import dgl
import torch
import torch.nn as nn
import dgl.function as Fn
import torch.nn.functional as F

from dgl.ops import edge_softmax
from dgl.nn.pytorch import TypedLinear
import math
import torch
import torch.nn as nn


class TimeEncoder(nn.Module):
    """
    Time2Vec z poprawioną inicjalizacją i stabilizacją.

    Zmiany względem poprzedniej wersji:
    - Częstotliwości log-spaced w zakresie [0.01, 50] zamiast Uniform(-0.1, 0.1)
    - Fazy rozłożone równomiernie po [0, 2π] dla dywersyfikacji
    - LayerNorm na wyjściu
    """
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.w = nn.Parameter(torch.empty(dim))
        self.b = nn.Parameter(torch.empty(dim))
        self.norm = nn.LayerNorm(dim)
        self._init_params()

    def _init_params(self):
        with torch.no_grad():
            self.w[0] = 1.0       # komponent liniowy (trend)
            self.b[0] = 0.0
            if self.dim > 1:
                # log-space: 0.01 → 50, pokrywa minuty/godziny/dni po normalizacji
                log_f = torch.linspace(math.log(0.01), math.log(50.0), self.dim - 1)
                self.w[1:] = torch.exp(log_f)
                self.b[1:] = torch.linspace(0.0, 2.0 * math.pi, self.dim - 1)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """t: (E,) znormalizowane timestamps → (E, dim)"""
        t = t.float().unsqueeze(-1)         # (E, 1)
        x = t * self.w + self.b             # (E, dim)
        emb = torch.cat([x[..., :1],
                         torch.sin(x[..., 1:])], dim=-1)
        return self.norm(emb)               # (E, dim), stabilne gradienty


class SimpleHGN(nn.Module):
    r"""
    This is a model SimpleHGN from `Are we really making much progress? Revisiting, benchmarking, and
    refining heterogeneous graph neural networks
    <https://dl.acm.org/doi/pdf/10.1145/3447548.3467350>`__, extended with an optional temporal
    module that encodes edge timestamps (Time2Vec) and fuses them directly into the attention score
    as a separate component with its own attention vector a_t. This avoids corrupting the shared
    edge-type embeddings.

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    num_etypes: int
        the number of the edge type
    in_dim: int
        the input dimension
    hidden_dim: int
        the output dimension
    num_classes: int
        the number of the output classes
    num_layers: int
        the number of layers we used in the computing
    heads: list
        the list of the number of heads in each layer
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    beta: float
        the hyperparameter used in edge residual
    ntypes: list
        the list of node type
    use_time: bool
        whether to enable the temporal module. default: False
    """
    @classmethod
    def build_model_from_args(cls, args, hg):
        heads = [args.num_heads] * args.num_layers + [1]
        return cls(args.edge_dim,
                   len(hg.etypes),
                   [args.hidden_dim],
                   args.hidden_dim // args.num_heads,
                   args.out_dim,
                   args.num_layers,
                   heads,
                   args.feats_drop_rate,
                   args.slope,
                   True,
                   args.beta,
                   hg.ntypes,
                   getattr(args, 'use_time', False),
                   getattr(args, 'contrastive', False)
                   )

    def __init__(self, edge_dim, num_etypes, in_dim, hidden_dim, num_classes,
                num_layers, heads, feat_drop, negative_slope,
                residual, beta, ntypes, use_time=False, contrastive=False):
        super(SimpleHGN, self).__init__()
        self.ntypes = ntypes
        self.num_layers = num_layers
        self.use_time = use_time
        self.hgn_layers = nn.ModuleList()
        self.activation = F.elu
        self.contrastive=contrastive


        # input projection (no residual)
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                in_dim[0],
                hidden_dim,
                heads[0],
                num_etypes,
                feat_drop,
                negative_slope,
                False,
                self.activation,
                beta=beta,
                use_time=use_time,
            )
        )
        # hidden layers
        for l in range(1, num_layers - 1):  # noqa E741
            self.hgn_layers.append(
                SimpleHGNConv(
                    edge_dim,
                    hidden_dim * heads[l - 1],
                    hidden_dim,
                    heads[l],
                    num_etypes,
                    feat_drop,
                    negative_slope,
                    residual,
                    self.activation,
                    beta=beta,
                    use_time=use_time,
                )
            )
        # output projection
        self.hgn_layers.append(
            SimpleHGNConv(
                edge_dim,
                hidden_dim * heads[-2],
                num_classes,
                heads[-1],
                num_etypes,
                feat_drop,
                negative_slope,
                residual,
                None,
                beta=beta,
                use_time=use_time,
            )
        )

    def forward(self, hg, h_dict, return_hidden=False):
        """
        The forward part of the SimpleHGN.

        Parameters
        ----------
        hg : object
            the dgl heterogeneous graph. Edges with timestamps should carry
            a 'time' feature under hg.edata['time']; other edge types should
            have 'time' set to 0.
        h_dict: dict
            the feature dict of different node types

        Returns
        -------
        dict
            The embeddings after the output projection.
        """
        with hg.local_scope():
            hg.ndata['h'] = h_dict
            edata_keys = ['time'] if (self.use_time and 'time' in hg.edata) else []
            g = dgl.to_homogeneous(hg, ndata='h', edata=edata_keys)
            h = g.ndata['h']

            if self.use_time and 'time' in g.edata:
                t = g.edata['time'].float()
                time_valid = (t != 0)
                t_norm = torch.zeros_like(t)
                if time_valid.any():
                    t_vals = t[time_valid]
                    t_norm[time_valid] = (t_vals - t_vals.mean()) / (t_vals.std() + 1e-8)
                timestamps, timestamps_mask = t_norm, time_valid.float()
            else:
                timestamps = timestamps_mask = None

            h_hidden = None
            for l in range(self.num_layers):
                h = self.hgn_layers[l](g, h, g.ndata['_TYPE'], g.edata['_TYPE'], True,
                                    timestamps=timestamps, timestamps_mask=timestamps_mask)
                h = h.flatten(1)
                if return_hidden and l == self.num_layers - 2:   # embedding przed warstwą wyjściową
                    h_hidden = h

            h_dict = to_hetero_feat(h, g.ndata['_TYPE'], hg.ntypes)
            if return_hidden:
                return h_dict, to_hetero_feat(h_hidden, g.ndata['_TYPE'], hg.ntypes)
            return h_dict

    @property
    def to_homo_flag(self):
        return True


class SimpleHGNConv(nn.Module):
    r"""
    The SimpleHGN convolution layer, with an optional temporal module.

    When use_time=True, timestamps are encoded via Time2Vec and added directly
    to the attention score as a separate term with its own attention vector a_t.
    This keeps the shared edge-type embeddings intact.

    Attention with temporal component:

    .. math::
        e_{ij} = LeakyReLU(a_l^T W h_i + a_r^T W h_j + a_e^T W_r r_{\psi} + a_t^T \tau(t_{ij}))

    Parameters
    ----------
    edge_dim: int
        the edge dimension
    in_dim: int
        the input dimension
    out_dim: int
        the output dimension
    num_heads: int
        the number of heads
    num_etypes: int
        the number of edge type
    feat_drop: float
        the feature drop rate
    negative_slope: float
        the negative slope used in the LeakyReLU
    residual: boolean
        if we need the residual operation
    activation: str
        the activation function
    beta: float
        the hyperparameter used in edge residual
    use_time: bool
        whether to enable the temporal module. default: False
    """
    def __init__(self, edge_dim, in_dim, out_dim, num_heads, num_etypes, feat_drop=0.0,
                 negative_slope=0.2, residual=True, activation=F.elu, beta=0.0, use_time=False):
        super(SimpleHGNConv, self).__init__()
        self.edge_dim = edge_dim
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_etypes = num_etypes
        self.use_time = use_time

        self.edge_emb = nn.Parameter(torch.empty(size=(num_etypes, edge_dim)))

        self.W = nn.Parameter(torch.FloatTensor(in_dim, out_dim * num_heads))
        self.W_r = TypedLinear(edge_dim, edge_dim * num_heads, num_etypes)

        self.a_l = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_r = nn.Parameter(torch.empty(size=(1, num_heads, out_dim)))
        self.a_e = nn.Parameter(torch.empty(size=(1, num_heads, edge_dim)))

        if use_time:
            self.time_encoder = TimeEncoder(edge_dim)
            self.a_t = nn.Parameter(torch.zeros(size=(1, num_heads, edge_dim)))

        nn.init.xavier_uniform_(self.edge_emb, gain=1.414)
        nn.init.xavier_uniform_(self.W, gain=1.414)
        nn.init.xavier_uniform_(self.a_l.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_r.data, gain=1.414)
        nn.init.xavier_uniform_(self.a_e.data, gain=1.414)

        self.feat_drop = nn.Dropout(feat_drop)
        self.leakyrelu = nn.LeakyReLU(negative_slope)
        self.activation = activation

        if residual:
            self.residual = nn.Linear(in_dim, out_dim * num_heads)
        else:
            self.register_buffer("residual", None)

        self.beta = beta

    def forward(self, g, h, ntype, etype, presorted=False, timestamps=None, timestamps_mask=None):
        """
        Parameters
        ----------
        g : object
            the dgl homogeneous graph
        h: tensor
            the original features of the graph
        ntype: tensor
            the node type of the graph
        etype: tensor
            the edge type of the graph
        presorted: boolean
            if the ntype and etype are preordered, default: False
        timestamps: tensor, optional, shape (E,)
            normalized edge timestamps (non-transaction edges should be 0).

        Returns
        -------
        tensor
            The embeddings after aggregation.
        """
        emb = self.feat_drop(h)
        emb = torch.matmul(emb, self.W).view(-1, self.num_heads, self.out_dim)
        emb[torch.isnan(emb)] = 0.0

        # edge type embedding — unchanged, not polluted by timestamps
        edge_type_emb = self.edge_emb[etype]
        edge_emb = self.W_r(edge_type_emb, etype, presorted).view(-1, self.num_heads, self.edge_dim)

        row = g.edges()[0]
        col = g.edges()[1]

        h_l = (self.a_l * emb).sum(dim=-1)[row]
        h_r = (self.a_r * emb).sum(dim=-1)[col]
        h_e = (self.a_e * edge_emb).sum(dim=-1)

        if self.use_time and timestamps is not None:
            te = self.time_encoder(timestamps).view(-1, 1, self.edge_dim)  # (E, 1, edge_dim)
            # timestamps_mask zamiast (timestamps != 0) – brak buga z normalizacją
            mask = timestamps_mask.view(-1, 1, 1) if timestamps_mask is not None \
                else (timestamps != 0).float().view(-1, 1, 1)
            h_t = (self.a_t * te * mask).sum(dim=-1)                       # (E, num_heads)
            edge_attention = self.leakyrelu(h_l + h_r + h_e + h_t)
        else:
            edge_attention = self.leakyrelu(h_l + h_r + h_e)

        edge_attention = edge_softmax(g, edge_attention)

        if 'alpha' in g.edata.keys():
            res_attn = g.edata['alpha']
            edge_attention = edge_attention * (1 - self.beta) + res_attn * self.beta

        if self.num_heads == 1:
            edge_attention = edge_attention[:, 0]
            edge_attention = edge_attention.unsqueeze(1)

        with g.local_scope():
            emb = emb.permute(0, 2, 1).contiguous()
            g.edata['alpha'] = edge_attention
            g.srcdata['emb'] = emb
            g.update_all(Fn.u_mul_e('emb', 'alpha', 'm'),
                         Fn.sum('m', 'emb'))
            h_output = g.dstdata['emb'].view(-1, self.out_dim * self.num_heads)

        g.edata['alpha'] = edge_attention
        if g.is_block:
            h = h[:g.num_dst_nodes()]
        if self.residual:
            res = self.residual(h)
            h_output += res
        if self.activation is not None:
            h_output = self.activation(h_output)

        return h_output


def to_hetero_feat(h, ntype, ntypes):
    h_dict = {}
    for i, ntype_name in enumerate(ntypes):
        mask = (ntype == i)
        h_dict[ntype_name] = h[mask]
    return h_dict

class ProjectionHead(nn.Module):
    """Osobna głowa do przestrzeni kontrastywnej; odrzucana przy inferencji."""
    def __init__(self, in_dim, hidden_dim=64, out_dim=32):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class SupervisedContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature

    def forward(self, z, labels):
        B = z.shape[0]
        labels = labels.view(-1, 1)

        pos_mask = torch.eq(labels, labels.T).float()
        self_mask = torch.eye(B, device=z.device)
        pos_mask = pos_mask - self_mask

        sim = (z @ z.T) / self.temperature
        sim = sim - sim.max(dim=1, keepdim=True).values.detach()

        exp_sim = torch.exp(sim) * (1 - self_mask)
        log_prob = sim - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-12)

        pos_count = pos_mask.sum(dim=1)
        valid = pos_count > 0
        if not valid.any():
            return sim.sum() * 0.0   # brak crasha autograd, zerowy wkład

        loss = -(pos_mask * log_prob).sum(dim=1)[valid] / pos_count[valid]
        return loss.mean()


def sample_contrastive_subset(labels, neg_per_pos=8, max_size=4000):
    """Cały fraud + losowy multiplikat non-fraud. Bez tego: O(N²) na ~37k+ węzłach = OOM."""
    pos_idx = (labels == 1).nonzero(as_tuple=True)[0]
    neg_idx_all = (labels == 0).nonzero(as_tuple=True)[0]

    n_neg = min(len(neg_idx_all), len(pos_idx) * neg_per_pos, max(max_size - len(pos_idx), 0))
    perm = torch.randperm(len(neg_idx_all), device=neg_idx_all.device)[:n_neg]
    return torch.cat([pos_idx, neg_idx_all[perm]])


class DenoisingModule(nn.Module):
    """
    Denoising Module: rekonstrukcja zaszumionych cech wejściowych węzłów.

    Dla każdego typu węzła dodaje szum gaussowski N(0, sigma_t^2) do surowych
    cech (h_raw), koduje je współdzielonym InputEncoder (te same wagi co
    główna ścieżka), a następnie dekoduje z powrotem do przestrzeni cech
    wejściowych osobnym dekoderem per typ. Strata rekonstrukcji (MSE) działa
    jak trening denoising autoencodera i wymusza na InputEncoderze reprezentacje
    odporne na perturbacje wejścia — zgodnie z L_denoise = ||x_i - x_hat_i||^2.

    sigma jest różna per typ węzła, bo cechy poszczególnych modalności mają
    różną wiarygodność (np. merchant jest bardziej zaszumiony/rzadki niż
    transaction). Operuje na całym grafie i nie korzysta z etykiet fraud,
    więc nie ma ryzyka wycieku między splitami.
    """
    DEFAULT_SIGMA = {'user': 0.10, 'card': 0.10, 'transaction': 0.05, 'merchant': 0.15}

    def __init__(self, hg, proj_dim=32, sigma=None):
        super().__init__()
        self.sigma = sigma or self.DEFAULT_SIGMA
        self.decoders = nn.ModuleDict({
            ntype: nn.Linear(proj_dim, hg.nodes[ntype].data['h_raw'].shape[1])
            for ntype in hg.ntypes
        })

    def forward(self, encoder, hg):
        """
        encoder: InputEncoder współdzielony z główną ścieżką (nn.ModuleDict per typ)
        hg: graf z surowymi cechami w hg.nodes[ntype].data['h_raw']

        Zwraca stratę rekonstrukcji (MSE) uśrednioną po typach węzłów.
        """
        losses = []
        for ntype, enc in encoder.encoders.items():
            x_clean = hg.nodes[ntype].data['h_raw']
            sigma   = self.sigma.get(ntype, 0.1)
            x_noisy = x_clean + torch.randn_like(x_clean) * sigma

            x_hat = self.decoders[ntype](enc(x_noisy))
            losses.append(F.mse_loss(x_hat, x_clean))

        return torch.stack(losses).mean()