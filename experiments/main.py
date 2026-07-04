import pandas as pd
import numpy as np
import torch
from typing import Dict, Tuple
import graph.graph_creation_v2 as gc
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F
from collections import defaultdict
from evaluation.statistical_analysis import StatisticalAnalysis
from trainingloop import cross_validation as cv
from trainingloop.training_processor import TrainingProcessor
import time
from evaluation.file_logger import log
from models.external.simpleHGN import SimpleHGN
import os
from dgl import save_graphs, load_graphs

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def main():
    """
    GRAPH CREATION
    """



    graph_path = "my_saved_graph_u150.dgl"

    if os.path.exists(graph_path):
        glist, _ = load_graphs(graph_path)
        data = glist[0]
        print("Graf załadowany z pliku.")
    else:
        data = gc.build_dgl_graph(max_users=150) 
        save_graphs(graph_path, [data])
        print("Graf utworzony i zapisany do pliku.")


    # print(f"data = {data}")
    # print({ntype: data.nodes[ntype].data['h_raw'].shape for ntype in data.ntypes})


    data_masks = cv.split(data, 3, 0.6, 0.2)

    """
    TRAINING
    """

    # tp_hgn = TrainingProcessor("SimpleHGN")
    # tp_hgn_200_epochs = TrainingProcessor("SimpleHGN", n_epochs=300)
    # tp_hgn_low_loss = TrainingProcessor("SimpleHGN", loss_mul=0.5)
    tp_hgn_200_epochs_low_loss = TrainingProcessor("SimpleHGN", loss_mul=0.5, n_epochs=200, n_reps=10)


    # r1= tp_hgn.cross_validation_training(data_masks, data)

    # r2= tp_hgn_200_epochs.cross_validation_training(data_masks, data)
    data = data.to(device) 

    r_base = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, use_time=False, contrastive=False, denoise=False)
    
    r_time = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, use_time=True, contrastive=False, denoise=False)
    r_contrastive = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, use_time=False, contrastive=True, denoise=False)
    r_denoise = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, denoise=True)
    
    r_time_contrastive = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, use_time=True, contrastive=True, denoise=False)
    r_denoise_time = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, use_time=True, contrastive=False, denoise=True)
    r_denoise_contrastive = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, use_time=False, contrastive=True, denoise=True)
    
    r_all = tp_hgn_200_epochs_low_loss.cross_validation_training(data_masks, data, use_time=True, contrastive=True, denoise=True)

    # r4= tp_hgn_low_loss.cross_validation_training(data_masks, data)

    
    r_base = dict(r_base)
    
    r_time = dict(r_time)
    r_contrastive = dict(r_contrastive)
    r_denoise = dict(r_denoise)

    r_time_contrastive = dict(r_time_contrastive)
    r_denoise_time = dict(r_denoise_time)
    r_denoise_contrastive = dict(r_denoise_contrastive)
    
    r_all = dict(r_all)

    d = [r_base, r_time, r_contrastive, r_denoise, r_time_contrastive, r_denoise_time, r_denoise_contrastive, r_all]

    # log(d)

    # results = {
    #             'acc':          [np.float64(0.9846453407510432), np.float64(0.9873157162726007), np.float64(0.9894059209219153), np.float64(0.9866719650307967), np.float64(0.9723266441486192), np.float64(0.9897039539042322), np.float64(0.9871289489370157), np.float64(0.9858414464534075), np.float64(0.9907411086826942), np.float64(0.9886866679912577), np.float64(0.992362408106497), np.float64(0.9902523345916947), np.float64(0.9910033777071329), np.float64(0.9849115835485793), np.float64(0.9898787999205245)], 
    #             'recall':       [np.float64(0.1971064814814815), np.float64(0.19050925925925927), np.float64(0.27523148148148147), np.float64(0.22094907407407405), np.float64(0.22523148148148148), np.float64(0.21875), np.float64(0.21782407407407406), np.float64(0.21585648148148148), np.float64(0.20613425925925927), np.float64(0.21168981481481483), np.float64(0.29166666666666663), np.float64(0.23032407407407404), np.float64(0.22002314814814813), np.float64(0.3241898148148148), np.float64(0.2033564814814815)], 
    #             'precision':    [np.float64(0.025514433511202493), np.float64(0.03166649234221749), np.float64(0.052924988150066146), np.float64(0.02800746893759598), np.float64(0.006759252562912493), np.float64(0.030628616062598386), np.float64(0.03623765541352507), np.float64(0.02264386515791885), np.float64(0.031174315063842172), np.float64(0.02504296722417528), np.float64(0.044816427591547206), np.float64(0.03138913070513618), np.float64(0.04605349910208592), np.float64(0.057213780388013634), np.float64(0.04310527881790379)], 
    #             'f1_score':     [np.float64(0.04252098964271761), np.float64(0.049928510919492144), np.float64(0.08529647033775871), np.float64(0.04624556780118302), np.float64(0.013087230503847141), np.float64(0.051999966978026266), np.float64(0.057624201891724694), np.float64(0.039154713813131604), np.float64(0.05228550005233694), np.float64(0.04263968030990819), np.float64(0.0761264381332433), np.float64(0.053373279602787795), np.float64(0.0683333764301495), np.float64(0.09204170151328676), np.float64(0.06263493670886075)]
    #         }
    # results = [results, results, results]

    # d = [ 
    #         { #BASE
    #             'acc': [np.float64(0.9939146097300462), np.float64(0.9986635935945936), np.float64(0.9990604651251417), np.float64(0.9989578739156103), np.float64(0.9990361679873985), np.float64(0.9992251543045216), np.float64(0.9988147819879113), np.float64(0.9990874629361567), np.float64(0.9969708050934788), np.float64(0.9990793641745123)], 
    #             'recall': [np.float64(0.35271121104727104), np.float64(0.38705098809913424), np.float64(0.3690487080595141), np.float64(0.4786120873575797), np.float64(0.4586725605395728), np.float64(0.46851273153270806), np.float64(0.2988779805026096), np.float64(0.4411950437171496), np.float64(0.309892106321845), np.float64(0.4024781414234379)], 
    #             'precision': [np.float64(0.3956275826974931), np.float64(0.4770033475585751), np.float64(0.6665956727518451), np.float64(0.6670279565127893), np.float64(0.7389840433318565), np.float64(0.7900912322569299), np.float64(0.4657688360219933), np.float64(0.5757314349959269), np.float64(0.37953688280718917), np.float64(0.6735926908340589)], 
    #             'f1_score': [np.float64(0.304854614370894), np.float64(0.3604918475380434), np.float64(0.42721056496776916), np.float64(0.5044321668170423), np.float64(0.4972150354426684), np.float64(0.5564823396144184), np.float64(0.33135996538012985), np.float64(0.4975158128058783), np.float64(0.2962632570436748), np.float64(0.4597365362258375)], 
    #             'pr_auc': [np.float64(0.32429324680029536), np.float64(0.3746714210585205), np.float64(0.45382256863912745), np.float64(0.48508924825275107), np.float64(0.5167216036450254), np.float64(0.5270084598927859), np.float64(0.287084408804461), np.float64(0.44856556615925874), np.float64(0.30383503781307447), np.float64(0.4689387647016304)], 
    #             'mcen': [np.float64(0.1365799752649293), np.float64(0.15655674948332463), np.float64(0.160536164267678), np.float64(0.16707154238194202), np.float64(0.161310927726011), np.float64(0.1741932100854476), np.float64(0.16536498523320708), np.float64(0.18132729243336335), np.float64(0.15519331827583463), np.float64(0.16430788335850552)]
    #         }, 
    #         { #TIME
    #             'acc': [np.float64(0.9940522986055828), np.float64(0.9983720149089926), np.float64(0.9989659722399167), np.float64(0.9988714802293681), np.float64(0.9990361676812617), np.float64(0.9993304465153283), np.float64(0.9989632717853142), np.float64(0.9989524719349268), np.float64(0.996849314069235), np.float64(0.9990820639293734)], 
    #             'recall': [np.float64(0.33807685838474955), np.float64(0.41964237411338284), np.float64(0.3855189942644874), np.float64(0.4755549850538348), np.float64(0.45011847746862116), np.float64(0.4700221020234085), np.float64(0.28127898615942937), np.float64(0.41122353929428607), np.float64(0.33101886688522514), np.float64(0.44363934848703573)], 
    #             'precision': [np.float64(0.31643780695599627), np.float64(0.43469185293164325), np.float64(0.641108891108877), np.float64(0.6412966066191784), np.float64(0.6634539272624921), np.float64(0.845106486674504), np.float64(0.48846125675393165), np.float64(0.505122890403733), np.float64(0.32613631367093626), np.float64(0.6277528690716058)], 
    #             'f1_score': [np.float64(0.3122227807956161), np.float64(0.35438226453432325), np.float64(0.4138542153983218), np.float64(0.48960434199279695), np.float64(0.491444315444762), np.float64(0.5881459670928803), np.float64(0.3320176467607412), np.float64(0.45079775877037237), np.float64(0.2919758714896883), np.float64(0.49367723031798555)], 
    #             'pr_auc': [np.float64(0.32599181087215096), np.float64(0.37525363662934436), np.float64(0.4644377345829304), np.float64(0.4776983812765517), np.float64(0.4991426585875738), np.float64(0.5310385748173325), np.float64(0.30072564934799345), np.float64(0.41832205070004075), np.float64(0.3133183233128592), np.float64(0.46918660280133534)], 
    #             'mcen': [np.float64(0.14896970292651032), np.float64(0.1565243681272038), np.float64(0.15452344062614176), np.float64(0.16653804896114283), np.float64(0.16878394064162314), np.float64(0.1778224276485825), np.float64(0.15235553852087444), np.float64(0.18256526616066757), np.float64(0.1557816993865017), np.float64(0.17328427810035463)]
    #         },
    #         { #CONTRASTIVE
    #             'acc': [np.float64(0.9988228835047871), np.float64(0.9989065775673697), np.float64(0.999044266224237), np.float64(0.9990577666166947), np.float64(0.9990631657546795), np.float64(0.9992143551320084), np.float64(0.9988444797287227), np.float64(0.9989740706516906), np.float64(0.9989794709923556), np.float64(0.9991711576547467)], 
    #             'recall': [np.float64(0.43588741050253965), np.float64(0.4819485387778831), np.float64(0.36176989305059754), np.float64(0.4751202234005995), np.float64(0.42819203087791013), np.float64(0.4811419673850031), np.float64(0.35924001302317415), np.float64(0.46703483699880316), np.float64(0.3661381656795973), np.float64(0.4217104428372)], 
    #             'precision': [np.float64(0.6189255647245053), np.float64(0.6760073260073157), np.float64(0.6350648387968723), np.float64(0.7226650563606979), np.float64(0.7473562631457226), np.float64(0.7722906403940776), np.float64(0.5237145432885718), np.float64(0.5454773405125712), np.float64(0.5780227170365341), np.float64(0.7364779047869195)], 
    #             'f1_score': [np.float64(0.44936728078722404), np.float64(0.4893229031752449), np.float64(0.41682368606128056), np.float64(0.5215590734649244), np.float64(0.47816427191719835), np.float64(0.5610150394222465), np.float64(0.3884249434064449), np.float64(0.48390039799400114), np.float64(0.4053968253963944), np.float64(0.5059010729219033)], 
    #             'pr_auc': [np.float64(0.47047225700936673), np.float64(0.5194639444100271), np.float64(0.4482915202057843), np.float64(0.5119989053997459), np.float64(0.5142225306669429), np.float64(0.5325523704659816), np.float64(0.366395531700834), np.float64(0.4482025607614399), np.float64(0.42242351828304886), np.float64(0.49782493088489704)], 
    #             'mcen': [np.float64(0.16473697309832375), np.float64(0.15944945783878092), np.float64(0.16018749330757334), np.float64(0.16835230580117835), np.float64(0.1606333115021312), np.float64(0.1738373727871633), np.float64(0.171528619459403), np.float64(0.17340954297003638), np.float64(0.16137991938191468), np.float64(0.1738328935406442)]
    #         },
    #         { #CONTRASTIVE + TIME
    #             'acc': [np.float64(0.9989875701913385), np.float64(0.9985880010707536), np.float64(0.998868778440884), np.float64(0.9991090621995936), np.float64(0.9990847643402421), np.float64(0.9992926486352567), np.float64(0.9989929692199889), np.float64(0.9988687774568726), np.float64(0.9990118687941653), np.float64(0.9991333604744165)], 
    #             'recall': [np.float64(0.3965577106749692), np.float64(0.47756452817048567), np.float64(0.3819887689851629), np.float64(0.41856333896949344), np.float64(0.4764313937258544), np.float64(0.48157672903823845), np.float64(0.39748821864773065), np.float64(0.416209035763569), np.float64(0.42187716704585015), np.float64(0.45439428947453475)], 
    #             'precision': [np.float64(0.527144347566877), np.float64(0.6129498969398989), np.float64(0.5344494720965192), np.float64(0.7154690600593375), np.float64(0.6916101442417135), np.float64(0.7452500074451202), np.float64(0.5859647266313855), np.float64(0.46018251208928757), np.float64(0.5835211099747432), np.float64(0.6882173583732986)], 
    #             'f1_score': [np.float64(0.4426152064196378), np.float64(0.4440385062576994), np.float64(0.3823059540854981), np.float64(0.48579830754597025), np.float64(0.5198926205573962), np.float64(0.5842384554234202), np.float64(0.45252183867172735), np.float64(0.4362715623784082), np.float64(0.4564274637226113), np.float64(0.5155005200353114)], 
    #             'pr_auc': [np.float64(0.41561592158399013), np.float64(0.4940278170558193), np.float64(0.4337555579963693), np.float64(0.4894237332248532), np.float64(0.5139780928863837), np.float64(0.5421553506603117), np.float64(0.3883331045154294), np.float64(0.39257980997842584), np.float64(0.45439485249728745), np.float64(0.49585077495796354)], 
    #             'mcen': [np.float64(0.17974137006132537), np.float64(0.15321220182169767), np.float64(0.15152128513079613), np.float64(0.16983728679841145), np.float64(0.16815929424161066), np.float64(0.18051700771795007), np.float64(0.1785738863673801), np.float64(0.18212481985668796), np.float64(0.16831159950684924), np.float64(0.17244804051540683)]
    #         },
    #         { #DENOISE
    #             'acc': [np.float64(0.993763420921257), np.float64(0.9984395119949591), np.float64(0.9990199681899502), np.float64(0.9986959917900076), np.float64(0.9989470746556296), np.float64(0.9992656509773101), np.float64(0.998679792320563), np.float64(0.9989551731548713), np.float64(0.9981884263741057), np.float64(0.9987985850112949)], 
    #             'recall': [np.float64(0.3497082080442681), np.float64(0.4591117484992325), np.float64(0.37059643987255875), np.float64(0.4396000979689119), np.float64(0.4916470679654393), np.float64(0.47566318365531873), np.float64(0.3505846167162876), np.float64(0.4592671610359094), np.float64(0.3420644686612559), np.float64(0.4366015212723333)], 
    #             'precision': [np.float64(0.3604260767507153), np.float64(0.5835688297201841), np.float64(0.6585924815786046), np.float64(0.5365016003164595), np.float64(0.6933664680776248), np.float64(0.7614774114774008), np.float64(0.38930930071763553), np.float64(0.5521413789236466), np.float64(0.3420127464496024), np.float64(0.588798687986871)], 
    #             'f1_score': [np.float64(0.28486805651855124), np.float64(0.41830832055026795), np.float64(0.4177562323150977), np.float64(0.4191826401593383), np.float64(0.4971092441836329), np.float64(0.5656351603082354), np.float64(0.31467949498597425), np.float64(0.4765557226270925), np.float64(0.2890485653804599), np.float64(0.4283319308725331)], 
    #             'pr_auc': [np.float64(0.31279061310918044), np.float64(0.4920304072554689), np.float64(0.4586938478490941), np.float64(0.4226003658821133), np.float64(0.5227118511052836), np.float64(0.5253366701619552), np.float64(0.31902457899880937), np.float64(0.4671733597528544), np.float64(0.3225808701652662), np.float64(0.4708552894329176)], 
    #             'mcen': [np.float64(0.13483185475584294), np.float64(0.15212277666995314), np.float64(0.15799983280417915), np.float64(0.16415361450882412), np.float64(0.15382330046080206), np.float64(0.17487847153326075), np.float64(0.14323706548916543), np.float64(0.17282495336794879), np.float64(0.16351821558023852), np.float64(0.15527357465343752)]
    #         }, 
    #         { #DENOISE + TIME
    #             'acc': [np.float64(0.9939146095551109), np.float64(0.9978428553246451), np.float64(0.9989821709002853), np.float64(0.9988660803479082), np.float64(0.9990145691175661), np.float64(0.9992575517564605), np.float64(0.9988282819648976), np.float64(0.9988174832734565), np.float64(0.9984908046695581), np.float64(0.9986932922756825)], 
    #             'recall': [np.float64(0.3423910317130073), np.float64(0.35635406123450375), np.float64(0.37500061476477814), np.float64(0.45222933382120695), np.float64(0.497231591237483), np.float64(0.47730091703232497), np.float64(0.34499763438511927), np.float64(0.42524066811652445), np.float64(0.3566220986790889), np.float64(0.4674268110232697)], 
    #             'precision': [np.float64(0.3527227341500128), np.float64(0.35035714951788915), np.float64(0.6212121212121079), np.float64(0.6352152405732833), np.float64(0.6040642913635376), np.float64(0.7648518113077395), np.float64(0.4537699703607241), np.float64(0.5259059059058996), np.float64(0.3533128575827775), np.float64(0.49872651121657224)], 
    #             'f1_score': [np.float64(0.29811507724970837), np.float64(0.2504312006151778), np.float64(0.4075776382360372), np.float64(0.4658585858581517), np.float64(0.5148083315199746), np.float64(0.5675987905058144), np.float64(0.3283277435172342), np.float64(0.43111318601469045), np.float64(0.32434424383840815), np.float64(0.4317721960574598)], 
    #             'pr_auc': [np.float64(0.31489014162994416), np.float64(0.32812248935077276), np.float64(0.4582194117362232), np.float64(0.4943699243744459), np.float64(0.5113917571784862), np.float64(0.527304882729078), np.float64(0.32604181152989925), np.float64(0.4393524048061545), np.float64(0.3360538992494888), np.float64(0.47119878517156133)], 
    #             'mcen': [np.float64(0.1411441560791472), np.float64(0.11742910346950154), np.float64(0.15471103064188), np.float64(0.1648564763312651), np.float64(0.1684641257497461), np.float64(0.17645111476954486), np.float64(0.13055390306398712), np.float64(0.16951883726980363), np.float64(0.16778041811910135), np.float64(0.16111255774364217)]
    #         }, 
    #         { #DENOISE + CONTRASTIVE
    #             'acc': [np.float64(0.9987499879368432), np.float64(0.9984719098623692), np.float64(0.9989011766800315), np.float64(0.9988498817531403), np.float64(0.9991549593005149), np.float64(0.999190056725984), np.float64(0.9985636986412171), np.float64(0.9989470725564057), np.float64(0.99833961741332), np.float64(0.9990388674798568)], 
    #             'recall': [np.float64(0.3217732963884264), np.float64(0.4852422025694178), np.float64(0.339316224183565), np.float64(0.4390910727301128), np.float64(0.4663502349385049), np.float64(0.48354348452612067), np.float64(0.42818957181878553), np.float64(0.43433525238307075), np.float64(0.3244339983612803), np.float64(0.45890666296823807)], 
    #             'precision': [np.float64(0.40829363459007545), np.float64(0.6194200707658168), np.float64(0.5896961314729549), np.float64(0.6927319644532638), np.float64(0.7753535353535215), np.float64(0.7289322387085336), np.float64(0.49086748491782184), np.float64(0.4912517423579326), np.float64(0.3280739099072037), np.float64(0.6496851418162807)], 
    #             'f1_score': [np.float64(0.304756524224691), np.float64(0.44500290866743003), np.float64(0.35998140202772805), np.float64(0.45837370610085), np.float64(0.5251285519466224), np.float64(0.5516640111812631), np.float64(0.4068298771856907), np.float64(0.45979507436094935), np.float64(0.2562862365956303), np.float64(0.49843785146510733)], 
    #             'pr_auc': [np.float64(0.326226906027346), np.float64(0.5011121489072443), np.float64(0.42235952427166285), np.float64(0.4951094557529476), np.float64(0.5312378117986332), np.float64(0.5294429434108353), np.float64(0.36196939412844453), np.float64(0.44595673550569576), np.float64(0.3112185835936723), np.float64(0.4980127881681787)], 
    #             'mcen': [np.float64(0.12910137452548462), np.float64(0.15140384580610633), np.float64(0.14724634533295075), np.float64(0.1601265678926528), np.float64(0.16148935448875776), np.float64(0.17304217097780583), np.float64(0.16733142560339256), np.float64(0.18045499481741997), np.float64(0.13785259855519202), np.float64(0.1709400712874388)]    
    #         }, 
    #         { #DENOISE + TIME + CONTRASTIVE
    #             'acc': [np.float64(0.9985717994146177), np.float64(0.9984017149895639), np.float64(0.9987202894525568), np.float64(0.9988768796078892), np.float64(0.998874179809294), np.float64(0.9991630584557639), np.float64(0.9988795782256706), np.float64(0.9986986888989721), np.float64(0.9986338956132461), np.float64(0.9988093835715346)], 
    #             'recall': [np.float64(0.32970769655997034), np.float64(0.4883352071363823), np.float64(0.39771346846354705), np.float64(0.43330343117437425), np.float64(0.5135735145561505), np.float64(0.4854718786916587), np.float64(0.44258736299351725), np.float64(0.41752020608882384), np.float64(0.3288539112318974), np.float64(0.46269361402017434)], 
    #             'precision': [np.float64(0.3451923910327948), np.float64(0.5774302310631809), np.float64(0.514823937237719), np.float64(0.6548995091228942), np.float64(0.5883844934428527), np.float64(0.6809796740829842), np.float64(0.5690237422716348), np.float64(0.406585916993993), np.float64(0.3579373122629022), np.float64(0.5160689333954297)], 
    #             'f1_score': [np.float64(0.304276753817574), np.float64(0.44215057817957987), np.float64(0.36695946831694837), np.float64(0.4572737455807116), np.float64(0.5006626199762039), np.float64(0.5426974898455721), np.float64(0.4429631361567902), np.float64(0.40928556268949334), np.float64(0.2975330857574533), np.float64(0.44790155290523037)], 
    #             'pr_auc': [np.float64(0.32663195117478855), np.float64(0.48803200907213706), np.float64(0.4283570058421439), np.float64(0.49861683831629744), np.float64(0.521328136465689), np.float64(0.5254844550322478), np.float64(0.404591461714847), np.float64(0.4024181809971359), np.float64(0.31770422670171045), np.float64(0.49736605398659367)], 
    #             'mcen': [np.float64(0.16158489217081917), np.float64(0.1543412974377113), np.float64(0.14771186949319506), np.float64(0.16470438891546815), np.float64(0.16119968053434966), np.float64(0.173465679038484), np.float64(0.16391684122727151), np.float64(0.17968748351924724), np.float64(0.14688829952808868), np.float64(0.1661645693325077)]
    #         }
    #     ]

    """
    STATISTICAL ANALYSIS
    """

    stats = StatisticalAnalysis(d)


"""

"""

# print(data)

# model = HeteroGNN(data.metadata(), HIDDEN_CHANNELS, num_edge_features, data).to(device)

# optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
# class_weights = calculate_class_weights(transactions, device)

# ## 3. Trening i Ewaluacja
# run_training_loop(model, data, optimizer, class_weights)
# final_evaluation(model, data)



if __name__ == '__main__':

    start = time.perf_counter()

    main()


    end = time.perf_counter()
    print(f"Execution time: {end - start:.4f} seconds")
