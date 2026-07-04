import numpy as np
from scipy.stats import rankdata, studentized_range, wilcoxon, friedmanchisquare
import math
from collections import defaultdict
from itertools import combinations
from evaluation.file_logger import log
import matplotlib.pyplot as plt


class StatisticalAnalysis():
    def __init__(self, data):
        self.data = dict(self._prepare_data(data))
        self.statistics_summary = {}

        for key in self.data.keys():
            self.statistics_summary[key] = {}

        print(f"DATA = {self.data}")
        print(f"STAT_SUM = {self.statistics_summary}")


        # statistics summary:
        # {
        #  'acc': {'friedman_anova': 123, 'p_val': 0.05, 'eta_squared': 0.3, 'kendalls_w': 0.4,
        #          'nemenyi': {...}, 'wilcoxon_holm': [...]},
        #  (...)
        #  'recall': {'friedman_anova': 23, 'p_val': 0.05, 'eta_squared': 0.3, 'kendalls_w': 0.4,
        #             'nemenyi': {...}, 'wilcoxon_holm': [...]}
        # }

        print("friedman")
        self._set_friedman_p_val()
        print("friedman done")
        print(self.statistics_summary)
        print("proportion of variance")
        self._set_proportion_of_variance()
        print(self.statistics_summary)
        print("nemenyi test")
        self._set_nemenyi_test()
        print(f"____________________")
        self._wilcoxon_holm()




    def _prepare_data(self, data):
        # INPUT
        # data = [
        #   {'acc': [1.21, 1.22, 1.22], (...), 'recall': [0.43, 0.22, 0.11]},  <- G1
        #   {'acc': [2.21, 2.22, 2.22], (...), 'recall': [1.43, 1.22, 1.11]},  <- G2
        #   {'acc': [3.21, 3.22, 3.22], (...), 'recall': [2.43, 2.22, 2.11]},  <- G3
        # ]

        # IN CLASS
        # self.data = {
        #     'acc': [[1.21, 1.22, 1.22],[2.21, 2.22, 2.22],[3.21, 3.22, 3.22]],
        #     (...),
        #     'recall': [[0.43, 0.22, 0.11],[1.43, 1.22, 1.11],[2.43, 2.22, 2.11]]
        # }
        print(data)
        d = defaultdict(list)
        for g in data:
            for key in g.keys():
                d[key].append(g[key])

        return d

    def _set_friedman_p_val(self):
        for key in self.statistics_summary.keys():
            stats = self.data[key]
            friedman, p_value = friedmanchisquare(*stats)

            self.statistics_summary[key]['friedman_anova'] = friedman
            self.statistics_summary[key]['p_val'] = p_value

    def _set_proportion_of_variance(self):
        for key in self.statistics_summary.keys():
            results = self.data[key]

            data = np.array(results)   # shape: (k, n)
            k, n = data.shape

            stat = self.statistics_summary[key]['friedman_anova']

            eta_squared = (stat - k + 1) / (n * (k - 1))
            kendalls_w = stat / (n * (k - 1))

            self.statistics_summary[key]['eta_squared'] = eta_squared
            self.statistics_summary[key]['kendalls_w'] = kendalls_w

    def _set_nemenyi_test(self):
        print("_________ NEMENYI __________")
        for key in self.statistics_summary.keys():
            print(f"\n ANALYZING {key} \n")
            results = self.data[key]

            data = np.array(results)   # shape: (k, n)
            k, n = data.shape

            # rank methods within each experiment (column-wise)
            ranks = np.array([
                rankdata(-data[:, j])
                for j in range(n)
            ]).T   # shape: (k, n)
            print(f"k = {k}, n= {n}")

            log(f"ranks = " + str(ranks))

            avg_ranks = ranks.mean(axis=1)

            log(f"avg_ranks = " + str(avg_ranks))

            CD = self._nemenyi_cd(k, n)

            log("CD = " + str(CD))

            method_names = [f"G{i}" for i in range(k)]

            comparisons = []
            for j in range(1, k):
                diff = abs(avg_ranks[0] - avg_ranks[j])
                comparisons.append({
                    "method_1": method_names[0],
                    "method_2": method_names[j],
                    "rank_diff": float(diff),
                    "significant": float(diff) > float(CD)
                })

            self.statistics_summary[key]['nemenyi'] = {
                "avg_ranks": dict(zip(method_names, avg_ranks.tolist())),
                "CD": CD,
                "comparisons": comparisons,
            }

            log(f"NEMENYI RESULTS ({key}): ")
            for comp in comparisons:
                log(comp)
            log("___________________")

    def _nemenyi_cd(self, k, n, alpha=0.05):
        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / math.sqrt(2)
        return q_alpha * math.sqrt((k * (k + 1)) / (6 * n))

    def _holm_correction(self, p_values, alpha=0.05):
        m = len(p_values)
        sorted_indices = np.argsort(p_values)
        adjusted = np.zeros(m, dtype=bool)

        for rank, idx in enumerate(sorted_indices):
            threshold = alpha / (m - rank)
            if p_values[idx] <= threshold:
                adjusted[idx] = True
            else:
                break

        return adjusted

    def _wilcoxon_holm(self, alpha=0.05):

        for key in self.statistics_summary.keys():

            results = self.data[key]

            k = len(results)
            n = len(results[0])

            method_names = [f"G{i}" for i in range(k)]

            p_values = []
            pairs = []

            for j in range(1, k):
                stat, p = wilcoxon(
                    results[0],
                    results[j],
                    zero_method="pratt",
                    alternative="two-sided"
                )
                p_values.append(p)
                pairs.append((0, j, stat))

            significant = self._holm_correction(p_values, alpha)

            results_table = []
            for idx, ((i, j, stat), p, sig) in enumerate(
                zip(pairs, p_values, significant)
            ):
                results_table.append({
                    "key": key,
                    "method_1": method_names[0],
                    "method_2": method_names[j],
                    "W": stat,
                    "p_uncorrected": p,
                    "significant": sig
                })

            self.statistics_summary[key]['wilcoxon_holm'] = results_table

            print(f"\n {key}")
            for r in results_table:
                log(
                    str(r['method_1']) + " vs " + str(r['method_2']) + ": "
                    + "W= " + str(r['W']) + " p= " + str(round(r['p_uncorrected'], 4)) + ", "
                    + str('SIGNIFICANT' if r['significant'] else 'n.s.')
                )


def plot_nemenyi_cd(ranks, cd, title="Critical Difference (Nemenyi)"):
    # 1. Sort ranks
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
    methods, rank_values = zip(*sorted_ranks)
    k = len(methods)

    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(max(8, k), 3))
    ax.set_title(title, pad=20, fontsize=14)

    low, high = 1, k
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(0, 1)

    # Draw the main axis line
    ax.hlines(0.2, low, high, color='black', linewidth=1.5)

    # Add Tick Markers
    for i in range(low, high + 1):
        ax.vlines(i, 0.2, 0.25, color='black')
        ax.text(i, 0.28, str(i), ha='center', va='bottom', fontsize=10)

    # 3. Plot Methods and Ranks
    for i, (method, r) in enumerate(zip(methods, rank_values)):
        ax.plot(r, 0.2, 'o', color='black', markersize=6)

        level = 0.05 if i % 2 == 0 else 0.45  # Alternate label positions (top/bottom)
        va = 'top' if level < 0.2 else 'bottom'

        ax.plot([r, r], [0.2, level], '-', color='gray', linewidth=0.8)
        ax.text(r, level, f"{method}\n({r:.2f})", ha='center', va=va, fontsize=11, fontweight='bold')

    y_bar = 0.25
    bar_offset = 0.0
    first_bar = True
    start = 0
    while start < k - 1:
        end = start
        while end + 1 < k and (rank_values[end + 1] - rank_values[start]) < cd:
            end += 1
        if end > start:
            ax.hlines(
                y_bar + 0.02 + bar_offset,
                rank_values[start], rank_values[end],
                color='red', linewidth=3,
                label='Not Sig.' if first_bar else None,
            )
            first_bar = False
            bar_offset += 0.03
        start += 1

    # 5. Add CD ruler (visual guide)
    ax.hlines(0.8, low, low + cd, color='black', linewidth=2)
    ax.vlines(low, 0.78, 0.82, color='black')
    ax.vlines(low + cd, 0.78, 0.82, color='black')
    ax.text(low + cd/2, 0.85, f"CD = {cd:.3f}", ha='center', fontsize=10)

    # Clean up
    ax.axis('off')
    plt.tight_layout()
    plt.show()


# plot_nemenyi_cd(ranks_f1, cd_value, title="F1-Score Comparison (Nemenyi Test)")