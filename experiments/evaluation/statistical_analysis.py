import numpy as np
from scipy.stats import rankdata, studentized_range, wilcoxon, friedmanchisquare
import math
from collections import defaultdict
from itertools import combinations                                                                                                                                                                                                              
from evaluation.file_logger import log
import matplotlib.pyplot as plt
import networkx as nx


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
        #  'acc': {'friedman_anova': 123, 'p_val': 0.05, 'chi_square': 0.3, 'nemenyi': 34.3},
        #  (...)
        #  'recall': {'friedman_anova': 23, 'p_val': 0.05, 'chi_square': 0.3,'nemenyi': 24.1}
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
            ]).T   # shape: (k, n)            print(f"k = {k}, n= {n}")

            log(f"ranks = " + str(ranks))

            avg_ranks = ranks.mean(axis=1)

            log(f"avg_anks = " + str(avg_ranks))

            CD = self._nemenyi_cd(k, n)

            log("CD = " + str(CD))

            method_names = [f"G{i}" for i in range(k)]

            comparisons = []
            for i, j in combinations(range(k), 2):
                diff = abs(avg_ranks[i] - avg_ranks[j])
                comparisons.append({
                    "method_1": method_names[i],
                    "method_2": method_names[j],
                    "rank_diff": float(diff),
                    "significant": float(diff) > float(CD)
                })

        log("NEMENYI RESULTS: ")
        for comp in comparisons:
            log("key - " + str(key))
            log(comparisons)
            log("___________________")


    def _nemenyi_cd(self, k, n, alpha=0.05):

        q_alpha = studentized_range.ppf(1 - alpha, k, np.inf)
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

            for i, j in combinations(range(k), 2):
                stat, p = wilcoxon(
                    results[i],
                    results[j],
                    zero_method="wilcox",
                    alternative="two-sided"
                )
                p_values.append(p)
                pairs.append((i, j, stat))

            significant = self._holm_correction(p_values, alpha)

            results_table = []
            for idx, ((i, j, stat), p, sig) in enumerate(
                zip(pairs, p_values, significant)
            ):
                results_table.append({
                    "key": key,
                    "method_1": method_names[i],
                    "method_2": method_names[j],
                    "W": stat,
                    "p_uncorrected": p,
                    "significant": sig
                })
            print(f"\n {key}")
            for r in results_table:
                log(
                    str(r['method_1']) + " vs " + str(r['method_2']) + ": " 
                    + "W= " + str(r['W']) + " p= " + str(round(r['p_uncorrected'], 4)) + ", "
                    + str('SIGNIFICANT' if r['significant'] else 'n.s.')
                )


def plot_nemenyi_cd(ranks, cd, title="Critical Difference (Nemenyi)"):
    """
    Plots a Critical Difference diagram for the Nemenyi test.
    
    Args:
        ranks (dict): Dictionary of {method_name: average_rank}
        cd (float): The Critical Difference value (e.g., 1.21)
        title (str): Title of the plot
    """
    # 1. Sort ranks
    sorted_ranks = sorted(ranks.items(), key=lambda x: x[1])
    methods, rank_values = zip(*sorted_ranks)
    
    # 2. Setup Plot
    fig, ax = plt.subplots(figsize=(8, 3))
    ax.set_title(title, pad=20, fontsize=14)
    
    # Define range for the axis (invert so 1 is on the left or right depending on preference)
    # Standard CD diagrams usually put lower rank (better) on the left or right.
    # We will put Best (1.0) on the Left.
    low, high = 1, len(methods)
    ax.set_xlim(low - 0.5, high + 0.5)
    ax.set_ylim(0, 1)
    
    # Draw the main axis line
    ax.hlines(0.2, low, high, color='black', linewidth=1.5)
    
    # Add Tick Markers
    for i in range(low, high + 1):
        ax.vlines(i, 0.2, 0.25, color='black')
        ax.text(i, 0.28, str(i), ha='center', va='bottom', fontsize=10)
    
    # 3. Plot Methods and Ranks
    # We stagger heights to prevent text overlap if ranks are close
    levels = [0.2, 0.05, 0.2, 0.05] # Toggle heights for labels
    
    for i, (method, r) in enumerate(zip(methods, rank_values)):
        # Draw marker on line
        ax.plot(r, 0.2, 'o', color='black', markersize=6)
        
        # Draw text label with line pointing to marker
        level = 0.05 if i % 2 == 0 else 0.45 # Alternate label positions (top/bottom)
        va = 'top' if level < 0.2 else 'bottom'
        
        # Line from axis to label
        ax.plot([r, r], [0.2, level], '-', color='gray', linewidth=0.8)
        
        # Label text
        ax.text(r, level, f"{method}\n({r:.2f})", ha='center', va=va, fontsize=11, fontweight='bold')

    # 4. Draw Connecting Bars for Non-Significant Differences
    # We look for groups where max_rank - min_rank < CD
    # Simple logic: Connect consecutive models if diff < CD
    # (For 3 models, we just check pairs manually for the line drawing)
    
    y_bar = 0.25 # Height for connection bars
    
    # Check cliques/groups
    # Group 1: G0 & G1? (2.93 - 1.87 = 1.06 < 1.21) -> YES
    # Group 2: G1 & G2? (1.87 - 1.20 = 0.67 < 1.21) -> YES
    # Group 3: G0 & G2? (2.93 - 1.20 = 1.73 > 1.21) -> NO
    
    # Draw bar for G2-G1
    if abs(ranks['G1'] - ranks['G2']) < cd:
        ax.hlines(y_bar + 0.02, ranks['G2'], ranks['G1'], color='red', linewidth=3, label='Not Sig.')
        
    # Draw bar for G1-G0
    if abs(ranks['G0'] - ranks['G1']) < cd:
         # Shift y slightly up if overlapping, or use same line if distinct
        ax.hlines(y_bar + 0.02, ranks['G1'], ranks['G0'], color='red', linewidth=3)

    # 5. Add CD ruler (visual guide)
    ax.hlines(0.8, low, low + cd, color='black', linewidth=2)
    ax.vlines(low, 0.78, 0.82, color='black')
    ax.vlines(low + cd, 0.78, 0.82, color='black')
    ax.text(low + cd/2, 0.85, f"CD = {cd:.3f}", ha='center', fontsize=10)

    # Clean up
    ax.axis('off')
    plt.tight_layout()
    plt.show()

# --- INPUT DATA FROM YOUR RESULTS ---
# Ranks for F1-Score
ranks_f1 = {
    'G0': 2.933,
    'G1': 1.867,
    'G2': 1.200
}
cd_value = 1.210

# plot_nemenyi_cd(ranks_f1, cd_value, title="F1-Score Comparison (Nemenyi Test)")