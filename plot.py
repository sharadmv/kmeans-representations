import matplotlib.pyplot as plt
import seaborn as sns
from argparse import ArgumentParser
from itertools import cycle

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('file')
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    experiments = {}
    with open(args.file, 'r') as fp:
        for line in fp:
            row = line.strip().split('\t')
            total, top, kmeans, _, train, test = row
            total = int(total)
            top = int(top)
            train = float(train)
            kmeans = kmeans == "True"
            test = float(test)
            frac = float(top) / total
            if total not in experiments:
                experiments[total] = {}
            if kmeans not in experiments[total]:
                experiments[total][kmeans] = {}
            experiments[total][kmeans][frac] = (train, test)
    plt.figure()
    colors = cycle(sns.color_palette())
    for total in experiments.keys():
        color = next(colors)
        for k, results in experiments[total].items():
            result_values = sorted(results.items(), key=lambda x: x[0])
            if k:
                plt.plot([r[0] for r in result_values], [r[1][1] for r in result_values], label="%s (kmeans)" % total, color=color)
            else:
                plt.plot([r[0] for r in result_values], [r[1][1] for r in result_values], '--', label="%s (random)" % total, color=color)
    plt.legend(loc='best')
    plt.ylabel("Test Accuracy")
    plt.xlabel("Fraction of non-zero clusters")
    plt.show()
