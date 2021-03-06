from argparse import ArgumentParser
from sh import python
cmd = "scripts/train_mnist_kmeans.py"

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('logfile')
    argparser.add_argument('--random', action='store_true')
    argparser.add_argument('--metric', default='cosine')
    argparser.add_argument('--type', default='similarity')
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for num_centers in [10, 20, 50, 100, 1000, 2000, 4000, 8000]:
        if num_centers < 100:
            increment = int(num_centers / 10)
        else:
            increment = int(num_centers / 100)
        for top_centers in range(increment, num_centers + increment, increment):
            print("Running with parameters: %u / %u" % (top_centers, num_centers))
            python(cmd,
                    **{
                        "num_centers": num_centers,
                        "top_centers":top_centers,
                        "logfile": args.logfile,
                        "random": args.random,
                        "metric": args.metric,
                        "type": args.type,
                        "_long_sep":" "
                    })
