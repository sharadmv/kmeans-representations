from argparse import ArgumentParser
from sh import python
cmd = "scripts/train_cifar_kmeans.py"

def parse_args():
    argparser = ArgumentParser()
    argparser.add_argument('logfile')
    argparser.add_argument('--random', action='store_true')
    return argparser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    for num_centers in [8000]:
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
                        "_long_sep":" "
                    })
