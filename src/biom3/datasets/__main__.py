import sys


def run_fetch_dataset():
    from biom3.datasets.fetch import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    sys.exit(main(args))


if __name__ == "__main__":
    run_fetch_dataset()
