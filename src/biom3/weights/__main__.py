import sys


def run_fetch_weights():
    from biom3.weights.fetch import parse_arguments, main
    args = parse_arguments(sys.argv[1:])
    sys.exit(main(args))


if __name__ == "__main__":
    run_fetch_weights()
