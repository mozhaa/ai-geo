import argparse

from .panoload.args import setup_parser as panoload_setup_parser
from .sample.args import setup_parser as sample_setup_parser


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    subparsers = parser.add_subparsers(dest="subcommand")
    panoload_setup_parser(
        subparsers.add_parser(
            "panoload",
            help="tool for loading full panoramas from Google API",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    sample_setup_parser(
        subparsers.add_parser(
            "sample",
            help="tool for sampling images from panoramas, loaded by panoload",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.subcommand == "panoload":
        from .panoload.main import main

        main(args)
    elif args.subcommand == "sample":
        from .sample.main import main

        main(args)


if __name__ == "__main__":
    main()
