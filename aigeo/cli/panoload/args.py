import argparse


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "infile",
        type=str,
        help="input JSON with locations. Each location must have either panoid "
        + "or lat/lng for obtaining panoid through Google single_image_search API call",
    )
    parser.add_argument(
        "-z",
        "--zoom",
        type=int,
        choices=range(7),
        default=3,
        help="panorama zoom level",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        type=int,
        default=8,
        help="max number of simultaneously processed locations",
    )
    parser.add_argument(
        "-l",
        "--conn-limit",
        type=int,
        default=64,
        help="max number of simultaneous TCP connections",
    )
    parser.add_argument(
        "--conn-limit-per-host",
        type=int,
        default=0,
        help="max number of simultaneous TCP connections per host",
    )
    parser.add_argument("--json-filename", type=str, default="storage.json", help="name of output JSON")
    parser.add_argument("--images-dir", type=str, default="panoramas", help="name of images directory")
    parser.add_argument("-o", "--output-dir", type=str, required=True, help="output directory for JSON and images")
