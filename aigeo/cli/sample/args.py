import argparse


def setup_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("input", type=str, help="input JSON with locations")
    parser.add_argument("-s", "--size", type=int, default=512, help="size of generated images")
    parser.add_argument("-b", "--batch-size", type=int, default=8, help="batch size for converting")
    parser.add_argument("-d", "--device", type=str, default="cpu", help="torch device for converting")
    parser.add_argument("-c", "--count", type=int, default=None, help="number of samples")
    parser.add_argument(
        "-p", "--phi", type=float, default=0, help="horizontal angle of camera ([-180, 180], 0=forward)"
    )
    parser.add_argument("-t", "--theta", type=float, default=0, help="vertical angle of camera ([-90, 90], 0=forward)")
    parser.add_argument("-f", "--fov", type=float, default=0.5, help="FOV of camera ([0, 180])")
    parser.add_argument("--json-filename", type=str, default="sample.json", help="name of output JSON")
    parser.add_argument("--images-dir", type=str, default="images", help="name of images directory")
    parser.add_argument("-a", "--append", action="store_true", help="append locations to output instead of overwriting")
    parser.add_argument("-o", "--output", type=str, required=True, help="output directory")
