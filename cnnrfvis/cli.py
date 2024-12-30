import argparse
from cnnrfvis.core import process_conv_file
from cnnrfvis.core import process_module_file


def parse_arguments():
    parser = argparse.ArgumentParser(description="CNNRFVis command line tool")

    parser.add_argument('--model_file', type=str, help="Path to the model file")
    parser.add_argument('--conv_file', type=str, help="Path to the conv file")
    parser.add_argument('--layer_idx',
                        type=int,
                        help="Convolution layer index to visualize",
                        required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_arguments()
    if args.model_file is not None:
        process_module_file(args.model_file, args.layer_idx)
    else:
        process_conv_file(args.conv_file, args.layer_idx)


if __name__ == "__main__":
    main()
