import argparse
import yaml

def get_config():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--batch_size', type=int)
    parser.add_argument('--seq_length', type=int)
    parser.add_argument('--model_type', type=str)
    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Overwrite YAML with CLI args
    for k, v in vars(args).items():
        if v is not None:
            config[k] = v

    return config