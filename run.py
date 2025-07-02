import torch
import time
from src.utils.parser import get_config
from src.models.encoderModel import EncoderModel
from src.models.decoderModel import DecoderModel
from src.models.encoderDecoderModel import EncoderDecoderModel

def main():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Bulding model ...")

    if config['model_type'] == 'bert':
        model = EncoderModel(config)
        dummy_input = torch.randint(0, config['vocab_size'], 
                                    (config['batch_size'], config['seq_length']), 
                                    device=device)
        run = lambda: model(dummy_input)

    elif config['model_type'] == 'gpt':
        model = DecoderModel(config)
        dummy_input = torch.randint(0, config['vocab_size'], 
                                    (config['batch_size'], config['seq_length']), 
                                    device=device)
        run = lambda: model(dummy_input, dummy_input)

    elif config['model_type'] == 'bart':
        model = EncoderDecoderModel(config)
        dummy_input = torch.randint(0, config['vocab_size'], 
                                    (config['batch_size'], config['seq_length']), 
                                    device=device)
        run = lambda: model(dummy_input, dummy_input)

    else:
        raise ValueError(f"Unknown model_type: {config['model_type']}")

    model = model.to(device)

    if device.type == "cuda":
        torch.cuda.synchronize()

    print(f"Running on device: {device}")
    start = time.time()
    with torch.no_grad():
        run()
    if device.type == "cuda":
        torch.cuda.synchronize()
    end = time.time()

    print(f"Latency: {(end - start) / 10:.4f} seconds")

if __name__ == '__main__':
    main()
