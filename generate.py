import torch
from model import GPT
from dataset import download_file,CharacterTokenizer


SHAKESPEARE_URL = "https://raw.githubusercontent.com/atilsamancioglu/ShakespeareInput/refs/heads/main/input.txt"


def load_model(check_point_path : str = "checkpoints"):

    device = "cuda" if torch.cuda.is_available() else "cpu"

    download_file(SHAKESPEARE_URL)
    with open("data/shakespeare", "r") as file:
        text = file.read()

    tokenizer = CharacterTokenizer(text)

    checkpoint = torch.load(check_point_path,weights_only=False,map_location=device)

    config = checkpoint["config"]


    model = GPT(
        vocab_size=config['vocab_size'],
        embeding_dim=config['embedding_dim'],
        heads=config['num_heads'],
        num_layers=config['num_layers'],
        block_size=config['block_size'],
        dropout=0.0
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    model = model.to(device)
    model.eval() # make prepared for generate step

    return model,tokenizer,device

def generate(model, tokenizer,prompt : str, device ,max_tokens  : int = 500,temperature : float = 0.5):


    prompt_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(prompt_ids,dtype=torch.long,device=device).unsqueeze(0)

    output_ids = model.generate(input_ids,max_new_tokens=max_tokens,temperature=temperature)


    return tokenizer.decode(output_ids[0])


if __name__ == "__main__":
    model, tokenizer, device = load_model(r"C:\Users\omen\PycharmProjects\ShakespeareGPT\checkpoints\model.pt")
    while True:
        try:
            prompt = input("\nYour prompt: ")

            if prompt.lower() in ['quit', 'exit', 'q']:
                print("Farewell!")
                break

            if not prompt:
                continue

            generated_text = generate(
                model=model,
                tokenizer=tokenizer,
                device=device,
                prompt=prompt,
                max_tokens=500,
                temperature=0.8
            )

            print("\n" + generated_text)

        except KeyboardInterrupt:
            print("\n\nFarewell!")
            break