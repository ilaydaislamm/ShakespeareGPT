import os
import time
import torch
from model import GPT
from dataset import load_data, get_batch


EMBEDDING_DIM = 384    # Size of embeddings (how big each vector is)
NUM_HEADS = 6          # Number of attention heads
NUM_LAYERS = 6         # Number of transformer blocks
BLOCK_SIZE = 256       # Maximum sequence length
DROPOUT = 0.1          # Dropout rate for regularization

# Training settings
BATCH_SIZE = 64        # Number of sequences per batch
MAX_ITERS = 5000       # Total training iterations
EVAL_INTERVAL = 500    # Evaluate every N iterations
LEARNING_RATE = 3e-4   # Learning rate
WARMUP_ITERS = 100     # Warmup iterations (gradually increase LR)

# System
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CHECKPOINT_PATH = "checkpoints/model.pt"

def get_learning_rate(iteration):
    if iteration < WARMUP_ITERS:
        return LEARNING_RATE * (iteration / WARMUP_ITERS)
    return LEARNING_RATE

@torch.no_grad()
def evaluate(model, train_data,test_data):

    model.eval()

    results = {}

    data = [("train", train_data),("test", test_data)]

    for name, data in data:

        total_loss = 0
        for _ in range(100):
            X,y = get_batch(data,BLOCK_SIZE,BATCH_SIZE)
            X,y = X.to(DEVICE),y.to(DEVICE)

            logits,loss = model(X,y)

            total_loss += loss.item()

        results[name] = total_loss / 100

    model.train()
    return results

def generate_sample(model, tokenizer):

    prompt = "ROMEO : "

    prompt_ids = tokenizer.encode(prompt)

    input_ids = torch.tensor(prompt_ids,dtype=torch.long,device=DEVICE).unsqueeze(0)

    output_ids = model.generate(input_ids,max_new_tokens=200,temperature=0.8)

    model.train()
    return tokenizer.decode(output_ids[0])





def train():

    train_data,test_data,tokenizer = load_data() #train-test-split
    vocab_size = tokenizer.vocabulary_size

    model = GPT(
        vocab_size=vocab_size,
        embeding_dim=EMBEDDING_DIM,
        heads=NUM_HEADS,
        num_layers=NUM_LAYERS,
        block_size=BLOCK_SIZE,
        dropout=DROPOUT
    )

    model = model.to(DEVICE)

    optim = torch.optim.AdamW(model.parameters(),lr=LEARNING_RATE)

    os.makedirs("checkpoints",exist_ok=True)
    start_time = time.time()

    for iteration in range(MAX_ITERS): #train iteration

        lr = get_learning_rate(iteration) #dynamic lr
        for param_groups in optim.param_groups:
            param_groups["lr"] = lr


        X,y = get_batch(train_data,BLOCK_SIZE,BATCH_SIZE) #divide data into batches (randomly on each iteration)

        X,y = X.to(DEVICE),y.to(DEVICE)

        logits,loss = model(X,y)

        optim.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(),1)

        optim.step()

        if iteration % EVAL_INTERVAL == 0:
            results = evaluate(model,train_data,test_data)

            elapsed = time.time() - start_time

            print(f"Iter {iteration:5d} | "
                  f"Train Loss: {results['train']:.4f} | "
                  f"Val Loss: {results['test']:.4f} | "
                  f"LR: {lr:.2e} | "
                  f"Time: {elapsed:.0f}s")

            # Show a sample generation
            if iteration > 0:
                print("\n--- Sample ---")
                print(generate_sample(model, tokenizer)[:400])
                print("--------------\n")

    checkpoint = {
        'model_state_dict': model.state_dict(),
        'iteration': MAX_ITERS,
        'test_loss': results['test'],
        'config': {
            'vocab_size': vocab_size,
            'embedding_dim': EMBEDDING_DIM,
            'num_heads': NUM_HEADS,
            'num_layers': NUM_LAYERS,
            'block_size': BLOCK_SIZE,
        }
    }
    torch.save(checkpoint, CHECKPOINT_PATH)

    total_time = time.time() - start_time
    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Total time: {total_time / 60:.1f} minutes")
    print(f"Final Test loss: {results['test_loss']:.4f}")
    print(f"Model saved to: {CHECKPOINT_PATH}")


if __name__ == "__main__":
    train()