"""
This script is designed for text generation in Direct Preference Optimization (DPO) training using a pretrained GPT-2 model. 
It loads a dataset, processes it into batches, and generates text based on user-item interactions.
"""
import os
import json
import torch
import argparse
from transformers import GPT2TokenizerFast
from AspectsModule import ContinuousPromptLearning
from utils import DataLoader2, Batchify3, now_time, ids2tokensReal

# Argument Parsing
parser = argparse.ArgumentParser(description='Text Generation for DPO Training')
parser.add_argument('--data_path', type=str, default='/content/data.pkl', help='Path to data file')
parser.add_argument('--index_dir', type=str, default='/content/index/', help='Path to index directory')
# parser.add_argument('--model_path', type=str, default='/content/model.pt', help='Path to the saved model')
parser.add_argument('--output_json', type=str, default='/content/generated_data.json', help='Output JSON file for generated text')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size for data loading')
parser.add_argument('--cuda', action='store_true', help='Use CUDA for inference')
args = parser.parse_args()

# Device Configuration
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

# Load Tokenizer and Data
print(now_time() + 'Loading tokenizer and data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
corpus = DataLoader2(args.data_path, args.index_dir, tokenizer, tokenizer, seq_len=20)
train_data = Batchify3(corpus.train, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, 20, args.batch_size)
val_data = Batchify3(corpus.valid, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,20, args.batch_size)
test_data = Batchify3(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,20, args.batch_size)

# Load Model
print(now_time() + 'Loading model')
# if not os.path.exists(args.model_path):
#     raise FileNotFoundError(f"Model file not found: {args.model_path}")
model_path = '/content/drive/MyDrive/SFTmodel/tripadvisor/model.pt'
model = torch.load(model_path, map_location=device).to(device)
# Generate Text
def generate_text(data, model, tokenizer, eos_token="<eos>", pad_token="<pad>"):
    """
    Generates text using a trained model by autoregressively predicting tokens.

    This function takes a dataset iterator, a trained language model, and a tokenizer to generate 
    text sequences. The generated responses are compared to ground truth sequences, forming 
    pairs of "chosen" (ground truth) and "rejected" (model-generated) responses.

    Args:
        data (DatasetIterator): A dataset iterator that provides user-item interaction data.
        model (torch.nn.Module): The trained language model used for text generation.
        tokenizer (transformers.PreTrainedTokenizer): Tokenizer to convert token IDs to text.
        eos_token (str, optional): The end-of-sequence token. Default is "<eos>".
        pad_token (str, optional): The padding token. Default is "<pad>".

    Returns:
        list of dict: A list of dictionaries, each containing:
            - "chosen" (str): The ground truth text.
            - "rejected" (str): The model-generated text.

    Example:
        >>> generated_pairs = generate_text(train_data, model, tokenizer)
        >>> print(generated_pairs[0])
        {"chosen": "This is the expected response.", "rejected": "This is the generated response."}
    """
    model.eval()
    generated_pairs = []
    with torch.no_grad():
        while True:
            # Retrieve batch data
            user, item, _, seq, feat, _ = data.next_batch()
            user = user.to(device)
            item = item.to(device)
            text = seq[:, :1].to(device)  # BOS (beginning of sequence) token

            # Autoregressive text generation
            for _ in range(seq.size(1)):
                outputs = model(user, item, text, feat, None)
                last_token = outputs.logits[:, -1, :]
                next_token = torch.argmax(torch.softmax(last_token, dim=-1), dim=1, keepdim=True)
                text = torch.cat([text, next_token], dim=1)

            # Convert generated IDs to text
            predicted_ids = text[:, 1:].tolist()  # Remove BOS token
            ground_truth_ids = seq[:, 1:].tolist()  # Remove BOS token

            ground_truth_texts = [ids2tokensReal(ids, tokenizer, eos_token, pad_token) for ids in ground_truth_ids]
            predicted_texts = [ids2tokensReal(ids, tokenizer, eos_token, pad_token) for ids in predicted_ids]

            # Store results as {chosen: ground truth, rejected: generated response}
            generated_pairs.extend([
                {"chosen": " ".join(gt), "rejected": " ".join(pred)}
                for gt, pred in zip(ground_truth_texts, predicted_texts)
            ])

            # Stop when the dataset is fully processed
            if data.step == data.total_step:
                break

    return generated_pairs


# Generate text for training data
print(now_time() + " Generating text for training data")
generated_data = generate_text(train_data, model, tokenizer)

# Save Generated Data as JSON
with open(args.output_json, "w", encoding="utf-8") as f:
    json.dump(generated_data, f, indent=4, ensure_ascii=False)

print(now_time() + f" Generated data saved to {args.output_json}")
