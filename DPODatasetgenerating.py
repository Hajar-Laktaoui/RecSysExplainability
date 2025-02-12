import os
import json
import pickle
import torch
import time
import argparse
from transformers import GPT2TokenizerFast
from AspectsModule import ContinuousPromptLearning
from utils import DataLoader2, Batchify3, now_time, ids2tokensReal

# Argument Parsing
parser = argparse.ArgumentParser(description='Generate Sentences')
parser.add_argument('--data_path', type=str, default='/content/drive/MyDrive/TripAdvisor/reviews.pickle', help='Path to dataset')
parser.add_argument('--index_dir', type=str, default='/content/index/', help='Path to index directory')
parser.add_argument('--output_json', type=str, default='/content/generated_sentences.json', help='Output JSON file')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for data loading')
parser.add_argument('--cuda', action='store_true', help='Use CUDA for inference')
args = parser.parse_args()

# Device Configuration
device = torch.device('cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

# Load Tokenizer
print(now_time() + ' Loading tokenizer...')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)

# Load Dataset
print(now_time() + ' Loading dataset...')
corpus = DataLoader2(args.data_path, args.index_dir, tokenizer, tokenizer, seq_len=20)
# train_data = Batchify3(corpus.train, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, 20, args.batch_size)
# val_data = Batchify3(corpus.valid, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, 20, args.batch_size)
# test_data = Batchify3(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, 20, args.batch_size)

full_dataset =  corpus.train + corpus.valid + corpus.test
# Convert dataset into batches
data_batches = Batchify3(full_dataset, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos, 20, args.batch_size)

# Load Model
print(now_time() + ' Loading model...')
model_path = '/home/hajar.laktaoui/lustre/robust_ml-um6p-st-sccs-iqcbvkbobtq/users/hajar.laktaoui/Implementation4TkL/PEPLERSASRec/MoviesAndTVij/model.pt'
# '/home/hajar.laktaoui/lustre/robust_ml-um6p-st-sccs-iqcbvkbobtq/users/hajar.laktaoui/Implementation4TkL/PEPLERSASRec/TripAdvisor/tripadvisorij/model.pt'
model = torch.load(model_path, map_location=device).to(device)

# Generate Sentences
def generate_sentences(data, model, tokenizer, corpus, eos_token="<eos>", pad_token="<pad>"):
    model.eval()
    generated_sentences = []
    
    with torch.no_grad():
        for batch_idx in range(data.total_step):
            user, item, _, seq, feat, _ = data.next_batch()
            text = seq[:, :1].to(device)  # Start with <bos>
            
            max_length = min(20, seq.size(1))  # Limit to 10 tokens max
            
            for _ in range(max_length):
                outputs = model(user.to(device), item.to(device), text, feat.to(device), None)
                last_token_logits = outputs.logits[:, -1, :]
                next_token = torch.argmax(last_token_logits, dim=-1, keepdim=True)  # Select most probable token
                
                # Stop if all sentences generate <eos>
                if (next_token == tokenizer.eos_token_id).all():
                    break
                
                text = torch.cat([text, next_token], dim=1)
            
            predicted_ids = text[:, 1:].tolist()
            ground_truth_ids = seq[:, 1:].tolist()
            
            ground_truth_texts = [ids2tokensReal(ids, tokenizer, eos_token, pad_token) for ids in ground_truth_ids]
            predicted_texts = [ids2tokensReal(ids, tokenizer, eos_token, pad_token) for ids in predicted_ids]

            for i in range(len(user)):  # Maintain loop
                original_user_id = corpus.user_dict.idx2entity[user[i].item()]  # Retrieve original user ID
                original_item_id = corpus.item_dict.idx2entity[item[i].item()]  # Retrieve original item ID

                generated_sentences.append({
                    "user_id": original_user_id,  # Store the original dataset user ID
                    "item_id": original_item_id,  # Store the original dataset item ID
                    "chosen": " ".join(ground_truth_texts[i]),
                    "rejected": " ".join(predicted_texts[i])
                })

            # Periodic Checkpoint Save
            if batch_idx % 10 == 0:
                with open(args.output_json, 'w', encoding='utf-8') as f:
                    json.dump(generated_sentences, f, indent=4, ensure_ascii=False)
                print(now_time() + f" Checkpoint saved at batch {batch_idx}")

    return generated_sentences


# Generate and save sentences
print(now_time() + ' Generating sentences...')
start_time = time.time()
generated_data = generate_sentences(data_batches, model, tokenizer, corpus)
print(f"Generation time: {time.time() - start_time:.2f} seconds")

# Save final results
with open(args.output_json, 'w', encoding='utf-8') as f:
    json.dump(generated_data, f, indent=4, ensure_ascii=False)

print(now_time() + f' Final generated sentences saved to {args.output_json}')
