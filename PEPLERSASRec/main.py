import os
import math
import torch
import argparse
from transformers import GPT2Tokenizer, AdamW
from transformers import GPT2TokenizerFast
import torch.nn.functional as F
# from VersionModule import ContinuousPromptLearning
# from module import ContinuousPromptLearning
from AspectsModule import ContinuousPromptLearning
# from ModuleHistory import ContinuousPromptLearning
from utils import rouge_score, bleu_score, DataLoader2, Batchify3, now_time, ids2tokens, ids2tokensReal, unique_sentence_percent, \
    feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity
# from utils import rouge_score, bleu_score, DataLoader, Batchify, now_time, ids2tokens, unique_sentence_percent, \
#     feature_detect, feature_matching_ratio, feature_coverage_ratio, feature_diversity


parser = argparse.ArgumentParser(description='PErsonalized Prompt Learning for Explainable Recommendation (PEPLER)')
parser.add_argument('--data_path', type=str, default=None,
                    help='path for loading the pickle data')
parser.add_argument('--index_dir', type=str, default=None,
                     help='load indexes')
parser.add_argument('--lr', type=float, default=0.001,
                    help='learning rate')
parser.add_argument('--epochs', type=int, default=100,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=128, #128
                    help='batch size')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_interval', type=int, default=200,
                    help='report interval')
parser.add_argument('--checkpoint', type=str, default='./pepler/',
                    help='directory to save the final model')
parser.add_argument('--outf', type=str, default='generated.txt',
                    help='output file for generated text')
parser.add_argument('--endure_times1', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--endure_times', type=int, default=5,
                    help='the maximum endure times of loss increasing on validation')
parser.add_argument('--words', type=int, default=20,
                    help='number of words to generate for each sample')
args = parser.parse_args()

if args.data_path is None:
    parser.error('--data_path should be provided for loading data')
if args.index_dir is None:
    parser.error('--index_dir should be provided for loading data splits')

print('-' * 40 + 'ARGUMENTS' + '-' * 40)
for arg in vars(args):
    print('{:40} {}'.format(arg, getattr(args, arg)))
print('-' * 40 + 'ARGUMENTS' + '-' * 40)

if torch.cuda.is_available():
    if not args.cuda:
        print(now_time() + 'WARNING: You have a CUDA device, so you should probably run with --cuda')
device = torch.device('cuda' if args.cuda else 'cpu')

if not os.path.exists(args.checkpoint):
    os.makedirs(args.checkpoint)
model_path = os.path.join(args.checkpoint, 'model.pt')
prediction_path = os.path.join(args.checkpoint, args.outf)

###############################################################################
# Load data
###############################################################################

print(now_time() + 'Loading data')
bos = '<bos>'
eos = '<eos>'
pad = '<pad>'
# tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
tokenizer = GPT2TokenizerFast.from_pretrained('gpt2', bos_token=bos, eos_token=eos, pad_token=pad)
tokenizerFast = GPT2TokenizerFast.from_pretrained("gpt2")
corpus = DataLoader2(args.data_path, args.index_dir, tokenizer, tokenizerFast, args.words)
# corpus = DataLoader2(args.data_path, tokenizer, tokenizerFast, args.words)
feature_set = corpus.feature_set
train_data = Batchify3(corpus.train, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,args.words, args.batch_size, shuffle=True) # tokenizerFast,
val_data = Batchify3(corpus.valid, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,args.words, args.batch_size)
test_data = Batchify3(corpus.test, corpus.user2feature, corpus.item2feature, tokenizer, bos, eos,args.words, args.batch_size)

###############################################################################
# Build the model
###############################################################################
nuser = len(corpus.user_dict)
nitem = len(corpus.item_dict)
ntoken = len(tokenizer)
# Path to the saved model
# saved_model_path = "/content/drive/MyDrive/SFTmodel/tripadvisor/model.pt"
# # Check if a saved model exists
# if os.path.exists(saved_model_path):
#     print(now_time() + "Loading the saved model from:", saved_model_path)
#     model = torch.load(saved_model_path).to(device)
#     # You may need to reinitialize the optimizer after loading the model
#     optimizer = AdamW(model.parameters(), lr=args.lr)
# else:
#     print(now_time() + "No saved model found. Initializing a new model.")
#     model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)
#     model.resize_token_embeddings(ntoken)  # Update embedding table
#     model.to(device)
#     optimizer = AdamW(model.parameters(), lr=args.lr)

# Proceed with the training as before
model = ContinuousPromptLearning.from_pretrained('gpt2', nuser, nitem)#gpt2-large
model.resize_token_embeddings(ntoken)  # three tokens added, update embedding table
model.to(device)
optimizer = AdamW(model.parameters(), lr=args.lr)

###############################################################################
# Training code
###############################################################################

def train(data):
    # Turn on training mode which enables dropout.
    model.train()
    text_loss = 0.
    total_sample = 0
    while True:
        user, item, _ , seq, feat , mask = data.next_batch()  # data.step += 1
        user = user.to(device)  # (batch_size,)
        item = item.to(device)
        seq = seq.to(device)  # (batch_size, seq_len)
        mask = mask.to(device)
        # context = context.to(device)
        feat = feat.to(device)
        optimizer.zero_grad()
        outputs = model(user, item, seq, feat, mask)
        loss = outputs.loss
        # torch.cuda.empty_cache()
        loss.backward()
        optimizer.step()

        batch_size = user.size(0)
        text_loss += batch_size * loss.item()
        total_sample += batch_size

        if data.step % args.log_interval == 0 or data.step == data.total_step:
            cur_t_loss = text_loss / total_sample
            print(now_time() + 'text ppl {:4.4f} | {:5d}/{:5d} batches'.format(math.exp(cur_t_loss), data.step, data.total_step))
            text_loss = 0.
            total_sample = 0
        if data.step == data.total_step:
            break

def evaluate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    text_loss = 0.
    total_sample = 0
    with torch.no_grad():
        while True:
            user, item, _, seq, feat, mask = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            seq = seq.to(device)  # (batch_size, seq_len)
            mask = mask.to(device)
            # context = context.to(device)
            feat = feat.to(device)
            outputs = model(user, item, seq, feat, mask)
            loss = outputs.loss

            batch_size = user.size(0)
            text_loss += batch_size * loss.item()
            total_sample += batch_size

            if data.step == data.total_step:
                break
    return text_loss / total_sample


def generate(data):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    idss_predict = []
    with torch.no_grad():
        while True:
            user, item, _, seq, feat, _ = data.next_batch()  # data.step += 1
            user = user.to(device)  # (batch_size,)
            item = item.to(device)
            text = seq[:, :1].to(device)  # bos, (batch_size, 1)
            for idx in range(seq.size(1)):
                # produce a word at each step
                outputs = model(user, item, text, feat, None)
                last_token = outputs.logits[:, -1, :]  # the last token, (batch_size, ntoken)
                word_prob = torch.softmax(last_token, dim=-1)
                token = torch.argmax(word_prob, dim=1, keepdim=True)  # (batch_size, 1), pick the one with the largest probability
                text = torch.cat([text, token], 1)  # (batch_size, len++)
            ids = text[:, 1:].tolist()  # remove bos, (batch_size, seq_len)
            idss_predict.extend(ids)

            if data.step == data.total_step:
                break
    return idss_predict

'''
print(now_time() + 'Tuning Prompt Only')
# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times1:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break

# Load the best saved model.

with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


print(now_time() + 'Tuning both Prompt and LM')
for param in model.parameters():
    param.requires_grad = True
optimizer = AdamW(model.parameters(), lr=args.lr)

# Loop over epochs.
best_val_loss = float('inf')
endure_count = 0
for epoch in range(1, args.epochs + 1):
    print(now_time() + 'epoch {}'.format(epoch))
    train(train_data)
    val_loss = evaluate(val_data)
    print(now_time() + 'text ppl {:4.4f} | valid loss {:4.4f} on validation'.format(math.exp(val_loss), val_loss))
    # Save the model if the validation loss is the best we've seen so far.
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        with open(model_path, 'wb') as f:
            torch.save(model, f)
    else:
        endure_count += 1
        print(now_time() + 'Endured {} time(s)'.format(endure_count))
        if endure_count == args.endure_times:
            print(now_time() + 'Cannot endure it anymore | Exiting from early stop')
            break
'''
model_path = '/content/drive/MyDrive/SFTmodel/tripadvisor/model.pt'
# Load the best saved model.
with open(model_path, 'rb') as f:
    model = torch.load(f).to(device)


# Run on test data.
test_loss = evaluate(test_data)
# test_loss = evaluate(val_data)
print('=' * 89)
print(now_time() + 'text ppl {:4.4f} on test | End of training'.format(math.exp(test_loss)))
print(now_time() + 'Generating text')
idss_predicted = generate(test_data) #test_data

tokens_test = [ids2tokensReal(ids[1:], tokenizer, eos_token="<eos>", pad_token="<pad>") for ids in test_data.seq.tolist()]
tokens_predict = [ids2tokens(ids, tokenizer, eos) for ids in idss_predicted]
BLEU1 = bleu_score(tokens_test, tokens_predict, n_gram=1, smooth=False)
print(now_time() + 'BLEU-1 {:7.4f}'.format(BLEU1))
BLEU4 = bleu_score(tokens_test, tokens_predict, n_gram=4, smooth=False)
print(now_time() + 'BLEU-4 {:7.4f}'.format(BLEU4))
USR, USN = unique_sentence_percent(tokens_predict)
print(now_time() + 'USR {:7.4f} | USN {:7}'.format(USR, USN))
feature_batch = feature_detect(tokens_predict, feature_set)
# print("Feature Batch:", feature_batch[:5])  # Print first 5 batches of detected features
# print("Test Features:", test_data.feature[:5].tolist())  # Ensure this is a list for easy readability
DIV = feature_diversity(feature_batch)  # time-consuming
print(now_time() + 'DIV {:7.4f}'.format(DIV))
FCR = feature_coverage_ratio(feature_batch, feature_set)
print(now_time() + 'FCR {:7.4f}'.format(FCR))
FMR = feature_matching_ratio(feature_batch, test_data.feature)
print(now_time() + 'FMR {:7.4f}'.format(FMR))
text_test = [' '.join(tokens) for tokens in tokens_test]
text_predict = [' '.join(tokens) for tokens in tokens_predict]
ROUGE = rouge_score(text_test, text_predict)  # a dictionary
for (k, v) in ROUGE.items():
    print(now_time() + '{} {:7.4f}'.format(k, v))
text_out = ''

for (real, fake) in zip(text_test, text_predict):
    text_out += '{}\n{}\n\n'.format(real, fake)
with open(prediction_path, 'w', encoding='utf-8') as f:
    f.write(text_out)
print(now_time() + 'Generated text saved to ({})'.format(prediction_path))
