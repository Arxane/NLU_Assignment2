import torch
import torch.nn as nn
import torch.optim as optim
import random
import os
import matplotlib.pyplot as plt

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
torch.backends.cudnn.benchmark = True

# config
HIDDEN_SIZE_RNN = 256
HIDDEN_SIZE_BLSTM = 128   # keep this lower to reduce overfitting
BATCH_SIZE = 128
EPOCHS_RNN = 40
EPOCHS_BLSTM = 20
EPOCHS_ATTN = 40
STEPS_PER_EPOCH = 200
LR = 0.002
TEMPERATURE = 0.9

OUTPUT_DIR = "problem2_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# load data
with open("TrainingNames.txt", "r", encoding="utf-8") as f:
    names = [line.strip().lower() for line in f if line.strip()]

# vocab
SPECIAL = ["<PAD>", "<SOS>", "<EOS>"]
chars = sorted(list(set("".join(names))))
chars = SPECIAL + chars

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for ch, i in stoi.items()}
vocab_size = len(chars)

PAD_IDX = stoi["<PAD>"]

def encode(name):
    return [stoi["<SOS>"]] + [stoi[c] for c in name] + [stoi["<EOS>"]]

encoded = [encode(n) for n in names]

# batching
def get_batch():
    batch = random.sample(encoded, BATCH_SIZE)
    X, Y = [], []
    for seq in batch:
        X.append(seq[:-1])
        Y.append(seq[1:])
    return X, Y

def pad(seqs):
    max_len = max(len(s) for s in seqs)
    return torch.tensor(
        [s + [PAD_IDX]*(max_len-len(s)) for s in seqs],
        device=device
    )

# models
class RNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, HIDDEN_SIZE_RNN)
        self.rnn = nn.RNN(HIDDEN_SIZE_RNN, HIDDEN_SIZE_RNN, batch_first=True)
        self.fc = nn.Linear(HIDDEN_SIZE_RNN, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)
        return self.fc(out), hidden


class BLSTMModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, HIDDEN_SIZE_BLSTM)
        self.dropout = nn.Dropout(0.3)

        self.lstm = nn.LSTM(
            HIDDEN_SIZE_BLSTM,
            HIDDEN_SIZE_BLSTM,
            batch_first=True,
            bidirectional=True
        )

        self.fc = nn.Linear(HIDDEN_SIZE_BLSTM * 2, vocab_size)

    def forward(self, x, hidden=None):
        x = self.dropout(self.embed(x))
        out, hidden = self.lstm(x, hidden)
        return self.fc(out), hidden


class AttentionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, HIDDEN_SIZE_RNN)
        self.rnn = nn.RNN(HIDDEN_SIZE_RNN, HIDDEN_SIZE_RNN, batch_first=True)
        self.attn = nn.Linear(HIDDEN_SIZE_RNN, 1)
        self.fc = nn.Linear(HIDDEN_SIZE_RNN, vocab_size)

    def forward(self, x, hidden=None):
        x = self.embed(x)
        out, hidden = self.rnn(x, hidden)

        weights = torch.softmax(self.attn(out), dim=1)
        context = torch.sum(weights * out, dim=1, keepdim=True)

        out = out + context
        return self.fc(out), hidden

# training
def train(model, epochs, early_stop=False):
    optimizer = optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)

    model.train()

    for epoch in range(epochs):
        total_loss = 0

        for _ in range(STEPS_PER_EPOCH):
            X, Y = get_batch()
            X = pad(X)
            Y = pad(Y)

            logits, _ = model(X)

            loss = loss_fn(
                logits.view(-1, vocab_size),
                Y.view(-1)
            )

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / STEPS_PER_EPOCH
        print(f"Epoch {epoch+1}/{epochs} | Loss {avg_loss:.4f}")

        # early stop for blstm
        if early_stop and avg_loss < 0.5:
            print("Early stopping triggered")
            break

    # generation
def generate(model, max_len=20):
    model.eval()

    ch = "<SOS>"
    result = ""
    hidden = None
    used = set()

    for _ in range(max_len):
        x = torch.tensor([[stoi[ch]]], device=device)

        with torch.no_grad():
            logits, hidden = model(x, hidden)

        probs = torch.softmax(logits[0, -1] / TEMPERATURE, dim=0)

        # light repetition penalty
        for c in used:
            probs[stoi[c]] *= 0.6

        probs = probs / probs.sum()

        idx = torch.multinomial(probs, 1).item()
        ch = itos[idx]

        if ch == "<EOS>":
            break

        if ch not in ["<PAD>", "<SOS>"]:
            result += ch
            used.add(ch)

    return result

# evaluation
def evaluate(model, train_set):
    generated = [generate(model) for _ in range(1000)]

    novelty = len([n for n in generated if n not in train_set]) / len(generated)
    diversity = len(set(generated)) / len(generated)

    return novelty, diversity, generated

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# run
models = {
    "RNN": (RNNModel().to(device), EPOCHS_RNN, False),
    "BLSTM": (BLSTMModel().to(device), EPOCHS_BLSTM, True),
    "ATTN": (AttentionModel().to(device), EPOCHS_ATTN, False)
}

results = {}

for name, (model, epochs, early_stop) in models.items():
    print(f"\nTraining {name}")
    train(model, epochs, early_stop)

    params = count_params(model)
    novelty, diversity, samples = evaluate(model, set(names))

    print(f"{name} → Novelty {novelty:.3f}, Diversity {diversity:.3f}")

    with open(f"{OUTPUT_DIR}/{name}_samples.txt", "w") as f:
        for s in samples:
            f.write(s + "\n")

    results[name] = {
        "novelty": novelty,
        "diversity": diversity,
        "params": params
    }

# save metrics
with open(f"{OUTPUT_DIR}/metrics.txt", "w") as f:
    for m, v in results.items():
        f.write(f"{m}\n")
        f.write(f"Params: {v['params']}\n")
        f.write(f"Novelty: {v['novelty']:.4f}\n")
        f.write(f"Diversity: {v['diversity']:.4f}\n\n")

# plots
names_list = list(results.keys())
nov = [results[m]["novelty"] for m in names_list]
div = [results[m]["diversity"] for m in names_list]

plt.figure()
plt.bar(names_list, nov)
plt.title("Novelty Comparison")
plt.savefig(f"{OUTPUT_DIR}/novelty.png")
plt.close()

plt.figure()
plt.bar(names_list, div)
plt.title("Diversity Comparison")
plt.savefig(f"{OUTPUT_DIR}/diversity.png")
plt.close()

print("\nDone. Outputs saved in problem2_outputs/")