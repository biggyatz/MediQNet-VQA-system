"""Lightweight MediQNet for the browser.

Image: MobileNetV3-Large (ImageNet-pretrained) fine-tuned on VQA-Med 2019 images.
Question: bag-of-words over the training vocabulary -> linear embedding.
Fusion: concat(image 960-d, question 128-d) -> MLP -> logits over the answer space.
At inference the answer is restricted to the question's category (modality / plane /
organ / abnormality), predicted from the question text by a small logistic model.

Outputs (in out/): mediqnet.onnx, vocab.json (tokens, answers, categories, masks), metrics.json
"""
import json, os, random, re, sys, time
import numpy as np, torch, torch.nn as nn
from PIL import Image
from torchvision import models, transforms

torch.set_num_threads(4); torch.manual_seed(0); random.seed(0); np.random.seed(0)
D = os.environ.get("VQA_DATA", "data"); OUT = os.environ.get("VQA_OUT", "out"); os.makedirs(OUT, exist_ok=True)
CATS = ["modality", "plane", "organ", "abnormality"]
CAT_FILES = {"modality": "C1_Modality", "plane": "C2_Plane", "organ": "C3_Organ", "abnormality": "C4_Abnormality"}

def load_split(split):
    rows = []
    if split == "test":
        for l in open(f"{D}/VQAMed2019Test/VQAMed2019_Test_Questions_w_Ref_Answers.txt"):
            i, c, q, a = l.rstrip("\n").split("|"); rows.append((i, c, q, a))
        img_dir = f"{D}/VQAMed2019Test/VQAMed2019_Test_Images"
    else:
        base = f"{D}/ImageClef-2019-VQA-Med-{'Training' if split == 'train' else 'Validation'}"
        for c in CATS:
            fn = f"{base}/QAPairsByCategory/{CAT_FILES[c]}_{split}.txt"
            for l in open(fn):
                i, q, a = l.rstrip("\n").split("|"); rows.append((i, c, q, a))
        img_dir = f"{base}/{'Train' if split == 'train' else 'Val'}_images"
    return rows, img_dir

tok = lambda q: re.findall(r"[a-z0-9]+", q.lower())
train, tr_dir = load_split("train"); val, va_dir = load_split("val"); test, te_dir = load_split("test")
FULL = "--full" in sys.argv          # retrain on train+val for the deployed model
fit_rows = train + (val if FULL else [])

vocab = sorted({w for _, _, q, _ in fit_rows for w in tok(q)})
widx = {w: i for i, w in enumerate(vocab)}
answers = sorted({a for _, _, _, a in fit_rows})
aidx = {a: i for i, a in enumerate(answers)}
cat_mask = np.zeros((4, len(answers)), dtype=np.float32)
for _, c, _, a in fit_rows: cat_mask[CATS.index(c), aidx[a]] = 1
print(f"train {len(train)} val {len(val)} test {len(test)} | vocab {len(vocab)} answers {len(answers)}", flush=True)

def bow(q):
    v = np.zeros(len(vocab), dtype=np.float32)
    for w in tok(q):
        if w in widx: v[widx[w]] = 1
    return v

# --- question category classifier (logistic regression on BoW, exported as weights) ---
from sklearn.linear_model import LogisticRegression
Xq = np.stack([bow(q) for _, _, q, _ in fit_rows]); yq = [CATS.index(c) for _, c, _, _ in fit_rows]
qcls = LogisticRegression(max_iter=2000, C=5).fit(Xq, yq)
for name, rows in (("val", val), ("test", test)):
    acc = np.mean(qcls.predict(np.stack([bow(q) for _, _, q, _ in rows])) == [CATS.index(c) for _, c, _, _ in rows])
    print(f"category classifier acc {name}: {acc:.4f}", flush=True)

# --- image preprocessing; images are cached as tensors once ---
norm = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
base_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor(), norm])
aug_tf = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
                             transforms.ColorJitter(0.15, 0.15), transforms.ToTensor(), norm])
_cache = {}
def img(path, aug=False):
    if aug:
        return aug_tf(Image.open(path).convert("RGB"))
    if path not in _cache:
        _cache[path] = base_tf(Image.open(path).convert("RGB"))
    return _cache[path]

class Net(nn.Module):
    def encode(self, image):
        return self.pool(self.features(image)).flatten(1)
    def answer(self, feats, question):
        return self.head(torch.cat([feats, self.q(question)], 1))
    def __init__(self, nv, na):
        super().__init__()
        mb = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.IMAGENET1K_V2)
        self.features, self.pool = mb.features, nn.AdaptiveAvgPool2d(1)
        self.q = nn.Sequential(nn.Linear(nv, 128), nn.ReLU())
        self.head = nn.Sequential(nn.Linear(960 + 128, 512), nn.ReLU(), nn.Dropout(0.3), nn.Linear(512, na))
    def forward(self, image, question):
        f = self.pool(self.features(image)).flatten(1)
        return self.head(torch.cat([f, self.q(question)], 1))

net = Net(len(vocab), len(answers))
def path_of(i, split):
    return f"{ {'train': tr_dir, 'val': va_dir, 'test': te_dir}[split] }/{i}.jpg"
fit = [(path_of(i, "train"), c, q, a) for i, c, q, a in train] + ([(path_of(i, "val"), c, q, a) for i, c, q, a in val] if FULL else [])

def evaluate(rows, split):
    net.eval(); correct = {c: [0, 0] for c in CATS}; preds = []
    with torch.no_grad():
        for k in range(0, len(rows), 64):
            batch = rows[k:k + 64]
            X = torch.stack([img(path_of(i, split)) for i, _, _, _ in batch])
            Q = torch.tensor(np.stack([bow(q) for _, _, q, _ in batch]))
            logits = net(X, Q).numpy()
            pc = qcls.predict(Q.numpy())
            for (i, c, q, a), lg, cc in zip(batch, logits, pc):
                lg = np.where(cat_mask[cc] > 0, lg, -1e9)
                p = answers[int(lg.argmax())]; preds.append(p)
                correct[c][0] += p == a; correct[c][1] += 1
    acc = {c: round(v[0] / max(1, v[1]), 4) for c, v in correct.items()}
    acc["overall"] = round(sum(v[0] for v in correct.values()) / len(rows), 4)
    return acc, preds

EPOCHS = int(os.environ.get("EPOCHS", 8))
opt = torch.optim.AdamW([{"params": net.features.parameters(), "lr": 3e-4}, {"params": list(net.q.parameters()) + list(net.head.parameters()), "lr": 1e-3}], weight_decay=1e-4)
lossf = nn.CrossEntropyLoss(label_smoothing=0.1)
# Group QA pairs by image so each image is encoded once per step and shared by its questions.
by_img = {}
for p, c, q, a in fit: by_img.setdefault(p, []).append((q, a))
img_list = list(by_img)
steps = EPOCHS * ((len(img_list) + 31) // 32)
sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[3e-4, 1e-3], total_steps=steps)
for ep in range(EPOCHS):
    net.train(); random.shuffle(img_list); t0 = time.time(); tl = n = 0
    for k in range(0, len(img_list), 32):
        paths = img_list[k:k + 32]
        feats = net.encode(torch.stack([img(p, aug=True) for p in paths]))
        idx, Q, y = [], [], []
        for j, p in enumerate(paths):
            for q, a in by_img[p]:
                idx.append(j); Q.append(bow(q)); y.append(aidx[a])
        logits = net.answer(feats[torch.tensor(idx)], torch.tensor(np.stack(Q)))
        loss = lossf(logits, torch.tensor(y)); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        tl += loss.item() * len(y); n += len(y)
    msg = f"epoch {ep + 1}/{EPOCHS} loss {tl / n:.3f} {time.time() - t0:.0f}s"
    if not FULL:
        msg += f" val {evaluate(val, 'val')[0]}"
    print(msg, flush=True)

test_acc, test_preds = evaluate(test, "test")
print("TEST", test_acc, flush=True)
json.dump({"test": test_acc, "full": FULL, "epochs": EPOCHS, "n_train": len(fit)}, open(f"{OUT}/metrics{'_full' if FULL else ''}.json", "w"), indent=1)

if FULL:
    net.eval()
    torch.onnx.export(net, (torch.zeros(1, 3, 224, 224), torch.zeros(1, len(vocab))), f"{OUT}/mediqnet.onnx",
                      input_names=["image", "question"], output_names=["logits"], opset_version=17,
                      dynamic_axes={"image": {0: "b"}, "question": {0: "b"}, "logits": {0: "b"}})
    json.dump({"vocab": vocab, "answers": answers, "categories": CATS, "cat_mask": cat_mask.astype(int).tolist(),
               "qcls_coef": qcls.coef_.tolist(), "qcls_intercept": qcls.intercept_.tolist(),
               "mean": [0.485, 0.456, 0.406], "std": [0.229, 0.224, 0.225]}, open(f"{OUT}/vocab.json", "w"))
    json.dump([{"id": i, "category": c, "question": q, "answer": a, "pred": p} for (i, c, q, a), p in zip(test, test_preds)],
              open(f"{OUT}/test_predictions.json", "w"))
    print("exported", flush=True)
