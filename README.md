# MediQNet — Medical Visual Question Answering

**Live demo:** <https://biggyatz.github.io/MediQNet-VQA-system/>. Pick a radiology image (or upload one), ask about its modality, plane, organ system or abnormality, and get an answer with confidence scores. The network runs entirely in your browser.

MediQNet answers natural-language questions about radiology images
("what plane is this MRI taken in?", "is this a T1-weighted image?",
"what organ system is shown?"). It was built as a capstone project on the
**ImageCLEF VQA-Med 2019** benchmark.

## How it works

```
question ──► BioBERT (dmis-lab/biobert-v1.1) ──► text embedding ─┐
                                                                 ├─► concat ─► fusion MLP ─► classifier (1,748 answers)
image ─────► Swin-Tiny (microsoft/swin-tiny-patch4-window7-224) ─► image embedding ─┘
```

* **Text encoder** — BioBERT, a BERT model pre-trained on PubMed abstracts, so
  medical vocabulary is understood out of the box.
* **Image encoder** — Swin Transformer (tiny, 224×224).
* **Fusion** — the two pooled embeddings are concatenated and passed through a
  fusion layer (Linear 512 → ReLU → Dropout 0.5), then a classifier over the answer space
  (`answer_space.txt`, 1,748 possible answers).
* **Metrics** — accuracy, macro F1 / precision / recall and **WUPS**
  (Wu-Palmer similarity over WordNet), which gives partial credit for
  semantically close answers.

## Repository layout

| Path | What it is |
| --- | --- |
| `data_exploration.ipynb` | EDA: question categories (modality, plane, organ system, abnormality), answer distribution, train/val/test CSV export |
| `model_train.ipynb` | Multimodal collator, `MultimodalVQAModel`, training with the 🤗 `Trainer`, evaluation |
| `cudacheck.ipynb` | Quick check that PyTorch sees the GPU |
| `web/` | In-browser demo (ONNX model, sample test images, page) |
| `training/train_vqa.py` | Trains and exports the compact demo model |
| `data_train.csv` / `data_val.csv` / `data_test.csv` | 12,792 / 2,000 / 500 question–answer pairs (`img_id,question,answer`) |
| `answer_space.txt` | All answer labels the classifier predicts over |
| `README-VQA-Med-2019.md`, `README-VQA-Med-2019-Data.txt` | The original dataset documentation |

## Running it

1. Download the VQA-Med 2019 images from Zenodo
   (<https://zenodo.org/records/10499039>) and unzip them so each image is at
   `train/Train_images/<img_id>.jpg` (the `train/` folder is git-ignored).
2. Create an environment with a CUDA GPU and install dependencies:
   ```bash
   python -m venv .venv && source .venv/bin/activate
   pip install -r requirements.txt
   python -c "import nltk; nltk.download('wordnet')"
   ```
3. Run `cudacheck.ipynb`, then `data_exploration.ipynb`, then `model_train.ipynb`.
   Training logs to Weights & Biases if you are logged in (`wandb login`);
   set `WANDB_MODE=disabled` to skip it.

## In-browser demo (`web/`)

The full BioBERT + Swin model is too large for a web page, so the demo uses a
compact version of the same design, trained with `training/train_vqa.py`:

* **Image:** MobileNetV3-Large (ImageNet-pretrained), fine-tuned on VQA-Med 2019 → 960-d features
* **Question:** bag of words over the training vocabulary → 128-d, plus a logistic
  question-category detector (100% accurate on validation and test)
* **Fusion:** MLP over 1,701 answers, restricted at inference to the detected category
* Exported to ONNX (17.6 MB) and run with ONNX Runtime Web; images never leave the device

Trained on the training and validation sets (14,792 QA pairs, 8 epochs on CPU),
evaluated on the official **500-question test set** (exact match):

| Category | MediQNet demo | Best 2019 team (Hanlin) |
| --- | --- | --- |
| Modality | 75.2% | 80.8% |
| Plane | **78.4%** | 76.8% |
| Organ system | 72.0% | 73.6% |
| Abnormality | 15.2% | 18.4% |
| **Overall** | **60.2%** | 62.4% |

60.2% would rank **#5 of 18** on the 2019 leaderboard (Ben Abacha et al. 2019,
Table 3; that table's per-category values are shares of overall accuracy and are
multiplied by 4 here). The ONNX model matches PyTorch on all 500 test questions;
in the browser it matched on 23 of 24 sample images (one borderline plane answer
differs due to canvas resizing).

```bash
# Retrain (downloads VQA-Med 2019 from Zenodo first; CC BY 4.0)
pip install torch torchvision onnx scikit-learn pillow numpy
VQA_DATA=data VQA_OUT=out EPOCHS=8 python training/train_vqa.py --full
cp out/mediqnet.onnx out/vocab.json web/
```

`.github/workflows/pages.yml` publishes `web/` to GitHub Pages on every push to `main`.
**Research demo only. Not for clinical use.**

## Dataset credit

Asma Ben Abacha, Sadid A. Hasan, Vivek V. Datla, Joey Liu, Dina Demner-Fushman,
Henning Müller. *VQA-Med: Overview of the Medical Visual Question Answering
Task at ImageCLEF 2019.* CLEF 2019 Working Notes, CEUR-WS Vol. 2380.
<https://ceur-ws.org/Vol-2380/paper_272.pdf>
