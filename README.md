# MediQNet — Medical Visual Question Answering

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

> **Deployment:** trained weights are not stored in this repository (they are
> several hundred MB), so there is no hosted demo. To publish one, push the
> trained model to the Hugging Face Hub and wrap it in a Gradio Space.

## Dataset credit

Asma Ben Abacha, Sadid A. Hasan, Vivek V. Datla, Joey Liu, Dina Demner-Fushman,
Henning Müller. *VQA-Med: Overview of the Medical Visual Question Answering
Task at ImageCLEF 2019.* CLEF 2019 Working Notes, CEUR-WS Vol. 2380.
<https://ceur-ws.org/Vol-2380/paper_272.pdf>
