# COVID-19 Sentiment NLP

End-to-end pipeline for sentiment classification on COVID-19 tweets.  
The project covers preprocessing, baseline evaluations, and final fine-tuning with BERTweet and DeBERTa, including pruning and quantization.


##  Repository Structure
```

├─ data/ # Input CSVs and processed DataFrames
├─ notebooks/ # All Jupyter notebooks
│ ├─ part_1_eda_preprocessing_transformation_final_data.ipynb
│ ├─ baseline_benchmark_zero_shot_BART_raw_and_transformed.ipynb
│ ├─ baseline_benchmark_few_shot_MiniLM_raw_and_transformed.ipynb
│ ├─ baseline_benchmark_BERTweet_raw.ipynb
│ ├─ part_2_fine_tuning_torch_hf_compression_BERTweet_DEBERTA.ipynb
│ └─ inference_notebook.ipynb # batch metrics for saved .pt models
├─ fine_tuned_models/ # Final model weights (.pt) tracked via Git LFS
├─ setup_links.py # Creates links so notebooks run unchanged
├─ requirements.txt # Dependencies
└─ Covid19_sentiment_nlp.pdf # Report

````

---

##  Setup

Clone the repository and install dependencies:

```bash
git lfs install
git clone https://github.com/iluley/covid19-sentiment-nlp.git
cd covid19-sentiment-nlp
pip install -r requirements.txt
````

### 0. (One-time) Fix legacy notebook paths

The original notebooks expect data/models in the same directory. Run:

```bash
python setup_links.py
```

This creates links inside `notebooks/` so you can run all notebooks (including inference) unchanged.

---

##  How to Run

### 1. Preprocessing

Run:

```
notebooks/part_1_eda_preprocessing_transformation_final_data.ipynb
```

This reads the original CSVs (`Corona_NLP_train.csv`, `Corona_NLP_test.csv`) and produces:

* `data/df_train_final.csv`
* `data/df_test_final.csv`

### 2. Baselines

Run in any order (all consume the final DataFrames):

* `baseline_benchmark_zero_shot_BART_raw_and_transformed.ipynb`
* `baseline_benchmark_few_shot_MiniLM_raw_and_transformed.ipynb`
* `baseline_benchmark_BERTweet_raw.ipynb`

### 3. Fine-Tuning

Run:

```
notebooks/part_2_fine_tuning_torch_hf_compression_BERTweet_DEBERTA.ipynb
```

Trains BERTweet/DeBERTa (+ distilled/pruned/quantized variants) and writes weights to fine_tuned_models/


### 4. Inference

```
notebooks/inference_notebook.ipynb
```

* Discovers all *.pt files (state_dicts) under fine_tuned_models/ (or the same dir as the notebook if you copy weights there).
* Infers the correct backbone per filename (BERTweet / DeBERTa / student models).
* Loads the matching tokenizer (use_fast=False, same as training to avoid SentencePiece/TikToken pitfalls).
* Evaluates on data/df_test_final.csv (labels come from the notebook’s label map).
* Produces a final metrics table for each model: macro-F1, accuracy, precision, recall, AUC (AUC is reported when probability scores are available).
* Optionally saves a CSV of the results (see the last cell).

Where to place weights:
Put your .pt files in fine_tuned_models/ (recommended).
If you prefer “same folder as the notebook”, place them in notebooks/; setup_links.py makes both layouts work.


---

##  Model Weights

All `.pt` weights are tracked with **Git LFS**.
If you see pointer files instead of actual weights, run:

```bash
git lfs install
git lfs pull
```
If Git LFS is slow or blocked, you can download the same `.pt` files here:  
[Google Drive backup — model weights](https://drive.google.com/drive/folders/15osPdIYL2JYlAXhbO4FDNamhG_vguA62?usp=sharing)

Place the downloaded files under `fine_tuned_models/` (keep the exact filenames).  
The repo’s Git LFS remains the source of truth; the Drive folder is a convenience mirror.


---

##  Results (summary)

* **Baselines:**

  * Zero-shot BART (MNLI)
  * Few-shot MiniLM (cosine prototypes)
  * BERTweet baseline

* **Fine-tuned models:**

  * BERTweet and DeBERTa (with pruned/quantized/distilled versions)

See notebooks and the PDF report for detailed metrics.

---

##  Dependencies

* numpy, pandas, scikit-learn, matplotlib, seaborn, tqdm
* torch, transformers, datasets, accelerate, sentencepiece
* sentence-transformers (for MiniLM few-shot)
* pyarrow (needed by HF datasets)
* optuna
* wandb
