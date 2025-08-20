# COVID-19 Sentiment NLP

End-to-end pipeline for sentiment classification on COVID-19 tweets.  
The project covers preprocessing, baseline evaluations, and final fine-tuning with BERTweet and DeBERTa, including pruning and quantization.


##  Repository Structure
```

├─ data/                  # Input CSVs and processed DataFrames
├─ notebooks/             # All Jupyter notebooks
├─ fine\_tuned\_models/     # Final model weights (.pt) tracked via Git LFS
├─ setup\_links.py         # Creates links so notebooks run unchanged
├─ requirements.txt       # Dependencies
└─ Covid19\_sentiment\_nlp.pdf   # Report

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

This creates links inside `notebooks/` so you can run all notebooks unchanged.

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

This notebook trains/fine-tunes:

* BERTweet
* DeBERTa
* Distilled, pruned, and quantized variants

Weights are saved in `fine_tuned_models/`.

---

##  Model Weights

All `.pt` weights are tracked with **Git LFS**.
If you see pointer files instead of actual weights, run:

```bash
git lfs install
git lfs pull
```

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

Do you want me to also add a **“Quick sanity check” snippet** at the very end (e.g., 3 lines of Python to verify the CSVs and models are accessible)? That way your instructor can test the environment right away.
```
