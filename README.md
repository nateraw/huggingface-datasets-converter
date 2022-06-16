# Hugging Face Datasets Converter

Scripts to convert datasets from various sources to Hugging Face `datasets`.

## Usage

### Setup

```
git clone https://github.com/nateraw/huggingface-datasets-converter.git
cd huggingface-datasets-converter
pip install -r requirements.txt
```

Make sure to authenticate with Hugging Face Hub

```
huggingface-cli login
```

### Convert Kaggle Dataset

Make sure you have your `kaggle.json` file in `~/.kaggle`. Then...

Provide the kaggle dataset ID and the Hugging Face Hub repo ID that you'd like to upload to (it will be created if it doesn't exist).
```
python run_kaggle.py --kaggle_id evangower/airbnb-stock-price --repo_id nateraw/airbnb-stock-price
```


### Convert Zenodo Dataset

Provide the record ID and the name of the repo on Hugging Face Hub you'd like to upload to (it will be created if it doesn't exist).

```
python run_zenodo.py --zenodo_record 6606485 --repo_id nateraw/espeni
```

For zenodo, you can also pass `--workers` flag if you want to do this with multiprocessing.

```
python run_zenodo.py --zenodo_record 6606485 --repo_id nateraw/espeni --workers 2
```
