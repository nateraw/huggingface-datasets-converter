from multiprocessing import Pool
from functools import partial
import os
from tempfile import TemporaryDirectory
import requests
from bs4 import BeautifulSoup as bs
import json
from huggingface_hub import create_repo, upload_folder
import kaggle

from .utils import download_url, download_and_extract_archive

def _dl_wrap(root: str, url:str) -> None:
    download_url(url, root)

def download_urls(urls, root='./data', num_download_workers=1):
    if not os.path.exists:
        os.makedirs(root, exist_ok=True)
    if num_download_workers == 1:
        for url in urls:
            download_url(url, root)
    else:
        part = partial(_dl_wrap, root)
        poolproc = Pool(num_download_workers)
        poolproc.map(part, urls)

def zenodo_to_hf(zenodo_id, repo_id, num_download_workers=1):
    url = f'https://zenodo.org/record/{zenodo_id}'
    r = requests.get(url, headers={'Accept': 'application/json'})
    soup = bs(r.text, 'lxml')
    json_str = soup.find('script').text
    zenodo_record_data = json.loads(json_str)
    
    zenodo_files = zenodo_record_data.get('distribution')
    urls_to_download = [x['contentUrl'] for x in zenodo_files]

    with TemporaryDirectory() as temp_dir:
        download_urls(urls_to_download, temp_dir, num_download_workers)
        url = create_repo(repo_id, repo_type='dataset', exist_ok=True)
        upload_folder(
            folder_path=temp_dir,
            path_in_repo="",
            repo_id=repo_id,
            token=None,
            repo_type='dataset'
        )
    print(f"Uploaded your files. Check it out here: {url}")


def kaggle_to_hf(kaggle_id, repo_id, token=None, unzip=True, path_in_repo=None):
    path_in_repo = path_in_repo or ""
    with TemporaryDirectory() as temp_dir:
        kaggle.api.dataset_download_files(kaggle_id, temp_dir, unzip=unzip, quiet=False)
        url = create_repo(repo_id, repo_type='dataset', exist_ok=True)
        upload_folder(
            folder_path=temp_dir,
            path_in_repo="",
            repo_id=repo_id,
            token=None,
            repo_type='dataset'
        )
    print(f"Uploaded your files. Check it out here: {url}")
