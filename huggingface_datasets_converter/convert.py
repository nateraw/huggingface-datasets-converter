import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from re import TEMPLATE
from tempfile import TemporaryDirectory

import requests
from bs4 import BeautifulSoup as bs
from huggingface_hub import create_repo, upload_folder
from modelcards import CardData, ModelCard

from .utils import download_and_extract_archive, download_url

TEMPLATE_DATASHEET_PATH = Path(__file__).parent / "datasheet_template.md"

# Mapping from kaggle license identifiers to Hugging Face license identifiers
# Note: all licenses in datasets inside Kaggle allow re-sharing of datasets; what we are doing here.
kaggle_license_map = {
    'CC0-1.0': 'cc0-1.0',
    'CC-BY-SA-3.0': 'cc-by-sa-3.0',
    'CC-BY-SA-4.0': 'cc-by-sa-4.0',
    'CC-BY-NC-SA-4.0': 'cc-by-nc-sa-4.0',
    'GPL-2.0': 'gpl-2.0',
    'GPL-3.0': 'gpl-3.0',
    'ODC Public Domain Dedication and Licence (PDDL)': 'pddl',
    'ODC Attribution License (ODC-By)': 'odc-by',
    'ODbL-1.0': 'odbl-1.0',
    'DbCL-1.0': 'odbl-1.0',  # Note - this isn't exactly right, but dbcl-1.0 inherits from it.
    'other': 'other',
    'unknown': 'unknown',
}


def _dl_wrap(root: str, unzip_archives: bool, url: str) -> None:
    if unzip_archives and os.path.basename(url).endswith('.zip'):
        download_and_extract_archive(url, root, remove_finished=True)
    else:
        download_url(url, root)


def download_urls(urls, root='./data', num_download_workers=1, unzip_archives=True):
    if not os.path.exists:
        os.makedirs(root, exist_ok=True)
    if num_download_workers == 1:
        for url in urls:
            download_url(url, root)
    else:
        part = partial(_dl_wrap, root, unzip_archives)
        poolproc = Pool(num_download_workers)
        poolproc.map(part, urls)


def get_bibtex_citation_from_zenodo(zenodo_id):
    url = f'https://zenodo.org/record/{zenodo_id}/export/hx'
    r = requests.get(url)
    soup = bs(r.text, 'lxml')
    citation = soup.find('pre')
    return citation.text


def get_zenodo_metadata(zenodo_id):
    url = f'https://zenodo.org/record/{zenodo_id}'
    r = requests.get(url, headers={'Accept': 'application/json'})
    soup = bs(r.text, 'lxml')
    json_str = soup.find('script').text
    data = json.loads(json_str)
    meta = dict(
        dataset_name=data.get('name'),
        authors=", ".join([x.get('name') for x in data.get('creator')]) if 'creator' in data else "Unknown",
        description=data.get('description'),
        language=data.get('inLanguage', {}).get('name'),  # Ex. 'English'. will have to be converted to HF taxonomy ('en')
        license=data.get('license'),  # Returns a URL: http://creativecommons.org/licenses/by-nc/2.0/
        homepage=data.get('url'),
        citation=get_bibtex_citation_from_zenodo(zenodo_id),
        zenodo_id=zenodo_id,
        zenodo_files=[x.get('contentUrl') for x in data.get('distribution')],
    )
    # meta['language'] = languages_map.get(meta['language'])
    return meta


def kaggle_username_to_markdown(username):
    return f"[@{username}](https://kaggle.com/{username})"


def get_kaggle_metadata(kaggle_id):
    import kaggle
    user, dataset_name = kaggle_id.split('/')
    data = kaggle.api.metadata_get(user, dataset_name)
    info = data['info']
    try:
        license_kaggle = data['info']['licenses'][0].get('name')
        license = kaggle_license_map.get(license_kaggle, 'unknown')
    except:
        license = 'unknown'

    if license == 'unknown' or license == 'other':
        raise NameError(
            f"The license of the {kaggle_id} dataset is unknown."
            " No one can use, share, distribute, re-post, add to,"
            " transform or change the dataset if it has not a specified"
            " a license."
        )

    meta = dict(
        dataset_name=info.get('title'),
        homepage=f"https://kaggle.com/datasets/{user}/{dataset_name}",
        description=info.get('description'),
        authors=", ".join([kaggle_username_to_markdown(user)]),
        license=license,
        citation="[More Information Needed]",
        language=None,
        kaggle_id=kaggle_id,
    )
    return meta


def zenodo_to_hf(zenodo_id, repo_id, num_download_workers=1, unzip_archives=True):
    meta = get_zenodo_metadata(zenodo_id)
    urls_to_download = meta.pop('zenodo_files')
    with TemporaryDirectory() as temp_dir:
        download_urls(urls_to_download, temp_dir, num_download_workers, unzip_archives)
        url = create_repo(repo_id, repo_type='dataset', exist_ok=True)
        upload_folder(folder_path=temp_dir, path_in_repo="", repo_id=repo_id, token=None, repo_type='dataset')

    # Try to make dataset card as well!
    card = ModelCard.from_template(
        card_data=CardData(
            zenodo_id=zenodo_id,
            license=['unknown'],
        ),
        template_path=TEMPLATE_DATASHEET_PATH,
        **meta,
    )
    card.push_to_hub(repo_id, repo_type='dataset')

    print(f"Uploaded your files. Check it out here: {url}")


def kaggle_to_hf(kaggle_id, repo_id, token=None, unzip=True, path_in_repo=None):
    import kaggle
    path_in_repo = path_in_repo or ""
    meta = get_kaggle_metadata(kaggle_id)
    with TemporaryDirectory() as temp_dir:
        kaggle.api.dataset_download_files(kaggle_id, temp_dir, unzip=unzip, quiet=False)
        url = create_repo(repo_id, repo_type='dataset', exist_ok=True)
        upload_folder(folder_path=temp_dir, path_in_repo="", repo_id=repo_id, token=None, repo_type='dataset')
    # Try to make dataset card as well!
    card = ModelCard.from_template(
        card_data=CardData(
            kaggle_id=kaggle_id,
            license=[meta.get('license')],
        ),
        template_path=TEMPLATE_DATASHEET_PATH,
        **meta,
    )
    card.push_to_hub(repo_id, repo_type='dataset')
    print(f"Uploaded your files. Check it out here: {url}")
