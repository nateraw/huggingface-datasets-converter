import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from re import TEMPLATE
from tempfile import TemporaryDirectory

import requests
from bs4 import BeautifulSoup as bs
from huggingface_hub import create_repo, upload_folder, DatasetCardData, DatasetCard

from .utils import download_and_extract_archive, download_url

TEMPLATE_DATASHEET_PATH = Path(__file__).parent / "datasheet_template.md"

# Mapping from kaggle license identifiers to Hugging Face license identifiers
# Note: all Kaggle dataset licenses allow re-sharing of datasets, which is required to use this tool.
# When license is not specified or is 'other', then re-sharing is not allowed and thus this tool will fail.
kaggle_license_map = {
    'CC0-1.0': 'cc0-1.0',
    'CC-BY-SA-3.0': 'cc-by-sa-3.0',
    'CC-BY-SA-4.0': 'cc-by-sa-4.0',
    'CC-BY-NC-SA-4.0': 'cc-by-nc-sa-4.0',
    'GPL-2.0': 'gpl-2.0',
    'GNU Lesser General Public License 3.0': 'lgpl-3.0',
    'GNU Affero General Public License 3.0': 'agpl-3.0',
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
            f"The license of the {kaggle_id} dataset is unknown or is not supported in the Hugging Face Hub."
            " No one can use, share, distribute, re-post, add to,"
            " transform or change the dataset if it has not a specified"
            " a license. You can ask the dataset author to specify a"
            " license in the 'Discussion' section of the dataset's"
            " Kaggle page."
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
    card = DatasetCard.from_template(
        card_data=DatasetCardData(
            license=['unknown'],
            # These are huggingface-datasets-converter specific kwargs so we can filter for them on the Hub
            zenodo_id=zenodo_id,
            converted_from='zenodo',

        ),
        template_path=TEMPLATE_DATASHEET_PATH,
        **meta,
    )
    card.push_to_hub(repo_id)

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
    card = DatasetCard.from_template(
        card_data=DatasetCardData(
            license=[meta.get('license')],
            # These are huggingface-datasets-converter specific kwargs so we can filter for them on the Hub
            converted_from="kaggle",
            kaggle_id=kaggle_id,
        ),
        template_path=TEMPLATE_DATASHEET_PATH,
        **meta,
    )
    card.push_to_hub(repo_id)
    print(f"Uploaded your files. Check it out here: {url}")


NOTEBOOK_CONVERTER_HTML = """<center> <img
src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg
alt='Hugging Face'> <br> Copy a dataset ID from Kaggle's 
<a href="https://www.kaggle.com/datasets?fileType=csv" target="_blank">csv</a> or 
<a href="https://www.kaggle.com/datasets?fileType=json" target="_blank">json</a> datasets and paste it below
<br> Then, provide the ID of the Hugging Face repo to create when converting
<br> Both IDs should look something like this: <b>username/dataset-or-repo-name</b> </center>"""

def notebook_converter_kaggle():
    """
    Displays a widget to login to the HF website and store the token.
    """
    try:
        import ipywidgets.widgets as widgets
        from IPython.display import clear_output, display
    except ImportError:
        raise ImportError(
            "The `notebook_login` function can only be used in a notebook (Jupyter or"
            " Colab) and you need the `ipywidgets` module: `pip install ipywidgets`."
        )

    box_layout = widgets.Layout(
        display="flex", flex_flow="column", align_items="center", width="50%"
    )

    kaggle_id_widget = widgets.Password(description="Kaggle ID:")
    hf_repo_id_widget = widgets.Password(description="Repo ID:")
    finish_button = widgets.Button(description="Login")

    login_token_widget = widgets.VBox(
        [
            widgets.HTML(NOTEBOOK_CONVERTER_HTML),
            kaggle_id_widget,
            hf_repo_id_widget,
            finish_button,
        ],
        layout=box_layout,
    )
    display(login_token_widget)

    # On click events
    def login_token_event(t):
        kaggle_id = kaggle_id_widget.value
        repo_id = hf_repo_id_widget.value
        clear_output()
        kaggle_to_hf(kaggle_id, repo_id)
        print(f"Kaggle ID: {kaggle_id}")
        print(f"Repo ID: {repo_id}")

    finish_button.on_click(login_token_event)
