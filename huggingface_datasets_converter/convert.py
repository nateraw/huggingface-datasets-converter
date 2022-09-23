import json
import os
from functools import partial
from multiprocessing import Pool
from pathlib import Path
from re import TEMPLATE
from tempfile import TemporaryDirectory

import requests
from bs4 import BeautifulSoup as bs
from huggingface_hub import create_repo, upload_folder, DatasetCardData, DatasetCard, upload_file, logging

from .utils import download_and_extract_archive, download_url

logging.set_verbosity_debug()

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
    'ODbL-1.0': 'odbl',
    'DbCL-1.0': 'odbl',  # Note - this isn't exactly right, but dbcl-1.0 inherits from it.
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
            converted_from='zenodo',
            zenodo_id=zenodo_id,
        ),
        template_path=TEMPLATE_DATASHEET_PATH,
        **meta,
    )
    card.push_to_hub(repo_id)
    return url

_gitattributes_text = """
*.7z filter=lfs diff=lfs merge=lfs -text
*.arrow filter=lfs diff=lfs merge=lfs -text
*.bin filter=lfs diff=lfs merge=lfs -text
*.bz2 filter=lfs diff=lfs merge=lfs -text
*.ftz filter=lfs diff=lfs merge=lfs -text
*.gz filter=lfs diff=lfs merge=lfs -text
*.h5 filter=lfs diff=lfs merge=lfs -text
*.joblib filter=lfs diff=lfs merge=lfs -text
*.lfs.* filter=lfs diff=lfs merge=lfs -text
*.lz4 filter=lfs diff=lfs merge=lfs -text
*.model filter=lfs diff=lfs merge=lfs -text
*.msgpack filter=lfs diff=lfs merge=lfs -text
*.npy filter=lfs diff=lfs merge=lfs -text
*.npz filter=lfs diff=lfs merge=lfs -text
*.onnx filter=lfs diff=lfs merge=lfs -text
*.ot filter=lfs diff=lfs merge=lfs -text
*.parquet filter=lfs diff=lfs merge=lfs -text
*.pb filter=lfs diff=lfs merge=lfs -text
*.pickle filter=lfs diff=lfs merge=lfs -text
*.pkl filter=lfs diff=lfs merge=lfs -text
*.pt filter=lfs diff=lfs merge=lfs -text
*.pth filter=lfs diff=lfs merge=lfs -text
*.rar filter=lfs diff=lfs merge=lfs -text
saved_model/**/* filter=lfs diff=lfs merge=lfs -text
*.tar.* filter=lfs diff=lfs merge=lfs -text
*.tflite filter=lfs diff=lfs merge=lfs -text
*.tgz filter=lfs diff=lfs merge=lfs -text
*.wasm filter=lfs diff=lfs merge=lfs -text
*.xz filter=lfs diff=lfs merge=lfs -text
*.zip filter=lfs diff=lfs merge=lfs -text
*.zst filter=lfs diff=lfs merge=lfs -text
*tfevents* filter=lfs diff=lfs merge=lfs -text
# Audio files - uncompressed
*.pcm filter=lfs diff=lfs merge=lfs -text
*.sam filter=lfs diff=lfs merge=lfs -text
*.raw filter=lfs diff=lfs merge=lfs -text
# Audio files - compressed
*.aac filter=lfs diff=lfs merge=lfs -text
*.flac filter=lfs diff=lfs merge=lfs -text
*.mp3 filter=lfs diff=lfs merge=lfs -text
*.ogg filter=lfs diff=lfs merge=lfs -text
*.wav filter=lfs diff=lfs merge=lfs -text
# Image files - uncompressed
*.bmp filter=lfs diff=lfs merge=lfs -text
*.gif filter=lfs diff=lfs merge=lfs -text
*.png filter=lfs diff=lfs merge=lfs -text
*.tiff filter=lfs diff=lfs merge=lfs -text
# Image files - compressed
*.jpg filter=lfs diff=lfs merge=lfs -text
*.jpeg filter=lfs diff=lfs merge=lfs -text
*.webp filter=lfs diff=lfs merge=lfs -text
*.json filter=lfs diff=lfs merge=lfs -text
*.csv filter=lfs diff=lfs merge=lfs -text
""".strip()


def kaggle_to_hf(kaggle_id, repo_id, token=None, unzip=True, path_in_repo=None):
    import kaggle
    path_in_repo = path_in_repo or ""
    meta = get_kaggle_metadata(kaggle_id)
    with TemporaryDirectory() as temp_dir:
        kaggle.api.dataset_download_files(kaggle_id, temp_dir, unzip=unzip, quiet=False)
        url = create_repo(repo_id, repo_type='dataset', exist_ok=True)

        # HACK - try to update gitattributes to avoid upload_folder failures...
        gitattributes_file = Path(temp_dir) / '.gitattributes'
        gitattributes_file.write_text(_gitattributes_text)
        upload_file(path_or_fileobj=gitattributes_file.as_posix(), path_in_repo=".gitattributes", repo_id=repo_id, token=token, repo_type='dataset')

        upload_folder(folder_path=temp_dir, path_in_repo="", repo_id=repo_id, token=token, repo_type='dataset')
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
    return url

NOTEBOOK_CONVERTER_HTML = """<center> <img
src=https://huggingface.co/front/assets/huggingface_logo-noborder.svg
alt='Hugging Face'> <br> Copy a dataset ID from Kaggle's 
<a href="https://www.kaggle.com/datasets?fileType=csv" target="_blank">csv</a> or 
<a href="https://www.kaggle.com/datasets?fileType=json" target="_blank">json</a> datasets and paste it below.
<br> Then, provide the Hugging Face repo ID of the dataset repo you'd like to upload to.
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

    kaggle_id_widget = widgets.Text(description="Kaggle ID:")
    hf_repo_id_widget = widgets.Text(description="Repo ID:")
    finish_button = widgets.Button(description="Convert")

    login_token_widget = widgets.VBox(
        [
            widgets.HTML(NOTEBOOK_CONVERTER_HTML),
            kaggle_id_widget,
            hf_repo_id_widget,
            finish_button,
        ],
        layout=box_layout,
    )

    # On click events
    output = widgets.Output()

    @output.capture()
    def login_token_event(t):
        kaggle_id = kaggle_id_widget.value
        repo_id = hf_repo_id_widget.value
        print("Converting...")
        print(f"\t- Kaggle ID: {kaggle_id}")
        print(f"\t- Repo ID: {repo_id}")
        url = kaggle_to_hf(kaggle_id, repo_id)
        output.clear_output()
        print(f"You can view your dataset here: {url}")

    with output:
        display(login_token_widget)
    finish_button.on_click(login_token_event)
    display(output)
