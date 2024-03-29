import six
import requests
import csv
from tqdm import tqdm
import os
import tarfile
import logging
import re


def reporthook(t):
    """https://github.com/tqdm/tqdm"""
    last_b = [0]

    def inner(b=1, bsize=1, tsize=None):
        """
        b: int, optional
        Number of blocks just transferred [default: 1].
        bsize: int, optional
        Size of each block (in tqdm units) [default: 1].
        tsize: int, optional
        Total size (in tqdm units). If [default: None] remains unchanged.
        """
        if tsize is not None:
            t.total = tsize
        t.update((b - last_b[0]) * bsize)
        last_b[0] = b
    return inner


def download_from_url(url, path=None, root='.data', overwrite=False):
    """Download file, with logic (from tensor2tensor) for Google Drive.
    Returns the path to the downloaded file.

    Arguments:
        url: the url of the file
        path: explicitly set the filename, otherwise attempts to
            detect the file name from URL header. (None)
        root: download folder used to store the file in (.data)
        overwrite: overwrite existing files (False)

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> torchtext.utils.download_from_url(url)
        >>> '.data/validation.tar.gz'
    """

    def _process_response(r, root, filename):
        chunk_size = 16 * 1024
        total_size = int(r.headers.get('Content-length', 0))
        if filename is None:
            d = r.headers['content-disposition']
            filename = re.findall("filename=\"(.+)\"", d)
            if filename is None:
                raise RuntimeError("Filename could not be autodetected")
            filename = filename[0]
        path = os.path.join(root, filename)
        if os.path.exists(path):
            logging.info('File %s already exists.' % path)
            if not overwrite:
                return path
            logging.info('Overwriting file %s.' % path)
        logging.info('Downloading file {} to {}.'.format(filename, path))
        with open(path, "wb") as file:
            with tqdm(total=total_size, unit='B',
                      unit_scale=1, desc=path.split('/')[-1]) as t:
                for chunk in r.iter_content(chunk_size):
                    if chunk:
                        file.write(chunk)
                        t.update(len(chunk))
        logging.info('File {} downloaded.'.format(path))
        return path

    filename = None
    if path is not None:
        root, filename = os.path.split(path)

    if not os.path.exists(root):
        raise RuntimeError(
            "Download directory {} does not exist. "
            "Did you create it?".format(root))

    if 'drive.google.com' not in url:
        response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, stream=True)
        return _process_response(response, root, filename)

    logging.info('Downloading from Google Drive; may take a few minutes')
    confirm_token = None
    session = requests.Session()
    response = session.get(url, stream=True)
    for k, v in response.cookies.items():
        if k.startswith("download_warning"):
            confirm_token = v

    if confirm_token:
        url = url + "&confirm=" + confirm_token
        response = session.get(url, stream=True)

    return _process_response(response, root, filename)


def unicode_csv_reader(unicode_csv_data, **kwargs):
    """Since the standard csv library does not handle unicode in Python 2, we need a wrapper.
    Borrowed and slightly modified from the Python docs:
    https://docs.python.org/2/library/csv.html#csv-examples"""
    if six.PY2:
        # csv.py doesn't do Unicode; encode temporarily as UTF-8:
        csv_reader = csv.reader(utf_8_encoder(unicode_csv_data), **kwargs)
        for row in csv_reader:
            # decode UTF-8 back to Unicode, cell by cell:
            yield [cell.decode('utf-8') for cell in row]
    else:
        for line in csv.reader(unicode_csv_data, **kwargs):
            yield line


def utf_8_encoder(unicode_csv_data):
    for line in unicode_csv_data:
        yield line.encode('utf-8')


def extract_archive(from_path, to_path=None, overwrite=False, archive='tar'):
    """Extract archive.

    Arguments:
        from_path: the path of the archive.
        to_path: the root path of the extraced files (directory of from_path)
        overwrite: overwrite existing files (False)
        archive: the archive format to extract (tar)

    Returns:
        List of paths to extracted files even if not overwritten.

    Examples:
        >>> url = 'http://www.quest.dcs.shef.ac.uk/wmt16_files_mmt/validation.tar.gz'
        >>> from_path = './validation.tar.gz'
        >>> to_path = './'
        >>> torchtext.utils.download_from_url(url, from_path)
        >>> torchtext.utils.extract_archive(from_path, to_path)
    """
    if to_path is None:
        to_path = os.path.dirname(from_path)

    if archive != 'tar':
        raise NotImplementedError("We currently only support tar achives.")

    logging.info('Opening tar file {}.'.format(from_path))
    with tarfile.open(from_path, 'r') as tar:
        files = []
        for file_ in tar:
            file_path = os.path.join(to_path, file_.name)
            if file_.isfile():
                files.append(file_path)
                if os.path.exists(file_path):
                    logging.info('{} already extracted.'.format(file_path))
                    if not overwrite:
                        continue
            tar.extract(file_, to_path)
        return files
