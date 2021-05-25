# ------------------------------------------------------------------------------
# This code is (modified) from
# Assessing out-of-domain generalization for robust building damage detection
# https://github.com/ecker-lab/robust-bdd.git
# Licensed under the CC BY-NC-SA 4.0 License.
# Written by Vitus Benson (vbenson@bgc-jena.mpg.de)
# ------------------------------------------------------------------------------
import os
import json
import urllib.request
from tqdm import tqdm


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_weights() -> str:
    dl_path = "https://data.goettingen-research-online.de/api/access/datafile/20240?gbrecs=true"
    outfile = "twostream-resnet50_all_plain.pt"
    with open('exp_settings.json', 'r') as JSON:
        settings_dict = json.load(JSON)
    filepath = settings_dict['resnet']['path'] + '/' + outfile
    if not os.path.isfile(filepath):
        print("Downloading from {} to {}".format(dl_path, filepath))
        with DownloadProgressBar(unit='B', unit_scale=True, miniters=1, desc=dl_path.split('/')[-1]) as t:
            urllib.request.urlretrieve(dl_path, filename = filepath, reporthook=t.update_to)
        print("Downloaded!")
    else:
        print("File exists already!")
    return filepath