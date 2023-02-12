import sys
import json
import os
import shutil

from src.dataset import make_dataset
from src.preprocessing import build_features
from src.models import train_model
from src.visualizations import visualize


def main(targets):
    if "test" in targets:
        if not os.path.exists("data/temp"):
            os.makedirs("data/temp")
        if not os.path.exists("data/out"):
            os.makedirs("data/out")

        test_path_metadata = "test/test_metadata.tsv"
        test_path_fungi = "test/test_fungi_data.tsv"



if __name__ == "__main__":
    # python run.py test
    targets = sys.argv[1:]
    main(targets)
    #generates graph in final_figure.png