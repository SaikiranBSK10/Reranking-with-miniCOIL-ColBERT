
import os
from beir import util
from beir.datasets.data_loader import GenericDataLoader
from ..logging import get_logger
logger=get_logger(__name__)

def _find(dataset,out_dir,base=None):
    d=dataset.lower()
    cands=[]
    if base: cands+=[base, os.path.join(base,d), os.path.join(base,d,d)]
    cands+=[os.path.join(out_dir,d), os.path.join(out_dir,d,d)]
    for c in cands:
        if os.path.isfile(os.path.join(c,"corpus.jsonl")): return c
    return None

def load_beir(dataset,out_dir,split="test"):
    os.makedirs(out_dir,exist_ok=True)
    url=f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{dataset.lower()}.zip"
    logger.info("Loading BEIR dataset: %s", dataset)
    base=util.download_and_unzip(url,out_dir)
    local=_find(dataset,out_dir,base) or _find(dataset,out_dir)
    if not local: raise FileNotFoundError("corpus.jsonl not found after unzip")
    logger.info("Using BEIR dataset dir: %s", local)
    return GenericDataLoader(local).load(split=split)
