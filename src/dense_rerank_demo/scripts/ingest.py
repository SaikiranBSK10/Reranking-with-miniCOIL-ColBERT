
from ..logging import get_logger
from ..config import DATASET, DATA_DIR
from ..data.loader import load_beir
from ..qdrant.client import get_client
from ..qdrant.index import index_corpus_dense
logger=get_logger(__name__)
def main():
    corpus,_,_=load_beir(DATASET, DATA_DIR, split="test")
    client=get_client()
    index_corpus_dense(client, corpus)
if __name__=="__main__":
    main()
