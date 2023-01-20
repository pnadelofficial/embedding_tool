import pandas as pd
import nltk
from umap import UMAP
from sklearn.pipeline import make_pipeline 
from embetter.text import SentenceEncoder

class Embedder:
    def __init__(self, model_name, sentences):
        self.model_name = model_name
        self.sentences = sentences
    
    def _make_embedding_pipeline(self):
        text_emb_pipeline = make_pipeline(
            SentenceEncoder(self.model_name),
            UMAP()
        )
        return text_emb_pipeline
    
    def _get_embeddings(self, pipeline):
        return pipeline.fit_transform(self.sentences)

    def _make_df(self, embeddings):
        df = pd.DataFrame({'text':self.sentences})
        df.text = df.text.str.wrap(30).apply(lambda x: x.replace('\n', '<br>'))
        df['sent_size'] = df.text.apply(lambda x: len(x.split()))
        df['x'] = embeddings[:, 0]
        df['y'] = embeddings[:, 1]
        return df
    
    def embed(self):
        pl = self._make_embedding_pipeline()
        em = self._get_embeddings(pl)
        df = self._make_df(em)
        return df
    
class Preprocesser:
    def __init__(self, text):
        self.text = text
    
    def sentence_tokenize(self):
        return nltk.sent_tokenize(self.text)