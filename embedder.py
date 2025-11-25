# # # # # # # # # # # # # # # # # # # # #
#                                       #
#               Embedder                #
#   Loads a model from HuggingFace to   #
#   used for embedding                  #
#                                       #
# # # # # # # # # # # # # # # # # # # # #

from sentence_transformers import SentenceTransformer

class Embedder:
    def __init__(self,model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.model_name = model_name
        self.model = SentenceTransformer(self.model_name)
        
    def embed(self,text):
        return self.model.encode(text)
