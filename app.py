from embedder import Embedder
from loader import Loader

def app():
    pdf = "PDFs/BrainPlot.pdf"
    embedder = Embedder()
    loader = Loader(pdf)
    
    embedded_chunks = embedder.embed(loader.get_chunks())
    
    print(embedded_chunks[0])
    print(embedded_chunks.shape)
    
if __name__ == "__main__":
    app()