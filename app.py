from embedder import Embedder
from loader import Loader
from model import LLMModel
import faiss
import numpy as np
import gradio as gr

def app():
    pdf = "PDFs/BrainPlot.pdf"
    embedder = Embedder()
    loader = Loader(pdf)
    model = LLMModel()
    
    text_chunks = loader.get_chunks()
    embedded_chunks = embedder.embed(text_chunks)
    dimension = embedded_chunks.shape[1] # MiniLM has a dimension of 384
    
    index = faiss.IndexFlatL2(dimension) 
    index.add(embedded_chunks)

    state_data = {
        "text_chunks": text_chunks,
        "model": model,
        "index": index,
        "embedder": embedder
    }

    demo = gr.Interface(
        fn=query,
        inputs=[
            gr.Textbox(lines=2, placeholder="Enter your question here..."),
            gr.State(state_data),
        ],
        outputs=[
            gr.Textbox(label="Response", lines = 5, max_lines=30),
            gr.State()
        ],
        title="Document Question Answering with LLMs",
        description="Ask questions about the content of a PDF document."
    )
    
    demo.launch()
    
def retrieve_chunks(question, index, chunks, embedder, k=3):
    query_vec = embedder.embed(question) # returns a (dimension,) shape
    query_vec = np.array(query_vec, dtype='float32').reshape(1, -1) # converts (dimension,) to (1, dimension) - this shape is required for FAISS
    
    distances, indices = index.search(query_vec, k)
    
    results = [chunks[i] for i in indices[0]]
    
    return results

def construct_prompt(question, retrieved_chunks):
    context_str = "\n".join(retrieved_chunks)
    prompt = f"""Read the following context and explain the answer to the question in your own words,
    combining information across all sentences. If the context doesn’t contain the answer,
    say "I don’t know".
    Context:
    {context_str}

    Question: {question}
    """
    return prompt

def query(question, state):
    text_chunks = state["text_chunks"]
    model = state["model"]
    index = state["index"]
    embedder = state["embedder"]
    
    retrieved_chunks = retrieve_chunks(question, index, text_chunks, embedder)
    
    prompt = construct_prompt(question, retrieved_chunks)
    answer = model.generate_response(prompt) # this line is modular enough to swap out different LLMs
    
    formatted = f"""Question: {question}
Answer: {answer}
Chunks used:
{'\n---------------\n'.join(retrieved_chunks)}
    """
    return formatted, state

if __name__ == "__main__":
    app()