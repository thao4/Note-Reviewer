from embedder import Embedder
from loader import Loader
from model import LLMModel
import faiss
import numpy as np
import gradio as gr

def app():
    state = None
    
    def upload_and_process(pdf_file):
        nonlocal state
        state, message = process_pdf(pdf_file)
        return message

    def answer_question(question):
        nonlocal state
        if state is None:
            return "Please upload and process a PDF file first."
        response, state = query(question, state)
        return response

    with gr.Blocks() as demo:
        gr.Markdown("# PDF Question Answering App")
        
        with gr.Row():
            pdf_input = gr.File(label="Upload PDF", file_types=['.pdf'])
            upload_button = gr.Button("Process PDF")
        
        output_message = gr.Textbox(label="Status", interactive=False)
        
        upload_button.click(upload_and_process, inputs=pdf_input, outputs=output_message)
        
        question_input = gr.Textbox(label="Enter your question here")
        answer_button = gr.Button("Get Answer")
        answer_output = gr.Textbox(label="Answer", interactive=False, lines=10, max_lines=30)
        
        answer_button.click(answer_question, inputs=question_input, outputs=answer_output)
    
    demo.launch()
    
def process_pdf(pdf_file):
    if pdf_file is None:
        return None, "Please upload a PDF file."
    
    # load + embed + index
    embedder = Embedder()
    loader = Loader(pdf_file)
    model = LLMModel()
    
    text_chunks = loader.get_chunks()
    embedded_chunks = embedder.embed(text_chunks)
    dimension = embedded_chunks.shape[1] # MiniLM has a dimension of 384
    
    index = faiss.IndexFlatL2(dimension) 
    index.add(embedded_chunks)
    
    state_data = {
        "text_chunks": text_chunks,
        "embedded_chunks": embedded_chunks,
        "model": model,
        "index": index,
        "embedder": embedder
    }
    return state_data, "PDF loaded and processed successfully."

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