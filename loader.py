# # # # # # # # # # # # # # # # # # # # #
#                                       #
#           PDF Loader/Chunker          #
#   Loads in a PDF and chunks it to be  #
#   used in a RAG Pipeline              #
#                                       #
# # # # # # # # # # # # # # # # # # # # #


from PyPDF2 import PdfReader
import re

class Loader: 
    def __init__(self,file,chunk_size=500,overlap_ratio=0.15):
        self.file = PdfReader(file)
        self.chunk_size = chunk_size
        self.overlap_ratio = overlap_ratio
        self.full_text = ""
        for page in self.file.pages:
            text = page.extract_text()
            
            # Verify if the page is not empty
            if text:
                self.full_text += text
                
        self.sentences = self.split_into_sentences()
        self.chunks = self.chunk_sentences()

    def get_full_text(self):
        return self.full_text
    
    def split_into_sentences(self):
        # Splits the sentences in the text using lightweight regular expressions
        sentence_endings = r'(?<=[.!?])\s+' 
        sentences = re.split(sentence_endings,self.full_text)
        return [s.strip() for s in sentences if s.strip()] # removes whitespaces and items that are only whitespaces
    
    def chunk_sentences(self):
        # Overlap allows additional context to be added to the chunks
        chunks = []
        current_chunk = []
        current_length = 0
        
        # number of sentences to keep as overlap
        overlap_sentence_count = 2
        
        for sentence in self.sentences:
            sentence_length = len(sentence)
            
            # If adding the sentence keeps us under the chunk size, then add it
            if current_length + sentence_length <= self.chunk_size:
                current_chunk.append(sentence)
                current_length += sentence_length
            else:
                # Save current chunk
                chunks.append(" ".join(current_chunk).strip())
                
                # Create overlap from the end of the previous chunk
                if overlap_sentence_count > 0:
                    current_chunk = current_chunk[-overlap_sentence_count:] # grabs the last couple of sentences from the previous chunk
                    current_length = sum(len(s) for s in current_chunk)
                else:
                    current_chunk = []
                    current_length = 0
                
                # Add the new sentence (might already exceed size if long)
                current_chunk.append(sentence)
                current_length += sentence_length
            
        # Add last chunk
        if current_chunk:
            chunks.append(" ".join(current_chunk).strip())
        
        return chunks
    
    def get_chunks(self):
        return self.chunks