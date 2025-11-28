from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

class LLMModel:
    def __init__(self,model_name="google/flan-t5-large"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        
    def generate_response(self, prompt):
        inputs = self.tokenizer(prompt, return_tensors="pt") # pt = PyTorch Tensors
        outputs = self.model.generate(**inputs, 
                                      max_new_tokens=512, # max output length
                                      num_beams=5, # beam search keeps top N candidate sequences at each step, higher num_beam -> better quality but slower generation
                                      no_repeat_ngram_size=2, # prevents the model from repeating
                                      early_stopping=True, # stops beam search when all beams finish
                                      temperature=0.7, # controls randomness, lower temp -> deterministic, higher temp -> random and more creativity
                                      do_sample=True, # enables stochastic sampling instead of picking the most likely token
                                      top_p=0.9, # sample from smallest set of tokens
                                      top_k=50 # sample from top K most likely tokens only
                                      )
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response