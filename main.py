from fastapi import FastAPI, Request
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load model and tokenizer
model_name = "MiniMaxAI/MiniMax-M1-80k"
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True).cuda()

class PromptInput(BaseModel):
    prompt: str
    max_tokens: int = 100

@app.get("/")
def root():
    return {"message": "MiniMax-M1 API is running"}

@app.post("/generate")
def generate(prompt_input: PromptInput):
    input_ids = tokenizer(prompt_input.prompt, return_tensors="pt").input_ids.cuda()

    with torch.no_grad():
        output = model.generate(
            input_ids,
            max_new_tokens=prompt_input.max_tokens,
            do_sample=True,
            top_p=0.95,
            temperature=0.8
        )

    generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
    return {"output": generated_text}
