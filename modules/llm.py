"""
Flask API for generating text using Hugging Face LLaMA 3.1 in chat format.

Example usage:
curl -X POST -H "Content-Type: application/json" \
     -d '{"max_length": 256, "messages": [{"role": "system", "content": "You are a pirate bot."}, {"role": "user", "content": "What is your name?"}]}' \
     http://0.0.0.0:5001/generate

source ~/venvs/ollama-env/bin/activate


curl -X POST -H "Content-Type: application/json" \
-d '{
  "max_length": 2048,
  "messages": [
    {
      "role": "system",
      "content": "You are a highly capable reasoning assistant. Your task is to analyze complex legal, logical, or policy-based scenarios and provide clear, concise, and accurate answers. Break down reasoning into logically sound steps if needed. Always prioritize factual correctness over politeness, and avoid speculative assumptions."
    },
    {
      "role": "user",
      "content": "A new law passed in Germany in January 2023 states that \"Freelancers earning below €22,000 annually are exempt from VAT registration.\" Lisa is a freelance translator who earned €19,000 in 2022 and €25,000 in 2023. In April 2024, the German tax office sends Lisa a letter demanding an explanation for why she did not register for VAT in 2023. Lisa replies, citing the 2023 law and her 2022 income. Question: Based on the new law and Lisa’s timeline, was she legally obligated to register for VAT in 2023? Why or why not?"
    }
  ]
}' \
http://0.0.0.0:5001/generate


Author: Adnan
Edited by: Cursor/Philine on 2026-03-18
"""




import os
import sys

# Prefer the non-Xet download codepath for stability.
# You can still override this from the shell by setting it explicitly.
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

# Prevent local `modules/logging.py` from shadowing stdlib `logging`.
# When running `python modules/llm.py`, Python puts the script directory
# (`.../modules`) at the front of sys.path, so `import logging` inside
# Flask/Werkzeug resolves to our local file. Remove that entry before
# importing Flask or anything that imports `logging`.
_script_dir = os.path.dirname(os.path.abspath(__file__))
if sys.path and os.path.abspath(sys.path[0]) == _script_dir:
    sys.path.pop(0)

from flask import Flask, request
from transformers import pipeline, AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import snapshot_download
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import torch
torch.set_float32_matmul_precision('high')

def _cuda_works() -> bool:
    try:
        return bool(torch.cuda.is_available() and torch.cuda.device_count() > 0)
    except Exception:
        return False


app = Flask(__name__)
app.config['ENV'] = 'development'

try:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
except Exception:
    # CUDA can be present but unusable (e.g., driver too old). In that case,
    # continue in CPU mode instead of crashing at import time.
    pass



class TextGenerator:
    def __init__(self, model_id, cache_dir):
        """
        Load the model from Hugging Face and cache it locally.
        """
        self.model_id = model_id
        self.cache_dir = cache_dir

        # ✅ Download & cache model files
        self.model_path = snapshot_download(
            repo_id=model_id,
            cache_dir=cache_dir,
            local_files_only=False
        )

        # Load model. If CUDA init fails (e.g. driver too old), fall back to CPU.
        use_cuda = _cuda_works()
        device_map = "cuda:0" if use_cuda else "cpu"
        dtype = torch.float16 if use_cuda else torch.float32
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map=device_map,
            torch_dtype=dtype,
            use_safetensors=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        # Build the pipeline once; pass generation params per request.
        self.generator = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            return_full_text=True,
            clean_up_tokenization_spaces=True,
            truncation=True,
        )

    def generate_text(self, messages, max_length, top_k, top_p, temperature, do_sample):
        """
        Generate a response using a list of role-based messages.
        """

        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
        except Exception:
            # CUDA can be present but unusable (e.g., driver too old). In that case,
            # continue in CPU mode instead of crashing at import time.
            pass

        with torch.inference_mode():
            output = self.generator(
                messages,
                return_full_text=False,
                max_new_tokens=max_length,
                do_sample=do_sample,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                use_cache=True,
            )[0]["generated_text"]

        return {"generate": output}


# ✅ Model setup
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"
#model_id = "google/gemma-3-12b-it"
local_cache_dir = "./hf_models"
generator = TextGenerator(model_id, local_cache_dir)

@app.route('/generate', methods=['POST'])
def generate():
    try:
        data = request.json
        messages = data.get('messages', [])
        if not messages:
            return {"error": "Missing `messages` array."}

        max_length = int(data.get('max_length', 2048))
        top_k = int(data.get('top_k', 10))
        top_p = float(data.get('top_p', 1.0))
        temperature = float(data.get('temperature', 0.1))
        do_sample = data.get('do_sample', True)
        if isinstance(do_sample, str):
            do_sample = do_sample.strip().lower() in ("1", "true", "yes", "y", "on")
        else:
            do_sample = bool(do_sample)

        if do_sample and temperature <= 0:
            return {"error": "If do_sample == True, temperature must be > 0"}

        return generator.generate_text(
            messages, max_length, top_k, top_p, temperature, do_sample
        )

    except Exception as e:
        print("Error:", str(e))
        return {"error": str(e)}

if __name__ == '__main__':
    os.environ['FLASK_ENV'] = 'development'
    app.run(host='0.0.0.0', port=5001, debug=False)

     