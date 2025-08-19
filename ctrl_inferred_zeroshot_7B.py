#!/usr/bin/env python3
import os
import io
import contextlib
import pandas as pd
from datetime import datetime
from tqdm import tqdm
from llama_cpp import Llama

def main():
    try:
        print("Starting main generation pipeline...")

        # ================================
        # Load Persona Descriptions
        # ================================
        persona_df = pd.read_csv("ctrl_Persona.csv")  # Replace with actual path if needed
        personas = persona_df["Generated_persona"].dropna().tolist()
        total_posts = len(personas)

        # ================================
        # Model Initialization
        # ================================
        model_path = os.path.expanduser(
            './models/Mistral-7B-Instruct-v0.2.Q4_K_M.gguf'
        )

        llm = Llama(
            model_path=model_path,
            n_ctx=2048,
            n_threads=48,
            n_gpu_layers=35,
            verbose=True
        )

        # ================================
        # Generation Settings
        # ================================
        data_prompt_template = """
Now independently imagine yourself as a mentally healthy reddit user with no history of mental illness and write exactly one reddit comment.
Your comment must be approximately 80 words long. Do not use any hashtags in your comment. Do not use greetings like "hello" or exclamations like "wow"
at the start. Exclude preambles. You must not reference any mental health condition or terms related to mental illness.
Be creative with your response. Focus on general behavior, expression, and tone. Do not use a title.
"""

        diversity_prompt = """
Provide something more diverse than the previous posts.
Change the structure at the beginning of your response: it shouldn't follow the format of your previous posts.
"""

        gen_config = {
            "max_tokens": 200,
            "temperature": 0.8,
            "top_p": 0.95,
            "top_k": 40,
            "repeat_penalty": 1.1,
            "presence_penalty": 0.0,
            "frequency_penalty": 0.0,
            "stop": ["</s>"]
        }

        # ================================
        # Generate Prompts
        # ================================
        prompts = []

        for i, generated_persona in enumerate(personas, start=1):
            if i % 5 == 0:
                body = f"{data_prompt_template}\n\n{diversity_prompt}"
                ptype = "diversity"
            else:
                body = data_prompt_template
                ptype = "normal"

            persona_line = f"You are: {generated_persona}."
            full_prompt = f"[INST] {persona_line}\n\n{body} [/INST]"

            prompts.append({
                "index": i,
                "type": ptype,
                "prompt": full_prompt,
                "persona_line": persona_line,
            })

        # ================================
        # Run Model Inference
        # ================================
        results = []
        start = datetime.now()

        for idx, entry in enumerate(tqdm(prompts, desc="Generating comments"), start=1):
            buf = io.StringIO()
            try:
                with contextlib.redirect_stdout(buf):
                    out = llm(
                        entry["prompt"],
                        max_tokens=gen_config["max_tokens"],
                        temperature=gen_config["temperature"],
                        top_p=gen_config["top_p"],
                        top_k=gen_config["top_k"],
                        repeat_penalty=gen_config["repeat_penalty"],
                        presence_penalty=gen_config["presence_penalty"],
                        frequency_penalty=gen_config["frequency_penalty"],
                        stop=gen_config["stop"]
                    )
                text = out["choices"][0]["text"].strip()
            except Exception as e:
                text = f"[Error: {e}]"

            results.append({
                "TID": idx,
                "Persona": entry["persona_line"],
                "Prompt": entry["prompt"],
                "Generated Text": text,
                "max_tokens": gen_config["max_tokens"],
                "temperature": gen_config["temperature"],
                "top_p": gen_config["top_p"],
                "top_k": gen_config["top_k"],
                "repeat_penalty": gen_config["repeat_penalty"],
                "presence_penalty": gen_config["presence_penalty"],
                "frequency_penalty": gen_config["frequency_penalty"],
                "stop": str(gen_config["stop"]),
                "model": os.path.basename(model_path)
            })

        end = datetime.now()
        print(f"\nDone in {end - start}")

        # ================================
        # Save Output to CSV
        # ================================
        df = pd.DataFrame(results)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        outfn = f"ctrl_inf_zero_7B.csv"
        df.to_csv(outfn, index=False)
        print(f"Saved {len(df)} rows to {outfn}")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
