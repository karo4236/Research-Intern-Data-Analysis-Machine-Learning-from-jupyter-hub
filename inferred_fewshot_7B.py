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
        persona_df = pd.read_csv("mdd_Persona.csv")#################
        personas = persona_df["Generated_persona"].dropna().tolist()
        total_posts = len(personas)

        # ================================
        # Load Few-Shot Examples
        # ================================
        few_shot_df = pd.read_csv("sampled_1000_rows.csv")

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
            verbose=False
        )

        # ================================
        # Generation Settings
        # ================================
        data_prompt_template = """
Now independently imagine yourself as a normal reddit user and write exactly one reddit comment. You are diagnosed with the condition: MDD.
Your comment must be approximately 80 words long. Do not use any hashtags in your comment. Do not use greetings like "hello" or exclamations like "wow"
at the start. Exclude preambles. The mention of any explicit reference to MDD is not allowed. 
You must avoid using any language related to mental health such as a condition or general terms like "mental illness", 
"diagnosed with", or "suffering from". Be creative with your response. Focus on general behavior, expression, and tone. Do not use a title.
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
            few_shot_examples = few_shot_df['text'].dropna().sample(n=10).tolist()
            few_shot_block = "\n\n".join(f"- {ex}" for ex in few_shot_examples)
            if i % 5 == 0:
                body = f"{data_prompt_template}\n\n{diversity_prompt}"
                ptype = "diversity"
            else:
                body = data_prompt_template
                ptype = "normal"

            persona_line = f"You are: {generated_persona}."
            full_prompt = (
                f"[INST] {persona_line}\n\n"
                f"Here are some example comments:\n{few_shot_block}\n\n"
                f"Do not copy or repeat any example comments verbatim. Instead, use them as inspiration to mimic the tone, structure, or style in your own unique way.\n\n"
                f"{body} [/INST]"
            )

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
        outfn = f"mdd_inf_Few_7B.csv"
        df.to_csv(outfn, index=False)
        print(f"Saved {len(df)} rows to {outfn}")

    except Exception as e:
        print(f"Error in main: {e}")

if __name__ == "__main__":
    main()
