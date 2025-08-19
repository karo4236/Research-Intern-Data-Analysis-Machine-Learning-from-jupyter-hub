#!/usr/bin/env python3

import multiprocessing
import random
import uuid
import io
import contextlib
import pandas as pd
from datetime import datetime
from tqdm import tqdm

from wk7_task2_persona_selector import generate_attribute_controlled_persona, load_inferred_personas
from wk7_task2_shot_selector import build_zero_shot_prompt, build_few_shot_prompt
from wk7_task2_model_selector import init_model_7b, init_model_70b, run_llm_inference

# Load global data once
subreddit_df = pd.read_csv("subreddits.csv")
subreddits = subreddit_df["Subreddits"].dropna().unique().tolist()

few_shot_df = pd.read_csv("balanced_depression_part1.csv")
few_shot_examples_pool = few_shot_df[["TID", "text", "user_id", "post_id"]].dropna().to_dict(orient="records")

inferred_personas_all = load_inferred_personas("mhc_demographics_cleaned (1).csv", max_n=100)


def run_condition(persona_type, prompting_method, model_size, total_posts_per_condition=15):
    print(f"\n[START] Persona: {persona_type} | Prompting: {prompting_method} | Model: {model_size}")

    # Initialize model
    llm = init_model_7b() if model_size == "7B" else init_model_70b()

    # Generate personas
    if persona_type == "attribute_controlled":
        personas = [generate_attribute_controlled_persona() for _ in range(total_posts_per_condition)]
    else:
        personas = (random.choices(inferred_personas_all, k=total_posts_per_condition)
                    if len(inferred_personas_all) < total_posts_per_condition
                    else inferred_personas_all[:total_posts_per_condition])

    # Pick subreddits
    sub_choices = [random.choice(subreddits) for _ in range(total_posts_per_condition)]

    # Build prompts
    prompts = []
    for i, (profile, sub_choice) in enumerate(zip(personas, sub_choices), start=1):
        if prompting_method == "zero_shot":
            prompt = build_zero_shot_prompt(profile, sub_choice)
        else:
            few_shot_examples = random.sample(few_shot_examples_pool, min(10, len(few_shot_examples_pool)))
            prompt = build_few_shot_prompt(few_shot_examples, sub_choice, profile)

        prompts.append({
            "index": i,
            "persona_type": persona_type,
            "prompting_method": prompting_method,
            "model_size": model_size,
            "profile": profile,
            "subreddit": sub_choice,
            "prompt": prompt
        })

    # Run inference
    results = []
    start_time = datetime.now()
    for entry in tqdm(prompts, desc=f"Running {persona_type} | {prompting_method} | {model_size}"):
        try:
            with contextlib.redirect_stdout(io.StringIO()):
                out = run_llm_inference(llm, entry["prompt"])
            generated_text = out["choices"][0]["text"].strip()
        except Exception as e:
            generated_text = f"[Error: {e}]"

        user_id = f"user_{uuid.uuid4().hex[:8]}"
        post_id = f"post_{uuid.uuid4().hex[:8]}"
        tid = f"{user_id}_{post_id}"

        results.append({
            "TID": tid,
            "user_id": user_id,
            "post_id": post_id,
            "label": "mdd",
            "text": generated_text,
            "language": "en"
        })

    end_time = datetime.now()
    print(f"[DONE] {persona_type} | {prompting_method} | {model_size} in {end_time - start_time}")

    # Save results
    df = pd.DataFrame(results)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"synthetic_posts_{persona_type}_{prompting_method}_{model_size}_{ts}.csv"
    df.to_csv(filename, index=False)
    print(f"Saved: {filename}")
    return df


def main():
    persona_types = ["attribute_controlled", "inferred"]
    prompting_methods = ["zero_shot", "few_shot"]
    model_sizes = ["7B", "70B"]

    tasks = []
    for persona_type in persona_types:
        for prompting_method in prompting_methods:
            for model_size in model_sizes:
                tasks.append((persona_type, prompting_method, model_size))

    print(f"Running {len(tasks)} experiments in parallel...\n")

    with multiprocessing.Pool(processes=min(8, multiprocessing.cpu_count())) as pool:
        all_results = pool.starmap(run_condition, tasks)

    # Combine and save all outputs
    combined_df = pd.concat(all_results, ignore_index=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    combined_df.to_csv(f"synthetic_posts_all_conditions_{ts}.csv", index=False)
    print(f"\nSaved ALL results to synthetic_posts_all_conditions_{ts}.csv")


if __name__ == "__main__":
    main()
