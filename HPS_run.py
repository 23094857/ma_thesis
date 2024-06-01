#! /usr/bin/env python3

from HPS import HPS

from utils.llm import LLM
from utils.data_loader import DataLoader


llm_evaluation = LLM("fireworks-llama-v3-8b-instruct", request_timeout=500, temperature=0.7)
llm_assistent = LLM("fireworks-llama-v3-70b-instruct", request_timeout=500, temperature=0.7)
dataset = DataLoader(dataset_name="math")


HPS_object = HPS(n_trials=15,
                    llm_evaluation=llm_evaluation,
                    llm_assistent=llm_assistent,
                    dataset=dataset,
                    population_size=8,
                    evaluations_per_generation=250,
                    max_evaluations_per_prompt=250,
                    max_llm_calls=int(150000),
                    n_final_evals_per_prompt=500,
                    baseline_accuracy=0.415,
                    pruning=False)

HPS_object.run_HPS()

