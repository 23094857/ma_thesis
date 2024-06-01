#! /usr/bin/env python3

from GA import GA
from utils.llm import LLM
from utils.data_loader import DataLoader


llm_evaluation = LLM("fireworks-llama-v3-8b-instruct", request_timeout=50000, temperature=0.7)
llm_assistent = LLM("fireworks-llama-v3-70b-instruct", request_timeout=50000, temperature=0.7)
dataset = DataLoader(dataset_name="math")

population_HPS = [
    'The governor of the university needs able citizens above all.',
    'Study science for others at Saracens Presidency.',
    'Review both.',
    'Read by Gyp is corrected to: Read it by Gyp.',
    'Come to the last spot to mount the situation.',
    'Robert saw a capillary with a head.',
    'Choose on Monday.',
    'The statement says on the output.'
 ]

population_best5_GA = [
    "Study science for your test in Saracens' machinery class.",
    "Review it in Italy.",
    "Read for the L'Turu Saracens Presidency.",
    "Review the time in others' work on Mondays.",
    "Do it on this day."
]

GA_object = GA(llm_evaluation, llm_assistent, dataset,
                population = population_best5_GA,
                max_generations=0,
                evaluations_per_generation=5000,
                max_evaluations_per_prompt=5000,
                #    population_size=8,
                baseline_accuracy=0.415,
                # n_final_evals_per_prompt=5000,
                max_llm_calls=int(150000),
                mutation_probability=0.49,
                cross_over_ratio=0.77,
                # reset_evals_every_n_llm_calls=40000
            )

GA_object.run_GA()




