import dextrous.launch.launch as launch

models = [
    "ex/LlamaTracker/EnormousTeth/1",
]

for model in models:
    for num_beams in [1, 3, 5, 10]:
        for repetition_penalty in [1.1, 1.2, 1.3]:
            launch.LlamaLaunch(
                model_path=model,
                approach='LlamaTracker',
                train_data=None,
                epochs=0,
                eval_data='data/mwoz2.4/valid',
                eval_dialogue_downsample=300,
                eval_exclude_speakers='bot',
                eval_all_slots_per_domain=True,
                calculate_eval_perplexity=False,
                train_percent_with_description=1.0,
                train_percent_description_only=0.0,
                train_percent_with_categories=0.0,
                train_percent_with_value_exs=0.0,
                train_percent_value_ex_includes_actual_value=None,
                train_remove_request_token_percent=None,
                prediction_lowercase=True,
                prediction_fuzzy_match_candidates=True,
                exclude_speakers=['bot'],
                num_beams=num_beams,
                repetition_penalty=repetition_penalty,
                gen_batch_size=25 // num_beams,
                lora_merge_on_load=False,
                yield_every_x_epochs=0.03,
                notifications=True,
            )