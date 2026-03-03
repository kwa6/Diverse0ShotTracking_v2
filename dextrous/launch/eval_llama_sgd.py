
from dextrous.launch import launch


machine = 'h100'

opt_args = dict(
    h100=dict(
        gen_batch_size=6,
        gradient_accumulation_steps=128,
    ),
    tebuna=dict(
        gen_batch_size=3,
        gradient_accumulation_steps=256,
    ),
)


for groupcode, base in [
    # ('llama_pretrained_dsg5k_finetune_sgd', 'ex/LlamaTracker/IconicChirrut/11'),
    ('llama_pretrained_dsg5k_finetune_sgd', 'ex/LlamaTracker/IconicChirrut/5')
]:
    launch.LlamaLaunch(
        groupcode=groupcode,
        base=base,
        approach='LlamaTracker',
        param_magnitude='13b',
        train_data='data/sgd/train',
        eval_data='data/sgd/unseen',
        train_downsample=None,
        eval_dialogue_downsample=None,
        eval_exclude_speakers='bot',
        test_domains=None,
        eval_all_slots_per_domain=True,
        prediction_lowercase=True,
        prediction_fuzzy_match_candidates=True,
        epochs=0,
        max_sequence_length=1024,
        train_batch_size=256,
        gradient_accumulation_steps=opt_args[machine]['gradient_accumulation_steps'],
        warmup_steps=100,
        learning_rate=5e-5,
        weight_decay=0.0,
        quantize='nf4',
        lora_merge_on_load=False,
        lora=32,
        lora_alpha=64,
        lora_dropout=0.0,
        train_all_slots_per_domain=True,
        neg_examples_ratio=0.0,
        exclude_speakers=['bot'],
        train_prop_add_continuation=1.0,
        train_percent_with_description=1.0,
        train_percent_description_only=0.0,
        train_percent_with_categories=0.0,
        train_percent_with_value_exs=0.0,
        train_percent_value_ex_includes_actual_value=None,
        train_remove_request_token_percent=None,
        train_filters_out_descriptions_with_actual_value=None,
        eval_with_categories=False,
        eval_with_value_exs=False,
        uncased=False,
        num_beams=3,
        repetition_penalty=1.0,
        gen_batch_size=opt_args[machine]['gen_batch_size'],
        max_output_length=16,
        rng_seed=21,
        do_eval_after_all_training=True,
        calculate_eval_perplexity=False,
        yield_every_x_epochs=0.10,
        dynamic_tokenization=True,
        notifications=True,
    )
