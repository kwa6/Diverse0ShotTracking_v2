
from dextrous.launch import launch


machine = 'h100'
opt_args = dict(
    h100=dict(
        gen_batch_size=6,
        gradient_accumulation_steps=128,
        max_sequence_length=1024,
    ),
    tebuna=dict(
        gen_batch_size=6,
        gradient_accumulation_steps=32,
    ),
)

opt_arg = opt_args[machine]
for domain in ['hotel', 'taxi', 'attraction', 'train', 'restaurant']:
    launch.T5Launch(
        **opt_arg,
        groupcode='t5-lora-all-dsg5k-finetune-mwoz',
        base='ex/T5Tracker/UnconquerableGrievous/21',
        approach='T5Tracker',
        param_magnitude='11b',
        train_data='data/mwoz2.4/train',
        eval_data='data/mwoz2.4/valid',
        train_downsample=None,
        eval_dialogue_downsample=None,
        eval_exclude_speakers='bot',
        test_domains=[domain],
        eval_all_slots_per_domain=True,
        prediction_lowercase=True,
        prediction_fuzzy_match_candidates=True,
        epochs=1,
        train_batch_size=256,
        warmup_steps=100,
        format='',
        optimizer='adafactor',
        learning_rate=1e-2,
        weight_decay=0.0,
        quantize='nf4',
        lora=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_modules=['q', 'v', 'k', 'o', 'wi_0', 'wi_1', 'wo'],
        lora_merge_on_load=True,
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
        max_output_length=16,
        rng_seed=21,
        do_eval_after_all_training=False,
        calculate_eval_perplexity=False,
        yield_every_x_epochs=0.10,
        dynamic_tokenization=True,
        notifications=True,
    )