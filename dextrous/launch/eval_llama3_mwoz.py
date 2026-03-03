
from dextrous.launch import launch
from language_model.llama import llama3format


machine = 'h100'

opt_args = dict(
    h100=dict(
        gen_batch_size=6,
        gradient_accumulation_steps=64,
    ),
    tebuna=dict(
        gen_batch_size=3,
        gradient_accumulation_steps=256,
    ),
)

for groupcode, domain, base in [
    ('llama3-baseline-mwoz', 'hotel', 'ex/LlamaTracker/HeroicQuarren/9'),
    ('llama3-baseline-mwoz', 'restaurant', 'ex/LlamaTracker/RadiantOrson/10'),
    ('llama3-baseline-mwoz', 'attraction', 'ex/LlamaTracker/ResoluteNute/11'),
    ('llama3-baseline-mwoz', 'train', 'ex/LlamaTracker/SerendipitousOrdMantell/6'),
    ('llama3-baseline-mwoz', 'taxi', 'ex/LlamaTracker/ElusiveChirrut/4'),

    ('llama3-finetune-mwoz', 'hotel', 'ex/LlamaTracker/IntrepidAlderaan/3'),
    ('llama3-finetune-mwoz', 'restaurant', 'ex/LlamaTracker/ThunderousDantooine/2'),
    ('llama3-finetune-mwoz', 'attraction', 'ex/LlamaTracker/LimitlessDromundKaas/6'),
    ('llama3-finetune-mwoz', 'train', 'ex/LlamaTracker/DaringClakdorVII/3'),
    ('llama3-finetune-mwoz', 'taxi', 'ex/LlamaTracker/UntamedKorriban/8'),
]:
    launch.LlamaLaunch(
        groupcode=groupcode,
        base=base,
        format=llama3format,
        approach='LlamaTracker',
        param_magnitude='8B',
        train_data='data/mwoz2.4/train',
        eval_data='data/mwoz2.4/test',
        train_downsample=None,
        eval_dialogue_downsample=None,
        eval_exclude_speakers='bot',
        test_domains=[domain],
        eval_all_slots_per_domain=True,
        prediction_lowercase=True,
        prediction_fuzzy_match_candidates=True,
        epochs=0,
        max_sequence_length=1024,
        protected_input_length=900,
        train_batch_size=256,
        gradient_accumulation_steps=opt_args[machine]['gradient_accumulation_steps'],
        warmup_steps=100,
        optimizer='adafactor',
        learning_rate=1e-2,
        weight_decay=0.0,
        quantize='bf16',
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
        tokenizer_reponame='meta-llama/Meta-Llama-3.1-8B-Instruct',
    )
