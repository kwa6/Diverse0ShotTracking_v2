
from dextrous.launch import launch


machine = 'h100'
opt_args = dict(
    h100=dict(
        gen_batch_size=6,
        gradient_accumulation_steps=128,
    ),
    tebuna=dict(
        gen_batch_size=6,
        gradient_accumulation_steps=32,
    ),
)


for groupcode, domain, base in [
    # ('hotel', 'ex/T5Tracker/ExoticGeonosis/8'),
    # ('taxi', 'ex/T5Tracker/CaptivatingVentress/9'),
    # ('attraction', 'ex/T5Tracker/ResilientTatooine/10'),
    # ('train', 'ex/T5Tracker/HarmoniousTauntaun/3'),
    # ('restaurant', 'ex/T5Tracker/GalacticKylo/3'),
    # ('hotel', 'ex/T5Tracker/TimelessOrson/21'),
    # ('taxi', 'ex/T5Tracker/TimelessOrson/21'),
    # ('attraction', 'ex/T5Tracker/TimelessOrson/21'),
    # ('train', 'ex/T5Tracker/TimelessOrson/21'),
    # ('restaurant', 'ex/T5Tracker/TimelessOrson/21'),

    # ('hotel', 'ex/T5Tracker/UnyieldingIego/9'),
    # ('taxi', 'ex/T5Tracker/ValiantIthor/6'),
    # ('attraction', 'ex/T5Tracker/VibrantCerea/11'),
    # ('train', 'ex/T5Tracker/FieryZolan/8'),
    # ('restaurant', 'ex/T5Tracker/DaringFinn/4'),

    # ('hotel', 'ex/T5Tracker/TimelessOrson/10'),
    # ('taxi', 'ex/T5Tracker/TimelessOrson/10'),
    # ('attraction', 'ex/T5Tracker/TimelessOrson/10'),
    # ('train', 'ex/T5Tracker/TimelessOrson/10'),
    # ('restaurant', 'ex/T5Tracker/TimelessOrson/10'),
    
    # ('hotel', 'ex/T5Tracker/TimelessOrson/5'),
    # ('taxi', 'ex/T5Tracker/TimelessOrson/5'),
    # ('attraction', 'ex/T5Tracker/TimelessOrson/5'),
    # ('train', 'ex/T5Tracker/TimelessOrson/5'),
    # ('restaurant', 'ex/T5Tracker/TimelessOrson/5'),

    # # flan-lora-all-baseline-mwoz
    # ('flan-lora-all-baseline-mwoz', 'hotel', 'ex/T5Tracker/MysticalEzra/5'),
    # ('flan-lora-all-baseline-mwoz', 'taxi', 'ex/T5Tracker/SereneVandor/3'),
    # ('flan-lora-all-baseline-mwoz', 'attraction', 'ex/T5Tracker/FierceEndor/10'),
    # ('flan-lora-all-baseline-mwoz', 'train', 'ex/T5Tracker/LegendaryNaboo/6'),
    # ('flan-lora-all-baseline-mwoz', 'restaurant', 'ex/T5Tracker/LimitlessRex/7'),
    # # flan-lora-all-dsg5k-finetune-mwoz
    # ('flan-lora-all-dsg5k-finetune-mwoz', 'hotel', 'ex/T5Tracker/RelentlessPasaana/8'),
    # ('flan-lora-all-dsg5k-finetune-mwoz', 'taxi', 'ex/T5Tracker/FiercePoe/6'),
    # ('flan-lora-all-dsg5k-finetune-mwoz', 'attraction', 'ex/T5Tracker/NobleMortis/10'),
    # ('flan-lora-all-dsg5k-finetune-mwoz', 'train', 'ex/T5Tracker/UnforgettableSullust/4'),
    # ('flan-lora-all-dsg5k-finetune-mwoz', 'restaurant', 'ex/T5Tracker/ResplendentWatto/10'),

    # # t5-lora-all-baseline-mwoz
    # ('t5-lora-all-baseline-mwoz', 'hotel', 'ex/T5Tracker/EmboldenedBodhi/11'),
    # ('t5-lora-all-baseline-mwoz', 'taxi', 'ex/T5Tracker/NebulousKessel/6'),
    # ('t5-lora-all-baseline-mwoz', 'attraction', 'ex/T5Tracker/DazzlingGreef/7'),
    # ('t5-lora-all-baseline-mwoz', 'train', 'ex/T5Tracker/CaptivatingAtollon/9'),
    # ('t5-lora-all-baseline-mwoz', 'restaurant', 'ex/T5Tracker/ThunderousZolan/11'),
    # # t5-lora-all-dsg5k-finetune-mwoz
    ('t5-lora-all-dsg5k-finetune-mwoz', 'hotel', 'ex/T5Tracker/UnchartedGreef/10'),
    ('t5-lora-all-dsg5k-finetune-mwoz', 'taxi', 'ex/T5Tracker/RadiantDantooine/8'),
    ('t5-lora-all-dsg5k-finetune-mwoz', 'attraction', 'ex/T5Tracker/LivelyClakdorVII/11'),
    ('t5-lora-all-dsg5k-finetune-mwoz', 'train', 'ex/T5Tracker/ResilientRyndellia/3'),
    ('t5-lora-all-dsg5k-finetune-mwoz', 'restaurant', 'ex/T5Tracker/IconicZolan/10'),


    # # t5-baseline-mwoz
    # ('t5-baseline-mwoz', 'hotel', 'ex/T5Tracker/CharismaticDQar/4'),
    # ('t5-baseline-mwoz', 'taxi', 'ex/T5Tracker/UnforgettableCatoNeimoidia/6'),
    # ('t5-baseline-mwoz', 'attraction', 'ex/T5Tracker/CaptivatingVandor/3'),
    # ('t5-baseline-mwoz', 'train', 'ex/T5Tracker/UnchartedPillio/2'),
    # ('t5-baseline-mwoz', 'restaurant', 'ex/T5Tracker/LimitlessPonda/2'),
    #  # t5-finetune-mwoz
    # ('t5-finetune-mwoz', 'hotel', 'ex/T5Tracker/EnthrallingPillio/5'),
    # ('t5-finetune-mwoz', 'taxi', 'ex/T5Tracker/ElectricManaan/1'),
    # ('t5-finetune-mwoz', 'attraction', 'ex/T5Tracker/HyperspaceLahmu/1'),
    # ('t5-finetune-mwoz', 'train', 'ex/T5Tracker/UnchartedLahmu/1'),
    # ('t5-finetune-mwoz', 'restaurant', 'ex/T5Tracker/DaringUtapau/8')
]:
    launch.T5Launch(
        groupcode=groupcode,
        base=base,
        approach='T5Tracker',
        param_magnitude='11b',
        format='',
        train_data='data/mwoz2.4/train',
        eval_data='data/mwoz2.1/test',
        train_downsample=None,
        eval_dialogue_downsample=None,
        eval_exclude_speakers='bot',
        test_domains=[domain],
        eval_all_slots_per_domain=True,
        prediction_lowercase=True,
        prediction_fuzzy_match_candidates=True,
        epochs=0,
        max_sequence_length=1024,
        train_batch_size=256,
        gradient_accumulation_steps=opt_args[machine]['gradient_accumulation_steps'],
        warmup_steps=100,
        
        optimizer='adafactor',
        learning_rate=1e-2,
        weight_decay=0.0,
        quantize='nf4',
        lora=32,
        lora_alpha=64,
        lora_dropout=0.0,
        lora_merge_on_load=False,
        lora_modules=['q', 'v', 'k', 'o', 'wi_0', 'wi_1', 'wo'],
        # learning_rate=1e-3,
        # weight_decay=5e-3,
        # optimizer='adafactor',
        # dropout=0.0,
        # lora=None,
        # quantize='bf16',

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
