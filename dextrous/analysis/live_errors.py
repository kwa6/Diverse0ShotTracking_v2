
import dextrous.utils as utils
import ezpyzy as ez
import dextrous.dst_data as dst
import dextrous.tracker as dex
import dextrous.preprocessing as dpp
import dextrous.metrics as metrics
import random as rng


def analyze(
    tracker,
    data: dst.Data,
    domains=None,
    downsample_dialogues=5,
    display_prompts=False,
    seed=42,
):
    if seed:
        rng.seed(seed)
    if domains:
        dpp.drop_domains(data, domains, include_specified=True)
    dpp.add_neg_slot_targets(data, per_domain=True)
    dpp.exclude_speakers(data, {'bot'})
    if downsample_dialogues:
        dpp.downsample_dialogues(data, downsample_dialogues)
    tracker.predict(data)
    for dialogue in data.examples():
        if not display_prompts:
            print('\n\n')
            for text, speaker, domain, predicted, actual, prompts, generations in dialogue:
                print(f'{speaker}: {text} ({domain})')
                if predicted or actual:
                    print('   P: ', ", ".join(f"{k}: {v}" for k, v in predicted.items()))
                    print('   A: ', ", ".join(f"{k}: {v}" for k, v in actual.items()))
        else:
            for text, speaker, domain, predicted, actual, prompts, generations in dialogue:
                if not domains or domain in domains:
                    for prompt, generation in zip(prompts, generations):
                        print(prompt)
                        print('Generated:', generation)
                        if predicted or actual:
                            print('   P: ', ", ".join(f"{k}: {v}" for k, v in predicted.items()))
                            print('   A: ', ", ".join(f"{k}: {v}" for k, v in actual.items()))
                        print('\n\n'+'-'*80+'\n\n')
    jga = metrics.joint_goal_accuracy(data, speakers={'user'}, domains=domains)
    print(f'Joint Goal Accuracy: {jga:.3f}')
    return data


def main():
    tracker = dex.LlamaTracker(dex.LlamaTrackerHyperparameters(
        'ex/LlamaTracker/SupernovaNute/1', lora_merge_on_load=False,
        gen_batch_size=1, max_sequence_length=256, protected_input_length=98
    ))
    data = dst.Data('data/mwoz2.4/valid')
    analyze(tracker, data, ['taxi'], 5, display_prompts=True)

if __name__ == '__main__':
    main()