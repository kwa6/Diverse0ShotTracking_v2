import random

import ezpyzy as ez
import pathlib as pl
import dextrous.dst_data as dst
import json
import dextrous.utils as du


def create_error_analysis(*experiment_paths, n=100):
    base = None
    rows = []
    experiment_paths = [f"ex/LlamaTracker/{p}/0" for p in experiment_paths]
    for experiment_path in experiment_paths:
        experiment_path = pl.Path(experiment_path)
        predictions_path = experiment_path/'predictions.csv'
        predictions = dst.Prediction.of(predictions_path)
        experiment_hyperparams = experiment_path.parent / 'experiment.csv'
        params, values = ez.File(experiment_hyperparams).load()
        hyperparams = {p: json.loads(v) for p, v in zip(params, values)}
        if base is None:
            base = hyperparams['base']
        else:
            if base != hyperparams['base']:
                print(f'Warning, base {base} != {hyperparams["base"]}')
        eval_data_path = pl.Path(hyperparams['eval_data'])
        eval_data = dst.Data(str(eval_data_path))
        eval_data.predictions = predictions
        for dialogue in eval_data.examples():
            context = []
            for text, speaker, domain, predicted, state, prompts, generations in dialogue:
                context.append(text)
                print(text)
                goldsv = {(s, v) for s, v in state.items()}
                predsv = {(s, v) for s, v in predicted.items()}
                for s, _ in goldsv ^ predsv:
                    print('   ', s, ':', state.get(s, '___'), '/', predicted.get(s, '___'))
                    rows.append(dict(
                        context="\n".join(context),
                        slot=s,
                        gold=state.get(s, ''),
                        pred=predicted.get(s, ''),
                        label=''
                    ))
    if n:
        rows = random.sample(rows, min(n, len(rows)))
    table = ez.Table.of(rows)
    save_path = f'analysis/{base.replace("/","_")}_errors.csv'
    print('Saving to', save_path)
    table().save(save_path, json_cells=False)
    return table

def examples_by_slot(path, slot=None, n=10):
    data = dst.Data(path)
    examples = {}
    for dialogue in data.samples(accumulate_state=False):
        for ex in dialogue:
            for s, v in ex.state.items():
                examples.setdefault(s, []).append((ex.context, s, v))
    if slot is None:
        examples = [ex for exs in examples.values() for ex in random.sample(exs, min(n, len(exs)))]
    else:
        examples = random.sample(examples[slot], min(n, len(examples[slot])))
    rows = []
    for context, slot, value in examples:
        rows.append(dict(context='\n'.join(context), slot=slot, value=value, label=None))
    return ez.Table.of(rows)

def get_analysis_results(path):
    path = pl.Path(path)
    error_counts = {}
    data = ez.File(path).load()
    columns = dict(zip(data[0], zip(*data[1:])))
    table = ez.Table.of(columns)
    for error in table.label:
        error_counts[error] = error_counts.get(error, 0) + 1
    error_counts = dict(sorted(error_counts.items(), key=lambda x: x[1], reverse=True))
    error_percents = {error: count/len(table) for error, count in error_counts.items()}
    print('Got', len(table), 'errors')
    with open(path.with_suffix('.json'), 'w') as f:
        json.dump(error_counts, f, indent=4)



if __name__ == '__main__':
    # create_error_analysis(
    #     'WondrousPolisMassa', 'DashingStewjon', 'IllustriousGeneral',
    #     'IntrepidBossk', 'MysteriousRishi'
    # )
    # du.download('h100',
    # "analysis/ex_LlamaTracker_DazzlingDengar_5_errors.csv")
    # get_analysis_results("analysis/ex_LlamaTracker_DazzlingDengar_5_errors.csv")

    # exs = examples_by_slot('data/mwoz2.4/valid')
    # exs().save('analysis/mwoz_examples.csv', json_cells=False)

    # ----------------------------------------------------------------------------

    # create_error_analysis(
    #     'MysteriousWicket', 'VibrantShaak', 'ResilientMalachor',
    #     'SwiftPadmé', 'HyperspaceVulpter', n=100
    # )

    du.download('h100', 'analysis/ex_LlamaTracker_SereneBaze_8_errors.csv')

