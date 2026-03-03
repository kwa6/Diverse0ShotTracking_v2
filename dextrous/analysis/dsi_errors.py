

import ezpyzy as ez
import dextrous.induction.cluster as dcl
import random as rng


def error_analysis_dsg5k_noise(path, n=30):
    induced = dcl.Clustered.of(path)
    induced = ~induced[induced.cluster_id == -1]
    context_slot_value = list(zip(
        induced.context, induced.slot, induced.value, [None]*len(induced)))
    sampled = rng.sample(context_slot_value, n)
    header = ['context', 'slot', 'value', 'annotation']
    ez.File('analysis/dsg5k_noise.csv').save([header] + sampled)


def error_analysis_missing_slots(path):
    table = ez.File(path).load()
    header = table[0]
    table = table[1:]
    columns = dict(zip(header, zip(*table)))
    errors = {}
    for error in columns['error']:
        errors[error] = errors.get(error, 0) + 1
    print('\n'.join(f"{k}: {v}" for k,v in errors.items()))
    ez.File('analysis/mwoz_errors.json').save(errors)


if __name__ == '__main__':
    # error_analysis_dsg5k_noise('data/dsg5k/dsg5k_induced.csv')
    error_analysis_missing_slots('results/mwoz_induction_error_analysis.csv')