
import ezpyzy as ez
import json

test_expt_names = []

test_result_files = [
    'mwoz2.1_3s_test_results.csv',
    'mwoz2.1_test_results.csv',
    'mwoz2.4_3s_test_results.csv',
    'mwoz2.4_test_results.csv',
]

for test_result_file in test_result_files:
    data = ez.File(f'results/{test_result_file}').load(format='csv')
    header, rows = data[0], data[1:]
    for colname, *column in zip(header, *rows):
        if colname.endswith('_ex'):
            for experiment_name in column:
                splitter = experiment_name.rfind('_')
                if splitter != -1:
                    experiment_name = experiment_name[:splitter] + '/' + experiment_name[splitter+1:]
                if experiment_name:
                    test_expt_names.append(experiment_name)

results_table = ez.Table.of('results/results.csv')

test_expts_set = set(test_expt_names)

whitelisted = set()
base_models_set = set()

for experiment, iteration, basemodel in zip(
    results_table.experiment,
    results_table.i,
    results_table.base
):
    name = f'{experiment}/{iteration}'
    if name in test_expts_set:
        # print(experiment, iteration, basemodel)
        whitelisted.add(basemodel)
        basemodel_name = '/'.join(basemodel.split('/')[-2:])
        base_models_set.add(basemodel_name)

print(f'Found {len(whitelisted)} initial whitelisted models.')
initial_whitelisted = set(whitelisted)

for experiment, iteration, basemodel in zip(
    results_table.experiment,
    results_table.i,
    results_table.base
):
    name = f'{experiment}/{iteration}'
    if name in base_models_set:
        # print(experiment, iteration, basemodel)
        whitelisted.add(basemodel)

print(f'Found {len(whitelisted)} total whitelisted models.')
whitelisted = {x for x in whitelisted if x.startswith('ex/')}

whitelist = '\n'.join(whitelisted)
ez.File('results/whitelist.txt').save(whitelist)


all_expt_paths = set()
for approach, experiment, iteration in zip(
    results_table.approach, results_table.experiment, results_table.i
):
    path = f'ex/{approach}/{experiment}/{iteration}'
    all_expt_paths.add(path)

blacklist_candidates = set(all_expt_paths)
for whitelisted_path in whitelisted:
    blacklist_candidates.remove(whitelisted_path)

blacklisted = list(blacklist_candidates)

blacklist = '\n'.join(blacklisted)

ez.File('results/blacklist.txt').save(blacklist)







