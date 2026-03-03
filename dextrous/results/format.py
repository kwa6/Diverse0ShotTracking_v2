
import ezpyzy as ez
from dextrous.results.collect import ExperimentTable
import dextrous.utils as ut


def format(results:ExperimentTable):
    formatting = [
        (results.joint_goal_accuracy, 'JGA', 5),
        (results.slot_update_accuracy, 'SUA', 5),
        (results.base, 'base'),
        (results.param_magnitude, 'pm', 2),
        (results.lora, 'lora'),
        (results.quantize, 'q'),
        (results.learning_rate, 'LR'),
        (results.train_batch_size, 'BS'),
        (results.neg_examples_ratio, 'neg'),
        (results.train_percent_with_description, 'desc'),
        (results.train_percent_description_only, 'd.only'),
        (results.train_percent_with_categories, 'cat'),
        (results.train_percent_with_value_exs, 'exs'),
        (results.train_percent_value_ex_includes_actual_value, 'v.exs'),
        (results.train_filters_out_descriptions_with_actual_value, 'f.desc'),
        (results.train_remove_request_token_percent, 'no ?'),
        (results.uncased, 'ucase', 1),
        (results.machine, 'M', 1),
        (results.gradient_accumulation_steps, 'GAS'),
        (results.gen_batch_size, 'GBS'),
        (results.num_beams, 'beam'),
        (results.gpu_usage_after_training_gb, 'tGPU', 3),
        (results.gpu_usage_after_generation_gb, 'gGPU', 3),
        (results.train_data, 'train'),
        (results.eval_data, 'eval'),
        (results.train_downsample, 'DS'),
        (results.test_domains, 'domain'),
        (results.train_prop_add_continuation, 'cont'),
        (results.yield_every_x_epochs, 'Y'),
        (results.train_perplexity, 'tPP'),
    ]
    for column in results():
        for col, *fmt in formatting:
            if column is col:
                name, *fmt = fmt if fmt else [None]
                width, *fmt = fmt if fmt else [None]
                if name:
                    column.name = name
                if width:
                    column[:] = [str(x)[:width] for x in column]
    results.base[:] = ['/'.join(x.split('/')[-2:]).replace('-{param_magnitude}', '') for x in results.base]
    results.train_data[:] = [x and x.replace('data/', '') for x in results.train_data]
    results.eval_data[:] = [x.replace('data/', '') for x in results.eval_data]
    results.test_domains[:] = [', '.join(x) if isinstance(x, list) else x for x in results.test_domains]
    return results

def no_results(results:ExperimentTable):
    results = ~results[results.joint_goal_accuracy == None]
    results = format(results)
    results = results[
        results.experiment,
        results.machine,
        results.train_data,
        results.eval_data,
        results.train_downsample,
        results.base,
        results.param_magnitude,
        results.lora,
        results.quantize,
        results.train_batch_size,
        results.gradient_accumulation_steps,
        results.gen_batch_size,
        results.num_beams,
        results.gpu_usage_after_training_gb,
        results.gpu_usage_after_generation_gb,
    ]
    results().save('results/no_results.csv', json_cells=False)
    return results

def resource_results(results:ExperimentTable):
    results = ~results
    results = format(results)
    results = results[
        results.experiment,
        results.i,
        results.machine,
        results.base,
        results.param_magnitude,
        results.lora,
        results.quantize,
        results.train_batch_size,
        results.gradient_accumulation_steps,
        results.gen_batch_size,
        results.num_beams,
        results.gpu_usage_after_training_gb,
        results.gpu_usage_after_generation_gb,
    ]
    results().save('results/resource_results.csv', json_cells=False)
    return results


def dsg5k_on_mwoz_valid(results:ExperimentTable):
    results = ~results[results.joint_goal_accuracy != None]
    results = ~results[['mwoz' in x and 'valid' in x for x in results.eval_data]]
    results:ExperimentTable = ~results[[x and 'dsg5k' in x for x in results.train_data]]
    results = format(results)
    results().sort(results.slot_update_accuracy, reverse=True)
    results = results[
        results.experiment,
        results.i,
        results.joint_goal_accuracy,
        results.slot_update_accuracy,
        results.base,
        results.param_magnitude,
        results.lora,
        results.quantize,
        results.train_downsample,
        results.learning_rate,
        results.train_batch_size,
        results.neg_examples_ratio,
        results.train_percent_with_description,
        results.train_percent_description_only,
        results.train_percent_with_categories,
        results.train_percent_with_value_exs,
        results.train_percent_value_ex_includes_actual_value,
        results.train_filters_out_descriptions_with_actual_value,
        results.train_remove_request_token_percent,
        results.uncased,
        results.machine
    ]
    results().save('results/dsg5k_to_mwoz_valid.csv', json_cells=False)
    return results

def dsg5k_on_mwoz_hotel(results:ExperimentTable):
    results = ~results[results.joint_goal_accuracy != None]
    results = ~results[['mwoz' in x and 'valid' in x for x in results.eval_data]]
    results = ~results[[bool(x and 'hotel' in x) for x in results.test_domains]]
    results:ExperimentTable = ~results[[x and 'dsg5k' in x for x in results.train_data]]
    results = format(results)
    results().sort(results.joint_goal_accuracy, reverse=True)
    results = results[
        results.machine,
        results.experiment,
        results.i,
        results.joint_goal_accuracy,
        results.train_perplexity,
        results.base,
        results.param_magnitude,
        results.lora,
        results.lora_alpha,
        results.train_downsample,
        results.yield_every_x_epochs,
        results.neg_examples_ratio,
    ]
    results().save('results/dsg5k_to_mwoz_hotel.csv', json_cells=False)
    performances = {}
    models = results().group(results[results.base, results.param_magnitude, results.lora, results.lora_alpha])
    for model, perfs in models.items():
        iters = {}
        for row in perfs:
            iters[row.i()] = max(
                float(row.joint_goal_accuracy()),
                iters.get(float(row.joint_goal_accuracy()), 0)) # noqa
        iters = sorted(iters.items())
        model = str(model)
        performances[model] = [x[1] for x in iters]
    # ut.plot_lines(**performances)
    return results

def gen_param_selection(results:ExperimentTable):
    results = ~results[results.joint_goal_accuracy != None]
    results = ~results[['mwoz' in x and 'valid' in x for x in results.eval_data]]
    results: ExperimentTable = ~results[results.train_data == None]
    results = format(results)
    results().sort(results.slot_update_accuracy, reverse=True)
    results = results[
        results.experiment,
        results.i,
        results.joint_goal_accuracy,
        results.slot_update_accuracy,
        results.model_path,
        results.base,
        results.param_magnitude,
        results.lora,
        results.quantize,
        results.train_downsample,
        results.repetition_penalty,
        results.num_beams,
        results.machine
    ]
    results().save('results/gen_param_on_mwoz.csv', json_cells=False)
    return results

def mwoz_valid_results(results:ExperimentTable):
    results = ~results[results.joint_goal_accuracy != None]
    results = ~results[['mwoz' in x and 'valid' in x for x in results.eval_data]]
    results: ExperimentTable = ~results[[x and 'mwoz' in x for x in results.train_data]]
    by_expt = results().group(results[
        results.base,
        results.param_magnitude,
        results.lora,
        results.lora_alpha,
        results.quantize,
        results.learning_rate,
        results.train_batch_size,
        results.neg_examples_ratio,
        results.train_percent_with_description,
        results.train_percent_description_only,
        results.train_percent_with_categories,
        results.train_percent_with_value_exs,
        results.train_percent_value_ex_includes_actual_value,
        results.train_filters_out_descriptions_with_actual_value,
        results.train_remove_request_token_percent,
        results.train_prop_add_continuation,
        results.uncased,
        results.train_data,
        results.eval_data
    ])
    pivoted = []
    for _, rows in by_expt.items():
        expts = list(rows.experiment)
        iters = list(rows.i)
        machines = list(rows.machine)
        domains = list(rows.test_domains)
        jgas = list(rows.joint_goal_accuracy)
        first_row = rows[0]().dict()
        jga_scores = {}
        for (domain,), jga, expt, i, m in zip(domains, jgas, expts, iters, machines):
            if jga > jga_scores.get(domain, -1):
                jga_scores[domain] = jga
                first_row[domain] = f'{jga:.3f}'
                first_row[domain[:2] + '_ex'] = str(expt) + f'_{i}'
                first_row[domain[:2] + '_m'] = m[:1]
        avg_jga = sum(jga_scores.values()) / len(jga_scores)
        first_row['joint_goal_accuracy'] = avg_jga
        pivoted.append(first_row)
    results = ExperimentTable.of(pivoted)
    results = format(results)
    results().sort(results.joint_goal_accuracy, reverse=True)
    results = results[
        results.base,
        results.param_magnitude,
        results.joint_goal_accuracy,
        results.hotel,
        results.ho_ex,
        results.ho_m,
        results.restaurant,
        results.re_ex,
        results.re_m,
        results.attraction,
        results.at_ex,
        results.at_m,
        results.train,
        results.tr_ex,
        results.tr_m,
        results.taxi,
        results.ta_ex,
        results.ta_m,
        results.lora,
        results.quantize,
        results.learning_rate,
        results.train_batch_size,
    ]
    results().save('results/mwoz_valid_results.csv', json_cells=False)
    return results


def mwoz_test_results(results:ExperimentTable, version='mwoz2.1', split='test', segregate_bases=False):
    results = ~results[results.joint_goal_accuracy != None]
    results = ~results[[f'{version}/{split}' in x for x in results.eval_data]]
    setups = dict(
        t5bl = {
            'UnyieldingIego', 'ValiantIthor', 'VibrantCerea', 'FieryZolan', 'DaringFinn',},
        t5ft = {
            'ExoticGeonosis', 'CaptivatingVentress',
            'ResilientTatooine', 'HarmoniousTauntaun', 'GalacticKylo'},
        t5pt={
            'TimelessOrson/21', },
        llbl = {
            'UnforgettableGreedo', 'EnthrallingScarif',
            'IllustriousGreedo', 'LuminousLuke', 'PulsarAqualish'},
        llft = {
            'SupernovaOssus', 'SpiritedMon',
            'SpiritedPamarthe', 'UntamedCatoNeimoidia', 'HeroicLandosFarm'},
        llpt={
            'DazzlingDengar/5'},
        llbl3s={
            'DazzlingAhsoka', 'RogueNevarro', 'ResplendentMace', 'AstralPorg', 'ThunderousIlum'},
        llft3s = {
            'UnchartedTatooII', 'VibrantTaris', 'UnchartedPaz', 'SereneBaze', 'BoldBB9E'},
        llpt3s={
            'SupernovaPlo/5'},
    )
    setup_col = []
    for base, gc in zip(results.base, results.groupcode):
        for groupcode, bases in setups.items():
            if any(b in base for b in bases):
                setup_col.append(groupcode)
                break
        else:
            setup_col.append(gc)
    results.setup = ez.Column(setup_col)
    by_setup = results().group(list(zip(
        results.setup,
        results.approach,
        *([results.base,] if segregate_bases else []),
        results.param_magnitude,
        results.learning_rate,
        results.optimizer,
        results.weight_decay,
    )))
    pivoted = []
    for setup, rows in by_setup.items():
        expts = list(rows.experiment)
        iters = list(rows.i)
        machines = list(rows.machine)
        domains = list(rows.test_domains)
        jgas = list(rows.joint_goal_accuracy)
        first_row = rows[0]().dict()
        jga_scores = {}
        for ds, jga, expt, i, m in zip(domains, jgas, expts, iters, machines):
            if not ds or ds[0] not in {'hotel', 'restaurant', 'attraction', 'train', 'taxi'}:
                continue
            domain = ds[0]
            if jga > jga_scores.get(domain, -1):
                jga_scores[domain] = jga
                first_row[domain] = f'{jga:.3f}'
                first_row[domain[:2] + '_ex'] = str(expt) + f'_{i}'
                first_row[domain[:2] + '_m'] = m[:1]
        if not jga_scores:
            continue
        avg_jga = sum(jga_scores.values()) / len(jga_scores)
        first_row['joint_goal_accuracy'] = avg_jga
        pivoted.append(first_row)
    results = ExperimentTable.of(pivoted)
    results = format(results)
    results().sort(results.joint_goal_accuracy, reverse=True)
    results = results[
        results.setup,
        results.joint_goal_accuracy,
        results.hotel,
        results.ho_ex,
        results.ho_m,
        results.restaurant,
        results.re_ex,
        results.re_m,
        results.attraction,
        results.at_ex,
        results.at_m,
        results.train,
        results.tr_ex,
        results.tr_m,
        results.taxi,
        results.ta_ex,
        results.ta_m,
    ]
    results().save(f'results/{version}_{split}_results.csv', json_cells=False)


def sgd_eval_results(results:ExperimentTable, split='valid'):
    eval = f'data/sgd/{split}'
    results = ~results[results.eval_data == eval]
    results = ~results[results.joint_goal_accuracy != None]
    results = format(results)
    results().sort(results.joint_goal_accuracy, reverse=True)
    results = results[
        results.groupcode,
        results.experiment,
        results.i,
        results.base,
        results.lora,
        results.quantize,
        results.train_data,
        results.eval_data,
        results.joint_goal_accuracy,
        results.machine,
    ]
    results().save(f'results/sgd_{split}_results.csv', json_cells=False)



def main():
    results = ExperimentTable.of('results/results.csv')
    dsg5k_on_mwoz_valid(results)
    mwoz_valid_results(results)
    no_results(results)
    resource_results(results)
    gen_param_selection(results)
    dsg5k_on_mwoz_hotel(results)
    mwoz_test_results(results, 'mwoz2.1')
    mwoz_test_results(results, 'mwoz2.4')
    mwoz_test_results(results, 'mwoz2.4_3s')
    mwoz_test_results(results, 'mwoz2.1_3s')
    mwoz_test_results(results, 'mwoz2.4', 'valid', segregate_bases=True)
    # mwoz_test_results(results, 'mwoz2.1', 'valid')
    sgd_eval_results(results, 'valid')

if __name__ == '__main__':
    main()