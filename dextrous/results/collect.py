
import ezpyzy as ez
import dextrous.experiment as exp
import pathlib as pl
import socket as sk
import datetime as dt

class ExperimentTable(exp.LlamaTrackerExperimentTable, exp.T5TrackerExperimentTable):pass

def collect():
    machine = sk.gethostname()
    table = ExperimentTable.of([], fill=None)
    experiments_iter = [
        e for path in ('ex/LlamaTracker', 'ex/T5Tracker')
        for e in pl.Path(path).iterdir() if e.is_dir()
    ]
    rows = []
    columns = set(table().column_names)
    for experiment in experiments_iter:
        i = 1
        while True:
            iteration = experiment/str(i)
            if i == 1 and not iteration.exists():
                i = 0
                continue
            elif not iteration.exists():
                break
            hyperparam_path = experiment/'experiment.csv'
            results_path = iteration/'result.csv'
            performance_path = experiment/'performance.csv'
            if hyperparam_path.exists():
                hyperparam_row = ExperimentTable.of(hyperparam_path)
                file_save_time = hyperparam_path.stat().st_mtime
                file_save_datetime = dt.datetime.fromtimestamp(file_save_time).timetuple()
                if results_path.exists():
                    result_row = exp.Result.of(results_path)
                else:
                    result_row = ez.Table.of(dict(_=[None]))
                if performance_path.exists():
                    performance_row = exp.Performance.of(performance_path)
                else:
                    performance_row = ez.Table.of(dict(_=[None]))
                for row in (hyperparam_row, result_row, performance_row):
                    columns.update(row().column_names)
                row = hyperparam_row - result_row - performance_row
                rowdict = row().dict()
                rowdict['datetime'] = file_save_datetime
                rowdict['i'] = i
                rows.append(rowdict)
            if i == 0:
                break
            i += 1
    table = ExperimentTable.of(rows)
    filename = f'{machine}_results.csv'
    table().save(filename)
    table().save(f"~/dextrous/{filename}")
    return



if __name__ == '__main__':
    collect()