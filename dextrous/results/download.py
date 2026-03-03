
import ezpyzy as ez
import fabric as fab
from dextrous.results.collect import ExperimentTable

altusrs = [
    'sfillwo'
]

hosts = dict(
    tebuna='localhost:55555',
    h100='localhost:55556'
)
password = ez.File('~/.pw/emory').read().strip()
project = '/local/scratch/jdfinch/dextrous/'
credentials = dict(
    user='jdfinch',
    connect_kwargs=dict(password=password)
)

def download_results() -> ExperimentTable:
    for machine, host in hosts.items():
        print(f'Connected to {machine}')
        with fab.Connection(host, **credentials) as conn:
            conn.get(
                f'{project}/{machine}_results.csv',
                f'results/{machine}_results.csv'
            )
            for usr in altusrs:
                try:
                    conn.get(
                        f'/home/{usr}/dextrous/{machine}_results.csv',
                        f'results/{machine}_{usr}_results.csv'
                    )
                    print(f'Got {machine}_{usr}_results.csv')
                except Exception:
                    pass
            print(f'Got {machine}_results.csv')
    tebuna_results = ExperimentTable.of('results/tebuna_results.csv')
    tebuna_results.machine = ez.Column(['tebuna']*len(tebuna_results))
    h100_results = ExperimentTable.of('results/h100_results.csv')
    h100_results.machine = ez.Column(['h100']*len(h100_results))
    s_h100_results = ExperimentTable.of('results/h100_sfillwo_results.csv')
    s_h100_results.machine = ez.Column(['h100']*len(s_h100_results))
    results = ExperimentTable.of(tebuna_results().dicts() + h100_results().dicts() + s_h100_results().dicts())
    results().sort(results.datetime, reverse=True)
    results().save('results/results.csv')
    return results

if __name__ == '__main__':
    results = download_results()