
import ezpyzy as ez
from dextrous.dst_data import Data
import csv

def domains_to_csv():
    with ez.check('Loading dsg5k...'):
        data = Data('data/dsg5k/train')
    unique_domains = set(data.slots.domain)
    print(len(unique_domains))
    csvwriter = csv.writer(open('unique_domains.csv', 'w'))
    csvwriter.writerow(['domain','exclude'])
    csvwriter.writerows([[ud,0] for ud in unique_domains])

if __name__ == '__main__':
    with ez.check('Loading dsg5k...'):
        data = Data('data/dsg5k/train')

    # print(data.turns[:5]().display(30), '\n')
    # print(data.slot_values[:5]().display(30), '\n')
    # print(data.slots[:5]().display(30))