
import ezpyzy as ez
import pathlib as pl


def insert_few_shots_descriptions(slots_csv_path, few_shots_text, shots=3):
    assert shots <= 3
    slots_table = ez.Table.of(slots_csv_path)
    descmap = {d: None for d in slots_table.slot}
    original = None
    new = []
    for line in few_shots_text.splitlines():
        if line.strip('"') in descmap:
            original = line.strip('"')
            if original not in descmap:
                print(f'Warning: {original} not found in slots table')
        elif original and line.strip():
            if line.strip().startswith('ex. '):
                if len([
                    l for l in new if l.strip().startswith('ex. ')
                ]) < shots:
                    new.append(line)
            else:
                new.append(line)
        elif original and new:
            descmap[original] = '\n'.join(new)
            original = None
            new = []
    for i, row in enumerate(slots_table):
        if row.slot() in descmap:
            slots_table[i].description(descmap[row.slot()])
            print(f'Replaced description for slot {row.slot()}')
    slots_table().save(slots_csv_path)
    for split in ('train', 'valid', 'test'):
        slots_table().save(pl.Path(slots_csv_path).parent/split/'slot.csv')


if __name__ == '__main__':
    few_shots_text = ez.File('data/mwoz2.4_3s/few_shots_descriptions.txt').load()
    insert_few_shots_descriptions(
        'data/mwoz2.1_1s/slot.csv',
        few_shots_text,
        shots=1
    )