import dextrous.dst_data as dst
import pathlib as pl
import random as rng
import ezpyzy as ez

"""
The area of the city the attraction is located
  ex. where can i go on the west side of town ? -> attraction area? west
  ex. do you have anywhere interesting in the centre ? -> attraction area? centre
  ex. can you recommend an attraction ? any area is fine . -> attraction area? any
 [north, south, east, west, centre, any]
"""

def fewshot_sampling(data_path):
    data_path = pl.Path(data_path)
    train = dst.Data(data_path / 'train')
    valid = dst.Data(data_path / 'valid')
    test = dst.Data(data_path / 'test')
    print('Data loaded!')
    slots = test.slots
    examples_by_slot = {}

    for data in [train, valid, test]:

        tid_to_text = {tid: text for tid, text in zip(data.turns.turn_id, data.turns.text)}

        for slot, value, svid, tid in zip(
            data.slot_values.slot, data.slot_values.value, 
            data.slot_values.slot_value_id, data.slot_values.turn_id
        ):
            examples_by_slot.setdefault(slot, []).append((
                tid_to_text[tid], value
            ))
    
    fewshots_by_slot = {}
    for slot, examples in examples_by_slot.items():
        rng.shuffle(examples)
        examples.sort(key=lambda x: int(x[1] in x[0]), reverse=True)
        fewshots = examples[:3]
        fewshots_by_slot[slot] = fewshots
    
    slot_descriptions = {}
    for slot, description in zip(slots.slot, slots.description):
        if slot in fewshots_by_slot:
            splitter = description.find('(e.g.')
            if splitter == -1:
                splitter = description.find('[')
                if splitter == -1:
                    raise AssertionError(f'Invalid description: {description}')
            left, right = description[:splitter], description[splitter:]
            new_description = left.rstrip()
            for shot_text, shot_value in fewshots_by_slot[slot]:
                new_description += f'\n  ex. {shot_text} -> {slot}? {shot_value}'
            new_description += '\n ' + right
            slot_descriptions[slot] = new_description
        else:
            slot_descriptions[slot] = description

    for data, split in [
        (train, 'train'), 
        (valid, 'valid'), 
        (test, 'test')
    ]:
        description_column = [slot_descriptions[slot] for slot in data.slots.slot]
        data.slots.description[:] = description_column
        data.save(data_path.with_name(data_path.name + '_3s') / split)

if __name__ == '__main__':
    fewshot_sampling('data/sgd')