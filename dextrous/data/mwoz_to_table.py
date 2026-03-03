
import ezpyzy as ez
import dataclasses as dc
import os
import pathlib as pl

import dextrous.dst_data as dst


descriptions = {
    "attraction-area": "The area of the city the attraction is located [north, south, east, west, centre, any]",
    "attraction-name": "The name or title of the attraction (e.g. clare hall, cambridge arts theater, scott polar museum, any)",
    "attraction-type": "The category or type of attraction (e.g. museum, theatre, park, any)",
    "bus-arriveBy": "The arrival time at the destination at the end of the ride (e.g. 12:45, 02:20, 20:00, any)",
    "bus-book people": "The number of people riding the bus that need tickets [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]",
    "bus-day": "The day of the week for the bus ride [monday, tuesday, wednesday, thursday, friday, saturday, sunday, tomorrow, any]",
    "bus-departure": "The starting location the person is coming from and departing (e.g. aylesbray lodge guest, cineworld, birmingham new street, any)",
    "bus-destination": "The destination the person is going to or arriving at (e.g. huntingdon marriott hotel, curry prince, london liverpool street, any)",
    "bus-leaveAt": "The departure time when the ride will start (e.g. 11:15, 01:30, 18:10, any)",
    "hospital-department": "The category or type of hospital department (e.g. cardiology, diabetes and endocrinology, transplant high dependency unity, any)",
    "hotel-area": "The area of the city the hotel is located [north, south, east, west, centre, any]",
    "hotel-book day": "The day of the week for the hotel booking [monday, tuesday, wednesday, thursday, friday, saturday, sunday, any]",
    "hotel-book people": "The number of people staying at the hotel [1, 2, 3, 4, 5, 6, 7, 8]",
    "hotel-book stay": "How many nights for the duration of the stay [1, 2, 3, 4, 5, 6, 7, 8]",
    "hotel-internet": "Whether the hotel provides free internet wifi [yes, no, any]",
    "hotel-name": "The name or title of the hotel (e.g. avalon, lovell lodge, holiday inn, any)",
    "hotel-parking": "Whether the hotel provides free parking [yes, no, any]",
    "hotel-pricerange": "The price range or cost of the hotel [cheap, moderate, expensive, any]",
    "hotel-stars": "The rating or number of stars of the hotel [0, 1, 2, 3, 4, 5, any]",
    "hotel-type": "The category or type of hotel [bed and breakfast, guest house, hotel, any]",
    "restaurant-area": "The area of the city the restaurant is located [north, south, east, west, centre, any]",
    "restaurant-book day": "The day of the week for the restaurant reservation [monday, tuesday, wednesday, thursday, friday, saturday, sunday, tomorrow, any]",
    "restaurant-book people": "The number of people eating at the restaurant [1, 2, 3, 4, 5, 6, 7, 8].",
    "restaurant-book time": "The arrival time for the restaurant reservation (e.g. 12:45, 02:20, 20:00, any)",
    "restaurant-food": "The category or type of food cuisine offered at the restaurant (e.g. mexican, seafood, modern global, any)",
    "restaurant-name": "The name or title of the restaurant (e.g. golden wok, the oak bistro, the varsity restaurant, any)",
    "restaurant-pricerange": "The price range or cost of the restaurant [cheap, moderate, expensive, any]",
    "taxi-arriveBy": "The arrival time at the destination at the end of the ride (e.g. 12:45, 02:20, 20:00, any)",
    "taxi-departure": "The starting location the person is coming from and departing (e.g. aylesbray lodge guest, cineworld, birmingham new street train station, any)",
    "taxi-destination": "The destination the person is going to or arriving at (e.g. huntingdon marriott hotel, curry prince, london liverpool street train station, any)",
    "taxi-leaveAt": "The departure time when the ride will start (e.g. 11:15, 01:30, 18:10, any)",
    "train-arriveBy": "The arrival time at the destination at the end of the ride (e.g. 12:45, 02:20, 20:00, any)",
    "train-book people": "The number of people riding the train that need tickets [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15]",
    "train-day": "The day of the week for the train ride [monday, tuesday, wednesday, thursday, friday, saturday, sunday, tomorrow, any]",
    "train-departure": "The starting location the person is coming from and departing (e.g. aylesbray lodge guest, cineworld, birmingham new street, any)",
    "train-destination": "The destination the person is going to or arriving at (e.g. huntingdon marriott hotel, curry prince, london liverpool street, any)",
    "train-leaveAt": "The departure time when the ride will start (e.g. 11:15, 01:30, 18:10, any)"
}

descriptions = {k.replace(' ', '-'): v for k, v in descriptions.items()}

slot_name_map = {
    'train-leaveAt': 'train leave at',
    'train leaveat': 'train leave at',
    'train-arriveBy': 'train arrive by',
    'train arriveby': 'train arrive by',
    'taxi-leaveAt': 'taxi leave at',
    'taxi leaveat': 'taxi leave at',
    'taxi-arriveBy': 'taxi arrive by',
    'taxi arriveby': 'taxi arrive by',
    'bus-leaveAt': 'bus leave at',
    'bus-arriveBy': 'bus arrive by',
    'hotel-pricerange': 'hotel price range',
    'restaurant-pricerange': 'restaurant price range',
}

fixed_descriptions = {slot_name_map.get(k, k).replace('-', ' '): v for k, v in descriptions.items()}

def mwoz_to_table(folder):
    ontology = ez.File(f'{folder}/ontology.json').load()
    slot_rows = []
    value_candidate_rows = []
    slot_map = {}
    for slot, values in ontology.items():
        slot = slot.replace('semi-', '').replace(' ', '-')
        domain = slot.split('-')[0]
        slot = dst.Slot(
            slot=slot,
            domain=domain,
            description=descriptions[slot]
        )
        slot_rows.append(slot)
        slot_id = slot.slot_id()
        slot_map[slot.slot().lower()] = slot_id
        for value in values:
            value = ", ".join(value.split('|'))
            value_candidate = dst.ValueCandidate(
                candidate_value=value,
                slot_id=slot_id,
                is_provided=True
            )
            value_candidate_rows.append(value_candidate)
    slot_table = dst.Slot.of(slot_rows) # noqa
    slot_value_candidate_table = dst.ValueCandidate.of(value_candidate_rows)
    slot_table_file = f'{folder}/slot.csv'
    slot_value_candidate_file = f'{folder}/value_candidate.csv'
    slot_table().save(slot_table_file)
    slot_value_candidate_table().save(slot_value_candidate_file)
    for split in ('dev', 'train', 'test',):
        mwoz = ez.File(f'{folder}/{split}_dials.json').load()
        if split == 'dev':
            split = 'valid'
        turn_rows = []
        slot_value_rows = []
        for dialogue in mwoz:
            dialogue_id = dialogue['dialogue_idx']
            dialogue_domains = dialogue['domains']
            belief_state = {}
            for turn in dialogue['dialogue']:
                turn_index = turn['turn_idx']
                system_turn_index = turn_index * 2
                user_turn_index = system_turn_index + 1
                turn_domain = turn['domain']
                system_transcript = turn['system_transcript']
                user_transcript = turn['transcript']
                updated_belief_state = {}
                for intent in turn['belief_state']:
                    for slot, value in intent['slots']:
                        slot = slot.replace('semi-', '').replace(' ', '-')
                        updated_belief_state[slot] = value
                belief_state_update = {
                    k: v for k, v in updated_belief_state.items()
                    if v != belief_state.get(k)
                }
                belief_state = updated_belief_state
                if system_transcript:
                    turn_rows.append(dst.Turn(
                        text=system_transcript,
                        dialogue=dialogue_id,
                        turn_index=system_turn_index,
                        speaker='bot',
                        domain=turn_domain
                    ))
                user_turn_row = dst.Turn(
                    text=user_transcript,
                    dialogue=dialogue_id,
                    turn_index=user_turn_index,
                    speaker='user',
                    domain=turn_domain
                )
                user_turn_id = user_turn_row.turn_id()
                turn_rows.append(user_turn_row)
                for slot, value in belief_state_update.items():
                    slot_value_rows.append(dst.SlotValue(
                        slot=slot,
                        value=", ".join(value.split('|')),
                        turn_id=user_turn_id,
                        slot_id=slot_map[slot]
                    ))
        turn_table = dst.Turn.of(turn_rows)
        slot_value_table = dst.SlotValue.of(slot_value_rows)
        turn_table().save(f'{folder}/{split}/turn.csv')
        slot_value_table().save(f'{folder}/{split}/slot_value.csv')
        os.chdir(f'{folder}/{split}')
        os.symlink('../slot.csv', f'slot.csv')
        os.symlink('../value_candidate.csv', f'value_candidate.csv')
        os.chdir('../../..')


def update_slot_descriptions(path):
    data = dst.Data(path)
    slots = data.slots
    # descs = {s.lower(): d for s, d in descriptions.items()}
    descs = fixed_descriptions
    for slot in slots:
        slot_name = slot_name_map.get(slot.slot(), slot.slot()).replace('-', ' ')
        slot.description(descs[slot_name])
        slot.slot(slot_name)
    slots().save(f'{path}/slot.csv')

def update_slot_values_slot_names(path):
    data = dst.Data(path)
    slot_values = data.slot_values
    for slot in slot_values:
        slot_name = slot_name_map.get(slot.slot(), slot.slot()).replace('-', ' ')
        slot.slot(slot_name)
    slot_values().save(f'{path}/slot_value.csv')

def update_dontcare(path):
    data = dst.Data(path)
    anys_mask = data.value_candidates.candidate_value == 'dontcare'
    num_anys = sum([int(bool(x)) for x in anys_mask])
    data.value_candidates.candidate_value[anys_mask] = ['any'] * num_anys
    anys_mask = data.slot_values.value == 'dontcare'
    num_anys = sum([int(bool(x)) for x in anys_mask])
    data.slot_values.value[anys_mask] = ['any'] * num_anys
    data.value_candidates().save(f'{path}/value_candidate.csv')
    data.slot_values().save(f'{path}/slot_value.csv')

if __name__ == '__main__':
    folder = 'data/mwoz2.1'
    mwoz_to_table(folder)
    update_slot_descriptions(folder)
    for split in ('train', 'valid', 'test'):
        update_dontcare(f'{folder}/{split}')
        update_slot_values_slot_names(f'{folder}/{split}')