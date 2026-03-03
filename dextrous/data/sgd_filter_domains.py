
import dextrous.dst_data as dst

domains_to_drop = {
    'Trains_', 'Hotels_', 'RideSharing_', 'Travel_', 'Restaurants_',
}

data = dst.Data('data/sgd/train')

turns_to_drop = [
    t for t, d in zip(data.turns.turn_id, data.turns.domain)
    if any(dtd in d for dtd in domains_to_drop)
]

set_of_turns_to_drop = set(turns_to_drop)

slot_values_to_drop = [
    sv for sv, t in zip(data.slot_values.slot_value_id, data.slot_values.turn_id)
    if t in set_of_turns_to_drop
]

slots_to_drop = [
    s for s, d in zip(data.slots.slot_id, data.slots.domain)
    if any(dtd in d for dtd in domains_to_drop)
]

set_of_slots_to_drop = set(slots_to_drop)

candidates_to_drop = [
    c for c, s in zip(data.value_candidates.value_candidate_id, data.value_candidates.slot_id)
    if s in set_of_slots_to_drop
]


del data.turns[turns_to_drop]
del data.slot_values[slot_values_to_drop]
del data.slots[slots_to_drop]
del data.value_candidates[candidates_to_drop]

data.save('data/sgd_filtered/train')