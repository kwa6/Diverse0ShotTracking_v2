
import ezpyzy as ez
import dataclasses as dc
import os
import pathlib as pl
import random as rng

import dextrous.dst_data as dst


speaker_map = dict(USER='user',SYSTEM='bot')

def sgd_to_table(folder, domainless=False):
    folder = pl.Path(folder)
    collected_slots = {}
    collected_value_candidates = []
    all_slot_values = {}
    slot_name_map = {}
    for split in ["train", "dev", "test"]:
        split_folder = folder / split
        schema = ez.File(split_folder / "schema.json").load()
        for service in schema:
            domain = service["service_name"]
            for slot in service["slots"]:
                slot_name = slot["name"].replace('_', ' ')
                slot_description = slot["description"]
                slot_is_categorical = slot["is_categorical"]
                slot_possible_values = slot["possible_values"]
                if slot_is_categorical:
                    slot_description = f"{slot_description} [{', '.join(slot_possible_values)}]"
                domain_slot_name = f"{domain.split('_')[0].rstrip('s')} {slot_name}".replace('Buse ', 'Bus ')
                slot_name_map[domain, slot_name] = domain_slot_name
                slot_obj = dst.Slot(
                    slot=domain_slot_name,
                    domain=domain,
                    description=slot_description,
                )
                collected_slots[domain, slot_name] = slot_obj
    for split in ["train", "dev", "test"]:
        split_folder = folder / split
        dialogues = ez.File(split_folder / f"{split_folder.name}.json").load()
        if split == 'dev':
            split = 'valid'
        collected_turns_by_domain = {}
        collected_slot_values_by_domain = {}
        for dialogue in dialogues:
            original_dialogue_id = dialogue["dialogue_id"]
            for dialogue_domain in ([''] if domainless else dialogue["services"]):
                dialogue_id = f'{dialogue_domain}{original_dialogue_id}'
                previous_state = {}
                for i, turn in enumerate(dialogue["turns"]):
                    turn_id = f"{dialogue_id}-{i+1}"
                    turn_text = turn["utterance"]
                    turn_speaker = speaker_map[turn["speaker"]]
                    frames = turn["frames"]
                    current_state = dict(previous_state)
                    service_is_active = False
                    for frame in frames:
                        domain = frame["service"]
                        if dialogue_domain not in domain or turn_speaker != 'user':
                            continue
                        service_is_active = True
                        state = frame["state"]
                        slot_values = state["slot_values"]
                        for slot_name, values_list in slot_values.items():
                            slot_name = slot_name.replace('_', ' ')
                            value = values_list[0]
                            if value == 'dontcare':
                                value = 'any'
                            current_state[domain, slot_name] = value
                    state_update = {
                        (d, s): v for (d, s), v in current_state.items() # noqa
                        if v != previous_state.get((d, s))
                    }
                    previous_state = current_state
                    turn_obj = dst.Turn(
                        text=turn_text,
                        dialogue=dialogue_id,
                        turn_index=i + 1,
                        speaker=turn_speaker,
                        domain=dialogue_domain if service_is_active else '',
                        turn_id=turn_id
                    )
                    collected_turns_by_domain.setdefault(dialogue_domain, []).append(turn_obj)
                    for (domain, slot), values in state_update.items():
                        value = values
                        slot_value_obj = dst.SlotValue(
                            slot=slot_name_map[domain, slot],
                            value=value,
                            turn_id=turn_obj.turn_id(),
                            slot_id=collected_slots[domain, slot].slot_id()
                        )
                        collected_slot_values_by_domain.setdefault(dialogue_domain, []).append(slot_value_obj)
                        all_slot_values.setdefault((domain, slot), set()).add(value)
        if split == 'test' and not domainless:
            for domain in ['Alarm_1', 'Messaging_1', 'Payment_1', 'Trains_1']:
                turn_table = dst.Turn.of(collected_turns_by_domain[domain])
                slot_value_table = dst.SlotValue.of(collected_slot_values_by_domain[domain])
                turn_table().save(f"data/sgd/test_{domain}/turn.csv")
                slot_value_table().save(f"data/sgd/test_{domain}/slot_value.csv")
        else:
            turn_table = dst.Turn.of([x for xs in collected_turns_by_domain.values() for x in xs])
            slot_value_table = dst.SlotValue.of([x for xs in collected_slot_values_by_domain.values() for x in xs])
            if domainless:
                turn_table().save(f"data/sgd_wo_domains/{split}/turn.csv")
                slot_value_table().save(f"data/sgd_wo_domains/{split}/slot_value.csv")
            else:
                turn_table().save(f"data/sgd/{split}/turn.csv")
                slot_value_table().save(f"data/sgd/{split}/slot_value.csv")
    for (domain, slot), values in all_slot_values.items():
        for candidate_value in values:
            candidate_obj = dst.ValueCandidate(
                candidate_value=candidate_value,
                slot_id=collected_slots[domain, slot].slot_id(),
                is_provided=False,
            )
            collected_value_candidates.append(candidate_obj)
    descriptions_with_examples = []
    reverse_slot_name_map = {v: k for k, v in slot_name_map.items()}
    slot_table = dst.Slot.of(list(collected_slots.values()))
    for domain, slot, description, in zip(
        slot_table.domain, slot_table.slot, slot_table.description,
    ):
        original_slot_name = reverse_slot_name_map.get(slot)[1]
        examples = all_slot_values.get((domain, original_slot_name), set())
        four_examples = rng.sample(list(examples), min(4, len(examples)))
        if not description.endswith(']') and four_examples:
            descriptions_with_examples.append(
                f"{description} (e.g. {', '.join(four_examples)})"
            )
        else:
            descriptions_with_examples.append(description)
    slot_table.description = ez.Column(descriptions_with_examples)
    value_candidate_table = dst.ValueCandidate.of(collected_value_candidates)
    slots_with_values = [d.endswith(']') or d.endswith(')') for d in slot_table.description]
    slot_table = ~slot_table[slots_with_values]
    slot_ids_with_values = set(slot_table.slot_id)
    value_cands_with_values = [s in slot_ids_with_values for s in value_candidate_table.slot_id]
    value_candidate_table = ~value_candidate_table[value_cands_with_values]
    if domainless:
        for split in ('train', 'valid', 'test'):
            slot_table().save(f"data/sgd_wo_domains/{split}/slot.csv")
            value_candidate_table().save(f"data/sgd_wo_domains/{split}/value_candidate.csv")
    else:
        for split in ('train', 'valid', 'test'):
            if split == 'test':
                for domain in ['Alarm_1', 'Messaging_1', 'Payment_1', 'Trains_1']:
                    slot_table().save(f"data/sgd/test_{domain}/slot.csv")
                    value_candidate_table().save(f"data/sgd/test_{domain}/value_candidate.csv")
            else:
                slot_table().save(f"data/sgd/{split}/slot.csv")
                value_candidate_table().save(f"data/sgd/{split}/value_candidate.csv")


if __name__ == '__main__':
    sgd_to_table('data/sgd', domainless=True)




