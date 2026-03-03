
import dextrous.old.gptdst5k_format as old
from dextrous.old.gptdst5k_format import DstData
import sys
sys.modules['dst'] = old
sys.modules['dst.data'] = old
sys.modules['dst.data.dst_data'] = old
old.data = old
old.dst_data = old

import dextrous.dst_data as fmt
import ezpyzy as ez


def main():
    dsg5k = DstData.load('data/dsg5k/gptdst5k.pkl')
    slot_rows = []
    value_candidate_rows = []
    turn_rows = []
    slot_value_rows = []
    for dialogue in dsg5k.dialogues:
        domains = dialogue.domains()
        if not domains:
            continue
        domain, = domains
        dialogue_id = ez.uuid()
        for turn_index, turn in enumerate(dialogue.turns):
            text = turn.turn
            speaker = turn.speaker
            turn_id = ez.uuid()
            turn_row = fmt.Turn(
                text=text,
                dialogue=dialogue_id,
                turn_index=turn_index,
                speaker=speaker,
                domain=domain,
                turn_id=turn_id
            )
            turn_rows.append(turn_row)
            for slot, values in turn.slots.items():
                if values is None:
                    continue
                slot_name = slot.name
                slot_value = ', '.join(values)
                slot_description = slot.description
                slot_is_categorical = slot.categorical
                slot_row = fmt.Slot(
                    slot=slot_name,
                    domain=domain,
                    description=slot_description
                )
                slot_rows.append(slot_row)
                slot_id = slot_row.slot_id()
                slot_value_row = fmt.SlotValue(
                    slot=slot_name,
                    value=slot_value,
                    turn_id=turn_id,
                    slot_id=slot_id
                )
                slot_value_rows.append(slot_value_row)
                slot_value_candidates = slot.values
                for slot_value_candidate in slot_value_candidates:
                    slot_value_candidate_row = fmt.ValueCandidate(
                        candidate_value=slot_value_candidate,
                        slot_id=slot_id,
                        is_provided=True
                    )
                    value_candidate_rows.append(slot_value_candidate_row)
    slot_table = fmt.Slot.of(slot_rows)
    slot_value_candidate_table = fmt.ValueCandidate.of(value_candidate_rows)
    turn_table = fmt.Turn.of(turn_rows)
    slot_value_table = fmt.SlotValue.of(slot_value_rows)
    slot_table().save('data/dsg5k/train/slot.csv')
    slot_value_candidate_table().save('data/dsg5k/train/value_candidate.csv')
    turn_table().save('data/dsg5k/train/turn.csv')
    slot_value_table().save('data/dsg5k/train/slot_value.csv')


def get_qa_pairs():
    existing_turn_table = fmt.Turn.of('data/dsg5k/train/turn.csv')
    existing_slot_value_table = fmt.SlotValue.of('data/dsg5k/train/slot_value.csv')
    existing_table = existing_slot_value_table.turn_id << existing_turn_table.turn_id
    signatures_to_ids = {}
    for svid, turn_text, slot_name_text, value_text in zip(
        existing_table.slot_value_id,
        existing_table.text,
        existing_table.slot,
        existing_table.value
    ):
        signature = (turn_text, slot_name_text, value_text)
        signatures_to_ids[signature].append(svid)
    dsg5k = DstData.load('data/dsg5k/gptdst5k.pkl')
    slot_rows = []
    value_candidate_rows = []
    turn_rows = []
    slot_value_rows = []
    for dialogue in dsg5k.dialogues:
        domains = dialogue.domains()
        if not domains:
            continue
        domain, = domains
        dialogue_id = ez.uuid()
        for turn_index, turn in enumerate(dialogue.turns):
            text = turn.turn
            speaker = turn.speaker
            turn_id = ez.uuid()
            turn_row = fmt.Turn(
                text=text,
                dialogue=dialogue_id,
                turn_index=turn_index,
                speaker=speaker,
                domain=domain,
                turn_id=turn_id
            )
            turn_rows.append(turn_row)
            for slot, values in turn.slots.items():
                if values is None:
                    continue
                slot_name = slot.name
                slot_value = ', '.join(values)
                slot_description = slot.description
                slot_is_categorical = slot.categorical
                slot_row = fmt.Slot(
                    slot=slot_name,
                    domain=domain,
                    description=slot_description
                )
                slot_rows.append(slot_row)
                slot_id = slot_row.slot_id()
                slot_value_row = fmt.SlotValue(
                    slot=slot_name,
                    value=slot_value,
                    turn_id=turn_id,
                    slot_id=slot_id
                )
                slot_value_rows.append(slot_value_row)
                slot_value_candidates = slot.values
                for slot_value_candidate in slot_value_candidates:
                    slot_value_candidate_row = fmt.ValueCandidate(
                        candidate_value=slot_value_candidate,
                        slot_id=slot_id,
                        is_provided=True
                    )
                    value_candidate_rows.append(slot_value_candidate_row)
    slot_table = fmt.Slot.of(slot_rows)
    slot_value_candidate_table = fmt.ValueCandidate.of(value_candidate_rows)
    turn_table = fmt.Turn.of(turn_rows)
    slot_value_table = fmt.SlotValue.of(slot_value_rows)
    slot_table().save('data/dsg5k/train/slot.csv')
    slot_value_candidate_table().save('data/dsg5k/train/value_candidate.csv')
    turn_table().save('data/dsg5k/train/turn.csv')
    slot_value_table().save('data/dsg5k/train/slot_value.csv')


if __name__ == '__main__':
    with ez.check('Processing DSG5K data'):
        main()