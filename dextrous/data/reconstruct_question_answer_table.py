import json

import ezpyzy as ez


def main():
    slot_value_table = ez.Table.of('data/dsg5k/train/slot_value.csv')
    turn_table = ez.Table.of('data/dsg5k/train/turn.csv')
    turn_slot_value_table = slot_value_table.turn_id << turn_table.turn_id
    turn_slot_value_to_qa = ez.File('data/dsg5k/turn_slot_value_to_QA.json').load()
    qa_rows = []
    for turn_text, slot_text, value_text, svid in zip(
        turn_slot_value_table.text,
        turn_slot_value_table.slot,
        turn_slot_value_table.value,
        turn_slot_value_table.slot_value_id
    ):
        question, answer = turn_slot_value_to_qa['|'.join((turn_text, slot_text, value_text))]
        qa_rows.append(dict(
            slot=slot_text,
            value=value_text,
            question=question,
            answer=answer,
            slot_value_id=svid
        ))
    qa_table = ez.Table.of(qa_rows)
    qa_table().save('data/dsg5k/train/qa.csv')
    return qa_table


if __name__ == '__main__':
    main()