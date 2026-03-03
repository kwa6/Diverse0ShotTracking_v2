
import ezpyzy as ez
import dextrous.dst_data as dst


def eval_str_postproc(string):
    if string is None:
        return string
    string = string.lower()
    string = string.replace('_', ' ').strip()
    if string == '?' or string == 'none':
        string = None
    return string


# def joint_goal_accuracy(data: dst.Data) -> float:
#     preds_labels = data.predictions.slot_value_id << data.slot_values.slot_value_id
#     preds_labels = preds_labels.turn_id << data.turns.turn_id
#     preds_labels.dialogue = ez.Column(
#         data.turns[turn_id].dialogue() for turn_id in preds_labels.turn_id
#     )
#     preds_by_dial = preds_labels().group(preds_labels.dialogue)
#     dialogues = data.turns().group(data.turns.dialogue)
#     count_correct = 0
#     count_total = 0
#     for dialogue, preds_labels in preds_by_dial.items():
#         turns = dialogues[dialogue]
#         turns().sort(turns.turn_index)
#         for pred_label in preds_labels:
#             history_slots = preds_labels[preds_labels.turn_index <= pred_label.turn_index()]
#             predictions = [eval_str_postproc(x) for x in history_slots.prediction]
#             values = list(history_slots.value)
#             correct = all(pred == value for pred, value in zip(predictions, values))
#             count_correct += int(correct)
#             count_total += 1
#     return count_correct / count_total

def joint_goal_accuracy(data: dst.Data, speakers=None, domains=None) -> float:
    count_correct, count_total = 0, 0
    for dialogue in data.examples(n=None, accumulate_states=True):
        for text, speaker, domain, predicted, actual, _, _ in dialogue:
            if speakers and speaker not in speakers:
                continue
            if domains and domain not in domains:
                continue
            actual_slot_values = {(k, eval_str_postproc(v)) for k,v in actual.items()}
            predicted_slot_values = {(k, eval_str_postproc(v)) for k,v in predicted.items()}
            count_correct += int(actual_slot_values == predicted_slot_values)
            count_total += 1
    return count_correct / count_total


def slot_accuracy(data: dst.Data) -> float:
    preds_labels = data.predictions.slot_value_id << data.slot_values.slot_value_id
    preds_labels = preds_labels.turn_id << data.turns.turn_id
    preds_labels.dialogue = ez.Column(
        data.turns[turn_id].dialogue() for turn_id in preds_labels.turn_id
    )
    preds_by_dial = preds_labels().group(preds_labels.dialogue)
    dialogues = data.turns().group(data.turns.dialogue)
    count_correct = 0
    count_total = 0
    for dialogue, preds_labels in preds_by_dial.items():
        turns = dialogues[dialogue]
        turns().sort(turns.turn_index)
        for pred_label in preds_labels:
            history_slots = preds_labels[preds_labels.turn_index <= pred_label.turn_index()]
            predictions = [eval_str_postproc(x) for x in history_slots.prediction]
            values = list(history_slots.value)
            for prediction, value in zip(predictions, values):
                correct = prediction == value
                count_correct += int(correct)
                count_total += 1
    return count_correct / count_total

def slot_update_accuracy(data: dst.Data) -> float:
    preds_labels = data.predictions.slot_value_id << data.slot_values.slot_value_id
    predictions = [eval_str_postproc(x) for x in preds_labels.prediction]
    values = list(preds_labels.value)
    matches = [pred == value for pred, value in zip(predictions, values)]
    return sum([int(i) for i in matches]) / len(matches)



if __name__ == '__main__':
    pass