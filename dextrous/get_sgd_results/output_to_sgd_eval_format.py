# Official SGD Evaluation script: https://github.com/google-research/google-research/blob/master/schema_guided_dst/evaluate.py

import dextrous.dst_data as dst
import json
import os
from tqdm import tqdm

def clear_groundtruth(data):
    # clear all ground truth values
    for dialogue in transformed_data:
        for turn in dialogue['turns']:
            for turn_frame in turn['frames']:
                if 'state' in turn_frame:
                    turn_frame['state'] = {'slot_values': {}}
                    turn_frame['slots'] = []
                    turn_frame['actions'] = []
                else:
                    turn_frame['actions'] = []
                    turn_frame['slots'] = []
                    turn_frame['service_results'] = []
                    turn_frame['service_call'] = {}


all_predicted_slot_names = set()

if __name__ == '__main__':

    data_split_original = 'test'
    data_split = 'unseen'
    modelname = 'LlamaTracker/MajesticZeb/0'
    # modelname = 'LlamaTracker/LegendaryTusken/0'


    outputpath = f'ex/{modelname}'
    data = dst.Data(data_path=f'data/sgd/{data_split}', prediction_path=f'{outputpath}/predictions.csv')

    if not os.path.exists(f'{outputpath}/formatted_results'):
        os.mkdir(f'{outputpath}/formatted_results')
    if not os.path.exists(f'{outputpath}/formatted_results/{data_split}'):
        os.mkdir(f'{outputpath}/formatted_results/{data_split}')
    
    for file in tqdm(os.listdir(f'data/original_sgd/{data_split_original}'), desc='Formatting'):
        if file.startswith('dialogues'):
            original_data = json.load(open(f'data/original_sgd/{data_split_original}/{file}'))
            transformed_data = json.load(open(f'data/original_sgd/{data_split_original}/{file}'))
            clear_groundtruth(transformed_data)
            for dialogue_idx, dialogue in enumerate(original_data):
                dialogue_id = dialogue['dialogue_id']
                predicted_state = {}
                for turn_idx, turn in enumerate(dialogue['turns']):
                    for frame_idx, turn_frame in enumerate(turn['frames']):
                        if 'state' in turn_frame:
                            turn_slot_service = turn_frame['service']
                            turn_slot_state = turn_frame['state']
                            turn_slot_values = turn_slot_state['slot_values']
                            predicted_turn_id = f'{turn_slot_service}{dialogue_id}-{turn_idx+1}'
                            predictions_for_turn = data.predictions[data.predictions.turn_id == predicted_turn_id]
                            if len(predictions_for_turn) > 0:
                                predicted_for_turn = {
                                    '_'.join(s.split(' ')[1:]): [v] for s,v in zip(predictions_for_turn.slot, predictions_for_turn.value)
                                    if v not in {'n/a', None}
                                }
                                predicted_state.update(predicted_for_turn)
                                transformed_data[dialogue_idx]['turns'][turn_idx]['frames'][frame_idx]['state']['slot_values'] = {
                                    k:v for k,v in predicted_state.items()
                                }
                                all_predicted_slot_names.update(predicted_state.keys())
    
            json.dump(transformed_data, open(f'{outputpath}/formatted_results/{data_split}/{file}', 'w'), indent=2)



