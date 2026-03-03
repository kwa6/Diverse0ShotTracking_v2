
import ezpyzy as ez
import dextrous.utils as utils
import dataclasses as dc
import dextrous.tracker as dex
import dextrous.dst_data as dst
import random as rng

def interact(tracker:str|dex.LlamaTracker, slots=None, turn=None, **hypers):
    if isinstance(tracker, str):
        tracker = dex.LlamaTracker(dex.LlamaTrackerHyperparameters(
            tracker, lora_merge_on_load=False,
            **hypers
        ))
    slots = slots or {}
    turn = turn or ''
    while (text:=input('>>> ')) != 'exit':
        if text.startswith('{'):
            slots = ez.JSON.deserialize(text)
        else:
            turn = text
        data = dst.Data.of(turn, slots)
        tracker.predict(data)
        for dialogue in data.examples():
            print('\n\n')
            for text, predicted, actual in dialogue:
                print(text)
                if predicted:
                    print('   P: ', ", ".join(f"{k}: {v}" for k, v in predicted.items()))
                if actual:
                    print('   A: ', ", ".join(f"{k}: {v}" for k, v in actual.items()))



if __name__ == '__main__':
    ...
    # utils.transfer_remote2remote(
    #     from_machine='h100',
    #     from_path='/local/scratch/jdfinch/dextrous/ex/LlamaTracker/DazzlingDengar',
    #     to_machine='tebuna',
    #     is_folder=True
    # )
    # utils.download('h100', 'ex/LlamaTracker/SupernovaOssus/7')
    # utils.download('h100', 'ex/LlamaTracker/SupernovaOssus/experiment.csv')
    # interact('ex/LlamaTracker/SupernovaOssus/7', dict(
    #     quantity='The number of items to buy (e.g. 3, 5)'
    # ), repetition_penalty=1.0, num_beams=3)
    utils.download('tebuna', 'results/dsg5k_induced.csv')
