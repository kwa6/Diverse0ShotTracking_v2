import dextrous.dst_data as dst
import dextrous.metrics as metrics
import dextrous.experiment as exp
import dextrous.tracker as dex
import pathlib as pl
import ezpyzy as ez

if __name__ == '__main__':

    models = [
        'ex/LlamaTracker/CelestialHan/1',
        'ex/LlamaTracker/MesmerizingAlderaan/1',
        'ex/LlamaTracker/MysticalChewbacca/1',
        'ex/LlamaTracker/RadiantUtapau/1',
        'ex/LlamaTracker/IconicVulpter/1',
    ]

    models = ['ex/LlamaTracker/hRadiantAurra/1']

    for model in models:
        exp.LlamaExperimentRun(
            experiment=f'{pl.Path(model).parent.name}_preds',
            model_path=model,
            approach='LlamaTracker',
            lora_merge_on_load=False,
            epochs=0,
            calculate_eval_perplexity=False,
            calculate_eval_gen_metrics=True,
            train_data=None,
            eval_data='data/mwoz2.4/valid',
            notifications=True,
            gen_batch_size=1,
            max_sequence_length=128,
            protected_input_length=98,
            eval_with_value_exs=False,
            eval_with_categories=False,
        )