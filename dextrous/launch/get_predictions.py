import dextrous.launch.launch as launch
import pathlib as pl

models = [
        # 'ex/LlamaTracker/CelestialHan/1',
        'ex/LlamaTracker/MesmerizingAlderaan/1',
        'ex/LlamaTracker/MysticalChewbacca/1',
        'ex/LlamaTracker/RadiantUtapau/1',
        'ex/LlamaTracker/IconicVulpter/1',
]

for model in models:
    launch.LlamaLaunch(
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
    )