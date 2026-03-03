
import ezpyzy as ez
import pathlib as pl
import shutil

blacklist = ez.File('results/blacklist.txt').load().split('\n')
whitelist = set(ez.File('results/whitelist.txt').load().split('\n'))

assert set(blacklist) & set(whitelist) == set()

prefix = '/local/scratch/jdfinch/dextrous'


def remove_blacklisted():
    for blacklisted in blacklist:
        path = pl.Path(f'{prefix}/{blacklisted}')
        if path.exists():
            for file in path.iterdir():
                if file.is_file() and file.suffix in ('.bin', '.safetensors') and 'model' in file.name:
                    print(f'Removing {file} ')
                    # file.unlink()


def remove_all_not_whitelisted():
    for approach in (x for x in (pl.Path(prefix) / 'ex').iterdir() if x.is_dir()):
        for experiment in (x for x in approach.iterdir() if x.is_dir()):
            for iteration in (x for x in experiment.iterdir() if x.is_dir()):
                path = f'{approach.name}/{experiment.name}/{iteration.name}'
                if f'ex/{path}' not in whitelist:
                    for file in iteration.iterdir():
                        if file.is_file() and file.suffix in ('.bin', '.safetensors') and 'model' in file.name:
                            print(f'Removing {file} ')
                            # file.unlink(missing_ok=True)


def validate_whitelist():
    print(f"Validating whitelist...")
    print('\n'.join(whitelist), '\n\n')
    for whitelisted in whitelist:
        path = pl.Path(f'{prefix}/{whitelisted}')
        found_model = False
        for file in path.iterdir():
            if file.is_file() and file.suffix in ('.bin', '.safetensors') and 'model' in file.name:
                found_model = True
                break
        if not found_model:
            print(f'Warning: {path} does not contain a model file.')


def oops():
    # Define the source and destination directories
    source_base = pl.Path('/local/scratch/jdfinch/dextrous/ex/T5Tracker/ex/LlamaTracker')
    destination_base = pl.Path('/local/scratch/jdfinch/dextrous/ex/LlamaTracker')

    # Iterate over all directories in the source directory
    for folder in source_base.iterdir():
        if folder.is_dir():  # Check if the current item is a directory
            destination_folder = destination_base / folder.name

            if not destination_folder.exists():
                # Move the directory if it doesn't already exist in the destination
                shutil.move(str(folder), str(destination_folder))
                # print(f'Moved {folder} to {destination_folder}')
            else:
                # Print a warning if the directory already exists
                print(f'Warning: {destination_folder} already exists. Skipping.')


if __name__ == '__main__':
    # remove_all_not_whitelisted()
    remove_all_not_whitelisted()
