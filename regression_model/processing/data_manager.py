import typing
import typing as t
from pathlib import Path

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config


def load_dataset(*, file_name: str) -> pd.DataFrame:
    data = pd.read_csv(Path(DATASET_DIR, file_name))
    data['MSSubClass'] = data['MSSubClass'].astype('O')
    data.rename(columns=config.model_config.variables_to_rename, inplace=True)
    return data


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:

    """Persist the pipeline and overwrites any previous
        saved models. This ensures that when the package is
         published, there is only one trained model that can be called,
         and we know exactly how it was build."""
    save_file_name = f'{config.app_config.pipeline_save_file}{_version}.pkl'
    save_path = Path(TRAINED_MODEL_DIR, save_file_name)
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """To ensure thefile    re is one-to-one mapping between model
     and package versions"""
    do_note_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_note_delete:
            model_file.unlink()


def load_pipeline(*, file_name: str) -> Pipeline:

    file_path = Path(TRAINED_MODEL_DIR, file_name)
    trained_model = joblib.load(file_path)
    return trained_model
