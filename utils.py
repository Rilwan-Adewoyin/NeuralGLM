from typing import Dict, Any, Optional
import yaml
from pytorch_lightning.utilities.cloud_io import get_filesystem
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities import _OMEGACONF_AVAILABLE, AttributeDict
if _OMEGACONF_AVAILABLE:
    from omegaconf import OmegaConf
    from omegaconf.dictconfig import DictConfig
    from omegaconf.errors import UnsupportedValueType, ValidationError
import pytorch_lightning as pl
import regex as re

def load_hparams_from_yaml(config_yaml: str, use_omegaconf: bool = True) -> Dict[str, Any]:
    """Load hparams from a file.

        Args:
            config_yaml: Path to config yaml file
            use_omegaconf: If omegaconf is available and ``use_omegaconf=True``,
                the hparams will be converted to ``DictConfig`` if possible.

    >>> hparams = Namespace(batch_size=32, learning_rate=0.001, data_root='./any/path/here')
    >>> path_yaml = './testing-hparams.yaml'
    >>> save_hparams_to_yaml(path_yaml, hparams)
    >>> hparams_new = load_hparams_from_yaml(path_yaml)
    >>> vars(hparams) == hparams_new
    True
    >>> os.remove(path_yaml)
    """
    fs = get_filesystem(config_yaml)
    if not fs.exists(config_yaml):
        rank_zero_warn(f"Missing Tags: {config_yaml}.", category=RuntimeWarning)
        return {}

    with fs.open(config_yaml, "r") as fp:
        hparams = yaml.unsafe_load(fp)

    if _OMEGACONF_AVAILABLE:
        if use_omegaconf:
            try:
                return OmegaConf.create(hparams)
            except (UnsupportedValueType, ValidationError):
                pass
    return hparams

pl.core.saving.load_hparams_from_yaml = load_hparams_from_yaml


def _format_checkpoint_name(
    cls,
    filename: Optional[str],
    metrics, #Dict[str, _METRIC],
    prefix: str = "",
    auto_insert_metric_name: bool = True,
    ) -> str:
    if not filename:
        # filename is not set, use default name
        filename = "{epoch}" + cls.CHECKPOINT_JOIN_CHAR + "{step}"

    # check and parse user passed keys in the string
    groups = re.findall(r"(\{.*?)[:\}]", filename)
    if len(groups) >= 0:
        for group in groups:
            name = group[1:]

            if auto_insert_metric_name:
                filename = filename.replace(group, name.replace("/","_") + "={" + name)

            if name not in metrics:
                metrics[name] = 0
        filename = filename.format(**metrics)

    if prefix:
        filename = cls.CHECKPOINT_JOIN_CHAR.join([prefix, filename])

    return filename
