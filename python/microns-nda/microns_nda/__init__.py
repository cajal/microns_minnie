import warnings
from microns_utils import config_utils

_repo = 'microns-nda'
_package = _repo

try:
    __version__ = config_utils.get_package_version(repo=_repo, package=_package)
except:
    warnings.warn('Package version not able to be determined.')