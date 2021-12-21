from microns_utils import version_utils, config_utils
from . import schemas
from .config import SCHEMAS

__version__ = version_utils.check_package_version(
    package='microns-nda-api', 
    check_if_latest=True, 
    check_if_latest_kwargs=dict(
        owner='cajal', 
        repo='microns-nda', 
        source='tag', 
    )
)

check_latest_version_from_github = version_utils.latest_github_version_checker(owner='cajal', repo='microns-nda')

def create_virtual_module(schema_name:str):
    """
    Converts module into a datajoint_plus virtual module.
    """
    module = getattr(schemas, SCHEMAS(schema_name).name.lower())
    module.schema.spawn_missing_classes()
    module.schema.connection.dependencies.load()
    config_utils.add_datajoint_plus(module, virtual=True)
    return module
