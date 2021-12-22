"""
Configuration package/module for microns-nda.
"""
import datajoint.datajoint_plus as djp
from microns_utils.config_utils import SchemaConfig
from . import adapters
from . import externals

djp.enable_datajoint_flags()

minnie_nda_config = SchemaConfig(
    module_name='minnie_nda',
    schema_name='microns_minnie_nda',
    externals=externals.minnie_nda,
    adapters=adapters.minnie_nda
)

minnie_function_config = SchemaConfig(
    module_name='minnie_function',
    schema_name='microns_minnie_function',
    externals=externals.minnie_function,
    adapters=adapters.minnie_function
)
