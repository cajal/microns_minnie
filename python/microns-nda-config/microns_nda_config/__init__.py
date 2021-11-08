"""
Configuration package/module for microns-coregistration.
"""

from . import adapters
from . import externals

import traceback

try:
    import datajoint as dj
except:
    traceback.print_exc()
    raise ImportError('DataJoint package not found.')

from enum import Enum

# Also important to run the dj flags at SOME point during normal datajoint initialization, probably don't want to have this in every config though?

# TODO: place in microns-utils
def enable_datajoint_flags(enable_python_native_blobs=True):
    """
    Enable experimental datajoint features
    
    These flags are required by 0.12.0+ (for now).
    """
    dj.config['enable_python_native_blobs'] = enable_python_native_blobs
    dj.errors._switch_filepath_types(True)
    dj.errors._switch_adapted_types(True)
    
enable_datajoint_flags()


def register_externals(schema_name:str=None, stores_config:dict=None):
    """
    TODO: Check logic/ add validation if passing both stores_config and schema_name

    Registers the external stores for a schema_name in this module.
    """
    if schema_name is None and stores_config is None:
        raise Exception("Either schema_name or stores_config must be set")

    if schema_name is not None: 
        stores_config = externals_mapping[schema_name]

    if 'stores' not in dj.config:
        dj.config['stores'] = stores_config
    else:
        dj.config['stores'].update(stores_config)


def register_adapters(schema_name:str=None, adapter_objects:dict=None, context=None):
    """
    TODO: Check logic/ add validation if passing both stores_config and schema_name

    Imports the adapters for a schema_name into the global namespace.
    
    This function is probably not necessary, but standardization is nice.
    """
    if schema_name is None and adapter_objects is None:
        raise Exception("Either schema_name or adapter_objects must be set")

    if schema_name is not None: 
        adapter_objects = adapters_mapping[schema_name]
    
    if context is None:
        # if context is missing, use the calling namespace
        import inspect
        frame = inspect.currentframe().f_back
        context = frame.f_locals
        del frame
    
    for name, adapter in adapter_objects.items():
        context[name] = adapter

# IMPORTANT: You can organize several schemas' config files (or folders) however you want as long as
# the correct schema configurations are plugged in for the enum variations (obviously).

class SCHEMAS(Enum):
    MINNIE_NDA = 'microns_minnie_nda'
    MINNIE_FUNCTION = 'microns_minnie_function'


config_mapping = {
    SCHEMAS.MINNIE_NDA: {
        "externals": None,
        "adapters": None
    },

    SCHEMAS.MINNIE_FUNCTION: {
        "externals": None,
        "adapters": None
    }
}

adapters_mapping = {SCHEMA.name: config_mapping[SCHEMA]['adapters'] for SCHEMA in SCHEMAS}

externals_mapping = {SCHEMA.name: config_mapping[SCHEMA]['externals'] for SCHEMA in SCHEMAS}

def create_vm(schema_name:str):
    """
    Creates a virtual module after registering the external stores, and includes the adapter objects in the vm.
    """
    
    if externals is not None:
        register_externals(schema_name)
    
    return dj.create_virtual_module(schema_name, schema_name, add_objects=adapters_mapping[schema_name])