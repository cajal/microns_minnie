"""
Configuration package/module for microns-nda.
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

from microns_utils import config_utils
    
config_utils.enable_datajoint_flags()

def register_externals(schema_name:str):
    """
    Registers the external stores for a schema_name in this module.
    """
    external_stores = config_mapping[SCHEMAS(schema_name)]["externals"]
    
    if external_stores is not None:
        config_utils.register_externals(external_stores)


def register_adapters(schema_name:str, context=None):
    """
    Imports the adapters for a schema_name into the global namespace.
    """
    adapter_objects = config_mapping[SCHEMAS(schema_name)]["adapters"]
    
    if adapter_objects is not None:
        config_utils.register_adapters(adapter_objects, context=context)


def create_vm(schema_name:str):
    """
    Creates a virtual module after registering the external stores, and includes the adapter objects in the vm.
    """
    schema = SCHEMAS(schema_name)
    return config_utils.create_vm(schema.value, external_stores=config_mapping[schema]["externals"], adapter_objects=config_mapping[schema]["adapters"])

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