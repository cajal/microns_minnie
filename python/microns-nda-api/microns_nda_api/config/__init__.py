"""
Configuration package/module for microns-nda.
"""
import inspect
import traceback
from enum import Enum
from . import adapters
from . import externals
from . import bases
try:
    import datajoint as dj
except:
    traceback.print_exc()
    raise ImportError('DataJoint package not found.')
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


def register_bases(schema_name:str, module):
    """
    Maps base classes to DataJoint tables.
    """
    bases = config_mapping[SCHEMAS(schema_name)]["bases"]

    if bases is not None:
        for base in bases:
            config_utils.register_bases(base, module)
        return module


def create_vm(schema_name:str):
    """
    Creates a virtual module after registering the external stores, adapter objects, DatajointPlus and base classes.
    """
    schema = SCHEMAS(schema_name)
    vm = config_utils._create_vm(schema.value, external_stores=config_mapping[schema]["externals"], adapter_objects=config_mapping[schema]["adapters"])
    config_utils.add_datajoint_plus(vm)
    register_bases(schema_name, vm)
    return vm


class SCHEMAS(Enum):
    MINNIE_NDA = 'microns_minnie_nda'
    MINNIE_FUNCTION = 'microns_minnie_function'


config_mapping = {
    SCHEMAS.MINNIE_NDA: {
        "externals" : None,
        "adapters" : None,
        "bases" : None
    },

    SCHEMAS.MINNIE_FUNCTION: {
        "externals": None,
        "adapters": None,
        "bases" : None
    }
}
