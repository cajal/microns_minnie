import datajoint as dj
from datajoint import datajoint_plus as djp

from microns_nda_api import config
schema_obj = config.SCHEMAS.MINNIE_FUNCTION

config.register_adapters(schema_obj, context=locals())
config.register_externals(schema_obj)

schema = dj.schema(schema_obj.value)