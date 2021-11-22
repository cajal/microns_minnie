from microns_nda_api import config
from . import minnie_nda, minnie_function

config.register_bases(config.SCHEMAS.MINNIE_NDA, minnie_nda)
config.register_bases(config.SCHEMAS.MINNIE_FUNCTION, minnie_function)