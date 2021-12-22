import datajoint.datajoint_plus as djp
from . import minnie_function, minnie_nda

djp.reassign_master_attribute(minnie_nda)
djp.reassign_master_attribute(minnie_function)
