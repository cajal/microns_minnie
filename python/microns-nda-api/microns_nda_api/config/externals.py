
"""
Externals for DataJoint tables.
"""

from pathlib import Path
import datajoint_plus as djp

base_path = Path() / '/mnt' / 'dj-stor01' / 'microns' / 'minnie65' / 'nda'
minnie_nda = {}

function_path = base_path / 'function'
minnie_function = {
    'resp_array': djp.make_store_dict(function_path / 'resp_array'),
    'corr_array': djp.make_store_dict(function_path / 'corr_array'),
}
