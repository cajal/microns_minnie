import datajoint as dj
from datajoint import datajoint_plus as djp

import numpy as np
import pandas as pd

from microns_nda_api import config
schema_obj = config.SCHEMAS.MINNIE_NDA

config.register_adapters(schema_obj, context=locals())
config.register_externals(schema_obj)

schema = dj.schema(schema_obj.value)
# schema.spawn_missing_classes()


@schema
class Animal(djp.Lookup):
    definition = """
    animal_id            : int                          # id number 
    """

    contents = [[17797]]


@schema
class Scan(djp.Lookup):
    definition = """
    # scans from pipeline_experiment.Scan
    -> Animal
    scan_session         : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    """

    class Experiment(djp.Part):
        definition = """
        # info from pipeline_experiment.Scan
        ->master
        ---
        lens                 : char(4)                      # objective lens
        brain_area           : char(12)                     # short name for cortical area
        aim="2pScan"         : varchar(36)                  # short name for the experiment scan
        filename             : varchar(255)                 # file base name
        depth=0              : int                          # manual depth measurement
        scan_notes           : varchar(4095)                # free-notes
        site_number=0        : tinyint                      # site number
        software             : varchar(20)                  # name of the software
        version              : char(10)                     # version
        scan_ts=CURRENT_TIMESTAMP : timestamp                    # don't edit
        """

    class Reso(djp.Part):
        definition = """
        # info from pipeline_reso.ScanInfo
        ->master
        ---
        pipe_version         : smallint                     # 
        nfields              : tinyint                      # number of slices
        nchannels            : tinyint                      # number of recorded channels
        nframes              : int                          # number of recorded frames
        nframes_requested    : int                          # number of frames (from header)
        px_height            : smallint                     # lines per frame
        px_width             : smallint                     # pixels per line
        um_height            : float                        # height in microns
        um_width             : float                        # width in microns
        x                    : float                        # (um) center of scan in the motor coordinate system
        y                    : float                        # (um) center of scan in the motor coordinate system
        fps                  : float                        # (Hz) frames per second
        zoom                 : decimal(5,2)                 # zoom factor
        bidirectional        : tinyint                      # true = bidirectional scanning
        usecs_per_line       : float                        # microseconds per scan line
        fill_fraction        : float                        # raster scan temporal fill fraction (see scanimage)
        valid_depth=0        : tinyint                      # whether depth has been manually check
        """

    class Meso(djp.Part):
        definition = """
        # info from pipeline_meso.ScanInfo
        ->master
        ---
        pipe_version         : smallint                     # 
        nfields              : tinyint                      # number of fields
        nchannels            : tinyint                      # number of channels
        nframes              : int                          # number of recorded frames
        nframes_requested    : int                          # number of requested frames (from header)
        x                    : float                        # (um) ScanImage's 0 point in the motor coordinate system
        y                    : float                        # (um) ScanImage's 0 point in the motor coordinate system
        fps                  : float                        # (Hz) frames per second
        bidirectional        : tinyint                      # true = bidirectional scanning
        usecs_per_line       : float                        # microseconds per scan line
        fill_fraction        : float                        # raster scan temporal fill fraction (see scanimage)
        nrois                : tinyint                      # number of ROIs (see scanimage)
        valid_depth=0        : tinyint                      # whether depth has been manually check
        """

    
    @classmethod
    def fill(cls):
        experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
        reso = dj.create_virtual_module('reso', 'pipeline_reso')
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        cls.insert(experiment.Scan.proj(..., scan_session='session') & Animal, ignore_extra_fields=True, skip_duplicates=True)
        cls.Experiment.insert(experiment.Scan.proj(..., scan_session='session') & Animal, skip_duplicates=True)
        cls.Reso.insert(reso.ScanInfo.proj(..., scan_session='session') & Animal, skip_duplicates=True)
        cls.Meso.insert(meso.ScanInfo.proj(..., scan_session='session') & Animal, skip_duplicates=True)
        

@schema
class Stack(djp.Lookup):    
    definition = f"""
    # stacks from pipeline_experiment.Stack
    -> Animal
    stack_session        : smallint                     # session index for the mouse
    stack_idx            : smallint                     # id of the stack
    """

    class Experiment(djp.Part):
        definition = """
        # info from pipeline_experiment.Stack
        ->master
        ---
        lens                 : char(4)                      # objective lens
        brain_area           : char(12)                     # short name for cortical area
        aim                  : varchar(36)                  # short name for the purpose of the scan
        software             : varchar(20)                  # name of the software
        version              : char(10)                     # version
        surf_depth=0         : smallint                     # (um) depth of the surface of the cortex
        top_depth            : smallint                     # (um) depth at top of the stack
        bottom_depth         : smallint                     # (um) depth at bottom of stack
        stack_notes          : varchar(4095)                # free notes
        stack_ts=CURRENT_TIMESTAMP : timestamp                    # don't edit
        """

    class Corrected(djp.Part):
        definition = """
        # info from pipeline_stack.StackInfo * pipeline_stack.CorrectedStack
        ->master
        volume_id            : tinyint                      # id of this volume
        ---
        nrois                : tinyint                      # number of ROIs
        nchannels            : tinyint                      # number of channels
        fill_fraction        : float                        # raster scan temporal fill fraction (see scanimage)
        z                    : float                        # (um) center of volume in the motor coordinate system (cortex is at 0)
        y                    : float                        # (um) center of volume in the motor coordinate system
        x                    : float                        # (um) center of volume in the motor coordinate system
        px_depth             : smallint                     # number of slices
        px_height            : smallint                     # lines per frame
        px_width             : smallint                     # pixels per line
        um_depth             : float                        # depth in microns
        um_height            : float                        # height in microns
        um_width             : float                        # width in microns
        surf_z               : float                        # (um) depth of first slice - half a z step (cortex is at z=0)
        """

    
    @classmethod
    def fill(cls):
        experiment = dj.create_virtual_module('experiment', 'pipeline_experiment')
        stack = dj.create_virtual_module('stack', 'pipeline_stack')
        cls.insert(experiment.Stack.proj(..., stack_session='session') & Animal, ignore_extra_fields=True, skip_duplicates=True)
        cls.Experiment.insert(experiment.Stack.proj(..., stack_session='session') & Animal, skip_duplicates=True)
        cls.Corrected.insert((stack.StackInfo * stack.CorrectedStack).proj(..., stack_session='session') & Animal, skip_duplicates=True)


@schema
class Field(djp.Lookup):
    definition = """
    # Scan fields
    -> Scan
    field                : smallint                     # Field Number
    """
    class Reso(djp.Part):
        definition = """
        # information from pipeline_reso.ScanInfo.Field
        -> master
        -> Scan.Reso
        field                : tinyint                      # slice in scan
        ---
        pipe_version         : smallint                     # 
        z                    : float                        # (um) absolute depth with respect to the surface of the cortex
        delay_image          : longblob                     # 
        """
    
    class Meso(djp.Part):
        definition = """
        # information from pipeline_meso.ScanInfo.Field
        -> master
        -> Scan.Meso
        field                : tinyint                      # 
        ---
        pipe_version         : smallint                     # 
        px_height            : smallint                     # height in pixels
        px_width             : smallint                     # width in pixels
        um_height            : float                        # height in microns
        um_width             : float                        # width in microns
        x                    : float                        # (um) center of field in the motor coordinate system
        y                    : float                        # (um) center of field in the motor coordinate system
        z                    : float                        # (um) absolute depth with respect to the surface of the cortex
        delay_image          : longblob                     # (ms) delay between the start of the scan and pixels in this field
        roi                  : tinyint                      # ROI to which this field belongs
        """
    
    
    @classmethod
    def fill(cls):
        reso = dj.create_virtual_module('reso', 'pipeline_reso')
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        for pipe, part in zip([reso, meso], [cls.Reso, cls.Meso]):
            cls.insert(pipe.ScanInfo.Field.proj(scan_session='session') & Scan, ignore_extra_fields=True, skip_duplicates=True)
            part.insert(pipe.ScanInfo.Field.proj(..., scan_session='session') & Scan, skip_duplicates=True)


@schema
class SummaryImages(djp.Lookup):
    definition = """
    # Summary images from pipeline_meso/reso.SummaryImages
    -> Field
    scan_channel              : tinyint                      #
    """
    
    class Average(djp.Part):
        definition = """
        # mean of each pixel across time
        -> master
        ---
        average_image         : longblob                     # 
        """
    
    class Correlation(djp.Part):
        definition = """
        # average temporal correlation between each pixel and its eight neighbors
        -> master
        ---
        correlation_image      : longblob                     # 
        """
    
    class L6Norm(djp.Part):
        definition = """
        # l6-norm of each pixel across time
        -> master
        ---
        l6norm_image           : longblob                     # 
        """
    
    @classmethod
    def fill(cls):
        reso = dj.create_virtual_module('reso', 'pipeline_reso')
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        for pipe in [reso, meso]:
            cls.insert(pipe.SummaryImages.proj(scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
            cls.Average.insert(pipe.SummaryImages.Average.proj(..., scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
            cls.Correlation.insert(pipe.SummaryImages.Correlation.proj(..., scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
            cls.L6Norm.insert(pipe.SummaryImages.L6Norm.proj(..., scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
    
    class All:
        def __new__(cls):
            return SummaryImages * SummaryImages.Average * SummaryImages.Correlation * SummaryImages.L6Norm  


@schema
class SegmentationMethod(djp.Lookup):
    definition = """
    # methods for mask extraction for multi-photon scans 
    segmentation_method  : tinyint                      # 
    ---
    name                 : varchar(16)                  # 
    details              : varchar(255)                 #
    """

    contents = [
        [6,	'nmf-test',	'test nmf with diff parameters']
    ]

    
@schema
class Segmentation(djp.Lookup):
    definition = """
    # segmentation results
    -> Field
    -> SegmentationMethod
    ---
    scan_channel              : tinyint                      #
    segmentation_time=CURRENT_TIMESTAMP : timestamp
    """
    
    class CNMFCaImAn(djp.Part):
        definition = """
        # CNMF segmentation params from pipeline_meso/reso.Segmentation.CNMF
        -> master
        ---
        params               : varchar(1024)                # parameters send to CNMF as JSON array
        """
    

    class Mask(djp.Part):
        definition = """
        # masks resulting from Segmentation
        -> master
        mask_id              : smallint                     #
        ---
        pixels               : longblob                     # indices into the image in column major (Fortran) order
        weights              : longblob                     # weights of the mask at the indices above
        """
    
    
    @classmethod
    def fill(cls):
        reso = dj.create_virtual_module('reso', 'pipeline_reso')
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        for segmentation_method in [6]:
            for pipe in [reso, meso]:
                cls.insert(pipe.Segmentation.proj(..., scan_session='session', scan_channel='channel') & Field & {'segmentation_method': segmentation_method}, skip_duplicates=True, ignore_extra_fields=True)
                cls.CNMFCaImAn.insert(pipe.Segmentation.CNMF.proj(..., scan_session='session') & Field & {'segmentation_method': segmentation_method}, skip_duplicates=True, ignore_extra_fields=True)
                cls.Mask.insert(pipe.Segmentation.Mask.proj(..., scan_session='session') & Field & {'segmentation_method': segmentation_method}, skip_duplicates=True, ignore_extra_fields=True)
            


@schema
class MaskClassificationMethod(djp.Lookup):
    definition = """
    # methods to classify extracted masks from pipeline_shared.ClassificationMethod
    mask_classification_method : tinyint                      # 
    ---
    name                 : varchar(16)                  # 
    details              : varchar(255)                 # 
    """
    
    contents = [
        [2, 'cnn-caiman', 'classification made by a trained convolutional network']
    ]


@schema
class MaskClassification(djp.Lookup):
    definition = """
    # mask classification from pipeline_meso/reso.MaskClassification
    -> Segmentation
    -> MaskClassificationMethod
    ---
    classif_time=CURRENT_TIMESTAMP : timestamp                    # automatic
    """
    
    class Type(djp.Part):
        definition = """
        # mask classification results
        -> master
        -> Segmentation.Mask
        --- 
        mask_type            : varchar(16)                  # mask classification
        """
    
    @classmethod
    def fill(cls):
        reso = dj.create_virtual_module('reso', 'pipeline_reso')
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        for mask_classification_method in [2]:
            for pipe in [reso, meso]:
                cls.insert(pipe.MaskClassification.proj(..., scan_session='session', mask_classification_method='classification_method') & Segmentation & {'mask_classification_method': mask_classification_method}, skip_duplicates=True, ignore_extra_fields=True)
                cls.Type.insert(pipe.MaskClassification.Type.proj(..., scan_session='session', mask_classification_method='classification_method', mask_type='type') & Segmentation & {'mask_classification_method': mask_classification_method}, skip_duplicates=True, ignore_extra_fields=True)


@schema
class Unit(djp.Lookup):
    definition = """
    # Re-indexing of masks to ensure uniqueness per scan, from pipeline_meso/reso.ScanSet.Unit
    -> Scan
    -> SegmentationMethod
    unit_id              : int                          # unique per scan
    ---
    -> Segmentation.Mask
    um_x                 : int                          # x-coordinate of centroid in motor coordinate system
    um_y                 : int                          # y-coordinate of centroid in motor coordinate system
    um_z                 : smallint                     # z-coordinate of mask relative to surface of the cortex
    px_x                 : smallint                     # x-coordinate of centroid in the frame
    px_y                 : smallint                     # y-coordinate of centroid in the frame
    ms_delay             : smallint                     # (ms) delay from start of frame to recording of this unit
    """

    @classmethod
    def fill(cls):
        reso = dj.create_virtual_module('reso', 'pipeline_reso')
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        for pipe in [reso, meso]:
            cls.insert((pipe.ScanSet.Unit * pipe.ScanSet.UnitInfo).proj(..., scan_session='session') & Scan & SegmentationMethod, skip_duplicates=True, ignore_extra_fields=True)
    

@schema
class RegistrationMethod(djp.Lookup):
    definition = """
    # method for registration. Note: Re-indexed from pipeline
    registration_method  : tinyint                      # method used for registration
    ---
    name                 : varchar(16)                  # short name to identify the registration method
    details              : varchar(255)                 # more details
    """
    
    contents = [
        [1, 'rigid', '3-d cross-correlation'],
        [2, 'affine', 'exhaustive search of 3-d rotations + cross-correlation'],
        [3, 'nonrigid', 'affine plus deformation field learnt via gradient ascent on correlation']
    ]

@schema
class Registration(djp.Lookup):
    definition = """
    # align a 2-d scan field to a stack from pipeline_stack.Registration
    -> Stack.Corrected
    -> Field
    -> RegistrationMethod
    ---
    scan_channel         : tinyint                      # scan channel used for registration
    stack_channel        : tinyint                      # stack channel used for registration
    """
    
    class Rigid(djp.Part):
        definition = """
        # 3-d template matching keeping the stack straight from pipeline_stack.Registration.Rigid
        -> master
        ---
        reg_x                : float                        # (um) center of field in motor coordinate system
        reg_y                : float                        # (um) center of field in motor coordinate system
        reg_z                : float                        # (um) center of field in motor coordinate system
        score                : float                        # cross-correlation score (-1 to 1)
        reg_field            : longblob                     # extracted field from the stack in the specified position
        """
    
    class Affine(djp.Part):
        definition = """
        # affine matrix learned via gradient ascent from pipeline_stack.Registration.Affine
        -> master
        ---
        a11                  : float                        # (um) element in row 1, column 1 of the affine matrix
        a21                  : float                        # (um) element in row 2, column 1 of the affine matrix
        a31                  : float                        # (um) element in row 3, column 1 of the affine matrix
        a12                  : float                        # (um) element in row 1, column 2 of the affine matrix
        a22                  : float                        # (um) element in row 2, column 2 of the affine matrix
        a32                  : float                        # (um) element in row 3, column 2 of the affine matrix
        reg_x                : float                        # (um) element in row 1, column 4 of the affine matrix
        reg_y                : float                        # (um) element in row 2, column 4 of the affine matrix
        reg_z                : float                        # (um) element in row 3, column 4 of the affine matrix
        score                : float                        # cross-correlation score (-1 to 1)
        reg_field            : longblob                     # extracted field from the stack in the specified position
        """
    
    class NonRigid(djp.Part):
        definition = """
        # affine plus deformation field learned via gradient descent from pipeline_stack.Registration.NonRigid
        -> master
        ---
        a11                  : float                        # (um) element in row 1, column 1 of the affine matrix
        a21                  : float                        # (um) element in row 2, column 1 of the affine matrix
        a31                  : float                        # (um) element in row 3, column 1 of the affine matrix
        a12                  : float                        # (um) element in row 1, column 2 of the affine matrix
        a22                  : float                        # (um) element in row 2, column 2 of the affine matrix
        a32                  : float                        # (um) element in row 3, column 2 of the affine matrix
        reg_x                : float                        # (um) element in row 1, column 4 of the affine matrix
        reg_y                : float                        # (um) element in row 2, column 4 of the affine matrix
        reg_z                : float                        # (um) element in row 3, column 4 of the affine matrix
        landmarks            : longblob                     # (um) x, y position of each landmark (num_landmarks x 2) assuming center of field is at (0, 0)
        deformations         : longblob                     # (um) x, y, z deformations per landmark (num_landmarks x 3)
        score                : float                        # cross-correlation score (-1 to 1)
        reg_field            : longblob                     # extracted field from the stack in the specified position
        """
        
    @classmethod
    def fill(cls):
        stack = dj.create_virtual_module('stack', 'pipeline_stack')       
        cls.insert((stack.Registration & Stack.Corrected).proj(registration_method='NULL').proj() * RegistrationMethod, skip_duplicates=True, ignore_extra_fields=True)
        cls.Rigid.insert(stack.Registration.Rigid.proj(..., registration_method='1')  & Stack.Corrected, skip_duplicates=True, ignore_extra_fields=True)
        cls.Affine.insert(stack.Registration.Affine.proj(..., registration_method='2') & Stack.Corrected, skip_duplicates=True, ignore_extra_fields=True)
        cls.NonRigid.insert(stack.Registration.NonRigid.proj(..., registration_method='3') & Stack.Corrected, skip_duplicates=True, ignore_extra_fields=True)


@schema
class AreaMaskMethod(djp.Lookup):
    definition = """
    # method for assigning cortex to visual areas
    mask_method          : tinyint                      # 
    ---
    name                 : varchar(16)                  # 
    details              : varchar(255)                 # 
    """
    
    contents = [
        [1, 'manual', ''], [2, 'manual', 'extrap to stack edges']
    ]


@schema
class Area(djp.Lookup):
    definition = """
    # area segmentation from pipeline_stack.Area
    ret_hash             : varchar(32)                  # single attribute representation of the key (used to avoid going over 16 attributes in the key)
    ---
    -> Stack.Corrected
    -> Scan
    -> RegistrationMethod
    -> AreaMaskMethod
    ret_idx              : smallint                     # retinotopy map index for each animal
    scan_channel         : tinyint                      # scan channel used for registration
    stack_channel        : tinyint                      # stack channel used for registration
    """
    
    class Mask(djp.Part):
        definition = """
        # area masks from pipeline_stack.Area.Mask
        -> master 
        brain_area           : varchar(256)                 # area name
        ---
        mask                 : blob                         # 2D mask of pixel area membership
        """
        
    @classmethod
    def fill(cls):
        stack = dj.create_virtual_module('stack', 'pipeline_stack')
        source = stack.Area.Mask & Stack.Corrected & Scan & 'registration_method=5'
        to_insert = source.proj(..., reg_method='registration_method').proj(..., reg_method='NULL').proj(..., registration_method='2')
        cls.insert(to_insert, skip_duplicates=True, ignore_extra_fields=True)
        cls.Mask.insert(to_insert, skip_duplicates=True, ignore_extra_fields=True)

@schema
class Grid(djp.Lookup):
    definition = """
    -> Registration
    ---
    grid                 : longblob                     # field grid in motor coordinates after registration into stack
    """
    
    @classmethod
    def fill(cls):
        from pipeline import stack
        # must be run with torch == 1.2.0
        for key in Registration:
            # get registration type
            reg_type = (RegistrationMethod & key).fetch1('name')
            
            # compatibility with pipeline
            key['registration_method'] = 5 
            
            # get grid
            key['grid'] = (stack.Registration & key).get_grid(type=reg_type)
            
            # convert registration method back to schema convention
            key['registration_method'] = (RegistrationMethod & {'name': reg_type}).fetch1('registration_method')
            
            cls.insert1(key, skip_duplicates=True, ignore_extra_fields=True)


@schema
class AreaMembership(djp.Lookup):
    definition = """
    # cell membership in visual areas according to stack registered retinotopy from pipeline_meso.AreaMembership.UnitInfo
    ret_hash             : varchar(32)                  # single attribute representation of the key (used to avoid going over 16 attributes in the key)
    -> Registration
    -> Unit
    ---
    -> Area.Mask
    confidence           : float                        # confidence in area assignment
    """


    @classmethod
    def fill(cls):
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        source = meso.AreaMembership.UnitInfo.proj(..., scan_session='session') & Animal
        source &= Registration.proj(registration_method='NULL').proj()
        source = source.proj(..., registration_method='2') & Unit
        cls.insert(source, skip_duplicates=True, ignore_extra_fields=True)


@schema
class StackUnit(djp.Lookup):
    definition = """
    # Unit xyz in stack from pipeline_meso.StackCoordinates (Note: stack_xyz renamed motor_xyz)
    -> Registration
    -> Unit
    ---
    motor_x            : float    # centroid x stack coordinates with motor offset (microns)
    motor_y            : float    # centroid y stack coordinates with motor offset (microns)
    motor_z            : float    # centroid z stack coordinates with motor offset (microns)
    stack_x            : float    # centroid x stack coordinates with 0,0,0 in top, left, back corner of stack (microns)
    stack_y            : float    # centroid y stack coordinates with 0,0,0 in top, left, back corner of stack (microns)
    stack_z            : float    # centroid z stack coordinates with 0,0,0 in top, left, back corner of stack (microns)   
    """
    
    @classmethod
    def fill(cls):
        meso = dj.create_virtual_module('meso', 'pipeline_meso')
        source = meso.StackCoordinates.UnitInfo.proj(..., scan_session='session', motor_x='stack_x', motor_y='stack_y', motor_z='stack_z') & Animal
        source &= Registration.proj(registration_method='NULL').proj() & 'segmentation_method=6'
        source = (source * Stack.Corrected).proj('motor_x', 'motor_y', 'motor_z', registration_method='2', stack_x = 'round(motor_x - x + um_width/2, 2)', stack_y = 'round(motor_y - y + um_height/2, 2)', stack_z = 'round(motor_z - z + um_depth/2, 2)')
        cls.insert(source, skip_duplicates=True, ignore_extra_fields=True)


@schema
class UnitSource(djp.Manual):
    definition = """
    # Convenience table to subset units for analysis and consolidate unit info
    animal_id            : int                          # id number 
    scan_session         : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    unit_id              : int                          # unique per scan
    ---
    um_x                 : int                          # x-coordinate of centroid in motor coordinate system
    um_y                 : int                          # y-coordinate of centroid in motor coordinate system
    um_z                 : smallint                     # z-coordinate of mask relative to surface of the cortex
    px_x                 : smallint                     # x-coordinate of centroid in the frame
    px_y                 : smallint                     # y-coordinate of centroid in the frame
    ms_delay             : smallint                     # (ms) delay from start of frame to recording of this unit
    motor_x              : float                        # centroid x stack coordinates with motor offset (microns)
    motor_y              : float                        # centroid y stack coordinates with motor offset (microns)
    motor_z              : float                        # centroid z stack coordinates with motor offset (microns)
    stack_x              : float                        # centroid x stack coordinates (microns)
    stack_y              : float                        # centroid y stack coordinates (microns)
    stack_z              : float                        # centroid z stack coordinates (microns)
    mask_type            : varchar(16)                  # mask classification
    brain_area           : varchar(256)                 # area name
    -> Unit
    -> StackUnit
    -> MaskClassification.Type
    -> AreaMembership
    """
        
    @classmethod
    def fill(cls): 
        key = {}
        key['animal_id'] = 17797
        key['stack_session'] = 9
        key['stack_idx'] = 19
        key['volume_id'] = 1
        key['segmentation_method'] = 6
        key['registration_method'] = 2
        key['mask_classification_method'] = 2
        key['ret_hash'] = "edec10b648420dd1dc8007b607a5046b"

        cls.insert(Unit * StackUnit * MaskClassification.Type * AreaMembership & key, ignore_extra_fields=True, skip_duplicates=True)



