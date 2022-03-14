from datajoint import datajoint_plus as djp

from microns_nda_api.schemas import minnie_nda as nda


class Animal(nda.Animal):
    contents = [[17797]]


class Scan(nda.Scan):

    class Experiment(nda.Scan.Experiment): pass

    class Reso(nda.Scan.Reso): pass

    class Meso(nda.Scan.Meso): pass

    @classmethod
    def fill(cls):
        experiment = djp.create_djp_module('experiment', 'pipeline_experiment')
        reso = djp.create_djp_module('reso', 'pipeline_reso')
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        cls.insert(experiment.Scan.proj(..., scan_session='session') & Animal, ignore_extra_fields=True, skip_duplicates=True)
        cls.Experiment.insert(experiment.Scan.proj(..., scan_session='session') & Animal, skip_duplicates=True)
        cls.Reso.insert(reso.ScanInfo.proj(..., scan_session='session') & Animal, skip_duplicates=True)
        cls.Meso.insert(meso.ScanInfo.proj(..., scan_session='session') & Animal, skip_duplicates=True)
        

class Stack(nda.Stack):    

    class Experiment(nda.Stack.Experiment): pass

    class Corrected(nda.Stack.Corrected): pass
    
    @classmethod
    def fill(cls):
        experiment = djp.create_djp_module('experiment', 'pipeline_experiment')
        stack = djp.create_djp_module('stack', 'pipeline_stack')
        cls.insert(experiment.Stack.proj(..., stack_session='session') & Animal, ignore_extra_fields=True, skip_duplicates=True)
        cls.Experiment.insert(experiment.Stack.proj(..., stack_session='session') & Animal, skip_duplicates=True)
        cls.Corrected.insert((stack.StackInfo * stack.CorrectedStack).proj(..., stack_session='session') & Animal, skip_duplicates=True)


class Field(nda.Field):

    class Reso(nda.Field.Reso): pass
    
    class Meso(nda.Field.Meso): pass
    
    @classmethod
    def fill(cls):
        reso = djp.create_djp_module('reso', 'pipeline_reso')
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        for pipe, part in zip([reso, meso], [cls.Reso, cls.Meso]):
            cls.insert(pipe.ScanInfo.Field.proj(scan_session='session') & Scan, ignore_extra_fields=True, skip_duplicates=True)
            part.insert(pipe.ScanInfo.Field.proj(..., scan_session='session') & Scan, skip_duplicates=True)


class SummaryImages(nda.SummaryImages):
    
    class Average(nda.SummaryImages.Average): pass
    
    class Correlation(nda.SummaryImages.Correlation): pass
    
    class L6Norm(nda.SummaryImages.L6Norm): pass

    class All(nda.SummaryImages.All): pass
    
    @classmethod
    def fill(cls):
        reso = djp.create_djp_module('reso', 'pipeline_reso')
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        for pipe in [reso, meso]:
            cls.insert(pipe.SummaryImages.proj(scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
            cls.Average.insert(pipe.SummaryImages.Average.proj(..., scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
            cls.Correlation.insert(pipe.SummaryImages.Correlation.proj(..., scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
            cls.L6Norm.insert(pipe.SummaryImages.L6Norm.proj(..., scan_session='session', scan_channel='channel') & Field, skip_duplicates=True, ignore_extra_fields=True)
    

class SegmentationMethod(nda.SegmentationMethod):
    contents = [
        [6,	'nmf-test',	'test nmf with diff parameters']
    ]

    
class Segmentation(nda.Segmentation):

    class CNMFCaImAn(nda.Segmentation.CNMFCaImAn): pass

    class Mask(nda.Segmentation.Mask): pass
    
    @classmethod
    def fill(cls):
        reso = djp.create_djp_module('reso', 'pipeline_reso')
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        for segmentation_method in [6]:
            for pipe in [reso, meso]:
                cls.insert(pipe.Segmentation.proj(..., scan_session='session', scan_channel='channel') & Field & {'segmentation_method': segmentation_method}, skip_duplicates=True, ignore_extra_fields=True)
                cls.CNMFCaImAn.insert(pipe.Segmentation.CNMF.proj(..., scan_session='session') & Field & {'segmentation_method': segmentation_method}, skip_duplicates=True, ignore_extra_fields=True)
                cls.Mask.insert(pipe.Segmentation.Mask.proj(..., scan_session='session') & Field & {'segmentation_method': segmentation_method}, skip_duplicates=True, ignore_extra_fields=True)
            

class MaskClassificationMethod(nda.MaskClassificationMethod):
    contents = [
        [2, 'cnn-caiman', 'classification made by a trained convolutional network']
    ]


class MaskClassification(nda.MaskClassification):

    class Type(nda.MaskClassification.Type): pass

    @classmethod
    def fill(cls):
        reso = djp.create_djp_module('reso', 'pipeline_reso')
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        for mask_classification_method in [2]:
            for pipe in [reso, meso]:
                cls.insert(pipe.MaskClassification.proj(..., scan_session='session', mask_classification_method='classification_method') & Segmentation & {'mask_classification_method': mask_classification_method}, skip_duplicates=True, ignore_extra_fields=True)
                cls.Type.insert(pipe.MaskClassification.Type.proj(..., scan_session='session', mask_classification_method='classification_method', mask_type='type') & Segmentation & {'mask_classification_method': mask_classification_method}, skip_duplicates=True, ignore_extra_fields=True)


class Unit(nda.Unit):

    @classmethod
    def fill(cls):
        reso = djp.create_djp_module('reso', 'pipeline_reso')
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        for pipe in [reso, meso]:
            cls.insert((pipe.ScanSet.Unit * pipe.ScanSet.UnitInfo).proj(..., scan_session='session') & Scan & SegmentationMethod, skip_duplicates=True, ignore_extra_fields=True)
    

class RegistrationMethod(nda.RegistrationMethod):
    contents = [
        [1, 'rigid', '3-d cross-correlation'],
        [2, 'affine', 'exhaustive search of 3-d rotations + cross-correlation'],
        [3, 'nonrigid', 'affine plus deformation field learnt via gradient ascent on correlation']
    ]

class Registration(nda.Registration):
    
    class Rigid(nda.Registration.Rigid): pass
    
    class Affine(nda.Registration.Affine): pass
    
    class NonRigid(nda.Registration.NonRigid): pass
        
    @classmethod
    def fill(cls):
        stack = djp.create_djp_module('stack', 'pipeline_stack')       
        cls.insert((stack.Registration & Stack.Corrected).proj(registration_method='NULL').proj() * RegistrationMethod, skip_duplicates=True, ignore_extra_fields=True)
        cls.Rigid.insert(stack.Registration.Rigid.proj(..., registration_method='1')  & Stack.Corrected, skip_duplicates=True, ignore_extra_fields=True)
        cls.Affine.insert(stack.Registration.Affine.proj(..., registration_method='2') & Stack.Corrected, skip_duplicates=True, ignore_extra_fields=True)
        cls.NonRigid.insert(stack.Registration.NonRigid.proj(..., registration_method='3') & Stack.Corrected, skip_duplicates=True, ignore_extra_fields=True)


class AreaMaskMethod(nda.AreaMaskMethod):
    contents = [
        [1, 'manual', ''], [2, 'manual', 'extrap to stack edges']
    ]


class Area(nda.Area):
    
    class Mask(nda.Area.Mask): pass
        
    @classmethod
    def fill(cls):
        stack = djp.create_djp_module('stack', 'pipeline_stack')
        source = stack.Area.Mask & Stack.Corrected & Scan & 'registration_method=5'
        to_insert = source.proj(..., reg_method='registration_method').proj(..., reg_method='NULL').proj(..., registration_method='2')
        cls.insert(to_insert, skip_duplicates=True, ignore_extra_fields=True)
        cls.Mask.insert(to_insert, skip_duplicates=True, ignore_extra_fields=True)

class Grid(nda.Grid):
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


class AreaMembership(nda.AreaMembership):
    @classmethod
    def fill(cls):
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        source = meso.AreaMembership.UnitInfo.proj(..., scan_session='session') & Animal
        source &= Registration.proj(registration_method='NULL').proj()
        source = source.proj(..., registration_method='2') & Unit
        cls.insert(source, skip_duplicates=True, ignore_extra_fields=True)


class StackUnit(nda.StackUnit):
    @classmethod
    def fill(cls):
        meso = djp.create_djp_module('meso', 'pipeline_meso')
        source = meso.StackCoordinates.UnitInfo.proj(..., scan_session='session', motor_x='stack_x', motor_y='stack_y', motor_z='stack_z') & Animal
        source &= Registration.proj(registration_method='NULL').proj() & 'segmentation_method=6'
        source = (source * Stack.Corrected).proj('motor_x', 'motor_y', 'motor_z', registration_method='2', stack_x = 'round(motor_x - x + um_width/2, 2)', stack_y = 'round(motor_y - y + um_height/2, 2)', stack_z = 'round(motor_z - z + um_depth/2, 2)')
        cls.insert(source, skip_duplicates=True, ignore_extra_fields=True)


class UnitSource(nda.UnitSource):        
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



