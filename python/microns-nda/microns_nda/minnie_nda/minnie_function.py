import functools
import datajoint as dj
from datajoint.datajoint_plus import classproperty
from microns_nda_api.schemas import minnie_function, minnie_nda

# Utility tables
class ScanSet(minnie_function.ScanSet):

    class Member(minnie_function.ScanSet.Member): pass

    @classmethod
    def fill(cls, keys, name, description):
        n_members = len(minnie_nda.Scan & keys)
        keys = (minnie_nda.Scan & keys).fetch("KEY")
        cls.insert(
            keys,
            insert_to_parts=cls.Member,
            ignore_extra_fields=True,
            skip_duplicates=True,
            constant_attrs={
                "name": name,
                "description": description,
                "n_members": n_members,
            },
        )


class ResponseType(minnie_function.ResponseType):
    contents = [
        ("in_vivo", "Tuning properties extracted from in vivo responses."),
        ("in_silico", "Tuning properties extracted from in silico model responses."),
    ]


class StimType(minnie_function.StimType):
    @property
    def contents(self):
        return [[s] for s in Orientation.all_stimulus_types()]

    def fill(self):
        self.insert(self.contents, skip_duplicates=True)


class StimTypeGrp(minnie_function.StimTypeGrp):

    class Member(minnie_function.StimTypeGrp.Member): pass

    @classmethod
    def fill(cls, keys):
        n_members = len(StimType & keys)
        keys = (StimType & keys).fetch("KEY")
        stim_types = (StimType & keys).fetch("stimulus_type")
        cls.insert(
            keys,
            insert_to_parts=cls.Member,
            ignore_extra_fields=True,
            skip_duplicates=True,
            constant_attrs={
                "stim_types": ", ".join(stim_types),
                "n_members": n_members,
            },
        )


# Faithful copy of functional properties from external database
## Orientation
class OrientationDV11521GD(minnie_function.OrientationDV11521GD):

    class Unit(minnie_function.OrientationDV11521GD.Unit): pass

    @classmethod
    def update_virtual_modules(cls, module_name, schema_name):
        if module_name not in cls.virtual_modules:
            cls.virtual_modules[module_name] = dj.create_virtual_module(
                module_name, schema_name
            )

    @classmethod
    def spawn_virtual_modules(cls):
        if not hasattr(cls, "virtual_modules"):
            cls.virtual_modules = {}
        cls.update_virtual_modules("dv_tunings_v2_direction", "dv_tunings_v2_direction")
        cls.update_virtual_modules("dv_tunings_v2_response", "dv_tunings_v2_response")
        cls.update_virtual_modules(
            "dv_scans_v1_scan_dataset", "dv_scans_v1_scan_dataset"
        )
        cls.update_virtual_modules(
            "dv_scans_v1_scan", "dv_scans_v1_scan"
        )
        cls.update_virtual_modules("dv_nns_v5_scan", "dv_nns_v5_scan")
        cls.update_virtual_modules("dv_stimuli_v1_stimulus", "dv_stimuli_v1_stimulus")
        cls.update_virtual_modules("pipeline_stimulus", "pipeline_stimulus")

    @classmethod
    def fill(cls):
        cls.spawn_virtual_modules()
        master = (
            cls.virtual_modules["dv_tunings_v2_direction"].BiVonMises()
            * cls.virtual_modules["dv_tunings_v2_direction"]
            .BiVonMisesBootstrap()
            .proj(..., bs_samples="n_samples", bs_seed="seed")
            * cls.virtual_modules["dv_tunings_v2_direction"]
            .BiVonMisesPermutation()
            .proj(..., permute_samples="n_samples", permute_seed="seed")
            * cls.virtual_modules["dv_tunings_v2_direction"].Tuning()
            * cls.virtual_modules["dv_tunings_v2_direction"]
            .GlobalDiscreteTrialTuning()
            .proj(..., tuning_curve_radians="radians")
        ) & (
            cls.virtual_modules["dv_tunings_v2_response"].ResponseInfo.Scan.proj(..., scan_session="session")
            & ScanSet.Member()
        )
        target = master.proj() - cls.proj()
        if len(target) == 0:
            return
        cls.insert(target, ignore_extra_fields=True, skip_duplicates=True)
        unit = (
            cls.virtual_modules["dv_tunings_v2_direction"].BiVonMises().Unit
            * cls.virtual_modules["dv_tunings_v2_direction"]
            .BiVonMisesBootstrap()
            .Unit()
            .proj(..., bs_samples="n_samples", bs_seed="seed")
            * cls.virtual_modules["dv_tunings_v2_direction"]
            .BiVonMisesPermutation()
            .Unit()
            .proj(..., permute_samples="n_samples", permute_seed="seed")
            * cls.virtual_modules["dv_tunings_v2_direction"].Tuning()
            * cls.virtual_modules["dv_tunings_v2_direction"]
            .GlobalDiscreteTrialTuning()
            .Unit()
            .proj(..., tuning_curve_mu="mu", tuning_curve_sigma="sigma")
        ).proj(..., scan_session="session") & ScanSet.Member() & target.proj()
        # Add back all functional units (some were removed in dyanmic vision because they were not unique units)
        unique_id = (
            cls.virtual_modules["dv_nns_v5_scan"].ScanInfo
            * cls.virtual_modules["dv_nns_v5_scan"].ScanConfig
            * cls.virtual_modules["dv_scans_v1_scan_dataset"].UnitConfig().Unique()
        )
        neuron2unit = (
            cls.virtual_modules["dv_scans_v1_scan"].Unique.Neuron.proj(
                ..., scan_session="session"
            )
            & unique_id
            & ScanSet.Member()
        ).proj(..., unique_unit_id="unit_id") * (
            cls.virtual_modules["dv_scans_v1_scan"].Unique.Unit.proj(
                scan_session="session"
            )
            & unique_id
            & ScanSet.Member()
        )
        unit = unit.proj(..., unique_unit_id="unit_id") * neuron2unit
        cls.Unit.insert(unit, ignore_extra_fields=True, skip_duplicates=True)

    def stimulus_type(self, key=None):
        # returns list of stimuli used in the tuning
        # elements of the list are stimulus_type in the pipeline_stimulus.Condition table
        self.spawn_virtual_modules()
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return list(
            (
                dj.U("stimulus_type")
                & (
                    (
                        self.virtual_modules["dv_tunings_v2_response"].ResponseSet.Slice
                        & key
                    )
                    * self.virtual_modules["dv_stimuli_v1_stimulus"].StimulusCondition
                    * self.virtual_modules["pipeline_stimulus"].Condition
                )
            ).fetch("stimulus_type")
        )

    def response_type(self, key=None):
        # returns {'in_vivo', 'in_silico'}
        self.spawn_virtual_modules()
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        response_type = (
            self.virtual_modules["dv_tunings_v2_response"].ResponseConfig & key
        ).fetch1("response_type")
        assert response_type in {
            "Scan1Mean",
            "Nn5Pure",
        }, f"response_type not supported, consider delete entry {key}"
        response_mapping = {
            "Scan1Mean": "in_vivo",
            "Nn5Pure": "in_silico",
        }
        return response_mapping[response_type]

    def scan(self, key=None):
        self.spawn_virtual_modules()
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return ((self & key).proj() * self.virtual_modules[
            "dv_tunings_v2_response"
        ].ResponseInfo.Scan.proj(scan_session="session")).fetch1('animal_id', 'scan_session', 'scan_idx')

    def len_sec(self, key=None):
        self.spawn_virtual_modules()
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return self.aggr(
            (
                self.virtual_modules["dv_tunings_v2_response"].ResponseSet.Slice & key
            ).proj(sec="n_frames / hz"),
            len_sec="sum(sec)",
        ).fetch1("len_sec")


class Orientation(minnie_function.Orientation):
    @classmethod
    def fill(cls):
        for part in cls.parts(as_cls=True):
            part.fill()

    def stimulus_type(self, key=None):
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return (self & key).part_table().stimulus_type()

    def response_type(self, key=None):
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return (self & key).part_table().response_type()

    def scan(self, key=None):
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return (self & key).part_table().scan()

    def len_sec(self, key=None):
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return (self & key).part_table().len_sec()
        
    @classmethod
    def all_stimulus_types(cls):
        return functools.reduce(
            lambda a, b: {*a, *b},
            [(cls & key).stimulus_type() for key in cls.fetch("KEY")],
        )

    class DV11521GD(minnie_function.Orientation.DV11521GD):
        @classproperty
        def source(cls):
            return eval(super()._source)

        @classmethod
        def fill(cls):
            constant_attrs = {
                "orientation_type": Orientation.DV11521GD.__name__,
            }
            cls.insert(
                cls.source,
                insert_to_master=True,
                constant_attrs=constant_attrs,
                ignore_extra_fields=True,
            )
        
        def stimulus_type(self, key=None):
            key = self.fetch1() if key is None else (self & key).fetch1()
            return (self.source & key).stimulus_type()

        def response_type(self, key=None):
            key = self.fetch1() if key is None else (self & key).fetch1()
            return (self.source & key).response_type()

        def scan(self, key=None):
            key = self.fetch1() if key is None else (self & key).fetch1()
            return (self.source & key).scan()

        def len_sec(self, key=None):
            key = self.fetch1() if key is None else (self & key).fetch1()
            return (self.source & key).len_sec()


class OrientationScanInfo(minnie_function.OrientationScanInfo):
    @property
    def key_source(self):
        return Orientation

    def make(self, key):
        stim_type_grp = (Orientation & key).stimulus_type()
        stim_type_grp = [{"stimulus_type": s} for s in stim_type_grp]
        stim_type_grp_hash = StimTypeGrp.add_hash_to_rows(stim_type_grp)[
            StimTypeGrp.hash_name
        ].unique()
        assert len(stim_type_grp_hash) == 1
        stim_type_grp_hash = stim_type_grp_hash[0]
        assert StimTypeGrp.restrict_with_hash(
            stim_type_grp_hash
        ), "stim_type_grp_hash does not exist in StimTypeGrp"
        response_type = (Orientation & key).response_type()
        animal_id, scan_session, scan_idx = (
            (Orientation & key).scan()
        )
        stimulus_length = round((Orientation & key).len_sec(), 2)
        self.insert1(
            dict(
                key,
                stim_type_grp_hash=stim_type_grp_hash,
                response_type=response_type,
                stimulus_length=stimulus_length,
                animal_id=animal_id,
                scan_session=scan_session,
                scan_idx=scan_idx,
            )
        )


class OrientationScanSet(minnie_function.OrientationScanSet):
    
    class Member(minnie_function.OrientationScanSet.Member): pass

    @classmethod
    def fill(cls, keys, description=""):
        keys = (OrientationScanInfo.proj() & keys).fetch("KEY")
        # check here all scans are unique
        assert len(OrientationScanInfo & keys) == len(
            minnie_nda.Scan * OrientationScanInfo & keys
        )
        scan_keys = (minnie_nda.Scan & (OrientationScanInfo & keys)).fetch("KEY")
        scan_set_hash = ScanSet.add_hash_to_rows(scan_keys)[ScanSet.hash_name].unique()[
            0
        ]
        response_type = (dj.U("response_type") & (OrientationScanInfo & keys)).fetch1(
            "response_type"
        )
        stim_type_grp_hash = (
            dj.U("stim_type_grp_hash") & (OrientationScanInfo & keys)
        ).fetch1("stim_type_grp_hash")
        cls.insert(
            keys,
            constant_attrs={
                "description": description,
                "scan_set_hash": scan_set_hash,
                "response_type": response_type,
                "stim_type_grp_hash": stim_type_grp_hash,
            },
            insert_to_parts=cls.Member,
            ignore_extra_fields=True,
            skip_duplicates=True,
        )
