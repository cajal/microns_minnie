import functools
import datajoint as dj
import pandas as pd
from tqdm import tqdm
from datajoint.datajoint_plus import classproperty
from microns_nda_api.schemas import minnie_function, minnie_nda

# Utility functions
class VMMixin:
    ## virtual module management
    virtual_modules = {}

    @classmethod
    def update_virtual_modules(cls, module_name, schema_name):
        if module_name not in cls.virtual_modules:
            cls.virtual_modules[module_name] = dj.create_virtual_module(
                module_name, schema_name
            )

    @classmethod
    def spawn_virtual_modules(cls, virtual_module_dict):
        for model_name, schema_name in virtual_module_dict.items():
            cls.update_virtual_modules(model_name, schema_name)


# Utility tables
# WARNING: delete and repopulate!
class ScanSet(minnie_function.ScanSet):
    class Member(minnie_function.ScanSet.Member):
        pass

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
            insert_to_parts_kws={"skip_duplicates": True, "ignore_extra_fields": True},
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
    class Member(minnie_function.StimTypeGrp.Member):
        pass

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


# Orientation
## Faithful copy of functional properties from external database
class OrientationDV11521GD(VMMixin, minnie_function.OrientationDV11521GD):
    class Unit(minnie_function.OrientationDV11521GD.Unit):
        pass

    virtual_module_dict = {
        "dv_tunings_v2_direction": "dv_tunings_v2_direction",
        "dv_tunings_v2_response": "dv_tunings_v2_response",
        "dv_scans_v1_scan_dataset": "dv_scans_v1_scan_dataset",
        "dv_scans_v1_scan": "dv_scans_v1_scan",
        "dv_nns_v5_scan": "dv_nns_v5_scan",
        "dv_stimuli_v1_stimulus": "dv_stimuli_v1_stimulus",
        "pipeline_stimulus": "pipeline_stimulus",
    }

    @classmethod
    def fill(cls):
        cls.spawn_virtual_modules(cls.virtual_module_dict)
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
            cls.virtual_modules["dv_tunings_v2_response"].ResponseInfo.Scan.proj(
                ..., scan_session="session"
            )
            & ScanSet.Member()
        )
        target = master.proj() - cls.proj()
        if len(target) == 0:
            return
        cls.insert(target, ignore_extra_fields=True, skip_duplicates=True)
        unit = (
            (
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
            ).proj(..., scan_session="session")
            & ScanSet.Member()
            & target.proj()
        )
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
        self.spawn_virtual_modules(self.virtual_module_dict)
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
        self.spawn_virtual_modules(self.virtual_module_dict)
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
        self.spawn_virtual_modules(self.virtual_module_dict)
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return (
            (self & key).proj()
            * self.virtual_modules["dv_tunings_v2_response"].ResponseInfo.Scan.proj(
                scan_session="session"
            )
        ).fetch1("animal_id", "scan_session", "scan_idx")

    def len_sec(self, key=None):
        self.spawn_virtual_modules(self.virtual_module_dict)
        key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
        return self.aggr(
            (
                self.virtual_modules["dv_tunings_v2_response"].ResponseSet.Slice & key
            ).proj(sec="n_frames / hz"),
            len_sec="sum(sec)",
        ).fetch1("len_sec")


## Aggregation tables
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
        animal_id, scan_session, scan_idx = (Orientation & key).scan()
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
    class Member(minnie_function.OrientationScanSet.Member):
        pass

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
            insert_to_parts_kws={"skip_duplicates": True, "ignore_extra_fields": True},
        )

# Oracle
## Faithful copy of data
class OracleDVScan1(VMMixin, minnie_function.OracleDVScan1):

    virtual_module_dict = {
        "dv_scans_v1_oracle": "dv_scans_v1_oracle",
    }

    class Unit(minnie_function.OracleDVScan1.Unit):
        pass

    @classmethod
    def fill(cls):
        cls.spawn_virtual_modules(cls.virtual_module_dict)
        keys = minnie_nda.Scan.fetch("KEY")
        keys = (
            (cls.virtual_modules["dv_scans_v1_oracle"].TrialVsOracle() & keys)
            .proj(..., scan_session="session")
            .fetch("KEY")
        )
        for k in keys:
            cls.insert1(k, skip_duplicates=True, ignore_extra_fields=True)
            cls.Unit.insert(
                (cls.virtual_modules["dv_scans_v1_oracle"].TrialVsOracle.Unit & k)
                .proj(..., scan_session="session")
                .fetch(),
                skip_duplicates=True,
                ignore_extra_fields=True,
            )


class OracleTuneMovieOracle(VMMixin, minnie_function.OracleTuneMovieOracle):

    virtual_module_dict = {
        "pipeline_tune": "pipeline_tune",
    }

    class Unit(minnie_function.OracleTuneMovieOracle.Unit):
        pass

    @classmethod
    def fill(cls):
        cls.spawn_virtual_modules(cls.virtual_module_dict)
        scan_keys = minnie_nda.Scan.fetch("KEY")
        scan_keys = [
            {**key, "segmentation_method": 6, "spike_method": 5} for key in scan_keys
        ]
        scan_rel = dj.U(*cls.heading) & (
            cls.virtual_modules["pipeline_tune"].MovieOracle & scan_keys
        ).proj(..., scan_session="session")
        cls.insert(scan_rel, skip_duplicates=True, ignore_extra_fields=True)
        unit_rel = dj.U(*cls.Unit.heading) & (
            cls.virtual_modules["pipeline_tune"].MovieOracle.Total & scan_keys
        ).proj(..., scan_session="session")
        cls.Unit.insert(unit_rel, skip_duplicates=True, ignore_extra_fields=True)


class Oracle(minnie_function.Oracle):
    @classmethod
    def fill(cls):
        for part in cls.parts(as_cls=True):
            part.fill()

    class DVScan1(minnie_function.Oracle.DVScan1):
        @classproperty
        def source(cls):
            return eval(super()._source)

        @classmethod
        def fill(cls):
            constant_attrs = {
                "oracle_type": cls.source.__name__,
            }
            cls.insert(
                cls.source,
                insert_to_master=True,
                constant_attrs=constant_attrs,
                ignore_extra_fields=True,
            )

    class TuneMovieOracle(minnie_function.Oracle.TuneMovieOracle):
        @classproperty
        def source(cls):
            return eval(super()._source)

        @classmethod
        def fill(cls):
            constant_attrs = {
                "oracle_type": cls.source.__name__,
            }
            cls.insert(
                cls.source,
                insert_to_master=True,
                constant_attrs=constant_attrs,
                ignore_extra_fields=True,
            )


class OracleScanInfo(minnie_function.OracleScanInfo):
    def make(self, key):
        animal_id, scan_session, scan_idx = (Oracle & key).scan()
        self.insert(
            [
                {**key, "animal_id": a, "scan_session": s, "scan_idx": c}
                for k, a, s, c in zip(key, animal_id, scan_session, scan_idx)
            ],
        )


class OracleScanSet(minnie_function.OracleScanSet):
    class Member(minnie_function.OracleScanSet.Member):
        pass

    @classmethod
    def fill(cls, keys, description=""):
        keys = (OracleScanInfo.proj() & keys).fetch("KEY")
        # check here all scans are unique
        assert len(OracleScanInfo & keys) == len(
            minnie_nda.Scan * OracleScanInfo & keys
        )
        # check all members of a set share the same oracle_type
        assert (
            len(dj.U("oracle_type") & (OracleScanInfo & keys)) == 1
        ), "All members of a set must share the same oracle_type"
        scan_keys = (minnie_nda.Scan & (OracleScanInfo & keys)).fetch("KEY")
        scan_set_hash = ScanSet.add_hash_to_rows(scan_keys)[ScanSet.hash_name].unique()[
            0
        ]
        cls.insert(
            keys,
            constant_attrs={
                "description": description,
                "scan_set_hash": scan_set_hash,
            },
            insert_to_parts=cls.Member,
            ignore_extra_fields=True,
            skip_duplicates=True,
            insert_to_parts_kws={"skip_duplicates": True, "ignore_extra_fields": True},
        )


# # Predictive model performance and parameters
## Aggregation tables
class DynamicModel(minnie_function.DynamicModel):
    class NnsV5(VMMixin, minnie_function.DynamicModel.NnsV5):
        virtual_module_dict = {
            "dv_nns_v5_scan": "dv_nns_v5_scan",
        }

        @classmethod
        def fill(cls):
            cls.spawn_virtual_modules(cls.virtual_module_dict)
            keys = minnie_nda.Scan.fetch("KEY")
            scan_keys = (
                (
                    cls.virtual_modules["dv_nns_v5_scan"].Readout.proj(
                        ..., scan_session="session"
                    )
                    & keys
                )
                .fetch(as_dict=True)
            )
            for scan_key in scan_keys:
                cls.insert1(
                    scan_key,
                    insert_to_master=True,
                    constant_attrs={"dynamic_model_type": cls.__name__},
                    ignore_extra_fields=True,
                )
                unit_keys = (
                    cls.virtual_modules["dv_nns_v5_scan"].Readout.Unit.proj(
                        ..., scan_session="session"
                    )
                    & scan_key
                ).fetch(format='frame').reset_index()
                unit_keys = cls.add_hash_to_rows(unit_keys)
                DynamicModel.NnsV5UnitReadout.insert(
                    unit_keys,
                    constant_attrs={"dynamic_model_type": cls.__name__},
                    ignore_extra_fields=True,
                )

    class NnsV5UnitReadout(minnie_function.DynamicModel.NnsV5UnitReadout): pass

    @classmethod
    def fill(cls):
        for p in cls.parts(as_cls=True):
            try:
                p.fill()
            except AttributeError:
                pass


class DynamicModelScore(minnie_function.DynamicModelScore):
    class NnsV5(VMMixin, minnie_function.DynamicModelScore.NnsV5):
        virtual_module_dict = {
            "dv_nns_v5_scan": "dv_nns_v5_scan",
            "dv_nns_v5_model": "dv_nns_v5_model",
        }

        @classmethod
        def fill(cls, key=None):
            model_maker = (DynamicModel & f"dynamic_model_type='{cls.__name__}'").maker()
            model_maker = model_maker if key is None else (model_maker & key)
            cls.spawn_virtual_modules(cls.virtual_module_dict)
            scan_keys = (
                (
                    (
                        cls.virtual_modules["dv_nns_v5_scan"].TrialVsModel
                        * cls.virtual_modules["dv_nns_v5_model"].BehaviorConfig.Scan
                    ).proj(..., scan_session="session")
                    * model_maker
                )
                .fetch(as_dict=True)
            )  # 8 sec
            for scan_key in tqdm(scan_keys):
                cls.insert1(
                    scan_key,
                    insert_to_master=True,
                    constant_attrs={"dynamic_score_type": cls.__name__},
                    ignore_extra_fields=True,
                )
                unit_keys = (
                (
                    (
                        cls.virtual_modules["dv_nns_v5_scan"].TrialVsModel.Unit
                        * cls.virtual_modules["dv_nns_v5_model"].BehaviorConfig.Scan
                    ).proj(..., scan_session="session") * model_maker
                    & scan_key
                )
                .fetch(format='frame').reset_index()
                )
                unit_keys = cls.add_hash_to_rows(unit_keys)
                DynamicModelScore.NnsV5UnitScore.insert(
                    unit_keys,
                    constant_attrs={"dynamic_model_type": cls.__name__},
                    ignore_extra_fields=True,
                )
            
    class NnsV5UnitScore(minnie_function.DynamicModelScore.NnsV5UnitScore): pass

    @classmethod
    def fill(cls):
        for p in cls.parts(as_cls=True):
            try:
                p.fill()
            except AttributeError:
                pass


class DynamicModelScanSet(minnie_function.DynamicModelScanSet):
    class Member(minnie_function.DynamicModelScanSet.Member):
        pass

    @classmethod
    def fill(cls, keys, name, description=""):
        keys = (DynamicModel.proj() & keys).fetch("KEY")
        # check if all scans are unique
        assert len(DynamicModel & keys) == len(
            minnie_nda.Scan * DynamicModel & keys
        )
        # check if all members of a set share the same readout_type
        assert (
            type((DynamicModel & keys).maker()) != list
        ), "All members of a set must share the same maker"
        scan_keys = (minnie_nda.Scan & (DynamicModel & keys)).fetch("KEY")
        scan_set_hash = ScanSet.hash1(scan_keys, unique=True)
        cls.insert(
            keys,
            constant_attrs={
                "name": name,
                "description": description,
                "scan_set_hash": scan_set_hash,
                "n_members": len(keys),
            },
            insert_to_parts=cls.Member,
            ignore_extra_fields=True,
            skip_duplicates=True,
            insert_to_parts_kws={"skip_duplicates": True, "ignore_extra_fields": True},
        )
