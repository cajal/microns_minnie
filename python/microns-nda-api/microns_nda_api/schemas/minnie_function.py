import datajoint as dj
import datajoint_plus as djp
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import decimal

from functools import cached_property

from ..config import minnie_function_config as config
from ..utils.function_utils import pcorr, xcorr, pcorr_p
from ..utils import datajoint_utils
from . import minnie_nda
from microns_nda_api.utils import function_utils
from scipy.spatial.distance import squareform

netflix = djp.create_dj_virtual_module("netflix", "pipeline_netflix")
stimulus = djp.create_dj_virtual_module("stimulus", "pipeline_stimulus")

config.register_externals()
config.register_adapters(context=locals())

schema = dj.schema(config.schema_name, create_schema=True)

# Utility mixins
class MakerMixin:
    def maker(self, key=None, return_list=False):
        rel = self if key is None else self & key
        makers = [
            getattr(self, m)
            for m in (dj.U(self.maker_name) & rel).fetch(self.maker_name)
            if hasattr(self, m)
        ]
        if len(makers) == 1:
            return makers[0]
        elif return_list:
            return makers
        raise Exception(
            "MakerMixin: Multiple or none makers found for:\n {}\nSet return_list to True if a list of makers is expected!".format(
                rel.__repr__
            )
        )


# Utility tables
@schema
class ScanSet(djp.Lookup):
    enable_hashing = True
    hash_name = "scan_set_hash"
    hashed_attrs = minnie_nda.Scan.primary_key
    hash_group = True
    definition = """
    scan_set_hash     : varchar(32)       #  unique identifier for the group
    ---
    name           : varchar(48)          #  name of the group
    description    : varchar(450)         #  description of the group
    n_members      : int                  #  number of members in the group
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class Member(djp.Part):
        definition = """
        -> master
        -> minnie_nda.Scan
        """


@schema
class ResponseType(djp.Lookup):
    definition = """
    # response type
    response_type         : varchar(32)
    ---
    description           : varchar(450)
    """


@schema
class StimType(djp.Manual):
    definition = """ 
    # stimulus type, entries must exist in the pipeline_stimulus.Condition table
    stimulus_type    : varchar(255)
    """


@schema
class StimTypeGrp(djp.Lookup):
    enable_hashing = True
    hash_name = "stim_type_grp_hash"
    hashed_attrs = StimType.primary_key
    hash_group = True
    definition = """
    stim_type_grp_hash     : varchar(32)          #  unique identifier for the group
    ---
    stim_types             : varchar(450)          #  list of stimulus types
    n_members              : int                  #  number of members in the group
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class Member(djp.Part):
        definition = """
        -> master
        -> StimType
        """


# Orientation
## Faithful copy of functional properties from external database
@schema
class OrientationDV11521GD(djp.Lookup):
    definition = """
    # Orientation tuning (global motion, discrete direction changes) extracted with the tuning pipeline within dynamic vision (version 1.1.5.2.1). 
    tuning_hash          : varchar(256)                 # unique identifier for tuning configuration
    response_hash        : varchar(256)                 # unique identifier for response configuration
    slice_set_hash       : varchar(256)                 # policy for defining groups of stimulus slices
    sample_hash          : varchar(256)                 # unique identifier for sample configuration
    bs_seed              : int                          # random seed for bootstrapping
    permute_seed         : int                          # random seed for permutation
    bs_samples           : int unsigned                 # number of bootstrap samples attempted
    permute_samples      : int unsigned                 # number of permutation samples attempted
    confidence_interval  : decimal(6,4)                 # confidence interval percentage
    ---
    tuning_curve_radians : longblob                     # [Data] directions of the stimuli in radians in ascending order
    n_radians            : int                          # [Data] number of directions
    n_samples            : longblob                     # [Data] number of response samples for each direction
    min_bs_times         : int                          # [Quality] minimum number of times the bootstrap test was run successfully
    min_permute_times    : int                          # [Quality] minimum number of times the permutation test was run successfully
    min_samples          : int                          # [Quality] minimum number of response samples for each direction
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        tuning_curve_mu      : longblob                     # [Data] mean responses over repeats of oriented stimuli, the corresponding direction of the stimuli is stored in the master table
        tuning_curve_sigma   : longblob                     # [Data] standard deviation of responses over repeats of oriented stimuli, the corresponding direction of the stimuli is stored in the master table
        mu                   : float                        # [Params] center of the first von Mises distribution. the center of the second is mu + pi. mu is in [0, pi). THIS IS NOT THE PREFERRED DIRECTION!
        phi                  : float                        # [Params] weight of the first von Mises distribution. the weight of the second is (1-phi)
        kappa                : float                        # [Params] dispersion of both von Mises distributions
        success              : tinyint                      # [Quality] success of the bounded minimization
        kuiper_v             : float                        # [Quality] statistics of the Kuiper GoF test between the empirical cdf and the bimodal von-Mises fit
        kuiper_v_uniform     : float                        # [Quality] statistics of the Kuiper GoF test between the empirical cdf and a uniform distribution
        bimodal              : tinyint                      # [Quality] whether the fitted BiVonMises distribution is bimodal
        sse_tuning           : float                        # [Quality] sum of squared error of predicted tuning curve and observed tuning curve, tuning curves are normalized to have sum of 1
        sse_frame            : float                        # [Quality] sum of squared error of predicted single frame responses and observed single frame responses, single frame responses are normalized to the same scale as the normalized tuning curve
        var_tuning           : float                        # [Quality] variance of the observed tuning curve, tuning curves are normalized to have sum of 1
        var_frame            : float                        # [Quality] variance of the observed single frame responses, single frame responses are normalized to the same scale as the normalized tuning curve
        mu_ci_lower          : float                        # [Quality] lower bound of confidence interval for mu
        mu_ci_upper          : float                        # [Quality] upper bound of confidence interval for mu
        phi_ci_lower         : float                        # [Quality] lower bound of confidence interval for phi
        phi_ci_upper         : float                        # [Quality] upper bound of confidence interval for phi
        kappa_ci_lower       : float                        # [Quality] lower bound of confidence interval for kappa
        kappa_ci_upper       : float                        # [Quality] upper bound of confidence interval for kappa
        bs_times             : int                          # [Quality] number of times the bootstrap test was run successfully
        mu_p                 : float                        # [Quality] p value for mu by permutation test
        phi_p                : float                        # [Quality] p value for phi by permutation test
        kappa_p              : float                        # [Quality] p value for kappa by permutation test
        fvu_tuning_diff      : float                        # [Quality] difference between the fraction of variance unexplained (tuning curve) of permuted fits and actual fit (mean of permuted fvu - actual fvu)
        fvu_tuning_p         : float                        # [Quality] p value for fraction of variance unexplained by permutation test (tuning curve)
        fvu_frame_diff       : float                        # [Quality] difference between the fraction of variance unexplained (single frame responses) of permuted fits and actual fit (mean of permuted fvu - actual fvu)
        fvu_frame_p          : float                        # [Quality] p value for fraction of variance unexplained by permutation test (single frame responses)
        permute_times        : int                          # [Quality] number of times the permutation test was run successfully
        """

    def pref_ori(self, unit_key=None):
        unit_key = {} if unit_key is None else unit_key
        return (
            (
                dj.U(*minnie_nda.UnitSource.primary_key, "pref_ori")
                & (self.Unit & self & unit_key).proj(pref_ori="mu")
            )
            .fetch(format="frame")
            .reset_index()
        )

    def selectivity(self, unit_key=None, percentile=True):
        unit_key = {} if unit_key is None else unit_key
        df = (
            (
                dj.U(*minnie_nda.UnitSource.primary_key, "selectivity")
                & (self.Unit & self & unit_key).proj(selectivity="kappa")
            )
            .fetch(format="frame")
            .reset_index()
        )
        if percentile:
            df["selectivity"] = df["selectivity"].rank(pct=True)
        return df


@schema
class OrientationDV231042(djp.Manual):
    definition = """
    # Orientation tuning extracted with the tuning pipeline within dynamic vision (version 2.3.10.4.2)
    animal_id            : int                          # id number
    scan_session         : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    timing_id            : tinyint                      # align neural and stimulus timing
    behavior_id          : tinyint                      # behavior resampling
    response_id          : tinyint                      # neural response resampling
    valid_id             : tinyint                      # neural response resampling
    unique_id            : tinyint                      # unique unit parameters
    pipe_version         : smallint                     # 
    segmentation_method  : tinyint                      # 
    direction_hash       : varchar(256)                 # unique identifier for direction configuration
    ---
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        success              : tinyint                      # success of least squares optimization
        mu                   : float                        # center of the first von Mises distribution, this is the preferred direction
        phi                  : float                        # weight of the first von Mises distribution
        kappa                : float                        # dispersion of both von Mises distributions
        scale                : float                        # von Mises amplitude
        bias                 : float                        # uniform amplitude
        bvm_mse              : float                        # mean squared error
        osi                  : float                        # orientation selectivity index
        dsi                  : float                        # direction selectivity index
        amp                  : float                        # amplitude
        uniform_mse          : float                        # mean squared error
        """

    def pref_ori(self, unit_key=None, clock_convention=True):
        unit_key = {} if unit_key is None else unit_key
        unit_df = (
            (
                dj.U(*minnie_nda.UnitSource.primary_key, "pref_ori")
                & (self.Unit & self & unit_key).proj(pref_ori="mu")
            )
            .fetch(format="frame")
            .reset_index()
        )
        if clock_convention:
            unit_df["pref_ori"] = (
                -unit_df["pref_ori"] + np.pi / 2
            ) % np.pi  # convert to clock convention (horizontal bar moving upward is 0 and orientation increases clockwise)
        else:
            unit_df["pref_ori"] = unit_df["pref_ori"] % np.pi
        return unit_df

    def selectivity(self, unit_key=None, percentile=True):
        unit_key = {} if unit_key is None else unit_key
        df = (
            (
                dj.U(*minnie_nda.UnitSource.primary_key, "selectivity")
                & (self.Unit & self & unit_key).proj(selectivity="osi")
            )
            .fetch(format="frame")
            .reset_index()
        )
        if percentile:
            df["selectivity"] = df["selectivity"].rank(pct=True)
        return df

    def tuning(self, unit_key=None, convention="ori"):
        tuning_dir = djp.create_dj_virtual_module(
            "dv_tunings_v4_direction", "dv_tunings_v4_direction"
        )
        unit_direction_df = (
            (
                tuning_dir.DirectionResponse.Unit * tuning_dir.DirectionConfig.Mean
                & self
                & unit_key
            )
            .fetch(format="frame")
            .reset_index()
        )
        unit_df = (
            unit_direction_df.groupby(["session", "scan_idx", "unit_id"])
            .apply(lambda df: df.sort_values("direction").direction.values)
            .to_frame(name="direction")
        )
        unit_df["response_mean"] = (
            unit_direction_df.groupby(["session", "scan_idx", "unit_id"])
            .apply(lambda df: df.sort_values("direction").response_mean.values)
            .to_frame()
        )
        return unit_df


## Aggregation tables
@schema
class Orientation(djp.Lookup):
    hash_part_table_names = True
    hash_name = "orientation_hash"
    definition = """
    # orientation
    orientation_hash    : varchar(32)
    ---
    orientation_type    : varchar(48)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    def part_table(self, key=None):
        key = self.fetch("KEY") if key is None else (self & key).fetch("KEY")
        part = (self & key).fetch("orientation_type")
        part = set(part)
        assert len(part) == 1
        part = part.pop()
        part = getattr(self, part)
        part_key = (part & key).fetch()
        return part & part_key

    def _pref_ori(self, key=None, unit_key=None):
        key = self.fetch("KEY") if key is None else (self & key).fetch("KEY")
        unit_key = {} if unit_key is None else unit_key
        return (self & key).part_table()._pref_ori(unit_key=unit_key)

    def _selectivity(self, key=None, unit_key=None, percentile=False):
        key = self.fetch("KEY") if key is None else (self & key).fetch("KEY")
        unit_key = {} if unit_key is None else unit_key
        return (
            (self & key)
            .part_table()
            ._selectivity(unit_key=unit_key, percentile=percentile)
        )

    class DV11521GD(djp.Part):
        _source = "OrientationDV11521GD"
        source = eval(_source)
        enable_hashing = True
        hash_name = "orientation_hash"
        hashed_attrs = source.primary_key
        definition = """
        #
        -> master
        ---
        -> OrientationDV11521GD
        """

        def _pref_ori(self, key=None, unit_key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            unit_key = {} if unit_key is None else unit_key
            return (self.source & key).pref_ori(unit_key=unit_key)

        def _selectivity(self, key=None, unit_key=None, percentile=False):
            key = self.fetch() if key is None else (self & key).fetch()
            unit_key = {} if unit_key is None else unit_key
            return (self.source & key).selectivity(
                unit_key=unit_key, percentile=percentile
            )

    class DV231042(djp.Part):
        _source = "OrientationDV231042"
        source = eval(_source)
        enable_hashing = True
        hash_name = "orientation_hash"
        hashed_attrs = source.primary_key
        definition = """
        #
        -> master
        ---
        -> OrientationDV231042
        """

        def _pref_ori(self, key=None, unit_key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            unit_key = {} if unit_key is None else unit_key
            return (self.source & key).pref_ori(unit_key=unit_key)

        def _selectivity(self, key=None, unit_key=None, percentile=False):
            key = self.fetch() if key is None else (self & key).fetch()
            unit_key = {} if unit_key is None else unit_key
            return (self.source & key).selectivity(
                unit_key=unit_key, percentile=percentile
            )


@schema
class OrientationScanInfo(djp.Computed):
    definition = """ # Provide useful information about the orientation tuning
    -> Orientation
    ---
    -> StimTypeGrp
    -> ResponseType
    -> minnie_nda.Scan
    stimulus_length   : float    # length of stimulus in seconds
    """

    def pref_ori(self, unit_key=None):
        # return the preferred orientation of requested units, return all units by default
        unit_key = {} if unit_key is None else unit_key
        assert minnie_nda.Scan().aggr(self, count="count(*)").fetch("count").max() == 1
        return (Orientation & self)._pref_ori(unit_key=unit_key)

    def selectivity(self, unit_key=None, percentile=False):
        unit_key = {} if unit_key is None else unit_key
        assert minnie_nda.Scan().aggr(self, count="count(*)").fetch("count").max() == 1
        return (Orientation & self)._selectivity(
            unit_key=unit_key, percentile=percentile
        )


@schema
class OrientationScanSet(djp.Lookup):
    enable_hashing = True
    hash_name = "ori_scan_set_hash"
    hashed_attrs = OrientationScanInfo.primary_key
    hash_group = True
    definition = """ # A group of orientation with the same response type and stimulus type
    ori_scan_set_hash      : varchar(32)
    ---
    -> ScanSet
    -> ResponseType
    -> StimTypeGrp
    description            : varchar(128)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class Member(djp.Part):
        definition = """
        -> master
        -> OrientationScanInfo
        """

    @property
    def info(self):
        return (
            self
            * StimTypeGrp().proj(
                ...,
                stim_type_grp_ts="timestamp",
                n_stim_type="n_members",
            )
            * ResponseType().proj(..., response_type_desc="description")
            * ScanSet().proj(
                ...,
                scan_set_ts="timestamp",
                n_scan="n_members",
                scan_set_desc="description",
            )
        )

    def pref_ori(self, unit_key=None):
        return (OrientationScanInfo & (self * self.Member).proj()).pref_ori(
            unit_key=unit_key
        )

    def selectivity(self, unit_key=None, percentile=False):
        return (OrientationScanInfo & (self * self.Member).proj()).selectivity(
            unit_key=unit_key, percentile=percentile
        )


@schema
class OrientationFilter(djp.Lookup):
    definition = """
    ori_filter_id           : int           # id for an orientation filter
    ---
    note                    : varchar(128)  # note for the orientation filter
    """
    contents = [
        [1, "horizontal cardinal (phi < pi/4 or pi >= pi*3/4)"],
        [2, "vertical cardinal (pi/4 <= phi < pi*3/4)"],
    ]

    def get_filter(self, key=None):
        key = self.fetch1() if key is None else (self & key).fetch1()
        if key["ori_filter_id"] == 1:
            return lambda x: x < np.pi / 4 or np.pi * 3 / 4 <= x
        elif key["ori_filter_id"] == 2:
            return lambda x: np.pi / 4 <= x and x < np.pi * 3 / 4


# Oracle
## Faithful copy of data
@schema
class OracleDVScan1(djp.Manual):
    definition = """
    # oracle value computed with dynamic vision scan v1
    animal_id            : int                          # id number
    scan_session         : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    preprocess_hash      : varchar(128)                 # scan and stimulus preprocessing
    trial_hash           : varchar(256)                 # unique identifier for trial configuration
    slice_set_hash       : varchar(256)                 # policy for defining groups of stimulus slices
    tier                 : varchar(20)                  # data tier
    min_trials_per_slice : int unsigned                 # minimum number of trials per slice
    first_response       : int unsigned                 # first sampled response index
    statistic_type       : varchar(64)                  # statistic type
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        statistic       : float                         # statistic summarizing relationship between trial and oracle
        """

    def scan(self, key=None):
        key = self.fetch("KEY") if key is None else (self & key).fetch("KEY")
        return (self & key).fetch("animal_id", "scan_session", "scan_idx")

    def oracle(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return (
            dj.U(*minnie_nda.UnitSource.heading.primary_key, "statistic")
            & (self.Unit & key)
        ).proj(oracle="statistic")


@schema
class OracleDVScan3(djp.Manual):
    definition = """
    # oracle value computed with dynamic vision scan v1
    animal_id            : int                          # id number
    scan_session         : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    timing_id            : tinyint                      # align neural and stimulus timing
    behavior_id          : tinyint                      # behavior resampling
    response_id          : tinyint                      # neural response resampling
    valid_id             : tinyint                      # neural response resampling
    trial_group_hash     : varchar(256)                 # unique identifier for trial group grouping
    downsample           : int unsigned                 # response downsampling
    start_index          : int unsigned                 # index of first sampled response
    statistic_type       : varchar(64)                  # statistic type
    segmentation_method  : tinyint                      # segmentation method
    pipe_version         : smallint                     # 
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        statistic       : float                         # statistic summarizing relationship between trial and oracle
        """

    def scan(self, key=None):
        key = self.fetch("KEY") if key is None else (self & key).fetch("KEY")
        return (self & key).fetch("animal_id", "scan_session", "scan_idx")

    def oracle(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return (
            dj.U(*minnie_nda.UnitSource.heading.primary_key, "statistic")
            & (self.Unit & key)
        ).proj(oracle="statistic")


@schema
class OracleTuneMovieOracle(djp.Manual):
    definition = """
    # oracle score imported from pipeline_tune.__movie_oracle__total
    animal_id            : int                          # id number
    scan_session         : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    pipe_version         : smallint                     # 
    segmentation_method  : tinyint                      # 
    spike_method         : tinyint                      # spike inference method
    """

    class Unit(djp.Part):
        definition = """
        -> minnie_nda.UnitSource
        ---
        trials               : int                          # number of trials used
        pearson              : float                        # per unit oracle pearson correlation over all movies
        """

    def scan(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return (self & key).fetch("animal_id", "scan_session", "scan_idx")

    def oracle(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return (
            dj.U(*minnie_nda.UnitSource.heading.primary_key, "pearson")
            & (self.Unit & key)
        ).proj(oracle="pearson")

@schema
class Nn10Reliability(djp.Manual):
    definition = """
    # reliability score computed with dynamic vision nns v10
    animal_id            : int                          # id number
    scan_session         : smallint                     # session index for the mouse
    scan_idx             : smallint                     # number of TIFF stack file
    stimulus_to_response : smallint unsigned            # ratio of stimulus hz to response hz
    start_index          : smallint unsigned            # index of first sampled response
    response_hash        : varchar(256)                 # unique identifier for response configuration
    transform_hash       : varchar(256)                 # unique identifier for transform configuration
    reliability_hash     : varchar(256)                 # unique identifier for reliability configuration
    trial_group_hash     : varchar(256)                 # unique identifier for trial group grouping  
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        reliability        : float                        # unit reliability score
        """
    
    def scan(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return (self & key).fetch("animal_id", "scan_session", "scan_idx")

    def oracle(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return (
            dj.U(*minnie_nda.UnitSource.heading.primary_key, "reliability")
            & (self.Unit & key)
        ).proj(oracle="reliability")

## Aggregation tables
@schema
class Oracle(djp.Lookup):
    hash_part_table_names = True
    hash_name = "oracle_hash"
    definition = """
    # oracle
    oracle_hash    : varchar(32)
    ---
    oracle_type    : varchar(48)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    def scan(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return self.r1p(key).scan()

    def oracle(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        return self.r1p(key).oracle()

    class DVScan1(djp.Part):
        _source = "OracleDVScan1"
        source = eval(_source)
        enable_hashing = True
        hash_name = "oracle_hash"
        hashed_attrs = source.primary_key
        definition = """
        #
        -> master
        ---
        -> OracleDVScan1
        """

        def scan(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).scan()

        def oracle(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).oracle()

    class DVScan3(djp.Part):
        _source = "OracleDVScan3"
        source = eval(_source)
        enable_hashing = True
        hash_name = "oracle_hash"
        hashed_attrs = source.primary_key
        definition = """
        #
        -> master
        ---
        -> OracleDVScan3
        """

        def scan(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).scan()

        def oracle(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).oracle()

    class TuneMovieOracle(djp.Part):
        _source = "OracleTuneMovieOracle"
        source = eval(_source)
        enable_hashing = True
        hash_name = "oracle_hash"
        hashed_attrs = source.primary_key
        definition = """
        #
        -> master
        ---
        -> OracleTuneMovieOracle
        """

        def scan(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).scan()

        def oracle(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).oracle()
    
    class Nn10Reliability(djp.Part):
        _source = "Nn10Reliability"
        source = eval(_source)
        enable_hashing = True
        hash_name = "oracle_hash"
        hashed_attrs = source.primary_key
        definition = """
        #
        -> master
        ---
        -> OracleTuneMovieOracle
        """

        def scan(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).scan()

        def oracle(self, key=None):
            key = self.fetch() if key is None else (self & key).fetch()
            return (self.source & key).oracle()


@schema
class OracleScanInfo(djp.Computed):
    definition = """ # Provide useful information about the oracle scores
    -> Oracle
    ---
    -> minnie_nda.Scan
    """


@schema
class OracleScanSet(djp.Lookup):
    enable_hashing = True
    hash_name = "oracle_scan_set_hash"
    hashed_attrs = OracleScanInfo.primary_key
    hash_group = True
    definition = """ # A group of orientation with the same response type and stimulus type
    oracle_scan_set_hash      : varchar(32)
    ---
    -> ScanSet
    description            : varchar(128)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class Member(djp.Part):
        definition = """
        -> master
        -> OracleScanInfo
        """

    def oracle(self, key=None):
        key = self.fetch() if key is None else (self & key).fetch()
        oracle_key = (
            (self.Member * Oracle.proj(*list(set(Oracle.heading) - {"timestamp"})))
            & key
        ).fetch()
        # assume all members of a set have the same oracle type
        assert len({key["oracle_type"] for key in oracle_key}) == 1
        return Oracle.r1p(oracle_key).oracle()


# # Predictive model performance and parameters
## Aggregation tables
@schema
class DynamicModel(djp.Lookup, MakerMixin):
    hash_part_table_names = True
    hash_name = "dynamic_model_hash"
    maker_name = "dynamic_model_type"
    definition = """
    # dynamic predictive models
    -> minnie_nda.Scan
    dynamic_model_hash    : varchar(32)
    ---
    dynamic_model_type    : varchar(48)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    def readout(self, part_key=None):
        return (self.maker() & self).readout(part_key)

    def readout_location(self, part_key=None):
        return (self.maker() & self).readout_location(part_key)

    class NnsV5(djp.Part):
        definition = """
        # dynamic model saved in `dv_nns_v5_scan.__scan_model`
        -> master
        ---
        scan_hash            : varchar(256)                 # configuration of scan dataset
        model_hash           : varchar(256)                 # unique identifier for model configuration
        instance_hash        : varchar(128)                 # model instance
        """
        enable_hashing = True
        hash_name = "dynamic_model_hash"
        hashed_attrs = [
            "scan_hash",
            "model_hash",
            "instance_hash",
            "animal_id",
            "scan_session",
            "scan_idx",
        ]

        def readout(self, part_key=None):
            part_key = {} if part_key is None else part_key
            return DynamicModel.NnsV5UnitReadout & self & part_key
        
        def readout_location(self, part_key=None):
            raise NotImplementedError

    class NnsV5UnitReadout(djp.Part):
        definition = """
        # readout saved in `dv_nns_v5_scan.__readout`
        -> DynamicModel.NnsV5
        -> minnie_nda.UnitSource
        ---
        ro_x                 : float                        # readout x coordinate
        ro_y                 : float                        # readout y coordinate
        ro_weight            : longblob                     # readout weight, [head, feature]
        """

    class NnsV10ScanV3Unique(djp.Part):
        definition = """
        # dynamic model saved in `dv_nns_v5_scan.__scan_model`
        -> master
        ---
        scan_hash            : varchar(256)                 # configuration of scan dataset
        nn_hash              : varchar(256)                 # configuration of neural network
        instance_hash        : varchar(128)                 # nn instance configuration
        unique_id            : varchar(128)                 # unique unit parameters
        """
        enable_hashing = True
        hash_name = "dynamic_model_hash"
        hashed_attrs = [
            "scan_hash",
            "nn_hash",
            "instance_hash",
            "unique_id",
            "animal_id",
            "scan_session",
            "scan_idx",
            "dynamic_model_type",
        ]

        def readout(self, part_key=None):
            part_key = {} if part_key is None else part_key
            return DynamicModel.NnsV10ScanV3UniqueUnitReadout & self & part_key
        
        def readout_location(self, part_key=None):
            scan3_perspective = dj.FreeTable(
                dj.conn(), '`dv_nns_v10_scan`.`__perspective__unit`'
            ).proj(..., scan_session='session')
            return scan3_perspective & self

    class NnsV10ScanV3UniqueUnitReadout(djp.Part):
        definition = """
        # readout saved in `dv_nns_v10_scan.__readout`
        -> DynamicModel.NnsV10ScanV3Unique
        -> minnie_nda.UnitSource
        ---
        ro_x                 : float                        # readout x coordinate
        ro_y                 : float                        # readout y coordinate
        ro_weight            : longblob                     # readout weight, [head, feature]
        """

    class NnsV10ScanV3All(djp.Part):
        definition = """
        # dynamic model saved in `dv_nns_v5_scan.__scan_model`
        -> master
        ---
        scan_hash            : varchar(256)                 # configuration of scan dataset
        nn_hash              : varchar(256)                 # configuration of neural network
        instance_hash        : varchar(128)                 # nn instance configuration
        """
        enable_hashing = True
        hash_name = "dynamic_model_hash"
        hashed_attrs = [
            "scan_hash",
            "nn_hash",
            "instance_hash",
            "animal_id",
            "scan_session",
            "scan_idx",
            "dynamic_model_type",
        ]

        def readout(self, part_key=None):
            part_key = {} if part_key is None else part_key
            return DynamicModel.NnsV10ScanV3AllUnitReadout & self & part_key
        def readout_location(self, part_key=None):
            scan3_perspective = dj.FreeTable(
                dj.conn(), '`dv_nns_v10_scan`.`__perspective__unit`'
            ).proj(..., scan_session='session')
            return scan3_perspective & self

    class NnsV10ScanV3AllUnitReadout(djp.Part):
        definition = """
        # readout saved in `dv_nns_v10_scan.__readout`
        -> DynamicModel.NnsV10ScanV3All
        -> minnie_nda.UnitSource
        ---
        ro_x                 : float                        # readout x coordinate
        ro_y                 : float                        # readout y coordinate
        ro_weight            : longblob                     # readout weight, [head, feature]
        """


@schema
class DynamicModelScore(djp.Lookup, MakerMixin):
    hash_part_table_names = True
    hash_name = "dynamic_score_hash"
    maker_name = "dynamic_score_type"
    definition = """
    # dynamic predictive models
    -> DynamicModel
    dynamic_score_hash    : varchar(32)
    ---
    dynamic_score_type    : varchar(48)
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    def unit_score(self, part_key=None):
        return (self.maker() & self).unit_score(part_key)

    class NnsV5(djp.Part):
        definition = """
        # Predictive model performance and parameters
        -> master
        ---
        slice_set_hash       : varchar(256)                 # policy for defining groups of stimulus slices
        tier                 : varchar(20)                  # data tier
        min_trials_per_slice : int unsigned                 # minimum number of trials per slice
        statistic_type       : varchar(64)                  # statistic type
        behavior_hash        : varchar(256)                 # nn behavior configuration
        eye_position         : bool                         # use eye position data
        behavior             : bool                         # use behavioral data
        """
        enable_hashing = True
        hash_name = "dynamic_score_hash"
        hashed_attrs = [
            "slice_set_hash",
            "tier",
            "min_trials_per_slice",
            "statistic_type",
            "behavior_hash",
        ]

        def unit_score(self, part_key=None):
            part_key = {} if part_key is None else part_key
            return (DynamicModelScore.NnsV5UnitScore * self) & part_key

    class NnsV5UnitScore(djp.Part):
        definition = """
        -> DynamicModelScore.NnsV5
        -> minnie_nda.UnitSource
        ---
        statistic            : float                        # statistic summarizing relationship between scan and behavioral model responses
        """

    class NnsV10ScanV3Unique(djp.Part):
        definition = """
        # Predictive model performance and parameters
        -> master
        ---
        trial_group_hash       : varchar(256)                 # policy for defining groups of stimulus slices
        behavior_hash          : varchar(256)                 # nn behavior configuration
        eye_position           : bool                         # use eye position data
        behavior               : bool                         # use behavioral data
        statistic_type         : varchar(64)                  # statistic type
        """
        enable_hashing = True
        hash_name = "dynamic_score_hash"
        hashed_attrs = [
            "trial_group_hash",
            "statistic_type",
            "behavior_hash",
            "dynamic_score_type",
        ]

        def unit_score(self, part_key=None):
            part_key = {} if part_key is None else part_key
            return (DynamicModelScore.NnsV10ScanV3UniqueUnitScore * self) & part_key

    class NnsV10ScanV3UniqueUnitScore(djp.Part):
        definition = """
        -> DynamicModelScore.NnsV10ScanV3Unique
        -> minnie_nda.UnitSource
        ---
        statistic            : float                        # statistic summarizing relationship between scan and behavioral model responses
        """


@schema
class DynamicModelScanSet(djp.Lookup):
    enable_hashing = True
    hash_name = "dynamic_model_scan_set_hash"
    hashed_attrs = DynamicModel.primary_key
    hash_group = True
    definition = """
    # a set of dynamic predictive models
    dynamic_model_scan_set_hash    : varchar(32)
    ---
    -> ScanSet
    name           : varchar(48)          #  name of the group
    description    : varchar(450)         #  description of the group
    n_members      : int                  #  number of members in the group
    timestamp=CURRENT_TIMESTAMP : timestamp
    """

    class Member(djp.Part):
        definition = """
        # Predictive model performance and parameters
        -> master
        -> DynamicModel
        """

    def readout(self, part_key=None):
        master_key = (
            (DynamicModel & (self.Member().proj() & self.proj())).proj().fetch()
        )
        return (DynamicModel & master_key).readout(part_key)

    def unit_score(self, part_key=None):
        master_key = (
            (DynamicModel & (self.Member().proj() & self.proj())).proj().fetch()
        )
        return (DynamicModelScore & master_key).unit_score(part_key)

    def readout_location(self, part_key=None):
        master_key = (
            (DynamicModel & (self.Member().proj() & self.proj())).proj().fetch()
        )
        return (DynamicModel & master_key).readout_location(part_key)


# WIP
@schema
class RespArrNnsV10(djp.Manual):
    definition = """
    resp_array_idx          : int unsigned
    ---
    -> DynamicModelScanSet
    resp_array              : blob@resp_array
    description             : varchar(255)          # description of the response array
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        row_idx             : int unsigned          # row index in the response array
        """

    class Condition(djp.Part):
        stimulus = djp.create_dj_virtual_module(
            "pipeline_stimulus", "pipeline_stimulus"
        )
        definition = """
        -> master
        -> djp.create_dj_virtual_module('pipeline_stimulus', 'pipeline_stimulus').Condition
        ---
        col_idx_start             : int unsigned     # start index of the condition in the response array
        col_idx_end               : int unsigned     # end index of the condition in the response array
        """


@schema
class RespCorr(djp.Lookup):
    hash_part_table_names = True
    hash_name = "resp_corr_hash"
    definition = """
    resp_corr_hash                  : varchar(32)
    ---
    -> ScanSet
    resp_corr_type                  : varchar(48)
    resp_corr_ts=CURRENT_TIMESTAMP  : timestamp
    """

    class RespArrNnsV10(djp.Part):
        definition = """
        -> master
        ---
        -> RespArrNnsV10
        """
        enable_hashing = True
        hash_name = "resp_corr_hash"
        hashed_attrs = [
            "resp_array_idx",
        ]

        @property
        def resp_array(self):
            key = self.fetch1("KEY")
            if not (
                hasattr(self.__class__, "_cache_key")
                and self.__class__._cache_key == key
            ):
                self.__class__._cache_resp_array = (RespArrNnsV10 & self).fetch1(
                    "resp_array"
                )
                self.__class__._cache_key = key
            return self.__class__._cache_resp_array

        def get_resp(self, unit_df):
            unit_index_df = (
                (RespArrNnsV10.Unit & self).fetch(format="frame").reset_index()
            )
            row_idx = unit_df.merge(unit_index_df, how="left")[
                ["row_idx"]
            ].values.squeeze()
            return self.resp_array[row_idx, :]

        def get_corr(self, unit_df1, unit_df2, max_batch_size=100_000):
            # unit_df = (RespArrNnsV10.Unit & self).fetch(format='frame').reset_index()
            # row_idx1 = unit_df1.merge(unit_df, how='left')[['row_idx']].values.squeeze()
            # row_idx2 = unit_df2.merge(unit_df, how='left')[['row_idx']].values.squeeze()
            # assert not np.isnan(row_idx1).any() or not np.isnan(row_idx2).any(), "units not found in the response array"
            assert len(unit_df1) == len(
                unit_df2
            ), "unit_df1 and unit_df2 must have the same length"
            corr = []
            unit_df1_batch = np.array_split(
                unit_df1, np.ceil(len(unit_df1) / max_batch_size)
            )
            unit_df2_batch = np.array_split(
                unit_df2, np.ceil(len(unit_df2) / max_batch_size)
            )
            for _unit_df1, _unit_df2 in zip(
                tqdm(unit_df1_batch, desc="batch"), unit_df2_batch
            ):
                resp_1 = self.get_resp(_unit_df1)
                resp_2 = self.get_resp(_unit_df2)
                corr.append(pcorr(resp_1, resp_2))
            corr = np.concatenate(corr)
            assert len(corr) == len(unit_df1)
            return corr

        def get_xcorr(self, unit_df):
            resp = self.get_resp(unit_df)
            return xcorr(resp)

    def get_corr(self, unit_df1, unit_df2):
        return self.r1p(self).get_corr(unit_df1, unit_df2)


@schema
class RunningID(djp.Lookup):
    definition = """
    running_id          : int unsigned
    ---
    description         : varchar(255)
    """
    contents = [
        (
            1,
            "minimum speed threshold = 0.5 cm/s, fetch data from dv scans v3 with timing_id=0, behavior_id=1, valid_id=0",
        ),
    ]

    @staticmethod
    def compute_running_time(scan_key, running_id):
        if running_id == 1:
            min_velocity_threshold = 0.5
            scan = djp.create_dj_virtual_module("scan", "dv_scans_v3_scan")
            scan_dataset = djp.create_dj_virtual_module(
                "scan_dataset", "dv_scans_v3_scan_dataset"
            )
            stimulus = djp.create_dj_virtual_module(
                "pipeline_stimulus", "pipeline_stimulus"
            )
            rel = (
                scan.Behavior.Trial
                * scan.Response.Trial
                * scan.Valid.Trial
                * scan.TimingInfo
                * stimulus.Trial
                * stimulus.Condition
                & stimulus.Clip
                & "valid"
            ).proj(..., scan_session="session") & scan_key
            behavior, hz = (rel & scan_key).fetch("behavior", "hz")
            treadmill = np.concatenate(behavior)[:, 2]
            hz = float(np.unique(hz))
            running_idx = np.nonzero(treadmill > min_velocity_threshold)[0]
            running_sec = len(running_idx) / hz
            total_sec = len(treadmill) / hz
            return running_sec, total_sec
        else:
            raise NotImplementedError


@schema
class Running(djp.Computed):
    definition = """
    -> RunningID
    -> minnie_nda.Scan
    ---
    running_time       : float                 # total running time in seconds
    scan_time          : float                 # total scan time included in the analysis in seconds
    """

    @property
    def key_source(self):
        return RunningID * minnie_nda.Scan & ScanSet.Member

    def make(self, key):
        running_sec, total_sec = RunningID.compute_running_time(
            key, running_id=key["running_id"]
        )
        self.insert1({**key, "running_time": running_sec, "scan_time": total_sec})


scan3 = djp.create_dj_virtual_module("scan", "dv_scans_v3_scan")


@datajoint_utils.config_table(schema, config_type="loc_corr")
class LocCorrConfig(djp.Lookup):
    """
    Configuration for computing behavior modulation for single neurons
    """

    class Scan3(djp.Part):
        definition = """
        -> master
        ---
        -> scan3.TimingId
        -> scan3.BehaviorId
        -> scan3.ResponseId
        -> scan3.ValidId
        binning_window    : decimal(4,2)            # binning window in seconds
        n_perm            : int unsigned            # number of permutations
        seed              : int unsigned            # random seed
        """
        content = [
            {
                "binning_window": decimal.Decimal("0.50"),
                "timing_id": 0,
                "behavior_id": 0,
                "response_id": 1,
                "valid_id": 0,
                "n_perm": 1000,
                "seed": 0,
            },
        ]

        def load_data(self, scan_key):
            params = self.fetch1()
            binning_window = float(self.fetch1("binning_window"))
            rel = (
                scan3.Behavior.Trial
                * scan3.Response.Trial
                * scan3.Valid.Trial
                * scan3.TimingInfo
                * stimulus.Trial
                * stimulus.Condition
                & stimulus.Clip
                & params
                & "valid"
            ).proj(..., scan_session="session") & scan_key
            data = pd.DataFrame(
                (rel & scan_key).fetch(
                    "trial_idx",
                    "animal_id",
                    "scan_session",
                    "scan_idx",
                    "response",
                    "behavior",
                    "hz",
                    as_dict=True,
                )
            )
            # restrict behavior to treadmill only
            data["behavior"] = data["behavior"].apply(lambda b: b[:, 2])
            data["treadmill"] = data["behavior"]
            del data["behavior"]
            # bin treadmill and response
            hz = float(np.unique(data["hz"].values))
            w = int(np.ceil(binning_window * hz))
            data["treadmill"] = data["treadmill"].apply(
                lambda x: np.mean(x[: int(len(x) // w * w)].reshape(-1, w), axis=1)
            )
            data["response"] = data["response"].apply(
                lambda x: np.mean(
                    x[: int(len(x) // w * w), :].reshape(-1, w, x.shape[1]), axis=1
                )
            )
            unit_df = pd.DataFrame(
                (scan3.Unit().proj(..., scan_session="session") & scan_key).fetch(
                    as_dict=True, order_by="response_index"
                )
            )
            return data, unit_df 

        def compute(self, scan_key):
            params = self.fetch1()
            data, unit_df = self.load_data(scan_key)
            treadmill = np.concatenate(data[f"treadmill"].values)
            response = np.concatenate(data[f"response"].values)
            treadmill = np.repeat(treadmill[None,:], response.shape[1], axis=0)
            unit_df[f"beh_mod"], unit_df[f"p_permute"] = pcorr_p(
                treadmill,
                response.T,
                n_perm=params["n_perm"],
                rng=np.random.default_rng(params["seed"]),
            )
            return unit_df


@schema
class LocCorr(djp.Computed):
    definition = """
    -> LocCorrConfig
    -> minnie_nda.Scan
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        loc_corr           : float                 # correlation between running and response
        p_permute          : float                 # p-value from permutation test
        """

    key_source = LocCorrConfig * minnie_nda.Scan & ScanSet.Member

    def make(self, key):
        unit_df = (LocCorrConfig & key).part_table().compute(scan_key=key)
        self.insert1(key)
        self.Unit().insert(
            [{**d, **key} for d in unit_df.to_dict("records")], ignore_extra_fields=True
        )


@datajoint_utils.config_table(schema, config_type="noise_corr")
class NoiseCorrConfig(djp.Lookup):
    class Scan3OracleCond(djp.Part):
        definition = """
        -> master
        ---
        -> scan3.TimingId
        -> scan3.BehaviorId
        -> scan3.ResponseId
        -> scan3.ValidId
        -> stimulus.Condition
        binning_window    : decimal(4,2)            # binning window in seconds
        n_perm            : int unsigned            # number of permutations
        seed              : int unsigned            # random seed
        """
        content = [
            {
                **d,
                "binning_window": decimal.Decimal("0.50"),
                "n_perm": 1000,
                "seed": 0,
                "timing_id": 0,
                "behavior_id": 0,
                "response_id": 1,
                "valid_id": 0,
            }
            for d in (stimulus.Clip & netflix.OracleSet()).fetch(
                "condition_hash", as_dict=True
            )
        ]

        def compute(self, scan_key):
            params = self.fetch1()
            binning_window = float(params["binning_window"])

            unit_df = pd.DataFrame(
                (scan3.Unit & params & scan_key)
                .proj(..., scan_session="session")
                .fetch(
                    "animal_id",
                    "scan_session",
                    "scan_idx",
                    "unit_id",
                    "response_index",
                    order_by="response_index",
                    as_dict=True,
                )
            )
            unit_df["corr_idx"] = unit_df["response_index"]

            oracle_trials = (
                (
                    scan3.Response.Trial
                    * scan3.Valid.Trial
                    * scan3.TimingId
                    * scan3.TimingInfo
                    * stimulus.Trial
                    * stimulus.Condition
                ).proj(..., scan_session="session")
                & params
                & "valid"
                & scan_key
            )
            response, hz = oracle_trials.fetch("response", "hz")
            assert len(np.unique(hz)) == 1
            hz = hz[0]
            w = int(np.ceil(binning_window * hz))
            # bin response
            response = [
                np.mean(r[: len(r) // w * w].reshape(-1, w, r.shape[1]), axis=1)
                for r in response
            ]
            response = np.stack(response, axis=0)  # n_trials x n_bins x n_units
            # get residue from mean
            response = response - np.mean(response, axis=0, keepdims=True)
            # z-score single trial residues
            response_z = stats.zscore(response, axis=1)
            response_z = response_z.reshape(
                -1, response_z.shape[-1]
            )  # n_trials*n_bins x n_units
            rho, p = function_utils.xcorr_p(response_z.T, n_perm=params["n_perm"], rng=np.random.default_rng(params["seed"]))
            rho, p = squareform(rho, checks=False), squareform(p, checks=False)

            return unit_df, rho, p

    class Scan3Oracle(djp.Part):
        definition = """
        -> master
        ---
        -> scan3.TimingId
        -> scan3.BehaviorId
        -> scan3.ResponseId
        -> scan3.ValidId
        binning_window    : decimal(4,2)            # binning window in seconds
        min_repeat        : int                     # minimum number of repeats to be considered
        n_perm            : int unsigned            # number of permutations
        seed              : int unsigned            # random seed
        """
        content = [
            {
                "binning_window": decimal.Decimal("0.50"),
                "timing_id": 0,
                "behavior_id": 0,
                "response_id": 1,
                "valid_id": 0,
                "min_repeat": 4,
                "n_perm": 1000,
                "seed": 0,
            },
        ]

        def compute(self, scan_key):
            params = self.fetch1()
            binning_window = float(params["binning_window"])

            unit_df = pd.DataFrame(
                (scan3.Unit & params & scan_key)
                .proj(..., scan_session="session")
                .fetch(
                    "animal_id",
                    "scan_session",
                    "scan_idx",
                    "unit_id",
                    "response_index",
                    order_by="response_index",
                    as_dict=True,
                )
            )
            unit_df["corr_idx"] = unit_df["response_index"]

            oracle_trial = (
                scan3.Response.Trial
                * scan3.Valid.Trial
                * scan3.TimingId
                * scan3.TimingInfo
                * stimulus.Trial
                * stimulus.Condition
                & "valid"
                & params
                & scan_key
            ).proj(..., scan_session="session")
            oracle_trial_df = pd.DataFrame(
                oracle_trial.fetch(
                    "animal_id",
                    "scan_session",
                    "scan_idx",
                    "condition_hash",
                    "response",
                    "hz",
                    as_dict=True,
                )
            )
            # bin to 500ms
            assert len(oracle_trial_df.hz.unique()) == 1
            hz = oracle_trial_df.hz.unique()[0]
            bins = int(np.ceil(hz * binning_window))
            oracle_trial_df["response"] = oracle_trial_df.response.apply(
                lambda x: np.mean(
                    x[: x.shape[0] // bins * bins, :].reshape(-1, x.shape[1], bins),
                    axis=-1,
                )
            )
            oracle_df = (
                oracle_trial_df[
                    ["animal_id", "scan_session", "scan_idx", "condition_hash"]
                ]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            oracle_df = oracle_df.merge(
                oracle_trial_df.groupby("condition_hash")
                .apply(lambda df: np.stack(df["response"], axis=0))
                .to_frame("single_trial")
                .reset_index()
            )
            oracle_df = oracle_df.loc[
                oracle_df.single_trial.apply(lambda x: x.shape[0])
                >= params["min_repeat"]
            ]
            oracle_df["mean"] = oracle_df["single_trial"].apply(
                lambda row: row.mean(axis=0)
            )
            oracle_df["residue"] = oracle_df["single_trial"] - oracle_df["mean"]
            oracle_df["residue_z"] = oracle_df["residue"].apply(
                lambda row: stats.zscore(row, axis=1)
            )
            oracle_df["residue_z"] = oracle_df["residue_z"].apply(
                lambda row: row.reshape(-1, row.shape[-1])
            )
            residue_z = np.concatenate(oracle_df["residue_z"].to_numpy())
            rho, p = function_utils.xcorr_p(residue_z.T, n_perm=params["n_perm"], rng=np.random.default_rng(params["seed"]))
            rho, p = squareform(rho, checks=False), squareform(p, checks=False)
            return unit_df, rho, p

    class Scan3Repeat(djp.Part):
        definition = """ # noise correlation computed with 2 repeats clips
        -> master
        ---
        -> scan3.TimingId
        -> scan3.BehaviorId
        -> scan3.ResponseId
        -> scan3.ValidId
        binning_window    : decimal(4,2)            # binning window in seconds
        min_repeat        : int                     # minimum number of repeats to be considered
        n_perm            : int unsigned            # number of permutations
        seed              : int unsigned            # random seed
        """
        content = [
            {
                "binning_window": decimal.Decimal("0.50"),
                "timing_id": 0,
                "behavior_id": 0,
                "response_id": 1,
                "valid_id": 0,
                "min_repeat": 4,
                "n_perm": 1000,
                "seed": 0,
            },
        ]

        def compute(self, scan_key):
            netflix = dj.create_virtual_module('netflix', 'pipeline_netflix')
            repeat_clip = stimulus.Clip & netflix.RepeatSet()
            params = self.fetch1()
            binning_window = float(params["binning_window"])

            unit_df = pd.DataFrame(
                (scan3.Unit & params & scan_key)
                .proj(..., scan_session="session")
                .fetch(
                    "animal_id",
                    "scan_session",
                    "scan_idx",
                    "unit_id",
                    "response_index",
                    order_by="response_index",
                    as_dict=True,
                )
            )
            unit_df["corr_idx"] = unit_df["response_index"]

            trials = (
                scan3.Response.Trial
                * scan3.Valid.Trial
                * scan3.TimingId
                * scan3.TimingInfo
                * stimulus.Trial
                * stimulus.Condition
                & "valid"
                & params
                & scan_key
            ).proj(..., scan_session="session")
            repeat_cond = stimulus.Condition.aggr(trials.proj(), repeats="count(*)") & "repeats > 1" & repeat_clip
            repeat_trial = trials & repeat_cond.proj()
            repeat_trial_df = pd.DataFrame(
                repeat_trial.fetch(
                    "animal_id",
                    "scan_session",
                    "scan_idx",
                    "condition_hash",
                    "response",
                    "hz",
                    as_dict=True,
                )
            )
            # bin to 500ms
            repeat_trial_df = pd.DataFrame(
                repeat_trial.fetch(
                    "animal_id", "session", "scan_idx", "condition_hash", "response", "hz", as_dict=True
                )
            )
            # bin to 500ms
            assert len(repeat_trial_df.hz.unique()) == 1
            hz = repeat_trial_df.hz.unique()[0]
            bins = int(np.ceil(hz * binning_window))
            repeat_trial_df["response"] = repeat_trial_df.response.apply(lambda x : np.mean(x[:x.shape[0]//bins*bins,:].reshape(-1, x.shape[1], bins), axis=-1))
            repeat_df = (
                repeat_trial_df[["animal_id", "session", "scan_idx", "condition_hash"]]
                .drop_duplicates()
                .reset_index(drop=True)
            )

            repeat_df = repeat_df.merge(
                repeat_trial_df.groupby("condition_hash")
                .apply(
                    lambda df: np.stack(df["response"], axis=0)
                )
                .to_frame("single_trial")
                .reset_index()
            )

            repeat_df['mean'] = repeat_df['single_trial'].apply(
                lambda row: row.mean(axis=0)
            )
            repeat_df['residue'] = repeat_df['single_trial'] - repeat_df['mean']
            # z-score the residue
            repeat_df['residue_z'] = repeat_df['residue'].apply(
                lambda row: stats.zscore(row, axis=1)
            )
            # concatenate all residue
            repeat_df['residue_z'] = repeat_df['residue_z'].apply(
                lambda row: row.reshape(-1, row.shape[-1])
            )
            residue_z = np.concatenate(repeat_df["residue_z"].to_numpy())
            rho, p = function_utils.xcorr_p(residue_z.T, n_perm=params["n_perm"], rng=np.random.default_rng(params["seed"]))
            rho, p = squareform(rho, checks=False), squareform(p, checks=False)
            return unit_df, rho, p

@schema
class NoiseCorr(djp.Computed):
    definition = """
    -> NoiseCorrConfig
    -> minnie_nda.Scan
    """

    class Unit(djp.Part):
        definition = """
        -> master
        -> minnie_nda.UnitSource
        ---
        corr_idx           : int               # index in the correlation matrix
        """

    class CorrMatrix(djp.Part):
        definition = """
        -> master
        ---
        corr_matrix        : blob@corr_array          # correlation matrix, stored in vector form
        p_matrix           : blob@corr_array          # p-value matrix, stored in vector form
        """

    @property
    def key_source(self):
        scan_oracle = (
            ScanSet.Member.proj(session="scan_session") * stimulus.Condition
        ) & (stimulus.Trial & (stimulus.Clip * netflix.OracleSet))
        return NoiseCorrConfig * minnie_nda.Scan & [
            scan_oracle.proj(scan_session="session") * NoiseCorrConfig.Scan3OracleCond,
            scan_oracle.proj(scan_session="session") * NoiseCorrConfig.Scan3Oracle,
            scan_oracle.proj(scan_session="session") * NoiseCorrConfig.Scan3Repeat,
        ]

    def make(self, key):
        unit_df, corr_matrix, p_matrix = (
            NoiseCorrConfig().part_table(key).compute(scan_key=key)
        )
        self.insert1(key)
        self.Unit().insert(
            [{**key, **d} for d in unit_df.to_dict("records")], ignore_extra_fields=True
        )
        self.CorrMatrix().insert1(dict(key, corr_matrix=corr_matrix, p_matrix=p_matrix))
