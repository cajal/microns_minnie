import inspect
import logging as log
import re
from itertools import chain

import datajoint as dj
import datajoint_plus as djp
import numpy as np
import pandas as pd
from networkx import NetworkXError


def grouping_table(schema, source, ordered=False, hash_comment=None, aggregator=None):
    def decorator(
        cls,
    ):
        nonlocal aggregator, hash_comment
        cls.source = source
        cls.ordered = ordered
        table_name = cls.__name__
        split_table_name = re.findall("[A-Z][^A-Z]*", table_name)
        hash_comment = hash_comment or "unique identifier for {} grouping".format(
            " ".join(split_table_name).lower()
        )
        grouping_type = "_".join(split_table_name).lower()
        if cls.ordered:
            grouping_idx = grouping_type + "_idx"
        # attributes for hashing
        cls.enable_hashing = True
        cls.hash_name = grouping_type + "_hash"
        cls.hash_group = True
        cls.hashed_attrs = list(set(*chain([m.primary_key for m in cls.source])))
        all_member_keys = None
        for m in cls.source:
            all_member_keys = (
                all_member_keys * m.proj() if all_member_keys else m.proj()
            )
        cls.all_member_keys = all_member_keys
        cls.definition = """
        #
        {hn}                             : varchar(32)   # {comment}
        ---
        {gt}_note                        : varchar(1024) # note about the grouping
        {gt}_ts=CURRENT_TIMESTAMP        : timestamp     # automatic
        """.format(
            gt=grouping_type,
            hn=cls.hash_name,
            comment=hash_comment,
        )

        # Add fields for aggregation functions if exits
        aggregator = [] if aggregator is None else aggregator
        default_aggregator = [
            {
                "name": f"{grouping_type}_members",
                "type": "int unsigned",
                "callable": len,
                "comment": "number of members in the grouping",
            }
        ]
        aggregator = default_aggregator + aggregator
        for field in aggregator:
            assert {"name", "type", "callable"} <= {
                *field.keys()
            }, "aggregation function must have field 'name','type' and 'callable'"
            if "comment" not in field:
                field["comment"] = ""
            cls.definition += "{:<33}:{:<15}# {}\n".format(
                field["name"], field["type"], field["comment"]
            )

        # Define Member part table
        context = (
            schema.context.copy()
            if schema.context
            else inspect.currentframe().f_back.f_locals
        )

        deps = []
        for s in cls.source:
            if s.__name__ in context:
                deps.append(s.__name__)
                continue
            elif s.database not in context:
                context[s.database] = djp.create_dj_virtual_module(
                    s.database, s.database
                )
            deps.append(f"{s.database}.{s.__name__}")
        deps = "\n".join([f"-> {d}" for d in deps])

        if cls.ordered:

            class Member(djp.Part):
                definition = """
                -> master
                {} : int unsigned #  index of the member in the grouping
                {}
                """.format(
                    grouping_idx, deps
                )

        else:

            class Member(djp.Part):
                definition = """
                -> master
                {}
                """.format(
                    deps
                )

        cls.Member = Member

        # Define fill function
        @staticmethod
        def add_aggr_to_keys(keys):
            for field in aggregator:
                value = field["callable"](keys)
                keys = [{**k, **{field["name"]: value}} for k in keys]
            return keys

        cls.add_aggr_to_keys = add_aggr_to_keys

        if cls.ordered:

            @classmethod
            def fill(cls, keys, note=""):
                keys = cls.add_aggr_to_keys(keys)
                keys = [{**k, grouping_idx: i} for i, k in enumerate(keys)]
                cls.insert(
                    keys,
                    insert_to_parts=cls.Member,
                    ignore_extra_fields=True,
                    skip_duplicates=True,
                    constant_attrs={
                        f"{grouping_type}_note": note,
                    },
                    insert_to_parts_kws={
                        "skip_duplicates": True,
                        "ignore_extra_fields": True,
                    },
                )

        else:

            @classmethod
            def fill(cls, keys, note=""):
                keys = (cls.all_member_keys & keys).fetch("KEY")
                keys = cls.add_aggr_to_keys(keys)
                cls.insert(
                    keys,
                    insert_to_parts=cls.Member,
                    ignore_extra_fields=True,
                    skip_duplicates=True,
                    constant_attrs={
                        f"{grouping_type}_note": note,
                    },
                    insert_to_parts_kws={
                        "skip_duplicates": True,
                        "ignore_extra_fields": True,
                    },
                )

        cls.fill = fill
        cls._init_validation()  # update table definition with hash info
        cls = schema(cls, context=context)
        return cls

    return decorator


def config_table(schema, hash_comment=None, config_type=None):
    def decorator(cls):
        nonlocal hash_comment
        table_name = cls().__class__.__name__
        assert table_name.endswith("Config"), "Config Table must be named {Name}Config"
        split_table_name = re.findall("[A-Z][^A-Z]*", table_name.split("Config")[0])
        hash_comment = hash_comment or "unique identifier for {} configuration".format(
            " ".join(split_table_name).lower()
        )

        cls.config_type = config_type or "_".join(split_table_name).lower()
        cls.hash_part_table_names = True
        cls.hash_name = cls.config_type + "_hash"
        cls.definition = """
            # parameters for {tn}
            {cn}                        : varchar(32) # {descr}
            ---
            {ct}_type                   : varchar(50)  # type
            {ct}_ts=CURRENT_TIMESTAMP   : timestamp    # automatic
            """.format(
            ct=cls.config_type, cn=cls.hash_name, tn=table_name, descr=hash_comment
        )
        schema(cls, context=schema.context or inspect.currentframe().f_back.f_locals)

        @staticmethod
        def fill():
            for rel in cls.parts(as_cls=True):
                log.info("Checking " + rel.__name__)
                keys_df = pd.DataFrame(
                    data=[c for c in rel().content if len(rel & c) == 0],
                    columns=rel.heading.secondary_attributes,
                    dtype=object,
                )
                if len(keys_df) == 0:
                    log.info("\tNo keys to insert")
                    continue
                content_attr = keys_df.columns.values
                table_attr = rel().heading.secondary_attributes
                assert np.isin(content_attr, table_attr).all()
                keys_df[cls.config_type + "_type"] = rel.__name__
                keys = keys_df.to_dict(orient="records")
                log.info("\tInserting {} keys".format(len(keys)))
                rel().insert(
                    keys,
                    insert_to_master=True,
                    ignore_extra_fields=True,
                    skip_duplicates=True,
                )

        @staticmethod
        def clean():
            ctype = "{}_type".format(cls.config_type)
            keys = cls & [{ctype: p.__name__} for p in cls.parts(as_cls=True)]
            invalid = (keys - cls.parts(as_cls=True)).fetch("KEY")
            (cls & invalid).delete()

        def part_type(self, key=None):
            rel = self if key is None else self & key
            config_type = "{}_type".format(self.config_type)
            config_types = dj.U(config_type) & rel
            assert (
                len(config_types) == 1
            ), "Table must restricted to a single config type"
            return config_types.fetch1(config_type)

        def part_table(self, key=None):
            key = self.fetch1("KEY") if key is None else (self & key).fetch1("KEY")
            return getattr(self, self.part_type(key))() & key

        def parameters(self, key=None):
            return self.part_table(key).parameters()

        cls.fill = fill
        cls.part_type = part_type
        cls.part_table = part_table
        cls.parameters = parameters
        cls.clean = clean

        def configure_part(part_cls):
            part_cls.enable_hashing = True
            part_cls.hash_name = cls.hash_name
            part_cls.hashed_attrs = part_cls.heading.secondary_attributes

            def parameters(self, key=None):
                if key is None:
                    key = self.fetch1("KEY")
                elif isinstance(key, dj.table.Table):
                    key = key.fetch1("KEY")
                raw_params = (self & key).fetch1()
                raw_params = {
                    k: v
                    for k, v in raw_params.items()
                    if k in self.heading.secondary_attributes
                }
                params = dict()
                for k in raw_params.keys():
                    params[k] = raw_params[k]
                return params

            assert hasattr(part_cls, "content"), "content is not defined"
            assert issubclass(part_cls, dj.Part)

            if hasattr(part_cls, "extra_parameters"):

                def extra_parameters(self, key=None):
                    return part_cls.extra_parameters(self, parameters(self, key))

                part_cls.parameters = extra_parameters
            else:
                part_cls.parameters = parameters

            return part_cls

        for part in cls.parts(as_cls=True):
            setattr(cls, part.__name__, configure_part(part))

        def modify_header(cls):
            # modify table comments
            inds, contents, _ = djp.heading.parse_definition(cls.definition)
            contents["headers"] = ["""#"""]
            cls.definition = djp.heading.reform_definition(inds, contents).strip("\n")
            # modify header
            hash_info_dict = dict(
                add_class_name=True,
                hash_name=cls.hash_name,
                hashed_attrs=cls.hashed_attrs,
            )
            cls._modify_header(**hash_info_dict)
            inds, contents, _ = djp.heading.parse_definition(cls.definition)
            new_comment = contents["headers"][0]
            old_comment = cls.heading.table_info["comment"]
            if new_comment != old_comment:
                log.info(
                    "Updating table %s comment to %s", cls.full_table_name, new_comment
                )
                schema.connection.query(
                    (f'ALTER TABLE {cls.full_table_name} COMMENT = "{new_comment}"')
                )

        modify_header(cls)
        for part in cls.parts(as_cls=True):
            modify_header(part)
        return cls

    return decorator


def versioned_part_table(version):
    def decorator(cls):
        cls.part_name = re.sub(r"(?<!^)(?=[A-Z])", "_", cls.__name__).lower()
        cls.version = version
        cls.definition = cls.definition + "{:<33}:{:<15}# {}\n".format(
            cls.part_name + "_version", "int", f"version of the {cls.__name__} code"
        )
        cls.original_content = cls.content

        @property
        def content(self):
            return [
                {**key, self.part_name + "_version": self.version}
                for key in self.original_content
            ]

        cls.content = content

        def __call__(self):
            return self & f'{self.part_name + "_version"} = {self.version}'

        cls.__call__ = __call__
        return cls

    return decorator


def drop_all_table(module):
    for name, obj in inspect.getmembers(module):
        if (
            isinstance(obj, type)
            and issubclass(obj, dj.Table)
            and not issubclass(obj, dj.Part)
        ):
            log.info("Checking %s", name)
            try:
                obj.drop()
            except NetworkXError as e:
                if str(e) != f"The node {obj.full_table_name} is not in the digraph.":
                    log.warning(e)
                continue
