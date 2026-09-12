"""Broadcast a graph-theoretic bag oracle across point-row size boundaries."""
import os

import pandas as pd
import pytest

import graphistry
from graphistry.compute.ast import e_forward, e_reverse, n, rows, select
from graphistry.tests.compute.gfql.routes.registry import to_engine
from graphistry.tests.compute.gfql.routes.switch import ROUTES, routes_off


@pytest.mark.parametrize("engine", ["pandas", "polars", "cudf"])
@pytest.mark.parametrize("count", [0, 1, 2, 8, 9, 32, 33])
@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("aliases", [("a", "b"), ("seed_node", "target_node")])
def test_point_joined_bag_matches_input_edge_oracle(engine, count, reverse, aliases):
    if engine == "cudf" and os.environ.get("TEST_CUDF") != "1":
        pytest.skip("cuDF lane runs with TEST_CUDF=1")
    # Repeated destinations are distinct paths, and a different edge type is excluded.
    targets = [1 + i % 3 for i in range(count)]
    nodes = pd.DataFrame({"key": [0, 1, 2, 3], "value": [11, 22, 33, 44]})
    starts, ends = [0] * (count + 1), targets + [3]
    if reverse:
        starts, ends = ends, starts
    edges = pd.DataFrame({"s": starts, "d": ends, "eid": range(count + 1),
                          "type": ["X"] * count + ["Y"]})
    g = graphistry.nodes(to_engine(nodes, engine), "key").edges(
        to_engine(edges, engine), "s", "d", "eid").gfql_index_all(engine=engine)
    a, b = aliases
    ops = [n({"key": 0}, name=a), (e_reverse if reverse else e_forward)({"type": "X"}, name="link"),
           n(name=b), rows(), select([("seed", f"{a}.value"), ("target", f"{b}.value"),
                                      ("edge", "link.eid")])]
    expected = sorted((11, (target + 1) * 11, i) for i, target in enumerate(targets))

    def values(result):
        frame = result._nodes
        if hasattr(frame, "to_dicts"):
            return sorted((r["seed"], r["target"], r["edge"]) for r in frame.to_dicts())
        if hasattr(frame, "to_pandas"):
            frame = frame.to_pandas()
        return sorted(frame[["seed", "target", "edge"]].itertuples(index=False, name=None))

    assert values(g.gfql(ops, engine=engine, index_policy="force")) == expected
    with routes_off(ROUTES):
        assert values(g.gfql(ops, engine=engine, index_policy="force")) == expected
