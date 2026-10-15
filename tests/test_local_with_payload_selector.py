"""Offline regression test cases for issue #1582; no provider or server required."""

import pytest

from qdrant_client import QdrantClient, grpc, models
from qdrant_client.local.local_collection import LocalCollection


@pytest.fixture
def memory_client():
    client = QdrantClient(":memory:")
    client.create_collection(
        "test_payload",
        vectors_config=models.VectorParams(size=2, distance=models.Distance.DOT),
    )
    client.upsert(
        "test_payload",
        [
            models.PointStruct(
                id=1,
                vector=[1.0, 2.0],
                payload={"a": 1, "b": 2, "nested": {"x": 10, "y": 20}},
            )
        ],
    )
    return client


def test_local_process_payload_with_grpc_selectors():
    payload = {"a": 1, "b": 2, "c": 3}

    # 1. include selector
    include_selector = grpc.WithPayloadSelector(
        include=grpc.PayloadIncludeSelector(fields=["a", "c"])
    )
    res = LocalCollection._process_payload(payload, include_selector)
    assert res == {"a": 1, "c": 3}

    # 2. exclude selector
    exclude_selector = grpc.WithPayloadSelector(
        exclude=grpc.PayloadExcludeSelector(fields=["b"])
    )
    res = LocalCollection._process_payload(payload, exclude_selector)
    assert res == {"a": 1, "c": 3}

    # 3. enable=True
    enable_true = grpc.WithPayloadSelector(enable=True)
    res = LocalCollection._process_payload(payload, enable_true)
    assert res == payload

    # 4. enable=False
    enable_false = grpc.WithPayloadSelector(enable=False)
    res = LocalCollection._process_payload(payload, enable_false)
    assert res is None


def test_local_retrieve_with_grpc_payload_selector(memory_client):
    # include
    selector_include = grpc.WithPayloadSelector(
        include=grpc.PayloadIncludeSelector(fields=["a"])
    )
    points = memory_client.retrieve("test_payload", [1], with_payload=selector_include)
    assert len(points) == 1
    assert points[0].payload == {"a": 1}

    # exclude
    selector_exclude = grpc.WithPayloadSelector(
        exclude=grpc.PayloadExcludeSelector(fields=["b"])
    )
    points = memory_client.retrieve("test_payload", [1], with_payload=selector_exclude)
    assert len(points) == 1
    assert points[0].payload == {"a": 1, "nested": {"x": 10, "y": 20}}

    # enable=False
    selector_false = grpc.WithPayloadSelector(enable=False)
    points = memory_client.retrieve("test_payload", [1], with_payload=selector_false)
    assert len(points) == 1
    assert points[0].payload is None

    # enable=True
    selector_true = grpc.WithPayloadSelector(enable=True)
    points = memory_client.retrieve("test_payload", [1], with_payload=selector_true)
    assert len(points) == 1
    assert points[0].payload == {"a": 1, "b": 2, "nested": {"x": 10, "y": 20}}


def test_local_scroll_with_grpc_payload_selector(memory_client):
    selector_include = grpc.WithPayloadSelector(
        include=grpc.PayloadIncludeSelector(fields=["b"])
    )
    records, _ = memory_client.scroll("test_payload", with_payload=selector_include)
    assert len(records) == 1
    assert records[0].payload == {"b": 2}

    selector_false = grpc.WithPayloadSelector(enable=False)
    records, _ = memory_client.scroll("test_payload", with_payload=selector_false)
    assert len(records) == 1
    assert records[0].payload is None
