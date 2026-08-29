# Copyright 2019 The Vearch Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Client-side throughput benchmark for a Vearch HNSW index.

By default, the test creates and populates its own space before measuring
search. Setup and index construction are deliberately outside the measurement
window. The benchmark accepts the same common ``VEARCH_PERF_*`` settings as
the IVFPQ benchmark, with ``VEARCH_HNSW_PERF_*`` taking precedence for
HNSW-specific settings. ``VEARCH_HNSW_PERF_TRUST_ENV`` controls whether
Requests reads proxy and CA-related environment variables (default: ``false``).
Set ``VEARCH_HNSW_PERF_KEEP_INDEX=true`` on the first run to flush and retain
the created space, then use ``VEARCH_HNSW_PERF_REUSE_INDEX=true`` with the same
explicit database and space names to benchmark that saved index without
rebuilding it.
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests

from utils.data_utils import DatasetSift1M
from utils.vearch_utils import (
    add,
    create_db,
    create_space,
    destroy,
    index_flush,
    logger,
    password,
    router_url,
    username,
)


logger.handlers.clear()
logger.propagate = True
logger.setLevel("INFO")


def _env(name, default):
    """Read an index-specific setting, falling back to common benchmark settings."""
    return os.getenv(
        "VEARCH_HNSW_PERF_" + name,
        os.getenv("VEARCH_PERF_" + name, str(default)),
    )


def _env_int(name, default, minimum=None):
    value = int(_env(name, default))
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %s, got %s" % (name, minimum, value))
    return value


def _env_float(name, default, minimum=None):
    value = float(_env(name, default))
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %s, got %s" % (name, minimum, value))
    return value


def _env_csv(name, default):
    values = [item.strip() for item in _env(name, default).split(",")]
    return [item for item in values if item]


def _env_bool(name, default=False):
    value = _env(name, default)
    normalized = value.strip().lower()
    if normalized in ("1", "true", "yes", "on"):
        return True
    if normalized in ("0", "false", "no", "off"):
        return False
    raise ValueError("%s must be a boolean, got %r" % (name, value))


QUERY_COUNT = _env_int("QUERY_COUNT", 100, minimum=1)
TOP_K = _env_int("TOP_K", 100, minimum=1)
INGEST_BATCH_SIZE = _env_int("INGEST_BATCH_SIZE", 100, minimum=1)
WARMUP_ROUNDS = _env_int("WARMUP_ROUNDS", 10, minimum=0)
TRIALS = _env_int("TRIALS", 5, minimum=1)
MEASURE_SECONDS = _env_float("SECONDS", 30, minimum=0.1)
SETTLE_SECONDS = _env_float("SETTLE_SECONDS", 10, minimum=0)
INDEX_TIMEOUT_SECONDS = _env_float("INDEX_TIMEOUT_SECONDS", 1800, minimum=1)
INDEX_POLL_SECONDS = _env_float("INDEX_POLL_SECONDS", 2, minimum=0.1)
READY_CONSECUTIVE_POLLS = _env_int("READY_CONSECUTIVE_POLLS", 2, minimum=1)
REQUEST_TIMEOUT_SECONDS = _env_float("REQUEST_TIMEOUT_SECONDS", 120, minimum=0.1)
NLINKS = _env_int("NLINKS", 32, minimum=1)
EF_CONSTRUCTION = _env_int("EF_CONSTRUCTION", 200, minimum=1)
EF_SEARCH = _env_int("EF_SEARCH", 64)
TRUST_ENV = _env_bool("TRUST_ENV")
REUSE_INDEX = _env_bool("REUSE_INDEX")
KEEP_INDEX = _env_bool("KEEP_INDEX")
REUSE_WARMUP_SECONDS = _env_float("REUSE_WARMUP_SECONDS", 30, minimum=0)
MODES = _env_csv("MODES", "single,batch")
CONCURRENCIES = [int(value) for value in _env_csv("CONCURRENCY", "1")]

if not MODES or any(mode not in ("single", "batch") for mode in MODES):
    raise ValueError("VEARCH_PERF_MODES must contain only 'single' and/or 'batch'")
if not CONCURRENCIES or any(value < 1 for value in CONCURRENCIES):
    raise ValueError("VEARCH_PERF_CONCURRENCY values must be positive integers")


def _assert_api_ok(response, action):
    response.raise_for_status()
    payload = response.json()
    if payload.get("code", 0) != 0:
        raise AssertionError("%s failed: %s" % (action, payload))
    return payload


def _create_hnsw_space(db_name, space_name, dimension, metric_type):
    _assert_api_ok(create_db(router_url, db_name), "create database")
    params = {
        "metric_type": metric_type,
        "nlinks": NLINKS,
        "efConstruction": EF_CONSTRUCTION,
    }
    if EF_SEARCH > 0:
        params["efSearch"] = EF_SEARCH
    space_config = {
        "name": space_name,
        "partition_num": 1,
        "replica_num": 1,
        "fields": [
            {
                "name": "field_int",
                "type": "integer",
                "index": {"name": "field_int", "type": "SCALAR"},
            },
            {
                "name": "field_vector",
                "type": "vector",
                "dimension": dimension,
                "store_type": "MemoryOnly",
                "index": {"name": "gamma", "type": "HNSW", "params": params},
            },
        ],
    }
    try:
        _assert_api_ok(create_space(router_url, db_name, space_config), "create space")
    except Exception:
        destroy(router_url, db_name, space_name)
        raise


def _ingest(db_name, space_name, vectors):
    full_batches, remainder = divmod(vectors.shape[0], INGEST_BATCH_SIZE)
    if full_batches:
        add(
            full_batches,
            INGEST_BATCH_SIZE,
            vectors[: full_batches * INGEST_BATCH_SIZE],
            with_id=True,
            db_name=db_name,
            space_name=space_name,
            max_workers=1,
        )
    if remainder:
        offset = full_batches * INGEST_BATCH_SIZE
        add(
            remainder,
            1,
            vectors[offset:],
            with_id=True,
            db_name=db_name,
            space_name=space_name,
            offset=offset,
            max_workers=1,
        )


def _wait_for_index(
    db_name,
    space_name,
    expected_count,
    require_complete_at_start=False,
):
    """Wait for one fully searchable partition.

    In reuse mode a live partition with all documents but only a partial index
    means Vearch is rebuilding after restart, not reusing the flushed graph.
    Reject that state instead of silently benchmarking a different graph.
    """
    url = "%s/dbs/%s/spaces/%s" % (router_url, db_name, space_name)
    deadline = time.perf_counter() + INDEX_TIMEOUT_SECONDS
    ready_polls = 0
    last_state = "no response"
    with requests.Session() as session:
        session.auth = (username, password)
        while True:
            payload = _assert_api_ok(
                session.get(url, timeout=REQUEST_TIMEOUT_SECONDS),
                "get index status",
            )
            partitions = payload.get("data", {}).get("partitions", [])
            indexed_count = sum(
                partition.get("index_num", 0) for partition in partitions
            )
            document_count = sum(
                partition.get("doc_num", 0) for partition in partitions
            )
            partition_statuses = [
                partition.get("status", 0) for partition in partitions
            ]
            index_statuses = [
                partition.get("index_status", 0) for partition in partitions
            ]
            colors = [partition.get("color", "") for partition in partitions]
            last_state = (
                "partitions=%d documents=%d indexed=%d "
                "partition_status=%s index_status=%s color=%s"
                % (
                    len(partitions),
                    document_count,
                    indexed_count,
                    partition_statuses,
                    index_statuses,
                    colors,
                )
            )

            partition_recovered = (
                len(partitions) == 1
                and partition_statuses[0] in (3, 4)  # PA_READONLY/PA_READWRITE
            )
            partition_live = (
                partition_recovered
                and partition_statuses == [4]  # entity.PA_READWRITE
                and colors == ["green"]
            )
            if require_complete_at_start and partition_recovered:
                if document_count != expected_count:
                    raise AssertionError(
                        "reusable space document count mismatch after restart: "
                        "%s expected=%d" % (last_state, expected_count)
                    )
                if indexed_count != expected_count:
                    raise AssertionError(
                        "retained HNSW dump is incomplete after restart; refusing "
                        "to benchmark a graph rebuilt during REUSE_INDEX: %s expected=%d"
                        % (last_state, expected_count)
                    )

            ready = (
                partition_live
                and document_count == expected_count
                and indexed_count == expected_count
                and index_statuses == [2]  # INDEXED
            )
            if ready:
                ready_polls += 1
                if ready_polls >= READY_CONSECUTIVE_POLLS:
                    logger.info("index ready: %s", last_state)
                    return
            else:
                ready_polls = 0
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    "index did not become ready in %.1fs: %s expected=%d"
                    % (INDEX_TIMEOUT_SECONDS, last_state, expected_count)
                )
            time.sleep(min(INDEX_POLL_SECONDS, remaining))


def _flush_retained_index(db_name, space_name):
    """Synchronously flush every shard and reject partial shard failures."""
    payload = _assert_api_ok(
        index_flush(router_url, db_name, space_name),
        "flush retained index",
    )
    shards = payload.get("data", {}).get("_shards")
    if not isinstance(shards, dict):
        raise AssertionError(
            "flush retained index returned no shard status: %s" % payload
        )
    total = shards.get("total")
    successful = shards.get("successful")
    # SearchStatus uses protobuf/JSON zero-value omission, so a successful
    # response commonly has no explicit "failed": 0 field.
    failed = shards.get("failed", 0)
    if total != 1 or successful != total or failed != 0:
        raise AssertionError("flush retained index failed: %s" % shards)
    logger.info("retained index flush confirmed: %s", shards)


def _validate_reusable_space(
    db_name,
    space_name,
    expected_count,
    expected_dimension,
    expected_metric,
    expected_nlinks,
    expected_ef_construction,
):
    """Ensure a retained space is the exact dataset/index shape we expect."""
    url = "%s/dbs/%s/spaces/%s" % (router_url, db_name, space_name)
    with requests.Session() as session:
        session.auth = (username, password)
        payload = _assert_api_ok(
            session.get(url, timeout=REQUEST_TIMEOUT_SECONDS),
            "describe reusable space",
        )

    data = payload.get("data", {})
    partitions = data.get("partitions", [])
    if len(partitions) != 1:
        raise AssertionError(
            "reusable space must have exactly one partition, got %d"
            % len(partitions)
        )
    indexed_count = sum(partition.get("index_num", 0) for partition in partitions)
    if indexed_count != expected_count:
        raise AssertionError(
            "reusable space vector count mismatch: indexed=%d expected=%d"
            % (indexed_count, expected_count)
        )
    if all("doc_num" in partition for partition in partitions):
        document_count = sum(partition["doc_num"] for partition in partitions)
        if document_count != expected_count:
            raise AssertionError(
                "reusable space document count mismatch: documents=%d expected=%d"
                % (document_count, expected_count)
            )

    fields = data.get("schema", {}).get("fields", [])
    vector_field = next(
        (field for field in fields if field.get("name") == "field_vector"),
        None,
    )
    if vector_field is None:
        raise AssertionError("reusable space is missing field_vector")
    if vector_field.get("dimension") != expected_dimension:
        raise AssertionError(
            "reusable space dimension mismatch: actual=%s expected=%d"
            % (vector_field.get("dimension"), expected_dimension)
        )

    index = vector_field.get("index", {})
    if index.get("type") != "HNSW":
        raise AssertionError(
            "reusable space index type mismatch: actual=%s expected=HNSW"
            % index.get("type")
        )
    params = index.get("params", {})
    metric = params.get("metric_type", "L2")
    if str(metric).casefold() != expected_metric.casefold():
        raise AssertionError(
            "reusable space metric mismatch: actual=%s expected=%s"
            % (metric, expected_metric)
        )
    if params.get("nlinks", 32) != expected_nlinks:
        raise AssertionError(
            "reusable space nlinks mismatch: actual=%s expected=%d"
            % (params.get("nlinks", 32), expected_nlinks)
        )
    if params.get("efConstruction", 100) != expected_ef_construction:
        raise AssertionError(
            "reusable space efConstruction mismatch: actual=%s expected=%d"
            % (params.get("efConstruction", 100), expected_ef_construction)
        )

    return data


def _base_query(db_name, space_name, metric_type):
    index_params = {"metric_type": metric_type}
    if EF_SEARCH >= 0:
        index_params["efSearch"] = EF_SEARCH
    return {
        "vectors": [],
        "index_params": index_params,
        "vector_value": False,
        "fields": ["field_int"],
        "limit": TOP_K,
        "db_name": db_name,
        "space_name": space_name,
    }


def _request_specs(mode, query_vectors, base_query):
    if mode == "batch":
        query = dict(base_query)
        query["vectors"] = [
            {"field": "field_vector", "feature": query_vectors.reshape(-1).tolist()}
        ]
        return [(json.dumps(query, separators=(",", ":")), query_vectors.shape[0])]
    specs = []
    for vector in query_vectors:
        query = dict(base_query)
        query["vectors"] = [{"field": "field_vector", "feature": vector.tolist()}]
        specs.append((json.dumps(query, separators=(",", ":")), 1))
    return specs


def _new_session():
    session = _TimedSession()
    # The benchmark targets the local Router. Requests otherwise scans the
    # process environment for proxy settings on every request, adding
    # serialized Python/GIL work to the closed-loop client measurement.
    session.trust_env = TRUST_ENV
    session.mount("http://", _TimedHTTPAdapter())
    session.mount("https://", _TimedHTTPAdapter())
    session.auth = (username, password)
    session.headers.update({"Content-Type": "application/json"})
    return session


def _preconnect(session):
    """Establish this session's Router connection outside the trial window."""
    response = session.get(router_url + "/", timeout=REQUEST_TIMEOUT_SECONDS)
    _assert_api_ok(response, "preconnect to Router")


class _TimedSession(requests.Session):
    """Session using an adapter that measures the HTTP exchange."""

    def send(self, request, **kwargs):
        response = super().send(request, **kwargs)
        started = getattr(response, "_vearch_http_started", None)
        if started is not None:
            response._vearch_http_elapsed = time.perf_counter() - started
        return response


class _TimedHTTPAdapter(requests.adapters.HTTPAdapter):
    """Mark the point immediately before Requests enters urllib3."""

    def send(self, request, **kwargs):
        started = time.perf_counter()
        response = super().send(request, **kwargs)
        response._vearch_http_started = started
        return response


def _execute_search(session, body, expected_query_count, return_latency=False):
    response = session.post(
        router_url + "/document/search?timeout=2000000",
        data=body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    payload = _assert_api_ok(response, "search")
    documents = payload.get("data", {}).get("documents")
    if not isinstance(documents, list) or len(documents) != expected_query_count:
        raise AssertionError(
            "search returned %s query result groups, expected %d"
            % (len(documents) if isinstance(documents, list) else None, expected_query_count)
        )
    if return_latency:
        return documents, response._vearch_http_elapsed
    return documents


def _verify_recall(batch_spec, groundtruth):
    body, expected_query_count = batch_spec
    with _new_session() as session:
        documents = _execute_search(session, body, expected_query_count)
    neighbors = []
    for results in documents:
        ids = [result["field_int"] for result in results[:TOP_K]]
        neighbors.append(ids + [-1] * (TOP_K - len(ids)))
    neighbors = np.asarray(neighbors)
    recalls = {}
    recall_at = 1
    while recall_at <= TOP_K:
        recalls[recall_at] = float(
            (neighbors[:, :recall_at] == groundtruth[:, :1]).sum()
            / neighbors.shape[0]
        )
        recall_at *= 10
    logger.info(
        "correctness check: %s",
        ", ".join(
            "recall@%d=%.2f%%" % (key, value * 100)
            for key, value in recalls.items()
        ),
    )


def _run_warmup(request_specs, concurrency, minimum_seconds=0):
    """Warm until both the configured round and duration targets are met."""
    def worker(worker_id):
        errors = 0
        error_examples = []
        successes = 0
        completed_rounds = 0
        started = time.perf_counter()
        deadline = started + minimum_seconds
        with _new_session() as session:
            warmup_failed = False
            while (
                completed_rounds < WARMUP_ROUNDS
                or time.perf_counter() < deadline
            ):
                for offset in range(len(request_specs)):
                    body, expected = request_specs[
                        (worker_id + offset) % len(request_specs)
                    ]
                    try:
                        _execute_search(session, body, expected)
                    except Exception as error:
                        errors += 1
                        if len(error_examples) < 3:
                            error_examples.append(
                                "warmup: %s: %s" % (type(error).__name__, error)
                            )
                        warmup_failed = True
                        break
                    else:
                        successes += 1
                if warmup_failed:
                    break
                completed_rounds += 1
        return {
            "errors": errors,
            "error_examples": error_examples,
            "successes": successes,
            "rounds": completed_rounds,
            "finished": time.perf_counter(),
            "started": started,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(worker, worker_id)
            for worker_id in range(concurrency)
        ]
        worker_results = [future.result() for future in futures]
    return {
        "errors": sum(result["errors"] for result in worker_results),
        "error_examples": [
            error
            for result in worker_results
            for error in result["error_examples"]
        ][:3],
        "successes": sum(result["successes"] for result in worker_results),
        "rounds": min(result["rounds"] for result in worker_results),
        "elapsed": max(result["finished"] for result in worker_results)
        - min(result["started"] for result in worker_results),
    }


def _run_trial(request_specs, concurrency):
    state = {}

    def start_measurement():
        state["start"] = time.perf_counter()
        state["deadline"] = state["start"] + MEASURE_SECONDS

    barrier = threading.Barrier(concurrency + 1, action=start_measurement)

    def worker(worker_id):
        latencies_ms = []
        errors = 0
        error_examples = []
        request_index = worker_id
        with _new_session() as session:
            try:
                _preconnect(session)
                barrier.wait()
            except BaseException:
                # Do not leave the other workers or the coordinator blocked if
                # establishing one worker's connection fails.
                barrier.abort()
                raise
            while time.perf_counter() < state["deadline"]:
                body, expected = request_specs[request_index % len(request_specs)]
                try:
                    _, http_elapsed = _execute_search(
                        session,
                        body,
                        expected,
                        return_latency=True,
                    )
                except Exception as error:
                    errors += 1
                    if len(error_examples) < 3:
                        error_examples.append("%s: %s" % (type(error).__name__, error))
                else:
                    latencies_ms.append(http_elapsed * 1000.0)
                request_index += concurrency
        return {
            "latencies_ms": latencies_ms,
            "errors": errors,
            "error_examples": error_examples,
            "finished": time.perf_counter(),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(concurrency)]
        barrier.wait()
        worker_results = [future.result() for future in futures]
    elapsed = max(result["finished"] for result in worker_results) - state["start"]
    latencies_ms = [latency for result in worker_results for latency in result["latencies_ms"]]
    errors = sum(result["errors"] for result in worker_results)
    error_examples = [
        error for result in worker_results for error in result["error_examples"]
    ][:3]
    successes = len(latencies_ms)
    attempts = successes + errors
    return {
        "latencies_ms": latencies_ms,
        "elapsed": elapsed,
        "successes": successes,
        "errors": errors,
        "error_rate": errors * 100.0 / attempts if attempts else 100.0,
        "error_examples": error_examples,
    }


def _latency_stats(latencies_ms):
    values = np.asarray(latencies_ms, dtype=np.float64)
    if not values.size:
        return {key: float("nan") for key in ("avg", "p50", "p95", "p99")}
    return {
        "avg": float(np.mean(values)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
    }


def _log_trial(mode, concurrency, trial_number, result, vectors_per_request):
    stats = _latency_stats(result["latencies_ms"])
    request_qps = result["successes"] / result["elapsed"]
    vector_qps = request_qps * vectors_per_request
    logger.info(
        "PERF trial=%d mode=%s concurrency=%d samples=%d "
        "latency_ms(avg/p50/p95/p99)=%.2f/%.2f/%.2f/%.2f "
        "request_qps=%.2f vector_qps=%.2f errors=%d error_rate=%.4f%%",
        trial_number,
        mode,
        concurrency,
        result["successes"],
        stats["avg"],
        stats["p50"],
        stats["p95"],
        stats["p99"],
        request_qps,
        vector_qps,
        result["errors"],
        result["error_rate"],
    )
    return vector_qps


def _log_summary(mode, concurrency, trial_results, vector_qps):
    latencies = [latency for result in trial_results for latency in result["latencies_ms"]]
    stats = _latency_stats(latencies)
    qps_values = np.asarray(vector_qps, dtype=np.float64)
    qps_mean = float(np.mean(qps_values))
    qps_stddev = (
        float(np.std(qps_values, ddof=1)) if qps_values.size > 1 else 0.0
    )
    qps_cv = qps_stddev * 100.0 / qps_mean if qps_mean else float("nan")
    logger.info(
        "PERF SUMMARY mode=%s concurrency=%d trials=%d samples=%d "
        "latency_ms(avg/p50/p95/p99)=%.2f/%.2f/%.2f/%.2f "
        "median_vector_qps=%.2f vector_qps_range(min/max)=%.2f/%.2f "
        "vector_qps_cv=%.2f%%",
        mode,
        concurrency,
        len(trial_results),
        len(latencies),
        stats["avg"],
        stats["p50"],
        stats["p95"],
        stats["p99"],
        float(np.median(qps_values)),
        float(np.min(qps_values)),
        float(np.max(qps_values)),
        qps_cv,
    )


def test_vearch_index_hnsw_performance():
    explicit_db_name = _env("DB_NAME", "").strip()
    if (REUSE_INDEX or KEEP_INDEX) and not explicit_db_name:
        raise ValueError(
            "VEARCH_HNSW_PERF_DB_NAME (or VEARCH_PERF_DB_NAME) must be set "
            "when KEEP_INDEX or REUSE_INDEX is enabled"
        )
    db_name = explicit_db_name or "ts_hnsw_perf_%d" % os.getpid()
    space_name = _env("SPACE_NAME", "ts_hnsw_perf_space")

    dataset = DatasetSift1M()
    expected_count = dataset.nb
    dimension = dataset.d
    vectors = None
    if not REUSE_INDEX:
        vectors = dataset.get_database()
        if vectors.shape != (expected_count, dimension):
            raise AssertionError(
                "database shape mismatch: actual=%s expected=(%d, %d)"
                % (vectors.shape, expected_count, dimension)
            )
    query_count = min(QUERY_COUNT, dataset.nq)
    query_vectors = dataset.get_queries()[:query_count]
    groundtruth = np.asarray(dataset.get_groundtruth()[:query_count])

    logger.info(
        "PERF CONFIG index=HNSW vectors=%d dimension=%d queries=%d top_k=%d "
        "modes=%s concurrency=%s warmup_rounds=%d trials=%d seconds=%.1f "
        "nlinks=%d efConstruction=%d efSearch=%s "
        "reuse_index=%s keep_index=%s reuse_warmup_seconds=%.1f "
        "ready_consecutive_polls=%d latency_scope=http_send trust_env=%s",
        expected_count,
        dimension,
        query_count,
        TOP_K,
        ",".join(MODES),
        ",".join(str(value) for value in CONCURRENCIES),
        WARMUP_ROUNDS,
        TRIALS,
        MEASURE_SECONDS,
        NLINKS,
        EF_CONSTRUCTION,
        "server-default" if EF_SEARCH < 0 else EF_SEARCH,
        str(REUSE_INDEX).lower(),
        str(KEEP_INDEX).lower(),
        REUSE_WARMUP_SECONDS,
        READY_CONSECUTIVE_POLLS,
        str(TRUST_ENV).lower(),
    )

    database_created = False
    index_persisted = False
    try:
        started = time.perf_counter()
        if REUSE_INDEX:
            _wait_for_index(
                db_name,
                space_name,
                expected_count,
                require_complete_at_start=True,
            )
            _validate_reusable_space(
                db_name,
                space_name,
                expected_count,
                dimension,
                dataset.metric,
                NLINKS,
                EF_CONSTRUCTION,
            )
            logger.info(
                "PERF SETUP index_source=reused total_ready_seconds=%.2f",
                time.perf_counter() - started,
            )
        else:
            _create_hnsw_space(db_name, space_name, dimension, dataset.metric)
            database_created = True
            _ingest(db_name, space_name, vectors)
            _wait_for_index(db_name, space_name, expected_count)
            if KEEP_INDEX:
                _flush_retained_index(db_name, space_name)
                index_persisted = True
            logger.info(
                "PERF SETUP index_source=built retained=%s total_ready_seconds=%.2f",
                str(index_persisted).lower(),
                time.perf_counter() - started,
            )
        if SETTLE_SECONDS:
            time.sleep(SETTLE_SECONDS)

        base_query = _base_query(db_name, space_name, dataset.metric)
        batch_specs = _request_specs("batch", query_vectors, base_query)
        _verify_recall(batch_specs[0], groundtruth)
        for mode in MODES:
            specs = (
                batch_specs
                if mode == "batch"
                else _request_specs(mode, query_vectors, base_query)
            )
            vectors_per_request = query_count if mode == "batch" else 1
            for concurrency in CONCURRENCIES:
                minimum_warmup_seconds = REUSE_WARMUP_SECONDS if REUSE_INDEX else 0
                warmup_result = _run_warmup(
                    specs,
                    concurrency,
                    minimum_seconds=minimum_warmup_seconds,
                )
                logger.info(
                    "PERF WARMUP mode=%s concurrency=%d rounds=%d requests=%d "
                    "elapsed_seconds=%.2f minimum_seconds=%.1f",
                    mode,
                    concurrency,
                    warmup_result["rounds"],
                    warmup_result["successes"],
                    warmup_result["elapsed"],
                    minimum_warmup_seconds,
                )
                if warmup_result["error_examples"]:
                    logger.error(
                        "benchmark warmup error examples: %s",
                        warmup_result["error_examples"],
                    )
                assert warmup_result["errors"] == 0
                trial_results = []
                trial_qps = []
                for trial_number in range(1, TRIALS + 1):
                    result = _run_trial(specs, concurrency)
                    trial_results.append(result)
                    trial_qps.append(
                        _log_trial(
                            mode,
                            concurrency,
                            trial_number,
                            result,
                            vectors_per_request,
                        )
                    )
                    if result["error_examples"]:
                        logger.error("benchmark error examples: %s", result["error_examples"])
                    assert result["errors"] == 0
                    assert result["successes"] > 0
                _log_summary(mode, concurrency, trial_results, trial_qps)
    finally:
        # Keep only a completely built and successfully flushed index. Failed
        # setup must not leave a partial space that a later reuse run accepts.
        if database_created and not index_persisted:
            destroy(router_url, db_name, space_name)
