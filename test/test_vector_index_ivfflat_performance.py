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

"""Client-side throughput benchmark for a Vearch IVFFLAT index.

The database, ingestion, and IVFFLAT training steps are excluded from search
measurements.  Common ``VEARCH_PERF_*`` settings are supported, while
``VEARCH_IVFFLAT_PERF_*`` values take precedence for this index.
``VEARCH_IVFFLAT_PERF_TRUST_ENV`` controls whether Requests reads proxy and
CA-related environment variables (default: ``false``).
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
    logger,
    password,
    router_url,
    username,
)


logger.handlers.clear()
logger.propagate = True
logger.setLevel("INFO")


def _env(name, default):
    return os.getenv(
        "VEARCH_IVFFLAT_PERF_" + name,
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
    return [item.strip() for item in _env(name, default).split(",") if item.strip()]


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
REQUEST_TIMEOUT_SECONDS = _env_float("REQUEST_TIMEOUT_SECONDS", 120, minimum=0.1)
NCENTROIDS = _env_int("NCENTROIDS", 256, minimum=1)
NPROBE = _env_int("NPROBE", 10)
TRUST_ENV = _env_bool("TRUST_ENV")
TRAINING_THRESHOLD = _env_int("TRAINING_THRESHOLD", NCENTROIDS * 39, minimum=1)
MODES = _env_csv("MODES", "single,batch")
CONCURRENCIES = [int(value) for value in _env_csv("CONCURRENCY", "1")]

if not MODES or any(mode not in ("single", "batch") for mode in MODES):
    raise ValueError("VEARCH_PERF_MODES must contain only 'single' and/or 'batch'")
if not CONCURRENCIES or any(value < 1 for value in CONCURRENCIES):
    raise ValueError("VEARCH_PERF_CONCURRENCY values must be positive integers")
if NPROBE < -1:
    raise ValueError("NPROBE must be >= -1")


def _assert_api_ok(response, action):
    response.raise_for_status()
    payload = response.json()
    if payload.get("code", 0) != 0:
        raise AssertionError("%s failed: %s" % (action, payload))
    return payload


def _create_ivfflat_space(db_name, space_name, dimension, metric_type):
    _assert_api_ok(create_db(router_url, db_name), "create database")
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
                # IVFFLAT currently requires the RocksDB raw-vector backend.
                "store_type": "RocksDB",
                "index": {
                    "name": "gamma",
                    "type": "IVFFLAT",
                    "params": {
                        "metric_type": metric_type,
                        "ncentroids": NCENTROIDS,
                        "nprobe": NPROBE,
                        "training_threshold": TRAINING_THRESHOLD,
                    },
                },
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
            db_name=db_name,
            space_name=space_name,
        )
    if remainder:
        offset = full_batches * INGEST_BATCH_SIZE
        add(
            remainder,
            1,
            vectors[offset:],
            db_name=db_name,
            space_name=space_name,
            offset=offset,
        )


def _wait_for_index(db_name, space_name, expected_count):
    url = "%s/dbs/%s/spaces/%s" % (router_url, db_name, space_name)
    deadline = time.perf_counter() + INDEX_TIMEOUT_SECONDS
    with requests.Session() as session:
        session.auth = (username, password)
        while True:
            payload = _assert_api_ok(
                session.get(url, timeout=REQUEST_TIMEOUT_SECONDS),
                "get index status",
            )
            indexed_count = sum(
                partition.get("index_num", 0)
                for partition in payload["data"]["partitions"]
            )
            if indexed_count >= expected_count:
                return
            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    "index did not become ready in %.1fs: indexed=%d expected=%d"
                    % (INDEX_TIMEOUT_SECONDS, indexed_count, expected_count)
                )
            time.sleep(min(INDEX_POLL_SECONDS, remaining))


def _base_query(db_name, space_name, metric_type):
    index_params = {"metric_type": metric_type}
    if NPROBE >= 0:
        index_params["nprobe"] = NPROBE
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
        ", ".join("recall@%d=%.2f%%" % (key, value * 100) for key, value in recalls.items()),
    )


def _run_warmup(request_specs, concurrency):
    """Run the configured warmup passes once before the measurement trials."""
    def worker(worker_id):
        errors = 0
        error_examples = []
        with _new_session() as session:
            warmup_failed = False
            for _ in range(WARMUP_ROUNDS):
                for offset in range(len(request_specs)):
                    body, expected = request_specs[(worker_id + offset) % len(request_specs)]
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
                if warmup_failed:
                    break
        return errors, error_examples

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(worker, worker_id)
            for worker_id in range(concurrency)
        ]
        worker_results = [future.result() for future in futures]
    return {
        "errors": sum(errors for errors, _ in worker_results),
        "error_examples": [
            error
            for _, examples in worker_results
            for error in examples
        ][:3],
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
            barrier.wait()
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
    logger.info(
        "PERF SUMMARY mode=%s concurrency=%d trials=%d samples=%d "
        "latency_ms(avg/p50/p95/p99)=%.2f/%.2f/%.2f/%.2f median_vector_qps=%.2f",
        mode,
        concurrency,
        len(trial_results),
        len(latencies),
        stats["avg"],
        stats["p50"],
        stats["p95"],
        stats["p99"],
        float(np.median(np.asarray(vector_qps, dtype=np.float64))),
    )


def test_vearch_index_ivfflat_performance():
    dataset = DatasetSift1M()
    vectors = dataset.get_database()
    query_count = min(QUERY_COUNT, dataset.nq)
    query_vectors = dataset.get_queries()[:query_count]
    groundtruth = np.asarray(dataset.get_groundtruth()[:query_count])
    db_name = _env("DB_NAME", "ts_ivfflat_perf_%d" % os.getpid())
    space_name = _env("SPACE_NAME", "ts_ivfflat_perf_space")

    logger.info(
        "PERF CONFIG index=IVFFLAT vectors=%d dimension=%d queries=%d top_k=%d "
        "modes=%s concurrency=%s warmup_rounds=%d trials=%d seconds=%.1f "
        "ncentroids=%d nprobe=%s training_threshold=%d "
        "latency_scope=http_send trust_env=%s",
        vectors.shape[0],
        vectors.shape[1],
        query_count,
        TOP_K,
        ",".join(MODES),
        ",".join(str(value) for value in CONCURRENCIES),
        WARMUP_ROUNDS,
        TRIALS,
        MEASURE_SECONDS,
        NCENTROIDS,
        "server-default" if NPROBE < 0 else NPROBE,
        TRAINING_THRESHOLD,
        str(TRUST_ENV).lower(),
    )

    database_created = False
    try:
        started = time.perf_counter()
        _create_ivfflat_space(db_name, space_name, vectors.shape[1], dataset.metric)
        database_created = True
        _ingest(db_name, space_name, vectors)
        _wait_for_index(db_name, space_name, vectors.shape[0])
        logger.info("PERF SETUP total_ready_seconds=%.2f", time.perf_counter() - started)
        if SETTLE_SECONDS:
            time.sleep(SETTLE_SECONDS)

        base_query = _base_query(db_name, space_name, dataset.metric)
        batch_specs = _request_specs("batch", query_vectors, base_query)
        _verify_recall(batch_specs[0], groundtruth)
        for mode in MODES:
            specs = batch_specs if mode == "batch" else _request_specs(mode, query_vectors, base_query)
            vectors_per_request = query_count if mode == "batch" else 1
            for concurrency in CONCURRENCIES:
                warmup_result = _run_warmup(specs, concurrency)
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
                    trial_qps.append(_log_trial(mode, concurrency, trial_number, result, vectors_per_request))
                    if result["error_examples"]:
                        logger.error("benchmark error examples: %s", result["error_examples"])
                    assert result["errors"] == 0
                    assert result["successes"] > 0
                _log_summary(mode, concurrency, trial_results, trial_qps)
    finally:
        if database_created:
            destroy(router_url, db_name, space_name)
