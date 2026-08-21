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

"""Stable client-side benchmark for a Vearch IVFPQ index.

This is intentionally separate from the functional IVFPQ tests. Database
creation, ingestion, index construction, and settling are excluded from the
search measurement window.

The benchmark is configured through environment variables so that the same
test can be used for quick checks and longer performance runs:

* VEARCH_PERF_MODES: comma-separated ``single`` and/or ``batch`` (default:
  ``single,batch``)
* VEARCH_PERF_CONCURRENCY: comma-separated positive integers (default: ``1``)
* VEARCH_PERF_WARMUP_ROUNDS: full passes over the query set per worker
  (default: ``10``)
* VEARCH_PERF_TRIALS: number of independent measurement trials (default: ``5``)
* VEARCH_PERF_SECONDS: duration of each trial (default: ``30``)
* VEARCH_PERF_QUERY_COUNT: number of dataset queries in a batch (default: ``100``)
* VEARCH_PERF_SETTLE_SECONDS: wait after the index becomes ready (default: ``10``)
* VEARCH_PERF_NPROBE: query nprobe; values below zero use the server default
  (default: ``10`` for the SIFT1M/256 configuration)
"""

import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
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

# ``vearch_utils`` installs its own uncolored console handler. Let pytest be
# the only console log renderer so INFO records are emitted once and colored.
logger.handlers.clear()
logger.propagate = True
logger.setLevel("INFO")


def _env_int(name, default, minimum=None):
    value = int(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %s, got %s" % (name, minimum, value))
    return value


def _env_float(name, default, minimum=None):
    value = float(os.getenv(name, str(default)))
    if minimum is not None and value < minimum:
        raise ValueError("%s must be >= %s, got %s" % (name, minimum, value))
    return value


def _env_csv(name, default):
    values = [item.strip() for item in os.getenv(name, default).split(",")]
    return [item for item in values if item]


QUERY_COUNT = _env_int("VEARCH_PERF_QUERY_COUNT", 100, minimum=1)
TOP_K = _env_int("VEARCH_PERF_TOP_K", 100, minimum=1)
INGEST_BATCH_SIZE = _env_int("VEARCH_PERF_INGEST_BATCH_SIZE", 100, minimum=1)
WARMUP_ROUNDS = _env_int("VEARCH_PERF_WARMUP_ROUNDS", 10, minimum=0)
TRIALS = _env_int("VEARCH_PERF_TRIALS", 5, minimum=1)
MEASURE_SECONDS = _env_float("VEARCH_PERF_SECONDS", 30, minimum=0.1)
SETTLE_SECONDS = _env_float("VEARCH_PERF_SETTLE_SECONDS", 10, minimum=0)
INDEX_TIMEOUT_SECONDS = _env_float(
    "VEARCH_PERF_INDEX_TIMEOUT_SECONDS", 1800, minimum=1
)
INDEX_POLL_SECONDS = _env_float("VEARCH_PERF_INDEX_POLL_SECONDS", 2, minimum=0.1)
REQUEST_TIMEOUT_SECONDS = _env_float(
    "VEARCH_PERF_REQUEST_TIMEOUT_SECONDS", 120, minimum=0.1
)
NPROBE = _env_int("VEARCH_PERF_NPROBE", 10)
MODES = _env_csv("VEARCH_PERF_MODES", "single,batch")
CONCURRENCIES = [
    int(value) for value in _env_csv("VEARCH_PERF_CONCURRENCY", "1")
]

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


def _create_ivfpq_space(db_name, space_name, dimension, store_type, index_params):
    response = create_db(router_url, db_name)
    _assert_api_ok(response, "create database")

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
                "store_type": store_type,
                "index": {
                    "name": "gamma",
                    "type": "IVFPQ",
                    "params": index_params,
                },
            },
        ],
    }
    try:
        response = create_space(router_url, db_name, space_config)
        _assert_api_ok(response, "create space")
    except Exception:
        destroy(router_url, db_name, space_name)
        raise


def _ingest(db_name, space_name, xb):
    full_batches, remainder = divmod(xb.shape[0], INGEST_BATCH_SIZE)
    if full_batches:
        add(
            full_batches,
            INGEST_BATCH_SIZE,
            xb[: full_batches * INGEST_BATCH_SIZE],
            db_name=db_name,
            space_name=space_name,
        )
    if remainder:
        offset = full_batches * INGEST_BATCH_SIZE
        add(
            remainder,
            1,
            xb[offset:],
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
            response = session.get(url, timeout=REQUEST_TIMEOUT_SECONDS)
            payload = _assert_api_ok(response, "get index status")
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


def _serialize_query(base_query, features):
    query = dict(base_query)
    query["vectors"] = [{"field": "field_vector", "feature": features}]
    return json.dumps(query, separators=(",", ":"))


def _request_specs(mode, query_vectors, base_query):
    if mode == "batch":
        body = _serialize_query(base_query, query_vectors.reshape(-1).tolist())
        return [(body, query_vectors.shape[0])]

    return [
        (_serialize_query(base_query, vector.tolist()), 1)
        for vector in query_vectors
    ]


def _new_session():
    session = requests.Session()
    session.auth = (username, password)
    session.headers.update({"Content-Type": "application/json"})
    return session


def _execute_search(session, body, expected_query_count):
    url = router_url + "/document/search?timeout=2000000"
    response = session.post(
        url,
        data=body,
        timeout=REQUEST_TIMEOUT_SECONDS,
    )
    payload = _assert_api_ok(response, "search")
    documents = payload.get("data", {}).get("documents")
    if not isinstance(documents, list) or len(documents) != expected_query_count:
        raise AssertionError(
            "search returned %s query result groups, expected %d"
            % (
                len(documents) if isinstance(documents, list) else None,
                expected_query_count,
            )
        )
    return documents


def _verify_recall(batch_spec, groundtruth):
    body, expected_query_count = batch_spec
    with _new_session() as session:
        documents = _execute_search(session, body, expected_query_count)

    result_ids = []
    for results in documents:
        ids = [result["field_int"] for result in results[:TOP_K]]
        ids.extend([-1] * (TOP_K - len(ids)))
        result_ids.append(ids)

    neighbors = np.asarray(result_ids)
    recalls = {}
    recall_at = 1
    while recall_at <= TOP_K:
        recalls[recall_at] = (
            (neighbors[:, :recall_at] == groundtruth[:, :1]).sum()
            / float(neighbors.shape[0])
        )
        recall_at *= 10
    logger.info(
        "correctness check: %s",
        ", ".join(
            "recall@%d=%.2f%%" % (key, value * 100)
            for key, value in recalls.items()
        ),
    )


def _error_text(error):
    text = "%s: %s" % (type(error).__name__, error)
    return text[:500]


def _run_trial(request_specs, concurrency):
    state = {}

    def start_measurement():
        state["start"] = time.perf_counter()
        state["deadline"] = state["start"] + MEASURE_SECONDS

    barrier = threading.Barrier(concurrency + 1, action=start_measurement)

    def worker(worker_id):
        latencies_ms = []
        errors = 0
        warmup_errors = 0
        error_examples = []
        request_index = worker_id

        with _new_session() as session:
            warmup_failed = False
            for _ in range(WARMUP_ROUNDS):
                for offset in range(len(request_specs)):
                    body, expected_query_count = request_specs[
                        (worker_id + offset) % len(request_specs)
                    ]
                    try:
                        _execute_search(session, body, expected_query_count)
                    except Exception as error:
                        warmup_errors += 1
                        error_examples.append("warmup: " + _error_text(error))
                        warmup_failed = True
                        break
                if warmup_failed:
                    break

            barrier.wait()

            if not warmup_failed:
                while time.perf_counter() < state["deadline"]:
                    body, expected_query_count = request_specs[
                        request_index % len(request_specs)
                    ]
                    started = time.perf_counter()
                    try:
                        _execute_search(session, body, expected_query_count)
                    except Exception as error:
                        errors += 1
                        if len(error_examples) < 3:
                            error_examples.append(_error_text(error))
                    else:
                        latencies_ms.append((time.perf_counter() - started) * 1000.0)
                    request_index += concurrency

        return {
            "latencies_ms": latencies_ms,
            "errors": errors,
            "warmup_errors": warmup_errors,
            "error_examples": error_examples,
            "finished": time.perf_counter(),
        }

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(worker, worker_id) for worker_id in range(concurrency)]
        barrier.wait()
        worker_results = [future.result() for future in futures]

    elapsed = max(result["finished"] for result in worker_results) - state["start"]
    latencies_ms = [
        latency
        for result in worker_results
        for latency in result["latencies_ms"]
    ]
    errors = sum(result["errors"] for result in worker_results)
    warmup_errors = sum(result["warmup_errors"] for result in worker_results)
    error_examples = [
        error
        for result in worker_results
        for error in result["error_examples"]
    ][:3]
    successes = len(latencies_ms)
    attempts = successes + errors

    return {
        "latencies_ms": latencies_ms,
        "elapsed": elapsed,
        "successes": successes,
        "errors": errors,
        "warmup_errors": warmup_errors,
        "error_rate": errors * 100.0 / attempts if attempts else 100.0,
        "error_examples": error_examples,
    }


def _latency_stats(latencies_ms):
    values = np.asarray(latencies_ms, dtype=np.float64)
    if values.size == 0:
        return {
            "avg": float("nan"),
            "p50": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "stddev": float("nan"),
            "cv": float("nan"),
        }

    average = float(np.mean(values))
    stddev = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "avg": average,
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
        "p99": float(np.percentile(values, 99)),
        "stddev": stddev,
        "cv": stddev * 100.0 / average if average else 0.0,
    }


def _log_trial(mode, concurrency, trial_number, result, vectors_per_request):
    stats = _latency_stats(result["latencies_ms"])
    request_qps = result["successes"] / result["elapsed"]
    vector_qps = request_qps * vectors_per_request
    logger.info(
        "PERF trial=%d mode=%s concurrency=%d samples=%d "
        "latency_ms(avg/p50/p95/p99)=%.2f/%.2f/%.2f/%.2f "
        "stddev_ms=%.2f cv=%.2f%% request_qps=%.2f vector_qps=%.2f "
        "errors=%d error_rate=%.4f%%",
        trial_number,
        mode,
        concurrency,
        result["successes"],
        stats["avg"],
        stats["p50"],
        stats["p95"],
        stats["p99"],
        stats["stddev"],
        stats["cv"],
        request_qps,
        vector_qps,
        result["errors"],
        result["error_rate"],
    )
    return {
        "stats": stats,
        "request_qps": request_qps,
        "vector_qps": vector_qps,
    }


def _log_summary(mode, concurrency, trial_results, calculated_results):
    all_latencies = [
        latency
        for trial_result in trial_results
        for latency in trial_result["latencies_ms"]
    ]
    latency = _latency_stats(all_latencies)
    vector_qps_values = np.asarray(
        [result["vector_qps"] for result in calculated_results],
        dtype=np.float64,
    )
    throughput_average = float(np.mean(vector_qps_values))
    throughput_stddev = (
        float(np.std(vector_qps_values, ddof=1))
        if vector_qps_values.size > 1
        else 0.0
    )
    throughput_cv = (
        throughput_stddev * 100.0 / throughput_average
        if throughput_average
        else 0.0
    )

    logger.info(
        "PERF SUMMARY mode=%s concurrency=%d trials=%d samples=%d "
        "latency_ms(avg/p50/p95/p99)=%.2f/%.2f/%.2f/%.2f "
        "median_vector_qps=%.2f vector_qps_cv=%.2f%%",
        mode,
        concurrency,
        len(trial_results),
        len(all_latencies),
        latency["avg"],
        latency["p50"],
        latency["p95"],
        latency["p99"],
        float(np.median(vector_qps_values)),
        throughput_cv,
    )


@pytest.mark.parametrize(
    ("store_type", "ncentroids"),
    [
        ("MemoryOnly", 256),
        ("RocksDB", 256),
    ],
)
def test_vearch_index_ivfpq_performance(store_type, ncentroids):
    dataset = DatasetSift1M()
    xb = dataset.get_database()
    query_count = min(QUERY_COUNT, dataset.nq)
    xq = dataset.get_queries()[:query_count]
    groundtruth = np.asarray(dataset.get_groundtruth()[:query_count])

    db_name = os.getenv(
        "VEARCH_PERF_DB_NAME",
        "ts_ivfpq_perf_%s_%d" % (store_type.lower(), os.getpid()),
    )
    space_name = os.getenv("VEARCH_PERF_SPACE_NAME", "ts_ivfpq_perf_space")
    index_params = {
        "metric_type": dataset.metric,
        "ncentroids": ncentroids,
        "nsubvector": 32,
    }

    logger.info(
        "PERF CONFIG store_type=%s vectors=%d dimension=%d queries=%d top_k=%d "
        "modes=%s concurrency=%s warmup_rounds=%d trials=%d seconds=%.1f "
        "nprobe=%s",
        store_type,
        xb.shape[0],
        xb.shape[1],
        query_count,
        TOP_K,
        ",".join(MODES),
        ",".join(str(value) for value in CONCURRENCIES),
        WARMUP_ROUNDS,
        TRIALS,
        MEASURE_SECONDS,
        "server-default" if NPROBE < 0 else NPROBE,
    )

    database_created = False
    try:
        setup_started = time.perf_counter()
        _create_ivfpq_space(
            db_name,
            space_name,
            xb.shape[1],
            store_type,
            index_params,
        )
        database_created = True

        ingest_started = time.perf_counter()
        _ingest(db_name, space_name, xb)
        ingest_elapsed = time.perf_counter() - ingest_started

        index_wait_started = time.perf_counter()
        _wait_for_index(db_name, space_name, xb.shape[0])
        index_wait_elapsed = time.perf_counter() - index_wait_started
        setup_elapsed = time.perf_counter() - setup_started

        logger.info(
            "PERF SETUP ingest_seconds=%.2f ingest_ack_docs_per_second=%.2f "
            "post_ingest_index_wait_seconds=%.2f total_ready_seconds=%.2f",
            ingest_elapsed,
            xb.shape[0] / ingest_elapsed,
            index_wait_elapsed,
            setup_elapsed,
        )

        if SETTLE_SECONDS:
            logger.info(
                "waiting %.1fs for caches and background work to settle",
                SETTLE_SECONDS,
            )
            time.sleep(SETTLE_SECONDS)

        base_query = _base_query(db_name, space_name, dataset.metric)
        batch_specs = _request_specs("batch", xq, base_query)
        _verify_recall(batch_specs[0], groundtruth)

        for mode in MODES:
            specs = (
                batch_specs
                if mode == "batch"
                else _request_specs(mode, xq, base_query)
            )
            vectors_per_request = query_count if mode == "batch" else 1

            for concurrency in CONCURRENCIES:
                trial_results = []
                calculated_results = []
                for trial_number in range(1, TRIALS + 1):
                    result = _run_trial(specs, concurrency)
                    trial_results.append(result)
                    calculated_results.append(
                        _log_trial(
                            mode,
                            concurrency,
                            trial_number,
                            result,
                            vectors_per_request,
                        )
                    )
                    if result["error_examples"]:
                        logger.error(
                            "benchmark error examples: %s",
                            result["error_examples"],
                        )
                    assert result["warmup_errors"] == 0
                    assert result["errors"] == 0
                    assert result["successes"] > 0

                _log_summary(
                    mode,
                    concurrency,
                    trial_results,
                    calculated_results,
                )
    finally:
        if database_created:
            destroy(router_url, db_name, space_name)
