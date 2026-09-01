"""External-Ray and MI35x inventory checks for the AMD GLM-5.2 launcher."""

import json
import os
import re
import socket
import subprocess
from pathlib import Path
from typing import Protocol
from urllib.parse import urlparse

import ray
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

import miles.utils.external_utils.command_utils as U

_EXPECTED_FULL_NUM_NODES = 8
_EXPECTED_GPUS_PER_NODE = 8
_PROXY_ENV_VARS = (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
)


class _ClusterArgs(Protocol):
    hardware: str
    num_gpus_per_node: int | None
    num_nodes: int


def _sysfs_product_names() -> list[str]:
    product_names = []
    for path in sorted(Path("/sys/class/drm").glob("card*/device/product_name")):
        if re.fullmatch(r"card\d+", path.parents[1].name):
            try:
                product_names.append(path.read_text().strip())
            except OSError:
                continue
    return [name for name in product_names if name]


def _find_named_values(value: object, names: set[str]) -> list[str]:
    if isinstance(value, dict):
        found = [str(item) for key, item in value.items() if key.lower() in names and isinstance(item, str)]
        for item in value.values():
            found.extend(_find_named_values(item, names))
        return found
    if isinstance(value, list):
        return [found for item in value for found in _find_named_values(item, names)]
    return []


def _amd_smi_product_names(output: str) -> list[str]:
    try:
        payload = json.loads(output)
    except json.JSONDecodeError:
        return re.findall(r"^\s*(?:MARKET_NAME|PRODUCT_NAME):\s*(.+?)\s*$", output, flags=re.MULTILINE)

    if isinstance(payload, dict):
        gpu_data = payload.get("gpu_data")
        records = gpu_data if isinstance(gpu_data, list) else [payload]
    else:
        records = payload
    records = records if isinstance(records, list) else [records]
    product_names = []
    for record in records:
        market_names = _find_named_values(record, {"market_name"})
        fallback_names = _find_named_values(record, {"product_name"})
        candidates = market_names or fallback_names
        matching_name = next((name for name in candidates if re.search(r"\bMI3(?:50|55)X\b", name, re.I)), None)
        if matching_name:
            product_names.append(matching_name)
    return product_names


def _product_sku(product_name: object) -> str | None:
    match = re.search(r"\b(MI350X|MI355X)\b", str(product_name), re.I)
    return match.group(1).upper() if match else None


def _probe_product_names(expected_count: int) -> tuple[list[str], str, str | None]:
    sysfs_names = _sysfs_product_names()
    if len(sysfs_names) == expected_count and all(_product_sku(name) for name in sysfs_names):
        return sysfs_names, "sysfs", None

    try:
        result = subprocess.run(
            ["amd-smi", "static", "--asic", "--json"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return sysfs_names, "sysfs-incomplete", repr(error)

    amd_smi_names = _amd_smi_product_names(result.stdout)
    error = None if result.returncode == 0 else f"amd-smi exited with {result.returncode}"
    return amd_smi_names, "amd-smi", error


def _visible_rocm_product_names() -> tuple[str, ...]:
    try:
        import torch
    except ImportError as error:
        raise RuntimeError("Single-node GLM-5.2 execution requires a ROCm PyTorch runtime") from error
    if torch.version.hip is None or not torch.cuda.is_available():
        raise RuntimeError("Single-node GLM-5.2 execution requires visible ROCm GPUs")
    return tuple(torch.cuda.get_device_name(rank) for rank in range(torch.cuda.device_count()))


def _validate_local_hardware(args: _ClusterArgs) -> None:
    if args.num_nodes != 1:
        return
    assert args.num_gpus_per_node is not None
    product_names = _visible_rocm_product_names()
    product_skus = tuple(_product_sku(name) for name in product_names)
    if len(product_names) != args.num_gpus_per_node or product_skus != (args.hardware,) * len(product_names):
        raise RuntimeError(
            f"Single-node GLM-5.2 requires exactly {args.num_gpus_per_node} visible homogeneous "
            f"{args.hardware} GPUs; found {product_names}"
        )
    print(f"Validated {len(product_names)} visible {args.hardware} GPUs: {product_names}")


def _probe_mi35x_node() -> dict[str, object]:
    """Inspect one Ray node without allocating a GPU or invoking a shell."""
    try:
        result = subprocess.run(
            ["rocminfo"],
            check=False,
            capture_output=True,
            text=True,
            timeout=60,
        )
    except (OSError, subprocess.TimeoutExpired) as error:
        return {
            "node_id": str(ray.get_runtime_context().get_node_id()),
            "hostname": socket.gethostname(),
            "gfx950_count": 0,
            "product_names": [],
            "product_source": None,
            "error": repr(error),
        }

    gfx950_count = len(re.findall(r"^\s*Name:\s+gfx950\s*$", result.stdout, flags=re.MULTILINE))
    product_names, product_source, product_error = _probe_product_names(gfx950_count)
    errors = []
    if result.returncode != 0:
        errors.append(f"rocminfo exited with {result.returncode}")
    if product_error:
        errors.append(product_error)
    return {
        "node_id": str(ray.get_runtime_context().get_node_id()),
        "hostname": socket.gethostname(),
        "gfx950_count": gfx950_count,
        "product_names": product_names,
        "product_source": product_source,
        "error": "; ".join(errors) or None,
    }


def _collect_ray_cluster_inventory() -> tuple[
    list[dict[str, object]], list[dict[str, object]], float, str
]:
    """Collect immutable node metadata plus a hard-affinity hardware probe per node."""
    initialized_here = not ray.is_initialized()
    if initialized_here:
        # Multi-node preparation already relies on address="auto" in
        # exec_command_multi_node. Requiring the same local cluster discovery
        # here prevents inspecting one cluster and submitting to another.
        ray.init(address="auto", log_to_driver=False)

    try:
        nodes = sorted(
            (node for node in ray.nodes() if node.get("Alive")),
            key=lambda node: str(node["NodeManagerAddress"]),
        )
        remote_probe = ray.remote(num_cpus=0)(_probe_mi35x_node)
        probe_refs = [
            remote_probe.options(
                scheduling_strategy=NodeAffinitySchedulingStrategy(node_id=str(node["NodeID"]), soft=False)
            ).remote()
            for node in nodes
        ]
        probes = ray.get(probe_refs, timeout=120)
        available_gpus = float(ray.available_resources().get("GPU", 0))
        return nodes, probes, available_gpus, ray.util.get_node_ip_address()
    finally:
        if initialized_here:
            ray.shutdown()


def _ray_address_host(ray_address: str) -> str | None:
    parsed = urlparse(ray_address)
    return parsed.hostname if parsed.scheme in {"http", "https"} else None


def _resolved_dashboard_addresses(host: str, driver_node_address: str) -> set[str]:
    try:
        addresses = {item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return set()
    if host == "localhost" or addresses & {"127.0.0.1", "::1"}:
        addresses.add(driver_node_address)
    return addresses


def _resolved_master_addresses(host: str, node_addresses: tuple[str, ...]) -> set[str]:
    if host in node_addresses:
        return {host}
    try:
        return {item[4][0] for item in socket.getaddrinfo(host, None, type=socket.SOCK_STREAM)}
    except socket.gaierror:
        return set()


def _configure_external_ray_network(node_addresses: tuple[str, ...] = ()) -> None:
    """Keep control-plane and collectives traffic out of HTTP proxy paths."""
    for name in _PROXY_ENV_VARS:
        os.environ.pop(name, None)

    ray_host = _ray_address_host(os.environ.get("RAY_ADDRESS", ""))
    configured = os.environ.get("no_proxy", "") + "," + os.environ.get("NO_PROXY", "")
    bypass = ["127.0.0.1", "localhost"]
    bypass.extend(entry.strip() for entry in configured.split(",") if entry.strip())
    bypass.extend(address for address in (ray_host, *node_addresses) if address)
    value = ",".join(dict.fromkeys(bypass))
    os.environ["no_proxy"] = value
    os.environ["NO_PROXY"] = value


def _validate_external_ray_cluster(args: _ClusterArgs) -> tuple[str, ...]:
    try:
        nodes, probes, available_gpus, driver_node_address = _collect_ray_cluster_inventory()
    except Exception as error:
        raise RuntimeError(
            "Could not inspect the external Ray cluster. Run the launcher on a joined Ray node "
            "where ray.init(address='auto') reaches the same cluster as RAY_ADDRESS."
        ) from error

    assert args.num_nodes == _EXPECTED_FULL_NUM_NODES
    assert args.num_gpus_per_node == _EXPECTED_GPUS_PER_NODE
    expected_total_gpus = args.num_nodes * args.num_gpus_per_node
    errors = []
    if len(nodes) != args.num_nodes:
        errors.append(f"expected exactly {args.num_nodes} alive nodes, found {len(nodes)}")
    if available_gpus != expected_total_gpus:
        errors.append(f"expected {expected_total_gpus} available GPUs, found {available_gpus:g}")

    probes_by_node = {str(probe["node_id"]): probe for probe in probes}
    records = []
    product_skus = []
    for node in nodes:
        node_id = str(node["NodeID"])
        address = str(node["NodeManagerAddress"])
        gpu_resources = float(node.get("Resources", {}).get("GPU", 0))
        probe = probes_by_node.get(node_id)
        if gpu_resources != args.num_gpus_per_node:
            errors.append(
                f"node {address} advertises {gpu_resources:g} GPUs instead of {args.num_gpus_per_node}"
            )
        if probe is None:
            errors.append(f"node {address} did not return its hard-affinity hardware probe")
            continue
        if probe.get("error"):
            errors.append(f"node {address} hardware probe failed: {probe['error']}")
        if probe.get("gfx950_count") != args.num_gpus_per_node:
            errors.append(
                f"node {address} reported {probe.get('gfx950_count', 0)} gfx950 agents "
                f"instead of {args.num_gpus_per_node}"
            )
        product_names = probe.get("product_names")
        if not isinstance(product_names, list) or len(product_names) != args.num_gpus_per_node:
            errors.append(
                f"node {address} reported {len(product_names) if isinstance(product_names, list) else 0} "
                f"GPU product names instead of {args.num_gpus_per_node}"
            )
            product_names = []
        normalized_skus = [sku for product_name in product_names if (sku := _product_sku(product_name))]
        if normalized_skus != [args.hardware] * args.num_gpus_per_node:
            errors.append(
                f"node {address} product names do not match requested {args.hardware}: {product_names}"
            )
        product_skus.extend(normalized_skus)
        records.append(
            {
                "node_id": node_id,
                "address": address,
                "hostname": probe.get("hostname"),
                "gpu_resources": gpu_resources,
                "gfx950_agents": probe.get("gfx950_count"),
                "product_names": product_names,
                "product_source": probe.get("product_source"),
            }
        )

    unexpected_probe_ids = set(probes_by_node) - {str(node["NodeID"]) for node in nodes}
    if unexpected_probe_ids:
        errors.append(f"hardware probes returned unexpected node IDs: {sorted(unexpected_probe_ids)}")
    if set(product_skus) != {args.hardware}:
        errors.append(f"cluster GPU products are not homogeneous {args.hardware}: {sorted(set(product_skus))}")

    node_addresses = tuple(str(node["NodeManagerAddress"]) for node in nodes)
    ray_host = _ray_address_host(os.environ["RAY_ADDRESS"])
    if ray_host is None:
        errors.append("RAY_ADDRESS must be an HTTP(S) Ray dashboard URL with an explicit host")
    else:
        resolved_dashboard = _resolved_dashboard_addresses(ray_host, driver_node_address)
        if not resolved_dashboard:
            errors.append(f"RAY_ADDRESS host {ray_host!r} could not be resolved")
        elif not resolved_dashboard.intersection(node_addresses):
            errors.append(
                f"RAY_ADDRESS host {ray_host!r} resolves outside the inspected cluster: "
                f"{sorted(resolved_dashboard)}"
            )
    master_host = os.environ["MASTER_ADDR"].strip()
    resolved_master = _resolved_master_addresses(master_host, node_addresses)
    if not resolved_master:
        errors.append(f"MASTER_ADDR host {master_host!r} could not be resolved")
    elif not resolved_master.intersection(node_addresses):
        errors.append(
            f"MASTER_ADDR host {master_host!r} resolves outside the inspected cluster: "
            f"{sorted(resolved_master)}"
        )
    if errors:
        raise RuntimeError(
            f"External Ray cluster is not a dedicated homogeneous 8x8 {args.hardware} allocation: "
            + "; ".join(errors)
        )

    print("Validated dedicated GLM-5.2 Ray cluster: " + json.dumps(records, sort_keys=True))
    return node_addresses


def _require_external_ray(args: _ClusterArgs, action: str) -> None:
    if args.num_nodes <= 1:
        if U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"):
            raise RuntimeError(
                f"Single-node GLM-5.2 {action} refuses MILES_SCRIPT_EXTERNAL_RAY=1; "
                "unset it so execution cannot be submitted to an arbitrary external cluster."
            )
        if os.environ.get("RAY_ADDRESS", "").strip():
            raise RuntimeError(
                f"Single-node GLM-5.2 {action} refuses a preconfigured RAY_ADDRESS; "
                "unset it so Miles creates and targets its local Ray runtime."
            )
        master_addr = os.environ.get("MASTER_ADDR", "").strip()
        if master_addr not in {"", "127.0.0.1", "localhost", "::1"}:
            raise RuntimeError(
                f"Single-node GLM-5.2 {action} requires a loopback MASTER_ADDR, got {master_addr!r}."
            )
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        return
    if not U.get_bool_env_var("MILES_SCRIPT_EXTERNAL_RAY"):
        raise RuntimeError(
            f"Multi-node GLM-5.2 {action} requires an already joined Ray cluster. "
            "Export MILES_SCRIPT_EXTERNAL_RAY=1 first."
        )
    if not os.environ.get("RAY_ADDRESS", "").strip():
        raise RuntimeError(
            f"Multi-node GLM-5.2 {action} requires an explicit RAY_ADDRESS for the existing cluster."
        )
    if _ray_address_host(os.environ["RAY_ADDRESS"]) is None:
        raise RuntimeError(
            f"Multi-node GLM-5.2 {action} requires RAY_ADDRESS to be an HTTP(S) Ray dashboard URL."
        )
    if not os.environ.get("MASTER_ADDR", "").strip():
        raise RuntimeError(
            f"Multi-node GLM-5.2 {action} requires an explicit MASTER_ADDR on an inspected Ray node."
        )

    _configure_external_ray_network()
    node_addresses = _validate_external_ray_cluster(args)
    _configure_external_ray_network(node_addresses)
