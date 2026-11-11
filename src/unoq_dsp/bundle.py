"""Versioned, bounded compiled model files. Checksums detect corruption, not trust.

No pickle, executable model code, archive extraction, or implicit downloads.
The native loader additionally validates all graph operations before execution.
"""
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import struct

import numpy as np

from .errors import InvalidBundle
from .limits import MAX_TENSORS, MAX_OPERATORS, ARENA_BYTES, MAX_PLAN_BYTES

MAGIC = b"UQMODEL\0"
HEADER = struct.Struct("<8sIII32s")
MAX_MANIFEST = 4 * 1024 * 1024
MAX_PLAN = MAX_PLAN_BYTES
DTYPES = {1: "<f4", 2: "|u1", 3: "|i1", 4: "<i4"}


def plan_specs(plan):
    if not 32 <= len(plan) <= MAX_PLAN:
        raise InvalidBundle("Invalid plan length")
    magic, version, nt, no, arena, inp, out, flags = struct.unpack_from("<8I", plan)
    if (magic != 0x55474e31 or version not in (2, 3) or not 1 <= nt <= MAX_TENSORS
            or not 1 <= no <= MAX_OPERATORS or not 0 < arena <= ARENA_BYTES
            or max(inp, out) >= nt or flags > 1 or len(plan) != 32 + 48*nt + 32*no + arena):
        raise InvalidBundle("Unsupported or malformed DSP plan header")
    result = {}
    end = 0
    for i in range(nt):
        offset, size, typ, rank, *tail = struct.unpack_from("<12I", plan, 32+48*i)
        dims, constant, reserved = tail[:4], tail[4], tail[5:]
        shape = tuple(dims[:rank])
        if (typ not in DTYPES or rank > 4 or any(d <= 0 for d in shape)
                or constant > 1 or any(reserved) or offset % 128 or (version < 3 and offset < end)
                or offset + size > arena or math.prod(shape)*np.dtype(DTYPES[typ]).itemsize != size):
            raise InvalidBundle(f"Invalid tensor descriptor {i}")
        end = offset + size
        for name, index in (("input", inp), ("output", out)):
            if i == index:
                if constant:
                    raise InvalidBundle(f"{name} cannot be constant")
                result[name] = {"shape": shape, "dtype": np.dtype(DTYPES[typ]), "bytes": size}
    result.update(arena_bytes=arena, plan_bytes=len(plan), dsp_threads=1+flags)
    return result


def json_value(value):
    if isinstance(value, np.dtype):
        return value.str
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


@dataclass(frozen=True)
class CompiledModel:
    plan: bytes
    metadata: dict

    def save(self, path):
        canonical = plan_specs(self.plan)
        for key in ("input", "output"):
            spec = self.metadata[key]
            if tuple(spec["shape"]) != canonical[key]["shape"] or np.dtype(spec["dtype"]) != canonical[key]["dtype"]:
                raise InvalidBundle("Metadata and plan disagree")
        # Retain only the portable API contract, not development snapshots.
        metadata = {k: self.metadata[k] for k in (
            "input", "output", "arena_bytes", "plan_bytes", "macs", "dsp_threads",
            "source_sha256", "source_format", "compiler_version", "memory_plan", "outputs", "task") if k in self.metadata}
        manifest = json.dumps(metadata, default=json_value, allow_nan=False,
                              sort_keys=True, separators=(",", ":")).encode()
        if len(manifest) > MAX_MANIFEST:
            raise InvalidBundle("Manifest too large")
        digest = hashlib.sha256(manifest+self.plan).digest()
        Path(path).write_bytes(HEADER.pack(MAGIC, 1, len(manifest), len(self.plan), digest)+manifest+self.plan)


def load(path):
    with Path(path).open("rb") as f:
        header = f.read(HEADER.size)
        if len(header) != HEADER.size:
            raise InvalidBundle("Truncated model header")
        magic, version, nm, np_, digest = HEADER.unpack(header)
        if magic != MAGIC or version != 1 or not 1 <= nm <= MAX_MANIFEST or not 32 <= np_ <= MAX_PLAN:
            raise InvalidBundle("Unsupported model format or length")
        payload = f.read(nm + np_ + 1)
    if len(payload) != nm+np_ or hashlib.sha256(payload).digest() != digest:
        raise InvalidBundle("Model length/checksum mismatch")
    try:
        metadata = json.loads(payload[:nm])
        if not isinstance(metadata, dict):
            raise InvalidBundle("Expected a model manifest")
        plan = payload[nm:]
        canonical = plan_specs(plan)
        for name in ("input", "output"):
            spec = metadata[name]
            checked = canonical[name]
            if tuple(spec["shape"]) != checked["shape"] or np.dtype(spec["dtype"]) != checked["dtype"]:
                raise InvalidBundle("Manifest I/O does not match the plan")
            if "quantization" in spec:
                q = spec["quantization"]
                if (not math.isfinite(q["scale"]) or q["scale"] <= 0
                        or type(q["zero_point"]) is not int
                        or not np.iinfo(checked['dtype']).min <= q["zero_point"] <= np.iinfo(checked['dtype']).max):
                    raise InvalidBundle("Invalid I/O quantization")
            spec.update(checked)
        for key in ("arena_bytes", "plan_bytes", "dsp_threads"):
            metadata[key] = canonical[key]
        if "outputs" in metadata:
            outputs = metadata["outputs"]
            if not isinstance(outputs, list) or not 1 <= len(outputs) <= 32 or canonical["output"]["dtype"] not in (np.dtype("i1"),np.dtype("u1")):
                raise InvalidBundle("Invalid multiple-output manifest")
            offset, names = 0, set()
            for spec in outputs:
                shape = spec["shape"]
                if (not isinstance(spec["name"], str) or spec["name"] in names or len(shape)>4
                        or any(type(d) is not int or d<=0 for d in shape)
                        or np.dtype(spec["dtype"]) not in (np.dtype("i1"),np.dtype("u1"))
                        or math.prod(shape) != spec["bytes"] or spec["packed_offset"] != offset):
                    raise InvalidBundle("Invalid packed output view")
                names.add(spec["name"])
                offset += spec["bytes"]
                spec["shape"] = tuple(shape)
                spec["dtype"] = np.dtype(spec["dtype"])
                if spec.get("dequantize") or 'quantization' in spec:
                    q=spec["quantization"];bounds=np.iinfo(spec["dtype"])
                    if not math.isfinite(q["scale"]) or q["scale"]<=0 or type(q["zero_point"]) is not int or not bounds.min<=q["zero_point"]<=bounds.max:
                        raise InvalidBundle("Invalid output dequantization")
            if offset != canonical["output"]["bytes"]:
                raise InvalidBundle("Packed output length does not match plan")
    except (KeyError, TypeError, ValueError, OverflowError, RecursionError) as exc:
        if isinstance(exc, InvalidBundle):
            raise
        raise InvalidBundle(f"Invalid manifest: {exc}") from exc
    return CompiledModel(plan, metadata)
