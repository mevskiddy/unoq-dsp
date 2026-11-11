"""Static lifetime allocation for bounded graph plans (ABI version 3).

Inputs and outputs of the same operation never alias. Constants remain resident.
The DSP independently validates all overlaps against graph lifetimes on load.
"""
import struct

from .errors import InvalidBundle


def compact_plan(plan, metadata):
    header = list(struct.unpack_from("<8I", plan))
    _, version, nt, no, arena_bytes, inp, out, _ = header
    if version != 2:
        raise InvalidBundle("Memory planner expects a version 2 source plan")
    tensors = [list(struct.unpack_from("<12I", plan, 32+48*i)) for i in range(nt)]
    start_ops = 32+48*nt
    ops = [struct.unpack_from("<8I", plan, start_ops+32*i) for i in range(no)]
    arena_start = start_ops+32*no
    born = [-1]*nt
    died = [-1]*nt
    for index, op in enumerate(ops):
        code, a, b, z, p0, _, p2, _ = op
        dependencies = [a, b]
        if 14 <= code <= 18 or code in (25,27):
            dependencies.append(p0)
        if code in (14,15) and op[5]:dependencies.append(op[5]-1)
        elif code in (6, 9):
            dependencies.append(p2)
        elif code == 19:
            table = tensors[b]
            dependencies.extend(struct.unpack_from("<"+"I"*(table[1]//4), plan, arena_start+table[0]))
        for i in dependencies:
            died[i] = max(index, died[i])
        born[z] = index
        died[z] = max(index, died[z])
    died[out] = no
    offsets = [None]*nt
    cursor = 0
    def aligned(n):
        return (n+127)&~127
    for i, tensor in enumerate(tensors):
        if tensor[8]:
            offsets[i] = cursor
            cursor += aligned(tensor[1])
    constant_bytes = cursor
    active = []
    free = []
    for i in sorted((i for i in range(nt) if not tensors[i][8]), key=lambda i: (born[i], i)):
        still_live = []
        for end, offset, size in active:
            if end < born[i]:
                free.append((offset, size))
            else:
                still_live.append((end, offset, size))
        active = still_live
        merged = []
        for offset, size in sorted(free):
            if merged and merged[-1][0]+merged[-1][1] == offset:
                old, length = merged.pop()
                merged.append((old, length+size))
            else:
                merged.append((offset, size))
        free = merged
        size = aligned(tensors[i][1])
        candidates = [(length, offset, j) for j, (offset, length) in enumerate(free) if length >= size]
        if candidates:
            length, offset, j = min(candidates)
            free.pop(j)
            if length > size:
                free.append((offset+size, length-size))
        else:
            offset = cursor
            cursor += size
        offsets[i] = offset
        active.append((died[i], offset, size))
    arena = bytearray(cursor)
    result_meta = dict(metadata)
    result_meta["tensors"] = [dict(t) for t in metadata["tensors"]]
    for i, tensor in enumerate(tensors):
        old = tensor[0]
        tensor[0] = offsets[i]
        if tensor[8]:
            arena[offsets[i]:offsets[i]+tensor[1]] = plan[arena_start+old:arena_start+old+tensor[1]]
        result_meta["tensors"][i]["offset"] = offsets[i]
    header[1] = 3
    header[4] = len(arena)
    packed = (struct.pack("<8I", *header)
              + b"".join(struct.pack("<12I", *t) for t in tensors)
              + plan[start_ops:arena_start]+arena)
    result_meta.update(input=result_meta["tensors"][inp], output=result_meta["tensors"][out],
                       arena_bytes=len(arena), plan_bytes=len(packed),
                       memory_plan={"original_arena_bytes": arena_bytes,
                                    "resident_constant_bytes_aligned": constant_bytes,
                                    "activation_pool_bytes": len(arena)-constant_bytes,
                                    "reuses_intermediate_buffers": True})
    # End-of-inference snapshots of overwritten intermediates are unavailable.
    result_meta.pop("snapshots", None)
    return bytes(packed), result_meta
