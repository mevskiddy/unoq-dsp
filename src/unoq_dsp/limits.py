"""Bounded model contract, mirrored by runtime/graph.h and qnn.h."""
MAX_TENSORS = 1024
MAX_OPERATORS = 512
ARENA_BYTES = 16*1024*1024
MAX_PLAN_BYTES = 17*1024*1024
MAX_WORK = 2_000_000_000
MAX_CHANNELS = 2048
