from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


BG = 0
RING_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
]
CARDINAL_GAP_TO_VEC = {
    1: (-1, 0),
    3: (0, 1),
    5: (1, 0),
    7: (0, -1),
}
DIR_TO_GAP = {vec: gap for gap, vec in CARDINAL_GAP_TO_VEC.items()}
CORNER_GAPS = (0, 2, 4, 6)
CARDINAL_DIRS = tuple(CARDINAL_GAP_TO_VEC.values())


def _detect_rings(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    rings: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if int(g[r, c]) != BG:
                continue
            patch = g[r - 1 : r + 2, c - 1 : c + 2]
            perimeter = [int(patch[dr + 1, dc + 1]) for dr, dc in RING_OFFSETS]
            colors = {value for value in perimeter if value != BG}
            if len(colors) != 1:
                continue
            color = next(iter(colors))
            if perimeter.count(color) != 7:
                continue
            missing = [idx for idx, value in enumerate(perimeter) if value == BG]
            if len(missing) != 1:
                continue
            if not np.all((patch == BG) | (patch == color)):
                continue
            gap_idx = int(missing[0])
            pixels = [
                (r + dr, c + dc)
                for idx, (dr, dc) in enumerate(RING_OFFSETS)
                if idx != gap_idx
            ]
            rings.append(
                {
                    "center": (int(r), int(c)),
                    "color": int(color),
                    "gap_idx": gap_idx,
                    "pixels": pixels,
                }
            )
    return rings


def _paint_ring(grid: np.ndarray, ring: dict, color: int) -> None:
    for r, c in ring["pixels"]:
        grid[int(r), int(c)] = int(color)


def _clear_ring(grid: np.ndarray, ring: dict) -> None:
    r, c = ring["center"]
    grid[int(r) - 1 : int(r) + 2, int(c) - 1 : int(c) + 2] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each ring's missing edge cell as a cardinal pointer, transfer colors to the
    immediately adjacent ring in that direction, then erase all emitting rings.
    """
    g = np.asarray(grid, dtype=int)
    rings = _detect_rings(g)
    if not rings:
        return g.copy()

    by_center = {tuple(ring["center"]): ring for ring in rings}
    incoming: dict[tuple[int, int], list[int]] = defaultdict(list)
    senders: list[dict] = []

    for ring in rings:
        gap_idx = int(ring["gap_idx"])
        if gap_idx not in CARDINAL_GAP_TO_VEC:
            continue
        senders.append(ring)
        dr, dc = CARDINAL_GAP_TO_VEC[gap_idx]
        rr, cc = ring["center"]
        target = (int(rr + 4 * dr), int(cc + 4 * dc))
        if target in by_center:
            incoming[target].append(int(ring["color"]))

    out = g.copy()
    for center, colors in incoming.items():
        counts = Counter(map(int, colors))
        best = max(counts.values())
        winners = [color for color, count in counts.items() if int(count) == int(best)]
        if len(winners) == 1:
            _paint_ring(out, by_center[center], int(winners[0]))

    for ring in senders:
        _clear_ring(out, ring)

    return out


SLOT_ROWS_RANGE = (3, 5)
SLOT_COLS_RANGE = (3, 5)
PAD_RANGE = (0, 3)
RING_COUNT_RANGE = (4, 8)
MAX_GEN_ATTEMPTS = 400

SCENARIOS = (
    ("simple", "inert"),
    ("simple", "miss", "inert"),
    ("tie", "inert"),
    ("chain", "inert"),
    ("tie", "miss"),
    ("simple", "chain"),
)


def _inside(slot: tuple[int, int], rows: int, cols: int) -> bool:
    r, c = slot
    return 0 <= int(r) < int(rows) and 0 <= int(c) < int(cols)


def _add(slot: tuple[int, int], vec: tuple[int, int]) -> tuple[int, int]:
    return int(slot[0] + vec[0]), int(slot[1] + vec[1])


def _unused_slots(spec: dict[tuple[int, int], int | None], rows: int, cols: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(rows)
        for c in range(cols)
        if (r, c) not in spec
    ]


def _place_simple(spec: dict[tuple[int, int], int | None], rows: int, cols: int) -> bool:
    candidates = []
    for src in _unused_slots(spec, rows, cols):
        for vec in CARDINAL_DIRS:
            target = _add(src, vec)
            if _inside(target, rows, cols) and target not in spec:
                candidates.append((src, vec, target))
    if not candidates:
        return False
    src, vec, target = random.choice(candidates)
    spec[src] = DIR_TO_GAP[vec]
    spec[target] = None
    return True


def _place_chain(spec: dict[tuple[int, int], int | None], rows: int, cols: int) -> bool:
    candidates = []
    for start in _unused_slots(spec, rows, cols):
        for vec in CARDINAL_DIRS:
            mid = _add(start, vec)
            end = _add(mid, vec)
            if _inside(mid, rows, cols) and _inside(end, rows, cols):
                if mid not in spec and end not in spec:
                    candidates.append((start, vec, mid, end))
    if not candidates:
        return False
    start, vec, mid, end = random.choice(candidates)
    spec[start] = DIR_TO_GAP[vec]
    spec[mid] = DIR_TO_GAP[vec]
    spec[end] = random.choice(CORNER_GAPS)
    return True


def _place_tie(spec: dict[tuple[int, int], int | None], rows: int, cols: int) -> bool:
    source_pairs = (
        ((-1, 0), (1, 0)),
        ((0, -1), (0, 1)),
        ((-1, 0), (0, -1)),
        ((-1, 0), (0, 1)),
        ((1, 0), (0, -1)),
        ((1, 0), (0, 1)),
    )
    candidates = []
    for target in _unused_slots(spec, rows, cols):
        for off1, off2 in source_pairs:
            s1 = _add(target, off1)
            s2 = _add(target, off2)
            if _inside(s1, rows, cols) and _inside(s2, rows, cols):
                if s1 not in spec and s2 not in spec:
                    candidates.append((target, s1, s2))
    if not candidates:
        return False
    target, s1, s2 = random.choice(candidates)
    spec[target] = random.choice(CORNER_GAPS)
    spec[s1] = DIR_TO_GAP[(int(target[0] - s1[0]), int(target[1] - s1[1]))]
    spec[s2] = DIR_TO_GAP[(int(target[0] - s2[0]), int(target[1] - s2[1]))]
    return True


def _place_miss(spec: dict[tuple[int, int], int | None], rows: int, cols: int) -> bool:
    candidates = []
    for r, c in _unused_slots(spec, rows, cols):
        if r == 0:
            candidates.append(((r, c), (-1, 0)))
        if r == rows - 1:
            candidates.append(((r, c), (1, 0)))
        if c == 0:
            candidates.append(((r, c), (0, -1)))
        if c == cols - 1:
            candidates.append(((r, c), (0, 1)))
    if not candidates:
        return False
    slot, vec = random.choice(candidates)
    spec[slot] = DIR_TO_GAP[vec]
    return True


def _place_inert(spec: dict[tuple[int, int], int | None], rows: int, cols: int) -> bool:
    unused = _unused_slots(spec, rows, cols)
    if not unused:
        return False
    spec[random.choice(unused)] = random.choice(CORNER_GAPS)
    return True


def _fill_unspecified_gaps(spec: dict[tuple[int, int], int | None]) -> None:
    for slot, gap in list(spec.items()):
        if gap is not None:
            continue
        if random.random() < 0.65:
            spec[slot] = random.choice(CORNER_GAPS)
        else:
            spec[slot] = random.choice(tuple(range(8)))


def _add_fillers(spec: dict[tuple[int, int], int | None], rows: int, cols: int, target_count: int) -> None:
    while len(spec) < target_count:
        unused = _unused_slots(spec, rows, cols)
        if not unused:
            return
        slot = random.choice(unused)
        kind = random.choices(("inert", "hit", "miss"), weights=(0.5, 0.3, 0.2), k=1)[0]
        if kind == "inert":
            spec[slot] = random.choice(CORNER_GAPS)
            continue
        if kind == "hit":
            neighbors = []
            for vec in CARDINAL_DIRS:
                target = _add(slot, vec)
                if target in spec:
                    neighbors.append(vec)
            if neighbors:
                spec[slot] = DIR_TO_GAP[random.choice(neighbors)]
                continue
        outward = []
        r, c = slot
        if r == 0:
            outward.append((-1, 0))
        if r == rows - 1:
            outward.append((1, 0))
        if c == 0:
            outward.append((0, -1))
        if c == cols - 1:
            outward.append((0, 1))
        if outward:
            spec[slot] = DIR_TO_GAP[random.choice(outward)]
        else:
            spec[slot] = random.choice(CORNER_GAPS)


def _slot_features(spec: dict[tuple[int, int], int]) -> dict[str, object]:
    active = {slot for slot, gap in spec.items() if int(gap) in CARDINAL_GAP_TO_VEC}
    incoming: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    hits = 0
    misses = 0
    for slot in active:
        vec = CARDINAL_GAP_TO_VEC[int(spec[slot])]
        target = _add(slot, vec)
        if target in spec:
            incoming[target].append(slot)
            hits += 1
        else:
            misses += 1
    return {
        "active": active,
        "incoming": incoming,
        "hits": hits,
        "misses": misses,
        "inert": len(spec) - len(active),
        "conflict": any(len(sources) >= 2 for sources in incoming.values()),
        "sender_target": any(target in active for target in incoming),
    }


def _scenario_ok(scenario: tuple[str, ...], features: dict[str, object]) -> bool:
    if "simple" in scenario and int(features["hits"]) < 1:
        return False
    if "miss" in scenario and int(features["misses"]) < 1:
        return False
    if "inert" in scenario and int(features["inert"]) < 1:
        return False
    if "tie" in scenario and not bool(features["conflict"]):
        return False
    if "chain" in scenario and not bool(features["sender_target"]):
        return False
    return True


def _paint_input(
    spec: dict[tuple[int, int], int], rows: int, cols: int
) -> tuple[np.ndarray, dict[tuple[int, int], tuple[int, int]]]:
    top_pad = random.randint(*PAD_RANGE)
    bottom_pad = random.randint(*PAD_RANGE)
    left_pad = random.randint(*PAD_RANGE)
    right_pad = random.randint(*PAD_RANGE)
    height = top_pad + 4 * (rows - 1) + 3 + bottom_pad
    width = left_pad + 4 * (cols - 1) + 3 + right_pad
    grid = np.zeros((height, width), dtype=int)

    colors = random.sample(range(1, 10), len(spec))
    items = list(spec.items())
    random.shuffle(items)
    centers: dict[tuple[int, int], tuple[int, int]] = {}
    for color, (slot, gap_idx) in zip(colors, items):
        center = (top_pad + 1 + 4 * slot[0], left_pad + 1 + 4 * slot[1])
        centers[slot] = center
        for idx, (dr, dc) in enumerate(RING_OFFSETS):
            if int(idx) == int(gap_idx):
                continue
            rr = int(center[0] + dr)
            cc = int(center[1] + dc)
            grid[rr, cc] = int(color)
    return grid, centers


def _gen_attempt() -> np.ndarray | None:
    rows = random.randint(*SLOT_ROWS_RANGE)
    cols = random.randint(*SLOT_COLS_RANGE)
    scenario = random.choice(SCENARIOS)
    spec: dict[tuple[int, int], int | None] = {}

    for motif in scenario:
        if motif == "simple" and not _place_simple(spec, rows, cols):
            return None
        if motif == "chain" and not _place_chain(spec, rows, cols):
            return None
        if motif == "tie" and not _place_tie(spec, rows, cols):
            return None
        if motif == "miss" and not _place_miss(spec, rows, cols):
            return None
        if motif == "inert" and not _place_inert(spec, rows, cols):
            return None

    max_rings = min(9, rows * cols, RING_COUNT_RANGE[1])
    min_rings = min(max_rings, max(len(spec), RING_COUNT_RANGE[0]))
    if min_rings > max_rings:
        return None
    target_count = random.randint(min_rings, max_rings)
    _add_fillers(spec, rows, cols, target_count)
    _fill_unspecified_gaps(spec)

    final_spec = {slot: int(gap) for slot, gap in spec.items() if gap is not None}
    features = _slot_features(final_spec)
    if not _scenario_ok(scenario, features):
        return None

    grid, centers = _paint_input(final_spec, rows, cols)
    detected = _detect_rings(grid)
    if len(detected) != len(final_spec):
        return None

    by_center = {ring["center"]: ring for ring in detected}
    centers_expected = {center: int(final_spec[slot]) for slot, center in centers.items()}
    if set(centers_expected) != set(by_center):
        return None
    if any(int(by_center[center]["gap_idx"]) != gap for center, gap in centers_expected.items()):
        return None

    out = main(grid)
    if np.array_equal(grid, out):
        return None
    if int(np.count_nonzero(out)) == 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
