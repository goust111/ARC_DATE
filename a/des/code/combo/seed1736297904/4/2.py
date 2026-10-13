from __future__ import annotations

import random

import numpy as np


BG = 0
PAIR_COUNT_RANGE = (2, 5)
HEIGHT_RANGE = (10, 24)
WIDTH_RANGE = (10, 24)
MAX_DISTANCE = 5
OVERLAP_PROB = 0.45
MIXED_ORIENTATION_PROB = 0.7
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250
MAX_FILL_RATIO = 0.55
MIN_FILL_RATIO = 0.08


def _spec_rect(spec: dict) -> tuple[int, int, int, int]:
    if spec["orientation"] == "h":
        return spec["r"] - spec["d"], spec["c0"], spec["r"] + spec["d"], spec["c1"]
    return spec["r0"], spec["c"] - spec["d"], spec["r1"], spec["c"] + spec["d"]


def _spec_seeds(spec: dict) -> list[tuple[int, int]]:
    if spec["orientation"] == "h":
        return [(spec["r"], spec["c0"]), (spec["r"], spec["c1"])]
    return [(spec["r0"], spec["c"]), (spec["r1"], spec["c"])]


def _spec_area(spec: dict) -> int:
    r0, c0, r1, c1 = _spec_rect(spec)
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _seed_pair_to_spec(color: int, points: list[tuple[int, int]], order: int, shape: tuple[int, int]) -> dict:
    assert len(points) == 2
    (r1, c1), (r2, c2) = points
    if r1 == r2:
        c0, c1 = sorted((c1, c2))
        d = c1 - c0 - 1
        spec = {"color": int(color), "orientation": "h", "r": int(r1), "c0": int(c0), "c1": int(c1), "d": int(d)}
    else:
        assert c1 == c2, "same-color seeds must align horizontally or vertically"
        r0, r1 = sorted((r1, r2))
        d = r1 - r0 - 1
        spec = {"color": int(color), "orientation": "v", "r0": int(r0), "r1": int(r1), "c": int(c1), "d": int(d)}

    assert spec["d"] >= 0
    rr0, cc0, rr1, cc1 = _spec_rect(spec)
    height, width = shape
    assert 0 <= rr0 <= rr1 < height
    assert 0 <= cc0 <= cc1 < width
    spec["area"] = _spec_area(spec)
    spec["order"] = int(order)
    return spec


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    coords_by_color: dict[int, list[tuple[int, int]]] = {}
    first_seen: dict[int, int] = {}
    for order, (r, c) in enumerate(np.argwhere(g != BG)):
        color = int(g[r, c])
        coords_by_color.setdefault(color, []).append((int(r), int(c)))
        first_seen.setdefault(color, int(order))

    specs = [
        _seed_pair_to_spec(color, points, first_seen[color], g.shape)
        for color, points in coords_by_color.items()
    ]
    specs.sort(key=lambda spec: (-int(spec["area"]), int(spec["order"])))

    out = np.zeros_like(g)
    for spec in specs:
        r0, c0, r1, c1 = _spec_rect(spec)
        out[r0 : r1 + 1, c0 : c1 + 1] = int(spec["color"])
    return out


def _distance_cap(pair_count: int) -> int:
    if pair_count >= 5:
        return 3
    if pair_count == 4:
        return 4
    return MAX_DISTANCE


def _sample_shape(pair_count: int) -> tuple[int, int]:
    extra = max(0, pair_count - 2)
    h_lo = max(HEIGHT_RANGE[0], 10 + 2 * extra)
    h_hi = min(HEIGHT_RANGE[1], 17 + 2 * extra)
    w_lo = max(WIDTH_RANGE[0], 10 + 2 * extra)
    w_hi = min(WIDTH_RANGE[1], 17 + 2 * extra)
    return random.randint(h_lo, h_hi), random.randint(w_lo, w_hi)


def _random_spec(height: int, width: int, orientation: str, distance_cap: int) -> dict | None:
    if orientation == "h":
        max_d = min(distance_cap, (height - 1) // 2, width - 2)
        if max_d < 1:
            return None
        d = random.randint(1, max_d)
        r = random.randint(d, height - 1 - d)
        c0 = random.randint(0, width - (d + 2))
        return {"orientation": "h", "r": int(r), "c0": int(c0), "c1": int(c0 + d + 1), "d": int(d)}

    max_d = min(distance_cap, height - 2, (width - 1) // 2)
    if max_d < 1:
        return None
    d = random.randint(1, max_d)
    c = random.randint(d, width - 1 - d)
    r0 = random.randint(0, height - (d + 2))
    return {"orientation": "v", "r0": int(r0), "r1": int(r0 + d + 1), "c": int(c), "d": int(d)}


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0] - pad
        or b[2] + pad < a[0] - pad
        or a[3] + pad < b[1] - pad
        or b[3] + pad < a[1] - pad
    )


def _seeds_are_isolated(existing: list[dict], spec: dict) -> bool:
    for r1, c1 in _spec_seeds(spec):
        for other in existing:
            for r2, c2 in _spec_seeds(other):
                if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                    return False
    return True


def _can_place_nonoverlap(existing: list[dict], spec: dict) -> bool:
    if not _seeds_are_isolated(existing, spec):
        return False
    rect = _spec_rect(spec)
    for other in existing:
        if _rects_overlap(rect, _spec_rect(other), pad=1):
            return False
    return True


def _random_cross_spec(height: int, width: int, base: dict, distance_cap: int) -> dict | None:
    if base["orientation"] == "h":
        max_d = min(distance_cap, height - 2, (width - 1) // 2)
        choices = [d for d in range(1, max_d + 1) if d != base["d"]]
        random.shuffle(choices)
        for d in choices:
            cols = [c for c in range(base["c0"] + 1, base["c1"]) if d <= c < width - d]
            if not cols:
                continue
            c = random.choice(cols)
            lo = max(0, base["r"] - d)
            hi = min(height - (d + 2), base["r"] - 1)
            if lo > hi:
                continue
            r0 = random.randint(lo, hi)
            return {"orientation": "v", "r0": int(r0), "r1": int(r0 + d + 1), "c": int(c), "d": int(d)}
        return None

    max_d = min(distance_cap, (height - 1) // 2, width - 2)
    choices = [d for d in range(1, max_d + 1) if d != base["d"]]
    random.shuffle(choices)
    for d in choices:
        rows = [r for r in range(base["r0"] + 1, base["r1"]) if d <= r < height - d]
        if not rows:
            continue
        r = random.choice(rows)
        lo = max(0, base["c"] - d)
        hi = min(width - (d + 2), base["c"] - 1)
        if lo > hi:
            continue
        c0 = random.randint(lo, hi)
        return {"orientation": "h", "r": int(r), "c0": int(c0), "c1": int(c0 + d + 1), "d": int(d)}
    return None


def _make_crossing_pair(height: int, width: int, distance_cap: int) -> list[dict] | None:
    first_orientation = random.choice(("h", "v"))
    for _ in range(PLACEMENT_ATTEMPTS):
        first = _random_spec(height, width, first_orientation, distance_cap)
        if first is None:
            return None
        second = _random_cross_spec(height, width, first, distance_cap)
        if second is None:
            continue
        if _spec_area(first) == _spec_area(second):
            continue
        if not _seeds_are_isolated([first], second):
            continue
        if not _rects_overlap(_spec_rect(first), _spec_rect(second), pad=0):
            continue
        return [first, second]
    return None


def _sample_orientations(count: int) -> list[str]:
    orientations = [random.choice(("h", "v")) for _ in range(count)]
    if count >= 2 and len(set(orientations)) == 1 and random.random() < MIXED_ORIENTATION_PROB:
        flip = random.randrange(count)
        orientations[flip] = "v" if orientations[flip] == "h" else "h"
    return orientations


def _overlap_pairs(specs: list[dict]) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for i in range(len(specs)):
        rect_i = _spec_rect(specs[i])
        for j in range(i + 1, len(specs)):
            if _rects_overlap(rect_i, _spec_rect(specs[j]), pad=0):
                pairs.append((i, j))
    return pairs


def _generate_attempt() -> np.ndarray | None:
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    height, width = _sample_shape(pair_count)
    distance_cap = _distance_cap(pair_count)
    specs: list[dict] = []

    if random.random() < OVERLAP_PROB:
        overlap_pair = _make_crossing_pair(height, width, distance_cap)
        if overlap_pair is None:
            return None
        specs.extend(overlap_pair)

    for orientation in _sample_orientations(pair_count - len(specs)):
        placed = None
        for _ in range(PLACEMENT_ATTEMPTS):
            candidate = _random_spec(height, width, orientation, distance_cap)
            if candidate is not None and _can_place_nonoverlap(specs, candidate):
                placed = candidate
                break
        if placed is None:
            return None
        specs.append(placed)

    if len({int(spec["d"]) for spec in specs}) < 2:
        return None

    overlaps = _overlap_pairs(specs)
    if len(overlaps) > 1:
        return None
    if overlaps:
        i, j = overlaps[0]
        if _spec_area(specs[i]) == _spec_area(specs[j]):
            return None

    colors = random.sample(list(range(1, 10)), len(specs))
    random.shuffle(colors)

    grid = np.zeros((height, width), dtype=int)
    for color, spec in zip(colors, specs):
        for r, c in _spec_seeds(spec):
            if grid[r, c] != BG:
                return None
            grid[r, c] = int(color)

    out = main(grid)
    fill_ratio = float(np.count_nonzero(out)) / float(height * width)
    if not (MIN_FILL_RATIO <= fill_ratio <= MAX_FILL_RATIO):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
