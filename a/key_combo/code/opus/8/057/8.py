from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


MARKER_COLORS = (1, 2, 3, 4)
LENGTH_TO_LINE_COLOR = {
    2: 7,
    3: 7,
    4: 6,
    5: 6,
    6: 5,
    7: 5,
    8: 9,
}
BUCKETS = {
    "short": (2, 3),
    "mid": (4, 5),
    "long": (6, 7),
    "max": (8,),
}
GEN_RETRIES = 500
PLACEMENT_RETRIES = 300
GRID_MIN = 12
GRID_MAX = 16


def _pair_records(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    present = sorted(int(color) for color in np.unique(g) if int(color) != 0)
    assert present, "expected colored markers"
    assert set(present).issubset(MARKER_COLORS), "input may only contain marker colors on black background"

    records: list[dict] = []
    occupied: set[tuple[int, int]] = set()
    for color in present:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), f"marker color {color} must appear exactly twice"
        (r1, c1), (r2, c2) = [tuple(map(int, coord)) for coord in coords]
        same_row = r1 == r2
        same_col = c1 == c2
        assert same_row ^ same_col, f"marker color {color} must align in one axis"

        if same_row:
            row = r1
            c0, c1_sorted = sorted((c1, c2))
            distance = c1_sorted - c0 - 1
            interior = [(row, c) for c in range(c0 + 1, c1_sorted)]
            segment = {(row, c) for c in range(c0, c1_sorted + 1)}
            endpoints = ((row, c0), (row, c1_sorted))
            orientation = "H"
        else:
            col = c1
            r0, r1_sorted = sorted((r1, r2))
            distance = r1_sorted - r0 - 1
            interior = [(r, col) for r in range(r0 + 1, r1_sorted)]
            segment = {(r, col) for r in range(r0, r1_sorted + 1)}
            endpoints = ((r0, col), (r1_sorted, col))
            orientation = "V"

        assert 2 <= distance <= 8, f"marker spacing must be 2-8, got {distance}"
        for r, c in interior:
            assert g[r, c] == 0, "the path between paired markers must be empty in the input"
        assert occupied.isdisjoint(segment), "marker-pair segments must not intersect"
        occupied |= segment

        records.append(
            {
                "color": color,
                "orientation": orientation,
                "distance": distance,
                "endpoints": endpoints,
                "interior": interior,
            }
        )

    assert 2 <= len(records) <= 4, "expected 2-4 marker pairs"
    return sorted(records, key=lambda item: item["color"])


def _buffer_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _bucket_name(distance: int) -> str:
    for name, values in BUCKETS.items():
        if distance in values:
            return name
    raise AssertionError(f"unexpected distance {distance}")


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-color aligned marker pair, then recolor the connecting line by distance bucket.

    Input structure:
    - The grid is a black background with 2-4 marker colors chosen from blue, red, green, and yellow.
    - Each used marker color appears exactly twice, and its two pixels lie on the same row or the same column.
    - The cells strictly between paired markers are black, and different future line segments do not intersect.

    Rule:
    - For each marker color, locate its two pixels and connect them along their shared row or column.
    - Keep the two endpoint markers unchanged, and recolor only the black cells strictly between them.
    - The fill color depends only on the number of interior cells: 2-3 -> orange(7), 4-5 -> magenta(6), 6-7 -> grey(5), 8 -> maroon(9).

    Key insight:
    - Marker colors identify which endpoints belong together, but the drawn segment color ignores those marker colors and is determined solely by the pair's gap length.
    """
    g = np.asarray(grid, dtype=int)
    pairs = _pair_records(g)

    out = g.copy()
    for pair in pairs:
        line_color = LENGTH_TO_LINE_COLOR[pair["distance"]]
        for r, c in pair["interior"]:
            out[r, c] = line_color
    return out


def _choose_pair_count() -> int:
    return random.choices([2, 3, 4], weights=[2, 5, 3], k=1)[0]


def _sample_distances(pair_count: int) -> list[int]:
    bucket_names = list(BUCKETS)
    if pair_count == 2:
        chosen_buckets = random.sample(bucket_names, 2)
    elif pair_count == 3 and random.random() < 0.65:
        repeated = random.choice(["short", "mid", "long"])
        other = random.choice([name for name in bucket_names if name != repeated])
        chosen_buckets = [repeated, repeated, other]
    elif pair_count == 4 and random.random() < 0.55:
        chosen_buckets = bucket_names[:]
        random.shuffle(chosen_buckets)
    else:
        distinct = random.sample(bucket_names, min(pair_count, len(bucket_names)))
        while len(distinct) < pair_count:
            distinct.append(random.choice(bucket_names))
        chosen_buckets = distinct
        random.shuffle(chosen_buckets)

    by_bucket: Counter[str] = Counter(chosen_buckets)
    distances: list[int] = []
    for bucket in chosen_buckets:
        values = list(BUCKETS[bucket])
        if by_bucket[bucket] > 1 and len(values) > 1:
            existing = [d for d in distances if _bucket_name(d) == bucket]
            remaining = [d for d in values if d not in existing]
            distances.append(random.choice(remaining or values))
        else:
            distances.append(random.choice(values))

    assert len({_bucket_name(d) for d in distances}) >= 2
    return distances


def _sample_orientations(pair_count: int) -> list[str]:
    if pair_count == 2:
        return random.sample(["H", "V"], 2)
    if pair_count == 3:
        return random.choice([["H", "H", "V"], ["H", "V", "V"]])
    return random.sample(["H", "H", "V", "V"], 4)


def _segment_cells(spec: dict) -> set[tuple[int, int]]:
    if spec["orientation"] == "H":
        return {(spec["row"], c) for c in range(spec["start"], spec["end"] + 1)}
    return {(r, spec["col"]) for r in range(spec["start"], spec["end"] + 1)}


def _make_candidate(color: int, orientation: str, distance: int, shape: tuple[int, int]) -> dict | None:
    h, w = shape
    span = distance + 2
    if orientation == "H":
        if w < span + 2:
            return None
        row = random.randint(1, h - 2)
        start = random.randint(1, w - span - 1)
        end = start + span - 1
        return {"color": color, "orientation": orientation, "distance": distance, "row": row, "start": start, "end": end}

    if h < span + 2:
        return None
    col = random.randint(1, w - 2)
    start = random.randint(1, h - span - 1)
    end = start + span - 1
    return {"color": color, "orientation": orientation, "distance": distance, "col": col, "start": start, "end": end}


def _endpoints(spec: dict) -> tuple[tuple[int, int], tuple[int, int]]:
    if spec["orientation"] == "H":
        return (spec["row"], spec["start"]), (spec["row"], spec["end"])
    return (spec["start"], spec["col"]), (spec["end"], spec["col"])


def _has_alignment_confuser(specs: list[dict]) -> bool:
    row_colors: defaultdict[int, set[int]] = defaultdict(set)
    col_colors: defaultdict[int, set[int]] = defaultdict(set)
    for spec in specs:
        for r, c in _endpoints(spec):
            row_colors[r].add(spec["color"])
            col_colors[c].add(spec["color"])
    return any(len(colors) >= 2 for colors in row_colors.values()) or any(len(colors) >= 2 for colors in col_colors.values())


def _render_input(specs: list[dict], shape: tuple[int, int]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for spec in specs:
        for r, c in _endpoints(spec):
            grid[r, c] = spec["color"]
    return grid


def _validate_example(grid: np.ndarray) -> None:
    pairs = _pair_records(grid)
    assert len({_bucket_name(pair["distance"]) for pair in pairs}) >= 2
    orientations = {pair["orientation"] for pair in pairs}
    assert orientations == {"H", "V"}, "examples must expose both horizontal and vertical pairing"

    row_colors: defaultdict[int, set[int]] = defaultdict(set)
    col_colors: defaultdict[int, set[int]] = defaultdict(set)
    for pair in pairs:
        for r, c in pair["endpoints"]:
            row_colors[r].add(pair["color"])
            col_colors[c].add(pair["color"])
    assert any(len(colors) >= 2 for colors in row_colors.values()) or any(
        len(colors) >= 2 for colors in col_colors.values()
    ), "need at least one misleading shared row/column among different colors"

    out = main(grid)
    changed = set(map(tuple, np.argwhere(out != grid)))
    expected_changed = {cell for pair in pairs for cell in pair["interior"]}
    assert changed == expected_changed, "only connector interiors should change"

    for pair in pairs:
        line_color = LENGTH_TO_LINE_COLOR[pair["distance"]]
        for r, c in pair["interior"]:
            assert out[r, c] == line_color
        for r, c in pair["endpoints"]:
            assert out[r, c] == grid[r, c] == pair["color"]


def _gen_attempt() -> np.ndarray | None:
    pair_count = _choose_pair_count()
    colors = random.sample(list(MARKER_COLORS), pair_count)
    orientations = _sample_orientations(pair_count)
    distances = _sample_distances(pair_count)

    dims = list(range(GRID_MIN, GRID_MAX + 1))
    random.shuffle(dims)
    for h in dims:
        for w in random.sample(dims, k=len(dims)):
            specs: list[dict] = []
            blocked: set[tuple[int, int]] = set()
            order = sorted(
                zip(colors, orientations, distances),
                key=lambda item: item[2],
                reverse=True,
            )
            ok = True
            for color, orientation, distance in order:
                placed = None
                for _ in range(PLACEMENT_RETRIES):
                    candidate = _make_candidate(color, orientation, distance, (h, w))
                    if candidate is None:
                        ok = False
                        break
                    cells = _segment_cells(candidate)
                    if cells & blocked:
                        continue
                    placed = candidate
                    specs.append(candidate)
                    blocked |= _buffer_cells(cells, (h, w))
                    break
                if not ok or placed is None:
                    ok = False
                    break
            if not ok:
                continue
            if not _has_alignment_confuser(specs):
                continue

            grid = _render_input(specs, (h, w))
            try:
                _validate_example(grid)
            except AssertionError:
                continue
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid distance-coded connector puzzle")
