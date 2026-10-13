import random

import numpy as np


BG = 0
RED = 2
GREY = 5
PINK = 6
TEAL = 8

MIN_HEIGHT = 16
MAX_HEIGHT = 30
MIN_WIDTH = 19
MAX_WIDTH = 30
GENERATION_ATTEMPTS = 500


def _extract_marker_and_bars(g: np.ndarray) -> tuple[int, list[dict]]:
    h, w = map(int, g.shape)
    assert h >= 3 and w >= 3, "expected a nontrivial grid"
    assert set(map(int, np.unique(g))) <= {BG, RED, GREY}, "unexpected input color"

    marker = np.argwhere(g == GREY)
    assert marker.shape == (1, 2), "expected exactly one grey marker"
    marker_r, marker_c = map(int, marker[0])
    assert marker_r == h - 1, "grey marker must be on the bottom border"

    bars: list[dict] = []
    red_cols = sorted(int(c) for c in np.unique(np.argwhere(g == RED)[:, 1]))
    assert red_cols, "expected at least one red bar"
    for col in red_cols:
        assert col != marker_c, "marker column must not contain a bar"
        rows = np.flatnonzero(g[:, col] == RED)
        assert rows.size > 0, "bar column must contain red pixels"
        expected_rows = np.arange(int(rows[0]), int(rows[-1]) + 1)
        assert np.array_equal(rows, expected_rows), "each red bar must be contiguous"

        length = int(rows.size)
        distance = abs(col - marker_c)
        assert distance > 0, "bar must be left or right of the marker"
        assert length > distance, "bar length must exceed marker distance"
        bars.append(
            {
                "col": int(col),
                "r0": int(rows[0]),
                "r1": int(rows[-1]),
                "length": length,
                "distance": distance,
                "side": "left" if col < marker_c else "right",
            }
        )

    assert sum(bar["length"] for bar in bars) == int(np.count_nonzero(g == RED))
    return marker_c, bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the end of each floating red vertical bar by its horizontal distance from the bottom grey marker.

    Input structure:
    - A black background contains separated one-cell-wide red vertical bars floating at varied row intervals.
    - The bottom border contains exactly one grey marker pixel, and its column contains no red bar.
    - Every red bar is longer than its horizontal distance from the grey marker column.

    Rule:
    - For each red bar, let d be the absolute column distance from the bar to the grey marker.
    - Bars left of the marker have their lowest d cells recolored teal.
    - Bars right of the marker have their highest d cells recolored pink.
    - All other red cells, the grey marker, and the black background remain unchanged.

    Key insight:
    - The recolored height is not a property of the bar itself; the bottom marker supplies an invisible horizontal ruler, while the side of the marker selects which bar endpoint is recolored.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    _marker_c, bars = _extract_marker_and_bars(g)
    out = g.copy()
    for bar in bars:
        c = bar["col"]
        d = bar["distance"]
        if bar["side"] == "left":
            out[bar["r1"] - d + 1 : bar["r1"] + 1, c] = TEAL
        else:
            out[bar["r0"] : bar["r0"] + d, c] = PINK
    return out


def _spaced_distances(limit: int, count: int, required: int) -> list[int] | None:
    values = list(range(1, int(limit) + 1))
    for _ in range(200):
        chosen = {int(required)}
        random.shuffle(values)
        for d in values:
            if len(chosen) >= count:
                break
            if d in chosen:
                continue
            if all(abs(d - old) >= 2 for old in chosen):
                chosen.add(int(d))
        if len(chosen) == count:
            return sorted(chosen)
    return None


def _has_mixed_length_distance_order(records: list[dict]) -> bool:
    shorter_farther = False
    longer_farther = False
    for i, a in enumerate(records):
        for j, b in enumerate(records):
            if i == j or a["distance"] >= b["distance"]:
                continue
            if a["length"] > b["length"]:
                shorter_farther = True
            if a["length"] < b["length"]:
                longer_farther = True
    return shorter_farther and longer_farther


def _choose_lengths(records: list[dict], height: int) -> list[int] | None:
    possible = list(range(4, height - 3))
    if len(possible) < len(records):
        return None

    for _ in range(300):
        lengths = random.sample(possible, len(records))
        if any(length <= rec["distance"] for length, rec in zip(lengths, records, strict=False)):
            continue
        for rec, length in zip(records, lengths, strict=False):
            rec["length"] = int(length)
        if _has_mixed_length_distance_order(records):
            return lengths
    return None


def _choose_row_intervals(records: list[dict], height: int) -> bool:
    order = list(range(len(records)))
    for _ in range(300):
        random.shuffle(order)
        used_tops: set[int] = set()
        used_bottoms: set[int] = set()
        assigned: list[tuple[int, int] | None] = [None] * len(records)
        ok = True
        for idx in order:
            length = int(records[idx]["length"])
            candidates = list(range(1, height - length))
            random.shuffle(candidates)
            for r0 in candidates:
                r1 = r0 + length - 1
                if r0 in used_tops or r1 in used_bottoms:
                    continue
                used_tops.add(r0)
                used_bottoms.add(r1)
                assigned[idx] = (r0, r1)
                break
            else:
                ok = False
                break
        if ok:
            for rec, interval in zip(records, assigned, strict=False):
                assert interval is not None
                rec["r0"], rec["r1"] = interval
            return True
    return False


def _expected_from_records(grid: np.ndarray, records: list[dict]) -> np.ndarray:
    out = grid.copy()
    for rec in records:
        c = int(rec["col"])
        d = int(rec["distance"])
        if rec["side"] == "left":
            out[int(rec["r1"]) - d + 1 : int(rec["r1"]) + 1, c] = TEAL
        else:
            out[int(rec["r0"]) : int(rec["r0"]) + d, c] = PINK
    return out


def _semantic_checks(grid: np.ndarray, records: list[dict], expected: np.ndarray) -> bool:
    if grid.ndim != 2 or expected.shape != grid.shape or max(grid.shape) > 30:
        return False
    if not records or np.array_equal(grid, expected):
        return False

    marker = np.argwhere(grid == GREY)
    if marker.shape != (1, 2) or int(marker[0, 0]) != grid.shape[0] - 1:
        return False
    marker_c = int(marker[0, 1])

    left_ds = {rec["distance"] for rec in records if rec["side"] == "left"}
    right_ds = {rec["distance"] for rec in records if rec["side"] == "right"}
    if not left_ds or not right_ds or not (left_ds & right_ds):
        return False

    lengths = [int(rec["length"]) for rec in records]
    tops = [int(rec["r0"]) for rec in records]
    bottoms = [int(rec["r1"]) for rec in records]
    if len(set(lengths)) != len(lengths):
        return False
    if len(set(tops)) != len(tops) or len(set(bottoms)) != len(bottoms):
        return False
    if not _has_mixed_length_distance_order(records):
        return False

    teal_count = 0
    pink_count = 0
    for rec in records:
        c = int(rec["col"])
        r0 = int(rec["r0"])
        r1 = int(rec["r1"])
        d = int(rec["distance"])
        length = int(rec["length"])
        if c == marker_c or abs(c - marker_c) != d or length != r1 - r0 + 1:
            return False
        if not (0 < d < length and 0 < r0 <= r1 < grid.shape[0] - 1):
            return False
        if not np.all(grid[r0 : r1 + 1, c] == RED):
            return False
        if rec["side"] == "left":
            teal_count += d
            if not np.all(expected[r1 - d + 1 : r1 + 1, c] == TEAL):
                return False
            if not np.all(expected[r0 : r1 - d + 1, c] == RED):
                return False
        else:
            pink_count += d
            if not np.all(expected[r0 : r0 + d, c] == PINK):
                return False
            if not np.all(expected[r0 + d : r1 + 1, c] == RED):
                return False

    return (
        teal_count > 0
        and pink_count > 0
        and int(np.count_nonzero(expected == TEAL)) == teal_count
        and int(np.count_nonzero(expected == PINK)) == pink_count
    )


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(MIN_HEIGHT, MAX_HEIGHT)
    width = random.randint(MIN_WIDTH, MAX_WIDTH)
    marker_col = random.randint(7, width - 8)

    left_limit = min(marker_col, height - 7)
    right_limit = min(width - marker_col - 1, height - 7)
    common_limit = min(left_limit, right_limit)
    if common_limit < 5:
        return None

    left_count = random.randint(2, 3)
    right_count = random.randint(2, 3)
    shared_distance = random.randint(1, common_limit)
    left_distances = _spaced_distances(left_limit, left_count, shared_distance)
    right_distances = _spaced_distances(right_limit, right_count, shared_distance)
    if left_distances is None or right_distances is None:
        return None

    records: list[dict] = []
    for d in left_distances:
        records.append({"side": "left", "distance": int(d), "col": int(marker_col - d)})
    for d in right_distances:
        records.append({"side": "right", "distance": int(d), "col": int(marker_col + d)})
    random.shuffle(records)

    if _choose_lengths(records, height) is None:
        return None
    if not _choose_row_intervals(records, height):
        return None

    grid = np.full((height, width), BG, dtype=int)
    grid[height - 1, marker_col] = GREY
    for rec in records:
        grid[int(rec["r0"]) : int(rec["r1"]) + 1, int(rec["col"])] = RED

    expected = _expected_from_records(grid, records)
    if not _semantic_checks(grid, records, expected):
        return None

    try:
        marker_c, extracted = _extract_marker_and_bars(grid)
        if marker_c != marker_col or len(extracted) != len(records):
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except Exception:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid suspended-ruler input")
