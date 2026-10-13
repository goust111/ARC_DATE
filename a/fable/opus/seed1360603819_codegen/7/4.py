from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
RNG = np.random.default_rng()

ROW_COUNT_RANGE = (4, 10)
PAYLOAD_LEN_RANGE = (2, 8)
LEFT_MARGIN_RANGE = (1, 5)
RIGHT_MARGIN_RANGE = (1, 5)
MAX_GENERATION_ATTEMPTS = 300
COUNT_SAMPLE_ATTEMPTS = 200
PAYLOAD_SAMPLE_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    if np.any(grid == BG):
        return BG
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _extract_rows(grid: np.ndarray) -> tuple[int, int, int, list[dict[str, np.ndarray | int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, _ = g.shape
    bg = _background_color(g)

    rows: list[dict[str, np.ndarray | int]] = []
    left_col: int | None = None
    right_col: int | None = None
    payload_len: int | None = None

    for r in range(h):
        row = g[r]
        cols = np.flatnonzero(row != bg)
        assert cols.size >= 3, "each row must contain two markers and at least one payload cell"

        left = int(cols[0])
        right = int(cols[-1])
        expected = np.arange(left, right + 1)
        assert np.array_equal(cols, expected), "row content must be one contiguous segment"

        marker = int(row[left])
        assert marker != bg
        assert int(row[right]) == marker, "row endpoints must share the marker color"

        payload = row[left + 1 : right].copy()
        assert payload.size >= 1
        assert bool(np.all(payload != bg)), "payload may not contain background"
        assert bool(np.all(payload != marker)), "payload may not reuse its row marker color"
        assert int(np.count_nonzero(row == marker)) == 2, "marker color must appear exactly twice in its row"

        if left_col is None:
            left_col = left
            right_col = right
            payload_len = int(payload.size)
        else:
            assert left == left_col and right == right_col, "marker columns must align across rows"
            assert int(payload.size) == payload_len, "all payloads must have the same length"

        rows.append({"marker": marker, "payload": payload})

    assert left_col is not None and right_col is not None
    return bg, left_col, right_col, rows


def _payload_order(markers: list[int]) -> list[int]:
    counts = Counter(markers)
    return sorted(range(len(markers)), key=lambda i: (-counts[markers[i]], i))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder row payloads by descending global frequency of each row's marker color.

    Each row contains one contiguous non-background segment:
    marker, payload..., marker. The marker columns stay fixed, as do the marker
    colors; only the payload strings are reassigned after sorting donor rows by
    marker-color frequency, using stable top-to-bottom order for ties.
    """
    g = np.asarray(grid, dtype=int)
    _, left_col, right_col, rows = _extract_rows(g)

    markers = [int(row["marker"]) for row in rows]
    donor_order = _payload_order(markers)

    out = g.copy()
    for dst_r, src_r in enumerate(donor_order):
        out[dst_r, left_col + 1 : right_col] = rows[src_r]["payload"]
    return out


def _sample_counts(row_count: int, tie_mode: bool) -> list[int] | None:
    min_colors = 3 if tie_mode else 2
    max_colors = min(4, row_count)
    if min_colors > max_colors:
        return None

    for _ in range(COUNT_SAMPLE_ATTEMPTS):
        color_count = int(RNG.integers(min_colors, max_colors + 1))
        cut_points = np.sort(
            RNG.choice(np.arange(1, row_count), size=color_count - 1, replace=False)
        )
        counts = np.diff(np.concatenate(([0], cut_points, [row_count]))).astype(int).tolist()
        distinct = len(set(counts))
        if tie_mode and 1 < distinct < color_count:
            return counts
        if not tie_mode and distinct == color_count:
            return counts
    return None


def _sample_payload(marker: int, payload_len: int, used: set[bytes]) -> np.ndarray | None:
    choices = [color for color in COLOR_POOL if color != marker]
    for _ in range(PAYLOAD_SAMPLE_ATTEMPTS):
        payload = RNG.choice(choices, size=payload_len, replace=True).astype(int)
        if payload_len >= 3 and len(set(map(int, payload))) == 1 and RNG.random() < 0.6:
            continue
        key = payload.tobytes()
        if key in used:
            continue
        used.add(key)
        return payload
    return None


def _build_grid(
    markers: list[int], payloads: list[np.ndarray], left_margin: int, right_margin: int
) -> np.ndarray:
    row_count = len(markers)
    payload_len = int(payloads[0].size)
    width = left_margin + 1 + payload_len + 1 + right_margin
    grid = np.full((row_count, width), BG, dtype=int)
    for r, (marker, payload) in enumerate(zip(markers, payloads)):
        grid[r, left_margin] = int(marker)
        grid[r, left_margin + 1 : left_margin + 1 + payload_len] = payload
        grid[r, left_margin + 1 + payload_len] = int(marker)
    return grid


def _gen_attempt() -> np.ndarray | None:
    row_count = int(RNG.integers(*ROW_COUNT_RANGE))
    payload_len = int(RNG.integers(*PAYLOAD_LEN_RANGE))
    left_margin = int(RNG.integers(*LEFT_MARGIN_RANGE))
    right_margin = int(RNG.integers(*RIGHT_MARGIN_RANGE))
    tie_mode = bool(RNG.integers(0, 2))

    counts = _sample_counts(row_count, tie_mode)
    if counts is None:
        return None

    marker_colors = RNG.choice(COLOR_POOL, size=len(counts), replace=False).astype(int).tolist()
    markers: list[int] = []
    for color, count in zip(marker_colors, counts):
        markers.extend([int(color)] * int(count))
    RNG.shuffle(markers)

    donor_order = _payload_order(markers)
    if donor_order == list(range(row_count)):
        return None

    used_payloads: set[bytes] = set()
    payloads: list[np.ndarray] = []
    for marker in markers:
        payload = _sample_payload(marker, payload_len, used_payloads)
        if payload is None:
            return None
        payloads.append(payload)

    grid = _build_grid(markers, payloads, left_margin, right_margin)
    out = main(grid)

    if np.array_equal(grid, out):
        return None

    _, parsed_left, parsed_right, rows = _extract_rows(grid)
    assert parsed_left == left_margin
    assert parsed_right == left_margin + payload_len + 1
    assert len(rows) == row_count
    assert len({row["payload"].tobytes() for row in rows}) == row_count

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
