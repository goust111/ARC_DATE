from __future__ import annotations

from collections import Counter

import numpy as np


BG = 0
TEAL = 8
NON_BG_COLORS = [1, 2, 3, 4, 5, 6, 7, 8, 9]
SEGMENT_COLORS = [1, 2, 3, 4, 5, 6, 7, 9]
MAX_GENERATION_ATTEMPTS = 500


def _most_common_color(values: np.ndarray) -> int:
    return int(Counter(map(int, values.ravel())).most_common(1)[0][0])


def _runs(cols: np.ndarray) -> list[tuple[int, int]]:
    if cols.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(cols[0])
    prev = int(cols[0])
    for col in map(int, cols[1:]):
        if col == prev + 1:
            prev = col
            continue
        runs.append((start, prev))
        start = prev = col
    runs.append((start, prev))
    return runs


def _parse_rows(grid: np.ndarray) -> tuple[int, list[int], list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 4, "expected a 2D ARC grid"

    scale = [int(v) for v in g[0]]
    assert len(set(scale)) == len(scale), "top-row color scale must contain unique colors"

    bg = _most_common_color(g[1:] if g.shape[0] > 1 else g)
    entries: list[dict[str, object]] = []
    segment_width: int | None = None
    seen_left_colors: set[int] = set()

    for r in range(1, g.shape[0]):
        cols = np.flatnonzero(g[r] != bg)
        if cols.size == 0:
            continue

        runs = _runs(cols)
        assert len(runs) == 1, "each marked row must contain one contiguous bracketed object"
        c0, c1 = runs[0]
        assert c1 - c0 + 1 >= 3, "each bracketed object needs two markers and a segment"
        assert int(g[r, c1]) == TEAL, "the right control pixel must be teal"

        left_color = int(g[r, c0])
        assert left_color not in {bg, TEAL}, "the left control pixel must be a non-teal foreground color"
        assert left_color in scale, "every left control color must appear in the scale"
        assert left_color not in seen_left_colors, "left control colors must be unique"
        seen_left_colors.add(left_color)

        segment = g[r, c0 + 1 : c1].copy()
        assert segment.size >= 1, "segment width must be positive"
        if segment_width is None:
            segment_width = int(segment.size)
        assert int(segment.size) == segment_width, "all marked rows must share one segment width"

        address = scale.index(left_color)
        entries.append(
            {
                "grid_row": r,
                "left_col": c0,
                "right_col": c1,
                "left_color": left_color,
                "address": address,
                "segment": segment,
            }
        )

    assert entries, "expected at least one marked row"
    return bg, scale, entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each bracketed row segment to the output row addressed by its left control color in the top scale.

    Input structure:
    - The top row is a full-width color scale whose cells are all different.
    - Each lower source row contains one contiguous non-background run.
    - In each such run, the left endpoint is a non-teal control color, the right endpoint is teal, and the cells strictly between them form the payload segment.

    Rule:
    - Ignore the top-row scale itself and drop both control pixels from every source row.
    - For each source row, look up the left control color in the top scale and read its column index `k`.
    - Copy the enclosed segment unchanged into output row `k`; scale colors with no matching source row stay as background rows.

    Key insight:
    - The segment is not ordered by its own row position; the real address is obtained in two jumps: control color first, then that color's slot in the top scale.
    """
    bg, scale, entries = _parse_rows(grid)
    width = int(entries[0]["segment"].shape[0])
    out = np.full((len(scale), width), bg, dtype=int)

    used_addresses: set[int] = set()
    for entry in entries:
        address = int(entry["address"])
        assert address not in used_addresses, "each scale slot can receive at most one segment"
        used_addresses.add(address)
        out[address] = np.asarray(entry["segment"], dtype=int)

    return out


def _random_segment(width: int, existing: set[bytes]) -> np.ndarray:
    for _ in range(200):
        segment = np.random.choice(SEGMENT_COLORS, size=width, replace=True).astype(int)
        if np.unique(segment).size == 1 and np.random.rand() < 0.6:
            continue
        key = segment.tobytes()
        if key not in existing:
            existing.add(key)
            return segment
    raise RuntimeError("failed to generate a distinct segment")


def _choose_scale() -> list[int]:
    width = int(np.random.randint(6, 10))
    extra = list(np.random.choice(SEGMENT_COLORS, size=width - 1, replace=False))
    scale = [TEAL, *map(int, extra)]
    np.random.shuffle(scale)
    return [int(v) for v in scale]


def _choose_used_indices(scale: list[int]) -> list[int]:
    nonteal = [i for i, color in enumerate(scale) if color != TEAL]
    marker_count = int(np.random.randint(2, min(5, len(nonteal) - 1) + 1))
    for _ in range(200):
        used = sorted(map(int, np.random.choice(nonteal, size=marker_count, replace=False)))
        unused = sorted(i for i in nonteal if i not in used)
        if any(used[0] < idx < used[-1] for idx in unused):
            return used
    raise RuntimeError("failed to keep an internal blank scale slot")


def _build_case() -> tuple[np.ndarray, np.ndarray]:
    scale = _choose_scale()
    width = len(scale)
    used_indices = _choose_used_indices(scale)
    marker_count = len(used_indices)
    segment_width = int(np.random.randint(2, min(5, width - 3) + 1))

    height = 1 + marker_count + int(np.random.randint(2, 6))
    marker_rows = sorted(map(int, np.random.choice(np.arange(1, height), size=marker_count, replace=False)))

    left_positions = list(range(0, width - segment_width - 1))
    for _ in range(200):
        left_cols = [int(v) for v in np.random.choice(left_positions, size=marker_count, replace=True)]
        if len(set(left_cols)) < 2:
            continue
        if not any(left != addr for left, addr in zip(left_cols, used_indices)):
            continue
        break
    else:
        raise RuntimeError("failed to diversify marker columns")

    row_order_addresses = [used_indices[i] for i in np.random.permutation(marker_count)]
    if row_order_addresses == sorted(row_order_addresses):
        row_order_addresses = [used_indices[i] for i in np.random.permutation(marker_count)]
    if row_order_addresses == sorted(row_order_addresses):
        row_order_addresses = list(reversed(sorted(row_order_addresses)))

    row_specs = list(zip(marker_rows, row_order_addresses, left_cols))
    row_specs.sort(key=lambda item: item[0])
    if [addr for _, addr, _ in row_specs] == sorted(used_indices):
        raise RuntimeError("row order accidentally matches address order")

    grid = np.full((height, width), BG, dtype=int)
    grid[0] = np.array(scale, dtype=int)

    out = np.full((width, segment_width), BG, dtype=int)
    seen_segments: set[bytes] = set()
    for row, address, left_col in row_specs:
        left_color = int(scale[address])
        segment = _random_segment(segment_width, seen_segments)
        right_col = left_col + segment_width + 1
        grid[row, left_col] = left_color
        grid[row, left_col + 1 : right_col] = segment
        grid[row, right_col] = TEAL
        out[address] = segment

    return grid, out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, scale, entries = _parse_rows(inp)
    except AssertionError:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    if expected.shape != (inp.shape[1], expected.shape[1]):
        return False

    addresses = [int(entry["address"]) for entry in entries]
    rows = [int(entry["grid_row"]) for entry in entries]
    left_cols = [int(entry["left_col"]) for entry in entries]
    segments = [np.asarray(entry["segment"], dtype=int) for entry in entries]
    nonblank_rows = [i for i in range(expected.shape[0]) if np.any(expected[i] != bg)]

    if len(set(addresses)) != len(addresses):
        return False
    if len({seg.tobytes() for seg in segments}) != len(segments):
        return False
    if len(set(left_cols)) < 2:
        return False
    if not any(left != addr for left, addr in zip(left_cols, addresses)):
        return False
    if [addr for _, addr in sorted(zip(rows, addresses))] == sorted(addresses):
        return False
    if not any(nonblank_rows[0] < idx < nonblank_rows[-1] for idx in range(len(scale)) if idx not in addresses):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            inp, expected = _build_case()
        except RuntimeError:
            continue
        if _validate_semantics(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 6")
