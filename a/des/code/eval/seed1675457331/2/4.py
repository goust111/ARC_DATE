from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                pts.append((int(rr), int(cc)))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(pts)
    return comps


def _mode(values: np.ndarray) -> int:
    uniq, counts = np.unique(values, return_counts=True)
    return int(uniq[counts.argmax()])


def _wall_records(grid: np.ndarray, wall_color: int) -> list[dict]:
    wall_mask = grid[:-1] == int(wall_color)
    comps = _components(wall_mask, connectivity=8)
    records: list[dict] = []

    for pts in comps:
        rows = [r for r, _c in pts]
        r0, r1 = min(rows), max(rows)
        row_to_col: dict[int, int] = {}
        for r, c in pts:
            assert r not in row_to_col, "wall rows must be single-pixel thick"
            row_to_col[r] = int(c)
        assert len(row_to_col) == r1 - r0 + 1, "wall must span consecutive rows"

        cols = [row_to_col[r] for r in range(r0, r1 + 1)]
        assert len(cols) >= 2, "funnel walls must have depth >= 2"
        diffs = [cols[i + 1] - cols[i] for i in range(len(cols) - 1)]
        assert all(abs(d) == 1 for d in diffs), "wall must be a unit diagonal"

        if all(d == 1 for d in diffs):
            kind = "left"
        else:
            assert all(d == -1 for d in diffs), "wall direction must be monotone"
            kind = "right"

        records.append(
            {
                "kind": kind,
                "r0": int(r0),
                "r1": int(r1),
                "top_col": int(cols[0]),
                "bottom_col": int(cols[-1]),
                "row_to_col": row_to_col,
            }
        )

    records.sort(key=lambda rec: rec["top_col"])
    return records


def _pair_walls(records: list[dict]) -> list[tuple[dict, dict]]:
    assert len(records) % 2 == 0 and records, "expected an even number of wall segments"
    pairs: list[tuple[dict, dict]] = []
    for i in range(0, len(records), 2):
        left = records[i]
        right = records[i + 1]
        assert left["kind"] == "left", "paired funnel must start with a left wall"
        assert right["kind"] == "right", "paired funnel must end with a right wall"
        assert left["r0"] == right["r0"] and left["r1"] == right["r1"], "walls must share row span"
        for r in range(left["r0"], left["r1"] + 1):
            assert left["row_to_col"][r] < right["row_to_col"][r], "walls must face inward"
        pairs.append((left, right))
    return pairs


def _interior_rows(left: dict, right: dict) -> list[tuple[int, int, int]]:
    rows: list[tuple[int, int, int]] = []
    for r in range(left["r0"], left["r1"] + 1):
        lc = left["row_to_col"][r]
        rc = right["row_to_col"][r]
        assert rc - lc >= 2, "funnel must have nonnegative interior width"
        rows.append((int(r), int(lc), int(rc)))
    return rows


def _settled_cells(rows: list[tuple[int, int, int]], count: int) -> tuple[list[tuple[int, int]], int]:
    remaining = int(count)
    filled: list[tuple[int, int]] = []
    height = 0
    for r, lc, rc in reversed(rows):
        if remaining <= 0:
            break
        capacity = int(rc - lc - 1)
        take = min(capacity, remaining)
        if take > 0:
            filled.extend((int(r), int(c)) for c in range(lc + 1, lc + 1 + take))
            remaining -= take
            height += 1
    assert remaining == 0, "pixel count exceeds funnel capacity"
    return filled, height


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each funnel's colored pixels down into a bottom-up pile, then recolor
    the lower wall segments according to the pile height.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    wall_color = _mode(g[-1])

    out = g.copy()
    for left, right in _pair_walls(_wall_records(g, wall_color)):
        rows = _interior_rows(left, right)

        colors: set[int] = set()
        count = 0
        for r, lc, rc in rows:
            segment = g[r, lc + 1 : rc]
            for value in segment:
                value = int(value)
                if value != BG:
                    colors.add(value)
                    count += 1

        assert len(colors) == 1, "each funnel must contain exactly one non-background color"
        fill_color = int(next(iter(colors)))
        assert fill_color != wall_color
        assert count > 0

        for r, lc, rc in rows:
            out[r, lc + 1 : rc] = BG

        settled, height = _settled_cells(rows, count)
        for r, c in settled:
            out[r, c] = fill_color

        for r, lc, rc in reversed(rows[-height:]):
            out[r, lc] = fill_color
            out[r, rc] = fill_color

    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 23)
GRID_W_RANGE = (16, 31)
FUNNEL_COUNT_RANGE = (1, 4)
DEPTH_RANGE = (2, 6)
SIDE_MARGIN = 1
MIN_FUNNEL_GAP = 2
COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_GENERATION_ATTEMPTS = 400


def _composition(total: int, parts: int) -> list[int]:
    if total <= 0:
        return [0] * parts
    return list(np.random.multinomial(int(total), [1.0 / parts] * parts))


def _pack_top_lefts(width: int, spans: list[int]) -> list[int] | None:
    slack = int(width) - 2 * SIDE_MARGIN - sum(spans) - MIN_FUNNEL_GAP * (len(spans) - 1)
    if slack < 0:
        return None
    gaps = _composition(slack, len(spans) + 1)
    lefts: list[int] = []
    pos = SIDE_MARGIN + gaps[0]
    for i, span in enumerate(spans):
        lefts.append(int(pos))
        pos += int(span)
        if i + 1 < len(spans):
            pos += MIN_FUNNEL_GAP + gaps[i + 1]
    return lefts


def _rows_from_spec(top_left: int, depth: int, bottom_row: int) -> list[tuple[int, int, int]]:
    top_row = int(bottom_row) - int(depth) + 1
    rows: list[tuple[int, int, int]] = []
    for level in range(depth):
        r = top_row + level
        left = int(top_left) + level
        right = int(top_left) + 2 * int(depth) - level
        rows.append((int(r), int(left), int(right)))
    return rows


def _all_interior_cells(rows: list[tuple[int, int, int]]) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r, lc, rc in rows:
        cells.extend((int(r), int(c)) for c in range(lc + 1, rc))
    return cells


def _sample_scatter_cells(rows: list[tuple[int, int, int]], count: int) -> list[tuple[int, int]]:
    all_cells = _all_interior_cells(rows)
    settled, _height = _settled_cells(rows, count)
    settled_set = set(settled)

    if count == len(all_cells):
        return list(all_cells)

    for _ in range(30):
        choice = random.sample(all_cells, int(count))
        if set(choice) != settled_set:
            return choice

    outside = [cell for cell in all_cells if cell not in settled_set]
    assert outside, "non-full piles must have at least one non-settled interior cell"
    prefix = random.sample(settled, int(count) - 1) if count > 1 else []
    return prefix + [random.choice(outside)]


def _sample_specs(height: int, width: int) -> list[dict] | None:
    max_depth = min(DEPTH_RANGE[1], height - 2)
    if max_depth < DEPTH_RANGE[0]:
        return None

    max_count = min(FUNNEL_COUNT_RANGE[1], len(COLORS))
    count = random.randint(FUNNEL_COUNT_RANGE[0], max_count)
    depths = [random.randint(DEPTH_RANGE[0], max_depth) for _ in range(count)]
    spans = [2 * depth + 1 for depth in depths]
    top_lefts = _pack_top_lefts(width, spans)
    if top_lefts is None:
        return None

    colors = random.sample(COLORS, count)
    specs: list[dict] = []
    for top_left, depth, color in zip(top_lefts, depths, colors, strict=False):
        pile_height = random.randint(1, depth)
        low = (pile_height - 1) ** 2 + 1
        high = pile_height ** 2
        if pile_height == depth and high > low:
            high -= 1
        count_pixels = random.randint(low, high)
        specs.append(
            {
                "top_left": int(top_left),
                "depth": int(depth),
                "color": int(color),
                "count": int(count_pixels),
            }
        )
    return specs


def _render_specs(height: int, width: int, specs: list[dict]) -> np.ndarray:
    grid = np.full((int(height), int(width)), BG, dtype=int)
    grid[-1, :] = WALL
    bottom_row = int(height) - 2

    for spec in specs:
        rows = _rows_from_spec(spec["top_left"], spec["depth"], bottom_row)
        for r, lc, rc in rows:
            grid[r, lc] = WALL
            grid[r, rc] = WALL

        scatter = _sample_scatter_cells(rows, spec["count"])
        for r, c in scatter:
            grid[r, c] = int(spec["color"])

    return grid


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    specs = _sample_specs(height, width)
    if specs is None:
        return None

    grid = _render_specs(height, width, specs)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid funnel puzzle")
