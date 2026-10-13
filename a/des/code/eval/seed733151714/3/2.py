from __future__ import annotations

from collections import deque
import math

import numpy as np


BG = 0
GRID = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
MAX_ATTEMPTS = 500
_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        if seen[r, c]:
            continue
        q = deque([(int(r), int(c))])
        seen[r, c] = True
        coords = []
        while q:
            cr, cc = q.popleft()
            coords.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _segment_info(coords: np.ndarray) -> dict[str, object]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    area = int(coords.shape[0])
    if h == 1 and w >= 2 and area == w:
        return {
            "orientation": "h",
            "length": w,
            "anchor": (int(r0), int(c0)),
        }
    if w == 1 and h >= 2 and area == h:
        return {
            "orientation": "v",
            "length": h,
            "anchor": (int(r0), int(c0)),
        }
    raise AssertionError("each colored component must be a straight segment")


def _nearest_index(values: np.ndarray, target: int) -> int:
    return int(np.argmin(np.abs(values - int(target))))


def _euclid(p0: tuple[int, int], p1: tuple[float, float]) -> float:
    return math.hypot(float(p0[0]) - p1[0], float(p0[1]) - p1[1])


def _infer_grid(grid: np.ndarray) -> tuple[int, int, np.ndarray, np.ndarray, tuple[float, float]]:
    bg = _most_common_color(grid)
    colors, counts = np.unique(grid[grid != bg], return_counts=True)
    assert len(colors) >= 2, "expected a grid color plus at least one instruction color"
    grid_color = int(colors[np.argmax(counts)])

    mask = grid == grid_color
    row_counts = mask.sum(axis=1)
    col_counts = mask.sum(axis=0)
    line_rows = np.flatnonzero(row_counts == row_counts.max())
    line_cols = np.flatnonzero(col_counts == col_counts.max())
    assert len(line_rows) >= 2 and len(line_cols) >= 2

    row_steps = np.diff(line_rows)
    col_steps = np.diff(line_cols)
    assert np.all(row_steps == row_steps[0]) and np.all(col_steps == col_steps[0])
    assert int(row_steps[0]) == int(col_steps[0])

    r0, r1 = int(line_rows[0]), int(line_rows[-1])
    c0, c1 = int(line_cols[0]), int(line_cols[-1])
    expected = np.zeros_like(mask)
    expected[line_rows[:, None], np.arange(c0, c1 + 1)] = True
    expected[np.arange(r0, r1 + 1)[:, None], line_cols] = True
    assert np.array_equal(mask, expected), "grey pixels must form one clean orthogonal grid"

    center = ((r0 + r1) / 2.0, (c0 + c1) / 2.0)
    return bg, grid_color, line_rows, line_cols, center


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode paired horizontal/vertical instruction segments into woven checkerboard
    fills on a grey orthogonal grid, then erase the instruction segments.
    """
    g = np.asarray(grid, dtype=int)
    bg, grid_color, line_rows, line_cols, center = _infer_grid(g)

    out = np.where(g == grid_color, grid_color, bg)
    instructions = []

    for color in np.unique(g):
        color = int(color)
        if color in (bg, grid_color):
            continue
        comps = _components(g == color)
        assert len(comps) == 2, "each color must appear as exactly two segments"

        horiz = None
        vert = None
        for comp in comps:
            info = _segment_info(comp)
            if info["orientation"] == "h":
                assert horiz is None
                horiz = info
            else:
                assert vert is None
                vert = info

        assert horiz is not None and vert is not None
        width = int(horiz["length"])
        height = int(vert["length"])
        start_col = _nearest_index(line_cols, int(horiz["anchor"][1]))
        start_row = _nearest_index(line_rows, int(vert["anchor"][0]))
        assert start_col + width <= len(line_cols)
        assert start_row + height <= len(line_rows)

        # The line endpoints are the visible anchors that encode the grid position.
        dist = 0.5 * (
            _euclid(horiz["anchor"], center)
            + _euclid(vert["anchor"], center)
        )
        instructions.append(
            {
                "distance": dist,
                "color": color,
                "start_row": start_row,
                "start_col": start_col,
                "height": height,
                "width": width,
            }
        )

    distances = [float(item["distance"]) for item in instructions]
    assert len(set(round(d, 6) for d in distances)) == len(distances), "instruction order must be unambiguous"
    instructions.sort(key=lambda item: float(item["distance"]))

    for item in instructions:
        color = int(item["color"])
        start_row = int(item["start_row"])
        start_col = int(item["start_col"])
        height = int(item["height"])
        width = int(item["width"])
        for dr in range(height):
            rr = int(line_rows[start_row + dr])
            for dc in range(width):
                cc = int(line_cols[start_col + dc])
                out[rr, cc] = color if (dr + dc) % 2 == 0 else grid_color

    return out


def _sample_region(rng: np.random.Generator, n_rows: int, n_cols: int) -> dict[str, int]:
    height = int(rng.integers(2, min(5, n_rows) + 1))
    width = int(rng.integers(2, min(5, n_cols) + 1))
    start_row = int(rng.integers(0, n_rows - height + 1))
    start_col = int(rng.integers(0, n_cols - width + 1))
    return {
        "start_row": start_row,
        "start_col": start_col,
        "height": height,
        "width": width,
    }


def _sample_overlapping_region(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
    base: dict[str, int],
) -> dict[str, int] | None:
    for _ in range(100):
        height = int(rng.integers(2, min(5, n_rows) + 1))
        width = int(rng.integers(2, min(5, n_cols) + 1))
        overlap_row = int(rng.integers(base["start_row"], base["start_row"] + base["height"]))
        overlap_col = int(rng.integers(base["start_col"], base["start_col"] + base["width"]))

        row_lo = max(0, overlap_row - height + 1)
        row_hi = min(overlap_row, n_rows - height)
        col_lo = max(0, overlap_col - width + 1)
        col_hi = min(overlap_col, n_cols - width)
        if row_lo > row_hi or col_lo > col_hi:
            continue

        start_row = int(rng.integers(row_lo, row_hi + 1))
        start_col = int(rng.integers(col_lo, col_hi + 1))
        region = {
            "start_row": start_row,
            "start_col": start_col,
            "height": height,
            "width": width,
        }
        if region != base:
            return region
    return None


def _region_overlap(a: dict[str, int], b: dict[str, int]) -> bool:
    row_ok = max(a["start_row"], b["start_row"]) < min(a["start_row"] + a["height"], b["start_row"] + b["height"])
    col_ok = max(a["start_col"], b["start_col"]) < min(a["start_col"] + a["width"], b["start_col"] + b["width"])
    return bool(row_ok and col_ok)


def _build_input_from_spec(specs: list[dict[str, int]], n_rows: int, n_cols: int, step: int, margins: tuple[int, int, int, int]) -> np.ndarray | None:
    top, bottom, left, right = margins
    grid_h = step * (n_rows - 1) + 1
    grid_w = step * (n_cols - 1) + 1
    h = top + grid_h + bottom
    w = left + grid_w + right
    grid = np.full((h, w), BG, dtype=int)

    line_rows = top + np.arange(n_rows) * step
    line_cols = left + np.arange(n_cols) * step
    grid_top, grid_bottom = int(line_rows[0]), int(line_rows[-1])
    grid_left, grid_right = int(line_cols[0]), int(line_cols[-1])

    grid[line_rows[:, None], np.arange(grid_left, grid_right + 1)] = GRID
    grid[np.arange(grid_top, grid_bottom + 1)[:, None], line_cols] = GRID

    row_candidates = [grid_top - off for off in range(2, top)] + [grid_bottom + off for off in range(2, bottom)]
    col_candidates = [grid_left - off for off in range(2, left)] + [grid_right + off for off in range(2, right)]
    if len(row_candidates) < len(specs) or len(col_candidates) < len(specs):
        return None

    h_rows = list(map(int, _RNG.choice(row_candidates, size=len(specs), replace=False)))
    v_cols = list(map(int, _RNG.choice(col_candidates, size=len(specs), replace=False)))
    _RNG.shuffle(h_rows)
    _RNG.shuffle(v_cols)

    center = ((grid_top + grid_bottom) / 2.0, (grid_left + grid_right) / 2.0)
    distances = []

    for spec, h_row, v_col in zip(specs, h_rows, v_cols):
        color = int(spec["color"])
        start_row = int(spec["start_row"])
        start_col = int(spec["start_col"])
        height = int(spec["height"])
        width = int(spec["width"])

        h_anchor = (int(h_row), int(line_cols[start_col]))
        v_anchor = (int(line_rows[start_row]), int(v_col))
        dist = 0.5 * (_euclid(h_anchor, center) + _euclid(v_anchor, center))
        distances.append(dist)

        grid[h_row, h_anchor[1] : h_anchor[1] + width] = color
        grid[v_anchor[0] : v_anchor[0] + height, v_col] = color

    if len(set(round(d, 6) for d in distances)) != len(distances):
        return None

    if any((grid == color).sum() != spec["width"] + spec["height"] for color, spec in ((int(s["color"]), s) for s in specs)):
        return None
    return grid


def _gen_attempt() -> np.ndarray | None:
    rng = _RNG
    n_rows = int(rng.integers(4, 8))
    n_cols = int(rng.integers(4, 8))
    step = int(rng.integers(3, 5))

    base_tb = int(rng.integers(5, 9))
    base_lr = int(rng.integers(5, 9))
    margins = (
        base_tb + int(rng.integers(0, 2)),
        base_tb + int(rng.integers(0, 2)),
        base_lr + int(rng.integers(0, 2)),
        base_lr + int(rng.integers(0, 2)),
    )

    n_colors = int(rng.integers(2, 5))
    colors = list(map(int, rng.choice(COLOR_POOL, size=n_colors, replace=False)))
    force_overlap = n_colors >= 2 and bool(rng.integers(0, 2))

    specs: list[dict[str, int]] = []
    first = _sample_region(rng, n_rows, n_cols)
    first["color"] = colors[0]
    specs.append(first)

    if force_overlap:
        second = _sample_overlapping_region(rng, n_rows, n_cols, first)
        if second is None:
            return None
        second["color"] = colors[1]
        specs.append(second)
        start_idx = 2
    else:
        start_idx = 1

    for idx in range(start_idx, n_colors):
        spec = _sample_region(rng, n_rows, n_cols)
        spec["color"] = colors[idx]
        specs.append(spec)

    if force_overlap and not any(_region_overlap(specs[0], specs[i]) for i in range(1, len(specs))):
        return None

    grid = _build_input_from_spec(specs, n_rows, n_cols, step, margins)
    if grid is None:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    _, _, line_rows, line_cols, _ = _infer_grid(grid)
    for spec in specs:
        sr = int(spec["start_row"])
        sc = int(spec["start_col"])
        height = int(spec["height"])
        width = int(spec["width"])
        color = int(spec["color"])
        colored_hits = 0
        for dr in range(height):
            rr = int(line_rows[sr + dr])
            for dc in range(width):
                cc = int(line_cols[sc + dc])
                if (dr + dc) % 2 == 0 and out[rr, cc] == color:
                    colored_hits += 1
        if colored_hits == 0:
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
