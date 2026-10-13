from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


COLOR_VALUES = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]

    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _legend_candidate(row: np.ndarray, bg: int) -> tuple[int, int, list[int]] | None:
    cols = np.flatnonzero(row != bg)
    if cols.size < 2:
        return None
    c0 = int(cols[0])
    c1 = int(cols[-1])
    if c1 - c0 + 1 != cols.size:
        return None
    colors = [int(v) for v in row[c0 : c1 + 1]]
    if any(color == bg for color in colors):
        return None
    if len(set(colors)) != len(colors):
        return None
    return c0, c1, colors


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, list[int]]:
    candidates = []
    for row_idx in (0, grid.shape[0] - 1):
        candidate = _legend_candidate(grid[row_idx], bg)
        if candidate is not None:
            c0, c1, colors = candidate
            candidates.append((row_idx, c0, c1, colors))
    assert len(candidates) == 1, "expected exactly one edge legend row"
    return candidates[0]


def _is_rectangular_frame(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
        return False

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros_like(sub, dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.all(sub[expected] == color) and np.all(sub[~expected] != color))


def _find_frames(
    grid: np.ndarray, bg: int, legend_len: int
) -> tuple[int, list[dict[str, object]]]:
    candidates = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        frames = []
        for coords in _components(grid == color, connectivity=4):
            if _is_rectangular_frame(grid, color, coords):
                r0, c0, r1, c1 = _bbox(coords)
                frames.append(
                    {
                        "bbox": (r0, c0, r1, c1),
                        "coords": coords,
                    }
                )
        if len(frames) == legend_len:
            candidates.append((color, frames))
    assert len(candidates) == 1, "expected a unique frame color"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    legend_row, legend_c0, legend_c1, legend_colors = _find_legend(g, bg)
    frame_color, frames = _find_frames(g, bg, len(legend_colors))

    counts = []
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        object_mask = (interior != bg) & (interior != frame_color)
        area = int(np.count_nonzero(object_mask))
        assert area > 0, "each frame must contain an interior object"
        counts.append(area)
        frame["area"] = area

    assert len(set(counts)) == len(counts), "object areas must be unique"

    out = g.copy()
    out[legend_row, legend_c0 : legend_c1 + 1] = bg

    ranked = sorted(frames, key=lambda item: int(item["area"]))
    for rank, frame in enumerate(ranked):
        color = legend_colors[rank]
        coords = frame["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _random_connected_shape(height: int, width: int, area: int) -> np.ndarray:
    assert 1 <= area <= height * width
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    patch = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        patch[r, c] = True
    return patch


def _separated(rect: tuple[int, int, int, int], others: list[tuple[int, int, int, int]]) -> bool:
    r0, c0, r1, c1 = rect
    for or0, oc0, or1, oc1 in others:
        if not (r1 < or0 - 1 or or1 < r0 - 1 or c1 < oc0 - 1 or oc1 < c0 - 1):
            return False
    return True


def _place_frames(
    height: int, width: int, n_frames: int, legend_top: bool
) -> list[tuple[int, int, int, int]] | None:
    rects: list[tuple[int, int, int, int]] = []
    top_margin = 2 if legend_top else 1
    bottom_limit = height - 3 if not legend_top else height - 2

    for _ in range(120):
        rects.clear()
        dims = []
        for _idx in range(n_frames):
            for _tries in range(40):
                fh = random.randint(4, 8)
                fw = random.randint(4, 9)
                if (fh - 2) * (fw - 2) >= n_frames:
                    dims.append((fh, fw))
                    break
            else:
                dims = []
                break
        if not dims:
            continue

        random.shuffle(dims)
        ok = True
        for fh, fw in dims:
            row_min = top_margin
            row_max = bottom_limit - fh + 1
            col_min = 1
            col_max = width - fw - 1
            if row_min > row_max or col_min > col_max:
                ok = False
                break

            candidates = []
            for r0 in range(row_min, row_max + 1):
                for c0 in range(col_min, col_max + 1):
                    rect = (r0, c0, r0 + fh - 1, c0 + fw - 1)
                    if _separated(rect, rects):
                        candidates.append(rect)
            if not candidates:
                ok = False
                break
            rects.append(random.choice(candidates))

        if ok:
            return rects[:]
    return None


def _assign_unique_areas(capacities: list[int]) -> list[int] | None:
    order = sorted(range(len(capacities)), key=lambda i: capacities[i])
    chosen = [0] * len(capacities)
    used: set[int] = set()

    def rec(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        candidates = [area for area in range(1, capacities[idx] + 1) if area not in used]
        random.shuffle(candidates)
        for area in candidates:
            used.add(area)
            chosen[idx] = area
            if rec(pos + 1):
                return True
            used.remove(area)
        return False

    if rec(0):
        return chosen
    return None


def _expected_output(
    grid: np.ndarray,
    legend_row: int,
    legend_c0: int,
    legend_c1: int,
    legend_colors: list[int],
    rects: list[tuple[int, int, int, int]],
    areas: list[int],
) -> np.ndarray:
    out = grid.copy()
    bg = _most_common_color(grid)
    out[legend_row, legend_c0 : legend_c1 + 1] = bg
    ranked = sorted(range(len(rects)), key=lambda i: areas[i])
    for rank, idx in enumerate(ranked):
        r0, c0, r1, c1 = rects[idx]
        color = legend_colors[rank]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 28)
    width = random.randint(18, 30)
    n_frames = random.randint(2, 4)
    legend_top = random.choice([True, False])

    bg, frame_color = random.sample(COLOR_VALUES, 2)
    remaining = [color for color in COLOR_VALUES if color not in (bg, frame_color)]
    legend_colors = random.sample(remaining, n_frames)
    object_palette = [color for color in remaining if color not in legend_colors]

    rects = _place_frames(height, width, n_frames, legend_top)
    if rects is None:
        return None

    capacities = [(r1 - r0 - 1) * (c1 - c0 - 1) for r0, c0, r1, c1 in rects]
    areas = _assign_unique_areas(capacities)
    if areas is None:
        return None

    grid = np.full((height, width), bg, dtype=int)
    legend_row = 0 if legend_top else height - 1
    legend_c0 = random.randint(1, width - n_frames - 1)
    legend_c1 = legend_c0 + n_frames - 1
    grid[legend_row, legend_c0 : legend_c1 + 1] = np.array(legend_colors, dtype=int)

    for idx, (r0, c0, r1, c1) in enumerate(rects):
        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color

        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        area = areas[idx]

        box_options = [
            (bh, bw)
            for bh in range(1, inner_h + 1)
            for bw in range(1, inner_w + 1)
            if bh * bw >= area
        ]
        box_h, box_w = random.choice(box_options)
        shape = _random_connected_shape(box_h, box_w, area)
        obj_color = random.choice(object_palette)
        pr = r0 + 1 + random.randint(0, inner_h - box_h)
        pc = c0 + 1 + random.randint(0, inner_w - box_w)
        patch = np.full((box_h, box_w), bg, dtype=int)
        patch[shape] = obj_color
        grid[pr : pr + box_h, pc : pc + box_w] = np.where(
            shape, patch, grid[pr : pr + box_h, pc : pc + box_w]
        )

    if _most_common_color(grid) != bg:
        return None

    expected = _expected_output(
        grid, legend_row, legend_c0, legend_c1, legend_colors, rects, areas
    )
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
