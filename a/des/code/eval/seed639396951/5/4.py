from __future__ import annotations

import random
from typing import Iterable

import numpy as np


ROTATION_TO_CCW_K = {0: 0, 1: 3, 2: 2, 3: 1}
LEGEND_ORDER = [((0, 0), 0), ((0, 1), 1), ((1, 1), 2), ((1, 0), 3)]
CORNERS = ("tl", "tr", "bl", "br")
SIDES = ("top", "right", "bottom", "left")


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _corner_origin(corner: str, h: int, w: int) -> tuple[int, int]:
    r0 = 0 if corner in ("tl", "tr") else h - 2
    c0 = 0 if corner in ("tl", "bl") else w - 2
    return r0, c0


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    assert h >= 2 and w >= 2
    candidates: list[tuple[np.ndarray, int, int]] = []

    for corner in CORNERS:
        r0, c0 = _corner_origin(corner, h, w)
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        colors = [int(x) for x in patch.ravel()]
        if bg in colors or len(set(colors)) != 4:
            continue
        candidates.append((patch.copy(), r0, c0))

    assert len(candidates) == 1, "grid must contain exactly one 2x2 legend in a corner"
    patch, r0, c0 = candidates[0]

    mapping: dict[int, int] = {}
    for (rr, cc), rotation_id in LEGEND_ORDER:
        mapping[int(patch[rr, cc])] = ROTATION_TO_CCW_K[rotation_id]

    cells = {(r0 + dr, c0 + dc) for dr in range(2) for dc in range(2)}
    return mapping, cells


def _border_mask(size: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _frame_candidate(
    grid: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    bg: int,
    legend_colors: set[int],
) -> dict | None:
    sub = grid[r0 : r0 + size, c0 : c0 + size]
    border = _border_mask(size)
    border_vals = sub[border]
    colors, counts = np.unique(border_vals, return_counts=True)
    if len(colors) != 2:
        return None

    major_idx = int(np.argmax(counts))
    minor_idx = 1 - major_idx
    border_color = int(colors[major_idx])
    marker_color = int(colors[minor_idx])
    if counts[major_idx] != border_vals.size - 1 or counts[minor_idx] != 1:
        return None
    if border_color == bg or marker_color == bg or marker_color not in legend_colors:
        return None

    marker_rel = np.argwhere(border & (sub == marker_color))
    if marker_rel.shape[0] != 1:
        return None
    mr, mc = map(int, marker_rel[0])
    mid = size // 2
    if (mr, mc) not in {(0, mid), (mid, size - 1), (size - 1, mid), (mid, 0)}:
        return None

    interior = sub[1:-1, 1:-1]
    if not np.any(interior != bg):
        return None

    return {
        "r0": r0,
        "c0": c0,
        "size": size,
        "border_color": border_color,
        "marker_color": marker_color,
    }


def _boxes_overlap(a: dict, b: dict) -> bool:
    a_r1 = a["r0"] + a["size"] - 1
    a_c1 = a["c0"] + a["size"] - 1
    b_r1 = b["r0"] + b["size"] - 1
    b_c1 = b["c0"] + b["size"] - 1
    return not (a_r1 < b["r0"] or b_r1 < a["r0"] or a_c1 < b["c0"] or b_c1 < a["c0"])


def _find_frames(grid: np.ndarray, bg: int, rotation_map: dict[int, int]) -> list[dict]:
    h, w = grid.shape
    legend_colors = set(rotation_map)
    candidates: list[dict] = []

    for size in range(5, min(h, w) + 1, 2):
        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                candidate = _frame_candidate(grid, r0, c0, size, bg, legend_colors)
                if candidate is not None:
                    candidates.append(candidate)

    assert candidates, "no framed square regions found"

    groups: dict[int, list[dict]] = {}
    for candidate in candidates:
        groups.setdefault(int(candidate["border_color"]), []).append(candidate)

    border_color = max(
        groups,
        key=lambda color: (len(groups[color]), sum(item["size"] * item["size"] for item in groups[color])),
    )

    selected: list[dict] = []
    for candidate in sorted(groups[border_color], key=lambda item: (-item["size"], item["r0"], item["c0"])):
        if any(_boxes_overlap(candidate, other) for other in selected):
            continue
        selected.append(candidate)

    assert selected, "failed to isolate square regions"
    return selected


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each framed pattern according to the marker color's position in the 2x2 legend.

    Input structure:
    - Background is the dominant color.
    - Several odd-sized square regions have a uniform border color except for one marker pixel
      at the midpoint of one side.
    - A 2x2 corner legend maps four colors to rotations by position:
      top-left=0, top-right=clockwise 90, bottom-right=180, bottom-left=counterclockwise 90.

    Rule:
    - For each square, find its marker color in the legend to determine the rotation.
    - Rotate only the interior pattern; keep the border color unchanged.
    - Restore the marker pixel to the border color in the output.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    rotation_map, _legend_cells = _find_legend(g, bg)
    frames = _find_frames(g, bg, rotation_map)

    out = g.copy()
    for frame in frames:
        r0 = int(frame["r0"])
        c0 = int(frame["c0"])
        size = int(frame["size"])
        border_color = int(frame["border_color"])
        marker_color = int(frame["marker_color"])
        k = rotation_map[marker_color]

        sub = g[r0 : r0 + size, c0 : c0 + size]
        interior = sub[1:-1, 1:-1]
        out[r0 : r0 + size, c0 : c0 + size] = sub

        border = _border_mask(size)
        out[r0 : r0 + size, c0 : c0 + size][border] = border_color
        out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = np.rot90(interior, k)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (20, 30)
GRID_W_RANGE = (20, 30)
REGION_COUNT_CHOICES = (2, 3, 4, 5)
SQUARE_SIZE_CHOICES = (5, 7)
GEN_ATTEMPTS = 500
PLACEMENT_PADDING = 1


def _neighbors4(r: int, c: int, h: int, w: int) -> Iterable[tuple[int, int]]:
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _grow_mask(size: int, target: int) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    start = (random.randrange(size), random.randrange(size))
    mask[start] = True
    frontier = set(_neighbors4(start[0], start[1], size, size))

    while int(mask.sum()) < target:
        if not frontier:
            return mask
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        mask[cell] = True
        for nbr in _neighbors4(cell[0], cell[1], size, size):
            if not mask[nbr]:
                frontier.add(nbr)
    return mask


def _is_nonsymmetric(pattern: np.ndarray) -> bool:
    return all(not np.array_equal(pattern, np.rot90(pattern, k)) for k in (1, 2, 3))


def _make_pattern(size: int, bg: int, palette: list[int]) -> np.ndarray | None:
    max_cells = size * size - 1
    min_cells = 3 if size <= 3 else 4
    if max_cells < min_cells:
        return None

    for _ in range(200):
        target = random.randint(min_cells, max_cells)
        mask = _grow_mask(size, target)
        if int(mask.sum()) != target:
            continue

        pattern = np.full((size, size), bg, dtype=int)
        color_count = random.randint(1, min(3, len(palette)))
        colors = random.sample(palette, color_count)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        order_mode = random.randrange(4)
        if order_mode == 0:
            coords.sort(key=lambda rc: (rc[0], rc[1]))
        elif order_mode == 1:
            coords.sort(key=lambda rc: (rc[1], rc[0]))
        elif order_mode == 2:
            coords.sort(key=lambda rc: (rc[0] + rc[1], rc[0], rc[1]))
        else:
            random.shuffle(coords)

        split_points = sorted(random.sample(range(1, len(coords)), color_count - 1)) if color_count > 1 else []
        groups = np.split(np.array(coords, dtype=int), split_points)
        for color, group in zip(colors, groups):
            for r, c in group:
                pattern[int(r), int(c)] = color

        if _is_nonsymmetric(pattern):
            return pattern
    return None


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> None:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    occupied[rr0:rr1, cc0:cc1] = True


def _place_square(occupied: np.ndarray, size: int) -> tuple[int, int] | None:
    h, w = occupied.shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(1, h - size):
        for c0 in range(1, w - size):
            rr0 = r0 - PLACEMENT_PADDING
            cc0 = c0 - PLACEMENT_PADDING
            rr1 = r0 + size + PLACEMENT_PADDING
            cc1 = c0 + size + PLACEMENT_PADDING
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            candidates.append((r0, c0))
    if not candidates:
        return None
    return random.choice(candidates)


def _marker_rel(side: str, size: int) -> tuple[int, int]:
    mid = size // 2
    if side == "top":
        return 0, mid
    if side == "right":
        return mid, size - 1
    if side == "bottom":
        return size - 1, mid
    return mid, 0


def _rotation_ids(n: int) -> list[int]:
    base = list(range(4))
    random.shuffle(base)
    out = base[: min(4, n)]
    while len(out) < n:
        out.append(random.randrange(4))
    random.shuffle(out)
    return out


def _paint_input(
    h: int,
    w: int,
    bg: int,
    border_color: int,
    legend_corner: str,
    legend_colors: list[int],
    regions: list[dict],
) -> np.ndarray:
    grid = np.full((h, w), bg, dtype=int)
    lr0, lc0 = _corner_origin(legend_corner, h, w)
    for (dr, dc), rotation_id in LEGEND_ORDER:
        grid[lr0 + dr, lc0 + dc] = legend_colors[rotation_id]

    for region in regions:
        r0 = int(region["r0"])
        c0 = int(region["c0"])
        size = int(region["size"])
        pattern = region["pattern"]
        grid[r0 : r0 + size, c0 : c0 + size] = bg
        grid[r0, c0 : c0 + size] = border_color
        grid[r0 + size - 1, c0 : c0 + size] = border_color
        grid[r0 : r0 + size, c0] = border_color
        grid[r0 : r0 + size, c0 + size - 1] = border_color
        grid[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = pattern

        mr, mc = _marker_rel(region["marker_side"], size)
        grid[r0 + mr, c0 + mc] = legend_colors[int(region["rotation_id"])]

    return grid


def _expected_output(input_grid: np.ndarray, border_color: int, regions: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for region in regions:
        r0 = int(region["r0"])
        c0 = int(region["c0"])
        size = int(region["size"])
        pattern = region["pattern"]
        k = ROTATION_TO_CCW_K[int(region["rotation_id"])]
        out[r0, c0 : c0 + size] = border_color
        out[r0 + size - 1, c0 : c0 + size] = border_color
        out[r0 : r0 + size, c0] = border_color
        out[r0 : r0 + size, c0 + size - 1] = border_color
        out[r0 + 1 : r0 + size - 1, c0 + 1 : c0 + size - 1] = np.rot90(pattern, k)
    return out


def _bg_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    best = int(np.argmax(counts))
    return int(values[best]) == bg and int(counts[best]) > int(np.max(counts[values != bg], initial=0))


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    bg = random.randrange(10)

    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    border_color = colors[0]
    legend_colors = colors[1:5]
    interior_palette = colors[5:]
    if len(legend_colors) != 4 or not interior_palette:
        return None

    grid_count = random.choice(REGION_COUNT_CHOICES)
    sizes = sorted([random.choice(SQUARE_SIZE_CHOICES) for _ in range(grid_count)], reverse=True)
    rotations = _rotation_ids(grid_count)
    legend_corner = random.choice(CORNERS)

    occupied = np.zeros((h, w), dtype=bool)
    lr0, lc0 = _corner_origin(legend_corner, h, w)
    _reserve(occupied, lr0, lc0, 2, 2, pad=1)

    regions: list[dict] = []
    for size, rotation_id in zip(sizes, rotations):
        pattern = _make_pattern(size - 2, bg, interior_palette)
        if pattern is None:
            return None
        placement = _place_square(occupied, size)
        if placement is None:
            return None
        r0, c0 = placement
        _reserve(occupied, r0, c0, size, size, pad=PLACEMENT_PADDING)
        regions.append(
            {
                "r0": r0,
                "c0": c0,
                "size": size,
                "pattern": pattern,
                "rotation_id": rotation_id,
                "marker_side": random.choice(SIDES),
            }
        )

    input_grid = _paint_input(h, w, bg, border_color, legend_corner, legend_colors, regions)
    if not _bg_is_dominant(input_grid, bg):
        return None

    expected = _expected_output(input_grid, border_color, regions)
    try:
        actual = main(input_grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_ATTEMPTS} attempts")
