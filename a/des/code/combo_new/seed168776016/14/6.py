from __future__ import annotations

import random

import numpy as np


BG = 0
DIVIDER = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)

# Distinct connected polyomino-like masks. The generator samples four different
# shape classes and then randomizes each with rotation / reflection.
SHAPE_BANK = (
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0, 0], [1, 1, 1, 1]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
)

MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False
    start = tuple(int(x) for x in coords[0])
    seen = {start}
    stack = [start]
    h, w = mask.shape
    while stack:
        r, c = stack.pop()
        for dr, dc in DIR4:
            rr = r + dr
            cc = c + dc
            if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and (rr, cc) not in seen:
                seen.add((rr, cc))
                stack.append((rr, cc))
    return len(seen) == int(mask.sum())


def _find_dividers(grid: np.ndarray, bg: int) -> tuple[int, int, int]:
    h, w = grid.shape
    row_candidates = [
        (r, int(grid[r, 0]))
        for r in range(h)
        if int(grid[r, 0]) != bg and bool(np.all(grid[r, :] == grid[r, 0]))
    ]
    col_candidates = [
        (c, int(grid[0, c]))
        for c in range(w)
        if int(grid[0, c]) != bg and bool(np.all(grid[:, c] == grid[0, c]))
    ]
    matches = [(r, c, color_r) for r, color_r in row_candidates for c, color_c in col_candidates if color_r == color_c]
    assert len(matches) == 1, "expected exactly one grey cross divider"
    return matches[0]


def _quadrant_slices(mid_r: int, mid_c: int) -> tuple[tuple[slice, slice], ...]:
    return (
        (slice(0, mid_r), slice(0, mid_c)),
        (slice(0, mid_r), slice(mid_c + 1, None)),
        (slice(mid_r + 1, None), slice(0, mid_c)),
        (slice(mid_r + 1, None), slice(mid_c + 1, None)),
    )


def _parse_quadrant(q: np.ndarray, bg: int) -> dict[str, object]:
    colors = [int(color) for color in np.unique(q) if int(color) != bg]
    assert len(colors) == 2, "each quadrant must contain one shape color and one label color"

    label_color = None
    label_pos = None
    shape_color = None
    shape_coords = None

    for color in colors:
        coords = np.argwhere(q == color)
        if len(coords) == 1:
            assert label_color is None
            label_color = color
            label_pos = tuple(int(x) for x in coords[0])
        else:
            assert shape_color is None
            assert 3 <= len(coords) <= 8
            mask = q == color
            assert _is_connected(mask), "shape must be 4-connected"
            shape_color = color
            shape_coords = [tuple(int(x) for x in coord) for coord in coords.tolist()]

    assert label_color is not None and label_pos is not None
    assert shape_color is not None and shape_coords is not None
    return {
        "label_color": label_color,
        "label_pos": label_pos,
        "shape_color": shape_color,
        "shape_coords": shape_coords,
        "shape": q.shape,
    }


def _rotate_180(coords: list[tuple[int, int]], shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    return [(h - 1 - r, w - 1 - c) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each quadrant's shape from its diagonal label and rotate it 180 degrees inside that quadrant.

    Input structure:
    - The background is black.
    - One full grey row and one full grey column split the grid into four quadrants.
    - Each quadrant contains exactly one monochrome 4-connected shape of area 3-8 and one single-pixel label of a different color.
    - Shapes and labels stay inside their quadrants and do not touch the grey divider.

    Rule:
    - Number the quadrants as top-left A, top-right B, bottom-left C, bottom-right D.
    - Remove every label pixel from the output.
    - In each quadrant, rotate the shape 180 degrees around the center of that quadrant's usable subgrid.
    - Repaint the rotated shape with the label color from the diagonal quadrant: A gets D's label color, B gets C's, C gets B's, and D gets A's.
    - Keep the grey divider unchanged.

    Key insight:
    - The color mapping and the geometry are independent: diagonal labels determine the new colors, while a quadrant-local 180-degree rotation determines the new shape placement and orientation.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    mid_r, mid_c, divider_color = _find_dividers(g, bg)
    quadrants = _quadrant_slices(mid_r, mid_c)
    infos = [_parse_quadrant(g[rs, cs], bg) for rs, cs in quadrants]

    diagonal = {0: 3, 1: 2, 2: 1, 3: 0}
    out = np.full_like(g, bg)
    out[mid_r, :] = divider_color
    out[:, mid_c] = divider_color

    for idx, (rs, cs) in enumerate(quadrants):
        info = infos[idx]
        rotated = _rotate_180(info["shape_coords"], info["shape"])
        new_color = int(infos[diagonal[idx]]["label_color"])
        q = out[rs, cs]
        for r, c in rotated:
            q[r, c] = new_color

    return out


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    return np.array(out, dtype=bool, copy=True)


def _coord_set(mask: np.ndarray, origin_r: int, origin_c: int) -> set[tuple[int, int]]:
    points = set()
    for r_raw, c_raw in np.argwhere(mask):
        points.add((origin_r + int(r_raw), origin_c + int(c_raw)))
    return points


def _dividing_margins(index: int) -> tuple[int, int, int, int]:
    if index == 0:
        return 0, 1, 0, 1
    if index == 1:
        return 0, 1, 1, 0
    if index == 2:
        return 1, 0, 0, 1
    return 1, 0, 1, 0


def _too_close(cell: tuple[int, int], coords: set[tuple[int, int]]) -> bool:
    r, c = cell
    for rr, cc in coords:
        if max(abs(r - rr), abs(c - cc)) <= 1:
            return True
    return False


def _place_quadrant_object(
    qh: int,
    qw: int,
    index: int,
    mask: np.ndarray,
    shape_color: int,
    label_color: int,
) -> dict[str, object] | None:
    top_margin, bottom_margin, left_margin, right_margin = _dividing_margins(index)
    h, w = mask.shape
    r_min = top_margin
    r_max = qh - bottom_margin - h
    c_min = left_margin
    c_max = qw - right_margin - w
    if r_max < r_min or c_max < c_min:
        return None

    for _ in range(60):
        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        shape_coords = _coord_set(mask, r0, c0)
        rotated_coords = {(qh - 1 - r, qw - 1 - c) for r, c in shape_coords}
        if shape_coords == rotated_coords:
            continue

        label_candidates = []
        for r in range(top_margin, qh - bottom_margin):
            for c in range(left_margin, qw - right_margin):
                cell = (r, c)
                if cell in shape_coords or cell in rotated_coords:
                    continue
                if _too_close(cell, shape_coords) or _too_close(cell, rotated_coords):
                    continue
                label_candidates.append(cell)
        if not label_candidates:
            continue

        label_pos = random.choice(label_candidates)
        return {
            "shape_color": shape_color,
            "label_color": label_color,
            "shape_coords": sorted(shape_coords),
            "rotated_coords": sorted(rotated_coords),
            "label_pos": label_pos,
            "shape": (qh, qw),
        }

    return None


def _paint_quadrant(grid: np.ndarray, rs: slice, cs: slice, info: dict[str, object]) -> None:
    q = grid[rs, cs]
    for r, c in info["shape_coords"]:
        q[r, c] = int(info["shape_color"])
    lr, lc = info["label_pos"]
    q[lr, lc] = int(info["label_color"])


def _build_expected(
    canvas_shape: tuple[int, int],
    mid_r: int,
    mid_c: int,
    states: list[dict[str, object]],
) -> np.ndarray:
    out = np.full(canvas_shape, BG, dtype=int)
    out[mid_r, :] = DIVIDER
    out[:, mid_c] = DIVIDER
    diagonal = {0: 3, 1: 2, 2: 1, 3: 0}
    slices = _quadrant_slices(mid_r, mid_c)

    for idx, (rs, cs) in enumerate(slices):
        q = out[rs, cs]
        color = int(states[diagonal[idx]]["label_color"])
        for r, c in states[idx]["rotated_coords"]:
            q[r, c] = color
    return out


def _generate_attempt() -> np.ndarray | None:
    top_h = random.randint(6, 12)
    bottom_h = random.randint(6, 12)
    left_w = random.randint(6, 12)
    right_w = random.randint(6, 12)

    mid_r = top_h
    mid_c = left_w
    grid = np.full((top_h + 1 + bottom_h, left_w + 1 + right_w), BG, dtype=int)
    grid[mid_r, :] = DIVIDER
    grid[:, mid_c] = DIVIDER

    colors = list(PALETTE)
    random.shuffle(colors)
    shape_colors = colors[:4]
    label_colors = colors[4:]
    shape_ids = random.sample(range(len(SHAPE_BANK)), 4)

    sizes = ((top_h, left_w), (top_h, right_w), (bottom_h, left_w), (bottom_h, right_w))
    states: list[dict[str, object]] = []

    for idx, ((qh, qw), shape_id) in enumerate(zip(sizes, shape_ids, strict=True)):
        info = _place_quadrant_object(
            qh=qh,
            qw=qw,
            index=idx,
            mask=_transform_mask(SHAPE_BANK[shape_id]),
            shape_color=shape_colors[idx],
            label_color=label_colors[idx],
        )
        if info is None:
            return None
        states.append(info)

    for (rs, cs), info in zip(_quadrant_slices(mid_r, mid_c), states, strict=True):
        _paint_quadrant(grid, rs, cs, info)

    expected = _build_expected(grid.shape, mid_r, mid_c, states)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
