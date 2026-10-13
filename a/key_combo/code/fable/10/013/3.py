from __future__ import annotations

import random
from collections import deque

import numpy as np


CORNER_SPECS = {
    "tl": {"step": (-1, -1), "opposite": "br"},
    "tr": {"step": (-1, 1), "opposite": "bl"},
    "bl": {"step": (1, -1), "opposite": "tr"},
    "br": {"step": (1, 1), "opposite": "tl"},
}
CORNER_ORDER = ("tl", "tr", "bl", "br")

GEN_RETRIES = 600
GRID_MIN = 18
GRID_MAX = 27
ORIG_SIDE_MIN = 3
ORIG_SIDE_MAX = 7
EXPANSION_MIN = 2
EXPANSION_MAX = 6
FINAL_MARGIN = 3
MAX_FINAL_SIDE = 13
EXTRA_DISTRACTOR_RANGE = (1, 3)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _edge_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _draw_framed_rectangle(
    grid: np.ndarray,
    box: tuple[int, int, int, int],
    border_color: int,
    interior_color: int,
) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = interior_color
    grid[r0, c0 : c1 + 1] = border_color
    grid[r1, c0 : c1 + 1] = border_color
    grid[r0 : r1 + 1, c0] = border_color
    grid[r0 : r1 + 1, c1] = border_color


def _rectangle_info(
    grid: np.ndarray,
    bg: int,
    coords: np.ndarray,
) -> dict[str, int | tuple[int, int, int, int]] | None:
    box = _bbox_coords(coords)
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or coords.shape[0] != h * w:
        return None
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    mask = _edge_mask(h, w)
    border_colors = set(map(int, sub[mask]))
    interior_colors = set(map(int, sub[~mask]))
    if len(border_colors) != 1 or len(interior_colors) != 1:
        return None
    border_color = border_colors.pop()
    interior_color = interior_colors.pop()
    if border_color == interior_color or border_color == bg or interior_color == bg:
        return None
    expected = np.full((h, w), interior_color, dtype=int)
    expected[mask] = border_color
    if not np.array_equal(sub, expected):
        return None
    return {
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
        "border_color": border_color,
        "interior_color": interior_color,
    }


def _first_non_bg_on_ray(
    grid: np.ndarray,
    bg: int,
    start: tuple[int, int],
    step: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    dr, dc = step
    r = start[0] + dr
    c = start[1] + dc
    while 0 <= r < h and 0 <= c < w:
        if int(grid[r, c]) != bg:
            return int(r), int(c)
        r += dr
        c += dc
    return None


def _is_isolated_pixel(grid: np.ndarray, bg: int, cell: tuple[int, int]) -> bool:
    r, c = cell
    if int(grid[r, c]) == bg:
        return False
    h, w = grid.shape
    return all(int(grid[nr, nc]) == bg for nr, nc in _neighbors4(r, c, h, w))


def _corner_points(box: tuple[int, int, int, int]) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = box
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "bl": (r1, c0),
        "br": (r1, c1),
    }


def _analyze_puzzle(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a nonempty 2D grid"
    bg = _get_bg(g)

    components = _components(g != bg)
    rectangle_candidates = []
    for coords in components:
        info = _rectangle_info(g, bg, coords)
        if info is not None:
            rectangle_candidates.append(info)
    assert len(rectangle_candidates) == 1, "expected one framed rectangle"
    rect = rectangle_candidates[0]
    rect_box = (rect["r0"], rect["c0"], rect["r1"], rect["c1"])

    for coords in components:
        if coords.shape[0] == 1:
            continue
        info = _rectangle_info(g, bg, coords)
        assert info is not None, "non-rectangle foreground components are not allowed"

    corners = _corner_points(rect_box)
    visible_hits: list[tuple[str, tuple[int, int]]] = []
    for corner_name in CORNER_ORDER:
        step = CORNER_SPECS[corner_name]["step"]
        hit = _first_non_bg_on_ray(g, bg, corners[corner_name], step)
        if hit is None:
            continue
        assert _is_isolated_pixel(g, bg, hit), "diagonal ray must hit a single pixel"
        visible_hits.append((corner_name, hit))
    assert len(visible_hits) == 1, "expected exactly one visible diagonal marker"

    selected_corner, marker = visible_hits[0]
    opposite_corner = CORNER_SPECS[selected_corner]["opposite"]
    anchor = corners[opposite_corner]
    final_box = (
        min(marker[0], anchor[0]),
        min(marker[1], anchor[1]),
        max(marker[0], anchor[0]),
        max(marker[1], anchor[1]),
    )

    return {
        "bg": bg,
        "rect_box": rect_box,
        "border_color": rect["border_color"],
        "interior_color": rect["interior_color"],
        "selected_corner": selected_corner,
        "marker": marker,
        "step": CORNER_SPECS[selected_corner]["step"],
        "final_box": final_box,
        "corners": corners,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Stretch the framed rectangle diagonally until the unique visible corner marker becomes the new corner.

    Input structure:
    - The background is the most common color.
    - One 4-connected object is a filled axis-aligned rectangle with one border color and one interior color.
    - The remaining foreground cells are isolated marker pixels scattered around the rectangle.

    Rule:
    - From each rectangle corner, look only along that corner's outward diagonal and stop at the first non-background cell on that ray.
    - Exactly one corner sees an isolated pixel through pure background; keep the opposite corner fixed and use that visible pixel as the moved corner of the expanded rectangle.
    - Redraw the larger rectangle with the same border and interior colors, covering the chosen marker while leaving all other distractor pixels unchanged.

    Key insight:
    - The correct marker is determined by diagonal line-of-sight from a corner, so one pixel simultaneously fixes both the new height and the new width.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_puzzle(g)
    out = g.copy()
    _draw_framed_rectangle(
        out,
        info["final_box"],
        int(info["border_color"]),
        int(info["interior_color"]),
    )
    return out


def _cell_in_box(cell: tuple[int, int], box: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _can_place_pixel(grid: np.ndarray, cell: tuple[int, int], bg: int) -> bool:
    r, c = cell
    h, w = grid.shape
    if not (0 <= r < h and 0 <= c < w):
        return False
    if int(grid[r, c]) != bg:
        return False
    return all(int(grid[nr, nc]) == bg for nr, nc in _neighbors4(r, c, h, w))


def _max_steps_from(cell: tuple[int, int], step: tuple[int, int], shape: tuple[int, int]) -> int:
    h, w = shape
    dr, dc = step
    steps = 0
    r, c = cell
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return steps
        steps += 1


def _shrink_from_final(
    corner_name: str,
    final_box: tuple[int, int, int, int],
    d: int,
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[int, int]]:
    fr0, fc0, fr1, fc1 = final_box
    if corner_name == "tl":
        rect_box = (fr0 + d, fc0 + d, fr1, fc1)
        marker = (fr0, fc0)
        moved_corner = (fr0 + d, fc0 + d)
    elif corner_name == "tr":
        rect_box = (fr0 + d, fc0, fr1, fc1 - d)
        marker = (fr0, fc1)
        moved_corner = (fr0 + d, fc1 - d)
    elif corner_name == "bl":
        rect_box = (fr0, fc0 + d, fr1 - d, fc1)
        marker = (fr1, fc0)
        moved_corner = (fr1 - d, fc0 + d)
    else:
        rect_box = (fr0, fc0, fr1 - d, fc1 - d)
        marker = (fr1, fc1)
        moved_corner = (fr1 - d, fc1 - d)
    return rect_box, marker, moved_corner


def _horizontal_misdirect_candidates(
    shape: tuple[int, int],
    corner_name: str,
    rect_box: tuple[int, int, int, int],
    final_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect_box
    fr0, fc0, fr1, fc1 = final_box
    if corner_name in {"tl", "tr"}:
        row = r0
    else:
        row = r1
    if corner_name in {"tl", "bl"}:
        cols = range(0, fc0)
    else:
        cols = range(fc1 + 1, w)
    return [(row, c) for c in cols if 0 <= row < h]


def _vertical_misdirect_candidates(
    shape: tuple[int, int],
    corner_name: str,
    rect_box: tuple[int, int, int, int],
    final_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = shape
    r0, c0, r1, c1 = rect_box
    fr0, fc0, fr1, fc1 = final_box
    if corner_name in {"tl", "bl"}:
        col = c0
    else:
        col = c1
    if corner_name in {"tl", "tr"}:
        rows = range(0, fr0)
    else:
        rows = range(fr1 + 1, h)
    return [(r, col) for r in rows if 0 <= col < w]


def _pick_from_candidates(
    grid: np.ndarray,
    bg: int,
    candidates: list[tuple[int, int]],
) -> tuple[int, int] | None:
    shuffled = candidates[:]
    random.shuffle(shuffled)
    for cell in shuffled:
        if _can_place_pixel(grid, cell, bg):
            return cell
    return None


def _random_outside_box_cells(
    grid: np.ndarray,
    bg: int,
    forbidden_box: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    h, w = grid.shape
    cells = []
    for r in range(h):
        for c in range(w):
            cell = (r, c)
            if _cell_in_box(cell, forbidden_box):
                continue
            if _can_place_pixel(grid, cell, bg):
                cells.append(cell)
    random.shuffle(cells)
    return cells


def _has_blocked_far_distractor(grid: np.ndarray, info: dict[str, object]) -> bool:
    marker = info["marker"]
    step = info["step"]
    bg = int(info["bg"])
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    r = int(marker[0]) + int(step[0])
    c = int(marker[1]) + int(step[1])
    while 0 <= r < h and 0 <= c < w:
        if int(g[r, c]) != bg:
            return True
        r += int(step[0])
        c += int(step[1])
    return False


def _has_edge_aligned_distractor(grid: np.ndarray, info: dict[str, object]) -> bool:
    g = np.asarray(grid, dtype=int)
    bg = int(info["bg"])
    marker = tuple(map(int, info["marker"]))
    r0, c0, r1, c1 = map(int, info["rect_box"])
    for coords in _components(g != bg):
        if coords.shape[0] != 1:
            continue
        cell = tuple(map(int, coords[0]))
        if cell == marker:
            continue
        if cell[0] in (r0, r1) or cell[1] in (c0, c1):
            return True
    return False


def _expected_output(
    grid: np.ndarray,
    final_box: tuple[int, int, int, int],
    border_color: int,
    interior_color: int,
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    _draw_framed_rectangle(out, final_box, border_color, interior_color)
    return out


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    values, counts = np.unique(grid, return_counts=True)
    bg_count = 0
    non_bg_max = 0
    for value, count in zip(values, counts):
        if int(value) == bg:
            bg_count = int(count)
        else:
            non_bg_max = max(non_bg_max, int(count))
    return bg_count > non_bg_max


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    bg = random.randint(0, 9)

    palette = [color for color in range(10) if color != bg]
    border_color, interior_color = random.sample(palette, 2)
    point_palette = random.sample([c for c in palette if c not in {border_color, interior_color}], k=3)

    corner_name = random.choice(CORNER_ORDER)
    rect_h = random.randint(ORIG_SIDE_MIN, ORIG_SIDE_MAX)
    rect_w = random.randint(ORIG_SIDE_MIN, ORIG_SIDE_MAX)
    d_max = min(EXPANSION_MAX, MAX_FINAL_SIDE - rect_h, MAX_FINAL_SIDE - rect_w)
    if d_max < EXPANSION_MIN:
        return None
    d = random.randint(EXPANSION_MIN, d_max)

    final_h = rect_h + d
    final_w = rect_w + d
    if final_h + 2 * FINAL_MARGIN > h or final_w + 2 * FINAL_MARGIN > w:
        return None

    fr0 = random.randint(FINAL_MARGIN, h - FINAL_MARGIN - final_h)
    fc0 = random.randint(FINAL_MARGIN, w - FINAL_MARGIN - final_w)
    final_box = (fr0, fc0, fr0 + final_h - 1, fc0 + final_w - 1)
    rect_box, marker, moved_corner = _shrink_from_final(corner_name, final_box, d)

    grid = np.full((h, w), bg, dtype=int)
    _draw_framed_rectangle(grid, rect_box, border_color, interior_color)

    marker_color = random.choice(point_palette)
    grid[marker] = marker_color

    step = CORNER_SPECS[corner_name]["step"]
    max_behind = _max_steps_from(marker, step, grid.shape)
    if max_behind < 1:
        return None
    k = random.randint(1, max_behind)
    blocked_far = (marker[0] + step[0] * k, marker[1] + step[1] * k)
    if not _can_place_pixel(grid, blocked_far, bg):
        return None
    grid[blocked_far] = marker_color

    horiz = _pick_from_candidates(
        grid,
        bg,
        _horizontal_misdirect_candidates(grid.shape, corner_name, rect_box, final_box),
    )
    if horiz is None:
        return None
    grid[horiz] = random.choice(point_palette)

    vert = _pick_from_candidates(
        grid,
        bg,
        _vertical_misdirect_candidates(grid.shape, corner_name, rect_box, final_box),
    )
    if vert is None:
        return None
    grid[vert] = random.choice(point_palette)

    extra_needed = random.randint(*EXTRA_DISTRACTOR_RANGE)
    extras = 0
    for cell in _random_outside_box_cells(grid, bg, final_box):
        if extras >= extra_needed:
            break
        if not _can_place_pixel(grid, cell, bg):
            continue
        grid[cell] = random.choice(point_palette)
        extras += 1
    if extras < extra_needed:
        return None

    if not _background_is_dominant(grid, bg):
        return None

    try:
        info = _analyze_puzzle(grid)
    except AssertionError:
        return None

    if info["selected_corner"] != corner_name:
        return None
    if tuple(map(int, info["marker"])) != marker:
        return None
    if tuple(map(int, info["final_box"])) != final_box:
        return None

    for coords in _components(grid != bg):
        if coords.shape[0] != 1:
            continue
        cell = tuple(map(int, coords[0]))
        if cell != marker and _cell_in_box(cell, final_box):
            return None

    if not _has_blocked_far_distractor(grid, info):
        return None
    if not _has_edge_aligned_distractor(grid, info):
        return None

    expected = _expected_output(grid, final_box, border_color, interior_color)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if np.array_equal(grid, expected):
        return None
    if not np.array_equal(actual, expected):
        return None

    mr, mc = marker
    sr, sc = moved_corner
    if abs(mr - sr) != abs(mc - sc) or abs(mr - sr) < EXPANSION_MIN:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
