from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _crop_non_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return np.array(grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1], copy=True)


def _contiguous_run(grid: np.ndarray, start_r: int, start_c: int, dr: int, dc: int, color: int) -> int:
    h, w = grid.shape
    r, c = int(start_r), int(start_c)
    length = 0
    while 0 <= r < h and 0 <= c < w and int(grid[r, c]) == int(color):
        length += 1
        r += int(dr)
        c += int(dc)
    return length


def _find_legend_box(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], int]:
    h, w = grid.shape
    corner_specs = [
        ("tl", 0, 0, 0, 1, 1, 0),
        ("tr", 0, w - 1, 0, -1, 1, 0),
        ("bl", h - 1, 0, 0, 1, -1, 0),
        ("br", h - 1, w - 1, 0, -1, -1, 0),
    ]
    candidates: list[tuple[tuple[int, int, int, int], int]] = []

    for _name, ar, ac, hdr, hdc, vdr, vdc in corner_specs:
        color = int(grid[ar, ac])
        if color == bg:
            continue
        width = _contiguous_run(grid, ar, ac, hdr, hdc, color)
        height = _contiguous_run(grid, ar, ac, vdr, vdc, color)
        if width < 3 or height < 3:
            continue

        if ar == 0:
            r0, r1 = 0, height - 1
        else:
            r0, r1 = h - height, h - 1
        if ac == 0:
            c0, c1 = 0, width - 1
        else:
            c0, c1 = w - width, w - 1

        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(box, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if not np.all(box[border] == color):
            continue
        interior = box[1:-1, 1:-1]
        if interior.size == 0 or not np.any(interior != bg):
            continue
        candidates.append(((int(r0), int(c0), int(r1), int(c1)), int(color)))

    assert len(candidates) == 1, f"expected one legend box, got {candidates}"
    return candidates[0]


def _read_top_bars(
    grid: np.ndarray, guide: int, exclude_box: tuple[int, int, int, int] | None
) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    col_positions: list[int] = []
    col_counts: list[int] = []
    for c in range(w):
        if exclude_box is not None:
            r0, c0, r1, c1 = exclude_box
            if r0 == 0 and c0 <= c <= c1:
                continue
        if int(grid[0, c]) != guide:
            continue
        length = 0
        while length < h and int(grid[length, c]) == guide:
            length += 1
        col_positions.append(int(c))
        col_counts.append(int(length))
    return col_positions, col_counts


def _read_left_bars(
    grid: np.ndarray, guide: int, exclude_box: tuple[int, int, int, int] | None
) -> tuple[list[int], list[int]]:
    h, w = grid.shape
    row_positions: list[int] = []
    row_counts: list[int] = []
    for r in range(h):
        if exclude_box is not None:
            r0, c0, r1, c1 = exclude_box
            if c0 == 0 and r0 <= r <= r1:
                continue
        if int(grid[r, 0]) != guide:
            continue
        length = 0
        while length < w and int(grid[r, length]) == guide:
            length += 1
        row_positions.append(int(r))
        row_counts.append(int(length))
    return row_positions, row_counts


def _is_contiguous(indices: list[int]) -> bool:
    return bool(indices) and indices == list(range(indices[0], indices[-1] + 1))


def _profile(arr: np.ndarray, bg: int) -> tuple[tuple[int, ...], tuple[int, ...]]:
    mask = arr != bg
    rows = tuple(int(x) for x in mask.sum(axis=1))
    cols = tuple(int(x) for x in mask.sum(axis=0))
    return rows, cols


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate the legend sprite into the unique canvas slot whose row and column projections match the edge bars.

    Input structure:
    - The background is dominant. Short guide-colored bars hang from the top edge and extend from the left edge.
    - A closed guide-colored box in one corner contains a legend sprite sample.
    - The intended target area is empty; only its row and column projection counts are shown by the bars.

    Rule:
    - Read each top bar length as the occupied-pixel count of the target sprite in that global column.
    - Read each left bar length as the occupied-pixel count of the target sprite in that global row.
    - Crop the legend sprite, try its four 90-degree rotations, and place the unique rotation whose per-row and per-column occupied counts exactly match both projection sequences.

    Key insight:
    - The bars encode the target sprite's two one-dimensional projections at absolute canvas rows and columns, so matching both projections simultaneously determines the correct rotation and placement.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    box, guide = _find_legend_box(g, bg)
    r0, c0, r1, c1 = box
    legend = _crop_non_bg(g[r0 + 1 : r1, c0 + 1 : c1], bg)

    col_positions, col_counts = _read_top_bars(g, guide, box)
    row_positions, row_counts = _read_left_bars(g, guide, box)
    assert _is_contiguous(col_positions)
    assert _is_contiguous(row_positions)

    expected_rows = tuple(row_counts)
    expected_cols = tuple(col_counts)
    matches: list[np.ndarray] = []
    for k in range(4):
        rot = np.rot90(legend, k)
        rows, cols = _profile(rot, bg)
        if rows == expected_rows and cols == expected_cols:
            matches.append(rot)

    assert len(matches) == 1, f"expected exactly one matching rotation, got {len(matches)}"
    sprite = matches[0]
    sh, sw = sprite.shape
    assert sh == len(row_positions) and sw == len(col_positions)

    place_r = row_positions[0]
    place_c = col_positions[0]
    out = np.array(g, copy=True)
    patch = out[place_r : place_r + sh, place_c : place_c + sw]
    mask = sprite != bg
    assert np.all(patch[mask] == bg)
    patch[mask] = sprite[mask]
    out[place_r : place_r + sh, place_c : place_c + sw] = patch
    return out


GRID_MIN = 16
GRID_MAX = 30
SPRITE_SIDE_MIN = 3
SPRITE_SIDE_MAX = 7
TARGET_FILL_RATIO_MIN = 0.35
TARGET_FILL_RATIO_MAX = 0.75
PLACEMENT_GAP = 2
GEN_RETRIES = 400


def _random_sprite_mask() -> np.ndarray:
    h = random.randint(SPRITE_SIDE_MIN, SPRITE_SIDE_MAX)
    w = random.randint(SPRITE_SIDE_MIN, SPRITE_SIDE_MAX)
    max_cells = h * w - 1
    min_cells = max(h + w - 1, int(np.ceil(h * w * TARGET_FILL_RATIO_MIN)))
    max_cells = max(min_cells, int(np.floor(h * w * TARGET_FILL_RATIO_MAX)))
    target_cells = random.randint(min_cells, max_cells)

    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = {start}
    while len(cells) < target_cells:
        seed = random.choice(tuple(frontier))
        candidates = []
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = seed[0] + dr
            cc = seed[1] + dc
            if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                candidates.append((rr, cc))
        if not candidates:
            frontier.remove(seed)
            if not frontier:
                frontier = set(cells)
            continue
        new_cell = random.choice(candidates)
        cells.add(new_cell)
        frontier.add(new_cell)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite(mask: np.ndarray, bg: int, guide: int) -> np.ndarray:
    available = [c for c in range(1, 10) if c not in (bg, guide)]
    palette_size = random.choice([1, 1, 2])
    palette = random.sample(available, palette_size)
    sprite = np.full(mask.shape, bg, dtype=int)
    coords = np.argwhere(mask)
    assignments = [palette[i % len(palette)] for i in range(len(coords))]
    random.shuffle(assignments)
    for (r, c), color in zip(coords, assignments):
        sprite[int(r), int(c)] = int(color)
    return sprite


def _rotation_profiles(sprite: np.ndarray, bg: int) -> list[tuple[tuple[int, ...], tuple[int, ...]]]:
    return [_profile(np.rot90(sprite, k), bg) for k in range(4)]


def _rect_too_close(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_box(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _corner_box(h: int, w: int, box_h: int, box_w: int, corner: str) -> tuple[int, int, int, int]:
    if corner == "tl":
        return 0, 0, box_h - 1, box_w - 1
    if corner == "tr":
        return 0, w - box_w, box_h - 1, w - 1
    if corner == "bl":
        return h - box_h, 0, h - 1, box_w - 1
    if corner == "br":
        return h - box_h, w - box_w, h - 1, w - 1
    raise ValueError(corner)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    guide = random.randint(1, 9)

    mask = _random_sprite_mask()
    sprite = _make_sprite(mask, bg, guide)
    if len(set(_rotation_profiles(sprite, bg))) != 4:
        return None

    rot_k = random.randrange(4)
    target = np.rot90(sprite, rot_k)
    row_counts, col_counts = _profile(target, bg)
    sh, sw = target.shape

    box_h = sprite.shape[0] + 2
    box_w = sprite.shape[1] + 2
    h = random.randint(max(GRID_MIN, box_h + sh + 4), GRID_MAX)
    w = random.randint(max(GRID_MIN, box_w + sw + 4), GRID_MAX)
    corner = random.choice(["tl", "tr", "bl", "br"])
    box = _corner_box(h, w, box_h, box_w, corner)

    min_r = max(col_counts) + PLACEMENT_GAP
    min_c = max(row_counts) + PLACEMENT_GAP
    if corner in ("tl", "bl"):
        min_c = max(min_c, box[3] + PLACEMENT_GAP + 1)
    if corner in ("tl", "tr"):
        min_r = max(min_r, box[2] + PLACEMENT_GAP + 1)

    max_r = h - sh
    max_c = w - sw
    if corner == "bl":
        max_r = min(max_r, box[0] - PLACEMENT_GAP - sh)
    if corner == "tr":
        max_c = min(max_c, box[1] - PLACEMENT_GAP - sw)

    if min_r > max_r or min_c > max_c:
        return None

    candidate_rs = list(range(min_r, max_r + 1))
    candidate_cs = list(range(min_c, max_c + 1))
    random.shuffle(candidate_rs)
    random.shuffle(candidate_cs)

    place_r = None
    place_c = None
    for r in candidate_rs[:20]:
        for c in candidate_cs[:20]:
            target_box = (r, c, r + sh - 1, c + sw - 1)
            if _rect_too_close(target_box, box, gap=1):
                continue
            place_r = r
            place_c = c
            break
        if place_r is not None:
            break
    if place_r is None or place_c is None:
        return None

    grid = np.full((h, w), bg, dtype=int)
    for dc, count in enumerate(col_counts):
        grid[:count, place_c + dc] = guide
    for dr, count in enumerate(row_counts):
        grid[place_r + dr, :count] = guide

    _draw_box(grid, box, guide)
    br0, bc0, br1, bc1 = box
    grid[br0 + 1 : br1, bc0 + 1 : bc1] = sprite

    expected = np.array(grid, copy=True)
    patch = expected[place_r : place_r + sh, place_c : place_c + sw]
    mask_target = target != bg
    patch[mask_target] = target[mask_target]
    expected[place_r : place_r + sh, place_c : place_c + sw] = patch

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 5")
