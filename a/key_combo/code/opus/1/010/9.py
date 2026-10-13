import random
from collections import deque

import numpy as np


BG_COLOR = 0
FRAME_COLOR = 5
OBJECT_COLOR = 3
SHADOW_COLOR = 2

GRID_HEIGHT_RANGE = (8, 10)
GRID_WIDTH_RANGE = (16, 24)
CHANNEL_WIDTH_RANGE = (4, 7)
OBJECT_SIZE_RANGE = (3, 8)
MAX_GENERATION_ATTEMPTS = 500

MODE_WEIGHTS = (
    ("left", 0.6),
    ("right", 0.2),
    ("interior", 0.2),
)


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(np.array(cells, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _cell_set(coords: np.ndarray) -> set[tuple[int, int]]:
    return {tuple(map(int, rc)) for rc in coords}


def _is_top_bracket(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    width = c1 - c0 + 1
    if r0 != 0 or r1 - r0 + 1 != 2 or width < 3:
        return False
    expected = {(r0, c) for c in range(c0, c1 + 1)}
    expected |= {(r1, c0), (r1, c1)}
    return _cell_set(coords) == expected


def _is_bottom_bracket(coords: np.ndarray, height: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    width = c1 - c0 + 1
    if r1 != height - 1 or r1 - r0 + 1 != 2 or width < 3:
        return False
    expected = {(r1, c) for c in range(c0, c1 + 1)}
    expected |= {(r0, c0), (r0, c1)}
    return _cell_set(coords) == expected


def _parse_scene(grid: np.ndarray) -> tuple[int, int, tuple[int, int, int, int], np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    height, _width = g.shape
    bg = _get_bg(g)
    assert SHADOW_COLOR not in np.unique(g), "input must not already contain the shadow color"

    bracket_candidates = []
    non_bg_colors = [int(color) for color in np.unique(g) if int(color) != bg]
    for color in non_bg_colors:
        components = _components(g == color)
        tops = [component for component in components if _is_top_bracket(component)]
        bottoms = [
            component for component in components if _is_bottom_bracket(component, height)
        ]
        if len(components) != 2 or len(tops) != 1 or len(bottoms) != 1:
            continue
        top_bbox = _bbox(tops[0])
        bottom_bbox = _bbox(bottoms[0])
        if top_bbox[1] != bottom_bbox[1] or top_bbox[3] != bottom_bbox[3]:
            continue
        bracket_candidates.append((color, top_bbox, bottom_bbox))

    assert len(bracket_candidates) == 1, "expected one aligned bracket pair"
    frame_color, top_bbox, bottom_bbox = bracket_candidates[0]

    crop_r0 = top_bbox[2]
    crop_r1 = bottom_bbox[0]
    crop_c0 = top_bbox[1] + 1
    crop_c1 = top_bbox[3] - 1
    assert crop_r0 <= crop_r1 and crop_c0 <= crop_c1

    fg_colors = [color for color in non_bg_colors if color != frame_color]
    assert len(fg_colors) == 1, "expected one foreground color inside the bracket lane"
    fg_color = fg_colors[0]
    fg_components = _components(g == fg_color)
    assert len(fg_components) == 1, "foreground must be a single 4-connected object"
    fg_coords = fg_components[0]
    assert fg_coords.size > 0
    assert np.all((crop_r0 <= fg_coords[:, 0]) & (fg_coords[:, 0] <= crop_r1))
    assert np.all((crop_c0 <= fg_coords[:, 1]) & (fg_coords[:, 1] <= crop_c1))
    return bg, fg_color, (crop_r0, crop_r1, crop_c0, crop_c1), fg_coords


def _make_channel_tile(
    bg: int,
    fg_color: int,
    crop_box: tuple[int, int, int, int],
    fg_coords: np.ndarray,
) -> np.ndarray:
    crop_r0, crop_r1, crop_c0, crop_c1 = crop_box
    tile = np.full((crop_r1 - crop_r0 + 1, crop_c1 - crop_c0 + 1), bg, dtype=int)
    local_rows = fg_coords[:, 0] - crop_r0
    local_cols = fg_coords[:, 1] - crop_c0
    tile[local_rows, local_cols] = fg_color
    return tile


def _add_wrapped_shadow(
    tile: np.ndarray,
    fg_color: int,
    require_in_bounds: bool = True,
) -> np.ndarray:
    shaded = tile.copy()
    width = tile.shape[1]
    fg_coords = np.argwhere(tile == fg_color)
    assert fg_coords.size > 0
    for r_raw, c_raw in fg_coords:
        r = int(r_raw)
        c = int(c_raw)
        if r == 0:
            if require_in_bounds:
                raise AssertionError("foreground must stay below the crop's top row")
            continue
        sr = r - 1
        sc = (c - 1 + width) % width
        if shaded[sr, sc] != fg_color:
            shaded[sr, sc] = SHADOW_COLOR
    return shaded


def _solve_bbox_crop(grid: np.ndarray) -> np.ndarray:
    bg, fg_color, _crop_box, fg_coords = _parse_scene(grid)
    r0, c0, r1, c1 = _bbox(fg_coords)
    tile = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    tile[fg_coords[:, 0] - r0, fg_coords[:, 1] - c0] = fg_color
    return np.tile(_add_wrapped_shadow(tile, fg_color, require_in_bounds=False), (3, 3))


def _solve_without_wrap(grid: np.ndarray) -> np.ndarray:
    bg, fg_color, crop_box, fg_coords = _parse_scene(grid)
    tile = _make_channel_tile(bg, fg_color, crop_box, fg_coords)
    shaded = tile.copy()
    for r_raw, c_raw in np.argwhere(tile == fg_color):
        r = int(r_raw)
        c = int(c_raw)
        if r == 0 or c == 0:
            continue
        if shaded[r - 1, c - 1] != fg_color:
            shaded[r - 1, c - 1] = SHADOW_COLOR
    return np.tile(shaded, (3, 3))


def _solve_trimmed_channel(grid: np.ndarray) -> np.ndarray:
    bg, fg_color, crop_box, fg_coords = _parse_scene(grid)
    crop_r0, crop_r1, crop_c0, crop_c1 = crop_box
    tile = np.full((crop_r1 - crop_r0 - 1, crop_c1 - crop_c0 + 1), bg, dtype=int)
    keep = fg_coords[:, 0] < crop_r1
    trimmed = fg_coords[keep]
    tile[trimmed[:, 0] - (crop_r0 + 1), trimmed[:, 1] - crop_c0] = fg_color
    return np.tile(_add_wrapped_shadow(tile, fg_color, require_in_bounds=False), (3, 3))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop the bracket-defined lane, add a wrapped offset shadow, and tile the result 3x3.

    Input structure:
    - A black background contains exactly one aligned pair of grey brackets: a top-edge bracket open downward and a bottom-edge bracket open upward.
    - The bracket pair defines a vertical lane whose interior column span is the crop width.
    - Inside that lane sits one irregular 4-connected green object, and no other non-background object appears.

    Rule:
    - Crop the full rectangle given by the brackets' interior columns and the rows between their inner sides; do not crop to the green object's own bbox.
    - Within that crop, keep only green foreground and black background.
    - For each green pixel at (r, c) in a crop of width W, place a red shadow at (r-1, (c-1+W) mod W), unless that target already contains green.
    - Tile the shaded crop three times horizontally and three times vertically.

    Key insight:
    - The brackets, not the foreground bbox, determine the crop width, and that width is the modulus used by the shadow's column wrap.
    """
    g = np.asarray(grid, dtype=int)
    bg, fg_color, crop_box, fg_coords = _parse_scene(g)
    tile = _make_channel_tile(bg, fg_color, crop_box, fg_coords)
    shaded = _add_wrapped_shadow(tile, fg_color)
    return np.tile(shaded, (3, 3))


def _random_mode(rng: random.Random) -> str:
    x = rng.random()
    total = 0.0
    for mode, weight in MODE_WEIGHTS:
        total += weight
        if x <= total:
            return mode
    return MODE_WEIGHTS[-1][0]


def _shape_domain(crop_h: int, crop_w: int, mode: str) -> tuple[range, range]:
    row_range = range(1, crop_h - 1)
    if mode == "left":
        col_range = range(0, crop_w - 1)
    elif mode == "right":
        col_range = range(1, crop_w)
    else:
        col_range = range(1, crop_w - 1)
    return row_range, col_range


def _seed_cell(rng: random.Random, crop_h: int, crop_w: int, mode: str) -> tuple[int, int]:
    top_favor = rng.random() < 0.5
    row = 1 if top_favor else rng.randint(1, crop_h - 2)
    if mode == "left":
        return row, 0
    if mode == "right":
        return row, crop_w - 1
    return row, rng.randint(1, crop_w - 2)


def _valid_object_shape(cells: set[tuple[int, int]], crop_h: int, crop_w: int, mode: str) -> bool:
    rows = sorted({r for r, _c in cells})
    cols = sorted({c for _r, c in cells})
    if len(rows) < 2 or len(cols) < 2:
        return False
    if min(rows) <= 0 or max(rows) >= crop_h - 1:
        return False
    if mode == "left" and max(cols) >= crop_w - 1:
        return False
    if mode == "right" and min(cols) <= 0:
        return False
    if mode == "interior" and (min(cols) <= 0 or max(cols) >= crop_w - 1):
        return False
    bbox_area = (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)
    if bbox_area == len(cells):
        return False
    shadow_targets = {((r - 1), (c - 1 + crop_w) % crop_w) for r, c in cells}
    if shadow_targets & cells:
        return False
    return True


def _grow_object(
    rng: random.Random,
    crop_h: int,
    crop_w: int,
    mode: str,
) -> set[tuple[int, int]] | None:
    row_range, col_range = _shape_domain(crop_h, crop_w, mode)
    allowed = {(r, c) for r in row_range for c in col_range}
    if len(allowed) < OBJECT_SIZE_RANGE[0]:
        return None
    target_size = rng.randint(OBJECT_SIZE_RANGE[0], min(OBJECT_SIZE_RANGE[1], len(allowed)))
    cells = {_seed_cell(rng, crop_h, crop_w, mode)}
    while len(cells) < target_size:
        frontier = set()
        for r, c in cells:
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nxt = (r + dr, c + dc)
                if nxt in allowed and nxt not in cells:
                    frontier.add(nxt)
        if not frontier:
            return None
        cells.add(rng.choice(tuple(frontier)))
    return cells if _valid_object_shape(cells, crop_h, crop_w, mode) else None


def _paint_brackets(grid: np.ndarray, left: int, outer_width: int) -> None:
    right = left + outer_width - 1
    grid[0, left:right + 1] = FRAME_COLOR
    grid[1, left] = FRAME_COLOR
    grid[1, right] = FRAME_COLOR
    grid[-1, left:right + 1] = FRAME_COLOR
    grid[-2, left] = FRAME_COLOR
    grid[-2, right] = FRAME_COLOR


def _build_input(
    grid_h: int,
    grid_w: int,
    channel_w: int,
    left: int,
    cells: set[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((grid_h, grid_w), BG_COLOR, dtype=int)
    _paint_brackets(grid, left, channel_w + 2)
    for r, c in cells:
        grid[r + 1, left + 1 + c] = OBJECT_COLOR
    return grid


def _semantics_hold(grid: np.ndarray) -> bool:
    try:
        correct = main(grid)
        bbox_shortcut = _solve_bbox_crop(grid)
        nowrap_shortcut = _solve_without_wrap(grid)
        trimmed_shortcut = _solve_trimmed_channel(grid)
    except Exception:
        return False

    if correct.ndim != 2 or max(correct.shape) > 30:
        return False
    if not np.any(correct == SHADOW_COLOR):
        return False
    if np.array_equal(correct, bbox_shortcut):
        return False
    if np.array_equal(correct, nowrap_shortcut):
        return False
    if np.array_equal(correct, trimmed_shortcut):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid_h = rng.randint(*GRID_HEIGHT_RANGE)
    crop_h = grid_h - 2
    grid_w = rng.randint(*GRID_WIDTH_RANGE)
    channel_w = rng.randint(*CHANNEL_WIDTH_RANGE)
    if crop_h * 3 > 30 or channel_w * 3 > 30:
        return None

    outer_w = channel_w + 2
    if outer_w + 4 > grid_w:
        return None
    left = rng.randint(2, grid_w - outer_w - 2)
    mode = _random_mode(rng)

    for _ in range(80):
        cells = _grow_object(rng, crop_h, channel_w, mode)
        if cells is None:
            continue
        grid = _build_input(grid_h, grid_w, channel_w, left, cells)
        if _semantics_hold(grid):
            return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 9")
