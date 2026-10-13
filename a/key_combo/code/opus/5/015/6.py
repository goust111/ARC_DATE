import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
COLOR_POOL = [3, 4, 6, 7, 8, 9]
MAX_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _flood_mask(passable: np.ndarray, start: tuple[int, int]) -> np.ndarray:
    h, w = passable.shape
    sr, sc = start
    inside = np.zeros((h, w), dtype=bool)
    stack = [(int(sr), int(sc))]
    while stack:
        r, c = stack.pop()
        if not (0 <= r < h and 0 <= c < w):
            continue
        if inside[r, c] or not passable[r, c]:
            continue
        inside[r, c] = True
        for nr, nc in neighbors4(r, c):
            stack.append((nr, nc))
    return inside


def _split_segments(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == BLUE)]
    assert len(sep_rows) == 2
    r0, r1 = sep_rows
    assert r0 > 0 and r1 > r0 + 1 and r1 < grid.shape[0] - 1
    return grid[:r0, :], grid[r0 + 1 : r1, :], grid[r1 + 1 :, :]


def _analyze_top(top: np.ndarray) -> tuple[int, np.ndarray]:
    red_cells = np.argwhere(top == RED)
    assert red_cells.shape[0] == 1
    start = tuple(int(x) for x in red_cells[0])
    passable = top != GREY
    inside = _flood_mask(passable, start)
    external = passable & (~inside)
    n_external = len(_components_from_mask(external))
    return int(n_external), inside


def _extract_template(mid: np.ndarray) -> np.ndarray:
    fg_mask = mid != BLACK
    coords = np.argwhere(fg_mask)
    assert coords.size > 0
    return crop_bbox(mid, bbox_coords(coords))


def _palette_colors(bottom: np.ndarray) -> tuple[int, int]:
    used = set()
    records = []
    for color in np.unique(bottom):
        color = int(color)
        if color in (BLACK, BLUE):
            continue
        if color in used:
            continue
        mask = bottom == color
        comps = _components_from_mask(mask)
        assert len(comps) == 1
        used.add(color)
        records.append((bbox_coords(comps[0]), color))
    assert len(records) == 2
    records.sort(key=lambda item: item[0][1])
    return int(records[0][1]), int(records[1][1])


def _build_output(template_crop: np.ndarray, repeat: int, fg_color: int, bg_color: int) -> np.ndarray:
    recolored = np.where(template_crop != BLACK, fg_color, bg_color).astype(int)
    h, w = recolored.shape
    out = np.full((repeat * h + (repeat - 1), w), bg_color, dtype=int)
    for i in range(repeat):
        r0 = i * (h + 1)
        out[r0 : r0 + h, :] = recolored
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Repeat a recolored template as many times as the flood-fill leaves external wall-separated fragments.

    Input structure:
    - Two full blue separator rows split the canvas into a top wall panel, a middle template panel, and a bottom palette panel.
    - The top panel has black open cells, grey wall pixels, and one red indicator inside one non-wall region.
    - The middle panel contains one small 4-connected monochrome template on black; the bottom panel contains two solid color swatches placed side by side.

    Rule:
    - Flood-fill 4-connected from the red indicator inside the top panel, blocked by grey walls and the panel boundary, to obtain the internal region.
    - Treat all remaining non-wall cells in the top panel as external and count their 4-connected components; call that count N.
    - Crop the middle template to its foreground bounding box, recolor its foreground with the left swatch color and its in-bbox background with the right swatch color, then stack N copies vertically with one full right-color separator row between adjacent copies.

    Key insight:
    - The wall topology is used only structurally: the number of open fragments unreachable from the indicator determines how many recolored template copies appear in the output.
    """
    g = np.asarray(grid, dtype=int)
    top, mid, bottom = _split_segments(g)
    repeat, _inside = _analyze_top(top)
    assert 1 <= repeat <= 9
    template_crop = _extract_template(mid)
    fg_color, bg_color = _palette_colors(bottom)
    return _build_output(template_crop, repeat, fg_color, bg_color)


def _sample_positions(rng: random.Random, count: int, limit: int) -> list[int] | None:
    candidates = list(range(1, limit - 1))
    for _ in range(80):
        picks = sorted(rng.sample(candidates, count))
        if all(b - a >= 2 for a, b in zip(picks, picks[1:])):
            return picks
    return None


def _make_top_panel(
    rng: random.Random, height: int, width: int, n_external: int
) -> np.ndarray | None:
    total_regions = n_external + 1
    branch_count = total_regions - 2
    top = np.full((height, width), BLACK, dtype=int)
    family = rng.choice(["vertical", "horizontal"])

    if family == "vertical":
        if height < 2 * branch_count + 1 or width < 6:
            return None
        spine_c = rng.randint(2, width - 3)
        left_count = rng.randint(0, branch_count)
        right_count = branch_count - left_count
        left_rows = _sample_positions(rng, left_count, height)
        right_rows = _sample_positions(rng, right_count, height)
        if left_rows is None or right_rows is None:
            return None
        top[:, spine_c] = GREY
        for r in left_rows:
            top[r, : spine_c + 1] = GREY
        for r in right_rows:
            top[r, spine_c:] = GREY
    else:
        if width < 2 * branch_count + 1 or height < 6:
            return None
        spine_r = rng.randint(2, height - 3)
        upper_count = rng.randint(0, branch_count)
        lower_count = branch_count - upper_count
        upper_cols = _sample_positions(rng, upper_count, width)
        lower_cols = _sample_positions(rng, lower_count, width)
        if upper_cols is None or lower_cols is None:
            return None
        top[spine_r, :] = GREY
        for c in upper_cols:
            top[: spine_r + 1, c] = GREY
        for c in lower_cols:
            top[spine_r:, c] = GREY

    open_components = _components_from_mask(top != GREY)
    if len(open_components) != total_regions:
        return None
    comp_index = rng.randrange(len(open_components))
    coords = open_components[comp_index]
    rr, cc = coords[rng.randrange(coords.shape[0])]
    top[rr, cc] = RED
    external, inside = _analyze_top(top)
    if external != n_external:
        return None
    if inside.sum() == 0 or any(comp.shape[0] == 0 for comp in open_components):
        return None
    return top


def _sample_connected_template(
    rng: random.Random, height: int, width: int, area: int
) -> np.ndarray | None:
    cells = {(rng.randrange(height), rng.randrange(width))}
    for _ in range(60):
        if len(cells) == area:
            sprite = np.zeros((height, width), dtype=int)
            for r, c in cells:
                sprite[r, c] = 1
            if len(_components_from_mask(sprite == 1)) == 1:
                return sprite
            return None
        frontier = []
        for r, c in cells:
            for nr, nc in neighbors4(r, c):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(rng.choice(frontier))
    return None


def _make_template_panel(
    rng: random.Random, width: int, color: int
) -> tuple[np.ndarray, np.ndarray] | None:
    box_h = rng.randint(2, 4)
    box_w = rng.randint(2, 4)
    min_area = 3
    max_area = min(8, box_h * box_w)
    if max_area < min_area:
        return None
    want_hole = rng.random() < 0.7 and box_h * box_w > min_area
    if want_hole:
        area = rng.randint(min_area, max_area - 1)
    else:
        area = rng.randint(min_area, max_area)
    sprite = None
    for _ in range(80):
        sprite = _sample_connected_template(rng, box_h, box_w, area)
        if sprite is None:
            continue
        if want_hole and sprite.sum() == box_h * box_w:
            continue
        break
    if sprite is None:
        return None

    mid_h = rng.randint(max(5, box_h + 2), max(6, box_h + 3))
    panel = np.full((mid_h, width), BLACK, dtype=int)
    top_margin = rng.randint(1, mid_h - box_h - 1)
    left_margin = rng.randint(1, width - box_w - 1)
    panel[top_margin : top_margin + box_h, left_margin : left_margin + box_w][sprite == 1] = color
    crop = sprite.astype(int) * color
    return panel, crop


def _make_palette_panel(
    rng: random.Random, width: int, left_color: int, right_color: int
) -> np.ndarray | None:
    rect_h = rng.randint(2, 3)
    panel_h = rect_h + rng.randint(1, 2)
    left_w = rng.randint(2, 4)
    right_w = rng.randint(2, 4)
    gap = 1
    total_w = left_w + gap + right_w
    if total_w + 2 > width:
        return None
    panel = np.full((panel_h, width), BLACK, dtype=int)
    r0 = rng.randint(0, panel_h - rect_h)
    c0 = rng.randint(1, width - total_w - 1)
    panel[r0 : r0 + rect_h, c0 : c0 + left_w] = left_color
    panel[r0 : r0 + rect_h, c0 + left_w + gap : c0 + total_w] = right_color
    return panel


def _validate_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    top, mid, bottom = _split_segments(grid)
    if np.count_nonzero(top == RED) != 1:
        return False
    if len(_components_from_mask(top == GREY)) != 1:
        return False
    repeat, inside = _analyze_top(top)
    if repeat < 2 or repeat > 4:
        return False
    if inside.sum() == 0:
        return False
    mid_colors = {int(v) for v in np.unique(mid) if int(v) != BLACK}
    if len(mid_colors) != 1:
        return False
    crop = _extract_template(mid)
    fg_pixels = int(np.count_nonzero(crop))
    if not (3 <= fg_pixels <= 8):
        return False
    if len(_components_from_mask(crop != BLACK)) != 1:
        return False
    if len(_components_from_mask((bottom != BLACK) & (bottom != BLUE))) != 2:
        return False
    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return False
    template = _extract_template(mid)
    left_color, right_color = _palette_colors(bottom)
    expected = _build_output(template, repeat, left_color, right_color)
    if not np.array_equal(out, expected):
        return False
    if len({left_color, right_color} | mid_colors) != 3:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_external = rng.choice([2, 3, 4])
    width = rng.randint(10, 14)
    top_h = rng.randint(max(7, 2 * (n_external - 1) + 2), 9)
    top = _make_top_panel(rng, top_h, width, n_external)
    if top is None:
        return None

    template_color, left_color, right_color = rng.sample(COLOR_POOL, 3)
    mid_data = _make_template_panel(rng, width, template_color)
    if mid_data is None:
        return None
    mid, template_crop = mid_data
    if np.unique(template_crop).size < 2:
        return None

    bottom = _make_palette_panel(rng, width, left_color, right_color)
    if bottom is None:
        return None

    grid_h = top.shape[0] + 1 + mid.shape[0] + 1 + bottom.shape[0]
    grid = np.full((grid_h, width), BLACK, dtype=int)
    r = 0
    grid[r : r + top.shape[0], :] = top
    r += top.shape[0]
    grid[r, :] = BLUE
    r += 1
    grid[r : r + mid.shape[0], :] = mid
    r += mid.shape[0]
    grid[r, :] = BLUE
    r += 1
    grid[r : r + bottom.shape[0], :] = bottom

    if not _validate_grid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
