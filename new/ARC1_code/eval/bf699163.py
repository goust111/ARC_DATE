import numpy as np

from arc_utils import bbox, bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste
import random


def _is_hollow_3x3_square(g: np.ndarray, comp: dict, bg: int) -> bool:
    box = comp["bbox"]
    if comp["area"] != 8 or bbox_h(box) != 3 or bbox_w(box) != 3:
        return False

    r0, c0, r1, c1 = box
    color = int(comp["color"])
    expected = np.full((3, 3), color, dtype=int)
    expected[1, 1] = bg
    return np.array_equal(g[r0 : r1 + 1, c0 : c1 + 1], expected)


def main(grid: np.ndarray) -> np.ndarray:
    """Return the hollow 3x3 sprite that is enclosed by the sparse guide mask.

    Input structure
    - The background is the dominant color.
    - Several monochrome 3x3 hollow-square sprites sit on the background.
    - The remaining foreground pixels are a single-color sparse guide mask whose bbox frames one sprite.

    Rule
    - Find every hollow 3x3 square sprite.
    - Build the guide mask from all non-background pixels that are not part of those sprites.
    - Return the tight bbox crop of the unique sprite fully contained in the guide mask bbox.

    Key insight
    - The guide pixels do not draw the output; their bounding rectangle selects which existing hollow-square sprite to crop.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    sprite_mask = np.zeros(g.shape, dtype=bool)
    sprites = []
    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        if _is_hollow_3x3_square(g, comp, bg):
            sprites.append(comp["bbox"])
            for r, c in comp["coords"]:
                sprite_mask[int(r), int(c)] = True

    assert len(sprites) >= 2, "expected multiple hollow-square candidate sprites"

    guide_mask = (g != bg) & ~sprite_mask
    assert np.any(guide_mask), "expected sparse guide pixels"
    assert len(set(map(int, np.unique(g[guide_mask])))) == 1, "guide pixels should share one color"

    guide_box = bbox(guide_mask)
    assert guide_box is not None
    gr0, gc0, gr1, gc1 = guide_box

    selected = []
    for box in sprites:
        r0, c0, r1, c1 = box
        if gr0 <= r0 and gc0 <= c0 and r1 <= gr1 and c1 <= gc1:
            selected.append(box)

    assert len(selected) == 1, "expected exactly one sprite inside the guide bbox"
    return crop_bbox(g, selected[0])

# --------------- generator (o5x) ---------------
# CFG: colors and sprite shape.
_GEN_BG = 5
_GEN_GUIDE_COLOR = 7
_GEN_SPRITE_COLORS = [1, 2, 3, 4, 6, 8]
_GEN_SPRITE_SIZE = 3
_GEN_SPRITE_GAP = 1

# CFG: inclusive bounds and structural counts.
_GEN_GRID_H = (13, 17)
_GEN_GRID_W = (14, 18)
_GEN_GUIDE_H = (5, 10)
_GEN_GUIDE_W = (6, 8)
_GEN_SPRITE_COUNT = (4, 5)
_GEN_GUIDE_FRAGMENTS = (3, 6)

# CFG: guide density and retry limits.
_GEN_GUIDE_AREA_RATIO = (0.35, 0.60)
_GEN_GUIDE_FILL = (0.40, 0.65)
_GEN_GUIDE_RETRIES = 50
_GEN_ATTEMPTS = 300
_GenBox = tuple[int, int, int, int]
_GEN_GEOMETRY_CACHE = {}


def _gen_hollow_square(color: int) -> np.ndarray:
    sprite = np.full((_GEN_SPRITE_SIZE, _GEN_SPRITE_SIZE), color, dtype=int)
    sprite[_GEN_SPRITE_SIZE // 2, _GEN_SPRITE_SIZE // 2] = _GEN_BG
    return sprite


def _gen_too_close(a: _GenBox, b: _GenBox, gap: int = _GEN_SPRITE_GAP) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _gen_sprite_box(r: int, c: int) -> _GenBox:
    return (r, c, r + _GEN_SPRITE_SIZE - 1, c + _GEN_SPRITE_SIZE - 1)


def _gen_valid_outside_sprite_boxes(
    h: int, w: int, guide_box: _GenBox
) -> list[_GenBox]:
    boxes = []
    for r in range(h - _GEN_SPRITE_SIZE + 1):
        for c in range(w - _GEN_SPRITE_SIZE + 1):
            box = _gen_sprite_box(r, c)
            if not _gen_too_close(box, guide_box):
                boxes.append(box)
    return boxes


def _gen_choose_sprite_boxes(
    boxes: list[_GenBox], count: int, *, randomize: bool
) -> list[_GenBox] | None:
    order = list(boxes)
    if randomize:
        random.shuffle(order)

    def rec(candidates: list[_GenBox], chosen: list[_GenBox]):
        if len(chosen) == count:
            return chosen
        if len(candidates) < count - len(chosen):
            return None

        next_candidates = list(candidates)
        if randomize:
            random.shuffle(next_candidates)
        for i, box in enumerate(next_candidates):
            if any(_gen_too_close(box, old) for old in chosen):
                continue
            remaining = [
                other
                for other in next_candidates[i + 1 :]
                if not _gen_too_close(other, box)
            ]
            result = rec(remaining, chosen + [box])
            if result is not None:
                return result
        return None

    return rec(order, [])


def _gen_geometry_options(sprite_count: int) -> dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]]:
    if sprite_count in _GEN_GEOMETRY_CACHE:
        return _GEN_GEOMETRY_CACHE[sprite_count]

    by_grid: dict[tuple[int, int], list[tuple[int, int, list[tuple[int, int]]]]] = {}
    needed_outside = sprite_count - 1
    for h in range(_GEN_GRID_H[0], _GEN_GRID_H[1] + 1):
        for w in range(_GEN_GRID_W[0], _GEN_GRID_W[1] + 1):
            for gh in range(_GEN_GUIDE_H[0], _GEN_GUIDE_H[1] + 1):
                for gw in range(_GEN_GUIDE_W[0], _GEN_GUIDE_W[1] + 1):
                    ratio = gh * gw / (h * w)
                    if not _GEN_GUIDE_AREA_RATIO[0] <= ratio <= _GEN_GUIDE_AREA_RATIO[1]:
                        continue
                    positions = []
                    for gr0 in range(h - gh + 1):
                        for gc0 in range(w - gw + 1):
                            guide_box = (gr0, gc0, gr0 + gh - 1, gc0 + gw - 1)
                            boxes = _gen_valid_outside_sprite_boxes(h, w, guide_box)
                            if _gen_choose_sprite_boxes(boxes, needed_outside, randomize=False) is not None:
                                positions.append((gr0, gc0))
                    if positions:
                        by_grid.setdefault((h, w), []).append((gh, gw, positions))
    _GEN_GEOMETRY_CACHE[sprite_count] = by_grid
    return by_grid


def _gen_choose_geometry(sprite_count: int) -> tuple[int, int, _GenBox] | None:
    geometry_options = _gen_geometry_options(sprite_count)
    if not geometry_options:
        return None

    h, w = random.choice(list(geometry_options))
    gh, gw, guide_positions = random.choice(geometry_options[(h, w)])
    gr0, gc0 = random.choice(guide_positions)
    return h, w, (gr0, gc0, gr0 + gh - 1, gc0 + gw - 1)


def _gen_selected_location(guide_box: _GenBox) -> tuple[int, int]:
    gr0, gc0, gr1, gc1 = guide_box
    row_choices = list(range(gr0 + _GEN_SPRITE_GAP, gr1 - _GEN_SPRITE_SIZE + 1))
    col_choices = list(range(gc0 + _GEN_SPRITE_GAP, gc1 - _GEN_SPRITE_SIZE + 1))
    return random.choice(row_choices), random.choice(col_choices)


def _gen_perimeter_cells(box: _GenBox) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = [(r0, c) for c in range(c0, c1 + 1)]
    cells.extend((r, c1) for r in range(r0 + 1, r1 + 1))
    cells.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    cells.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return cells


def _gen_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    values = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _gen_sides_covered(cells: list[tuple[int, int]], box: _GenBox) -> bool:
    r0, c0, r1, c1 = box
    return (
        any(r == r0 for r, _ in cells)
        and any(r == r1 for r, _ in cells)
        and any(c == c0 for _, c in cells)
        and any(c == c1 for _, c in cells)
    )


def _gen_guide_cells(box: _GenBox) -> list[tuple[int, int]] | None:
    perimeter = _gen_perimeter_cells(box)
    perim_len = len(perimeter)
    min_count = int(np.ceil(_GEN_GUIDE_FILL[0] * perim_len))
    max_count = int(np.floor(_GEN_GUIDE_FILL[1] * perim_len))

    possible_fragments = []
    for fragments in range(_GEN_GUIDE_FRAGMENTS[0], _GEN_GUIDE_FRAGMENTS[1] + 1):
        lo = max(min_count, fragments)
        hi = min(max_count, perim_len - fragments)
        if lo <= hi:
            possible_fragments.append(fragments)

    for _ in range(_GEN_GUIDE_RETRIES):
        fragments = random.choice(possible_fragments)
        lo = max(min_count, fragments)
        hi = min(max_count, perim_len - fragments)
        guide_count = random.randint(lo, hi)
        runs = _gen_composition(guide_count, fragments)
        gaps = _gen_composition(perim_len - guide_count, fragments)

        pattern = []
        for run_len, gap_len in zip(runs, gaps):
            pattern.extend([1] * run_len)
            pattern.extend([0] * gap_len)

        shifts = list(range(perim_len))
        random.shuffle(shifts)
        for shift in shifts:
            cells = [
                perimeter[(i + shift) % perim_len]
                for i, is_guide in enumerate(pattern)
                if is_guide
            ]
            if _gen_sides_covered(cells, box):
                return cells
    return None


def _gen_render_grid(
    h: int,
    w: int,
    selected_loc: tuple[int, int],
    selected_color: int,
    other_boxes: list[_GenBox],
    other_colors: list[int],
    guide_cells: list[tuple[int, int]],
) -> np.ndarray:
    grid = np.full((h, w), _GEN_BG, dtype=int)
    selected_r, selected_c = selected_loc
    paste(grid, _gen_hollow_square(selected_color), selected_r, selected_c)
    for box, color in zip(other_boxes, other_colors):
        paste(grid, _gen_hollow_square(color), box[0], box[1])

    guide_rc = np.asarray(guide_cells, dtype=int)
    grid[guide_rc[:, 0], guide_rc[:, 1]] = _GEN_GUIDE_COLOR
    return grid


def _gen_passes_oracle(grid: np.ndarray, selected_color: int) -> bool:
    expected = _gen_hollow_square(selected_color)
    try:
        return np.array_equal(main(grid), expected)
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    sprite_count = random.randint(_GEN_SPRITE_COUNT[0], _GEN_SPRITE_COUNT[1])
    geometry = _gen_choose_geometry(sprite_count)
    if geometry is None:
        return None
    h, w, guide_box = geometry
    selected_loc = _gen_selected_location(guide_box)
    outside_boxes = _gen_valid_outside_sprite_boxes(h, w, guide_box)
    other_boxes = _gen_choose_sprite_boxes(outside_boxes, sprite_count - 1, randomize=True)
    if other_boxes is None:
        return None

    colors = random.sample(_GEN_SPRITE_COLORS, sprite_count)
    selected_color, other_colors = colors[0], colors[1:]

    guide_cells = _gen_guide_cells(guide_box)
    if guide_cells is None:
        return None

    grid = _gen_render_grid(h, w, selected_loc, selected_color, other_boxes, other_colors, guide_cells)
    if not _gen_passes_oracle(grid, selected_color):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid bf699163 input")
