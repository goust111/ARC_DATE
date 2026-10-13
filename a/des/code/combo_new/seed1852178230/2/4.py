import random
from itertools import combinations

import numpy as np


SIDES = ("top", "right", "bottom", "left")
SIDE_TO_TEMPLATE_POS = {
    "top": (0, 1),
    "right": (1, 2),
    "bottom": (2, 1),
    "left": (1, 0),
}
CORNER_SLICES = {
    "tl": lambda h, w: (slice(0, 3), slice(0, 3)),
    "tr": lambda h, w: (slice(0, 3), slice(w - 3, w)),
    "bl": lambda h, w: (slice(h - 3, h), slice(0, 3)),
    "br": lambda h, w: (slice(h - 3, h), slice(w - 3, w)),
}
SIDE_SETS = {
    k: [tuple(side_set) for side_set in combinations(SIDES, k)]
    for k in (1, 2, 3)
}

MIN_GRID = 18
MAX_GRID = 30
MIN_RECT_SIDE = 5
MAX_RECT_SIDE = 13
BOX_PADDING = 1
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    for getter in CORNER_SLICES.values():
        rs, cs = getter(h, w)
        mask[rs, cs] = True
    return mask


def _extract_templates(grid: np.ndarray, bg: int) -> dict[int, dict[str, int]]:
    h, w = grid.shape
    templates: dict[int, dict[str, int]] = {}
    for getter in CORNER_SLICES.values():
        rs, cs = getter(h, w)
        patch = grid[rs, cs]
        non_bg = np.argwhere(patch != bg)
        count = int(len(non_bg))
        if count == 0:
            continue
        assert count in (1, 2, 3), "corner template sizes must key the gap count"
        allowed = set(SIDE_TO_TEMPLATE_POS.values())
        assert all(tuple(map(int, cell)) in allowed for cell in non_bg), "templates use the four cardinal slots"
        assert count not in templates, "each gap count must map to one template"
        templates[count] = {
            side: int(patch[pos])
            for side, pos in SIDE_TO_TEMPLATE_POS.items()
        }
    assert set(templates) == {1, 2, 3}, "expected one template for each gap count 1..3"
    return templates


def _gap_positions(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    return {
        "top": (r0, mid_c),
        "right": (mid_r, c1),
        "bottom": (r1, mid_c),
        "left": (mid_r, c0),
    }


def _validate_box(
    grid: np.ndarray,
    bg: int,
    color: int,
    forbidden: np.ndarray,
    r0: int,
    c0: int,
    r1: int,
    c1: int,
):
    if r1 - r0 < 4 or c1 - c0 < 4:
        return None
    if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
        return None
    if forbidden[r0 : r1 + 1, c0 : c1 + 1].any():
        return None
    if np.any(grid[r0 + 1 : r1, c0 + 1 : c1] != bg):
        return None

    gaps_by_pos = {coord: side for side, coord in _gap_positions(r0, c0, r1, c1).items()}
    gap_sides: list[str] = []
    color_cells: list[tuple[int, int]] = []

    for c in range(c0, c1 + 1):
        for r in (r0, r1):
            val = int(grid[r, c])
            coord = (r, c)
            if coord in gaps_by_pos:
                if val == bg:
                    gap_sides.append(gaps_by_pos[coord])
                elif val == color:
                    color_cells.append(coord)
                else:
                    return None
            elif val == color:
                color_cells.append(coord)
            else:
                return None

    for r in range(r0 + 1, r1):
        for c in (c0, c1):
            val = int(grid[r, c])
            coord = (r, c)
            if coord in gaps_by_pos:
                if val == bg:
                    gap_sides.append(gaps_by_pos[coord])
                elif val == color:
                    color_cells.append(coord)
                else:
                    return None
            elif val == color:
                color_cells.append(coord)
            else:
                return None

    if not (1 <= len(gap_sides) <= 3):
        return None
    return {
        "bbox": (r0, c0, r1, c1),
        "gaps": tuple(side for side in SIDES if side in gap_sides),
        "color_cells": tuple(color_cells),
        "area": (r1 - r0 + 1) * (c1 - c0 + 1),
    }


def _find_rectangles(grid: np.ndarray, bg: int, templates: dict[int, dict[str, int]]):
    h, w = grid.shape
    forbidden = _corner_mask((h, w))
    colors = sorted(int(v) for v in np.unique(grid) if int(v) != bg)
    matches = []

    for color in colors:
        candidates = []
        for r0 in range(0, h - 4):
            for c0 in range(0, w - 4):
                if int(grid[r0, c0]) != color or forbidden[r0, c0]:
                    continue
                for r1 in range(r0 + 4, h, 2):
                    if int(grid[r1, c0]) != color:
                        continue
                    for c1 in range(c0 + 4, w, 2):
                        if int(grid[r0, c1]) != color or int(grid[r1, c1]) != color:
                            continue
                        candidate = _validate_box(grid, bg, color, forbidden, r0, c0, r1, c1)
                        if candidate is not None:
                            candidates.append(candidate)

        if not candidates:
            continue

        candidates.sort(key=lambda item: (-item["area"], item["bbox"]))
        occupied = np.zeros((h, w), dtype=bool)
        selected = []
        for candidate in candidates:
            r0, c0, r1, c1 = candidate["bbox"]
            if occupied[r0 : r1 + 1, c0 : c1 + 1].any():
                continue
            if any(len(candidate["gaps"]) == count for count in templates):
                selected.append(candidate)
                occupied[r0 : r1 + 1, c0 : c1 + 1] = True

        if not selected:
            continue

        color_mask = (grid == color) & (~forbidden)
        covered = np.zeros((h, w), dtype=bool)
        for candidate in selected:
            for r, c in candidate["color_cells"]:
                covered[r, c] = True
        if np.array_equal(color_mask, covered):
            matches.append((color, selected))

    assert len(matches) == 1, "expected a unique frame color"
    rectangles = matches[0][1]
    assert rectangles, "expected at least one hollow rectangle"
    return rectangles


def _ray_cells(bbox: tuple[int, int, int, int], side: str):
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side in ("top", "bottom"):
        return [(r, mid_c) for r in range(r0 + 1, r1)]
    return [(mid_r, c) for c in range(c0 + 1, c1)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow rectangle interior with rays chosen from the corner template whose non-background count matches its number of border gaps.

    Input structure:
    - The grid has a dominant background color.
    - Several one-pixel-thick hollow rectangles share one border color; each border has 1 to 3 single-cell gaps placed at side midpoints.
    - Three 3x3 corner templates act as a library: each non-empty template has only cardinal-direction pixels and its non-background pixel count is 1, 2, or 3.

    Rule:
    - Read the corner template whose non-background count equals a rectangle's number of gaps.
    - For every gap, determine which side it lies on and look up the color at the same direction in that template.
    - Paint a one-cell-wide ray from the gap inward through the rectangle interior until the opposite border, while leaving the border and the gap cell unchanged.

    Key insight:
    - The corner sprites are not objects to copy directly; they are an indexed legend where pixel count chooses the template and directional placement inside that template chooses the ray color for each gap side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _get_bg(g)
    templates = _extract_templates(g, bg)
    rectangles = _find_rectangles(g, bg, templates)

    out = g.copy()
    for rect in rectangles:
        gap_count = len(rect["gaps"])
        template = templates[gap_count]
        for side in rect["gaps"]:
            color = int(template[side])
            assert color != bg, "gap directions must map to visible template colors"
            for r, c in _ray_cells(rect["bbox"], side):
                if int(out[r, c]) == bg:
                    out[r, c] = color
                else:
                    assert int(out[r, c]) == color, "overlapping rays must agree in color"
    return out


def _template_patch(bg: int, color: int, sides: tuple[str, ...]) -> np.ndarray:
    patch = np.full((3, 3), bg, dtype=int)
    for side in sides:
        patch[SIDE_TO_TEMPLATE_POS[side]] = color
    return patch


def _odd_choices(limit: int) -> list[int]:
    upper = min(MAX_RECT_SIDE, limit)
    return [size for size in range(MIN_RECT_SIDE, upper + 1, 2)]


def _expanded_slice(start: int, stop: int, limit: int, pad: int = BOX_PADDING) -> slice:
    return slice(max(0, start - pad), min(limit, stop + pad))


def _try_place_rectangle(
    grid: np.ndarray,
    blocked: np.ndarray,
    bg: int,
    frame_color: int,
    gap_sides: tuple[str, ...],
) -> bool:
    h, w = grid.shape
    heights = _odd_choices(h - 2)
    widths = _odd_choices(w - 2)
    if not heights or not widths:
        return False

    for _ in range(200):
        height = random.choice(heights)
        width = random.choice(widths)
        if height >= h - 1 or width >= w - 1:
            continue

        r0 = random.randint(1, h - height - 1)
        c0 = random.randint(1, w - width - 1)
        r1 = r0 + height - 1
        c1 = c0 + width - 1

        rs = _expanded_slice(r0, r1 + 1, h)
        cs = _expanded_slice(c0, c1 + 1, w)
        if blocked[rs, cs].any():
            continue

        grid[r0, c0 : c1 + 1] = frame_color
        grid[r1, c0 : c1 + 1] = frame_color
        grid[r0 : r1 + 1, c0] = frame_color
        grid[r0 : r1 + 1, c1] = frame_color
        for side, (gr, gc) in _gap_positions(r0, c0, r1, c1).items():
            if side in gap_sides:
                grid[gr, gc] = bg

        blocked[rs, cs] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    palette = list(range(10))
    bg, frame_color, col1, col2, col3 = random.sample(palette, 5)
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    side_sets = {
        1: random.choice(SIDE_SETS[1]),
        2: random.choice(SIDE_SETS[2]),
        3: random.choice(SIDE_SETS[3]),
    }
    template_colors = {1: col1, 2: col2, 3: col3}

    corners = list(CORNER_SLICES)
    random.shuffle(corners)
    placements = {count: corners[count - 1] for count in (1, 2, 3)}
    for count, corner in placements.items():
        rs, cs = CORNER_SLICES[corner](h, w)
        grid[rs, cs] = _template_patch(bg, template_colors[count], side_sets[count])

    reserve = np.zeros((h, w), dtype=bool)
    reserve[:4, :4] = True
    reserve[:4, max(0, w - 4) : w] = True
    reserve[max(0, h - 4) : h, :4] = True
    reserve[max(0, h - 4) : h, max(0, w - 4) : w] = True
    blocked |= reserve

    counts = [1, 2, 3]
    extra = random.randint(0, 2)
    counts.extend(random.choice((1, 2, 3)) for _ in range(extra))
    random.shuffle(counts)

    for count in counts:
        if not _try_place_rectangle(grid, blocked, bg, frame_color, side_sets[count]):
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-template ray puzzle")
