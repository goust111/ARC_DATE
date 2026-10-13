import random
from collections import Counter

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
PINK = 6
TEAL = 8
MARKER_COLORS = (BLUE, RED, GREEN)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int,
    monochromatic: bool,
) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    dirs = DIR4 if connectivity == 4 else DIR8
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in dirs:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) == bg:
                        continue
                    if monochromatic and int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))
            components.append(comp)
    return components


def _component_colors(grid: np.ndarray, coords: list[tuple[int, int]]) -> set[int]:
    return {int(grid[r, c]) for r, c in coords}


def _infer_structure(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], dict[int, tuple[int, int]], dict[int, list[tuple[int, int]]], list[list[tuple[int, int]]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    components = _find_components(g, bg=bg, connectivity=8, monochromatic=False)
    assert len(components) >= 3, "expected one rectangle and at least two sprites"
    components.sort(key=len, reverse=True)

    rect_coords = components[0]
    r0, c0, r1, c1 = _bbox(rect_coords)
    rect = g[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(rect != bg), "target rectangle must be solidly filled"
    assert YELLOW in rect, "rectangle must contain yellow fill"
    assert set(map(int, np.unique(rect))) <= {YELLOW, *MARKER_COLORS}, "unexpected colors inside rectangle"

    markers: dict[int, tuple[int, int]] = {}
    for rr, cc in np.argwhere(np.isin(rect, MARKER_COLORS)):
        color = int(rect[rr, cc])
        assert color not in markers, "marker colors inside rectangle must be unique"
        markers[color] = (r0 + int(rr), c0 + int(cc))
    assert 2 <= len(markers) <= 3, "expected 2-3 rectangle markers"

    sprites = components[1:]
    templates: dict[int, list[tuple[int, int]]] = {}
    usable_sprites: list[list[tuple[int, int]]] = []
    for comp in sprites:
        colors = _component_colors(g, comp)
        if PINK not in colors:
            continue
        marker_colors = [color for color in colors if color not in {bg, PINK}]
        assert len(marker_colors) == 1, "each sprite must have exactly one marker color"
        marker_color = marker_colors[0]
        assert marker_color not in templates, "sprite marker colors must be unique"
        marker_cells = [(r, c) for r, c in comp if int(g[r, c]) == marker_color]
        pink_cells = [(r, c) for r, c in comp if int(g[r, c]) == PINK]
        assert len(marker_cells) == 1 and pink_cells, "sprite must contain one marker and at least one pink cell"
        mr, mc = marker_cells[0]
        templates[marker_color] = [(r - mr, c - mc) for r, c in pink_cells]
        usable_sprites.append(comp)

    assert set(templates) == set(markers), "every internal marker must match exactly one external sprite"
    return (r0, c0, r1, c1), markers, templates, usable_sprites


def _stamp_then_find_pairs(
    base: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    markers: dict[int, tuple[int, int]],
    templates: dict[int, list[tuple[int, int]]],
) -> tuple[np.ndarray, list[tuple[tuple[int, int], tuple[int, int]]]]:
    out = base.copy()
    r0, c0, r1, c1 = rect_bbox
    for color, (mr, mc) in markers.items():
        for dr, dc in templates[color]:
            rr = mr + dr
            cc = mc + dc
            assert r0 <= rr <= r1 and c0 <= cc <= c1, "stamp target left rectangle"
            out[rr, cc] = PINK
        out[mr, mc] = color

    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            if int(out[rr, cc]) != PINK:
                continue
            for dr, dc in DIR4:
                nr = rr + dr
                nc = cc + dc
                if r0 <= nr <= r1 and c0 <= nc <= c1 and int(out[nr, nc]) == YELLOW:
                    pairs.append(((rr, cc), (nr, nc)))
    return out, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each external pink sprite into the matching yellow rectangle, then erase every new pink-yellow domino by turning the yellow side teal.

    Input structure:
    - The grid has a black background, one large yellow filled rectangle, and 2-3 small external sprites.
    - The rectangle contains one isolated marker pixel of each used marker color among blue, red, and green.
    - Each external sprite contains pink stamp pixels plus exactly one marker pixel whose color matches one rectangle marker.

    Rule:
    - Find the largest non-black component and treat its full bounding box as the target rectangle.
    - For each rectangle marker, find the external sprite with the same marker color and copy that sprite's pink offsets relative to its marker into the rectangle, preserving the marker itself.
    - After all pink pixels are stamped, scan the rectangle for every 4-neighbor pink-yellow pair.
    - For each such domino, recolor the yellow cell to teal and erase the pink cell to black; erase all external sprites to black as well.

    Key insight:
    - The pink-yellow dominoes do not exist in the input scene; they are created only after aligning the off-rectangle sprite masks to their same-colored markers inside the filled rectangle.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    rect_bbox, markers, templates, sprite_components = _infer_structure(g)

    out = g.copy()
    for comp in sprite_components:
        for r, c in comp:
            out[r, c] = bg

    stamped, pairs = _stamp_then_find_pairs(out, rect_bbox, markers, templates)
    for (pr, pc), (yr, yc) in pairs:
        stamped[pr, pc] = bg
        stamped[yr, yc] = TEAL
    return stamped


_CORNER_SPECS = {
    "nw": [((0, 1), (0, -1)), ((1, 0), (-1, 0))],
    "ne": [((0, -2), (0, 1)), ((1, -1), (-1, 0))],
    "sw": [((-2, 0), (1, 0)), ((-1, 1), (0, -1))],
    "se": [((-2, -1), (1, 0)), ((-1, -2), (0, 1))],
}


def _actual_corner_option(corner: str, rh: int, rw: int) -> tuple[tuple[int, int], tuple[int, int]]:
    marker_rel, offset = random.choice(_CORNER_SPECS[corner])
    mr, mc = marker_rel
    if mr < 0:
        mr += rh
    if mc < 0:
        mc += rw
    return (mr, mc), offset


def _non_adjacent(points: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(points):
        for r2, c2 in points[i + 1 :]:
            if abs(r1 - r2) + abs(c1 - c2) == 1:
                return False
    return True


def _make_sprite_patch(marker_color: int, offset: tuple[int, int]) -> np.ndarray:
    raw = [(0, 0), offset]
    min_r = min(r for r, _ in raw)
    max_r = max(r for r, _ in raw)
    min_c = min(c for _, c in raw)
    max_c = max(c for _, c in raw)
    span_h = max_r - min_r + 1
    span_w = max_c - min_c + 1

    patch_h = random.randint(max(2, span_h), 3)
    patch_w = random.randint(max(2, span_w), 3)
    top_pad = random.randint(0, patch_h - span_h)
    left_pad = random.randint(0, patch_w - span_w)

    marker_r = top_pad - min_r
    marker_c = left_pad - min_c
    pink_r = marker_r + offset[0]
    pink_c = marker_c + offset[1]

    patch = np.zeros((patch_h, patch_w), dtype=int)
    patch[marker_r, marker_c] = marker_color
    patch[pink_r, pink_c] = PINK
    return patch


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    for r in range(ph):
        for c in range(pw):
            if int(patch[r, c]) == BG:
                continue
            rr = top + r
            cc = left + c
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            for nr in range(rr - 1, rr + 2):
                for nc in range(cc - 1, cc + 2):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG:
                        return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray) -> tuple[int, int] | None:
    h, w = grid.shape
    ph, pw = patch.shape
    candidates = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(candidates)
    for r, c in candidates:
        if not _can_place_patch(grid, patch, r, c):
            continue
        mask = patch != BG
        grid[r : r + ph, c : c + pw][mask] = patch[mask]
        return r, c
    return None


def _validate_constructed(
    grid: np.ndarray,
    rect_bbox: tuple[int, int, int, int],
    markers: dict[int, tuple[int, int]],
    templates: dict[int, list[tuple[int, int]]],
    sprite_cells: list[tuple[int, int]],
) -> bool:
    r0, c0, r1, c1 = rect_bbox
    rect = grid[r0 : r1 + 1, c0 : c1 + 1]
    if np.any(rect == PINK):
        return False
    if not _non_adjacent(list(markers.values())):
        return False

    cleaned = grid.copy()
    for r, c in sprite_cells:
        cleaned[r, c] = BG
    stamped, pairs = _stamp_then_find_pairs(cleaned, rect_bbox, markers, templates)

    pink_cells = [(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1) if int(stamped[r, c]) == PINK]
    if len(pink_cells) != len(markers):
        return False
    if any(abs(r1 - r2) + abs(c1 - c2) == 1 for i, (r1, c1) in enumerate(pink_cells) for r2, c2 in pink_cells[i + 1 :]):
        return False
    if len(pairs) != len(markers):
        return False

    pair_counts: Counter[tuple[int, int]] = Counter(pair[0] for pair in pairs)
    yellow_counts: Counter[tuple[int, int]] = Counter(pair[1] for pair in pairs)
    if any(count != 1 for count in pair_counts.values()):
        return False
    if any(count != 1 for count in yellow_counts.values()):
        return False

    for (pr, pc), (yr, yc) in pairs:
        yellow_neighbors = [
            (nr, nc)
            for dr, dc in DIR4
            for nr, nc in [(pr + dr, pc + dc)]
            if r0 <= nr <= r1 and c0 <= nc <= c1 and int(stamped[nr, nc]) == YELLOW
        ]
        if yellow_neighbors != [(yr, yc)] and set(yellow_neighbors) != {(yr, yc)}:
            return False

    out = main(grid)
    if np.count_nonzero(out == TEAL) != len(markers):
        return False
    for r, c in sprite_cells:
        if int(out[r, c]) != BG:
            return False
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    rh = random.randint(8, min(12, h - 6))
    rw = random.randint(8, min(12, w - 6))
    top = random.randint(3, h - rh - 3)
    left = random.randint(3, w - rw - 3)
    grid = np.zeros((h, w), dtype=int)
    grid[top : top + rh, left : left + rw] = YELLOW

    marker_count = random.choice([2, 3])
    chosen_colors = random.sample(list(MARKER_COLORS), marker_count)
    chosen_corners = random.sample(list(_CORNER_SPECS), marker_count)

    markers: dict[int, tuple[int, int]] = {}
    templates: dict[int, list[tuple[int, int]]] = {}
    sprite_cells: list[tuple[int, int]] = []

    rel_positions: list[tuple[int, int]] = []
    pending: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color, corner in zip(chosen_colors, chosen_corners):
        marker_rel, offset = _actual_corner_option(corner, rh, rw)
        rel_positions.append(marker_rel)
        pending.append((color, marker_rel, offset))
    if not _non_adjacent(rel_positions):
        return None

    for color, (mr, mc), offset in pending:
        markers[color] = (top + mr, left + mc)
        templates[color] = [offset]
        grid[top + mr, left + mc] = color

    for color in chosen_colors:
        patch = _make_sprite_patch(color, templates[color][0])
        loc = _place_patch(grid, patch)
        if loc is None:
            return None
        base_r, base_c = loc
        for rr, cc in np.argwhere(patch != BG):
            sprite_cells.append((base_r + int(rr), base_c + int(cc)))

    comps = _find_components(grid, bg=BG, connectivity=8, monochromatic=False)
    if len(comps) != 1 + marker_count:
        return None

    rect_bbox = (top, left, top + rh - 1, left + rw - 1)
    if not _validate_constructed(grid, rect_bbox, markers, templates, sprite_cells):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
