from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PATH_COLOR = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]

GEN_RETRIES = 400
DROP_COUNT_RANGE = (4, 6)
GRID_SIDE_RANGE = (18, 30)
SPRITE_SIDE_RANGE = (4, 6)
SPRITE_EXTRA_PIXELS = (3, 7)
PATH_WIDTH_RANGE = (6, 11)
PATH_ROWS_RANGE = (3, 5)
DROP_GAP_RANGE = (1, 3)


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: list[Coord] | np.ndarray) -> BBox:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _grow_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    while True:
        mask = np.zeros((h, w), dtype=bool)
        start = (random.randrange(h), random.randrange(w))
        mask[start] = True
        frontier = set()
        frontier.update(_neighbors4(*start))
        while int(mask.sum()) < area:
            candidates = [
                (r, c)
                for r, c in frontier
                if 0 <= r < h and 0 <= c < w and not mask[r, c]
            ]
            if not candidates:
                break
            cell = random.choice(candidates)
            mask[cell] = True
            frontier.remove(cell)
            frontier.update(_neighbors4(*cell))
        if int(mask.sum()) != area:
            continue
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        cropped = mask[r0 : r1 + 1, c0 : c1 + 1]
        if cropped.shape[0] < 3 or cropped.shape[1] < 3:
            continue
        if not cropped[0].any() or not cropped[-1].any() or not cropped[:, 0].any() or not cropped[:, -1].any():
            continue
        return cropped


def _make_sprite(drop_count: int) -> tuple[np.ndarray, dict[int, Coord]]:
    for _ in range(200):
        h = random.randint(*SPRITE_SIDE_RANGE)
        w = random.randint(*SPRITE_SIDE_RANGE)
        max_area = min(h * w - 1, drop_count + SPRITE_EXTRA_PIXELS[1])
        min_area = max(drop_count + SPRITE_EXTRA_PIXELS[0], h + w - 1)
        if min_area > max_area:
            continue
        area = random.randint(min_area, max_area)
        mask = _grow_connected_mask(h, w, area)
        sh, sw = mask.shape
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if len(coords) < drop_count + 2:
            continue

        grid = np.full((sh, sw), BG, dtype=int)
        anchor_colors = random.sample(PALETTE, drop_count)
        anchor_cells = random.sample(coords, drop_count)
        anchor_map: dict[int, Coord] = {}
        for color, cell in zip(anchor_colors, anchor_cells):
            grid[cell] = int(color)
            anchor_map[int(color)] = cell

        filler_cells = [cell for cell in coords if cell not in set(anchor_cells)]
        filler_palette = [c for c in PALETTE if c not in anchor_colors]
        filler_colors = random.sample(filler_palette, random.randint(1, min(3, len(filler_palette))))
        for cell in filler_cells:
            grid[cell] = int(random.choice(filler_colors))

        if len(set(map(int, grid[grid != BG].ravel()))) < drop_count + 1:
            continue
        for color in anchor_colors:
            if int(np.sum(grid == color)) != 1:
                break
        else:
            return grid, anchor_map
    raise RuntimeError("failed to generate sprite")


def _corner_bbox(top: int, left: int, h: int, w: int) -> BBox:
    return int(top), int(left), int(top + h - 1), int(left + w - 1)


def _expand_bbox(bbox: BBox, margin: int, h: int, w: int) -> BBox:
    r0, c0, r1, c1 = bbox
    return max(0, r0 - margin), max(0, c0 - margin), min(h - 1, r1 + margin), min(w - 1, c1 + margin)


def _inside_bbox(cell: Coord, bbox: BBox) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _stamp_mask(sprite: np.ndarray, top: int, left: int) -> set[Coord]:
    coords = np.argwhere(sprite != BG)
    return {(int(top + r), int(left + c)) for r, c in coords}


def _paint_stamp(out: np.ndarray, sprite: np.ndarray, top: int, left: int, bg: int) -> None:
    sh, sw = sprite.shape
    patch = out[top : top + sh, left : left + sw]
    mask = sprite != bg
    patch[mask] = sprite[mask]


def _simulate_stamps(shape: tuple[int, int], sprite: np.ndarray, drops: list[tuple[Coord, int]], bg: int) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    for (r, c), color in drops:
        anchors = np.argwhere(sprite == int(color))
        assert len(anchors) == 1
        ar, ac = map(int, anchors[0])
        top = int(r - ar)
        left = int(c - ac)
        _paint_stamp(out, sprite, top, left, bg)
    return out


def _parse_input(grid: np.ndarray) -> tuple[list[Coord], np.ndarray, BBox]:
    bg = _get_bg(grid)
    comps = _components(grid != bg)
    assert len(comps) == 2, "expected exactly one path component and one sprite component"

    path_comp = None
    sprite_comp = None
    for comp in comps:
        colors = {int(grid[r, c]) for r, c in comp}
        if PATH_COLOR in colors:
            assert path_comp is None
            path_comp = comp
        else:
            assert sprite_comp is None
            sprite_comp = comp

    assert path_comp is not None and sprite_comp is not None
    sprite_bbox = _bbox_from_coords(sprite_comp)
    sprite = _crop_bbox(grid, sprite_bbox)
    return path_comp, sprite, sprite_bbox


def _traverse_path(grid: np.ndarray, path_comp: list[Coord]) -> list[tuple[Coord, int]]:
    path_set = set(path_comp)
    adjacency: dict[Coord, list[Coord]] = {}
    endpoints: list[Coord] = []
    for cell in path_comp:
        nbrs = [nbr for nbr in _neighbors4(*cell) if nbr in path_set]
        adjacency[cell] = nbrs
        assert len(nbrs) <= 2, "path must have no branches"
        if len(nbrs) == 1:
            endpoints.append(cell)
    assert len(endpoints) == 2, "path must have exactly two endpoints"

    start = min(endpoints)
    ordered: list[Coord] = []
    prev = None
    cur = start
    while True:
        ordered.append(cur)
        nxt = [nbr for nbr in adjacency[cur] if nbr != prev]
        if not nxt:
            break
        assert len(nxt) == 1
        prev, cur = cur, nxt[0]
    assert len(ordered) == len(path_comp), "path traversal must cover every cell exactly once"

    bg = _get_bg(grid)
    drops = []
    for r, c in ordered:
        color = int(grid[r, c])
        if color not in (bg, PATH_COLOR):
            drops.append(((int(r), int(c)), color))
    assert drops, "path must contain at least one colored drop point"
    return drops


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp the same multicolor sprite onto each colored snake waypoint in traversal order, with later stamps overwriting earlier ones.

    Input structure:
    - Black background with one disconnected multicolor sprite in a corner.
    - Elsewhere there is one 4-connected unbranched snake path made of gray cells, except some path cells are replaced by non-gray drop colors.
    - The sprite contains exactly one pixel of every drop color.

    Rule:
    - Find the snake's two endpoints and start from the one with smaller (row, col).
    - Traverse the snake cell by cell; whenever a colored drop cell is encountered, align the sprite so its unique same-colored pixel lands on that drop cell, then stamp the whole non-background sprite.
    - Start from a blank background canvas and keep only the accumulated stamps, so the input snake and corner legend disappear; later stamps overwrite earlier ones in overlaps.

    Key insight:
    - The snake is a hidden time axis: the sprite placement comes from color matching, but the visible overlap colors depend on the endpoint-selected traversal order.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    path_comp, sprite, _sprite_bbox = _parse_input(g)
    drops = _traverse_path(g, path_comp)
    return _simulate_stamps(g.shape, sprite, drops, bg)


def _build_snake_path(r0: int, c0: int, width: int, lanes: int, start_from_left: bool) -> list[Coord]:
    cur_r = int(r0)
    cur_c = int(c0 if start_from_left else c0 + width - 1)
    direction = 1 if start_from_left else -1
    order = [(cur_r, cur_c)]
    for lane in range(lanes):
        end_c = c0 + width - 1 if direction == 1 else c0
        while cur_c != end_c:
            cur_c += direction
            order.append((cur_r, cur_c))
        if lane == lanes - 1:
            break
        order.append((cur_r + 1, cur_c))
        cur_r += 2
        order.append((cur_r, cur_c))
        direction *= -1
    return order


def _safe_bounds_for_anchors(shape: tuple[int, int], sprite: np.ndarray, anchors: dict[int, Coord]) -> BBox:
    h, w = shape
    sh, sw = sprite.shape
    rows = [cell[0] for cell in anchors.values()]
    cols = [cell[1] for cell in anchors.values()]
    r0 = max(rows)
    c0 = max(cols)
    r1 = h - sh + min(rows)
    c1 = w - sw + min(cols)
    return int(r0), int(c0), int(r1), int(c1)


def _choose_corner_position(h: int, w: int, sh: int, sw: int) -> tuple[str, int, int]:
    corner = random.choice(["tl", "tr", "bl", "br"])
    if corner == "tl":
        return corner, 0, 0
    if corner == "tr":
        return corner, 0, w - sw
    if corner == "bl":
        return corner, h - sh, 0
    return corner, h - sh, w - sw


def _find_path_candidates(
    shape: tuple[int, int],
    safe_bbox: BBox,
    forbidden: BBox,
) -> list[tuple[int, int, int, int, bool]]:
    h, w = shape
    safe_r0, safe_c0, safe_r1, safe_c1 = safe_bbox
    candidates: list[tuple[int, int, int, int, bool]] = []
    for lanes in range(PATH_ROWS_RANGE[0], PATH_ROWS_RANGE[1] + 1):
        band_h = 2 * lanes - 1
        if band_h > (safe_r1 - safe_r0 + 1):
            continue
        for width in range(PATH_WIDTH_RANGE[0], PATH_WIDTH_RANGE[1] + 1):
            if width > (safe_c1 - safe_c0 + 1):
                continue
            for r0 in range(safe_r0, safe_r1 - band_h + 2):
                for c0 in range(safe_c0, safe_c1 - width + 2):
                    for start_left in (False, True):
                        order = _build_snake_path(r0, c0, width, lanes, start_left)
                        if any(_inside_bbox(cell, forbidden) for cell in order):
                            continue
                        end0, end1 = order[0], order[-1]
                        if min(end0, end1) != end0:
                            continue
                        if not all(0 <= rr < h and 0 <= cc < w for rr, cc in order):
                            continue
                        candidates.append((r0, c0, width, lanes, start_left))
    return candidates


def _choose_drop_sequence(
    path_order: list[Coord],
    sprite: np.ndarray,
    anchor_map: dict[int, Coord],
) -> tuple[list[tuple[Coord, int]], np.ndarray]:
    colors = list(anchor_map)
    sprite_area = int(np.sum(sprite != BG))
    usable = list(range(1, len(path_order) - 1))
    for _ in range(300):
        random.shuffle(colors)
        start_idx = random.choice(usable[:-len(colors)])
        idxs = [start_idx]
        for _color in colors[1:]:
            idxs.append(idxs[-1] + random.randint(*DROP_GAP_RANGE))
        if idxs[-1] >= len(path_order) - 1:
            continue

        drops = [(path_order[idx], int(color)) for idx, color in zip(idxs, colors)]
        prev_mask = None
        ok = True
        for (r, c), color in drops:
            ar, ac = anchor_map[color]
            mask = _stamp_mask(sprite, r - ar, c - ac)
            if prev_mask is not None:
                overlap = len(prev_mask & mask)
                if overlap == 0 or overlap > max(1, sprite_area // 2):
                    ok = False
                    break
            prev_mask = mask
        if not ok:
            continue

        forward = _simulate_stamps((30, 30), sprite, drops, BG)
        backward = _simulate_stamps((30, 30), sprite, list(reversed(drops)), BG)
        if np.array_equal(forward, backward):
            continue
        return drops, forward
    raise RuntimeError("failed to choose non-degenerate drop sequence")


def _render_input(
    shape: tuple[int, int],
    sprite: np.ndarray,
    sprite_top: int,
    sprite_left: int,
    path_order: list[Coord],
    drops: list[tuple[Coord, int]],
) -> np.ndarray:
    h, w = shape
    grid = np.full((h, w), BG, dtype=int)
    grid[sprite_top : sprite_top + sprite.shape[0], sprite_left : sprite_left + sprite.shape[1]][sprite != BG] = sprite[sprite != BG]
    for r, c in path_order:
        grid[r, c] = PATH_COLOR
    for (r, c), color in drops:
        grid[r, c] = int(color)
    return grid


def _semantic_checks(
    grid: np.ndarray,
    sprite: np.ndarray,
    drops: list[tuple[Coord, int]],
    expected: np.ndarray,
) -> bool:
    try:
        solved = main(grid)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    path_comp, parsed_sprite, _ = _parse_input(grid)
    if not np.array_equal(parsed_sprite, sprite):
        return False

    ordered = _traverse_path(grid, path_comp)
    if ordered != drops:
        return False

    reverse = _simulate_stamps(grid.shape, sprite, list(reversed(drops)), BG)
    if np.array_equal(reverse, expected):
        return False

    for i in range(len(drops) - 1):
        (r0, c0), col0 = drops[i]
        (r1, c1), col1 = drops[i + 1]
        a0 = np.argwhere(sprite == col0)
        a1 = np.argwhere(sprite == col1)
        if len(a0) != 1 or len(a1) != 1:
            return False
        mask0 = _stamp_mask(sprite, r0 - int(a0[0, 0]), c0 - int(a0[0, 1]))
        mask1 = _stamp_mask(sprite, r1 - int(a1[0, 0]), c1 - int(a1[0, 1]))
        if not (mask0 & mask1):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    drop_count = random.randint(*DROP_COUNT_RANGE)
    sprite, anchor_map = _make_sprite(drop_count)
    sh, sw = sprite.shape
    if sh >= h - 6 or sw >= w - 6:
        return None

    _corner, sprite_top, sprite_left = _choose_corner_position(h, w, sh, sw)
    sprite_bbox = _corner_bbox(sprite_top, sprite_left, sh, sw)
    forbidden = _expand_bbox(sprite_bbox, 1, h, w)
    safe_bbox = _safe_bounds_for_anchors((h, w), sprite, anchor_map)
    sr0, sc0, sr1, sc1 = safe_bbox
    if sr0 > sr1 or sc0 > sc1:
        return None

    candidates = _find_path_candidates((h, w), safe_bbox, forbidden)
    if not candidates:
        return None
    r0, c0, width, lanes, start_left = random.choice(candidates)
    path_order = _build_snake_path(r0, c0, width, lanes, start_left)

    try:
        drops, _dummy = _choose_drop_sequence(path_order, sprite, anchor_map)
    except RuntimeError:
        return None

    expected = _simulate_stamps((h, w), sprite, drops, BG)
    grid = _render_input((h, w), sprite, sprite_top, sprite_left, path_order, drops)

    non_bg_comps = _components(grid != BG)
    if len(non_bg_comps) != 2:
        return None
    if not _semantic_checks(grid, sprite, drops, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
