from __future__ import annotations

from collections import Counter, deque
import random
from typing import Iterable

import numpy as np


PLACEHOLDER = 3
MAX_SIDE = 30
MAX_GEN_TRIES = 400
GEN_BG_CHOICES = [0, 1, 2, 4, 5, 6, 7, 8, 9]
GEN_COLOR_POOL = [1, 2, 4, 5, 6, 7, 8, 9]


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_coords(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = list(coords)
    rs = [int(r) for r, _c in coords]
    cs = [int(c) for _r, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors(r: int, c: int, connectivity: int) -> Iterable[tuple[int, int]]:
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1
    if int(connectivity) == 8:
        yield r - 1, c - 1
        yield r - 1, c + 1
        yield r + 1, c - 1
        yield r + 1, c + 1


def _find_components(
    grid: np.ndarray,
    *,
    bg: int,
    connectivity: int,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    out: list[dict] = []
    for r in range(H):
        for c in range(W):
            if seen[r, c] or int(g[r, c]) == int(bg):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors(int(rr), int(cc), connectivity):
                    if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) != int(bg):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            out.append(
                {
                    "coords": coords,
                    "bbox": _bbox_coords(coords),
                    "area": len(coords),
                }
            )
    out.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1]), int(comp["area"])))
    return out


def _is_connected(coords: Iterable[tuple[int, int]], *, connectivity: int = 4) -> bool:
    pts = {tuple(map(int, rc)) for rc in coords}
    if not pts:
        return False
    start = next(iter(pts))
    seen = {start}
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nb in _neighbors(int(r), int(c), connectivity):
            if nb in pts and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(pts)


def _map_from_component(grid: np.ndarray, comp: dict) -> dict[tuple[int, int], int]:
    g = np.asarray(grid, dtype=int)
    return {(int(r), int(c)): int(g[int(r), int(c)]) for r, c in comp["coords"]}


def _tight_grid_from_map(color_map: dict[tuple[int, int], int], bg: int) -> np.ndarray:
    bbox = _bbox_coords(color_map)
    r0, c0, r1, c1 = bbox
    out = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
    for (r, c), color in color_map.items():
        out[int(r - r0), int(c - c0)] = int(color)
    return out


def _is_horizontally_symmetric(color_map: dict[tuple[int, int], int]) -> bool:
    if not color_map:
        return False
    cols = [int(c) for _r, c in color_map]
    x2 = min(cols) + max(cols)
    for (r, c), color in color_map.items():
        if color_map.get((int(r), int(x2 - c))) != int(color):
            return False
    return True


def _validate_alignment(
    frag_a: dict[tuple[int, int], int],
    frag_b: dict[tuple[int, int], int],
    *,
    bg: int,
    dr: int,
    dc: int,
) -> dict | None:
    moved_b: dict[tuple[int, int], int] = {}
    for (r, c), color in frag_b.items():
        pos = (int(r + dr), int(c + dc))
        if pos in frag_a or pos in moved_b:
            return None
        moved_b[pos] = int(color)

    full_map = dict(frag_a)
    full_map.update(moved_b)

    cols = [int(c) for _r, c in full_map]
    x2 = min(cols) + max(cols)
    for (r, c), color in full_map.items():
        partner = (int(r), int(x2 - c))
        partner_color = full_map.get(partner)
        if partner_color is None:
            return None
        if int(color) == int(PLACEHOLDER):
            if int(partner_color) in (int(bg), int(PLACEHOLDER)):
                return None
            continue
        if int(partner_color) == int(PLACEHOLDER):
            continue
        if int(partner_color) != int(color):
            return None

    return {
        "dr": int(dr),
        "dc": int(dc),
        "x2": int(x2),
        "full_map": full_map,
    }


def _find_alignments(
    frag_a: dict[tuple[int, int], int],
    frag_b: dict[tuple[int, int], int],
    *,
    bg: int,
    stop_after: int = 2,
) -> list[dict]:
    a_bbox = _bbox_coords(frag_a)
    b_bbox = _bbox_coords(frag_b)
    a_r0, a_c0, a_r1, a_c1 = a_bbox
    b_r0, b_c0, b_r1, b_c1 = b_bbox

    dr_min = int(a_r0 - b_r1 - (MAX_SIDE - 1))
    dr_max = int(a_r1 - b_r0 + (MAX_SIDE - 1))
    dc_min = int(a_c0 - b_c1 - (MAX_SIDE - 1))
    dc_max = int(a_c1 - b_c0 + (MAX_SIDE - 1))

    out: list[dict] = []
    for dr in range(dr_min, dr_max + 1):
        for dc in range(dc_min, dc_max + 1):
            cand = _validate_alignment(frag_a, frag_b, bg=bg, dr=dr, dc=dc)
            if cand is None:
                continue
            out.append(cand)
            if len(out) >= int(stop_after):
                return out
    return out


def _render_alignment(candidate: dict, *, bg: int) -> np.ndarray:
    x2 = int(candidate["x2"])
    full_map = candidate["full_map"]
    filled: dict[tuple[int, int], int] = {}
    for (r, c), color in full_map.items():
        if int(color) == int(PLACEHOLDER):
            partner = (int(r), int(x2 - c))
            filled[(int(r), int(c))] = int(full_map[partner])
        else:
            filled[(int(r), int(c))] = int(color)
    return _tight_grid_from_map(filled, bg)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Translate one fragment so the two fragments form a unique left-right mirror whole, then repair every green placeholder from its mirror mate.

    Input structure:
    - The dominant color is background.
    - Exactly two separated multicolor fragments appear on the grid.
    - Some fragment pixels are green placeholders; green is not part of the final sprite.
    - Ignoring green, neither fragment is self-symmetric, but one unique relative translation makes the combined non-green pixels left-right mirror symmetric.

    Rule:
    - Keep one fragment fixed and translate the other without rotating or flipping.
    - Choose the unique translation whose combined non-green pixels are horizontally mirror symmetric.
    - After alignment, replace each green pixel by the color at its horizontal mirror position.
    - Return the repaired combined sprite as a tight crop.

    Key insight:
    The docking location is not marked directly; horizontal symmetry of the combined non-green pattern is the only alignment cue, and the placeholder colors become recoverable only after that alignment is found.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _find_components(g, bg=bg, connectivity=8)
    assert len(comps) == 2, "expected exactly two foreground fragments"

    frag_a = _map_from_component(g, comps[0])
    frag_b = _map_from_component(g, comps[1])
    candidates = _find_alignments(frag_a, frag_b, bg=bg, stop_after=2)
    assert len(candidates) == 1, "expected a unique mirror-symmetric alignment"
    return _render_alignment(candidates[0], bg=bg)


def _random_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    start = (random.randrange(h), random.randrange(w))
    cells = {start}
    frontier = set()

    def add_frontier(cell: tuple[int, int]) -> None:
        r, c = cell
        for nr, nc in _neighbors(int(r), int(c), 4):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((int(nr), int(nc)))

    add_frontier(start)
    while len(cells) < int(area) and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        if cell in cells:
            continue
        cells.add(cell)
        add_frontier(cell)

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[int(r), int(c)] = True
    return mask


def _gen_left_half() -> np.ndarray | None:
    for _ in range(100):
        h = random.randint(5, 9)
        w = random.randint(3, 6)
        area_lo = max(h + 1, w + 2, (h * w + 2) // 3)
        area_hi = min(h * w - 1, int(h * w * 0.78))
        if area_lo > area_hi:
            continue
        mask = _random_connected_mask(h, w, random.randint(area_lo, area_hi))
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        if not coords:
            continue
        if _bbox_coords(coords) != (0, 0, h - 1, w - 1):
            continue
        right_rows = [int(r) for r in range(h) if bool(mask[int(r), w - 1])]
        if len(right_rows) < 2:
            continue
        if _is_horizontally_symmetric({(r, c): 1 for r, c in coords}):
            continue
        return mask
    return None


def _colorize_left_half(
    mask: np.ndarray,
    *,
    bg: int,
) -> tuple[dict[tuple[int, int], int], list[int]] | None:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    if len(coords) < 6:
        return None
    usable = [int(color) for color in GEN_COLOR_POOL if int(color) != int(bg)]
    palette_size = random.randint(3, min(5, len(usable)))
    palette = random.sample(usable, palette_size)
    random.shuffle(coords)
    color_map: dict[tuple[int, int], int] = {}
    for idx, coord in enumerate(coords):
        if idx < len(palette):
            color_map[coord] = int(palette[idx])
        else:
            color_map[coord] = int(random.choice(palette))
    return color_map, palette


def _build_solved_example(
    *,
    bg: int,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int], np.ndarray] | None:
    left_mask = _gen_left_half()
    if left_mask is None:
        return None
    h, w = left_mask.shape
    colored = _colorize_left_half(left_mask, bg=bg)
    if colored is None:
        return None
    left_map, palette = colored
    right_rows = [int(r) for r in range(h) if bool(left_mask[int(r), w - 1])]
    axis_rows = random.sample(right_rows, random.randint(1, min(3, len(right_rows))))
    axis_map: dict[tuple[int, int], int] = {}
    for r in axis_rows:
        left_edge_color = int(left_map[(int(r), w - 1)])
        axis_choices = [int(col) for col in palette if int(col) != int(left_edge_color)]
        if not axis_choices:
            return None
        axis_map[(int(r), int(w))] = int(random.choice(axis_choices))

    full_map = dict(left_map)
    full_map.update(axis_map)
    for (r, c), color in left_map.items():
        full_map[(int(r), int(2 * w - c))] = int(color)

    axis_owner = random.choice(["left", "right"])
    frag_a: dict[tuple[int, int], int] = {}
    frag_b: dict[tuple[int, int], int] = {}
    for pos, color in full_map.items():
        _r, c = pos
        if int(c) < int(w):
            frag_a[pos] = int(color)
        elif int(c) > int(w):
            frag_b[pos] = int(color)
        elif axis_owner == "left":
            frag_a[pos] = int(color)
        else:
            frag_b[pos] = int(color)

    if not _is_connected(frag_a, connectivity=4) or not _is_connected(frag_b, connectivity=4):
        return None
    expected = _tight_grid_from_map(full_map, bg=bg)
    return frag_a, frag_b, expected


def _add_placeholders(
    frag_a: dict[tuple[int, int], int],
    frag_b: dict[tuple[int, int], int],
    expected: np.ndarray,
) -> tuple[dict[tuple[int, int], int], dict[tuple[int, int], int]] | None:
    width = int(expected.shape[1])
    axis_col = int(width // 2)
    off_axis_a = [pos for pos in frag_a if int(pos[1]) != int(axis_col)]
    off_axis_b = [pos for pos in frag_b if int(pos[1]) != int(axis_col)]
    if not off_axis_a or not off_axis_b:
        return None

    out_a = dict(frag_a)
    out_b = dict(frag_b)
    count_a = random.randint(1, min(3, max(1, len(off_axis_a) // 2)))
    count_b = random.randint(1, min(3, max(1, len(off_axis_b) // 2)))
    for pos in random.sample(off_axis_a, count_a):
        out_a[pos] = int(PLACEHOLDER)
    for pos in random.sample(off_axis_b, count_b):
        out_b[pos] = int(PLACEHOLDER)
    return out_a, out_b


def _patch_from_map(color_map: dict[tuple[int, int], int], *, bg: int) -> np.ndarray:
    return _tight_grid_from_map(color_map, bg)


def _place_two_patches(
    patch_a: np.ndarray,
    patch_b: np.ndarray,
    *,
    bg: int,
) -> np.ndarray | None:
    ha, wa = patch_a.shape
    hb, wb = patch_b.shape
    orient = random.choice(["horizontal", "vertical"])

    if orient == "horizontal":
        min_h = max(ha, hb, 12)
        max_h = min(30, min_h + 10)
        min_w = max(wa + wb + 5, 16)
        max_w = min(30, min_w + 10)
        if min_h > max_h or min_w > max_w:
            return None
        H = random.randint(min_h, max_h)
        W = random.randint(min_w, max_w)
        left_first = random.choice([True, False])
        left_patch = patch_a if left_first else patch_b
        right_patch = patch_b if left_first else patch_a
        hl, wl = left_patch.shape
        hr, wr = right_patch.shape
        c_left = random.randint(1, max(1, W - (wl + wr + 3)))
        c_right = random.randint(c_left + wl + 2, W - wr - 1)
        r_left = random.randint(1, H - hl - 1)
        r_right = random.randint(1, H - hr - 1)
        grid = np.full((H, W), int(bg), dtype=int)
        grid[int(r_left) : int(r_left) + hl, int(c_left) : int(c_left) + wl] = left_patch
        grid[int(r_right) : int(r_right) + hr, int(c_right) : int(c_right) + wr] = right_patch
        return grid

    min_h = max(ha + hb + 5, 16)
    max_h = min(30, min_h + 10)
    min_w = max(wa, wb, 12)
    max_w = min(30, min_w + 10)
    if min_h > max_h or min_w > max_w:
        return None
    H = random.randint(min_h, max_h)
    W = random.randint(min_w, max_w)
    top_first = random.choice([True, False])
    top_patch = patch_a if top_first else patch_b
    bot_patch = patch_b if top_first else patch_a
    ht, wt = top_patch.shape
    hb2, wb2 = bot_patch.shape
    r_top = random.randint(1, max(1, H - (ht + hb2 + 3)))
    r_bot = random.randint(r_top + ht + 2, H - hb2 - 1)
    c_top = random.randint(1, W - wt - 1)
    c_bot = random.randint(1, W - wb2 - 1)
    grid = np.full((H, W), int(bg), dtype=int)
    grid[int(r_top) : int(r_top) + ht, int(c_top) : int(c_top) + wt] = top_patch
    grid[int(r_bot) : int(r_bot) + hb2, int(c_bot) : int(c_bot) + wb2] = bot_patch
    return grid


def _semantic_checks(grid: np.ndarray, expected: np.ndarray, *, bg: int) -> bool:
    comps = _find_components(grid, bg=bg, connectivity=8)
    if len(comps) != 2:
        return False
    frag_a = _map_from_component(grid, comps[0])
    frag_b = _map_from_component(grid, comps[1])
    if sum(int(v == PLACEHOLDER) for v in frag_a.values()) == 0:
        return False
    if sum(int(v == PLACEHOLDER) for v in frag_b.values()) == 0:
        return False
    non_green_a = {pos: col for pos, col in frag_a.items() if int(col) != int(PLACEHOLDER)}
    non_green_b = {pos: col for pos, col in frag_b.items() if int(col) != int(PLACEHOLDER)}
    if _is_horizontally_symmetric(non_green_a) or _is_horizontally_symmetric(non_green_b):
        return False
    aligns = _find_alignments(frag_a, frag_b, bg=bg, stop_after=2)
    if len(aligns) != 1:
        return False
    if not np.array_equal(_render_alignment(aligns[0], bg=bg), expected):
        return False
    out = main(grid)
    if not np.array_equal(out, expected):
        return False
    solved_comps = _find_components(expected, bg=bg, connectivity=4)
    return len(solved_comps) == 1


def _generate_one() -> np.ndarray | None:
    bg = random.choice(GEN_BG_CHOICES)
    built = _build_solved_example(bg=bg)
    if built is None:
        return None
    frag_a, frag_b, expected = built
    with_green = _add_placeholders(frag_a, frag_b, expected)
    if with_green is None:
        return None
    frag_a, frag_b = with_green

    patch_a = _patch_from_map(frag_a, bg=bg)
    patch_b = _patch_from_map(frag_b, bg=bg)

    grid = _place_two_patches(patch_a, patch_b, bg=bg)
    if grid is None:
        return None
    if max(grid.shape) > MAX_SIDE or max(expected.shape) > MAX_SIDE:
        return None
    if not _semantic_checks(grid, expected, bg=bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_TRIES):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed")
