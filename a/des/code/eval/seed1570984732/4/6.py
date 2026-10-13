from __future__ import annotations

from collections import Counter, deque
import itertools
import random

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDE_TO_DIR = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
DIR_TO_SIDE = {value: key for key, value in SIDE_TO_DIR.items()}

FENCE_COLOR = 5
GRID_H_RANGE = (14, 28)
GRID_W_RANGE = (14, 28)
DEPTH_CHOICES = (1, 2, 3, 4)
GEN_ATTEMPTS = 400
POSITION_MARGIN = 6


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _touches_border(cells: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in cells)


def _find_interior_and_fence(grid: np.ndarray) -> tuple[set[tuple[int, int]], int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])

    candidates: list[tuple[int, set[tuple[int, int]], int]] = []
    for comp in _components(g == bg):
        if _touches_border(comp, h, w):
            continue
        boundary_colors: set[int] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and int(g[nr, nc]) != bg:
                    boundary_colors.add(int(g[nr, nc]))
        if len(boundary_colors) == 1:
            fence_color = next(iter(boundary_colors))
            candidates.append((len(comp), set(comp), int(fence_color)))

    assert candidates, "expected one enclosed background region"
    _size, interior, fence_color = max(candidates, key=lambda item: item[0])
    return interior, int(fence_color)


def _stripe_mapping(grid: np.ndarray, bg: int, fence_color: int) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    mask = (g != bg) & (g != fence_color)
    mapping: dict[int, int] = {}

    for comp in _components(mask):
        colors = {int(g[r, c]) for r, c in comp}
        if len(colors) != 1:
            continue
        rows = sorted({r for r, _c in comp})
        cols = sorted({c for _r, c in comp})
        if len(rows) == 1:
            length = cols[-1] - cols[0] + 1
            if length != len(comp):
                continue
        elif len(cols) == 1:
            length = rows[-1] - rows[0] + 1
            if length != len(comp):
                continue
        else:
            continue

        if not any(
            0 <= r + dr < h and 0 <= c + dc < w and int(g[r + dr, c + dc]) == fence_color
            for r, c in comp
            for dr, dc in DIR4
        ):
            continue

        color = next(iter(colors))
        mapping[int(length)] = int(color)

    return mapping


def _interior_layers(interior: set[tuple[int, int]], fence_mask: np.ndarray) -> dict[tuple[int, int], int]:
    h, w = fence_mask.shape
    dist: dict[tuple[int, int], int] = {}
    q = deque()
    for r, c in interior:
        if any(
            0 <= r + dr < h and 0 <= c + dc < w and bool(fence_mask[r + dr, c + dc])
            for dr, dc in DIR4
        ):
            dist[(r, c)] = 1
            q.append((r, c))

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in interior and (nr, nc) not in dist:
                dist[(nr, nc)] = dist[(r, c)] + 1
                q.append((nr, nc))

    return dist


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill onion layers inside a closed fence using outside stripe lengths as layer ids.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    interior, fence_color = _find_interior_and_fence(g)
    layer_to_color = _stripe_mapping(g, bg, fence_color)
    fence_mask = g == fence_color
    layers = _interior_layers(interior, fence_mask)

    out = g.copy()
    for (r, c), layer in layers.items():
        if layer in layer_to_color:
            out[r, c] = int(layer_to_color[layer])
    return out


def _dilate(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out = set(cells)
    for r, c in list(cells):
        for dr, dc in DIR4:
            out.add((r + dr, c + dc))
    return out


def _boundary(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r, c in cells
        if any((r + dr, c + dc) not in cells for dr, dc in DIR4)
    }


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    min_r = min(r for r, _c in cells)
    min_c = min(c for _r, c in cells)
    return {(r - min_r, c - min_c) for r, c in cells}


def _translate(cells: set[tuple[int, int]], dr: int, dc: int) -> set[tuple[int, int]]:
    return {(r + dr, c + dc) for r, c in cells}


def _component_count(cells: set[tuple[int, int]]) -> int:
    if not cells:
        return 0
    remaining = set(cells)
    count = 0
    while remaining:
        count += 1
        start = next(iter(remaining))
        q = deque([start])
        remaining.remove(start)
        while q:
            r, c = q.popleft()
            for dr, dc in DIR4:
                nxt = (r + dr, c + dc)
                if nxt in remaining:
                    remaining.remove(nxt)
                    q.append(nxt)
    return count


def _inner_regions_from_fence(fence: set[tuple[int, int]], h: int, w: int) -> list[set[tuple[int, int]]]:
    bg_mask = np.ones((h, w), dtype=bool)
    for r, c in fence:
        bg_mask[r, c] = False
    regions: list[set[tuple[int, int]]] = []
    for comp in _components(bg_mask):
        if not _touches_border(comp, h, w):
            regions.append(set(comp))
    return regions


def _max_layer(interior: set[tuple[int, int]], fence: set[tuple[int, int]], h: int, w: int) -> int:
    fence_mask = np.zeros((h, w), dtype=bool)
    for r, c in fence:
        fence_mask[r, c] = True
    dists = _interior_layers(interior, fence_mask)
    return max(dists.values()) if dists else 0


def _build_solid(depth: int) -> set[tuple[int, int]]:
    base_h = random.randint(2 * depth + 1, 2 * depth + 6)
    base_w = random.randint(2 * depth + 1, 2 * depth + 6)
    cells = {(r, c) for r in range(base_h) for c in range(base_w)}

    for side in ("top", "bottom", "left", "right"):
        if random.random() >= 0.6:
            continue
        if side in ("top", "bottom"):
            thickness = random.randint(1, depth + 2)
            seg_w = random.randint(max(2, base_w // 3), base_w)
            c0 = random.randint(0, base_w - seg_w)
            rows = range(-thickness, 0) if side == "top" else range(base_h, base_h + thickness)
            for r in rows:
                for c in range(c0, c0 + seg_w):
                    cells.add((r, c))
        else:
            thickness = random.randint(1, depth + 2)
            seg_h = random.randint(max(2, base_h // 3), base_h)
            r0 = random.randint(0, base_h - seg_h)
            cols = range(-thickness, 0) if side == "left" else range(base_w, base_w + thickness)
            for r in range(r0, r0 + seg_h):
                for c in cols:
                    cells.add((r, c))

    cells = _normalize(cells)
    height = max(r for r, _c in cells) + 1
    width = max(c for _r, c in cells) + 1

    for side in ("top", "bottom", "left", "right"):
        if random.random() >= 0.55:
            continue
        if side in ("top", "bottom"):
            max_depth = min(depth, height - 2)
            if max_depth < 1:
                continue
            notch_depth = random.randint(1, max_depth)
            notch_width = random.randint(1, max(1, width - 2))
            c0 = random.randint(0, width - notch_width)
            rows = range(0, notch_depth) if side == "top" else range(height - notch_depth, height)
            for r in rows:
                for c in range(c0, c0 + notch_width):
                    cells.discard((r, c))
        else:
            max_depth = min(depth, width - 2)
            if max_depth < 1:
                continue
            notch_depth = random.randint(1, max_depth)
            notch_height = random.randint(1, max(1, height - 2))
            r0 = random.randint(0, height - notch_height)
            cols = range(0, notch_depth) if side == "left" else range(width - notch_depth, width)
            for r in range(r0, r0 + notch_height):
                for c in cols:
                    cells.discard((r, c))

    return cells


def _stripe_candidates(
    fence: set[tuple[int, int]],
    solid: set[tuple[int, int]],
    h: int,
    w: int,
) -> list[tuple[tuple[int, int], tuple[int, int], int]]:
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]] = []
    for r, c in fence:
        outside_dirs = [direction for direction in DIR4 if (r + direction[0], c + direction[1]) not in solid]
        if len(outside_dirs) != 1:
            continue
        direction = outside_dirs[0]
        max_len = 0
        for length in range(1, 5):
            ok = True
            for step in range(1, length + 1):
                rr = r + direction[0] * step
                cc = c + direction[1] * step
                if not (0 <= rr < h and 0 <= cc < w) or (rr, cc) in solid:
                    ok = False
                    break
                if step >= 2 and any((rr + dr, cc + dc) in solid for dr, dc in DIR4):
                    ok = False
                    break
            if ok:
                max_len = length
        if max_len:
            candidates.append(((r, c), direction, int(max_len)))
    random.shuffle(candidates)
    return candidates


def _pick_stripes(
    candidates: list[tuple[tuple[int, int], tuple[int, int], int]],
    depth: int,
) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    by_side = {
        side: [cand for cand in candidates if DIR_TO_SIDE[cand[1]] == side]
        for side in SIDE_TO_DIR
    }
    lengths = list(range(1, depth + 1))
    side_orders = list(itertools.permutations(list(SIDE_TO_DIR.keys()), depth))
    random.shuffle(side_orders)

    for side_order in side_orders:
        used: set[tuple[int, int]] = set()
        picks: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
        ok = True
        for length, side in zip(lengths, side_order, strict=False):
            slot = None
            random.shuffle(by_side[side])
            for anchor, direction, max_len in by_side[side]:
                if max_len < length:
                    continue
                stripe_cells = {
                    (anchor[0] + direction[0] * step, anchor[1] + direction[1] * step)
                    for step in range(1, length + 1)
                }
                if stripe_cells & used:
                    continue
                slot = (length, anchor, direction)
                used |= stripe_cells
                break
            if slot is None:
                ok = False
                break
            picks.append(slot)
        if ok:
            return picks
    raise RuntimeError("failed to choose non-overlapping stripes")


def _render_expected(
    inp: np.ndarray,
    depth_to_color: dict[int, int],
    interior: set[tuple[int, int]],
    fence: set[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    fence_mask = np.zeros(inp.shape, dtype=bool)
    for r, c in fence:
        fence_mask[r, c] = True
    layers = _interior_layers(interior, fence_mask)
    for (r, c), layer in layers.items():
        out[r, c] = int(depth_to_color[layer])
    return out


def _attempt_generate() -> np.ndarray | None:
    depth = random.choice(DEPTH_CHOICES)
    bg_choices = [color for color in range(10) if color != FENCE_COLOR]
    bg = random.choice(bg_choices)

    solid = _build_solid(depth)
    if not solid or _component_count(solid) != 1:
        return None

    solid = _normalize(solid)
    solid_h = max(r for r, _c in solid) + 1
    solid_w = max(c for _r, c in solid) + 1

    min_h = max(GRID_H_RANGE[0], solid_h + 2 * POSITION_MARGIN)
    min_w = max(GRID_W_RANGE[0], solid_w + 2 * POSITION_MARGIN)
    max_h = max(min_h, GRID_H_RANGE[1])
    max_w = max(min_w, GRID_W_RANGE[1])
    if min_h > max_h or min_w > max_w:
        return None
    height = random.randint(min_h, max_h)
    width = random.randint(min_w, max_w)

    top = random.randint(POSITION_MARGIN, height - solid_h - POSITION_MARGIN)
    left = random.randint(POSITION_MARGIN, width - solid_w - POSITION_MARGIN)
    solid = _translate(solid, top, left)
    fence = _boundary(solid)

    inner_regions = _inner_regions_from_fence(fence, height, width)
    if len(inner_regions) != 1:
        return None
    interior = inner_regions[0]
    if interior != solid - fence:
        return None
    if _max_layer(interior, fence, height, width) != depth:
        return None

    candidates = _stripe_candidates(fence, solid, height, width)
    try:
        stripe_plan = _pick_stripes(candidates, depth)
    except RuntimeError:
        return None

    color_pool = [color for color in range(10) if color not in (bg, FENCE_COLOR)]
    random.shuffle(color_pool)
    depth_to_color = {layer: color_pool[layer - 1] for layer in range(1, depth + 1)}

    inp = np.full((height, width), bg, dtype=int)
    for r, c in fence:
        inp[r, c] = FENCE_COLOR

    for length, anchor, direction in stripe_plan:
        color = depth_to_color[length]
        for step in range(1, length + 1):
            rr = anchor[0] + direction[0] * step
            cc = anchor[1] + direction[1] * step
            inp[rr, cc] = int(color)

    expected = _render_expected(inp, depth_to_color, interior, fence)
    if not np.array_equal(main(inp), expected):
        return None
    if Counter(map(int, inp.ravel())).most_common(1)[0][0] != bg:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate puzzle input")
