from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)

GEN_RETRIES = 300
PLACEMENT_RETRIES = 200


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components_mask(mask: np.ndarray, connectivity: int = 4) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    if connectivity == 4:
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        steps = [
            (-1, 0),
            (1, 0),
            (0, -1),
            (0, 1),
            (-1, -1),
            (-1, 1),
            (1, -1),
            (1, 1),
        ]
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in steps:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _touches_boundary(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _extract_templates(g: np.ndarray) -> tuple[dict[int, np.ndarray], np.ndarray]:
    h, w = g.shape
    bbox_mask = np.zeros((h, w), dtype=bool)
    templates: dict[int, np.ndarray] = {}
    for color in map(int, np.unique(g)):
        if color in (BLACK, GREY):
            continue
        edge_comps = [comp for comp in _components_mask(g == color) if _touches_boundary(comp, h, w)]
        if not edge_comps:
            continue
        assert len(edge_comps) == 1, "expected one edge template per color"
        comp = edge_comps[0]
        r0, c0, r1, c1 = _bbox(comp)
        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all((sub == BLACK) | (sub == color)), "template bbox must contain only color and black"
        assert np.all(sub[0, :] == color) and np.all(sub[-1, :] == color), "template top/bottom border must be colored"
        assert np.all(sub[:, 0] == color) and np.all(sub[:, -1] == color), "template left/right border must be colored"
        pattern = sub == BLACK
        assert pattern.any(), "template must contain a visible black texture"
        templates[color] = pattern
        bbox_mask[r0 : r1 + 1, c0 : c1 + 1] = True
    assert templates, "expected at least one edge template"
    return templates, bbox_mask


def _fill_region(out: np.ndarray, coords: list[tuple[int, int]], color: int, pattern: np.ndarray) -> None:
    r0, c0, _r1, _c1 = _bbox(coords)
    ph, pw = pattern.shape
    for r, c in coords:
        if pattern[(r - r0) % ph, (c - c0) % pw]:
            out[r, c] = BLACK
        else:
            out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use the seed color inside each enclosed region to select the same-colored border template,
    then tile that template's black texture across the region.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    templates, template_bbox_mask = _extract_templates(g)
    region_mask = (g != GREY) & (~template_bbox_mask)

    out = g.copy()
    for comp in _components_mask(region_mask):
        if _touches_boundary(comp, h, w):
            continue
        colors = {int(g[r, c]) for r, c in comp if int(g[r, c]) != BLACK}
        if not colors:
            continue
        assert len(colors) == 1, "each room must contain exactly one seed color"
        color = colors.pop()
        assert color in templates, "every seed color must have a matching template"
        _fill_region(out, comp, color, templates[color])
    return out


def _room_boundary(region_mask: np.ndarray) -> np.ndarray:
    ch, cw = region_mask.shape
    boundary = np.zeros((ch + 2, cw + 2), dtype=bool)
    room = {(r + 1, c + 1) for r, c in zip(*np.where(region_mask), strict=False)}
    for r, c in room:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) not in room:
                boundary[nr, nc] = True
    boundary[[r for r, _c in room], [c for _r, c in room]] = False
    return boundary


def _grow_blob(height: int, width: int, target: int) -> np.ndarray | None:
    start = (random.randrange(height), random.randrange(width))
    blob = {start}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in blob:
                frontier.add((nr, nc))

    add_frontier(*start)
    while len(blob) < target and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        blob.add(cell)
        add_frontier(*cell)
    if len(blob) != target:
        return None
    mask = np.zeros((height, width), dtype=bool)
    for r, c in blob:
        mask[r, c] = True
    return mask


def _pick_seed_cells(region_mask: np.ndarray, size: int) -> list[tuple[int, int]] | None:
    cells = [(int(r), int(c)) for r, c in np.argwhere(region_mask)]
    region = set(cells)
    for _ in range(50):
        start = random.choice(cells)
        chosen = [start]
        chosen_set = {start}
        frontier = [start]
        while len(chosen) < size and frontier:
            r, c = random.choice(frontier)
            nbrs = [
                (nr, nc)
                for nr, nc in _neighbors4(r, c)
                if (nr, nc) in region and (nr, nc) not in chosen_set
            ]
            if not nbrs:
                frontier.remove((r, c))
                continue
            nxt = random.choice(nbrs)
            chosen.append(nxt)
            chosen_set.add(nxt)
            frontier.append(nxt)
        if len(chosen) == size:
            return chosen
    return None


def _make_room_patch(color: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    for _ in range(200):
        core_h = random.randint(4, 8)
        core_w = random.randint(4, 8)
        area_min = max(6, (core_h * core_w) // 3)
        area_max = max(area_min, (core_h * core_w * 3) // 4)
        target = random.randint(area_min, area_max)
        region_mask = _grow_blob(core_h, core_w, target)
        if region_mask is None:
            continue
        boundary = _room_boundary(region_mask)
        if len(_components_mask(boundary, connectivity=8)) != 1:
            continue
        seed_size = random.randint(2, 3)
        seed = _pick_seed_cells(region_mask, seed_size)
        if seed is None:
            continue

        patch = np.full((core_h + 2, core_w + 2), BLACK, dtype=int)
        patch[boundary] = GREY
        region_coords = [(r + 1, c + 1) for r, c in np.argwhere(region_mask)]
        for r, c in region_coords:
            patch[r, c] = BLACK
        for r, c in seed:
            patch[r + 1, c + 1] = color
        return patch, region_coords
    raise RuntimeError("failed to generate a valid room patch")


def _center_spots(length: int) -> list[int]:
    if length <= 0:
        return []
    if length % 2 == 1:
        return [length // 2]
    return [length // 2 - 1, length // 2]


def _diag_mask(height: int, width: int, reverse: bool = False) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    for r in range(height):
        if height == 1:
            c = width // 2
        else:
            pos = r * (width - 1) / (height - 1)
            c = int(round(pos))
        if reverse:
            c = width - 1 - c
        mask[r, c] = True
    return mask


def _pattern_mask(height: int, width: int) -> np.ndarray:
    inner_h = height - 2
    inner_w = width - 2
    rr, cc = np.indices((inner_h, inner_w))
    variants: list[np.ndarray] = []

    center = np.zeros((inner_h, inner_w), dtype=bool)
    for r in _center_spots(inner_h):
        for c in _center_spots(inner_w):
            center[r, c] = True
    variants.append(center)

    variants.append((rr + cc) % 2 == 0)
    variants.append((rr + cc) % 2 == 1)

    cross = np.zeros((inner_h, inner_w), dtype=bool)
    cross[_center_spots(inner_h), :] = True
    cross[:, _center_spots(inner_w)] = True
    variants.append(cross)

    stripe_h = np.zeros((inner_h, inner_w), dtype=bool)
    stripe_h[_center_spots(inner_h), :] = True
    variants.append(stripe_h)

    stripe_v = np.zeros((inner_h, inner_w), dtype=bool)
    stripe_v[:, _center_spots(inner_w)] = True
    variants.append(stripe_v)

    variants.append(_diag_mask(inner_h, inner_w, reverse=False))
    variants.append(_diag_mask(inner_h, inner_w, reverse=True))

    corners = np.zeros((inner_h, inner_w), dtype=bool)
    corners[0, 0] = True
    corners[0, -1] = True
    corners[-1, 0] = True
    corners[-1, -1] = True
    variants.append(corners)

    choices = [m for m in variants if m.any()]
    inner = random.choice(choices).copy()
    full = np.zeros((height, width), dtype=bool)
    full[1:-1, 1:-1] = inner
    return full


def _make_template(color: int) -> tuple[np.ndarray, np.ndarray]:
    height = random.randint(3, 5)
    width = random.randint(3, 5)
    pattern = _pattern_mask(height, width)
    template = np.full((height, width), color, dtype=int)
    template[pattern] = BLACK
    return template, pattern


def _layout_line(total: int, sizes: list[int], margin: int = 1) -> list[int] | None:
    need = sum(sizes) + (len(sizes) - 1) + 2 * margin
    if need > total:
        return None
    extra = total - need
    slots = [0] * (len(sizes) + 1)
    for _ in range(extra):
        slots[random.randrange(len(slots))] += 1
    starts = []
    pos = margin + slots[0]
    for i, size in enumerate(sizes):
        starts.append(pos)
        pos += size
        if i < len(sizes) - 1:
            pos += 1 + slots[i + 1]
    return starts


def _place_templates(
    grid: np.ndarray,
    side: str,
    templates: list[np.ndarray],
) -> list[tuple[np.ndarray, np.ndarray, int, int]]:
    placed: list[tuple[np.ndarray, np.ndarray, int, int]] = []
    if side == "bottom":
        starts = _layout_line(grid.shape[1], [tpl.shape[1] for tpl in templates])
        if starts is None:
            raise RuntimeError("templates do not fit horizontally")
        for tpl, c0 in zip(templates, starts, strict=False):
            h, _w = tpl.shape
            r0 = grid.shape[0] - h
            grid[r0 : r0 + tpl.shape[0], c0 : c0 + tpl.shape[1]] = tpl
            placed.append((tpl, tpl == BLACK, r0, c0))
    else:
        starts = _layout_line(grid.shape[0], [tpl.shape[0] for tpl in templates])
        if starts is None:
            raise RuntimeError("templates do not fit vertically")
        for tpl, r0 in zip(templates, starts, strict=False):
            _h, w = tpl.shape
            c0 = grid.shape[1] - w
            grid[r0 : r0 + tpl.shape[0], c0 : c0 + tpl.shape[1]] = tpl
            placed.append((tpl, tpl == BLACK, r0, c0))
    return placed


def _can_place(occupied: np.ndarray, r0: int, c0: int, patch: np.ndarray) -> bool:
    h, w = occupied.shape
    ph, pw = patch.shape
    if r0 < 0 or c0 < 0 or r0 + ph > h or c0 + pw > w:
        return False
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h, r0 + ph + 1)
    cc1 = min(w, c0 + pw + 1)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _mark_occupied(occupied: np.ndarray, r0: int, c0: int, patch: np.ndarray) -> None:
    occupied[r0 : r0 + patch.shape[0], c0 : c0 + patch.shape[1]] = True


def _render_expected(
    inp: np.ndarray,
    rooms: list[tuple[list[tuple[int, int]], int]],
    patterns: dict[int, np.ndarray],
) -> np.ndarray:
    out = inp.copy()
    for coords, color in rooms:
        _fill_region(out, coords, color, patterns[color])
    return out


def _gen_attempt() -> np.ndarray | None:
    side = random.choice(["bottom", "right"])
    room_count = random.randint(2, 4)
    colors = random.sample(COLOR_POOL, room_count)

    template_data = {color: _make_template(color) for color in colors}
    template_arrays = [template_data[color][0] for color in colors]
    template_patterns = {color: template_data[color][1] for color in colors}

    size_base = 18 + 2 * max(0, room_count - 2)
    height = random.randint(size_base, 28)
    width = random.randint(size_base, 28)

    if side == "bottom":
        strip_h = max(tpl.shape[0] for tpl in template_arrays)
        if height - strip_h - 3 < 7:
            return None
    else:
        strip_w = max(tpl.shape[1] for tpl in template_arrays)
        if width - strip_w - 3 < 7:
            return None

    grid = np.full((height, width), BLACK, dtype=int)
    try:
        _place_templates(grid, side, template_arrays)
    except RuntimeError:
        return None

    occupied = np.zeros((height, width), dtype=bool)
    if side == "bottom":
        occupied[height - strip_h - 1 :, :] = True
        row_hi = height - strip_h - 2
        col_hi = width - 2
    else:
        occupied[:, width - strip_w - 1 :] = True
        row_hi = height - 2
        col_hi = width - strip_w - 2

    placed_rooms: list[tuple[list[tuple[int, int]], int]] = []
    for color in colors:
        patch, local_region = _make_room_patch(color)
        ph, pw = patch.shape
        if row_hi - ph < 1 or col_hi - pw < 1:
            return None
        candidates = []
        for r0 in range(1, row_hi - ph + 2):
            for c0 in range(1, col_hi - pw + 2):
                if _can_place(occupied, r0, c0, patch):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        _mark_occupied(occupied, r0, c0, patch)
        global_region = [(r0 + r, c0 + c) for r, c in local_region]
        placed_rooms.append((global_region, color))

    expected = _render_expected(grid, placed_rooms, template_patterns)
    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected) and not np.array_equal(grid, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid input")
