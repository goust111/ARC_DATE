import random
from collections import deque

import numpy as np


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _color_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(pts, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": _bbox(arr),
                    "area": int(len(arr)),
                }
            )
    return comps


def _component_local_mask(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = comp["coords"] - np.array([r0, c0], dtype=int)
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _enclosed_regions(component_mask: np.ndarray) -> list[np.ndarray]:
    free = ~component_mask
    h, w = free.shape
    seen = np.zeros((h, w), dtype=bool)
    regions: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not free[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts = []
            touches_edge = False
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                if cr == 0 or cr == h - 1 or cc == 0 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if free[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            if not touches_edge:
                regions.append(np.array(pts, dtype=int))
    return regions


def _sample_mapping_entry(
    grid: np.ndarray, comp: dict, bg: int, fence_color: int
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = comp["bbox"]
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    regions = _enclosed_regions(_component_local_mask(comp))
    if len(regions) != 1:
        return None
    pts = regions[0]
    vals = [int(sub[r, c]) for r, c in pts]
    interior_colors = {v for v in vals if v not in (bg, fence_color)}
    if len(interior_colors) != 1:
        return None
    colored_count = sum(v not in (bg, fence_color) for v in vals)
    if colored_count != 1:
        return None
    bg_area = sum(v == bg for v in vals)
    if bg_area <= 0:
        return None
    return int(bg_area), int(next(iter(interior_colors)))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _bg_color(g)
    comps = _color_components(g, bg)
    assert comps

    largest_area = max(comp["area"] for comp in comps)
    largest_candidates = [comp for comp in comps if comp["area"] == largest_area]
    assert len(largest_candidates) == 1
    fence = largest_candidates[0]
    fence_color = int(fence["color"])

    mapping: dict[int, int] = {}
    for comp in comps:
        if comp is fence or int(comp["color"]) != fence_color:
            continue
        entry = _sample_mapping_entry(g, comp, bg, fence_color)
        if entry is None:
            continue
        area, color = entry
        assert area not in mapping or mapping[area] == color
        mapping[area] = color
    assert mapping

    r0, c0, r1, c1 = fence["bbox"]
    sub = g[r0 : r1 + 1, c0 : c1 + 1]
    out = g.copy()
    for region in _enclosed_regions(_component_local_mask(fence)):
        vals = [int(sub[r, c]) for r, c in region]
        if any(v not in (bg, fence_color) for v in vals):
            continue
        room = [(r0 + int(r), c0 + int(c)) for r, c in region if int(sub[r, c]) == bg]
        if not room:
            continue
        color = mapping[len(room)]
        for rr, cc in room:
            out[rr, cc] = color
    return out


_GEN_RETRIES = 400
_GRID_H_RANGE = (20, 30)
_GRID_W_RANGE = (22, 32)
_INNER_H_RANGE = (6, 9)
_INNER_W_RANGE = (7, 11)
_ROOM_COUNT_CHOICES = (3, 4)
_MAX_ROOM_AREA = 24
_PAD = 1


def _random_colors(room_count: int) -> tuple[int, int, list[int]]:
    colors = list(range(10))
    bg = random.choice(colors)
    rest = [c for c in colors if c != bg]
    fence = random.choice(rest)
    fill_colors = random.sample([c for c in rest if c != fence], room_count)
    return int(bg), int(fence), [int(c) for c in fill_colors]


def _room_area(room: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = room
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _paint_walls(
    patch: np.ndarray, walls: list[tuple[str, int, int, int]], fence_color: int
) -> None:
    for orient, axis, start, end in walls:
        if orient == "v":
            patch[start : end + 1, axis] = fence_color
        else:
            patch[axis, start : end + 1] = fence_color


def _three_room_candidates(inner_h: int, inner_w: int):
    candidates = []
    for wall_c in range(2, inner_w):
        for wall_r in range(2, inner_h):
            left = wall_c - 1
            right = inner_w - wall_c
            top = wall_r - 1
            bottom = inner_h - wall_r
            if min(left, right, top, bottom) < 1:
                continue

            rooms = [
                (1, 1, wall_r - 1, wall_c - 1),
                (wall_r + 1, 1, inner_h, wall_c - 1),
                (1, wall_c + 1, inner_h, inner_w),
            ]
            walls = [("v", wall_c, 1, inner_h), ("h", wall_r, 1, wall_c - 1)]
            candidates.append((rooms, walls))

            rooms = [
                (1, 1, inner_h, wall_c - 1),
                (1, wall_c + 1, wall_r - 1, inner_w),
                (wall_r + 1, wall_c + 1, inner_h, inner_w),
            ]
            walls = [("v", wall_c, 1, inner_h), ("h", wall_r, wall_c + 1, inner_w)]
            candidates.append((rooms, walls))

            rooms = [
                (1, 1, wall_r - 1, wall_c - 1),
                (1, wall_c + 1, wall_r - 1, inner_w),
                (wall_r + 1, 1, inner_h, inner_w),
            ]
            walls = [("h", wall_r, 1, inner_w), ("v", wall_c, 1, wall_r - 1)]
            candidates.append((rooms, walls))

            rooms = [
                (1, 1, wall_r - 1, inner_w),
                (wall_r + 1, 1, inner_h, wall_c - 1),
                (wall_r + 1, wall_c + 1, inner_h, inner_w),
            ]
            walls = [("h", wall_r, 1, inner_w), ("v", wall_c, wall_r + 1, inner_h)]
            candidates.append((rooms, walls))
    return candidates


def _four_room_candidates(inner_h: int, inner_w: int):
    candidates = []
    for wall_c in range(2, inner_w):
        for wall_r in range(2, inner_h):
            left = wall_c - 1
            right = inner_w - wall_c
            top = wall_r - 1
            bottom = inner_h - wall_r
            if min(left, right, top, bottom) < 1:
                continue
            rooms = [
                (1, 1, wall_r - 1, wall_c - 1),
                (1, wall_c + 1, wall_r - 1, inner_w),
                (wall_r + 1, 1, inner_h, wall_c - 1),
                (wall_r + 1, wall_c + 1, inner_h, inner_w),
            ]
            walls = [("v", wall_c, 1, inner_h), ("h", wall_r, 1, inner_w)]
            candidates.append((rooms, walls))
    return candidates


def _make_fence_patch(room_count: int, bg: int, fence_color: int):
    for _ in range(120):
        inner_h = random.randint(*_INNER_H_RANGE)
        inner_w = random.randint(*_INNER_W_RANGE)
        if room_count == 3:
            candidates = _three_room_candidates(inner_h, inner_w)
        else:
            candidates = _four_room_candidates(inner_h, inner_w)

        valid = []
        for rooms, walls in candidates:
            areas = [_room_area(room) for room in rooms]
            if len(set(areas)) != room_count or max(areas) > _MAX_ROOM_AREA:
                continue
            valid.append((rooms, walls, areas))
        if not valid:
            continue

        rooms, walls, areas = random.choice(valid)
        patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
        patch[0, :] = fence_color
        patch[-1, :] = fence_color
        patch[:, 0] = fence_color
        patch[:, -1] = fence_color
        _paint_walls(patch, walls, fence_color)
        return patch, rooms, areas
    return None


def _factor_pairs(n: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, int(n**0.5) + 1):
        if n % h != 0:
            continue
        w = n // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    return pairs


def _make_sample_patch(area: int, color: int, bg: int, fence_color: int) -> np.ndarray:
    total = area + 1
    pairs = _factor_pairs(total)
    pairs.sort(key=lambda hw: (abs(hw[0] - hw[1]), hw[0] + hw[1]))
    top = pairs[: min(3, len(pairs))]
    inner_h, inner_w = random.choice(top)
    patch = np.full((inner_h + 2, inner_w + 2), bg, dtype=int)
    patch[0, :] = fence_color
    patch[-1, :] = fence_color
    patch[:, 0] = fence_color
    patch[:, -1] = fence_color
    pr = random.randint(1, inner_h)
    pc = random.randint(1, inner_w)
    patch[pr, pc] = color
    return patch


_SPRITE_SHAPES = (
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
)


def _rotate_bool(mask: np.ndarray) -> np.ndarray:
    return np.rot90(mask, random.randint(0, 3))


def _make_sprite_patch(color: int, bg: int) -> np.ndarray:
    mask = _rotate_bool(random.choice(_SPRITE_SHAPES))
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = color
    return patch


def _fits(occ: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int) -> bool:
    H, W = occ.shape
    r1 = r0 + h
    c1 = c0 + w
    if r0 < 1 or c0 < 1 or r1 > H - 1 or c1 > W - 1:
        return False
    rs = max(0, r0 - pad)
    cs = max(0, c0 - pad)
    re = min(H, r1 + pad)
    ce = min(W, c1 + pad)
    return not occ[rs:re, cs:ce].any()


def _place_rect_patch(
    grid: np.ndarray, occ: np.ndarray, patch: np.ndarray, pad: int
) -> tuple[int, int] | None:
    H, W = grid.shape
    ph, pw = patch.shape
    positions = []
    for r0 in range(1, H - ph):
        for c0 in range(1, W - pw):
            if _fits(occ, r0, c0, ph, pw, pad):
                positions.append((r0, c0))
    if not positions:
        return None
    r0, c0 = random.choice(positions)
    grid[r0 : r0 + ph, c0 : c0 + pw] = patch
    occ[r0 : r0 + ph, c0 : c0 + pw] = True
    return int(r0), int(c0)


def _sample_grid_dims(patches: list[np.ndarray]) -> tuple[int, int] | None:
    min_h = max(p.shape[0] for p in patches) + 2
    min_w = max(p.shape[1] for p in patches) + 2
    H = random.randint(max(_GRID_H_RANGE[0], min_h), _GRID_H_RANGE[1])
    W = random.randint(max(_GRID_W_RANGE[0], min_w), _GRID_W_RANGE[1])
    total_box_area = sum((p.shape[0] + 2 * _PAD) * (p.shape[1] + 2 * _PAD) for p in patches)
    if H * W < total_box_area:
        return None
    return H, W


def _build_expected(
    inp: np.ndarray,
    fence_pos: tuple[int, int],
    rooms: list[tuple[int, int, int, int]],
    room_colors: list[int],
) -> np.ndarray:
    out = inp.copy()
    base_r, base_c = fence_pos
    for (r0, c0, r1, c1), color in zip(rooms, room_colors):
        out[base_r + r0 : base_r + r1 + 1, base_c + c0 : base_c + c1 + 1] = color
    return out


def _coord_in_component(comp: dict, coord: tuple[int, int]) -> bool:
    target = np.array(coord, dtype=int)
    return bool(np.any(np.all(comp["coords"] == target, axis=1)))


def _gen_attempt() -> np.ndarray | None:
    room_count = random.choice(_ROOM_COUNT_CHOICES)
    bg, fence_color, room_colors = _random_colors(room_count)

    fence_data = _make_fence_patch(room_count, bg, fence_color)
    if fence_data is None:
        return None
    fence_patch, rooms, areas = fence_data

    room_color_by_area = {int(area): int(color) for area, color in zip(areas, room_colors)}
    room_colors = [room_color_by_area[int(area)] for area in areas]

    legend_items = list(room_color_by_area.items())
    random.shuffle(legend_items)

    sample_patches = [_make_sample_patch(area, color, bg, fence_color) for area, color in legend_items]
    sprite_patches = [_make_sprite_patch(color, bg) for color in room_colors]
    all_patches = [fence_patch, *sample_patches, *sprite_patches]

    dims = _sample_grid_dims(all_patches)
    if dims is None:
        return None
    H, W = dims
    inp = np.full((H, W), bg, dtype=int)
    occ = np.zeros((H, W), dtype=bool)

    fence_pos = _place_rect_patch(inp, occ, fence_patch, pad=_PAD)
    if fence_pos is None:
        return None

    place_order = sample_patches + sprite_patches
    place_order.sort(key=lambda p: p.shape[0] * p.shape[1], reverse=True)
    for patch in place_order:
        if _place_rect_patch(inp, occ, patch, pad=_PAD) is None:
            return None

    if _bg_color(inp) != bg:
        return None

    comps = _color_components(inp, bg)
    if not comps:
        return None
    intended = None
    for comp in comps:
        if int(comp["color"]) == fence_color and _coord_in_component(comp, fence_pos):
            intended = comp
            break
    if intended is None:
        return None
    largest_area = max(comp["area"] for comp in comps)
    if intended["area"] != largest_area:
        return None
    if sum(comp["area"] == largest_area for comp in comps) != 1:
        return None

    expected = _build_expected(inp, fence_pos, rooms, room_colors)
    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
