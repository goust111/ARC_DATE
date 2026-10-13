import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

# Generator configuration.
GRID_H_RANGE = (28, 36)
GRID_W_RANGE = (28, 36)
REGION_COUNT_CHOICES = (2, 3, 4)
EDGE_BAND = 4
REGION_GAP = 2
THICKNESS_CHOICES = (2, 3)
CAVITY_H_RANGE = (5, 8)
CAVITY_W_RANGE = (5, 8)
GEN_ATTEMPTS = 400
REGION_PLACE_TRIES = 120
KEY_PLACE_TRIES = 160


def _majority_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((int(cr), int(cc)))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            parts.append(pts)
    return parts


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in points]
    cs = [c for _, c in points]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize_mask(points: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(points)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in points:
        mask[r - r0, c - c0] = True
    return mask


def _touches_edge(points: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in points)


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _connected_mask(mask: np.ndarray) -> bool:
    pts = list(map(tuple, np.argwhere(mask)))
    if not pts:
        return False
    need = set((int(r), int(c)) for r, c in pts)
    start = next(iter(need))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) in need and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == len(need)


def _build_key_shapes() -> list[np.ndarray]:
    shapes: dict[tuple[tuple[int, ...], ...], np.ndarray] = {}
    for bits in range(1, 1 << 9):
        raw = np.zeros((3, 3), dtype=bool)
        for idx in range(9):
            if bits & (1 << idx):
                raw[idx // 3, idx % 3] = True
        if not _connected_mask(raw):
            continue
        coords = np.argwhere(raw)
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        cropped = raw[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        shapes.setdefault(_mask_signature(cropped), cropped.copy())
    return list(shapes.values())


KEY_SHAPES = _build_key_shapes()


def _nearest_side(marker: tuple[int, int], inner_bbox: tuple[int, int, int, int]) -> str:
    mr, mc = marker
    ir0, ic0, ir1, ic1 = inner_bbox
    dists = {
        "top": mr - ir0,
        "bottom": ir1 - mr,
        "left": mc - ic0,
        "right": ic1 - mc,
    }
    return min(SIDES, key=lambda side: (dists[side], SIDES.index(side)))


def _find_regions(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    regions: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for pts in _components(grid == color):
            r0, c0, r1, c1 = _bbox(pts)
            bh = r1 - r0 + 1
            bw = c1 - c0 + 1
            if bh < 5 or bw < 5:
                continue
            if _touches_edge(pts, h, w):
                continue
            sub = (grid[r0 : r1 + 1, c0 : c1 + 1] == color)
            max_t = min(3, bh // 2, bw // 2)
            for t in range(1, max_t + 1):
                if bh - 2 * t < 1 or bw - 2 * t < 1:
                    continue
                frame = np.zeros((bh, bw), dtype=bool)
                frame[:t, :] = True
                frame[-t:, :] = True
                frame[:, :t] = True
                frame[:, -t:] = True
                if not np.all(sub[frame]):
                    continue
                if np.any(sub[~frame]):
                    continue
                inner = grid[r0 + t : r1 - t + 1, c0 + t : c1 - t + 1]
                non_bg = np.argwhere(inner != bg)
                if len(non_bg) != 1:
                    continue
                mr_rel, mc_rel = map(int, non_bg[0])
                marker_color = int(inner[mr_rel, mc_rel])
                if marker_color in (bg, color):
                    continue
                marker = (r0 + t + mr_rel, c0 + t + mc_rel)
                regions.append(
                    {
                        "region_color": int(color),
                        "outer_bbox": (r0, c0, r1, c1),
                        "inner_bbox": (r0 + t, c0 + t, r1 - t, c1 - t),
                        "thickness": int(t),
                        "marker_pos": marker,
                        "marker_color": marker_color,
                    }
                )
                break
    regions.sort(key=lambda item: item["outer_bbox"])
    return regions


def _find_key_component(
    grid: np.ndarray,
    color: int,
    marker_pos: tuple[int, int],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    h, w = grid.shape
    candidates = []
    for pts in _components(grid == color):
        if marker_pos in set(pts):
            continue
        mask = _normalize_mask(pts)
        if mask.shape[0] > 3 or mask.shape[1] > 3:
            continue
        if not _touches_edge(pts, h, w):
            continue
        candidates.append((pts, mask))
    if len(candidates) != 1:
        raise ValueError(f"expected exactly one edge key for color {color}, found {len(candidates)}")
    return candidates[0]


def _analyze(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("grid must be 2D")
    bg = _majority_color(g)
    regions = _find_regions(g, bg)
    if not regions:
        raise ValueError("no valid closed regions found")

    marker_colors = [reg["marker_color"] for reg in regions]
    if len(set(marker_colors)) != len(marker_colors):
        raise ValueError("marker colors must be unique")
    region_colors = [reg["region_color"] for reg in regions]
    if len(set(region_colors)) != len(region_colors):
        raise ValueError("region colors must be unique")

    for reg in regions:
        key_points, key_mask = _find_key_component(g, reg["marker_color"], reg["marker_pos"])
        reg["key_points"] = key_points
        reg["key_mask"] = key_mask
        reg["side"] = _nearest_side(reg["marker_pos"], reg["inner_bbox"])
    return bg, regions


def _stamp_hole(out: np.ndarray, bg: int, region: dict) -> None:
    r0, c0, r1, c1 = region["outer_bbox"]
    ir0, ic0, ir1, ic1 = region["inner_bbox"]
    mr, mc = region["marker_pos"]
    key = region["key_mask"]
    kh, kw = key.shape
    side = region["side"]

    if side == "top":
        base_r = ir0 - kh
        base_c = mc - kw // 2
    elif side == "bottom":
        base_r = ir1 + 1
        base_c = mc - kw // 2
    elif side == "left":
        base_r = mr - kh // 2
        base_c = ic0 - kw
    else:
        base_r = mr - kh // 2
        base_c = ic1 + 1

    for rr, cc in np.argwhere(key):
        tr = int(base_r + rr)
        tc = int(base_c + cc)
        if r0 <= tr <= r1 and c0 <= tc <= c1:
            out[tr, tc] = bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each cavity marker to the same-colored edge key, erase the key and marker,
    then carve the key's shape into the nearest cavity wall.
    """
    g = np.asarray(grid, dtype=int)
    bg, regions = _analyze(g)

    out = g.copy()
    for reg in regions:
        mr, mc = reg["marker_pos"]
        out[mr, mc] = bg
        for r, c in reg["key_points"]:
            out[r, c] = bg
        _stamp_hole(out, bg, reg)
    return out


def _random_shape(rng: random.Random, side: str, thickness: int) -> np.ndarray:
    candidates = []
    for shape in KEY_SHAPES:
        kh, kw = shape.shape
        if side in ("top", "bottom"):
            if kh <= thickness and kw % 2 == 1:
                candidates.append(shape)
        else:
            if kw <= thickness and kh % 2 == 1:
                candidates.append(shape)
    return rng.choice(candidates).copy()


def _marker_candidates(
    cavity_h: int,
    cavity_w: int,
    side: str,
    key_mask: np.ndarray,
) -> list[tuple[int, int]]:
    kh, kw = key_mask.shape
    inner_bbox = (0, 0, cavity_h - 1, cavity_w - 1)
    candidates: list[tuple[int, int]] = []
    for r in range(cavity_h):
        for c in range(cavity_w):
            if side in ("top", "bottom"):
                left = c - kw // 2
                if left < 0 or left + kw > cavity_w:
                    continue
            else:
                top = r - kh // 2
                if top < 0 or top + kh > cavity_h:
                    continue
            here = (r, c)
            dists = {
                "top": r - inner_bbox[0],
                "bottom": inner_bbox[2] - r,
                "left": c - inner_bbox[1],
                "right": inner_bbox[3] - c,
            }
            if all(dists[side] < dists[other] for other in SIDES if other != side):
                candidates.append(here)
    return candidates


def _make_region_specs(rng: random.Random, count: int, colors: list[int]) -> list[dict] | None:
    region_colors = rng.sample(colors, count)
    remaining = [c for c in colors if c not in region_colors]
    marker_colors = rng.sample(remaining, count)

    region_sides = list(rng.sample(SIDES, min(count, 4)))
    while len(region_sides) < count:
        region_sides.append(rng.choice(SIDES))
    rng.shuffle(region_sides)

    key_edge_sides = list(rng.sample(SIDES, min(count, 4)))
    while len(key_edge_sides) < count:
        key_edge_sides.append(rng.choice(SIDES))
    rng.shuffle(key_edge_sides)

    specs: list[dict] = []
    for idx in range(count):
        side = region_sides[idx]
        for _ in range(80):
            thickness = rng.choice(THICKNESS_CHOICES)
            cavity_h = rng.randint(CAVITY_H_RANGE[0], CAVITY_H_RANGE[1])
            cavity_w = rng.randint(CAVITY_W_RANGE[0], CAVITY_W_RANGE[1])
            key_mask = _random_shape(rng, side, thickness)
            kh, kw = key_mask.shape
            if kh > cavity_h or kw > cavity_w:
                continue
            marker_choices = _marker_candidates(cavity_h, cavity_w, side, key_mask)
            if not marker_choices:
                continue
            marker_rel = rng.choice(marker_choices)
            specs.append(
                {
                    "region_color": region_colors[idx],
                    "marker_color": marker_colors[idx],
                    "side": side,
                    "edge_side": key_edge_sides[idx],
                    "thickness": thickness,
                    "cavity_h": cavity_h,
                    "cavity_w": cavity_w,
                    "outer_h": cavity_h + 2 * thickness,
                    "outer_w": cavity_w + 2 * thickness,
                    "marker_rel": marker_rel,
                    "key_mask": key_mask,
                }
            )
            break
        else:
            return None
    return specs


def _place_regions(rng: random.Random, h: int, w: int, specs: list[dict]) -> bool:
    blocked = np.zeros((h, w), dtype=bool)
    order = sorted(range(len(specs)), key=lambda i: specs[i]["outer_h"] * specs[i]["outer_w"], reverse=True)
    for idx in order:
        spec = specs[idx]
        oh = spec["outer_h"]
        ow = spec["outer_w"]
        max_r0 = h - EDGE_BAND - oh
        max_c0 = w - EDGE_BAND - ow
        if max_r0 < EDGE_BAND or max_c0 < EDGE_BAND:
            return False
        for _ in range(REGION_PLACE_TRIES):
            r0 = rng.randint(EDGE_BAND, max_r0)
            c0 = rng.randint(EDGE_BAND, max_c0)
            r1 = r0 + oh - 1
            c1 = c0 + ow - 1
            rr0 = max(0, r0 - REGION_GAP)
            cc0 = max(0, c0 - REGION_GAP)
            rr1 = min(h - 1, r1 + REGION_GAP)
            cc1 = min(w - 1, c1 + REGION_GAP)
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            blocked[r0 : r1 + 1, c0 : c1 + 1] = True
            spec["outer_bbox"] = (r0, c0, r1, c1)
            spec["inner_bbox"] = (
                r0 + spec["thickness"],
                c0 + spec["thickness"],
                r1 - spec["thickness"],
                c1 - spec["thickness"],
            )
            mr = r0 + spec["thickness"] + spec["marker_rel"][0]
            mc = c0 + spec["thickness"] + spec["marker_rel"][1]
            spec["marker_pos"] = (mr, mc)
            break
        else:
            return False
    return True


def _key_origin(edge_side: str, h: int, w: int, key_mask: np.ndarray, along: int) -> tuple[int, int]:
    kh, kw = key_mask.shape
    if edge_side == "top":
        return 0, along
    if edge_side == "bottom":
        return h - kh, along
    if edge_side == "left":
        return along, 0
    return along, w - kw


def _place_keys(rng: random.Random, grid: np.ndarray, specs: list[dict], bg: int) -> bool:
    h, w = grid.shape
    blocked = grid != bg
    for spec in specs:
        key_mask = spec["key_mask"]
        kh, kw = key_mask.shape
        edge_side = spec["edge_side"]
        if edge_side in ("top", "bottom"):
            coords = list(range(0, w - kw + 1))
        else:
            coords = list(range(0, h - kh + 1))
        rng.shuffle(coords)
        placed = False
        for along in coords[:KEY_PLACE_TRIES]:
            r0, c0 = _key_origin(edge_side, h, w, key_mask, along)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(h - 1, r0 + kh)
            cc1 = min(w - 1, c0 + kw)
            if blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue
            for rr, cc in np.argwhere(key_mask):
                grid[r0 + rr, c0 + cc] = spec["marker_color"]
            blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break
        if not placed:
            return False
    return True


def _paint_regions(grid: np.ndarray, specs: list[dict], bg: int) -> None:
    for spec in specs:
        r0, c0, r1, c1 = spec["outer_bbox"]
        ir0, ic0, ir1, ic1 = spec["inner_bbox"]
        grid[r0 : r1 + 1, c0 : c1 + 1] = spec["region_color"]
        grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = bg
        mr, mc = spec["marker_pos"]
        grid[mr, mc] = spec["marker_color"]


def _valid_grid(grid: np.ndarray, expected_regions: int, bg: int) -> bool:
    if _majority_color(grid) != bg:
        return False
    try:
        found_bg, regions = _analyze(grid)
    except ValueError:
        return False
    if found_bg != bg or len(regions) != expected_regions:
        return False
    out = main(grid)
    if np.array_equal(out, grid):
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_H_RANGE[0], GRID_H_RANGE[1])
    w = rng.randint(GRID_W_RANGE[0], GRID_W_RANGE[1])
    bg = rng.choice(list(range(10)))
    colors = [c for c in range(10) if c != bg]
    count = rng.choice(REGION_COUNT_CHOICES)

    specs = _make_region_specs(rng, count, colors)
    if specs is None:
        return None
    if not _place_regions(rng, h, w, specs):
        return None

    grid = np.full((h, w), bg, dtype=int)
    _paint_regions(grid, specs, bg)
    if not _place_keys(rng, grid, specs, bg):
        return None
    if not _valid_grid(grid, count, bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted attempts")
