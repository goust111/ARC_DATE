import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 300


def _neighbors4(r: int, c: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        yield r + dr, c + dc


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _frame_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    pts = []
    for c in range(c0, c1 + 1):
        pts.append((r0, c))
        pts.append((r1, c))
    for r in range(r0 + 1, r1):
        pts.append((r, c0))
        pts.append((r, c1))
    return pts


def _is_rect_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return False
    want = set(_frame_coords(r0, c0, r1, c1))
    have = set(coords)
    if have != want:
        return False
    return True


def _find_separator(grid: np.ndarray) -> int:
    rows = [r for r in range(grid.shape[0]) if np.all(grid[r] == GREY)]
    assert len(rows) == 1, "expected one separator row"
    return rows[0]


def _find_panels(top: np.ndarray):
    panels = []
    for color in COLOR_POOL:
        comps = _components(top == color)
        for comp in comps:
            if not _is_rect_frame(top, color, comp):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            inner = top[r0 + 1 : r1, c0 + 1 : c1]
            vals = set(int(v) for v in np.unique(inner))
            if BG not in vals:
                continue
            fill_vals = vals - {BG}
            if len(fill_vals) != 1:
                continue
            hole_comps = _components(inner == BG)
            if len(hole_comps) != 1:
                continue
            hole_mask = _crop_mask(hole_comps[0])
            hr0, hc0, hr1, hc1 = _bbox(hole_comps[0])
            panels.append(
                {
                    "border": color,
                    "fill": next(iter(fill_vals)),
                    "bbox": (r0, c0, r1, c1),
                    "hole_bbox_inner": (hr0, hc0, hr1, hc1),
                    "hole_mask": hole_mask,
                }
            )
    assert panels, "expected at least one panel"
    assert len({p["border"] for p in panels}) == len(panels), "panel colors must be unique"
    return panels


def _find_chests(bottom: np.ndarray):
    chests = []
    comps = _components(bottom == GREY)
    for comp in comps:
        if not _is_rect_frame(bottom, GREY, comp):
            continue
        r0, c0, r1, c1 = _bbox(comp)
        inner = bottom[r0 + 1 : r1, c0 + 1 : c1]
        if inner.size == 0:
            continue
        if np.any(inner == GREY) or np.any(inner == BG):
            continue
        chests.append({"bbox": (r0, c0, r1, c1), "pattern": inner.copy()})
    assert chests, "expected at least one chest"
    return chests


def _find_keys(bottom: np.ndarray, chests):
    blocked = np.zeros_like(bottom, dtype=bool)
    for chest in chests:
        r0, c0, r1, c1 = chest["bbox"]
        blocked[r0 : r1 + 1, c0 : c1 + 1] = True
    keys = []
    for color in COLOR_POOL:
        comps = _components((bottom == color) & (~blocked))
        for comp in comps:
            mask = _crop_mask(comp)
            r0, c0, r1, c1 = _bbox(comp)
            keys.append({"color": color, "bbox": (r0, c0, r1, c1), "mask": mask})
    assert keys, "expected at least one key"
    assert len({k["color"] for k in keys}) == len(keys), "key colors must be unique"
    return keys


def _rotation_to_match(src: np.ndarray, dst: np.ndarray) -> int:
    matches = [k for k in range(4) if np.array_equal(np.rot90(src, k), dst)]
    assert len(matches) == 1, "key rotation must be unique"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    top = g[:sep].copy()
    bottom = g[sep + 1 :]

    panels = _find_panels(top)
    chests = _find_chests(bottom)
    keys = _find_keys(bottom, chests)

    key_by_color = {k["color"]: k for k in keys}
    chest_by_color = {}
    for key in keys:
        kr0, kc0, kr1, kc1 = key["bbox"]
        kh, kw = key["mask"].shape
        matches = []
        for chest in chests:
            r0, c0, r1, c1 = chest["bbox"]
            inner_h = r1 - r0 - 1
            inner_w = c1 - c0 - 1
            if inner_h != kh or inner_w != kw:
                continue
            if r0 != kr1 + 2:
                continue
            if c0 + 1 != kc0:
                continue
            matches.append(chest)
        assert len(matches) == 1, "each key must have one chest directly below it"
        chest_by_color[key["color"]] = matches[0]

    out = top.copy()
    for panel in panels:
        color = panel["border"]
        assert color in key_by_color, "missing matching key color"
        key = key_by_color[color]
        chest = chest_by_color[color]
        rot = _rotation_to_match(key["mask"], panel["hole_mask"])
        rotated_pattern = np.rot90(chest["pattern"], rot)
        assert rotated_pattern.shape == panel["hole_mask"].shape

        r0, c0, _r1, _c1 = panel["bbox"]
        hr0, hc0, _hr1, _hc1 = panel["hole_bbox_inner"]
        hole_top = r0 + 1 + hr0
        hole_left = c0 + 1 + hc0
        ph, pw = panel["hole_mask"].shape
        region = out[hole_top : hole_top + ph, hole_left : hole_left + pw]
        region[panel["hole_mask"]] = rotated_pattern[panel["hole_mask"]]

    return out


def _random_shape(rng: random.Random) -> np.ndarray:
    for _ in range(100):
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        cells = {(rng.randrange(h), rng.randrange(w))}
        target = rng.randint(3, min(h * w - 1, 6))
        while len(cells) < target:
            r, c = rng.choice(tuple(cells))
            options = []
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    options.append((nr, nc))
            if not options:
                break
            cells.add(rng.choice(options))
        if len(cells) < 3:
            continue
        coords = list(cells)
        mask = _crop_mask(coords)
        if mask.all():
            continue
        rots = {(np.rot90(mask, k).shape, np.rot90(mask, k).tobytes()) for k in range(4)}
        if len(rots) != 4:
            continue
        return mask
    raise RuntimeError("failed to build asymmetric key shape")


def _random_pattern(rng: random.Random, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    for _ in range(100):
        palette = rng.sample(COLOR_POOL, rng.randint(2, min(4, len(COLOR_POOL))))
        arr = np.array([[rng.choice(palette) for _ in range(w)] for _ in range(h)], dtype=int)
        if len(set(int(v) for v in arr.flat)) < 2:
            continue
        return arr
    raise RuntimeError("failed to build chest pattern")


def _draw_panel(
    grid: np.ndarray,
    top: int,
    left: int,
    border_color: int,
    fill_color: int,
    hole_mask: np.ndarray,
    pad_t: int,
    pad_l: int,
):
    hh, hw = hole_mask.shape
    inner_h = hh + pad_t + 1 + 1
    inner_w = hw + pad_l + 1 + 1
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    grid[top : top + outer_h, left : left + outer_w] = border_color
    grid[top + 1 : top + outer_h - 1, left + 1 : left + outer_w - 1] = fill_color
    hole_top = top + 1 + pad_t
    hole_left = left + 1 + pad_l
    grid[hole_top : hole_top + hh, hole_left : hole_left + hw][hole_mask] = BG
    return outer_h, outer_w


def _can_place(mask: np.ndarray, top: int, left: int, shape: tuple[int, int]) -> bool:
    h, w = shape
    if top < 0 or left < 0 or top + h > mask.shape[0] or left + w > mask.shape[1]:
        return False
    return not np.any(mask[top : top + h, left : left + w])


def _occupy(mask: np.ndarray, top: int, left: int, shape: tuple[int, int], margin: int = 0):
    h, w = shape
    r0 = max(0, top - margin)
    c0 = max(0, left - margin)
    r1 = min(mask.shape[0], top + h + margin)
    c1 = min(mask.shape[1], left + w + margin)
    mask[r0:r1, c0:c1] = True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n = rng.randint(2, 3)
    colors = rng.sample(COLOR_POOL, 2 * n)
    border_colors = colors[:n]
    fill_colors = colors[n:]

    items = []
    for i in range(n):
        key_mask = _random_shape(rng)
        rot = rng.randrange(4)
        hole_mask = np.rot90(key_mask, rot)
        chest_pattern = _random_pattern(rng, key_mask.shape)
        pad_t = rng.randint(1, 2)
        pad_l = rng.randint(1, 2)
        panel_h = hole_mask.shape[0] + pad_t + 4
        panel_w = hole_mask.shape[1] + pad_l + 4
        items.append(
            {
                "border": border_colors[i],
                "fill": fill_colors[i],
                "key": key_mask,
                "rot": rot,
                "hole": hole_mask,
                "pattern": chest_pattern,
                "pad_t": pad_t,
                "pad_l": pad_l,
                "panel_h": panel_h,
                "panel_w": panel_w,
            }
        )

    top_order = items[:]
    rng.shuffle(top_order)
    bottom_order = items[:]
    rng.shuffle(bottom_order)

    top_gaps = [rng.randint(2, 4) for _ in range(n - 1)]
    bottom_gaps = [rng.randint(3, 5) for _ in range(n - 1)]
    top_span = sum(item["panel_w"] for item in top_order) + sum(top_gaps)
    bottom_span = sum(item["key"].shape[1] + 2 for item in bottom_order) + sum(bottom_gaps)

    grid_w = max(top_span, bottom_span, 18) + rng.randint(3, 6)
    top_h = max(item["panel_h"] for item in items) + 2
    bottom_h = max(item["key"].shape[0] + item["pattern"].shape[0] + 5 for item in items) + 2
    grid_h = top_h + 1 + bottom_h
    if grid_w > 30 or grid_h > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    sep = top_h
    grid[sep] = GREY

    x = rng.randint(1, grid_w - top_span - 1)
    for idx, item in enumerate(top_order):
        top = rng.randint(0, top_h - item["panel_h"] - 1)
        _draw_panel(
            grid,
            top,
            x,
            item["border"],
            item["fill"],
            item["hole"],
            item["pad_t"],
            item["pad_l"],
        )
        x += item["panel_w"]
        if idx < len(top_gaps):
            x += top_gaps[idx]

    occupied = np.zeros((bottom_h, grid_w), dtype=bool)
    x = rng.randint(1, grid_w - bottom_span - 1)
    for idx, item in enumerate(bottom_order):
        key = item["key"]
        kh, kw = key.shape
        chest = item["pattern"]
        outer_h, outer_w = chest.shape[0] + 2, chest.shape[1] + 2
        total_h = kh + 1 + outer_h
        y = rng.randint(1, bottom_h - total_h - 1)
        while not _can_place(occupied, y, x, (total_h, outer_w)):
            y = rng.randint(1, bottom_h - total_h - 1)
            x += 1
            if x + outer_w >= grid_w - 1:
                return None
        base_r = sep + 1 + y
        key_top = base_r
        key_left = x + 1
        for rr in range(kh):
            for cc in range(kw):
                if key[rr, cc]:
                    grid[key_top + rr, key_left + cc] = item["border"]
        chest_top = key_top + kh + 1
        chest_left = x
        grid[chest_top : chest_top + outer_h, chest_left : chest_left + outer_w] = GREY
        grid[chest_top + 1 : chest_top + outer_h - 1, chest_left + 1 : chest_left + outer_w - 1] = chest
        _occupy(occupied, y, x, (total_h, outer_w), margin=1)
        x += outer_w
        if idx < len(bottom_gaps):
            x += bottom_gaps[idx]

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generation failed")
