import numpy as np


BG = 8
MAX_RETRIES = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _full_edge_count(mask: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return sum(
        (
            bool(np.all(sub[0, :])),
            bool(np.all(sub[-1, :])),
            bool(np.all(sub[:, 0])),
            bool(np.all(sub[:, -1])),
        )
    )


def _find_rectangle_colors(grid: np.ndarray) -> list[int]:
    colors = [int(x) for x in np.unique(grid) if int(x) != BG]
    rect_colors = []
    for color in colors:
        mask = grid == color
        box = _bbox(mask)
        if _full_edge_count(mask, box) >= 2:
            rect_colors.append(color)
    assert 3 <= len(rect_colors) <= 5
    return rect_colors


def main(grid: np.ndarray) -> np.ndarray:
    """Separate overlapping colored rectangles into restored layers sorted by area."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rect_colors = _find_rectangle_colors(g)
    rect_color_set = set(rect_colors)

    layers = []
    for color in rect_colors:
        mask = g == color
        r0, c0, r1, c1 = _bbox(mask)
        h, w = r1 - r0 + 1, c1 - c0 + 1

        rect = np.full((h, w), color, dtype=int)
        inside = g[r0 : r1 + 1, c0 : c1 + 1]
        pattern_cells = np.argwhere(
            (inside != BG) & (inside != color) & ~np.isin(inside, list(rect_color_set))
        )
        if len(pattern_cells) > 0:
            pattern_colors = {int(inside[r, c]) for r, c in pattern_cells}
            assert len(pattern_colors) == 1
            pattern_color = next(iter(pattern_colors))
            pat = np.zeros((h, w), dtype=bool)
            for rr, cc in pattern_cells:
                rr = int(rr)
                cc = int(cc)
                pat[rr, cc] = True
                pat[h - 1 - rr, w - 1 - cc] = True
            rect[pat] = pattern_color

        layers.append((h * w, h, w, rect))

    layers.sort(key=lambda item: item[0])
    out_h = max(h for _, h, _, _ in layers)
    out_w = sum(w for _, _, w, _ in layers) + len(layers) - 1
    out = np.full((out_h, out_w), BG, dtype=int)

    c = 0
    for _, h, w, rect in layers:
        out[:h, c : c + w] = rect
        c += w + 1
    return out


def _mirror(rc: tuple[int, int], h: int, w: int) -> tuple[int, int]:
    r, c = rc
    return h - 1 - r, w - 1 - c


def _rect_mask(rect, shape):
    r0, c0, h, w = rect["r"], rect["c"], rect["h"], rect["w"]
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r0 + h, c0 : c0 + w] = True
    return mask


def _visible_edge_count(rect, covered: np.ndarray) -> int:
    r0, c0, h, w = rect["r"], rect["c"], rect["h"], rect["w"]
    sub = covered[r0 : r0 + h, c0 : c0 + w]
    return sum(
        (
            bool(np.all(~sub[0, :])),
            bool(np.all(~sub[-1, :])),
            bool(np.all(~sub[:, 0])),
            bool(np.all(~sub[:, -1])),
        )
    )


def _build_pattern(rect, rng: np.random.Generator):
    h, w = rect["h"], rect["w"]
    prev_occ = rect["prev_overlap"]
    next_occ = rect["next_overlap"]

    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    allowed = []
    for rc in interior:
        mr, mc = _mirror(rc, h, w)
        if prev_occ[rc] or prev_occ[mr, mc]:
            continue
        allowed.append(rc)
    rng.shuffle(allowed)

    used = set()
    pattern = set()

    forced_done = False
    if next_occ.any():
        forced = []
        for rc in allowed:
            mr, mc = _mirror(rc, h, w)
            if next_occ[rc] or next_occ[mr, mc]:
                forced.append(rc)
        rng.shuffle(forced)
        for rc in forced:
            if rc in used:
                continue
            mr = _mirror(rc, h, w)
            used.add(rc)
            used.add(mr)
            pattern.add(rc)
            pattern.add(mr)
            forced_done = True
            break

    max_pairs = max(2, min(len(allowed) // 2, (h * w) // 8))
    target_pairs = int(rng.integers(1 + int(forced_done), max_pairs + 1))
    for rc in allowed:
        if len(pattern) // 2 >= target_pairs:
            break
        if rc in used:
            continue
        mr = _mirror(rc, h, w)
        used.add(rc)
        used.add(mr)
        pattern.add(rc)
        pattern.add(mr)

    if h % 2 == 1 and w % 2 == 1 and not prev_occ[h // 2, w // 2] and rng.random() < 0.4:
        pattern.add((h // 2, w // 2))

    if len(pattern) < 2:
        return None
    if next_occ.any() and not any(next_occ[rc] for rc in pattern):
        return None
    return pattern


def _paint_scene(rects, shape):
    grid = np.full(shape, BG, dtype=int)
    for rect in rects:
        r0, c0, h, w = rect["r"], rect["c"], rect["h"], rect["w"]
        grid[r0 : r0 + h, c0 : c0 + w] = rect["base"]
        for rr, cc in rect["pattern"]:
            grid[r0 + rr, c0 + cc] = rect["pattern_color"]
    return grid


def _gen_attempt():
    rng = np.random.default_rng()
    n = int(rng.integers(3, 6))

    palette = [0, 1, 2, 3, 4, 5, 6, 7, 9]
    rng.shuffle(palette)
    base_colors = palette[:n]
    pattern_pool = palette[n:]
    if not pattern_pool:
        return None

    dims = []
    used_areas = set()
    for _ in range(n):
        for _ in range(100):
            h = int(rng.integers(5, 10))
            w = int(rng.integers(5, 10))
            area = h * w
            if area not in used_areas:
                used_areas.add(area)
                dims.append((h, w))
                break
        else:
            return None

    direction = tuple(rng.choice([-1, 1], size=2))
    rects = [{"r": 0, "c": 0, "h": dims[0][0], "w": dims[0][1], "base": base_colors[0]}]
    rel_overlaps = []

    for i in range(1, n):
        prev = rects[-1]
        h, w = dims[i]
        oy_max = min(3, prev["h"] - 2, h - 2)
        ox_max = min(3, prev["w"] - 2, w - 2)
        if oy_max < 1 or ox_max < 1:
            return None
        oy = int(rng.integers(1, oy_max + 1))
        ox = int(rng.integers(1, ox_max + 1))
        dr, dc = direction

        if dr == 1:
            new_r = prev["r"] + prev["h"] - oy
            lower_r = prev["h"] - oy
            upper_r = 0
        else:
            new_r = prev["r"] - h + oy
            lower_r = 0
            upper_r = h - oy

        if dc == 1:
            new_c = prev["c"] + prev["w"] - ox
            lower_c = prev["w"] - ox
            upper_c = 0
        else:
            new_c = prev["c"] - w + ox
            lower_c = 0
            upper_c = w - ox

        rects.append({"r": new_r, "c": new_c, "h": h, "w": w, "base": base_colors[i]})
        rel_overlaps.append((i - 1, i, lower_r, lower_c, upper_r, upper_c, oy, ox))

    min_r = min(rect["r"] for rect in rects)
    min_c = min(rect["c"] for rect in rects)
    for rect in rects:
        rect["r"] += 2 - min_r
        rect["c"] += 2 - min_c

    max_r = max(rect["r"] + rect["h"] for rect in rects)
    max_c = max(rect["c"] + rect["w"] for rect in rects)
    if max_r + 2 > 30 or max_c + 2 > 30:
        return None
    shape = (max_r + 2, max_c + 2)

    pair_overlaps = {}
    for i in range(n):
        for j in range(i + 1, n):
            a, b = rects[i], rects[j]
            rr0 = max(a["r"], b["r"])
            cc0 = max(a["c"], b["c"])
            rr1 = min(a["r"] + a["h"], b["r"] + b["h"])
            cc1 = min(a["c"] + a["w"], b["c"] + b["w"])
            area = max(0, rr1 - rr0) * max(0, cc1 - cc0)
            if area > 0:
                pair_overlaps[(i, j)] = area

    if len(pair_overlaps) != n - 1:
        return None
    if any(abs(i - j) != 1 for i, j in pair_overlaps):
        return None

    for rect in rects:
        rect["prev_overlap"] = np.zeros((rect["h"], rect["w"]), dtype=bool)
        rect["next_overlap"] = np.zeros((rect["h"], rect["w"]), dtype=bool)

    for low, high, lr, lc, hr, hc, oy, ox in rel_overlaps:
        rects[low]["next_overlap"][lr : lr + oy, lc : lc + ox] = True
        rects[high]["prev_overlap"][hr : hr + oy, hc : hc + ox] = True

    for i, rect in enumerate(rects):
        rect["pattern_color"] = pattern_pool[i % len(pattern_pool)]
        pattern = _build_pattern(rect, rng)
        if pattern is None:
            return None
        rect["pattern"] = pattern

    grid = _paint_scene(rects, shape)

    covered = np.zeros(shape, dtype=bool)
    for i in range(n - 1, -1, -1):
        rects[i]["visible_edges"] = _visible_edge_count(rects[i], covered)
        covered |= _rect_mask(rects[i], shape)
    if any(rect["visible_edges"] < 2 for rect in rects):
        return None

    for rect in rects:
        mask = grid == rect["base"]
        if not mask.any():
            return None
        if _bbox(mask) != (rect["r"], rect["c"], rect["r"] + rect["h"] - 1, rect["c"] + rect["w"] - 1):
            return None

    for rect in rects:
        r0, c0, h, w = rect["r"], rect["c"], rect["h"], rect["w"]
        inside = grid[r0 : r0 + h, c0 : c0 + w]
        visible = np.argwhere(inside == rect["pattern_color"])
        if len(visible) == 0:
            return None
        if any(rect["prev_overlap"][int(r), int(c)] for r, c in visible):
            return None

    try:
        out = main(grid)
    except Exception:
        return None
    if not isinstance(out, np.ndarray):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
