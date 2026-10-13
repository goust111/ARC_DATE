import random
from collections import Counter, deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
SIDES = ("top", "bottom", "left", "right")

GEN_ATTEMPTS = 400


def _mode(arr: np.ndarray) -> int:
    vals, counts = np.unique(arr, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
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
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _find_frame(grid: np.ndarray):
    bg = _mode(grid)
    h, w = grid.shape
    best = None
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        pts = np.argwhere(grid == color)
        if pts.size == 0:
            continue
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        if r1 - r0 < 3 or c1 - c0 < 3:
            continue
        perim = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        if len(pts) != perim:
            continue
        ok = True
        for c in range(c0, c1 + 1):
            ok &= grid[r0, c] == color and grid[r1, c] == color
        for r in range(r0, r1 + 1):
            ok &= grid[r, c0] == color and grid[r, c1] == color
        if not ok:
            continue
        inside = grid[r0 + 1 : r1, c0 + 1 : c1]
        if inside.size == 0:
            continue
        inside_vals = np.unique(inside)
        if len(inside_vals) != 1:
            continue
        cand = (perim, r0, c0, r1, c1, color, int(inside_vals[0]), bg)
        if best is None or cand[0] > best[0]:
            best = cand
    assert best is not None
    _, r0, c0, r1, c1, frame_color, fill_color, bg = best
    return (r0, c0, r1, c1), frame_color, fill_color, bg


def _touch_side(touch, bbox):
    r0, c0, r1, c1 = bbox
    r, c = touch
    if r == r0:
        return "top"
    if r == r1:
        return "bottom"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("touch not on frame")


def _map_touch(touch, bbox):
    r0, c0, r1, c1 = bbox
    r, c = touch
    out_h = r1 - r0 - 1
    out_w = c1 - c0 - 1
    if r == r0:
        return (0, c - c0 - 1)
    if r == r1:
        return (out_h - 1, c - c0 - 1)
    if c == c0:
        return (r - r0 - 1, 0)
    if c == c1:
        return (r - r0 - 1, out_w - 1)
    raise AssertionError("touch not on frame")


def _segment(a, b):
    r0, c0 = a
    r1, c1 = b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise AssertionError("segment must be axis-aligned")


def _ordered_l_path(a, b, side_a=None, side_b=None):
    if a == b:
        return [a]
    if a[0] == b[0] or a[1] == b[1]:
        return _segment(a, b)
    horiz = {"top", "bottom"}
    vert = {"left", "right"}
    bend = None
    if side_a in horiz and side_b in vert:
        bend = (b[0], a[1])
    elif side_b in horiz and side_a in vert:
        bend = (a[0], b[1])
    else:
        bend = (a[0], b[1])
    p1 = _segment(a, bend)
    p2 = _segment(bend, b)
    return p1 + p2[1:]


def _draw_alternating(out: np.ndarray, path, color_a: int, color_b: int):
    for i, (r, c) in enumerate(path):
        out[r, c] = color_a if i % 2 == 0 else color_b
    if out[path[-1]] != color_b:
        for i, (r, c) in enumerate(reversed(path)):
            out[r, c] = color_b if i % 2 == 0 else color_a
        if out[path[0]] != color_a:
            out[path[0]] = color_a
            out[path[-1]] = color_b


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer interior Manhattan drawings from exterior clue components touching a rectangular frame.
    """
    g = np.asarray(grid, dtype=int)
    bbox, frame_color, fill_color, bg = _find_frame(g)
    r0, c0, r1, c1 = bbox

    out = np.full((r1 - r0 - 1, c1 - c0 - 1), fill_color, dtype=int)

    clue_mask = (g != bg) & (g != frame_color) & (g != fill_color)
    for comp in _components(clue_mask):
        touch_colors = {}
        for r, c in comp:
            v = int(g[r, c])
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and g[nr, nc] == frame_color:
                    touch_colors[(nr, nc)] = v
        if len(touch_colors) != 2:
            continue
        touches = sorted(touch_colors)
        mapped = [_map_touch(t, bbox) for t in touches]
        sides = [_touch_side(t, bbox) for t in touches]
        colors = sorted({int(g[r, c]) for r, c in comp})
        if len(colors) == 1:
            color = colors[0]
            ra = min(mapped[0][0], mapped[1][0])
            rb = max(mapped[0][0], mapped[1][0])
            ca = min(mapped[0][1], mapped[1][1])
            cb = max(mapped[0][1], mapped[1][1])
            out[ra : rb + 1, ca : cb + 1] = color
        else:
            path = _ordered_l_path(mapped[0], mapped[1], sides[0], sides[1])
            _draw_alternating(out, path, touch_colors[touches[0]], touch_colors[touches[1]])
    return out


def _side_position(side, idx, bbox):
    r0, c0, r1, c1 = bbox
    if side == "top":
        return (r0, c0 + 1 + idx)
    if side == "bottom":
        return (r1, c0 + 1 + idx)
    if side == "left":
        return (r0 + 1 + idx, c0)
    if side == "right":
        return (r0 + 1 + idx, c1)
    raise AssertionError


def _perimeter_cycle(layer, bbox):
    r0, c0, r1, c1 = bbox
    tr0, tc0, tr1, tc1 = r0 - layer, c0 - layer, r1 + layer, c1 + layer
    cycle = []
    for c in range(tc0, tc1 + 1):
        cycle.append((tr0, c))
    for r in range(tr0 + 1, tr1 + 1):
        cycle.append((r, tc1))
    for c in range(tc1 - 1, tc0 - 1, -1):
        cycle.append((tr1, c))
    for r in range(tr1 - 1, tr0, -1):
        cycle.append((r, tc0))
    return cycle


def _connector(touch, side, layer, bbox):
    r0, c0, r1, c1 = bbox
    r, c = touch
    if side == "top":
        return [(rr, c) for rr in range(r0 - 1, r0 - layer - 1, -1)]
    if side == "bottom":
        return [(rr, c) for rr in range(r1 + 1, r1 + layer + 1)]
    if side == "left":
        return [(r, cc) for cc in range(c0 - 1, c0 - layer - 1, -1)]
    if side == "right":
        return [(r, cc) for cc in range(c1 + 1, c1 + layer + 1)]
    raise AssertionError


def _arc_between(cycle, a, b, rng):
    ia = cycle.index(a)
    ib = cycle.index(b)
    if ia <= ib:
        arc1 = cycle[ia : ib + 1]
        arc2 = cycle[ib:] + cycle[: ia + 1]
    else:
        arc1 = cycle[ia:] + cycle[: ib + 1]
        arc2 = cycle[ib : ia + 1][::-1]
    if len(arc1) == len(arc2):
        return arc1 if rng.random() < 0.5 else arc2[::-1]
    best = arc1 if len(arc1) < len(arc2) else arc2[::-1]
    return best


def _build_clue_path(touch1, side1, touch2, side2, layer, bbox, rng):
    conn1 = _connector(touch1, side1, layer, bbox)
    conn2 = _connector(touch2, side2, layer, bbox)
    track1 = conn1[-1]
    track2 = conn2[-1]
    cycle = _perimeter_cycle(layer, bbox)
    arc = _arc_between(cycle, track1, track2, rng)
    path = conn1 + arc[1:] + list(reversed(conn2[:-1]))
    deduped = []
    seen = set()
    for cell in path:
        if deduped and deduped[-1] == cell:
            continue
        if cell in seen:
            return None
        deduped.append(cell)
        seen.add(cell)
    return deduped


def _footprint(kind, p1, s1, p2, s2):
    if kind == "mono":
        ra = min(p1[0], p2[0])
        rb = max(p1[0], p2[0])
        ca = min(p1[1], p2[1])
        cb = max(p1[1], p2[1])
        return {(r, c) for r in range(ra, rb + 1) for c in range(ca, cb + 1)}
    return set(_ordered_l_path(p1, p2, s1, s2))


def _sample_component(kind, bbox, out_shape, used_mask, rng):
    h, w = out_shape
    pairs_bi = [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    pairs_mono = [
        ("top", "bottom"),
        ("left", "right"),
        ("top", "left"),
        ("top", "right"),
        ("bottom", "left"),
        ("bottom", "right"),
    ]
    pairs = pairs_bi if kind == "bi" else pairs_mono
    for _ in range(120):
        side1, side2 = rng.choice(pairs)
        idx1 = rng.randrange(w) if side1 in ("top", "bottom") else rng.randrange(h)
        idx2 = rng.randrange(w) if side2 in ("top", "bottom") else rng.randrange(h)
        t1 = _side_position(side1, idx1, bbox)
        t2 = _side_position(side2, idx2, bbox)
        p1 = _map_touch(t1, bbox)
        p2 = _map_touch(t2, bbox)
        if p1 == p2:
            continue
        if kind == "bi":
            dist = abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])
            if dist < 3 or dist % 2 == 0:
                continue
        else:
            area = (abs(p1[0] - p2[0]) + 1) * (abs(p1[1] - p2[1]) + 1)
            if area < 3:
                continue
        fp = _footprint(kind, p1, side1, p2, side2)
        if any(used_mask[r, c] for r, c in fp):
            continue
        return {
            "kind": kind,
            "touches": (t1, t2),
            "sides": (side1, side2),
            "mapped": (p1, p2),
            "footprint": fp,
        }
    return None


def _color_path(grid, path, colors):
    if len(colors) == 1:
        for r, c in path:
            grid[r, c] = colors[0]
        return
    a, b = colors
    for i, (r, c) in enumerate(path):
        grid[r, c] = a if i % 2 == 0 else b
    if grid[path[-1]] != b:
        grid[path[-1]] = b


def _gen_attempt():
    rng = random.Random()
    bg, frame_color, fill_color = rng.sample(range(10), 3)
    min_margin = rng.randint(4, 6)
    top = min_margin + rng.randint(0, 2)
    bottom = min_margin + rng.randint(0, 2)
    left = min_margin + rng.randint(0, 2)
    right = min_margin + rng.randint(0, 2)
    inner_h = rng.randint(5, 9)
    inner_w = rng.randint(5, 10)
    h = top + 2 + inner_h + bottom
    w = left + 2 + inner_w + right
    grid = np.full((h, w), bg, dtype=int)

    bbox = (top, left, top + inner_h + 1, left + inner_w + 1)
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 + 1 : r1, c0 + 1 : c1] = fill_color

    max_components = min(4, min_margin - 1)
    n = rng.randint(2, max_components)
    types = ["mono"] * (n // 2) + ["bi"] * (n - n // 2)
    rng.shuffle(types)
    if "mono" not in types:
        types[0] = "mono"
    if "bi" not in types:
        types[-1] = "bi"

    used_inside = np.zeros((inner_h, inner_w), dtype=bool)
    comps = []
    for kind in types:
        comp = _sample_component(kind, bbox, (inner_h, inner_w), used_inside, rng)
        if comp is None:
            return None
        for r, c in comp["footprint"]:
            used_inside[r, c] = True
        comps.append(comp)

    layers = list(range(2, 2 + n))
    rng.shuffle(layers)
    occupied = np.zeros_like(grid, dtype=bool)

    palette = [c for c in range(10) if c not in (bg, frame_color, fill_color)]
    for comp, layer in zip(comps, layers):
        t1, t2 = comp["touches"]
        s1, s2 = comp["sides"]
        path = _build_clue_path(t1, s1, t2, s2, layer, bbox, rng)
        if path is None:
            return None
        for r, c in path:
            if occupied[r, c] or grid[r, c] != bg:
                return None
        if comp["kind"] == "mono":
            color = rng.choice(palette)
            colors = [color]
        else:
            a, b = rng.sample(palette, 2)
            colors = [a, b]
        _color_path(grid, path, colors)
        for r, c in path:
            occupied[r, c] = True

    clue_mask = (grid != bg) & (grid != frame_color) & (grid != fill_color)
    found = _components(clue_mask)
    if len(found) != n:
        return None
    for comp in found:
        touches = set()
        for r, c in comp:
            for dr, dc in DIR4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == frame_color:
                    touches.add((nr, nc))
        if len(touches) != 2:
            return None

    out = main(grid)
    if out.shape != (inner_h, inner_w):
        return None
    if not np.any(out != fill_color):
        return None
    if len(np.unique(out)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
