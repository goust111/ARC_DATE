from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _background(g: np.ndarray) -> int:
    counts = Counter(map(int, g.flat))
    return int(counts.most_common(1)[0][0])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q = deque([(sr, sc)])
        seen[sr, sc] = True
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        out.append(np.array(pts, dtype=int))
    return out


def _edge_state(g: np.ndarray, color: int, side: str, r0: int, c0: int, r1: int, c1: int) -> str:
    if side == "up":
        vals = g[r0, c0 : c1 + 1]
        return "full" if bool(np.all(vals == color)) else "open" if (
            int(g[r0, c0]) == color
            and int(g[r0, c1]) == color
            and bool(np.all(g[r0, c0 + 1 : c1] != color))
        ) else "bad"
    if side == "down":
        vals = g[r1, c0 : c1 + 1]
        return "full" if bool(np.all(vals == color)) else "open" if (
            int(g[r1, c0]) == color
            and int(g[r1, c1]) == color
            and bool(np.all(g[r1, c0 + 1 : c1] != color))
        ) else "bad"
    if side == "left":
        vals = g[r0 : r1 + 1, c0]
        return "full" if bool(np.all(vals == color)) else "open" if (
            int(g[r0, c0]) == color
            and int(g[r1, c0]) == color
            and bool(np.all(g[r0 + 1 : r1, c0] != color))
        ) else "bad"
    vals = g[r0 : r1 + 1, c1]
    return "full" if bool(np.all(vals == color)) else "open" if (
        int(g[r0, c1]) == color
        and int(g[r1, c1]) == color
        and bool(np.all(g[r0 + 1 : r1, c1] != color))
    ) else "bad"


def _open_edge_cells(side: str, r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    if side == "up":
        return [(r0, c) for c in range(c0 + 1, c1)]
    if side == "down":
        return [(r1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0) for r in range(r0 + 1, r1)]
    return [(r, c1) for r in range(r0 + 1, r1)]


def _detect_u_frame(g: np.ndarray, bg: int, pts: np.ndarray) -> dict:
    rows = pts[:, 0]
    cols = pts[:, 1]
    r0, r1 = int(rows.min()), int(rows.max())
    c0, c1 = int(cols.min()), int(cols.max())
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h >= 3 and w >= 3, "U frame must have a real interior"

    color = int(g[int(pts[0, 0]), int(pts[0, 1])])
    local = g[r0 : r1 + 1, c0 : c1 + 1]
    assert bool(np.all(g[rows, cols] == color)), "each frame component must be monochrome"
    assert bool(np.all(local[1:-1, 1:-1] == bg)), "frame interior must be background"

    states = {side: _edge_state(g, color, side, r0, c0, r1, c1) for side in DIR_VECS}
    open_sides = [side for side, state in states.items() if state == "open"]
    assert len(open_sides) == 1, "each frame must have exactly one open side"
    assert all(state in ("full", "open") for state in states.values()), "invalid frame edge"
    assert sum(state == "full" for state in states.values()) == 3, "three frame sides must be closed"

    side = open_sides[0]
    cells = _open_edge_cells(side, r0, c0, r1, c1)
    assert cells, "open side must contain at least one missing edge cell"
    assert all(int(g[r, c]) == bg for r, c in cells), "open side cells must be background"
    return {"bbox": (r0, c0, r1, c1), "color": color, "side": side, "open_cells": cells}


def _ray_distance(g: np.ndarray, bg: int, start: tuple[int, int], side: str) -> int:
    h, w = g.shape
    dr, dc = DIR_VECS[side]
    r, c = start
    dist = 0
    while True:
        r += dr
        c += dc
        dist += 1
        assert 0 <= r < h and 0 <= c < w, "each open-side ray must hit another frame"
        if int(g[r, c]) != bg:
            return dist


def _close_side(out: np.ndarray, color: int, side: str, r0: int, c0: int, r1: int, c1: int) -> None:
    if side == "up":
        out[r0, c0 : c1 + 1] = color
    elif side == "down":
        out[r1, c0 : c1 + 1] = color
    elif side == "left":
        out[r0 : r1 + 1, c0] = color
    else:
        out[r0 : r1 + 1, c1] = color


def _draw_stripes(out: np.ndarray, color: int, side: str, d: int, r0: int, c0: int, r1: int, c1: int) -> None:
    assert d >= 1
    if side == "up":
        for r in range(r0 + 1, r1, d):
            out[r, c0 + 1 : c1] = color
    elif side == "down":
        for r in range(r1 - 1, r0, -d):
            out[r, c0 + 1 : c1] = color
    elif side == "left":
        for c in range(c0 + 1, c1, d):
            out[r0 + 1 : r1, c] = color
    else:
        for c in range(c1 - 1, c0, -d):
            out[r0 + 1 : r1, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Close each open rectangular frame and stripe its interior using the distance to the first frame seen through the opening.

    Input structure:
    - The background is the dominant color.
    - Several foreground colors form separate one-cell-thick rectangular U frames.
    - Each U frame is hollow, has exactly one missing side, and every ray cast outward from that open side hits another frame wall.

    Rule:
    - Find each monochrome U-frame component and identify which of its four border sides is open.
    - From every missing edge cell, cast a ray in the opening direction and take the shortest hit distance D to any non-background pixel.
    - Repaint the missing side in the frame color, turning the U into a closed rectangle.
    - Inside the rectangle, draw same-color stripes perpendicular to the opening direction, starting just inside the original opening and repeating every D cells.

    Key insight: The open side gives both the ray direction and the stripe orientation, while the nearest wall along those rays encodes the stripe spacing.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = _background(g)

    frames: list[dict] = []
    for color in [c for c in Counter(map(int, g.flat)) if int(c) != bg]:
        for pts in _components(g == int(color)):
            frames.append(_detect_u_frame(g, bg, pts))
    assert frames, "expected at least one U frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        side = frame["side"]
        color = int(frame["color"])
        distances = [_ray_distance(g, bg, cell, side) for cell in frame["open_cells"]]
        d = min(distances)
        _close_side(out, color, side, r0, c0, r1, c1)
        _draw_stripes(out, color, side, d, r0, c0, r1, c1)
    return out


# --------------- generator ---------------

GEN_RETRIES = 500
BG_COLORS = list(range(10))
MIN_SIDE, MAX_SIDE = 12, 30


def _draw_u(g: np.ndarray, r: int, c: int, h: int, w: int, side: str, color: int, bg: int) -> None:
    r1 = r + h - 1
    c1 = c + w - 1
    g[r, c : c1 + 1] = color
    g[r1, c : c1 + 1] = color
    g[r : r1 + 1, c] = color
    g[r : r1 + 1, c1] = color
    if side == "up":
        g[r, c + 1 : c1] = bg
    elif side == "down":
        g[r1, c + 1 : c1] = bg
    elif side == "left":
        g[r + 1 : r1, c] = bg
    elif side == "right":
        g[r + 1 : r1, c1] = bg


def _paint_specs(bg: int, shape: tuple[int, int], specs: list[dict]) -> np.ndarray:
    g = np.full(shape, bg, dtype=int)
    for spec in specs:
        sprite = np.full((spec["h"], spec["w"]), bg, dtype=int)
        _draw_u(sprite, 0, 0, spec["h"], spec["w"], spec["side"], int(spec["color"]), bg)
        r, c = int(spec["r"]), int(spec["c"])
        mask = sprite != bg
        region = g[r : r + spec["h"], c : c + spec["w"]]
        if bool(np.any(mask & (region != bg))):
            raise ValueError("overlap")
        region[mask] = sprite[mask]
    return g


def _bbox(spec: dict) -> tuple[int, int, int, int]:
    return (
        int(spec["r"]),
        int(spec["c"]),
        int(spec["r"] + spec["h"] - 1),
        int(spec["c"] + spec["w"] - 1),
    )


def _bbox_gap(a: dict, b: dict) -> int:
    ar0, ac0, ar1, ac1 = _bbox(a)
    br0, bc0, br1, bc1 = _bbox(b)
    row_gap = max(0, br0 - ar1 - 1, ar0 - br1 - 1)
    col_gap = max(0, bc0 - ac1 - 1, ac0 - bc1 - 1)
    return max(row_gap, col_gap) if row_gap == 0 or col_gap == 0 else max(row_gap, col_gap)


def _boxes_overlap(a: dict, b: dict) -> bool:
    ar0, ac0, ar1, ac1 = _bbox(a)
    br0, bc0, br1, bc1 = _bbox(b)
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _basic_specs_ok(specs: list[dict], h: int, w: int) -> bool:
    for spec in specs:
        if spec["h"] < 4 or spec["w"] < 4:
            return False
        if spec["r"] < 1 or spec["c"] < 1:
            return False
        if spec["r"] + spec["h"] >= h or spec["c"] + spec["w"] >= w:
            return False
    for i, a in enumerate(specs):
        for b in specs[i + 1 :]:
            if _boxes_overlap(a, b):
                return False
            if _bbox_gap(a, b) < 1:
                return False
    return True


def _vertical_pair(h: int, w: int, colors: list[int], c_min: int, c_max: int) -> list[dict] | None:
    fh = random.randint(4, 8)
    th = random.randint(4, 8)
    fw = random.randint(4, 7)
    gap = random.randint(1, 4)
    total = fh + gap + th
    if total + 2 > h or c_max - c_min + 1 < fw:
        return None
    r0 = random.randint(1, h - total - 1)
    c0 = random.randint(c_min, c_max - fw + 1)
    return [
        {"r": r0, "c": c0, "h": fh, "w": fw, "side": "down", "color": colors[0]},
        {"r": r0 + fh + gap, "c": c0, "h": th, "w": fw, "side": "up", "color": colors[1]},
    ]


def _horizontal_pair(h: int, w: int, colors: list[int], c_min: int, c_max: int) -> list[dict] | None:
    fh = random.randint(4, 7)
    lw = random.randint(4, 7)
    rw = random.randint(4, 7)
    gap = random.randint(1, 4)
    total = lw + gap + rw
    if fh + 2 > h or c_max - c_min + 1 < total:
        return None
    r0 = random.randint(1, h - fh - 1)
    c0 = random.randint(c_min, c_max - total + 1)
    return [
        {"r": r0, "c": c0, "h": fh, "w": lw, "side": "right", "color": colors[0]},
        {"r": r0, "c": c0 + lw + gap, "h": fh, "w": rw, "side": "left", "color": colors[1]},
    ]


def _extra_horizontal_against(target: dict, color: int, grid_w: int) -> dict | None:
    eh = random.randint(4, min(7, int(target["h"])))
    ew = random.randint(4, 7)
    rr = random.randint(int(target["r"]), int(target["r"] + target["h"] - eh))
    gap = random.randint(1, 4)
    left_c = int(target["c"] - gap - ew)
    right_c = int(target["c"] + target["w"] + gap)
    choices: list[dict] = []
    if left_c >= 1:
        choices.append({"r": rr, "c": left_c, "h": eh, "w": ew, "side": "right", "color": color})
    if right_c + ew < grid_w:
        choices.append({"r": rr, "c": right_c, "h": eh, "w": ew, "side": "left", "color": color})
    return random.choice(choices) if choices else None


def _extra_vertical_against(target: dict, color: int, grid_h: int) -> dict | None:
    ew = random.randint(4, min(7, int(target["w"])))
    eh = random.randint(4, 7)
    cc = random.randint(int(target["c"]), int(target["c"] + target["w"] - ew))
    gap = random.randint(1, 4)
    top_r = int(target["r"] - gap - eh)
    bottom_r = int(target["r"] + target["h"] + gap)
    choices: list[dict] = []
    if top_r >= 1:
        choices.append({"r": top_r, "c": cc, "h": eh, "w": ew, "side": "down", "color": color})
    if bottom_r + eh < grid_h:
        choices.append({"r": bottom_r, "c": cc, "h": eh, "w": ew, "side": "up", "color": color})
    return random.choice(choices) if choices else None


def _attempt_specs(n: int, colors: list[int]) -> tuple[tuple[int, int], list[dict]] | None:
    if n == 4:
        h = random.randint(24, MAX_SIDE)
        w = random.randint(26, MAX_SIDE)
        v = _vertical_pair(h, w, colors[:2], 1, min(8, w - 1))
        if v is None:
            return None
        min_c = max(1, int(v[0]["c"] + v[0]["w"] + 2))
        hp = _horizontal_pair(h, w, colors[2:], min_c, w - 2)
        if hp is None:
            return None
        specs = v + hp
        return (h, w), specs

    if n == 3:
        h = random.randint(18, MAX_SIDE)
        w = random.randint(18, MAX_SIDE)
        if random.random() < 0.5:
            v = _vertical_pair(h, w, colors[:2], 7, max(7, w - 8))
            if v is None:
                return None
            extra = _extra_horizontal_against(random.choice(v), colors[2], w)
            if extra is None:
                return None
            return (h, w), v + [extra]
        hp = _horizontal_pair(h, w, colors[:2], 1, w - 2)
        if hp is None:
            return None
        extra = _extra_vertical_against(random.choice(hp), colors[2], h)
        if extra is None:
            return None
        return (h, w), hp + [extra]

    h = random.randint(MIN_SIDE, 24)
    w = random.randint(MIN_SIDE, 24)
    if random.random() < 0.5:
        specs = _vertical_pair(h, w, colors[:2], 1, w - 2)
    else:
        specs = _horizontal_pair(h, w, colors[:2], 1, w - 2)
    if specs is None:
        return None
    return (h, w), specs


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([2, 3, 4])
    bg = random.choice(BG_COLORS)
    palette = [c for c in BG_COLORS if c != bg]
    colors = random.sample(palette, n)
    packed = _attempt_specs(n, colors)
    if packed is None:
        return None
    shape, specs = packed
    h, w = shape
    if not _basic_specs_ok(specs, h, w):
        return None
    try:
        grid = _paint_specs(bg, shape, specs)
        out = main(grid)
    except Exception:
        return None
    if grid.shape != out.shape or max(grid.shape) > 30:
        return None
    if np.array_equal(grid, out):
        return None
    if _background(grid) != bg:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed1030756833_6_3 input")
