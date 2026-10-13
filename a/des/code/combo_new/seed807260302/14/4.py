from __future__ import annotations

import random

import numpy as np


BG = 0
MARK = 9
FULL_H = 30
FULL_W = 30
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
GEN_RETRIES = 500
PLACEMENT_RETRIES = 250

Coord = tuple[int, int]


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: list[Coord] | set[Coord]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _find_rectangles(g: np.ndarray) -> dict[int, tuple[int, int, int, int]]:
    rects: dict[int, tuple[int, int, int, int]] = {}
    for color in sorted(int(v) for v in np.unique(g) if int(v) not in (BG, MARK)):
        pts = [(int(r), int(c)) for r, c in np.argwhere(g == color)]
        assert pts, "foreground rectangle color must appear"
        r0, c0, r1, c1 = _bbox(pts)
        area = (r1 - r0 + 1) * (c1 - c0 + 1)
        assert area == len(pts), "each non-marker foreground color must form one solid rectangle"
        rects[int(color)] = (int(r0), int(c0), int(r1), int(c1))
    return rects


def _marker_corner(cells: list[Coord]) -> Coord:
    cell_set = set(cells)
    for r, c in cells:
        deg = 0
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (r + dr, c + dc) in cell_set:
                deg += 1
        if deg == 2:
            return int(r), int(c)
    raise AssertionError("3-cell L marker must have exactly one corner")


def _decode_marker(g: np.ndarray, rects: dict[int, tuple[int, int, int, int]], cells: list[Coord]) -> dict:
    r0, c0, r1, c1 = _bbox(cells)
    assert r1 - r0 == 1 and c1 - c0 == 1 and len(cells) == 3, "marker must be a 2x2-minus-1 L"

    corner = _marker_corner(cells)
    cr, cc = corner
    h, w = g.shape

    adj: list[tuple[int, int, int]] = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = cr + dr, cc + dc
        if 0 <= nr < h and 0 <= nc < w:
            color = int(g[nr, nc])
            if color not in (BG, MARK):
                adj.append((int(nr), int(nc), color))
    assert len(adj) == 1, "marker corner must touch exactly one rectangle"
    nr, nc, color = adj[0]
    rect = rects[color]

    if nr == cr + 1 and nc == cc:
        side = "top"
        along = [p for p in cells if p != corner and p[0] == cr]
        perp = [p for p in cells if p != corner and p[1] == cc]
        assert len(along) == 1 and len(perp) == 1
        assert perp[0][0] == cr - 1
        seg0, seg1 = sorted([cc, along[0][1]])
        assert seg1 == seg0 + 1
        assert rect[1] <= seg0 and seg1 <= rect[3]
        return {
            "corner": corner,
            "side": side,
            "color": color,
            "rect": rect,
            "seg": (int(seg0), int(seg1)),
        }
    if nr == cr - 1 and nc == cc:
        side = "bottom"
        along = [p for p in cells if p != corner and p[0] == cr]
        perp = [p for p in cells if p != corner and p[1] == cc]
        assert len(along) == 1 and len(perp) == 1
        assert perp[0][0] == cr + 1
        seg0, seg1 = sorted([cc, along[0][1]])
        assert seg1 == seg0 + 1
        assert rect[1] <= seg0 and seg1 <= rect[3]
        return {
            "corner": corner,
            "side": side,
            "color": color,
            "rect": rect,
            "seg": (int(seg0), int(seg1)),
        }
    if nr == cr and nc == cc + 1:
        side = "left"
        along = [p for p in cells if p != corner and p[1] == cc]
        perp = [p for p in cells if p != corner and p[0] == cr]
        assert len(along) == 1 and len(perp) == 1
        assert perp[0][1] == cc - 1
        seg0, seg1 = sorted([cr, along[0][0]])
        assert seg1 == seg0 + 1
        assert rect[0] <= seg0 and seg1 <= rect[2]
        return {
            "corner": corner,
            "side": side,
            "color": color,
            "rect": rect,
            "seg": (int(seg0), int(seg1)),
        }

    assert nr == cr and nc == cc - 1
    side = "right"
    along = [p for p in cells if p != corner and p[1] == cc]
    perp = [p for p in cells if p != corner and p[0] == cr]
    assert len(along) == 1 and len(perp) == 1
    assert perp[0][1] == cc + 1
    seg0, seg1 = sorted([cr, along[0][0]])
    assert seg1 == seg0 + 1
    assert rect[0] <= seg0 and seg1 <= rect[2]
    return {
        "corner": corner,
        "side": side,
        "color": color,
        "rect": rect,
        "seg": (int(seg0), int(seg1)),
    }


def _mirrored_targets(info: dict) -> list[tuple[int, int, int]]:
    r0, c0, r1, c1 = info["rect"]
    color = int(info["color"])
    side = info["side"]
    a0, a1 = info["seg"]
    out: list[tuple[int, int, int]] = []

    if side == "top":
        for r in range(r0, r1 + 1):
            rr = 2 * r0 - 1 - r
            for c in range(a0, a1 + 1):
                out.append((int(rr), int(c), color))
        return out
    if side == "bottom":
        for r in range(r0, r1 + 1):
            rr = 2 * r1 + 1 - r
            for c in range(a0, a1 + 1):
                out.append((int(rr), int(c), color))
        return out
    if side == "left":
        for r in range(a0, a1 + 1):
            for c in range(c0, c1 + 1):
                cc = 2 * c0 - 1 - c
                out.append((int(r), int(cc), color))
        return out

    for r in range(a0, a1 + 1):
        for c in range(c0, c1 + 1):
            cc = 2 * c1 + 1 - c
            out.append((int(r), int(cc), color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold a marked 2-cell edge flap of each solid rectangle outward across its touched boundary.

    Input structure:
    - The background is black.
    - Several non-overlapping solid rectangles appear in distinct non-maroon colors.
    - Some rectangle edges have an external 3-pixel maroon L marker whose corner touches the edge; one arm runs along the edge and one arm points outward.

    Rule:
    - For each maroon L, identify which rectangle edge its corner touches.
    - The corner pixel together with the arm that runs along the edge marks a 2-cell hinge segment on that edge.
    - Mirror the whole rectangle strip directly inside that 2-cell hinge segment across the boundary line to the outward side indicated by the perpendicular arm.
    - Keep the original rectangle in place, erase every maroon marker, and only paint mirrored cells onto background so earlier foreground stays visible.

    Key insight:
    - The L marker does not ask for mirroring the whole panel: it specifies one exact 2-cell edge segment and an outward folding direction, so the solver must decode a local flap reflection.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    rects = _find_rectangles(g)
    marker_comps = _components(g == MARK)
    markers = [_decode_marker(g, rects, comp) for comp in marker_comps]
    markers.sort(key=lambda info: (info["corner"][0], info["corner"][1]))

    out = g.copy()
    out[out == MARK] = BG

    for info in markers:
        for rr, cc, color in _mirrored_targets(info):
            if 0 <= rr < h and 0 <= cc < w and out[rr, cc] == BG:
                out[rr, cc] = int(color)
    return out


def _expand(cells: set[Coord], h: int, w: int, radius: int = 1) -> set[Coord]:
    out: set[Coord] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((int(rr), int(cc)))
    return out


def _shift(cells: set[Coord], origin: Coord) -> set[Coord]:
    orow, ocol = origin
    return {(int(r + orow), int(c + ocol)) for r, c in cells}


def _marked_spec(rng: random.Random, color: int) -> dict:
    side = rng.choice(["top", "bottom", "left", "right"])
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)

    rect = {(r, c) for r in range(h) for c in range(w)}
    marker: set[Coord] = set()
    copy: set[Coord] = set()

    if side == "top":
        start = rng.randint(0, w - 2)
        if rng.random() < 0.5:
            marker = {(-1, start), (-1, start + 1), (-2, start)}
        else:
            marker = {(-1, start + 1), (-1, start), (-2, start + 1)}
        for r in range(h):
            rr = -1 - r
            for c in range(start, start + 2):
                copy.add((int(rr), int(c)))
    elif side == "bottom":
        start = rng.randint(0, w - 2)
        if rng.random() < 0.5:
            marker = {(h, start), (h, start + 1), (h + 1, start)}
        else:
            marker = {(h, start + 1), (h, start), (h + 1, start + 1)}
        for r in range(h):
            rr = 2 * h - 1 - r
            for c in range(start, start + 2):
                copy.add((int(rr), int(c)))
    elif side == "left":
        start = rng.randint(0, h - 2)
        if rng.random() < 0.5:
            marker = {(start, -1), (start + 1, -1), (start, -2)}
        else:
            marker = {(start + 1, -1), (start, -1), (start + 1, -2)}
        for r in range(start, start + 2):
            for c in range(w):
                copy.add((int(r), int(-1 - c)))
    else:
        start = rng.randint(0, h - 2)
        if rng.random() < 0.5:
            marker = {(start, w), (start + 1, w), (start, w + 1)}
        else:
            marker = {(start + 1, w), (start, w), (start + 1, w + 1)}
        for r in range(start, start + 2):
            for c in range(w):
                copy.add((int(r), int(2 * w - 1 - c)))

    footprint = rect | marker | copy
    return {
        "color": int(color),
        "rect": rect,
        "marker": marker,
        "copy": copy,
        "footprint": footprint,
    }


def _plain_rect_spec(rng: random.Random, color: int) -> dict:
    h = rng.randint(3, 6)
    w = rng.randint(3, 6)
    rect = {(r, c) for r in range(h) for c in range(w)}
    return {
        "color": int(color),
        "rect": rect,
        "marker": set(),
        "copy": set(),
        "footprint": set(rect),
    }


def _place_spec(grid: np.ndarray, blocked: np.ndarray, spec: dict, rng: random.Random) -> bool:
    cells = spec["footprint"]
    min_r = min(r for r, _ in cells)
    max_r = max(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_c = max(c for _, c in cells)

    r_lo = -min_r
    r_hi = FULL_H - 1 - max_r
    c_lo = -min_c
    c_hi = FULL_W - 1 - max_c
    if r_lo > r_hi or c_lo > c_hi:
        return False

    for _ in range(PLACEMENT_RETRIES):
        origin = (rng.randint(r_lo, r_hi), rng.randint(c_lo, c_hi))
        placed_footprint = _shift(spec["footprint"], origin)
        halo = _expand(placed_footprint, FULL_H, FULL_W, radius=1)
        if any(blocked[r, c] for r, c in halo):
            continue

        for r, c in _shift(spec["rect"], origin):
            grid[r, c] = int(spec["color"])
        for r, c in _shift(spec["marker"], origin):
            grid[r, c] = MARK
        for r, c in halo:
            blocked[r, c] = True
        return True
    return False


def _crop_pair(inp: np.ndarray, out: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    union = (inp != BG) | (out != BG)
    rs, cs = np.where(union)
    assert rs.size > 0
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    top_pad = rng.randint(0, min(2, r0))
    left_pad = rng.randint(0, min(2, c0))
    bot_pad = rng.randint(0, min(2, inp.shape[0] - 1 - r1))
    right_pad = rng.randint(0, min(2, inp.shape[1] - 1 - c1))

    rr0 = r0 - top_pad
    rr1 = r1 + bot_pad
    cc0 = c0 - left_pad
    cc1 = c1 + right_pad
    return inp[rr0 : rr1 + 1, cc0 : cc1 + 1].copy(), out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    grid = np.zeros((FULL_H, FULL_W), dtype=int)
    blocked = np.zeros((FULL_H, FULL_W), dtype=bool)

    total = rng.randint(2, 4)
    n_marked = rng.randint(1, total)
    colors = rng.sample(COLOR_POOL, total)

    marked_specs = [_marked_spec(rng, colors[i]) for i in range(n_marked)]
    plain_specs = [_plain_rect_spec(rng, colors[n_marked + i]) for i in range(total - n_marked)]

    for spec in marked_specs + plain_specs:
        if not _place_spec(grid, blocked, spec, rng):
            return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    cropped_in, cropped_out = _crop_pair(grid, out, rng)
    if max(cropped_in.shape) > 30 or max(cropped_out.shape) > 30:
        return None
    if not np.array_equal(main(cropped_in), cropped_out):
        return None
    return cropped_in


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid folding-panel sample")
