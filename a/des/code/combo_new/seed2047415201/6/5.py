from __future__ import annotations

import math
import random

import numpy as np


BG = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]


def _components_by_color(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r, c in np.argwhere(g != BG):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(g[r, c])
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append({"color": color, "coords": coords})
    return comps


def _square_offset(bbox_h: int, bbox_w: int, side: int) -> tuple[int, int]:
    return (bbox_h - side) // 2, (bbox_w - side) // 2


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace each foreground object by an equal-area square colored by bbox fullness.

    Input structure:
    - Black background with several separated monochrome 4-connected foreground
      objects, each using a distinct non-background color.
    - Every object area is a perfect square.
    - Some objects fill their whole bounding box; other objects leave black cells
      inside their bounding box.

    Rule:
    - For each object, compute its area A and square side S = sqrt(A).
    - Clear all original foreground pixels.
    - Draw a solid SxS square centered on the object's original bounding box.
    - If the bounding box was completely filled by the object color, keep the
      original color; otherwise draw the replacement square in grey.

    Key insight:
    - Area determines only the replacement square's size, while bounding-box
      fullness independently determines whether the color is preserved or grey.
    """
    g = np.asarray(grid, dtype=int)
    out = np.full_like(g, BG)

    for comp in _components_by_color(g):
        color = int(comp["color"])
        coords = comp["coords"]
        area = int(len(coords))
        side = int(math.isqrt(area))
        assert side * side == area, "object area must be a perfect square"

        pts = np.array(coords, dtype=int)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))

        bbox = g[r0 : r1 + 1, c0 : c1 + 1]
        full_bbox = bool((bbox == color).all())
        draw_color = color if full_bbox else GREY

        bbox_h = int(r1 - r0 + 1)
        bbox_w = int(c1 - c0 + 1)
        off_r, off_c = _square_offset(bbox_h, bbox_w, side)
        sr = int(r0 + off_r)
        sc = int(c0 + off_c)
        assert 0 <= sr and 0 <= sc and sr + side <= g.shape[0] and sc + side <= g.shape[1]
        out[sr : sr + side, sc : sc + side] = int(draw_color)

    return out


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_CONCAVE_BASES = [
    _mask(
        [
            "00100",
            "00100",
            "11111",
            "00100",
            "00100",
        ]
    ),
    _mask(
        [
            "11111",
            "10000",
            "10000",
            "10000",
            "10000",
        ]
    ),
    _mask(
        [
            "111111",
            "100001",
            "100001",
            "111111",
        ]
    ),
    _mask(
        [
            "111111",
            "100000",
            "100000",
            "100000",
            "100000",
            "111111",
        ]
    ),
]


def _transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            cur = np.rot90(base, k)
            key = (tuple(map(int, cur.shape)), cur.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(cur.copy())
    return variants


_CONCAVE_VARIANTS = [variant for base in _CONCAVE_BASES for variant in _transforms(base)]


def _convex_mask(force_non_square: bool = False) -> np.ndarray:
    choices = [
        (2, 2, 2),
        (3, 3, 3),
        (4, 4, 4),
        (3, 1, 9),
        (3, 9, 1),
        (4, 2, 8),
        (4, 8, 2),
    ]
    if force_non_square:
        choices = [item for item in choices if item[1] != item[2]]
    elif random.random() < 0.70:
        nonsquare = [item for item in choices if item[1] != item[2]]
        choices = nonsquare
    _side, h, w = random.choice(choices)
    return np.ones((int(h), int(w)), dtype=bool)


def _concave_mask() -> np.ndarray:
    return random.choice(_CONCAVE_VARIANTS).copy()


def _placement_footprint(mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = map(int, mask.shape)
    area = int(mask.sum())
    side = int(math.isqrt(area))
    assert side * side == area
    off_r, off_c = _square_offset(h, w, side)

    footprint = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    for r in range(off_r, off_r + side):
        for c in range(off_c, off_c + side):
            footprint.add((int(r), int(c)))
    return footprint


def _try_place(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray, color: int) -> bool:
    h, w = grid.shape
    footprint = _placement_footprint(mask)
    min_r = min(r for r, _c in footprint)
    max_r = max(r for r, _c in footprint)
    min_c = min(c for _r, c in footprint)
    max_c = max(c for _r, c in footprint)

    lo_r = int(1 - min_r)
    hi_r = int(h - 2 - max_r)
    lo_c = int(1 - min_c)
    hi_c = int(w - 2 - max_c)
    if lo_r > hi_r or lo_c > hi_c:
        return False

    positions = [(r, c) for r in range(lo_r, hi_r + 1) for c in range(lo_c, hi_c + 1)]
    random.shuffle(positions)

    for top_r, top_c in positions:
        if any(bool(blocked[top_r + r, top_c + c]) for r, c in footprint):
            continue

        for rr, cc in np.argwhere(mask):
            grid[int(top_r) + int(rr), int(top_c) + int(cc)] = int(color)

        for rr, cc in footprint:
            ar = int(top_r + rr)
            ac = int(top_c + cc)
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    br = ar + dr
                    bc = ac + dc
                    if 0 <= br < h and 0 <= bc < w:
                        blocked[br, bc] = True
        return True

    return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 30)
    w = random.randint(13, 30)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    n_objects = random.randint(2, 6)
    n_convex = random.randint(1, n_objects - 1)
    kinds = ["convex"] * n_convex + ["concave"] * (n_objects - n_convex)
    random.shuffle(kinds)

    colors = random.sample(PALETTE, n_objects)
    used_forced_convex = False
    items: list[tuple[str, np.ndarray, int]] = []
    for kind, color in zip(kinds, colors):
        if kind == "convex":
            force = not used_forced_convex
            mask = _convex_mask(force_non_square=force)
            used_forced_convex = True
        else:
            mask = _concave_mask()
        items.append((kind, mask, int(color)))

    random.shuffle(items)
    items.sort(key=lambda item: int(len(_placement_footprint(item[1]))), reverse=True)

    for _kind, mask, color in items:
        if not _try_place(grid, blocked, mask, color):
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    if not bool((out == GREY).any()):
        return None
    if np.array_equal(grid, out):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
