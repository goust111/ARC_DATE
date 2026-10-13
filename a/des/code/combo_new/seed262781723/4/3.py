from __future__ import annotations

from collections import Counter

import numpy as np


_RNG = np.random.default_rng()
_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_SIDE_PRIORITY = ["top", "right", "bottom", "left"]
_OPPOSITE = {
    "top": "bottom",
    "right": "left",
    "bottom": "top",
    "left": "right",
}


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in _DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _uniform_sides(arr: np.ndarray, bg: int | None = None) -> list[str]:
    edges = {
        "top": arr[0, :],
        "right": arr[:, -1],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
    }
    out: list[str] = []
    for side in _SIDE_PRIORITY:
        vals = list(map(int, edges[side]))
        if len(set(vals)) != 1:
            continue
        if bg is not None and vals[0] == int(bg):
            continue
        out.append(side)
    return out


def _rotated_side(side: str, k: int) -> str:
    order = ["top", "left", "bottom", "right"]
    idx = order.index(side)
    return order[(idx + k) % 4]


def _rotation_to_side(src: str, dst: str) -> int:
    for k in range(4):
        if _rotated_side(src, k) == dst:
            return k
    raise ValueError(f"cannot rotate {src} to {dst}")


def _anchor_on_side(shape: tuple[int, int], side: str) -> tuple[int, int]:
    h, w = map(int, shape)
    if side == "top":
        return 0, w // 2
    if side == "bottom":
        return h - 1, w // 2
    if side == "left":
        return h // 2, 0
    return h // 2, w - 1


def _paste(dst: np.ndarray, src: np.ndarray, r0: int, c0: int) -> None:
    h, w = dst.shape
    sh, sw = src.shape
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(h, int(r0 + sh))
    cc1 = min(w, int(c0 + sw))
    if rr0 >= rr1 or cc0 >= cc1:
        return
    dst[rr0:rr1, cc0:cc1] = src[rr0 - r0 : rr1 - r0, cc0 - c0 : cc1 - c0]


def _exposed_side(mask: np.ndarray, r: int, c: int) -> str | None:
    h, w = mask.shape
    exposed: list[str] = []
    if r == 0 or not bool(mask[r - 1, c]):
        exposed.append("top")
    if c == w - 1 or not bool(mask[r, c + 1]):
        exposed.append("right")
    if r == h - 1 or not bool(mask[r + 1, c]):
        exposed.append("bottom")
    if c == 0 or not bool(mask[r, c - 1]):
        exposed.append("left")
    for side in _SIDE_PRIORITY:
        if side in exposed:
            return side
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cover a large solid polygon's boundary with rotated copies of a corner stamp.

    Input structure:
    - The background is a dominant grey-like color.
    - One large 4-connected monochrome polygon occupies the main interior of the grid.
    - A smaller rectangular two-color stamp sits in one corner, separated from the polygon.
    - One edge of the stamp's rectangle is a uniform-color base edge.

    Rule:
    - Detect the large polygon and the small stamp prototype.
    - Find every polygon pixel that has at least one non-polygon 4-neighbor; this is a boundary pixel.
    - Classify each boundary pixel by its exposed side with priority top > right > bottom > left.
    - For each boundary pixel, align the center of the stamp's base edge to that pixel, rotate the stamp so the base edge faces the polygon interior, and paint it; later stamps overwrite earlier ones in left-to-right then top-to-bottom order.

    Key insight:
    - The corner object is not part of the result scene; it is a reusable stamp whose unique base edge determines how each rotated copy attaches to the polygon boundary.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    comps = _components(g != bg)
    assert len(comps) >= 2, "expected one large polygon and one separate stamp"

    comps.sort(key=len, reverse=True)
    poly_coords = comps[0]
    stamp_coords = [pt for comp in comps[1:] for pt in comp]
    assert stamp_coords, "expected a stamp component"

    poly_mask = np.zeros_like(g, dtype=bool)
    for r, c in poly_coords:
        poly_mask[r, c] = True
    poly_color = int(Counter(int(g[r, c]) for r, c in poly_coords).most_common(1)[0][0])

    sr0, sc0, sr1, sc1 = _bbox(stamp_coords)
    stamp = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
    base_candidates = _uniform_sides(stamp, bg=bg)
    assert base_candidates, "expected a unique uniform stamp base edge"
    base_side = base_candidates[0]

    out = np.full_like(g, bg)
    out[poly_mask] = poly_color

    boundary = [(r, c) for r, c in poly_coords if _exposed_side(poly_mask, r, c) is not None]
    for r, c in sorted(boundary, key=lambda pt: (pt[1], pt[0])):
        exposed = _exposed_side(poly_mask, r, c)
        assert exposed is not None
        target_base_side = _OPPOSITE[exposed]
        k = _rotation_to_side(base_side, target_base_side)
        rotated = np.rot90(stamp, k)
        ar, ac = _anchor_on_side(rotated.shape, target_base_side)
        _paste(out, rotated, r - ar, c - ac)

    return out


def _edge_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    edge = np.zeros_like(mask, dtype=bool)
    pts = np.argwhere(mask)
    for r, c in pts:
        for dr, dc in _DIRS:
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w) or not bool(mask[nr, nc]):
                edge[r, c] = True
                break
    return edge


def _is_asymmetric(arr: np.ndarray) -> bool:
    for k in range(1, 4):
        rot = np.rot90(arr, k)
        if rot.shape == arr.shape and np.array_equal(arr, rot):
            return False
    if np.array_equal(arr, np.fliplr(arr)):
        return False
    if np.array_equal(arr, np.flipud(arr)):
        return False
    return True


def _make_canonical_stamp(base_color: int, accent_color: int) -> np.ndarray:
    for _ in range(200):
        h = int(_RNG.integers(2, 5))
        w = int(_RNG.choice(np.array([3, 5], dtype=int)))
        stamp = np.full((h, w), accent_color, dtype=int)
        stamp[-1, :] = base_color
        if h > 1:
            top_mask = _RNG.integers(0, 2, size=(h - 1, w), dtype=int)
            stamp[:-1, :] = np.where(top_mask == 0, accent_color, base_color)
        if not bool((stamp[:-1, :] == base_color).any()):
            continue
        if not bool((stamp[:-1, :] == accent_color).any()):
            continue
        if len(_uniform_sides(stamp)) != 1 or _uniform_sides(stamp)[0] != "bottom":
            continue
        if not _is_asymmetric(stamp):
            continue
        return stamp
    raise RuntimeError("failed to build a valid stamp")


def _random_poly_mask(height: int, width: int) -> np.ndarray | None:
    min_width = max(4, width // 3)
    if min_width >= width:
        return None

    span = int(_RNG.integers(min_width, width))
    left = int(_RNG.integers(0, width - span + 1))
    right = left + span - 1
    intervals = [(left, right)]

    for _ in range(1, height):
        found = False
        for _ in range(80):
            nl = int(np.clip(left + int(_RNG.integers(-2, 3)), 0, width - min_width))
            nr_min = nl + min_width - 1
            nr = int(np.clip(right + int(_RNG.integers(-2, 3)), nr_min, width - 1))
            if max(nl, left) <= min(nr, right):
                left, right = nl, nr
                intervals.append((left, right))
                found = True
                break
        if not found:
            return None

    if len({l for l, _ in intervals}) == 1 and len({r for _, r in intervals}) == 1:
        return None

    mask = np.zeros((height, width), dtype=bool)
    for r, (l, rr) in enumerate(intervals):
        mask[r, l : rr + 1] = True

    if bool(_RNG.integers(0, 2)):
        mask = mask.T

    if _components(mask).__len__() != 1:
        return None

    edge = _edge_mask(mask)
    area = int(mask.sum())
    interior = int((mask & ~edge).sum())
    if area < 35 or interior < 4:
        return None
    return mask


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, int(r0 - pad)),
        max(0, int(c0 - pad)),
        min(h - 1, int(r1 + pad)),
        min(w - 1, int(c1 + pad)),
    )


def _corner_position(grid_h: int, grid_w: int, stamp_h: int, stamp_w: int, corner: str) -> tuple[int, int]:
    if corner == "tl":
        return 1, 1
    if corner == "tr":
        return 1, grid_w - stamp_w - 1
    if corner == "bl":
        return grid_h - stamp_h - 1, 1
    return grid_h - stamp_h - 1, grid_w - stamp_w - 1


def _gen_attempt() -> np.ndarray | None:
    bg = 5
    color_pool = np.array([0, 1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
    poly_color, base_color, accent_color = map(int, _RNG.choice(color_pool, size=3, replace=False))

    canonical = _make_canonical_stamp(base_color, accent_color)
    stamp = np.rot90(canonical, int(_RNG.integers(0, 4)))
    stamp_h, stamp_w = map(int, stamp.shape)
    reach = max(stamp_h, stamp_w)
    pad = reach + 1

    grid_h = int(_RNG.integers(max(20, 2 * pad + 8), 31))
    grid_w = int(_RNG.integers(max(20, 2 * pad + 8), 31))
    corner = str(_RNG.choice(np.array(["tl", "tr", "bl", "br"])))
    sr0, sc0 = _corner_position(grid_h, grid_w, stamp_h, stamp_w, corner)
    stamp_box = (sr0, sc0, sr0 + stamp_h - 1, sc0 + stamp_w - 1)

    for _ in range(200):
        mask_h = int(_RNG.integers(7, min(15, grid_h - 2 * pad) + 1))
        mask_w = int(_RNG.integers(7, min(15, grid_w - 2 * pad) + 1))
        poly_mask_local = _random_poly_mask(mask_h, mask_w)
        if poly_mask_local is None:
            continue
        ph, pw = map(int, poly_mask_local.shape)
        if grid_h - 2 * pad < ph or grid_w - 2 * pad < pw:
            continue

        pr0 = int(_RNG.integers(pad, grid_h - pad - ph + 1))
        pc0 = int(_RNG.integers(pad, grid_w - pad - pw + 1))
        poly_box = (pr0, pc0, pr0 + ph - 1, pc0 + pw - 1)
        if _boxes_intersect(_expand_box(poly_box, reach, grid_h, grid_w), _expand_box(stamp_box, 1, grid_h, grid_w)):
            continue

        grid = np.full((grid_h, grid_w), bg, dtype=int)
        grid[pr0 : pr0 + ph, pc0 : pc0 + pw][poly_mask_local] = poly_color
        grid[sr0 : sr0 + stamp_h, sc0 : sc0 + stamp_w] = stamp

        comps = _components(grid != bg)
        if len(comps) != 2:
            continue
        comps.sort(key=len, reverse=True)
        if len(comps[0]) <= len(comps[1]):
            continue

        poly_global = np.zeros_like(grid, dtype=bool)
        for r, c in comps[0]:
            poly_global[r, c] = True
        edge = _edge_mask(poly_global)
        if int((poly_global & ~edge).sum()) < 4:
            continue

        out = main(grid)
        if out.shape != grid.shape:
            continue
        if not bool((out == base_color).any()) or not bool((out == accent_color).any()):
            continue
        if not bool((out == poly_color).any()):
            continue
        prototype_area = out[sr0 : sr0 + stamp_h, sc0 : sc0 + stamp_w]
        if not bool((prototype_area == bg).all()):
            continue
        return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
