from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
ARC_MAX = 30
MAX_GENERATE_ATTEMPTS = 600
COLOR_POOL = tuple(range(1, 10))


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return int((r1 - r0 + 1) * (c1 - c0 + 1))


def _contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    or0, oc0, or1, oc1 = outer
    ir0, ic0, ir1, ic1 = inner
    return (
        (or0, oc0, or1, oc1) != (ir0, ic0, ir1, ic1)
        and or0 <= ir0
        and oc0 <= ic0
        and or1 >= ir1
        and oc1 >= ic1
    )


def _components_by_color(g: np.ndarray, bg: int = BG) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[dict] = []

    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or int(g[sr, sc]) == bg:
                continue
            color = int(g[sr, sc])
            q: deque[tuple[int, int]] = deque([(sr, sc)])
            seen[sr, sc] = True
            pts: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                pts.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(pts, dtype=int)
            out.append({"color": color, "coords": coords, "bbox": _bbox_coords(coords)})

    return out


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        count += 1
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        while q:
            r, c = q.popleft()
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    q.append((nr, nc))

    return count


def _crop(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size, "cannot crop an empty mask"
    r0, c0, r1, c1 = _bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _erode_once(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    out = np.ones_like(mask, dtype=bool)
    h, w = mask.shape
    for dr in range(3):
        for dc in range(3):
            out &= padded[dr : dr + h, dc : dc + w]
    return out


def _core_from_component(mask: np.ndarray) -> np.ndarray:
    cur = mask.astype(bool).copy()
    assert cur.any() and _component_count(cur) == 1, "expected one connected component"

    while True:
        nxt = _erode_once(cur)
        if not nxt.any() or _component_count(nxt) != 1:
            return _crop(cur)
        cur = nxt


def _center_paste(out: np.ndarray, mask: np.ndarray, color: int) -> None:
    oh, ow = out.shape
    mh, mw = mask.shape
    assert mh <= oh and mw <= ow, "inner core must fit inside the outer core bbox"
    r0 = (oh - mh) // 2
    c0 = (ow - mw) // 2
    out[r0 : r0 + mh, c0 : c0 + mw][mask] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel each colored region to its core and redraw the nested cores from outside to inside.

    Input structure:
    - The background is black.
    - Several non-black colors form separate monochromatic 4-connected regions.
    - Region bounding boxes form one visible containment hierarchy, with an outer
      region's bbox enclosing each deeper region's bbox.

    Rule:
    - For each region, repeatedly apply one-pixel square erosion to its binary
      mask until another erosion would make the region disappear or split.
    - Treat the last connected mask before that failure as the region's core.
    - Sort regions by bbox containment depth from outermost to innermost.
    - Create a black canvas the size of the outermost core bbox and center-paint
      every core on it in that order, letting inner colors overwrite outer colors.

    Key insight:
    The large colored blobs are dilated shells around smaller motifs; the task is
    to strip away the dilation and then use spatial containment to stack the motifs.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _components_by_color(g, BG)
    assert components, "expected at least one colored region"

    records: list[dict] = []
    for comp in components:
        r0, c0, r1, c1 = comp["bbox"]
        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        coords = comp["coords"]
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        records.append(
            {
                "color": int(comp["color"]),
                "bbox": comp["bbox"],
                "core": _core_from_component(local),
            }
        )

    for rec in records:
        rec["depth"] = sum(1 for other in records if _contains(other["bbox"], rec["bbox"]))

    roots = [rec for rec in records if rec["depth"] == 0]
    assert len(roots) == 1, "expected one outermost region"
    root = roots[0]

    draw_order = sorted(records, key=lambda rec: (rec["depth"], -_bbox_area(rec["bbox"]), rec["bbox"]))
    out = np.full(root["core"].shape, BG, dtype=int)
    for rec in draw_order:
        _center_paste(out, rec["core"], int(rec["color"]))
    return out


def _dilate_once(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h + 2, w + 2), dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out[dr : dr + h, dc : dc + w] |= mask
    return out


def _dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    out = mask.astype(bool).copy()
    for _ in range(radius):
        out = _dilate_once(out)
    return out


def _draw_orthogonal_path(mask: np.ndarray, pts: list[tuple[int, int]]) -> None:
    for (r0, c0), (r1, c1) in zip(pts, pts[1:], strict=False):
        if r0 == r1:
            for c in range(min(c0, c1), max(c0, c1) + 1):
                mask[r0, c] = True
        else:
            assert c0 == c1
            for r in range(min(r0, r1), max(r0, r1) + 1):
                mask[r, c0] = True


def _rectangle_loop(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _chamfer_loop(h: int, w: int, k: int) -> np.ndarray:
    pts: list[tuple[int, int]] = [(0, k), (0, w - k - 1)]
    for i in range(k):
        pts.append((i, w - k + i))
        pts.append((i + 1, w - k + i))
    pts.append((h - k - 1, w - 1))
    for i in range(k):
        pts.append((h - k + i, w - 1 - i))
        pts.append((h - k + i, w - 2 - i))
    pts.append((h - 1, k))
    for i in range(k):
        pts.append((h - 1 - i, k - 1 - i))
        pts.append((h - 2 - i, k - 1 - i))
    pts.append((k, 0))
    for i in range(k):
        pts.append((k - 1 - i, i))
        pts.append((k - 1 - i, i + 1))
    pts.append((0, k))

    mask = np.zeros((h, w), dtype=bool)
    _draw_orthogonal_path(mask, pts)
    return mask


def _ring_core(h: int, w: int) -> np.ndarray:
    max_k = min(3, h // 4, w // 4)
    if max_k >= 2 and random.random() < 0.45:
        return _chamfer_loop(h, w, random.randint(2, max_k))
    return _rectangle_loop(h, w)


def _leaf_templates() -> list[np.ndarray]:
    plus = np.zeros((5, 5), dtype=bool)
    plus[2, :] = True
    plus[:, 2] = True

    zig = np.zeros((4, 5), dtype=bool)
    zig[0, :3] = True
    zig[:, 2] = True
    zig[3, 2:] = True

    ell = np.zeros((5, 5), dtype=bool)
    ell[:, 0] = True
    ell[-1, :] = True

    tee = np.zeros((5, 5), dtype=bool)
    tee[0, :] = True
    tee[:, 2] = True

    short = np.zeros((3, 5), dtype=bool)
    short[1, :] = True
    short[0, 1] = True
    short[2, 3] = True

    return [plus, zig, ell, tee, short]


def _leaf_core() -> np.ndarray:
    mask = random.choice(_leaf_templates()).copy()
    mask = np.rot90(mask, random.randint(0, 3))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    return _crop(mask)


def _paint_centered(base: np.ndarray, patch: np.ndarray) -> None:
    bh, bw = base.shape
    ph, pw = patch.shape
    r0 = (bh - ph) // 2
    c0 = (bw - pw) // 2
    base[r0 : r0 + ph, c0 : c0 + pw] = patch


def _expected_output(outer_to_inner: list[dict]) -> np.ndarray:
    root = outer_to_inner[0]
    out = np.full(root["core"].shape, BG, dtype=int)
    for rec in outer_to_inner:
        _center_paste(out, rec["core"], int(rec["color"]))
    return out


def _gen_attempt() -> np.ndarray | None:
    n_regions = random.choice((2, 2, 3, 3, 4))
    colors = random.sample(COLOR_POOL, n_regions)

    leaf_core = _leaf_core()
    leaf_radius = random.choice((1, 1, 2))
    leaf_mask = _dilate(leaf_core, leaf_radius)
    composite = np.full(leaf_mask.shape, BG, dtype=int)
    composite[leaf_mask] = colors[-1]
    inner_to_outer: list[dict] = [{"color": colors[-1], "core": leaf_core, "radius": leaf_radius}]

    for color in reversed(colors[:-1]):
        child_h, child_w = composite.shape
        radius = random.choice((1, 1, 2))
        gap = random.randint(1, 2)
        min_core_h = max(7, child_h + 2 * radius + 2 + 2 * gap)
        min_core_w = max(7, child_w + 2 * radius + 2 + 2 * gap)
        core_h = min_core_h + random.randint(0, 4)
        core_w = min_core_w + random.randint(0, 4)

        core = _ring_core(core_h, core_w)
        parent_mask = _dilate(core, radius)
        if max(parent_mask.shape) > ARC_MAX:
            return None

        new_composite = np.full(parent_mask.shape, BG, dtype=int)
        new_composite[parent_mask] = color
        r0 = (parent_mask.shape[0] - child_h) // 2
        c0 = (parent_mask.shape[1] - child_w) // 2
        if r0 <= 0 or c0 <= 0:
            return None
        if parent_mask[r0 : r0 + child_h, c0 : c0 + child_w].any():
            return None
        new_composite[r0 : r0 + child_h, c0 : c0 + child_w] = composite

        composite = new_composite
        inner_to_outer.append({"color": color, "core": core, "radius": radius})

    comp_h, comp_w = composite.shape
    if comp_h > ARC_MAX or comp_w > ARC_MAX:
        return None

    height = random.randint(comp_h, ARC_MAX)
    width = random.randint(comp_w, ARC_MAX)
    grid = np.full((height, width), BG, dtype=int)
    r0 = random.randint(0, height - comp_h)
    c0 = random.randint(0, width - comp_w)
    grid[r0 : r0 + comp_h, c0 : c0 + comp_w] = composite

    expected = _expected_output(list(reversed(inner_to_outer)))
    try:
        actual = main(grid)
    except Exception:
        return None
    if actual.shape != expected.shape or max(actual.shape) > ARC_MAX:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATE_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-core input")
