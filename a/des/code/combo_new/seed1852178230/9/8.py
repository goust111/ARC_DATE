from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
MARKER_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
SIDES = ("top", "bottom", "left", "right")
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _is_nonsymmetric(mask: np.ndarray) -> bool:
    variants = [np.rot90(mask, k) for k in (1, 2, 3)]
    variants.extend(np.fliplr(np.rot90(mask, k)) for k in range(4))
    return all(mask.shape != variant.shape or not np.array_equal(mask, variant) for variant in variants)


def _make_top_shape(rng: random.Random) -> np.ndarray:
    for _ in range(200):
        h = rng.randint(3, 6)
        w = rng.randint(3, 6)
        family = rng.choice(("corner", "double_cut", "skyline"))

        if family == "corner":
            mask = np.ones((h, w), dtype=bool)
            cut_h = rng.randint(1, h - 1)
            cut_w = rng.randint(1, w - 1)
            if rng.random() < 0.5:
                mask[h - cut_h :, :cut_w] = False
            else:
                mask[h - cut_h :, w - cut_w :] = False
        elif family == "double_cut":
            mask = np.ones((h, w), dtype=bool)
            left_w = rng.randint(1, w - 2)
            right_w = rng.randint(1, w - left_w - 1)
            left_h = rng.randint(1, h - 1)
            right_h = rng.randint(1, h - 1)
            mask[h - left_h :, :left_w] = False
            mask[h - right_h :, w - right_w :] = False
        else:
            heights = [rng.randint(1, h) for _ in range(w)]
            heights[rng.randrange(w)] = h
            mask = np.zeros((h, w), dtype=bool)
            for c, height in enumerate(heights):
                mask[:height, c] = True

        if mask.sum() < 5:
            continue
        if len(_components(mask)) != 1:
            continue
        if not _is_nonsymmetric(mask):
            continue
        return mask
    raise RuntimeError("failed to generate a non-symmetric shape")


def _orient_top_shape(top_mask: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return top_mask
    if side == "bottom":
        return np.rot90(top_mask, 2)
    if side == "left":
        return np.rot90(top_mask, 1)
    if side == "right":
        return np.rot90(top_mask, 3)
    raise ValueError(side)


def _layout_for_side(h: int, w: int, side: str) -> tuple[int, int, int, int]:
    if side == "top":
        return 2 * h, w, h, 0
    if side == "bottom":
        return 2 * h, w, 0, 0
    if side == "left":
        return h, 2 * w, 0, w
    if side == "right":
        return h, 2 * w, 0, 0
    raise ValueError(side)


def _pick_marker(shape: np.ndarray, side: str, rng: random.Random) -> tuple[int, int]:
    h, w = shape.shape
    if side == "top":
        return 0, rng.randint(1, w - 2)
    if side == "bottom":
        return h - 1, rng.randint(1, w - 2)
    if side == "left":
        return rng.randint(1, h - 2), 0
    if side == "right":
        return rng.randint(1, h - 2), w - 1
    raise ValueError(side)


def _choose_sides(rng: random.Random, count: int) -> list[str]:
    if count == 4 and rng.random() < 0.7:
        sides = list(SIDES)
        rng.shuffle(sides)
        return sides
    if count >= 2 and rng.random() < 0.4:
        return rng.sample(list(SIDES), count)
    return [rng.choice(SIDES) for _ in range(count)]


def _find_placement(
    reserved: np.ndarray,
    union_h: int,
    union_w: int,
    rng: random.Random,
) -> tuple[int, int] | None:
    h, w = reserved.shape
    candidates: list[tuple[int, int]] = []
    for top in range(h - union_h + 1):
        for left in range(w - union_w + 1):
            r0 = max(0, top - 1)
            r1 = min(h, top + union_h + 1)
            c0 = max(0, left - 1)
            c1 = min(w, left + union_w + 1)
            if not reserved[r0:r1, c0:c1].any():
                candidates.append((top, left))
    if not candidates:
        return None
    return rng.choice(candidates)


def _reserve(reserved: np.ndarray, top: int, left: int, union_h: int, union_w: int) -> None:
    h, w = reserved.shape
    r0 = max(0, top - 1)
    r1 = min(h, top + union_h + 1)
    c0 = max(0, left - 1)
    c1 = min(w, left + union_w + 1)
    reserved[r0:r1, c0:c1] = True


def main(grid: np.ndarray) -> np.ndarray:
    """Reflect each marked grey object across the marked bounding-box edge and paint the copy in the marker color.

    Input structure:
    - The background is black.
    - Each foreground object is a 4-connected grey polyomino with exactly one boundary cell recolored to a non-grey marker color.
    - Objects are separated from one another, and the marker may lie on the top, bottom, left, or right outer boundary of its object.

    Rule:
    - For each foreground component, locate its unique non-grey marker cell and compute the component's bounding box.
    - Any direction whose neighboring cell is black or outside the grid is a candidate boundary side for the marker; if several directions qualify, choose the candidate whose bounding-box edge is nearest to the marker.
    - Reflect the full component across that chosen bounding-box edge onto the outside of the box, recolor every reflected cell with the marker color, and keep the original object unchanged.

    Key insight:
    - The marker carries two visible cues at once: its position selects the reflection axis, and its color paints the entire reflected copy.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    for comp in _components(g != BG):
        marker_pts = [(r, c) for r, c in comp if int(g[r, c]) != GREY]
        assert len(marker_pts) == 1, "each object must contain exactly one marker"
        mr, mc = marker_pts[0]
        marker_color = int(g[mr, mc])
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        candidates: list[tuple[int, str]] = []
        if mr == 0 or int(g[mr - 1, mc]) == BG:
            candidates.append((mr - r0, "top"))
        if mr == h - 1 or int(g[mr + 1, mc]) == BG:
            candidates.append((r1 - mr, "bottom"))
        if mc == 0 or int(g[mr, mc - 1]) == BG:
            candidates.append((mc - c0, "left"))
        if mc == w - 1 or int(g[mr, mc + 1]) == BG:
            candidates.append((c1 - mc, "right"))
        assert candidates, "marker must sit on an outer boundary"
        side = min(candidates, key=lambda item: (item[0], SIDES.index(item[1])))[1]

        reflected: list[tuple[int, int]] = []
        for r, c in comp:
            if side == "top":
                rr, cc = 2 * r0 - 1 - r, c
            elif side == "bottom":
                rr, cc = 2 * r1 + 1 - r, c
            elif side == "left":
                rr, cc = r, 2 * c0 - 1 - c
            else:
                rr, cc = r, 2 * c1 + 1 - c
            reflected.append((rr, cc))

        assert all(0 <= rr < h and 0 <= cc < w for rr, cc in reflected), "reflection must stay in bounds"
        for rr, cc in reflected:
            assert int(out[rr, cc]) == BG, "reflected copies must not overlap other content"
            out[rr, cc] = marker_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    object_count = rng.choice((1, 2, 2, 3, 3, 4))
    min_size = 10 + 4 * object_count
    h = rng.randint(min_size, 30)
    w = rng.randint(min_size, 30)

    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    sides = _choose_sides(rng, object_count)
    marker_colors = rng.sample(MARKER_COLORS, object_count)

    for side, marker_color in zip(sides, marker_colors):
        shape = _orient_top_shape(_make_top_shape(rng), side)
        sh, sw = shape.shape
        union_h, union_w, orig_r0, orig_c0 = _layout_for_side(sh, sw, side)
        placement = _find_placement(reserved, union_h, union_w, rng)
        if placement is None:
            return None
        top, left = placement
        marker_r, marker_c = _pick_marker(shape, side, rng)

        for r, c in np.argwhere(shape):
            rr = top + orig_r0 + int(r)
            cc = left + orig_c0 + int(c)
            grid[rr, cc] = GREY

        grid[top + orig_r0 + marker_r, left + orig_c0 + marker_c] = marker_color
        _reserve(reserved, top, left, union_h, union_w)

    if len(_components(grid != BG)) != object_count:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
