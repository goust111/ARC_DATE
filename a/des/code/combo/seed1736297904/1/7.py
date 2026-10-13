from __future__ import annotations

import random
from collections import deque

import numpy as np


GEN_RETRIES = 400
PLACEMENT_RETRIES = 250
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)


def _mode_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _read_legend(grid: np.ndarray, bg: int) -> list[int]:
    h, w = grid.shape
    assert w >= 3 and int(grid[0, 0]) != bg

    legend: list[int] = []
    c = 0
    while c < w and int(grid[0, c]) != bg:
        legend.append(int(grid[0, c]))
        c += 1

    assert 3 <= len(legend) <= 5
    assert len(set(legend)) == len(legend)
    if h > 1:
        assert np.all(grid[1, : len(legend)] == bg)
    return legend


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray], list[bool]]:
    h, w = mask.shape
    labels = np.full((h, w), -1, dtype=int)
    comps: list[np.ndarray] = []
    touches_border: list[bool] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or labels[r, c] != -1:
                continue

            cid = len(comps)
            q = deque([(r, c)])
            labels[r, c] = cid
            pts: list[tuple[int, int]] = []
            border = False

            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    border = True

                if rr > 0 and bool(mask[rr - 1, cc]) and labels[rr - 1, cc] == -1:
                    labels[rr - 1, cc] = cid
                    q.append((rr - 1, cc))
                if rr + 1 < h and bool(mask[rr + 1, cc]) and labels[rr + 1, cc] == -1:
                    labels[rr + 1, cc] = cid
                    q.append((rr + 1, cc))
                if cc > 0 and bool(mask[rr, cc - 1]) and labels[rr, cc - 1] == -1:
                    labels[rr, cc - 1] = cid
                    q.append((rr, cc - 1))
                if cc + 1 < w and bool(mask[rr, cc + 1]) and labels[rr, cc + 1] == -1:
                    labels[rr, cc + 1] = cid
                    q.append((rr, cc + 1))

            comps.append(np.array(pts, dtype=int))
            touches_border.append(border)

    return labels, comps, touches_border


def _exterior_background(fg_mask: np.ndarray) -> np.ndarray:
    h, w = fg_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    q = deque[tuple[int, int]]()

    for r in range(h):
        for c in (0, w - 1):
            if not bool(fg_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not bool(fg_mask[r, c]) and not bool(seen[r, c]):
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        if r > 0 and not bool(fg_mask[r - 1, c]) and not bool(seen[r - 1, c]):
            seen[r - 1, c] = True
            q.append((r - 1, c))
        if r + 1 < h and not bool(fg_mask[r + 1, c]) and not bool(seen[r + 1, c]):
            seen[r + 1, c] = True
            q.append((r + 1, c))
        if c > 0 and not bool(fg_mask[r, c - 1]) and not bool(seen[r, c - 1]):
            seen[r, c - 1] = True
            q.append((r, c - 1))
        if c + 1 < w and not bool(fg_mask[r, c + 1]) and not bool(seen[r, c + 1]):
            seen[r, c + 1] = True
            q.append((r, c + 1))

    return seen


def _foreground_depths(fg_mask: np.ndarray) -> tuple[list[int], list[np.ndarray]]:
    remaining = fg_mask.copy()
    depths: list[int] = []
    comps_out: list[np.ndarray] = []
    depth = 0

    while np.any(remaining):
        exterior = _exterior_background(remaining)
        _labels, comps, _ = _label_components(remaining)
        peel: list[np.ndarray] = []

        for pts in comps:
            exposed = False
            for r, c in pts:
                if (
                    (r > 0 and exterior[r - 1, c])
                    or (r + 1 < remaining.shape[0] and exterior[r + 1, c])
                    or (c > 0 and exterior[r, c - 1])
                    or (c + 1 < remaining.shape[1] and exterior[r, c + 1])
                ):
                    exposed = True
                    break
            if exposed:
                peel.append(pts)

        assert peel
        for pts in peel:
            remaining[pts[:, 0], pts[:, 1]] = False
            depths.append(depth)
            comps_out.append(pts)
        depth += 1

    return depths, comps_out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor nested same-color foreground regions by topological depth using the
    horizontal legend in the top-left corner, then erase that legend to bg.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    legend = _read_legend(g, bg)

    work = g.copy()
    work[0, : len(legend)] = bg
    fg_mask = work != bg
    if not np.any(fg_mask):
        return work

    depths, comps = _foreground_depths(fg_mask)
    out = work.copy()

    for depth, pts in zip(depths, comps):
        assert depth < len(legend)
        out[pts[:, 0], pts[:, 1]] = int(legend[depth])

    return out


def _tight_crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    assert len(rows) > 0
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _add_outer_bars(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    h, w = mask.shape
    pads = {
        "top": rng.randint(0, 2),
        "bottom": rng.randint(0, 2),
        "left": rng.randint(0, 2),
        "right": rng.randint(0, 2),
    }
    candidates = [side for side, extent in pads.items() if extent > 0]
    if not candidates:
        return mask.copy()

    canvas = np.zeros(
        (h + pads["top"] + pads["bottom"], w + pads["left"] + pads["right"]),
        dtype=bool,
    )
    r0 = pads["top"]
    c0 = pads["left"]
    canvas[r0 : r0 + h, c0 : c0 + w] = mask

    rng.shuffle(candidates)
    n_bars = rng.randint(0, min(2, len(candidates)))
    for side in candidates[:n_bars]:
        if side in ("top", "bottom"):
            extent = pads[side]
            min_span = max(2, w // 3)
            span = rng.randint(min_span, w)
            start = rng.randint(0, w - span)
            cc0 = c0 + start
            cc1 = cc0 + span
            if side == "top":
                canvas[:extent, cc0:cc1] = True
            else:
                canvas[r0 + h :, cc0:cc1] = True
        else:
            extent = pads[side]
            min_span = max(2, h // 3)
            span = rng.randint(min_span, h)
            start = rng.randint(0, h - span)
            rr0 = r0 + start
            rr1 = rr0 + span
            if side == "left":
                canvas[rr0:rr1, :extent] = True
            else:
                canvas[rr0:rr1, c0 + w :] = True

    return _tight_crop(canvas)


def _carve_corner_holes(
    mask: np.ndarray,
    top: int,
    bottom: int,
    left: int,
    right: int,
    rng: random.Random,
) -> None:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    if top >= 2 and left >= 2:
        candidates.append((1, 1))
    if top >= 2 and right >= 2:
        candidates.append((1, w - 2))
    if bottom >= 2 and left >= 2:
        candidates.append((h - 2, 1))
    if bottom >= 2 and right >= 2:
        candidates.append((h - 2, w - 2))

    rng.shuffle(candidates)
    n_holes = rng.randint(0, min(2, len(candidates)))
    for r, c in candidates[:n_holes]:
        mask[r, c] = False


def _make_solid_seed(rng: random.Random) -> np.ndarray:
    mask = np.ones((rng.randint(2, 5), rng.randint(2, 6)), dtype=bool)
    if rng.random() < 0.8:
        mask = _add_outer_bars(mask, rng)
    return mask


def _make_ring_seed(rng: random.Random) -> np.ndarray:
    hole_h = rng.randint(2, 5)
    hole_w = rng.randint(2, 5)
    top = rng.choice([1, 1, 2, 2, 3])
    bottom = rng.choice([1, 1, 2, 2, 3])
    left = rng.choice([1, 1, 2, 2, 3])
    right = rng.choice([1, 1, 2, 2, 3])

    mask = np.ones((hole_h + top + bottom, hole_w + left + right), dtype=bool)
    mask[top : top + hole_h, left : left + hole_w] = False
    _carve_corner_holes(mask, top, bottom, left, right, rng)
    if rng.random() < 0.8:
        mask = _add_outer_bars(mask, rng)
    return mask


def _wrap_patch(child: np.ndarray, rng: random.Random) -> np.ndarray:
    ch, cw = child.shape
    gap_top = rng.randint(1, 2)
    gap_bottom = rng.randint(1, 2)
    gap_left = rng.randint(1, 2)
    gap_right = rng.randint(1, 2)

    top = rng.choice([1, 1, 2, 2, 3])
    bottom = rng.choice([1, 1, 2, 2, 3])
    left = rng.choice([1, 1, 2, 2, 3])
    right = rng.choice([1, 1, 2, 2, 3])

    hole_h = ch + gap_top + gap_bottom
    hole_w = cw + gap_left + gap_right
    mask = np.ones((hole_h + top + bottom, hole_w + left + right), dtype=bool)
    mask[top : top + hole_h, left : left + hole_w] = False
    mask[
        top + gap_top : top + gap_top + ch,
        left + gap_left : left + gap_left + cw,
    ] = child

    _carve_corner_holes(mask, top, bottom, left, right, rng)
    if rng.random() < 0.8:
        mask = _add_outer_bars(mask, rng)
    return mask


def _random_transform(mask: np.ndarray, rng: random.Random) -> np.ndarray:
    out = mask.copy()
    if rng.random() < 0.5:
        out = out[::-1, :]
    if rng.random() < 0.5:
        out = out[:, ::-1]
    if rng.random() < 0.5:
        out = out.T
    return out.copy()


def _make_object_patch(depth: int, rng: random.Random) -> np.ndarray:
    if depth == 0 or rng.random() < 0.35:
        patch = _make_ring_seed(rng)
    else:
        patch = _make_solid_seed(rng)

    for _ in range(depth):
        patch = _wrap_patch(patch, rng)

    return _random_transform(_tight_crop(patch), rng)


def _boxes_conflict(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _place_objects(
    grid: np.ndarray,
    patches: list[np.ndarray],
    fg: int,
    rng: random.Random,
) -> bool:
    h, w = grid.shape
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[np.ndarray, int, int]] = []

    for patch in sorted(patches, key=lambda m: m.shape[0] * m.shape[1], reverse=True):
        ph, pw = patch.shape
        max_r0 = h - ph - 1
        max_c0 = w - pw - 1
        if max_r0 < 3 or max_c0 < 1:
            return False

        found = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(3, max_r0)
            c0 = rng.randint(1, max_c0)
            box = (r0, c0, r0 + ph - 1, c0 + pw - 1)
            if any(_boxes_conflict(box, other, pad=1) for other in placed_boxes):
                continue
            placements.append((patch, r0, c0))
            placed_boxes.append(box)
            found = True
            break

        if not found:
            return False

    for patch, r0, c0 in placements:
        ph, pw = patch.shape
        grid[r0 : r0 + ph, c0 : c0 + pw][patch] = fg

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_len = rng.randint(3, 5)
    color_pool = rng.sample(list(range(10)), legend_len + 2)
    bg = color_pool[0]
    fg = color_pool[1]
    legend = color_pool[2:]

    deepest = rng.randint(0, min(4, legend_len - 1))
    min_side = max(18, 12 + 4 * deepest)
    h = rng.randint(min_side, GRID_H_RANGE[1])
    w = rng.randint(min_side, GRID_W_RANGE[1])
    grid = np.full((h, w), bg, dtype=int)
    grid[0, :legend_len] = np.array(legend, dtype=int)

    max_objects = 3 if min(h, w) >= 24 and deepest <= 2 else 2
    n_objects = rng.randint(1, max_objects)
    depths = [deepest]
    for _ in range(n_objects - 1):
        depths.append(rng.randint(0, min(4, legend_len - 1)))
    rng.shuffle(depths)

    patches: list[np.ndarray] = []
    for depth in depths:
        patch = _make_object_patch(depth, rng)
        if patch.shape[0] > h - 4 or patch.shape[1] > w - 2:
            return None
        patches.append(patch)

    if not _place_objects(grid, patches, fg, rng):
        return None

    if np.any(grid[1, :legend_len] != bg):
        return None
    if not np.any(grid == fg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if not np.all(out[0, :legend_len] == bg):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-depth ARC input")
