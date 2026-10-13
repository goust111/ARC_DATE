from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
PAIR_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
GEN_ATTEMPTS = 400

HOLE_PATTERNS = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1)),
)


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            arr = np.asarray(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[arr[:, 0] - r0, arr[:, 1] - c0] = True
            comps.append({"coords": arr, "bbox": (r0, c0, r1, c1), "mask": local})

    return comps


def _hole_mask(local_mask: np.ndarray) -> np.ndarray:
    h, w = local_mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes = np.zeros((h, w), dtype=bool)

    for r in range(h):
        for c in range(w):
            if local_mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if not local_mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            if not touches_border:
                for rr, cc in cells:
                    holes[rr, cc] = True

    return holes


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[int(np.argmax(counts))])


def _yellow_owner(grid: np.ndarray, bg: int) -> tuple[int | None, tuple[int, int] | None]:
    yellow = np.argwhere(grid == YELLOW)
    assert yellow.shape == (1, 2), "expected exactly one yellow marker"
    yr, yc = map(int, yellow[0])
    neighbors = {
        int(grid[nr, nc])
        for nr, nc in _neighbors4(yr, yc, *grid.shape)
        if int(grid[nr, nc]) not in (bg, YELLOW)
    }
    assert len(neighbors) == 1, "yellow marker must belong to a unique source object"
    return next(iter(neighbors)), (yr, yc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color source/target objects, use the target hole layout to copy the aligned
    source subregion into the target, then erase the source and the yellow marker.
    """
    g = np.asarray(grid, dtype=int)
    bg = BG if np.any(g == BG) else _most_common_color(g)
    yellow_owner, yellow_pos = _yellow_owner(g, bg)
    yr, yc = yellow_pos

    out = g.copy()
    pair_colors = [int(c) for c in np.unique(g) if int(c) not in (bg, YELLOW)]
    assert pair_colors, "expected at least one colored pair"

    for color in pair_colors:
        mask = g == color
        if color == yellow_owner:
            mask[yr, yc] = True

        comps = _components(mask)
        assert len(comps) == 2, "each pair color must form exactly one source and one target"

        enriched: list[tuple[dict, np.ndarray]] = []
        for comp in comps:
            holes = _hole_mask(comp["mask"])
            enriched.append((comp, holes))

        source = [(comp, holes) for comp, holes in enriched if not np.any(holes)]
        target = [(comp, holes) for comp, holes in enriched if np.any(holes)]
        assert len(source) == 1 and len(target) == 1, "need one solid source and one hollow target"
        src, _ = source[0]
        tgt, tgt_holes = target[0]

        assert src["mask"].shape == tgt["mask"].shape, "paired shapes must share the same bbox"
        assert np.array_equal(src["mask"], tgt["mask"] | tgt_holes), "outer contours must match"

        sr0, sc0, sr1, sc1 = src["bbox"]
        tr0, tc0, tr1, tc1 = tgt["bbox"]
        src_crop = g[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        src_crop[src_crop == YELLOW] = color
        assert np.all(np.isin(src_crop[tgt_holes], (color,)))

        tgt_slice = out[tr0 : tr1 + 1, tc0 : tc1 + 1]
        assert np.all(tgt_slice[tgt_holes] == bg)
        tgt_slice[tgt_holes] = src_crop[tgt_holes]

        out[src["coords"][:, 0], src["coords"][:, 1]] = bg

    out[out == YELLOW] = bg
    return out


def _paint_tab(mask: np.ndarray, base_box: tuple[int, int, int, int], side: str) -> None:
    r0, c0, h, w = base_box
    if side == "top":
        depth = random.randint(1, r0)
        span = random.randint(2, w)
        start = c0 + random.randint(0, w - span)
        mask[r0 - depth : r0, start : start + span] = True
    elif side == "bottom":
        depth = random.randint(1, mask.shape[0] - (r0 + h))
        span = random.randint(2, w)
        start = c0 + random.randint(0, w - span)
        mask[r0 + h : r0 + h + depth, start : start + span] = True
    elif side == "left":
        depth = random.randint(1, c0)
        span = random.randint(2, h)
        start = r0 + random.randint(0, h - span)
        mask[start : start + span, c0 - depth : c0] = True
    else:
        depth = random.randint(1, mask.shape[1] - (c0 + w))
        span = random.randint(2, h)
        start = r0 + random.randint(0, h - span)
        mask[start : start + span, c0 + w : c0 + w + depth] = True


def _gen_outer_shape() -> tuple[np.ndarray, tuple[int, int, int, int]]:
    base_h = random.randint(5, 8)
    base_w = random.randint(5, 8)
    top = random.randint(0, 2)
    bottom = random.randint(0, 2)
    left = random.randint(0, 2)
    right = random.randint(0, 2)

    mask = np.zeros((base_h + top + bottom, base_w + left + right), dtype=bool)
    base_box = (top, left, base_h, base_w)
    r0, c0, h, w = base_box
    mask[r0 : r0 + h, c0 : c0 + w] = True

    sides = []
    if top > 0:
        sides.append("top")
    if bottom > 0:
        sides.append("bottom")
    if left > 0:
        sides.append("left")
    if right > 0:
        sides.append("right")

    random.shuffle(sides)
    painted = False
    for side in sides:
        if random.random() < 0.75:
            _paint_tab(mask, base_box, side)
            painted = True
    if sides and not painted:
        _paint_tab(mask, base_box, random.choice(sides))

    return mask, base_box


def _pattern_shape(pattern: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [r for r, _ in pattern]
    cols = [c for _, c in pattern]
    return max(rows) + 1, max(cols) + 1


def _gen_target_mask(source_mask: np.ndarray, base_box: tuple[int, int, int, int]) -> np.ndarray | None:
    target_mask = source_mask.copy()
    forbidden = np.zeros_like(source_mask, dtype=bool)
    r0, c0, h, w = base_box
    desired_holes = random.choice((1, 2, 3))
    placed = 0

    for _ in range(200):
        if placed == desired_holes:
            break
        pattern = random.choice(HOLE_PATTERNS)
        ph, pw = _pattern_shape(pattern)
        if h - ph < 2 or w - pw < 2:
            continue
        rr = random.randint(r0 + 1, r0 + h - ph - 1)
        cc = random.randint(c0 + 1, c0 + w - pw - 1)
        cells = [(rr + dr, cc + dc) for dr, dc in pattern]
        if any(forbidden[r, c] for r, c in cells):
            continue
        for r, c in cells:
            target_mask[r, c] = False
        for r, c in cells:
            for nr in range(r - 1, r + 2):
                for nc in range(c - 1, c + 2):
                    if 0 <= nr < target_mask.shape[0] and 0 <= nc < target_mask.shape[1]:
                        forbidden[nr, nc] = True
        placed += 1

    if placed == 0 or len(_components(target_mask)) != 1:
        return None
    if not np.any(_hole_mask(target_mask)):
        return None
    return target_mask


def _transform_pair(source_mask: np.ndarray, target_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if random.random() < 0.5:
        source_mask = np.flipud(source_mask)
        target_mask = np.flipud(target_mask)
    if random.random() < 0.5:
        source_mask = np.fliplr(source_mask)
        target_mask = np.fliplr(target_mask)
    if random.random() < 0.5:
        source_mask = source_mask.T
        target_mask = target_mask.T
    return np.ascontiguousarray(source_mask), np.ascontiguousarray(target_mask)


def _marker_candidates(mask: np.ndarray) -> list[tuple[int, int]]:
    h, w = mask.shape
    candidates: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            continue
        if not all(mask[nr, nc] for nr, nc in _neighbors4(r, c, h, w)):
            continue
        probe = mask.copy()
        probe[r, c] = False
        if len(_components(probe)) == 1:
            candidates.append((r, c))
    return candidates


def _paint_patch(mask: np.ndarray, color: int, marker: tuple[int, int] | None = None) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    if marker is not None:
        patch[marker] = YELLOW
    return patch


def _place_patches(patches: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(int(p.shape[0]) for p in patches) + 2
    max_w = max(int(p.shape[1]) for p in patches) + 2
    need_area = sum((int(p.shape[0]) + 2) * (int(p.shape[1]) + 2) for p in patches)

    for _ in range(150):
        h_lo = max(18, max_h)
        w_lo = max(18, max_w)
        h = random.randint(h_lo, max(h_lo, 30))
        w = random.randint(w_lo, max(w_lo, 30))
        if h * w < need_area + 20:
            continue

        grid = np.full((h, w), BG, dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        order = list(range(len(patches)))
        random.shuffle(order)

        ok = True
        for idx in order:
            patch = patches[idx]
            ph, pw = map(int, patch.shape)
            mask = patch != BG
            placed = False
            for _ in range(250):
                r0 = random.randint(0, h - ph)
                c0 = random.randint(0, w - pw)
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(h, r0 + ph + 1)
                cc1 = min(w, c0 + pw + 1)
                if np.any(occupied[rr0:rr1, cc0:cc1]):
                    continue
                sub = grid[r0 : r0 + ph, c0 : c0 + pw]
                if np.any(sub[mask] != BG):
                    continue
                sub[mask] = patch[mask]
                occupied[rr0:rr1, cc0:cc1] = True
                placed = True
                break
            if not placed:
                ok = False
                break

        if ok:
            return grid

    return None


def _gen_pair(color: int, with_marker: bool) -> tuple[np.ndarray, np.ndarray] | None:
    source_mask, base_box = _gen_outer_shape()
    target_mask = _gen_target_mask(source_mask, base_box)
    if target_mask is None:
        return None
    source_mask, target_mask = _transform_pair(source_mask, target_mask)

    marker = None
    if with_marker:
        candidates = _marker_candidates(source_mask)
        if not candidates:
            return None
        marker = random.choice(candidates)

    source_patch = _paint_patch(source_mask, color, marker)
    target_patch = _paint_patch(target_mask, color)

    if len(_components(source_patch == color)) != 1 or len(_components(target_patch == color)) != 1:
        return None
    if with_marker:
        yr, yc = marker
        touching = {
            int(source_patch[nr, nc])
            for nr, nc in _neighbors4(yr, yc, *source_patch.shape)
            if int(source_patch[nr, nc]) not in (BG, YELLOW)
        }
        if touching != {color}:
            return None
    return source_patch, target_patch


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.choice((1, 2, 3))
    colors = random.sample(list(PAIR_COLORS), pair_count)
    marker_pair = random.randrange(pair_count)
    patches: list[np.ndarray] = []

    for idx, color in enumerate(colors):
        pair = _gen_pair(color, with_marker=idx == marker_pair)
        if pair is None:
            return None
        source_patch, target_patch = pair
        patches.extend([source_patch, target_patch])

    grid = _place_patches(patches)
    if grid is None:
        return None

    if np.count_nonzero(grid == YELLOW) != 1:
        return None
    for color in colors:
        if len(_components(grid == color)) != 2:
            return None

    try:
        out = main(grid)
    except Exception:
        return None

    if np.count_nonzero(out == YELLOW) != 0:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
