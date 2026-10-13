import random
from collections import Counter, deque

import numpy as np


BG = 0
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
PALETTE = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), max(rows) + 1, min(cols), max(cols) + 1


def _edge_values(rect: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return rect[0, :]
    if side == "bottom":
        return rect[-1, :]
    if side == "left":
        return rect[:, 0]
    if side == "right":
        return rect[:, -1]
    raise ValueError(side)


def _is_seal_edge(edge: np.ndarray, envelope: int) -> tuple[bool, int | None]:
    others = [int(v) for v in np.unique(edge) if int(v) != envelope]
    if len(others) != 1:
        return False, None
    seal = others[0]
    parity0 = np.all(edge[0::2] == seal) and np.all(edge[1::2] == envelope)
    parity1 = np.all(edge[0::2] == envelope) and np.all(edge[1::2] == seal)
    if parity0 or parity1:
        return True, seal
    return False, None


def _find_seal(rect: np.ndarray, envelope: int) -> tuple[str, int]:
    candidates: list[tuple[str, int]] = []
    for side in SIDES:
        ok, seal = _is_seal_edge(_edge_values(rect, side), envelope)
        if ok and seal is not None:
            candidates.append((side, seal))
    assert len(candidates) == 1, f"expected one seal edge, got {candidates}"
    return candidates[0]


def _mirror_target(side: str, h: int, w: int, r: int, c: int) -> tuple[int, int]:
    if side in ("top", "bottom"):
        return h - 1 - r, c
    return r, w - 1 - c


def _pattern_half_mask(shape: tuple[int, int], side: str) -> np.ndarray:
    h, w = shape
    mask = np.zeros((h, w), dtype=bool)
    if side == "top":
        mask[0 : h // 2, :] = True
    elif side == "bottom":
        mask[h // 2 :, :] = True
    elif side == "left":
        mask[:, 0 : w // 2] = True
    elif side == "right":
        mask[:, w // 2 :] = True
    else:
        raise ValueError(side)
    return mask


def _paint_open_edge(rect: np.ndarray, side: str, envelope: int) -> None:
    if side == "top":
        rect[0, :] = envelope
    elif side == "bottom":
        rect[-1, :] = envelope
    elif side == "left":
        rect[:, 0] = envelope
    elif side == "right":
        rect[:, -1] = envelope
    else:
        raise ValueError(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open each serrated-edge envelope and mirror its half-hidden interior pattern across the fold axis.

    Input structure:
    - The background is black.
    - Each non-background object is a fully filled rectangle separated from the others by background cells.
    - Inside each rectangle, one dominant envelope color fills most cells, one other color forms an alternating serrated pattern along exactly one border, and any remaining color pixels are interior pattern pixels near that border.

    Rule:
    - Detect each rectangular envelope object independently.
    - The border carrying the alternating serrated seal determines the fold direction, and the fold axis is the rectangle's central horizontal or vertical midline.
    - Mirror every interior pattern pixel across that fold axis into the opposite half while keeping the original pattern, then replace the entire serrated border with the envelope color.

    Key insight:
    - The only asymmetric clue is the zigzag seal edge: once that edge identifies which side was folded over, the hidden half is recovered by a simple reflection across the rectangle center.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for comp in _components(g != BG):
        r0, r1, c0, c1 = _bbox(comp)
        rect = g[r0:r1, c0:c1]
        assert np.all(rect != BG), "component bbox must be a solid rectangle"

        h, w = rect.shape
        counts = Counter(int(v) for v in rect.ravel())
        envelope = counts.most_common(1)[0][0]
        side, seal = _find_seal(rect, envelope)
        if side in ("top", "bottom"):
            assert h % 2 == 0, "folded height must be even"
        else:
            assert w % 2 == 0, "folded width must be even"

        pattern_mask = (rect != envelope) & (rect != seal)
        near_half = _pattern_half_mask(rect.shape, side)
        assert np.all(~pattern_mask | near_half), "pattern must stay on the seal side"

        out_rect = out[r0:r1, c0:c1]
        _paint_open_edge(out_rect, side, envelope)

        for lr, lc in np.argwhere(pattern_mask):
            tr, tc = _mirror_target(side, h, w, int(lr), int(lc))
            out_rect[tr, tc] = rect[lr, lc]

    return out


def _sample_dimensions(rng: random.Random, side: str) -> tuple[int, int]:
    if side in ("top", "bottom"):
        h = rng.choice((4, 6, 8, 10))
        w = rng.randint(4, 10)
    else:
        h = rng.randint(4, 10)
        w = rng.choice((4, 6, 8, 10))
    return h, w


def _near_half_cells(h: int, w: int, side: str) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    if side == "top":
        rows = range(1, h // 2)
        cols = range(1, w - 1)
    elif side == "bottom":
        rows = range(h // 2, h - 1)
        cols = range(1, w - 1)
    elif side == "left":
        rows = range(1, h - 1)
        cols = range(1, w // 2)
    elif side == "right":
        rows = range(1, h - 1)
        cols = range(w // 2, w - 1)
    else:
        raise ValueError(side)
    for r in rows:
        for c in cols:
            cells.append((r, c))
    return cells


def _grow_walk(
    rng: random.Random,
    allowed: set[tuple[int, int]],
    start: tuple[int, int],
    target: int,
) -> set[tuple[int, int]]:
    pts = {start}
    frontier = [start]
    while frontier and len(pts) < target:
        base = frontier.pop(rng.randrange(len(frontier)))
        nbrs = []
        for dr, dc in DIRS4:
            cand = (base[0] + dr, base[1] + dc)
            if cand in allowed and cand not in pts:
                nbrs.append(cand)
        rng.shuffle(nbrs)
        for cand in nbrs:
            pts.add(cand)
            frontier.append(cand)
            if len(pts) >= target:
                break
        if not frontier and len(pts) < target:
            frontier = list(pts)
    return pts


def _sample_pattern(rng: random.Random, h: int, w: int, side: str) -> set[tuple[int, int]]:
    cells = _near_half_cells(h, w, side)
    allowed = set(cells)
    assert allowed

    max_cells = max(1, min(len(cells), max(2, len(cells) * 2 // 3)))
    target = rng.randint(1, max_cells)
    mode = rng.choice(("points", "walk", "blocks"))

    if mode == "points":
        return set(rng.sample(cells, target))

    if mode == "walk":
        start = rng.choice(cells)
        return _grow_walk(rng, allowed, start, target)

    pts: set[tuple[int, int]] = set()
    for _ in range(rng.randint(1, 3)):
        sr, sc = rng.choice(cells)
        bh = rng.randint(1, 2)
        bw = rng.randint(1, 3)
        for r in range(sr, sr + bh):
            for c in range(sc, sc + bw):
                if (r, c) in allowed:
                    pts.add((r, c))
                    if len(pts) >= target:
                        return pts
    if not pts:
        pts.add(rng.choice(cells))
    while len(pts) < target:
        pts |= _grow_walk(rng, allowed, rng.choice(tuple(pts)), target)
    return pts


def _paint_object(
    grid: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    side: str,
    envelope: int,
    seal: int,
    pattern: int,
    pattern_cells: set[tuple[int, int]],
) -> None:
    grid[top : top + h, left : left + w] = envelope
    rect = grid[top : top + h, left : left + w]

    edge = _edge_values(rect, side)
    edge[0::2] = seal
    edge[1::2] = envelope

    for r, c in pattern_cells:
        rect[r, c] = pattern


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    h0, w0 = occupied.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h0, top + h + 1)
    c1 = min(w0, left + w + 1)
    return not occupied[r0:r1, c0:c1].any()


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        out = main(grid)
    except AssertionError:
        return False
    if grid.shape != out.shape:
        return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 30)
    w = rng.randint(14, 30)
    grid = np.zeros((h, w), dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    num_objects = rng.randint(1, 4)
    placed = 0
    for _ in range(num_objects):
        success = False
        for _ in range(120):
            side = rng.choice(SIDES)
            oh, ow = _sample_dimensions(rng, side)
            if oh + 2 > h or ow + 2 > w:
                continue
            top = rng.randint(1, h - oh - 1)
            left = rng.randint(1, w - ow - 1)
            if not _can_place(occupied, top, left, oh, ow):
                continue

            colors = rng.sample(PALETTE, 3)
            pattern_cells = _sample_pattern(rng, oh, ow, side)
            if not pattern_cells:
                continue

            _paint_object(
                grid,
                top,
                left,
                oh,
                ow,
                side,
                envelope=colors[0],
                seal=colors[1],
                pattern=colors[2],
                pattern_cells=pattern_cells,
            )
            occupied[top : top + oh, left : left + ow] = True
            placed += 1
            success = True
            break
        if not success:
            break

    if placed == 0:
        return None
    if rng.random() < 0.45 and placed < 2:
        return None
    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
