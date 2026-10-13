from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
ORANGE = 7

TILE_COLORS = (BLUE, RED, GREEN, GREY)
SOURCE_COLORS = (YELLOW, ORANGE)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            out.append(comp)
    return out


def _extract_source_offsets(
    grid: np.ndarray,
) -> tuple[list[tuple[int, int, int]], tuple[int, int]]:
    source_mask = np.isin(grid, SOURCE_COLORS)
    comps = _components(source_mask)
    assert len(comps) == 1, "expected one yellow/orange source component"
    source_cells = comps[0]
    red_neighbors: set[tuple[int, int]] = set()
    for r, c in source_cells:
        for nr, nc in _neighbors4(r, c):
            if _in_bounds(grid.shape, nr, nc) and int(grid[nr, nc]) == RED:
                red_neighbors.add((nr, nc))
    assert len(red_neighbors) == 1, "expected one reference red anchor"
    anchor = next(iter(red_neighbors))
    ar, ac = anchor
    offsets = sorted((r - ar, c - ac, int(grid[r, c])) for r, c in source_cells)
    return offsets, anchor


def _candidate_template(
    grid: np.ndarray, period_h: int, period_w: int
) -> np.ndarray | None:
    template = np.full((period_h, period_w), -1, dtype=int)
    visible = np.isin(grid, TILE_COLORS)
    if not np.any(visible):
        return None
    for r, c in np.argwhere(visible):
        rr = int(r) % period_h
        cc = int(c) % period_w
        color = int(grid[r, c])
        prev = int(template[rr, cc])
        if prev == -1:
            template[rr, cc] = color
        elif prev != color:
            return None
    if np.any(template == -1):
        return None
    return template


def _infer_tile_template(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    periods = sorted(
        ((ph, pw) for ph in range(1, h + 1) for pw in range(1, w + 1)),
        key=lambda p: (p[0] * p[1], p[0], p[1]),
    )
    for ph, pw in periods:
        template = _candidate_template(grid, ph, pw)
        if template is not None:
            return template
    raise AssertionError("failed to infer a full periodic tile")


def _tile_from_template(template: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    rr = np.arange(h) % template.shape[0]
    cc = np.arange(w) % template.shape[1]
    return template[np.ix_(rr, cc)]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair the hidden repeating tile, then stamp the source motif at every red tile anchor.

    Input structure:
    - A black-background grid is mostly covered by a repeating tile that uses blue, green, grey, and one fixed red residue per period.
    - Some rectangular black regions occlude tile cells, hiding parts of the lattice and some red residues.
    - One connected yellow/orange source motif sits beside a visible red tile pixel that acts as its reference anchor.

    Rule:
    - Treat black cells as unknown tile occlusions and ignore the yellow/orange motif while inferring the smallest rectangular tile period consistent with the visible tile-colored cells.
    - Tile that recovered period across the whole grid to reveal every red anchor, including the ones hidden under black rectangles.
    - Take the yellow/orange motif's non-black pixels as offsets relative to its adjacent red anchor, then stamp those colored offsets at every red tile pixel.

    Key insight:
    - The red anchors are part of the periodic tile itself, so the hidden stamp targets only become knowable after repairing the occluded lattice.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    source_offsets, _anchor = _extract_source_offsets(g)
    template = _infer_tile_template(g)
    out = _tile_from_template(template, g.shape)

    for ar, ac in np.argwhere(out == RED):
        ar = int(ar)
        ac = int(ac)
        for dr, dc, color in source_offsets:
            rr = ar + dr
            cc = ac + dc
            assert _in_bounds(out.shape, rr, cc), "stamp must stay inside the grid"
            out[rr, cc] = color
    return out


_TILE_FORMULAS = (
    (1, 1, 0),
    (1, 2, 0),
    (2, 1, 1),
    (2, 2, 2),
)

_SOURCE_PATTERNS = (
    (
        (0, 1, YELLOW),
        (-1, 1, ORANGE),
        (1, 1, ORANGE),
        (-1, 2, YELLOW),
        (0, 2, ORANGE),
        (1, 2, YELLOW),
    ),
    (
        (0, 1, ORANGE),
        (0, 2, YELLOW),
        (-1, 2, ORANGE),
        (1, 2, ORANGE),
        (-1, 3, YELLOW),
        (1, 3, YELLOW),
    ),
    (
        (0, 1, YELLOW),
        (1, 1, ORANGE),
        (-1, 2, YELLOW),
        (0, 2, ORANGE),
        (1, 2, YELLOW),
    ),
)


def _rotate_offsets(
    offsets: tuple[tuple[int, int, int], ...], turns: int
) -> list[tuple[int, int, int]]:
    rotated = [(dr, dc, color) for dr, dc, color in offsets]
    for _ in range(turns % 4):
        rotated = [(-dc, dr, color) for dr, dc, color in rotated]
    return rotated


def _make_tile(period_h: int, period_w: int) -> tuple[np.ndarray, tuple[int, int]]:
    a, b, shift = random.choice(_TILE_FORMULAS)
    cycle = random.sample([BLUE, GREEN, GREY], 3)
    tile = np.zeros((period_h, period_w), dtype=int)
    for r in range(period_h):
        for c in range(period_w):
            tile[r, c] = cycle[(a * r + b * c + shift) % 3]
    anchor_residue = (random.randrange(period_h), random.randrange(period_w))
    tile[anchor_residue] = RED
    return tile, anchor_residue


def _grid_anchors(
    shape: tuple[int, int], period: tuple[int, int], residue: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    rh, rw = residue
    return [
        (r, c)
        for r in range(rh, h, period[0])
        for c in range(rw, w, period[1])
    ]


def _tile_period_shape(template: np.ndarray) -> tuple[int, int]:
    inferred = _infer_tile_template(_tile_from_template(template, (17, 19)))
    return tuple(map(int, inferred.shape))


def _choose_reference_anchor(
    base: np.ndarray, anchors: list[tuple[int, int]], source_offsets: list[tuple[int, int, int]]
) -> tuple[int, int] | None:
    h, w = base.shape
    candidates: list[tuple[int, int, int]] = []
    for ar, ac in anchors:
        if ar < h // 2 or ac < w // 2:
            continue
        ok = True
        covered: set[tuple[int, int]] = set()
        for dr, dc, _color in source_offsets:
            rr = ar + dr
            cc = ac + dc
            if not _in_bounds(base.shape, rr, cc):
                ok = False
                break
            if int(base[rr, cc]) == RED:
                ok = False
                break
            covered.add((rr, cc))
        if not ok:
            continue
        adjacent_reds: set[tuple[int, int]] = set()
        for rr, cc in covered:
            for nr, nc in _neighbors4(rr, cc):
                if _in_bounds(base.shape, nr, nc) and int(base[nr, nc]) == RED:
                    adjacent_reds.add((nr, nc))
        if adjacent_reds != {(ar, ac)}:
            continue
        candidates.append((ar + ac, ar, ac))
    if not candidates:
        return None
    best_sum = max(score for score, _r, _c in candidates)
    best = [(r, c) for score, r, c in candidates if score == best_sum]
    return random.choice(best)


def _stamp_fits_all(
    shape: tuple[int, int],
    anchors: list[tuple[int, int]],
    source_offsets: list[tuple[int, int, int]],
) -> bool:
    for ar, ac in anchors:
        for dr, dc, _color in source_offsets:
            if not _in_bounds(shape, ar + dr, ac + dc):
                return False
    return True


def _protected_mask(
    shape: tuple[int, int],
    anchor: tuple[int, int],
    source_offsets: list[tuple[int, int, int]],
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    ar, ac = anchor
    mask[ar, ac] = True
    for dr, dc, _color in source_offsets:
        mask[ar + dr, ac + dc] = True
    return mask


def _residues_visible(
    shape: tuple[int, int],
    period: tuple[int, int],
    source_cells: set[tuple[int, int]],
    occluded: np.ndarray,
) -> bool:
    seen = np.zeros(period, dtype=bool)
    for r in range(shape[0]):
        for c in range(shape[1]):
            if occluded[r, c] or (r, c) in source_cells:
                continue
            seen[r % period[0], c % period[1]] = True
    return bool(np.all(seen))


def _place_occlusions(
    base: np.ndarray,
    period: tuple[int, int],
    anchor_residue: tuple[int, int],
    ref_anchor: tuple[int, int],
    source_offsets: list[tuple[int, int, int]],
) -> np.ndarray | None:
    h, w = base.shape
    total = h * w
    anchors = _grid_anchors(base.shape, period, anchor_residue)
    protected = _protected_mask(base.shape, ref_anchor, source_offsets)
    source_cells = {
        (ref_anchor[0] + dr, ref_anchor[1] + dc)
        for dr, dc, _color in source_offsets
    }

    for _ in range(200):
        rect_count = random.randint(2, 4)
        occluded = np.zeros((h, w), dtype=bool)
        rects: list[tuple[int, int, int, int]] = []
        for _rect_idx in range(rect_count):
            placed = False
            for _ in range(80):
                rh = random.randint(2, min(6, h))
                rw = random.randint(2, min(6, w))
                r0 = random.randint(0, h - rh)
                c0 = random.randint(0, w - rw)
                r1 = r0 + rh
                c1 = c0 + rw
                if np.any(protected[r0:r1, c0:c1]) or np.any(occluded[r0:r1, c0:c1]):
                    continue
                occluded[r0:r1, c0:c1] = True
                rects.append((r0, c0, r1, c1))
                placed = True
                break
            if not placed:
                break
        if len(rects) != rect_count:
            continue
        coverage = int(np.sum(occluded))
        if not (int(0.20 * total) <= coverage <= int(0.40 * total)):
            continue
        hidden = [pos for pos in anchors if pos != ref_anchor and occluded[pos]]
        visible = [pos for pos in anchors if pos != ref_anchor and not occluded[pos]]
        if not hidden or not visible:
            continue
        if not _residues_visible(base.shape, period, source_cells, occluded):
            continue
        return occluded
    return None


def _stamp_all_anchors(
    base: np.ndarray,
    anchor_residue: tuple[int, int],
    period: tuple[int, int],
    source_offsets: list[tuple[int, int, int]],
) -> np.ndarray:
    out = base.copy()
    for ar, ac in _grid_anchors(base.shape, period, anchor_residue):
        for dr, dc, color in source_offsets:
            out[ar + dr, ac + dc] = color
    return out


def _attempt() -> np.ndarray | None:
    period_h = random.randint(4, 5)
    period_w = random.randint(4, 5)
    tile, anchor_residue = _make_tile(period_h, period_w)
    if set(np.unique(tile)) != {BLUE, RED, GREEN, GREY}:
        return None
    if _tile_period_shape(tile) != (period_h, period_w):
        return None

    h = random.randint(max(14, period_h * 3), min(18, period_h * 4 + 1))
    w = random.randint(max(14, period_w * 3), min(18, period_w * 4 + 1))
    base = _tile_from_template(tile, (h, w))
    anchors = _grid_anchors(base.shape, (period_h, period_w), anchor_residue)
    if len(anchors) < 4:
        return None

    source_offsets: list[tuple[int, int, int]] | None = None
    ref_anchor: tuple[int, int] | None = None
    pattern_options = list(_SOURCE_PATTERNS)
    random.shuffle(pattern_options)
    turn_options = [0, 1, 2, 3]
    for pattern in pattern_options:
        random.shuffle(turn_options)
        for turns in turn_options:
            candidate_offsets = _rotate_offsets(pattern, turns)
            if not _stamp_fits_all(base.shape, anchors, candidate_offsets):
                continue
            candidate_anchor = _choose_reference_anchor(base, anchors, candidate_offsets)
            if candidate_anchor is not None:
                source_offsets = candidate_offsets
                ref_anchor = candidate_anchor
                break
        if source_offsets is not None:
            break
    if source_offsets is None or ref_anchor is None:
        return None

    occluded = _place_occlusions(
        base, (period_h, period_w), anchor_residue, ref_anchor, source_offsets
    )
    if occluded is None:
        return None

    inp = base.copy()
    for dr, dc, color in source_offsets:
        inp[ref_anchor[0] + dr, ref_anchor[1] + dc] = color
    inp[occluded] = BLACK

    expected = _stamp_all_anchors(base, anchor_residue, (period_h, period_w), source_offsets)
    try:
        pred = main(inp)
    except Exception:
        return None
    if not np.array_equal(pred, expected):
        return None
    if np.array_equal(inp, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
