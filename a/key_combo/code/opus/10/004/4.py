import random
from typing import Iterable

import numpy as np


BG = 5
PANEL = 1
OBJ = 2
SIDE_TO_COLOR = {
    "top": 3,
    "right": 4,
    "bottom": 7,
    "left": 6,
}
SIDE_ORDER = ("top", "right", "bottom", "left")


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _panel_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    bg = int(np.bincount(grid.ravel()).argmax())
    box = _bbox(grid != bg)
    assert box is not None, "expected a foreground panel"
    r0, c0, r1, c1 = box
    assert r0 > 0 and c0 > 0 and r1 < grid.shape[0] - 1 and c1 < grid.shape[1] - 1
    return box


def _canonical_quadrant(grid: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _panel_bbox(grid)
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = panel.shape
    assert h % 2 == 0 and w % 2 == 0, "panel dimensions must be even"
    qh, qw = h // 2, w // 2
    ul = panel[:qh, :qw]
    assert np.array_equal(panel[:qh, qw:], np.fliplr(ul)), "top quadrants must mirror horizontally"
    assert np.array_equal(panel[qh:, :qw], np.flipud(ul)), "left quadrants must mirror vertically"
    assert np.array_equal(panel[qh:, qw:], np.flipud(np.fliplr(ul))), "bottom-right quadrant must be 180-degree reflected"
    return ul.copy()


def _contact_side(coords: np.ndarray, h: int, w: int) -> str:
    sides = []
    if np.any(coords[:, 0] == 0):
        sides.append("top")
    if np.any(coords[:, 1] == w - 1):
        sides.append("right")
    if np.any(coords[:, 0] == h - 1):
        sides.append("bottom")
    if np.any(coords[:, 1] == 0):
        sides.append("left")
    assert len(sides) == 1, "each object must touch exactly one quadrant side"
    return sides[0]


def _recolor_quadrant(quadrant: np.ndarray) -> np.ndarray:
    q = np.asarray(quadrant, dtype=int)
    h, w = q.shape
    out = q.copy()
    comps = _components(q == OBJ)
    assert 2 <= len(comps) <= 4, "expected 2 to 4 red objects in the canonical quadrant"
    seen_sides = set()
    for comp in comps:
        assert 2 <= len(comp) <= 5, "object size must be 2 to 5"
        side = _contact_side(comp, h, w)
        assert side not in seen_sides, "objects in a quadrant must touch different sides"
        seen_sides.add(side)
        out[comp[:, 0], comp[:, 1]] = SIDE_TO_COLOR[side]
    assert np.any(out != q), "expected at least one recolor"
    return out


def _mirror_panel(quadrant: np.ndarray) -> np.ndarray:
    top = np.concatenate([quadrant, np.fliplr(quadrant)], axis=1)
    bottom = np.concatenate([np.flipud(quadrant), np.flipud(np.fliplr(quadrant))], axis=1)
    return np.concatenate([top, bottom], axis=0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Crop a four-way mirrored panel to its canonical quadrant and recolor each boundary-touching object by the side it touches.

    Input structure:
    - The grid has a grey background containing one larger blue foreground panel.
    - The panel has even height and width, and its content forms four reflective quadrants.
    - Red 4-connected objects are embedded in the panel; in the upper-left quadrant each such object touches exactly one quadrant side.

    Rule:
    - Crop to the foreground panel's bounding box and verify four-way reflective symmetry.
    - Extract the upper-left quadrant as the canonical quadrant.
    - Find each red 4-connected object in that quadrant and identify whether it touches the top, right, bottom, or left edge of the quadrant.
    - Recolor touched-top objects to green, touched-right to yellow, touched-bottom to orange, and touched-left to magenta; keep blue pixels unchanged.

    Key insight:
    - The relevant side markers are implicit: the quadrant boundaries created by the symmetry crop determine each object's output color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    quadrant = _canonical_quadrant(g)
    return _recolor_quadrant(quadrant)


TOP_TEMPLATES = [
    {(0, 0), (1, 0)},
    {(0, 0), (1, 0), (2, 0)},
    {(0, 0), (1, 0), (1, -1)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (1, -1)},
    {(0, 0), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, -1), (2, -1)},
    {(0, 0), (1, 0), (2, 0), (1, -1), (1, 1)},
    {(0, 0), (1, 0), (2, 0), (2, -1), (1, -1)},
    {(0, 0), (1, 0), (2, 0), (2, 1), (1, 1)},
]


def _side_cells(side: str, anchor: int, template: Iterable[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    cells = set()
    for dr, dc in template:
        if side == "top":
            rr, cc = dr, anchor + dc
        elif side == "bottom":
            rr, cc = h - 1 - dr, anchor + dc
        elif side == "left":
            rr, cc = anchor + dc, dr
        elif side == "right":
            rr, cc = anchor + dc, w - 1 - dr
        else:
            raise ValueError(side)
        cells.add((int(rr), int(cc)))
    return cells


def _touches_one_side(cells: set[tuple[int, int]], h: int, w: int, side: str) -> bool:
    coords = np.array(sorted(cells), dtype=int)
    if coords.size == 0:
        return False
    try:
        return _contact_side(coords, h, w) == side
    except AssertionError:
        return False


def _far_enough(cells: set[tuple[int, int]], occupied: set[tuple[int, int]]) -> bool:
    for r, c in cells:
        if (r, c) in occupied:
            return False
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if (nr, nc) in occupied:
                return False
    return True


def _candidate_anchors(side: str, template: Iterable[tuple[int, int]], h: int, w: int) -> list[int]:
    dcs = [dc for _dr, dc in template]
    low, high = min(dcs), max(dcs)
    if side in ("top", "bottom"):
        edge_len = w
    else:
        edge_len = h
    start = 1 - low
    stop = edge_len - 2 - high
    return list(range(start, stop + 1))


def _choose_sides() -> tuple[str, ...]:
    roll = random.random()
    if roll < 0.55:
        return SIDE_ORDER
    if roll < 0.85:
        if random.random() < 0.5:
            return tuple(random.sample(("top", "right", "bottom"), 3))
        return tuple(random.sample(("right", "bottom", "left"), 3))
    outer = random.choice(("top", "left"))
    inner = random.choice(("right", "bottom"))
    return (outer, inner)


def _build_quadrant() -> np.ndarray | None:
    h = random.randint(6, 8)
    w = random.randint(6, 8)
    q = np.full((h, w), PANEL, dtype=int)
    occupied: set[tuple[int, int]] = set()

    chosen_sides = list(_choose_sides())
    random.shuffle(chosen_sides)
    for side in chosen_sides:
        candidates = []
        for template in TOP_TEMPLATES:
            anchors = _candidate_anchors(side, template, h, w)
            random.shuffle(anchors)
            for anchor in anchors:
                cells = _side_cells(side, anchor, template, h, w)
                if not _touches_one_side(cells, h, w, side):
                    continue
                if not _far_enough(cells, occupied):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        cells = random.choice(candidates)
        for r, c in cells:
            q[r, c] = OBJ
        occupied |= cells

    comps = _components(q == OBJ)
    if len(comps) != len(chosen_sides):
        return None
    return q


def _validate_generated(grid: np.ndarray, expected: np.ndarray) -> None:
    box = _panel_bbox(grid)
    r0, c0, r1, c1 = box
    panel = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert 12 <= panel.shape[0] <= 16 and 12 <= panel.shape[1] <= 16

    q = _canonical_quadrant(grid)
    assert np.array_equal(_mirror_panel(q), panel), "generator must create exact mirrored panel"

    comps = _components(q == OBJ)
    assert 2 <= len(comps) <= 4
    sides = [_contact_side(comp, *q.shape) for comp in comps]
    assert len(set(sides)) == len(sides)
    assert any(side in {"right", "bottom"} for side in sides), "need at least one seam-touching object"

    actual = main(grid)
    assert np.array_equal(actual, expected), "solver output must match constructed target"
    assert np.array_equal(expected, _recolor_quadrant(q))
    assert not np.array_equal(q, expected)


def generate_input() -> np.ndarray:
    for _ in range(500):
        quadrant = _build_quadrant()
        if quadrant is None:
            continue
        expected = _recolor_quadrant(quadrant)
        panel = _mirror_panel(quadrant)
        ph, pw = panel.shape

        grid_h = random.randint(max(18, ph + 2), min(22, ph + 6))
        grid_w = random.randint(max(18, pw + 2), min(22, pw + 6))
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        r0 = random.randint(1, grid_h - ph - 1)
        c0 = random.randint(1, grid_w - pw - 1)
        grid[r0 : r0 + ph, c0 : c0 + pw] = panel

        try:
            _validate_generated(grid, expected)
        except AssertionError:
            continue
        return grid

    raise RuntimeError("failed to generate a valid puzzle input")
