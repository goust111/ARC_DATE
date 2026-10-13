import random
from collections import deque

import numpy as np


BLACK = 0
ORANGE = 7
PATTERN_COLOR_POOL = (1, 2, 3, 5, 6, 8)
MARKER_COLORS = (4, 9)
GRID_SIZES = (19, 21, 23)
HOLE_SIZES = (6, 7, 8)
MAX_GEN_ATTEMPTS = 400

LOCAL_TO_CORNER = {
    frozenset({(0, 0), (0, 1), (1, 0)}): "top_left",
    frozenset({(0, 0), (0, 1), (1, 1)}): "top_right",
    frozenset({(0, 0), (1, 0), (1, 1)}): "bottom_left",
    frozenset({(0, 1), (1, 0), (1, 1)}): "bottom_right",
}
CORNER_TO_LOCAL = {corner: cells for cells, corner in LOCAL_TO_CORNER.items()}


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _inside_bbox(r: int, c: int, box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return r0 <= r <= r1 and c0 <= c <= c1


def _components_by_color(
    grid: np.ndarray,
    allowed: np.ndarray,
) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros_like(allowed, dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if not allowed[r, c] or seen[r, c]:
                continue
            color = int(grid[r, c])
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and allowed[nr, nc] and not seen[nr, nc]:
                        if int(grid[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            components.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return components


def _marker_corner(component: dict) -> str:
    r0, c0, _r1, _c1 = component["bbox"]
    local = frozenset((int(r) - r0, int(c) - c0) for r, c in component["coords"])
    assert local in LOCAL_TO_CORNER
    return LOCAL_TO_CORNER[local]


def _expansion_cells(
    hole_box: tuple[int, int, int, int],
    marker_box: tuple[int, int, int, int],
    corner: str,
) -> set[tuple[int, int]]:
    hr0, hc0, hr1, hc1 = hole_box
    mr0, mc0, mr1, mc1 = marker_box
    cells: set[tuple[int, int]] = set()

    if corner == "top_left":
        for c in range(hc0, mc1 + 1):
            cells.add((mr0, c))
        for r in range(hr0, mr1 + 1):
            cells.add((r, mc0))
    elif corner == "top_right":
        for c in range(mc0, hc1 + 1):
            cells.add((mr0, c))
        for r in range(hr0, mr1 + 1):
            cells.add((r, mc1))
    elif corner == "bottom_left":
        for c in range(hc0, mc1 + 1):
            cells.add((mr1, c))
        for r in range(mr0, hr1 + 1):
            cells.add((r, mc0))
    else:
        assert corner == "bottom_right"
        for c in range(mc0, hc1 + 1):
            cells.add((mr1, c))
        for r in range(mr0, hr1 + 1):
            cells.add((r, mc1))
    return cells


def _infer_axis2(grid: np.ndarray, hole_box: tuple[int, int, int, int], *, rows: bool) -> int:
    h, w = grid.shape
    limit = h if rows else w
    best_axis2 = None
    best_score = -1

    for axis2 in range(2 * limit - 1):
        score = 0
        mismatch = False
        for r in range(h):
            for c in range(w):
                if _inside_bbox(r, c, hole_box):
                    continue
                rr, cc = (axis2 - r, c) if rows else (r, axis2 - c)
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if _inside_bbox(rr, cc, hole_box):
                    continue
                if int(grid[r, c]) != int(grid[rr, cc]):
                    mismatch = True
                    break
                score += 1
            if mismatch:
                break
        if not mismatch and score > best_score:
            best_axis2 = axis2
            best_score = score

    assert best_axis2 is not None and best_score > 0
    return int(best_axis2)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each L marker toward its hole corner, then restore the rest of the orange rectangle by bilateral reflection.

    Input structure:
    - A square grid has a black outer background and a visible colored pattern with horizontal and vertical reflection symmetry.
    - Near the center, one orange rectangular hole hides part of the pattern.
    - Inside that hole, there are one or two 3-pixel monochrome L markers, each fully inside the hole and not touching the hole border.

    Rule:
    - Find the orange hole rectangle from the bounding box of the orange cells.
    - For each 3-pixel L marker, read its corner orientation from its 2x2 shape and extend its two arms outward until they reach the corresponding sides of the hole.
    - Fill those expanded corner-L pixels with the marker color.
    - For every remaining hole cell, copy the color from the same-row horizontal reflection position; if that reflected position is still inside the hole, use the same-column vertical reflection position instead.

    Key insight:
    - The hole is solved by two cooperating mechanisms: L markers claim specific corner strips, and the untouched remainder is uniquely determined by the visible bilateral symmetry outside the hole.
    """
    g = np.asarray(grid, dtype=int)
    hole_box = _bbox(g == ORANGE)
    assert hole_box is not None
    hr0, hc0, hr1, hc1 = hole_box

    row_axis2 = _infer_axis2(g, hole_box, rows=True)
    col_axis2 = _infer_axis2(g, hole_box, rows=False)

    hole_mask = np.zeros_like(g, dtype=bool)
    hole_mask[hr0 : hr1 + 1, hc0 : hc1 + 1] = True
    marker_mask = hole_mask & (g != ORANGE) & (g != BLACK)
    markers = _components_by_color(g, marker_mask)
    assert 1 <= len(markers) <= 2

    out = g.copy()
    covered = np.zeros_like(g, dtype=bool)
    for marker in markers:
        assert marker["area"] == 3
        r0, c0, r1, c1 = marker["bbox"]
        assert (r1 - r0, c1 - c0) == (1, 1)
        corner = _marker_corner(marker)
        cells = _expansion_cells(hole_box, marker["bbox"], corner)
        color = int(marker["color"])
        for r, c in cells:
            out[r, c] = color
            covered[r, c] = True

    h, w = g.shape
    for r in range(hr0, hr1 + 1):
        for c in range(hc0, hc1 + 1):
            if covered[r, c]:
                continue
            mirror_c = col_axis2 - c
            if 0 <= mirror_c < w and not _inside_bbox(r, mirror_c, hole_box):
                out[r, c] = g[r, mirror_c]
                continue
            mirror_r = row_axis2 - r
            assert 0 <= mirror_r < h
            assert not _inside_bbox(mirror_r, c, hole_box)
            out[r, c] = g[mirror_r, c]

    return out


def _build_pattern_grid(size: int, colors: tuple[int, ...]) -> np.ndarray:
    grid = np.zeros((size, size), dtype=int)
    center = size // 2
    for r in range(1, center + 1):
        for c in range(1, center + 1):
            color = random.choice(colors)
            for rr in {r, size - 1 - r}:
                for cc in {c, size - 1 - c}:
                    grid[rr, cc] = color
    return grid


def _choose_hole_box(size: int, h: int, w: int) -> tuple[tuple[int, int, int, int], bool]:
    center = size // 2
    cross_vertical = random.random() < 0.5
    candidates: list[tuple[int, int, int, int]] = []
    if cross_vertical:
        c0 = center - w // 2
        c1 = c0 + w - 1
        r_top = center - 1
        top_box = (r_top - h + 1, c0, r_top, c1)
        bottom_box = (center + 1, c0, center + h, c1)
        for box in (top_box, bottom_box):
            r0, cc0, r1, cc1 = box
            if 1 <= r0 <= r1 <= size - 2 and 1 <= cc0 <= cc1 <= size - 2:
                candidates.append(box)
    else:
        for gap in (1, 2):
            left_c1 = center - gap
            left_c0 = left_c1 - w + 1
            right_c0 = center + gap
            right_c1 = right_c0 + w - 1
            for c0, c1 in ((left_c0, left_c1), (right_c0, right_c1)):
                for shift in (-1, 0, 1):
                    r0 = center - h // 2 + shift
                    r1 = r0 + h - 1
                    if 1 <= r0 <= r1 <= size - 2 and 1 <= c0 <= c1 <= size - 2:
                        candidates.append((r0, c0, r1, c1))
    assert candidates
    box = random.choice(candidates)
    return tuple(map(int, box)), cross_vertical


def _marker_top_left_candidates(
    hole_box: tuple[int, int, int, int],
    corner: str,
) -> list[tuple[int, int]]:
    hr0, hc0, hr1, hc1 = hole_box
    hh = hr1 - hr0 + 1
    ww = hc1 - hc0 + 1
    max_rel_r = hh - 3
    max_rel_c = ww - 3
    top_rel_max = max(1, (hh - 3) // 2)
    left_rel_max = max(1, (ww - 3) // 2)
    bottom_rel_min = min(max_rel_r, (hh - 2) // 2)
    right_rel_min = min(max_rel_c, (ww - 2) // 2)

    rel_rows = range(1, max_rel_r + 1)
    rel_cols = range(1, max_rel_c + 1)
    out = []
    for rr in rel_rows:
        for cc in rel_cols:
            if "top" in corner and rr > top_rel_max:
                continue
            if "bottom" in corner and rr < bottom_rel_min:
                continue
            if "left" in corner and cc > left_rel_max:
                continue
            if "right" in corner and cc < right_rel_min:
                continue
            out.append((hr0 + rr, hc0 + cc))
    random.shuffle(out)
    return out


def _place_markers(
    hole_box: tuple[int, int, int, int],
    marker_count: int,
) -> list[dict] | None:
    if marker_count == 1:
        corners = [random.choice(tuple(CORNER_TO_LOCAL))]
    else:
        corners = random.choice(
            [
                ["top_left", "bottom_right"],
                ["top_right", "bottom_left"],
            ]
        )

    chosen = []
    occupied: set[tuple[int, int]] = set()
    used_colors = list(MARKER_COLORS)
    random.shuffle(used_colors)
    for idx, corner in enumerate(corners):
        placed = False
        for mr0, mc0 in _marker_top_left_candidates(hole_box, corner):
            marker_box = (mr0, mc0, mr0 + 1, mc0 + 1)
            expansion = _expansion_cells(hole_box, marker_box, corner)
            if expansion & occupied:
                continue
            occupied |= expansion
            chosen.append(
                {
                    "corner": corner,
                    "color": used_colors[idx],
                    "box": marker_box,
                    "expansion": expansion,
                }
            )
            placed = True
            break
        if not placed:
            return None
    return chosen


def _paint_marker(grid: np.ndarray, marker: dict) -> None:
    r0, c0, _r1, _c1 = marker["box"]
    color = int(marker["color"])
    for dr, dc in CORNER_TO_LOCAL[marker["corner"]]:
        grid[r0 + dr, c0 + dc] = color


def _horizontal_hidden_count(
    hole_box: tuple[int, int, int, int],
    covered: np.ndarray,
    col_axis2: int,
) -> int:
    hr0, hc0, hr1, hc1 = hole_box
    total = 0
    for r in range(hr0, hr1 + 1):
        for c in range(hc0, hc1 + 1):
            if covered[r, c]:
                continue
            if _inside_bbox(r, col_axis2 - c, hole_box):
                total += 1
    return total


def _gen_attempt() -> np.ndarray | None:
    size = random.choice(GRID_SIZES)
    pattern_colors = tuple(random.sample(PATTERN_COLOR_POOL, k=random.choice((3, 4))))
    clean = _build_pattern_grid(size, pattern_colors)
    if len(set(map(int, clean[1:-1, 1:-1].flatten()))) < 3:
        return None

    hole_h = random.choice(HOLE_SIZES)
    hole_w = random.choice(HOLE_SIZES)
    hole_box, cross_vertical = _choose_hole_box(size, hole_h, hole_w)

    marker_count = random.choice((1, 2))
    markers = _place_markers(hole_box, marker_count)
    if markers is None:
        return None

    expected = clean.copy()
    covered = np.zeros_like(clean, dtype=bool)
    for marker in markers:
        for r, c in marker["expansion"]:
            expected[r, c] = int(marker["color"])
            covered[r, c] = True

    hr0, hc0, hr1, hc1 = hole_box
    inp = clean.copy()
    inp[hr0 : hr1 + 1, hc0 : hc1 + 1] = ORANGE
    for marker in markers:
        _paint_marker(inp, marker)

    center = size // 2
    remaining = (hr1 - hr0 + 1) * (hc1 - hc0 + 1) - int(covered[hr0 : hr1 + 1, hc0 : hc1 + 1].sum())
    if remaining <= 0:
        return None
    if cross_vertical:
        hidden_horiz = _horizontal_hidden_count(hole_box, covered, center * 2)
        if hidden_horiz == 0:
            return None
    else:
        hidden_horiz = _horizontal_hidden_count(hole_box, covered, center * 2)
        if hidden_horiz >= remaining:
            return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
