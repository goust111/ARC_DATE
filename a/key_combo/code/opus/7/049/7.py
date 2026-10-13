import random
from collections import Counter, defaultdict

import numpy as np


BG = 0
FRAME = 5
FILL = 4
SOURCE_PALETTE = (1, 2, 3, 6, 7, 8, 9)
TARGET_PALETTE = (1, 2, 3, 6, 7, 8, 9)
CORNERS = ("tl", "tr", "br", "bl")


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(grid == FRAME)
    assert coords.size > 0, "missing gray frame"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame too small"
    assert np.all(grid[r0, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r1, c0 : c1 + 1] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c0] == FRAME)
    assert np.all(grid[r0 : r1 + 1, c1] == FRAME)
    return r0, c0, r1, c1


def _corner_anchor(bbox: tuple[int, int, int, int], kind: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if kind == "tl":
        return r0, c0
    if kind == "tr":
        return r0, c1
    if kind == "br":
        return r1, c1
    if kind == "bl":
        return r1, c0
    raise ValueError(kind)


def _to_canonical(
    rr: int, cc: int, bbox: tuple[int, int, int, int], kind: str
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if kind == "tl":
        return rr - r0, cc - c0
    if kind == "tr":
        return rr - r0, c1 - cc
    if kind == "br":
        return r1 - rr, c1 - cc
    if kind == "bl":
        return r1 - rr, cc - c0
    raise ValueError(kind)


def _from_canonical(
    bbox: tuple[int, int, int, int], kind: str, dr: int, dc: int
) -> tuple[int, int]:
    ar, ac = _corner_anchor(bbox, kind)
    if kind == "tl":
        return ar + dr, ac + dc
    if kind == "tr":
        return ar + dr, ac - dc
    if kind == "br":
        return ar - dr, ac - dc
    if kind == "bl":
        return ar - dr, ac + dc
    raise ValueError(kind)


def _assign_corner(
    rr: int, cc: int, bbox: tuple[int, int, int, int]
) -> tuple[str, int, int]:
    candidates = []
    for kind in CORNERS:
        dr, dc = _to_canonical(rr, cc, bbox, kind)
        if dr >= 1 and dc >= 1:
            candidates.append((max(dr, dc), dr + dc, kind, dr, dc))
    assert candidates, "colored cell is not inside the frame"
    _, _, kind, dr, dc = min(candidates)
    return kind, dr, dc


def _scanline_fill(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = bbox
    for r in range(r0 + 1, r1):
        cols = np.flatnonzero(out[r, c0 : c1 + 1] != BG) + c0
        for left, right in zip(cols, cols[1:]):
            if right - left <= 1:
                continue
            gap = out[r, left + 1 : right]
            gap[gap == BG] = FILL
    return out


def _extract_structure(
    grid: np.ndarray,
) -> tuple[tuple[int, int, int, int], set[tuple[int, int]], int, str, dict[str, int]]:
    bbox = _frame_bbox(grid)
    groups: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    r0, c0, r1, c1 = bbox
    for rr, cc in np.argwhere((grid != BG) & (grid != FRAME)):
        rr, cc = int(rr), int(cc)
        assert r0 < rr < r1 and c0 < cc < c1, "colored pixels must stay inside frame"
        kind, dr, dc = _assign_corner(rr, cc, bbox)
        groups[kind].append((dr, dc, int(grid[rr, cc])))

    assert set(groups) == set(CORNERS), "expected all four corners to be decorated or marked"

    source_kinds = [kind for kind, cells in groups.items() if len(cells) > 1]
    assert len(source_kinds) == 1, "expected exactly one multi-pixel source corner"
    source_kind = source_kinds[0]
    source_cells = groups[source_kind]
    source_colors = {color for _, _, color in source_cells}
    assert len(source_colors) == 1, "source pattern must be monochrome"
    source_color = next(iter(source_colors))

    pattern = {(dr, dc) for dr, dc, color in source_cells if color == source_color}
    diag_depths = [dr for dr, dc in pattern if dr == dc]
    assert diag_depths, "source pattern must contain diagonal anchors"
    k = max(diag_depths)
    assert (k, k) in pattern
    assert all((dc, dr) in pattern for dr, dc in pattern), "pattern must be diagonal-symmetric"

    targets: dict[str, int] = {}
    for kind in CORNERS:
        if kind == source_kind:
            continue
        cells = groups[kind]
        assert len(cells) == 1, "target corners must contain one marker"
        dr, dc, color = cells[0]
        assert (dr, dc) == (k, k), "marker must sit at the deepest diagonal offset"
        targets[kind] = color

    return bbox, pattern, source_color, source_kind, targets


def _paint_pattern(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    kind: str,
    pattern: set[tuple[int, int]],
    color: int,
) -> None:
    for dr, dc in pattern:
        rr, cc = _from_canonical(bbox, kind, dr, dc)
        grid[rr, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replicate the one fully drawn corner motif to the other three corners, then scanline-fill the gaps inside the frame.

    Input structure:
    - The background is black, with one gray hollow rectangle acting as a frame.
    - Exactly one corner contains a multi-pixel interior motif, while the other three corners contain a single marker pixel.
    - All colored motif or marker pixels lie strictly inside the frame and are associated with one of its four corners.

    Rule:
    - Find the unique source corner with the multi-pixel motif and extract its offsets relative to that corner.
    - Rotate that same offset pattern into the other three corners, using each target corner's marker color and overwriting the marker.
    - Treat the frame border plus all four corner motifs as row-wise boundary pixels; on every interior frame row, fill each black gap between consecutive non-black pixels with yellow.

    Key insight:
    - The source motif is recognized structurally by being the only multi-pixel corner decoration; once copied to all corners, those motif pixels become the scanline endpoints that determine the final filled shape.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bbox, pattern, _source_color, _source_kind, targets = _extract_structure(g)
    out = g.copy()
    for kind, color in targets.items():
        _paint_pattern(out, bbox, kind, pattern, color)
    return _scanline_fill(out, bbox)


def _pattern_options(k: int) -> list[set[tuple[int, int]]]:
    if k == 2:
        return [{(1, 1), (1, 2), (2, 1), (2, 2)}]
    return [
        {(1, 1), (2, 2), (3, 3)},
        {(1, 1), (2, 2), (3, 3), (1, 2), (2, 1)},
        {(1, 1), (2, 2), (3, 3), (2, 3), (3, 2)},
    ]


def _build_input() -> tuple[np.ndarray, np.ndarray]:
    H = random.randint(13, 18)
    W = random.randint(13, 18)
    k = random.choice((2, 3))

    min_inner = 2 * k + 3
    inner_h = random.randint(min_inner, H - 3)
    inner_w = random.randint(min_inner, W - 3)
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    r0 = random.randint(0, H - outer_h)
    c0 = random.randint(0, W - outer_w)
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    bbox = (r0, c0, r1, c1)

    pattern = random.choice(_pattern_options(k))
    source_kind = random.choice(CORNERS)
    source_color = random.choice(SOURCE_PALETTE)
    target_pool = [color for color in TARGET_PALETTE if color not in {source_color, FILL, FRAME}]
    target_colors = random.sample(target_pool, 3)

    inp = np.zeros((H, W), dtype=int)
    inp[r0, c0 : c1 + 1] = FRAME
    inp[r1, c0 : c1 + 1] = FRAME
    inp[r0 : r1 + 1, c0] = FRAME
    inp[r0 : r1 + 1, c1] = FRAME
    _paint_pattern(inp, bbox, source_kind, pattern, source_color)

    expected = inp.copy()
    for kind, color in zip([kind for kind in CORNERS if kind != source_kind], target_colors):
        rr, cc = _from_canonical(bbox, kind, k, k)
        inp[rr, cc] = color
        _paint_pattern(expected, bbox, kind, pattern, color)

    expected = _scanline_fill(expected, bbox)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(300):
        inp, expected = _build_input()
        try:
            actual = main(inp)
        except Exception:
            continue
        if not np.array_equal(actual, expected):
            continue
        if not np.any(actual == FILL):
            continue
        return inp
    raise RuntimeError("failed to generate a valid input")
