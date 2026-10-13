from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


BG = 0
MAX_GENERATION_ATTEMPTS = 500
TEMPLATE_LIBRARY = (
    ((0, 0, -1), (-1, 0, 0), (-1, 1, 1), (0, 1, 0)),
    ((0, 0, -1), (-1, 0, 0), (-2, 0, 0), (-2, 1, 1)),
    ((0, 0, -1), (-1, -1, 0), (-1, 0, 1), (-1, 1, 0), (-2, 0, 1)),
    ((0, 0, -1), (0, 1, 0), (-1, 1, 1), (-2, 1, 0), (-2, 0, 1)),
    ((0, 0, -1), (-1, 0, 0), (0, 1, 1), (1, 1, 0)),
    ((0, 0, -1), (-1, -1, 0), (0, -1, 1), (0, 1, 0), (1, 0, 1)),
)


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ]


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components(mask: np.ndarray, connectivity: int = 8) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    if connectivity == 8:
        deltas = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
    else:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _find_marker_info(grid: np.ndarray) -> tuple[int, tuple[int, int], set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    candidates: list[tuple[int, tuple[int, int], set[tuple[int, int]]]] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) != bg):
        coords = [tuple(map(int, rc)) for rc in np.argwhere(g == color)]
        if len(coords) < 2:
            continue

        decorated = []
        isolated = []
        for r, c in coords:
            neighbors = [
                int(g[nr, nc])
                for nr, nc in _neighbors8(r, c)
                if _in_bounds(g.shape, nr, nc)
            ]
            if any(v not in (bg, color) for v in neighbors):
                decorated.append((r, c))
            elif all(v == bg for v in neighbors):
                isolated.append((r, c))

        if len(decorated) != 1 or not isolated:
            continue

        anchor = decorated[0]
        comp = None
        for cells in _components(g != bg, connectivity=8):
            cell_set = set(cells)
            if anchor in cell_set:
                comp = cell_set
                break
        if comp is None:
            continue
        if sum(1 for rc in comp if int(g[rc]) == color) != 1:
            continue
        candidates.append((color, anchor, comp))

    assert len(candidates) == 1, "expected one decorated marker color"
    return candidates[0]


def _piece_offsets(grid: np.ndarray, anchor: tuple[int, int], component: set[tuple[int, int]]) -> list[tuple[int, int, int]]:
    ar, ac = anchor
    offsets = []
    for r, c in sorted(component):
        offsets.append((r - ar, c - ac, int(grid[r, c])))
    return sorted(offsets)


def _drop_piece(
    shape: tuple[int, int],
    offsets: list[tuple[int, int, int]],
    anchor: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> tuple[list[tuple[int, int, int]], int]:
    ar, ac = anchor
    cells = [(ar + dr, ac + dc, color) for dr, dc, color in offsets]
    assert all(_in_bounds(shape, r, c) for r, c, _color in cells)
    assert all((r, c) not in occupied for r, c, _color in cells)

    drop = 0
    while True:
        shifted = [(r + drop + 1, c) for r, c, _color in cells]
        if any(not _in_bounds(shape, r, c) or (r, c) in occupied for r, c in shifted):
            break
        drop += 1
    return [(r + drop, c, color) for r, c, color in cells], drop


def _simulate_markers(
    shape: tuple[int, int],
    fixed_cells: dict[tuple[int, int], int],
    offsets: list[tuple[int, int, int]],
    markers: list[tuple[int, int]],
    order: list[tuple[int, int]] | None = None,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    out = np.full(shape, BG, dtype=int)
    for (r, c), color in fixed_cells.items():
        out[r, c] = int(color)

    occupied = set(fixed_cells)
    pieces: list[dict[str, object]] = []
    if order is None:
        order = sorted(markers, key=lambda rc: (-rc[0], rc[1]))

    for anchor in order:
        landed_cells, drop = _drop_piece(shape, offsets, anchor, occupied)
        support: set[tuple[int, int]] = set()
        for r, c, _color in landed_cells:
            nr = r + 1
            if nr >= shape[0]:
                support.add((shape[0], c))
            elif (nr, c) in occupied:
                support.add((nr, c))
        for r, c, color in landed_cells:
            out[r, c] = int(color)
            occupied.add((r, c))
        pieces.append(
            {
                "anchor": anchor,
                "cells": landed_cells,
                "drop": drop,
                "support": support,
            }
        )
    return out, pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop a rigid copy of the exemplar stamp from every bare marker, letting lower copies settle first and later ones stack on them.

    Input structure:
    - The grid has a background plus static obstacle bars that act as floor and platforms.
    - Exactly one marker pixel is decorated by a small multicolor exemplar component touching it.
    - Additional pixels of the same marker color appear as isolated bare markers with no surrounding template.

    Rule:
    - Treat the decorated marker as the origin and record every colored cell in its 8-connected component as a relative offset.
    - For each bare marker, instantiate that same rigid component and let it fall straight downward as one piece.
    - Process bare markers from lower row to higher row, breaking ties by the leftmost marker; a piece stops when any of its cells would move into the grid boundary, a static obstacle, the exemplar, or an already settled copy.
    - Remove the bare markers from their starting cells, keep the exemplar fixed, and paint each settled copy in its landing position.

    Key insight:
    - The visible exemplar is a stamp template, but the other markers do not stamp in place: they become identical rigid bodies whose landing order changes later supports and therefore changes the final stack.
    """
    g = np.asarray(grid, dtype=int)
    marker_color, exemplar_anchor, exemplar_component = _find_marker_info(g)
    offsets = _piece_offsets(g, exemplar_anchor, exemplar_component)

    fixed_cells: dict[tuple[int, int], int] = {}
    markers: list[tuple[int, int]] = []
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color == BG:
                continue
            if color == marker_color and (r, c) not in exemplar_component:
                markers.append((r, c))
            else:
                fixed_cells[(r, c)] = color

    out, _pieces = _simulate_markers(g.shape, fixed_cells, offsets, markers)
    return out


def _instantiate_template(template: tuple[tuple[int, int, int], ...], marker_color: int, fill_colors: list[int]) -> list[tuple[int, int, int]]:
    offsets = []
    for dr, dc, idx in template:
        color = marker_color if idx < 0 else fill_colors[idx]
        offsets.append((dr, dc, color))
    return sorted(offsets)


def _bbox_from_offsets(offsets: list[tuple[int, int, int]]) -> tuple[int, int, int, int]:
    rows = [dr for dr, _dc, _color in offsets]
    cols = [dc for _dr, dc, _color in offsets]
    return min(rows), min(cols), max(rows), max(cols)


def _place_platforms(rng: np.random.Generator, grid: np.ndarray, avoid: set[tuple[int, int]]) -> None:
    h, w = grid.shape
    obstacle = int(next(color for color in np.unique(grid) if color != BG))
    platform_count = int(rng.integers(0, 4))
    used_rows: set[int] = set()

    for _ in range(platform_count):
        for _retry in range(50):
            row = int(rng.integers(max(5, h // 3), h - 3))
            if row in used_rows:
                continue
            length = int(rng.integers(4, min(9, w)))
            c0 = int(rng.integers(0, w - length + 1))
            cells = {(row, c) for c in range(c0, c0 + length)}
            halo = {
                (r, c)
                for rr, cc in cells
                for r in range(rr - 1, rr + 2)
                for c in range(cc - 1, cc + 2)
                if _in_bounds(grid.shape, r, c)
            }
            if halo & avoid:
                continue
            if any(grid[row, c] != BG for c in range(c0, c0 + length)):
                continue
            for c in range(c0, c0 + length):
                grid[row, c] = obstacle
            used_rows.add(row)
            break


def _piece_cells(anchor: tuple[int, int], offsets: list[tuple[int, int, int]]) -> set[tuple[int, int]]:
    ar, ac = anchor
    return {(ar + dr, ac + dc) for dr, dc, _color in offsets}


def _visible_marker_isolation(grid: np.ndarray, anchor: tuple[int, int]) -> bool:
    r, c = anchor
    return all(
        not _in_bounds(grid.shape, nr, nc) or int(grid[nr, nc]) == BG
        for nr, nc in _neighbors8(r, c)
    )


def _candidate_anchors(grid: np.ndarray, offsets: list[tuple[int, int, int]]) -> list[dict[str, object]]:
    h, w = grid.shape
    candidates: list[dict[str, object]] = []
    row_min, col_min, row_max, col_max = _bbox_from_offsets(offsets)

    for r in range(-row_min, h - row_max):
        for c in range(-col_min, w - col_max):
            anchor = (r, c)
            cells = _piece_cells(anchor, offsets)
            if any(int(grid[rr, cc]) != BG for rr, cc in cells):
                continue
            if int(grid[r, c]) != BG:
                continue
            if not _visible_marker_isolation(grid, anchor):
                continue
            cols = {cc for _rr, cc in cells}
            candidates.append({"anchor": anchor, "cells": cells, "cols": cols})
    return candidates


def _difference_order_exists(
    shape: tuple[int, int],
    fixed_cells: dict[tuple[int, int], int],
    offsets: list[tuple[int, int, int]],
    markers: list[tuple[int, int]],
) -> bool:
    expected, _ = _simulate_markers(shape, fixed_cells, offsets, markers)
    reverse_order = sorted(markers, key=lambda rc: (rc[0], rc[1]))
    try:
        other, _ = _simulate_markers(shape, fixed_cells, offsets, markers, order=reverse_order)
    except AssertionError:
        return True
    return not np.array_equal(expected, other)


def _supports_platform(piece: dict[str, object], floor_row: int, fixed_cells: dict[tuple[int, int], int]) -> bool:
    for cell in piece["support"]:
        if cell[0] == floor_row + 1:
            continue
        if cell in fixed_cells and cell[0] < floor_row:
            return True
    return False


def _pick_core_pair(
    shape: tuple[int, int],
    fixed_cells: dict[tuple[int, int], int],
    offsets: list[tuple[int, int, int]],
    candidates: list[dict[str, object]],
) -> list[tuple[int, int]] | None:
    pairs = list(itertools.combinations(candidates, 2))
    np.random.shuffle(pairs)
    for a, b in pairs:
        ar, ac = a["anchor"]
        br, bc = b["anchor"]
        if ar == br:
            continue
        if not (a["cols"] & b["cols"]):
            continue
        if a["cells"] & b["cells"]:
            continue
        markers = [a["anchor"], b["anchor"]]
        if _difference_order_exists(shape, fixed_cells, offsets, markers):
            return markers
    return None


def _pick_platform_marker(
    shape: tuple[int, int],
    fixed_cells: dict[tuple[int, int], int],
    offsets: list[tuple[int, int, int]],
    candidates: list[dict[str, object]],
    used_cells: set[tuple[int, int]],
    floor_row: int,
) -> tuple[int, int] | None:
    items = candidates[:]
    np.random.shuffle(items)
    for item in items:
        if item["cells"] & used_cells:
            continue
        _out, pieces = _simulate_markers(shape, fixed_cells, offsets, [item["anchor"]])
        if _supports_platform(pieces[0], floor_row, fixed_cells):
            return item["anchor"]
    return None


def _validate_generated(
    inp: np.ndarray,
    expected: np.ndarray,
    marker_color: int,
    exemplar_anchor: tuple[int, int],
    offsets: list[tuple[int, int, int]],
    markers: list[tuple[int, int]],
    fixed_cells: dict[tuple[int, int], int],
    require_platform: bool,
) -> bool:
    try:
        actual = main(inp)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    detected_color, detected_anchor, component = _find_marker_info(inp)
    if detected_color != marker_color or detected_anchor != exemplar_anchor:
        return False
    if _piece_offsets(inp, detected_anchor, component) != offsets:
        return False
    if len(markers) < 2:
        return False
    if not _difference_order_exists(inp.shape, fixed_cells, offsets, markers):
        return False
    if np.array_equal(inp, expected):
        return False

    for r, c in markers:
        if int(inp[r, c]) != marker_color:
            return False
        if not _visible_marker_isolation(inp, (r, c)):
            return False

    oracle, pieces = _simulate_markers(inp.shape, fixed_cells, offsets, markers)
    if not np.array_equal(oracle, expected):
        return False
    if not any(int(piece["drop"]) > 0 for piece in pieces):
        return False
    if require_platform and not any(_supports_platform(piece, inp.shape[0] - 1, fixed_cells) for piece in pieces):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 25))
    w = int(rng.integers(14, 25))
    palette = rng.choice(np.arange(1, 10), size=5, replace=False).astype(int).tolist()
    obstacle_color = palette[0]
    marker_color = palette[1]
    fill_colors = palette[2:5]
    template = TEMPLATE_LIBRARY[int(rng.integers(0, len(TEMPLATE_LIBRARY)))]
    offsets = _instantiate_template(template, marker_color, fill_colors)
    row_min, col_min, row_max, col_max = _bbox_from_offsets(offsets)

    grid = np.full((h, w), BG, dtype=int)
    grid[h - 1, :] = obstacle_color

    exemplar_side = int(rng.integers(0, 2))
    exemplar_col_lo = 1 - col_min
    exemplar_col_hi = w - col_max - 1
    if exemplar_col_lo >= exemplar_col_hi:
        return None
    if exemplar_side == 0:
        exemplar_col = int(rng.integers(exemplar_col_lo, min(exemplar_col_hi, max(exemplar_col_lo + 1, w // 3))))
    else:
        exemplar_col = int(rng.integers(max(exemplar_col_lo, (2 * w) // 3), exemplar_col_hi))
    exemplar_row = int(rng.integers(max(2, -row_min), min(7, h - row_max - 3)))
    exemplar_anchor = (exemplar_row, exemplar_col)
    exemplar_cells = _piece_cells(exemplar_anchor, offsets)
    for r, c, color in [(exemplar_row + dr, exemplar_col + dc, color) for dr, dc, color in offsets]:
        grid[r, c] = int(color)

    moat = {
        (r, c)
        for rr, cc in exemplar_cells
        for r in range(rr - 1, rr + 2)
        for c in range(cc - 1, cc + 2)
        if _in_bounds(grid.shape, r, c)
    }
    _place_platforms(rng, grid, moat)

    base_grid = grid.copy()
    candidates = _candidate_anchors(base_grid, offsets)
    if len(candidates) < 2:
        return None

    fixed_cells = {(r, c): int(base_grid[r, c]) for r, c in np.argwhere(base_grid != BG)}
    markers = _pick_core_pair(base_grid.shape, fixed_cells, offsets, candidates)
    if markers is None:
        return None

    used_cells = set()
    for item in candidates:
        if item["anchor"] in markers:
            used_cells |= set(item["cells"])
    require_platform = bool(rng.random() < 0.55 and np.any(base_grid[:-1] == obstacle_color))
    if require_platform:
        extra = _pick_platform_marker(base_grid.shape, fixed_cells, offsets, candidates, used_cells, h - 1)
        if extra is None or extra in markers:
            return None
        markers.append(extra)
        for item in candidates:
            if item["anchor"] == extra:
                used_cells |= set(item["cells"])
                break

    rng.shuffle(candidates)
    target_count = int(rng.integers(len(markers), min(5, len(candidates) + 1)))
    for item in candidates:
        if len(markers) >= target_count:
            break
        anchor = item["anchor"]
        if anchor in markers or item["cells"] & used_cells:
            continue
        trial = markers + [anchor]
        if not _difference_order_exists(base_grid.shape, fixed_cells, offsets, trial):
            continue
        markers.append(anchor)
        used_cells |= set(item["cells"])

    if not (2 <= len(markers) <= 4):
        return None

    inp = base_grid.copy()
    for r, c in markers:
        inp[r, c] = marker_color

    expected, _pieces = _simulate_markers(inp.shape, fixed_cells, offsets, markers)
    if not _validate_generated(
        inp,
        expected,
        marker_color,
        exemplar_anchor,
        offsets,
        markers,
        fixed_cells,
        require_platform,
    ):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt(rng)
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid stamp-drop stacking input")
