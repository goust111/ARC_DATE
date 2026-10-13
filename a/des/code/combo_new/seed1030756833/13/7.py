from __future__ import annotations

from collections import defaultdict, deque
import random

import numpy as np


BG = 0
YELLOW = 4
PALETTE = tuple(c for c in range(10) if c not in (BG, YELLOW))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int):
    for dr, dc in DIR4:
        yield r + dr, c + dc


def _edge_cells(h: int, w: int):
    seen: set[tuple[int, int]] = set()
    for c in range(w):
        seen.add((0, c))
        seen.add((h - 1, c))
    for r in range(h):
        seen.add((r, 0))
        seen.add((r, w - 1))
    return sorted(seen)


def _infer_axes(g: np.ndarray) -> list[dict]:
    h, w = g.shape
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in _edge_cells(h, w):
        color = int(g[r, c])
        if color != BG:
            by_color[color].append((r, c))

    axes: list[dict] = []
    marker_cells: set[tuple[int, int]] = set()
    for color, pts in by_color.items():
        assert len(pts) == 2, "each edge marker color must appear exactly twice"
        (r0, c0), (r1, c1) = pts

        is_vertical = c0 == c1 and {r0, r1} == {0, h - 1}
        is_horizontal = r0 == r1 and {c0, c1} == {0, w - 1}
        assert is_vertical != is_horizontal, "marker pair must span opposite edges on one row or column"

        if is_vertical:
            axes.append({"orient": "V", "index": int(c0), "color": int(color), "markers": set(pts)})
        else:
            axes.append({"orient": "H", "index": int(r0), "color": int(color), "markers": set(pts)})
        marker_cells.update(pts)

    assert axes, "expected at least one marker-defined mirror axis"
    return axes


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if bool(seen[sr, sc]):
            continue
        seen[sr, sc] = True
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        pts: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            pts.append((r, c))
            for nr, nc in _neighbors4(r, c):
                if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                    seen[nr, nc] = True
                    q.append((nr, nc))
        comps.append(np.array(pts, dtype=int))
    return comps


def _fragment_components(g: np.ndarray, marker_colors: set[int]) -> list[dict]:
    fragments: list[dict] = []
    for color in sorted(int(c) for c in np.unique(g)):
        if color == BG or color in marker_colors:
            continue
        for pts in _components(g == color):
            assert pts.size > 0
            fragments.append({"color": int(color), "coords": pts})
    return fragments


def _axis_for_fragment(coords: np.ndarray, axes: list[dict]) -> dict:
    candidates: list[dict] = []
    for axis in axes:
        index = int(axis["index"])
        vals = coords[:, 1] if axis["orient"] == "V" else coords[:, 0]
        if bool(np.all(vals < index)) or bool(np.all(vals > index)):
            candidates.append(axis)
    assert len(candidates) == 1, "each fragment must belong to exactly one mirror axis"
    return candidates[0]


def _reflect_cell(axis: dict, r: int, c: int) -> tuple[int, int]:
    index = int(axis["index"])
    if axis["orient"] == "V":
        return r, 2 * index - c
    return 2 * index - r, c


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror interior colored fragments across marker-defined edge axes, coloring original-copy overlaps yellow.

    Input structure:
    - The background is Black.
    - Pairs of same-color marker pixels sit on opposite grid edges, either top-bottom
      in one column or left-right in one row; each pair defines a vertical or horizontal hidden mirror axis.
    - Away from the edge markers, the remaining foreground consists of 4-connected
      monochrome fragments whose colors differ from all marker colors.
    - Each fragment lies wholly on one side of exactly one hidden axis.

    Rule:
    - Keep the marker pixels and all original fragment pixels unless an overlap rule recolors them.
    - For every fragment, infer its unique axis and draw its mirror image across that axis in the fragment's color.
    - Mirrored pixels never overwrite edge markers.
    - Where any mirrored copy lands on an original fragment pixel, paint that cell Yellow.

    Key insight:
    - The mirror axes are not drawn directly; they must be inferred from the paired edge markers before assigning fragments to axes and resolving overlap cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    h, w = g.shape
    assert h <= 30 and w <= 30, "ARC grids must be at most 30 by 30"

    axes = _infer_axes(g)
    marker_colors = {int(axis["color"]) for axis in axes}
    marker_cells = set().union(*(axis["markers"] for axis in axes))
    fragments = _fragment_components(g, marker_colors)
    assert fragments, "expected at least one interior fragment"

    original_fragment = np.zeros(g.shape, dtype=bool)
    for frag in fragments:
        pts = frag["coords"]
        original_fragment[pts[:, 0], pts[:, 1]] = True

    proposals: dict[tuple[int, int], list[int]] = defaultdict(list)
    for frag in fragments:
        color = int(frag["color"])
        axis = _axis_for_fragment(frag["coords"], axes)
        for r, c in frag["coords"]:
            rr, cc = _reflect_cell(axis, int(r), int(c))
            assert 0 <= rr < h and 0 <= cc < w, "reflected fragment must stay in bounds"
            if (rr, cc) in marker_cells:
                continue
            proposals[(rr, cc)].append(color)

    out = g.copy()
    for (r, c), colors in proposals.items():
        if (r, c) in marker_cells:
            continue
        if bool(original_fragment[r, c]):
            out[r, c] = YELLOW
            continue

        first = int(colors[0])
        assert all(int(color) == first for color in colors), "copy-only overlaps must not conflict"
        out[r, c] = first

    return out


def _normalize_mask(mask: np.ndarray) -> np.ndarray:
    pts = np.argwhere(mask)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _make_shape(rng: random.Random, span: str | None) -> np.ndarray:
    if span == "vertical":
        h = rng.randint(3, 5)
        w = rng.randint(2, 4)
        mask = np.zeros((h, w), dtype=bool)
        c = rng.randrange(w)
        mask[:, c] = True
    elif span == "horizontal":
        h = rng.randint(2, 4)
        w = rng.randint(3, 5)
        mask = np.zeros((h, w), dtype=bool)
        r = rng.randrange(h)
        mask[r, :] = True
    else:
        h = rng.randint(2, 4)
        w = rng.randint(2, 4)
        mask = np.zeros((h, w), dtype=bool)
        mask[rng.randrange(h), rng.randrange(w)] = True

    frontier = {
        (nr, nc)
        for r, c in np.argwhere(mask)
        for nr, nc in _neighbors4(int(r), int(c))
        if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not bool(mask[nr, nc])
    }
    target = rng.randint(max(3, int(mask.sum())), min(mask.size, int(mask.sum()) + rng.randint(2, 5)))
    while int(mask.sum()) < target and frontier:
        r, c = rng.choice(tuple(frontier))
        frontier.remove((r, c))
        mask[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and not bool(mask[nr, nc]):
                frontier.add((nr, nc))

    return _normalize_mask(mask)


def _other_axis_index(axes: list[dict], orient: str) -> int | None:
    for axis in axes:
        if axis["orient"] == orient:
            return int(axis["index"])
    return None


def _rand_range(rng: random.Random, lo: int, hi: int) -> int | None:
    if lo > hi:
        return None
    return rng.randint(lo, hi)


def _origin_for_axis(
    rng: random.Random,
    axis: dict,
    axes: list[dict],
    mask: np.ndarray,
    side: int,
    h: int,
    w: int,
) -> tuple[int, int] | None:
    mh, mw = map(int, mask.shape)
    index = int(axis["index"])

    if axis["orient"] == "V":
        row_lo, row_hi = 1, h - mh - 1
        h_axis = _other_axis_index(axes, "H")
        if h_axis is not None:
            row_lo = max(row_lo, h_axis - mh + 2)
            row_hi = min(row_hi, h_axis - 1)

        if side < 0:
            col_lo = max(1, 2 * index - (w - 2))
            col_hi = index - mw
        else:
            col_lo = index + 1
            col_hi = min(w - mw - 1, 2 * index - mw)

        r0 = _rand_range(rng, row_lo, row_hi)
        c0 = _rand_range(rng, col_lo, col_hi)
    else:
        col_lo, col_hi = 1, w - mw - 1
        v_axis = _other_axis_index(axes, "V")
        if v_axis is not None:
            col_lo = max(col_lo, v_axis - mw + 2)
            col_hi = min(col_hi, v_axis - 1)

        if side < 0:
            row_lo = max(1, 2 * index - (h - 2))
            row_hi = index - mh
        else:
            row_lo = index + 1
            row_hi = min(h - mh - 1, 2 * index - mh)

        r0 = _rand_range(rng, row_lo, row_hi)
        c0 = _rand_range(rng, col_lo, col_hi)

    if r0 is None or c0 is None:
        return None
    return int(r0), int(c0)


def _shape_cells(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {(int(r0 + r), int(c0 + c)) for r, c in np.argwhere(mask)}


def _reflect_cells(axis: dict, cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {_reflect_cell(axis, r, c) for r, c in cells}


def _span_for_axis(axis: dict, axes: list[dict]) -> str | None:
    if axis["orient"] == "V" and _other_axis_index(axes, "H") is not None:
        return "vertical"
    if axis["orient"] == "H" and _other_axis_index(axes, "V") is not None:
        return "horizontal"
    return None


def _clear_cells(
    cells: set[tuple[int, int]],
    occupied: np.ndarray,
    reserved: np.ndarray,
    forbidden: np.ndarray,
) -> bool:
    h, w = occupied.shape
    for r, c in cells:
        if not (1 <= r < h - 1 and 1 <= c < w - 1):
            return False
        if bool(occupied[r, c]) or bool(reserved[r, c]) or bool(forbidden[r, c]):
            return False
    return True


def _mark_piece(
    grid: np.ndarray,
    cells: set[tuple[int, int]],
    color: int,
    occupied: np.ndarray,
    reserved: np.ndarray,
) -> None:
    h, w = grid.shape
    for r, c in cells:
        grid[r, c] = int(color)
        occupied[r, c] = True
        reserved[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                reserved[nr, nc] = True


def _mark_targets(mask: np.ndarray, cells: set[tuple[int, int]]) -> None:
    h, w = mask.shape
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            mask[r, c] = True


def _add_random_piece(
    rng: random.Random,
    grid: np.ndarray,
    axis: dict,
    axes: list[dict],
    fragment_colors: tuple[int, ...],
    occupied: np.ndarray,
    reserved: np.ndarray,
    copy_targets: np.ndarray,
    forbidden: np.ndarray,
    paired: bool,
) -> bool:
    h, w = grid.shape
    span = _span_for_axis(axis, axes)
    sides = [-1, 1]

    for _ in range(160):
        mask = _make_shape(rng, span)
        rng.shuffle(sides)
        for side in sides:
            origin = _origin_for_axis(rng, axis, axes, mask, side, h, w)
            if origin is None:
                continue
            cells = _shape_cells(mask, *origin)
            mirrored = _reflect_cells(axis, cells)
            if not _clear_cells(cells, occupied, reserved, forbidden):
                continue

            if paired:
                temp_occ = occupied.copy()
                temp_res = reserved.copy()
                _mark_piece(np.zeros_like(grid), cells, 1, temp_occ, temp_res)
                if not _clear_cells(mirrored, temp_occ, temp_res, forbidden):
                    continue
                if any(bool(copy_targets[r, c]) for r, c in mirrored):
                    continue

                color_a = rng.choice(fragment_colors)
                color_b = rng.choice(fragment_colors)
                _mark_piece(grid, cells, color_a, occupied, reserved)
                _mark_piece(grid, mirrored, color_b, occupied, reserved)
                _mark_targets(copy_targets, cells | mirrored)
                _mark_targets(forbidden, cells | mirrored)
                return True

            if any(bool(occupied[r, c]) or bool(copy_targets[r, c]) for r, c in mirrored):
                continue
            color = rng.choice(fragment_colors)
            _mark_piece(grid, cells, color, occupied, reserved)
            _mark_targets(copy_targets, mirrored)
            _mark_targets(forbidden, mirrored)
            return True

    return False


def _draw_markers(grid: np.ndarray, axes: list[dict]) -> None:
    h, w = grid.shape
    for axis in axes:
        color = int(axis["color"])
        index = int(axis["index"])
        if axis["orient"] == "V":
            grid[0, index] = color
            grid[h - 1, index] = color
        else:
            grid[index, 0] = color
            grid[index, w - 1] = color


def _choose_axes(rng: random.Random, h: int, w: int) -> list[dict]:
    kind = rng.choice(("V", "H", "VH"))
    colors = list(PALETTE)
    rng.shuffle(colors)
    axes: list[dict] = []
    if "V" in kind:
        axes.append({"orient": "V", "index": rng.randint(5, w - 6), "color": colors.pop(), "markers": set()})
    if "H" in kind:
        axes.append({"orient": "H", "index": rng.randint(5, h - 6), "color": colors.pop(), "markers": set()})
    return axes


def _expected_marker_sets(grid: np.ndarray, axes: list[dict]) -> None:
    h, w = grid.shape
    for axis in axes:
        index = int(axis["index"])
        if axis["orient"] == "V":
            axis["markers"] = {(0, index), (h - 1, index)}
        else:
            axis["markers"] = {(index, 0), (index, w - 1)}


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(15, 25)
    w = rng.randint(15, 25)
    grid = np.full((h, w), BG, dtype=int)
    axes = _choose_axes(rng, h, w)
    _expected_marker_sets(grid, axes)
    _draw_markers(grid, axes)

    marker_colors = {int(axis["color"]) for axis in axes}
    fragment_colors = tuple(c for c in PALETTE if c not in marker_colors)
    occupied = np.zeros((h, w), dtype=bool)
    reserved = np.zeros((h, w), dtype=bool)
    copy_targets = np.zeros((h, w), dtype=bool)
    forbidden = np.zeros((h, w), dtype=bool)

    want_overlap = rng.random() < 0.5
    if want_overlap:
        axis = rng.choice(axes)
        if not _add_random_piece(
            rng, grid, axis, axes, fragment_colors, occupied, reserved, copy_targets, forbidden, paired=True
        ):
            return None

    for axis in axes:
        for _ in range(rng.randint(1, 2)):
            if not _add_random_piece(
                rng, grid, axis, axes, fragment_colors, occupied, reserved, copy_targets, forbidden, paired=False
            ):
                return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if not bool((out != grid).any()):
        return None
    if not bool(((out != grid) & (out != YELLOW)).any()):
        return None
    has_yellow = bool(np.any(out == YELLOW))
    if want_overlap != has_yellow:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marker-axis reflection input")
