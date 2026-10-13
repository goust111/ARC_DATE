from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
STAMP_COLOR = 4
CODE_COLORS = (1, 2, 3, 6, 7, 8, 9)
GRID_SIZES = tuple(range(15, 30, 2))
RECT_SIZES = ((2, 2), (2, 3), (3, 2))
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    mask = grid == int(color)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            cells = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _grid_coord(idx: int) -> int:
    return 1 + 2 * int(idx)


def _local_to_grid(i: int, j: int, corner: int, lattice_n: int) -> tuple[int, int]:
    ii = int(i)
    jj = int(j)
    if corner in (1, 3):
        jj = lattice_n - 1 - jj
    if corner in (2, 3):
        ii = lattice_n - 1 - ii
    return _grid_coord(ii), _grid_coord(jj)


def _choose_corner(code_pixels: list[tuple[int, int, int]], lattice_n: int) -> list[tuple[int, int, int]]:
    orientations = []
    for flip_r in (False, True):
        for flip_c in (False, True):
            local = []
            for lr, lc, color in code_pixels:
                i = lattice_n - 1 - lr if flip_r else lr
                j = lattice_n - 1 - lc if flip_c else lc
                local.append((int(i), int(j), int(color)))
            score = (
                max(max(i, j) for i, j, _ in local),
                sum(i + j for i, j, _ in local),
            )
            orientations.append((score, local))
    orientations.sort(key=lambda item: item[0])
    assert len(orientations) >= 2 and orientations[0][0] < orientations[1][0], "corner code must be unambiguous"
    return orientations[0][1]


def _decode_code(local_pixels: list[tuple[int, int, int]], lattice_n: int):
    corner_colors: dict[int, int] = {}
    horizontal_colors: dict[int, int] = {}
    vertical_colors: dict[int, int] = {}

    for i, j, color in local_pixels:
        assert 0 <= i < lattice_n and 0 <= j < lattice_n
        if i == j:
            assert i not in corner_colors or corner_colors[i] == color
            corner_colors[i] = int(color)
        elif i < j:
            depth = int(i)
            assert depth < j < lattice_n - depth
            assert depth not in horizontal_colors or horizontal_colors[depth] == color
            horizontal_colors[depth] = int(color)
        else:
            depth = int(j)
            assert depth < i < lattice_n - depth
            assert depth not in vertical_colors or vertical_colors[depth] == color
            vertical_colors[depth] = int(color)

    depths = sorted(corner_colors)
    assert depths == list(range(len(depths))), "rings must start at the outermost layer and be consecutive"
    assert len(depths) in (2, 3), "expected 2-3 coded rings"
    for depth in set(horizontal_colors) | set(vertical_colors):
        assert depth in corner_colors, "side colors must belong to an existing ring"
    return corner_colors, horizontal_colors, vertical_colors


def _render_frames(
    shape: tuple[int, int],
    corner_colors: dict[int, int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
    bg: int = BACKGROUND,
) -> np.ndarray:
    h, w = shape
    assert h == w and h % 2 == 1
    lattice_n = (h - 1) // 2
    out = np.full(shape, int(bg), dtype=int)

    for depth, color in corner_colors.items():
        lo = int(depth)
        hi = lattice_n - 1 - lo
        for ri in (lo, hi):
            for ci in (lo, hi):
                out[_grid_coord(ri), _grid_coord(ci)] = int(color)

    for depth, color in horizontal_colors.items():
        lo = int(depth)
        hi = lattice_n - 1 - lo
        for ci in range(lo + 1, hi):
            out[_grid_coord(lo), _grid_coord(ci)] = int(color)
            out[_grid_coord(hi), _grid_coord(ci)] = int(color)

    for depth, color in vertical_colors.items():
        lo = int(depth)
        hi = lattice_n - 1 - lo
        for ri in range(lo + 1, hi):
            out[_grid_coord(ri), _grid_coord(lo)] = int(color)
            out[_grid_coord(ri), _grid_coord(hi)] = int(color)

    return out


def _find_stamp_rect(grid: np.ndarray, bg: int):
    comps = _components_of_color(grid, STAMP_COLOR)
    rects = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 2 or w < 2:
            continue
        if int(coords.shape[0]) != h * w:
            continue
        if np.any(grid[r0 : r1 + 1, c0 : c1 + 1] != STAMP_COLOR):
            continue
        rects.append((r0, c0, h, w))
    assert len(rects) == 1, "expected exactly one solid yellow rectangle"
    r0, c0, h, w = rects[0]
    return r0, c0, h, w


def _stamp_positions(h: int, w: int, r0: int, c0: int, rect_h: int, rect_w: int):
    positions = [(int(r0), int(c0), "origin")]
    for direction, dc in (("left", -rect_w), ("right", rect_w)):
        rr = r0 + rect_h
        cc = c0 + dc
        while rr < h and cc < w and cc + rect_w > 0:
            positions.append((int(rr), int(cc), direction))
            rr += rect_h
            cc += dc
    return positions


def _apply_stamp_overlays(frame: np.ndarray, r0: int, c0: int, rect_h: int, rect_w: int) -> np.ndarray:
    h, w = frame.shape
    out = frame.copy()
    for top, left, _tag in _stamp_positions(h, w, r0, c0, rect_h, rect_w):
        rr0 = max(0, top)
        cc0 = max(0, left)
        rr1 = min(h, top + rect_h)
        cc1 = min(w, left + rect_w)
        if rr0 >= rr1 or cc0 >= cc1:
            continue
        stamp_view = frame[rr0:rr1, cc0:cc1]
        out[rr0:rr1, cc0:cc1] = np.where(stamp_view != BACKGROUND, stamp_view, STAMP_COLOR)
    return out


def _decode_instance(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h == w and h % 2 == 1
    bg = _get_bg(g)
    assert int(bg) == BACKGROUND

    rect_r0, rect_c0, rect_h, rect_w = _find_stamp_rect(g, bg)
    code_mask = g != bg
    code_mask[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = False
    code_coords = np.argwhere(code_mask)
    assert len(code_coords) >= 3, "expected a non-empty corner code"
    assert all(int(r) % 2 == 1 and int(c) % 2 == 1 for r, c in code_coords), "corner code must lie on odd lattice sites"
    assert np.count_nonzero(g == STAMP_COLOR) == rect_h * rect_w, "yellow should only appear in the source stamp"

    lattice_n = (h - 1) // 2
    code_pixels = [((int(r) - 1) // 2, (int(c) - 1) // 2, int(g[r, c])) for r, c in code_coords]
    local_pixels = _choose_corner(code_pixels, lattice_n)
    corner_colors, horizontal_colors, vertical_colors = _decode_code(local_pixels, lattice_n)
    frame = _render_frames(g.shape, corner_colors, horizontal_colors, vertical_colors, bg=bg)
    return frame, (rect_r0, rect_c0, rect_h, rect_w), (corner_colors, horizontal_colors, vertical_colors)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a corner-coded concentric frame set, then recolor diagonal stamp copies wherever they land on the frame.

    Input structure:
    - The grid is an odd-sized square on a black background.
    - One corner contains a sparse odd-lattice code for 2-3 concentric square rings.
    - Away from that code there is exactly one solid yellow rectangle that acts as the stamp source.

    Rule:
    - Decode the nearest-corner code: diagonal code pixels set each ring's four corner color, horizontal-offset pixels set the top/bottom side color of that ring, and vertical-offset pixels set the left/right side color.
    - Draw the completed concentric frame set on the same grid.
    - Keep the original yellow rectangle and copy it repeatedly at offsets (k*h, -k*w) and (k*h, +k*w), where h and w are the rectangle's height and width, clipping partial copies and stopping each trajectory once a copy is fully outside the grid.
    - For every stamped pixel, if the destination cell is also a frame pixel, replace the yellow with that frame color; otherwise keep it yellow.

    Key insight:
    - The answer is the spatial coupling of two independent constructions: a symmetric frame decoded from one corner, and a diagonal stamp whose pixels are recolored only where the two structures intersect.
    """
    frame, (rect_r0, rect_c0, rect_h, rect_w), _decoded = _decode_instance(grid)
    return _apply_stamp_overlays(frame, rect_r0, rect_c0, rect_h, rect_w)


def _build_code_spec(num_rings: int):
    slots = [("h", depth) for depth in range(num_rings)] + [("v", depth) for depth in range(num_rings)]
    min_extra = 2 if num_rings == 2 else 1
    max_extra = min(3, len(slots))
    extra_count = random.randint(min_extra, max_extra)

    if extra_count >= 2 and random.random() < 0.7:
        h_depth = random.randrange(num_rings)
        v_depth = random.randrange(num_rings)
        chosen = {("h", h_depth), ("v", v_depth)}
        remaining = [slot for slot in slots if slot not in chosen]
        need = extra_count - len(chosen)
        if need > 0:
            chosen.update(random.sample(remaining, need))
    else:
        chosen = set(random.sample(slots, extra_count))

    used_colors = random.sample(CODE_COLORS, num_rings + len(chosen))
    corner_colors = {depth: used_colors[depth] for depth in range(num_rings)}
    horizontal_colors = {}
    vertical_colors = {}
    cursor = num_rings
    for axis, depth in sorted(chosen, key=lambda item: (item[1], item[0])):
        if axis == "h":
            horizontal_colors[depth] = used_colors[cursor]
        else:
            vertical_colors[depth] = used_colors[cursor]
        cursor += 1
    return corner_colors, horizontal_colors, vertical_colors


def _make_code_input(
    n: int,
    corner: int,
    corner_colors: dict[int, int],
    horizontal_colors: dict[int, int],
    vertical_colors: dict[int, int],
) -> tuple[np.ndarray, set[tuple[int, int]]]:
    grid = np.full((n, n), BACKGROUND, dtype=int)
    lattice_n = (n - 1) // 2
    code_cells: set[tuple[int, int]] = set()
    for depth, color in corner_colors.items():
        r, c = _local_to_grid(depth, depth, corner, lattice_n)
        grid[r, c] = int(color)
        code_cells.add((r, c))
    for depth, color in horizontal_colors.items():
        r, c = _local_to_grid(depth, depth + 1, corner, lattice_n)
        grid[r, c] = int(color)
        code_cells.add((r, c))
    for depth, color in vertical_colors.items():
        r, c = _local_to_grid(depth + 1, depth, corner, lattice_n)
        grid[r, c] = int(color)
        code_cells.add((r, c))
    return grid, code_cells


def _frame_corner_mask(
    n: int,
    corner_colors: dict[int, int],
) -> np.ndarray:
    lattice_n = (n - 1) // 2
    mask = np.zeros((n, n), dtype=bool)
    for depth in corner_colors:
        lo = depth
        hi = lattice_n - 1 - depth
        for ri in (lo, hi):
            for ci in (lo, hi):
                mask[_grid_coord(ri), _grid_coord(ci)] = True
    return mask


def _clipped_cells(n: int, top: int, left: int, rect_h: int, rect_w: int) -> set[tuple[int, int]]:
    rr0 = max(0, top)
    cc0 = max(0, left)
    rr1 = min(n, top + rect_h)
    cc1 = min(n, left + rect_w)
    if rr0 >= rr1 or cc0 >= cc1:
        return set()
    return {(r, c) for r in range(rr0, rr1) for c in range(cc0, cc1)}


def _trajectory_stats(
    frame: np.ndarray,
    code_cells: set[tuple[int, int]],
    rect_r0: int,
    rect_c0: int,
    rect_h: int,
    rect_w: int,
    corner_mask: np.ndarray,
):
    n = frame.shape[0]
    positions = _stamp_positions(n, n, rect_r0, rect_c0, rect_h, rect_w)
    direction_counts = Counter(tag for _r, _c, tag in positions)
    all_cells: set[tuple[int, int]] = set()
    overlap_cells: set[tuple[int, int]] = set()
    copy_overlap = {"left": 0, "right": 0}
    partial_count = 0
    for top, left, tag in positions:
        if top < 0 or left < 0 or top + rect_h > n or left + rect_w > n:
            partial_count += 1
        cells = _clipped_cells(n, top, left, rect_h, rect_w)
        if tag != "origin":
            copy_overlap[tag] += sum(frame[r, c] != BACKGROUND for r, c in cells)
        all_cells |= cells
        overlap_cells |= {(r, c) for r, c in cells if frame[r, c] != BACKGROUND}

    side_overlap = {(r, c) for r, c in overlap_cells if not corner_mask[r, c]}
    frame_cells = {(int(r), int(c)) for r, c in np.argwhere(frame != BACKGROUND)}
    absorbed_colors = {int(frame[r, c]) for r, c in overlap_cells}
    return {
        "positions": positions,
        "direction_counts": direction_counts,
        "partial_count": partial_count,
        "all_cells": all_cells,
        "overlap_cells": overlap_cells,
        "side_overlap": side_overlap,
        "frame_only": frame_cells - all_cells,
        "stamp_only": all_cells - frame_cells,
        "absorbed_colors": absorbed_colors,
        "copy_overlap": copy_overlap,
        "hits_code": bool(all_cells & code_cells),
    }


def _pick_stamp_location(
    frame: np.ndarray,
    code_cells: set[tuple[int, int]],
    rect_h: int,
    rect_w: int,
    corner_colors: dict[int, int],
) -> tuple[int, int] | None:
    n = frame.shape[0]
    corner_mask = _frame_corner_mask(n, corner_colors)
    want_partial = random.random() < 0.45
    want_multi_color = random.random() < 0.5

    code_array = np.array(sorted(code_cells), dtype=int)
    code_box = _bbox(code_array)
    candidates = []
    for r0 in range(0, n - rect_h + 1):
        if r0 + 2 * rect_h > n:
            continue
        for c0 in range(0, n - rect_w + 1):
            if c0 < rect_w or c0 + 2 * rect_w > n:
                continue

            rect_cells = {(r, c) for r in range(r0, r0 + rect_h) for c in range(c0, c0 + rect_w)}
            if rect_cells & code_cells:
                continue

            rr0, cc0, rr1, cc1 = code_box
            row_gap = max(0, max(rr0 - (r0 + rect_h), r0 - (rr1 + 1)))
            col_gap = max(0, max(cc0 - (c0 + rect_w), c0 - (cc1 + 1)))
            if row_gap + col_gap < 3:
                continue

            stats = _trajectory_stats(frame, code_cells, r0, c0, rect_h, rect_w, corner_mask)
            if stats["direction_counts"]["left"] == 0 or stats["direction_counts"]["right"] == 0:
                continue
            if stats["hits_code"]:
                continue
            if not stats["overlap_cells"] or not stats["stamp_only"] or not stats["frame_only"]:
                continue
            if not stats["side_overlap"]:
                continue
            if stats["copy_overlap"]["left"] == 0 or stats["copy_overlap"]["right"] == 0:
                continue
            if len(stats["overlap_cells"]) > max(10, len(stats["all_cells"]) // 2):
                continue
            if want_partial and stats["partial_count"] == 0:
                continue
            if not want_partial and stats["partial_count"] > 1:
                continue
            if want_multi_color and len(stats["absorbed_colors"]) < 2:
                continue

            score = (
                abs(len(stats["overlap_cells"]) - 4),
                abs(len(stats["absorbed_colors"]) - 2),
                stats["partial_count"],
                random.random(),
            )
            candidates.append((score, r0, c0))

    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    top_band = candidates[: min(10, len(candidates))]
    _score, r0, c0 = random.choice(top_band)
    return int(r0), int(c0)


def _semantic_checks(
    inp: np.ndarray,
    oracle_out: np.ndarray,
    frame: np.ndarray,
    rect_r0: int,
    rect_c0: int,
    rect_h: int,
    rect_w: int,
) -> bool:
    try:
        frame2, (rr, cc, hh, ww), decoded = _decode_instance(inp)
    except AssertionError:
        return False
    if not np.array_equal(frame, frame2):
        return False
    if (rr, cc, hh, ww) != (rect_r0, rect_c0, rect_h, rect_w):
        return False
    if not np.array_equal(main(inp), oracle_out):
        return False

    frame_only = frame.copy()
    naive_stamp = np.full_like(frame, BACKGROUND)
    for top, left, _tag in _stamp_positions(inp.shape[0], inp.shape[1], rect_r0, rect_c0, rect_h, rect_w):
        for r, c in _clipped_cells(inp.shape[0], top, left, rect_h, rect_w):
            naive_stamp[r, c] = STAMP_COLOR
    naive_overlay = frame.copy()
    naive_overlay[naive_stamp == STAMP_COLOR] = STAMP_COLOR

    if np.array_equal(oracle_out, frame_only):
        return False
    if np.array_equal(oracle_out, naive_stamp):
        return False
    if np.array_equal(oracle_out, naive_overlay):
        return False

    left_only = frame.copy()
    for top, left, tag in _stamp_positions(inp.shape[0], inp.shape[1], rect_r0, rect_c0, rect_h, rect_w):
        if tag not in ("origin", "left"):
            continue
        for r, c in _clipped_cells(inp.shape[0], top, left, rect_h, rect_w):
            left_only[r, c] = frame[r, c] if frame[r, c] != BACKGROUND else STAMP_COLOR
    right_only = frame.copy()
    for top, left, tag in _stamp_positions(inp.shape[0], inp.shape[1], rect_r0, rect_c0, rect_h, rect_w):
        if tag not in ("origin", "right"):
            continue
        for r, c in _clipped_cells(inp.shape[0], top, left, rect_h, rect_w):
            right_only[r, c] = frame[r, c] if frame[r, c] != BACKGROUND else STAMP_COLOR
    if np.array_equal(oracle_out, left_only) or np.array_equal(oracle_out, right_only):
        return False

    corner_colors, horizontal_colors, vertical_colors = decoded
    if not horizontal_colors and not vertical_colors:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    n = random.choice(GRID_SIZES)
    num_rings = random.choice((2, 3))
    corner = random.randrange(4)
    corner_colors, horizontal_colors, vertical_colors = _build_code_spec(num_rings)
    inp, code_cells = _make_code_input(n, corner, corner_colors, horizontal_colors, vertical_colors)
    frame = _render_frames(inp.shape, corner_colors, horizontal_colors, vertical_colors)
    rect_h, rect_w = random.choice(RECT_SIZES)
    location = _pick_stamp_location(frame, code_cells, rect_h, rect_w, corner_colors)
    if location is None:
        return None
    rect_r0, rect_c0 = location
    inp[rect_r0 : rect_r0 + rect_h, rect_c0 : rect_c0 + rect_w] = STAMP_COLOR
    oracle_out = _apply_stamp_overlays(frame, rect_r0, rect_c0, rect_h, rect_w)
    if not _semantic_checks(inp, oracle_out, frame, rect_r0, rect_c0, rect_h, rect_w):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
