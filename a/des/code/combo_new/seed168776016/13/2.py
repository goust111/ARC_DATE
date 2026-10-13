from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE = (1, 2, 3, 4, 6, 7, 8)
DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in points]
    cols = [c for _r, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_separator(g: np.ndarray) -> int:
    h, w = g.shape
    candidates = []
    for c in range(1, w - 1):
        if int(g[0, c]) != GREY:
            continue
        if not np.all(g[:, c] == GREY):
            continue
        left = g[:, :c]
        right = g[:, c + 1 :]
        if left.size == 0 or right.size == 0:
            continue
        left_has_fragment = bool(np.isin(left, PALETTE).any())
        right_has_frame = bool((right == GREY).any())
        if left_has_fragment and right_has_frame:
            candidates.append(c)
    assert len(candidates) == 1, "expected a unique full-height grey separator"
    return int(candidates[0])


def _fragment_from_points(g: np.ndarray, points: list[tuple[int, int]]) -> dict[str, object]:
    pts = [(int(r), int(c)) for r, c in points]
    rows = sorted({r for r, _c in pts})
    cols = sorted({c for _r, c in pts})
    assert len(pts) in (2, 3, 4), "fragment length must be 2-4"
    assert len(rows) == 1 or len(cols) == 1, "fragment must be a straight line"
    if len(rows) == 1:
        r = rows[0]
        assert cols == list(range(cols[0], cols[-1] + 1)), "horizontal fragment must be contiguous"
        cells = [(r, c) for c in cols]
    else:
        c = cols[0]
        assert rows == list(range(rows[0], rows[-1] + 1)), "vertical fragment must be contiguous"
        cells = [(r, c) for r in rows]

    endpoints = [cells[0], cells[-1]]
    interface_candidates = []
    for endpoint in endpoints:
        if any(0 <= nr < g.shape[0] and 0 <= nc < g.shape[1] and int(g[nr, nc]) == MAROON for nr, nc in _neighbors4(*endpoint)):
            interface_candidates.append(endpoint)
    assert len(interface_candidates) == 1, "expected exactly one maroon-marked endpoint"

    interface = interface_candidates[0]
    if interface == cells[0]:
        cells_from_interface = cells
        cells_from_noninterface = list(reversed(cells))
    else:
        cells_from_interface = list(reversed(cells))
        cells_from_noninterface = cells

    return {
        "key": min(pts),
        "cells_from_interface": cells_from_interface,
        "cells_from_noninterface": cells_from_noninterface,
    }


def _fragments_by_color(g: np.ndarray, sep: int) -> dict[int, list[dict[str, object]]]:
    left = g[:, :sep]
    colors = sorted(int(v) for v in np.unique(left) if int(v) in PALETTE)
    out: dict[int, list[dict[str, object]]] = {}
    for color in colors:
        comps = _components(left == color)
        fragments = [_fragment_from_points(left, comp) for comp in comps]
        fragments.sort(key=lambda frag: frag["key"])
        out[color] = fragments
    return out


def _path_from_fragments(fragments: list[dict[str, object]]) -> list[tuple[int, int]]:
    assert fragments, "each color must have at least one fragment"
    path = [tuple(cell) for cell in fragments[0]["cells_from_noninterface"]]
    current = path[-1]

    for fragment in fragments[1:]:
        local = [tuple(cell) for cell in fragment["cells_from_interface"]]
        dr = int(current[0] - local[0][0])
        dc = int(current[1] - local[0][1])
        shifted = [(r + dr, c + dc) for r, c in local]
        path.extend(shifted[1:])
        current = shifted[-1]

    start = path[0]
    return [(int(r - start[0]), int(c - start[1])) for r, c in path]


def _find_boxes(g: np.ndarray, sep: int) -> dict[int, tuple[int, int, int, int]]:
    right = g[:, sep + 1 :]
    boxes: dict[int, tuple[int, int, int, int]] = {}
    for comp in _components(right == GREY):
        abs_pts = [(r, c + sep + 1) for r, c in comp]
        r0, c0, r1, c1 = _bbox(abs_pts)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        if height < 4 or width < 4:
            continue
        if len(abs_pts) != 2 * height + 2 * width - 4:
            continue
        border_ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                on_border = r in (r0, r1) or c in (c0, c1)
                if on_border and int(g[r, c]) != GREY:
                    border_ok = False
                if not on_border and int(g[r, c]) == GREY:
                    border_ok = False
        if not border_ok:
            continue

        marker = int(g[r0 + 1, c0 + 1])
        if marker not in PALETTE:
            continue
        interior = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        interior[0, 0] = BLACK
        if not np.all(interior == BLACK):
            continue
        assert marker not in boxes, "each output box color must be unique"
        boxes[marker] = (r0, c0, r1, c1)
    return boxes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each color's scattered path from left-side fragments and draw it into the matching right-side frame.

    Input structure:
    - The background is black, and a full-height grey column separates the grid into left and right regions.
    - The left region contains several colored straight fragments; each fragment is a 2-4 cell horizontal or vertical line.
    - Exactly one maroon cell touches one endpoint of each fragment, marking that endpoint as the fragment's interface.
    - The right region contains grey rectangular frames whose interior top-left cell is a color marker matching one left-side fragment color.

    Rule:
    - For each color, collect all left-side fragments of that color and sort them by appearance order from top to bottom, then left to right.
    - The first fragment is traversed from its non-interface endpoint to its interface endpoint; every later fragment is traversed from its interface endpoint to its non-interface endpoint, with consecutive fragments joined interface-to-interface.
    - This produces one orthogonal polyline per color. Align its start to the cell directly below the matching frame's color marker and draw the path in that frame.
    - Leave the original fragments, separator, frames, and marker pixels unchanged.

    Key insight:
    - The maroon neighbor identifies which endpoint of each segment is the connector, while spatial reading order decides how same-color fragments must be chained.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator(g)
    fragments = _fragments_by_color(g, sep)
    boxes = _find_boxes(g, sep)

    out = g.copy()
    for color, group in fragments.items():
        assert color in boxes, "every fragment color must have a matching output frame"
        r0, c0, r1, c1 = boxes[color]
        path = _path_from_fragments(group)
        for pr, pc in path:
            rr = int(r0 + 2 + pr)
            cc = int(c0 + 1 + pc)
            assert r0 < rr < r1 and c0 < cc < c1, "projected path must stay inside its frame"
            if (rr, cc) == (r0 + 1, c0 + 1):
                continue
            out[rr, cc] = color
    return out


def _sample_runs(n_runs: int, max_row: int, max_col: int) -> tuple[list[dict[str, int]], list[tuple[int, int]]] | None:
    for _ in range(200):
        runs: list[dict[str, int]] = []
        occupied = {(0, 0)}
        path = [(0, 0)]
        current = (0, 0)
        ok = True
        for i in range(n_runs):
            options: list[tuple[tuple[int, int], int, list[tuple[int, int]]]] = []
            lengths = [2, 3, 4]
            random.shuffle(lengths)
            dirs = list(DIRS)
            random.shuffle(dirs)
            for dr, dc in dirs:
                if i == 0 and (dr, dc) in {(-1, 0), (0, -1)}:
                    continue
                for length in lengths:
                    new_cells = []
                    valid = True
                    for step in range(1, length):
                        rr = current[0] + dr * step
                        cc = current[1] + dc * step
                        if rr < 0 or cc < 0 or rr > max_row or cc > max_col:
                            valid = False
                            break
                        if (rr, cc) in occupied:
                            valid = False
                            break
                        new_cells.append((rr, cc))
                    if valid:
                        options.append(((dr, dc), length, new_cells))
            if not options:
                ok = False
                break
            drdc, length, new_cells = random.choice(options)
            runs.append({"dr": drdc[0], "dc": drdc[1], "length": length})
            for cell in new_cells:
                occupied.add(cell)
                path.append(cell)
            current = path[-1]
        if ok:
            return runs, path
    return None


def _marker_candidates(interface: tuple[int, int], segment: set[tuple[int, int]]) -> list[tuple[int, int]]:
    out = []
    for nr, nc in _neighbors4(*interface):
        if (nr, nc) not in segment:
            out.append((nr, nc))
    return out


def _patch_from_run(run: dict[str, int], is_first: bool) -> dict[str, object]:
    dr = int(run["dr"])
    dc = int(run["dc"])
    length = int(run["length"])

    if dr != 0:
        segment = [(k, 0) for k in range(length)]
        low = segment[0]
        high = segment[-1]
        if is_first:
            interface = high if dr > 0 else low
        else:
            interface = low if dr > 0 else high
    else:
        segment = [(0, k) for k in range(length)]
        low = segment[0]
        high = segment[-1]
        if is_first:
            interface = high if dc > 0 else low
        else:
            interface = low if dc > 0 else high

    marker = random.choice(_marker_candidates(interface, set(segment)))
    all_cells = segment + [marker]
    min_r = min(r for r, _c in all_cells)
    min_c = min(c for _r, c in all_cells)
    norm_segment = [(r - min_r, c - min_c) for r, c in segment]
    norm_marker = (marker[0] - min_r, marker[1] - min_c)
    frag_r0 = min(r for r, _c in norm_segment)
    frag_c0 = min(c for _r, c in norm_segment)

    return {
        "height": max(r for r, _c in norm_segment + [norm_marker]) + 1,
        "width": max(c for _r, c in norm_segment + [norm_marker]) + 1,
        "segment": norm_segment,
        "marker": norm_marker,
        "frag_key_offset": (frag_r0, frag_c0),
    }


def _box_dims(path: list[tuple[int, int]]) -> tuple[int, int]:
    max_r = max(r for r, _c in path)
    max_c = max(c for _r, c in path)
    return max_r + 4, max_c + 3


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    left_w: int,
    color: int,
    patch: dict[str, object],
    last_key: tuple[int, int],
) -> tuple[int, int] | None:
    h, w = grid.shape
    ph = int(patch["height"])
    pw = int(patch["width"])
    key_off_r, key_off_c = patch["frag_key_offset"]
    valid_positions = []
    for top in range(h - ph + 1):
        for left in range(left_w - pw + 1):
            frag_key = (top + key_off_r, left + key_off_c)
            if frag_key <= last_key:
                continue
            cells = [(top + r, left + c) for r, c in patch["segment"]]
            cells.append((top + patch["marker"][0], left + patch["marker"][1]))
            if any(bool(blocked[r, c]) for r, c in cells):
                continue
            valid_positions.append((top, left))
    if not valid_positions:
        return None

    top, left = random.choice(valid_positions)
    cells = [(top + r, left + c) for r, c in patch["segment"]]
    marker = (top + patch["marker"][0], left + patch["marker"][1])
    for r, c in cells:
        grid[r, c] = color
    grid[marker] = MAROON
    for r, c in cells + [marker]:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(left_w, c + 2)):
                blocked[rr, cc] = True
    return top + key_off_r, left + key_off_c


def _place_boxes(
    grid: np.ndarray,
    sep: int,
    right_w: int,
    box_specs: list[tuple[int, int, int]],
    gap: int,
) -> dict[int, tuple[int, int, list[tuple[int, int]]]]:
    h, _w = grid.shape
    order = box_specs[:]
    random.shuffle(order)

    needed = sum(bh for _color, bh, _bw in order) + gap * (len(order) - 1)
    extra = h - needed
    assert extra >= 0
    slots = [0] * (len(order) + 1)
    for _ in range(extra):
        slots[random.randrange(len(slots))] += 1

    placements: dict[int, tuple[int, int, list[tuple[int, int]]]] = {}
    r = slots[0]
    for i, (color, box_h, box_w) in enumerate(order):
        c = sep + 1 + random.randint(0, right_w - box_w)
        grid[r : r + box_h, c] = GREY
        grid[r : r + box_h, c + box_w - 1] = GREY
        grid[r, c : c + box_w] = GREY
        grid[r + box_h - 1, c : c + box_w] = GREY
        grid[r + 1, c + 1] = color
        placements[color] = (r, c, [])
        r += box_h
        if i != len(order) - 1:
            r += gap + slots[i + 1]
    return placements


def _draw_expected(
    base: np.ndarray,
    box_info: dict[int, tuple[int, int, list[tuple[int, int]]]],
    paths: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    out = base.copy()
    for color, path in paths.items():
        r0, c0, _ = box_info[color]
        for pr, pc in path:
            out[r0 + 2 + pr, c0 + 1 + pc] = color
    return out


def _generate_candidate() -> tuple[np.ndarray, np.ndarray] | None:
    n_colors = random.randint(2, 3)
    colors = random.sample(PALETTE, n_colors)

    paths: dict[int, list[tuple[int, int]]] = {}
    patches_by_color: dict[int, list[dict[str, object]]] = {}
    box_specs: list[tuple[int, int, int]] = []
    total_patches = 0

    for color in colors:
        n_runs = random.randint(1, 4)
        sampled = _sample_runs(n_runs, max_row=4, max_col=6)
        if sampled is None:
            return None
        runs, path = sampled
        paths[color] = path
        patches = [_patch_from_run(run, is_first=(i == 0)) for i, run in enumerate(runs)]
        patches_by_color[color] = patches
        box_h, box_w = _box_dims(path)
        box_specs.append((color, box_h, box_w))
        total_patches += len(patches)

    right_w = max(box_w for _color, _box_h, box_w in box_specs) + random.randint(1, 3)
    gap = random.randint(1, 2)
    needed_h = sum(box_h for _color, box_h, _box_w in box_specs) + gap * (len(box_specs) - 1)
    h_min = max(needed_h, 12)
    h_max = min(30, max(h_min, 16 + total_patches))
    if h_min > h_max:
        return None
    h = random.randint(h_min, h_max)

    left_w_max = 30 - 1 - right_w
    left_w_min = 10
    if left_w_max < left_w_min:
        return None
    left_w = random.randint(left_w_min, left_w_max)
    w = left_w + 1 + right_w

    grid = np.full((h, w), BLACK, dtype=int)
    sep = left_w
    grid[:, sep] = GREY

    box_info = _place_boxes(grid, sep, right_w, box_specs, gap)
    for color, path in paths.items():
        r0, c0, _ = box_info[color]
        box_info[color] = (r0, c0, path)

    blocked = np.zeros((h, left_w), dtype=bool)
    color_order = sorted(colors, key=lambda col: -len(patches_by_color[col]))
    for color in color_order:
        last_key = (-1, -1)
        for patch in patches_by_color[color]:
            placed_key = _place_patch(grid, blocked, left_w, color, patch, last_key)
            if placed_key is None:
                return None
            last_key = placed_key

    expected = _draw_expected(grid, box_info, paths)
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _generate_candidate()
        if candidate is None:
            continue
        grid, expected = candidate
        try:
            actual = main(grid)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return grid
    raise RuntimeError("failed to generate a valid input for ARC task 2")
