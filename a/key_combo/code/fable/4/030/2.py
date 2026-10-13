from __future__ import annotations

import random
from collections import Counter

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

DIRS4: tuple[Coord, ...] = ((-1, 0), (1, 0), (0, -1), (0, 1))
PALETTE: tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> list[Coord]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _bbox_from_coords(coords: list[Coord] | set[Coord]) -> BBox:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _components(mask: np.ndarray) -> list[list[Coord]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[Coord]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _is_rect_frame(coords: list[Coord]) -> bool:
    if len(coords) < 8:
        return False
    coord_set = set(coords)
    r0, c0, r1, c1 = _bbox_from_coords(coord_set)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    expected: set[Coord] = set()
    for c in range(c0, c1 + 1):
        expected.add((r0, c))
        expected.add((r1, c))
    for r in range(r0, r1 + 1):
        expected.add((r, c0))
        expected.add((r, c1))
    return coord_set == expected


def _find_frame(grid: np.ndarray, bg: int) -> tuple[int, set[Coord], BBox]:
    frame_hits: list[tuple[int, set[Coord], BBox]] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for comp in _components(grid == color):
            if _is_rect_frame(comp):
                comp_set = set(comp)
                frame_hits.append((color, comp_set, _bbox_from_coords(comp_set)))
    assert len(frame_hits) == 1, "expected exactly one hollow rectangular frame"
    return frame_hits[0]


def _ordered_path(path_cells: set[Coord], start: Coord) -> list[Coord]:
    degrees = {
        cell: sum((nb in path_cells) for nb in _neighbors4(*cell))
        for cell in path_cells
    }
    endpoints = [cell for cell, degree in degrees.items() if degree == 1]
    assert len(endpoints) == 2, "path must have exactly two endpoints"
    assert start in endpoints, "start must be a path endpoint"
    assert all(degree in (1, 2) for degree in degrees.values()), "path must not branch"

    ordered: list[Coord] = []
    prev: Coord | None = None
    cur = start
    while True:
        ordered.append(cur)
        next_cells = [nb for nb in _neighbors4(*cur) if nb in path_cells and nb != prev]
        if not next_cells:
            break
        assert len(next_cells) == 1, "path traversal must be unique"
        prev, cur = cur, next_cells[0]
    assert len(ordered) == len(path_cells), "path traversal must cover all path cells"
    return ordered


def _runs(values: list[int]) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for value in values:
        if out and out[-1][0] == value:
            out[-1] = (value, out[-1][1] + 1)
        else:
            out.append((value, 1))
    return out


def _paint_ring(out: np.ndarray, box: BBox, color: int, thickness: int) -> BBox:
    r0, c0, r1, c1 = box
    assert thickness >= 1
    assert r0 + thickness <= r1 - thickness + 1, "ring must leave remaining rows"
    assert c0 + thickness <= c1 - thickness + 1, "ring must leave remaining cols"
    out[r0 : r0 + thickness, c0 : c1 + 1] = color
    out[r1 - thickness + 1 : r1 + 1, c0 : c1 + 1] = color
    out[r0 : r1 + 1, c0 : c0 + thickness] = color
    out[r0 : r1 + 1, c1 - thickness + 1 : c1 + 1] = color
    return r0 + thickness, c0 + thickness, r1 - thickness, c1 - thickness


def main(grid: np.ndarray) -> np.ndarray:
    """Translate the color runs of the frame-touching path into concentric bands inside the frame.

    Input structure:
    - The background is empty except for one one-pixel-thick hollow rectangular frame.
    - Outside the frame there is one multicolor one-pixel-wide 4-connected path.
    - Exactly one path endpoint touches the outer wall of the frame; the other endpoint is free.

    Rule:
    - Identify the hollow frame and erase the entire path from the output.
    - Traverse the path starting from the endpoint that touches the frame, then compress the visited
      colors into maximal same-color runs.
    - Fill the frame interior from outermost to innermost: each run paints the next rectangular band
      with that run color, using the run length as the band's thickness, and the last run fills the
      remaining center.

    Key insight:
    - The path geometry is a distraction; only the ordered color runs, read from the frame-touching
      endpoint, matter, and they specify the interior band colors and thicknesses.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    _frame_color, frame_cells, frame_bbox = _find_frame(g, bg)
    r0, c0, r1, c1 = frame_bbox
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame interior must be non-empty"

    path_cells = {
        (int(r), int(c))
        for r, c in np.argwhere(g != bg)
        if (int(r), int(c)) not in frame_cells
    }
    assert path_cells, "expected a path outside the frame"
    path_components = _components(_build_path_mask(path_cells, g.shape))
    assert len(path_components) == 1, "expected exactly one path component"
    path_cells = set(path_components[0])

    touching_endpoints = []
    for cell in path_cells:
        degree = sum((nb in path_cells) for nb in _neighbors4(*cell))
        if degree != 1:
            continue
        if any(nb in frame_cells for nb in _neighbors4(*cell)):
            touching_endpoints.append(cell)
    assert len(touching_endpoints) == 1, "expected exactly one frame-touching endpoint"
    ordered = _ordered_path(path_cells, touching_endpoints[0])
    values = [int(g[r, c]) for r, c in ordered]
    run_spec = _runs(values)
    assert run_spec, "expected at least one color run"

    out = g.copy()
    for r, c in path_cells:
        out[r, c] = bg

    inner_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    cur_box = inner_box
    for idx, (color, length) in enumerate(run_spec):
        cr0, cc0, cr1, cc1 = cur_box
        assert cr0 <= cr1 and cc0 <= cc1, "bands exceed frame interior"
        if idx == len(run_spec) - 1:
            out[cr0 : cr1 + 1, cc0 : cc1 + 1] = color
        else:
            cur_box = _paint_ring(out, cur_box, color, length)

    return out


def _partition_depth(total: int) -> list[int]:
    max_parts = min(5, total)
    parts = random.randint(2, max_parts) if total >= 2 else 1
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out = []
    for cut in cuts + [total]:
        out.append(cut - prev)
        prev = cut
    return out


def _build_path_mask(path_cells: set[Coord], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in path_cells:
        mask[r, c] = True
    return mask


def _turn_positions(path: list[Coord]) -> set[int]:
    turns: set[int] = set()
    for i in range(1, len(path) - 1):
        dr1 = path[i][0] - path[i - 1][0]
        dc1 = path[i][1] - path[i - 1][1]
        dr2 = path[i + 1][0] - path[i][0]
        dc2 = path[i + 1][1] - path[i][1]
        if (dr1, dc1) != (dr2, dc2):
            turns.add(i)
    return turns


def _shape_decoupled_from_runs(path: list[Coord], run_lengths: list[int]) -> bool:
    if len(path) < 4:
        return True
    turns = _turn_positions(path)
    if not turns:
        return False
    boundary_positions = set()
    acc = 0
    for length in run_lengths[:-1]:
        acc += length
        boundary_positions.add(acc - 1)
    return bool(turns - boundary_positions) or bool(boundary_positions - turns)


def _touches_frame(cell: Coord, frame_cells: set[Coord]) -> bool:
    return any(nb in frame_cells for nb in _neighbors4(*cell))


def _dir_priority(prev_dir: Coord | None) -> list[Coord]:
    dirs = list(DIRS4)
    random.shuffle(dirs)
    if prev_dir is None:
        return dirs
    straight = [d for d in dirs if d == prev_dir]
    turns = [d for d in dirs if d != prev_dir and d != (-prev_dir[0], -prev_dir[1])]
    back = [d for d in dirs if d == (-prev_dir[0], -prev_dir[1])]
    if random.random() < 0.75:
        return turns + straight + back
    return straight + turns + back


def _grow_path(
    shape: tuple[int, int],
    frame_cells: set[Coord],
    blocked_box: BBox,
    path_len: int,
    start: Coord,
    outward: Coord,
) -> list[Coord] | None:
    h, w = shape
    r0, c0, r1, c1 = blocked_box

    def in_blocked(cell: Coord) -> bool:
        r, c = cell
        return r0 <= r <= r1 and c0 <= c <= c1

    second = (start[0] + outward[0], start[1] + outward[1])
    if not (0 <= second[0] < h and 0 <= second[1] < w):
        return None
    if in_blocked(second) or _touches_frame(second, frame_cells):
        return None

    path = [start, second]
    used = {start, second}

    def valid_next(cur: Coord, nxt: Coord) -> bool:
        nr, nc = nxt
        if not (0 <= nr < h and 0 <= nc < w):
            return False
        if nxt in used or in_blocked(nxt):
            return False
        if _touches_frame(nxt, frame_cells):
            return False
        for nb in _neighbors4(nr, nc):
            if nb in used and nb != cur:
                return False
        return True

    def dfs(prev_dir: Coord) -> list[Coord] | None:
        if len(path) == path_len:
            return list(path)
        cur = path[-1]
        for dr, dc in _dir_priority(prev_dir):
            nxt = (cur[0] + dr, cur[1] + dc)
            if not valid_next(cur, nxt):
                continue
            path.append(nxt)
            used.add(nxt)
            found = dfs((dr, dc))
            if found is not None:
                return found
            used.remove(nxt)
            path.pop()
        return None

    return dfs(outward)


def _fill_from_runs(grid: np.ndarray, path: list[Coord], run_colors: list[int]) -> np.ndarray:
    inp = grid.copy()
    for (r, c), color in zip(path, run_colors):
        inp[r, c] = color
    return inp


def _sample_case() -> tuple[np.ndarray, np.ndarray] | None:
    inner_h = random.randint(4, 12)
    inner_w = random.randint(4, 12)
    total_depth = (min(inner_h, inner_w) + 1) // 2
    run_lengths = _partition_depth(total_depth)
    path_len = sum(run_lengths)

    frame_color = random.choice(PALETTE)
    path_palette = [color for color in PALETTE if color != frame_color]
    if len(run_lengths) > len(path_palette):
        return None
    run_colors_unique = random.sample(path_palette, len(run_lengths))
    if run_colors_unique == run_colors_unique[::-1]:
        return None

    path_colors: list[int] = []
    for color, length in zip(run_colors_unique, run_lengths):
        path_colors.extend([color] * length)

    side = random.choice(["top", "bottom", "left", "right"])
    side_margin = random.randint(path_len + 1, path_len + 7)
    opp_margin = random.randint(1, 6)
    other_margin_a = random.randint(2, 6)
    other_margin_b = random.randint(2, 6)

    outer_h = inner_h + 2
    outer_w = inner_w + 2
    if side in ("top", "bottom"):
        top_margin, bottom_margin = (side_margin, opp_margin) if side == "top" else (opp_margin, side_margin)
        left_margin, right_margin = other_margin_a, other_margin_b
    else:
        left_margin, right_margin = (side_margin, opp_margin) if side == "left" else (opp_margin, side_margin)
        top_margin, bottom_margin = other_margin_a, other_margin_b

    height = outer_h + top_margin + bottom_margin
    width = outer_w + left_margin + right_margin
    if height > 30 or width > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    r0 = top_margin
    c0 = left_margin
    r1 = r0 + outer_h - 1
    c1 = c0 + outer_w - 1
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    frame_cells = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }

    if side == "top":
        start = (r0 - 1, random.randint(c0 + 1, c1 - 1))
        outward = (-1, 0)
    elif side == "bottom":
        start = (r1 + 1, random.randint(c0 + 1, c1 - 1))
        outward = (1, 0)
    elif side == "left":
        start = (random.randint(r0 + 1, r1 - 1), c0 - 1)
        outward = (0, -1)
    else:
        start = (random.randint(r0 + 1, r1 - 1), c1 + 1)
        outward = (0, 1)

    path = _grow_path(grid.shape, frame_cells, (r0, c0, r1, c1), path_len, start, outward)
    if path is None:
        return None
    if not _shape_decoupled_from_runs(path, run_lengths):
        return None

    inp = _fill_from_runs(grid, path, path_colors)
    out = main(inp)
    if out.shape != inp.shape:
        return None

    expected = grid.copy()
    cur_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
    for idx, (color, length) in enumerate(zip(run_colors_unique, run_lengths)):
        if idx == len(run_lengths) - 1:
            er0, ec0, er1, ec1 = cur_box
            expected[er0 : er1 + 1, ec0 : ec1 + 1] = color
        else:
            cur_box = _paint_ring(expected, cur_box, color, length)
    if not np.array_equal(out, expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _sample_case()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate valid input for puzzle 2")
