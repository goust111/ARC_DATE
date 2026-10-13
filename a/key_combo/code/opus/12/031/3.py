from __future__ import annotations

import random

import numpy as np


BG = 0
BRACKET_COLOR = 5
MARKER_COLORS = (1, 2, 3)
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
ORIENTATIONS = ("tl", "tr", "bl", "br")
GEN_RETRIES = 400


def _components_of_color(grid: np.ndarray, color: int) -> list[set[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[set[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp = set()
            while stack:
                rr, cc = stack.pop()
                comp.add((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _make_bracket_cells(bbox: tuple[int, int, int, int], orientation: str) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if orientation == "tl":
        return {(r0, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    if orientation == "tr":
        return {(r0, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    if orientation == "bl":
        return {(r1, c) for c in range(c0, c1 + 1)} | {(r, c0) for r in range(r0, r1 + 1)}
    if orientation == "br":
        return {(r1, c) for c in range(c0, c1 + 1)} | {(r, c1) for r in range(r0, r1 + 1)}
    raise ValueError(f"unknown orientation: {orientation}")


def _match_corner_bracket(coords: set[tuple[int, int]]) -> dict | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, c0, r1, c1 = min(rows), min(cols), max(rows), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 2 or w < 2 or len(coords) != h + w - 1:
        return None
    bbox = (r0, c0, r1, c1)
    for orientation in ORIENTATIONS:
        cells = _make_bracket_cells(bbox, orientation)
        if coords == cells:
            return {
                "bbox": bbox,
                "orientation": orientation,
                "coords": cells,
                "vertical_len": h,
                "horizontal_len": w,
                "short_len": min(h, w),
                "long_len": max(h, w),
            }
    return None


def _detect_puzzle_objects(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    candidates = []
    for comp in _components_of_color(g, BRACKET_COLOR):
        match = _match_corner_bracket(comp)
        if match is None:
            continue
        r0, c0, r1, c1 = match["bbox"]
        markers = [
            (r, c)
            for r in range(r0, r1 + 1)
            for c in range(c0, c1 + 1)
            if (r, c) not in match["coords"] and int(g[r, c]) not in (BG, BRACKET_COLOR)
        ]
        if len(markers) != 1:
            continue
        marker = markers[0]
        marker_color = int(g[marker])
        if marker_color not in MARKER_COLORS:
            continue
        match["marker"] = marker
        match["marker_color"] = marker_color
        candidates.append(match)
    assert len(candidates) == 1, "expected exactly one grey corner bracket with one internal marker"
    return candidates[0]


def _apply_rule_with_metadata(
    grid: np.ndarray,
    bracket_cells: set[tuple[int, int]],
    marker: tuple[int, int],
    shift: int,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    rolled = np.roll(g, shift=shift, axis=0)
    marker_after = ((marker[0] + shift) % h, marker[1])
    marker_color = int(rolled[marker_after])

    out = rolled.copy()
    out[out == marker_color] = BG
    out[marker_after] = marker_color
    for r, c in bracket_cells:
        out[(r + shift) % h, c] = BG
    return out


def _apply_wrong_bracket_reference(
    grid: np.ndarray,
    bracket_cells: set[tuple[int, int]],
    marker: tuple[int, int],
    shift: int,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    marker_color = int(g[marker])
    tmp = g.copy()
    tmp[tmp == marker_color] = BG
    tmp[marker] = marker_color
    rolled = np.roll(tmp, shift=shift, axis=0)
    for r, c in bracket_cells:
        rolled[r, c] = BG
    return rolled


def main(grid: np.ndarray) -> np.ndarray:
    """
    Roll the whole grid downward by the corner bracket's short arm, then erase the shifted marker color everywhere else and remove the bracket.

    Input structure:
    - The background is black.
    - There is exactly one grey L-shaped corner bracket in a corner region, with unequal arm lengths and one non-grey marker pixel inside its bounding box.
    - The marker color is blue, red, or green, and the rest of the grid contains a multicolor pattern including more pixels of that same color.

    Rule:
    - Measure the two grey bracket arms and take the shorter arm length L.
    - Cyclically shift every row of the entire grid downward by L, including the bracket and the marker.
    - In the shifted grid, erase every pixel whose color matches the shifted marker, except for the marker pixel itself.
    - Erase the shifted grey bracket pixels and keep all other shifted pixels unchanged.

    Key insight:
    - The same bracket carries two independent controls: its geometry encodes the roll amount, while its enclosed marker names the color to erase after the roll.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    info = _detect_puzzle_objects(g)
    return _apply_rule_with_metadata(g, info["coords"], info["marker"], info["short_len"])


def _normalize_shape(cells: set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    return tuple(sorted((r - min_r, c - min_c) for r, c in cells))


def _grow_connected_shape(size: int) -> tuple[tuple[int, int], ...]:
    shape = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(shape) < size:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        shape.add(cell)
        for dr, dc in DIR4:
            nbr = (cell[0] + dr, cell[1] + dc)
            if nbr not in shape:
                frontier.add(nbr)
    return _normalize_shape(shape)


def _halo4(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    halo = set(cells)
    for r, c in cells:
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                halo.add((nr, nc))
    return halo


def _expand_bbox(bbox: tuple[int, int, int, int], pad: int, h: int, w: int) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {
        (r, c)
        for r in range(max(0, r0 - pad), min(h - 1, r1 + pad) + 1)
        for c in range(max(0, c0 - pad), min(w - 1, c1 + pad) + 1)
    }


def _place_shape(
    grid: np.ndarray,
    color: int,
    shape: tuple[tuple[int, int], ...],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = grid.shape
    max_r = max(r for r, _ in shape)
    max_c = max(c for _, c in shape)
    placements = []
    for ar in range(h - max_r):
        for ac in range(w - max_c):
            cells = {(ar + r, ac + c) for r, c in shape}
            if any(cell in blocked for cell in cells):
                continue
            placements.append(cells)
    if not placements:
        return None
    cells = random.choice(placements)
    for r, c in cells:
        grid[r, c] = color
    blocked.update(_halo4(cells, h, w))
    return cells


def _make_bracket_spec(h: int, w: int) -> dict | None:
    orientation = random.choice(ORIENTATIONS)
    short_len = random.randint(2, 4)
    short_vertical = random.choice((True, False))
    if short_vertical:
        vlen = short_len
        hlen_lo = max(4, short_len + 1)
        hlen_hi = min(7, w)
        if hlen_lo > hlen_hi:
            return None
        hlen = random.randint(hlen_lo, hlen_hi)
    else:
        hlen = short_len
        vlen_lo = max(4, short_len + 1)
        vlen_hi = min(7, h)
        if vlen_lo > vlen_hi:
            return None
        vlen = random.randint(vlen_lo, vlen_hi)

    top_margin = random.randint(0, min(2, h - vlen))
    left_margin = random.randint(0, min(2, w - hlen))
    bottom_margin = random.randint(0, min(2, h - vlen))
    right_margin = random.randint(0, min(2, w - hlen))

    if orientation == "tl":
        r0, c0 = top_margin, left_margin
    elif orientation == "tr":
        r0, c0 = top_margin, w - hlen - right_margin
    elif orientation == "bl":
        r0, c0 = h - vlen - bottom_margin, left_margin
    else:
        r0, c0 = h - vlen - bottom_margin, w - hlen - right_margin

    bbox = (r0, c0, r0 + vlen - 1, c0 + hlen - 1)
    cells = _make_bracket_cells(bbox, orientation)
    interior = [
        (r, c)
        for r in range(bbox[0], bbox[2] + 1)
        for c in range(bbox[1], bbox[3] + 1)
        if (r, c) not in cells
    ]
    if not interior:
        return None
    marker = random.choice(interior)
    return {
        "orientation": orientation,
        "bbox": bbox,
        "coords": cells,
        "marker": marker,
        "short_len": short_len,
        "long_len": max(vlen, hlen),
    }


def _marker_specs_blob(marker_color: int) -> list[tuple[int, int]]:
    total = random.randint(2, 5)
    if total >= 4 and random.random() < 0.5:
        first = random.randint(2, total - 2)
        return [(marker_color, first), (marker_color, total - first)]
    return [(marker_color, total)]


def _marker_specs_scatter(marker_color: int) -> list[tuple[int, int]]:
    total = random.randint(2, 5)
    specs = []
    while total > 0:
        size = 2 if total >= 2 and random.random() < 0.55 else 1
        size = min(size, total)
        specs.append((marker_color, size))
        total -= size
    return specs


def _paint_pattern(
    grid: np.ndarray,
    bracket_bbox: tuple[int, int, int, int],
    marker_color: int,
) -> bool:
    h, w = grid.shape
    blocked = _expand_bbox(bracket_bbox, 1, h, w)
    mode = random.choice(("blobs", "scatter"))
    num_colors = random.randint(3, 5)
    other_colors = random.sample([c for c in PATTERN_COLORS if c != marker_color], num_colors - 1)
    palette = [marker_color] + other_colors

    if mode == "blobs":
        specs = _marker_specs_blob(marker_color)
        for color in other_colors:
            specs.append((color, random.randint(2, 4)))
        for _ in range(random.randint(0, 2)):
            specs.append((random.choice(palette), random.randint(1, 3)))
        random.shuffle(specs)
        for color, size in specs:
            shape = _grow_connected_shape(size)
            if _place_shape(grid, color, shape, blocked) is None:
                return False
        return True

    specs = _marker_specs_scatter(marker_color)
    for color in other_colors:
        count = random.randint(2, 4)
        while count > 0:
            size = 2 if count >= 2 and random.random() < 0.45 else 1
            size = min(size, count)
            specs.append((color, size))
            count -= size
    for _ in range(random.randint(0, 3)):
        specs.append((random.choice(palette), random.choice((1, 2))))
    random.shuffle(specs)
    shape_bank = {
        1: [((0, 0),)],
        2: [((0, 0), (0, 1)), ((0, 0), (1, 0))],
    }
    for color, size in specs:
        shape = random.choice(shape_bank[size])
        if _place_shape(grid, color, shape, blocked) is None:
            return False
    return True


def _oracle_output(grid: np.ndarray, meta: dict) -> np.ndarray:
    return _apply_rule_with_metadata(grid, meta["coords"], meta["marker"], meta["short_len"])


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(9, 14)
    w = random.randint(9, 14)
    meta = _make_bracket_spec(h, w)
    if meta is None:
        return None

    marker_color = random.choice(MARKER_COLORS)
    grid = np.zeros((h, w), dtype=int)
    for r, c in meta["coords"]:
        grid[r, c] = BRACKET_COLOR
    grid[meta["marker"]] = marker_color

    if not _paint_pattern(grid, meta["bbox"], marker_color):
        return None

    visible_colors = {int(v) for v in np.unique(grid) if int(v) not in (BG, BRACKET_COLOR)}
    if len(visible_colors) < 3:
        return None
    if int(np.count_nonzero(grid == marker_color)) < 3:
        return None

    detected = _detect_puzzle_objects(grid)
    if detected["coords"] != meta["coords"]:
        return None
    if detected["marker"] != meta["marker"]:
        return None
    if detected["short_len"] != meta["short_len"] or detected["long_len"] != meta["long_len"]:
        return None

    rolled = np.roll(grid, shift=meta["short_len"], axis=0)
    expected = _oracle_output(grid, meta)
    marker_after = ((meta["marker"][0] + meta["short_len"]) % h, meta["marker"][1])
    if int(rolled[marker_after]) != marker_color:
        return None
    if int(np.count_nonzero(rolled == marker_color)) < 2:
        return None
    if int(np.count_nonzero(expected == marker_color)) != 1:
        return None
    if int(np.count_nonzero(rolled == BRACKET_COLOR)) != len(meta["coords"]):
        return None
    if int(np.count_nonzero(expected == BRACKET_COLOR)) != 0:
        return None

    long_shift_out = _apply_rule_with_metadata(grid, meta["coords"], meta["marker"], meta["long_len"])
    if np.array_equal(expected, long_shift_out):
        return None
    wrong_frame_out = _apply_wrong_bracket_reference(grid, meta["coords"], meta["marker"], meta["short_len"])
    if np.array_equal(expected, wrong_frame_out):
        return None
    roll_only = rolled.copy()
    if np.array_equal(expected, roll_only):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 3")
