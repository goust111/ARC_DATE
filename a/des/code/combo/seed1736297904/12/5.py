from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
NON_SPECIAL_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

MIN_PAIRS = 2
MAX_PAIRS = 5
MIN_PATTERN_LEN = 2
MAX_PATTERN_LEN = 4
INTERIOR_WIDTH = 6  # key + up to 4 pattern cells + trailing black pad
MIN_SIZE = 16
MAX_SIZE = 28
MIN_STRAIGHT_LEN = 6  # inclusive anchor-to-anchor span
MAX_PLACEMENT_ATTEMPTS = 600
MAX_GENERATION_ATTEMPTS = 400
PAIR_TYPES = ("horizontal", "vertical", "elbow")
FRAME_CORNERS = ("tl", "tr", "bl", "br")


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> BBox:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _find_dictionary_bbox(grid: np.ndarray) -> BBox:
    grey_components = _components(grid == GREY)
    candidates: list[BBox] = []
    h, w = grid.shape
    for coords in grey_components:
        r0, c0, r1, c1 = _bbox(coords)
        if not ((r0 == 0 or r1 == h - 1) and (c0 == 0 or c1 == w - 1)):
            continue
        if r1 - r0 < 2 or c1 - c0 < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        border = np.zeros_like(sub, dtype=bool)
        border[0, :] = True
        border[-1, :] = True
        border[:, 0] = True
        border[:, -1] = True
        if np.all(sub[border] == GREY) and np.all(sub[~border] != GREY):
            candidates.append((r0, c0, r1, c1))
    assert len(candidates) == 1, "expected one grey dictionary frame"
    return candidates[0]


def _parse_dictionary(grid: np.ndarray, bbox: BBox) -> dict[int, tuple[int, ...]]:
    r0, c0, r1, c1 = bbox
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    mapping: dict[int, tuple[int, ...]] = {}
    for row in inner:
        non_black = np.flatnonzero(row != BLACK)
        if len(non_black) == 0:
            continue
        assert int(non_black[0]) == 0, "dictionary key must be the leftmost interior cell"
        key = int(row[0])
        assert key not in {BLACK, GREY}
        pat: list[int] = []
        idx = 1
        while idx < row.shape[0] and int(row[idx]) != BLACK:
            color = int(row[idx])
            assert color != GREY and color != key
            pat.append(color)
            idx += 1
        assert MIN_PATTERN_LEN <= len(pat) <= MAX_PATTERN_LEN
        assert np.all(row[idx:] == BLACK)
        mapping[key] = tuple(pat)
    assert MIN_PAIRS <= len(mapping) <= MAX_PAIRS
    return mapping


def _anchor_pairs(grid: np.ndarray, mapping: dict[int, tuple[int, ...]], bbox: BBox) -> dict[int, tuple[Coord, Coord]]:
    r0, c0, r1, c1 = bbox
    outside = np.ones(grid.shape, dtype=bool)
    outside[r0 : r1 + 1, c0 : c1 + 1] = False

    outside_colors = {
        int(color)
        for color in np.unique(grid[outside])
        if int(color) not in {BLACK, GREY}
    }
    assert outside_colors == set(mapping), "outside colors must match dictionary keys"

    pairs: dict[int, tuple[Coord, Coord]] = {}
    for color in mapping:
        coords = np.argwhere(outside & (grid == color))
        assert coords.shape[0] == 2, "each anchor color must appear exactly twice outside the dictionary"
        pairs[color] = ((int(coords[0, 0]), int(coords[0, 1])), (int(coords[1, 0]), int(coords[1, 1])))
    return pairs


def _path_cells(a: Coord, b: Coord) -> list[Coord]:
    ar, ac = a
    br, bc = b
    if ar == br:
        if ac > bc:
            ac, bc = bc, ac
            ar, br = br, ar
        return [(ar, c) for c in range(ac, bc + 1)]
    if ac == bc:
        if ar > br:
            ar, br = br, ar
            ac, bc = bc, ac
        return [(r, ac) for r in range(ar, br + 1)]

    if ac < bc:
        sr, sc, er, ec = ar, ac, br, bc
    else:
        sr, sc, er, ec = br, bc, ar, ac

    horiz = [(sr, c) for c in range(sc, ec + 1)]
    if er > sr:
        vert = [(r, ec) for r in range(sr + 1, er + 1)]
    else:
        vert = [(r, ec) for r in range(sr - 1, er - 1, -1)]
    return horiz + vert


def _paint_paths(
    grid: np.ndarray,
    mapping: dict[int, tuple[int, ...]],
    pairs: dict[int, tuple[Coord, Coord]],
) -> np.ndarray:
    out = grid.copy()
    anchor_cells = {pos for pair in pairs.values() for pos in pair}
    for color, pattern in mapping.items():
        cells = _path_cells(*pairs[color])
        for i, (r, c) in enumerate(cells[1:-1]):
            if (r, c) in anchor_cells:
                continue
            if int(out[r, c]) in {BLACK, *pattern}:
                out[r, c] = pattern[i % len(pattern)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the path between each same-color anchor pair with the repeating motif
    stored in the grey-framed dictionary.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bbox = _find_dictionary_bbox(g)
    mapping = _parse_dictionary(g, bbox)
    pairs = _anchor_pairs(g, mapping, bbox)
    return _paint_paths(g, mapping, pairs)


def _dictionary_bbox_for_corner(h: int, w: int, frame_h: int, frame_w: int, corner: str) -> BBox:
    if corner == "tl":
        return (0, 0, frame_h - 1, frame_w - 1)
    if corner == "tr":
        return (0, w - frame_w, frame_h - 1, w - 1)
    if corner == "bl":
        return (h - frame_h, 0, h - 1, frame_w - 1)
    assert corner == "br"
    return (h - frame_h, w - frame_w, h - 1, w - 1)


def _dilate(coords: list[Coord], h: int, w: int, pad: int = 1) -> list[Coord]:
    out: set[Coord] = set()
    for r, c in coords:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return list(out)


def _valid_path(coords: list[Coord], reserved: np.ndarray) -> bool:
    seen: set[Coord] = set()
    for r, c in coords:
        if reserved[r, c]:
            return False
        if (r, c) in seen:
            return False
        seen.add((r, c))
    return True


def _random_horizontal(h: int, w: int) -> list[Coord]:
    r = random.randrange(h)
    left = random.randrange(w - MIN_STRAIGHT_LEN + 1)
    right = random.randrange(left + MIN_STRAIGHT_LEN - 1, w)
    return [(r, c) for c in range(left, right + 1)]


def _random_vertical(h: int, w: int) -> list[Coord]:
    c = random.randrange(w)
    top = random.randrange(h - MIN_STRAIGHT_LEN + 1)
    bottom = random.randrange(top + MIN_STRAIGHT_LEN - 1, h)
    return [(r, c) for r in range(top, bottom + 1)]


def _random_elbow(h: int, w: int) -> list[Coord]:
    left = random.randrange(w - 4)
    right = random.randrange(left + 3, w)
    row1 = random.randrange(h)
    row2_choices = [r for r in range(h) if r != row1]
    row2 = random.choice(row2_choices)
    return _path_cells((row1, left), (row2, right))


def _sample_pattern(pattern_palette: tuple[int, ...], used: set[tuple[int, ...]]) -> tuple[int, ...]:
    for _ in range(200):
        length = random.randint(MIN_PATTERN_LEN, MAX_PATTERN_LEN)
        seq = [random.choice(pattern_palette)]
        while len(seq) < length:
            choices = [color for color in pattern_palette if color != seq[-1]]
            seq.append(random.choice(choices))
        if len(set(seq)) < 2:
            continue
        pattern = tuple(seq)
        if pattern not in used:
            used.add(pattern)
            return pattern
    raise RuntimeError("failed to sample a nontrivial pattern")


def _place_pairs(
    h: int,
    w: int,
    colors: list[int],
    reserved: np.ndarray,
) -> dict[int, tuple[Coord, Coord]] | None:
    pair_types: list[str]
    if len(colors) >= 3:
        pair_types = ["horizontal", "vertical", "elbow"]
        while len(pair_types) < len(colors):
            pair_types.append(random.choice(PAIR_TYPES))
        random.shuffle(pair_types)
    else:
        pair_types = [random.choice(PAIR_TYPES) for _ in colors]

    pairs: dict[int, tuple[Coord, Coord]] = {}
    working = reserved.copy()
    generators = {
        "horizontal": _random_horizontal,
        "vertical": _random_vertical,
        "elbow": _random_elbow,
    }

    for color, pair_type in zip(colors, pair_types):
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            coords = generators[pair_type](h, w)
            if len(coords) < MIN_STRAIGHT_LEN:
                continue
            if not _valid_path(coords, working):
                continue
            a = coords[0]
            b = coords[-1]
            pairs[color] = (a, b)
            for rr, cc in _dilate(coords, h, w, pad=1):
                working[rr, cc] = True
            break
        else:
            return None
    return pairs


def _paint_dictionary(
    grid: np.ndarray,
    mapping: dict[int, tuple[int, ...]],
    corner: str,
) -> BBox:
    frame_h = len(mapping) + 2
    frame_w = INTERIOR_WIDTH + 2
    r0, c0, r1, c1 = _dictionary_bbox_for_corner(grid.shape[0], grid.shape[1], frame_h, frame_w, corner)
    grid[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    rows = list(mapping.items())
    random.shuffle(rows)
    for idx, (key, pattern) in enumerate(rows):
        rr = r0 + 1 + idx
        cc = c0 + 1
        grid[rr, cc] = key
        for j, color in enumerate(pattern, start=1):
            grid[rr, cc + j] = color
    return (r0, c0, r1, c1)


def _build_reserved(h: int, w: int, bbox: BBox) -> np.ndarray:
    reserved = np.zeros((h, w), dtype=bool)
    r0, c0, r1, c1 = bbox
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(h - 1, r1 + 1)
    cc1 = min(w - 1, c1 + 1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
    return reserved


def _generate_one() -> np.ndarray | None:
    n_pairs = random.randint(MIN_PAIRS, MAX_PAIRS)
    h = random.randint(max(MIN_SIZE, n_pairs + 8), MAX_SIZE)
    w = random.randint(max(MIN_SIZE, INTERIOR_WIDTH + 8), MAX_SIZE)

    palette = list(NON_SPECIAL_COLORS)
    random.shuffle(palette)
    anchor_colors = palette[:n_pairs]
    pattern_palette = tuple(palette[n_pairs : n_pairs + 3])
    if len(pattern_palette) < 2:
        return None

    used_patterns: set[tuple[int, ...]] = set()
    mapping = {color: _sample_pattern(pattern_palette, used_patterns) for color in anchor_colors}
    corner = random.choice(FRAME_CORNERS)

    frame_h = len(mapping) + 2
    frame_w = INTERIOR_WIDTH + 2
    if h < frame_h + 4 or w < frame_w + 4:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    bbox = _paint_dictionary(grid, mapping, corner)
    reserved = _build_reserved(h, w, bbox)
    pairs = _place_pairs(h, w, anchor_colors, reserved)
    if pairs is None:
        return None

    for color, (a, b) in pairs.items():
        grid[a] = color
        grid[b] = color

    expected = _paint_paths(grid, mapping, pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
