from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
SEP_COLOR = 5
TOP_PALETTE = (1, 2, 3, 4)
BOTTOM_PALETTE = (6, 7, 8, 9)

GEN_HEIGHTS = (15, 17)
GEN_WIDTH_RANGE = (15, 19)
GEN_TOP_COUNT_RANGE = (3, 4)
GEN_BOTTOM_COUNT_RANGE = (3, 4)
GEN_MATCHED_HOLE_PROB = 0.6
GEN_OTHER_HOLE_PROB = 0.2
GEN_ATTEMPTS = 400
GEN_SHAPE_ATTEMPTS = 120
GEN_PLACEMENT_ATTEMPTS = 220


Coord = tuple[int, int]


def _get_bg(grid: np.ndarray) -> int:
    counts = Counter(int(v) for v in np.asarray(grid).ravel())
    return int(counts.most_common(1)[0][0])


def _neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def _normalize(coords: set[Coord] | list[Coord]) -> list[Coord]:
    pts = list(coords)
    min_r = min(r for r, _ in pts)
    min_c = min(c for _, c in pts)
    return sorted((r - min_r, c - min_c) for r, c in pts)


def _bbox(coords: list[Coord] | set[Coord]) -> tuple[int, int, int, int]:
    pts = list(coords)
    return (
        min(r for r, _ in pts),
        min(c for _, c in pts),
        max(r for r, _ in pts),
        max(c for _, c in pts),
    )


def _translate(coords: list[Coord], r0: int, c0: int) -> list[Coord]:
    return [(r0 + r, c0 + c) for r, c in coords]


def _mirror_across_separator(coords: list[Coord] | set[Coord], sep: int) -> list[Coord]:
    return sorted((2 * sep - r, c) for r, c in coords)


def _component_patch(coords: list[Coord] | set[Coord]) -> tuple[np.ndarray, int, int]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        patch[r - r0, c - c0] = True
    return patch, r0, c0


def _hole_cells(coords: list[Coord] | set[Coord]) -> set[Coord]:
    patch, r0, c0 = _component_patch(coords)
    h, w = patch.shape
    seen = np.zeros((h, w), dtype=bool)
    q: deque[Coord] = deque()

    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if not patch[r, c] and not seen[r, c]:
                seen[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not patch[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))

    holes = set()
    for r in range(h):
        for c in range(w):
            if not patch[r, c] and not seen[r, c]:
                holes.add((r0 + r, c0 + c))
    return holes


def _halo_cells(coords: list[Coord] | set[Coord], shape: tuple[int, int], sep: int) -> set[Coord]:
    H, W = shape
    obj = set(coords)
    holes = _hole_cells(coords)
    halo = set()
    for r, c in obj:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if nr == sep:
                    continue
                if (nr, nc) in obj or (nr, nc) in holes:
                    continue
                halo.add((nr, nc))
    return halo


def _extract_components(grid: np.ndarray, row_lo: int, row_hi: int, banned: set[int]) -> list[dict]:
    H, W = grid.shape
    seen = np.zeros((H, W), dtype=bool)
    comps: list[dict] = []

    for r in range(row_lo, row_hi):
        for c in range(W):
            color = int(grid[r, c])
            if color in banned or seen[r, c]:
                continue

            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if row_lo <= nr < row_hi and 0 <= nc < W and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            coords = sorted(coords)
            r0, c0, r1, c1 = _bbox(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "coord_key": frozenset(coords),
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Mirror-match each bottom sprite across the separator, dilating matched ones with the mirrored top color and erasing non-matches.

    Input structure:
    - The grid has black background and exactly one solid non-background separator row splitting equal-height top and bottom halves.
    - The top half contains several non-touching monochrome 4-connected sprites, and the bottom half contains several other monochrome 4-connected sprites.
    - Some bottom sprites are exact vertical reflections of top sprites across the separator row, while others are not.

    Rule:
    - For each bottom sprite, reflect its cells across the separator row and compare the reflected shape-and-position mask against the top-half sprites.
    - If the reflected mask exactly matches one top sprite, keep the bottom sprite and add a one-pixel halo around it using that top sprite's color, but only on background cells and never inside enclosed black holes or on the separator row.
    - If the reflected mask matches no top sprite, erase that bottom sprite to background.
    - Leave the entire top half and the separator row unchanged.

    Key insight:
    - Mirror matching is both the keep/delete test and the source of the halo color, so the reflection step must be solved before the lower-half edit is known.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    H, W = g.shape

    bg = _get_bg(g)
    separator_rows = [r for r in range(H) if int(g[r, 0]) != bg and np.all(g[r] == g[r, 0])]
    assert len(separator_rows) == 1, "expected exactly one solid separator row"
    sep = separator_rows[0]
    assert H == 2 * sep + 1, "separator must split equal-height halves"

    top = _extract_components(g, 0, sep, {bg, int(g[sep, 0])})
    bottom = _extract_components(g, sep + 1, H, {bg, int(g[sep, 0])})
    assert top and bottom

    top_by_coords = {comp["coord_key"]: comp for comp in top}
    out = g.copy()

    for comp in bottom:
        mirrored = frozenset(_mirror_across_separator(comp["coords"], sep))
        match = top_by_coords.get(mirrored)
        if match is None:
            for r, c in comp["coords"]:
                out[r, c] = bg
            continue

        halo_color = int(match["color"])
        for r, c in _halo_cells(comp["coords"], g.shape, sep):
            if int(g[r, c]) == bg:
                out[r, c] = halo_color

    return out


def _make_ring_shape() -> list[Coord]:
    return [(r, c) for r in range(3) for c in range(3) if (r, c) != (1, 1)]


def _make_polyomino() -> list[Coord] | None:
    for _ in range(GEN_SHAPE_ATTEMPTS):
        area = random.randint(3, 7)
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = set()
            for r, c in cells:
                for nr, nc in _neighbors4(r, c):
                    if -3 <= nr <= 3 and -3 <= nc <= 3 and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(tuple(frontier)))

        if len(cells) != area:
            continue

        shape = _normalize(cells)
        r0, c0, r1, c1 = _bbox(shape)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h > 4 or w > 4:
            continue
        if h * w == area and area >= 5 and random.random() < 0.7:
            continue
        return shape
    return None


def _make_shape(prefer_hole: bool) -> list[Coord] | None:
    if prefer_hole:
        return _make_ring_shape()
    return _make_polyomino()


def _can_place(coords: list[Coord], reserved: np.ndarray, row_lo: int, row_hi: int, W: int) -> bool:
    for r, c in coords:
        if not (row_lo <= r < row_hi and 0 <= c < W):
            return False
        if reserved[r, c]:
            return False
    return True


def _reserve(coords: list[Coord], reserved: np.ndarray) -> None:
    H, W = reserved.shape
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < H and 0 <= nc < W:
                    reserved[nr, nc] = True


def _place_random(shape: list[Coord], row_lo: int, row_hi: int, W: int, reserved: np.ndarray) -> list[Coord] | None:
    _, _, max_r, max_c = _bbox(shape)
    h = max_r + 1
    w = max_c + 1
    row_choices = list(range(row_lo, row_hi - h + 1))
    col_choices = list(range(0, W - w + 1))
    if not row_choices or not col_choices:
        return None

    for _ in range(GEN_PLACEMENT_ATTEMPTS):
        r0 = random.choice(row_choices)
        c0 = random.choice(col_choices)
        coords = _translate(shape, r0, c0)
        if _can_place(coords, reserved, row_lo, row_hi, W):
            return coords
    return None


def _build_expected(inp: np.ndarray, sep: int, top_objs: list[dict], bottom_objs: list[dict]) -> np.ndarray:
    out = inp.copy()
    top_lookup = {frozenset(obj["coords"]): obj for obj in top_objs}

    for obj in bottom_objs:
        mirrored = frozenset(_mirror_across_separator(obj["coords"], sep))
        match = top_lookup.get(mirrored)
        if match is None:
            for r, c in obj["coords"]:
                out[r, c] = BG
            continue

        halo = _halo_cells(obj["coords"], inp.shape, sep)
        for r, c in halo:
            if int(inp[r, c]) == BG:
                out[r, c] = int(match["color"])
    return out


def _validate_semantics(inp: np.ndarray, expected: np.ndarray, sep: int, top_objs: list[dict], bottom_objs: list[dict]) -> bool:
    try:
        solved = main(inp)
    except Exception:
        return False
    if not np.array_equal(solved, expected):
        return False

    top_lookup = {frozenset(obj["coords"]): obj for obj in top_objs}
    matched = 0
    unmatched = 0
    for obj in bottom_objs:
        mirrored = frozenset(_mirror_across_separator(obj["coords"], sep))
        match = top_lookup.get(mirrored)
        if match is None:
            unmatched += 1
            if any(int(expected[r, c]) != BG for r, c in obj["coords"]):
                return False
        else:
            matched += 1
            holes = _hole_cells(obj["coords"])
            for r, c in holes:
                if int(expected[r, c]) != BG:
                    return False
            halo = _halo_cells(obj["coords"], inp.shape, sep)
            for r, c in halo:
                if (r, c) in holes:
                    return False
                if int(inp[r, c]) == BG and int(expected[r, c]) not in (BG, int(match["color"])):
                    return False

    if matched < 1 or unmatched < 1:
        return False
    return True


def _generate_case() -> np.ndarray | None:
    H = random.choice(GEN_HEIGHTS)
    W = random.randint(*GEN_WIDTH_RANGE)
    sep = H // 2

    n_top = random.randint(*GEN_TOP_COUNT_RANGE)
    n_bottom = random.randint(*GEN_BOTTOM_COUNT_RANGE)
    max_match = min(n_top, n_bottom - 1)
    if max_match < 1:
        return None
    n_match = random.randint(1, max_match)

    top_colors = random.sample(TOP_PALETTE, n_top)
    bottom_colors = random.sample(BOTTOM_PALETTE, n_bottom)

    matched_top_colors = top_colors[:n_match]
    extra_top_colors = top_colors[n_match:]
    matched_bottom_colors = bottom_colors[:n_match]
    unmatched_bottom_colors = bottom_colors[n_match:]
    random.shuffle(matched_top_colors)
    random.shuffle(extra_top_colors)
    random.shuffle(matched_bottom_colors)
    random.shuffle(unmatched_bottom_colors)

    top_reserved = np.zeros((H, W), dtype=bool)
    bottom_reserved = np.zeros((H, W), dtype=bool)
    top_objs: list[dict] = []
    bottom_objs: list[dict] = []

    hole_pair_remaining = bool(matched_top_colors) and random.random() < GEN_MATCHED_HOLE_PROB

    for top_color, bottom_color in zip(matched_top_colors, matched_bottom_colors):
        placed = False
        for _ in range(GEN_SHAPE_ATTEMPTS):
            prefer_hole = hole_pair_remaining
            shape = _make_shape(prefer_hole)
            if shape is None:
                continue

            for _ in range(GEN_PLACEMENT_ATTEMPTS):
                top_coords = _place_random(shape, 0, sep, W, top_reserved)
                if top_coords is None:
                    break
                bottom_coords = _mirror_across_separator(top_coords, sep)
                if not _can_place(bottom_coords, bottom_reserved, sep + 1, H, W):
                    continue

                _reserve(top_coords, top_reserved)
                _reserve(bottom_coords, bottom_reserved)
                top_objs.append({"coords": top_coords, "color": top_color})
                bottom_objs.append({"coords": bottom_coords, "color": bottom_color})
                placed = True
                hole_pair_remaining = False
                break
            if placed:
                break
        if not placed:
            return None

    for top_color in extra_top_colors:
        placed = False
        for _ in range(GEN_SHAPE_ATTEMPTS):
            shape = _make_shape(random.random() < GEN_OTHER_HOLE_PROB)
            if shape is None:
                continue
            coords = _place_random(shape, 0, sep, W, top_reserved)
            if coords is None:
                continue
            _reserve(coords, top_reserved)
            top_objs.append({"coords": coords, "color": top_color})
            placed = True
            break
        if not placed:
            return None

    top_lookup = {frozenset(obj["coords"]): obj for obj in top_objs}

    for bottom_color in unmatched_bottom_colors:
        placed = False
        for _ in range(GEN_SHAPE_ATTEMPTS):
            shape = _make_shape(random.random() < GEN_OTHER_HOLE_PROB)
            if shape is None:
                continue
            coords = _place_random(shape, sep + 1, H, W, bottom_reserved)
            if coords is None:
                continue
            mirrored = frozenset(_mirror_across_separator(coords, sep))
            if mirrored in top_lookup:
                continue
            _reserve(coords, bottom_reserved)
            bottom_objs.append({"coords": coords, "color": bottom_color})
            placed = True
            break
        if not placed:
            return None

    inp = np.zeros((H, W), dtype=int)
    inp[sep, :] = SEP_COLOR
    for obj in top_objs:
        for r, c in obj["coords"]:
            inp[r, c] = int(obj["color"])
    for obj in bottom_objs:
        for r, c in obj["coords"]:
            inp[r, c] = int(obj["color"])

    expected = _build_expected(inp, sep, top_objs, bottom_objs)
    if not _validate_semantics(inp, expected, sep, top_objs, bottom_objs):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid mirror-match dilation input")
