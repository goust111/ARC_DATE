from __future__ import annotations

from collections import defaultdict

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
RNG = np.random.default_rng()
BG = 0
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 250


def _normalize(coords) -> tuple[tuple[int, int], ...]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0 = min(rows)
    c0 = min(cols)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))


def _bbox_shape(coords) -> tuple[int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return max(rows) - min(rows) + 1, max(cols) - min(cols) + 1


def _is_connected(coords) -> bool:
    cells = set(coords)
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                stack.append(nxt)
    return len(seen) == len(cells)


def _shape_type(coords) -> str | None:
    cells = set(_normalize(coords))
    rows = sorted({r for r, _ in cells})
    cols = sorted({c for _, c in cells})

    if len(rows) == 1:
        bar_cols = sorted(c for _, c in cells)
        if bar_cols == list(range(bar_cols[0], bar_cols[-1] + 1)):
            return "bar"
    if len(cols) == 1:
        bar_rows = sorted(r for r, _ in cells)
        if bar_rows == list(range(bar_rows[0], bar_rows[-1] + 1)):
            return "bar"

    h, w = _bbox_shape(cells)
    candidates = (
        {(0, cc) for cc in range(w)} | {(rr, 0) for rr in range(h)},
        {(0, cc) for cc in range(w)} | {(rr, w - 1) for rr in range(h)},
        {(h - 1, cc) for cc in range(w)} | {(rr, 0) for rr in range(h)},
        {(h - 1, cc) for cc in range(w)} | {(rr, w - 1) for rr in range(h)},
    )
    return "L" if any(cells == candidate for candidate in candidates) else None


def _rotate180(coords) -> tuple[tuple[int, int], ...]:
    norm = _normalize(coords)
    h, w = _bbox_shape(norm)
    rotated = [(h - 1 - r, w - 1 - c) for r, c in norm]
    return _normalize(rotated)


def _candidate_rectangle_dims(piece_a, piece_b) -> set[tuple[int, int]]:
    a = set(_normalize(piece_a))
    b = set(_normalize(piece_b))
    dims: set[tuple[int, int]] = set()
    for dr in range(-8, 9):
        for dc in range(-8, 9):
            shifted = {(r + dr, c + dc) for r, c in b}
            if a & shifted:
                continue
            union = a | shifted
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            h = max(rows) - min(rows) + 1
            w = max(cols) - min(cols) + 1
            if len(union) == h * w:
                dims.add((int(h), int(w)))
    return dims


def _reconstruct_rectangle(piece_a, piece_b) -> tuple[int, int]:
    dims = _candidate_rectangle_dims(piece_a, piece_b)
    assert len(dims) == 1, "each color pair must reconstruct one rectangle size"
    return next(iter(dims))


def _color_components(grid: np.ndarray) -> list[dict[str, object]]:
    components: list[dict[str, object]] = []
    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BG):
        mask = grid == int(color)
        visited = np.zeros_like(mask, dtype=bool)
        seeds = np.argwhere(mask)
        for r0, c0 in seeds:
            rr0 = int(r0)
            cc0 = int(c0)
            if visited[rr0, cc0]:
                continue
            stack = [(rr0, cc0)]
            visited[rr0, cc0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in NEIGHBORS4:
                    nr = r + dr
                    nc = c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append({"color": int(color), "coords": tuple(coords)})
    return components


def _build_output(rectangles) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: int(item[0]))
    out_h = sum(int(h) for _, h, _, _ in ordered)
    out_w = max(int(w) for _, _, w, _ in ordered)
    out = np.zeros((out_h, out_w), dtype=int)
    row = out_h
    for _, h, w, color in ordered:
        h = int(h)
        w = int(w)
        row -= h
        col = (out_w - w) // 2
        out[row : row + h, col : col + w] = int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-colored scattered fragments, rebuild their solid rectangles, and stack those rectangles by area.

    Input structure:
    - The background is black.
    - The foreground consists of monochromatic 4-connected objects of several colors.
    - Each non-black color appears in exactly two separate objects.
    - Within each color pair, the two objects are 180-degree rotations of one another and can be translated to tile one solid rectangle.

    Rule:
    - Split the input into monochromatic 4-connected components and group them by color.
    - For each color pair, translate one piece relative to the other without rotating either one until their union becomes a filled rectangle; this determines the rectangle's width and height.
    - Replace the pair by that solid rectangle of the same color.
    - Sort the reconstructed rectangles by increasing area and stack them from bottom to top with no vertical gaps, centering each rectangle horizontally on the output canvas.

    Key insight:
    - The crucial step is to recognize that equal-color pieces are complementary 180-degree partners whose original positions are irrelevant; once each hidden rectangle is recovered, a second rule reorders the rectangles by area into a bottom stack.
    """
    g = np.asarray(grid, dtype=int)
    components = _color_components(g)
    by_color: dict[int, list[tuple[tuple[int, int], ...]]] = defaultdict(list)
    for comp in components:
        by_color[int(comp["color"])].append(_normalize(comp["coords"]))

    rectangles = []
    for color, pieces in by_color.items():
        assert len(pieces) == 2, "each color must appear in exactly two objects"
        rect_h, rect_w = _reconstruct_rectangle(pieces[0], pieces[1])
        rectangles.append((rect_h * rect_w, rect_h, rect_w, color))

    assert rectangles, "expected at least one color pair"
    return _build_output(rectangles)


def _rotation_pairs(rect_h: int, rect_w: int) -> list[tuple[tuple[int, int], tuple[int, int]]] | None:
    pairs = []
    seen = set()
    for r in range(rect_h):
        for c in range(rect_w):
            other = (rect_h - 1 - r, rect_w - 1 - c)
            here = (r, c)
            if other == here:
                return None
            key = tuple(sorted((here, other)))
            if key not in seen:
                seen.add(key)
                pairs.append((here, other))
    return pairs


def _build_catalog() -> list[dict[str, object]]:
    catalog: list[dict[str, object]] = []
    seen = set()
    for rect_h in range(1, 7):
        for rect_w in range(1, 7):
            area = rect_h * rect_w
            if not (4 <= area <= 12 and area % 2 == 0):
                continue
            cell_pairs = _rotation_pairs(rect_h, rect_w)
            if cell_pairs is None:
                continue
            for bits in range(1 << len(cell_pairs)):
                piece_a = []
                piece_b = []
                for idx, (cell0, cell1) in enumerate(cell_pairs):
                    if (bits >> idx) & 1:
                        piece_a.append(cell0)
                        piece_b.append(cell1)
                    else:
                        piece_a.append(cell1)
                        piece_b.append(cell0)
                if not (_is_connected(piece_a) and _is_connected(piece_b)):
                    continue
                norm_a = _normalize(piece_a)
                norm_b = _normalize(piece_b)
                shape_a = _shape_type(norm_a)
                shape_b = _shape_type(norm_b)
                if shape_a not in {"L", "bar"} or shape_b not in {"L", "bar"}:
                    continue
                if _rotate180(norm_a) != norm_b:
                    continue
                if _candidate_rectangle_dims(norm_a, norm_b) != {(rect_h, rect_w)}:
                    continue
                key = (area, tuple(sorted((norm_a, norm_b))))
                if key in seen:
                    continue
                seen.add(key)
                catalog.append(
                    {
                        "area": int(area),
                        "rect": (int(rect_h), int(rect_w)),
                        "pieces": (norm_a, norm_b),
                        "shape": shape_a,
                    }
                )
    return catalog


CATALOG = _build_catalog()
CATALOG_BY_AREA: dict[int, list[dict[str, object]]] = defaultdict(list)
for entry in CATALOG:
    CATALOG_BY_AREA[int(entry["area"])].append(entry)

AREA_OPTIONS = sorted(CATALOG_BY_AREA)
assert AREA_OPTIONS == [8, 10, 12]


def _mark_blocked(blocked: np.ndarray, coords) -> None:
    h, w = blocked.shape
    for r, c in coords:
        for rr in range(max(0, r - 1), min(h, r + 2)):
            for cc in range(max(0, c - 1), min(w, c + 2)):
                blocked[rr, cc] = True


def _place_piece(grid: np.ndarray, blocked: np.ndarray, piece, color: int) -> bool:
    piece = tuple(piece)
    piece_h, piece_w = _bbox_shape(piece)
    max_r = grid.shape[0] - piece_h
    max_c = grid.shape[1] - piece_w
    if max_r < 0 or max_c < 0:
        return False

    for _ in range(PLACEMENT_ATTEMPTS):
        r0 = int(RNG.integers(0, max_r + 1))
        c0 = int(RNG.integers(0, max_c + 1))
        placed = [(r0 + r, c0 + c) for r, c in piece]
        if any(blocked[r, c] for r, c in placed):
            continue
        for r, c in placed:
            grid[r, c] = int(color)
        _mark_blocked(blocked, placed)
        return True
    return False


def _generate_attempt() -> np.ndarray | None:
    pair_count = int(RNG.integers(2, min(4, len(AREA_OPTIONS) + 1)))
    areas = list(RNG.choice(AREA_OPTIONS, size=pair_count, replace=False))
    colors = list(RNG.choice(np.arange(1, 10), size=pair_count, replace=False))

    selected = []
    rectangles = []
    max_piece_h = 0
    max_piece_w = 0
    for area, color in zip(areas, colors):
        choices = CATALOG_BY_AREA[int(area)]
        entry = choices[int(RNG.integers(0, len(choices)))]
        pieces = list(entry["pieces"])
        if int(RNG.integers(0, 2)) == 1:
            pieces.reverse()
        selected.append((int(color), tuple(pieces)))
        rect_h, rect_w = entry["rect"]
        rectangles.append((int(area), int(rect_h), int(rect_w), int(color)))
        for piece in pieces:
            piece_h, piece_w = _bbox_shape(piece)
            max_piece_h = max(max_piece_h, int(piece_h))
            max_piece_w = max(max_piece_w, int(piece_w))

    min_h = max(14, max_piece_h + 4, 8 + 3 * pair_count)
    min_w = max(14, max_piece_w + 4, 8 + 3 * pair_count)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = int(RNG.integers(min_h, 31))
    grid_w = int(RNG.integers(min_w, 31))
    grid = np.zeros((grid_h, grid_w), dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    pieces_to_place = []
    for color, pieces in selected:
        for piece in pieces:
            pieces_to_place.append((color, piece))
    RNG.shuffle(pieces_to_place)

    for color, piece in pieces_to_place:
        if not _place_piece(grid, blocked, piece, color):
            return None

    expected = _build_output(rectangles)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
