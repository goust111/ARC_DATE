from __future__ import annotations

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5
INPUT_COLORS = np.array([1, 2, 3, 6, 7, 8, 9], dtype=int)
GEN_RETRY_LIMIT = 400

_RNG = np.random.default_rng()
_SEEN_INPUTS: set[tuple[tuple[int, int], bytes]] = set()
_SEEN_OUTPUTS: set[tuple[tuple[int, int], bytes]] = set()


def _is_isolated(mask: np.ndarray, r: int, c: int) -> bool:
    h, w = mask.shape
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
            return False
    return True


def _extract_pairs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg_mask = g != BLACK
    fg_coords = np.argwhere(fg_mask)
    assert fg_coords.shape[0] == 6, "expected exactly six non-black pixels"
    for r, c in fg_coords:
        assert _is_isolated(fg_mask, int(r), int(c)), "foreground pixels must be isolated singletons"

    colors = [int(color) for color in np.unique(g) if int(color) != BLACK]
    assert len(colors) == 3, "expected exactly three foreground colors"

    pairs = []
    for color in colors:
        coords = np.argwhere(g == color)
        assert coords.shape == (2, 2), "each color must appear exactly twice"
        (r0, c0), (r1, c1) = [tuple(map(int, rc)) for rc in coords]
        same_row = r0 == r1
        same_col = c0 == c1
        assert same_row ^ same_col, "same-color pixels must share exactly one axis"
        if same_row:
            dist = abs(c1 - c0)
            midpoint = (r0, (c0 + c1) // 2)
            orientation = "H"
        else:
            dist = abs(r1 - r0)
            midpoint = ((r0 + r1) // 2, c0)
            orientation = "V"
        assert dist >= 2 and dist % 2 == 0, "each pair must meet at a unique grid midpoint"
        pairs.append(
            {
                "color": color,
                "coords": ((r0, c0), (r1, c1)),
                "midpoint": midpoint,
                "orientation": orientation,
            }
        )
    return pairs


def _infer_rectangle(midpoints: list[tuple[int, int]]) -> tuple[int, int, int, int, tuple[int, int]]:
    points = {tuple(map(int, pt)) for pt in midpoints}
    assert len(points) == 3, "collision points must be distinct"

    rows = sorted({r for r, _ in points})
    cols = sorted({c for _, c in points})
    assert len(rows) == 2 and len(cols) == 2, "collision points must define a rectangle by rows and columns"

    all_corners = {(rows[0], cols[0]), (rows[0], cols[1]), (rows[1], cols[0]), (rows[1], cols[1])}
    assert points <= all_corners, "collision points must occupy rectangle corners"
    missing = next(iter(all_corners - points))
    return rows[0], cols[0], rows[1], cols[1], missing


def _build_output(shape: tuple[int, int], pairs: list[dict]) -> np.ndarray:
    point_to_color = {tuple(pair["midpoint"]): int(pair["color"]) for pair in pairs}
    assert len(point_to_color) == 3, "midpoints must be unique"
    r0, c0, r1, c1, missing = _infer_rectangle(list(point_to_color))

    out = np.zeros(shape, dtype=int)
    out[r0 : r1 + 1, c0 : c1 + 1] = YELLOW
    for point, color in point_to_color.items():
        out[point] = color
    out[missing] = GREY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the missing fourth rectangle corner from three pairwise collision points and paint the hidden rectangle.

    Input structure:
    - A black grid contains exactly three colors, each appearing as one isolated pair of single pixels.
    - Within each color, the two pixels lie on one shared row or one shared column and are separated so their midpoint is a single cell.
    - The three pair midpoints land on three corners of one axis-aligned rectangle.

    Rule:
    - Group the pixels into same-color pairs and compute each pair's collision midpoint.
    - Erase the input pixels entirely.
    - Use the three midpoint corners to infer the rectangle's missing fourth corner.
    - Fill the whole rectangle yellow, repaint the three observed corners with their pair colors, and paint the inferred fourth corner grey.

    Key insight:
    - Nothing in the input draws the rectangle itself; the only usable geometry is the three hidden collision corners induced by midpoint computation, and the fourth corner must be inferred from their row-column pattern.
    """
    return _build_output(np.asarray(grid, dtype=int).shape, _extract_pairs(grid))


def _sample_margins() -> tuple[int, int, int, int]:
    return tuple(int(_RNG.integers(2, 8)) for _ in range(4))


def _pair_endpoints(corner: tuple[int, int], orientation: str, distance: int) -> tuple[tuple[int, int], tuple[int, int]]:
    r, c = corner
    if orientation == "H":
        return (r, c - distance), (r, c + distance)
    return (r - distance, c), (r + distance, c)


def _inside_rect(rc: tuple[int, int], rect: tuple[int, int, int, int]) -> bool:
    r, c = rc
    r0, c0, r1, c1 = rect
    return r0 <= r <= r1 and c0 <= c <= c1


def _validate_case(grid: np.ndarray) -> bool:
    try:
        pairs = _extract_pairs(grid)
        orientations = {pair["orientation"] for pair in pairs}
        assert orientations == {"H", "V"}, "generator must cover both horizontal and vertical pair cases in each sample"

        point_to_color = {tuple(pair["midpoint"]): int(pair["color"]) for pair in pairs}
        r0, c0, r1, c1, missing = _infer_rectangle(list(point_to_color))
        rect = (r0, c0, r1, c1)

        all_endpoints = [rc for pair in pairs for rc in pair["coords"]]
        assert len(set(all_endpoints)) == 6, "input endpoints must stay distinct"
        rows = [r for r, _ in all_endpoints]
        cols = [c for _, c in all_endpoints]
        assert (min(rows), min(cols), max(rows), max(cols)) != rect, "input bbox must not equal the hidden rectangle"

        for pair in pairs:
            inside_count = sum(_inside_rect(rc, rect) for rc in pair["coords"])
            assert inside_count == 1, "each pair should have one interior and one exterior endpoint"

        out = main(grid)
        expected = _build_output(grid.shape, pairs)
        assert np.array_equal(out, expected), "solver output must match the geometry implied by the input"
        assert out[missing] == GREY
        assert np.count_nonzero(out == GREY) == 1
        assert np.all(out[r0 : r1 + 1, c0 : c1 + 1] != BLACK)
        return True
    except AssertionError:
        return False


def _choose_orientation_pattern() -> tuple[str, str, str]:
    patterns = (
        ("H", "H", "V"),
        ("H", "V", "H"),
        ("V", "H", "H"),
        ("V", "V", "H"),
        ("V", "H", "V"),
        ("H", "V", "V"),
    )
    return patterns[int(_RNG.integers(len(patterns)))]


def _make_candidate() -> np.ndarray | None:
    side_h = int(_RNG.integers(4, 10))
    side_w = int(_RNG.integers(4, 10))
    top, bottom, left, right = _sample_margins()

    h = top + side_h + 1 + bottom
    w = left + side_w + 1 + right
    r0, c0 = top, left
    r1, c1 = r0 + side_h, c0 + side_w

    all_corners = ((r0, c0), (r0, c1), (r1, c0), (r1, c1))
    missing_i = int(_RNG.integers(4))
    visible_corners = [corner for i, corner in enumerate(all_corners) if i != missing_i]
    orientations = _choose_orientation_pattern()
    colors = [int(c) for c in _RNG.choice(INPUT_COLORS, size=3, replace=False)]

    grid = np.zeros((h, w), dtype=int)
    rect = (r0, c0, r1, c1)

    used: set[tuple[int, int]] = set()
    for corner, orientation, color in zip(visible_corners, orientations, colors):
        r, c = corner
        if orientation == "H":
            outward_room = left if c == c0 else right
            max_d = min(outward_room, side_w - 1)
        else:
            outward_room = top if r == r0 else bottom
            max_d = min(outward_room, side_h - 1)
        if max_d < 1:
            return None

        distances = list(range(1, max_d + 1))
        _RNG.shuffle(distances)
        placed = False
        for distance in distances:
            a, b = _pair_endpoints(corner, orientation, distance)
            if a in used or b in used:
                continue
            if abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1:
                continue
            if any(abs(a[0] - ur) + abs(a[1] - uc) == 1 for ur, uc in used):
                continue
            if any(abs(b[0] - ur) + abs(b[1] - uc) == 1 for ur, uc in used):
                continue
            if sum(_inside_rect(rc, rect) for rc in (a, b)) != 1:
                continue
            used.add(a)
            used.add(b)
            grid[a] = color
            grid[b] = color
            placed = True
            break
        if not placed:
            return None

    if not _validate_case(grid):
        return None

    input_sig = (grid.shape, grid.tobytes())
    output = main(grid)
    output_sig = (output.shape, output.tobytes())
    if input_sig in _SEEN_INPUTS or output_sig in _SEEN_OUTPUTS:
        return None
    _SEEN_INPUTS.add(input_sig)
    _SEEN_OUTPUTS.add(output_sig)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _make_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 3")
