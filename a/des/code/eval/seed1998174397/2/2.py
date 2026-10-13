import numpy as np


BLACK = 0
ORANGE = 7
LINE_COLORS = (1, 2, 3, 4, 5, 6, 8, 9)

GEN_ATTEMPTS = 500
PLACE_ATTEMPTS = 120
GRID_H_RANGE = (14, 22)
GRID_W_RANGE = (14, 22)
PAIR_COUNT_CHOICES = (2, 3, 4)
RULER_LENGTH_CHOICES = (2, 3, 4)
PAIR_ORIENTATIONS = ("horizontal", "vertical")
RULER_CORNERS = ("tl", "tr", "bl", "br")


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp: list[tuple[int, int]] = []
        while stack:
            cr, cc = stack.pop()
            comp.append((cr, cc))
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(sorted(comp))

    comps.sort(key=lambda comp: _bbox(comp))
    return comps


def _segment_orientation(comp: list[tuple[int, int]]) -> str | None:
    r0, c0, r1, c1 = _bbox(comp)
    pts = set(comp)
    if r0 == r1:
        expected = {(r0, c) for c in range(c0, c1 + 1)}
        if pts == expected and 2 <= len(comp) <= 5:
            return "horizontal"
    if c0 == c1:
        expected = {(r, c0) for r in range(r0, r1 + 1)}
        if pts == expected and 2 <= len(comp) <= 5:
            return "vertical"
    return None


def _is_corner_ruler(comp: list[tuple[int, int]], h: int, w: int) -> bool:
    r0, c0, r1, c1 = _bbox(comp)
    if r0 != r1:
        return False
    if not (2 <= len(comp) <= 4):
        return False
    expected = {(r0, c) for c in range(c0, c1 + 1)}
    if set(comp) != expected:
        return False
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _extract_scene(
    grid: np.ndarray,
) -> tuple[int, int, list[tuple[int, list[list[tuple[int, int]]], str]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _background_color(g)
    colors = [int(color) for color in np.unique(g) if int(color) != bg]

    comps_by_color = {color: _components_of_color(g, color) for color in colors}

    ruler_candidates: list[tuple[int, list[tuple[int, int]]]] = []
    for color, comps in comps_by_color.items():
        if len(comps) == 1 and _is_corner_ruler(comps[0], h, w):
            ruler_candidates.append((color, comps[0]))

    assert len(ruler_candidates) == 1, "expected exactly one corner ruler"
    ruler_color, ruler_comp = ruler_candidates[0]
    _, c0, _, c1 = _bbox(ruler_comp)
    stride = c1 - c0 + 1

    pairs: list[tuple[int, list[list[tuple[int, int]]], str]] = []
    for color, comps in comps_by_color.items():
        if color == ruler_color:
            continue
        assert len(comps) == 2, "each line color must form exactly two segments"
        orientations = [_segment_orientation(comp) for comp in comps]
        assert orientations[0] is not None and orientations[0] == orientations[1], (
            "same-color segments must be straight and parallel"
        )
        pairs.append((color, comps, orientations[0]))

    pairs.sort(key=lambda item: item[0])
    return bg, stride, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair same-color line segments and fill the rectangle between them with periodic stripes.

    The unique corner ruler gives the stripe period K.
    Horizontal segment pairs produce vertical stripes from the left edge.
    Vertical segment pairs produce horizontal stripes from the top edge.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    _, stride, pairs = _extract_scene(g)

    for color, comps, orientation in pairs:
        pts = comps[0] + comps[1]
        r0, c0, r1, c1 = _bbox(pts)
        if orientation == "horizontal":
            for c in range(c0, c1 + 1, stride):
                out[r0 : r1 + 1, c] = color
        else:
            for r in range(r0, r1 + 1, stride):
                out[r, c0 : c1 + 1] = color

    return out


def _reserve(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> None:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    mask[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _free(mask: np.ndarray, r0: int, c0: int, r1: int, c1: int, pad: int = 1) -> bool:
    h, w = mask.shape
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(h - 1, r1 + pad)
    cc1 = min(w - 1, c1 + pad)
    return not bool(mask[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _ruler_cells(h: int, w: int, corner: str, length: int) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(0, c) for c in range(length)]
    if corner == "tr":
        return [(0, w - length + c) for c in range(length)]
    if corner == "bl":
        return [(h - 1, c) for c in range(length)]
    return [(h - 1, w - length + c) for c in range(length)]


def _orientation_schedule(count: int, rng: np.random.Generator) -> list[str]:
    if count == 2:
        picks = ["horizontal", "vertical"]
    elif count == 3:
        picks = ["horizontal", "vertical", str(rng.choice(PAIR_ORIENTATIONS))]
    else:
        picks = ["horizontal", "horizontal", "vertical", "vertical"]
    rng.shuffle(picks)
    return picks


def _place_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    orientation: str,
    rng: np.random.Generator,
) -> bool:
    h, w = grid.shape
    for _ in range(PLACE_ATTEMPTS):
        line_len = int(rng.integers(2, 6))
        if orientation == "horizontal":
            box_h = int(rng.integers(3, 9))
            if box_h > h - 2 or line_len > w - 2:
                continue
            r0 = int(rng.integers(1, h - box_h))
            c0 = int(rng.integers(1, w - line_len))
            r1 = r0 + box_h - 1
            c1 = c0 + line_len - 1
            if not _free(reserved, r0, c0, r1, c1, pad=1):
                continue
            grid[r0, c0 : c1 + 1] = color
            grid[r1, c0 : c1 + 1] = color
            _reserve(reserved, r0, c0, r1, c1, pad=1)
            return True

        box_w = int(rng.integers(3, 9))
        if line_len > h - 2 or box_w > w - 2:
            continue
        r0 = int(rng.integers(1, h - line_len))
        c0 = int(rng.integers(1, w - box_w))
        r1 = r0 + line_len - 1
        c1 = c0 + box_w - 1
        if not _free(reserved, r0, c0, r1, c1, pad=1):
            continue
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        _reserve(reserved, r0, c0, r1, c1, pad=1)
        return True

    return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    w = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    grid = np.full((h, w), BLACK, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    stride = int(rng.choice(RULER_LENGTH_CHOICES))
    corner = str(rng.choice(RULER_CORNERS))
    for r, c in _ruler_cells(h, w, corner, stride):
        grid[r, c] = ORANGE
    ruler_bbox = _bbox(_ruler_cells(h, w, corner, stride))
    _reserve(reserved, *ruler_bbox, pad=1)

    pair_count = int(rng.choice(PAIR_COUNT_CHOICES))
    orientations = _orientation_schedule(pair_count, rng)
    colors = [int(v) for v in rng.choice(LINE_COLORS, size=pair_count, replace=False)]

    for color, orientation in zip(colors, orientations):
        if not _place_pair(grid, reserved, color, orientation, rng):
            return None

    try:
        bg, parsed_stride, pairs = _extract_scene(grid)
    except AssertionError:
        return None

    if bg != BLACK or parsed_stride != stride or len(pairs) != pair_count:
        return None
    if np.array_equal(main(grid), grid):
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {GEN_ATTEMPTS} attempts")
