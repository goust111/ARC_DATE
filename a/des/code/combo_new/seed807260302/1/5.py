import random

import numpy as np


BLACK = 0
GREY = 5
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]


def _sign(x: int) -> int:
    return (int(x) > 0) - (int(x) < 0)


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _line_step_and_length(a: tuple[int, int], b: tuple[int, int]) -> tuple[tuple[int, int], int]:
    dr = int(b[0] - a[0])
    dc = int(b[1] - a[1])
    assert dr != 0 or dc != 0, "paired marker pixels must be distinct"

    if dr == 0:
        return (0, _sign(dc)), abs(dc)
    if dc == 0:
        return (_sign(dr), 0), abs(dr)

    assert abs(dr) == abs(dc), "marker pair must be horizontal, vertical, or diagonal"
    return (_sign(dr), _sign(dc)), abs(dr)


def _mirror_info(g: np.ndarray) -> tuple[str, int, set[tuple[int, int]]]:
    grey = np.argwhere(g == GREY)
    assert grey.size > 0, "expected a grey mirror segment"
    rows = np.unique(grey[:, 0])
    cols = np.unique(grey[:, 1])

    if len(rows) == 1 and len(cols) > 1:
        cs = np.sort(cols)
        assert np.array_equal(cs, np.arange(int(cs[0]), int(cs[-1]) + 1))
        cells = {(int(rows[0]), int(c)) for c in cs}
        return "horizontal", int(rows[0]), cells

    if len(cols) == 1 and len(rows) > 1:
        rs = np.sort(rows)
        assert np.array_equal(rs, np.arange(int(rs[0]), int(rs[-1]) + 1))
        cells = {(int(r), int(cols[0])) for r in rs}
        return "vertical", int(cols[0]), cells

    raise AssertionError("grey mirror must be one horizontal or vertical segment")


def _marker_pairs(g: np.ndarray) -> list[tuple[int, tuple[int, int], tuple[int, int]]]:
    pairs: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    for color in np.unique(g):
        col = int(color)
        if col in (BLACK, GREY):
            continue
        pts = np.argwhere(g == col)
        assert pts.shape[0] == 2, "each marker color must appear exactly twice"
        p0 = (int(pts[0, 0]), int(pts[0, 1]))
        p1 = (int(pts[1, 0]), int(pts[1, 1]))
        pairs.append((col, p0, p1))
    assert pairs, "expected at least one colored marker pair"
    return pairs


def _directed_pair(
    orient: str,
    wall_line: int,
    p0: tuple[int, int],
    p1: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    if orient == "horizontal":
        s0 = _sign(p0[0] - wall_line)
        s1 = _sign(p1[0] - wall_line)
        d0 = abs(p0[0] - wall_line)
        d1 = abs(p1[0] - wall_line)
    else:
        s0 = _sign(p0[1] - wall_line)
        s1 = _sign(p1[1] - wall_line)
        d0 = abs(p0[1] - wall_line)
        d1 = abs(p1[1] - wall_line)

    assert s0 != 0 and s0 == s1, "paired markers must be on the same side of the mirror"
    assert d0 != d1, "one marker must be nearer to the mirror"
    return (p0, p1) if d0 > d1 else (p1, p0)


def _ray_from_pair(
    g: np.ndarray,
    orient: str,
    wall_line: int,
    far: tuple[int, int],
    near: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int], int]:
    (dr, dc), length = _line_step_and_length(far, near)
    r, c = near
    seen = 0

    while True:
        r += dr
        c += dc
        seen += 1
        assert seen <= g.shape[0] + g.shape[1], "incoming ray did not reach the mirror"
        assert _in_bounds(g.shape, r, c), "incoming ray left the grid before reaching the mirror"
        if int(g[r, c]) == GREY:
            reflection = (int(r), int(c))
            break

    if orient == "horizontal":
        out_dir = (-dr, dc)
    else:
        out_dir = (dr, -dc)
    assert out_dir != (0, 0)
    return reflection, out_dir, int(length)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored marker pair from a gray mirror and draw equal-length outgoing rays with cross endpoints.

    Input structure:
    - The background is black.
    - A single gray segment is horizontal or vertical and acts as a mirror wall.
    - Every other non-black color appears as exactly two marker pixels on the same side of the mirror.
    - Each marker pair lies on one horizontal, vertical, or diagonal direction whose extension reaches the mirror.

    Rule:
    - For each color, aim from the marker farther from the mirror toward the nearer marker.
    - Continue from the nearer marker along that direction until the first gray mirror pixel is reached.
    - Reflect the direction by flipping the row component for a horizontal mirror or the column component for a vertical mirror.
    - Draw from the reflection point for the same step length as the marker-pair spacing, then draw a 3x3 orthogonal cross at the ray endpoint.
    - Keep the original marker pixels and the gray mirror unchanged.

    Key insight:
    - The two same-color pixels encode both the incoming direction and length; the gray line only flips the perpendicular component of that direction.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    orient, wall_line, _wall_cells = _mirror_info(g)
    pairs = _marker_pairs(g)
    marker_cells = [(r, c, color) for color, p0, p1 in pairs for r, c in (p0, p1)]
    grey_mask = g == GREY

    out = g.copy()
    for color, p0, p1 in pairs:
        far, near = _directed_pair(orient, wall_line, p0, p1)
        (rr, rc), (dr, dc), length = _ray_from_pair(g, orient, wall_line, far, near)

        r, c = rr, rc
        for _ in range(length):
            r += dr
            c += dc
            assert _in_bounds((H, W), r, c), "reflected ray must remain inside the grid"
            if not grey_mask[r, c]:
                out[r, c] = int(color)

        for ar, ac in ((r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            assert _in_bounds((H, W), ar, ac), "endpoint cross must fit inside the grid"
            if not grey_mask[ar, ac]:
                out[ar, ac] = int(color)

    out[grey_mask] = GREY
    for r, c, color in marker_cells:
        out[int(r), int(c)] = int(color)
    return out


# --------------- generator ---------------

MIN_SIZE, MAX_SIZE = 14, 30
PAIR_COUNT_RANGE = (1, 4)
WALL_MIN_LEN = 5
WALL_MAX_LEN = 18
RAY_LEN_MIN, RAY_LEN_MAX = 2, 6
GENERATION_ATTEMPTS = 500
PAIR_PLACEMENT_ATTEMPTS = 500


def _cross_cells(end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = map(int, end)
    return [(r, c), (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _wall_span(axis_len: int) -> tuple[int, int]:
    max_len = min(WALL_MAX_LEN, int(axis_len) - 6)
    length = random.randint(WALL_MIN_LEN, max_len)
    start = random.randint(2, int(axis_len) - length - 3)
    return int(start), int(start + length - 1)


def _choose_wall(H: int, W: int) -> tuple[str, int, tuple[int, int], set[tuple[int, int]]]:
    orient = random.choice(["horizontal", "vertical"])
    if orient == "horizontal":
        row = random.randint(6, H - 7)
        c0, c1 = _wall_span(W)
        cells = {(row, c) for c in range(c0, c1 + 1)}
        return orient, int(row), (int(c0), int(c1)), cells

    col = random.randint(6, W - 7)
    r0, r1 = _wall_span(H)
    cells = {(r, col) for r in range(r0, r1 + 1)}
    return orient, int(col), (int(r0), int(r1)), cells


def _candidate_horizontal(
    H: int,
    W: int,
    wall_row: int,
    wall_span: tuple[int, int],
) -> dict | None:
    side = random.choice([-1, 1])
    side_space = wall_row - 1 if side < 0 else H - 2 - wall_row
    max_len = min(RAY_LEN_MAX, (side_space - 1) // 2)
    if max_len < RAY_LEN_MIN:
        return None

    length = random.randint(RAY_LEN_MIN, max_len)
    gap = random.randint(length + 1, side_space - length)
    dc = random.choice([-1, 0, 1])
    dr = -side
    ref_c = random.randint(int(wall_span[0]), int(wall_span[1]))
    ref = (int(wall_row), int(ref_c))
    near = (ref[0] - gap * dr, ref[1] - gap * dc)
    far = (ref[0] - (gap + length) * dr, ref[1] - (gap + length) * dc)
    out_dir = (side, dc)
    endpoint = (ref[0] + length * out_dir[0], ref[1] + length * out_dir[1])
    incoming = [(near[0] + s * dr, near[1] + s * dc) for s in range(1, gap)]
    ray = [(ref[0] + s * out_dir[0], ref[1] + s * out_dir[1]) for s in range(1, length + 1)]

    return {
        "far": far,
        "near": near,
        "ref": ref,
        "length": length,
        "incoming": incoming,
        "ray": ray,
        "cross": _cross_cells(endpoint),
    }


def _candidate_vertical(
    H: int,
    W: int,
    wall_col: int,
    wall_span: tuple[int, int],
) -> dict | None:
    side = random.choice([-1, 1])
    side_space = wall_col - 1 if side < 0 else W - 2 - wall_col
    max_len = min(RAY_LEN_MAX, (side_space - 1) // 2)
    if max_len < RAY_LEN_MIN:
        return None

    length = random.randint(RAY_LEN_MIN, max_len)
    gap = random.randint(length + 1, side_space - length)
    dr = random.choice([-1, 0, 1])
    dc = -side
    ref_r = random.randint(int(wall_span[0]), int(wall_span[1]))
    ref = (int(ref_r), int(wall_col))
    near = (ref[0] - gap * dr, ref[1] - gap * dc)
    far = (ref[0] - (gap + length) * dr, ref[1] - (gap + length) * dc)
    out_dir = (dr, side)
    endpoint = (ref[0] + length * out_dir[0], ref[1] + length * out_dir[1])
    incoming = [(near[0] + s * dr, near[1] + s * dc) for s in range(1, gap)]
    ray = [(ref[0] + s * out_dir[0], ref[1] + s * out_dir[1]) for s in range(1, length + 1)]

    return {
        "far": far,
        "near": near,
        "ref": ref,
        "length": length,
        "incoming": incoming,
        "ray": ray,
        "cross": _cross_cells(endpoint),
    }


def _valid_candidate(
    cand: dict,
    shape: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> bool:
    markers = {tuple(cand["far"]), tuple(cand["near"])}
    incoming = {tuple(p) for p in cand["incoming"]}
    draw = {tuple(p) for p in cand["ray"]} | {tuple(p) for p in cand["cross"]}
    all_cells = markers | incoming | draw | {tuple(cand["ref"])}

    if any(not _in_bounds(shape, r, c) for r, c in all_cells):
        return False
    if markers & reserved:
        return False
    if draw & reserved:
        return False
    if incoming & occupied_input:
        return False
    if incoming & reserved:
        return False
    if markers & draw:
        return False
    return True


def _sample_pair(
    H: int,
    W: int,
    orient: str,
    wall_line: int,
    wall_span: tuple[int, int],
    occupied_input: set[tuple[int, int]],
    reserved: set[tuple[int, int]],
) -> dict | None:
    for _ in range(PAIR_PLACEMENT_ATTEMPTS):
        if orient == "horizontal":
            cand = _candidate_horizontal(H, W, wall_line, wall_span)
        else:
            cand = _candidate_vertical(H, W, wall_line, wall_span)
        if cand is not None and _valid_candidate(cand, (H, W), occupied_input, reserved):
            return cand
    return None


def _render_expected(inp: np.ndarray, specs: list[dict]) -> np.ndarray:
    out = inp.copy()
    grey_mask = inp == GREY
    marker_cells: list[tuple[int, int, int]] = []

    for spec in specs:
        color = int(spec["color"])
        for r, c in spec["ray"]:
            if not grey_mask[int(r), int(c)]:
                out[int(r), int(c)] = color
        for r, c in spec["cross"]:
            if not grey_mask[int(r), int(c)]:
                out[int(r), int(c)] = color
        marker_cells.append((int(spec["far"][0]), int(spec["far"][1]), color))
        marker_cells.append((int(spec["near"][0]), int(spec["near"][1]), color))

    out[grey_mask] = GREY
    for r, c, color in marker_cells:
        out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(MIN_SIZE, MAX_SIZE)
    W = random.randint(MIN_SIZE, MAX_SIZE)
    orient, wall_line, span, wall_cells = _choose_wall(H, W)
    pair_count = random.randint(*PAIR_COUNT_RANGE)
    colors = random.sample(MARKER_COLORS, pair_count)

    grid = np.full((H, W), BLACK, dtype=int)
    for r, c in wall_cells:
        grid[int(r), int(c)] = GREY

    occupied_input = set(wall_cells)
    reserved = set(wall_cells)
    specs: list[dict] = []

    for color in colors:
        cand = _sample_pair(H, W, orient, wall_line, span, occupied_input, reserved)
        if cand is None:
            return None

        far = tuple(cand["far"])
        near = tuple(cand["near"])
        grid[far] = int(color)
        grid[near] = int(color)
        cand["color"] = int(color)
        specs.append(cand)

        occupied_input.update({far, near})
        semantic_cells = {far, near, tuple(cand["ref"])}
        semantic_cells.update(tuple(p) for p in cand["incoming"])
        semantic_cells.update(tuple(p) for p in cand["ray"])
        semantic_cells.update(tuple(p) for p in cand["cross"])
        reserved.update(semantic_cells)

    expected = _render_expected(grid, specs)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid reflected-mirror input")
