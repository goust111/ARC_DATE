import itertools

import numpy as np


BG = 0
PATH = 7
RED = 2
GREEN = 3
MAROON = 9

RUN_LEN = 4
VERTICAL_GAP = 2
VALID_RUN_COUNTS = (5, 6, 7)
MAX_MARKERS = 4
MAX_GENERATION_ATTEMPTS = 300

_RNG = np.random.default_rng()
_PROFILE_CACHE: dict[int, list[tuple[int, ...]]] = {}


def _rot_cw(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return dc, -dr


def _rot_ccw(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return -dc, dr


def _turn_kind(prev_vec: tuple[int, int], next_vec: tuple[int, int]) -> str | None:
    if _rot_ccw(prev_vec) == next_vec:
        return "L"
    if _rot_cw(prev_vec) == next_vec:
        return "R"
    return None


def _neighbors4(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    r, c = cell
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _bbox(points: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _trace_path(grid: np.ndarray) -> list[tuple[int, int]]:
    pts = [tuple(map(int, rc)) for rc in np.argwhere(grid != BG)]
    assert pts, "grid must contain a non-background path"

    point_set = set(pts)
    green = [tuple(map(int, rc)) for rc in np.argwhere(grid == GREEN)]
    assert len(green) == 2, "expected exactly two green endpoints"

    adj = {
        pt: [nbr for nbr in _neighbors4(pt) if nbr in point_set]
        for pt in point_set
    }
    degrees = {pt: len(nbrs) for pt, nbrs in adj.items()}
    endpoints = [pt for pt, deg in degrees.items() if deg == 1]
    assert len(endpoints) == 2, "path must have exactly two graph endpoints"
    assert set(endpoints) == set(green), "green cells must be the path endpoints"
    assert all(deg == 2 for pt, deg in degrees.items() if pt not in endpoints), "path may not branch"

    start = green[0]
    order = [start]
    prev = None
    cur = start
    while True:
        nxts = [nbr for nbr in adj[cur] if nbr != prev]
        if not nxts:
            break
        assert len(nxts) == 1, "path ordering must be unique"
        nxt = nxts[0]
        order.append(nxt)
        prev, cur = cur, nxt

    assert len(order) == len(point_set), "path must be connected"
    assert order[-1] in green and order[-1] != start, "path must end at the other green endpoint"
    return order


def _fold_points(order: list[tuple[int, int]], marker_indices: tuple[int, ...]) -> list[tuple[int, int]] | None:
    pts = [list(pt) for pt in order]
    for pivot_idx in marker_indices:
        if pivot_idx <= 0 or pivot_idx >= len(pts) - 1:
            return None
        prev_vec = (pts[pivot_idx][0] - pts[pivot_idx - 1][0], pts[pivot_idx][1] - pts[pivot_idx - 1][1])
        next_vec = (pts[pivot_idx + 1][0] - pts[pivot_idx][0], pts[pivot_idx + 1][1] - pts[pivot_idx][1])
        turn = _turn_kind(prev_vec, next_vec)
        if turn is None:
            return None
        rotate = _rot_ccw if turn == "L" else _rot_cw
        pr, pc = pts[pivot_idx]
        for j in range(pivot_idx + 1, len(pts)):
            dr = pts[j][0] - pr
            dc = pts[j][1] - pc
            rr, cc = rotate((dr, dc))
            pts[j] = [pr + rr, pc + cc]
    return [tuple(pt) for pt in pts]


def _render_folded(
    order: list[tuple[int, int]], marker_indices: tuple[int, ...]
) -> tuple[np.ndarray, dict[str, float | int]] | None:
    folded = _fold_points(order, marker_indices)
    if folded is None:
        return None

    last_index: dict[tuple[int, int], int] = {}
    for idx, pt in enumerate(folded):
        last_index[pt] = idx

    r0, c0, r1, c1 = _bbox(list(last_index))
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)

    marker_set = set(marker_indices)
    for idx, (r, c) in enumerate(folded):
        if idx == 0 or idx == len(folded) - 1:
            color = GREEN
        elif idx in marker_set:
            color = MAROON
        else:
            color = PATH
        out[r - r0, c - c0] = color

    # Endpoints stay green even if an overlap would otherwise hide them.
    for idx in (0, len(folded) - 1):
        r, c = folded[idx]
        out[r - r0, c - c0] = GREEN

    visible_markers = sum(1 for idx in marker_indices if last_index[folded[idx]] == idx)
    occupied = int(len(last_index))
    area = int(out.shape[0] * out.shape[1])
    meta = {
        "occupied": occupied,
        "area": area,
        "holes": area - occupied,
        "visible_markers": visible_markers,
        "height": int(out.shape[0]),
        "width": int(out.shape[1]),
        "fill": occupied / area,
    }
    return out, meta


def _candidate_output(order: list[tuple[int, int]], red_cells: set[tuple[int, int]]) -> tuple[np.ndarray, dict[str, float | int]]:
    marker_indices = tuple(i for i, pt in enumerate(order) if pt in red_cells)
    rendered = _render_folded(order, marker_indices)
    assert rendered is not None, "red markers must remain turn pivots throughout the fold sequence"
    return rendered


def _output_key(out: np.ndarray, meta: dict[str, float | int]) -> tuple:
    return (
        int(meta["area"]),
        int(meta["holes"]),
        -int(meta["visible_markers"]),
        abs(int(meta["height"]) - int(meta["width"])),
        int(meta["height"]),
        int(meta["width"]),
        out.tobytes(),
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold a marked snake path into its compact cropped result.

    The grid contains one non-branching 4-connected path on a black background.
    Two green endpoints define the path order. Red cells mark turn pivots. For each
    marked turn, the suffix after that pivot is rotated by 90 degrees so the first
    post-pivot step folds back over the incoming segment; left turns rotate CCW and
    right turns rotate CW. The final output is the tight crop of the folded path,
    using orange for path pixels, maroon for visible markers, and green for endpoints.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    order = _trace_path(g)
    red_cells = {tuple(map(int, rc)) for rc in np.argwhere(g == RED)}

    candidates = [
        _candidate_output(order, red_cells),
        _candidate_output(list(reversed(order)), red_cells),
    ]
    if np.array_equal(candidates[0][0], candidates[1][0]):
        return candidates[0][0]
    return min(candidates, key=lambda item: _output_key(item[0], item[1]))[0]


def _build_spaced_snake(run_count: int, run_len: int = RUN_LEN, gap: int = VERTICAL_GAP) -> list[tuple[int, int]]:
    pts = [(0, 0)]
    r = 0
    c = 0
    direction = 1
    for run_idx in range(run_count):
        target_c = run_len - 1 if direction == 1 else 0
        while c != target_c:
            c += direction
            pts.append((r, c))
        if run_idx == run_count - 1:
            break
        for _ in range(gap):
            r += 1
            pts.append((r, c))
        direction *= -1
    return pts


def _turn_indices(order: list[tuple[int, int]]) -> list[int]:
    turns = []
    for i in range(1, len(order) - 1):
        prev_vec = (order[i][0] - order[i - 1][0], order[i][1] - order[i - 1][1])
        next_vec = (order[i + 1][0] - order[i][0], order[i + 1][1] - order[i][1])
        if prev_vec != next_vec:
            turns.append(i)
    return turns


def _valid_marker_profiles(run_count: int) -> list[tuple[int, ...]]:
    if run_count in _PROFILE_CACHE:
        return _PROFILE_CACHE[run_count]

    order = _build_spaced_snake(run_count)
    turns = _turn_indices(order)
    profiles: list[tuple[int, ...]] = []
    for count in range(1, min(MAX_MARKERS, len(turns)) + 1):
        for marks in itertools.combinations(turns, count):
            kinds = {
                _turn_kind(
                    (order[idx][0] - order[idx - 1][0], order[idx][1] - order[idx - 1][1]),
                    (order[idx + 1][0] - order[idx][0], order[idx + 1][1] - order[idx][1]),
                )
                for idx in marks
            }
            if kinds != {"L", "R"}:
                continue

            rendered = _render_folded(order, marks)
            if rendered is None:
                continue
            out, meta = rendered
            if int(meta["visible_markers"]) != len(marks):
                continue
            if int(meta["height"]) < 3 or int(meta["width"]) < 3:
                continue
            if float(meta["fill"]) < 0.8:
                continue

            rev_order = list(reversed(order))
            rev_marks = tuple(sorted(len(order) - 1 - idx for idx in marks))
            rev_rendered = _render_folded(rev_order, rev_marks)
            if rev_rendered is None or not np.array_equal(out, rev_rendered[0]):
                continue
            profiles.append(marks)

    assert profiles, f"no valid fold profiles for run_count={run_count}"
    _PROFILE_CACHE[run_count] = profiles
    return profiles


def _apply_random_transform(order: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = np.array(order, dtype=int)
    if bool(_RNG.integers(2)):
        pts[:, 1] *= -1
    rot = int(_RNG.integers(4))
    for _ in range(rot):
        pts = np.stack((pts[:, 1], -pts[:, 0]), axis=1)
    return [tuple(map(int, rc)) for rc in pts]


def _paint_input(order: list[tuple[int, int]], marks: tuple[int, ...]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(order)
    top = int(_RNG.integers(1, 5))
    bottom = int(_RNG.integers(1, 5))
    left = int(_RNG.integers(1, 5))
    right = int(_RNG.integers(1, 5))
    grid = np.full((r1 - r0 + 1 + top + bottom, c1 - c0 + 1 + left + right), BG, dtype=int)

    mark_set = set(marks)
    for idx, (r, c) in enumerate(order):
        rr = r - r0 + top
        cc = c - c0 + left
        if idx == 0 or idx == len(order) - 1:
            grid[rr, cc] = GREEN
        elif idx in mark_set:
            grid[rr, cc] = RED
        else:
            grid[rr, cc] = PATH
    return grid


def _gen_attempt() -> np.ndarray | None:
    run_count = int(_RNG.choice(np.array(VALID_RUN_COUNTS)))
    profiles = _valid_marker_profiles(run_count)
    marks = profiles[int(_RNG.integers(len(profiles)))]

    order = _build_spaced_snake(run_count)
    order = _apply_random_transform(order)
    grid = _paint_input(order, marks)

    expected = _render_folded(order, marks)
    if expected is None:
        return None
    out_expected, _ = expected
    if not np.array_equal(main(grid), out_expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid foldable snake input")
