from __future__ import annotations

import numpy as np

BLACK = 0
BLUE = 1
GREEN = 3
ORANGE = 7
TEAL = 8

_WEDGE_KINDS = (
    ("down", (1, -1), (1, 1)),
    ("up", (-1, -1), (-1, 1)),
    ("right", (-1, 1), (1, 1)),
    ("left", (-1, -1), (1, -1)),
)

_GEN_HEIGHT_RANGE = (11, 15)
_GEN_PANEL_WIDTH_RANGE = (11, 14)
_GEN_BLUE_AREA_RANGE = (15, 28)
_GEN_WEDGE_HEIGHT_RANGE = (2, 4)
_GEN_CASES = ("one_accept", "one_reject", "two_mixed")
_GEN_CASE_WEIGHTS = np.array([0.22, 0.22, 0.56], dtype=float)
_GEN_RETRY_LIMIT = 400
_PLACEMENT_CACHE: dict[tuple[int, int], list[dict[str, object]]] = {}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _ray_marker_distances(grid: np.ndarray, r: int, c: int, dr: int, dc: int) -> list[int]:
    distances: list[int] = []
    d = 1
    while _in_bounds(grid.shape, r + dr * d, c + dc * d):
        if int(grid[r + dr * d, c + dc * d]) == GREEN:
            distances.append(d)
        d += 1
    return distances


def _triangle_cells(
    kind: str, tip_r: int, tip_c: int, height: int, shape: tuple[int, int]
) -> frozenset[tuple[int, int]]:
    cells = {(tip_r, tip_c)}
    if kind == "down":
        for d in range(1, height + 1):
            rr = tip_r + d
            for cc in range(tip_c - d, tip_c + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.add((rr, cc))
    elif kind == "up":
        for d in range(1, height + 1):
            rr = tip_r - d
            for cc in range(tip_c - d, tip_c + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.add((rr, cc))
    elif kind == "right":
        for d in range(1, height + 1):
            cc = tip_c + d
            for rr in range(tip_r - d, tip_r + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.add((rr, cc))
    elif kind == "left":
        for d in range(1, height + 1):
            cc = tip_c - d
            for rr in range(tip_r - d, tip_r + d + 1):
                if _in_bounds(shape, rr, cc):
                    cells.add((rr, cc))
    else:
        raise AssertionError(f"unexpected wedge kind: {kind}")
    return frozenset(cells)


def _pairwise_manhattan_gap(
    a: frozenset[tuple[int, int]], b: frozenset[tuple[int, int]]
) -> int:
    return min(abs(r1 - r2) + abs(c1 - c2) for r1, c1 in a for r2, c2 in b)


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    height, width = map(int, g.shape)
    assert width % 2 == 1 and width >= 5, "expected two equal panels and one separator"

    sep_col = width // 2
    assert np.all(g[:, sep_col] == ORANGE), "expected a full orange separator column"
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape == (height, sep_col)
    return left, right


def _detect_wedges(right: np.ndarray) -> list[dict[str, object]]:
    candidates: list[dict[str, object]] = []
    for r_raw, c_raw in np.argwhere(right == GREEN):
        r = int(r_raw)
        c = int(c_raw)
        for kind, side_a, side_b in _WEDGE_KINDS:
            da = _ray_marker_distances(right, r, c, side_a[0], side_a[1])
            db = _ray_marker_distances(right, r, c, side_b[0], side_b[1])
            common = set(da) & set(db)
            if not common:
                continue
            height = max(common)
            if height < 2:
                continue
            candidates.append(
                {
                    "tip": (r, c),
                    "kind": kind,
                    "height": height,
                    "triangle": _triangle_cells(kind, r, c, height, right.shape),
                }
            )

    best_by_tip: dict[tuple[int, int], dict[str, object]] = {}
    for cand in candidates:
        tip = cand["tip"]
        current = best_by_tip.get(tip)
        if current is None:
            best_by_tip[tip] = cand
            continue
        score = (int(cand["height"]), len(cand["triangle"]), str(cand["kind"]))
        cur_score = (int(current["height"]), len(current["triangle"]), str(current["kind"]))
        if score > cur_score:
            best_by_tip[tip] = cand

    deduped = sorted(
        best_by_tip.values(),
        key=lambda item: (-len(item["triangle"]), int(item["tip"][0]), int(item["tip"][1])),
    )

    wedges: list[dict[str, object]] = []
    used: set[tuple[int, int]] = set()
    for cand in deduped:
        triangle = set(cand["triangle"])
        if triangle & used:
            continue
        wedges.append(cand)
        used.update(triangle)
    return wedges


def main(grid: np.ndarray) -> np.ndarray:
    """Infer teal triangular wedges from green diagonal markers and overlay only the non-overlapping wedges onto the left panel.

    Input structure:
    - The grid is split into two equal panels by a full-height orange separator column.
    - The left panel is a black background with one interior blue 4-connected shape.
    - The right panel is a black background with sparse green pixels lying on two diagonal rays from one or two wedge tips.

    Rule:
    - In the right panel, treat each valid green tip as the vertex of a 90-degree wedge whose two sides are 45-degree diagonal rays.
    - A wedge is valid when both rays contain green markers at a shared distance from the tip; the wedge height is the farthest such shared distance.
    - Fill every non-green cell inside that triangular wedge teal while keeping the green markers green.
    - Remove the separator column and use the left panel as the output canvas.
    - For each inferred wedge region as a whole, copy its green and teal cells onto the left panel only if every corresponding left-panel cell is black; otherwise ignore that wedge completely.

    Key insight:
    - The right panel does not contain the overlay mask directly; the sparse green diagonal markers first define triangular wedge regions, and those inferred regions are then filtered by a no-overlap test against the left-panel blue shape.
    """
    left, right = _split_panels(grid)
    out = left.copy()
    left_foreground = left != BLACK

    for wedge in _detect_wedges(right):
        triangle = wedge["triangle"]
        if any(left_foreground[r, c] for r, c in triangle):
            continue
        for r, c in triangle:
            out[r, c] = GREEN if int(right[r, c]) == GREEN else TEAL
    return out


def _randint(lo: int, hi: int) -> int:
    return int(np.random.randint(lo, hi + 1))


def _choice(seq):
    return seq[int(np.random.randint(0, len(seq)))]


def _weighted_choice(options: tuple[str, ...], weights: np.ndarray) -> str:
    idx = int(np.random.choice(len(options), p=weights / weights.sum()))
    return str(options[idx])


def _generate_blue_shape(height: int, width: int) -> set[tuple[int, int]] | None:
    cells_domain = [
        (r, c)
        for r in range(2, height - 2)
        for c in range(2, width - 2)
    ]
    if len(cells_domain) < _GEN_BLUE_AREA_RANGE[0]:
        return None

    domain = set(cells_domain)
    target = _randint(_GEN_BLUE_AREA_RANGE[0], min(_GEN_BLUE_AREA_RANGE[1], len(cells_domain) - 2))
    shape = {_choice(cells_domain)}

    while len(shape) < target:
        frontier = sorted(
            {
                nbr
                for r, c in shape
                for nbr in _neighbors4(r, c)
                if nbr in domain and nbr not in shape
            }
        )
        if not frontier:
            return None
        shape.add(_choice(frontier))

    rows = [r for r, _ in shape]
    cols = [c for _, c in shape]
    if max(rows) - min(rows) < 3 or max(cols) - min(cols) < 3:
        return None
    if any(len([1 for nr, nc in _neighbors4(r, c) if (nr, nc) in shape]) == 0 for r, c in shape):
        return None
    return shape


def _ray_cells(kind: str, tip_r: int, tip_c: int, height: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    for name, side_a, side_b in _WEDGE_KINDS:
        if name != kind:
            continue
        ray_a = [(tip_r + side_a[0] * d, tip_c + side_a[1] * d) for d in range(1, height + 1)]
        ray_b = [(tip_r + side_b[0] * d, tip_c + side_b[1] * d) for d in range(1, height + 1)]
        return ray_a, ray_b
    raise AssertionError(f"unexpected wedge kind: {kind}")


def _placement_catalog(shape: tuple[int, int]) -> list[dict[str, object]]:
    cached = _PLACEMENT_CACHE.get(shape)
    if cached is not None:
        return cached

    height, width = shape
    placements: list[dict[str, object]] = []
    for kind, _side_a, _side_b in _WEDGE_KINDS:
        for wedge_height in range(_GEN_WEDGE_HEIGHT_RANGE[0], _GEN_WEDGE_HEIGHT_RANGE[1] + 1):
            for tip_r in range(height):
                for tip_c in range(width):
                    triangle = _triangle_cells(kind, tip_r, tip_c, wedge_height, shape)
                    expected_area = wedge_height * (wedge_height + 2) + 1
                    if len(triangle) != expected_area:
                        continue
                    ray_a, ray_b = _ray_cells(kind, tip_r, tip_c, wedge_height)
                    if not all(_in_bounds(shape, r, c) for r, c in ray_a + ray_b):
                        continue
                    placements.append(
                        {
                            "kind": kind,
                            "tip": (tip_r, tip_c),
                            "height": wedge_height,
                            "triangle": triangle,
                            "ray_a": tuple(ray_a),
                            "ray_b": tuple(ray_b),
                        }
                    )

    _PLACEMENT_CACHE[shape] = placements
    return placements


def _marker_cells(placement: dict[str, object]) -> set[tuple[int, int]]:
    height = int(placement["height"])
    ray_a = list(placement["ray_a"])
    ray_b = list(placement["ray_b"])
    distances = list(range(1, height))
    shared_count = 0 if not distances else _randint(0, len(distances))
    shared = set()
    if shared_count:
        picks = np.random.permutation(distances)[:shared_count]
        shared.update(int(d) for d in picks)
    shared.add(height)

    markers = {tuple(placement["tip"])}
    for d in range(1, height + 1):
        if d in shared or np.random.random() < 0.35:
            markers.add(ray_a[d - 1])
        if d in shared or np.random.random() < 0.35:
            markers.add(ray_b[d - 1])
    return markers


def _render_left_panel(shape: set[tuple[int, int]], height: int, width: int) -> np.ndarray:
    panel = np.zeros((height, width), dtype=int)
    for r, c in shape:
        panel[r, c] = BLUE
    return panel


def _render_right_panel(
    placements: list[dict[str, object]], height: int, width: int
) -> tuple[np.ndarray, list[set[tuple[int, int]]]]:
    panel = np.zeros((height, width), dtype=int)
    markers_per_wedge: list[set[tuple[int, int]]] = []
    for placement in placements:
        markers = _marker_cells(placement)
        markers_per_wedge.append(markers)
        for r, c in markers:
            panel[r, c] = GREEN
    return panel, markers_per_wedge


def _render_expected_output(
    left: np.ndarray,
    placements: list[dict[str, object]],
    markers_per_wedge: list[set[tuple[int, int]]],
    accepted: list[bool],
) -> np.ndarray:
    out = left.copy()
    for placement, markers, keep in zip(placements, markers_per_wedge, accepted):
        if not keep:
            continue
        for r, c in placement["triangle"]:
            out[r, c] = GREEN if (r, c) in markers else TEAL
    return out


def _choose_case(
    placements: list[dict[str, object]], blue_shape: set[tuple[int, int]]
) -> tuple[list[dict[str, object]], list[bool]] | None:
    blue = set(blue_shape)
    accept_pool = []
    reject_pool = []
    for placement in placements:
        triangle = set(placement["triangle"])
        overlap = len(triangle & blue)
        remaining = len(triangle) - overlap
        if overlap == 0:
            accept_pool.append(placement)
        elif overlap >= 2 and remaining >= 3:
            reject_pool.append(placement)

    case = _weighted_choice(_GEN_CASES, _GEN_CASE_WEIGHTS)
    if case == "one_accept":
        if not accept_pool:
            return None
        return [_choice(accept_pool)], [True]
    if case == "one_reject":
        if not reject_pool:
            return None
        return [_choice(reject_pool)], [False]

    if not accept_pool or not reject_pool:
        return None

    accept_order = [accept_pool[int(i)] for i in np.random.permutation(len(accept_pool))]
    reject_order = [reject_pool[int(i)] for i in np.random.permutation(len(reject_pool))]
    for accept in accept_order:
        for reject in reject_order:
            tri_a = accept["triangle"]
            tri_b = reject["triangle"]
            if tri_a & tri_b:
                continue
            if _pairwise_manhattan_gap(tri_a, tri_b) < 2:
                continue
            return [accept, reject], [True, False]
    return None


def _attempt_generate() -> np.ndarray | None:
    height = _randint(*_GEN_HEIGHT_RANGE)
    panel_width = _randint(*_GEN_PANEL_WIDTH_RANGE)

    blue_shape = _generate_blue_shape(height, panel_width)
    if blue_shape is None:
        return None

    all_placements = _placement_catalog((height, panel_width))
    chosen = _choose_case(all_placements, blue_shape)
    if chosen is None:
        return None
    placements, accepted = chosen

    left = _render_left_panel(blue_shape, height, panel_width)
    right, markers_per_wedge = _render_right_panel(placements, height, panel_width)

    grid = np.zeros((height, panel_width * 2 + 1), dtype=int)
    grid[:, :panel_width] = left
    grid[:, panel_width] = ORANGE
    grid[:, panel_width + 1 :] = right

    expected = _render_expected_output(left, placements, markers_per_wedge, accepted)
    if not np.array_equal(main(grid), expected):
        return None
    if len(_detect_wedges(right)) != len(placements):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRY_LIMIT):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid wedge-overlay ARC input")
