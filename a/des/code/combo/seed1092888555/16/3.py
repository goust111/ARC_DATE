import numpy as np


BG = 8
PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 9]
GRID_MIN = 14
GRID_MAX = 22
CLUSTER_COUNT_RANGE = (2, 4)
CLUSTER_SIZE_RANGE = (3, 6)
MARGIN = 4
GEN_ATTEMPTS = 400
CLUSTER_ATTEMPTS = 200

BASE_TEMPLATES = {
    "diag": [
        [(-3, -1), (-1, 2), (4, -1)],
        [(-4, -1), (-1, 3), (2, 1), (3, -3)],
    ],
    "axis": [
        [(-3, 0), (0, -2), (3, 2)],
        [(-4, 0), (0, 3), (2, -1), (2, -2)],
        [(-3, 0), (3, 0), (0, -3), (0, 3)],
    ],
    "merge": [
        [(-1, 0), (1, 0), (-2, -1), (2, 1)],
        [(0, -1), (0, 1), (-2, 2), (2, -2)],
        [(-1, -1), (1, 1), (-3, 0), (3, 0)],
    ],
}

EXTRA_PAIRS = [
    [(-2, -1), (2, 1)],
    [(-2, 1), (2, -1)],
    [(-3, -1), (3, 1)],
    [(-3, 1), (3, -1)],
    [(-1, -3), (1, 3)],
    [(-1, 3), (1, -3)],
    [(0, -2), (0, 2)],
    [(-2, 0), (2, 0)],
]


def _round_half_up(value: float) -> int:
    return int(np.floor(value + 0.5))


def _cluster_center(points: np.ndarray) -> tuple[int, int]:
    return (
        _round_half_up(float(points[:, 0].mean())),
        _round_half_up(float(points[:, 1].mean())),
    )


def _step_toward(value: int, target: int) -> int:
    if value < target:
        return value + 1
    if value > target:
        return value - 1
    return value


def _move_cluster(points: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    center_r, center_c = _cluster_center(points)
    moved = np.array(
        [
            [_step_toward(int(r), center_r), _step_toward(int(c), center_c)]
            for r, c in points
        ],
        dtype=int,
    )
    return (center_r, center_c), moved


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each same-color cluster one step toward its own rounded centroid.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = np.full_like(g, BG)
    occupied: dict[tuple[int, int], int] = {}

    for color in sorted(int(c) for c in np.unique(g) if int(c) != BG):
        points = np.argwhere(g == color)
        _center, moved = _move_cluster(points)
        for r, c in np.unique(moved, axis=0):
            pos = (int(r), int(c))
            prev = occupied.get(pos)
            assert prev in (None, color), "different colors cannot converge to the same output cell"
            occupied[pos] = color
            out[pos] = color

    return out


def _transform(rels: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    swap_axes = bool(rng.integers(0, 2))
    sign_r = -1 if bool(rng.integers(0, 2)) else 1
    sign_c = -1 if bool(rng.integers(0, 2)) else 1

    out = []
    for dr, dc in rels:
        rr, cc = (dc, dr) if swap_axes else (dr, dc)
        out.append((sign_r * rr, sign_c * cc))
    return out


def _classify(center: tuple[int, int], points: np.ndarray, moved: np.ndarray) -> str:
    center_r, center_c = center
    if len(np.unique(moved, axis=0)) < len(points):
        return "merge"
    if any(int(r) == center_r or int(c) == center_c for r, c in points):
        return "axis"
    return "diag"


def _neighbors(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    expanded = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                expanded.add((r + dr, c + dc))
    return expanded


def _build_cluster(
    family: str,
    center: tuple[int, int],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray] | None:
    templates = BASE_TEMPLATES[family]
    base = list(templates[int(rng.integers(0, len(templates)))])
    rels = _transform(base, rng)

    target_size = int(rng.integers(CLUSTER_SIZE_RANGE[0], CLUSTER_SIZE_RANGE[1] + 1))
    while len(rels) + 2 <= target_size:
        pair = _transform(list(EXTRA_PAIRS[int(rng.integers(0, len(EXTRA_PAIRS)))]), rng)
        if len(set(rels).intersection(pair)) == 0:
            rels.extend(pair)

    pts = np.array([(center[0] + dr, center[1] + dc) for dr, dc in rels], dtype=int)
    computed_center, moved = _move_cluster(pts)
    if computed_center != center:
        return None
    if _classify(computed_center, pts, moved) != family:
        return None
    return pts, moved


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    w = int(rng.integers(GRID_MIN, GRID_MAX + 1))
    grid = np.full((h, w), BG, dtype=int)

    cluster_count = int(rng.integers(CLUSTER_COUNT_RANGE[0], CLUSTER_COUNT_RANGE[1] + 1))
    colors = list(rng.choice(PALETTE, size=cluster_count, replace=False))
    families = list(rng.choice(["diag", "axis", "merge"], size=cluster_count, replace=True))

    reserved = set()
    moved_reserved = set()
    for color, family in zip(colors, families):
        built = None
        for _ in range(CLUSTER_ATTEMPTS):
            center = (
                int(rng.integers(MARGIN, h - MARGIN)),
                int(rng.integers(MARGIN, w - MARGIN)),
            )
            cluster = _build_cluster(family, center, rng)
            if cluster is None:
                continue
            points, moved = cluster
            if (
                np.any(points[:, 0] < 0)
                or np.any(points[:, 0] >= h)
                or np.any(points[:, 1] < 0)
                or np.any(points[:, 1] >= w)
            ):
                continue
            point_cells = {(int(r), int(c)) for r, c in points}
            moved_cells = {(int(r), int(c)) for r, c in np.unique(moved, axis=0)}
            union = point_cells | moved_cells
            if union & reserved:
                continue
            if moved_cells & moved_reserved:
                continue
            built = (points, moved)
            reserved |= _neighbors(union)
            moved_reserved |= moved_cells
            break
        if built is None:
            return None
        points, _moved = built
        for r, c in points:
            grid[int(r), int(c)] = color

    if len(np.unique(grid)) <= 2:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
