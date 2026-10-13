import numpy as np


BG = 0
COLOR_POOL = tuple(range(1, 10))
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

GENERATION_ATTEMPTS = 500
LATTICE_ROWS = (3, 4)
LATTICE_COLS = (3, 4)
LATTICE_SPACING = 7
LATTICE_MARGIN = 3
ARROW_COUNT_RANGE = (3, 6)


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _arrow_cells(center: tuple[int, int], direction: tuple[int, int]) -> np.ndarray:
    r, c = center
    dr, dc = direction
    pr, pc = -dc, dr
    return np.array(
        [
            (r, c),
            (r + dr, c + dc),
            (r + 2 * dr, c + 2 * dc),
            (r + pr, c + pc),
            (r - pr, c - pc),
        ],
        dtype=int,
    )


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _expanded_corners(coords: np.ndarray) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    return [
        (r0 - 1, c0 - 1),
        (r0 - 1, c1 + 1),
        (r1 + 1, c0 - 1),
        (r1 + 1, c1 + 1),
    ]


def _parse_arrow(coords: np.ndarray, color: int) -> dict[str, object]:
    cells = {tuple(map(int, coord)) for coord in coords}
    assert len(cells) == 5, "each arrow must occupy exactly five cells"

    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        deg = 0
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in cells:
                deg += 1
        degrees[(r, c)] = deg

    assert sorted(degrees.values()) == [1, 1, 1, 2, 3], "arrow must be a T pentomino"
    centers = [cell for cell, deg in degrees.items() if deg == 3]
    assert len(centers) == 1
    center = centers[0]

    stem_neighbors = []
    for dr, dc in DIRS4:
        nbr = (center[0] + dr, center[1] + dc)
        if nbr in cells and degrees[nbr] == 2:
            stem_neighbors.append(nbr)
    assert len(stem_neighbors) == 1, "arrow must have one forward stem neighbor"
    stem = stem_neighbors[0]
    direction = (stem[0] - center[0], stem[1] - center[1])
    assert direction in DIRS4

    expected = {tuple(map(int, coord)) for coord in _arrow_cells(center, direction)}
    assert cells == expected, "component does not match the expected arrow footprint"

    return {
        "color": int(color),
        "center": center,
        "direction": direction,
        "coords": np.array(sorted(cells), dtype=int),
    }


def _detect_arrows(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    arrows: list[dict[str, object]] = []
    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == int(color)).astype(int)
        arrows.append(_parse_arrow(coords, int(color)))
    assert arrows, "expected at least one arrow"
    return bg, arrows


def _successor_map(shape: tuple[int, int], arrows: list[dict[str, object]]) -> list[int]:
    owner: dict[tuple[int, int], int] = {}
    for idx, arrow in enumerate(arrows):
        coords = np.asarray(arrow["coords"], dtype=int)
        for r, c in coords:
            pos = (int(r), int(c))
            assert pos not in owner, "arrows must not overlap"
            owner[pos] = idx

    successors = [-1] * len(arrows)
    for idx, arrow in enumerate(arrows):
        center_r, center_c = arrow["center"]
        dr, dc = arrow["direction"]
        r = center_r + 3 * dr
        c = center_c + 3 * dc
        while 0 <= r < shape[0] and 0 <= c < shape[1]:
            hit = owner.get((int(r), int(c)))
            if hit is not None:
                assert hit != idx
                successors[idx] = int(hit)
                break
            r += dr
            c += dc
    return successors


def _analyze_chains(successors: list[int]) -> tuple[list[int], dict[int, int], dict[int, int]] | None:
    n = len(successors)
    indegree = [0] * n
    for nxt in successors:
        if nxt != -1:
            indegree[nxt] += 1
            if indegree[nxt] > 1:
                return None

    heads = [idx for idx, deg in enumerate(indegree) if deg == 0]
    if not heads:
        return None

    tails: dict[int, int] = {}
    lengths: dict[int, int] = {}
    seen: set[int] = set()
    for head in heads:
        cur = head
        local_seen: set[int] = set()
        length = 1
        while successors[cur] != -1:
            if cur in local_seen:
                return None
            local_seen.add(cur)
            cur = successors[cur]
            length += 1
        tails[head] = cur
        lengths[head] = length
        seen.update(local_seen)
        seen.add(cur)

    if len(seen) != n:
        return None
    return heads, tails, lengths


def _render_grid(shape: tuple[int, int], arrows: list[dict[str, object]]) -> np.ndarray | None:
    grid = np.full(shape, BG, dtype=int)
    for arrow in arrows:
        coords = _arrow_cells(arrow["center"], arrow["direction"])
        if np.any(coords[:, 0] < 0) or np.any(coords[:, 0] >= shape[0]) or np.any(coords[:, 1] < 0) or np.any(coords[:, 1] >= shape[1]):
            return None
        if np.any(grid[coords[:, 0], coords[:, 1]] != BG):
            return None
        grid[coords[:, 0], coords[:, 1]] = int(arrow["color"])
    return grid


def _expected_output(grid: np.ndarray, arrows: list[dict[str, object]], successors: list[int]) -> np.ndarray | None:
    analyzed = _analyze_chains(successors)
    if analyzed is None:
        return None
    heads, tails, _lengths = analyzed

    out = grid.copy()
    used_corners: dict[tuple[int, int], int] = {}
    for head in heads:
        tail = tails[head]
        color = int(arrows[head]["color"])
        for r, c in _expanded_corners(np.asarray(arrows[tail]["coords"], dtype=int)):
            if not (0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]):
                return None
            if int(grid[r, c]) != BG:
                return None
            previous = used_corners.get((r, c))
            if previous is not None and previous != color:
                return None
            used_corners[(r, c)] = color
            out[r, c] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Follow arrow-to-arrow ray chains and stamp each chain head color onto its tail's outer corners.

    Input structure:
    - The background is black.
    - Each non-background color forms exactly one 5-cell arrow-shaped T object.
    - Every arrow's T stem points in one of the four cardinal directions, and arrows do not overlap.

    Rule:
    - Decode each arrow's pointing direction from its T shape.
    - From the front tip of each arrow, cast a one-cell-wide ray forward and link it to the first other arrow hit by that ray, if any.
    - Treat the resulting directed graph as one or more chains; for each chain, keep every arrow unchanged and place four single-pixel markers around the last arrow's bbox, using the first arrow's color.

    Key insight:
    - The T geometry is both the object identity and the direction code: once the directions are decoded, the forward first-hit links reveal color-transfer chains whose only visible effect is the tail-corner marking.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, arrows = _detect_arrows(g)
    successors = _successor_map(g.shape, arrows)
    analyzed = _analyze_chains(successors)
    assert analyzed is not None, "arrow links must form disjoint chains"
    heads, tails, _lengths = analyzed

    out = g.copy()
    for head in heads:
        tail = tails[head]
        color = int(arrows[head]["color"])
        for r, c in _expanded_corners(np.asarray(arrows[tail]["coords"], dtype=int)):
            assert 0 <= r < g.shape[0] and 0 <= c < g.shape[1]
            assert int(g[r, c]) == bg
            out[r, c] = color
    return out


def _sample_lattice_shape(rng: np.random.Generator) -> tuple[int, int, list[int], list[int]]:
    row_count = int(rng.integers(LATTICE_ROWS[0], LATTICE_ROWS[1] + 1))
    col_count = int(rng.integers(LATTICE_COLS[0], LATTICE_COLS[1] + 1))
    extra_rows = int(rng.integers(0, 31 - LATTICE_SPACING * row_count))
    extra_cols = int(rng.integers(0, 31 - LATTICE_SPACING * col_count))
    height = LATTICE_SPACING * row_count + extra_rows
    width = LATTICE_SPACING * col_count + extra_cols
    rows = [LATTICE_MARGIN + LATTICE_SPACING * i for i in range(row_count)]
    cols = [LATTICE_MARGIN + LATTICE_SPACING * i for i in range(col_count)]
    return height, width, rows, cols


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    height, width, rows, cols = _sample_lattice_shape(rng)
    lattice = [(r, c) for r in rows for c in cols]
    max_arrows = min(ARROW_COUNT_RANGE[1], len(lattice), len(COLOR_POOL))
    arrow_count = int(rng.integers(ARROW_COUNT_RANGE[0], max_arrows + 1))

    chosen = rng.choice(len(lattice), size=arrow_count, replace=False)
    colors = [int(v) for v in rng.choice(COLOR_POOL, size=arrow_count, replace=False)]

    arrows: list[dict[str, object]] = []
    for idx, lattice_idx in enumerate(chosen):
        direction = DIRS4[int(rng.integers(len(DIRS4)))]
        arrows.append(
            {
                "color": colors[idx],
                "center": lattice[int(lattice_idx)],
                "direction": direction,
            }
        )

    grid = _render_grid((height, width), arrows)
    if grid is None:
        return None

    generated_arrows = []
    for arrow in arrows:
        generated_arrows.append(
            {
                "color": int(arrow["color"]),
                "center": tuple(map(int, arrow["center"])),
                "direction": tuple(map(int, arrow["direction"])),
                "coords": _arrow_cells(tuple(map(int, arrow["center"])), tuple(map(int, arrow["direction"]))),
            }
        )

    successors = _successor_map(grid.shape, generated_arrows)
    analyzed = _analyze_chains(successors)
    if analyzed is None:
        return None
    _heads, _tails, lengths = analyzed

    if not any(succ != -1 for succ in successors):
        return None
    if max(lengths.values()) < 2:
        return None

    expected = _expected_output(grid, generated_arrows, successors)
    if expected is None:
        return None

    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GENERATION_ATTEMPTS} attempts")
