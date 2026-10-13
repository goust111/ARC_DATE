import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8)
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
GENERATION_ATTEMPTS = 400

TEMPLATE_BANK = (
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 0], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 1, 0], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[0, 0, 1], [0, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    np.array([[1, 0, 1], [1, 1, 1], [0, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [0, 0, 0]], dtype=bool),
)


def _parse_legend(grid: np.ndarray) -> tuple[int, dict[int, np.ndarray]]:
    rows_with_grey = np.where(np.any(grid == GREY, axis=1))[0]
    assert rows_with_grey.size > 0, "legend rows must exist"
    legend_h = int(rows_with_grey[-1] + 1)
    assert legend_h == 3, "legend must encode 3x3 templates"
    assert np.array_equal(rows_with_grey, np.arange(legend_h)), "grey must appear only in the top legend band"

    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:legend_h, c] == GREY)]
    assert len(sep_cols) >= 2, "legend needs at least one cell"
    assert all(b - a == 4 for a, b in zip(sep_cols, sep_cols[1:])), "legend separators must split 3x3 cells"

    templates: dict[int, np.ndarray] = {}
    for c0, c1 in zip(sep_cols, sep_cols[1:]):
        cell = grid[:legend_h, c0 + 1 : c1]
        assert cell.shape == (3, 3)
        colors = set(map(int, np.unique(cell))) - {BLACK}
        assert len(colors) == 1, "each legend cell must use exactly one key color"
        color = int(next(iter(colors)))
        assert color not in {GREY, MAROON}
        assert color not in templates, "legend colors must be unique"
        mask = cell == color
        assert mask.any(), "template cannot be empty"
        templates[color] = mask

    allowed = {BLACK, GREY, *templates.keys()}
    assert set(map(int, np.unique(grid[:legend_h]))).issubset(allowed)
    return legend_h, templates


def _find_seeds(grid: np.ndarray, legend_h: int, templates: dict[int, np.ndarray]) -> list[tuple[int, int, int]]:
    template_colors = set(templates.keys())
    main_area = grid[legend_h:, :]
    allowed = {BLACK, MAROON, *template_colors}
    assert set(map(int, np.unique(main_area))).issubset(allowed), "main area must contain only bg, walls, and seed colors"

    seeds: list[tuple[int, int, int]] = []
    for rr, cc in np.argwhere(np.isin(main_area, list(template_colors))):
        r = int(rr + legend_h)
        c = int(cc)
        color = int(grid[r, c])
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and nr >= legend_h:
                assert int(grid[nr, nc]) != color, "seed pixels must be isolated"
        seeds.append((r, c, color))

    assert seeds, "expected at least one seed pixel"
    return seeds


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    legend_h, templates = _parse_legend(g)
    seeds = _find_seeds(g, legend_h, templates)

    out = g.copy()
    h, w = g.shape
    for r, c, color in seeds:
        mask = templates[color]
        for mr in range(3):
            for mc in range(3):
                if not bool(mask[mr, mc]):
                    continue
                rr = r + mr - 1
                cc = c + mc - 1
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if rr < legend_h or int(g[rr, cc]) == MAROON:
                    continue
                prev = int(out[rr, cc])
                assert prev in {BLACK, color}, "stamp footprints must not conflict"
                out[rr, cc] = color
        out[r, c] = color
    return out


def _random_templates(rng: np.random.Generator, count: int) -> dict[int, np.ndarray]:
    colors = list(map(int, rng.choice(COLOR_POOL, size=count, replace=False)))
    template_ids = rng.choice(len(TEMPLATE_BANK), size=count, replace=False)
    return {color: TEMPLATE_BANK[int(idx)].copy() for color, idx in zip(colors, template_ids)}


def _place_legend(grid: np.ndarray, templates: dict[int, np.ndarray], rng: np.random.Generator) -> int:
    legend_w = 4 * len(templates) + 1
    left = int(rng.integers(0, grid.shape[1] - legend_w + 1))
    for i in range(len(templates) + 1):
        grid[:3, left + 4 * i] = GREY
    for i, color in enumerate(templates):
        cell = grid[:3, left + 4 * i + 1 : left + 4 * i + 4]
        cell[templates[color]] = int(color)
    return left


def _can_place_center(center: tuple[int, int], centers: list[tuple[int, int]]) -> bool:
    r, c = center
    return all(max(abs(r - r0), abs(c - c0)) >= 3 for r0, c0 in centers)


def _protected_zone(centers: list[tuple[int, int]], h: int, w: int) -> np.ndarray:
    protected = np.zeros((h, w), dtype=bool)
    for r, c in centers:
        protected[r - 1 : r + 2, c - 1 : c + 2] = True
    return protected


def _blocking_wall_cells(
    center: tuple[int, int],
    mask: np.ndarray,
    grid: np.ndarray,
    protected: np.ndarray,
    main_top: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    r, c = center
    candidates: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for mr in range(3):
        for mc in range(3):
            if (mr, mc) == (1, 1) or not bool(mask[mr, mc]):
                continue
            rr = r + mr - 1
            cc = c + mc - 1
            if grid[rr, cc] != BLACK:
                continue
            outward: list[tuple[int, int]] = []
            for dr, dc in DIR4:
                nr, nc = rr + dr, cc + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if nr < main_top or grid[nr, nc] != BLACK or bool(protected[nr, nc]):
                    continue
                outward.append((nr, nc))
            if outward:
                candidates.append(((rr, cc), outward))

    if not candidates:
        return None

    start, outward = candidates[int(rng.integers(len(candidates)))]
    cells = [start, outward[int(rng.integers(len(outward)))]]
    tail = cells[-1]
    extra_steps = int(rng.integers(0, 3))
    for _ in range(extra_steps):
        next_options: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nr, nc = tail[0] + dr, tail[1] + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if nr < main_top or grid[nr, nc] != BLACK or bool(protected[nr, nc]) or (nr, nc) in cells:
                continue
            next_options.append((nr, nc))
        if not next_options:
            break
        tail = next_options[int(rng.integers(len(next_options)))]
        cells.append(tail)
    return cells


def _grow_extra_wall(
    grid: np.ndarray,
    protected: np.ndarray,
    main_top: int,
    rng: np.random.Generator,
) -> bool:
    candidates = np.argwhere((grid == BLACK) & (~protected))
    candidates = candidates[candidates[:, 0] >= main_top]
    if len(candidates) == 0:
        return False

    start_r, start_c = map(int, candidates[int(rng.integers(len(candidates)))])
    target_size = int(rng.integers(2, 6))
    wall = [(start_r, start_c)]

    frontier = [(start_r, start_c)]
    while frontier and len(wall) < target_size:
        base = frontier[int(rng.integers(len(frontier)))]
        next_options: list[tuple[int, int]] = []
        for dr, dc in DIR4:
            nr, nc = base[0] + dr, base[1] + dc
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                continue
            if nr < main_top or grid[nr, nc] != BLACK or bool(protected[nr, nc]) or (nr, nc) in wall:
                continue
            next_options.append((nr, nc))
        if not next_options:
            frontier.remove(base)
            continue
        nxt = next_options[int(rng.integers(len(next_options)))]
        wall.append(nxt)
        frontier.append(nxt)

    if len(wall) < 2:
        return False
    for r, c in wall:
        grid[r, c] = MAROON
    return True


def _seed_records(rng: np.random.Generator, templates: dict[int, np.ndarray], h: int, w: int, main_top: int) -> list[dict]:
    seed_count = int(rng.integers(len(templates), min(len(templates) + 4, 8)))
    centers: list[tuple[int, int]] = []
    for _ in range(seed_count):
        placed = False
        for _ in range(300):
            center = (
                int(rng.integers(main_top + 1, h - 1)),
                int(rng.integers(1, w - 1)),
            )
            if _can_place_center(center, centers):
                centers.append(center)
                placed = True
                break
        if not placed:
            raise RuntimeError("failed to place enough seeds")

    seed_colors = list(templates.keys())
    extra = seed_count - len(seed_colors)
    if extra > 0:
        seed_colors.extend(map(int, rng.choice(list(templates.keys()), size=extra, replace=True)))
    rng.shuffle(seed_colors)

    blocked_count = int(rng.integers(1, seed_count))
    blocked_ids = set(map(int, rng.choice(seed_count, size=blocked_count, replace=False)))

    return [
        {
            "center": centers[i],
            "color": int(seed_colors[i]),
            "mask": templates[int(seed_colors[i])],
            "blocked": i in blocked_ids,
        }
        for i in range(seed_count)
    ]


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    template_count = int(rng.integers(2, 6))
    templates = _random_templates(rng, template_count)

    legend_w = 4 * template_count + 1
    h = int(rng.integers(14, 25))
    w = int(rng.integers(max(legend_w + 2, 18), 31))
    spacer = int(rng.integers(1, 4))
    main_top = 3 + spacer
    if main_top + 3 >= h:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    _place_legend(grid, templates, rng)
    try:
        seeds = _seed_records(rng, templates, h, w, main_top)
    except RuntimeError:
        return None

    centers = [seed["center"] for seed in seeds]
    protected = _protected_zone(centers, h, w)

    for seed in seeds:
        r, c = seed["center"]
        grid[r, c] = seed["color"]

    blocked_hits = 0
    clear_hits = 0
    for seed in seeds:
        if not seed["blocked"]:
            clear_hits += 1
            continue
        cells = _blocking_wall_cells(seed["center"], seed["mask"], grid, protected, main_top, rng)
        if cells is None:
            return None
        for r, c in cells:
            grid[r, c] = MAROON
        blocked_hits += 1

    extra_wall_attempts = int(rng.integers(0, 3))
    for _ in range(extra_wall_attempts):
        _grow_extra_wall(grid, protected, main_top, rng)

    if blocked_hits == 0 or clear_hits == 0:
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_ATTEMPTS} attempts")
