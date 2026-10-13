import numpy as np


BG = 7
RED = 2
GREY = 5
BLUE = 1


def _run_start_indices(lines: np.ndarray) -> list[int]:
    keep = [0]
    for i in range(1, lines.shape[0]):
        if not np.array_equal(lines[i], lines[i - 1]):
            keep.append(i)
    return keep


def _compress_grid(grid: np.ndarray) -> np.ndarray:
    row_starts = _run_start_indices(grid)
    col_starts = _run_start_indices(grid.T)
    return grid[np.ix_(row_starts, col_starts)].copy()


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _color_components(grid: np.ndarray, color: int) -> list[np.ndarray]:
    mask = grid == color
    seen = np.zeros(grid.shape, dtype=bool)
    components: list[np.ndarray] = []
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors8(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _diamond_outline(cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    pts = set()
    for dr in range(-radius, radius + 1):
        dc = radius - abs(dr)
        pts.add((cr + dr, cc + dc))
        pts.add((cr + dr, cc - dc))
    return pts


def _fit_diamond(coords: np.ndarray) -> tuple[int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    assert height == width and height % 2 == 1
    radius = height // 2
    cr = int((r0 + r1) // 2)
    cc = int((c0 + c1) // 2)
    assert {tuple(map(int, pt)) for pt in coords} == _diamond_outline(cr, cc, radius)
    return cr, cc, radius


def _paint(out: np.ndarray, pts: set[tuple[int, int]], color: int) -> None:
    for r, c in pts:
        out[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compress duplicate row and column runs, then shrink red diamond outlines and expand gray ones.

    Input structure:
    - The canvas is an orange background with 2-3 one-cell-thick Manhattan-diamond outlines in red or gray, plus an optional blue diamond that stays fixed.
    - The visible input is a stretched version of a smaller true grid: each true row and each true column has been repeated 1-3 times, creating adjacent identical row and column runs.
    - The red and gray objects are complete, unclipped diamond outlines whose geometry is only correct after removing the duplicate runs.

    Rule:
    - Keep the first row from each maximal run of identical neighboring rows, and the first column from each maximal run of identical neighboring columns.
    - In that compressed grid, identify each red and gray one-pixel-thick Manhattan diamond outline by its center and radius.
    - Clear all red and gray pixels to orange, redraw each red diamond at radius r-1, and redraw each gray diamond at radius r+1.
    - Leave blue diamonds unchanged and return the compressed grid.

    Key insight:
    - The repeated rows and columns are only a visual stretch; compressing them first restores the true Manhattan geometry, which makes the diamond center and radius directly readable.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] > 0 and g.shape[1] > 0

    compressed = _compress_grid(g)
    out = compressed.copy()
    out[np.isin(out, [RED, GREY])] = BG

    for comp in _color_components(compressed, RED):
        cr, cc, radius = _fit_diamond(comp)
        assert radius >= 2
        _paint(out, _diamond_outline(cr, cc, radius - 1), RED)

    for comp in _color_components(compressed, GREY):
        cr, cc, radius = _fit_diamond(comp)
        _paint(out, _diamond_outline(cr, cc, radius + 1), GREY)

    return out


GEN_ATTEMPTS = 500
PLACE_TRIES = 300
GRID_MIN = 11
GRID_MAX = 15
BLUE_PROB = 0.55


def _outline_in_bounds(shape: tuple[int, int], cr: int, cc: int, radius: int) -> set[tuple[int, int]]:
    h, w = shape
    pts = _diamond_outline(cr, cc, radius)
    if any(r < 0 or r >= h or c < 0 or c >= w for r, c in pts):
        return set()
    return pts


def _halo(shape: tuple[int, int], pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in pts:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _adjacent_lines_distinct(grid: np.ndarray) -> bool:
    for i in range(1, grid.shape[0]):
        if np.array_equal(grid[i], grid[i - 1]):
            return False
    for i in range(1, grid.shape[1]):
        if np.array_equal(grid[:, i], grid[:, i - 1]):
            return False
    return True


def _choose_object_specs() -> list[dict]:
    specs = []
    colors = [RED, GREY]
    if int(np.random.randint(0, 2)) == 1:
        colors.append(int(np.random.choice([RED, GREY])))
    if np.random.random() < BLUE_PROB:
        colors.append(BLUE)

    for color in colors:
        if color == RED:
            out_radius = int(np.random.randint(1, 4))
            in_radius = out_radius + 1
        elif color == GREY:
            out_radius = int(np.random.randint(3, 6))
            in_radius = out_radius - 1
        else:
            out_radius = int(np.random.randint(2, 5))
            in_radius = out_radius
        specs.append(
            {
                "color": color,
                "in_radius": in_radius,
                "out_radius": out_radius,
                "margin": max(in_radius, out_radius),
            }
        )
    specs.sort(key=lambda spec: (-spec["margin"], spec["color"]))
    return specs


def _choose_shape(specs: list[dict]) -> tuple[int, int]:
    max_margin = max(spec["margin"] for spec in specs)
    min_dim = max(GRID_MIN, 2 * max_margin + 1)
    max_dim = max(min_dim, GRID_MAX)
    return (
        int(np.random.randint(min_dim, max_dim + 1)),
        int(np.random.randint(min_dim, max_dim + 1)),
    )


def _place_objects(shape: tuple[int, int], specs: list[dict]) -> list[dict] | None:
    h, w = shape
    forbidden: set[tuple[int, int]] = set()
    placed: list[dict] = []
    for spec in specs:
        margin = spec["margin"]
        if 2 * margin + 1 > h or 2 * margin + 1 > w:
            return None
        found = False
        for _ in range(PLACE_TRIES):
            cr = int(np.random.randint(margin, h - margin))
            cc = int(np.random.randint(margin, w - margin))
            in_pts = _outline_in_bounds(shape, cr, cc, spec["in_radius"])
            out_pts = _outline_in_bounds(shape, cr, cc, spec["out_radius"])
            footprint = in_pts | out_pts
            if not footprint or footprint & forbidden:
                continue
            found = True
            placed.append(
                {
                    "color": spec["color"],
                    "cr": cr,
                    "cc": cc,
                    "in_pts": in_pts,
                    "out_pts": out_pts,
                }
            )
            forbidden |= _halo(shape, footprint)
            break
        if not found:
            return None
    return placed


def _render_base(shape: tuple[int, int], placed: list[dict], key: str) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for obj in placed:
        _paint(grid, obj[key], obj["color"])
    return grid


def _choose_runs(length: int) -> np.ndarray | None:
    for _ in range(200):
        runs = np.random.randint(1, 4, size=length)
        if runs.sum() <= 30 and runs.max() > 1 and len(set(map(int, runs))) > 1:
            return runs.astype(int)
    return None


def _stretch(grid: np.ndarray, row_runs: np.ndarray, col_runs: np.ndarray) -> np.ndarray:
    return np.repeat(np.repeat(grid, row_runs, axis=0), col_runs, axis=1)


def _gen_attempt() -> np.ndarray | None:
    specs = _choose_object_specs()
    shape = _choose_shape(specs)
    placed = _place_objects(shape, specs)
    if placed is None:
        return None

    compressed_input = _render_base(shape, placed, "in_pts")
    compressed_output = _render_base(shape, placed, "out_pts")

    if not _adjacent_lines_distinct(compressed_input):
        return None

    row_runs = _choose_runs(shape[0])
    col_runs = _choose_runs(shape[1])
    if row_runs is None or col_runs is None:
        return None

    stretched_input = _stretch(compressed_input, row_runs, col_runs)
    if not np.array_equal(main(stretched_input), compressed_output):
        return None
    return stretched_input


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
