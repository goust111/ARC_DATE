import numpy as np
from collections import Counter
from scipy import ndimage


STRUCT8 = np.ones((3, 3), dtype=bool)



def _background_color(grid: np.ndarray) -> int:
    """Return the most common color in grid."""
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _component_sizes(mask: np.ndarray) -> np.ndarray:
    """Return per-component sizes for a boolean mask using 8-connectivity."""
    if not np.any(mask):
        return np.zeros(0, dtype=int)
    labeled, n = ndimage.label(mask.astype(np.int8), structure=STRUCT8)
    if n == 0:
        return np.zeros(0, dtype=int)
    counts = np.bincount(labeled.ravel())
    return counts[1:].astype(int)


def _draw_ring(board: np.ndarray, color: int, side_len: int, dashed: bool) -> None:
    """Draw a centered square ring with side length `side_len` into `board`."""
    assert side_len % 2 == 1 and side_len >= 3
    center = board.shape[0] // 2
    radius = side_len // 2
    lo = center - radius
    hi = center + radius
    parity = (lo + lo) & 1

    for c in range(lo, hi + 1):
        if not dashed or ((lo + c) & 1) == parity:
            board[lo, c] = color
            board[hi, c] = color

    for r in range(lo + 1, hi):
        if not dashed or ((r + lo) & 1) == parity:
            board[r, lo] = color
            board[r, hi] = color


def main(grid: np.ndarray) -> np.ndarray:
    """Convert each foreground color into a concentric square ring centered at the canvas origin.

    Input structure:
    - The background color is the most common color in the grid.
    - Each non-background color forms several 8-connected monochrome foreground objects.
    - Each object color has a distinct number of objects, and objects of the same color do not touch.

    Rule:
    - For each foreground color, count its 8-connected object count n.
    - Let N be the largest such count; output a square of size (2N+1)x(2N+1), filled with background.
    - Draw color-specific concentric square frames with side length 2n+1, centered in the output.
    - If all objects of that color are singleton pixels, draw a dashed ring by filling only border cells whose parity matches the corners.
    - Otherwise draw a solid ring.

    Key insight:
    - Output is a concentric summary of each color, where object count determines ring radius and singleton-only color decides dashed vs solid texture.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    specs = []  # (count, dashed, color)
    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        sizes = _component_sizes(g == color)
        if sizes.size == 0:
            continue
        count = int(sizes.size)
        dashed = bool(np.all(sizes == 1))
        specs.append((count, dashed, color))

    assert specs, "expected at least one foreground color"
    counts = [count for count, _, _ in specs]
    assert len(set(counts)) == len(counts), "foreground object counts must be distinct"

    max_count = int(max(counts))
    out_size = 2 * max_count + 1
    out = np.full((out_size, out_size), bg, dtype=int)

    for count, dashed, color in sorted(specs, key=lambda x: -x[0]):
        side_len = 2 * count + 1
        _draw_ring(out, color, side_len, dashed)

    return out


def _random_connected_shape(rng: np.random.Generator, area: int) -> np.ndarray:
    """Create a random connected 8-neighborhood mask with exactly `area` pixels."""
    assert area >= 1
    if area == 1:
        return np.ones((1, 1), dtype=bool)

    cells = [(0, 0)]
    cell_set = {(0, 0)}
    neighbors = [
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    ]

    while len(cells) < area:
        r, c = cells[int(rng.integers(0, len(cells)))]
        dr, dc = neighbors[int(rng.integers(0, len(neighbors)))]
        nr, nc = r + dr, c + dc
        if (nr, nc) in cell_set:
            continue
        cell_set.add((nr, nc))
        cells.append((nr, nc))

    arr = np.array(cells, dtype=int)
    arr[:, 0] -= int(arr[:, 0].min())
    arr[:, 1] -= int(arr[:, 1].min())
    h = int(arr[:, 0].max()) + 1
    w = int(arr[:, 1].max()) + 1
    mask = np.zeros((h, w), dtype=bool)
    mask[arr[:, 0], arr[:, 1]] = True
    return mask


def _shape_area_for_count(rng: np.random.Generator, n_objects: int, all_singleton: bool) -> list[int]:
    if all_singleton:
        return [1 for _ in range(n_objects)]

    # Keep objects small to satisfy ARC canvas limits while still allowing rich motifs.
    base = int(rng.integers(2, 6))
    areas = [int(rng.integers(2, base + 1)) for _ in range(n_objects)]
    # At least one object should be visibly non-singleton for non-singleton colors.
    if max(areas) == 1:
        areas[0] = int(rng.integers(2, base + 1))
    return areas


def _place_objects(
    rng: np.random.Generator,
    h: int,
    w: int,
    color_specs: dict[int, list[np.ndarray]],
    bg: int,
) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)
    blocked = np.zeros((h, w), dtype=bool)

    order = []
    for color, shapes in color_specs.items():
        for shape in shapes:
            order.append((color, shape))
    rng.shuffle(order)

    for color, shape in order:
        shape_cells = np.argwhere(shape)
        hh, ww = shape.shape
        rr = shape_cells[:, 0]
        cc = shape_cells[:, 1]
        placed = False

        for _ in range(400):
            r0 = int(rng.integers(0, h - hh + 1))
            c0 = int(rng.integers(0, w - ww + 1))

            ar = rr + r0
            ac = cc + c0
            if np.any(blocked[ar, ac]):
                continue

            # Place this object.
            grid[ar, ac] = color
            occupied[ar, ac] = True
            blocked[ar, ac] = True

            # Expand blocked by one cell (8-neighbor), preventing touching between objects.
            for r, c in zip(ar, ac):
                blocked[max(r - 1, 0) : min(h, r + 2), max(c - 1, 0) : min(w, c + 2)] = True

            placed = True
            break

        if not placed:
            return None

    return grid


def _analyze_colors(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    stats = {}
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        stats[color] = _component_sizes(grid == color)
    return stats


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(300):
        bg = int(rng.integers(0, 10))
        available = [c for c in range(10) if c != bg]
        max_count = int(rng.integers(4, 15))  # implies output size <=29

        n_colors = int(rng.integers(2, min(6, max_count + 1)))
        if n_colors > len(available):
            n_colors = len(available)
        counts = sorted([int(x) for x in rng.choice(np.arange(1, max_count + 1), size=n_colors, replace=False)])

        colors = rng.choice(available, size=n_colors, replace=False).astype(int).tolist()
        all_singleton = [bool(rng.random() < 0.5) for _ in colors]
        if all(all_singleton):
            all_singleton[int(rng.integers(0, n_colors))] = False
        if not any(all_singleton):
            all_singleton[int(rng.integers(0, n_colors))] = True

        n_cells = 0
        color_specs: dict[int, list[np.ndarray]] = {}
        for color, count in zip(colors, counts):
            all_single = all_singleton[colors.index(color)]
            areas = _shape_area_for_count(rng, count, all_single)
            n_cells += sum(areas)
            color_specs[color] = [_random_connected_shape(rng, area) for area in areas]

        # Choose canvas large enough for component packing with spacing.
        total_objects = sum(counts)
        min_dim = max(12, int(np.ceil(np.sqrt(total_objects * 5 + n_cells)) + 2))
        h = int(rng.integers(min_dim, 31))
        w = int(rng.integers(min_dim, 31))

        grid = _place_objects(rng, h, w, color_specs, bg)
        if grid is None:
            continue

        stats = _analyze_colors(grid, bg)
        if len(stats) != n_colors:
            continue

        valid = True
        for color, expected in zip(colors, counts):
            sizes = stats.get(color, np.zeros(0, dtype=int))
            if sizes.size != expected:
                valid = False
                break
            if all_singleton[colors.index(color)] and not np.all(sizes == 1):
                valid = False
                break
            if not all_singleton[colors.index(color)] and np.all(sizes == 1):
                valid = False
                break

        if not valid:
            continue

        out = main(grid)
        if out.ndim != 2:
            continue
        if max(out.shape) > 30:
            continue
        if out.shape[0] != 2 * max(counts) + 1 or out.shape[1] != 2 * max(counts) + 1:
            continue

        return grid

    raise RuntimeError("failed to generate a valid input")
