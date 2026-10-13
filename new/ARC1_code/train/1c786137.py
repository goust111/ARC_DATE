import numpy as np


def _find_largest_frame_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    candidates: list[tuple[int, int, int, int, int]] = []

    for color in np.unique(g):
        coords = np.argwhere(g == color)
        rows = np.unique(coords[:, 0])
        cols = np.unique(coords[:, 1])

        for r0_index, r0 in enumerate(rows):
            for r1 in rows[r0_index + 1 :]:
                if r1 - r0 < 2:
                    continue
                for c0_index, c0 in enumerate(cols):
                    for c1 in cols[c0_index + 1 :]:
                        if c1 - c0 < 2:
                            continue
                        if (
                            np.all(g[r0, c0 : c1 + 1] == color)
                            and np.all(g[r1, c0 : c1 + 1] == color)
                            and np.all(g[r0 : r1 + 1, c0] == color)
                            and np.all(g[r0 : r1 + 1, c1] == color)
                        ):
                            area = int((r1 - r0 - 1) * (c1 - c0 - 1))
                            candidates.append((area, int(r0), int(c0), int(r1), int(c1)))

    assert candidates, "expected at least one monochrome rectangular frame"
    max_area = max(area for area, *_ in candidates)
    largest = [candidate for candidate in candidates if candidate[0] == max_area]
    assert len(largest) == 1, "expected a unique largest rectangular frame"
    _area, r0, c0, r1, c1 = largest[0]
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Crop out the interior of the largest complete monochrome rectangular frame.

    Input structure
    A noisy grid contains a closed one-cell-thick rectangular frame whose border
    is one color; the desired image is inside that frame.

    Rule
    Find every monochrome rectangular border, require a unique largest one by
    interior area, and return the subgrid strictly inside its border.

    Key insight
    The frame color is identified by the closed rectangle it forms, not by its
    numeric color id; the output is the unchanged frame interior.
    """
    g = np.asarray(grid, dtype=int)
    r0, c0, r1, c1 = _find_largest_frame_bbox(g)
    return g[r0 + 1 : r1, c0 + 1 : c1].copy()

# --------------- generator (o5x) ---------------

# Canvas/frame ranges; randint upper bounds are exclusive.
INTERIOR_HEIGHT_RANGE, INTERIOR_WIDTH_RANGE = (5, 9), (3, 11)
GRID_HEIGHT_RANGE, GRID_WIDTH_RANGE = (13, 24), (16, 22)
FRAME_THICKNESS, MIN_FRAME_MARGIN = 1, 1

# Palette, density, and retry controls.
BACKGROUND_COLOR, COLOR_RANGE = 0, (1, 10)
CONTENT_COLOR_COUNT_RANGE = (3, 5)
INTERIOR_DENSITY_RANGE, NOISE_DENSITY_RANGE = (0.66, 0.73), (0.57, 0.73)
MAX_GENERATION_ATTEMPTS = 300


def _gen_palette_fill(height: int, width: int, palette: list[int], density: float) -> np.ndarray:
    area = height * width
    nonzero = [color for color in palette if color != BACKGROUND_COLOR]
    nonzero_count = int(round(area * density))
    nonzero_count = min(area - 1, max(len(nonzero), nonzero_count))

    flat = np.zeros(area, dtype=int)
    shuffled = np.random.choice(area, size=nonzero_count, replace=False)
    np.random.shuffle(shuffled)
    split = len(nonzero)
    flat[shuffled[:split]] = nonzero

    remaining = shuffled[split:]
    if len(remaining):
        flat[remaining] = np.random.choice(nonzero, size=len(remaining))

    return flat.reshape(height, width)


def _gen_attempt() -> np.ndarray | None:
    h_int = int(np.random.randint(*INTERIOR_HEIGHT_RANGE))
    w_int = int(np.random.randint(*INTERIOR_WIDTH_RANGE))

    color_count = int(np.random.randint(*CONTENT_COLOR_COUNT_RANGE))
    nonzero_colors = np.random.choice(np.arange(*COLOR_RANGE), size=color_count, replace=False)
    content_palette = [BACKGROUND_COLOR] + [int(color) for color in nonzero_colors]
    frame_color = int(np.random.choice([color for color in range(*COLOR_RANGE) if color not in content_palette]))

    interior_density = float(np.random.uniform(*INTERIOR_DENSITY_RANGE))
    interior = _gen_palette_fill(h_int, w_int, content_palette, interior_density)

    h_outer = h_int + 2 * FRAME_THICKNESS
    w_outer = w_int + 2 * FRAME_THICKNESS
    min_h_grid = max(GRID_HEIGHT_RANGE[0], h_outer + 2 * MIN_FRAME_MARGIN)
    min_w_grid = max(GRID_WIDTH_RANGE[0], w_outer + 2 * MIN_FRAME_MARGIN)
    h_grid = int(np.random.randint(min_h_grid, GRID_HEIGHT_RANGE[1]))
    w_grid = int(np.random.randint(min_w_grid, GRID_WIDTH_RANGE[1]))

    r0 = int(np.random.randint(MIN_FRAME_MARGIN, h_grid - h_outer))
    c0 = int(np.random.randint(MIN_FRAME_MARGIN, w_grid - w_outer))

    outside_mask = np.ones((h_grid, w_grid), dtype=bool)
    outside_mask[r0 : r0 + h_outer, c0 : c0 + w_outer] = False
    outside_count = int(outside_mask.sum())
    noise_density = float(np.random.uniform(*NOISE_DENSITY_RANGE))
    noise_area = _gen_palette_fill(1, outside_count, content_palette, noise_density).reshape(-1)

    grid = np.empty((h_grid, w_grid), dtype=int)
    grid[outside_mask] = noise_area
    grid[r0 : r0 + h_outer, c0 : c0 + w_outer] = frame_color
    grid[r0 + 1 : r0 + 1 + h_int, c0 + 1 : c0 + 1 + w_int] = interior

    try:
        if np.array_equal(main(grid), interior):
            return grid
    except AssertionError:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for 1c786137")
