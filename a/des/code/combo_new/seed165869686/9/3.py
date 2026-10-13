import random
from collections import deque

import numpy as np


BG_COLOR = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7

GENERATION_RETRIES = 300
PLACEMENT_RETRIES = 200
PATTERN_RETRIES = 200

GRID_HEIGHT_RANGE = (20, 30)
GRID_WIDTH_RANGE = (20, 30)
REGION_COUNT_RANGE = (3, 4)
LEGEND_GAP = 1

LEGEND_PATTERNS = {
    "lr": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 1],
        ],
        dtype=bool,
    ),
    "ud": np.array(
        [
            [1, 0, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    "both": np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
}

LEGEND_COLOR_BY_SYMMETRY = {"lr": RED, "ud": BLUE, "both": GREEN}


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    height, width = grid.shape
    mask = grid == color
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        seen[r, c] = True
        queue = deque([(r, c)])
        component: list[tuple[int, int]] = []
        while queue:
            cr, cc = queue.popleft()
            component.append((cr, cc))
            for nr, nc in _neighbors4(cr, cc, height, width):
                if mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
        components.append(component)

    return components


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _classify_symmetry(mask: np.ndarray) -> str | None:
    if not np.any(mask):
        return None
    mirror_lr = np.array_equal(mask, np.fliplr(mask))
    mirror_ud = np.array_equal(mask, np.flipud(mask))
    if mirror_lr and mirror_ud:
        return "both"
    if mirror_lr:
        return "lr"
    if mirror_ud:
        return "ud"
    return None


def _find_legend_mapping(grid: np.ndarray, bg: int) -> tuple[dict[str, int], set[tuple[int, int]]]:
    height, _ = grid.shape
    mapping: dict[str, int] = {}
    legend_cells: set[tuple[int, int]] = set()

    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for component in _components_of_color(grid, color):
            r0, c0, r1, c1 = _bbox(component)
            if r1 != height - 1 or r1 - r0 != 2 or c1 - c0 != 2:
                continue
            patch = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            if int(patch.sum()) != len(component):
                continue
            symmetry = _classify_symmetry(patch)
            if symmetry is None:
                continue
            assert symmetry not in mapping, "legend must have one tile per symmetry type"
            mapping[symmetry] = color
            legend_cells.update(component)

    assert set(mapping) == {"lr", "ud", "both"}, "bottom legend must encode all three symmetry types"
    return mapping, legend_cells


def _is_frame_component(
    grid: np.ndarray, color: int, component: list[tuple[int, int]], bg: int
) -> bool:
    r0, c0, r1, c1 = _bbox(component)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return False
    if len(component) != 2 * (height + width) - 4:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not (
        np.all(patch[0, :] == color)
        and np.all(patch[-1, :] == color)
        and np.all(patch[:, 0] == color)
        and np.all(patch[:, -1] == color)
    ):
        return False
    if height > 2 and width > 2 and np.any(patch[1:-1, 1:-1] == color):
        return False
    if height > 2 and width > 2 and not np.any(patch[1:-1, 1:-1] != bg):
        return False
    return True


def _strictly_inside(
    inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]
) -> bool:
    return inner[0] > outer[0] and inner[1] > outer[1] and inner[2] < outer[2] and inner[3] < outer[3]


def _find_frames(grid: np.ndarray, bg: int, legend_colors: set[int]) -> list[tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, int, int]] = []

    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in legend_colors and int(v) != bg):
        for component in _components_of_color(grid, color):
            if _is_frame_component(grid, color, component, bg):
                candidates.append(_bbox(component))

    frames: list[tuple[int, int, int, int]] = []
    for bbox in candidates:
        if any(_strictly_inside(bbox, other) for other in candidates if other != bbox):
            continue
        frames.append(bbox)

    frames.sort()
    assert frames, "expected at least one enclosed rectangular region"
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect each enclosed region's symmetry type and recolor its border with the matching legend color.

    Input structure:
    - The grid has a dominant background color.
    - The bottom three rows contain a three-tile legend: each 3x3 monochrome tile visually demonstrates one symmetry class.
    - Above the legend sit one-cell-thick rectangular frames; each frame encloses an orange pattern on the background.

    Rule:
    - Read the three legend tiles by their visible symmetry: left-right mirror only, up-down mirror only, or both.
    - For every framed region, inspect the full interior pattern and determine whether it mirrors only left-right, only up-down, or both.
    - Recolor that frame's border to the color of the legend tile with the same symmetry type, leaving the interior pattern and background unchanged.

    Key insight:
    - The bottom examples are not labels by position; their own shapes encode the three symmetry classes that must be matched against the framed patterns above.
    """
    g = np.asarray(grid, dtype=int)
    bg = _majority_color(g)
    legend_map, _ = _find_legend_mapping(g, bg)
    frames = _find_frames(g, bg, set(legend_map.values()))

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        interior_mask = g[r0 + 1 : r1, c0 + 1 : c1] != bg
        symmetry = _classify_symmetry(interior_mask)
        assert symmetry is not None, "every region pattern must have one supported symmetry type"
        color = legend_map[symmetry]
        out[r0, c0 : c1 + 1] = color
        out[r1, c0 : c1 + 1] = color
        out[r0 : r1 + 1, c0] = color
        out[r0 : r1 + 1, c1] = color

    return out


def _build_lr_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    half_width = (width + 1) // 2
    for _ in range(PATTERN_RETRIES):
        density = rng.uniform(0.25, 0.6)
        left = np.array(
            [[rng.random() < density for _ in range(half_width)] for _ in range(height)],
            dtype=bool,
        )
        if width % 2 == 0:
            mask = np.concatenate([left, left[:, ::-1]], axis=1)
        else:
            mask = np.concatenate([left, left[:, :-1][:, ::-1]], axis=1)
        if 0 < int(mask.sum()) < mask.size and _classify_symmetry(mask) == "lr":
            return mask
    raise RuntimeError("failed to generate lr-only pattern")


def _build_ud_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    half_height = (height + 1) // 2
    for _ in range(PATTERN_RETRIES):
        density = rng.uniform(0.25, 0.6)
        top = np.array(
            [[rng.random() < density for _ in range(width)] for _ in range(half_height)],
            dtype=bool,
        )
        if height % 2 == 0:
            mask = np.concatenate([top, top[::-1, :]], axis=0)
        else:
            mask = np.concatenate([top, top[:-1, :][::-1, :]], axis=0)
        if 0 < int(mask.sum()) < mask.size and _classify_symmetry(mask) == "ud":
            return mask
    raise RuntimeError("failed to generate ud-only pattern")


def _build_both_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    quarter_height = (height + 1) // 2
    quarter_width = (width + 1) // 2
    for _ in range(PATTERN_RETRIES):
        density = rng.uniform(0.2, 0.55)
        corner = np.array(
            [[rng.random() < density for _ in range(quarter_width)] for _ in range(quarter_height)],
            dtype=bool,
        )
        if width % 2 == 0:
            top = np.concatenate([corner, corner[:, ::-1]], axis=1)
        else:
            top = np.concatenate([corner, corner[:, :-1][:, ::-1]], axis=1)
        if height % 2 == 0:
            mask = np.concatenate([top, top[::-1, :]], axis=0)
        else:
            mask = np.concatenate([top, top[:-1, :][::-1, :]], axis=0)
        if 0 < int(mask.sum()) < mask.size and _classify_symmetry(mask) == "both":
            return mask
    raise RuntimeError("failed to generate both-axes pattern")


def _build_pattern(height: int, width: int, symmetry: str, rng: random.Random) -> np.ndarray:
    if symmetry == "lr":
        return _build_lr_mask(height, width, rng)
    if symmetry == "ud":
        return _build_ud_mask(height, width, rng)
    if symmetry == "both":
        return _build_both_mask(height, width, rng)
    raise ValueError(f"unknown symmetry type: {symmetry}")


def _draw_region(
    grid: np.ndarray, top: int, left: int, interior_mask: np.ndarray
) -> None:
    height, width = interior_mask.shape
    bottom = top + height + 1
    right = left + width + 1
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY
    interior = grid[top + 1 : bottom, left + 1 : right]
    interior[interior_mask] = ORANGE


def _draw_legend(grid: np.ndarray, start_col: int) -> None:
    legend_row = grid.shape[0] - 3
    order = ["lr", "ud", "both"]
    for index, symmetry in enumerate(order):
        color = LEGEND_COLOR_BY_SYMMETRY[symmetry]
        patch = LEGEND_PATTERNS[symmetry]
        col = start_col + index * 4
        grid[legend_row : legend_row + 3, col : col + 3][patch] = color


def _choose_region_specs(rng: random.Random) -> list[tuple[int, int, str]]:
    region_count = rng.randint(*REGION_COUNT_RANGE)
    symmetries = ["lr", "ud", "both"]
    if region_count == 4:
        symmetries.append(rng.choice(symmetries))
    rng.shuffle(symmetries)

    specs: list[tuple[int, int, str]] = []
    for symmetry in symmetries:
        if region_count == 4:
            height = rng.randint(4, 6)
            width = rng.randint(4, 6)
        else:
            height = rng.randint(4, 7)
            width = rng.randint(4, 7)
        specs.append((height, width, symmetry))
    return specs


def _place_regions(
    grid: np.ndarray, specs: list[tuple[int, int, str]], rng: random.Random
) -> bool:
    height, width = grid.shape
    legend_top = height - 3
    top_limit = legend_top - LEGEND_GAP
    occupied = np.zeros_like(grid, dtype=bool)
    occupied[legend_top:, :] = True
    occupied[top_limit:legend_top, :] = True

    for interior_height, interior_width, symmetry in specs:
        outer_height = interior_height + 2
        outer_width = interior_width + 2
        placed = False
        if outer_height > top_limit or outer_width > width:
            return False

        for _ in range(PLACEMENT_RETRIES):
            r0 = rng.randint(0, top_limit - outer_height)
            c0 = rng.randint(0, width - outer_width)
            r1 = r0 + outer_height - 1
            c1 = c0 + outer_width - 1
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(height - 1, r1 + 1)
            cc1 = min(width - 1, c1 + 1)
            if occupied[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            pattern = _build_pattern(interior_height, interior_width, symmetry, rng)
            _draw_region(grid, r0, c0, pattern)
            occupied[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break

        if not placed:
            return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(*GRID_HEIGHT_RANGE)
    width = rng.randint(*GRID_WIDTH_RANGE)
    if width < 11 or height < 8:
        return None

    grid = np.full((height, width), BG_COLOR, dtype=int)
    legend_start = rng.randint(0, width - 11)
    _draw_legend(grid, legend_start)

    specs = _choose_region_specs(rng)
    if not _place_regions(grid, specs, rng):
        return None

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
