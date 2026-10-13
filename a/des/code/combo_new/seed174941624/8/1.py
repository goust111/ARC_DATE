import random

import numpy as np


_DIR_STEPS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

_PALETTE = tuple(range(10))
_GRID_MIN = 12
_GRID_MAX = 30
_MAX_REGIONS = 4
_OUTER_MIN = 5
_OUTER_MAX = 10
_PLACEMENT_PAD = 1
_MAX_SAMPLE_ATTEMPTS = 400
_MAX_REGION_ATTEMPTS = 200


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in _DIR_STEPS.values():
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(cells)
    return components


def _frame_info(
    grid: np.ndarray,
    cells: list[tuple[int, int]],
    wall_color: int,
    bg: int,
) -> dict[str, object] | None:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    cell_set = set(cells)
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))

    if not cell_set <= border:
        return None
    missing = list(border - cell_set)
    if len(missing) != 1:
        return None

    gap = missing[0]
    if int(grid[gap]) != bg:
        return None

    if gap[0] == r0 and c0 < gap[1] < c1:
        side = "up"
    elif gap[0] == r1 and c0 < gap[1] < c1:
        side = "down"
    elif gap[1] == c0 and r0 < gap[0] < r1:
        side = "left"
    elif gap[1] == c1 and r0 < gap[0] < r1:
        side = "right"
    else:
        return None

    seeds: list[tuple[int, int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            color = int(grid[r, c])
            if color != bg and color != wall_color:
                seeds.append((r, c, color))
    if not seeds:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "side": side,
        "gap": gap,
        "seeds": seeds,
    }


def _regions(grid: np.ndarray) -> tuple[int, int, list[dict[str, object]]]:
    bg = _background_color(grid)
    best_color = None
    best_regions: list[dict[str, object]] = []
    best_score: tuple[int, int] | None = None

    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        regions: list[dict[str, object]] = []
        for cells in _color_components(grid, color):
            info = _frame_info(grid, cells, color, bg)
            if info is not None:
                regions.append(info)
        if not regions:
            continue
        score = (len(regions), int(np.count_nonzero(grid == color)))
        if best_score is None or score > best_score:
            best_score = score
            best_color = color
            best_regions = regions

    if best_color is None or not best_regions:
        raise AssertionError("expected at least one framed region")
    return bg, int(best_color), best_regions


def _paint_region(out: np.ndarray, region: dict[str, object]) -> None:
    r0, c0, r1, c1 = region["bbox"]  # type: ignore[index]
    side = region["side"]  # type: ignore[index]
    for r, c, color in region["seeds"]:  # type: ignore[index]
        if side == "up":
            out[r0 + 1 : r + 1, c] = color
        elif side == "down":
            out[r:r1, c] = color
        elif side == "left":
            out[r, c0 + 1 : c + 1] = color
        else:
            out[r, c:c1] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read each rectangular wall gap as a direction code, then extend the interior seed pixels into rays toward that wall side.

    Input structure:
    - The grid has a dominant background color and one wall color used by several disjoint one-pixel-thick rectangular frames.
    - Every frame is missing exactly one non-corner wall pixel on one side, leaving a single-cell gap in an otherwise closed rectangle.
    - Inside each framed region there are one to three single-pixel colored seeds on background cells.

    Rule:
    - Identify each rectangular frame and locate its unique wall gap.
    - The side containing the gap determines the direction for every seed inside that same frame: top gap means up, bottom means down, left means left, right means right.
    - For each seed, paint a straight ray of that seed's color from the seed to the corresponding interior boundary of the frame, while leaving the wall pixels and the gap cell unchanged.

    Key insight:
    - The gap is not an opening to pass through; it is a structural direction marker that tells all seeds in the same box which way to extend.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    _bg, _wall, regions = _regions(g)
    out = g.copy()
    for region in regions:
        _paint_region(out, region)
    return out


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = _PLACEMENT_PAD) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0)


def _sample_seed_positions(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    side: str,
    count: int,
) -> list[tuple[int, int]]:
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))

    if side in ("up", "down"):
        cols = random.sample(inner_cols, count)
        seeds: list[tuple[int, int]] = []
        for idx, col in enumerate(cols):
            if side == "up" and idx == 0:
                row = random.choice(list(range(r0 + 2, r1)))
            elif side == "down" and idx == 0:
                row = random.choice(list(range(r0 + 1, r1 - 1)))
            else:
                row = random.choice(inner_rows)
            seeds.append((row, col))
        return seeds

    rows = random.sample(inner_rows, count)
    seeds = []
    for idx, row in enumerate(rows):
        if side == "left" and idx == 0:
            col = random.choice(list(range(c0 + 2, c1)))
        elif side == "right" and idx == 0:
            col = random.choice(list(range(c0 + 1, c1 - 1)))
        else:
            col = random.choice(inner_cols)
        seeds.append((row, col))
    return seeds


def _sample_region(
    h: int,
    w: int,
    occupied: list[tuple[int, int, int, int]],
    wall_color: int,
    bg: int,
) -> dict[str, object] | None:
    seed_palette = [color for color in _PALETTE if color not in (bg, wall_color)]

    for _ in range(_MAX_REGION_ATTEMPTS):
        outer_h = random.randint(_OUTER_MIN, min(_OUTER_MAX, h - 2))
        outer_w = random.randint(_OUTER_MIN, min(_OUTER_MAX, w - 2))
        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        bbox = (r0, c0, r1, c1)
        if any(_boxes_touch(bbox, other) for other in occupied):
            continue

        side = random.choice(("up", "down", "left", "right"))
        interior_h = outer_h - 2
        interior_w = outer_w - 2
        axial_capacity = interior_w if side in ("up", "down") else interior_h
        seed_count = random.randint(1, min(3, axial_capacity))
        if len(seed_palette) < seed_count:
            continue

        if side == "up":
            gap = (r0, random.randint(c0 + 1, c1 - 1))
        elif side == "down":
            gap = (r1, random.randint(c0 + 1, c1 - 1))
        elif side == "left":
            gap = (random.randint(r0 + 1, r1 - 1), c0)
        else:
            gap = (random.randint(r0 + 1, r1 - 1), c1)

        seeds = _sample_seed_positions(r0, c0, r1, c1, side, seed_count)
        colors = random.sample(seed_palette, seed_count)
        return {
            "bbox": bbox,
            "side": side,
            "gap": gap,
            "seeds": [(r, c, color) for (r, c), color in zip(seeds, colors)],
        }
    return None


def _draw_region(grid: np.ndarray, region: dict[str, object], wall_color: int, bg: int) -> None:
    r0, c0, r1, c1 = region["bbox"]  # type: ignore[index]
    grid[r0, c0 : c1 + 1] = wall_color
    grid[r1, c0 : c1 + 1] = wall_color
    grid[r0 : r1 + 1, c0] = wall_color
    grid[r0 : r1 + 1, c1] = wall_color
    gap_r, gap_c = region["gap"]  # type: ignore[index]
    grid[gap_r, gap_c] = bg
    for r, c, color in region["seeds"]:  # type: ignore[index]
        grid[r, c] = color


def _render_scene(
    h: int,
    w: int,
    bg: int,
    wall_color: int,
    regions: list[dict[str, object]],
) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), bg, dtype=int)
    for region in regions:
        _draw_region(inp, region, wall_color, bg)

    out = inp.copy()
    for region in regions:
        _paint_region(out, region)
    return inp, out


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    bg_count = int(np.count_nonzero(grid == bg))
    for color in map(int, np.unique(grid)):
        if color != bg and int(np.count_nonzero(grid == color)) >= bg_count:
            return False
    return True


def _sample_scene() -> np.ndarray | None:
    h = random.randint(_GRID_MIN, _GRID_MAX)
    w = random.randint(_GRID_MIN, _GRID_MAX)
    bg, wall_color = random.sample(_PALETTE, 2)

    max_regions = min(_MAX_REGIONS, max(1, (h * w) // 120 + 1))
    target_regions = random.randint(1, max_regions)

    regions: list[dict[str, object]] = []
    occupied: list[tuple[int, int, int, int]] = []
    for _ in range(target_regions):
        region = _sample_region(h, w, occupied, wall_color, bg)
        if region is None:
            return None
        regions.append(region)
        occupied.append(region["bbox"])  # type: ignore[arg-type]

    inp, expected = _render_scene(h, w, bg, wall_color, regions)
    if not _background_is_dominant(inp, bg):
        return None
    if np.array_equal(inp, expected):
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_MAX_SAMPLE_ATTEMPTS):
        sample = _sample_scene()
        if sample is not None:
            return sample
    raise RuntimeError("failed to generate a valid input")
