from __future__ import annotations

import random

import numpy as np


BG = 0
SIDES = ("up", "down", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _touches_corner(bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    h, w = shape
    return (
        (r0 == 0 and c0 == 0)
        or (r0 == 0 and c1 == w - 1)
        or (r1 == h - 1 and c0 == 0)
        or (r1 == h - 1 and c1 == w - 1)
    )


def _frame_info(
    grid: np.ndarray, color: int, bg: int
) -> tuple[tuple[int, int, int, int], tuple[int, int], tuple[str, ...]] | None:
    coords = np.argwhere(grid == color)
    if coords.size == 0:
        return None

    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5 or height % 2 == 0 or width % 2 == 0:
        return None

    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    center_local = (height // 2, width // 2)
    if int(sub[center_local]) != color:
        return None

    if not (
        int(sub[0, 0]) == color
        and int(sub[0, -1]) == color
        and int(sub[-1, 0]) == color
        and int(sub[-1, -1]) == color
    ):
        return None

    allowed = np.zeros_like(sub, dtype=bool)
    allowed[0, :] = True
    allowed[-1, :] = True
    allowed[:, 0] = True
    allowed[:, -1] = True
    allowed[center_local] = True
    if np.any((sub == color) & ~allowed):
        return None

    interior = sub[1:-1, 1:-1].copy()
    interior[center_local[0] - 1, center_local[1] - 1] = bg
    if np.any(interior != bg):
        return None

    gap_dirs: list[str] = []
    side_views = {
        "up": sub[0, 1:-1],
        "down": sub[-1, 1:-1],
        "left": sub[1:-1, 0],
        "right": sub[1:-1, -1],
    }
    for side, cells in side_views.items():
        if np.any((cells != color) & (cells != bg)):
            return None
        gap_count = int(np.count_nonzero(cells == bg))
        if gap_count > 1:
            return None
        if gap_count == 1:
            gap_dirs.append(side)

    if not gap_dirs:
        return None

    center = (r0 + center_local[0], c0 + center_local[1])
    return (r0, c0, r1, c1), center, tuple(gap_dirs)


def _grow_branches(
    out: np.ndarray,
    bbox: tuple[int, int, int, int],
    center: tuple[int, int],
    color: int,
    gap_dirs: tuple[str, ...],
) -> None:
    r0, c0, r1, c1 = bbox
    cr, cc = center
    top = r0 + 1
    bottom = r1 - 1
    left = c0 + 1
    right = c1 - 1

    if "up" in gap_dirs:
        out[top : cr + 1, cc] = color
        if cc - 1 >= left:
            out[top, cc - 1] = color
        if cc + 1 <= right:
            out[top, cc + 1] = color
    if "down" in gap_dirs:
        out[cr : bottom + 1, cc] = color
        if cc - 1 >= left:
            out[bottom, cc - 1] = color
        if cc + 1 <= right:
            out[bottom, cc + 1] = color
    if "left" in gap_dirs:
        out[cr, left : cc + 1] = color
        if cr - 1 >= top:
            out[cr - 1, left] = color
        if cr + 1 <= bottom:
            out[cr + 1, left] = color
    if "right" in gap_dirs:
        out[cr, cc : right + 1] = color
        if cr - 1 >= top:
            out[cr - 1, right] = color
        if cr + 1 <= bottom:
            out[cr + 1, right] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow T-tipped branches from each frame's central seed in the directions indicated by edge gaps.

    Input structure:
    - bg = black. The grid contains several non-touching one-pixel-thick rectangular hollow frames of distinct colors.
    - Each ordinary frame has a same-colored seed pixel at its exact center, and 1 to 4 single-cell gaps on its top, bottom, left, and/or right edges.
    - A small example legend sits in one grid corner as a pre-filled miniature frame that demonstrates the growth rule.

    Rule:
    - Ignore the corner legend and process every other colored frame independently.
    - Detect which of the four frame sides contain a gap.
    - From the central seed, draw a one-pixel-wide straight branch toward each gapped side until the branch reaches the corresponding inner wall cell.
    - At each branch endpoint, extend one pixel in the two perpendicular directions, clipped to the frame interior, so the tip becomes a T shape; leave the gap itself black.

    Key insight:
    - The legend shows the local growth pattern, but each real frame chooses its branch directions only from the set of sides whose borders contain a gap.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != bg):
        coords = np.argwhere(g == color)
        bbox = _bbox(coords)
        if _touches_corner(bbox, g.shape):
            continue
        info = _frame_info(g, color, bg)
        if info is None:
            continue
        frame_bbox, center, gap_dirs = info
        _grow_branches(out, frame_bbox, center, color, gap_dirs)

    return out


def _reserve(occupied: np.ndarray, r0: int, c0: int, h: int, w: int, pad: int = 1) -> bool:
    rr0 = max(0, r0 - pad)
    cc0 = max(0, c0 - pad)
    rr1 = min(occupied.shape[0], r0 + h + pad)
    cc1 = min(occupied.shape[1], c0 + w + pad)
    if np.any(occupied[rr0:rr1, cc0:cc1]):
        return False
    occupied[rr0:rr1, cc0:cc1] = True
    return True


def _side_gap_index(length: int, rng: random.Random) -> int:
    candidates = list(range(1, length - 1))
    center = length // 2
    off_center = [idx for idx in candidates if idx != center]
    if off_center and rng.random() < 0.7:
        return rng.choice(off_center)
    return rng.choice(candidates)


def _gap_positions_for_sides(
    height: int, width: int, sides: tuple[str, ...], rng: random.Random
) -> dict[str, int]:
    positions: dict[str, int] = {}
    for side in sides:
        if side in {"up", "down"}:
            positions[side] = _side_gap_index(width, rng)
        else:
            positions[side] = _side_gap_index(height, rng)
    return positions


def _draw_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    height: int,
    width: int,
    color: int,
    gap_positions: dict[str, int],
    *,
    prefill_growth: bool,
    gap_dirs: tuple[str, ...],
) -> None:
    r1 = r0 + height - 1
    c1 = c0 + width - 1
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color

    if "up" in gap_positions:
        grid[r0, c0 + gap_positions["up"]] = BG
    if "down" in gap_positions:
        grid[r1, c0 + gap_positions["down"]] = BG
    if "left" in gap_positions:
        grid[r0 + gap_positions["left"], c0] = BG
    if "right" in gap_positions:
        grid[r0 + gap_positions["right"], c1] = BG

    center = (r0 + height // 2, c0 + width // 2)
    grid[center] = color
    if prefill_growth:
        _grow_branches(grid, (r0, c0, r1, c1), center, color, gap_dirs)


def _corner_bbox(corner: str, height: int, width: int, grid_h: int, grid_w: int) -> tuple[int, int]:
    if corner == "tl":
        return 0, 0
    if corner == "tr":
        return 0, grid_w - width
    if corner == "bl":
        return grid_h - height, 0
    if corner == "br":
        return grid_h - height, grid_w - width
    raise ValueError(corner)


def _legend_sides(size: int, rng: random.Random) -> tuple[str, ...]:
    vertical = rng.choice([("up",), ("down",)])
    horizontal = rng.choice([("left",), ("right",)])
    base = list(vertical + horizontal)
    remaining = [side for side in SIDES if side not in base]
    extra_budget = 1 if size >= 7 and rng.random() < 0.5 else 0
    extras = rng.sample(remaining, extra_budget)
    return tuple(base + extras)


def _ordinary_specs(
    grid_h: int, grid_w: int, count: int, rng: random.Random
) -> list[tuple[int, int, tuple[str, ...]]]:
    gap_counts = rng.sample([1, 2, 3, 4], count)
    specs: list[tuple[int, int, tuple[str, ...]]] = []
    max_side = 11 if count == 2 else 9
    h_choices = [v for v in range(5, min(max_side, grid_h - 2) + 1, 2)]
    w_choices = [v for v in range(5, min(max_side, grid_w - 2) + 1, 2)]
    for gap_count in gap_counts:
        height = rng.choice(h_choices)
        width = rng.choice(w_choices)
        sides = tuple(rng.sample(SIDES, gap_count))
        specs.append((height, width, sides))
    specs.sort(key=lambda item: item[0] * item[1], reverse=True)
    return specs


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    grid_h = rng.randint(18, 30)
    grid_w = rng.randint(18, 30)
    ordinary_count = rng.randint(2, 4)
    legend_size = rng.choice([5, 7])
    legend_corner = rng.choice(["tl", "tr", "bl", "br"])

    if ordinary_count + 1 > 9:
        return None
    colors = rng.sample(list(range(1, 10)), ordinary_count + 1)
    legend_color = colors[0]
    ordinary_colors = colors[1:]

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    legend_r0, legend_c0 = _corner_bbox(legend_corner, legend_size, legend_size, grid_h, grid_w)
    if not _reserve(occupied, legend_r0, legend_c0, legend_size, legend_size):
        return None
    legend_dirs = _legend_sides(legend_size, rng)
    legend_gaps = _gap_positions_for_sides(legend_size, legend_size, legend_dirs, rng)
    _draw_frame(
        grid,
        legend_r0,
        legend_c0,
        legend_size,
        legend_size,
        legend_color,
        legend_gaps,
        prefill_growth=True,
        gap_dirs=legend_dirs,
    )

    specs = _ordinary_specs(grid_h, grid_w, ordinary_count, rng)
    for color, (height, width, gap_dirs) in zip(ordinary_colors, specs, strict=False):
        placed = False
        for _ in range(200):
            r0 = rng.randint(1, grid_h - height - 1)
            c0 = rng.randint(1, grid_w - width - 1)
            if not _reserve(occupied, r0, c0, height, width):
                continue
            gap_positions = _gap_positions_for_sides(height, width, gap_dirs, rng)
            _draw_frame(
                grid,
                r0,
                c0,
                height,
                width,
                color,
                gap_positions,
                prefill_growth=False,
                gap_dirs=gap_dirs,
            )
            placed = True
            break
        if not placed:
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(solved, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-growth input")
