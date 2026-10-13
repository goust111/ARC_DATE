from __future__ import annotations

from collections import Counter
import random

import numpy as np


PALETTE = tuple(range(10))
DIR_VECS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
GEN_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 160
GRID_MIN = 12
GRID_MAX = 30
INNER_MIN = 1
INNER_MAX = 5


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _perimeter_coords(r0: int, c0: int, r1: int, c1: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(c0, c1 + 1):
        coords.append((r0, c))
    for r in range(r0 + 1, r1):
        coords.append((r, c1))
    if r1 > r0:
        for c in range(c1, c0 - 1, -1):
            coords.append((r1, c))
    if c1 > c0:
        for r in range(r1 - 1, r0, -1):
            coords.append((r, c0))
    return coords


def _gap_direction(
    gap: tuple[int, int], bbox_: tuple[int, int, int, int]
) -> tuple[int, int] | None:
    r, c = gap
    r0, c0, r1, c1 = bbox_
    if r == r0 and c0 < c < c1:
        return DIR_VECS["up"]
    if r == r1 and c0 < c < c1:
        return DIR_VECS["down"]
    if c == c0 and r0 < r < r1:
        return DIR_VECS["left"]
    if c == c1 and r0 < r < r1:
        return DIR_VECS["right"]
    return None


def _ray_cells(
    gap: tuple[int, int], direction: tuple[int, int], length: int, h: int, w: int
) -> list[tuple[int, int]]:
    dr, dc = direction
    cells: list[tuple[int, int]] = []
    r, c = gap
    for step in range(1, length + 1):
        nr = r + dr * step
        nc = c + dc * step
        if not (0 <= nr < h and 0 <= nc < w):
            break
        cells.append((nr, nc))
    return cells


def _detect_rectangles(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    counts = Counter(map(int, g.flatten()))
    border_candidates = [color for color, _ in counts.most_common() if color != bg]
    assert border_candidates, "expected a non-background border color"
    border = int(border_candidates[0])

    rectangles: list[dict[str, object]] = []
    for comp in _components(g == border):
        if len(comp) < 4:
            continue
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue

        perim = _perimeter_coords(r0, c0, r1, c1)
        if len(comp) != len(perim) - 2:
            continue
        if any(not (r in (r0, r1) or c in (c0, c1)) for r, c in comp):
            continue

        gap_cells = [(r, c) for r, c in perim if int(g[r, c]) == bg]
        marker_cells = [(r, c) for r, c in perim if int(g[r, c]) not in (bg, border)]
        if len(gap_cells) != 1 or len(marker_cells) != 1:
            continue

        gap = gap_cells[0]
        marker = marker_cells[0]
        direction = _gap_direction(gap, (r0, c0, r1, c1))
        if direction is None:
            continue
        if abs(gap[0] - marker[0]) + abs(gap[1] - marker[1]) != 1:
            continue
        if direction[0] != 0 and gap[0] != marker[0]:
            continue
        if direction[1] != 0 and gap[1] != marker[1]:
            continue

        seed_color = int(g[marker])
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        if interior.size == 0:
            continue
        seed_positions = np.argwhere(interior == seed_color)
        if seed_positions.shape[0] != 1:
            continue
        if np.any((interior != bg) & (interior != seed_color)):
            continue

        area = int(interior.shape[0] * interior.shape[1])
        rectangles.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap": gap,
                "direction": direction,
                "seed_color": seed_color,
                "area": area,
            }
        )

    rectangles.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    assert rectangles, "expected at least one rectangle"
    return rectangles


def main(grid: np.ndarray) -> np.ndarray:
    """
    Draw a seed-colored outward ray from each framed rectangle's gap, with length equal to the rectangle's interior area.

    Input structure:
    - The grid has a dominant background color plus several axis-aligned one-cell-thick rectangular frames in one shared border color.
    - Each frame has exactly one background-colored gap on one side and exactly one border-adjacent marker pixel recolored to the unique seed color used inside that frame.
    - Inside each rectangle, all cells are background except for one seed pixel whose color matches the border marker.

    Rule:
    - For each rectangle, locate the unique interior seed color and the matching marker pixel on the frame perimeter.
    - Use the adjacent background gap on that same side to determine the outward direction.
    - Compute the rectangle's interior area as interior_height × interior_width, then draw a ray of that many cells starting one step outside the gap and extending outward, truncating at the grid edge.
    - Leave the original frame, gap, and seed pixel unchanged; only add the ray cells.

    Key insight:
    - The repeated seed color links two independent pieces of information: it identifies which perimeter marker belongs to which interior seed, while the gap beside that marker gives direction and the rectangle's enclosed area gives length.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for rect in _detect_rectangles(g):
        gap = rect["gap"]  # type: ignore[assignment]
        direction = rect["direction"]  # type: ignore[assignment]
        length = int(rect["area"])
        color = int(rect["seed_color"])
        for r, c in _ray_cells(gap, direction, length, h, w):
            out[r, c] = color
    return out


def _position_bounds(
    h: int,
    w: int,
    box_h: int,
    box_w: int,
    area: int,
    direction_name: str,
    truncated: bool,
) -> tuple[int, int, int, int] | None:
    row_lo, row_hi = 1, h - box_h - 1
    col_lo, col_hi = 1, w - box_w - 1
    if row_lo > row_hi or col_lo > col_hi:
        return None

    if direction_name == "up":
        if truncated:
            row_hi = min(row_hi, area - 1)
        else:
            row_lo = max(row_lo, area)
    elif direction_name == "down":
        if truncated:
            row_lo = max(row_lo, h - box_h - (area - 1))
        else:
            row_hi = min(row_hi, h - box_h - area)
    elif direction_name == "left":
        if truncated:
            col_hi = min(col_hi, area - 1)
        else:
            col_lo = max(col_lo, area)
    else:
        assert direction_name == "right"
        if truncated:
            col_lo = max(col_lo, w - box_w - (area - 1))
        else:
            col_hi = min(col_hi, w - box_w - area)

    if row_lo > row_hi or col_lo > col_hi:
        return None
    return row_lo, row_hi, col_lo, col_hi


def _reserve_bbox(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = reserved.shape
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            reserved[r, c] = True


def _reserve_ray(reserved: np.ndarray, ray: list[tuple[int, int]]) -> None:
    h, w = reserved.shape
    for r, c in ray:
        reserved[r, c] = True
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w:
                reserved[nr, nc] = True


def _try_place_rectangle(
    grid: np.ndarray,
    reserved: np.ndarray,
    rng: random.Random,
    border_color: int,
    seed_color: int,
    direction_name: str,
    truncated: bool,
) -> dict[str, object] | None:
    h, w = grid.shape
    direction = DIR_VECS[direction_name]

    for _ in range(PLACEMENT_ATTEMPTS):
        inner_h = rng.randint(INNER_MIN, INNER_MAX)
        inner_w = rng.randint(INNER_MIN, INNER_MAX)
        box_h = inner_h + 2
        box_w = inner_w + 2
        area = inner_h * inner_w

        bounds = _position_bounds(h, w, box_h, box_w, area, direction_name, truncated)
        if bounds is None:
            continue
        row_lo, row_hi, col_lo, col_hi = bounds
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        r1 = r0 + box_h - 1
        c1 = c0 + box_w - 1

        if direction_name in ("up", "down"):
            gap_offset = rng.randint(1, box_w - 2)
            gap = (r0, c0 + gap_offset) if direction_name == "up" else (r1, c0 + gap_offset)
            marker_options = []
            if gap_offset - 1 >= 0:
                marker_options.append(-1)
            if gap_offset + 1 <= box_w - 1:
                marker_options.append(1)
            marker_delta = rng.choice(marker_options)
            marker = (gap[0], gap[1] + marker_delta)
        else:
            gap_offset = rng.randint(1, box_h - 2)
            gap = (r0 + gap_offset, c0) if direction_name == "left" else (r0 + gap_offset, c1)
            marker_options = []
            if gap_offset - 1 >= 0:
                marker_options.append(-1)
            if gap_offset + 1 <= box_h - 1:
                marker_options.append(1)
            marker_delta = rng.choice(marker_options)
            marker = (gap[0] + marker_delta, gap[1])

        seed = (rng.randint(r0 + 1, r1 - 1), rng.randint(c0 + 1, c1 - 1))
        ray = _ray_cells(gap, direction, area, h, w)
        if not ray:
            continue

        footprint = {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}
        footprint.update(ray)
        if any(reserved[r, c] for r, c in footprint):
            continue

        for r, c in _perimeter_coords(r0, c0, r1, c1):
            if (r, c) == gap:
                continue
            if (r, c) == marker:
                grid[r, c] = seed_color
            else:
                grid[r, c] = border_color
        grid[seed] = seed_color

        _reserve_bbox(reserved, r0, c0, r1, c1)
        _reserve_ray(reserved, ray)
        return {
            "bbox": (r0, c0, r1, c1),
            "gap": gap,
            "direction": direction,
            "seed_color": seed_color,
            "area": area,
            "ray": ray,
        }

    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(GRID_MIN, GRID_MAX)
    w = rng.randint(GRID_MIN, GRID_MAX)
    max_objects = 4 if min(h, w) >= 18 else 3
    object_count = rng.randint(1, max_objects)

    bg = rng.choice(PALETTE)
    border_choices = [color for color in PALETTE if color != bg]
    border = rng.choice(border_choices)
    seed_pool = [color for color in PALETTE if color not in (bg, border)]
    rng.shuffle(seed_pool)
    seed_colors = seed_pool[:object_count]

    grid = np.full((h, w), bg, dtype=int)
    reserved = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []

    directions = [rng.choice(tuple(DIR_VECS)) for _ in range(object_count)]
    truncations = [rng.choice([False, True]) for _ in range(object_count)]
    if object_count >= 2 and len(set(directions)) == 1:
        directions[1] = rng.choice([name for name in DIR_VECS if name != directions[0]])

    for seed_color, direction_name, truncated in zip(seed_colors, directions, truncations):
        spec = _try_place_rectangle(
            grid,
            reserved,
            rng,
            border,
            seed_color,
            direction_name,
            truncated,
        )
        if spec is None:
            return None
        placed.append(spec)

    if _most_common_color(grid) != bg:
        return None
    detected = _detect_rectangles(grid)
    if len(detected) != len(placed):
        return None

    expected = grid.copy()
    for spec in placed:
        color = int(spec["seed_color"])
        for r, c in spec["ray"]:  # type: ignore[index]
            expected[r, c] = color
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
