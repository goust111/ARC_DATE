from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import Counter

import numpy as np


BLACK = 0
YELLOW = 4
ARM_COLORS = (1, 2, 3, 5, 6, 7, 8, 9)
PLUS_OFFSETS = ((0, 0), (-1, 0), (1, 0), (0, -1), (0, 1))
CARDINALS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_MIN = 13
GRID_MAX = 18
MARKER_COUNT_RANGE = (4, 6)
TARGET_COUNT_OPTIONS = (2, 3, 4)
NOISE_DENSITY_RANGE = (0.10, 0.15)
MAX_GENERATION_ATTEMPTS = 600

_SEEN_OUTPUT_SIGNATURES: set[str] = set()


def _neighbors4(r: int, c: int):
    for dr, dc in CARDINALS:
        yield r + dr, c + dc


def _plus_cells(center: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = center
    return [(r + dr, c + dc) for dr, dc in PLUS_OFFSETS]


def _quadrant_of_cell(r: int, c: int, h: int, w: int) -> int:
    top_end = h // 2
    left_end = w // 2
    bottom_start = top_end + (h % 2)
    right_start = left_end + (w % 2)

    if r < top_end:
        row_part = 0
    elif r >= bottom_start:
        row_part = 1
    else:
        return 0

    if c < left_end:
        col_part = 0
    elif c >= right_start:
        col_part = 1
    else:
        return 0

    return 1 + row_part * 2 + col_part


def _quadrant_slices(h: int, w: int, quadrant: int) -> tuple[slice, slice]:
    top_end = h // 2
    left_end = w // 2
    bottom_start = top_end + (h % 2)
    right_start = left_end + (w % 2)

    if quadrant == 1:
        return slice(0, top_end), slice(0, left_end)
    if quadrant == 2:
        return slice(0, top_end), slice(right_start, w)
    if quadrant == 3:
        return slice(bottom_start, h), slice(0, left_end)
    if quadrant == 4:
        return slice(bottom_start, h), slice(right_start, w)
    raise ValueError(f"invalid quadrant: {quadrant}")


def _detect_plus_markers(grid: np.ndarray) -> list[dict[str, int | tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    markers: list[dict[str, int | tuple[int, int]]] = []
    for r, c in np.argwhere(g == YELLOW):
        r = int(r)
        c = int(c)
        if r == 0 or r == h - 1 or c == 0 or c == w - 1:
            continue
        arms = [int(g[r - 1, c]), int(g[r + 1, c]), int(g[r, c - 1]), int(g[r, c + 1])]
        if len(set(arms)) != 1:
            continue
        arm_color = arms[0]
        if arm_color in (BLACK, YELLOW):
            continue
        markers.append({"center": (r, c), "arm_color": arm_color})
    return markers


def _render_output(h: int, w: int, target_color: int, target_count: int) -> np.ndarray:
    out = np.zeros((h, w), dtype=int)
    rs, cs = _quadrant_slices(h, w, target_count)
    out[rs, cs] = target_color
    return out


def _grid_signature(grid: np.ndarray) -> str:
    return hashlib.md5(grid.tobytes() + str(grid.shape).encode()).hexdigest()


def main(grid: np.ndarray) -> np.ndarray:
    """Fill the quadrant indexed by the repeated yellow-centered plus-marker count with that arm color.

    Input structure:
    - The grid has a black background with scattered non-yellow noise pixels.
    - It contains 4 to 6 plus markers made of a yellow center pixel and four orthogonally adjacent arm pixels of one shared non-black, non-yellow color.
    - Exactly one arm color appears on more than one detected plus marker.

    Rule:
    - Detect every valid yellow-centered plus marker.
    - Count how many markers use each arm color; the unique repeated arm color is the target color.
    - Let N be the number of markers using that target color, clear the whole grid to black, and split the grid by its row and column midpoints.
    - Fill quadrant N (1=top-left, 2=top-right, 3=bottom-left, 4=bottom-right) with the target color; if a dimension is odd, its middle row or column stays black.

    Key insight:
    - Repetition of an arm color determines the fill color, while the number of markers carrying that repeated color determines which quadrant to paint.
    """
    g = np.asarray(grid, dtype=int)
    markers = _detect_plus_markers(g)
    assert markers, "expected at least one yellow-centered plus marker"

    arm_counts = Counter(int(marker["arm_color"]) for marker in markers)
    repeated = [color for color, count in arm_counts.items() if count > 1]
    assert len(repeated) == 1, "expected exactly one repeated arm color"
    target_color = int(repeated[0])
    target_count = int(arm_counts[target_color])
    assert 1 <= target_count <= 4, "target count must index one of four quadrants"

    return _render_output(g.shape[0], g.shape[1], target_color, target_count)


def _candidate_centers_by_quadrant(h: int, w: int) -> dict[int, list[tuple[int, int]]]:
    centers = {1: [], 2: [], 3: [], 4: []}
    mid_r = h // 2
    mid_c = w // 2
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if h % 2 == 1 and abs(r - mid_r) <= 1:
                continue
            if w % 2 == 1 and abs(c - mid_c) <= 1:
                continue
            quadrant = _quadrant_of_cell(r, c, h, w)
            if quadrant:
                centers[quadrant].append((r, c))
    return centers


def _can_place_marker(center: tuple[int, int], occupied: set[tuple[int, int]], shape: tuple[int, int]) -> bool:
    h, w = shape
    for r, c in _plus_cells(center):
        if not (0 <= r < h and 0 <= c < w):
            return False
        if (r, c) in occupied:
            return False
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                return False
    return True


def _place_centers(
    rng: random.Random,
    h: int,
    w: int,
    total_markers: int,
    target_quadrant: int,
) -> list[tuple[int, int]] | None:
    centers_by_quadrant = _candidate_centers_by_quadrant(h, w)
    quadrant_count = rng.choice((3, 4))
    other_quadrants = [q for q in range(1, 5) if q != target_quadrant]
    used_quadrants = {target_quadrant, *rng.sample(other_quadrants, quadrant_count - 1)}

    per_quadrant = {q: 1 for q in used_quadrants}
    remaining = total_markers - len(used_quadrants)
    quadrants_cycle = list(used_quadrants)
    while remaining > 0:
        per_quadrant[rng.choice(quadrants_cycle)] += 1
        remaining -= 1

    occupied: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int]] = []
    for quadrant in rng.sample(sorted(used_quadrants), len(used_quadrants)):
        candidates = centers_by_quadrant[quadrant][:]
        rng.shuffle(candidates)
        needed = per_quadrant[quadrant]
        placed_here = 0
        for center in candidates:
            if _can_place_marker(center, occupied, (h, w)):
                chosen.append(center)
                occupied.update(_plus_cells(center))
                placed_here += 1
                if placed_here == needed:
                    break
        if placed_here != needed:
            return None

    return chosen if len(chosen) == total_markers else None


def _pick_target_centers(
    centers: list[tuple[int, int]],
    h: int,
    w: int,
    target_count: int,
    target_quadrant: int,
    rng: random.Random,
) -> set[tuple[int, int]] | None:
    indexed = centers[:]
    combinations = list(itertools.combinations(indexed, target_count))
    rng.shuffle(combinations)
    all_quadrants = {center: _quadrant_of_cell(center[0], center[1], h, w) for center in centers}

    for combo in combinations:
        target_centers = set(combo)
        target_quads = [all_quadrants[center] for center in combo]
        quad_counter = Counter(target_quads)
        max_count = max(quad_counter.values())
        selected_quad_count = quad_counter.get(target_quadrant, 0)
        selected_is_unique_max = selected_quad_count == max_count and sum(
            1 for count in quad_counter.values() if count == max_count
        ) == 1
        non_target_in_selected = any(
            all_quadrants[center] == target_quadrant and center not in target_centers for center in centers
        )
        if len(set(target_quads)) < 2:
            continue
        if not any(all_quadrants[center] != target_quadrant for center in target_centers):
            continue
        if not non_target_in_selected:
            continue
        if selected_is_unique_max:
            continue
        return target_centers
    return None


def _noise_count_bounds(h: int, w: int, target_count: int) -> tuple[int, int]:
    area = h * w
    lo = math.ceil(area * NOISE_DENSITY_RANGE[0])
    hi = math.floor(area * NOISE_DENSITY_RANGE[1])
    lo = max(lo, 4 * target_count + 1)
    return lo, hi


def _paint_candidate(
    rng: random.Random,
    h: int,
    w: int,
    centers: list[tuple[int, int]],
    target_centers: set[tuple[int, int]],
    target_color: int,
) -> tuple[np.ndarray, dict[str, object]] | None:
    grid = np.zeros((h, w), dtype=int)
    target_count = len(target_centers)
    non_target_centers = [center for center in centers if center not in target_centers]
    extra_colors = rng.sample([c for c in ARM_COLORS if c != target_color], len(non_target_centers))

    center_to_color: dict[tuple[int, int], int] = {}
    for center in centers:
        center_to_color[center] = target_color if center in target_centers else extra_colors.pop()

    occupied = set()
    clear_zone = set()
    for center in centers:
        color = center_to_color[center]
        cr, cc = center
        grid[cr, cc] = YELLOW
        for nr, nc in _neighbors4(cr, cc):
            grid[nr, nc] = color
        occupied.update(_plus_cells(center))
        for rr in range(cr - 1, cr + 2):
            for cc2 in range(cc - 1, cc + 2):
                if 0 <= rr < h and 0 <= cc2 < w:
                    clear_zone.add((rr, cc2))

    noise_lo, noise_hi = _noise_count_bounds(h, w, target_count)
    if noise_lo > noise_hi:
        return None

    available_noise = [(r, c) for r in range(h) for c in range(w) if (r, c) not in clear_zone]
    if len(available_noise) < noise_lo:
        return None

    noise_count = rng.randint(noise_lo, min(noise_hi, len(available_noise)))
    heavy_noise_color = rng.choice([c for c in ARM_COLORS if c != target_color])
    heavy_noise_count = rng.randint(4 * target_count + 1, noise_count)
    noise_positions = rng.sample(available_noise, noise_count)
    heavy_positions = set(noise_positions[:heavy_noise_count])
    supporting_colors = list({target_color, heavy_noise_color, *center_to_color.values(), *ARM_COLORS})

    for r, c in noise_positions:
        if (r, c) in heavy_positions:
            grid[r, c] = heavy_noise_color
        else:
            grid[r, c] = rng.choice(supporting_colors)

    meta = {
        "centers": centers,
        "target_centers": target_centers,
        "center_to_color": center_to_color,
        "target_color": target_color,
        "target_count": target_count,
        "heavy_noise_color": heavy_noise_color,
    }
    return grid, meta


def _validate_candidate(grid: np.ndarray, meta: dict[str, object]) -> bool:
    h, w = grid.shape
    centers = meta["centers"]
    target_centers = meta["target_centers"]
    target_color = int(meta["target_color"])
    target_count = int(meta["target_count"])

    markers = _detect_plus_markers(grid)
    if len(markers) != len(centers):
        return False

    detected_counts = Counter(int(marker["arm_color"]) for marker in markers)
    repeated = [color for color, count in detected_counts.items() if count > 1]
    if repeated != [target_color]:
        return False
    if detected_counts[target_color] != target_count:
        return False
    if len(centers) <= target_count:
        return False

    marker_centers = {tuple(marker["center"]) for marker in markers}
    if marker_centers != set(centers):
        return False

    target_quadrant = target_count
    all_quadrants = {center: _quadrant_of_cell(center[0], center[1], h, w) for center in centers}
    if len({all_quadrants[center] for center in centers}) < 3:
        return False
    target_quad_counts = Counter(all_quadrants[center] for center in target_centers)
    if len(target_quad_counts) < 2:
        return False
    if target_quad_counts.get(target_quadrant, 0) == max(target_quad_counts.values()) and list(
        target_quad_counts.values()
    ).count(max(target_quad_counts.values())) == 1:
        return False
    if not any(all_quadrants[center] != target_quadrant for center in target_centers):
        return False
    if not any(all_quadrants[center] == target_quadrant and center not in target_centers for center in centers):
        return False

    total_color_counts = Counter(int(value) for value in grid.ravel() if value not in (BLACK, YELLOW))
    best_other = max(
        (count for color, count in total_color_counts.items() if color != target_color),
        default=-1,
    )
    if total_color_counts[target_color] > best_other:
        return False

    expected = _render_output(h, w, target_color, target_count)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return False

    return True


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    parity_h = rng.choice((0, 1))
    parity_w = rng.choice((0, 1))
    h_choices = [dim for dim in range(GRID_MIN, GRID_MAX + 1) if dim % 2 == parity_h]
    w_choices = [dim for dim in range(GRID_MIN, GRID_MAX + 1) if dim % 2 == parity_w]
    h = rng.choice(h_choices)
    w = rng.choice(w_choices)

    target_count = rng.choice(TARGET_COUNT_OPTIONS)
    total_markers = rng.randint(max(MARKER_COUNT_RANGE[0], target_count + 1), MARKER_COUNT_RANGE[1])
    target_quadrant = target_count
    centers = _place_centers(rng, h, w, total_markers, target_quadrant)
    if centers is None:
        return None

    target_centers = _pick_target_centers(centers, h, w, target_count, target_quadrant, rng)
    if target_centers is None:
        return None

    target_color = rng.choice(ARM_COLORS)
    painted = _paint_candidate(rng, h, w, centers, target_centers, target_color)
    if painted is None:
        return None
    grid, meta = painted

    if not _validate_candidate(grid, meta):
        return None

    output_signature = _grid_signature(main(grid))
    if output_signature in _SEEN_OUTPUT_SIGNATURES:
        return None
    _SEEN_OUTPUT_SIGNATURES.add(output_signature)
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
