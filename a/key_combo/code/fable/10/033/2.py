from __future__ import annotations

from collections import deque
import random

import numpy as np


PALETTE = list(range(10))
GEN_ATTEMPTS = 500
VARIANTS = ("open", "room", "double_room", "thick")
VARIANT_WEIGHTS = (3, 3, 2, 2)


def _compress_runs(row: np.ndarray) -> list[int]:
    values = [int(v) for v in row]
    if not values:
        return []
    out = [values[0]]
    for value in values[1:]:
        if value != out[-1]:
            out.append(value)
    return out


def _random_partition(total: int, parts: int) -> list[int]:
    assert total >= parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    widths = []
    prev = 0
    for cut in cuts + [total]:
        widths.append(cut - prev)
        prev = cut
    return widths


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 3 and g.shape[1] >= 2

    separator_row = g[-2]
    assert np.all(separator_row == separator_row[0]), "separator row must be uniform"
    separator_color = int(separator_row[0])

    legend = _compress_runs(g[-1])
    assert len(legend) >= 2, "legend must contain at least two colors"
    assert len(set(legend)) == len(legend), "legend colors must be distinct"

    region = g[:-2].copy()
    forbidden = set(legend)
    forbidden.add(separator_color)
    body_colors = sorted(set(map(int, np.unique(region))) - forbidden)
    assert len(body_colors) == 3, "main region must contain background, walls, and one source color"

    counts = {color: int(np.count_nonzero(region == color)) for color in body_colors}
    source_candidates = [color for color, count in counts.items() if count == 1]
    assert len(source_candidates) == 1, "source color must appear exactly once"
    source_color = source_candidates[0]
    source_pos = tuple(map(int, np.argwhere(region == source_color)[0]))

    others = [color for color in body_colors if color != source_color]
    assert len(others) == 2
    bg_color = max(others, key=lambda color: counts[color])
    wall_color = others[0] if others[1] == bg_color else others[1]
    assert counts[bg_color] > counts[wall_color], "background must be the dominant non-source color"

    return {
        "grid": g,
        "region": region,
        "separator_color": separator_color,
        "legend": legend,
        "k": len(legend),
        "bg_color": int(bg_color),
        "wall_color": int(wall_color),
        "source_color": int(source_color),
        "source_pos": source_pos,
    }


def _bfs_distances(region: np.ndarray, wall_color: int, source_pos: tuple[int, int]) -> np.ndarray:
    h, w = region.shape
    dist = np.full((h, w), -1, dtype=int)
    sr, sc = source_pos
    dist[sr, sc] = 0
    queue: deque[tuple[int, int]] = deque([(sr, sc)])

    while queue:
        r, c = queue.popleft()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= rr < h and 0 <= cc < w and dist[rr, cc] < 0 and int(region[rr, cc]) != wall_color:
                dist[rr, cc] = dist[r, c] + 1
                queue.append((rr, cc))
    return dist


def _apply_rule(parsed: dict, dist: np.ndarray) -> np.ndarray:
    out = parsed["grid"].copy()
    region = out[:-2]
    bg_color = parsed["bg_color"]
    legend = parsed["legend"]
    for r, c in np.argwhere((parsed["region"] == bg_color) & (dist >= 1)):
        region[r, c] = legend[(int(dist[r, c]) - 1) % len(legend)]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Color each reachable background cell by geodesic distance from the unique source using the bottom legend cycle.

    Input structure:
    - The bottom row is a legend bar made of k distinct color runs, and the row above it is a uniform separator.
    - Above the separator lies the main region containing one dominant background color, one wall color, and exactly one source pixel.
    - Legend colors and separator color are not reused inside the main region.

    Rule:
    - Work only inside the main region.
    - Starting from the unique source, compute 4-neighbor shortest-path distances through all non-wall cells.
    - Every reachable background cell at distance d >= 1 is recolored with the ((d-1) mod k)-th legend color, while the source, walls, unreachable background, separator row, and legend bar stay unchanged.

    Key insight:
    - The colored bands follow maze geodesic shells rather than Euclidean circles, and the bottom legend fixes both the cycle order and its phase.
    """
    parsed = _parse_grid(grid)
    dist = _bfs_distances(parsed["region"], parsed["wall_color"], parsed["source_pos"])
    return _apply_rule(parsed, dist)


def _add_closed_rooms(region: np.ndarray, bg_color: int, wall_color: int, source_pos: tuple[int, int], room_count: int) -> int:
    h, w = region.shape
    placed = 0
    for _ in range(250):
        if placed >= room_count:
            break
        outer_h = random.randint(3, 5)
        outer_w = random.randint(3, 5)
        if h < outer_h + 2 or w < outer_w + 2:
            break
        r0 = random.randint(1, h - outer_h - 1)
        c0 = random.randint(1, w - outer_w - 1)
        r1 = r0 + outer_h - 1
        c1 = c0 + outer_w - 1
        sr, sc = source_pos
        if r0 - 1 <= sr <= r1 + 1 and c0 - 1 <= sc <= c1 + 1:
            continue
        surround = region[r0 - 1 : r1 + 2, c0 - 1 : c1 + 2]
        if surround.shape != (outer_h + 2, outer_w + 2):
            continue
        if not np.all(surround == bg_color):
            continue

        region[r0 : r1 + 1, c0] = wall_color
        region[r0 : r1 + 1, c1] = wall_color
        region[r0, c0 : c1 + 1] = wall_color
        region[r1, c0 : c1 + 1] = wall_color
        placed += 1
    return placed


def _transform_region(region: np.ndarray) -> np.ndarray:
    out = np.rot90(region, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=int, copy=True)


def _make_region(height: int, width: int, bg_color: int, wall_color: int, source_color: int, variant: str) -> np.ndarray | None:
    region = np.full((height, width), bg_color, dtype=int)

    thick_bias = 0.6 if variant == "thick" else 0.25
    first_gap_bottom = random.choice([True, False])
    wall_specs: list[tuple[int, int, bool, int]] = []
    col = random.randint(2, 3)
    while col < width - 2 and len(wall_specs) < 6:
        thickness = 2 if random.random() < thick_bias and col < width - 3 else 1
        if col + thickness >= width - 1:
            break
        gap_size = random.randint(1, min(3, height - 2))
        wall_specs.append((col, thickness, first_gap_bottom, gap_size))
        col += thickness + random.randint(2, 4)
        first_gap_bottom = not first_gap_bottom

    if len(wall_specs) < 3:
        return None

    for col, thickness, gap_bottom, gap_size in wall_specs:
        if gap_bottom:
            region[: height - gap_size, col : col + thickness] = wall_color
        else:
            region[gap_size:, col : col + thickness] = wall_color

    # Open extra doorways on some internal walls so shortest-path choices are not all single-corridor forced.
    for col, thickness, gap_bottom, gap_size in wall_specs[1:-1]:
        if random.random() < 0.45:
            if gap_bottom:
                low = 1
                high = max(low, height - gap_size - 2)
            else:
                low = min(height - 2, gap_size + 1)
                high = height - 2
            if low <= high:
                door_r = random.randint(low, high)
                region[door_r, col : col + thickness] = bg_color

    left_room_width = wall_specs[0][0]
    if left_room_width < 2:
        return None
    source_r = random.randint(0, min(2, height - 1)) if wall_specs[0][2] else random.randint(max(0, height - 3), height - 1)
    source_c = random.randint(0, left_room_width - 1)
    region[source_r, source_c] = source_color

    requested_rooms = {"open": 0, "room": 1, "double_room": 2, "thick": random.choice([0, 1])}[variant]
    placed_rooms = _add_closed_rooms(region, bg_color, wall_color, (source_r, source_c), requested_rooms)
    if variant in {"room", "double_room"} and placed_rooms < requested_rooms:
        return None

    return region


def _legend_row(width: int, legend: list[int]) -> np.ndarray:
    widths = _random_partition(width, len(legend))
    row = np.empty(width, dtype=int)
    c0 = 0
    for color, block_w in zip(legend, widths):
        row[c0 : c0 + block_w] = int(color)
        c0 += block_w
    return row


def _manhattan_output(grid: np.ndarray) -> np.ndarray:
    parsed = _parse_grid(grid)
    out = parsed["grid"].copy()
    region = out[:-2]
    src_r, src_c = parsed["source_pos"]
    legend = parsed["legend"]
    bg_mask = parsed["region"] == parsed["bg_color"]
    for r, c in np.argwhere(bg_mask):
        d = abs(int(r) - src_r) + abs(int(c) - src_c)
        if d >= 1:
            region[r, c] = legend[(d - 1) % len(legend)]
    return out


def _is_valid_sample(grid: np.ndarray, expect_rooms: bool) -> bool:
    try:
        parsed = _parse_grid(grid)
        dist = _bfs_distances(parsed["region"], parsed["wall_color"], parsed["source_pos"])
        out = _apply_rule(parsed, dist)
    except AssertionError:
        return False

    bg_mask = parsed["region"] == parsed["bg_color"]
    reachable_mask = bg_mask & (dist >= 1)
    unreachable_mask = bg_mask & (dist < 0)
    if np.count_nonzero(reachable_mask) < max(14, parsed["k"] * 5):
        return False

    max_dist = int(dist.max())
    if max_dist < parsed["k"] + 2:
        return False

    if any(color not in set(map(int, np.unique(out[:-2]))) for color in parsed["legend"]):
        return False

    if not any(np.any(dist == d) and np.any(dist == d + parsed["k"]) for d in range(1, parsed["k"] + 1)):
        return False

    src_r, src_c = parsed["source_pos"]
    rr, cc = np.indices(parsed["region"].shape)
    manhattan = np.abs(rr - src_r) + np.abs(cc - src_c)
    if not np.any(reachable_mask & (dist > manhattan)):
        return False

    if np.array_equal(out, grid):
        return False

    if np.array_equal(out, _manhattan_output(grid)):
        return False

    if expect_rooms and np.count_nonzero(unreachable_mask) == 0:
        return False

    if not np.array_equal(main(grid), out):
        return False

    return True


def _assemble_grid() -> np.ndarray | None:
    variant = random.choices(VARIANTS, weights=VARIANT_WEIGHTS, k=1)[0]
    region_h = random.randint(9, 17)
    region_w = random.randint(12, 24)

    k = random.randint(2, 5)
    colors = random.sample(PALETTE, k + 4)
    bg_color, wall_color, source_color, separator_color = colors[:4]
    legend = colors[4:]

    region = _make_region(region_h, region_w, bg_color, wall_color, source_color, variant)
    if region is None:
        return None

    region = _transform_region(region)
    if max(region.shape) > 28:
        return None

    grid = np.empty((region.shape[0] + 2, region.shape[1]), dtype=int)
    grid[:-2] = region
    grid[-2] = separator_color
    grid[-1] = _legend_row(region.shape[1], legend)

    expect_rooms = variant in {"room", "double_room"}
    if not _is_valid_sample(grid, expect_rooms):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _assemble_grid()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid geodesic ring input")
