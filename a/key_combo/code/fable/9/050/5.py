from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
YELLOW = 4
GREY = 5

MAX_GENERATION_ATTEMPTS = 400
MIN_FRAMES = 2
MAX_FRAMES = 5
MAX_OUTER_RADIUS = 11


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _label_components(mask: np.ndarray) -> tuple[np.ndarray, int]:
    labels = np.zeros(mask.shape, dtype=int)
    comp_id = 0
    height, width = mask.shape

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            queue = deque([(r, c)])
            labels[r, c] = comp_id
            while queue:
                rr, cc = queue.popleft()
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        queue.append((nr, nc))
    return labels, comp_id


def _analyze_regions(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert set(map(int, np.unique(g))) <= {BLACK, GREY}, "input should contain only black and grey"

    grey_mask = g == GREY
    assert grey_mask.any(), "expected at least one grey frame"
    r0, c0, r1, c1 = _bbox(grey_mask)
    assert r1 - r0 == c1 - c0, "outer frame bbox must be square"
    assert (r0 + r1) % 2 == 0 and (c0 + c1) % 2 == 0, "frame center must be a grid cell"

    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2

    inside_outer = np.zeros_like(grey_mask, dtype=bool)
    inside_outer[r0 + 1 : r1, c0 + 1 : c1] = True
    black_regions = inside_outer & (g == BLACK)

    labels, component_count = _label_components(black_regions)
    assert component_count >= 1, "expected black area inside the outer frame"

    center_component = int(labels[center_r, center_c])
    assert center_component > 0, "the center cell must belong to the innermost black region"

    graph = {comp_id: set() for comp_id in range(1, component_count + 1)}
    height, width = g.shape
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if g[r, c] != GREY:
                continue
            for (r_a, c_a), (r_b, c_b) in (((r - 1, c), (r + 1, c)), ((r, c - 1), (r, c + 1))):
                if not (0 <= r_a < height and 0 <= c_a < width and 0 <= r_b < height and 0 <= c_b < width):
                    continue
                if not (inside_outer[r_a, c_a] and inside_outer[r_b, c_b]):
                    continue
                a = int(labels[r_a, c_a])
                b = int(labels[r_b, c_b])
                if a > 0 and b > 0 and a != b:
                    graph[a].add(b)
                    graph[b].add(a)

    distances = {center_component: 0}
    queue = deque([center_component])
    while queue:
        current = queue.popleft()
        for neighbor in sorted(graph[current]):
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)

    assert len(distances) == component_count, "all black regions must form one nested chain"
    depth_histogram = Counter(distances.values())
    assert set(depth_histogram) == set(range(component_count)), "region depths must be consecutive"
    assert all(count == 1 for count in depth_histogram.values()), "each depth should contain exactly one region"

    ordered_components = [comp_id for comp_id, _depth in sorted(distances.items(), key=lambda item: item[1])]
    return {
        "bbox": (r0, c0, r1, c1),
        "center": (center_r, center_c),
        "labels": labels,
        "component_count": component_count,
        "distances": distances,
        "ordered_components": ordered_components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill every other black connected region inside concentric grey frames, counting merged regions created by breaches.

    Input structure:
    - The grid is black except for several concentric hollow grey square frames.
    - The frames partition the area inside the outermost frame into a center region and nested black ring regions.
    - Some non-outer frames may have a one-cell breach that connects the black regions on its two sides.

    Rule:
    - Restrict attention to black cells strictly inside the outermost frame.
    - Split those black cells into 4-connected regions, then order the regions from the one containing the center outward.
    - Fill the 1st, 3rd, 5th, ... such regions yellow, leave the even-indexed regions black, and preserve all grey frame cells.

    Key insight:
    - The alternation unit is the black connected region, not the geometric ring: a single breach merges two adjacent bands and flips the odd/even phase of every region outside it.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_regions(g)
    labels = info["labels"]

    out = g.copy()
    for comp_id, depth in info["distances"].items():
        if depth % 2 == 0:
            out[labels == comp_id] = YELLOW
    return out


def _sample_radii() -> list[int]:
    for _ in range(200):
        frame_count = random.randint(MIN_FRAMES, MAX_FRAMES)
        radii = [random.randint(1, 3)]
        for _frame_index in range(1, frame_count):
            radii.append(radii[-1] + random.randint(2, 4))
        if radii[-1] <= MAX_OUTER_RADIUS:
            return radii
    raise RuntimeError("failed to sample concentric frame radii")


def _sample_breaches(radii: list[int]) -> dict[int, tuple[str, int]]:
    candidates = list(range(len(radii) - 1))  # keep the outermost frame closed
    if not candidates:
        return {}

    roll = random.random()
    if roll < 0.2:
        chosen = []
    elif roll < 0.6 or len(candidates) == 1:
        chosen = [random.choice(candidates)]
    else:
        lower = 2 if len(candidates) >= 2 else 1
        count = random.randint(lower, len(candidates))
        chosen = random.sample(candidates, count)

    breaches: dict[int, tuple[str, int]] = {}
    for frame_index in chosen:
        radius = radii[frame_index]
        side = random.choice(("top", "bottom", "left", "right"))
        offset = random.randint(-radius + 1, radius - 1)
        breaches[frame_index] = (side, offset)
    return breaches


def _draw_frame(grid: np.ndarray, center: tuple[int, int], radius: int, breach: tuple[str, int] | None) -> None:
    cr, cc = center
    r0, r1 = cr - radius, cr + radius
    c0, c1 = cc - radius, cc + radius

    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    if breach is None:
        return

    side, offset = breach
    if side == "top":
        grid[r0, cc + offset] = BLACK
    elif side == "bottom":
        grid[r1, cc + offset] = BLACK
    elif side == "left":
        grid[cr + offset, c0] = BLACK
    else:
        grid[cr + offset, c1] = BLACK


def _geometric_band_index(center: tuple[int, int], radii: list[int], r: int, c: int) -> int:
    cr, cc = center
    depth = max(abs(r - cr), abs(c - cc))
    return 1 + sum(radius < depth for radius in radii)


def _naive_ring_fill(grid: np.ndarray, center: tuple[int, int], radii: list[int]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    cr, cc = center
    outer_radius = radii[-1]
    for r in range(cr - outer_radius + 1, cr + outer_radius):
        for c in range(cc - outer_radius + 1, cc + outer_radius):
            if out[r, c] != BLACK:
                continue
            if max(abs(r - cr), abs(c - cc)) >= outer_radius:
                continue
            if _geometric_band_index(center, radii, r, c) % 2 == 1:
                out[r, c] = YELLOW
    return out


def _validate_generated_case(
    grid: np.ndarray,
    radii: list[int],
    breaches: dict[int, tuple[str, int]],
    center: tuple[int, int],
) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) != {BLACK, GREY}:
        return False

    grey_count_expected = sum(8 * radius for radius in radii) - len(breaches)
    if int((grid == GREY).sum()) != grey_count_expected:
        return False

    analysis = _analyze_regions(grid)
    if analysis["center"] != center:
        return False

    expected_components = len(radii) - len(breaches)
    if expected_components < 2 or analysis["component_count"] != expected_components:
        return False

    solved = main(grid)
    naive = _naive_ring_fill(grid, center, radii)
    if len(breaches) == 0:
        if not np.array_equal(solved, naive):
            return False
    else:
        if np.array_equal(solved, naive):
            return False
        labels = analysis["labels"]
        merged_component_exists = False
        for comp_id in analysis["ordered_components"]:
            cells = np.argwhere(labels == comp_id)
            band_ids = {
                _geometric_band_index(center, radii, int(r), int(c))
                for r, c in cells
            }
            if len(band_ids) > 1:
                merged_component_exists = True
                break
        if not merged_component_exists:
            return False

    inner_area = (analysis["bbox"][2] - analysis["bbox"][0] - 1) * (analysis["bbox"][3] - analysis["bbox"][1] - 1)
    yellow_count = int((solved == YELLOW).sum())
    if not (0 < yellow_count < inner_area):
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    radii = _sample_radii()
    breaches = _sample_breaches(radii)

    top_margin = random.randint(1, 3)
    bottom_margin = random.randint(1, 3)
    left_margin = random.randint(1, 3)
    right_margin = random.randint(1, 3)

    outer_radius = radii[-1]
    height = 2 * outer_radius + 1 + top_margin + bottom_margin
    width = 2 * outer_radius + 1 + left_margin + right_margin
    if height > 30 or width > 30:
        return None

    center = (top_margin + outer_radius, left_margin + outer_radius)
    grid = np.zeros((height, width), dtype=int)

    for frame_index, radius in enumerate(radii):
        _draw_frame(grid, center, radius, breaches.get(frame_index))

    return grid if _validate_generated_case(grid, radii, breaches, center) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 5")
