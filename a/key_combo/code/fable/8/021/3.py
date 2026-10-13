from __future__ import annotations

import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg  # noqa: E402


BACKGROUND = 0
PALETTE = tuple(range(1, 10))
GEN_RETRIES = 400
MAX_GRID = 30


def _frame_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _distance_to_frame_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, bbox)
    dr = max(r0 - int(r), 0, int(r) - r1)
    dc = max(c0 - int(c), 0, int(c) - c1)
    return int(max(dr, dc))


def _find_frame(grid: np.ndarray) -> dict:
    bg = int(get_bg(grid))
    candidates: list[dict] = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        b = comp["bbox"]
        r0, c0, r1, c1 = map(int, b)
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue
        patch = crop_bbox(grid, b)
        border = _frame_mask(h, w)
        color = int(comp["color"])
        if int(comp["area"]) != int(border.sum()):
            continue
        if not np.array_equal(patch == color, border):
            continue
        candidates.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "coords": comp["coords"],
                "inner_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )
    assert len(candidates) == 1, "expected exactly one hollow rectangular frame"
    return candidates[0]


def _extract_layers(grid: np.ndarray, frame: dict) -> list[dict]:
    bg = int(get_bg(grid))
    frame_cells = {tuple(map(int, rc)) for rc in frame["coords"]}
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in np.argwhere(grid != bg):
        rr, cc = int(r), int(c)
        if (rr, cc) in frame_cells:
            continue
        by_color[int(grid[rr, cc])].append((rr, cc))

    assert by_color, "expected scattered color pixels outside the frame"

    layers: list[dict] = []
    min_distances = []
    r0, c0, r1, c1 = map(int, frame["bbox"])
    for color, pts in by_color.items():
        assert pts, "each color must contribute at least one pixel"
        assert all(not (r0 < r < r1 and c0 < c < c1) for r, c in pts), "noise pixels must stay outside frame"
        dists = [_distance_to_frame_bbox(r, c, frame["bbox"]) for r, c in pts]
        min_dist = int(min(dists))
        min_distances.append(min_dist)
        layers.append(
            {
                "color": int(color),
                "thickness": int(len(pts)),
                "distance": min_dist,
                "points": pts,
            }
        )

    assert len(set(min_distances)) == len(min_distances), "color-to-frame nearest distances must be unique"
    layers.sort(key=lambda item: item["distance"])
    return layers


def _render_output(shape: tuple[int, int], frame: dict, layers: list[dict]) -> np.ndarray:
    out = np.full(shape, BACKGROUND, dtype=int)
    r0, c0, r1, c1 = map(int, frame["bbox"])
    border = _frame_mask(r1 - r0 + 1, c1 - c0 + 1)
    out_patch = out[r0 : r1 + 1, c0 : c1 + 1]
    out_patch[border] = int(frame["color"])

    ir0, ic0, ir1, ic1 = map(int, frame["inner_bbox"])
    for layer in layers:
        t = int(layer["thickness"])
        assert ir0 <= ir1 and ic0 <= ic1, "rings exceed available frame interior"
        assert 2 * t <= (ir1 - ir0 + 1) and 2 * t <= (ic1 - ic0 + 1), "generator must leave room for later rings"

        color = int(layer["color"])
        out[ir0 : ir0 + t, ic0 : ic1 + 1] = color
        out[ir1 - t + 1 : ir1 + 1, ic0 : ic1 + 1] = color
        out[ir0 + t : ir1 - t + 1, ic0 : ic0 + t] = color
        out[ir0 + t : ir1 - t + 1, ic1 - t + 1 : ic1 + 1] = color
        ir0 += t
        ic0 += t
        ir1 -= t
        ic1 -= t

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Decode scattered color pixels into concentric rings inside the hollow frame.

    Input structure:
    - A black background contains exactly one 1-pixel-thick hollow rectangular frame.
    - The frame interior is empty in the input, while colored pixels are scattered outside the frame.
    - Each scattered color may appear multiple times, and a color's nearest Chebyshev distance to the frame is unique.

    Rule:
    - Ignore the frame color itself and group all remaining non-black pixels by color.
    - For each color, take the minimum Chebyshev distance from any of its pixels to the frame border; sort colors by this distance from near to far.
    - Draw concentric rectangular rings inside the frame, from outermost inward, using that sorted order; each ring's thickness equals the total number of input pixels of that color.
    - Erase all scattered pixels outside the frame, preserve the frame, and leave any leftover center of the frame black.

    Key insight:
    - Each color encodes two independent pieces of information at once: nearest distance decides ring order, while pixel count decides ring thickness.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    frame = _find_frame(grid)
    layers = _extract_layers(grid, frame)
    return _render_output(grid.shape, frame, layers)


def _shell_cells(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    distance: int,
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    cells = []
    rr0 = max(0, r0 - distance)
    rr1 = min(h - 1, r1 + distance)
    cc0 = max(0, c0 - distance)
    cc1 = min(w - 1, c1 + distance)
    for r in range(rr0, rr1 + 1):
        for c in range(cc0, cc1 + 1):
            if r0 <= r <= r1 and c0 <= c <= c1:
                continue
            if _distance_to_frame_bbox(r, c, bbox) == int(distance):
                cells.append((int(r), int(c)))
    random.shuffle(cells)
    return cells


def _sample_thicknesses(num_colors: int) -> list[int] | None:
    for _ in range(80):
        total = random.randint(num_colors + 1, 7)
        remain = total
        vals = []
        for idx in range(num_colors):
            left = num_colors - idx - 1
            lo = max(1, remain - 3 * left)
            hi = min(3, remain - left)
            if lo > hi:
                break
            val = random.randint(lo, hi)
            vals.append(val)
            remain -= val
        if len(vals) != num_colors or remain != 0:
            continue
        if max(vals) == 1:
            continue
        if len(set(vals)) == 1:
            continue
        random.shuffle(vals)
        return vals
    return None


def _far_shell_limit(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    return int(max(r0, c0, h - 1 - r1, w - 1 - c1))


def _place_noise(
    shape: tuple[int, int],
    bbox: tuple[int, int, int, int],
    assignments: list[tuple[int, int, int]],
) -> dict[int, list[tuple[int, int]]] | None:
    occupied: list[tuple[int, int]] = []
    by_color: dict[int, list[tuple[int, int]]] = {}
    shell_cache: dict[int, list[tuple[int, int]]] = {}
    for color, min_distance, count in assignments:
        far_limit = _far_shell_limit(shape, bbox)
        must_spread = count >= 2 and far_limit > min_distance and random.random() < 0.7
        spread_budget = random.randint(1, count - 1) if must_spread else 0

        planned_distances = [min_distance]
        for idx in range(1, count):
            if idx <= spread_budget:
                planned_distances.append(random.randint(min_distance + 1, far_limit))
            else:
                planned_distances.append(random.randint(min_distance, far_limit))
        random.shuffle(planned_distances)
        if min(planned_distances) != min_distance:
            planned_distances[0] = min_distance

        points: list[tuple[int, int]] = []
        for dist in planned_distances:
            if dist not in shell_cache:
                shell_cache[dist] = _shell_cells(shape, bbox, dist)
            candidates = shell_cache[dist][:]
            random.shuffle(candidates)
            placed = None
            for cell in candidates:
                if cell in points:
                    continue
                if any(max(abs(cell[0] - rr), abs(cell[1] - cc)) <= 1 for rr, cc in occupied):
                    continue
                placed = cell
                break
            if placed is None:
                return None
            points.append(placed)
            occupied.append(placed)
        if min(_distance_to_frame_bbox(r, c, bbox) for r, c in points) != min_distance:
            return None
        by_color[int(color)] = points
    return by_color


def _attempt_generate() -> np.ndarray | None:
    num_colors = random.randint(2, 4)
    thicknesses = _sample_thicknesses(num_colors)
    if thicknesses is None:
        return None

    total_thickness = int(sum(thicknesses))
    center_h = random.randint(1, 4)
    center_w = random.randint(1, 4)
    inner_h = 2 * total_thickness + center_h
    inner_w = 2 * total_thickness + center_w
    frame_h = inner_h + 2
    frame_w = inner_w + 2
    if frame_h > MAX_GRID - 4 or frame_w > MAX_GRID - 4:
        return None

    max_possible_distance = min((MAX_GRID - frame_h) // 2, (MAX_GRID - frame_w) // 2)
    if max_possible_distance < num_colors:
        return None
    max_distance = random.randint(num_colors, max_possible_distance)
    distances = sorted(random.sample(range(1, max_distance + 1), num_colors))

    min_h = frame_h + 2 * max_distance
    min_w = frame_w + 2 * max_distance
    if min_h > MAX_GRID or min_w > MAX_GRID:
        return None
    grid_h = random.randint(min_h, MAX_GRID)
    grid_w = random.randint(min_w, MAX_GRID)
    top = random.randint(max_distance, grid_h - frame_h - max_distance)
    left = random.randint(max_distance, grid_w - frame_w - max_distance)
    bbox = (top, left, top + frame_h - 1, left + frame_w - 1)

    colors = random.sample(PALETTE, num_colors + 1)
    frame_color = int(colors[0])
    ring_colors = [int(c) for c in colors[1:]]
    assignments = [(ring_colors[i], distances[i], thicknesses[i]) for i in range(num_colors)]
    random.shuffle(assignments)

    noise = _place_noise((grid_h, grid_w), bbox, assignments)
    if noise is None:
        return None

    grid = np.full((grid_h, grid_w), BACKGROUND, dtype=int)
    r0, c0, r1, c1 = map(int, bbox)
    border = _frame_mask(frame_h, frame_w)
    patch = np.full((frame_h, frame_w), BACKGROUND, dtype=int)
    patch[border] = frame_color
    grid[r0 : r1 + 1, c0 : c1 + 1] = patch
    for color, pts in noise.items():
        for r, c in pts:
            grid[r, c] = int(color)

    frame = _find_frame(grid)
    layers = _extract_layers(grid, frame)
    expected = _render_output(grid.shape, frame, layers)
    if not np.array_equal(main(grid), expected):
        return None

    distances_seen = [layer["distance"] for layer in layers]
    thickness_seen = [layer["thickness"] for layer in layers]
    if len(set(distances_seen)) != num_colors:
        return None
    if max(thickness_seen) <= 1 or len(set(thickness_seen)) == 1:
        return None
    if not any(
        len({_distance_to_frame_bbox(r, c, frame["bbox"]) for r, c in layer["points"]}) > 1
        for layer in layers
        if layer["thickness"] >= 2
    ):
        return None
    if not bool((expected == BACKGROUND).any()):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input for puzzle 3")
