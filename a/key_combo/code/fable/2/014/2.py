from __future__ import annotations

from collections import deque
from importlib.machinery import SourceFileLoader
import random

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
components_mask = _ARC_UTILS.components_mask
get_bg = _ARC_UTILS.get_bg
neighbors4 = _ARC_UTILS.neighbors4

BG = Color.BLACK
GREEN = Color.GREEN
GRAY = Color.GREY
LEGEND_PALETTE = [c for c in Color.ALL_COLORS if c not in {BG, GREEN}]

WORK_SIZE = 30
FAMILY_GAP = 2
MARKER_SEP = 2
GEN_RETRIES = 400


def _boundary_color(rect: np.ndarray, top: int, bottom: int, left: int, right: int) -> int:
    parts = [
        rect[top, left:right],
        rect[bottom - 1, left:right],
        rect[top + 1 : bottom - 1, left],
        rect[top + 1 : bottom - 1, right - 1],
    ]
    boundary = np.concatenate([part.reshape(-1) for part in parts if part.size])
    assert boundary.size > 0
    colors = np.unique(boundary)
    assert colors.size == 1
    return int(colors[0])


def _read_legend_layers(rect: np.ndarray) -> list[int]:
    top, left = 0, 0
    bottom, right = map(int, rect.shape)
    layers: list[int] = []

    while top < bottom and left < right:
        color = _boundary_color(rect, top, bottom, left, right)
        layers.append(color)
        previous = (top, bottom, left, right)

        while top < bottom and np.all(rect[top, left:right] == color):
            top += 1
        while top < bottom and np.all(rect[bottom - 1, left:right] == color):
            bottom -= 1
        while left < right and top < bottom and np.all(rect[top:bottom, left] == color):
            left += 1
        while left < right and top < bottom and np.all(rect[top:bottom, right - 1] == color):
            right -= 1

        assert (top, bottom, left, right) != previous

    assert len(layers) >= 2
    assert len(layers) == len(set(layers))
    return layers


def _region_depth_data(grid: np.ndarray) -> tuple[np.ndarray, dict[int, int], np.ndarray]:
    green_mask = grid == GREEN
    region_coords = components_mask(~green_mask, connectivity=4)
    region_ids = np.full(grid.shape, -1, dtype=int)
    border_regions: set[int] = set()

    for region_id, coords in enumerate(region_coords):
        rows = coords[:, 0]
        cols = coords[:, 1]
        region_ids[rows, cols] = region_id
        touches_border = (
            np.any(rows == 0)
            or np.any(rows == grid.shape[0] - 1)
            or np.any(cols == 0)
            or np.any(cols == grid.shape[1] - 1)
        )
        if touches_border:
            border_regions.add(region_id)

    graph = {region_id: set() for region_id in range(len(region_coords))}
    for coords in components_mask(green_mask, connectivity=8):
        adjacent: set[int] = set()
        for r, c in coords:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1]:
                    region_id = int(region_ids[rr, cc])
                    if region_id >= 0:
                        adjacent.add(region_id)
        assert len(adjacent) == 2
        a, b = sorted(adjacent)
        graph[a].add(b)
        graph[b].add(a)

    depths: dict[int, int] = {}
    queue = deque()
    for region_id in sorted(border_regions):
        depths[region_id] = 0
        queue.append(region_id)

    while queue:
        region_id = queue.popleft()
        for other in sorted(graph[region_id]):
            if other in depths:
                continue
            depths[other] = depths[region_id] + 1
            queue.append(other)

    assert len(depths) == len(region_coords)
    depth_map = np.full(grid.shape, -1, dtype=int)
    for region_id, coords in enumerate(region_coords):
        depth_map[coords[:, 0], coords[:, 1]] = depths[region_id]
    return region_ids, depths, depth_map


def _find_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]:
    mask = (grid != bg) & (grid != GREEN)
    candidates: list[tuple[np.ndarray, tuple[int, int, int, int], np.ndarray]] = []

    for coords in components_mask(mask, connectivity=4):
        if int(coords.shape[0]) <= 1:
            continue
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        rect = grid[r0 : r1 + 1, c0 : c1 + 1]
        if int(coords.shape[0]) != rect.size:
            continue
        try:
            _read_legend_layers(rect)
        except AssertionError:
            continue
        legend_mask = np.zeros_like(grid, dtype=bool)
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        candidates.append((legend_mask, bbox, rect))

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each gray marker by the number of enclosing green loops, aligned to the legend's own submerged depth.

    Input structure:
    - The background is black; several non-intersecting green closed curves may appear and can be nested.
    - Gray single-pixel markers lie in various regions, either outside all loops or inside nested interiors.
    - One connected solid rectangle is a concentric multi-layer legend whose boundary colors name successive depths.

    Rule:
    - Partition the non-green space into regions and assign each region a depth equal to how many green closed curves must be crossed to reach the exterior.
    - Let the legend's region depth be j; its outermost layer stands for depth j, the next layer for depth j+1, and so on.
    - Recolor every gray marker at depth k with the legend layer at index k-j, keep every green curve unchanged, and erase the legend to background.

    Key insight:
    - The legend is not depth-zero by default: it is immersed inside the same nesting structure, so its own enclosing-loop count is the offset for reading all marker depths.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    region_ids, region_depths, _depth_map = _region_depth_data(g)
    legend_mask, legend_bbox, legend_rect = _find_legend(g, bg)
    layer_colors = _read_legend_layers(legend_rect)

    legend_cell = tuple(map(int, np.argwhere(legend_mask)[0]))
    legend_depth = region_depths[int(region_ids[legend_cell])]

    out = g.copy()
    out[legend_mask] = bg

    marker_coords = np.argwhere((g == GRAY) & (~legend_mask))
    assert marker_coords.size > 0
    for r, c in marker_coords:
        depth = region_depths[int(region_ids[int(r), int(c)])]
        layer_index = depth - legend_depth
        assert 0 <= layer_index < len(layer_colors)
        out[int(r), int(c)] = layer_colors[layer_index]

    return out


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_loop(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = GREEN
    grid[r1, c0 : c1 + 1] = GREEN
    grid[r0 : r1 + 1, c0] = GREEN
    grid[r0 : r1 + 1, c1] = GREEN


def _loop_bboxes(outer_bbox: tuple[int, int, int, int], depth: int) -> list[tuple[int, int, int, int]]:
    r0, c0, r1, c1 = outer_bbox
    return [(r0 + 2 * i, c0 + 2 * i, r1 - 2 * i, c1 - 2 * i) for i in range(depth)]


def _core_bbox(outer_bbox: tuple[int, int, int, int], depth: int) -> tuple[int, int, int, int]:
    inset = 1 + 2 * (depth - 1)
    r0, c0, r1, c1 = outer_bbox
    return r0 + inset, c0 + inset, r1 - inset, c1 - inset


def _build_legend(layer_count: int) -> tuple[np.ndarray, list[int]]:
    colors = random.sample(LEGEND_PALETTE, layer_count)
    thicknesses = [1] * (layer_count - 1)
    if thicknesses and random.random() < 0.35:
        thicknesses[random.randrange(len(thicknesses))] = 2
    inner_limit = 3 if layer_count <= 3 else 2
    inner_h = random.randint(1, inner_limit)
    inner_w = random.randint(1, inner_limit)

    height = inner_h + 2 * sum(thicknesses)
    width = inner_w + 2 * sum(thicknesses)
    patch = np.full((height, width), BG, dtype=int)

    top, left = 0, 0
    bottom, right = height, width
    for i, color in enumerate(colors):
        patch[top:bottom, left:right] = color
        if i < len(thicknesses):
            t = thicknesses[i]
            top += t
            left += t
            bottom -= t
            right -= t

    return patch, colors


def _family_specs(legend_depth: int, family_depths: list[int], legend_shape: tuple[int, int]) -> list[dict[str, int]]:
    specs: list[dict[str, int]] = []
    legend_h, legend_w = legend_shape
    for depth in family_depths:
        if depth == legend_depth and depth > 0:
            core_h = random.randint(legend_h + 2, min(11, legend_h + 4))
            core_w = random.randint(legend_w + 2, min(11, legend_w + 4))
        else:
            core_h = random.randint(5, 8)
            core_w = random.randint(5, 8)
        outer_h = core_h + 4 * depth - 2
        outer_w = core_w + 4 * depth - 2
        specs.append(
            {
                "depth": depth,
                "core_h": core_h,
                "core_w": core_w,
                "outer_h": outer_h,
                "outer_w": outer_w,
            }
        )
    return specs


def _place_families(specs: list[dict[str, int]]) -> list[dict[str, int]] | None:
    placed: list[dict[str, int]] = []
    for spec in sorted(specs, key=lambda item: item["outer_h"] * item["outer_w"], reverse=True):
        outer_h = spec["outer_h"]
        outer_w = spec["outer_w"]
        placed_here = False
        for _ in range(300):
            r0 = random.randint(2, WORK_SIZE - outer_h - 3)
            c0 = random.randint(2, WORK_SIZE - outer_w - 3)
            outer_bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_overlap(outer_bbox, other["outer_bbox"], gap=FAMILY_GAP) for other in placed):
                continue
            entry = dict(spec)
            entry["outer_r0"] = r0
            entry["outer_c0"] = c0
            entry["outer_r1"] = outer_bbox[2]
            entry["outer_c1"] = outer_bbox[3]
            core_r0, core_c0, core_r1, core_c1 = _core_bbox(outer_bbox, spec["depth"])
            entry["core_r0"] = core_r0
            entry["core_c0"] = core_c0
            entry["core_r1"] = core_r1
            entry["core_c1"] = core_c1
            entry["outer_bbox"] = outer_bbox
            placed.append(entry)
            placed_here = True
            break
        if not placed_here:
            return None
    return placed


def _rect_positions(
    grid: np.ndarray,
    allowed_mask: np.ndarray,
    patch_shape: tuple[int, int],
    margin: int,
) -> list[tuple[int, int]]:
    height, width = patch_shape
    positions: list[tuple[int, int]] = []
    for r0 in range(margin, grid.shape[0] - height - margin + 1):
        for c0 in range(margin, grid.shape[1] - width - margin + 1):
            r1 = r0 + height
            c1 = c0 + width
            if not np.all(allowed_mask[r0:r1, c0:c1]):
                continue
            if np.any(grid[r0:r1, c0:c1] != BG):
                continue
            rr0 = max(0, r0 - margin)
            cc0 = max(0, c0 - margin)
            rr1 = min(grid.shape[0], r1 + margin)
            cc1 = min(grid.shape[1], c1 + margin)
            if np.any(grid[rr0:rr1, cc0:cc1] != BG):
                continue
            positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _cell_score(grid: np.ndarray, r: int, c: int) -> int:
    score = 0
    for rr in range(r - 1, r + 2):
        for cc in range(c - 1, c + 2):
            if 0 <= rr < grid.shape[0] and 0 <= cc < grid.shape[1] and grid[rr, cc] == BG:
                score += 1
    return score


def _pick_marker_positions(
    grid: np.ndarray,
    depth_map: np.ndarray,
    plan: list[int],
) -> dict[tuple[int, int], int] | None:
    chosen: dict[tuple[int, int], int] = {}
    for depth in plan:
        candidates = [
            (int(r), int(c))
            for r, c in np.argwhere((grid == BG) & (depth_map == depth))
            if 1 <= int(r) < grid.shape[0] - 1 and 1 <= int(c) < grid.shape[1] - 1
        ]
        comfy = [pos for pos in candidates if _cell_score(grid, *pos) >= 6]
        pool = comfy if len(comfy) >= 1 else candidates
        if not pool:
            return None
        ordered = sorted(pool, key=lambda pos: (_cell_score(grid, *pos), random.random()), reverse=True)

        picked = None
        for pos in ordered:
            if all(max(abs(pos[0] - rr), abs(pos[1] - cc)) >= MARKER_SEP for rr, cc in chosen):
                picked = pos
                break
        if picked is None:
            return None
        chosen[picked] = depth
        grid[picked] = GRAY
    return chosen


def _expected_output(
    grid: np.ndarray,
    legend_mask: np.ndarray,
    marker_depths: dict[tuple[int, int], int],
    legend_depth: int,
    layer_colors: list[int],
) -> np.ndarray:
    out = grid.copy()
    out[legend_mask] = BG
    for (r, c), depth in marker_depths.items():
        out[r, c] = layer_colors[depth - legend_depth]
    return out


def _crop_pair(
    grid: np.ndarray,
    out: np.ndarray,
    marker_depths: dict[tuple[int, int], int],
) -> tuple[np.ndarray, np.ndarray, dict[tuple[int, int], int]]:
    r0, c0, r1, c1 = bbox_coords(np.argwhere(grid != BG))
    top = random.randint(1, min(4, r0))
    left = random.randint(1, min(4, c0))
    bottom = random.randint(1, min(4, grid.shape[0] - 1 - r1))
    right = random.randint(1, min(4, grid.shape[1] - 1 - c1))

    rr0 = r0 - top
    cc0 = c0 - left
    rr1 = r1 + bottom
    cc1 = c1 + right

    cropped_grid = grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
    cropped_out = out[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
    cropped_markers = {(r - rr0, c - cc0): depth for (r, c), depth in marker_depths.items()}
    return cropped_grid, cropped_out, cropped_markers


def _sample_case() -> tuple[int, list[int], list[int]]:
    roll = random.random()
    if roll < 0.35:
        legend_depth = 0
        max_depth = random.choice([1, 2, 2, 3, 3])
        marker_depths = {max_depth}
        if random.random() < 0.65:
            marker_depths.add(0)
        if max_depth > 1 and random.random() < 0.55:
            marker_depths.add(random.randint(1, max_depth - 1))
    elif roll < 0.8:
        legend_depth = 1
        max_depth = random.choice([2, 3])
        marker_depths = {1, max_depth}
    else:
        legend_depth = 2
        max_depth = 3
        marker_depths = {2, 3}

    family_depths = sorted(set(depth for depth in marker_depths if depth > 0) | ({legend_depth} if legend_depth > 0 else set()))
    return legend_depth, sorted(marker_depths), family_depths


def _marker_plan(marker_depths: list[int], legend_depth: int) -> list[int]:
    plan = list(marker_depths)
    extras = random.randint(1, 3)
    weights = [3 if depth == max(marker_depths) else 2 if depth == legend_depth else 1 for depth in marker_depths]
    for _ in range(extras):
        plan.append(random.choices(marker_depths, weights=weights, k=1)[0])
    random.shuffle(plan)
    return plan


def _gen_attempt() -> np.ndarray | None:
    legend_depth, marker_depths, family_depths = _sample_case()
    layer_count = max(marker_depths) - legend_depth + 1
    legend_patch, layer_colors = _build_legend(layer_count)

    family_specs = _family_specs(legend_depth, family_depths, legend_patch.shape)
    families = _place_families(family_specs)
    if families is None:
        return None

    grid = np.full((WORK_SIZE, WORK_SIZE), BG, dtype=int)
    for family in families:
        for loop_bbox in _loop_bboxes(family["outer_bbox"], family["depth"]):
            _draw_loop(grid, loop_bbox)

    legend_mask = np.zeros_like(grid, dtype=bool)
    if legend_depth == 0:
        allowed = grid == BG
        positions = _rect_positions(grid, allowed, legend_patch.shape, margin=1)
    else:
        family = next(family for family in families if family["depth"] == legend_depth)
        allowed = np.zeros_like(grid, dtype=bool)
        allowed[
            family["core_r0"] : family["core_r1"] + 1,
            family["core_c0"] : family["core_c1"] + 1,
        ] = True
        positions = _rect_positions(grid, allowed, legend_patch.shape, margin=1)
    if not positions:
        return None

    legend_r0, legend_c0 = positions[0]
    legend_r1 = legend_r0 + legend_patch.shape[0]
    legend_c1 = legend_c0 + legend_patch.shape[1]
    grid[legend_r0:legend_r1, legend_c0:legend_c1] = legend_patch
    legend_mask[legend_r0:legend_r1, legend_c0:legend_c1] = True

    _region_ids, _region_depths, depth_map = _region_depth_data(grid)
    marker_depth_plan = _marker_plan(marker_depths, legend_depth)
    placed_markers = _pick_marker_positions(grid, depth_map, marker_depth_plan)
    if placed_markers is None:
        return None

    expected = _expected_output(grid, legend_mask, placed_markers, legend_depth, layer_colors)
    cropped_grid, cropped_expected, cropped_markers = _crop_pair(grid, expected, placed_markers)

    try:
        _validate_case(cropped_grid, cropped_expected, cropped_markers, legend_depth, layer_colors)
    except AssertionError:
        return None

    return cropped_grid


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    marker_depths: dict[tuple[int, int], int],
    legend_depth: int,
    layer_colors: list[int],
) -> None:
    assert grid.ndim == 2 and max(grid.shape) <= 30
    assert expected.ndim == 2 and max(expected.shape) <= 30
    assert get_bg(grid) == BG

    legend_mask, _legend_bbox, legend_rect = _find_legend(grid, BG)
    assert _read_legend_layers(legend_rect) == layer_colors

    region_ids, region_depths, depth_map = _region_depth_data(grid)
    legend_cell = tuple(map(int, np.argwhere(legend_mask)[0]))
    assert region_depths[int(region_ids[legend_cell])] == legend_depth

    for (r, c), depth in marker_depths.items():
        assert grid[r, c] == GRAY
        assert depth_map[r, c] == depth
        assert expected[r, c] == layer_colors[depth - legend_depth]

    assert np.array_equal(main(grid), expected)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid submerged-depth-ruler input")
