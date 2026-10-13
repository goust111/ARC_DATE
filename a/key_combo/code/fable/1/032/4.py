from __future__ import annotations

from importlib.machinery import SourceFileLoader
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    "/home/lixinyu/snap/new/solvers/arc_utils.py",
).load_module()
get_bg = _ARC_UTILS.get_bg
find_components = _ARC_UTILS.find_components
label_mask = _ARC_UTILS.label_mask


def _legend_components(grid: np.ndarray, bg: int) -> list[dict]:
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    legends = [comp for comp in components if int(comp["bbox"][0]) == 0]
    assert legends, "expected top-edge legend bars"
    return sorted(legends, key=lambda comp: (int(comp["bbox"][1]), int(comp["bbox"][0])))


def _role_colors(grid: np.ndarray, bg: int, legends: list[dict]) -> tuple[int, int, set[int]]:
    legend_colors = {int(comp["color"]) for comp in legends}
    components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )

    by_color: dict[int, list[dict]] = {}
    for comp in components:
        color = int(comp["color"])
        if color in legend_colors:
            continue
        by_color.setdefault(color, []).append(comp)

    dot_candidates = [
        color for color, parts in by_color.items() if all(int(part["area"]) == 1 for part in parts)
    ]
    assert len(dot_candidates) == 1, "expected one singleton dot color"
    dot_color = int(dot_candidates[0])

    contour_candidates = [color for color in by_color if color != dot_color]
    assert len(contour_candidates) == 1, "expected one contour color"
    contour_color = int(contour_candidates[0])
    return contour_color, dot_color, legend_colors


def _enclosed_regions(
    grid: np.ndarray,
    contour_color: int,
    legend_colors: set[int],
    dot_color: int,
) -> list[dict]:
    barrier = (grid == int(contour_color)) | np.isin(grid, list(legend_colors))
    open_mask = ~barrier
    labels, n_regions = label_mask(open_mask, connectivity=4)

    regions: list[dict] = []
    for region_id in range(1, n_regions + 1):
        coords = np.argwhere(labels == region_id)
        if coords.size == 0:
            continue
        touches_border = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == grid.shape[0] - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == grid.shape[1] - 1)
        )
        if touches_border:
            continue
        region_mask = labels == region_id
        dot_count = int(np.sum(grid[region_mask] == int(dot_color)))
        regions.append(
            {
                "id": int(region_id),
                "mask": region_mask,
                "dot_count": dot_count,
            }
        )

    assert regions, "expected enclosed regions"
    counts = [region["dot_count"] for region in regions]
    assert len(set(counts)) == len(counts), "dot counts must be unique"
    return sorted(regions, key=lambda region: region["dot_count"], reverse=True)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the rank-selected enclosed compartments with the legend colors.

    Input structure:
    - The background is the dominant color.
    - One contour color forms several closed loops, creating enclosed compartments and
      one exterior region.
    - A single dot color appears as isolated marker pixels inside the compartments, and
      the top-left edge contains several monochrome vertical legend bars whose heights
      encode rank indices.

    Rule:
    - Read each legend bar as (color K, height k), where k is a rank and not a size to copy.
    - Treat contour cells and legend bars as barriers, but count the dot-color pixels inside
      each enclosed compartment.
    - Rank the enclosed compartments by dot count from most to fewest, then recolor every
      cell of the k-th compartment, including its dots, to K; leave unselected compartments,
      the contour, and the exterior unchanged.

    Key insight:
    - The legends do not point to geometry directly: bar height names a rank in the ordering
      induced by compartment dot counts, and selected dots are absorbed by the fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    legends = _legend_components(g, bg)
    contour_color, dot_color, legend_colors = _role_colors(g, bg, legends)
    ranked_regions = _enclosed_regions(g, contour_color, legend_colors, dot_color)

    out = g.copy()
    for legend in legends:
        rank = int(legend["bbox"][2]) - int(legend["bbox"][0]) + 1
        assert 1 <= rank <= len(ranked_regions), "legend rank must select a region"
        region = ranked_regions[rank - 1]
        out[region["mask"]] = int(legend["color"])

    return out


_GEN_MAX_ATTEMPTS = 400
_GEN_ROOM_MARGIN = 2
_GEN_SIZE_RANGE = (22, 30)
_GEN_ROOM_COUNT_CHOICES = (3, 3, 4, 4)
_GEN_OUTER_H_RANGE = (6, 10)
_GEN_OUTER_W_RANGE = (6, 10)
_GEN_BAR_GAP = 1
_GEN_BAR_WIDTH = 1
_GEN_FAR_LEFT = 0


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + margin < br0
        or br1 + margin < ar0
        or ac1 + margin < bc0
        or bc1 + margin < ac0
    )


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _dot_candidates(bbox: tuple[int, int, int, int], count: int) -> list[tuple[int, int]] | None:
    r0, c0, r1, c1 = bbox
    inner_rows = list(range(r0 + 1, r1))
    inner_cols = list(range(c0 + 1, c1))
    offsets = [(0, 0), (0, 1), (1, 0), (1, 1)]
    random.shuffle(offsets)
    for row_offset, col_offset in offsets:
        cells = [
            (rr, cc)
            for rr in inner_rows[row_offset::2]
            for cc in inner_cols[col_offset::2]
        ]
        if len(cells) >= count:
            return cells
    return None


def _legend_count(room_count: int) -> int:
    if room_count == 1:
        return 1
    if random.random() < 0.75:
        return random.randint(1, room_count - 1)
    return room_count


def _place_rooms(
    height: int,
    width: int,
    room_count: int,
    top_clearance: int,
) -> list[tuple[int, int, int, int]] | None:
    rooms: list[tuple[int, int, int, int]] = []
    for _ in range(room_count):
        placed = False
        for _attempt in range(200):
            outer_h = random.randint(*_GEN_OUTER_H_RANGE)
            outer_w = random.randint(*_GEN_OUTER_W_RANGE)
            if top_clearance + outer_h + 1 >= height or outer_w + 2 >= width:
                continue

            r0 = random.randint(top_clearance, height - outer_h - 1)
            c0 = random.randint(1, width - outer_w - 1)
            bbox = (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1)
            if any(_boxes_too_close(bbox, other, _GEN_ROOM_MARGIN) for other in rooms):
                continue
            rooms.append(bbox)
            placed = True
            break
        if not placed:
            return None
    return rooms


def _room_capacity(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    return ((inner_h + 1) // 2) * ((inner_w + 1) // 2)


def _choose_counts(rooms: list[tuple[int, int, int, int]]) -> list[int] | None:
    min_capacity = min(_room_capacity(room) for room in rooms)
    if min_capacity < len(rooms):
        return None

    count_pool = list(range(1, min_capacity + 1))
    counts = random.sample(count_pool, len(rooms))
    random.shuffle(counts)

    areas = [(room[2] - room[0] - 1) * (room[3] - room[1] - 1) for room in rooms]
    area_order = sorted(range(len(rooms)), key=lambda idx: areas[idx], reverse=True)
    count_order = sorted(range(len(rooms)), key=lambda idx: counts[idx], reverse=True)
    if area_order == count_order:
        return None
    return counts


def _expected_output(
    grid: np.ndarray,
    rooms: list[tuple[int, int, int, int]],
    counts: list[int],
    legend_specs: list[tuple[int, int]],
) -> np.ndarray:
    ranked_room_ids = sorted(range(len(rooms)), key=lambda idx: counts[idx], reverse=True)
    out = grid.copy()
    for legend_color, rank in legend_specs:
        room_id = ranked_room_ids[rank - 1]
        r0, c0, r1, c1 = rooms[room_id]
        out[r0 + 1 : r1, c0 + 1 : c1] = int(legend_color)
    return out


def _generate_attempt() -> np.ndarray | None:
    room_count = random.choice(_GEN_ROOM_COUNT_CHOICES)
    legend_count = _legend_count(room_count)
    legend_ranks = random.sample(range(1, room_count + 1), legend_count)

    height = random.randint(*_GEN_SIZE_RANGE)
    width = random.randint(*_GEN_SIZE_RANGE)
    top_clearance = max(legend_ranks) + 2
    if width < 2 * legend_count + 10:
        return None

    rooms = _place_rooms(height, width, room_count, top_clearance)
    if rooms is None:
        return None

    counts = _choose_counts(rooms)
    if counts is None:
        return None

    colors = random.sample(range(10), 3 + legend_count)
    bg = int(colors[0])
    contour_color = int(colors[1])
    dot_color = int(colors[2])
    legend_colors = [int(color) for color in colors[3:]]

    grid = np.full((height, width), bg, dtype=int)
    legend_specs = list(zip(legend_colors, legend_ranks))
    random.shuffle(legend_specs)
    for idx, (legend_color, rank) in enumerate(legend_specs):
        c0 = _GEN_FAR_LEFT + idx * (_GEN_BAR_WIDTH + _GEN_BAR_GAP)
        grid[:rank, c0 : c0 + _GEN_BAR_WIDTH] = int(legend_color)

    for bbox in rooms:
        _draw_frame(grid, bbox, contour_color)

    for bbox, dot_count in zip(rooms, counts):
        cells = _dot_candidates(bbox, dot_count)
        if cells is None:
            return None
        for rr, cc in random.sample(cells, dot_count):
            grid[rr, cc] = int(dot_color)

    expected = _expected_output(grid, rooms, counts, legend_specs)

    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid puzzle input")
