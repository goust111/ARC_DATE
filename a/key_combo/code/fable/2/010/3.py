from __future__ import annotations

from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "arc_utils_fable_2_010_3",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
bbox_h = _ARC_UTILS.bbox_h
bbox_w = _ARC_UTILS.bbox_w
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _has_adjacency(coords: np.ndarray, mask: np.ndarray) -> bool:
    height, width = mask.shape
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        if rr > 0 and mask[rr - 1, cc]:
            return True
        if rr + 1 < height and mask[rr + 1, cc]:
            return True
        if cc > 0 and mask[rr, cc - 1]:
            return True
        if cc + 1 < width and mask[rr, cc + 1]:
            return True
    return False


def _shift_coords(coords: np.ndarray, dr: int, dc: int, steps: int) -> np.ndarray:
    return coords + np.array([int(dr) * int(steps), int(dc) * int(steps)], dtype=int)


def _slide_candidate(
    coords: np.ndarray,
    target_mask: np.ndarray,
    obstacle_mask: np.ndarray,
    shape: tuple[int, int],
    dr: int,
    dc: int,
) -> tuple[int, np.ndarray] | None:
    height, width = shape
    for steps in range(1, max(height, width) + 1):
        shifted = _shift_coords(coords, dr, dc, steps)
        rows = shifted[:, 0]
        cols = shifted[:, 1]
        if int(rows.min()) < 0 or int(cols.min()) < 0 or int(rows.max()) >= height or int(cols.max()) >= width:
            return None
        if obstacle_mask[rows, cols].any():
            return None
        if _has_adjacency(shifted, target_mask):
            return int(steps), shifted
    return None


def _find_legend(grid: np.ndarray, bg: int) -> tuple[np.ndarray, list[int]]:
    height, width = grid.shape
    candidates = []
    for component in find_components(grid, bg=bg, connectivity=4, monochromatic=False):
        box = component["bbox"]
        r0, c0, r1, c1 = box
        if bbox_w(box) != 1 or int(component["area"]) < 2:
            continue
        if not ((r0 == 0 or r1 == height - 1) and (c0 == 0 or c1 == width - 1)):
            continue
        sep_row = r1 + 1 if r0 == 0 else r0 - 1
        if not (0 <= sep_row < height) or np.any(grid[sep_row, :] != bg):
            continue
        colors = [int(grid[r, c]) for r, c in component["coords"]]
        if any(color in (bg, Color.GREY) for color in colors):
            continue
        if len(set(colors)) != len(colors):
            continue
        candidates.append(component)

    assert len(candidates) == 1, "expected exactly one corner legend"
    legend = candidates[0]
    reverse = int(legend["bbox"][2]) == height - 1
    ordered_cells = sorted(
        (tuple(map(int, rc)) for rc in legend["coords"]),
        key=lambda rc: rc[0],
        reverse=reverse,
    )
    legend_mask = np.zeros_like(grid, dtype=bool)
    for r, c in ordered_cells:
        legend_mask[r, c] = True
    legend_colors = [int(grid[r, c]) for r, c in ordered_cells]
    return legend_mask, legend_colors


def _find_bar_mask(grid: np.ndarray, bg: int) -> np.ndarray:
    grey_components = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[Color.GREY],
    )
    assert len(grey_components) == 1, "expected exactly one grey attractor bar"
    return _mask_from_coords(grid.shape, grey_components[0]["coords"])


def _find_blocks(grid: np.ndarray, bg: int, legend_mask: np.ndarray, legend_colors: list[int]) -> dict[int, np.ndarray]:
    blocks: dict[int, np.ndarray] = {}
    for color in legend_colors:
        matches = []
        for component in find_components(
            grid,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[color],
        ):
            coords = component["coords"]
            if legend_mask[coords[:, 0], coords[:, 1]].all():
                continue
            matches.append(coords.copy())
        assert len(matches) == 1, f"expected one source block for color {color}"
        blocks[int(color)] = matches[0]
    return blocks


def _solve_with_trace(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = int(get_bg(g))
    legend_mask, legend_colors = _find_legend(g, bg)
    bar_mask = _find_bar_mask(g, bg)
    blocks = _find_blocks(g, bg, legend_mask, legend_colors)

    state = g.copy()
    parked_mask = bar_mask.copy()
    parked_blocks_mask = np.zeros_like(g, dtype=bool)
    trace: list[dict] = []

    for color in legend_colors:
        coords = blocks[int(color)]
        assert not _has_adjacency(coords, parked_mask), "source block must start away from the aggregate"

        obstacle_mask = state != bg
        obstacle_mask[coords[:, 0], coords[:, 1]] = False

        candidates = []
        for direction, (dr, dc) in DIRS.items():
            candidate = _slide_candidate(coords, parked_mask, obstacle_mask, g.shape, dr, dc)
            if candidate is None:
                continue
            steps, shifted = candidate
            candidates.append((int(steps), direction, shifted))

        assert candidates, f"block color {color} cannot reach the aggregate"
        candidates.sort(key=lambda item: item[0])
        assert len(candidates) == 1 or candidates[0][0] < candidates[1][0], "movement direction must be unique"

        steps, direction, shifted = candidates[0]
        state[coords[:, 0], coords[:, 1]] = bg
        state[shifted[:, 0], shifted[:, 1]] = int(color)
        blocks[int(color)] = shifted

        touches_bar = _has_adjacency(shifted, bar_mask)
        touches_parked = _has_adjacency(shifted, parked_blocks_mask)
        parked_mask[shifted[:, 0], shifted[:, 1]] = True
        parked_blocks_mask[shifted[:, 0], shifted[:, 1]] = True
        trace.append(
            {
                "color": int(color),
                "direction": direction,
                "distance": int(steps),
                "touches_bar": bool(touches_bar),
                "touches_parked": bool(touches_parked),
                "coords": shifted.copy(),
            }
        )

    return state, trace


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each colored block toward the grey bar in legend order, letting earlier arrivals become later stopping faces.

    Input structure
    - The background is black.
    - One grey horizontal or vertical strip is the attractor.
    - Several separated solid colored blocks appear in the main scene, with exactly one source block for each legend color.
    - A vertical multicolor legend sits in one corner and is separated from the scene by an empty row.

    Rule
    - Read the legend from the corner end outward: top-to-bottom for a top-corner legend and bottom-to-top for a bottom-corner legend.
    - For each listed color, move the matching block in the unique cardinal direction that reaches 4-neighbor contact with the current aggregate (grey bar plus already parked blocks) in the fewest unobstructed steps.
    - While one block moves, the legend and every other still-visible block remain fixed obstacles; the moving block stops at its first non-overlapping contact with the aggregate.
    - Keep the legend and grey bar unchanged, clear each source location, and paint each block only at its parked location.

    Key insight
    The corner color strip is a time axis rather than a palette: once a block docks, it joins the attractor and can stop later blocks earlier than the grey bar alone would.
    """
    out, _trace = _solve_with_trace(grid)
    return out


def _paint_rect(grid: np.ndarray, rect: tuple[int, int, int, int], color: int) -> None:
    r0, c0, h, w = rect
    grid[r0 : r0 + h, c0 : c0 + w] = int(color)


def _random_composition(total: int, parts: int) -> list[int]:
    pieces = [1] * int(parts)
    for _ in range(int(total) - int(parts)):
        pieces[random.randrange(parts)] += 1
    random.shuffle(pieces)
    return pieces


def _interleave_lane_blocks(lane_blocks: list[list[dict]]) -> list[dict]:
    queues = [blocks[:] for blocks in lane_blocks]
    ordered: list[dict] = []
    while any(queues):
        available = [index for index, blocks in enumerate(queues) if blocks]
        lane_index = random.choice(available)
        ordered.append(queues[lane_index].pop(0))
    return ordered


def _draw_legend(grid: np.ndarray, colors: list[int], *, top: bool, left: bool) -> None:
    col = 0 if left else grid.shape[1] - 1
    if top:
        rows = list(range(len(colors)))
        ordered_colors = colors
    else:
        rows = list(range(grid.shape[0] - len(colors), grid.shape[0]))
        ordered_colors = list(reversed(colors))
    for r, color in zip(rows, ordered_colors):
        grid[int(r), int(col)] = int(color)


def _build_horizontal_candidate(num_blocks: int, legend_top: bool, legend_left: bool) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    lane_count = random.randint(2, min(3, num_blocks - 1))
    counts = _random_composition(num_blocks, lane_count)
    lanes = []
    for count in counts:
        lanes.append(
            {
                "count": int(count),
                "side": random.choice(["up", "down"]),
                "h": random.randint(1, 2),
                "w": random.randint(1, 3),
                "start_gap": random.randint(2, 4),
                "inter_gap": random.randint(1, 2),
            }
        )

    if len({lane["side"] for lane in lanes}) == 1 and lane_count > 1 and random.random() < 0.8:
        lanes[random.randrange(lane_count)]["side"] = "down" if lanes[0]["side"] == "up" else "up"

    gaps = [random.randint(1, 2) for _ in range(lane_count - 1)]
    bar_len = sum(lane["w"] for lane in lanes) + sum(gaps)
    left_pad = random.randint(2, 4)
    right_pad = random.randint(2, 4)
    width = left_pad + bar_len + right_pad

    top_need = max(
        (
            lane["start_gap"] + lane["count"] * lane["h"] + (lane["count"] - 1) * lane["inter_gap"]
            for lane in lanes
            if lane["side"] == "up"
        ),
        default=0,
    )
    bottom_need = max(
        (
            lane["start_gap"] + lane["count"] * lane["h"] + (lane["count"] - 1) * lane["inter_gap"]
            for lane in lanes
            if lane["side"] == "down"
        ),
        default=0,
    )
    slack_up = random.randint(1, 3)
    slack_down = random.randint(1, 3)
    scene_height = top_need + slack_up + 1 + bottom_need + slack_down
    height = scene_height + num_blocks + 1

    if height > 30 or width > 30:
        return None

    scene_top = num_blocks + 1 if legend_top else 0
    bar_row = scene_top + top_need + slack_up
    bar_col = left_pad

    cursor = bar_col
    for index, lane in enumerate(lanes):
        lane["c0"] = cursor
        cursor += lane["w"]
        if index < len(gaps):
            cursor += gaps[index]

    lane_blocks: list[list[dict]] = []
    for lane_index, lane in enumerate(lanes):
        blocks = []
        for rank in range(lane["count"]):
            if lane["side"] == "up":
                src_top = bar_row - lane["start_gap"] - (rank + 1) * lane["h"] - rank * lane["inter_gap"]
                dst_top = bar_row - (rank + 1) * lane["h"]
                move_dir = "down"
            else:
                src_top = bar_row + lane["start_gap"] + 1 + rank * (lane["h"] + lane["inter_gap"])
                dst_top = bar_row + 1 + rank * lane["h"]
                move_dir = "up"
            blocks.append(
                {
                    "lane": lane_index,
                    "rank": rank,
                    "move_dir": move_dir,
                    "src": (int(src_top), int(lane["c0"]), int(lane["h"]), int(lane["w"])),
                    "dst": (int(dst_top), int(lane["c0"]), int(lane["h"]), int(lane["w"])),
                    "touches_prev_expected": bool(rank > 0),
                    "touches_bar_expected": bool(rank == 0),
                }
            )
        lane_blocks.append(blocks)

    ordered_blocks = _interleave_lane_blocks(lane_blocks)
    colors = random.sample(COLOR_POOL, num_blocks)
    for block, color in zip(ordered_blocks, colors):
        block["color"] = int(color)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    expected = np.full((height, width), Color.BLACK, dtype=int)
    grid[bar_row, bar_col : bar_col + bar_len] = Color.GREY
    expected[bar_row, bar_col : bar_col + bar_len] = Color.GREY

    for block in ordered_blocks:
        _paint_rect(grid, block["src"], block["color"])
        _paint_rect(expected, block["dst"], block["color"])

    _draw_legend(grid, [block["color"] for block in ordered_blocks], top=legend_top, left=legend_left)
    _draw_legend(expected, [block["color"] for block in ordered_blocks], top=legend_top, left=legend_left)
    return grid, expected, ordered_blocks


def _build_vertical_candidate(num_blocks: int, legend_top: bool, legend_left: bool) -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    lane_count = random.randint(2, min(3, num_blocks - 1))
    counts = _random_composition(num_blocks, lane_count)
    lanes = []
    for count in counts:
        lanes.append(
            {
                "count": int(count),
                "side": random.choice(["left", "right"]),
                "h": random.randint(1, 3),
                "w": random.randint(1, 2),
                "start_gap": random.randint(2, 4),
                "inter_gap": random.randint(1, 2),
            }
        )

    if len({lane["side"] for lane in lanes}) == 1 and lane_count > 1 and random.random() < 0.8:
        lanes[random.randrange(lane_count)]["side"] = "right" if lanes[0]["side"] == "left" else "left"

    gaps = [random.randint(1, 2) for _ in range(lane_count - 1)]
    bar_len = sum(lane["h"] for lane in lanes) + sum(gaps)
    top_pad = random.randint(1, 3)
    bottom_pad = random.randint(1, 3)
    scene_height = top_pad + bar_len + bottom_pad
    height = scene_height + num_blocks + 1

    left_need = max(
        (
            lane["start_gap"] + lane["count"] * lane["w"] + (lane["count"] - 1) * lane["inter_gap"]
            for lane in lanes
            if lane["side"] == "left"
        ),
        default=0,
    )
    right_need = max(
        (
            lane["start_gap"] + lane["count"] * lane["w"] + (lane["count"] - 1) * lane["inter_gap"]
            for lane in lanes
            if lane["side"] == "right"
        ),
        default=0,
    )
    slack_left = random.randint(1, 3)
    slack_right = random.randint(1, 3)
    width = left_need + slack_left + 1 + right_need + slack_right

    if height > 30 or width > 30:
        return None

    scene_top = num_blocks + 1 if legend_top else 0
    bar_row = scene_top + top_pad
    bar_col = left_need + slack_left

    cursor = bar_row
    for index, lane in enumerate(lanes):
        lane["r0"] = cursor
        cursor += lane["h"]
        if index < len(gaps):
            cursor += gaps[index]

    lane_blocks: list[list[dict]] = []
    for lane_index, lane in enumerate(lanes):
        blocks = []
        for rank in range(lane["count"]):
            if lane["side"] == "left":
                src_left = bar_col - lane["start_gap"] - (rank + 1) * lane["w"] - rank * lane["inter_gap"]
                dst_left = bar_col - (rank + 1) * lane["w"]
                move_dir = "right"
            else:
                src_left = bar_col + lane["start_gap"] + 1 + rank * (lane["w"] + lane["inter_gap"])
                dst_left = bar_col + 1 + rank * lane["w"]
                move_dir = "left"
            blocks.append(
                {
                    "lane": lane_index,
                    "rank": rank,
                    "move_dir": move_dir,
                    "src": (int(lane["r0"]), int(src_left), int(lane["h"]), int(lane["w"])),
                    "dst": (int(lane["r0"]), int(dst_left), int(lane["h"]), int(lane["w"])),
                    "touches_prev_expected": bool(rank > 0),
                    "touches_bar_expected": bool(rank == 0),
                }
            )
        lane_blocks.append(blocks)

    ordered_blocks = _interleave_lane_blocks(lane_blocks)
    colors = random.sample(COLOR_POOL, num_blocks)
    for block, color in zip(ordered_blocks, colors):
        block["color"] = int(color)

    grid = np.full((height, width), Color.BLACK, dtype=int)
    expected = np.full((height, width), Color.BLACK, dtype=int)
    grid[bar_row : bar_row + bar_len, bar_col] = Color.GREY
    expected[bar_row : bar_row + bar_len, bar_col] = Color.GREY

    for block in ordered_blocks:
        _paint_rect(grid, block["src"], block["color"])
        _paint_rect(expected, block["dst"], block["color"])

    _draw_legend(grid, [block["color"] for block in ordered_blocks], top=legend_top, left=legend_left)
    _draw_legend(expected, [block["color"] for block in ordered_blocks], top=legend_top, left=legend_left)
    return grid, expected, ordered_blocks


def _validate_candidate(grid: np.ndarray, expected: np.ndarray, ordered_blocks: list[dict]) -> bool:
    try:
        actual, trace = _solve_with_trace(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False
    if len(trace) != len(ordered_blocks):
        return False
    saw_mass_effect = False
    for block, step in zip(ordered_blocks, trace):
        if int(block["color"]) != int(step["color"]):
            return False
        if block["move_dir"] != step["direction"]:
            return False
        if bool(block["touches_prev_expected"]) != bool(step["touches_parked"]):
            return False
        if bool(block["touches_bar_expected"]) != bool(step["touches_bar"]):
            return False
        if block["touches_prev_expected"]:
            saw_mass_effect = True
    return saw_mass_effect


def _gen_attempt() -> np.ndarray | None:
    num_blocks = random.randint(3, 5)
    legend_top = random.choice([False, True])
    legend_left = random.choice([False, True])
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        built = _build_horizontal_candidate(num_blocks, legend_top, legend_left)
    else:
        built = _build_vertical_candidate(num_blocks, legend_top, legend_left)
    if built is None:
        return None

    grid, expected, ordered_blocks = built
    if not _validate_candidate(grid, expected, ordered_blocks):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid color-order attraction input")
