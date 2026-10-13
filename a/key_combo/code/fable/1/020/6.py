from __future__ import annotations

import itertools
import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_h, bbox_w, crop_bbox, find_components, get_bg, in_bounds, paste


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]

_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
_ACTIONS = ("L", "S", "R")


def _turn(direction: Coord, action: str) -> Coord:
    idx = _DIRS.index(tuple(direction))
    if action == "L":
        return _DIRS[(idx - 1) % 4]
    assert action == "R"
    return _DIRS[(idx + 1) % 4]


def _is_border_cell(cell: Coord, shape: tuple[int, int]) -> bool:
    r, c = cell
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _neighbors4(cell: Coord) -> list[Coord]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _box_coords(box: BBox, shape: tuple[int, int], margin: int = 0) -> set[Coord]:
    h, w = shape
    r0 = max(0, int(box[0]) - int(margin))
    c0 = max(0, int(box[1]) - int(margin))
    r1 = min(h - 1, int(box[2]) + int(margin))
    c1 = min(w - 1, int(box[3]) + int(margin))
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def _panel_info(grid: np.ndarray, component: dict, bg: int) -> dict | None:
    box = component["bbox"]
    if bbox_h(box) != 3 or bbox_w(box) != 5 or int(component["area"]) != 12:
        return None

    patch = crop_bbox(grid, box)
    border_color = int(component["color"])
    if not (
        np.all(patch[0, :] == border_color)
        and np.all(patch[2, :] == border_color)
        and int(patch[1, 0]) == border_color
        and int(patch[1, 4]) == border_color
    ):
        return None

    legend_row = [int(v) for v in patch[1, 1:4]]
    if len(set(legend_row)) != 3:
        return None
    if any(color in {bg, border_color} for color in legend_row):
        return None

    return {
        "bbox": tuple(map(int, box)),
        "border_color": border_color,
        "action_to_color": {"L": legend_row[0], "S": legend_row[1], "R": legend_row[2]},
        "color_to_action": {legend_row[0]: "L", legend_row[1]: "S", legend_row[2]: "R"},
    }


def _parse_panel(grid: np.ndarray, bg: int) -> dict:
    candidates = []
    for component in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        info = _panel_info(grid, component, bg)
        if info is not None:
            candidates.append(info)
    assert len(candidates) == 1, "expected exactly one 3x5 legend panel"
    return candidates[0]


def _outside_box_mask(shape: tuple[int, int], box: BBox) -> np.ndarray:
    mask = np.ones(shape, dtype=bool)
    r0, c0, r1, c1 = box
    mask[r0 : r1 + 1, c0 : c1 + 1] = False
    return mask


def _parse_walker(grid: np.ndarray, panel_box: BBox) -> tuple[Coord, Coord, Coord]:
    outside_panel = _outside_box_mask(grid.shape, panel_box)
    red = np.argwhere((grid == Color.RED) & outside_panel)
    grey = np.argwhere((grid == Color.GREY) & outside_panel)
    assert red.shape == (1, 2), "expected exactly one red head pixel"
    assert grey.shape == (1, 2), "expected exactly one grey tail pixel"

    head = tuple(map(int, red[0]))
    tail = tuple(map(int, grey[0]))
    direction = (head[0] - tail[0], head[1] - tail[1])
    assert abs(direction[0]) + abs(direction[1]) == 1, "walker head and tail must be orthogonally adjacent"
    return head, tail, direction


def _simulate(
    grid: np.ndarray,
    *,
    color_to_action: dict[int, str] | None = None,
) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    panel = _parse_panel(g, bg)
    head, tail, direction = _parse_walker(g, panel["bbox"])
    color_to_action = panel["color_to_action"] if color_to_action is None else dict(color_to_action)

    out = g.copy()
    r0, c0, r1, c1 = panel["bbox"]
    position = head
    path = [head]
    hits: list[dict] = []
    termination = "edge"

    while True:
        nr = int(position[0] + direction[0])
        nc = int(position[1] + direction[1])
        if not in_bounds(g.shape, nr, nc):
            break
        assert not (r0 <= nr <= r1 and c0 <= nc <= c1), "walker path must not enter the legend panel"

        position = (nr, nc)
        path.append(position)
        stepped_color = int(g[nr, nc])
        out[nr, nc] = Color.RED

        action = color_to_action.get(stepped_color)
        if action is None:
            continue
        hits.append({"pos": position, "action": action, "color": stepped_color})
        if action == "S":
            termination = "stop"
            break
        direction = _turn(direction, action)

    return out, {
        "bg": bg,
        "panel": panel,
        "head": head,
        "tail": tail,
        "path": path,
        "hits": hits,
        "termination": termination,
        "final_direction": direction,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the corner legend, then march the red-headed walker forward and repaint its executed route.

    Input structure
    - A dominant background surrounds one two-pixel walker: a red head and a gray tail whose offset defines the initial heading.
    - One corner contains a 3x5 single-color border panel; its inner row holds three distinct instruction colors in left, middle, right positions.
    - Elsewhere, single instruction pixels of exactly those three colors are scattered away from the legend panel.

    Rule
    - Read the legend row by position: left color means turn left, middle color means stop, right color means turn right.
    - Starting from the red head, move one cell at a time in the heading from tail to head and paint every visited cell red.
    - When the walker steps onto an instruction pixel, execute only that stepped-on pixel's decoded command: left/right rotate the heading by 90 degrees, while stop halts immediately after painting that cell; if the next step would leave the grid, halt at the edge instead.

    Key insight
    - The instruction colors have no fixed meaning across examples; only their positions inside the legend panel define the command, and only the instruction pixels actually touched by the route matter.
    """
    out, _trace = _simulate(grid)
    return out


# --------------- generator (o5x) ---------------

GEN_BG = Color.BLACK
GEN_HEAD = Color.RED
GEN_TAIL = Color.GREY
GEN_PANEL_SHAPE = (3, 5)
GEN_PANEL_MARGIN = 1
GEN_HEIGHT_RANGE = (12, 24)
GEN_WIDTH_RANGE = (12, 24)
GEN_TURN_COUNT_RANGE = (2, 4)
GEN_TURN_SEG_RANGE = (2, 6)
GEN_STOP_SEG_RANGE = (1, 5)
GEN_FINAL_STOP_PROB = 0.55
GEN_DISTRACTOR_COUNT_RANGE = (3, 6)
GEN_MIN_PATH_LEN = 8
GEN_MAX_ATTEMPTS = 700
GEN_COLOR_POOL = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
GEN_CORNERS = ("tl", "tr", "bl", "br")


def _panel_bbox(shape: tuple[int, int], corner: str) -> BBox:
    h, w = shape
    ph, pw = GEN_PANEL_SHAPE
    if corner == "tl":
        return 0, 0, ph - 1, pw - 1
    if corner == "tr":
        return 0, w - pw, ph - 1, w - 1
    if corner == "bl":
        return h - ph, 0, h - 1, pw - 1
    assert corner == "br"
    return h - ph, w - pw, h - 1, w - 1


def _make_panel(border_color: int, action_to_color: dict[str, int]) -> np.ndarray:
    panel = np.full(GEN_PANEL_SHAPE, int(border_color), dtype=int)
    panel[1, 1] = int(action_to_color["L"])
    panel[1, 2] = int(action_to_color["S"])
    panel[1, 3] = int(action_to_color["R"])
    return panel


def _sample_turn_actions() -> list[str]:
    count = random.randint(*GEN_TURN_COUNT_RANGE)
    actions = ["L", "R"]
    while len(actions) < count:
        actions.append(random.choice(("L", "R")))
    random.shuffle(actions)
    return actions


def _segment_cells(
    start: Coord,
    direction: Coord,
    steps: int,
    *,
    occupied: set[Coord],
    forbidden: set[Coord],
    shape: tuple[int, int],
) -> list[Coord] | None:
    cells: list[Coord] = []
    local_occupied = set(occupied)
    prev = start
    dr, dc = direction
    for _ in range(int(steps)):
        nxt = (int(prev[0] + dr), int(prev[1] + dc))
        if not in_bounds(shape, nxt[0], nxt[1]):
            return None
        if nxt in forbidden or nxt in local_occupied:
            return None
        for neighbor in _neighbors4(nxt):
            if neighbor in local_occupied and neighbor != prev:
                return None
        cells.append(nxt)
        local_occupied.add(nxt)
        prev = nxt
    return cells


def _build_path_recursive(
    *,
    current: Coord,
    direction: Coord,
    turn_actions: list[str],
    final_stop: bool,
    occupied: set[Coord],
    forbidden: set[Coord],
    shape: tuple[int, int],
) -> dict | None:
    if not turn_actions:
        if final_stop:
            lengths = list(range(GEN_STOP_SEG_RANGE[0], GEN_STOP_SEG_RANGE[1] + 1))
            random.shuffle(lengths)
            for steps in lengths:
                segment = _segment_cells(
                    current,
                    direction,
                    steps,
                    occupied=occupied,
                    forbidden=forbidden,
                    shape=shape,
                )
                if segment is None:
                    continue
                stop_cell = segment[-1]
                if _is_border_cell(stop_cell, shape):
                    continue
                return {"cells": segment, "instructions": [(stop_cell, "S")], "termination": "stop"}
            return None

        segment: list[Coord] = []
        local_occupied = set(occupied)
        prev = current
        dr, dc = direction
        while True:
            nxt = (int(prev[0] + dr), int(prev[1] + dc))
            if not in_bounds(shape, nxt[0], nxt[1]):
                break
            if nxt in forbidden or nxt in local_occupied:
                return None
            for neighbor in _neighbors4(nxt):
                if neighbor in local_occupied and neighbor != prev:
                    return None
            segment.append(nxt)
            local_occupied.add(nxt)
            prev = nxt
        if len(segment) < 2 or not _is_border_cell(segment[-1], shape):
            return None
        return {"cells": segment, "instructions": [], "termination": "edge"}

    action = turn_actions[0]
    lengths = list(range(GEN_TURN_SEG_RANGE[0], GEN_TURN_SEG_RANGE[1] + 1))
    random.shuffle(lengths)
    for steps in lengths:
        segment = _segment_cells(
            current,
            direction,
            steps,
            occupied=occupied,
            forbidden=forbidden,
            shape=shape,
        )
        if segment is None:
            continue
        joint = segment[-1]
        if _is_border_cell(joint, shape):
            continue

        new_direction = _turn(direction, action)
        future_probe = _segment_cells(
            joint,
            new_direction,
            1,
            occupied=occupied | set(segment),
            forbidden=forbidden,
            shape=shape,
        )
        if future_probe is None:
            continue

        rest = _build_path_recursive(
            current=joint,
            direction=new_direction,
            turn_actions=turn_actions[1:],
            final_stop=final_stop,
            occupied=occupied | set(segment),
            forbidden=forbidden,
            shape=shape,
        )
        if rest is None:
            continue
        return {
            "cells": segment + rest["cells"],
            "instructions": [(joint, action)] + rest["instructions"],
            "termination": rest["termination"],
        }
    return None


def _candidate_heads(shape: tuple[int, int], forbidden: set[Coord]) -> list[tuple[Coord, Coord]]:
    h, w = shape
    candidates = []
    cells = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    random.shuffle(cells)
    directions = list(_DIRS)
    for head in cells:
        if head in forbidden:
            continue
        random.shuffle(directions)
        for direction in directions:
            tail = (head[0] - direction[0], head[1] - direction[1])
            if not in_bounds(shape, tail[0], tail[1]):
                continue
            if tail in forbidden or _is_border_cell(tail, shape):
                continue
            candidates.append((head, direction))
    return candidates


def _place_distractors(
    grid: np.ndarray,
    *,
    path: list[Coord],
    forbidden: set[Coord],
    action_to_color: dict[str, int],
    tail: Coord,
) -> bool:
    path_set = set(path)
    free_cells = []
    for r in range(grid.shape[0]):
        for c in range(grid.shape[1]):
            cell = (r, c)
            if cell in forbidden or cell in path_set or cell == tail:
                continue
            if int(grid[r, c]) != GEN_BG:
                continue
            free_cells.append(cell)
    if len(free_cells) < 3:
        return False

    def path_distance(cell: Coord) -> int:
        return min(abs(cell[0] - pr) + abs(cell[1] - pc) for pr, pc in path)

    near = [cell for cell in free_cells if 1 <= path_distance(cell) <= 2]
    far = [cell for cell in free_cells if cell not in near]
    random.shuffle(near)
    random.shuffle(far)
    ordered = near + far

    chosen: list[Coord] = []
    used: set[Coord] = set()
    for action in _ACTIONS:
        for cell in ordered:
            if cell in used:
                continue
            chosen.append(cell)
            used.add(cell)
            grid[cell] = int(action_to_color[action])
            break
        else:
            return False

    extra_count = random.randint(*GEN_DISTRACTOR_COUNT_RANGE) - 3
    for cell in ordered:
        if extra_count <= 0:
            break
        if cell in used:
            continue
        grid[cell] = int(action_to_color[random.choice(_ACTIONS)])
        used.add(cell)
        extra_count -= 1
    return True


def _build_attempt() -> tuple[np.ndarray, dict] | None:
    shape = (random.randint(*GEN_HEIGHT_RANGE), random.randint(*GEN_WIDTH_RANGE))
    corner = random.choice(GEN_CORNERS)
    panel_box = _panel_bbox(shape, corner)
    forbidden = _box_coords(panel_box, shape, margin=GEN_PANEL_MARGIN)

    colors = random.sample(GEN_COLOR_POOL, 4)
    instruction_colors = colors[:3]
    border_color = colors[3]
    action_to_color = {"L": instruction_colors[0], "S": instruction_colors[1], "R": instruction_colors[2]}

    head_candidates = _candidate_heads(shape, forbidden)
    random.shuffle(head_candidates)
    turn_actions = _sample_turn_actions()
    final_stop = bool(random.random() < GEN_FINAL_STOP_PROB)

    for head, direction in head_candidates[:180]:
        tail = (head[0] - direction[0], head[1] - direction[1])
        build = _build_path_recursive(
            current=head,
            direction=direction,
            turn_actions=turn_actions,
            final_stop=final_stop,
            occupied={head, tail},
            forbidden=forbidden,
            shape=shape,
        )
        if build is None:
            continue

        path = [head] + build["cells"]
        if len(path) < GEN_MIN_PATH_LEN:
            continue

        grid = np.full(shape, GEN_BG, dtype=int)
        paste(grid, _make_panel(border_color, action_to_color), panel_box[0], panel_box[1])
        grid[head] = GEN_HEAD
        grid[tail] = GEN_TAIL
        for cell, action in build["instructions"]:
            grid[cell] = int(action_to_color[action])
        if not _place_distractors(grid, path=path, forbidden=forbidden, action_to_color=action_to_color, tail=tail):
            continue

        plan = {
            "path": path,
            "instructions": build["instructions"],
            "termination": build["termination"],
            "turn_actions": turn_actions,
            "action_to_color": action_to_color,
            "panel_box": panel_box,
            "head": head,
            "tail": tail,
        }
        return grid, plan
    return None


def _semantic_checks(grid: np.ndarray, plan: dict) -> bool:
    try:
        out, trace = _simulate(grid)
    except Exception:
        return False

    if trace["path"] != plan["path"]:
        return False
    actual_hits = [(entry["pos"], entry["action"]) for entry in trace["hits"]]
    if actual_hits != plan["instructions"]:
        return False
    if trace["termination"] != plan["termination"]:
        return False
    if len(trace["path"]) < GEN_MIN_PATH_LEN:
        return False

    turn_hits = [entry["action"] for entry in trace["hits"] if entry["action"] != "S"]
    if len(turn_hits) < 2 or set(turn_hits) != {"L", "R"}:
        return False

    panel = trace["panel"]
    instruction_colors = set(panel["action_to_color"].values())
    scatter_positions = {
        tuple(map(int, coord))
        for coord in np.argwhere(np.isin(grid, list(instruction_colors)))
        if not (
            panel["bbox"][0] <= int(coord[0]) <= panel["bbox"][2]
            and panel["bbox"][1] <= int(coord[1]) <= panel["bbox"][3]
        )
    }
    path_set = set(trace["path"])
    off_path = scatter_positions - path_set
    if len(off_path) < 3:
        return False
    if {int(grid[cell]) for cell in off_path} != instruction_colors:
        return False

    colors_in_order = [panel["action_to_color"][action] for action in _ACTIONS]
    for perm in itertools.permutations(_ACTIONS):
        if perm == _ACTIONS:
            continue
        alt_map = {color: perm[idx] for idx, color in enumerate(colors_in_order)}
        try:
            alt_out, _alt_trace = _simulate(grid, color_to_action=alt_map)
        except Exception:
            continue
        if np.array_equal(alt_out, out):
            return False

    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        built = _build_attempt()
        if built is None:
            continue
        grid, plan = built
        if _semantic_checks(grid, plan):
            return grid
    raise RuntimeError("failed to generate a valid instruction-walker example")
