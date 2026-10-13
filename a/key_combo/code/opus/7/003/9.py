from __future__ import annotations

from collections import deque
from pathlib import Path
import random
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import find_components, get_bg, neighbors4


BG = 0
WALL = 1
MARKER_COLORS = (2, 3, 4, 6, 7, 8, 9)
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (16, 24)
BOX_SIZES = ((7, 7), (7, 8), (8, 7), (8, 8))
MAX_GEN_ATTEMPTS = 400

SHAPES = (
    {"cells": ((0, 0), (1, 0), (2, 0), (2, 1)), "marker": (2, 1)},
    {"cells": ((0, 0), (0, 1), (1, 1), (2, 1)), "marker": (0, 0)},
    {"cells": ((0, 1), (1, 0), (1, 1), (2, 1)), "marker": (1, 0)},
    {"cells": ((0, 0), (0, 1), (1, 1), (1, 2), (2, 1)), "marker": (1, 2)},
    {"cells": ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)), "marker": (0, 0)},
    {"cells": ((0, 1), (1, 0), (1, 1), (1, 2), (2, 1), (3, 1)), "marker": (0, 1)},
)


def _normalize_shape(cells: set[tuple[int, int]], marker: tuple[int, int]) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    r0 = min(r for r, _ in cells)
    c0 = min(c for _, c in cells)
    norm_cells = tuple(sorted((r - r0, c - c0) for r, c in cells))
    norm_marker = (marker[0] - r0, marker[1] - c0)
    return norm_cells, norm_marker


def _transform_shape(shape: dict) -> tuple[tuple[tuple[int, int], ...], tuple[int, int]]:
    cells = set(shape["cells"])
    marker = shape["marker"]

    if random.random() < 0.5:
        max_c = max(c for _, c in cells)
        cells = {(r, max_c - c) for r, c in cells}
        marker = (marker[0], max_c - marker[1])

    turns = random.randrange(4)
    for _ in range(turns):
        max_r = max(r for r, _ in cells)
        cells = {(c, max_r - r) for r, c in cells}
        marker = (marker[1], max_r - marker[0])

    return _normalize_shape(cells, marker)


def _shape_dims(cells: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in cells) + 1, max(c for _, c in cells) + 1


def _translate(cells: tuple[tuple[int, int], ...], top: int, left: int) -> tuple[tuple[int, int], ...]:
    return tuple((top + r, left + c) for r, c in cells)


def _bbox_cells(cells: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    return (
        min(r for r, _ in cells),
        min(c for _, c in cells),
        max(r for r, _ in cells),
        max(c for _, c in cells),
    )


def _in_box(cells: tuple[tuple[int, int], ...], box: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = box
    return all(r0 <= r <= r1 and c0 <= c <= c1 for r, c in cells)


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _sample_boxes(height: int, width: int, count: int) -> list[tuple[int, int, int, int]] | None:
    for _ in range(200):
        boxes: list[tuple[int, int, int, int]] = []
        dims = [random.choice(BOX_SIZES) for _ in range(count)]
        dims.sort(key=lambda x: x[0] * x[1], reverse=True)
        ok = True
        for h, w in dims:
            placed = False
            for _ in range(120):
                r0 = random.randint(2, height - h - 3)
                c0 = random.randint(2, width - w - 3)
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_boxes_separate(box, other) for other in boxes):
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            random.shuffle(boxes)
            return boxes
    return None


def _wall_cells(box: tuple[int, int, int, int], gap: tuple[int, int] | None) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = box
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    if gap is not None:
        cells.remove(gap)
    return cells


def _sample_gap(box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    side = random.choice(("top", "bottom", "left", "right"))
    if side == "top":
        return (r0, random.randint(c0 + 1, c1 - 1))
    if side == "bottom":
        return (r1, random.randint(c0 + 1, c1 - 1))
    if side == "left":
        return (random.randint(r0 + 1, r1 - 1), c0)
    return (random.randint(r0 + 1, r1 - 1), c1)


def _marker_exposed(cells: tuple[tuple[int, int], ...], marker: tuple[int, int], inner_box: tuple[int, int, int, int]) -> bool:
    cell_set = set(cells)
    for nr, nc in neighbors4(*marker):
        if inner_box[0] <= nr <= inner_box[2] and inner_box[1] <= nc <= inner_box[3] and (nr, nc) not in cell_set:
            return True
    return False


def _sample_sprite_placement(
    box: tuple[int, int, int, int],
    axis: str,
) -> dict | None:
    local_cells, local_marker = _transform_shape(random.choice(SHAPES))
    shape_h, shape_w = _shape_dims(local_cells)
    r0, c0, r1, c1 = box
    inner_box = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)

    feasible: list[tuple[int, int]] = []
    for top in range(inner_box[0], inner_box[2] - shape_h + 2):
        for left in range(inner_box[1], inner_box[3] - shape_w + 2):
            global_cells = _translate(local_cells, top, left)
            marker = (top + local_marker[0], left + local_marker[1])
            if _marker_exposed(global_cells, marker, inner_box):
                feasible.append((top, left))
    random.shuffle(feasible)

    for final_top, final_left in feasible:
        final_cells = _translate(local_cells, final_top, final_left)
        final_marker = (final_top + local_marker[0], final_left + local_marker[1])
        initial_choices = []
        for top, left in feasible:
            if axis == "H" and top == final_top and left != final_left:
                initial_choices.append((top, left))
            if axis == "V" and left == final_left and top != final_top:
                initial_choices.append((top, left))
        random.shuffle(initial_choices)
        for init_top, init_left in initial_choices:
            init_cells = _translate(local_cells, init_top, init_left)
            init_marker = (init_top + local_marker[0], init_left + local_marker[1])
            if axis == "H" and abs(final_marker[1] - init_marker[1]) < 2:
                continue
            if axis == "V" and abs(final_marker[0] - init_marker[0]) < 2:
                continue
            if not (_in_box(init_cells, inner_box) and _in_box(final_cells, inner_box)):
                continue
            return {
                "local_cells": local_cells,
                "local_marker": local_marker,
                "initial_cells": init_cells,
                "final_cells": final_cells,
                "initial_marker": init_marker,
                "final_marker": final_marker,
                "box": box,
                "inner_box": inner_box,
                "axis": axis,
            }
    return None


def _flood_reaches_anchor(grid: np.ndarray, start: tuple[int, int], anchor: tuple[int, int]) -> bool:
    height, width = grid.shape
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if not (0 <= nr < height and 0 <= nc < width):
                continue
            pt = (int(nr), int(nc))
            if pt in seen:
                continue
            if pt == anchor or int(grid[nr, nc]) == BG:
                seen.add(pt)
                q.append(pt)
    return anchor in seen


def _find_anchors(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    height, width = grid.shape
    anchors: dict[int, tuple[int, int]] = {}
    for r in range(height):
        for c in range(width):
            if r not in (0, height - 1) and c not in (0, width - 1):
                continue
            color = int(grid[r, c])
            if color in (bg, WALL):
                continue
            if all(
                not (0 <= nr < height and 0 <= nc < width) or int(grid[nr, nc]) == bg
                for nr, nc in neighbors4(r, c)
            ):
                anchors[color] = (int(r), int(c))
    return anchors


def _extract_sprites(grid: np.ndarray, anchors: dict[int, tuple[int, int]], bg: int) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (bg, WALL)]
    comps = find_components(
        grid,
        bg=bg,
        colors=colors,
        connectivity=4,
        monochromatic=False,
        return_format="dict",
    )
    sprites: list[dict] = []
    for comp in comps:
        coords = tuple((int(r), int(c)) for r, c in comp["coords"])
        if len(coords) < 2:
            continue
        if any(r in (0, grid.shape[0] - 1) or c in (0, grid.shape[1] - 1) for r, c in coords):
            continue
        comp_colors = {int(grid[r, c]) for r, c in coords}
        marker_colors = sorted(color for color in comp_colors if color in anchors)
        if len(marker_colors) != 1:
            continue
        marker_color = marker_colors[0]
        marker_pixels = [(r, c) for r, c in coords if int(grid[r, c]) == marker_color]
        if len(marker_pixels) != 1:
            continue
        sprites.append(
            {
                "coords": coords,
                "marker_color": marker_color,
                "marker": marker_pixels[0],
                "anchor": anchors[marker_color],
            }
        )
    return sprites


def _solve_scene(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    anchors = _find_anchors(g, bg)
    sprites = _extract_sprites(g, anchors, bg)

    out = g.copy()
    for sprite in sprites:
        for r, c in sprite["coords"]:
            out[r, c] = bg

    moves: list[dict] = []
    for sprite in sprites:
        marker = sprite["marker"]
        anchor = sprite["anchor"]
        reachable = _flood_reaches_anchor(g, marker, anchor)
        ar, ac = anchor
        if ar in (0, g.shape[0] - 1) and ac not in (0, g.shape[1] - 1):
            dr, dc = 0, ac - marker[1]
            axis = "H"
        else:
            dr, dc = ar - marker[0], 0
            axis = "V"
        moved_cells = tuple((r + dr, c + dc) for r, c in sprite["coords"]) if reachable else sprite["coords"]
        moves.append(
            {
                "marker_color": sprite["marker_color"],
                "reachable": reachable,
                "axis": axis,
                "dr": int(dr if reachable else 0),
                "dc": int(dc if reachable else 0),
                "coords": moved_cells,
                "anchor": anchor,
            }
        )

    for move, sprite in zip(moves, sprites):
        colors = [int(g[r, c]) for r, c in sprite["coords"]]
        for (rr, cc), color in zip(move["coords"], colors):
            out[rr, cc] = color

    return out, moves


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each marked sprite only when its matching border anchor is visible through wall gaps.

    Input structure:
    - Black background contains 2-3 interior multi-color sprites, each with one unique marker-color pixel.
    - Blue wall segments form internal cages or barriers, and same-color singleton anchors sit on the border.
    - Each anchor is isolated on the boundary and matches exactly one sprite marker color.

    Rule:
    - Pair every sprite with the border anchor sharing its marker color.
    - Flood-fill from the sprite's marker pixel through black cells only, treating blue walls and all non-black non-anchor cells as blockers; the flood may step onto the matched anchor cell.
    - If the matched anchor is unreachable, keep the sprite fixed. If reachable, slide the whole sprite horizontally for top/bottom anchors or vertically for left/right anchors so the marker aligns with the anchor's column or row.
    - Preserve all walls and anchor pixels in the output.

    Key insight:
    - The walls do not physically stop the final slide; they only gate whether the sprite is allowed to use its matching anchor as an alignment target.
    """
    out, _moves = _solve_scene(grid)
    return out


def _sample_cases(count: int) -> list[dict]:
    if count == 2:
        cases = random.choice(
            (
                ({"axis": "H", "reachable": True}, {"axis": "V", "reachable": False}),
                ({"axis": "V", "reachable": True}, {"axis": "H", "reachable": False}),
            )
        )
        cases = [dict(case) for case in cases]
        random.shuffle(cases)
        return cases

    cases = [
        {"axis": "H", "reachable": True},
        {"axis": "V", "reachable": True},
        {"axis": random.choice(("H", "V")), "reachable": False},
    ]
    random.shuffle(cases)
    return cases


def _choose_body_colors(marker_colors: list[int]) -> list[int]:
    palette = [c for c in range(2, 10) if c not in marker_colors and c != WALL]
    random.shuffle(palette)
    return palette[: len(marker_colors)]


def _render_scene(height: int, width: int, sprites: list[dict], anchors: dict[int, tuple[int, int]], boxes: list[dict], moved: bool) -> np.ndarray:
    grid = np.zeros((height, width), dtype=int)

    for box in boxes:
        for r, c in _wall_cells(box["box"], box["gap"]):
            grid[r, c] = WALL

    for color, (r, c) in anchors.items():
        grid[r, c] = int(color)

    for sprite in sprites:
        cells = sprite["final_cells"] if moved and sprite["reachable"] else sprite["initial_cells"]
        for r, c in cells:
            grid[r, c] = int(sprite["body_color"])
        mr, mc = sprite["final_marker"] if moved and sprite["reachable"] else sprite["initial_marker"]
        grid[mr, mc] = int(sprite["marker_color"])

    return grid


def _anchors_are_isolated(grid: np.ndarray, anchors: dict[int, tuple[int, int]]) -> bool:
    height, width = grid.shape
    anchor_cells = set(anchors.values())
    for color, (r, c) in anchors.items():
        if int(grid[r, c]) != int(color):
            return False
        if r not in (0, height - 1) and c not in (0, width - 1):
            return False
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < height and 0 <= nc < width:
                if (nr, nc) in anchor_cells:
                    return False
                if int(grid[nr, nc]) != BG:
                    return False
    return True


def _scene_valid(inp: np.ndarray, expected: np.ndarray, specs: list[dict], count: int) -> bool:
    parsed_out, moves = _solve_scene(inp)
    if not np.array_equal(parsed_out, expected):
        return False
    if np.array_equal(inp, expected):
        return False

    by_color = {spec["marker_color"]: spec for spec in specs}
    if set(by_color) != {move["marker_color"] for move in moves}:
        return False

    reachables = 0
    blocked = 0
    moved_axes: set[str] = set()
    occupied: set[tuple[int, int]] = set()
    for move in moves:
        spec = by_color[move["marker_color"]]
        if bool(move["reachable"]) != bool(spec["reachable"]):
            return False
        if move["axis"] != spec["axis"]:
            return False
        if move["anchor"] != spec["anchor"]:
            return False
        if move["reachable"]:
            reachables += 1
            moved_axes.add(move["axis"])
            if move["axis"] == "H" and move["dc"] == 0:
                return False
            if move["axis"] == "V" and move["dr"] == 0:
                return False
        else:
            blocked += 1
            if spec["would_move_by"] == 0:
                return False
        for cell in move["coords"]:
            if cell in occupied:
                return False
            occupied.add(cell)

    if reachables < 1 or blocked < 1:
        return False
    if count == 3 and moved_axes != {"H", "V"}:
        return False

    return True


def _sample_scene() -> tuple[np.ndarray, np.ndarray, list[dict]] | None:
    count = 3 if random.random() < 0.7 else 2
    height = random.randint(*GRID_H_RANGE)
    width = random.randint(*GRID_W_RANGE)
    boxes_raw = _sample_boxes(height, width, count)
    if boxes_raw is None:
        return None

    marker_colors = random.sample(MARKER_COLORS, count)
    body_colors = _choose_body_colors(marker_colors)
    cases = _sample_cases(count)

    sprite_specs: list[dict] = []
    wall_specs: list[dict] = []
    anchors: dict[int, tuple[int, int]] = {}

    for box, marker_color, body_color, case in zip(boxes_raw, marker_colors, body_colors, cases):
        placement = _sample_sprite_placement(box, case["axis"])
        if placement is None:
            return None

        if case["axis"] == "H":
            anchor = (random.choice((0, height - 1)), placement["final_marker"][1])
            would_move_by = abs(placement["final_marker"][1] - placement["initial_marker"][1])
        else:
            anchor = (placement["final_marker"][0], random.choice((0, width - 1)))
            would_move_by = abs(placement["final_marker"][0] - placement["initial_marker"][0])

        if anchor in anchors.values():
            return None

        sprite_specs.append(
            {
                "marker_color": marker_color,
                "body_color": body_color,
                "axis": case["axis"],
                "reachable": case["reachable"],
                "initial_cells": placement["initial_cells"],
                "final_cells": placement["final_cells"],
                "initial_marker": placement["initial_marker"],
                "final_marker": placement["final_marker"],
                "anchor": anchor,
                "would_move_by": would_move_by,
            }
        )
        anchors[marker_color] = anchor
        wall_specs.append({"box": box, "gap": _sample_gap(box) if case["reachable"] else None})

    inp = _render_scene(height, width, sprite_specs, anchors, wall_specs, moved=False)
    expected = _render_scene(height, width, sprite_specs, anchors, wall_specs, moved=True)

    if not _anchors_are_isolated(inp, anchors):
        return None
    if not _scene_valid(inp, expected, sprite_specs, count):
        return None

    return inp, expected, sprite_specs


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        scene = _sample_scene()
        if scene is not None:
            return scene[0]
    raise RuntimeError("failed to generate a valid ARC scene")
