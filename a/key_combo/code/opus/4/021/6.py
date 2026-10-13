import importlib.util
import random
from collections import defaultdict
from pathlib import Path

import numpy as np


def _load_arc_utils():
    path = Path("/home/lixinyu/snap/new/solvers/arc_utils.py")
    spec = importlib.util.spec_from_file_location("arc_utils_shared_021_6", path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)
    return module


_ARC = _load_arc_utils()
Color = _ARC.Color
find_components = _ARC.find_components
bbox_coords = _ARC.bbox_coords

BG = Color.BLACK
DASH = Color.BLUE
OBJECT_COLORS = [Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]

_SHAPE_MASKS = [
    np.array([[1, 1], [1, 0]], dtype=int),
    np.array([[1, 1], [0, 1]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1], [1, 1]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[1, 1, 1], [0, 1, 1]], dtype=int),
    np.array([[1, 1], [1, 0], [1, 0]], dtype=int),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=int),
]
_LOW_SHAPE_MASKS = [mask for mask in _SHAPE_MASKS if mask.shape[0] == 2]


def _mask_to_coords(mask: np.ndarray) -> list[tuple[int, int]]:
    return [tuple(map(int, rc)) for rc in np.argwhere(mask > 0)]


def _bottom_profile(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    bottom_by_col: dict[int, int] = {}
    for r, c in coords:
        bottom_by_col[c] = max(bottom_by_col.get(c, -1), int(r))
    return [(int(r), int(c)) for c, r in sorted(bottom_by_col.items())]


_SHAPES = []
for mask in _SHAPE_MASKS:
    coords = _mask_to_coords(mask)
    _SHAPES.append(
        {
            "coords": coords,
            "height": int(mask.shape[0]),
            "width": int(mask.shape[1]),
            "bottom": _bottom_profile(coords),
            "bottom_max": max(r for r, _c in coords),
            "area": len(coords),
        }
    )
_LOW_SHAPES = [shape for shape in _SHAPES if shape["height"] == 2]


def _dash_rows(grid: np.ndarray) -> list[dict]:
    rows = []
    for r in range(grid.shape[0]):
        blue_cols = np.flatnonzero(grid[r] == DASH)
        if blue_cols.size == 0:
            continue
        left = int(blue_cols[0])
        right = int(blue_cols[-1])
        gaps = [
            c
            for c in range(left + 1, right)
            if int(grid[r, c]) == BG and int(grid[r, c - 1]) == DASH and int(grid[r, c + 1]) == DASH
        ]
        rows.append(
            {
                "row": int(r),
                "left": left,
                "right": right,
                "blue_cols": [int(c) for c in blue_cols],
                "gap_cols": gaps,
            }
        )
    return rows


def _extract_objects(grid: np.ndarray) -> list[dict]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BG, DASH)]
    objects = find_components(
        grid,
        bg=BG,
        colors=colors,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    out = []
    for obj in objects:
        coords = [(int(r), int(c)) for r, c in obj["coords"]]
        r0, c0, r1, c1 = bbox_coords(obj["coords"])
        out.append(
            {
                "color": int(obj["color"]),
                "coords": coords,
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
            }
        )
    out.sort(key=lambda obj: (obj["bbox"][0], obj["bbox"][1], obj["color"]))
    return out


def _landing_for_object(coords: list[tuple[int, int]], dash_rows: list[dict]) -> tuple[int, int]:
    bottoms = _bottom_profile(coords)
    best_shift = None
    best_row = None
    for br, bc in bottoms:
        for dash in dash_rows:
            delta = int(dash["row"]) - int(br) - 1
            if delta < 0:
                continue
            if int(bc) in dash["blue_cols"]:
                if best_shift is None or delta < best_shift or (delta == best_shift and dash["row"] < best_row):
                    best_shift = int(delta)
                    best_row = int(dash["row"])
                break
    assert best_shift is not None and best_row is not None, "every object must land on a dashed row"
    return best_shift, best_row


def _translate(coords: list[tuple[int, int]], dr: int) -> list[tuple[int, int]]:
    return [(int(r + dr), int(c)) for r, c in coords]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop each colored object onto the first dashed blue row it can touch, then color that row's one-cell gaps with the landed-object colors.

    Input structure:
    - The background is black.
    - Several horizontal blue dashed rows lie at distinct heights; each row consists of blue pixels separated by single black gaps.
    - A few small monochrome non-blue objects float above one or more dashed rows.

    Rule:
    - Move every colored object straight downward until one of its bottom-profile pixels is exactly one row above a blue pixel on the first dashed row encountered below it.
    - Erase the old object positions and draw the translated objects in their landing positions.
    - For each dashed row, collect the objects that land on it, sort them by the leftmost landed column, and fill the row's single-pixel black gaps from left to right by cycling through those object colors.
    - Leave the blue dashed pixels unchanged; dashed rows with no landed objects keep their gaps black.

    Key insight:
    - Gap colors are determined indirectly by gravity: only the objects physically intercepted by a dashed row contribute colors, and multi-object rows assign those colors cyclically in landing order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    dash_rows = _dash_rows(g)
    assert dash_rows
    objects = _extract_objects(g)
    assert objects

    out = g.copy()
    for obj in objects:
        for r, c in obj["coords"]:
            out[r, c] = BG

    landed_by_row: dict[int, list[dict]] = defaultdict(list)
    occupied = set()
    for obj in objects:
        shift, landing_row = _landing_for_object(obj["coords"], dash_rows)
        moved = _translate(obj["coords"], shift)
        for r, c in moved:
            assert (r, c) not in occupied
            occupied.add((r, c))
            out[r, c] = obj["color"]
        c0 = min(c for _r, c in moved)
        landed_by_row[int(landing_row)].append({"color": obj["color"], "left": int(c0)})

    for dash in dash_rows:
        landed = landed_by_row.get(int(dash["row"]), [])
        if not landed:
            continue
        landed.sort(key=lambda item: (item["left"], item["color"]))
        colors = [item["color"] for item in landed]
        for idx, c in enumerate(dash["gap_cols"]):
            out[int(dash["row"]), int(c)] = int(colors[idx % len(colors)])

    return out


def _split_total(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    dividers = sorted(random.sample(range(total + parts - 1), parts - 1))
    out = []
    prev = -1
    for cut in dividers + [total + parts - 1]:
        out.append(int(cut - prev - 1))
        prev = cut
    return out


def _sample_dash_span(width: int) -> tuple[int, int]:
    left = random.randint(1, 2)
    max_len = width - left - 1
    if max_len % 2 == 0:
        max_len -= 1
    min_len = max(9, max_len - 4)
    if min_len % 2 == 0:
        min_len += 1
    if min_len > max_len:
        min_len = max_len
    choices = [length for length in range(min_len, max_len + 1, 2)]
    length = random.choice(choices)
    return int(left), int(left + length - 1)


def _sample_dash_rows(height: int, count: int) -> list[int]:
    min_row = 2
    max_row = height - 2
    base_gap = 5
    slack = max_row - min_row - base_gap * (count - 1)
    assert slack >= 0
    extras = _split_total(slack, count + 1)
    rows = []
    row = min_row + extras[0]
    for idx in range(count):
        if idx > 0:
            row += base_gap + extras[idx]
        rows.append(int(row))
    return rows


def _dash_blue_cols(left: int, right: int) -> list[int]:
    return [int(c) for c in range(left, right + 1, 2)]


def _dash_gap_cols(left: int, right: int) -> list[int]:
    return [int(c) for c in range(left + 1, right, 2)]


def _choose_assignments(n_objects: int, n_lines: int) -> list[int]:
    assert n_lines >= 3
    if n_objects == 2:
        return [0, 0]
    if n_objects == 3:
        return random.choice([[0, 0, 0], [0, 0, 1]])
    return random.choice([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 1]])


def _expanded(coords: list[tuple[int, int]], shape: tuple[int, int], pad: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    out = set()
    for r, c in coords:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _render_input(plan: dict) -> np.ndarray:
    grid = np.zeros(plan["shape"], dtype=int)
    for dash in plan["lines"]:
        r = dash["row"]
        for c in dash["blue_cols"]:
            grid[r, c] = DASH
    for obj in plan["objects"]:
        for r, c in obj["input_cells"]:
            grid[r, c] = obj["color"]
    return grid


def _render_output(plan: dict) -> np.ndarray:
    grid = np.zeros(plan["shape"], dtype=int)
    landed_by_row: dict[int, list[dict]] = defaultdict(list)
    for dash in plan["lines"]:
        r = dash["row"]
        for c in dash["blue_cols"]:
            grid[r, c] = DASH
    for obj in plan["objects"]:
        for r, c in obj["final_cells"]:
            grid[r, c] = obj["color"]
        landed_by_row[int(obj["target_row"])].append({"left": obj["final_left"], "color": obj["color"]})
    for dash in plan["lines"]:
        row = int(dash["row"])
        landed = landed_by_row.get(row, [])
        if not landed:
            continue
        landed.sort(key=lambda item: (item["left"], item["color"]))
        colors = [item["color"] for item in landed]
        for idx, c in enumerate(dash["gap_cols"]):
            grid[row, int(c)] = int(colors[idx % len(colors)])
    return grid


def _pick_object(
    target_idx: int,
    lines: list[dict],
    final_blocked: set[tuple[int, int]],
    input_blocked: set[tuple[int, int]],
    shape_hw: tuple[int, int],
) -> dict | None:
    target = lines[target_idx]
    prev_row = None if target_idx == 0 else int(lines[target_idx - 1]["row"])
    shape_pool = _LOW_SHAPES if target_idx > 0 else _SHAPES

    for _ in range(400):
        shape = random.choice(shape_pool)
        final_top = int(target["row"]) - 1 - int(shape["bottom_max"])
        if target_idx > 0 and final_top < prev_row + 2:
            continue

        support_r, support_c = random.choice(shape["bottom"])
        support_col = random.choice(target["blue_cols"])
        final_left = int(support_col - support_c)
        if final_left < 0 or final_left + shape["width"] > shape_hw[1]:
            continue

        final_cells = [(final_top + r, final_left + c) for r, c in shape["coords"]]
        if any(r < 0 or r >= shape_hw[0] or c < 0 or c >= shape_hw[1] for r, c in final_cells):
            continue
        if final_blocked.intersection(final_cells):
            continue

        min_input_top = 0 if target_idx == 0 else prev_row + 2
        max_input_top = final_top - 1
        if min_input_top > max_input_top:
            continue
        input_top = random.randint(min_input_top, max_input_top)
        lift = final_top - input_top
        input_cells = [(r - lift, c) for r, c in final_cells]
        if any(r < 0 or r >= shape_hw[0] or c < 0 or c >= shape_hw[1] for r, c in input_cells):
            continue
        if input_blocked.intersection(input_cells):
            continue

        touches_dash = False
        for r, c in input_cells:
            if r + 1 < shape_hw[0] and any((r + 1) == int(line["row"]) and c in line["blue_cols"] for line in lines):
                touches_dash = True
                break
        if touches_dash:
            continue

        return {
            "shape": shape,
            "target_row": int(target["row"]),
            "target_idx": int(target_idx),
            "final_top": int(final_top),
            "final_left": int(final_left),
            "input_top": int(input_top),
            "lift": int(lift),
            "final_cells": [(int(r), int(c)) for r, c in final_cells],
            "input_cells": [(int(r), int(c)) for r, c in input_cells],
        }
    return None


def _oracle_landing_row(input_cells: list[tuple[int, int]], lines: list[dict]) -> int:
    bottoms: dict[int, int] = {}
    for r, c in input_cells:
        bottoms[c] = max(bottoms.get(c, -1), int(r))
    best = None
    for c, r in bottoms.items():
        for line in lines:
            delta = int(line["row"]) - int(r) - 1
            if delta < 0:
                continue
            if int(c) in line["blue_cols"]:
                if best is None or delta < best[0] or (delta == best[0] and line["row"] < best[1]):
                    best = (int(delta), int(line["row"]))
                break
    assert best is not None
    return int(best[1])


def _validate_plan(plan: dict, inp: np.ndarray, out: np.ndarray) -> bool:
    if inp.shape != out.shape or inp.ndim != 2 or max(inp.shape) > 30:
        return False

    lines = plan["lines"]
    objects = plan["objects"]
    if not objects or not lines:
        return False

    landed_counts = defaultdict(int)
    multi_land = False
    empty_land = False
    cycle_visible = False
    for obj in objects:
        if _oracle_landing_row(obj["input_cells"], lines) != obj["target_row"]:
            return False
        landed_counts[obj["target_row"]] += 1
    for line in lines:
        row = int(line["row"])
        count = landed_counts.get(row, 0)
        if count >= 2:
            multi_land = True
            if len(line["gap_cols"]) > count:
                cycle_visible = True
        if count == 0:
            empty_land = True
        if not any(min(r for r, _c in obj["input_cells"]) < row for obj in objects):
            return False
    if not multi_land or not empty_land or not cycle_visible:
        return False

    for obj in objects:
        for r, c in obj["input_cells"]:
            if inp[r, c] != obj["color"]:
                return False
            if r + 1 < inp.shape[0] and inp[r + 1, c] == DASH:
                return False
    try:
        solved = main(inp)
    except Exception:
        return False
    return np.array_equal(solved, out)


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(15, 20)

    feasible_counts = [count for count in range(3, 6) if 2 + 5 * (count - 1) <= height - 2]
    if not feasible_counts:
        return None
    n_lines = random.choice(feasible_counts)
    n_objects = random.randint(2, 4)
    assignments = _choose_assignments(n_objects, n_lines)
    colors = random.sample(OBJECT_COLORS, n_objects)

    rows = _sample_dash_rows(height, n_lines)
    lines = []
    for row in rows:
        left, right = _sample_dash_span(width)
        lines.append(
            {
                "row": int(row),
                "left": int(left),
                "right": int(right),
                "blue_cols": _dash_blue_cols(left, right),
                "gap_cols": _dash_gap_cols(left, right),
            }
        )

    final_blocked: set[tuple[int, int]] = set()
    input_blocked: set[tuple[int, int]] = set()
    objects = []
    for target_idx, color in zip(assignments, colors):
        placed = _pick_object(target_idx, lines, final_blocked, input_blocked, (height, width))
        if placed is None:
            return None
        placed["color"] = int(color)
        objects.append(placed)
        final_blocked.update(_expanded(placed["final_cells"], (height, width), pad=1))
        input_blocked.update(_expanded(placed["input_cells"], (height, width), pad=1))

    plan = {"shape": (height, width), "lines": lines, "objects": objects}
    inp = _render_input(plan)
    out = _render_output(plan)
    if not _validate_plan(plan, inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 6")
