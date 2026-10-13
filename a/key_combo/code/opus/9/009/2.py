from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import Color, find_components, get_bg


BG = Color.BLACK
COUNT_COLOR = Color.GREY
OBJ_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
SIDES = ("top", "bottom", "left", "right")
PATTERNS = (
    (1, 2),
    (1, 3),
    (1, 2, 2),
    (1, 2, 3),
    (1, 3, 3),
    (1, 2, 2, 3),
)
GENERATE_RETRIES = 400


def _component_center(component: dict) -> tuple[float, float]:
    coords = component["coords"]
    return float(coords[:, 0].mean()), float(coords[:, 1].mean())


def _classify_sides(components: list[dict]) -> tuple[dict[str, dict], tuple[float, float]]:
    centers = np.array([_component_center(component) for component in components], dtype=float)
    global_center = tuple(np.mean(centers, axis=0))
    side_map: dict[str, dict] = {}
    for component in components:
        r, c = _component_center(component)
        dr = r - global_center[0]
        dc = c - global_center[1]
        if abs(dr) == abs(dc):
            raise AssertionError("component must be clearly aligned to one cardinal side")
        if abs(dr) > abs(dc):
            side = "bottom" if dr > 0 else "top"
        else:
            side = "right" if dc > 0 else "left"
        if side in side_map:
            raise AssertionError("expected one object per side")
        side_map[side] = component
    if set(side_map) != set(SIDES):
        raise AssertionError("expected exactly four side objects")
    return side_map, global_center


def _tip_and_growth_profile(side: str, coords: np.ndarray) -> dict:
    if side == "top":
        tip = coords[coords[:, 0] == coords[:, 0].max()]
        if tip.shape[0] != 1:
            raise AssertionError("top object must have a unique inward tip")
        outer_row = int(coords[:, 0].min())
        cols = sorted(int(c) for c in coords[coords[:, 0] == outer_row][:, 1])
        return {"tip": tuple(map(int, tip[0])), "axis": "col", "line": int(tip[0, 1]), "cross": [(outer_row, c) for c in cols], "step": (-1, 0)}
    if side == "bottom":
        tip = coords[coords[:, 0] == coords[:, 0].min()]
        if tip.shape[0] != 1:
            raise AssertionError("bottom object must have a unique inward tip")
        outer_row = int(coords[:, 0].max())
        cols = sorted(int(c) for c in coords[coords[:, 0] == outer_row][:, 1])
        return {"tip": tuple(map(int, tip[0])), "axis": "col", "line": int(tip[0, 1]), "cross": [(outer_row, c) for c in cols], "step": (1, 0)}
    if side == "left":
        tip = coords[coords[:, 1] == coords[:, 1].max()]
        if tip.shape[0] != 1:
            raise AssertionError("left object must have a unique inward tip")
        outer_col = int(coords[:, 1].min())
        rows = sorted(int(r) for r in coords[coords[:, 1] == outer_col][:, 0])
        return {"tip": tuple(map(int, tip[0])), "axis": "row", "line": int(tip[0, 0]), "cross": [(r, outer_col) for r in rows], "step": (0, -1)}
    if side == "right":
        tip = coords[coords[:, 1] == coords[:, 1].min()]
        if tip.shape[0] != 1:
            raise AssertionError("right object must have a unique inward tip")
        outer_col = int(coords[:, 1].max())
        rows = sorted(int(r) for r in coords[coords[:, 1] == outer_col][:, 0])
        return {"tip": tuple(map(int, tip[0])), "axis": "row", "line": int(tip[0, 0]), "cross": [(r, outer_col) for r in rows], "step": (0, 1)}
    raise ValueError(f"unknown side: {side}")


def _line_count(grid: np.ndarray, axis: str, line: int) -> int:
    if axis == "row":
        return int(np.count_nonzero(grid[int(line), :] == COUNT_COLOR))
    if axis == "col":
        return int(np.count_nonzero(grid[:, int(line)] == COUNT_COLOR))
    raise ValueError(f"unknown axis: {axis}")


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    bg = int(get_bg(g))
    if bg != BG:
        raise AssertionError("background must be black")
    colors = sorted(int(color) for color in np.unique(g) if int(color) not in (BG, COUNT_COLOR))
    if len(colors) != 4:
        raise AssertionError("expected four foreground object colors")
    components = find_components(g, bg=bg, connectivity=4, monochromatic=True, colors=colors, return_format="dict")
    if len(components) != 4:
        raise AssertionError("expected four monochromatic 4-connected objects")
    side_map, global_center = _classify_sides(components)

    info: dict[str, dict] = {}
    for side, component in side_map.items():
        profile = _tip_and_growth_profile(side, component["coords"])
        profile["color"] = int(component["color"])
        profile["bbox"] = tuple(map(int, component["bbox"]))
        profile["area"] = int(component["area"])
        profile["center"] = _component_center(component)
        profile["count"] = _line_count(g, profile["axis"], profile["line"])
        if not 1 <= profile["count"] <= 4:
            raise AssertionError("each side count must be between 1 and 4")
        info[side] = profile

    counts = [info[side]["count"] for side in SIDES]
    if sorted(counts) != [1, 2, 3, 4]:
        raise AssertionError("the four count values must be exactly 1,2,3,4")
    winner_side = max(SIDES, key=lambda side: info[side]["count"])
    if list(counts).count(info[winner_side]["count"]) != 1:
        raise AssertionError("winner must be unique")

    H, W = g.shape
    for side, profile in info.items():
        if side == winner_side:
            dr, dc = profile["step"]
            for layer in range(1, profile["count"] + 1):
                for r, c in profile["cross"]:
                    rr = r + dr * layer
                    cc = c + dc * layer
                    if not (0 <= rr < H and 0 <= cc < W):
                        raise AssertionError("winner growth must stay in bounds")
                    if int(g[rr, cc]) != BG:
                        raise AssertionError("winner growth path must be clear background")

    return {"grid": g, "bg": bg, "center": global_center, "sides": info, "winner": winner_side}


def _apply_growth(out: np.ndarray, profile: dict) -> None:
    dr, dc = profile["step"]
    color = int(profile["color"])
    for layer in range(1, profile["count"] + 1):
        for r, c in profile["cross"]:
            out[r + dr * layer, c + dc * layer] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow the uniquely highest-count object outward and retract the other three center-facing tips.

    Input structure:
    - Black is the background and gray pixels are count markers.
    - Four different 4-connected monochromatic foreground objects surround one shared central gap from the top, bottom, left, and right.
    - Each object has a unique one-pixel inward tip pointing toward the gap, and gray markers lie on that tip's row or column depending on the object's orientation.

    Rule:
    - Identify the top, bottom, left, and right objects by their positions around the common center.
    - For each object, count gray pixels on the inward-tip column for vertical objects and on the inward-tip row for horizontal objects.
    - The unique largest count object grows away from the center by that many layers, each layer copying the object's outermost cross-section; the other three objects each lose their single inward tip pixel.
    - Remove every gray count pixel in the output.

    Key insight:
    - The same visible count markers encode both the winner selection and the winner's growth amount, while the non-winners all retract by the same one-pixel inward edit.
    """
    analysis = _analyze_grid(grid)
    out = analysis["grid"].copy()
    out[out == COUNT_COLOR] = BG

    winner = analysis["winner"]
    for side, profile in analysis["sides"].items():
        if side == winner:
            _apply_growth(out, profile)
        else:
            r, c = profile["tip"]
            out[r, c] = BG
    return out


def _nested_spans(tip_pos: int, widths: tuple[int, ...]) -> list[tuple[int, int]]:
    spans = [(int(tip_pos), int(tip_pos))]
    for width in widths[1:]:
        prev_l, prev_r = spans[-1]
        prev_w = prev_r - prev_l + 1
        extra = int(width) - prev_w
        if extra < 0:
            raise AssertionError("width sequence must be nondecreasing")
        if extra == 0:
            spans.append((prev_l, prev_r))
            continue
        left_add = random.randint(0, extra)
        right_add = extra - left_add
        spans.append((prev_l - left_add, prev_r + right_add))
    return spans


def _make_vertical_object(tip_r: int, tip_c: int, outward_up: bool) -> set[tuple[int, int]]:
    widths = random.choice(PATTERNS)
    spans = _nested_spans(tip_c, widths)
    coords: set[tuple[int, int]] = set()
    for depth, (c0, c1) in enumerate(spans):
        row = tip_r - depth if outward_up else tip_r + depth
        for col in range(c0, c1 + 1):
            coords.add((int(row), int(col)))
    return coords


def _make_horizontal_object(tip_r: int, tip_c: int, outward_left: bool) -> set[tuple[int, int]]:
    heights = random.choice(PATTERNS)
    spans = _nested_spans(tip_r, heights)
    coords: set[tuple[int, int]] = set()
    for depth, (r0, r1) in enumerate(spans):
        col = tip_c - depth if outward_left else tip_c + depth
        for row in range(r0, r1 + 1):
            coords.add((int(row), int(col)))
    return coords


def _paint_cells(grid: np.ndarray, cells: set[tuple[int, int]], color: int) -> None:
    for r, c in cells:
        grid[r, c] = int(color)


def _ordered_slots(side: str, values: list[int]) -> list[int]:
    if side in ("top", "right"):
        return sorted(values)
    return sorted(values, reverse=True)


def _place_count_pixels(
    grid: np.ndarray,
    side: str,
    count: int,
    top_tip_c: int,
    bottom_tip_c: int,
    left_tip_r: int,
    right_tip_r: int,
    gap_r0: int,
    gap_r1: int,
    gap_c0: int,
    gap_c1: int,
) -> None:
    if side == "top":
        rows = [r for r in range(gap_r0 + 1, gap_r1) if r not in (left_tip_r, right_tip_r)]
        for r in _ordered_slots(side, rows)[:count]:
            grid[r, top_tip_c] = COUNT_COLOR
        return
    if side == "bottom":
        rows = [r for r in range(gap_r0 + 1, gap_r1) if r not in (left_tip_r, right_tip_r)]
        for r in _ordered_slots(side, rows)[:count]:
            grid[r, bottom_tip_c] = COUNT_COLOR
        return
    if side == "left":
        cols = [c for c in range(gap_c0 + 1, gap_c1) if c not in (top_tip_c, bottom_tip_c)]
        for c in _ordered_slots(side, cols)[:count]:
            grid[left_tip_r, c] = COUNT_COLOR
        return
    if side == "right":
        cols = [c for c in range(gap_c0 + 1, gap_c1) if c not in (top_tip_c, bottom_tip_c)]
        for c in _ordered_slots(side, cols)[:count]:
            grid[right_tip_r, c] = COUNT_COLOR
        return
    raise ValueError(f"unknown side: {side}")


def _count_adjacency_ok(grid: np.ndarray) -> bool:
    H, W = grid.shape
    gray_coords = np.argwhere(grid == COUNT_COLOR)
    for r, c in gray_coords:
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            rr = int(r + dr)
            cc = int(c + dc)
            if 0 <= rr < H and 0 <= cc < W and int(grid[rr, cc]) not in (BG, COUNT_COLOR):
                return False
    return True


def _validate_generated(grid: np.ndarray) -> bool:
    try:
        analysis = _analyze_grid(grid)
    except AssertionError:
        return False
    if not _count_adjacency_ok(grid):
        return False

    out = main(grid)
    if np.any(out == COUNT_COLOR):
        return False

    winner = analysis["winner"]
    winner_profile = analysis["sides"][winner]
    for side, profile in analysis["sides"].items():
        r, c = profile["tip"]
        if side == winner:
            for layer in range(1, profile["count"] + 1):
                for rr, cc in profile["cross"]:
                    if int(out[rr + profile["step"][0] * layer, cc + profile["step"][1] * layer]) != int(profile["color"]):
                        return False
        else:
            if int(out[r, c]) != BG:
                return False

    added = int(np.count_nonzero((out != BG) & (grid == BG)))
    removed = int(np.count_nonzero((grid != BG) & (out == BG) & (grid != COUNT_COLOR)))
    if added != winner_profile["count"] * len(winner_profile["cross"]):
        return False
    if removed != 3:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(24, 29)
    W = random.randint(24, 29)
    max_gap_h = min(10, H - 16)
    max_gap_w = min(10, W - 16)
    if max_gap_h < 8 or max_gap_w < 8:
        return None
    gap_h = random.randint(8, max_gap_h)
    gap_w = random.randint(8, max_gap_w)
    gap_r0 = random.randint(8, H - gap_h - 8)
    gap_c0 = random.randint(8, W - gap_w - 8)
    gap_r1 = gap_r0 + gap_h - 1
    gap_c1 = gap_c0 + gap_w - 1

    top_tip_c, bottom_tip_c = random.sample(list(range(gap_c0 + 1, gap_c1)), 2)
    left_tip_r, right_tip_r = random.sample(list(range(gap_r0 + 1, gap_r1)), 2)

    top_cells = _make_vertical_object(gap_r0 - 1, top_tip_c, outward_up=True)
    bottom_cells = _make_vertical_object(gap_r1 + 1, bottom_tip_c, outward_up=False)
    left_cells = _make_horizontal_object(left_tip_r, gap_c0 - 1, outward_left=True)
    right_cells = _make_horizontal_object(right_tip_r, gap_c1 + 1, outward_left=False)

    objects = {
        "top": top_cells,
        "bottom": bottom_cells,
        "left": left_cells,
        "right": right_cells,
    }
    all_cells = set().union(*objects.values())
    if len(all_cells) != sum(len(cells) for cells in objects.values()):
        return None
    if any(r < 0 or c < 0 or r >= H or c >= W for r, c in all_cells):
        return None

    grid = np.full((H, W), BG, dtype=int)
    colors = list(OBJ_COLORS)
    random.shuffle(colors)
    for side, color in zip(SIDES, colors):
        _paint_cells(grid, objects[side], color)

    counts = dict(zip(random.sample(list(SIDES), len(SIDES)), (1, 2, 3, 4)))
    _place_count_pixels(grid, "top", counts["top"], top_tip_c, bottom_tip_c, left_tip_r, right_tip_r, gap_r0, gap_r1, gap_c0, gap_c1)
    _place_count_pixels(grid, "bottom", counts["bottom"], top_tip_c, bottom_tip_c, left_tip_r, right_tip_r, gap_r0, gap_r1, gap_c0, gap_c1)
    _place_count_pixels(grid, "left", counts["left"], top_tip_c, bottom_tip_c, left_tip_r, right_tip_r, gap_r0, gap_r1, gap_c0, gap_c1)
    _place_count_pixels(grid, "right", counts["right"], top_tip_c, bottom_tip_c, left_tip_r, right_tip_r, gap_r0, gap_r1, gap_c0, gap_c1)

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
