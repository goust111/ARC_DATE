from __future__ import annotations

import random
import sys

import numpy as np

try:
    from arc_utils import Color, bbox, get_bg
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import Color, bbox, get_bg


GREY = Color.GREY
RED = Color.RED
ORANGE = Color.ORANGE
MAROON = Color.MAROON

MAX_GENERATION_ATTEMPTS = 500


def _solid_rect_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int]:
    mask = grid == color
    box = bbox(mask)
    assert box is not None, f"missing color {color}"
    r0, c0, r1, c1 = box
    area = (r1 - r0 + 1) * (c1 - c0 + 1)
    assert int(mask.sum()) == area, f"color {color} must form one solid rectangle"
    assert np.all(mask[r0 : r1 + 1, c0 : c1 + 1]), f"color {color} must fill its bbox"
    return int(r0), int(c0), int(r1), int(c1)


def _paint_rect(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = box
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _wall_positions_horizontal(grid: np.ndarray, row_box: tuple[int, int], c_start: int, c_stop: int) -> list[tuple[int, int]]:
    r0, r1 = row_box
    walls: list[tuple[int, int]] = []
    step = 1 if c_start <= c_stop else -1
    for c in range(c_start, c_stop + step, step):
        if np.all(grid[r0 : r1 + 1, c] == GREY):
            parity = int(np.count_nonzero(grid[:, c] == GREY) % 2)
            walls.append((int(c), parity))
    return walls


def _wall_positions_vertical(grid: np.ndarray, col_box: tuple[int, int], r_start: int, r_stop: int) -> list[tuple[int, int]]:
    c0, c1 = col_box
    walls: list[tuple[int, int]] = []
    step = 1 if r_start <= r_stop else -1
    for r in range(r_start, r_stop + step, step):
        if np.all(grid[r, c0 : c1 + 1] == GREY):
            parity = int(np.count_nonzero(grid[r, :] == GREY) % 2)
            walls.append((int(r), parity))
    return walls


def _extract_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    src = _solid_rect_bbox(g, ORANGE)
    dst = _solid_rect_bbox(g, MAROON)
    bg = get_bg(g)

    src_h = src[2] - src[0] + 1
    src_w = src[3] - src[1] + 1
    dst_h = dst[2] - dst[0] + 1
    dst_w = dst[3] - dst[1] + 1
    assert (src_h, src_w) == (dst_h, dst_w), "source and target rectangles must match"

    if src[0] == dst[0] and src[2] == dst[2]:
        direction = 1 if src[1] < dst[1] else -1
        if direction == 1:
            walls = _wall_positions_horizontal(g, (src[0], src[2]), src[3] + 1, dst[1] - 1)
        else:
            walls = _wall_positions_horizontal(g, (src[0], src[2]), src[1] - 1, dst[3] + 1)
        return {
            "grid": g,
            "bg": int(bg),
            "orientation": "horizontal",
            "direction": direction,
            "source": src,
            "target": dst,
            "height": src_h,
            "width": src_w,
            "walls": walls,
        }

    if src[1] == dst[1] and src[3] == dst[3]:
        direction = 1 if src[0] < dst[0] else -1
        if direction == 1:
            walls = _wall_positions_vertical(g, (src[1], src[3]), src[2] + 1, dst[0] - 1)
        else:
            walls = _wall_positions_vertical(g, (src[1], src[3]), src[0] - 1, dst[2] + 1)
        return {
            "grid": g,
            "bg": int(bg),
            "orientation": "vertical",
            "direction": direction,
            "source": src,
            "target": dst,
            "height": src_h,
            "width": src_w,
            "walls": walls,
        }

    raise AssertionError("source and target must share either rows or columns")


def _final_box(scene: dict[str, object]) -> tuple[tuple[int, int, int, int], list[int]]:
    source = scene["source"]
    target = scene["target"]
    orientation = scene["orientation"]
    direction = int(scene["direction"])
    width = int(scene["width"])
    height = int(scene["height"])

    crossed: list[int] = []
    stop_wall: int | None = None
    for axis_pos, parity in scene["walls"]:
        if parity % 2 == 1:
            crossed.append(int(axis_pos))
            continue
        stop_wall = int(axis_pos)
        break

    if stop_wall is None:
        return target, crossed

    if orientation == "horizontal":
        r0, _c0, r1, _c1 = source
        if direction == 1:
            return (r0, stop_wall - width, r1, stop_wall - 1), crossed
        return (r0, stop_wall + 1, r1, stop_wall + width), crossed

    _r0, c0, _r1, c1 = source
    if direction == 1:
        return (stop_wall - height, c0, stop_wall - 1, c1), crossed
    return (stop_wall + 1, c0, stop_wall + height, c1), crossed


def _render(scene: dict[str, object]) -> np.ndarray:
    grid = np.array(scene["grid"], copy=True)
    bg = int(scene["bg"])
    source = scene["source"]
    orientation = scene["orientation"]

    _paint_rect(grid, source, bg)
    final_box, crossed = _final_box(scene)

    if orientation == "horizontal":
        r0, _c0, r1, _c1 = source
        for c in crossed:
            grid[r0 : r1 + 1, c] = RED
    else:
        _r0, c0, _r1, c1 = source
        for r in crossed:
            grid[r, c0 : c1 + 1] = RED

    _paint_rect(grid, final_box, ORANGE)
    return grid


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Slide the orange rectangle toward the maroon rectangle through parity-gated grey walls.

    Odd-size walls are passable and leave a red trace only on the corridor-crossing
    segment. The first even-size wall blocks the motion, so the orange rectangle
    stops flush against it. If every encountered wall is odd, the orange rectangle
    reaches the maroon target and replaces it.
    """
    scene = _extract_scene(np.asarray(input_grid, dtype=int))
    return _render(scene)


def _sample_extensions(rect_len: int, parity: int, before_cap: int, after_cap: int) -> tuple[int, int]:
    candidates = [
        (before, after)
        for before in range(before_cap + 1)
        for after in range(after_cap + 1)
        if (rect_len + before + after) % 2 == parity
    ]
    extended = [(before, after) for before, after in candidates if before + after > 0]
    return random.choice(extended or candidates)


def _build_horizontal_case() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rect_h = random.randint(2, 4)
        rect_w = random.randint(2, 4)
        n_walls = random.randint(1, 3)

        if random.random() < 0.35:
            first_even = None
            parities = [1] * n_walls
        else:
            first_even = random.randrange(n_walls)
            parities = [1 if i < first_even else 0 if i == first_even else random.randint(0, 1) for i in range(n_walls)]

        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        top_margin = random.randint(1, 4)
        bottom_margin = random.randint(1, 4)

        source_left = left_margin
        source_right = source_left + rect_w - 1
        wall_cols: list[int] = []
        cursor = source_right
        for idx in range(n_walls):
            gap_low = 1
            if first_even is not None and idx == first_even and idx > 0:
                gap_low = rect_w
            gap = random.randint(gap_low, gap_low + 3)
            wall_col = cursor + gap + 1
            wall_cols.append(wall_col)
            cursor = wall_col

        end_gap = random.randint(2, 5)
        target_left = cursor + end_gap + 1
        width = target_left + rect_w + right_margin
        height = top_margin + rect_h + bottom_margin
        if width > 30 or height > 30:
            continue

        bg_choices = [color for color in Color.ALL_COLORS if color not in {GREY, RED, ORANGE, MAROON}]
        bg = random.choice(bg_choices)
        grid = np.full((height, width), bg, dtype=int)

        r0 = top_margin
        r1 = r0 + rect_h - 1
        source = (r0, source_left, r1, source_right)
        target = (r0, target_left, r1, target_left + rect_w - 1)
        _paint_rect(grid, source, ORANGE)
        _paint_rect(grid, target, MAROON)

        for wall_col, parity in zip(wall_cols, parities):
            ext_top, ext_bottom = _sample_extensions(rect_h, parity, top_margin, bottom_margin)
            grid[r0 - ext_top : r1 + ext_bottom + 1, wall_col] = GREY

        if np.count_nonzero(grid == bg) <= max(grid.size // 2, 20):
            continue

        if random.random() < 0.5:
            grid = np.fliplr(grid).copy()
        if random.random() < 0.5:
            grid = grid.T.copy()
        return grid

    raise RuntimeError("failed to generate a valid parity-gate instance")


def generate_input() -> np.ndarray:
    return _build_horizontal_case()


def _validate_solution(input_grid: np.ndarray, output_grid: np.ndarray) -> None:
    scene = _extract_scene(input_grid)
    source = scene["source"]
    target = scene["target"]
    final_box, crossed = _final_box(scene)

    assert isinstance(output_grid, np.ndarray)
    assert output_grid.ndim == 2
    assert output_grid.shape == input_grid.shape
    assert output_grid.shape[0] <= 30 and output_grid.shape[1] <= 30

    orange_box = _solid_rect_bbox(output_grid, ORANGE)
    assert orange_box == final_box

    expected_area = int(scene["height"]) * int(scene["width"])
    assert int(np.count_nonzero(output_grid == ORANGE)) == expected_area

    orientation = scene["orientation"]
    if orientation == "horizontal":
        r0, _c0, r1, _c1 = source
        encountered = {axis for axis, _parity in scene["walls"]}
        crossed_set = set(crossed)
        for c in encountered:
            segment = output_grid[r0 : r1 + 1, c]
            if c in crossed_set:
                assert np.all(segment == RED)
            else:
                assert np.all(segment == GREY)
    else:
        _r0, c0, _r1, c1 = source
        encountered = {axis for axis, _parity in scene["walls"]}
        crossed_set = set(crossed)
        for r in encountered:
            segment = output_grid[r, c0 : c1 + 1]
            if r in crossed_set:
                assert np.all(segment == RED)
            else:
                assert np.all(segment == GREY)

    if final_box == target:
        assert not np.any(output_grid[np.s_[target[0] : target[2] + 1, target[1] : target[3] + 1]] == MAROON)
    else:
        assert np.all(output_grid[np.s_[target[0] : target[2] + 1, target[1] : target[3] + 1]] == MAROON)


def _self_check() -> None:
    for _ in range(40):
        sample = generate_input()
        solved = main(sample)
        _validate_solution(sample, solved)


if __name__ == "__main__":
    _self_check()
