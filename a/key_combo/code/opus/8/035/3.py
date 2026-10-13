from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

_TOOLBOX_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_TOOLBOX_DIR) not in sys.path:
    sys.path.append(str(_TOOLBOX_DIR))

from arc_utils import Color, find_components, get_bg


FRAME_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
FRAME_SIDE_OPTIONS = (5, 7, 9, 11, 13, 15)
INPUT_GRID_MIN = 18
INPUT_GRID_MAX = 30
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 500
RNG = np.random.default_rng()


def _box_for(side: int, r: int, c: int) -> tuple[int, int, int, int]:
    return int(r), int(c), int(r + side - 1), int(c + side - 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int, color: int) -> None:
    grid[r0, c0 : c0 + side] = int(color)
    grid[r0 + side - 1, c0 : c0 + side] = int(color)
    grid[r0 : r0 + side, c0] = int(color)
    grid[r0 : r0 + side, c0 + side - 1] = int(color)


def _is_hollow_square_frame(component: dict, grid: np.ndarray, bg: int) -> bool:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    side_h = int(r1 - r0 + 1)
    side_w = int(c1 - c0 + 1)
    if side_h != side_w or side_h < 3 or side_h % 2 == 0:
        return False
    if int(component["area"]) != 4 * side_h - 4:
        return False

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    if not np.isin(patch, (int(bg), color)).all():
        return False

    mask = patch == color
    border = np.zeros((side_h, side_h), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    frames: list[dict[str, int]] = []
    for component in find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict"):
        if not _is_hollow_square_frame(component, g, bg):
            raise AssertionError("every foreground component must be a hollow odd square frame")
        r0, c0, r1, c1 = map(int, component["bbox"])
        side = int(r1 - r0 + 1)
        frames.append(
            {
                "color": int(component["color"]),
                "side": side,
                "r0": r0,
                "c0": c0,
                "center_r": int(r0 + side // 2),
                "center_c": int(c0 + side // 2),
            }
        )

    if not frames:
        raise AssertionError("expected at least one frame")

    sides = [frame["side"] for frame in frames]
    if len(set(sides)) != len(sides):
        raise AssertionError("frame side lengths must be distinct")
    if any(side % 2 == 0 for side in sides):
        raise AssertionError("frame side lengths must be odd")
    return bg, frames


def _render_nested(frame_order: list[dict[str, int]], color_order: list[int], bg: int) -> np.ndarray:
    max_side = max(int(frame["side"]) for frame in frame_order)
    out = np.full((max_side, max_side), int(bg), dtype=int)
    for frame, color in zip(frame_order, color_order):
        side = int(frame["side"])
        offset = (max_side - side) // 2
        _draw_frame(out, offset, offset, side, int(color))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Nest scattered hollow square frames by size and reverse their layer colors.

    Input structure:
    - The background is black.
    - The foreground contains 3-4 disjoint one-cell-thick hollow square frames.
    - Every frame has a distinct odd side length, a single monochrome border color, and a black interior.

    Rule:
    - Identify every hollow square frame and measure its outer side length.
    - Sort the frames from largest to smallest, then center-align them into one concentric stack whose canvas size equals the largest frame.
    - Read the resulting outer-to-inner color sequence and reverse it, so the outermost slot receives the original innermost frame color, the innermost slot receives the original outermost color, and middle layers swap accordingly.
    - Keep all non-frame cells in the output black.

    Key insight:
    - The input positions do not define depth; depth appears only after size-sorting and concentric reassembly, so the color reversal must be applied to that newly created nesting order.
    """
    bg, frames = _extract_frames(grid)
    ordered = sorted(frames, key=lambda frame: int(frame["side"]), reverse=True)
    reversed_colors = [int(frame["color"]) for frame in reversed(ordered)]
    return _render_nested(ordered, reversed_colors, bg)


def _place_sides(sides: tuple[int, ...], grid_side: int, *, randomize: bool) -> list[tuple[int, int, int]] | None:
    ordered_sides = tuple(sorted(map(int, sides), reverse=True))
    candidates_by_side = [
        [(r, c) for r in range(grid_side - side + 1) for c in range(grid_side - side + 1)]
        for side in ordered_sides
    ]
    restarts = 120 if randomize else 1
    for _ in range(restarts):
        placements: list[tuple[int, int, int]] = []
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for side, candidates in zip(ordered_sides, candidates_by_side):
            pool = list(candidates)
            if randomize:
                RNG.shuffle(pool)

            placed = False
            for r, c in pool:
                box = _box_for(side, r, c)
                if all(_boxes_separated(box, prev) for prev in boxes):
                    placements.append((int(r), int(c), int(side)))
                    boxes.append(box)
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if ok:
            return placements
    return None


def _lower_grid_side(sides: tuple[int, ...]) -> int:
    total_area = sum(side * side for side in sides)
    return max(INPUT_GRID_MIN, max(sides) + 2, int(math.ceil(math.sqrt(total_area))) + 2)


def _sample_sides(count: int) -> tuple[int, ...]:
    for _ in range(200):
        sides = tuple(sorted(int(side) for side in RNG.choice(FRAME_SIDE_OPTIONS, size=count, replace=False)))
        gaps = [b - a for a, b in zip(sides, sides[1:])]
        if RNG.random() < 0.7 and all(gap == 2 for gap in gaps):
            continue
        return sides
    raise AssertionError("failed to sample non-degenerate side lengths")


def _position_orders(frames: list[dict[str, int]]) -> list[tuple[int, ...]]:
    return [
        tuple(frame["side"] for frame in sorted(frames, key=lambda frame: (frame["center_c"], frame["center_r"]))),
        tuple(frame["side"] for frame in sorted(frames, key=lambda frame: (frame["center_r"], frame["center_c"]))),
        tuple(frame["side"] for frame in sorted(frames, key=lambda frame: (frame["r0"], frame["c0"]))),
    ]


def _anti_shortcut_checks(frames: list[dict[str, int]], expected: np.ndarray, bg: int) -> bool:
    size_desc = tuple(frame["side"] for frame in sorted(frames, key=lambda frame: frame["side"], reverse=True))
    size_asc = tuple(reversed(size_desc))
    for order in _position_orders(frames):
        if order == size_desc or order == size_asc:
            return False

    by_size = sorted(frames, key=lambda frame: frame["side"], reverse=True)
    same_colors = [frame["color"] for frame in by_size]
    if np.array_equal(expected, _render_nested(by_size, same_colors, bg)):
        return False

    for key in (
        lambda frame: (frame["center_c"], frame["center_r"]),
        lambda frame: (frame["center_r"], frame["center_c"]),
    ):
        positional = sorted(frames, key=key)
        wrong_geom = _render_nested(positional, [frame["color"] for frame in reversed(positional)], bg)
        wrong_colors = _render_nested(by_size, [frame["color"] for frame in positional], bg)
        if np.array_equal(expected, wrong_geom) or np.array_equal(expected, wrong_colors):
            return False
    return True


def _build_input_grid(frames: list[dict[str, int]], grid_side: int) -> np.ndarray:
    grid = np.full((grid_side, grid_side), Color.BLACK, dtype=int)
    for frame in frames:
        _draw_frame(grid, int(frame["r0"]), int(frame["c0"]), int(frame["side"]), int(frame["color"]))
    return grid


def _gen_attempt() -> np.ndarray | None:
    frame_count = int(RNG.integers(3, 5))
    sides = _sample_sides(frame_count)
    colors = [int(color) for color in RNG.choice(FRAME_COLORS, size=frame_count, replace=False)]
    RNG.shuffle(colors)

    lower = _lower_grid_side(sides)
    if lower > INPUT_GRID_MAX:
        return None
    side_choices = list(range(lower, min(INPUT_GRID_MAX, lower + 8) + 1))
    RNG.shuffle(side_choices)
    placements = None
    grid_side = None
    for candidate_side in side_choices:
        placements = _place_sides(sides, int(candidate_side), randomize=True)
        if placements is not None:
            grid_side = int(candidate_side)
            break
    if placements is None or grid_side is None:
        return None

    frames = []
    for color, (r0, c0, side) in zip(colors, placements):
        frames.append(
            {
                "color": int(color),
                "side": int(side),
                "r0": int(r0),
                "c0": int(c0),
                "center_r": int(r0 + side // 2),
                "center_c": int(c0 + side // 2),
            }
        )

    expected = _render_nested(
        sorted(frames, key=lambda frame: frame["side"], reverse=True),
        [int(frame["color"]) for frame in sorted(frames, key=lambda frame: frame["side"])],
        Color.BLACK,
    )
    if not _anti_shortcut_checks(frames, expected, Color.BLACK):
        return None

    inp = _build_input_grid(frames, grid_side)
    if np.array_equal(inp, expected):
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame reversal input")
