from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox_coords, colors_present, find_components, get_bg, in_bounds


DIRECTIONS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
ARROW_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.PINK, Color.ORANGE]
FRAME_COLOR = Color.GREY
BACKGROUND = Color.BLACK
GRID_MIN = 17
GRID_MAX = 21
FRAME_MIN = 7
FRAME_MAX = 9
MARGIN_MIN = 4
GEN_MAX_ATTEMPTS = 400
PLACE_TRIES = 120


def _frame_bbox(grid: np.ndarray) -> tuple[int, int, int, int]:
    mask = grid == FRAME_COLOR
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected one gray frame"
    r0, c0, r1, c1 = bbox_coords(coords)
    assert r1 - r0 + 1 >= FRAME_MIN and c1 - c0 + 1 >= FRAME_MIN, "frame too small"

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    assert np.array_equal(mask, expected), "gray pixels must form exactly one 1-pixel rectangular frame"
    return r0, c0, r1, c1


def _parse_arrow(coords: np.ndarray) -> tuple[str, int, tuple[int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for rr, cc in coords:
        patch[int(rr) - r0, int(cc) - c0] = True

    h, w = patch.shape
    area = int(patch.sum())

    if w == 3 and h in (3, 4):
        row_sums = patch.sum(axis=1)
        cross_rows = [idx for idx, total in enumerate(row_sums) if int(total) == 3]
        if (
            len(cross_rows) == 1
            and np.all(patch[:, 1])
            and area == h + 2
            and sorted(int(x) for x in row_sums) == [1] * (h - 1) + [3]
        ):
            cross_r = cross_rows[0]
            d_top = cross_r
            d_bottom = h - 1 - cross_r
            expected = [0, 2] if h == 3 else [1, 2]
            if sorted((d_top, d_bottom)) == expected:
                if d_top < d_bottom:
                    return "up", h, (r0, c0 + 1)
                return "down", h, (r1, c0 + 1)

    if h == 3 and w in (3, 4):
        col_sums = patch.sum(axis=0)
        cross_cols = [idx for idx, total in enumerate(col_sums) if int(total) == 3]
        if (
            len(cross_cols) == 1
            and np.all(patch[1, :])
            and area == w + 2
            and sorted(int(x) for x in col_sums) == [1] * (w - 1) + [3]
        ):
            cross_c = cross_cols[0]
            d_left = cross_c
            d_right = w - 1 - cross_c
            expected = [0, 2] if w == 3 else [1, 2]
            if sorted((d_left, d_right)) == expected:
                if d_left < d_right:
                    return "left", w, (r0 + 1, c0)
                return "right", w, (r0 + 1, c1)

    raise AssertionError("expected an asymmetric T arrow with shaft length 3 or 4")


def _cast_to_frame(grid: np.ndarray, tip: tuple[int, int], direction: str) -> tuple[int, int]:
    dr, dc = DIRECTIONS[direction]
    r, c = tip
    while True:
        r += dr
        c += dc
        assert in_bounds(grid.shape, r, c), "ray left the grid before reaching the frame"
        value = int(grid[r, c])
        if value == FRAME_COLOR:
            return r, c
        assert value == BACKGROUND, "arrow path to the frame must be unobstructed"


def _opposite_frame_point(hit: tuple[int, int], frame_bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = frame_bbox
    r, c = hit
    if r == r0:
        return r1, c
    if r == r1:
        return r0, c
    if c == c0:
        return r, c1
    if c == c1:
        return r, c0
    raise AssertionError("hit point must lie on the frame perimeter")


def _periodic_points(
    start: tuple[int, int], direction: str, step: int, shape: tuple[int, int]
) -> list[tuple[int, int]]:
    dr, dc = DIRECTIONS[direction]
    out = []
    r0, c0 = start
    mul = 1
    while True:
        r = r0 + dr * step * mul
        c = c0 + dc * step * mul
        if not in_bounds(shape, r, c):
            break
        out.append((r, c))
        mul += 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Mark each arrow's frame endpoint and extend matching periodic dots outside the frame.

    Input structure:
    - The grid is mostly black, with one one-pixel gray rectangular frame away from the canvas border.
    - Inside the frame are 2 or 3 disconnected monochrome asymmetric T arrows of distinct colors.
    - Each arrow has a shaft of length 3 or 4 and a length-3 crossbar offset toward the pointing end.

    Rule:
    - Determine each arrow's direction from which shaft end is nearer to the crossbar.
    - Follow that direction from the arrow tip to the frame; paint the struck frame pixel with the arrow color.
    - Let k be the shaft length. Starting from that struck frame pixel, paint dots every k cells farther in the same direction outside the frame.
    - Paint the opposite frame pixel on the same row or column black, and leave all other cells unchanged.

    Key insight:
    - Each arrow encodes two independent parameters at once: crossbar offset gives the direction, while shaft length gives the periodic spacing of the outward ray.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(get_bg(g)) == BACKGROUND, "expected black background"

    frame_bbox = _frame_bbox(g)
    r0, c0, r1, c1 = frame_bbox
    assert r0 > 0 and c0 > 0 and r1 < g.shape[0] - 1 and c1 < g.shape[1] - 1, "frame must not touch the border"
    assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] != FRAME_COLOR), "frame interior must not contain gray"

    sprite_colors = colors_present(g, exclude={BACKGROUND, FRAME_COLOR})
    assert 2 <= len(sprite_colors) <= 3, "expected 2 or 3 arrow colors"
    components = find_components(g, bg=BACKGROUND, connectivity=4, monochromatic=True, colors=sprite_colors)
    assert len(components) == len(sprite_colors), "expected one component per arrow color"

    frame_updates: dict[tuple[int, int], int] = {}
    ray_updates: dict[tuple[int, int], int] = {}
    seen_directions = set()

    for component in components:
        color = int(component["color"])
        direction, step, tip = _parse_arrow(component["coords"])
        seen_directions.add(direction)
        hit = _cast_to_frame(g, tip, direction)
        opposite = _opposite_frame_point(hit, frame_bbox)
        assert hit not in frame_updates and opposite not in frame_updates, "frame edits must be unique"
        frame_updates[hit] = color
        frame_updates[opposite] = BACKGROUND

        for coord in _periodic_points(hit, direction, step, g.shape):
            assert int(g[coord]) == BACKGROUND, "periodic ray must land on black cells only"
            assert coord not in ray_updates, "periodic rays must not intersect"
            ray_updates[coord] = color

    assert len(seen_directions) == len(components), "arrows must point to different frame sides"

    out = g.copy()
    for coord, color in frame_updates.items():
        out[coord] = color
    for coord, color in ray_updates.items():
        out[coord] = color
    return out


def _make_sprite(color: int, direction: str, step: int) -> tuple[np.ndarray, tuple[int, int]]:
    if direction in ("up", "down"):
        patch = np.zeros((step, 3), dtype=int)
        patch[:, 1] = color
        cross_r = 0 if (direction == "up" and step == 3) else 1 if direction == "up" else step - 1 if step == 3 else step - 2
        patch[cross_r, :] = color
        tip = (0, 1) if direction == "up" else (step - 1, 1)
        return patch, tip

    patch = np.zeros((3, step), dtype=int)
    patch[1, :] = color
    cross_c = 0 if (direction == "left" and step == 3) else 1 if direction == "left" else step - 1 if step == 3 else step - 2
    patch[:, cross_c] = color
    tip = (1, 0) if direction == "left" else (1, step - 1)
    return patch, tip


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR


def _sample_steps(count: int) -> list[int]:
    if count == 2:
        if random.random() < 0.75:
            steps = [3, 4]
            random.shuffle(steps)
            return steps
        return [random.choice((3, 4)) for _ in range(count)]

    steps = [3, 4, random.choice((3, 4))]
    random.shuffle(steps)
    return steps


def _ray_to_frame_clear(
    grid: np.ndarray, tip: tuple[int, int], direction: str, expected_hit: tuple[int, int]
) -> bool:
    dr, dc = DIRECTIONS[direction]
    r, c = tip
    while True:
        r += dr
        c += dc
        if not in_bounds(grid.shape, r, c):
            return False
        if (r, c) == expected_hit:
            return int(grid[r, c]) == FRAME_COLOR
        if int(grid[r, c]) != BACKGROUND:
            return False


def _placement_ok(
    sprite_mask: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    frame_bbox: tuple[int, int, int, int],
) -> bool:
    h, w = patch.shape
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(sprite_mask.shape[0], top + h + 1)
    c1 = min(sprite_mask.shape[1], left + w + 1)
    if np.any(sprite_mask[r0:r1, c0:c1]):
        return False

    fr0, fc0, fr1, fc1 = frame_bbox
    return top >= fr0 + 1 and left >= fc0 + 1 and top + h - 1 <= fr1 - 1 and left + w - 1 <= fc1 - 1


def _try_place_arrow(
    grid: np.ndarray,
    sprite_mask: np.ndarray,
    frame_bbox: tuple[int, int, int, int],
    plan: dict,
    frame_used: set[tuple[int, int]],
    ray_used: set[tuple[int, int]],
) -> dict | None:
    direction = plan["direction"]
    color = plan["color"]
    step = plan["step"]
    patch, tip_local = _make_sprite(color, direction, step)
    fr0, fc0, fr1, fc1 = frame_bbox

    for _ in range(PLACE_TRIES):
        if direction == "up":
            hit = (fr0, random.randint(fc0 + 1, fc1 - 1))
            tip_r = random.randint(fr0 + 1, fr1 - step)
            top, left = tip_r, hit[1] - 1
        elif direction == "down":
            hit = (fr1, random.randint(fc0 + 1, fc1 - 1))
            tip_r = random.randint(fr0 + step, fr1 - 1)
            top, left = tip_r - step + 1, hit[1] - 1
        elif direction == "left":
            hit = (random.randint(fr0 + 1, fr1 - 1), fc0)
            tip_c = random.randint(fc0 + 1, fc1 - step)
            top, left = hit[0] - 1, tip_c
        else:
            hit = (random.randint(fr0 + 1, fr1 - 1), fc1)
            tip_c = random.randint(fc0 + step, fc1 - 1)
            top, left = hit[0] - 1, tip_c - step + 1

        if not _placement_ok(sprite_mask, patch, top, left, frame_bbox):
            continue

        view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
        if view.shape != patch.shape or np.any(view[patch != 0] != BACKGROUND):
            continue

        tip = (top + tip_local[0], left + tip_local[1])
        if not _ray_to_frame_clear(grid, tip, direction, hit):
            continue

        opposite = _opposite_frame_point(hit, frame_bbox)
        if hit in frame_used or opposite in frame_used:
            continue

        ray_points = _periodic_points(hit, direction, step, grid.shape)
        if not ray_points:
            continue
        if any(point in ray_used for point in ray_points):
            continue

        grid[top : top + patch.shape[0], left : left + patch.shape[1]][patch != 0] = color
        sprite_mask[top : top + patch.shape[0], left : left + patch.shape[1]][patch != 0] = True
        frame_used.add(hit)
        frame_used.add(opposite)
        ray_used.update(ray_points)
        return {
            "color": color,
            "direction": direction,
            "step": step,
            "tip": tip,
            "hit": hit,
            "opposite": opposite,
            "ray_points": ray_points,
        }

    return None


def _oracle_from_plan(grid: np.ndarray, placements: list[dict]) -> np.ndarray:
    out = grid.copy()
    for item in placements:
        out[item["hit"]] = item["color"]
        out[item["opposite"]] = BACKGROUND
        for point in item["ray_points"]:
            out[point] = item["color"]
    return out


def _validate_generated_input(grid: np.ndarray, placements: list[dict], frame_bbox: tuple[int, int, int, int]) -> bool:
    try:
        parsed_frame = _frame_bbox(grid)
        if parsed_frame != frame_bbox:
            return False

        colors = colors_present(grid, exclude={BACKGROUND, FRAME_COLOR})
        if sorted(colors) != sorted(item["color"] for item in placements):
            return False

        frame_cells = set()
        ray_cells = set()
        for item in placements:
            if _cast_to_frame(grid, item["tip"], item["direction"]) != item["hit"]:
                return False
            if _periodic_points(item["hit"], item["direction"], item["step"], grid.shape) != item["ray_points"]:
                return False
            if item["hit"] in frame_cells or item["opposite"] in frame_cells:
                return False
            frame_cells.add(item["hit"])
            frame_cells.add(item["opposite"])
            for point in item["ray_points"]:
                if point in ray_cells:
                    return False
                ray_cells.add(point)

        return np.array_equal(main(grid), _oracle_from_plan(grid, placements))
    except Exception:
        return False


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    frame_h = random.randint(FRAME_MIN, FRAME_MAX)
    frame_w = random.randint(FRAME_MIN, FRAME_MAX)
    if h < frame_h + 2 * MARGIN_MIN or w < frame_w + 2 * MARGIN_MIN:
        return None

    top = random.randint(MARGIN_MIN, h - frame_h - MARGIN_MIN)
    left = random.randint(MARGIN_MIN, w - frame_w - MARGIN_MIN)
    frame_bbox = (top, left, top + frame_h - 1, left + frame_w - 1)

    grid = np.zeros((h, w), dtype=int)
    _draw_frame(grid, frame_bbox)
    sprite_mask = np.zeros_like(grid, dtype=bool)

    count = random.choice((2, 3))
    directions = random.sample(list(DIRECTIONS), count)
    colors = random.sample(ARROW_COLORS, count)
    steps = _sample_steps(count)
    plans = [{"direction": d, "color": c, "step": s} for d, c, s in zip(directions, colors, steps)]
    random.shuffle(plans)

    placements = []
    frame_used: set[tuple[int, int]] = set()
    ray_used: set[tuple[int, int]] = set()
    for plan in plans:
        placed = _try_place_arrow(grid, sprite_mask, frame_bbox, plan, frame_used, ray_used)
        if placed is None:
            return None
        placements.append(placed)

    if not _validate_generated_input(grid, placements, frame_bbox):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
