from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


BLACK = 0
ORANGE = 7
SQUARE_COLORS = (1, 2, 3, 4, 6)
VALID_SIDES = (1, 3, 5, 7, 9)
IN_TO_OUT = {1: 9, 3: 1, 5: 3, 7: 5, 9: 7}
OUT_TO_IN = {v: k for k, v in IN_TO_OUT.items()}

GRID_MIN = 17
GRID_MAX = 22
MAX_GENERATION_ATTEMPTS = 900


def _bbox_from_center(center_r: int, center_c: int, side: int) -> tuple[int, int, int, int]:
    half = side // 2
    return center_r - half, center_c - half, center_r + half, center_c + half


def _boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 0
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _draw_square(
    grid: np.ndarray, color: int, center_r: int, center_c: int, side: int
) -> None:
    r0, c0, r1, c1 = _bbox_from_center(center_r, center_c, side)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _flood_background(square_grid: np.ndarray, seeds: list[tuple[int, int]]) -> np.ndarray:
    h, w = map(int, square_grid.shape)
    reached = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r, c in seeds:
        if 0 <= r < h and 0 <= c < w and square_grid[r, c] == BLACK and not reached[r, c]:
            reached[r, c] = True
            queue.append((int(r), int(c)))
    while queue:
        r, c = queue.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and square_grid[nr, nc] == BLACK and not reached[nr, nc]:
                reached[nr, nc] = True
                queue.append((int(nr), int(nc)))
    return reached


def _extract_square_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(int(c) for c in np.unique(grid) if c not in (BLACK, ORANGE)):
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, "every foreground color must appear at least once"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        side_r = int(r1 - r0 + 1)
        side_c = int(c1 - c0 + 1)
        assert side_r == side_c, "each object must have a square bbox"
        assert side_r in VALID_SIDES and side_r % 2 == 1, "square side must be odd and in the cycle"
        patch = grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
        assert np.all(patch == int(color)), "each object must be a solid monochrome square"
        assert int(coords.shape[0]) == side_r * side_r, "each square must fill its bbox"
        objects.append(
            {
                "color": int(color),
                "center_r": int((r0 + r1) // 2),
                "center_c": int((c0 + c1) // 2),
                "in_side": int(side_r),
                "out_side": int(IN_TO_OUT[int(side_r)]),
            }
        )
    assert 3 <= len(objects) <= 5, "expected 3 to 5 colored squares"
    return objects


def _render_transformed(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    h, w = map(int, shape)
    out = np.zeros((h, w), dtype=int)
    painted = np.zeros((h, w), dtype=bool)
    order = sorted(
        objects,
        key=lambda obj: (
            -int(obj["out_side"]) * int(obj["out_side"]),
            int(obj["center_r"]),
            int(obj["center_c"]),
        ),
    )
    for obj in order:
        r0, c0, r1, c1 = _bbox_from_center(
            int(obj["center_r"]), int(obj["center_c"]), int(obj["out_side"])
        )
        assert 0 <= r0 <= r1 < h and 0 <= c0 <= c1 < w, "transformed square must stay in bounds"
        for r in range(int(r0), int(r1) + 1):
            for c in range(int(c0), int(c1) + 1):
                if not painted[r, c]:
                    out[r, c] = int(obj["color"])
                    painted[r, c] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Cycle odd solid squares to new centered sizes, then flood-fill the transformed background corridors from the orange seed pixels.

    Input structure:
    - The grid is black background with 3 to 5 separated solid monochrome odd-sided squares.
    - Square colors are non-black and non-orange, each square keeps a fixed center and has side length 1, 3, 5, 7, or 9.
    - One or two orange seed pixels lie on background cells outside every square.

    Rule:
    - Clear the grid to black.
    - For each square, keep its center and color, but cycle its side length by 1->9->7->5->3->1.
    - Redraw the transformed squares in descending transformed area order, breaking ties by center row then center column, and do not let later squares overwrite earlier painted cells.
    - Starting from every orange seed position, flood-fill all 4-connected black cells in the transformed grid with orange; square pixels are never entered.

    Key insight:
    - The orange region cannot be determined from the input corridors directly: shrinking squares opens new black channels while 1x1 squares explode to 9x9 blockers, so the fill must happen after the size cycle.
    """
    g = np.asarray(grid, dtype=int)
    seeds = [tuple(map(int, rc)) for rc in np.argwhere(g == ORANGE)]
    assert 1 <= len(seeds) <= 2, "expected 1 or 2 orange seed pixels"

    objects = _extract_square_objects(g)
    transformed = _render_transformed(g.shape, objects)

    for r, c in seeds:
        assert transformed[r, c] == BLACK, "seed must remain on transformed background"

    reached = _flood_background(transformed, seeds)
    transformed[reached] = ORANGE
    return transformed


def _sample_output_sides(count: int, include_expander: bool) -> list[int]:
    sides: list[int] = []
    if include_expander:
        sides.append(9)
    if len(sides) < count and random.random() < 0.45:
        sides.append(7)
    pool = [1, 1, 3, 3, 3, 5, 5, 7]
    while len(sides) < count:
        sides.append(int(random.choice(pool)))
    random.shuffle(sides)
    return sides


def _place_objects(
    h: int, w: int, count: int, include_expander: bool
) -> list[dict] | None:
    colors = random.sample(list(SQUARE_COLORS), count)
    out_sides = _sample_output_sides(count, include_expander)
    objects = [
        {
            "color": int(color),
            "out_side": int(out_side),
            "in_side": int(OUT_TO_IN[int(out_side)]),
        }
        for color, out_side in zip(colors, out_sides)
    ]
    total_input_area = sum(int(obj["in_side"]) ** 2 for obj in objects)
    if total_input_area >= int(0.55 * h * w):
        return None

    objects.sort(
        key=lambda obj: (max(int(obj["in_side"]), int(obj["out_side"])), int(obj["out_side"])),
        reverse=True,
    )

    placed: list[dict] = []
    input_gap = 1
    output_gap = random.choice((0, 1))
    for obj in objects:
        side_limit = max(int(obj["in_side"]), int(obj["out_side"]))
        margin = side_limit // 2
        centers = [(r, c) for r in range(margin, h - margin) for c in range(margin, w - margin)]
        random.shuffle(centers)
        for center_r, center_c in centers:
            in_box = _bbox_from_center(int(center_r), int(center_c), int(obj["in_side"]))
            out_box = _bbox_from_center(int(center_r), int(center_c), int(obj["out_side"]))
            if all(
                _boxes_separated(in_box, prev["in_box"], input_gap)
                and _boxes_separated(out_box, prev["out_box"], output_gap)
                for prev in placed
            ):
                placed.append(
                    {
                        **obj,
                        "center_r": int(center_r),
                        "center_c": int(center_c),
                        "in_box": in_box,
                        "out_box": out_box,
                    }
                )
                break
        else:
            return None
    random.shuffle(placed)
    return placed


def _render_input_squares(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for obj in objects:
        _draw_square(
            grid,
            int(obj["color"]),
            int(obj["center_r"]),
            int(obj["center_c"]),
            int(obj["in_side"]),
        )
    return grid


def _build_expected_output(
    transformed_squares: np.ndarray, seeds: list[tuple[int, int]]
) -> np.ndarray:
    out = transformed_squares.copy()
    out[_flood_background(transformed_squares, seeds)] = ORANGE
    return out


def _pick_seeds(
    input_squares: np.ndarray, transformed_squares: np.ndarray, include_expander: bool
) -> list[tuple[int, int]] | None:
    h, w = map(int, input_squares.shape)
    common_bg = (input_squares == BLACK) & (transformed_squares == BLACK)
    candidates = [tuple(map(int, rc)) for rc in np.argwhere(common_bg)]
    if len(candidates) < 1:
        return None
    random.shuffle(candidates)

    newly_exposed = (input_squares != BLACK) & (transformed_squares == BLACK)
    sampled = candidates[: min(len(candidates), 80)]
    infos: list[dict] = []
    for seed in sampled:
        reach_out = _flood_background(transformed_squares, [seed])
        reach_in = _flood_background(input_squares, [seed])
        exposed_count = int(np.count_nonzero(reach_out & newly_exposed))
        delta = int(np.count_nonzero(reach_out ^ reach_in))
        filled = int(np.count_nonzero(reach_out))
        lost = int(np.count_nonzero(reach_in & ~reach_out))
        if exposed_count < 4:
            continue
        if delta < 6:
            continue
        if filled < 12 or filled > int(0.78 * h * w):
            continue
        if include_expander and lost < 10:
            continue
        infos.append(
            {
                "seed": seed,
                "reach_out": reach_out,
                "reach_in": reach_in,
                "score": (exposed_count, delta, filled),
            }
        )

    if not infos:
        return None

    want_two = random.random() < 0.38
    if want_two and len(infos) >= 2:
        ordered = sorted(infos, key=lambda item: item["score"], reverse=True)[: min(20, len(infos))]
        random.shuffle(ordered)
        for a, b in combinations(ordered, 2):
            if abs(a["seed"][0] - b["seed"][0]) + abs(a["seed"][1] - b["seed"][1]) < 3:
                continue
            union_out = a["reach_out"] | b["reach_out"]
            union_in = a["reach_in"] | b["reach_in"]
            if int(np.count_nonzero(a["reach_out"] & ~b["reach_out"])) < 3:
                continue
            if int(np.count_nonzero(b["reach_out"] & ~a["reach_out"])) < 3:
                continue
            if int(np.count_nonzero(union_out & newly_exposed)) < 6:
                continue
            if int(np.count_nonzero(union_out ^ union_in)) < 8:
                continue
            if include_expander and int(np.count_nonzero(union_in & ~union_out)) < 12:
                continue
            return [a["seed"], b["seed"]]
        for a, b in combinations(ordered, 2):
            if abs(a["seed"][0] - b["seed"][0]) + abs(a["seed"][1] - b["seed"][1]) < 5:
                continue
            union_out = a["reach_out"] | b["reach_out"]
            union_in = a["reach_in"] | b["reach_in"]
            if int(np.count_nonzero(union_out & newly_exposed)) < 6:
                continue
            if int(np.count_nonzero(union_out ^ union_in)) < 8:
                continue
            if include_expander and int(np.count_nonzero(union_in & ~union_out)) < 12:
                continue
            return [a["seed"], b["seed"]]

    infos.sort(key=lambda item: item["score"], reverse=True)
    return [infos[0]["seed"]]


def _semantic_checks(
    input_grid: np.ndarray, transformed_squares: np.ndarray, seeds: list[tuple[int, int]], include_expander: bool
) -> bool:
    input_squares = input_grid.copy()
    input_squares[input_squares == ORANGE] = BLACK
    reach_out = _flood_background(transformed_squares, seeds)
    reach_in = _flood_background(input_squares, seeds)
    newly_exposed = (input_squares != BLACK) & (transformed_squares == BLACK)
    if int(np.count_nonzero(reach_out & newly_exposed)) < 4:
        return False
    if int(np.count_nonzero(reach_out ^ reach_in)) < 6:
        return False
    if include_expander and int(np.count_nonzero(reach_in & ~reach_out)) < 10:
        return False
    if not np.all(transformed_squares[np.array([r for r, _ in seeds]), np.array([c for _, c in seeds])] == BLACK):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_MIN, GRID_MAX)
    w = random.randint(GRID_MIN, GRID_MAX)
    count = random.randint(3, 5)
    include_expander = random.random() < 0.58

    objects = _place_objects(h, w, count, include_expander)
    if objects is None:
        return None

    input_squares = _render_input_squares((h, w), objects)
    transformed_squares = _render_transformed((h, w), objects)

    if int(np.count_nonzero(transformed_squares == BLACK)) < 20:
        return None

    seeds = _pick_seeds(input_squares, transformed_squares, include_expander)
    if seeds is None:
        return None

    grid = input_squares.copy()
    for r, c in seeds:
        grid[r, c] = ORANGE

    expected = _build_expected_output(transformed_squares, seeds)
    if not _semantic_checks(grid, transformed_squares, seeds, include_expander):
        return None
    try:
        predicted = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(predicted, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid odd-square flood-fill example")
