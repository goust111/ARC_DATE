from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PIXEL_COLORS = [3, 4, 5, 6, 7, 8, 9]


def _contiguous_segment(mask: np.ndarray, name: str) -> tuple[int, int]:
    idx = np.flatnonzero(mask)
    assert idx.size > 0, f"missing {name} marker segment"
    assert idx.size < mask.size, f"{name} marker must not fill the whole edge"
    assert np.all(np.diff(idx) == 1), f"{name} marker must be one contiguous segment"
    return int(idx[0]), int(idx[-1])


def _segments(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int]]:
    red_cols = _contiguous_segment(g[0, :] == RED, "red top")
    blue_rows = _contiguous_segment(g[:, 0] == BLUE, "blue left")

    red_coords = np.argwhere(g == RED)
    blue_coords = np.argwhere(g == BLUE)
    assert red_coords.size > 0 and blue_coords.size > 0
    assert all(int(r) == 0 and red_cols[0] <= int(c) <= red_cols[1] for r, c in red_coords)
    assert all(int(c) == 0 and blue_rows[0] <= int(r) <= blue_rows[1] for r, c in blue_coords)
    return red_cols, blue_rows


def _pixel_route(
    r: int,
    c: int,
    red_cols: tuple[int, int],
    blue_rows: tuple[int, int],
) -> str:
    in_red = red_cols[0] <= c <= red_cols[1]
    in_blue = blue_rows[0] <= r <= blue_rows[1]
    if in_red and in_blue:
        assert r != c, "double-captured pixels must have unequal edge distances"
        return "red" if r < c else "blue"
    if in_red:
        return "red"
    if in_blue:
        return "blue"
    return "stay"


def _classified_pixels(
    g: np.ndarray,
) -> tuple[
    tuple[int, int],
    tuple[int, int],
    list[tuple[int, int, int, str]],
]:
    red_cols, blue_rows = _segments(g)
    pixels: list[tuple[int, int, int, str]] = []

    H, W = g.shape
    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == BLACK or color in (RED, BLUE):
                continue
            assert r > 0 and c > 0, "scattered pixels must be interior"
            route = _pixel_route(r, c, red_cols, blue_rows)
            pixels.append((r, c, color, route))

    assert pixels, "expected at least one scattered pixel"
    return red_cols, blue_rows, pixels


def _solve(g: np.ndarray) -> np.ndarray:
    red_cols, blue_rows, pixels = _classified_pixels(g)
    out = np.full_like(g, BLACK)
    out[g == RED] = RED
    out[g == BLUE] = BLUE

    red_groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    blue_groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for r, c, color, route in pixels:
        if route == "red":
            red_groups[c].append((r, c, color))
        elif route == "blue":
            blue_groups[r].append((r, c, color))
        else:
            assert int(out[r, c]) == BLACK, "stationary pixels must not collide"
            out[r, c] = color

    for c, group in red_groups.items():
        assert red_cols[0] <= c <= red_cols[1]
        for k, (r, _c, color) in enumerate(sorted(group, key=lambda item: item[0]), start=1):
            assert k < g.shape[0]
            assert int(out[k, c]) == BLACK, "red and blue stacks must not collide"
            out[k, c] = color

    for r, group in blue_groups.items():
        assert blue_rows[0] <= r <= blue_rows[1]
        for k, (_r, c, color) in enumerate(sorted(group, key=lambda item: item[1]), start=1):
            assert k < g.shape[1]
            assert int(out[r, k]) == BLACK, "red and blue stacks must not collide"
            out[r, k] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide pixels captured by top and left edge marker intervals toward the owning edge.

    Input structure:
    - The background is black.
    - A single red segment lies on the top edge and a single blue segment lies on the left edge; neither fills its whole edge.
    - Interior non-red, non-blue pixels are isolated scattered pixels.

    Rule:
    - A pixel whose column is inside the red top segment is captured by red; a pixel whose row is inside the blue left segment is captured by blue.
    - Pixels captured by only one segment slide straight to that segment's edge. Pixels captured by both compare their row distance to the top edge with their column distance to the left edge and slide to the nearer edge.
    - In each captured row or column, nearer pixels arrive first and the rest stack behind them. Pixels captured by neither segment remain fixed.

    Key insight:
    - The edge markers define two capture corridors, and the only ambiguous-looking pixels are resolved by their own distances to the two marked edges.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    return _solve(g)


# ----------------------------- generator -----------------------------


GEN_RETRIES = 800
MIN_SIZE = 14
MAX_SIZE = 24
MIN_SEG_LEN = 4
MAX_SEG_LEN = 7
EDGE_GAP = 5
MAX_STACK_PER_LINE = 4


def _interior_range(limit: int) -> range:
    return range(2, limit - 1)


def _far_enough(cell: tuple[int, int], occupied: set[tuple[int, int]]) -> bool:
    r, c = cell
    return all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in occupied)


def _choose_separated(values: list[int], count: int) -> list[int] | None:
    values = values[:]
    random.shuffle(values)
    chosen: list[int] = []
    for v in values:
        if all(abs(v - u) > 1 for u in chosen):
            chosen.append(int(v))
            if len(chosen) == count:
                return chosen
    return None


def _route_name(
    r: int,
    c: int,
    red_cols: tuple[int, int],
    blue_rows: tuple[int, int],
) -> str | None:
    in_red = red_cols[0] <= c <= red_cols[1]
    in_blue = blue_rows[0] <= r <= blue_rows[1]
    if in_red and in_blue:
        if r == c:
            return None
        return "both_red" if r < c else "both_blue"
    if in_red:
        return "red_only"
    if in_blue:
        return "blue_only"
    return "stay"


def _stack_counts(
    pixels: list[tuple[int, int, int]],
    red_cols: tuple[int, int],
    blue_rows: tuple[int, int],
) -> tuple[dict[int, int], dict[int, int]]:
    red_counts: dict[int, int] = defaultdict(int)
    blue_counts: dict[int, int] = defaultdict(int)
    for r, c, _color in pixels:
        route = _route_name(r, c, red_cols, blue_rows)
        if route in ("red_only", "both_red"):
            red_counts[c] += 1
        elif route in ("blue_only", "both_blue"):
            blue_counts[r] += 1
    return red_counts, blue_counts


def _can_add(
    cell: tuple[int, int],
    pixels: list[tuple[int, int, int]],
    occupied: set[tuple[int, int]],
    red_cols: tuple[int, int],
    blue_rows: tuple[int, int],
) -> bool:
    r, c = cell
    if not _far_enough(cell, occupied):
        return False
    if _route_name(r, c, red_cols, blue_rows) is None:
        return False

    trial = pixels + [(r, c, PIXEL_COLORS[0])]
    red_counts, blue_counts = _stack_counts(trial, red_cols, blue_rows)
    return (
        max(red_counts.values(), default=0) <= MAX_STACK_PER_LINE
        and max(blue_counts.values(), default=0) <= MAX_STACK_PER_LINE
    )


def _add_pixel(
    pixels: list[tuple[int, int, int]],
    occupied: set[tuple[int, int]],
    cell: tuple[int, int],
    color: int,
) -> None:
    occupied.add(cell)
    pixels.append((cell[0], cell[1], int(color)))


def _sample_from(
    candidates: list[tuple[int, int]],
    pixels: list[tuple[int, int, int]],
    occupied: set[tuple[int, int]],
    red_cols: tuple[int, int],
    blue_rows: tuple[int, int],
) -> tuple[int, int] | None:
    candidates = candidates[:]
    random.shuffle(candidates)
    for cell in candidates:
        if _can_add(cell, pixels, occupied, red_cols, blue_rows):
            return cell
    return None


def _semantic_profile(g: np.ndarray) -> dict[str, object]:
    red_cols, blue_rows, classified = _classified_pixels(g)
    branch_counts = {
        "red_only": 0,
        "blue_only": 0,
        "both_red": 0,
        "both_blue": 0,
        "stay": 0,
    }
    red_groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    blue_groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)

    for r, c, color, route in classified:
        branch = _route_name(r, c, red_cols, blue_rows)
        assert branch is not None
        branch_counts[branch] += 1
        if route == "red":
            red_groups[c].append((r, c, color))
        elif route == "blue":
            blue_groups[r].append((r, c, color))

    red_stack = max((len(v) for v in red_groups.values()), default=0)
    blue_stack = max((len(v) for v in blue_groups.values()), default=0)
    boundary_stays = 0
    for r, c, _color, route in classified:
        if route != "stay":
            continue
        near_red_edge = c in {red_cols[0] - 1, red_cols[1] + 1}
        near_blue_edge = r in {blue_rows[0] - 1, blue_rows[1] + 1}
        if near_red_edge or near_blue_edge:
            boundary_stays += 1

    return {
        "branches": branch_counts,
        "red_stack": red_stack,
        "blue_stack": blue_stack,
        "boundary_stays": boundary_stays,
    }


def _check_solution_semantics(g: np.ndarray, out: np.ndarray) -> bool:
    try:
        red_cols, blue_rows, classified = _classified_pixels(g)
    except AssertionError:
        return False

    if not np.array_equal(out[g == RED], g[g == RED]):
        return False
    if not np.array_equal(out[g == BLUE], g[g == BLUE]):
        return False

    expected_nonzero = int(np.count_nonzero(g))
    if int(np.count_nonzero(out)) != expected_nonzero:
        return False

    red_groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    blue_groups: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for r, c, color, route in classified:
        if route == "red":
            red_groups[c].append((r, c, color))
        elif route == "blue":
            blue_groups[r].append((r, c, color))
        else:
            if int(out[r, c]) != color:
                return False

    for c, group in red_groups.items():
        ordered = sorted(group, key=lambda item: item[0])
        for k, (_r, _c, color) in enumerate(ordered, start=1):
            if int(out[k, c]) != color:
                return False

    for r, group in blue_groups.items():
        ordered = sorted(group, key=lambda item: item[1])
        for k, (_r, _c, color) in enumerate(ordered, start=1):
            if int(out[r, k]) != color:
                return False

    return True


def _valid_generated(g: np.ndarray) -> bool:
    try:
        out = main(g)
        profile = _semantic_profile(g)
    except AssertionError:
        return False

    branches = profile["branches"]
    assert isinstance(branches, dict)
    required = ("red_only", "blue_only", "both_red", "both_blue", "stay")
    if any(int(branches[name]) == 0 for name in required):
        return False
    if int(profile["red_stack"]) < 2 or int(profile["blue_stack"]) < 2:
        return False
    if int(profile["boundary_stays"]) < 2:
        return False
    if np.array_equal(g, out):
        return False
    if len(set(map(int, g.ravel())) - {BLACK, RED, BLUE}) < 4:
        return False
    return _check_solution_semantics(g, out)


def _make_attempt() -> np.ndarray | None:
    H = random.randint(MIN_SIZE, MAX_SIZE)
    W = random.randint(MIN_SIZE, MAX_SIZE)

    max_red_len = min(MAX_SEG_LEN, W - EDGE_GAP - 2)
    max_blue_len = min(MAX_SEG_LEN, H - EDGE_GAP - 2)
    if max_red_len < MIN_SEG_LEN or max_blue_len < MIN_SEG_LEN:
        return None

    red_len = random.randint(MIN_SEG_LEN, max_red_len)
    blue_len = random.randint(MIN_SEG_LEN, max_blue_len)
    red_c0 = random.randint(EDGE_GAP, W - red_len - 2)
    blue_r0 = random.randint(EDGE_GAP, H - blue_len - 2)
    red_cols = (red_c0, red_c0 + red_len - 1)
    blue_rows = (blue_r0, blue_r0 + blue_len - 1)

    red_col_values = list(range(red_cols[0], red_cols[1] + 1))
    blue_row_values = list(range(blue_rows[0], blue_rows[1] + 1))
    both_red_candidates = [(r, c) for r in blue_row_values for c in red_col_values if r < c]
    both_blue_candidates = [(r, c) for r in blue_row_values for c in red_col_values if r > c]
    if not both_red_candidates or not both_blue_candidates:
        return None

    pixels: list[tuple[int, int, int]] = []
    occupied: set[tuple[int, int]] = set()
    colors = PIXEL_COLORS[:]
    random.shuffle(colors)
    color_i = 0

    def next_color(exclude: set[int] | None = None) -> int:
        nonlocal color_i
        exclude = exclude or set()
        choices = [c for c in colors if c not in exclude]
        if not choices:
            choices = colors
        color = choices[color_i % len(choices)]
        color_i += 1
        return int(color)

    red_stack_col = random.choice(red_col_values)
    red_rows = [r for r in _interior_range(H) if not (blue_rows[0] <= r <= blue_rows[1])]
    red_pair_rows = _choose_separated(red_rows, 2)
    if red_pair_rows is None:
        return None
    first_color = next_color()
    second_color = next_color({first_color})
    for r, color in zip(red_pair_rows, (first_color, second_color)):
        cell = (r, red_stack_col)
        if not _can_add(cell, pixels, occupied, red_cols, blue_rows):
            return None
        _add_pixel(pixels, occupied, cell, color)

    blue_stack_row = random.choice(blue_row_values)
    blue_cols = [c for c in _interior_range(W) if not (red_cols[0] <= c <= red_cols[1])]
    blue_pair_cols = _choose_separated(blue_cols, 2)
    if blue_pair_cols is None:
        return None
    first_color = next_color()
    second_color = next_color({first_color})
    for c, color in zip(blue_pair_cols, (first_color, second_color)):
        cell = (blue_stack_row, c)
        if not _can_add(cell, pixels, occupied, red_cols, blue_rows):
            return None
        _add_pixel(pixels, occupied, cell, color)

    cell = _sample_from(both_red_candidates, pixels, occupied, red_cols, blue_rows)
    if cell is None:
        return None
    _add_pixel(pixels, occupied, cell, next_color())

    cell = _sample_from(both_blue_candidates, pixels, occupied, red_cols, blue_rows)
    if cell is None:
        return None
    _add_pixel(pixels, occupied, cell, next_color())

    red_boundary_cols = [c for c in (red_cols[0] - 1, red_cols[1] + 1) if 2 <= c < W - 1]
    blue_boundary_rows = [r for r in (blue_rows[0] - 1, blue_rows[1] + 1) if 2 <= r < H - 1]
    red_boundary_stays = [
        (r, c)
        for c in red_boundary_cols
        for r in _interior_range(H)
        if not (blue_rows[0] <= r <= blue_rows[1])
    ]
    blue_boundary_stays = [
        (r, c)
        for r in blue_boundary_rows
        for c in _interior_range(W)
        if not (red_cols[0] <= c <= red_cols[1])
    ]
    for candidates in (red_boundary_stays, blue_boundary_stays):
        cell = _sample_from(candidates, pixels, occupied, red_cols, blue_rows)
        if cell is None:
            return None
        _add_pixel(pixels, occupied, cell, next_color())

    all_candidates = [
        (r, c)
        for r in _interior_range(H)
        for c in _interior_range(W)
        if _route_name(r, c, red_cols, blue_rows) is not None
    ]
    random.shuffle(all_candidates)
    extra_goal = random.randint(0, 6)
    for cell in all_candidates:
        if extra_goal == 0:
            break
        if _can_add(cell, pixels, occupied, red_cols, blue_rows):
            _add_pixel(pixels, occupied, cell, next_color())
            extra_goal -= 1

    g = np.full((H, W), BLACK, dtype=int)
    g[0, red_cols[0] : red_cols[1] + 1] = RED
    g[blue_rows[0] : blue_rows[1] + 1, 0] = BLUE
    for r, c, color in pixels:
        assert int(g[r, c]) == BLACK
        g[r, c] = color

    return g if _valid_generated(g) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _make_attempt()
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid capture-interval double-gravity input")
