from __future__ import annotations

from collections import Counter
import random

import numpy as np


BACKGROUND = 0
ANCHOR_COLOR = 2
LEFT_HINGE_COLOR = 3
RIGHT_HINGE_COLOR = 7
LINE_COLOR = 8

TURN_BY_COLOR = {
    LEFT_HINGE_COLOR: "left",
    RIGHT_HINGE_COLOR: "right",
}

DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GENERATION_RETRIES = 500


def _rotate_left(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return -dc, dr


def _rotate_right(vec: tuple[int, int]) -> tuple[int, int]:
    dr, dc = vec
    return dc, -dr


def _rotate_vec(vec: tuple[int, int], turn: str) -> tuple[int, int]:
    if turn == "left":
        return _rotate_left(vec)
    if turn == "right":
        return _rotate_right(vec)
    raise ValueError(f"unknown turn: {turn}")


def _most_common_color(values: list[int]) -> int:
    counts = Counter(map(int, values))
    common = counts.most_common()
    assert common
    if len(common) > 1:
        assert common[0][1] > common[1][1], "line color must be uniquely most frequent"
    return int(common[0][0])


def _ordered_input_chain(grid: np.ndarray) -> tuple[int, int, int, list[tuple[int, int]], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g.ravel().tolist())
    coords = np.argwhere(g != bg)
    assert coords.size > 0, "expected a non-background rod"

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if r0 == r1:
        ordered = [(int(r0), int(c)) for c in sorted(map(int, coords[:, 1]))]
        assert len(ordered) == int(c1 - c0 + 1), "horizontal rod must be gap-free"
    else:
        assert c0 == c1, "rod must be straight"
        ordered = [(int(r), int(c0)) for r in sorted(map(int, coords[:, 0]))]
        assert len(ordered) == int(r1 - r0 + 1), "vertical rod must be gap-free"

    colors = [int(g[r, c]) for r, c in ordered]
    line_color = _most_common_color(colors)

    endpoint_ids = [0, len(ordered) - 1]
    anchor_ids = [idx for idx in endpoint_ids if colors[idx] != line_color]
    assert len(anchor_ids) == 1, "exactly one endpoint must be the anchor"
    if anchor_ids[0] == len(ordered) - 1:
        ordered.reverse()
        colors.reverse()

    anchor_color = int(colors[0])
    assert anchor_color != line_color
    extra_colors = {color for color in colors[1:] if color != line_color}
    assert extra_colors <= set(TURN_BY_COLOR), "unexpected hinge color"
    return bg, line_color, anchor_color, ordered, colors


def _render_path(
    shape: tuple[int, int],
    bg: int,
    positions: list[tuple[int, int]],
    colors: list[int],
    anchor_color: int,
) -> np.ndarray:
    out = np.full(shape, bg, dtype=int)
    cover = Counter(positions)
    for (r, c), color in zip(positions, colors):
        out[r, c] = anchor_color if cover[(r, c)] >= 2 else int(color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fold the marked straight rod hinge by hinge and draw the final polyline.

    Input structure:
    - The grid has a dominant background and one one-cell-wide straight rod.
    - One endpoint is a unique anchor pixel, the rod body is mostly one line color,
      and some body cells are hinge markers in two fixed hinge colors.
    - The rod is initially horizontal or vertical, with all marked cells lying on
      the same uninterrupted row or column.

    Rule:
    - Start at the anchor and traverse the rod toward the free end.
    - When a hinge is reached, rotate the entire still-unfolded suffix around that
      hinge by ninety degrees; the two hinge colors encode left turn versus right turn.
    - After all hinges have acted in order, render the final chain: anchor keeps its
      anchor color, hinges keep their hinge colors, ordinary rod cells keep the line
      color, and any cell occupied by two or more final rod cells is recolored to the
      anchor color.

    Key insight:
    - Each hinge changes the local coordinate frame for all later hinges, so the task
      is a sequential suffix-rotation simulation rather than a single global transform.
    """
    g = np.asarray(grid, dtype=int)
    bg, line_color, anchor_color, ordered, colors = _ordered_input_chain(g)
    assert line_color != anchor_color

    positions = list(ordered)
    for pivot_idx in range(1, len(positions) - 1):
        turn = TURN_BY_COLOR.get(colors[pivot_idx])
        if turn is None:
            continue
        pivot_r, pivot_c = positions[pivot_idx]
        for j in range(pivot_idx + 1, len(positions)):
            rel = (positions[j][0] - pivot_r, positions[j][1] - pivot_c)
            rot = _rotate_vec(rel, turn)
            positions[j] = (pivot_r + rot[0], pivot_c + rot[1])

    h, w = g.shape
    assert all(0 <= r < h and 0 <= c < w for r, c in positions), "fold left the grid bounds"
    return _render_path(g.shape, bg, positions, colors, anchor_color)


def _build_straight_input(
    initial_dir: tuple[int, int],
    segment_steps: list[int],
    turns: list[str],
) -> tuple[list[tuple[int, int]], list[int]]:
    hinge_step_to_turn = {}
    walked = 0
    for steps, turn in zip(segment_steps[:-1], turns):
        walked += steps
        hinge_step_to_turn[walked] = turn

    coords = [(0, 0)]
    colors = [ANCHOR_COLOR]
    dr, dc = initial_dir
    for step in range(1, sum(segment_steps) + 1):
        r, c = coords[-1]
        coords.append((r + dr, c + dc))
        turn = hinge_step_to_turn.get(step)
        if turn == "left":
            colors.append(LEFT_HINGE_COLOR)
        elif turn == "right":
            colors.append(RIGHT_HINGE_COLOR)
        else:
            colors.append(LINE_COLOR)
    return coords, colors


def _build_final_chain(
    initial_dir: tuple[int, int],
    segment_steps: list[int],
    turns: list[str],
) -> tuple[list[tuple[int, int]], list[int]]:
    coords = [(0, 0)]
    colors = [ANCHOR_COLOR]
    direction = initial_dir

    for seg_idx, steps in enumerate(segment_steps):
        for step in range(steps):
            r, c = coords[-1]
            coords.append((r + direction[0], c + direction[1]))
            is_hinge = seg_idx < len(turns) and step == steps - 1
            if not is_hinge:
                colors.append(LINE_COLOR)
            elif turns[seg_idx] == "left":
                colors.append(LEFT_HINGE_COLOR)
            else:
                colors.append(RIGHT_HINGE_COLOR)
        if seg_idx < len(turns):
            direction = _rotate_vec(direction, turns[seg_idx])
    return coords, colors


def _translate(coords: list[tuple[int, int]], offset: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = offset
    return [(r + dr, c + dc) for r, c in coords]


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _sample_turns(hinge_count: int) -> list[str]:
    if hinge_count == 1:
        return [random.choice(["left", "right"])]

    for _ in range(20):
        turns = [random.choice(["left", "right"]) for _ in range(hinge_count)]
        if len(set(turns)) == 2 or random.random() < 0.25:
            return turns
    return turns


def _sample_overlap_case() -> tuple[tuple[int, int], list[int], list[str]]:
    initial_dir = random.choice(DIRS)
    turn = random.choice(["left", "right"])
    first = random.randint(3, 6)
    second = random.randint(2, 5)
    third = random.randint(1, first - 1)
    fourth = second
    return initial_dir, [first, second, third, fourth], [turn, turn, turn]


def _sample_nonoverlap_case() -> tuple[tuple[int, int], list[int], list[str]]:
    hinge_count = random.randint(1, 4)
    segment_steps = [random.randint(2, 5) for _ in range(hinge_count + 1)]
    initial_dir = random.choice(DIRS)
    turns = _sample_turns(hinge_count)
    return initial_dir, segment_steps, turns


def _gen_attempt() -> np.ndarray | None:
    want_overlap = random.random() < 0.45
    if want_overlap:
        initial_dir, segment_steps, turns = _sample_overlap_case()
    else:
        initial_dir, segment_steps, turns = _sample_nonoverlap_case()

    final_rel, final_colors = _build_final_chain(initial_dir, segment_steps, turns)
    cover = Counter(final_rel)
    overlap_cells = [cell for cell, count in cover.items() if count >= 2]
    has_overlap = bool(overlap_cells)
    if has_overlap != want_overlap:
        return None
    if len(overlap_cells) > 3 or max(cover.values()) > 2:
        return None
    if any(cover[cell] >= 2 for cell, color in zip(final_rel, final_colors) if color in TURN_BY_COLOR):
        return None

    final_box = _bbox(final_rel)
    if final_box[0] == final_box[2] or final_box[1] == final_box[3]:
        return None

    input_rel, input_colors = _build_straight_input(initial_dir, segment_steps, turns)
    all_coords = final_rel + input_rel
    r0, c0, r1, c1 = _bbox(all_coords)
    top = random.randint(1, 4)
    left = random.randint(1, 4)
    bottom = random.randint(1, 4)
    right = random.randint(1, 4)
    height = (r1 - r0 + 1) + top + bottom
    width = (c1 - c0 + 1) + left + right
    if max(height, width) > 30:
        return None

    offset = (top - r0, left - c0)
    input_abs = _translate(input_rel, offset)
    final_abs = _translate(final_rel, offset)

    inp = np.full((height, width), BACKGROUND, dtype=int)
    for (r, c), color in zip(input_abs, input_colors):
        inp[r, c] = color

    expected = _render_path(inp.shape, BACKGROUND, final_abs, final_colors, ANCHOR_COLOR)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fold-chain input")
