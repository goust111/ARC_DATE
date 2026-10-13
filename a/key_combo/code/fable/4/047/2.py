from __future__ import annotations

import random
from collections import Counter

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
BALL_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.asarray(coords, dtype=int))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _infer_frame_color(grid: np.ndarray, bg: int) -> int:
    counts = Counter(map(int, grid.flatten()))
    counts.pop(int(bg), None)
    assert counts, "expected at least one foreground color"
    return max(counts.items(), key=lambda item: item[1])[0]


def _parse_frames(grid: np.ndarray, frame_color: int) -> list[dict]:
    comps = _components(grid == int(frame_color))
    assert comps, "expected at least one frame"

    frames = []
    for coords in comps:
        r0, c0, r1, c1 = _bbox(coords)
        sub = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        sub[coords[:, 0] - r0, coords[:, 1] - c0] = True

        height, width = sub.shape
        assert height >= 3 and width >= 3, "frame must have a non-empty interior"
        assert np.all(sub[:, 0]) and np.all(sub[:, -1]), "frame sides must be solid"
        assert np.all(sub[-1, :]), "frame bottom edge must be solid"
        if height > 2 and width > 2:
            assert np.all(~sub[1:-1, 1:-1]), "frame interior must be empty"

        top = sub[0]
        gap_candidates = np.flatnonzero(~top)
        assert gap_candidates.size == 1, "expected exactly one top-edge gate"
        gap_idx = int(gap_candidates[0])
        assert 0 < gap_idx < width - 1, "gate must be away from frame corners"
        assert np.all(top[np.arange(width) != gap_idx]), "top edge must be solid except for the gate"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gate": (r0, c0 + gap_idx),
            }
        )

    frames.sort(
        key=lambda frame: (
            -((frame["bbox"][2] - frame["bbox"][0] + 1) * (frame["bbox"][3] - frame["bbox"][1] + 1)),
            frame["bbox"][0],
            frame["bbox"][1],
        )
    )

    for outer, inner in zip(frames, frames[1:]):
        r0, c0, r1, c1 = outer["bbox"]
        rr0, cc0, rr1, cc1 = inner["bbox"]
        assert r0 < rr0 and c0 < cc0 and rr1 < r1 and cc1 < c1, "frames must be strictly nested"

    return frames


def _extract_balls(grid: np.ndarray, bg: int, frame_color: int) -> list[tuple[int, int, int]]:
    mask = (grid != int(bg)) & (grid != int(frame_color))
    balls = [(int(r), int(c), int(grid[r, c])) for r, c in np.argwhere(mask)]
    rows = [r for r, _c, _color in balls]
    assert len(rows) == len(set(rows)), "ball rows must be unique"
    balls.sort(key=lambda item: item[0], reverse=True)
    return balls


def _simulate(
    grid: np.ndarray,
    bg: int,
    frame_color: int,
    frames: list[dict],
    balls: list[tuple[int, int, int]],
) -> tuple[np.ndarray, list[tuple[int, int, int]]]:
    out = np.array(grid, dtype=int, copy=True)
    ball_mask = (out != int(bg)) & (out != int(frame_color))
    out[ball_mask] = int(bg)

    solid = out == int(frame_color)
    settled = np.zeros_like(solid, dtype=bool)
    open_gates = {tuple(frame["gate"]) for frame in frames}
    h, _w = out.shape
    finals: list[tuple[int, int, int]] = []

    for start_r, c, color in balls:
        r = int(start_r)
        while True:
            nr = r + 1
            if nr >= h:
                break
            if (nr, c) in open_gates:
                open_gates.remove((nr, c))
                solid[nr, c] = True
                out[nr, c] = int(frame_color)
                r = nr
                continue
            if solid[nr, c] or settled[nr, c]:
                break
            r = nr

        assert not settled[r, c], "balls may stop on top of objects, not inside them"
        settled[r, c] = True
        out[r, c] = int(color)
        finals.append((r, c, color))

    return out, finals


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop colored balls through nested frame gates that close after first use.

    Input structure:
    - The grid has a dominant background, several nested one-cell-thick hollow rectangular frames,
      and single-cell colored balls above the outer frame.
    - Every frame uses the same border color and has exactly one missing cell on its top edge.
    - Ball rows are distinct, but multiple balls may share a column.

    Rule:
    - Identify the frame color, parse the nested frames, and process the balls from lower start row to higher start row.
    - Each ball falls straight down, stopping on the cell above a frame edge or an already settled ball.
    - If a ball meets an open top-edge gate, it passes through that opening and the gate immediately closes for later balls.
    - Return the final settled balls together with the frames after all used gates have been filled.

    Key insight:
    - The transformation is a sequential simulation, not an independent per-ball projection:
      the first ball to use a gate permanently changes the obstacle map for every later ball.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    frame_color = _infer_frame_color(g, bg)
    frames = _parse_frames(g, frame_color)
    balls = _extract_balls(g, bg, frame_color)
    out, _finals = _simulate(g, bg, frame_color, frames, balls)
    return out


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], gate_col: int, color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, gate_col] = BACKGROUND


def _shift_rows(
    bboxes: list[tuple[int, int, int, int]],
    delta: int,
) -> list[tuple[int, int, int, int]]:
    return [(r0 + delta, c0, r1 + delta, c1) for r0, c0, r1, c1 in bboxes]


def _gen_attempt() -> np.ndarray | None:
    layer_count = random.randint(2, 4)
    transitions = []
    for _ in range(layer_count - 1):
        transitions.append(
            (
                random.randint(2, 3),
                random.randint(2, 3),
                random.randint(2, 3),
                random.randint(2, 3),
            )
        )

    inner_h = random.randint(5, 8)
    inner_w = random.randint(5, 10)
    outer_h = inner_h + sum(top + bottom for top, _left, bottom, _right in transitions)
    outer_w = inner_w + sum(left + right for _top, left, _bottom, right in transitions)

    left_margin = random.randint(1, 4)
    right_margin = random.randint(1, 4)
    bottom_margin = random.randint(1, 3)
    width = left_margin + outer_w + right_margin
    if width > 30:
        return None

    bboxes = []
    bbox = (0, left_margin, outer_h - 1, left_margin + outer_w - 1)
    bboxes.append(bbox)
    for top_gap, left_gap, bottom_gap, right_gap in transitions:
        r0, c0, r1, c1 = bbox
        bbox = (r0 + top_gap, c0 + left_gap, r1 - bottom_gap, c1 - right_gap)
        bboxes.append(bbox)

    allowed_cols = [list(range(c0 + 1, c1)) for _r0, c0, _r1, c1 in bboxes]
    assert all(cols for cols in allowed_cols)

    shared_depth = random.randint(1, layer_count - 1)
    shared_cols = set(allowed_cols[0])
    for idx in range(1, shared_depth):
        shared_cols &= set(allowed_cols[idx])
    if not shared_cols:
        return None
    shaft_col = random.choice(sorted(shared_cols))

    gates = [None] * layer_count
    for idx in range(shared_depth):
        gates[idx] = shaft_col

    mismatch_choices = [col for col in allowed_cols[shared_depth] if col != shaft_col]
    if not mismatch_choices:
        return None
    gates[shared_depth] = random.choice(mismatch_choices)

    for idx in range(shared_depth + 1, layer_count):
        choices = list(allowed_cols[idx])
        if len(choices) > 1 and random.random() < 0.7:
            non_shaft = [col for col in choices if col != shaft_col]
            gates[idx] = random.choice(non_shaft or choices)
        else:
            gates[idx] = random.choice(choices)

    outer_cols = allowed_cols[0]
    blocked_choices = [col for col in outer_cols if col != shaft_col]
    if not blocked_choices:
        return None
    blocked_col = random.choice(blocked_choices)
    extra_choices = [col for col in outer_cols if col not in {shaft_col, blocked_col}]

    ball_columns = [shaft_col, blocked_col, shaft_col, blocked_col]
    if random.random() < 0.8:
        ball_columns.append(shaft_col)
    if extra_choices and random.random() < 0.6:
        insert_at = random.randint(1, len(ball_columns))
        ball_columns.insert(insert_at, random.choice(extra_choices))

    ball_count = len(ball_columns)
    top_margin = ball_count + random.randint(1, 3)
    height = top_margin + outer_h + bottom_margin
    if height > 30:
        return None

    shifted_bboxes = _shift_rows(bboxes, top_margin)
    grid = np.full((height, width), BACKGROUND, dtype=int)
    for bbox, gate_col in zip(shifted_bboxes, gates):
        _draw_frame(grid, bbox, int(gate_col), FRAME_COLOR)

    rows_desc = sorted(random.sample(range(top_margin), ball_count), reverse=True)
    colors = random.sample(BALL_COLORS, ball_count)
    for row, col, color in zip(rows_desc, ball_columns, colors):
        grid[row, col] = int(color)

    try:
        out = main(grid)
        frames = _parse_frames(grid, FRAME_COLOR)
        balls = _extract_balls(grid, BACKGROUND, FRAME_COLOR)
        _replay, finals = _simulate(grid, BACKGROUND, FRAME_COLOR, frames, balls)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None

    closed_gates = sum(int(grid[r, c] == BACKGROUND and out[r, c] == FRAME_COLOR) for r, c in (frame["gate"] for frame in frames))
    if closed_gates < 1:
        return None

    distinct_final_rows = len({r for r, _c, _color in finals})
    if distinct_final_rows < 3:
        return None

    outer_top = shifted_bboxes[0][0]
    deep_ball_exists = any(outer_top < r for r, _c, _color in finals)
    if not deep_ball_exists:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gate-chain input")
