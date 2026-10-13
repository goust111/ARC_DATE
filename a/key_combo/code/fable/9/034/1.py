from __future__ import annotations

import random

import numpy as np


BG = 0
TAIL = 5
DIAGONALS = ((-1, -1), (-1, 1), (1, -1), (1, 1))
HEAD_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRY_LIMIT = 500


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _is_edge(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return r == 0 or c == 0 or r == h - 1 or c == w - 1


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = np.zeros_like(mask, dtype=bool)
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            rs = slice(max(0, dr), mask.shape[0] + min(0, dr))
            cs = slice(max(0, dc), mask.shape[1] + min(0, dc))
            rt = slice(max(0, -dr), mask.shape[0] + min(0, -dr))
            ct = slice(max(0, -dc), mask.shape[1] + min(0, -dc))
            out[rt, ct] |= mask[rs, cs]
    return out


def _trace_ray(
    shape: tuple[int, int],
    head: tuple[int, int],
    direction: tuple[int, int],
    stop_hits: int,
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    h, w = shape
    r, c = head
    dr, dc = direction
    path: list[tuple[int, int]] = []
    hits: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()

    while True:
        r += dr
        c += dc
        assert 0 <= r < h and 0 <= c < w, "ray stepped outside the grid"
        cell = (r, c)
        assert cell not in seen, "ray revisited a cell before stopping"
        seen.add(cell)
        path.append(cell)

        if _is_edge(shape, r, c):
            hits.append(cell)
            if len(hits) == stop_hits:
                return path, hits
            if r == 0 or r == h - 1:
                dr = -dr
            if c == 0 or c == w - 1:
                dc = -dc


def _head_neighbors(grid: np.ndarray, head: tuple[int, int]) -> list[tuple[int, int]]:
    bg = _get_bg(grid)
    r, c = head
    out = []
    for rr in range(max(0, r - 1), min(grid.shape[0], r + 2)):
        for cc in range(max(0, c - 1), min(grid.shape[1], c + 2)):
            if (rr, cc) != (r, c) and int(grid[rr, cc]) != bg:
                out.append((rr, cc))
    return out


def _parse_emitters(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    heads = np.argwhere((g != bg) & (g != TAIL))
    assert heads.size > 0, "expected at least one colored head pixel"

    emitters: list[dict[str, object]] = []
    claimed_tail: set[tuple[int, int]] = set()

    for r_raw, c_raw in heads:
        r = int(r_raw)
        c = int(c_raw)
        color = int(g[r, c])
        tail_steps = []
        for tr, tc in DIAGONALS:
            rr = r + tr
            cc = c + tc
            if _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == TAIL:
                tail_steps.append((tr, tc))
        assert len(tail_steps) == 1, "each head must attach to exactly one diagonal tail"

        tail_step = tail_steps[0]
        neighbors = _head_neighbors(g, (r, c))
        assert neighbors == [(r + tail_step[0], c + tail_step[1])], "head must only touch its own tail"

        tail: list[tuple[int, int]] = []
        rr = r + tail_step[0]
        cc = c + tail_step[1]
        while _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == TAIL:
            cell = (rr, cc)
            assert cell not in claimed_tail, "tails must not overlap"
            claimed_tail.add(cell)
            tail.append(cell)
            rr += tail_step[0]
            cc += tail_step[1]

        assert 1 <= len(tail) <= 4, "tail length must encode 1 to 4 wall hits"
        assert _in_bounds(g.shape, rr, cc) and int(g[rr, cc]) == bg, "tail must end cleanly in background"

        emitters.append(
            {
                "head": (r, c),
                "color": color,
                "direction": (-tail_step[0], -tail_step[1]),
                "tail": tail,
                "hits": len(tail),
            }
        )

    gray_cells = {tuple(map(int, coord)) for coord in np.argwhere(g == TAIL)}
    assert gray_cells == claimed_tail, "all grey cells must belong to emitter tails"
    return emitters


def main(grid: np.ndarray) -> np.ndarray:
    """
    Launch each colored head along the diagonal indicated by its grey tail and stop on the tail-length-th wall hit.

    Input structure:
    - A black background contains several isolated emitters.
    - Each emitter is one colored head pixel attached to one straight grey tail bar on a diagonal.
    - Tail bars have length 1 to 4 and different emitters do not touch.

    Rule:
    - For each emitter, move from the head one cell at a time in the diagonal direction from tail toward head.
    - Paint every visited background cell with the head color; when the ray reaches a border cell, reflect by flipping the touched axis component and continue.
    - If the tail has length L, stop exactly on the border cell of the L-th wall contact, while leaving the original head and grey tail unchanged.

    Key insight:
    - The grey tail carries both pieces of information: its diagonal orientation determines the launch direction, and its length determines how many reflections the ray survives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    emitters = _parse_emitters(g)
    out = g.copy()

    for emitter in emitters:
        head = emitter["head"]
        color = int(emitter["color"])
        direction = emitter["direction"]
        hits = int(emitter["hits"])
        path, _ = _trace_ray(g.shape, head, direction, hits)

        for rr, cc in path:
            assert int(g[rr, cc]) == bg, "ray paths must travel only through background"
            assert int(out[rr, cc]) == bg, "ray paths must not overlap"
            out[rr, cc] = color

    return out


def _sample_lengths(count: int) -> list[int]:
    if count == 2:
        lengths = [random.choice((1, 2)), random.choice((3, 4))]
    elif count == 3:
        lengths = random.sample([1, 2, 3, 4], 3)
    else:
        lengths = [1, 2, 3, 4]
    random.shuffle(lengths)
    return lengths


def _candidate_emitter(
    shape: tuple[int, int],
    head: tuple[int, int],
    direction: tuple[int, int],
    tail_len: int,
) -> dict[str, object] | None:
    h, w = shape
    r, c = head
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return None

    tail = []
    for k in range(1, tail_len + 1):
        rr = r - k * direction[0]
        cc = c - k * direction[1]
        if not _in_bounds(shape, rr, cc) or _is_edge(shape, rr, cc):
            return None
        tail.append((rr, cc))

    try:
        path, hits = _trace_ray(shape, head, direction, tail_len)
    except AssertionError:
        return None

    if len(path) < 3:
        return None
    if any(_is_edge(shape, hr, hc) and ((hr in (0, h - 1)) and (hc in (0, w - 1))) for hr, hc in hits):
        return None

    input_cells = [head, *tail]
    input_set = set(input_cells)
    if any(cell in input_set for cell in path):
        return None

    for idx, (pr, pc) in enumerate(path):
        for tr, tc in tail:
            if max(abs(pr - tr), abs(pc - tc)) <= 1:
                return None
        if idx > 0 and max(abs(pr - head[0]), abs(pc - head[1])) <= 1:
            return None

    return {
        "head": head,
        "direction": direction,
        "tail_len": tail_len,
        "tail": tail,
        "path": path,
        "output_cells": input_cells + path,
        "input_cells": input_cells,
    }


def _coords_to_mask(shape: tuple[int, int], coords: list[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[r, c] = True
    return mask


def _place_emitters(shape: tuple[int, int], lengths: list[int], directions: list[tuple[int, int]]) -> list[dict[str, object]] | None:
    input_mask = np.zeros(shape, dtype=bool)
    output_mask = np.zeros(shape, dtype=bool)
    placed: list[dict[str, object]] = []

    order = sorted(range(len(lengths)), key=lambda idx: lengths[idx], reverse=True)
    for idx in order:
        tail_len = lengths[idx]
        direction = directions[idx]
        blocked_input = _dilate8(input_mask)
        blocked_output = _dilate8(output_mask)
        candidates = []

        heads = [
            (r, c)
            for r in range(1, shape[0] - 1)
            for c in range(1, shape[1] - 1)
        ]
        random.shuffle(heads)

        for head in heads:
            cand = _candidate_emitter(shape, head, direction, tail_len)
            if cand is None:
                continue

            if any(blocked_input[r, c] for r, c in cand["input_cells"]):
                continue
            if any(blocked_output[r, c] for r, c in cand["output_cells"]):
                continue
            candidates.append(cand)

        if not candidates:
            return None

        chosen = random.choice(candidates)
        placed.append(chosen)
        for r, c in chosen["input_cells"]:
            input_mask[r, c] = True
            output_mask[r, c] = True
        for r, c in chosen["path"]:
            output_mask[r, c] = True

    return placed


def _semantic_checks(grid: np.ndarray) -> bool:
    try:
        emitters = _parse_emitters(grid)
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(grid, out):
        return False

    lengths = {int(emitter["hits"]) for emitter in emitters}
    directions = {tuple(emitter["direction"]) for emitter in emitters}
    if len(emitters) < 2 or len(directions) < 2:
        return False
    if max(lengths) < 2:
        return False

    bg = _get_bg(grid)
    colored_heads = {int(emitter["color"]) for emitter in emitters}
    if len(colored_heads) != len(emitters):
        return False

    painted = np.argwhere((out != bg) & (out != grid))
    if painted.size == 0:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(12, 24)
    w = random.randint(12, 24)
    count = random.randint(2, 4)
    lengths = _sample_lengths(count)
    directions = list(DIAGONALS)
    random.shuffle(directions)
    directions = directions[:count]

    placed = _place_emitters((h, w), lengths, directions)
    if placed is None:
        return None

    colors = random.sample(HEAD_COLORS, count)
    grid = np.full((h, w), BG, dtype=int)
    for emitter, color in zip(placed, colors):
        hr, hc = emitter["head"]
        grid[hr, hc] = color
        for tr, tc in emitter["tail"]:
            grid[tr, tc] = TAIL

    return grid if _semantic_checks(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid tail-billiards input")
