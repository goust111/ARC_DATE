from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0

_GEN_CALL_COUNT = 0
_CASE_CYCLE: tuple[tuple[str, bool], ...] = (
    ("split", True),
    ("split", False),
    ("thick", True),
    ("thick", False),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
    return count


def _peel_rounds(mask: np.ndarray) -> tuple[np.ndarray, int]:
    current = np.array(mask, dtype=bool, copy=True)
    rounds = np.zeros(mask.shape, dtype=int)
    step = 0
    h, w = mask.shape
    while bool(current.any()):
        step += 1
        boundary = np.zeros_like(current)
        for r, c in np.argwhere(current):
            rr, cc = int(r), int(c)
            for nr, nc in _neighbors4(rr, cc):
                if not (0 <= nr < h and 0 <= nc < w) or not current[nr, nc]:
                    boundary[rr, cc] = True
                    break
        assert bool(boundary.any()), "peeling stalled on a non-empty object"
        rounds[boundary] = step
        current[boundary] = False
    return rounds, step


def _row_runs(legend: np.ndarray) -> list[tuple[int, int]]:
    if legend.size == 0:
        raise AssertionError("legend must be non-empty")
    h, _w = legend.shape
    row_colors: list[int] = []
    for r in range(h):
        color = int(legend[r, 0])
        if not bool((legend[r] == color).all()):
            raise AssertionError("legend rows must be full-width solid stripes")
        row_colors.append(color)

    runs: list[tuple[int, int]] = []
    r = 0
    while r < h:
        color = row_colors[r]
        r2 = r + 1
        while r2 < h and row_colors[r2] == color:
            r2 += 1
        runs.append((color, r2 - r))
        r = r2
    return runs


def _parse_instance(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, int]], int, bool]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise AssertionError("expected a 2D grid")
    h, w = g.shape
    candidates: list[tuple[np.ndarray, list[tuple[int, int]], int, bool]] = []

    for sep_c in range(w):
        sep_color = int(g[0, sep_c])
        if sep_color == BLACK or not bool((g[:, sep_c] == sep_color).all()):
            continue
        legend = g[:, :sep_c]
        canvas = g[:, sep_c + 1 :]
        if legend.shape[1] == 0 or canvas.shape[1] == 0:
            continue
        try:
            runs = _row_runs(legend)
        except AssertionError:
            continue
        if len(runs) < 2:
            continue
        if any(not (1 <= height <= 3) for _color, height in runs):
            continue
        run_colors = [color for color, _height in runs]
        if len(set(run_colors)) != len(run_colors):
            continue
        canvas_colors = set(map(int, np.unique(canvas)))
        if BLACK not in canvas_colors:
            continue
        fg_colors = sorted(canvas_colors - {BLACK})
        if len(fg_colors) != 1:
            continue
        obj_color = fg_colors[0]
        anchor_indices = [i for i, (color, _height) in enumerate(runs) if color == obj_color]
        if len(anchor_indices) != 1:
            continue
        anchor_idx = anchor_indices[0]
        if anchor_idx not in (0, len(runs) - 1):
            continue
        mask = canvas == obj_color
        if _component_count(mask) != 1:
            continue
        candidates.append((canvas, runs, obj_color, anchor_idx == 0))

    assert len(candidates) == 1, "expected exactly one valid separator/legend interpretation"
    return candidates[0]


def _render_output(canvas: np.ndarray, runs: list[tuple[int, int]], obj_color: int, anchor_top: bool) -> np.ndarray:
    out = np.array(canvas, dtype=int, copy=True)
    mask = out == obj_color
    rounds, depth = _peel_rounds(mask)

    read_runs = list(reversed(runs[1:])) if anchor_top else list(runs[:-1])
    total_recolored = sum(height for _color, height in read_runs)
    assert total_recolored < depth, "anchor must leave a non-empty core"

    cursor = 1
    for color, height in read_runs:
        band = (rounds >= cursor) & (rounds < cursor + height)
        assert bool(band.any()), "each stripe must recolor at least one peel layer"
        out[band] = int(color)
        cursor += height
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor an onion-peeled solid object using stripe groups from a side legend.

    Input structure:
    - A single full-height separator column splits the grid into a narrow left legend and a right main canvas.
    - The legend is a vertical stack of full-width horizontal stripes with distinct colors and heights 1-3.
    - Exactly one stripe matches the unique monochrome 4-connected solid object on the main canvas, and that matching stripe is either the topmost or bottommost stripe.
    - The rest of the main canvas is black background.

    Rule:
    - Repeatedly peel the object by 4-neighbor boundary layers from outside to inside.
    - Read the legend from the end farthest from the matching-color anchor stripe.
    - Each non-anchor stripe recolors a consecutive group of peel layers whose thickness equals that stripe's height.
    - When the anchor stripe is reached, stop recoloring and leave the remaining inner core in the object's original color.

    Key insight:
    - The anchor stripe does not paint anything directly; it only fixes which end of the stripe stack is the core end, so stripe order and stripe heights jointly determine the object's outer-to-inner recoloring.
    """
    canvas, runs, obj_color, anchor_top = _parse_instance(np.asarray(grid, dtype=int))
    return _render_output(canvas, runs, obj_color, anchor_top)


def _dilate4(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.array(mask, dtype=bool, copy=True)
    for r, c in np.argwhere(mask):
        rr, cc = int(r), int(c)
        for nr, nc in _neighbors4(rr, cc):
            if 0 <= nr < h and 0 <= nc < w:
                out[nr, nc] = True
    return out


def _build_object_from_line(core_length: int, peel_depth: int) -> np.ndarray:
    core = np.zeros((1, core_length), dtype=bool)
    core[0, :] = True
    mask = np.pad(core, ((peel_depth, peel_depth), (peel_depth, peel_depth)), mode="constant")
    for _ in range(peel_depth):
        mask = _dilate4(mask)
    return mask


def _place_mask(canvas: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    canvas[top : top + h, left : left + w][mask] = int(color)


def _rand_colors(count: int) -> list[int]:
    return random.sample(list(range(1, 10)), count)


def _build_grid(case_name: str, anchor_top: bool) -> np.ndarray:
    peel_depth = 2
    anchor_height = 3
    height = peel_depth + anchor_height
    core_length = random.randint(1, 7)
    object_mask = _build_object_from_line(core_length, peel_depth)
    obj_h, obj_w = object_mask.shape
    assert obj_h == height

    canvas_width = random.randint(obj_w + 1, min(20, obj_w + 8))
    left_pad = random.randint(0, canvas_width - obj_w)

    legend_width = random.randint(2, 4)
    if case_name == "split":
        obj_color, outer_color, inner_color, sep_color = _rand_colors(4)
        read_runs = [(outer_color, 1), (inner_color, 1)]
    else:
        obj_color, outer_color, sep_color = _rand_colors(3)
        read_runs = [(outer_color, 2)]

    top_runs = [(obj_color, anchor_height), *reversed(read_runs)] if anchor_top else [*read_runs, (obj_color, anchor_height)]

    legend = np.zeros((height, legend_width), dtype=int)
    start = 0
    for color, stripe_h in top_runs:
        legend[start : start + stripe_h, :] = int(color)
        start += stripe_h
    assert start == height

    canvas = np.zeros((height, canvas_width), dtype=int)
    _place_mask(canvas, object_mask, 0, left_pad, obj_color)

    separator = np.full((height, 1), sep_color, dtype=int)
    grid = np.hstack([legend, separator, canvas])
    assert max(grid.shape) <= 30
    return grid


def _reversed_non_anchor_output(canvas: np.ndarray, runs: list[tuple[int, int]], obj_color: int, anchor_top: bool) -> np.ndarray:
    if anchor_top:
        modified_runs = [runs[0], *runs[:0:-1]]
    else:
        modified_runs = [*runs[:-1][::-1], runs[-1]]
    return _render_output(canvas, modified_runs, obj_color, anchor_top)


def _validate_generated_instance(grid: np.ndarray, case_name: str) -> None:
    canvas, runs, obj_color, anchor_top = _parse_instance(grid)
    mask = canvas == obj_color
    rounds, depth = _peel_rounds(mask)
    assert depth >= 3
    out = main(grid)
    assert out.shape == canvas.shape
    assert not np.array_equal(out, canvas)

    read_runs = list(reversed(runs[1:])) if anchor_top else list(runs[:-1])
    total_recolored = sum(height for _color, height in read_runs)
    assert total_recolored == 2
    assert bool((rounds == 1).any()) and bool((rounds == 2).any()) and bool((rounds >= 3).any())

    if case_name == "split":
        assert len(read_runs) == 2 and read_runs[0][1] == read_runs[1][1] == 1
        alt = _reversed_non_anchor_output(canvas, runs, obj_color, anchor_top)
        assert not np.array_equal(out, alt), "anchor orientation must matter in split cases"
        outer_color, inner_color = read_runs[0][0], read_runs[1][0]
        assert outer_color != inner_color
        assert set(map(int, np.unique(out[rounds == 1]))) == {outer_color}
        assert set(map(int, np.unique(out[rounds == 2]))) == {inner_color}
    else:
        assert len(read_runs) == 1 and read_runs[0][1] == 2
        outer_color = read_runs[0][0]
        assert set(map(int, np.unique(out[(rounds == 1) | (rounds == 2)]))) == {outer_color}

    assert set(map(int, np.unique(out[rounds >= 3]))) == {obj_color}


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    for _ in range(300):
        case_name, anchor_top = _CASE_CYCLE[_GEN_CALL_COUNT % len(_CASE_CYCLE)]
        _GEN_CALL_COUNT += 1
        grid = _build_grid(case_name, anchor_top)
        try:
            _validate_generated_instance(grid, case_name)
        except AssertionError:
            continue
        return grid
    raise RuntimeError("failed to generate a valid onion-peeling instance")
