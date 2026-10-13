from __future__ import annotations

import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _frame_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None

    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    if int(r1 - r0) < 2 or int(c1 - c0) < 2:
        return None

    expected = np.zeros_like(mask, dtype=bool)
    expected[r0 : r1 + 1, c0] = True
    expected[r0 : r1 + 1, c1] = True
    expected[r0, c0 : c1 + 1] = True
    expected[r1, c0 : c1 + 1] = True
    if not np.array_equal(mask, expected):
        return None

    return int(r0), int(c0), int(r1), int(c1)


def _paint_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict[str, int | tuple[int, int, int, int]]]]:
    bg = _get_bg(grid)
    frames: list[dict[str, int | tuple[int, int, int, int]]] = []

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        bbox = _frame_bbox(grid == color)
        assert bbox is not None, "every foreground color must form one rectangular frame"
        frames.append({"color": color, "bbox": bbox})

    assert 3 <= len(frames) <= 5, "expected 3 to 5 nested frames"
    frames.sort(key=lambda item: tuple(map(int, item["bbox"])))

    centers = {(int(b[0]) + int(b[2]), int(b[1]) + int(b[3])) for b in (item["bbox"] for item in frames)}
    assert len(centers) == 1, "all frames must share the same center"

    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = map(int, outer["bbox"])
        ir0, ic0, ir1, ic1 = map(int, inner["bbox"])
        assert (ir0, ic0, ir1, ic1) == (or0 + 2, oc0 + 2, or1 - 2, oc1 - 2), "frames must be strictly nested with one-cell gaps"

    ir0, ic0, ir1, ic1 = map(int, frames[-1]["bbox"])
    assert np.all(grid[ir0 + 1 : ir1, ic0 + 1 : ic1] == bg), "innermost frame interior must be background"
    return bg, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reverse the colors of strictly nested rectangular frames while preserving their geometry.
    """
    g = np.asarray(grid, dtype=int)
    _bg, frames = _extract_frames(g)

    out = g.copy()
    reversed_colors = [int(frame["color"]) for frame in reversed(frames)]
    for frame, new_color in zip(frames, reversed_colors):
        _paint_frame(out, tuple(map(int, frame["bbox"])), new_color)
    return out


# --------------- generator ---------------

BG_COLOR = 0
COLOR_POOL = tuple(range(1, 10))
OUTER_BOX_MAX = 26
MARGIN_RANGE = (1, 5)
GENERATION_ATTEMPTS = 300

_PARITY_TOGGLE = 0
_SHAPE_TOGGLE = 0


def _sample_layer_count() -> int:
    global _PARITY_TOGGLE
    want_even = _PARITY_TOGGLE % 2 == 0
    _PARITY_TOGGLE += 1
    return 4 if want_even else random.choice((3, 5))


def _sample_hole_dims(layer_count: int) -> tuple[int, int]:
    global _SHAPE_TOGGLE
    mode = _SHAPE_TOGGLE % 3
    _SHAPE_TOGGLE += 1

    max_dim = OUTER_BOX_MAX - (4 * int(layer_count) - 2)
    assert max_dim >= 1

    if mode == 0:
        base = random.randint(1, max_dim)
        delta = random.randint(0, min(2, max_dim - base))
        if random.random() < 0.5:
            return base, base + delta
        return base + delta, base
    if mode == 1:
        h = random.randint(1, max(1, max_dim - 1))
        w = random.randint(h + 1, max_dim)
        return h, w

    w = random.randint(1, max(1, max_dim - 1))
    h = random.randint(w + 1, max_dim)
    return h, w


def _make_input() -> tuple[np.ndarray, list[tuple[int, int, int, int]], list[int]]:
    layer_count = _sample_layer_count()
    hole_h, hole_w = _sample_hole_dims(layer_count)
    outer_h = hole_h + 4 * layer_count - 2
    outer_w = hole_w + 4 * layer_count - 2

    top = random.randint(*MARGIN_RANGE)
    bottom = random.randint(*MARGIN_RANGE)
    left = random.randint(*MARGIN_RANGE)
    right = random.randint(*MARGIN_RANGE)

    grid = np.full((outer_h + top + bottom, outer_w + left + right), BG_COLOR, dtype=int)
    colors = random.sample(COLOR_POOL, layer_count)
    bboxes: list[tuple[int, int, int, int]] = []

    for depth, color in enumerate(colors):
        r0 = top + 2 * depth
        c0 = left + 2 * depth
        r1 = top + outer_h - 1 - 2 * depth
        c1 = left + outer_w - 1 - 2 * depth
        bbox = (r0, c0, r1, c1)
        bboxes.append(bbox)
        _paint_frame(grid, bbox, color)

    return grid, bboxes, colors


def _expected_output(inp: np.ndarray, bboxes: list[tuple[int, int, int, int]], colors: list[int]) -> np.ndarray:
    out = inp.copy()
    for bbox, color in zip(bboxes, reversed(colors)):
        _paint_frame(out, bbox, color)
    return out


def _gen_attempt() -> np.ndarray | None:
    inp, bboxes, colors = _make_input()
    expected = _expected_output(inp, bboxes, colors)

    if _get_bg(inp) != BG_COLOR:
        return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid nested-frame puzzle")
