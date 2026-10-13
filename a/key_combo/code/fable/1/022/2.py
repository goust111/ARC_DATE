from __future__ import annotations

import itertools
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _frame_mask(shape: tuple[int, int], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = map(int, bbox)
    mask = np.zeros(shape, dtype=bool)
    mask[r0 : r1 + 1, c0] = True
    mask[r0 : r1 + 1, c1] = True
    mask[r0, c0 : c1 + 1] = True
    mask[r1, c0 : c1 + 1] = True
    return mask


def _render_stack(
    shape: tuple[int, int],
    masks: list[np.ndarray],
    colors: list[int],
    order: tuple[int, ...],
    bg: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for idx in order:
        out[masks[idx]] = int(colors[idx])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild each occluded hollow rectangle, count its hidden border cells, and redraw the stack by that hidden-area order.

    Input structure:
    - The background is a single dominant color.
    - Each foreground color belongs to one one-cell-thick hollow rectangle frame.
    - Frames overlap, so each overlap cell shows only the color of the topmost frame, while every frame still has some visible pixels.

    Rule:
    - For each foreground color, take the bbox of its visible pixels and reconstruct the full rectangular border of that bbox.
    - Count how many cells of that full border are not visible in the input for that color.
    - Redraw all reconstructed frames on a blank canvas of the same size from smallest hidden-count to largest hidden-count, so the most hidden frame ends up on top.

    Key insight:
    - The new z-order is not the original one or its reverse; it is determined only after reconstructing each full frame and counting how much of that frame was hidden.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    colors = sorted(int(color) for color in np.unique(g) if int(color) != bg)
    assert colors, "expected at least one foreground frame"

    masks: dict[int, np.ndarray] = {}
    hidden_counts: dict[int, int] = {}
    for color in colors:
        visible = g == int(color)
        box = _bbox(visible)
        assert box is not None, "each frame color must remain visible"
        frame = _frame_mask(g.shape, box)
        assert np.all(frame[visible]), "visible pixels must lie on the reconstructed rectangle border"
        masks[color] = frame
        hidden_counts[color] = int(frame.sum() - visible.sum())

    assert len(set(hidden_counts.values())) == len(hidden_counts), "hidden counts must be unique"

    out = np.full_like(g, bg)
    for color in sorted(colors, key=lambda value: hidden_counts[value]):
        out[masks[color]] = int(color)
    return out


GEN_BG = 0
GEN_COLORS = tuple(range(1, 10))
GEN_HEIGHT_RANGE = (14, 24)
GEN_WIDTH_RANGE = (14, 24)
GEN_FRAME_COUNTS = (3, 4, 5)
GEN_EDGE_GAP = 1
GEN_SIDE_MIN = 4
GEN_SIDE_MAX = 11
GEN_PAIR_OVERLAP_CAP = 6
GEN_FRAME_PLACEMENT_TRIES = 400
GENERATION_RETRIES = 300


def _randint(rng: np.random.Generator, lo: int, hi: int) -> int:
    return int(rng.integers(int(lo), int(hi) + 1))


def _sample_bboxes(
    rng: np.random.Generator, shape: tuple[int, int], count: int
) -> tuple[list[tuple[int, int, int, int]], list[np.ndarray]] | None:
    height, width = map(int, shape)
    bboxes: list[tuple[int, int, int, int]] = []
    masks: list[np.ndarray] = []

    max_h = min(GEN_SIDE_MAX, height - 2 * GEN_EDGE_GAP)
    max_w = min(GEN_SIDE_MAX, width - 2 * GEN_EDGE_GAP)
    if max_h < GEN_SIDE_MIN or max_w < GEN_SIDE_MIN:
        return None

    for _ in range(GEN_FRAME_PLACEMENT_TRIES):
        frame_h = _randint(rng, GEN_SIDE_MIN, max_h)
        frame_w = _randint(rng, GEN_SIDE_MIN, max_w)
        r0 = _randint(rng, GEN_EDGE_GAP, height - GEN_EDGE_GAP - frame_h)
        c0 = _randint(rng, GEN_EDGE_GAP, width - GEN_EDGE_GAP - frame_w)
        bbox = (r0, c0, r0 + frame_h - 1, c0 + frame_w - 1)
        if bbox in bboxes:
            continue

        mask = _frame_mask(shape, bbox)
        if any(int((mask & old).sum()) > GEN_PAIR_OVERLAP_CAP for old in masks):
            continue

        bboxes.append(bbox)
        masks.append(mask)
        if len(bboxes) == count:
            break

    if len(bboxes) != count:
        return None

    cover = np.zeros(shape, dtype=int)
    for mask in masks:
        cover += mask.astype(int)
    if int(cover.max()) > 2:
        return None
    if int(np.count_nonzero(cover >= 2)) < count - 1:
        return None

    seen = {0}
    stack = [0]
    while stack:
        cur = stack.pop()
        for nxt in range(count):
            if nxt in seen or nxt == cur:
                continue
            if int((masks[cur] & masks[nxt]).sum()) > 0:
                seen.add(nxt)
                stack.append(nxt)
    if len(seen) != count:
        return None

    return bboxes, masks


def _truth_from_geometry(
    shape: tuple[int, int],
    bboxes: list[tuple[int, int, int, int]],
    masks: list[np.ndarray],
    colors: list[int],
    input_order: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray] | None:
    inp = _render_stack(shape, masks, colors, input_order, GEN_BG)
    if int(np.count_nonzero(inp != GEN_BG)) > (shape[0] * shape[1]) // 2:
        return None

    hidden_counts: list[int] = []
    visible_counts: list[int] = []
    border_sizes: list[int] = []
    for idx, color in enumerate(colors):
        visible = inp == int(color)
        if _bbox(visible) != bboxes[idx]:
            return None
        visible_count = int(visible.sum())
        border_size = int(masks[idx].sum())
        hidden_counts.append(border_size - visible_count)
        visible_counts.append(visible_count)
        border_sizes.append(border_size)

    if len(set(hidden_counts)) != len(hidden_counts):
        return None

    output_order = tuple(idx for _, idx in sorted((hidden_counts[idx], idx) for idx in range(len(colors))))
    out = _render_stack(shape, masks, colors, output_order, GEN_BG)

    reverse_out = _render_stack(shape, masks, colors, tuple(reversed(input_order)), GEN_BG)
    if np.array_equal(out, inp) or np.array_equal(out, reverse_out):
        return None

    if len(set(visible_counts)) == len(visible_counts):
        visible_order = tuple(idx for _, idx in sorted((-visible_counts[idx], idx) for idx in range(len(colors))))
        if visible_order == output_order:
            return None

    if len(set(border_sizes)) == len(border_sizes):
        size_order = tuple(idx for _, idx in sorted((border_sizes[idx], idx) for idx in range(len(colors))))
        if size_order == output_order:
            return None

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    shape = (
        _randint(rng, *GEN_HEIGHT_RANGE),
        _randint(rng, *GEN_WIDTH_RANGE),
    )
    count = int(rng.choice(np.array(GEN_FRAME_COUNTS, dtype=int)))
    sampled = _sample_bboxes(rng, shape, count)
    if sampled is None:
        return None
    bboxes, masks = sampled

    colors = random.sample(list(GEN_COLORS), count)
    orders = list(itertools.permutations(range(count)))
    random.shuffle(orders)
    for input_order in orders:
        truth = _truth_from_geometry(shape, bboxes, masks, colors, input_order)
        if truth is not None:
            inp, _out = truth
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid frame-stack input")
