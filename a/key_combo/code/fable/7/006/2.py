from __future__ import annotations

import random

import numpy as np


Coord = tuple[int, int]
BBox = tuple[int, int, int, int]  # (r0, c0, r1, c1), inclusive

MAX_ATTEMPTS = 500
PALETTE = tuple(range(10))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[Coord] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(comp, dtype=int))
    return components


def _find_legend(grid: np.ndarray, bg: int) -> tuple[int, int, int, tuple[int, ...]]:
    h, _w = map(int, grid.shape)
    candidates: list[tuple[int, int, int, tuple[int, ...]]] = []
    for row in (0, h - 1):
        cols = np.flatnonzero(grid[row] != int(bg))
        if len(cols) == 0:
            continue
        assert np.all(cols == np.arange(int(cols[0]), int(cols[-1]) + 1)), "legend must be one contiguous edge run"
        c0 = int(cols[0])
        c1 = int(cols[-1])
        seq = tuple(int(v) for v in grid[row, c0 : c1 + 1])
        assert len(seq) >= 3, "legend must have length >= 3"
        candidates.append((int(row), c0, c1, seq))
    assert len(candidates) == 1, "expected exactly one top- or bottom-edge legend"
    return candidates[0]


def _extract_rectangles(
    grid: np.ndarray,
    bg: int,
    legend_row: int,
    legend_c0: int,
    legend_c1: int,
) -> list[dict]:
    mask = grid != int(bg)
    mask[int(legend_row), int(legend_c0) : int(legend_c1) + 1] = False
    components = _components_mask(mask)
    assert components, "expected at least one solid rectangle"

    rectangles: list[dict] = []
    for coords in components:
        bbox = _bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        colors = np.unique(patch)
        colors = colors[colors != int(bg)]
        assert len(colors) == 1, "each object must be monochromatic"
        color = int(colors[0])
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert int(coords.shape[0]) == h * w, "each object must be a solid rectangle"
        assert np.all(patch == int(color)), "rectangle interior must be filled"
        rectangles.append({"bbox": bbox, "color": color})

    rect_colors = {int(rect["color"]) for rect in rectangles}
    assert len(rect_colors) == 1, "all rectangles must share the same anchor color"
    return rectangles


def _oriented_legend(seq_lr: tuple[int, ...], rect_color: int) -> tuple[int, ...]:
    count = sum(int(v) == int(rect_color) for v in seq_lr)
    assert count == 1, "rectangle color must appear exactly once in the legend"
    if int(seq_lr[0]) == int(rect_color):
        return seq_lr
    assert int(seq_lr[-1]) == int(rect_color), "rectangle color must anchor one legend endpoint"
    return tuple(reversed(seq_lr))


def _depth_grid(bbox: BBox) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    rows = np.arange(r0, r1 + 1)[:, None]
    cols = np.arange(c0, c1 + 1)[None, :]
    return np.minimum(
        np.minimum(rows - int(r0), int(r1) - rows),
        np.minimum(cols - int(c0), int(c1) - cols),
    )


def _periodic_index(depth: np.ndarray, k: int) -> np.ndarray:
    return depth % int(k)


def _clamp_index(depth: np.ndarray, k: int) -> np.ndarray:
    return np.minimum(depth, int(k) - 1)


def _bounce_index(depth: np.ndarray, k: int) -> np.ndarray:
    period = 2 * (int(k) - 1)
    mod = depth % period
    return np.where(mod < int(k), mod, period - mod)


def _recolor_with_mode(
    grid: np.ndarray,
    rectangles: list[dict],
    oriented_seq: tuple[int, ...],
    *,
    mode: str,
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    seq = np.asarray(oriented_seq, dtype=int)
    k = int(len(oriented_seq))

    for rect in rectangles:
        bbox = rect["bbox"]
        r0, c0, r1, c1 = bbox
        depth = _depth_grid(bbox)
        if mode == "periodic":
            idx = _periodic_index(depth, k)
        elif mode == "clamp":
            idx = _clamp_index(depth, k)
        elif mode == "bounce":
            idx = _bounce_index(depth, k)
        else:
            raise ValueError(f"unknown recolor mode: {mode}")
        out[r0 : r1 + 1, c0 : c1 + 1] = seq[idx]

    return out


def _parse_puzzle(grid: np.ndarray) -> tuple[int, tuple[int, ...], list[dict]]:
    bg = _get_bg(grid)
    legend_row, legend_c0, legend_c1, seq_lr = _find_legend(grid, bg)
    rectangles = _extract_rectangles(grid, bg, legend_row, legend_c0, legend_c1)
    rect_color = int(rectangles[0]["color"])
    oriented_seq = _oriented_legend(seq_lr, rect_color)
    return bg, oriented_seq, rectangles


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Recolor each solid rectangle by indexing an edge legend with rectangular layer depth.

    Input structure:
    - The background is dominant, and exactly one contiguous horizontal legend sits on the top or bottom edge.
    - Away from the legend there are one or more solid axis-aligned rectangles, all in the same color.
    - That rectangle color appears exactly once inside the legend and is located at one endpoint of the legend.

    Rule:
    - Determine the reading direction from the endpoint containing the rectangle color.
    - For every rectangle pixel, compute its depth from the rectangle boundary, where the outer border has depth 0.
    - Recolor depth d with the d mod k entry of the directed legend, and keep the legend plus the background unchanged.

    Key insight:
    - The visible alternation comes from depth-based table lookup, not from a fixed two-color toggle, and the lookup direction is anchored by the rectangle color endpoint.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"
    assert max(grid.shape) <= 30, "ARC grids must be at most 30x30"

    _bg, oriented_seq, rectangles = _parse_puzzle(grid)
    return _recolor_with_mode(grid, rectangles, oriented_seq, mode="periodic")


def _depth_max(bbox: BBox) -> int:
    r0, c0, r1, c1 = bbox
    return (min(int(r1 - r0 + 1), int(c1 - c0 + 1)) - 1) // 2


def _rect_gap_ok(candidate: BBox, placed: list[BBox]) -> bool:
    r0, c0, r1, c1 = candidate
    for pr0, pc0, pr1, pc1 in placed:
        separated = (
            int(r1) + 1 < int(pr0)
            or int(pr1) + 1 < int(r0)
            or int(c1) + 1 < int(pc0)
            or int(pc1) + 1 < int(c0)
        )
        if not separated:
            return False
    return True


def _sample_rect_positions(
    height: int,
    width: int,
    dims: list[tuple[int, int]],
    *,
    top_legend: bool,
) -> list[BBox] | None:
    placed: list[BBox] = []
    row_min = 2 if top_legend else 1
    row_max = (height - 2) if top_legend else (height - 3)

    for rect_h, rect_w in dims:
        candidates: list[BBox] = []
        for r0 in range(int(row_min), int(row_max) - int(rect_h) + 2):
            for c0 in range(1, int(width) - int(rect_w)):
                bbox = (int(r0), int(c0), int(r0 + rect_h - 1), int(c0 + rect_w - 1))
                if _rect_gap_ok(bbox, placed):
                    candidates.append(bbox)
        if not candidates:
            return None
        bbox = random.choice(candidates)
        placed.append(bbox)
    return placed


def _build_legend(bg: int, rect_color: int, k: int, anchor_left: bool) -> tuple[int, ...]:
    others = [color for color in PALETTE if color not in (int(bg), int(rect_color))]
    inner = [random.choice(others) for _ in range(int(k) - 1)]
    if len(set(inner)) == len(inner) and random.random() < 0.7:
        dup_src = random.randrange(len(inner))
        dup_dst = random.randrange(len(inner))
        while dup_dst == dup_src:
            dup_dst = random.randrange(len(inner))
        inner[dup_dst] = inner[dup_src]
    directed = (int(rect_color), *map(int, inner))
    return directed if anchor_left else tuple(reversed(directed))


def _legend_interval(legend_start: int, k: int) -> tuple[int, int]:
    return int(legend_start), int(legend_start + k - 1)


def _wrong_outputs(
    grid: np.ndarray,
    rectangles: list[dict],
    oriented_seq: tuple[int, ...],
) -> list[np.ndarray]:
    return [
        _recolor_with_mode(grid, rectangles, tuple(reversed(oriented_seq)), mode="periodic"),
        _recolor_with_mode(grid, rectangles, oriented_seq, mode="clamp"),
        _recolor_with_mode(grid, rectangles, oriented_seq, mode="bounce"),
    ]


def _validate_sample(grid: np.ndarray, expected: np.ndarray) -> bool:
    try:
        bg, oriented_seq, rectangles = _parse_puzzle(grid)
    except AssertionError:
        return False

    if int(bg) not in PALETTE:
        return False
    if max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if not rectangles:
        return False
    if max(_depth_max(rect["bbox"]) for rect in rectangles) < len(oriented_seq):
        return False
    if np.array_equal(grid, expected):
        return False

    for wrong in _wrong_outputs(grid, rectangles, oriented_seq):
        if np.array_equal(wrong, expected):
            return False

    return np.array_equal(main(grid), expected)


def _gen_attempt() -> np.ndarray | None:
    k = random.randint(3, 6)
    rect_count = random.choice([1, 2, 2, 3, 3, 4])
    top_legend = bool(random.getrandbits(1))
    anchor_left = bool(random.getrandbits(1))

    bg, rect_color = random.sample(PALETTE, 2)
    legend = _build_legend(int(bg), int(rect_color), k, anchor_left)
    directed = legend if int(legend[0]) == int(rect_color) else tuple(reversed(legend))

    big_min_side = 2 * k + 1
    height = random.randint(max(14, big_min_side + 4), 30)
    width = random.randint(max(14, big_min_side + 4), 30)
    if width <= k + 2:
        return None

    legend_row = 0 if top_legend else height - 1
    legend_start = random.randint(1, width - k - 1)
    legend_c0, legend_c1 = _legend_interval(legend_start, k)

    dims = []
    big_h = random.randint(big_min_side, min(big_min_side + 4, height - 4))
    big_w = random.randint(big_min_side, min(big_min_side + 6, width - 2))
    dims.append((int(big_h), int(big_w)))
    for _ in range(rect_count - 1):
        rect_h = random.randint(3, min(10, height - 3))
        rect_w = random.randint(3, min(10, width - 2))
        dims.append((int(rect_h), int(rect_w)))
    dims.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    rects = _sample_rect_positions(height, width, dims, top_legend=top_legend)
    if rects is None:
        return None

    area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in rects)
    total = height * width
    if area >= (total - k) / 2:
        return None

    grid = np.full((height, width), int(bg), dtype=int)
    grid[legend_row, legend_c0 : legend_c1 + 1] = np.asarray(legend, dtype=int)
    for r0, c0, r1, c1 in rects:
        grid[r0 : r1 + 1, c0 : c1 + 1] = int(rect_color)

    rectangles = [{"bbox": bbox, "color": int(rect_color)} for bbox in rects]
    expected = _recolor_with_mode(grid, rectangles, directed, mode="periodic")
    return grid if _validate_sample(grid, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/7/006/2")
