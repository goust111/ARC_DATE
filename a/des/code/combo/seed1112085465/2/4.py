from __future__ import annotations

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_MAX_GENERATION_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_area(bbox: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = bbox
    return int(r1 - r0 + 1) * int(c1 - c0 + 1)


def _on_frame(bbox: tuple[int, int, int, int], r: int, c: int) -> bool:
    r0, c0, r1, c1 = bbox
    if not (r0 <= r <= r1 and c0 <= c <= c1):
        return False
    return r == r0 or r == r1 or c == c0 or c == c1


def _bbox_from_center(center_r2: int, center_c2: int, height: int, width: int) -> tuple[int, int, int, int]:
    r0 = (int(center_r2) - int(height) + 1) // 2
    c0 = (int(center_c2) - int(width) + 1) // 2
    r1 = r0 + int(height) - 1
    c1 = c0 + int(width) - 1
    return int(r0), int(c0), int(r1), int(c1)


def _draw_frame(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    out[r0, c0 : c1 + 1] = int(color)
    out[r1, c0 : c1 + 1] = int(color)
    out[r0 : r1 + 1, c0] = int(color)
    out[r0 : r1 + 1, c1] = int(color)


def _recover_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == int(bg):
            continue
        coords = np.argwhere(grid == int(color))
        assert coords.size > 0, "foreground colors must be present"
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 >= 2 and c1 - c0 >= 2, "frames need a non-empty interior"

        rel_r = coords[:, 0]
        rel_c = coords[:, 1]
        on_border = (rel_r == r0) | (rel_r == r1) | (rel_c == c0) | (rel_c == c1)
        assert bool(np.all(on_border)), "visible pixels must lie on a rectangular frame perimeter"

        frames.append(
            {
                "color": int(color),
                "bbox": bbox,
                "area": _bbox_area(bbox),
            }
        )

    assert 3 <= len(frames) <= 5, "expected 3 to 5 colored frames"
    return frames


def _infer_layer_order(grid: np.ndarray, frames: list[dict]) -> list[int]:
    colors = [int(frame["color"]) for frame in frames]
    edges: dict[int, set[int]] = {color: set() for color in colors}

    height, width = grid.shape
    for r in range(int(height)):
        for c in range(int(width)):
            covering = [int(frame["color"]) for frame in frames if _on_frame(frame["bbox"], int(r), int(c))]
            if len(covering) < 2:
                continue

            visible = int(grid[int(r), int(c)])
            assert visible in covering, "an overlap cell must show one of its covering frame colors"
            for lower in covering:
                if int(lower) != int(visible):
                    edges[int(lower)].add(int(visible))

    indegree = {color: 0 for color in colors}
    for uppers in edges.values():
        for upper in uppers:
            indegree[int(upper)] += 1

    order: list[int] = []
    ready = sorted(color for color in colors if indegree[int(color)] == 0)
    while ready:
        assert len(ready) == 1, "stacking order must be uniquely recoverable from occlusion"
        color = int(ready.pop(0))
        order.append(int(color))
        for upper in sorted(edges[int(color)]):
            indegree[int(upper)] -= 1
            if indegree[int(upper)] == 0:
                ready.append(int(upper))
        ready.sort()

    assert len(order) == len(colors), "frame-overlap graph must be acyclic and complete enough to order"
    return order


def _output_center(frames: list[dict]) -> tuple[int, int]:
    min_area = min(int(frame["area"]) for frame in frames)
    smallest = [frame for frame in frames if int(frame["area"]) == int(min_area)]
    assert len(smallest) == 1, "the smallest frame area must be unique"
    r0, c0, r1, c1 = smallest[0]["bbox"]
    center_r2 = int(r0 + r1)
    center_c2 = int(c0 + c1)
    assert center_r2 % 2 == center_c2 % 2, "the smallest frame center must admit centered squares"
    return center_r2, center_c2


def _render_output(
    shape: tuple[int, int],
    bg: int,
    bottom_to_top: list[int],
    center_r2: int,
    center_c2: int,
) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    side = 3 if int(center_r2) % 2 == 0 else 4
    side += 4 * (len(bottom_to_top) - 1)

    for color in bottom_to_top:
        bbox = _bbox_from_center(center_r2, center_c2, side, side)
        r0, c0, r1, c1 = bbox
        assert 0 <= r0 and 0 <= c0 and r1 < shape[0] and c1 < shape[1], "output frame must fit"
        _draw_frame(out, bbox, int(color))
        side -= 4

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild overlapping rectangular frames as reversed-layer concentric square frames.

    Input structure:
    - Background is black.
    - There are 3-5 unique colors, each a partially occluded 1px rectangular frame.
    - Visible colors at overlaps belong to the upper frame.

    Rule:
    - Recover each full rectangular frame from the bbox of its visible pixels.
    - Infer the unique bottom-to-top order from overlap visibility.
    - Use the smallest recovered frame's center as the common center.
    - Clear the grid and redraw concentric square frames with 1px gaps.
    - Map bottom layer to the outermost square and top layer to the innermost.

    Key insight:
    - Occlusion determines color order, while the output geometry is a separate
      centered square nesting.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    frames = _recover_frames(g, bg)
    bottom_to_top = _infer_layer_order(g, frames)
    center_r2, center_c2 = _output_center(frames)
    return _render_output(g.shape, int(bg), bottom_to_top, center_r2, center_c2)


def _sample_dims(count: int, even_sides: bool) -> list[tuple[int, int]] | None:
    min_side = 4 if even_sides else 3
    outer_square = int(min_side + 4 * (count - 1))
    max_side = max(int(outer_square + 4), int(min_side + 2 * (count + 3)))
    choices = list(range(int(min_side), int(max_side) + 1, 2))
    if len(choices) < count:
        return None

    heights = sorted(int(v) for v in _RNG.choice(np.array(choices, dtype=int), size=count, replace=False))
    widths = sorted((int(v) for v in _RNG.choice(np.array(choices, dtype=int), size=count, replace=False)), reverse=True)
    dims = [(int(h), int(w)) for h, w in zip(heights, widths)]

    if int(_RNG.integers(0, 2)) == 1:
        dims = [(int(w), int(h)) for h, w in dims]

    areas = [int(h * w) for h, w in dims]
    if len(set(areas)) != count:
        return None
    return dims


def _hidden_frames(
    dims: list[tuple[int, int]],
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> tuple[tuple[int, int], tuple[int, int], list[dict]]:
    heights = [int(h) for h, _w in dims]
    widths = [int(w) for _h, w in dims]
    count = len(dims)
    min_side = 4 if heights[0] % 2 == 0 else 3
    outer_square = int(min_side + 4 * (count - 1))

    required_h = max(max(heights), outer_square)
    required_w = max(max(widths), outer_square)
    shape = (
        int(required_h + top_margin + bottom_margin),
        int(required_w + left_margin + right_margin),
    )
    center_r2 = int(2 * top_margin + required_h - 1)
    center_c2 = int(2 * left_margin + required_w - 1)

    frames: list[dict] = []
    for height, width in dims:
        bbox = _bbox_from_center(center_r2, center_c2, int(height), int(width))
        frames.append(
            {
                "bbox": bbox,
                "area": int(height * width),
            }
        )

    return shape, (center_r2, center_c2), frames


def _render_input(shape: tuple[int, int], frame_specs: list[dict], bottom_to_top: list[int]) -> np.ndarray:
    grid = np.full(shape, _BG, dtype=int)
    by_color = {int(frame["color"]): frame for frame in frame_specs}
    for color in bottom_to_top:
        _draw_frame(grid, by_color[int(color)]["bbox"], int(color))
    return grid


def _build_expected(shape: tuple[int, int], bottom_to_top: list[int], center_r2: int, center_c2: int) -> np.ndarray:
    return _render_output(shape, _BG, bottom_to_top, center_r2, center_c2)


def _gen_attempt() -> np.ndarray | None:
    count = int(_RNG.integers(3, 6))
    even_sides = bool(int(_RNG.integers(0, 2)))

    dims = _sample_dims(count, even_sides)
    if dims is None:
        return None

    top_margin = int(_RNG.integers(1, 6))
    bottom_margin = int(_RNG.integers(1, 6))
    left_margin = int(_RNG.integers(1, 6))
    right_margin = int(_RNG.integers(1, 6))

    shape, center2, frames = _hidden_frames(dims, top_margin, bottom_margin, left_margin, right_margin)
    if shape[0] > 30 or shape[1] > 30:
        return None

    palette = [int(v) for v in _RNG.choice(np.arange(1, 10, dtype=int), size=count, replace=False)]
    perm = [int(v) for v in _RNG.permutation(count)]
    layer_ranks = [palette[idx] for idx in perm]

    frame_specs: list[dict] = []
    for color, frame in zip(palette, frames):
        item = dict(frame)
        item["color"] = int(color)
        frame_specs.append(item)

    inp = _render_input(shape, frame_specs, layer_ranks)
    expected = _build_expected(shape, layer_ranks, center2[0], center2[1])

    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
