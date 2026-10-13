from __future__ import annotations

import numpy as np


_DIRECTION_ORDER = {"top": 0, "right": 1, "bottom": 2, "left": 3}
_DIRECTIONS = tuple(_DIRECTION_ORDER.keys())

_GEN_BG = 0
_GEN_COLORS = tuple(range(1, 10))
_GEN_MAX_ATTEMPTS = 500


def _render_patch(height: int, width: int, color: int, gap_cell: tuple[int, int], bg: int) -> np.ndarray:
    patch = np.full((height, width), int(bg), dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    patch[int(gap_cell[0]), int(gap_cell[1])] = int(bg)
    return patch


def _extract_frame_item(grid: np.ndarray, color: int) -> dict | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None

    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    rel = coords - np.array([r0, c0], dtype=int)
    on_border = (
        (rel[:, 0] == 0)
        | (rel[:, 0] == height - 1)
        | (rel[:, 1] == 0)
        | (rel[:, 1] == width - 1)
    )
    if not bool(np.all(on_border)):
        return None

    if int(coords.shape[0]) != 2 * height + 2 * width - 5:
        return None

    if not (
        int(grid[r0, c0]) == int(color)
        and int(grid[r0, c1]) == int(color)
        and int(grid[r1, c0]) == int(color)
        and int(grid[r1, c1]) == int(color)
    ):
        return None

    side_missing = {
        "top": int(np.sum(grid[r0, c0 + 1 : c1] != int(color))),
        "right": int(np.sum(grid[r0 + 1 : r1, c1] != int(color))),
        "bottom": int(np.sum(grid[r1, c0 + 1 : c1] != int(color))),
        "left": int(np.sum(grid[r0 + 1 : r1, c0] != int(color))),
    }
    gap_sides = [side for side, count in side_missing.items() if count == 1]
    if len(gap_sides) != 1 or sum(side_missing.values()) != 1:
        return None

    return {
        "color": int(color),
        "bbox": (r0, c0, r1, c1),
        "height": height,
        "width": width,
        "area": height * width,
        "coords": rel,
        "gap_side": gap_sides[0],
    }


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    colors = [int(color) for color in np.unique(grid)]
    frames = []
    frame_colors = set()
    for color in colors:
        item = _extract_frame_item(grid, int(color))
        if item is None:
            continue
        frames.append(item)
        frame_colors.add(int(color))

    assert 2 <= len(frames) <= 4, "expected 2 to 4 colored frames"

    bg_candidates = [int(color) for color in colors if int(color) not in frame_colors]
    assert len(bg_candidates) == 1, "expected exactly one background color"
    bg = int(bg_candidates[0])

    frames.sort(key=lambda item: int(item["area"]), reverse=True)
    for outer, inner in zip(frames, frames[1:]):
        or0, oc0, or1, oc1 = outer["bbox"]
        ir0, ic0, ir1, ic1 = inner["bbox"]
        assert or0 < ir0 and oc0 < ic0 and or1 > ir1 and oc1 > ic1, "frames must be nested"

    gap_sides = [item["gap_side"] for item in frames]
    assert len(gap_sides) == len(set(gap_sides)), "gap directions must be unique"
    return bg, frames


def _frame_patch_from_item(item: dict, bg: int) -> np.ndarray:
    patch = np.full((int(item["height"]), int(item["width"])), int(bg), dtype=int)
    rel = np.asarray(item["coords"], dtype=int)
    patch[rel[:, 0], rel[:, 1]] = int(item["color"])
    return patch


def _assemble_output(items: list[dict], bg: int) -> np.ndarray:
    ordered = sorted(items, key=lambda item: _DIRECTION_ORDER[item["gap_side"]])
    patches = [_frame_patch_from_item(item, bg) for item in ordered]

    out_height = max(int(patch.shape[0]) for patch in patches)
    out_width = sum(int(patch.shape[1]) for patch in patches) + len(patches) - 1
    out = np.full((out_height, out_width), int(bg), dtype=int)

    col = 0
    for patch in patches:
        height, width = map(int, patch.shape)
        out[:height, col : col + width] = patch
        col += width + 1
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Peel nested one-gap rectangular frames apart and reorder them by gap side.

    Input structure:
    - The grid contains 2 to 4 axis-aligned rectangular frames on a black background.
    - Each frame has a distinct color, is one pixel thick, and is nested inside a larger frame.
    - Every frame is a rectangular border with exactly one single-cell gap on one non-corner side.

    Rule:
    - Identify each colored frame from its tight bounding box and isolate that frame on background cells.
    - Determine which side contains the unique missing border cell for each frame.
    - Sort the isolated frame crops by gap direction in clockwise order top, right, bottom, left.
    - Place the cropped frames left-to-right with one background column between neighbors.

    Key insight:
    - The colors only separate layers; the decisive feature is the geometric side of each frame's gap, so the output order ignores nesting depth.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, frames = _extract_frames(g)
    return _assemble_output(frames, bg)


def _sample_inset(rng: np.random.Generator, n_layers: int) -> int:
    if n_layers == 2:
        return int(rng.integers(1, 4))
    return int(rng.choice([1, 1, 2]))


def _sample_bboxes(rng: np.random.Generator, n_layers: int) -> tuple[tuple[int, int], list[tuple[int, int, int, int]]] | None:
    inner_height_max = {2: 12, 3: 9, 4: 7}[n_layers]
    inner_width_max = {2: 12, 3: 6, 4: 3}[n_layers]

    inner_height = int(rng.integers(3, inner_height_max + 1))
    inner_width = int(rng.integers(3, inner_width_max + 1))

    boxes = [(0, 0, inner_height - 1, inner_width - 1)]
    for _ in range(n_layers - 1):
        r0, c0, r1, c1 = boxes[-1]
        top = _sample_inset(rng, n_layers)
        bottom = _sample_inset(rng, n_layers)
        left = _sample_inset(rng, n_layers)
        right = _sample_inset(rng, n_layers)
        boxes.append((r0 - top, c0 - left, r1 + bottom, c1 + right))

    min_r = min(box[0] for box in boxes)
    min_c = min(box[1] for box in boxes)
    normalized = [(r0 - min_r, c0 - min_c, r1 - min_r, c1 - min_c) for r0, c0, r1, c1 in boxes]
    normalized.reverse()

    widths = [int(c1 - c0 + 1) for _r0, c0, _r1, c1 in normalized]
    heights = [int(r1 - r0 + 1) for r0, _c0, r1, _c1 in normalized]
    if sum(widths) + n_layers - 1 > 30 or max(heights) > 30:
        return None

    outer_height = heights[0]
    outer_width = widths[0]

    top_margin = int(rng.integers(0, min(4, 30 - outer_height + 1)))
    bottom_margin = int(rng.integers(0, min(4, 30 - outer_height - top_margin + 1)))
    left_margin = int(rng.integers(0, min(4, 30 - outer_width + 1)))
    right_margin = int(rng.integers(0, min(4, 30 - outer_width - left_margin + 1)))

    shifted = [
        (r0 + top_margin, c0 + left_margin, r1 + top_margin, c1 + left_margin)
        for r0, c0, r1, c1 in normalized
    ]
    canvas_shape = (outer_height + top_margin + bottom_margin, outer_width + left_margin + right_margin)
    return canvas_shape, shifted


def _make_gap_cell(
    rng: np.random.Generator,
    height: int,
    width: int,
    gap_side: str,
) -> tuple[int, int]:
    if gap_side == "top":
        return 0, int(rng.integers(1, width - 1))
    if gap_side == "right":
        return int(rng.integers(1, height - 1)), width - 1
    if gap_side == "bottom":
        return height - 1, int(rng.integers(1, width - 1))
    return int(rng.integers(1, height - 1)), 0


def _build_specs(
    rng: np.random.Generator,
    bboxes: list[tuple[int, int, int, int]],
    colors: np.ndarray,
    gap_sides: np.ndarray,
) -> list[dict]:
    specs = []
    for bbox, color, gap_side in zip(bboxes, colors, gap_sides):
        r0, c0, r1, c1 = map(int, bbox)
        height = int(r1 - r0 + 1)
        width = int(c1 - c0 + 1)
        specs.append(
            {
                "color": int(color),
                "bbox": (r0, c0, r1, c1),
                "height": height,
                "width": width,
                "gap_side": str(gap_side),
                "gap_cell": _make_gap_cell(rng, height, width, str(gap_side)),
            }
        )
    return specs


def _paint_input(specs: list[dict], shape: tuple[int, int], bg: int) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        color = int(spec["color"])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color

        gap_r = r0 + int(spec["gap_cell"][0])
        gap_c = c0 + int(spec["gap_cell"][1])
        grid[gap_r, gap_c] = int(bg)
    return grid


def _expected_output(specs: list[dict], bg: int) -> np.ndarray:
    items = []
    for spec in specs:
        patch = _render_patch(
            int(spec["height"]),
            int(spec["width"]),
            int(spec["color"]),
            tuple(spec["gap_cell"]),
            int(bg),
        )
        items.append(
            {
                "color": int(spec["color"]),
                "height": int(spec["height"]),
                "width": int(spec["width"]),
                "coords": np.argwhere(patch == int(spec["color"])),
                "gap_side": str(spec["gap_side"]),
            }
        )
    return _assemble_output(items, int(bg))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_layers = int(rng.integers(2, 5))
    sampled = _sample_bboxes(rng, n_layers)
    if sampled is None:
        return None
    shape, bboxes = sampled

    colors = rng.choice(_GEN_COLORS, size=n_layers, replace=False)
    gap_sides = rng.permutation(np.array(_DIRECTIONS, dtype=object))[:n_layers]
    specs = _build_specs(rng, bboxes, colors, gap_sides)

    inp = _paint_input(specs, shape, _GEN_BG)
    expected = _expected_output(specs, _GEN_BG)

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if actual.shape != expected.shape or not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame input")
