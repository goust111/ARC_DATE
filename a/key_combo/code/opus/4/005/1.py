from __future__ import annotations

import random
import sys

import numpy as np

_ARC_UTILS_DIR = "/home/lixinyu/snap/new/solvers"
if _ARC_UTILS_DIR not in sys.path:
    sys.path.append(_ARC_UTILS_DIR)

from arc_utils import Color, bbox_coords, components_mask, crop_bbox, get_bg, paste


_BG = Color.BLACK
_BLOCK_COLOR = Color.YELLOW
_MARKER_COLORS = [Color.BLUE, Color.RED, Color.GREEN, Color.GREY, Color.PINK, Color.ORANGE, Color.TEAL, Color.MAROON]
_VALID_DIMS = [(2, 2), (2, 3), (3, 2), (2, 4), (4, 2)]
_GRID_H_RANGE = (15, 22)
_GRID_W_RANGE = (15, 22)
_BLOCK_COUNT_RANGE = (3, 4)
_DISTANCES = (2, 3)
_GEN_MAX_ATTEMPTS = 400


def _expanded_box(block: dict) -> tuple[int, int, int, int]:
    r0 = int(block["r0"])
    c0 = int(block["c0"])
    h = int(block["h"])
    w = int(block["w"])
    d = int(block["d"])
    if block["axis"] == "h":
        return r0, c0 - d, r0 + h - 1, c0 + w + d - 1
    return r0 - d, c0, r0 + h + d - 1, c0 + w - 1


def _make_patch(block: dict) -> np.ndarray:
    patch = np.full((int(block["h"]), int(block["w"])), _BLOCK_COLOR, dtype=int)
    d = int(block["d"])
    color = int(block["color"])

    if block["axis"] == "h":
        row = random.randrange(int(block["h"]))
        start = random.randrange(int(block["w"]) - d + 1)
        patch[row, start : start + d] = color
    else:
        col = random.randrange(int(block["w"]))
        start = random.randrange(int(block["h"]) - d + 1)
        patch[start : start + d, col] = color
    return patch


def _parse_blocks(grid: np.ndarray) -> list[dict]:
    bg = int(get_bg(grid))
    comps = components_mask(grid != bg, connectivity=4)
    assert comps, "expected at least one foreground rectangle"

    blocks: list[dict] = []
    for coords in comps:
        r0, c0, r1, c1 = bbox_coords(coords)
        region = crop_bbox(grid, (r0, c0, r1, c1))
        assert int(coords.shape[0]) == int(region.size), "each object must be a solid rectangle"

        colors = set(map(int, np.unique(region)))
        assert _BLOCK_COLOR in colors, "each object must contain yellow cells"
        marker_colors = colors - {_BLOCK_COLOR}
        assert len(marker_colors) == 1, "each rectangle must have exactly one control color"
        marker_color = marker_colors.pop()

        markers = np.argwhere(region == marker_color)
        d = int(markers.shape[0])
        assert d >= 1, "expected at least one control pixel"
        same_row = len(set(map(int, markers[:, 0]))) == 1
        same_col = len(set(map(int, markers[:, 1]))) == 1
        assert same_row ^ same_col, "control pixels must encode a unique axis"

        blocks.append(
            {
                "r0": int(r0),
                "c0": int(c0),
                "h": int(region.shape[0]),
                "w": int(region.shape[1]),
                "d": d,
                "axis": "h" if same_row else "v",
                "color": int(marker_color),
            }
        )
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand each yellow marker-bearing rectangle along the axis encoded by its control-pixel line.

    Input structure:
    - The background is black and the foreground consists of several separated solid rectangles.
    - Every rectangle contains yellow cells plus 1 color of embedded non-yellow control pixels.
    - The control pixels of a rectangle form a straight horizontal or vertical line inside that rectangle.

    Rule:
    - For each rectangle, count its control pixels; that count is the expansion distance d and their color is the paint color.
    - If the control pixels lie in one row, expand the rectangle's bbox d columns left and right; if they lie in one column, expand d rows up and down.
    - Paint the expanded bbox with the control color, then restore the original rectangle so yellow cells and control pixels remain visible.

    Key insight:
    - The same embedded marker set carries 2 independent instructions at once: marker count gives the distance, and marker orientation gives the single expansion axis.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    occupied = np.zeros(g.shape, dtype=bool)
    for block in _parse_blocks(g):
        er0, ec0, er1, ec1 = _expanded_box(block)
        assert 0 <= er0 <= er1 < g.shape[0], "expanded block must stay within the grid"
        assert 0 <= ec0 <= ec1 < g.shape[1], "expanded block must stay within the grid"
        assert not bool(np.any(occupied[er0 : er1 + 1, ec0 : ec1 + 1])), "expanded regions must be disjoint"

        out[er0 : er1 + 1, ec0 : ec1 + 1] = int(block["color"])
        occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True

    fg = g != int(get_bg(g))
    out[fg] = g[fg]
    return out


def _valid_dims(axis: str, d: int) -> list[tuple[int, int]]:
    if axis == "h":
        return [(h, w) for h, w in _VALID_DIMS if w >= d]
    return [(h, w) for h, w in _VALID_DIMS if h >= d]


def _seed_specs(n_blocks: int) -> list[dict]:
    shared_d = random.choice(_DISTANCES)
    other_d = _DISTANCES[0] if shared_d == _DISTANCES[1] else _DISTANCES[1]
    dominant_axis = random.choice(("h", "v"))
    other_axis = "v" if dominant_axis == "h" else "h"

    specs = [
        {"axis": dominant_axis, "d": shared_d},
        {"axis": other_axis, "d": shared_d},
        {"axis": dominant_axis, "d": other_d},
    ]
    while len(specs) < n_blocks:
        specs.append({"axis": random.choice(("h", "v")), "d": random.choice(_DISTANCES)})

    random.shuffle(specs)
    return specs


def _can_place(used: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(used.shape[0], r0 + h + 1)
    cc1 = min(used.shape[1], c0 + w + 1)
    return not bool(np.any(used[rr0:rr1, cc0:cc1]))


def _place_expanded_boxes(shape: tuple[int, int], blocks: list[dict]) -> bool:
    used = np.zeros(shape, dtype=bool)
    order = sorted(range(len(blocks)), key=lambda i: blocks[i]["eh"] * blocks[i]["ew"], reverse=True)

    def rec(k: int) -> bool:
        if k == len(order):
            return True

        idx = order[k]
        block = blocks[idx]
        starts = [
            (r, c)
            for r in range(shape[0] - int(block["eh"]) + 1)
            for c in range(shape[1] - int(block["ew"]) + 1)
            if _can_place(used, r, c, int(block["eh"]), int(block["ew"]))
        ]
        random.shuffle(starts)

        for er0, ec0 in starts:
            used[er0 : er0 + int(block["eh"]), ec0 : ec0 + int(block["ew"])] = True
            block["er0"] = int(er0)
            block["ec0"] = int(ec0)
            if block["axis"] == "h":
                block["r0"] = int(er0)
                block["c0"] = int(ec0 + int(block["d"]))
            else:
                block["r0"] = int(er0 + int(block["d"]))
                block["c0"] = int(ec0)
            if rec(k + 1):
                return True
            used[er0 : er0 + int(block["eh"]), ec0 : ec0 + int(block["ew"])] = False
            for key in ("er0", "ec0", "r0", "c0"):
                block.pop(key, None)
        return False

    return rec(0)


def _render_output(input_grid: np.ndarray, blocks: list[dict]) -> np.ndarray:
    out = input_grid.copy()
    for block in blocks:
        er0, ec0, er1, ec1 = _expanded_box(block)
        out[er0 : er1 + 1, ec0 : ec1 + 1] = int(block["color"])
    for block in blocks:
        paste(out, block["patch"], int(block["r0"]), int(block["c0"]))
    return out


def _assert_generator_semantics(shape: tuple[int, int], blocks: list[dict], input_grid: np.ndarray, output_grid: np.ndarray) -> None:
    axes = [str(block["axis"]) for block in blocks]
    ds = [int(block["d"]) for block in blocks]

    assert set(axes) == {"h", "v"}, "generator must expose both axis branches"
    assert any(blocks[i]["axis"] != blocks[j]["axis"] and blocks[i]["d"] == blocks[j]["d"] for i in range(len(blocks)) for j in range(i + 1, len(blocks))), "need same-distance blocks with different axes"
    assert any(blocks[i]["axis"] == blocks[j]["axis"] and blocks[i]["d"] != blocks[j]["d"] for i in range(len(blocks)) for j in range(i + 1, len(blocks))), "need same-axis blocks with different distances"
    assert len(set(int(block["color"]) for block in blocks)) == len(blocks), "marker colors must distinguish rectangles"
    assert set(ds).issubset(set(_DISTANCES))

    occupied = np.zeros(shape, dtype=bool)
    for block in blocks:
        er0, ec0, er1, ec1 = _expanded_box(block)
        assert 0 <= er0 <= er1 < shape[0]
        assert 0 <= ec0 <= ec1 < shape[1]
        assert not bool(np.any(occupied[er0 : er1 + 1, ec0 : ec1 + 1])), "expanded boxes must be disjoint"
        occupied[er0 : er1 + 1, ec0 : ec1 + 1] = True

        markers = np.argwhere(block["patch"] == int(block["color"]))
        same_row = len(set(map(int, markers[:, 0]))) == 1
        same_col = len(set(map(int, markers[:, 1]))) == 1
        assert same_row ^ same_col
        assert int(markers.shape[0]) == int(block["d"])

    parsed = _parse_blocks(input_grid)
    assert len(parsed) == len(blocks), "input must contain the planned number of rectangles"
    assert np.array_equal(main(input_grid), output_grid), "solver must match the semantic renderer"


def _gen_attempt() -> np.ndarray | None:
    shape = (random.randint(*_GRID_H_RANGE), random.randint(*_GRID_W_RANGE))
    n_blocks = random.randint(*_BLOCK_COUNT_RANGE)
    colors = random.sample(_MARKER_COLORS, n_blocks)

    blocks = _seed_specs(n_blocks)
    for block, color in zip(blocks, colors):
        block["color"] = int(color)
        block["h"], block["w"] = random.choice(_valid_dims(str(block["axis"]), int(block["d"])))
        if block["axis"] == "h":
            block["eh"] = int(block["h"])
            block["ew"] = int(block["w"] + 2 * block["d"])
        else:
            block["eh"] = int(block["h"] + 2 * block["d"])
            block["ew"] = int(block["w"])

    if not _place_expanded_boxes(shape, blocks):
        return None

    input_grid = np.full(shape, _BG, dtype=int)
    for block in blocks:
        block["patch"] = _make_patch(block)
        paste(input_grid, block["patch"], int(block["r0"]), int(block["c0"]))

    output_grid = _render_output(input_grid, blocks)
    _assert_generator_semantics(shape, blocks, input_grid, output_grid)
    return input_grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
