from __future__ import annotations

import numpy as np

from arc_utils import Color, find_components, paste


def _infer_background_and_objects(g: np.ndarray) -> tuple[int, list[dict]]:
    colors, counts = np.unique(g, return_counts=True)
    min_count = int(counts.min())
    marker_colors = [int(c) for c, n in zip(colors, counts) if int(n) == min_count]
    assert len(marker_colors) == 1
    marker = marker_colors[0]
    assert int(np.sum(g == marker)) == 2

    candidates: list[tuple[int, list[dict]]] = []
    for bg in map(int, colors):
        if bg == marker:
            continue
        objects = find_components(g, bg=bg, connectivity=4, monochromatic=False)
        if len(objects) != 2:
            continue
        ok = True
        for obj in objects:
            coords = obj["coords"]
            vals = g[coords[:, 0], coords[:, 1]]
            if int(np.sum(vals == marker)) != 1:
                ok = False
            if len(set(map(int, vals)) - {marker}) != 1:
                ok = False
        if ok:
            candidates.append((bg, objects))

    assert len(candidates) == 1
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the background, rotate the whole input 90 degrees counterclockwise, and stack two framed copies vertically.

    Input structure:
    - Blue is absent from the input.
    - One rare marker color appears in exactly two pixels.
    - The background is the unique non-marker color whose removal leaves exactly two 4-connected foreground objects, each with one marker pixel and one body color.
    - The whole grid, including background and foreground objects, is the sprite to transform.

    Rule:
    - Rotate the entire input grid 90 degrees counterclockwise.
    - Create a blue one-pixel frame around two vertical copies of the rotated grid.
    - Use the input background color for the two endpoints of every horizontal frame/separator row.

    Key insight: The output is not a crop of the foreground objects; it is two copies of the complete rotated input, separated and enclosed by a newly added blue border.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg, objects = _infer_background_and_objects(g)
    assert bg != Color.BLUE
    assert Color.BLUE not in set(map(int, g.ravel()))

    copies = len(objects)

    tile = np.rot90(g, 1)
    tile_h, tile_w = map(int, tile.shape)

    out_h = copies * tile_h + copies + 1
    out_w = tile_w + 2
    out = np.full((out_h, out_w), Color.BLUE, dtype=int)

    for i in range(copies):
        row = i * (tile_h + 1)
        out[row, 0] = bg
        out[row, -1] = bg
        paste(out, tile, row + 1, 1)

    out[-1, 0] = bg
    out[-1, -1] = bg
    return out

# --------------- generator (att2, o5x) ---------------
# CFG: ranges use numpy's high-exclusive convention where applicable.
CANVAS_SHAPE_RANGES = ((5, 10), (5, 11))
OBJECT_SHAPE_RANGES = ((2, 5), (2, 5))
MIN_SEPARATION_GAP = 1
MAX_OUTPUT_SIZE = 30
INPUT_COLORS = [0, 2, 3, 4, 5, 6, 7, 8, 9]
PALETTE_SIZE = 3
FRAME_COLOR = Color.BLUE
OUTPUT_TILE_COPIES = 2
FRAME_THICKNESS = 1
MAX_GENERATION_ATTEMPTS = 300

def _edge_cells(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    ]

def _expected_output(inp: np.ndarray, bg: int) -> np.ndarray:
    tile = np.rot90(inp, 1)
    tile_h, tile_w = tile.shape
    out_h = OUTPUT_TILE_COPIES * tile_h + (OUTPUT_TILE_COPIES + 1) * FRAME_THICKNESS
    out_w = tile_w + 2 * FRAME_THICKNESS
    out = np.full((out_h, out_w), FRAME_COLOR, dtype=int)
    out[0, 0] = out[0, -1] = bg
    out[tile_h + FRAME_THICKNESS, 0] = out[tile_h + FRAME_THICKNESS, -1] = bg
    out[-1, 0] = out[-1, -1] = bg
    out[FRAME_THICKNESS : FRAME_THICKNESS + tile_h, FRAME_THICKNESS : FRAME_THICKNESS + tile_w] = tile
    second_top = tile_h + 2 * FRAME_THICKNESS
    out[second_top : second_top + tile_h, FRAME_THICKNESS : FRAME_THICKNESS + tile_w] = tile
    return out

def _sample_object_dims(canvas_h: int, canvas_w: int) -> tuple[int, int] | None:
    feasible_dims = [
        (oh, ow)
        for oh in range(*OBJECT_SHAPE_RANGES[0])
        for ow in range(*OBJECT_SHAPE_RANGES[1])
        if oh <= canvas_h and ow <= canvas_w and (
            canvas_h - 2 * oh >= MIN_SEPARATION_GAP
            or canvas_w - 2 * ow >= MIN_SEPARATION_GAP
        )
    ]
    if not feasible_dims:
        return None
    return feasible_dims[int(np.random.randint(len(feasible_dims)))]


def _gen_attempt() -> np.ndarray | None:
    H = int(np.random.randint(*CANVAS_SHAPE_RANGES[0]))
    W = int(np.random.randint(*CANVAS_SHAPE_RANGES[1]))

    dims = _sample_object_dims(H, W)
    if dims is None:
        return None
    oh, ow = dims

    bg, body, marker = map(int, np.random.choice(INPUT_COLORS, size=PALETTE_SIZE, replace=False))
    inp = np.full((H, W), bg, dtype=int)
    inp[:oh, :ow] = body
    inp[H - oh : H, W - ow : W] = body
    mr, mc = _edge_cells(oh, ow)[int(np.random.randint(2 * oh + 2 * ow - 4))]
    inp[mr, mc] = marker
    inp[H - oh + mr, W - ow + mc] = marker

    expected = _expected_output(inp, bg)
    if expected.shape[0] > MAX_OUTPUT_SIZE or expected.shape[1] > MAX_OUTPUT_SIZE:
        return None

    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate valid input")
