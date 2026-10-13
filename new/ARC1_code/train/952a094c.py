import numpy as np

from arc_utils import find_components, get_bg


def _rectangular_frame_candidates(grid: np.ndarray, bg: int) -> list[dict]:
    candidates = []
    for comp in find_components(grid, bg=bg, connectivity=4, monochromatic=True):
        r0, c0, r1, c1 = comp["bbox"]
        if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
            continue

        border = np.zeros(grid.shape, dtype=bool)
        border[r0 : r1 + 1, c0] = True
        border[r0 : r1 + 1, c1] = True
        border[r0, c0 : c1 + 1] = True
        border[r1, c0 : c1 + 1] = True

        mask = np.zeros(grid.shape, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0], coords[:, 1]] = True
        if np.array_equal(mask, border):
            candidates.append(comp)
    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the four interior-corner markers to the opposite corners just outside the rectangular frame.

    Input structure
    - The grid has one dominant background color.
    - The foreground contains a unique one-pixel-thick monochrome rectangular frame.
    - The only other foreground pixels are four markers at the frame interior's four corners.
    Rule
    - Keep the frame unchanged and clear the four marker pixels to the background color.
    - Place each marker at the opposite corner of the frame's one-cell-expanded bounding box.
    Key insight
    - The frame defines both the source marker slots just inside it and the destination slots just outside it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    frames = _rectangular_frame_candidates(g, bg)
    assert len(frames) == 1, "expected exactly one rectangular frame"

    frame = frames[0]
    frame_color = int(frame["color"])
    r0, c0, r1, c1 = frame["bbox"]
    height, width = map(int, g.shape)
    assert 0 < r0 and 0 < c0 and r1 < height - 1 and c1 < width - 1

    src_dst = [
        ((r0 + 1, c0 + 1), (r1 + 1, c1 + 1)),
        ((r0 + 1, c1 - 1), (r1 + 1, c0 - 1)),
        ((r1 - 1, c0 + 1), (r0 - 1, c1 + 1)),
        ((r1 - 1, c1 - 1), (r0 - 1, c0 - 1)),
    ]
    assert len({src for src, _dst in src_dst}) == 4, "expected four distinct interior corner slots"

    expected_foreground = np.zeros(g.shape, dtype=bool)
    frame_coords = frame["coords"]
    expected_foreground[frame_coords[:, 0], frame_coords[:, 1]] = True
    for (sr, sc), (dr, dc) in src_dst:
        assert 0 <= sr < height and 0 <= sc < width
        assert 0 <= dr < height and 0 <= dc < width
        assert int(g[sr, sc]) not in {bg, frame_color}, "expected a marker at each interior corner"
        assert int(g[dr, dc]) == bg, "expected outside destination corner to be background"
        expected_foreground[sr, sc] = True
    assert np.array_equal(g != bg, expected_foreground), "unexpected extra foreground pixels"

    out = g.copy()
    for src, dst in src_dst:
        color = int(g[src])
        out[src] = bg
        out[dst] = color
    return out

# --------------- generator (o5x) ---------------

# Canvas, palette, frame sampling, marker count, and retry bounds.
GRID_SHAPE = (10, 10)
BG = 0
COLOR_IDS = np.arange(1, 10)
FRAME_SIZE_RANGE = (4, 7)
FRAME_EDGE_MARGIN = 1
MARKER_COUNT = 4
GENERATION_RETRIES = 300


def _gen_attempt() -> np.ndarray | None:
    height, width = GRID_SHAPE

    frame_h = int(np.random.randint(*FRAME_SIZE_RANGE))
    frame_w = int(np.random.randint(*FRAME_SIZE_RANGE))
    r0 = int(np.random.randint(FRAME_EDGE_MARGIN, height - frame_h - FRAME_EDGE_MARGIN + 1))
    c0 = int(np.random.randint(FRAME_EDGE_MARGIN, width - frame_w - FRAME_EDGE_MARGIN + 1))
    r1 = r0 + frame_h - 1
    c1 = c0 + frame_w - 1

    colors = np.random.permutation(COLOR_IDS)[: MARKER_COUNT + 1]
    frame_color = int(colors[0])
    marker_colors = [int(color) for color in colors[1:]]

    out = np.full(GRID_SHAPE, BG, dtype=int)
    out[r0 : r1 + 1, c0] = frame_color
    out[r0 : r1 + 1, c1] = frame_color
    out[r0, c0 : c1 + 1] = frame_color
    out[r1, c0 : c1 + 1] = frame_color

    src_dst = [
        ((r0 + 1, c0 + 1), (r1 + 1, c1 + 1)),
        ((r0 + 1, c1 - 1), (r1 + 1, c0 - 1)),
        ((r1 - 1, c0 + 1), (r0 - 1, c1 + 1)),
        ((r1 - 1, c1 - 1), (r0 - 1, c0 - 1)),
    ]

    for color, (src, dst) in zip(marker_colors, src_dst):
        out[dst] = color

    inp = out.copy()
    for src, dst in src_dst:
        inp[src] = int(out[dst])
        inp[dst] = BG

    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 952a094c input")
