import numpy as np

from arc_utils import Color, bbox_coords, components_mask, get_bg, paste
import random


def main(grid: np.ndarray) -> np.ndarray:
    """Paint a same-color halo around each yellow block, with halo thickness equal to its embedded marker count.

    Input structure
    - The background is the dominant color.
    - Each foreground object is a solid rectangular block made of yellow cells plus control pixels of exactly one other color.

    Rule
    - For each block, count its non-yellow control pixels.
    - Expand the block's bounding box by that count on all four sides and paint the expanded rectangle with the control color.
    - Restore all original foreground cells so the yellow block and its control pixels remain visible.

    Key insight
    - The repeated control color inside a block encodes both the output color and the padding distance for that block's surrounding rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = get_bg(g)
    foreground = g != bg
    components = components_mask(foreground, connectivity=4)
    assert components, "expected at least one foreground block"

    out = g.copy()
    painted = np.zeros(g.shape, dtype=bool)
    height, width = map(int, g.shape)

    for coords in components:
        r0, c0, r1, c1 = bbox_coords(coords)
        region = g[r0 : r1 + 1, c0 : c1 + 1]
        assert coords.shape[0] == region.size, "expected each foreground object to be a solid rectangle"

        colors = set(map(int, np.unique(region)))
        assert Color.YELLOW in colors, "expected yellow block cells"
        marker_colors = colors - {Color.YELLOW}
        assert len(marker_colors) == 1, "expected exactly one control color per block"
        marker_color = marker_colors.pop()

        pad = int(np.sum(region == marker_color))
        assert pad > 0, "expected at least one control pixel"
        assert int(np.sum(region == Color.YELLOW)) + pad == region.size

        er0, ec0 = r0 - pad, c0 - pad
        er1, ec1 = r1 + pad, c1 + pad
        assert 0 <= er0 <= er1 < height, "expanded block must stay in bounds"
        assert 0 <= ec0 <= ec1 < width, "expanded block must stay in bounds"
        assert not bool(np.any(painted[er0 : er1 + 1, ec0 : ec1 + 1])), "expanded blocks must not overlap"

        out[er0 : er1 + 1, ec0 : ec1 + 1] = marker_color
        painted[er0 : er1 + 1, ec0 : ec1 + 1] = True

    out[foreground] = g[foreground]
    return out

# --------------- generator (o5x) ---------------
# Canvas and block bounds.
GRID_SIDE_RANGE, BLOCK_COUNT_RANGE = (25, 29), (2, 3)
BLOCK_HEIGHT_RANGE, BLOCK_WIDTH_RANGE = (3, 8), (3, 10)
# Marker padding and block interior constraints.
PAD_RANGE, INTERIOR_MARGIN = (1, 5), 1
# Retry limit and color choices.
GENERATION_RETRIES = 300
MARKER_COLORS = [c for c in Color.NOT_BLACK if c != Color.YELLOW]

def _sample_block_specs():
    blocks = []
    for _ in range(random.randint(*BLOCK_COUNT_RANGE)):
        h = random.randint(*BLOCK_HEIGHT_RANGE)
        w = random.randint(*BLOCK_WIDTH_RANGE)
        interior_area = (h - 2 * INTERIOR_MARGIN) * (w - 2 * INTERIOR_MARGIN)
        pad = random.randint(PAD_RANGE[0], min(PAD_RANGE[1], interior_area))
        blocks.append({"h": h, "w": w, "pad": pad, "eh": h + 2 * pad, "ew": w + 2 * pad})
    return blocks

def _gen_place_expanded_boxes(side, blocks):
    """Assign non-overlapping expanded-box origins in-place."""
    used = np.zeros((side, side), dtype=bool)
    order = sorted(range(len(blocks)), key=lambda i: blocks[i]["eh"] * blocks[i]["ew"], reverse=True)

    def rec(k):
        if k == len(order):
            return True

        idx = order[k]
        block = blocks[idx]
        eh, ew = block["eh"], block["ew"]
        starts = []
        for r in range(side - eh + 1):
            for c in range(side - ew + 1):
                if not bool(np.any(used[r : r + eh, c : c + ew])):
                    starts.append((r, c))
        random.shuffle(starts)

        for r, c in starts:
            used[r : r + eh, c : c + ew] = True
            block["er0"], block["ec0"] = r, c
            if rec(k + 1):
                return True
            used[r : r + eh, c : c + ew] = False
            block.pop("er0", None)
            block.pop("ec0", None)
        return False

    return rec(0)

def _render_input_output(side, blocks, colors):
    inp = np.zeros((side, side), dtype=int)
    out = np.zeros((side, side), dtype=int)

    for block, color in zip(blocks, colors):
        er0, ec0 = block["er0"], block["ec0"]
        out[er0 : er0 + block["eh"], ec0 : ec0 + block["ew"]] = color

    for block, color in zip(blocks, colors):
        pad = block["pad"]
        r0 = block["er0"] + pad
        c0 = block["ec0"] + pad
        patch = np.full((block["h"], block["w"]), Color.YELLOW, dtype=int)
        interior = [(r, c) for r in range(1, block["h"] - 1) for c in range(1, block["w"] - 1)]
        for r, c in random.sample(interior, pad):
            patch[r, c] = color
        paste(inp, patch, r0, c0)
        paste(out, patch, r0, c0)

    return inp, out

def _gen_attempt():
    side = random.randint(*GRID_SIDE_RANGE)
    blocks = _sample_block_specs()
    if not _gen_place_expanded_boxes(side, blocks):
        return None

    colors = random.sample(MARKER_COLORS, len(blocks))
    inp, out = _render_input_output(side, blocks, colors)
    return inp if np.array_equal(main(inp), out) else None

def generate_input():
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 52fd389e input")
