import random

import numpy as np


BLACK = 0
GAP = 1
COLOR_POOL = tuple(range(1, 10))
LAYOUTS = (
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
    (3, 3),
)
SIDE_MIN = 4
SIDE_MAX = 7
MARGIN_MIN = 1
MARGIN_MAX = 3
MAX_ATTEMPTS = 400
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _component_bboxes(mask: np.ndarray) -> list[tuple[int, int, int, int]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    boxes: list[tuple[int, int, int, int]] = []

    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            r0 = r1 = r
            c0 = c1 = c

            while stack:
                cr, cc = stack.pop()
                r0 = min(r0, cr)
                r1 = max(r1, cr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            boxes.append((r0, c0, r1, c1))

    return boxes


def _extract_blocks(grid: np.ndarray) -> tuple[list[dict], list[int], list[int], int]:
    g = np.asarray(grid, dtype=int)
    components = _component_bboxes(g != BLACK)
    assert components, "expected at least one colored block"

    blocks: list[dict] = []
    side = None
    for r0, c0, r1, c1 in components:
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        h, w = map(int, patch.shape)
        assert h == w and h >= 3, "each block must occupy a square bbox"

        colors = np.unique(patch[patch != BLACK])
        assert len(colors) == 1, "each block must use exactly one foreground color"
        color = int(colors[0])

        assert bool((patch[0, :] == color).all())
        assert bool((patch[-1, :] == color).all())
        assert bool((patch[:, 0] == color).all())
        assert bool((patch[:, -1] == color).all())

        if side is None:
            side = h
        else:
            assert h == side, "all blocks must share the same size"

        blocks.append({"r0": int(r0), "c0": int(c0), "color": color, "patch": patch})

    assert side is not None

    row_positions = sorted({int(block["r0"]) for block in blocks})
    col_positions = sorted({int(block["c0"]) for block in blocks})
    assert len(blocks) == len(row_positions) * len(col_positions), "blocks must form a full matrix"

    for positions in (row_positions, col_positions):
        if len(positions) >= 2:
            diffs = [int(b - a) for a, b in zip(positions, positions[1:])]
            assert all(diff == side + GAP for diff in diffs), "adjacent blocks must be separated by one black cell"

    return blocks, row_positions, col_positions, int(side)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Invert the cut-out pattern inside each colored square and pack the results.

    Input structure:
    - Black background.
    - Several equal-sized colored square blocks arranged on a regular matrix with
      one black cell of spacing between neighboring blocks.
    - Inside each block, some interior cells are cut out to black.

    Rule:
    - For each block, invert the two-color bitmap inside its bbox: black cells
      become the block color and colored cells become black.
    - Keep the original row/column arrangement of the blocks.
    - Remove the spacing and tightly mosaic the inverted blocks in the output.
    """
    g = np.asarray(grid, dtype=int)
    blocks, row_positions, col_positions, side = _extract_blocks(g)
    row_index = {r0: i for i, r0 in enumerate(row_positions)}
    col_index = {c0: i for i, c0 in enumerate(col_positions)}

    out = np.full((len(row_positions) * side, len(col_positions) * side), BLACK, dtype=int)
    for block in blocks:
        rr = row_index[int(block["r0"])] * side
        cc = col_index[int(block["c0"])] * side
        patch = np.asarray(block["patch"], dtype=int)
        color = int(block["color"])
        stamp = np.where(patch == BLACK, color, BLACK)
        out[rr : rr + side, cc : cc + side] = stamp

    return out


def _random_blob(height: int, width: int) -> np.ndarray:
    target_max = min(height * width - 1, max(1, (height * width * 2) // 3))
    target = random.randint(1, target_max)

    start = (random.randrange(height), random.randrange(width))
    cells = {start}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _rectangle_mask(height: int, width: int) -> np.ndarray:
    rect_h = random.randint(1, height)
    rect_w = random.randint(1, width)
    if rect_h == height and rect_w == width and height * width > 1:
        if rect_h >= rect_w and rect_h > 1:
            rect_h -= 1
        else:
            rect_w -= 1

    r0 = random.randint(0, height - rect_h)
    c0 = random.randint(0, width - rect_w)
    mask = np.zeros((height, width), dtype=bool)
    mask[r0 : r0 + rect_h, c0 : c0 + rect_w] = True
    return mask


def _line_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    if random.random() < 0.5:
        row = random.randrange(height)
        length = random.randint(1, width)
        c0 = random.randint(0, width - length)
        mask[row, c0 : c0 + length] = True
    else:
        col = random.randrange(width)
        length = random.randint(1, height)
        r0 = random.randint(0, height - length)
        mask[r0 : r0 + length, col] = True
    return mask


def _l_mask(height: int, width: int) -> np.ndarray:
    arm_h = random.randint(1, height)
    arm_w = random.randint(1, width)
    r0 = random.randint(0, height - arm_h)
    c0 = random.randint(0, width - arm_w)
    corner_r = r0 if random.random() < 0.5 else r0 + arm_h - 1
    corner_c = c0 if random.random() < 0.5 else c0 + arm_w - 1

    mask = np.zeros((height, width), dtype=bool)
    mask[r0 : r0 + arm_h, corner_c] = True
    mask[corner_r, c0 : c0 + arm_w] = True
    return mask


def _plus_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    center_r = random.randrange(height)
    center_c = random.randrange(width)
    up = random.randint(0, center_r)
    down = random.randint(0, height - center_r - 1)
    left = random.randint(0, center_c)
    right = random.randint(0, width - center_c - 1)
    mask[center_r - up : center_r + down + 1, center_c] = True
    mask[center_r, center_c - left : center_c + right + 1] = True
    return mask


def _make_hole_mask(side: int) -> np.ndarray:
    height = side - 2
    width = side - 2
    builders = [_random_blob, _rectangle_mask, _line_mask, _l_mask]
    if height >= 3 and width >= 3:
        builders.append(_plus_mask)

    for _ in range(20):
        mask = random.choice(builders)(height, width)
        count = int(mask.sum())
        if 1 <= count < height * width:
            return mask

    fallback = np.zeros((height, width), dtype=bool)
    fallback[random.randrange(height), random.randrange(width)] = True
    return fallback


def _build_expected(blocks: list[np.ndarray], colors: list[int], rows: int, cols: int, side: int) -> np.ndarray:
    out = np.full((rows * side, cols * side), BLACK, dtype=int)
    idx = 0
    for r in range(rows):
        for c in range(cols):
            block = blocks[idx]
            color = int(colors[idx])
            out[r * side : (r + 1) * side, c * side : (c + 1) * side] = np.where(block == BLACK, color, BLACK)
            idx += 1
    return out


def _gen_attempt() -> np.ndarray | None:
    rows, cols = random.choice(LAYOUTS)
    block_count = rows * cols
    side = random.randint(SIDE_MIN, SIDE_MAX)
    top = random.randint(MARGIN_MIN, MARGIN_MAX)
    bottom = random.randint(MARGIN_MIN, MARGIN_MAX)
    left = random.randint(MARGIN_MIN, MARGIN_MAX)
    right = random.randint(MARGIN_MIN, MARGIN_MAX)
    colors = random.sample(COLOR_POOL, block_count)

    blocks: list[np.ndarray] = []
    signatures: set[bytes] = set()
    for _ in range(block_count):
        block = np.full((side, side), BLACK, dtype=int)
        color = colors[len(blocks)]
        block[:, :] = color
        holes = _make_hole_mask(side)
        block[1:-1, 1:-1][holes] = BLACK
        blocks.append(block)
        signatures.add(holes.tobytes())

    if block_count >= 2 and len(signatures) < 2:
        return None

    height = top + rows * side + (rows - 1) * GAP + bottom
    width = left + cols * side + (cols - 1) * GAP + right
    grid = np.full((height, width), BLACK, dtype=int)

    idx = 0
    for r in range(rows):
        for c in range(cols):
            r0 = top + r * (side + GAP)
            c0 = left + c * (side + GAP)
            grid[r0 : r0 + side, c0 : c0 + side] = blocks[idx]
            idx += 1

    expected = _build_expected(blocks, colors, rows, cols, side)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
