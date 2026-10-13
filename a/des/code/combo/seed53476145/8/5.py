import random
import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

MARKER_COLORS = (RED, BLUE, GREEN, YELLOW)
SPRITE_COLORS = (6, 7, 8, 9)

STAGING_SHAPE = (30, 30)
PANE_COUNT_CHOICES = (1, 2, 2, 3, 3, 4)
QUADRANT_SIZE_RANGE = (3, 5)
PLACEMENT_ATTEMPTS = 120
MAX_GENERATION_ATTEMPTS = 300
MAX_CROP_MARGIN = 3
QUADRANT_ATTEMPTS = 80


def _apply_transform(block: np.ndarray, marker_color: int) -> np.ndarray:
    if marker_color == RED:
        return block.copy()
    if marker_color == BLUE:
        return block[::-1, :].copy()
    if marker_color == GREEN:
        return block[:, ::-1].copy()
    if marker_color == YELLOW:
        return block[::-1, ::-1].copy()
    raise ValueError(f"unsupported marker color: {marker_color}")


def _find_panes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    marker_mask = np.isin(g, MARKER_COLORS)
    panes: list[tuple[int, int, int, int]] = []

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not marker_mask[r0, c0]:
                continue
            for c1 in range(c0 + 2, w):
                if not marker_mask[r0, c1]:
                    continue
                if not np.all(g[r0, c0 + 1 : c1] == GREY):
                    continue
                for r1 in range(r0 + 2, h):
                    if not marker_mask[r1, c0] or not marker_mask[r1, c1]:
                        continue
                    if not np.all(g[r1, c0 + 1 : c1] == GREY):
                        continue
                    if not np.all(g[r0 + 1 : r1, c0] == GREY):
                        continue
                    if not np.all(g[r0 + 1 : r1, c1] == GREY):
                        continue
                    if np.any(g[r0 + 1 : r1, c0 + 1 : c1] == GREY):
                        continue
                    panes.append((r0, c0, r1, c1))

    panes.sort()
    return panes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Transform each pane quadrant independently according to the pane-corner marker colors.

    Each grey rectangular frame has four marker-colored corners. The top-left, top-right,
    bottom-left, and bottom-right markers control the corresponding interior quadrants:
    red keeps the quadrant unchanged, blue flips it vertically, green flips it
    horizontally, and yellow rotates it 180 degrees.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for r0, c0, r1, c1 in _find_panes(g):
        inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        ih, iw = inner.shape
        top_end = ih // 2
        bottom_start = ih - ih // 2
        left_end = iw // 2
        right_start = iw - iw // 2

        new_inner = inner.copy()
        quadrants = [
            ((slice(0, top_end), slice(0, left_end)), int(g[r0, c0])),
            ((slice(0, top_end), slice(right_start, iw)), int(g[r0, c1])),
            ((slice(bottom_start, ih), slice(0, left_end)), int(g[r1, c0])),
            ((slice(bottom_start, ih), slice(right_start, iw)), int(g[r1, c1])),
        ]

        for (row_slice, col_slice), marker in quadrants:
            block = inner[row_slice, col_slice]
            new_inner[row_slice, col_slice] = _apply_transform(block, marker)

        out[r0 + 1 : r1, c0 + 1 : c1] = new_inner

    return out


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _random_quadrant_mask(rng: random.Random, h: int, w: int, marker_color: int) -> np.ndarray:
    area = h * w
    min_cells = max(2, area // 4)
    max_cells = max(min_cells, min(area - 1, int(np.ceil(area * 0.65))))

    for _ in range(QUADRANT_ATTEMPTS):
        target = rng.randint(min_cells, max_cells)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], h, w))

        while len(cells) < target and frontier:
            nxt = rng.choice(tuple(frontier))
            frontier.remove(nxt)
            if nxt in cells:
                continue
            cells.add(nxt)
            for nbr in _neighbors4(nxt[0], nxt[1], h, w):
                if nbr not in cells:
                    frontier.add(nbr)

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        if not mask.any():
            continue
        if marker_color != RED and np.array_equal(mask, _apply_transform(mask, marker_color)):
            continue
        return mask

    mask = np.zeros((h, w), dtype=bool)
    mask[0, 0] = True
    if h > 1:
        mask[1, 0] = True
    if w > 1:
        mask[0, w - 1] = True
    if marker_color != RED and np.array_equal(mask, _apply_transform(mask, marker_color)):
        mask[-1, 0] = True
    return mask


def _assemble_pane(
    qh: int,
    qw: int,
    corners: tuple[int, int, int, int],
    target_inner: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inner_h = 2 * qh
    inner_w = 2 * qw

    input_inner = target_inner.copy()
    quadrant_specs = [
        ((slice(0, qh), slice(0, qw)), corners[0]),
        ((slice(0, qh), slice(qw, inner_w)), corners[1]),
        ((slice(qh, inner_h), slice(0, qw)), corners[2]),
        ((slice(qh, inner_h), slice(qw, inner_w)), corners[3]),
    ]
    for (row_slice, col_slice), marker in quadrant_specs:
        block = target_inner[row_slice, col_slice]
        input_inner[row_slice, col_slice] = _apply_transform(block, marker)

    pane_h = inner_h + 2
    pane_w = inner_w + 2
    pane_in = np.full((pane_h, pane_w), BG, dtype=int)
    pane_out = np.full((pane_h, pane_w), BG, dtype=int)

    pane_in[0, 1:-1] = GREY
    pane_in[-1, 1:-1] = GREY
    pane_in[1:-1, 0] = GREY
    pane_in[1:-1, -1] = GREY
    pane_in[0, 0], pane_in[0, -1], pane_in[-1, 0], pane_in[-1, -1] = corners
    pane_in[1:-1, 1:-1] = input_inner

    pane_out[:] = pane_in
    pane_out[1:-1, 1:-1] = target_inner

    return pane_in, pane_out


def _make_target_inner(
    rng: random.Random, qh: int, qw: int, corners: tuple[int, int, int, int], sprite_color: int
) -> np.ndarray:
    inner = np.zeros((2 * qh, 2 * qw), dtype=int)
    quadrant_specs = [
        ((slice(0, qh), slice(0, qw)), corners[0]),
        ((slice(0, qh), slice(qw, 2 * qw)), corners[1]),
        ((slice(qh, 2 * qh), slice(0, qw)), corners[2]),
        ((slice(qh, 2 * qh), slice(qw, 2 * qw)), corners[3]),
    ]
    for (row_slice, col_slice), marker in quadrant_specs:
        mask = _random_quadrant_mask(rng, qh, qw, marker)
        block = inner[row_slice, col_slice].copy()
        block[mask] = sprite_color
        inner[row_slice, col_slice] = block
    return inner


def _place_panes(
    rng: random.Random, pane_inputs: list[np.ndarray], pane_outputs: list[np.ndarray]
) -> tuple[np.ndarray, np.ndarray] | None:
    stage_h, stage_w = STAGING_SHAPE
    grid_in = np.zeros((stage_h, stage_w), dtype=int)
    grid_out = np.zeros((stage_h, stage_w), dtype=int)
    occupied = np.zeros((stage_h, stage_w), dtype=bool)

    order = sorted(range(len(pane_inputs)), key=lambda idx: pane_inputs[idx].size, reverse=True)
    for idx in order:
        pane_in = pane_inputs[idx]
        pane_out = pane_outputs[idx]
        ph, pw = pane_in.shape
        if ph + 2 > stage_h or pw + 2 > stage_w:
            return None

        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            r0 = rng.randint(1, stage_h - ph - 1)
            c0 = rng.randint(1, stage_w - pw - 1)
            r1 = r0 + ph
            c1 = c0 + pw
            if occupied[r0 - 1 : r1 + 1, c0 - 1 : c1 + 1].any():
                continue

            grid_in[r0:r1, c0:c1] = pane_in
            grid_out[r0:r1, c0:c1] = pane_out
            occupied[r0:r1, c0:c1] = True
            placed = True
            break

        if not placed:
            return None

    points = np.argwhere(occupied)
    r0 = int(points[:, 0].min())
    r1 = int(points[:, 0].max())
    c0 = int(points[:, 1].min())
    c1 = int(points[:, 1].max())

    top_margin = rng.randint(0, min(MAX_CROP_MARGIN, r0))
    bottom_margin = rng.randint(0, min(MAX_CROP_MARGIN, stage_h - 1 - r1))
    left_margin = rng.randint(0, min(MAX_CROP_MARGIN, c0))
    right_margin = rng.randint(0, min(MAX_CROP_MARGIN, stage_w - 1 - c1))

    rows = slice(r0 - top_margin, r1 + bottom_margin + 1)
    cols = slice(c0 - left_margin, c1 + right_margin + 1)
    return grid_in[rows, cols], grid_out[rows, cols]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pane_count = rng.choice(PANE_COUNT_CHOICES)
    marker_pool = list(MARKER_COLORS) * pane_count
    rng.shuffle(marker_pool)

    pane_inputs: list[np.ndarray] = []
    pane_outputs: list[np.ndarray] = []
    for idx in range(pane_count):
        qh = rng.randint(*QUADRANT_SIZE_RANGE)
        qw = rng.randint(*QUADRANT_SIZE_RANGE)
        corners = tuple(marker_pool[4 * idx : 4 * idx + 4])
        sprite_color = rng.choice(SPRITE_COLORS)
        target_inner = _make_target_inner(rng, qh, qw, corners, sprite_color)
        pane_in, pane_out = _assemble_pane(qh, qw, corners, target_inner)
        pane_inputs.append(pane_in)
        pane_outputs.append(pane_out)

    placed = _place_panes(rng, pane_inputs, pane_outputs)
    if placed is None:
        return None
    grid_in, grid_out = placed

    panes = _find_panes(grid_in)
    if len(panes) != pane_count:
        return None
    if not np.array_equal(main(grid_in), grid_out):
        return None
    return grid_in


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pane-quadrant ARC input")
