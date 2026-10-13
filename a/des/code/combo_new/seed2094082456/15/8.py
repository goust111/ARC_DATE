from __future__ import annotations

import random

import numpy as np


BG = 0
SOURCE_CENTER = 8
RECEIVER_CENTER = 7
TYPE_COLORS = (1, 2, 3, 4, 5, 6, 9)

BLOCK_SIZE = 4
INNER_SIZE = 2
MIN_GAP = 1
MAX_GAP = 8
PAIR_TRIES = 120
GEN_ATTEMPTS = 500


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _parse_blocks(grid: np.ndarray) -> list[dict]:
    bg = _background_color(grid)
    blocks: list[dict] = []

    for coords in _components4(grid != bg):
        pts = np.asarray(coords, dtype=int)
        r0, c0 = pts.min(axis=0)
        r1, c1 = pts.max(axis=0)
        size_r = int(r1 - r0 + 1)
        size_c = int(c1 - c0 + 1)
        if size_r != size_c or size_r < 3 or len(coords) != size_r * size_c:
            continue

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        border_mask = np.zeros((size_r, size_c), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True
        inner = patch[1:-1, 1:-1]
        if inner.size == 0:
            continue

        border_colors = np.unique(patch[border_mask])
        inner_colors = np.unique(inner)
        if len(border_colors) != 1 or len(inner_colors) != 1:
            continue

        border_color = int(border_colors[0])
        center_color = int(inner_colors[0])
        if border_color == bg or center_color == bg or border_color == center_color:
            continue
        if center_color not in (SOURCE_CENTER, RECEIVER_CENTER):
            continue

        blocks.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "inner_bbox": (int(r0 + 1), int(c0 + 1), int(r1 - 1), int(c1 - 1)),
                "border": border_color,
                "center": center_color,
                "inner_width": int(size_r - 2),
            }
        )

    return blocks


def _paint_horizontal(
    out: np.ndarray,
    rows: tuple[int, int],
    src_bbox: tuple[int, int, int, int],
    dst_bbox: tuple[int, int, int, int],
    color: int,
    bg: int,
) -> None:
    top, bottom = rows
    src_left = src_bbox[1] < dst_bbox[1]
    if src_left:
        positions = list(range(int(src_bbox[3] + 1), int(dst_bbox[1])))
    else:
        positions = list(range(int(src_bbox[1] - 1), int(dst_bbox[3]), -1))
    assert positions, "expected a non-empty horizontal gap"

    c0 = min(positions)
    c1 = max(positions)
    out[int(top) : int(bottom) + 1, int(c0) : int(c1) + 1] = int(bg)
    for offset, c in enumerate(positions):
        if offset % 2 == 0:
            out[int(top) : int(bottom) + 1, int(c)] = int(color)


def _paint_vertical(
    out: np.ndarray,
    cols: tuple[int, int],
    src_bbox: tuple[int, int, int, int],
    dst_bbox: tuple[int, int, int, int],
    color: int,
    bg: int,
) -> None:
    left, right = cols
    src_above = src_bbox[0] < dst_bbox[0]
    if src_above:
        positions = list(range(int(src_bbox[2] + 1), int(dst_bbox[0])))
    else:
        positions = list(range(int(src_bbox[0] - 1), int(dst_bbox[2]), -1))
    assert positions, "expected a non-empty vertical gap"

    r0 = min(positions)
    r1 = max(positions)
    out[int(r0) : int(r1) + 1, int(left) : int(right) + 1] = int(bg)
    for offset, r in enumerate(positions):
        if offset % 2 == 0:
            out[int(r), int(left) : int(right) + 1] = int(color)


def _paint_link(out: np.ndarray, source: dict, receiver: dict, bg: int) -> None:
    src_inner = source["inner_bbox"]
    dst_inner = receiver["inner_bbox"]
    src_bbox = source["bbox"]
    dst_bbox = receiver["bbox"]
    color = int(source["border"])

    same_inner_rows = src_inner[0] == dst_inner[0] and src_inner[2] == dst_inner[2]
    same_inner_cols = src_inner[1] == dst_inner[1] and src_inner[3] == dst_inner[3]
    assert same_inner_rows or same_inner_cols, "paired blocks must align horizontally or vertically"

    if same_inner_rows:
        _paint_horizontal(
            out,
            (int(src_inner[0]), int(src_inner[2])),
            src_bbox,
            dst_bbox,
            color,
            bg,
        )
        return

    _paint_vertical(
        out,
        (int(src_inner[1]), int(src_inner[3])),
        src_bbox,
        dst_bbox,
        color,
        bg,
    )


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair source and receiver blocks by frame color, then paint alternating stripes between them.

    Input structure:
    - The background is black.
    - The foreground consists of small solid squares whose outer ring is a single frame color and whose inner region is uniformly azure or orange.
    - Each frame color appears exactly twice, once with an azure center and once with an orange center; each matched pair is aligned in one row band or one column band.

    Rule:
    - Group squares by their shared frame color.
    - In each pair, treat the azure-centered square as the source and the orange-centered square as the receiver.
    - Fill the empty corridor between them with alternating one-cell stripes that start at the source side: frame color, then black, then frame color again, using the full inner width of the squares as the stripe thickness.

    Key insight:
    - The frame color encodes pairing, while the inner marker color encodes which end of the pair starts the alternating pattern.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _background_color(g)
    blocks = _parse_blocks(g)
    assert blocks, "expected at least one framed block"

    by_color: dict[int, list[dict]] = {}
    for block in blocks:
        by_color.setdefault(int(block["border"]), []).append(block)

    out = g.copy()
    for border_color, pair in by_color.items():
        assert len(pair) == 2, f"frame color {border_color} must appear exactly twice"
        sources = [block for block in pair if int(block["center"]) == SOURCE_CENTER]
        receivers = [block for block in pair if int(block["center"]) == RECEIVER_CENTER]
        assert len(sources) == 1 and len(receivers) == 1, "each pair needs one source and one receiver"
        source = sources[0]
        receiver = receivers[0]
        assert int(source["inner_width"]) == int(receiver["inner_width"])
        _paint_link(out, source, receiver, bg)

    return out


def _draw_block(grid: np.ndarray, top: int, left: int, border: int, center: int) -> None:
    grid[int(top) : int(top + BLOCK_SIZE), int(left) : int(left + BLOCK_SIZE)] = int(border)
    grid[
        int(top + 1) : int(top + 1 + INNER_SIZE),
        int(left + 1) : int(left + 1 + INNER_SIZE),
    ] = int(center)


def _reserve_ok(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> bool:
    h, w = reserved.shape
    rr0 = max(0, int(r0 - 1))
    cc0 = max(0, int(c0 - 1))
    rr1 = min(h - 1, int(r1 + 1))
    cc1 = min(w - 1, int(c1 + 1))
    return not bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any())


def _reserve(reserved: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> None:
    h, w = reserved.shape
    rr0 = max(0, int(r0 - 1))
    cc0 = max(0, int(c0 - 1))
    rr1 = min(h - 1, int(r1 + 1))
    cc1 = min(w - 1, int(c1 + 1))
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _place_horizontal_pair(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    border: int,
) -> bool:
    h, w = grid.shape
    max_gap = min(MAX_GAP, w - 2 * BLOCK_SIZE)
    if max_gap < MIN_GAP:
        return False

    for _ in range(PAIR_TRIES):
        gap = rng.randint(MIN_GAP, max_gap)
        total_w = 2 * BLOCK_SIZE + gap
        top = rng.randint(0, h - BLOCK_SIZE)
        left = rng.randint(0, w - total_w)
        r0, c0 = top, left
        r1, c1 = top + BLOCK_SIZE - 1, left + total_w - 1
        if not _reserve_ok(reserved, r0, c0, r1, c1):
            continue

        source_on_left = bool(rng.getrandbits(1))
        left_border = left
        right_border = left + BLOCK_SIZE + gap
        if source_on_left:
            source_left = left_border
            receiver_left = right_border
        else:
            receiver_left = left_border
            source_left = right_border

        _draw_block(grid, top, source_left, border, SOURCE_CENTER)
        _draw_block(grid, top, receiver_left, border, RECEIVER_CENTER)
        _reserve(reserved, r0, c0, r1, c1)
        return True

    return False


def _place_vertical_pair(
    rng: random.Random,
    grid: np.ndarray,
    reserved: np.ndarray,
    border: int,
) -> bool:
    h, w = grid.shape
    max_gap = min(MAX_GAP, h - 2 * BLOCK_SIZE)
    if max_gap < MIN_GAP:
        return False

    for _ in range(PAIR_TRIES):
        gap = rng.randint(MIN_GAP, max_gap)
        total_h = 2 * BLOCK_SIZE + gap
        top = rng.randint(0, h - total_h)
        left = rng.randint(0, w - BLOCK_SIZE)
        r0, c0 = top, left
        r1, c1 = top + total_h - 1, left + BLOCK_SIZE - 1
        if not _reserve_ok(reserved, r0, c0, r1, c1):
            continue

        source_on_top = bool(rng.getrandbits(1))
        upper_top = top
        lower_top = top + BLOCK_SIZE + gap
        if source_on_top:
            source_top = upper_top
            receiver_top = lower_top
        else:
            receiver_top = upper_top
            source_top = lower_top

        _draw_block(grid, source_top, left, border, SOURCE_CENTER)
        _draw_block(grid, receiver_top, left, border, RECEIVER_CENTER)
        _reserve(reserved, r0, c0, r1, c1)
        return True

    return False


def _pair_count_and_shape(rng: random.Random) -> tuple[int, int, int]:
    pair_count = rng.choice((2, 3, 4))
    if pair_count == 2:
        height = rng.randint(16, 24)
        width = rng.randint(16, 24)
    elif pair_count == 3:
        height = rng.randint(18, 28)
        width = rng.randint(18, 28)
    else:
        height = rng.randint(24, 30)
        width = rng.randint(24, 30)
    return pair_count, height, width


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count, height, width = _pair_count_and_shape(rng)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    colors = rng.sample(TYPE_COLORS, pair_count)
    horizontal_count = rng.randint(1, pair_count - 1)
    orientations = ["H"] * horizontal_count + ["V"] * (pair_count - horizontal_count)
    rng.shuffle(orientations)

    for border, orientation in zip(colors, orientations):
        if orientation == "H":
            ok = _place_horizontal_pair(rng, grid, reserved, border)
        else:
            ok = _place_vertical_pair(rng, grid, reserved, border)
        if not ok:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
