import random
from collections import Counter

import numpy as np


BLACK = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400


def _spans(active: np.ndarray) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = None
    for idx, flag in enumerate(active.tolist()):
        if flag and start is None:
            start = idx
        elif not flag and start is not None:
            spans.append((start, idx - 1))
            start = None
    if start is not None:
        spans.append((start, len(active) - 1))
    return spans


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the shared block template by majority vote, then overlay every anomalous replacement pixel onto one empty block.

    Input structure:
    - The background is black.
    - Foreground is arranged as a regular grid of same-sized rectangular blocks, with black separator rows and columns between neighboring blocks.
    - Every block contains the same black-and-one-color base sprite.
    - Some blocks replace exactly one base-sprite pixel with a second foreground color, creating a one-pixel anomaly.

    Rule:
    - Detect the repeated block row bands and column bands by the non-black spans between black separator lines.
    - Crop every block and recover the common base sprite by majority vote at each local coordinate across all blocks.
    - For each block that differs from the recovered base sprite, record the local coordinate of its single differing pixel.
    - Output one empty block-sized grid and paint the anomaly color at every recorded local coordinate.

    Key insight:
    - The correct output is distributed across many nearly identical blocks: first infer the normal template from the repeated structure, then aggregate the local outlier coordinates into one shared coordinate frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    row_spans = _spans((g != BLACK).any(axis=1))
    col_spans = _spans((g != BLACK).any(axis=0))
    assert row_spans and col_spans and len(row_spans) * len(col_spans) >= 2

    row_heights = {r1 - r0 + 1 for r0, r1 in row_spans}
    col_widths = {c1 - c0 + 1 for c0, c1 in col_spans}
    assert len(row_heights) == 1 and len(col_widths) == 1
    tile_h = row_heights.pop()
    tile_w = col_widths.pop()

    tiles = []
    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            tile = g[r0 : r1 + 1, c0 : c1 + 1]
            assert tile.shape == (tile_h, tile_w)
            tiles.append(tile.copy())
    stack = np.stack(tiles, axis=0)

    base = np.zeros((tile_h, tile_w), dtype=int)
    for r in range(tile_h):
        for c in range(tile_w):
            counts = Counter(map(int, stack[:, r, c]))
            base[r, c] = int(counts.most_common(1)[0][0])

    out = np.zeros((tile_h, tile_w), dtype=int)
    anomaly_colors = set()
    for tile in tiles:
        diffs = np.argwhere(tile != base)
        assert diffs.shape[0] <= 1
        if diffs.shape[0] == 0:
            continue
        r, c = map(int, diffs[0])
        value = int(tile[r, c])
        assert value != BLACK
        out[r, c] = value
        anomaly_colors.add(value)

    if anomaly_colors:
        assert len(anomaly_colors) == 1
    return out


def _choose_layout() -> tuple[int, int, int, int, int, int, int, int]:
    layouts = []
    for tile_rows in range(1, 5):
        for tile_cols in range(1, 5):
            block_count = tile_rows * tile_cols
            if block_count < 4:
                continue
            for tile_h in range(4, 8):
                for tile_w in range(4, 8):
                    for pad_top in range(0, 3):
                        for pad_bottom in range(0, 3):
                            for pad_left in range(0, 3):
                                for pad_right in range(0, 3):
                                    height = pad_top + tile_rows * tile_h + (tile_rows - 1) + pad_bottom
                                    width = pad_left + tile_cols * tile_w + (tile_cols - 1) + pad_right
                                    if height <= 30 and width <= 30:
                                        layouts.append(
                                            (
                                                tile_rows,
                                                tile_cols,
                                                tile_h,
                                                tile_w,
                                                pad_top,
                                                pad_bottom,
                                                pad_left,
                                                pad_right,
                                            )
                                        )
    return random.choice(layouts)


def _neighbors(h: int, w: int, r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr, nc = r + dr, c + dc
        if 0 <= nr < h and 0 <= nc < w:
            out.append((nr, nc))
    return out


def _cross_blob_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    spine_r = random.randrange(h)
    spine_c = random.randrange(w)
    mask[spine_r, :] = True
    mask[:, spine_c] = True

    target = random.randint(max(mask.sum(), int(np.ceil(0.35 * h * w))), max(mask.sum(), int(np.floor(0.6 * h * w))))
    while int(mask.sum()) < target:
        frontier = [tuple(map(int, pt)) for pt in np.argwhere(mask)]
        r, c = random.choice(frontier)
        candidates = [(nr, nc) for nr, nc in _neighbors(h, w, r, c) if not mask[nr, nc]]
        if not candidates:
            continue
        nr, nc = random.choice(candidates)
        mask[nr, nc] = True
    return mask


def _ring_blob_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True

    additions = random.randint(1, 4)
    for _ in range(additions):
        if random.random() < 0.5 and h > 2:
            r = random.randint(1, h - 2)
            if random.random() < 0.5:
                end = random.randint(1, w - 2)
                mask[r, 1 : end + 1] = True
            else:
                start = random.randint(1, w - 2)
                mask[r, start:w - 1] = True
        elif w > 2:
            c = random.randint(1, w - 2)
            if random.random() < 0.5:
                end = random.randint(1, h - 2)
                mask[1 : end + 1, c] = True
            else:
                start = random.randint(1, h - 2)
                mask[start:h - 1, c] = True
    return mask


def _sample_base_mask(h: int, w: int) -> np.ndarray:
    for _ in range(80):
        branch = random.choice(("cross", "ring"))
        if branch == "cross":
            mask = _cross_blob_mask(h, w)
        else:
            mask = _ring_blob_mask(h, w)

        if np.all(mask.any(axis=0)) and np.all(mask.any(axis=1)):
            if 0 < int(mask.sum()) < h * w:
                return mask
    raise RuntimeError("failed to sample a valid base mask")


def _tile_origin(tile_r: int, tile_c: int, tile_h: int, tile_w: int, pad_top: int, pad_left: int) -> tuple[int, int]:
    return pad_top + tile_r * (tile_h + 1), pad_left + tile_c * (tile_w + 1)


def _render_input(
    tile_rows: int,
    tile_cols: int,
    tile_h: int,
    tile_w: int,
    pad_top: int,
    pad_bottom: int,
    pad_left: int,
    pad_right: int,
    mask: np.ndarray,
    base_color: int,
    anomaly_color: int,
    anomaly_blocks: list[int],
    anomaly_positions: list[tuple[int, int]],
) -> tuple[np.ndarray, np.ndarray]:
    height = pad_top + tile_rows * tile_h + (tile_rows - 1) + pad_bottom
    width = pad_left + tile_cols * tile_w + (tile_cols - 1) + pad_right
    grid = np.zeros((height, width), dtype=int)

    block_positions = [(idx // tile_cols, idx % tile_cols) for idx in range(tile_rows * tile_cols)]
    for tile_r, tile_c in block_positions:
        r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w, pad_top, pad_left)
        patch = np.where(mask, base_color, BLACK)
        grid[r0 : r0 + tile_h, c0 : c0 + tile_w] = patch

    for block_idx, (ar, ac) in zip(anomaly_blocks, anomaly_positions):
        tile_r, tile_c = block_positions[block_idx]
        r0, c0 = _tile_origin(tile_r, tile_c, tile_h, tile_w, pad_top, pad_left)
        assert mask[ar, ac]
        grid[r0 + ar, c0 + ac] = anomaly_color

    expected = np.zeros((tile_h, tile_w), dtype=int)
    for ar, ac in anomaly_positions:
        expected[ar, ac] = anomaly_color
    return grid, expected


def _gen_attempt() -> np.ndarray | None:
    tile_rows, tile_cols, tile_h, tile_w, pad_top, pad_bottom, pad_left, pad_right = _choose_layout()
    block_count = tile_rows * tile_cols

    base_color, anomaly_color = random.sample(PALETTE, 2)
    mask = _sample_base_mask(tile_h, tile_w)
    base_positions = [tuple(map(int, pt)) for pt in np.argwhere(mask)]

    max_anomalies = min(len(base_positions), (block_count - 1) // 2)
    if max_anomalies <= 0:
        return None
    anomaly_count = random.randint(1, max_anomalies)
    anomaly_positions = random.sample(base_positions, anomaly_count)
    anomaly_blocks = random.sample(list(range(block_count)), anomaly_count)

    grid, expected = _render_input(
        tile_rows,
        tile_cols,
        tile_h,
        tile_w,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        mask,
        base_color,
        anomaly_color,
        anomaly_blocks,
        anomaly_positions,
    )

    row_spans = _spans((grid != BLACK).any(axis=1))
    col_spans = _spans((grid != BLACK).any(axis=0))
    if len(row_spans) != tile_rows or len(col_spans) != tile_cols:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
