import itertools
from collections import Counter

import numpy as np


BG = 0
FRAME = 3
ROW_COUNT = 4
STRIPE_HEIGHT = 2
PATTERN_WIDTH = 10
GEN_ATTEMPTS = 400
PALETTE = (1, 2, 4, 5, 6, 7, 8, 9)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _parse_single_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]):
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    full_rows = [r for r in range(sub.shape[0]) if np.all(sub[r, :] == FRAME)]
    full_cols = [c for c in range(sub.shape[1]) if np.all(sub[:, c] == FRAME)]
    if len(full_rows) != ROW_COUNT + 1 or len(full_cols) != 2:
        return None
    if full_rows[0] != 0 or full_rows[-1] != sub.shape[0] - 1:
        return None
    if full_cols[0] != 0 or full_cols[-1] != sub.shape[1] - 1:
        return None
    if any(b - a < 2 for a, b in zip(full_rows, full_rows[1:])):
        return None
    if full_cols[1] - full_cols[0] < 2:
        return None

    row_set = set(full_rows)
    col_set = set(full_cols)
    for rr, cc in np.argwhere(sub == FRAME):
        if rr not in row_set and cc not in col_set:
            return None

    slot_colors: list[int | None] = []
    for i in range(ROW_COUNT):
        slot = sub[full_rows[i] + 1 : full_rows[i + 1], full_cols[0] + 1 : full_cols[1]]
        if slot.size == 0:
            return None
        colors = set(map(int, np.unique(slot)))
        colors.discard(BG)
        colors.discard(FRAME)
        if len(colors) > 1:
            return None
        if colors:
            color = next(iter(colors))
            if not np.all(slot == color):
                return None
            slot_colors.append(color)
        else:
            if not np.all(slot == BG):
                return None
            slot_colors.append(None)

    return {"bbox": bbox, "slot_colors": slot_colors}


def _parse_frames(grid: np.ndarray) -> list[dict]:
    frames = []
    for coords in _components(grid == FRAME):
        parsed = _parse_single_frame(grid, _bbox_coords(coords))
        if parsed is not None:
            frames.append(parsed)
    assert len(frames) == 2, "expected exactly two lattice frames"
    frames.sort(key=lambda frame: (frame["bbox"][1], frame["bbox"][0]))
    return frames


def _infer_row_colors(frames: list[dict]) -> list[int]:
    inferred = []
    for row in range(ROW_COUNT):
        seen = [frame["slot_colors"][row] for frame in frames if frame["slot_colors"][row] is not None]
        assert seen, "every logical row needs at least one color hint"
        assert len(set(seen)) == 1, "conflicting row hints are not allowed"
        inferred.append(int(seen[0]))
    assert len(set(inferred)) == ROW_COUNT, "the four inferred row colors must be distinct"
    return inferred


def _right_pattern_bbox(grid: np.ndarray, frames: list[dict]) -> tuple[int, int, int, int]:
    mask = grid != BG
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mask[r0 : r1 + 1, c0 : c1 + 1] = False
    coords = np.argwhere(mask)
    assert coords.size > 0, "right pattern area is missing"
    bbox = _bbox_coords(coords)
    r0, c0, r1, c1 = bbox
    sub = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(sub != BG), "pattern area must be fully colored in the input"
    assert sub.shape[0] % ROW_COUNT == 0, "pattern area height must split into four stripes"
    return bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer one color per lattice row and use it as the row inversion key for the right-side pattern.

    Input structure:
    - The background is black.
    - Two green rectangular lattice frames appear on the left; each frame has 4 logical rows and 1 logical column of slot interiors.
    - Some slot interiors are solid-colored hints, and a separate colored pattern block on the right contains 4 horizontal stripes.

    Rule:
    - Parse the two lattice frames and, for each logical row, collect the solid slot color shown by any frame in that row.
    - Use that inferred row color as the key for the correspondingly ordered right-side stripe.
    - In each stripe, replace key-color pixels with black and replace every other pixel in the stripe with the key color; keep the lattices unchanged.

    Key insight:
    - The right-side stripes are deliberately ambiguous under row-local statistics, so the lattice-row color inference is the only way to choose which of the two stripe colors must be inverted.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames = _parse_frames(g)
    keys = _infer_row_colors(frames)
    r0, c0, r1, c1 = _right_pattern_bbox(g, frames)
    stripe_h = (r1 - r0 + 1) // ROW_COUNT

    out = g.copy()
    for row, key in enumerate(keys):
        rs = r0 + row * stripe_h
        re = rs + stripe_h
        stripe = g[rs:re, c0 : c1 + 1]
        colors = [color for color in np.unique(stripe) if color != BG]
        assert len(colors) == 2 and key in colors, "each stripe must contain the inferred key and one other color"
        assert stripe.shape[1] % 2 == 0, "stripe width must allow an equal-color tie"
        for band_row in stripe:
            counts = Counter(map(int, band_row))
            assert len(counts) == 2 and all(v == stripe.shape[1] // 2 for v in counts.values()), (
                "each stripe row must have a perfect 5:5-style color tie"
            )
        out[rs:re, c0 : c1 + 1] = np.where(stripe == key, BG, key)

    return out


def _draw_frame(
    grid: np.ndarray,
    top: int,
    left: int,
    heights: list[int],
    slot_w: int,
    hinted_rows: set[int],
    row_colors: list[int],
):
    total_h = sum(heights) + ROW_COUNT + 1
    total_w = slot_w + 2
    grid[top, left : left + total_w] = FRAME
    row_cursor = top
    for idx, height in enumerate(heights):
        grid[row_cursor : row_cursor + height + 2, left] = FRAME
        grid[row_cursor : row_cursor + height + 2, left + total_w - 1] = FRAME
        grid[row_cursor + height + 1, left : left + total_w] = FRAME
        if idx in hinted_rows:
            grid[row_cursor + 1 : row_cursor + height + 1, left + 1 : left + total_w - 1] = row_colors[idx]
        row_cursor += height + 1
    return top + total_h - 1, left + total_w - 1


def _max_run(mask: np.ndarray) -> int:
    best = 0
    current = 0
    for value in mask:
        if value:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


def _sample_masks(rng: np.random.Generator) -> list[np.ndarray] | None:
    masks = []
    for cols in itertools.combinations(range(PATTERN_WIDTH), PATTERN_WIDTH // 2):
        mask = np.zeros(PATTERN_WIDTH, dtype=bool)
        mask[list(cols)] = True
        masks.append(mask)
    choice = rng.choice(len(masks), size=ROW_COUNT, replace=False)
    picked = [masks[i] for i in choice]
    if not any(mask[0] for mask in picked):
        return None
    if not any(not mask[0] for mask in picked):
        return None
    if not any(_max_run(mask) >= 3 for mask in picked):
        return None
    if not any(_max_run(mask) <= 2 for mask in picked):
        return None
    return picked


def _manual_output(grid: np.ndarray, right_bbox: tuple[int, int, int, int], keys: list[int]) -> np.ndarray:
    out = grid.copy()
    r0, c0, r1, c1 = right_bbox
    for row, key in enumerate(keys):
        rs = r0 + row * STRIPE_HEIGHT
        re = rs + STRIPE_HEIGHT
        stripe = grid[rs:re, c0 : c1 + 1]
        out[rs:re, c0 : c1 + 1] = np.where(stripe == key, BG, key)
    return out


def _validate_generated(
    grid: np.ndarray,
    expected: np.ndarray,
    keys: list[int],
    right_bbox: tuple[int, int, int, int],
):
    parsed_frames = _parse_frames(grid)
    parsed_keys = _infer_row_colors(parsed_frames)
    assert parsed_keys == keys
    assert len(set(parsed_keys)) == ROW_COUNT
    assert _right_pattern_bbox(grid, parsed_frames) == right_bbox

    leftmost_key = []
    r0, c0, _r1, c1 = right_bbox
    for row, key in enumerate(keys):
        rs = r0 + row * STRIPE_HEIGHT
        stripe = grid[rs : rs + STRIPE_HEIGHT, c0 : c1 + 1]
        colors = [color for color in np.unique(stripe) if color != BG]
        assert len(colors) == 2 and key in colors
        other = colors[0] if colors[1] == key else colors[1]
        correct = np.where(stripe == key, BG, key)
        wrong = np.where(stripe == other, BG, other)
        assert not np.array_equal(correct, wrong), "choosing the other stripe color must change the result"
        assert np.any(correct != stripe), "the inversion step must visibly matter"
        leftmost_key.append(int(stripe[0, 0]) == key)
        for band_row in stripe:
            counts = Counter(map(int, band_row))
            assert counts[key] == counts[other] == PATTERN_WIDTH // 2

    assert any(leftmost_key) and not all(leftmost_key), "leftmost color cannot solve every stripe"
    actual = main(grid)
    assert np.array_equal(actual, expected)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    palette = list(rng.permutation(PALETTE))
    row_colors = palette[:ROW_COUNT]
    stripe_others = palette[ROW_COUNT:]

    masks = _sample_masks(rng)
    if masks is None:
        return None

    heights1 = rng.integers(1, 4, size=ROW_COUNT).tolist()
    heights2 = rng.integers(1, 4, size=ROW_COUNT).tolist()
    slot_w1 = int(rng.integers(2, 4))
    slot_w2 = int(rng.integers(2, 4))
    frame_h1 = sum(heights1) + ROW_COUNT + 1
    frame_h2 = sum(heights2) + ROW_COUNT + 1
    frame_w1 = slot_w1 + 2
    frame_w2 = slot_w2 + 2

    if heights1 == heights2 and slot_w1 == slot_w2:
        return None

    left_margin = int(rng.integers(1, 3))
    gap_frames = int(rng.integers(1, 3))
    gap_right = int(rng.integers(2, 5))
    right_margin = int(rng.integers(1, 3))
    width = left_margin + frame_w1 + gap_frames + frame_w2 + gap_right + PATTERN_WIDTH + right_margin
    if width > 30:
        return None

    top1 = int(rng.integers(1, 4))
    top2 = int(rng.integers(1, 4))
    right_top = int(rng.integers(1, 5))
    bottom_margin = int(rng.integers(1, 4))
    height = max(top1 + frame_h1, top2 + frame_h2, right_top + ROW_COUNT * STRIPE_HEIGHT) + bottom_margin
    if height > 30:
        return None

    grid = np.zeros((height, width), dtype=int)
    frame1_left = left_margin
    frame2_left = frame1_left + frame_w1 + gap_frames
    right_left = frame2_left + frame_w2 + gap_right
    frame1_rows = set(rng.choice(ROW_COUNT, size=2, replace=False).tolist())
    frame2_rows = set(range(ROW_COUNT)) - frame1_rows

    _draw_frame(grid, top1, frame1_left, heights1, slot_w1, frame1_rows, row_colors)
    _draw_frame(grid, top2, frame2_left, heights2, slot_w2, frame2_rows, row_colors)

    for row, (mask, key, other) in enumerate(zip(masks, row_colors, stripe_others)):
        band = np.where(mask, key, other).astype(int)
        rs = right_top + row * STRIPE_HEIGHT
        grid[rs : rs + STRIPE_HEIGHT, right_left : right_left + PATTERN_WIDTH] = band

    right_bbox = (
        right_top,
        right_left,
        right_top + ROW_COUNT * STRIPE_HEIGHT - 1,
        right_left + PATTERN_WIDTH - 1,
    )
    expected = _manual_output(grid, right_bbox, row_colors)

    try:
        _validate_generated(grid, expected, row_colors, right_bbox)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
