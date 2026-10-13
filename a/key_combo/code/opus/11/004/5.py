from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
PALETTE_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ALLOWED_K = (3, 4)
ALLOWED_S = (2, 3)

GEN_GRID_ROWS = (16, 24)
GEN_GRID_COLS = (16, 24)
GEN_RELATIONS = ("above", "below", "left", "right")
GEN_MARGIN_RANGE = (1, 4)
GEN_MAX_ATTEMPTS = 600

_SEEN_OUTPUT_SIGS: set[tuple[tuple[int, int], bytes]] = set()


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _palette_from_grid(grid: np.ndarray) -> list[int]:
    coords = np.argwhere((grid != BLACK) & (grid != GREY))
    assert coords.size > 0, "expected a horizontal palette strip"
    rows = np.unique(coords[:, 0])
    assert rows.size == 1, "palette must lie on a single row"
    row = int(rows[0])
    cols = sorted(int(c) for c in coords[:, 1])
    assert cols == list(range(cols[0], cols[-1] + 1)), "palette pixels must be contiguous"
    palette = [int(grid[row, c]) for c in cols]
    assert 2 <= len(palette) <= 4, "palette length must be between 2 and 4"
    assert len(set(palette)) == len(palette), "palette colors must be distinct"
    return palette


def _infer_k_s(size: int) -> tuple[int, int]:
    candidates = [(k, s) for k in ALLOWED_K for s in ALLOWED_S if k * s == int(size)]
    assert len(candidates) == 1, f"ambiguous expanded mask size: {size}"
    return candidates[0]


def _downsample_mask(grid: np.ndarray) -> np.ndarray:
    box = _bbox(grid == GREY)
    assert box is not None, "expected a grey expanded pattern"
    r0, c0, r1, c1 = box
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w, "expanded pattern bbox must be square"

    k, s = _infer_k_s(h)
    sprite = grid[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(np.isin(sprite, (BLACK, GREY))), "expanded pattern must contain only black and grey"

    mask = np.zeros((k, k), dtype=int)
    for r in range(k):
        for c in range(k):
            block = sprite[r * s : (r + 1) * s, c * s : (c + 1) * s]
            if np.any(block == GREY):
                assert np.all(block == GREY), "bright source cells must be full SxS grey blocks"
                mask[r, c] = 1
            else:
                assert np.all(block == BLACK), "dark source cells must be full SxS black blocks"

    assert np.count_nonzero(mask) >= 3, "mask must contain at least three bright cells"
    assert np.count_nonzero(mask == 0) >= 2, "mask must contain at least two dark cells"
    return mask


def _render_from_mask(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    k = int(mask.shape[0])
    out = np.zeros((2 * k, 2 * k), dtype=int)
    period = len(palette)
    for tile_r in range(2):
        for tile_c in range(2):
            base_r = tile_r * k
            base_c = tile_c * k
            for r in range(k):
                for c in range(k):
                    if not mask[r, c]:
                        continue
                    depth = min(r, c, k - 1 - r, k - 1 - c)
                    out[base_r + r, base_c + c] = int(palette[depth % period])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Downsample the grey block pattern into a mask, tile it 2x2, and recolor bright cells by cyclic ring depth.

    Input structure:
    - The grid is mostly black background.
    - One side contains a grey block-expanded square pattern: a hidden KxK binary mask
      whose cells were enlarged into uniform SxS grey or black blocks.
    - Elsewhere there is one horizontal palette strip of contiguous non-grey, non-black colors.

    Rule:
    - Crop the grey pattern to its tight bounding box and infer the integer block scale from
      the square size; then downsample one SxS block per source cell to recover the KxK mask.
    - Read the palette colors from left to right.
    - Tile the binary mask 2x2 to form a 2K x 2K output.
    - For every bright cell, compute its local coordinates inside its own KxK tile and color it
      by palette[min(r, c, K-1-r, K-1-c) mod palette_length]; dark cells stay black.

    Key insight:
    - The grey source object supplies only the binary occupancy mask, while the palette and the
      tile-local concentric rectangle depth jointly determine the output colors.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette = _palette_from_grid(g)
    mask = _downsample_mask(g)
    return _render_from_mask(mask, palette)


def _wrong_prefix_output(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    k = int(mask.shape[0])
    out = np.zeros((2 * k, 2 * k), dtype=int)
    for tile_r in range(2):
        for tile_c in range(2):
            base_r = tile_r * k
            base_c = tile_c * k
            for r in range(k):
                for c in range(k):
                    if mask[r, c]:
                        out[base_r + r, base_c + c] = int(palette[min(r, c) % len(palette)])
    return out


def _wrong_global_depth_output(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    k = int(mask.shape[0])
    out = np.zeros((2 * k, 2 * k), dtype=int)
    for r in range(2 * k):
        for c in range(2 * k):
            if mask[r % k, c % k]:
                depth = min(r, c, 2 * k - 1 - r, 2 * k - 1 - c)
                out[r, c] = int(palette[depth % len(palette)])
    return out


def _wrong_full_square_output(mask: np.ndarray, palette: list[int]) -> np.ndarray:
    return _render_from_mask(np.ones_like(mask, dtype=int), palette)


def _output_sig(out: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, out.shape)), out.tobytes()


def _edge_cells(k: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(k)]
    if side == "bottom":
        return [(k - 1, c) for c in range(k)]
    if side == "left":
        return [(r, 0) for r in range(k)]
    if side == "right":
        return [(r, k - 1) for r in range(k)]
    raise ValueError(side)


def _random_mask(k: int) -> np.ndarray:
    inner_cells = [(r, c) for r in range(1, k - 1) for c in range(1, k - 1)]
    discriminating = [(k - 1, c) for c in range(1, k - 1)] + [(r, k - 1) for r in range(1, k - 1)]

    for _ in range(200):
        chosen: set[tuple[int, int]] = set()
        for side in ("top", "bottom", "left", "right"):
            chosen.add(random.choice(_edge_cells(k, side)))
        chosen.add(random.choice(inner_cells))
        chosen.add(random.choice(discriminating))

        for r in range(k):
            for c in range(k):
                if random.random() < 0.35:
                    chosen.add((r, c))

        if len(chosen) > k * k - 2:
            continue

        mask = np.zeros((k, k), dtype=int)
        for r, c in chosen:
            mask[r, c] = 1

        if np.count_nonzero(mask) < 3 or np.count_nonzero(mask == 0) < 2:
            continue
        if not np.any(mask[1 : k - 1, 1 : k - 1]):
            continue
        if not np.any(mask[k - 1, 1 : k - 1]) and not np.any(mask[1 : k - 1, k - 1]):
            continue
        if not all(np.any(mask[rng_r, rng_c]) for rng_r, rng_c in (
            (0, slice(None)),
            (k - 1, slice(None)),
            (slice(None), 0),
            (slice(None), k - 1),
        )):
            continue
        return mask
    raise RuntimeError("failed to build a non-degenerate mask")


def _expanded_pattern(mask: np.ndarray, scale: int) -> np.ndarray:
    return np.kron(mask * GREY, np.ones((scale, scale), dtype=int))


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    return (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _place_pattern_and_palette(
    rows: int,
    cols: int,
    pattern_shape: tuple[int, int],
    palette_len: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    ph, pw = map(int, pattern_shape)
    relation = random.choice(GEN_RELATIONS)
    margin = random.randint(*GEN_MARGIN_RANGE)

    for _ in range(120):
        if relation == "above":
            if rows < ph + margin + 1:
                return None
            pr = random.randint(margin + 1, rows - ph)
            pc = random.randint(0, cols - pw)
            lo, hi = 0, pr - margin - 1
            if lo > hi or cols < palette_len:
                continue
            rr = random.randint(lo, hi)
            cc = random.randint(0, cols - palette_len)
        elif relation == "below":
            if rows < ph + margin + 1:
                return None
            pr = random.randint(0, rows - ph - margin - 1)
            pc = random.randint(0, cols - pw)
            lo, hi = pr + ph + margin, rows - 1
            if lo > hi or cols < palette_len:
                continue
            rr = random.randint(lo, hi)
            cc = random.randint(0, cols - palette_len)
        elif relation == "left":
            if cols < pw + palette_len + margin:
                return None
            pr = random.randint(0, rows - ph)
            lo, hi = palette_len + margin, cols - pw
            if lo > hi:
                continue
            pc = random.randint(lo, hi)
            rr = random.randint(0, rows - 1)
            cc = random.randint(0, pc - margin - palette_len)
        else:
            if cols < pw + palette_len + margin:
                return None
            pr = random.randint(0, rows - ph)
            pc = random.randint(0, cols - pw - margin - palette_len)
            rr = random.randint(0, rows - 1)
            cc = random.randint(pc + pw + margin, cols - palette_len)

        pattern_box = (pr, pc, pr + ph - 1, pc + pw - 1)
        palette_box = (rr, cc, rr, cc + palette_len - 1)
        if _rect_separated(pattern_box, palette_box, pad=1):
            return (pr, pc), (rr, cc)
    return None


def _validate_sample(grid: np.ndarray, mask: np.ndarray, palette: list[int]) -> np.ndarray | None:
    try:
        out = main(grid)
    except Exception:
        return None

    expected = _render_from_mask(mask, palette)
    if not np.array_equal(out, expected):
        return None
    if np.array_equal(out, _wrong_prefix_output(mask, palette)):
        return None
    if np.array_equal(out, _wrong_global_depth_output(mask, palette)):
        return None
    if np.array_equal(out, _wrong_full_square_output(mask, palette)):
        return None

    used = set(map(int, np.unique(out)))
    if used != {BLACK, int(palette[0]), int(palette[1])}:
        return None

    sig = _output_sig(out)
    if sig in _SEEN_OUTPUT_SIGS:
        return None
    return out


def _attempt_generate() -> np.ndarray | None:
    k = random.choice(ALLOWED_K)
    s = random.choice(ALLOWED_S)
    mask = _random_mask(k)
    pattern = _expanded_pattern(mask, s)

    palette = random.sample(PALETTE_COLORS, 2)
    rows = random.randint(max(GEN_GRID_ROWS[0], pattern.shape[0] + 3), GEN_GRID_ROWS[1])
    cols = random.randint(max(GEN_GRID_COLS[0], pattern.shape[1] + 4), GEN_GRID_COLS[1])

    placement = _place_pattern_and_palette(rows, cols, pattern.shape, len(palette))
    if placement is None:
        return None
    (pr, pc), (rr, cc) = placement

    grid = np.zeros((rows, cols), dtype=int)
    grid[pr : pr + pattern.shape[0], pc : pc + pattern.shape[1]] = pattern
    grid[rr, cc : cc + len(palette)] = np.array(palette, dtype=int)

    out = _validate_sample(grid, mask, palette)
    if out is None:
        return None

    _SEEN_OUTPUT_SIGS.add(_output_sig(out))
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
