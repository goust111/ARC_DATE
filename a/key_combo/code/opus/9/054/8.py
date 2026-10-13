import random

import numpy as np


BLACK = 0
GREEN = 3
YELLOW = 4
RING_PALETTE = (1, 2, 3, 5, 6, 7, 8, 9)

GEN_ATTEMPTS = 400
SEQ_ATTEMPTS = 200


def _ring_coords(height: int, width: int, offset: int) -> list[tuple[int, int]]:
    r0, c0 = offset, offset
    r1, c1 = height - 1 - offset, width - 1 - offset
    coords = [(r0, c) for c in range(c0, c1 + 1)]
    if r1 > r0:
        coords.extend((r, c1) for r in range(r0 + 1, r1 + 1))
        coords.extend((r1, c) for c in range(c1 - 1, c0 - 1, -1))
    if c1 > c0:
        coords.extend((r, c0) for r in range(r1 - 1, r0, -1))
    return coords


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert coords.size > 0, "expected at least one foreground cell"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _read_ring_colors(section: np.ndarray) -> list[int]:
    non_bg = section != BLACK
    assert np.any(non_bg), "each half must contain a ring object"
    assert not np.any(section == YELLOW), "separator color may only appear on the divider row"

    r0, c0, r1, c1 = _bbox(non_bg)
    crop = np.asarray(section[r0 : r1 + 1, c0 : c1 + 1], dtype=int)

    assert np.all(crop != BLACK), "ring bbox must be fully filled by concentric rings"
    outside = section.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = BLACK
    assert np.all(outside == BLACK), "only the ring object may occupy each half"

    n_rings = (min(crop.shape) + 1) // 2
    assert 2 <= n_rings <= 5, "ring count must be between 2 and 5"

    colors: list[int] = []
    for offset in range(n_rings):
        values = {int(crop[r, c]) for r, c in _ring_coords(crop.shape[0], crop.shape[1], offset)}
        assert len(values) == 1, "each ring must be monochromatic"
        color = values.pop()
        assert color not in (BLACK, YELLOW), "ring colors must differ from background and separator"
        colors.append(color)
    return colors


def _xor_color_grid(top_colors: list[int], bottom_colors: list[int]) -> np.ndarray:
    top = np.array(top_colors, dtype=int)[:, None]
    bottom = np.array(bottom_colors, dtype=int)[None, :]
    out = np.zeros((len(top_colors), len(bottom_colors)), dtype=int)
    out[top != bottom] = GREEN
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the output size from the concentric-ring counts above and below the divider, then fill by ring-color mismatch.

    Input structure:
    - A black background contains one full yellow separator row that splits the grid into equal top and bottom halves.
    - Each half contains exactly one filled concentric rectangular-ring object; every ring is one cell thick and monochromatic.
    - Reading each half from the outer ring inward yields an ordered color sequence.

    Rule:
    - Locate the yellow divider row and split the grid into the top and bottom halves.
    - Read the top-half ring colors from outermost to innermost to get sequence `T` and count `R`.
    - Read the bottom-half ring colors from outermost to innermost to get sequence `B` and count `C`.
    - Produce an `R x C` output where cell `(r, c)` is black if `T[r] == B[c]`, otherwise green.

    Key insight:
    - The same concentric-ring structure encodes both the output dimensions (number of layers) and the output contents (pairwise equality of ring colors).
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and min(g.shape) > 0, "expected a non-empty 2D grid"

    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == YELLOW)]
    assert len(divider_rows) == 1, "expected exactly one full yellow separator row"
    sep = divider_rows[0]

    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape[0] == bottom.shape[0] and top.shape[0] > 0, "halves must be non-empty and equal-height"

    top_colors = _read_ring_colors(top)
    bottom_colors = _read_ring_colors(bottom)
    return _xor_color_grid(top_colors, bottom_colors)


def _paint_rings(grid: np.ndarray, r0: int, c0: int, height: int, width: int, colors: list[int]) -> None:
    for offset, color in enumerate(colors):
        rr0, cc0 = r0 + offset, c0 + offset
        rr1, cc1 = r0 + height - 1 - offset, c0 + width - 1 - offset
        grid[rr0, cc0 : cc1 + 1] = color
        grid[rr1, cc0 : cc1 + 1] = color
        grid[rr0 : rr1 + 1, cc0] = color
        grid[rr0 : rr1 + 1, cc1] = color


def _sample_sequence(length: int, require_repeat: bool) -> list[int]:
    require_repeat = bool(require_repeat and length >= 3)
    for _ in range(SEQ_ATTEMPTS):
        if require_repeat:
            pool_size = random.randint(2, min(4, length))
            pool = random.sample(RING_PALETTE, k=pool_size)
            seq = [random.choice(pool) for _ in range(length)]
            if len(set(seq)) == length:
                continue
        else:
            seq = random.sample(RING_PALETTE, k=length)
        if len(set(seq)) >= 2:
            return seq
    raise RuntimeError("failed to sample a non-degenerate ring-color sequence")


def _sample_sequences(n_top: int, n_bottom: int) -> tuple[list[int], list[int]]:
    want_repeat = random.random() < 0.5 and max(n_top, n_bottom) >= 3
    for _ in range(SEQ_ATTEMPTS):
        top_repeat = want_repeat and n_top >= 3 and random.random() < 0.65
        bottom_repeat = want_repeat and n_bottom >= 3 and random.random() < 0.65
        if want_repeat and not (top_repeat or bottom_repeat):
            if n_top >= 3 and (n_bottom < 3 or random.random() < 0.5):
                top_repeat = True
            elif n_bottom >= 3:
                bottom_repeat = True

        top = _sample_sequence(n_top, require_repeat=top_repeat)
        bottom = _sample_sequence(n_bottom, require_repeat=bottom_repeat)

        if not (set(top) & set(bottom)):
            shared = random.choice(top)
            bottom[random.randrange(n_bottom)] = shared

        output = _xor_color_grid(top, bottom)
        row_patterns = {tuple(row.tolist()) for row in output}
        col_patterns = {tuple(col.tolist()) for col in output.T}
        has_repeat = len(set(top)) < len(top) or len(set(bottom)) < len(bottom)

        if GREEN not in output or BLACK not in output:
            continue
        if len(row_patterns) < 2 or len(col_patterns) < 2:
            continue
        if want_repeat and not has_repeat:
            continue
        if not want_repeat and has_repeat:
            continue
        return top, bottom
    raise RuntimeError("failed to sample compatible top/bottom color sequences")


def _sample_object_dims(layers: int, half_h: int, grid_w: int, prefer_line: bool) -> tuple[int, int]:
    min_dim = 2 * layers - 1
    choices: list[tuple[int, int]] = []

    for width in range(min_dim, min(grid_w, min_dim + 6) + 1):
        if not prefer_line or width > min_dim:
            choices.append((min_dim, width))
    for height in range(min_dim, min(half_h, min_dim + 4) + 1):
        if not prefer_line or height > min_dim:
            choices.append((height, min_dim))
    if not prefer_line:
        choices.append((min_dim, min_dim))

    if not choices:
        choices.append((min_dim, min_dim))
    return random.choice(choices)


def _validate_generated(grid: np.ndarray, top_colors: list[int], bottom_colors: list[int]) -> None:
    g = np.asarray(grid, dtype=int)
    divider_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == YELLOW)]
    assert len(divider_rows) == 1
    sep = divider_rows[0]
    top = _read_ring_colors(g[:sep, :])
    bottom = _read_ring_colors(g[sep + 1 :, :])
    assert top == top_colors
    assert bottom == bottom_colors

    out = main(g)
    expected = _xor_color_grid(top_colors, bottom_colors)
    assert np.array_equal(out, expected)
    assert out.ndim == 2 and max(out.shape) <= 30
    assert BLACK in out and GREEN in out
    assert len({tuple(row.tolist()) for row in out}) >= 2
    assert len({tuple(col.tolist()) for col in out.T}) >= 2


def _gen_attempt() -> np.ndarray | None:
    n_top = random.randint(2, 5)
    n_bottom = random.randint(2, 5)

    min_needed = max(2 * n_top - 1, 2 * n_bottom - 1)
    half_h = random.randint(min_needed, min(11, min_needed + 2))
    grid_w = random.randint(max(6, min_needed), min(18, min_needed + 7))

    top_colors, bottom_colors = _sample_sequences(n_top, n_bottom)
    top_h, top_w = _sample_object_dims(n_top, half_h, grid_w, prefer_line=random.random() < 0.6)
    bottom_h, bottom_w = _sample_object_dims(n_bottom, half_h, grid_w, prefer_line=random.random() < 0.6)

    grid = np.zeros((2 * half_h + 1, grid_w), dtype=int)
    grid[half_h, :] = YELLOW

    top_r0 = random.randint(0, half_h - top_h)
    top_c0 = random.randint(0, grid_w - top_w)
    bottom_r0 = half_h + 1 + random.randint(0, half_h - bottom_h)
    bottom_c0 = random.randint(0, grid_w - bottom_w)

    _paint_rings(grid, top_r0, top_c0, top_h, top_w, top_colors)
    _paint_rings(grid, bottom_r0, bottom_c0, bottom_h, bottom_w, bottom_colors)

    try:
        _validate_generated(grid, top_colors, bottom_colors)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid concentric-ring XOR example")
