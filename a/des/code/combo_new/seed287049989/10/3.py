from __future__ import annotations

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
PATTERN_COLORS = (3, 4, 5, 6, 7, 8, 9)
MAX_ATTEMPTS = 400
STEP_MAX = 6
TILE_MAX = 12


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    red_mask = grid == RED
    box = _bbox(red_mask)
    assert box is not None, "expected one red frame"
    r0, c0, r1, c1 = box
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must have a non-empty interior"

    border = np.zeros_like(red_mask, dtype=bool)
    border[r0, c0 : c1 + 1] = True
    border[r1, c0 : c1 + 1] = True
    border[r0 : r1 + 1, c0] = True
    border[r0 : r1 + 1, c1] = True

    assert np.array_equal(red_mask, border), "red pixels must form exactly one rectangular frame"
    return r0, c0, r1, c1


def _repeat_patch(patch: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    ph, pw = map(int, patch.shape)
    h, w = map(int, shape)
    rows = np.arange(h)[:, None] % ph
    cols = np.arange(w)[None, :] % pw
    return patch[rows, cols].copy()


def _period_mismatch(tile: np.ndarray, step_r: int, step_c: int) -> int:
    recon = _repeat_patch(tile[:step_r, :step_c], tile.shape)
    return int(np.count_nonzero(tile != recon))


def _choose_steps(tile: np.ndarray, marker: tuple[int, int]) -> tuple[int, int]:
    mr, mc = marker
    h, w = map(int, tile.shape)
    candidates: list[tuple[int, int, int, int]] = []

    direct = (mr, mc)
    if 1 <= direct[0] <= h and 1 <= direct[1] <= w:
        candidates.append((_period_mismatch(tile, direct[0], direct[1]), 0, direct[0], direct[1]))

    plus_one = (mr + 1, mc + 1)
    if 1 <= plus_one[0] <= h and 1 <= plus_one[1] <= w:
        candidates.append((_period_mismatch(tile, plus_one[0], plus_one[1]), 1, plus_one[0], plus_one[1]))

    assert candidates, "marker must encode at least one positive tiling step"
    best = min(candidates, key=lambda item: (item[0], item[1]))
    return int(best[2]), int(best[3])


def _render_output(base: np.ndarray, shape: tuple[int, int], anchor: tuple[int, int]) -> np.ndarray:
    h, w = map(int, shape)
    ar, ac = map(int, anchor)
    bh, bw = map(int, base.shape)
    rows = (np.arange(h)[:, None] - ar) % bh
    cols = (np.arange(w)[None, :] - ac) % bw
    return base[rows, cols].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the framed motif across the whole grid using the inner blue marker as the tiling offset.

    Input structure:
    - The background is black.
    - A single red one-pixel-thick rectangular frame appears somewhere in the grid.
    - Inside the frame is a multicolor motif on a black interior plus exactly one blue marker pixel.

    Rule:
    - Extract the frame interior as the tile and locate the blue marker inside it.
    - Replace that marker cell with black; its local row/column position encodes the tiling steps.
    - Repeat the tile pattern over the entire grid from the tile's original interior position using those steps.
    - Remove the red frame itself in the output.

    Key insight:
    - The blue pixel is control metadata rather than motif content: its offset inside the frame tells you the wallpaper period.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    r0, c0, r1, c1 = _find_frame(g)
    tile = g[r0 + 1 : r1, c0 + 1 : c1].copy()
    assert tile.size > 0, "frame interior must be non-empty"
    assert not np.any(tile == RED), "red should only appear on the frame"

    marker_cells = np.argwhere(tile == BLUE)
    assert marker_cells.shape[0] == 1, "expected exactly one blue marker"
    mr, mc = map(int, marker_cells[0])
    tile[mr, mc] = BLACK

    step_r, step_c = _choose_steps(tile, (mr, mc))
    base = tile[:step_r, :step_c]
    return _render_output(base, g.shape, (r0 + 1, c0 + 1))


def _valid_steps() -> list[tuple[int, int]]:
    return [(sr, sc) for sr in range(1, STEP_MAX + 1) for sc in range(1, STEP_MAX + 1) if sr * sc >= 3]


def _sample_base(rng: np.random.Generator, step_r: int, step_c: int) -> np.ndarray | None:
    area = step_r * step_c
    marker = (step_r - 1, step_c - 1)
    free_cells = [(r, c) for r in range(step_r) for c in range(step_c) if (r, c) != marker]
    if len(free_cells) < 2:
        return None

    base = np.full((step_r, step_c), BLACK, dtype=int)
    max_colors = min(3, len(free_cells))
    n_colors = int(rng.integers(2, max_colors + 1))
    colors = [int(v) for v in rng.choice(PATTERN_COLORS, size=n_colors, replace=False)]

    perm = rng.permutation(len(free_cells))
    chosen = [free_cells[int(i)] for i in perm[:n_colors]]
    for color, (r, c) in zip(colors, chosen):
        base[r, c] = color

    leftovers = [free_cells[int(i)] for i in perm[n_colors:]]
    if leftovers:
        extra_hi = len(leftovers) + 1
        extra_count = int(rng.integers(0, extra_hi))
        if extra_count:
            fill_perm = rng.permutation(len(leftovers))[:extra_count]
            for idx in np.atleast_1d(fill_perm):
                r, c = leftovers[int(idx)]
                base[r, c] = colors[int(rng.integers(0, len(colors)))]

    if len(set(map(int, np.unique(base))) - {BLACK}) < 2:
        return None
    return base


def _sample_tile_shape(rng: np.random.Generator, step_r: int, step_c: int) -> tuple[int, int]:
    rep_r = int(rng.integers(1, 4))
    rep_c = int(rng.integers(1, 4))
    extra_r = int(rng.integers(0, step_r))
    extra_c = int(rng.integers(0, step_c))
    tile_h = step_r * rep_r + extra_r
    tile_w = step_c * rep_c + extra_c
    tile_h = max(tile_h, step_r)
    tile_w = max(tile_w, step_c)
    tile_h = min(tile_h, TILE_MAX)
    tile_w = min(tile_w, TILE_MAX)
    return int(tile_h), int(tile_w)


def _ambiguous_with_direct(tile: np.ndarray, marker: tuple[int, int]) -> bool:
    mr, mc = marker
    if mr <= 0 or mc <= 0:
        return False
    return _period_mismatch(tile, mr, mc) == 0


def _build_input(
    base: np.ndarray,
    grid_shape: tuple[int, int],
    frame_top: int,
    frame_left: int,
    tile_shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    grid_h, grid_w = map(int, grid_shape)
    tile_h, tile_w = map(int, tile_shape)
    step_r, step_c = map(int, base.shape)
    marker = (step_r - 1, step_c - 1)

    tile = _repeat_patch(base, (tile_h, tile_w))
    expected = _render_output(base, (grid_h, grid_w), (frame_top + 1, frame_left + 1))

    inp = np.full((grid_h, grid_w), BLACK, dtype=int)
    inp[frame_top, frame_left : frame_left + tile_w + 2] = RED
    inp[frame_top + tile_h + 1, frame_left : frame_left + tile_w + 2] = RED
    inp[frame_top : frame_top + tile_h + 2, frame_left] = RED
    inp[frame_top : frame_top + tile_h + 2, frame_left + tile_w + 1] = RED
    inp[frame_top + 1 : frame_top + 1 + tile_h, frame_left + 1 : frame_left + 1 + tile_w] = tile
    inp[frame_top + 1 + marker[0], frame_left + 1 + marker[1]] = BLUE
    return inp, expected


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    step_r, step_c = _valid_steps()[int(rng.integers(0, len(_valid_steps())))]
    base = _sample_base(rng, step_r, step_c)
    if base is None:
        return None

    tile_h, tile_w = _sample_tile_shape(rng, step_r, step_c)
    tile = _repeat_patch(base, (tile_h, tile_w))
    marker = (step_r - 1, step_c - 1)
    if _ambiguous_with_direct(tile, marker):
        return None

    frame_h, frame_w = tile_h + 2, tile_w + 2
    min_grid_h = max(8, frame_h)
    min_grid_w = max(8, frame_w)
    if min_grid_h > 30 or min_grid_w > 30:
        return None

    grid_h = int(rng.integers(min_grid_h, 31))
    grid_w = int(rng.integers(min_grid_w, 31))
    frame_top = int(rng.integers(0, grid_h - frame_h + 1))
    frame_left = int(rng.integers(0, grid_w - frame_w + 1))

    inp, expected = _build_input(base, (grid_h, grid_w), frame_top, frame_left, (tile_h, tile_w))
    if np.count_nonzero(inp == BLUE) != 1:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
