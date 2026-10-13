import numpy as np


_RNG = np.random.default_rng()
_GEN_ATTEMPTS = 400


def _boundary_coords(height: int, width: int) -> list[tuple[int, int]]:
    if height == 1:
        return [(0, c) for c in range(width)]
    if width == 1:
        return [(r, 0) for r in range(height)]

    coords = [(0, c) for c in range(width)]
    coords.extend((r, width - 1) for r in range(1, height - 1))
    coords.extend((height - 1, c) for c in range(width - 1, -1, -1))
    coords.extend((r, 0) for r in range(height - 2, 0, -1))
    return coords


def _background_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[int(np.argmax(counts))])


def _foreground_color(grid: np.ndarray, bg: int) -> int:
    colors = [int(color) for color in np.unique(grid) if int(color) != bg]
    assert len(colors) == 1, "expected exactly one foreground color"
    return colors[0]


def _boundary_mask(grid: np.ndarray, fg: int) -> tuple[list[tuple[int, int]], np.ndarray]:
    coords = _boundary_coords(*grid.shape)
    mask = np.array([int(grid[r, c]) == fg for r, c in coords], dtype=bool)

    boundary = set(coords)
    fg_coords = {tuple(map(int, rc)) for rc in np.argwhere(grid == fg)}
    assert fg_coords, "expected at least one foreground cell"
    assert fg_coords <= boundary, "foreground must lie entirely on the boundary"
    return coords, mask


def _segment_start_and_length(mask: np.ndarray) -> tuple[int, int]:
    mask = np.asarray(mask, dtype=bool)
    length = int(mask.size)
    assert length > 0

    total = int(mask.sum())
    assert total > 0, "expected a non-empty boundary segment"
    if total == length:
        return 0, length

    starts = [idx for idx in range(length) if mask[idx] and not mask[(idx - 1) % length]]
    assert len(starts) == 1, "foreground must form exactly one cyclic run"

    start = int(starts[0])
    run_length = 0
    while run_length < length and mask[(start + run_length) % length]:
        run_length += 1

    assert run_length == total, "foreground must be contiguous on the cyclic boundary"
    return start, run_length


def _replicated_mask(path_len: int, start: int, seg_len: int) -> np.ndarray:
    out = np.zeros(path_len, dtype=bool)
    period = 2 * seg_len
    for offset in range(path_len):
        if offset % period < seg_len:
            out[(start + offset) % path_len] = True
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a boundary segment around the rectangular perimeter with period 2n.

    Input structure:
    - The background is the dominant color.
    - Exactly one foreground color is present.
    - Foreground cells form one continuous segment on the rectangular boundary,
      possibly wrapping across the boundary cycle origin.

    Rule:
    - Unroll the boundary clockwise into a cyclic path.
    - Let the foreground segment length be n.
    - Along one full lap of that cyclic path, copy the segment every 2n
      positions, preserving the original phase.

    Key insight:
    - The object lives on the perimeter cycle, so repetition follows boundary
      order rather than row/column extension.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _background_color(g)
    fg = _foreground_color(g, bg)
    coords, mask = _boundary_mask(g, fg)
    start, seg_len = _segment_start_and_length(mask)

    out = np.full(g.shape, bg, dtype=int)
    repeated = _replicated_mask(len(coords), start, seg_len)
    for keep, (r, c) in zip(repeated, coords):
        if keep:
            out[r, c] = fg
    return out


def _sample_segment_length(path_len: int, cell_count: int) -> int | None:
    max_len = min((path_len - 1) // 2, (cell_count - 1) // 2)
    if max_len < 1:
        return None

    small_cap = min(max_len, max(1, path_len // 6))
    if small_cap == max_len or _RNG.random() < 0.5:
        return int(_RNG.integers(1, small_cap + 1))
    return int(_RNG.integers(small_cap + 1, max_len + 1))


def _make_input(
    height: int,
    width: int,
    bg: int,
    fg: int,
    start: int,
    seg_len: int,
) -> np.ndarray:
    coords = _boundary_coords(height, width)
    grid = np.full((height, width), bg, dtype=int)
    for offset in range(seg_len):
        r, c = coords[(start + offset) % len(coords)]
        grid[r, c] = fg
    return grid


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(4, 13))
    width = int(_RNG.integers(4, 13))
    coords = _boundary_coords(height, width)
    path_len = len(coords)

    seg_len = _sample_segment_length(path_len, height * width)
    if seg_len is None:
        return None

    wrap = bool(seg_len > 1 and _RNG.random() < 0.5)
    if wrap:
        start = int(_RNG.integers(path_len - seg_len + 1, path_len))
    else:
        start = int(_RNG.integers(0, path_len - seg_len + 1))

    bg, fg = map(int, _RNG.choice(np.arange(10), size=2, replace=False))
    grid = _make_input(height, width, bg, fg, start, seg_len)

    _, mask = _boundary_mask(grid, fg)
    found_start, found_len = _segment_start_and_length(mask)
    if found_start != start or found_len != seg_len:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None

    out_fg = int(np.count_nonzero(out == fg))
    in_fg = int(np.count_nonzero(grid == fg))
    if out_fg <= in_fg:
        return None

    if wrap != (start + seg_len > path_len):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid boundary-period sample")
