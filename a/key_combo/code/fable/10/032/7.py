import numpy as np


def _parse_grid(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    total_h, w = map(int, g.shape)
    assert total_h % 2 == 1, "expected a separator row between equal-height halves"
    h = total_h // 2
    assert 1 <= h and 1 <= w

    sep_row = h
    sep_vals = np.unique(g[sep_row])
    assert sep_vals.size == 1, "separator row must be solid"
    sep_color = int(sep_vals[0])

    top = g[:h, :]
    bottom = g[sep_row + 1 :, :]
    top_colors = set(map(int, np.unique(top)))
    bottom_colors = set(map(int, np.unique(bottom)))
    assert sep_color not in top_colors and sep_color not in bottom_colors, "separator color must be unique"

    shared = top_colors & bottom_colors
    assert len(shared) == 1, "the two halves must share exactly one background color"
    bg = next(iter(shared))

    top_fg_colors = top_colors - {bg}
    bottom_fg_colors = bottom_colors - {bg}
    assert len(top_fg_colors) == 1, "top half must have exactly one foreground color"
    assert len(bottom_fg_colors) == 1, "bottom half must have exactly one foreground color"
    top_fg = next(iter(top_fg_colors))
    bottom_fg = next(iter(bottom_fg_colors))

    top_mask = top == top_fg
    bottom_mask = bottom == bottom_fg
    top_count = int(top_mask.sum())
    bottom_count = int(bottom_mask.sum())
    assert 0 < top_count < h * w
    assert 0 < bottom_count < h * w
    assert top_count != bottom_count, "foreground pixel counts must differ"

    return g, h, w, bg, top_fg, bottom_fg, top_mask, bottom_mask


def _render_output(
    select_mask: np.ndarray,
    stamp_mask: np.ndarray,
    bg: int,
    stamp_color: int,
) -> np.ndarray:
    h, w = map(int, select_mask.shape)
    out = np.full((h * h, w * w), bg, dtype=int)
    tile = np.full((h, w), bg, dtype=int)
    tile[stamp_mask] = stamp_color

    for r, c in np.argwhere(select_mask):
        r0 = int(r) * h
        c0 = int(c) * w
        out[r0 : r0 + h, c0 : c0 + w] = tile
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Use the overlap of two separator-split halves to choose slots, then stamp the sparser half into each selected block.

    Input structure:
    - A solid separator row splits the grid into equal-size top and bottom halves.
    - The two halves share one background color, and each half has exactly one distinct foreground color.
    - The two foreground masks have different pixel counts.

    Rule:
    - Compare the two halves cell by cell and mark every position where both halves are foreground.
    - Create an output grid of size (h*h) x (w*w), divided into h x w block slots of size h x w.
    - In every selected slot, stamp the entire foreground-sparser half using its own foreground color; leave all other cells as the shared background color.

    Key insight:
    - The overlap mask only chooses where blocks appear; the image drawn inside each block is the whole sparser original half, not the overlap itself and not whichever half owns the slot.
    """
    _g, h, w, bg, top_fg, bottom_fg, top_mask, bottom_mask = _parse_grid(grid)
    overlap = top_mask & bottom_mask
    assert overlap.any(), "expected at least one overlapping foreground position"

    if int(top_mask.sum()) < int(bottom_mask.sum()):
        stamp_mask = top_mask
        stamp_color = top_fg
    else:
        stamp_mask = bottom_mask
        stamp_color = bottom_fg

    out = _render_output(overlap, stamp_mask, bg, stamp_color)
    assert out.shape == (h * h, w * w)
    return out


def _mask_bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    assert coords.size > 0
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _spans_two_axes(mask: np.ndarray) -> bool:
    r0, c0, r1, c1 = _mask_bbox(mask)
    return (r1 > r0) and (c1 > c0)


def _has_adjacent_pair(mask: np.ndarray) -> bool:
    return bool(np.any(mask[:, 1:] & mask[:, :-1]) or np.any(mask[1:, :] & mask[:-1, :]))


def _no_full_foreground_row(mask: np.ndarray) -> bool:
    return not np.any(mask.all(axis=1))


def _build_masks(
    h: int,
    w: int,
    sparse_is_top: bool,
    rng: np.random.Generator,
):
    area = h * w
    cells = np.arange(area)
    min_bg = 2

    for _ in range(400):
        overlap_count = int(rng.integers(2, min(6, area - 4) + 1))
        sparse_only = int(rng.integers(1, min(4, area - overlap_count - 3) + 1))
        dense_min = sparse_only + 1
        dense_max = min(6, area - overlap_count - sparse_only - min_bg)
        if dense_min > dense_max:
            continue
        dense_only = int(rng.integers(dense_min, dense_max + 1))
        bg_only = area - overlap_count - sparse_only - dense_only
        if bg_only < min_bg:
            continue

        perm = rng.permutation(cells)
        overlap_idx = perm[:overlap_count]
        sparse_only_idx = perm[overlap_count : overlap_count + sparse_only]
        dense_only_idx = perm[overlap_count + sparse_only : overlap_count + sparse_only + dense_only]

        overlap = np.zeros(area, dtype=bool)
        overlap[overlap_idx] = True
        sparse_mask = overlap.copy()
        dense_mask = overlap.copy()
        sparse_mask[sparse_only_idx] = True
        dense_mask[dense_only_idx] = True

        overlap = overlap.reshape(h, w)
        sparse_mask = sparse_mask.reshape(h, w)
        dense_mask = dense_mask.reshape(h, w)

        if not (_spans_two_axes(overlap) and _spans_two_axes(sparse_mask) and _spans_two_axes(dense_mask)):
            continue
        if not (_has_adjacent_pair(sparse_mask) and _has_adjacent_pair(dense_mask)):
            continue
        if not (_no_full_foreground_row(sparse_mask) and _no_full_foreground_row(dense_mask)):
            continue

        if sparse_is_top:
            return sparse_mask, dense_mask
        return dense_mask, sparse_mask

    raise RuntimeError("failed to sample non-degenerate masks")


def _pick_colors(rng: np.random.Generator):
    palette = rng.choice(np.arange(10), size=4, replace=False)
    bg, sep, top_fg, bottom_fg = map(int, palette)
    return bg, sep, top_fg, bottom_fg


def _assemble_grid(
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    bg: int,
    sep: int,
    top_fg: int,
    bottom_fg: int,
) -> np.ndarray:
    h, w = map(int, top_mask.shape)
    grid = np.full((2 * h + 1, w), bg, dtype=int)
    grid[h, :] = sep
    grid[:h, :][top_mask] = top_fg
    grid[h + 1 :, :][bottom_mask] = bottom_fg
    return grid


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    top_mask: np.ndarray,
    bottom_mask: np.ndarray,
    bg: int,
    top_fg: int,
    bottom_fg: int,
) -> bool:
    try:
        actual = main(grid)
    except Exception:
        return False
    if not np.array_equal(actual, expected):
        return False

    overlap = top_mask & bottom_mask
    union = top_mask | bottom_mask
    top_count = int(top_mask.sum())
    bottom_count = int(bottom_mask.sum())
    sparse_mask, sparse_color = (top_mask, top_fg) if top_count < bottom_count else (bottom_mask, bottom_fg)
    dense_mask, dense_color = (bottom_mask, bottom_fg) if top_count < bottom_count else (top_mask, top_fg)

    wrong_outputs = [
        _render_output(union, sparse_mask, bg, sparse_color),
        _render_output(top_mask, sparse_mask, bg, sparse_color),
        _render_output(bottom_mask, sparse_mask, bg, sparse_color),
        _render_output(overlap, dense_mask, bg, dense_color),
        _render_output(top_mask, top_mask, bg, top_fg),
        _render_output(bottom_mask, bottom_mask, bg, bottom_fg),
    ]
    if any(np.array_equal(expected, wrong) for wrong in wrong_outputs):
        return False

    return True


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    shapes = [(h, w) for h in range(3, 6) for w in range(3, 6)]

    for _ in range(500):
        h, w = shapes[int(rng.integers(len(shapes)))]
        sparse_is_top = bool(rng.integers(2))
        top_mask, bottom_mask = _build_masks(h, w, sparse_is_top, rng)
        bg, sep, top_fg, bottom_fg = _pick_colors(rng)
        grid = _assemble_grid(top_mask, bottom_mask, bg, sep, top_fg, bottom_fg)

        overlap = top_mask & bottom_mask
        if int(overlap.sum()) < 2:
            continue

        top_count = int(top_mask.sum())
        bottom_count = int(bottom_mask.sum())
        stamp_mask, stamp_color = (top_mask, top_fg) if top_count < bottom_count else (bottom_mask, bottom_fg)
        expected = _render_output(overlap, stamp_mask, bg, stamp_color)

        if _semantic_checks(grid, expected, top_mask, bottom_mask, bg, top_fg, bottom_fg):
            return grid

    raise RuntimeError("failed to generate a valid sample")
