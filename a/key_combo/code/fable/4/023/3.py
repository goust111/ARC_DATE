import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
TEAL = 8
BAR_COLORS = (BLUE, TEAL)

GEN_RETRY_LIMIT = 300


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _local_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_border:
                holes += 1
    return holes


def _blob_period(grid: np.ndarray) -> int:
    grey_components = _components(grid == GREY)
    assert grey_components, "expected a grey blob"
    blob = max(grey_components, key=lambda coords: coords.shape[0])
    period = _hole_count(_local_mask(blob))
    assert period >= 1, "grey blob must contain at least one hole"
    return int(period)


def _ranked_bars(grid: np.ndarray) -> list[tuple[int, int, int, np.ndarray]]:
    bar_mask = (grid != BLACK) & (grid != GREY)
    bars = []
    for coords in _components(bar_mask):
        r0, c0, r1, c1 = _bbox_coords(coords)
        bars.append((int(c1 - c0 + 1), int(r0), int(c0), coords))
    bars.sort(key=lambda item: (item[0], item[1], item[2]))
    return bars


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor every N-th horizontal bar in length order, where N is the grey blob's hole count.

    Input structure:
    - Black cells form the background.
    - Several non-touching solid horizontal bars appear in blue and teal, with pairwise distinct lengths.
    - A single grey blob sits in a corner and contains one or more enclosed black holes.

    Rule:
    - Count the number of holes inside the grey blob, using 4-connected black regions in the blob's bounding box that do not touch that box border.
    - Sort the horizontal bars from shortest to longest and number them 1, 2, 3, ...
    - Recolor every bar whose rank is a multiple of the hole count to red, keep the other bars unchanged, and erase the grey blob to black.

    Key insight:
    - The grey shape is only a parameter carrier: its topology gives the period, and the selected bars are determined by length rank rather than by top-to-bottom position or color.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    out[g == GREY] = BLACK

    period = _blob_period(g)
    for rank, (_, _, _, coords) in enumerate(_ranked_bars(g), start=1):
        if rank % period == 0:
            out[coords[:, 0], coords[:, 1]] = RED
    return out


def _crop_nonblack(patch: np.ndarray) -> np.ndarray:
    coords = np.argwhere(patch != BLACK)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return patch[r0 : r1 + 1, c0 : c1 + 1]


def _make_blob_patch(rng: np.random.Generator, hole_count: int) -> np.ndarray:
    width = 2 * hole_count + 1
    patch = np.zeros((5, width + 2), dtype=int)
    patch[1:4, 1 : width + 1] = GREY
    patch[2, 2 : width + 1 : 2] = BLACK

    top_cols = np.flatnonzero(rng.random(width) < 0.45) + 1
    bottom_cols = np.flatnonzero(rng.random(width) < 0.45) + 1
    left_rows = np.flatnonzero(rng.random(3) < 0.65) + 1
    right_rows = np.flatnonzero(rng.random(3) < 0.65) + 1

    if top_cols.size + bottom_cols.size + left_rows.size + right_rows.size == 0:
        side = int(rng.integers(0, 4))
        if side == 0:
            top_cols = np.asarray([int(rng.integers(1, width + 1))])
        elif side == 1:
            bottom_cols = np.asarray([int(rng.integers(1, width + 1))])
        elif side == 2:
            left_rows = np.asarray([int(rng.integers(1, 4))])
        else:
            right_rows = np.asarray([int(rng.integers(1, 4))])

    patch[0, top_cols] = GREY
    patch[4, bottom_cols] = GREY
    patch[left_rows, 0] = GREY
    patch[right_rows, width + 1] = GREY

    cropped = _crop_nonblack(patch)
    assert _hole_count(cropped == GREY) == hole_count
    return cropped


def _choose_row_order(
    rng: np.random.Generator, sorted_lengths: list[int], mode: str
) -> list[int]:
    if mode == "ascending":
        return list(sorted_lengths)
    if mode == "descending":
        return list(reversed(sorted_lengths))

    perm = list(map(int, rng.permutation(sorted_lengths)))
    while perm == sorted_lengths or perm == list(reversed(sorted_lengths)):
        perm = list(map(int, rng.permutation(sorted_lengths)))
    return perm


def _sample_bar_colors(
    rng: np.random.Generator, n_bars: int, selected_rows: set[int]
) -> list[int]:
    for _ in range(80):
        colors = list(map(int, rng.choice(np.asarray(BAR_COLORS), size=n_bars, replace=True)))
        if len(set(colors)) < 2:
            continue
        sel_colors = {colors[i] for i in selected_rows}
        other_colors = {colors[i] for i in range(n_bars) if i not in selected_rows}
        if len(selected_rows) >= 2 and len(sel_colors) < 2:
            continue
        if n_bars - len(selected_rows) >= 2 and len(other_colors) < 2:
            continue
        return colors

    colors = [BAR_COLORS[i % 2] for i in range(n_bars)]
    if len({colors[i] for i in selected_rows}) < 2:
        for i in sorted(selected_rows)[1::2]:
            colors[i] = BAR_COLORS[1]
        for i in sorted(selected_rows)[::2]:
            colors[i] = BAR_COLORS[0]
    remaining = [i for i in range(n_bars) if i not in selected_rows]
    if len({colors[i] for i in remaining}) < 2:
        for j, i in enumerate(remaining):
            colors[i] = BAR_COLORS[j % 2]
    return list(map(int, colors))


def _bar_rows_top(
    rng: np.random.Generator, start_row: int, n_bars: int
) -> list[int]:
    gaps = list(map(int, rng.integers(1, 3, size=max(0, n_bars - 1))))
    rows = [int(start_row)]
    for gap in gaps:
        rows.append(rows[-1] + 1 + gap)
    return rows


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    hole_count = int(rng.integers(2, 5))
    n_bars = int(2 * hole_count + rng.integers(0, 3))

    blob_patch = _make_blob_patch(rng, hole_count)
    blob_h, blob_w = map(int, blob_patch.shape)

    min_width = max(18, blob_w + 4, n_bars + 6)
    if min_width > 30:
        return None
    width = int(rng.integers(min_width, 31))

    max_bar_length = min(18, width - 2)
    choices = np.arange(3, max_bar_length + 1)
    if choices.size < n_bars:
        return None
    sorted_lengths = sorted(map(int, rng.choice(choices, size=n_bars, replace=False)))

    mode = str(rng.choice(np.asarray(["ascending", "descending", "shuffled"])))
    row_lengths = _choose_row_order(rng, sorted_lengths, mode)

    selected_lengths = set(sorted_lengths[hole_count - 1 :: hole_count])
    selected_rows = {i for i, length in enumerate(row_lengths) if length in selected_lengths}
    if not selected_rows or len(selected_rows) == n_bars:
        return None

    bar_colors = _sample_bar_colors(rng, n_bars, selected_rows)

    top_blob = bool(rng.integers(0, 2))
    left_blob = bool(rng.integers(0, 2))

    if top_blob:
        bar_rows = _bar_rows_top(rng, blob_h + int(rng.integers(2, 4)), n_bars)
        height = bar_rows[-1] + 2
        blob_r0 = 1
    else:
        bar_rows = _bar_rows_top(rng, int(rng.integers(1, 3)), n_bars)
        blob_r0 = bar_rows[-1] + int(rng.integers(2, 4))
        height = blob_r0 + blob_h + 1

    if height > 30:
        return None

    blob_c0 = 1 if left_blob else width - blob_w - 1
    if blob_c0 < 1:
        return None

    grid = np.zeros((height, width), dtype=int)
    grid[blob_r0 : blob_r0 + blob_h, blob_c0 : blob_c0 + blob_w] = blob_patch

    for row, length, color in zip(bar_rows, row_lengths, bar_colors):
        c0 = int(rng.integers(1, width - length))
        grid[row, c0 : c0 + length] = int(color)

    if len(_components(grid == GREY)) != 1:
        return None
    if _blob_period(grid) != hole_count:
        return None

    out = main(grid)
    if np.array_equal(out, grid):
        return None
    if int((out == RED).sum()) == 0:
        return None
    if not np.all(out[grid == GREY] == BLACK):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
