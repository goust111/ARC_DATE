import random
from itertools import product

import numpy as np


BLACK = 0
AZURE = 8
MAROON = 9
BAND_COLORS = tuple(range(1, 8))

MIN_AXIS_SIZE = 6
MAX_AXIS_SIZE = 30
MAX_BANDS_PER_AXIS = 4
GEN_ATTEMPTS = 300


def _unique_nonblack(values: np.ndarray) -> int:
    vals = {int(v) for v in np.asarray(values).reshape(-1) if int(v) != BLACK}
    assert len(vals) == 1, "expected exactly one visible band color"
    return vals.pop()


def _band_starts(full_flags: np.ndarray) -> list[int]:
    starts: list[int] = []
    i = 0
    while i < len(full_flags) - 1:
        if bool(full_flags[i]) and bool(full_flags[i + 1]) and (i == 0 or not bool(full_flags[i - 1])):
            starts.append(i)
            i += 2
        else:
            i += 1
    return starts


def _bands_mask(length: int, starts: list[int]) -> np.ndarray:
    mask = np.zeros(length, dtype=bool)
    for start in starts:
        assert 0 <= start <= length - 2
        mask[start : start + 2] = True
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Redraw each 2x2 crossing of horizontal and vertical color bands according to its nearby flip marker.

    Input structure:
    - The background is black.
    - Several horizontal bands and several vertical bands, each exactly 2 pixels wide,
      span the grid; every band has its own non-black, non-Azure, non-Maroon color.
    - Each horizontal/vertical crossing forms a 2x2 overlap block whose current visible
      color is whichever band is on top there.
    - One cell down-right from every crossing sits a single marker pixel colored Azure
      or Maroon.

    Rule:
    - Identify the 2-row horizontal bands, the 2-column vertical bands, and every 2x2 crossing.
    - Read the current visible crossing color to infer which band is presently on top.
    - If the marker is Azure, keep that visible color; if the marker is Maroon, replace
      the whole 2x2 crossing with the other band's color.
    - Erase all marker pixels to black and leave every non-crossing band pixel unchanged.

    Key insight:
    - The crossing already encodes the current front/back ordering, while the nearby
      Azure-versus-Maroon marker only decides whether that ordering stays or flips.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    marker_mask = np.isin(g, (AZURE, MAROON))
    work = g.copy()
    work[marker_mask] = BLACK

    full_rows = np.all(work != BLACK, axis=1)
    full_cols = np.all(work != BLACK, axis=0)
    h_starts = _band_starts(full_rows)
    v_starts = _band_starts(full_cols)
    assert h_starts and v_starts, "expected both horizontal and vertical bands"

    h_mask = _bands_mask(h, h_starts)
    v_mask = _bands_mask(w, v_starts)
    assert np.array_equal(full_rows, h_mask), "full rows must match horizontal 2-row bands"
    assert np.array_equal(full_cols, v_mask), "full columns must match vertical 2-column bands"
    assert np.any(~h_mask) and np.any(~v_mask), "bands need uncovered rows and columns"

    h_colors: dict[int, int] = {}
    for r in h_starts:
        h_colors[r] = _unique_nonblack(work[r : r + 2, ~v_mask])

    v_colors: dict[int, int] = {}
    for c in v_starts:
        v_colors[c] = _unique_nonblack(work[~h_mask, c : c + 2])

    band_colors = list(h_colors.values()) + list(v_colors.values())
    assert len(set(band_colors)) == len(band_colors), "every band must have its own color"
    assert not set(band_colors) & {BLACK, AZURE, MAROON}

    expected_markers = {(r + 2, c + 2) for r, c in product(h_starts, v_starts)}
    actual_markers = {tuple(map(int, rc)) for rc in np.argwhere(marker_mask)}
    assert actual_markers == expected_markers, "markers must sit one cell down-right from each crossing"

    out = work.copy()
    for r, c in product(h_starts, v_starts):
        h_color = h_colors[r]
        v_color = v_colors[c]
        visible = _unique_nonblack(work[r : r + 2, c : c + 2])
        assert visible in (h_color, v_color), "crossing must show one of the two band colors"

        marker = int(g[r + 2, c + 2])
        if marker == AZURE:
            target = visible
        else:
            assert marker == MAROON
            target = v_color if visible == h_color else h_color
        out[r : r + 2, c : c + 2] = target

    return out


def _split_total(total: int, parts: int) -> list[int]:
    assert total >= 0 and parts >= 1
    if parts == 1:
        return [total]
    cuts = sorted(random.sample(range(total + parts - 1), parts - 1))
    out: list[int] = []
    prev = -1
    for cut in cuts + [total + parts - 1]:
        out.append(cut - prev - 1)
        prev = cut
    return out


def _sample_band_starts(length: int, count: int) -> list[int]:
    slack = length - 3 * count
    assert slack >= 0
    extras = _split_total(slack, count + 1)
    starts: list[int] = []
    pos = extras[0]
    for i in range(count):
        starts.append(pos)
        pos += 3 + extras[i + 1]
    return starts


def _balanced_choices(count: int, a, b) -> list:
    if count == 0:
        return []
    n_a = count // 2
    if count % 2 and random.random() < 0.5:
        n_a += 1
    values = [a] * n_a + [b] * (count - n_a)
    random.shuffle(values)
    return values


def _expected_output(
    grid: np.ndarray,
    h_starts: list[int],
    v_starts: list[int],
    h_colors: dict[int, int],
    v_colors: dict[int, int],
) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    out[np.isin(out, (AZURE, MAROON))] = BLACK
    for r, c in product(h_starts, v_starts):
        marker = int(grid[r + 2, c + 2])
        current = _unique_nonblack(grid[r : r + 2, c : c + 2])
        if marker == MAROON:
            current = v_colors[c] if current == h_colors[r] else h_colors[r]
        out[r : r + 2, c : c + 2] = current
    return out


def _count_options() -> list[tuple[int, int]]:
    return [
        (n_h, n_v)
        for n_h in range(1, MAX_BANDS_PER_AXIS + 1)
        for n_v in range(1, MAX_BANDS_PER_AXIS + 1)
        if n_h + n_v <= len(BAND_COLORS)
    ]


COUNT_OPTIONS = _count_options()


def _gen_attempt() -> np.ndarray | None:
    n_h, n_v = random.choice(COUNT_OPTIONS)

    min_h = max(MIN_AXIS_SIZE, 3 * n_h)
    min_w = max(MIN_AXIS_SIZE, 3 * n_v)
    if min_h > MAX_AXIS_SIZE or min_w > MAX_AXIS_SIZE:
        return None

    h = random.randint(min_h, MAX_AXIS_SIZE)
    w = random.randint(min_w, MAX_AXIS_SIZE)
    h_starts = _sample_band_starts(h, n_h)
    v_starts = _sample_band_starts(w, n_v)

    colors = random.sample(BAND_COLORS, n_h + n_v)
    h_colors = {r: color for r, color in zip(h_starts, colors[:n_h])}
    v_colors = {c: color for c, color in zip(v_starts, colors[n_h:])}

    grid = np.zeros((h, w), dtype=int)
    for r in h_starts:
        grid[r : r + 2, :] = h_colors[r]
    for c in v_starts:
        grid[:, c : c + 2] = v_colors[c]

    pairs = list(product(h_starts, v_starts))
    visible_choices = _balanced_choices(len(pairs), "h", "v")
    marker_choices = _balanced_choices(len(pairs), AZURE, MAROON)

    for (r, c), visible_kind, marker in zip(pairs, visible_choices, marker_choices):
        grid[r : r + 2, c : c + 2] = h_colors[r] if visible_kind == "h" else v_colors[c]
        grid[r + 2, c + 2] = marker

    expected = _expected_output(grid, h_starts, v_starts, h_colors, v_colors)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
