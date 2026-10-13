import random
from collections import Counter

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5

MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_separator_col(grid: np.ndarray) -> int:
    cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == GREY)]
    assert len(cols) == 1, "expected exactly one solid grey separator column"
    return int(cols[0])


def _extract_legend_square(legend: np.ndarray, color: int) -> int:
    coords = np.argwhere(legend == color)
    assert coords.size > 0, f"legend must contain color {color}"
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    assert h == w and h in (2, 3), "legend sprite must be a 2x2 or 3x3 solid square"
    assert coords.shape[0] == h * w, "legend sprite must be solid"
    patch = legend[r0 : r1 + 1, c0 : c1 + 1]
    assert np.all(patch == color), "legend sprite must fill its bbox"
    return int(h)


def _extract_marker(work: np.ndarray, color: int) -> tuple[int, int]:
    coords = np.argwhere(work == color)
    assert coords.shape[0] == 1, f"workspace must contain exactly one marker of color {color}"
    return int(coords[0, 0]), int(coords[0, 1])


def _periodic_origins(center: int, size: int, limit: int) -> list[int]:
    start = int(center) - size // 2
    period = size + 1
    origins = [start]

    pos = start - period
    while pos + size > 0:
        origins.append(int(pos))
        pos -= period

    pos = start + period
    while pos < limit:
        origins.append(int(pos))
        pos += period

    return sorted(origins)


def _paint_square(dst: np.ndarray, r0: int, c0: int, size: int, color: int) -> None:
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + size)
    cc1 = min(dst.shape[1], int(c0) + size)
    if rr0 < rr1 and cc0 < cc1:
        dst[rr0:rr1, cc0:cc1] = int(color)


def _overlay_blue(dst: np.ndarray, r0: int, c0: int, size: int) -> None:
    rr0 = max(0, int(r0))
    cc0 = max(0, int(c0))
    rr1 = min(dst.shape[0], int(r0) + size)
    cc1 = min(dst.shape[1], int(c0) + size)
    if rr0 >= rr1 or cc0 >= cc1:
        return
    view = dst[rr0:rr1, cc0:cc1]
    overlap = view == RED
    view[~overlap] = BLUE
    view[overlap] = YELLOW


def _apply_workspace_rule(
    work: np.ndarray,
    red_size: int,
    blue_size: int,
    red_marker: tuple[int, int],
    blue_marker: tuple[int, int],
) -> np.ndarray:
    out = np.array(work, copy=True)
    assert set(np.unique(out)).issubset({BLACK, BLUE, RED})

    red_r, red_c = map(int, red_marker)
    blue_r, blue_c = map(int, blue_marker)
    red_row0 = red_r - red_size // 2
    blue_col0 = blue_c - blue_size // 2

    red_cols = _periodic_origins(red_c, red_size, out.shape[1])
    for col0 in red_cols:
        _paint_square(out, red_row0, int(col0), red_size, RED)

    blue_rows = _periodic_origins(blue_r, blue_size, out.shape[0])
    for row0 in blue_rows:
        _overlay_blue(out, int(row0), blue_col0, blue_size)

    return out


def _stamp_count(center: int, size: int, limit: int) -> int:
    return len(_periodic_origins(center, size, limit))


def _sample_square_row(height: int, size: int, top_min: int, top_max: int) -> int:
    lo = max(0, int(top_min))
    hi = min(height - size, int(top_max))
    assert lo <= hi
    return random.randint(lo, hi)


def _sample_legend(height: int, panel_w: int, red_size: int, blue_size: int) -> tuple[int, int, int, int]:
    red_col = random.randint(0, panel_w - red_size)
    blue_col = random.randint(0, panel_w - blue_size)

    max_red_top = height - (red_size + blue_size + 1)
    red_top = _sample_square_row(height, red_size, 0, max_red_top)
    blue_min = red_top + red_size + 1
    blue_top = _sample_square_row(height, blue_size, blue_min, height - blue_size)
    return red_top, red_col, blue_top, blue_col


def _marker_bias_positions(limit: int, size: int, prefer_edge: bool) -> list[int]:
    if not prefer_edge or limit <= 2 * size + 4:
        return list(range(limit))
    edge_band = min(size + 1, limit - 1)
    positions = list(range(edge_band)) + list(range(limit - edge_band, limit))
    return sorted(set(int(x) for x in positions))


def _sample_markers(
    work_h: int,
    work_w: int,
    red_size: int,
    blue_size: int,
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    prefer_red_edge = random.random() < 0.5
    prefer_blue_edge = random.random() < 0.5

    red_rows = list(range(work_h))
    red_cols = _marker_bias_positions(work_w, red_size, prefer_red_edge)
    blue_rows = _marker_bias_positions(work_h, blue_size, prefer_blue_edge)
    blue_cols = list(range(work_w))

    attempts = 300
    min_gap = 2 * max(red_size, blue_size)
    for _ in range(attempts):
        red_marker = (random.choice(red_rows), random.choice(red_cols))
        blue_marker = (random.choice(blue_rows), random.choice(blue_cols))
        if red_marker[0] == blue_marker[0] or red_marker[1] == blue_marker[1]:
            continue
        if max(abs(red_marker[0] - blue_marker[0]), abs(red_marker[1] - blue_marker[1])) < min_gap:
            continue

        out = _apply_workspace_rule(
            np.zeros((work_h, work_w), dtype=int),
            red_size,
            blue_size,
            red_marker,
            blue_marker,
        )
        yellow_count = int((out == YELLOW).sum())
        if yellow_count == 0:
            continue
        if yellow_count > (red_size * blue_size * 3):
            continue
        if _stamp_count(red_marker[1], red_size, work_w) < 3:
            continue
        if _stamp_count(blue_marker[0], blue_size, work_h) < 3:
            continue
        return red_marker, blue_marker
    return None


def _build_input(
    work_h: int,
    work_w: int,
    panel_w: int,
    red_size: int,
    blue_size: int,
    red_marker: tuple[int, int],
    blue_marker: tuple[int, int],
) -> np.ndarray:
    grid = np.zeros((work_h, panel_w + 1 + work_w), dtype=int)
    sep = panel_w
    grid[:, sep] = GREY

    red_top, red_col, blue_top, blue_col = _sample_legend(work_h, panel_w, red_size, blue_size)
    grid[red_top : red_top + red_size, red_col : red_col + red_size] = RED
    grid[blue_top : blue_top + blue_size, blue_col : blue_col + blue_size] = BLUE

    rr, rc = red_marker
    br, bc = blue_marker
    grid[rr, sep + 1 + rc] = RED
    grid[br, sep + 1 + bc] = BLUE
    return grid


def _semantic_checks(inp: np.ndarray, out: np.ndarray) -> bool:
    try:
        sep = _find_separator_col(inp)
        legend = inp[:, :sep]
        work = inp[:, sep + 1 :]
        assert _get_bg(inp) == BLACK
        red_size = _extract_legend_square(legend, RED)
        blue_size = _extract_legend_square(legend, BLUE)
        red_marker = _extract_marker(work, RED)
        blue_marker = _extract_marker(work, BLUE)
        assert red_marker[0] != blue_marker[0] and red_marker[1] != blue_marker[1]
        expected = _apply_workspace_rule(work, red_size, blue_size, red_marker, blue_marker)
        assert np.array_equal(out[:, :sep], inp[:, :sep])
        assert np.array_equal(out[:, sep], inp[:, sep])
        assert np.array_equal(out[:, sep + 1 :], expected)
        assert int((expected == YELLOW).sum()) >= 1
        assert _stamp_count(red_marker[1], red_size, work.shape[1]) >= 3
        assert _stamp_count(blue_marker[0], blue_size, work.shape[0]) >= 3
        assert not np.array_equal(inp, out)
    except Exception:
        return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeat the legend squares periodically from the two workspace markers and recolor their orthogonal overlaps yellow.

    Input structure:
    - A single solid grey column splits the grid into a left legend panel and a right black workspace.
    - The legend panel contains one solid red square and one solid blue square, each of size 2x2 or 3x3.
    - The workspace contains exactly one red marker pixel and one blue marker pixel, and they are in different rows and columns.

    Rule:
    - Read the red and blue square sizes from the legend by color matching.
    - Center the red square on the red marker and repeat it left and right along that marker's row with center spacing equal to the square size plus one, clipping at the workspace boundary.
    - Then center the blue square on the blue marker and repeat it up and down along that marker's column with the same size-plus-one spacing for blue.
    - Draw blue after red; whenever a blue stamp lands on an already red cell, recolor that overlap cell yellow instead of blue. Keep the legend panel and grey separator unchanged.

    Key insight:
    - The left panel is a color legend for stamp size, while the two workspace markers choose orthogonal periodic stamp sequences whose crossings create the yellow cells.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    sep = _find_separator_col(g)
    assert 0 < sep < g.shape[1] - 1
    legend = g[:, :sep]
    work = g[:, sep + 1 :]

    red_size = _extract_legend_square(legend, RED)
    blue_size = _extract_legend_square(legend, BLUE)
    assert set(np.unique(work)).issubset({BLACK, BLUE, RED})

    red_marker = _extract_marker(work, RED)
    blue_marker = _extract_marker(work, BLUE)
    assert red_marker[0] != blue_marker[0]
    assert red_marker[1] != blue_marker[1]

    out = np.array(g, copy=True)
    out[:, sep + 1 :] = _apply_workspace_rule(work, red_size, blue_size, red_marker, blue_marker)
    return out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        red_size = random.choice((2, 3))
        blue_size = random.choice((2, 3))
        max_size = max(red_size, blue_size)

        work_h = random.randint(max(12, 4 * max_size), 22)
        work_w = random.randint(max(12, 4 * max_size), 20)
        panel_w = random.choice([w for w in (3, 4) if w >= max_size])

        markers = _sample_markers(work_h, work_w, red_size, blue_size)
        if markers is None:
            continue
        red_marker, blue_marker = markers

        inp = _build_input(
            work_h,
            work_w,
            panel_w,
            red_size,
            blue_size,
            red_marker,
            blue_marker,
        )
        out = main(inp)
        if _semantic_checks(inp, out):
            return inp
    raise RuntimeError("failed to generate a valid input for puzzle 7")
