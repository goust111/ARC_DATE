import random
from dataclasses import dataclass

import numpy as np


BLACK = 0
GREY = 5
TEAL = 8
BAR_COLORS = [1, 2, 3, 4, 6, 7, 9]
GEN_RETRIES = 400


@dataclass(frozen=True)
class Bar:
    color: int
    col: int
    top: int
    bottom: int
    height: int
    side: str
    distance: int
    final_center_row: int


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _template_offsets(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid == TEAL)
    assert coords.size > 0, "expected a teal stencil"
    r0, c0, r1, c1 = _bbox(coords)
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    return coords - np.array([center_r, center_c], dtype=int)


def _grey_line_row(grid: np.ndarray) -> int:
    rows = np.flatnonzero(np.all(grid == GREY, axis=1))
    assert rows.size == 1, "expected exactly one full grey row"
    return int(rows[0])


def _extract_bars(grid: np.ndarray, grey_row: int) -> list[Bar]:
    h, w = grid.shape
    bars: list[Bar] = []
    valid_colors = set(int(v) for v in np.unique(grid)) - {BLACK, GREY, TEAL}
    for c in range(w):
        rows = [r for r in range(h) if int(grid[r, c]) in valid_colors]
        if not rows:
            continue
        colors = {int(grid[r, c]) for r in rows}
        assert len(colors) == 1, "each bar column must contain a single color"
        assert rows == list(range(rows[0], rows[-1] + 1)), "bar must be contiguous"
        top = int(rows[0])
        bottom = int(rows[-1])
        color = next(iter(colors))
        height = bottom - top + 1
        assert height >= 1
        if bottom < grey_row:
            side = "above"
            final_bottom = grey_row - 1
            final_top = final_bottom - height + 1
            distance = final_top - top
        else:
            assert top > grey_row, "bars cannot cross the grey line"
            side = "below"
            final_top = grey_row + 1
            distance = top - final_top
        assert distance >= 1, "bars must start away from the grey line"
        final_center_row = final_top + (height - 1) // 2
        bars.append(
            Bar(
                color=color,
                col=c,
                top=top,
                bottom=bottom,
                height=height,
                side=side,
                distance=int(distance),
                final_center_row=int(final_center_row),
            )
        )
    assert bars, "expected at least one vertical bar"
    return bars


def _stamp_order(bar: Bar) -> tuple[int, int, int]:
    side_rank = 1 if bar.side == "above" else 0
    return (bar.distance, side_rank, -bar.col)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide colored vertical bars to the grey docking line, then stamp the teal template at each stop using movement-priority overlaps.

    Input structure:
    - A black background contains one full-width horizontal grey line, one teal connected stencil in a corner area, and several 1-column colored vertical bars above and below the line.
    - Each bar is a contiguous vertical segment in its own column and is separated from the grey line by at least one black row.

    Rule:
    - Treat the grey row as a collision boundary and the teal component as the stencil.
    - Slide bars above the line downward until their bottoms touch the row just above the grey line; slide bars below upward until their tops touch the row just below it.
    - For each slid bar, align the stencil bbox center to the bar's final center cell and recolor the stencil to the bar color.
    - If stamps overlap, the bar with longer slide distance overwrites the other; ties prefer bars that came from above, then smaller column index.
    - Erase the original grey line, teal stencil, and bar starting positions; keep only the stamped copies on black.

    Key insight:
    - The bars do not remain as the output objects: their slide destinations define where recolored copies of the teal template are placed, and travel distance determines overlap priority.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    grey_row = _grey_line_row(g)
    offsets = _template_offsets(g)
    bars = _extract_bars(g, grey_row)

    out = np.full_like(g, BLACK)
    h, w = out.shape
    for bar in sorted(bars, key=_stamp_order):
        rows = offsets[:, 0] + bar.final_center_row
        cols = offsets[:, 1] + bar.col
        keep = (0 <= rows) & (rows < h) & (0 <= cols) & (cols < w)
        out[rows[keep], cols[keep]] = bar.color
    return out


TEMPLATE_PATTERNS = [
    np.array(
        [
            [8, 0, 0],
            [8, 8, 8],
            [0, 8, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 0, 8],
            [8, 8, 8],
            [0, 8, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 8, 0],
            [8, 8, 8],
            [8, 0, 0],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 8, 0],
            [8, 8, 8],
            [0, 0, 8],
        ],
        dtype=int,
    ),
    np.array(
        [
            [0, 8, 0],
            [8, 8, 0],
            [0, 8, 8],
        ],
        dtype=int,
    ),
]


def _place_template(grid: np.ndarray, template: np.ndarray, corner: str, rng: random.Random) -> tuple[int, int]:
    h, w = grid.shape
    th, tw = template.shape
    if corner == "tl":
        r0 = rng.randint(0, 1)
        c0 = rng.randint(0, 1)
    elif corner == "tr":
        r0 = rng.randint(0, 1)
        c0 = rng.randint(max(0, w - tw - 1), w - tw)
    elif corner == "bl":
        r0 = rng.randint(max(0, h - th - 1), h - th)
        c0 = rng.randint(0, 1)
    else:
        r0 = rng.randint(max(0, h - th - 1), h - th)
        c0 = rng.randint(max(0, w - tw - 1), w - tw)
    mask = template == TEAL
    grid[r0 : r0 + th, c0 : c0 + tw][mask] = TEAL
    return r0, c0


def _bar_cluster_columns(
    w: int, template_col_range: tuple[int, int], count: int, rng: random.Random
) -> list[int]:
    t0, t1 = template_col_range
    candidates = [c for c in range(w) if not (t0 - 1 <= c <= t1 + 1)]
    assert len(candidates) >= count
    mode = rng.choices(["left", "center", "right"], weights=[1, 3, 1], k=1)[0]
    band_width = rng.randint(max(count, 3), min(max(count, 3) + 2, len(candidates)))
    if mode == "left":
        start = 0
    elif mode == "right":
        start = max(0, len(candidates) - band_width)
    else:
        start = rng.randint(0, len(candidates) - band_width)
    band = candidates[start : start + band_width]
    cols = sorted(rng.sample(band, count))
    return cols


def _distance_choices(max_distance: int, rng: random.Random) -> int:
    assert max_distance >= 1
    if max_distance == 1:
        return 1
    if rng.random() < 0.35:
        return max_distance
    return rng.randint(1, max_distance)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 21)
    w = rng.randint(14, 21)
    grey_row = rng.randint(5, h - 6)
    grid = np.full((h, w), BLACK, dtype=int)
    grid[grey_row, :] = GREY

    template = rng.choice(TEMPLATE_PATTERNS).copy()
    corner = rng.choice(["tl", "tr", "bl", "br"])
    template_r0, template_c0 = _place_template(grid, template, corner, rng)
    template_h, template_w = template.shape
    template_rows = (template_r0, template_r0 + template_h - 1)
    template_cols = (template_c0, template_c0 + template_w - 1)

    total_bars = rng.randint(2, min(5, len(BAR_COLORS)))
    above_count = rng.randint(1, total_bars - 1)
    below_count = total_bars - above_count
    cols = _bar_cluster_columns(w, template_cols, total_bars, rng)
    colors = rng.sample(BAR_COLORS, total_bars)

    if corner.startswith("t") and grey_row <= template_rows[1] + 2:
        return None
    if corner.startswith("b") and grey_row >= template_rows[0] - 2:
        return None

    bar_specs = []
    for idx, col in enumerate(cols):
        side = "above" if idx < above_count else "below"
        height = rng.choice([2, 3])
        if side == "above":
            final_top = grey_row - height
            max_distance = final_top
            if max_distance < 1:
                return None
            distance = _distance_choices(max_distance, rng)
            top = final_top - distance
            bottom = top + height - 1
        else:
            final_top = grey_row + 1
            max_distance = h - (final_top + height)
            if max_distance < 1:
                return None
            distance = _distance_choices(max_distance, rng)
            top = final_top + distance
            bottom = top + height - 1
        if top < 0 or bottom >= h:
            return None
        if template_cols[0] <= col <= template_cols[1] and not (bottom < template_rows[0] or top > template_rows[1]):
            return None
        bar_specs.append((colors[idx], col, top, bottom))

    for color, col, top, bottom in bar_specs:
        grid[top : bottom + 1, col] = color

    teal_pixels = int(np.count_nonzero(grid == TEAL))
    if teal_pixels < 3 or teal_pixels > 8:
        return None

    bars = _extract_bars(grid, grey_row)
    if sum(bar.side == "above" for bar in bars) == 0 or sum(bar.side == "below" for bar in bars) == 0:
        return None

    out = main(grid)
    if np.count_nonzero(out) < teal_pixels:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        rng = random.Random()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
