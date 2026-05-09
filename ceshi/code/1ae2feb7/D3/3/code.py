# Variant of 1ae2feb7
# Dimension: D3 — 射线源对象类型 (src)
# Change: 水平单色色条——分隔条一侧的水平连续同色像素段，每行1-2个色条，紧邻分隔条排列... → 离散像素组

from __future__ import annotations

"""
Design analysis for the D3 variant
==================================

a) Concrete behavior and trace
The original task extracts horizontal monochrome runs on the bar side. Each run
stays on one row, and its width is used as the periodic step for painting dots
on the opposite side. The variant keeps the separator column, bar side /
extension side split, nearest-first priority, and no-overwrite rule, but
changes the source object itself:

- Each source is now one monochrome 8-connected component shaped like a sparse
  diagonal chain across a 2-row band, e.g. {(0,0), (1,1), (0,2)}.
- The component is visually discrete: pixels only touch diagonally, so it is
  clearly not a horizontal solid bar.
- The step is no longer a row-run width. It is the total pixel count of the
  component.
- The component paints on every row it occupies, using the same step on each of
  those rows.

Small trace on the left-bar-side case:
- Original row source: "6666|........" gives step 4, so output dots are at
  columns 0,4,8,... on the extension side of that row.
- Variant source component (rows r,r+1): top row has cells at columns [0,2],
  bottom row has [1]. This is one 8-connected component of size 3.
- The correct variant output paints both occupied rows with period 3. A solver
  that still measures per-row runs would incorrectly see only length-1 runs and
  would paint every column instead of every third column.

b) Complexity calibration
Using 2-row diagonal chains keeps the object count small and the period values
in a narrow, human-countable range (3-5 pixels per source). This avoids noisy
free-form blobs while still making the variant visibly different from the
original bar logic. Some bands contain two components to preserve the original
"nearer source claims cells first" interaction.

c) Hidden-assumption audit of the original main()
- `_iter_runs_1d(...)` assumes each relevant source is a contiguous row run.
  This fails immediately for sparse diagonal chains.
- `length = c1 - c0 + 1` assumes the source-defining measure is horizontal span.
  In the variant the period must be `component_pixel_count`, not bbox width.
- The original loop only paints the current row. That is correct for 1-row bars
  but wrong for multi-row components, so the variant must paint every occupied
  row of each component.
- Sorting by distance to separator still makes sense, but the distance must be
  derived from the nearest pixel of a component, not from a run endpoint.

d) Placement legality
Generator constraints keep every component inside the bar side, separated from
the next component in the same band by at least one blank column, and separated
from other 2-row bands by at least one blank row. Because the components are
8-connected diagonal chains, a 1-column horizontal gap is enough to prevent
accidental merging, and a 1-row vertical gap between bands prevents cross-band
touching.

e) Visual observability
Humans can see that the source is no longer a solid bar: each object is a short
zig-zag of isolated-looking pixels. The output period matches the total number
of source pixels, not any single row span. Since sizes stay in 3-5, counting is
practical from 2-4 demonstrations.

f) Visual cleanliness
The grid is partitioned the same way as the original: source objects live only
on one side of the separator and output dots appear only on the other side. The
objects are sparse but compact, and blank rows between bands keep the display
readable.

g) Color inference
No color remapping is introduced. Each source component keeps its own color, and
that same color is copied into the periodic dots. Colors are used only as
visible grouping cues, never as hidden ordering keys.

h) Quantitative anti-degeneration target
If the variant degenerates back into the original task, then the original solver
and this variant solver would often agree on generated inputs. Here the source
components are all size >= 3 and every occupied row contains only short
single-pixel runs, so the original run-based solver produces a different output
on the vast majority of samples. The post-edit verification script checks this.

i) New solver assumptions introduced by the variant
- All non-separator non-background cells on the source side belong to source
  components, and the generator guarantees that.
- Components are identified with 8-connectivity, matching the diagonal-chain
  construction.
- A component may occupy multiple rows; the generator restricts this to compact
  2-row bands so the rule stays legible.
"""

import numpy as np

from arc_utils import colors_present, get_bg, neighbors8
from arc_utils import get_bg as _bg_color


def _find_sep_col(g, bg):
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _extract_sources(
    g: np.ndarray,
    sep_col: int,
    src_on_left: bool,
    bg: int,
    sep_color: int,
) -> list[tuple[int, int, int, list[int], int, int]]:
    """Return source components as (dist, top_row, side_key, rows, size, color)."""
    h, w = g.shape
    if src_on_left:
        c_lo, c_hi = 0, sep_col
    else:
        c_lo, c_hi = sep_col + 1, w

    visited = np.zeros_like(g, dtype=bool)
    sources = []

    for r in range(h):
        for c in range(c_lo, c_hi):
            if visited[r, c]:
                continue
            color = int(g[r, c])
            if color == bg or color == sep_color:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            row_set: set[int] = set()
            min_dist = w
            near_col = -1 if src_on_left else w

            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                row_set.add(rr)

                dist = (sep_col - 1 - cc) if src_on_left else (cc - (sep_col + 1))
                min_dist = min(min_dist, dist)
                if src_on_left:
                    near_col = max(near_col, cc)
                else:
                    near_col = min(near_col, cc)

                for nr, nc in neighbors8(rr, cc):
                    if not (0 <= nr < h and c_lo <= nc < c_hi):
                        continue
                    if visited[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    stack.append((nr, nc))

            side_key = -near_col if src_on_left else near_col
            sources.append(
                (min_dist, min(row_set), side_key, sorted(row_set), len(cells), color)
            )

    sources.sort(key=lambda t: (t[0], t[1], t[2]))
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    bg = _bg_color(g)

    # Find the vertical separator column: a column whose non-bg cells are all the same color.
    # (May not span full height; some rows can be bg.)
    sep_col, sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    # Decide which side contains the sources (the other side is the painting target).
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    src_on_left = left_mass >= right_mass

    sources = _extract_sources(g, sep_col, src_on_left, bg, sep_color)

    # Process source components from closest to separator outward; do not overwrite.
    for _, _top_row, _side_key, rows, size, color in sources:
        step = size
        if step <= 0:
            continue
        for r in rows:
            if src_on_left:
                for cc in range(sep_col + 1, w, step):
                    if int(out[r, cc]) == bg:
                        out[r, cc] = color
            else:
                for cc in range(sep_col - 1, -1, -step):
                    if int(out[r, cc]) == bg:
                        out[r, cc] = color

    return out


# --------------- generator ---------------

# Canvas size bounds (inclusive ranges).
H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

# Source-side width bounds (clamped so ext_side stays in [EXT_SIDE_MIN, EXT_SIDE_MAX]).
SRC_SIDE_MIN, SRC_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 2, 14

# Palette sizes.
PALETTE_MIN, PALETTE_MAX = 5, 8

# Two-row source band counts.
BAND_MIN = 2

# Per-band structure.
P_TWO_SOURCES = 0.4
P_SEP_GAP = 0.15
P_SEP_COLUMN_GAP = 0.2

# Sparse source sizes.
MIN_SRC_SIZE, MAX_SRC_SIZE = 3, 5
MIN_INTER_GAP, MAX_INTER_GAP = 1, 2

# Retry budget for generation.
MAX_ATTEMPTS = 300


def _choose_band_starts(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` 2-row bands with at least one empty row between bands."""
    candidates = list(range(h - 1))
    for _ in range(200):
        starts = sorted(map(int, rng.choice(candidates, size=count, replace=False)))
        if all(b - a >= 3 for a, b in zip(starts, starts[1:])):
            return starts
    raise RuntimeError("failed to choose spaced source bands")


def _sample_palette(
    rng: np.random.Generator, needed_colors: int
) -> tuple[int, list[int]]:
    """Pick separator color and a disjoint source-color palette."""
    sep_color = int(rng.integers(1, 10))
    src_pool = [c for c in range(1, 10) if c != sep_color]
    size = max(needed_colors, PALETTE_MIN)
    size = min(size, min(PALETTE_MAX, len(src_pool)))
    if size < needed_colors:
        raise RuntimeError("not enough colors for requested number of sources")
    palette = list(map(int, rng.choice(src_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_band_sources(
    rng: np.random.Generator, src_side_len: int
) -> tuple[list[dict[str, int]], int]:
    """Return (sources, sep_gap) for one 2-row band, ordered near->far."""
    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = src_side_len - sep_gap
    if capacity < MIN_SRC_SIZE:
        raise RuntimeError("source side too narrow for sparse component")

    two_sources = capacity >= 2 * MIN_SRC_SIZE + MIN_INTER_GAP and rng.random() < P_TWO_SOURCES
    if not two_sources:
        size = int(rng.integers(MIN_SRC_SIZE, min(MAX_SRC_SIZE, capacity) + 1))
        return [{"size": size, "phase": int(rng.integers(0, 2)), "gap_after": 0}], sep_gap

    for _ in range(100):
        near_size = int(rng.integers(MIN_SRC_SIZE, min(MAX_SRC_SIZE, capacity - MIN_SRC_SIZE - MIN_INTER_GAP) + 1))
        remaining = capacity - near_size
        inter_gap = int(rng.integers(MIN_INTER_GAP, min(MAX_INTER_GAP, remaining - MIN_SRC_SIZE) + 1))
        far_capacity = remaining - inter_gap
        if far_capacity < MIN_SRC_SIZE:
            continue
        far_size = int(rng.integers(MIN_SRC_SIZE, min(MAX_SRC_SIZE, far_capacity) + 1))
        return [
            {"size": near_size, "phase": int(rng.integers(0, 2)), "gap_after": inter_gap},
            {"size": far_size, "phase": int(rng.integers(0, 2)), "gap_after": 0},
        ], sep_gap

    raise RuntimeError("failed to sample two sparse sources for a band")


def _chain_cells(size: int, phase: int) -> list[tuple[int, int]]:
    """Sparse 8-connected diagonal chain within a 2xsize box."""
    return [((phase + i) % 2, i) for i in range(size)]


def _place_chain(
    out: np.ndarray,
    band_top: int,
    start_col: int,
    color: int,
    size: int,
    phase: int,
) -> list[tuple[int, int]]:
    cells = []
    for dr, dc in _chain_cells(size, phase):
        rr = band_top + dr
        cc = start_col + dc
        out[rr, cc] = color
        cells.append((rr, cc))
    return cells


def _paint_extension(
    out: np.ndarray,
    rows: list[int],
    sep_col: int,
    src_on_left: bool,
    step: int,
    color: int,
) -> None:
    """Simulate the periodic extension onto the ext side (nearest-first, no overwrite)."""
    w = out.shape[1]
    for row in rows:
        if src_on_left:
            positions = range(sep_col + 1, w, step)
        else:
            positions = range(sep_col - 1, -1, -step)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    # src_side_len constrained so ext_side_len = w - 1 - src_side_len is in [EXT_SIDE_MIN, EXT_SIDE_MAX].
    lo = max(SRC_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(SRC_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    src_side_len = int(rng.integers(lo, hi + 1))

    src_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = src_side_len if src_on_left else (w - 1 - src_side_len)

    max_bands = min((h + 1) // 3, 4)
    if max_bands < BAND_MIN:
        return None
    band_count = int(rng.integers(BAND_MIN, max_bands + 1))
    try:
        band_starts = _choose_band_starts(rng, h, band_count)
    except RuntimeError:
        return None

    band_specs: list[tuple[int, list[dict[str, int]], int]] = []
    total_sources = 0
    for band_top in band_starts:
        try:
            sources, sep_gap = _sample_band_sources(rng, src_side_len)
        except RuntimeError:
            return None
        band_specs.append((band_top, sources, sep_gap))
        total_sources += len(sources)

    try:
        sep_color, palette = _sample_palette(rng, total_sources)
    except RuntimeError:
        return None

    out = np.zeros((h, w), dtype=int)
    out[:, sep_col] = sep_color

    occupied_rows = {band_top + dr for band_top in band_starts for dr in (0, 1)}
    inactive_rows = [r for r in range(h) if r not in occupied_rows]
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        out[gap_row, sep_col] = 0

    color_iter = iter(map(int, rng.permutation(palette)[:total_sources]))
    painted_sources: list[tuple[int, list[int], int]] = []

    for band_top, sources, sep_gap in band_specs:
        if src_on_left:
            cursor = sep_col - sep_gap
            for src in sources:
                size = int(src["size"])
                phase = int(src["phase"])
                color = next(color_iter)
                start_col = cursor - size
                cells = _place_chain(out, band_top, start_col, color, size, phase)
                rows = sorted({r for r, _ in cells})
                painted_sources.append((size, rows, color))
                cursor = start_col - int(src["gap_after"])
        else:
            cursor = sep_col + 1 + sep_gap
            for src in sources:
                size = int(src["size"])
                phase = int(src["phase"])
                color = next(color_iter)
                start_col = cursor
                cells = _place_chain(out, band_top, start_col, color, size, phase)
                rows = sorted({r for r, _ in cells})
                painted_sources.append((size, rows, color))
                cursor = start_col + size + int(src["gap_after"])

    for size, rows, color in painted_sources:
        _paint_extension(out, rows, sep_col, src_on_left, size, color)

    inp = out.copy()
    if src_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
