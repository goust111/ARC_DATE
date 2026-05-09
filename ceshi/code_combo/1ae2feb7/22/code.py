# Combo variant of 1ae2feb7 (index 22)
# combo_spec: D3/3+D7/2
# Name: 离散像素计数决定优先级

from __future__ import annotations

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and trace
The original task reads horizontal bars on one side of a vertical separator.
Each bar paints periodic dots on the opposite side, with bar length as the
period and near-to-separator bars processed first.

This combo changes both the source object and the processing order:
- A source is now a sparse 8-connected diagonal chain across a 2-row band.
  Its occupied cells are visually discrete single pixels, not a solid bar.
- The period is the total pixel count of the chain.
- Sources are processed by pixel count from large to small, not by distance.
- No-overwrite stays unchanged, so the first source to reach an ext-side cell
  keeps that cell.

Concrete left-side trace:
- Near source: size 3, color A, occupying the two rows of a band.
- Far source: size 5, color B, occupying the same two rows.
- Original task would see row runs of length 1 and would paint almost every
  ext-side cell, which is wrong here.
- D3-only behavior would count component sizes correctly but still process the
  near size-3 source first, so the first ext-side cell on both occupied rows
  would become color A.
- This combo processes size 5 before size 3, so that same first ext-side cell
  becomes color B. Later size-3 hits cannot overwrite it.

b) Complexity calibration
The sparse chains use only sizes 3-5, so humans can count them. Each puzzle
contains at least two 2-row bands with two sources each: one band where the
nearer source is larger, and one where the farther source is larger. This
prevents the ordering rule from collapsing into "always near first" or "always
far first" and gives repeated evidence for "count pixels, then compare sizes".

c) Hidden-assumption audit of the original main()
- `_iter_runs_1d(...)` in the original solver assumed every source was a
  contiguous 1-row bar. That fails because every occupied row here contains
  separated single pixels instead.
- `length = c1 - c0 + 1` assumed period equals horizontal run length. The combo
  needs connected-component pixel count instead.
- The original painter only updated the current row. Here one source occupies
  two rows, so both occupied rows must be painted using the same period.
- Distance-based sorting no longer matches the task. The solver must sort by
  component size first and use distance only as a deterministic tie-break.
- New solver assumption: sources are recoverable as 8-connected monochrome
  components on the source side. The generator guarantees this.

d) Placement legality
Every source stays fully inside the source side. Two-source bands leave at least
one blank column between chains so they cannot merge even diagonally. Different
bands are separated by at least one empty row, so components from neighboring
bands also cannot connect accidentally. Forced evidence bands use zero separator
gap to preserve capacity for two unequal sources.

e) Visual observability
The first ext-side cell next to the separator is the clearest conflict witness,
because every source tries to paint there. In a band where the farther source
has more pixels, that first ext-side cell visibly adopts the farther color,
which rules out the original near-first logic. In another band where the nearer
source has more pixels, the nearer color wins, ruling out a fixed far-first
guess. The sparse shape also makes it apparent that the relevant measure is
pixel count, not horizontal span.

f) Visual cleanliness
All source objects remain on one side of the separator, and all generated dots
remain on the other side. Components are compact 2-row zig-zags with blank rows
between bands, so the grid stays readable despite the added multi-row behavior.

g) Color inference
Colors are only labels that help group cells into visible sources and track
which source painted which dots. They do not determine order. The generator uses
distinct source colors to avoid any hidden same-color grouping ambiguity.

h) Quantitative anti-degeneration target
The generator rejects any sample whose size-descending output matches the
distance-ascending output on the same sparse sources. This guarantees the D7
part is active. The D3 part remains active because every occupied row contains
only isolated single-pixel runs, so the original run-length solver would not
match the component-count behavior.

i) New solver assumptions introduced by the combo
- All non-separator, non-background cells on the source side belong to sparse
  source components.
- Components are identified with 8-connectivity.
- Components may occupy two rows, and bands do not touch each other.
- Same-size conflicts are allowed only as deterministic ties; the generator
  avoids relying on them as visible evidence by forcing unequal sizes in the
  key two-source bands.
"""

import numpy as np

from arc_utils import colors_present, get_bg, neighbors8
from arc_utils import get_bg as _bg_color


Source = tuple[int, int, int, list[int], int, int]


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, _w = g.shape
    for c in range(g.shape[1]):
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
) -> list[Source]:
    """Return source components as (dist, top_row, side_key, rows, size, color)."""
    h, w = g.shape
    if src_on_left:
        c_lo, c_hi = 0, sep_col
    else:
        c_lo, c_hi = sep_col + 1, w

    visited = np.zeros_like(g, dtype=bool)
    sources: list[Source] = []

    for r in range(h):
        for c in range(c_lo, c_hi):
            if visited[r, c]:
                continue
            color = int(g[r, c])
            if color == bg or color == sep_color:
                continue

            stack = [(r, c)]
            visited[r, c] = True
            row_set: set[int] = set()
            size = 0
            min_dist = w
            near_col = -1 if src_on_left else w

            while stack:
                rr, cc = stack.pop()
                size += 1
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
            sources.append((min_dist, min(row_set), side_key, sorted(row_set), size, color))

    return sources


def _ordered_sources(sources: list[Source], order: str) -> list[Source]:
    if order == "size_desc":
        return sorted(sources, key=lambda t: (-t[4], t[0], t[1], t[2], t[5]))
    if order == "near_first":
        return sorted(sources, key=lambda t: (t[0], t[1], t[2], t[5]))
    raise ValueError(order)


def _paint_sources(
    out: np.ndarray,
    sources: list[Source],
    sep_col: int,
    src_on_left: bool,
    bg: int,
    order: str,
) -> None:
    _h, w = out.shape
    for _dist, _top_row, _side_key, rows, size, color in _ordered_sources(sources, order):
        if size <= 0:
            continue
        for r in rows:
            if src_on_left:
                positions = range(sep_col + 1, w, size)
            else:
                positions = range(sep_col - 1, -1, -size)
            for cc in positions:
                if int(out[r, cc]) == bg:
                    out[r, cc] = color


def _render_output(
    base: np.ndarray,
    sources: list[Source],
    sep_col: int,
    src_on_left: bool,
    bg: int,
    order: str,
) -> np.ndarray:
    out = base.copy()
    _paint_sources(out, sources, sep_col, src_on_left, bg, order)
    return out


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if row[c] == bg:
            c += 1
            continue
        color = int(row[c])
        c0 = c
        while c < c_hi and row[c] == color:
            c += 1
        yield c0, c - 1, color


def _baseline_original_output(
    g: np.ndarray,
    sep_col: int,
    src_on_left: bool,
    bg: int,
) -> np.ndarray:
    """Original 1ae2feb7 behavior: row runs, near-first, run length as period."""
    out = g.copy()
    h, w = g.shape
    bars: list[tuple[int, int, int, int, int, int]] = []
    for r in range(h):
        if src_on_left:
            for c0, c1, color in _iter_runs_1d(g[r, :], 0, sep_col, bg):
                bars.append(((sep_col - 1) - c1, r, c0, c1, c1 - c0 + 1, color))
        else:
            for c0, c1, color in _iter_runs_1d(g[r, :], sep_col + 1, w, bg):
                bars.append((c0 - (sep_col + 1), r, c0, c1, c1 - c0 + 1, color))
    bars.sort(key=lambda t: (t[0], t[1], t[2]))
    for _dist, r, _c0, _c1, step, color in bars:
        if src_on_left:
            positions = range(sep_col + 1, w, step)
        else:
            positions = range(sep_col - 1, -1, -step)
        for cc in positions:
            if int(out[r, cc]) == bg:
                out[r, cc] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Overview:
    Find sparse monochrome source components on one side of a vertical separator
    and extend each component's color periodically onto the opposite side.

    Input structure:
    - One mostly solid separator column of a single non-background color.
    - All source pixels lie on exactly one side of the separator.
    - Each source is a sparse 8-connected monochrome component, typically a
      short diagonal chain spanning two neighboring rows.
    - The opposite side is empty in the input.

    Rule:
    - Count the number of pixels in each source component.
    - Use that pixel count as the periodic step for painting dots on every row
      occupied by the component.
    - Process components from largest pixel count to smallest.
    - If two components try to paint the same target cell, keep the color from
      the component processed first; do not overwrite existing non-background
      cells.

    Key insight:
    The important quantity is not a row-run length but the total size of a
    sparse connected component, and that same size also determines priority.
    """
    g = np.asarray(grid, dtype=int)

    bg = _bg_color(g)
    sep_col, sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    src_on_left = left_mass >= right_mass

    sources = _extract_sources(g, sep_col, src_on_left, bg, int(sep_color))
    _paint_sources(out, sources, sep_col, src_on_left, bg, order="size_desc")
    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

SRC_SIDE_MIN, SRC_SIDE_MAX = 7, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 12

PALETTE_MIN, PALETTE_MAX = 5, 8

BAND_MIN, BAND_MAX = 2, 3
P_EXTRA_BAND = 0.45
P_TWO_SOURCES_EXTRA = 0.45
P_SEP_GAP = 0.12
P_SEP_COLUMN_GAP = 0.2

MIN_SRC_SIZE, MAX_SRC_SIZE = 3, 5
MIN_INTER_GAP, MAX_INTER_GAP = 1, 2

FORCED_RELATIONS = ("near_larger", "far_larger")
MAX_ATTEMPTS = 400


def _choose_band_starts(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` 2-row bands with at least one empty row between them."""
    candidates = list(range(h - 1))
    for _ in range(300):
        starts = sorted(map(int, rng.choice(candidates, size=count, replace=False)))
        if all(b - a >= 3 for a, b in zip(starts, starts[1:])):
            return starts
    raise RuntimeError("failed to choose spaced 2-row bands")


def _sample_palette(
    rng: np.random.Generator, needed_colors: int
) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    src_pool = [c for c in range(1, 10) if c != sep_color]
    size = max(needed_colors, PALETTE_MIN)
    size = min(size, min(PALETTE_MAX, len(src_pool)))
    if size < needed_colors:
        raise RuntimeError("not enough source colors")
    palette = list(map(int, rng.choice(src_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_two_source_layout(
    rng: np.random.Generator,
    capacity: int,
    relation: str | None,
) -> list[dict[str, int]] | None:
    layouts: list[tuple[int, int, int]] = []
    for near_size in range(MIN_SRC_SIZE, min(MAX_SRC_SIZE, capacity) + 1):
        for inter_gap in range(MIN_INTER_GAP, MAX_INTER_GAP + 1):
            for far_size in range(MIN_SRC_SIZE, min(MAX_SRC_SIZE, capacity) + 1):
                total = near_size + inter_gap + far_size
                if total > capacity or near_size == far_size:
                    continue
                if relation == "near_larger" and near_size <= far_size:
                    continue
                if relation == "far_larger" and far_size <= near_size:
                    continue
                layouts.append((near_size, inter_gap, far_size))
    if not layouts:
        return None
    near_size, inter_gap, far_size = layouts[int(rng.integers(0, len(layouts)))]
    return [
        {"size": near_size, "phase": int(rng.integers(0, 2)), "gap_after": inter_gap},
        {"size": far_size, "phase": int(rng.integers(0, 2)), "gap_after": 0},
    ]


def _sample_band_sources(
    rng: np.random.Generator,
    src_side_len: int,
    relation: str | None,
) -> tuple[list[dict[str, int]], int] | None:
    if relation is not None:
        sources = _sample_two_source_layout(rng, src_side_len, relation)
        if sources is None:
            return None
        return sources, 0

    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = src_side_len - sep_gap
    if capacity < MIN_SRC_SIZE:
        return None

    if capacity >= 2 * MIN_SRC_SIZE + MIN_INTER_GAP and rng.random() < P_TWO_SOURCES_EXTRA:
        sources = _sample_two_source_layout(rng, capacity, relation=None)
        if sources is not None:
            return sources, sep_gap

    size = int(rng.integers(MIN_SRC_SIZE, min(MAX_SRC_SIZE, capacity) + 1))
    return [{"size": size, "phase": int(rng.integers(0, 2)), "gap_after": 0}], sep_gap


def _chain_cells(size: int, phase: int) -> list[tuple[int, int]]:
    """Sparse 8-connected diagonal chain inside a 2 x size box."""
    return [((phase + i) % 2, i) for i in range(size)]


def _place_chain(
    out: np.ndarray,
    band_top: int,
    start_col: int,
    color: int,
    size: int,
    phase: int,
) -> None:
    for dr, dc in _chain_cells(size, phase):
        out[band_top + dr, start_col + dc] = color


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(SRC_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(SRC_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    src_side_len = int(rng.integers(lo, hi + 1))

    src_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = src_side_len if src_on_left else (w - 1 - src_side_len)

    max_bands = min(BAND_MAX, (h + 1) // 3)
    if max_bands < BAND_MIN:
        return None
    band_count = BAND_MIN
    if max_bands > BAND_MIN and rng.random() < P_EXTRA_BAND:
        band_count += 1
    try:
        band_starts = _choose_band_starts(rng, h, band_count)
    except RuntimeError:
        return None

    forced_relations = list(FORCED_RELATIONS)
    rng.shuffle(forced_relations)
    relation_map = {band_starts[i]: forced_relations[i] for i in range(len(FORCED_RELATIONS))}

    band_specs: list[tuple[int, list[dict[str, int]], int]] = []
    total_sources = 0
    for band_top in band_starts:
        sampled = _sample_band_sources(rng, src_side_len, relation=relation_map.get(band_top))
        if sampled is None:
            return None
        sources, sep_gap = sampled
        band_specs.append((band_top, sources, sep_gap))
        total_sources += len(sources)

    try:
        sep_color, palette = _sample_palette(rng, total_sources)
    except RuntimeError:
        return None

    base = np.zeros((h, w), dtype=int)
    base[:, sep_col] = sep_color

    occupied_rows = {band_top + dr for band_top in band_starts for dr in (0, 1)}
    inactive_rows = [r for r in range(h) if r not in occupied_rows]
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        base[gap_row, sep_col] = 0

    color_iter = iter(map(int, rng.permutation(palette)[:total_sources]))
    for band_top, sources, sep_gap in band_specs:
        if src_on_left:
            cursor = sep_col - sep_gap
            for src in sources:
                size = int(src["size"])
                start_col = cursor - size
                _place_chain(base, band_top, start_col, next(color_iter), size, int(src["phase"]))
                cursor = start_col - int(src["gap_after"])
        else:
            cursor = sep_col + 1 + sep_gap
            for src in sources:
                size = int(src["size"])
                start_col = cursor
                _place_chain(base, band_top, start_col, next(color_iter), size, int(src["phase"]))
                cursor = start_col + size + int(src["gap_after"])

    bg = 0
    sources = _extract_sources(base, sep_col, src_on_left, bg, sep_color)
    out_variant = _render_output(base, sources, sep_col, src_on_left, bg, order="size_desc")
    out_distance = _render_output(base, sources, sep_col, src_on_left, bg, order="near_first")

    inp = base.copy()
    if src_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.array_equal(out_variant, out_distance):
        return None
    if np.array_equal(out_variant, _baseline_original_output(inp, sep_col, src_on_left, bg)):
        return None
    if not np.array_equal(main(inp), out_variant):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
