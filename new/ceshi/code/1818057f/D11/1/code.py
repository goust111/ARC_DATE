# Variant of 1818057f
# Dimension: D11 — 前景像素的连通性利用方式 (meta)
# Change: 不直接使用连通分量分析，仅在每个像素的局部4-邻域上做判断... → 每个连通分量独立着不同颜色

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import get_bg, label_mask


# Design analysis for the D11 variant
# a) Concrete behavior trace:
#    The input still has one foreground color on a dominant background, and a '+' is
#    still defined locally by a center with four orthogonal foreground neighbors.
#    The change happens after detection: '+' marks now inherit a color from the
#    connected foreground component that contains them. Example: if the top component
#    contains two '+' centers and the bottom component contains one, the old rule
#    recolors all three motifs teal, while the new rule recolors both top motifs with
#    the first palette color and the bottom motif with the second palette color.
# b) Complexity calibration:
#    To make component membership visible, every generated grid contains 2 or 3 large
#    disconnected foreground components, and each component contains at least one '+';
#    at least one component contains multiple '+' motifs so "same component -> same
#    color" is supported by repeated evidence inside a single sample.
# c) Hidden assumptions from the original solver:
#    The original `fg = colors[0]` assumes a single non-background color, which still
#    holds here. The original unconditional `out[...] = 8` is no longer safe because
#    different components need different colors, so the solver must first label the
#    foreground components and map each detected '+' to its host component before
#    recoloring. We also avoid relying on library label ids directly by sorting the
#    plus-containing components by their visible top-left anchor.
# d) Placement legality:
#    The generator places each component inside its own horizontal band with at least
#    one background row between bands. '+' centers stay at least one cell away from
#    their band border, centers on the same component are spaced by >= 4 columns, and
#    extra branches only extend outward from existing arms. This prevents overlap,
#    out-of-bounds placement, and accidental merging between components.
# e) Visual observability:
#    Component order is tied to a visible scan order: top band first, then the next
#    band below it. The output palette is fixed and short, so the user can infer that
#    color depends on which disconnected blob owns the '+', not on hidden ids.
# f) Visual cleanliness:
#    Using separated horizontal bands keeps the scene readable. Each band contains a
#    single connected "branchy line" object, so recolored '+' footprints remain easy
#    to attribute to one blob without heavy overlap or clutter.
# g) Color-change inference:
#    Only three recolor colors are used: teal, green, pink. They are assigned in
#    top-to-bottom component order among components that actually contain '+'. This
#    gives repeated, low-entropy color behavior across samples.
# h) Generator-side degeneration check:
#    After drawing the foreground, the generator recomputes all '+' centers and
#    rejects the sample unless they match the planned centers exactly. This blocks
#    accidental extra '+' motifs created by branches or connectors.
# i) New solver assumptions and how the generator satisfies them:
#    The solver assumes every recolored '+' belongs to exactly one 4-connected
#    foreground component and that the number of plus-containing components does not
#    exceed the palette length. The generator enforces 2-3 separated components, each
#    in its own band, so those assumptions always hold.

PLUS_DELTAS = ((0, 0), (1, 0), (-1, 0), (0, 1), (0, -1))
PALETTE = (8, 3, 6)
BG = 2
FG = 4


def _find_plus_centers(mask: np.ndarray) -> list[tuple[int, int]]:
    centers: list[tuple[int, int]] = []
    H, W = mask.shape
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            if not bool(mask[r, c]):
                continue
            if bool(mask[r - 1, c]) and bool(mask[r + 1, c]) and bool(mask[r, c - 1]) and bool(mask[r, c + 1]):
                centers.append((int(r), int(c)))
    return centers


def _anchor_key(coords: np.ndarray) -> tuple[int, int]:
    top = coords[np.lexsort((coords[:, 1], coords[:, 0]))][0]
    return int(top[0]), int(top[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect '+' motifs in disconnected foreground blobs and recolor each blob's motifs with its own color.

    Input structure:
    - bg is the most frequent color.
    - A single foreground color forms several separate 4-connected branchy components.
    - Some components contain one or more embedded thin '+' motifs.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - Label the foreground into 4-connected components and keep only the components that contain at least one '+' center.
    - Order those components by the position of their topmost-leftmost pixel and assign colors teal, green, then pink.
    - Recolor each '+' center together with its 4 orthogonal neighbors to that component color; leave all other pixels unchanged.

    Key insight:
    The local '+' test is unchanged, but the recolor is no longer global: motifs inherit their color from the disconnected foreground component that owns them.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    fg_mask = g == fg
    centers = _find_plus_centers(fg_mask)
    if not centers:
        return out

    comp_lab, comp_n = label_mask(fg_mask, connectivity=4)
    comp_coords = {
        cid: np.argwhere(comp_lab == cid)
        for cid in range(1, int(comp_n) + 1)
    }
    plus_comp_ids = sorted({int(comp_lab[r, c]) for r, c in centers})
    ordered_comp_ids = sorted(plus_comp_ids, key=lambda cid: _anchor_key(comp_coords[int(cid)]))
    comp_color = {
        int(cid): int(PALETTE[idx % len(PALETTE)])
        for idx, cid in enumerate(ordered_comp_ids)
    }

    for r, c in centers:
        color = int(comp_color[int(comp_lab[r, c])])
        for dr, dc in PLUS_DELTAS:
            out[r + dr, c + dc] = color
    return out


def _pick_spaced_columns(rng: np.random.Generator, low: int, high: int, count: int) -> list[int] | None:
    values = np.arange(low, high + 1, dtype=int)
    for _ in range(80):
        chosen = sorted(map(int, rng.choice(values, size=count, replace=False)))
        if all(chosen[i + 1] - chosen[i] >= 4 for i in range(len(chosen) - 1)):
            return chosen
    return None


def _extension_capacity(r: int, c: int, dr: int, dc: int, band: tuple[int, int, int, int]) -> int:
    r0, r1, c0, c1 = band
    steps = 0
    nr = int(r + 2 * dr)
    nc = int(c + 2 * dc)
    while r0 <= nr <= r1 and c0 <= nc <= c1:
        steps += 1
        nr += dr
        nc += dc
    return int(steps)


def _add_plus(cells: set[tuple[int, int]], r: int, c: int) -> None:
    for dr, dc in PLUS_DELTAS:
        cells.add((int(r + dr), int(c + dc)))


def _build_component(
    rng: np.random.Generator,
    band: tuple[int, int, int, int],
    min_centers: int,
) -> tuple[set[tuple[int, int]], list[tuple[int, int]]] | None:
    r0, r1, c0, c1 = band
    if r1 - r0 + 1 < 5 or c1 - c0 + 1 < 5:
        return None

    center_row = int(rng.integers(r0 + 2, r1 - 1))
    low_col = int(c0 + 2)
    high_col = int(c1 - 2)
    if high_col < low_col:
        return None

    max_centers = 1 + (high_col - low_col) // 4
    max_centers = int(min(3, max_centers))
    if max_centers < min_centers:
        return None

    count = int(rng.integers(min_centers, max_centers + 1))
    cols = _pick_spaced_columns(rng, low_col, high_col, count)
    if cols is None:
        return None

    centers = [(int(center_row), int(col)) for col in cols]
    cells: set[tuple[int, int]] = set()
    for _r, _c in centers:
        _add_plus(cells, _r, _c)

    for idx in range(len(cols) - 1):
        for cc in range(int(cols[idx] + 1), int(cols[idx + 1])):
            cells.add((int(center_row), int(cc)))

    for idx, (_r, _c) in enumerate(centers):
        candidates: list[tuple[int, int, int]] = []
        for dr, dc in ((-1, 0), (1, 0)):
            cap = _extension_capacity(_r, _c, dr, dc, band)
            if cap > 0:
                candidates.append((dr, dc, cap))
        if idx == 0:
            cap = _extension_capacity(_r, _c, 0, -1, band)
            if cap > 0:
                candidates.append((0, -1, cap))
        if idx == len(centers) - 1:
            cap = _extension_capacity(_r, _c, 0, 1, band)
            if cap > 0:
                candidates.append((0, 1, cap))
        if not candidates:
            continue

        if len(centers) == 1 and len(candidates) >= 2:
            pick_count = 2
        else:
            pick_count = int(rng.integers(1, min(2, len(candidates)) + 1))

        chosen_idx = np.atleast_1d(rng.choice(len(candidates), size=pick_count, replace=False))
        for raw_idx in chosen_idx:
            dr, dc, cap = candidates[int(raw_idx)]
            length = int(rng.integers(1, min(3, cap) + 1))
            for step in range(2, 2 + length):
                cells.add((int(_r + dr * step), int(_c + dc * step)))

    return cells, centers


def _band_layout(H: int, K: int) -> list[tuple[int, int]]:
    usable = int(H - 2 - (K - 1))
    base = usable // K
    extra = usable % K
    rows: list[tuple[int, int]] = []
    start = 1
    for idx in range(K):
        height = int(base + (1 if idx < extra else 0))
        end = int(start + height - 1)
        rows.append((int(start), int(end)))
        start = int(end + 2)
    return rows


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(16, 23))
    W = int(rng.integers(14, 23))

    max_components = 3 if H >= 19 else 2
    K = int(rng.integers(2, max_components + 1))
    band_rows = _band_layout(H, K)
    grid = np.full((H, W), BG, dtype=int)

    all_centers: list[tuple[int, int]] = []
    component_count_with_multiple = 0
    for idx, (r0, r1) in enumerate(band_rows):
        band = (int(r0), int(r1), 1, int(W - 2))
        min_centers = 2 if idx == 0 else 1
        built = _build_component(rng, band, min_centers=min_centers)
        if built is None:
            return None
        cells, centers = built
        if len(centers) >= 2:
            component_count_with_multiple += 1
        for r, c in cells:
            grid[int(r), int(c)] = FG
        all_centers.extend(centers)

    if component_count_with_multiple == 0:
        return None

    fg_mask = grid == FG
    actual_centers = _find_plus_centers(fg_mask)
    if sorted(actual_centers) != sorted(all_centers):
        return None

    comp_lab, comp_n = label_mask(fg_mask, connectivity=4)
    plus_component_ids = {int(comp_lab[r, c]) for r, c in all_centers}
    if int(comp_n) != K or len(plus_component_ids) != K:
        return None

    total = int(H * W)
    fg_total = int(np.sum(fg_mask))
    if fg_total >= total // 2:
        return None

    out = main(grid)
    recolor_values = sorted(set(map(int, out.flatten())) - {BG, FG})
    if len(recolor_values) != K:
        return None
    if not np.array_equal(out, main(np.array(grid, dtype=int))):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError("generate_input failed after 300 attempts")
