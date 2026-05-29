# Combo variant of 65b59efc (index 33)
# combo_spec: D3/1+D6/1+D15/1
# Name: 密集对角OR叠加
from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solvers.arc_utils import paste, random_sprite


# Design analysis for D3/1 + D6/1 + D15/1:
# - The original N x N marker band cannot encode all N*N slots with visually diagonal 2-pixel markers:
#   for N=3 there are only 8 diagonal adjacencies but 9 output slots. The fused variant therefore enlarges
#   the middle band to an (N+1) x (N+1) marker lattice. Slot (r, c) is encoded by the pair at
#   [(r, c), (r+1, c+1)], which gives exactly N*N distinct diagonal markers.
# - Dense placement would normally make same-color pairs merge into ambiguous blobs. The solver now assumes
#   each tile column uses a single row-mod-3 class (all encoded slots have r % 3 fixed). That stronger
#   spacing keeps intended pairs three diagonal steps apart on each marker chain, so scanning for
#   zone[r, c] == zone[r+1, c+1] recovers the intended slots without false positives.
# - Pixelwise OR on colored stamps is ambiguous if two templates want different colors on the same cell.
#   The generator avoids that clash by only overlapping templates whose binary masks are disjoint. The
#   solver can then merge by taking the union of non-background pixels inside each output block.
# - Full D15 coverage is implemented by three base templates: one stamps each row-mod-3 class. A fourth
#   template reuses exactly one of those classes, which guarantees overlap blocks exist while preserving
#   decodability and keeping the middle band visually dense.
# - New implicit assumptions checked in main(): each marker tile has one non-separator color, every decoded
#   tile has at least one pair, all decoded slots in that tile share the same row-mod-3 class, and the
#   colored cells count equals twice the number of decoded pairs. generate_input() constructs inputs to
#   satisfy all of these conditions.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a dense stamp map with diagonal marker pairs, then OR-merge overlapping stamps.

    Input structure:
    - The grid is split into K tile columns separated by 1-column dashed separators.
    - Top band (rows 0..N-1): K template masks, each drawn inside an (N+1)-wide tile column.
    - Middle band (rows N+1..2N+1): an (N+1)x(N+1) marker lattice per tile column.
      Every marker is a same-color down-right diagonal pair.
    - Bottom band (rows 2N+3..): K palette swatches aligned with the template columns.
    - Separator rows appear at rows N and 2N+2.

    Rule:
    - In a marker lattice, the pair at local cells (r, c) and (r+1, c+1) targets output block (r, c).
    - The marker color identifies which template column to use; the palette recolors that template.
    - Every output block receives at least one stamp, and some blocks receive multiple stamps.
    - When several stamps hit the same block, combine them by pixelwise OR on foreground occupancy:
      any foreground cell contributed by any stamp stays foreground in the merged block.

    Key insight: The middle band is no longer a sparse set of single pixels. It is a dense lattice of
    same-color diagonal pairs, so you must read pair positions instead of isolated cells and then union
    compatible template masks whenever multiple pairs point to the same output block.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if (g[:, c] == sep).sum() >= h // 2]
    sep_rows = [r for r in range(h) if (g[r, :] == sep).sum() >= w // 2]
    assert sep_cols and len(sep_rows) >= 2

    tile_span = sep_cols[0]
    n = tile_span - 1
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)
    assert sep_rows[0] == n and sep_rows[-1] == 2 * n + 2

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st:st + tile_span]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub[:, :n] == col)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st:st + tile_span]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    mid_r0 = n + 1
    out = np.full((n * n, n * n), bg, dtype=int)
    seen_slots: set[tuple[int, int]] = set()

    for st in starts:
        zone = g[mid_r0:mid_r0 + n + 1, st:st + tile_span]
        vals = {int(v) for v in zone.flat} - {bg, sep}
        assert len(vals) == 1
        tile_color = vals.pop()
        j = tcol_to_j[tile_color]

        slots: list[tuple[int, int]] = []
        for rr in range(n):
            for cc in range(n):
                if int(zone[rr, cc]) == tile_color and int(zone[rr + 1, cc + 1]) == tile_color:
                    slots.append((rr, cc))

        assert slots
        row_class = slots[0][0] % 3
        assert all(rr % 3 == row_class for rr, _ in slots)
        assert int((zone == tile_color).sum()) == 2 * len(slots)

        tile = np.where(tmasks[j], pcols[j], bg)
        for rr, cc in slots:
            seen_slots.add((rr, cc))
            block = out[rr * n:(rr + 1) * n, cc * n:(cc + 1) * n]
            active = tmasks[j]
            clash = active & (block != bg) & (block != pcols[j])
            assert not clash.any()
            block[active & (block == bg)] = pcols[j]

    assert len(seen_slots) == n * n
    return out


# --------------- generator ---------------

# -- CFG: colors --
_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

# -- CFG: layout --
_N_RANGE = (3, 5)
_K_CHOICES = [4]
_PAL_H_CHOICES = [2, 3]

# -- CFG: templates --
_DENSITY_RANGE = (0.18, 0.42)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_MIN_TEMPLATE_PIXELS = 2

# -- CFG: retries --
_TEMPLATE_RETRIES = 120
_MAX_RETRIES = 400


def _dash_cell(pos: int, tile_span: int) -> bool:
    period = tile_span + 1
    k = pos % period
    return k == tile_span or k % 2 == 0


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    tile_span = n + 1
    for r in (n, 2 * n + 2):
        for c in range(core_w):
            if _dash_cell(c, tile_span):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = tile_span + i * (tile_span + 1)
        for r in range(core_h):
            if _dash_cell(r, tile_span):
                grid[r, c] = _SEP


def _sample_template(n: int, color: int, avoid: np.ndarray | None = None) -> np.ndarray | None:
    for _ in range(_TEMPLATE_RETRIES):
        try:
            sprite = random_sprite(
                n,
                n,
                density=random.uniform(*_DENSITY_RANGE),
                symmetry=random.choice(_SYMMETRY_OPTIONS),
                color_palette=[color],
                connectivity=_TEMPLATE_CONNECTIVITY,
                background=_BG,
            )
        except Exception:
            continue
        mask = sprite == color
        if int(mask.sum()) < _MIN_TEMPLATE_PIXELS:
            continue
        if avoid is not None and np.any(mask & avoid):
            continue
        return mask
    return None


def _sample_templates(n: int, tcols: list[int], extra_classes: list[int]) -> list[np.ndarray] | None:
    templates: list[np.ndarray] = []
    base0 = _sample_template(n, tcols[0])
    if base0 is None:
        return None
    templates.append(base0)

    base1 = _sample_template(n, tcols[1])
    if base1 is None:
        return None
    templates.append(base1)

    base2 = _sample_template(n, tcols[2])
    if base2 is None:
        return None
    templates.append(base2)

    for j in range(3, len(tcols)):
        avoid = templates[extra_classes[j - 3]]
        mask = _sample_template(n, tcols[j], avoid=avoid)
        if mask is None:
            return None
        templates.append(mask)
    return templates


def _eligible_slots(n: int, row_class: int) -> list[tuple[int, int]]:
    return [(rr, cc) for rr in range(n) for cc in range(n) if rr % 3 == row_class]


def _choose_extra_subset(slots: list[tuple[int, int]]) -> list[tuple[int, int]]:
    assert slots
    if len(slots) == 1:
        return slots[:]
    # Keep OR-overlap visible in the output: whenever a row-class offers multiple
    # slots, reuse at least two of them instead of letting the overlap collapse to
    # a single merged block.
    lo = max(2, len(slots) // 3)
    hi = max(lo, len(slots) - 1)
    count = random.randint(lo, hi)
    return random.sample(slots, count)


def _build_template_slots(n: int, k: int) -> tuple[list[list[tuple[int, int]]], list[int]]:
    template_slots: list[list[tuple[int, int]]] = [
        _eligible_slots(n, 0),
        _eligible_slots(n, 1),
        _eligible_slots(n, 2),
    ]
    extra_classes = [random.choice([0, 1, 2])]
    for row_class in extra_classes:
        template_slots.append(_choose_extra_subset(_eligible_slots(n, row_class)))
    assert len(template_slots) == k
    return template_slots, extra_classes


def _build_expected(n: int, templates: list[np.ndarray], pcols: list[int],
                    template_slots: list[list[tuple[int, int]]]) -> np.ndarray:
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, slots in enumerate(template_slots):
        active = templates[j]
        for rr, cc in slots:
            block = expected[rr * n:(rr + 1) * n, cc * n:(cc + 1) * n]
            clash = active & (block != _BG) & (block != pcols[j])
            assert not clash.any()
            block[active & (block == _BG)] = pcols[j]
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.choice(_K_CHOICES)
    colors = random.sample(_AVAILABLE_COLORS, 2 * k)
    tcols = colors[:k]
    pcols = colors[k:]

    template_slots, extra_classes = _build_template_slots(n, k)
    templates = _sample_templates(n, tcols, extra_classes)
    if templates is None:
        return None

    expected = _build_expected(n, templates, pcols, template_slots)

    pal_h = random.choice(_PAL_H_CHOICES)
    tile_span = n + 1
    core_h = 2 * n + 3 + pal_h
    core_w = k * tile_span + (k - 1)
    grid = np.full((core_h, core_w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        st = j * (tile_span + 1)
        paste(grid, np.where(mask, tcols[j], _BG), 0, st)

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    mid_r0 = n + 1
    for j, slots in enumerate(template_slots):
        st = tile_assign[j] * (tile_span + 1)
        for rr, cc in slots:
            grid[mid_r0 + rr, st + cc] = tcols[j]
            grid[mid_r0 + rr + 1, st + cc + 1] = tcols[j]

    swatch_r = core_h - 1
    for j, col in enumerate(pcols):
        st = j * (tile_span + 1)
        size = random.choice([1, 2])
        c0 = st + max(0, (tile_span - size) // 2)
        grid[swatch_r, c0:c0 + size] = col

    overlap_slots = sum(
        1
        for rr in range(n)
        for cc in range(n)
        if sum((rr, cc) in slots for slots in template_slots) >= 2
    )
    assert overlap_slots >= 2

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo 65b59efc input")
