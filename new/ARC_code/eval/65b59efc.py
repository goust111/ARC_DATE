from __future__ import annotations
from collections import Counter
import numpy as np
import random
from arc_utils import paste, random_sprite


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a stamp map: read templates, markers, and palette from a tiled grid, then render stamps.

    Input structure:
    - Grid divided into K tile columns (each N wide) separated by 1-col dashed separators.
    - Top band (rows 0..N-1): K template masks, one per tile column.
    - Middle band (rows N+1..2N): marker pixels whose color picks a template.
    - Bottom band (rows 2N+2..): K palette swatches giving the output color per template.
    - Separator rows at rows N and 2N+1 (dashed pattern of sep color).

    Rule:
    - Each marker's local position (lr, lc) within its tile column determines which N×N output block to stamp.
    - The marker's color identifies which template to use; the palette recolors the template.

    Key insight: Marker color encodes template index; local position within tile column encodes stamp block.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(W) if (g[:, c] == sep).sum() >= H // 2]
    sep_rows = [r for r in range(H) if (g[r, :] == sep).sum() >= W // 2]
    assert sep_cols and sep_rows

    N = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    K = len(starts)

    # Read K templates from the top band
    tcols = []
    tmasks = []
    for st in starts:
        sub = g[0:N, st:st + N]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == K
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    # Read palette colors from the bottom band
    pal_r0 = sep_rows[-1] + 1
    pcols = []
    for st in starts:
        sub = g[pal_r0:H, st:st + N]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    # Read markers from the middle band and build stamp placements
    mid_r0 = N + 1
    out = np.full((N * N, N * N), bg, dtype=int)
    slot_owner = {}

    for st in starts:
        tile_color = None
        for lr in range(N):
            for lc in range(N):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = tcol_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates must occupy different slots"
                slot_owner[(lr, lc)] = j
                tile = np.where(tmasks[j], pcols[j], bg)
                out[lr * N:(lr + 1) * N, lc * N:(lc + 1) * N] = tile

    return out

# --------------- generator (att2) ---------------

# -- CFG: colors --
_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]  # all non-bg, non-sep

# -- CFG: grid dimensions --
_N_RANGE = (3, 5)          # tile size
_K_RANGE = (3, 4)          # number of templates
_PAL_H_CHOICES = [2, 3, 4]       # palette band height options
_PAL_H_WEIGHTS = [8, 1, 1]       # probability weights for palette height
_MAX_ROW_PAD = 2           # max trailing bg row padding
_MAX_COL_PAD = 7           # max trailing bg col padding

# -- CFG: template generation --
_DENSITY_RANGE = (0.55, 0.92)    # template pixel density
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

# -- CFG: marker placement --
_MIN_SLOT_COVERAGE = 0.60  # minimum fraction of NxN slots filled with markers

# -- CFG: palette swatch --
_SWATCH_DOUBLE_PROB = 0.35  # probability of 2-pixel swatch (vs 1-pixel)

# -- CFG: retry --
_MAX_RETRIES = 300


def _dash_cell(pos: int, n: int) -> bool:
    """True if position `pos` gets a separator pip in the dashed pattern for tile size `n`."""
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    """Count how many separator pips appear in a span of `length` cells."""
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    """Map (tile column index, local column) to absolute grid column."""
    return tile_col * (n + 1) + local_c


def _sample_template(n: int, color: int) -> np.ndarray:
    """Generate a random NxN boolean template mask via random_sprite."""
    sprite = random_sprite(
        n, n,
        density=random.uniform(*_DENSITY_RANGE),
        symmetry=random.choice(_SYMMETRY_OPTIONS),
        color_palette=[color],
        connectivity=_TEMPLATE_CONNECTIVITY,
        background=_BG,
    )
    return sprite == color


def _build_expected(n: int, templates: list[np.ndarray], pcols: list[int],
                    placements: list[tuple[int, int, int]]) -> np.ndarray:
    """Render the expected output grid from stamp placements."""
    expected = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], _BG)
        paste(expected, tile, rr * n, cc * n)
    return expected


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    """Draw dashed horizontal and vertical separator lines in-place."""
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = random.sample(_AVAILABLE_COLORS, k)

    # Grid dimensions with optional bg padding
    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = [_sample_template(n, col) for col in tcols]

    # Assign random slots to templates for marker placement
    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]
    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k))[:mark_count]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(n, templates, pcols, placements)

    # Assemble input grid
    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    # Top band: templates
    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    # Middle band: markers (shuffled tile-column assignment)
    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    # Bottom band: palette swatches
    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0:c0 + size] = col

    # Oracle check
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
    raise RuntimeError("failed to generate a valid 65b59efc input")
