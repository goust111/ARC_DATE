# Combo variant of 65b59efc (index 27)
# combo_spec: D4/0+D7/2+D12/3
# Name: 空心万花筒网格

from __future__ import annotations

# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior trace
#    - Base task: a marker color chooses one top-band template, the marker's local slot
#      (lr, lc) chooses one output block, and the bottom swatch gives the repaint color.
#      The stamped tile is written unchanged into a plain (N*N) x (N*N) output board.
#    - Fused task: the same legend decoding is preserved, but the output board is now a
#      separator-colored lattice. Logical block (lr, lc) no longer starts at pixel
#      (lr*N, lc*N); it starts at (lr*(N+1), lc*(N+1)) because one separator row/col is
#      inserted after every block except the last.
#    - Example with N=4:
#        * A marker at local slot (1,2) still targets logical block (1,2).
#        * In the fused output that block occupies rows 5..8 and cols 10..13 because
#          separator lines sit at indices 4, 9, 14.
#        * The decoded template is a hollow border-only frame with one missing border
#          cell. Since block row 1 is odd, the frame is flipped vertically. Since block
#          col 2 is even, no horizontal flip is applied. If the slot were (1,1), both
#          flips would apply, moving the missing border cell to the diagonally opposite
#          side and creating a 180-degree turned appearance.
#    - Difference summary: D4 changes the output canvas geometry, D7 constrains the
#      templates to hollow border objects, and D12 changes where the frame's missing
#      border notch appears inside each output block.
#
# b) Complexity calibration
#    - A perfect gapless square ring would make D12 invisible because left/right and
#      up/down flips would produce the same picture. To avoid that degeneration while
#      keeping the intended "hollow frame" look, generation uses border-only frames with
#      exactly one missing border cell. The interior still stays completely empty.
#    - Each sample uses 3-4 templates and dense multi-block placement, and every template
#      is forced to appear in odd-row and odd-col blocks. The generator also forces at
#      least one even-even block and one odd-odd block overall, so all four parity
#      branches of the kaleidoscope rule are visible in every sample rather than only by
#      chance.
#    - N is restricted to 4 or 5. That keeps the hollow frame readable and gives enough
#      perimeter cells to place distinct notches without making the outputs messy.
#
# c) Hidden assumptions audit against the original solver
#    - `bg = most_common_color` still holds because the input layout is unchanged and the
#      generator keeps background dominant.
#    - `sep = most_common_non_bg_color` still holds because only the input separators use
#      the separator color and their dashed rows/cols remain denser than any single other
#      non-background color.
#    - `n = sep_cols[0]` still works because the first vertical separator is still placed
#      immediately after the first N-wide legend tile.
#    - The template-reading loop still assumes each top-band tile contains one template
#      color. The generator preserves that by drawing each hollow frame monochromatically.
#    - The palette-reading loop still assumes exactly one palette color per bottom tile.
#      The generator keeps one contiguous swatch per column.
#    - The slot-owner constraint remains keyed by logical block (lr, lc). D12 changes the
#      orientation of the stamped tile, not which logical block owns that slot.
#    - The original output write `out[lr*N:(lr+1)*N, lc*N:(lc+1)*N]` is no longer valid.
#      The solver must route every logical block through the enlarged lattice coordinates.
#
# d) Placement legality
#    - Input legality is unchanged structurally: templates live in rows 0..N-1, markers
#      live in rows N+1..2N, palette swatches live below row 2N+1, and separators only
#      occupy the dedicated dashed gaps.
#    - Output legality is enforced by construction: logical block (rr, cc) always pastes
#      at `(rr*(N+1), cc*(N+1))`, which keeps each N x N stamp fully inside its own cell
#      of the `(N*N+N-1) x (N*N+N-1)` lattice and never overwrites separator lines.
#    - Marker placements remain globally disjoint by logical slot, so different templates
#      never compete for the same output block.
#
# e) Visual observability
#    - The output lattice is obvious to the eye because separator-colored rows/cols repeat
#      at every block boundary.
#    - The hollow-frame restriction is visible both in the top legend and in the output:
#      every stamped object is an empty frame rather than a filled blob.
#    - The single-notch asymmetry makes the odd-row / odd-col flips readable from a few
#      examples: the notch hops from one side/corner of the frame to another in a regular
#      parity pattern.
#
# f) Visual cleanliness
#    - Only one separator style exists in the input (the original dashed legend grid) and
#      only one separator style exists in the output (full lattice lines). No extra glyphs
#      or overlay annotations are introduced.
#    - Hollow frames are thin border objects, so even dense placement yields a geometric
#      texture rather than large solid areas fighting the separator lattice.
#    - Palette swatches remain small centered runs in the bottom band, which keeps the
#      input readable despite the stronger output structure.
#
# g) New assumptions introduced by main(), and how generate_input() supports them
#    - main() assumes the output separator color can be reused from the input separator
#      color. Generation uses a fixed separator color consistently in both places.
#    - main() assumes parity-based flips matter. Generation therefore rejects any template
#      that is mirror-symmetric and guarantees placements in mirrored parity classes.
#    - main() does not explicitly verify "hollow frame" structure, so the generator must
#      enforce border-only masks with empty interiors. It does so deterministically by
#      starting from a square ring and removing exactly one border cell.

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from solvers.arc_utils import paste  # noqa: E402


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a hollow-frame stamp legend and rebuild it as a kaleidoscopic lattice.

    Input structure:
    - The input is divided into K legend columns of width N, separated by 1-cell dashed separator columns.
    - Rows 0..N-1 contain K monochrome hollow-frame templates; each frame uses only border cells and leaves
      its interior empty.
    - Rows N+1..2N contain sparse marker pixels; each legend column uses exactly one marker color.
    - Rows below the second dashed separator contain one palette swatch per legend column.

    Rule:
    - A marker's color identifies which top template/palette pair to use, and its local slot (lr, lc) inside
      the legend column identifies which logical output block to fill.
    - The output is not a plain N-by-N block board anymore: 1-pixel separator lines are inserted between all
      adjacent blocks, so logical block (rr, cc) starts at pixel (rr*(N+1), cc*(N+1)).
    - Before stamping, odd logical block rows flip the template vertically and odd logical block columns flip
      it horizontally; if both parities are odd, both flips apply.

    Key insight:
    - The legend decoding itself is still color lookup plus local-slot lookup. The new work is to map each
      logical block into a larger separator-lined canvas and apply block-parity mirroring to the decoded
      hollow-frame mask before recoloring it.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flat))
    bg = int(counts.most_common(1)[0][0])
    sep = int(max((cnt, col) for col, cnt in counts.items() if col != bg)[1])

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and len(sep_rows) >= 2

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)
    assert k >= 1

    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        col = vals.pop()
        tcols.append(col)
        tmasks.append(sub == col)
    assert len(set(tcols)) == k
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    pal_r0 = sep_rows[-1] + 1
    pcols: list[int] = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat} - {bg, sep}
        assert len(vals) == 1
        pcols.append(vals.pop())

    out = _blank_gridded_output(n, bg, sep)
    mid_r0 = n + 1
    slot_owner: dict[tuple[int, int], int] = {}

    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg or v == sep:
                    continue
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = tcol_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates must occupy different slots"
                slot_owner[(lr, lc)] = j
                tile = _render_tile(tmasks[j], pcols[j], bg, lr, lc)
                _paste_output_block(out, tile, lr, lc, n)

    return out


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_CHOICES = [4, 5]
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_MIN_SLOT_COVERAGE = 0.70
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 400


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _block_origin(index: int, n: int) -> int:
    return index * (n + 1)


def _separator_positions(n: int) -> list[int]:
    return [cut * (n + 1) - 1 for cut in range(1, n)]


def _blank_gridded_output(n: int, bg: int, sep: int) -> np.ndarray:
    size = n * n + n - 1
    out = np.full((size, size), bg, dtype=int)
    for idx in _separator_positions(n):
        out[idx, :] = sep
        out[:, idx] = sep
    return out


def _paste_output_block(out: np.ndarray, tile: np.ndarray, rr: int, cc: int, n: int) -> None:
    paste(out, tile, _block_origin(rr, n), _block_origin(cc, n))


def _kaleidoscope(mask: np.ndarray, block_r: int, block_c: int) -> np.ndarray:
    cur = mask
    if block_r % 2 == 1:
        cur = np.flipud(cur)
    if block_c % 2 == 1:
        cur = np.fliplr(cur)
    return cur


def _render_tile(mask: np.ndarray, color: int, bg: int, block_r: int, block_c: int) -> np.ndarray:
    warped = _kaleidoscope(mask, block_r, block_c)
    return np.where(warped, color, bg)


def _ring_mask(n: int) -> np.ndarray:
    mask = np.zeros((n, n), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _perimeter_cells(n: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(n):
        cells.append((0, c))
    for r in range(1, n - 1):
        cells.append((r, n - 1))
    for c in range(n - 1, -1, -1):
        cells.append((n - 1, c))
    for r in range(n - 2, 0, -1):
        cells.append((r, 0))
    return cells


def _flip_pos_h(pos: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = pos
    return r, n - 1 - c


def _flip_pos_v(pos: tuple[int, int], n: int) -> tuple[int, int]:
    r, c = pos
    return n - 1 - r, c


def _notch_orbit_size(pos: tuple[int, int], n: int) -> int:
    orbit = {
        pos,
        _flip_pos_h(pos, n),
        _flip_pos_v(pos, n),
        _flip_pos_v(_flip_pos_h(pos, n), n),
    }
    return len(orbit)


def _sample_templates(n: int, colors: list[int]) -> list[np.ndarray]:
    count = len(colors)
    ring = _ring_mask(n)
    candidates = [pos for pos in _perimeter_cells(n) if _notch_orbit_size(pos, n) == 4]
    chosen = random.sample(candidates, count)
    templates: list[np.ndarray] = []
    for notch_r, notch_c in chosen:
        mask = ring.copy()
        mask[notch_r, notch_c] = False
        templates.append(mask)
    return templates


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = _SEP
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = _SEP


def _sample_placements(n: int, k: int) -> list[tuple[int, int, int]] | None:
    odd_even = [(r, c) for r in range(n) for c in range(n) if r % 2 == 1 and c % 2 == 0]
    even_odd = [(r, c) for r in range(n) for c in range(n) if r % 2 == 0 and c % 2 == 1]
    odd_odd = [(r, c) for r in range(n) for c in range(n) if r % 2 == 1 and c % 2 == 1]
    even_even = [(r, c) for r in range(n) for c in range(n) if r % 2 == 0 and c % 2 == 0]
    if len(odd_even) < k or len(even_odd) < k or not odd_odd or not even_even:
        return None

    random.shuffle(odd_even)
    random.shuffle(even_odd)
    random.shuffle(odd_odd)
    random.shuffle(even_even)

    placements: list[tuple[int, int, int]] = []
    used_slots: set[tuple[int, int]] = set()
    for j in range(k):
        for slot in (odd_even.pop(), even_odd.pop()):
            placements.append((j, slot[0], slot[1]))
            used_slots.add(slot)

    extras_pool = odd_odd + even_even + odd_even + even_odd
    random.shuffle(extras_pool)
    min_marks = max(len(placements), int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)

    for pool in (odd_odd, even_even):
        slot = pool.pop()
        if slot not in used_slots and len(placements) < mark_count:
            placements.append((random.randrange(k), slot[0], slot[1]))
            used_slots.add(slot)

    for slot in extras_pool:
        if len(placements) >= mark_count:
            break
        if slot in used_slots:
            continue
        placements.append((random.randrange(k), slot[0], slot[1]))
        used_slots.add(slot)

    if len(placements) != mark_count:
        return None
    return placements


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
    *,
    use_mirror: bool,
) -> np.ndarray:
    expected = _blank_gridded_output(n, _BG, _SEP)
    for j, rr, cc in placements:
        if use_mirror:
            tile = _render_tile(templates[j], pcols[j], _BG, rr, cc)
        else:
            tile = np.where(templates[j], pcols[j], _BG)
        _paste_output_block(expected, tile, rr, cc, n)
    return expected


def _block_view(grid: np.ndarray, rr: int, cc: int, n: int) -> np.ndarray:
    r0 = _block_origin(rr, n)
    c0 = _block_origin(cc, n)
    return grid[r0 : r0 + n, c0 : c0 + n]


def _count_changed_blocks(a: np.ndarray, b: np.ndarray, n: int) -> int:
    diff = 0
    for rr in range(n):
        for cc in range(n):
            if not np.array_equal(_block_view(a, rr, cc, n), _block_view(b, rr, cc, n)):
                diff += 1
    return diff


def _fusion_metrics(
    n: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> dict[str, int]:
    combo = _build_expected(n, templates, pcols, placements, use_mirror=True)
    no_mirror = _build_expected(n, templates, pcols, placements, use_mirror=False)
    mirrored_blocks = 0
    odd_parity_blocks = 0
    for j, rr, cc in placements:
        plain = np.where(templates[j], pcols[j], _BG)
        warped = _render_tile(templates[j], pcols[j], _BG, rr, cc)
        if rr % 2 == 1 or cc % 2 == 1:
            odd_parity_blocks += 1
        if not np.array_equal(plain, warped):
            mirrored_blocks += 1
    exact_rings = sum(int(mask.sum()) == 4 * n - 4 for mask in templates)
    return {
        "mirrored_blocks": mirrored_blocks,
        "odd_parity_blocks": odd_parity_blocks,
        "diff_vs_no_mirror": _count_changed_blocks(combo, no_mirror, n),
        "exact_rings": exact_rings,
        "template_count": len(templates),
    }


def _gen_attempt() -> tuple[np.ndarray, dict[str, int]] | None:
    n = random.choice(_N_CHOICES)
    k = random.randint(*_K_RANGE)
    tcols = random.sample(_AVAILABLE_COLORS, k)
    pcols = random.sample(_AVAILABLE_COLORS, k)

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    templates = _sample_templates(n, tcols)
    placements = _sample_placements(n, k)
    if placements is None:
        return None

    metrics = _fusion_metrics(n, templates, pcols, placements)
    if metrics["mirrored_blocks"] < k + 1:
        return None
    if metrics["diff_vs_no_mirror"] < k + 1:
        return None
    if metrics["exact_rings"] != 0:
        return None

    expected = _build_expected(n, templates, pcols, placements, use_mirror=True)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], _BG), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = tcols[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid, metrics
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        built = _gen_attempt()
        if built is not None:
            grid, _metrics = built
            return grid
    raise RuntimeError("failed to generate a valid combo 65b59efc input")
