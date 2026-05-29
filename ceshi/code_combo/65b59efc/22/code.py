# Combo variant of 65b59efc (index 22)
# combo_spec: D5/1+D7/1+D8/0
# Name: 多色双色全映射

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


# Design analysis for the fused variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original 65b59efc:
#      * Top band: each tile column stores one single-color binary template.
#      * Marker color directly chooses that template.
#      * Bottom band only gives one repaint color, so output tiles are "template color vs global bg".
#    - Fused rule here:
#      * Each top-band template now has 2-3 source colors plus optional black holes.
#      * The bottom band becomes a 2-row legend of unordered color pairs per template column:
#        upper pair = {template key color, output fill color}, lower pair = {template key color, output
#        foreground color}. The key may appear on either side.
#      * A middle-band marker still chooses the template by its color, but that same color must now be
#        recovered from the repeated pair member: only template cells of the keyed source color become the
#        output foreground; every other template cell, including other source colors and blank cells, becomes
#        the output fill color.
#    - Small example with N=4:
#        template =
#          [[1, 1, 7, 0],
#           [1, 8, 7, 0],
#           [0, 8, 1, 1],
#           [0, 0, 1, 1]]
#        bottom legend pairs = {8, 6} and {8, 3}
#        marker local slot = (2, 0), marker color = 8
#      Original logic would treat the whole non-black template as one mask. The fused logic instead paints
#      only the two source-8 cells with 3, and paints every 1/7/0 cell with 6, producing a full 4x4
#      two-color tile whose internal split depends on the template's multicolor structure.
#
# b) Complexity calibration
#    - The output still uses exactly two colors per stamped block, so the task does not explode into an
#      unrestricted recoloring problem.
#    - The input templates visibly use 2-3 colors, and each sample guarantees at least one true 3-color
#      template, so the D7 change is easy to notice.
#    - The bottom legend uses exactly two pair rows with fixed row roles (`fill pair`, then `fg pair`) but
#      keeps the key/value order inside each pair unordered. That keeps the new codebook small while still
#      making the D8 color-pair role visible.
#
# c) Implicit-assumption audit against the original main()
#    - "Each template tile has exactly one non-background color" is false; parsing must keep the entire
#      multicolor tile, not reduce it to a boolean mask.
#    - "The palette contributes one output color" is false; parsing must decode two stacked unordered pairs.
#    - "Non-template cells stay global background in the output" is false; output cells not equal to the
#      keyed source color become the per-template fill color.
#    - "Marker color alone completely determines output repainting" is false; marker color now works
#      together with the repeated pair member to decide which source color is foreground.
#
# d) Placement legality
#    - The original three-band geometry is preserved, so templates, markers, and palette legend remain
#      in separate non-overlapping regions.
#    - Marker slots are still unique across templates, so at most one stamp owns each output block.
#    - The 2-row palette legend lives entirely below the second separator row and fits inside each N-wide
#      tile column with no spill into separator columns.
#
# e) Visual observability
#    - A human sees multicolor templates on top and fully two-color tiles on the output, so it is clear
#      that some source colors are collapsing together.
#    - The repeated color across the two legend rows identifies one of the template's source colors, giving
#      a visible cue for which source color survives as output foreground.
#    - Because at least one template has three source colors in every sample, the rule "all non-key colors
#      merge into one fill color" can be inferred from just a few examples.
#
# f) Visual cleanliness
#    - No extra overlays or dense text-like codes are added; only two compact palette pairs per template
#      column appear in the bottom band.
#    - Accent colors are drawn from a small shared pool, so the top band is varied but not noisy.
#    - Output blocks are full tiles with only two colors, which keeps the final mosaic easy to read.
#
# g) New assumptions introduced by main(), and generator support
#    - main() assumes the palette band has exactly two informative rows and that the template key color is
#      the unique color shared by the two unordered pairs; generate_input() always builds that exact legend.
#    - main() assumes each marker column is color-uniform and that the marker color matches exactly one
#      template key; generate_input() keeps distinct keys and the original one-template-per-marker-column
#      constraint.
#    - main() assumes every template contains its keyed source color and at least one non-key source color;
#      generate_input() rejects templates that would collapse back to the binary behavior.
#    - main() assumes samples include both 2-color and 3-color templates and mixed pair directions;
#      generate_input() enforces both so the fusion remains observable.
#    - main() assumes separator detection can rely on one color forming both repeated rows and repeated
#      columns; generate_input() keeps the original dashed separator geometry and avoids trailing padding.


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_CHOICES = [4, 5]
_K = 3
_PAL_H_CHOICES = [2, 3]
_DENSITY_RANGE = (0.48, 0.82)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_MIN_SLOT_COVERAGE = 0.65
_MAX_RETRIES = 500


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _non_special_vals(arr: np.ndarray, bg: int, sep: int) -> set[int]:
    return {int(v) for v in arr.flat if int(v) not in {bg, sep}}


def _infer_layout(g: np.ndarray) -> tuple[int, int, int, list[int], int, int, int]:
    h, w = g.shape
    bg = int(Counter(map(int, g.flatten())).most_common(1)[0][0])
    candidates: list[tuple[int, int, list[int], list[int]]] = []

    for color in sorted(set(map(int, g.flatten())) - {bg}):
        rows = [r for r in range(h) if int((g[r] == color).sum()) >= max(3, w // 3)]
        cols = [c for c in range(w) if int((g[:, c] == color).sum()) >= max(3, h // 3)]
        if len(rows) >= 2 and len(cols) >= 1:
            candidates.append((len(rows) + len(cols), color, rows, cols))

    assert candidates, "failed to identify separator color"
    _, sep, sep_rows, sep_cols = max(candidates, key=lambda item: item[0])
    sep_rows = sorted(sep_rows)
    sep_cols = sorted(sep_cols)
    n = sep_rows[0]
    assert sep_cols[0] == n
    assert len(sep_rows) >= 2 and sep_rows[1] == 2 * n + 1
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)
    mid_r0 = n + 1
    pal_r0 = 2 * n + 2
    assert pal_r0 + 1 < h
    return bg, sep, n, starts, k, mid_r0, pal_r0


def _parse_grid(g: np.ndarray) -> dict:
    bg, sep, n, starts, k, mid_r0, pal_r0 = _infer_layout(g)

    templates: list[np.ndarray] = []
    keys: list[int] = []
    palette_fg: list[int] = []
    palette_fill: list[int] = []

    for st in starts:
        tpl = g[0:n, st : st + n].copy()
        vals = _non_special_vals(tpl, bg, sep)
        assert 2 <= len(vals) <= 3
        templates.append(tpl)

    palette_rows = [
        r
        for r in range(pal_r0, g.shape[0])
        if _non_special_vals(g[r : r + 1, :], bg, sep)
    ]
    assert len(palette_rows) == 2
    fill_row, fg_row = palette_rows

    for j, st in enumerate(starts):
        vals = _non_special_vals(templates[j], bg, sep)
        fill_pair = _non_special_vals(g[fill_row : fill_row + 1, st : st + n], bg, sep)
        fg_pair = _non_special_vals(g[fg_row : fg_row + 1, st : st + n], bg, sep)
        assert len(fill_pair) == 2 and len(fg_pair) == 2

        shared = fill_pair & fg_pair
        assert len(shared) == 1
        key = next(iter(shared))
        fill = next(v for v in fill_pair if v != key)
        fg = next(v for v in fg_pair if v != key)
        assert key in vals
        assert fg != fill
        keys.append(key)
        palette_fg.append(fg)
        palette_fill.append(fill)

    assert len(set(keys)) == k
    key_to_j = {key: j for j, key in enumerate(keys)}

    placements: list[tuple[int, int, int]] = []
    slot_owner: dict[tuple[int, int], int] = {}
    for st in starts:
        marker_key = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v in {bg, sep}:
                    continue
                assert marker_key is None or v == marker_key, "marker column must be color-uniform"
                marker_key = v
                assert v in key_to_j, "marker must match one legend key"
                j = key_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
                slot_owner[(lr, lc)] = j
                placements.append((j, lr, lc))

    return {
        "bg": bg,
        "sep": sep,
        "n": n,
        "starts": starts,
        "k": k,
        "mid_r0": mid_r0,
        "pal_r0": pal_r0,
        "templates": templates,
        "keys": keys,
        "palette_fg": palette_fg,
        "palette_fill": palette_fill,
        "placements": placements,
    }


def _render_keyed_tile(template: np.ndarray, key: int, fg: int, fill: int) -> np.ndarray:
    return np.where(template == key, fg, fill)


def _render_output(meta: dict) -> np.ndarray:
    n = meta["n"]
    bg = meta["bg"]
    out = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in meta["placements"]:
        tile = _render_keyed_tile(
            meta["templates"][j],
            meta["keys"][j],
            meta["palette_fg"][j],
            meta["palette_fill"][j],
        )
        paste(out, tile, rr * n, cc * n)
    return out


def _render_binary_baseline(meta: dict) -> np.ndarray:
    n = meta["n"]
    bg = meta["bg"]
    out = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in meta["placements"]:
        tile = np.where(meta["templates"][j] != bg, meta["palette_fg"][j], meta["palette_fill"][j])
        paste(out, tile, rr * n, cc * n)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend where multicolor templates collapse into keyed two-color output tiles.

    Input structure:
    - The grid keeps the original three stacked bands separated by dashed divider lines, with K template
      columns of width N and 1-cell separator columns between them.
    - Top band (rows 0..N-1): K multicolor templates, each using 2-3 non-separator colors.
    - Middle band (rows N+1..2N): sparse marker pixels; the local marker slot still chooses the output
      block location.
    - Bottom band contains exactly two informative palette rows. In each template column, the upper row is
      an unordered pair {template key color, output fill color}; the lower row is an unordered pair
      {template key color, output foreground color}.

    Rule:
    - A marker's local coordinates (lr, lc) still choose the N x N output block at block position (lr, lc).
    - The marker color identifies which template/legend stack to use. The template key color is the unique
      color shared by the two palette pairs in that column. In that template, only cells whose original
      source color equals that key become the legend foreground color.
    - Every other cell in the template tile, including other source colors and empty background cells,
      becomes the legend fill color, so each stamped block is fully recolored with exactly two colors.

    Key insight: The top band is no longer a binary mask. You must read the bottom legend to find which
    template source color survives as foreground, and you must decode each palette row as an unordered
    key-value pair rather than by fixed left/right position; all remaining template colors collapse together
    into the second output color.
    """
    meta = _parse_grid(np.array(grid, dtype=int))
    return _render_output(meta)


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


def _pair_cols(n: int) -> tuple[int, int]:
    left = min(n - 2, n // 2)
    return left, left + 1


def _draw_pair(grid: np.ndarray, row: int, st: int, n: int, key: int, value: int, key_left: bool) -> None:
    c0, c1 = _pair_cols(n)
    if key_left:
        grid[row, st + c0] = key
        grid[row, st + c1] = value
    else:
        grid[row, st + c0] = value
        grid[row, st + c1] = key


def _side_sets(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    coords = np.argwhere(mask)
    rs = coords[:, 0]
    cs = coords[:, 1]
    r0, r1 = int(rs.min()), int(rs.max())
    c0, c1 = int(cs.min()), int(cs.max())

    raw_sets = [
        [(int(r), int(c)) for r, c in coords if int(r) == r0],
        [(int(r), int(c)) for r, c in coords if int(r) == r1],
        [(int(r), int(c)) for r, c in coords if int(c) == c0],
        [(int(r), int(c)) for r, c in coords if int(c) == c1],
    ]

    seen: set[tuple[tuple[int, int], ...]] = set()
    out: list[list[tuple[int, int]]] = []
    for cells in raw_sets:
        key = tuple(sorted(cells))
        if len(cells) < 2 or key in seen:
            continue
        seen.add(key)
        out.append(list(key))
    return out


def _sample_multicolor_template(
    n: int,
    key_color: int,
    accent_colors: list[int],
    color_count: int,
) -> np.ndarray | None:
    for _ in range(240):
        sprite = random_sprite(
            n,
            n,
            density=random.uniform(*_DENSITY_RANGE),
            symmetry=random.choice(_SYMMETRY_OPTIONS),
            color_palette=[1],
            connectivity=_TEMPLATE_CONNECTIVITY,
            background=_BG,
        )
        mask = sprite == 1
        area = int(mask.sum())
        if area < n + 3:
            continue

        side_sets = _side_sets(mask)
        if not side_sets:
            continue

        out = np.full((n, n), _BG, dtype=int)
        out[mask] = key_color

        accent1 = random.choice(side_sets)
        for rr, cc in accent1:
            out[rr, cc] = accent_colors[0]

        used = set(accent1)
        if color_count == 3:
            random.shuffle(side_sets)
            accent2 = None
            for cells in side_sets:
                diff = [cell for cell in cells if cell not in used]
                if diff:
                    accent2 = diff
                    break
            if accent2 is None:
                continue
            for rr, cc in accent2:
                out[rr, cc] = accent_colors[1]

        key_count = int((out == key_color).sum())
        vals = {int(v) for v in out.flat if int(v) != _BG}
        if key_count < max(2, area // 3):
            continue
        if len(vals) != color_count:
            continue
        return out
    return None


def _sample_templates(n: int, keys: list[int], accent_pool: list[int]) -> list[np.ndarray] | None:
    color_counts = [2, 3] + [random.choice([2, 3]) for _ in range(_K - 2)]
    random.shuffle(color_counts)
    templates: list[np.ndarray] = []
    seen: set[bytes] = set()

    for j, key in enumerate(keys):
        target_colors = color_counts[j]
        tpl = None
        for _ in range(120):
            accents = random.sample(accent_pool, target_colors - 1)
            candidate = _sample_multicolor_template(n, key, accents, target_colors)
            if candidate is None:
                continue
            sig = candidate.tobytes()
            if sig in seen:
                continue
            tpl = candidate
            seen.add(sig)
            break
        if tpl is None:
            return None
        templates.append(tpl)
    return templates


def _seeded_placements(n: int, k: int) -> list[tuple[int, int, int]]:
    available = {(rr, cc) for rr in range(n) for cc in range(n)}
    placements: list[tuple[int, int, int]] = []

    for j in range(k):
        for rr in (j % n, (j + 1) % n):
            candidates = [(rr, cc) for cc in range(n) if (rr, cc) in available]
            if not candidates:
                return []
            slot = random.choice(candidates)
            available.remove(slot)
            placements.append((j, slot[0], slot[1]))

    remaining = list(available)
    random.shuffle(remaining)
    min_marks = max(len(placements), int(np.ceil(_MIN_SLOT_COVERAGE * n * n)))
    mark_count = random.randint(min_marks, n * n)
    extra = mark_count - len(placements)
    for rr, cc in remaining[:extra]:
        placements.append((random.randrange(k), rr, cc))

    random.shuffle(placements)
    return placements


def _build_meta(
    n: int,
    templates: list[np.ndarray],
    keys: list[int],
    palette_fg: list[int],
    palette_fill: list[int],
    placements: list[tuple[int, int, int]],
) -> dict:
    return {
        "bg": _BG,
        "sep": _SEP,
        "n": n,
        "templates": templates,
        "keys": keys,
        "palette_fg": palette_fg,
        "palette_fill": palette_fill,
        "placements": placements,
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_N_CHOICES)
    k = _K
    pal_h = random.choice(_PAL_H_CHOICES)

    key_colors = random.sample(_AVAILABLE_COLORS, k)
    remaining = [c for c in _AVAILABLE_COLORS if c not in key_colors]
    palette_fg = random.sample(remaining, k)
    accent_pool = [c for c in remaining if c not in palette_fg]
    if len(accent_pool) < 2:
        return None

    templates = _sample_templates(n, key_colors, accent_pool)
    if templates is None:
        return None

    palette_fill: list[int] = []
    for j in range(k):
        choices = [c for c in _AVAILABLE_COLORS if c not in {key_colors[j], palette_fg[j]}]
        palette_fill.append(random.choice(choices))

    placements = _seeded_placements(n, k)
    if not placements:
        return None

    meta = _build_meta(n, templates, key_colors, palette_fg, palette_fill, placements)
    expected = _render_output(meta)
    baseline = _render_binary_baseline(meta)
    if np.array_equal(expected, baseline):
        return None

    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    grid = np.full((core_h, core_w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, tpl in enumerate(templates):
        paste(grid, tpl, 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = key_colors[j]

    fill_orders = [random.choice([False, True]) for _ in range(k)]
    fg_orders = [random.choice([False, True]) for _ in range(k)]
    if all(fill_orders) or not any(fill_orders):
        return None
    if all(fg_orders) or not any(fg_orders):
        return None

    fill_row = 2 * n + pal_h
    fg_row = fill_row + 1
    for j in range(k):
        st = j * (n + 1)
        _draw_pair(grid, fill_row, st, n, key_colors[j], palette_fill[j], fill_orders[j])
        _draw_pair(grid, fg_row, st, n, key_colors[j], palette_fg[j], fg_orders[j])

    try:
        actual = main(grid)
    except Exception:
        return None
    if np.array_equal(actual, expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
