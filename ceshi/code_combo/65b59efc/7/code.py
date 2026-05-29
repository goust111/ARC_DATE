# Combo variant of 65b59efc (index 7)
# combo_spec: D5/2+D7/1
# Name: 多色反转印章

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
#    - Original 65b59efc: each top template is a single-color binary mask. A middle-band marker color
#      names the template, and the marker's local (lr, lc) chooses the output block. That block is drawn
#      as "mask cells = palette color, non-mask cells = background".
#    - Fused task here: the top template is itself multi-color. Example with N=4:
#        top template j =
#          [[2,2,0,0],
#           [2,7,7,0],
#           [2,7,8,0],
#           [0,0,0,0]]
#        marker color = 2, palette color = 4, marker local slot = (1,3)
#      The solver must still use marker color 2 to pick this template and slot (1,3) to pick the output
#      block, but the rendered 4x4 stamp becomes
#          [[2,2,4,4],
#           [2,7,7,4],
#           [2,7,8,4],
#           [4,4,4,4]]
#      instead of recoloring the foreground to one palette color. The visible difference is exactly the
#      D5/2 inversion merged with D7/1 multi-color structure.
#
# b) Complexity calibration
#    - Each template has one unique dominant "key color" plus 1-2 shared accent colors. Marker colors use
#      only the dominant key colors, so template lookup stays recoverable while the templates are genuinely
#      multi-color.
#    - N is restricted to 4-5 so the 2-3 color template structure and the palette-filled complement are
#      both easy to see. Every template keeps at least two background cells, so the inversion is always
#      visible in the output.
#    - Accent colors are reused across templates within a sample, which gives repeated evidence that not
#      every template color is a template identifier.
#
# c) Implicit-assumption audit against the original main()
#    - Original assumption "top template contains exactly one non-bg non-sep color" is false. The new
#      solver must read the full N x N color patch, not a boolean mask.
#    - Original assumption "rendering is np.where(mask, palette, bg)" is false. Rendering is now
#      "template color where active, palette color where inactive".
#    - Original assumption "the one template color is the marker color" is weakened. The generator now
#      guarantees that the marker color equals the template's dominant key color, so main() maps markers
#      through that dominant color only.
#    - Original assumption "full templates may be solid" is unsafe now because the inverted fill would
#      disappear. The generator rejects solid masks and guarantees inactive cells exist.
#
# d) Placement legality
#    - Top templates occupy disjoint N-wide columns separated by dashed separator columns, exactly as in
#      the base layout.
#    - Middle-band placements still use disjoint local slots across templates, so at most one stamp owns
#      each output block.
#    - Palette swatches stay inside the bottom band and trailing padding stays background-only, so layout
#      parsing remains stable.
#
# e) Visual observability
#    - Humans can see 2-3 colors inside one top template and then see those same colors preserved inside
#      the output stamp instead of being flattened to a single palette color.
#    - At the same time, the cells that were blank inside the top template become palette-colored in the
#      output block, so the foreground/background role swap is explicit.
#    - Because accent colors repeat across different templates, the consistent marker clue is the dominant
#      key color rather than "any color that appears in the template".
#
# f) Visual cleanliness
#    - The scene keeps the original three-band legend layout and one-color separator system.
#    - Template accents are small contiguous patches rather than random pepper noise, so each template still
#      reads as one coherent sprite.
#    - Palette swatches remain single-color and centered, keeping the extra color load localized to the top
#      band and stamped output blocks only.
#
# g) New assumptions introduced by main(), and generator support
#    - main() assumes each template has a unique dominant key color; generate_input() enforces that by
#      giving every template its own primary color and keeping accent counts smaller.
#    - main() assumes each template has 2-3 non-bg colors; generate_input() always paints 1 or 2 accent
#      clusters on top of the dominant color.
#    - main() assumes every template has at least one inactive cell so the palette complement exists;
#      generate_input() rejects full masks.
#    - main() assumes each middle tile column contains markers of one key color only; generator keeps the
#      original one-template-per-physical-column construction.


_BG = 0
_SEP = 5
_AVAILABLE_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]

_N_RANGE = (4, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.82)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8
_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
_MAX_RETRIES = 500


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _sep_count(length: int, n: int) -> int:
    return sum(1 for x in range(length) if _dash_cell(x, n))


def _raw_col(tile_col: int, local_c: int, n: int) -> int:
    return tile_col * (n + 1) + local_c


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((r + dr, c + dc))
    return out


def _sample_mask(n: int) -> np.ndarray:
    sprite = random_sprite(
        n,
        n,
        density=random.uniform(*_DENSITY_RANGE),
        symmetry=random.choice(_SYMMETRY_OPTIONS),
        color_palette=[1],
        connectivity=_TEMPLATE_CONNECTIVITY,
        background=_BG,
    )
    return sprite == 1


def _pick_cluster(mask: np.ndarray, allowed: set[tuple[int, int]], size: int) -> set[tuple[int, int]] | None:
    if len(allowed) < size:
        return None
    for _ in range(40):
        seed = random.choice(tuple(allowed))
        cluster = {seed}
        while len(cluster) < size:
            frontier = []
            for r, c in cluster:
                for nr, nc in _neighbors8(r, c):
                    if (nr, nc) in allowed and (nr, nc) not in cluster:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cluster.add(random.choice(frontier))
        if len(cluster) == size:
            return cluster
    return None


def _sample_multicolor_template(
    n: int,
    primary: int,
    shared_accents: tuple[int, int],
    want_three_colors: bool,
) -> np.ndarray | None:
    for _ in range(120):
        mask = _sample_mask(n)
        area = int(mask.sum())
        if area < 6 or area > n * n - 2:
            continue

        out = np.full((n, n), _BG, dtype=int)
        out[mask] = primary
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        remaining = set(coords)

        accent_total = 2 if want_three_colors else 1
        accent_sizes: list[int] = []
        free_budget = area - 3
        if free_budget < 2 * accent_total:
            continue
        for idx in range(accent_total):
            max_size = min(4, max(2, area // (4 if accent_total == 1 else 5)))
            min_size = 2
            size = random.randint(min_size, max_size)
            accent_sizes.append(size)
        if sum(accent_sizes) > free_budget:
            continue

        ok = True
        for color, size in zip(shared_accents[:accent_total], accent_sizes):
            cluster = _pick_cluster(mask, remaining, size)
            if cluster is None:
                ok = False
                break
            for r, c in cluster:
                out[r, c] = color
            remaining -= cluster
        if not ok:
            continue

        counts = Counter(int(v) for v in out.flat if int(v) != _BG)
        if len(counts) != accent_total + 1:
            continue
        if counts[primary] <= max(cnt for col, cnt in counts.items() if col != primary):
            continue
        return out
    return None


def _render_stamp(template: np.ndarray, palette_color: int) -> np.ndarray:
    return np.where(template != _BG, template, palette_color)


def _build_expected(
    n: int,
    templates: list[np.ndarray],
    palette_colors: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), _BG, dtype=int)
    for j, rr, cc in placements:
        paste(out, _render_stamp(templates[j], palette_colors[j]), rr * n, cc * n)
    return out


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


def _sample_templates(n: int, k: int) -> tuple[list[np.ndarray], list[int], list[int]] | None:
    primary_colors = random.sample(_AVAILABLE_COLORS, k)
    accent_pool = [c for c in _AVAILABLE_COLORS if c not in primary_colors]
    if len(accent_pool) < 2:
        return None
    shared_accents = tuple(random.sample(accent_pool, 2))

    # Always mix 2-color and 3-color templates in one puzzle so both branches are observable.
    three_color_js = set(random.sample(range(k), random.randint(1, k - 1)))
    templates: list[np.ndarray] = []
    seen_templates: set[bytes] = set()
    for j, primary in enumerate(primary_colors):
        tpl = _sample_multicolor_template(n, primary, shared_accents, j in three_color_js)
        if tpl is None:
            return None
        sig = tpl.tobytes()
        if sig in seen_templates:
            return None
        seen_templates.add(sig)
        templates.append(tpl)

    palette_colors = []
    for tpl in templates:
        active = {int(v) for v in tpl.flat if int(v) != _BG}
        choices = [c for c in _AVAILABLE_COLORS if c not in active]
        palette_colors.append(random.choice(choices))

    return templates, primary_colors, palette_colors


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)

    sampled = _sample_templates(n, k)
    if sampled is None:
        return None
    templates, key_colors, palette_colors = sampled

    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h = 2 * n + 2 + pal_h
    core_w = k * n + (k - 1)
    max_rpad = max(0, min(_MAX_ROW_PAD, 2 * _sep_count(core_h, n) + 1 - core_h))
    max_cpad = max(0, min(_MAX_COL_PAD, 2 * _sep_count(core_w, n) + 1 - core_w))
    h = core_h + random.randint(0, max_rpad)
    w = core_w + random.randint(0, max_cpad)

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = int(np.ceil(_MIN_SLOT_COVERAGE * n * n))
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]

    assign_pool = list(range(k))
    while len(assign_pool) < mark_count:
        assign_pool.extend(range(k))
    assign_pool = assign_pool[:mark_count]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(n, templates, palette_colors, placements)

    grid = np.full((h, w), _BG, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w)

    for j, tpl in enumerate(templates):
        paste(grid, tpl, 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, _raw_col(tile_assign[j], cc, n)] = key_colors[j]

    swatch_r = 2 * n + 2 + pal_h - 1
    for j, col in enumerate(palette_colors):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a tiled stamp legend whose templates are multi-color and whose blank cells invert into palette fill.

    Input structure:
    - The grid is divided into K tile columns of width N, separated by 1-column dashed separators.
    - Top band (rows 0..N-1): one N x N multi-color template per tile column, each using 2-3 non-background colors.
    - Middle band (rows N+1..2N): marker pixels. A tile column uses only one marker color, and a marker's local
      position (lr, lc) still names one output block.
    - Bottom band (rows 2N+2..): one palette swatch per tile column, giving the fill color paired with that template.
    - Dashed separator rows remain at rows N and 2N+1.

    Rule:
    - Marker color selects the template by matching the template's dominant key color from the top band.
    - Marker local position (lr, lc) chooses which N x N output block receives the stamp.
    - Stamping is inverted relative to the base task: active template cells keep their original template colors,
      while inactive cells inside the N x N block are filled with the template's palette color.

    Key insight: The top-band object is no longer a binary mask to recolor. Read the whole multi-color template,
    use the marker color only as a key into the dominant template color, and render each stamped block by preserving
    template colors on the foreground while using the palette color as the background inside that block.
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = Counter({c: n for c, n in counts.items() if c != bg}).most_common(1)[0][0]

    sep_cols = [c for c in range(w) if int((g[:, c] == sep).sum()) >= h // 2]
    sep_rows = [r for r in range(h) if int((g[r, :] == sep).sum()) >= w // 2]
    assert sep_cols and sep_rows

    n = sep_cols[0]
    starts = [0] + [c + 1 for c in sep_cols]
    k = len(starts)

    templates: list[np.ndarray] = []
    key_colors: list[int] = []
    for st in starts:
        sub = g[0:n, st : st + n].copy()
        vals = [int(v) for v in np.unique(sub) if int(v) not in {bg, sep}]
        assert 2 <= len(vals) <= 3
        tpl_counts = Counter(int(v) for v in sub.flat if int(v) not in {bg, sep})
        primary, primary_count = tpl_counts.most_common(1)[0]
        assert all(primary_count > cnt for col, cnt in tpl_counts.items() if col != primary)
        templates.append(sub)
        key_colors.append(primary)
    assert len(set(key_colors)) == k
    key_to_j = {col: j for j, col in enumerate(key_colors)}

    pal_r0 = sep_rows[-1] + 1
    palette_colors = []
    for st in starts:
        sub = g[pal_r0:h, st : st + n]
        vals = {int(v) for v in sub.flat if int(v) not in {bg, sep}}
        assert len(vals) == 1
        palette_colors.append(vals.pop())

    mid_r0 = n + 1
    out = np.full((n * n, n * n), bg, dtype=int)
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
                j = key_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j, "different templates must occupy different slots"
                slot_owner[(lr, lc)] = j
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = _render_stamp(templates[j], palette_colors[j])

    return out


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 65b59efc combo input")
