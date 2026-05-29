# Combo variant of 65b59efc (index 13)
# combo_spec: D17/0+D20/0
# Name: 可变配色噪声干扰
#
# Design analysis:
# a) Concretization and behavior trace
#    - Original task: `bg=0`, `sep=5`, and trailing padding is pure background, so the solver can
#      recover the scaffold by simple frequency heuristics and then decode templates / markers / palette.
#    - Combo task: keep the same three-band stamp-map layout, but let `(bg, sep)` vary per instance and
#      plant a few stray colored pixels only in the trailing padding. Example with `N=4`, `K=3`:
#      rows `4` and `9` still contain the dashed separator row, columns `4` and `9` still contain the
#      dashed separator columns, yet the last padded row/cols may contain random red / teal specks.
#      The solver must first recover the top-left structural core, then ignore everything beyond it.
#    - Difference: the useful region is no longer recoverable from fixed color ids, and the padding is
#      no longer a clean all-background suffix.
# b) Complexity calibration
#    - `N` and `K` remain in the original small ranges, so the human still sees a familiar tiled scaffold.
#    - Noise is sparse and confined to padding, which makes the interference visible but not dominant.
#    - The generator rejects the original `(bg=0, sep=5)` pair, so the color-variation dimension is always active.
# c) Broken original assumptions
#    - `sep == second-most-common color` is no longer trusted; noise can alter non-background frequencies.
#    - `padding is entirely background` is false; any suffix scan that assumes purity can be fooled.
#    - The solver must therefore validate candidate `(sep, N, K, pal_h)` tuples against the dashed scaffold
#      and the three functional bands, instead of relying on a single count-based guess.
# d) Placement legality
#    - Templates, markers, palette swatches, and separator lines still occupy the same disjoint bands as
#      the base task, so functional objects never overlap.
#    - Noise is sampled only from cells with `r >= core_h or c >= core_w`, i.e. genuine padding cells.
# e) Visual observability
#    - Across 2-4 examples, a person can see two stable cues: separator / background colors vary, and the
#      extra colored specks appear only in the right/bottom padding rather than inside the scaffold.
# f) Visual tidiness
#    - Noise remains isolated and sparse, so the meaningful three-band structure is still easy to parse.
#    - Separator color is not reused for noise, avoiding fake continuation lines in the padding.
# g) Inferability of changes
#    - No hidden numeric ordering is used. The new rule is entirely visible: identify the dominant
#      background, find the dashed separator scaffold, crop away noisy suffix padding, then decode normally.
# New solver assumptions and generator support
#    - Solver assumption: the true core starts at the top-left corner and padding is trailing-only.
#      Generator support: noise is injected only after the fully constructed core.
#    - Solver assumption: exactly one candidate with maximal core area satisfies the scaffold + band checks.
#      Generator support: the base task's structural constraints remain intact inside the core.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import math
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


def _dash_cell(pos: int, n: int) -> bool:
    k = pos % (n + 1)
    return k == n or k % 2 == 0


def _core_dims(n: int, k: int, pal_h: int) -> tuple[int, int]:
    return 2 * n + 2 + pal_h, k * n + (k - 1)


def _tile_starts(n: int, k: int) -> list[int]:
    return [j * (n + 1) for j in range(k)]


def _non_bg_colors(sub: np.ndarray, bg: int) -> set[int]:
    return {int(v) for v in sub.flat if int(v) != bg}


def _validate_sep_row(g: np.ndarray, row: int, n: int, core_w: int, bg: int, sep: int) -> bool:
    return all(int(g[row, c]) == (sep if _dash_cell(c, n) else bg) for c in range(core_w))


def _validate_sep_col(g: np.ndarray, col: int, n: int, core_h: int, bg: int, sep: int) -> bool:
    return all(int(g[r, col]) == (sep if _dash_cell(r, n) else bg) for r in range(core_h))


def _candidate_layout(
    g: np.ndarray,
    bg: int,
    sep: int,
    n: int,
    k: int,
    pal_h: int,
) -> dict[str, object] | None:
    h, w = g.shape
    core_h, core_w = _core_dims(n, k, pal_h)
    if core_h > h or core_w > w:
        return None

    if not _validate_sep_row(g, n, n, core_w, bg, sep):
        return None
    if not _validate_sep_row(g, 2 * n + 1, n, core_w, bg, sep):
        return None

    sep_cols = [n + i * (n + 1) for i in range(k - 1)]
    if not all(_validate_sep_col(g, c, n, core_h, bg, sep) for c in sep_cols):
        return None

    starts = _tile_starts(n, k)
    tcols: list[int] = []
    tmasks: list[np.ndarray] = []
    for st in starts:
        sub = g[0:n, st : st + n]
        vals = _non_bg_colors(sub, bg)
        if len(vals) != 1:
            return None
        col = next(iter(vals))
        if col == sep:
            return None
        tcols.append(col)
        tmasks.append(sub == col)
    if len(set(tcols)) != k:
        return None

    mid_colors: list[int] = []
    for st in starts:
        sub = g[n + 1 : 2 * n + 1, st : st + n]
        vals = _non_bg_colors(sub, bg)
        if len(vals) != 1:
            return None
        col = next(iter(vals))
        if col == sep or col not in tcols:
            return None
        mid_colors.append(col)
    if set(mid_colors) != set(tcols):
        return None

    pcols: list[int] = []
    for st in starts:
        sub = g[2 * n + 2 : core_h, st : st + n]
        vals = _non_bg_colors(sub, bg)
        if len(vals) != 1:
            return None
        col = next(iter(vals))
        if col == sep:
            return None
        pcols.append(col)

    return {
        "bg": bg,
        "sep": sep,
        "n": n,
        "k": k,
        "pal_h": pal_h,
        "core_h": core_h,
        "core_w": core_w,
        "starts": starts,
        "tcols": tcols,
        "tmasks": tmasks,
        "pcols": pcols,
    }


def _infer_layout(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    h, w = g.shape
    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])

    candidates: list[dict[str, object]] = []
    for sep, _cnt in counts.most_common():
        sep = int(sep)
        if sep == bg:
            continue
        for n in range(3, min(5, h - 2, w) + 1):
            for k in range(3, 5):
                for pal_h in range(2, 5):
                    info = _candidate_layout(g, bg, sep, n, k, pal_h)
                    if info is not None:
                        candidates.append(info)
    if not candidates:
        raise ValueError("unable to infer layout")
    candidates.sort(key=lambda info: (int(info["core_w"]) * int(info["core_h"]), int(info["core_w"])), reverse=True)
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode a stamp map whose separator/background colors vary and whose trailing padding may contain noise.

    Input structure:
    - The useful content still starts at the top-left and forms K tile columns of width N, separated by
      1-cell dashed separator columns, plus two dashed separator rows.
    - Top band: K monochrome template masks. Middle band: marker pixels. Bottom band: K palette swatches.
    - Extra rows and columns may trail on the bottom/right as padding, and that padding can contain a few
      arbitrary non-background specks that do not carry any meaning.

    Rule:
    - First identify the dominant background color, then test candidate separator colors and small layout
      sizes against the dashed scaffold to recover the true top-left core and discard noisy padding.
    - Within that recovered core, marker color still chooses the template, marker local coordinates still
      choose the output block, and the aligned palette swatch still provides the repaint color.

    Key insight:
    - The only globally rigid cue is the dashed separator scaffold. Once that scaffold is found, everything
      beyond the inferred core rectangle is padding noise and can be ignored before applying the original decode.
    """
    g = np.array(grid, dtype=int)
    info = _infer_layout(g)

    bg = int(info["bg"])
    n = int(info["n"])
    starts = list(info["starts"])
    tcols = list(info["tcols"])
    tmasks = list(info["tmasks"])
    pcols = list(info["pcols"])
    tcol_to_j = {col: j for j, col in enumerate(tcols)}

    out = np.full((n * n, n * n), bg, dtype=int)
    slot_owner: dict[tuple[int, int], int] = {}
    mid_r0 = n + 1

    for st in starts:
        tile_color = None
        for lr in range(n):
            for lc in range(n):
                v = int(g[mid_r0 + lr, st + lc])
                if v == bg:
                    continue
                assert v in tcol_to_j
                assert tile_color is None or v == tile_color, "tile column must have uniform marker color"
                tile_color = v
                j = tcol_to_j[v]
                assert (lr, lc) not in slot_owner or slot_owner[(lr, lc)] == j
                slot_owner[(lr, lc)] = j
                tile = np.where(tmasks[j], pcols[j], bg)
                out[lr * n : (lr + 1) * n, lc * n : (lc + 1) * n] = tile

    return out


# --------------- generator (combo D17/0 + D20/0) ---------------

_ALL_COLORS = list(range(10))
_ORIGINAL_BG = 0
_ORIGINAL_SEP = 5

_N_RANGE = (3, 5)
_K_RANGE = (3, 4)
_PAL_H_CHOICES = [2, 3, 4]
_PAL_H_WEIGHTS = [8, 1, 1]
_MAX_ROW_PAD = 2
_MAX_COL_PAD = 7

_DENSITY_RANGE = (0.55, 0.92)
_SYMMETRY_OPTIONS = [None, "horizontal", "vertical", "mirror", "not_symmetric"]
_TEMPLATE_CONNECTIVITY = 8

_MIN_SLOT_COVERAGE = 0.60
_SWATCH_DOUBLE_PROB = 0.35
_MAX_NOISE_PIXELS = 6
_MAX_RETRIES = 400


def _sample_bg_sep() -> tuple[int, int]:
    while True:
        bg, sep = random.sample(_ALL_COLORS, 2)
        if (bg, sep) != (_ORIGINAL_BG, _ORIGINAL_SEP):
            return bg, sep


def _sample_template(n: int, color: int, bg: int) -> np.ndarray:
    sprite = random_sprite(
        n,
        n,
        density=random.uniform(*_DENSITY_RANGE),
        symmetry=random.choice(_SYMMETRY_OPTIONS),
        color_palette=[color],
        connectivity=_TEMPLATE_CONNECTIVITY,
        background=bg,
    )
    return sprite == color


def _build_expected(
    n: int,
    bg: int,
    templates: list[np.ndarray],
    pcols: list[int],
    placements: list[tuple[int, int, int]],
) -> np.ndarray:
    out = np.full((n * n, n * n), bg, dtype=int)
    for j, rr, cc in placements:
        tile = np.where(templates[j], pcols[j], bg)
        paste(out, tile, rr * n, cc * n)
    return out


def _draw_separators(grid: np.ndarray, n: int, k: int, core_h: int, core_w: int, sep: int) -> None:
    for r in (n, 2 * n + 1):
        for c in range(core_w):
            if _dash_cell(c, n):
                grid[r, c] = sep
    for i in range(k - 1):
        c = n + i * (n + 1)
        for r in range(core_h):
            if _dash_cell(r, n):
                grid[r, c] = sep


def _padding_coords(h: int, w: int, core_h: int, core_w: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(h) for c in range(w) if r >= core_h or c >= core_w]


def _inject_padding_noise(grid: np.ndarray, core_h: int, core_w: int, bg: int, sep: int) -> int:
    coords = _padding_coords(grid.shape[0], grid.shape[1], core_h, core_w)
    assert coords
    noise_colors = [c for c in _ALL_COLORS if c not in {bg, sep}]
    random.shuffle(coords)
    noise_count = random.randint(1, min(len(coords), _MAX_NOISE_PIXELS))
    for r, c in coords[:noise_count]:
        grid[r, c] = random.choice(noise_colors)
    return noise_count


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(*_N_RANGE)
    k = random.randint(*_K_RANGE)
    pal_h = random.choices(_PAL_H_CHOICES, weights=_PAL_H_WEIGHTS)[0]
    core_h, core_w = _core_dims(n, k, pal_h)

    row_pad = random.randint(0, _MAX_ROW_PAD)
    col_pad = random.randint(0, _MAX_COL_PAD)
    if row_pad == 0 and col_pad == 0:
        if random.random() < 0.5:
            row_pad = 1
        else:
            col_pad = 1
    h = core_h + row_pad
    w = core_w + col_pad

    bg, sep = _sample_bg_sep()
    usable = [c for c in _ALL_COLORS if c not in {bg, sep}]
    tcols = random.sample(usable, k)
    pcols = random.sample(usable, k)
    templates = [_sample_template(n, col, bg) for col in tcols]

    all_slots = [(rr, cc) for rr in range(n) for cc in range(n)]
    random.shuffle(all_slots)
    min_marks = math.ceil(_MIN_SLOT_COVERAGE * n * n)
    mark_count = random.randint(min_marks, n * n)
    used_slots = all_slots[:mark_count]
    assign_pool = (list(range(k)) * ((mark_count + k - 1) // k))[:mark_count]
    random.shuffle(assign_pool)
    placements = [(j, rr, cc) for (rr, cc), j in zip(used_slots, assign_pool)]

    expected = _build_expected(n, bg, templates, pcols, placements)

    grid = np.full((h, w), bg, dtype=int)
    _draw_separators(grid, n, k, core_h, core_w, sep)

    for j, mask in enumerate(templates):
        paste(grid, np.where(mask, tcols[j], bg), 0, j * (n + 1))

    tile_assign = list(range(k))
    random.shuffle(tile_assign)
    for j, rr, cc in placements:
        grid[n + 1 + rr, tile_assign[j] * (n + 1) + cc] = tcols[j]

    swatch_r = core_h - 1
    for j, col in enumerate(pcols):
        start = j * (n + 1)
        size = 2 if random.random() < _SWATCH_DOUBLE_PROB else 1
        c0 = start + max(0, (n - size) // 2)
        grid[swatch_r, c0 : c0 + size] = col

    _inject_padding_noise(grid, core_h, core_w, bg, sep)

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
    raise RuntimeError("failed to generate a valid 65b59efc combo input")


def _measure_fusion_features(grid: np.ndarray) -> dict[str, int]:
    info = _infer_layout(np.array(grid, dtype=int))
    bg = int(info["bg"])
    sep = int(info["sep"])
    core_h = int(info["core_h"])
    core_w = int(info["core_w"])
    g = np.array(grid, dtype=int)
    noise = sum(
        1
        for r in range(g.shape[0])
        for c in range(g.shape[1])
        if (r >= core_h or c >= core_w) and int(g[r, c]) != bg
    )
    return {
        "bg": bg,
        "sep": sep,
        "core_h": core_h,
        "core_w": core_w,
        "padding_noise": noise,
    }
