# Variant of 2ba387bc
# Dimension: D4 — 源对象的类型标识物 (src)
# Change: 框架(frame)通过1像素彩色边框+背景色内部识别；实心(solid)通过全部同色填充识别... → 实心方块可含少量噪声像素

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis
# ---------------
# Source-task note:
# - The prompt points to `b99e7126` as the anomaly-detection inspiration, but in this
#   workspace only `/ARC-data/.../b99e7126.json` is present; no readable solver source
#   exists. I therefore use the visible task intent as the baseline: recover the true
#   dominant pattern while ignoring 1-2 anomalous pixels.
#
# a) Concrete behavior and trace
# - Original objects are still scattered 4x4 tiles on a single-color background.
# - Frames stay unchanged: 12 border pixels share one non-bg color and the 2x2 center
#   is exactly background.
# - Solids are now "nearly solid": among the 16 cells, 14-15 (or all 16) share one
#   non-bg base color, and 1-2 cells may be different non-bg colors. Example:
#     input solid:
#       2222
#       2722
#       2222
#       2223
#   The original task would stack that raw noisy patch on the right. For this variant,
#   the key D4 cue is that the tile is identified by its majority fill color, so the
#   output stacks the canonical recovered solid:
#       2222
#       2222
#       2222
#       2222
#   This extra cleanup is necessary; otherwise the old `if frame else solid` logic from
#   the seed task would still solve the noisy inputs unchanged, making the D4 change
#   functionally invisible.
#
# b) Complexity calibration
# - Noise is capped at 1-2 pixels per 4x4 solid. That is enough to be visibly different
#   from the original pure solid class, but not enough to make the object ambiguous.
# - Frames remain exact. This keeps only two object categories, so the task does not
#   drift into a distractor-filtering or multi-rule puzzle.
# - Solids keep a strict 14/16 majority at minimum, so "majority fill" is a clean and
#   repeated cue rather than a fuzzy heuristic.
#
# c) Implicit-assumption audit of the original `main()`
# - `find_components(... monochromatic=False)`: still valid because noisy solids are
#   multicolor but remain one 4-connected component.
# - `if bbox_h != 4 or bbox_w != 4: continue`: still valid because every source object is
#   still a 4x4 tile.
# - `if len(bcols)==1 and interior==bg -> frame else solid`: this is the critical old
#   assumption. It relied on "the only non-frame objects are perfect solids". In the
#   variant, a non-frame object must additionally be checked for a 14+ cell non-bg
#   majority, and the output should use that recovered majority color rather than the raw
#   noisy patch.
# - Sorting by `(row, col)` and stacking by columns is unchanged and remains visible.
#
# d) Placement legality
# - Generator places each 4x4 tile with `padding=1`, so components never touch or merge.
# - Noisy solids use only non-bg colors, which avoids accidental transparent holes during
#   placement and guarantees their bbox stays 4x4.
# - The generator verifies `main(input)` against an independently composed expected
#   output before accepting a sample.
#
# e) Visual observability
# - Every generated solid contains 1-2 visible anomalies, so at least one output block is
#   "cleaner" than its input counterpart. This makes the new rule observable in 2-4 ARC
#   examples rather than hidden inside classification internals.
# - No sorting uses hidden numeric color order; order is still purely spatial.
#
# f) Visual cleanliness
# - Objects remain isolated 4x4 tiles with one-cell padding. There are no added legends,
#   markers, or detached noise components, so the scene stays close to the original task.
#
# g) Color inferability
# - The output solid color is simply the visually dominant color already occupying almost
#   all cells of that tile. Noise colors are not semantically mapped; they are anomalies
#   to be ignored. This is easy to infer from few examples.
#
# New solver assumptions introduced here, which the generator guarantees:
# - Every non-frame 4x4 tile intended as a solid has a unique non-bg majority color with
#   count at least 14.
# - There are no third kinds of 4x4 multicolor tiles.


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    a = np.full((4, 4), col, dtype=int)
    a[1:3, 1:3] = bg
    return a


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), col, dtype=int)


def _frame_color(sub: np.ndarray, bg: int) -> int | None:
    border = np.concatenate([sub[0, :], sub[-1, :], sub[1:-1, 0], sub[1:-1, -1]])
    bcols = set(map(int, border.tolist()))
    if len(bcols) != 1:
        return None
    col = int(next(iter(bcols)))
    if col == int(bg):
        return None
    if not bool((sub[1:3, 1:3] == int(bg)).all()):
        return None
    return col


def _solid_majority_color(sub: np.ndarray, bg: int) -> int | None:
    vals, counts = np.unique(sub, return_counts=True)
    best_idx = int(np.argmax(counts))
    col = int(vals[best_idx])
    cnt = int(counts[best_idx])
    if col == int(bg):
        return None
    if cnt < 14:
        return None
    return col


def _compose_output(
    bg: int,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> np.ndarray:
    frames = sorted(frames, key=lambda t: (int(t[0]), int(t[1])))
    solids = sorted(solids, key=lambda t: (int(t[0]), int(t[1])))
    out_h = 4 * max(int(len(frames)), int(len(solids)))
    out = np.full((out_h, 8), int(bg), dtype=int)
    for i, (_r0, _c0, sub) in enumerate(frames):
        paste(out, sub, int(i * 4), 0)
    for i, (_r0, _c0, sub) in enumerate(solids):
        paste(out, sub, int(i * 4), 4)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered 4x4 tiles into two columns by type.

    Variant rule:
    - Frames are still exact 1px colored borders with bg-filled 2x2 centers.
    - Solids may contain 1-2 anomalous pixels. Recover the solid's true color by the
      4x4 majority fill and output the canonical cleaned solid block.
    - Frames stack on the left; recovered solids stack on the right; each group keeps
      top-to-bottom order from the input.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        b = comp["bbox"]
        if bbox_h(b) != 4 or bbox_w(b) != 4:
            continue
        r0, c0, _r1, _c1 = b
        sub = crop_bbox(g, b)

        frame_col = _frame_color(sub, bg)
        if frame_col is not None:
            frames.append((int(r0), int(c0), sub))
            continue

        solid_col = _solid_majority_color(sub, bg)
        if solid_col is not None:
            solids.append((int(r0), int(c0), _make_solid(solid_col)))

    return _compose_output(bg, frames, solids)


def _make_noisy_solid(col: int, bg: int = 0) -> tuple[np.ndarray, int]:
    del bg  # kept only for signature symmetry with `_make_frame`
    a = _make_solid(col)
    n_noise = int(np.random.randint(1, 3))
    interior = [(1, 1), (1, 2), (2, 1), (2, 2)]
    positions = [interior[int(np.random.randint(0, len(interior)))]]
    while len(positions) < n_noise:
        pos = (int(np.random.randint(0, 4)), int(np.random.randint(0, 4)))
        if pos not in positions:
            positions.append(pos)
    palette = [c for c in range(1, 10) if c != int(col)]
    for rr, cc in positions:
        a[rr, cc] = int(np.random.choice(palette))
    return a, n_noise


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    h = int(np.random.randint(18, 30))
    w = int(np.random.randint(18, 30))
    grid = np.full((h, w), bg, dtype=int)
    n_frames = int(np.random.randint(1, 5))
    n_solids = int(np.random.randint(1, 5))

    frames_raw: list[tuple[int, int, np.ndarray]] = []
    solids_clean: list[tuple[int, int, np.ndarray]] = []
    solids_raw: list[tuple[int, int, np.ndarray]] = []
    total_noise = 0

    try:
        object_colors = list(np.random.permutation(np.arange(1, 10))[: n_frames + n_solids])
        color_idx = 0

        for _ in range(n_frames):
            col = int(object_colors[color_idx])
            color_idx += 1
            patch = _make_frame(col, bg)
            r0, c0 = random_free_location(
                grid, patch, transparent=bg, border=1, padding=1, padding_connectivity=4
            )
            paste(grid, patch, r0, c0, transparent=bg)
            frames_raw.append((int(r0), int(c0), patch.copy()))

        for _ in range(n_solids):
            col = int(object_colors[color_idx])
            color_idx += 1
            raw_patch, noise_count = _make_noisy_solid(col, bg)
            r0, c0 = random_free_location(
                grid, raw_patch, transparent=bg, border=1, padding=1, padding_connectivity=4
            )
            paste(grid, raw_patch, r0, c0)
            solids_raw.append((int(r0), int(c0), raw_patch.copy()))
            solids_clean.append((int(r0), int(c0), _make_solid(col)))
            total_noise += int(noise_count)

        if total_noise < n_solids:
            return None

        expected = _compose_output(bg, frames_raw, solids_clean)
        if not np.array_equal(main(grid), expected):
            return None

        old_style_expected = _compose_output(bg, frames_raw, solids_raw)
        if np.array_equal(expected, old_style_expected):
            return None

        return grid
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 300 attempts")
