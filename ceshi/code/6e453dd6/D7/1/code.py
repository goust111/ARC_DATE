# Variant of 6e453dd6
# Dimension: D7 — 射线终止条件 (dst)
# Change: 射线延伸到网格右边缘（grid.right_edge）——填充从sep+1到w-1的所有列... → 射线长度等于洞的属性值

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        sys.path.append(str(_arc_utils.parent))
        break

from arc_utils import components_mask, paste


# Design analysis for the D7 variant
# - Concrete behavior: the left half still contains black connected components that slide right until
#   their right edge becomes flush with the grey separator. After sliding, only enclosed background
#   holes that touch the separator-side interior column (sep-2) emit. The D7 change applies only to
#   the emission stage: each emitting row now gets a short red segment whose length equals that hole's
#   height, not a full sweep to the right border.
# - Example trace: if a sprite becomes a 5x4 rectangular shell after sliding, its enclosed hole spans
#   rows 1..3 inside the shell, so the hole height is 3. The original task would paint every cell in
#   those rows on the right pane; the variant paints exactly three red cells per triggering row, leaving
#   trailing background visible.
# - Complexity calibration: the key count is deliberately small and repeated. Emitter holes have heights
#   1-5 and the right pane is widened to 4-7 cells so different beam lengths are visually distinguishable.
#   Distractor sprites remain open or solid shapes, so "hole vs non-hole" is still part of the puzzle.
# - Implicit-assumption audit for the original main():
#   * Sliding code assumes every black component can move independently within the left pane. This remains
#     true because generator row bands keep components vertically disjoint and entirely left of sep.
#   * Hole detection assumes true holes are background components that avoid the outer left-pane boundary.
#     That still holds because emitters are closed shells and distractors are open/solid.
#   * The original row-filling step relied on a constant destination `sep+1:w`; under D7 that becomes wrong
#     because it would erase the new trailing background. The new solver therefore stores a per-hole length
#     and fills only `sep+1 : sep+1+hole_height`.
#   * The original assertion that each trigger row appears once is no longer safe if future inputs ever had
#     multiple trigger holes with the same row. The new solver records per-row beam lengths and asserts only
#     that duplicate rows agree on the same encoded length.
# - Placement legality: emitters are generated with height <= right_w so their hole height is <= right_w-1,
#   guaranteeing the short beam fits fully in the right pane and usually stops before the border. Row bands
#   are still separated by at least one background row, so sprites never overlap before or after unsliding.
# - Visual observability: a human can compare the hole's vertical extent against the horizontal red segment.
#   Because generator enforces `beam_length < right_w`, the variant cannot silently collapse back to the
#   original "paint all the way to the edge" behavior on its emitting rows.
# - Visual cleanliness: only one red layer exists on the right pane, beams never overlap across sprite bands,
#   and the separator remains a single clean vertical divider.


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = 5) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = 5) -> int:
    h, w = g.shape
    border = []
    for c in range(w):
        if c != sep_col and g[0, c] != sep_color:
            border.append(int(g[0, c]))
        if c != sep_col and g[h - 1, c] != sep_color:
            border.append(int(g[h - 1, c]))
    for r in range(h):
        if g[r, 0] != sep_color:
            border.append(int(g[r, 0]))
        if g[r, w - 1] != sep_color:
            border.append(int(g[r, w - 1]))
    assert border, "expected non-separator border pixels to infer the background"
    vals, counts = np.unique(np.array(border, dtype=int), return_counts=True)
    winners = vals[counts == counts.max()]
    assert len(winners) == 1, f"background color must be unique on the border, got {winners.tolist()}"
    return int(winners[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the grey divider, then fire short red beams whose length is encoded by hole height.

    Input structure:
    - A single full-height grey (5) separator column splits the board into a left sprite pane and an empty right pane.
    - Black connected components sit on the left over one background color, while the right pane starts as pure background.

    Rule:
    - Move every black connected component horizontally so its rightmost black pixel becomes flush with the separator.
    - In the slid layout, find enclosed background holes; only holes that touch the separator-side interior column emit.
    - For each row occupied by an emitting hole, paint a red segment immediately to the right of the separator, with length equal to that hole's height.

    Key insight: The same separator-adjacent hole rows still determine where beams appear, but the hole's vertical size now determines how far each beam travels.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    left = out[:, :sep].copy()
    comps = _components4(left == black)
    left[left == black] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = black

    out[:, :sep] = left

    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_col = sep - 2
    beam_by_row: dict[int, int] = {}
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        hole_rows = sorted({int(r) for r, _ in pts})
        ray_len = len(hole_rows)
        assert ray_len >= 1, "emitting holes must span at least one row"
        for r in hole_rows:
            prev = beam_by_row.get(r)
            assert prev in (None, ray_len), f"conflicting beam lengths on row {r}: {prev} vs {ray_len}"
            beam_by_row[r] = ray_len

    for r, ray_len in sorted(beam_by_row.items()):
        end = min(w, sep + 1 + ray_len)
        assert end > sep + 1, f"beam on row {r} must paint at least one cell"
        assert np.all(out[r, sep + 1 : end] == bg), f"beam path on row {r} must be background before emission"
        out[r, sep + 1 : end] = red

    return out


BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 19
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 4, 7

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

MAX_RETRIES = 400


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n):
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters, n_total):
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(h, is_emitter, right_w):
    n = len(is_emitter)
    min_heights = [3 if e else 2 for e in is_emitter]
    max_emitter_h = min(SPRITE_H_MAX, right_w + 1)
    max_heights = [max_emitter_h if e else SPRITE_H_MAX for e in is_emitter]
    if any(lo > hi for lo, hi in zip(min_heights, max_heights)):
        return None

    base_gaps = [MIN_BAND_GAP] * (n + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    extra_cap = sum(hi - lo for lo, hi in zip(min_heights, max_heights))
    extra = random.randint(0, min(extra_cap, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i, (x, hi) in enumerate(zip(heights, max_heights)) if x < hi]
        i = random.choice(choices)
        heights[i] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _emitter_patch(sh, max_sw):
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    return patch


def _distractor_patch(sh, max_sw):
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), BLACK, dtype=int)

    min_sw = 3 if kind.startswith("c_") or sh == 2 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    if kind == "c_left":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, -1] = BLACK
    elif kind == "c_right":
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[:, 0] = BLACK
    else:
        patch[:, -1] = BLACK
        if random.random() < 0.5:
            patch[0, :] = BLACK
        else:
            patch[-1, :] = BLACK
    return patch


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep, right_w


def _choose_emitter_flags(h):
    valid_counts = [
        n
        for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
        if _min_rows_needed(_min_emitters_for(n), n) <= h
    ]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = min_em
    for k in range(min_em + 1, n + 1):
        if _min_rows_needed(k, n) <= h:
            max_em = k
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _render_output(h, w, sep, right_w, bands, flags):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            ray_len = sh - 2
            assert 1 <= ray_len < right_w, f"emitter height {sh} must encode a short beam within right width {right_w}"
            out[r0 + 1 : r0 + sh - 1, sep + 1 : sep + 1 + ray_len] = RED
        sprites.append((patch, r0, c0))
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _gen_attempt() -> np.ndarray | None:
    h, w, sep, right_w = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags, right_w)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, right_w, bands, flags)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 6e453dd6 D7 input")
