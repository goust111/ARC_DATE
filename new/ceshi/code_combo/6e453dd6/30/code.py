# Combo variant of 6e453dd6 (index 30)
# combo_spec: D15/0+D24/0
# Name: 小网格+统一尺寸精灵

from __future__ import annotations

# Design analysis for combo D15/0 + D24/0
# a) Concrete behavior and trace:
#    - The base task already has the core rule we keep: black sprites on the left slide
#      horizontally until flush with the grey separator, then only enclosed holes that
#      touch the separator-side inner column emit red rays on the right.
#    - D15/0 changes the scene scale: the whole board is now compact, only 8..12 rows
#      by 8..12 cols, so the large 2..8 by 2..6 sprite variety from the source task is
#      no longer appropriate.
#    - D24/0 changes the sprite family: every sprite uses the same 3x3 bbox. The only
#      meaningful difference between sprites is their internal black/bg pattern.
#    - Concrete trace:
#      * Suppose a 9x10 grid has separator at col 5.
#      * A 3x3 ring sprite at rows 1..3 starts at cols 0..2 and slides to cols 2..4.
#      * Its single hole is the center cell, so after sliding that hole lands at col 3,
#        exactly sep-2, and row 2 emits a red ray across the right pane.
#      * A 3x3 C-shape at rows 5..7 slides the same way, but its center bg touches the
#        bbox boundary, so it is not an enclosed hole and emits nothing.
#      * Relative to the original task, the logical transformation is unchanged, but
#        the visual evidence is now "same-size teaching icons: ring fires, open icon does not".
# b) Complexity calibration:
#    - Uniform 3x3 sprites are simpler, but not trivialized: the generator always mixes
#      emitter and non-emitter icons, so the solver must still read enclosure rather than
#      a fixed row index or a color cue.
#    - The distractor vocabulary is intentionally small (C/L-like 3x3 outlines). This
#      gives repeated support across samples instead of too many one-off special cases.
#    - The compact board keeps 2 or 3 row-separated sprites. That is enough variety to
#      show the rule while staying visually legible in 2..4 examples.
# c) Implicit assumptions review against the original main():
#    - `_find_separator_col(...)` still assumes one full-height separator column. Kept.
#    - `_background_color(...)` still assumes the border is dominated by background. Kept;
#      generator preserves a pure-bg right pane and bg-heavy outer border.
#    - The original hole finder scanned the whole left pane after sliding. That remains
#      legal, but in this combo a per-component bbox analysis is cleaner because every
#      sprite has the same compact footprint and row bands make components independent.
#    - The original `trigger_col = sep - 2` assumption still holds exactly. For the 3x3
#      ring emitter, the enclosed center cell always lands at sep-2 after the flush slide.
#    - The original ray fill assumes the right pane is empty before painting. Kept;
#      generator guarantees the right pane is all background in the input.
# d) Placement legality:
#    - Every sprite occupies a disjoint 3-row band, with at least one full bg row between
#      neighboring bands. This prevents overlap both before and after sliding.
#    - With separator in cols 5..7, each 3x3 sprite has at least two legal horizontal
#      placements on the left, so the input can visibly differ from the flush output.
#    - All placements are stamped with transparent background, so bbox size stays 3x3
#      while black pixels never overlap or leave the left pane.
# e) Visual observability:
#    - A human can directly see both fused dimensions:
#      * the board is visibly smaller than the source-task style
#      * all black sprites are the same 3x3 size, so only hole-vs-open structure matters
#    - Because emitter and distractor icons share size, the firing condition is easier to
#      infer from shape structure alone: closed center fires, open center does not.
# f) Visual cleanliness:
#    - Black icons are confined to separated left-side bands, the grey separator is a
#      single full column, and red only appears on the empty right pane.
#    - No extra control marks or overlay colors are introduced, so the compact layout
#      stays readable instead of cluttered.
# g) Change inferability:
#    - No hidden numeric/color ordering is used. The fusion is entirely visible in the
#      geometry: smaller boards and identical sprite bboxes across the whole sample set.
#    - Repeated canonical 3x3 patterns provide enough structural evidence to infer the
#      enclosure condition from a few training pairs.
# h) Quantitative anti-degeneration check:
#    - Generation rejects samples unless all of these hold:
#      * 8 <= H,W <= 12
#      * all sprite bboxes are exactly one shared shape, here 3x3
#      * there is at least one emitter and at least one distractor
#      * at least one sprite actually slides left-to-right in the input/output pair
#    - These metrics are later easy to measure over many samples to confirm the combo is
#      not silently falling back to the original broad-size regime.
# i) New solver assumptions and generator guarantees:
#    - Solver assumption: all left-side components share one bbox shape.
#      Generator guarantee: every sprite patch is drawn from a 3x3 template family.
#    - Solver assumption: each moved component can be analyzed independently for holes.
#      Generator guarantee: sprites occupy disjoint row bands, so components never merge.
#    - Solver assumption: trigger rows are unique.
#      Generator guarantee: each 3x3 emitter has a single-cell hole on its middle row,
#      and row bands keep different emitters on different rows.

from pathlib import Path
import random
import sys

import numpy as np


for _parent in Path(__file__).resolve().parents:
    _arc_utils = _parent / "solvers" / "arc_utils.py"
    if _arc_utils.exists():
        if str(_arc_utils.parent) not in sys.path:
            sys.path.insert(0, str(_arc_utils.parent))
        break

from arc_utils import components_mask, paste


def _components4(mask: np.ndarray):
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


def _bbox_of(pts) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in pts]
    cols = [int(c) for _, c in pts]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_shape(pts) -> tuple[int, int]:
    r0, c0, r1, c1 = _bbox_of(pts)
    return int(r1 - r0 + 1), int(c1 - c0 + 1)


def _slid_points(comp, delta: int):
    return [(int(r), int(c + delta)) for r, c in comp]


def _trigger_rows_for_component(pts, sep: int) -> list[int]:
    r0, c0, r1, c1 = _bbox_of(pts)
    box_h = r1 - r0 + 1
    box_w = c1 - c0 + 1
    occ = np.zeros((box_h, box_w), dtype=bool)
    for r, c in pts:
        occ[int(r) - r0, int(c) - c0] = True

    rows = set()
    for hole in _components4(~occ):
        if any(lr == 0 or lr == box_h - 1 or lc == 0 or lc == box_w - 1 for lr, lc in hole):
            continue
        global_hole = [(r0 + int(lr), c0 + int(lc)) for lr, lc in hole]
        if any(c == sep - 2 for _, c in global_hole):
            rows.update(int(r) for r, _ in global_hole)
    return sorted(rows)


def _slide_components(left: np.ndarray, sep: int, black: int, bg: int):
    comps = _components4(left == black)
    assert comps, "expected at least one black sprite"

    bbox_shapes = {_bbox_shape(comp) for comp in comps}
    assert len(bbox_shapes) == 1, f"expected uniform sprite bbox, got {sorted(bbox_shapes)}"

    left_out = np.full(left.shape, bg, dtype=int)
    moved_comps = []
    target_right = sep - 1
    for comp in comps:
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        moved = _slid_points(comp, delta)
        for r, c in moved:
            assert 0 <= c < sep, f"shifted pixel {(r, c)} left the pane"
            left_out[r, c] = black
        moved_comps.append(moved)
    return left_out, moved_comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide same-size black sprites to the grey divider, then fire red rays from compact enclosed holes.

    Input structure:
    - A single full-height grey (5) separator column splits a small board into a left sprite pane and an empty right pane.
    - The left pane contains several black connected sprites, and in this teaching variant every sprite shares the same 3x3 bounding box on one common background color.

    Rule:
    - Move each black connected component horizontally until its rightmost black pixel is flush with the separator.
    - In that moved layout, inspect each sprite's own bbox and find enclosed background holes that reach the inner column just left of the separator.
    - For every row occupied by such a triggering hole, paint a red ray across the right pane while leaving non-triggering same-size sprites as simple moved black shapes.

    Key insight: Size is no longer informative because every sprite is equally large; the deciding feature is whether the slid 3x3 icon truly encloses a hole against the separator-side inner face.
    """
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    left_out, moved_comps = _slide_components(g[:, :sep], sep, black=black, bg=bg)

    out = g.copy()
    out[:, :sep] = left_out

    trigger_rows = []
    for moved in moved_comps:
        trigger_rows.extend(_trigger_rows_for_component(moved, sep))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"

    for r in trigger_rows:
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = red

    return out


# --------------- generator (combo D15/0 + D24/0) ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 8, 12
W_MIN, W_MAX = 8, 12
SEP_MIN, SEP_MAX = 5, 7
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

SPRITE_H = 3
SPRITE_W = 3
MIN_BAND_GAP = 1
MAX_RETRIES = 400


EMITTER_PATCH = np.array(
    [
        [BLACK, BLACK, BLACK],
        [BLACK, BG, BLACK],
        [BLACK, BLACK, BLACK],
    ],
    dtype=int,
)


DISTRACTOR_PATCHES = [
    np.array(
        [
            [BLACK, BLACK, BLACK],
            [BG, BG, BLACK],
            [BLACK, BLACK, BLACK],
        ],
        dtype=int,
    ),
    np.array(
        [
            [BLACK, BLACK, BLACK],
            [BLACK, BG, BG],
            [BLACK, BLACK, BLACK],
        ],
        dtype=int,
    ),
    np.array(
        [
            [BLACK, BLACK, BLACK],
            [BG, BG, BLACK],
            [BG, BG, BLACK],
        ],
        dtype=int,
    ),
    np.array(
        [
            [BG, BG, BLACK],
            [BG, BG, BLACK],
            [BLACK, BLACK, BLACK],
        ],
        dtype=int,
    ),
]


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _choose_dims():
    h = random.randint(H_MIN, H_MAX)
    valid = []
    for sep in range(SEP_MIN, SEP_MAX + 1):
        for right_w in range(RIGHT_W_MIN, RIGHT_W_MAX + 1):
            w = sep + 1 + right_w
            if W_MIN <= w <= W_MAX:
                valid.append((w, sep))
    w, sep = random.choice(valid)
    return h, w, sep


def _choose_n_sprites(h: int) -> int:
    # Favor the 3-sprite branch when the compact grid is tall enough so the
    # training set does not collapse into the simpler 2-sprite layout.
    if h >= 11 and random.random() < 0.75:
        return 3
    return 2


def _sprite_bands(h: int, n: int):
    base_used = n * SPRITE_H + (n - 1) * MIN_BAND_GAP
    if base_used > h:
        return None
    extras = h - base_used
    base_gaps = [0] + [MIN_BAND_GAP] * (n - 1) + [0]
    gaps = [a + b for a, b in zip(base_gaps, _scatter(extras, n + 1))]

    bands = []
    r = gaps[0]
    for i in range(n):
        bands.append((r, SPRITE_H))
        r += SPRITE_H + gaps[i + 1]
    return bands


def _choose_flags(n: int) -> list[bool]:
    if n == 2:
        flags = [True, False]
    else:
        # Cover both "one emitter" and "two emitters" 3-sprite cases regularly.
        flags = [True, True, False] if random.random() < 0.5 else [True, False, False]
    random.shuffle(flags)
    return flags


def _render_output(h: int, w: int, sep: int, bands, flags):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    flush_c0 = sep - SPRITE_W
    assert flush_c0 >= 0, "left pane too narrow for 3x3 sprites"

    sprites = []
    for (r0, _), is_emitter in zip(bands, flags):
        patch = EMITTER_PATCH.copy() if is_emitter else random.choice(DISTRACTOR_PATCHES).copy()
        paste(out, patch, r0, flush_c0, transparent=BG)
        if is_emitter:
            out[r0 + 1, sep + 1 :] = RED
        sprites.append((patch, int(r0), int(flush_c0), bool(is_emitter)))
    return out, sprites


def _unslide_sprites(h: int, w: int, sep: int, sprites):
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    movable = [i for i, (_, _, c0, _) in enumerate(sprites) if c0 > 0]
    assert movable, "expected at least one movable sprite"
    forced = random.choice(movable)

    moved_any = False
    for i, (patch, r0, c0, _) in enumerate(sprites):
        if i == forced:
            delta = random.randint(1, c0)
        else:
            delta = random.randint(0, c0)
        moved_any = moved_any or delta > 0
        paste(inp, patch, r0, c0 - delta, transparent=BG)

    assert moved_any, "expected at least one sprite to move"
    return inp


def _count_emitters_after_slide(inp: np.ndarray) -> int:
    sep = _find_separator_col(inp, sep_color=SEP_COLOR)
    bg = _background_color(inp, sep_col=sep, sep_color=SEP_COLOR)
    _, moved_comps = _slide_components(inp[:, :sep], sep, black=BLACK, bg=bg)
    return sum(1 for moved in moved_comps if _trigger_rows_for_component(moved, sep))


def fusion_metrics(inp: np.ndarray) -> dict:
    g = np.asarray(inp, dtype=int)
    h, w = g.shape
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    left = g[:, :sep]
    comps = _components4(left == BLACK)
    bbox_shapes = sorted({_bbox_shape(comp) for comp in comps})
    emitters = _count_emitters_after_slide(g)
    shifted = sum(1 for comp in comps if max(int(c) for _, c in comp) < sep - 1)
    return {
        "shape": (int(h), int(w)),
        "bbox_shapes": bbox_shapes,
        "n_sprites": len(comps),
        "n_emitters": int(emitters),
        "n_distractors": int(len(comps) - emitters),
        "shifted_sprites": int(shifted),
        "is_small_grid": bool(H_MIN <= h <= H_MAX and W_MIN <= w <= W_MAX),
        "has_uniform_bbox": bool(len(bbox_shapes) == 1 and bbox_shapes[0] == (SPRITE_H, SPRITE_W)),
    }


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_dims()
    n = _choose_n_sprites(h)
    bands = _sprite_bands(h, n)
    if bands is None:
        return None

    flags = _choose_flags(n)
    out, sprites = _render_output(h, w, sep, bands, flags)
    inp = _unslide_sprites(h, w, sep, sprites)

    metrics = fusion_metrics(inp)
    if not metrics["is_small_grid"]:
        return None
    if not metrics["has_uniform_bbox"]:
        return None
    if metrics["n_emitters"] < 1 or metrics["n_distractors"] < 1:
        return None
    if metrics["shifted_sprites"] < 1:
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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")
