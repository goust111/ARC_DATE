# Combo variant of 6e453dd6 (index 22)
# combo_spec: D1/0+D2/3
# Name: 反向平移+射线先垂直后转弯

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


# Design analysis for combo D1/0 + D2/3
# --------------------------------------
# a) Concrete behavior and trace
# - Start from the D1 mirror geometry: sprites live on the right of the grey separator and slide
#   left until their separator-facing wall is flush at global column `sep+1`.
# - Original task after sliding:
#     triggering hole rows = every row of an enclosed bg hole touching pane-local col 1
#     output = fill the entire left pane red on those same rows
# - Fused task after sliding:
#     triggering object = the same kind of enclosed hole touching pane-local col 1
#     emitted path = one bent ray per triggering hole, not one full-width row per hole row
#     ray start = the hole's top row on the left side of the separator
#     ray leg 1 = a vertical segment downward in column `sep-1`
#     leg-1 length = the hole height
#     turn point = the bottom row of that hole
#     ray leg 2 = a horizontal segment leftward from the turn point to column 0
# - Small trace:
#     if a slid right-side shell has a hole spanning rows 7..9 (height 3),
#     original D1 output would paint rows 7, 8, 9 fully red on the left.
#     This combo instead paints:
#       column sep-1 at rows 7, 8, 9
#       plus row 9 from col 0 through sep-1
#   The causal variable is still "hole adjacent to separator", but the visible signature is now an
#   L-shaped path whose vertical leg length equals the hole height.
#
# b) Complexity calibration
# - The generator keeps the original sprite-count regime and still mixes emitters with C/L/solid
#   distractors, so the puzzle is not reduced to "every shell emits".
# - Emitter hole heights vary over a small repeated catalog {2, 3, 4}. That is enough to make the
#   height-controlled turn obvious across a few examples without turning the task into a large codebook.
# - Left pane width is widened slightly to 3-5 columns so the horizontal leg is always visible; with
#   the original 2-column ray pane, the turned path would look too close to a trivial stub.
#
# c) Implicit-assumption audit
# - `_find_separator_col` and `_background_color` still hold.
# - Every original "active pane is left" assumption from 6e453dd6 must flip to the right pane:
#   movement happens in `out[:, sep+1:]`, trigger holes are searched there, and rays are painted in
#   `out[:, :sep]`.
# - The old emission step `out[r, :sep] = red` is invalid under D2/3 because it hard-codes both the
#   path shape and the set of painted rows. The new solver must work hole-by-hole, compute height, and
#   paint a two-segment path.
# - The old uniqueness assertion on trigger rows no longer applies because one hole now contributes a
#   whole vertical segment plus one shelf row. Overlap is legal as long as the overlapping pixels are
#   all red; the solver therefore allows repeated writes to already-red cells.
#
# d) Placement legality
# - Sprites remain in disjoint row bands, so they never merge during right-to-left unsliding or the
#   final leftward slide.
# - Output is still generated first: sprites are stamped flush against the separator on the right, the
#   bent rays are painted on the left, then each sprite is shifted right to make the input.
# - Because the vertical leg is anchored to the hole's own row span, it never needs extra top/bottom
#   margin beyond the sprite band itself; no beam can leave the grid vertically.
#
# e) Visual observability
# - A human can see two coupled changes at once: sprites begin on the right and end near the separator,
#   and the red signal is no longer a set of full-width rows but an L-shaped trace hugging the divider.
# - The variable that controls the new path, hole height, is directly visible in the black shell and is
#   repeated by the red vertical leg right across the separator.
# - The generator rejects layouts where multiple emitters all have the same height when there is room for
#   variation, so the key control variable is exercised in typical samples.
#
# f) Visual cleanliness
# - The separator still cleanly isolates black sprite motion on the right from red ray drawing on the left.
# - Only one output color is used for rays, and overlaps happen only among red pixels, so the left pane
#   stays legible even when multiple emitters contribute nearby L-shapes.
#
# g) Inferability of the change
# - No hidden color-ID logic is used. Grey stays the divider, black stays the moving sprite, magenta is
#   the background, and red is the emitted path.
# - The only new inference is geometric: first follow the divider direction for a length equal to the hole
#   height, then turn left to the border.
#
# h) New solver assumptions and generator guarantees
# - Solver assumptions:
#   * all black components lie strictly to the right of the separator in the input;
#   * the left pane is pure background before ray emission;
#   * emitting holes are enclosed bg components touching pane-local column 1 after sliding.
# - Generator guarantees:
#   * all sprites are placed only in the right pane;
#   * the left pane is untouched in inputs;
#   * emitter patches are simple mirrored shells with one clean separator-facing hole;
#   * the output-first renderer paints bent rays exactly from those hole geometries.


BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
LEFT_W_MIN, LEFT_W_MAX = 3, 5
RIGHT_W_MIN, RIGHT_W_MAX = 5, 10

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

EMITTER_H_MIN, EMITTER_H_MAX = 4, 6      # hole heights 2..4
DISTRACTOR_H_MIN, DISTRACTOR_H_MAX = 2, 6

EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

MAX_RETRIES = 400


def _components4(mask: np.ndarray):
    return components_mask(mask, connectivity=4)


def _find_separator_col(g: np.ndarray, sep_color: int = SEP_COLOR) -> int:
    _, w = g.shape
    cols = [c for c in range(w) if np.all(g[:, c] == sep_color)]
    assert len(cols) == 1, f"expected a unique full separator column, got {cols}"
    return int(cols[0])


def _background_color(g: np.ndarray, sep_col: int, sep_color: int = SEP_COLOR) -> int:
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


def _mirror_patch(patch: np.ndarray) -> np.ndarray:
    return patch[:, ::-1].copy()


def _slide_right_pane_left(right: np.ndarray, bg: int) -> np.ndarray:
    pane = right.copy()
    comps = _components4(pane == BLACK)
    pane[pane == BLACK] = bg
    for comp in comps:
        min_c = min(c for _, c in comp)
        for r, c in comp:
            nc = c - min_c
            assert 0 <= nc < pane.shape[1], f"shifted pixel {(r, c)} left the pane"
            pane[r, nc] = BLACK
    return pane


def _emitting_hole_infos(right: np.ndarray, bg: int) -> list[dict[str, int]]:
    h, pane_w = right.shape
    infos = []
    for pts in _components4(right == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == pane_w - 1 for r, c in pts):
            continue
        cols = {int(c) for _, c in pts}
        if 1 not in cols:
            continue
        rows = sorted({int(r) for r, _ in pts})
        hole_h = len(rows)
        top = rows[0]
        bottom = top + hole_h - 1
        assert bottom == rows[-1], f"hole rows must be contiguous, got {rows}"
        infos.append({"top": top, "bottom": bottom, "height": hole_h})
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide right-side black sprites into the grey divider, then emit leftward bent rays from enclosed holes.

    Input structure:
    - A full-height grey (5) separator column splits a narrow left ray pane from a wider right sprite pane.
    - Black connected components appear only on the right over one background color, while the entire left pane starts empty.

    Rule:
    - Translate each right-side black connected component leftward until its separator-facing edge is flush with the divider.
    - In that shifted right pane, find enclosed background holes that touch the inner column immediately next to the separator-facing wall.
    - For each such hole, draw one red ray on the left: start beside the separator at the hole's top row, move vertically downward for a distance equal to the hole height, then turn 90 degrees and continue horizontally left to the grid edge.

    Key insight: This is not just a mirrored version of the original task. The same separator-adjacent hole still decides whether emission happens, but the output path is now a two-stage shape whose vertical leg explicitly encodes the hole's height before the ray turns left.
    """
    g = np.asarray(grid, dtype=int)
    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()
    right = _slide_right_pane_left(out[:, sep + 1 :], bg=bg)
    out[:, sep + 1 :] = right

    for info in _emitting_hole_infos(right, bg=bg):
        top = info["top"]
        bottom = info["bottom"]
        height = info["height"]

        vertical = out[top : top + height, sep - 1]
        assert np.all((vertical == bg) | (vertical == RED)), "vertical leg must overwrite only background/red"
        out[top : top + height, sep - 1] = RED

        shelf = out[bottom, :sep]
        assert np.all((shelf == bg) | (shelf == RED)), "horizontal leg must overwrite only background/red"
        out[bottom, :sep] = RED

    return out


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n: int) -> int:
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters: int, n_total: int) -> int:
    return n_emitters * EMITTER_H_MIN + (n_total - n_emitters) * DISTRACTOR_H_MIN + (n_total + 1) * MIN_BAND_GAP


def _choose_grid_dims() -> tuple[int, int, int, int]:
    h = random.randint(H_MIN, H_MAX)
    left_w = random.randint(LEFT_W_MIN, LEFT_W_MAX)
    valid_right_ws = [
        right_w
        for right_w in range(RIGHT_W_MIN, RIGHT_W_MAX + 1)
        if W_MIN <= left_w + 1 + right_w <= W_MAX
    ]
    right_w = random.choice(valid_right_ws)
    sep = left_w
    w = left_w + 1 + right_w
    return h, w, sep, right_w


def _choose_emitter_flags(h: int) -> list[bool]:
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


def _sprite_bands(h: int, flags: list[bool]) -> tuple[list[tuple[int, int]], list[int]] | None:
    min_heights = [EMITTER_H_MIN if flag else DISTRACTOR_H_MIN for flag in flags]
    max_heights = [EMITTER_H_MAX if flag else DISTRACTOR_H_MAX for flag in flags]

    base_gaps = [MIN_BAND_GAP] * (len(flags) + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    extra_cap = sum(hi - lo for lo, hi in zip(min_heights, max_heights))
    extra = random.randint(0, min(extra_cap, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i, (cur, hi) in enumerate(zip(heights, max_heights)) if cur < hi]
        heights[random.choice(choices)] += 1

    if sum(flag for flag in flags) >= 2:
        emitter_heights = [ht for ht, flag in zip(heights, flags) if flag]
        if len(set(emitter_heights)) < 2:
            return None

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), len(base_gaps)))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands, heights


def _emitter_patch(sh: int, max_sw: int) -> tuple[np.ndarray, dict[str, int]]:
    hole_h = sh - 2
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = hole_w + 2
    assert sw <= max_sw, f"emitter width {sw} exceeds allowance {max_sw}"

    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    patch[:, -1] = BLACK
    return patch, {"hole_top": 1, "hole_height": hole_h}


def _distractor_patch_core(sh: int, max_sw: int) -> np.ndarray:
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


def _distractor_patch(sh: int, max_sw: int) -> np.ndarray:
    return _mirror_patch(_distractor_patch_core(sh, max_sw))


def _render_output(
    h: int,
    w: int,
    sep: int,
    bands: list[tuple[int, int]],
    heights: list[int],
    flags: list[bool],
    right_w: int,
) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]]]:
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites: list[tuple[np.ndarray, int, int]] = []
    max_sw = min(SPRITE_W_MAX, right_w - 1)
    assert max_sw >= 3, "right pane must allow both emitters and rightward unsliding"

    for (r0, _), sh, is_emitter in zip(bands, heights, flags):
        if is_emitter:
            patch, meta = _emitter_patch(sh, max_sw=max_sw)
        else:
            patch = _distractor_patch(sh, max_sw=max_sw)
            meta = None

        c0 = sep + 1
        paste(out, patch, r0, c0, transparent=BG)

        if meta is not None:
            hole_top = meta["hole_top"]
            hole_h = meta["hole_height"]
            top = r0 + hole_top
            bottom = top + hole_h - 1
            out[top : top + hole_h, sep - 1] = RED
            out[bottom, :sep] = RED

        sprites.append((patch, r0, c0))

    return out, sprites


def _unslide_sprites(h: int, w: int, sep: int, sprites: list[tuple[np.ndarray, int, int]]) -> np.ndarray | None:
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    n = len(sprites)
    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, (patch, r0, c0) in enumerate(sprites):
        sw = patch.shape[1]
        max_shift = w - (c0 + sw)
        if max_shift < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, max_shift)
        paste(inp, patch, r0, c0 + delta, transparent=BG)
    return inp


def _combo_metrics(inp: np.ndarray, out: np.ndarray) -> dict[str, int]:
    sep = _find_separator_col(inp, sep_color=SEP_COLOR)
    bg = _background_color(inp, sep_col=sep, sep_color=SEP_COLOR)
    right = _slide_right_pane_left(inp[:, sep + 1 :], bg=bg)
    infos = _emitting_hole_infos(right, bg=bg)
    heights = [info["height"] for info in infos]
    partial_rows = int(
        sum(
            np.any(out[r, :sep] == RED) and not np.all(out[r, :sep] == RED)
            for r in range(out.shape[0])
        )
    )
    return {
        "emitters": len(infos),
        "distinct_heights": len(set(heights)),
        "vertical_red": int(np.sum(out[:, sep - 1] == RED)),
        "left_red_rows": int(np.sum(np.any(out[:, :sep] == RED, axis=1))),
        "partial_rows": partial_rows,
    }


def _gen_attempt() -> np.ndarray | None:
    h, w, sep, right_w = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    layout = _sprite_bands(h, flags)
    if layout is None:
        return None
    bands, heights = layout

    out, sprites = _render_output(h, w, sep, bands, heights, flags, right_w)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    try:
        if not np.array_equal(main(inp), out):
            return None
        metrics = _combo_metrics(inp, out)
        if metrics["emitters"] < 1:
            return None
        if metrics["partial_rows"] < 1:
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
