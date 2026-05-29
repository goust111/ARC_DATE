# Combo variant of 6e453dd6 (index 13)
# combo_spec: D2/2+D5/1
# Name: 四方向射线配宽松触发

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concretization + behavior trace
#    - Original task: slide every left black component until its right edge reaches the grey
#      separator, then only fully enclosed bg holes that touch `sep-2` emit. Each triggered
#      row paints a red strip only to the right.
#    - Combo task: sliding is unchanged, but triggering is widened from "closed hole" to
#      "separator-facing recess". Inside each slid component bbox, any bg region that reaches
#      the inner trigger face (`sep-2`) counts if that face is backed by a black right wall
#      and the same bg region also touches some other black edge (top, bottom, or left).
#      This keeps true holes active and additionally activates `C-left` and `L` recesses,
#      while excluding a plain vertical wall and `C-right` shapes that open away from the
#      separator.
#    - Small trace:
#        * Closed shell `000 / 0.0 / 000` flush to the separator still triggers.
#        * Open `C-left` shell `000 / ..0 / 000` used to fail because the bg region leaks
#          leftward; now it triggers because its `sep-2` cells touch the right wall and the
#          same recess also touches the top and bottom caps.
#        * Right-open `C-right` shell `000 / 0.. / 000` still does not trigger because its
#          `sep-2` cells do not sit directly against a black right wall.
#      After any trigger, the visible ray is a four-direction cross in the right chamber:
#      the triggered row becomes fully red, and a shared launch column also becomes fully red.
# b) Complexity calibration
#    - The source logic is broader than the original but still only has two visible families:
#      separator-facing recesses emit, away-facing/solid distractors do not.
#    - The path logic stays low-entropy: every trigger contributes a horizontal strip and all
#      triggers share one vertical launch column. This creates a stable "rows + one column"
#      pattern that is easy to compare across 2-4 examples.
#    - The generator enforces one open emitter, one closed emitter, and one distractor in
#      every sample so the combo does not collapse back to either single-variant rule.
# c) Implicit-assumption audit of `main`
#    - Separator detection and border-based bg inference still rely on the input having a
#      unique full-height grey column and an otherwise bg border. The generator preserves this.
#    - Sliding black components assumes horizontal motion cannot merge separate objects. The
#      generator keeps sprite row bands disjoint, so shifted components remain disjoint.
#    - Loose trigger detection assumes each slid component is flush with the separator and is
#      interpreted inside its own bbox. The generator always renders output in that flush form.
#    - Ray painting now allows overlaps: several trigger rows can share one vertical column.
#      Therefore cells are only required to be bg or already red before repainting.
# d) Placement legality
#    - Every sprite is generated within one vertical band, bands are separated by at least one
#      full bg row, and widths stay at most `sep-1`, so unsliding always has legal leftward
#      room and no sprite overlap occurs.
# e) Visual observability
#    - Open emitters (`C-left`, `L`) are visibly different from closed shells in the left pane,
#      and they still emit after sliding. That makes the relaxed trigger condition readable.
#    - The extra full-height red column makes the D2 path change visible even when only one row
#      fires.
# f) Visual cleanliness
#    - Right-side overlaps collapse to one color (red), so the output remains legible rather
#      than noisy. Using one shared launch column avoids a thicket of competing vertical bars.
# g) Inferability of the change
#    - No hidden numeric ordering is used. The rule can be inferred from geometry alone:
#      move-to-separator, detect separator-facing recesses, then emit row strips plus one
#      column in the right chamber.
#    - The generator rejects samples whose outputs match either the D2-only or D5-only
#      single-variant baselines, so the fused behavior is always visible.


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


def _slide_left_components(left: np.ndarray, sep: int, bg: int, black: int) -> np.ndarray:
    slid = left.copy()
    comps = _components4(slid == black)
    slid[slid == black] = bg

    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            slid[r, nc] = black
    return slid


def _has_secondary_black_contact(patch: np.ndarray, pts, black: int) -> bool:
    h, w = patch.shape
    for r, c in pts:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1)):
            if 0 <= nr < h and 0 <= nc < w and patch[nr, nc] == black:
                return True
    return False


def _trigger_rows_from_left(left: np.ndarray, sep: int, bg: int, black: int) -> list[int]:
    trigger_rows = set()
    for comp in _components4(left == black):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        assert c1 == sep - 1, "slid components must be flush with the separator"
        if c1 - c0 + 1 < 2:
            continue

        patch = left[r0 : r1 + 1, c0 : c1 + 1]
        local_trigger = patch.shape[1] - 2
        for pts in _components4(patch == bg):
            port_rows = {
                pr
                for pr, pc in pts
                if pc == local_trigger and patch[pr, pc + 1] == black
            }
            if not port_rows:
                continue
            if not _has_secondary_black_contact(patch, pts, black):
                continue
            trigger_rows.update(r0 + pr for pr in port_rows)
    return sorted(trigger_rows)


def _closed_trigger_rows_from_left(left: np.ndarray, sep: int, bg: int) -> list[int]:
    holes = []
    h = left.shape[0]
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_rows = []
    trigger_col = sep - 2
    for pts in holes:
        if not any(c == trigger_col for _, c in pts):
            continue
        trigger_rows.extend(sorted({int(r) for r, _ in pts}))
    assert len(trigger_rows) == len(set(trigger_rows)), f"trigger rows must be unique, got {trigger_rows}"
    return trigger_rows


def _launch_col(sep: int, w: int) -> int:
    launch = min(sep + 2, w - 1)
    assert launch > sep, "right pane must contain a launch column"
    return int(launch)


def _paint_row_rays(out: np.ndarray, sep: int, trigger_rows: list[int], bg: int, red: int) -> None:
    for r in trigger_rows:
        seg = out[r, sep + 1 :]
        assert np.all((seg == bg) | (seg == red)), f"right half of row {r} must be free before ray emission"
        out[r, sep + 1 :] = red


def _paint_cross_rays(out: np.ndarray, sep: int, trigger_rows: list[int], bg: int, red: int) -> None:
    if not trigger_rows:
        return

    _, w = out.shape
    launch = _launch_col(sep, w)

    col_vals = out[:, launch]
    assert np.all((col_vals == bg) | (col_vals == red)), "launch column must be free before emission"
    out[:, launch] = red
    _paint_row_rays(out, sep, trigger_rows, bg, red)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites to the separator, then let separator-facing recesses emit four-direction rays.

    Input structure:
    - One full-height grey (5) separator column splits the grid into a left sprite field and a narrow right chamber.
    - The left field contains black connected components on a uniform background; the right chamber is empty background in the input.

    Rule:
    - Move each left-side black component horizontally until its rightmost black cell is flush with the separator.
    - In that slid state, inspect each component's bbox and treat any background recess that reaches the inner face next to the separator as a trigger, even if the recess is open like a C-shape or L-shape, provided it is backed by a black right wall and also touches some other black edge.
    - For every triggered row, paint the whole right-side row red and also paint a shared right-side launch column red from top to bottom, producing four-direction cross rays.

    Key insight: The motion step is the same as the original puzzle, but the source detector is looser: any separator-facing concavity can act as a nozzle, not just a fully enclosed hole, and every activation now adds both a horizontal strip and a vertical bar.
    """
    g = np.asarray(grid, dtype=int)
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = _slide_left_components(out[:, :sep], sep, bg, black)
    out[:, :sep] = left

    trigger_rows = _trigger_rows_from_left(left, sep, bg, black)
    _paint_cross_rays(out, sep, trigger_rows, bg, red)
    return out


# --------------- generator (combo variant) ---------------

BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

EXTRA_EMITTERS_PROB = 0.3
FLUSH_PROB = 0.35
OPEN_EMITTER_PROB = 0.7

MAX_RETRIES = 400


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n: int) -> int:
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters: int, n_total: int) -> int:
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(h: int, is_emitter: list[bool]):
    n = len(is_emitter)
    min_heights = [3 if e else 2 for e in is_emitter]
    base_gaps = [MIN_BAND_GAP] * (n + 1)
    height_cap = h - sum(base_gaps)
    if sum(min_heights) > height_cap:
        return None

    heights = min_heights[:]
    extra_cap = sum(SPRITE_H_MAX - x for x in heights)
    extra = random.randint(0, min(extra_cap, height_cap - sum(heights)))
    for _ in range(extra):
        choices = [i for i, x in enumerate(heights) if x < SPRITE_H_MAX]
        heights[random.choice(choices)] += 1

    gaps = [a + b for a, b in zip(base_gaps, _scatter(h - sum(heights) - sum(base_gaps), n + 1))]
    bands = []
    r = gaps[0]
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh + gaps[i + 1]
    return bands


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_emitter_flags(h: int) -> list[bool]:
    min_sprites = max(3, N_SPRITES_MIN)
    valid_counts = [n for n in range(min_sprites, N_SPRITES_MAX + 1) if _min_rows_needed(_min_emitters_for(n), n) <= h]
    n = random.choice(valid_counts)

    min_em = _min_emitters_for(n)
    max_em = min_em
    for k in range(min_em + 1, n + 1):
        if _min_rows_needed(k, n) <= h:
            max_em = k
    max_em = min(max_em, n - 1)

    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em

    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _closed_emitter_patch(sh: int, max_sw: int) -> np.ndarray:
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    left_wall = sw - hole_w - 2
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[1:-1, left_wall] = BLACK
    patch[1:-1, -1] = BLACK
    return patch


def _c_left_emitter_patch(sh: int, max_sw: int) -> np.ndarray:
    min_sw = 3 if max_sw >= 3 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, -1] = BLACK
    return patch


def _l_emitter_patch(sh: int, max_sw: int, top_bar: bool) -> np.ndarray:
    min_sw = 3 if max_sw >= 3 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[:, -1] = BLACK
    if top_bar:
        patch[0, :] = BLACK
    else:
        patch[-1, :] = BLACK
    return patch


def _solid_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(2, min(3, max_sw))
    return np.full((sh, sw), BLACK, dtype=int)


def _c_right_patch(sh: int, max_sw: int) -> np.ndarray:
    sw = random.randint(3, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    patch[0, :] = BLACK
    patch[-1, :] = BLACK
    patch[:, 0] = BLACK
    return patch


def _build_patch(kind: str, sh: int, max_sw: int) -> np.ndarray:
    if kind == "closed":
        return _closed_emitter_patch(sh, max_sw)
    if kind == "c_left":
        return _c_left_emitter_patch(sh, max_sw)
    if kind == "l_top":
        return _l_emitter_patch(sh, max_sw, top_bar=True)
    if kind == "l_bottom":
        return _l_emitter_patch(sh, max_sw, top_bar=False)
    if kind == "c_right":
        assert max_sw >= 3
        return _c_right_patch(sh, max_sw)
    if kind == "solid":
        return _solid_patch(sh, max_sw)
    raise ValueError(f"unknown sprite kind: {kind}")


def _choose_sprite_kinds(flags: list[bool], max_sw: int) -> list[str]:
    emitter_ids = [i for i, flag in enumerate(flags) if flag]
    open_anchor = random.choice(emitter_ids)
    closed_anchor = random.choice([i for i in emitter_ids if i != open_anchor])
    kinds = []
    for i, flag in enumerate(flags):
        if flag:
            if i == closed_anchor:
                kinds.append("closed")
            elif i == open_anchor or random.random() < OPEN_EMITTER_PROB:
                kinds.append(random.choice(["c_left", "l_top", "l_bottom"]))
            else:
                kinds.append("closed")
        else:
            distractors = ["solid"]
            if max_sw >= 3:
                distractors.append("c_right")
            kinds.append(random.choice(distractors))
    return kinds


def _render_output(h: int, w: int, sep: int, bands, kinds: list[str]):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), kind in zip(bands, kinds):
        patch = _build_patch(kind, sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0))

    trigger_rows = _trigger_rows_from_left(out[:, :sep], sep, BG, BLACK)
    assert trigger_rows, "generator must create at least one emitting recess"
    _paint_cross_rays(out, sep, trigger_rows, BG, RED)
    return out, sprites, trigger_rows


def _variant_baselines(h: int, w: int, sep: int, sprites):
    base = np.full((h, w), BG, dtype=int)
    base[:, sep] = SEP_COLOR
    for patch, r0, c0 in sprites:
        paste(base, patch, r0, c0, transparent=BG)

    d2_out = base.copy()
    d2_rows = _closed_trigger_rows_from_left(base[:, :sep], sep, BG)
    _paint_cross_rays(d2_out, sep, d2_rows, BG, RED)

    d5_out = base.copy()
    d5_rows = _trigger_rows_from_left(base[:, :sep], sep, BG, BLACK)
    _paint_row_rays(d5_out, sep, d5_rows, BG, RED)
    return d2_out, d5_out


def _unslide_sprites(h: int, w: int, sep: int, sprites) -> np.ndarray | None:
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
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    max_sw = min(SPRITE_W_MAX, sep - 1)
    if max_sw < 3:
        return None
    kinds = _choose_sprite_kinds(flags, max_sw)

    out, sprites, _ = _render_output(h, w, sep, bands, kinds)
    d2_out, d5_out = _variant_baselines(h, w, sep, sprites)
    if np.array_equal(out, d2_out) or np.array_equal(out, d5_out):
        return None

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
    raise RuntimeError("failed to generate a valid 6e453dd6 combo input")
