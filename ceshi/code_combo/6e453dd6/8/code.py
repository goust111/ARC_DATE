# Combo variant of 6e453dd6 (index 8)
# combo_spec: D2/0+D3/0
# Name: 对角线周期虚线射线

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and step trace
# - The left pane keeps the seed task's object logic: black components slide right until their
#   right edge reaches the grey separator, then enclosed background holes are recomputed in the
#   slid state.
# - The trigger condition stays geometric rather than color-based: a hole emits only if some
#   hole cell lies at column sep-2 after the slide, i.e. the hole touches the inner face of the
#   flush right wall.
# - The fused change happens at emission time and combines D2 with D3 in one trajectory rule:
#   each trigger row launches two 45-degree branches, but only every other visited cell is
#   colored red. The skipped cells remain background, so the output shows diagonal dashed rays
#   instead of either solid horizontal bars (seed) or solid diagonal fans (pure D2).
# - Small trace: suppose sep=9 and a slid emitter exposes one mouth row r=10. The seed task
#   would paint row 10 solid red at cols 10..end. Pure D2 would paint solid diagonals
#   (9,10),(8,11),(7,12),... and (11,10),(12,11),(13,12),.... The fused rule paints
#   (9,10), skip (8,11), paint (7,12), ... and symmetrically downward, giving a red/bg/red/bg
#   alternation along each diagonal path.
#
# b) Complexity calibration
# - The periodic trajectory is fixed to one simple cycle, [red, bg], rather than introducing a
#   hole-width-dependent family. That keeps the rule learnable while still making the D3 change
#   visually obvious.
# - The right pane is widened to 5..7 columns. With the seed width 2..4, a dashed diagonal
#   often collapses to one or two red cells, which is too weak to distinguish from incidental
#   noise.
# - Emitters are generated with a single trigger row per hole. This avoids a common failure mode
#   where nearby branches refill one another's skipped cells and visually degenerate back toward
#   solid diagonals.
#
# c) Implicit-assumption audit of the original main()
# - Separator detection and border-based background inference remain valid because the separator
#   is still a unique full-height grey column and the border is still dominated by background.
# - Component sliding remains valid because sprites still live in disjoint row bands, so sliding
#   them flush cannot merge components.
# - Hole detection as "background component not touching the left-pane boundary" still matches
#   the new task; the enclosure semantics do not change.
# - The original row-wise assertion that each trigger row's entire right half is background no
#   longer works. Diagonal branches can revisit cells from different emitters, so the solver now
#   checks only that the input right pane starts as pure background and then allows later writes
#   onto already-red cells.
# - The original emission loop depended only on trigger rows. The fused rule additionally needs
#   path stepping with parity, so emission must be traced cell-by-cell.
#
# d) Placement legality
# - Generation stays output-first: build the slid left pane and emitted right pane, then unslide
#   each sprite independently to obtain the input.
# - Row bands remain disjoint and at least one background row apart, so unsliding cannot create
#   overlaps.
# - All diagonal tracing is clipped by grid bounds; no ray can cross the separator or write
#   outside the grid.
#
# e) Visual observability
# - The solver-visible change is strong in two ways at once: the rays are diagonal rather than
#   horizontal, and they contain periodic gaps rather than solid fills.
# - The generator rejects samples where dashed and solid diagonal emission would look identical or
#   nearly identical. This explicitly prevents D3 from collapsing away.
#
# f) Visual cleanliness
# - Only one output color is used on the rays, so the right pane stays legible: grey divider,
#   magenta gaps, and red energy.
# - Avoiding multi-row emitter mouths keeps the right pane from becoming a dense red thicket.
#
# g) Change inferability
# - No hidden numeric ordering is used. The only cue for the periodic rule is spatial: along each
#   diagonal path, every second location is left blank. That is directly visible in 2-4 examples.
#
# h) New solver assumptions and generator guarantees
# - The solver assumes the input right pane is entirely background before emission; the generator
#   enforces that.
# - The solver permits branch overlap only between background and red; the generator never places
#   non-background colors on the right side.
# - The solver does not require one-row holes in principle, but the generator prefers them to keep
#   the dashed pattern observable.


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


def _trace_diagonal_branch(shape: tuple[int, int], start_r: int, start_c: int, dr: int) -> list[tuple[int, int]]:
    h, w = shape
    rr = int(start_r)
    cc = int(start_c)
    cells = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += int(dr)
        cc += 1
    return cells


def _paint_branch(out: np.ndarray, cells: list[tuple[int, int]], bg: int, red: int, dashed: bool) -> tuple[int, int]:
    painted = 0
    for step, (r, c) in enumerate(cells):
        assert out[r, c] in (bg, red), f"unexpected right-pane color {out[r, c]} at {(r, c)}"
        if (not dashed) or (step % 2 == 0):
            out[r, c] = red
            painted += 1
    return len(cells), painted


def _emit_diagonal_fan(
    out: np.ndarray,
    mouth_rows: list[int],
    sep: int,
    bg: int,
    red: int,
    dashed: bool,
) -> dict:
    stats = {
        "branch_lengths": [],
        "painted_per_branch": [],
    }
    for r in mouth_rows:
        for start_r, dr in ((r - 1, -1), (r + 1, 1)):
            cells = _trace_diagonal_branch(out.shape, start_r, sep + 1, dr)
            length, painted = _paint_branch(out, cells, bg=bg, red=red, dashed=dashed)
            stats["branch_lengths"].append(int(length))
            stats["painted_per_branch"].append(int(painted))
    return stats


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites against the divider, then emit dashed diagonal rays from enclosed mouths.

    Input structure:
    - A single full-height grey (5) separator column splits the board into a left sprite pane and a right empty pane.
    - Black connected components appear only on the left over one background color; the right side starts as pure background.

    Rule:
    - Shift every black component rightward until its right edge is flush with the separator.
    - In the shifted layout, find enclosed background holes and keep only those with hole pixels at column sep-2.
    - For each emitting mouth row, launch one up-right and one down-right 45-degree ray into the right pane, but color only alternating cells along each branch to make a red dashed diagonal.

    Key insight: The slide still determines which holes are active nozzles, but the signal is now encoded jointly by direction and rhythm: diagonal fan paths plus a red/background alternation along each path.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    assert np.all(out[:, sep + 1 :] == bg), "input right pane must start as background only"

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
    mouth_rows = []
    for pts in holes:
        rows = sorted({int(r) for r, c in pts if int(c) == trigger_col})
        if rows:
            mouth_rows.extend(rows)
    assert len(mouth_rows) == len(set(mouth_rows)), f"trigger rows must be unique, got {mouth_rows}"

    _emit_diagonal_fan(out, mouth_rows, sep=sep, bg=bg, red=red, dashed=True)
    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 11, 19
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 5, 7

# ── Sprite constraints ──
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# ── Probabilities ──
EXTRA_EMITTERS_PROB = 0.2
FLUSH_PROB = 0.35

# ── Retry budget ──
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


def _sprite_bands(h, is_emitter):
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
        i = random.choice(choices)
        heights[i] += 1

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


def _choose_emitter_flags(h):
    valid_counts = [n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
                    if _min_rows_needed(_min_emitters_for(n), n) <= h]
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


def _emitter_patch(sh: int, max_sw: int) -> tuple[np.ndarray, list[int]]:
    """Single-row-hole emitter: easy to detect and keeps dashed rays visually sparse."""
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BLACK, dtype=int)
    hole_row = random.randint(1, sh - 2)
    hole_start = sw - hole_w - 1
    patch[hole_row, hole_start:sw - 1] = BG
    return patch, [hole_row]


def _distractor_patch(sh: int, max_sw: int) -> np.ndarray:
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


def _render_output(h, w, sep, bands, flags):
    base = np.full((h, w), BG, dtype=int)
    base[:, sep] = SEP_COLOR

    sprites = []
    global_mouth_rows = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        if is_emitter:
            patch, mouth_rows = _emitter_patch(sh, max_sw)
        else:
            patch = _distractor_patch(sh, max_sw)
            mouth_rows = []
        sw = patch.shape[1]
        c0 = sep - sw
        paste(base, patch, r0, c0, transparent=BG)
        global_mouth_rows.extend([r0 + lr for lr in mouth_rows])
        sprites.append((patch, r0, c0))

    dashed = base.copy()
    dashed_stats = _emit_diagonal_fan(dashed, global_mouth_rows, sep=sep, bg=BG, red=RED, dashed=True)

    solid = base.copy()
    _emit_diagonal_fan(solid, global_mouth_rows, sep=sep, bg=BG, red=RED, dashed=False)

    right_slice = (slice(None), slice(sep + 1, None))
    diff_cells = int(np.sum(dashed[right_slice] != solid[right_slice]))
    red_cells = int(np.sum(dashed[right_slice] == RED))
    long_branches = sum(1 for x in dashed_stats["branch_lengths"] if x >= 4)
    two_sided_mouths = 0
    for i in range(0, len(dashed_stats["branch_lengths"]), 2):
        pair = dashed_stats["branch_lengths"][i:i + 2]
        if len(pair) == 2 and min(pair) >= 3:
            two_sided_mouths += 1
    stats = {
        "mouth_rows": len(global_mouth_rows),
        "branch_lengths": list(dashed_stats["branch_lengths"]),
        "red_cells": red_cells,
        "diff_vs_solid": diff_cells,
        "long_branches": int(long_branches),
        "two_sided_mouths": int(two_sided_mouths),
    }
    return dashed, sprites, stats


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


def _visible_enough(stats, w, sep):
    right_w = w - sep - 1
    return (
        stats["mouth_rows"] >= 1
        and stats["diff_vs_solid"] >= max(3, right_w // 2)
        and stats["red_cells"] >= 4
        and stats["long_branches"] >= 2
        and stats["two_sided_mouths"] >= 1
    )


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    out, sprites, stats = _render_output(h, w, sep, bands, flags)
    if not _visible_enough(stats, w, sep):
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
