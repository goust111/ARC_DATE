# Combo variant of 6e453dd6 (index 1)
# combo_spec: D2/0+D7/0
# Name: 对角射线遇障碍停止

import random
from collections import Counter

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and step trace
# - The left pane keeps the seed task's first two stages: slide every black 4-connected sprite
#   right until it is flush with the grey separator, then detect enclosed background holes.
# - The trigger geometry also stays the same: a hole becomes an emitter only through the rows
#   whose hole pixels lie at column sep-2 after the slide.
# - The fused change happens entirely on the right pane. Each trigger row now launches two
#   45-degree branches instead of one horizontal stripe:
#     mouth row r -> up-right branch starts at (r-1, sep+1)
#                    down-right branch starts at (r+1, sep+1)
# - Those branches no longer run all the way to the border. The right pane already contains
#   colored blocker pixels, and each branch stops immediately before the first blocker on its
#   own diagonal path.
# - Example with sep=9 and a mouth row r=7:
#     seed task output on the right: row 7 becomes red across cols 10..W-1
#     D2-only output: (6,10),(5,11),... and (8,10),(9,11),...
#     fused output: if blockers sit at (4,12) and (10,12), then the red cells are
#       up-right  -> (6,10),(5,11)
#       down-right-> (8,10),(9,11)
#     and the blockers themselves stay colored. The learner must combine left-side mouth
#     detection with right-side diagonal collision stopping.
#
# b) Complexity calibration
# - The right pane is widened to 5..7 columns so diagonal motion and stopping distance are both
#   visually legible; the seed width 2..4 is too cramped for this fusion.
# - Top and bottom margins are raised to 2 background rows. That keeps emitter mouths away from
#   the border, so both diagonal branches usually have at least one visible red pixel before the
#   blocker instead of degenerating into zero- or one-cell stubs.
# - Every emitted branch gets its own interior blocker on the branch path. This guarantees the
#   D7 effect is active on every branch, not just occasionally.
#
# c) Implicit-assumption audit of the original main()
# - Separator detection and background inference still work because the separator is unchanged
#   and blockers are forbidden from the outer border, so border-majority background remains valid.
# - Sliding black components still works because the generator preserves disjoint row bands, so
#   rightward motion cannot create left-pane overlaps.
# - Hole detection via "bg component not touching the left-pane boundary" still works because
#   enclosure semantics are unchanged.
# - The original row-wise fill assumption `out[r, sep+1:] == bg` is invalid twice over now:
#   diagonal rays can overlap other red rays, and pre-existing colored blockers occupy the right
#   pane. The combo solver must therefore read blockers from the original input grid while writing
#   red into a separate output grid.
# - The old "one trigger row -> one independent right-side row fill" assumption also fails.
#   After D2, emission is path-based, so stopping must be checked cell by cell along each branch.
#
# d) Placement legality
# - Sprites are still generated output-first in the slid state, then unslid leftward per sprite.
# - Blockers are placed only on interior right-pane cells and never on the separator, outer
#   border, or left pane.
# - Each branch's blocker is chosen from cells that belong to that branch alone, so one branch's
#   target cannot accidentally truncate another branch.
# - Decoy blockers, when present, are placed outside all emitted branch paths.
#
# e) Visual observability
# - The left half still teaches "slide then find enclosed mouths"; the right half now teaches
#   two independent cues at once: diagonal direction and collision-based stopping.
# - Because blockers remain visible in both input and output while red grows toward them only in
#   the output, humans can infer that colored pixels are obstacles rather than emitted content.
# - Samples are rejected unless every emitted branch paints at least one red cell before stopping,
#   and unless the overall right-side red footprint is large enough to make the diagonal pattern
#   obvious across a few examples.
#
# f) Visual cleanliness
# - All emitted energy stays red and all blockers use a small palette disjoint from black/red/grey/bg.
# - Overlapping red branches are acceptable because overlap does not change color; blockers are kept
#   off shared branch cells, so the stop points remain readable.
# - Decoy blockers are sparse and never sit on active ray paths, avoiding clutter.
#
# g) Change inferability
# - Blocker colors carry no individual meaning: any non-background colored pixel on the right
#   blocks identically. This avoids hidden color-to-behavior mappings.
# - Repetition exists along both dimensions of change: multiple examples can show several mouths,
#   both slopes, and several different stopping distances using the same blocker palette.
#
# New solver assumptions introduced by the combo
# - The input right pane contains only background plus isolated colored blockers; the generator
#   never places red there.
# - Every emitted branch has an interior blocker somewhere on its diagonal path, and the cells
#   before that blocker on the same path are background in the input.
# - Blockers that matter to active branches are unique to one branch, so the first non-background
#   cell encountered by the solver is the intended stopping point.


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


def _branch_path(shape: tuple[int, int], start_r: int, start_c: int, dr: int) -> list[tuple[int, int]]:
    h, w = shape
    rr = int(start_r)
    cc = int(start_c)
    cells = []
    while 0 <= rr < h and 0 <= cc < w:
        cells.append((rr, cc))
        rr += int(dr)
        cc += 1
    return cells


def _paint_branch_until_blocker(
    out: np.ndarray,
    blockers: np.ndarray,
    start_r: int,
    start_c: int,
    dr: int,
    bg: int,
    red: int,
) -> tuple[int, bool]:
    """Paint one 45-degree branch until the first blocker cell on that path."""
    h, w = out.shape
    rr = int(start_r)
    cc = int(start_c)
    painted = 0
    hit_blocker = False
    while 0 <= rr < h and 0 <= cc < w:
        if int(blockers[rr, cc]) != int(bg):
            hit_blocker = True
            break
        assert out[rr, cc] in (bg, red), f"unexpected right-pane color {out[rr, cc]} at {(rr, cc)}"
        out[rr, cc] = red
        painted += 1
        rr += int(dr)
        cc += 1
    return painted, hit_blocker


def _emit_diagonal_rays(
    out: np.ndarray,
    blockers: np.ndarray,
    mouth_rows: list[int],
    sep: int,
    bg: int,
    red: int,
) -> list[tuple[int, bool]]:
    stats = []
    for r in mouth_rows:
        stats.append(_paint_branch_until_blocker(out, blockers, r - 1, sep + 1, -1, bg, red))
        stats.append(_paint_branch_until_blocker(out, blockers, r + 1, sep + 1, +1, bg, red))
    return stats


def _find_holes(left: np.ndarray, h: int, sep: int, bg: int) -> list[list[tuple[int, int]]]:
    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)
    return holes


def _mouth_rows_from_holes(holes: list[list[tuple[int, int]]], trigger_col: int) -> list[int]:
    mouth_rows = []
    for pts in holes:
        rows = sorted({int(r) for r, c in pts if int(c) == int(trigger_col)})
        if rows:
            mouth_rows.extend(rows)
    assert len(mouth_rows) == len(set(mouth_rows)), f"trigger rows must be unique, got {mouth_rows}"
    return mouth_rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide black sprites against the grey divider, then send red diagonal rays that stop at blockers.

    Input structure:
    - One full-height grey (5) separator column splits the board into a left sprite pane and a right blocker pane.
    - Black shapes live only on the left, while the right pane contains background plus isolated colored obstacle pixels.

    Rule:
    - Shift every black 4-connected component right until its right edge is flush with the separator.
    - In that slid arrangement, locate enclosed background holes; rows whose hole pixels reach column sep-2 are emitter mouths.
    - Each mouth row fires two red 45-degree branches into the right pane, one up-right and one down-right, and each branch stops immediately before the first non-background pixel on its diagonal path.

    Key insight: The left half still determines which mouths activate, but the right half now controls where each diagonal branch ends because the colored pixels act as collision blockers.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5
    black = 0
    red = 2

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()

    # Step 1: move black components on the left so that their right edge touches the separator.
    left = out[:, :sep].copy()
    left_mask = left == black
    comps = _components4(left_mask)

    # Clear old black pixels (leave other colors intact).
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

    # Step 2: find enclosed background regions (holes) in the slid left pane.
    holes = _find_holes(left, h, sep, bg)

    # Step 3: hole pixels at sep-2 define mouth rows. Each mouth fires two diagonal branches.
    mouth_rows = _mouth_rows_from_holes(holes, trigger_col=sep - 2)
    _emit_diagonal_rays(out, g, mouth_rows, sep, bg, red)

    return out


# --------------- generator ---------------

# ── Colors ──
BG = 6
BLACK = 0
RED = 2
SEP_COLOR = 5
TARGET_COLORS = [1, 3, 4, 7, 8, 9]

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
TOP_BOTTOM_MARGIN = 2

# ── Probabilities ──
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
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
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2 * TOP_BOTTOM_MARGIN


def _sprite_bands(h, is_emitter):
    """Partition grid rows into disjoint vertical bands for each sprite."""
    n = len(is_emitter)
    min_heights = [3 if e else 2 for e in is_emitter]
    base_gaps = [TOP_BOTTOM_MARGIN] + [MIN_BAND_GAP] * (n - 1) + [TOP_BOTTOM_MARGIN]
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


def _emitter_patch(sh, max_sw):
    """Build an emitter sprite patch: a closed shell with an interior hole at the right edge."""
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w)
    else:
        patch[0, :] = BLACK
        patch[-1, :] = BLACK
        patch[1:-1, left_wall] = BLACK
        patch[1:-1, right_wall] = BLACK
    return patch


def _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w):
    """Fill *patch* in-place with a split emitter (mini section + main section)."""
    place_top = random.random() < 0.5
    mini_h = random.randint(3, min(4, sh - 2))
    main_h = sh - mini_h + 1
    if place_top:
        patch[0, :mini_w] = BLACK
        for r in range(mini_h):
            patch[r, 0] = BLACK
            patch[r, mini_w - 1] = BLACK
        patch[mini_h - 1, :] = BLACK
        patch[-1, :] = BLACK
        for r in range(mini_h, sh):
            patch[r, left_wall] = BLACK
            patch[r, right_wall] = BLACK
    else:
        patch[0, :] = BLACK
        for r in range(main_h - 1):
            patch[r, left_wall] = BLACK
            patch[r, right_wall] = BLACK
        patch[main_h - 1, :] = BLACK
        patch[-1, :mini_w] = BLACK
        for r in range(main_h, sh):
            patch[r, 0] = BLACK
            patch[r, mini_w - 1] = BLACK


def _distractor_patch(sh, max_sw):
    """Build a non-emitter distractor sprite."""
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
    return h, w, sep


def _choose_emitter_flags(h):
    valid_counts = [
        n for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
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


def _emitter_mouth_rows(patch: np.ndarray) -> list[int]:
    sh, sw = patch.shape
    if sw < 2:
        return []
    return [r for r in range(sh) if patch[r, sw - 1] == BLACK and patch[r, sw - 2] == BG]


def _branch_specs_for_rows(mouth_rows: list[int], sep: int, shape: tuple[int, int]) -> list[dict]:
    specs = []
    for r in mouth_rows:
        for dr, start_r in [(-1, r - 1), (+1, r + 1)]:
            specs.append(
                {
                    "mouth_row": int(r),
                    "dr": int(dr),
                    "start_r": int(start_r),
                    "start_c": int(sep + 1),
                    "path": _branch_path(shape, start_r, sep + 1, dr),
                }
            )
    return specs


def _choose_branch_target_cells(branch_specs: list[dict], h: int, w: int) -> bool:
    counts = Counter(cell for spec in branch_specs for cell in spec["path"])
    for spec in branch_specs:
        candidates = []
        for idx, cell in enumerate(spec["path"]):
            r, c = cell
            if idx < 1:
                continue
            if not (0 < r < h - 1 and c < w - 1):
                continue
            if counts[cell] != 1:
                continue
            candidates.append((idx, cell))
        if not candidates:
            return False
        near = candidates[: min(4, len(candidates))]
        paint_len, target_cell = random.choice(near if random.random() < 0.8 else candidates)
        spec["target"] = target_cell
        spec["paint_len"] = int(paint_len)
    return True


def _stamp_branch_targets(out: np.ndarray, branch_specs: list[dict]) -> None:
    for spec in branch_specs:
        r, c = spec["target"]
        assert out[r, c] == BG, f"target cell {(r, c)} must be empty before stamping"
        out[r, c] = random.choice(TARGET_COLORS)


def _stamp_decoy_targets(out: np.ndarray, sep: int, branch_specs: list[dict]) -> int:
    h, w = out.shape
    forbidden = {cell for spec in branch_specs for cell in spec["path"]}
    candidates = [
        (r, c)
        for r in range(1, h - 1)
        for c in range(sep + 1, w - 1)
        if out[r, c] == BG and (r, c) not in forbidden
    ]
    if not candidates:
        return 0
    random.shuffle(candidates)
    n_decoys = random.randint(0, min(2, len(candidates), max(1, len(branch_specs) // 3)))
    for r, c in candidates[:n_decoys]:
        out[r, c] = random.choice(TARGET_COLORS)
    return n_decoys


def _emit_branch_specs(out: np.ndarray, blockers: np.ndarray, branch_specs: list[dict], bg: int, red: int) -> dict:
    lengths = []
    blocked = 0
    for spec in branch_specs:
        painted, hit = _paint_branch_until_blocker(
            out, blockers, spec["start_r"], spec["start_c"], spec["dr"], bg, red
        )
        lengths.append(int(painted))
        blocked += int(hit)
        assert hit, "every generated branch must hit an intended blocker"
        assert painted == spec["paint_len"], f"branch painted {painted}, expected {spec['paint_len']}"
    return {
        "branch_lengths": lengths,
        "blocked_branches": int(blocked),
        "visible_branches": sum(1 for x in lengths if x >= 1),
        "long_branches": sum(1 for x in lengths if x >= 2),
        "max_branch": max(lengths, default=0),
    }


def _render_output(h, w, sep, bands, flags):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    mouth_rows = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), is_emitter in zip(bands, flags):
        patch = _emitter_patch(sh, max_sw) if is_emitter else _distractor_patch(sh, max_sw)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        if is_emitter:
            mouth_rows.extend(r0 + lr for lr in _emitter_mouth_rows(patch))
        sprites.append((patch, r0, c0))

    mouth_rows = sorted(mouth_rows)
    assert mouth_rows, "at least one emitter row is required"
    assert len(mouth_rows) == len(set(mouth_rows)), "sprite bands should make trigger rows unique"

    branch_specs = _branch_specs_for_rows(mouth_rows, sep, out.shape)
    if not _choose_branch_target_cells(branch_specs, h, w):
        return None
    _stamp_branch_targets(out, branch_specs)
    n_decoys = _stamp_decoy_targets(out, sep, branch_specs)

    blocker_template = out.copy()
    emit_stats = _emit_branch_specs(out, blocker_template, branch_specs, BG, RED)
    emit_stats.update(
        {
            "mouth_rows": len(mouth_rows),
            "branch_count": len(branch_specs),
            "red_cells": int(np.sum(out[:, sep + 1 :] == RED)),
            "decoy_targets": int(n_decoys),
        }
    )
    return out, sprites, emit_stats


def _unslide_sprites(h, w, sep, sprites, right_template):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR
    right = right_template[:, sep + 1 :].copy()
    right[right == RED] = BG
    inp[:, sep + 1 :] = right

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
        and stats["branch_count"] >= 2
        and stats["blocked_branches"] == stats["branch_count"]
        and stats["visible_branches"] == stats["branch_count"]
        and stats["long_branches"] >= 2
        and stats["max_branch"] >= 2
        and stats["red_cells"] >= right_w + 3
    )


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    bands = _sprite_bands(h, flags)
    if bands is None:
        return None

    rendered = _render_output(h, w, sep, bands, flags)
    if rendered is None:
        return None
    out, sprites, stats = rendered
    if not _visible_enough(stats, w, sep):
        return None

    inp = _unslide_sprites(h, w, sep, sprites, out)
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
