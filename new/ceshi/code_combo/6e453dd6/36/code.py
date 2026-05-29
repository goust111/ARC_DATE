# Combo variant of 6e453dd6 (index 36)
# combo_spec: D2/3+D28/2
# Name: 射线转弯+穿越分隔线

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    - Original: slide each left black sprite until it is flush with the grey separator; a hole that
#      touches sep-2 emits a straight horizontal ray only on the right half.
#    - Fused: the same slide-and-detect stage stays intact, but each triggering hole now produces a
#      bent polyline. The hole's vertical size controls the stem length, and the hole's vertical
#      position inside its owner sprite controls whether the stem goes upward or downward.
#    - Example: a top-side 2-cell-tall hole at sep-2 yields a red stem at sep-2 that rises 2 cells
#      from the hole's top edge, then a full-width horizontal red bar on the turn row. A bottom-side
#      3-cell-tall hole analogously drops 3 cells from the hole's bottom edge and then turns.
# b) Complexity calibration:
#    - The rule uses only two visible control axes: hole height in {2,3} and hole side {upper,lower}.
#      This is richer than the source task but still learnable from a few examples.
#    - The horizontal leg always spans the whole row, so the "cross the separator" effect is obvious
#      and does not rely on subtle separator-local coloring.
# c) Hidden assumptions that changed:
#    - The original solver assumed rays only touch the right pane and preserve the separator. That no
#      longer holds; rays may overwrite separator cells and may also appear on the left pane.
#    - The original solver used every row inside the hole as an emitter row. The fused rule instead
#      treats the hole as one emitter with a single turn row derived from hole geometry.
#    - Direction is no longer implicit from "touches sep-2"; it depends on the hole's location inside
#      the enclosing sprite bbox, so the generator must build clearly upper or lower holes.
# d) Placement legality:
#    - Emitter bands reserve extra blank rows on the shooting side so the turn row lands in empty
#      space rather than inside another sprite. That keeps horizontal bars non-overlapping with black
#      bodies except for the intended separator crossing.
# e) Visual observability:
#    - Upper holes always bend upward and lower holes always bend downward. Height-2 and height-3
#      holes visibly produce different stem lengths. Both cues are directly inspectable in the pixels.
# f) Visual cleanliness:
#    - Sprites remain in disjoint row bands. Turn rows are forced into blank gaps, so the full-width
#      red bars read as deliberate fold lines instead of noisy overwrite artifacts.
# g) Non-degeneracy target:
#    - Generated samples must have red on the left of the separator and on the separator row itself,
#      otherwise they collapse back toward the original right-only straight-ray task.


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

HOLE_HEIGHTS = (2, 3)
EXTRA_EMITTERS_PROB = 0.3
FLUSH_PROB = 0.3
MAX_RETRIES = 400


def _components4(mask):
    return components_mask(mask, connectivity=4)


def _bbox(points):
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


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


def _hole_specs(left: np.ndarray, bg: int) -> list[dict]:
    h, sep = left.shape
    specs = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        rows = sorted({r for r, _ in pts})
        cols = sorted({c for _, c in pts})
        specs.append(
            {
                "pts": pts,
                "bbox": _bbox(pts),
                "rows": rows,
                "cols": cols,
            }
        )
    return specs


def _owner_component(hole_bbox: tuple[int, int, int, int], comps: list[dict]) -> dict:
    hr0, hc0, hr1, hc1 = hole_bbox
    matches = []
    for rec in comps:
        r0, c0, r1, c1 = rec["bbox"]
        if r0 < hr0 and hr1 < r1 and c0 < hc0 and hc1 < c1:
            matches.append(rec)
    assert len(matches) == 1, f"expected one owner for hole {hole_bbox}, got {len(matches)}"
    return matches[0]


def _ray_spec(hole: dict, owner: dict) -> dict:
    hr0, _hc0, hr1, _hc1 = hole["bbox"]
    r0, _c0, r1, _c1 = owner["bbox"]
    hole_h = hr1 - hr0 + 1
    hole_center = (hr0 + hr1) / 2.0
    comp_center = (r0 + r1) / 2.0
    direction = -1 if hole_center < comp_center else 1
    if direction < 0:
        start_row = hr0
        turn_row = hr0 - hole_h
    else:
        start_row = hr1
        turn_row = hr1 + hole_h
    return {
        "direction": direction,
        "hole_h": hole_h,
        "start_row": start_row,
        "turn_row": turn_row,
    }


def _paint_bent_ray(out: np.ndarray, trigger_col: int, ray: dict) -> None:
    lo = min(ray["start_row"], ray["turn_row"])
    hi = max(ray["start_row"], ray["turn_row"])
    out[lo : hi + 1, trigger_col] = RED
    out[ray["turn_row"], :] = RED


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide the left black sprites into the separator, then convert separator-facing holes into bent red rays.

    Input structure:
    - A single full-height grey separator column splits the grid into a wider left pane and a narrow right pane.
    - Several black sprites sit in the left pane on a uniform background, and some sprites contain enclosed background holes near their right wall.

    Rule:
    - Translate each black connected component rightward until its right edge becomes flush with the separator.
    - Find enclosed background holes whose rightmost cells touch the inner trigger column just left of the separator.
    - For each such hole, use its height as the vertical travel distance; if the hole is in the upper half of its sprite, the ray rises first, otherwise it drops first. At the turn row, paint a horizontal red bar across the entire grid, crossing the separator and extending into the left pane.

    Key insight:
    - The hole does not emit one row-per-row anymore. Instead, each qualifying hole defines one folded path: upper holes bend upward, lower holes bend downward, and taller holes create longer stems before the full-width horizontal sweep.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    out = g.copy()
    left = out[:, :sep].copy()
    comps = _components4(left == BLACK)

    left[left == BLACK] = bg
    target_right = sep - 1
    for comp in comps:
        max_c = max(c for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = c + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[r, nc] = BLACK

    out[:, :sep] = left

    comp_records = [{"pts": comp, "bbox": _bbox(comp)} for comp in _components4(left == BLACK)]
    trigger_col = sep - 2
    for hole in _hole_specs(left, bg):
        if trigger_col not in hole["cols"]:
            continue
        owner = _owner_component(hole["bbox"], comp_records)
        ray = _ray_spec(hole, owner)
        assert 0 <= ray["turn_row"] < h, f"turn row left grid: {ray}"
        _paint_bent_ray(out, trigger_col, ray)

    return out


def _scatter(total, slots):
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n):
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters, n_total):
    return n_emitters * 5 + (n_total - n_emitters) * 2 + (n_total + 1) * MIN_BAND_GAP


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_emitter_flags(h):
    valid_counts = [
        n
        for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
        if _min_rows_needed(_min_emitters_for(n), n) <= h
    ]
    n = random.choice(valid_counts)
    min_em = _min_emitters_for(n)
    max_em = n
    if max_em > min_em and random.random() < EXTRA_EMITTERS_PROB:
        n_emitters = random.randint(min_em + 1, max_em)
    else:
        n_emitters = min_em
    flags = [True] * n_emitters + [False] * (n - n_emitters)
    random.shuffle(flags)
    return flags


def _branch_cycle(values, count):
    seq = []
    while len(seq) < count:
        chunk = list(values)
        random.shuffle(chunk)
        seq.extend(chunk)
    return seq[:count]


def _emitter_branches(n_emitters):
    if n_emitters <= 0:
        return []
    if n_emitters == 1:
        return [{"hole_h": random.choice(HOLE_HEIGHTS), "direction": random.choice(["up", "down"])}]

    hole_heights = _branch_cycle(HOLE_HEIGHTS, n_emitters)
    directions = _branch_cycle(("up", "down"), n_emitters)
    branches = [
        {"hole_h": hole_h, "direction": direction}
        for hole_h, direction in zip(hole_heights, directions)
    ]
    random.shuffle(branches)
    return branches


def _emitter_spec(max_sw, branch):
    hole_h = branch["hole_h"]
    direction = branch["direction"]
    hole_w = 2 if max_sw >= 4 and random.random() < 0.45 else 1
    min_sw = max(4, hole_w + 2)
    sw = random.randint(min_sw, max_sw)
    min_sh = hole_h + 3
    sh = random.randint(min_sh, SPRITE_H_MAX)
    return {
        "is_emitter": True,
        "sh": sh,
        "sw": sw,
        "hole_h": hole_h,
        "hole_w": hole_w,
        "direction": direction,
    }


def _distractor_spec(max_sw):
    kind_choices = ["l_shape", "solid"]
    if max_sw >= 3:
        kind_choices.extend(["c_left", "c_right"])
    kind = random.choice(kind_choices)
    min_sw = 2 if kind in {"solid", "l_shape"} else 3
    sw = random.randint(min_sw, max_sw)
    sh = random.randint(2, min(5, SPRITE_H_MAX))
    return {
        "is_emitter": False,
        "sh": sh,
        "sw": sw,
        "kind": kind,
    }


def _spec_gap_needs(spec):
    above = MIN_BAND_GAP
    below = MIN_BAND_GAP
    if spec["is_emitter"]:
        extra = spec["hole_h"] - 1
        if spec["direction"] == "up":
            above = max(above, extra)
        else:
            below = max(below, extra)
    return above, below


def _make_specs(flags, sep):
    max_sw = min(SPRITE_W_MAX, sep - 1)
    if max_sw < 4:
        return None
    branches = _emitter_branches(sum(flags))
    specs = []
    for is_emitter in flags:
        if is_emitter:
            spec = _emitter_spec(max_sw, branches.pop())
        else:
            spec = _distractor_spec(max_sw)
        specs.append(spec)
    return specs


def _sprite_bands(h, specs):
    n = len(specs)
    needs = [_spec_gap_needs(spec) for spec in specs]
    min_gaps = [0] * (n + 1)
    min_gaps[0] = needs[0][0]
    for i in range(1, n):
        min_gaps[i] = max(needs[i - 1][1], needs[i][0], MIN_BAND_GAP)
    min_gaps[n] = needs[-1][1]

    total = sum(spec["sh"] for spec in specs) + sum(min_gaps)
    if total > h:
        return None

    gaps = [a + b for a, b in zip(min_gaps, _scatter(h - total, n + 1))]
    bands = []
    row = gaps[0]
    for i, spec in enumerate(specs):
        bands.append((row, spec["sh"]))
        row += spec["sh"] + gaps[i + 1]
    return bands


def _emitter_patch(spec):
    sh, sw = spec["sh"], spec["sw"]
    hole_h, hole_w = spec["hole_h"], spec["hole_w"]
    patch = np.full((sh, sw), BLACK, dtype=int)
    hole_c0 = sw - hole_w - 1
    hole_c1 = sw - 2
    if spec["direction"] == "up":
        hole_r0 = 1
        hole_r1 = hole_h
    else:
        hole_r0 = sh - hole_h - 1
        hole_r1 = sh - 2
    patch[hole_r0 : hole_r1 + 1, hole_c0 : hole_c1 + 1] = BG
    return patch, {"hole_r0": hole_r0, "hole_r1": hole_r1, "hole_c0": hole_c0, "hole_c1": hole_c1}


def _distractor_patch(spec):
    sh, sw, kind = spec["sh"], spec["sw"], spec["kind"]
    if kind == "solid":
        return np.full((sh, sw), BLACK, dtype=int)

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


def _render_output(h, w, sep, bands, specs):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR
    sprites = []
    turn_rows = []

    for (r0, _sh), spec in zip(bands, specs):
        if spec["is_emitter"]:
            patch, hole = _emitter_patch(spec)
        else:
            patch = _distractor_patch(spec)
            hole = None
        c0 = sep - patch.shape[1]
        paste(out, patch, r0, c0, transparent=BG)

        sprite = {"patch": patch, "r0": r0, "c0": c0, "spec": spec}
        if spec["is_emitter"]:
            hole_r0 = r0 + hole["hole_r0"]
            hole_r1 = r0 + hole["hole_r1"]
            if spec["direction"] == "up":
                start_row = hole_r0
                turn_row = hole_r0 - spec["hole_h"]
            else:
                start_row = hole_r1
                turn_row = hole_r1 + spec["hole_h"]
            if not (0 <= turn_row < h):
                return None, None
            sprite["turn_row"] = turn_row
            sprite["start_row"] = start_row
            sprite["trigger_col"] = sep - 2
            turn_rows.append(turn_row)
            lo = min(start_row, turn_row)
            hi = max(start_row, turn_row)
            out[lo : hi + 1, sep - 2] = RED
            out[turn_row, :] = RED
        sprites.append(sprite)

    if len(turn_rows) != len(set(turn_rows)):
        return None, None
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(len(sprites)) if random.random() < FLUSH_PROB else -1
    for i, sprite in enumerate(sprites):
        patch, r0, c0 = sprite["patch"], sprite["r0"], sprite["c0"]
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _variant_signature(inp, out, sep):
    left_red = int(np.sum(out[:, :sep] == RED))
    sep_red = int(np.sum(out[:, sep] == RED))
    right_red = int(np.sum(out[:, sep + 1 :] == RED))
    return {"left_red": left_red, "sep_red": sep_red, "right_red": right_red}


def _gen_attempt():
    h, w, sep = _choose_grid_dims()
    flags = _choose_emitter_flags(h)
    specs = _make_specs(flags, sep)
    if specs is None:
        return None
    bands = _sprite_bands(h, specs)
    if bands is None:
        return None

    out, sprites = _render_output(h, w, sep, bands, specs)
    if out is None:
        return None
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    sig = _variant_signature(inp, out, sep)
    if sig["left_red"] == 0 or sig["sep_red"] == 0 or sig["right_red"] == 0:
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
    raise RuntimeError("failed to generate a valid fused 6e453dd6 input")
