# Combo variant of 6e453dd6 (index 16)
# combo_spec: D3/1+D17/0+D19/2
# Name: 多色精灵+洞内颜色编码射线

import random

import numpy as np

from arc_utils import components_mask, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior and trace:
#    - Base task: monochrome black shells slide right until flush with the grey separator;
#      a pure-background trigger hole then emits a fixed red ray.
#    - Combo task:
#      1. Each sprite shell has its own visible shell color.
#      2. Trigger cavities are mixed-color holes: they still contain background, but also a
#         marker color inside the hole.
#      3. After sliding, the cavity touching `sep-2` emits a ray whose color equals the
#         cavity marker, not the shell.
#    - Example trace:
#      a blue shell containing an orange cavity marker slides flush to the separator.
#      The enclosing hole reaches the trigger column, so the output paints orange on the
#      corresponding right-side rows, not blue. A green shell with a red marker emits red.
#      Relative to the original task, both the moving object identity and the ray color source
#      have changed.
# b) Complexity calibration:
#    - Shell colors come from a small fixed shell palette, marker/ray colors from a second
#      small palette. The palettes are disjoint, so "shell color" and "ray color" are visibly
#      different roles instead of noisy arbitrary recolorings.
#    - Every generated sample includes at least two emitters and uses both marker/ray colors,
#      so the cavity-color rule is visible within a single example.
#    - More than two emitters naturally create repeated ray colors across different shell colors,
#      which helps show that ray color does not come from shell identity.
# c) Implicit-assumption audit:
#    - Original `left == black` sprite detection fails because shells are now multi-colored
#      across sprites.
#    - Original pixelwise shifting fails because cavity markers must move with the shell even
#      though they are not shell-colored pixels.
#    - Original `left == bg` hole detection fails because trigger cavities are mixed
#      background+marker regions.
#    - Original fixed `red` painting fails because the ray color is now per-cavity.
#    - Fix:
#      detect shell components from the shell-color mask, shift each shell's full bounding-box
#      patch, then detect enclosed non-shell regions and read the unique marker color from each
#      triggering cavity.
# d) Placement legality:
#    - Sprites stay in disjoint row bands with at least one background row between them.
#    - Shell colors are unique within each sample, so shell components never merge by color.
#    - Markers are placed only inside enclosed trigger cavities, so border-background inference
#      and cavity detection remain valid.
# e) Visual observability:
#    - A human can compare input and output and see three distinct roles:
#      shell color, cavity marker color, and right-side ray color.
#    - Because shell and ray palettes are disjoint, samples directly show that matching the shell
#      color would be wrong.
# f) Visual cleanliness:
#    - Rays occupy empty right-side rows only, and row bands keep different sprites from sharing
#      rows, so outputs remain easy to parse.
#    - Colored markers appear only inside holes, which localizes the new legend cue.
# g) Change inferability:
#    - The changed color rule is structural and local: "read the color written inside the trigger
#      hole". No hidden ordering, numeric color ids, or arbitrary global mapping is used.
# New solver assumptions checked against the generator:
#    - Shell pixels use colors from SHELL_COLORS; marker/ray pixels use colors from MARKER_COLORS.
#    - Every shell is a single monochrome 4-connected component whose bbox contains all of its
#      internal marker pixels.
#    - Every triggering cavity contains exactly one marker color and at least one remaining
#      background cell, so the cavity is visibly mixed rather than solid-filled.
#    - Triggering rows from different sprites do not overlap because sprites occupy disjoint
#      vertical bands.


# ── Colors ──
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
SEP_COLOR = 5
BG = 6
ORANGE = 7
TEAL = 8
MAROON = 9

SHELL_COLORS = (BLUE, GREEN, YELLOW, TEAL, MAROON)
MARKER_COLORS = (RED, ORANGE)

# ── Grid dimensions ──
H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

# ── Sprite constraints ──
N_SPRITES_MIN, N_SPRITES_MAX = 2, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6
MIN_BAND_GAP = 1

# ── Probabilities ──
SPLIT_PROB = 0.4
EXTRA_EMITTERS_PROB = 0.25
FLUSH_PROB = 0.35

# ── Retry budget ──
MAX_RETRIES = 300


def _components4(mask: np.ndarray) -> list[np.ndarray]:
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


def _bbox_from_component(comp: np.ndarray) -> tuple[int, int, int, int]:
    rows = comp[:, 0]
    cols = comp[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _shell_mask(left: np.ndarray) -> np.ndarray:
    return np.isin(left, SHELL_COLORS)


def _component_color(left: np.ndarray, comp: np.ndarray) -> int:
    colors = {int(left[r, c]) for r, c in comp}
    assert len(colors) == 1, f"shell component must be monochrome, got {sorted(colors)}"
    return next(iter(colors))


def _slide_left_sprites(left_src: np.ndarray, bg: int) -> np.ndarray:
    left_dst = np.full_like(left_src, bg)
    target_right = left_src.shape[1] - 1

    comps = _components4(_shell_mask(left_src))
    assert comps, "expected at least one shell component"

    for comp in comps:
        _component_color(left_src, comp)
        r0, c0, r1, c1 = _bbox_from_component(comp)
        delta = target_right - c1
        assert delta >= 0, "left-side shells must stay left of the separator"
        patch = left_src[r0 : r1 + 1, c0 : c1 + 1].copy()
        paste(left_dst, patch, r0, c0 + delta)
    return left_dst


def _enclosed_cavities(left: np.ndarray) -> list[np.ndarray]:
    h, w = left.shape
    cavities = []
    for pts in _components4(~_shell_mask(left)):
        if any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in pts):
            continue
        cavities.append(pts)
    return cavities


def _cavity_ray_color(left: np.ndarray, cavity: np.ndarray, bg: int) -> int:
    colors = sorted({int(left[r, c]) for r, c in cavity if int(left[r, c]) != bg})
    assert len(colors) == 1, f"trigger cavity must contain exactly one marker color, got {colors}"
    color = colors[0]
    assert color in MARKER_COLORS, f"unexpected marker color {color}"
    return color


def _trigger_rows_to_colors(left: np.ndarray, bg: int) -> dict[int, int]:
    trigger_col = left.shape[1] - 2
    row_to_color: dict[int, int] = {}
    for cavity in _enclosed_cavities(left):
        if not any(c == trigger_col for _, c in cavity):
            continue
        ray_color = _cavity_ray_color(left, cavity, bg)
        for r in sorted({int(r) for r, _ in cavity}):
            prev = row_to_color.get(r)
            assert prev is None or prev == ray_color, f"conflicting ray colors on row {r}: {prev} vs {ray_color}"
            row_to_color[r] = ray_color
    return row_to_color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-colored shell sprites to the separator, then fire cavity-coded rays.

    Input structure:
    - A single grey (5) separator column splits the grid into a left sprite pane and an empty right pane.
    - Each left-side sprite has a monochrome shell color chosen independently from the other sprites.
    - Some enclosed holes inside those shells also contain a second, special marker color in addition to the background.

    Rule:
    - Move each sprite rightward until the shell's right edge is flush with the separator, carrying any hole markers along with it.
    - After sliding, find enclosed non-shell cavities; only cavities that touch the column just left of the separator emit.
    - For every row covered by an emitting cavity, paint the whole right side with that cavity's marker color.

    Key insight:
    - Shell color identifies which pixels belong to the same moving sprite, but ray color is not taken from that shell.
    - The decisive cue is the colored marker inside the trigger hole, so the solver must preserve the whole sprite patch and then read the cavity interior after the slide.
    """
    g = np.asarray(grid, dtype=int)

    sep = _find_separator_col(g, sep_color=SEP_COLOR)
    bg = _background_color(g, sep_col=sep, sep_color=SEP_COLOR)

    left_dst = _slide_left_sprites(g[:, :sep], bg)

    out = g.copy()
    out[:, :sep] = left_dst

    for r, ray_color in _trigger_rows_to_colors(left_dst, bg).items():
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = ray_color
    return out


def _scatter(total: int, slots: int) -> list[int]:
    vals = [0] * slots
    for _ in range(total):
        vals[random.randrange(slots)] += 1
    return vals


def _min_emitters_for(n: int) -> int:
    return (3 * n + 4) // 5


def _min_rows_needed(n_emitters: int, n_total: int) -> int:
    return n_emitters * 3 + (n_total - n_emitters) * 2 + (n_total - 1) + 2


def _sprite_bands(h: int, roles: list[int | None]) -> list[tuple[int, int]] | None:
    n = len(roles)
    min_heights = [3 if role is not None else 2 for role in roles]
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


def _build_split_emitter(
    patch: np.ndarray,
    sh: int,
    sw: int,
    left_wall: int,
    right_wall: int,
    mini_w: int,
    shell_color: int,
) -> None:
    place_top = random.random() < 0.5
    mini_h = random.randint(3, min(4, sh - 2))
    main_h = sh - mini_h + 1
    if place_top:
        patch[0, :mini_w] = shell_color
        for r in range(mini_h):
            patch[r, 0] = shell_color
            patch[r, mini_w - 1] = shell_color
        patch[mini_h - 1, :] = shell_color
        patch[-1, :] = shell_color
        for r in range(mini_h, sh):
            patch[r, left_wall] = shell_color
            patch[r, right_wall] = shell_color
    else:
        patch[0, :] = shell_color
        for r in range(main_h - 1):
            patch[r, left_wall] = shell_color
            patch[r, right_wall] = shell_color
        patch[main_h - 1, :] = shell_color
        patch[-1, :mini_w] = shell_color
        for r in range(main_h, sh):
            patch[r, 0] = shell_color
            patch[r, mini_w - 1] = shell_color


def _mark_trigger_cavity(patch: np.ndarray, ray_color: int) -> bool:
    trigger_col = patch.shape[1] - 2
    cavities = [pts for pts in _enclosed_cavities(patch) if any(c == trigger_col for _, c in pts)]
    if len(cavities) != 1:
        return False

    cavity = sorted((int(r), int(c)) for r, c in cavities[0])
    if len(cavity) < 2:
        return False

    keep_bg = cavity[0]
    for r, c in cavity:
        patch[r, c] = ray_color
    patch[keep_bg] = BG
    return True


def _emitter_patch(sh: int, max_sw: int, shell_color: int, ray_color: int) -> np.ndarray:
    for _ in range(60):
        hole_width_choices = [1, 2] if max_sw >= 4 else [1]
        if sh == 3 and 2 in hole_width_choices:
            hole_width_choices = [2]

        hole_w = random.choice(hole_width_choices)
        sw = random.randint(hole_w + 2, max_sw)
        patch = np.full((sh, sw), BG, dtype=int)

        right_wall = sw - 1
        left_wall = sw - hole_w - 2

        right_shift = random.randint(1, max(1, sw - 3))
        mini_w = sw - right_shift
        can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
        if can_split and random.random() < SPLIT_PROB:
            _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w, shell_color)
        else:
            patch[0, :] = shell_color
            patch[-1, :] = shell_color
            patch[1:-1, left_wall] = shell_color
            patch[1:-1, right_wall] = shell_color

        if _mark_trigger_cavity(patch, ray_color):
            return patch
    raise RuntimeError("failed to build an emitter patch with a marked trigger cavity")


def _distractor_patch(sh: int, max_sw: int, shell_color: int) -> np.ndarray:
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), shell_color, dtype=int)

    min_sw = 3 if kind.startswith("c_") or sh == 2 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    if kind == "c_left":
        patch[0, :] = shell_color
        patch[-1, :] = shell_color
        patch[:, -1] = shell_color
    elif kind == "c_right":
        patch[0, :] = shell_color
        patch[-1, :] = shell_color
        patch[:, 0] = shell_color
    else:
        patch[:, -1] = shell_color
        if random.random() < 0.5:
            patch[0, :] = shell_color
        else:
            patch[-1, :] = shell_color
    return patch


def _choose_grid_dims() -> tuple[int, int, int]:
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_sprite_roles(h: int) -> list[int | None]:
    valid_counts = [
        n
        for n in range(N_SPRITES_MIN, N_SPRITES_MAX + 1)
        if _min_rows_needed(_min_emitters_for(n), n) <= h and n <= len(SHELL_COLORS)
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

    assert n_emitters >= 2, "combo variant needs both marker colors visible"
    emitter_colors = [MARKER_COLORS[i % len(MARKER_COLORS)] for i in range(n_emitters)]
    random.shuffle(emitter_colors)

    roles = emitter_colors + [None] * (n - n_emitters)
    random.shuffle(roles)
    return roles


def _choose_shell_colors(n: int) -> list[int]:
    return random.sample(list(SHELL_COLORS), n)


def _render_output(
    h: int,
    w: int,
    sep: int,
    bands: list[tuple[int, int]],
    roles: list[int | None],
    shell_colors: list[int],
) -> tuple[np.ndarray, list[tuple[np.ndarray, int, int]]]:
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    for (r0, sh), ray_color, shell_color in zip(bands, roles, shell_colors):
        if ray_color is None:
            patch = _distractor_patch(sh, max_sw, shell_color)
        else:
            patch = _emitter_patch(sh, max_sw, shell_color, ray_color)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)
        sprites.append((patch, r0, c0))

    for r, ray_color in _trigger_rows_to_colors(out[:, :sep], BG).items():
        assert np.all(out[r, sep + 1 :] == BG), f"right half of row {r} must be empty before ray emission"
        out[r, sep + 1 :] = ray_color
    return out, sprites


def _unslide_sprites(h: int, w: int, sep: int, sprites: list[tuple[np.ndarray, int, int]]) -> np.ndarray | None:
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
    roles = _choose_sprite_roles(h)
    bands = _sprite_bands(h, roles)
    if bands is None:
        return None

    shell_colors = _choose_shell_colors(len(roles))
    out, sprites = _render_output(h, w, sep, bands, roles, shell_colors)
    inp = _unslide_sprites(h, w, sep, sprites)
    if inp is None:
        return None

    right = out[:, sep + 1 :]
    if not (np.any(right == MARKER_COLORS[0]) and np.any(right == MARKER_COLORS[1])):
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
