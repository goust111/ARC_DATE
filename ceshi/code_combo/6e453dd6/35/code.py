# Combo variant of 6e453dd6 (index 35)
# combo_spec: D17/0+D27/2
# Name: 多色精灵+间距编码分组
#
# Design analysis
# - a) Concrete behavior:
#   - Base task: each left-side sprite slides right until flush with the grey
#     separator; enclosed holes that reach the inner face of that separator emit
#     horizontal rays on the right.
#   - Fusion change:
#     1. Every sprite is monochrome, but different sprites use different visible
#        colors.
#     2. Vertical spacing between consecutive sprites encodes grouping:
#        exactly 1 empty row means "same group", while 3+ empty rows means
#        "new group". Gap 2 is intentionally never generated.
#     3. A group's ray color is the color of its topmost sprite, not necessarily
#        the color of the sprite that actually emits.
#   - Small trace:
#     - Suppose three sprites from top to bottom are blue, red, yellow.
#     - If blue/red have gap 1, they form one group. If red/yellow have gap 4,
#       yellow starts a new group.
#     - After sliding, if the red sprite has a triggering hole, it emits a blue
#       ray because blue is the leader color of that gap-1 group; if yellow emits,
#       it emits yellow because it is its own group leader.
#     - Difference from D17-only: ray color no longer follows the emitting
#       sprite's own body color. Difference from the original task: both sprite
#       identity and inter-sprite spacing now matter.
# - b) Complexity calibration:
#   - Colors come from a small fixed palette and remain unique per sprite, so
#     human observers can track bodies unambiguously.
#   - Group encoding uses only two visually separable gap classes: 1 versus 3+.
#     This keeps the rule readable in 2-4 examples and avoids fragile counting.
#   - Every sample contains at least one multi-sprite group with a follower
#     emitter whose ray color differs from its own body color, so the fusion is
#     not degenerate back to the D17-only behavior.
# - c) Implicit-assumption audit:
#   - `left == black` is invalid once sprites are multi-colored; sprite detection
#     must use "non-background and non-separator".
#   - Clearing and restamping can no longer hard-code black; each component must
#     preserve its own body color through the slide.
#   - The old fixed red ray is invalid; ray color must be derived from the
#     emitting sprite's group leader.
#   - A new assumption appears: consecutive sprite bands must have gaps only in
#     the supported classes (1 or >=3). The generator guarantees that.
#   - Another new assumption is that each sprite color is unique, so the hole's
#     enclosing wall color identifies exactly one sprite; the generator also
#     guarantees that.
# - d) Placement legality:
#   - Sprites are still placed in disjoint row bands, so horizontal unsliding
#     cannot create overlaps.
#   - Same-group neighbors keep exactly one background row; different groups keep
#     at least three. This preserves both non-overlap and unambiguous grouping.
#   - The right half starts as pure background, so group-colored rays never clash
#     with pre-existing right-side structure.
# - e) Visual observability:
#   - The solver-visible cue is local and human-readable: "tight adjacency means
#     reuse the top sprite's color across the cluster; wide gap resets the color
#     source".
#   - Because at least two groups appear and each group has at least one emitter,
#     examples show both shared-color-within-group and color-reset-across-groups.
# - f) Visual cleanliness:
#   - Sprites remain in vertical bands and rays occupy only those rows, so output
#     rows do not overwrite each other.
#   - The palette excludes background and separator colors, keeping strong visual
#     contrast between group leaders, follower emitters, holes, separator, and
#     rays.
# - g) Inferability of the color change:
#   - The color mapping is structural rather than arbitrary: output color is
#     copied from the topmost sprite of the local gap-defined group.
#   - That topmost sprite is physically adjacent to the follower emitters via
#     exactly one blank row, so the support for the mapping is visible directly in
#     the grid without relying on hidden numeric color ids.
# - New solver assumptions checked against the generator:
#   - Each sprite is a single-color 4-connected component.
#   - Sprite row intervals are strictly ordered and disjoint.
#   - Holes are enclosed by exactly one sprite color.
#   - At least one emitter belongs to a multi-sprite group whose leader color is
#     different from the emitter's own color.

import random

import numpy as np

from arc_utils import components_mask, paste


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


def _sprite_mask(left: np.ndarray, bg: int, sep_color: int) -> np.ndarray:
    return (left != bg) & (left != sep_color)


def _component_color(left: np.ndarray, comp) -> int:
    colors = {int(left[r, c]) for r, c in comp}
    assert len(colors) == 1, f"sprite component must be monochrome, got {sorted(colors)}"
    return next(iter(colors))


def _component_infos(left: np.ndarray, bg: int, sep_color: int) -> list[dict]:
    sprite_mask = _sprite_mask(left, bg=bg, sep_color=sep_color)
    comps = _components4(sprite_mask)
    assert comps, "expected at least one sprite on the left side"

    infos = []
    seen_colors = set()
    for comp in comps:
        color = _component_color(left, comp)
        assert color not in seen_colors, f"sprite colors must be unique, got duplicate {color}"
        seen_colors.add(color)
        rows = [int(r) for r, _ in comp]
        infos.append(
            {
                "cells": comp,
                "color": color,
                "r0": min(rows),
                "r1": max(rows),
            }
        )
    infos.sort(key=lambda info: (info["r0"], info["r1"], info["color"]))

    for prev, curr in zip(infos, infos[1:]):
        assert prev["r1"] < curr["r0"], "sprite row spans must be disjoint"
    return infos


def _group_color_by_sprite_color(infos: list[dict]) -> dict[int, int]:
    mapping = {}
    leader_color = None
    prev = None
    for info in infos:
        if prev is None:
            leader_color = int(info["color"])
        else:
            gap = int(info["r0"]) - int(prev["r1"]) - 1
            assert gap == SAME_GROUP_GAP or gap >= DIFF_GROUP_GAP_MIN, (
                "consecutive sprite gaps must encode grouping with 1 or >=3 empty rows"
            )
            if gap >= DIFF_GROUP_GAP_MIN:
                leader_color = int(info["color"])
        mapping[int(info["color"])] = int(leader_color)
        prev = info
    return mapping


def _hole_wall_color(left: np.ndarray, hole, bg: int) -> int:
    h, w = left.shape
    neighbors = set()
    for r, c in hole:
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w:
                val = int(left[nr, nc])
                if val != bg:
                    neighbors.add(val)
    assert len(neighbors) == 1, f"triggering hole must belong to exactly one sprite color, got {sorted(neighbors)}"
    return next(iter(neighbors))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide multi-colored sprites rightward to a grey separator, then emit group-colored rays from enclosed holes.

    Input structure:
    - A full-height grey (5) separator splits the grid into a wide left pane and a narrow right pane.
    - The left pane contains several monochrome sprites with different body colors, arranged in vertically disjoint bands.
    - Consecutive bands with exactly one blank row belong to the same group; a gap of at least three blank rows starts a new group.

    Rule:
    - Move every left-side sprite horizontally until its right edge becomes flush with the separator, preserving that sprite's own body color.
    - After sliding, detect enclosed background holes that touch the column just left of the separator.
    - Each such hole emits across the right pane on all of its rows, but the ray color is not the emitter's own color unless it is the group leader: use the color of the topmost sprite in that gap-defined group.

    Key insight: The hole geometry still decides which rows fire, but vertical spacing decides how colors propagate: tight stacks share the leader's color, while a wide gap resets the color source for the next stack.
    """
    g = np.asarray(grid, dtype=int)
    h, _ = g.shape
    sep_color = 5

    sep = _find_separator_col(g, sep_color=sep_color)
    bg = _background_color(g, sep_col=sep, sep_color=sep_color)

    out = g.copy()
    left = out[:, :sep].copy()

    infos = _component_infos(left, bg=bg, sep_color=sep_color)
    group_color = _group_color_by_sprite_color(infos)

    sprite_mask = _sprite_mask(left, bg=bg, sep_color=sep_color)
    left[sprite_mask] = bg

    target_right = sep - 1
    for info in infos:
        comp = info["cells"]
        color = int(info["color"])
        max_c = max(int(c) for _, c in comp)
        delta = target_right - max_c
        assert delta >= 0, "left-side components must stay left of the separator"
        for r, c in comp:
            nc = int(c) + delta
            assert 0 <= nc < sep, f"shifted pixel {(r, c)} left the pane"
            left[int(r), nc] = color

    out[:, :sep] = left

    holes = []
    for pts in _components4(left == bg):
        if any(r == 0 or r == h - 1 or c == 0 or c == sep - 1 for r, c in pts):
            continue
        holes.append(pts)

    trigger_col = sep - 2
    row_to_color = {}
    for pts in holes:
        if not any(int(c) == trigger_col for _, c in pts):
            continue
        source_color = _hole_wall_color(left, pts, bg=bg)
        ray_color = int(group_color[source_color])
        for r in sorted({int(r) for r, _ in pts}):
            if r in row_to_color:
                assert row_to_color[r] == ray_color, f"conflicting ray colors for row {r}: {row_to_color[r]} vs {ray_color}"
            row_to_color[r] = ray_color

    for r, color in sorted(row_to_color.items()):
        assert np.all(out[r, sep + 1 :] == bg), f"right half of row {r} must be background before ray emission"
        out[r, sep + 1 :] = color

    return out


# --------------- generator ---------------

BG = 6
SEP_COLOR = 5
SPRITE_COLORS = [1, 2, 3, 4, 7, 8, 9]

H_MIN, H_MAX = 16, 25
W_MIN, W_MAX = 10, 16
SEP_MIN, SEP_MAX = 5, 11
RIGHT_W_MIN, RIGHT_W_MAX = 2, 4

N_SPRITES_MIN, N_SPRITES_MAX = 3, 5
SPRITE_H_MAX = 8
SPRITE_W_MAX = 6

SAME_GROUP_GAP = 1
DIFF_GROUP_GAP_MIN = 3

FOLLOWER_EMITTER_PROB = 0.75
LEADER_EMITTER_PROB = 0.55
SPLIT_PROB = 0.4
FLUSH_PROB = 0.35

MAX_RETRIES = 400


def _group_patterns(n: int) -> list[list[int]]:
    if n == 3:
        return [[2, 1], [1, 2]]
    if n == 4:
        return [[2, 2], [3, 1], [1, 3], [2, 1, 1], [1, 2, 1], [1, 1, 2]]
    if n == 5:
        return [[2, 3], [3, 2], [2, 1, 2], [1, 2, 2], [2, 2, 1], [3, 1, 1], [1, 3, 1], [1, 1, 3]]
    raise AssertionError(f"unsupported sprite count {n}")


def _choose_grid_dims():
    h = random.randint(H_MIN, H_MAX)
    right_w = random.randint(RIGHT_W_MIN, RIGHT_W_MAX)
    valid_seps = [s for s in range(SEP_MIN, SEP_MAX + 1) if W_MIN <= s + 1 + right_w <= W_MAX]
    sep = random.choice(valid_seps)
    w = sep + 1 + right_w
    return h, w, sep


def _choose_group_layout(h: int):
    for _ in range(100):
        n = random.randint(N_SPRITES_MIN, N_SPRITES_MAX)
        groups = random.choice(_group_patterns(n))
        flags = _choose_emitter_flags(groups)
        if _min_rows_needed(groups, flags) <= h:
            group_ids = _group_ids(groups)
            return groups, flags, group_ids
    return None


def _group_ids(groups: list[int]) -> list[int]:
    ids = []
    for gid, size in enumerate(groups):
        ids.extend([gid] * size)
    return ids


def _choose_emitter_flags(groups: list[int]):
    delegated_group = random.choice([i for i, size in enumerate(groups) if size >= 2])
    flags = []

    for gid, size in enumerate(groups):
        group_flags = [False] * size
        if size == 1:
            group_flags[0] = True
        elif gid == delegated_group:
            group_flags[0] = False
            group_flags[1] = True
            for i in range(2, size):
                group_flags[i] = random.random() < FOLLOWER_EMITTER_PROB
        else:
            group_flags[0] = random.random() < LEADER_EMITTER_PROB
            for i in range(1, size):
                group_flags[i] = random.random() < FOLLOWER_EMITTER_PROB
            if not any(group_flags):
                group_flags[random.randrange(size)] = True
        flags.extend(group_flags)

    assert any(flags), "expected at least one emitter"
    return flags


def _min_rows_needed(groups: list[int], flags: list[bool]) -> int:
    sprite_heights = [3 if flag else 2 for flag in flags]
    internal_gaps = sum(size - 1 for size in groups) * SAME_GROUP_GAP
    break_gaps = (len(groups) - 1) * DIFF_GROUP_GAP_MIN
    outer_gaps = 2
    return sum(sprite_heights) + internal_gaps + break_gaps + outer_gaps


def _sprite_bands(h: int, groups: list[int], flags: list[bool]):
    n = len(flags)
    heights = [3 if flag else 2 for flag in flags]
    group_ids = _group_ids(groups)

    gaps = []
    for i in range(n - 1):
        if group_ids[i] == group_ids[i + 1]:
            gaps.append(SAME_GROUP_GAP)
        else:
            gaps.append(DIFF_GROUP_GAP_MIN)

    top_gap = 1
    bottom_gap = 1
    min_total = top_gap + bottom_gap + sum(heights) + sum(gaps)
    if min_total > h:
        return None

    free = h - min_total
    while free > 0:
        choices = [("top", 0), ("bottom", 0)]
        for i, sh in enumerate(heights):
            if sh < SPRITE_H_MAX:
                choices.append(("height", i))
        for i, gap in enumerate(gaps):
            if gap >= DIFF_GROUP_GAP_MIN:
                choices.append(("break", i))
        kind, idx = random.choice(choices)
        if kind == "top":
            top_gap += 1
        elif kind == "bottom":
            bottom_gap += 1
        elif kind == "height":
            heights[idx] += 1
        else:
            gaps[idx] += 1
        free -= 1

    bands = []
    r = top_gap
    for i, sh in enumerate(heights):
        bands.append((r, sh))
        r += sh
        if i < n - 1:
            r += gaps[i]
    assert r + bottom_gap == h
    return bands


def _choose_sprite_colors(n: int) -> list[int]:
    return random.sample(SPRITE_COLORS, n)


def _emitter_patch(sh, max_sw, color):
    hole_w = random.choice([1, 2]) if max_sw >= 4 else 1
    sw = random.randint(hole_w + 2, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)

    right_wall = sw - 1
    left_wall = sw - hole_w - 2

    right_shift = random.randint(1, max(1, sw - 3))
    mini_w = sw - right_shift
    can_split = sh >= 5 and mini_w >= 3 and right_shift >= 1
    if can_split and random.random() < SPLIT_PROB:
        _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w, color)
    else:
        patch[0, :] = color
        patch[-1, :] = color
        patch[1:-1, left_wall] = color
        patch[1:-1, right_wall] = color
    return patch


def _build_split_emitter(patch, sh, sw, left_wall, right_wall, mini_w, color):
    place_top = random.random() < 0.5
    mini_h = random.randint(3, min(4, sh - 2))
    main_h = sh - mini_h + 1
    if place_top:
        patch[0, :mini_w] = color
        for r in range(mini_h):
            patch[r, 0] = color
            patch[r, mini_w - 1] = color
        patch[mini_h - 1, :] = color
        patch[-1, :] = color
        for r in range(mini_h, sh):
            patch[r, left_wall] = color
            patch[r, right_wall] = color
    else:
        patch[0, :] = color
        for r in range(main_h - 1):
            patch[r, left_wall] = color
            patch[r, right_wall] = color
        patch[main_h - 1, :] = color
        patch[-1, :mini_w] = color
        for r in range(main_h, sh):
            patch[r, 0] = color
            patch[r, mini_w - 1] = color


def _distractor_patch(sh, max_sw, color):
    choices = ["l_shape"]
    if sh >= 3 and max_sw >= 3:
        choices.extend(["c_left", "c_right"])
    if sh <= 3:
        choices.append("solid")
    kind = random.choice(choices)

    if kind == "solid":
        sw = random.randint(2, min(3, max_sw))
        return np.full((sh, sw), color, dtype=int)

    min_sw = 3 if kind.startswith("c_") or sh == 2 else 2
    sw = random.randint(min_sw, max_sw)
    patch = np.full((sh, sw), BG, dtype=int)
    if kind == "c_left":
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, -1] = color
    elif kind == "c_right":
        patch[0, :] = color
        patch[-1, :] = color
        patch[:, 0] = color
    else:
        patch[:, -1] = color
        if random.random() < 0.5:
            patch[0, :] = color
        else:
            patch[-1, :] = color
    return patch


def _trigger_rows_from_patch(patch: np.ndarray, color: int) -> list[int]:
    rows = []
    for r in range(patch.shape[0]):
        if patch[r, -1] == color and patch[r, -2] == BG:
            rows.append(r)
    return rows


def _render_output(h, w, sep, bands, flags, group_ids, colors):
    out = np.full((h, w), BG, dtype=int)
    out[:, sep] = SEP_COLOR

    sprites = []
    max_sw = min(SPRITE_W_MAX, sep - 1)
    leader_color_by_group = {}

    for idx, ((r0, sh), is_emitter, gid, color) in enumerate(zip(bands, flags, group_ids, colors)):
        if gid not in leader_color_by_group:
            leader_color_by_group[gid] = color

        patch = _emitter_patch(sh, max_sw, color) if is_emitter else _distractor_patch(sh, max_sw, color)
        sw = patch.shape[1]
        c0 = sep - sw
        paste(out, patch, r0, c0, transparent=BG)

        trigger_rows = _trigger_rows_from_patch(patch, color) if is_emitter else []
        group_color = leader_color_by_group[gid]
        for lr in trigger_rows:
            assert np.all(out[r0 + lr, sep + 1 :] == BG)
            out[r0 + lr, sep + 1 :] = group_color

        sprites.append(
            {
                "index": idx,
                "patch": patch,
                "r0": r0,
                "c0": c0,
                "color": color,
                "group_id": gid,
                "group_color": group_color,
                "is_emitter": bool(is_emitter),
                "trigger_rows": trigger_rows,
            }
        )
    return out, sprites


def _unslide_sprites(h, w, sep, sprites):
    n = len(sprites)
    inp = np.full((h, w), BG, dtype=int)
    inp[:, sep] = SEP_COLOR

    flush_i = random.randrange(n) if random.random() < FLUSH_PROB else -1
    for i, sprite in enumerate(sprites):
        patch = sprite["patch"]
        r0 = int(sprite["r0"])
        c0 = int(sprite["c0"])
        if c0 < 1:
            return None
        delta = 0 if i == flush_i else random.randint(1, c0)
        paste(inp, patch, r0, c0 - delta, transparent=BG)
    return inp


def _has_required_fusion_behavior(sprites) -> bool:
    return any(
        sprite["is_emitter"]
        and sprite["trigger_rows"]
        and sprite["color"] != sprite["group_color"]
        for sprite in sprites
    )


def _gen_attempt() -> np.ndarray | None:
    h, w, sep = _choose_grid_dims()
    layout = _choose_group_layout(h)
    if layout is None:
        return None
    groups, flags, group_ids = layout
    bands = _sprite_bands(h, groups, flags)
    if bands is None:
        return None

    colors = _choose_sprite_colors(len(flags))
    out, sprites = _render_output(h, w, sep, bands, flags, group_ids, colors)
    if not _has_required_fusion_behavior(sprites):
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
