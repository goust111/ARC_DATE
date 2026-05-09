# Combo variant of a251c730 (index 19)
# combo_spec: D3/3+D4/1
# Name: 方向标记+镜像翻转

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, paste, random_free_location

# Design analysis
# fused_variant:
#   concrete_behavior:
#     source_side:
#       - keep the original wallpaper + two bordered rectangles layout
#       - keep 1-2 sprite types in the source frame, each with one singleton anchor color
#     destination_side:
#       - replace singleton markers with 2-cell domino markers
#       - one cell is the anchor-colored marker pixel
#       - the adjacent pointer cell uses the destination border color
#       - unlike the pure D3 rotation variant, the pointer no longer chooses a quarter-turn
#       - instead, pointer axis chooses which mirror channel is even eligible:
#         * horizontal domino (left/right neighbor) -> inspect the left-half rule
#         * vertical domino (up/down neighbor) -> inspect the upper-half rule
#     fused_transform_rule:
#       - if the chosen channel is horizontal and the anchor lies in the left half of the destination interior,
#         apply a left-right mirror
#       - if the chosen channel is vertical and the anchor lies in the upper half of the destination interior,
#         apply an up-down mirror
#       - otherwise keep the source orientation
#       - after that optional mirror, align the transformed anchor back onto the anchor-colored marker cell
#   behavior_trace:
#     example:
#       source_sprite:
#         shape_rows: ["AB.", "BBB", ".BB"]
#         anchor_rc: (0,0)
#       original_task:
#         - a red singleton at (r,c) simply receives the raw sprite with A still at local (0,0)
#       fused_task_case_1:
#         - a red+border horizontal domino in the left half says "use the horizontal channel here"
#         - left-half test passes, so the sprite is mirrored with `np.fliplr`
#         - anchor moves from (0,0) to (0,2), and that new anchor is aligned to the red pixel
#       fused_task_case_2:
#         - a red+border vertical domino in the lower half says "use the vertical channel here"
#         - upper-half test fails, so no mirror happens
#         - the same sprite now stamps unchanged even though the marker is still a domino
#       difference:
#         - relative to the original task, direction markers are parsed as structured objects, not singleton cues
#         - relative to the single D3 variant, pointer direction no longer means rotation amount
#         - relative to the single D4 variant, destination position alone is insufficient; the pointer axis gates
#           which half-test matters
#   complexity_calibration:
#     - the visible rule space stays bounded: only two semantic marker states (horizontal-axis or vertical-axis)
#     - the position test is also binary, so the fused rule is still learnable from a few examples
#     - the generator always includes one horizontal-active marker and one vertical-active marker, so both channels
#       receive repeated support instead of appearing as isolated corner cases
#     - sprites are chosen from shapes asymmetric under both LR and UD reflection, preventing visual collapse
#   implicit_assumption_audit:
#     - original dst detection by singleton components breaks; the solver now expects every destination component
#       to be a 2-cell 4-connected marker
#     - iterating every non-background interior pixel would double-count a marker, so the solver must iterate
#       connected components instead
#     - original anchor coordinates are not stable after reflection, so transformed anchor coordinates must be
#       recomputed before paste
#     - D4's raw "position decides all flips" logic would allow both flips in the top-left quadrant; that clashes
#       with the fused brief where the pointer encodes the mirror axis. The resolution here is deliberate:
#       position only turns a chosen channel on or off, while the pointer selects the channel itself
#   placement_legality:
#     - generator remains output-first: place already-transformed sprites into the destination output canvas,
#       then back-project each stamp into one 2-cell input marker
#     - every transformed patch must fit fully inside the destination interior
#     - marker dominoes must stay 4-disconnected from one another, so a destination component is always exactly size 2
#     - pointer direction sign is redundant but systematic: it points inward along the chosen axis, keeping all
#       pointers in-bounds without introducing a hidden extra semantic degree of freedom
#   visual_observability:
#     - anchor color still visibly matches a source sprite type to a destination cue
#     - the extra border-colored neighbor visibly tells the viewer whether to think "horizontal mirror channel"
#       or "vertical mirror channel"
#     - because the actual flip depends on half-position, examples show that the same style of marker can sometimes
#       flip and sometimes not, but only along its own axis
#   visual_cleanliness:
#     - input destination remains sparse: only 2-3 domino cues on a plain background
#     - output sprites are non-overlapping, and marker cells are kept Manhattan distance > 1 apart
#     - pointer cells reuse the border color instead of adding another palette role
#   inference_guardrail:
#     - no numeric color ordering is used for any visible decision
#     - the only matching relation is anchor-color equality, and the only transform controls are marker axis plus
#       the marker's half-position inside the destination rectangle


AXIAL_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]


def _interior_bg(grid: np.ndarray, rect: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    return int(Counter(map(int, inner.flatten())).most_common(1)[0][0])


def _rect_components(grid: np.ndarray, rect: tuple[int, int, int, int]) -> tuple[int, list[dict]]:
    r0, c0, r1, c1 = map(int, rect)
    inner = grid[int(r0) + 1 : int(r1), int(c0) + 1 : int(c1)]
    ibg = int(Counter(map(int, inner.flatten())).most_common(1)[0][0])
    comps: list[dict] = []

    for coords in components_mask(inner != int(ibg), connectivity=4):
        if coords.size == 0:
            continue
        rr0, cc0, rr1, cc1 = bbox_coords(coords)
        mask = np.zeros((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), dtype=bool)
        mask[coords[:, 0] - int(rr0), coords[:, 1] - int(cc0)] = True
        patch = inner[int(rr0) : int(rr1) + 1, int(cc0) : int(cc1) + 1].copy()
        patch[~mask] = 0
        vals = [int(inner[int(r), int(c)]) for r, c in coords]
        comps.append(
            {
                "size": int(coords.shape[0]),
                "colors": dict(Counter(vals)),
                "bbox": (int(rr0), int(cc0), int(rr1), int(cc1)),
                "mask": mask,
                "patch": patch,
                "coords": coords.copy(),
            }
        )

    comps.sort(key=lambda d: -int(d["size"]))
    return int(ibg), comps


def _find_rectangles(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int], int, int]]:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    rects: list[tuple[int, tuple[int, int, int, int], int, int]] = []
    min_h = 6
    min_w = 6

    for r0 in range(int(H)):
        for c0 in range(int(W)):
            bc = int(g[int(r0), int(c0)])
            for r1 in range(int(r0 + min_h - 1), int(H)):
                for c1 in range(int(c0 + min_w - 1), int(W)):
                    if (
                        int(g[int(r0), int(c1)]) != int(bc)
                        or int(g[int(r1), int(c0)]) != int(bc)
                        or int(g[int(r1), int(c1)]) != int(bc)
                    ):
                        continue
                    if not bool((g[int(r0), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r1), int(c0) : int(c1) + 1] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c0)] == int(bc)).all()):
                        continue
                    if not bool((g[int(r0) : int(r1) + 1, int(c1)] == int(bc)).all()):
                        continue
                    ibg = _interior_bg(g, (int(r0), int(c0), int(r1), int(c1)))
                    if int(ibg) == int(bc):
                        continue
                    area = int((int(r1) - int(r0) + 1) * (int(c1) - int(c0) + 1))
                    rects.append((int(area), (int(r0), int(c0), int(r1), int(c1)), int(bc), int(ibg)))

    rects.sort(reverse=True, key=lambda x: int(x[0]))
    return rects[:24]


def _anchor_color_from_component(comp: dict) -> int | None:
    counts = {int(col): int(ct) for col, ct in comp["colors"].items()}
    if not counts:
        return None
    min_ct = min(counts.values())
    candidates = [int(col) for col, ct in counts.items() if int(ct) == int(min_ct)]
    if len(candidates) != 1:
        return None
    return int(candidates[0])


def _transform_template(
    patch: np.ndarray, mask: np.ndarray, anchor_rc: tuple[int, int], flip_lr: bool, flip_ud: bool
) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    out_patch = np.array(patch, dtype=int).copy()
    out_mask = np.array(mask, dtype=bool).copy()
    ar, ac = map(int, anchor_rc)
    h, w = map(int, out_patch.shape)
    if bool(flip_ud):
        out_patch = np.flipud(out_patch).copy()
        out_mask = np.flipud(out_mask).copy()
        ar = int(h - 1 - ar)
    if bool(flip_lr):
        out_patch = np.fliplr(out_patch).copy()
        out_mask = np.fliplr(out_mask).copy()
        ac = int(w - 1 - ac)
    return out_patch, out_mask, (int(ar), int(ac))


def _marker_axis(direction: tuple[int, int]) -> str | None:
    dr, dc = map(int, direction)
    if abs(int(dc)) == 1 and int(dr) == 0:
        return "h"
    if abs(int(dr)) == 1 and int(dc) == 0:
        return "v"
    return None


def _marker_flip_flags(
    rr: int, cc: int, inner_h: int, inner_w: int, direction: tuple[int, int], fused: bool
) -> tuple[bool, bool]:
    if not bool(fused):
        return False, False
    axis = _marker_axis(direction)
    if axis == "h":
        return bool(int(cc) < (int(inner_w) / 2.0)), False
    if axis == "v":
        return False, bool(int(rr) < (int(inner_h) / 2.0))
    return False, False


def _extract_templates(src_comps: list[dict]) -> dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]]:
    templates: dict[int, tuple[np.ndarray, np.ndarray, tuple[int, int]]] = {}
    for comp in src_comps:
        anchor_col = _anchor_color_from_component(comp)
        if anchor_col is None:
            continue
        patch = np.array(comp["patch"], dtype=int)
        mask = np.array(comp["mask"], dtype=bool)
        anchors = np.argwhere(mask & (patch == int(anchor_col)))
        if anchors.shape[0] != 1:
            continue
        ar, ac = map(int, anchors[0])
        if int(anchor_col) not in templates or int(mask.sum()) > int(templates[int(anchor_col)][1].sum()):
            templates[int(anchor_col)] = (patch.copy(), mask.copy(), (int(ar), int(ac)))
    return templates


def _solve(grid: np.ndarray, *, fused: bool = True) -> np.ndarray:
    g = np.array(grid, dtype=int)
    rects = _find_rectangles(g)
    assert rects, "must find candidate rectangles"

    src = None
    dst = None
    for _area, rect, bc, _ibg in rects:
        ibg2, comps = _rect_components(g, rect)
        sizes = [int(comp["size"]) for comp in comps]
        if not sizes:
            continue
        if max(sizes) > 2 and src is None:
            cand_templates = _extract_templates(comps)
            if cand_templates:
                src = (rect, bc, ibg2, comps, cand_templates)
        if min(sizes) == 2 and max(sizes) == 2 and dst is None:
            dst = (rect, bc, ibg2, comps)

    assert src is not None and dst is not None, "must find source sprites and 2-cell destination markers"

    (_sr, _sc, _sr1, _sc1), _src_bc, _src_bg, src_comps, templates = src
    (dr, dc, dr1, dc1), dst_bc, dst_bg, _dst_comps = dst

    assert templates, "must recover at least one source template"

    out_h = int(dr1 - dr + 1)
    out_w = int(dc1 - dc + 1)
    out = np.full((out_h, out_w), int(dst_bg), dtype=int)
    out[0, :] = int(dst_bc)
    out[-1, :] = int(dst_bc)
    out[:, 0] = int(dst_bc)
    out[:, -1] = int(dst_bc)

    inner = g[int(dr) + 1 : int(dr1), int(dc) + 1 : int(dc1)]
    inner_h, inner_w = map(int, inner.shape)
    for coords in components_mask(inner != int(dst_bg), connectivity=4):
        if int(coords.shape[0]) != 2:
            continue
        cells = [(int(r), int(c), int(inner[int(r), int(c)])) for r, c in coords]
        anchor_cells = [(r, c, col) for r, c, col in cells if int(col) in templates]
        if len(anchor_cells) != 1:
            continue
        rr, cc, anchor_col = anchor_cells[0]
        pr, pc, _ptr_col = next((r, c, col) for r, c, col in cells if (r, c) != (rr, cc))
        direction = (int(pr - rr), int(pc - cc))
        if tuple(direction) not in AXIAL_DIRS:
            continue
        patch, mask, anchor_rc = templates[int(anchor_col)]
        flip_lr, flip_ud = _marker_flip_flags(int(rr), int(cc), inner_h, inner_w, tuple(direction), fused)
        tpatch, tmask, (tar, tac) = _transform_template(patch, mask, anchor_rc, flip_lr, flip_ud)
        top = int(1 + int(rr) - int(tar))
        left = int(1 + int(cc) - int(tac))
        paste(out, tpatch, int(top), int(left), mask=tmask, overwrite=True, clip=True)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the destination frame by stamping source sprites onto directional mirror markers.

    Input structure:
    - Two bordered rectangles lie on a striped wallpaper: the source rectangle contains full sprites, while the
      destination rectangle contains only 2-cell markers made from an anchor-colored pixel plus one adjacent pointer pixel.
    - Each source sprite has one unique least-frequent anchor color, and the destination anchor pixel reuses that color.

    Rule:
    - Match each destination marker to the source sprite with the same anchor color.
    - Read the marker's axis from the pointer: a horizontal neighbor selects the horizontal mirror channel, while a
      vertical neighbor selects the vertical mirror channel.
    - Apply that mirror only when the anchor pixel sits in the corresponding activating half of the destination interior
      (left half for horizontal, upper half for vertical), then align the transformed anchor onto the anchor-colored marker.

    Key insight: Color still chooses which sprite to copy, but the adjacent pointer chooses which positional half-test can
    trigger a mirror, so the final orientation depends on both the marker's local shape and its global placement.
    """
    return _solve(grid, fused=True)


# --------------- generator (O2) ---------------

# === CFG: Canvas & layout ===
GRID_SIZE = 30
WALLPAPER_COLORS = (4, 5)
TOTAL_COLORS_TARGET = (7, 8, 9)
GAP_RANGE = (1, 4)

# === CFG: Rectangle size bounds ===
SRC_H_RANGE = (9, 19)
SRC_W_RANGE = (8, 18)
DST_H_RANGE = (9, 19)
DST_W_RANGE = (8, 20)
DST_W_SIDE_MAX = 20

# === CFG: Sprite & placement ===
SPRITE_TYPE_CHOICES = (1, 2)
# Keep every sample at 3 markers so the fused rule always shows both active channels
# plus one explicit same-task "no mirror" counterexample.
MARKER_COUNT_RANGE = (3, 3)
SOURCE_SPRITE_RANGE = (2, 4)
SOURCE_SPRITE_PADDING = 1

# === CFG: Generation retries ===
MAX_GEN_ATTEMPTS = 600

SPRITE_SHAPES = [
    (("AB.", "BBB"), (0, 0)),
    (("ABB", "BBB"), (0, 0)),
    (("AB.", "BBB", ".BB"), (0, 0)),
    (("ABB", "BBB", "..B"), (0, 0)),
    ((".AB", "BBB", "BB."), (0, 1)),
    (("AB.", "BBB", ".B."), (0, 0)),
]


def _render_sprite(shape_rows: tuple[str, ...], body: int, anchor: int, bg: int) -> np.ndarray:
    char_map = {"B": body, "A": anchor}
    rows = [[char_map.get(ch, bg) for ch in row] for row in shape_rows]
    return np.array(rows, dtype=int)


def _make_wallpaper(direction: str, palette: list[int]) -> np.ndarray:
    period = len(palette)
    stripe = np.array(palette * ((GRID_SIZE + period - 1) // period), dtype=int)[:GRID_SIZE]
    if direction == "vertical":
        return np.tile(stripe, (GRID_SIZE, 1))
    return np.tile(stripe.reshape(-1, 1), (1, GRID_SIZE))


def _build_frame(h: int, w: int, border: int, bg: int) -> np.ndarray:
    rect = np.full((h, w), bg, dtype=int)
    rect[0, :] = rect[-1, :] = border
    rect[:, 0] = rect[:, -1] = border
    return rect


def _choose_counts(n_total: int, n_types: int) -> list[int]:
    counts = [1] * n_types
    for _ in range(n_total - n_types):
        counts[random.randrange(n_types)] += 1
    return counts


def _choose_dimensions(layout: str) -> tuple[int, int, int, int]:
    gap = random.randint(*GAP_RANGE)
    if layout == "side_by_side":
        src_w = random.randint(*SRC_W_RANGE)
        dst_w_max = min(DST_W_SIDE_MAX, GRID_SIZE - gap - src_w)
        if dst_w_max < DST_W_RANGE[0]:
            raise ValueError("no side-by-side width fit")
        dst_w = random.randint(DST_W_RANGE[0], dst_w_max)
        src_h = random.randint(*SRC_H_RANGE)
        dst_h = random.randint(*DST_H_RANGE)
    else:
        src_h = random.randint(*SRC_H_RANGE)
        dst_h_max = min(DST_H_RANGE[1], GRID_SIZE - gap - src_h)
        if dst_h_max < DST_H_RANGE[0]:
            raise ValueError("no stacked height fit")
        dst_h = random.randint(DST_H_RANGE[0], dst_h_max)
        src_w = random.randint(*SRC_W_RANGE)
        dst_w = random.randint(*DST_W_RANGE)
    return src_h, src_w, dst_h, dst_w


def _place_rectangles(
    layout: str, src_h: int, src_w: int, dst_h: int, dst_w: int
) -> tuple[tuple[int, int], tuple[int, int]]:
    if layout == "side_by_side":
        gap = random.randint(1, GRID_SIZE - src_w - dst_w)
        slack = GRID_SIZE - src_w - dst_w - gap
        left_margin = random.randint(0, slack)
        src_r = random.randint(0, GRID_SIZE - src_h)
        dst_r = random.randint(0, GRID_SIZE - dst_h)
        if random.random() < 0.5:
            src_c, dst_c = left_margin, left_margin + src_w + gap
        else:
            dst_c, src_c = left_margin, left_margin + dst_w + gap
    else:
        gap = random.randint(1, GRID_SIZE - src_h - dst_h)
        slack = GRID_SIZE - src_h - dst_h - gap
        top_margin = random.randint(0, slack)
        src_c = random.randint(0, GRID_SIZE - src_w)
        dst_c = random.randint(0, GRID_SIZE - dst_w)
        if random.random() < 0.5:
            src_r, dst_r = top_margin, top_margin + src_h + gap
        else:
            dst_r, src_r = top_margin, top_margin + dst_h + gap
    return (src_r, src_c), (dst_r, dst_c)


def _assign_palette(n_types: int) -> tuple[list[int], int, int, list[int], list[int], int | None] | None:
    wallpaper_n = 5 if n_types == 1 else random.choice(list(WALLPAPER_COLORS))
    used_total_target = random.choice(list(TOTAL_COLORS_TARGET))
    extras_needed = used_total_target - (wallpaper_n + 1 + n_types)
    if extras_needed < 0 or extras_needed > 2:
        return None

    all_colors = list(range(10))
    wallpaper = random.sample(all_colors, wallpaper_n)
    remaining = [c for c in all_colors if c not in wallpaper]
    random.shuffle(remaining)

    src_border = remaining.pop()
    anchors = [remaining.pop() for _ in range(n_types)]

    dst_border_extra = extras_needed >= 2 or (extras_needed == 1 and random.random() < 0.5)
    body_extra = extras_needed >= 2 or (extras_needed >= 1 and not dst_border_extra)

    if dst_border_extra:
        if not remaining:
            return None
        dst_border = remaining.pop()
    else:
        border_choices = [c for c in wallpaper if c != src_border]
        if not border_choices:
            return None
        dst_border = random.choice(border_choices)

    body_pool = wallpaper[:]
    forced_body: int | None = None
    if body_extra:
        if not remaining:
            return None
        forced_body = remaining.pop()
        body_pool.append(forced_body)

    return wallpaper, src_border, dst_border, anchors, body_pool, forced_body


def _marker_plan(n_markers: int) -> list[tuple[str, bool]]:
    plan: list[tuple[str, bool]] = [("h", True), ("v", True)]
    if int(n_markers) >= 3:
        # The third marker always demonstrates the branch where the chosen axis
        # is present but the half-position gate does not trigger a mirror.
        plan.append(random.choice([("h", False), ("v", False)]))
    while len(plan) < int(n_markers):
        plan.append(random.choice([("h", False), ("v", False), ("h", True), ("v", True)]))
    random.shuffle(plan)
    return plan


def _state_matches(rr: int, cc: int, inner_h: int, inner_w: int, axis: str, active: bool) -> bool:
    if axis == "h":
        return (int(cc) < (int(inner_w) / 2.0)) == bool(active)
    return (int(rr) < (int(inner_h) / 2.0)) == bool(active)


def _pointer_delta(rr: int, cc: int, inner_h: int, inner_w: int, axis: str) -> tuple[int, int]:
    if axis == "h":
        return 0, 1 if int(cc) < (int(inner_w) / 2.0) else -1
    return 1 if int(rr) < (int(inner_h) / 2.0) else -1, 0


def _marker_cells_are_clear(
    occupied: list[tuple[int, int]], new_cells: list[tuple[int, int]], shape: tuple[int, int]
) -> bool:
    H, W = map(int, shape)
    if len(set(new_cells)) != len(new_cells):
        return False
    for r, c in new_cells:
        if not (0 <= int(r) < H and 0 <= int(c) < W):
            return False
        for pr, pc in occupied:
            if abs(int(r) - int(pr)) + abs(int(c) - int(pc)) <= 1:
                return False
    return True


def _place_markers(
    sprite_types: list[dict], marker_type_ids: list[int], dst_h: int, dst_w: int, dst_bg: int
) -> tuple[np.ndarray, list[tuple[int, int, int, str, bool, tuple[int, int]]]] | None:
    inner_h = int(dst_h - 2)
    inner_w = int(dst_w - 2)
    inner = np.full((inner_h, inner_w), dst_bg, dtype=int)
    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    occupied_marker_cells: list[tuple[int, int]] = []
    marker_points: list[tuple[int, int, int, str, bool, tuple[int, int]]] = []

    plans = _marker_plan(len(marker_type_ids))
    marker_plan = list(zip(marker_type_ids, plans))
    marker_plan.sort(
        key=lambda item: (
            0 if item[1][1] else 1,
            -int(sprite_types[int(item[0])]["dst_mask"].sum()),
        )
    )

    for type_id, (axis, active) in marker_plan:
        sprite = sprite_types[int(type_id)]
        flip_lr = axis == "h" and bool(active)
        flip_ud = axis == "v" and bool(active)
        patch, mask, (ar, ac) = _transform_template(
            sprite["dst_patch"], sprite["dst_mask"], sprite["anchor_rc"], flip_lr, flip_ud
        )
        ph, pw = map(int, patch.shape)

        candidates: list[tuple[int, int, int, int, tuple[int, int]]] = []
        for rr in range(inner_h):
            for cc in range(inner_w):
                if not _state_matches(int(rr), int(cc), inner_h, inner_w, axis, bool(active)):
                    continue
                dr, dc = _pointer_delta(int(rr), int(cc), inner_h, inner_w, axis)
                pointer = (int(rr + dr), int(cc + dc))
                if not _marker_cells_are_clear(occupied_marker_cells, [(int(rr), int(cc)), pointer], inner.shape):
                    continue
                top = int(rr - ar)
                left = int(cc - ac)
                if top < 0 or left < 0 or top + ph > inner_h or left + pw > inner_w:
                    continue
                if bool((occupied[top : top + ph, left : left + pw] & mask).any()):
                    continue
                candidates.append((int(rr), int(cc), int(top), int(left), (int(dr), int(dc))))

        if not candidates:
            return None

        rr, cc, top, left, direction = random.choice(candidates)
        paste(inner, patch, int(top), int(left), mask=mask, overwrite=True, clip=False)
        occupied[top : top + ph, left : left + pw] |= mask
        occupied_marker_cells.extend([(int(rr), int(cc)), (int(rr + direction[0]), int(cc + direction[1]))])
        marker_points.append((int(rr), int(cc), int(type_id), axis, bool(active), direction))

    return inner, marker_points


def _place_sources(
    sprite_types: list[dict], source_type_ids: list[int], src_h: int, src_w: int, src_bg: int
) -> np.ndarray | None:
    inner = np.full((src_h - 2, src_w - 2), src_bg, dtype=int)
    for type_id in source_type_ids:
        sprite = sprite_types[int(type_id)]
        patch = sprite["src_patch"]
        try:
            r0, c0 = random_free_location(
                inner,
                patch,
                transparent=src_bg,
                border=SOURCE_SPRITE_PADDING,
                padding=SOURCE_SPRITE_PADDING,
            )
        except ValueError:
            return None
        paste(inner, patch, int(r0), int(c0), transparent=src_bg)
    return inner


def _gen_attempt() -> np.ndarray | None:
    n_types = random.choice(list(SPRITE_TYPE_CHOICES))

    palette_result = _assign_palette(n_types)
    if palette_result is None:
        return None
    wallpaper, src_border, dst_border, anchors, body_pool, forced_body = palette_result

    src_bg = random.choice(wallpaper)
    dst_bg_choices = [c for c in wallpaper if c != dst_border]
    if not dst_bg_choices:
        return None
    dst_bg = random.choice(dst_bg_choices)

    sprite_specs = random.sample(SPRITE_SHAPES, n_types)
    bodies: list[int] = []
    for i in range(n_types):
        choices = [c for c in body_pool if c not in anchors and c not in {src_bg, dst_bg, dst_border}]
        if not choices:
            return None
        if i == 0 and forced_body is not None and forced_body in choices:
            bodies.append(forced_body)
        else:
            bodies.append(random.choice(choices))

    sprite_types = []
    for (shape_rows, anchor_rc), body, anchor in zip(sprite_specs, bodies, anchors):
        src_patch = _render_sprite(shape_rows, body, anchor, src_bg)
        dst_patch = _render_sprite(shape_rows, body, anchor, dst_bg)
        sprite_types.append(
            {
                "shape_rows": shape_rows,
                "anchor_rc": anchor_rc,
                "body": body,
                "anchor": anchor,
                "src_patch": src_patch,
                "dst_patch": dst_patch,
                "dst_mask": dst_patch != int(dst_bg),
            }
        )

    n_markers = random.randint(*MARKER_COUNT_RANGE)
    marker_type_ids = [i for i, ct in enumerate(_choose_counts(n_markers, n_types)) for _ in range(ct)]
    random.shuffle(marker_type_ids)

    n_source = random.randint(max(2, n_types), SOURCE_SPRITE_RANGE[1])
    source_type_ids = [i for i, ct in enumerate(_choose_counts(n_source, n_types)) for _ in range(ct)]
    random.shuffle(source_type_ids)

    layout = random.choice(["side_by_side", "stacked"])
    try:
        src_h, src_w, dst_h, dst_w = _choose_dimensions(layout)
    except ValueError:
        return None
    if src_h * src_w <= dst_h * dst_w:
        return None

    marker_result = _place_markers(sprite_types, marker_type_ids, dst_h, dst_w, dst_bg)
    if marker_result is None:
        return None
    out_inner, marker_points = marker_result

    output = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    paste(output[1:-1, 1:-1], out_inner, 0, 0)

    source_inner = _place_sources(sprite_types, source_type_ids, src_h, src_w, src_bg)
    if source_inner is None:
        return None

    source_rect = _build_frame(src_h, src_w, src_border, src_bg)
    paste(source_rect[1:-1, 1:-1], source_inner, 0, 0)

    dest_rect = _build_frame(dst_h, dst_w, dst_border, dst_bg)
    for rr, cc, type_id, _axis, _active, direction in marker_points:
        dest_rect[1 + rr, 1 + cc] = sprite_types[int(type_id)]["anchor"]
        dest_rect[1 + rr + direction[0], 1 + cc + direction[1]] = dst_border

    try:
        src_pos, dst_pos = _place_rectangles(layout, src_h, src_w, dst_h, dst_w)
    except ValueError:
        return None

    grid = _make_wallpaper(random.choice(["vertical", "horizontal"]), wallpaper)
    paste(grid, source_rect, src_pos[0], src_pos[1])
    paste(grid, dest_rect, dst_pos[0], dst_pos[1])

    if not np.array_equal(main(grid), output):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a251c730 combo input after {MAX_GEN_ATTEMPTS} attempts")
