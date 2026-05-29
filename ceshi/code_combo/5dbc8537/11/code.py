# Combo variant of 5dbc8537 (index 11)
# combo_spec: D4/3+D7/0
# Name: 居中模板四方向镜像填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
)

# Design analysis for the combo variant
# -------------------------------------
# concrete behavior:
# - The original task used one edge-hugging strip template, so a hole could only be "early/late"
#   along one long axis. This combo replaces that with a floating template centered in the grid.
# - The filled template is built as a clean plus-like frame: top, bottom, left, and right arms
#   each contain rectangular holes, while the center and corners stay base-colored.
# - Outside sprites are scattered in the four side bands around the centered template, not in a
#   single L-shaped leftover region. A sprite lying above or below the template is mirrored
#   left-right before insertion; a sprite lying left or right of the template is mirrored top-bottom.
# - Small example trace: if the output template has a 3x4 top-arm hole filled with an asymmetric
#   patch `A`, the input shows the raw 3x4 patch above the template. The old solver would paste `A`
#   unchanged; the combo solver pastes `fliplr(A)`. A left-side 4x2 patch instead becomes `flipud(A)`.
#
# complexity calibration:
# - The four-direction layout must be visually obvious, so the generator always creates holes on all
#   four sides of the template and places source sprites in matching side bands.
# - Mirror visibility must also be obvious, so every non-dot sprite uses an asymmetric two-color
#   pattern that is guaranteed to change under its assigned mirror axis.
# - To avoid size-matching ambiguity after introducing source-side-dependent transforms, the solver
#   matches by `(side, bbox-size)`: a top sprite only fills a top hole, a left sprite only fills a
#   left hole, and within each side arm the two hole sizes are distinct.
#
# implicit assumption audit:
# - `base_col = largest non-bg connected component` still holds because the centered frame remains
#   much larger than any sprite and stays 8-connected.
# - `dst_bbox = bbox(base pixels)` still works even though the template no longer touches the grid
#   border; the bbox is simply an interior floating rectangle.
# - The original exact-cover DFS assumed rectangular hole tiles and rectangular sprites. That remains
#   true, but the generator now additionally guarantees one unique candidate slot per non-dot sprite.
# - The new solver step that infers the mirror axis from the sprite's side assumes every sprite lies
#   wholly in one cardinal side band (top / bottom / left / right) relative to the template. The
#   generator enforces this by never placing sprites in corner zones.
#
# legality / cleanliness / observability:
# - Holes never overlap and stay inside the centered template bbox.
# - Sprites never overlap each other or the template and keep a one-cell 8-neighborhood gap.
# - No hidden use of color ids controls behavior; only visible spatial side-of-template relations do.
# - The output stays tidy because each side arm contains a small number of rectangular holes separated
#   by base-colored dividers, so humans can clearly see which source sprite goes with which direction.

GRID_MIN, GRID_MAX = 21, 25
MARGIN_MIN = 4
TEMPLATE_CHOICES = (13, 15)
CENTER_CHOICES = (7,)
ARM_MIN, ARM_MAX = 2, 4
MIN_CHANGED_CELLS = 12
MAX_RETRIES = 500


def _side_to_kind(side: str) -> str:
    if side in {"top", "bottom"}:
        return "h"
    if side in {"left", "right"}:
        return "v"
    raise ValueError(f"unknown side: {side}")


def _apply_transform(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == "h":
        return np.fliplr(arr)
    if kind == "v":
        return np.flipud(arr)
    raise ValueError(f"unknown transform kind: {kind}")


def _sprite_side_from_bbox(bbox: tuple[int, int, int, int], dst_bbox: tuple[int, int, int, int]) -> str:
    pr0, pc0, pr1, pc1 = map(int, bbox)
    r0, c0, r1, c1 = map(int, dst_bbox)
    if pr1 < r0:
        return "top"
    if pr0 > r1:
        return "bottom"
    if pc1 < c0:
        return "left"
    if pc0 > c1:
        return "right"
    raise AssertionError("sprite is not fully outside the template bbox")


def _hole_side(rr0: int, cc0: int, h: int, w: int, dh: int, dw: int) -> str:
    center_r2 = 2 * int(rr0) + int(h) - 1
    center_c2 = 2 * int(cc0) + int(w) - 1
    mid_r2 = int(dh) - 1
    mid_c2 = int(dw) - 1
    dr2 = center_r2 - mid_r2
    dc2 = center_c2 - mid_c2
    if abs(dr2) > abs(dc2):
        return "top" if dr2 < 0 else "bottom"
    return "left" if dc2 < 0 else "right"


def _is_exact_hole_rect(hole_mask: np.ndarray, rr0: int, cc0: int, h: int, w: int) -> bool:
    dh, dw = map(int, hole_mask.shape)
    if not bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
        return False
    if rr0 > 0 and bool(hole_mask[rr0 - 1, cc0 : cc0 + w].any()):
        return False
    if rr0 + h < dh and bool(hole_mask[rr0 + h, cc0 : cc0 + w].any()):
        return False
    if cc0 > 0 and bool(hole_mask[rr0 : rr0 + h, cc0 - 1].any()):
        return False
    if cc0 + w < dw and bool(hole_mask[rr0 : rr0 + h, cc0 + w].any()):
        return False
    return True


def _axis_parts(inner: int) -> tuple[int, int, int] | None:
    options: list[tuple[int, int, int]] = []
    for center in CENTER_CHOICES:
        remain = int(inner) - int(center)
        if remain < 2 * ARM_MIN:
            continue
        for a in range(ARM_MIN, ARM_MAX + 1):
            b = remain - a
            if ARM_MIN <= b <= ARM_MAX:
                options.append((int(a), int(center), int(b)))
    if not options:
        return None
    return random.choice(options)


def _split_two(span: int) -> tuple[int, int] | None:
    options = [
        (a, span - 1 - a)
        for a in range(2, span - 2)
        if span - 1 - a >= 2 and a != span - 1 - a
    ]
    if not options:
        return None
    return random.choice(options)


def _build_rects(dh: int, dw: int) -> list[dict] | None:
    ih, iw = int(dh) - 2, int(dw) - 2
    row_parts = _axis_parts(ih)
    col_parts = _axis_parts(iw)
    if row_parts is None or col_parts is None:
        return None
    top_h, center_h, bottom_h = row_parts
    left_w, center_w, right_w = col_parts

    top_split = _split_two(center_w)
    bottom_split = _split_two(center_w)
    left_split = _split_two(center_h)
    right_split = _split_two(center_h)
    if None in {top_split, bottom_split, left_split, right_split}:
        return None
    top_w1, top_w2 = top_split
    bottom_w1, bottom_w2 = bottom_split
    left_h1, left_h2 = left_split
    right_h1, right_h2 = right_split

    mid_r0 = 1 + top_h
    mid_c0 = 1 + left_w
    bottom_r0 = 1 + top_h + center_h
    right_c0 = 1 + left_w + center_w

    rects = [
        {"r": 1, "c": mid_c0, "h": top_h, "w": top_w1, "side": "top"},
        {"r": 1, "c": mid_c0 + top_w1 + 1, "h": top_h, "w": top_w2, "side": "top"},
        {"r": bottom_r0, "c": mid_c0, "h": bottom_h, "w": bottom_w1, "side": "bottom"},
        {"r": bottom_r0, "c": mid_c0 + bottom_w1 + 1, "h": bottom_h, "w": bottom_w2, "side": "bottom"},
        {"r": mid_r0, "c": 1, "h": left_h1, "w": left_w, "side": "left"},
        {"r": mid_r0 + left_h1 + 1, "c": 1, "h": left_h2, "w": left_w, "side": "left"},
        {"r": mid_r0, "c": right_c0, "h": right_h1, "w": right_w, "side": "right"},
        {"r": mid_r0 + right_h1 + 1, "c": right_c0, "h": right_h2, "w": right_w, "side": "right"},
    ]

    dims_by_side: dict[str, set[tuple[int, int]]] = {}
    for rec in rects:
        side = str(rec["side"])
        dims_by_side.setdefault(side, set()).add((int(rec["h"]), int(rec["w"])))
    if any(len(dims_by_side.get(side, set())) != 2 for side in ("top", "bottom", "left", "right")):
        return None
    if min(int(rec["h"]) for rec in rects) < 2 or min(int(rec["w"]) for rec in rects) < 2:
        return None
    return rects


def _make_visible_patch(h: int, w: int, cols: tuple[int, int], kind: str) -> np.ndarray:
    a, b = map(int, cols)
    for _ in range(40):
        patch = np.full((h, w), a, dtype=int)
        if kind == "h":
            cut = random.randint(1, w - 1)
            patch[:, :cut] = b
            if h > 1:
                rr0 = random.randint(0, h - 1)
                cc0 = random.randint(0, cut - 1)
                patch[rr0:, cc0] = a
        else:
            cut = random.randint(1, h - 1)
            patch[:cut, :] = b
            if w > 1:
                rr0 = random.randint(0, cut - 1)
                cc0 = random.randint(0, w - 1)
                patch[rr0, cc0:] = a
        if not np.array_equal(patch, _apply_transform(patch, kind)):
            return patch
    raise AssertionError("failed to build asymmetric patch")


def _make_pieces(rects: list[dict], bg: int, base: int) -> dict[int, np.ndarray]:
    palette = [c for c in range(10) if c not in (bg, base)]
    assert len(palette) >= 6
    random.shuffle(palette)
    pieces: dict[int, np.ndarray] = {}
    for i, rec in enumerate(rects):
        a = int(palette[i % len(palette)])
        choices = [c for c in palette if c != a]
        b = int(random.choice(choices))
        kind = _side_to_kind(str(rec["side"]))
        pieces[i] = _make_visible_patch(int(rec["h"]), int(rec["w"]), (a, b), kind)
    return pieces


def _render_output(dh: int, dw: int, base: int, rects: list[dict], pieces: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, rec in enumerate(rects):
        patch = _apply_transform(pieces[i], _side_to_kind(str(rec["side"])))
        paste(out, patch, int(rec["r"]), int(rec["c"]))
    return out


def _render_old_output(dh: int, dw: int, base: int, rects: list[dict], pieces: dict[int, np.ndarray]) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, rec in enumerate(rects):
        paste(out, pieces[i], int(rec["r"]), int(rec["c"]))
    return out


def _can_place(occupied: np.ndarray, r: int, c: int, h: int, w: int, padding: int = 1) -> bool:
    H, W = occupied.shape
    if r < 0 or c < 0 or r + h > H or c + w > W:
        return False
    rr0 = max(0, r - padding)
    rr1 = min(H, r + h + padding)
    cc0 = max(0, c - padding)
    cc1 = min(W, c + w + padding)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _place_in_zone(
    inp: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    zone: tuple[int, int, int, int],
) -> bool:
    zr0, zc0, zr1, zc1 = map(int, zone)
    h, w = map(int, patch.shape)
    if zr1 - zr0 + 1 < h or zc1 - zc0 + 1 < w:
        return False
    candidates = [
        (r, c)
        for r in range(zr0, zr1 - h + 2)
        for c in range(zc0, zc1 - w + 2)
    ]
    random.shuffle(candidates)
    for r, c in candidates:
        if _can_place(occupied, r, c, h, w, padding=1):
            paste(inp, patch, r, c)
            occupied[r : r + h, c : c + w] = True
            return True
    return False


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[dict],
    pieces: dict[int, np.ndarray],
) -> np.ndarray | None:
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for rec in rects:
        r = tr0 + int(rec["r"])
        c = tc0 + int(rec["c"])
        h = int(rec["h"])
        w = int(rec["w"])
        inp[r : r + h, c : c + w] = bg

    occupied = np.zeros((H, W), dtype=bool)
    occupied[tr0 : tr0 + dh, tc0 : tc0 + dw] = True
    zones = {
        "top": (0, tc0, tr0 - 1, tc0 + dw - 1),
        "bottom": (tr0 + dh, tc0, H - 1, tc0 + dw - 1),
        "left": (tr0, 0, tr0 + dh - 1, tc0 - 1),
        "right": (tr0, tc0 + dw, tr0 + dh - 1, W - 1),
    }

    for side in ("top", "bottom", "left", "right"):
        ids = [i for i, rec in enumerate(rects) if rec["side"] == side]
        ids.sort(key=lambda i: pieces[i].size, reverse=True)
        for i in ids:
            if not _place_in_zone(inp, occupied, pieces[i], zones[side]):
                return None
    return inp


def _behavior_metrics(rects: list[dict], pieces: dict[int, np.ndarray]) -> tuple[dict[str, int], int]:
    side_counts = {"top": 0, "bottom": 0, "left": 0, "right": 0}
    changed_pieces = 0
    for i, rec in enumerate(rects):
        side = str(rec["side"])
        side_counts[side] += 1
        kind = _side_to_kind(side)
        if not np.array_equal(pieces[i], _apply_transform(pieces[i], kind)):
            changed_pieces += 1
    return side_counts, changed_pieces


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a centered floating template by inserting outside sprites, with mirror direction taken from the sprite side.

    Input structure:
    - The most common color is the background. A large single-color template floats away from all four grid edges.
    - Inside that template are rectangular background holes arranged around a solid center, while rectangular source sprites are scattered above, below, left, and right of the template.

    Rule:
    - Find the template from the largest non-background connected component and crop to its bbox.
    - Match each outside sprite to a same-sized rectangular hole on the corresponding side of the template, then mirror the sprite according to which side of the template it came from.
    - Sprites from above or below are flipped left-right; sprites from the left or right are flipped top-bottom.

    Key insight:
    - The size-matching logic from the original jigsaw task still solves the placement, but the final pasted orientation is no longer fixed: the sprite's visible cardinal position around the centered template tells you which mirror to apply.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(max_area, int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size > 0
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst == int(bg)
    out = np.full((dh, dw), int(base_col), dtype=int)

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        bbox = tuple(map(int, comp["bbox"]))
        pr0, pc0, pr1, pc1 = bbox
        patch = crop_bbox(g, bbox)
        ph, pw = map(int, patch.shape)
        mask = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        side = _sprite_side_from_bbox(bbox, dst_bbox)
        sprites.append(
            {
                "h": int(ph),
                "w": int(pw),
                "mask": mask,
                "patch": patch,
                "side": side,
                "kind": _side_to_kind(side),
            }
        )

    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))

    areas = [int(s["h"] * s["w"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())

    candidates: list[list[tuple[int, int]]] = []
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        side = str(sp["side"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(dh - h + 1):
                for cc0 in range(dw - w + 1):
                    if not _is_exact_hole_rect(hole_mask, rr0, cc0, h, w):
                        continue
                    if _hole_side(rr0, cc0, h, w, dh, dw) != side:
                        continue
                    pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)
    used = ~hole_mask
    placement: list[tuple[int, int] | None] = [None] * len(sprites)

    def dfs(i: int) -> bool:
        if i == len(sprites):
            return True
        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(i + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)

    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        sp = sprites[i]
        patch = _apply_transform(np.array(sp["patch"], dtype=int), str(sp["kind"]))
        mask = _apply_transform(np.array(sp["mask"], dtype=int), str(sp["kind"])).astype(bool)
        paste(out, patch, rr0, cc0, mask=mask, overwrite=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(GRID_MIN, GRID_MAX)
    W = random.randint(GRID_MIN, GRID_MAX)

    dh_choices = [d for d in TEMPLATE_CHOICES if d <= H - 2 * MARGIN_MIN]
    dw_choices = [d for d in TEMPLATE_CHOICES if d <= W - 2 * MARGIN_MIN]
    if not dh_choices or not dw_choices:
        return None
    dh = random.choice(dh_choices)
    dw = random.choice(dw_choices)
    tr0 = (H - dh) // 2
    tc0 = (W - dw) // 2

    rects = _build_rects(dh, dw)
    if rects is None:
        return None

    bg, base = random.sample(range(10), 2)
    pieces = _make_pieces(rects, bg, base)
    side_counts, changed_pieces = _behavior_metrics(rects, pieces)
    if changed_pieces != len(rects):
        return None
    if any(side_counts[side] == 0 for side in side_counts):
        return None

    out = _render_output(dh, dw, base, rects, pieces)
    old_out = _render_old_output(dh, dw, base, rects, pieces)
    if int(np.sum(out != old_out)) < MIN_CHANGED_CELLS:
        return None

    inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces)
    if inp is None:
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
