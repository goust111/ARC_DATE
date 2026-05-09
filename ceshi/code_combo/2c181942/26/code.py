# Combo variant of 2c181942 (index 26)
# combo_spec: D6/1+D16/0
# Name: 同色干扰+严格4连通

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - Base task: a hollow 4x4 hub sits near the center; 2-4 monochrome sprites are
#      scattered elsewhere. A true sprite must be rotated so that one side has a
#      contiguous contact segment of length 2 and the whole mask is mirror-symmetric
#      across the contact axis. Output = hub plus the attached true sprites.
#    - D6/1 alone would introduce a same-color distractor, so color is no longer
#      sufficient for matching. D16/0 alone would change object extraction from
#      8-connectivity to 4-connectivity.
#    - In the fused version, the distractor is not a separate far-away shape. Instead,
#      one or two same-color single-pixel satellites are placed so that they touch the
#      true sprite only diagonally. Under 8-connectivity the union looks like one
#      plausible same-color silhouette; under 4-connectivity it splits into the true
#      sprite plus tiny distractor components.
#    - Small trace for one side color c:
#      original:
#        1. collect all c-colored cells outside the hub
#        2. extract one 8-connected component
#        3. rotate that component until the geometry matches
#      fused:
#        1. collect all c-colored cells outside the hub
#        2. extract 4-connected components
#        3. tiny diagonal satellites split off automatically
#        4. test each 4-component against the same attachment geometry
#        5. keep the unique large valid component and ignore the small invalid ones
#      difference:
#        the apparent same-color "whole sprite" in the input is an 8-connected illusion.
#        The correct parsing must happen at 4-connectivity before geometric matching.
# b) Complexity calibration
#    - Real sprites still come from a tiny canonical family with one valid rotation,
#      so the core task remains learnable from 2-4 examples.
#    - Every generated input contains at least one disguised sprite, making the visual
#      effect of the combo repeat across examples instead of appearing as a rare edge case.
#    - The distractor satellites are only 1-2 pixels each, so they are visible but do
#      not create a second full-scale object that would overload the scene.
# c) Implicit assumption audit of the original main()
#    - `m = (g == col) & (~hub_mask)` still holds: hub-edge color still selects the
#      candidate pool for that side.
#    - `label_mask(..., connectivity=8)` no longer holds. It would merge the diagonally
#      touching satellites into the true sprite and destroy the intended decomposition.
#    - "pick the largest component first" also no longer holds in the original sense,
#      because after switching to 4-connectivity there can be multiple same-color
#      components for the same side. The solver must test every 4-component and then
#      choose among geometrically valid candidates.
#    - Symmetry checks, contact-edge length checks, and placement conflict checks still
#      hold and remain the actual semantic test for a true sprite.
# d) Placement legality
#    - Output is still built first: place the hub, then attach only the true sprites.
#    - Input uses `random_free_location(..., padding=2, padding_connectivity=8)` on the
#      entire scattered patch, so different color groups stay separated from the hub and
#      from each other even when one patch internally contains diagonal satellites.
#    - Satellites are generated inside the scattered patch and only diagonally adjacent
#      to the core sprite, so the fused illusion is local and deterministic.
# e) Visual observability
#    - A human sees a same-color blob whose outline has one or two small diagonal nubs.
#      Across examples, those nubs disappear in the output while the main body attaches
#      to the matching hub edge. That is exactly the intended observable rule:
#      diagonal-only same-color pieces are not part of the true sprite.
#    - No hidden ordering by raw color id or any non-visual statistic is used.
# f) Visual tidiness
#    - There is no overlap and no mixed-color tagging inside a patch.
#    - The hub remains clean, the true sprite remains a simple monochrome object, and
#      the distractor satellites are small enough that the grid stays readable.
# g) Change inferability
#    - Color semantics are still structured: a hub side color identifies which scattered
#      patch belongs to that side.
#    - The only new color behavior is reuse of the same side color by diagonal satellites.
#      Because the satellites always sit adjacent to a matching-color sprite and vanish in
#      the output, the rule is inferable from position and connectivity, not from hidden
#      color-value conventions.


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.where(mask, 1, 0).astype(int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


def _edge_segment(mask: np.ndarray, side: str):
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    if side == "top":
        rr = int(pts[:, 0].min())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "bot":
        rr = int(pts[:, 0].max())
        cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
        return ("h", int(rr), cols)
    if side == "left":
        cc = int(pts[:, 1].min())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    if side == "right":
        cc = int(pts[:, 1].max())
        rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
        return ("v", int(cc), rows)
    return None


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


def _need_edge_for_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _valid_rotations_for_side(mask: np.ndarray, need_edge: str, target: int) -> list[tuple[np.ndarray, tuple]]:
    valid: list[tuple[np.ndarray, tuple]] = []
    for krot in range(4):
        rot = _crop_mask(np.rot90(mask, int(krot)))
        if need_edge in ("left", "right"):
            if not bool(np.array_equal(rot, np.flipud(rot))):
                continue
        else:
            if not bool(np.array_equal(rot, np.fliplr(rot))):
                continue
        seg = _edge_segment(rot, need_edge)
        if seg is None:
            continue
        if need_edge in ("left", "right"):
            _typ, _edge_c, rows = seg
            if len(rows) != int(target) or not _is_contig(rows):
                continue
        else:
            _typ, _edge_r, cols = seg
            if len(cols) != int(target) or not _is_contig(cols):
                continue
        valid.append((rot, seg))
    return valid


def _aligned_translation(mask: np.ndarray, need_edge: str, fixed: int, span: list[int], target: int) -> tuple[int, int] | None:
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None
    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target) or not _is_contig(rows):
            return None
        return int(span[0] - rows[0]), int(fixed - edge_c)
    _typ, edge_r, cols = seg
    if len(cols) != int(target) or not _is_contig(cols):
        return None
    return int(fixed - edge_r), int(span[0] - cols[0])


def _attach_specs(r: int, c: int, n: int, top_col: int, bot_col: int, left_col: int, right_col: int) -> dict[str, tuple[int, tuple[str, int, list[int]]]]:
    return {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
    for n in range(3, 7):
        for r in range(0, int(H - n + 1)):
            for c in range(0, int(W - n + 1)):
                if (
                    int(g[int(r), int(c)]) != int(bg)
                    or int(g[int(r), int(c + n - 1)]) != int(bg)
                    or int(g[int(r + n - 1), int(c)]) != int(bg)
                    or int(g[int(r + n - 1), int(c + n - 1)]) != int(bg)
                ):
                    continue
                inner = g[int(r + 1) : int(r + n - 1), int(c + 1) : int(c + n - 1)]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[int(r), int(c + 1) : int(c + n - 1)]
                bot = g[int(r + n - 1), int(c + 1) : int(c + n - 1)]
                left = g[int(r + 1) : int(r + n - 1), int(c)]
                right = g[int(r + 1) : int(r + n - 1), int(c + n - 1)]

                if bool((top == int(bg)).any()) or bool((bot == int(bg)).any()) or bool((left == int(bg)).any()) or bool((right == int(bg)).any()):
                    continue
                if (
                    len(set(map(int, top))) != 1
                    or len(set(map(int, bot))) != 1
                    or len(set(map(int, left))) != 1
                    or len(set(map(int, right))) != 1
                ):
                    continue

                cr = float(r + (n - 1) / 2)
                cc = float(c + (n - 1) / 2)
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (
                    float(score),
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo solver:
    - parse same-color patches with 4-connectivity
    - ignore diagonally attached same-color satellites
    - attach only the geometrically valid 4-connected component for each hub side
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]
    target = int(n - 2)
    segs = _attach_specs(r, c, n, top_col, bot_col, left_col, right_col)

    for side in ("top", "bot", "left", "right"):
        col, (need_edge, fixed, span) = segs[side]
        color_mask = (g == int(col)) & (~hub_mask)
        if not bool(color_mask.any()):
            continue

        # New assumption: diagonal-only neighbors must split here.
        labc, nc = label_mask(color_mask, connectivity=4)
        candidates = []
        for cid in range(1, int(nc) + 1):
            comp_mask = labc == int(cid)
            comp_arr = np.where(comp_mask, int(col), int(bg)).astype(int)
            comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)

            valid_rots = _valid_rotations_for_side(comp_mask, need_edge, target)
            if len(valid_rots) != 1:
                continue

            placed_mask, _seg = valid_rots[0]
            shift = _aligned_translation(placed_mask, need_edge, fixed, span, target)
            if shift is None:
                continue
            dr, dc = shift

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(placed_mask)}
            if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
                continue

            ok = True
            for R, C in placed:
                if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != int(col):
                    ok = False
                    break
            if not ok:
                continue

            arr = np.where(placed_mask, int(col), int(bg)).astype(int)
            candidates.append((int(placed_mask.sum()), arr, placed_mask, int(dr), int(dc)))

        if not candidates:
            continue

        candidates.sort(key=lambda item: item[0], reverse=True)
        _area, arr, mask, dr, dc = candidates[0]
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Build a true 4-connected sprite from a tiny canonical family.

    Canonical frame:
    - height 4
    - contact edge on the right
    - flipud symmetric
    - right edge has exactly two contiguous cells
    - not fliplr symmetric
    - exactly one valid rotation for left-side attachment tests
    """
    if int(edge_len) != 2:
        return None

    left_attach_caps = [
        np.array([1, 0], dtype=bool),
        np.array([1, 0, 0], dtype=bool),
        np.array([1, 1, 0], dtype=bool),
        np.array([1, 0, 0, 0], dtype=bool),
    ]
    rng.shuffle(left_attach_caps)

    for cap in left_attach_caps:
        w = int(cap.size)
        grid = np.zeros((4, w), dtype=bool)
        grid[0] = cap
        grid[1] = True
        grid[2] = True
        grid[3] = cap
        grid = _crop_mask(grid)
        if int(grid.sum()) > 10:
            continue
        if label_mask(grid, connectivity=4)[1] != 1:
            continue
        if len(_valid_rotations_for_side(grid, "right", edge_len)) != 1:
            continue

        if side == "left":
            return grid.copy()
        if side == "right":
            return np.fliplr(grid)
        if side == "top":
            return np.rot90(grid, -1)
        return np.rot90(grid, 1)

    return None


def _eight_neighbors(r: int, c: int) -> list[tuple[int, int]]:
    out = []
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            out.append((int(r + dr), int(c + dc)))
    return out


def _four_neighbors(r: int, c: int) -> list[tuple[int, int]]:
    return [(int(r - 1), int(c)), (int(r + 1), int(c)), (int(r), int(c - 1)), (int(r), int(c + 1))]


def _build_diagonal_disguise(core_mask: np.ndarray, rng: random.Random) -> np.ndarray | None:
    """
    Add 1-2 same-color satellite pixels that touch the core only diagonally.
    Result:
    - 8-connected as a whole patch
    - split into multiple components under 4-connectivity
    """
    h, w = core_mask.shape
    canvas = np.zeros((h + 4, w + 4), dtype=bool)
    off_r = 2
    off_c = 2
    canvas[off_r : off_r + h, off_c : off_c + w] = core_mask
    core = canvas.copy()
    H, W = canvas.shape

    candidates: list[tuple[int, int]] = []
    for r in range(H):
        for c in range(W):
            if bool(canvas[r, c]):
                continue
            if any(0 <= nr < H and 0 <= nc < W and bool(core[nr, nc]) for nr, nc in _four_neighbors(r, c)):
                continue
            if not any(0 <= nr < H and 0 <= nc < W and bool(core[nr, nc]) for nr, nc in _eight_neighbors(r, c)):
                continue
            candidates.append((int(r), int(c)))

    rng.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    want = rng.randint(1, 2)
    for r, c in candidates:
        if any(abs(r - rr) + abs(c - cc) == 1 for rr, cc in chosen):
            continue
        chosen.append((int(r), int(c)))
        if len(chosen) >= want:
            break

    if not chosen:
        return None

    for r, c in chosen:
        canvas[int(r), int(c)] = True

    patch = _crop_mask(canvas)
    _lab8, n8 = label_mask(patch, connectivity=8)
    _lab4, n4 = label_mask(patch, connectivity=4)
    if int(n8) != 1 or int(n4) <= 1:
        return None
    return patch


def _scatter_patch_from_mask(mask: np.ndarray, color: int, bg: int) -> np.ndarray:
    return np.where(mask, int(color), int(bg)).astype(int)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(m.shape[0] for m in sprite_masks.values())
    max_sprite_w = max(m.shape[1] for m in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 3

    hub_r_min = margin
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = top_col
        out[hub_r + n - 1, hub_c + ci] = bot_col
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = left_col
        out[hub_r + ri, hub_c + n - 1] = right_col

    segs = _attach_specs(hub_r, hub_c, n, top_col, bot_col, left_col, right_col)
    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        color = side_colors[side]
        arr = np.where(mask, color, bg).astype(int)
        _col, (need_edge, fixed, span) = segs[side]
        shift = _aligned_translation(mask, need_edge, fixed, span, edge_len)
        if shift is None:
            return None
        sr, sc = shift
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr, cc in np.argwhere(mask):
            if int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                return None
        paste(out, arr, sr, sc, transparent=bg)
        attached_sprites[side] = (arr, mask)

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    disguise_count = rng.randint(1, min(2, K))
    disguise_sides = set(rng.sample(sprite_sides, disguise_count))

    for side in sprite_sides:
        _arr, mask = attached_sprites[side]
        scatter_mask = mask.copy()
        if side in disguise_sides:
            disguised = _build_diagonal_disguise(mask, rng)
            if disguised is None:
                return None
            scatter_mask = disguised

        k = rng.choice([1, 2, 3])
        rotated_mask = _crop_mask(np.rot90(scatter_mask, k))
        patch = _scatter_patch_from_mask(rotated_mask, side_colors[side], bg)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    # Generator-side guarantee for the new implicit assumptions:
    # every disguised patch must be one 8-connected blob but multiple 4-components.
    found_disguised = False
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[hub_r : hub_r + n, hub_c : hub_c + n] = True
    for side in disguise_sides:
        color = side_colors[side]
        color_mask = (inp == int(color)) & (~hub_mask)
        _lab8, n8 = label_mask(color_mask, connectivity=8)
        _lab4, n4 = label_mask(color_mask, connectivity=4)
        if int(n8) < int(n4):
            found_disguised = True
            break
    if not found_disguised:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
