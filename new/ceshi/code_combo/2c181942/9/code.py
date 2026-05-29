# Combo variant of 2c181942 (index 9)
# combo_spec: D3/0+D7/2
# Name: 八变换空间+非对称精灵
#
# Design analysis
# ---------------
# 1. Concrete behavior trace:
#    - Original task:
#      a) match each sprite to a hub side by color
#      b) try 4 rotations
#      c) accept the first rotation whose contact edge has length N-2 and whose
#         whole silhouette is mirror-symmetric across the contact axis
#    - This combo:
#      a) color matching stays the same, so the visible pairing cue remains human-readable
#      b) symmetry is removed entirely; sprites are fully asymmetric across the full
#         dihedral orbit (4 rotations + 4 mirrored rotations)
#      c) for each matched sprite, main() now tries all 8 transforms and keeps the
#         unique one whose exposed contact side has exactly N-2 contiguous cells in
#         the correct place against the hub, plus the correct one-sided inner anchor
#         beside that contact run
#    - Small example:
#      Suppose a red sprite belongs on the hub's left side. In its solved orientation
#      the sprite exposes exactly two cells on its rightmost column. In the input, that
#      same sprite might be mirrored+rotated so the two-cell run sits on a top edge.
#      The old solver would never recover it because no pure rotation restores the
#      solved state. The combo solver enumerates the full 8-transform space and finds
#      the single transform that moves that two-cell run back to the right edge.
#
# 2. Complexity calibration:
#    - The hub and color pairing remain unchanged so the task is still learnable from
#      2-4 examples.
#    - The visible novelty comes from lopsided sprites and from some inputs requiring
#      an actual mirror operation, not just a quarter-turn.
#    - Freedom is controlled: contact length is still N-2 (=2 here), the hub stays 4x4,
#      and there are only 8 candidate transforms per sprite.
#
# 3. Hidden assumption review against original main():
#    - The original symmetry checks assumed every valid sprite had a global mirror axis.
#      That assumption is invalid here and is removed.
#    - The original "break on first valid rotation" assumed the first local match was
#      safe. That is no longer safe once mirrors are allowed, so the new solver counts
#      all valid transforms and requires uniqueness.
#    - The original code also implicitly assumed same-color clutter would not compete
#      with the true sprite. The generator preserves that assumption by keeping each
#      matched color to one largest 8-connected component outside the hub.
#
# 4. Placement legality:
#    - Generator still builds the solved output first, verifies all attached sprites are
#      in-bounds and non-overlapping, then scatters transformed copies with padding.
#    - Input scattering uses random_free_location(..., padding=2, padding_connectivity=8)
#      so sprites remain visually separated from the hub and from each other.
#
# 5. Visual observability:
#    - Humans can still infer "same color attaches to same-colored hub edge".
#    - The new clue visible across examples is that the source sprite can be a mirrored
#      version of the attached one, while the attached silhouette itself is visibly
#      asymmetric. This makes the combo distinguishable from both the original task and
#      the rotation-only D7 variant.
#
# 6. Visual neatness:
#    - No extra markers or overlays are introduced.
#    - Sprites remain monochrome and spatially separated, so the asymmetry is easy to read.
#
# 7. Generator-side guarantees for this combo:
#    - every matched sprite has full dihedral orbit size 8
#    - for its target hub side, exactly one of the 8 transforms yields a legal contact
#      signature: the right contact-run length plus a one-sided inner anchor that
#      operationalizes the "position match" part of the combo spec
#    - every generated sample includes at least one matched sprite whose recovery transform
#      has flip=True, so the 8-transform rule is exercised rather than degenerating to
#      a rotation-only sample
#
# 8. New implicit assumptions introduced by the combo solver:
#    - main() assumes that for each matched color, the largest 8-connected component is
#      the intended sprite and that its valid attachment transform is unique over the full
#      8-transform search space. generate_input() constructs exactly that.
#    - main() assumes the unique valid transform can be pasted without conflict onto the
#      hub layout. generate_input() constructs output-first and checks overlap before
#      accepting a sample.

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


TRANSFORMS: tuple[tuple[int, bool], ...] = tuple(
    (int(k), bool(flip)) for k in range(4) for flip in (False, True)
)
NON_IDENTITY_TRANSFORMS: tuple[tuple[int, bool], ...] = tuple(
    tf for tf in TRANSFORMS if tf != (0, False)
)
FLIP_TRANSFORMS: tuple[tuple[int, bool], ...] = tuple(tf for tf in NON_IDENTITY_TRANSFORMS if tf[1])


def _apply_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if bool(flip) else out


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    dummy = np.zeros(mask.shape, dtype=int)
    _, cropped = _crop_to_mask(dummy, mask)
    return cropped


def _mask_signature(mask: np.ndarray) -> tuple[int, int, bytes]:
    cropped = _crop_mask(mask)
    return int(cropped.shape[0]), int(cropped.shape[1]), cropped.astype(np.uint8).tobytes()


def _orbit_size(mask: np.ndarray) -> int:
    return len({_mask_signature(_apply_transform(mask, k, flip)) for k, flip in TRANSFORMS})


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    return int(_orbit_size(mask)) == 8


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


def _attachment_candidate(
    arr: np.ndarray,
    mask: np.ndarray,
    *,
    need_edge: str,
    fixed: int,
    span: list[int],
    target: int,
):
    arr, mask = _crop_to_mask(arr, mask)
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None

    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        if len(rows) != int(target) or not _is_contig(rows):
            return None
        inner_c = int(edge_c + 1) if need_edge == "left" else int(edge_c - 1)
        if not (0 <= inner_c < mask.shape[1]):
            return None
        if not bool(mask[int(rows[0]), int(inner_c)]):
            return None
        if bool(mask[int(rows[-1]), int(inner_c)]):
            return None
        dr = int(span[0] - rows[0])
        dc = int(fixed - edge_c)
    else:
        _typ, edge_r, cols = seg
        if len(cols) != int(target) or not _is_contig(cols):
            return None
        inner_r = int(edge_r + 1) if need_edge == "top" else int(edge_r - 1)
        if not (0 <= inner_r < mask.shape[0]):
            return None
        if not bool(mask[int(inner_r), int(cols[0])]):
            return None
        if bool(mask[int(inner_r), int(cols[-1])]):
            return None
        dr = int(fixed - edge_r)
        dc = int(span[0] - cols[0])

    return arr, mask, int(dr), int(dc)


def _count_valid_transforms(mask: np.ndarray, need_edge: str, target: int) -> int:
    count = 0
    dummy = np.where(mask, 1, 0).astype(int)
    for k, flip in TRANSFORMS:
        cand = _attachment_candidate(
            _apply_transform(dummy, k, flip),
            _apply_transform(mask, k, flip),
            need_edge=need_edge,
            fixed=0,
            span=list(range(int(target))),
            target=int(target),
        )
        if cand is not None:
            count += 1
    return int(count)


def _contact_side_for_hub_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _canonical_to_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return _apply_transform(mask, 0, True)
    if side == "top":
        return _apply_transform(mask, 3, False)
    return _apply_transform(mask, 1, False)


def _find_hub(g: np.ndarray, bg: int):
    h, w = g.shape
    best = None
    for n in range(3, 7):
        for r in range(0, int(h - n + 1)):
            for c in range(0, int(w - n + 1)):
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
                if (
                    bool((top == int(bg)).any())
                    or bool((bot == int(bg)).any())
                    or bool((left == int(bg)).any())
                    or bool((right == int(bg)).any())
                ):
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
                score = abs(float(cr - (h - 1) / 2)) + abs(float(cc - (w - 1) / 2))
                cand = (
                    float(score),
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
    return best


def _largest_component(mask: np.ndarray) -> np.ndarray | None:
    lab, nc = label_mask(mask, connectivity=8)
    best_id = None
    best_sz = -1
    for cid in range(1, int(nc) + 1):
        sz = int((lab == int(cid)).sum())
        if sz > best_sz:
            best_sz = sz
            best_id = int(cid)
    if best_id is None or best_sz <= 0:
        return None
    return lab == int(best_id)


def _extract_sprite(g: np.ndarray, hub_mask: np.ndarray, color: int, bg: int) -> tuple[np.ndarray, np.ndarray] | None:
    comp = _largest_component((g == int(color)) & (~hub_mask))
    if comp is None:
        return None
    arr = g.copy()
    arr[~comp] = int(bg)
    return _crop_to_mask(arr, comp)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - pair sprites to hub sides by color
    - search the full 8-transform dihedral space for each matched sprite
    - keep the unique transform whose contact edge fits the target hub side
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((h, w), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    target = int(n - 2)
    segs = {
        "top": (int(top_col), ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1))))),
        "bot": (int(bot_col), ("top", int(r + n), list(range(int(c + 1), int(c + n - 1))))),
        "left": (int(left_col), ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1))))),
        "right": (int(right_col), ("left", int(c + n), list(range(int(r + 1), int(r + n - 1))))),
    }

    for _side, (color, (need_edge, fixed, span)) in segs.items():
        sprite = _extract_sprite(g, hub_mask, int(color), bg)
        if sprite is None:
            continue
        spr_arr, spr_mask = sprite

        candidates = []
        for k, flip in TRANSFORMS:
            cand = _attachment_candidate(
                _apply_transform(spr_arr, k, flip),
                _apply_transform(spr_mask, k, flip),
                need_edge=need_edge,
                fixed=int(fixed),
                span=span,
                target=int(target),
            )
            if cand is not None:
                candidates.append((k, flip, cand))

        if len(candidates) != 1:
            continue

        _k, _flip, (arr, mask, dr, dc) = candidates[0]
        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(mask)}
        if any(not (0 <= rr < h and 0 <= cc2 < w) for rr, cc2 in placed):
            continue

        ok = True
        for rr, cc2 in placed:
            val = int(arr[int(rr - dr), int(cc2 - dc)])
            if val == int(bg):
                continue
            if int(out[int(rr), int(cc2)]) != int(bg) and int(out[int(rr), int(cc2)]) != val:
                ok = False
                break
        if not ok:
            continue

        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _make_canonical_mask(edge_len: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(800):
        height = rng.choice([4, 5, 6])
        width = rng.choice([3, 4, 5])
        run_start = rng.randint(0, int(height - edge_len))
        max_pixels = min(12, int(height * width))
        target_pixels = rng.randint(5, max_pixels)

        mask = np.zeros((height, width), dtype=bool)
        mask[int(run_start) : int(run_start + edge_len), int(width - 1)] = True
        mask[int(run_start), int(width - 2)] = True
        forbidden = {(int(run_start + edge_len - 1), int(width - 2))}

        while int(mask.sum()) < int(target_pixels):
            candidates = set()
            for rr, cc in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if not (0 <= nr < height and 0 <= nc < width):
                            continue
                        if bool(mask[int(nr), int(nc)]):
                            continue
                        if (int(nr), int(nc)) in forbidden:
                            continue
                        if int(nc) == int(width - 1) and not (int(run_start) <= int(nr) < int(run_start + edge_len)):
                            continue
                        candidates.add((int(nr), int(nc)))
            if not candidates:
                break

            cand_list = list(candidates)
            weights = []
            mid_r = float(run_start + (edge_len - 1) / 2)
            for nr, nc in cand_list:
                weight = 1.0
                weight += float(width - 1 - nc) * 2.5
                weight += abs(float(nr) - mid_r) * 0.8
                if int(nc) <= int(width - 3):
                    weight += 2.0
                if not (int(run_start) <= int(nr) < int(run_start + edge_len)):
                    weight += 1.5
                weights.append(weight)
            nr, nc = rng.choices(cand_list, weights=weights, k=1)[0]
            mask[int(nr), int(nc)] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < 5 or int(mask.sum()) > 12:
            continue
        _lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_transforms(mask, "right", edge_len) != 1:
            continue
        return mask
    return None


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    need_edge = _contact_side_for_hub_side(side)
    for _ in range(300):
        canonical = _make_canonical_mask(edge_len, rng)
        if canonical is None:
            continue
        mask = _canonical_to_side(canonical, side)
        if not _is_fully_asymmetric(mask):
            continue
        if _count_valid_transforms(mask, need_edge, edge_len) != 1:
            continue
        return mask
    return None


def _behavior_metrics(inp: np.ndarray) -> dict[str, int]:
    g = np.array(inp, dtype=int)
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    if hub is None:
        return {"matched_sprites": 0, "flip_required_count": 0, "rotation_only_count": 0}

    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros(g.shape, dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True
    target = int(n - 2)
    segs = {
        "top": (int(top_col), "bot"),
        "bot": (int(bot_col), "top"),
        "left": (int(left_col), "right"),
        "right": (int(right_col), "left"),
    }

    matched = 0
    flip_required = 0
    rotation_only = 0
    for _side, (color, need_edge) in segs.items():
        sprite = _extract_sprite(g, hub_mask, color, bg)
        if sprite is None:
            continue
        matched += 1
        arr, mask = sprite
        valids = []
        for k, flip in TRANSFORMS:
            cand = _attachment_candidate(
                _apply_transform(arr, k, flip),
                _apply_transform(mask, k, flip),
                need_edge=need_edge,
                fixed=0,
                span=list(range(int(target))),
                target=int(target),
            )
            if cand is not None:
                valids.append((k, flip))
        if len(valids) == 1:
            _k, flip = valids[0]
            if flip:
                flip_required += 1
            else:
                rotation_only += 1

    return {
        "matched_sprites": int(matched),
        "flip_required_count": int(flip_required),
        "rotation_only_count": int(rotation_only),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    h = rng.randint(18, 26)
    w = rng.randint(20, 26)

    all_colors = [col for col in range(10) if col != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]

    sides = ["top", "bot", "left", "right"]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    sprite_sides = rng.sample(sides, rng.randint(2, 4))
    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = h - n - margin
    hub_c_min = margin
    hub_c_max = w - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((h, w), bg, dtype=int)
    for ci in range(1, n - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + n - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, n - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + n - 1] = int(right_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        sh, sw = mask.shape
        color = int(side_colors[side])
        arr = np.where(mask, color, bg).astype(int)

        if side == "top":
            filled = np.where(mask[-1])[0]
            sr = int(hub_r - sh)
            sc = int(hub_c + 1 - filled[0])
        elif side == "bot":
            filled = np.where(mask[0])[0]
            sr = int(hub_r + n)
            sc = int(hub_c + 1 - filled[0])
        elif side == "left":
            filled = np.where(mask[:, -1])[0]
            sr = int(hub_r + 1 - filled[0])
            sc = int(hub_c - sw)
        else:
            filled = np.where(mask[:, 0])[0]
            sr = int(hub_r + 1 - filled[0])
            sc = int(hub_c + n)

        if sr < 0 or sc < 0 or sr + sh > h or sc + sw > w:
            return None

        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[int(rr), int(cc)]) and int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None

        paste(out, arr, int(sr), int(sc), transparent=bg)
        attached_sprites[side] = (arr, mask)

    inp = np.full((h, w), bg, dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    forced_flip_side = rng.choice(sprite_sides)
    for side in sprite_sides:
        arr, mask = attached_sprites[side]
        if side == forced_flip_side:
            k, flip = rng.choice(FLIP_TRANSFORMS)
        else:
            k, flip = rng.choice(NON_IDENTITY_TRANSFORMS)
        transformed_arr = _apply_transform(arr, k, flip)
        transformed_mask = _apply_transform(mask, k, flip)
        cropped_arr, cropped_mask = _crop_to_mask(transformed_arr, transformed_mask)
        patch = np.where(cropped_mask, cropped_arr, bg).astype(int)

        try:
            pr, pc = random_free_location(
                canvas,
                patch,
                transparent=bg,
                padding=2,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.3:
        used_colors = set(side_colors.values())
        distractor_colors = [col for col in range(10) if col != bg and col not in used_colors]
        if distractor_colors:
            dcol = int(rng.choice(distractor_colors))
            dist_mask = _make_sprite_mask(edge_len, rng.choice(sides), rng)
            if dist_mask is not None:
                dk, dflip = rng.choice(NON_IDENTITY_TRANSFORMS)
                dist_arr = np.where(dist_mask, dcol, bg).astype(int)
                dist_arr = _apply_transform(dist_arr, dk, dflip)
                dist_mask = _apply_transform(dist_mask, dk, dflip)
                dist_arr, dist_mask = _crop_to_mask(dist_arr, dist_mask)
                patch = np.where(dist_mask, dist_arr, bg).astype(int)
                try:
                    pr, pc = random_free_location(
                        canvas,
                        patch,
                        transparent=bg,
                        padding=2,
                        padding_connectivity=8,
                    )
                    paste(canvas, patch, pr, pc, transparent=bg)
                    paste(inp, patch, pr, pc, transparent=bg)
                except ValueError:
                    pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None
    if _behavior_metrics(inp)["flip_required_count"] < 1:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
