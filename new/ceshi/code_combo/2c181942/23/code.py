# Combo variant of 2c181942 (index 23)
# combo_spec: D2/1+D7/2
# Name: 尺寸排序+非对称精灵
#
# Design analysis for this fused variant
# -------------------------------------
# spec_variant:
#   core_preserved:
#     - grid still uses a central hollow hub plus scattered monochrome sprites
#     - output still keeps only the hub and the correctly attached sprites
#     - sprites are still 8-connected and are detached/scattered in the input
#   d2_change_realized:
#     - the hub is no longer a uniform 4x4 square with equal side lengths
#     - instead, one common hollow rectangular box carries four colored side
#       segments, one per side, and the four segment lengths are all distinct
#       (generator uses a permutation of lengths 2,3,4,5)
#     - sprites no longer match sides by color; each sprite has an intrinsic
#       contact-edge length, and solver sorts sprite lengths against hub-side
#       lengths to determine the pairing
#   d7_change_realized:
#     - every sprite silhouette is fully asymmetric across all non-identity D4
#       transforms
#     - for the already-determined target side, solver tries all four rotations
#       and accepts the sprite only if exactly one rotation exposes the required
#       contact edge on the required side
#   behavior_trace:
#     - original task:
#       1. read hub side color
#       2. take the same-colored sprite
#       3. rely on symmetry to choose orientation
#       4. attach to a side of fixed length N-2
#     - fused task:
#       1. detect the four hub-side segment lengths, e.g. [5,4,3,2]
#       2. detect the four scattered sprites and measure each sprite's intrinsic
#          contact length, e.g. [2,5,3,4]
#       3. sort both multisets by length and pair equal ranks:
#          5<->5, 4<->4, 3<->3, 2<->2
#       4. for each paired sprite/side, test rot90 x 0..3 and keep the unique
#          rotation whose exposed contact side has the target run length and span
#       5. paste the rotated sprite flush against that side segment
#     - concrete visible difference:
#       the output pairing can no longer be inferred from color; a red sprite may
#       attach to a green hub side if their ranked lengths match
#   complexity_calibration:
#     - the sortable lengths come from a small repeated vocabulary {2,3,4,5},
#       which is visible across 2-4 examples and is enough for humans to infer
#       the ranking rule
#     - the hub remains one coherent central object, so the scene is still easy
#       to parse despite the added degrees of freedom
#     - exactly four sprites are generated so the full ranking structure is shown
#       in every sample instead of hiding part of the order
#   hidden_assumptions_review:
#     - original main() assumed color identified the side; that assumption is
#       removed, so solver must enumerate all sprites first and sort them by
#       measured geometry
#     - original _find_hub() assumed a square with full colored sides; that is
#       replaced by a search for a hollow box whose boundary has exactly one
#       contiguous colored segment per side
#     - original rotation loop stopped at the first valid orientation; here that
#       would be unsafe, so main() counts candidates and requires uniqueness
#     - new assumption introduced by solver:
#       each relevant sprite has one unique maximum boundary run length equal to
#       its intended contact length; generator enforces that explicitly
#     - new assumption introduced by solver:
#       the hub side lengths are pairwise distinct; generator fixes this with a
#       permutation of [2,3,4,5]
#   placement_legality:
#     - generator still works output-first, then scatters sprites
#     - attached placements are checked for bounds and overlap before acceptance
#     - scattered placements use random_free_location(..., padding=2,
#       padding_connectivity=8), preserving visual separation
#   visual_observability:
#     - humans can see that hub side colors and sprite colors do not match, so
#       color cannot be the rule
#     - they can also see repeated side-length categories 2/3/4/5 across examples
#       and that the attached sprite edge lengths line up with those side lengths
#     - asymmetry is visible because the silhouettes are visibly lopsided and must
#       be rotated into one precise orientation before attachment
#   neatness:
#     - there are no overlays, labels, or stacked signals in one cell
#     - the hub sits centrally with ample margin, and sprites are scattered with
#       padding so each object remains visually isolated
#   generator_guarantees_for_main:
#     - one hub candidate with four distinct side lengths exists near the center
#     - exactly four outside components represent sprites
#     - for each sprite, its intrinsic measured contact length is unique among
#       the four sprites and equals one hub side length
#     - for the paired hub side, exactly one rotation satisfies the contact-side
#       geometry checked by main()

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


SIDE_ORDER = ["top", "bot", "left", "right"]


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


def _is_contig(seq: list[int]) -> bool:
    return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))


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


def _edge_run_length(mask: np.ndarray, side: str) -> int:
    seg = _edge_segment(mask, side)
    if seg is None:
        return 0
    _typ, _fixed, seq = seg
    return int(len(seq)) if _is_contig(seq) else 0


def _all_edge_run_lengths(mask: np.ndarray) -> dict[str, int]:
    return {side: int(_edge_run_length(mask, side)) for side in SIDE_ORDER}


def _measure_contact_length(mask: np.ndarray) -> int:
    runs = list(_all_edge_run_lengths(mask).values())
    return int(max(runs)) if runs else 0


def _rotation_attachment(
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
        dr = int(span[0] - rows[0])
        dc = int(fixed - edge_c)
    else:
        _typ, edge_r, cols = seg
        if len(cols) != int(target) or not _is_contig(cols):
            return None
        dr = int(fixed - edge_r)
        dc = int(span[0] - cols[0])

    return arr, mask, int(dr), int(dc)


def _count_valid_rotations(mask: np.ndarray, need_edge: str, target: int) -> int:
    count = 0
    dummy = np.where(mask, 1, 0).astype(int)
    for krot in range(4):
        cand = _rotation_attachment(
            np.rot90(dummy, int(krot)),
            np.rot90(mask, int(krot)),
            need_edge=need_edge,
            fixed=0,
            span=list(range(int(target))),
            target=int(target),
        )
        if cand is not None:
            count += 1
    return int(count)


def _is_fully_asymmetric(mask: np.ndarray) -> bool:
    variants = [
        np.rot90(mask, 1),
        np.rot90(mask, 2),
        np.rot90(mask, 3),
        np.fliplr(mask),
        np.flipud(mask),
        mask.T,
        np.fliplr(mask.T),
    ]
    return all(not np.array_equal(mask, other) for other in variants)


def _contact_side_for_hub_side(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _rotate_canonical_to_side(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _side_span(side_info: dict[str, int]) -> list[int]:
    return list(range(int(side_info["start"]), int(side_info["end"]) + 1))


def _extract_line_segment(line: np.ndarray, bg: int) -> tuple[int, int, int] | None:
    idx = [int(i) for i, v in enumerate(line) if int(v) != int(bg)]
    if not idx:
        return None
    if not _is_contig(idx):
        return None
    cols = {int(line[i]) for i in idx}
    if len(cols) != 1:
        return None
    return int(idx[0]), int(idx[-1]), int(next(iter(cols)))


def _find_hub(grid: np.ndarray, bg: int):
    g = np.array(grid, dtype=int)
    h, w = g.shape
    best = None  # (score, info)
    for box_h in range(5, min(12, h) + 1):
        for box_w in range(5, min(12, w) + 1):
            for r0 in range(0, int(h - box_h + 1)):
                r1 = int(r0 + box_h - 1)
                for c0 in range(0, int(w - box_w + 1)):
                    c1 = int(c0 + box_w - 1)
                    inner = g[int(r0 + 1) : int(r1), int(c0 + 1) : int(c1)]
                    if inner.size == 0 or not bool((inner == int(bg)).all()):
                        continue
                    if (
                        int(g[int(r0), int(c0)]) != int(bg)
                        or int(g[int(r0), int(c1)]) != int(bg)
                        or int(g[int(r1), int(c0)]) != int(bg)
                        or int(g[int(r1), int(c1)]) != int(bg)
                    ):
                        continue

                    raw = {
                        "top": _extract_line_segment(g[int(r0), int(c0 + 1) : int(c1)], int(bg)),
                        "bot": _extract_line_segment(g[int(r1), int(c0 + 1) : int(c1)], int(bg)),
                        "left": _extract_line_segment(g[int(r0 + 1) : int(r1), int(c0)], int(bg)),
                        "right": _extract_line_segment(g[int(r0 + 1) : int(r1), int(c1)], int(bg)),
                    }
                    if any(raw[side] is None for side in SIDE_ORDER):
                        continue

                    side_infos: dict[str, dict[str, int]] = {}
                    lengths: list[int] = []
                    colors: list[int] = []
                    for side in SIDE_ORDER:
                        start_idx, end_idx, color = raw[side]
                        if side in ("top", "bot"):
                            start = int(c0 + 1 + start_idx)
                            end = int(c0 + 1 + end_idx)
                            fixed = int(r0 if side == "top" else r1)
                        else:
                            start = int(r0 + 1 + start_idx)
                            end = int(r0 + 1 + end_idx)
                            fixed = int(c0 if side == "left" else c1)
                        length = int(end - start + 1)
                        side_infos[side] = {
                            "side": side,
                            "color": int(color),
                            "start": int(start),
                            "end": int(end),
                            "fixed": int(fixed),
                            "length": int(length),
                        }
                        lengths.append(int(length))
                        colors.append(int(color))

                    if len(set(lengths)) != 4 or len(set(colors)) != 4:
                        continue

                    area = int(box_h * box_w)
                    center_r = float(r0 + r1) / 2.0
                    center_c = float(c0 + c1) / 2.0
                    center_dist = abs(center_r - float(h - 1) / 2.0) + abs(center_c - float(w - 1) / 2.0)
                    score = (-area, center_dist)
                    info = {
                        "r0": int(r0),
                        "c0": int(c0),
                        "r1": int(r1),
                        "c1": int(c1),
                        "h": int(box_h),
                        "w": int(box_w),
                        "sides": side_infos,
                    }
                    if best is None or score < best[0]:
                        best = (score, info)
    return None if best is None else best[1]


def _component_infos(grid: np.ndarray, bg: int, hub_box_mask: np.ndarray) -> list[dict[str, object]]:
    object_mask = (grid != int(bg)) & (~hub_box_mask)
    lab, nc = label_mask(object_mask, connectivity=8)
    infos: list[dict[str, object]] = []
    for cid in range(1, int(nc) + 1):
        mask = lab == int(cid)
        if not bool(mask.any()):
            continue
        arr = np.where(mask, grid, int(bg))
        arr, mask = _crop_to_mask(arr, mask)
        color_vals = {int(v) for v in arr[mask]}
        if len(color_vals) != 1:
            continue
        contact_len = int(_measure_contact_length(mask))
        infos.append(
            {
                "arr": arr,
                "mask": mask,
                "color": int(next(iter(color_vals))),
                "size": int(mask.sum()),
                "contact_len": int(contact_len),
            }
        )
    return infos


def _select_sprites_by_length(component_infos: list[dict[str, object]]) -> dict[int, dict[str, object]]:
    best_by_len: dict[int, dict[str, object]] = {}
    for info in component_infos:
        length = int(info["contact_len"])
        if length <= 0:
            continue
        prev = best_by_len.get(int(length))
        if prev is None or int(info["size"]) > int(prev["size"]):
            best_by_len[int(length)] = info
    return best_by_len


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused D2/1 + D7/2 solver.

    Pairing logic:
    - detect the central hollow hub with four distinct side-segment lengths
    - detect outside sprites and measure each sprite's intrinsic contact length
    - sort/pair by contact length rather than color
    - for each paired side, try all four rotations and keep the unique valid one
    """
    g = np.array(grid, dtype=int)
    h, w = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, int(bg))
    assert hub is not None

    hub_box_mask = np.zeros((h, w), dtype=bool)
    hub_box_mask[int(hub["r0"]) : int(hub["r1"]) + 1, int(hub["c0"]) : int(hub["c1"]) + 1] = True

    out = np.full_like(g, int(bg))
    out[int(hub["r0"]) : int(hub["r1"]) + 1, int(hub["c0"]) : int(hub["c1"]) + 1] = g[
        int(hub["r0"]) : int(hub["r1"]) + 1,
        int(hub["c0"]) : int(hub["c1"]) + 1,
    ]

    side_order = sorted(
        [hub["sides"][side] for side in SIDE_ORDER],
        key=lambda item: (-int(item["length"]), SIDE_ORDER.index(str(item["side"]))),
    )
    selected = _select_sprites_by_length(_component_infos(g, int(bg), hub_box_mask))
    sprite_order = sorted(
        [selected[int(side["length"])] for side in side_order if int(side["length"]) in selected],
        key=lambda item: (-int(item["contact_len"]), -int(item["size"]), int(item["color"])),
    )

    for side_info, sprite in zip(side_order, sprite_order):
        target = int(side_info["length"])
        if int(sprite["contact_len"]) != int(target):
            continue

        side = str(side_info["side"])
        need_edge = _contact_side_for_hub_side(side)
        fixed = int(side_info["fixed"]) + (-1 if side == "top" else 1 if side == "bot" else 0)
        if side == "left":
            fixed = int(side_info["fixed"]) - 1
        if side == "right":
            fixed = int(side_info["fixed"]) + 1
        span = _side_span(side_info)

        candidates = []
        for krot in range(4):
            cand = _rotation_attachment(
                np.rot90(np.array(sprite["arr"], dtype=int), int(krot)),
                np.rot90(np.array(sprite["mask"], dtype=bool), int(krot)),
                need_edge=need_edge,
                fixed=int(fixed),
                span=span,
                target=int(target),
            )
            if cand is not None:
                candidates.append(cand)

        if len(candidates) != 1:
            continue

        arr, mask, dr, dc = candidates[0]
        placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(mask)}
        if any(not (0 <= rr < h and 0 <= cc < w) for rr, cc in placed):
            continue

        ok = True
        for rr, cc in placed:
            val = int(arr[int(rr - dr), int(cc - dc)])
            if val == int(bg):
                continue
            if int(out[int(rr), int(cc)]) != int(bg) and int(out[int(rr), int(cc)]) != val:
                ok = False
                break
        if not ok:
            continue

        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _make_canonical_mask(contact_len: int, rng: random.Random) -> np.ndarray | None:
    for _ in range(1200):
        h = rng.randint(int(contact_len) + 1, int(contact_len) + 3)
        w = rng.randint(3, 6)
        run_start = rng.randint(0, int(h - contact_len))
        target_pixels = rng.randint(int(contact_len) + 3, min(14, int(h * w)))

        mask = np.zeros((h, w), dtype=bool)
        mask[int(run_start) : int(run_start + contact_len), int(w - 1)] = True
        anchor_row = rng.choice(list(range(int(run_start), int(run_start + contact_len))))
        mask[int(anchor_row), int(w - 2)] = True

        while int(mask.sum()) < int(target_pixels):
            candidates = set()
            for rr, cc in np.argwhere(mask):
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = int(rr + dr)
                        nc = int(cc + dc)
                        if not (0 <= nr < h and 0 <= nc < w):
                            continue
                        if bool(mask[int(nr), int(nc)]):
                            continue
                        if int(nc) == int(w - 1) and not (int(run_start) <= int(nr) < int(run_start + contact_len)):
                            continue
                        candidates.add((int(nr), int(nc)))
            if not candidates:
                break

            cand_list = list(candidates)
            weights = []
            for nr, nc in cand_list:
                weight = 1 + 2 * (int(w - 1) - int(nc))
                if not (int(run_start) <= int(nr) < int(run_start + contact_len)):
                    weight += 2
                if int(nr) in {0, h - 1}:
                    weight += 1
                weights.append(float(weight))
            nr, nc = rng.choices(cand_list, weights=weights, k=1)[0]
            mask[int(nr), int(nc)] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 2:
            continue
        if int(mask.sum()) < int(contact_len) + 2 or int(mask.sum()) > 14:
            continue

        lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue

        runs = _all_edge_run_lengths(mask)
        max_run = int(max(runs.values()))
        if int(runs["right"]) != int(contact_len):
            continue
        if max_run != int(contact_len):
            continue
        if sum(int(v == max_run) for v in runs.values()) != 1:
            continue
        if _count_valid_rotations(mask, "right", int(contact_len)) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue

        return mask
    return None


def _make_sprite_mask(contact_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    need_edge = _contact_side_for_hub_side(side)
    for _ in range(300):
        canonical = _make_canonical_mask(int(contact_len), rng)
        if canonical is None:
            continue
        mask = _rotate_canonical_to_side(canonical, side)
        if int(_measure_contact_length(mask)) != int(contact_len):
            continue
        if _count_valid_rotations(mask, need_edge, int(contact_len)) != 1:
            continue
        if not _is_fully_asymmetric(mask):
            continue
        return mask
    return None


def _make_hub_layout(rng: random.Random):
    lengths = [2, 3, 4, 5]
    rng.shuffle(lengths)
    by_side = {side: int(length) for side, length in zip(SIDE_ORDER, lengths)}

    inner_h = rng.randint(5, 7)
    inner_w = rng.randint(5, 7)
    box_h = int(inner_h + 2)
    box_w = int(inner_w + 2)

    side_specs: dict[str, dict[str, int]] = {}
    for side in SIDE_ORDER:
        length = int(by_side[side])
        avail = int(box_w - 2) if side in ("top", "bot") else int(box_h - 2)
        start_idx = rng.randint(0, int(avail - length))
        end_idx = int(start_idx + length - 1)
        side_specs[side] = {
            "side": side,
            "length": int(length),
            "start_idx": int(start_idx),
            "end_idx": int(end_idx),
        }
    return int(box_h), int(box_w), side_specs


def _build_hub(
    h: int,
    w: int,
    hub_r: int,
    hub_c: int,
    box_h: int,
    box_w: int,
    side_specs: dict[str, dict[str, int]],
    side_colors: dict[str, int],
) -> tuple[np.ndarray, dict[str, dict[str, int]]]:
    grid = np.full((h, w), 8, dtype=int)
    placed: dict[str, dict[str, int]] = {}

    for side in SIDE_ORDER:
        spec = side_specs[side]
        color = int(side_colors[side])
        if side == "top":
            fixed = int(hub_r)
            start = int(hub_c + 1 + spec["start_idx"])
            end = int(hub_c + 1 + spec["end_idx"])
            grid[int(fixed), int(start) : int(end) + 1] = int(color)
        elif side == "bot":
            fixed = int(hub_r + box_h - 1)
            start = int(hub_c + 1 + spec["start_idx"])
            end = int(hub_c + 1 + spec["end_idx"])
            grid[int(fixed), int(start) : int(end) + 1] = int(color)
        elif side == "left":
            fixed = int(hub_c)
            start = int(hub_r + 1 + spec["start_idx"])
            end = int(hub_r + 1 + spec["end_idx"])
            grid[int(start) : int(end) + 1, int(fixed)] = int(color)
        else:
            fixed = int(hub_c + box_w - 1)
            start = int(hub_r + 1 + spec["start_idx"])
            end = int(hub_r + 1 + spec["end_idx"])
            grid[int(start) : int(end) + 1, int(fixed)] = int(color)

        placed[side] = {
            "side": side,
            "color": int(color),
            "fixed": int(fixed),
            "start": int(start),
            "end": int(end),
            "length": int(spec["length"]),
        }

    return grid, placed


def _attached_origin(mask: np.ndarray, side_info: dict[str, int]) -> tuple[int, int]:
    side = str(side_info["side"])
    need_edge = _contact_side_for_hub_side(side)
    seg = _edge_segment(mask, need_edge)
    assert seg is not None
    if need_edge in ("left", "right"):
        _typ, edge_c, rows = seg
        return int(side_info["start"] - rows[0]), int(
            (int(side_info["fixed"]) - 1 if side == "left" else int(side_info["fixed"]) + 1) - edge_c
        )
    _typ, edge_r, cols = seg
    return int((int(side_info["fixed"]) - 1 if side == "top" else int(side_info["fixed"]) + 1) - edge_r), int(
        side_info["start"] - cols[0]
    )


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    h = rng.randint(20, 26)
    w = rng.randint(20, 26)

    box_h, box_w, side_specs = _make_hub_layout(rng)
    side_lengths = [int(side_specs[side]["length"]) for side in SIDE_ORDER]

    sprite_masks: dict[str, np.ndarray] = {}
    for side in SIDE_ORDER:
        mask = _make_sprite_mask(int(side_specs[side]["length"]), side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 3

    hub_r_min = margin
    hub_r_max = h - box_h - margin
    hub_c_min = margin
    hub_c_max = w - box_w - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    palette = [col for col in range(10) if col != bg]
    rng.shuffle(palette)
    hub_colors = {side: int(col) for side, col in zip(SIDE_ORDER, palette[:4])}
    sprite_colors = {side: int(col) for side, col in zip(SIDE_ORDER, palette[4:8])}

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out, placed_sides = _build_hub(
        int(h),
        int(w),
        int(hub_r),
        int(hub_c),
        int(box_h),
        int(box_w),
        side_specs,
        hub_colors,
    )

    if len(set(side_lengths)) != 4:
        return None

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in SIDE_ORDER:
        mask = sprite_masks[side]
        color = int(sprite_colors[side])
        arr = np.where(mask, color, bg).astype(int)
        sr, sc = _attached_origin(mask, placed_sides[side])
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > h or sc + sw > w:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[int(rr), int(cc)]) and int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None
        paste(out, arr, int(sr), int(sc), transparent=bg)
        attached[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((h, w), bg, dtype=int)
    inp[int(hub_r) : int(hub_r + box_h), int(hub_c) : int(hub_c + box_w)] = out[
        int(hub_r) : int(hub_r + box_h),
        int(hub_c) : int(hub_c + box_w),
    ]
    canvas = inp.copy()
    canvas[int(hub_r) : int(hub_r + box_h), int(hub_c) : int(hub_c + box_w)] = -1

    for side in SIDE_ORDER:
        arr, mask, _sr, _sc = attached[side]
        rot_k = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(rot_k))
        rotated_mask = np.rot90(mask, int(rot_k))
        cropped_arr, cropped_mask = _crop_to_mask(rotated_arr, rotated_mask)
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
        paste(canvas, patch, int(pr), int(pc), transparent=bg)
        paste(inp, patch, int(pr), int(pc), transparent=bg)

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
