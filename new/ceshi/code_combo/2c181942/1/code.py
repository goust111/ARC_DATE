# Combo variant of 2c181942 (index 1)
# combo_spec: D1/1+D2/1
# Name: 位置排序双重匹配

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage as _ndimage

from arc_utils import bbox_coords, get_bg, label_mask, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - The original task used a hollow 4x4 square hub; detached sprites were matched to
#   hub sides by color equality, then rotated so their contact edge length exactly
#   equaled the hub side length.
# - This combo changes two things together:
#   1. Matching is no longer color-based. Detached sprites are first grouped by
#      hub-relative position: above/below/left/right corridors.
#   2. The hub is now a non-square hollow rectangle, so horizontal and vertical sides
#      have different contact lengths. Within a corridor, the correct sprite is the
#      one whose contact-edge length falls into the same long-vs-short rank class as
#      that side.
# - Small trace:
#   original: a blue top edge searches for blue cells outside the hub, finds one blue
#   sprite, rotates it, and pastes it on top.
#   combo: the top corridor may contain two same-colored sprites. Position says both
#   are "top candidates", but only the one whose contact edge belongs to the longer
#   length class matches the top side of the rectangular hub. That sprite is rotated,
#   then recolored to the top-edge color.
# - Difference:
#   original matching path = color -> side.
#   combo matching path = side corridor -> length-rank filter -> side color repaint.
#
# b) Complexity calibration
# - I keep the original rotation/symmetry/contact-edge reasoning, so the task is still
#   about geometric attachment rather than becoming a trivial "move the nearest sprite".
# - All true detached sprites intentionally share one source color that does not appear
#   on the hub; color therefore cannot solve the puzzle.
# - To make the D2 signal actually matter, 1-3 side corridors also contain a decoy
#   sprite of the wrong contact-length class. Position alone only narrows the search;
#   long-vs-short ordering selects the correct candidate.
# - The hub remains visually simple: one central hollow rectangle plus side corridors.
#
# c) Original assumptions that break, line by line
# - `m = (g == col) & (~hub_mask)` no longer works because detached sprites do not use
#   hub edge colors. We must detect every outside component first, then classify it by
#   position relative to the hub.
# - `labc = label_mask(m, connectivity=8)` relied on the same color prefilter. In the
#   combo, multiple side-corridor candidates can share one color, so component labeling
#   must happen on all non-background outside cells.
# - `target = n - 2` assumed a square hub with one universal contact length. Here the
#   target depends on side: top/bottom use `hub_w - 2`, left/right use `hub_h - 2`.
# - `paste(out, a, ...)` assumed source and destination colors match. In the combo the
#   placed sprite must be recolored to the destination edge color after geometry is
#   decided.
# - A new assumption appears in the corridor selection loop: if a component overlaps
#   exactly one side corridor, it belongs to that side's candidate pool. The generator
#   guarantees this by keeping true sprites and decoys fully inside unambiguous side
#   corridors, with extra padding so components do not merge.
#
# d) Placement legality
# - Generator uses output-first construction: create the final attached scene, then
#   detach and scatter sprites back into side corridors.
# - Every placement checks bounds and collision against a dilated occupancy mask, so
#   hub, true sprites, and same-corridor decoys never overlap or touch.
# - Because the hub is non-square, the long and short target lengths are always
#   distinct; wrong-length decoys therefore cannot accidentally satisfy the side they
#   shadow.
#
# e) Visual observability
# - A human can see the combo rule from a few examples:
#   the sprite above stays associated with the top region, but when there are two
#   above-hub candidates only the one with the longer contact edge gets attached to the
#   long horizontal side.
# - The recoloring rule is also visible: all detached sprites share one source color,
#   while attached output sprites adopt the hub-side colors.
#
# f) Visual cleanliness
# - The hub stays near center with a margin based on the largest sprite extent.
# - Corridors keep candidates separated by side, and padding keeps components distinct.
# - No textual markers or hidden encodings are introduced.
#
# g) Change inferability
# - Relevant colors remain few: four hub-edge colors plus one detached-sprite color.
# - Length ranking is a human-visible property: long contact edge versus short contact
#   edge, not a hidden comparison on numeric color ids.
#
# h) Quantitative checks used after implementation
# - All true detached sprites use a non-hub source color, so color-match count should
#   stay 0 in generated samples.
# - Generated samples should contain at least one side corridor with a wrong-length
#   decoy in most cases, so a position-only ablation differs from the combo solver.
#
# i) New solver assumptions and generator guarantees
# - Solver assumption: the hub is a hollow rectangle with distinct height/width and
#   four solid monochrome edges. Generator guarantee: always true.
# - Solver assumption: each outside component lies in at most one side corridor.
#   Generator guarantee: corridor-specific placement helper enforces this.
# - Solver assumption: each true sprite has exactly one valid rotation for its intended
#   side. Generator guarantee: sprite templates are constructed in a canonical frame
#   that yields one valid rotation per side.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach detached sprites to a central hollow rectangle by side corridor + length rank.

    Combo rule:
    - Position gives the candidate side corridor: above/below/left/right the hub.
    - The hub is a non-square hollow rectangle, so horizontal and vertical sides have
      different contact lengths. Within a corridor, keep only candidates whose contact
      length belongs to the same descending length class as that side.
    - Rotate the chosen sprite into attachment orientation and repaint it with the
      destination side color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r, c, h, w), (top, bot, left, right))
        for h in range(4, 8):
            for w in range(4, 8):
                if h == w:
                    continue
                for r in range(0, H - h + 1):
                    for c in range(0, W - w + 1):
                        if (
                            int(g[r, c]) != bg
                            or int(g[r, c + w - 1]) != bg
                            or int(g[r + h - 1, c]) != bg
                            or int(g[r + h - 1, c + w - 1]) != bg
                        ):
                            continue
                        inner = g[r + 1 : r + h - 1, c + 1 : c + w - 1]
                        if not bool((inner == bg).all()):
                            continue

                        top = g[r, c + 1 : c + w - 1]
                        bot = g[r + h - 1, c + 1 : c + w - 1]
                        left = g[r + 1 : r + h - 1, c]
                        right = g[r + 1 : r + h - 1, c + w - 1]
                        if (
                            bool((top == bg).any())
                            or bool((bot == bg).any())
                            or bool((left == bg).any())
                            or bool((right == bg).any())
                        ):
                            continue
                        if (
                            len(set(map(int, top))) != 1
                            or len(set(map(int, bot))) != 1
                            or len(set(map(int, left))) != 1
                            or len(set(map(int, right))) != 1
                        ):
                            continue

                        colors = [int(top[0]), int(bot[0]), int(left[0]), int(right[0])]
                        if len(set(colors)) != 4:
                            continue

                        cr = float(r + (h - 1) / 2)
                        cc = float(c + (w - 1) / 2)
                        score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
                        cand = (
                            float(score),
                            (int(r), int(c), int(h), int(w)),
                            tuple(colors),
                        )
                        if best is None or float(cand[0]) < float(best[0]):
                            best = cand
        return best

    def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        pts = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(pts)
        return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]

    def _edge_segment(mask: np.ndarray, side: str):
        pts = np.argwhere(mask)
        if pts.size == 0:
            return None
        if side == "top":
            rr = int(pts[:, 0].min())
            cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
            return ("h", rr, cols)
        if side == "bot":
            rr = int(pts[:, 0].max())
            cols = sorted(int(x) for x in pts[pts[:, 0] == rr][:, 1])
            return ("h", rr, cols)
        if side == "left":
            cc = int(pts[:, 1].min())
            rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
            return ("v", cc, rows)
        if side == "right":
            cc = int(pts[:, 1].max())
            rows = sorted(int(x) for x in pts[pts[:, 1] == cc][:, 0])
            return ("v", cc, rows)
        return None

    def _is_contig(seq: list[int]) -> bool:
        return bool(seq) and all(int(seq[i]) + 1 == int(seq[i + 1]) for i in range(len(seq) - 1))

    def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
        return max(int(a0), int(b0)) <= min(int(a1), int(b1))

    hub = _find_hub()
    assert hub is not None

    _score, (r, c, hub_h, hub_w), (top_col, bot_col, left_col, right_col) = hub
    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + hub_h, c : c + hub_w] = True

    out = np.full_like(g, bg)
    out[r : r + hub_h, c : c + hub_w] = g[r : r + hub_h, c : c + hub_w]

    target_len = {
        "top": int(hub_w - 2),
        "bot": int(hub_w - 2),
        "left": int(hub_h - 2),
        "right": int(hub_h - 2),
    }
    length_order = sorted({int(hub_w - 2), int(hub_h - 2)}, reverse=True)
    desired_rank = {side: int(length_order.index(int(length))) for side, length in target_len.items()}

    side_specs = {
        "top": {
            "paint": int(top_col),
            "need_edge": "bot",
            "fixed": int(r - 1),
            "span": list(range(int(c + 1), int(c + hub_w - 1))),
        },
        "bot": {
            "paint": int(bot_col),
            "need_edge": "top",
            "fixed": int(r + hub_h),
            "span": list(range(int(c + 1), int(c + hub_w - 1))),
        },
        "left": {
            "paint": int(left_col),
            "need_edge": "right",
            "fixed": int(c - 1),
            "span": list(range(int(r + 1), int(r + hub_h - 1))),
        },
        "right": {
            "paint": int(right_col),
            "need_edge": "left",
            "fixed": int(c + hub_w),
            "span": list(range(int(r + 1), int(r + hub_h - 1))),
        },
    }

    inner_r0, inner_r1 = int(r + 1), int(r + hub_h - 2)
    inner_c0, inner_c1 = int(c + 1), int(c + hub_w - 2)

    outside = (g != bg) & (~hub_mask)
    lab, nc = label_mask(outside, connectivity=8)
    candidates_by_side: dict[str, list[dict[str, object]]] = {side: [] for side in side_specs}

    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        pts = np.argwhere(comp_mask)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)

        side_hits: list[str] = []
        if int(r1) < int(r) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
            side_hits.append("top")
        if int(r0) > int(r + hub_h - 1) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
            side_hits.append("bot")
        if int(c1) < int(c) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
            side_hits.append("left")
        if int(c0) > int(c + hub_w - 1) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
            side_hits.append("right")
        if len(side_hits) != 1:
            continue

        side = side_hits[0]
        comp_arr = g.copy()
        comp_arr[~comp_mask] = bg
        comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)

        side_spec = side_specs[side]
        rotations: list[dict[str, object]] = []
        for krot in range(4):
            rot_arr = np.rot90(comp_arr, int(krot))
            rot_mask = np.rot90(comp_mask, int(krot))
            rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)

            need_edge = str(side_spec["need_edge"])
            if need_edge in ("left", "right"):
                if not bool(np.array_equal(rot_mask, np.flipud(rot_mask))):
                    continue
            else:
                if not bool(np.array_equal(rot_mask, np.fliplr(rot_mask))):
                    continue

            seg = _edge_segment(rot_mask, need_edge)
            if seg is None:
                continue

            if need_edge in ("left", "right"):
                _typ, edge_c, rows = seg
                if not _is_contig(rows):
                    continue
                contact_len = int(len(rows))
                dr = int(side_spec["span"][0] - rows[0])
                dc = int(side_spec["fixed"] - edge_c)
            else:
                _typ, edge_r, cols = seg
                if not _is_contig(cols):
                    continue
                contact_len = int(len(cols))
                dr = int(side_spec["fixed"] - edge_r)
                dc = int(side_spec["span"][0] - cols[0])

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(rot_mask)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue

            rotations.append(
                {
                    "contact_len": int(contact_len),
                    "rank": int(length_order.index(contact_len)) if int(contact_len) in length_order else None,
                    "arr": rot_arr,
                    "mask": rot_mask,
                    "dr": int(dr),
                    "dc": int(dc),
                }
            )

        if not rotations:
            continue

        candidates_by_side[side].append(
            {
                "area": int(comp_mask.sum()),
                "rotations": rotations,
            }
        )

    for side, spec in side_specs.items():
        eligible: list[tuple[int, int, dict[str, object]]] = []
        for cand in candidates_by_side[side]:
            for rot in cand["rotations"]:
                if rot["rank"] is None:
                    continue
                if int(rot["rank"]) != int(desired_rank[side]):
                    continue
                if int(rot["contact_len"]) != int(target_len[side]):
                    continue
                eligible.append((int(cand["area"]), int(rot["contact_len"]), rot))
        if not eligible:
            continue

        _area, _length, chosen = max(eligible, key=lambda item: (int(item[1]), int(item[0])))
        rot_mask = chosen["mask"]
        paint = np.where(rot_mask, int(spec["paint"]), int(bg)).astype(int)
        paste(out, paint, int(chosen["dr"]), int(chosen["dc"]), mask=rot_mask, overwrite=True, clip=True)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _left_attach_mask(edge_len: int, rng: random.Random) -> np.ndarray:
    """
    Canonical sprite frame for left-side attachment:
    - flipud symmetric
    - contact edge is the rightmost column with exactly `edge_len` contiguous cells
    - not fliplr symmetric, so each side has exactly one valid rotation
    """
    h = int(edge_len + 2)
    if int(edge_len) >= 3 and rng.random() < 0.5:
        grid = np.zeros((h, 5), dtype=bool)
        grid[:, 3] = True
        grid[1 : 1 + edge_len, 4] = True
        grid[1 : 1 + edge_len, 2] = True
        if h >= 6:
            grid[2 : h - 2, 1] = True
        else:
            grid[h // 2, 1] = True
        grid[0, 0] = True
        grid[h - 1, 0] = True
        return grid

    grid = np.zeros((h, 4), dtype=bool)
    grid[:, 2] = True
    grid[1 : 1 + edge_len, 3] = True
    grid[1 : 1 + edge_len, 1] = True
    grid[0, 0] = True
    grid[h - 1, 0] = True
    return grid


def _orient_mask(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    if side == "bot":
        return np.rot90(mask, 1)
    raise ValueError(f"unknown side: {side}")


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray:
    return _orient_mask(_left_attach_mask(int(edge_len), rng), side)


def _crop_patch(arr: np.ndarray, bg: int) -> np.ndarray:
    mask = arr != int(bg)
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _intervals_overlap(a0: int, a1: int, b0: int, b1: int) -> bool:
    return max(int(a0), int(b0)) <= min(int(a1), int(b1))


def _candidate_positions(
    canvas: np.ndarray,
    patch: np.ndarray,
    region: str,
    hub_r: int,
    hub_c: int,
    hub_h: int,
    hub_w: int,
    bg: int,
    padding: int = 2,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    ph, pw = patch.shape
    occ = canvas != int(bg)
    if int(padding) > 0:
        occ = _ndimage.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=int(padding))
    mask = patch != int(bg)

    inner_r0, inner_r1 = int(hub_r + 1), int(hub_r + hub_h - 2)
    inner_c0, inner_c1 = int(hub_c + 1), int(hub_c + hub_w - 2)
    candidates: list[tuple[int, int]] = []

    for pr in range(0, H - ph + 1):
        for pc in range(0, W - pw + 1):
            if bool((occ[pr : pr + ph, pc : pc + pw] & mask).any()):
                continue

            rr0, rr1 = int(pr), int(pr + ph - 1)
            cc0, cc1 = int(pc), int(pc + pw - 1)
            if region == "top":
                if rr1 >= int(hub_r):
                    continue
                if not _intervals_overlap(cc0, cc1, inner_c0, inner_c1):
                    continue
            elif region == "bot":
                if rr0 <= int(hub_r + hub_h - 1):
                    continue
                if not _intervals_overlap(cc0, cc1, inner_c0, inner_c1):
                    continue
            elif region == "left":
                if cc1 >= int(hub_c):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            elif region == "right":
                if cc0 <= int(hub_c + hub_w - 1):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            else:
                raise ValueError(f"unknown region: {region}")

            candidates.append((int(pr), int(pc)))
    return candidates


def _attach_coords(side: str, mask: np.ndarray, hub_r: int, hub_c: int, hub_h: int, hub_w: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + hub_h), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + hub_w)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8

    H = rng.randint(22, 28)
    W = rng.randint(22, 28)

    length_choices = [2, 3, 4, 5]
    horiz_len, vert_len = rng.sample(length_choices, 2)
    hub_h = int(vert_len + 2)
    hub_w = int(horiz_len + 2)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = map(int, all_colors[:4])
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    remaining = [c for c in range(10) if c != bg and c not in side_colors.values()]
    if not remaining:
        return None
    src_col = int(rng.choice(remaining))

    side_lengths = {
        "top": int(horiz_len),
        "bot": int(horiz_len),
        "left": int(vert_len),
        "right": int(vert_len),
    }
    opposite_length = {
        "top": int(vert_len),
        "bot": int(vert_len),
        "left": int(horiz_len),
        "right": int(horiz_len),
    }
    if len({int(horiz_len), int(vert_len)}) != 2:
        return None

    sides = ["top", "bot", "left", "right"]
    true_masks = {side: _make_sprite_mask(int(side_lengths[side]), side, rng) for side in sides}

    max_extent = max(max(mask.shape) for mask in true_masks.values())
    margin = int(max_extent + 2)
    hub_r_min = int(margin)
    hub_r_max = int(H - hub_h - margin)
    hub_c_min = int(margin)
    hub_c_max = int(W - hub_w - margin)
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, hub_w - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + hub_h - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, hub_h - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + hub_w - 1] = int(right_col)

    for side in sides:
        mask = true_masks[side]
        sr, sc = _attach_coords(side, mask, hub_r, hub_c, hub_h, hub_w)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        if bool((out[sr : sr + sh, sc : sc + sw] != int(bg))[mask].any()):
            return None
        patch = np.where(mask, int(side_colors[side]), int(bg)).astype(int)
        paste(out, patch, int(sr), int(sc), transparent=int(bg))

    inp = np.full((H, W), int(bg), dtype=int)
    inp[hub_r : hub_r + hub_h, hub_c : hub_c + hub_w] = out[hub_r : hub_r + hub_h, hub_c : hub_c + hub_w]
    canvas = inp.copy()

    for side in sides:
        patch = np.where(true_masks[side], int(src_col), int(bg)).astype(int)
        k = rng.choice([1, 2, 3])
        patch = _crop_patch(np.rot90(patch, int(k)), int(bg))
        candidates = _candidate_positions(canvas, patch, side, hub_r, hub_c, hub_h, hub_w, int(bg), padding=1)
        if not candidates:
            return None
        pr, pc = rng.choice(candidates)
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    decoy_count = rng.randint(1, 2)
    decoy_sides = rng.sample(sides, decoy_count)
    for side in decoy_sides:
        mask = _make_sprite_mask(int(opposite_length[side]), side, rng)
        patch = np.where(mask, int(src_col), int(bg)).astype(int)
        patch = _crop_patch(np.rot90(patch, rng.choice([1, 2, 3])), int(bg))
        candidates = _candidate_positions(canvas, patch, side, hub_r, hub_c, hub_h, hub_w, int(bg), padding=1)
        if not candidates:
            return None
        pr, pc = rng.choice(candidates)
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    result = main(inp)
    if not np.array_equal(result, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
