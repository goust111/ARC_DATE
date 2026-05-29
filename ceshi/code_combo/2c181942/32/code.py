# Combo variant of 2c181942 (index 32)
# combo_spec: D1/1+D2/1+D5/0
# Name: 位置匹配+排序+矩形hub

from __future__ import annotations

import random

import numpy as np
from scipy import ndimage as _ndimage
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste

# ---------------------------------------------------------------------------
# Design analysis for the combo variant
# ---------------------------------------------------------------------------
#
# a) Concrete behavior and trace
# - The hub is no longer the original 4x4 square. It is a hollow 4x6 rectangle:
#   top/bottom sides have contact length 4, left/right sides have contact length 2.
# - Detached sprites no longer match hub sides by color. All true detached sprites
#   share one non-hub source color, so color equality is intentionally useless.
# - Matching now has two visible stages:
#   1. Position gives the corridor: a component above / below / left / right of the
#      hub can only attach to that side.
#   2. Within that corridor, contact-edge length rank decides which candidate is the
#      true one for the rectangle tier: long edges choose the longer contact segment,
#      short edges choose the shorter one.
# - Small trace:
#   original: "blue detached sprite" is found by color, rotated to a 2-cell contact,
#   then pasted onto a square side.
#   combo: above the hub there may be two same-colored candidates. The solver first
#   says "both belong to the top corridor", then prefers the one whose valid top-side
#   contact run is longer, because the top hub edge is a long rectangle edge. The
#   chosen mask is recolored to the top-edge color and attached.
#
# b) Complexity calibration
# - A strict four-way length ordering is impossible on a true rectangle because
#   opposite sides have equal lengths. I therefore realize D2 as a two-tier ordering
#   over the rectangle's observable edge classes: long horizontal edges (length 4)
#   outrank short vertical edges (length 2).
# - To keep that ordering non-vacuous, the generator always includes all four true
#   sides and also injects at least:
#   * one horizontal-corridor decoy from the short class, and
#   * one vertical-corridor decoy from the long class.
#   So position alone is insufficient whenever a corridor contains two candidates.
#
# c) Implicit assumptions review against the original code
# - `_find_hub()` originally assumed an `n x n` square. That is invalid now; the
#   solver must search a fixed 4x6 hollow rectangle.
# - `target = n - 2` originally gave one contact length for all sides. That fails for
#   a rectangle and becomes side-specific (`4` for top/bot, `2` for left/right).
# - `m = (g == col) & (~hub_mask)` assumed color isolates the intended sprite. That is
#   false once all true sprites share one source color distinct from hub colors.
# - Taking the largest same-color component also breaks, because the generator may put
#   multiple same-colored candidates into the same side corridor.
# - The paste color can no longer come from the source sprite; attached sprites must
#   be recolored to the destination hub side color.
#
# d) Placement legality
# - The generator is output-first: it first builds the exact final output
#   (hub + attached true sprites), then scatters detached copies into side corridors.
# - Each detached patch is placed by an explicit corridor-aware search with padding,
#   so objects stay in-bounds, disconnected under 8-connectivity, and visually tied to
#   exactly one side corridor.
# - Decoys are never attached in the output, but they are placed in the same corridor
#   geometry as true sprites so the solver must reject them by the length-tier rule.
#
# e) Visual observability
# - Across samples, humans can see:
#   * the hub is rectangular, not square;
#   * all detached true sprites share one source color, so color matching is gone;
#   * top/bottom outputs use longer contact spans than left/right outputs;
#   * some corridors contain two same-colored candidates, and only the one from the
#     correct long/short tier is kept.
# - No invisible numeric property such as raw color id is used for ranking.
#
# f) Visual cleanliness
# - The hub is kept near the center to leave balanced side corridors.
# - Corridor placement requires overlap with the hub's inner projection, so "top" and
#   "left" candidates are visually segregated instead of floating in ambiguous corners.
# - Padding prevents accidental merges between neighboring detached objects.
#
# g) Change inferability
# - Only five semantic colors are needed in normal samples: four hub-edge colors plus
#   one shared detached-sprite color. This keeps recoloring easy to infer.
# - The long-vs-short length distinction is directly visible because the rectangle
#   itself exposes two side-length classes, and detached candidates are shaped so their
#   valid contact edges visibly belong to one class or the other.
#
# h) Quantitative check used after implementation
# - Every generated sample should have zero detached sprites whose input color equals a
#   hub-edge color.
# - Every generated sample should have at least one vertical corridor containing both a
#   short-class true sprite and a long-class decoy. A naive "always choose the longest
#   candidate in each corridor" solver should therefore disagree with the combo solver
#   on many samples.
#
# i) New solver assumptions and generator guarantees
# - Solver assumption: a relevant detached component lies entirely in one side corridor
#   and overlaps the hub's inner span on the orthogonal axis.
#   Generator guarantee: corridor search enforces exactly that.
# - Solver assumption: in each side corridor, at least one candidate has the correct
#   long/short contact tier for that side.
#   Generator guarantee: each side always gets one true sprite of the correct class.
# - Solver assumption: opposite-class decoys remain disconnected and stay in the same
#   corridor as the true sprite.
#   Generator guarantee: they are placed with padding inside the same region.

HUB_H = 4
HUB_W = 6
HORIZ_TARGET = HUB_W - 2
VERT_TARGET = HUB_H - 2
SIDES = ("top", "bot", "left", "right")
LONG_SIDES = {"top", "bot"}
SHORT_SIDES = {"left", "right"}

_LONG_TOP_BASES = [
    np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=bool,
    ),
]

_SHORT_TOP_BASES = [
    np.array(
        [
            [1, 0, 0, 1],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [1, 0, 0, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [1, 0, 0, 1],
            [1, 1, 1, 1],
            [0, 1, 1, 0],
        ],
        dtype=bool,
    ),
]


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


def _need_edge(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _target_len_for_side(side: str) -> int:
    return HORIZ_TARGET if side in LONG_SIDES else VERT_TARGET


def _target_len_for_kind(kind: str) -> int:
    return HORIZ_TARGET if kind == "long" else VERT_TARGET


def _attachment_sym_ok(mask: np.ndarray, need_edge: str) -> bool:
    if need_edge in ("left", "right"):
        return bool(np.array_equal(mask, np.flipud(mask)))
    return bool(np.array_equal(mask, np.fliplr(mask)))


def _contact_len(mask: np.ndarray, need_edge: str) -> int | None:
    if not _attachment_sym_ok(mask, need_edge):
        return None
    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None
    if need_edge in ("left", "right"):
        _typ, _edge_c, rows = seg
        return int(len(rows)) if _is_contig(rows) else None
    _typ, _edge_r, cols = seg
    return int(len(cols)) if _is_contig(cols) else None


def _count_valid_rotations(mask: np.ndarray, need_edge: str, target_len: int) -> int:
    count = 0
    arr = mask.astype(int)
    for krot in range(4):
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _contact_len(rot_mask, need_edge) == int(target_len):
            count += 1
    return count


def _orient_from_top(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return mask.copy()
    if side == "bot":
        return np.flipud(mask)
    if side == "left":
        return np.rot90(mask, 1)
    return np.rot90(mask, -1)


def _make_class_mask(kind: str, side: str, rng: random.Random) -> np.ndarray:
    if kind == "long":
        base = rng.choice(_LONG_TOP_BASES).copy()
    else:
        base = rng.choice(_SHORT_TOP_BASES).copy()
    mask = _orient_from_top(base, side)

    need_edge = _need_edge(side)
    assert _count_valid_rotations(mask, need_edge, _target_len_for_kind(kind)) == 1
    labeled, num = _label(mask, structure=np.ones((3, 3), dtype=int))
    assert int(num) == 1
    return mask


def _make_sprite_mask(side: str, rng: random.Random) -> np.ndarray:
    return _make_class_mask("long" if side in LONG_SIDES else "short", side, rng)


def _make_opposite_class_mask(side: str, rng: random.Random) -> np.ndarray:
    if side in LONG_SIDES:
        return _make_class_mask("short", side, rng)
    return _make_class_mask("long", side, rng)


def _build_hub(out: np.ndarray, hub_r: int, hub_c: int, side_colors: dict[str, int], bg: int) -> None:
    out[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W] = int(bg)
    out[hub_r, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["top"])
    out[hub_r + HUB_H - 1, hub_c + 1 : hub_c + HUB_W - 1] = int(side_colors["bot"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c] = int(side_colors["left"])
    out[hub_r + 1 : hub_r + HUB_H - 1, hub_c + HUB_W - 1] = int(side_colors["right"])


def _attachment_offset(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int]:
    if side == "top":
        _typ, edge_r, cols = _edge_segment(mask, "bot")
        return int(hub_r - edge_r - 1), int(hub_c + 1 - cols[0])
    if side == "bot":
        _typ, edge_r, cols = _edge_segment(mask, "top")
        return int(hub_r + HUB_H - edge_r), int(hub_c + 1 - cols[0])
    if side == "left":
        _typ, edge_c, rows = _edge_segment(mask, "right")
        return int(hub_r + 1 - rows[0]), int(hub_c - edge_c - 1)
    _typ, edge_c, rows = _edge_segment(mask, "left")
    return int(hub_r + 1 - rows[0]), int(hub_c + HUB_W - edge_c)


def _crop_patch(arr: np.ndarray, bg: int) -> np.ndarray:
    mask = arr != int(bg)
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _scatter_patch(mask: np.ndarray, color: int, bg: int, rng: random.Random) -> np.ndarray:
    arr = np.where(mask, int(color), int(bg)).astype(int)
    krot = rng.choice([1, 2, 3])
    return _crop_patch(np.rot90(arr, int(krot)), bg)


def _candidate_positions(
    canvas: np.ndarray,
    patch: np.ndarray,
    region: str,
    hub_r: int,
    hub_c: int,
    bg: int,
    padding: int = 1,
) -> list[tuple[int, int]]:
    H, W = canvas.shape
    ph, pw = patch.shape
    occ = canvas != int(bg)
    if int(padding) > 0:
        occ = _ndimage.binary_dilation(occ, structure=np.ones((3, 3), dtype=bool), iterations=int(padding))
    mask = patch != int(bg)

    inner_r0, inner_r1 = int(hub_r + 1), int(hub_r + HUB_H - 2)
    inner_c0, inner_c1 = int(hub_c + 1), int(hub_c + HUB_W - 2)
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
                if rr0 <= int(hub_r + HUB_H - 1):
                    continue
                if not _intervals_overlap(cc0, cc1, inner_c0, inner_c1):
                    continue
            elif region == "left":
                if cc1 >= int(hub_c):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            elif region == "right":
                if cc0 <= int(hub_c + HUB_W - 1):
                    continue
                if not _intervals_overlap(rr0, rr1, inner_r0, inner_r1):
                    continue
            else:
                raise ValueError(f"unknown region: {region}")
            candidates.append((int(pr), int(pc)))
    return candidates


def _classify_component_side(
    r0: int,
    c0: int,
    r1: int,
    c1: int,
    hub_r: int,
    hub_c: int,
) -> str | None:
    inner_r0, inner_r1 = int(hub_r + 1), int(hub_r + HUB_H - 2)
    inner_c0, inner_c1 = int(hub_c + 1), int(hub_c + HUB_W - 2)
    candidates: list[str] = []
    if int(r1) < int(hub_r) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
        candidates.append("top")
    if int(r0) > int(hub_r + HUB_H - 1) and _intervals_overlap(int(c0), int(c1), inner_c0, inner_c1):
        candidates.append("bot")
    if int(c1) < int(hub_c) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
        candidates.append("left")
    if int(c0) > int(hub_c + HUB_W - 1) and _intervals_overlap(int(r0), int(r1), inner_r0, inner_r1):
        candidates.append("right")
    return candidates[0] if len(candidates) == 1 else None


def _hub_side_specs(hub_r: int, hub_c: int, side_colors: dict[str, int]) -> dict[str, tuple[int, str, int, list[int], int]]:
    return {
        "top": (
            int(side_colors["top"]),
            "bot",
            int(hub_r - 1),
            list(range(int(hub_c + 1), int(hub_c + HUB_W - 1))),
            HORIZ_TARGET,
        ),
        "bot": (
            int(side_colors["bot"]),
            "top",
            int(hub_r + HUB_H),
            list(range(int(hub_c + 1), int(hub_c + HUB_W - 1))),
            HORIZ_TARGET,
        ),
        "left": (
            int(side_colors["left"]),
            "right",
            int(hub_c - 1),
            list(range(int(hub_r + 1), int(hub_r + HUB_H - 1))),
            VERT_TARGET,
        ),
        "right": (
            int(side_colors["right"]),
            "left",
            int(hub_c + HUB_W),
            list(range(int(hub_r + 1), int(hub_r + HUB_H - 1))),
            VERT_TARGET,
        ),
    }


def _best_candidate_for_side(
    spr_arr: np.ndarray,
    spr_mask: np.ndarray,
    side: str,
    fixed: int,
    span: list[int],
    bg: int,
    out: np.ndarray,
) -> dict[str, object] | None:
    H, W = out.shape
    need_edge = _need_edge(side)
    variants: list[dict[str, object]] = []

    for krot in range(4):
        a = np.rot90(spr_arr, int(krot))
        msk = np.rot90(spr_mask, int(krot))
        a, msk = _crop_to_mask(a, msk)

        match_len = _contact_len(msk, need_edge)
        if match_len is None:
            continue

        seg = _edge_segment(msk, need_edge)
        if seg is None:
            continue
        if need_edge in ("left", "right"):
            _typ, edge_c, rows = seg
            dr = int(span[0] - rows[0])
            dc = int(fixed - edge_c)
        else:
            _typ, edge_r, cols = seg
            dr = int(fixed - edge_r)
            dc = int(span[0] - cols[0])

        placed = [(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(msk)]
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in placed):
            continue
        if any(int(out[rr, cc]) != int(bg) for rr, cc in placed):
            continue

        variants.append(
            {
                "arr": a,
                "mask": msk,
                "dr": int(dr),
                "dc": int(dc),
                "match_len": int(match_len),
                "area": int(msk.sum()),
            }
        )

    if not variants:
        return None

    if side in LONG_SIDES:
        variants.sort(key=lambda v: (-int(v["match_len"]), -int(v["area"])))
    else:
        variants.sort(key=lambda v: (int(v["match_len"]), -int(v["area"])))
    return variants[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach detached sprites to a hollow 4x6 rectangle by side corridor and length tier.

    Variant rule:
    - Position relative to the hub decides the candidate side corridor.
    - Because the hub is rectangular, top/bottom are long edges (contact length 4)
      while left/right are short edges (contact length 2).
    - If a corridor contains multiple candidates, long sides choose the longer valid
      contact-edge candidate and short sides choose the shorter valid candidate.
    - Attached sprites are recolored to the destination hub-edge color.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None  # (score, (r,c), colors_dict)
        for r in range(0, H - HUB_H + 1):
            for c in range(0, W - HUB_W + 1):
                if (
                    int(g[r, c]) != int(bg)
                    or int(g[r, c + HUB_W - 1]) != int(bg)
                    or int(g[r + HUB_H - 1, c]) != int(bg)
                    or int(g[r + HUB_H - 1, c + HUB_W - 1]) != int(bg)
                ):
                    continue

                inner = g[r + 1 : r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                if not bool((inner == int(bg)).all()):
                    continue

                top = g[r, c + 1 : c + HUB_W - 1]
                bot = g[r + HUB_H - 1, c + 1 : c + HUB_W - 1]
                left = g[r + 1 : r + HUB_H - 1, c]
                right = g[r + 1 : r + HUB_H - 1, c + HUB_W - 1]
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

                side_colors = {
                    "top": int(top[0]),
                    "bot": int(bot[0]),
                    "left": int(left[0]),
                    "right": int(right[0]),
                }
                if len(set(side_colors.values())) != 4:
                    continue

                cr = float(r + (HUB_H - 1) / 2)
                cc = float(c + (HUB_W - 1) / 2)
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (float(score), (int(r), int(c)), side_colors)
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
        return best

    hub = _find_hub()
    assert hub is not None

    _score, (hub_r, hub_c), side_colors = hub
    side_specs = _hub_side_specs(hub_r, hub_c, side_colors)

    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W] = True

    out = np.full_like(g, int(bg))
    out[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W] = g[hub_r : hub_r + HUB_H, hub_c : hub_c + HUB_W]

    outside = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(outside, connectivity=8)
    by_side: dict[str, list[dict[str, object]]] = {side: [] for side in SIDES}

    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        pts = np.argwhere(comp_mask)
        if pts.size == 0:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        side = _classify_component_side(int(r0), int(c0), int(r1), int(c1), hub_r, hub_c)
        if side is None:
            continue

        comp_arr = g.copy()
        comp_arr[~comp_mask] = int(bg)
        comp_arr, comp_mask = _crop_to_mask(comp_arr, comp_mask)

        _paint_col, _need_edge_name, fixed, span, target = side_specs[side]
        cand = _best_candidate_for_side(comp_arr, comp_mask, side, int(fixed), span, bg, out)
        if cand is None:
            continue
        cand["target"] = int(target)
        by_side[side].append(cand)

    for side in SIDES:
        paint_col, _need_edge_name, _fixed, _span, target = side_specs[side]
        side_cands = [cand for cand in by_side[side] if int(cand["match_len"]) in (HORIZ_TARGET, VERT_TARGET)]
        if not side_cands:
            continue

        if side in LONG_SIDES:
            side_cands.sort(key=lambda cand: (-int(cand["match_len"]), -int(cand["area"])))
        else:
            side_cands.sort(key=lambda cand: (int(cand["match_len"]), -int(cand["area"])))
        chosen = side_cands[0]
        if int(chosen["match_len"]) != int(target):
            continue

        mask = chosen["mask"]
        dr = int(chosen["dr"])
        dc = int(chosen["dc"])
        if any(int(out[int(rr + dr), int(cc + dc)]) != int(bg) for rr, cc in np.argwhere(mask)):
            continue
        paint = np.where(mask, int(paint_col), int(bg)).astype(int)
        paste(out, paint, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8

    H = rng.randint(20, 26)
    W = rng.randint(22, 28)

    all_colors = [c for c in range(10) if c != bg]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = all_colors[:4]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    remaining = [c for c in range(10) if c != bg and c not in side_colors.values()]
    if not remaining:
        return None
    src_col = int(rng.choice(remaining))

    true_masks = {side: _make_sprite_mask(side, rng) for side in SIDES}

    # Keep the hub near the center to leave balanced space for multi-candidate corridors.
    hub_r = int((H - HUB_H) // 2 + rng.choice([-1, 0, 1]))
    hub_c = int((W - HUB_W) // 2 + rng.choice([-1, 0, 1]))
    hub_r = max(4, min(int(hub_r), int(H - HUB_H - 4)))
    hub_c = max(5, min(int(hub_c), int(W - HUB_W - 5)))

    out = np.full((H, W), int(bg), dtype=int)
    _build_hub(out, hub_r, hub_c, side_colors, bg)

    for side in SIDES:
        mask = true_masks[side]
        arr = np.where(mask, int(side_colors[side]), int(bg)).astype(int)
        sr, sc = _attachment_offset(mask, side, hub_r, hub_c)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        if bool((out[sr : sr + sh, sc : sc + sw] != int(bg))[mask].any()):
            return None
        paste(out, arr, int(sr), int(sc), transparent=int(bg))

    inp = np.full((H, W), int(bg), dtype=int)
    _build_hub(inp, hub_r, hub_c, side_colors, bg)
    canvas = inp.copy()

    horizontal_decoy_side = rng.choice(("top", "bot"))
    vertical_decoy_side = rng.choice(("left", "right"))
    decoy_masks = {
        horizontal_decoy_side: _make_opposite_class_mask(horizontal_decoy_side, rng),
        vertical_decoy_side: _make_opposite_class_mask(vertical_decoy_side, rng),
    }

    for side in SIDES:
        patches: list[np.ndarray] = []
        true_patch = _scatter_patch(true_masks[side], src_col, bg, rng)
        patches.append(true_patch)
        if side in decoy_masks:
            decoy_patch = _scatter_patch(decoy_masks[side], src_col, bg, rng)
            # Place the larger patch first to reduce corridor-packing failures.
            if int((decoy_patch != int(bg)).sum()) >= int((true_patch != int(bg)).sum()):
                patches = [decoy_patch, true_patch]
            else:
                patches.append(decoy_patch)

        for patch in patches:
            candidates = _candidate_positions(canvas, patch, side, hub_r, hub_c, bg, padding=1)
            if not candidates:
                return None
            pr, pc = rng.choice(candidates)
            paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
            paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    # Optional corner distractor: different color and outside all side corridors.
    if rng.random() < 0.3:
        dcols = [c for c in remaining if int(c) != int(src_col)]
        if dcols:
            dmask = _make_sprite_mask(rng.choice(SIDES), rng)
            dpatch = _scatter_patch(dmask, int(rng.choice(dcols)), bg, rng)
            # Try a few unconstrained corner-ish placements by simple rejection.
            dh, dw = dpatch.shape
            dmask_bool = dpatch != int(bg)
            for _ in range(25):
                pr = rng.randint(0, H - dh)
                pc = rng.randint(0, W - dw)
                rr0, rr1 = int(pr), int(pr + dh - 1)
                cc0, cc1 = int(pc), int(pc + dw - 1)
                if _classify_component_side(rr0, cc0, rr1, cc1, hub_r, hub_c) is not None:
                    continue
                if bool(((canvas[pr : pr + dh, pc : pc + dw] != int(bg)) & dmask_bool).any()):
                    continue
                paste(canvas, dpatch, int(pr), int(pc), transparent=int(bg))
                paste(inp, dpatch, int(pr), int(pc), transparent=int(bg))
                break

    try:
        result = main(inp)
    except Exception:
        return None
    if not np.array_equal(result, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
