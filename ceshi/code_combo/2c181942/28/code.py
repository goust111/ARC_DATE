# Combo variant of 2c181942 (index 28)
# combo_spec: D9/1+D11/1
# Name: 边缘放置+间隔贴附

from __future__ import annotations

import random
from pathlib import Path
import sys

import numpy as np
from scipy.ndimage import label as _label

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import bbox_coords, get_bg, label_mask, paste


# Design analysis for the combo:
# - Original task: sprites are scattered arbitrarily, color determines which hub side
#   they belong to, and the output attaches them flush to the hub.
# - Combo change D9/1: each sprite now sits on one outer grid border in the input.
#   That border encodes the target side: top-border sprite goes above the hub, left-
#   border sprite goes left of the hub, etc. The color still stays consistent with the
#   matching hub edge, but direction must be readable from position.
# - Combo change D11/1: attached sprites no longer touch the hub. There must be a
#   single background pixel gap between the sprite contact edge and the hub side.
# - Concrete pixel example: if the hub top edge is at row r, then the matched top
#   sprite's contact row is no longer placed at r-1 (flush) but at r-2, leaving row
#   r-1 empty. Similarly, the left sprite's contact column is c-2 instead of c-1.
# - Complexity calibration: color remains useful as identity, but direction is now
#   repeated across samples through border placement, making the new behavior visible
#   to a human. Sprites are still rotated, so the solver must recover orientation; we
#   keep the original "contact edge length + mirror symmetry" constraint to avoid
#   rotational ambiguity.
# - Hidden-assumption audit of the original solver:
#   1. It assumed "largest component of a hub-edge color" was the right sprite.
#      That is no longer enough conceptually, because border position carries the
#      target side. The new solver therefore first filters by touched grid border and
#      only then rotates/attaches.
#   2. It assumed flush attachment coordinates. Those offsets are all shifted outward
#      by one cell here.
#   3. It assumed scattered placement far from borders/hub. The generator now instead
#      guarantees: one monochrome 8-connected component per chosen side, each touching
#      exactly one grid border and no corner, so side inference is unambiguous.
# - Placement legality: output is built first, with overlap/bounds checks for the
#   gapped attachment positions. Input is then derived by moving each sprite to its
#   corresponding outer grid edge while keeping a one-cell buffer from corners and
#   from other objects, preventing accidental multi-border touches or overlap.
# - Visual observability: in 2-4 examples, a human sees two stable signals:
#   sprites start on the frame edge corresponding to their final hub side, and every
#   output sprite floats one cell away from the hub instead of touching it.


SIDES = ("top", "bot", "left", "right")


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


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


def _touches_border(mask: np.ndarray, side: str) -> bool:
    if side == "top":
        return bool(mask[0, :].any())
    if side == "bot":
        return bool(mask[-1, :].any())
    if side == "left":
        return bool(mask[:, 0].any())
    return bool(mask[:, -1].any())


def _attachment_origin(mask: np.ndarray, hub_r: int, hub_c: int, n: int, side: str, gap: int = 1) -> tuple[int, int]:
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - gap - mask.shape[0]), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + n + gap), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - gap - mask.shape[1])
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + n + gap)


def _count_valid_rotations(mask: np.ndarray, need_edge: str, edge_len: int) -> int:
    count = 0
    for krot in range(4):
        msk = _trim_mask(np.rot90(mask, krot))
        if need_edge in ("left", "right"):
            if not np.array_equal(msk, np.flipud(msk)):
                continue
        else:
            if not np.array_equal(msk, np.fliplr(msk)):
                continue
        seg = _edge_segment(msk, need_edge)
        if seg is None:
            continue
        if need_edge in ("left", "right"):
            _typ, _edge_c, rows = seg
            if len(rows) == edge_len and _is_contig(rows):
                count += 1
        else:
            _typ, _edge_r, cols = seg
            if len(cols) == edge_len and _is_contig(cols):
                count += 1
    return count


def _rotate_canonical_left(mask: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return mask.copy()
    if side == "right":
        return np.fliplr(mask)
    if side == "top":
        return np.rot90(mask, -1)
    return np.rot90(mask, 1)


def _canonical_templates() -> list[np.ndarray]:
    templates = [
        np.array(
            [
                [0, 1, 0, 0],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [0, 1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 0, 0],
                [1, 1, 1],
                [1, 1, 1],
                [1, 0, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [0, 1, 1, 0],
                [1, 1, 0, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=bool,
        ),
        np.array(
            [
                [1, 1, 0, 0],
                [0, 1, 1, 1],
                [0, 1, 1, 1],
                [1, 1, 0, 0],
            ],
            dtype=bool,
        ),
    ]
    out = []
    for tmpl in templates:
        if np.array_equal(tmpl, np.flipud(tmpl)) and _count_valid_rotations(tmpl, "right", 2) == 1:
            labeled, num = _label(tmpl, structure=np.ones((3, 3), dtype=int))
            if num == 1:
                out.append(tmpl)
    return out


LEFT_TEMPLATES = _canonical_templates()


def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    usable = [tmpl for tmpl in LEFT_TEMPLATES if _count_valid_rotations(tmpl, "right", edge_len) == 1]
    if not usable:
        return None
    mask = rng.choice(usable)
    return _rotate_canonical_left(mask, side)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    def _find_hub():
        best = None
        for n in range(3, 7):
            for r in range(0, H - n + 1):
                for c in range(0, W - n + 1):
                    if g[r, c] != bg or g[r, c + n - 1] != bg or g[r + n - 1, c] != bg or g[r + n - 1, c + n - 1] != bg:
                        continue
                    inner = g[r + 1 : r + n - 1, c + 1 : c + n - 1]
                    if not (inner == bg).all():
                        continue
                    top = g[r, c + 1 : c + n - 1]
                    bot = g[r + n - 1, c + 1 : c + n - 1]
                    left = g[r + 1 : r + n - 1, c]
                    right = g[r + 1 : r + n - 1, c + n - 1]
                    if (top == bg).any() or (bot == bg).any() or (left == bg).any() or (right == bg).any():
                        continue
                    if len(set(map(int, top))) != 1 or len(set(map(int, bot))) != 1 or len(set(map(int, left))) != 1 or len(set(map(int, right))) != 1:
                        continue
                    cr = r + (n - 1) / 2
                    cc = c + (n - 1) / 2
                    score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
                    cand = (float(score), (r, c, n), (int(top[0]), int(bot[0]), int(left[0]), int(right[0])))
                    if best is None or cand[0] < best[0]:
                        best = cand
        return best

    hub = _find_hub()
    assert hub is not None
    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub

    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[r : r + n, c : c + n] = True

    out = np.full_like(g, bg)
    out[r : r + n, c : c + n] = g[r : r + n, c : c + n]
    edge_len = n - 2

    side_specs = {
        "top": {"color": int(top_col), "need_edge": "bot", "fixed": int(r - 2), "span": list(range(c + 1, c + n - 1))},
        "bot": {"color": int(bot_col), "need_edge": "top", "fixed": int(r + n + 1), "span": list(range(c + 1, c + n - 1))},
        "left": {"color": int(left_col), "need_edge": "right", "fixed": int(c - 2), "span": list(range(r + 1, r + n - 1))},
        "right": {"color": int(right_col), "need_edge": "left", "fixed": int(c + n + 1), "span": list(range(r + 1, r + n - 1))},
    }

    for side, spec in side_specs.items():
        col = spec["color"]
        m = (g == col) & (~hub_mask)
        if not bool(m.any()):
            continue
        labc, nc = label_mask(m, connectivity=8)
        candidates: list[tuple[int, np.ndarray]] = []
        for cid in range(1, int(nc) + 1):
            comp = labc == cid
            if not _touches_border(comp, side):
                continue
            if any(_touches_border(comp, other) for other in SIDES if other != side):
                continue
            candidates.append((int(comp.sum()), comp))
        if not candidates:
            continue
        candidates.sort(key=lambda x: x[0], reverse=True)
        spr_mask = candidates[0][1]
        spr_arr = g.copy()
        spr_arr[~spr_mask] = bg
        spr_arr, spr_mask = _crop_to_mask(spr_arr, spr_mask)

        found = None
        need_edge = str(spec["need_edge"])
        fixed = int(spec["fixed"])
        span = list(spec["span"])
        for krot in range(4):
            a = np.rot90(spr_arr, krot)
            msk = np.rot90(spr_mask, krot)
            a, msk = _crop_to_mask(a, msk)

            if need_edge in ("left", "right"):
                if not np.array_equal(msk, np.flipud(msk)):
                    continue
            else:
                if not np.array_equal(msk, np.fliplr(msk)):
                    continue

            seg = _edge_segment(msk, need_edge)
            if seg is None:
                continue
            if need_edge in ("left", "right"):
                _typ, edge_c, rows = seg
                if len(rows) != edge_len or not _is_contig(rows):
                    continue
                dr = int(span[0] - rows[0])
                dc = int(fixed - edge_c)
            else:
                _typ, edge_r, cols = seg
                if len(cols) != edge_len or not _is_contig(cols):
                    continue
                dr = int(fixed - edge_r)
                dc = int(span[0] - cols[0])

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue
            ok = True
            for rr, cc2 in placed:
                val = int(a[rr - dr, cc2 - dc])
                if val == bg:
                    continue
                if out[rr, cc2] != bg and out[rr, cc2] != val:
                    ok = False
                    break
            if not ok:
                continue
            found = (a, msk, dr, dc)
            break

        if found is None:
            continue
        a, msk, dr, dc = found
        paste(out, a, dr, dc, mask=msk, overwrite=True, clip=True)

    return out


def _place_patch_on_border(canvas: np.ndarray, patch: np.ndarray, side: str, bg: int, rng: random.Random) -> tuple[int, int] | None:
    H, W = canvas.shape
    ph, pw = patch.shape
    positions: list[tuple[int, int]] = []
    corner_pad = 2
    halo = 1

    if side == "top":
        if pw > W - 2 * corner_pad:
            return None
        sr = 0
        for sc in range(corner_pad, W - pw - corner_pad + 1):
            positions.append((sr, sc))
    elif side == "bot":
        if pw > W - 2 * corner_pad:
            return None
        sr = H - ph
        for sc in range(corner_pad, W - pw - corner_pad + 1):
            positions.append((sr, sc))
    elif side == "left":
        if ph > H - 2 * corner_pad:
            return None
        sc = 0
        for sr in range(corner_pad, H - ph - corner_pad + 1):
            positions.append((sr, sc))
    else:
        if ph > H - 2 * corner_pad:
            return None
        sc = W - pw
        for sr in range(corner_pad, H - ph - corner_pad + 1):
            positions.append((sr, sc))

    rng.shuffle(positions)
    occ = canvas != bg
    pmask = patch != bg
    for sr, sc in positions:
        if sr < 0 or sc < 0 or sr + ph > H or sc + pw > W:
            continue
        if np.any(occ[sr : sr + ph, sc : sc + pw] & pmask):
            continue
        rr0 = max(0, sr - halo)
        rr1 = min(H, sr + ph + halo)
        cc0 = max(0, sc - halo)
        cc1 = min(W, sc + pw + halo)
        region = occ[rr0:rr1, cc0:cc1].copy()
        region[sr - rr0 : sr - rr0 + ph, sc - cc0 : sc - cc0 + pw] &= ~pmask
        if region.any():
            continue
        return int(sr), int(sc)
    return None


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
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    k = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), k)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_sprite_h = max(mask.shape[0] for mask in sprite_masks.values())
    max_sprite_w = max(mask.shape[1] for mask in sprite_masks.values())
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

    attached: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        arr = np.where(mask, side_colors[side], bg).astype(int)
        sr, sc = _attachment_origin(mask, hub_r, hub_c, n, side, gap=1)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        if np.any((out[sr : sr + sh, sc : sc + sw] != bg) & mask):
            return None
        paste(out, arr, sr, sc, transparent=bg)
        attached[side] = (arr, mask)

    inp = np.full((H, W), bg, dtype=int)
    inp[hub_r : hub_r + n, hub_c : hub_c + n] = out[hub_r : hub_r + n, hub_c : hub_c + n]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask = attached[side]
        valid_rots = [rot for rot in (1, 2, 3)]
        rng.shuffle(valid_rots)
        placed = False
        for krot in valid_rots:
            rot_arr = np.rot90(arr, krot)
            rot_mask = np.rot90(mask, krot)
            rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
            patch = np.where(rot_mask, rot_arr, bg).astype(int)
            pos = _place_patch_on_border(canvas, patch, side, bg, rng)
            if pos is None:
                continue
            pr, pc = pos
            paste(canvas, patch, pr, pc, transparent=bg)
            paste(inp, patch, pr, pc, transparent=bg)
            placed = True
            break
        if not placed:
            return None

    result = main(inp)
    if not np.array_equal(result, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after 300 attempts")
