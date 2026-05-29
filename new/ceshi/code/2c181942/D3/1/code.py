# Variant of 2c181942
# Dimension: D3 — 变换操作：旋转规则 (transform)
# Change: 尝试rot90 x 0-3，选择满足对称性和接触边约束的唯一旋转角度，将精灵对齐到hub对应边... → 条件旋转：由标记像素决定旋转方向
#
# Design note:
# - The hub, color-to-side pairing, and flush attachment geometry stay in the
#   same task family as the source task.
# - The changed part is the rotation decision: each sprite carries one marker
#   pixel in a corner of its bbox. After the solver identifies which hub side
#   the sprite belongs to by body color, it reads that corner and applies a
#   side-relative CW/CCW rule instead of trying all four rotations.
# - The generator uses a small validated sprite vocabulary so every sampled
#   input still exhibits the D3 change while keeping the success rate high.

from __future__ import annotations

import random
import numpy as np
from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


BG = 8
HUB_N = 4
EDGE_LEN = HUB_N - 2
SIDES = ("top", "bot", "left", "right")


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    mask = arr != int(bg)
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


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
                score = abs(float(cr - (H - 1) / 2)) + abs(float(cc - (W - 1) / 2))
                cand = (
                    float(score),
                    (int(r), int(c), int(n)),
                    (int(top[0]), int(bot[0]), int(left[0]), int(right[0])),
                )
                if best is None or float(cand[0]) < float(best[0]):
                    best = cand
    return best


def _attachment_delta(core_mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int] | None:
    target = int(n - 2)
    segs = {
        "top": ("bot", int(hub_r - 1), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "bot": ("top", int(hub_r + n), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "left": ("right", int(hub_c - 1), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
        "right": ("left", int(hub_c + n), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
    }
    need_edge, fixed, span = segs[side]
    seg = _edge_segment(core_mask, need_edge)
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


def _marker_corner(arr: np.ndarray, body_color: int, bg: int) -> str | None:
    marker_pts = np.argwhere((arr != int(bg)) & (arr != int(body_color)))
    if marker_pts.shape[0] != 1:
        return None
    mr, mc = map(int, marker_pts[0])
    h, w = map(int, arr.shape)
    if mr == 0 and mc == 0:
        return "tl"
    if mr == 0 and mc == w - 1:
        return "tr"
    if mr == h - 1 and mc == 0:
        return "bl"
    if mr == h - 1 and mc == w - 1:
        return "br"
    return None


def _turn_from_corner(side: str, corner: str | None) -> int | None:
    if corner is None:
        return None
    if side == "top":
        return -1 if corner in {"tl", "bl"} else 1
    if side == "bot":
        return -1 if corner in {"tr", "br"} else 1
    if side == "left":
        return -1 if corner in {"bl", "br"} else 1
    return -1 if corner in {"tl", "tr"} else 1


def main(grid: np.ndarray) -> np.ndarray:
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

    comps, nc = label_mask((g != int(bg)) & (~hub_mask), connectivity=8)
    components: list[np.ndarray] = []
    for cid in range(1, int(nc) + 1):
        cmask = comps == int(cid)
        arr = np.full_like(g, int(bg))
        arr[cmask] = g[cmask]
        components.append(_crop_non_bg(arr, bg))

    side_colors = {
        "top": int(top_col),
        "bot": int(bot_col),
        "left": int(left_col),
        "right": int(right_col),
    }

    for side, col in side_colors.items():
        best_arr = None
        best_count = -1
        for arr in components:
            count = int((arr == int(col)).sum())
            if count > best_count:
                best_count = count
                best_arr = arr
        if best_arr is None or best_count <= 0:
            continue

        krot = _turn_from_corner(side, _marker_corner(best_arr, col, bg))
        if krot is None:
            continue

        rotated = _crop_non_bg(np.rot90(best_arr, int(krot)), bg)
        core_mask = rotated == int(col)
        full_mask = rotated != int(bg)
        delta = _attachment_delta(core_mask, side, int(r), int(c), int(n))
        if delta is None:
            continue
        dr, dc = map(int, delta)

        placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(full_mask)}
        if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
            continue
        if any(bool(hub_mask[int(R), int(C)]) for R, C in placed):
            continue

        ok = True
        for R, C in placed:
            val = int(rotated[int(R - dr), int(C - dc)])
            if val == int(bg):
                continue
            if int(out[int(R), int(C)]) != int(bg) and int(out[int(R), int(C)]) != val:
                ok = False
                break
        if not ok:
            continue

        paste(out, rotated, int(dr), int(dc), transparent=bg, overwrite=True, clip=True)

    return out


def _make_body_mask(width: int, side: str) -> np.ndarray:
    base = np.ones((EDGE_LEN, int(width)), dtype=bool)
    if side == "left":
        return base.copy()
    if side == "right":
        return np.fliplr(base)
    if side == "top":
        return np.rot90(base, -1)
    return np.rot90(base, 1)


def _embed_with_marker(body_mask: np.ndarray, anchor: str, body_color: int, marker_color: int, bg: int) -> np.ndarray:
    h, w = map(int, body_mask.shape)
    arr = np.full((h + 1, w + 1), int(bg), dtype=int)
    patch = np.where(body_mask, int(body_color), int(bg))
    if anchor == "tl":
        arr[1:, 1:] = patch
        arr[0, 0] = int(marker_color)
    elif anchor == "tr":
        arr[1:, :w] = patch
        arr[0, w] = int(marker_color)
    elif anchor == "bl":
        arr[:h, 1:] = patch
        arr[h, 0] = int(marker_color)
    else:
        arr[:h, :w] = patch
        arr[h, w] = int(marker_color)
    return _crop_non_bg(arr, bg)


def _corner_matches_turn(side: str, corner: str | None, turn_dir: str) -> bool:
    if corner is None:
        return False
    if side == "top":
        return corner in (("tl", "bl") if turn_dir == "cw" else ("tr", "br"))
    if side == "bot":
        return corner in (("tr", "br") if turn_dir == "cw" else ("tl", "bl"))
    if side == "left":
        return corner in (("bl", "br") if turn_dir == "cw" else ("tl", "tr"))
    return corner in (("tl", "tr") if turn_dir == "cw" else ("bl", "br"))


def _build_sprite_candidates(side: str, width: int, turn_dir: str, body_color: int, marker_color: int, bg: int) -> list[tuple[np.ndarray, np.ndarray]]:
    body = _make_body_mask(int(width), side)
    krot = -1 if turn_dir == "cw" else 1
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for anchor in ("tl", "tr", "bl", "br"):
        attached = _embed_with_marker(body, anchor, body_color, marker_color, bg)
        scattered = _crop_non_bg(np.rot90(attached, -int(krot)), bg)
        if _corner_matches_turn(side, _marker_corner(scattered, body_color, bg), turn_dir):
            candidates.append((scattered, attached))
    return candidates


def _place_attached_sprite(out: np.ndarray, arr: np.ndarray, side: str, body_color: int, hub_r: int, hub_c: int, n: int, bg: int) -> bool:
    core_mask = arr == int(body_color)
    full_mask = arr != int(bg)
    delta = _attachment_delta(core_mask, side, hub_r, hub_c, n)
    if delta is None:
        return False
    sr, sc = map(int, delta)
    H, W = out.shape
    placed = {(int(rr + sr), int(cc + sc)) for rr, cc in np.argwhere(full_mask)}
    if any(not (0 <= R < H and 0 <= C < W) for R, C in placed):
        return False
    if any(int(hub_r) <= R < int(hub_r + n) and int(hub_c) <= C < int(hub_c + n) for R, C in placed):
        return False
    for R, C in placed:
        val = int(arr[int(R - sr), int(C - sc)])
        if val == int(bg):
            continue
        if int(out[int(R), int(C)]) != int(bg):
            return False
    paste(out, arr, int(sr), int(sc), transparent=bg)
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG
    H = rng.randint(24, 28)
    W = rng.randint(24, 28)

    colors = [c for c in range(10) if c != int(bg)]
    rng.shuffle(colors)
    top_col, bot_col, left_col, right_col, marker_col = colors[:5]
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}

    sprite_sides = rng.sample(list(SIDES), rng.randint(2, 4))
    turn_dirs = {side: rng.choice(["cw", "ccw"]) for side in sprite_sides}
    if len(sprite_sides) >= 2 and len({turn_dirs[s] for s in sprite_sides}) == 1:
        side = rng.choice(sprite_sides)
        turn_dirs[side] = "ccw" if turn_dirs[side] == "cw" else "cw"

    hub_r = rng.randint(7, H - HUB_N - 7)
    hub_c = rng.randint(7, W - HUB_N - 7)

    out = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, HUB_N - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + HUB_N - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, HUB_N - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + HUB_N - 1] = int(right_col)

    sprite_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in SIDES:
        if side not in sprite_sides:
            continue
        candidates = _build_sprite_candidates(side, 4, turn_dirs[side], side_colors[side], marker_col, bg)
        rng.shuffle(candidates)
        placed_any = False
        for scattered, attached in candidates:
            tmp = out.copy()
            if _place_attached_sprite(tmp, attached, side, side_colors[side], hub_r, hub_c, HUB_N, bg):
                out = tmp
                sprite_pairs[side] = (scattered, attached)
                placed_any = True
                break
        if not placed_any:
            return None

    inp = np.full((H, W), int(bg), dtype=int)
    inp[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N] = out[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N]
    canvas = inp.copy()

    for side in sprite_sides:
        scattered = sprite_pairs[side][0]
        try:
            pr, pc = random_free_location(canvas, scattered, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, scattered, pr, pc, transparent=bg)
        paste(inp, scattered, pr, pc, transparent=bg)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(300):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 300 attempts")
