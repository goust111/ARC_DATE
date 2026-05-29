# Variant of 2c181942
# Dimension: D1 — 匹配条件：颜色配对机制 (match_cond)
# Change: s.head.color == d.edge.color — 精灵颜色与hub边缘颜色相同即配对，颜色唯一决定精灵应贴附... → 连接器标记匹配
#
# Design analysis
# ===============
# design:
#   concrete_behavior:
#     - 保留原题中心 hub：4x4 空心方框，四条边仍然用 4 个不同颜色表示；输出里的贴附几何保持原题不变。
#     - 去掉“sprite 主体颜色 == hub 边缘颜色”的配对方式。所有真实 sprite 改成统一 body 颜色，
#       每个 sprite 只在接触边上放 1 个特殊 connector 像素；hub 对应边的外侧，也放 1 个同色 marker 像素。
#     - hub marker 正好落在该 sprite 贴附后的接触边像素位置上，因此 output 中 marker 会被 sprite 覆盖，
#       最终可见的是“贴上去的 sprite 自己的 connector 像素”。
#     - 由于 N 固定为 4，所以每条边只有 2 个可接触槽位；hub marker 会占其中一个槽位，sprite connector
#       也必须落在对应槽位。这样“颜色一致”决定配哪一条边，“槽位/方向一致”决定贴附朝向是否正确。
#   behavior_trace:
#     - 原题局部流程：
#       1. 读取某条 hub 边颜色，例如 top=2。
#       2. 在 hub 外找颜色 2 的最大连通块。
#       3. 尝试旋转，找到满足对称性和接触边长度的朝向后贴到 top。
#     - 本变体局部流程：
#       1. 读取 top 边外侧 2 个候选格中唯一的 marker，例如 (r-1,c+2)=7，槽位=右槽。
#       2. 在 hub 外找“恰有一个单例颜色像素，且该单例颜色为 7”的 sprite 连通块。
#       3. 尝试旋转；除了原题的 mask 对称性和接触边长度外，还要求 connector 像素正好位于接触边的右槽。
#       4. 贴附后 connector 像素覆盖原 hub marker 位置。
#     - 差异：
#       1. 原题用主体颜色配边；变体用单像素 connector 颜色配边。
#       2. 原题只靠接触边几何决定旋转；变体额外要求 connector 槽位对齐。
#   complexity_calibration:
#     - 不把 connector 颜色在多个边之间复用，避免 2-4 个样例下人类难以从极少样本推断多重冲突。
#     - 但仍要求“方向/槽位”参与，因为每条边有两个接触槽位，hub marker 和 sprite connector 必须对齐；
#       否则同色但 connector 在另一槽位的旋转视为无效。
#     - 所有真实 sprite 共享同一个 body 颜色，使新规则肉眼可见：不再可能从主体颜色直接读出目标边。
#   implicit_assumption_review:
#     - 原 solver 的 `m = (g == col) & (~hub_mask)` 依赖“sprite 主体颜色就是边颜色”。变体中不成立，必须改成
#       “先按 8 连通抽出组件，再从组件内部识别 connector 单例颜色”。
#     - 原 solver 默认整块 sprite 单色；变体中不成立，因此旋转后不能再只看 mask，需要额外检查 connector
#       颜色像素在接触边上的精确位置。
#     - 原 solver 复制 hub 时只复制方框本体。这里保留该做法，因为 hub 外侧 marker 在 output 中会被贴上的
#       sprite 覆盖，不应作为独立元素保留。
#   placement_legality:
#     - generator 仍采用 output-first：先生成已贴附输出，再拆成输入。
#     - hub 周围保留足够 margin，确保 attached sprite、hub 外侧 marker、以及散落后的 sprite 都不会越界。
#     - 输入中 hub marker 是方框外的单像素组件；随机散落 sprite 时使用 padding>=2，防止它们与 hub marker
#       或其他 sprite 8 连通粘在一起，保证 solver 的组件抽取稳定。
#   visual_observability:
#     - 人类能直接看到：hub 四边颜色和 sprite 主体颜色脱钩，真正起作用的是每个 sprite 上那个小色点，
#       以及 hub 边外侧同色的小色点。
#     - output 中 connector 仍保留在贴附后的接触边上，因此“哪个小色点对到哪条边、对到哪一个槽位”是可见的。
#   visual_cleanliness:
#     - 每个对象最多两种颜色：统一 body 色 + 1 个 connector 色。hub 仍是四条彩边 + 少量外侧 marker。
#       没有多层重叠，也不在同一区域叠加多种细碎标记。
#   color_inference:
#     - 颜色总数受控：bg 1 种，body 1 种，hub edge 4 种，connector 最多 4 种。
#     - connector 颜色总是成对出现：输入里每种 active connector 色恰好出现两次，
#       一次在 hub 侧，一次在某个 sprite 上，利于人类归纳“同色连接器配对”。
#   new_solver_assumptions:
#     - 每个真实 sprite 是一个 8 连通组件，且恰有两种颜色：body 色若干像素 + 1 个 connector 单例像素。
#     - 每个 active hub 边外侧恰有 1 个 marker，inactive 边没有 marker。
#     - active connector 颜色在同一题内唯一对应 1 条边和 1 个 sprite。
#     - generator 显式保证上述条件；solver 按这些条件解析输入。

from __future__ import annotations

from collections import Counter
import random

import numpy as np
from scipy.ndimage import label as _label

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


SIDES = ("top", "bot", "left", "right")


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


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


def _hub_marker_candidates(r: int, c: int, n: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(int(r - 1), int(c + 1)), (int(r - 1), int(c + 2))]
    if side == "bot":
        return [(int(r + n), int(c + 1)), (int(r + n), int(c + 2))]
    if side == "left":
        return [(int(r + 1), int(c - 1)), (int(r + 2), int(c - 1))]
    return [(int(r + 1), int(c + n)), (int(r + 2), int(c + n))]


def _contact_side_for_hub(side: str) -> str:
    return {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]


def _sprite_connector_coord(arr: np.ndarray, mask: np.ndarray, side: str, conn_col: int, slot: int):
    seg = _edge_segment(mask, _contact_side_for_hub(side))
    if seg is None:
        return None
    if side in ("top", "bot"):
        _typ, edge_r, cols = seg
        if slot < 0 or slot >= len(cols):
            return None
        expect = (int(edge_r), int(cols[slot]))
    else:
        _typ, edge_c, rows = seg
        if slot < 0 or slot >= len(rows):
            return None
        expect = (int(rows[slot]), int(edge_c))
    pts = np.argwhere((arr == int(conn_col)) & mask)
    if pts.shape[0] != 1:
        return None
    rr, cc = map(int, pts[0])
    return (rr, cc) if (rr, cc) == expect else None


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
    Variant rule: sprites no longer match hub sides by body color.
    Each active hub side exposes one outside connector marker pixel; each sprite has one
    connector pixel of the same color on its contact edge. Attach the sprite whose
    connector color matches the hub marker, in the rotation where the connector falls on
    the same side-slot as the hub marker.
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

    side_markers: dict[str, tuple[int, int]] = {}
    for side in SIDES:
        found = []
        for slot, (rr, cc) in enumerate(_hub_marker_candidates(int(r), int(c), int(n), side)):
            if 0 <= rr < H and 0 <= cc < W and int(g[int(rr), int(cc)]) != int(bg):
                found.append((int(g[int(rr), int(cc)]), int(slot)))
        assert len(found) <= 1
        if found:
            side_markers[side] = found[0]

    outside_fg = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(outside_fg, connectivity=8)
    sprites_by_connector: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        size = int(comp_mask.sum())
        if size < 4:
            continue
        coords = np.argwhere(comp_mask)
        vals = [int(g[int(rr), int(cc)]) for rr, cc in coords]
        counts = Counter(vals)
        if len(counts) != 2:
            continue
        singleton_cols = [col for col, cnt in counts.items() if int(cnt) == 1]
        if len(singleton_cols) != 1:
            continue
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = g[comp_mask]
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        conn_col = int(singleton_cols[0])
        sprites_by_connector.setdefault(conn_col, []).append((arr, comp_mask))

    target = int(n - 2)
    segs = {
        "top": ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1)))),
        "bot": ("top", int(r + n), list(range(int(c + 1), int(c + n - 1)))),
        "left": ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1)))),
        "right": ("left", int(c + n), list(range(int(r + 1), int(r + n - 1)))),
    }

    for side in SIDES:
        marker = side_markers.get(side)
        if marker is None:
            continue
        conn_col, slot = marker
        candidates = sprites_by_connector.get(int(conn_col), [])
        assert len(candidates) == 1
        spr_arr, spr_mask = candidates[0]
        need_edge, fixed, span = segs[side]

        found = None
        for krot in range(4):
            a = np.rot90(spr_arr, int(krot))
            msk = np.rot90(spr_mask, int(krot))
            a, msk = _crop_to_mask(a, msk)

            if need_edge in ("left", "right"):
                if not bool(np.array_equal(msk, np.flipud(msk))):
                    continue
            else:
                if not bool(np.array_equal(msk, np.fliplr(msk))):
                    continue

            seg = _edge_segment(msk, need_edge)
            if seg is None:
                continue

            if need_edge in ("left", "right"):
                _typ, edge_c, rows = seg
                if len(rows) != int(target) or not _is_contig(rows):
                    continue
                if _sprite_connector_coord(a, msk, side, int(conn_col), int(slot)) is None:
                    continue
                dr = int(span[0] - rows[0])
                dc = int(fixed - edge_c)
            else:
                _typ, edge_r, cols = seg
                if len(cols) != int(target) or not _is_contig(cols):
                    continue
                if _sprite_connector_coord(a, msk, side, int(conn_col), int(slot)) is None:
                    continue
                dr = int(fixed - edge_r)
                dc = int(span[0] - cols[0])

            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(msk)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue

            ok = True
            for rr, cc2 in placed:
                val = int(a[int(rr - dr), int(cc2 - dc)])
                if val == int(bg):
                    continue
                cur = int(out[int(rr), int(cc2)])
                if cur != int(bg) and cur != val:
                    ok = False
                    break
            if not ok:
                continue
            found = (a, msk, int(dr), int(dc))
            break

        assert found is not None
        a, msk, dr, dc = found
        paste(out, a, int(dr), int(dc), mask=msk, overwrite=True, clip=True)

    return out


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

def _make_sprite_mask(edge_len: int, side: str, rng: random.Random) -> np.ndarray | None:
    """
    Reuse the original geometric family: attached mask has the right symmetry and a
    contact edge of length `edge_len`. Exactly one rotation is valid for the intended side.
    """
    e = edge_len
    h = 4
    w = rng.randint(2, 4)
    half_h = (h + 1) // 2

    for _attempt in range(100):
        grid = np.zeros((h, w), dtype=bool)
        for ri in range(half_h):
            mirror_ri = h - 1 - ri
            for ci in range(w):
                val = rng.random() < 0.5
                grid[ri, ci] = val
                grid[mirror_ri, ci] = val

        right_col = grid[:, -1]
        runs = []
        count = 0
        for v in right_col:
            if v:
                count += 1
            else:
                if count > 0:
                    runs.append(count)
                count = 0
        if count > 0:
            runs.append(count)
        if len(runs) != 1 or runs[0] != e:
            continue

        rows_any = grid.any(axis=1)
        cols_any = grid.any(axis=0)
        if not rows_any.any():
            continue
        r0, r1 = np.where(rows_any)[0][[0, -1]]
        c0, c1 = np.where(cols_any)[0][[0, -1]]
        grid = grid[r0 : r1 + 1, c0 : c1 + 1]

        if grid.shape[0] < 2 or grid.shape[1] < 2:
            continue
        if not np.array_equal(grid, np.flipud(grid)):
            continue

        labeled, num = _label(grid, structure=np.ones((3, 3), dtype=int))
        if int(num) != 1:
            continue
        if np.array_equal(grid, np.fliplr(grid)):
            continue

        px = int(grid.sum())
        if px < 4 or px > 12:
            continue

        valid_for_left = 0
        for k in range(4):
            rot = np.rot90(grid, int(k))
            ra = rot.any(axis=1)
            ca = rot.any(axis=0)
            rr0, rr1 = np.where(ra)[0][[0, -1]]
            cc0, cc1 = np.where(ca)[0][[0, -1]]
            cr = rot[rr0 : rr1 + 1, cc0 : cc1 + 1]
            if not np.array_equal(cr, np.flipud(cr)):
                continue
            rc = cr[:, -1]
            runs2 = []
            cnt2 = 0
            for v in rc:
                if v:
                    cnt2 += 1
                else:
                    if cnt2 > 0:
                        runs2.append(cnt2)
                    cnt2 = 0
            if cnt2 > 0:
                runs2.append(cnt2)
            if len(runs2) == 1 and runs2[0] == e:
                valid_for_left += 1
        if valid_for_left != 1:
            continue

        if side == "left":
            return grid.copy()
        if side == "right":
            return np.fliplr(grid)
        if side == "top":
            return np.rot90(grid, -1)
        return np.rot90(grid, 1)
    return None


def _build_sprite_arr(mask: np.ndarray, side: str, body_col: int, conn_col: int, slot: int, bg: int) -> np.ndarray | None:
    arr = np.where(mask, int(body_col), int(bg)).astype(int)
    seg = _edge_segment(mask, _contact_side_for_hub(side))
    if seg is None:
        return None
    if side in ("top", "bot"):
        _typ, edge_r, cols = seg
        if len(cols) != 2 or slot not in (0, 1):
            return None
        arr[int(edge_r), int(cols[slot])] = int(conn_col)
    else:
        _typ, edge_c, rows = seg
        if len(rows) != 2 or slot not in (0, 1):
            return None
        arr[int(rows[slot]), int(edge_c)] = int(conn_col)
    return arr


def _place_attached(mask: np.ndarray, side: str, hub_r: int, hub_c: int, hub_n: int) -> tuple[int, int] | None:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + hub_n), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + hub_n)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    N = 4
    edge_len = N - 2
    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    colors = [c for c in range(10) if c != bg]
    rng.shuffle(colors)
    body_col = int(colors[0])
    top_col, bot_col, left_col, right_col = map(int, colors[1:5])
    connector_colors = list(map(int, colors[5:9]))

    side_edge_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}
    side_conn_colors = dict(zip(SIDES, connector_colors))

    k = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), k)
    side_slots = {side: rng.randint(0, 1) for side in sprite_sides}

    sprite_masks: dict[str, np.ndarray] = {}
    sprite_arrays: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(edge_len, side, rng)
        if mask is None:
            return None
        arr = _build_sprite_arr(mask, side, int(body_col), int(side_conn_colors[side]), int(side_slots[side]), int(bg))
        if arr is None:
            return None
        sprite_masks[side] = mask
        sprite_arrays[side] = arr

    max_sprite_h = max(m.shape[0] for m in sprite_masks.values())
    max_sprite_w = max(m.shape[1] for m in sprite_masks.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - N - margin
    hub_c_min = margin
    hub_c_max = W - N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None
    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, N - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + N - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, N - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + N - 1] = int(right_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        mask = sprite_masks[side]
        arr = sprite_arrays[side]
        sh, sw = mask.shape
        sr, sc = _place_attached(mask, side, int(hub_r), int(hub_c), int(N))
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if mask[rr, cc] and int(out[sr + rr, sc + cc]) != int(bg):
                    return None
        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), int(bg), dtype=int)
    inp[hub_r : hub_r + N, hub_c : hub_c + N] = out[hub_r : hub_r + N, hub_c : hub_c + N]
    for side in sprite_sides:
        slot = int(side_slots[side])
        rr, cc = _hub_marker_candidates(int(hub_r), int(hub_c), int(N), side)[slot]
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        if int(inp[rr, cc]) != int(bg):
            return None
        inp[rr, cc] = int(side_conn_colors[side])

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        krot = rng.choice([1, 2, 3])
        rotated_arr = np.rot90(arr, int(krot))
        rotated_mask = np.rot90(mask, int(krot))
        rows_any = rotated_mask.any(axis=1)
        cols_any = rotated_mask.any(axis=0)
        r0_idx, r1_idx = np.where(rows_any)[0][[0, -1]]
        c0_idx, c1_idx = np.where(cols_any)[0][[0, -1]]
        patch = rotated_arr[r0_idx : r1_idx + 1, c0_idx : c1_idx + 1]
        try:
            pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
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
