# Combo variant of 2c181942 (index 5)
# combo_spec: D1/2+D3/1
# Name: 连接器标记配对+条件旋转
#
# Design analysis
# ===============
# design:
#   concrete_behavior:
#     - 保留原题中心 hub：4x4 空心方框、四条边四种不同颜色，输出中的贴附几何仍然是“与对应边齐平接触”。
#     - 原题的“sprite 主体颜色 == hub 边颜色”被取消。所有真实 sprite 共用一个 body 颜色。
#     - 每个真实 sprite 上有两个功能像素：
#       1. connector：位于未来接触边的 2 个槽位之一，颜色与目标 hub 边旁的单像素 marker 相同。
#       2. turn marker：位于 sprite bbox 的一个角点，颜色固定；它不参与配对，只编码旋转方向。
#     - 为了控制调色板，connector 颜色直接复用 hub 边颜色；因此视觉上能看见“主体色不再配边，
#       真正配边的是那一个小 connector 点以及 hub 外侧的同色点”。
#   behavior_trace:
#     - 原题：
#       1. 看 top 边颜色，例如 2。
#       2. 在 hub 外找颜色 2 的单色 sprite。
#       3. 试 4 个 rot90，找到唯一满足接触边长度与对称约束的朝向后贴到 top。
#     - 本融合变体：
#       1. 看 top 边外侧两个候选槽位中是否有 marker；若 (r-1,c+2) 处有颜色 2，则 top 激活、槽位=右槽。
#       2. 在 hub 外找含有唯一颜色 2 connector 的多色 sprite；主体色不再提供配对线索。
#       3. 读取该 sprite 的角点 turn marker：marker 在左列角点表示对散落态执行 CW 90 度，
#          marker 在右列角点表示执行 CCW 90 度。
#       4. 只做这一次条件旋转，然后检查 connector 是否恰好落到 top 的对应槽位，再贴附。
#     - 差异：
#       1. side 归属由 connector marker 决定，不再由主体颜色决定。
#       2. 旋转由角点 marker 决定，不再靠“枚举 4 个旋转后试错”。
#       3. connector 槽位与 hub marker 槽位还要一致，因此配对与旋转在同一条推理链上协同生效。
#   complexity_calibration:
#     - turn 规则只保留两档：左列角点= CW，右列角点= CCW，避免 2-4 个样例下过度稀疏。
#     - 仍保留 connector 槽位 0/1 的自由度，让“hub 外侧 marker 的具体位置”成为可见约束，
#       防止只看 connector 颜色就能忽略空间关系。
#     - sprite 几何使用 2x3/2x4 长条家族；形状简单，人的注意力集中在 connector 配对和 turn marker 上，
#       不会被额外几何噪声淹没。
#   implicit_assumption_review:
#     - 原 solver 的 `(g == edge_color)` 选块逻辑已失效，因为主体色不再等于边色；改为先抽取所有 hub 外组件，
#       再用“组件内部恰有 1 个 connector 色像素”去配 active hub side。
#     - 原 solver 依赖单色 sprite；这里不成立。组件现在应满足：1 个主色 + 1 个 connector 单例 + 1 个 turn
#       marker 单例。generator 显式保证这一点。
#     - 原 solver 通过试四个旋转寻找唯一解；这里不再枚举，而是假设 turn marker 总在 bbox 角点且编码有效。
#       generator 通过“先生成 attached，再反向旋转为 scattered，并验 corner 类别”保证该假设。
#     - 贴附后 output 不保留 hub 外侧 marker，因为它会被贴上的 connector 覆盖；因此 out 仍只先复制 hub 本体。
#   placement_legality:
#     - generator 仍采用 output-first。先放 hub，再把 attached sprite 贴好，最后拆成散落输入。
#     - hub 周围预留 margin，保证 attached sprite、hub 外侧 marker、散落 sprite 都不越界。
#     - 散落放置使用 `padding=2` 和 8 连通隔离，避免 sprite 与 hub marker 或其他 sprite 粘连。
#   visual_observability:
#     - 人类可直接看到：所有真实 sprite 主体都同色，但每块各自带一个与某条 hub 边同色的小 connector 点。
#     - 角点 turn marker 在不同样例中会落在左列或右列角，输出里对应的贴附方向变化可与之对照。
#     - 同一题里通常会同时出现 CW/CCW 两类 turn，避免规则退化成“总是同向旋转”。
#   visual_cleanliness:
#     - 每个真实 sprite 只用 3 种颜色：body、connector、turn marker；hub 附近每条 active side 只有 1 个外侧
#       marker，不会出现多层重叠标记。
#     - turn marker 和 connector 都是大块矩形上的单像素重着色，位置稳定，读图负担低。
#   color_inference:
#     - 颜色数受控：bg 1、hub edge/connector 4、body 1、turn marker 1，共 7 种核心颜色。
#     - connector 颜色成对出现：hub 外一次、对应 sprite 一次。turn marker 颜色在所有真实 sprite 上复用，
#       强化“这个角点只控制旋转、不控制配对”的观察。
#   new_solver_assumptions:
#     - 每个真实 sprite 是一个 8 连通矩形组件，且恰有 1 个 connector 单例和 1 个 turn marker 单例。
#     - turn marker 必在 sprite bbox 四角之一；左列角点表示 CW，右列角点表示 CCW。
#     - 每个 active side 外侧最多 1 个 marker；其槽位与目标 sprite 的 connector 槽位一致。
#     - generator 会过滤掉不满足这些条件的候选，因此 solver 可以按这些前提解析。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


BG = 8
HUB_N = 4
EDGE_LEN = HUB_N - 2
SIDES = ("top", "bot", "left", "right")


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


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


def _marker_corner(arr: np.ndarray, body_col: int, conn_col: int, bg: int) -> str | None:
    pts = np.argwhere((arr != int(bg)) & (arr != int(body_col)) & (arr != int(conn_col)))
    if pts.shape[0] != 1:
        return None
    mr, mc = map(int, pts[0])
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


def _turn_from_corner(corner: str | None) -> int | None:
    if corner in {"tl", "bl"}:
        return -1
    if corner in {"tr", "br"}:
        return 1
    return None


def _corner_matches_turn(turn_dir: str, corner: str | None) -> bool:
    if turn_dir == "cw":
        return corner in {"tl", "bl"}
    return corner in {"tr", "br"}


def _connector_coord(arr: np.ndarray, conn_col: int) -> tuple[int, int] | None:
    pts = np.argwhere(arr == int(conn_col))
    if pts.shape[0] != 1:
        return None
    rr, cc = map(int, pts[0])
    return int(rr), int(cc)


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


def _attachment_delta(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int] | None:
    target = int(n - 2)
    segs = {
        "top": ("bot", int(hub_r - 1), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "bot": ("top", int(hub_r + n), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "left": ("right", int(hub_c - 1), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
        "right": ("left", int(hub_c + n), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
    }
    need_edge, fixed, span = segs[side]
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


def _make_body_mask(width: int, side: str) -> np.ndarray:
    base = np.ones((EDGE_LEN, int(width)), dtype=bool)
    if side == "left":
        return base.copy()
    if side == "right":
        return np.fliplr(base)
    if side == "top":
        return np.rot90(base, -1)
    return np.rot90(base, 1)


def _corner_positions(mask: np.ndarray) -> dict[str, tuple[int, int]]:
    h, w = map(int, mask.shape)
    return {
        "tl": (0, 0),
        "tr": (0, int(w - 1)),
        "bl": (int(h - 1), 0),
        "br": (int(h - 1), int(w - 1)),
    }


def _build_attached_candidates(
    side: str,
    width: int,
    slot: int,
    body_col: int,
    conn_col: int,
    turn_col: int,
    bg: int,
    turn_dir: str,
) -> list[tuple[np.ndarray, np.ndarray]]:
    mask = _make_body_mask(int(width), side)
    arr0 = np.where(mask, int(body_col), int(bg)).astype(int)
    seg = _edge_segment(mask, _contact_side_for_hub(side))
    assert seg is not None
    if side in ("top", "bot"):
        _typ, edge_r, cols = seg
        conn_rc = (int(edge_r), int(cols[int(slot)]))
    else:
        _typ, edge_c, rows = seg
        conn_rc = (int(rows[int(slot)]), int(edge_c))
    arr0[conn_rc] = int(conn_col)

    krot = -1 if turn_dir == "cw" else 1
    out: list[tuple[np.ndarray, np.ndarray]] = []
    for _name, (rr, cc) in _corner_positions(mask).items():
        if (int(rr), int(cc)) == conn_rc:
            continue
        attached = arr0.copy()
        attached[int(rr), int(cc)] = int(turn_col)
        scattered = _crop_non_bg(np.rot90(attached, -int(krot)), int(bg))
        corner = _marker_corner(scattered, int(body_col), int(conn_col), int(bg))
        if _corner_matches_turn(turn_dir, corner):
            out.append((scattered, attached))
    return out


def _place_attached(out: np.ndarray, arr: np.ndarray, side: str, hub_r: int, hub_c: int, n: int, bg: int) -> tuple[int, int] | None:
    mask = arr != int(bg)
    delta = _attachment_delta(mask, side, int(hub_r), int(hub_c), int(n))
    if delta is None:
        return None
    sr, sc = map(int, delta)
    H, W = out.shape
    for rr, cc in np.argwhere(mask):
        R, C = int(rr + sr), int(cc + sc)
        if not (0 <= R < H and 0 <= C < W):
            return None
        if int(hub_r) <= R < int(hub_r + n) and int(hub_c) <= C < int(hub_c + n):
            return None
        if int(out[R, C]) != int(bg):
            return None
    paste(out, arr, int(sr), int(sc), transparent=int(bg))
    return int(sr), int(sc)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None
    _score, (r, c, n), (_top_col, _bot_col, _left_col, _right_col) = hub

    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    side_markers: dict[str, tuple[int, int, tuple[int, int]]] = {}
    for side in SIDES:
        found: list[tuple[int, int, tuple[int, int]]] = []
        for slot, (rr, cc) in enumerate(_hub_marker_candidates(int(r), int(c), int(n), side)):
            if 0 <= rr < H and 0 <= cc < W and int(g[int(rr), int(cc)]) != int(bg):
                found.append((int(g[int(rr), int(cc)]), int(slot), (int(rr), int(cc))))
        assert len(found) <= 1
        if found:
            side_markers[side] = found[0]

    outside = (g != int(bg)) & (~hub_mask)
    comps, nc = label_mask(outside, connectivity=8)
    components: list[np.ndarray] = []
    for cid in range(1, int(nc) + 1):
        cmask = comps == int(cid)
        if int(cmask.sum()) < 4:
            continue
        arr = np.full_like(g, int(bg))
        arr[cmask] = g[cmask]
        components.append(_crop_non_bg(arr, int(bg)))

    used_ids: set[int] = set()
    for side in SIDES:
        marker = side_markers.get(side)
        if marker is None:
            continue
        conn_col, _slot, marker_pos = marker
        candidate_idx = None
        body_col = None
        krot = None

        for idx, arr in enumerate(components):
            if idx in used_ids:
                continue
            vals = [int(v) for v in arr[arr != int(bg)].ravel()]
            counts = Counter(vals)
            if int(counts.get(int(conn_col), 0)) != 1:
                continue
            multi_cols = [col for col, cnt in counts.items() if int(cnt) > 1 and int(col) != int(conn_col)]
            single_other = [col for col, cnt in counts.items() if int(cnt) == 1 and int(col) != int(conn_col)]
            if len(multi_cols) != 1 or len(single_other) != 1:
                continue
            trial_body = int(multi_cols[0])
            corner = _marker_corner(arr, int(trial_body), int(conn_col), int(bg))
            trial_krot = _turn_from_corner(corner)
            if trial_krot is None:
                continue
            candidate_idx = int(idx)
            body_col = int(trial_body)
            krot = int(trial_krot)
            break

        assert candidate_idx is not None and body_col is not None and krot is not None
        used_ids.add(int(candidate_idx))

        rotated = _crop_non_bg(np.rot90(components[int(candidate_idx)], int(krot)), int(bg))
        mask = rotated != int(bg)
        delta = _attachment_delta(mask, side, int(r), int(c), int(n))
        assert delta is not None
        dr, dc = map(int, delta)

        conn_rc = _connector_coord(rotated, int(conn_col))
        assert conn_rc is not None
        assert (int(conn_rc[0] + dr), int(conn_rc[1] + dc)) == tuple(marker_pos)

        for rr, cc in np.argwhere(mask):
            R, C = int(rr + dr), int(cc + dc)
            assert 0 <= R < H and 0 <= C < W
            val = int(rotated[int(rr), int(cc)])
            cur = int(out[int(R), int(C)])
            assert cur == int(bg) or cur == val

        paste(out, rotated, int(dr), int(dc), transparent=int(bg), overwrite=True, clip=True)

    return out


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG
    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    colors = [c for c in range(10) if c != int(bg)]
    rng.shuffle(colors)
    top_col, bot_col, left_col, right_col = map(int, colors[:4])
    body_col = int(colors[4])
    turn_col = int(colors[5])
    side_colors = {
        "top": int(top_col),
        "bot": int(bot_col),
        "left": int(left_col),
        "right": int(right_col),
    }

    sprite_sides = rng.sample(list(SIDES), rng.randint(2, 4))
    side_slots = {side: rng.randint(0, 1) for side in sprite_sides}
    turn_dirs = {side: rng.choice(["cw", "ccw"]) for side in sprite_sides}
    if len(sprite_sides) >= 2 and len({turn_dirs[side] for side in sprite_sides}) == 1:
        flip_side = rng.choice(sprite_sides)
        turn_dirs[flip_side] = "ccw" if turn_dirs[flip_side] == "cw" else "cw"

    widths = {side: rng.choice([3, 4]) for side in sprite_sides}
    sprite_pairs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        candidates = _build_attached_candidates(
            side=side,
            width=int(widths[side]),
            slot=int(side_slots[side]),
            body_col=int(body_col),
            conn_col=int(side_colors[side]),
            turn_col=int(turn_col),
            bg=int(bg),
            turn_dir=turn_dirs[side],
        )
        if not candidates:
            return None
        sprite_pairs[side] = rng.choice(candidates)

    max_h = max(int(max(scattered.shape[0], attached.shape[0])) for scattered, attached in sprite_pairs.values())
    max_w = max(int(max(scattered.shape[1], attached.shape[1])) for scattered, attached in sprite_pairs.values())
    margin = max(max_h, max_w) + 2

    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None
    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, HUB_N - 1):
        out[hub_r, hub_c + ci] = int(top_col)
        out[hub_r + HUB_N - 1, hub_c + ci] = int(bot_col)
    for ri in range(1, HUB_N - 1):
        out[hub_r + ri, hub_c] = int(left_col)
        out[hub_r + ri, hub_c + HUB_N - 1] = int(right_col)

    for side in sprite_sides:
        attached = sprite_pairs[side][1]
        placed = _place_attached(out, attached, side, int(hub_r), int(hub_c), int(HUB_N), int(bg))
        if placed is None:
            return None

    inp = np.full((H, W), int(bg), dtype=int)
    inp[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N] = out[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N]
    for side in sprite_sides:
        slot = int(side_slots[side])
        rr, cc = _hub_marker_candidates(int(hub_r), int(hub_c), int(HUB_N), side)[slot]
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        if int(inp[rr, cc]) != int(bg):
            return None
        inp[rr, cc] = int(side_colors[side])

    canvas = inp.copy()
    for side in sprite_sides:
        scattered = sprite_pairs[side][0]
        try:
            pr, pc = random_free_location(canvas, scattered, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, scattered, int(pr), int(pc), transparent=int(bg))
        paste(inp, scattered, int(pr), int(pc), transparent=int(bg))

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
