# Combo variant of 2c181942 (index 8)
# combo_spec: D1/2+D4/0
# Name: 连接器配对+多色复合精灵
#
# Design analysis
# ===============
# a) 具体化与行为追踪
#    - 保留原题的 central hub：4x4 空心方框，四条边四种不同颜色，角和内部仍为 bg。
#    - 每个真实 sprite 现在是一个三色 8 连通组件：
#      1. 外层壳色 outer：覆盖主体轮廓；
#      2. 内层装饰色 inner：固定 2 个内部像素，输出中必须保留；
#      3. connector：1 个单像素，落在最终贴附时的接触边两格中的某一槽位。
#    - hub 的每条 active 边外侧也有 1 个 marker 单像素，颜色与目标 sprite 的 connector 相同。
#    - 原题流程是“按边色找同色单色块，再靠几何挑旋转”。融合后流程改成：
#      1. 先找到 hub 及四边外侧 marker；
#      2. 再从 hub 外全部前景里按 8 连通抽取整只三色 sprite，而不是按某一种颜色抽；
#      3. 用 connector 颜色把 sprite 和 hub 边配对；
#      4. 试 4 个旋转，只接受同时满足：
#         - 原题接触边长度/镜像对称约束；
#         - connector 恰好落在该边 marker 指定的槽位；
#      5. 平移整只 sprite 贴到 hub，对内层装饰色和 connector 都原样保留。
#    - 为了避免退化回 D4/0 的“外层色直接配边”，生成器刻意把 outer 设成另一个 active 边的 hub 颜色
#      （active sides 上做 derangement）。因此 outer 仍然是可见外壳，但不再决定目标边，真正起作用的是 connector。
# b) 复杂度校准
#    - connector 颜色在同一题内唯一，不跨边复用；marker 槽位只有 2 档，足以体现方向约束但不会过复杂。
#    - inner 色在同题内少量复用，支持“这是装饰，不负责匹配”的归纳。
#    - outer 色来自 hub 边色集合，但被故意打乱；这样视觉上仍和 hub 色系相关，同时又不会让 connector 失去作用。
# c) 隐式假设审查
#    - 原 solver 的 `m = (g == col) & (~hub_mask)` 假设 sprite 是单色且按边色可直接抽出；融合后失效。
#    - 原 solver 旋转时只看 mask；融合后还必须检查 connector 单像素在接触边的精确槽位。
#    - D4/0 单变体里“outer 色就是目标边色”的前提，在本融合里被有意取消，否则 D1/2 会沦为摆设。
#    - 新 solver 的关键前提：
#      1. 每个真实 sprite 恰有 3 色，且仅 connector 颜色是 singleton；
#      2. 每条 active 边外侧至多 1 个 marker，inactive 边没有 marker；
#      3. connector 颜色一一对应到 active 边和 sprite；
#      4. sprite 的 inner 恰好是 2 像素，outer 为主色，三者形成单个 8 连通组件。
#      generator 会显式保证这些条件。
# d) 放置合法性
#    - 仍采用 output-first：先在输出里放 hub 和已贴附 sprite，确认不越界、不重叠，再拆成输入并散放。
#    - hub 周围预留 `max_extent + 2` 边距；输入散放时用 `random_free_location(..., padding=2)`，避免 sprite 与 marker
#      或别的 sprite 8 连通粘在一起。
# e) 视觉可观察性
#    - 人类会同时看到三条线索：
#      1. sprite 是复合三色对象，输出贴附后 inner 装饰仍在；
#      2. hub 外侧的小 marker 与 sprite 上的小 connector 同色成对；
#      3. outer 色虽然来自 hub 边色集合，但并不指向同色边，因而 connector 明显才是决定配对的线索。
# f) 可视化整洁度
#    - marker 仅 1 像素且紧贴 hub 外侧，不引入额外大结构。
#    - sprite 模板数量少、尺寸小、配色受控；inner 固定在内部，不和 marker/hub 重叠。
# g) 变化可推断性
#    - 颜色种类上限受控：bg 1，hub edge 4，connector 2-4，inner 1，outer 不新增颜色而复用 hub edge 色。
#    - 关键控制变量可量化为：active sprite 的 outer 是否与目标边色不同。生成器保证这一值恒为 active sprite 数，
#      从而 connector 配对在样本中始终是必要条件，不会退化成原题或 D4/0 单变体。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


SIDES = ("top", "bot", "left", "right")


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_CANONICAL_TEMPLATES: tuple[tuple[np.ndarray, tuple[tuple[int, int], tuple[int, int]]], ...] = (
    (
        _rows_to_mask(
            [
                "1000",
                "1111",
                "1111",
                "1000",
            ]
        ),
        ((1, 1), (2, 1)),
    ),
    (
        _rows_to_mask(
            [
                "1100",
                "1111",
                "1111",
                "1100",
            ]
        ),
        ((1, 1), (2, 1)),
    ),
    (
        _rows_to_mask(
            [
                "10000",
                "11111",
                "11111",
                "10000",
            ]
        ),
        ((1, 1), (2, 1)),
    ),
)


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


def _orient_from_left(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, -1)
    return np.rot90(arr, 1)


def _sprite_connector_matches(arr: np.ndarray, mask: np.ndarray, side: str, conn_col: int, slot: int) -> bool:
    seg = _edge_segment(mask, _contact_side_for_hub(side))
    if seg is None:
        return False
    pts = np.argwhere((arr == int(conn_col)) & mask)
    if pts.shape[0] != 1:
        return False
    rr, cc = map(int, pts[0])
    if side in ("top", "bot"):
        _typ, edge_r, cols = seg
        if slot < 0 or slot >= len(cols):
            return False
        return (rr, cc) == (int(edge_r), int(cols[slot]))
    _typ, edge_c, rows = seg
    if slot < 0 or slot >= len(rows):
        return False
    return (rr, cc) == (int(rows[slot]), int(edge_c))


def _attachment_info(arr: np.ndarray, mask: np.ndarray, side: str, conn_col: int, slot: int, target_len: int):
    need_edge = _contact_side_for_hub(side)
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return None
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return None

    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return None

    if need_edge in ("left", "right"):
        _typ, _edge_c, rows = seg
        if len(rows) != int(target_len) or not _is_contig(rows):
            return None
    else:
        _typ, _edge_r, cols = seg
        if len(cols) != int(target_len) or not _is_contig(cols):
            return None

    if not _sprite_connector_matches(arr, mask, side, int(conn_col), int(slot)):
        return None
    return seg


def _placeable_rotation(
    arr: np.ndarray,
    mask: np.ndarray,
    side: str,
    conn_col: int,
    slot: int,
    fixed: int,
    span: list[int],
    target_len: int,
    out: np.ndarray,
    bg: int,
):
    H, W = out.shape
    need_edge = _contact_side_for_hub(side)
    for krot in range(4):
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        seg = _attachment_info(rot_arr, rot_mask, side, int(conn_col), int(slot), int(target_len))
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

        placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(rot_mask)}
        if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in placed):
            continue

        ok = True
        for rr, cc in placed:
            val = int(rot_arr[int(rr - dr), int(cc - dc)])
            if val == int(bg):
                continue
            cur = int(out[int(rr), int(cc)])
            if cur != int(bg) and cur != val:
                ok = False
                break
        if ok:
            return rot_arr, rot_mask, int(dr), int(dc)
    return None


def _extract_sprite_components(g: np.ndarray, hub_mask: np.ndarray, bg: int):
    fg = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(fg, connectivity=8)
    out = []
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        if int(comp_mask.sum()) < 4:
            continue
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = g[comp_mask]
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        vals = [int(v) for v in arr[comp_mask]]
        counts = Counter(vals)
        singleton_cols = [col for col, cnt in counts.items() if int(cnt) == 1]
        if len(counts) == 3 and len(singleton_cols) == 1:
            out.append((int(cid), arr, comp_mask, counts, int(singleton_cols[0])))
    return out


def _attachment_position(mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + n), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + n)


def _deranged_side_mapping(sides: list[str], rng: random.Random) -> dict[str, str] | None:
    if len(sides) < 2:
        return None
    if len(sides) == 2:
        return {sides[0]: sides[1], sides[1]: sides[0]}
    perm = sides[:]
    for _ in range(100):
        rng.shuffle(perm)
        if all(perm[i] != sides[i] for i in range(len(sides))):
            return {sides[i]: perm[i] for i in range(len(sides))}
    return None


def _build_composite_sprite(
    side: str,
    outer_color: int,
    inner_color: int,
    conn_col: int,
    slot: int,
    bg: int,
    rng: random.Random,
) -> tuple[np.ndarray, np.ndarray] | None:
    template_mask, inner_cells = rng.choice(_CANONICAL_TEMPLATES)
    left_arr = np.full(template_mask.shape, -1, dtype=int)
    left_arr[template_mask] = int(outer_color)
    for rr, cc in inner_cells:
        left_arr[int(rr), int(cc)] = int(inner_color)

    arr = _orient_from_left(left_arr, side)
    mask = arr != -1
    arr = np.where(mask, arr, int(bg)).astype(int)
    arr, mask = _crop_to_mask(arr, mask)

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
    return arr, mask


def _valid_rotation_count(arr: np.ndarray, mask: np.ndarray, side: str, conn_col: int, slot: int, target_len: int) -> int:
    count = 0
    for krot in range(4):
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if _attachment_info(rot_arr, rot_mask, side, int(conn_col), int(slot), int(target_len)) is not None:
            count += 1
    return count


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

    side_markers: dict[str, tuple[int, int]] = {}
    for side in SIDES:
        found = []
        for slot, (rr, cc) in enumerate(_hub_marker_candidates(int(r), int(c), int(n), side)):
            if 0 <= rr < H and 0 <= cc < W and int(g[int(rr), int(cc)]) != int(bg):
                found.append((int(g[int(rr), int(cc)]), int(slot)))
        assert len(found) <= 1
        if found:
            side_markers[side] = found[0]

    sprites_by_connector: dict[int, list[tuple[np.ndarray, np.ndarray]]] = {}
    for _cid, arr, mask, _counts, conn_col in _extract_sprite_components(g, hub_mask, bg):
        sprites_by_connector.setdefault(int(conn_col), []).append((arr, mask))

    target = int(n - 2)
    segs = {
        "top": (int(r - 1), list(range(int(c + 1), int(c + n - 1)))),
        "bot": (int(r + n), list(range(int(c + 1), int(c + n - 1)))),
        "left": (int(c - 1), list(range(int(r + 1), int(r + n - 1)))),
        "right": (int(c + n), list(range(int(r + 1), int(r + n - 1)))),
    }

    for side in SIDES:
        marker = side_markers.get(side)
        if marker is None:
            continue
        conn_col, slot = marker
        candidates = sprites_by_connector.get(int(conn_col), [])
        assert len(candidates) == 1
        spr_arr, spr_mask = candidates[0]
        fixed, span = segs[side]

        placed = _placeable_rotation(
            spr_arr,
            spr_mask,
            side,
            int(conn_col),
            int(slot),
            int(fixed),
            span,
            int(target),
            out,
            int(bg),
        )
        assert placed is not None
        arr, mask, dr, dc = placed
        paste(out, arr, int(dr), int(dc), mask=mask, overwrite=True, clip=True)

    return out


def _analyze_input_features(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    hub = _find_hub(g, bg)
    assert hub is not None
    _score, (r, c, n), (top_col, bot_col, left_col, right_col) = hub
    side_edge = {"top": int(top_col), "bot": int(bot_col), "left": int(left_col), "right": int(right_col)}

    hub_mask = np.zeros_like(g, dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    side_markers: dict[str, tuple[int, int]] = {}
    for side in SIDES:
        for slot, (rr, cc) in enumerate(_hub_marker_candidates(int(r), int(c), int(n), side)):
            if 0 <= rr < g.shape[0] and 0 <= cc < g.shape[1] and int(g[int(rr), int(cc)]) != int(bg):
                side_markers[side] = (int(g[int(rr), int(cc)]), int(slot))

    sprites = _extract_sprite_components(g, hub_mask, bg)
    connector_to_counts = {int(conn_col): counts for _cid, _arr, _mask, counts, conn_col in sprites}

    active = 0
    shell_mismatch = 0
    decorated = 0
    for side, (conn_col, _slot) in side_markers.items():
        counts = connector_to_counts[int(conn_col)]
        non_conn = {int(col): int(cnt) for col, cnt in counts.items() if int(col) != int(conn_col)}
        assert len(non_conn) == 2
        shell_col = max(non_conn.items(), key=lambda kv: kv[1])[0]
        inner_cnt = min(non_conn.values())
        active += 1
        decorated += inner_cnt
        if int(shell_col) != int(side_edge[side]):
            shell_mismatch += 1
    return {
        "active_sides": int(active),
        "shell_mismatch_count": int(shell_mismatch),
        "decorative_pixels": int(decorated),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    n = 4
    edge_len = n - 2

    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    all_colors = [c for c in range(10) if c != int(bg)]
    rng.shuffle(all_colors)
    top_col, bot_col, left_col, right_col = map(int, all_colors[:4])
    side_edge_colors = {"top": int(top_col), "bot": int(bot_col), "left": int(left_col), "right": int(right_col)}

    remaining = [int(c) for c in all_colors[4:]]
    sides = list(SIDES)
    k = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, int(k))

    if len(remaining) < int(k) + 1:
        return None
    connector_colors = [int(c) for c in remaining[:k]]
    inner_color = int(remaining[k])
    side_conn_colors = {side: int(connector_colors[i]) for i, side in enumerate(sprite_sides)}

    shell_side_map = _deranged_side_mapping(sprite_sides, rng)
    if shell_side_map is None:
        return None
    side_outer_colors = {side: int(side_edge_colors[shell_side_map[side]]) for side in sprite_sides}
    side_slots = {side: rng.randint(0, 1) for side in sprite_sides}

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        arr_mask = _build_composite_sprite(
            side,
            int(side_outer_colors[side]),
            int(inner_color),
            int(side_conn_colors[side]),
            int(side_slots[side]),
            int(bg),
            rng,
        )
        if arr_mask is None:
            return None
        arr, mask = arr_mask
        if _valid_rotation_count(arr, mask, side, int(side_conn_colors[side]), int(side_slots[side]), int(edge_len)) != 1:
            return None
        sprite_defs[side] = (arr, mask)

    max_sprite_h = max(int(arr.shape[0]) for arr, _mask in sprite_defs.values())
    max_sprite_w = max(int(arr.shape[1]) for arr, _mask in sprite_defs.values())
    margin = max(max_sprite_h, max_sprite_w) + 2

    hub_r_min = margin
    hub_r_max = H - n - margin
    hub_c_min = margin
    hub_c_max = W - n - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), int(bg), dtype=int)
    for ci in range(1, n - 1):
        out[int(hub_r), int(hub_c + ci)] = int(top_col)
        out[int(hub_r + n - 1), int(hub_c + ci)] = int(bot_col)
    for ri in range(1, n - 1):
        out[int(hub_r + ri), int(hub_c)] = int(left_col)
        out[int(hub_r + ri), int(hub_c + n - 1)] = int(right_col)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        arr, mask = sprite_defs[side]
        sr, sc = _attachment_position(mask, side, int(hub_r), int(hub_c), int(n))
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if mask[int(rr), int(cc)] and int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None
        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), int(bg), dtype=int)
    inp[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)] = out[int(hub_r) : int(hub_r + n), int(hub_c) : int(hub_c + n)]
    for side in sprite_sides:
        slot = int(side_slots[side])
        rr, cc = _hub_marker_candidates(int(hub_r), int(hub_c), int(n), side)[slot]
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        if int(inp[int(rr), int(cc)]) != int(bg):
            return None
        inp[int(rr), int(cc)] = int(side_conn_colors[side])

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        krot = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        scatter = np.where(rot_mask, rot_arr, int(bg)).astype(int)
        try:
            pr, pc = random_free_location(canvas, scatter, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, scatter, int(pr), int(pc), transparent=int(bg))
        paste(inp, scatter, int(pr), int(pc), transparent=int(bg))

    analysis = _analyze_input_features(inp)
    if analysis["active_sides"] != int(k):
        return None
    if analysis["shell_mismatch_count"] != int(k):
        return None
    if analysis["decorative_pixels"] != int(2 * k):
        return None

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
