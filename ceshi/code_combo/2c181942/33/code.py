# Combo variant of 2c181942 (index 33)
# combo_spec: D1/2+D3/1+D4/0
# Name: 连接器+条件旋转+多色精灵
#
# Design analysis
# ===============
# a) Concrete behavior / behavior trace
#    - 保留原题中央 4x4 空心 hub，四边四种不同颜色；输出中的贴附几何仍然是“沿某边紧贴”。
#    - 融合后每个真实 sprite 都是双色对象：
#      1. 外层壳色：所有真实 sprite 共用同一种壳色，故不能再靠主体颜色配边。
#      2. 第二种颜色：同一只 sprite 内同时承担三种角色：
#         - 接触边上的 connector 像素：与 hub 外侧单像素 marker 同色，用于配对；
#         - bbox 角上的 rotation marker：角点位置编码“顺时针/逆时针旋转 90 度”；
#         - 内层装饰像素：贴附后保留，体现 D4 的多色复合外观。
#    - 一个具体小例子：
#      1. 原题会看 top 边颜色=2，然后去找颜色 2 的单色精灵，再试四种旋转。
#      2. 本融合题先看 top 边外侧两格里唯一 marker，例如颜色 7 且在右槽位。
#      3. 在 hub 外找包含颜色 7 的双色组件；读取该组件哪个 bbox 角落是 7：
#         左侧角表示顺时针，右侧角表示逆时针（规则对 side 是相对的）。
#      4. 按该方向只旋转 90 度后，再检查颜色 7 是否正好落到 top 的接触边右槽位；
#         若是，则整只双色 sprite 贴到 top，内部装饰色一起保留。
#    - 与原题差异：
#      1. 配对不再由主体颜色完成，而由 hub marker 和 sprite connector 的同色关系完成。
#      2. 旋转不再靠“穷举四个朝向+几何唯一性”，而由角点 marker 直接决定 CW/CCW。
#      3. 粘贴时不再只保留单色块，而是保留壳色与内层装饰色的完整双色对象。
# b) Complexity calibration
#    - 若继续让主体颜色决定 side，则会退化回原题，因此这里故意让所有真实 sprite 共用壳色，
#      逼迫 connector 成为必要线索。
#    - 若 rotation marker 再单独占一种全局新颜色，会超出 10 色预算；因此把“连接器/角点/内层”
#      合并成同一第二色，既省颜色又让第二色明显成为信息载体。
#    - 角点规则只用两档：CW / CCW；hub 槽位也只用两档：左槽 / 右槽。2-4 个样例里足以重复支撑。
# c) Hidden assumptions audit
#    - 原代码的 `m = (g == edge_color)` 依赖“sprite 主色 == hub 边色”，现在失效；
#      solver 必须先做 8 连通分块，再按 connector 颜色识别组件。
#    - 原代码假设旋转要靠四向试探。这里改成先读角点，再只做一个 ±90 旋转；
#      因此 generator 也必须保证输入 sprite 恰好离贴附朝向一个四分之一转。
#    - 原代码可直接用整块 mask 做接触几何。这里额外有一个 bbox 角 marker 突出在 core 外侧，
#      所以 solver 先定位并移除这个单像素 marker，再用剩余 core mask 做对称性和接触边判断。
#    - 新 solver 的隐式假设：
#      1. 每个真实 sprite 恰有两种颜色：共享壳色 + 某个 connector/accent 色。
#      2. 每个真实 sprite 恰有一个 connector/accent 色落在 bbox 某个角上。
#      3. 每个 active hub side 外侧恰有一个单像素 marker，且颜色唯一对应一只 sprite。
#      4. generator 显式保证这些条件。
# d) Placement legality
#    - 继续用 output-first：先在输出里放 hub 和已定向 sprite，确认不越界、不重叠；
#      再把 sprite 逆向旋回散落输入，并单独添加 hub marker。
#    - hub 周围按最大 sprite 尺寸预留 margin；散放和干扰物都使用 `random_free_location(..., padding=2)`，
#      避免与 hub marker 或其他组件 8 连通粘连。
# e) Visual observability
#    - 解题者能直观看到：所有真实 sprite 的外壳色一样，不可能再靠主体颜色配边；
#      真正对应关系来自 hub 边外的小色点和 sprite 上的同色点。
#    - 同时，那个第二色在输入里还出现在某个 bbox 角上，输出里则表现为“旋好后贴上去并保留内部装饰”，
#      足以让人从少量样例归纳“同色配对 + 角点定旋转 + 保留内层”。
# f) Visual cleanliness
#    - 每只 sprite 只有两种颜色，hub 外每条 active 边最多一个单像素 marker；
#      没有多层拥挤叠加，角色区分清楚。
# g) Change inferability
#    - 颜色角色受控：bg 1 种，hub 边 4 种，共享壳色 1 种，active connector/accent 最多 4 种。
#      第二色总是“hub marker 与某 sprite 内部结构成对呼应”，让颜色变化具备明确结构线索。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


BG = 8
HUB_N = 4
EDGE_LEN = HUB_N - 2
SIDES = ("top", "bot", "left", "right")


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_CANONICAL_TEMPLATES: tuple[tuple[np.ndarray, tuple[tuple[int, int], ...]], ...] = (
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


def _crop_by_mask(arr: np.ndarray, mask: np.ndarray, *extra_masks: np.ndarray):
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    arr2 = arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    mask2 = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    extra2 = tuple(m[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] for m in extra_masks)
    return (arr2, mask2, *extra2)


def _orient_from_left(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, -1)
    return np.rot90(arr, 1)


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


def _rotation_corner(arr: np.ndarray, conn_col: int) -> str | None:
    h, w = map(int, arr.shape)
    hits = []
    corners = {
        "tl": (0, 0),
        "tr": (0, int(w - 1)),
        "bl": (int(h - 1), 0),
        "br": (int(h - 1), int(w - 1)),
    }
    for name, (rr, cc) in corners.items():
        if int(arr[int(rr), int(cc)]) == int(conn_col):
            hits.append(name)
    if len(hits) != 1:
        return None
    return hits[0]


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


def _attachment_delta(core_mask: np.ndarray, side: str, hub_r: int, hub_c: int, n: int) -> tuple[int, int] | None:
    target = int(n - 2)
    segs = {
        "top": ("bot", int(hub_r - 1), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "bot": ("top", int(hub_r + n), list(range(int(hub_c + 1), int(hub_c + n - 1)))),
        "left": ("right", int(hub_c - 1), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
        "right": ("left", int(hub_c + n), list(range(int(hub_r + 1), int(hub_r + n - 1)))),
    }
    need_edge, fixed, span = segs[side]

    if need_edge in ("left", "right"):
        if not bool(np.array_equal(core_mask, np.flipud(core_mask))):
            return None
    else:
        if not bool(np.array_equal(core_mask, np.fliplr(core_mask))):
            return None

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


def _crop_core(arr: np.ndarray, core_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, int]:
    pts = np.argwhere(core_mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        core_mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        int(r0),
        int(c0),
    )


def _connector_slot_ok(arr: np.ndarray, core_mask: np.ndarray, side: str, conn_col: int, slot: int) -> bool:
    seg = _edge_segment(core_mask, _contact_side_for_hub(side))
    if seg is None:
        return False
    if side in ("top", "bot"):
        _typ, edge_r, cols = seg
        if len(cols) != EDGE_LEN or slot < 0 or slot >= len(cols):
            return False
        hits = [int(arr[int(edge_r), int(cc)]) == int(conn_col) for cc in cols]
        return sum(hits) == 1 and hits[int(slot)]
    _typ, edge_c, rows = seg
    if len(rows) != EDGE_LEN or slot < 0 or slot >= len(rows):
        return False
    hits = [int(arr[int(rr), int(edge_c)]) == int(conn_col) for rr in rows]
    return sum(hits) == 1 and hits[int(slot)]


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int):
    fg = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(fg, connectivity=8)
    out = []
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = g[comp_mask]
        arr, comp_mask = _crop_by_mask(arr, comp_mask)
        out.append((int(cid), arr, comp_mask))
    return out


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

    components = _extract_components(g, hub_mask, bg)
    sprites_by_conn: dict[int, list[tuple[int, np.ndarray, np.ndarray]]] = {}
    for cid, arr, comp_mask in components:
        if int(comp_mask.sum()) < 5:
            continue
        vals = [int(v) for v in arr[comp_mask]]
        counts = Counter(vals)
        if len(counts) != 2:
            continue
        for conn_col in counts:
            if _rotation_corner(arr, int(conn_col)) is None:
                continue
            sprites_by_conn.setdefault(int(conn_col), []).append((int(cid), arr, comp_mask))

    used_ids: set[int] = set()
    for side in SIDES:
        marker = side_markers.get(side)
        if marker is None:
            continue
        conn_col, slot = marker
        candidates = sprites_by_conn.get(int(conn_col), [])

        found = None
        for cid, arr, _comp_mask in candidates:
            if cid in used_ids:
                continue
            corner = _rotation_corner(arr, int(conn_col))
            krot = _turn_from_corner(side, corner)
            if krot is None:
                continue

            marker_mask = np.zeros(arr.shape, dtype=bool)
            if corner == "tl":
                marker_mask[0, 0] = True
            elif corner == "tr":
                marker_mask[0, -1] = True
            elif corner == "bl":
                marker_mask[-1, 0] = True
            else:
                marker_mask[-1, -1] = True

            rot_arr = np.rot90(arr, int(krot))
            rot_full = np.rot90(arr != int(bg), int(krot))
            rot_marker = np.rot90(marker_mask, int(krot))
            rot_arr, rot_full, rot_marker = _crop_by_mask(rot_arr, rot_full, rot_marker)

            core_mask = rot_full & (~rot_marker)
            if not bool(core_mask.any()):
                continue
            core_arr, core_local, r0, c0 = _crop_core(rot_arr, core_mask)
            local_delta = _attachment_delta(core_local, side, int(r), int(c), int(n))
            if local_delta is None:
                continue
            if not _connector_slot_ok(core_arr, core_local, side, int(conn_col), int(slot)):
                continue

            dr, dc = int(local_delta[0] - r0), int(local_delta[1] - c0)
            placed = {(int(rr + dr), int(cc2 + dc)) for rr, cc2 in np.argwhere(rot_full)}
            if any(not (0 <= rr < H and 0 <= cc2 < W) for rr, cc2 in placed):
                continue

            ok = True
            for rr, cc2 in placed:
                val = int(rot_arr[int(rr - dr), int(cc2 - dc)])
                if val == int(bg):
                    continue
                cur = int(out[int(rr), int(cc2)])
                if cur != int(bg) and cur != val:
                    ok = False
                    break
            if not ok:
                continue
            found = (int(cid), rot_arr, rot_full, int(dr), int(dc))
            break

        assert found is not None
        cid, rot_arr, rot_full, dr, dc = found
        used_ids.add(int(cid))
        paste(out, rot_arr, int(dr), int(dc), mask=rot_full, overwrite=True, clip=True)

    return out


def _make_core_sprite(side: str, shell_col: int, conn_col: int, slot: int, bg: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    template_mask, inner_cells = rng.choice(_CANONICAL_TEMPLATES)
    left_arr = np.full(template_mask.shape, int(bg), dtype=int)
    left_arr[template_mask] = int(shell_col)

    inner_choices = [
        (inner_cells[0],),
        (inner_cells[1],),
        inner_cells,
    ]
    for rr, cc in rng.choice(inner_choices):
        left_arr[int(rr), int(cc)] = int(conn_col)

    arr = _orient_from_left(left_arr, side)
    mask = _orient_from_left(template_mask, side)
    arr, mask = _crop_by_mask(arr, mask)

    seg = _edge_segment(mask, _contact_side_for_hub(side))
    assert seg is not None
    if side in ("top", "bot"):
        _typ, edge_r, cols = seg
        assert len(cols) == EDGE_LEN and slot in (0, 1)
        arr[int(edge_r), int(cols[int(slot)])] = int(conn_col)
    else:
        _typ, edge_c, rows = seg
        assert len(rows) == EDGE_LEN and slot in (0, 1)
        arr[int(rows[int(slot)]), int(edge_c)] = int(conn_col)
    return arr, mask


def _embed_with_corner(
    core_arr: np.ndarray,
    core_mask: np.ndarray,
    anchor: str,
    conn_col: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = map(int, core_arr.shape)
    arr = np.full((int(h + 1), int(w + 1)), int(bg), dtype=int)
    core = np.zeros((int(h + 1), int(w + 1)), dtype=bool)
    marker = np.zeros((int(h + 1), int(w + 1)), dtype=bool)

    if anchor == "tl":
        arr[1:, 1:] = core_arr
        core[1:, 1:] = core_mask
        arr[0, 0] = int(conn_col)
        marker[0, 0] = True
    elif anchor == "tr":
        arr[1:, :w] = core_arr
        core[1:, :w] = core_mask
        arr[0, int(w)] = int(conn_col)
        marker[0, int(w)] = True
    elif anchor == "bl":
        arr[:h, 1:] = core_arr
        core[:h, 1:] = core_mask
        arr[int(h), 0] = int(conn_col)
        marker[int(h), 0] = True
    else:
        arr[:h, :w] = core_arr
        core[:h, :w] = core_mask
        arr[int(h), int(w)] = int(conn_col)
        marker[int(h), int(w)] = True

    full = arr != int(bg)
    arr, full, core, marker = _crop_by_mask(arr, full, core, marker)
    return arr, core, marker


def _build_sprite_pair(
    side: str,
    turn_dir: str,
    shell_col: int,
    conn_col: int,
    slot: int,
    bg: int,
    rng: random.Random,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    core_arr, core_mask = _make_core_sprite(side, int(shell_col), int(conn_col), int(slot), int(bg), rng)
    krot = -1 if turn_dir == "cw" else 1

    out = []
    for anchor in ("tl", "tr", "bl", "br"):
        attached_arr, attached_core, attached_marker = _embed_with_corner(core_arr, core_mask, anchor, int(conn_col), int(bg))
        scat_arr = np.rot90(attached_arr, -int(krot))
        scat_full = np.rot90(attached_arr != int(bg), -int(krot))
        scat_arr, scat_full = _crop_by_mask(scat_arr, scat_full)
        corner = _rotation_corner(scat_arr, int(conn_col))
        if _turn_from_corner(side, corner) == int(krot):
            out.append((scat_arr, attached_arr, attached_core, attached_marker))
    return out


def _place_attached_sprite(
    out: np.ndarray,
    arr: np.ndarray,
    core_mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    n: int,
    bg: int,
) -> tuple[int, int] | None:
    _core_arr, core_local, r0, c0 = _crop_core(arr, core_mask)
    delta = _attachment_delta(core_local, side, int(hub_r), int(hub_c), int(n))
    if delta is None:
        return None
    dr, dc = int(delta[0] - r0), int(delta[1] - c0)
    H, W = out.shape
    full_mask = arr != int(bg)
    placed = {(int(rr + dr), int(cc + dc)) for rr, cc in np.argwhere(full_mask)}
    if any(not (0 <= rr < H and 0 <= cc < W) for rr, cc in placed):
        return None
    if any(int(hub_r) <= rr < int(hub_r + n) and int(hub_c) <= cc < int(hub_c + n) for rr, cc in placed):
        return None
    for rr, cc in placed:
        val = int(arr[int(rr - dr), int(cc - dc)])
        if val == int(bg):
            continue
        if int(out[int(rr), int(cc)]) != int(bg):
            return None
    paste(out, arr, int(dr), int(dc), transparent=int(bg))
    return int(dr), int(dc)


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG
    H = rng.randint(22, 28)
    W = rng.randint(22, 28)

    colors = [c for c in range(10) if c != int(bg)]
    rng.shuffle(colors)
    top_col, bot_col, left_col, right_col, shell_col = map(int, colors[:5])
    side_colors = {"top": top_col, "bot": bot_col, "left": left_col, "right": right_col}
    connector_pool = list(map(int, colors[5:]))

    k = rng.randint(2, 4)
    sprite_sides = rng.sample(list(SIDES), int(k))
    if len(connector_pool) < len(sprite_sides):
        return None

    side_conn = {side: int(connector_pool[i]) for i, side in enumerate(sprite_sides)}
    side_slot = {side: rng.randint(0, 1) for side in sprite_sides}
    side_turn = {side: rng.choice(["cw", "ccw"]) for side in sprite_sides}
    if len(sprite_sides) >= 2 and len({side_turn[s] for s in sprite_sides}) == 1:
        flip_side = rng.choice(sprite_sides)
        side_turn[flip_side] = "ccw" if side_turn[flip_side] == "cw" else "cw"

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        candidates = _build_sprite_pair(
            side,
            side_turn[side],
            int(shell_col),
            int(side_conn[side]),
            int(side_slot[side]),
            int(bg),
            rng,
        )
        rng.shuffle(candidates)
        if not candidates:
            return None
        chosen = None
        for scattered_arr, attached_arr, attached_core, attached_marker in candidates:
            core_arr, core_local, _r0, _c0 = _crop_core(attached_arr, attached_core)
            core_ok = _attachment_delta(core_local, side, 10, 10, HUB_N)
            if core_ok is None:
                continue
            if not _connector_slot_ok(core_arr, core_local, side, int(side_conn[side]), int(side_slot[side])):
                continue
            if int(attached_marker.sum()) != 1:
                continue
            chosen = (scattered_arr, attached_arr, attached_core)
            break
        if chosen is None:
            return None
        sprite_defs[side] = chosen

    max_h = max(max(att.shape[0], scat.shape[0]) for scat, att, _core in sprite_defs.values())
    max_w = max(max(att.shape[1], scat.shape[1]) for scat, att, _core in sprite_defs.values())
    margin = max(int(max_h), int(max_w)) + 2

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

    attached_defs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        _scattered_arr, attached_arr, attached_core = sprite_defs[side]
        placed = _place_attached_sprite(out, attached_arr, attached_core, side, int(hub_r), int(hub_c), HUB_N, int(bg))
        if placed is None:
            return None
        attached_defs[side] = (attached_arr, attached_core)

    inp = np.full((H, W), int(bg), dtype=int)
    inp[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N] = out[hub_r : hub_r + HUB_N, hub_c : hub_c + HUB_N]
    for side in sprite_sides:
        rr, cc = _hub_marker_candidates(int(hub_r), int(hub_c), HUB_N, side)[int(side_slot[side])]
        if not (0 <= rr < H and 0 <= cc < W):
            return None
        if int(inp[int(rr), int(cc)]) != int(bg):
            return None
        inp[int(rr), int(cc)] = int(side_conn[side])

    canvas = inp.copy()
    for side in sprite_sides:
        scattered_arr, _attached_arr, _attached_core = sprite_defs[side]
        try:
            pr, pc = random_free_location(canvas, scattered_arr, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, scattered_arr, int(pr), int(pc), transparent=int(bg))
        paste(inp, scattered_arr, int(pr), int(pc), transparent=int(bg))

    unused = [col for col in connector_pool if int(col) not in {int(side_conn[s]) for s in sprite_sides}]
    if unused and rng.random() < 0.3:
        d_side = rng.choice(SIDES)
        d_slot = rng.randint(0, 1)
        d_turn = rng.choice(["cw", "ccw"])
        candidates = _build_sprite_pair(d_side, d_turn, int(shell_col), int(unused[0]), int(d_slot), int(bg), rng)
        if candidates:
            dist_arr = rng.choice(candidates)[0]
            try:
                pr, pc = random_free_location(canvas, dist_arr, transparent=int(bg), padding=2, padding_connectivity=8)
                paste(canvas, dist_arr, int(pr), int(pc), transparent=int(bg))
                paste(inp, dist_arr, int(pr), int(pc), transparent=int(bg))
            except ValueError:
                pass

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
