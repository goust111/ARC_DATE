# Combo variant of 2c181942 (index 6)
# combo_spec: D1/4+D4/0
# Name: 渐变序列匹配+多色复合精灵
"""
design_analysis:
  fused_rule:
    summary: |
      hub 仍是中心空心方块，但固定为 N=5，让每条边都有 3 个可见像素，从而能显示
      一个有方向的颜色序列。每个 sprite 仍满足原题的接触边长度与轴对称几何约束，
      但不再是单色，而是“多色外层壳 + 单色内层装饰”的复合对象。
      solver 需要比较 sprite 接触边上的有序外层颜色序列，与 hub 边缘序列完全一致时才配对；
      一旦配对成功，整只 sprite 都被旋转并贴附，内层装饰色保留。
  concretization_and_trace:
    concrete_example: |
      例如 hub top 边是 [2,4,7]，right 边是 [4,7,3]，bot 边是 [7,3,2]，left 边是 [3,2,4]。
      某个散落 sprite 本体是 8 连通小物体，外壳三条带的颜色依次为 [2,4,7]，内部再放 1-2 个颜色 9 的装饰像素。
      原题 solver 会按单色 2/4/7 去找对象，已经失效，因为一个 sprite 同时含有多种外层颜色和内层颜色；
      新 solver 必须先把 hub 外所有非背景像素按 8 连通提整只对象，再枚举旋转，检查：
      1) 该旋转是否满足原题接触边长度与镜像对称；
      2) 真正接触 hub 的那一侧，沿边读取到的颜色序列是否正好等于对应 hub 边。
      若等于 [2,4,7] 就贴到 top；即使 sprite 中也含有 9，这些 9 只是装饰，会原样保留。
    changed_execution_points:
      - "hub 检测从“每边单色”改成“每边是非背景序列，并记录整条序列”。"
      - "候选 sprite 提取从“按某个颜色筛最大连通块”改成“提所有前景连通块”。"
      - "匹配条件从“含某个边色”改成“某个旋转下的接触边序列完全匹配”。"
      - "贴附对象从“单色 mask”改成“整只多色对象”，因此内层颜色一起移动。"
  complexity_calibration:
    choices: |
      我把 hub 固定到 N=5，而不是沿用原题常见的 N=4。原因是 D1/4 的本质是“顺序匹配”，
      若边长只有 2，视觉上很难和“只是两个不同颜色”区分，也不利于 2-4 个样例中形成稳定归纳。
      长度 3 足够表达方向，同时仍便于放置和保持 generator 成功率。
    ambiguity_control: |
      四条 hub 边序列使用 4 色循环窗口：
      top=[a,b,c], right=[b,c,d], bot=[c,d,a], left=[d,a,b]。
      这样单个颜色会在多条边重复，旧版“看某个颜色就知道贴哪边”的逻辑彻底失效；
      但完整序列仍互异，且不会与其它边的逆序重复，从而避免旋转方向歧义。
      sprite 模板只从少量手工不对称 mask 中挑选，并显式验证“对目标 side 只有 1 个旋转合法”。
  implicit_assumption_audit:
    original_main_breakages:
      - "_find_hub 中 `len(set(top)) == 1` 等单色前提在融合后不成立，必须改为提取完整序列。"
      - "`m = (g == col) & (~hub_mask)` 假设整只 sprite 只有一个颜色；融合后会把对象拆碎。"
      - "原逻辑通过颜色唯一决定 side；融合后同一个颜色会出现在多条边，只能靠完整顺序。"
      - "原 paste/冲突检测默认粘贴单色像素；融合后必须对整只多色数组做一致性检查。"
    new_main_assumptions:
      - "hub 固定 N=5，四边都是长度 3 的非背景序列。generator 保证。"
      - "每个可匹配 sprite 在 hub 外是单个 8 连通前景组件。generator 用 padding=2 保证互不粘连。"
      - "真正用于匹配的是接触边上的三格外层色序列；内层装饰不会落在接触边上。generator 保证。"
      - "不同 side 的序列唯一且不与彼此逆序相同；否则序列匹配可能退化。generator 保证。"
      - "对于每个有效 sprite，目标 side 的合法旋转唯一；generator 在生成时验证。"
  placement_legality:
    output_first: |
      generator 仍采用 output-first。先把 hub 和已定向 sprite 在输出图中贴好，检查不越界、不重叠；
      再把 sprite 拆下、随机旋转到非贴附朝向，散放到输入。
    spacing: |
      hub 周围保留至少 `max(sprite_extent)+2` 的 margin；散放使用 random_free_location，transparent=bg，
      padding=2，padding_connectivity=8，保证 sprite 之间以及 sprite 与 hub 不会接触，solver 才能稳定按组件提取。
    distractors: |
      干扰物也做成复合多色 sprite，但它的接触边序列不会等于任何 hub 边序列，也不会等于任何 hub 边序列的逆序。
  visual_observability:
    why_human_can_infer: |
      从样例上，人类能直接看到两条强线索：
      1) hub 四边不再单色，而是有顺序的三色边；
      2) sprite 贴附后保留了内部装饰色，说明不是只拿“同色部分”去贴。
      又因为同一个颜色会在多条 hub 边重复出现，观察者不能靠单个颜色做匹配，只能靠整段顺序。
      内层色在不同 sprite 上可复用，但始终不控制贴附 side，这会重复支撑“外层匹配、内层保留”的规则。
  visual_cleanliness:
    layout: |
      仍保持原题的“中心 hub + 四周散落 2-4 个对象 + 可选 1 个干扰物”的简洁版式。
      不引入额外标记、不叠加文字；对象数量和颜色池受控，避免画面混乱。
  inference_safety:
    no_hidden_ordering: |
      颜色顺序完全由空间位置定义：top/bot 按左到右，left/right 按上到下。
      实现中不依赖颜色编号大小规律等人类不可见因素。
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


N = 5
EDGE_LEN = N - 2
SIDES = ("top", "right", "bot", "left")


def _rows_to_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


# Canonical "left-attach" shapes:
# - contact edge is the rightmost column
# - mask is flipud-symmetric
# - exactly EDGE_LEN contiguous cells touch the contact edge
_CANONICAL_LEFT_SPRITES: tuple[tuple[np.ndarray, tuple[tuple[tuple[int, int], ...], ...]], ...] = (
    (
        _rows_to_mask(
            [
                "011",
                "111",
                "011",
            ]
        ),
        (((1, 1),), ((0, 1), (2, 1))),
    ),
    (
        _rows_to_mask(
            [
                "0011",
                "1111",
                "0011",
            ]
        ),
        (((1, 1),), ((1, 2),)),
    ),
    (
        _rows_to_mask(
            [
                "0111",
                "0011",
                "0111",
            ]
        ),
        (((1, 2),), ((0, 1), (2, 1))),
    ),
    (
        _rows_to_mask(
            [
                "111",
                "011",
                "111",
            ]
        ),
        (((1, 1),), ((0, 0), (2, 0))),
    ),
)


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return (
        arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
        mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1],
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    arr = np.where(mask, 1, 0).astype(int)
    _, cropped_mask = _crop_to_mask(arr, mask.astype(bool))
    return cropped_mask


def _orient_from_left(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "left":
        return arr.copy()
    if side == "right":
        return np.fliplr(arr)
    if side == "top":
        return np.rot90(arr, -1)
    if side == "bot":
        return np.rot90(arr, 1)
    raise ValueError(side)


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


def _edge_signature(arr: np.ndarray, mask: np.ndarray, side: str) -> tuple[int, ...] | None:
    seg = _edge_segment(mask, side)
    if seg is None:
        return None
    if side in ("top", "bot"):
        _typ, rr, cols = seg
        if len(cols) != EDGE_LEN or not _is_contig(cols):
            return None
        return tuple(int(arr[int(rr), int(cc)]) for cc in cols)
    _typ, cc, rows = seg
    if len(rows) != EDGE_LEN or not _is_contig(rows):
        return None
    return tuple(int(arr[int(rr), int(cc)]) for rr in rows)


def _mask_matches_attachment(mask: np.ndarray, need_edge: str, target_len: int) -> bool:
    if need_edge in ("left", "right"):
        if not bool(np.array_equal(mask, np.flipud(mask))):
            return False
    else:
        if not bool(np.array_equal(mask, np.fliplr(mask))):
            return False

    seg = _edge_segment(mask, need_edge)
    if seg is None:
        return False
    if need_edge in ("left", "right"):
        _typ, _cc, rows = seg
        return len(rows) == int(target_len) and _is_contig(rows)
    _typ, _rr, cols = seg
    return len(cols) == int(target_len) and _is_contig(cols)


def _find_hub(g: np.ndarray, bg: int):
    H, W = g.shape
    best = None
    for r in range(0, int(H - N + 1)):
        for c in range(0, int(W - N + 1)):
            if (
                int(g[int(r), int(c)]) != int(bg)
                or int(g[int(r), int(c + N - 1)]) != int(bg)
                or int(g[int(r + N - 1), int(c)]) != int(bg)
                or int(g[int(r + N - 1), int(c + N - 1)]) != int(bg)
            ):
                continue

            inner = g[int(r + 1) : int(r + N - 1), int(c + 1) : int(c + N - 1)]
            if not bool((inner == int(bg)).all()):
                continue

            side_seqs = {
                "top": tuple(int(x) for x in g[int(r), int(c + 1) : int(c + N - 1)]),
                "right": tuple(int(x) for x in g[int(r + 1) : int(r + N - 1), int(c + N - 1)]),
                "bot": tuple(int(x) for x in g[int(r + N - 1), int(c + 1) : int(c + N - 1)]),
                "left": tuple(int(x) for x in g[int(r + 1) : int(r + N - 1), int(c)]),
            }
            seqs = list(side_seqs.values())
            if any(int(bg) in seq for seq in seqs):
                continue
            if any(len(seq) != EDGE_LEN for seq in seqs):
                continue
            if any(len(set(seq)) < 2 for seq in seqs):
                continue
            if len(set(seqs)) != 4:
                continue
            if any(seq[::-1] in {other for other in seqs if other != seq} for seq in seqs):
                continue

            cr = float(r + (N - 1) / 2.0)
            cc = float(c + (N - 1) / 2.0)
            score = abs(float(cr - (H - 1) / 2.0)) + abs(float(cc - (W - 1) / 2.0))
            cand = (float(score), (int(r), int(c), int(N)), side_seqs)
            if best is None or float(cand[0]) < float(best[0]):
                best = cand
    return best


def _extract_components(g: np.ndarray, hub_mask: np.ndarray, bg: int) -> list[tuple[int, np.ndarray, np.ndarray]]:
    fg = (g != int(bg)) & (~hub_mask)
    lab, nc = label_mask(fg, connectivity=8)
    out = []
    for cid in range(1, int(nc) + 1):
        comp_mask = lab == int(cid)
        if not bool(comp_mask.any()):
            continue
        arr = np.full_like(g, int(bg))
        arr[comp_mask] = g[comp_mask]
        arr, comp_mask = _crop_to_mask(arr, comp_mask)
        out.append((int(cid), arr, comp_mask))
    return out


def _placeable_rotation(
    arr: np.ndarray,
    mask: np.ndarray,
    need_edge: str,
    fixed: int,
    span: list[int],
    target_seq: tuple[int, ...],
    out: np.ndarray,
    bg: int,
):
    H, W = out.shape
    for krot in range(4):
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)

        if not _mask_matches_attachment(rot_mask, need_edge, EDGE_LEN):
            continue
        if _edge_signature(rot_arr, rot_mask, need_edge) != target_seq:
            continue

        seg = _edge_segment(rot_mask, need_edge)
        assert seg is not None
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
            if int(out[int(rr), int(cc)]) != int(bg) and int(out[int(rr), int(cc)]) != val:
                ok = False
                break
        if ok:
            return rot_arr, rot_mask, int(dr), int(dc)
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Attach composite multi-color sprites to the hub side whose ordered contact-edge sequence they match.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None

    _score, (r, c, n), hub_sides = hub
    assert int(n) == int(N)

    hub_mask = np.zeros((H, W), dtype=bool)
    hub_mask[int(r) : int(r + n), int(c) : int(c + n)] = True

    out = np.full_like(g, int(bg))
    out[int(r) : int(r + n), int(c) : int(c + n)] = g[int(r) : int(r + n), int(c) : int(c + n)]

    components = _extract_components(g, hub_mask, bg)
    segs = {
        "top": ("bot", int(r - 1), list(range(int(c + 1), int(c + n - 1)))),
        "bot": ("top", int(r + n), list(range(int(c + 1), int(c + n - 1)))),
        "left": ("right", int(c - 1), list(range(int(r + 1), int(r + n - 1)))),
        "right": ("left", int(c + n), list(range(int(r + 1), int(r + n - 1)))),
    }

    used_ids: set[int] = set()
    for side in ("top", "bot", "left", "right"):
        target_seq = tuple(int(x) for x in hub_sides[side])
        need_edge, fixed, span = segs[side]

        found = None
        for cid, arr, mask in components:
            if cid in used_ids:
                continue
            if len({int(v) for v in arr[mask]}) < 2:
                continue
            placed = _placeable_rotation(arr, mask, need_edge, fixed, span, target_seq, out, bg)
            if placed is None:
                continue
            found = (cid, placed)
            break

        if found is None:
            continue

        cid, (arr2, mask2, dr, dc) = found
        used_ids.add(int(cid))
        paste(out, arr2, int(dr), int(dc), mask=mask2, overwrite=True, clip=True)

    return out


def _build_side_sequences(rng: random.Random, bg: int) -> dict[str, tuple[int, int, int]]:
    palette = rng.sample([c for c in range(10) if c != int(bg)], 4)
    a, b, c, d = map(int, palette)
    return {
        "top": (a, b, c),
        "right": (b, c, d),
        "bot": (c, d, a),
        "left": (d, a, b),
    }


def _colorize_shell(mask: np.ndarray, side: str, seq: tuple[int, int, int], bg: int) -> np.ndarray:
    arr = np.full(mask.shape, int(bg), dtype=int)
    if side in ("left", "right"):
        active_rows = [int(rr) for rr in range(mask.shape[0]) if bool(mask[int(rr)].any())]
        if len(active_rows) != len(seq):
            raise ValueError("row stripe count mismatch")
        for idx, rr in enumerate(active_rows):
            arr[int(rr), mask[int(rr)]] = int(seq[int(idx)])
        return arr

    active_cols = [int(cc) for cc in range(mask.shape[1]) if bool(mask[:, int(cc)].any())]
    if len(active_cols) != len(seq):
        raise ValueError("col stripe count mismatch")
    for idx, cc in enumerate(active_cols):
        arr[mask[:, int(cc)], int(cc)] = int(seq[int(idx)])
    return arr


def _make_composite_sprite(
    side: str,
    seq: tuple[int, int, int],
    accent_color: int,
    rng: random.Random,
    bg: int,
) -> tuple[np.ndarray, np.ndarray] | None:
    candidates: list[tuple[np.ndarray, np.ndarray]] = []
    for left_mask_raw, accent_groups in _CANONICAL_LEFT_SPRITES:
        left_mask = _crop_mask(left_mask_raw)
        left_shell = _colorize_shell(left_mask, "left", seq, bg)

        for accent_cells in accent_groups:
            accent_mask = np.zeros_like(left_mask, dtype=bool)
            for rr, cc in accent_cells:
                if not bool(left_mask[int(rr), int(cc)]):
                    accent_mask = None
                    break
                if int(cc) == int(left_mask.shape[1] - 1):
                    accent_mask = None
                    break
                accent_mask[int(rr), int(cc)] = True
            if accent_mask is None or not bool(accent_mask.any()):
                continue

            side_mask = _orient_from_left(left_mask.astype(int), side).astype(bool)
            side_shell = _orient_from_left(left_shell, side)
            side_accent = _orient_from_left(accent_mask.astype(int), side).astype(bool)

            arr = side_shell.copy()
            arr[side_accent] = int(accent_color)
            arr = np.where(side_mask, arr, int(bg)).astype(int)
            arr, side_mask = _crop_to_mask(arr, side_mask)
            candidates.append((arr, side_mask))

    if not candidates:
        return None
    return rng.choice(candidates)


def _valid_rotation_count(arr: np.ndarray, mask: np.ndarray, need_edge: str, target_seq: tuple[int, ...]) -> int:
    count = 0
    for krot in range(4):
        rot_arr = np.rot90(arr, int(krot))
        rot_mask = np.rot90(mask, int(krot))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        if not _mask_matches_attachment(rot_mask, need_edge, EDGE_LEN):
            continue
        if _edge_signature(rot_arr, rot_mask, need_edge) == target_seq:
            count += 1
    return count


def _attachment_position(mask: np.ndarray, side: str, hub_r: int, hub_c: int) -> tuple[int, int]:
    sh, sw = mask.shape
    if side == "top":
        filled = np.where(mask[-1])[0]
        return int(hub_r - sh), int(hub_c + 1 - int(filled[0]))
    if side == "bot":
        filled = np.where(mask[0])[0]
        return int(hub_r + N), int(hub_c + 1 - int(filled[0]))
    if side == "left":
        filled = np.where(mask[:, -1])[0]
        return int(hub_r + 1 - int(filled[0])), int(hub_c - sw)
    filled = np.where(mask[:, 0])[0]
    return int(hub_r + 1 - int(filled[0])), int(hub_c + N)


def _place_hub(out: np.ndarray, hub_r: int, hub_c: int, side_seqs: dict[str, tuple[int, int, int]], bg: int) -> None:
    out[int(hub_r) : int(hub_r + N), int(hub_c) : int(hub_c + N)] = int(bg)
    for idx, col in enumerate(side_seqs["top"], start=1):
        out[int(hub_r), int(hub_c + idx)] = int(col)
    for idx, col in enumerate(side_seqs["bot"], start=1):
        out[int(hub_r + N - 1), int(hub_c + idx)] = int(col)
    for idx, col in enumerate(side_seqs["left"], start=1):
        out[int(hub_r + idx), int(hub_c)] = int(col)
    for idx, col in enumerate(side_seqs["right"], start=1):
        out[int(hub_r + idx), int(hub_c + N - 1)] = int(col)


def _make_distractor_sequence(
    rng: random.Random,
    bg: int,
    forbidden: set[tuple[int, int, int]],
) -> tuple[int, int, int] | None:
    colors = [int(c) for c in range(10) if c != int(bg)]
    for _ in range(200):
        seq = tuple(int(x) for x in rng.sample(colors, EDGE_LEN))
        if len(set(seq)) < 2:
            continue
        if seq in forbidden or seq[::-1] in forbidden:
            continue
        return seq
    return None


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 8
    H = rng.randint(18, 26)
    W = rng.randint(20, 26)

    side_seqs = _build_side_sequences(rng, bg)
    sides = ["top", "bot", "left", "right"]
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    hub_colors = {int(c) for seq in side_seqs.values() for c in seq}
    accent_palette = [int(c) for c in range(10) if c != int(bg) and c not in hub_colors]
    if len(accent_palette) < 2:
        return None

    sprite_defs: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for side in sprite_sides:
        accent_color = int(rng.choice(accent_palette))
        sprite = _make_composite_sprite(side, side_seqs[side], accent_color, rng, bg)
        if sprite is None:
            return None
        arr, mask = sprite
        if len({int(v) for v in arr[mask]}) < 2:
            return None
        need_edge = {"top": "bot", "bot": "top", "left": "right", "right": "left"}[side]
        if _valid_rotation_count(arr, mask, need_edge, side_seqs[side]) != 1:
            return None
        sprite_defs[side] = (arr, mask)

    max_sprite_h = max(arr.shape[0] for arr, _ in sprite_defs.values())
    max_sprite_w = max(arr.shape[1] for arr, _ in sprite_defs.values())
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
    _place_hub(out, hub_r, hub_c, side_seqs, bg)

    attached_sprites: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        arr, mask = sprite_defs[side]
        sr, sc = _attachment_position(mask, side, hub_r, hub_c)
        sh, sw = mask.shape
        if sr < 0 or sc < 0 or sr + sh > H or sc + sw > W:
            return None
        for rr in range(sh):
            for cc in range(sw):
                if bool(mask[int(rr), int(cc)]) and int(out[int(sr + rr), int(sc + cc)]) != int(bg):
                    return None
        paste(out, arr, int(sr), int(sc), transparent=int(bg))
        attached_sprites[side] = (arr, mask, int(sr), int(sc))

    inp = np.full((H, W), int(bg), dtype=int)
    inp[int(hub_r) : int(hub_r + N), int(hub_c) : int(hub_c + N)] = out[int(hub_r) : int(hub_r + N), int(hub_c) : int(hub_c + N)]
    canvas = inp.copy()

    for side in sprite_sides:
        arr, mask, _sr, _sc = attached_sprites[side]
        k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, int(k))
        rot_mask = np.rot90(mask, int(k))
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, rot_arr, int(bg)).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
        paste(inp, patch, int(pr), int(pc), transparent=int(bg))

    if rng.random() < 0.3:
        forbidden = set(side_seqs.values())
        dist_seq = _make_distractor_sequence(rng, bg, forbidden)
        if dist_seq is not None:
            dist_side = rng.choice(sides)
            dist_accent = int(rng.choice(accent_palette))
            distractor = _make_composite_sprite(dist_side, dist_seq, dist_accent, rng, bg)
            if distractor is not None:
                dist_arr, dist_mask = distractor
                k = rng.choice([0, 1, 2, 3])
                dist_arr = np.rot90(dist_arr, int(k))
                dist_mask = np.rot90(dist_mask, int(k))
                dist_arr, dist_mask = _crop_to_mask(dist_arr, dist_mask)
                patch = np.where(dist_mask, dist_arr, int(bg)).astype(int)
                try:
                    pr, pc = random_free_location(canvas, patch, transparent=int(bg), padding=2, padding_connectivity=8)
                    paste(canvas, patch, int(pr), int(pc), transparent=int(bg))
                    paste(inp, patch, int(pr), int(pc), transparent=int(bg))
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
