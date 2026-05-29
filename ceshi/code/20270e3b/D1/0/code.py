# Variant of 20270e3b
# Dimension: D1 — 连接器匹配条件 (match_cond)
# Change: 两个精灵的连接器具有相同的归一化形状（相同长度的ctrl色水平条），按形状/颜色匹配... → 颜色匹配而非形状匹配

"""
design_analysis:
  variant_summary:
    kept:
      - 仍然只有两个 8 连通精灵：base 固定在输入左上角，plug 散落在别处。
      - 输出仍然是把 plug 平移到 base 上方一行进行对接，然后把连接器重涂成 body 色并紧裁剪。
      - 连接器仍然是单行水平条，base 的自由侧朝下，plug 的自由侧朝上。
    changed:
      - 原题要求两个连接器的归一化形状相同，本变体去掉该条件。
      - 两个连接器现在只要求“同色”，长度允许不同，并在生成器中强制大多数情况下都不同；这里直接强制 100% 不同。
      - 连接器颜色不再固定为 7，而是在若干非 bg / 非 body 颜色中随机选一个，使“颜色”成为显式可见的匹配线索。
      - 对接时不再依赖“两个条完全同长”，而是采用“左端点对齐、plug 连接器位于 base 连接器正上方一行”的几何。

  a_concretization_and_trace:
    input_look:
      - bg 固定为 1，body 固定为 4。
      - 两个精灵都含有一条水平连接器，连接器颜色是同一个随机颜色 ctrl in {2,3,5,6,7,8,9}。
      - base 连接器长度 1-3，plug 连接器长度也 1-3，但两者必不相等。
    output_look:
      - plug 被平移到 base 上方，plug 连接器整条位于 base 连接器上方一格。
      - plug 的第一行 body 会压到 base 的连接器上；所有 ctrl 最终都被改成 obj=4。
    concrete_trace:
      - 例：base 连接器是红色长度 3，位置在第 4 行第 2-4 列；plug 连接器也是红色但长度 1，位于 plug 顶行第 0 列。
      - 原题逻辑会先归一化两个连接器形状，得到 {0,1,2} vs {0}，随即断言失败。
      - 本变体逻辑先识别“唯一的非 bg / 非 obj 颜色”为 ctrl，再分别提取两个精灵中的 ctrl 水平条。
      - 平移量改为按左端点对齐：plug_conn_left 对齐到 base_conn_left，plug_conn_row 放到 base_conn_row-1。
      - 结果中，base 原来长度 3 的连接器有 1 格被 plug 的 body 压住，剩余 2 格虽未被覆盖，也会在最后统一改成 body 色。
      - 差异点就在“匹配条件”：原题依赖几何同形；变体依赖颜色，并允许几何不一致。

  b_complexity_calibration:
    - 如果只把长度放开但颜色仍固定为 7，视觉上会像“只是长度变了”，颜色线索太弱。
    - 因此 ctrl 颜色在多个备选颜色里随机，使解题者从 2-4 组样例里能看到：需要识别的是“那条特殊颜色的水平条”，而不是固定长度。
    - 仍然只保留 1 个连接器颜色，避免再引入多连接器 / 多配对分支，把复杂度控制在原题附近。
    - 长度只允许 1-3 三档，且样本中始终 base_len != plug_len，变化足够明显但分支很少。

  c_hidden_assumption_review_main:
    - `ctrl = 7`：旧前提是连接器颜色固定；变体中失效，必须从网格中推断唯一的额外颜色。
    - `shpA == shpB`：旧前提是两个连接器同形；变体中显式失效，必须删除。
    - `norm(conn)`：旧代码既用它检查形状，也用它拿左上锚点。变体中“形状比较”删除，但“取左端点作为对接锚点”仍然成立。
    - `cur == ctrl or val == ctrl` 的重叠断言仍然成立，但 `ctrl` 现在是推断值而不是常数。
    - `out[out == ctrl] = obj` 仍然成立，因为变体没有改变“连接器消失变成 body 色”。
    - `assert len(best_comps) == 2` 保留；生成器继续保证恰好两块主精灵，没有额外碎片。

  d_placement_legality:
    - base 仍然放在输入画布左上角。
    - plug 在输入中必须与 base 8-邻域完全分离，否则 solver 会把二者识别成同一连通块；生成器按像素检查，禁止任何 8 邻接。
    - docked 状态下只允许一种重叠：plug 的第一行 body 覆盖到 base 的连接器像素上。
    - 若 plug 更长，它多出来的 body 只能压在背景上，不能压到 base 的普通 body；若 base 更长，未被覆盖的 base 连接器在最后重涂为 body。
    - plug 连接器的左端局部坐标必须不大于 base 连接器左端列，这样 dock 后不会伸到 base 左边，保证紧裁剪后 base 仍在输出左上角。

  e_visual_observability:
    - 每个样例都强制 base_len != plug_len，因此“旧规则要求同长”会被明显打破。
    - 同时 ctrl 颜色跨样例变化，解题者能看到：真正稳定的是“那条特殊颜色条”的语义，而不是长度 1 / 2 / 3 的某个固定形状。
    - 对接方向仍然单一固定为竖直接缝，避免把观察难度转移到旋转/镜像上。

  f_visual_cleanliness:
    - 只使用 3 种可见角色：背景、body、连接器，不叠加额外标记。
    - plug 的最顶行除了连接器外不允许再长出 body，避免顶部边界混乱。
    - 输入放置要求 8-邻域隔离、bbox gap 1-5，既不挤在一起，也不会相距太远。

  g_color_inference:
    - 颜色变化只涉及“哪一种额外颜色扮演连接器”，不是任意映射。
    - body 始终为 4，背景始终为 1，解题者只需识别唯一的第三种颜色即可。
    - 这种颜色在两个精灵上重复出现，结构角色完全一致，足以支撑少样本推断。

  h_new_main_assumptions_and_generator_guarantees:
    - main 假设网格里恰有一种非 bg / 非 obj 的颜色；generator 保证只引入一个 ctrl 颜色。
    - main 假设每个精灵里的 ctrl 像素组成单行连续水平条；generator 显式检查。
    - main 假设 base 连接器上方是 body、下方无 body，plug 连接器下方是 body、上方无 body；generator 按候选位置和保留像素约束保证。
    - main 假设左端点对齐后，plug 不会越过 base 的上边/左边且不会与 base 普通 body 非法重叠；generator 在 docked 坐标系里逐像素检查。
"""

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _bar_info(conn_pts: set[tuple[int, int]]) -> tuple[int, int, int, list[int]]:
    assert conn_pts
    rows = sorted({int(r) for r, _c in conn_pts})
    assert len(rows) == 1
    cols = sorted(int(c) for _r, c in conn_pts)
    assert all(int(cols[i]) + 1 == int(cols[i + 1]) for i in range(len(cols) - 1))
    return int(rows[0]), int(cols[0]), int(len(cols)), cols


def _crop_non_bg(arr: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(arr != int(bg))
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _assemble_output(
    base_grid: np.ndarray,
    plug_grid: np.ndarray,
    *,
    ctrl: int,
    obj: int,
    bg: int,
    dock_r0: int,
    dock_c0: int,
) -> np.ndarray:
    bh, bw = map(int, base_grid.shape)
    ph, pw = map(int, plug_grid.shape)
    out_h = max(int(bh), int(dock_r0 + ph))
    out_w = max(int(bw), int(dock_c0 + pw))
    out = np.full((int(out_h), int(out_w)), int(bg), dtype=int)
    paste(out, base_grid, 0, 0, transparent=bg)
    for r in range(ph):
        for c in range(pw):
            val = int(plug_grid[int(r), int(c)])
            if val == int(bg):
                continue
            rr = int(dock_r0 + r)
            cc = int(dock_c0 + c)
            cur = int(out[int(rr), int(cc)])
            if cur != int(bg) and val != int(bg):
                assert cur == int(ctrl) or val == int(ctrl)
            out[int(rr), int(cc)] = int(val)
    out[out == int(ctrl)] = int(obj)
    return _crop_non_bg(out, int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect two sprites by their shared connector color, then tight-crop.

    Variant rule:
    - The two connectors share a color but may have different lengths.
    - Docking uses connector color to identify the two bars, then left-aligns their
      bars while placing the plug connector one row above the base connector.
    """
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4

    extra_cols = sorted({int(v) for v in g.flatten()} - {int(bg), int(obj)})
    assert len(extra_cols) == 1
    ctrl = int(extra_cols[0])

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    def conn(sprite: set[tuple[int, int]]) -> set[tuple[int, int]]:
        pts = {(r, c) for (r, c) in sprite if int(g[int(r), int(c)]) == int(ctrl)}
        _bar_info(pts)
        return pts

    connA = conn(A)
    connB = conn(B)

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r - 1), int(c)) in base_body
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r + 1), int(c)) in plug_body
        assert (int(r - 1), int(c)) not in plug_body

    dr0, dc0, _dlen, _dcols = _bar_info(connD)
    pr0, pc0, _plen, _pcols = _bar_info(connP)

    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))
    base_coords = np.array(list(base), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)
    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(g[int(r), int(c)])

    for r, c in plug_shift:
        val = int(g[int(r - tr), int(c - tc)])
        cur = int(out[int(r - rr0), int(c - cc0)])
        if cur != int(bg) and val != int(bg):
            assert cur == int(ctrl) or val == int(ctrl)
        out[int(r - rr0), int(c - cc0)] = int(val)

    out[out == int(ctrl)] = int(obj)
    return _crop_non_bg(out, int(bg))


def _grow_shape(
    mask: np.ndarray,
    target: int,
    frontier: list[tuple[int, int]],
    allow=None,
) -> None:
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (int(r + dr), int(c + dc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= int(r + dr) < h
            and 0 <= int(c + dc) < w
            and not bool(mask[int(r + dr), int(c + dc)])
            and (allow is None or bool(allow(int(r + dr), int(c + dc))))
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[int(nr), int(nc)] = True
        frontier.append((int(nr), int(nc)))
        count += 1


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    keep = keep or set()
    h, w = mask.shape
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        r = int(r)
        c = int(c)
        if (r, c) in keep:
            continue
        adj_bg = any(
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w) or not bool(mask[int(r + dr), int(c + dc)])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _rect_gap(bh: int, bw: int, pr: int, pc: int, ph: int, pw: int) -> int:
    base_r0, base_c0, base_r1, base_c1 = 0, 0, int(bh - 1), int(bw - 1)
    plug_r0, plug_c0, plug_r1, plug_c1 = int(pr), int(pc), int(pr + ph - 1), int(pc + pw - 1)
    row_gap = max(0, int(plug_r0 - base_r1 - 1), int(base_r0 - plug_r1 - 1))
    col_gap = max(0, int(plug_c0 - base_c1 - 1), int(base_c0 - plug_c1 - 1))
    return int(row_gap + col_gap)


def _valid_input_positions(
    canvas_h: int,
    canvas_w: int,
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    bg: int,
) -> list[tuple[int, int]]:
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    occupied = [(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))]
    plug_cells = [(int(r), int(c)) for r, c in np.argwhere(plug_sprite != int(bg))]
    out = []
    for pr in range(0, int(canvas_h - ph + 1)):
        for pc in range(0, int(canvas_w - pw + 1)):
            if pr + pc == 0:
                continue
            gap = _rect_gap(int(bh), int(bw), int(pr), int(pc), int(ph), int(pw))
            if gap < 1 or gap > 5:
                continue
            ok = True
            for r, c in plug_cells:
                rr = int(pr + r)
                cc = int(pc + c)
                for br, bc in occupied:
                    if abs(int(rr - br)) <= 1 and abs(int(cc - bc)) <= 1:
                        ok = False
                        break
                if not ok:
                    break
            if ok:
                out.append((int(pr), int(pc)))
    return out


def _base_connector_candidates(mask: np.ndarray, conn_len: int) -> list[tuple[int, list[int]]]:
    h, w = mask.shape
    cands = []
    for r in range(1, int(h)):
        for c0 in range(0, int(w - conn_len + 1)):
            cols = list(range(int(c0), int(c0 + conn_len)))
            if not all(bool(mask[int(r), int(c)]) for c in cols):
                continue
            if not all(bool(mask[int(r - 1), int(c)]) for c in cols):
                continue
            if not all(int(r + 1) >= h or not bool(mask[int(r + 1), int(c)]) for c in cols):
                continue
            cands.append((int(r), cols))
    return cands


def _gen_attempt(debug: bool = False):
    def fail(reason: str):
        if debug:
            return None, reason
        return None

    bg, obj = 1, 4
    ctrl = _rng.choice([2, 3, 5, 6, 7, 8, 9])

    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, int(base_h * base_w)))
    base = np.zeros((int(base_h), int(base_w)), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, int(base_target), [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return fail("base_not_contiguous_after_grow")
    _thin(base)

    all_base_cands: list[tuple[int, int, list[int], list[int]]] = []
    for conn_len in range(1, min(3, int(base_w)) + 1):
        for row, cols in _base_connector_candidates(base, int(conn_len)):
            compatible = []
            for plug_len in (1, 2, 3):
                if int(plug_len) == int(conn_len):
                    continue
                ok = True
                for k in range(int(plug_len)):
                    cc = int(cols[0] + k)
                    if cc >= int(base_w):
                        continue
                    if cc in cols:
                        continue
                    if bool(base[int(row), int(cc)]):
                        ok = False
                        break
                if ok:
                    compatible.append(int(plug_len))
            if compatible:
                all_base_cands.append((int(conn_len), int(row), cols, compatible))
    if not all_base_cands:
        return fail("no_base_connector_candidate")

    base_conn_len, conn_row, conn_cols, compatible_plug_lens = _rng.choice(all_base_cands)
    plug_conn_len = _rng.choice(compatible_plug_lens)

    base_grid = np.full((int(base_h), int(base_w)), int(bg), dtype=int)
    base_grid[base] = int(obj)
    for c in conn_cols:
        base_grid[int(conn_row), int(c)] = int(ctrl)
    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return fail("base_grid_not_contiguous")

    plug_h = _rng.randint(2, 7)
    plug_w_min = max(int(plug_conn_len), 2)
    plug_w_max = min(10, int(base_w + 4))
    if plug_w_min > plug_w_max:
        return fail("plug_width_range_empty")
    plug_w = _rng.randint(int(plug_w_min), int(plug_w_max))
    max_conn_c0 = min(int(plug_w - plug_conn_len), int(conn_cols[0]))
    if max_conn_c0 < 0:
        return fail("plug_would_extend_left")
    plug_conn_c0 = _rng.randint(0, int(max_conn_c0))
    dock_c0 = int(conn_cols[0] - plug_conn_c0)
    dock_r0 = int(conn_row - 1)

    plug = np.zeros((int(plug_h), int(plug_w)), dtype=bool)
    conn_set = set(range(int(plug_conn_c0), int(plug_conn_c0 + plug_conn_len)))
    keep_plug = set()
    for c in conn_set:
        plug[0, int(c)] = True
        plug[1, int(c)] = True
        keep_plug.add((0, int(c)))
        keep_plug.add((1, int(c)))

    plug_max = min(29, int(plug_h * plug_w))
    if plug_max < 5:
        return fail("plug_too_small")
    plug_target = _rng.randint(max(5, int(2 * plug_conn_len)), int(plug_max))

    def plug_allowed(r: int, c: int) -> bool:
        if int(r) == 0:
            return int(c) in conn_set
        rr = int(dock_r0 + r)
        cc = int(dock_c0 + c)
        if 0 <= rr < int(base_h) and 0 <= cc < int(base_w):
            base_val = int(base_grid[int(rr), int(cc)])
            if base_val == int(obj):
                return int(r) == 0 and int(c) in conn_set
            if base_val == int(ctrl):
                return int(r) == 1 and int(c) in conn_set
        return True

    _grow_shape(
        plug,
        int(plug_target),
        [(int(r), int(c)) for r, c in np.argwhere(plug)],
        allow=plug_allowed,
    )
    if int(plug.sum()) < int(plug_target):
        return fail("plug_could_not_reach_target")
    if not is_contiguous(plug, background=0, connectivity=8):
        return fail("plug_not_contiguous_after_grow")
    _thin(plug, keep=keep_plug)
    for c in range(int(plug_w)):
        if c not in conn_set:
            plug[0, int(c)] = False
    if not is_contiguous(plug, background=0, connectivity=8):
        return fail("plug_not_contiguous_after_thin")

    row0_cols = sorted(int(c) for c in np.where(plug[0])[0])
    if row0_cols != list(range(int(plug_conn_c0), int(plug_conn_c0 + plug_conn_len))):
        return fail("plug_top_row_not_clean")
    if not all(bool(plug[1, int(c)]) for c in conn_set):
        return fail("plug_support_row_broken")

    for r, c in np.argwhere(plug):
        rr = int(dock_r0 + r)
        cc = int(dock_c0 + c)
        if 0 <= rr < int(base_h) and 0 <= cc < int(base_w):
            base_val = int(base_grid[int(rr), int(cc)])
            if base_val == int(obj):
                if not (int(r) == 0 and int(c) in conn_set):
                    return fail("dock_overlaps_base_body")
            if base_val == int(ctrl):
                if not (int(r) == 1 and int(c) in conn_set):
                    return fail("dock_overlaps_base_ctrl_wrongly")

    plug_grid = np.full((int(plug_h), int(plug_w)), int(bg), dtype=int)
    plug_grid[plug] = int(obj)
    for c in conn_set:
        plug_grid[0, int(c)] = int(ctrl)
    if not is_contiguous(plug_grid != int(bg), background=0, connectivity=8):
        return fail("plug_grid_not_contiguous")

    if int(np.sum(base_grid != int(bg))) < 5 or int(np.sum(plug_grid != int(bg))) < 5:
        return fail("sprite_pixel_count_too_small")

    base_sprite = _crop_non_bg(base_grid, int(bg))
    plug_sprite = _crop_non_bg(plug_grid, int(bg))
    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)

    canvas_h = _rng.randint(max(7, max(int(bh), int(ph)) + 2), 11)
    canvas_w = _rng.randint(max(3, max(int(bw), int(pw))), 13)
    positions = _valid_input_positions(int(canvas_h), int(canvas_w), base_sprite, plug_sprite, int(bg))
    if not positions:
        return fail("no_input_positions")
    pr, pc = _rng.choice(positions)

    base_conn_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite == int(ctrl))}
    plug_conn_pts = {(int(r), int(c)) for r, c in np.argwhere(plug_sprite == int(ctrl))}
    b_conn_r0, b_conn_c0, _b_len, _b_cols = _bar_info(base_conn_pts)
    p_conn_r0, p_conn_c0, _p_len, _p_cols = _bar_info(plug_conn_pts)
    expected = _assemble_output(
        base_sprite,
        plug_sprite,
        ctrl=int(ctrl),
        obj=int(obj),
        bg=int(bg),
        dock_r0=int(b_conn_r0 - 1 - p_conn_r0),
        dock_c0=int(b_conn_c0 - p_conn_c0),
    )

    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=bg)
    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return fail("canvas_not_sparse_enough")

    try:
        result = main(canvas)
    except Exception:
        return fail("solver_failed")
    if not np.array_equal(result, expected):
        return fail("expected_mismatch")
    if debug:
        return canvas, "ok"
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(500):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 500 attempts")
