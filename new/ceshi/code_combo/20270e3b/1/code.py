# Combo variant of 20270e3b (index 1)
# combo_spec: D1/0+D8/1
# Name: 颜色配对链式对接

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location

"""
design_analysis:
  variant_summary:
    kept:
      - 仍然是“水平连接器竖直对接后，连接器消失并重涂成 body 色，再紧裁剪”。
      - base 仍然锚定在输入左上角，且最终输出左上角仍由 base 占据。
      - 所有连接器都仍是单行连续水平条；向下接口要求 body 在上、自由侧在下；向上接口相反。
    changed:
      - 从 2 精灵变成 3 精灵链式结构 A-B-C，需要做两次对接。
      - B 不再是“普通 plug”，而是中间件：上方一个向上接口、下方一个向下接口。
      - 两次对接各自使用不同的连接器颜色；配对依据从“同形”改成“同色”。
      - 每一对连接器长度允许不同，并在生成器中强制 A-B 与 B-C 两对都异长，使颜色成为必要线索。

  a_concretization_and_trace:
    input_look:
      - A 在原点，只有 1 条向下连接器，颜色记为 x。
      - B 有两条连接器：顶边向上接口颜色也是 x；底边向下接口颜色为 y，且 x != y。
      - C 只有 1 条向上连接器，颜色为 y。
      - x 与 y 都从 {2,3,5,6,7,8,9} 里选，body 固定 4，bg 固定 1。
    output_look:
      - 第一步：把 B 的 x 色顶接口左端对齐到 A 的 x 色接口左端，并放到 A 接口正上方一行。
      - 第二步：再把 C 的 y 色顶接口左端对齐到已平移 B 的 y 色底接口左端，并放到其正上方一行。
      - 两步完成后，所有 x/y 连接器颜色统一变成 body=4，并对非背景做紧裁剪。
    concrete_trace:
      - 例：A 的 x 色接口长度 3，B 顶部 x 色接口长度 1；B 底部 y 色接口长度 2，C 的 y 色接口长度 3。
      - 原题/原 solver 会先要求“两个接口同形”，且只处理 2 个精灵；这里会直接失效。
      - 融合版先识别 3 个前景组件，再识别 2 种额外颜色 x/y，并找到唯一的角色分工：
        A=原点+单 down，B=一 up 一 down，C=单 up。
      - 然后按颜色等式 A.x == B.top.x、B.bottom.y == C.y 来确定两条接缝，而不是按长度或颜色值大小排序。
      - A-B 对接与 B-C 对接都采用“左端点对齐 + plug 接口在 dock 目标上方 1 行”的几何。
      - 差异点在于：先做颜色配对消歧，再做两步空间对接；任何一步都不是原题能覆盖的。

  b_complexity_calibration:
    - 只用 2 种连接器颜色，避免颜色种类过多导致样例难以归纳。
    - 只用 3 个精灵、4 条接口，且中间件 B 的“双接口”视觉上非常明显，链式结构可从少量样例看出。
    - 每对接口长度都限制在 1-3，并强制异长，既能突出“不是按形状配对”，又不会把自由度放得过大。
    - 颜色配对和多步对接是协同的：若没有颜色匹配，B 的两条接口无法与 A/C 正确对应；若没有链式结构，也不需要第二种颜色。

  c_hidden_assumption_review_main:
    - `ctrl = 7` 失效：现在需要从网格中推断恰好 2 种额外颜色。
    - `assert len(best_comps) == 2` 失效：现在恰好 3 个 8 连通组件。
    - “每个精灵只有一个 connector 集合”失效：B 有两个互不接触的连接器，且颜色不同。
    - `assert shpA == shpB` 失效：颜色决定配对，长度可不同，因此不能再要求归一化形状一致。
    - 原始二元 role split 失效：现在要先根据原点锚定和接口朝向数目识别 A/B/C。
    - 单次平移不够：必须先求 B 相对 A 的平移，再求 C 相对已平移 B 的平移。
    - 重叠断言也要放宽到“两种 connector 色都可作为合法 seam 重叠色”，而不是单个 ctrl。

  d_placement_legality:
    - 输入放置时 3 个精灵必须两两 8 邻域分离，否则会在 solver 中并成同一组件。
    - A 固定粘在输入左上角；B、C 用 `random_free_location(..., padding=1)` 分散放置。
    - 生成 assembled 链时，B 和 C 都不允许伸到 A 的上方或左方，维持 base 仍是裁剪后的左上锚点。
    - 合法重叠只允许发生在两条接缝：A 的 x 接口与 B 的首层 body，B 的 y 接口与 C 的首层 body。
    - generator 在 grow 阶段按 dock 后坐标逐像素检查，禁止与普通 body 非法重叠。

  e_visual_observability:
    - 观察者能直接看到：一共有 3 个对象，其中 1 个对象带两条不同色的横条，明显是“中间件”。
    - 两个端点对象各只有一条横条，且各自只和 B 的某一条同色，这让“先配颜色，再接链”可从 2-4 组样例归纳出来。
    - 每对都异长，排除了“其实还是按同长度接”的误解。
    - 角色判定只依赖可见结构（原点锚定、接口朝向、颜色共享），不依赖颜色编号大小规律等不可见先验。

  f_visual_cleanliness:
    - 整个任务只使用 4 种角色色：bg、body、接口色 x、接口色 y。
    - 每条连接器都保持为干净的单行水平条；B 的顶行/底行只保留连接器，避免边界噪声。
    - 输入采用稀疏散落布局，输出只包含简单两级 seam，不会出现三重叠层。

  g_color_inference:
    - 颜色变化并非任意映射，而是“凡是承担连接器语义的两种颜色，输出里都变成 body 色 4”。
    - 两种颜色各自在两个不同精灵上重复出现，且都绑定在水平条结构上，能给人足够的结构提示。
    - 颜色数量被限制为 2，避免“过多颜色 + 过少样例”导致不可推断。

  h_new_main_assumptions_and_generator_guarantees:
    - main 假设输入中恰有 2 种非 bg / 非 obj 颜色；generator 显式只使用两种 connector 色。
    - main 假设恰有 3 个前景组件；generator 用间隔放置保证这一点。
    - main 假设 base 是唯一原点组件且只有 1 个 down 接口；generator 直接构造并固定 A 在 (0,0)。
    - main 假设 middle 是唯一同时拥有 1 个 up 与 1 个 down 接口的组件，且两条接口颜色不同；generator 显式构造。
    - main 假设每种 connector 色正好出现在 2 条接口上，分别形成 A-B 与 B-C 两个配对；generator 按 x/y 成对分配。
    - main 假设按左端点对齐后不会越过 A 的上/左边界，也不会与普通 body 非法相撞；generator 在 assembled 坐标里逐像素验证。
"""


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_non_bg(grid: np.ndarray, bg: int) -> tuple[np.ndarray, int, int]:
    nz = np.argwhere(grid != int(bg))
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy(), int(r0), int(c0)


def _connector_components(
    g: np.ndarray,
    sprite: set[tuple[int, int]],
    extra_colors: set[int],
) -> list[set[tuple[int, int]]]:
    r0, c0, r1, c1 = _bbox_of_pts(sprite)
    comps: list[set[tuple[int, int]]] = []
    for color in sorted(int(v) for v in extra_colors):
        mask = np.zeros((int(r1 - r0 + 1), int(c1 - c0 + 1)), dtype=bool)
        for r, c in sprite:
            if int(g[int(r), int(c)]) == int(color):
                mask[int(r - r0), int(c - c0)] = True
        lab, n = label_mask(mask, connectivity=4)
        for k in range(1, int(n) + 1):
            pts = {
                (int(r0 + rr), int(c0 + cc))
                for rr, cc in np.argwhere(lab == int(k))
            }
            comps.append(pts)
    return comps


def _classify_connector(
    sprite: set[tuple[int, int]],
    conn_pts: set[tuple[int, int]],
    g: np.ndarray,
    obj: int,
) -> dict[str, object]:
    rows = sorted({int(r) for r, _ in conn_pts})
    cols = sorted(int(c) for _, c in conn_pts)
    assert len(rows) == 1
    assert cols == list(range(int(cols[0]), int(cols[0] + len(cols))))
    color = int(g[next(iter(conn_pts))])
    assert color != int(obj)
    body = sprite - conn_pts
    row = int(rows[0])

    up = all((int(row + 1), int(c)) in body for c in cols) and all(
        (int(row - 1), int(c)) not in body for c in cols
    )
    down = all((int(row - 1), int(c)) in body for c in cols) and all(
        (int(row + 1), int(c)) not in body for c in cols
    )
    assert up ^ down
    return {
        "pts": conn_pts,
        "row": int(row),
        "c0": int(cols[0]),
        "len": int(len(cols)),
        "cols": cols,
        "color": int(color),
        "orientation": "up" if up else "down",
    }


def _sprite_desc(
    g: np.ndarray,
    sprite: set[tuple[int, int]],
    *,
    obj: int,
    extra_colors: set[int],
) -> dict[str, object]:
    connectors = [
        _classify_connector(sprite, comp, g, obj)
        for comp in _connector_components(g, sprite, extra_colors)
    ]
    r0, c0, r1, c1 = _bbox_of_pts(sprite)
    return {
        "pts": sprite,
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "top_left_sum": int(r0 + c0),
        "connectors": connectors,
    }


def _stamp_shifted(
    out: np.ndarray,
    patch: np.ndarray,
    top: int,
    left: int,
    *,
    bg: int,
    ctrl_colors: set[int],
) -> None:
    h, w = patch.shape
    for r in range(int(h)):
        for c in range(int(w)):
            val = int(patch[int(r), int(c)])
            if val == int(bg):
                continue
            rr, cc = int(top + r), int(left + c)
            cur = int(out[int(rr), int(cc)])
            if cur != int(bg):
                assert cur in ctrl_colors or val in ctrl_colors
            out[int(rr), int(cc)] = int(val)


def _render_chain(
    base_grid: np.ndarray,
    mid_grid: np.ndarray,
    end_grid: np.ndarray,
    *,
    mid_shift: tuple[int, int],
    end_shift: tuple[int, int],
    bg: int,
    ctrl_colors: set[int],
) -> np.ndarray | None:
    bh, bw = map(int, base_grid.shape)
    mh, mw = map(int, mid_grid.shape)
    eh, ew = map(int, end_grid.shape)
    tr_mid, tc_mid = map(int, mid_shift)
    tr_end, tc_end = map(int, end_shift)
    if min(tr_mid, tc_mid, tr_end, tc_end) < 0:
        return None
    H = max(int(bh), int(tr_mid + mh), int(tr_end + eh))
    W = max(int(bw), int(tc_mid + mw), int(tc_end + ew))
    out = np.full((int(H), int(W)), int(bg), dtype=int)
    try:
        _stamp_shifted(out, base_grid, 0, 0, bg=bg, ctrl_colors=ctrl_colors)
        _stamp_shifted(out, mid_grid, int(tr_mid), int(tc_mid), bg=bg, ctrl_colors=ctrl_colors)
        _stamp_shifted(out, end_grid, int(tr_end), int(tc_end), bg=bg, ctrl_colors=ctrl_colors)
    except AssertionError:
        return None
    return out


def _parse_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4
    extra_colors = {int(v) for v in g.flatten()} - {int(bg), int(obj)}
    assert len(extra_colors) == 2

    lab, n = label_mask(g != int(bg), connectivity=8)
    comps: list[set[tuple[int, int]]] = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 3

    sprites = [
        _sprite_desc(g, pts, obj=int(obj), extra_colors=extra_colors)
        for pts in comps
    ]

    origin = [spr for spr in sprites if int(spr["top_left_sum"]) == 0]
    assert len(origin) == 1
    base = origin[0]
    others = [spr for spr in sprites if spr is not base]

    assert len(base["connectors"]) == 1
    base_downs = [c for c in base["connectors"] if c["orientation"] == "down"]
    assert len(base_downs) == 1
    base_down = base_downs[0]

    middle_candidates = []
    end_candidates = []
    for spr in others:
        orientations = sorted(str(c["orientation"]) for c in spr["connectors"])
        if len(spr["connectors"]) == 2 and orientations == ["down", "up"]:
            colors = {int(c["color"]) for c in spr["connectors"]}
            assert len(colors) == 2
            middle_candidates.append(spr)
        elif len(spr["connectors"]) == 1 and orientations == ["up"]:
            end_candidates.append(spr)
    assert len(middle_candidates) == 1 and len(end_candidates) == 1
    middle = middle_candidates[0]
    end = end_candidates[0]

    mid_up = [c for c in middle["connectors"] if c["orientation"] == "up"]
    mid_down = [c for c in middle["connectors"] if c["orientation"] == "down"]
    end_up = [c for c in end["connectors"] if c["orientation"] == "up"]
    assert len(mid_up) == 1 and len(mid_down) == 1 and len(end_up) == 1
    mid_up = mid_up[0]
    mid_down = mid_down[0]
    end_up = end_up[0]

    assert int(base_down["color"]) == int(mid_up["color"])
    assert int(mid_down["color"]) == int(end_up["color"])
    assert int(base_down["color"]) != int(mid_down["color"])

    color_hist: dict[int, int] = {}
    for spr in sprites:
        for conn in spr["connectors"]:
            color = int(conn["color"])
            color_hist[color] = int(color_hist.get(color, 0) + 1)
            assert 1 <= int(conn["len"]) <= 3
    assert set(color_hist) == {int(v) for v in extra_colors}
    assert sorted(color_hist.values()) == [2, 2]

    return {
        "g": g,
        "bg": int(bg),
        "obj": int(obj),
        "extra_colors": {int(v) for v in extra_colors},
        "base": base,
        "middle": middle,
        "end": end,
        "base_down": base_down,
        "mid_up": mid_up,
        "mid_down": mid_down,
        "end_up": end_up,
    }


def main(grid: np.ndarray) -> np.ndarray:
    scene = _parse_scene(np.array(grid, dtype=int))
    g = scene["g"]
    bg = int(scene["bg"])
    obj = int(scene["obj"])
    ctrl_colors = {int(v) for v in scene["extra_colors"]}
    base = scene["base"]
    middle = scene["middle"]
    end = scene["end"]
    base_down = scene["base_down"]
    mid_up = scene["mid_up"]
    mid_down = scene["mid_down"]
    end_up = scene["end_up"]

    tr_mid = int(int(base_down["row"]) - 1 - int(mid_up["row"]))
    tc_mid = int(int(base_down["c0"]) - int(mid_up["c0"]))
    tr_end = int(int(mid_down["row"]) + tr_mid - 1 - int(end_up["row"]))
    tc_end = int(int(mid_down["c0"]) + tc_mid - int(end_up["c0"]))

    shifted_mid = {(int(r + tr_mid), int(c + tc_mid)) for r, c in middle["pts"]}
    shifted_end = {(int(r + tr_end), int(c + tc_end)) for r, c in end["pts"]}
    pts_all = set(base["pts"]) | shifted_mid | shifted_end
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))

    base_coords = np.array(list(base["pts"]), dtype=int)
    br0 = int(base_coords[:, 0].min())
    bc0 = int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(bg), dtype=int)

    for spr, tr, tc in (
        (base, 0, 0),
        (middle, int(tr_mid), int(tc_mid)),
        (end, int(tr_end), int(tc_end)),
    ):
        r0, c0, r1, c1 = map(int, spr["bbox"])
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        _stamp_shifted(
            out,
            patch,
            int(r0 + tr - rr0),
            int(c0 + tc - cc0),
            bg=bg,
            ctrl_colors=ctrl_colors,
        )

    for color in ctrl_colors:
        out[out == int(color)] = int(obj)
    return _crop_non_bg(out, int(bg))[0]


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
            not (0 <= int(r + dr) < h and 0 <= int(c + dc) < w)
            or not bool(mask[int(r + dr), int(c + dc)])
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[int(r), int(c)] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[int(r), int(c)] = True


def _down_connector_candidates(mask: np.ndarray, conn_len: int) -> list[tuple[int, int]]:
    h, w = mask.shape
    out: list[tuple[int, int]] = []
    for r in range(1, int(h)):
        for c0 in range(0, int(w - conn_len + 1)):
            cols = range(int(c0), int(c0 + conn_len))
            if not all(bool(mask[int(r), int(c)]) for c in cols):
                continue
            if not all(bool(mask[int(r - 1), int(c)]) for c in cols):
                continue
            if any(int(r + 1) < h and bool(mask[int(r + 1), int(c)]) for c in cols):
                continue
            out.append((int(r), int(c0)))
    return out


def _compatible_up_lengths(
    mask: np.ndarray,
    row: int,
    c0: int,
    base_len: int,
) -> list[int]:
    cols = set(range(int(c0), int(c0 + base_len)))
    out: list[int] = []
    for plug_len in (1, 2, 3):
        if int(plug_len) == int(base_len):
            continue
        ok = True
        for k in range(int(plug_len)):
            cc = int(c0 + k)
            if cc >= int(mask.shape[1]):
                continue
            if cc in cols:
                continue
            if bool(mask[int(row), int(cc)]):
                ok = False
                break
        if ok:
            out.append(int(plug_len))
    return out


def _gen_attempt() -> np.ndarray | None:
    bg, obj = 1, 4
    ctrl_ab, ctrl_bc = _rng.sample([2, 3, 5, 6, 7, 8, 9], 2)
    ctrl_colors = {int(ctrl_ab), int(ctrl_bc)}

    base_h = _rng.randint(3, 7)
    base_w = _rng.randint(4, 9)
    base_low = 6
    base_high = min(24, int(base_h * base_w))
    if base_low > base_high:
        return None
    base_target = _rng.randint(int(base_low), int(base_high))
    base_mask = np.zeros((int(base_h), int(base_w)), dtype=bool)
    base_mask[0, 0] = True
    _grow_shape(base_mask, int(base_target), [(0, 0)])
    if not is_contiguous(base_mask, background=0, connectivity=8):
        return None
    _thin(base_mask)

    base_choices: list[tuple[int, int, int, list[int]]] = []
    for base_len in (1, 2, 3):
        if int(base_len) > int(base_w):
            continue
        for row, c0 in _down_connector_candidates(base_mask, int(base_len)):
            mids = _compatible_up_lengths(base_mask, int(row), int(c0), int(base_len))
            if mids:
                base_choices.append((int(base_len), int(row), int(c0), mids))
    if not base_choices:
        return None
    base_len, base_row, base_c0, mid_top_lens = _rng.choice(base_choices)
    mid_top_len = _rng.choice(mid_top_lens)

    base_grid = np.full((int(base_h), int(base_w)), int(bg), dtype=int)
    base_grid[base_mask] = int(obj)
    base_grid[int(base_row), int(base_c0) : int(base_c0 + base_len)] = int(ctrl_ab)
    if not is_contiguous(base_grid != int(bg), background=0, connectivity=8):
        return None
    base_grid, _br0, _bc0 = _crop_non_bg(base_grid, int(bg))
    base_h, base_w = map(int, base_grid.shape)

    mid_bot_len = _rng.randint(1, 3)
    end_len_choices = [v for v in (1, 2, 3) if int(v) != int(mid_bot_len)]
    end_len = _rng.choice(end_len_choices)

    mid_h = _rng.randint(5, 8)
    mid_w = _rng.randint(max(3, int(max(mid_top_len, mid_bot_len))), 9)
    max_top_c0 = min(int(mid_w - mid_top_len), int(base_c0))
    if max_top_c0 < 0:
        return None
    mid_top_c0 = _rng.randint(0, int(max_top_c0))
    mid_bot_c0 = _rng.randint(0, int(mid_w - mid_bot_len))
    mid_shift = (int(base_row - 1), int(base_c0 - mid_top_c0))

    mid_mask = np.zeros((int(mid_h), int(mid_w)), dtype=bool)
    top_cols = set(range(int(mid_top_c0), int(mid_top_c0 + mid_top_len)))
    bot_cols = set(range(int(mid_bot_c0), int(mid_bot_c0 + mid_bot_len)))
    for c in top_cols:
        mid_mask[0, int(c)] = True
        mid_mask[1, int(c)] = True
    for c in bot_cols:
        mid_mask[int(mid_h - 1), int(c)] = True
        mid_mask[int(mid_h - 2), int(c)] = True
    top_center = int(mid_top_c0 + mid_top_len // 2)
    bot_center = int(mid_bot_c0 + mid_bot_len // 2)
    for r in range(1, int(mid_h - 1)):
        mid_mask[int(r), int(top_center)] = True
    for c in range(min(int(top_center), int(bot_center)), max(int(top_center), int(bot_center)) + 1):
        mid_mask[int(mid_h - 2), int(c)] = True

    mid_seed = int(mid_mask.sum())
    mid_low = max(8, int(mid_seed + 1))
    mid_high = min(25, int(mid_h * mid_w))
    if mid_low > mid_high:
        return None
    mid_target = _rng.randint(int(mid_low), int(mid_high))

    def mid_allowed(r: int, c: int) -> bool:
        rr = int(mid_shift[0] + r)
        cc = int(mid_shift[1] + c)
        if 0 <= rr < int(base_h) and 0 <= cc < int(base_w):
            base_val = int(base_grid[int(rr), int(cc)])
            if base_val == int(obj):
                return False
            if base_val == int(ctrl_ab):
                return int(r) == 1 and int(c) in top_cols
        return True

    _grow_shape(
        mid_mask,
        int(mid_target),
        [(int(r), int(c)) for r, c in np.argwhere(mid_mask)],
        allow=mid_allowed,
    )
    if int(mid_mask.sum()) < int(mid_target):
        return None
    if not is_contiguous(mid_mask, background=0, connectivity=8):
        return None

    keep_mid = {
        (0, int(c)) for c in top_cols
    } | {
        (1, int(c)) for c in top_cols
    } | {
        (int(mid_h - 1), int(c)) for c in bot_cols
    } | {
        (int(mid_h - 2), int(c)) for c in bot_cols
    }
    _thin(mid_mask, keep=keep_mid)
    for c in range(int(mid_w)):
        if c not in top_cols:
            mid_mask[0, int(c)] = False
        if c not in bot_cols:
            mid_mask[int(mid_h - 1), int(c)] = False
    if not all(bool(mid_mask[1, int(c)]) for c in top_cols):
        return None
    if not all(bool(mid_mask[int(mid_h - 2), int(c)]) for c in bot_cols):
        return None
    if not is_contiguous(mid_mask, background=0, connectivity=8):
        return None

    mid_grid = np.full((int(mid_h), int(mid_w)), int(bg), dtype=int)
    mid_grid[mid_mask] = int(obj)
    mid_grid[0, int(mid_top_c0) : int(mid_top_c0 + mid_top_len)] = int(ctrl_ab)
    mid_grid[int(mid_h - 1), int(mid_bot_c0) : int(mid_bot_c0 + mid_bot_len)] = int(ctrl_bc)
    if not is_contiguous(mid_grid != int(bg), background=0, connectivity=8):
        return None
    mid_grid, _mr0, mc0 = _crop_non_bg(mid_grid, int(bg))
    mid_top_c0 -= int(mc0)
    mid_bot_c0 -= int(mc0)
    mid_h, mid_w = map(int, mid_grid.shape)
    mid_shift = (int(base_row - 1), int(base_c0 - mid_top_c0))

    partial = _render_chain(
        base_grid,
        mid_grid,
        np.full((1, 1), int(bg), dtype=int),
        mid_shift=mid_shift,
        end_shift=(0, 0),
        bg=int(bg),
        ctrl_colors=ctrl_colors,
    )
    if partial is None:
        return None

    end_h = _rng.randint(3, 7)
    end_w = _rng.randint(max(2, int(end_len)), 8)
    global_mid_bot_c0 = int(mid_shift[1] + mid_bot_c0)
    max_end_c0 = min(int(end_w - end_len), int(global_mid_bot_c0))
    if max_end_c0 < 0:
        return None
    end_c0 = _rng.randint(0, int(max_end_c0))
    end_shift = (
        int(mid_shift[0] + mid_h - 2),
        int(global_mid_bot_c0 - end_c0),
    )

    end_mask = np.zeros((int(end_h), int(end_w)), dtype=bool)
    end_cols = set(range(int(end_c0), int(end_c0 + end_len)))
    for c in end_cols:
        end_mask[0, int(c)] = True
        end_mask[1, int(c)] = True
    if int(end_h) > 2:
        for c in end_cols:
            end_mask[2, int(c)] = True

    end_seed = int(end_mask.sum())
    end_low = max(6, int(end_seed + 1))
    end_high = min(22, int(end_h * end_w))
    if end_low > end_high:
        return None
    end_target = _rng.randint(int(end_low), int(end_high))

    def end_allowed(r: int, c: int) -> bool:
        rr = int(end_shift[0] + r)
        cc = int(end_shift[1] + c)
        if 0 <= rr < int(partial.shape[0]) and 0 <= cc < int(partial.shape[1]):
            val = int(partial[int(rr), int(cc)])
            if val == int(obj):
                return False
            if val == int(ctrl_bc):
                return int(r) == 1 and int(c) in end_cols
            if val == int(ctrl_ab):
                return False
        return True

    _grow_shape(
        end_mask,
        int(end_target),
        [(int(r), int(c)) for r, c in np.argwhere(end_mask)],
        allow=end_allowed,
    )
    if int(end_mask.sum()) < int(end_target):
        return None
    if not is_contiguous(end_mask, background=0, connectivity=8):
        return None
    keep_end = {
        (0, int(c)) for c in end_cols
    } | {
        (1, int(c)) for c in end_cols
    }
    if int(end_h) > 2:
        keep_end |= {(2, int(c)) for c in end_cols}
    _thin(end_mask, keep=keep_end)
    for c in range(int(end_w)):
        if c not in end_cols:
            end_mask[0, int(c)] = False
    if not all(bool(end_mask[1, int(c)]) for c in end_cols):
        return None
    if not is_contiguous(end_mask, background=0, connectivity=8):
        return None

    end_grid = np.full((int(end_h), int(end_w)), int(bg), dtype=int)
    end_grid[end_mask] = int(obj)
    end_grid[0, int(end_c0) : int(end_c0 + end_len)] = int(ctrl_bc)
    if not is_contiguous(end_grid != int(bg), background=0, connectivity=8):
        return None
    end_grid, _er0, ec0 = _crop_non_bg(end_grid, int(bg))
    end_c0 -= int(ec0)
    end_h, end_w = map(int, end_grid.shape)
    end_shift = (
        int(mid_shift[0] + mid_h - 2),
        int(global_mid_bot_c0 - end_c0),
    )

    chain = _render_chain(
        base_grid,
        mid_grid,
        end_grid,
        mid_shift=mid_shift,
        end_shift=end_shift,
        bg=int(bg),
        ctrl_colors=ctrl_colors,
    )
    if chain is None:
        return None
    chain_out = chain.copy()
    for color in ctrl_colors:
        chain_out[chain_out == int(color)] = int(obj)
    chain_out, _r0, _c0 = _crop_non_bg(chain_out, int(bg))
    if chain_out.size == 0:
        return None
    if chain_out.shape[0] > 12 or chain_out.shape[1] > 16:
        return None

    if int(np.sum(base_grid != int(bg))) < 6:
        return None
    if int(np.sum(mid_grid != int(bg))) < 8:
        return None
    if int(np.sum(end_grid != int(bg))) < 6:
        return None

    canvas_h_min = max(11, int(max(base_grid.shape[0], mid_grid.shape[0], end_grid.shape[0]) + 2))
    canvas_w_min = max(
        12,
        int(base_grid.shape[1] + mid_grid.shape[1] + end_grid.shape[1] + 2),
    )
    canvas_h = _rng.randint(int(canvas_h_min), max(int(canvas_h_min), 16))
    canvas_w = _rng.randint(int(canvas_w_min), max(int(canvas_w_min), 20))
    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    paste(canvas, base_grid, 0, 0, transparent=bg)
    try:
        mr, mc = random_free_location(canvas, mid_grid, transparent=bg, padding=1)
        paste(canvas, mid_grid, int(mr), int(mc), transparent=bg)
        er, ec = random_free_location(canvas, end_grid, transparent=bg, padding=1)
        paste(canvas, end_grid, int(er), int(ec), transparent=bg)
    except ValueError:
        return None
    if (int(mr) + int(mc)) == 0 or (int(er) + int(ec)) == 0:
        return None
    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return None

    try:
        result = main(canvas)
    except Exception:
        return None
    if not np.array_equal(result, chain_out):
        return None

    scene = _parse_scene(canvas)
    assert int(scene["base_down"]["len"]) != int(scene["mid_up"]["len"])
    assert int(scene["mid_down"]["len"]) != int(scene["end_up"]["len"])
    assert int(scene["base_down"]["color"]) != int(scene["mid_down"]["color"])
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(700):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 700 attempts")
