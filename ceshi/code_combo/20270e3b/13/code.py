# Combo variant of 20270e3b (index 13)
# combo_spec: D1/0+D10/1
# Name: 多色连接器颜色配对

"""
design_analysis:
  variant_summary:
    kept:
      - 仍然只有两个 8 连通前景精灵，base 放在输入左上角，plug 分离放置。
      - 输出仍然是把 plug 平移到 base 上方一行进行对接，再把所有连接器重涂成 body 色并紧裁剪。
      - 连接器仍然是水平单行条；base 连接器自由侧朝下，plug 连接器自由侧朝上。
    changed:
      - 每个精灵不再只有一个连接器，而是恰有两个不同颜色的连接器。
      - 两个精灵之间只有一种颜色是共享的；这一对同色连接器才是活跃对接对。
      - 活跃对的长度允许不同，并且生成器强制二者不同。
      - 为了防止退化成“按长度/形状配对”，干扰连接器采用交叉长度：
        base 的干扰条长度 = plug 活跃条长度，plug 的干扰条长度 = base 活跃条长度。

  a_concretization_and_trace:
    input_look:
      - 背景固定为 1，body 固定为 4。
      - base 例如有红色长度 3 的活跃连接器、绿色长度 1 的干扰连接器。
      - plug 例如有红色长度 1 的活跃连接器、灰色长度 3 的干扰连接器。
    original_trace:
      - 原题只有一对橙色连接器，solver 直接取唯一 ctrl 颜色并要求两条同形。
      - 原题平移时用两个连接器的左上锚点对齐；因为长度相同，这等价于整条重合。
    fused_trace:
      - 新 solver 先提取每个精灵中全部非 bg/non-obj 的水平色条。
      - 只看两边颜色交集，唯一共享颜色决定活跃条。
      - 平移时按活跃条左端点对齐，并把 plug 活跃条放到 base 活跃条正上方一行。
      - 活跃条长度不同，所以“整条同形重合”不再成立；左对齐后只有重叠前缀处形成对接口。
      - 干扰条虽然也可见，但因颜色不共享而不会被选中。
    key_difference:
      - 与原题相比，几何同形从必要条件降为无关项；真正稳定的是“唯一共享颜色”。
      - 与仅有单一彩色连接器的简单版本相比，这里必须先在多个颜色条中找同色对。

  b_complexity_calibration:
    - 每个精灵固定 2 条连接器，足以形成“找同色对”的可见推理，但不会像 3-4 条那样杂乱。
    - 长度只取 1-3 三档，并强制活跃对长度不同，变化明显但组合仍少。
    - 交叉长度让错误的“按形状配对”策略会挑到干扰条，从而保证颜色线索真的有用。
    - 颜色总数固定为 3 个连接器颜色：1 个共享色 + 2 个各自独有色，推断负担可控。

  c_hidden_assumption_review_main:
    - 原 `ctrl = 7` 失效：现在连接器颜色来自多个非 {bg,obj} 颜色，必须动态枚举。
    - 原 `assert len(best_comps) == 2` 仍保留：generator 继续保证只有两个主精灵。
    - 原“每个精灵只有一个连接器”失效：main 需提取每个精灵的全部连接器色条。
    - 原 `shpA == shpB` 失效：活跃对长度故意不同，不能再用归一化形状比较。
    - 原重叠断言 `cur == ctrl or val == ctrl` 需泛化为“重叠至少一侧是某种连接器色”。
    - 原 `out[out == ctrl] = obj` 需泛化为“所有非 {bg,obj} 颜色都重涂为 obj”。
    - 新隐式假设：两精灵颜色交集恰好 1 种，且该颜色在每个精灵中恰好对应 1 条水平连续色条。
      generator 会显式保证这一点。

  d_placement_legality:
    - base 仍裁剪后贴在输入画布原点。
    - plug 输入位置必须与 base 8 邻域分离，否则会被 solver 视为同一连通块。
    - 组装态中只允许一种重叠：plug 活跃连接器下方那一行 body 覆盖到 base 活跃连接器上。
    - plug 不得在组装后伸到 base 上边或左边；因此活跃条左端点对齐时要求 dock_c0 >= 0。
    - 干扰连接器在组装态不能与 base 任意 body/连接器发生重叠，generator 按像素验证。

  e_visual_observability:
    - 人类看样例时能直接看到：每个输入里都有两种以上连接器颜色，但只有一种颜色在两精灵上都出现。
    - 同时可见到同色活跃条长度经常不同，因此“找同色”会比“找同长”更稳定。
    - 干扰条保留在输入中、输出中消失为 body 色，使其角色既可见又不会误导输出结构。

  f_visual_cleanliness:
    - 每个精灵仅 2 条连接器，且同一精灵内连接器颜色不同。
    - 连接器只放在合法边界段，不叠加文本式标记，也不使用黑色 0。
    - 生成器要求同一行上的两条连接器不要相互贴边，避免视觉拥挤。

  g_color_inference:
    - 颜色变化不是任意映射；解题者只需识别“跨两精灵重复出现的那一种颜色”。
    - body 和 bg 固定，连接器色只从少量备选色中取值，降低无结构配色噪声。
    - 共享颜色在两个精灵上承担同一结构角色，干扰颜色只在单个精灵出现，线索明确。

  h_non_degeneracy_target:
    - 生成器强制 `active_base_len != active_plug_len`。
    - 生成器还强制 `base_distr_len == active_plug_len` 且 `plug_distr_len == active_base_len`。
    - 因此“按长度配对”会落到错误的异色对上；只有“按共享颜色配对”才稳定。

  i_main_assumptions_vs_generator_guarantees:
    - main 假设每个精灵恰有 2 条连接器色条；generator 只给每个精灵涂两条。
    - main 假设共享颜色唯一；generator 采用 1 个共享色 + 2 个独有色。
    - main 假设活跃 base 条满足上方为 body、下方为空；plug 条满足下方为 body、上方为空；
      generator 的候选段筛选和组装验证保证这一点。
    - main 假设左端点对齐后不会出现非法重叠；generator 在 docked 坐标系逐像素检查。
"""

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste


def _crop_nonbg(arr: np.ndarray, bg: int) -> np.ndarray:
    nz = np.argwhere(arr != int(bg))
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return arr[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _extract_components(grid: np.ndarray, bg: int) -> list[dict]:
    lab, n = label_mask(grid != int(bg), connectivity=8)
    comps: list[dict] = []
    for k in range(1, int(n) + 1):
        pts = np.argwhere(lab == int(k))
        r0, c0, r1, c1 = bbox_coords(pts)
        sprite = np.full((int(r1 - r0 + 1), int(c1 - c0 + 1)), int(bg), dtype=int)
        for r, c in pts:
            sprite[int(r - r0), int(c - c0)] = int(grid[int(r), int(c)])
        comps.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "sprite": sprite,
            }
        )
    comps.sort(key=lambda item: int(item["bbox"][0]) + int(item["bbox"][1]))
    return comps


def _connector_runs(sprite: np.ndarray, bg: int, obj: int) -> list[dict]:
    h, w = map(int, sprite.shape)
    runs: list[dict] = []
    for r in range(h):
        c = 0
        while c < w:
            color = int(sprite[int(r), int(c)])
            if color in (int(bg), int(obj)):
                c += 1
                continue
            c0 = int(c)
            while c + 1 < w and int(sprite[int(r), int(c + 1)]) == int(color):
                c += 1
            c1 = int(c)
            runs.append(
                {
                    "color": int(color),
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c1),
                    "pts": {(int(r), int(cc)) for cc in range(int(c0), int(c1) + 1)},
                }
            )
            c += 1
    return runs


def _unique_run_by_color(runs: list[dict], color: int) -> dict:
    matches = [run for run in runs if int(run["color"]) == int(color)]
    assert len(matches) == 1
    return matches[0]


def _validate_active_run(sprite: np.ndarray, run: dict, *, bg: int, obj: int, mode: str) -> None:
    r = int(run["row"])
    for _r, c in run["pts"]:
        c = int(c)
        if mode == "base":
            assert r > 0
            assert int(sprite[int(r - 1), int(c)]) == int(obj)
            assert int(r + 1) >= sprite.shape[0] or int(sprite[int(r + 1), int(c)]) == int(bg)
        else:
            assert int(r + 1) < sprite.shape[0]
            assert int(sprite[int(r + 1), int(c)]) == int(obj)
            assert int(r) == 0 or int(sprite[int(r - 1), int(c)]) == int(bg)


def _assemble_output(
    base_sprite: np.ndarray,
    plug_sprite: np.ndarray,
    *,
    bg: int,
    obj: int,
    match_color: int,
) -> np.ndarray:
    runs_base = _connector_runs(base_sprite, bg, obj)
    runs_plug = _connector_runs(plug_sprite, bg, obj)
    base_run = _unique_run_by_color(runs_base, int(match_color))
    plug_run = _unique_run_by_color(runs_plug, int(match_color))
    _validate_active_run(base_sprite, base_run, bg=bg, obj=obj, mode="base")
    _validate_active_run(plug_sprite, plug_run, bg=bg, obj=obj, mode="plug")

    tr = int(base_run["row"] - 1 - plug_run["row"])
    tc = int(base_run["c0"] - plug_run["c0"])
    conn_colors = {int(run["color"]) for run in runs_base + runs_plug}

    base_pts = {(int(r), int(c)) for r, c in np.argwhere(base_sprite != int(bg))}
    plug_pts = {(int(r + tr), int(c + tc)) for r, c in np.argwhere(plug_sprite != int(bg))}
    pts_all = base_pts | plug_pts
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(np.array(list(pts_all), dtype=int)))
    assert rr0 == 0 and cc0 == 0

    out = np.full((int(rr1 + 1), int(cc1 + 1)), int(bg), dtype=int)
    for r, c in base_pts:
        out[int(r), int(c)] = int(base_sprite[int(r), int(c)])

    for r2, c2 in plug_pts:
        src = int(plug_sprite[int(r2 - tr), int(c2 - tc)])
        cur = int(out[int(r2), int(c2)])
        if cur != int(bg):
            assert int(cur) in conn_colors or int(src) in conn_colors
        out[int(r2), int(c2)] = int(src)

    out[(out != int(bg)) & (out != int(obj))] = int(obj)
    return _crop_nonbg(out, int(bg))


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg, obj = 1, 4

    comps = _extract_components(g, int(bg))
    assert len(comps) == 2
    assert int(comps[0]["bbox"][0] + comps[0]["bbox"][1]) == 0
    base_sprite = comps[0]["sprite"]
    plug_sprite = comps[1]["sprite"]

    runs_base = _connector_runs(base_sprite, int(bg), int(obj))
    runs_plug = _connector_runs(plug_sprite, int(bg), int(obj))
    assert len(runs_base) == 2 and len(runs_plug) == 2

    base_colors = {int(run["color"]) for run in runs_base}
    plug_colors = {int(run["color"]) for run in runs_plug}
    shared = sorted(base_colors & plug_colors)
    assert len(shared) == 1
    match_color = int(shared[0])

    return _assemble_output(base_sprite, plug_sprite, bg=int(bg), obj=int(obj), match_color=int(match_color))


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


def _valid_segments(mask: np.ndarray, mode: str, conn_len: int) -> list[dict]:
    h, w = map(int, mask.shape)
    segs: list[dict] = []
    for r in range(h):
        for c0 in range(0, int(w - conn_len + 1)):
            cols = list(range(int(c0), int(c0 + conn_len)))
            if not all(bool(mask[int(r), int(c)]) for c in cols):
                continue
            if mode == "base":
                if int(r) == 0:
                    continue
                if not all(bool(mask[int(r - 1), int(c)]) for c in cols):
                    continue
                if not all(int(r + 1) >= h or not bool(mask[int(r + 1), int(c)]) for c in cols):
                    continue
            else:
                if int(r + 1) >= h:
                    continue
                if not all(bool(mask[int(r + 1), int(c)]) for c in cols):
                    continue
                if not all(int(r) == 0 or not bool(mask[int(r - 1), int(c)]) for c in cols):
                    continue
            segs.append(
                {
                    "row": int(r),
                    "c0": int(c0),
                    "c1": int(c0 + conn_len - 1),
                    "pts": {(int(r), int(c)) for c in cols},
                }
            )
    return segs


def _segments_compatible(a: dict, b: dict) -> bool:
    if set(a["pts"]) & set(b["pts"]):
        return False
    if int(a["row"]) == int(b["row"]):
        left = min(int(a["c1"]), int(b["c1"]))
        right = max(int(a["c0"]), int(b["c0"]))
        if right <= left + 1:
            return False
    return True


def _paint_segment(arr: np.ndarray, seg: dict, color: int) -> None:
    for r, c in seg["pts"]:
        arr[int(r), int(c)] = int(color)


def _build_base(
    active_len: int,
    distract_len: int,
    match_color: int,
    extra_color: int,
) -> tuple[np.ndarray, dict]:
    bg, obj = 1, 4
    min_w = max(4, int(active_len + distract_len + 1), int(max(active_len, distract_len)) + 1)
    for _ in range(400):
        base_w = _rng.randint(int(min_w), 10)
        base_h = _rng.randint(3, 7)
        slots = []
        for a0 in range(0, int(base_w - active_len + 1)):
            a1 = int(a0 + active_len - 1)
            for d0 in range(0, int(base_w - distract_len + 1)):
                d1 = int(d0 + distract_len - 1)
                if max(int(a0), int(d0)) <= min(int(a1), int(d1)):
                    continue
                if int(d0) == int(a1) + 1 or int(a0) == int(d1) + 1:
                    continue
                if int(a0) == 0 and int(base_w) > int(active_len):
                    continue
                slots.append((int(a0), int(d0)))
        if not slots:
            continue
        active_c0, distract_c0 = _rng.choice(slots)
        active_c1 = int(active_c0 + active_len - 1)
        distract_c1 = int(distract_c0 + distract_len - 1)
        active_height = _rng.randint(2, int(base_h))
        distract_height = _rng.randint(2, int(base_h))

        heights: list[int] = []
        for c in range(int(base_w)):
            if int(active_c0) <= c <= int(active_c1):
                heights.append(int(active_height))
                continue
            if int(distract_c0) <= c <= int(distract_c1):
                heights.append(int(distract_height))
                continue
            choices = list(range(1, int(base_h) + 1))
            if c in (int(active_c0) - 1, int(active_c1) + 1) and int(active_height) in choices:
                choices.remove(int(active_height))
            if c in (int(distract_c0) - 1, int(distract_c1) + 1) and int(distract_height) in choices:
                if int(distract_height) in choices:
                    choices.remove(int(distract_height))
            if not choices:
                choices = [1]
            heights.append(int(_rng.choice(choices)))

        if sum(int(h) for h in heights) > 29:
            continue

        base_grid = np.full((int(base_h), int(base_w)), int(bg), dtype=int)
        for c, h in enumerate(heights):
            base_grid[: int(h), int(c)] = int(obj)

        active_seg = {
            "row": int(active_height - 1),
            "c0": int(active_c0),
            "c1": int(active_c1),
            "pts": {(int(active_height - 1), int(c)) for c in range(int(active_c0), int(active_c1) + 1)},
        }
        distract_seg = {
            "row": int(distract_height - 1),
            "c0": int(distract_c0),
            "c1": int(distract_c1),
            "pts": {(int(distract_height - 1), int(c)) for c in range(int(distract_c0), int(distract_c1) + 1)},
        }
        _paint_segment(base_grid, active_seg, int(match_color))
        _paint_segment(base_grid, distract_seg, int(extra_color))

        runs = _connector_runs(base_grid, int(bg), int(obj))
        if len(runs) != 2:
            continue
        return base_grid, active_seg
    raise ValueError("base-build-failed")


def _build_plug(
    active_len: int,
    distract_len: int,
    match_color: int,
    extra_color: int,
    active_base: dict,
    base_grid: np.ndarray,
) -> np.ndarray:
    bg, obj = 1, 4
    base_h, base_w = map(int, base_grid.shape)
    base_heights = [int(np.sum(base_grid[:, int(c)] != int(bg))) for c in range(int(base_w))]
    dock_r0 = int(active_base["row"] - 1)
    dock_c0 = int(active_base["c0"])
    active_cols = set(range(0, int(active_len)))
    min_distr_c0 = max(int(active_len + 1), int(base_w - dock_c0))

    for _ in range(400):
        plug_h = _rng.randint(3, 7)
        distract_row = _rng.randint(1, int(plug_h - 2))
        plug_w_min = max(int(min_distr_c0 + distract_len), int(active_len + distract_len + 1))
        if plug_w_min > 10:
            continue
        plug_w = _rng.randint(int(plug_w_min), 10)
        distract_c0 = _rng.randint(int(min_distr_c0), int(plug_w - distract_len))
        distract_c1 = int(distract_c0 + distract_len - 1)

        starts: list[int] = []
        ok = True
        for c in range(int(plug_w)):
            if c in active_cols:
                starts.append(0)
                continue
            if int(distract_c0) <= c <= int(distract_c1):
                starts.append(int(distract_row))
                continue
            low = 1
            cc = int(dock_c0 + c)
            if 0 <= cc < int(base_w):
                low = max(int(low), int(base_heights[int(cc)] - dock_r0))
            if low > int(plug_h - 1):
                ok = False
                break
            choices = list(range(int(low), int(plug_h)))
            if c in (int(distract_c0) - 1, int(distract_c1) + 1) and int(distract_row) in choices:
                choices.remove(int(distract_row))
            if not choices:
                ok = False
                break
            starts.append(int(_rng.choice(choices)))
        if not ok:
            continue

        pixel_count = sum(int(plug_h - s) for s in starts)
        if pixel_count < 5 or pixel_count > 29:
            continue

        plug_grid = np.full((int(plug_h), int(plug_w)), int(bg), dtype=int)
        for c, start in enumerate(starts):
            plug_grid[int(start) :, int(c)] = int(obj)

        active_seg = {
            "row": 0,
            "c0": 0,
            "c1": int(active_len - 1),
            "pts": {(0, int(c)) for c in range(int(active_len))},
        }
        distract_seg = {
            "row": int(distract_row),
            "c0": int(distract_c0),
            "c1": int(distract_c1),
            "pts": {(int(distract_row), int(c)) for c in range(int(distract_c0), int(distract_c1) + 1)},
        }

        illegal = False
        for r, c in np.argwhere(plug_grid != int(bg)):
            rr = int(dock_r0 + r)
            cc = int(dock_c0 + c)
            if 0 <= rr < int(base_h) and 0 <= cc < int(base_w):
                base_val = int(base_grid[int(rr), int(cc)])
                if base_val == int(obj):
                    allowed = (
                        int(r) == 0
                        and int(c) in active_cols
                        and int(active_base["c0"]) <= int(cc) <= int(active_base["c1"])
                        and int(rr) == int(active_base["row"] - 1)
                    )
                    if not allowed:
                        illegal = True
                        break
                elif base_val != int(bg):
                    allowed = (
                        int(r) == 1
                        and int(c) in active_cols
                        and (int(rr), int(cc)) in set(active_base["pts"])
                        and base_val == int(match_color)
                    )
                    if not allowed:
                        illegal = True
                        break
        if illegal:
            continue

        _paint_segment(plug_grid, active_seg, int(match_color))
        _paint_segment(plug_grid, distract_seg, int(extra_color))
        runs = _connector_runs(plug_grid, int(bg), int(obj))
        if len(runs) != 2:
            continue
        return plug_grid
    raise ValueError("plug-build-failed")


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
            if int(pr + pc) == 0:
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


def _gen_attempt() -> tuple[np.ndarray, dict] | None:
    bg, obj = 1, 4
    palette = [2, 3, 5, 6, 7, 8, 9]
    match_color, base_extra_color, plug_extra_color = _rng.sample(palette, 3)
    active_base_len, active_plug_len = _rng.sample([1, 2, 3], 2)
    base_distr_len = int(active_plug_len)
    plug_distr_len = int(active_base_len)

    try:
        base_grid, active_base = _build_base(
            int(active_base_len),
            int(base_distr_len),
            int(match_color),
            int(base_extra_color),
        )
        plug_grid = _build_plug(
            int(active_plug_len),
            int(plug_distr_len),
            int(match_color),
            int(plug_extra_color),
            active_base,
            base_grid,
        )
    except ValueError:
        return None

    if int(np.sum(base_grid != int(bg))) < 5 or int(np.sum(plug_grid != int(bg))) < 5:
        return None

    base_sprite = _crop_nonbg(base_grid, int(bg))
    plug_sprite = _crop_nonbg(plug_grid, int(bg))
    base_runs = _connector_runs(base_sprite, int(bg), int(obj))
    plug_runs = _connector_runs(plug_sprite, int(bg), int(obj))
    if len(base_runs) != 2 or len(plug_runs) != 2:
        return None

    try:
        expected = _assemble_output(
            base_sprite,
            plug_sprite,
            bg=int(bg),
            obj=int(obj),
            match_color=int(match_color),
        )
    except Exception:
        return None
    if expected.size == 0 or not np.any(expected != int(bg)):
        return None

    bh, bw = map(int, base_sprite.shape)
    ph, pw = map(int, plug_sprite.shape)
    canvas_h = _rng.randint(max(7, max(int(bh), int(ph)) + 2), 11)
    canvas_w = _rng.randint(max(4, max(int(bw), int(pw)) + 1), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    positions = _valid_input_positions(int(canvas_h), int(canvas_w), base_sprite, plug_sprite, int(bg))
    if not positions:
        return None
    pr, pc = _rng.choice(positions)

    canvas = np.full((int(canvas_h), int(canvas_w)), int(bg), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=bg)
    paste(canvas, plug_sprite, int(pr), int(pc), transparent=bg)
    if np.sum(canvas == int(bg)) <= canvas.size // 2:
        return None

    try:
        actual = main(canvas)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    info = {
        "match_color": int(match_color),
        "active_base_len": int(active_base_len),
        "active_plug_len": int(active_plug_len),
        "base_distr_len": int(base_distr_len),
        "plug_distr_len": int(plug_distr_len),
    }
    return canvas, info


def generate_input() -> np.ndarray:
    for _ in range(2000):
        out = _gen_attempt()
        if out is not None:
            return out[0]
    raise RuntimeError("generate_input: failed after 2000 attempts")
