# Combo variant of 20270e3b (index 8)
# combo_spec: D5/2+D10/0
# Name: 双色精灵颜色交换

from __future__ import annotations

import random as _rng

import numpy as np

from arc_utils import bbox_coords, is_contiguous, label_mask, paste, random_free_location


"""
Design analysis for the combo variant
=====================================

design_analysis:
  fusion_rule:
    summary: |
      保留 20270e3b 的几何对接方式：识别左上角 base、识别另一个 plug，
      用匹配的水平 connector 对齐，让 plug 的 connector 落到 base connector 上方一行。
      融合后的唯一变化在着色阶段：
      1. base 的整只 sprite（含它自己的 connector 位置）改成 plug 的原 body 色；
      2. plug 的 body 改成 base 的原 body 色；
      3. 所有仍然可见的 connector 像素统一改成 base 的新 body 色。
      因为原题的几何里 base connector 会被 plug 的第一行 body 覆盖，所以真正会“存活”的
      connector 是上方那条 plug connector，它最终也必须变成 base 的新色，也就是 plug 的原色。
  a_concretization_and_trace:
    concrete_example: |
      例：bg=1，ctrl=7，base body=2，plug body=3，connector 长度=2。
      原题：dock 后所有 7 直接改成单一 obj=4，输出只剩一种前景色。
      D10/0 单变体：dock 后 base 仍是 2，plug body 仍是 3，幸存的上方 connector 改成 base 色 2。
      本融合：dock 后 base 整体变成 3；plug body 变成 2；幸存的上方 connector 也改成 base 新色 3。
      所以最终颜色分层是“下方 base 区域 = 3，上方 connector 条 = 3，connector 下方的 plug body = 2”。
      差异点不在平移，而在于 solver 必须同时追踪 base 原色、plug 原色、以及 swap 后的 base 新色。
  b_complexity_calibration:
    choices:
      - 仍只使用 4 种颜色：bg、ctrl、base_body、plug_body。
      - 强制两个 body 色不同，避免 swap 退化成无变化。
      - connector 仍固定唯一颜色 7，保证解题者先看出“怎么对接”，再看出“对接后如何换色”。
      - 不引入多种 connector 规则；connector 永远重着色到 base 的输出色，规律足够稳定。
  c_implicit_assumption_audit:
    original_assumptions_broken:
      - 原 solver 写死 obj=4，默认两个 sprite body 同色；融合后必须分别提取两个组件的唯一 body 色。
      - 原 solver 末尾 `out[out == ctrl] = obj` 假设 connector 统一变一个全局颜色；融合后应改为
        “connector -> base 输出色”，而这个输出色本身又依赖 color swap。
      - D5/2 的单变体如果按“对象归属”给 connector 着色，会让 plug connector 变成 plug 新色。
        这和本题顶层融合规格冲突；这里必须以 D10 的 connector 规则为准，即 connector 看 base 输出色，
        不是看 connector 属于谁。
      - 原 overlap 检查用颜色判断 `cur == ctrl or val == ctrl`。在双色情形下这太脆弱，
        必须改成基于几何身份检查：重叠只能发生在 base connector 或 plug connector。
    new_main_assumptions:
      - 输入恰有两个 8 连通前景组件。
      - 每个组件恰有一种非 bg/ctrl 的 body 色。
      - 两个 connector 规范化后形状一致，且都为单行水平短条。
      - base connector 下方无 body，plug connector 上方无 body。
      - dock 后唯一允许重叠的是“base connector 被 plug body 覆盖”。
  d_placement_legality:
    generator_policy: |
      仍采用 output-first：先构造满足几何约束的 docked 结构，再裁开并重新放回输入画布。
      base 固定贴在 (0,0)；plug 放在别处，确保两组件不重叠、不越界，且输入中只有这两只前景精灵。
      生成器在 docked 阶段显式排除非接缝重叠，并要求 plug 不得伸到 base 左边或上边，
      这样 tight crop 后 base 仍在输出左上角。
  e_visual_observability:
    rationale: |
      人类从样例里会先看到对接几何和原题一致，然后发现输出不是“保留双色”也不是“统一单色”：
      下方 base 区域拿到了上方精灵的原色，而上方 plug body 拿到了 base 原色；
      同时最上方那条 connector 颜色又跟下方 base 区域一致。三者合起来足以支持归纳。
  f_visual_cleanliness:
    rationale: |
      整体仍只有两层对象叠加：plug body 覆盖 base connector。颜色种类固定 4 个，输出只剩 2 个 body 色，
      且 connector 是短直条，不会制造难以分辨的纹理噪声。
  g_change_inferability:
    rationale: |
      颜色变化完全由结构角色决定：base 区域取 plug 原色，plug body 取 base 原色，connector 取 base 输出色。
      这里没有任意颜色映射表，也不依赖颜色数值大小。样例之间只要重复出现“两个 body 色不同”的情况，
      解题者就能从少量案例推断。
"""


BG = 1
CTRL = 7
BODY_PALETTE = (2, 3, 5, 6, 8, 9)


def _bbox_of_pts(pts: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = np.array(list(pts), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    return int(r0), int(c0), int(r1), int(c1)


def _conn_of(g: np.ndarray, pts: set[tuple[int, int]]) -> set[tuple[int, int]]:
    return {(int(r), int(c)) for (r, c) in pts if int(g[int(r), int(c)]) == int(CTRL)}


def _body_color_of(g: np.ndarray, pts: set[tuple[int, int]]) -> int:
    cols = {
        int(g[int(r), int(c)])
        for (r, c) in pts
        if int(g[int(r), int(c)]) not in (int(BG), int(CTRL))
    }
    assert len(cols) == 1
    return int(next(iter(cols)))


def _norm_conn(conn_pts: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    rs = [r for r, _c in conn_pts]
    cs = [c for _r, c in conn_pts]
    r0, c0 = int(min(rs)), int(min(cs))
    shape = {(int(r - r0), int(c - c0)) for r, c in conn_pts}
    return shape, (int(r0), int(c0))


def _extract_scene(
    g: np.ndarray,
) -> tuple[
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    set[tuple[int, int]],
    int,
    int,
    int,
    int,
]:
    lab, n = label_mask(g != int(BG), connectivity=8)
    comps = []
    for k in range(1, int(n) + 1):
        pts = {(int(r), int(c)) for r, c in np.argwhere(lab == int(k))}
        comps.append(pts)
    comps.sort(key=len, reverse=True)
    assert len(comps) == 2
    A, B = set(comps[0]), set(comps[1])

    connA = _conn_of(g, A)
    connB = _conn_of(g, B)
    assert connA and connB
    shpA, _ = _norm_conn(connA)
    shpB, _ = _norm_conn(connB)
    assert shpA == shpB

    r0a, c0a, _r1a, _c1a = _bbox_of_pts(A)
    r0b, c0b, _r1b, _c1b = _bbox_of_pts(B)
    assert int(r0a + c0a) == 0 or int(r0b + c0b) == 0
    if int(r0a + c0a) <= int(r0b + c0b):
        base, connD = A, connA
        plug, connP = B, connB
    else:
        base, connD = B, connB
        plug, connP = A, connA

    base_col = _body_color_of(g, base)
    plug_col = _body_color_of(g, plug)
    assert base_col != plug_col

    base_body = base - connD
    plug_body = plug - connP
    for r, c in connD:
        assert (int(r + 1), int(c)) not in base_body
    for r, c in connP:
        assert (int(r - 1), int(c)) not in plug_body

    _sp, (pr0, pc0) = _norm_conn(connP)
    _sd, (dr0, dc0) = _norm_conn(connD)
    tr = int((dr0 - 1) - pr0)
    tc = int(dc0 - pc0)
    return base, plug, connD, connP, base_col, plug_col, tr, tc


def _render_docked(
    base: set[tuple[int, int]],
    plug: set[tuple[int, int]],
    connD: set[tuple[int, int]],
    connP: set[tuple[int, int]],
    tr: int,
    tc: int,
    base_out_col: int,
    plug_body_out_col: int,
    connector_out_col: int,
) -> np.ndarray:
    plug_shift = {(int(r + tr), int(c + tc)) for r, c in plug}

    pts_all = base | plug_shift
    coords_all = np.array(list(pts_all), dtype=int)
    rr0, cc0, rr1, cc1 = map(int, bbox_coords(coords_all))

    base_coords = np.array(list(base), dtype=int)
    br0, bc0 = int(base_coords[:, 0].min()), int(base_coords[:, 1].min())
    assert rr0 == br0 and cc0 == bc0

    out = np.full((int(rr1 - rr0 + 1), int(cc1 - cc0 + 1)), int(BG), dtype=int)

    for r, c in base:
        out[int(r - rr0), int(c - cc0)] = int(base_out_col)

    for r, c in plug_shift:
        src = (int(r - tr), int(c - tc))
        if int(out[int(r - rr0), int(c - cc0)]) != int(BG):
            assert (int(r), int(c)) in connD or src in connP
        color = int(connector_out_col) if src in connP else int(plug_body_out_col)
        out[int(r - rr0), int(c - cc0)] = color

    nz = np.argwhere(out != int(BG))
    assert nz.size > 0
    r0, c0, r1, c1 = bbox_coords(nz)
    return out[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Dock the plug above the base connector, then apply the fused color rule:
    swap the two body colors and recolor surviving connector pixels to the
    base sprite's new body color.
    """
    g = np.array(grid, dtype=int)
    base, plug, connD, connP, base_col, plug_col, tr, tc = _extract_scene(g)

    # Fusion:
    # - base body/output color becomes plug's original body color
    # - plug body becomes base's original body color
    # - visible connector pixels become the base's new body color (= plug original)
    return _render_docked(
        base,
        plug,
        connD,
        connP,
        tr,
        tc,
        base_out_col=plug_col,
        plug_body_out_col=base_col,
        connector_out_col=plug_col,
    )


def _grow_shape(mask: np.ndarray, target: int, frontier: list[tuple[int, int]]) -> None:
    """Grow a connected 4-neighbor shape from frontier until target pixels."""
    count = int(mask.sum())
    while count < int(target) and frontier:
        idx = _rng.randint(0, len(frontier) - 1)
        r, c = frontier[idx]
        h, w = mask.shape
        nbrs = [
            (r + dr, c + dc)
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
            if 0 <= r + dr < h and 0 <= c + dc < w and not mask[r + dr, c + dc]
        ]
        if not nbrs:
            frontier[idx] = frontier[-1]
            frontier.pop()
            continue
        nr, nc = _rng.choice(nbrs)
        mask[nr, nc] = True
        count += 1
        frontier.append((nr, nc))


def _thin(mask: np.ndarray, keep: set[tuple[int, int]] | None = None) -> None:
    """Remove interior pixels while preserving 8-connectivity."""
    h, w = mask.shape
    keep = keep or set()
    pixels = list(zip(*np.where(mask)))
    _rng.shuffle(pixels)
    for r, c in pixels:
        if (int(r), int(c)) in keep:
            continue
        adj_bg = any(
            not (0 <= r + dr < h and 0 <= c + dc < w) or not mask[r + dr, c + dc]
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
        )
        if adj_bg:
            continue
        mask[r, c] = False
        if not is_contiguous(mask, background=0, connectivity=8):
            mask[r, c] = True


def _crop_non_bg(grid: np.ndarray) -> np.ndarray | None:
    nz = np.argwhere(grid != int(BG))
    if nz.size == 0:
        return None
    r0, c0, r1, c1 = bbox_coords(nz)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _gen_attempt() -> np.ndarray | None:
    base_col, plug_col = _rng.sample(BODY_PALETTE, 2)

    base_h = _rng.randint(2, 7)
    base_w = _rng.randint(3, 10)
    base_target = _rng.randint(5, min(29, base_h * base_w))
    base = np.zeros((base_h, base_w), dtype=bool)
    base[0, 0] = True
    _grow_shape(base, base_target, [(0, 0)])
    if not is_contiguous(base, background=0, connectivity=8):
        return None
    _thin(base)

    conn_len = _rng.randint(1, min(3, base_w))
    candidates = []
    for r in range(base_h):
        for c0 in range(base_w - conn_len + 1):
            cols = list(range(c0, c0 + conn_len))
            if not all(base[r, c] for c in cols):
                continue
            if r == 0:
                continue
            if not all(base[r - 1, c] for c in cols):
                continue
            candidates.append((r, cols))
    if not candidates:
        return None
    conn_row, conn_cols = _rng.choice(candidates)

    base_grid = np.full((base_h, base_w), int(BG), dtype=int)
    base_grid[base] = int(base_col)
    for c in conn_cols:
        base_grid[conn_row, c] = int(CTRL)
    if not is_contiguous(base_grid != int(BG), background=0, connectivity=8):
        return None

    plug_h = _rng.randint(2, 7)
    plug_w = _rng.randint(max(conn_len, 2), min(10, base_w + 4))
    plug_conn_c0 = _rng.randint(0, plug_w - conn_len)
    dc_offset = int(conn_cols[0] - plug_conn_c0)
    if dc_offset < 0:
        return None

    plug = np.zeros((plug_h, plug_w), dtype=bool)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug[0, c] = True
        if plug_h > 1:
            plug[1, c] = True

    plug_max = min(29, plug_h * plug_w)
    if plug_max < 5:
        return None
    plug_target = _rng.randint(5, plug_max)
    _grow_shape(plug, plug_target, list(zip(*np.where(plug))))
    if not is_contiguous(plug, background=0, connectivity=8):
        return None
    keep_plug = {(0, c) for c in range(plug_conn_c0, plug_conn_c0 + conn_len)}
    _thin(plug, keep=keep_plug)

    for r in range(plug_h):
        for c in range(plug_w):
            if not plug[r, c]:
                continue
            dr = int(conn_row + r)
            dc = int(dc_offset + c)
            if 0 <= dr < base_h and 0 <= dc < base_w and base_grid[dr, dc] == int(base_col):
                if r == 0 and plug_conn_c0 <= c < plug_conn_c0 + conn_len:
                    continue
                return None

    plug_grid = np.full((plug_h, plug_w), int(BG), dtype=int)
    plug_grid[plug] = int(plug_col)
    for c in range(plug_conn_c0, plug_conn_c0 + conn_len):
        plug_grid[0, c] = int(CTRL)
        if plug_h < 2 or plug_grid[1, c] != int(plug_col):
            return None
    if not is_contiguous(plug_grid != int(BG), background=0, connectivity=8):
        return None

    if int(np.sum(base_grid != int(BG))) < 5 or int(np.sum(plug_grid != int(BG))) < 5:
        return None

    base_sprite = _crop_non_bg(base_grid)
    plug_sprite = _crop_non_bg(plug_grid)
    if base_sprite is None or plug_sprite is None:
        return None

    bh, bw = base_sprite.shape
    ph, pw = plug_sprite.shape
    canvas_h = _rng.randint(max(7, max(bh, ph) + 2), 11)
    canvas_w = _rng.randint(max(3, max(bw, pw)), 13)
    if bh > canvas_h or bw > canvas_w or ph > canvas_h or pw > canvas_w:
        return None

    canvas = np.full((canvas_h, canvas_w), int(BG), dtype=int)
    paste(canvas, base_sprite, 0, 0, transparent=int(BG))
    try:
        pr, pc = random_free_location(canvas, plug_sprite, transparent=int(BG), padding=0)
    except ValueError:
        return None
    if int(pr + pc) == 0:
        return None
    paste(canvas, plug_sprite, pr, pc, transparent=int(BG))

    if np.sum(canvas == int(BG)) <= canvas.size // 2:
        return None

    try:
        combo_out = main(canvas)
        base_pts, plug_pts, connD, connP, seen_base_col, seen_plug_col, tr, tc = _extract_scene(canvas)
        d10_like_out = _render_docked(
            base_pts,
            plug_pts,
            connD,
            connP,
            tr,
            tc,
            base_out_col=seen_base_col,
            plug_body_out_col=seen_plug_col,
            connector_out_col=seen_base_col,
        )
    except Exception:
        return None

    if combo_out.size == 0 or not np.any(combo_out != int(BG)):
        return None

    # Guard against degeneration:
    # - combo output must visibly keep exactly the two sampled body colors
    # - combo output must differ from the non-swap D10-only rendering on the same input
    out_colors = {int(x) for x in np.unique(combo_out)} - {int(BG)}
    if out_colors != {int(base_col), int(plug_col)}:
        return None
    if np.array_equal(combo_out, d10_like_out):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(1000):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 1000 attempts")
