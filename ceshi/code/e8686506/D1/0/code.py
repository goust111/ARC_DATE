# Variant of e8686506
# Dimension: D1 — 匹配条件：边缘对齐拼合 vs 其他匹配逻辑 (match_cond)
# Change: s.edge.position aligns with d.adjacent_piece.edge — 碎片的边缘位置与... → 颜色匹配决定相邻关系

"""
Design analysis for the D1 variant
==================================

variant_spec:
  core_change:
    from: 原题主要依赖碎片形状与洞轮廓的几何贴合来唯一确定相邻关系
    to: 碎片主体几何形状故意重复，真正决定上下相邻顺序的是边缘提示色

  a_concretization_and_trace:
    objects:
      - 输出仍是竖向矩形，保持严格左右镜像。
      - 输入中保留一个已经摆成输出外框位置的矩形边框碎片。
      - 内部碎片不再是不规则洞块，而是若干个相同主体形状的横条：
        主体恒为 1x3 纯色条；其上边中心和下边中心各外挂一个 1 像素提示色。
      - 外框碎片在其内侧顶部/底部中心也各有一个提示色像素，作为链的起点和终点。
    behavior_trace:
      original_local_step: |
        原题在某个内部块上，solver 会枚举它能塞进哪些洞；洞形状和边框锁定使唯一位置出现。
      variant_local_step: |
        现在所有内部块的主体都同形（1x3），几何上都能放进任意一行；
        solver 改为读取“上提示色/下提示色”，从外框顶部锚点开始沿颜色链逐行串下去。
      difference: |
        旧逻辑的关键信息来自“这个形状能卡进哪里”；新逻辑的关键信息来自
        “哪个碎片的上提示色等于当前需要的连接色”。

  b_complexity_calibration:
    - 若仍使用不规则洞形，颜色提示会沦为装饰，因此这里主动把内部主体统一成重复的 1x3 横条。
    - 为避免过度简化成纯颜色列表，仍保留可见的外框碎片和矩形装配场景。
    - 连接色使用一条长度为 interior_rows+1 的唯一链；每个样例里有 7-9 个连接关系，重复支撑“按颜色接龙”这条规则。

  c_implicit_assumption_audit:
    main_old_assumptions:
      - 非背景同色连通块就是一个 piece。
      - piece 只有单色，因此颜色本身就是 piece 身份。
      - placement validity 由目标矩形内的空位、镜像约束、边框锁定和同色不接触给出。
    why_broken_now:
      - 变体 piece 是“单色主体 + 异色提示像素”的多色连通块，不能再按同色提取 piece。
      - 内部横条几何完全重复，若继续按原先的 can_place/propagation，会得到大量等价位置。
      - “同色不 4 邻接”不再是核心，因为提示色像素只存在输入、输出中会被剥离。
    new_assumptions_and_guards:
      - 每个输入 component 都有唯一多数色，且该多数色就是主体色。generator 通过让主体面积 >=3、提示像素每色仅 1 个来保证。
      - 只有一个主体为矩形边框的 component。generator 固定只生成一个外框。
      - 每个内部条形 piece 的主体都恰为 1x3，提示像素位于主体上下中心。generator 固定用该模板生成。
      - 顶部锚点颜色唯一对应第一条，底部锚点颜色唯一对应最后一条。generator 生成一条无分叉链。

  d_placement_legality:
    - generator 先构造最终输出，再把外框放入输入画布的一个矩形 bbox。
    - 整个 bbox 会先被临时阻塞，内部横条只能散落在 bbox 外，避免和外框/其空腔重叠。
    - 横条 patch 都是 3x3，使用 random_free_location 且带 padding，保证互不重叠。
    - 输出时会剥离提示像素，只保留主体，因此 solver 的重建结果与设计输出一致。

  e_visual_observability:
    - 人类可直接看到：很多内部碎片主体完全同形，单靠“塞形状”无法决定顺序；
      同时每条横条上下都有一个小色点，外框内侧也有起止色点，颜色链很显眼。
    - 不使用任何基于颜色 id 的排序；solver 仅做“当前需要的色 -> 找上提示色相同的那条”。

  f_visual_cleanliness:
    - 提示像素只占主体上下中心单点，不覆盖主体大面积区域，输入易读。
    - 输出中提示像素全部消失，只留下整洁的边框 + 水平色带。
    - 外框固定、内部条形散落在外框外，不会在同一区域多层叠加。

  g_color_inference:
    - 输出颜色数控制在 4-6，和原题接近；提示色可多，但它们总以“小连接点”角色重复出现。
    - 连接逻辑是“相同提示色接到一起”，而不是隐藏映射或按颜色编号计算，因此 2-3 组样例足以归纳。
"""

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_WIDTH = 5
_HEIGHTS = (9, 10)
_OUTPUT_N_COLORS_MIN = 4
_OUTPUT_N_COLORS_MAX = 6
_CANVAS_MIN = 15
_CANVAS_MAX = 17
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 200


def _normalize(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), coords


def _is_border_ring(coords: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    if h < 3 or w < 3:
        return False
    expected = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    return set(coords) == expected


def _component_info(grid: np.ndarray, pts: np.ndarray) -> dict:
    pts_list = [(int(r), int(c)) for r, c in pts]
    vals = [int(grid[r, c]) for r, c in pts_list]
    counts = Counter(vals)
    body_color, body_count = counts.most_common(1)[0]
    assert body_count >= 3, "each component must have a visible majority body color"

    body_pts = np.array([(r, c) for r, c in pts_list if int(grid[r, c]) == int(body_color)], dtype=int)
    h, w, body_coords = _normalize(body_pts)
    br0, bc0, _, _ = bbox_coords(body_pts)

    hints = []
    for r, c in pts_list:
        col = int(grid[r, c])
        if col == int(body_color):
            continue
        hints.append({"r": int(r - br0), "c": int(c - bc0), "col": col})

    return {
        "body_color": int(body_color),
        "body_h": int(h),
        "body_w": int(w),
        "body_coords": body_coords,
        "hints": hints,
    }


def _parse_input(grid: np.ndarray) -> tuple[dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=4)
    infos = [_component_info(g, pts) for pts in comps]

    frame = None
    rows = []
    for info in infos:
        if _is_border_ring(info["body_coords"], info["body_h"], info["body_w"]):
            assert frame is None, "frame must be unique"
            frame = info
            continue

        assert info["body_h"] == 1 and info["body_w"] == 3, "internal pieces must be 1x3 body strips"
        assert set(info["body_coords"]) == {(0, 0), (0, 1), (0, 2)}
        hints = sorted(info["hints"], key=lambda x: (x["r"], x["c"]))
        assert len(hints) == 2, "each strip must expose exactly two hint pixels"
        assert (hints[0]["r"], hints[0]["c"]) == (-1, 1), "top hint must sit above strip center"
        assert (hints[1]["r"], hints[1]["c"]) == (1, 1), "bottom hint must sit below strip center"
        rows.append(
            {
                "body_color": int(info["body_color"]),
                "top": int(hints[0]["col"]),
                "bottom": int(hints[1]["col"]),
            }
        )

    assert frame is not None, "missing frame component"
    fh, fw = int(frame["body_h"]), int(frame["body_w"])
    assert fw == _WIDTH and fh in _HEIGHTS, "frame dimensions must define the output"
    frame_hints = sorted(frame["hints"], key=lambda x: (x["r"], x["c"]))
    assert len(frame_hints) == 2, "frame must expose top/bottom anchors"
    assert (frame_hints[0]["r"], frame_hints[0]["c"]) == (1, fw // 2)
    assert (frame_hints[1]["r"], frame_hints[1]["c"]) == (fh - 2, fw // 2)
    frame["top_anchor"] = int(frame_hints[0]["col"])
    frame["bottom_anchor"] = int(frame_hints[1]["col"])
    return frame, rows


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered strip-fragments into a left-right symmetric rectangle by matching hint colors.

    Input structure:
    - bg is the dominant color.
    - One multicolor component already sketches the output border: its majority color is the frame body, and two minority pixels inside the frame mark the top/bottom chain anchors.
    - The remaining components are repeated 1x3 body strips; each has one colored hint pixel above its center and one below its center.

    Rule:
    - Treat each non-bg connected component as one piece; within a piece, the majority color is the body and minority pixels are edge hints.
    - The frame body gives the output size. Every 1x3 strip fills one interior row of that frame.
    - Start from the frame's top anchor color, repeatedly choose the strip whose upper hint matches the current color, place it on the next interior row, then continue with that strip's lower hint color.

    Key insight: The internal strips are geometrically interchangeable, so shape alone does not order them; the only reliable assembly signal is the color chain running from the frame's top hint to its bottom hint.
    """
    frame, rows = _parse_input(grid)
    H, W = int(frame["body_h"]), int(frame["body_w"])
    assert len(rows) == H - 2, "the number of strips must match the frame interior height"

    by_top = {}
    for row in rows:
        assert row["top"] not in by_top, "each incoming hint color must identify a unique strip"
        by_top[row["top"]] = row

    out = np.full((H, W), int(get_bg(np.asarray(grid, dtype=int))), dtype=int)
    frame_color = int(frame["body_color"])
    out[0, :] = frame_color
    out[H - 1, :] = frame_color
    out[:, 0] = frame_color
    out[:, W - 1] = frame_color

    current = int(frame["top_anchor"])
    used_tops = set()
    for r in range(1, H - 1):
        assert current in by_top, "broken color chain: no strip matches the current hint"
        row = by_top[current]
        assert row["top"] not in used_tops, "color chain revisited an already placed strip"
        out[r, 1 : W - 1] = int(row["body_color"])
        used_tops.add(int(row["top"]))
        current = int(row["bottom"])

    assert current == int(frame["bottom_anchor"]), "color chain must terminate at the frame bottom anchor"
    assert len(used_tops) == len(rows), "every strip must be used exactly once"
    assert np.array_equal(out, out[:, ::-1]), "output must remain left-right symmetric"
    return out.astype(int)


def _choose_body_colors(palette: list[int], n_rows: int, top_links: list[int], bottom_links: list[int]) -> tuple[int, list[int]]:
    k = random.randint(_OUTPUT_N_COLORS_MIN, _OUTPUT_N_COLORS_MAX)
    body_pool = random.sample(palette, k=k)
    frame_candidates = [c for c in body_pool if c not in {top_links[0], bottom_links[-1]}]
    if not frame_candidates:
        frame_candidates = [c for c in palette if c not in {top_links[0], bottom_links[-1]}]
    frame_color = random.choice(frame_candidates)

    row_colors = []
    prev = None
    for i in range(n_rows):
        forbid = {top_links[i], bottom_links[i]}
        if i == 0:
            forbid.add(frame_color)
        if i == n_rows - 1:
            forbid.add(frame_color)
        if prev is not None:
            forbid.add(prev)
        opts = [c for c in body_pool if c not in forbid]
        if not opts:
            opts = [c for c in palette if c not in forbid]
        row_col = random.choice(opts)
        row_colors.append(int(row_col))
        prev = int(row_col)

    used_colors = len(set([frame_color, *row_colors]))
    if used_colors < _OUTPUT_N_COLORS_MIN:
        unique_targets = [c for c in body_pool if c != frame_color]
        idxs = list(range(n_rows))
        random.shuffle(idxs)
        for idx in idxs:
            forbid = {top_links[idx], bottom_links[idx]}
            if idx == 0 or idx == n_rows - 1:
                forbid.add(frame_color)
            if idx > 0:
                forbid.add(row_colors[idx - 1])
            if idx + 1 < n_rows:
                forbid.add(row_colors[idx + 1])
            candidates = [c for c in unique_targets if c not in forbid and c != row_colors[idx]]
            if candidates:
                row_colors[idx] = random.choice(candidates)
                used_colors = len(set([frame_color, *row_colors]))
                if used_colors >= _OUTPUT_N_COLORS_MIN:
                    break

    assert _OUTPUT_N_COLORS_MIN <= len(set([frame_color, *row_colors])) <= _OUTPUT_N_COLORS_MAX
    return int(frame_color), row_colors


def _build_output_and_patches() -> tuple[np.ndarray, int, np.ndarray, list[np.ndarray]]:
    H = random.choice(_HEIGHTS)
    W = _WIDTH
    n_rows = H - 2

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)

    link_colors = palette[: n_rows + 1]
    top_links = link_colors[:-1]
    bottom_links = link_colors[1:]
    frame_color, row_colors = _choose_body_colors(palette, n_rows, top_links, bottom_links)

    out = np.full((H, W), bg, dtype=int)
    out[0, :] = frame_color
    out[H - 1, :] = frame_color
    out[:, 0] = frame_color
    out[:, W - 1] = frame_color
    for r in range(1, H - 1):
        out[r, 1 : W - 1] = row_colors[r - 1]

    frame_patch = np.full((H, W), bg, dtype=int)
    frame_patch[0, :] = frame_color
    frame_patch[H - 1, :] = frame_color
    frame_patch[:, 0] = frame_color
    frame_patch[:, W - 1] = frame_color
    frame_patch[1, W // 2] = top_links[0]
    frame_patch[H - 2, W // 2] = bottom_links[-1]

    row_patches = []
    for row_col, top_col, bot_col in zip(row_colors, top_links, bottom_links):
        patch = np.full((3, 3), bg, dtype=int)
        patch[1, :] = row_col
        patch[0, 1] = top_col
        patch[2, 1] = bot_col
        row_patches.append(patch)

    return out, bg, frame_patch, row_patches


def _scatter(frame_patch: np.ndarray, row_patches: list[np.ndarray], bg: int) -> np.ndarray:
    H, W = frame_patch.shape
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    max_r = rows - H - 2
    max_c = cols - W - 2
    if max_r < 1 or max_c < 1:
        raise ValueError("canvas too small for frame")
    off_r = random.randint(1, max_r)
    off_c = random.randint(1, max_c)

    canvas[off_r : off_r + H, off_c : off_c + W] = -1
    for patch in row_patches:
        r0, c0 = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r0, c0, transparent=bg)

    canvas[off_r : off_r + H, off_c : off_c + W] = bg
    paste(canvas, frame_patch, off_r, off_c, transparent=bg)
    return canvas


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    out, bg, frame_patch, row_patches = _build_output_and_patches()
    try:
        inp = _scatter(frame_patch, row_patches, bg)
    except Exception:
        return None
    got = main(inp)
    if np.array_equal(got, out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid D1 variant input for e8686506")
