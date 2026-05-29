# Combo variant of e8686506 (index 42)
# combo_spec: D1/0+D2/1+D11/0
# Name: 颜色匹配+锚点碎片+两阶段拼合

"""
Design analysis for the combo variant
=====================================

combo_summary:
  core_change:
    from: 原题把单色碎片直接塞回一个左右对称矩形，主要靠几何贴合与边框约束。
    to: 输入里的真实拼装单位变成“带锚点的多色条形碎片”。这些碎片要先在局部
      通过边缘颜色接龙组装成 3 个子组件，再利用子组件中的锚点列对齐到外框内部
      的目标标记，最终得到左右镜像对称的矩形。

  a_concretization_and_trace:
    objects:
      - 输出固定为 5 列竖向矩形，外层是一圈单色边框。
      - 内部恰有 3 个纵向堆叠的子组件；每个子组件高 2 或 3，总高度拼成整个内腔。
      - 子组件的每一行都来自一个同形 strip：中间一行是 `[body, anchor, body]`，
        上下各外挂一个连接提示色像素，所以输入中的每个碎片都是 3x3、5 个前景像素。
      - 外框内部预先放着目标锚点标记，只显示每个子组件最终中心列应出现的锚点颜色。
    behavior_trace:
      original_local_step: |
        原题先把每个单色连通块当作最终 piece，再直接尝试塞进目标矩形；边框和洞形状
        会把每块锁到唯一位置。
      combo_local_step: |
        现在所有散落 strip 的几何形状完全相同，不能直接塞回矩形。solver 先按锚点色
        把 strip 划分到 3 个候选子组件中，再在每个组内读取“上提示色 -> 下提示色”
        的链条，把 2-3 个 strip 排成竖直子组件；随后读取外框里的锚点标记块，把这 3 个
        子组件放到对应高度区间。
      difference: |
        原题的一步几何装配被拆成了两步：先做颜色链驱动的局部重建，再做锚点驱动的全局
        对位。单看形状时，第一阶段和第二阶段都会出现等价候选，必须把两套标记一起用上。

  b_complexity_calibration:
    - 所有 strip 共享同一个 3x3 模板，避免“靠形状就能拼完”，让 D1 的颜色匹配变成必要信息。
    - 子组件高度只允许 2 或 3，始终是 3 个子组件；每个样例都会重复出现 2-3 次相同的
      连接模板，解题者能在少量样例里看出“先接龙、再对锚点”的流程。
    - 至少两个子组件会有相同高度，因此第二阶段不能只看子组件尺寸，必须看锚点颜色。
    - 连接提示色只用 4 个固定角色色：TOP / MID1 / MID2 / BOT，在不同组之间复用，
      避免颜色自由度过大。

  c_implicit_assumption_audit:
    main_old_assumptions:
      - `components_mask(g == col, ...)` 可以把同色块直接视作 piece。
      - piece 是单色的，所以颜色既代表身份，也代表输出颜色。
      - 只要知道总前景面积，就能从因子分解反推出输出矩形。
      - 组装是一步完成的，不需要先恢复中间子组件。
    why_broken_now:
      - strip 是多色 4 连通块；按颜色分量提取会把一个 piece 撕开。
      - 输入里同形 strip 大量重复，几何位置不再唯一。
      - 外框内部单独放入了锚点目标标记，前景面积不再等于最终输出面积。
      - 连接提示色在不同子组件间复用，若不先按锚点色分组，颜色链会互相干扰。
    new_guards:
      - generator 固定每个散落 strip 恰为 3x3 模板，主体色只出现在左右两格，
        中心格是锚点色，上下中心格是连接提示色。
      - generator 固定只有一个单色外框环，其 bbox 就是输出尺寸。
      - 外框内只出现中心列锚点标记；它们可能上下相邻、形成多色竖链，因此 solver
        不能再把“内部非背景连通块”当作 piece，而要在找到外框后直接读取内腔像素。
      - 每个锚点色只对应一个子组件，且目标标记块高度与该组 strip 数完全一致。

  d_placement_legality:
    - 生成器先构造目标输出，再把外框放进大画布的一个 bbox；该 bbox 在散落阶段会被整体阻塞。
    - 所有 strip patch 都用 `random_free_location(..., padding=1)` 放在 bbox 外，保证互不重叠、
      不越界，也不会和外框或内部标记发生接触。
    - 外框内部只绘制最终锚点列，不绘制主体 body，避免和散落 strip 覆盖混杂。

  e_visual_observability:
    - 人眼能直接看到：外框已经给出最终矩形；外框里只有一列彩色锚点标记；外框外散落着许多
      完全同形的小条，每条同时带“上下连接点 + 中心锚点”。
    - 从 2-3 个样例里可以归纳出：先把同锚点色的小条按上下提示色接成更高的竖条，
      再把竖条放回外框中同色锚点标记所在的高度段。
    - 整个规则不依赖颜色编号大小、排序或坐标奇偶等不可见信息。

  f_visual_cleanliness:
    - 输入中只有三层信息：单色外框、中心列锚点标记、散落 strip；三者区域分离，几乎不重叠。
    - 输出里 strip 的连接提示色全部消失，只留下干净的边框和三段锚点带。
    - strip 模板固定且稀疏，不会出现复杂轮廓相互遮挡。

  g_inference_of_changes:
    - 锚点色只负责“哪个子组件放到哪个目标段”，连接提示色只负责“组内上下谁接谁”，
      各自职责单一且重复出现。
    - 因为不同子组件经常同高同形，若忽略锚点会无法确定最终纵向顺序；若忽略连接提示色，
      组内 2-3 个同形 strip 也无法排序。两阶段信号都必须成立，才能避免退化回原题。
"""

from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location


_WIDTH = 5
_HEIGHT_PATTERNS = [(2, 2, 3), (2, 3, 3)]
_CANVAS_MIN = 18
_CANVAS_MAX = 21
_SCATTER_PADDING = 1
_GEN_MAX_RETRIES = 400


def _normalize_coords(pts: np.ndarray) -> tuple[int, int, tuple[tuple[int, int], ...]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    coords = tuple(sorted((int(r) - r0, int(c) - c0) for r, c in pts))
    return int(r1 - r0 + 1), int(c1 - c0 + 1), coords


def _is_border_ring(coords: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    if h < 3 or w != _WIDTH:
        return False
    expected = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    return set(coords) == expected


def _paint_patch(g: np.ndarray, pts: np.ndarray, bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in pts:
        patch[int(r) - r0, int(c) - c0] = int(g[int(r), int(c)])
    return patch, (int(r0), int(c0), int(r1), int(c1))


def _find_frame(g: np.ndarray, comps: list[np.ndarray], bg: int) -> tuple[np.ndarray, tuple[int, int, int, int], int]:
    frame_pts = None
    frame_bbox = None
    frame_color = None
    for pts in comps:
        patch, bbox = _paint_patch(g, pts, bg)
        h, w = patch.shape
        if w != _WIDTH or h < 3:
            continue
        border = {
            int(patch[r, c])
            for r in range(h)
            for c in range(w)
            if r in (0, h - 1) or c in (0, w - 1)
        }
        if len(border) != 1 or bg in border:
            continue
        candidate = next(iter(border))
        if all(int(patch[r, c]) == int(candidate) for r in range(h) for c in range(w) if r in (0, h - 1) or c in (0, w - 1)):
            assert frame_pts is None, "frame must be unique"
            frame_pts = pts
            frame_bbox = bbox
            frame_color = int(candidate)
    assert frame_pts is not None and frame_bbox is not None and frame_color is not None, "missing frame ring"
    return frame_pts, frame_bbox, int(frame_color)


def _parse_strip(g: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    patch, bbox = _paint_patch(g, pts, bg)
    assert patch.shape == (3, 3), "every scattered fragment must use the 3x3 strip template"
    expected = {(0, 1), (1, 0), (1, 1), (1, 2), (2, 1)}
    coords = {(int(r), int(c)) for r, c in np.argwhere(patch != bg)}
    assert coords == expected, "strip foreground layout must match the cross template"

    body_left = int(patch[1, 0])
    anchor = int(patch[1, 1])
    body_right = int(patch[1, 2])
    top = int(patch[0, 1])
    bottom = int(patch[2, 1])

    assert body_left == body_right != bg, "body color must occupy both side cells"
    assert anchor not in (bg, body_left), "anchor color must be distinct from body"
    assert top not in (bg, body_left, anchor), "top connector must be a distinct color"
    assert bottom not in (bg, body_left, anchor), "bottom connector must be a distinct color"

    return {
        "bbox": bbox,
        "body_color": int(body_left),
        "anchor_color": int(anchor),
        "top_color": int(top),
        "bottom_color": int(bottom),
    }


def _extract_targets(g: np.ndarray, frame_bbox: tuple[int, int, int, int], frame_color: int, bg: int) -> dict[int, list[int]]:
    r0, c0, r1, c1 = frame_bbox
    H = r1 - r0 + 1
    W = c1 - c0 + 1
    mid = W // 2
    targets: dict[int, list[int]] = defaultdict(list)

    for rr in range(H):
        for cc in range(W):
            v = int(g[r0 + rr, c0 + cc])
            if v in (bg, frame_color):
                continue
            assert 1 <= rr < H - 1, "target anchors must lie inside the frame"
            assert cc == mid, "target anchors must occupy the frame center column"
            targets[v].append(int(rr))

    assert len(targets) == 3, "there must be exactly three target anchor blocks"
    for col, rows in targets.items():
        rows.sort()
        assert 2 <= len(rows) <= 3, "each target block height must be 2 or 3"
        assert rows == list(range(rows[0], rows[0] + len(rows))), f"anchor rows for {col} must be contiguous"
    return {int(col): rows for col, rows in targets.items()}


def _assemble_group(strips: list[dict]) -> dict:
    assert 2 <= len(strips) <= 3, "each group must contain 2-3 fragments"
    anchor_colors = {int(s["anchor_color"]) for s in strips}
    body_colors = {int(s["body_color"]) for s in strips}
    assert len(anchor_colors) == 1, "all fragments in a group must share one anchor color"
    assert len(body_colors) == 1, "all fragments in a group must share one body color"

    by_top = {}
    bottoms = Counter()
    tops = Counter()
    for strip in strips:
        top = int(strip["top_color"])
        bot = int(strip["bottom_color"])
        assert top not in by_top, "each connector top color must be unique within a group"
        by_top[top] = strip
        tops[top] += 1
        bottoms[bot] += 1

    starts = [col for col in by_top if bottoms[col] == 0]
    ends = [col for col in bottoms if tops[col] == 0]
    assert len(starts) == 1 and len(ends) == 1, "each group must expose exactly one chain start and one chain end"
    start_color = int(starts[0])
    end_color = int(ends[0])

    ordered = []
    current = start_color
    used = set()
    while True:
        assert current in by_top, "broken connector chain"
        strip = by_top[current]
        key = (int(strip["anchor_color"]), int(strip["top_color"]), int(strip["bottom_color"]))
        assert key not in used, "connector chain must not cycle"
        ordered.append(strip)
        used.add(key)
        current = int(strip["bottom_color"])
        if current == end_color:
            break

    assert len(ordered) == len(strips), "every strip in a group must appear exactly once in the chain"
    return {
        "anchor_color": int(next(iter(anchor_colors))),
        "body_color": int(next(iter(body_colors))),
        "height": int(len(ordered)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a left-right symmetric rectangle by first chaining same-anchor fragments into subcomponents, then placing those subcomponents via anchor targets.

    Input structure:
    - The dominant color is background.
    - One monochrome border ring already sits at the final output size; inside that ring, only a center-column stack of colored anchor markers is pre-drawn.
    - Outside the ring, every real fragment is the same 3x3 multicolor strip template: two body pixels, one center anchor pixel, and one connector pixel above and below.

    Rule:
    - Detect the border ring to obtain the output box, then read the interior anchor-marker blocks from its center column.
    - Group scattered strips by their shared anchor color and, within each group, use the top/bottom connector colors to order the strips into a vertical 2-3 row subcomponent.
    - Place each assembled subcomponent into the frame so its anchor column matches the same-colored target block, and paint every row as `[body, anchor, body]`.

    Key insight: Every scattered strip has the same geometry, so neither local order nor final vertical placement can come from shape alone. Connector colors resolve the first-stage chaining, while the pre-drawn anchor markers resolve the second-stage placement.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=4)
    assert comps, "grid must contain foreground objects"

    frame_pts, frame_bbox, frame_color = _find_frame(g, comps, bg)
    _frame_r0, _frame_c0, frame_r1, frame_c1 = frame_bbox
    H = int(frame_r1 - frame_bbox[0] + 1)
    W = int(frame_c1 - frame_bbox[1] + 1)

    targets = _extract_targets(g, frame_bbox, frame_color, bg)

    frame_key = frozenset((int(r), int(c)) for r, c in frame_pts)
    strips = []
    for pts in comps:
        key = frozenset((int(r), int(c)) for r, c in pts)
        if key == frame_key:
            continue
        r0, c0, r1, c1 = bbox_coords(pts)
        inside_frame = (
            frame_bbox[0] <= int(r0)
            and int(r1) <= frame_bbox[2]
            and frame_bbox[1] <= int(c0)
            and int(c1) <= frame_bbox[3]
        )
        if inside_frame:
            continue
        strips.append(_parse_strip(g, pts, bg))

    groups: dict[int, list[dict]] = defaultdict(list)
    for strip in strips:
        groups[int(strip["anchor_color"])].append(strip)

    assert set(groups) == set(targets), "anchor colors of strips and target blocks must match exactly"
    assembled = {int(anchor): _assemble_group(group) for anchor, group in groups.items()}

    out = np.full((H, W), bg, dtype=int)
    out[0, :] = frame_color
    out[H - 1, :] = frame_color
    out[:, 0] = frame_color
    out[:, W - 1] = frame_color

    for anchor_color, band in assembled.items():
        rows = targets[int(anchor_color)]
        assert len(rows) == int(band["height"]), "target block height must equal assembled subcomponent height"
        for rr in rows:
            out[int(rr), 1] = int(band["body_color"])
            out[int(rr), 2] = int(anchor_color)
            out[int(rr), 3] = int(band["body_color"])

    assert not np.any(out < 0)
    assert np.array_equal(out, out[:, ::-1]), "output must stay left-right symmetric"
    return out.astype(int)


def _choose_roles(bg: int) -> dict[str, int]:
    colors = [c for c in range(10) if c != bg]
    random.shuffle(colors)
    return {
        "frame": int(colors[0]),
        "body": int(colors[1]),
        "anchor_a": int(colors[2]),
        "anchor_b": int(colors[3]),
        "anchor_c": int(colors[4]),
        "top": int(colors[5]),
        "mid1": int(colors[6]),
        "mid2": int(colors[7]),
        "bottom": int(colors[8]),
    }


def _build_output_and_parts() -> tuple[np.ndarray, int, np.ndarray, list[np.ndarray]]:
    heights = list(random.choice(_HEIGHT_PATTERNS))
    random.shuffle(heights)
    H = 2 + sum(heights)
    W = _WIDTH

    bg = random.randint(0, 9)
    roles = _choose_roles(bg)
    anchors = [roles["anchor_a"], roles["anchor_b"], roles["anchor_c"]]
    random.shuffle(anchors)

    out = np.full((H, W), bg, dtype=int)
    out[0, :] = roles["frame"]
    out[H - 1, :] = roles["frame"]
    out[:, 0] = roles["frame"]
    out[:, W - 1] = roles["frame"]

    frame_patch = out.copy()
    pieces: list[np.ndarray] = []
    row0 = 1
    for anchor_color, height in zip(anchors, heights):
        for local_idx in range(height):
            out[row0 + local_idx, 1] = roles["body"]
            out[row0 + local_idx, 2] = int(anchor_color)
            out[row0 + local_idx, 3] = roles["body"]
            frame_patch[row0 + local_idx, 2] = int(anchor_color)

            patch = np.full((3, 3), bg, dtype=int)
            patch[1, 0] = roles["body"]
            patch[1, 1] = int(anchor_color)
            patch[1, 2] = roles["body"]
            if local_idx == 0:
                patch[0, 1] = roles["top"]
            elif local_idx == 1:
                patch[0, 1] = roles["mid1"]
            else:
                patch[0, 1] = roles["mid2"]

            if local_idx == height - 1:
                patch[2, 1] = roles["bottom"]
            elif local_idx == 0:
                patch[2, 1] = roles["mid1"]
            else:
                patch[2, 1] = roles["mid2"]
            pieces.append(patch)
        row0 += height

    return out.astype(int), int(bg), frame_patch.astype(int), pieces


def _scatter(frame_patch: np.ndarray, pieces: list[np.ndarray], bg: int) -> np.ndarray:
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
    random.shuffle(pieces)
    for patch in pieces:
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


def _gen_attempt() -> np.ndarray | None:
    out, bg, frame_patch, pieces = _build_output_and_parts()
    try:
        inp = _scatter(frame_patch, pieces, bg)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid combo input for e8686506")
