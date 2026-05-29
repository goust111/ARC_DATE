# Combo variant of e8686506 (index 2)
# combo_spec: D1/0+D12/0
# Name: 颜色匹配+允许旋转拼合

"""
Design analysis for the combo variant
=====================================

combo_variant:
  core_change:
    from: 原题主要依赖单色碎片的几何洞匹配；输入朝向就是输出朝向。
    to: 内部碎片改成“主体色横条 + 两个端点提示色”的多色部件，真正决定上下相邻顺序的是端点颜色；
      同时这些部件在输入里会被旋转 0/90/180/270 度，因此 solver 既要恢复颜色链，也要恢复每块朝向。

  a_concretization_and_trace:
    objects:
      - 输出仍是 5 列的竖向矩形，并保持严格左右镜像。
      - 输入中保留一个已经摆成输出位置的矩形外框组件；外框主体是单一多数色，框内上/下中心各有一个锚点色。
      - 其余组件全部是 3x3 patch：中间一行 3 格是主体色，中心上/下各 1 个提示色。
      - generator 会把这些 3x3 patch 随机旋转后再散落。
    behavior_trace:
      original_local_step: |
        原题看到一个单色碎片时，会问“它几何上能塞进哪个洞”，位置对了就结束。
      combo_local_step: |
        现在内部条带主体几何完全重复；solver 先对每个 3x3 patch 尝试四个旋转，
        找出哪些转回后能成为“水平主体条 + 上下端点”。这一步会给出该 piece 的两种候选朝向：
        正向(top->bottom) 与反向(bottom->top)。
        接着从外框顶部锚点颜色出发，沿颜色链选择“当前 top 色匹配”的未使用 piece 朝向，逐行放入输出。
      difference: |
        旧逻辑的决定性信息是洞形状；新逻辑里几何只负责判定 piece 的合法朝向，
        真正决定行序的是颜色链。旋转让同一个 piece 产生正反两种颜色解释，必须联合推理。

  b_complexity_calibration:
    - 如果保留原题不规则洞形，颜色提示会沦为装饰；因此这里故意把内部主体统一成重复的 1x3 条带。
    - 如果允许旋转但不给颜色链，任务会退化成纯朝向识别；如果只给颜色链不旋转，又退化成 D1 的单链排序。
      当前设计里每个内部链接色都会同时对应“上一条反向候选”和“下一条正向候选”，形成真实歧义。
    - 高度只取 9 或 10，链长 8 或 9，足够让颜色对应关系在每个样例中重复出现，同时不会把搜索空间放大到不稳定。

  c_implicit_assumption_audit:
    main_old_assumptions:
      - 非背景同色连通块就是一个 piece。
      - piece 是单色，颜色本身就是 piece 身份。
      - 输入中观测到的形状朝向就是输出朝向。
      - 位置唯一性来自几何洞约束。
    why_broken_now:
      - 现在一个 piece 是多色 4 连通组件，不能按“同色 component”提取。
      - 观测朝向不可信；180 度旋转会交换上下端点，90/270 度旋转会把上下端点变成左右端点。
      - 内部主体条几何重复，不能靠洞形直接排序。
    new_assumptions_and_guards:
      - 每个组件都有唯一多数色主体。generator 固定让条带主体面积=3，外框主体面积远大于 2 个提示色。
      - 外框组件唯一，且主体恰为 5 列矩形边框；generator 固定这样生成。
      - 每个内部 piece 的完整 bbox 恰为 3x3，四旋转后只有两种能还原成“水平主体 + 上下提示”的解释；
        solver 明确枚举并去重，不靠颜色 id 排序。
      - 顶锚和底锚唯一，所有内部 piece 必须刚好用完；generator 通过单链构造保证这一点。

  d_placement_legality:
    - generator 先构造最终输出，再生成对应 frame patch 与 row patches。
    - 画布上先整块阻塞 frame bbox，所有旋转后的 3x3 row patch 都只能散落在 bbox 外。
    - 使用 random_free_location + 8 邻域 padding，保证 piece 之间不重叠、不贴边混连。
    - 最后只把 frame patch 贴回 bbox，因此输入中不会出现外框和散落条带相互覆盖。

  e_visual_observability:
    - 人类能直接看到：很多内部 piece 形状完全相同，但有的横着、有的竖着，且端点小色块位置随旋转改变。
    - 从几组样例中可观察到：输出内部总是水平色带；输入中的竖向条带必须先旋回水平，
      再按外框给出的起止颜色把每一行串起来。
    - 规则不依赖颜色编号大小或排序；任何颜色只作为“相同颜色可连接”的可见标记。

  f_visual_cleanliness:
    - 提示色仅占条带中心上下单点，外框锚点也仅占两个内侧单点，主体色仍然清晰。
    - 散落 piece 全在外框 bbox 之外，减少区域叠加；外框本身完整留在原位，尺寸锚点一眼可见。
    - 输出去掉所有提示点，只保留整洁的对称外框和内部水平色带。

  g_inference_of_change:
    - 输出主体色种类仍控制在 4-6，和原题接近；输入额外出现的链路颜色虽然较多，但都以“小连接点”形式重复出现：
      内部链接色在链中至少出现两次（上一条的 bottom 与下一条的 top），能支持人类归纳“同色相接”。
    - 量化上，本设计故意让每个内部链接色都对应两个候选解释：前一条反向或后一条正向；
      因此“只看颜色链、不看旋转”不再唯一，而“只看旋转、不看颜色链”也不够决定顺序。
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
_CANVAS_MIN = 16
_CANVAS_MAX = 18
_SCATTER_PADDING = 1
_MIN_ROT90_VISIBLE = 1
_GEN_MAX_RETRIES = 500

_LAST_GEN_INFO: dict | None = None


def _is_border_ring(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[h - 1, :] = True
    expected[:, 0] = True
    expected[:, w - 1] = True
    return np.array_equal(mask.astype(bool), expected)


def _component_info(grid: np.ndarray, pts: np.ndarray, bg: int) -> dict:
    vals = [int(grid[int(r), int(c)]) for r, c in pts]
    counts = Counter(vals)
    body_color, body_count = counts.most_common(1)[0]
    r0, c0, r1, c1 = bbox_coords(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in pts:
        patch[int(r) - r0, int(c) - c0] = int(grid[int(r), int(c)])
    return {
        "patch": patch,
        "body_color": int(body_color),
        "body_count": int(body_count),
    }


def _frame_info(info: dict, bg: int) -> dict | None:
    patch = np.asarray(info["patch"], dtype=int)
    body_color = int(info["body_color"])
    H, W = map(int, patch.shape)
    if W != _WIDTH or H not in _HEIGHTS:
        return None
    body_mask = patch == body_color
    if not _is_border_ring(body_mask):
        return None
    hint_cells = [(int(r), int(c)) for r, c in np.argwhere((patch != bg) & (patch != body_color))]
    if len(hint_cells) != 2:
        return None
    hint_cells.sort()
    top_pos, bot_pos = hint_cells
    if top_pos != (1, W // 2) or bot_pos != (H - 2, W // 2):
        return None
    return {
        "H": H,
        "W": W,
        "body_color": body_color,
        "top_anchor": int(patch[top_pos]),
        "bottom_anchor": int(patch[bot_pos]),
    }


def _row_candidates(info: dict, bg: int) -> list[dict]:
    patch = np.asarray(info["patch"], dtype=int)
    body_color = int(info["body_color"])
    assert patch.shape == (3, 3), "row pieces must occupy a 3x3 bbox"

    candidates = []
    seen = set()
    for k in range(4):
        rp = np.rot90(patch, k)
        body_mask = rp == body_color
        if set(map(tuple, np.argwhere(body_mask))) != {(1, 0), (1, 1), (1, 2)}:
            continue
        other = [(int(r), int(c)) for r, c in np.argwhere((rp != bg) & (rp != body_color))]
        if sorted(other) != [(0, 1), (2, 1)]:
            continue
        top = int(rp[0, 1])
        bottom = int(rp[2, 1])
        assert top != body_color and bottom != body_color, "hint colors must differ from the body color"
        key = (top, bottom)
        if key in seen:
            continue
        seen.add(key)
        candidates.append({"top": top, "bottom": bottom, "rot_k": int(k)})

    assert len(candidates) == 2, "each row piece must have exactly two horizontal candidate orientations"
    return candidates


def _parse_input(grid: np.ndarray) -> tuple[dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=4)

    frame = None
    rows = []
    for pts in comps:
        info = _component_info(g, pts, bg)
        maybe_frame = _frame_info(info, bg)
        if maybe_frame is not None:
            assert frame is None, "frame must be unique"
            frame = maybe_frame
            continue
        assert int(info["body_count"]) == 3, "row-piece body must have area 3"
        rows.append(
            {
                "body_color": int(info["body_color"]),
                "candidates": _row_candidates(info, bg),
            }
        )

    assert frame is not None, "missing frame component"
    assert len(rows) == int(frame["H"]) - 2, "row-piece count must match the frame interior height"
    return frame, rows


def _solve_chain(frame: dict, rows: list[dict]) -> list[dict]:
    target_len = int(frame["H"]) - 2
    full_mask = (1 << len(rows)) - 1
    memo: set[tuple[int, int, int]] = set()

    def dfs(step: int, current: int, used_mask: int) -> list[dict] | None:
        key = (step, current, used_mask)
        if key in memo:
            return None
        if step == target_len:
            if current == int(frame["bottom_anchor"]) and used_mask == full_mask:
                return []
            memo.add(key)
            return None

        for idx, row in enumerate(rows):
            if (used_mask >> idx) & 1:
                continue
            for cand in row["candidates"]:
                if int(cand["top"]) != current:
                    continue
                tail = dfs(step + 1, int(cand["bottom"]), used_mask | (1 << idx))
                if tail is not None:
                    return [{"row_index": idx, "body_color": int(row["body_color"]), "cand": cand}] + tail

        memo.add(key)
        return None

    path = dfs(0, int(frame["top_anchor"]), 0)
    assert path is not None, "color chain with rotation-aware orientations must be solvable"
    assert len(path) == target_len, "solution path must fill every interior row"
    return path


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered multicolor strip-fragments into a left-right symmetric rectangle by combining edge-color matching with quarter-turn recovery.

    Input structure:
    - The background is the dominant color.
    - One multicolor component already sits in output position: its majority color forms a 5-column rectangular border, and two minority pixels inside that border mark the top and bottom chain anchors.
    - Every other non-background component is a 3x3 patch whose majority color is a 1x3 body strip; two minority pixels encode the strip's edge colors, and the whole patch may be rotated by 0/90/180/270 degrees.

    Rule:
    - Treat each non-background connected component as one piece, using its majority color as the body and the two minority pixels as edge hints.
    - The frame fixes the output size. For every scattered 3x3 piece, try all quarter-turns and keep the rotations that convert it back into a horizontal strip with one hint above and one hint below.
    - Starting from the frame's top anchor color, choose an unused strip orientation whose upper hint matches the current required color, place its body on the next interior row, then continue with that strip's lower hint until the bottom frame anchor is reached.

    Key insight:
    - Internal body strips are intentionally repetitive, so geometry alone cannot order them; but color matching alone is also ambiguous because rotating a strip can swap its incoming and outgoing colors. The correct assembly appears only when rotation recovery and color-chain matching are solved together.
    """
    frame, rows = _parse_input(grid)
    path = _solve_chain(frame, rows)

    H, W = int(frame["H"]), int(frame["W"])
    bg = int(get_bg(np.asarray(grid, dtype=int)))
    out = np.full((H, W), bg, dtype=int)
    frame_color = int(frame["body_color"])
    out[0, :] = frame_color
    out[H - 1, :] = frame_color
    out[:, 0] = frame_color
    out[:, W - 1] = frame_color

    for r, item in enumerate(path, start=1):
        out[r, 1 : W - 1] = int(item["body_color"])

    assert np.array_equal(out, out[:, ::-1]), "output must remain left-right symmetric"
    return out.astype(int)


def _choose_body_assignment(
    palette: list[int],
    n_rows: int,
    top_links: list[int],
    bottom_links: list[int],
) -> tuple[int, list[int]]:
    for _ in range(300):
        frame_opts = [c for c in palette if c not in {top_links[0], bottom_links[-1]}]
        if not frame_opts:
            continue
        frame_color = random.choice(frame_opts)
        row_colors = []
        ok = True
        for i in range(n_rows):
            forbid = {top_links[i], bottom_links[i]}
            if i == 0 or i == n_rows - 1:
                forbid.add(frame_color)
            if row_colors:
                forbid.add(row_colors[-1])
            opts = [c for c in palette if c not in forbid]
            if not opts:
                ok = False
                break
            used = set(row_colors)
            pref = [c for c in opts if c in used]
            if pref and random.random() < 0.5:
                row_colors.append(int(random.choice(pref)))
            else:
                row_colors.append(int(random.choice(opts)))
        if not ok:
            continue
        used_colors = len(set([frame_color, *row_colors]))
        if _OUTPUT_N_COLORS_MIN <= used_colors <= _OUTPUT_N_COLORS_MAX:
            return int(frame_color), row_colors
    raise RuntimeError("failed to sample body colors within the required distinct-color range")


def _build_output_and_patches() -> tuple[np.ndarray, int, np.ndarray, list[dict], dict]:
    H = random.choice(_HEIGHTS)
    W = _WIDTH
    n_rows = H - 2

    bg = random.randint(0, 9)
    palette = [c for c in range(10) if c != bg]
    random.shuffle(palette)

    link_colors = random.sample(palette, k=n_rows + 1)
    top_links = link_colors[:-1]
    bottom_links = link_colors[1:]
    frame_color, row_colors = _choose_body_assignment(palette, n_rows, top_links, bottom_links)

    out = np.full((H, W), bg, dtype=int)
    out[0, :] = frame_color
    out[H - 1, :] = frame_color
    out[:, 0] = frame_color
    out[:, W - 1] = frame_color
    for r in range(1, H - 1):
        out[r, 1 : W - 1] = int(row_colors[r - 1])

    frame_patch = out.copy()
    frame_patch[1 : H - 1, 1 : W - 1] = bg
    frame_patch[1, W // 2] = int(top_links[0])
    frame_patch[H - 2, W // 2] = int(bottom_links[-1])

    row_patches = []
    for idx, (row_col, top_col, bot_col) in enumerate(zip(row_colors, top_links, bottom_links)):
        patch = np.full((3, 3), bg, dtype=int)
        patch[1, :] = int(row_col)
        patch[0, 1] = int(top_col)
        patch[2, 1] = int(bot_col)
        row_patches.append(
            {
                "patch": patch,
                "body_color": int(row_col),
                "top": int(top_col),
                "bottom": int(bot_col),
                "row_index": int(idx),
            }
        )

    meta = {
        "height": int(H),
        "row_count": int(n_rows),
        "output_distinct_colors": int(len(np.unique(out))),
        "link_color_count": int(len(set(link_colors))),
        # Every internal chain color is ambiguous once rotations are allowed:
        # it can be read either as piece i+1's incoming color or piece i's incoming color after reversal.
        "ambiguous_top_color_count": int(n_rows - 1),
    }
    return out, bg, frame_patch, row_patches, meta


def _scatter(frame_patch: np.ndarray, row_patches: list[dict], bg: int) -> tuple[np.ndarray, dict]:
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

    order = row_patches[:]
    random.shuffle(order)
    force_n = random.randint(_MIN_ROT90_VISIBLE, min(3, len(order)))
    forced = set(random.sample(range(len(order)), k=force_n))

    rot90_visible_count = 0
    rotated_piece_count = 0
    quarter_turn_piece_count = 0

    for i, info in enumerate(order):
        if i in forced:
            k = random.choice([1, 3])
        else:
            k = random.randrange(4)
        rotated = np.rot90(np.asarray(info["patch"], dtype=int), k)
        if k != 0:
            rotated_piece_count += 1
        if k in (1, 3):
            quarter_turn_piece_count += 1
            rot90_visible_count += 1
        r0, c0 = random_free_location(
            canvas,
            rotated,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, rotated, r0, c0, transparent=bg)

    canvas[off_r : off_r + H, off_c : off_c + W] = bg
    paste(canvas, frame_patch, off_r, off_c, transparent=bg)

    meta = {
        "rot90_visible_count": int(rot90_visible_count),
        "quarter_turn_piece_count": int(quarter_turn_piece_count),
        "rotated_piece_count": int(rotated_piece_count),
        "frame_offset_r": int(off_r),
        "frame_offset_c": int(off_c),
    }
    if meta["rot90_visible_count"] < _MIN_ROT90_VISIBLE:
        raise ValueError("rotation visibility requirement not met")
    return canvas, meta


def _gen_attempt() -> tuple[np.ndarray | None, dict | None]:
    out, bg, frame_patch, row_patches, meta0 = _build_output_and_patches()
    try:
        inp, meta1 = _scatter(frame_patch, row_patches, bg)
        got = main(inp)
    except Exception:
        return None, None

    if np.array_equal(got, out):
        meta = dict(meta0)
        meta.update(meta1)
        return inp, meta
    return None, None


def generate_input() -> np.ndarray:
    global _LAST_GEN_INFO
    for _ in range(_GEN_MAX_RETRIES):
        grid, meta = _gen_attempt()
        if grid is not None:
            _LAST_GEN_INFO = dict(meta or {})
            return grid
    raise RuntimeError("failed to generate a valid combo input for e8686506")
