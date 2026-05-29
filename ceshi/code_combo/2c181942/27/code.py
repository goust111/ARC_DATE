# Combo variant of 2c181942 (index 27)
# combo_spec: D1/0+D11/0
# Name: 形状轮廓匹配+内嵌填充

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, get_bg, label_mask, paste, random_free_location


"""
design_analysis:
  fused_goal:
    summary: |
      原题靠“边颜色 = 精灵颜色”配对，并把精灵贴在 hub 外侧。本融合把这两点一起改掉：
      1) 全部前景统一成一种颜色，颜色不再携带配对信息；
      2) hub 内侧四边各有一个可见的接收槽模式，精灵要靠轮廓互补去找边；
      3) 输出时精灵不是贴外面，而是把接触边压在 hub 对应边上并向内部伸入。

  a_concretization_and_trace:
    concrete_geometry: |
      为了同时容纳“内嵌”与“可见轮廓编码”，hub 固定为 10x10 单色空心方框。
      外框仍是原题可识别的 hollow square：四角 bg、四边同色。
      另外在内部四边中央各放一段长度 4 的 receiving slot：
        - top slot    : row r+1, cols c+3..c+6
        - bottom slot : row r+8, cols c+3..c+6
        - left slot   : col c+1, rows r+3..r+6
        - right slot  : col c+8, rows r+3..r+6
      slot 上的 bg 单元表示“孔”，前景单元表示“实心”。四边的孔位模式各不相同。
    tiny_trace: |
      例：top slot 的孔位模式若为 1000，表示 row r+1 的四格里只有最左一格是 bg 孔。
      对应 top 精灵在正确朝向时：
        - 第 0 行必须是连续 4 格接触带，会压到 hub 顶边 row r 的中央四格；
        - 第 1 行必须只在最左位置有 1 格凸起，恰好填进 row r+1 的那个孔；
        - 更深的身体继续落入 hub 内部。
      原题在这里是“看颜色匹配哪一边，再往外贴”；融合后变成“看内槽孔位匹配哪一边，再往里嵌”。
    key_difference: |
      被改动的推理链是协同的：如果只做内嵌但还保留颜色，题目仍近似原题；
      如果只做单色形状匹配但仍外贴，视觉变化没这么强。现在必须先读 hub 内部槽型，
      再把同色散块旋转后塞进内部，两个变化不可分割。

  b_complexity_calibration:
    why_hub_is_larger: |
      N=6 虽能做简单内嵌，但四边中央槽型会太挤；N=10 可以把四个长度 4 的槽位
      放到内侧中央区，彼此完全错开，不会因为正交边干扰而抹掉孔位信息。
    controlled_freedom: |
      模式只用 4 个固定 4-bit 档位：
        top   = 1000
        bot   = 1100
        left  = 1010
        right = 1110
      它们互不相同，且 reverse 后也不与其它边重合，足够支持少量样例归纳，
      又不会让生成器需要处理过多离散情形。
    non_degenerate_sprite_family: |
      精灵不是完全固定模板，而是“接触带 + 凸起带 + 少量主体噪声”的一小族。
      这样既能保持唯一匹配，又保留旋转恢复与散落放置的自由度。

  c_implicit_assumptions_review:
    original_assumptions_broken:
      - 原 main 里按颜色 `m = (g == col) & (~hub_mask)` 取某边精灵；融合后完全失效。
      - 原 `_find_hub()` 假设 hub 内部全 bg；融合后不成立，因为接收槽本身就在内部可见。
      - 原 transform 假设 top 对应 sprite 的 bottom 接触边；融合后要改成 same-side 接触。
      - 原 generator 的干扰物可用新颜色规避；融合后必须同色且靠“不含任何合法槽型”规避。
    new_solver_assumptions:
      - hub 是唯一靠近中心、满足固定 10x10 几何的单色组件；
      - 四个内部槽位都位于固定偏移位置，且孔位模式两两不同；
      - 每个真精灵在四种旋转里只会对一个 side 呈现合法“接触带+凸起带”；
      - 干扰物不产生任何合法 side 匹配；
      - 输入中 hub 与散落精灵之间有间距，因此 8 连通标号可把 hub 单独分出。
    generator_guarantees: |
      下面的 output-first 生成器逐条构造这些条件，并在每次样本生成后用新 solver 做 oracle 校验。

  d_placement_legality:
    layout_rules: |
      输出先画空 hub，再把真精灵以内嵌方式粘回。因为四个槽位位于中央区但彼此错开，
      不同方向的接触带不会抢同一格。若主体在内部重合，也只是同色并集，不构成冲突。
    input_rules: |
      输入中只放空 hub；精灵先从输出取下、随机做非恒等旋转，再用
      `random_free_location(..., padding=2, connectivity=8)` 散到外部，避免越界或与 hub 连成一块。

  e_visual_observability:
    human_view: |
      解题者能直接看到：中心是单色方框，但方框内部四边中央各有不同的孔位槽；
      外面散着同色精灵，它们都有一条长度 4 的“嘴边”和一排凸点。
      输出里这些块被旋转后塞进 hub 里面，并把对应孔位补上。这一规律在人眼上是明确可见的。
    no_hidden_signal: |
      不使用颜色数值大小、生成顺序等不可见因素；所有匹配信息都来自可见轮廓。

  f_visual_cleanliness:
    cleanliness: |
      槽型只占 hub 内侧中央的 4 格短段，不把整个内部铺满；四角与大部分腔体仍为空，
      因而 hub 结构清楚。精灵也限制在 4x4 到 6x6 的小尺寸，避免散点过杂。

  g_inference_of_change:
    reasoning_path: |
      这个融合不要求人类猜“颜色如何映射”，因为颜色根本不区分对象。
      人类只需观察：哪块的凸起能补哪条内槽、且补进去后块体位于 hub 内部而非外部。

  h_metric_for_post_test:
    metric: |
      用两个指标验证没有退化：
        1) `slot_holes_filled`: 输出中，原本 input hub 的 bg 孔位被填上的数量；
        2) `inner_fill`: 输出在 hub 内框内新增的非 bg 像素数。
      若这两个值长期为 0，就说明没发生“形状轮廓匹配+内嵌填充”。

  i_solver_branch_audit:
    audit:
      - `_find_hub()` 依赖固定的外框和槽位偏移；generator 固定用同一几何。
      - `_contact_band()` 依赖 candidate 在合法朝向下，其接触带占据该朝向最外层一整段，
        且凸起带只出现在紧邻内侧的一层；generator 精灵专门按这个语法生成。
      - `_match_component_to_side()` 依赖同一组件对同一边至多一个旋转合法；generator 用
        `_count_valid_matches(mask) == [(0, side)]` 过滤。
      - side 之间的匹配顺序依赖组件不会一物两配；同样由上面的唯一匹配检查保证。
"""


HUB_N = 10
SPAN_LEN = 4
BG_COLOR = 8
SLOT_ROW_OFFSET = 1
SLOT_COL_OFFSET = 1
SLOT_SPAN_OFFSET = 3
SIDES = ("top", "bot", "left", "right")
PATTERNS = {
    "top": (1, 0, 0, 0),
    "bot": (1, 1, 0, 0),
    "left": (1, 0, 1, 0),
    "right": (1, 1, 1, 0),
}


def _crop_to_mask(arr: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(pts)
    return arr[r0 : r1 + 1, c0 : c1 + 1], mask[r0 : r1 + 1, c0 : c1 + 1]


def _side_span(hub_r: int, hub_c: int, side: str) -> tuple[int, int]:
    if side in ("top", "bot"):
        start = hub_c + SLOT_SPAN_OFFSET
    else:
        start = hub_r + SLOT_SPAN_OFFSET
    return int(start), int(start + SPAN_LEN - 1)


def _slot_coord(hub_r: int, hub_c: int, side: str) -> int:
    if side == "top":
        return int(hub_r + SLOT_ROW_OFFSET)
    if side == "bot":
        return int(hub_r + HUB_N - 1 - SLOT_ROW_OFFSET)
    if side == "left":
        return int(hub_c + SLOT_COL_OFFSET)
    return int(hub_c + HUB_N - 1 - SLOT_COL_OFFSET)


def _read_hub_patterns(g: np.ndarray, hub_r: int, hub_c: int, color: int, bg: int) -> dict[str, tuple[int, ...]] | None:
    pats: dict[str, tuple[int, ...]] = {}
    for side in SIDES:
        start, end = _side_span(hub_r, hub_c, side)
        fixed = _slot_coord(hub_r, hub_c, side)
        vals = []
        if side in ("top", "bot"):
            for cc in range(start, end + 1):
                cell = int(g[fixed, cc])
                if cell not in (bg, color):
                    return None
                vals.append(1 if cell == bg else 0)
        else:
            for rr in range(start, end + 1):
                cell = int(g[rr, fixed])
                if cell not in (bg, color):
                    return None
                vals.append(1 if cell == bg else 0)
        pats[side] = tuple(vals)
    uniq = set(pats.values())
    if len(uniq) != 4:
        return None
    if any(sum(p) <= 0 or sum(p) >= SPAN_LEN for p in pats.values()):
        return None
    return pats


def _find_hub(g: np.ndarray, bg: int) -> tuple[int, int, int, dict[str, tuple[int, ...]]] | None:
    H, W = g.shape
    best = None
    n = HUB_N
    for r in range(0, H - n + 1):
        for c in range(0, W - n + 1):
            if (
                int(g[r, c]) != bg
                or int(g[r, c + n - 1]) != bg
                or int(g[r + n - 1, c]) != bg
                or int(g[r + n - 1, c + n - 1]) != bg
            ):
                continue

            top = g[r, c + 1 : c + n - 1]
            bot = g[r + n - 1, c + 1 : c + n - 1]
            left = g[r + 1 : r + n - 1, c]
            right = g[r + 1 : r + n - 1, c + n - 1]
            if bool((top == bg).any()) or bool((bot == bg).any()) or bool((left == bg).any()) or bool((right == bg).any()):
                continue

            colors = set(map(int, np.concatenate([top, bot, left, right])))
            if len(colors) != 1:
                continue
            color = int(next(iter(colors)))
            if color == bg:
                continue

            pats = _read_hub_patterns(g, r, c, color, bg)
            if pats is None:
                continue

            ok = True
            for rr in range(r + 1, r + n - 1):
                for cc in range(c + 1, c + n - 1):
                    side_slot = False
                    for side in SIDES:
                        start, end = _side_span(r, c, side)
                        fixed = _slot_coord(r, c, side)
                        if side in ("top", "bot"):
                            if rr == fixed and start <= cc <= end:
                                side_slot = True
                                break
                        else:
                            if cc == fixed and start <= rr <= end:
                                side_slot = True
                                break
                    if side_slot:
                        continue
                    if int(g[rr, cc]) != bg:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                continue

            cr = r + (n - 1) / 2
            cc = c + (n - 1) / 2
            score = abs(cr - (H - 1) / 2) + abs(cc - (W - 1) / 2)
            cand = (float(score), int(r), int(c), int(color), pats)
            if best is None or cand[0] < best[0]:
                best = cand

    if best is None:
        return None
    _, hub_r, hub_c, color, pats = best
    return hub_r, hub_c, color, pats


def _contact_band(mask: np.ndarray, side: str, pattern: tuple[int, ...]) -> int | None:
    h, w = mask.shape
    L = len(pattern)
    hits: list[int] = []

    if side == "top":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[0, :j].any()) or bool(mask[0, j + L :].any()):
                continue
            if bool(mask[1, :j].any()) or bool(mask[1, j + L :].any()):
                continue
            if not bool(mask[0, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[1, j : j + L]) != pattern:
                continue
            hits.append(j)
    elif side == "bot":
        if h < 2 or w < L:
            return None
        for j in range(w - L + 1):
            if bool(mask[h - 1, :j].any()) or bool(mask[h - 1, j + L :].any()):
                continue
            if bool(mask[h - 2, :j].any()) or bool(mask[h - 2, j + L :].any()):
                continue
            if not bool(mask[h - 1, j : j + L].all()):
                continue
            if tuple(int(v) for v in mask[h - 2, j : j + L]) != pattern:
                continue
            hits.append(j)
    elif side == "left":
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, 0].any()) or bool(mask[j + L :, 0].any()):
                continue
            if bool(mask[:j, 1].any()) or bool(mask[j + L :, 1].any()):
                continue
            if not bool(mask[j : j + L, 0].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, 1]) != pattern:
                continue
            hits.append(j)
    else:
        if w < 2 or h < L:
            return None
        for j in range(h - L + 1):
            if bool(mask[:j, w - 1].any()) or bool(mask[j + L :, w - 1].any()):
                continue
            if bool(mask[:j, w - 2].any()) or bool(mask[j + L :, w - 2].any()):
                continue
            if not bool(mask[j : j + L, w - 1].all()):
                continue
            if tuple(int(v) for v in mask[j : j + L, w - 2]) != pattern:
                continue
            hits.append(j)

    if len(hits) != 1:
        return None
    return hits[0]


def _count_valid_matches(mask: np.ndarray) -> list[tuple[int, str]]:
    matches: list[tuple[int, str]] = []
    for krot in range(4):
        rot = np.rot90(mask, krot)
        rot = _crop_to_mask(rot.astype(int), rot)[1]
        for side, pattern in PATTERNS.items():
            if _contact_band(rot, side, pattern) is not None:
                matches.append((krot, side))
    return matches


def _placeable(out: np.ndarray, arr: np.ndarray, mask: np.ndarray, dr: int, dc: int, bg: int) -> bool:
    H, W = out.shape
    for rr, cc in np.argwhere(mask):
        R = int(rr + dr)
        C = int(cc + dc)
        if not (0 <= R < H and 0 <= C < W):
            return False
        val = int(arr[rr, cc])
        if val == bg:
            continue
        if int(out[R, C]) != bg and int(out[R, C]) != val:
            return False
    return True


def _attachment_shift(mask: np.ndarray, side: str, pattern: tuple[int, ...], hub_r: int, hub_c: int) -> tuple[int, int] | None:
    pos = _contact_band(mask, side, pattern)
    if pos is None:
        return None
    span_start, _span_end = _side_span(hub_r, hub_c, side)
    h, w = mask.shape

    if side == "top":
        return int(hub_r), int(span_start - pos)
    if side == "bot":
        return int(hub_r + HUB_N - h), int(span_start - pos)
    if side == "left":
        return int(span_start - pos), int(hub_c)
    return int(span_start - pos), int(hub_c + HUB_N - w)


def _match_component_to_side(
    comp_arr: np.ndarray,
    comp_mask: np.ndarray,
    side: str,
    pattern: tuple[int, ...],
    hub_r: int,
    hub_c: int,
    out: np.ndarray,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    hits = []
    for krot in range(4):
        arr = np.rot90(comp_arr, krot)
        mask = np.rot90(comp_mask, krot)
        arr, mask = _crop_to_mask(arr, mask)
        shift = _attachment_shift(mask, side, pattern, hub_r, hub_c)
        if shift is None:
            continue
        dr, dc = shift
        if not _placeable(out, arr, mask, dr, dc, bg):
            continue
        hits.append((arr, mask, dr, dc))
    if len(hits) != 1:
        return None
    return hits[0]


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))

    hub = _find_hub(g, bg)
    assert hub is not None
    hub_r, hub_c, hub_color, hub_patterns = hub

    lab, nc = label_mask(g != bg, connectivity=8)
    hub_seed = (hub_r, hub_c + 1)
    hub_id = int(lab[hub_seed])
    assert hub_id > 0
    hub_mask = lab == hub_id

    out = np.full((H, W), bg, dtype=int)
    out[hub_mask] = g[hub_mask]

    components = []
    for cid in range(1, int(nc) + 1):
        if cid == hub_id:
            continue
        mask = lab == cid
        arr = np.where(mask, hub_color, bg)
        arr, mask = _crop_to_mask(arr, mask)
        components.append((int(cid), arr, mask, int(mask.sum())))
    components.sort(key=lambda x: -x[3])

    used: set[int] = set()
    for side in SIDES:
        pattern = hub_patterns[side]
        found = None
        found_id = None
        for cid, arr, mask, _sz in components:
            if cid in used:
                continue
            match = _match_component_to_side(arr, mask, side, pattern, hub_r, hub_c, out, bg)
            if match is None:
                continue
            found = match
            found_id = cid
            break
        if found is None:
            continue
        arr, mask, dr, dc = found
        paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=True)
        used.add(int(found_id))

    return out


def _draw_hub(canvas: np.ndarray, hub_r: int, hub_c: int, color: int, bg: int) -> None:
    n = HUB_N
    canvas[hub_r : hub_r + n, hub_c : hub_c + n] = bg

    for cc in range(1, n - 1):
        canvas[hub_r, hub_c + cc] = color
        canvas[hub_r + n - 1, hub_c + cc] = color
    for rr in range(1, n - 1):
        canvas[hub_r + rr, hub_c] = color
        canvas[hub_r + rr, hub_c + n - 1] = color

    for side, pattern in PATTERNS.items():
        start, end = _side_span(hub_r, hub_c, side)
        fixed = _slot_coord(hub_r, hub_c, side)
        for i, bit in enumerate(pattern):
            if bit:
                continue
            if side in ("top", "bot"):
                canvas[fixed, start + i] = color
            else:
                canvas[start + i, fixed] = color


def _add_body_noise(mask: np.ndarray, side: str, band_start: int, rng: random.Random) -> None:
    h, w = mask.shape

    if side == "top":
        if h >= 3:
            stem_c = band_start + rng.randint(0, SPAN_LEN - 2)
            mask[2, stem_c : stem_c + rng.randint(1, 2)] = True
        for rr in range(3, h):
            for cc in range(max(0, band_start - 1), min(w, band_start + SPAN_LEN + 1)):
                if rng.random() < 0.28:
                    mask[rr, cc] = True
    elif side == "bot":
        if h >= 3:
            stem_c = band_start + rng.randint(0, SPAN_LEN - 2)
            mask[h - 3, stem_c : stem_c + rng.randint(1, 2)] = True
        for rr in range(0, max(0, h - 3)):
            for cc in range(max(0, band_start - 1), min(w, band_start + SPAN_LEN + 1)):
                if rng.random() < 0.28:
                    mask[rr, cc] = True
    elif side == "left":
        if w >= 3:
            stem_r = band_start + rng.randint(0, SPAN_LEN - 2)
            mask[stem_r : stem_r + rng.randint(1, 2), 2] = True
        for cc in range(3, w):
            for rr in range(max(0, band_start - 1), min(h, band_start + SPAN_LEN + 1)):
                if rng.random() < 0.28:
                    mask[rr, cc] = True
    else:
        if w >= 3:
            stem_r = band_start + rng.randint(0, SPAN_LEN - 2)
            mask[stem_r : stem_r + rng.randint(1, 2), w - 3] = True
        for cc in range(0, max(0, w - 3)):
            for rr in range(max(0, band_start - 1), min(h, band_start + SPAN_LEN + 1)):
                if rng.random() < 0.28:
                    mask[rr, cc] = True


def _make_sprite_mask(pattern: tuple[int, ...], side: str, rng: random.Random) -> np.ndarray | None:
    L = len(pattern)
    for _ in range(800):
        h = rng.randint(4, 6)
        w = rng.randint(4, 6)
        if side in ("top", "bot") and w < L:
            w = L
        if side in ("left", "right") and h < L:
            h = L

        mask = np.zeros((h, w), dtype=bool)
        if side in ("top", "bot"):
            band_start = rng.randint(0, w - L)
        else:
            band_start = rng.randint(0, h - L)

        if side == "top":
            mask[0, band_start : band_start + L] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[1, band_start + i] = True
        elif side == "bot":
            mask[h - 1, band_start : band_start + L] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[h - 2, band_start + i] = True
        elif side == "left":
            mask[band_start : band_start + L, 0] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[band_start + i, 1] = True
        else:
            mask[band_start : band_start + L, w - 1] = True
            for i, bit in enumerate(pattern):
                if bit:
                    mask[band_start + i, w - 2] = True

        _add_body_noise(mask, side, band_start, rng)

        if not bool(mask.any()):
            continue
        mask = _crop_to_mask(mask.astype(int), mask)[1]
        area = int(mask.sum())
        if area < 6 or area > 14:
            continue

        lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue

        matches = _count_valid_matches(mask)
        if matches != [(0, side)]:
            continue
        return mask
    return None


def _make_distractor_mask(rng: random.Random) -> np.ndarray | None:
    for _ in range(800):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        mask = np.zeros((h, w), dtype=bool)
        k = rng.randint(4, min(8, h * w))
        for rr, cc in rng.sample([(r, c) for r in range(h) for c in range(w)], k):
            mask[rr, cc] = True
        mask = _crop_to_mask(mask.astype(int), mask)[1]
        if int(mask.sum()) < 4:
            continue
        lab, nc = label_mask(mask, connectivity=8)
        if int(nc) != 1:
            continue
        if _count_valid_matches(mask):
            continue
        return mask
    return None


def _attach_sprite(
    out: np.ndarray,
    mask: np.ndarray,
    side: str,
    hub_r: int,
    hub_c: int,
    color: int,
    bg: int,
) -> tuple[np.ndarray, np.ndarray, int, int] | None:
    arr = np.where(mask, color, bg).astype(int)
    shift = _attachment_shift(mask, side, PATTERNS[side], hub_r, hub_c)
    if shift is None:
        return None
    dr, dc = shift
    if not _placeable(out, arr, mask, dr, dc, bg):
        return None
    paste(out, arr, dr, dc, mask=mask, overwrite=True, clip=False)
    return arr, mask, dr, dc


def _variant_metrics(inp: np.ndarray, solved: np.ndarray) -> dict[str, int]:
    bg = int(get_bg(inp))
    hub = _find_hub(inp, bg)
    assert hub is not None
    hub_r, hub_c, _color, patterns = hub

    hole_cells = []
    for side, pattern in patterns.items():
        start, _end = _side_span(hub_r, hub_c, side)
        fixed = _slot_coord(hub_r, hub_c, side)
        for i, bit in enumerate(pattern):
            if not bit:
                continue
            if side in ("top", "bot"):
                hole_cells.append((fixed, start + i))
            else:
                hole_cells.append((start + i, fixed))

    slot_holes_filled = sum(1 for rr, cc in hole_cells if int(solved[rr, cc]) != bg)
    inner_fill = int(np.count_nonzero(solved[hub_r + 1 : hub_r + HUB_N - 1, hub_c + 1 : hub_c + HUB_N - 1] != bg))
    return {
        "slot_holes_total": int(len(hole_cells)),
        "slot_holes_filled": int(slot_holes_filled),
        "inner_fill": int(inner_fill),
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = BG_COLOR
    color = int(rng.choice([c for c in range(10) if c != bg]))

    H = rng.randint(24, 30)
    W = rng.randint(24, 30)

    sides = list(SIDES)
    K = rng.randint(2, 4)
    sprite_sides = rng.sample(sides, K)

    sprite_masks: dict[str, np.ndarray] = {}
    for side in sprite_sides:
        mask = _make_sprite_mask(PATTERNS[side], side, rng)
        if mask is None:
            return None
        sprite_masks[side] = mask

    max_extent = max(max(mask.shape) for mask in sprite_masks.values())
    margin = max_extent + 3
    hub_r_min = margin
    hub_r_max = H - HUB_N - margin
    hub_c_min = margin
    hub_c_max = W - HUB_N - margin
    if hub_r_min > hub_r_max or hub_c_min > hub_c_max:
        return None

    hub_r = rng.randint(hub_r_min, hub_r_max)
    hub_c = rng.randint(hub_c_min, hub_c_max)

    out = np.full((H, W), bg, dtype=int)
    inp = np.full((H, W), bg, dtype=int)
    _draw_hub(out, hub_r, hub_c, color, bg)
    _draw_hub(inp, hub_r, hub_c, color, bg)

    attached: dict[str, tuple[np.ndarray, np.ndarray, int, int]] = {}
    for side in sprite_sides:
        placed = _attach_sprite(out, sprite_masks[side], side, hub_r, hub_c, color, bg)
        if placed is None:
            return None
        attached[side] = placed

    canvas = inp.copy()
    for side in sprite_sides:
        arr, mask, _sr, _sc = attached[side]
        k = rng.choice([1, 2, 3])
        rot_arr = np.rot90(arr, k)
        rot_mask = np.rot90(mask, k)
        rot_arr, rot_mask = _crop_to_mask(rot_arr, rot_mask)
        patch = np.where(rot_mask, rot_arr, bg).astype(int)
        try:
            pr, pc = random_free_location(canvas, patch, transparent=bg, padding=2, padding_connectivity=8)
        except ValueError:
            return None
        paste(canvas, patch, pr, pc, transparent=bg)
        paste(inp, patch, pr, pc, transparent=bg)

    if rng.random() < 0.35:
        dist_mask = _make_distractor_mask(rng)
        if dist_mask is not None:
            dist_arr = np.where(dist_mask, color, bg).astype(int)
            try:
                pr, pc = random_free_location(canvas, dist_arr, transparent=bg, padding=2, padding_connectivity=8)
                paste(canvas, dist_arr, pr, pc, transparent=bg)
                paste(inp, dist_arr, pr, pc, transparent=bg)
            except ValueError:
                pass

    result = main(inp)
    if not np.array_equal(result, out):
        return None

    metrics = _variant_metrics(inp, result)
    if metrics["slot_holes_filled"] <= 0:
        return None
    if metrics["inner_fill"] <= 0:
        return None

    non_bg_colors = sorted(set(map(int, inp[inp != bg].flatten())))
    if len(non_bg_colors) != 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 500 attempts")
