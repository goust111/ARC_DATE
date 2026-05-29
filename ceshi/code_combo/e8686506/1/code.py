# Combo variant of e8686506 (index 1)
# combo_spec: D1/0+D5/3
# Name: 颜色匹配拼合+特定颜色对禁邻

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, get_bg, paste, random_free_location

"""
design_analysis:
  a_concretization_and_trace:
    objects:
      - 输出仍是 5 列的竖向矩形，保持严格左右镜像。
      - 输入左上角新增一个固定 2x2 双色图例，显式给出一对“不能 4 邻接”的颜色。
      - 其余主体沿用 D1 的视觉骨架：一个已经摆成输出位置的矩形边框 piece，以及若干散落的 1x3 横条。
      - 每条横条主体只有 3 个同色像素，但其上/下中心各有 1 个提示色像素，表示它与上下相邻条之间的颜色接口。
    behavior_trace:
      - 原题依赖洞形状与 piece 轮廓的几何贴合来唯一定位。
      - D1 单变体把内部块故意做成同形横条，仅靠上下提示色把它们串成一条链。
      - 本融合版进一步让某个“上提示色/下提示色”接口重复出现两次；于是颜色链只能推出
        “这两条都该接在同一类接口上”，却无法决定谁更靠前。
      - 此时必须再读取左上角图例得到禁邻颜色对，用“某条横条若放在这里会与相邻行形成禁邻对”
        这一信息来打破顺序歧义。
      - 例如某样例里两条横条都有 top=t1,bottom=t2；若其中一条主体色是 A、另一条主体色是 X，
        且图例声明 A 与 C 不能相邻，同时前一行主体色正好是 C，那么那一格只能放 X，A 只能去后面的重复接口。

  b_complexity_calibration:
    - 如果每个接口都只出现一次，那么禁邻图例会沦为装饰；因此 generator 固定制造 1 组重复接口。
    - 如果重复接口出现太多，颜色链会退化成复杂图搜索；因此只保留 1 组可见重复，并把行数固定为 8 条，
      既能提供歧义，也能让推理规模保持很小。
    - 禁邻对固定只是一对颜色，且这两色在输出里会重复出现，训练样例能反复支持“是这两色不能相邻，
      不是所有同色/异色都受限”。

  c_implicit_assumption_audit:
    old_assumptions_broken:
      - 原 solver 假设非背景同色连通块就是一个单色 piece；D1 风格条形片是多色连通块，因此必须改成按整体 component 解析。
      - 原 solver 假设 piece 的几何形状足以决定位置；现在内部横条几何完全重复，这个前提失效。
      - D5 改动取消“同色不可邻接”，所以任何依赖 color_piece_count 推出同色禁邻的逻辑都必须删除。
    new_assumptions_added:
      - 左上角一定有一个与主谜面隔离的 2x2 图例；generator 通过预留 3x3 角区保证。
      - 唯一 frame component 的主体是 10x5 的矩形边框，且内部有上下两个锚点提示色。
      - 所有非 frame component 都是 1x3 主体横条，提示点固定在主体上下中心。
      - 恰有 8 条横条，且存在唯一一组重复接口；忽略禁邻时至少有 2 个合法顺序，加入禁邻后唯一。
      - 主体色不会等于本 piece 的提示色，否则多数色解析会失真；generator 在颜色分配时显式规避。

  d_placement_legality:
    - generator 先构造最终输出和每个 piece patch，再把 frame bbox 与 legend 角区一起临时阻塞。
    - 所有横条 patch 都通过 random_free_location(..., padding=1) 散落到阻塞区之外，保证互不重叠且与 frame/legend 分离。
    - 最后再把 frame patch 放回矩形 bbox，把图例放回左上角，整个输入没有越界与覆盖冲突。

  e_visual_observability:
    - 人眼能同时看到三个稳定信号：
      1. 左上角固定双色图例；
      2. 边框内部只有上下两个锚点；
      3. 多条内部横条拥有相同形状，且其中有两条的上下提示色组合完全相同。
    - 看到训练样例时，解题者能自然发现：几何上这些横条可交换，但图例指定的那对颜色在输出中从不贴边，
      因而必须用图例去决定重复接口的先后。

  f_visual_cleanliness:
    - 图例单独占据左上角，并有 1 格背景护城河，不会与拼图部件混淆。
    - 提示色只占中心单像素，主体仍然是大面积横条/边框，视觉层次清楚。
    - 输出中所有提示色都会消失，只留下干净的边框与水平色带。

  g_inference_of_change:
    - 颜色变化全部是可见的：同一对提示接口重复出现，且图例中的两色在输出里以主体色形式出现。
    - 不使用任何基于颜色编号大小、扫描顺序或隐藏排序的规则；solver 只使用肉眼可见的：
      接口颜色是否匹配，以及图例给出的禁邻对是否被违反。
"""


_WIDTH = 5
_HEIGHT = 10
_INTERIOR_ROWS = _HEIGHT - 2
_CANVAS_MIN = 15
_CANVAS_MAX = 17
_SCATTER_PADDING = 1
_LEGEND_BLOCK = 3
_GEN_MAX_RETRIES = 400

_LINK_PATTERN = (0, 1, 2, 3, 1, 2, 4, 1, 5)
_ROW_ROLE_SEQ = ("C", "X", "A", "D", "A", "B", "C", "D")
_FORBIDDEN_ROLE_PAIR = ("A", "C")


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


def _norm_pair(a: int, b: int) -> tuple[int, int]:
    a = int(a)
    b = int(b)
    return (a, b) if a <= b else (b, a)


def _touches_forbidden(a: int, b: int, forbidden_pair: tuple[int, int] | None) -> bool:
    return forbidden_pair is not None and _norm_pair(a, b) == forbidden_pair


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
        hints.append({"r": int(r - br0), "c": int(c - bc0), "col": int(col)})

    return {
        "body_color": int(body_color),
        "body_h": int(h),
        "body_w": int(w),
        "body_coords": body_coords,
        "hints": hints,
    }


def _read_forbidden_pair(grid: np.ndarray, bg: int) -> tuple[tuple[int, int], np.ndarray]:
    """
    Read the fixed top-left 2x2 legend and erase it before parsing puzzle pieces.

    Layout:
      A C .
      A C .
      . . .
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    assert H >= _LEGEND_BLOCK and W >= _LEGEND_BLOCK, "legend requires a 3x3 top-left corner"
    a0, a1 = int(g[0, 0]), int(g[1, 0])
    c0, c1 = int(g[0, 1]), int(g[1, 1])
    assert a0 == a1 and c0 == c1, "legend columns must be monochrome dominoes"
    assert a0 != bg and c0 != bg and a0 != c0, "legend must contain two distinct non-background colors"
    for r, c in ((0, 2), (1, 2), (2, 0), (2, 1), (2, 2)):
        assert int(g[r, c]) == bg, "legend moat must stay background"
    cleaned = g.copy()
    cleaned[0:2, 0:2] = bg
    return _norm_pair(a0, c0), cleaned


def _parse_cleaned_input(grid: np.ndarray) -> tuple[dict, list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = components_mask(g != bg, connectivity=4)
    infos = [_component_info(g, pts) for pts in comps]

    frame = None
    rows: list[dict] = []
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
    assert int(frame["body_h"]) == _HEIGHT and int(frame["body_w"]) == _WIDTH, "frame dimensions must define the output"
    frame_hints = sorted(frame["hints"], key=lambda x: (x["r"], x["c"]))
    assert len(frame_hints) == 2, "frame must expose exactly two anchor pixels"
    assert (frame_hints[0]["r"], frame_hints[0]["c"]) == (1, _WIDTH // 2)
    assert (frame_hints[1]["r"], frame_hints[1]["c"]) == (_HEIGHT - 2, _WIDTH // 2)
    frame["top_anchor"] = int(frame_hints[0]["col"])
    frame["bottom_anchor"] = int(frame_hints[1]["col"])
    assert len(rows) == _INTERIOR_ROWS, "the number of strips must match the frame interior height"
    return frame, rows


def _find_row_orders(
    frame: dict,
    rows: list[dict],
    forbidden_pair: tuple[int, int] | None,
    limit: int = 2,
) -> list[list[int]]:
    by_top: dict[int, list[int]] = {}
    for idx, row in enumerate(rows):
        by_top.setdefault(int(row["top"]), []).append(idx)
    for lst in by_top.values():
        lst.sort(key=lambda i: (int(rows[i]["bottom"]), int(rows[i]["body_color"]), i))

    solutions: list[list[int]] = []
    n = len(rows)
    frame_color = int(frame["body_color"])

    def dfs(pos: int, current_hint: int, prev_body: int | None, used_mask: int, order: list[int]) -> None:
        if len(solutions) >= limit:
            return
        if pos == n:
            if current_hint == int(frame["bottom_anchor"]):
                solutions.append(order.copy())
            return

        for idx in by_top.get(int(current_hint), []):
            bit = 1 << idx
            if used_mask & bit:
                continue
            row = rows[idx]
            body = int(row["body_color"])
            if pos == 0 and _touches_forbidden(frame_color, body, forbidden_pair):
                continue
            if prev_body is not None and _touches_forbidden(prev_body, body, forbidden_pair):
                continue
            if pos == n - 1 and _touches_forbidden(body, frame_color, forbidden_pair):
                continue
            order.append(idx)
            dfs(pos + 1, int(row["bottom"]), body, used_mask | bit, order)
            order.pop()

    dfs(0, int(frame["top_anchor"]), None, 0, [])
    return solutions


def _build_output(frame_color: int, ordered_rows: list[dict]) -> np.ndarray:
    out = np.full((_HEIGHT, _WIDTH), int(frame_color), dtype=int)
    for r, row in enumerate(ordered_rows, start=1):
        out[r, 1 : _WIDTH - 1] = int(row["body_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble scattered strip-fragments into a left-right symmetric rectangle by combining color chaining with a visible forbidden color pair.

    Input structure:
    - The dominant color is background.
    - A fixed 2x2 legend sits in the top-left corner, isolated by a 1-cell background moat; its two colors are the only pair that may not become 4-neighbors in the output.
    - One multicolor component already forms the output border: its majority color is the frame body, and two minority pixels inside the frame mark the top and bottom chain anchors.
    - The remaining components are repeated 1x3 body strips; each exposes one hint color above its center and one below its center.

    Rule:
    - Remove the legend after reading the forbidden pair.
    - Treat every remaining non-background connected component as one piece; within each piece, the majority color is the body and the minority pixels are edge hints.
    - Starting from the frame's top anchor, place strips row by row so consecutive hint colors match.
    - If multiple unused strips match the same hint interface, keep only the orderings in which adjacent row colors never form the legend's forbidden pair.

    Key insight: The repeated strip geometry is not enough, and even the hint-color chain is intentionally ambiguous because one interface appears twice; the legend's forbidden pair is what decides which of those look-alike interfaces comes first, yielding a unique mirrored rectangle.
    """
    raw = np.asarray(grid, dtype=int)
    bg = int(get_bg(raw))
    forbidden_pair, cleaned = _read_forbidden_pair(raw, bg)
    frame, rows = _parse_cleaned_input(cleaned)
    solutions = _find_row_orders(frame, rows, forbidden_pair, limit=2)
    assert len(solutions) == 1, "row order must become unique once the forbidden pair is enforced"
    ordered_rows = [rows[idx] for idx in solutions[0]]
    out = _build_output(int(frame["body_color"]), ordered_rows)
    assert np.array_equal(out, out[:, ::-1]), "output must remain left-right symmetric"
    return out.astype(int)


def _choose_colors(bg: int) -> tuple[dict[str, int], list[int]] | None:
    available = [c for c in range(10) if c != bg]
    roles = ("frame", "A", "B", "C", "D", "X")

    for _ in range(200):
        link_colors = random.sample(available, 6)
        link = {f"l{i}": int(link_colors[i]) for i in range(6)}
        forbids = {
            "frame": {link["l0"], link["l5"]},
            "A": {link["l1"], link["l2"], link["l3"]},
            "B": {link["l2"], link["l4"]},
            "C": {link["l0"], link["l1"], link["l4"]},
            "D": {link["l1"], link["l3"], link["l5"]},
            "X": {link["l1"], link["l2"]},
        }

        assigned: dict[str, int] = {}
        used: set[int] = set()
        order = sorted(roles, key=lambda role: (-len(forbids[role]), role))

        def backtrack(pos: int) -> bool:
            if pos == len(order):
                return True
            role = order[pos]
            candidates = [c for c in available if c not in used and c not in forbids[role]]
            random.shuffle(candidates)
            for col in candidates:
                assigned[role] = int(col)
                used.add(int(col))
                if backtrack(pos + 1):
                    return True
                used.remove(int(col))
                del assigned[role]
            return False

        if backtrack(0):
            return assigned, [int(link[f"l{i}"]) for i in range(6)]
    return None


def _build_specs() -> tuple[list[tuple[str, str]], list[str]]:
    transitions = []
    row_roles = list(_ROW_ROLE_SEQ)
    for i, role in enumerate(row_roles):
        transitions.append((f"l{_LINK_PATTERN[i]}", f"l{_LINK_PATTERN[i + 1]}"))
        assert isinstance(role, str)
    return transitions, row_roles


def _build_output_and_patches() -> tuple[np.ndarray, int, np.ndarray, list[np.ndarray], tuple[int, int], list[dict]] | None:
    bg = random.randint(0, 9)
    chosen = _choose_colors(bg)
    if chosen is None:
        return None
    role_colors, link_colors = chosen
    link = {f"l{i}": int(link_colors[i]) for i in range(6)}
    transitions, row_roles = _build_specs()

    rows: list[dict] = []
    for role, (top_key, bottom_key) in zip(row_roles, transitions):
        rows.append(
            {
                "body_color": int(role_colors[role]),
                "top": int(link[top_key]),
                "bottom": int(link[bottom_key]),
            }
        )

    frame = {
        "body_color": int(role_colors["frame"]),
        "top_anchor": int(link["l0"]),
        "bottom_anchor": int(link["l5"]),
    }
    forbidden_pair = _norm_pair(int(role_colors[_FORBIDDEN_ROLE_PAIR[0]]), int(role_colors[_FORBIDDEN_ROLE_PAIR[1]]))

    constrained = _find_row_orders(frame, rows, forbidden_pair, limit=3)
    unconstrained = _find_row_orders(frame, rows, None, limit=3)
    if len(constrained) != 1 or len(unconstrained) < 2:
        return None

    ordered_rows = [rows[idx] for idx in constrained[0]]
    out = _build_output(int(frame["body_color"]), ordered_rows)

    frame_patch = np.full((_HEIGHT, _WIDTH), bg, dtype=int)
    frame_patch[0, :] = int(frame["body_color"])
    frame_patch[_HEIGHT - 1, :] = int(frame["body_color"])
    frame_patch[:, 0] = int(frame["body_color"])
    frame_patch[:, _WIDTH - 1] = int(frame["body_color"])
    frame_patch[1, _WIDTH // 2] = int(frame["top_anchor"])
    frame_patch[_HEIGHT - 2, _WIDTH // 2] = int(frame["bottom_anchor"])

    row_patches = []
    for row in rows:
        patch = np.full((3, 3), bg, dtype=int)
        patch[1, :] = int(row["body_color"])
        patch[0, 1] = int(row["top"])
        patch[2, 1] = int(row["bottom"])
        row_patches.append(patch)

    return out, bg, frame_patch, row_patches, forbidden_pair, rows


def _place_legend(canvas: np.ndarray, bg: int, forbidden_pair: tuple[int, int]) -> None:
    a, c = forbidden_pair
    canvas[:_LEGEND_BLOCK, :_LEGEND_BLOCK] = bg
    canvas[0:2, 0] = int(a)
    canvas[0:2, 1] = int(c)


def _scatter(frame_patch: np.ndarray, row_patches: list[np.ndarray], bg: int, forbidden_pair: tuple[int, int]) -> np.ndarray:
    rows = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    cols = random.randint(_CANVAS_MIN, _CANVAS_MAX)
    canvas = np.full((rows, cols), bg, dtype=int)

    canvas[:_LEGEND_BLOCK, :_LEGEND_BLOCK] = -1

    max_r = rows - _HEIGHT - 1
    max_c = cols - _WIDTH - 1
    if max_r < _LEGEND_BLOCK or max_c < _LEGEND_BLOCK:
        raise ValueError("canvas too small for frame")
    off_r = random.randint(_LEGEND_BLOCK, max_r)
    off_c = random.randint(_LEGEND_BLOCK, max_c)

    canvas[off_r : off_r + _HEIGHT, off_c : off_c + _WIDTH] = -1
    for patch in row_patches:
        r0, c0 = random_free_location(
            canvas,
            patch,
            transparent=bg,
            padding=_SCATTER_PADDING,
            padding_connectivity=8,
        )
        paste(canvas, patch, r0, c0, transparent=bg)

    canvas[off_r : off_r + _HEIGHT, off_c : off_c + _WIDTH] = bg
    paste(canvas, frame_patch, off_r, off_c, transparent=bg)

    canvas[canvas == -1] = bg
    _place_legend(canvas, bg, forbidden_pair)
    return canvas


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    built = _build_output_and_patches()
    if built is None:
        return None
    out, bg, frame_patch, row_patches, forbidden_pair, _rows = built
    try:
        inp = _scatter(frame_patch, row_patches, bg, forbidden_pair)
        got = main(inp)
    except Exception:
        return None
    if np.array_equal(got, out):
        return inp, out
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid combo input for e8686506")
