# Variant of 195c6913
# Dimension: D7 — 模板查找机制 (symbol_instructions)
# Change: 左上角一排2x2单色块定义颜色循环顺序（从左到右），标记的起始循环位由其颜色在循环中的位置决定。
#   → 模板改为左上角的 L / T 形 2x2 色块符号，循环顺序由该角落符号的顺时针外圈读法决定。
#
# Design analysis
# a) 具体化与行为追踪
#    - 旧题模板是同一行上的 2x2 色块，读序几乎等于扫描顺序。
#    - 新题模板固定放在左上角背景区，但不再排成一行，只会出现两种小词典形状：
#      L: block top-left coords (r,c),(r,c+3),(r+3,c)
#      T: block top-left coords (r,c),(r,c+3),(r,c+6),(r+3,c+3)
#    - 每个 block 之间仍隔 1 列/1 行背景，保持“它们是独立 2x2 色块”的视觉线索。
#    - 读取规则具体化为“先读符号主干，再读分支”：
#      L 读成 [左上拐角, 左下腿端, 右上臂端]
#      T 读成 [顶中主干, 下方柄端, 左臂, 右臂]
#    - 小例子：若 L 三块颜色分别是 左上=红、左下=蓝、右上=黄，则循环是 红→蓝→黄→红。
#      旧逻辑会把扫描到的 2x2 块当作一排或按行扫描；新逻辑必须先识别形状，再按新的骨架读序组装循环。
# b) 复杂度校准
#    - 只允许 L/T 两类，避免“任意连通形状”导致匹配歧义或样例稀释。
#    - 模板色强制互异；否则 L/T 的读序变化在人眼上很难确认。
#    - 读序不能只是旧顺序的循环平移，因为循环序列在“按 marker 颜色定相位”的原规则下对输出是等价的。
#    - marker 允许取循环中的任意颜色，强化“起始相位由 marker 自身颜色决定”这一原有机制。
# c) 隐式假设审查
#    - 旧 main() 中 `pattern.append(...)` 依赖“模板就是扫描顺序”，在 L/T 下不成立，必须改成：
#      先收集左上角全部 2x2 候选块 → 归一化坐标 → 匹配词典形状 → 用形状专属顺序读色。
#    - 旧 generator 只给 marker 第一种模板色，会弱化新模板顺序的影响；这里改成从循环中抽 marker 颜色。
#    - 旧 `_build_band()` 只给 rows 1..3 留模板背景；T 形最低会占到 row 6，所以清空区要扩大到 rows 1..6。
# d) 放置合法性
#    - 模板 anchor 仍只取 row 1 或 2，col 固定为 1；这样始终在左上角，且与 path band 的预留空区对齐。
#    - T 最宽占到 col 8，最高/最低占到 row anchor..anchor+4；generator 会显式检查这些格子都在背景中。
#    - end block 仍只放右下背景区，避免被模板解析误认。
# e) 视觉可观察性
#    - 规则完全由可见形状决定，不使用颜色 id 排序。
#    - L 与 T 在 2-4 个样例里肉眼可区分，且输出上的循环换色会随该模板读序变化。
# f) 可视化整洁度
#    - 模板仍集中在左上角独立背景区，不与 path 或 marker 叠加。
#    - 仅增加模板内部布局复杂度，不新增额外小噪声对象。
# g) 颜色变化可推断性
#    - 模板色数量限制为 3 或 4，且互不重复。
#    - marker 颜色直接来自模板色集合，因此“哪种颜色从循环中的哪里开始”能由输入结构反推。

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import label_mask


_TEMPLATE_SHAPES: dict[str, dict[str, list[tuple[int, int]]]] = {
    "L": {
        "slots": [(0, 0), (0, 3), (3, 0)],
        "order": [(0, 0), (3, 0), (0, 3)],
    },
    "T": {
        "slots": [(0, 0), (0, 3), (0, 6), (3, 3)],
        "order": [(0, 3), (3, 3), (0, 0), (0, 6)],
    },
}


def _assert_path_single_component(g: np.ndarray, path_color: int) -> None:
    """Eval invariant: path_color forms a single 4-connected component."""
    _, n_path = label_mask(g == path_color, connectivity=4)
    assert n_path == 1, f"path_color must be a single 4-connected component, got {n_path}"


def _assert_marker_embedded_in_path(
    g: np.ndarray, markers: list[tuple[int, int, int]], path_color: int
) -> None:
    """Eval invariant: a col-0 marker keeps path cells above and below it."""
    H, _ = g.shape
    for r, c, _ in markers:
        if c != 0:
            continue
        if r > 0:
            assert int(g[r - 1, c]) == path_color, (
                f"marker at ({r},{c}) up-neighbor ({r-1},{c})={int(g[r-1,c])} "
                f"must be path_color {path_color}"
            )
        if r < H - 1:
            assert int(g[r + 1, c]) == path_color, (
                f"marker at ({r},{c}) down-neighbor ({r+1},{c})={int(g[r+1,c])} "
                f"must be path_color {path_color}"
            )


def _is_solid_2x2(g: np.ndarray, r: int, c: int, color: int) -> bool:
    return (
        0 <= r < g.shape[0] - 1
        and 0 <= c < g.shape[1] - 1
        and int(g[r, c]) == color
        and int(g[r + 1, c]) == color
        and int(g[r, c + 1]) == color
        and int(g[r + 1, c + 1]) == color
    )


def _parse_template(
    g: np.ndarray, template_bg: int, path_color: int
) -> tuple[list[int], list[tuple[int, int]], str, tuple[int, int]]:
    """Read the left-corner L/T template and return its cycle order."""
    H, W = g.shape
    blocks: dict[tuple[int, int], int] = {}
    seen: set[tuple[int, int]] = set()
    for r in range(min(10, H - 1)):
        for c in range(min(14, W - 1)):
            if (r, c) in seen:
                continue
            v = int(g[r, c])
            if v == template_bg or v == path_color:
                continue
            if _is_solid_2x2(g, r, c, v):
                blocks[(r, c)] = v
                for rr in (r, r + 1):
                    for cc in (c, c + 1):
                        seen.add((rr, cc))

    assert blocks, "template blocks not found in top-left region"
    min_r = min(r for r, _ in blocks)
    min_c = min(c for _, c in blocks)
    norm = {(r - min_r, c - min_c) for r, c in blocks}

    shape_name = None
    for name, spec in _TEMPLATE_SHAPES.items():
        if norm == set(spec["slots"]):
            shape_name = name
            break
    assert shape_name is not None, f"unrecognized template shape: {sorted(norm)}"

    order = _TEMPLATE_SHAPES[shape_name]["order"]
    ordered_blocks = [(min_r + dr, min_c + dc) for dr, dc in order]
    assert all(pos in blocks for pos in ordered_blocks), "template missing a required block"

    pattern = [int(blocks[pos]) for pos in ordered_blocks]
    pat_cells: list[tuple[int, int]] = []
    for br, bc in ordered_blocks:
        pat_cells.extend([(br, bc), (br + 1, bc), (br, bc + 1), (br + 1, bc + 1)])
    return pattern, pat_cells, shape_name, (min_r, min_c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Simulate path-following particles whose paint cycle is encoded by a corner symbol.

    Input structure:
    - Top-left: several 2x2 monochrome blocks arranged as an L or T symbol in a bg pocket.
    - Isolated single pixels whose colors come from that symbol: start markers embedded in the path.
    - One separate 2x2 monochrome block of another color: the terminal collision color.

    Rule:
    - First decode the cycle by recognizing whether the corner symbol is L-shaped or T-shaped.
    - Read the symbol by its visible backbone first and only then its side branches to obtain the repeating color cycle.
    - Each marker walks straight on the path, advances one step in the cycle per path cell, and still alternates CCW/CW turns on successive collisions.
    - If the scheduled turn is not a path cell, the particle stops; collisions with bg still receive the terminal color.

    Key insight: The motion rule is unchanged, but the usable cycle now depends on parsing a small corner glyph and using its shape-specific read order rather than scanning a horizontal row.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape

    template_bg = int(g[0, 0])

    counts = Counter(map(int, g.flatten()))
    path_color = int(
        max([(cnt, int(col)) for col, cnt in counts.items() if int(col) != template_bg])[1]
    )
    _assert_path_single_component(g, path_color)

    pattern, pat_cells, _shape_name, _anchor = _parse_template(g, template_bg, path_color)
    pat_set = set(map(int, pattern))

    end_color = None
    end_pos = None
    for r in range(H - 1):
        for c in range(W - 1):
            v = int(g[r, c])
            if v == template_bg or v == path_color or v in pat_set:
                continue
            if _is_solid_2x2(g, r, c, v):
                end_color = int(v)
                end_pos = (int(r), int(c))
                break
        if end_pos is not None:
            break

    markers: list[tuple[int, int, int]] = []
    for r in range(H):
        for c in range(W):
            v = int(g[r, c])
            if v not in pat_set:
                continue
            same = 0
            has_path = False
            for dr, dc in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < H and 0 <= cc < W:
                    if int(g[rr, cc]) == v:
                        same += 1
                    if int(g[rr, cc]) == path_color:
                        has_path = True
            if same == 0 and has_path:
                markers.append((int(r), int(c), int(v)))

    _assert_marker_embedded_in_path(g, markers, path_color)

    out = g.copy()
    for r, c in pat_cells:
        out[r, c] = template_bg
    if end_pos is not None:
        er, ec = end_pos
        out[er : er + 2, ec : ec + 2] = template_bg

    dir_order = [(0, 1), (1, 0), (0, -1), (-1, 0)]

    per_marker_cells: list[set[tuple[int, int]]] = []
    for sr, sc, sv in markers:
        init = None
        for dr, dc in dir_order:
            rr = int(sr + dr)
            cc = int(sc + dc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                init = (int(dr), int(dc))
                break
        if init is None or not pattern:
            per_marker_cells.append(set())
            continue

        r, c = int(sr), int(sc)
        dr, dc = init
        idx = int(pattern.index(int(sv))) if int(sv) in pattern else 0
        hits = 0
        visited: set[tuple[int, int, int, int, int, int]] = set()
        my_cells: set[tuple[int, int]] = set()

        for _ in range(H * W * 4):
            state = (int(r), int(c), int(dr), int(dc), int(idx), int(hits & 1))
            if state in visited:
                break
            visited.add(state)

            nr = int(r + dr)
            nc = int(c + dc)
            if not (0 <= nr < H and 0 <= nc < W):
                break

            if int(g[nr, nc]) == path_color:
                idx = int((idx + 1) % len(pattern))
                out[nr, nc] = int(pattern[idx])
                my_cells.add((int(nr), int(nc)))
                r, c = nr, nc
                continue

            if int(g[nr, nc]) == template_bg and end_color is not None:
                out[nr, nc] = int(end_color)

            left = (-int(dc), int(dr))
            right = (int(dc), -int(dr))
            tdr, tdc = left if (int(hits) % 2 == 0) else right
            rr = int(r + tdr)
            cc = int(c + tdc)
            if 0 <= rr < H and 0 <= cc < W and int(g[rr, cc]) == path_color:
                dr, dc = int(tdr), int(tdc)
                hits += 1
                continue
            break

        per_marker_cells.append(my_cells)

    for i in range(len(per_marker_cells)):
        for j in range(i + 1, len(per_marker_cells)):
            overlap = per_marker_cells[i] & per_marker_cells[j]
            assert not overlap, (
                f"marker trajectories {i},{j} overlap on {len(overlap)} cell(s)"
            )

    return out


def _build_band(H: int, W: int, lobe_rows: list[int] | None = None) -> np.ndarray:
    """Return a thick anti-diagonal path band with a clear top-left template pocket."""
    bot_center = random.randint(0, W // 4)
    centers = np.linspace(W - 1, bot_center, H).astype(float)
    half_w = np.array(
        [0.5 * (3 + (1.0 - abs(2.0 * r / max(1, H - 1) - 1.0)) * (W - 6)) for r in range(H)]
    )

    c_left = np.zeros(H, dtype=int)
    c_right = np.zeros(H, dtype=int)

    cc_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )
    w_jit = np.convolve(
        np.array([random.randint(-3, 3) for _ in range(H)], dtype=float),
        np.ones(3) / 3,
        mode="same",
    )

    for r in range(H):
        c_left_param = centers[r] - half_w[r]
        if c_left_param <= 0:
            c_left[r] = 0
            c_right[r] = int(round(centers[r] + cc_jit[r] + half_w[r] + w_jit[r] / 2))
        else:
            w = max(2, min(W, int(round(2 * half_w[r] + w_jit[r]))))
            cc = centers[r] + cc_jit[r]
            c_left[r] = int(round(cc - w / 2))
            c_right[r] = c_left[r] + w - 1

    if lobe_rows:
        lmin = min(lobe_rows) - 1
        lmax = H - 1
        for r in range(H):
            d_outside = max(0, lmin - r, r - lmax)
            c_left[r] = min(c_left[r], 2 * d_outside)

    # D7 adaptation: T-shaped templates can reach down to row 6, so the bg pocket
    # for the corner symbol must stay open through rows 1..6.
    for r in range(1, min(7, H)):
        c_left[r] = max(c_left[r], 12)
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_right[0] = W - 1
    c_left[0] = max(c_left[0], W - 6)
    c_left[H - 1] = 0
    c_right[H - 1] = max(5, min(W - 1, c_right[H - 1]))

    for _ in range(2):
        for r in range(1, H - 1):
            if c_left[r] > c_left[r - 1] and c_left[r] > c_left[r + 1]:
                c_left[r] = max(c_left[r - 1], c_left[r + 1])

    for r in range(1, H):
        c_left[r] = min(c_left[r], c_left[r - 1] + 2)
        c_right[r] = min(c_right[r], c_right[r - 1] + 2)
        if c_left[r] > c_right[r - 1]:
            c_left[r] = c_right[r - 1]
        if c_right[r] < c_left[r - 1]:
            c_right[r] = c_left[r - 1]
        if c_right[r] < c_left[r]:
            c_right[r] = c_left[r]

    c_left = np.clip(c_left, 0, W - 1)
    c_right = np.clip(c_right, 0, W - 1)
    c_right[0] = W - 1
    c_left[H - 1] = 0

    mask = np.zeros((H, W), dtype=bool)
    for r in range(H):
        mask[r, c_left[r] : c_right[r] + 1] = True
    return mask


def _carve_fjord(mask: np.ndarray, r1: int, r2: int) -> bool:
    """Carve a bg separator between two marker channels without disconnecting the path."""
    H, W = mask.shape
    sep_h = random.randint(2, 3)
    sr_min = r1 + 2
    sr_max = r2 - 1 - sep_h
    if sr_max < sr_min:
        return False
    sr = random.randint(sr_min, sr_max)

    cur_top = sr
    made_carve = False
    for k in range(W - 1):
        trial = mask.copy()
        for dr in range(sep_h):
            rr = cur_top + dr
            if 0 <= rr < H:
                trial[rr, k] = False
        _, n_trial = label_mask(trial, connectivity=4)
        if n_trial != 1:
            break
        np.copyto(mask, trial)
        made_carve = True
        if random.random() < 0.3 and cur_top > 0:
            cur_top -= 1
    return made_carve


def _template_blocks(shape_name: str, top_row: int, left_col: int = 1) -> list[tuple[int, int]]:
    return [(top_row + dr, left_col + dc) for dr, dc in _TEMPLATE_SHAPES[shape_name]["slots"]]


def _gen_attempt() -> np.ndarray | None:
    H = W = random.randint(20, 30)
    n_markers = 2 if random.random() < 0.7 else 1

    palette = list(range(10))
    random.shuffle(palette)
    bg = palette[0]
    path_color = palette[1]
    end_color = palette[2]

    template_shape = random.choice(["L", "T"])
    n_blocks = len(_TEMPLATE_SHAPES[template_shape]["order"])
    cycle_colors = random.sample(palette[3:], n_blocks)

    if n_markers == 1:
        r1 = random.randint(H // 2, H - 4)
        markers = [r1]
    else:
        r1 = random.randint(H // 3, H // 2)
        r2 = random.randint(max(r1 + 5, 2 * H // 3), H - 4)
        markers = [r1, r2]

    mask = _build_band(H, W, lobe_rows=markers)
    if mask[0, 0]:
        return None
    _, n_path = label_mask(mask, connectivity=4)
    if n_path != 1:
        return None
    if not mask[0, :].any() or not mask[:, W - 1].any():
        return None
    if not mask[H - 1, :].any() or not mask[:, 0].any():
        return None
    for r in markers:
        if not (mask[r, 0] and mask[r - 1, 0] and mask[r + 1, 0]):
            return None

    if n_markers == 2:
        trial = mask.copy()
        if not _carve_fjord(trial, r1, r2):
            return None
        use_interior = random.random() < 0.25
        if use_interior:
            K = random.randint(2, 4)
            pre_carve_cells = mask & ~trial
            for c in range(min(K, W)):
                trial[:, c] |= pre_carve_cells[:, c]
        _, nc = label_mask(trial, connectivity=4)
        if nc != 1:
            return None
        if not trial[0, :].any() or not trial[:, W - 1].any():
            return None
        if not trial[H - 1, :].any() or not trial[:, 0].any():
            return None
        for r in markers:
            if not (trial[r, 0] and trial[r - 1, 0] and trial[r + 1, 0]):
                return None
        mask = trial

    _, n_bg = label_mask(~mask, connectivity=4)
    if n_bg < 2:
        return None

    template_row = random.choice([1, 2])
    block_tops = _template_blocks(template_shape, template_row, 1)
    for br, bc in block_tops:
        if br + 1 >= H or bc + 1 >= W:
            return None
        for dr in range(2):
            for dc in range(2):
                if mask[br + dr, bc + dc]:
                    return None

    g = np.full((H, W), bg, dtype=int)
    g[mask] = path_color

    for (br, bc), color in zip(block_tops, cycle_colors):
        g[br : br + 2, bc : bc + 2] = color

    end_candidates = []
    for r in range(H // 2, H - 1):
        for c in range(W // 2, W - 1):
            if not (g[r : r + 2, c : c + 2] == bg).all():
                continue
            r0, r1_ = max(0, r - 1), min(H, r + 3)
            c0, c1_ = max(0, c - 1), min(W, c + 3)
            if (g[r0:r1_, c0:c1_] == path_color).any():
                continue
            end_candidates.append((r, c))
    if not end_candidates:
        return None
    er, ec = random.choice(end_candidates)
    g[er : er + 2, ec : ec + 2] = end_color

    if n_markers == 1:
        marker_indices = [random.randrange(n_blocks)]
    else:
        marker_indices = random.sample(range(n_blocks), 2)
    for r, idx in zip(markers, marker_indices):
        g[r, 0] = cycle_colors[idx]

    _, n_path_after = label_mask(g == path_color, connectivity=4)
    if n_path_after != 1:
        return None

    if g[0, 0] != bg:
        return None

    cnt = Counter(map(int, g.flatten()))
    non_bg = sorted([(c, col) for col, c in cnt.items() if col != bg], reverse=True)
    if not non_bg or non_bg[0][1] != path_color:
        return None

    try:
        _ = main(g)
    except Exception:
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(300):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("generate_input: failed after 300 attempts")
