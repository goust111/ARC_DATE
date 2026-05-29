# Combo variant of b5ca7ac4 (index 19)
# combo_spec: D8/3+D9/2
# Name: 碰撞检测+边框合并

from __future__ import annotations

import random

import numpy as np

from arc_utils import crop_bbox, get_bg, paste


# Design analysis
# a) 具体化与行为追踪
#    - 输入对象仍是原题那批 5x5 框：外圈 16 格必须是 8 或 2，中心 3x3 是单一纯色填充。
#    - 第一步完全保留原题的“移动到左右边缘”的判定：青框按输入列升序处理，红框按输入列降序处理；
#      若其 5 行跨度与同组某个已占据边缘列的框重叠，则该框退到第二列。
#    - 第二步不再把这些 5x5 patch 原样贴出。先把每个移动后的 5x5 全部填成自己的内部色，
#      再只在“该组所有移动后方块并集的外轮廓”上恢复框色。于是并排且纵向相撞的 edge/inward 对
#      会失去共享竖墙，输出变成连通拼接块；没有发生碰撞的方块仍保持普通 5x5 框样子。
#    - 具体例子：若两个青框最终落在 (r,0) 与 (r+1,5)，原题会看到两条紧挨的竖框线；本变体中，
#      它们重叠行段里的 col=4/5 共享墙会消失，只留下整块外边界。
# b) 复杂度校准
#    - 不引入新对象、新颜色映射或额外排序键，只在“完成全部移动之后”新增一次轮廓重算。
#    - 视觉差异集中在少量可重复的 seam 消失现象上；generator 强制 inward 框与某个 edge 框发生强碰撞，
#      避免退化成原题那种单纯并排摆放。
# c) 隐式假设审查
#    - 原 solver 默认 patch 可以直接 paste；融合后这个前提失效，必须改成“先算目标位置，再按组重绘”。
#    - 原代码把 row-overlap 仅当作 inward 判定条件；现在它还决定后续是否出现共享墙消除，所以 generator
#      需要保证至少一部分 overlap 足够强，不能只满足最弱的一行接触。
#    - 组别识别仍然只能基于输入边框色，因为输出里的可见结构已经被并集合并，不能再从单个完整 5x5 反推。
# d) 放置合法性
#    - 输入仍要求 5x5 方块彼此不重叠；输出仍只使用每侧两列，且同列对象不重叠，因此不会越界。
#    - 第二步的“边框合并”只发生在同组移动后区域内部，不会改变对象占据的 5x5 并集范围，所以不存在新越界。
# e) 视觉可观察性
#    - 人类能直接看到：对象先按原题方式靠边，再在发生碰撞的位置失去内部隔墙，形成更大的连通框架。
#    - 这一规律只依赖可见的几何接触，不依赖颜色编号大小或任何不可视属性。
# f) 可视化整洁度
#    - 合并只移除共享内墙，不额外添加标记；因此输出仍局限在左右两块堆叠区，没有遮挡到背景中的其它区域。
#    - 每个格子始终只属于一个 5x5 方块，不存在多层覆写同一像素导致的杂乱。
# g) 变化的可推断性
#    - 颜色本身不做查表变化；变化是“哪些原本是边框的格子在碰撞后变成了内部色”。
#    - generator 禁止内部色使用 2/8/bg，使共享墙消失后更容易肉眼看出“框色被去掉了”而不是颜色巧合。
# 9) 新隐式假设审查
#    - main() 假设输入中的真对象仍是完整、内部纯色的 5x5 框；generate_input() 保证这一点。
#    - main() 在边框合并阶段按同组 5x5 占据并集重算外轮廓；这要求输入推导出的输出最多只有原题那两列堆叠，
#      generator 继续沿用原题的 output-first 反推，确保该前提成立。
#    - main() 不尝试从已经合并后的输出中重新检测对象，只做单向构造；因此 generator 只需保证输入可检测，
#      不需要让输出仍可分解回独立 5x5。


GRID_H, GRID_W = 22, 22
SQ = 5
AZURE, RED = 8, 2
FRAME_COLORS = {AZURE, RED}
N_AZURE_RANGE = (3, 4)
N_RED_RANGE = (3, 5)
MAX_INWARD = 2
INPUT_COL_RETRIES = 50
ROW_ASSIGN_RETRIES = 80
GEN_RETRIES = 500
MIN_MERGE_DELTA = 4


def _make_patch(frame_color: int, inner_color: int) -> np.ndarray:
    """Create a SQxSQ bordered patch with a uniform 3x3 interior."""
    patch = np.full((SQ, SQ), int(frame_color), dtype=int)
    patch[1:-1, 1:-1] = int(inner_color)
    return patch


def _rows_overlap(r_a: int, r_b: int) -> bool:
    """Whether two SQ-tall row spans overlap at all."""
    return abs(int(r_a) - int(r_b)) < SQ


def _rows_merge_visibly(r_a: int, r_b: int) -> bool:
    """Strong enough overlap to remove at least two cells of shared wall."""
    return abs(int(r_a) - int(r_b)) <= 1


def _rects_overlap(r1: int, c1: int, r2: int, c2: int) -> bool:
    """Whether two SQxSQ rectangles overlap."""
    return abs(int(r1) - int(r2)) < SQ and abs(int(c1) - int(c2)) < SQ


def _any_pairwise_overlap(positions: list[tuple[int, int, np.ndarray]]) -> bool:
    """Return True if any two (row, col, patch) entries overlap as SQxSQ rectangles."""
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            if _rects_overlap(positions[i][0], positions[i][1], positions[j][0], positions[j][1]):
                return True
    return False


def _parse_square_patch(patch: np.ndarray) -> tuple[int, int] | None:
    """Return (frame_color, inner_color) for a valid input square; otherwise None."""
    border = np.concatenate([patch[0, :], patch[-1, :], patch[1:-1, 0], patch[1:-1, -1]])
    if int(border.min()) != int(border.max()):
        return None
    frame_color = int(border[0])
    if frame_color not in FRAME_COLORS:
        return None

    inner = patch[1:-1, 1:-1]
    if int(inner.min()) != int(inner.max()):
        return None
    inner_color = int(inner[0, 0])
    if inner_color == frame_color:
        return None
    return frame_color, inner_color


def _detect_squares(grid: np.ndarray) -> list[dict]:
    """Scan 5x5 windows and extract all valid bordered squares from the input."""
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    squares: list[dict] = []
    for r0 in range(h - SQ + 1):
        for c0 in range(w - SQ + 1):
            patch = crop_bbox(g, (int(r0), int(c0), int(r0 + SQ - 1), int(c0 + SQ - 1)))
            parsed = _parse_square_patch(patch)
            if parsed is None:
                continue
            frame_color, inner_color = parsed
            squares.append(
                {
                    "frame": int(frame_color),
                    "inner": int(inner_color),
                    "r0": int(r0),
                    "c0": int(c0),
                    "patch": patch,
                }
            )
    return squares


def _assign_destinations(squares: list[dict], width: int) -> list[dict]:
    """Apply the original edge/inward stacking rule and return moved-square metadata."""
    placements: list[dict] = []

    for frame_color, ascending, edge_col, inward_col in (
        (AZURE, True, 0, SQ),
        (RED, False, int(width - SQ), int(width - 2 * SQ)),
    ):
        group = [s for s in squares if int(s["frame"]) == frame_color]
        group.sort(key=lambda s: int(s["c0"]), reverse=not ascending)

        edge_rows: list[int] = []
        for s in group:
            r0 = int(s["r0"])
            is_edge = not any(_rows_overlap(r0, er) for er in edge_rows)
            dest_c = int(edge_col if is_edge else inward_col)
            placements.append(
                {
                    "frame": int(frame_color),
                    "inner": int(s["inner"]),
                    "r0": int(r0),
                    "c0": int(s["c0"]),
                    "dest_c": int(dest_c),
                    "is_edge": bool(is_edge),
                    "patch": s["patch"],
                }
            )
            if is_edge:
                edge_rows.append(r0)

    return placements


def _paint_group_with_merged_borders(out: np.ndarray, placements: list[dict]) -> None:
    """Render one side after movement, removing any shared internal walls."""
    if not placements:
        return

    h, w = out.shape
    union_mask = np.zeros((h, w), dtype=bool)

    for p in placements:
        r0 = int(p["r0"])
        c0 = int(p["dest_c"])
        out[r0 : r0 + SQ, c0 : c0 + SQ] = int(p["inner"])
        union_mask[r0 : r0 + SQ, c0 : c0 + SQ] = True

    for p in placements:
        r0 = int(p["r0"])
        c0 = int(p["dest_c"])
        frame_color = int(p["frame"])
        for dr in range(SQ):
            for dc in range(SQ):
                if dr not in {0, SQ - 1} and dc not in {0, SQ - 1}:
                    continue
                rr = int(r0 + dr)
                cc = int(c0 + dc)
                exposed = False
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if not (0 <= nr < h and 0 <= nc < w) or not bool(union_mask[nr, nc]):
                        exposed = True
                        break
                if exposed:
                    out[rr, cc] = frame_color


def _render_output(shape: tuple[int, int], bg: int, placements: list[dict], *, merge_shared: bool) -> np.ndarray:
    """Render the post-move scene either with or without the fusion step."""
    out = np.full(shape, int(bg), dtype=int)
    if not merge_shared:
        for p in placements:
            paste(out, p["patch"], int(p["r0"]), int(p["dest_c"]))
        return out

    for frame_color in (AZURE, RED):
        group = [p for p in placements if int(p["frame"]) == frame_color]
        _paint_group_with_merged_borders(out, group)
    return out


def _fusion_delta_from_input(grid: np.ndarray) -> int:
    """Number of output cells changed by the merge step relative to raw moved squares."""
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    placements = _assign_destinations(_detect_squares(g), g.shape[1])
    merged = _render_output(g.shape, bg, placements, merge_shared=True)
    plain = _render_output(g.shape, bg, placements, merge_shared=False)
    return int(np.count_nonzero(merged != plain))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move red/teal 5x5 frames to their side stacks, then erase any shared inner walls created by those moves.

    Input structure:
    - The background is the most common color.
    - Non-overlapping 5x5 squares have a uniform teal (8) or red (2) border and a uniform 3x3 interior color.

    Rule:
    - First perform the original stacking move: teal frames go to the left side and red frames go to the right side, each keeping its original top row; teal are processed by increasing input column and red by decreasing input column.
    - Within each side, a square that row-overlaps any already edge-placed square is shifted one square-width inward instead of staying on the edge column.
    - After all squares of one side have been moved, treat the moved 5x5 regions as a single union: refill every moved square with its own interior color, then redraw the frame color only on cells that lie on the outer boundary of that union.

    Key insight:
    - Ordering still matters for deciding edge vs inward placement, but the final image is not a direct paste of those moved squares; you must complete the movement first and then recompute the visible contour so any collision-created shared border disappears.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    placements = _assign_destinations(_detect_squares(g), g.shape[1])
    return _render_output(g.shape, bg, placements, merge_shared=True)


def _assign_output_rows(n: int) -> tuple[list[int], list[int]] | None:
    """Pick edge and inward rows while forcing at least one visibly merged edge/inward pair."""
    max_inward = min(MAX_INWARD, n - 1, n // 2)
    n_inward = random.randint(1, max_inward)
    n_edge = n - n_inward

    all_rows = list(range(GRID_H - SQ + 1))
    random.shuffle(all_rows)
    edge_rows: list[int] = []
    for row in all_rows:
        if all(not _rows_overlap(row, er) for er in edge_rows):
            edge_rows.append(int(row))
            if len(edge_rows) == n_edge:
                break
    if len(edge_rows) < n_edge:
        return None

    for _ in range(ROW_ASSIGN_RETRIES):
        targets = random.sample(edge_rows, n_inward)
        inward_rows: list[int] = []
        ok = True
        for er in targets:
            candidates = [r for r in (er - 1, er, er + 1) if 0 <= r <= GRID_H - SQ]
            random.shuffle(candidates)
            chosen = None
            for row in candidates:
                if any(_rows_overlap(row, ir) for ir in inward_rows):
                    continue
                chosen = int(row)
                break
            if chosen is None:
                ok = False
                break
            inward_rows.append(chosen)
        if not ok:
            continue
        if all(any(_rows_merge_visibly(ir, er) for er in edge_rows) for ir in inward_rows):
            return edge_rows, inward_rows

    return None


def _build_side_placements(
    edge_rows: list[int],
    inward_rows: list[int],
    colors: list[int],
    frame_color: int,
    edge_col: int,
    inward_col: int,
) -> list[dict]:
    """Create moved-square metadata for one side using fixed output rows and colors."""
    placements: list[dict] = []
    idx = 0
    for row in edge_rows:
        patch = _make_patch(frame_color, colors[idx])
        placements.append(
            {
                "frame": int(frame_color),
                "inner": int(colors[idx]),
                "r0": int(row),
                "dest_c": int(edge_col),
                "is_edge": True,
                "patch": patch,
            }
        )
        idx += 1
    for row in inward_rows:
        patch = _make_patch(frame_color, colors[idx])
        placements.append(
            {
                "frame": int(frame_color),
                "inner": int(colors[idx]),
                "r0": int(row),
                "dest_c": int(inward_col),
                "is_edge": False,
                "patch": patch,
            }
        )
        idx += 1
    return placements


def _derive_input_positions(
    placements: list[dict],
    ascending: bool,
) -> list[tuple[int, int, np.ndarray]] | None:
    """Assign input columns that reproduce the requested edge/inward placement pattern."""
    edges = [(int(p["r0"]), int(p["dest_c"]), p["patch"]) for p in placements if bool(p["is_edge"])]
    inwards = [(int(p["r0"]), int(p["dest_c"]), p["patch"]) for p in placements if not bool(p["is_edge"])]

    inward_deps = [[i for i, (er, _, _) in enumerate(edges) if _rows_overlap(ir, er)] for (ir, _, _) in inwards]

    n = len(placements)
    col_range = list(range(0, GRID_W - SQ + 1))
    if len(col_range) < n:
        return None

    for _ in range(INPUT_COL_RETRIES):
        edge_indices = list(range(len(edges)))
        random.shuffle(edge_indices)
        order: list[tuple[str, int]] = [("edge", i) for i in edge_indices]

        inward_indices = list(range(len(inwards)))
        random.shuffle(inward_indices)
        for inward_idx in inward_indices:
            latest_pos = max(
                (
                    pos
                    for dep in inward_deps[inward_idx]
                    for pos, (typ, idx) in enumerate(order)
                    if typ == "edge" and idx == dep
                ),
                default=-1,
            )
            insert_pos = random.randint(latest_pos + 1, len(order))
            order.insert(insert_pos, ("inward", inward_idx))

        ordered = [edges[idx] if typ == "edge" else inwards[idx] for typ, idx in order]

        chosen_cols = sorted(random.sample(col_range, n))
        if not ascending:
            chosen_cols = chosen_cols[::-1]

        result: list[tuple[int, int, np.ndarray]] = []
        ok = True
        for idx, (row, out_col, patch) in enumerate(ordered):
            in_col = int(chosen_cols[idx])
            if in_col == out_col:
                ok = False
                break
            result.append((int(row), in_col, patch))
        if ok and not _any_pairwise_overlap(result):
            return result

    return None


def _gen_attempt() -> np.ndarray | None:
    """Single attempt to generate a valid input grid for the combo behavior."""
    bg = random.choice([c for c in range(10) if c not in FRAME_COLORS])
    n_azure = random.randint(*N_AZURE_RANGE)
    n_red = random.randint(*N_RED_RANGE)

    azure_rows = _assign_output_rows(n_azure)
    if azure_rows is None:
        return None
    red_rows = _assign_output_rows(n_red)
    if red_rows is None:
        return None
    azure_edge_rows, azure_inward_rows = azure_rows
    red_edge_rows, red_inward_rows = red_rows

    usable_colors = [c for c in range(10) if c not in {bg, AZURE, RED}]
    if len(usable_colors) < max(n_azure, n_red):
        return None
    azure_colors = random.sample(usable_colors, n_azure)
    red_colors = random.sample(usable_colors, n_red)

    azure_placements = _build_side_placements(azure_edge_rows, azure_inward_rows, azure_colors, AZURE, 0, SQ)
    red_placements = _build_side_placements(
        red_edge_rows,
        red_inward_rows,
        red_colors,
        RED,
        GRID_W - SQ,
        GRID_W - 2 * SQ,
    )
    placements = azure_placements + red_placements

    output = _render_output((GRID_H, GRID_W), bg, placements, merge_shared=True)
    plain_output = _render_output((GRID_H, GRID_W), bg, placements, merge_shared=False)
    if int(np.count_nonzero(output != plain_output)) < MIN_MERGE_DELTA:
        return None

    azure_input = _derive_input_positions(azure_placements, ascending=True)
    if azure_input is None:
        return None
    red_input = _derive_input_positions(red_placements, ascending=False)
    if red_input is None:
        return None

    all_positions = azure_input + red_input
    if _any_pairwise_overlap(all_positions):
        return None

    inp = np.full((GRID_H, GRID_W), int(bg), dtype=int)
    for row, col, patch in all_positions:
        paste(inp, patch, int(row), int(col))

    try:
        if _fusion_delta_from_input(inp) < MIN_MERGE_DELTA:
            return None
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} attempts")
