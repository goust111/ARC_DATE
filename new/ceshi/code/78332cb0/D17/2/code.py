# Variant of 78332cb0
# Dimension: D17 — 符号编码/图例 (symbol_instructions)
# Change: 无符号编码，所有信息直接由块的空间结构和边缘像素位置表达
#   -> 控制像素指示输出顺序

from __future__ import annotations

from collections import Counter
import itertools
import random

import numpy as np

from arc_utils import get_bg, is_contiguous, paste


"""
design_analysis:
  variant_core:
    original_rule:
      - 输入里有 3-4 个 5x5 单色连通块。
      - 原题通过块边缘上的前景接触位置来推断唯一链顺序和输出方向。
    new_rule:
      - 输入最外层恰好有一条“控制边”：top / bottom / left / right 之一。
      - 这条控制边上排成一行的彩色单像素，按扫描顺序读取。
      - 每个控制像素颜色与一个块的唯一前景颜色一一对应。
      - 控制像素颜色顺序直接给出输出中块的排列顺序。
      - 控制边是横向(top/bottom)时输出为横向条带；控制边是纵向(left/right)时输出为纵向条带。

  a_concretization_and_trace:
    input_geometry:
      - 块区域仍保留原题的 5x5 块 + 6 色分隔线布局，布局可为 2x2、横条、竖条。
      - 在块区域外额外扩出 1 行或 1 列作为控制边，避免与块/分隔线重叠。
      - 控制边只放 3 或 4 个彩色像素，其余单元保持背景色。
    example_trace:
      original:
        - 若 4 个块颜色分别为 [2, 4, 8, 9]，原 solver 会比较相邻边缘接触偏移，找出唯一链，比如 [8, 2, 9, 4]，并按该顺序输出。
      variant:
        - 现在顶部控制边从左到右是 [9, 2, 4, 8]。
        - solver 不再做块间边缘匹配，而是先切出块，再按颜色映射得到 [9块, 2块, 4块, 8块]。
        - 因为控制边是横向，输出渲染成横向条带。
      difference:
        - 原逻辑在“排序环节”依赖块之间的结构关系。
        - 新逻辑在“排序环节”依赖边界图例；块结构只负责提供要被搬运的对象本体。

  b_complexity_calibration:
    anti_oversimplification:
      - 控制颜色不是抽象标签，而是直接复用块的主体颜色，人眼可立即建立对应关系。
      - 每个样例有 3-4 个块，控制序列长度与块数一致，信息量足够且不冗长。
      - 块颜色必须互异；否则颜色匹配会歧义，D17 变体不可解。
    anti_overfitting:
      - 控制边位置在 top/bottom/left/right 间变化。
      - 输入块布局在 2x2 与条带间变化；条带时仍尽量使用与输出相反的内层方向，保持原题“先切块再重排”的观感。

  c_implicit_assumption_audit_for_main:
    original_lines_affected:
      - `_contacts()` 和 `_find_chain()` 原本假设“排序信息在块边缘接触里”。该前提在变体中失效，因此整个链搜索逻辑必须删除。
      - 原 `_split_blocks()` 默认整个输入都由块区组成。变体增加了额外控制边，因此 solver 必须先识别并剥离那 1 行/列，再沿用分隔线切块。
      - 原代码允许块颜色重复，因为排序不靠颜色。变体改为颜色匹配后，solver 必须假设“每块恰有一种非背景色，且各块颜色互异”；generator 负责保证。
      - 原输出方向由链方向决定。变体中该信息改由控制边方向编码，因此 solver 需要把控制边朝向映射到渲染轴。

  d_placement_legality:
    guarantees_needed:
      - 控制边与块区完全分离，只占据额外的最外层 1 行或 1 列，因此不会覆盖块或分隔线。
      - 去掉控制边后，剩余区域仍是原题可被分隔线整齐切开的 5x5 块布局。
      - 控制像素数量与块数相等，颜色集合与块颜色集合完全相同。

  e_visual_observability:
    human_view:
      - 观察者能看到边界上有一串颜色标签，且这些颜色在块里各出现一次。
      - 输出只是在按这串颜色顺序把块排成条带，因此 2-3 个样例足以归纳规则。
    why_not_hidden:
      - 排序不依赖颜色编号大小，而依赖控制像素在线上的可见排列顺序。
      - 输出方向也由“控制线是横的还是竖的”这个可见结构给出。

  f_visual_cleanliness:
    choices:
      - 控制像素放在单独边带，不与块或分隔线叠加。
      - 仍保留原题的背景色和分隔线色，块区和图例区的职责清晰。

  g_color_inference:
    constraints:
      - 每块单色且颜色互异，控制像素直接复用这些颜色，不引入额外颜色映射。
      - 不使用颜色值大小、固定色义等不可见规则。

  generator_strategy:
    approach: output_first
    steps:
      - 选择块数 3 或 4，选择输出方向 h/v。
      - 为输出顺序挑选互异颜色，并为每种颜色生成一个 5x5 单色连通块。
      - 明确拒绝任何还能被“旧版边缘匹配链”唯一排序的样本，确保 D17 真正在起作用。
      - 先渲染目标输出条带，再打乱块顺序构造内层输入布局。
      - 最后在外侧追加控制边，把输出顺序编码为颜色序列。
      - 用新 `main()` 回归验证。

  post_change_main_assumptions_and_generator_support:
    assumption_1:
      - main 假设恰有一个可剥离的控制边候选。
      - generator 始终只添加一条控制边，并保证其他三条外边直接属于块区，不会形成第二个候选。
    assumption_2:
      - main 假设剥离控制边后能切出 3-4 个 5x5 块。
      - generator 沿用原题块布局模板，所以这个条件恒成立。
    assumption_3:
      - main 假设每块恰有一种非背景非分隔线颜色。
      - generator 的块始终单色。
    assumption_4:
      - main 假设控制颜色与块颜色一一对应。
      - generator 用输出顺序颜色直接绘制控制像素，因此集合与顺序都受控。
"""


BG = 7
SEP = 6
PALETTE = [1, 2, 3, 4, 5, 8, 9]

BLOCK_SIZE = 5
FG_AREA_MIN = 4
FG_AREA_MAX = 10
GRID_LAYOUT_PROB = 0.5
MAX_RETRIES = 400
MASK_RETRIES = 60

_OTHER = {"h": "v", "v": "h"}
_EDGE_PAIRS = {"h": ("right", "left"), "v": ("bot", "top")}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reorder 5x5 blocks using border control pixels instead of edge matching.

    Input structure:
    - bg = background, sep = divider color.
    - A normal 3- or 4-block layout from the original task is present inside.
    - One extra outer border row/col contains 3 or 4 colored control pixels.

    Rule:
    - Remove the control border.
    - Split the remaining region into 5x5 blocks using sep lines.
    - Match each control color to the uniquely colored block of that color.
    - Read control pixels left->right or top->bottom to get output order.
    - Horizontal control border => horizontal output strip.
      Vertical control border => vertical output strip.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def _find_sep(a: np.ndarray) -> int:
        cand: list[int] = []
        for r in range(int(a.shape[0])):
            if bool((a[r, :] == int(a[r, 0])).all()) and int(a[r, 0]) != int(bg):
                cand.append(int(a[r, 0]))
        for c in range(int(a.shape[1])):
            if bool((a[:, c] == int(a[0, c])).all()) and int(a[0, c]) != int(bg):
                cand.append(int(a[0, c]))
        assert cand, "separator not found"
        return int(Counter(cand).most_common(1)[0][0])

    sep = _find_sep(g)

    def _full_sep_rows(a: np.ndarray) -> list[int]:
        return [int(r) for r in range(int(a.shape[0])) if bool((a[r, :] == int(sep)).all())]

    def _full_sep_cols(a: np.ndarray) -> list[int]:
        return [int(c) for c in range(int(a.shape[1])) if bool((a[:, c] == int(sep)).all())]

    def _split_by_rows(a: np.ndarray) -> list[np.ndarray]:
        rows = _full_sep_rows(a)
        segs: list[np.ndarray] = []
        start = 0
        for r in rows + [int(a.shape[0])]:
            if int(r) > int(start):
                segs.append(a[start:r, :].copy())
            start = int(r + 1)
        return segs

    def _split_by_cols(a: np.ndarray) -> list[np.ndarray]:
        cols = _full_sep_cols(a)
        segs: list[np.ndarray] = []
        start = 0
        for c in cols + [int(a.shape[1])]:
            if int(c) > int(start):
                segs.append(a[:, start:c].copy())
            start = int(c + 1)
        return segs

    def _split_blocks(a: np.ndarray) -> list[np.ndarray] | None:
        rows = _full_sep_rows(a)
        cols = _full_sep_cols(a)
        blocks: list[np.ndarray]
        if rows and cols:
            if len(rows) != 1 or len(cols) != 1:
                return None
            r = int(rows[0])
            c = int(cols[0])
            blocks = [
                a[:r, :c].copy(),
                a[:r, c + 1 :].copy(),
                a[r + 1 :, :c].copy(),
                a[r + 1 :, c + 1 :].copy(),
            ]
        elif rows:
            blocks = _split_by_rows(a)
        elif cols:
            blocks = _split_by_cols(a)
        else:
            return None
        if len(blocks) not in (3, 4):
            return None
        if any(block.shape != (BLOCK_SIZE, BLOCK_SIZE) for block in blocks):
            return None
        return blocks

    def _block_color(block: np.ndarray) -> int | None:
        vals = {int(v) for v in np.unique(block)}
        vals.discard(int(bg))
        vals.discard(int(sep))
        if len(vals) != 1:
            return None
        return int(next(iter(vals)))

    def _read_controls(line: np.ndarray) -> list[int]:
        return [int(v) for v in line.tolist() if int(v) not in (int(bg), int(sep))]

    candidates: list[tuple[str, list[int], list[np.ndarray]]] = []
    border_specs = [
        ("top", g[0, :], g[1:, :]),
        ("bottom", g[-1, :], g[:-1, :]),
        ("left", g[:, 0], g[:, 1:]),
        ("right", g[:, -1], g[:, :-1]),
    ]
    for side, border, inner in border_specs:
        ctrl_colors = _read_controls(border)
        if len(ctrl_colors) not in (3, 4):
            continue
        if len(set(ctrl_colors)) != len(ctrl_colors):
            continue
        blocks = _split_blocks(inner)
        if blocks is None or len(blocks) != len(ctrl_colors):
            continue
        block_colors = [_block_color(block) for block in blocks]
        if any(color is None for color in block_colors):
            continue
        if set(map(int, block_colors)) != set(ctrl_colors):
            continue
        candidates.append((side, ctrl_colors, blocks))

    assert len(candidates) == 1, "expected exactly one control border"
    side, ctrl_colors, blocks = candidates[0]

    color_to_block = {int(_block_color(block)): block for block in blocks}
    ordered = [color_to_block[int(color)] for color in ctrl_colors]
    axis = 1 if side in {"top", "bottom"} else 0
    return _render_strip_axis(ordered, axis=axis)


def _render_strip_axis(blocks: list[np.ndarray], axis: int) -> np.ndarray:
    n = len(blocks)
    if axis == 0:
        out = np.full((n * BLOCK_SIZE + (n - 1), BLOCK_SIZE), BG, dtype=int)
        rr = 0
        for i, block in enumerate(blocks):
            paste(out, block, rr, 0)
            rr += BLOCK_SIZE
            if i < n - 1:
                out[rr, :] = SEP
                rr += 1
        return out
    out = np.full((BLOCK_SIZE, n * BLOCK_SIZE + (n - 1)), BG, dtype=int)
    cc = 0
    for i, block in enumerate(blocks):
        paste(out, block, 0, cc)
        cc += BLOCK_SIZE
        if i < n - 1:
            out[:, cc] = SEP
            cc += 1
    return out


def _edge_sets(block: np.ndarray) -> dict[str, set[int]]:
    mask = block != BG
    return {
        "top": set(np.where(mask[0, :])[0].tolist()),
        "bot": set(np.where(mask[-1, :])[0].tolist()),
        "left": set(np.where(mask[:, 0])[0].tolist()),
        "right": set(np.where(mask[:, -1])[0].tolist()),
    }


def _count_orders(blocks: list[np.ndarray], orient: str) -> int:
    contacts = [_edge_sets(block) for block in blocks]
    lead, trail = _EDGE_PAIRS[orient]

    def ok(a: int, b: int) -> bool:
        return bool(contacts[a][lead] & contacts[b][trail])

    total = 0
    for perm in itertools.permutations(range(len(blocks))):
        if all(ok(perm[i], perm[i + 1]) for i in range(len(perm) - 1)):
            total += 1
    return total


def _random_mask() -> np.ndarray | None:
    for _ in range(MASK_RETRIES):
        target_area = random.randint(FG_AREA_MIN, FG_AREA_MAX)
        cells = {(random.randrange(BLOCK_SIZE), random.randrange(BLOCK_SIZE))}
        while len(cells) < target_area:
            frontier: set[tuple[int, int]] = set()
            for r, c in cells:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    rr = int(r + dr)
                    cc = int(c + dc)
                    if 0 <= rr < BLOCK_SIZE and 0 <= cc < BLOCK_SIZE and (rr, cc) not in cells:
                        frontier.add((rr, cc))
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        if len(cells) != target_area:
            continue
        mask = np.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if any(mask[r, :].all() for r in range(BLOCK_SIZE)):
            continue
        if any(mask[:, c].all() for c in range(BLOCK_SIZE)):
            continue
        if not is_contiguous(mask.astype(int), background=0, connectivity=4):
            continue
        return mask
    return None


def _make_block(color: int) -> np.ndarray | None:
    mask = _random_mask()
    if mask is None:
        return None
    block = np.full((BLOCK_SIZE, BLOCK_SIZE), BG, dtype=int)
    block[mask] = int(color)
    return block


def _render_strip(blocks: list[np.ndarray], orient: str) -> np.ndarray:
    return _render_strip_axis(blocks, axis=1 if orient == "h" else 0)


def _render_inner_input(blocks: list[np.ndarray], output_orient: str) -> np.ndarray:
    n = len(blocks)
    perm = list(range(n))
    while perm == list(range(n)):
        random.shuffle(perm)
    scrambled = [blocks[i] for i in perm]

    if n == 4 and random.random() < GRID_LAYOUT_PROB:
        grid = np.full((2 * BLOCK_SIZE + 1, 2 * BLOCK_SIZE + 1), BG, dtype=int)
        grid[BLOCK_SIZE, :] = SEP
        grid[:, BLOCK_SIZE] = SEP
        starts = [(0, 0), (0, BLOCK_SIZE + 1), (BLOCK_SIZE + 1, 0), (BLOCK_SIZE + 1, BLOCK_SIZE + 1)]
        for block, (r0, c0) in zip(scrambled, starts):
            paste(grid, block, r0, c0)
        return grid

    return _render_strip(scrambled, _OTHER[output_orient])


def _add_controls(inner: np.ndarray, control_colors: list[int], output_orient: str) -> np.ndarray:
    n = len(control_colors)
    if output_orient == "h":
        side = random.choice(["top", "bottom"])
        H, W = inner.shape
        out = np.full((H + 1, W), BG, dtype=int)
        if side == "top":
            out[1:, :] = inner
            row = 0
        else:
            out[:-1, :] = inner
            row = H
        start = (W - n) // 2
        for i, color in enumerate(control_colors):
            out[row, start + i] = int(color)
        return out

    side = random.choice(["left", "right"])
    H, W = inner.shape
    out = np.full((H, W + 1), BG, dtype=int)
    if side == "left":
        out[:, 1:] = inner
        col = 0
    else:
        out[:, :-1] = inner
        col = W
    start = (H - n) // 2
    for i, color in enumerate(control_colors):
        out[start + i, col] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([3, 4])
    orient = random.choice(["h", "v"])
    colors = random.sample(PALETTE, n)

    blocks: list[np.ndarray] = []
    for color in colors:
        block = _make_block(color)
        if block is None:
            return None
        blocks.append(block)

    if _count_orders(blocks, "h") != 0 or _count_orders(blocks, "v") != 0:
        return None

    out = _render_strip(blocks, orient)
    inner = _render_inner_input(blocks, orient)
    inp = _add_controls(inner, colors, orient)
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
