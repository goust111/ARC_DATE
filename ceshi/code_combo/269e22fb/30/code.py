# Combo variant of 269e22fb (index 30)
# combo_spec: D2/1+D3/1+D17/2
# Name: 条形边界标记补集输出

from __future__ import annotations

import numpy as np

from arc_utils import Color, colors_present


# Design analysis
# a) 具体化与行为追踪:
#    - 原题输入是一个普通矩形内部裁剪，solver 直接把整块两色窗口拿去和 20x20 母图的
#      8 个二面体变体做精确匹配，最后输出完整 20x20 图案。
#    - 本融合版把输入改成“细长条 + 越界边带”。例如 5x18 输入若顶行全是第三色，
#      那么真正可见的母图片段其实是其下方 4x18 条带，并且这个条带一定贴着母图上边界。
#      相比原题，solver 不再做二维偏移搜索，而是先读出哪一侧出现越界边带，再只沿自由轴搜索。
#    - 输出也不再是完整图案，而是把刚才识别出的那条真实 4x18/18x4 可见带从完整图案中扣掉，
#      得到一个 U 形剩余区域。这里“扣掉”的是窗口与母图的相交部分，不包含越界第三色部分。
# b) 复杂度校准:
#    - 如果允许 3xN 条带再被边界截断成 2xN，可辨识度会明显下降。
#    - 因此生成器只使用 5x15..18 或 15..18x5 输入窗口，并且越界厚度固定为 1。
#      这样去掉第三色边带后，真实可见部分恰好总是 4x15..18 或 15..18x4，
#      保留了 D2 中“极端长条但仍可唯一匹配”的信息量，同时 D17 的边界线索又把搜索
#      从二维降成单轴。
#    - 只允许单边越界，不允许角越界。这样输出稳定呈现 U 形而不是更杂的缺角框形，
#      2-4 个样例就足以让人看出“第三色提示边界 + 输出未知补集”。
# c) 原 main() 的隐式假设审查:
#    - `len(colors) == 2` 不再成立；融合后输入必须有 3 色，其中第三色只出现在一整条外侧边带。
#    - 原代码默认裁剪完全落在 20x20 内部，所以会遍历所有内部 `(r, c)`。
#      这里该前提被 D17 打破；但第三色边带恰好告诉我们触碰的是哪一条真实边，所以
#      条带在母图中的一维位置被固定到 `r=0/16` 或 `c=0/16`。
#    - 原代码返回完整图案；这里必须保留匹配到的条带位置，因为 D3 需要据此抠除那块已知区域。
# d) 放置合法性:
#    - 生成器总是先选完整输出，再选一条贴边的 4x15..18 或 15..18x4 真实可见带，
#      最后在其外侧补一行或一列第三色形成输入窗口。
#    - 所有真实可见部分都完全位于 20x20 内，第三色只占窗口外扩出的 1 格边带，
#      不会与别的对象重叠，因为任务中根本没有第二个对象。
# e) 视觉可观察性:
#    - 人眼能直接看到输入是细长条，不是普通矩形。
#    - 第三色是一条完整边带，明确提示“这边已经越过真实边界”。
#    - 输出缺失的正是输入中已知的那条贴边窄带，因此不是补全，而是反向地输出剩余部分。
# f) 可视化整洁度:
#    - 输入只有三块语义层: 第三色边带、前景色、背景色；并且第三色只在单条外缘上出现，
#      没有多层遮挡。
#    - 输出仍然只有前景/背景两色，结构是干净的 U 形剩余区域。
# g) 变化的可推断性:
#    - 颜色规则完全依赖结构，不依赖颜色编号。第三色靠“整条外缘边带”被识别；
#      前景/背景仍通过母图匹配恢复。
# h) main() 新增隐式假设与生成器保证:
#    - 假设 1: 输入尺寸必须是 5x15..18 或 15..18x5 之一。生成器固定只产出这些尺寸。
#    - 假设 2: 第三色恰好占据一整条最外侧边带且只出现一次。生成器固定越界厚度为 1。
#    - 假设 3: 去掉第三色后，真实条带必须仍包含母图两种状态。生成器显式过滤掉单色条带。
#    - 假设 4: 在“边界已知 + 极端条带”条件下匹配唯一。生成器在返回前用 `_find_unique_match`
#      和 `main(input) == expected` 双重验证。


_BASE_PATTERN = (
    "11000000000000011000",
    "11111100000000011000",
    "11111110000000011100",
    "11000111000000011100",
    "00000011100000111110",
    "00000001110000100110",
    "00000000110011100110",
    "00000000010010100110",
    "00000000011110100110",
    "01111111110110100110",
    "01000000011110100110",
    "01111111110010100110",
    "01010000010011100110",
    "11101111110000100110",
    "01011000010000111110",
    "11101000010001101100",
    "01011000010001000110",
    "11101000010001100011",
    "01011000101000101110",
    "11100001000100111000",
)

_HORIZONTAL_VISIBLE_WIDTHS = tuple(range(15, 19))
_VERTICAL_VISIBLE_HEIGHTS = tuple(range(15, 19))
_VISIBLE_STRIP_SHAPES = tuple((4, w) for w in _HORIZONTAL_VISIBLE_WIDTHS) + tuple(
    (h, 4) for h in _VERTICAL_VISIBLE_HEIGHTS
)
_INPUT_WINDOW_SHAPES = tuple((5, w) for w in _HORIZONTAL_VISIBLE_WIDTHS) + tuple(
    (h, 5) for h in _VERTICAL_VISIBLE_HEIGHTS
)


def _pattern_mask() -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in _BASE_PATTERN], dtype=bool)


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    for k in range(4):
        rotated = np.rot90(mask, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    return variants


_ALL_PATTERNS = _dihedral_variants(_pattern_mask())


def _render_window(
    pattern: np.ndarray,
    H: int,
    W: int,
    window_r0: int,
    window_c0: int,
    fg: int,
    bg: int,
    border: int,
) -> np.ndarray:
    out = np.full((H, W), int(border), dtype=int)
    src_r0 = max(0, int(window_r0))
    src_r1 = min(20, int(window_r0 + H))
    src_c0 = max(0, int(window_c0))
    src_c1 = min(20, int(window_c0 + W))
    if src_r0 < src_r1 and src_c0 < src_c1:
        patch = np.where(pattern[src_r0:src_r1, src_c0:src_c1], int(fg), int(bg))
        out[src_r0 - window_r0 : src_r1 - window_r0, src_c0 - window_c0 : src_c1 - window_c0] = patch
    return out


def _render_complement(
    pattern: np.ndarray, fg: int, bg: int, strip_r0: int, strip_c0: int, H: int, W: int
) -> np.ndarray:
    keep = pattern.copy()
    keep[int(strip_r0) : int(strip_r0 + H), int(strip_c0) : int(strip_c0 + W)] = False
    return np.where(keep, int(fg), int(bg)).astype(int)


def _extract_edge_strip(grid: np.ndarray, border: int) -> tuple[np.ndarray, str] | None:
    g = np.asarray(grid, dtype=int)
    H, W = map(int, g.shape)
    border_mask = g == int(border)
    count = int(border_mask.sum())

    if H == 5 and W in _HORIZONTAL_VISIBLE_WIDTHS:
        if count == W and np.all(border_mask[0, :]):
            return g[1:, :], "top"
        if count == W and np.all(border_mask[-1, :]):
            return g[:-1, :], "bottom"

    if W == 5 and H in _VERTICAL_VISIBLE_HEIGHTS:
        if count == H and np.all(border_mask[:, 0]):
            return g[:, 1:], "left"
        if count == H and np.all(border_mask[:, -1]):
            return g[:, :-1], "right"

    return None


def _search_starts(side: str, strip_h: int, strip_w: int) -> tuple[list[int], list[int]]:
    if side == "top":
        return [0], list(range(0, 20 - strip_w + 1))
    if side == "bottom":
        return [20 - strip_h], list(range(0, 20 - strip_w + 1))
    if side == "left":
        return list(range(0, 20 - strip_h + 1)), [0]
    if side == "right":
        return list(range(0, 20 - strip_h + 1)), [20 - strip_w]
    raise AssertionError("unknown side")


def _find_unique_match(grid: np.ndarray) -> tuple[int, int, np.ndarray, int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    assert g.shape in _INPUT_WINDOW_SHAPES

    colors = colors_present(g)
    assert len(colors) == 3

    matches: list[tuple[int, int, np.ndarray, int, int, int, int]] = []
    for border in colors:
        extracted = _extract_edge_strip(g, int(border))
        if extracted is None:
            continue
        strip, side = extracted
        strip_colors = colors_present(strip)
        if len(strip_colors) != 2:
            continue
        strip_h, strip_w = map(int, strip.shape)
        assert (strip_h, strip_w) in _VISIBLE_STRIP_SHAPES
        row_starts, col_starts = _search_starts(side, strip_h, strip_w)
        for fg in strip_colors:
            bg = int(strip_colors[0] if int(strip_colors[1]) == int(fg) else strip_colors[1])
            visible = strip == int(fg)
            for pattern in _ALL_PATTERNS:
                for r in row_starts:
                    for c in col_starts:
                        if np.array_equal(pattern[r : r + strip_h, c : c + strip_w], visible):
                            matches.append((int(fg), int(bg), pattern, int(r), int(c), strip_h, strip_w))

    assert len(matches) == 1
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the motif orientation from a border-aware strip, then output only the unseen remainder.

    Input structure:
    - The grid itself is a long narrow window: either 5x15..18 or 15..18x5.
    - Exactly one full outer row or column is filled with a third color, marking the part of the window that lies outside the true 20x20 motif.
    - The remaining 4x15..18 or 15..18x4 two-color band is the actual visible slice of one dihedral orientation of the shared motif.

    Rule:
    - Try each input color as the possible out-of-bounds marker and identify whether it forms a complete top/bottom/left/right border band.
    - Remove that band, treat the remaining two colors as the possible motif foreground/background, and match the recovered strip against every dihedral motif variant.
    - The border band fixes which real edge of the motif was crossed, so only a one-axis offset search remains.
    - After finding the unique strip placement, repaint the full motif with the recovered colors and erase the matched visible strip, leaving only its complement.

    Key insight:
    - The third color does not describe motif content; it tells you which side of the motif the thin strip is attached to. That edge cue and the long-strip geometry jointly determine the strip's unique placement, and the output is the rest of the motif rather than the observed strip itself.
    """
    fg, bg, pattern, strip_r0, strip_c0, strip_h, strip_w = _find_unique_match(np.asarray(grid, dtype=int))
    return _render_complement(pattern, fg, bg, strip_r0, strip_c0, strip_h, strip_w)


def _generate_case(
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | str | tuple[int, int]]]:
    fg, bg, border = map(int, rng.choice(Color.ALL_COLORS, size=3, replace=False))
    pattern = _ALL_PATTERNS[int(rng.integers(len(_ALL_PATTERNS)))]

    horizontal = bool(int(rng.integers(2)) == 0)
    if horizontal:
        strip_h = 4
        strip_w = int(rng.choice(_HORIZONTAL_VISIBLE_WIDTHS))
        side = "top" if int(rng.integers(2)) == 0 else "bottom"
        strip_r0 = 0 if side == "top" else 20 - strip_h
        strip_c0 = int(rng.integers(1, 20 - strip_w))
        window_h, window_w = strip_h + 1, strip_w
        window_r0 = -1 if side == "top" else strip_r0
        window_c0 = strip_c0
    else:
        strip_h = int(rng.choice(_VERTICAL_VISIBLE_HEIGHTS))
        strip_w = 4
        side = "left" if int(rng.integers(2)) == 0 else "right"
        strip_r0 = int(rng.integers(1, 20 - strip_h))
        strip_c0 = 0 if side == "left" else 20 - strip_w
        window_h, window_w = strip_h, strip_w + 1
        window_r0 = strip_r0
        window_c0 = -1 if side == "left" else strip_c0

    grid = _render_window(pattern, window_h, window_w, window_r0, window_c0, fg, bg, border)
    if side == "top":
        visible_strip = grid[1:, :]
    elif side == "bottom":
        visible_strip = grid[:-1, :]
    elif side == "left":
        visible_strip = grid[:, 1:]
    else:
        visible_strip = grid[:, :-1]
    expected = _render_complement(pattern, fg, bg, strip_r0, strip_c0, strip_h, strip_w)

    meta = {
        "side": side,
        "input_shape": (window_h, window_w),
        "visible_shape": (strip_h, strip_w),
        "strip_offset": (strip_r0, strip_c0),
        "fg": fg,
        "bg": bg,
        "border": border,
    }

    assert colors_present(grid) == sorted((fg, bg, border))
    assert len(colors_present(visible_strip)) == 2
    assert int(strip_r0 == 0) + int(strip_r0 + strip_h == 20) + int(strip_c0 == 0) + int(strip_c0 + strip_w == 20) == 1
    assert not np.array_equal(expected, np.where(pattern, fg, bg).astype(int))
    return grid, expected, meta


def generate_input() -> np.ndarray:
    """
    Output-first generator for the combo variant.

    The generator enforces every solver-side assumption introduced above:
    - input size is always one of the sanctioned border-aware strip windows;
    - the third color is always a single complete outer band of thickness 1;
    - removing that band yields a 4x15..18 or 15..18x4 two-color strip with both motif states;
    - the resulting sample has exactly one valid match and produces the intended complement output.
    """
    rng = np.random.default_rng()

    for _ in range(2000):
        try:
            grid, expected, _meta = _generate_case(rng)
            _find_unique_match(grid)
            if not np.array_equal(main(grid), expected):
                continue
            return grid
        except AssertionError:
            continue

    raise AssertionError("failed to generate a unique border-aware complement strip")
