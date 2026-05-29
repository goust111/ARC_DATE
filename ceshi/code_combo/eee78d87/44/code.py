# Combo variant of eee78d87 (index 44)
# combo_spec: D8/1+D14/0
# Name: 背景色由字形色决定+前景色决定印章色

"""
设计分析：eee78d87 的融合变体（D8/1 + D14/0）。

a) 具体化与行为追踪
  - 原题输入是 6x6 橙底板，上面放一个单色 3x3 轮廓字形；输出是黑橙周期 3 晶格，
    中央 6x6 用栗色重绘该字形的 2x2 平铺。
  - 融合后，字形形状仍然只负责解码周期相位，但颜色获得双重语义：
    1) 输入背景色不再固定为橙色，而是由字形前景色决定；
    2) 输出中心印章也不再固定为栗色，而是由同一个前景色决定，但用另一张映射表。
  - 具体例子：若输入字形色为绿色(3)，则生成器把背景改成蓝色(1)，中心印章改成黄色(4)。
    若该字形是 plus，则原题输出会是“橙黑晶格 + 栗色 plus 2x2”，
    融合后则变成“橙黑晶格 + 黄色 plus 2x2”，同时输入已不再是橙底而是蓝底。
  - 差异点有两个且来自同一控制变量 `fg`：输入侧看见 `fg -> bg_in`，输出侧看见 `fg -> stamp_out`。

b) 复杂度校准
  - 不改动原题最核心的几何结构：仍然只有一个 3x3 字形，仍然只允许 4 种轮廓，仍然输出 16x16。
  - 为了让颜色映射可由少量样例归纳，把前景色压缩到 3 种：蓝(1)、绿(3)、粉(6)。
  - 两张映射表都使用 3 个档位，且不是同一映射：
      输入背景映射：1->8, 3->1, 6->4
      输出印章映射：1->2, 3->4, 6->5
    这样 2-4 个样例就有机会覆盖全部档位，而且肉眼能看出“同一前景色同时控制两处颜色”。

c) 隐式假设审查（逐行对应原 solver 的关键前提）
  - `bg = get_bg(g)`：原前提“背景是多数色”仍成立；融合后只是背景颜色换了，但仍远多于字形像素。
  - `assert bg == Color.ORANGE`：该前提被 D8/1 直接打破，必须替换成“背景色等于前景色经过背景映射后的结果”。
  - `len(colors_present(g, exclude={bg})) == 1`：仍成立；输入依旧只有背景和一种字形色。
  - `motif = crop_bbox(fg_mask, b)` 与 `motif.shape == (3, 3)`：仍成立；融合不改字形 bbox 尺寸与形状集合。
  - `phase_colors = np.where(phase, bg, Color.BLACK)`：这个前提被融合描述明确打破。输入背景只影响输入端，
    输出晶格必须继续固定为橙黑，而不是沿用输入背景色。
  - `... = Color.MAROON`：该前提被 D14/0 打破，必须改成“前景色映射出的印章色”。
  - 生成器里的 `np.full(..., Color.ORANGE)` 同样失效，必须按前景色反查背景色。

d) 放置合法性
  - 输入仍是一个 6x6 网格，只放一个 3x3 字形，左上角位置仍在 `0..3 x 0..3` 内随机选取，因此不会越界。
  - 除单个字形外没有第二对象，不存在对象间重叠问题。
  - 输出中心印章仍是 6x6，放在 16x16 画布中心，与周围平铺背景共享像素位但颜色覆盖规则单一且确定。

e) 视觉可观察性
  - 人类能直接看到：同样都是 3x3 轮廓字形，但输入背景在变，而且中心印章颜色也在变。
  - 同时，外部晶格仍稳定保持橙黑两色，所以解题者容易把“形状解码规则”和“颜色映射规则”分离观察。
  - 背景色与印章色的变化都由输入前景色触发，且这两个输出效果分处输入/输出两侧，变化足够醒目。

f) 可视化整洁度
  - 输入只有两种颜色，输出只有三种颜色（黑、橙、印章色），没有额外标记层，画面仍然干净。
  - 晶格与中心印章分区明确：外围是周期纹理，中间是 6x6 图章，不会出现多套颜色编码在同一区域互相遮挡。

g) 变化的可推断性
  - 这里不依赖“颜色编号加一”这类人类不可见的数值操作，而是显式有限映射表。
  - 档位数限制为 3，训练集中只要覆盖 3 个前景色，就能完整暴露两张映射表。
  - 为避免退化，生成器永远不会产生橙色背景，也不会把印章固定成栗色；因此样例肉眼上一定区别于原题。

Post-edit assumption audit
  - `main()` 的新假设是：输入恰有两种颜色；唯一非背景色属于 `FG_TO_INPUT_BG` 的键集；
    背景色等于该前景色映射得到的颜色；非背景像素的紧致 bbox 恰为 3x3 且属于 4 种合法轮廓之一。
  - `generate_input()` 直接按这些约束构造，因此不会触发新的歧义或断言失败。
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


FG_TO_INPUT_BG = {
    Color.BLUE: Color.TEAL,
    Color.GREEN: Color.BLUE,
    Color.PINK: Color.YELLOW,
}

FG_TO_STAMP = {
    Color.BLUE: Color.RED,
    Color.GREEN: Color.YELLOW,
    Color.PINK: Color.GREY,
}

GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = sorted(FG_TO_INPUT_BG)
MAX_RETRIES = 100

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the period-3 orange phase mask."""
    assert motif.shape == (3, 3)

    corners = np.array([motif[0, 0], motif[0, 2], motif[2, 0], motif[2, 2]], dtype=bool)
    edge_centers = np.array([motif[0, 1], motif[1, 0], motif[1, 2], motif[2, 1]], dtype=bool)
    phase = np.zeros((3, 3), dtype=bool)

    if not bool(corners.any()):
        assert bool(motif[1, 1])
        assert bool(edge_centers.all())
        phase[1:, 1:] = True
        return phase

    assert bool(corners.all())
    assert not bool(motif[0, 1])
    assert not bool(motif[2, 1])
    assert bool(motif[1, 0]) == bool(motif[1, 2])

    phase[1:, 0] = True
    if bool(motif[1, 0]):
        assert bool(motif[1, 1])
    else:
        phase[0, 1:] = True

    if not bool(motif[1, 1]):
        phase[0, 0] = True

    return phase


def _extract_fg_and_motif(g: np.ndarray) -> tuple[int, np.ndarray]:
    """Recover the foreground color and the 3x3 boolean glyph mask."""
    bg = int(get_bg(g))
    other_colors = colors_present(g, exclude={bg})
    assert len(other_colors) == 1
    fg_color = int(other_colors[0])
    assert fg_color in FG_TO_INPUT_BG
    assert bg == FG_TO_INPUT_BG[fg_color]

    fg_mask = g == fg_color
    b = bbox(fg_mask)
    assert b is not None
    motif = crop_bbox(fg_mask, b).astype(bool)
    assert motif.shape == (3, 3)
    return fg_color, motif


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into the usual orange phase lattice, while using the glyph color to validate the input background and choose the center stamp color.

    Input structure:
    - The grid is 6x6 and contains exactly two colors: one background color and one glyph color.
    - The non-background pixels form a tight 3x3 contour glyph chosen from the same four forms as the base task.
    - The background color is not fixed; it is determined by the glyph color through a learned mapping.

    Rule:
    - Ignore the input background when drawing the outer lattice: decode the glyph shape into the same orange/black period-3 tile and repeat it across a 16x16 canvas.
    - Use the glyph color twice: first to confirm which input background is legal, then to select the recolor target for the central 2x2 tiled stamp.
    - Place the 6x6 tiled stamp at the center, recoloring its true cells with the mapped stamp color.

    Key insight:
    - Shape and color play different roles. The 3x3 contour still encodes the orange lattice phase, but the single foreground color now also carries a second semantic channel: it predicts both the input background and the output stamp color via two distinct mappings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    fg_color, motif = _extract_fg_and_motif(g)
    phase = _decode_phase_tile(motif)

    phase_colors = np.where(phase, Color.ORANGE, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    stamp_color = FG_TO_STAMP[fg_color]
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = stamp_color
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    glyph_name = str(rng.choice(list(GLYPHS)))
    motif = GLYPHS[glyph_name]
    fg_color = int(rng.choice(FG_COLORS))
    bg_color = FG_TO_INPUT_BG[fg_color]
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), bg_color, dtype=int)
    patch = inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE]
    patch[motif] = fg_color

    try:
        out = main(inp)
    except Exception:
        return None

    center = out[5:11, 5:11]
    assert FG_TO_STAMP[fg_color] in colors_present(center)
    assert Color.MAROON not in colors_present(out)
    assert Color.ORANGE not in colors_present(inp)
    return inp


def generate_input() -> np.ndarray:
    """Direct constructor for the fused color-semantic variant."""
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo D8/1+D14/0")
