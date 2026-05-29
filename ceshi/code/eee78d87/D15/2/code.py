# Variant of eee78d87
# Dimension: D15 — 格栅的空间方向 (meta)
# Change: 格栅为正交方向（水平/垂直线交织），铺贴沿行列方向周期重复... → 格栅交替使用两种不同的瓦片

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg


# Design analysis
# ---------------
# a) 具体化与行为追踪
#    - 输入不变：6x6 橙色背景里放一个 3x3 轮廓字形，字形仍然只负责“编码”相位瓦片。
#    - 原题输出：先解码出 3x3 瓦片 A，然后整张 16x16 都按 A 做普通周期平铺。
#    - 本变体输出：先解码出瓦片 A，再构造 B = rot90(A)。输出按“3 行为一条带”平铺：
#      第 1 条带用 A，第 2 条带用 B，第 3 条带再用 A，依次交替；列方向仍按各自条带中的瓦片周期重复。
#    - 追踪示例：plus 字形会解码出右下 2x2 橙块。原题所有条带都重复这块；变体中偶数条带改成旋转后的左下 2x2，
#      所以横向仍有周期 3，纵向则出现 A/B/A/B 的交替方向。
# b) 复杂度校准
#    - 只引入两种瓦片 A/B，分支少且重复充分，不会因为自由度过高而难归纳。
#    - B 选 90 度旋转而不是转置，因为四种合法字形解码后的瓦片在旋转下都可见地改变；若用转置，部分字形会退化。
# c) 隐式假设审查
#    - 原 `out = tile_fill(...)` 默认“所有位置共享同一瓦片”，这是 D15 变更中失效的关键假设，必须替换。
#    - 原中心印章覆盖逻辑只依赖输出尺寸和中心 6x6 区域，不依赖格栅方向，因此保持不变。
#    - `_decode_phase_tile` 仍依赖输入字形属于四个原始轮廓类；D15 不修改输入编码规则，所以该前提仍成立。
# d) 放置合法性
#    - 生成器仍只放一个 3x3 字形，位置限制在左上角坐标 0..3，保证不越界且 bbox 恰为 3x3。
#    - 输出格栅由 solver 直接构造，不存在对象碰撞问题；中心 6x6 印章仍位于 (5:11, 5:11)。
# e) 视觉可观察性
#    - 交替规则完全依赖可见的空间条带，不依赖颜色编号等隐藏属性。
#    - 四种字形对应的 A 与 B 都不同，因此任意生成样例都能在输出上显示出双模式条带，而非偶然退化回原题。
# f) 可视化整洁度
#    - 仍然只有“黑/橙格栅 + 栗色中心印章”两层结构，条带变化只发生在背景格栅，不会造成局部过度叠加。
# g) 颜色变化可推断性
#    - 颜色系统不变：输入仍是一种前景色；输出仍只用黑、橙、栗色。变体只改变空间排布，不增加颜色映射负担。
# 9) main() 的新隐式假设
#    - solver 现在假设解码后的 A 与其 90 度旋转 B 对四个合法字形都具有可见差异；生成器通过固定原四种字形集合来保证。
#    - solver 按 3 行一条带交替，因此默认瓦片尺寸为 3x3；生成器与 `_decode_phase_tile` 共同保证该尺寸恒定。


def _decode_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the 3x3 contour glyph into the base period-3 orange phase mask."""
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


def _fill_alternating_row_bands(shape: tuple[int, int], tile_a: np.ndarray, tile_b: np.ndarray) -> np.ndarray:
    """Repeat tile A/B across the canvas, alternating which tile each 3-row band uses."""
    h, w = map(int, shape)
    th, tw = tile_a.shape
    assert tile_a.shape == tile_b.shape == (3, 3)

    out = np.full((h, w), Color.BLACK, dtype=int)
    for band_r0 in range(0, h, th):
        tile = tile_a if (band_r0 // th) % 2 == 0 else tile_b
        for c0 in range(0, w, tw):
            hh = min(th, h - band_r0)
            ww = min(tw, w - c0)
            out[band_r0 : band_r0 + hh, c0 : c0 + ww] = tile[:hh, :ww]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a 16x16 alternating lattice, then stamp the glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange color.
    - The non-orange cells form a single 3x3-bounded contour glyph.
    - The glyph is one of the four original code shapes: plus, H, X, or four corners.

    Rule:
    - Decode the glyph into a base 3x3 orange/black phase tile A.
    - Build a companion tile B by rotating A by 90 degrees.
    - Fill the 16x16 black canvas by repeating A on the 1st, 3rd, 5th... tile rows and B on the 2nd, 4th, 6th... tile rows.
    - Tile the original 3x3 glyph 2x2 and recolor its true cells to maroon in the centered 6x6 region.

    Key insight:
    - The input glyph still acts as a small code, but it now controls a pair of related lattice phases whose orientation alternates by horizontal band.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (6, 6)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (3, 3)

    phase_a = _decode_phase_tile(motif)
    phase_b = np.rot90(phase_a, k=-1)
    tile_a = np.where(phase_a, bg, Color.BLACK).astype(int)
    tile_b = np.where(phase_b, bg, Color.BLACK).astype(int)

    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = _fill_alternating_row_bands(out_shape, tile_a, tile_b)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
MOTIF_POS_MAX = 4
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]
MAX_RETRIES = 300

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif_name = str(rng.choice(list(GLYPHS.keys())))
    motif = GLYPHS[motif_name]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
        phase_a = _decode_phase_tile(motif)
        phase_b = np.rot90(phase_a, k=-1)
        assert not np.array_equal(phase_a, phase_b)
        assert out.shape == (16, 16)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 D15 variant")
