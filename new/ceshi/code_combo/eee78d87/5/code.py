# Combo variant of eee78d87 (index 5)
# combo_spec: D4/2+D4/3
# Name: 镜像+旋转混合对称印章

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill

# 融合设计分析: D4/2 + D4/3
# a) 具体化与行为追踪
#    - 原题: 输入里的 3x3 轮廓字形只负责解码 3x3 周期相位瓦片; 输出中心 6x6 只是把同一个 3x3 字形
#      直接做 2x2 平铺并统一重着色为栗色。
#    - 本融合: 周期格栅完全不变, 只改中心 6x6 的四个 3x3 象限。采用
#      左上=原始, 右上=水平翻转, 左下=顺时针90度, 右下=逆时针90度。
#    - 具体例子用最有信息量的 `H` 字形:
#        原始 H = 101/111/101
#        旋转后 = 111/010/111
#      原题中心印章四象限都相同; 本题上排仍是 H, 下排变成旋转版, 人眼能直接看出“复制规则变了”。
#
# b) 复杂度校准
#    - 不能把四象限变换做得更自由, 否则 2-4 个样例不足以让人归纳。
#    - 也不能继续均匀生成原题全部四种字形而不筛选, 因为 plus/X/4corners 在 D4 群下都与自身完全重合,
#      会把融合退化回原题或单一子变体。于是 generator 用定量指标拒绝退化样本。
#
# c) 原 main() 的隐式假设审查
#    - `bg == ORANGE`, `6x6`, `tight 3x3 bbox`, `single non-orange color`: 都仍然成立, 不需要改。
#    - `_decode_phase_tile(motif)`: 依赖 motif 仍是原题四种轮廓编码之一, 融合并未改输入编码, 仍成立。
#    - `center = np.tile(motif, (2, 2))`: 这是被打破的关键假设。融合后中心不再是简单复制, 必须替换为
#      按象限分别施加不同 D4 变换的 6x6 拼接。
#
# d) 放置合法性
#    - 输入端仍然只放一个 3x3 字形, top-left 在 0..3, 不会越界也不存在重叠问题。
#    - 输出端中心印章仍固定放在 16x16 的中心 6x6 窗口, 四个 3x3 象限正好无缝拼接。
#
# e) 视觉可观察性
#    - 核心变化集中在中心 6x6, 周围格栅保持原题结构, 对比清楚。
#    - 变换规则只使用“镜像/旋转”这种人眼直接可见的几何特征, 不依赖颜色数值或隐藏排序。
#    - generator 强制混合印章必须同时区别于原题平铺、纯镜像拼接、纯旋转拼接, 防止训练样本退化。
#
# f) 可视化整洁度
#    - 只有一层中心栗色印章覆盖在橙黑格栅上, 与原题同样整洁; 没有引入额外对象或多层标记冲突。
#
# g) 变化的可推断性
#    - 颜色规则不变: 中心字形仍统一重着色为栗色, 所以唯一新增信息就是几何拼接方式。
#      这让解题者可以把注意力集中在象限变换本身。
#
# h) 融合行为量化指标
#    - `mixed_plain_delta`: 混合印章与原题简单 2x2 平铺的像素差异数。
#    - `mixed_mirror_delta`: 混合印章与纯 D4/2 镜像拼接的差异数。
#    - `mixed_rotation_delta`: 混合印章与纯 D4/3 旋转拼接的差异数。
#    - 只有三者都 > 0 时, 才说明样本确实体现“镜像+旋转混合”而非退化。
#
# i) 新增隐式假设与生成保障
#    - 新 solver 假设输入字形经过所选混合布局后, 至少能产生一个不同于三种 baseline 的中心印章。
#    - generate_input 通过 `_fusion_metrics(...)` 显式验证这一点; 若不满足就丢弃样本重采样。


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


def _rot_cw(motif: np.ndarray) -> np.ndarray:
    return np.rot90(motif, -1)


def _rot_ccw(motif: np.ndarray) -> np.ndarray:
    return np.rot90(motif, 1)


def _compose_stamp(tl: np.ndarray, tr: np.ndarray, bl: np.ndarray, br: np.ndarray) -> np.ndarray:
    assert tl.shape == tr.shape == bl.shape == br.shape == (MOTIF_SIZE, MOTIF_SIZE)
    return np.block([[tl, tr], [bl, br]]).astype(bool)


def _plain_stamp(motif: np.ndarray) -> np.ndarray:
    return np.tile(motif, (2, 2)).astype(bool)


def _mirror_stamp(motif: np.ndarray) -> np.ndarray:
    return _compose_stamp(motif, np.fliplr(motif), np.flipud(motif), np.rot90(motif, 2))


def _rotation_stamp(motif: np.ndarray) -> np.ndarray:
    return _compose_stamp(motif, _rot_cw(motif), np.rot90(motif, 2), _rot_ccw(motif))


def _mixed_stamp(motif: np.ndarray) -> np.ndarray:
    return _compose_stamp(motif, np.fliplr(motif), _rot_cw(motif), _rot_ccw(motif))


def _fusion_metrics(motif: np.ndarray) -> dict[str, int]:
    mixed = _mixed_stamp(motif)
    quadrants = [mixed[:3, :3], mixed[:3, 3:], mixed[3:, :3], mixed[3:, 3:]]
    unique_quadrants = len({q.astype(np.uint8).tobytes() for q in quadrants})
    return {
        "mixed_plain_delta": int(np.count_nonzero(mixed != _plain_stamp(motif))),
        "mixed_mirror_delta": int(np.count_nonzero(mixed != _mirror_stamp(motif))),
        "mixed_rotation_delta": int(np.count_nonzero(mixed != _rotation_stamp(motif))),
        "quadrant_variant_count": int(unique_quadrants),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour cue into the original 16x16 periodic lattice, then stamp a mixed-symmetry 2x2 arrangement of that glyph in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one non-orange foreground color.
    - The non-orange pixels form a tight 3x3 contour glyph.
    - The glyph is one of the four task codes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the same period-3 orange/black phase tile as the original task.
    - Repeat that tile to fill the 16x16 output canvas.
    - In the centered 6x6 window, place four 3x3 maroon copies of the glyph using a mixed D4 layout:
      top-left original, top-right horizontal flip, bottom-left clockwise rotation, bottom-right counterclockwise rotation.

    Key insight:
    - The input glyph still serves two roles: its contour decodes the background lattice, and its geometry is reused at the center; only the center now reveals a quadrant-wise transform rule instead of plain repetition.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_H, GRID_W)

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (MOTIF_SIZE, MOTIF_SIZE)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    center = _mixed_stamp(motif)
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
    metrics = _fusion_metrics(motif)
    if min(metrics["mixed_plain_delta"], metrics["mixed_mirror_delta"], metrics["mixed_rotation_delta"]) <= 0:
        return None

    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        out = main(inp)
    except Exception:
        return None

    center = out[5:11, 5:11]
    if int((center == Color.MAROON).sum()) <= 0:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
