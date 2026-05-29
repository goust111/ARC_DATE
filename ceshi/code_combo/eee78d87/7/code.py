# Combo variant of eee78d87 (index 7)
# combo_spec: D6/1+D15/0
# Name: 行列不同周期+对角线格栅

from __future__ import annotations

import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../../"))
SOLVER_DIR = os.path.join(ROOT_DIR, "solvers")
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)
if SOLVER_DIR not in sys.path:
    sys.path.append(SOLVER_DIR)

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg

# Design analysis for combo D6/1 + D15/0
# ---------------------------------------
# - concretization:
#   - 输入保持原题骨架: 6x6 橙底, 只有一个非橙色 3x3 轮廓字形, 位置任意但 bbox 必须正好是 3x3。
#   - 输出仍是 16x16 黑/橙背景加中央 6x6 栗色印章, 但背景不再是正交 3x3 平铺。
#   - 融合后用对角坐标采样: u=(r+c) mod 3, v=(r-c) mod 2。于是两组 45 度对角线组成菱形格栅,
#     其中一组每 3 相位重复, 另一组每 2 相位重复。
#   - 具体追踪一个 H 字形: 原题先解出 3x3 相位 000/100/100, 再按行列正交平铺;
#     D15/0 单变体会把同一个 3x3 相位改为对角坐标下的 3x3 周期; 本融合版则进一步只保留前两列,
#     得到 3x2 相位 00/10/10, 再按 (u,v) 采样。结果不再是正方格重复, 而是沿两组对角线分别呈现
#     周期 3 和周期 2 的菱形链。
# - complexity_calibration:
#   - 仍只保留原题 4 种字形, 不引入新对象和新颜色, 防止规则支离破碎。
#   - 3x2 相位砖块直接来自原题 3x3 相位码的前两列: 这样 4 种字形仍然给出 4 个互异背景纹理,
#     同时第二个对角轴真的只有 2 个相位, 视觉上能稳定看出“两个方向周期不同”。
#   - 若简单把 3x3 相位硬映射成 2x2 或只看单一特征, 很容易让不同字形碰撞成同一输出; 这里避免了
#     这种简化退化。
# - implicit_assumption_review:
#   - `bbox(fg)`、`crop_bbox(...).shape == (3,3)`、输入恰有一种非背景色这些原假设仍然成立,
#     生成器继续保证。
#   - 原 solver 的关键假设 `tile_fill` 使用正交坐标已失效; 这里必须改成对角坐标采样。
#   - 原 solver 还默认两个平铺轴周期都等于 3; 融合后改成 `u` 轴长度 3、`v` 轴长度 2,
#     所以输出底纹不再能由单个 3x3 正交 tile 解释。
#   - 中央 6x6 栗色盖章仍是后覆盖层, 与底纹如何铺贴无冲突。
# - placement_legality:
#   - 输入生成器仍只放一个 3x3 字形, top-left 在 0..3 范围内, 不越界也不会有第二个物体冲突。
#   - 输出中央印章固定为 6x6, 放在 16x16 中心, 与原题相同, 始终完全落在画布内。
# - visual_observability:
#   - 只看 2-4 个样例, 人眼能直接看到背景橙点不是横竖重复, 而是斜向菱形格栅。
#   - 同时不同样例间, 某一组对角线的重复间距是 3, 另一组是 2; 这会明显区别于原题和仅 D15/0 的
#     等周期对角格栅。
#   - 四种字形仍会产生四种不同底纹, 所以解题者必须继续从 3x3 字形推断相位, 而不是只认“有斜纹”。
# - cleanliness:
#   - 画面仍只有两层: 黑/橙底纹和中央栗色印章。底纹虽变为菱形格栅, 但没有额外覆盖或多颜色混杂。
#   - 中心印章仍保持原字形 2x2 平铺, 与背景颜色和几何方向都明显区分, 视觉上整洁。
# - color_inference:
#   - 颜色语义完全没变: 橙/黑只用于底纹, 栗色只用于中心盖章。规则依赖几何与周期, 不依赖颜色编号。
# - post_edit_assumptions:
#   - `main()` 新增假设是: 有效字形经原题解码后取前两列, 四类都能形成非退化 3x2 相位砖块。
#   - `generate_input()` 只从这 4 个合法字形中采样, 因而该假设总被满足。

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


def _decode_base_phase_tile(motif: np.ndarray) -> np.ndarray:
    """Decode the original 3x3 contour catalog into the seed task's 3x3 phase mask."""
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


def _decode_anisotropic_phase_tile(motif: np.ndarray) -> np.ndarray:
    """
    Convert the seed 3x3 phase code into a 3x2 anisotropic phase code.

    Row phases remain the original 3-step ordering; column phases collapse to the
    first two seed columns so the second diagonal axis has period 2.
    """
    base_phase = _decode_base_phase_tile(motif)
    phase = base_phase[:, :2].copy()
    assert phase.shape == (3, 2)
    return phase


def _tile_fill_diagonal_anisotropic(shape: tuple[int, int], tile: np.ndarray) -> np.ndarray:
    """Repeat `tile` on diagonal axes using periods tile.shape == (3, 2)."""
    h, w = tile.shape
    rr, cc = np.indices(shape)
    return tile[(rr + cc) % h, (rr - cc) % w]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the 3x3 contour glyph into a 16x16 rhombic lattice whose two diagonal axes use different periods, then stamp the motif in maroon at the center.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels form one tight 3x3 contour glyph inside its bbox.
    - The glyph is one of the four seed-task codes: plus, H, X, or four corners.

    Rule:
    - Decode the glyph with the seed task's contour logic, then compress that 3x3 phase code into a 3x2 anisotropic phase tile.
    - Fill a black 16x16 canvas by sampling that tile in diagonal coordinates, with `u=(row+col) mod 3` and `v=(row-col) mod 2`.
    - Tile the original 3x3 glyph 2x2 and overwrite its true cells with maroon in the centered 6x6 region.

    Key insight:
    - The contour glyph now controls a diagonal coordinate system twice: its 3-row phase ordering sets one family of slanted lattice lines, while its first two phase columns set a different 2-phase alternation on the opposite diagonal family.
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

    phase = _decode_anisotropic_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)
    out = _tile_fill_diagonal_anisotropic(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[rng.choice(list(GLYPHS.keys()))]
    fg = int(rng.choice(FG_COLORS))
    r0 = int(rng.integers(0, MOTIF_POS_MAX))
    c0 = int(rng.integers(0, MOTIF_POS_MAX))

    inp = np.full((GRID_H, GRID_W), Color.ORANGE, dtype=int)
    inp[r0 : r0 + MOTIF_SIZE, c0 : c0 + MOTIF_SIZE][motif] = fg

    try:
        main(inp)
    except Exception:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_RETRIES):
        g = _gen_attempt(rng)
        if g is not None:
            return g
    raise RuntimeError("Failed to generate input for combo eee78d87 D6/1+D15/0")
