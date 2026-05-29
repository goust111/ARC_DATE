# Combo variant of eee78d87 (index 31)
# combo_spec: D9/2+D11/1
# Name: 单次字形不平铺+对齐瓦片边界

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


"""
design_analysis:
  a_concretization_and_trace:
    - 原题把输入里的 3x3 轮廓字形同时用于两件事：先解码成周期 3 的橙黑瓦片，再把该字形做 2x2 平铺，
      形成固定放在 (5,5)-(10,10) 的 6x6 栗色中心印章。
    - 本融合版保留第一件事不变，只改第二件事：印章不再 2x2 平铺，而是直接使用原始 3x3 字形一次。
    - 为了让 D9/2 真正起作用，不把单次字形直接几何居中；而是先沿用原题中心印章的旧锚点 (5,5)，
      再把这个 3x3 印章吸附到最近的 3x3 瓦片边界，得到 (6,6)。这样输出里会出现“只覆盖一个瓦片单元”
      的微小栗色标记。
    - 例子：若输入是 plus 字形，原题输出会在整张 16x16 相位格栅中央放一个 6x6 的 2x2 plus 印章；
      本题输出仍是同一格栅，但只在 (6:9,6:9) 这个单一瓦片里放一个 3x3 栗色 plus。
  b_complexity_calibration:
    - 仅保留 4 种原始字形，不新增额外颜色分支，也不让印章位置多档位漂移，避免规则过散。
    - D11/1 带来明显的尺寸变化：栗色区从 6x6 缩成 3x3；D9/2 带来明确的结构变化：印章边缘与周期 3 格栅精确对齐。
    - 若直接把 3x3 印章放到几何中心，D9/2 会退化成无效描述；因此显式采用“旧中心锚点 -> 最近边界”的两步逻辑。
  c_implicit_assumption_audit:
    - _decode_phase_tile() 仍假设 bbox 恰为 3x3 且字形属于 plus/H/X/4corners；generator 保持这一点。
    - 原 main() 默认中心印章尺寸是 6x6，并据此用 (H-6)//2 得到旧锚点 5；融合后不能再用印章自身尺寸做居中，
      否则会直接落在 6，掩盖 D9/2，所以改为单独保留 legacy anchor。
    - 覆盖栗色时依赖印章完全落在画布内；吸附候选仅取 0,3,6,9,12，且 3x3 印章因此不会越界。
  d_placement_legality:
    - 输入仍只有一个 3x3 字形，放在 6x6 橙底上任意可容纳位置，不会重叠或越界。
    - 输出格栅由 tile_fill 铺满；栗色印章只覆盖一个完整 3x3 瓦片单元，不与边界裁切区冲突。
  e_visual_observability:
    - 从少量样例中，人类能同时看到两件稳定事实：背景格栅仍由字形解码；栗色标记不再是 6x6 大块，而是一个
      与格栅边界重合的 3x3 小字形。
    - 对齐规则通过“栗色块正好占满一个格栅单元”体现，不依赖颜色编号或坐标文字。
  f_visual_cleanliness:
    - 输出只含黑/橙/栗三色，且栗色层只覆盖一个瓦片，叠加关系单纯，不会让中心区域变脏。
    - 小印章与周期格栅共边，视觉上像一个被替换颜色的单元，而不是漂浮在格栅上的噪声。
  g_inferability:
    - 颜色变化没有额外映射，唯一新行为是“原字形单次印章 + 吸附到瓦片边界”，并且在所有样例中稳定重复。
    - 生成器不需要编码隐藏条件；只要给出不同字形，解题者就能从输出同时观察到相位格栅和单瓦片印章的双重作用。
"""


GRID_H, GRID_W = 6, 6
MOTIF_SIZE = 3
TILE_PERIOD = 3
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


def _snap_to_tile_boundary(start: int, max_start: int, period: int) -> int:
    candidates = list(range(0, int(max_start) + 1, int(period)))
    assert candidates
    return min(candidates, key=lambda x: (abs(int(x) - int(start)), int(x)))


def _aligned_stamp_origin(out_shape: tuple[int, int], stamp_shape: tuple[int, int]) -> tuple[int, int]:
    """
    Keep the legacy center-stamp anchor, then snap it to the nearest period-3 boundary.

    The original task centered a 6x6 tiled stamp at top-left (5,5). In this combo the
    stamp shrinks to 3x3, but we still start from that legacy anchor so the later snap
    step is observable and not silently absorbed by ordinary centering.
    """
    out_h, out_w = map(int, out_shape)
    stamp_h, stamp_w = map(int, stamp_shape)
    legacy_span = 2 * MOTIF_SIZE
    legacy_r0 = (out_h - legacy_span) // 2
    legacy_c0 = (out_w - legacy_span) // 2
    r0 = _snap_to_tile_boundary(legacy_r0, out_h - stamp_h, TILE_PERIOD)
    c0 = _snap_to_tile_boundary(legacy_c0, out_w - stamp_w, TILE_PERIOD)
    return int(r0), int(c0)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour glyph into a 16x16 periodic lattice, then replace one aligned lattice cell with a maroon copy of the glyph.

    Input structure:
    - The grid is 6x6 with orange background and exactly one other color.
    - The non-orange pixels occupy a tight 3x3 bbox containing one contour glyph.
    - The glyph is one of the task's four phase codes: plus, H, X, or four corners.

    Rule:
    - Decode the 3x3 glyph into the period-3 orange/black phase tile and repeat it across a 16x16 canvas.
    - Keep the original task's centered-stamp anchor as the reference placement, but do not 2x2 tile the glyph.
    - Use the raw 3x3 glyph once, recolor its true pixels to maroon, and snap that 3x3 stamp to the nearest 3x3 tile boundary so it exactly covers one lattice cell.

    Key insight:
    - The same tiny contour serves as both a code for the global lattice phase and a local marker, but in this variant the marker is reduced to a single tile-sized stamp aligned with the lattice grid.
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
    out_shape = (TILE_PERIOD * g.shape[0] - 2, TILE_PERIOD * g.shape[1] - 2)
    out = tile_fill(out_shape, phase_colors)

    stamp = motif
    r0, c0 = _aligned_stamp_origin(out.shape, stamp.shape)
    out[r0 : r0 + stamp.shape[0], c0 : c0 + stamp.shape[1]][stamp] = Color.MAROON
    return out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    motif = GLYPHS[str(rng.choice(list(GLYPHS.keys())))]
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
    raise RuntimeError("Failed to generate input for eee78d87 combo variant")
