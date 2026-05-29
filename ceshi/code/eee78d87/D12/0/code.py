# Variant of eee78d87
# Dimension: D12 — 输入网格尺寸 (meta)
# Change: 固定6x6输入网格... → 输入网格尺寸可变

from __future__ import annotations

import numpy as np

from arc_utils import Color, bbox, colors_present, crop_bbox, get_bg, tile_fill


# Design analysis for the D12 variant
# -----------------------------------
# variant_spec:
#   preserved_structure:
#     - 输入仍然只有一个 3x3 轮廓字形，背景仍为橙色。
#     - 字形类别仍限定为 plus / H / X / 4corners 四种。
#     - 输出仍由周期 3 的橙黑底纹 + 居中的 2x2 栗色印章组成。
#   core_change:
#     - 唯一核心改动是输入从固定 6x6 改为可变 N x N，N 在生成时取 4..9。
#     - 输出同步改为 (3N-2) x (3N-2)，让解题者必须从样例中归纳尺寸公式。
#   concrete_behavior:
#     - 输入：橙色 N x N 画布上放一个 3x3 非橙色字形，位置仍可任意，只要完整落在网格内。
#     - 输出：先把字形解码为 3x3 相位瓦片，再平铺到 (3N-2) x (3N-2) 画布；最后把原字形 2x2 平铺成 6x6 栗色印章，放在输出正中央。
#   behavior_trace:
#     - 原始逻辑：若输入是 6x6，则输出固定为 16x16，中心印章总在 (5,5) 起始。
#     - 变体逻辑：若输入是 4x4，则输出为 10x10，中心印章起始变为 (2,2)；
#       若输入是 9x9，则输出为 25x25，中心印章起始变为 (9,9)。
#     - 差异不在字形解码本身，而在“同一解码结果被放到多大画布上，以及中心区域落点如何随 N 改变”。
#   complexity_calibration:
#     - 只引入一个自由度 N，保持四种字形分支不变，避免规则被其他新增机制稀释。
#     - N 的取值范围控制在 4..9，能明显看到输出尺寸变化，同时不会让输入过空或输出过大。
#   original_assumption_audit:
#     - `assert g.shape == (6, 6)` 依赖固定尺寸，在变体中必须移除，改为检查方形且边长至少 4。
#     - `out_shape = (3 * g.shape[0] - 2, 3 * g.shape[1] - 2)` 原本就具有泛化性，可以保留。
#     - 中心印章通过 `(out - 6) // 2` 定位；只要 N >= 4，则输出边长 >= 10，6x6 印章始终能完整放下。
#     - 生成器中的 `GRID_H/W` 与 `MOTIF_POS_MAX` 依赖固定 6x6，必须改为从随机 N 动态推导。
#   placement_legality:
#     - 生成器选 N 后，只在 `0..N-3` 内采样字形左上角，保证 3x3 bbox 不越界。
#     - 输入仍只有一个前景对象，因此不会出现对象重叠或 solver 误分割。
#   visual_observability:
#     - 两三个样例就能直接看见：输入画布变大时，输出底纹边长按 3N-2 同步放大，而中央栗色 6x6 印章保持不变。
#     - 若始终生成 N=6，会退化回原题；因此生成器必须覆盖多个 N，并且大量样本要落在 N!=6。
#   visual_cleanliness:
#     - 输出仍然只有“重复底纹 + 中心印章”两层，结构清晰，不引入额外覆盖冲突。
#   color_inference:
#     - 颜色逻辑完全不变：输入前景色只负责标记字形，输出只使用黑/橙/栗色三色，人类无需推断新的颜色映射。
#   new_solver_assumptions:
#     - 输入必须是橙色背景、恰有一种非橙色、其整体 bbox 恰为 3x3。
#     - 生成器通过单个 3x3 字形的直接放置来保证这些假设恒成立。


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the colored 3x3 contour motif into a variable-size periodic lattice whose size is determined by the input side length.

    Input structure:
    - The grid is square, orange-background, and its side length can vary across examples.
    - Exactly one non-orange color appears, forming a tight 3x3 bbox that contains one contour glyph.
    - The glyph is one of the task's four period-3 contour forms: plus, H, X, or four corners.

    Rule:
    - Read the 3x3 glyph as a code for a 3x3 orange/black phase tile.
    - Build an output canvas of size (3N-2)x(3N-2), where N is the input side length, and fill it by tiling that phase tile.
    - Tile the original 3x3 glyph 2x2 into a 6x6 pattern and recolor its true cells to maroon at the exact center of the output.

    Key insight:
    - The input object still encodes the lattice phase rather than a literal copy target, but now the surrounding canvas size must be inferred from the input grid size instead of treated as a fixed constant.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    assert h == w
    assert h >= 4

    bg = int(get_bg(g))
    assert bg == Color.ORANGE
    assert len(colors_present(g, exclude={bg})) == 1

    fg = g != bg
    b = bbox(fg)
    assert b is not None
    motif = crop_bbox(fg, b).astype(bool)
    assert motif.shape == (3, 3)

    phase = _decode_phase_tile(motif)
    phase_colors = np.where(phase, bg, Color.BLACK).astype(int)
    out_shape = (3 * h - 2, 3 * w - 2)
    out = tile_fill(out_shape, phase_colors)

    center = np.tile(motif, (2, 2))
    r0 = (out.shape[0] - center.shape[0]) // 2
    c0 = (out.shape[1] - center.shape[1]) // 2
    out[r0 : r0 + center.shape[0], c0 : c0 + center.shape[1]][center] = Color.MAROON
    return out


# --------------- generator (cx) ---------------

# --- CFG: tunable generation parameters ---
MIN_GRID_SIZE = 4                             # inclusive lower bound for N
MAX_GRID_SIZE = 9                             # inclusive upper bound for N
MOTIF_SIZE = 3                                # glyph bounding box side length
FG_COLORS = [1, 2, 3, 4, 5, 6, 8]            # valid foreground colors (exclude 0, 7, 9)
MAX_RETRIES = 300                             # oracle-validation retry budget

GLYPHS = {
    "plus": np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    "H": np.array([[1, 0, 1], [1, 1, 1], [1, 0, 1]], dtype=bool),
    "X": np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]], dtype=bool),
    "4corners": np.array([[1, 0, 1], [0, 0, 0], [1, 0, 1]], dtype=bool),
}
GLYPH_NAMES = tuple(GLYPHS)


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n = int(rng.integers(MIN_GRID_SIZE, MAX_GRID_SIZE + 1))
    motif = GLYPHS[str(rng.choice(GLYPH_NAMES))]
    fg = int(rng.choice(FG_COLORS))
    pos_max = n - MOTIF_SIZE + 1
    r0 = int(rng.integers(0, pos_max))
    c0 = int(rng.integers(0, pos_max))

    inp = np.full((n, n), Color.ORANGE, dtype=int)
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
    raise RuntimeError("Failed to generate input for eee78d87 D12 variant")
