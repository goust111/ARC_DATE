# Variant of dbff022c
# Dimension: D9 — 多步处理顺序——精灵的处理优先级 (multi_step)
# Change: 所有精灵独立处理，顺序无关（填充结果互不影响），调色板先定位再逐个精灵查表填充... → 两遍处理——第一遍填充匹配精灵，第二遍删除调色板

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import components_mask, neighbors4

# Design analysis
# ---
# spec_variant:
#   core_change:
#     original: 调色板在输出中保留，solver 只做“查表后填洞”。
#     variant: 必须先利用调色板完成所有匹配洞的填充，再把整块调色板擦成背景色。
#   concrete_pixel_behavior:
#     input:
#       - 一块 2xN 或 Nx2 的边缘调色板，外侧一行/列给边框色，内侧一行/列给填充色。
#       - 若干空心精灵，其中 1-3 个边框色出现在调色板 key 中，另有 1 个干扰精灵不在映射里。
#       - 所有可匹配精灵的洞在输入中仍是背景色。
#     output:
#       - 所有可匹配精灵的洞被映射色填满。
#       - 调色板整块被背景色覆盖，因此边缘只剩空白。
#   trace_example:
#     before:
#       - 顶边有 palette: 第一行 [2,6,4]，第二行 [7,1,8]。
#       - 网格中有边框色 2 的空心矩形和边框色 9 的干扰矩形。
#     original_flow:
#       - 定位顶部 palette，读出 2->7, 6->1, 4->8。
#       - 将色 2 的洞填成 7，保留 palette 原样。
#     variant_flow:
#       - 第一步同样读出映射并填充色 2 的洞。
#       - 第二步把 palette 占据的整块区域写回背景色 0。
#       - 差异是最终输出边缘的调色板消失，这要求“先读再删”的顺序语义。
#   complexity_calibration:
#     - 保留原题 palette 方向、长度、空心精灵形状与数量范围，避免引入新的匹配自由度。
#     - 变体的可观察差异固定而明显：每个样例都会同时出现“内部填充”和“边缘 palette 消失”。
#     - 不使用颜色 id 排序、大小排序等不可见条件；删除对象始终是可见的 palette 整块。
#   implicit_assumption_review:
#     - _bg_boundary 依赖输入边界上背景仍占多数。生成器继续保留原校验，确保 palette 虽在边缘但不主导边界统计。
#     - _best_palette / _palette_mapping 依赖 palette 在输入时存在且与 sprite 颜色隔离。变体不改变输入阶段，所以这些前提仍成立。
#     - _fill_enclosed_bg_regions 依赖 palette 在填充阶段不能被当成普通背景区域；因此仍需用 pal_mask 排除 palette。
#     - 原 main 在填充后直接返回，这默认“调色板在最终状态中保留”。该假设在变体中失效，必须显式增加第二步擦除。
#     - 第二步不能先做；若先删 palette，就失去读取映射的载体，solver 无法恢复 border->fill 对应关系。
#   placement_legality:
#     - palette 仍贴边放置，并在 occ 中占据 1 格缓冲区，保证与 sprites 至少隔 1 格背景。
#     - sprites 的洞不得接触外边界，保证 solver 识别到的可填区域始终是封闭背景连通块。
#     - output 删除 palette 后只会在边缘产生额外背景，不会与 sprite 内洞相连，因为 sprites 与 palette 之间已有缓冲。
#   visual_observability:
#     - 人类从 2-4 个样例里能同时看到两个稳定现象：匹配精灵被填色、边缘图例整块消失。
#     - 这个现象不依赖颜色数值，只依赖“哪块是图例、哪块是空心精灵”的几何结构。
#   cleanliness:
#     - 保留原有对象稀疏放置策略，删除 palette 后边缘更干净，不会增加重叠或遮挡。
#   color_inference:
#     - 颜色映射仍由 palette 的双行/双列结构直接提示；删除发生在读取之后，但输入里图例完整可见，推理线索未减少。
#   generator_guarantees_for_new_main:
#     - 至少 1 个可匹配精灵，避免样例只体现“删除 palette”而没有“先填后删”。
#     - palette 区域在输入中全部为非背景，在输出中全部变为背景，确保与原题行为显著不同。


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _best_palette(g: np.ndarray, bg: int) -> tuple[str, int, int]:
    """
    Find the 2xN or Nx2 palette block on a grid edge.

    Returns (side, start, end) where:
      - side in {"top","bottom","left","right"}
      - for top/bottom: start/end are column indices [start,end)
      - for left/right: start/end are row indices [start,end)
    """
    h, w = g.shape
    best = None  # (score_tuple, side, start, end)

    def isolated(side: str, start: int, end: int) -> bool:
        """
        Palette cells (excluding bg) should not be 4-neigh adjacent to same-colored cells
        outside the palette region; this prevents accidentally absorbing parts of large sprites
        that touch the grid edge.
        """
        h2, w2 = g.shape
        pal = np.zeros((h2, w2), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h2 - 2 : h2, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w2 - 2 : w2] = True

        for r, c in zip(*np.where(pal), strict=False):
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h2 and 0 <= cc < w2 and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
                        return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[0, :]
        b = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        a = g[h - 1, :]
        b = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_a = a[c0:c1]
                seg_b = b[c0:c1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, 0]
        b = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        a = g[:, w - 1]
        b = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_a = a[r0:r1]
                seg_b = b[r0:r1]
                da = set(map(int, np.unique(seg_a)))
                db = set(map(int, np.unique(seg_b)))
                da.discard(bg)
                db.discard(bg)
                if len(da) < 2 or len(db) < 2:
                    continue
                d = da | db
                if len(d) < 3:
                    continue
                border_keys = [int(x) for x in seg_a.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_a != bg) + np.sum(seg_b != bg))
                cand = ((len(set(border_keys)), len(d), -(r1 - r0), nonbg), "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray]:
    """Return (mapping border->fill, palette_mask)."""
    h, w = g.shape
    side, start, end = _best_palette(g, bg)
    pal_mask = np.zeros((h, w), dtype=bool)
    mapping: dict[int, int] = {}

    if side == "top":
        pal_mask[0:2, start:end] = True
        border_line = g[0, start:end]
        fill_line = g[1, start:end]
    elif side == "bottom":
        pal_mask[h - 2 : h, start:end] = True
        border_line = g[h - 1, start:end]
        fill_line = g[h - 2, start:end]
    elif side == "left":
        pal_mask[start:end, 0:2] = True
        border_line = g[start:end, 0]
        fill_line = g[start:end, 1]
    else:
        pal_mask[start:end, w - 2 : w] = True
        border_line = g[start:end, w - 1]
        fill_line = g[start:end, w - 2]

    for b, f in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        b = int(b)
        if b == bg:
            continue
        mapping[b] = int(f)

    return mapping, pal_mask


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Find non-palette background regions that do not touch the grid boundary,
    then fill each one by the dominant mapped border color around it.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(
            r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts
        )
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    v = int(out[rr, cc])
                    if v != bg:
                        adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col in mapping and cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        out[tuple(pts.T)] = int(mapping[border])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use an edge legend to color hollow sprites, then remove the legend itself.

    Input structure:
    - The background is the dominant border color, and a 2-cell-thick palette sits on one outer edge.
    - Elsewhere are several hollow single-color sprites; some border colors appear in the palette's key line, and one distractor sprite does not.

    Rule:
    - First detect the palette, read its outer line as border colors and its inner line as the corresponding fill colors, then fill every enclosed non-boundary hole whose surrounding border matches a palette key.
    - After all eligible holes have been filled, overwrite the entire palette block with the background color so the legend disappears from the final grid.

    Key insight:
    - The palette must be interpreted before it is erased: it is both the lookup table for the fills and the object removed in the second pass.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    out[pal_mask] = bg
    return out


# --------------- generator (c) ---------------

# -- Configuration constants --------------------------------------------------
_GRID_H_MIN, _GRID_H_MAX = 4, 13
_GRID_W_MIN, _GRID_W_MAX = 10, 15
_SMALL_H_THRESHOLD = 6
_SMALL_H_PALETTE_FLIP = 0.7
_TINY_H_THRESHOLD = 5

_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

_MAX_RETRIES = 300
# -- End configuration --------------------------------------------------------


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components inside the sprite bbox that avoid the bbox boundary."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped boolean mask."""
    s = np.zeros((3, 3), dtype=bool)
    s[0, 1] = s[1, 0] = s[1, 2] = s[2, 1] = True
    return s


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole and limited area."""
    sprite = np.ones((h, w), dtype=bool)
    if h >= 3 and w >= 3:
        sprite[1:-1, 1:-1] = False
    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        n = rng.randint(1, w - 2)
        s = rng.randint(1, w - n - 1)
        sprite[br, s : s + n] = True
    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        n = rng.randint(1, h - 2)
        s = rng.randint(1, h - n - 1)
        sprite[s : s + n, bc] = True
    if rng.random() < _P_BORDER_REMOVAL:
        bcells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(bcells)
        for cr, cc in bcells[: rng.randint(1, min(4, len(bcells)))]:
            test = sprite.copy()
            test[cr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test
    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(
    rng: random.Random, n_pairs: int, num_filled: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """Pick palette colors and the keys used by the filled sprites."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n_pairs]
    rng.shuffle(all_c)
    fill_values = all_c[:n_pairs]
    distractor_color = rng.choice([c for c in range(1, 10) if c not in border_keys])
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the given edge side."""
    n_pairs = len(border_keys)
    if side in ("top", "bottom"):
        pal = np.zeros((2, n_pairs), dtype=int)
        outer, inner = (0, 1) if side == "top" else (1, 0)
        pal[outer] = border_keys
        pal[inner] = fill_values
    else:
        pal = np.zeros((n_pairs, 2), dtype=int)
        outer, inner = (0, 1) if side == "left" else (1, 0)
        pal[:, outer] = border_keys
        pal[:, inner] = fill_values
    return pal


def _place_palette(
    grid: np.ndarray, occ: np.ndarray, pal: np.ndarray, side: str, rng: random.Random
) -> np.ndarray:
    """Place the palette and return its exact mask."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        r0, c0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        r0, c0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        r0, c0 = rng.randint(0, h - pal_h), 0
    else:
        r0, c0 = rng.randint(0, h - pal_h), w - pal_w

    grid[r0 : r0 + pal_h, c0 : c0 + pal_w] = pal
    occ[max(0, r0 - 1) : min(h, r0 + pal_h + 1), max(0, c0 - 1) : min(w, c0 + pal_w + 1)] = True

    pal_mask = np.zeros_like(grid, dtype=bool)
    pal_mask[r0 : r0 + pal_h, c0 : c0 + pal_w] = True
    return pal_mask


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find placements that avoid overlap and keep each hole away from the grid border."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hpts in holes
                for hr, hc in hpts
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int | None,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite: border in color, and optionally pre-filled holes."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + hr, c0 + hc] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    if h <= _SMALL_H_THRESHOLD and side in ("top", "bottom") and rng.random() < _SMALL_H_PALETTE_FLIP:
        side = rng.choice(["left", "right"])

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if h <= _TINY_H_THRESHOLD:
        max_sh = 3

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_filled + 1

    shapes = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(
        rng, n_pairs, num_filled
    )
    pal = _build_palette(side, border_keys, fill_values)

    solved = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    pal_mask = _place_palette(solved, occ, pal, side, rng)

    infos = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            color = filled_keys[i]
            fill = mapping[color]
        else:
            color = distractor_color
            fill = None
        infos.append((mask, color, fill))

    placed = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(solved, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill, holes))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = solved.copy()
    output[pal_mask] = bg

    inp = solved.copy()
    for r0, c0, mask, fill, holes in placed:
        if fill is None:
            continue
        for hpts in holes:
            for hr, hc in hpts:
                inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    if not np.all(inp[pal_mask] != bg):
        return None
    if not np.all(output[pal_mask] == bg):
        return None
    if np.array_equal(inp, output):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        candidate = _gen_attempt()
        if candidate is not None:
            return candidate
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
