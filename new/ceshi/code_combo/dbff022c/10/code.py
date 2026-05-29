# Combo variant of dbff022c (index 10)
# combo_spec: D7/2+D10/1
# Name: 未匹配精灵自填充 + 调色板条目少于精灵

import random

import numpy as np

from solvers.arc_utils import components_mask, neighbors4


# Design analysis for the combo variant
# -------------------------------------
# spec_variant:
#   concrete_example_trace:
#     - 设 palette 在顶边，给出 2 个条目：3->6、4->2；场上有 4 个空心精灵，边框色分别是 3、3、4、7。
#     - 原题逻辑：
#       1. 读取 palette 后得到 mapping={3:6, 4:2}。
#       2. 对边框为 3 的两个洞，都填 6；对边框为 4 的洞填 2；对边框为 7 的洞保持 0。
#     - 本融合逻辑：
#       1. 同样先读出 mapping={3:6, 4:2}。
#       2. 两个边框为 3 的精灵共享同一个 palette 条目，两个洞都填 6。
#       3. 边框为 4 的精灵正常按唯一映射填 2。
#       4. 边框为 7 的精灵不再保持空心，而是把洞填成 7，自身变成实心块。
#     - 差异点有三类：正常映射填充、多个同色精灵共享映射、无映射时自填充。
#   complexity_calibration:
#     - 为了让 D10/1 真正可见，generator 不允许“palette 条目不少于精灵”的退化情况，而是强制
#       `mapped_sprites = palette_entries + 1`，再额外加入 1 个未匹配精灵。
#     - 为了让“共享映射”和“普通映射”同时存在，所有 palette key 至少各出现 1 次，并额外随机挑 1 个 key
#       再出现 1 次；这样同一输入里一定同时有 singleton key 和 duplicated key。
#     - 为了让 D7/2 可见，恰好保留 1 个未匹配精灵，且其颜色优先避开 palette key 与 fill value，
#       这样输出里会出现明显的“自填充实心块”。
#   hidden_assumptions_review:
#     - `_best_palette` 依赖“palette 外侧 key 唯一”来读成字典；融合后仍保持 palette 条目本身唯一，
#       变化只发生在场上 sprite 数量与颜色复用关系，因此该假设仍成立。
#     - 原 `_fill_enclosed_bg_regions` 的旧假设是“只填有 mapping 的颜色，其他颜色忽略”。D7/2 打破了
#       这个前提，所以实现必须先确定洞对应的主边框色，再做 `mapping.get(border, border)`。
#     - 原 solver 实际上按“边框颜色类别”而不是“精灵实例身份”工作，因此 D10/1 不需要新的匹配机制；
#       多个同色精灵天然共享一个条目，这点必须由 generator 构造出来，而不是在 solver 里额外编码。
#     - 洞周围主边框色的判定仍依赖“每个内部洞主要被单一颜色包围”。generator 继续保证 sprite 与
#       sprite、sprite 与 palette 之间至少隔 1 格背景，避免一个洞同时邻接多个对象造成歧义。
#     - `out` 上原地回填仍安全，因为各洞互不接触；先填完一个洞不会改变另一个洞的主边框色统计。
#   placement_legality:
#     - palette 仍贴边放置，并通过 `occ` 给出 1 格缓冲区，防止与任何 sprite 4 邻接。
#     - 所有 sprite 仍要求不重叠、不越界、彼此隔开，且它们的内部洞不能碰到整张网格边界，
#       这样 solver 的“内部背景连通域”判定与生成条件完全一致。
#     - 由于 sprite 总数提高到 4 或 5，生成器缩小 palette 条目范围到 2-3，并偏向较大的网格、
#       略小的 sprite 尺寸，以提高合法放置成功率。
#   visual_observability:
#     - 每个样本都同时包含：
#       1. 至少 1 个正常 palette-mapped 精灵；
#       2. 至少 2 个共享同一 key 的精灵；
#       3. 至少 1 个未匹配且会自填充的精灵。
#     - 解题者在 2-4 个样例里能直接看到：同色边框的多个对象都变成同一种填充色，而另一种不在图例中的
#       边框则把内部补成自身颜色。这是结构线索，不依赖颜色编号大小之类的人类不可视信息。
#   cleanliness_and_inference:
#     - 画面仍只包含背景、边缘 palette、分离的空心精灵三类元素，没有叠加额外标记。
#     - 颜色总数被限制在少量档位：2-3 个 palette key、对应 fill，以及 1 个 unmatched 颜色；
#       颜色变化始终由“边框色是否出现在图例中”这个结构条件触发，推断成本可控。


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
        outside the palette region; this prevents accidentally absorbing parts of edge sprites.
        """
        pal = np.zeros((h, w), dtype=bool)
        if side == "top":
            pal[0:2, start:end] = True
        elif side == "bottom":
            pal[h - 2 : h, start:end] = True
        elif side == "left":
            pal[start:end, 0:2] = True
        else:
            pal[start:end, w - 2 : w] = True

        for r, c in zip(*np.where(pal), strict=False):
            value = int(g[r, c])
            if value == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == value:
                        return False
        return True

    def consider_top() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[0, :]
        inner = g[1, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                union = outer_colors | inner_colors
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(union), -(c1 - c0), nonbg), "top", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_bottom() -> None:
        nonlocal best
        if h < 2:
            return
        outer = g[h - 1, :]
        inner = g[h - 2, :]
        for c0 in range(w):
            for c1 in range(c0 + 2, w + 1):
                seg_outer = outer[c0:c1]
                seg_inner = inner[c0:c1]
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                union = outer_colors | inner_colors
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(union), -(c1 - c0), nonbg), "bottom", c0, c1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_left() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, 0]
        inner = g[:, 1]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                union = outer_colors | inner_colors
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(union), -(r1 - r0), nonbg), "left", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    def consider_right() -> None:
        nonlocal best
        if w < 2:
            return
        outer = g[:, w - 1]
        inner = g[:, w - 2]
        for r0 in range(h):
            for r1 in range(r0 + 2, h + 1):
                seg_outer = outer[r0:r1]
                seg_inner = inner[r0:r1]
                outer_colors = set(map(int, np.unique(seg_outer)))
                inner_colors = set(map(int, np.unique(seg_inner)))
                outer_colors.discard(bg)
                inner_colors.discard(bg)
                if len(outer_colors) < 2 or len(inner_colors) < 2:
                    continue
                union = outer_colors | inner_colors
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                cand = ((len(border_keys), len(union), -(r1 - r0), nonbg), "right", r0, r1)
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

    for border, fill in zip(border_line.tolist(), fill_line.tolist(), strict=False):
        border = int(border)
        fill = int(fill)
        if border == bg:
            continue
        mapping[border] = fill

    return mapping, pal_mask


def _fill_enclosed_bg_regions(
    out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray
) -> None:
    """
    Fill each enclosed background region using its dominant adjacent border color.

    Mapped border colors use the palette's target fill.
    Unmapped border colors fall back to self-fill.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w:
                    if pal_mask[rr, cc]:
                        continue
                    value = int(out[rr, cc])
                    if value == bg:
                        continue
                    adj_counts[value] = adj_counts.get(value, 0) + 1
        if not adj_counts:
            continue

        border = max(adj_counts.items(), key=lambda item: item[1])[0]
        fill = int(mapping.get(border, border))
        for r, c in pts:
            out[r, c] = fill


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading a compact edge legend, with shared mappings and self-fill fallback.

    Input structure:
    - The background dominates the outer border, and one grid edge contains a 2xN or Nx2 legend.
    - The legend's outer strip lists distinct border colors, while the inner strip lists the fill colors paired to them.
    - There are more hollow single-color sprites than legend entries: some sprites reuse the same border color, and at least one sprite uses a border color absent from the legend.

    Rule:
    - Detect the edge legend and read it as a border-color to fill-color mapping.
    - Find every enclosed background hole outside the legend area and infer the border color surrounding that hole from dominant adjacency.
    - If that border color is present in the legend, fill the hole with the mapped color; otherwise fill it with the border color itself.

    Key insight:
    - The legend classifies border colors, not individual sprite instances. So repeated border colors across multiple sprites all share one mapping, while a missing key implies a default self-mapping rather than leaving the hole empty.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    mapping, pal_mask = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)
    return out


# --------------- generator ---------------

# Grid dimensions: bias upward because the combo needs 4-5 sprites with spacing.
_GRID_H_MIN, _GRID_H_MAX = 9, 13
_GRID_W_MIN, _GRID_W_MAX = 12, 15

# Palette: keep entries few so sprite count can exceed it without overcrowding.
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 3

# Sprite geometry: slightly smaller than the source task to improve placement success.
_SPRITE_H_MAX = 5
_SPRITE_W_MAX = 5
_SPRITE_DIM_SUM_CAP = 8
_SPRITE_CELL_MAX = 14

# Shape mutation probabilities
_P_HORIZ_WALL = 0.25
_P_VERT_WALL = 0.25
_P_BORDER_REMOVAL = 0.35
_P_DIAMOND = 0.2

_MAX_RETRIES = 500


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return point arrays for bg components that are internal to the sprite mask."""
    h, w = sprite.shape
    return [
        pts
        for pts in components_mask(~sprite, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _make_diamond() -> np.ndarray:
    """Return a 3x3 diamond-shaped boolean mask."""
    sprite = np.zeros((3, 3), dtype=bool)
    sprite[0, 1] = True
    sprite[1, 0] = True
    sprite[1, 2] = True
    sprite[2, 1] = True
    return sprite


def _make_hollow_sprite(h: int, w: int, rng: random.Random) -> np.ndarray | None:
    """Create a hollow sprite with at least one enclosed hole and bounded size."""
    sprite = np.ones((h, w), dtype=bool)
    sprite[1:-1, 1:-1] = False

    if h >= 5 and w >= 4 and rng.random() < _P_HORIZ_WALL:
        br = rng.randint(2, h - 2)
        wall_len = rng.randint(1, w - 3)
        start = rng.randint(1, w - wall_len - 1)
        sprite[br, start : start + wall_len] = True

    if h >= 4 and w >= 5 and rng.random() < _P_VERT_WALL:
        bc = rng.randint(2, w - 2)
        wall_len = rng.randint(1, h - 3)
        start = rng.randint(1, h - wall_len - 1)
        sprite[start : start + wall_len, bc] = True

    if rng.random() < _P_BORDER_REMOVAL:
        border_cells = [
            (r, c)
            for r in range(h)
            for c in range(w)
            if sprite[r, c] and (r in (0, h - 1) or c in (0, w - 1))
        ]
        rng.shuffle(border_cells)
        for rr, cc in border_cells[: rng.randint(1, min(3, len(border_cells)))]:
            test = sprite.copy()
            test[rr, cc] = False
            if _get_enclosed_holes(test):
                sprite = test

    if not _get_enclosed_holes(sprite) or int(sprite.sum()) > _SPRITE_CELL_MAX:
        return None
    return sprite


def _choose_colors(
    rng: random.Random, n_pairs: int
) -> tuple[list[int], list[int], int, dict[int, int], list[int]]:
    """
    Pick palette colors and sprite border colors.

    Mapped sprite keys use every palette key once, plus one extra repeated key.
    """
    all_colors = list(range(1, 10))
    rng.shuffle(all_colors)
    border_keys = all_colors[:n_pairs]

    fill_candidates = [c for c in range(1, 10) if c not in border_keys]
    rng.shuffle(fill_candidates)
    fill_values = fill_candidates[:n_pairs]

    distractor_candidates = [c for c in range(1, 10) if c not in border_keys and c not in fill_values]
    if distractor_candidates:
        distractor_color = rng.choice(distractor_candidates)
    else:
        distractor_color = rng.choice([c for c in range(1, 10) if c not in border_keys])

    mapping = dict(zip(border_keys, fill_values))
    repeated_key = rng.choice(border_keys)
    mapped_keys = border_keys + [repeated_key]
    rng.shuffle(mapped_keys)
    return border_keys, fill_values, distractor_color, mapping, mapped_keys


def _build_palette(side: str, border_keys: list[int], fill_values: list[int]) -> np.ndarray:
    """Construct the 2xN or Nx2 palette array for the chosen edge."""
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
) -> None:
    """Place palette on the chosen edge and reserve a 1-cell padded occupancy zone."""
    h, w = grid.shape
    pal_h, pal_w = pal.shape
    if side == "top":
        pr0, pc0 = 0, rng.randint(0, w - pal_w)
    elif side == "bottom":
        pr0, pc0 = h - pal_h, rng.randint(0, w - pal_w)
    elif side == "left":
        pr0, pc0 = rng.randint(0, h - pal_h), 0
    else:
        pr0, pc0 = rng.randint(0, h - pal_h), w - pal_w

    grid[pr0 : pr0 + pal_h, pc0 : pc0 + pal_w] = pal
    occ[
        max(0, pr0 - 1) : min(h, pr0 + pal_h + 1),
        max(0, pc0 - 1) : min(w, pc0 + pal_w + 1),
    ] = True


def _find_valid_positions(
    h: int, w: int, occ: np.ndarray, mask: np.ndarray, holes: list[np.ndarray]
) -> list[tuple[int, int]]:
    """Find legal placements where the sprite fits, stays separated, and keeps holes internal."""
    sh, sw = mask.shape
    positions: list[tuple[int, int]] = []
    for r0 in range(h - sh + 1):
        for c0 in range(w - sw + 1):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            if any(
                r0 + hr == 0 or c0 + hc == 0 or r0 + hr == h - 1 or c0 + hc == w - 1
                for hole in holes
                for hr, hc in hole
            ):
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    border_color: int,
    fill_color: int,
    holes: list[np.ndarray],
) -> None:
    """Render a sprite onto the grid using one border color and one hole fill color."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = border_color
    for hole in holes:
        for hr, hc in hole:
            grid[r0 + hr, c0 + hc] = fill_color


def _make_shapes(total_sprites: int, max_sh: int, max_sw: int, rng: random.Random) -> list[np.ndarray] | None:
    """Generate several hollow sprite masks."""
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
    return shapes


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))
    mapped_keys_count = n_pairs + 1
    total_sprites = mapped_keys_count + 1  # plus one unmatched self-fill sprite

    max_sh = min(_SPRITE_H_MAX, h)
    max_sw = min(_SPRITE_W_MAX, w)
    if total_sprites >= 5:
        max_sh = min(max_sh, 4)
        max_sw = min(max_sw, 4)

    shapes = _make_shapes(total_sprites, max_sh, max_sw, rng)
    if shapes is None:
        return None

    border_keys, fill_values, distractor_color, mapping, mapped_keys = _choose_colors(rng, n_pairs)
    pal = _build_palette(side, border_keys, fill_values)

    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)
    _place_palette(grid, occ, pal, side, rng)

    infos = []
    for mask, key in zip(shapes[:mapped_keys_count], mapped_keys, strict=False):
        infos.append((mask, key, mapping[key]))
    infos.append((shapes[-1], distractor_color, distractor_color))
    infos.sort(key=lambda item: -int(item[0].sum()))

    placed: list[tuple[int, int, np.ndarray]] = []
    for mask, border_color, fill_color in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(h, w, occ, mask, holes)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(grid, r0, c0, mask, border_color, fill_color, holes)
        placed.append((r0, c0, mask))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)] = True

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, mask in placed:
        for hole in _get_enclosed_holes(mask):
            for hr, hc in hole:
                inp[r0 + hr, c0 + hc] = bg

    border = np.concatenate([inp[0, :], inp[-1, :], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
