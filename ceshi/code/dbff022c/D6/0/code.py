# Variant of dbff022c
# Dimension: D6 — 网格输出形状 (grid_shape)
# Change: 输出网格与输入网格完全相同大小，调色板区域保持不变，仅精灵内部被修改... → 裁剪移除调色板区域

import random

import numpy as np

from arc_utils import components_mask, neighbors4


# Design analysis for the D6 variant
# ----------------------------------
# yaml:
#   concretization:
#     variant_rule:
#       - palette 仍然是贴着某一侧边界的 2xN / Nx2 图例
#       - solver 仍先读取 border->fill 映射并填充空心精灵内部
#       - 新变化只发生在最后一步: 整条承载调色板的 2-cell 边带会被裁掉
#     traced_example:
#       - old: 顶边 palette 放在 rows 0..1; 精灵区在 rows 2..H-1; 输出仍保留 rows 0..1
#       - new: 同样先在完整网格上填洞, 然后输出改为 out[2:, :]
#       - difference: palette 不再出现在输出里, 输出高度减少 2
#   complexity_calibration:
#     - 不引入新的排序、颜色运算或多段式分支, 只增加一个与 palette side 对应的裁剪动作
#     - 为了让变化稳定可见, generator 保证所有精灵都落在保留区域内, 被裁掉的一侧只承担指令作用
#   implicit_assumptions_audit:
#     - _bg_boundary 依赖外边界大多是 bg; generator 让精灵远离完整输入的外边界, 仅 palette 占用被裁边带
#     - _best_palette 依赖 palette 是唯一可信的边缘 2-cell 图例; generator 不允许其他非 bg 物体触碰外边界
#     - _fill_enclosed_bg_regions 在裁剪前工作; 因此洞是否封闭必须相对于完整输入成立
#     - 新增的 _crop_palette_strip 假设 palette 总在整条 2-row/2-col 边带内; generator 明确按此构造
#   placement_legality:
#     - 先在“裁剪后画布”上放置精灵, 这样它们天然不会进入将被裁掉的边带
#     - 精灵之间仍保持至少 1 个 bg 间隔, 且离裁剪后画布边界至少 1 格, 便于 solver 稳定识别 bg 边界
#     - 再把该画布嵌回完整输入, 并仅在被裁边带放置 palette
#   visibility:
#     - 每个样例都会在一条轴上缩短 2 格, 人眼可直接看到“输入有 palette, 输出没有 palette”
#     - 精灵填色逻辑保持原样, 因此解题者能看出 palette 既决定颜色又决定裁掉哪一侧
#   generator_guardrails:
#     - 输出永远等于“完整填色结果裁掉 palette 边带”, 不会退化成原题的同尺寸输出
#     - 测试时可量化 shape delta = input.shape - output.shape, 必有且仅有一个维度差为 2


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
        outside the palette region; this prevents accidentally absorbing parts of sprites.
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
            v = int(g[int(r), int(c)])
            if v == bg:
                continue
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal[rr, cc]:
                    if int(g[rr, cc]) == v:
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
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("top", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(c1 - c0), nonbg)
                cand = (score, "top", c0, c1)
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
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("bottom", c0, c1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(c1 - c0), nonbg)
                cand = (score, "bottom", c0, c1)
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
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("left", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(r1 - r0), nonbg)
                cand = (score, "left", r0, r1)
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
                d_outer = set(map(int, np.unique(seg_outer)))
                d_inner = set(map(int, np.unique(seg_inner)))
                d_outer.discard(bg)
                d_inner.discard(bg)
                if len(d_outer) < 2 or len(d_inner) < 2:
                    continue
                union = d_outer | d_inner
                if len(union) < 3:
                    continue
                border_keys = [int(x) for x in seg_outer.tolist() if int(x) != bg]
                if not border_keys or len(border_keys) != len(set(border_keys)):
                    continue
                if not isolated("right", r0, r1):
                    continue
                nonbg = int(np.sum(seg_outer != bg) + np.sum(seg_inner != bg))
                score = (len(set(border_keys)), len(union), -(r1 - r0), nonbg)
                cand = (score, "right", r0, r1)
                if best is None or cand[0] > best[0]:
                    best = cand

    consider_top()
    consider_bottom()
    consider_left()
    consider_right()

    assert best is not None, "_best_palette must find a valid palette region"
    _, side, start, end = best
    return side, int(start), int(end)


def _palette_mapping(g: np.ndarray, bg: int) -> tuple[dict[int, int], np.ndarray, tuple[str, int, int]]:
    """Returns (mapping border->fill, palette_mask, palette_info)."""
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
        f = int(f)
        if b == bg:
            continue
        mapping[b] = f

    return mapping, pal_mask, (side, start, end)


def _fill_enclosed_bg_regions(out: np.ndarray, bg: int, mapping: dict[int, int], pal_mask: np.ndarray) -> None:
    """
    Find background regions (excluding the palette cells) that do not touch the grid boundary,
    and fill each region using the mapped color of its dominant adjacent border.
    """
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)
    for pts in components_mask(bg_mask, connectivity=4):
        touches_boundary = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
        if touches_boundary:
            continue

        adj_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w:
                    if pal_mask[rr, cc]:
                        continue
                    v = int(out[rr, cc])
                    if v == bg:
                        continue
                    adj_counts[v] = adj_counts.get(v, 0) + 1
        if not adj_counts:
            continue

        border = None
        best = -1
        for col, cnt in adj_counts.items():
            if col not in mapping:
                continue
            if cnt > best:
                best = cnt
                border = col
        if border is None:
            continue

        fill = int(mapping[border])
        for r, c in pts:
            out[int(r), int(c)] = fill


def _crop_palette_strip(out: np.ndarray, side: str) -> np.ndarray:
    """Remove the full 2-cell edge strip that contains the palette legend."""
    if side == "top":
        return out[2:, :].copy()
    if side == "bottom":
        return out[:-2, :].copy()
    if side == "left":
        return out[:, 2:].copy()
    return out[:, :-2].copy()


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites from an edge palette, then crop away the palette strip.

    Input structure:
    - The background dominates the outer border, while one edge contains a 2xN or Nx2 palette legend.
    - Hollow single-color sprites occupy the remaining field; some border colors are keyed by the legend and one sprite color may be a distractor.

    Rule:
    - Detect the edge legend, read it as a border-color to fill-color mapping, and fill each enclosed background hole whose surrounding border color appears in that mapping.
    - After the fills are complete, remove the entire 2-row or 2-column edge strip that carries the legend.

    Key insight: The palette is input-only control information. Use it to decide the interior fills on the full grid, then discard that whole side from the final output.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)

    mapping, pal_mask, pal_info = _palette_mapping(g, bg)
    assert mapping, "main: palette mapping must be non-empty"

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, mapping, pal_mask)

    side, _start, _end = pal_info
    return _crop_palette_strip(out, side)


# --------------- generator (D6 variant) ---------------

# Full input dimensions. One axis loses exactly 2 cells after cropping.
_GRID_H_MIN, _GRID_H_MAX = 7, 13
_GRID_W_MIN, _GRID_W_MAX = 12, 15

# Palette
_PALETTE_N_MIN, _PALETTE_N_MAX = 2, 5

# Sprite geometry
_SPRITE_H_MAX = 7
_SPRITE_W_MAX = 6
_SPRITE_DIM_SUM_CAP = 10
_SPRITE_CELL_MAX = 16
_FILLED_SPRITE_MAX = 3

# Sprite mutation probabilities
_P_HORIZ_WALL = 0.3
_P_VERT_WALL = 0.3
_P_BORDER_REMOVAL = 0.4
_P_DIAMOND = 0.2

# Generation
_MAX_RETRIES = 400


def _get_enclosed_holes(sprite: np.ndarray) -> list[np.ndarray]:
    """Return bg components that stay fully inside the sprite bbox."""
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
    """Create a hollow sprite with at least one enclosed hole."""
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
    """Pick palette pairs, a distractor border color, and which keys appear on filled sprites."""
    all_c = list(range(1, 10))
    rng.shuffle(all_c)
    border_keys = all_c[:n_pairs]
    rng.shuffle(all_c)
    fill_values = all_c[:n_pairs]
    dist_cands = [c for c in range(1, 10) if c not in border_keys]
    distractor_color = rng.choice(dist_cands)
    mapping = dict(zip(border_keys, fill_values))
    filled_keys = rng.sample(border_keys, num_filled)
    return border_keys, fill_values, distractor_color, mapping, filled_keys


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


def _place_palette(grid: np.ndarray, pal: np.ndarray, side: str, rng: random.Random) -> None:
    """Place the palette somewhere inside the removable 2-cell edge strip."""
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


def _cropped_shape(side: str, h: int, w: int) -> tuple[int, int]:
    """Shape after removing the palette strip."""
    if side in ("top", "bottom"):
        return h - 2, w
    return h, w - 2


def _embed_cropped(full: np.ndarray, crop: np.ndarray, side: str) -> None:
    """Paste the cropped sprite field back into the full input/output canvas."""
    if side == "top":
        full[2:, :] = crop
    elif side == "bottom":
        full[:-2, :] = crop
    elif side == "left":
        full[:, 2:] = crop
    else:
        full[:, :-2] = crop


def _find_valid_positions(h: int, w: int, occ: np.ndarray, mask: np.ndarray) -> list[tuple[int, int]]:
    """Find placements with 1-cell margin from the cropped canvas border and other sprites."""
    sh, sw = mask.shape
    positions = []
    for r0 in range(1, h - sh):
        for c0 in range(1, w - sw):
            if occ[max(0, r0 - 1) : min(h, r0 + sh + 1), max(0, c0 - 1) : min(w, c0 + sw + 1)].any():
                continue
            positions.append((r0, c0))
    return positions


def _paint_sprite(
    grid: np.ndarray, r0: int, c0: int, mask: np.ndarray, color: int, fill: int | None, holes: list[np.ndarray]
) -> None:
    """Render a sprite onto a cropped canvas."""
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    if fill is not None:
        for hpts in holes:
            for hr, hc in hpts:
                grid[r0 + int(hr), c0 + int(hc)] = fill


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0

    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    side = rng.choice(["top", "bottom", "left", "right"])
    crop_h, crop_w = _cropped_shape(side, h, w)

    if crop_h < 5 or crop_w < 8:
        return None

    max_n = h if side in ("left", "right") else w
    n_pairs = rng.randint(_PALETTE_N_MIN, min(_PALETTE_N_MAX, max_n))

    max_sh = min(_SPRITE_H_MAX, crop_h - 2)
    max_sw = min(_SPRITE_W_MAX, crop_w - 2)
    if max_sh < 3 or max_sw < 3:
        return None

    num_filled = rng.randint(1, min(_FILLED_SPRITE_MAX, n_pairs))
    total_sprites = num_filled + 1

    shapes: list[np.ndarray] = []
    for _ in range(total_sprites):
        sh = rng.randint(3, max_sh)
        sw = rng.randint(3, max_sw)
        if sh + sw > _SPRITE_DIM_SUM_CAP:
            sw = max(3, _SPRITE_DIM_SUM_CAP - sh)
        if sw > max_sw:
            sw = max_sw
        if sh == 3 and sw == 3 and rng.random() < _P_DIAMOND:
            mask = _make_diamond()
        else:
            mask = _make_hollow_sprite(sh, sw, rng)
        if mask is None:
            return None
        shapes.append(mask)

    border_keys, fill_values, distractor_color, mapping, filled_keys = _choose_colors(rng, n_pairs, num_filled)
    pal = _build_palette(side, border_keys, fill_values)

    crop_out = np.full((crop_h, crop_w), bg, dtype=int)
    occ = np.zeros((crop_h, crop_w), dtype=bool)

    infos: list[tuple[np.ndarray, int, int | None]] = []
    for i, mask in enumerate(shapes):
        if i < num_filled:
            key = filled_keys[i]
            infos.append((mask, key, mapping[key]))
        else:
            infos.append((mask, distractor_color, None))

    placed: list[tuple[int, int, np.ndarray, int | None]] = []
    for mask, color, fill in infos:
        holes = _get_enclosed_holes(mask)
        positions = _find_valid_positions(crop_h, crop_w, occ, mask)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        _paint_sprite(crop_out, r0, c0, mask, color, fill, holes)
        placed.append((r0, c0, mask, fill))
        sh, sw = mask.shape
        occ[max(0, r0 - 1) : min(crop_h, r0 + sh + 1), max(0, c0 - 1) : min(crop_w, c0 + sw + 1)] = True

    full_out = np.full((h, w), bg, dtype=int)
    _embed_cropped(full_out, crop_out, side)
    _place_palette(full_out, pal, side, rng)

    inp = full_out.copy()
    for r0, c0, mask, fill in placed:
        if fill is None:
            continue
        for hpts in _get_enclosed_holes(mask):
            for hr, hc in hpts:
                rr = r0 + int(hr)
                cc = c0 + int(hc)
                if side == "top":
                    rr += 2
                elif side == "left":
                    cc += 2
                inp[rr, cc] = bg

    border = np.concatenate([inp[0], inp[-1], inp[1:-1, 0], inp[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    if int(vals[cnts.argmax()]) != bg:
        return None

    try:
        if np.array_equal(main(inp), crop_out):
            return inp
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        r = _gen_attempt()
        if r is not None:
            return r
    raise RuntimeError(f"Failed to generate input after {_MAX_RETRIES} attempts")
