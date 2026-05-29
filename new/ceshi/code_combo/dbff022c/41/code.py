# Combo variant of dbff022c (index 41)
# combo_spec: D10/1+D12/2
# Name: 调色板条目少于精灵 + T/L形调色板

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np

_ROOT = Path(__file__).resolve().parents[5]
_SOLVERS = _ROOT / "solvers"
if str(_SOLVERS) not in sys.path:
    sys.path.insert(0, str(_SOLVERS))

from arc_utils import components_mask, neighbors4


# Design analysis for combo D10/1 + D12/2
# ----------------------------------------
# spec_variant:
#   concrete_trace:
#     - 设顶边图例是一个 T 形：横臂有两个条目，分别是 3->6、4->2；竖臂只有一个条目 3->8。
#     - 场上有 5 个空心精灵：三个宽框精灵，边框色分别为 3、3、4；两个高框精灵，边框色都为 3。
#     - 原题逻辑会把 palette 看成单一 2xN/Nx2 色条，只允许 `color -> fill` 一维映射，因此无法同时表达
#       `宽的 3 -> 6` 和 `高的 3 -> 8`。
#     - 本融合逻辑必须分两步：
#       1. 先把顶边连通 palette 解析成 T/L 几何，横臂读成“宽精灵映射”，竖臂读成“高精灵映射”。
#       2. 再对每个洞找到所属精灵，按该精灵 bbox 的宽高关系选择分支：宽精灵查横臂，高精灵查竖臂。
#     - 于是两个宽的 3 都填 6，两个高的 3 都填 8，宽的 4 填 2。这里同时体现：
#       1. palette 条目数 3 < 精灵数 5；
#       2. 多个同色精灵共享同一条目；
#       3. 同一边框色在不同分支上可以对应不同填充值。
#   complexity_calibration:
#     - 为了让 D10/1 可见，每个样本固定 5 个精灵而 palette 只有 3 个有效条目，严格保证 `sprites > entries`。
#     - 为了让 D12/2 可见，palette 不是装饰性的拐角，而是语义分支：
#       横臂控制宽精灵，竖臂控制高精灵，并强制至少一个共享 key 同时出现在两支且填充值不同。
#     - 为了让人类能在少量样例中看出规则，宽/高类别只保留两档，且模板都做成明显的长宽不等框，
#       避免出现正方形导致“走哪一支”不明确。
#   hidden_assumption_review:
#     - 原 `_best_palette` 假设 palette 是完整 2xN/Nx2 矩形色条；现在不成立，必须改成先找边界连通 palette，
#       再按 T/L 几何拆出横臂和竖臂。
#     - 原 `_palette_mapping` 假设颜色单独决定填充色；现在不成立，因为同一 key 会在两支上重复出现且映射不同。
#     - 原填充循环只统计相邻颜色即可；现在还必须识别“这个洞属于哪个具体精灵分量”，因为同色精灵可能有宽/高两类。
#     - 新 solver 的前提是：所有非背景且触边的像素都属于 palette；generator 通过让所有精灵离外边框至少 1 格来保证。
#     - 新 solver 还假设每个待填洞只邻接一个单色精灵分量；generator 只生成单层空心框并保持对象之间至少 1 格间距。
#   placement_legality:
#     - palette 固定贴顶边放置，并在其周围留出 1 格缓冲，防止与任何精灵 4 邻接。
#     - 所有精灵都放在两排固定槽位里：上排放宽框、下排放一宽两高，保证不重叠、不越界、不碰外边框，
#       且任意两对象之间至少隔 1 格背景。
#     - 模板都预验证为恰有 1 个内部洞，因此 input->output 只在这些洞上发生变化。
#   observability:
#     - 每个样本都同时包含：
#       1. 至少两个共享“横臂共享 key”的宽精灵；
#       2. 至少两个共享“竖臂共享 key”的高精灵；
#       3. 同一个共享 key 在宽/高两类上产生不同输出色。
#     - 这让解题者能直接看到：不是单看颜色，而是要结合图例分支和精灵形状类别。
#   cleanliness_and_inference:
#     - 画面只保留三层要素：顶边 T/L 图例、内部空心精灵、输出时被填充的洞，没有额外噪声标记。
#     - 颜色总量控制在 6 种语义色以内：2 个 border key、3 个 fill、1 个 connector。规则由显式几何结构触发，
#       不依赖颜色编号大小或任何人类不可视属性。
#   main_assumption_check:
#     - palette 解析阶段假设 row 0 和 row 1 形成一条连续横臂，而 row >= 2 只在两列 stem 上有像素；
#       generator 精确按这个模板构造。
#     - 分量分类阶段假设每个精灵 bbox 满足 `width != height`；模板库只提供明显的宽框或高框。
#     - 查表阶段假设宽精灵颜色一定出现在横臂映射、高精灵颜色一定出现在竖臂映射；generator 只使用这三类条目。


_GRID_H_MIN, _GRID_H_MAX = 13, 13
_GRID_W_MIN, _GRID_W_MAX = 15, 15
_MAX_RETRIES = 400

_BAR_WIDTH = 4
_PALETTE_HEIGHT = 3


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


_WIDE_TEMPLATES = [
    _mask_from_rows(["1111", "1001", "1111"]),
]
_TALL_TEMPLATES = [mask.T.copy() for mask in _WIDE_TEMPLATES]


def _bg_boundary(g: np.ndarray) -> int:
    """Most common color along the 1-cell outer border."""
    border = np.concatenate([g[0, :], g[-1, :], g[1:-1, 0], g[1:-1, -1]])
    vals, cnts = np.unique(border, return_counts=True)
    return int(vals[cnts.argmax()])


def _get_enclosed_holes(mask: np.ndarray) -> list[np.ndarray]:
    """Return enclosed background components inside a boolean sprite mask."""
    h, w = mask.shape
    return [
        pts
        for pts in components_mask(~mask, connectivity=4)
        if not any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts)
    ]


def _validate_templates() -> None:
    for group in (_WIDE_TEMPLATES, _TALL_TEMPLATES):
        for mask in group:
            holes = _get_enclosed_holes(mask)
            assert len(holes) == 1
            h, w = mask.shape
            assert h != w


_validate_templates()


def _boundary_palette_mask(g: np.ndarray, bg: int) -> np.ndarray:
    """Return the unique non-bg component that touches the outer boundary."""
    nonbg = g != bg
    comps = components_mask(nonbg, connectivity=4)
    touching = []
    h, w = g.shape
    for pts in comps:
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            touching.append(pts)
    assert len(touching) == 1, "expected exactly one boundary-touching palette component"
    pal_mask = np.zeros_like(g, dtype=bool)
    for r, c in touching[0]:
        pal_mask[int(r), int(c)] = True
    return pal_mask


def _parse_palette(
    g: np.ndarray, bg: int
) -> tuple[dict[int, int], dict[int, int], np.ndarray]:
    """
    Parse the top-edge T/L palette.

    Returns:
      - wide_mapping: used by wide sprites (horizontal arm)
      - tall_mapping: used by tall sprites (vertical arm)
      - pal_mask
    """
    pal_mask = _boundary_palette_mask(g, bg)
    rows, cols = np.where(pal_mask)
    rows = rows.astype(int)
    cols = cols.astype(int)

    assert int(rows.min()) == 0, "palette must start on the top edge"
    assert int(rows.max()) == _PALETTE_HEIGHT - 1, "palette height mismatch"

    bar_cols = sorted(int(c) for c in np.where(pal_mask[0])[0].tolist())
    assert len(bar_cols) == _BAR_WIDTH, "palette bar width mismatch"
    assert bar_cols == list(range(bar_cols[0], bar_cols[0] + _BAR_WIDTH))
    assert np.array_equal(pal_mask[0], pal_mask[1]), "rows 0 and 1 must share the horizontal bar"

    stem_rows = sorted(int(r) for r in np.where(pal_mask[:, :].any(axis=1))[0].tolist() if r >= 2)
    assert stem_rows == [2], "expected exactly one row below the top bar"

    stem_cols = sorted(int(c) for c in np.where(pal_mask[2])[0].tolist())
    assert len(stem_cols) == 2 and stem_cols[1] == stem_cols[0] + 1, "stem must be 2 contiguous columns"
    assert stem_cols[0] >= bar_cols[0] and stem_cols[1] <= bar_cols[-1], "stem must attach to the bar"

    wide_mapping: dict[int, int] = {}
    for c in bar_cols:
        if c in stem_cols:
            continue
        key = int(g[0, c])
        fill = int(g[1, c])
        assert key != bg and fill != bg
        wide_mapping[key] = fill

    tall_mapping: dict[int, int] = {}
    key = int(g[2, stem_cols[0]])
    fill = int(g[2, stem_cols[1]])
    assert key != bg and fill != bg
    tall_mapping[key] = fill

    assert len(wide_mapping) == 2 and len(tall_mapping) == 1
    return wide_mapping, tall_mapping, pal_mask


def _sprite_components(
    g: np.ndarray, bg: int, pal_mask: np.ndarray
) -> tuple[dict[tuple[int, int], int], list[dict[str, int | tuple[int, int, int, int]]]]:
    """Index each non-palette sprite component by its cells."""
    mask = (g != bg) & (~pal_mask)
    cell_to_comp: dict[tuple[int, int], int] = {}
    infos: list[dict[str, int | tuple[int, int, int, int]]] = []

    for comp_id, pts in enumerate(components_mask(mask, connectivity=4)):
        coords = [(int(r), int(c)) for r, c in pts]
        for rc in coords:
            cell_to_comp[rc] = comp_id
        r_vals = [r for r, _ in coords]
        c_vals = [c for _, c in coords]
        infos.append(
            {
                "color": int(g[coords[0]]),
                "r0": min(r_vals),
                "r1": max(r_vals),
                "c0": min(c_vals),
                "c1": max(c_vals),
            }
        )
    return cell_to_comp, infos


def _fill_enclosed_bg_regions(
    out: np.ndarray,
    bg: int,
    wide_mapping: dict[int, int],
    tall_mapping: dict[int, int],
    pal_mask: np.ndarray,
) -> None:
    """Fill each interior hole by combining border color with sprite aspect class."""
    cell_to_comp, infos = _sprite_components(out, bg, pal_mask)
    h, w = out.shape
    bg_mask = (out == bg) & (~pal_mask)

    for pts in components_mask(bg_mask, connectivity=4):
        if any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in pts):
            continue

        comp_counts: dict[int, int] = {}
        for r, c in pts:
            for rr, cc in neighbors4(int(r), int(c)):
                if 0 <= rr < h and 0 <= cc < w and not pal_mask[rr, cc]:
                    comp_id = cell_to_comp.get((rr, cc))
                    if comp_id is None:
                        continue
                    comp_counts[comp_id] = comp_counts.get(comp_id, 0) + 1

        assert comp_counts, "each hole must touch one sprite component"
        comp_id = max(comp_counts.items(), key=lambda item: item[1])[0]
        info = infos[comp_id]
        color = int(info["color"])
        width = int(info["c1"]) - int(info["c0"]) + 1
        height = int(info["r1"]) - int(info["r0"]) + 1
        mapping = wide_mapping if width > height else tall_mapping
        fill = mapping[color]

        for r, c in pts:
            out[int(r), int(c)] = int(fill)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill hollow sprites by reading a T/L edge legend whose branches encode different sprite classes.

    Input structure:
    - The outer border is mostly background, and the only boundary-touching non-background object is a compact T/L palette on the top edge.
    - The palette's horizontal arm lists mappings for wide hollow sprites, while its vertical stem lists mappings for tall hollow sprites.
    - There are more hollow sprites than palette entries, so several sprites share the same border color and therefore reuse the same branch-specific mapping.

    Rule:
    - Detect the top-edge palette as one connected T/L structure and read the horizontal arm and vertical stem separately.
    - Find every enclosed background hole outside the palette, identify the surrounding sprite component, and classify that sprite as wide or tall from its bounding box.
    - Fill the hole from the mapping on the matching branch: horizontal-arm pairs for wide sprites, vertical-stem pairs for tall sprites.

    Key insight:
    - The palette no longer means "one color gives one fill everywhere". The correct fill is determined by the combination of border color and which geometric branch the sprite belongs to, so identical border colors can map differently on wide versus tall sprites.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_boundary(g)
    wide_mapping, tall_mapping, pal_mask = _parse_palette(g, bg)

    out = g.copy()
    _fill_enclosed_bg_regions(out, bg, wide_mapping, tall_mapping, pal_mask)
    return out


def _choose_palette_colors(
    rng: random.Random,
) -> tuple[int, int, int, int, int, int]:
    """Return shared key, wide-only key, wide fill, wide-only fill, tall fill, connector."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    shared_key, wide_key, wide_fill, wide_only_fill, tall_fill, connector = colors[:6]
    return (
        int(shared_key),
        int(wide_key),
        int(wide_fill),
        int(wide_only_fill),
        int(tall_fill),
        int(connector),
    )


def _place_palette(
    grid: np.ndarray,
    occ: np.ndarray,
    shared_key: int,
    wide_key: int,
    wide_fill: int,
    wide_only_fill: int,
    tall_fill: int,
    connector: int,
    rng: random.Random,
) -> dict[str, int]:
    """Draw one top-edge T or L palette and reserve a padded area around it."""
    h, w = grid.shape
    c0 = rng.randint(0, w - _BAR_WIDTH)
    kind = rng.choice(["T", "L"])
    if kind == "T":
        stem_start = c0 + 1
    else:
        stem_start = c0 if rng.random() < 0.5 else c0 + 2

    entry_cols = [c for c in range(c0, c0 + _BAR_WIDTH) if c not in (stem_start, stem_start + 1)]
    rng.shuffle(entry_cols)
    shared_col, wide_only_col = entry_cols

    grid[0:2, c0 : c0 + _BAR_WIDTH] = connector
    grid[0, shared_col] = shared_key
    grid[1, shared_col] = wide_fill
    grid[0, wide_only_col] = wide_key
    grid[1, wide_only_col] = wide_only_fill
    grid[2, stem_start] = shared_key
    grid[2, stem_start + 1] = tall_fill

    occ[0 : min(h, _PALETTE_HEIGHT + 1), max(0, c0 - 1) : min(w, c0 + _BAR_WIDTH + 1)] = True
    return {"c0": int(c0), "stem_start": int(stem_start), "kind": 0 if kind == "T" else 1}


def _paint_sprite(
    grid: np.ndarray,
    r0: int,
    c0: int,
    mask: np.ndarray,
    color: int,
    fill: int,
) -> list[np.ndarray]:
    """Paint a single-color hollow sprite and return its hole coordinates in local space."""
    holes = _get_enclosed_holes(mask)
    grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]][mask] = color
    for hole in holes:
        for hr, hc in hole:
            grid[r0 + int(hr), c0 + int(hc)] = fill
    return holes


def _slot_layout() -> dict[str, list[tuple[int, int]]]:
    """
    Prepacked legal slots on a 13x15 board.

    Rows 0-2 are the palette, row 3 is a buffer.
    """
    return {
        "wide": [(4, 1), (4, 8), (8, 1)],
        "tall": [(8, 7), (8, 11)],
    }


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    bg = 0
    h = rng.randint(_GRID_H_MIN, _GRID_H_MAX)
    w = rng.randint(_GRID_W_MIN, _GRID_W_MAX)
    grid = np.full((h, w), bg, dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    (
        shared_key,
        wide_key,
        wide_fill,
        wide_only_fill,
        tall_fill,
        connector,
    ) = _choose_palette_colors(rng)

    _place_palette(
        grid,
        occ,
        shared_key,
        wide_key,
        wide_fill,
        wide_only_fill,
        tall_fill,
        connector,
        rng,
    )

    sprite_specs = [
        ("wide", shared_key, wide_fill),
        ("wide", shared_key, wide_fill),
        ("wide", wide_key, wide_only_fill),
        ("tall", shared_key, tall_fill),
        ("tall", shared_key, tall_fill),
    ]
    rng.shuffle(sprite_specs)
    slots = _slot_layout()
    rng.shuffle(slots["wide"])
    rng.shuffle(slots["tall"])

    placed: list[tuple[int, int, np.ndarray, list[np.ndarray]]] = []
    for shape_kind, color, fill in sprite_specs:
        mask = rng.choice(_WIDE_TEMPLATES if shape_kind == "wide" else _TALL_TEMPLATES).copy()
        if not slots[shape_kind]:
            return None
        r0, c0 = slots[shape_kind].pop()
        holes = _paint_sprite(grid, r0, c0, mask, color, fill)
        placed.append((r0, c0, mask, holes))

    output = grid.copy()
    inp = grid.copy()
    for r0, c0, _mask, holes in placed:
        for hole in holes:
            for hr, hc in hole:
                inp[r0 + int(hr), c0 + int(hc)] = bg

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate dbff022c combo D10/1+D12/2 input")
