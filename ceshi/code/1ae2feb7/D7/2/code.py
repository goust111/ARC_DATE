# Variant of 1ae2feb7
# Dimension: D7 — 多步处理顺序 (multi_step)
# Change: ordered_processing——按色条距分隔条的距离从近到远排序处理，先涂的不被后涂的覆盖（no-overwrite规则） → 按色条长度排序处理（长优先）
#
# Design analysis
# spec_variant:
#   unchanged_visual_core:
#     - 仍然有一列竖直分隔条，把网格分成 bar_side 和 ext_side。
#     - 活动行在 bar_side 上放 1-2 条水平纯色色条；ext_side 在输入中为空，在输出中被周期性打点。
#     - no-overwrite 规则保留：某个 ext_side 像素一旦先被涂上颜色，后处理的色条不能覆盖它。
#   changed_step_order:
#     - 原题每行按“离分隔条近 -> 远”处理色条。
#     - 本变体改为按“色条长度长 -> 短”处理；长度相同仅作确定性 tie-break，不作为可见规则的一部分。
#   concrete_trace:
#     - 例：同一行在分隔条左侧有 near=(长度2, 蓝色) 和 far=(长度5, 红色)。
#     - 原题：near 先涂，ext 第 1 格先变蓝；far 再尝试涂时不能覆盖，所以第 1 格仍蓝。
#     - 本题：far 更长，先涂，ext 第 1 格先变红；near 后到时不能覆盖，所以第 1 格改为红。
#     - 二者都保留“每隔自身长度打点”的几何规律，只是冲突归属从“位置近者赢”变为“长度长者赢”。
#   complexity_calibration:
#     - 如果所有双条行都满足 near 比 far 更长，那么新规则会经常与原规则同结果，视觉上退化。
#     - 因此生成器强制至少两条双条行：一条 near 更长，一条 far 更长。这样同一题里会同时出现
#       “近条获胜”和“远条获胜”的首格证据，排除“总是近条优先/总是远条优先”的误读。
#     - 仍允许少量单条行，保留原题整体观感，不把任务简化成只看一行。
#   implicit_assumption_audit_main:
#     - `_find_sep_col` 依赖“分隔条列的非背景像素同色且覆盖至少 H-1 个格”。变体未改这点，生成器继续保证。
#     - `bars_on_left = left_mass >= right_mass` 依赖“输入时 ext_side 全空，所有色条在同侧”。变体未改，生成器继续保证。
#     - `_iter_runs_1d` 会把每个同色连续段都当成候选色条；因此生成器必须继续保证一行最多 2 条、且它们水平连续放置。
#     - 原版 `bars.sort(key=(dist,...))` 依赖“顺序由距离决定”；这里必须替换为 `(-length, dist, ...)`，
#       否则会悄悄回到旧规则。
#     - no-overwrite 逻辑 `if out[r, cc] == bg` 仍然正确；变体只换先后顺序，不改变覆盖语义。
#     - 同长度双条若颜色不同，则 tie-break 会影响输出，但题面无法从可见结构推断该隐藏顺序。
#       为避免这个新隐式假设，生成器禁止同一行两条长度相等。
#   placement_legality:
#     - bar_side 宽度仍受限，双条行总长度不超过 bar_side 宽度；证据行不留 separator gap，避免容量不足。
#     - 活动行之间继续至少隔 1 个空行，避免不同模式在竖直方向粘连。
#     - 分隔条偶尔允许一个非活动行缺口，仍不破坏检测。
#   visual_observability:
#     - 冲突最稳定地发生在 ext_side 紧邻分隔条的第 1 格，因为所有射线都从这里开始。
#     - 因此只要双条行长度不等，首格颜色就直接暴露“谁先处理”；再结合后续周期点位，可肉眼推断规则。
#     - 生成器要求每题同时出现 near>far 和 far>near 两种长度关系，让“按长度排序”在 2-3 个样例内可见。
#   cleanliness:
#     - 变化仍局限在每个活动行自己的 ext_side；行间互不覆盖，视觉上保持原题整洁。
#   color_inference:
#     - 颜色只作为被传播的标签，不参与排序，也不编码任何不可见元信息。
#   generator_guards:
#     - `main(inp) == out_variant` 验证求解器正确。
#     - 额外要求 `out_variant != out_original_distance_order`，避免样本退化成原题行为。

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


def _find_sep_col(g: np.ndarray, bg: int):
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _iter_runs_1d(row: np.ndarray, c_lo: int, c_hi: int, bg: int):
    c = c_lo
    while c < c_hi:
        if row[c] == bg:
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < c_hi and row[c] == col:
            c += 1
        yield c0, c - 1, col


def _collect_bars(
    g: np.ndarray,
    sep_col: int,
    bg: int,
    bars_on_left: bool,
) -> list[tuple[int, int, int, int, int, int]]:
    bars: list[tuple[int, int, int, int, int, int]] = []
    h, w = g.shape
    for r in range(h):
        if bars_on_left:
            for c0, c1, col in _iter_runs_1d(g[r, :], 0, sep_col, bg=bg):
                length = c1 - c0 + 1
                dist = (sep_col - 1) - c1
                bars.append((dist, r, c0, c1, length, col))
        else:
            for c0, c1, col in _iter_runs_1d(g[r, :], sep_col + 1, w, bg=bg):
                length = c1 - c0 + 1
                dist = c0 - (sep_col + 1)
                bars.append((dist, r, c0, c1, length, col))
    return bars


def _paint_from_bars(
    out: np.ndarray,
    bars: list[tuple[int, int, int, int, int, int]],
    sep_col: int,
    bars_on_left: bool,
    bg: int,
) -> None:
    _, w = out.shape
    for _dist, r, _c0, _c1, length, col in bars:
        if length <= 0:
            continue
        if bars_on_left:
            for cc in range(sep_col + 1, w, length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col
        else:
            for cc in range(sep_col - 1, -1, -length):
                if int(out[r, cc]) == bg:
                    out[r, cc] = col


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    _, w = g.shape

    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    out = g.copy()

    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < w else 0
    bars_on_left = left_mass >= right_mass

    bars = _collect_bars(g, sep_col, bg, bars_on_left)

    # Variant rule: longer bars claim the periodic extension first.
    # Distance remains only as a deterministic tie-break, not the primary ordering cue.
    bars.sort(key=lambda t: (-t[4], t[0], t[1], t[2]))
    _paint_from_bars(out, bars, sep_col, bars_on_left, bg)
    return out


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

BAR_SIDE_MIN, BAR_SIDE_MAX = 3, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 5, 14

PALETTE_MIN, PALETTE_MAX = 5, 8

ACTIVE_MIN, ACTIVE_MAX = 3, 5
P_TWO_BARS_EXTRA = 0.55
P_SEP_GAP = 0.12
P_SEP_COLUMN_GAP = 0.2

FORCED_RELATIONS = ("near_longer", "far_longer")
MAX_ATTEMPTS = 400


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    bar_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(bar_pool)) + 1))
    palette = list(map(int, rng.choice(bar_pool, size=size, replace=False)))
    return sep_color, palette


def _pick_unequal_lengths(
    rng: np.random.Generator,
    capacity: int,
    relation: str | None = None,
) -> tuple[int, int] | None:
    pairs: list[tuple[int, int]] = []
    for near_len in range(1, capacity):
        max_far = capacity - near_len
        for far_len in range(1, max_far + 1):
            if near_len == far_len:
                continue
            if relation == "near_longer" and near_len <= far_len:
                continue
            if relation == "far_longer" and far_len <= near_len:
                continue
            pairs.append((near_len, far_len))
    if not pairs:
        return None
    idx = int(rng.integers(0, len(pairs)))
    return pairs[idx]


def _sample_row_bars(
    rng: np.random.Generator,
    palette: list[int],
    primary: int,
    bar_side_len: int,
    relation: str | None,
) -> tuple[list[tuple[int, int]], int] | None:
    if relation is not None:
        sep_gap = 0
        lengths = _pick_unequal_lengths(rng, bar_side_len, relation=relation)
        if lengths is None:
            return None
        near_len, far_len = lengths
        other_choices = [c for c in palette if c != primary]
        far_color = int(rng.choice(other_choices))
        return [(near_len, primary), (far_len, far_color)], sep_gap

    sep_gap = 1 if rng.random() < P_SEP_GAP else 0
    capacity = bar_side_len - sep_gap
    if capacity <= 0:
        return None

    if capacity >= 3 and rng.random() < P_TWO_BARS_EXTRA:
        lengths = _pick_unequal_lengths(rng, capacity, relation=None)
        if lengths is None:
            return None
        near_len, far_len = lengths
        other_choices = [c for c in palette if c != primary]
        far_color = int(rng.choice(other_choices))
        return [(near_len, primary), (far_len, far_color)], sep_gap

    length = int(rng.integers(1, capacity + 1))
    return [(length, primary)], sep_gap


def _place_bars(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    sep_gap: int,
) -> None:
    if bars_on_left:
        cursor = sep_col - sep_gap
        for length, color in bars:
            out[row, cursor - length : cursor] = color
            cursor -= length
    else:
        cursor = sep_col + 1 + sep_gap
        for length, color in bars:
            out[row, cursor : cursor + length] = color
            cursor += length


def _ordered_row_bars(
    bars: list[tuple[int, int]],
    order: str,
) -> list[tuple[int, int]]:
    indexed = list(enumerate(bars))
    if order == "length_desc":
        indexed.sort(key=lambda t: (-t[1][0], t[0]))
    elif order == "near_first":
        indexed.sort(key=lambda t: t[0])
    else:
        raise ValueError(order)
    return [bar for _, bar in indexed]


def _paint_extension(
    out: np.ndarray,
    row: int,
    sep_col: int,
    bars_on_left: bool,
    bars: list[tuple[int, int]],
    order: str,
) -> None:
    w = out.shape[1]
    for length, color in _ordered_row_bars(bars, order=order):
        if bars_on_left:
            positions = range(sep_col + 1, w, length)
        else:
            positions = range(sep_col - 1, -1, -length)
        for cc in positions:
            if out[row, cc] == 0:
                out[row, cc] = color


def _render_output(
    base: np.ndarray,
    sep_col: int,
    bars_on_left: bool,
    row_bars_map: dict[int, list[tuple[int, int]]],
    order: str,
) -> np.ndarray:
    out = base.copy()
    for row, bars in row_bars_map.items():
        _paint_extension(out, row, sep_col, bars_on_left, bars, order=order)
    return out


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(BAR_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(BAR_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    bar_side_len = int(rng.integers(lo, hi + 1))

    bars_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = bar_side_len if bars_on_left else (w - 1 - bar_side_len)

    sep_color, palette = _sample_palette(rng)

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    min_active = max(ACTIVE_MIN, len(FORCED_RELATIONS))
    if max_active < min_active:
        return None
    active_count = int(rng.integers(min_active, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)
    active_set = set(active_rows)
    inactive_rows = [r for r in range(h) if r not in active_set]

    base = np.zeros((h, w), dtype=int)
    base[:, sep_col] = sep_color
    if inactive_rows and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = inactive_rows[int(rng.integers(0, len(inactive_rows)))]
        base[gap_row, sep_col] = 0

    row_primary = list(map(int, rng.permutation(palette)[:active_count]))
    forced_rows = active_rows[: len(FORCED_RELATIONS)]
    forced_relations = list(FORCED_RELATIONS)
    rng.shuffle(forced_relations)
    forced_map = dict(zip(forced_rows, forced_relations))

    row_bars_map: dict[int, list[tuple[int, int]]] = {}
    for idx, row in enumerate(active_rows):
        relation = forced_map.get(row)
        sampled = _sample_row_bars(rng, palette, row_primary[idx], bar_side_len, relation=relation)
        if sampled is None:
            return None
        bars, sep_gap = sampled
        row_bars_map[row] = bars
        _place_bars(base, row, sep_col, bars_on_left, bars, sep_gap)

    out_variant = _render_output(base, sep_col, bars_on_left, row_bars_map, order="length_desc")

    inp = base.copy()
    if bars_on_left:
        inp[:, sep_col + 1 :] = 0
    else:
        inp[:, :sep_col] = 0

    out_original = _render_output(inp.copy(), sep_col, bars_on_left, row_bars_map, order="near_first")

    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.array_equal(out_variant, out_original):
        return None
    if not np.array_equal(main(inp), out_variant):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
