# Variant of 97d7923e
# Dimension: D4 — 源对象：垂直条的结构 (src)
# Change: 垂直条结构为cap-fill-cap：顶部和底部各一个帽色像素，中间为单一填充色的连续区域，所有条底部对齐在最后一行... → 条的填充区域为交替双色条纹

import itertools

import numpy as np

from arc_utils import get_bg, neighbors4, paste


# Design analysis for the D4 variant
# analysis:
#   concretization:
#     legend:
#       - 仍位于左上角，若某列顶部有颜色 K 的竖直块，高度 k 就表示“在 K 帽色这一组里选第 k 长的条”。
#     bars:
#       - 每根条仍是一列像素，底部对齐在最后一行，顶部和底部端点仍是帽色 K。
#       - 变化只发生在内部填充区：原题是 [F,F,F,...]，现在必须是 [A,B,A,B,...] 这类严格交替双色条纹。
#       - 条纹色 A/B 都不是背景，也都不等于帽色 K；同一组条可共享同一对条纹色，但解题逻辑不能依赖它们。
#     output:
#       - 被选中的那根条，其整个内部区依然被重涂成帽色 K，因此输出会把“条纹”抹平成纯色。
#   behavior_trace:
#     original_local_flow:
#       - `_vertical_sandwiches()` 取帽色 K 后，在每一列找两个相邻的 K，要求中间整段全等于同一个 fill 色。
#       - 因此像 [K,2,3,2,3,K] 这样的列会被拒绝，因为 `np.all(interior == fill)` 不成立。
#     variant_local_flow:
#       - 新检测器对同样的列应接受，只要中间不存在背景/帽色，且相邻像素总是不同、全段恰好只用两种颜色。
#       - 排序依据仍然是内部长度，而不是内部颜色；输出仍把整段改成 K。
#     key_difference:
#       - 旧逻辑把“内部单色”当成对象定义的一部分；新逻辑把它降级为无关细节，只保留“帽色 + 竖直长度”作为语义核心。
#   complexity_calibration:
#     - 为了让 D4 肉眼可见，生成器强制每根条的内部长度至少为 2，这样每根条都真实出现双色条纹，不会退化成单像素单色填充。
#     - 组内长度仍互异，避免排名歧义；组数、每组条数、图例深度维持原题档位，避免引入额外推理维度。
#     - 条纹色允许跨组复用，避免把“某组有专属填充色”错误固化为规则，同时也避免 4 组时颜色数超出 ARC 调色板。
#   implicit_assumption_audit:
#     - 原 `_vertical_sandwiches` 中 `fill = g[a+1,c]` + `np.all(...)` 依赖“整段单色”，在变体中失效，必须改为条纹合法性检查。
#     - 原 `main()` 最后直接 `out[top+1:bot,col] = cap` 并不依赖单色假设；在变体中仍然正确，因为目标仍是整段重涂。
#     - `_legend_entries()` 只依赖图例位于左上且为竖直连通块，不依赖条纹结构，可保持不变。
#   placement_legality:
#     - 图高、宽度、列间距、底部对齐规则沿用原题，仍保证条之间不重叠。
#     - 共享图例列的那根条仍需在图例下方留至少 1 行背景；因此该列可用的最大内部长度仍是 `MAX_FILL_LEN - rank`。
#     - 所有内部像素都非背景、非帽色，确保求解器不会把图例尾部到条顶之间的背景缝误识别为条内部。
#   observability:
#     - 输入中每根条都显式呈现 ABAB 条纹，输出中只有被选中的条纹列会被抹成纯帽色，因此“条纹不影响选中规则”是可观察的。
#     - 排序只看长度，长度是几何可见特征；不使用颜色 id、出现次数等不可见属性。
#   visual_cleanliness:
#     - 仍然是一列一对象，图例留在左上，条之间至少隔 1 列背景，避免条纹和图例相互缠绕。
#     - 每组只共享帽色，不额外叠加标记，视觉上仍接近原题，只是内部更花。
#   color_change_inference:
#     - 颜色变化仍只有一种：被选中的条内部变成对应帽色。
#     - 条纹色数量受限于 ARC 调色板，并由帽色位置/端点提供结构线索，不需要从无提示的大规模颜色置换中猜映射。


def _is_alternating_two_color(interior: np.ndarray, cap: int, bg: int) -> bool:
    """Return True iff the 1-D interior is a strict two-color alternation."""
    if interior.size < 2:
        return False
    vals = [int(v) for v in interior]
    if any(v == cap or v == bg for v in vals):
        return False
    if len(set(vals)) != 2:
        return False
    return all(vals[i] != vals[i - 1] for i in range(1, len(vals)))


def _vertical_striped_bars(g: np.ndarray, cap: int) -> list[tuple[int, int, int]]:
    """
    Find vertical bar candidates (top_r, bottom_r, col) where:
      - g[top_r, col] == cap and g[bottom_r, col] == cap
      - (top_r, bottom_r) are consecutive occurrences of cap in that column
      - interior is a non-empty strict alternation of exactly two non-bg, non-cap colors
      - length >= 2 so the stripe structure is visually observable
    """
    _h, w = g.shape
    bg = get_bg(g)
    segs: list[tuple[int, int, int]] = []
    for c in range(w):
        rows = [int(r) for r in np.flatnonzero(g[:, c] == cap)]
        if len(rows) < 2:
            continue
        for a, b in zip(rows, rows[1:]):
            if b <= a + 2:
                continue
            interior = g[a + 1 : b, c]
            if _is_alternating_two_color(interior, cap, bg):
                segs.append((a, b, c))
    return segs


def _legend_entries(g: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    """
    Legend is in the top-left: a cluster of non-bg pixels on the top row near the left edge,
    allowing small gaps (typically 1 bg column) between entries.

    Returns entries as (x_pos, legend_color, block_count), sorted by x_pos.
    block_count is the vertical height of the 4-connected component of that color that touches
    the top row within the legend cluster.
    """
    h, w = g.shape
    top_pos = [int(c) for c in np.flatnonzero(g[0, :] != bg)]
    assert top_pos, "legend row must be non-empty"
    top_pos.sort()

    cols = [top_pos[0]]
    for c in top_pos[1:]:
        if c - cols[-1] <= 2:
            cols.append(c)
        else:
            break
    legend_max_c = max(cols)

    entries: list[tuple[int, int, int]] = []
    seen = np.zeros((h, w), dtype=bool)
    for c in cols:
        col = int(g[0, c])
        if col == bg or seen[0, c]:
            continue
        st = [(0, c)]
        seen[0, c] = True
        pix = []
        while st:
            r, cc = st.pop()
            pix.append((r, cc))
            for rr, ccc in neighbors4(r, cc):
                if 0 <= rr < h and 0 <= ccc <= legend_max_c and (not seen[rr, ccc]) and int(g[rr, ccc]) == col:
                    seen[rr, ccc] = True
                    st.append((rr, ccc))
        x_pos = min(cc for _, cc in pix)
        rs = [r for r, _ in pix]
        block_count = max(rs) - min(rs) + 1
        entries.append((x_pos, col, int(block_count)))

    entries.sort(key=lambda t: t[0])
    return entries


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repaint ranked vertical bars chosen from cap-matched striped columns.

    Input structure:
    - bg is the dominant background. The top-left legend is still made of one-column color stacks,
      where a stack of color K and height k encodes rank k.
    - The main objects are bottom-aligned vertical bars with cap color K at both ends, while the
      interior is an alternating two-color stripe rather than a solid fill.

    Rule:
    - For each legend entry (K, k), collect every vertical bar whose two endpoints are color K.
    - Sort that cap-color group by interior length from longest to shortest.
    - Recolor the interior of the rank-k bar to K, ignoring whatever two stripe colors were inside it.

    Key insight: The legend still indexes a length ranking by cap color; the striped interior is
    deliberately irrelevant noise, so only the capped span and its length matter.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = g.copy()

    legend = _legend_entries(g, bg)
    assert legend, "legend must be non-empty"

    for _x, cap, k in legend:
        segs = _vertical_striped_bars(g, cap)
        assert segs, f"legend cap {cap} must have matching striped bars"
        segs.sort(key=lambda s: (-(s[1] - s[0] - 1), s[0], s[2]))
        assert 1 <= k <= len(segs), f"legend rank {k} out of range for {len(segs)} bars"
        top_r, bot_r, col = segs[k - 1]
        out[top_r + 1 : bot_r, col] = cap

    return out


# --------------- generator (O2) ---------------

# --- Configuration ---
GRID_H = 10
WIDTH_MIN = 10
WIDTH_MAX = 20
MAX_GROUPS = 4
MAX_TOTAL_BARS = 10
BARS_PER_GROUP = (2, 3)
MIN_FILL_LEN = 2
MAX_FILL_LEN = 7
COLOR_RANGE = (1, 10)
COL_SPACING = 2
MAX_RETRIES = 300

_BOTTOM_ROW = GRID_H - 1
_MAX_CAP_ROW = GRID_H - 2

_FEASIBLE_COUNTS = {
    ng: [c for c in itertools.product(BARS_PER_GROUP, repeat=ng) if sum(c) <= MAX_TOTAL_BARS]
    for ng in range(1, MAX_GROUPS + 1)
}


def _build_bar(cap: int, stripe_a: int, stripe_b: int, fill_len: int, phase: int) -> np.ndarray:
    """Return a (fill_len+2, 1) column sprite: cap / alternating stripe / cap."""
    bar = np.full((fill_len + 2, 1), cap, dtype=int)
    for idx in range(fill_len):
        bar[idx + 1, 0] = stripe_a if (idx + phase) % 2 == 0 else stripe_b
    return bar


def _assign_columns(rng, n_groups: int, counts: list[int], width: int) -> list[list[int]]:
    """Distribute bar columns across groups with random inter-group spacing."""
    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    gap_slots = total_bars - n_groups + 1
    extra_width = width - min_width
    extra_parts = rng.multinomial(extra_width, np.full(gap_slots, 1.0 / gap_slots))

    inter_gap_bonus = np.zeros(total_bars - 1, dtype=int)
    if total_bars > n_groups:
        inter_gap_bonus[n_groups - 1 :] = extra_parts[:-1]

    occupied = [0]
    for idx in range(total_bars - 1):
        occupied.append(occupied[-1] + COL_SPACING + int(inter_gap_bonus[idx]))

    legend_cols = occupied[:n_groups]
    extra_cols = occupied[n_groups:]

    group_columns = [[legend_cols[i]] for i in range(n_groups)]
    owners = [i for i, count in enumerate(counts) for _ in range(count - 1)]
    rng.shuffle(owners)
    for col, owner in zip(extra_cols, owners):
        group_columns[owner].append(col)

    return group_columns


def _choose_fill_lengths(rng, count: int, rank: int) -> list[int]:
    """Pick distinct striped interior lengths for one group."""
    max_legend_len = MAX_FILL_LEN - rank
    assert MIN_FILL_LEN <= max_legend_len
    shared_len = int(rng.integers(MIN_FILL_LEN, max_legend_len + 1))
    pool = [l for l in range(MIN_FILL_LEN, MAX_FILL_LEN + 1) if l != shared_len]
    rng.shuffle(pool)
    return [shared_len] + pool[: count - 1]


def _expected_output(
    inp: np.ndarray,
    caps: list[int],
    group_columns: list[list[int]],
    group_lengths: list[list[int]],
    ranks: list[int],
) -> np.ndarray:
    """Compute the expected output by recoloring the rank-k bar's full interior to cap."""
    out = inp.copy()
    for cap, cols, lengths, rank in zip(caps, group_columns, group_lengths, ranks):
        ranked = sorted(zip(cols, lengths), key=lambda item: -item[1])
        target_col, target_len = ranked[rank - 1]
        top_r = _MAX_CAP_ROW - target_len
        out[top_r + 1 : _BOTTOM_ROW, target_col] = cap
    return out


def _gen_attempt() -> np.ndarray | None:
    """Try once to build a valid striped-bar grid. Returns None on oracle mismatch."""
    rng = np.random.default_rng()

    n_groups = int(rng.integers(1, MAX_GROUPS + 1))
    feasible = _FEASIBLE_COUNTS[n_groups]
    counts = list(feasible[int(rng.integers(len(feasible)))])
    ranks = [int(rng.integers(1, c + 1)) for c in counts]

    cap_palette = rng.choice(np.arange(*COLOR_RANGE), size=n_groups, replace=False)
    caps = [int(v) for v in cap_palette]
    non_caps = [c for c in range(*COLOR_RANGE) if c not in caps]
    stripe_pairs = []
    for cap in caps:
        choices = rng.choice(non_caps, size=2, replace=False)
        a, b = (int(choices[0]), int(choices[1]))
        assert a != cap and b != cap
        stripe_pairs.append((a, b))

    total_bars = sum(counts)
    min_width = max(WIDTH_MIN, COL_SPACING * total_bars - 1)
    width = int(rng.integers(min_width, WIDTH_MAX + 1))
    group_columns = _assign_columns(rng, n_groups, counts, width)

    grid = np.zeros((GRID_H, width), dtype=int)
    group_lengths: list[list[int]] = []
    for cap, stripes, count, rank, cols in zip(caps, stripe_pairs, counts, ranks, group_columns):
        paste(grid, np.full((rank, 1), cap, dtype=int), 0, cols[0])

        lengths = _choose_fill_lengths(rng, count, rank)
        group_lengths.append(lengths)

        for col, fill_len in zip(cols, lengths):
            phase = int(rng.integers(0, 2))
            paste(grid, _build_bar(cap, stripes[0], stripes[1], fill_len, phase), _MAX_CAP_ROW - fill_len, col)

    expected = _expected_output(grid, caps, group_columns, group_lengths, ranks)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_RETRIES} attempts")
