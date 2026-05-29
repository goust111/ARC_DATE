# Combo variant of 1ae2feb7 (index 6)
# combo_spec: D3/0+D8/0
# Name: 位置编码双重语义

"""
Design analysis for the D3/0 + D8/0 fusion
===========================================

a) 具体化与行为追踪
-------------------
- 原题里，source side 的每个活跃行有 1-2 条水平色条；每条色条的颜色就是输出颜色，
  色条长度就是 target side 上的周期步长。
- 这次融合后，source object 不再是水平条，而是单个彩色像素控制点。控制点的颜色仍然
  直接决定 target side 的涂色颜色，但周期不再由“长度”给出，因为单像素没有长度可读；
  周期改为“控制点到分隔列的列距离”，即 `step = abs(c - sep_col)`。
- 仍然保留原题的核心执行顺序：同一行若有多个 source object，就按离分隔列最近到最远的顺序
  处理；每个对象都从分隔列另一侧的第一格开始，按自己的步长周期性落点；已经被更近对象涂过
  的格子不再覆盖。
- 小例子：若分隔列在 `c=6`，source side 在左。
  * 原题一行上若有色条 `444` 贴着分隔列，则颜色 4 以步长 3 涂到 `c=7,10,13,...`。
  * 融合后若这一行有两个控制点，分别在 `c=5` 颜色 4 和 `c=3` 颜色 7，
    则它们的步长分别是 1 和 3。先处理更近的 4，把整行 target side 连续涂满；
    再处理更远的 7 时，因为所有落点都已非背景，7 不再出现。
  * 若只有远点 `(r,3)=7` 而没有近点，则输出会在 `c=7,10,13,...` 出现稀疏的 7。
- 差异的关键是：原题解题者看“条长”，本题解题者看“点离分隔条有多远”。颜色和步长来自同一
  个像素的两个不同属性，形成“位置编码双重语义”。

b) 复杂度校准
-------------
- 若把所有控制点都固定在分隔条旁边，那么所有步长都会退化成 1，D8 的改动失效。因此生成器
  强制使用至少两个不同列距离，并保证至少一个控制点的距离 > 1。
- 若每行只放一个点，规则仍成立，但“最近优先且不覆盖”的原题机制就很难被看见。因此保留
  “每行 1-2 个 source object”的骨架，并强制至少一行出现 2 个控制点，让优先级仍有作用。
- 同时不把自由度放得太大：控制点仍只出现在 source side，活跃行仍有空行间隔，步长词表在
  每个样例里只取 2-3 个距离值并跨行复用，这样肉眼容易看出“同列距 => 同步长”。

c) 原 solver 的隐式假设审查
---------------------------
- `_find_sep_col(...)` 假设存在一列几乎满高且单色的竖直分隔列。该假设保留。
- 原题通过扫描连续 run 并取 `length = c1 - c0 + 1` 来得到步长；融合后这个前提失效，
  因为 source object 全是离散单像素。继续用原逻辑会把所有步长误读为 1。
- 原题按 `dist` 排序是为了让同一行更靠近分隔列的条先占据 target side。这个前提仍然有价值，
  只是 `dist` 不再只是优先级，还同时成为步长本身。
- `bars_on_left = left_mass >= right_mass` 依赖输入的 target side 为空。生成器继续保证：
  输入只在 source side 与 separator 上有非背景色。

d) 放置合法性
-------------
- 控制点放在分隔列同侧，距离取值满足 `1 <= step <= source_side_width`，因此不会越界。
- 同一行若有两个控制点，它们必须使用不同的距离值，所以不会重叠到同一格。
- 活跃行沿用原题的“至少隔一空行”采样，不同行控制点天然不冲突。
- 分隔列偶尔允许在非活跃的顶行或底行缺一个像素，但不会和控制点同格，也不会破坏识别。
- 生成器要求 `step <= ext_side_len - 1`，让每个控制点至少能在 target side 产生两个落点，
  避免出现不可观察的单点周期。

e) 视觉可观察性
----------------
- 输入里 source side 变成稀疏的控制点阵列，而不是水平条；输出里每个活跃行的疏密程度直接与
  控制点离分隔列的远近对应，肉眼可见。
- 因为步长词表在同一网格里重复使用，解题者能看到多个行共享同一列偏移和同一周期，而不需要
  依赖颜色 id 做不可见推理。
- 至少会同时出现一步长 1 的连续填充行和一步长 > 1 的稀疏打点行，差异显著。

f) 可视化整洁度
---------------
- 分隔列仍把 source side 与 target side 明确切开；source side 只有单像素，没有长条堆叠，
  视觉上比原题更简洁。
- 只有同一行的 1-2 个控制点会共同影响同一行输出，不会跨行干扰；target side 也继续遵守
  “不覆盖”规则，因此不会产生难以解释的颜色混杂。

g) 变化的可推断性
-------------------
- 颜色规则保持最直接的继承：控制点颜色就是它在 target side 产生的颜色，不引入隐藏映射。
- 真正变化的是“同一个控制点的哪个属性编码步长”：原题是水平长度，这里是列距离。
- 生成器使用 2-3 个重复出现的距离档位，并强制样本与“所有控制点步长都当成 1”时的输出不同，
  防止退化成看起来还像原题但实际上没用到距离编码的弱样本。

9) 新 solver 的隐式假设审查
---------------------------
- solver 假设 source side 上所有有用像素都是离散控制点，而不是更长的连续 run。生成器保证
  source side 除这些点外全为空白。
- solver 假设同一行若有多个控制点，它们到分隔列的距离两两不同。生成器按距离集合采样并逐行
  去重，保证这一点。
- solver 假设 target side 在输入中为空，因此“left_mass >= right_mass”能可靠识别 source side。
- solver 假设步长至少为 1；由于距离是按列差定义，最小合法距离正好就是 1。
"""

from __future__ import annotations

import numpy as np

from arc_utils import colors_present, get_bg
from arc_utils import get_bg as _bg_color


def _find_sep_col(g: np.ndarray, bg: int) -> tuple[int | None, int | None]:
    h, w = g.shape
    for c in range(w):
        col = g[:, c]
        vals = col[col != bg]
        if vals.size >= h - 1 and (vals == vals[0]).all():
            return c, int(vals[0])
    return None, None


def _source_side_is_left(g: np.ndarray, sep_col: int, bg: int) -> bool:
    left_mass = int(np.sum(g[:, :sep_col] != bg)) if sep_col > 0 else 0
    right_mass = int(np.sum(g[:, sep_col + 1 :] != bg)) if sep_col + 1 < g.shape[1] else 0
    return left_mass >= right_mass


def _iter_control_points(
    g: np.ndarray,
    sep_col: int,
    controls_on_left: bool,
    bg: int,
) -> list[tuple[int, int, int, int]]:
    """Return (dist, row, col, color) for each source-side control pixel."""
    h, w = g.shape
    if controls_on_left:
        col_range = range(0, sep_col)
    else:
        col_range = range(sep_col + 1, w)

    pts: list[tuple[int, int, int, int]] = []
    for r in range(h):
        for c in col_range:
            color = int(g[r, c])
            if color == bg:
                continue
            dist = abs(int(c) - int(sep_col))
            if dist <= 0:
                continue
            pts.append((dist, int(r), int(c), color))
    return pts


def _paint_target(
    out: np.ndarray,
    row: int,
    sep_col: int,
    controls_on_left: bool,
    step: int,
    color: int,
    bg: int,
) -> None:
    w = out.shape[1]
    if controls_on_left:
        positions = range(sep_col + 1, w, step)
    else:
        positions = range(sep_col - 1, -1, -step)
    for cc in positions:
        if int(out[row, cc]) == bg:
            out[row, cc] = color


def _simulate_variant(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    controls_on_left = _source_side_is_left(g, sep_col, bg)
    out = g.copy()

    controls = _iter_control_points(g, sep_col, controls_on_left, bg)
    controls.sort(key=lambda t: (t[1], t[0], t[2]))
    for dist, row, _col, color in controls:
        _paint_target(out, row, sep_col, controls_on_left, dist, color, bg)

    return out


def _simulate_unit_step_baseline(grid: np.ndarray) -> np.ndarray:
    """Baseline that ignores the distance code and treats every control pixel as step 1."""
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    sep_col, _sep_color = _find_sep_col(g, bg)
    if sep_col is None:
        return g.copy()

    controls_on_left = _source_side_is_left(g, sep_col, bg)
    out = g.copy()

    controls = _iter_control_points(g, sep_col, controls_on_left, bg)
    controls.sort(key=lambda t: (t[1], t[0], t[2]))
    for _dist, row, _col, color in controls:
        _paint_target(out, row, sep_col, controls_on_left, 1, color, bg)

    return out


def _distance_effect_cells(grid: np.ndarray) -> int:
    fused = _simulate_variant(grid)
    baseline = _simulate_unit_step_baseline(grid)
    return int(np.sum(fused != baseline))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each source-side control pixel across the separator, using its column distance as the period.

    Input structure:
    - A near-solid vertical separator column divides the grid into a sparse source side and an empty target side.
    - Source objects are single colored pixels, not horizontal bars.
    - Active rows contain 1-2 control pixels on the source side; their colors may differ, and their column offsets from the separator may differ.

    Rule:
    - For each active row, process its control pixels from nearest to farthest from the separator.
    - A control pixel with color `k` and separator distance `d` paints color `k` on the opposite side at positions
      `1, 1+d, 1+2d, ...` away from the separator along that same row.
    - Never overwrite a cell that was already painted by a nearer control pixel.

    Key insight:
    The single source pixel carries two meanings at once: its color chooses the paint color, while its horizontal
    position relative to the separator encodes the repeat interval.
    """
    return _simulate_variant(grid)


# --------------- generator ---------------

H_MIN, H_MAX = 8, 13
W_MIN, W_MAX = 15, 20

CTRL_SIDE_MIN, CTRL_SIDE_MAX = 4, 12
EXT_SIDE_MIN, EXT_SIDE_MAX = 6, 14

PALETTE_MIN, PALETTE_MAX = 6, 8

ACTIVE_MIN, ACTIVE_MAX = 3, 4
DISTANCE_VOCAB_MIN, DISTANCE_VOCAB_MAX = 2, 3

P_EXTRA_TWO_CONTROL_ROW = 0.35
P_SEP_COLUMN_GAP = 0.2

MAX_ATTEMPTS = 500


def _choose_active_rows(rng: np.random.Generator, h: int, count: int) -> list[int]:
    """Pick `count` row indices in [0, h) with pairwise spacing >= 2."""
    base = np.sort(rng.choice(h - count + 1, size=count, replace=False))
    return list(map(int, base + np.arange(count)))


def _sample_palette(rng: np.random.Generator) -> tuple[int, list[int]]:
    sep_color = int(rng.integers(1, 10))
    ctrl_pool = [c for c in range(1, 10) if c != sep_color]
    size = int(rng.integers(PALETTE_MIN, min(PALETTE_MAX, len(ctrl_pool)) + 1))
    palette = list(map(int, rng.choice(ctrl_pool, size=size, replace=False)))
    return sep_color, palette


def _sample_distance_vocab(rng: np.random.Generator, max_step: int) -> list[int]:
    if max_step < 3:
        return []
    size = int(rng.integers(DISTANCE_VOCAB_MIN, min(DISTANCE_VOCAB_MAX, max_step) + 1))
    far_choices = list(range(2, max_step + 1))
    far_anchor = int(rng.choice(far_choices))
    vocab = {1, far_anchor}
    extra_needed = size - len(vocab)
    if extra_needed > 0:
        extras_pool = [d for d in range(2, max_step + 1) if d not in vocab]
        extras = list(map(int, rng.choice(extras_pool, size=extra_needed, replace=False)))
        vocab.update(extras)
    return sorted(vocab)


def _assign_row_distances(
    rng: np.random.Generator,
    active_rows: list[int],
    distance_vocab: list[int],
) -> dict[int, list[int]]:
    active_count = len(active_rows)
    if active_count < 3 or len(distance_vocab) < 2:
        return {}

    row_two = int(active_rows[int(rng.integers(0, active_count))])
    other_rows = [r for r in active_rows if r != row_two]
    row_near = int(other_rows[int(rng.integers(0, len(other_rows)))])
    remaining = [r for r in other_rows if r != row_near]
    row_far = int(remaining[int(rng.integers(0, len(remaining)))]) if remaining else row_two

    far_options = [d for d in distance_vocab if d > 1]
    far_anchor = int(rng.choice(far_options))

    row_map: dict[int, list[int]] = {row_two: [1, far_anchor], row_near: [1], row_far: [far_anchor]}
    for row in active_rows:
        if row in row_map:
            continue
        use_two = len(distance_vocab) >= 2 and rng.random() < P_EXTRA_TWO_CONTROL_ROW
        if use_two:
            ds = sorted(map(int, rng.choice(distance_vocab, size=2, replace=False)))
        else:
            ds = [int(rng.choice(distance_vocab))]
        row_map[int(row)] = ds
    return row_map


def _dist_to_col(step: int, sep_col: int, controls_on_left: bool) -> int:
    return int(sep_col - step) if controls_on_left else int(sep_col + step)


def _place_controls(
    inp: np.ndarray,
    sep_col: int,
    controls_on_left: bool,
    row_map: dict[int, list[tuple[int, int]]],
) -> None:
    for row, controls in row_map.items():
        for step, color in controls:
            col = _dist_to_col(step, sep_col, controls_on_left)
            inp[row, col] = int(color)


def _build_row_controls(
    rng: np.random.Generator,
    active_rows: list[int],
    distance_vocab: list[int],
    palette: list[int],
) -> dict[int, list[tuple[int, int]]]:
    distance_map = _assign_row_distances(rng, active_rows, distance_vocab)
    if not distance_map:
        return {}

    total_controls = sum(len(ds) for ds in distance_map.values())
    if total_controls > len(palette):
        return {}

    colors = list(map(int, rng.choice(palette, size=total_controls, replace=False)))
    row_controls: dict[int, list[tuple[int, int]]] = {}
    idx = 0
    for row in active_rows:
        pairs = []
        for step in distance_map[row]:
            pairs.append((int(step), colors[idx]))
            idx += 1
        row_controls[int(row)] = pairs
    return row_controls


def _control_summary(
    row_controls: dict[int, list[tuple[int, int]]],
) -> tuple[list[int], int]:
    steps = [step for controls in row_controls.values() for step, _color in controls]
    two_control_rows = sum(1 for controls in row_controls.values() if len(controls) == 2)
    return steps, int(two_control_rows)


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    h = int(rng.integers(H_MIN, H_MAX + 1))
    w = int(rng.integers(W_MIN, W_MAX + 1))

    lo = max(CTRL_SIDE_MIN, w - 1 - EXT_SIDE_MAX)
    hi = min(CTRL_SIDE_MAX, w - 1 - EXT_SIDE_MIN)
    if lo > hi:
        return None
    ctrl_side_len = int(rng.integers(lo, hi + 1))
    ext_side_len = int(w - 1 - ctrl_side_len)
    max_step = min(ctrl_side_len, ext_side_len - 1)
    if max_step < 3:
        return None

    controls_on_left = bool(rng.integers(0, 2) == 0)
    sep_col = int(ctrl_side_len if controls_on_left else (w - 1 - ctrl_side_len))

    max_active = min(ACTIVE_MAX, (h + 1) // 2)
    if max_active < ACTIVE_MIN:
        return None
    active_count = int(rng.integers(ACTIVE_MIN, max_active + 1))
    active_rows = _choose_active_rows(rng, h, active_count)

    distance_vocab = _sample_distance_vocab(rng, max_step)
    if len(distance_vocab) < 2 or max(distance_vocab) <= 1:
        return None

    sep_color, palette = _sample_palette(rng)
    row_controls = _build_row_controls(rng, active_rows, distance_vocab, palette)
    if not row_controls:
        return None

    steps, two_control_rows = _control_summary(row_controls)
    if len(set(steps)) < 2 or max(steps) <= 1 or two_control_rows < 1:
        return None

    inp = np.zeros((h, w), dtype=int)
    inp[:, sep_col] = sep_color

    active_set = set(active_rows)
    terminal_inactive = [r for r in (0, h - 1) if r not in active_set]
    if terminal_inactive and rng.random() < P_SEP_COLUMN_GAP:
        gap_row = int(rng.choice(terminal_inactive))
        inp[gap_row, sep_col] = 0

    _place_controls(inp, sep_col, controls_on_left, row_controls)

    out = _simulate_variant(inp)
    if get_bg(inp) != 0:
        return None
    if sep_color not in colors_present(inp, exclude=[0]):
        return None
    if np.array_equal(inp, out):
        return None
    if _distance_effect_cells(inp) <= 0:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_ATTEMPTS} attempts")
