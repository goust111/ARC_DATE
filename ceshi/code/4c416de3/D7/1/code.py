# Variant of 4c416de3
# Dimension: D7 — 框架结构：形状与数量 (meta)
# Change: 2-3个空心方形0框架，边长5-8，轴对齐矩形，可能有部分截断（L形）... → 嵌套框架（框中框）

import random
from collections import Counter

import numpy as np

from arc_utils import draw_line, get_bg


# Design analysis
# a) 具体化与行为追踪
#    - 变体不再把 2-3 个 0 方框分散摆放，而是把它们做成一组同心嵌套方框。
#      每一层都是完整正方形 0 边框，内外层之间固定留出 3 格“护城河”，这样同方位角点
#      (例如 outer TL / inner TL) 会在对角线上排成一串。
#    - 输入里仍然保留“source 角点已有完整图案、marker 角点只保留最深对角像素”的语法，
#      但复制不再面向“所有带 marker 的角点”，而是只发生在“同方位、相邻嵌套层”的角点对之间。
#    - 小例子追踪：
#      原题：任意 frame 上只要某角点有 marker，就会被全局 source 图案补全。
#      新题：outer TL 若是 source，只会把图案复制到下一层的 TL；另一处孤立 marker 即使存在，
#      也不会变化。差异就在于 target 是否与 source 形成“嵌套同角位配对”。
# b) 复杂度校准
#    - 若只改成“有内外框，但所有 marker 仍然全部补全”，任务会退化回原题。
#      因此生成器固定加入至少一个 decoy marker：它带颜色但没有相邻层 source 配对，输出保持单点。
#    - 为了避免规则过多分支，每个角位 (TL/TR/BL/BR) 最多只出现一种状态：
#      outer-source、inner-source、decoy、blank 之一。这样人眼可以按四个角位逐个归纳。
# c) 隐式假设审查
#    - 原 `_find_corners()` / “所有 marker 都是 target” 的前提已经失效；嵌套后必须先恢复 frame 层级。
#    - 原 target 校验默认邻域内所有 pattern-offset 颜色都属于该角点；嵌套后相邻层 source 很近，
#      所以图案模板不能再含 (-1,-1)，否则相邻 source 会落进 target 的 canonical pattern 集合中。
#      这里把模板收缩为只含 {(-1,0),(0,-1),(0,0),(1,1),(2,2)} 的子集，保证相邻层互不混淆。
#    - solver 新增的关键假设：
#      1. frame 是完整正方形边框；
#      2. 所有 frame 同心嵌套；
#      3. 同一角位至多一个 source；
#      4. source 只会驱动相邻层的同角位 marker。
#      generator 会显式满足这些条件。
# d) 放置合法性
#    - 相邻层 gap 固定为 3，足以容纳 k_max<=2 的 inward 对角轨迹，并避免不同层同角位图案重叠。
#    - 生成时先选内层尺寸，再按 gap 外扩；所以所有层天然不重叠且严格包含。
#    - output-first 构造后总是用 `main(input)` 回验，避免隐藏越界/覆盖冲突。
# e) 视觉可观察性
#    - 样例中会同时出现“会被补全的 marker”和“不会被补全的 decoy marker”，两者差别只在
#      是否有相邻层同角位 source，可直接从 I/O 对照里看到。
#    - 颜色仅用于区分各个角点自己的输出颜色，不参与排序、大小比较或任何不可见计算。
# f) 可视化整洁度
#    - 只保留一组同心嵌套 frame，避免多个局部区域同时叠加。
#    - 每个角位最多一种状态，source / target / decoy 不会在同一角位堆叠。
# g) 颜色变化可推断性
#    - 仍沿用原题的“target 用自己的 marker 颜色着色”。
#    - 每个样例只给 2-3 个 active pair 和至少 1 个 decoy；着色关系由位置与 marker 直接提示，
#      不需要额外的颜色映射推理。


CORNER_KINDS = ("TL", "TR", "BL", "BR")
_CORNER_OFFSETS = {"TL": (0, 0), "TR": (0, 1), "BL": (1, 0), "BR": (1, 1)}

_NEIGH_R = 3
MAX_ATTEMPTS = 400

H_RANGE = (19, 24)
W_RANGE = (19, 24)
BORDER = 1
NEST_GAP = 3
INNER_SIZE_RANGE = (5, 8)
N_FRAMES_3_PROB = 0.45

# Restrict templates so adjacent nested corners (3,3) apart do not project into each
# other's canonical pattern offsets. In particular, omit (-1,-1), and always include
# (0,0) so a true source is distinguishable from a nearby marker-only target.
PATTERN_TEMPLATES = (
    [(-1, 0), (0, -1), (0, 0), (1, 1)],
    [(-1, 0), (0, -1), (0, 0), (1, 1), (2, 2)],
)


def _rot(dr: int, dc: int, kind: str) -> tuple[int, int]:
    """Map actual offset <-> canonical TL-oriented offset; self-inverse."""
    if kind == "TL":
        return dr, dc
    if kind == "TR":
        return dr, -dc
    if kind == "BL":
        return -dr, dc
    if kind == "BR":
        return -dr, -dc
    raise ValueError(kind)


def _corner_coord(frame: dict, kind: str) -> tuple[int, int]:
    dr, dc = _CORNER_OFFSETS[kind]
    span = frame["size"] - 1
    return frame["r"] + dr * span, frame["c"] + dc * span


def _colored_around(g: np.ndarray, r: int, c: int, kind: str, bg: int) -> list[tuple[tuple[int, int], int]]:
    H, W = g.shape
    items: list[tuple[tuple[int, int], int]] = []
    for dr in range(-_NEIGH_R, _NEIGH_R + 1):
        for dc in range(-_NEIGH_R, _NEIGH_R + 1):
            rr, cc = r + dr, c + dc
            if not (0 <= rr < H and 0 <= cc < W):
                continue
            v = int(g[rr, cc])
            if v == bg or v == 0:
                continue
            items.append((_rot(dr, dc, kind), v))
    return items


def _scan_frames(g: np.ndarray, bg: int) -> list[dict]:
    """Detect complete square 0-frames whose edge interiors are all 0."""
    H, W = g.shape
    frames: list[dict] = []
    max_size = min(H, W)
    for size in range(5, max_size + 1):
        span = size - 1
        for r in range(0, H - size + 1):
            for c in range(0, W - size + 1):
                r2, c2 = r + span, c + span
                corners = (int(g[r, c]), int(g[r, c2]), int(g[r2, c]), int(g[r2, c2]))
                if any(v == bg for v in corners):
                    continue
                if not (
                    np.all(g[r, c + 1 : c2] == 0)
                    and np.all(g[r2, c + 1 : c2] == 0)
                    and np.all(g[r + 1 : r2, c] == 0)
                    and np.all(g[r + 1 : r2, c2] == 0)
                ):
                    continue
                frames.append({"r": r, "c": c, "size": size})
    if not frames:
        return []

    center_groups: dict[tuple[int, int], list[dict]] = {}
    for frame in frames:
        key = (2 * frame["r"] + frame["size"] - 1, 2 * frame["c"] + frame["size"] - 1)
        center_groups.setdefault(key, []).append(frame)

    # Generator makes exactly one concentric stack with a fixed 3-cell inset per layer,
    # so true successive frame sizes differ by 2 * NEST_GAP.
    best_group: list[dict] = []
    best_score: tuple[int, int, int] = (-1, -1, -1)
    for group in center_groups.values():
        dedup: dict[tuple[int, int, int], dict] = {}
        for frame in group:
            dedup[(frame["r"], frame["c"], frame["size"])] = frame
        ordered = sorted(dedup.values(), key=lambda f: f["size"], reverse=True)
        by_size = {frame["size"]: frame for frame in ordered}
        sizes = sorted(by_size, reverse=True)

        candidates: list[list[dict]] = []
        for outer in sizes:
            chain = [by_size[outer]]
            next_size = outer - 2 * NEST_GAP
            while next_size in by_size and len(chain) < 3:
                chain.append(by_size[next_size])
                next_size -= 2 * NEST_GAP
            if len(chain) >= 2:
                candidates.append(chain)

        if not candidates:
            continue
        chain = max(candidates, key=lambda frames: (len(frames), frames[0]["size"], -frames[-1]["size"]))
        score = (len(chain), chain[0]["size"], -chain[-1]["size"])
        if score > best_score:
            best_score = score
            best_group = chain

    return best_group


def _draw_frame(grid: np.ndarray, r: int, c: int, size: int) -> None:
    r2, c2 = r + size - 1, c + size - 1
    draw_line(grid, r, c, end_r=r, end_c=c2, color=0)
    draw_line(grid, r, c, end_r=r2, end_c=c, color=0)
    draw_line(grid, r2, c, end_r=r2, end_c=c2, color=0)
    draw_line(grid, r, c2, end_r=r2, end_c=c2, color=0)


def _paint_pattern(
    grid: np.ndarray, r: int, c: int, kind: str, pattern: list[tuple[int, int]], color: int
) -> None:
    for dr, dc in pattern:
        rr, cc = _rot(dr, dc, kind)
        grid[r + rr, c + cc] = color


def _choose_pattern() -> tuple[list[tuple[int, int]], int]:
    pattern = list(random.choice(PATTERN_TEMPLATES))
    k_max = max(dr for dr, dc in pattern if dr == dc and dr > 0)
    return pattern, k_max


def _build_nested_frames(H: int, W: int, n_frames: int, min_inner_size: int) -> list[dict] | None:
    outer_extra = 2 * NEST_GAP * (n_frames - 1)
    max_inner = min(H, W) - 2 * BORDER - outer_extra
    lo = max(INNER_SIZE_RANGE[0], min_inner_size)
    hi = min(INNER_SIZE_RANGE[1], max_inner)
    if hi < lo:
        return None
    inner_size = random.randint(lo, hi)
    outer_size = inner_size + outer_extra
    r0 = random.randint(BORDER, H - outer_size - BORDER)
    c0 = random.randint(BORDER, W - outer_size - BORDER)
    frames = []
    for level in range(n_frames):
        inset = level * NEST_GAP
        size = outer_size - 2 * inset
        frames.append({"r": r0 + inset, "c": c0 + inset, "size": size})
    return frames


def _allocate_states(n_frames: int) -> tuple[dict[tuple[int, str], tuple[str, int | None]], set[tuple[int, str]], set[tuple[int, str]]]:
    """
    Return:
      roles[(level, kind)] = ('source'|'marker'|'blank', color placeholder)
      source_keys
      active_target_keys
    """
    kinds = list(CORNER_KINDS)
    random.shuffle(kinds)
    n_active = random.randint(2, 3)
    n_decoy = 1 if n_active == 3 else random.randint(1, 2)
    active_kinds = kinds[:n_active]
    decoy_kinds = kinds[n_active : n_active + n_decoy]

    roles: dict[tuple[int, str], tuple[str, int | None]] = {}
    source_keys: set[tuple[int, str]] = set()
    active_target_keys: set[tuple[int, str]] = set()

    for level in range(n_frames):
        for kind in CORNER_KINDS:
            roles[(level, kind)] = ("blank", None)

    for kind in active_kinds:
        if n_frames == 2:
            src_level = random.choice((0, 1))
            tgt_level = 1 - src_level
        else:
            # Outer source -> middle target, or inner source -> middle target.
            if random.random() < 0.5:
                src_level, tgt_level = 0, 1
            else:
                src_level, tgt_level = n_frames - 1, n_frames - 2
        roles[(src_level, kind)] = ("source", None)
        roles[(tgt_level, kind)] = ("marker", None)
        source_keys.add((src_level, kind))
        active_target_keys.add((tgt_level, kind))

    for kind in decoy_kinds:
        level = random.randrange(n_frames)
        roles[(level, kind)] = ("marker", None)

    return roles, source_keys, active_target_keys


def _assign_colors(
    roles: dict[tuple[int, str], tuple[str, int | None]],
    bg: int,
) -> dict[tuple[int, str], tuple[str, int | None]]:
    colored_keys = [key for key, (role, _color) in roles.items() if role != "blank"]
    palette = [c for c in range(1, 10) if c != bg]
    assert len(colored_keys) <= len(palette)
    random.shuffle(palette)
    out = dict(roles)
    for key, color in zip(colored_keys, palette):
        role, _ = out[key]
        out[key] = (role, color)
    return out


def _render_output(
    base: np.ndarray,
    frames: list[dict],
    roles: dict[tuple[int, str], tuple[str, int | None]],
    pattern: list[tuple[int, int]],
    marker_canon: tuple[int, int],
) -> np.ndarray:
    out = base.copy()
    for level, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            role, color = roles[(level, kind)]
            if role == "blank":
                continue
            assert color is not None
            r, c = _corner_coord(frame, kind)
            if role == "source":
                _paint_pattern(out, r, c, kind, pattern, color)
            else:
                dr, dc = _rot(*marker_canon, kind)
                out[r + dr, c + dc] = color
    return out


def _derive_input(
    output: np.ndarray,
    base: np.ndarray,
    frames: list[dict],
    roles: dict[tuple[int, str], tuple[str, int | None]],
    pattern: list[tuple[int, int]],
    marker_canon: tuple[int, int],
    active_target_keys: set[tuple[int, str]],
) -> np.ndarray:
    inp = output.copy()
    for level, kind in active_target_keys:
        r, c = _corner_coord(frames[level], kind)
        mdr, mdc = _rot(*marker_canon, kind)
        marker = (r + mdr, c + mdc)
        for dr, dc in pattern:
            rr, cc = _rot(dr, dc, kind)
            rr += r
            cc += c
            if (rr, cc) != marker:
                inp[rr, cc] = base[rr, cc]
    return inp


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy a corner glyph only across matched corners of a nested frame stack.

    Input structure:
    - The grid contains one concentric stack of 2-3 hollow square 0-frames.
    - Some corners are sources: they already show the full diagonally symmetric glyph in one color.
    - Some corners are only marker corners: they keep just the deepest inward-diagonal pixel.
      Marker corners may be true targets or inert decoys.

    Rule:
    - Recover the shared canonical corner glyph from the source corners.
    - For each source, look only at the same-orientation corner on an adjacent nested layer.
      If that adjacent corner carries the marker pixel, paint the full glyph there in the marker's color.
    - Marker corners without an adjacent same-orientation source are left unchanged.

    Key insight:
    - Nesting, not mere marker presence, determines which corners expand.
      The only productive relation is "same corner kind on neighboring nested layers";
      lone markers are decoys and must stay as single pixels.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))

    frames = _scan_frames(g, bg)
    assert 2 <= len(frames) <= 3, f"expected 2-3 nested frames, found {len(frames)}"

    center_key = (2 * frames[0]["r"] + frames[0]["size"] - 1, 2 * frames[0]["c"] + frames[0]["size"] - 1)
    for i in range(1, len(frames)):
        cur = frames[i]
        assert (
            2 * cur["r"] + cur["size"] - 1,
            2 * cur["c"] + cur["size"] - 1,
        ) == center_key, "frames are not concentric"
        prev = frames[i - 1]
        assert prev["r"] < cur["r"] and prev["c"] < cur["c"], "frames are not properly nested"

    per_corner: dict[tuple[int, str], list[tuple[tuple[int, int], int]]] = {}
    for level, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            per_corner[(level, kind)] = _colored_around(g, r, c, kind, bg)

    sources: list[tuple[int, str]] = []
    for level, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            r, c = _corner_coord(frame, kind)
            col = int(g[r, c])
            if col == bg or col == 0:
                continue
            ar1, ac1 = _rot(-1, 0, kind)
            ar2, ac2 = _rot(0, -1, kind)
            v1 = int(g[r + ar1, c + ac1])
            v2 = int(g[r + ar2, c + ac2])
            if v1 == col and v2 == col:
                sources.append((level, kind))
    assert sources, "at least one source corner is required"

    sources_by_kind: dict[str, list[int]] = {kind: [] for kind in CORNER_KINDS}
    for level, kind in sources:
        sources_by_kind[kind].append(level)
    for kind, levels in sources_by_kind.items():
        assert len(levels) <= 1, f"multiple nested sources on {kind}: {levels}"

    def _extract(corner_key: tuple[int, str]) -> tuple[frozenset[tuple[int, int]], int]:
        level, kind = corner_key
        r, c = _corner_coord(frames[level], kind)
        pat_col = int(g[r, c])
        items = per_corner[corner_key]
        return frozenset(off for off, color in items if color == pat_col), pat_col

    canon, _ = _extract(sources[0])
    for source in sources[1:]:
        other, _ = _extract(source)
        assert other == canon, f"source patterns disagree: {sorted(canon)} vs {sorted(other)}"

    for dr, dc in canon:
        assert (dc, dr) in canon, f"pattern not diagonally symmetric: ({dr},{dc}) has no mirror"

    diag_ks = [dr for dr, dc in canon if dr == dc and dr > 0]
    assert diag_ks, "pattern has no inward diagonal marker anchor"
    marker_canon = (max(diag_ks), max(diag_ks))

    marker_colors: dict[tuple[int, str], int] = {}
    for level, frame in enumerate(frames):
        for kind in CORNER_KINDS:
            if (level, kind) in sources:
                continue
            r, c = _corner_coord(frame, kind)
            mdr, mdc = _rot(*marker_canon, kind)
            mr, mc = r + mdr, c + mdc
            v = int(g[mr, mc])
            if v == bg or v == 0:
                continue
            for off, color in per_corner[(level, kind)]:
                if off in canon:
                    assert color == v, (
                        f"marker corner {(level, kind)} has conflicting color {color} at {off}; expected {v}"
                    )
            marker_colors[(level, kind)] = v

    out = g.copy()
    for kind, levels in sources_by_kind.items():
        if not levels:
            continue
        src_level = levels[0]
        for tgt_level in (src_level - 1, src_level + 1):
            if not (0 <= tgt_level < len(frames)):
                continue
            if (tgt_level, kind) not in marker_colors:
                continue
            tcol = marker_colors[(tgt_level, kind)]
            r, c = _corner_coord(frames[tgt_level], kind)
            for cdr, cdc in canon:
                adr, adc = _rot(cdr, cdc, kind)
                rr, cc = r + adr, c + adc
                v = int(out[rr, cc])
                assert v == bg or v == 0 or v == tcol, (
                    f"paint conflict at ({rr},{cc}) for target {(tgt_level, kind)}: {v} vs {tcol}"
                )
                out[rr, cc] = tcol

    return out


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*H_RANGE)
    W = random.randint(*W_RANGE)
    bg = random.randint(1, 9)
    pattern, k_max = _choose_pattern()

    can_have_three = min(H, W) - 2 * BORDER >= INNER_SIZE_RANGE[0] + 4 * NEST_GAP
    if can_have_three and random.random() < N_FRAMES_3_PROB:
        n_frames = 3
    else:
        n_frames = 2

    min_inner_size = 6 if k_max == 2 else INNER_SIZE_RANGE[0]
    frames = _build_nested_frames(H, W, n_frames, min_inner_size)
    if frames is None:
        return None

    base = np.full((H, W), bg, dtype=int)
    for frame in frames:
        _draw_frame(base, frame["r"], frame["c"], frame["size"])

    roles, source_keys, active_target_keys = _allocate_states(n_frames)
    roles = _assign_colors(roles, bg)

    output = _render_output(base, frames, roles, pattern, (k_max, k_max))

    # Expand only the truly active paired markers in the finished output.
    for level, kind in active_target_keys:
        _role, color = roles[(level, kind)]
        assert color is not None
        r, c = _corner_coord(frames[level], kind)
        _paint_pattern(output, r, c, kind, pattern, color)

    inp = _derive_input(output, base, frames, roles, pattern, (k_max, k_max), active_target_keys)

    # Variant-specific guarantee: at least one decoy marker must remain single in the output.
    decoy_count = 0
    for key, (role, color) in roles.items():
        if role != "marker" or key in active_target_keys:
            continue
        level, kind = key
        r, c = _corner_coord(frames[level], kind)
        mdr, mdc = _rot(k_max, k_max, kind)
        if int(output[r + mdr, c + mdc]) == int(color):
            decoy_count += 1
    if decoy_count == 0:
        return None

    if not np.array_equal(main(inp), output):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
