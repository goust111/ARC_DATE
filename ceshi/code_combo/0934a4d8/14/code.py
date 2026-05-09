# Combo variant of 0934a4d8 (index 14)
# combo_spec: D4/2+D14/1
# Name: 多层级回退恢复+标记跨轴

from __future__ import annotations

import numpy as np

from arc_utils import bbox

# Design analysis:
# a) 具体化与行为追踪
#    - 画面仍是 30x30 方阵、唯一一个实心矩形 marker、输出该矩形被遮挡的原始子图。
#    - 清洁底图继续保持原题的 LR+UD 双轴镜像骨架；变化出现在“怎么恢复 marker 内像素”和“marker 放哪”。
#    - 融合后 marker 必须横跨至少一条中心对称轴，因此像 `(r,c)` 的第一候选 LR 镜像 `(r,mc)` 常常也落在 marker 里。
#    - 对每个被遮挡像素，solver 不再收集整条 LR/UD 轨道的可见值，而是按固定优先级逐层尝试：
#      1. 左右镜像 `(r, mc)`
#      2. 上下镜像 `(mr, c)`
#      3. 偏移主对角反射 `(c + 1, r - 1)`
#      4. 主对角转置 `(c, r)`
#    - 具体小例子：若 marker 跨越竖轴并覆盖 `(10,15)` 与 `(10,16)`，则 `(10,15)` 的 LR 候选 `(10,16)` 被挡住；
#      若同一矩形还跨越横轴，则 UD 候选 `(21,15)` 也可能被挡住；这时 solver 继续查看 `(16,9)` 的偏移对角来源，
#      若那格仍被挡，再退到 `(15,10)`。原题在这种 central cross-axis 情况下通常不会走到第三、四层。
# b) 复杂度校准
#    - 保持原题的单 marker 与双轴对称视觉骨架，不额外增加新对象或颜色规则，避免把任务变成全新题。
#    - 为了让“多层级回退”在样例中可观察，生成器不接受只用 LR/UD 就能恢复的简单跨轴矩形；
#      它会均衡抽样水平跨轴与垂直跨轴两类放置，保证样例里稳定出现 `diag` 深层回退，而 `V` 类样例还会继续暴露最终 `transpose`。
#    - 偏移对角使用 `r = c + 1` 这条轴，避免与最终转置 `g[c,r]` 重合为固定点，仍能形成稳定且可见的第三层来源。
# c) 隐式假设审查
#    - 原 `_reflection_orbit_values()` 的假设是“只要 LR/UD 轨道里有可见值，合并起来就是唯一答案”；融合后必须保留顺序，
#      因为 LR 被挡住时不能直接跳去轨道中的 UD 值之外的任意成员，而要先尝试 UD、再尝试对角、最后才转置。
#    - 原 `else: assert r < lo or c < lo` 假设只有外边框才会需要回退；跨轴矩形使内层像素也会触发深层回退，这条断言失效。
#    - 原生成器只需保证“整条 LR/UD 轨道全被挡住”时有转置备份；融合后必须逐像素保证优先级链中最早可见的那一层存在，
#      并对 `diag` / `transpose` 分支单独做颜色合并。
# d) 放置合法性
#    - 矩形必须在边界内，且它是唯一显式对象，因此不存在对象间重叠问题。
#    - 合法放置需同时满足：
#      1. 横跨至少一条中心对称轴；
#      2. marker 内每个像素的四层候选中至少有一层落在 marker 外；
#      3. `H`/`V` 两类跨轴放置都要反复出现；其中 `H` 至少走到 `diag`，`V` 至少走到 `transpose`，避免退化。
# e) 视觉可观察性
#    - 人类从 2-4 个样例中能看到 marker 总是卡在中心镜像轴上，而不是像原题那样可随意落在边缘。
#    - 同一个矩形内部会混合“直接镜像可见”“镜像被挡但更深层可见”的多种像素，因此输出与周围可见备份位置存在可追踪联系。
# f) 可视化整洁度
#    - 视觉上仍只有背景纹理和一个纯色矩形 marker，没有额外覆盖层；复杂度来自恢复链，而不是画面元素堆叠。
#    - 所有额外约束都通过底图颜色同步实现，不引入额外符号，所以输入仍然整洁。
# g) 变化的可推断性
#    - 没有颜色置换规则；颜色只通过空间对应关系重复出现。解题者需要推断的是位置优先级链，而不是颜色 id 含义。
#    - 生成器偏向让 `diag` 和 `transpose` 分支重复出现，避免深层回退只在单个像素上偶发；同时交替覆盖 `H`/`V` 两种跨轴方向。
# h) 新 solver 假设与 generator 保证
#    - solver 假设 `[lo..hi]` 的 LR/UD 镜像轴仍可从可见部分推回；生成器继续保持原题的双轴对称底图。
#    - solver 假设 marker 确实跨越至少一条中心轴；生成器用 placement 过滤保证这一点。
#    - solver 假设第三层偏移对角来源与第四层转置来源在需要时可见且与目标同色；
#      生成器通过把目标像素的 LR/UD 轨道与相应备份来源的 LR/UD 轨道并入同一个颜色组来保证。
# i) 对修改后 `main()` 的逐段假设复查
#    - `_solid_rectangle_marker()` 依赖“唯一实心矩形颜色”仍成立；生成器若意外造出别的实心色块，会在校验阶段重试。
#    - `_find_reflected_strip_start()` 依赖“忽略 marker 后，某个最小 lo 使 LR/UD 一致”；底图构造保持该条件不变。
#    - `_recover_from_priority_chain()` 依赖“按固定顺序总能遇到第一个非 marker 候选”；合法放置筛选正是为这条性质服务。
#    - `main()` 里新增的跨轴断言依赖“placement 一定跨轴”；生成器明确过滤并只抽取跨轴放置。


def _solid_rectangle_marker(g: np.ndarray) -> tuple[int, tuple[int, int, int, int]]:
    candidates: list[tuple[int, int, tuple[int, int, int, int]]] = []
    for color in np.unique(g):
        mask = g == int(color)
        b = bbox(mask)
        assert b is not None
        r0, c0, r1, c1 = b
        area = int((r1 - r0 + 1) * (c1 - c0 + 1))
        if area > 1 and int(mask.sum()) == area and bool(np.all(mask[r0 : r1 + 1, c0 : c1 + 1])):
            candidates.append((area, int(color), b))

    assert len(candidates) == 1
    _area, color, b = candidates[0]
    return color, b


def _find_reflected_strip_start(g: np.ndarray, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m
    hi = n - 1

    for lo in range(n // 2 + 1):
        strip = g[:, lo : hi + 1]
        strip_reflected = strip[:, ::-1]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        strip = g[lo : hi + 1, :]
        strip_reflected = strip[::-1, :]
        known = (strip != marker) & (strip_reflected != marker)
        if bool(np.any((strip != strip_reflected) & known)):
            continue

        return lo

    raise AssertionError("no reflected strip found")


def _mirror(x: int, lo: int, hi: int) -> int:
    return int(lo + hi - x)


def _diag_reflect(r: int, c: int) -> tuple[int, int]:
    """Reflect across the offset diagonal r = c + 1."""
    return int(c + 1), int(r - 1)


def _axis_pair(lo: int, n: int) -> tuple[int, int]:
    hi = n - 1
    left = (lo + hi) // 2
    return left, left + 1


def _crosses_axis(start: int, end: int, lo: int, n: int) -> bool:
    axis_lo, axis_hi = _axis_pair(lo, n)
    return int(start) <= axis_lo and int(end) >= axis_hi


def _inside_marker(r: int, c: int, r0: int, c0: int, h: int, w: int) -> bool:
    return r0 <= r < r0 + h and c0 <= c < c0 + w


def _recovery_candidates(r: int, c: int, lo: int, n: int) -> list[tuple[str, bool, tuple[int, int]]]:
    hi = n - 1
    dr, dc = _diag_reflect(int(r), int(c))
    return [
        ("lr", lo <= c <= hi, (int(r), _mirror(int(c), lo, hi))),
        ("ud", lo <= r <= hi, (_mirror(int(r), lo, hi), int(c))),
        ("diag", 0 <= dr < n and 0 <= dc < n, (dr, dc)),
        ("transpose", 0 <= c < n and 0 <= r < n, (int(c), int(r))),
    ]


def _recover_from_priority_chain(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> int:
    n, m = map(int, g.shape)
    assert n == m

    for _name, enabled, (rr, cc) in _recovery_candidates(int(r), int(c), lo, n):
        if not enabled:
            continue
        value = int(g[rr, cc])
        if value != marker:
            return value

    raise AssertionError("no recovery source found")


def _legacy_reflection_orbit_values(g: np.ndarray, r: int, c: int, lo: int, marker: int) -> set[int]:
    hi = int(g.shape[0]) - 1
    seen: set[tuple[int, int]] = set()
    values: set[int] = set()
    stack = [(int(r), int(c))]

    while stack:
        rr, cc = stack.pop()
        if (rr, cc) in seen:
            continue
        seen.add((rr, cc))

        value = int(g[rr, cc])
        if value != marker:
            values.add(value)

        if lo <= cc <= hi:
            stack.append((rr, _mirror(cc, lo, hi)))
        if lo <= rr <= hi:
            stack.append((_mirror(rr, lo, hi), cc))

    return values


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    """Original 0934a4d8 recovery, used only to reject degenerate combo samples."""
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            r = int(r0 + rr)
            c = int(c0 + cc)
            values = _legacy_reflection_orbit_values(g, r, c, lo, marker)
            if values:
                assert len(values) == 1
                out[rr, cc] = values.pop()
            else:
                assert r < lo or c < lo
                value = int(g[c, r])
                assert value != marker
                out[rr, cc] = value

    return out


def _branch_for_cell(r: int, c: int, r0: int, c0: int, h: int, w: int) -> str | None:
    for name, enabled, (rr, cc) in _recovery_candidates(int(r), int(c), _LO, _N):
        if enabled and not _inside_marker(rr, cc, r0, c0, h, w):
            return name
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the rectangle hidden by the solid marker using a fixed four-step backup chain on a cross-axis placement.

    Input structure:
    - A square colored grid contains one solid rectangular marker color that covers the answer patch.
    - The visible background still follows the original bilateral mirror scaffold on the large [lo..N-1] strip.
    - The marker is forced to straddle at least one center symmetry axis, so many covered cells also hide their first mirror witness.

    Rule:
    - Detect the unique solid rectangle color and use its bounding box as the output window.
    - Infer the mirrored strip start from the smallest offset whose lower/right strip stays LR- and UD-consistent whenever both compared cells are visible.
    - For each covered cell, probe recovery sources in strict order: LR mirror, UD mirror, offset-diagonal reflection, then transpose; the first non-marker cell supplies the answer.

    Key insight:
    The cross-axis marker deliberately blocks shallow mirror partners, so the hidden patch is recoverable only by following the same priority ladder everywhere rather than by treating the reflection orbit as an unordered set.
    """
    g = np.asarray(grid, dtype=int)
    marker, (r0, c0, r1, c1) = _solid_rectangle_marker(g)
    lo = _find_reflected_strip_start(g, marker)
    n = int(g.shape[0])

    assert _crosses_axis(r0, r1, lo, n) or _crosses_axis(c0, c1, lo, n)

    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    out = np.empty((h, w), dtype=int)

    for rr in range(h):
        for cc in range(w):
            out[rr, cc] = _recover_from_priority_chain(g, int(r0 + rr), int(c0 + cc), lo, marker)

    return out


# --------------- generator ---------------

# --- CFG: canvas & structure ---
_N = 30
_LO = 2
_HI = _N - 1
_AXIS_LO, _AXIS_HI = _axis_pair(_LO, _N)

# --- CFG: colors ---
_PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 9], dtype=int)
_MARKER = 8

# --- CFG: marker rectangle size ---
_MARKER_DIM_LO = 3
_MARKER_DIM_HI = 9
_MARKER_AREA_LO = 16
_MARKER_AREA_HI = 36

# --- CFG: generation retries ---
_MAX_RETRIES = 300

_RNG = np.random.default_rng()
_SIZE_PAIRS = [
    (h, w)
    for h in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    for w in range(_MARKER_DIM_LO, _MARKER_DIM_HI + 1)
    if _MARKER_AREA_LO <= h * w <= _MARKER_AREA_HI
]
_PLACEMENT_CACHE: dict[
    tuple[int, int],
    dict[str, list[tuple[int, int, str, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]]],
] = {}


def _orbit(r: int, c: int) -> set[tuple[int, int]]:
    rows = {r}
    cols = {c}
    if _LO <= r <= _HI:
        rows.add(_mirror(r, _LO, _HI))
    if _LO <= c <= _HI:
        cols.add(_mirror(c, _LO, _HI))
    return {(rr, cc) for rr in rows for cc in cols}


def _placement_mode(r0: int, c0: int, h: int, w: int) -> str:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    cross_h = _crosses_axis(r0, r1, _LO, _N)
    cross_v = _crosses_axis(c0, c1, _LO, _N)
    if cross_h and cross_v:
        return "HV"
    if cross_h:
        return "H"
    if cross_v:
        return "V"
    return ""


def _placement_branch_data(
    h: int, w: int, r0: int, c0: int
) -> tuple[str, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    mode = _placement_mode(r0, c0, h, w)
    if not mode:
        return None

    counts = {"lr": 0, "ud": 0, "diag": 0, "transpose": 0}
    diag_cells: list[tuple[int, int]] = []
    transpose_cells: list[tuple[int, int]] = []
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            branch = _branch_for_cell(r, c, r0, c0, h, w)
            if branch is None:
                return None
            counts[branch] += 1
            if branch == "diag":
                diag_cells.append((r, c))
            elif branch == "transpose":
                transpose_cells.append((r, c))

    if any(counts[name] == 0 for name in ("lr", "ud", "diag")):
        return None
    deep_total = counts["diag"] + counts["transpose"]
    if deep_total < 4:
        return None
    if mode == "H" and counts["transpose"] != 0:
        return None
    if mode == "V" and counts["transpose"] == 0:
        return None
    if mode == "HV":
        # With this priority chain, both-axis central placements leave fully hidden cells
        # with no visible source outside the marker, so we keep only H/V families.
        return None

    return mode, counts, diag_cells, transpose_cells


def _valid_placements(
    h: int, w: int
) -> dict[str, list[tuple[int, int, str, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]]]:
    key = (h, w)
    if key not in _PLACEMENT_CACHE:
        placements: dict[
            str, list[tuple[int, int, str, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]]
        ] = {"H": [], "V": [], "HV": []}
        for r0 in range(_N - h + 1):
            for c0 in range(_N - w + 1):
                data = _placement_branch_data(h, w, r0, c0)
                if data is None:
                    continue
                mode, counts, diag_cells, transpose_cells = data
                placements[mode].append((r0, c0, mode, counts, diag_cells, transpose_cells))
        _PLACEMENT_CACHE[key] = placements
    return _PLACEMENT_CACHE[key]


def _build_clean() -> np.ndarray:
    half = (_N - _LO) // 2
    clean = np.empty((_N, _N), dtype=int)

    q = _RNG.choice(_PALETTE, size=(half, half))
    top = np.concatenate([q, q[:, ::-1]], axis=1)
    clean[_LO:_N, _LO:_N] = np.concatenate([top, top[::-1, :]], axis=0)

    for r in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[r, _LO : _LO + half] = vals
        clean[r, _LO + half : _N] = vals[::-1]

    for c in range(_LO):
        vals = _RNG.choice(_PALETTE, size=half)
        clean[_LO : _LO + half, c] = vals
        clean[_LO + half : _N, c] = vals[::-1]

    clean[:_LO, :_LO] = _RNG.choice(_PALETTE, size=(_LO, _LO))
    return clean


def _unify_groups(clean: np.ndarray, groups: list[set[tuple[int, int]]]) -> None:
    parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(x: tuple[int, int]) -> tuple[int, int]:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for group in groups:
        coords = list(group)
        if not coords:
            continue
        for coord in coords[1:]:
            union(coords[0], coord)

    merged: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for coord in parent:
        merged.setdefault(find(coord), []).append(coord)

    for coords in merged.values():
        color = int(_RNG.choice(_PALETTE))
        for r, c in coords:
            clean[r, c] = color


def _pick_shape_and_placement(
) -> tuple[int, int, int, int, str, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]] | None:
    choices_by_mode: dict[
        str, list[tuple[int, int, int, int, str, dict[str, int], list[tuple[int, int]], list[tuple[int, int]]]]
    ] = {"H": [], "V": []}
    for h, w in _SIZE_PAIRS:
        placements = _valid_placements(h, w)
        for mode in ("H", "V"):
            for r0, c0, _mode, counts, diag_cells, transpose_cells in placements[mode]:
                choices_by_mode[mode].append((h, w, r0, c0, mode, counts, diag_cells, transpose_cells))

    available_modes = [mode for mode in ("H", "V") if choices_by_mode[mode]]
    if not available_modes:
        return None

    mode = available_modes[int(_RNG.integers(len(available_modes)))]
    choices = choices_by_mode[mode]
    weights = np.array(
        [
            1.0
            + 1.5 * min(item[5]["diag"], 4)
            + 2.0 * min(item[5]["transpose"], 4)
            for item in choices
        ],
        dtype=float,
    )
    weights /= weights.sum()
    idx = int(_RNG.choice(len(choices), p=weights))
    return choices[idx]


def _gen_attempt() -> np.ndarray | None:
    clean = _build_clean()
    picked = _pick_shape_and_placement()
    if picked is None:
        return None

    h, w, r0, c0, _mode, _counts, diag_cells, transpose_cells = picked
    groups = [_orbit(r, c) | _orbit(*_diag_reflect(r, c)) for r, c in diag_cells]
    groups.extend(_orbit(r, c) | _orbit(c, r) for r, c in transpose_cells)
    _unify_groups(clean, groups)

    expected = clean[r0 : r0 + h, c0 : c0 + w].copy()
    inp = clean.copy()
    inp[r0 : r0 + h, c0 : c0 + w] = _MARKER

    try:
        if not np.array_equal(main(inp), expected):
            return None
        try:
            if np.array_equal(_legacy_main(inp), expected):
                return None
        except Exception:
            pass
        return inp
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid 0934a4d8 combo input")
