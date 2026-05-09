# Combo variant of 5dbc8537 (index 3)
# combo_spec: D4/1+D15/2
# Name: 方向标记驱动旋转填充

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import (
    bbox_coords,
    colors_present,
    components_mask,
    crop_bbox,
    find_components,
    get_bg,
    paste,
    random_free_location,
)

# Design analysis for the combo variant
# a) Concreteization and behavior trace:
#    - 原题里，外部每个 sprite 就是一个实心矩形，bbox 直接等于目标洞的矩形尺寸；solver 做 exact cover 后原样贴回。
#    - 融合后，外部对象变成“矩形 core + 一个斜对角粘连的单像素角标”。角标不参与输出，只用来指示要把 core
#      旋转多少再填洞。输出中仍然只出现矩形 core，不出现角标。
#    - 小例子：某个输出洞里是一个 2x4 的双色 L 形 patch，但不同 piece 的内部朝向本身也会变化。
#      输入里真正散落在外部的会是该输出 patch 的逆旋转版本，例如 4x2 的 core，再在其右上角外侧放一个 marker。
#      原题会把这个 5x3 的稀疏 bbox 误当成 sprite 本体；
#      融合后 solver 必须先识别出“右上角 marker + 4x2 core”，再按 corner->rotation 规则旋回 2x4，最后再去匹配洞。
# b) Complexity calibration:
#    - 角标只提供 4 个离散档位，对应 0/90/180/270 度，避免自由度过高。
#    - 每个样例强制四种角标都出现，且至少 4 个非正方形矩形，使 90/270 度不只是“图案朝向变化”，还会改变 bbox 尺寸。
#    - 所有 core 都用同一种非旋转对称的双色 L 形母板，但输出中的朝向会混合出现；
#      对非正方形块至少混入 0/180 两种朝向，避免退化成“全部转正再填”。
# c) Hidden-assumption audit against the original main():
#    - 原 solver 假设 source component bbox 本身就是要填的矩形尺寸；现在 source bbox 会额外多一行一列，因此失效。
#    - 原 solver 假设 component 内部是满矩形；现在 component 是稀疏形状，必须先用 mask 识别角标位置并提取实心 core。
#    - 原 solver 的候选只按输入 bbox 尺寸枚举；现在需要先旋转 core，再按旋转后的 (h,w) 枚举候选矩形。
#    - 原 solver 不需要从符号里读取额外规则；现在必须先识别 corner，并按固定图例
#      `tl=0, tr=90, br=180, bl=270` 旋回后再去做 exact cover。
# d) Placement legality:
#    - 角标与 core 仅对角接触，因此整个 source sprite 仍是单个 8 连通 component。
#    - 模板内 hole 仍由一串互相搭接的矩形组成，保证 hole 区域连通且可被矩形精确覆盖。
#    - 外部散布使用 `transparent=bg` 的随机放置，因此稀疏 bbox 不会用背景像素覆盖别的对象。
# e) Visual observability:
#    - 规则完全由人眼可见的“角标在哪个角”和“图案被转回后才能匹配洞尺寸/朝向”决定，不用颜色编号排序。
#      因为目标洞中的图案朝向并不统一，解题者不能忽略角标、只靠把所有 core 统一转成某个标准朝向。
#    - 每个样例至少出现一次 tl/tr/br/bl 四种角标；非正方形块上的 90/270 度会把 2x4 与 4x2 互换，映射关系可直接观察。
# f) Visual cleanliness:
#    - 角标固定放在矩形 core 外侧一个角，不遮挡 core 图案；模板条带和外部 sprite 之间仍保留背景间隔。
#    - 图案层次只有“template / hole / core / single-pixel marker”四层，视觉上清楚，不会混成杂乱块。
# g) Change inferability:
#    - 颜色不承载隐藏语义；角标颜色在同一网格内共享，只表示“这是指令像素”，真正的角度信息来自位置。
#      core 的双色只负责让朝向可见，但不会单独决定一个全局“标准朝向”。
#    - 生成器要求四种角标都出现，并要求多个非零旋转样本，所以规则不会退化成“原样填回也行”。
# h) New-main assumption audit:
#    - `_extract_symbol_sprite()` 假设每个 source component 都恰好是“一个矩形 core + 一个角标”；生成器按该模板构造。
#    - `_solve_with_offset()` 假设旋转后的每块面积总和等于 hole 面积；生成器仍保持 exact-cover 面积守恒。
#    - `main()` 假设整套任务都共享同一个固定角标图例；generator 在每个样本里反复使用四种 corner，保证该图例可从样例学习。

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 10
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 6, 8
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_RECT_LEN = 2
MIN_RECT_WIDTH = 2
MIN_DISTINCT_SHAPES = 3
MIN_NON_SQUARE = 4
WIDTH_TOLERANCE = 2
MAX_RETRIES = 500

CORNERS = ("tl", "tr", "br", "bl")


def _corner_index(corner: str) -> int:
    return CORNERS.index(str(corner))


def _corner_to_rotation(corner: str, offset: int = 0) -> int:
    """Return the CCW quarter-turn count implied by a corner under a cyclic legend offset."""
    return (_corner_index(corner) + int(offset)) % 4


def _corner_bbox_spec(ph: int, pw: int, corner: str) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    if corner == "tl":
        return (0, 0), (1, 1, ph - 1, pw - 1)
    if corner == "tr":
        return (0, pw - 1), (1, 0, ph - 1, pw - 2)
    if corner == "br":
        return (ph - 1, pw - 1), (0, 0, ph - 2, pw - 2)
    if corner == "bl":
        return (ph - 1, 0), (0, 1, ph - 2, pw - 1)
    raise ValueError(f"unknown corner {corner}")


def _extract_symbol_sprite(patch: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, str]:
    """Parse a sparse corner-marked component into its solid rectangular core and marker corner."""
    ph, pw = map(int, patch.shape)
    assert ph >= 3 and pw >= 3
    for corner in CORNERS:
        (mr, mc), (r0, c0, r1, c1) = _corner_bbox_spec(ph, pw, corner)
        expected = np.zeros((ph, pw), dtype=bool)
        expected[r0 : r1 + 1, c0 : c1 + 1] = True
        expected[mr, mc] = True
        if np.array_equal(mask, expected):
            core = patch[r0 : r1 + 1, c0 : c1 + 1].copy()
            return core, corner
    raise AssertionError("source component is not a valid corner-marked rectangle")


def _candidate_positions(hole_mask: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    dh, dw = map(int, hole_mask.shape)
    pos: list[tuple[int, int]] = []
    if h > dh or w > dw:
        return pos
    for rr0 in range(dh - h + 1):
        for cc0 in range(dw - w + 1):
            if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                pos.append((int(rr0), int(cc0)))
    return pos


def _extract_problem(grid: np.ndarray) -> tuple[int, np.ndarray, list[dict[str, object]]]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int]] = []
    for col in colors_present(g, exclude={bg}):
        max_area = 0
        for coords in components_mask(g == int(col), connectivity=8):
            max_area = max(int(max_area), int(coords.shape[0]))
        base_candidates.append((int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][1])

    base_coords = np.argwhere(g == int(base_col))
    assert base_coords.size
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst = crop_bbox(g, (int(r0), int(c0), int(r1), int(c1)))
    hole_mask = dst != int(base_col)

    masked = g.copy()
    masked[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict[str, object]] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        mask = np.zeros_like(patch, dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        core, corner = _extract_symbol_sprite(patch, mask)
        sprites.append({"core": core, "corner": corner})

    assert sprites
    return int(base_col), hole_mask, sprites


def _solve_with_offset(
    sprites: list[dict[str, object]],
    hole_mask: np.ndarray,
    base_col: int,
    offset: int,
) -> np.ndarray | None:
    cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
    transformed: list[dict[str, object]] = []
    total_area = 0
    for sp in sprites:
        rot = _corner_to_rotation(str(sp["corner"]), offset)
        patch = np.rot90(np.array(sp["core"], dtype=int), k=int(rot))
        h, w = map(int, patch.shape)
        total_area += h * w
        key = (h, w)
        if key not in cache:
            cache[key] = _candidate_positions(hole_mask, h, w)
        if not cache[key]:
            return None
        transformed.append({"patch": patch, "h": h, "w": w, "cands": cache[key]})

    if total_area != int(hole_mask.sum()):
        return None

    order = sorted(
        range(len(transformed)),
        key=lambda i: (len(transformed[i]["cands"]), -int(transformed[i]["h"]) * int(transformed[i]["w"])),
    )
    used = ~hole_mask.copy()
    placement: list[tuple[int, int] | None] = [None] * len(transformed)

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        i = order[k]
        h, w = int(transformed[i]["h"]), int(transformed[i]["w"])
        for rr0, cc0 in transformed[i]["cands"]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(k + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    if not dfs(0):
        return None

    out = np.full(hole_mask.shape, int(base_col), dtype=int)
    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        paste(out, np.array(transformed[i]["patch"], dtype=int), rr0, cc0, overwrite=True)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild the strip template by reading a corner instruction and rotating each scattered core before insertion.

    Input structure:
    - The dominant non-background color forms a strip-shaped template whose hole region is still an exact cover of axis-aligned rectangles.
    - Every outside sprite is sparse: a solid rectangular core plus one diagonally attached corner marker pixel sitting just outside that core.

    Rule:
    - Detect the template as before, but parse each source component into its core rectangle and marker corner instead of treating the whole bbox as a hole-sized block.
    - Interpret the marker corner as a quarter-turn instruction, rotate the core accordingly, and exact-cover the hole rectangles with those rotated cores.
    - Return the tight template bbox filled with the rotated cores only; the corner markers never appear in the output.

    Key insight:
    - The puzzle is no longer solved by raw bbox matching: the visible corner tag first changes each sprite's orientation, so matching sizes only works after applying the fixed corner-to-rotation legend.
    """
    base_col, hole_mask, sprites = _extract_problem(np.array(grid, dtype=int))
    out = _solve_with_offset(sprites, hole_mask, base_col, offset=0)
    assert out is not None, "no valid fill under the canonical corner legend"
    return out


def _split_lengths(total: int, count: int) -> list[int] | None:
    if count <= 0 or count * MIN_RECT_LEN > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [MIN_RECT_LEN] * count
    remaining = total - count * MIN_RECT_LEN
    while remaining:
        choices = [i for i, v in enumerate(lengths) if v < MAX_RECT_LEN]
        if not choices:
            return None
        i = random.choice(choices)
        add = random.randint(1, min(remaining, MAX_RECT_LEN - lengths[i]))
        lengths[i] += add
        remaining -= add
    random.shuffle(lengths)
    return lengths


def _choose_widths(lengths: list[int], thick: int, full_area: int) -> list[int] | None:
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    if max_w < MIN_RECT_WIDTH:
        return None
    lo = math.ceil(HOLE_FRAC_LO * full_area)
    hi = math.floor(HOLE_FRAC_HI * full_area)
    paths: dict[int, list[int]] = {0: []}
    for h in lengths:
        opts = list(range(MIN_RECT_WIDTH, max_w + 1))
        random.shuffle(opts)
        items = list(paths.items())
        random.shuffle(items)
        nxt: dict[int, list[int]] = {}
        for cur, path in items:
            for w in opts:
                s = cur + h * w
                if s <= hi and s not in nxt:
                    nxt[s] = path + [w]
        paths = nxt
        if not paths:
            return None
    valid = [(s, p) for s, p in paths.items() if lo <= s <= hi]
    if not valid:
        return None
    target = random.randint(lo, hi)
    best = min(abs(s - target) for s, _ in valid)
    near = [p for s, p in valid if abs(s - target) <= best + WIDTH_TOLERANCE]
    return random.choice(near)


def _corridor_rects(long: int, thick: int, count: int) -> list[tuple[int, int, int, int]] | None:
    min_len = max(count * MIN_RECT_LEN, round(CORRIDOR_COVERAGE * long))
    min_len = min(min_len, long - 1)
    if min_len > long - 1:
        return None
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, long * thick)
    if widths is None:
        return None

    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int]] = []
    for h, w in zip(lengths, widths):
        c_hi = thick - w - 1
        if c_hi < 1:
            return None
        if prev_c is None:
            c = random.randint(1, c_hi)
        else:
            c_lo = max(1, prev_c - w + 1)
            c_hi = min(c_hi, prev_c + prev_w - 1)
            if c_lo > c_hi:
                return None
            c = random.randint(c_lo, c_hi)
        rects.append((r, c, h, w))
        r += h
        prev_c, prev_w = c, w
    return rects


def _make_oriented_patch(h: int, w: int, primary: int, secondary: int) -> np.ndarray:
    out = np.full((h, w), int(primary), dtype=int)
    out[0, :] = int(secondary)
    out[:, 0] = int(secondary)
    if h > 2 and w > 2:
        out[1, -1] = int(secondary)
    return out


def _build_symbol_sprite(core: np.ndarray, marker: int, corner: str, bg: int) -> np.ndarray:
    h, w = map(int, core.shape)
    out = np.full((h + 1, w + 1), int(bg), dtype=int)
    if corner == "tl":
        out[0, 0] = int(marker)
        out[1:, 1:] = core
    elif corner == "tr":
        out[0, w] = int(marker)
        out[1:, :w] = core
    elif corner == "br":
        out[:h, :w] = core
        out[h, w] = int(marker)
    elif corner == "bl":
        out[:h, 1:] = core
        out[h, 0] = int(marker)
    else:
        raise ValueError(f"unknown corner {corner}")
    return out


def _assign_color_pairs(n: int, bg: int, base: int, marker: int) -> list[tuple[int, int]] | None:
    available = [c for c in range(10) if c not in (bg, base, marker)]
    if len(available) < 4:
        return None
    pairs = [(a, b) for a in available for b in available if a != b]
    if len(pairs) < n:
        return None
    return random.sample(pairs, n)


def _build_piece_sets(
    rects: list[tuple[int, int, int, int]],
    bg: int,
    base: int,
    marker: int,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray], list[int]] | None:
    if len({(h, w) for _, _, h, w in rects}) < MIN_DISTINCT_SHAPES:
        return None

    non_square = [i for i, (_, _, h, w) in enumerate(rects) if h != w]
    if len(non_square) < MIN_NON_SQUARE:
        return None

    color_pairs = _assign_color_pairs(len(rects), bg, base, marker)
    if color_pairs is None:
        return None

    pattern_rotations: list[int] = []
    non_square_pattern_rotations: set[int] = set()
    for _, _, h, w in rects:
        allowed = (0, 2) if h != w else (0, 1, 2, 3)
        rot = random.choice(allowed)
        pattern_rotations.append(int(rot))
        if h != w:
            non_square_pattern_rotations.add(int(rot))
    if len(non_square_pattern_rotations) < 2:
        return None

    out_pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w) in enumerate(rects):
        primary, secondary = color_pairs[i]
        patch = np.rot90(_make_oriented_patch(h, w, primary, secondary), k=int(pattern_rotations[i]))
        assert all(not np.array_equal(patch, np.rot90(patch, k)) for k in (1, 2, 3))
        out_pieces[i] = patch

    rotations = [random.randrange(4) for _ in rects]
    anchors = random.sample(non_square, 4)
    for rot, idx in enumerate(anchors):
        rotations[idx] = rot

    src_pieces: dict[int, np.ndarray] = {}
    for i in range(len(rects)):
        core = np.rot90(out_pieces[i], k=(4 - int(rotations[i])) % 4)
        src_pieces[i] = _build_symbol_sprite(core, marker, CORNERS[int(rotations[i])], bg)
    return out_pieces, src_pieces, rotations


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full((dh, dw), int(base), dtype=int)
    for i, (r, c, _, _) in enumerate(rects):
        paste(out, pieces[i], int(r), int(c), overwrite=True)
    return out


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    inp = np.full((H, W), int(bg), dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    for r, c, h, w in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)

    placement = np.full((H, W), int(bg), dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(base)
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].shape[0] * pieces[i].shape[1], reverse=True)
    for i in order:
        patch = pieces[i]
        rr, cc = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, rr, cc, transparent=bg, overwrite=True)
        paste(placement, patch, rr, cc, transparent=bg, overwrite=True)
    return inp


def _gen_attempt() -> np.ndarray | None:
    vertical = random.random() < 0.5
    thick = random.randint(THICK_MIN, THICK_MAX)
    if vertical:
        H = random.randint(GRID_MIN, GRID_MAX)
        W = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        dh, dw = H, thick
        tr0, tc0 = 0, (0 if random.random() < 0.5 else W - thick)
    else:
        H = random.randint(max(GRID_MIN, thick + OUTER_GAP), GRID_MAX)
        W = random.randint(GRID_MIN, GRID_MAX)
        dh, dw = thick, W
        tr0, tc0 = (0 if random.random() < 0.5 else H - thick), 0
    long = dh if vertical else dw

    total = random.randint(SPRITE_COUNT_MIN, SPRITE_COUNT_MAX)
    rects_long = _corridor_rects(long, thick, total)
    if rects_long is None:
        return None
    rects = rects_long if vertical else [(c, r, w, h) for r, c, h, w in rects_long]

    bg, base, marker = random.sample(range(10), 3)
    built = _build_piece_sets(rects, bg, base, marker)
    if built is None:
        return None
    out_pieces, src_pieces, rotations = built

    non_zero = sum(int(rot != 0) for rot in rotations)
    odd_non_square = sum(
        int(rotations[i] in (1, 3))
        for i, (_, _, h, w) in enumerate(rects)
        if h != w
    )
    if non_zero < 4 or odd_non_square < 2:
        return None

    out = _render_output(dh, dw, base, rects, out_pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, src_pieces)
    except Exception:
        return None
    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
