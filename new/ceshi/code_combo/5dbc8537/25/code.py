# Combo variant of 5dbc8537 (index 25)
# combo_spec: D5/2+D6/0+D11/0
# Name: 源位置标记替换+全网格输出+固定基底色

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import bbox_coords, crop_bbox, find_components, get_bg, paste, random_free_location


"""
design_analysis:
  a_concretization_and_trace:
    - 原题的动作是“移动并裁剪”：模板外的彩色矩形精灵被塞进模板洞里，输出只保留模板 bbox，外部源位置完全消失。
    - 本融合变体把三处同时改掉：
      1. D11/0: 模板基底色不再靠“最大连通块颜色”推断，而是固定为灰色 5。
      2. D6/0: 输出不再裁剪，整张输入画布都要保留。
      3. D5/2: 每个源精灵移入洞后，原地不变成背景，而是被统一的标记色实心矩形覆盖。
    - 具体例子：若输入右边有一条灰色竖模板，洞里缺一个 2x3 条纹块和一个 1x1 小块，左侧外区散落这两个精灵。
      原题输出只会给出被填满的灰色模板紧裁图；本题输出仍是整张 HxW 网格，模板内新增那两个精灵，
      而左侧原来两个源位置分别变成同尺寸的纯标记色矩形。
  b_complexity_calibration:
    - 我保留原题最重要的难点：洞区仍由多个矩形 bbox 完全铺满，solver 仍需做精确覆盖搜索，而不是按颜色直接配对。
    - 为避免规则退化成“只是全图输出”，输出里额外引入统一标记矩形，并且输入中完全禁用该标记色；
      这样解题者能直接看见“源被替换”，而不是把变化误读为简单复制或简单不裁剪。
    - 为避免规则退化成“只是固定灰色模板”，生成器仍保留 7-15 个矩形、重复尺寸组、双色纹理和若干 1x1 小块。
  c_implicit_assumption_audit:
    - 原 `main()` 先从所有非背景色里挑最大连通块颜色作为 base。D11/0 让这个假设失效；新 solver 直接使用固定灰色 5，
      并要求生成器不把 5 用在任何精灵或背景上。
    - 原 `hole_mask = dst != base_col` 隐含“模板 bbox 内除 base 外只有洞”。本变体仍保留这一点，生成器保证输入模板内部只有 base 和 bg。
    - 原 `out = full(base)` 假设输出只看模板局部。D6/0 后必须改成整图 `out = g.copy()`，并在全局坐标上粘贴洞内精灵。
    - 原 solver 不关心源位置后续如何处理。D5/2 新增假设：每个源组件的 bbox 是实心矩形，便于整体替换成统一标记矩形。
      生成器继续只产生“bbox 内无背景孔洞”的矩形精灵，因此这一点成立。
    - 原 `find_components(masked, monochromatic=False)` 依赖模板与源精灵至少隔 1 格背景，避免误并。生成器继续沿用这一分离约束。
  d_placement_legality:
    - 模板仍沿整行或整列铺开，并贴住三条边；外部精灵只落在剩余的 L 形外区。
    - `random_free_location(..., padding=1)` 保证精灵彼此以及与模板之间至少隔 1 格背景，不重叠、不越界、不会被 8 邻接并成一团。
    - 标记矩形只出现在输出，不参与输入对象放置，因此不会破坏输入可读性。
  e_visual_observability:
    - 固定灰色模板让“哪一块是承载洞的框架”一眼可见，不必靠面积推断。
    - 输出与输入同尺寸，且模板外所有源位置都变成统一色块；这一变化在 2-3 个样例里就足够稳定、显眼。
    - 标记色在输入中完全缺席，解题者能把它理解成“已使用/已搬走”的状态，而不是普通第三类对象。
  f_visual_cleanliness:
    - 输出新增内容分成两个清晰区：模板内是彩色填充结果，模板外是纯色标记矩形；没有轨迹、叠影或多层覆盖。
    - 灰色基底、彩色精灵、纯黑标记三类视觉角色完全分离，避免外部区域看起来脏乱。
  g_inferability:
    - 颜色规则只有两条，而且都具备结构提示：灰色永远是模板，黑色永远只出现在输出中的源替换处。
    - 不使用“颜色数值大小”“黑色等于 0 所以特殊”之类人类不可见信息。即使解题者不知道颜色编号，也能从样例视觉上推断规则。
  h_new_solver_assumptions_and_generator_support:
    - solver 假设灰色像素集合的 bbox 就是唯一模板 bbox；生成器保证灰色只用于模板。
    - solver 假设模板外所有非背景组件都是待填矩形精灵；生成器不在外区额外放置装饰或标记。
    - solver 假设源精灵 bbox 面积之和等于洞面积，且这些 bbox 能在洞区中做无重叠精确覆盖；生成器继续采用 output-first 构造并用 solver 回验。
"""


FIXED_BASE = 5
SOURCE_MARK = 0


def _template_bbox(g: np.ndarray) -> tuple[int, int, int, int]:
    base_coords = np.argwhere(g == int(FIXED_BASE))
    assert base_coords.size > 0
    return tuple(map(int, bbox_coords(base_coords)))


def _extract_sprites(g: np.ndarray, bg: int, tpl_bbox: tuple[int, int, int, int]) -> list[dict]:
    tr0, tc0, tr1, tc1 = map(int, tpl_bbox)
    masked = g.copy()
    masked[tr0 : tr1 + 1, tc0 : tc1 + 1] = int(bg)
    comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in comps:
        sr0, sc0, sr1, sc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (sr0, sc0, sr1, sc1))
        h, w = map(int, patch.shape)
        assert np.all(patch != int(bg))
        mask = np.ones((h, w), dtype=bool)
        sprites.append(
            {
                "bbox": (int(sr0), int(sc0), int(sr1), int(sc1)),
                "h": int(h),
                "w": int(w),
                "patch": patch,
                "mask": mask,
            }
        )

    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["h"]), -int(s["w"])))
    return sprites


def _solve_exact_cover(hole_mask: np.ndarray, sprites: list[dict]) -> list[tuple[int, int]]:
    dh, dw = map(int, hole_mask.shape)
    assert int(sum(int(sp["h"] * sp["w"]) for sp in sprites)) == int(hole_mask.sum())

    candidates: list[list[tuple[int, int]]] = []
    for sp in sprites:
        h, w = int(sp["h"]), int(sp["w"])
        pos: list[tuple[int, int]] = []
        if h <= dh and w <= dw:
            for rr0 in range(int(dh - h + 1)):
                for cc0 in range(int(dw - w + 1)):
                    if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                        pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)
    used = ~hole_mask
    placement: list[tuple[int, int] | None] = [None] * len(sprites)

    def dfs(i: int) -> bool:
        if i == len(sprites):
            return True

        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(i + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)
    assert all(pos is not None for pos in placement)
    return [tuple(map(int, pos)) for pos in placement if pos is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill a fixed-gray strip template with matching outside rectangles, then replace each source with a marker block.

    Input structure:
    - The background is the most common color, while gray cells form the only large strip-shaped template.
    - Inside that gray template bbox, the missing regions are background-colored rectangular holes.
    - Outside the template, separated colored rectangles act as source sprites, and the source-mark color does not appear yet.

    Rule:
    - Treat gray as the fixed template base color and use the bbox of all gray pixels as the destination strip.
    - Match each outside rectangle to a same-sized hole rectangle inside the strip and paste the sprite into that hole.
    - Return the full original-size grid: keep the surrounding background, keep the filled template in place, and replace every source bbox by a solid marker rectangle.

    Key insight:
    - The geometric matching rule is still the original size-based exact cover, but the fused variant changes both identification and rendering: gray is predefined as the frame, the canvas is not cropped, and used source locations are turned into uniform marker blocks.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    tpl_bbox = _template_bbox(g)
    tr0, tc0, tr1, tc1 = map(int, tpl_bbox)

    tpl = crop_bbox(g, tpl_bbox)
    hole_mask = tpl == int(bg)
    sprites = _extract_sprites(g, bg, tpl_bbox)
    placement = _solve_exact_cover(hole_mask, sprites)

    out = g.copy()
    for sp, (rr0, cc0) in zip(sprites, placement):
        paste(out, sp["patch"], int(tr0 + rr0), int(tc0 + cc0), mask=sp["mask"], overwrite=True)

    for sp in sprites:
        sr0, sc0, sr1, sc1 = map(int, sp["bbox"])
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = int(SOURCE_MARK)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 15
UNIT_MIN, UNIT_MAX = 1, 4
UNIT_NON_UNIT_BUFFER = 6
UNIT_LONG_BUFFER = 4
HOLE_FRAC_LO, HOLE_FRAC_HI = 0.39, 0.50
CORRIDOR_COVERAGE = 0.78
MAX_RECT_LEN = 8
MAX_RECT_WIDTH = 6
MIN_SHAPE_GROUPS = 3
MIN_PALETTE_SIZE = 5
BICHROMATIC_FRAC = 0.3
WIDTH_TOLERANCE = 2
MAX_RETRIES = 300


def _split_lengths(total: int, count: int) -> list[int] | None:
    if count <= 0 or count > total or count * MAX_RECT_LEN < total:
        return None
    lengths = [1] * count
    remaining = total - count
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


def _choose_widths(lengths: list[int], thick: int, units: int, full_area: int) -> list[int] | None:
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = max(0, math.ceil(HOLE_FRAC_LO * full_area) - units)
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area) - units)
    paths: dict[int, list[int]] = {0: []}
    for h in lengths:
        opts = list(range(1, max_w + 1))
        if h == 1 and max_w >= 2:
            opts = list(range(2, max_w + 1))
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


def _place_unit_cells(
    rects: list[tuple[int, int, int, int, bool]],
    occupied: np.ndarray,
    thick: int,
    units: int,
) -> list[tuple[int, int, int, int, bool]] | None:
    for _ in range(units):
        choices: list[tuple[int, int]] = []
        for r, c, h, w, _ in rects:
            for rr in range(r, r + h):
                for cc in (c - 1, c + w):
                    if 0 <= cc < thick and not occupied[rr, cc]:
                        choices.append((rr, cc))
        if not choices:
            return None
        rr, cc = random.choice(choices)
        occupied[rr, cc] = True
        rects.append((int(rr), int(cc), 1, 1, True))
    return rects


def _corridor_rects(long: int, thick: int, count: int, units: int) -> list[tuple[int, int, int, int, bool]] | None:
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, units, long * thick)
    if widths is None:
        return None

    start = random.randint(0, long - hole_len)
    r = start
    prev_c = prev_w = None
    rects: list[tuple[int, int, int, int, bool]] = []
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
        rects.append((int(r), int(c), int(h), int(w), False))
        r += h
        prev_c, prev_w = c, w

    occupied = np.zeros((long, thick), dtype=bool)
    for r0, c0, h, w, _ in rects:
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return _place_unit_cells(rects, occupied, thick, units)


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out

    b = cols[1]
    pattern = random.choice(["checker", "dots", "split"])
    rr, cc = np.indices((h, w))
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = b
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = b
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = b
    else:
        out[:, random.randint(1, w - 1) :] = b
    return out


def _make_pieces(rects: list[tuple[int, int, int, int, bool]], bg: int) -> dict[int, np.ndarray] | None:
    non_unit_groups: set[tuple[int, int]] = set()
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w, is_unit) in enumerate(rects):
        key = (h, w)
        groups.setdefault(key, []).append(i)
        if not is_unit:
            non_unit_groups.add(key)
    if len(non_unit_groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, FIXED_BASE, SOURCE_MARK)]
    max_colors = min(len(available), 2 * len(groups))
    if max_colors < MIN_PALETTE_SIZE:
        return None
    palette = random.sample(available, random.randint(MIN_PALETTE_SIZE, max_colors))

    keys = list(groups)
    random.shuffle(keys)
    bi_candidates = [key for key in keys if key != (1, 1)]
    needed_bi = max(0, len(palette) - len(keys))
    bi_count = min(len(bi_candidates), max(needed_bi, round(BICHROMATIC_FRAC * len(bi_candidates))))
    bi_keys = set(random.sample(bi_candidates, bi_count)) if bi_count else set()

    unused = palette[:]
    random.shuffle(unused)
    group_cols: dict[tuple[int, int], list[int]] = {}
    for key in keys:
        group_cols[key] = [unused.pop() if unused else random.choice(palette)]
    for key in bi_keys:
        primary = group_cols[key][0]
        second_choices = [c for c in palette if c != primary]
        second = unused.pop() if unused else random.choice(second_choices)
        group_cols[key].append(second)

    group_patches: dict[tuple[int, int], np.ndarray] = {}
    for key in keys:
        h, w = key
        if key == (1, 1):
            group_patches[key] = np.array([[group_cols[key][0]]], dtype=int)
        else:
            group_patches[key] = _make_patch(h, w, group_cols[key])

    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w, _) in enumerate(rects):
        pieces[i] = np.array(group_patches[(h, w)], copy=True)
    return pieces


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(FIXED_BASE)
    for r, c, h, w, _ in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = int(bg)

    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = int(FIXED_BASE)

    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)

    src_bboxes: list[tuple[int, int, int, int] | None] = [None] * len(rects)
    for i in order:
        patch = pieces[i]
        r0, c0 = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        h, w = map(int, patch.shape)
        paste(inp, patch, r0, c0)
        paste(placement, patch, r0, c0)
        src_bboxes[i] = (int(r0), int(c0), int(r0 + h - 1), int(c0 + w - 1))

    assert all(b is not None for b in src_bboxes)
    return inp, [tuple(map(int, b)) for b in src_bboxes if b is not None]


def _render_expected_output(
    inp: np.ndarray,
    tr0: int,
    tc0: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
    src_bboxes: list[tuple[int, int, int, int]],
) -> np.ndarray:
    out = np.array(inp, copy=True)
    for i, (r, c, _, _, _) in enumerate(rects):
        paste(out, pieces[i], int(tr0 + r), int(tc0 + c), overwrite=True)
    for sr0, sc0, sr1, sc1 in src_bboxes:
        out[sr0 : sr1 + 1, sc0 : sc1 + 1] = int(SOURCE_MARK)
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
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
    min_units = max(UNIT_MIN, total - (long - UNIT_LONG_BUFFER))
    max_units = min(UNIT_MAX, total - UNIT_NON_UNIT_BUFFER)
    if min_units > max_units:
        return None
    units = random.randint(min_units, max_units)

    canon = _corridor_rects(long, thick, total - units, units)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h, is_unit) for r, c, h, w, is_unit in canon]

    bg = random.choice([1, 2, 3, 4, 6, 7, 8, 9])
    pieces = _make_pieces(rects, bg)
    if pieces is None:
        return None

    try:
        inp, src_bboxes = _scatter_sprites(H, W, bg, tr0, tc0, dh, dw, rects, pieces)
        out = _render_expected_output(inp, tr0, tc0, rects, pieces, src_bboxes)
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, out):
        return None

    meta = {
        "grid_shape": (int(H), int(W)),
        "template_bbox": (int(tr0), int(tc0), int(tr0 + dh - 1), int(tc0 + dw - 1)),
        "sprite_count": int(len(rects)),
        "marker_area": int(sum((sr1 - sr0 + 1) * (sc1 - sc0 + 1) for sr0, sc0, sr1, sc1 in src_bboxes)),
        "hole_area": int(sum(piece.size for piece in pieces.values())),
        "base_color": int(FIXED_BASE),
        "marker_color": int(SOURCE_MARK),
    }
    assert meta["marker_area"] == meta["hole_area"]
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        sample = _gen_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate valid fused 5dbc8537 input")
