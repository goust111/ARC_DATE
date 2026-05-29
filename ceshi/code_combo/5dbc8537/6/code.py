# Combo variant of 5dbc8537 (index 6)
# combo_spec: D5/0+D6/0
# Name: 就地复制填充（保留源+全网格输出）

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

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


"""
design_analysis:
  a_concretization_and_trace:
    - 原题输入里有一条厚条带模板，内部若干矩形洞，模板外散落若干矩形精灵。原 solver 会把外部精灵按 bbox
      尺寸配对塞回洞里，然后只返回模板 bbox 的紧裁结果，因此外部源精灵与模板外背景都会消失。
    - 融合后仍然做同一套“按矩形尺寸精确覆盖洞区”的配对，但执行动作改成复制而不是搬移：
      先保留整张输入，再把每个外部精灵复制到模板内对应洞中。
    - 具体例子：若一个 2x3 条纹块和一个 1x1 黑点散落在模板外，原题输出只剩填满后的模板条带；
      本题输出仍是原尺寸整图，外部的 2x3 条纹块和黑点还在原位，同时模板中的 2x3 洞和 1x1 洞也被填上。
  b_complexity_calibration:
    - 没有引入新的匹配维度，仍以矩形 bbox 尺寸做唯一拼图依据，避免规则变成过度自由的“任意复制”。
    - 视觉差异足够强：输出尺寸固定等于输入尺寸，且模板外源精灵稳定保留；与原题的裁剪输出明显不同。
    - 生成器保留原题 7-15 个精灵、重复形状组、多色纹理和黑点标记，使“复制填充”发生在足够丰富的对象集上，
      不会退化成单一矩形的平凡拷贝。
  c_implicit_assumption_audit:
    - 原 main() 的 `out = full(base)` 依赖“输出只看模板 bbox”；融合后此假设失效，必须改为 `out = g.copy()`。
    - 原 main() 在模板 bbox 外直接抹成背景再找组件，假设外部精灵不会与模板连通；生成器继续保证模板和精灵之间
      至少隔 1 格背景，所以这个前提仍成立。
    - 原 main() 依赖 `sum(sprite bbox area) == hole area`；融合后仍然需要，因为只是从移动改成复制，配对结构未变。
    - 原 main() 最后粘贴到局部坐标 `out[dh,dw]`；融合后必须改为带上模板偏移量 `(r0, c0)` 粘贴到整图。
  d_placement_legality:
    - 模板仍然占满整行或整列并贴三边，精灵仍只散布在 L 形外区。
    - `random_free_location(..., padding=1)` 继续保证各精灵彼此及与模板至少 1 格背景分离，避免误并连通块。
    - 每个候选样本都经过新 `main()` 回算校验。
  e_visual_observability:
    - 解题者能直接看到输出与输入同尺寸，并且模板外的彩色精灵一个不少地保留下来；同时模板内新出现第二份拷贝。
    - 由于模板本身仍带大面积 base 色框架，新增填充位置与原精灵位置分处不同区域，复制语义比“单纯不裁剪”更明显。
  f_visual_cleanliness:
    - 输出只新增模板内部内容，不在模板外叠加新层，也不在同一区域重复覆盖多个对象。
    - 外部精灵和模板之间始终有背景缓冲，整图虽然更大，但结构仍清晰分成“源区 + 模板区”。
  g_inferability:
    - 没有任何依赖颜色数值大小的隐藏规则；颜色只作为人眼可见的纹理签名帮助区分同尺寸矩形。
    - 关键控制变量只有两个：源是否保留、输出是否全尺寸。两者在每个样例里都会同时显现，并且与原题形成稳定对比。
"""


def _find_base_bbox(g: np.ndarray, bg: int) -> tuple[int, tuple[int, int, int, int]]:
    """Return the template base color and bbox of its full extent."""
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
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    return base_col, (int(r0), int(c0), int(r1), int(c1))


def _extract_sprites(g: np.ndarray, bg: int, dst_bbox: tuple[int, int, int, int]) -> list[dict]:
    """Find all non-bg sprite components outside the template bbox."""
    r0, c0, r1, c1 = map(int, dst_bbox)
    masked = g.copy()
    masked[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        mask = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append({"h": int(ph), "w": int(pw), "mask": mask, "patch": patch})

    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))
    return sprites


def _solve_exact_cover(hole_mask: np.ndarray, sprites: list[dict]) -> list[tuple[int, int]]:
    """Match each sprite bbox to one hole rectangle via DFS exact cover."""
    dh, dw = map(int, hole_mask.shape)
    areas = [int(sp["h"] * sp["w"]) for sp in sprites]
    assert sum(areas) == int(hole_mask.sum())

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
    Copy scattered rectangular sprites into matching template holes while keeping the originals.

    Input structure:
    - The background is the most common color, and one dominant non-background color forms a long strip template with rectangular gaps.
    - Rectangular sprite components sit outside that strip, separated from one another and from the template by background cells.

    Rule:
    - Use the largest non-background connected component color as the template base, and treat its bbox as the destination strip.
    - Match outside sprites to inside holes by exact bounding-box size, paste each sprite into its hole, and return the full grid.
    - The source sprites are not erased; the output is the original scene plus a filled template.

    Key insight:
    - The same size-based jigsaw fit from the original task still determines which piece goes where, but the action is duplication instead of movement and the canvas is not cropped.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col, dst_bbox = _find_base_bbox(g, bg)
    r0, c0, r1, c1 = map(int, dst_bbox)

    dst = crop_bbox(g, dst_bbox)
    hole_mask = dst != int(base_col)
    sprites = _extract_sprites(g, bg, dst_bbox)
    placement = _solve_exact_cover(hole_mask, sprites)

    out = g.copy()
    for sp, (rr0, cc0) in zip(sprites, placement):
        paste(out, sp["patch"], int(r0 + rr0), int(c0 + cc0), mask=sp["mask"], overwrite=True)
    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
THICK_MIN, THICK_MAX = 7, 10
OUTER_GAP = 8
SPRITE_COUNT_MIN, SPRITE_COUNT_MAX = 7, 15
DOT_MIN, DOT_MAX = 1, 4
DOT_NON_DOT_BUFFER = 6
DOT_LONG_BUFFER = 4
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
    """Partition total into count positive integers, each <= MAX_RECT_LEN."""
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


def _choose_widths(lengths: list[int], thick: int, dots: int, full_area: int) -> list[int] | None:
    """Pick widths so the total hole area stays inside the original fraction band."""
    max_w = min(MAX_RECT_WIDTH, thick - 2)
    lo = max(0, math.ceil(HOLE_FRAC_LO * full_area) - dots)
    hi = max(lo, math.floor(HOLE_FRAC_HI * full_area) - dots)
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
                area = cur + h * w
                if area <= hi and area not in nxt:
                    nxt[area] = path + [w]
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


def _place_dot_markers(
    rects: list[tuple[int, int, int, int, bool]],
    occupied: np.ndarray,
    thick: int,
    dots: int,
) -> list[tuple[int, int, int, int, bool]] | None:
    """Attach dot-sized rectangles adjacent to existing corridor rectangles."""
    for _ in range(dots):
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
        rects.append((rr, cc, 1, 1, True))
    return rects


def _corridor_rects(long: int, thick: int, count: int, dots: int) -> list[tuple[int, int, int, int, bool]] | None:
    """Generate a connected run of rectangles along the long axis."""
    min_len = max(count + 2, round(CORRIDOR_COVERAGE * long))
    if min_len > long - 1:
        min_len = long - 1
    hole_len = random.randint(min_len, long - 1)
    lengths = _split_lengths(hole_len, count)
    if lengths is None:
        return None
    widths = _choose_widths(lengths, thick, dots, long * thick)
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
        rects.append((r, c, h, w, False))
        r += h
        prev_c, prev_w = c, w

    occupied = np.zeros((long, thick), dtype=bool)
    for r0, c0, h, w, _ in rects:
        occupied[r0 : r0 + h, c0 : c0 + w] = True

    return _place_dot_markers(rects, occupied, thick, dots)


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Fill a rectangle with one or two colors using a simple structured pattern."""
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out

    other = cols[1]
    pattern = random.choice(["checker", "dots", "split"])
    rr, cc = np.indices((h, w))
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = other
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = other
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = other
    else:
        out[:, random.randint(1, w - 1) :] = other
    return out


def _make_pieces(
    rects: list[tuple[int, int, int, int, bool]],
    bg: int,
    base: int,
) -> dict[int, np.ndarray] | None:
    """Assign colors and textures while keeping same-size sprites interchangeable."""
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if not is_dot:
            groups.setdefault((h, w), []).append(i)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None

    available = [c for c in range(1, 10) if c not in (bg, base)]
    max_colors = min(len(available), 2 * len(groups))
    if max_colors < MIN_PALETTE_SIZE:
        return None
    palette = random.sample(available, random.randint(MIN_PALETTE_SIZE, max_colors))

    keys = list(groups)
    random.shuffle(keys)
    needed_bi = max(0, len(palette) - len(keys))
    bi_count = min(len(keys), max(needed_bi, round(BICHROMATIC_FRAC * len(keys))))
    bi_keys = set(random.sample(keys, bi_count))

    unused = palette[:]
    random.shuffle(unused)
    group_cols: dict[tuple[int, int], list[int]] = {}
    for key in keys:
        group_cols[key] = [unused.pop() if unused else random.choice(palette)]
    for key in bi_keys:
        primary = group_cols[key][0]
        second = unused.pop() if unused else random.choice([c for c in palette if c != primary])
        group_cols[key].append(second)

    patch_by_shape: dict[tuple[int, int], np.ndarray] = {}
    pieces: dict[int, np.ndarray] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if is_dot:
            pieces[i] = np.array([[0]], dtype=int)
            continue
        key = (h, w)
        if key not in patch_by_shape:
            patch_by_shape[key] = _make_patch(h, w, group_cols[key])
        pieces[i] = patch_by_shape[key].copy()
    return pieces


def _scatter_sprites(
    H: int,
    W: int,
    bg: int,
    base: int,
    tr0: int,
    tc0: int,
    dh: int,
    dw: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    """Build the input grid with a holed template and scattered sprite sources."""
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for r, c, h, w, _ in rects:
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg

    placement = np.full((H, W), bg, dtype=int)
    placement[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    order = list(range(len(rects)))
    random.shuffle(order)
    order.sort(key=lambda i: pieces[i].size, reverse=True)
    for i in order:
        patch = pieces[i]
        r, c = random_free_location(placement, patch, transparent=bg, padding=1, padding_connectivity=8)
        paste(inp, patch, r, c)
        paste(placement, patch, r, c)
    return inp


def _render_expected_output(
    inp: np.ndarray,
    tr0: int,
    tc0: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    """Expected combo output: keep full input, then fill the template holes in place."""
    out = inp.copy()
    for i, (r, c, _, _, _) in enumerate(rects):
        paste(out, pieces[i], int(tr0 + r), int(tc0 + c))
    return out


def _gen_attempt() -> np.ndarray | None:
    """Sample one instance, then oracle-check it against the fused solver."""
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
    min_dots = max(DOT_MIN, total - (long - DOT_LONG_BUFFER))
    max_dots = min(DOT_MAX, total - DOT_NON_DOT_BUFFER)
    if min_dots > max_dots:
        return None
    dots = random.randint(min_dots, max_dots)

    canon = _corridor_rects(long, thick, total - dots, dots)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h, d) for r, c, h, w, d in canon]

    bg, base = random.sample(range(1, 10), 2)
    pieces = _make_pieces(rects, bg, base)
    if pieces is None:
        return None

    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces)
        expected = _render_expected_output(inp, tr0, tc0, rects, pieces)
        solved = main(inp)
    except Exception:
        return None

    if not np.array_equal(solved, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
