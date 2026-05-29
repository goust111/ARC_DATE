# Variant of 5dbc8537
# Dimension: D12 — 精灵放置歧义性与搜索策略 (meta)
# Change: 使用DFS回溯搜索进行非重叠精灵放置，精灵按面积降序处理，每个精灵可能有多个候选位置但最终有唯一解... → 多步消歧

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

# Design analysis
# spec_variant:
#   visual:
#     - 仍然是一个沿整行或整列延伸的 base 色模板条带，内部由若干矩形槽位拼成连通洞区，外部散落矩形精灵。
#     - 与原题不同，重复尺寸的槽位不再全部是纯 bg 空洞；这些槽位内部会预填 1-3 个彩色线索像素。
#     - 线索像素直接取自目标精灵在相同相对坐标处的颜色，因此视觉上像“槽位里提前露出了一小部分图案”。
#   flow_trace:
#     - 原逻辑：同尺寸 sprite -> 枚举所有同尺寸 hole rectangle -> 只靠 DFS 的全局非重叠约束决定放置。
#     - 变体逻辑：同尺寸 sprite 先得到同一批几何候选；然后读取每个候选槽位里已露出的彩色像素，只保留与 sprite 局部颜色一致的候选；最后再做非重叠覆盖。
#     - 小例子：若有两个 3x2 精灵 A/B，对应洞区里也有两个 3x2 候选窗口，原题中 A/B 只看尺寸会互换；现在左槽位预填 A 的左上角和右下角颜色，右槽位预填 B 的对应颜色，因此尺寸筛选后还需再做一次颜色/模式匹配。
#   complexity_calibration:
#     - 只对“重复尺寸组”添加线索，唯一尺寸组仍保持原任务风格，避免把所有洞都变成直接拷贝题。
#     - 线索数量控制在 1-3 个，既足够消歧，又不会把整块图案提前展示出来。
#     - 生成器强制至少一个非 dot 重复尺寸组，并要求至少两个精灵满足“尺寸候选数 > 1 且线索过滤后唯一”，确保 D12 行为稳定出现。
#   hidden_assumptions_review:
#     - 原 `base` 识别依赖“最大非 bg 连通块就是模板”。线索像素很少，不会超过模板条带面积，因此该前提仍成立。
#     - 原 `hole_mask = dst != base` 默认洞区全是 bg。变体中洞区含 bg 与彩色线索，但“非 base 都属于洞区”仍成立，因此这行可保留其语义。
#     - 原 `sum(sprite areas) == hole_mask.sum()` 依赖每个洞格都属于某个 sprite。线索像素只是预填的 sprite 像素，仍占用洞格，所以面积守恒不变。
#     - 原“候选只按尺寸”在变体中失效，因为重复尺寸 sprite 可以交换位置而仍满足几何覆盖；因此必须新增线索一致性过滤。
#     - 原最终 `paste(... overwrite=True)` 在变体中仍安全，因为模板内线索像素就是目标 sprite 的子图，覆盖后不会引入冲突。
#   placement_legality:
#     - 所有 sprite 仍是完整矩形、互不接触地散落在模板外。
#     - 模板中的每个槽位仍是 axis-aligned rectangle；线索只改槽位内部颜色，不改 bbox 和面积。
#     - 生成器在输出先行构造后，会反推输入并再次调用 `main()` 验证。
#   observability:
#     - 人类可先看到“有几块外部精灵尺寸相同”，再看到模板中相同尺寸的槽位里藏着不同颜色的小碎片，从而推断必须两轮匹配。
#     - 所有判定都基于可见的局部颜色/图案，不使用颜色 id 的排序或其他不可见元信息。
#   cleanliness:
#     - 线索像素只出现在槽位内部，不会叠到模板边框或外部散点上；外部 sprite 仍保持 1 格缓冲，视觉分层清楚。
#   color_inference:
#     - 颜色不做全局映射；线索颜色就是目标 sprite 的原色。解题者只需要做“局部同色/同图案对应”，不需要额外学习颜色置换规则。
#   new_main_assumptions:
#     - 对于每个带歧义的尺寸组，至少存在一个共同的线索坐标集合，使不同 sprite 在这些坐标上的颜色元组互不相同。
#     - 线索不会引入 bg/base 之外的新语义：洞区中的非 bg 像素都应被视为“预填的目标图案”。
#     - 生成器通过布局验证保证：重复尺寸组中，真正体现变体的 sprite 具有多个尺寸候选，但恰有一个线索兼容候选。


def _find_base_color(g: np.ndarray, bg: int) -> int:
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
    return int(base_candidates[0][1])


def _extract_sprites(g: np.ndarray, bg: int, dst_bbox: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = map(int, dst_bbox)
    masked = g.copy()
    masked[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (pr0, pc0, pr1, pc1))
        ph, pw = map(int, patch.shape)
        m = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        m[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append(
            {
                "h": int(ph),
                "w": int(pw),
                "mask": m,
                "patch": patch,
            }
        )
    return sprites


def _candidate_positions(hole_mask: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    dh, dw = map(int, hole_mask.shape)
    out: list[tuple[int, int]] = []
    if h > dh or w > dw:
        return out
    for rr0 in range(dh - h + 1):
        for cc0 in range(dw - w + 1):
            if bool(hole_mask[rr0 : rr0 + h, cc0 : cc0 + w].all()):
                out.append((int(rr0), int(cc0)))
    return out


def _clue_compatible(region: np.ndarray, patch: np.ndarray, bg: int) -> bool:
    clue_mask = region != int(bg)
    if not bool(clue_mask.any()):
        return True
    return bool(np.array_equal(region[clue_mask], patch[clue_mask]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the filled template by placing outside sprites into matching interior slots.

    Input structure:
    - The dominant non-background color forms a long rectangular template strip.
    - Outside the strip sit several rectangular sprites; some share the same bounding-box size.
    - Inside the strip, the cavity union is still rectangularly tileable, but repeated-size slots may contain a few prefilled clue pixels.

    Rule:
    - Use the largest non-bg connected component to locate the template bbox.
    - Round 1: for each outside sprite, enumerate every same-sized rectangle fully contained in the non-base cavity region.
    - Round 2: among those geometric candidates, keep only the slots whose visible clue pixels match the sprite's local colors/pattern, then solve the remaining non-overlapping cover and crop to the template bbox.

    Key insight:
    - Size no longer determines a unique slot by itself; it only narrows the search to a repeated-size group.
    - The decisive information is the small colored preview already visible inside each slot, which must agree with the correct sprite.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col = _find_base_color(g, bg)

    base_coords = np.argwhere(g == int(base_col))
    assert not (base_coords.size == 0)
    r0, c0, r1, c1 = bbox_coords(base_coords)
    dst_bbox = (int(r0), int(c0), int(r1), int(c1))

    dst = crop_bbox(g, dst_bbox)
    dh, dw = map(int, dst.shape)
    hole_mask = dst != int(base_col)
    out = np.full((dh, dw), int(base_col), dtype=int)

    sprites = _extract_sprites(g, bg, dst_bbox)
    sprites.sort(key=lambda s: (-int(s["h"] * s["w"]), -int(s["mask"].sum())))

    areas = [int(s["h"] * s["w"]) for s in sprites]
    assert sum(areas) == int(hole_mask.sum())

    positions_by_size: dict[tuple[int, int], list[tuple[int, int]]] = {}
    candidates: list[list[tuple[int, int]]] = []
    for sp in sprites:
        size = (int(sp["h"]), int(sp["w"]))
        if size not in positions_by_size:
            positions_by_size[size] = _candidate_positions(hole_mask, *size)
        pos: list[tuple[int, int]] = []
        h, w = size
        for rr0, cc0 in positions_by_size[size]:
            region = dst[rr0 : rr0 + h, cc0 : cc0 + w]
            if _clue_compatible(region, sp["patch"], bg):
                pos.append((int(rr0), int(cc0)))
        candidates.append(pos)

    assert all(candidates)

    used = ~hole_mask
    placement: list[tuple[int, int] | None] = [None] * len(sprites)
    order = sorted(
        range(len(sprites)),
        key=lambda i: (
            len(candidates[i]),
            -int(sprites[i]["h"] * sprites[i]["w"]),
            -int((dst[candidates[i][0][0] : candidates[i][0][0] + sprites[i]["h"], candidates[i][0][1] : candidates[i][0][1] + sprites[i]["w"]] != bg).sum())
            if candidates[i]
            else 0,
        ),
    )

    def dfs(k: int) -> bool:
        if k == len(order):
            return True
        i = int(order[k])
        h, w = int(sprites[i]["h"]), int(sprites[i]["w"])
        for rr0, cc0 in candidates[i]:
            if bool(used[rr0 : rr0 + h, cc0 : cc0 + w].any()):
                continue
            used[rr0 : rr0 + h, cc0 : cc0 + w] = True
            placement[i] = (int(rr0), int(cc0))
            if dfs(k + 1):
                return True
            placement[i] = None
            used[rr0 : rr0 + h, cc0 : cc0 + w] = False
        return False

    assert dfs(0)

    for i, pos in enumerate(placement):
        assert pos is not None
        rr0, cc0 = map(int, pos)
        sp = sprites[i]
        paste(out, sp["patch"], rr0, cc0, mask=sp["mask"], overwrite=True)

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

MIN_DUPLICATE_GROUPS = 1
MIN_RESOLVED_AMBIGUOUS = 2
CLUE_COUNT_MAX = 3


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


def _choose_widths(lengths: list[int], thick: int, dots: int, full_area: int) -> list[int] | None:
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


def _place_dot_markers(
    rects: list[tuple[int, int, int, int, bool]],
    occupied: np.ndarray,
    thick: int,
    dots: int,
) -> list[tuple[int, int, int, int, bool]] | None:
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
    for r, c, h, w, _ in rects:
        occupied[r : r + h, c : c + w] = True
    return _place_dot_markers(rects, occupied, thick, dots)


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), cols[0], dtype=int)
    if len(cols) == 1:
        return out
    b = cols[1]
    rr, cc = np.indices((h, w))
    pattern = random.choice(["checker", "dots", "split"])
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = b
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not bool(mask.any()) or bool(mask.all()):
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = b
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = b
    else:
        out[:, random.randint(1, w - 1) :] = b
    return out


def _shape_groups(rects: list[tuple[int, int, int, int, bool]]) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = {}
    for i, (_, _, h, w, is_dot) in enumerate(rects):
        if not is_dot:
            groups.setdefault((int(h), int(w)), []).append(i)
    return groups


def _signature_positions(h: int, w: int, count: int) -> list[tuple[int, int]]:
    raw = [
        (0, 0),
        (0, w - 1),
        (h - 1, 0),
        (h - 1, w - 1),
        (h // 2, w // 2),
        (0, w // 2),
        (h // 2, 0),
        (h - 1, w // 2),
        (h // 2, w - 1),
    ]
    out: list[tuple[int, int]] = []
    for pos in raw:
        if pos not in out:
            out.append(pos)
        if len(out) >= count:
            break
    assert len(out) >= count
    return out


def _signature_pool(palette: list[int], count: int) -> list[tuple[int, ...]]:
    if count == 1:
        pool = [(a,) for a in palette]
    elif count == 2:
        pool = [(a, b) for a in palette for b in palette]
    else:
        pool = [(a, b, c) for a in palette for b in palette for c in palette]
    random.shuffle(pool)
    return pool


def _random_piece_patch(h: int, w: int, palette: list[int]) -> np.ndarray:
    cols = [random.choice(palette)]
    if h * w > 1 and random.random() < BICHROMATIC_FRAC:
        other = random.choice([c for c in palette if c != cols[0]])
        cols.append(other)
    return _make_patch(h, w, cols)


def _make_pieces_and_clues(
    rects: list[tuple[int, int, int, int, bool]],
    bg: int,
    base: int,
) -> tuple[dict[int, np.ndarray], dict[int, list[tuple[int, int, int]]]] | None:
    groups = _shape_groups(rects)
    if len(groups) < MIN_SHAPE_GROUPS:
        return None
    duplicate_groups = [ids for ids in groups.values() if len(ids) >= 2]
    if len(duplicate_groups) < MIN_DUPLICATE_GROUPS:
        return None

    available = [c for c in range(10) if c not in (bg, base)]
    if len(available) < MIN_PALETTE_SIZE:
        return None
    palette = random.sample(available, random.randint(MIN_PALETTE_SIZE, len(available)))

    pieces: dict[int, np.ndarray] = {}
    clues: dict[int, list[tuple[int, int, int]]] = {}

    for i, (_, _, _, _, is_dot) in enumerate(rects):
        if is_dot:
            pieces[i] = np.array([[0]], dtype=int)
            clues[i] = []

    keys = list(groups)
    random.shuffle(keys)
    for key in keys:
        ids = groups[key]
        h, w = map(int, key)
        if len(ids) == 1:
            i = int(ids[0])
            pieces[i] = _random_piece_patch(h, w, palette)
            clues[i] = []
            continue

        clue_count = 2 if h * w >= 2 else 1
        if len(ids) >= 4 and h * w >= 4:
            clue_count = 3
        clue_count = min(clue_count, CLUE_COUNT_MAX)
        positions = _signature_positions(h, w, clue_count)
        pool = _signature_pool(palette, clue_count)
        if len(pool) < len(ids):
            return None
        signatures = pool[: len(ids)]
        for i, sig in zip(ids, signatures):
            patch = _random_piece_patch(h, w, palette)
            for (rr, cc), col in zip(positions, sig):
                patch[int(rr), int(cc)] = int(col)
            pieces[int(i)] = patch
            clues[int(i)] = [(int(rr), int(cc), int(col)) for (rr, cc), col in zip(positions, sig)]

    return pieces, clues


def _render_output(
    dh: int,
    dw: int,
    base: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
) -> np.ndarray:
    out = np.full((dh, dw), base, dtype=int)
    for i, (r, c, _, _, _) in enumerate(rects):
        paste(out, pieces[i], r, c)
    return out


def _render_template_input_crop(
    dh: int,
    dw: int,
    bg: int,
    base: int,
    rects: list[tuple[int, int, int, int, bool]],
    clues: dict[int, list[tuple[int, int, int]]],
) -> np.ndarray:
    dst = np.full((dh, dw), base, dtype=int)
    for i, (r, c, h, w, _) in enumerate(rects):
        dst[r : r + h, c : c + w] = bg
        for dr, dc, col in clues.get(i, []):
            dst[r + dr, c + dc] = int(col)
    return dst


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
    clues: dict[int, list[tuple[int, int, int]]],
) -> np.ndarray:
    inp = np.full((H, W), bg, dtype=int)
    inp[tr0 : tr0 + dh, tc0 : tc0 + dw] = base
    for i, (r, c, h, w, _) in enumerate(rects):
        inp[tr0 + r : tr0 + r + h, tc0 + c : tc0 + c + w] = bg
        for dr, dc, col in clues.get(i, []):
            inp[tr0 + r + dr, tc0 + c + dc] = int(col)

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


def _variant_metrics_from_layout(
    dh: int,
    dw: int,
    bg: int,
    base: int,
    rects: list[tuple[int, int, int, int, bool]],
    pieces: dict[int, np.ndarray],
    clues: dict[int, list[tuple[int, int, int]]],
) -> dict[str, int]:
    dst = _render_template_input_crop(dh, dw, bg, base, rects, clues)
    hole_mask = dst != int(base)
    groups = _shape_groups(rects)
    pos_cache: dict[tuple[int, int], list[tuple[int, int]]] = {}
    ambiguous_pieces = 0
    resolved_pieces = 0
    duplicate_groups = 0
    for size, ids in groups.items():
        if len(ids) < 2:
            continue
        duplicate_groups += 1
        if size not in pos_cache:
            pos_cache[size] = _candidate_positions(hole_mask, *size)
        positions = pos_cache[size]
        for i in ids:
            h, w = size
            clue_positions = [
                (rr0, cc0)
                for rr0, cc0 in positions
                if _clue_compatible(dst[rr0 : rr0 + h, cc0 : cc0 + w], pieces[i], bg)
            ]
            if len(positions) > 1:
                ambiguous_pieces += 1
            if len(positions) > 1 and len(clue_positions) == 1:
                resolved_pieces += 1
    return {
        "duplicate_groups": int(duplicate_groups),
        "ambiguous_pieces": int(ambiguous_pieces),
        "resolved_pieces": int(resolved_pieces),
    }


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
    min_dots = max(DOT_MIN, total - (long - DOT_LONG_BUFFER))
    max_dots = min(DOT_MAX, total - DOT_NON_DOT_BUFFER)
    if min_dots > max_dots:
        return None
    dots = random.randint(min_dots, max_dots)

    canon = _corridor_rects(long, thick, total - dots, dots)
    if canon is None:
        return None
    rects = canon if vertical else [(c, r, w, h, d) for r, c, h, w, d in canon]

    if len([ids for ids in _shape_groups(rects).values() if len(ids) >= 2]) < MIN_DUPLICATE_GROUPS:
        return None

    bg, base = random.sample(range(1, 10), 2)
    piece_info = _make_pieces_and_clues(rects, bg, base)
    if piece_info is None:
        return None
    pieces, clues = piece_info

    metrics = _variant_metrics_from_layout(dh, dw, bg, base, rects, pieces, clues)
    if metrics["resolved_pieces"] < MIN_RESOLVED_AMBIGUOUS:
        return None

    out = _render_output(dh, dw, base, rects, pieces)
    try:
        inp = _scatter_sprites(H, W, bg, base, tr0, tc0, dh, dw, rects, pieces, clues)
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
    raise RuntimeError("failed to generate valid 5dbc8537 input")
