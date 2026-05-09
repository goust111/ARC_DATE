from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from arc_utils import paste, random_sprite  # noqa: E402


PALETTE = list(range(10))
BASE_N_CHOICES = [4, 5]

VARIANT_TITLES = {
    "D1/0": "形状匹配：标记区域的形状必须匹配模板轮廓",
    "D1/1": "边缘颜色匹配：标记像素所在行/列的边框颜色决定模板",
    "D1/3": "大小排序匹配：标记组中像素数量排序对应模板排序",
    "D2/0": "反向模运算映射",
    "D2/1": "转置映射",
    "D2/2": "旋转映射",
    "D3/1": "对角线像素对标记",
    "D3/2": "标记使用调色板颜色而非模板颜色",
    "D4/0": "tile块之间添加分隔线",
    "D4/1": "紧凑裁剪输出",
    "D5/0": "保留模板原色不重绘",
    "D5/1": "双色重绘",
    "D5/2": "颜色交换",
    "D7/1": "模板允许多色",
    "D8/0": "调色板使用颜色对编码",
    "D8/2": "调色板色样大小编码额外参数",
    "D9/0": "取消tile列统一色约束",
    "D10/0": "实线分隔",
    "D10/1": "无分隔线",
    "D11/0": "水平三带布局",
    "D11/1": "调色板带与模板带交换位置",
    "D12/0": "根据输出块位置旋转模板",
    "D12/1": "根据标记所在tile列翻转模板",
    "D12/3": "模板按块位置做镜像",
    "D14/2": "K随示例变化",
    "D16/1": "先印章后擦除",
    "D17/0": "背景色和分隔色在不同示例中变化",
    "D18/0": "空块填充默认图案",
    "D19/0": "调色板通过颜色匹配对应模板",
    "D19/1": "调色板顺序与模板顺序相反",
    "D20/2": "添加额外的无用模板作为干扰",
}


def _variant_doc(variant: str) -> str:
    title = VARIANT_TITLES[variant]
    return (
        f"Variant decoder for 65b59efc ({variant}).\n\n"
        f"Input structure:\n"
        f"- A tiled legend grid containing templates, markers, and palette hints.\n"
        f"- The exact legend layout depends on the selected single-dimension variant.\n"
        f"- Markers still indicate which stamp to place and where to place it.\n\n"
        f"Rule:\n"
        f"- Decode the variant-specific legend signals, recover the template-to-stamp mapping,\n"
        f"  then render the output stamp grid under the modified rule `{title}`.\n\n"
        f"Key insight: This variant changes one visible coding rule while keeping the\n"
        f"overall 'read legend then stamp blocks' task family intact."
    )


def _choice_excluding(exclude: set[int]) -> int:
    return random.choice([c for c in PALETTE if c not in exclude])


def _sample_distinct(count: int, exclude: set[int]) -> list[int]:
    return random.sample([c for c in PALETTE if c not in exclude], count)


def _sample_allow_overlap(count: int, exclude: set[int]) -> list[int]:
    pool = [c for c in PALETTE if c not in exclude]
    return [random.choice(pool) for _ in range(count)]


def _transform(arr: np.ndarray, op: str) -> np.ndarray:
    if op == "id":
        return arr.copy()
    if op == "rot90":
        return np.rot90(arr, 3)
    if op == "rot180":
        return np.rot90(arr, 2)
    if op == "rot270":
        return np.rot90(arr, 1)
    if op == "hflip":
        return np.fliplr(arr)
    if op == "vflip":
        return np.flipud(arr)
    raise ValueError(op)


def _component_masks(mask: np.ndarray) -> list[tuple[tuple[int, int], np.ndarray]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            rs = [p[0] for p in cells]
            cs = [p[1] for p in cells]
            r0, c0 = min(rs), min(cs)
            sub = np.zeros((max(rs) - r0 + 1, max(cs) - c0 + 1), dtype=bool)
            for rr, cc in cells:
                sub[rr - r0, cc - c0] = True
            out.append(((r0, c0), sub))
    return out


def _bbox_crop(arr: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(arr != bg)
    if coords.size == 0:
        return arr[:1, :1].copy()
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _sample_binary_mask(n: int, color: int, *, bg: int = 0, symmetric: str | None = None) -> np.ndarray:
    sprite = random_sprite(
        n,
        n,
        density=random.uniform(0.45, 0.78),
        symmetry=symmetric or random.choice([None, "horizontal", "vertical", "not_symmetric"]),
        color_palette=[color],
        connectivity=8,
        background=bg,
    )
    return sprite


def _sample_multicolor_template(n: int, colors: tuple[int, int], *, bg: int = 0) -> np.ndarray:
    primary, accent = colors
    base = _sample_binary_mask(n, primary, bg=bg)
    coords = np.argwhere(base == primary)
    accent_count = max(2, len(coords) // 4)
    random.shuffle(coords)
    out = base.copy()
    for rr, cc in coords[:accent_count]:
        out[rr, cc] = accent
    return out


def _sample_shape_template(n: int, color: int, *, bg: int = 0) -> np.ndarray:
    inner = random.choice([2, 3])
    sprite = random_sprite(
        inner,
        inner,
        density=random.uniform(0.45, 0.75),
        symmetry=random.choice(["not_symmetric", None]),
        color_palette=[color],
        connectivity=8,
        background=bg,
    )
    out = np.full((n, n), bg, dtype=int)
    r0 = random.randint(0, n - inner)
    c0 = random.randint(0, n - inner)
    paste(out, sprite, r0, c0)
    return out


def _template_primary(template: np.ndarray, bg: int) -> int:
    vals = [int(v) for v in np.unique(template) if int(v) != bg]
    if len(vals) == 1:
        return vals[0]
    counts = Counter(int(v) for v in template.flat if int(v) != bg)
    return counts.most_common(1)[0][0]


def _draw_sep_line(grid: np.ndarray, axis: str, idx: int, color: int, dashed: bool) -> None:
    if axis == "row":
        for c in range(grid.shape[1]):
            if (not dashed) or c % 2 == 0:
                grid[idx, c] = color
        return
    for r in range(grid.shape[0]):
        if (not dashed) or r % 2 == 0:
            grid[r, idx] = color


def _default_empty_tile(n: int, bg: int, sep: int) -> np.ndarray:
    tile = np.full((n, n), bg, dtype=int)
    mid = n // 2
    tile[mid, :] = sep
    tile[:, mid] = sep
    return tile


def _coord_map(variant: str, n: int, rr: int, cc: int) -> tuple[int, int]:
    if variant == "D2/0":
        return n - 1 - rr, n - 1 - cc
    if variant == "D2/1":
        return cc, rr
    if variant == "D2/2":
        return cc, n - 1 - rr
    return rr, cc


def _geom_op(variant: str, rr: int, cc: int, physical_idx: int, swatch_mode: str | None) -> str:
    if variant == "D12/0":
        return ["id", "rot90", "rot180", "rot270"][rr % 4]
    if variant == "D12/1":
        ops = ["id", "hflip", "vflip"]
        return ops[physical_idx % len(ops)]
    if variant == "D12/3":
        if rr % 2 == 1 and cc % 2 == 1:
            return "rot180"
        if rr % 2 == 1:
            return "vflip"
        if cc % 2 == 1:
            return "hflip"
        return "id"
    if variant == "D8/2" and swatch_mode is not None:
        return {"id": "id", "flip": "hflip", "turn": "rot90"}[swatch_mode]
    return "id"


def _render_template(
    variant: str,
    template: np.ndarray,
    bg: int,
    primary: int,
    palette_fg: int,
    palette_bg: int | None,
    rr: int,
    cc: int,
    physical_idx: int,
    swatch_mode: str | None,
) -> np.ndarray:
    op = _geom_op(variant, rr, cc, physical_idx, swatch_mode)
    cur = _transform(template, op)

    if variant == "D7/1":
        vals = [int(v) for v in np.unique(cur) if int(v) != bg]
        vals.sort()
        if len(vals) != 2:
            raise AssertionError("expected two template colors")
        return np.where(cur == vals[0], palette_fg, np.where(cur == vals[1], palette_bg, bg))

    if variant == "D5/0":
        return np.where(cur != bg, cur, bg)
    if variant == "D5/1":
        assert palette_bg is not None
        return np.where(cur != bg, palette_fg, palette_bg)
    if variant == "D5/2":
        return np.where(cur != bg, cur, palette_fg)
    return np.where(cur != bg, palette_fg, bg)


def _render_from_meta(meta: dict) -> np.ndarray:
    n = meta["n"]
    bg = meta["bg"]
    sep = meta["sep"]
    variant = meta["variant"]
    block_tiles = {}

    if variant == "D18/0":
        empty_tile = _default_empty_tile(n, bg, sep)
    else:
        empty_tile = None

    for j, rr, cc, physical_idx in meta["placements"]:
        dr, dc = _coord_map(variant, n, rr, cc)
        tile = _render_template(
            variant,
            meta["templates"][j],
            bg,
            meta["primaries"][j],
            meta["palette_fg"][j],
            meta["palette_bg"][j],
            dr,
            dc,
            physical_idx,
            meta["swatch_modes"][j],
        )
        block_tiles[(dr, dc)] = tile

    for rr, cc in meta["erase_slots"]:
        block_tiles[(rr, cc)] = np.full((n, n), bg, dtype=int)

    out = np.full((n * n, n * n), bg, dtype=int)
    for rr in range(n):
        for cc in range(n):
            tile = block_tiles.get((rr, cc), empty_tile)
            if tile is not None:
                paste(out, tile, rr * n, cc * n)

    if variant == "D4/0":
        big = np.full((n * n + n - 1, n * n + n - 1), bg, dtype=int)
        for rr in range(n):
            for cc in range(n):
                patch = out[rr * n : (rr + 1) * n, cc * n : (cc + 1) * n]
                paste(big, patch, rr * (n + 1), cc * (n + 1))
        for idx in range(1, n):
            r = idx * n + (idx - 1)
            c = idx * n + (idx - 1)
            big[r, :] = sep
            big[:, c] = sep
        out = big
    elif variant == "D4/1":
        out = _bbox_crop(out, bg)

    return out


def _mark_slots(meta: dict) -> dict[int, list[tuple[int, int]]]:
    by_j = defaultdict(list)
    for j, rr, cc, _ in meta["placements"]:
        by_j[j].append((rr, cc))
    return by_j


def _build_standard_vertical(meta: dict) -> np.ndarray:
    n = meta["n"]
    k = meta["k_total"]
    bg = meta["bg"]
    sep = meta["sep"]
    variant = meta["variant"]
    dashed = variant not in {"D10/0"}
    no_sep = variant == "D10/1"
    color_stripes = variant == "D1/1"
    swapped = variant == "D11/1"
    frame_cols = meta["frame_cols"]

    if no_sep:
        h = 2 * n + 2
        w = k * n
        grid = np.full((h, w), bg, dtype=int)
        starts = [j * n for j in range(k)]
        top_r0, mid_r0, pal_r0 = 0, n, 2 * n
    elif color_stripes:
        h = 2 * n + 4
        w = k * (n + 1)
        grid = np.full((h, w), bg, dtype=int)
        starts = [j * (n + 1) for j in range(k)]
        top_r0, mid_r0, pal_r0 = 0, n + 1, 2 * n + 2
        _draw_sep_line(grid, "row", n, sep, True)
        _draw_sep_line(grid, "row", 2 * n + 1, sep, True)
        for j, col in enumerate(frame_cols):
            c = starts[j] + n
            for r in range(h):
                if r in {n, 2 * n + 1}:
                    continue
                grid[r, c] = col
    else:
        h = 2 * n + 4
        w = k * n + (k - 1)
        grid = np.full((h, w), bg, dtype=int)
        starts = [j * (n + 1) for j in range(k)]
        top_r0, mid_r0, pal_r0 = 0, n + 1, 2 * n + 2
        if not no_sep:
            _draw_sep_line(grid, "row", n, sep, dashed)
            _draw_sep_line(grid, "row", 2 * n + 1, sep, dashed)
            for j in range(k - 1):
                _draw_sep_line(grid, "col", starts[j] + n, sep, dashed)

    if swapped:
        top_r0, pal_r0 = pal_r0, 0
        mid_r0 = 3
        grid[:] = bg
        _draw_sep_line(grid, "row", 2, sep, dashed)
        _draw_sep_line(grid, "row", 3 + n, sep, dashed)
        pal_r0, top_r0 = 0, 4 + n

    for j, st in enumerate(starts):
        paste(grid, meta["templates"][j], top_r0, st)

    by_j = _mark_slots(meta)
    for j, slots in by_j.items():
        if variant == "D1/0":
            for rr, cc in slots:
                shape = meta["marker_shapes"][j]
                mark = _transform(shape, random.choice(["id", "rot90", "hflip"]))
                h0, w0 = mark.shape
                r0 = mid_r0 + rr
                c0 = starts[meta["marker_home"][j]] + cc
                paste(grid, np.where(mark, meta["marker_colors"][j], bg), r0, c0)
            continue
        for rr, cc in slots:
            phys = meta["marker_home"][j]
            if variant == "D3/1":
                c0 = starts[phys] + cc
                r0 = mid_r0 + rr
                col = meta["marker_colors"][j]
                grid[r0, c0] = col
                grid[r0 + 1, c0 + 1] = col
            else:
                col = meta["marker_colors"][j]
                grid[mid_r0 + rr, starts[phys] + cc] = col

    if variant == "D9/0":
        for j, rr, cc, phys in meta["placements"]:
            grid[mid_r0 + rr, starts[phys] + cc] = meta["marker_colors"][j]

    if variant == "D16/1":
        for rr, cc in meta["erase_slots"]:
            grid[mid_r0 + rr, starts[random.randrange(k)] + cc] = meta["erase_color"]

    for j, st in enumerate(starts):
        if variant == "D8/0":
            row = pal_r0 + 1
            grid[row, st + max(0, n // 2 - 1)] = meta["palette_keys"][j]
            grid[row, st + min(n - 1, n // 2)] = meta["palette_fg"][j]
        elif variant == "D8/2":
            row = pal_r0 + 1
            size = {"id": 1, "flip": 2, "turn": 3}[meta["swatch_modes"][j]]
            c0 = st + (n - size) // 2
            grid[row, c0 : c0 + size] = meta["palette_fg"][j]
        elif variant == "D19/0":
            row = pal_r0 + 1
            col_slot = meta["palette_slots"][j]
            st2 = starts[col_slot]
            grid[row, st2 + max(0, n // 2 - 1)] = meta["primaries"][j]
            grid[row, st2 + min(n - 1, n // 2)] = meta["palette_fg"][j]
        elif variant in {"D5/1", "D7/1"}:
            grid[pal_r0, st + n // 2] = meta["palette_bg"][j]
            grid[pal_r0 + 1, st + n // 2] = meta["palette_fg"][j]
        else:
            row = pal_r0 + 1
            col = meta["palette_fg"][j]
            if variant == "D19/1":
                col = meta["palette_fg"][k - 1 - j]
            grid[row, st + n // 2] = col

    if variant == "D5/0":
        for j, st in enumerate(starts):
            grid[pal_r0 + 1, st + n // 2] = meta["noise_palette"][j]

    return grid


def _build_horizontal(meta: dict) -> np.ndarray:
    n = meta["n"]
    k = meta["k_total"]
    bg = meta["bg"]
    sep = meta["sep"]
    h = k * n + (k - 1)
    w = 3 * n + 2
    grid = np.full((h, w), bg, dtype=int)
    for i in range(k - 1):
        _draw_sep_line(grid, "row", i * (n + 1) + n, sep, True)
    _draw_sep_line(grid, "col", n, sep, True)
    _draw_sep_line(grid, "col", 2 * n + 1, sep, True)
    starts = [i * (n + 1) for i in range(k)]

    for j, r0 in enumerate(starts):
        paste(grid, meta["templates"][j], r0, 0)
        grid[r0 + n // 2, 2 * n + 2 + n // 2] = meta["palette_fg"][j]

    for j, rr, cc, phys in meta["placements"]:
        r0 = starts[phys]
        grid[r0 + rr, n + 1 + cc] = meta["marker_colors"][j]

    return grid


def _compute_meta(variant: str) -> dict:
    n = random.choice(BASE_N_CHOICES)
    bg = 0
    sep = 5
    if variant == "D17/0":
        bg, sep = random.sample(PALETTE, 2)

    if variant == "D12/1":
        k_total = 3
    elif variant == "D14/2":
        k_total = random.choice([2, 3, 4, 5])
    elif variant == "D20/2":
        k_total = random.choice([4, 5])
    else:
        k_total = random.choice([3, 4])

    k_used = k_total
    if variant == "D20/2":
        k_used = max(2, k_total - random.choice([1, 2]))

    reserved = {bg, sep}
    if variant == "D16/1":
        erase_color = _choice_excluding(reserved)
        reserved.add(erase_color)
    else:
        erase_color = None

    if variant == "D7/1":
        template_cols = _sample_distinct(2 * k_total, reserved)
        fg_cols = _sample_distinct(k_total, reserved)
        bg_cols = _sample_allow_overlap(k_total, reserved)
        templates = []
        primaries = []
        secondaries = []
        palette_fg = []
        palette_bg = []
        for j in range(k_total):
            c1, c2 = template_cols[2 * j : 2 * j + 2]
            p1 = fg_cols[j]
            p2 = bg_cols[j]
            tpl = _sample_multicolor_template(n, (c1, c2), bg=bg)
            templates.append(tpl)
            primaries.append(c1)
            secondaries.append(c2)
            palette_fg.append(p1)
            palette_bg.append(p2)
        marker_shapes = None
    elif variant == "D1/0":
        prim_cols = _sample_distinct(k_total, reserved)
        pal_cols = _sample_distinct(k_total, reserved)
        templates = []
        primaries = []
        palette_fg = []
        palette_bg = [None] * k_total
        marker_shapes = []
        for j in range(k_total):
            c1 = prim_cols[j]
            p1 = pal_cols[j]
            tpl = _sample_shape_template(n, c1, bg=bg)
            templates.append(tpl)
            primaries.append(c1)
            palette_fg.append(p1)
            palette_bg.append(None)
            marker_shapes.append(_bbox_crop(tpl != 0, False))
        secondaries = [None] * k_total
    else:
        prim_cols = _sample_distinct(k_total, reserved)
        pal_cols = _sample_distinct(k_total, reserved)
        templates = []
        primaries = []
        palette_fg = []
        palette_bg = [None] * k_total
        for j in range(k_total):
            c1 = prim_cols[j]
            p1 = pal_cols[j]
            symmetric = "horizontal" if variant == "D12/1" else None
            if variant == "D12/0":
                symmetric = "not_symmetric"
            tpl = _sample_binary_mask(n, c1, bg=bg, symmetric=symmetric)
            templates.append(tpl)
            primaries.append(c1)
            palette_fg.append(p1)
        secondaries = [None] * k_total
        marker_shapes = None

    if variant == "D5/1":
        extra = _sample_allow_overlap(k_total, reserved)
        palette_bg = extra
    if variant == "D5/0":
        noise_palette = _sample_distinct(k_total, reserved | set(primaries))
    else:
        noise_palette = [None] * k_total

    if variant == "D3/2":
        marker_colors = palette_fg[:]
    elif variant == "D1/3":
        marker_colors = _sample_distinct(k_total, reserved | set(primaries))
    elif variant == "D1/1":
        marker_colors = [_choice_excluding(reserved | set(primaries))] * k_total
    else:
        marker_colors = primaries[:]

    frame_cols = [None] * k_total
    if variant == "D1/1":
        frame_cols = _sample_distinct(k_total, reserved | set(primaries))

    template_order = list(range(k_total))
    random.shuffle(template_order)
    if variant == "D1/1":
        template_order = list(range(k_total))
    marker_home = {j: template_order[j] for j in range(k_total)}

    used_js = template_order[:k_used]
    slots = [(r, c) for r in range(n) for c in range(n)]
    if variant == "D3/1":
        slots = [(r, c) for r in range(n - 1) for c in range(n - 1) if (r + c) % 2 == 0]
    if variant == "D1/0":
        slots = [(r, c) for r in range(n - 2) for c in range(n - 2)]
    random.shuffle(slots)

    placements: list[tuple[int, int, int, int]] = []
    erase_slots: set[tuple[int, int]] = set()

    if variant == "D1/0":
        for j in used_js:
            phys = marker_home[j]
            count = random.choice([1, 2])
            random.shuffle(slots)
            for rr, cc in slots[:count]:
                placements.append((j, rr, cc, phys))
    elif variant == "D1/3":
        sorted_js = sorted(used_js, key=lambda j: int(np.count_nonzero(templates[j] != bg)))
        counts = list(range(1, len(sorted_js) + 1))
        for j, cnt in zip(sorted_js, counts):
            phys = marker_home[j]
            for rr, cc in slots[:cnt]:
                placements.append((j, rr, cc, phys))
            slots = slots[cnt:]
    else:
        min_marks = max(k_used, int(0.45 * len(slots)))
        max_marks = max(min_marks, int(0.7 * len(slots)))
        count = random.randint(min_marks, max_marks)
        chosen = slots[:count]
        js = [random.choice(used_js) for _ in chosen]
        for j in used_js:
            if j not in js:
                js[random.randrange(len(js))] = j
        for (rr, cc), j in zip(chosen, js):
            phys = random.randrange(k_total) if variant == "D9/0" else marker_home[j]
            placements.append((j, rr, cc, phys))

    if variant == "D16/1":
        sample = random.sample(placements, k=min(2, len(placements)))
        erase_slots = {_coord_map(variant, n, rr, cc) for _, rr, cc, _ in sample}

    if variant == "D8/2":
        modes = ["id", "flip", "turn"]
        swatch_modes = [random.choice(modes) for _ in range(k_total)]
    else:
        swatch_modes = [None] * k_total

    if variant == "D19/0":
        palette_slots = list(range(k_total))
        random.shuffle(palette_slots)
    else:
        palette_slots = list(range(k_total))

    if variant == "D8/0":
        palette_keys = primaries[:]
    else:
        palette_keys = [None] * k_total

    meta = {
        "variant": variant,
        "n": n,
        "bg": bg,
        "sep": sep,
        "k_total": k_total,
        "templates": templates,
        "primaries": primaries,
        "secondaries": secondaries,
        "palette_fg": palette_fg,
        "palette_bg": palette_bg,
        "marker_colors": marker_colors,
        "marker_home": marker_home,
        "placements": placements,
        "erase_slots": erase_slots,
        "erase_color": erase_color,
        "frame_cols": frame_cols,
        "swatch_modes": swatch_modes,
        "palette_slots": palette_slots,
        "palette_keys": palette_keys,
        "marker_shapes": marker_shapes,
        "noise_palette": noise_palette,
    }
    return meta


def _parse_standard(variant: str, grid: np.ndarray) -> dict:
    g = np.array(grid, dtype=int)
    h, w = g.shape

    if variant == "D11/0":
        n = (w - 2) // 3
        k = (h + 1) // (n + 1)
        starts = [i * (n + 1) for i in range(k)]
        bg = Counter(map(int, g.flatten())).most_common(1)[0][0]
        sep = [int(v) for v, _ in Counter(map(int, g[:, n].tolist())).items()][0]
        templates = [g[r0 : r0 + n, 0:n].copy() for r0 in starts]
        primaries = [_template_primary(t, bg) for t in templates]
        palette_fg = [int(g[r0 + n // 2, 2 * n + 2 + n // 2]) for r0 in starts]
        marker_records = []
        for phys, r0 in enumerate(starts):
            for rr in range(n):
                for cc in range(n):
                    v = int(g[r0 + rr, n + 1 + cc])
                    if v == bg or v == sep:
                        continue
                    marker_records.append((v, rr, cc, phys))
        return {
            "variant": variant,
            "n": n,
            "bg": bg,
            "sep": sep,
            "templates": templates,
            "primaries": primaries,
            "palette_fg": palette_fg,
            "palette_bg": [None] * k,
            "swatch_modes": [None] * k,
            "marker_records": marker_records,
        }

    n_candidates = []
    if variant == "D10/1":
        for n in BASE_N_CHOICES:
            if h == 2 * n + 2 and w % n == 0:
                n_candidates.append((n, w // n))
    elif variant == "D1/1":
        for n in BASE_N_CHOICES:
            if h == 2 * n + 4 and w % (n + 1) == 0:
                n_candidates.append((n, w // (n + 1)))
    else:
        for n in BASE_N_CHOICES:
            if h == 2 * n + 4 and (w + 1) % (n + 1) == 0:
                n_candidates.append((n, (w + 1) // (n + 1)))
    if not n_candidates:
        raise AssertionError("unable to infer N/K")
    n, k = n_candidates[0]

    counts = Counter(map(int, g.flatten()))
    bg = int(counts.most_common(1)[0][0])
    sep = 5
    if variant == "D17/0":
        if variant == "D10/1":
            sep = _choice_excluding({bg})
        else:
            row_idx = 2 if variant == "D11/1" else n
            row_counts = Counter(int(v) for v in g[row_idx, :])
            sep = row_counts.most_common(1)[0][0]

    if variant == "D10/1":
        starts = [j * n for j in range(k)]
        top_r0, mid_r0, pal_r0 = 0, n, 2 * n
    elif variant == "D11/1":
        starts = [j * (n + 1) for j in range(k)]
        top_r0, mid_r0, pal_r0 = 4 + n, 3, 0
    elif variant == "D1/1":
        starts = [j * (n + 1) for j in range(k)]
        top_r0, mid_r0, pal_r0 = 0, n + 1, 2 * n + 2
    else:
        starts = [j * (n + 1) for j in range(k)]
        top_r0, mid_r0, pal_r0 = 0, n + 1, 2 * n + 2

    templates = [g[top_r0 : top_r0 + n, st : st + n].copy() for st in starts]
    primaries = [_template_primary(t, bg) for t in templates]

    if variant == "D7/1":
        palette_fg = [int(g[pal_r0 + 1, st + n // 2]) for st in starts]
        palette_bg = [int(g[pal_r0, st + n // 2]) for st in starts]
    elif variant == "D5/1":
        palette_fg = [int(g[pal_r0 + 1, st + n // 2]) for st in starts]
        palette_bg = [int(g[pal_r0, st + n // 2]) for st in starts]
    elif variant == "D19/0":
        palette_fg = [None] * k
        for st in starts:
            row = pal_r0 + 1
            vals = [int(g[row, st + x]) for x in range(n) if int(g[row, st + x]) != bg]
            if len(vals) >= 2:
                key, pal = vals[0], vals[1]
                palette_fg[primaries.index(key)] = pal
        palette_bg = [None] * k
    elif variant == "D8/0":
        palette_fg = [int(g[pal_r0 + 1, st + min(n - 1, n // 2)]) for st in starts]
        palette_bg = [None] * k
    elif variant == "D19/1":
        raw = [int(g[pal_r0 + 1, st + n // 2]) for st in starts]
        palette_fg = list(reversed(raw))
        palette_bg = [None] * k
    else:
        palette_fg = [int(g[pal_r0 + 1, st + n // 2]) for st in starts]
        palette_bg = [None] * k

    swatch_modes = [None] * k
    if variant == "D8/2":
        for j, st in enumerate(starts):
            row = g[pal_r0 + 1, st : st + n]
            cells = [int(v) for v in row if int(v) != bg]
            palette_fg[j] = cells[0]
            size = len(cells)
            swatch_modes[j] = {1: "id", 2: "flip", 3: "turn"}[size]

    marker_records = []
    if variant == "D1/0":
        shape_db = defaultdict(list)
        for j, tpl in enumerate(templates):
            mask = tpl != bg
            for op in ["id", "rot90", "rot180", "rot270", "hflip", "vflip"]:
                sub = _bbox_crop(_transform(mask.astype(int), op), 0) != 0
                shape_db[sub.tobytes()].append(j)
        for phys, st in enumerate(starts):
            mask = g[mid_r0 : mid_r0 + n, st : st + n] != bg
            comps = _component_masks(mask)
            for (rr, cc), sub in comps:
                matches = shape_db.get(sub.tobytes(), [])
                if not matches:
                    continue
                marker_records.append((matches[0], rr, cc, phys))
    elif variant == "D3/1":
        for phys, st in enumerate(starts):
            sub = g[mid_r0 : mid_r0 + n, st : st + n]
            for rr in range(n - 1):
                for cc in range(n - 1):
                    v = int(sub[rr, cc])
                    if v == bg or v == sep:
                        continue
                    if int(sub[rr + 1, cc + 1]) == v:
                        marker_records.append((v, rr, cc, phys))
    else:
        for phys, st in enumerate(starts):
            sub = g[mid_r0 : mid_r0 + n, st : st + n]
            for rr in range(sub.shape[0]):
                for cc in range(sub.shape[1]):
                    v = int(sub[rr, cc])
                    if v == bg or (variant != "D10/1" and v == sep):
                        continue
                    marker_records.append((v, rr, cc, phys))

    if variant == "D16/1":
        erase_color = None
        known = set(primaries) | set(palette_fg) | {bg, sep}
        for v, _, _, _ in marker_records:
            if v not in known:
                erase_color = v
                break
    else:
        erase_color = None

    return {
        "variant": variant,
        "n": n,
        "bg": bg,
        "sep": sep,
        "templates": templates,
        "primaries": primaries,
        "palette_fg": palette_fg,
        "palette_bg": palette_bg,
        "swatch_modes": swatch_modes,
        "marker_records": marker_records,
        "starts": starts,
        "erase_color": erase_color,
        "pal_r0": pal_r0,
        "mid_r0": mid_r0,
    }


def solve_variant(variant: str, grid: np.ndarray) -> np.ndarray:
    parsed = _parse_standard(variant, np.array(grid, dtype=int))
    n = parsed["n"]
    bg = parsed["bg"]
    sep = parsed["sep"]
    templates = parsed["templates"]
    primaries = parsed["primaries"]
    palette_fg = parsed["palette_fg"]
    palette_bg = parsed["palette_bg"]
    swatch_modes = parsed["swatch_modes"]
    records = parsed["marker_records"]

    if variant == "D1/3":
        tmpl_order = sorted(range(len(templates)), key=lambda j: int(np.count_nonzero(templates[j] != bg)))
        counts = Counter(v for v, _, _, _ in records)
        col_order = [color for color, _ in sorted(counts.items(), key=lambda item: item[1])]
        color_to_j = {col: j for col, j in zip(col_order, tmpl_order)}
    elif variant == "D3/2":
        color_to_j = {palette_fg[j]: j for j in range(len(templates))}
    else:
        color_to_j = {primaries[j]: j for j in range(len(templates))}

    if variant == "D1/1":
        starts = parsed["starts"]
        frame_to_j = {}
        top_r0 = 0
        mid_r0 = parsed["mid_r0"]
        for j, st in enumerate(starts):
            frame_to_j[int(grid[top_r0, st + n])] = j
        placements = []
        for _, rr, cc, phys in records:
            stripe = int(grid[mid_r0, starts[phys] + n])
            j = frame_to_j[stripe]
            placements.append((j, rr, cc, phys))
    else:
        placements = []
        for v, rr, cc, phys in records:
            if variant == "D16/1" and v == parsed["erase_color"]:
                continue
            if variant == "D1/0":
                j = v
            else:
                j = color_to_j[v]
            placements.append((j, rr, cc, phys))

    erase_slots = set()
    if variant == "D16/1":
        erase_slots = {_coord_map(variant, n, rr, cc) for v, rr, cc, _ in records if v == parsed["erase_color"]}

    meta = {
        "variant": variant,
        "n": n,
        "bg": bg,
        "sep": sep,
        "templates": templates,
        "primaries": primaries,
        "palette_fg": palette_fg,
        "palette_bg": palette_bg,
        "swatch_modes": swatch_modes,
        "placements": placements,
        "erase_slots": erase_slots,
    }
    return _render_from_meta(meta)


def generate_variant(variant: str) -> np.ndarray:
    for _ in range(300):
        meta = _compute_meta(variant)
        if variant == "D11/0":
            grid = _build_horizontal(meta)
        else:
            grid = _build_standard_vertical(meta)
        expected = _render_from_meta(meta)
        try:
            actual = solve_variant(variant, grid)
        except Exception:
            continue
        if np.array_equal(expected, actual):
            return grid
    raise RuntimeError(f"failed to generate a valid 65b59efc {variant} input")


def make_variant(variant: str):
    def main(grid: np.ndarray) -> np.ndarray:
        return solve_variant(variant, np.array(grid, dtype=int))

    def generate_input() -> np.ndarray:
        return generate_variant(variant)

    main.__doc__ = _variant_doc(variant)
    return main, generate_input
