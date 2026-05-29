# Variant of 64efde09
# Dimension: D14 — 端口到模板的映射方式 (meta)
# Change: 通过二面体变换的ID追踪实现端口映射——每个模板像素有唯一ID，变换后ID跟随，标记标注的模板ID在所有精灵上找到对应位... → 通过非活跃边颜色间接映射

from __future__ import annotations

"""
Design analysis for the D14 variant
===================================

variant_spec:
  preserved_backbone:
    - 背景仍是最常见色。
    - 仍然有 3-4 个由同一条 N×2 模板经过二面体变换得到的致密矩形精灵。
    - 输出仍然是在每个精灵的活跃长边端口向外发射有色射线。
  changed_core:
    - 原题里，标记点直接落在活跃边外侧，所以“看到哪个标记”就等于“这个活跃端口被上色”。
    - 本变体里，标记点改为落在非活跃边外侧。标记先指向一条 2 格宽槽位的“非活跃边成员”，
      再通过同一行/列的对齐关系找到该槽位另一侧的活跃端口。
    - 为了在所有变换精灵之间稳定转移这个信息，不再给每个像素分配模板 ID；
      改为给每条槽位分配一个可见签名 `(非活跃边颜色, 距离最近端点的距离)`。
      该距离在翻转后保持不变，因此不需要像素 ID 追踪。

behavior_trace:
  old_flow:
    - 例：竖直精灵左列是活跃边，某一行左侧边界上有单像素标记色 6。
    - solver 沿左侧射线看到 6，就把“这一行左列像素的模板 ID”记成颜色 6。
    - 之后去所有变换精灵里找同一个模板 ID，并从这些位置继续向左/上/右/下发射 6。
  new_flow:
    - 同样的精灵里，标记色 6 现在出现在右侧边界，先命中的是同一行右列的非活跃像素。
    - solver 不会直接把 6 赋给右列；而是读取该非活跃像素的颜色，以及它离最近端点有多远。
    - 这个签名唯一确定一条槽位；槽位另一侧同一行的左列像素才是真正要发射射线的活跃端口。
    - 在其它变换精灵上，找到签名相同的非活跃槽位，再取其对齐的活跃成员发射 6。
  visible_difference:
    - 输入里标记与输出射线总是在精灵两侧分离出现。
    - 人类解题者不能再把“标记所在那一侧”当作发射起点，必须先跨过 2 格宽模板做一次配对。

complexity_calibration:
  - 只用“非活跃边单个颜色”会退化，因为颜色可重复；因此增加一个很弱但可见的几何线索：
    到最近端点的距离。它只有 0..4 这类少量取值，且条带很短，肉眼可数。
  - 不使用颜色 id 排序、数值大小规律等不可见因素；距离完全来自几何位置，颜色来自画面本身。
  - 保持模板仍是 N×2 细长条，避免引入额外对象类型或高自由度结构。

implicit_assumptions_review:
  - 原 `_first_marker_on_ray(...)` 默认“标记在活跃侧”这一点在变体中失效，因此现在要从非活跃侧扫描。
  - 原先 `template_id = sprite["ids"][lr, lc]` 假设每个被标注像素可以直接映射到模板 ID；
    变体不再成立，因为被标注的是非活跃边像素，而且题意要求不要靠像素 ID 追踪。
  - 原先 `assert (lr, lc) in sprite["active"]` 在输出阶段天然成立；现在必须先从签名找槽位，
    再显式取槽位的活跃成员。
  - 新 solver 额外假设：每个精灵上 `(非活跃边颜色, 最近端距离)` 对槽位唯一。
    generator 会显式构造并验证这一点。

placement_legality:
  - 精灵放在画布内部，而不是贴某一侧边界；这样同一条槽位的两侧都能各自连到边界：
    一侧放输入标记，另一侧画输出射线。
  - 生成器要求精灵之间带间隔，并在放好标记后检查：
    1. 标记到其目标非活跃边之间视线全是背景。
    2. 每个标记恰好只会被一个非活跃槽位看到，避免歧义。
    3. 新 solver 能完整跑通，借此过滤所有射线冲突。

visual_readability:
  - 单像素标记仍然在边界上，和原题视觉语言一致。
  - 标记始终位于输出射线的对侧，因此输入不会出现“标记像素与输出颜色已经在同一侧排开”的假象。
  - 仍只使用 3 个控制标记，避免边界被过多小点塞满。

generator_guarantees_for_solver:
  - 所有槽位签名在模板内唯一。
  - 所有标记颜色唯一，且与模板颜色集合不相交。
  - 每个标记都能被某个非活跃槽位看到，且不能被别的槽位误看到。
"""

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg


def _dihedral_variants(template: np.ndarray) -> list[tuple[str, np.ndarray]]:
    variants: list[tuple[str, np.ndarray]] = []
    for k in range(4):
        rotated = np.rot90(template, k)
        variants.append((f"rot{k}", rotated))
        variants.append((f"rot{k}_fliplr", np.fliplr(rotated)))
    return variants


def _active_side(sub: np.ndarray) -> tuple[tuple[int, int], set[tuple[int, int]]]:
    h, w = sub.shape
    assert h != w, "sprites must be non-square rectangles"

    if h > w:
        sides = [
            ((0, -1), [(r, 0) for r in range(h)]),
            ((0, 1), [(r, w - 1) for r in range(h)]),
        ]
    else:
        sides = [
            ((-1, 0), [(0, c) for c in range(w)]),
            ((1, 0), [(h - 1, c) for c in range(w)]),
        ]

    scored = []
    for direction, coords in sides:
        vals = [int(sub[r, c]) for r, c in coords]
        scored.append((max(Counter(vals).values()), direction, set(coords)))

    scored.sort(key=lambda item: item[0], reverse=True)
    assert scored[0][0] > scored[1][0], "active side must be structurally unique"
    return scored[0][1], scored[0][2]


def _first_marker_on_ray(
    g: np.ndarray,
    bg: int,
    markers: dict[tuple[int, int], int],
    r: int,
    c: int,
    dr: int,
    dc: int,
) -> int | None:
    H, W = g.shape
    r += dr
    c += dc
    while 0 <= r < H and 0 <= c < W:
        if int(g[r, c]) != bg:
            return markers.get((r, c))
        r += dr
        c += dc
    return None


def _slot_descriptors(sub: np.ndarray) -> tuple[tuple[int, int], tuple[int, int], list[dict]]:
    direction, active = _active_side(sub)
    dr, dc = direction
    label_direction = (-dr, -dc)
    h, w = sub.shape
    slots: list[dict] = []

    if h > w:
        active_col = 0 if direction == (0, -1) else w - 1
        label_col = w - 1 - active_col
        expected_active = {(r, active_col) for r in range(h)}
        assert active == expected_active
        for idx in range(h):
            signature = (int(sub[idx, label_col]), min(idx, h - 1 - idx))
            slots.append(
                {
                    "active": (idx, active_col),
                    "label": (idx, label_col),
                    "signature": signature,
                }
            )
    else:
        active_row = 0 if direction == (-1, 0) else h - 1
        label_row = h - 1 - active_row
        expected_active = {(active_row, c) for c in range(w)}
        assert active == expected_active
        for idx in range(w):
            signature = (int(sub[label_row, idx]), min(idx, w - 1 - idx))
            slots.append(
                {
                    "active": (active_row, idx),
                    "label": (label_row, idx),
                    "signature": signature,
                }
            )

    signatures = [slot["signature"] for slot in slots]
    assert len(signatures) == len(set(signatures)), "slot signatures must be unique"
    return direction, label_direction, slots


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read control markers from the non-active side of each strip sprite, then fire them from the aligned active side.

    Input structure:
    - bg is the most common color.
    - Several dense 2-by-N or N-by-2 rectangular sprites are dihedral transforms of one narrow template.
    - Isolated single-pixel markers sit on boundary rays leaving the non-active long edge, not the active one.

    Rule:
    - For each sprite, identify the active long edge as the one with the stronger same-color majority.
    - A marker seen first from a non-active-edge cell labels that whole 2-cell slot via the signature `(non-active color, distance to nearest end)`.
    - On every sprite, find the slot with that signature and draw the marker color outward from the aligned active-edge cell until blocked or out of bounds.

    Key insight:
    - The marker does not name the firing port directly. It first selects the opposite-side member of a slot, and the actual launch point is the cell aligned across the strip.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    comps = find_components(g, bg=bg, connectivity=8, monochromatic=False)

    sprites: list[dict] = []
    markers: dict[tuple[int, int], int] = {}
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        area = int(comp["area"])
        if area == 1:
            r, c = map(int, comp["coords"][0])
            markers[(r, c)] = int(g[r, c])
        else:
            assert area == h * w, "sprite components must be dense rectangles"
            sub = crop_bbox(g, (r0, c0, r1, c1))
            direction, label_direction, slots = _slot_descriptors(sub)
            slot_map = {slot["signature"]: slot for slot in slots}
            sprites.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "sub": sub,
                    "direction": direction,
                    "label_direction": label_direction,
                    "slots": slots,
                    "slot_map": slot_map,
                }
            )

    assert len(sprites) >= 2, "expected multiple transformed rectangular sprites"
    assert markers, "expected isolated control markers"
    assert len(set(markers.values())) == len(markers), "control marker colors must be unique"

    color_to_signature: dict[int, tuple[int, int]] = {}
    signature_to_color: dict[tuple[int, int], int] = {}
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        ldr, ldc = sprite["label_direction"]
        for slot in sprite["slots"]:
            lr, lc = slot["label"]
            marker_color = _first_marker_on_ray(g, bg, markers, r0 + lr, c0 + lc, ldr, ldc)
            if marker_color is None:
                continue
            signature = slot["signature"]
            if marker_color in color_to_signature:
                assert color_to_signature[marker_color] == signature
            if signature in signature_to_color:
                assert signature_to_color[signature] == marker_color
            color_to_signature[marker_color] = signature
            signature_to_color[signature] = marker_color

    assert set(color_to_signature) == set(markers.values()), "every control marker must label one non-active slot"

    out = g.copy()
    H, W = g.shape
    for sprite in sprites:
        r0, c0, _r1, _c1 = sprite["bbox"]
        dr, dc = sprite["direction"]
        for color, signature in color_to_signature.items():
            slot = sprite["slot_map"].get(signature)
            assert slot is not None, "every labeled slot signature must exist on every sprite"
            lr, lc = slot["active"]

            r = r0 + lr + dr
            c = c0 + lc + dc
            while 0 <= r < H and 0 <= c < W:
                if int(g[r, c]) != bg:
                    break
                assert int(out[r, c]) in (bg, color), "control rays must not conflict"
                out[r, c] = color
                r += dr
                c += dc

    return out


# --------------- generator ---------------

_BG = 8
_NON_BG_COLORS = [0, 1, 2, 3, 4, 5, 6, 7, 9]
_GRID_H_RANGE = (21, 29)
_GRID_W_RANGE = (27, 29)
_TEMPLATE_N_RANGE = (7, 9)
_TEMPLATE_COLOR_COUNT = (3, 4)
_ACTIVE_MAJORITY_RANGE = (5, 9)
_SPRITE_COUNT_RANGE = (3, 4)
_NUM_MARKERS = 3
_PLACEMENT_PAD = 2
_BOUNDARY_MARGIN = 2
_PLACEMENT_RETRIES = 100
_TEMPLATE_RETRIES = 200
_GEN_RETRIES = 300


def _build_column(n: int, colors: list[int], majority: int, dominant: int) -> list[int]:
    col = [dominant] * majority
    others = [c for c in colors if c != dominant]
    fill = others if others else colors
    for _ in range(n - majority):
        col.append(random.choice(fill))
    random.shuffle(col)
    return col


def _canonical_slot_signatures(right_col: list[int]) -> list[tuple[int, int]]:
    n = len(right_col)
    return [(int(right_col[i]), min(i, n - 1 - i)) for i in range(n)]


def _design_template(N: int) -> tuple[np.ndarray, list[int]] | None:
    for _ in range(_TEMPLATE_RETRIES):
        num_colors = random.randint(*_TEMPLATE_COLOR_COUNT)
        template_colors = random.sample(_NON_BG_COLORS, num_colors)

        active_majority = random.randint(_ACTIVE_MAJORITY_RANGE[0], min(N, _ACTIVE_MAJORITY_RANGE[1]))
        left_col = _build_column(N, template_colors, active_majority, random.choice(template_colors))

        max_right_majority = active_majority - 1
        min_right_majority = max(2, N // num_colors)
        if min_right_majority > max_right_majority:
            continue
        right_col = _build_column(
            N,
            template_colors,
            random.randint(min_right_majority, max_right_majority),
            random.choice(template_colors),
        )

        if max(Counter(left_col).values()) <= max(Counter(right_col).values()):
            continue

        signatures = _canonical_slot_signatures(right_col)
        if len(signatures) != len(set(signatures)):
            continue

        template = np.column_stack([left_col, right_col]).astype(int)
        if len(set(map(int, template.flat))) < num_colors:
            continue
        return template, template_colors
    return None


def _deduplicated_variants(template: np.ndarray) -> list[np.ndarray]:
    seen: list[np.ndarray] = []
    result: list[np.ndarray] = []
    for _name, sub in _dihedral_variants(template):
        if any(s.shape == sub.shape and np.array_equal(s, sub) for s in seen):
            continue
        try:
            _slot_descriptors(sub)
        except AssertionError:
            continue
        seen.append(sub)
        result.append(sub)
    return result


def _random_sprite_position(H: int, W: int, h: int, w: int) -> tuple[int, int]:
    return (
        random.randint(_BOUNDARY_MARGIN, H - h - _BOUNDARY_MARGIN),
        random.randint(_BOUNDARY_MARGIN, W - w - _BOUNDARY_MARGIN),
    )


def _occupied_set(placements: list[tuple]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        for r in range(h):
            for c in range(w):
                cells.add((r0 + r, c0 + c))
    return cells


def _place_sprites(H: int, W: int, variants: list[np.ndarray]) -> list[tuple] | None:
    placements: list[tuple] = []
    pad = _PLACEMENT_PAD
    for sub in variants:
        h, w = sub.shape
        direction, label_direction, slots = _slot_descriptors(sub)
        placed = False
        for _ in range(_PLACEMENT_RETRIES):
            r0, c0 = _random_sprite_position(H, W, h, w)
            occ = _occupied_set(placements)
            if any((r0 + r, c0 + c) in occ for r in range(-pad, h + pad) for c in range(-pad, w + pad)):
                continue
            placements.append((r0, c0, sub, direction, label_direction, slots))
            placed = True
            break
        if not placed:
            return None
    return placements


def _boundary_position(r: int, c: int, dr: int, dc: int, H: int, W: int) -> tuple[int, int]:
    if dr == 0 and dc == -1:
        return r, 0
    if dr == 0 and dc == 1:
        return r, W - 1
    if dr == -1 and dc == 0:
        return 0, c
    return H - 1, c


def _line_clear_to_boundary(grid: np.ndarray, r: int, c: int, dr: int, dc: int, stop: tuple[int, int]) -> bool:
    bg = int(_BG)
    r += dr
    c += dc
    while 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
        if (r, c) == stop:
            return int(grid[r, c]) == bg
        if int(grid[r, c]) != bg:
            return False
        r += dr
        c += dc
    return False


def _validate_marker_visibility(
    grid: np.ndarray,
    placements: list[tuple],
    marker_pos_to_color: dict[tuple[int, int], int],
    color_to_signature: dict[int, tuple[int, int]],
) -> bool:
    seen_counts = Counter()
    seen_sigs: dict[int, set[tuple[int, int]]] = {color: set() for color in color_to_signature}
    for r0, c0, _sub, _direction, label_direction, slots in placements:
        ldr, ldc = label_direction
        for slot in slots:
            lr, lc = slot["label"]
            color = _first_marker_on_ray(grid, _BG, marker_pos_to_color, r0 + lr, c0 + lc, ldr, ldc)
            if color is None:
                continue
            seen_counts[color] += 1
            seen_sigs.setdefault(color, set()).add(slot["signature"])

    for color, signature in color_to_signature.items():
        if seen_counts[color] != 1:
            return False
        if seen_sigs.get(color) != {signature}:
            return False
    return True


def _place_markers(grid: np.ndarray, sprite_placements: list[tuple], template_colors: list[int]) -> bool:
    H, W = grid.shape
    signatures = [slot["signature"] for slot in sprite_placements[0][5]]
    if len(signatures) < _NUM_MARKERS:
        return False

    labeled_signatures = random.sample(signatures, _NUM_MARKERS)
    available = [c for c in range(10) if c != _BG and c not in template_colors]
    if len(available) < _NUM_MARKERS:
        return False
    marker_colors = random.sample(available, _NUM_MARKERS)

    marker_pos_to_color: dict[tuple[int, int], int] = {}
    color_to_signature: dict[int, tuple[int, int]] = {}
    for color, signature in zip(marker_colors, labeled_signatures):
        r0, c0, _sub, _direction, label_direction, slots = random.choice(sprite_placements)
        matched = [slot for slot in slots if slot["signature"] == signature]
        if len(matched) != 1:
            return False
        slot = matched[0]
        lr, lc = slot["label"]
        ldr, ldc = label_direction
        marker_r, marker_c = _boundary_position(r0 + lr, c0 + lc, ldr, ldc, H, W)

        if not _line_clear_to_boundary(grid, r0 + lr, c0 + lc, ldr, ldc, (marker_r, marker_c)):
            return False
        if int(grid[marker_r, marker_c]) != _BG:
            return False

        grid[marker_r, marker_c] = color
        marker_pos_to_color[(marker_r, marker_c)] = color
        color_to_signature[color] = signature

    return _validate_marker_visibility(grid, sprite_placements, marker_pos_to_color, color_to_signature)


def _gen_attempt() -> np.ndarray | None:
    N = random.randint(*_TEMPLATE_N_RANGE)
    H = random.randint(*_GRID_H_RANGE)
    W = random.randint(*_GRID_W_RANGE)

    design = _design_template(N)
    if design is None:
        return None
    template, template_colors = design

    unique_variants = _deduplicated_variants(template)
    num_sprites = random.randint(*_SPRITE_COUNT_RANGE)
    if len(unique_variants) < num_sprites:
        return None
    chosen = random.sample(unique_variants, num_sprites)

    placements = _place_sprites(H, W, chosen)
    if placements is None:
        return None

    grid = np.full((H, W), _BG, dtype=int)
    for r0, c0, sub, *_ in placements:
        h, w = sub.shape
        grid[r0 : r0 + h, c0 : c0 + w] = sub

    if not _place_markers(grid, placements, template_colors):
        return None

    try:
        main(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
