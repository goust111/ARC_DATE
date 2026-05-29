# Combo variant of 8698868d (index 5)
# combo_spec: D1/2+D8/0
# Name: 面积匹配多像素框架洞

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location

# Design analysis
# ===============
# spec_variant:
#   preserved:
#     - 背景仍固定为 8；输入仍由一片紧密排列的大方框网格和若干散落的小 sprite 组成。
#     - 输出仍只保留框架网格，并把匹配到的 sprite 填进对应框架内部。
#     - sprite 内部的洞像素在输出中仍会被重着色为对应 frame 的颜色。
#   fused_change:
#     - frame_key 不再由“洞像素个数”决定，而是由 frame 内部 4-连通洞分量个数决定。
#     - 这些 frame 洞分量现在允许是多像素连通块，不再限制为单像素点。
#     - sprite_key 改为 sprite 内全部洞像素的总面积，而不是洞分量个数。
#   concrete_trace:
#     - 例如某个 sprite 只有 1 个洞分量，但这个洞块占 4 个像素，则 sprite_key = 4。
#     - 与之匹配的 frame 需要有 4 个洞分量；其中一个分量可以是 2 格连通块，另外
#       3 个分量是单格，于是这个 frame 的洞像素总数是 5，但 frame_key 仍然是 4。
#     - 原题会在 frame 与 sprite 两侧都数“几个洞块”；本融合变体要求在 sprite 一侧
#       数像素面积，在 frame 一侧数连通分量，还要忽略 frame 中多像素洞带来的视觉噪声。
#   complexity_calibration:
#     - 为了让 D1/2 可观察，生成器让每个 sprite 默认只有 1 个多像素洞分量，其面积取
#       连续整数 2..N+1。这样 sprite 的“面积”和“分量数”稳定分离。
#     - 为了让 D8/0 可观察，生成器强制每个 frame 至少有 1 个多像素洞分量，因此 frame
#       的“洞像素总数”通常严格大于“洞分量数”。
#     - key 取连续的 2..N+1 而非从 0 开始，避免旧规则在 0/1 上偶然与新规则重合。
#   implicit_assumptions_review:
#     - `find_components(... monochromatic=True)` 依赖 frame 和 sprite 主体均为单色连通块。
#       生成器继续只使用主体色 + 背景色两种像素，保证该假设成立。
#     - 方形筛选 `area >= 0.5 * h * h` 依赖对象填充率不过低。这里 sprite 最大洞面积受
#       内层尺寸限制，frame 只做少量洞扩张，因此所有对象都远高于 0.5。
#     - frame 颜色仍通过边框众数恢复；这要求四边完全实心且不被洞触及。生成器只在内部
#       区域生长洞块，边框保持纯净。
#     - solver 用 `inner != frame_color` / `region != sprite_color` 抽取洞；这要求对象内部
#       的非主体色像素都是真洞。生成器不引入第三种颜色，故可直接成立。
#     - 匹配逻辑依赖 key 唯一。生成器令 frame 分量数与 sprite 洞面积都恰好是连续整数
#       2..N+1 的一个排列，因此配对唯一。
#     - 网格恢复逻辑依赖最大的方块尺寸是 frame，次大的方块尺寸是 sprite。生成器保留
#       `sprite_size = frame_size - 2` 且两类对象都为正方形，满足该前提。
#   placement_legality:
#     - frame 网格仍无缝贴合地整体落在大画布某处。
#     - sprite 仍通过 `random_free_location(..., padding=1)` 放在背景空区，保证不重叠、
#       不越界、也不接触 frame 或其他 sprite。
#   visual_observability:
#     - 人类会看到 sprite 侧通常是“一个连成块的洞”，frame 侧则是“若干洞分量，其中有
#       的分量占多格”。输出若正确，说明不能只数洞块，也不能只数洞像素。
#     - 同一组样例里 frame 的洞像素总数常常不等于 sprite 的洞面积；真正稳定对应的是
#       “frame 洞分量数 = sprite 洞面积”，因此融合规则在 2-4 个样例中可被肉眼归纳。
#   cleanliness:
#     - 画面中仍只有两类方形对象；洞都只出现在对象内部，没有额外标记叠加。
#     - frame 洞块虽然变复杂，但仍被限制在小面积、少量扩张的范围内，避免视觉失控。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair scattered sprites with a packed frame grid by matching sprite hole area
    to frame hole-component count, then place each sprite into its frame.

    Input structure:
    - Background is azure (8). Large square frames form a tight grid; smaller square
      sprites appear elsewhere on the background.
    - Each object is monochrome except for bg-colored holes. Sprite holes may form
      one connected blob; frame holes may be several connected components, and a
      frame hole component may itself span multiple pixels.

    Rule:
    - Recover the frame grid from the largest square components.
    - For each frame, count how many 4-connected hole components appear inside it.
    - For each sprite, measure the total number of hole pixels inside it.
    - Match equal keys, insert the sprite into the frame interior, and recolor the
      sprite's hole pixels to the frame color in the output grid.

    Key insight:
    - The two sides use different measurements: sprites are matched by hole area,
      while frames are matched by the number of hole components, even when a frame
      component occupies multiple pixels.
    """
    g = np.array(grid, dtype=int)
    BG = 8

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((h, comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != frame_color
            _labels, hole_components = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "color": frame_color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "hole_components": int(hole_components),
                }
            )

    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        sprites.append(
            {
                "color": sprite_color,
                "hole_area": int(np.sum(hole_mask)),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict] = {}
    used_sprites: set[int] = set()
    for frame in frame_grid:
        for i, sprite in enumerate(sprites):
            if i not in used_sprites and frame["hole_components"] == sprite["hole_area"]:
                matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        sprite = matches.get((fr, fc))
        if sprite is None:
            continue

        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
GEN_RETRIES = 300
HOLE_KEY_START = 2

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` interior cells in a size x size patch, mutually 4-isolated."""
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _grow_connected_hole(size: int, hole_area: int) -> set[tuple[int, int]] | None:
    """Return one 4-connected interior hole of exact area `hole_area`."""
    inner_side = size - 2
    if hole_area < 0 or hole_area > inner_side * inner_side:
        return None
    if hole_area == 0:
        return set()

    seed = (random.randrange(inner_side), random.randrange(inner_side))
    shape = {seed}
    while len(shape) < hole_area:
        frontier: set[tuple[int, int]] = set()
        for r, c in shape:
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < inner_side and 0 <= nc < inner_side and (nr, nc) not in shape:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        shape.add(random.choice(list(frontier)))

    return {(r + 1, c + 1) for r, c in shape}


def _component_frontier(
    size: int,
    components: list[set[tuple[int, int]]],
    occupied: set[tuple[int, int]],
    comp_idx: int,
) -> list[tuple[int, int]]:
    """Cells that can grow component `comp_idx` without merging into another component."""
    frontier: set[tuple[int, int]] = set()
    for r, c in components[comp_idx]:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr = r + dr
            nc = c + dc
            if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                continue
            if (nr, nc) in occupied:
                continue
            ok = True
            for other_idx, other in enumerate(components):
                if other_idx == comp_idx:
                    continue
                if (nr - 1, nc) in other or (nr + 1, nc) in other or (nr, nc - 1) in other or (nr, nc + 1) in other:
                    ok = False
                    break
            if ok:
                frontier.add((nr, nc))
    return list(frontier)


def _hole_component_sizes(patch: np.ndarray, color: int) -> list[int]:
    hole_mask = patch != color
    labels, num = label_mask(hole_mask, connectivity=4)
    sizes: list[int] = []
    for idx in range(1, int(num) + 1):
        sizes.append(int(np.sum(labels == idx)))
    return sizes


def _make_sprite(size: int, hole_area: int, sprite_color: int) -> np.ndarray | None:
    """
    Build a sprite with a solid border and exactly one 4-connected bg hole whose
    total area equals `hole_area`.
    """
    sprite = np.full((size, size), sprite_color, dtype=int)
    hole_cells = _grow_connected_hole(size, hole_area)
    if hole_cells is None:
        return None

    for r, c in hole_cells:
        sprite[r, c] = BG

    actual_area = int(np.sum(sprite != sprite_color))
    _labels, num_holes = label_mask(sprite != sprite_color, connectivity=4)
    if actual_area != hole_area or num_holes != 1:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """
    Build a frame with exactly `hole_count` 4-connected hole components and force
    at least one of those components to occupy multiple pixels.
    """
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    components = [{seed} for seed in seeds]
    occupied = set(seeds)
    max_extra = (frame_size - 2) ** 2 - hole_count
    if max_extra <= 0:
        return None

    extra_cells = random.randint(1, min(3, max_extra))
    grew = False
    for _ in range(extra_cells):
        growable = [idx for idx in range(len(components)) if _component_frontier(frame_size, components, occupied, idx)]
        if not growable:
            break
        comp_idx = random.choice(growable)
        cell = random.choice(_component_frontier(frame_size, components, occupied, comp_idx))
        components[comp_idx].add(cell)
        occupied.add(cell)
        grew = True

    if not grew:
        return None

    for r, c in occupied:
        frame[r, c] = BG

    component_sizes = _hole_component_sizes(frame, frame_color)
    if len(component_sizes) != hole_count:
        return None
    if max(component_sizes, default=0) <= 1:
        return None
    return frame


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_area = {key: patch for key, patch in sprites}
    for idx, (key, _patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = ri * frame_size
        cc = ci * frame_size
        frame_color = frame_colors[idx]
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_area[key].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _feasible_ns(frame_size: int) -> list[int]:
    sprite_size = frame_size - 2
    max_sprite_hole_area = (sprite_size - 2) ** 2
    max_n_from_sprite = max_sprite_hole_area - HOLE_KEY_START + 1
    return [n for n in sorted(_LAYOUTS_BY_N) if n <= max_n_from_sprite]


def _assign_sprite_colors(frame_colors_by_key: dict[int, int], sprite_keys: list[int]) -> list[int] | None:
    """
    Choose unique sprite colors so that the sprite matched to each key never shares
    the same color as its target frame. This keeps the new output visibly distinct
    from the original count-count solver.
    """
    available = [c for c in range(10) if c != BG]
    order = list(range(len(sprite_keys)))
    random.shuffle(order)
    sprite_colors = [0] * len(sprite_keys)
    used: set[int] = set()

    for idx in order:
        forbidden = frame_colors_by_key[sprite_keys[idx]]
        choices = [c for c in available if c not in used and c != forbidden]
        if not choices:
            return None
        sprite_colors[idx] = random.choice(choices)
        used.add(sprite_colors[idx])
    return sprite_colors


def _sample_exhibits_fusion(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    sprite_colors: list[int],
) -> bool:
    """
    Reject degenerate samples that visually collapse back toward the original task.
    """
    frame_has_multi = False
    for (key, patch), color in zip(frames, frame_colors):
        component_sizes = _hole_component_sizes(patch, color)
        if len(component_sizes) != key:
            return False
        if max(component_sizes, default=0) > 1:
            frame_has_multi = True

    sprite_all_single_component = True
    sprite_areas_match_keys = True
    for (key, patch), color in zip(sprites, sprite_colors):
        hole_mask = patch != color
        hole_area = int(np.sum(hole_mask))
        _labels, num = label_mask(hole_mask, connectivity=4)
        if hole_area != key:
            sprite_areas_match_keys = False
        if num != 1:
            sprite_all_single_component = False

    return frame_has_multi and sprite_all_single_component and sprite_areas_match_keys


def _gen_attempt() -> np.ndarray | None:
    frame_size = random.choice(FRAME_SIZE_RANGE)
    feasible_ns = _feasible_ns(frame_size)
    if not feasible_ns:
        return None

    N = random.choice(feasible_ns)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, N)

    keys = list(range(HOLE_KEY_START, HOLE_KEY_START + N))
    frame_key_perm = keys[:]
    sprite_key_perm = keys[:]
    random.shuffle(frame_key_perm)
    random.shuffle(sprite_key_perm)

    frame_colors_by_key = {key: frame_colors[idx] for idx, key in enumerate(frame_key_perm)}
    sprite_colors = _assign_sprite_colors(frame_colors_by_key, sprite_key_perm)
    if sprite_colors is None:
        return None

    sprites: list[tuple[int, np.ndarray]] = []
    for i, area in enumerate(sprite_key_perm):
        patch = _make_sprite(sprite_size, area, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((area, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(frame_key_perm):
        patch = _make_frame(frame_size, hole_count, frame_colors[i])
        if patch is None:
            return None
        frames.append((hole_count, patch))

    if not _sample_exhibits_fusion(frames, sprites, frame_colors, sprite_colors):
        return None

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_key, frame_patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = frame_patch

    for _key, sprite_patch in sprites:
        try:
            r, c = random_free_location(canvas, sprite_patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, sprite_patch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
