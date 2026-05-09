from __future__ import annotations

# Variant of 8698868d
# Dimension: D13 — 背景色与固定参数 (meta)
# Change: 背景色固定为 azure (8)，solver 硬编码 BG=8... → 背景色随任务变化

from collections import Counter
import random

import numpy as np

from arc_utils import (
    bbox_h,
    bbox_w,
    crop_bbox,
    find_components,
    get_bg,
    label_mask,
    paste,
    random_free_location,
)

# Design analysis for D13
# a) Concrete behavior and trace
#    原题里所有“空白/洞”都恰好是 8，因此 solver 可以把 8 当成背景并忽略它。
#    这个变体要求背景色每题随机，所以视觉上仍然是：
#    - 一块大背景
#    - 若干大方框组成紧密网格
#    - 若干较小 sprite 散落在背景上
#    但“背景像素是什么颜色”不再固定。例如一题里背景可能是 3：
#    - 原逻辑: `find_components(..., bg=8)` 会把整块真实背景 3 也当成前景对象，后续尺寸分组失真。
#    - 新逻辑: 先统计全图颜色频率，取最常见颜色 3 作为背景，再做原有连通域/孔洞计数流程。
#    差异只发生在“预处理识别背景色”这一步；之后的匹配与输出构造保持原题一致。
# b) Complexity calibration
#    不引入新的匹配维度，仍只按 hole-count 配对，避免把任务改成多因素对应。
#    为了避免“背景不是 8 但旧 solver 仍可能侥幸成功”的退化，生成器在 bg != 8 时固定让至少一个
#    sprite 使用颜色 8，同时禁止 frame 使用 8；这样把 8 错认成背景会直接漏掉一个可见 sprite，
#    且不会被同色 frame 掩盖，D13 的差异会稳定生效。
# c) Implicit assumption audit
#    - 原 `BG = 8` 直接依赖旧属性，在 D13 下必错。
#    - 原 `find_components(... bg=BG)`、洞像素构造、sprite 放置透明色、oracle 输出构造都依赖 BG；
#      因此 solver 和 generator 里所有使用 BG 的位置都必须改为动态值。
#    - 原 `out = np.full(..., BG)` 在输出中最终会被 frame 全覆盖，本来“恰好无害”；这里保留动态 BG，
#      避免把旧常量继续藏在实现里。
# d) Placement legality
#    - frame grid 仍为紧密无缝网格。
#    - sprites 仍通过 `random_free_location(... transparent=bg)` 放在背景区域，不重叠、不越界。
#    - 新增 `bg must be most common` 校验，确保 solver 的频率推断前提被 generator 明确满足。
# e) Visual observability
#    人类能从 2-4 个样例中看到：虽然底色不同，但“把小图按洞数量塞回大框里”的规则不变。
#    变体可观察点是：不能再把某个固定颜色当作永远的背景，必须先识别哪种颜色铺满了大部分空间。
# f) Visual cleanliness
#    不增加新标记层，不改变对象结构；只是换底色，因此画面整洁度与原题基本一致。
# g) Color inference
#    颜色变化只有一个维度：背景色。它由“大面积底色 + 所有洞都与该底色同色”双重结构共同提示，
#    不是依赖颜色编号本身。


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, then nest each sprite inside its frame.

    D13 change:
    - Background is no longer fixed to color 8.
    - Infer the background as the most common color before running the original logic.
    """
    g = np.array(grid, dtype=int)
    BG = int(get_bg(g))

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Frames and sprites are both roughly square monochromatic components;
    # holey sprites need a relaxed fill threshold.
    MIN_FILL_RATIO = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * MIN_FILL_RATIO):
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

    # Assumption preserved from the original family: visible frames define a tight regular grid.
    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frames]
    min_r = min(r for r, _ in frame_positions)
    min_c = min(c for _, c in frame_positions)
    max_r = max(r for r, _ in frame_positions)
    max_c = max(c for _, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict[str, int]] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = g[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])

            inner = region[1:-1, 1:-1]
            hole_mask = inner != frame_color
            _lab, num_holes = label_mask(hole_mask, connectivity=4)

            frame_grid.append(
                {"color": frame_color, "out_r": int(ri * frame_size), "out_c": int(ci * frame_size), "holes": num_holes}
            )

    sprites: list[dict[str, int | np.ndarray]] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": num_holes,
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], tuple[dict, dict]] = {}
    used_sprites: set[int] = set()
    for f in frame_grid:
        for i, s in enumerate(sprites):
            if i not in used_sprites and int(f["holes"]) == int(s["holes"]):
                matches[(int(f["out_r"]), int(f["out_c"]))] = (f, s)
                used_sprites.add(i)
                break

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color

        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != sprite_color] = frame_color

            sh, sw = int(sprite["h"]), int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator (D13 variant) ---------------

FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
GEN_RETRIES = 300

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
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


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int, bg: int) -> np.ndarray | None:
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite
    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = bg

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_hole:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(comps)
                        if j != ci
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((nr, nc))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = bg

    _, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int, bg: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    if hole_count == 0:
        return frame
    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = bg

    _, num = label_mask(frame != frame_color, connectivity=4)
    if num != hole_count:
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
    bg: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), bg, dtype=int)
    sprite_by_k = {k: spatch for k, spatch in sprites}
    for idx, (k, _) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        fc = frame_colors[idx]
        expected[rr : rr + frame_size, cc : cc + frame_size] = fc
        content = sprite_by_k[k].copy()
        content[content == bg] = fc
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _bg_is_most_common(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    return int(counts.most_common(1)[0][0]) == int(bg)


def _gen_attempt() -> np.ndarray | None:
    bg = random.randint(0, 9)

    N = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[N])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None

    H = random.randint(min_H, CANVAS_MAX)
    W = random.randint(min_W, CANVAS_MAX)

    available = [c for c in range(10) if c != bg]
    if bg != 8:
        frame_available = [c for c in available if c != 8]
        frame_colors = random.sample(frame_available, N)
    else:
        frame_colors = random.sample(available, N)
    if bg != 8:
        other_sprite_colors = [c for c in available if c != 8]
        sprite_colors = [8, *random.sample(other_sprite_colors, N - 1)]
        random.shuffle(sprite_colors)
    else:
        sprite_colors = random.sample(available, N)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - N)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + N))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, k, sprite_colors[i], bg)
        if patch is None:
            return None
        sprites.append((k, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(hole_counts):
        patch = _make_frame(frame_size, k, frame_colors[i], bg)
        if patch is None:
            return None
        frames.append((k, patch))

    canvas = np.full((H, W), bg, dtype=int)
    r0 = random.randint(0, H - grid_pixel_h)
    c0 = random.randint(0, W - grid_pixel_w)
    for idx, (_, fpatch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    for _, spatch in sprites:
        try:
            r, c = random_free_location(canvas, spatch, transparent=bg, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=bg)

    # D13-specific legality: the solver infers background by frequency, so the
    # generator must only emit samples where that inference is unambiguous.
    if not _bg_is_most_common(canvas, bg):
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, frame_colors, rows_grid, cols_grid, frame_size, sprite_size, bg)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
