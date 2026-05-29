# Combo variant of 8f215267 (index 11)
# combo_spec: D4/3+D5/1
# Name: 洞计数+垂直放置

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, label_mask
from arc_utils import paste, random_free_location

# Design analysis for the combo variant
# -------------------------------------
# combo_variant_spec:
#   concrete_trace:
#     - 原题中，若红框右侧有 3 个红色精灵，则输出在红框中心行画 3 个横向点。
#     - 本融合版中，若红色精灵分别含有 2 个洞、1 个洞、0 个洞，则红框的 N=3，
#       因为统计的是“内部连通洞总数”而不是“精灵个数”。输出改为在中心列自底向上画 3 个竖点。
#     - 因此同样是两个红精灵，也可能得到 3 个点；同时点的轴向从横向刻度改成纵向刻度。
#   complexity_calibration:
#     - 若保留原题 5x11 框，竖直方向最多只能稳定容纳 3 个可见内点，会削弱原题 0-4 的计数跨度。
#     - 这里把框高提升到 6、宽度保持 11，这样框内恰好有 4 个可见内部行位，仍能表达 0-4。
#     - 右侧精灵不再用任意小连通块，而改用少量模板化的 1 洞/2 洞单色精灵，避免洞计数退化成噪声。
#   solver_assumption_audit:
#     - 原 solver 把“大且稀疏”的单色连通域当作框。本版进一步依赖“框明显更宽”，
#       因此生成器把所有右侧精灵限制在宽度 <= 4，避免带洞精灵被误判为框。
#     - 原 solver 的计数行 `sum(1 for p in patterns if color match)` 已失效，
#       必须改成“逐个匹配色精灵，按 bbox 内部且不接触 bbox 边界的背景连通域数求和”。
#     - 原 solver 只依赖中心行和从右向左列坐标；本版改为中心列和自底向上的行坐标。
#   placement_legality:
#     - generator 仍采用 output-first：先按每个框的洞总数构造带竖点的输出，再删除竖点生成输入。
#     - 随后在右侧 9 列区域放置匹配精灵与干扰精灵，使用 `random_free_location(..., padding=1)`
#       保证各精灵互不重叠也不粘连。
#     - 每个匹配颜色的洞分配使用有限分拆，如 (2,1) 或 (2,2)，从而显式保证“总洞数”与“精灵个数”
#       在至少一个框上不同，避免退化回原题。
#   observability_and_cleanliness:
#     - 洞通过单色闭环里的背景孔洞直接可见，人类可以目测每个精灵是 1 洞还是 2 洞。
#     - 左侧框仍整齐纵向堆叠，右侧精灵仍集中在独立区域；输出只保留框和竖点，视觉层次清晰。
#     - 颜色仍只承担匹配关系，不依赖颜色编号大小等不可视因素。


def _vertical_dot_rows(bbox: tuple[int, int, int, int], n_dots: int) -> list[int]:
    """Rows for bottom-up dots inside a hollow frame bbox."""
    r0, _c0, r1, _c1 = map(int, bbox)
    return [int(r1 - 1 - i) for i in range(int(n_dots)) if int(r1 - 1 - i) > int(r0)]


def _count_enclosed_holes(patch: np.ndarray, color: int) -> int:
    """Count non-color 4-components that stay strictly inside the tight bbox."""
    hole_mask = np.array(patch != int(color), dtype=bool)
    lab, n_comp = label_mask(hole_mask, connectivity=4)
    if n_comp == 0:
        return 0

    h, w = map(int, patch.shape)
    holes = 0
    for k in range(1, int(n_comp) + 1):
        coords = np.argwhere(lab == k)
        if coords.size == 0:
            continue
        if np.all(
            (coords[:, 0] > 0)
            & (coords[:, 0] < h - 1)
            & (coords[:, 1] > 0)
            & (coords[:, 1] < w - 1)
        ):
            holes += 1
    return int(holes)


def _pattern_hole_count(grid: np.ndarray, comp: dict) -> int:
    """Hole count of a monochrome pattern component."""
    patch = crop_bbox(grid, comp["bbox"])
    return _count_enclosed_holes(patch, int(comp["color"]))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sum enclosed holes of same-color sprites per frame and show the total as vertical dots.

    Input structure:
    - bg = most common color.
    - Several wide hollow rectangular frames with distinct colors are stacked on the left.
    - Small monochrome sprites are scattered on the right; some are matching colors, some are distractors.
    - Matching sprites may contain 1 or more enclosed background holes inside their own tight bounding boxes.

    Rule:
    - Remove every scattered sprite from the grid.
    - For each frame color C, inspect all sprites of color C and count the total number of enclosed 4-connected holes across them.
    - Draw that total inside the frame on the center column, stacking dots from the bottom upward one row at a time.

    Key insight:
    - The tally is no longer based on how many sprites exist; it is based on the topology inside those sprites.
    - The in-frame display is also rotated relative to the original task: the answer appears as a vertical column rather than a horizontal row.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 9) and (area < h * w * 0.55)
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    pattern_holes = [_pattern_hole_count(g, p) for p in patterns]

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        color = int(f["color"])
        total_holes = sum(
            int(hole_count)
            for p, hole_count in zip(patterns, pattern_holes)
            if int(p["color"]) == int(color)
        )
        if total_holes <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        c_mid = (c0 + c1) // 2
        for rr in _vertical_dot_rows((r0, c0, r1, c1), total_holes):
            out[int(rr), int(c_mid)] = color

    return out


# --------------- generator (codex-att2, D4 hole-count + D5 vertical tally) ---------------

# --- layout geometry ---
FRAME_H = 6
FRAME_W = 11
FRAME_R0 = 0
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 3)
BOTTOM_PAD_RANGE = (0, 7)
MAX_DOTS_PER_FRAME = 4
MIN_TOTAL_PATTERNS = 5
MAX_TOTAL_PATTERNS = 8
MIN_TOTAL_HOLES = 5

# --- generator logic ---
MAX_ATTEMPTS = 400
MATCH_PARTITIONS = [(), (1,), (2,), (1, 1), (2, 1), (2, 2)]

_SOLID_TEMPLATES = [
    ("11", "11"),
    ("111", "111"),
    ("111", "110"),
    ("1111", "1110"),
]
_ONE_HOLE_TEMPLATES = [
    ("111", "101", "111"),
    ("111", "101", "101", "111"),
    ("1111", "1001", "1001", "1111"),
    ("111", "101", "101", "101", "111"),
]
_TWO_HOLE_TEMPLATES = [
    ("111", "101", "111", "101", "111"),
    ("1111", "1001", "1111", "1001", "1111"),
    ("1111", "1011", "1111", "1101", "1111"),
]


def _template_to_patch(template: tuple[str, ...], color: int, bg: int) -> np.ndarray:
    patch = np.full((len(template), len(template[0])), bg, dtype=int)
    for r, row in enumerate(template):
        for c, ch in enumerate(row):
            if ch == "1":
                patch[int(r), int(c)] = int(color)
    return patch


def _make_sprite_patch(
    rng: np.random.Generator,
    color: int,
    bg: int,
    hole_count: int,
) -> np.ndarray:
    if int(hole_count) == 0:
        template = _SOLID_TEMPLATES[int(rng.integers(0, len(_SOLID_TEMPLATES)))]
    elif int(hole_count) == 1:
        template = _ONE_HOLE_TEMPLATES[int(rng.integers(0, len(_ONE_HOLE_TEMPLATES)))]
    elif int(hole_count) == 2:
        template = _TWO_HOLE_TEMPLATES[int(rng.integers(0, len(_TWO_HOLE_TEMPLATES)))]
    else:
        raise ValueError(f"unsupported hole_count: {hole_count}")

    patch = _template_to_patch(template, color, bg)
    if bool(rng.integers(0, 2)):
        patch = np.flipud(patch)
    if bool(rng.integers(0, 2)):
        patch = np.fliplr(patch)
    return np.array(patch, dtype=int)


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    """Build a FRAME_H x FRAME_W hollow rectangle with n_dots on the center column."""
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color

    c_mid = FRAME_W // 2
    for rr in range(FRAME_H - 2, FRAME_H - 2 - int(n_dots), -1):
        patch[int(rr), int(c_mid)] = color
    return patch


def _sample_partitions(
    rng: np.random.Generator,
    n_frames: int,
    n_distractors: int,
) -> list[tuple[int, ...]] | None:
    for _ in range(200):
        parts = [
            tuple(int(x) for x in MATCH_PARTITIONS[int(rng.integers(0, len(MATCH_PARTITIONS)))])
            for _ in range(int(n_frames))
        ]
        total_holes = sum(sum(part) for part in parts)
        total_match_sprites = sum(len(part) for part in parts)
        total_patterns = total_match_sprites + int(n_distractors)
        active_frames = sum(1 for part in parts if part)
        has_fused_evidence = any((len(part) >= 2) and (2 in part) for part in parts)

        if active_frames < 2:
            continue
        if not has_fused_evidence:
            continue
        if not (MIN_TOTAL_PATTERNS <= total_patterns <= MAX_TOTAL_PATTERNS):
            continue
        if total_holes < MIN_TOTAL_HOLES:
            continue
        if any(sum(part) > MAX_DOTS_PER_FRAME for part in parts):
            continue
        return parts
    return None


def _count_matching_patterns(inp: np.ndarray) -> dict[int, int]:
    """Diagnostic helper: count same-color pattern components per color."""
    bg = int(get_bg(inp))
    comps = find_components(inp, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    counts: dict[int, int] = {}
    for comp in comps:
        h = int(bbox_h(comp["bbox"]))
        w = int(bbox_w(comp["bbox"]))
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 9) and (area < h * w * 0.55)
        if is_frame:
            continue
        color = int(comp["color"])
        counts[color] = counts.get(color, 0) + 1
    return counts


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    partitions = _sample_partitions(rng, n_frames, n_distractors)
    if partitions is None:
        return None

    palette = rng.choice(np.arange(10), size=1 + n_frames + n_distractors, replace=False)
    bg = int(palette[0])
    frame_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    distractor_colors = [int(x) for x in palette[1 + n_frames :]]

    hole_totals = [int(sum(part)) for part in partitions]
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (color, total_holes) in enumerate(zip(frame_colors, hole_totals)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, total_holes), r0, FRAME_C0)

    inp = out.copy()
    c_mid = FRAME_C0 + FRAME_W // 2
    for r0, total_holes in zip(frame_rows, hole_totals):
        for rr in range(r0 + FRAME_H - 2, r0 + FRAME_H - 2 - int(total_holes), -1):
            inp[int(rr), int(c_mid)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    sprite_specs: list[tuple[int, int]] = []
    for color, part in zip(frame_colors, partitions):
        sprite_specs.extend((int(color), int(hole_count)) for hole_count in part)
    sprite_specs.extend(
        (int(color), int(rng.integers(0, 3)))
        for color in distractor_colors
    )
    rng.shuffle(sprite_specs)

    try:
        for color, hole_count in sprite_specs:
            sprite = _make_sprite_patch(rng, color, bg, hole_count)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=1,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)

    # Oracle validation plus anti-degeneracy check against raw sprite counting.
    if not np.array_equal(main(inp), out):
        return None

    pattern_counts = _count_matching_patterns(inp)
    if not any(
        int(pattern_counts.get(int(color), 0)) != int(total_holes)
        for color, total_holes in zip(frame_colors, hole_totals)
    ):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
