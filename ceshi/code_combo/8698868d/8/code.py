# Combo variant of 8698868d (index 8)
# combo_spec: D2/2+D4/3
# Name: 多色精灵第三色填洞

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The original task already has two object families:
#      1) large monochrome square frames in a tight grid,
#      2) smaller square sprites scattered elsewhere.
#    - The D2 change modifies only the sprites: they keep a solid 1-cell border,
#      but their non-hole pixels are no longer all one color. Each sprite now has
#      a two-color checker-style pattern, so after nesting the internal pattern is
#      preserved instead of collapsing to one color.
#    - The D4 change modifies only the hole fill after matching: a compact legend
#      maps each frame color to a third color. After a sprite is nested into its
#      matched frame, only the sprite's bg-colored holes are recolored, and they
#      become the legend color rather than the frame color.
#    - Small trace:
#         original behavior:
#           red frame + green sprite with 2 hole-components -> holes become red.
#         combo behavior:
#           legend shows red above yellow, sprite is green/cyan checker ->
#           same hole-count match still happens, the green/cyan pattern stays,
#           and the holes become yellow.
#      The combined effect is visible in one output cell block: preserved
#      multi-color sprite pattern plus a new fill color that is neither sprite
#      color nor frame border color.
#
# b) Complexity calibration:
#    - Sprites use exactly two non-bg colors, not an arbitrary palette. This is
#      enough to visibly differ from the original task while keeping the pattern
#      easy to parse from 2-3 examples.
#    - The legend stays a tiny 2-row strip of aligned 2x2 chips. That keeps the
#      new color-transfer signal explicit and structurally separate from frames
#      and sprites.
#    - Frame count is capped at 2-3. This keeps the one-to-one legend readable
#      and leaves enough color budget to keep frame colors, sprite colors, and
#      hole-fill colors globally separated.
#
# c) Hidden assumptions in the original solver:
#    - The original sprite detection assumed each sprite is a single monochrome
#      component, because it used `find_components(..., monochromatic=True)` and
#      then treated the sprite component color as "the sprite color". That fails
#      here: one sprite splits into several monochrome components.
#    - The original hole counting line
#          hole_mask = sprite_region != sprite_color
#      also fails, because non-hole accent pixels would be mistaken for holes.
#      In the combo variant, sprite holes must be detected as bg cells inside a
#      square sprite bbox, not as "anything not equal to one anchor color".
#    - The original recolor line
#          content[content != sprite_color] = frame_color
#      depended on both old assumptions at once. The new solver must preserve all
#      non-bg sprite colors and recolor only `BG` cells via the legend mapping.
#    - Legend decoding introduces a new assumption: the only aligned two-row set
#      of 2x2 monochrome blocks is the legend. The generator enforces a fixed
#      layout and uses checker-style sprite interiors to avoid accidental extra
#      2x2 monochrome chips.
#
# d) Placement legality:
#    - Frames still occupy one tight rectangle on the canvas.
#    - The legend is placed as an isolated patch with padding, so its chips do not
#      merge with frames or sprites even if colors repeat.
#    - Sprites are placed in background with 1-cell padding. That lets the solver
#      safely recover them by scanning for isolated square patches of size
#      `frame_size - 2` outside the frame-grid bbox and legend bbox.
#
# e) Visual observability:
#    - The output differs from the original in two human-visible ways at once:
#      checker pixels survive inside the nested sprite, and hole pixels switch to
#      a third color exposed by the legend.
#    - Because the same legend format repeats for every frame in one sample, the
#      color-transfer rule is inferable spatially rather than numerically.
#
# f) Visual cleanliness:
#    - The sprite pattern is limited to a regular two-color checker. That prevents
#      busy, noisy interiors while still making "multi-color preservation" obvious.
#    - Metadata remains confined to one compact legend block, so the input still
#      cleanly separates frames, sprites, and mapping information.
#
# g) Change inferability:
#    - Hole-fill colors are globally disjoint from every frame color and every
#      sprite color. That prevents degenerate samples where the combo would
#      accidentally look like the original task or where the "third color"
#      semantics would be visually muddied by reuse elsewhere on the canvas.
#    - Hole-count matching is unchanged, so the puzzle's structural logic stays
#      stable while the fused color rule is learned from repeated examples.
#
# 9) New solver assumptions checked against the generator:
#    - Largest square monochrome components are always the frames; sprites are
#      smaller and multi-color, so their monochrome fragments never define the
#      frame size.
#    - Every sprite has size `frame_size - 2`, a solid border, at least two
#      visible non-bg colors, and a clear 1-cell bg moat around its bbox.
#    - Every frame color that appears in the grid is represented exactly once in
#      the legend.
#    - The legend occupies the only aligned two-row family of 2x2 monochrome
#      chips, and the solver uses that whole structure rather than arbitrary chips.


BG = 8

FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 3)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
LEGEND_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 50
SPRITE_PATTERN_RETRIES = 80
GEN_RETRIES = 400
LEGEND_BLOCK = 2
LEGEND_GAP = 1

_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _bbox_union(bboxes: list[tuple[int, int, int, int]]) -> tuple[int, int, int, int]:
    return (
        min(int(b[0]) for b in bboxes),
        min(int(b[1]) for b in bboxes),
        max(int(b[2]) for b in bboxes),
        max(int(b[3]) for b in bboxes),
    )


def _legend_dims(n: int) -> tuple[int, int]:
    return 2 * LEGEND_BLOCK + LEGEND_GAP, 3 * n - 1


def _extract_legend_mapping(comps: list[dict]) -> tuple[dict[int, int], tuple[int, int, int, int]]:
    """Decode the aligned two-row 2x2-chip legend."""
    chips = []
    for comp in comps:
        b = comp["bbox"]
        if int(bbox_h(b)) == LEGEND_BLOCK and int(bbox_w(b)) == LEGEND_BLOCK and int(comp["area"]) == 4:
            chips.append(comp)

    assert len(chips) >= 4
    chips_by_r: dict[int, list[dict]] = {}
    for comp in chips:
        chips_by_r.setdefault(int(comp["bbox"][0]), []).append(comp)

    best_pair: tuple[list[dict], list[dict]] | None = None
    best_cols: list[int] = []
    row_starts = sorted(chips_by_r)
    for top_r in row_starts:
        bottom_r = int(top_r + LEGEND_BLOCK + LEGEND_GAP)
        if bottom_r not in chips_by_r:
            continue
        top_by_c = {int(comp["bbox"][1]): comp for comp in chips_by_r[top_r]}
        bottom_by_c = {int(comp["bbox"][1]): comp for comp in chips_by_r[bottom_r]}
        common_cols = sorted(set(top_by_c) & set(bottom_by_c))
        if len(common_cols) < N_RANGE[0]:
            continue
        if len(common_cols) != len(top_by_c) or len(common_cols) != len(bottom_by_c):
            continue
        if common_cols != list(range(common_cols[0], common_cols[0] + 3 * len(common_cols), 3)):
            continue
        if len(common_cols) > len(best_cols):
            best_cols = common_cols
            best_pair = ([top_by_c[c0] for c0 in common_cols], [bottom_by_c[c0] for c0 in common_cols])

    assert best_pair is not None
    top_row, bottom_row = best_pair
    mapping: dict[int, int] = {}
    for top_chip, bottom_chip in zip(top_row, bottom_row):
        frame_color = int(top_chip["color"])
        fill_color = int(bottom_chip["color"])
        assert frame_color not in mapping
        mapping[frame_color] = fill_color

    legend_bbox = _bbox_union([comp["bbox"] for comp in top_row + bottom_row])
    return mapping, legend_bbox


def _find_frame_components(comps: list[dict]) -> tuple[int, list[dict]]:
    """Frames are the largest square monochrome regions with moderate fill."""
    square_regions: list[tuple[int, dict]] = []
    min_fill_ratio = 0.5
    for comp in comps:
        b = comp["bbox"]
        h = int(bbox_h(b))
        w = int(bbox_w(b))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((h, comp))

    assert square_regions
    frame_size = max(size for size, _ in square_regions)
    frames = [comp for size, comp in square_regions if int(size) == int(frame_size)]
    assert frames
    return int(frame_size), frames


def _frame_grid_bbox(frames: list[dict], frame_size: int) -> tuple[int, int, int, int, int, int]:
    positions = [(int(comp["bbox"][0]), int(comp["bbox"][1])) for comp in frames]
    min_r = min(r for r, _ in positions)
    min_c = min(c for _, c in positions)
    max_r = max(r for r, _ in positions)
    max_c = max(c for _, c in positions)
    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)
    return min_r, min_c, n_rows, n_cols, max_r + frame_size - 1, max_c + frame_size - 1


def _build_blocked_mask(
    shape: tuple[int, int],
    frame_box: tuple[int, int, int, int],
    legend_box: tuple[int, int, int, int],
) -> np.ndarray:
    blocked = np.zeros(shape, dtype=bool)
    fr0, fc0, fr1, fc1 = frame_box
    lr0, lc0, lr1, lc1 = legend_box
    blocked[fr0 : fr1 + 1, fc0 : fc1 + 1] = True
    blocked[lr0 : lr1 + 1, lc0 : lc1 + 1] = True
    return blocked


def _ring_is_clear(g: np.ndarray, r0: int, c0: int, size: int) -> bool:
    H, W = g.shape
    rr0 = max(0, int(r0 - 1))
    cc0 = max(0, int(c0 - 1))
    rr1 = min(H - 1, int(r0 + size))
    cc1 = min(W - 1, int(c0 + size))
    for rr in range(rr0, rr1 + 1):
        for cc in range(cc0, cc1 + 1):
            inside = int(r0) <= rr < int(r0 + size) and int(c0) <= cc < int(c0 + size)
            if not inside and int(g[rr, cc]) != BG:
                return False
    return True


def _scan_sprite_at(
    g: np.ndarray,
    r0: int,
    c0: int,
    size: int,
    blocked: np.ndarray,
) -> dict | None:
    patch = g[r0 : r0 + size, c0 : c0 + size]
    if patch.shape != (size, size):
        return None
    if bool(blocked[r0 : r0 + size, c0 : c0 + size].any()):
        return None

    border = np.concatenate([patch[0, :], patch[-1, :], patch[:, 0], patch[:, -1]])
    border_color = int(border[0])
    if border_color == BG or not np.all(border == border_color):
        return None
    if not _ring_is_clear(g, r0, c0, size):
        return None

    non_bg = patch[patch != BG]
    if non_bg.size == 0:
        return None
    colors = {int(v) for v in np.unique(non_bg)}
    if len(colors) < 2:
        return None

    non_bg_mask = patch != BG
    _, support_components = label_mask(non_bg_mask, connectivity=4)
    if int(support_components) != 1:
        return None

    hole_mask = patch == BG
    _, hole_count = label_mask(hole_mask, connectivity=4)
    return {
        "top": int(r0),
        "left": int(c0),
        "holes": int(hole_count),
        "content": patch.copy(),
        "h": int(size),
        "w": int(size),
    }


def _detect_sprites(
    g: np.ndarray,
    sprite_size: int,
    blocked: np.ndarray,
) -> list[dict]:
    sprites: list[dict] = []
    H, W = g.shape
    for r0 in range(H - sprite_size + 1):
        for c0 in range(W - sprite_size + 1):
            sprite = _scan_sprite_at(g, r0, c0, sprite_size, blocked)
            if sprite is not None:
                sprites.append(sprite)
    return sprites


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match patterned sprites to frames by connected-hole count, then nest each sprite
    while filling its holes with a legend-defined third color.

    Input structure:
    - bg = azure (8).
    - Large monochrome square frames form one tight grid.
    - Smaller square sprites are scattered elsewhere, each with a solid border and a
      preserved two-color interior pattern plus bg-colored holes.
    - A compact two-row legend of 2x2 chips maps frame border colors to hole-fill
      colors.

    Rule:
    - Read the legend to obtain a frame-color -> fill-color mapping.
    - Count 4-connected hole components inside each frame interior and inside each
      sprite.
    - Pair each frame with the sprite having the same hole-component count.
    - Place the matched sprite inside the frame, keep its non-bg pattern unchanged,
      and recolor only its bg holes to the legend color for that frame.

    Key insight:
    - The combinational change is that sprite matching still depends only on hole
      count, but sprite detection and recoloring no longer depend on a single sprite
      color. Holes are the bg cells inside a patterned square sprite, and their new
      color comes from the separate legend rather than from the frame itself.
    """
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    legend_map, legend_bbox = _extract_legend_mapping(comps)
    frame_size, frames = _find_frame_components(comps)
    sprite_size = int(frame_size - 2)

    min_r, min_c, n_rows, n_cols, max_r1, max_c1 = _frame_grid_bbox(frames, frame_size)
    frame_box = (int(min_r), int(min_c), int(max_r1), int(max_c1))
    blocked = _build_blocked_mask(g.shape, frame_box, legend_bbox)

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
            _, num_holes = label_mask(hole_mask, connectivity=4)
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
                }
            )

    assert len(frame_grid) == len(frames) == len(legend_map)
    for frame in frame_grid:
        assert int(frame["color"]) in legend_map

    sprites = _detect_sprites(g, sprite_size, blocked)
    assert len(sprites) == len(frame_grid)

    sprites_by_hole = {int(sprite["holes"]): sprite for sprite in sprites}
    assert len(sprites_by_hole) == len(sprites)

    out_h = int(n_rows * frame_size)
    out_w = int(n_cols * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        fill_color = int(legend_map[frame_color])
        sprite = sprites_by_hole[int(frame["holes"])]

        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color
        content = np.array(sprite["content"], dtype=int).copy()
        content[content == BG] = fill_color
        out[fr + 1 : fr + 1 + sprite_size, fc + 1 : fc + 1 + sprite_size] = content

    return out


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place interior seed cells that are pairwise 4-separated."""
    inner_positions = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1)]
    for _ in range(SEED_RETRIES):
        random.shuffle(inner_positions)
        seeds: list[tuple[int, int]] = []
        for r, c in inner_positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((int(r), int(c)))
                if len(seeds) == count:
                    return seeds
    return None


def _make_hole_mask(size: int, hole_count: int) -> np.ndarray | None:
    """Build a hole mask with exactly `hole_count` 4-connected components."""
    hole = np.zeros((size, size), dtype=bool)
    if hole_count == 0:
        return hole

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        hole[r, c] = True

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow = random.randint(0, min(max_grow, inner_cells - len(all_hole)))
        for _ in range(grow):
            frontier = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = int(r + dr), int(c + dc)
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
            hole[cell[0], cell[1]] = True

    _, num = label_mask(hole, connectivity=4)
    if int(num) != hole_count:
        return None
    return hole


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build a monochrome frame patch with isolated bg holes inside."""
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    if hole_count == 0:
        return frame

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None
    for r, c in seeds:
        frame[r, c] = BG

    _, num = label_mask(frame != frame_color, connectivity=4)
    if int(num) != hole_count:
        return None
    return frame


def _make_multicolor_sprite(
    size: int,
    hole_count: int,
    border_color: int,
    accent_color: int,
    phase: int,
) -> np.ndarray | None:
    """Build a sprite with a solid border, checker interior, and bg holes."""
    for _ in range(SPRITE_PATTERN_RETRIES):
        hole_mask = _make_hole_mask(size, hole_count)
        if hole_mask is None:
            return None

        sprite = np.full((size, size), int(border_color), dtype=int)
        for r in range(1, size - 1):
            for c in range(1, size - 1):
                if hole_mask[r, c]:
                    sprite[r, c] = BG
                elif (r + c + phase) % 2 == 0:
                    sprite[r, c] = int(border_color)
                else:
                    sprite[r, c] = int(accent_color)

        if not np.any(sprite == int(accent_color)):
            continue
        non_bg_colors = {int(v) for v in np.unique(sprite) if int(v) != BG}
        if len(non_bg_colors) < 2:
            continue
        _, hole_components = label_mask(sprite == BG, connectivity=4)
        if int(hole_components) != hole_count:
            continue
        return sprite
    return None


def _make_legend(frame_colors: list[int], fill_colors: list[int]) -> np.ndarray:
    legend_h, legend_w = _legend_dims(len(frame_colors))
    legend = np.full((legend_h, legend_w), BG, dtype=int)
    bottom_r = LEGEND_BLOCK + LEGEND_GAP
    for i, (frame_color, fill_color) in enumerate(zip(frame_colors, fill_colors)):
        c0 = int(3 * i)
        legend[0:LEGEND_BLOCK, c0 : c0 + LEGEND_BLOCK] = int(frame_color)
        legend[bottom_r : bottom_r + LEGEND_BLOCK, c0 : c0 + LEGEND_BLOCK] = int(fill_color)
    return legend


def _choose_fill_colors(
    available: list[int],
    forbidden: set[int],
    count: int,
) -> list[int] | None:
    """Pick distinct fill colors disjoint from all visible non-fill object colors."""
    pool = [int(c) for c in available if int(c) not in forbidden]
    if len(pool) < count:
        return None
    return [int(c) for c in random.sample(pool, count)]


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_colors: list[int],
    fill_colors: list[int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_hole = {int(k): patch for k, patch in sprites}
    for idx, (hole_count, _) in enumerate(frames):
        rr = int((idx // cols_grid) * frame_size)
        cc = int((idx % cols_grid) * frame_size)
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_colors[idx])
        content = np.array(sprite_by_hole[int(hole_count)], dtype=int).copy()
        content[content == BG] = int(fill_colors[idx])
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = int(frame_size - 2)
    legend_h, legend_w = _legend_dims(n)

    grid_h = int(rows_grid * frame_size)
    grid_w = int(cols_grid * frame_size)
    min_h = max(CANVAS_MIN, grid_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_w + sprite_size + SPRITE_PADDING)
    if grid_h + legend_h + 1 <= CANVAS_MAX:
        min_h = max(min_h, grid_h + legend_h + 1)
    if grid_w + legend_w + 1 <= CANVAS_MAX:
        min_w = max(min_w, grid_w + legend_w + 1)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None

    H = random.randint(min_h, CANVAS_MAX)
    W = random.randint(min_w, CANVAS_MAX)
    available = [c for c in range(10) if c != BG]

    frame_colors = random.sample(available, n)
    sprite_color_pool = [int(c) for c in available if int(c) not in frame_colors]
    if len(sprite_color_pool) < 2:
        return None
    sprite_border_color, accent_color = random.sample(sprite_color_pool, 2)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    max_start = max(0, max_hole - n)
    start = random.randint(0, min(max_start, HOLE_START_MAX))
    hole_counts = list(range(start, start + n))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)
    phase = random.randint(0, 1)

    fill_colors = _choose_fill_colors(
        available,
        forbidden={int(c) for c in frame_colors} | {int(sprite_border_color), int(accent_color)},
        count=n,
    )
    if fill_colors is None:
        return None

    sprites: list[tuple[int, np.ndarray]] = []
    for hole_count in sprite_hole_perm:
        patch = _make_multicolor_sprite(
            sprite_size,
            int(hole_count),
            int(sprite_border_color),
            int(accent_color),
            phase,
        )
        if patch is None:
            return None
        sprites.append((int(hole_count), patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, hole_count in enumerate(hole_counts):
        patch = _make_frame(frame_size, int(hole_count), int(frame_colors[i]))
        if patch is None:
            return None
        frames.append((int(hole_count), patch))

    canvas = np.full((H, W), BG, dtype=int)
    r0 = random.randint(0, H - grid_h)
    c0 = random.randint(0, W - grid_w)
    for idx, (_, patch) in enumerate(frames):
        rr = int(r0 + (idx // cols_grid) * frame_size)
        cc = int(c0 + (idx % cols_grid) * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = patch

    legend = _make_legend(frame_colors, fill_colors)
    try:
        lr, lc = random_free_location(
            canvas,
            legend,
            transparent=BG,
            padding=LEGEND_PLACEMENT_PAD,
        )
    except ValueError:
        return None
    paste(canvas, legend, lr, lc, transparent=BG)

    for _, patch in sprites:
        try:
            sr, sc = random_free_location(
                canvas,
                patch,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, patch, sr, sc, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        frame_colors,
        fill_colors,
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
