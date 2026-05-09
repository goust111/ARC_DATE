# Variant of 8f215267
# Dimension: D15 — 符号编码参数 (symbol_instructions)
# Change: 无符号编码——点的数量和位置完全由精灵计数和固定间距规则决定
#   -> 图例指定颜色映射。输入顶部边缘加入 2xN 颜色图例：上排给出精灵颜色，
#      下排给出其对应的框架颜色。solver 必须先查图例，再把精灵计数写入对应框架。
#
# Design analysis
# spec_variant:
#   preserved_structure:
#     grid_width: 23 fixed
#     frames:
#       shape: 5x11 hollow rectangles
#       count: 3-4
#       layout: vertically stacked on the left, same as original
#       tally_rendering: dots still go on the center row from right to left with step 2
#     sprites:
#       location: right half
#       shape: small monochrome connected components
#       output_behavior: all sprites are erased in the output
#   d15_change:
#     legend:
#       geometry: 2xN strip on the top edge, N = number of frames
#       location: rows 0-1, cols 13..(13+N-1)
#       encoding:
#         row0: sprite colors
#         row1: corresponding frame colors
#       semantics: a sprite of color legend[row0, c] contributes one dot to the frame
#         whose border color is legend[row1, c]
#     visual_effect:
#       original: count direct color matches between sprites and frames
#       variant: count via a visible lookup table
#   behavior_trace:
#     original_example:
#       frame_blue: 3 scattered blue sprites -> blue frame gets 3 dots
#     variant_example:
#       legend column: top=red, bottom=blue
#       input: 3 red sprites, blue frame
#       output: red sprites disappear, blue frame gets 3 blue dots
#       difference: direct same-color matching would give the blue frame 0 dots, so the
#         legend is now the only valid bridge between right-side sprites and left-side frames
#   complexity_calibration:
#     why_not_too_simple:
#       - The mapping is global and explicit, not hidden in color ids.
#       - All counted sprites are recolored away from their target frames, so the change is
#         visible in every sample rather than only in occasional non-identity cases.
#       - The legend has only 3-4 columns, which keeps it inferable from 2-3 examples.
#     why_not_too_complex:
#       - The legend is a single fixed-format strip.
#       - Mapping is one-to-one and uses only color, not shape, order-by-id, or sorting.
#   implicit_assumption_review:
#     original_main_lines_affected:
#       - "num_patterns = sum(... p['color'] == f['color'])" assumed direct color equality.
#         This is false in the variant; it must become a legend lookup.
#       - "remove all non-frame components" was harmless before, because every small component
#         outside frames was a sprite. In the variant, legend cells are also small components,
#         so solver must explicitly exclude the legend area from pattern removal.
#       - Frame detection by large hollow components remains valid, because the legend stays
#         small and isolated from frames.
#   placement_legality:
#     guarantees:
#       - Rows 0-1 are reserved for the legend by moving the first frame down to row 2.
#       - Sprites are placed only in rows >= 2, so they never overlap the legend.
#       - Right region starts at col 13, while frames end at col 11, so sprites and legend
#         never touch frames.
#       - Generator validates every sample with main(input) == output.
#   observability:
#     human_solver_view:
#       - Each sample shows the same stable 2-row codebook at the top.
#       - Frame colors and counted sprite colors are disjoint, so direct matching visibly fails.
#       - The output preserves the legend, making the color remapping clue persistent.
#   cleanliness:
#     - Left = frames, top-right = legend, right/below = sprites.
#     - No element types share the same footprint or overlay each other.
#   new_main_assumptions_to_guarantee:
#     - Legend occupies exactly rows 0-1 and has columns where both cells are non-bg.
#     - No other non-bg cells appear in rows 0-1.
#     - Each frame color appears once in the legend bottom row and each sprite color appears
#       once in the legend top row.
#     - Counted sprites use only legend top-row colors; distractors use colors absent from the legend.
#     - Generator enforces all of the above.

from __future__ import annotations

import numpy as np
from arc_utils import bbox_h, bbox_w, find_components, get_bg
from arc_utils import paste, random_free_location


# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 2
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = GRID_W - RIGHT_C0
LEGEND_R0 = 0
LEGEND_H = 2

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_DISTRACTORS_RANGE = (1, 2)
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 6)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _find_legend(grid: np.ndarray, bg: int) -> tuple[dict[int, int], set[tuple[int, int]]]:
    """
    Read the 2xN legend from the top edge.

    Legend format enforced by the generator:
    - rows 0 and 1
    - each legend column has non-bg cells in both rows
    - top cell = sprite color, bottom cell = frame color
    """
    if grid.shape[0] < LEGEND_H:
        return {}, set()

    mapping: dict[int, int] = {}
    legend_cells: set[tuple[int, int]] = set()
    for c in range(grid.shape[1]):
        top = int(grid[LEGEND_R0, c])
        bottom = int(grid[LEGEND_R0 + 1, c])
        if top == bg and bottom == bg:
            continue
        if top == bg or bottom == bg:
            continue
        mapping[top] = bottom
        legend_cells.add((LEGEND_R0, c))
        legend_cells.add((LEGEND_R0 + 1, c))
    return mapping, legend_cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count mapped-color sprites per frame and render the tally as interior dots.

    Variant rule:
    - The top 2 rows contain a legend. Each legend column maps sprite-color -> frame-color.
    - Remove all small sprites from the grid, but keep the legend intact.
    - For each frame of color F, count sprites whose color maps to F via the legend.
    - Draw that many dots inside the frame on the center row, spaced by 2 from the right.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))
    sprite_to_frame, legend_cells = _find_legend(g, bg)

    regions = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")

    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in regions:
        coords = [(int(r), int(c)) for r, c in comp["coords"]]
        if coords and all(cell in legend_cells for cell in coords):
            continue

        b = comp["bbox"]
        h = bbox_h(b)
        w = bbox_w(b)
        area = int(comp["area"])
        is_frame = (h >= 5) and (w >= 5) and (area < (h * w * 0.7))
        if is_frame:
            frames.append(comp)
        else:
            patterns.append(comp)

    for p in patterns:
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for f in frames:
        frame_color = int(f["color"])
        num_patterns = sum(1 for p in patterns if sprite_to_frame.get(int(p["color"])) == frame_color)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, f["bbox"])
        r_mid = (r0 + r1) // 2
        for i in range(num_patterns):
            cc = int(c1 - 2 - 2 * i)
            if cc > int(c0):
                out[int(r_mid), int(cc)] = frame_color

    return out


def _frame_patch(color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    mid_r = FRAME_H // 2
    dot_cols = [FRAME_W - 3 - 2 * i for i in range(n_dots)]
    patch[mid_r, dot_cols] = color
    return patch


def _sample_counts(rng: np.random.Generator, n_frames: int, total_match: int) -> list[int]:
    counts = [0] * n_frames
    order = rng.permutation(n_frames)
    remaining = total_match
    for pos, idx in enumerate(order):
        frames_left = n_frames - pos - 1
        lo = max(0, remaining - MAX_DOTS_PER_FRAME * frames_left)
        hi = min(MAX_DOTS_PER_FRAME, remaining)
        val = int(rng.integers(lo, hi + 1))
        counts[int(idx)] = val
        remaining -= val
    return counts


def _grow_sprite(rng: np.random.Generator, h: int, w: int, area: int) -> list[tuple[int, int]]:
    seed = (int(rng.integers(0, h)), int(rng.integers(0, w)))
    cells = {seed}
    while len(cells) < area:
        frontier = []
        for r, c in cells:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nb = (r + dr, c + dc)
                if 0 <= nb[0] < h and 0 <= nb[1] < w and nb not in cells:
                    frontier.append(nb)
        cells.add(frontier[int(rng.integers(0, len(frontier)))])
    return list(cells)


def _make_sprite(rng: np.random.Generator, color: int, bg: int) -> np.ndarray:
    area = int(rng.integers(*SPRITE_AREA_RANGE))
    dims = [
        (h, w)
        for h in range(1, SPRITE_MAX_H + 1)
        for w in range(1, SPRITE_MAX_W + 1)
        if h * w >= area
    ]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[r - r_min, c - c_min] = color
    return patch


def _legend_patch(sprite_colors: list[int], frame_colors: list[int], bg: int) -> np.ndarray:
    patch = np.full((LEGEND_H, len(sprite_colors)), bg, dtype=int)
    patch[0, :] = np.array(sprite_colors, dtype=int)
    patch[1, :] = np.array(frame_colors, dtype=int)
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    n_needed_colors = 1 + n_frames + n_frames + n_distractors
    palette = rng.choice(np.arange(10), size=n_needed_colors, replace=False)
    bg = int(palette[0])
    sprite_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    frame_colors = [int(x) for x in palette[1 + n_frames : 1 + 2 * n_frames]]
    distractor_colors = [int(x) for x in palette[1 + 2 * n_frames :]]

    counts = _sample_counts(rng, n_frames, total_match)
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    perm = rng.permutation(n_frames)
    legend_sprite_colors = [sprite_colors[int(i)] for i in range(n_frames)]
    legend_frame_colors = [frame_colors[int(i)] for i in perm]
    sprite_for_frame = {legend_frame_colors[i]: legend_sprite_colors[i] for i in range(n_frames)}

    out = np.full((height, GRID_W), bg, dtype=int)
    paste(out, _legend_patch(legend_sprite_colors, legend_frame_colors, bg), LEGEND_R0, RIGHT_C0)

    frame_rows: list[int] = []
    for i, (color, count) in enumerate(zip(frame_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        dot_cols = [FRAME_C0 + FRAME_W - 3 - 2 * i for i in range(count)]
        inp[mid_r, dot_cols] = bg

    sprite_canvas = np.full((height - LEGEND_H, RIGHT_W), bg, dtype=int)
    pattern_colors = [sprite_for_frame[frame_color] for frame_color, n in zip(frame_colors, counts) for _ in range(n)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, color, bg)
            r0, c0 = random_free_location(
                sprite_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(sprite_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, sprite_canvas, LEGEND_H, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
