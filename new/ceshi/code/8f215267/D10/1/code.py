# Variant of 8f215267
# Dimension: D10 — 颜色使用约束 (meta)
# Change: 每种颜色最多用于一个框架，干扰精灵使用不与任何框架重复的颜色；总颜色数 = 1(背景) + 3-4(框架) + 1-2(... → 框架使用双色边框

from __future__ import annotations

import sys

import numpy as np

try:
    from arc_utils import bbox_h, bbox_w, find_components, get_bg
    from arc_utils import paste, random_free_location
except ModuleNotFoundError:
    sys.path.append("/home/lixinyu/snap/new/solvers")
    from arc_utils import bbox_h, bbox_w, find_components, get_bg
    from arc_utils import paste, random_free_location


# Design analysis for the D10 variant
# ==================================
# a) Concrete appearance and behavior
# - Original task: each frame is a single-color 5x11 hollow rectangle, and scattered
#   sprites match the frame by exact color equality.
# - Variant: each frame remains a 5x11 hollow rectangle in the same left-side stack,
#   but its border uses two colors:
#   - top and bottom edges use the frame's main color,
#   - left and right edges (excluding the corners) use an accent color.
# - Because the frame is much wider than tall, the main color occupies 22 border
#   cells while the accent color occupies 6 border cells. The output dots are drawn
#   in the main color only. Sprites outside the frames also use only the main color.
# - Small trace example:
#   - Original input: a red frame plus three red sprites -> output adds three red dots.
#   - Variant input: a frame whose horizontal edges are red and vertical edges are
#     blue, plus three red sprites -> output adds three red dots, while the blue side
#     color remains only a visual cue on the frame.
#   - Difference: the solver can no longer treat every monochrome component as a full
#     frame. It must identify a multicolor rectangle, recover its dominant border
#     color, then count sprites by that dominant color.
#
# b) Complexity calibration
# - The frame shape, frame count, sprite size, and placement strategy stay the same,
#   so the task is still recognizably 8f215267.
# - The new difficulty is local and visible: every frame now contains one functional
#   color and one non-functional color. Accent colors are reused across frames, which
#   prevents the scene from looking like "every border color matters equally".
# - Distractor sprites never use any frame color, including accent colors, to avoid
#   ambiguous evidence about whether the accent color should be counted.
#
# c) Hidden-assumption review against the original main()
# - `find_components(... monochromatic=True)`: invalid now, because one frame would be
#   split into four monochrome pieces. Replaced with multicolor connected components.
# - `comp["color"]` as the frame color: invalid now, because a frame has two colors.
#   Replaced with border-frequency analysis over the frame bbox.
# - `num_patterns = sum(... p["color"] == col)`: still valid in spirit, but `col`
#   must now mean the frame's main color, not the raw component color.
# - Clearing patterns by component coords is still valid because sprites remain small,
#   isolated, and monochrome.
#
# d) Placement legality
# - Frames remain in fixed left-side slots with two empty rows between them, so they
#   never touch each other.
# - Sprites are still generated only in the right-side canvas and placed with 8-neighbor
#   padding, so sprite components do not merge with each other.
# - The solver recognizes a frame only if its bbox has a complete border, exactly two
#   border colors, uniform top/bottom color, uniform left/right color, and no interior
#   foreground outside the allowed dot positions.
#
# e) Visual observability
# - In 2-4 training pairs, the user can see that dots and matching sprites always use
#   the more prevalent frame color, while the thinner side color never participates.
# - No rule depends on numeric color ids, only on visible area dominance and border
#   location.
#
# f) Visual cleanliness
# - Frames stay in the left block and sprites stay in the right block, preserving the
#   original spatial organization.
# - Accent colors appear only on side edges, which is visually clean and easy to parse.
#
# g) Color inference
# - Total colors are kept moderate: 1 background + 3-4 main colors + 1-2 accent colors
#   + 1-2 distractor colors.
# - Accent colors are reused and never appear as sprites, giving the solver enough
#   structural evidence that only the dominant border color controls counting.
#
# h) New solver assumptions introduced here
# - Frames are exactly the generator's border template: top/bottom one color, side
#   edges another, corners belonging to the top/bottom color.
# - Any interior marks that belong to a frame occur only on the center row in the
#   original tally positions. The generator guarantees this.
# - Non-frame objects are small enough that they cannot satisfy the full-border test.


# --- layout geometry ---
FRAME_H = 5
FRAME_W = 11
FRAME_R0 = 1
FRAME_C0 = 1
FRAME_GAP = 2
GRID_W = 23
RIGHT_C0 = 13
RIGHT_W = 9

# --- structural counts ---
N_FRAMES_RANGE = (3, 5)
N_ACCENT_COLORS_RANGE = (1, 3)
N_DISTRACTORS_RANGE = (1, 3)
TOTAL_PATTERNS_RANGE = (5, 11)
MAX_DOTS_PER_FRAME = 4
BOTTOM_PAD_RANGE = (0, 7)

# --- sprite shape ---
SPRITE_AREA_RANGE = (2, 9)
SPRITE_MAX_H = 4
SPRITE_MAX_W = 3
SPRITE_PADDING = 1

# --- retry budget ---
MAX_ATTEMPTS = 300


def _dot_columns(width: int) -> list[int]:
    return [int(width - 3 - 2 * i) for i in range(MAX_DOTS_PER_FRAME) if int(width - 3 - 2 * i) > 0]


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _allowed_dot_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mid_r = h // 2
    for cc in _dot_columns(w):
        mask[mid_r, int(cc)] = True
    return mask


def _component_color(grid: np.ndarray, comp: dict) -> int:
    coords = comp["coords"]
    vals = np.unique(grid[coords[:, 0], coords[:, 1]])
    if len(vals) != 1:
        raise ValueError("non-frame component is not monochrome")
    return int(vals[0])


def _frame_info(grid: np.ndarray, bbox: tuple[int, int, int, int], bg: int) -> dict | None:
    r0, c0, r1, c1 = map(int, bbox)
    h = bbox_h(bbox)
    w = bbox_w(bbox)
    if h < FRAME_H or w < FRAME_W:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    fg = patch != bg
    border = _border_mask(h, w)
    allowed_dots = _allowed_dot_mask(h, w)

    if int(np.count_nonzero(fg & border)) != int(border.sum()):
        return None
    if bool((fg & ~(border | allowed_dots)).any()):
        return None

    top_vals = set(map(int, patch[0, :]))
    bottom_vals = set(map(int, patch[-1, :]))
    left_vals = set(map(int, patch[1:-1, 0]))
    right_vals = set(map(int, patch[1:-1, -1]))
    if len(top_vals) != 1 or len(bottom_vals) != 1 or top_vals != bottom_vals:
        return None
    if len(left_vals) != 1 or len(right_vals) != 1 or left_vals != right_vals:
        return None

    main_candidate = int(next(iter(top_vals)))
    accent_candidate = int(next(iter(left_vals)))
    if main_candidate == bg or accent_candidate == bg or main_candidate == accent_candidate:
        return None

    border_vals = patch[border]
    colors, counts = np.unique(border_vals, return_counts=True)
    if len(colors) != 2:
        return None
    order = np.argsort(counts)[::-1]
    main_color = int(colors[int(order[0])])
    accent_color = int(colors[int(order[1])])
    if int(counts[int(order[0])]) <= int(counts[int(order[1])]):
        return None
    if main_color != main_candidate or accent_color != accent_candidate:
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "main_color": main_color,
        "accent_color": accent_color,
    }


def _split_objects(grid: np.ndarray, bg: int) -> tuple[list[dict], list[dict]]:
    objects = find_components(grid, bg=bg, connectivity=4, monochromatic=False, return_format="dict")
    frames: list[dict] = []
    patterns: list[dict] = []
    for comp in objects:
        info = _frame_info(grid, comp["bbox"], bg)
        if info is not None:
            frames.append(info)
        else:
            patterns.append(comp)
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1]))
    return frames, patterns


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count same-main-color sprites per two-color frame and show the tally as dots.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    frames, patterns = _split_objects(g, bg)

    pattern_colors = []
    for p in patterns:
        pattern_colors.append(_component_color(g, p))
        for r, c in p["coords"]:
            out[int(r), int(c)] = bg

    for frame in frames:
        main_color = int(frame["main_color"])
        num_patterns = sum(1 for color in pattern_colors if int(color) == main_color)
        if num_patterns <= 0:
            continue

        r0, c0, r1, c1 = map(int, frame["bbox"])
        r_mid = (r0 + r1) // 2
        for i, local_c in enumerate(_dot_columns(bbox_w(frame["bbox"]))):
            if i >= num_patterns:
                break
            out[int(r_mid), int(c0 + local_c)] = main_color

    return out


def _frame_patch(main_color: int, accent_color: int, bg: int, n_dots: int) -> np.ndarray:
    patch = np.full((FRAME_H, FRAME_W), bg, dtype=int)
    patch[0, :] = main_color
    patch[-1, :] = main_color
    patch[1:-1, 0] = accent_color
    patch[1:-1, -1] = accent_color
    mid_r = FRAME_H // 2
    for cc in _dot_columns(FRAME_W)[:n_dots]:
        patch[mid_r, int(cc)] = main_color
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
    dims = [(h, w) for h in range(1, SPRITE_MAX_H + 1) for w in range(1, SPRITE_MAX_W + 1) if h * w >= area]
    h, w = dims[int(rng.integers(0, len(dims)))]
    cells = _grow_sprite(rng, h, w, area)
    rs, cs = zip(*cells)
    r_min, c_min = min(rs), min(cs)
    patch = np.full((max(rs) - r_min + 1, max(cs) - c_min + 1), bg, dtype=int)
    for r, c in cells:
        patch[int(r - r_min), int(c - c_min)] = color
    return patch


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()

    n_frames = int(rng.integers(*N_FRAMES_RANGE))
    n_accent_colors = int(rng.integers(*N_ACCENT_COLORS_RANGE))
    n_distractors = int(rng.integers(*N_DISTRACTORS_RANGE))
    total_patterns = int(rng.integers(*TOTAL_PATTERNS_RANGE))
    total_match = total_patterns - n_distractors

    palette_size = 1 + n_frames + n_accent_colors + n_distractors
    palette = rng.choice(np.arange(10), size=palette_size, replace=False)
    bg = int(palette[0])
    main_colors = [int(x) for x in palette[1 : 1 + n_frames]]
    accent_palette = [int(x) for x in palette[1 + n_frames : 1 + n_frames + n_accent_colors]]
    distractor_colors = [int(x) for x in palette[1 + n_frames + n_accent_colors :]]
    accent_colors = [int(accent_palette[int(i)]) for i in rng.integers(0, len(accent_palette), size=n_frames)]

    counts = _sample_counts(rng, n_frames, total_match)
    bottom_pad = int(rng.integers(*BOTTOM_PAD_RANGE)) if n_frames == 3 else 0
    height = FRAME_R0 + n_frames * FRAME_H + (n_frames - 1) * FRAME_GAP + bottom_pad

    out = np.full((height, GRID_W), bg, dtype=int)
    frame_rows: list[int] = []
    for i, (main_color, accent_color, count) in enumerate(zip(main_colors, accent_colors, counts)):
        r0 = FRAME_R0 + i * (FRAME_H + FRAME_GAP)
        frame_rows.append(r0)
        paste(out, _frame_patch(main_color, accent_color, bg, count), r0, FRAME_C0)

    inp = out.copy()
    for r0, count in zip(frame_rows, counts):
        mid_r = r0 + FRAME_H // 2
        for cc in _dot_columns(FRAME_W)[:count]:
            inp[int(mid_r), int(FRAME_C0 + cc)] = bg

    right_canvas = np.full((height, RIGHT_W), bg, dtype=int)
    pattern_colors = [color for color, count in zip(main_colors, counts) for _ in range(count)]
    pattern_colors.extend(distractor_colors)
    rng.shuffle(pattern_colors)

    try:
        for color in pattern_colors:
            sprite = _make_sprite(rng, int(color), bg)
            r0, c0 = random_free_location(
                right_canvas,
                sprite,
                transparent=bg,
                padding=SPRITE_PADDING,
                padding_connectivity=8,
            )
            paste(right_canvas, sprite, r0, c0, transparent=bg)
    except ValueError:
        return None

    paste(inp, right_canvas, 0, RIGHT_C0, transparent=bg)
    if np.array_equal(main(inp), out):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
