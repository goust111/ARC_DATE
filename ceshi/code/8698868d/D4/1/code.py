# Variant of 8698868d
# Dimension: D4 — 变换方式：嵌套后的重着色 (transform)
# Change: recolor — 精灵嵌套入框架后，精灵内部的洞（背景色像素）重着色为框架的边框颜色... → 洞重着色为精灵自身颜色（自填充）

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the D4 variant
# ----------------------------------
# a) Concreteized behavior and trace
#    Input geometry stays the same as the original family:
#    - large square frames in a tight grid
#    - smaller square sprites elsewhere on the canvas
#    - both are matched by the number of 4-connected interior hole components
#    Example:
#    - frame color = 4, sprite color = 2, matched hole count = 2
#    - original output placed the sprite in the frame, then recolored every sprite
#      hole pixel to 4, so the interior stayed visibly two-colored (2 + 4)
#    - this variant instead recolors every non-sprite pixel inside the nested sprite
#      to 2, so the whole inserted square becomes solid 2
#    The changed stage is only the final nesting/recolor step; frame detection, sprite
#    detection, hole counting, and matching all remain unchanged.
#
# b) Complexity calibration
#    The matching signal remains the connected-hole count, so the puzzle does not
#    collapse into a trivial color-copy task. To make the D4 change reliably visible,
#    the generator enforces:
#    - every matched sprite/frame pair uses different colors
#    - every sprite has at least one hole component
#    Otherwise some samples could accidentally look identical to the original task.
#
# c) Hidden-assumption review of the original solver
#    - component discovery by monochromatic connected regions still holds because
#      the input sprites remain monochrome objects with BG holes only
#    - using the largest square size as frame size still holds because frames remain
#      larger than sprites by exactly 2 cells per side
#    - counting holes via `region != object_color` still holds for both frames and
#      sprites because the input encoding is unchanged
#    - the only invalid old assumption is the final assignment
#      `content[content != sprite_color] = frame_color`; in this variant those pixels
#      must become `sprite_color`
#
# d) Placement legality
#    The generator keeps the original output-first structure:
#    - choose frame layout and colors
#    - build frames and sprites with legal isolated/grown hole components
#    - place the frame grid as a compact block
#    - place sprites only in remaining BG space using `random_free_location`
#    This guarantees no overlap, no out-of-bounds placement, and solver-readable input.
#
# e) Visual observability
#    Human-solvable samples need an obvious before/after cue. The output therefore has
#    a frame-colored border but a fully solid sprite-colored square inside each frame.
#    This makes the new rule visible in 2-3 examples without relying on hidden values.
#
# f) Visual cleanliness
#    The output contains only two regions per tile: the outer frame border and the
#    solid inner sprite square. There is no extra overlay or ambiguous marker.
#
# g) Color-change inferability
#    The color change is structurally anchored: the inserted object's own color fills
#    its former holes. The generator keeps sprite/frame colors distinct per match so
#    the self-fill rule is actually observable.


BG = 8


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by connected-hole count, nest each sprite into its frame,
    and self-fill sprite holes with the sprite's own color.
    """
    g = np.array(grid, dtype=int)

    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    # Find square regions; sprites may have sizeable hole area, so the fill ratio stays loose.
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

    frame_grid: list[dict] = []
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
                {
                    "color": frame_color,
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(num_holes),
                }
            )

    sprites: list[dict] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _lab, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
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

            # D4 change: former hole pixels become the sprite's own color, yielding a
            # completely solid sprite-colored square inside the frame.
            content[content != sprite_color] = sprite_color

            sh, sw = int(sprite["h"]), int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


# --------------- generator ---------------

FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
HOLE_START = 1
SEED_RETRIES = 50
COLOR_RETRIES = 100
GEN_RETRIES = 300


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
    """Place `count` interior cells that are pairwise 4-separated."""
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


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    """Create a sprite with a solid border and exactly `hole_count` 4-connected BG holes."""
    sprite = np.full((size, size), sprite_color, dtype=int)
    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{seed} for seed in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

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
            sprite[cell[0], cell[1]] = BG

    _, num = label_mask(sprite != sprite_color, connectivity=4)
    if num != hole_count:
        return None
    return sprite


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Create an input frame with exactly `hole_count` isolated BG holes."""
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = BG

    _, num = label_mask(frame != frame_color, connectivity=4)
    if num != hole_count:
        return None
    return frame


def _assign_colors(hole_counts: list[int]) -> tuple[dict[int, int], dict[int, int]] | None:
    """Assign distinct frame colors and sprite colors; matched pairs must differ."""
    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, len(hole_counts))
    frame_color_by_k = {k: frame_colors[i] for i, k in enumerate(hole_counts)}

    for _ in range(COLOR_RETRIES):
        sprite_colors = random.sample(available, len(hole_counts))
        sprite_color_by_k = {k: sprite_colors[i] for i, k in enumerate(hole_counts)}
        if all(sprite_color_by_k[k] != frame_color_by_k[k] for k in hole_counts):
            return frame_color_by_k, sprite_color_by_k
    return None


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_color_by_k: dict[int, int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    """Variant output: nested sprite holes are self-filled with sprite color."""
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {k: patch for k, patch in sprites}
    for idx, (k, _) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * frame_size, ci * frame_size
        frame_color = int(frame_color_by_k[k])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        content = sprite_by_k[k].copy()
        sprite_color = int(np.bincount(content[content != BG].ravel()).argmax())
        content[content == BG] = sprite_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _build_legacy_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    frame_color_by_k: dict[int, int],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    """Original-task output, used only to reject degenerate samples."""
    legacy = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {k: patch for k, patch in sprites}
    for idx, (k, _) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr, cc = ri * frame_size, ci * frame_size
        frame_color = int(frame_color_by_k[k])
        legacy[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        content = sprite_by_k[k].copy()
        content[content == BG] = frame_color
        legacy[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return legacy


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    rows_grid, cols_grid = random.choice(_LAYOUTS_BY_N[n])
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2

    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size
    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size + SPRITE_PADDING)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size + SPRITE_PADDING)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    canvas_h = random.randint(min_h, CANVAS_MAX)
    canvas_w = random.randint(min_w, CANVAS_MAX)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    if HOLE_START + n - 1 > max_hole:
        return None
    hole_counts = list(range(HOLE_START, HOLE_START + n))
    random.shuffle(hole_counts)

    color_assignment = _assign_colors(hole_counts)
    if color_assignment is None:
        return None
    frame_color_by_k, sprite_color_by_k = color_assignment

    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for k in sprite_hole_perm:
        patch = _make_sprite_interior(sprite_size, k, sprite_color_by_k[k])
        if patch is None:
            return None
        sprites.append((k, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for k in hole_counts:
        patch = _make_frame(frame_size, k, frame_color_by_k[k])
        if patch is None:
            return None
        frames.append((k, patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_, patch) in enumerate(frames):
        ri, ci = divmod(idx, cols_grid)
        rr = r0 + ri * frame_size
        cc = c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = patch

    for _, patch in sprites:
        try:
            r, c = random_free_location(canvas, patch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, patch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        frame_color_by_k,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    legacy = _build_legacy_output(
        frames,
        sprites,
        frame_color_by_k,
        rows_grid,
        cols_grid,
        frame_size,
        sprite_size,
    )
    if np.array_equal(expected, legacy):
        return None

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
