# Combo variant of 8698868d (index 25)
# combo_spec: D2/2+D4/3+D16/0
# Name: 多色精灵+图例+第三色填洞

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask
from arc_utils import paste, random_free_location


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    - The original task had monochrome sprites, hole-count matching, and hole fill
#      equal to the destination frame color.
#    - This combo changes all three links in one chain:
#      1) each sprite keeps a solid border color but its interior foreground becomes
#         a 2-3 color pattern,
#      2) a top-edge legend maps sprite border color -> frame color, so pairing is
#         read from symbols rather than from hole counts,
#      3) the same legend entry also provides a third color for filling that sprite's
#         holes after nesting.
#    - Concrete trace:
#         input legend entry: blue -> red -> yellow
#         scattered sprite: blue border, blue/green checker interior, 2 bg holes
#         frame in grid: red border/frame, maybe 0 or 1 interior bg holes
#         output: the blue/green sprite is nested into the red frame and only its bg
#         hole cells become yellow. The blue/green non-hole pattern stays unchanged.
#      Compared with the original task, both the matching key and the recolor target
#      move from structural counting to legend decoding, while multicolor content
#      makes "keep non-hole colors as-is" a visible rule instead of a trivial one.
#
# b) Complexity calibration:
#    - The legend is a compact 3-row top band: row 0 = sprite border color,
#      row 1 = destination frame color, row 2 = hole-fill color. This is the
#      smallest clean encoding that exposes both changed mappings.
#    - Object count is limited to 2-3. That keeps the legend readable and allows
#      three disjoint color roles per entry (sprite border / frame / fill) within
#      the 9 non-background ARC colors.
#    - Every sprite is forced to use at least one accent color beyond its border
#      color, so the D2 change is always visible instead of silently collapsing back
#      to a monochrome sprite.
#
# c) Hidden assumptions broken from the original solver:
#    - Original assumption: one monochrome component equals one whole sprite.
#      Broken by D2. The new solver instead detects the sprite via its solid border
#      color component and then crops the full multicolor bbox from the grid.
#    - Original assumption: "sprite holes are everything not equal to sprite_color".
#      Broken by D2 because accent colors are not holes. The new solver treats only
#      background-colored cells as holes.
#    - Original assumption: pairing is by equal hole-component count.
#      Broken by D16/0. Hole counts remain present only as distractors; pairing now
#      comes from the legend's sprite-color -> frame-color relation.
#    - Original assumption: filled holes become the frame color.
#      Broken by D4/3. Fill now comes from the legend's third row.
#
# d) Placement legality:
#    - The legend is fixed to the top edge and frames start below reserved rows, so
#      legend/frame overlap cannot happen.
#    - Sprites are placed by `random_free_location(..., padding=1)` after the legend
#      and frame grid are drawn, so they cannot overlap either region or each other.
#    - Each legend entry uses a 1-column gap, preventing accidental merging.
#
# e) Visual observability:
#    - A human can see the same sprite-border color both in the legend top row and on
#      the scattered multicolor sprite border. The corresponding frame color appears
#      in the legend middle row and on the frame border. The fill color appears only
#      in the legend bottom row and later inside holes in the output.
#    - Hole counts are deliberately deranged between frames and sprites, so the old
#      structural matching rule visibly fails on generated samples.
#
# f) Visual cleanliness:
#    - The legend is confined to 3 rows at the top; the frame grid stays compact
#      below; sprites stay in the remaining background. This keeps metadata, targets,
#      and movable pieces separated into readable zones.
#
# g) Color-change inferability:
#    - Each legend entry provides repeated spatial evidence for a triple relation:
#      sprite border color -> frame color -> fill color.
#    - Fill colors are distinct from the matched sprite palette and from the matched
#      frame color, so the "third-color fill" is always visually explicit.
#
# 9) New solver assumptions and generator guarantees:
#    - Assumption: every real sprite/frame has a solid 1-pixel border of a single
#      color. Guarantee: generator constructs both objects that way.
#    - Assumption: every sprite border color appears exactly once in the legend and
#      exactly once as a sprite. Guarantee: generator uses distinct border colors.
#    - Assumption: the top 3 rows contain only the legend band. Guarantee: generator
#      reserves those rows and never places sprites/frames there.
#    - Assumption: sprite border colors differ from frame colors for the same entry,
#      and fill colors differ from both. Guarantee: generator samples disjoint role
#      colors per entry.


BG = 8

FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 3)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
HOLE_START_MAX = 1
SEED_RETRIES = 60
GEN_RETRIES = 400

LEGEND_H = 3
LEGEND_CELL_W = 2
LEGEND_GAP = 1
LEGEND_RESERVED_ROWS = 4


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _legend_width(count: int) -> int:
    return int(count * LEGEND_CELL_W + max(0, count - 1) * LEGEND_GAP)


def _read_legend_map(grid: np.ndarray, bg: int) -> dict[int, tuple[int, int]]:
    """Read the top-edge 3-row legend: sprite_border -> (frame_color, fill_color)."""
    h, w = grid.shape
    assert h >= LEGEND_H

    band = (grid[:LEGEND_H, :] != int(bg)).any(axis=0)
    assert bool(band.any())

    mapping: dict[int, tuple[int, int]] = {}
    c = 0
    while c < w:
        if not bool(band[int(c)]):
            c += 1
            continue
        c0 = int(c)
        while c + 1 < w and bool(band[int(c + 1)]):
            c += 1
        c1 = int(c)

        sub = grid[:LEGEND_H, c0 : c1 + 1]
        assert sub.shape == (LEGEND_H, LEGEND_CELL_W)
        assert bool((sub != int(bg)).all())
        top_vals = {int(v) for v in sub[0, :].tolist()}
        mid_vals = {int(v) for v in sub[1, :].tolist()}
        bot_vals = {int(v) for v in sub[2, :].tolist()}
        assert len(top_vals) == len(mid_vals) == len(bot_vals) == 1

        sprite_border = next(iter(top_vals))
        frame_color = next(iter(mid_vals))
        fill_color = next(iter(bot_vals))
        assert sprite_border not in mapping
        mapping[int(sprite_border)] = (int(frame_color), int(fill_color))
        c += 1

    assert mapping
    return mapping


def _find_square_candidates(grid: np.ndarray) -> list[dict]:
    """Detect square frames and multicolor sprites via their solid monochrome borders."""
    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True, return_format="dict")
    candidates: list[dict] = []
    for comp in comps:
        b = comp["bbox"]
        size = int(bbox_h(b))
        if size != int(bbox_w(b)) or not (4 <= size <= 10):
            continue
        if int(comp["area"]) < int(4 * size - 4):
            continue

        region = crop_bbox(grid, b)
        border_color = int(comp["color"])
        border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
        if not bool((border == int(border_color)).all()):
            continue

        candidates.append(
            {
                "bbox": b,
                "size": int(size),
                "border_color": int(border_color),
            }
        )
    return candidates


def _extract_objects(grid: np.ndarray) -> tuple[int, int, list[dict], dict[int, dict]]:
    candidates = _find_square_candidates(grid)
    assert candidates

    sizes: dict[int, list[dict]] = {}
    for cand in candidates:
        sizes.setdefault(int(cand["size"]), []).append(cand)

    size_list = sorted(sizes.keys(), reverse=True)
    assert len(size_list) >= 2
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1])

    frame_candidates = sizes[int(frame_size)]
    sprite_candidates = sizes[int(sprite_size)]
    assert frame_candidates and sprite_candidates

    frame_positions = [(int(f["bbox"][0]), int(f["bbox"][1])) for f in frame_candidates]
    min_r = min(r for r, _c in frame_positions)
    min_c = min(c for _r, c in frame_positions)
    max_r = max(r for r, _c in frame_positions)
    max_c = max(c for _r, c in frame_positions)

    n_rows = int((max_r - min_r) // frame_size + 1)
    n_cols = int((max_c - min_c) // frame_size + 1)

    frame_grid: list[dict] = []
    for ri in range(n_rows):
        for ci in range(n_cols):
            inp_r = int(min_r + ri * frame_size)
            inp_c = int(min_c + ci * frame_size)
            region = grid[inp_r : inp_r + frame_size, inp_c : inp_c + frame_size]
            assert region.shape == (frame_size, frame_size)

            border = np.concatenate([region[0, :], region[-1, :], region[:, 0], region[:, -1]])
            frame_color = int(Counter(map(int, border.flatten())).most_common(1)[0][0])
            frame_grid.append(
                {
                    "color": int(frame_color),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                }
            )

    sprites_by_border: dict[int, dict] = {}
    for cand in sprite_candidates:
        region = crop_bbox(grid, cand["bbox"])
        border_color = int(cand["border_color"])
        sprites_by_border[int(border_color)] = {
            "border_color": int(border_color),
            "content": region.copy(),
            "h": int(region.shape[0]),
            "w": int(region.shape[1]),
        }

    return frame_size, sprite_size, frame_grid, sprites_by_border


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a color legend, place each multicolor sprite into its designated frame,
    and recolor only the sprite's holes with a legend-defined third color.

    Input structure:
    - bg = azure (8).
    - The top edge contains a 3-row legend band; each 2-column entry encodes
      sprite-border color on row 0, frame color on row 1, and hole-fill color on
      row 2.
    - Large square frames of frame colors form a tight grid below the legend.
    - Smaller square sprites are scattered elsewhere; each sprite has a solid border
      in one color but keeps a 2-3 color foreground pattern inside, plus bg holes.

    Rule:
    - Decode the legend to obtain sprite-border -> frame and sprite-border -> fill.
    - Detect frames by the large square grid and sprites by their smaller square
      border-colored shells.
    - Nest each sprite inside the frame whose color is specified by the legend.
    - Preserve all non-hole sprite colors, but recolor the sprite's bg hole cells to
      the legend's third color.

    Key insight:
    - The legend replaces hole-count matching, and multicolor sprites mean "non-border
      colors are real content, not holes". Only background cells inside a sprite are
      transformed, while the visible color pattern is carried unchanged into the frame.
    """
    g = np.array(grid, dtype=int)

    legend_map = _read_legend_map(g, BG)
    frame_to_sprite = {int(frame): int(sprite) for sprite, (frame, _fill) in legend_map.items()}
    frame_to_fill = {int(frame): int(fill) for sprite, (frame, fill) in legend_map.items()}
    assert len(frame_to_sprite) == len(legend_map) == len(frame_to_fill)

    frame_size, _sprite_size, frame_grid, sprites_by_border = _extract_objects(g)

    out_h = int(max(f["out_r"] for f in frame_grid) + frame_size)
    out_w = int(max(f["out_c"] for f in frame_grid) + frame_size)
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for frame in frame_grid:
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = int(frame_color)

        sprite_border = frame_to_sprite.get(int(frame_color))
        fill_color = frame_to_fill.get(int(frame_color))
        assert sprite_border is not None and fill_color is not None
        sprite = sprites_by_border[int(sprite_border)]

        content = np.array(sprite["content"], dtype=int).copy()
        content[content == int(BG)] = int(fill_color)
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


def _random_derangement(values: list[int]) -> list[int] | None:
    if len(values) <= 1:
        return None
    if len(values) == 2:
        return [int(values[1]), int(values[0])]
    for _ in range(120):
        perm = values[:]
        random.shuffle(perm)
        if all(int(a) != int(b) for a, b in zip(values, perm)):
            return [int(v) for v in perm]
    return None


def _place_isolated_seeds(size: int, count: int) -> list[tuple[int, int]] | None:
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


def _make_holey_base_sprite(size: int, hole_count: int, border_color: int) -> np.ndarray | None:
    """Build a square sprite shell with exactly `hole_count` 4-connected bg holes."""
    sprite = np.full((size, size), int(border_color), dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_holes = set(seeds)
    for r, c in seeds:
        sprite[r, c] = int(BG)

    max_grow = max(1, inner_cells // max(hole_count, 1) - 1)
    for ci, comp in enumerate(comps):
        grow_steps = random.randint(0, min(max_grow, inner_cells - len(all_holes)))
        for _ in range(grow_steps):
            frontier: list[tuple[int, int]] = []
            for r, c in comp:
                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = int(r + dr), int(c + dc)
                    if not (1 <= nr <= size - 2 and 1 <= nc <= size - 2):
                        continue
                    if (nr, nc) in all_holes:
                        continue
                    if any(
                        (nr + dr2, nc + dc2) in other
                        for j, other in enumerate(comps)
                        if j != ci
                        for dr2, dc2 in [(-1, 0), (1, 0), (0, -1), (0, 1), (0, 0)]
                    ):
                        continue
                    frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_holes.add(cell)
            sprite[cell[0], cell[1]] = int(BG)

    _, num = label_mask(sprite == int(BG), connectivity=4)
    if int(num) != int(hole_count):
        return None
    return sprite


def _paint_multicolor_pattern(
    sprite: np.ndarray,
    border_color: int,
    accent_pool: list[int],
) -> np.ndarray | None:
    """Recolor interior foreground cells with 1-2 accent colors while keeping the border."""
    out = np.array(sprite, dtype=int).copy()
    size = int(out.shape[0])
    inner_fg = [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if int(out[r, c]) != int(BG)]
    if not inner_fg:
        return None

    max_accents = 2 if len(inner_fg) >= 3 and len(accent_pool) >= 2 else 1
    n_accents = 2 if max_accents == 2 and random.random() < 0.5 else 1
    accent_colors = [int(v) for v in random.sample(accent_pool, n_accents)]

    if n_accents == 1:
        labelers = [
            lambda r, c: (r + c) % 2,
            lambda r, c: c % 2,
            lambda r, c: r % 2,
        ]
    else:
        labelers = [
            lambda r, c: (r + 2 * c) % 3,
            lambda r, c: (2 * r + c) % 3,
            lambda r, c: (r + c) % 3,
        ]

    chosen = None
    for labeler in labelers:
        seen = {int(labeler(r, c)) for r, c in inner_fg}
        needed = set(range(1, n_accents + 1))
        if needed.issubset(seen):
            chosen = labeler
            break

    if chosen is None:
        for idx, (r, c) in enumerate(inner_fg):
            if idx < n_accents:
                out[r, c] = int(accent_colors[idx])
        if any(int(v) != int(border_color) for v in out.flatten() if int(v) != int(BG)):
            return out
        return None

    for r, c in inner_fg:
        label = int(chosen(r, c))
        if 1 <= label <= n_accents:
            out[r, c] = int(accent_colors[label - 1])

    non_bg_colors = {int(v) for v in out.flatten() if int(v) != int(BG)}
    if len(non_bg_colors) < 2:
        return None
    return out


def _make_multicolor_sprite(
    size: int,
    hole_count: int,
    border_color: int,
    accent_pool: list[int],
) -> np.ndarray | None:
    base = _make_holey_base_sprite(size, hole_count, border_color)
    if base is None:
        return None
    return _paint_multicolor_pattern(base, border_color, accent_pool)


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    frame = np.full((frame_size, frame_size), int(frame_color), dtype=int)
    if hole_count == 0:
        return frame

    inner_cells = (frame_size - 2) ** 2
    if inner_cells < hole_count:
        return None

    seeds = _place_isolated_seeds(frame_size, hole_count)
    if seeds is None:
        return None

    for r, c in seeds:
        frame[r, c] = int(BG)

    _, num = label_mask(frame == int(BG), connectivity=4)
    if int(num) != int(hole_count):
        return None
    return frame


def _make_legend(sprite_borders: list[int], frame_colors: list[int], fill_colors: list[int]) -> np.ndarray:
    width = _legend_width(len(sprite_borders))
    legend = np.full((LEGEND_H, width), int(BG), dtype=int)
    for i, (sprite_border, frame_color, fill_color) in enumerate(zip(sprite_borders, frame_colors, fill_colors)):
        c0 = int(i * (LEGEND_CELL_W + LEGEND_GAP))
        legend[0, c0 : c0 + LEGEND_CELL_W] = int(sprite_border)
        legend[1, c0 : c0 + LEGEND_CELL_W] = int(frame_color)
        legend[2, c0 : c0 + LEGEND_CELL_W] = int(fill_color)
    return legend


def _build_expected_output(
    frames: list[tuple[int, int, np.ndarray]],
    sprites: list[tuple[int, int, int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), int(BG), dtype=int)
    sprite_by_frame: dict[int, tuple[int, np.ndarray]] = {}
    for sprite_border, frame_color, fill_color, sprite_patch in sprites:
        sprite_by_frame[int(frame_color)] = (int(fill_color), np.array(sprite_patch, dtype=int).copy())

    for idx, (frame_color, _hole_count, _patch) in enumerate(frames):
        rr = int((idx // cols_grid) * frame_size)
        cc = int((idx % cols_grid) * frame_size)
        expected[rr : rr + frame_size, cc : cc + frame_size] = int(frame_color)
        fill_color, sprite_patch = sprite_by_frame[int(frame_color)]
        content = np.array(sprite_patch, dtype=int).copy()
        content[content == int(BG)] = int(fill_color)
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def _gen_attempt() -> np.ndarray | None:
    n = random.choice(_VALID_NS)
    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    legend_w = _legend_width(n)

    feasible_layouts = []
    for rows_grid, cols_grid in _LAYOUTS_BY_N[n]:
        grid_pixel_h = int(rows_grid * frame_size)
        grid_pixel_w = int(cols_grid * frame_size)
        min_h = max(CANVAS_MIN, LEGEND_RESERVED_ROWS + grid_pixel_h + sprite_size + 1)
        min_w = max(CANVAS_MIN, max(grid_pixel_w, legend_w) + sprite_size + 1)
        if min_h <= CANVAS_MAX and min_w <= CANVAS_MAX:
            feasible_layouts.append((int(rows_grid), int(cols_grid), int(min_h), int(min_w)))
    if not feasible_layouts:
        return None

    rows_grid, cols_grid, min_h, min_w = random.choice(feasible_layouts)
    h = random.randint(min_h, CANVAS_MAX)
    w = random.randint(min_w, CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    picked = random.sample(available, 3 * n)
    sprite_borders = [int(c) for c in picked[:n]]
    frame_colors = [int(c) for c in picked[n : 2 * n]]
    fill_colors = [int(c) for c in picked[2 * n : 3 * n]]

    sprite_inner = sprite_size - 2
    frame_inner = frame_size - 2
    max_sprite_holes = (sprite_inner * sprite_inner + 1) // 2
    max_frame_holes = (frame_inner * frame_inner + 1) // 2
    max_hole = min(int(max_sprite_holes), int(max_frame_holes))
    # We need n distinct positive hole counts to keep the third-color fill
    # visibly active on every sprite while still breaking the old hole-count
    # pairing rule via a derangement.
    if max_hole < n:
        return None

    max_start = max(0, max_hole - n + 1)
    # Keep at least one hole per sprite whenever the geometry allows it, so the
    # third-color fill behavior is visible on every paired object.
    min_start = 1 if max_hole >= n else 0
    start_hi = min(max_start, max(HOLE_START_MAX, min_start))
    start = random.randint(min_start, start_hi)
    frame_holes = list(range(start, start + n))
    sprite_holes = _random_derangement(frame_holes)
    if sprite_holes is None:
        return None

    all_role_colors = {int(c) for c in sprite_borders + frame_colors + fill_colors}
    sprites: list[tuple[int, int, int, np.ndarray]] = []
    for i in range(n):
        anchor = int(sprite_borders[i])
        frame_color = int(frame_colors[i])
        fill_color = int(fill_colors[i])
        preferred_accents = [c for c in available if c not in all_role_colors]
        fallback_accents = [c for c in available if c not in {anchor, frame_color, fill_color}]
        accent_pool = preferred_accents if preferred_accents else fallback_accents
        if not accent_pool:
            return None
        patch = _make_multicolor_sprite(sprite_size, int(sprite_holes[i]), anchor, accent_pool)
        if patch is None:
            return None
        if len({int(v) for v in patch.flatten() if int(v) != int(BG)}) < 2:
            return None
        sprites.append((anchor, frame_color, fill_color, patch))

    frames: list[tuple[int, int, np.ndarray]] = []
    for i in range(n):
        patch = _make_frame(frame_size, int(frame_holes[i]), int(frame_colors[i]))
        if patch is None:
            return None
        frames.append((int(frame_colors[i]), int(frame_holes[i]), patch))

    canvas = np.full((h, w), int(BG), dtype=int)

    legend = _make_legend(sprite_borders, frame_colors, fill_colors)
    legend_c = random.randint(0, w - legend_w)
    paste(canvas, legend, 0, legend_c, transparent=BG)

    grid_pixel_h = int(rows_grid * frame_size)
    grid_pixel_w = int(cols_grid * frame_size)
    r0 = random.randint(LEGEND_RESERVED_ROWS, h - grid_pixel_h)
    c0 = random.randint(0, w - grid_pixel_w)
    for idx, (_frame_color, _hole_count, patch) in enumerate(frames):
        rr = int(r0 + (idx // cols_grid) * frame_size)
        cc = int(c0 + (idx % cols_grid) * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = patch

    for anchor, _frame_color, _fill_color, patch in sprites:
        footprint = np.full(patch.shape, int(anchor), dtype=int)
        try:
            sr, sc = random_free_location(
                canvas,
                footprint,
                transparent=BG,
                padding=SPRITE_PLACEMENT_PAD,
            )
        except ValueError:
            return None
        paste(canvas, patch, sr, sc, transparent=BG)

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)

    try:
        out = main(canvas)
    except Exception:
        return None

    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
