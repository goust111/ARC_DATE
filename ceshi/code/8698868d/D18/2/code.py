# Variant of 8698868d
# Dimension: D18 — 框架边框厚度 (meta)
# Change: 框架边框为 1 像素厚度，内部空间为 (frame_size - 2)^2... → 边框带有缺口

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste, random_free_location


# Design analysis for the D18 variant
# ==================================
# a) Concrete behavior / trace
#    - Original large objects are monochrome squares whose outer ring is a perfect 1-pixel border.
#      The solver counts hole components by inspecting only region[1:-1, 1:-1].
#    - In this variant, each frame still occupies a square bbox, but its border has a 1-2 pixel gap
#      on one side. The gap is background-colored. In addition, the generator may extend a thin
#      background corridor 1-2 cells inward from that gap. This produces a boundary-connected
#      background component inside the nominal interior.
#    - Example: a 7x7 frame may have a top-edge gap at (0,3) and an inward corridor at (1,3).
#      Original logic on inner = region[1:-1,1:-1] would count that corridor as one "hole".
#      Variant logic must instead label all non-frame-color cells in the full 7x7 region and ignore
#      any component that touches the outer boundary. The open corridor is therefore not a hole.
#    - Output stays a tight grid of matched frames with sprites nested inside, but now the output
#      preserves the frame's border gaps. The sprite still fills the entire interior and its own
#      holes are recolored to the frame color.
#
# b) Complexity calibration
#    - The gap is visible on every frame, so the variant is not a hidden parser-only change.
#    - The inward corridor is narrow (width 1, depth 1-2) so the frame remains easy to recognize
#      as a square component, and enclosed-hole matching is still the only pairing key.
#    - Hole counts remain a consecutive unique set 0..N-1, preserving bijective matching.
#
# c) Implicit-assumption audit of the original main()
#    - `frame_color = border majority`: still valid because each border contains only 1-2 bg gap
#      pixels while the rest of the border remains frame-colored.
#    - `inner = region[1:-1,1:-1]; hole_mask = inner != frame_color`: no longer valid, because a
#      boundary-connected corridor can enter the interior through a border gap.
#    - `out[frame box] = frame_color`: no longer appropriate if we want the D18 change to remain
#      visible in the output. We therefore copy the actual frame patch, then overwrite only the
#      interior with the matched sprite.
#
# d) Placement legality
#    - Frames remain axis-aligned same-sized square components in a tight grid.
#    - Sprites are still placed by `random_free_location`, so they do not overlap frames or each
#      other and remain outside the grid footprint.
#    - Enclosed frame holes are placed only on interior cells that are not adjacent to the open
#      corridor, ensuring the intended enclosed-hole count is exact.
#
# e) Visual observability
#    - Across 2-4 ARC examples a human can see that border gaps persist on the output frames, but
#      those gap structures do not affect which sprite is inserted; only fully enclosed holes do.
#    - The contrast between an open notch and a sealed interior hole is spatial, not color-coded,
#      so the rule is inferable from a few examples.
#
# f) Visual cleanliness
#    - Each frame has only one short gap segment and one thin inward corridor, avoiding clutter.
#    - Sprite interiors replace the noisy filled interior of the input frame, so the output stays
#      clean and grid-like.
#
# g) Color reasoning
#    - No new color mapping is introduced. Only the original recoloring rule remains: sprite holes
#      become frame-colored after insertion.


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
GRID_DIM_RANGE = (1, 4)
N_RANGE = (2, 6)
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PADDING = 2
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 80
GEN_RETRIES = 400


def _count_legacy_inner_components(region: np.ndarray, frame_color: int) -> int:
    inner = region[1:-1, 1:-1]
    _lab, n = label_mask(inner != int(frame_color), connectivity=4)
    return int(n)


def _count_enclosed_frame_holes(region: np.ndarray, frame_color: int) -> int:
    """Count non-frame-color components that are fully enclosed by frame_color."""
    mask = region != int(frame_color)
    lab, n = label_mask(mask, connectivity=4)
    holes = 0
    h, w = region.shape
    for k in range(1, int(n) + 1):
        coords = np.argwhere(lab == k)
        if coords.size == 0:
            continue
        touches_boundary = any(r in (0, h - 1) or c in (0, w - 1) for r, c in coords)
        if not touches_boundary:
            holes += 1
    return int(holes)


def _candidate_positions(size: int, forbidden: set[tuple[int, int]] | None = None) -> list[tuple[int, int]]:
    forbidden = forbidden or set()
    blocked = set(forbidden)
    for r, c in forbidden:
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = r + dr, c + dc
            if 1 <= nr <= size - 2 and 1 <= nc <= size - 2:
                blocked.add((nr, nc))
    return [(r, c) for r in range(1, size - 1) for c in range(1, size - 1) if (r, c) not in blocked]


def _place_isolated_seeds(
    size: int,
    count: int,
    forbidden: set[tuple[int, int]] | None = None,
) -> list[tuple[int, int]] | None:
    """Place `count` interior cells, pairwise 4-isolated and not adjacent to `forbidden`."""
    positions = _candidate_positions(size, forbidden=forbidden)
    if count == 0:
        return []
    for _ in range(SEED_RETRIES):
        random.shuffle(positions)
        seeds: list[tuple[int, int]] = []
        for r, c in positions:
            if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in seeds):
                seeds.append((r, c))
                if len(seeds) == count:
                    return seeds
    return None


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    """Build a sprite with exactly `hole_count` 4-connected bg-hole components."""
    sprite = np.full((size, size), sprite_color, dtype=int)
    if hole_count == 0:
        return sprite

    inner_cells = (size - 2) ** 2
    if hole_count > inner_cells:
        return None

    seeds = _place_isolated_seeds(size, hole_count)
    if seeds is None:
        return None

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
    all_hole = set(seeds)
    for r, c in seeds:
        sprite[r, c] = BG

    max_grow = max(1, inner_cells // hole_count - 1)
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

    _lab, num = label_mask(sprite != int(sprite_color), connectivity=4)
    if int(num) != int(hole_count):
        return None
    return sprite


def _make_open_notch(frame_size: int) -> set[tuple[int, int]]:
    """Create a boundary-connected bg notch: 1-2 border-gap cells + 1-2 inward corridor cells."""
    side = random.choice(["top", "bottom", "left", "right"])
    gap_len = random.randint(1, 2)
    depth = random.randint(1, 2)
    notch: set[tuple[int, int]] = set()

    if side in {"top", "bottom"}:
        start = random.randint(1, frame_size - 1 - gap_len)
        mouth = start + random.randint(0, gap_len - 1)
        border_r = 0 if side == "top" else frame_size - 1
        step = 1 if side == "top" else -1
        for j in range(gap_len):
            notch.add((border_r, start + j))
        for d in range(1, depth + 1):
            notch.add((border_r + step * d, mouth))
    else:
        start = random.randint(1, frame_size - 1 - gap_len)
        mouth = start + random.randint(0, gap_len - 1)
        border_c = 0 if side == "left" else frame_size - 1
        step = 1 if side == "left" else -1
        for j in range(gap_len):
            notch.add((start + j, border_c))
        for d in range(1, depth + 1):
            notch.add((mouth, border_c + step * d))

    return notch


def _make_frame(frame_size: int, hole_count: int, frame_color: int) -> np.ndarray | None:
    """Build a frame patch with a border gap/open notch plus `hole_count` enclosed bg holes."""
    frame = np.full((frame_size, frame_size), frame_color, dtype=int)
    notch = _make_open_notch(frame_size)
    for r, c in notch:
        frame[r, c] = BG

    seeds = _place_isolated_seeds(frame_size, hole_count, forbidden=notch)
    if seeds is None:
        return None
    for r, c in seeds:
        frame[r, c] = BG

    enclosed = _count_enclosed_frame_holes(frame, frame_color)
    legacy = _count_legacy_inner_components(frame, frame_color)
    if int(enclosed) != int(hole_count):
        return None
    if int(legacy) <= int(enclosed):
        return None
    return frame


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match sprites to frames by enclosed-hole count, then nest each sprite inside its frame.

    D18 change:
    - frame borders may contain 1-2 bg gap pixels;
    - those gaps can connect to a thin background notch entering the interior;
    - only fully enclosed non-frame-color components count as frame holes.
    """
    g = np.array(grid, dtype=int)
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
            non_bg = [int(x) for x in region.flatten() if int(x) != BG]
            frame_color = int(Counter(non_bg).most_common(1)[0][0]) if non_bg else int(BG)
            holes = _count_enclosed_frame_holes(region, frame_color)
            frame_grid.append(
                {
                    "color": frame_color,
                    "patch": region.copy(),
                    "out_r": int(ri * frame_size),
                    "out_c": int(ci * frame_size),
                    "holes": int(holes),
                }
            )

    sprites = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        _lab, num_holes = label_mask(sprite_region != int(sprite_color), connectivity=4)
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
    out = np.full((out_h, out_w), int(BG), dtype=int)

    for f in frame_grid:
        fr, fc = int(f["out_r"]), int(f["out_c"])
        frame_color = int(f["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = np.array(f["patch"], dtype=int)

        if (fr, fc) in matches:
            _f, sprite = matches[(fr, fc)]
            content = np.array(sprite["content"], dtype=int).copy()
            sprite_color = int(sprite["color"])
            content[content != int(sprite_color)] = int(frame_color)

            sh, sw = int(sprite["h"]), int(sprite["w"])
            out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content

    return out


_LAYOUTS_BY_N: dict[int, list[tuple[int, int]]] = {}
for _r in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
    for _c in range(GRID_DIM_RANGE[0], GRID_DIM_RANGE[1]):
        if N_RANGE[0] <= _r * _c <= N_RANGE[1]:
            _LAYOUTS_BY_N.setdefault(_r * _c, []).append((_r, _c))
_VALID_NS = sorted(_LAYOUTS_BY_N.keys())


def _build_expected_output(
    frames: list[tuple[int, np.ndarray]],
    sprites: list[tuple[int, np.ndarray]],
    rows_grid: int,
    cols_grid: int,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    """Build the output grid while preserving frame border gaps."""
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_k = {k: spatch for k, spatch in sprites}
    for idx, (k, frame_patch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = ri * frame_size, ci * frame_size
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_patch
        frame_color = int(Counter(int(x) for x in frame_patch.flatten() if int(x) != BG).most_common(1)[0][0])
        content = sprite_by_k[k].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


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

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, n)
    sprite_colors = random.sample(available, n)

    hole_counts = list(range(n))
    random.shuffle(hole_counts)
    sprite_hole_perm = hole_counts[:]
    random.shuffle(sprite_hole_perm)

    sprites: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(sprite_hole_perm):
        patch = _make_sprite_interior(sprite_size, k, sprite_colors[i])
        if patch is None:
            return None
        sprites.append((k, patch))

    frames: list[tuple[int, np.ndarray]] = []
    for i, k in enumerate(hole_counts):
        patch = _make_frame(frame_size, k, frame_colors[i])
        if patch is None:
            return None
        frames.append((k, patch))

    canvas = np.full((canvas_h, canvas_w), BG, dtype=int)
    r0 = random.randint(0, canvas_h - grid_pixel_h)
    c0 = random.randint(0, canvas_w - grid_pixel_w)
    for idx, (_, fpatch) in enumerate(frames):
        ri, ci = idx // cols_grid, idx % cols_grid
        rr, cc = r0 + ri * frame_size, c0 + ci * frame_size
        canvas[rr : rr + frame_size, cc : cc + frame_size] = fpatch

    for _, spatch in sprites:
        try:
            r, c = random_free_location(canvas, spatch, transparent=BG, padding=SPRITE_PLACEMENT_PAD)
        except ValueError:
            return None
        paste(canvas, spatch, r, c, transparent=BG)

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, rows_grid, cols_grid, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
