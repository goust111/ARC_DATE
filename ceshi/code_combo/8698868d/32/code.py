# Combo variant of 8698868d (index 32)
# combo_spec: D11/0+D19/2
# Name: L/T形网格+行列转置输出

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste


# ---------------------------------------------------------------------------
# Design analysis for combo D11/0 + D19/2
#
# a) Concrete behavior and trace
# - Base task: large square frames form a full rows x cols rectangle, each small sprite
#   is matched to one frame by connected-hole count, and the solved output keeps the same
#   row-major frame arrangement.
# - D11/0 changes the frame layout: only some lattice slots are occupied, and the visible
#   slots form an L or T polyomino. Missing slots are real absences, not invisible frames.
# - D19/2 changes placement after solving: a frame read at logical slot (ri, ci) must be
#   written to output slot (ci, ri).
# - Combined trace on a 2x3 mask:
#     input occupied slots = {(0,0), (0,1), (0,2), (1,1)}  # a T shape
#     old solver assumption = all 6 slots exist, then output keeps 2x3 order
#     new solver behavior   = keep only those 4 true slots, match by holes, and place
#                             them at {(0,0), (1,0), (2,0), (1,1)} in the output.
#   The transpose acts on slot coordinates, so a non-rectangular mask becomes a different
#   non-rectangular mask; a horizontal T becomes a vertical T, and an L can become a
#   reflected/rotated-looking counterpart depending on the chosen orientation.
#
# b) Complexity calibration
# - Layouts are restricted to a small, readable family of L/T masks on 2x2, 2x3, 3x2, or
#   3x3 slot lattices. This keeps the rule visually obvious while still breaking the full
#   rectangle assumption.
# - Hole-count matching remains unique and unchanged from the original task, so the new
#   difficulty is purely spatial: detect missing slots first, then transpose the solved
#   slot map.
# - The generator rejects degenerate cases where transposed and non-transposed outputs are
#   accidentally identical on the same input.
#
# c) Implicit-assumption audit of the original solver
# - Original `n_rows/n_cols` inference from min/max coordinates is still usable only as a
#   lattice extent, but iterating over every slot inside that extent is no longer valid.
# - Original border-majority reading of every bounding-rectangle slot would invent bogus
#   frames on missing positions; the combo solver instead iterates only over true frame
#   components and maps their observed top-left coordinates to lattice indices.
# - Original matching keyed by output coordinates is also brittle once output positions are
#   no longer equal to input positions. The combo solver keeps the logical slot identity
#   `(ri, ci)` separate from the output coordinates `(ci, ri)`.
#
# d) Placement legality
# - Every real frame occupies one slot of a tight lattice with step `frame_size`.
# - Missing slots are left completely background-colored.
# - Sprites are placed strictly outside the entire frame-layout bounding rectangle and with
#   padding between sprite rectangles, so they cannot overlap frames or visually sit inside
#   a missing lattice slot.
#
# e) Visual observability
# - Humans can see the L/T layout because empty slots remain blank both in input and output.
# - Humans can see the transpose because the occupied shape itself changes orientation, and
#   on rectangular masks the output dimensions flip as well.
# - No part of the rule depends on numeric color ids or hidden ordering; only visible slot
#   positions and visible hole counts matter.
#
# f) Visual cleanliness
# - The frame layout is kept in one corner as a compact block.
# - Sprites are pushed outside the entire layout rectangle, which preserves a clean visual
#   silhouette for the L/T mask and its transposed output counterpart.
#
# g) Color/change inference
# - No new color remapping is introduced. As in the original task, only sprite hole pixels
#   are recolored to the matched frame color after nesting.
#
# h) New solver assumptions and generator guarantees
# - All large frame components share one square size; all sprite components share the next
#   smaller square size `frame_size - 2`.
# - Distinct observed frame top-left rows/cols form the logical slot lattice.
# - Hole counts are unique across frames and sprites, so dictionary matching by hole count
#   is safe. The generator enforces all three assumptions.
# ---------------------------------------------------------------------------


BG = 8
FRAME_SIZE_RANGE = [6, 7, 8]
CANVAS_MIN = 15
CANVAS_MAX = 30
SPRITE_PLACEMENT_PAD = 1
SEED_RETRIES = 50
GEN_RETRIES = 400


def _mask_variants(mask: np.ndarray) -> list[np.ndarray]:
    out: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in (mask, np.fliplr(mask)):
        cur = np.array(base, dtype=int)
        for _ in range(4):
            key = tuple(tuple(int(v) for v in row) for row in cur.tolist())
            if key not in seen:
                seen.add(key)
                out.append(cur.copy())
            cur = np.rot90(cur)
    return out


_BASE_LAYOUTS = [
    ("L", np.array([[1, 1], [1, 0]], dtype=int)),
    ("L", np.array([[1, 1, 1], [1, 0, 0]], dtype=int)),
    ("T", np.array([[1, 1, 1], [0, 1, 0]], dtype=int)),
    ("L", np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=int)),
    ("T", np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=int)),
]

LAYOUTS: list[dict[str, object]] = []
for kind, base in _BASE_LAYOUTS:
    for mask in _mask_variants(base):
        LAYOUTS.append({"kind": kind, "mask": mask, "count": int(mask.sum())})


def _occupied_slots(mask: np.ndarray) -> list[tuple[int, int]]:
    rows, cols = map(int, mask.shape)
    return [(r, c) for r in range(rows) for c in range(cols) if int(mask[r, c]) == 1]


def _slot_to_out_coords(slot_r: int, slot_c: int, frame_size: int, *, transpose: bool) -> tuple[int, int]:
    if transpose:
        return int(slot_c * frame_size), int(slot_r * frame_size)
    return int(slot_r * frame_size), int(slot_c * frame_size)


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


def _make_sprite_interior(size: int, hole_count: int, sprite_color: int) -> np.ndarray | None:
    sprite = np.full((size, size), int(sprite_color), dtype=int)
    if hole_count == 0:
        return sprite

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
            frontier: list[tuple[int, int]] = []
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
                    frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _, num_holes = label_mask(sprite != int(sprite_color), connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return sprite


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
        frame[r, c] = BG

    _, num_holes = label_mask(frame != int(frame_color), connectivity=4)
    if int(num_holes) != int(hole_count):
        return None
    return frame


def _rects_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + pad <= br0
        or br1 + pad <= ar0
        or ac1 + pad <= bc0
        or bc1 + pad <= ac0
    )


def _place_sprites_outside_layout(
    canvas: np.ndarray,
    sprite_patches: Iterable[np.ndarray],
    layout_rect: tuple[int, int, int, int],
) -> bool:
    placed_rects: list[tuple[int, int, int, int]] = []
    height, width = map(int, canvas.shape)
    for spatch in sprite_patches:
        sh, sw = map(int, spatch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(height - sh + 1):
            for c0 in range(width - sw + 1):
                rect = (int(r0), int(c0), int(r0 + sh), int(c0 + sw))
                if _rects_overlap(rect, layout_rect, pad=SPRITE_PLACEMENT_PAD):
                    continue
                if any(_rects_overlap(rect, other, pad=SPRITE_PLACEMENT_PAD) for other in placed_rects):
                    continue
                if np.any(canvas[r0 : r0 + sh, c0 : c0 + sw] != BG):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        paste(canvas, spatch, r0, c0, transparent=BG)
        placed_rects.append((int(r0), int(c0), int(r0 + sh), int(c0 + sw)))
    return True


def _build_expected_output(
    frames: list[dict[str, object]],
    sprites: list[dict[str, object]],
    mask: np.ndarray,
    frame_size: int,
    sprite_size: int,
    *,
    transpose: bool,
) -> np.ndarray:
    rows_grid, cols_grid = map(int, mask.shape)
    out_h = int((cols_grid if transpose else rows_grid) * frame_size)
    out_w = int((rows_grid if transpose else cols_grid) * frame_size)
    expected = np.full((out_h, out_w), BG, dtype=int)
    sprite_by_holes = {int(sprite["holes"]): np.array(sprite["patch"], dtype=int) for sprite in sprites}

    for frame in frames:
        slot_r = int(frame["slot_r"])
        slot_c = int(frame["slot_c"])
        rr, cc = _slot_to_out_coords(slot_r, slot_c, frame_size, transpose=transpose)
        frame_color = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_holes[int(frame["holes"])].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content

    return expected


# Match sprites to frames by connected-hole count, then nest each sprite inside its frame.
#
# Input structure:
# - Background is azure (8). Large square frames occupy only some slots of a tight lattice,
#   and the occupied slots form an L or T shape rather than a full rectangle.
# - Smaller square sprites of various colors are scattered elsewhere on the canvas, outside
#   the frame-layout rectangle. Sprite interiors may contain one or more hole components.
#
# Rule:
# - Recover the logical slot lattice from the real frame components only; missing lattice
#   slots stay missing.
# - Count 4-connected hole components inside each frame interior and inside each sprite,
#   then pair equal counts.
# - Nest each matched sprite into its frame, recoloring the sprite's hole pixels to the
#   frame color, and write the solved frame to the transposed slot position `(ci, ri)`.
#
# Key insight:
# - The solver must separate two coordinate systems: detect the irregular occupied slot set
#   from the input, then transpose those slot coordinates in the output. The matching cue is
#   still the number of connected hole components, not color identity or hole area.
def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

    min_fill_ratio = 0.5
    square_regions: list[tuple[int, dict]] = []
    for comp in comps:
        bbox = comp["bbox"]
        h = int(bbox_h(bbox))
        w = int(bbox_w(bbox))
        area = int(comp["area"])
        if h == w and 4 <= h <= 10 and area >= int(h * h * min_fill_ratio):
            square_regions.append((int(h), comp))

    assert square_regions

    sizes: dict[int, list[dict]] = {}
    for size, comp in square_regions:
        sizes.setdefault(int(size), []).append(comp)

    size_list = sorted(sizes.keys(), reverse=True)
    frame_size = int(size_list[0])
    sprite_size = int(size_list[1]) if len(size_list) > 1 else int(frame_size - 2)

    frames_raw = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames_raw
    assert sprites_raw

    row_vals = sorted({int(comp["bbox"][0]) for comp in frames_raw})
    col_vals = sorted({int(comp["bbox"][1]) for comp in frames_raw})
    row_to_idx = {row: idx for idx, row in enumerate(row_vals)}
    col_to_idx = {col: idx for idx, col in enumerate(col_vals)}

    frames: list[dict[str, object]] = []
    for comp in frames_raw:
        bbox = comp["bbox"]
        top_r = int(bbox[0])
        top_c = int(bbox[1])
        region = crop_bbox(g, bbox)
        frame_color = int(comp["color"])
        inner = region[1:-1, 1:-1]
        hole_mask = inner != int(frame_color)
        _, num_holes = label_mask(hole_mask, connectivity=4)
        frames.append(
            {
                "slot_r": int(row_to_idx[top_r]),
                "slot_c": int(col_to_idx[top_c]),
                "color": int(frame_color),
                "holes": int(num_holes),
            }
        )

    sprites: list[dict[str, object]] = []
    for comp in sprites_raw:
        sprite_region = crop_bbox(g, comp["bbox"])
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != int(sprite_color)
        _, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": int(sprite_color),
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    sprite_by_holes = {int(sprite["holes"]): sprite for sprite in sprites}
    assert len(sprite_by_holes) == len(sprites)

    out_h = int(len(col_vals) * frame_size)
    out_w = int(len(row_vals) * frame_size)
    out = np.full((out_h, out_w), BG, dtype=int)

    for frame in sorted(frames, key=lambda item: (int(item["slot_r"]), int(item["slot_c"]))):
        rr, cc = _slot_to_out_coords(int(frame["slot_r"]), int(frame["slot_c"]), frame_size, transpose=True)
        frame_color = int(frame["color"])
        out[rr : rr + frame_size, cc : cc + frame_size] = frame_color

        sprite = sprite_by_holes[int(frame["holes"])]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != int(sprite_color)] = frame_color

        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[rr + 1 : rr + 1 + sh, cc + 1 : cc + 1 + sw] = content

    return out


def _gen_attempt() -> np.ndarray | None:
    layout = random.choice(LAYOUTS)
    mask = np.array(layout["mask"], dtype=int)
    kind = str(layout["kind"])
    frame_count = int(layout["count"])
    rows_grid, cols_grid = map(int, mask.shape)

    frame_size = random.choice(FRAME_SIZE_RANGE)
    sprite_size = frame_size - 2
    grid_pixel_h = rows_grid * frame_size
    grid_pixel_w = cols_grid * frame_size

    min_h = max(CANVAS_MIN, grid_pixel_h + sprite_size)
    min_w = max(CANVAS_MIN, grid_pixel_w + sprite_size)
    if min_h > CANVAS_MAX or min_w > CANVAS_MAX:
        return None
    height = random.randint(max(min_h, CANVAS_MAX - 2), CANVAS_MAX)
    width = random.randint(max(min_w, CANVAS_MAX - 2), CANVAS_MAX)

    available = [color for color in range(10) if color != BG]
    frame_colors = random.sample(available, frame_count)
    sprite_colors = random.sample(available, frame_count)

    slots = _occupied_slots(mask)
    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    if max_hole < frame_count - 1:
        return None

    missing_indices = [
        idx
        for idx in range(rows_grid * cols_grid)
        if int(mask[idx // cols_grid, idx % cols_grid]) == 0
    ]
    if not missing_indices:
        return None

    earliest_missing = min(missing_indices)
    zero_candidates = [(r, c) for r, c in slots if int(r * cols_grid + c) > earliest_missing]
    if not zero_candidates:
        return None

    zero_slot = random.choice(zero_candidates)
    other_holes = list(range(1, frame_count))
    random.shuffle(other_holes)
    frame_holes_by_slot: dict[tuple[int, int], int] = {}
    for slot in slots:
        if slot == zero_slot:
            frame_holes_by_slot[slot] = 0
        else:
            frame_holes_by_slot[slot] = int(other_holes.pop())

    sprite_hole_perm = list(range(frame_count))
    random.shuffle(sprite_hole_perm)

    frames: list[dict[str, object]] = []
    for (slot_r, slot_c), color in zip(slots, frame_colors):
        hole_count = int(frame_holes_by_slot[(slot_r, slot_c)])
        patch = _make_frame(frame_size, hole_count, int(color))
        if patch is None:
            return None
        frames.append(
            {
                "slot_r": int(slot_r),
                "slot_c": int(slot_c),
                "holes": int(hole_count),
                "color": int(color),
                "patch": patch,
            }
        )

    sprites: list[dict[str, object]] = []
    for hole_count, color in zip(sprite_hole_perm, sprite_colors):
        patch = _make_sprite_interior(sprite_size, int(hole_count), int(color))
        if patch is None:
            return None
        sprites.append({"holes": int(hole_count), "color": int(color), "patch": patch})

    canvas = np.full((height, width), BG, dtype=int)

    corner_choices = [
        (0, 0),
        (0, width - grid_pixel_w),
        (height - grid_pixel_h, 0),
        (height - grid_pixel_h, width - grid_pixel_w),
    ]
    r0, c0 = random.choice(corner_choices)
    layout_rect = (int(r0), int(c0), int(r0 + grid_pixel_h), int(c0 + grid_pixel_w))

    for frame in frames:
        rr = int(r0 + int(frame["slot_r"]) * frame_size)
        cc = int(c0 + int(frame["slot_c"]) * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    sprite_patches = [np.array(sprite["patch"], dtype=int) for sprite in sprites]
    random.shuffle(sprite_patches)
    if not _place_sprites_outside_layout(canvas, sprite_patches, layout_rect):
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(
        frames,
        sprites,
        mask,
        frame_size,
        sprite_size,
        transpose=True,
    )
    old_expected = _build_expected_output(
        frames,
        sprites,
        mask,
        frame_size,
        sprite_size,
        transpose=False,
    )

    if np.array_equal(expected, old_expected):
        return None
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None
    if kind not in {"L", "T"}:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
