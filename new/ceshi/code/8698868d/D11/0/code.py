# Variant of 8698868d
# Dimension: D11 — 框架网格布局 (meta)
# Change: 框架排成 rows x cols 的紧密矩形网格（1-3行 x 1-3列），总数 2-6 个... → L 形或 T 形网格布局

from __future__ import annotations

import random
from typing import Iterable

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, label_mask, paste


# ---------------------------------------------------------------------------
# Design analysis for the D11 variant
#
# a) Concrete behavior
# - Original: the large frames occupy every slot of a tight rows x cols rectangle.
# - Variant: the frames still lie on a tight slot lattice with step = frame_size, but
#   only some slots are occupied. The occupied slots form an L or T polyomino.
# - Output is the minimal slot-aligned bounding rectangle of that polyomino. Occupied
#   slots are filled as before; missing slots stay BG, so output can now contain BG.
# - Small trace:
#   Old 2x2 full grid: frame slots {(0,0),(0,1),(1,0),(1,1)}.
#   New 2x2 L grid:    frame slots {(0,0),(0,1),(1,0)} and slot (1,1) is empty.
#   The old solver would derive n_rows=2,n_cols=2 from min/max coords and then read
#   slot (1,1) as if a frame existed there, manufacturing a bogus extra frame from BG.
#   The new solver must instead keep only the actual frame components and map them onto
#   the unique lattice rows/cols observed in the input.
#
# b) Complexity calibration
# - Layout family is restricted to a small visible set: L/T masks on 2x2, 2x3/3x2, or
#   3x3 lattices, rotated/flipped. Frame count is therefore 3-5.
# - Matching signal stays unchanged: unique connected-hole counts still give a bijection
#   between frames and sprites, so the new difficulty is specifically spatial layout.
#
# c) Implicit-assumption audit of the original main()
# - `n_rows = (max_r-min_r)//frame_size + 1` and the analogous `n_cols` assume every
#   lattice slot in the bounding rectangle is occupied. This fails for L/T layouts.
# - The nested loop over all `(ri,ci)` then samples absent slots; on an empty slot the
#   border majority is BG, so the solver invents a non-frame with arbitrary hole count.
# - The rest of the pipeline (hole counting, sprite matching, nesting) remains valid if
#   frame slots are taken only from real components.
#
# d) Placement legality
# - Generator aligns each frame to one slot of a 2-3 by 2-3 lattice with step=frame_size.
# - Missing lattice slots are left fully BG.
# - Sprites are placed strictly outside the entire layout bounding rectangle, not merely
#   outside occupied frames, so they can never visually sit in a missing slot.
# - Sprite rectangles are non-overlapping and separated by a small padding.
#
# e) Visual observability
# - The missing slot is visible in both input and output. Because the output keeps the
#   slot-aligned bounding rectangle, the human solver must infer the irregular L/T shape,
#   not just an unordered set of frames.
# - Rotations/flips are varied so the task is not reducible to one fixed corner/stem.
# - To prevent accidental degeneration back to the old behavior, generation always keeps
#   a real 0-hole frame after at least one missing slot in row-major order. The original
#   rectangle-assuming solver will then consume the 0-hole sprite on a bogus empty slot,
#   producing a visibly different output on the same input.
#
# f) Visual cleanliness
# - Sprites are kept away from the whole frame-layout rectangle. This avoids clutter in
#   the exact region where the new D11 signal (missing slots) needs to be read.
#
# g) Color-change inference
# - No extra color mapping is introduced. The only recoloring remains the original one:
#   sprite hole pixels become the matched frame color after nesting.
#
# h) New solver assumptions and generator guarantees
# - Solver assumption: all large frame components share one square size and their top-left
#   coordinates lie on a single slot lattice.
# - Solver assumption: each hole count appears exactly once among frames and sprites.
# - Generator enforces both assumptions directly.
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
    return [(int(r), int(c)) for r in range(mask.shape[0]) for c in range(mask.shape[1]) if int(mask[r, c]) == 1]


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

    comps: list[set[tuple[int, int]]] = [{s} for s in seeds]
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
                    frontier.append((int(nr), int(nc)))
            if not frontier:
                break
            cell = random.choice(frontier)
            comp.add(cell)
            all_hole.add(cell)
            sprite[cell[0], cell[1]] = BG

    _, num = label_mask(sprite != int(sprite_color), connectivity=4)
    if int(num) != int(hole_count):
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

    _, num = label_mask(frame != int(frame_color), connectivity=4)
    if int(num) != int(hole_count):
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
    H, W = canvas.shape
    for spatch in sprite_patches:
        h, w = map(int, spatch.shape)
        candidates: list[tuple[int, int]] = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                rect = (int(r0), int(c0), int(r0 + h), int(c0 + w))
                if _rects_overlap(rect, layout_rect, pad=SPRITE_PLACEMENT_PAD):
                    continue
                if any(_rects_overlap(rect, other, pad=SPRITE_PLACEMENT_PAD) for other in placed_rects):
                    continue
                if np.any(canvas[r0 : r0 + h, c0 : c0 + w] != BG):
                    continue
                candidates.append((int(r0), int(c0)))
        if not candidates:
            return False
        r0, c0 = random.choice(candidates)
        paste(canvas, spatch, r0, c0, transparent=BG)
        placed_rects.append((int(r0), int(c0), int(r0 + h), int(c0 + w)))
    return True


def _build_expected_output(
    frames: list[dict[str, object]],
    sprites: list[dict[str, object]],
    mask: np.ndarray,
    frame_size: int,
    sprite_size: int,
) -> np.ndarray:
    rows_grid, cols_grid = map(int, mask.shape)
    expected = np.full((rows_grid * frame_size, cols_grid * frame_size), BG, dtype=int)
    sprite_by_holes = {int(s["holes"]): np.array(s["patch"], dtype=int) for s in sprites}
    for frame in frames:
        rr = int(frame["slot_r"]) * frame_size
        cc = int(frame["slot_c"]) * frame_size
        frame_color = int(frame["color"])
        expected[rr : rr + frame_size, cc : cc + frame_size] = frame_color
        content = sprite_by_holes[int(frame["holes"])].copy()
        content[content == BG] = frame_color
        expected[rr + 1 : rr + 1 + sprite_size, cc + 1 : cc + 1 + sprite_size] = content
    return expected


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    comps = find_components(g, bg=BG, connectivity=4, monochromatic=True, return_format="dict")

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

    frames_raw = sizes.get(frame_size, [])
    sprites_raw = sizes.get(sprite_size, [])
    assert frames_raw
    assert sprites_raw

    row_vals = sorted({int(f["bbox"][0]) for f in frames_raw})
    col_vals = sorted({int(f["bbox"][1]) for f in frames_raw})
    row_to_idx = {r: i for i, r in enumerate(row_vals)}
    col_to_idx = {c: i for i, c in enumerate(col_vals)}

    frames: list[dict[str, object]] = []
    for comp in frames_raw:
        b = comp["bbox"]
        top_r = int(b[0])
        top_c = int(b[1])
        region = crop_bbox(g, b)
        frame_color = int(comp["color"])
        inner = region[1:-1, 1:-1]
        hole_mask = inner != frame_color
        _, num_holes = label_mask(hole_mask, connectivity=4)
        frames.append(
            {
                "color": frame_color,
                "holes": int(num_holes),
                "out_r": int(row_to_idx[top_r] * frame_size),
                "out_c": int(col_to_idx[top_c] * frame_size),
            }
        )

    sprites: list[dict[str, object]] = []
    for comp in sprites_raw:
        b = comp["bbox"]
        sprite_region = crop_bbox(g, b)
        sprite_color = int(comp["color"])
        hole_mask = sprite_region != sprite_color
        _, num_holes = label_mask(hole_mask, connectivity=4)
        sprites.append(
            {
                "color": sprite_color,
                "holes": int(num_holes),
                "content": sprite_region.copy(),
                "h": int(sprite_region.shape[0]),
                "w": int(sprite_region.shape[1]),
            }
        )

    matches: dict[tuple[int, int], dict[str, object]] = {}
    used_sprites: set[int] = set()
    for frame in sorted(frames, key=lambda f: (int(f["out_r"]), int(f["out_c"]))):
        for i, sprite in enumerate(sprites):
            if i not in used_sprites and int(frame["holes"]) == int(sprite["holes"]):
                matches[(int(frame["out_r"]), int(frame["out_c"]))] = sprite
                used_sprites.add(i)
                break

    out = np.full((len(row_vals) * frame_size, len(col_vals) * frame_size), BG, dtype=int)
    for frame in sorted(frames, key=lambda f: (int(f["out_r"]), int(f["out_c"]))):
        fr = int(frame["out_r"])
        fc = int(frame["out_c"])
        frame_color = int(frame["color"])
        out[fr : fr + frame_size, fc : fc + frame_size] = frame_color
        sprite = matches[(fr, fc)]
        content = np.array(sprite["content"], dtype=int).copy()
        sprite_color = int(sprite["color"])
        content[content != sprite_color] = frame_color
        sh = int(sprite["h"])
        sw = int(sprite["w"])
        out[fr + 1 : fr + 1 + sh, fc + 1 : fc + 1 + sw] = content
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

    min_H = max(CANVAS_MIN, grid_pixel_h + sprite_size)
    min_W = max(CANVAS_MIN, grid_pixel_w + sprite_size)
    if min_H > CANVAS_MAX or min_W > CANVAS_MAX:
        return None
    H = random.randint(max(min_H, CANVAS_MAX - 2), CANVAS_MAX)
    W = random.randint(max(min_W, CANVAS_MAX - 2), CANVAS_MAX)

    available = [c for c in range(10) if c != BG]
    frame_colors = random.sample(available, frame_count)
    sprite_colors = random.sample(available, frame_count)

    inner = frame_size - 2
    max_hole = (inner * inner + 1) // 2
    slots = _occupied_slots(mask)
    missing_indices = [idx for idx in range(rows_grid * cols_grid) if int(mask[idx // cols_grid, idx % cols_grid]) == 0]
    if not missing_indices:
        return None
    earliest_missing = min(missing_indices)
    zero_candidates = [(r, c) for r, c in slots if int(r * cols_grid + c) > earliest_missing]
    if not zero_candidates:
        return None
    if max_hole < frame_count - 1:
        return None

    zero_slot = random.choice(zero_candidates)
    other_holes = list(range(1, frame_count))
    random.shuffle(other_holes)
    frame_holes_by_slot: dict[tuple[int, int], int] = {}
    for slot in slots:
        if slot == zero_slot:
            frame_holes_by_slot[slot] = 0
        else:
            frame_holes_by_slot[slot] = other_holes.pop()

    sprite_hole_perm = list(range(frame_count))
    random.shuffle(sprite_hole_perm)

    frames: list[dict[str, object]] = []
    for (slot_r, slot_c), color in zip(slots, frame_colors):
        hole_count = int(frame_holes_by_slot[(slot_r, slot_c)])
        patch = _make_frame(frame_size, hole_count, color)
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
        patch = _make_sprite_interior(sprite_size, hole_count, color)
        if patch is None:
            return None
        sprites.append({"holes": int(hole_count), "color": int(color), "patch": patch})

    canvas = np.full((H, W), BG, dtype=int)

    corner_choices = [
        (0, 0),
        (0, W - grid_pixel_w),
        (H - grid_pixel_h, 0),
        (H - grid_pixel_h, W - grid_pixel_w),
    ]
    r0, c0 = random.choice(corner_choices)
    layout_rect = (int(r0), int(c0), int(r0 + grid_pixel_h), int(c0 + grid_pixel_w))

    for frame in frames:
        rr = int(r0 + int(frame["slot_r"]) * frame_size)
        cc = int(c0 + int(frame["slot_c"]) * frame_size)
        canvas[rr : rr + frame_size, cc : cc + frame_size] = np.array(frame["patch"], dtype=int)

    sprite_patches = [np.array(s["patch"], dtype=int) for s in sprites]
    random.shuffle(sprite_patches)
    if not _place_sprites_outside_layout(canvas, sprite_patches, layout_rect):
        return None

    try:
        out = main(canvas)
    except Exception:
        return None

    expected = _build_expected_output(frames, sprites, mask, frame_size, sprite_size)
    if out.shape != expected.shape or not np.array_equal(out, expected):
        return None

    # Keep a no-op reference so the layout family is part of the concrete generation path.
    # This is useful when auditing the file and makes the intended branch explicit.
    if kind not in {"L", "T"}:
        return None
    return canvas


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        g = _gen_attempt()
        if g is not None:
            return g
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES} attempts")
