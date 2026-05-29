# Combo variant of 5dbc8537 (index 7)
# combo_spec: D3/0+D7/2
# Name: 双模板多框架分组匹配

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and flow trace
-----------------------------------
- The original task had one strip-shaped template bbox and one global cavity set. Every outside sprite could be
  matched directly against the same destination region.
- This combo replaces that with two left-right symmetric template groups. Each group contains 2-3 disconnected
  square frames stacked vertically, and each frame contains exactly two rectangular holes.
- One hole size is repeated inside every frame of both groups. That forces a local frame-band decision inside a
  template. The other hole size is shared only by mirrored left/right frames at the same vertical level. That forces
  a template-side decision before local matching.
- Small trace:
  1. Original: detect one base bbox -> treat every non-base cell inside it as one destination mass -> exact-cover all sprites.
  2. Combo: detect the shared base color -> split it into frame components -> group frames into left and right
     symmetric template columns -> assign each sprite to a side by its visible placement outside the arrangement ->
     assign it to one frame by row-band overlap -> match size inside that frame.
- Difference: matching is now explicitly hierarchical. Global packing is no longer valid because identical rectangle
  sizes intentionally appear both across the two templates and across different frames within one template.

b) Complexity calibration
-------------------------
- The design is fixed to two symmetric template groups so the D7 change is always visible.
- Each template has multiple disconnected frames, so the D3 change is also always visible.
- The common repeated size across all frames prevents the task from collapsing into "size uniquely identifies slot".
- The mirrored repeated size per row prevents the task from collapsing into "once row is known, side is irrelevant".
- Each frame still has only two holes, so the visual scene remains readable and the solver logic stays bounded.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- Original `bbox(all base pixels)` still gives the correct output crop, but it can no longer be treated as one
  monolithic destination mask because bg gaps between frames must remain bg in the output.
- Original `hole_mask = dst != base_col` is invalid: it would wrongly absorb inter-frame gaps and the large center gap.
- Original global area equality and DFS over one cavity are invalid: repeated sizes across frames make a one-stage
  exact cover ambiguous and logically wrong.
- Safe replacement:
  recover every base-colored frame component, extract its own rectangular hole list, group the frames into the two
  template columns, then perform side assignment and row-band assignment before local size matching.

d) Placement legality
---------------------
- Frames are axis-aligned squares with a 1-cell base border and at least a 2-cell bg gap between neighboring frames.
- The two template columns are separated by a bg center gap, and their positions are horizontally symmetric.
- Every hole is a filled rectangle strictly inside one frame, and the two holes in a frame are separated by at least
  one base cell.
- Sprites are placed strictly outside the entire arrangement bbox, on the same side as their owning template, with
  1-cell 8-connected padding from other sprites and from the arrangement.

e) Visual observability
-----------------------
- A human can see two mirrored columns of frames in the input and the same crop in the output.
- Sprites for the left template appear only on the left side of the arrangement; sprites for the right template
  appear only on the right side. This makes template ownership visually recoverable.
- Within one side, each sprite overlaps exactly one frame's row band, so frame ownership is also recoverable by sight.
- Repeated rectangle sizes appear in multiple places, making the grouping cues necessary rather than decorative.

f) Visual cleanliness
---------------------
- The layout is only two columns and 2-3 rows of frames, with constant gaps, so the scene stays structured.
- Sprites never intrude into the crop area or into the center gap between templates.
- Limiting each frame to two holes avoids clutter and keeps repeated-size patterns legible.

g) Color inferability
---------------------
- Colors are not used as routing metadata. Pieces are matched by visible side and row alignment plus bbox size.
- Piece fills use one or two colors only, drawn from the normal ARC palette, and base color never appears inside a
  sprite. Repeated sizes may have different colors, so color is visibly non-authoritative.

h) New solver assumptions and generator guarantees
--------------------------------------------------
- Assumption: the dominant non-bg color by total area is the shared frame color.
  Guarantee: generator rejects samples where any sprite color reaches or exceeds the base count.
- Assumption: base-colored components are exactly the small destination frames.
  Guarantee: sprites never use the base color.
- Assumption: frames form two vertical columns of equal count.
  Guarantee: generator always builds 2-3 stacked frames on the left and mirrors their placements on the right.
- Assumption: each sprite lies on the correct outer side and overlaps exactly one frame row band.
  Guarantee: generator places sprites with side-specific column ranges and frame-specific row ranges.
- Assumption: hole sizes are unique within one frame.
  Guarantee: each frame contains exactly two different rectangle sizes.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild two symmetric multi-frame templates by routing each outside sprite to the correct template and frame.

    Input structure:
    - bg is the most common color. One shared base color forms several disconnected square frames arranged as two left-right symmetric columns.
    - Each column is one template group with 2-3 stacked frames. Rectangular sprites sit outside the arrangement on the same side as their owning template, and each sprite overlaps one frame's row band.

    Rule:
    - Recover the shared base color, split it into frame components, and crop the union bbox of the whole arrangement.
    - Group frames into left and right template columns by horizontal position. Then assign each outside sprite first to a side-template by side placement, and next to one frame by row-band overlap.
    - Inside that frame, match sprite bbox size to the unique local hole of the same size and paste it there. Keep bg in all gaps between frames and between the two templates.

    Key insight: size matching is now only the final step; because equal rectangle sizes are intentionally reused across mirrored templates and across frames, visible side-grouping and frame-band grouping must happen first.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    base_candidates: list[tuple[int, int, int]] = []
    for col in colors_present(g, exclude={bg}):
        comps = components_mask(g == int(col), connectivity=8)
        if not comps:
            continue
        total_area = sum(int(coords.shape[0]) for coords in comps)
        max_area = max(int(coords.shape[0]) for coords in comps)
        base_candidates.append((int(total_area), int(max_area), int(col)))

    assert base_candidates
    base_candidates.sort(reverse=True)
    if len(base_candidates) > 1:
        assert base_candidates[0][0] > base_candidates[1][0]
    base_col = int(base_candidates[0][2])

    frame_coords = components_mask(g == int(base_col), connectivity=8)
    assert len(frame_coords) >= 4 and len(frame_coords) % 2 == 0

    all_base = np.argwhere(g == int(base_col))
    ur0, uc0, ur1, uc1 = bbox_coords(all_base)
    out = np.full((int(ur1 - ur0 + 1), int(uc1 - uc0 + 1)), int(bg), dtype=int)

    frames: list[dict] = []
    for coords in frame_coords:
        fr0, fc0, fr1, fc1 = bbox_coords(coords)
        patch = crop_bbox(g, (int(fr0), int(fc0), int(fr1), int(fc1)))
        base_mask = patch == int(base_col)
        hole_mask = patch == int(bg)
        assert bool(hole_mask.any())

        hole_rects: list[tuple[int, int, int, int]] = []
        seen_sizes: set[tuple[int, int]] = set()
        for hole in components_mask(hole_mask, connectivity=4):
            hr0, hc0, hr1, hc1 = bbox_coords(hole)
            h = int(hr1 - hr0 + 1)
            w = int(hc1 - hc0 + 1)
            assert int(hole.shape[0]) == int(h * w)
            size = (int(h), int(w))
            assert size not in seen_sizes
            seen_sizes.add(size)
            hole_rects.append((int(hr0), int(hc0), int(h), int(w)))

        assert len(hole_rects) >= 2
        base_patch = np.full(patch.shape, int(base_col), dtype=int)
        paste(out, base_patch, int(fr0 - ur0), int(fc0 - uc0), mask=base_mask, overwrite=True)
        frames.append(
            {
                "bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
                "hole_rects": hole_rects,
                "center_c2": int(fc0 + fc1),
            }
        )

    frames.sort(key=lambda f: (int(f["center_c2"]), int(f["bbox"][0]), int(f["bbox"][1])))
    half = len(frames) // 2
    left_frames = sorted(frames[:half], key=lambda f: (int(f["bbox"][0]), int(f["bbox"][1])))
    right_frames = sorted(frames[half:], key=lambda f: (int(f["bbox"][0]), int(f["bbox"][1])))
    assert len(left_frames) == len(right_frames)

    split_col = 0.5 * (
        max(int(f["bbox"][3]) for f in left_frames) + min(int(f["bbox"][1]) for f in right_frames)
    )

    masked = g.copy()
    masked[g == int(base_col)] = int(bg)
    sprite_comps = find_components(masked, bg=bg, connectivity=8, monochromatic=False, return_format="dict")

    sprites: list[dict] = []
    for comp in sprite_comps:
        pr0, pc0, pr1, pc1 = map(int, comp["bbox"])
        patch = crop_bbox(g, (int(pr0), int(pc0), int(pr1), int(pc1)))
        ph, pw = map(int, patch.shape)
        mask = np.zeros((ph, pw), dtype=bool)
        coords = comp["coords"]
        mask[coords[:, 0] - pr0, coords[:, 1] - pc0] = True
        sprites.append(
            {
                "bbox": (int(pr0), int(pc0), int(pr1), int(pc1)),
                "h": int(ph),
                "w": int(pw),
                "mask": mask,
                "patch": patch,
            }
        )

    assert sprites
    expected_sprite_count = sum(len(frame["hole_rects"]) for frame in frames)
    assert len(sprites) == expected_sprite_count

    assigned: dict[tuple[int, int], list[dict]] = {}
    for side_idx, side_frames in enumerate([left_frames, right_frames]):
        for frame_idx in range(len(side_frames)):
            assigned[(int(side_idx), int(frame_idx))] = []

    for sp in sprites:
        pr0, pc0, pr1, pc1 = sp["bbox"]
        side_idx = 0 if 0.5 * (pc0 + pc1) < split_col else 1
        side_frames = left_frames if side_idx == 0 else right_frames
        best_score: tuple[int, int, int] | None = None
        best_idx: int | None = None
        for frame_idx, frame in enumerate(side_frames):
            fr0, _fc0, fr1, _fc1 = map(int, frame["bbox"])
            overlap = max(0, min(pr1, fr1) - max(pr0, fr0) + 1)
            band_dist = abs((pr0 + pr1) - (fr0 + fr1))
            score = (int(overlap), -int(band_dist), -int(sp["h"] * sp["w"]))
            if best_score is None or score > best_score:
                best_score = score
                best_idx = frame_idx
        assert best_idx is not None
        assert best_score is not None and best_score[0] > 0
        assigned[(int(side_idx), int(best_idx))].append(sp)

    for side_idx, side_frames in enumerate([left_frames, right_frames]):
        for frame_idx, frame in enumerate(side_frames):
            rects = frame["hole_rects"]
            pieces = assigned[(int(side_idx), int(frame_idx))]
            assert len(pieces) == len(rects)
            by_size: dict[tuple[int, int], dict] = {}
            for sp in pieces:
                size = (int(sp["h"]), int(sp["w"]))
                assert size not in by_size
                by_size[size] = sp
            fr0, fc0, _fr1, _fc1 = map(int, frame["bbox"])
            for rr0, cc0, h, w in rects:
                sp = by_size[(int(h), int(w))]
                paste(
                    out,
                    sp["patch"],
                    int(fr0 - ur0 + rr0),
                    int(fc0 - uc0 + cc0),
                    mask=sp["mask"],
                    overwrite=True,
                )

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 15, 25
FRAME_COUNT_CHOICES = [2, 3]
FRAME_SIZE_CHOICES = [6, 7]
FRAME_GAP = 2
CENTER_GAP = 3
SIDE_MARGIN_MIN = 4
SIDE_MARGIN_MAX = 5
BICHROMATIC_FRAC = 0.4
MAX_RETRIES = 500

RECT_LIBRARY = [
    (1, 1),
    (1, 2),
    (2, 1),
    (1, 3),
    (3, 1),
    (2, 2),
    (2, 3),
    (3, 2),
]


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    """Fill an (h, w) rectangle with 1 or 2 colors using a visible non-coding pattern."""
    out = np.full((h, w), int(cols[0]), dtype=int)
    if len(cols) == 1:
        return out

    alt = int(cols[1])
    rr, cc = np.indices((h, w))
    pattern = random.choice(["checker", "dots", "split"])
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = int(alt)
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = int(alt)
    elif h > 1 and (w == 1 or random.random() < 0.5):
        out[random.randint(1, h - 1) :, :] = int(alt)
    else:
        out[:, random.randint(1, w - 1) :] = int(alt)
    return out


def _pack_rectangles_in_frame(size: int, dims: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    """Pack axis-aligned rectangles into the inside of one frame with a 1-cell gap."""
    inner = int(size - 2)
    ordered = sorted([(int(h), int(w)) for h, w in dims], key=lambda x: (-x[0] * x[1], -x[0], -x[1]))
    for _ in range(150):
        occupied = np.zeros((inner, inner), dtype=bool)
        rects: list[tuple[int, int, int, int]] = []
        ok = True
        for h, w in ordered:
            candidates: list[tuple[int, int]] = []
            for r in range(inner - h + 1):
                for c in range(inner - w + 1):
                    rr0 = max(0, r - 1)
                    cc0 = max(0, c - 1)
                    rr1 = min(inner, r + h + 1)
                    cc1 = min(inner, c + w + 1)
                    if bool(occupied[rr0:rr1, cc0:cc1].any()):
                        continue
                    candidates.append((int(r), int(c)))
            if not candidates:
                ok = False
                break
            r, c = random.choice(candidates)
            occupied[r : r + h, c : c + w] = True
            rects.append((int(r + 1), int(c + 1), int(h), int(w)))
        if ok:
            return rects
    return None


def _build_frame_layouts(
    frame_count: int, size: int
) -> tuple[list[list[tuple[int, int, int, int]]], list[list[tuple[int, int, int, int]]], tuple[int, int], list[tuple[int, int]]] | None:
    """
    Build paired left/right frame layouts.

    Every frame gets:
    - one common size shared by all frames in both templates
    - one row-specific size shared by the mirrored left/right frame pair
    """
    library = [(h, w) for h, w in RECT_LIBRARY if h <= size - 2 and w <= size - 2]
    common_pool = [d for d in library if d[0] * d[1] >= 2]
    if not common_pool:
        return None

    for _ in range(300):
        common_dim = random.choice(common_pool)
        row_pool = [d for d in library if d != common_dim]
        if len(row_pool) < frame_count:
            return None
        row_dims = random.sample(row_pool, frame_count)
        left_layouts: list[list[tuple[int, int, int, int]]] = []
        right_layouts: list[list[tuple[int, int, int, int]]] = []
        ok = True
        for row_dim in row_dims:
            dims = [common_dim, row_dim]
            packed_left = _pack_rectangles_in_frame(size, dims)
            packed_right = _pack_rectangles_in_frame(size, dims)
            if packed_left is None or packed_right is None:
                ok = False
                break
            left_layouts.append(packed_left)
            right_layouts.append(packed_right)
        if ok:
            return left_layouts, right_layouts, common_dim, [(int(h), int(w)) for h, w in row_dims]
    return None


def _band_location(
    occupied: np.ndarray,
    h: int,
    w: int,
    row_lo: int,
    row_hi: int,
    col_lo: int,
    col_hi: int,
    *,
    padding: int = 1,
) -> tuple[int, int] | None:
    """Find a non-overlapping location inside one side-band."""
    H, W = occupied.shape
    row_lo = max(0, int(row_lo))
    col_lo = max(0, int(col_lo))
    row_hi = min(int(row_hi), int(H - h))
    col_hi = min(int(col_hi), int(W - w))
    if row_lo > row_hi or col_lo > col_hi:
        return None

    candidates: list[tuple[int, int]] = []
    for r in range(row_lo, row_hi + 1):
        for c in range(col_lo, col_hi + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(H, r + h + padding)
            cc1 = min(W, c + w + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                candidates.append((int(r), int(c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_side_sprites(
    inp: np.ndarray,
    occupied: np.ndarray,
    patches_by_frame: list[list[np.ndarray]],
    frame_bboxes: list[tuple[int, int, int, int]],
    arrangement_bbox: tuple[int, int, int, int],
    side: str,
) -> bool:
    """Place all sprites for one template on its owning side, aligned to the correct frame rows."""
    H, W = inp.shape
    _ar0, ac0, _ar1, ac1 = map(int, arrangement_bbox)
    frame_order = list(range(len(frame_bboxes)))
    random.shuffle(frame_order)

    trial_occ = occupied.copy()
    placed: list[tuple[np.ndarray, int, int]] = []
    for frame_idx in frame_order:
        fr0, _fc0, fr1, _fc1 = map(int, frame_bboxes[frame_idx])
        order = list(range(len(patches_by_frame[frame_idx])))
        random.shuffle(order)
        order.sort(key=lambda i: int(patches_by_frame[frame_idx][i].shape[0] * patches_by_frame[frame_idx][i].shape[1]), reverse=True)
        for patch_idx in order:
            patch = patches_by_frame[frame_idx][patch_idx]
            h, w = map(int, patch.shape)
            if side == "left":
                loc = _band_location(trial_occ, h, w, fr0, fr1 - h + 1, 0, ac0 - w - 1)
            else:
                loc = _band_location(trial_occ, h, w, fr0, fr1 - h + 1, ac1 + 2, W - w)
            if loc is None:
                return False
            r, c = loc
            trial_occ[r : r + h, c : c + w] = True
            placed.append((patch, int(r), int(c)))

    for patch, r, c in placed:
        paste(inp, patch, int(r), int(c), overwrite=True)
        occupied[r : r + patch.shape[0], c : c + patch.shape[1]] = True
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    random.shuffle(FRAME_SIZE_CHOICES)

    for frame_size in FRAME_SIZE_CHOICES:
        out_h = int(frame_count * frame_size + (frame_count - 1) * FRAME_GAP)
        out_w = int(2 * frame_size + CENTER_GAP)
        side_margin_cap = (GRID_MAX - out_w) // 2
        if out_h > GRID_MAX or side_margin_cap < SIDE_MARGIN_MIN:
            continue

        side_margin = random.randint(SIDE_MARGIN_MIN, min(SIDE_MARGIN_MAX, side_margin_cap))
        W = int(out_w + 2 * side_margin)
        H = max(GRID_MIN, out_h)
        if H > GRID_MAX or W > GRID_MAX:
            continue
        tr0 = int((H - out_h) // 2)
        tc0 = int(side_margin)

        built = _build_frame_layouts(frame_count, frame_size)
        if built is None:
            continue
        left_layouts, right_layouts, _common_dim, _row_dims = built

        bg, base = random.sample(range(10), 2)
        available = [c for c in range(10) if c not in (bg, base)]
        random.shuffle(available)

        expected = np.full((out_h, out_w), int(bg), dtype=int)
        inp = np.full((H, W), int(bg), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        occupied[tr0 : tr0 + out_h, tc0 : tc0 + out_w] = True

        left_global_bboxes: list[tuple[int, int, int, int]] = []
        right_global_bboxes: list[tuple[int, int, int, int]] = []
        left_patches: list[list[np.ndarray]] = []
        right_patches: list[list[np.ndarray]] = []
        primary_cursor = 0
        dim_records: dict[tuple[int, int], list[np.ndarray]] = {}

        for side_idx, layouts in enumerate([left_layouts, right_layouts]):
            local_c0 = 0 if side_idx == 0 else frame_size + CENTER_GAP
            for frame_idx, rects in enumerate(layouts):
                local_r0 = int(frame_idx * (frame_size + FRAME_GAP))
                gr0 = int(tr0 + local_r0)
                gc0 = int(tc0 + local_c0)
                bbox = (int(gr0), int(gc0), int(gr0 + frame_size - 1), int(gc0 + frame_size - 1))
                if side_idx == 0:
                    left_global_bboxes.append(bbox)
                else:
                    right_global_bboxes.append(bbox)

                expected[local_r0 : local_r0 + frame_size, local_c0 : local_c0 + frame_size] = int(base)
                inp[gr0 : gr0 + frame_size, gc0 : gc0 + frame_size] = int(base)

                frame_patches: list[np.ndarray] = []
                for r, c, h, w in rects:
                    primary = int(available[primary_cursor % len(available)])
                    primary_cursor += 1
                    cols = [primary]
                    if h * w > 1 and random.random() < BICHROMATIC_FRAC:
                        second_choices = [x for x in available if x != primary]
                        cols.append(int(random.choice(second_choices)))
                    patch = _make_patch(int(h), int(w), cols)
                    frame_patches.append(patch)
                    dim_records.setdefault((int(h), int(w)), []).append(patch.copy())
                    paste(expected, patch, int(local_r0 + r), int(local_c0 + c), overwrite=True)
                    inp[gr0 + r : gr0 + r + h, gc0 + c : gc0 + c + w] = int(bg)
                if side_idx == 0:
                    left_patches.append(frame_patches)
                else:
                    right_patches.append(frame_patches)

        # Keep repeated sizes visibly decorative rather than exact duplicates whenever possible.
        changed = False
        for dim, patches in list(dim_records.items()):
            if len(patches) < 2:
                continue
            first = patches[0]
            if all(np.array_equal(first, patch) for patch in patches[1:]):
                h, w = dim
                alt_primary = int(available[(primary_cursor + h + w) % len(available)])
                replacement = _make_patch(int(h), int(w), [alt_primary])
                for side_idx, patch_groups in enumerate([left_patches, right_patches]):
                    for frame_idx, group in enumerate(patch_groups):
                        for patch_idx, patch in enumerate(group):
                            if patch.shape == replacement.shape and np.array_equal(patch, first):
                                group[patch_idx] = replacement.copy()
                                local_c0 = 0 if side_idx == 0 else frame_size + CENTER_GAP
                                local_r0 = int(frame_idx * (frame_size + FRAME_GAP))
                                rr, cc, hh, ww = (
                                    left_layouts[frame_idx][patch_idx]
                                    if side_idx == 0
                                    else right_layouts[frame_idx][patch_idx]
                                )
                                paste(expected, replacement, int(local_r0 + rr), int(local_c0 + cc), overwrite=True)
                                changed = True
                                break
                        if changed:
                            break
                    if changed:
                        break
            if changed:
                break

        arrangement_bbox = (int(tr0), int(tc0), int(tr0 + out_h - 1), int(tc0 + out_w - 1))
        if not _place_side_sprites(inp, occupied, left_patches, left_global_bboxes, arrangement_bbox, "left"):
            continue
        if not _place_side_sprites(inp, occupied, right_patches, right_global_bboxes, arrangement_bbox, "right"):
            continue

        counts = {int(col): int((inp == int(col)).sum()) for col in colors_present(inp, exclude={bg})}
        if any(col != int(base) and counts[col] >= counts[int(base)] for col in counts):
            continue

        try:
            solved = main(inp)
        except Exception:
            continue
        if not np.array_equal(solved, expected):
            continue
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
