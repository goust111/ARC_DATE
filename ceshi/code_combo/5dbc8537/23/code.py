# Combo variant of 5dbc8537 (index 23)
# combo_spec: D3/0+D8/0+D12/0
# Name: 多框架少量大精灵颜色签名唯一匹配

"""
Design analysis for the combo variant
=====================================

a) Concrete behavior and flow trace
-----------------------------------
- Original 5dbc8537 uses one long base-colored strip, one global hole region, and a DFS exact-cover search over
  many small scattered rectangles.
- This combo keeps the same local mechanic, "outside filled rectangle goes back into a same-size rectangular hole",
  but changes three interacting parts at once:
  1. the destination is now 2-3 disconnected square frames instead of one strip;
  2. there are only 3-5 sprites, and each sprite is much larger;
  3. every sprite has a globally unique color signature, so pieces are visually distinct and no global backtracking
     over interchangeable pieces is needed.
- Concrete small trace:
  original flow:
  1. find one dominant base-color component;
  2. crop one bbox around it;
  3. treat every non-base cell inside that bbox as one shared cavity mass;
  4. DFS place many rectangles into that mass.

  combo flow:
  1. find the shared base color by total base area across several disconnected frame components;
  2. split that base color into individual frames and extract each frame's own rectangular bg holes;
  3. assign each outside sprite to a frame using row/column band alignment;
  4. inside each frame, place sprites by direct size match only.

  difference:
  - the original solver can ignore ownership because there is only one template;
  - the combo must first recover which frame a sprite belongs to;
  - the original needs search because many pieces can compete for many slots;
  - the combo is deterministic because there are few pieces, sizes are unique inside a frame, and the pieces are
    visually non-interchangeable due to unique color signatures.

b) Complexity calibration
-------------------------
- Frame count is forced to 2 or 3, so the task cannot collapse back to the single-template original.
- Total sprite count is forced to 3-5, so the "few large pieces" behavior is visible in every sample.
- Frames 0 and 1 always share at least one repeated hole size across different frames. That makes the extra
  frame-grouping stage logically necessary instead of being cosmetic.
- Within one frame, hole sizes are unique. This keeps the local step deterministic and prevents the variant from
  becoming an underconstrained "guess which equal hole gets which color pattern" puzzle.
- Sprite patches are 1-2 colors only, but each sprite's color signature is globally unique. This preserves the
  source-task look while removing indistinguishable duplicates.

c) Implicit-assumption audit of the original main()
---------------------------------------------------
- Original `base_coords -> one bbox -> hole_mask = dst != base`:
  broken, because the union bbox now contains intentional bg gaps between separate frames.
- Original "all holes live in one connected destination":
  broken, because holes are partitioned by frame.
- Original "sum(sprite area) == one global hole area":
  no longer the right abstraction. The correct invariant is per-frame rectangle matching after frame assignment.
- Original DFS exact cover:
  unnecessary and misleading here. The combo generator guarantees per-frame unique sizes, so deterministic placement
  is the correct solver behavior.
- New solver assumptions introduced here:
  1. every base-colored 8-connected component is a real frame;
  2. each sprite overlaps exactly one frame band;
  3. each frame's hole components are rectangles and have unique sizes inside that frame;
  4. sprite color signatures are globally unique.
  The generator enforces all four conditions explicitly.

d) Placement legality
---------------------
- Frames are separated by a 2-cell bg gap and never overlap.
- Each hole stays strictly inside a 1-cell frame border.
- Holes inside one frame are separated by at least one base cell.
- Sprites are scattered only outside the arrangement bbox, with 1-cell padding from other sprites and from frames.
- Every accepted sample is oracle-checked by running `main(input)` and comparing against the constructed output.

e) Visual observability
-----------------------
- Multiple disconnected destination frames are immediately visible.
- With only 3-5 large sprites, the puzzle reads as "route these few blocks back to their frames" rather than as
  dense packing noise.
- Repeated sizes across different frames prevent a human from explaining the sample as one global cavity again.
- Unique color signatures make the source pieces look intentionally distinct instead of like interchangeable clones.

f) Visual cleanliness
---------------------
- The arrangement is a single row or a single column of frames, so the crop remains easy to read.
- Sprite scattering is restricted to the aligned outer band of the owning frame, avoiding clutter in the gaps between
  frames and keeping frame affiliation visually obvious.
- Because there are only a few pieces and each one is large, the input remains legible even when bichromatic patterns
  are used.

g) Change inferability
----------------------
- No hidden use of color ids is required for the rule. Colors serve only as visible piece identity.
- The actual placement rule remains geometric and human-readable: identify the frame, then match the rectangle size.
- Color-signature uniqueness matters by preventing source-side ambiguity; it does not create an invisible ordering
  rule that a human would have to guess from raw ids.

h) Solver/generator contract
----------------------------
- Solver side:
  detect the shared base color, split frames, find local rectangular holes, group sprites by frame band, then fill
  each frame by size.
- Generator side:
  keep frame count >= 2, sprite count in [3, 5], sprite sizes large, repeated sizes across different frames, unique
  signatures across all sprites, and enough base area that the shared base color is recoverable.
"""

from __future__ import annotations

import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, crop_bbox, find_components, get_bg, paste


Rect = tuple[int, int, int, int]


def _color_signature(patch: np.ndarray) -> tuple[tuple[int, int], ...]:
    vals, counts = np.unique(np.array(patch, dtype=int), return_counts=True)
    pairs = [(int(v), int(c)) for v, c in zip(vals.tolist(), counts.tolist())]
    pairs.sort(key=lambda x: (-x[1], x[0]))
    return tuple(pairs)


def _sprite_key(sp: dict) -> tuple[tuple[tuple[int, int], ...], int, int, tuple[int, int, int, int]]:
    return (
        sp["signature"],
        int(sp["h"]),
        int(sp["w"]),
        tuple(map(int, sp["bbox"])),
    )


def _detect_base_color(g: np.ndarray, bg: int) -> int:
    candidates: list[tuple[int, int, int]] = []
    for col in colors_present(g, exclude={bg}):
        comps = components_mask(g == int(col), connectivity=8)
        if not comps:
            continue
        total_area = sum(int(coords.shape[0]) for coords in comps)
        max_area = max(int(coords.shape[0]) for coords in comps)
        candidates.append((int(total_area), int(max_area), int(col)))
    assert candidates
    candidates.sort(reverse=True)
    if len(candidates) > 1:
        assert candidates[0][0] > candidates[1][0]
    return int(candidates[0][2])


def _frame_holes_from_patch(frame_patch: np.ndarray, bg: int) -> list[Rect]:
    hole_rects: list[Rect] = []
    seen_sizes: set[tuple[int, int]] = set()
    for hole in components_mask(frame_patch == int(bg), connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(hole)
        h = int(hr1 - hr0 + 1)
        w = int(hc1 - hc0 + 1)
        assert int(hole.shape[0]) == int(h * w)
        size = (int(h), int(w))
        assert size not in seen_sizes
        seen_sizes.add(size)
        hole_rects.append((int(hr0), int(hc0), int(h), int(w)))
    hole_rects.sort(key=lambda x: (x[0], x[1], x[2], x[3]))
    assert hole_rects
    return hole_rects


def _detect_frames(g: np.ndarray, bg: int, base_col: int) -> tuple[list[dict], tuple[int, int, int, int]]:
    frame_coords = components_mask(g == int(base_col), connectivity=8)
    assert len(frame_coords) >= 2

    all_base = np.argwhere(g == int(base_col))
    ur0, uc0, ur1, uc1 = bbox_coords(all_base)

    frames: list[dict] = []
    for coords in frame_coords:
        fr0, fc0, fr1, fc1 = bbox_coords(coords)
        patch = crop_bbox(g, (int(fr0), int(fc0), int(fr1), int(fc1)))
        base_mask = patch == int(base_col)
        hole_rects = _frame_holes_from_patch(patch, bg)
        frames.append(
            {
                "bbox": (int(fr0), int(fc0), int(fr1), int(fc1)),
                "base_mask": base_mask,
                "hole_rects": hole_rects,
            }
        )
    return frames, (int(ur0), int(uc0), int(ur1), int(uc1))


def _arrangement_is_horizontal(frames: list[dict]) -> bool:
    row_spread = max(int(f["bbox"][0]) for f in frames) - min(int(f["bbox"][0]) for f in frames)
    col_spread = max(int(f["bbox"][1]) for f in frames) - min(int(f["bbox"][1]) for f in frames)
    return bool(col_spread >= row_spread)


def _extract_sprites(g: np.ndarray, bg: int, base_col: int) -> list[dict]:
    masked = np.array(g, dtype=int).copy()
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
                "signature": _color_signature(patch),
            }
        )

    assert sprites
    signatures = [sp["signature"] for sp in sprites]
    assert len(set(signatures)) == len(signatures)
    sprites.sort(key=_sprite_key)
    return sprites


def _frame_score(sprite_bbox: tuple[int, int, int, int], frame_bbox: tuple[int, int, int, int], horizontal: bool) -> tuple[int, int, int]:
    pr0, pc0, pr1, pc1 = map(int, sprite_bbox)
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    if horizontal:
        overlap = max(0, min(pc1, fc1) - max(pc0, fc0) + 1)
        band_dist = abs((pc0 + pc1) - (fc0 + fc1))
        orth_dist = min(abs(pr1 - fr0), abs(pr0 - fr1))
    else:
        overlap = max(0, min(pr1, fr1) - max(pr0, fr0) + 1)
        band_dist = abs((pr0 + pr1) - (fr0 + fr1))
        orth_dist = min(abs(pc1 - fc0), abs(pc0 - fc1))
    return int(overlap), -int(band_dist), -int(orth_dist)


def _assign_sprites_to_frames(sprites: list[dict], frames: list[dict], horizontal: bool) -> dict[int, list[dict]]:
    assigned: dict[int, list[dict]] = {i: [] for i in range(len(frames))}
    for sp in sorted(sprites, key=_sprite_key):
        best_idx: int | None = None
        best_score: tuple[int, int, int] | None = None
        for i, frame in enumerate(frames):
            score = _frame_score(sp["bbox"], frame["bbox"], horizontal)
            if best_score is None or score > best_score:
                best_score = score
                best_idx = i
        assert best_idx is not None
        assert best_score is not None and best_score[0] > 0
        assigned[best_idx].append(sp)
    for i, frame in enumerate(frames):
        assert len(assigned[i]) == len(frame["hole_rects"])
        assigned[i].sort(key=_sprite_key)
    return assigned


def _analyze_input(grid: np.ndarray) -> dict[str, int]:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col = _detect_base_color(g, bg)
    frames, _ = _detect_frames(g, bg, base_col)
    horizontal = _arrangement_is_horizontal(frames)
    frames = sorted(frames, key=lambda f: int(f["bbox"][1] if horizontal else f["bbox"][0]))
    sprites = _extract_sprites(g, bg, base_col)

    size_to_frames: dict[tuple[int, int], set[int]] = {}
    max_area = 0
    for fi, frame in enumerate(frames):
        for _r, _c, h, w in frame["hole_rects"]:
            size_to_frames.setdefault((int(h), int(w)), set()).add(int(fi))
            max_area = max(max_area, int(h * w))

    repeated_cross_frame = sum(1 for owners in size_to_frames.values() if len(owners) >= 2)
    return {
        "frame_count": int(len(frames)),
        "sprite_count": int(len(sprites)),
        "max_sprite_area": int(max(int(sp["h"] * sp["w"]) for sp in sprites)),
        "max_hole_area": int(max_area),
        "repeated_cross_frame_sizes": int(repeated_cross_frame),
        "unique_signatures": int(len({sp["signature"] for sp in sprites}) == len(sprites)),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill several disconnected frame templates by routing each large outside sprite to its owning frame and local hole.

    Input structure:
    - bg is the most common color. One shared base color forms 2-3 disconnected square frames, each containing one or two rectangular bg holes.
    - Only 3-5 large filled sprites appear outside the frame arrangement, and each sprite has a globally unique 1-color or 2-color signature.

    Rule:
    - Recover the shared base color, split it into individual frame components, and crop the union bbox containing the whole arrangement.
    - Assign each outside sprite to the frame whose row/column band it overlaps, then match the sprite bbox size to one rectangular hole inside that frame.
    - Paste every sprite into its matching hole and keep bg only in the intentional gaps between different frames.

    Key insight: This is no longer one global exact-cover search; the real work is first finding frame ownership, after which each frame is solved deterministically by direct rectangle-size matching among a few visually distinct large pieces.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    base_col = _detect_base_color(g, bg)

    frames, union_bbox = _detect_frames(g, bg, base_col)
    horizontal = _arrangement_is_horizontal(frames)
    frames = sorted(frames, key=lambda f: int(f["bbox"][1] if horizontal else f["bbox"][0]))

    ur0, uc0, ur1, uc1 = map(int, union_bbox)
    out = np.full((int(ur1 - ur0 + 1), int(uc1 - uc0 + 1)), int(bg), dtype=int)

    for frame in frames:
        fr0, fc0, fr1, fc1 = map(int, frame["bbox"])
        base_patch = np.full((int(fr1 - fr0 + 1), int(fc1 - fc0 + 1)), int(base_col), dtype=int)
        paste(out, base_patch, int(fr0 - ur0), int(fc0 - uc0), mask=frame["base_mask"], overwrite=True)

    sprites = _extract_sprites(g, bg, base_col)
    assigned = _assign_sprites_to_frames(sprites, frames, horizontal)

    for i, frame in enumerate(frames):
        pieces = assigned[i]
        by_size: dict[tuple[int, int], dict] = {}
        for sp in pieces:
            size = (int(sp["h"]), int(sp["w"]))
            assert size not in by_size
            by_size[size] = sp

        fr0, fc0, _fr1, _fc1 = map(int, frame["bbox"])
        for rr0, cc0, h, w in frame["hole_rects"]:
            sp = by_size[(int(h), int(w))]
            paste(out, sp["patch"], int(fr0 - ur0 + rr0), int(fc0 - uc0 + cc0), mask=sp["mask"], overwrite=True)

    return out


# --------------- generator ---------------

GRID_MIN, GRID_MAX = 20, 30
FRAME_SIZE_CHOICES = [8, 9, 10]
FRAME_GAP = 2
HOLE_COUNT_BY_TOTAL = {
    2: {
        3: [[1, 2], [2, 1]],
        4: [[2, 2]],
    },
    3: {
        3: [[1, 1, 1]],
        4: [[1, 1, 2], [1, 2, 1], [2, 1, 1]],
        5: [[1, 2, 2], [2, 1, 2], [2, 2, 1]],
    },
}
FRAME_COUNT_WEIGHTS = {2: 1, 3: 1}
SPRITE_TOTAL_WEIGHTS = {
    2: {3: 1, 4: 1},
    3: {3: 1, 4: 2, 5: 3},
}
BICHROMATIC_FRAC = 0.45
MAX_RETRIES = 500


def _make_patch(h: int, w: int, cols: list[int]) -> np.ndarray:
    out = np.full((h, w), int(cols[0]), dtype=int)
    if len(cols) == 1:
        return out

    b = int(cols[1])
    rr, cc = np.indices((h, w))
    pattern = random.choice(["checker", "dots", "split"])
    if pattern == "checker":
        out[(rr + cc + random.randint(0, 1)) % 2 == 0] = b
    elif pattern == "dots":
        step = random.choice([2, 3])
        mask = (rr * 2 + cc + random.randint(0, step - 1)) % step == 0
        if not mask.any() or mask.all():
            mask = np.zeros((h, w), dtype=bool)
            mask[random.randrange(h), random.randrange(w)] = True
        out[mask] = b
    elif h > 1 and (w == 1 or random.random() < 0.5):
        split = random.randint(1, h - 1)
        out[split:, :] = b
    else:
        split = random.randint(1, w - 1)
        out[:, split:] = b
    return out


def _candidate_dims(size: int) -> list[tuple[int, int]]:
    inner = int(size - 2)
    dims: list[tuple[int, int]] = []
    for h in range(2, inner + 1):
        for w in range(2, inner + 1):
            area = int(h * w)
            if area < max(8, inner + 1):
                continue
            if area >= inner * inner:
                continue
            dims.append((int(h), int(w)))
    dims.sort(key=lambda x: (-x[0] * x[1], -x[0], -x[1]))
    return dims


def _choose_hole_counts() -> tuple[int, list[int]]:
    frame_counts = sorted(HOLE_COUNT_BY_TOTAL)
    frame_count = random.choices(frame_counts, weights=[FRAME_COUNT_WEIGHTS[n] for n in frame_counts], k=1)[0]
    total_weights = SPRITE_TOTAL_WEIGHTS[int(frame_count)]
    totals = sorted(HOLE_COUNT_BY_TOTAL[int(frame_count)])
    sprite_total = random.choices(totals, weights=[total_weights[t] for t in totals], k=1)[0]
    patterns = HOLE_COUNT_BY_TOTAL[int(frame_count)][int(sprite_total)]
    return int(frame_count), list(random.choice(patterns))


def _frame_geometry(frame_count: int, frame_size: int, horizontal: bool, sprite_total: int) -> tuple[int, int, int, int, int] | None:
    out_h = int(frame_size if horizontal else frame_count * frame_size + (frame_count - 1) * FRAME_GAP)
    out_w = int(frame_count * frame_size + (frame_count - 1) * FRAME_GAP if horizontal else frame_size)
    margin = int(frame_size + min(2, max(0, sprite_total - 3)))
    min_h = max(GRID_MIN, out_h + 2 * margin if horizontal else out_h)
    min_w = max(GRID_MIN, out_w if horizontal else out_w + 2 * margin)
    if min_h > GRID_MAX or min_w > GRID_MAX:
        return None
    return int(out_h), int(out_w), int(margin), int(min_h), int(min_w)


def _pack_rectangles_in_frame(size: int, dims: list[tuple[int, int]]) -> list[Rect] | None:
    inner = int(size - 2)
    ordered = sorted([(int(h), int(w)) for h, w in dims], key=lambda x: (-x[0] * x[1], -x[0], -x[1]))

    for _ in range(80):
        occupied = np.zeros((inner, inner), dtype=bool)
        placed: list[Rect] = []

        def dfs(i: int) -> bool:
            if i == len(ordered):
                return True
            h, w = ordered[i]
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
            random.shuffle(candidates)
            for r, c in candidates:
                occupied[r : r + h, c : c + w] = True
                placed.append((int(r + 1), int(c + 1), int(h), int(w)))
                if dfs(i + 1):
                    return True
                placed.pop()
                occupied[r : r + h, c : c + w] = False
            return False

        if dfs(0):
            return placed
    return None


def _build_frame_layouts(frame_count: int, size: int, hole_counts: list[int]) -> tuple[list[list[Rect]], tuple[int, int]] | None:
    library = _candidate_dims(size)
    shared_pool = [d for d in library if d[0] * d[1] >= max(10, size)]
    if not shared_pool:
        return None

    for _ in range(250):
        shared_dim = random.choice(shared_pool)
        layouts: list[list[Rect]] = []
        ok = True
        for fi in range(frame_count):
            need = int(hole_counts[fi])
            assert 1 <= need <= 2
            packed = None
            for _ in range(160):
                dims = [shared_dim] if fi < 2 else []
                if need < len(dims):
                    break
                while len(dims) < need:
                    pool = [d for d in library if d not in dims]
                    if not pool:
                        dims = []
                        break
                    dims.append(random.choice(pool))
                if len(dims) != need:
                    continue
                if len(set(dims)) != len(dims):
                    continue
                packed = _pack_rectangles_in_frame(size, dims)
                if packed is not None:
                    break
            if packed is None:
                ok = False
                break
            layouts.append(packed)
        if ok:
            return layouts, shared_dim
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
    H, W = occupied.shape
    row_lo = int(max(0, row_lo))
    col_lo = int(max(0, col_lo))
    row_hi = int(min(H - h, row_hi))
    col_hi = int(min(W - w, col_hi))
    if row_lo > row_hi or col_lo > col_hi:
        return None
    candidates: list[tuple[int, int]] = []
    for r in range(int(row_lo), int(row_hi) + 1):
        for c in range(int(col_lo), int(col_hi) + 1):
            rr0 = max(0, r - padding)
            cc0 = max(0, c - padding)
            rr1 = min(H, r + h + padding)
            cc1 = min(W, c + w + padding)
            if not bool(occupied[rr0:rr1, cc0:cc1].any()):
                candidates.append((int(r), int(c)))
    if not candidates:
        return None
    return random.choice(candidates)


def _place_frame_sprites(
    inp: np.ndarray,
    occupied: np.ndarray,
    patches: list[np.ndarray],
    frame_bbox: tuple[int, int, int, int],
    arrangement_bbox: tuple[int, int, int, int],
    horizontal: bool,
) -> bool:
    H, W = inp.shape
    fr0, fc0, fr1, fc1 = map(int, frame_bbox)
    ar0, ac0, ar1, ac1 = map(int, arrangement_bbox)

    def try_side(side: str) -> bool:
        trial_occ = occupied.copy()
        placed: list[tuple[np.ndarray, int, int]] = []
        order = list(range(len(patches)))
        random.shuffle(order)
        order.sort(key=lambda i: int(patches[i].shape[0] * patches[i].shape[1]), reverse=True)
        for idx in order:
            patch = patches[idx]
            h, w = map(int, patch.shape)
            if horizontal and side == "top":
                loc = _band_location(trial_occ, h, w, 0, ar0 - h - 1, fc0, fc1 - w + 1)
            elif horizontal and side == "bottom":
                loc = _band_location(trial_occ, h, w, ar1 + 2, H - h, fc0, fc1 - w + 1)
            elif (not horizontal) and side == "left":
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

    sides = ["top", "bottom"] if horizontal else ["left", "right"]
    random.shuffle(sides)
    for side in sides:
        if try_side(side):
            return True
    return False


def _make_patches(rects: list[Rect], bg: int, base: int) -> list[np.ndarray] | None:
    available = [c for c in range(10) if c not in (bg, base)]
    random.shuffle(available)
    if len(rects) > len(available):
        return None

    patches: list[np.ndarray] = []
    signatures: set[tuple[tuple[int, int], ...]] = set()
    for i, (_r, _c, h, w) in enumerate(rects):
        primary = int(available[i])
        cols = [primary]
        if h * w >= 10 and random.random() < BICHROMATIC_FRAC:
            secondary_choices = [c for c in available if c != primary]
            cols.append(int(random.choice(secondary_choices)))
        patch = _make_patch(int(h), int(w), cols)
        sig = _color_signature(patch)
        if sig in signatures:
            return None
        signatures.add(sig)
        patches.append(patch)

    assert len(signatures) == len(patches)
    return patches


def _gen_attempt() -> np.ndarray | None:
    horizontal = random.random() < 0.5
    frame_count, hole_counts = _choose_hole_counts()
    sprite_total = int(sum(hole_counts))

    size_choices = [size for size in FRAME_SIZE_CHOICES if _frame_geometry(frame_count, size, horizontal, sprite_total) is not None]
    if not size_choices:
        return None
    random.shuffle(size_choices)
    for frame_size in size_choices:
        geometry = _frame_geometry(frame_count, frame_size, horizontal, sprite_total)
        if geometry is None:
            continue
        out_h, out_w, margin, min_h, min_w = geometry

        built = _build_frame_layouts(frame_count, frame_size, hole_counts)
        if built is None:
            continue
        frame_rects, _shared_dim = built

        H = random.randint(min_h, GRID_MAX)
        W = random.randint(min_w, GRID_MAX)
        if horizontal:
            tr0 = random.randint(margin, H - out_h - margin)
            tc0 = random.randint(0, W - out_w)
        else:
            tr0 = random.randint(0, H - out_h)
            tc0 = random.randint(margin, W - out_w - margin)

        bg, base = random.sample(range(10), 2)
        all_rects = [rect for rects in frame_rects for rect in rects]
        patches = _make_patches(all_rects, bg, base)
        if patches is None:
            continue

        expected = np.full((out_h, out_w), int(bg), dtype=int)
        inp = np.full((H, W), int(bg), dtype=int)
        occupied = np.zeros((H, W), dtype=bool)
        occupied[tr0 : tr0 + out_h, tc0 : tc0 + out_w] = True

        frame_bboxes_global: list[tuple[int, int, int, int]] = []
        patches_by_frame: list[list[np.ndarray]] = []
        cursor = 0

        for fi, rects in enumerate(frame_rects):
            local_r0 = 0 if horizontal else fi * (frame_size + FRAME_GAP)
            local_c0 = fi * (frame_size + FRAME_GAP) if horizontal else 0
            gr0 = tr0 + local_r0
            gc0 = tc0 + local_c0

            frame_bboxes_global.append((int(gr0), int(gc0), int(gr0 + frame_size - 1), int(gc0 + frame_size - 1)))
            expected[local_r0 : local_r0 + frame_size, local_c0 : local_c0 + frame_size] = int(base)
            inp[gr0 : gr0 + frame_size, gc0 : gc0 + frame_size] = int(base)

            local_patches: list[np.ndarray] = []
            for r, c, h, w in rects:
                patch = patches[cursor]
                cursor += 1
                local_patches.append(patch)
                paste(expected, patch, int(local_r0 + r), int(local_c0 + c), overwrite=True)
                inp[gr0 + r : gr0 + r + h, gc0 + c : gc0 + c + w] = int(bg)
            patches_by_frame.append(local_patches)

        arrangement_bbox = (int(tr0), int(tc0), int(tr0 + out_h - 1), int(tc0 + out_w - 1))
        for fi in range(frame_count):
            if not _place_frame_sprites(inp, occupied, patches_by_frame[fi], frame_bboxes_global[fi], arrangement_bbox, horizontal):
                return None

        size_to_frames: dict[tuple[int, int], set[int]] = {}
        for fi, rects in enumerate(frame_rects):
            for _r, _c, h, w in rects:
                size_to_frames.setdefault((int(h), int(w)), set()).add(int(fi))
        repeated_cross_frame = sum(1 for owners in size_to_frames.values() if len(owners) >= 2)
        signature_unique = int(len({_color_signature(p) for p in patches}) == len(patches))
        max_sprite_area = max(int(p.shape[0] * p.shape[1]) for p in patches)

        if frame_count < 2:
            return None
        if not (3 <= len(all_rects) <= 5):
            return None
        if max_sprite_area < 8:
            return None
        if repeated_cross_frame < 1:
            return None
        if signature_unique != 1:
            return None

        counts = {int(col): int((inp == int(col)).sum()) for col in colors_present(inp, exclude={bg})}
        if any(col != int(base) and counts[col] >= counts[int(base)] for col in counts):
            return None

        try:
            solved = main(inp)
        except Exception:
            return None
        if not np.array_equal(solved, expected):
            return None
        return inp

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid 5dbc8537 combo input")
