# Combo variant of 67e490f4 (index 25)
# combo_spec: D14/0+D15/2
# Name: 多色框架+大框架

"""
Design analysis for the combo variant
=====================================

Source-task grounding
- The base task is the original 67e490f4 shape-to-hole color transfer puzzle.
- This combo merges D14/0 (two-color frame material) with D15/2 (frame bbox enlarged so much that
  sprites are forced into a very narrow outer ring).

a) Concretization and behavior trace
- Original crop logic:
    2 2 2 2 2
    2 0 0 2 2
    2 2 2 2 2
  The old solver found one monochrome frame color `2`, defined holes by `!= 2`, and rendered a solid
  output base full of `2`.
- New combo crop logic:
    2 3 2 3 2
    3 0 0 2 3
    2 3 2 3 2
  Now the frame material is the pair `{2,3}`. True holes are cells that are neither 2 nor 3, so the
  old `sub != frame_col` rule would falsely treat every color-3 frame cell as a hole. The solver must
  first infer the color pair, preserve the visible pattern, and only recolor the actual voids.
- The D15 part changes the global layout: instead of a medium rectangle floating in 30x30, the frame is
  a near-full 24x24 template centered inside the grid. After enforcing a one-cell frame/sprite gap, the
  remaining legal sprite region becomes only a two-cell-thick perimeter ring. This physically constrains
  the sprite library to compact shapes that still fit in that ring.

b) Complexity calibration
- The combo must stay visibly different from the source task in only 2-3 examples. To make that obvious,
  the frame always uses exactly two colors with a simple repeating motif (`checker`, row stripes, or
  column stripes), and the frame is always very large.
- The sprite band is intentionally narrow, but not so narrow that generation becomes impossible. A fixed
  24x24 frame inside 30x30 leaves a three-cell border before padding and an effective two-cell usable
  ring after the mandatory one-cell separation. This still feels cramped while remaining packable.
- Shape freedom is reduced on purpose: only compact shapes with at least one dimension <= 2 are used.
  That gives enough variety for matching while avoiding impossible 3x3-like sprites in the narrow ring.

c) Implicit-assumption audit against the original `main()`
- `find_components(... monochromatic=True)` for frame detection assumed one frame color. That is broken by
  D14; the solver now searches over non-bg color pairs and scores the largest valid paired component.
- `hole_mask = (sub != frame_col)` assumed every non-frame cell in the bbox was a hole. With a two-color
  template, that would misclassify half the frame. The correct test is `~np.isin(sub, frame_cols)`.
- `out = np.full((th, tw), frame_col, ...)` assumed the output template was solid. That is also broken by
  D14; output must start from the cropped two-color pattern itself.
- The original generator assumed ample outside space for arbitrary small sprites. D15/2 removes that
  assumption, so sprite generation must use a narrower shape library and lower sprite counts.
- The solver now implicitly assumes the frame colors are not reused by sprites, that the real frame bbox
  is much larger than any sprite component, and that its bbox border is entirely covered by the paired
  frame colors. The generator enforces each of those conditions.

d) Placement legality
- Holes remain strictly inside the frame border and 4-disconnected from each other, so the two-color
  frame remains one large 8-connected object under the union of its colors.
- Sprites are pasted only after the frame is fixed in place. `random_free_location(..., padding=1)` keeps
  them separated from the frame and from each other by at least one background cell.
- Because the usable outer band is thin, the generator places larger sprite batches first and restricts
  sprite shapes to compact masks that fit within a two-cell-thick strip in some orientation.

e) Visual observability
- A human can see both changes immediately: the output preserves a two-color template rather than a solid
  slab, and that template nearly fills the whole 30x30 grid.
- The edge crowding is also visible in the input: small sprites hug the outer perimeter, while the large
  central template dominates the canvas. This gives repeated evidence that the outer scraps are still the
  color keys even though they are spatially compressed.

f) Visual cleanliness
- The frame occupies one clear central region, and the sprites live entirely outside it. There is no
  overlap or mixed layering inside a shared area.
- Hole fill colors exclude both frame colors, so frame material, holes, and sprites remain visually
  separable even with the dense large template.

g) Change inferability
- No hidden ordering by numeric color id is required for correctness. The generator guarantees a unique
  plurality sprite color for each shape, and the solver asserts that uniqueness instead of relying on an
  arbitrary tie-break that a human could not infer from examples.
- The D14 contribution is structurally inferable because the second frame color survives in the output
  everywhere outside holes. The D15 contribution is structurally inferable because the frame bbox is
  consistently near-full and the sprites are consistently compressed into the edge ring.

Generator/solver contract after modification
- Solver assumptions:
  1. Exactly one large two-color frame exists and its bbox border is fully made from those two colors.
  2. The frame bbox is large (at least 22x22), so tiny paired sprite components cannot win frame search.
  3. Sprite colors never equal either frame color, and every hole shape has a unique plurality color.
- Generator guarantees:
  1. The frame is a centered 24x24 two-color template with holes carved from compact symmetric shapes.
  2. The outside sprite region is only the narrow perimeter ring around that large frame.
  3. For every hole shape, the plurality sprite color is unique and matches the intended fill color.
"""

from __future__ import annotations

from collections import Counter
from itertools import combinations
import random

import numpy as np

from arc_utils import bbox_coords, components_mask, crop_bbox, get_bg, paste, random_free_location


def _all_mask_keys(mask: np.ndarray):
    """Generate keys for all rotations and vertical-flip variants of a boolean mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes())


def _all_rot_flip(mask: np.ndarray):
    """Yield (key, oriented_mask) for all 8 rotation/flip variants of a bool mask."""
    m0 = mask.astype(bool)
    for flip in (False, True):
        m1 = np.flipud(m0) if flip else m0
        for k in range(4):
            m2 = np.rot90(m1, k)
            yield (m2.shape, m2.tobytes()), m2


def _canonical_key(mask: np.ndarray):
    """Canonical representative for a shape under rotation/flip."""
    return min(key for key, _ in _all_rot_flip(mask))


def _all_orientations(mask: np.ndarray):
    """Deduplicated list of all distinct rotation/flip variants."""
    seen, result = set(), []
    for key, m in _all_rot_flip(mask):
        if key not in seen:
            seen.add(key)
            result.append(m)
    return result


def _border_is_full(mask: np.ndarray) -> bool:
    """Return True iff every border cell of `mask` is True."""
    return bool(mask[0, :].all() and mask[-1, :].all() and mask[:, 0].all() and mask[:, -1].all())


def _find_patterned_frame(g: np.ndarray, bg: int) -> tuple[tuple[int, int], tuple[int, int, int, int]]:
    """Locate the large paired-color frame by searching valid color-pair components."""
    colors = sorted(set(map(int, g.flatten())) - {int(bg)})
    assert len(colors) >= 2

    best_score = None
    best_pair = None
    best_bbox = None

    for cols in combinations(colors, 2):
        pair_mask = np.isin(g, cols)
        if not bool(pair_mask.any()):
            continue
        for coords in components_mask(pair_mask, connectivity=8):
            bbox = bbox_coords(coords)
            sub = crop_bbox(g, bbox)
            if sub.shape[0] < LARGE_FRAME_MIN_DIM or sub.shape[1] < LARGE_FRAME_MIN_DIM:
                continue
            sub_pair_mask = np.isin(sub, cols)
            if not _border_is_full(sub_pair_mask):
                continue
            hole_area = int((~sub_pair_mask).sum())
            if hole_area <= 0:
                continue
            bbox_area = int(sub.shape[0] * sub.shape[1])
            pair_area = int(sub_pair_mask.sum())
            score = (bbox_area, pair_area, hole_area)
            if best_score is None or score > best_score:
                best_score = score
                best_pair = (int(cols[0]), int(cols[1]))
                best_bbox = bbox

    assert best_pair is not None and best_bbox is not None
    return best_pair, best_bbox


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a near-full template whose visible material alternates between two frame colors.

    Input structure:
    - The background is the most common color. One very large central template is made from exactly two
      alternating frame colors and occupies most of the 30x30 grid.
    - Inside that template bbox, the holes are connected regions using neither frame color.
    - Outside the bbox, compact sprites are squeezed into the thin edge ring and encode fill colors by shape.

    Rule:
    - Find the large two-color template, crop to its bounding box, and keep its existing two-color pattern.
    - Detect each hole as a 4-connected shape formed by cells that are not either frame color.
    - Match each hole shape to the outside sprites up to rotation/flip, then fill the hole with the unique
      most frequent sprite color for that shape.

    Key insight: The central object is no longer a solid frame and it now almost fills the grid, so the
    solver must infer a large paired-color template first, preserve that pattern, and use the cramped edge
    sprites only as shape-coded color votes for the holes.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    # STEP 1) Find the large two-color frame.
    frame_cols, fb = _find_patterned_frame(g, bg)
    r0, c0, r1, c1 = map(int, fb)
    sub = crop_bbox(g, fb)

    # STEP 2) Extract hole components inside the bbox (cells that are neither frame color), 4-connected.
    frame_mask = np.isin(sub, frame_cols)
    hole_mask = ~frame_mask
    holes: list[tuple[tuple[tuple[int, int], bytes], tuple[int, int], np.ndarray]] = []
    for coords in components_mask(hole_mask, connectivity=4):
        hr0, hc0, hr1, hc1 = bbox_coords(coords)
        rel = coords - np.array([hr0, hc0])
        mask = np.zeros((int(hr1 - hr0 + 1), int(hc1 - hc0 + 1)), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True
        holes.append((((mask.shape[0], mask.shape[1]), mask.tobytes()), (int(hr0), int(hc0)), mask))

    # STEP 3) Gather outside sprites by shape and count their colors.
    outside = g.copy()
    outside[r0 : r1 + 1, c0 : c1 + 1] = int(bg)
    sprite_mask = outside != int(bg)
    sprites_by_key: dict[tuple[tuple[int, int], bytes], dict[int, int]] = {}

    for coords in components_mask(sprite_mask, connectivity=8):
        vals = [int(g[rr, cc]) for rr, cc in coords]
        scol = int(Counter(vals).most_common(1)[0][0])
        if scol in frame_cols:
            continue

        sr0, sc0, sr1, sc1 = bbox_coords(coords)
        rel = coords - np.array([sr0, sc0])
        mask = np.zeros((int(sr1 - sr0 + 1), int(sc1 - sc0 + 1)), dtype=bool)
        mask[rel[:, 0], rel[:, 1]] = True

        for key in _all_mask_keys(mask):
            bucket = sprites_by_key.setdefault(key, {})
            bucket[scol] = int(bucket.get(scol, 0)) + 1

    # STEP 4) Preserve the two-color template and fill the holes by matched sprite plurality.
    out = sub.copy()
    for key, (hr0, hc0), hole in holes:
        cmap = sprites_by_key.get(key)
        assert cmap is not None
        max_count = max(cmap.values())
        winners = [color for color, count in cmap.items() if count == max_count]
        assert len(winners) == 1
        patch = np.full(hole.shape, int(winners[0]), dtype=int)
        paste(out, patch, int(hr0), int(hc0), mask=hole)

    return out


# --------------- generator ---------------

NUM_COLORS = 10
GRID_SIZE = 30

# Large frame geometry
FRAME_MARGIN = 3
FRAME_H = GRID_SIZE - 2 * FRAME_MARGIN
FRAME_W = GRID_SIZE - 2 * FRAME_MARGIN
FRAME_BBOX = (FRAME_MARGIN, FRAME_MARGIN, FRAME_MARGIN + FRAME_H - 1, FRAME_MARGIN + FRAME_W - 1)
LARGE_FRAME_MIN_DIM = 22

# Hole layout: compact shapes so they can also appear as edge sprites.
SHAPE_COUNT_RANGE = (3, 4)
MIN_SHAPES = 3
SHAPE_MAX_AREA = 4
HOLE_CELL_RANGE = (34, 88)
MIN_TOTAL_HOLES = 12
PLACEMENT_RETRIES = 120
MULT_VALUES = [1, 2]
MULT_WEIGHTS = [4, 1]

# Visible frame appearance
PATTERN_STYLES = ("checker", "row", "col")

# Sprite scattering in the narrow edge ring
WINNING_SPRITE_RANGE = (2, 3)
LOSER_COUNT_RANGE = (1, 1)
SPRITE_PADDING = 1

# Generation retry budget
MAX_GEN_ATTEMPTS = 900


def _build_shape_library():
    raw = [
        [[1]],
        [[1, 1]],
        [[1], [1]],
        [[1, 1, 1]],
        [[1], [1], [1]],
        [[1, 1, 1, 1]],
        [[1], [1], [1], [1]],
        [[1, 1], [1, 1]],
        [[1, 0], [1, 1]],
        [[1, 1], [1, 0]],
        [[1, 1, 0], [0, 1, 1]],
        [[0, 1, 1], [1, 1, 0]],
        [[1, 1, 1], [1, 0, 0]],
        [[1, 1, 1], [0, 0, 1]],
        [[1, 0, 1], [1, 1, 1]],
    ]
    seen, dedup = set(), []
    for s in raw:
        m = np.array(s, dtype=bool)
        if int(m.sum()) > SHAPE_MAX_AREA:
            continue
        if min(m.shape) > 2:
            continue
        key = _canonical_key(m)
        if key not in seen:
            seen.add(key)
            dedup.append(m)
    return dedup


SHAPE_LIBRARY = _build_shape_library()


def _mirror_pos(r: int, c: int, sh: int, sw: int, th: int, tw: int):
    """Return 4-way mirrored placements for an even-by-even template."""
    return list(
        {
            (r, c),
            (r, tw - sw - c),
            (th - sh - r, c),
            (th - sh - r, tw - sw - c),
        }
    )


def _validate_positions(occ: np.ndarray, shape: np.ndarray, positions, th: int, tw: int):
    """Check bounds, non-overlap, and 4-disconnection from existing holes."""
    sh, sw = shape.shape
    new_cells = set()
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    new_cells.add((pr + dr, pc + dc))

    for pr, pc in positions:
        if pr < 1 or pc < 1 or pr + sh > th - 1 or pc + sw > tw - 1:
            return None
        for dr in range(sh):
            for dc in range(sw):
                if not shape[dr, dc]:
                    continue
                rr, cc = pr + dr, pc + dc
                if not occ[rr, cc]:
                    return None
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < th and 0 <= nc < tw and not occ[nr, nc] and (nr, nc) not in new_cells:
                        return None
    return positions


def _try_place_quadrant(occ: np.ndarray, shape: np.ndarray, th: int, tw: int):
    """Place shape in the upper-left quadrant and mirror it into four copies."""
    sh, sw = shape.shape
    r_max = th // 2 - sh
    c_max = tw // 2 - sw
    if r_max < 1 or c_max < 1:
        return None
    r = random.randint(1, r_max)
    c = random.randint(1, c_max)
    return _validate_positions(occ, shape, _mirror_pos(r, c, sh, sw, th, tw), th, tw)


def _carve_hole(occ: np.ndarray, shape: np.ndarray, positions):
    """Mark all hole cells as unavailable in the occupancy grid."""
    sh, sw = shape.shape
    for pr, pc in positions:
        for dr in range(sh):
            for dc in range(sw):
                if shape[dr, dc]:
                    occ[pr + dr, pc + dc] = False


def _build_hole_layout(th: int, tw: int):
    """Place mirrored compact holes inside the large template."""
    available = [s for s in SHAPE_LIBRARY if s.shape[0] <= th - 2 and s.shape[1] <= tw - 2]
    if len(available) < MIN_SHAPES:
        return None

    target = random.randint(SHAPE_COUNT_RANGE[0], min(SHAPE_COUNT_RANGE[1], len(available)))
    random.shuffle(available)

    occ = np.ones((th, tw), dtype=bool)
    hole_placements = []

    for shape in available:
        if len(hole_placements) >= target:
            break
        mult = random.choices(MULT_VALUES, weights=MULT_WEIGHTS)[0]
        placed_groups = []
        for _ in range(mult):
            for _ in range(PLACEMENT_RETRIES):
                positions = _try_place_quadrant(occ, shape, th, tw)
                if positions is not None:
                    _carve_hole(occ, shape, positions)
                    placed_groups.append(positions)
                    break
        if placed_groups:
            hole_placements.append((shape, placed_groups))

    if len(hole_placements) < MIN_SHAPES:
        return None

    hole_cells = int((~occ).sum())
    if not (HOLE_CELL_RANGE[0] <= hole_cells <= HOLE_CELL_RANGE[1]):
        return None

    total_holes = sum(len(group) for _, groups in hole_placements for group in groups)
    if total_holes < MIN_TOTAL_HOLES:
        return None

    return occ, hole_placements


def _make_frame_pattern(th: int, tw: int, frame_cols: tuple[int, int]):
    """Create the visible two-color pattern for the large template."""
    a, b = map(int, frame_cols)
    rr, cc = np.indices((th, tw))
    style = random.choice(PATTERN_STYLES)
    if style == "checker":
        selector = (rr + cc) % 2 == 0
    elif style == "row":
        selector = rr % 2 == 0
    else:
        selector = cc % 2 == 0
    return np.where(selector, a, b).astype(int), style


def _build_output(occ: np.ndarray, hole_placements, frame_cols: tuple[int, int], bg: int):
    """Render the target output and record per-shape fill metadata."""
    disallowed = {int(bg), int(frame_cols[0]), int(frame_cols[1])}
    avail_colors = [c for c in range(NUM_COLORS) if c not in disallowed]
    if len(avail_colors) < len(hole_placements):
        return None

    output, style = _make_frame_pattern(occ.shape[0], occ.shape[1], frame_cols)
    shape_colors = random.sample(avail_colors, len(hole_placements))
    shape_info = []

    for (shape, groups), fill_color in zip(hole_placements, shape_colors):
        all_positions = []
        for positions in groups:
            for pr, pc in positions:
                paste(output, np.full(shape.shape, int(fill_color), dtype=int), pr, pc, mask=shape)
                all_positions.append((pr, pc))
        shape_info.append((shape, int(fill_color), all_positions))

    return output, shape_info, style


def _place_sprite(inp: np.ndarray, placement: np.ndarray, ori: np.ndarray, color: int, bg: int):
    """Try to place one sprite in the narrow edge ring."""
    patch = np.full(ori.shape, int(bg), dtype=int)
    patch[ori] = int(color)
    try:
        r, c = random_free_location(placement, patch, transparent=bg, padding=SPRITE_PADDING)
    except ValueError:
        return False
    paste(placement, patch, r, c, transparent=bg)
    paste(inp, patch, r, c, transparent=bg)
    return True


def _scatter_sprites(inp: np.ndarray, shape_info, bg: int, frame_cols: tuple[int, int]):
    """Place compact sprite batches in the outer ring; larger shapes go first."""
    placement = inp.copy()
    disallowed = {int(bg), int(frame_cols[0]), int(frame_cols[1])}
    other_pool = [c for c in range(NUM_COLORS) if c not in disallowed]

    ordered = sorted(shape_info, key=lambda item: (-max(item[0].shape), -int(item[0].sum())))
    for shape, fill_color, _positions in ordered:
        orientations = _all_orientations(shape)
        random.shuffle(orientations)
        n_winning = random.randint(*WINNING_SPRITE_RANGE)

        for _ in range(n_winning):
            if not any(_place_sprite(inp, placement, ori, fill_color, bg) for ori in random.sample(orientations, len(orientations))):
                return False

        losers = [c for c in other_pool if c != fill_color]
        if losers:
            loser_color = random.choice(losers)
            n_loser = random.randint(*LOSER_COUNT_RANGE)
            n_loser = min(n_loser, n_winning - 1)
            for _ in range(n_loser):
                for ori in random.sample(orientations, len(orientations)):
                    if _place_sprite(inp, placement, ori, loser_color, bg):
                        break
                else:
                    return False
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single generation attempt: build the large patterned frame, then squeeze sprites into the edge ring."""
    bg = random.randint(0, NUM_COLORS - 1)
    frame_cols = tuple(random.sample([c for c in range(NUM_COLORS) if c != bg], 2))

    layout = _build_hole_layout(FRAME_H, FRAME_W)
    if layout is None:
        return None
    occ, hole_placements = layout

    built = _build_output(occ, hole_placements, frame_cols, bg)
    if built is None:
        return None
    output, shape_info, _style = built

    frame_pattern = np.where(occ, output, bg).astype(int)
    inp = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    paste(inp, frame_pattern, FRAME_MARGIN, FRAME_MARGIN)

    if not _scatter_sprites(inp, shape_info, bg, frame_cols):
        return None

    try:
        if np.array_equal(main(inp), output):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate input after {MAX_GEN_ATTEMPTS} attempts")
