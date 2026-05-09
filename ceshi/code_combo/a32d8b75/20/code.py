# Combo variant of a32d8b75 (index 20)
# combo_spec: D1/3+D2/3
# Name: 边缘对齐拼接+多精灵交替

import random

import numpy as np
from arc_utils import bbox_coords, neighbors4, random_sprite
from scipy.ndimage import label as _scipy_label


"""
Design analysis for the combo variant.

fusion_summary:
  base_task:
    - read one unit sprite, one frame mask, one anchor square, and one rotation patch from each legend
    - rotate the frame, tile the same sprite into every active frame cell, then place the enlarged result on the canvas
  combo_task:
    - each legend now contains two stacked 5x5 sprites A and B
    - frame cells alternate between A and B by checkerboard parity on the rotated frame lattice
    - each chosen sprite copy is not stamped verbatim: among its 4 rotations, the solver must pick the unique one whose touching edge sequence matches already-placed neighbors

pixel_trace:
  small_example:
    frame_mask_rot:
      [[1, 1],
       [1, 1]]
    original:
      [[S, S],
       [S, S]]
    D2_only:
      [[A, B],
       [B, A]]
    fused:
      - seed at (0,0): A(0 deg)
      - (0,1) must be sprite B because parity flips; if B(0 deg) has the wrong left edge, replace it with the unique rotated B whose left edge equals A(0).right
      - (1,0) must be sprite B as well and is chosen by matching A(0).bottom
      - (1,1) must be sprite A and must satisfy both upper and left contact constraints simultaneously
      result:
        [[A(0),   B(k1)],
         [B(k2),  A(k3)]]
  difference_from_original:
    - the changed step is no longer "repeat one tile everywhere"
    - parity decides which sprite family appears, and edge signatures decide the orientation inside that family

complexity_calibration:
  why_fixed_5x5:
    - with 2x2 or 3x3 binary sprites, many rotations share the same edge signatures, so edge matching becomes ambiguous
    - fixing both sprites to 5x5 gives enough border capacity to make all 4 rotations uniquely identifiable
  why_shared_border_code:
    - A and B need to alternate, but adjacency is always A-to-B or B-to-A
    - giving both families the same four border codes keeps cross-family matching deterministic while still letting their interiors differ visibly
  why_4_connected_frames_in_generation:
    - edge constraints only propagate along side contacts, so diagonal-only adjacency would hide the new rule
    - the generator therefore builds 4-connected frames and rejects masks that do not use both parities
  anti-degeneracy_checks:
    - A and B must differ by several interior pixels and cannot be mere rotations of one another
    - at least one A tile and one B tile must use a non-zero rotation in the generated composite
    - the edge-matched composite must differ from the naive checkerboard composite with no per-tile rotation

implicit_assumption_audit:
  original_assumptions_broken:
    - "top-most non-{0,6} component is the only unit sprite" is false because there are now two stacked sprites
    - "frame starts below one sprite" is false; it must be searched below the lower of the two sprite boxes
    - "all active frame cells use the same tile orientation" is false because both sprite family and rotation vary per cell
  new_solver_assumptions:
    - the two top-most 4-connected components above the marker are the two unit sprites
    - both sprites are square, fully filled, share the same size, and still use the original two-color swap rule
    - within each sprite family, the 4 rotated border signatures are distinct
    - cross-family edge matching is locally deterministic
  generator_guarantees:
    - sprites are stacked with a zero spacer row so extraction is unambiguous
    - both sprites are generated from the same border code but different interior motifs
    - every accepted instance is revalidated by `main(input) == output`

placement_legality:
  - the generator constructs the true edge-matched alternating composite before anchor placement
  - chosen frame sizes ensure the composite fits fully inside the canvas after the legend-directed frame rotation
  - two-legend cases still reject overlapping placed bounding boxes

visual_observability_and_cleanliness:
  - A and B share border grammar but have clearly different interiors, so the checkerboard alternation is visible
  - asymmetric border codes make rotations visually detectable from only a few examples
  - marker square, rotation patch, frame color, and canvas background stay separated from the sprite interiors, keeping the scene readable
"""


STRUCT4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=int)
STRUCT8 = np.ones((3, 3), dtype=int)


def _separator_runs(g: np.ndarray) -> list[tuple[int, int]]:
    """Return contiguous runs of full-height columns that are all color 6."""
    cols6 = [c for c in range(g.shape[1]) if np.all(g[:, c] == 6)]
    if not cols6:
        return []
    cols6.sort()
    runs: list[tuple[int, int]] = []
    s = e = cols6[0]
    for c in cols6[1:]:
        if c == e + 1:
            e = c
        else:
            runs.append((s, e))
            s = e = c
    runs.append((s, e))
    return runs


def _split_canvas(g: np.ndarray) -> tuple[tuple[int, int], tuple[int, int] | None]:
    """Identify the left legend separator and an optional right legend separator."""
    runs = _separator_runs(g)
    assert runs, "must have at least one color-6 separator run"
    left = runs[0]
    right: tuple[int, int] | None = None
    for s, e in reversed(runs[1:]):
        if s >= g.shape[1] - 10 and s > left[1] + 1:
            right = (s, e)
            break
    return left, right


def _all_marker_panel_squares(g: np.ndarray) -> list[tuple[int, int]]:
    """List all 5x5 border-6 squares that look like the top of an 8x5 marker panel."""
    h, w = g.shape
    out: list[tuple[int, int]] = []
    for r0 in range(h - 7):
        for c0 in range(w - 4):
            sub = g[r0 : r0 + 5, c0 : c0 + 5]
            if not (np.all(sub[0, :] == 6) and np.all(sub[4, :] == 6) and np.all(sub[:, 0] == 6) and np.all(sub[:, 4] == 6)):
                continue
            inner = sub[1:4, 1:4]
            if not np.any((inner != 0) & (inner != 6)):
                continue
            rot = g[r0 + 5 : r0 + 8, c0 + 1 : c0 + 4]
            if rot.shape != (3, 3) or not np.any((rot != 0) & (rot != 6)):
                continue
            out.append((int(r0), int(c0)))
    out.sort()
    return out


def _swap_two_colors(sprite: np.ndarray) -> np.ndarray | None:
    cols = sorted({int(x) for x in np.unique(sprite) if int(x) not in (0, 6)})
    if len(cols) != 2:
        return None
    a, b = cols
    out = sprite.copy()
    out[sprite == a] = -1
    out[sprite == b] = a
    out[out == -1] = b
    return out


def _extract_unit_sprites(
    legend: np.ndarray, border6_r0: int
) -> tuple[tuple[np.ndarray, tuple[int, int, int, int]], tuple[np.ndarray, tuple[int, int, int, int]]] | None:
    """Extract the two top-most unit sprites stacked above the frame and marker."""
    scan = legend[:border6_r0, :]
    mask = (scan != 0) & (scan != 6)
    labels, n4 = _scipy_label(mask, structure=STRUCT4)
    if n4 < 2:
        return None

    comps: list[tuple[int, int, int, int]] = []
    for idx in range(1, n4 + 1):
        pts = np.argwhere(labels == idx)
        if pts.size == 0:
            continue
        comps.append(bbox_coords(pts))
    comps.sort(key=lambda bb: (bb[0], bb[1], bb[2], bb[3]))
    if len(comps) < 2:
        return None

    result = []
    for bb in comps[:2]:
        r0, c0, r1, c1 = bb
        spr = legend[r0 : r1 + 1, c0 : c1 + 1].copy()
        if spr.shape[0] != spr.shape[1]:
            return None
        if not np.all((spr != 0) & (spr != 6)):
            return None
        swapped = _swap_two_colors(spr)
        if swapped is None:
            return None
        result.append((swapped, bb))

    if result[0][0].shape != result[1][0].shape:
        return None
    return result[0], result[1]


def _extract_frame_bbox(
    legend: np.ndarray,
    sprite_bbs: list[tuple[int, int, int, int]],
    border6_r0: int,
) -> tuple[int, int, int, int] | None:
    """Frame pixels live below the lower sprite and above the marker square."""
    r_start = max(bb[2] for bb in sprite_bbs) + 1
    r_end = border6_r0 - 1
    if r_start > r_end:
        return None
    sub = legend[r_start : r_end + 1, :]
    mask = (sub != 0) & (sub != 6)
    pts = np.argwhere(mask)
    if pts.size == 0:
        return None
    pts[:, 0] += r_start
    return bbox_coords(pts)


def _position_indices_from_border_square(legend: np.ndarray, border6_r0: int, border6_c0: int) -> tuple[int, int] | None:
    sq = legend[border6_r0 : border6_r0 + 5, border6_c0 : border6_c0 + 5]
    inner = sq[1:4, 1:4]
    pts = np.argwhere((inner != 0) & (inner != 6))
    if pts.size == 0:
        return None
    r, c = int(pts[0, 0]), int(pts[0, 1])
    return r, c


def _rotation_from_clockhand(legend: np.ndarray, border6_r0: int, border6_c0: int) -> int:
    """Decode the 3x3 clock-hand rotation mark as 0/1/2/3 quarter-turns clockwise."""
    h, w = legend.shape
    r0 = border6_r0 + 5
    c0 = border6_c0 + 1
    assert r0 + 2 < h and c0 + 2 < w, "rotation marker patch out of bounds"
    sub = legend[r0 : r0 + 3, c0 : c0 + 3]
    pts = np.argwhere((sub != 0) & (sub != 6))
    assert pts.size > 0, "no rotation marker pixel found"
    center = np.array([1, 1], dtype=int)
    dists = np.abs(pts - center).sum(axis=1)
    p = pts[int(dists.argmax())]
    dr, dc = int(p[0] - center[0]), int(p[1] - center[1])
    if abs(dr) >= abs(dc):
        if dr < 0:
            return 0
        if dr > 0:
            return 2
    else:
        if dc > 0:
            return 1
        if dc < 0:
            return 3
    raise AssertionError("rotation marker pixel at center, direction ambiguous")


def _anchor_to_top_left(anchor_idx: int, size: int, place_size: int) -> int:
    """Convert a start/middle/end anchor index into the placed object's top-left coordinate."""
    if anchor_idx <= 0:
        canvas_anchor = 0
        obj_anchor = 0
    elif anchor_idx >= 2:
        canvas_anchor = max(0, size - 1)
        obj_anchor = max(0, place_size - 1)
    else:
        canvas_anchor = (size - 1) // 2 if size > 1 else 0
        obj_anchor = (place_size - 1) // 2 if place_size > 1 else 0
    return canvas_anchor - obj_anchor


def _rotated_variants(sprite: np.ndarray) -> list[np.ndarray]:
    return [np.rot90(sprite, (-k) % 4).copy() for k in range(4)]


def _edge_signatures(sprite: np.ndarray) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    top = tuple(int(x) for x in sprite[0, :])
    right = tuple(int(x) for x in sprite[:, -1])
    bottom = tuple(int(x) for x in sprite[-1, :])
    left = tuple(int(x) for x in sprite[:, 0])
    return top, right, bottom, left


def _parity_family(r: int, c: int) -> int:
    return 0 if (r + c) % 2 == 0 else 1


def _candidate_rotations(
    source_edges: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    target_edges: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]],
    source_k: int,
    dr: int,
    dc: int,
) -> list[int]:
    src_top, src_right, src_bottom, src_left = source_edges[source_k]
    if dr == -1 and dc == 0:
        target_side = 2
        required = src_top
    elif dr == 1 and dc == 0:
        target_side = 0
        required = src_bottom
    elif dr == 0 and dc == -1:
        target_side = 1
        required = src_left
    elif dr == 0 and dc == 1:
        target_side = 3
        required = src_right
    else:
        raise AssertionError("edge matching only supports 4-neighbor propagation")
    return [k for k, sig in enumerate(target_edges) if sig[target_side] == required]


def _build_checkerboard_composite(frame_mask: np.ndarray, sprite_a: np.ndarray, sprite_b: np.ndarray) -> np.ndarray:
    """Tile the two sprites by parity without per-tile rotation."""
    assert sprite_a.shape == sprite_b.shape
    uh, uw = sprite_a.shape
    fh, fw = frame_mask.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    mask_a = (sprite_a != 0) & (sprite_a != 6)
    mask_b = (sprite_b != 0) & (sprite_b != 6)
    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j]:
                continue
            rr0 = i * uh
            cc0 = j * uw
            if _parity_family(i, j) == 0:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mask_a] = sprite_a[mask_a]
            else:
                comp[rr0 : rr0 + uh, cc0 : cc0 + uw][mask_b] = sprite_b[mask_b]
    return comp


def _build_alternating_edge_matched_composite(
    frame_mask: np.ndarray, sprite_a: np.ndarray, sprite_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """
    Alternate sprite families by parity, then propagate edge constraints to choose rotations.

    Each 4-connected frame component seeds its first visited tile with rotation 0; all
    remaining tiles must take the unique rotation of their parity-chosen family whose
    touching edge matches every already-fixed neighbor.
    """
    variants = [_rotated_variants(sprite_a), _rotated_variants(sprite_b)]
    edges = [[_edge_signatures(v) for v in fam] for fam in variants]
    fh, fw = frame_mask.shape
    chosen = np.full((fh, fw), -1, dtype=int)

    for i in range(fh):
        for j in range(fw):
            if not frame_mask[i, j] or chosen[i, j] >= 0:
                continue
            chosen[i, j] = 0
            queue = [(i, j)]
            qpos = 0
            while qpos < len(queue):
                r, c = queue[qpos]
                qpos += 1
                src_family = _parity_family(r, c)
                src_k = int(chosen[r, c])
                for nr, nc in neighbors4(r, c):
                    if not (0 <= nr < fh and 0 <= nc < fw and frame_mask[nr, nc]):
                        continue
                    dst_family = _parity_family(nr, nc)
                    cands = _candidate_rotations(edges[src_family], edges[dst_family], src_k, nr - r, nc - c)
                    assert cands, "no rotation satisfies local edge-matching constraints"
                    assert len(cands) == 1, "edge-matching is ambiguous for the alternating sprite family"
                    cand = cands[0]
                    if chosen[nr, nc] < 0:
                        chosen[nr, nc] = cand
                        queue.append((nr, nc))
                    else:
                        assert int(chosen[nr, nc]) == cand, "edge-matching constraints disagree across neighbors"

    uh, uw = sprite_a.shape
    comp = np.zeros((fh * uh, fw * uw), dtype=int)
    masks = [
        (sprite_a != 0) & (sprite_a != 6),
        (sprite_b != 0) & (sprite_b != 6),
    ]
    for i in range(fh):
        for j in range(fw):
            k = int(chosen[i, j])
            if k < 0:
                continue
            fam = _parity_family(i, j)
            tile = variants[fam][k]
            rr0 = i * uh
            cc0 = j * uw
            comp[rr0 : rr0 + uh, cc0 : cc0 + uw][masks[fam]] = tile[masks[fam]]
    return comp, chosen


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble and place enlarged composites from two alternating legend sprites with edge-matched rotations.

    Input structure:
    - Color-6 separator columns still split the full grid into legend panel(s) and the central canvas.
    - Each legend now shows two stacked 5x5 unit sprites above one frame template, followed by the usual 5x5 anchor square and 3x3 rotation clock-hand.

    Rule:
    - Read both sprites, color-swap them, rotate the frame layout with the clock-hand, then assign sprite A to even-parity active frame cells and sprite B to odd-parity active frame cells.
    - For each chosen A/B tile, keep the base orientation only if its touching border sequence matches already-fixed neighbors; otherwise replace it with the unique rotated version that satisfies the edge constraints, then place the completed composite by the anchor marker.

    Key insight:
    - The legend encodes two coupled decisions: parity tells you which sprite family appears at each frame cell, and the asymmetric borders of that family tell you which rotation is forced so adjacent enlarged tiles share identical contact-edge color sequences.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w >= 2, "grid width must be >= 2"

    left_sep, right_sep = _split_canvas(g)
    canvas_c0 = left_sep[1] + 1
    canvas_c1 = right_sep[0] if right_sep is not None else w
    main_canvas = g[:, canvas_c0:canvas_c1].copy()

    marker_squares = _all_marker_panel_squares(g)
    assert marker_squares, "no marker panel found"

    eff_h, eff_w = main_canvas.shape
    full6_cols = [c for c in range(eff_w) if np.all(main_canvas[:, c] == 6)]
    if full6_cols:
        eff_w = min(full6_cols)
    full6_rows = [r for r in range(eff_h) if np.all(main_canvas[r, :] == 6)]
    if full6_rows:
        eff_h = min(full6_rows)

    out = main_canvas.copy()
    composite_written = np.zeros(out.shape, dtype=bool)

    for marker_r0, marker_c0 in marker_squares:
        if canvas_c0 <= marker_c0 < canvas_c1:
            continue
        if marker_c0 + 5 > w:
            continue

        panel = g[:, marker_c0 : marker_c0 + 5]
        sprites = _extract_unit_sprites(panel, marker_r0)
        if sprites is None:
            continue
        (sprite_a, bb_a), (sprite_b, bb_b) = sprites

        frame_bb = _extract_frame_bbox(panel, [bb_a, bb_b], marker_r0)
        if frame_bb is None:
            continue
        fr0, fc0, fr1, fc1 = frame_bb
        frame = panel[fr0 : fr1 + 1, fc0 : fc1 + 1]
        frame_mask0 = (frame != 0) & (frame != 6)
        _, n8 = _scipy_label(frame_mask0, structure=STRUCT8)
        assert n8 == 1, f"frame must be 8-connected, got {n8} components"

        dir_k = _rotation_from_clockhand(g, marker_r0, marker_c0)
        frame_mask = np.rot90(frame_mask0, (-dir_k) % 4)
        comp, _tile_rotations = _build_alternating_edge_matched_composite(frame_mask, sprite_a, sprite_b)

        pos = _position_indices_from_border_square(g, marker_r0, marker_c0)
        if pos is None:
            continue
        pr, pc = pos
        top = _anchor_to_top_left(pr, eff_h, comp.shape[0])
        left = _anchor_to_top_left(pc, eff_w, comp.shape[1])

        ch, cw = comp.shape
        for r in range(ch):
            rr = top + r
            assert 0 <= rr < out.shape[0], "sprite row out of canvas bounds"
            for c in range(cw):
                cc = left + c
                assert 0 <= cc < out.shape[1], "sprite col out of canvas bounds"
                if out[rr, cc] == 6:
                    continue
                val = int(comp[r, c])
                if val != 0 and val != 6:
                    assert not composite_written[rr, cc], "composites must not overlap"
                    composite_written[rr, cc] = True
                    out[rr, cc] = val
    return out


# --------------- generator (edge-matched alternating dual sprites) ---------------

GRID_H_RANGE = (23, 30)
GRID_W = 30
LEGEND_COUNT_CHOICES = (1, 2)
UNIT_SIZE = 5
MIN_CANVAS_W = 10
FRAME_H_RANGE = (2, 5)          # 2..4
FRAME_W_RANGE = (2, 5)          # 2..4
FRAME_DENSITY = 0.65
BAND_WIDTH_MIN = 4
MARKER_PANEL_HEIGHT = 8
USABLE_COLORS = [1, 2, 3, 4, 5, 7, 8, 9]
ROTATION_DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
GEN_RETRIES = 500
SPRITE_STACK_GAP = 1
MIN_FRAME_ACTIVE = 4
MIN_SPRITE_DIFF = 5

_EDGE_PATTERNS = [
    (0, 0, 0, 0, 0),
    (0, 0, 1, 0, 0),
    (0, 1, 0, 1, 0),
    (0, 1, 1, 1, 0),
]

_INTERIOR_MOTIFS = [
    np.array([[1, 0, 1], [0, 1, 1], [1, 0, 0]], dtype=int),
    np.array([[0, 1, 1], [1, 0, 1], [0, 1, 0]], dtype=int),
    np.array([[1, 1, 0], [0, 1, 0], [1, 0, 1]], dtype=int),
    np.array([[1, 0, 0], [1, 1, 0], [0, 1, 1]], dtype=int),
    np.array([[0, 1, 0], [1, 1, 0], [1, 0, 1]], dtype=int),
]
_EDGE_PERMUTATIONS_B = [
    (1, 0, 3, 2),
    (3, 2, 1, 0),
]


def _allocate_colors(num_legends: int):
    """Pick per-legend unit palettes, canvas colors, and frame colors."""
    pool = list(USABLE_COLORS)
    random.shuffle(pool)

    unit_color_sets = []
    used_unit = set()
    for _ in range(num_legends):
        if len(pool) < 2:
            return None
        c1, c2 = pool.pop(), pool.pop()
        unit_color_sets.append((c1, c2))
        used_unit.update([c1, c2])

    bg_cands = [c for c in pool if c not in used_unit]
    if len(bg_cands) < 2:
        return None
    random.shuffle(bg_cands)
    bg_colors = (bg_cands[0], bg_cands[1])

    frame_colors = []
    all_set = set(USABLE_COLORS)
    for uc in unit_color_sets:
        candidates = list(all_set - set(uc))
        frame_colors.append(random.choice(candidates))

    return unit_color_sets, bg_colors, frame_colors


def _color_swap(sprite: np.ndarray, c1: int, c2: int) -> np.ndarray:
    out = sprite.copy()
    out[sprite == c1] = -1
    out[sprite == c2] = c1
    out[out == -1] = c2
    return out


def _bit_row(bits: tuple[int, ...], c1: int, c2: int) -> np.ndarray:
    return np.array([c2 if b else c1 for b in bits], dtype=int)


def _apply_random_transform(motif: np.ndarray) -> np.ndarray:
    out = motif.copy()
    out = np.rot90(out, random.randint(0, 3))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return out


def _build_sprite_with_shared_edges(
    c1: int,
    c2: int,
    edge_bits: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    interior: np.ndarray,
) -> np.ndarray:
    top_bits, right_bits, bottom_bits, left_bits = edge_bits
    sprite = np.full((UNIT_SIZE, UNIT_SIZE), c1, dtype=int)
    sprite[1:4, 1:4] = np.where(interior == 1, c2, c1)
    sprite[0, :] = _bit_row(top_bits, c1, c2)
    sprite[-1, :] = _bit_row(bottom_bits, c1, c2)
    sprite[:, 0] = _bit_row(left_bits, c1, c2)
    sprite[:, -1] = _bit_row(right_bits, c1, c2)
    sprite[0, :] = _bit_row(top_bits, c1, c2)
    sprite[-1, :] = _bit_row(bottom_bits, c1, c2)
    return sprite


def _permute_edge_bits(
    edge_bits: tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]],
    order: tuple[int, int, int, int],
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Reassign the same four edge codes to different sides while preserving the code multiset."""
    return tuple(edge_bits[i] for i in order)


def _design_sprite_pair(unit_colors: tuple[int, int]) -> tuple[np.ndarray, np.ndarray] | None:
    c1, c2 = unit_colors
    edge_bits_a = tuple(random.sample(_EDGE_PATTERNS, 4))
    edge_bits_b = _permute_edge_bits(edge_bits_a, random.choice(_EDGE_PERMUTATIONS_B))
    for _ in range(120):
        interior_a = _apply_random_transform(random.choice(_INTERIOR_MOTIFS))
        interior_b = _apply_random_transform(random.choice(_INTERIOR_MOTIFS))
        sprite_a = _build_sprite_with_shared_edges(c1, c2, edge_bits_a, interior_a)
        sprite_b = _build_sprite_with_shared_edges(c1, c2, edge_bits_b, interior_b)
        if np.array_equal(sprite_a, sprite_b):
            continue
        if int(np.sum(sprite_a != sprite_b)) < MIN_SPRITE_DIFF:
            continue
        if any(np.array_equal(sprite_a, np.rot90(sprite_b, k)) for k in range(4)):
            continue
        vars_a = _rotated_variants(sprite_a)
        vars_b = _rotated_variants(sprite_b)
        top_a = {tuple(int(x) for x in v[0, :]) for v in vars_a}
        left_a = {tuple(int(x) for x in v[:, 0]) for v in vars_a}
        top_b = {tuple(int(x) for x in v[0, :]) for v in vars_b}
        left_b = {tuple(int(x) for x in v[:, 0]) for v in vars_b}
        if len(top_a) != 4 or len(left_a) != 4 or len(top_b) != 4 or len(left_b) != 4:
            continue
        return sprite_a, sprite_b
    return None


def _random_frame_mask(h: int, w: int) -> np.ndarray | None:
    for _ in range(120):
        mask = random_sprite(
            h,
            w,
            density=FRAME_DENSITY,
            connectivity=4,
            color_palette=[1],
            symmetry="not_symmetric",
        ) != 0
        labels4, n4 = _scipy_label(mask, structure=STRUCT4)
        if n4 != 1:
            continue
        active = int(mask.sum())
        if active < MIN_FRAME_ACTIVE:
            continue
        rr, cc = np.indices(mask.shape)
        even = int(np.sum(mask & (((rr + cc) % 2) == 0)))
        odd = int(np.sum(mask & (((rr + cc) % 2) == 1)))
        if even == 0 or odd == 0:
            continue
        if not (np.any(mask[:, :-1] & mask[:, 1:]) or np.any(mask[:-1, :] & mask[1:, :])):
            continue
        return mask
    return None


def _design_legend(leg: dict, unit_colors: tuple[int, int], frame_color: int, marker_r0: int, grid_h: int, canvas_w: int) -> bool:
    sprites = _design_sprite_pair(unit_colors)
    if sprites is None:
        return False
    sprite_a, sprite_b = sprites
    uc1, uc2 = unit_colors
    legend_sprite_a = _color_swap(sprite_a, uc1, uc2)
    legend_sprite_b = _color_swap(sprite_b, uc1, uc2)

    after_sprites = 2 * UNIT_SIZE + SPRITE_STACK_GAP + 1
    avail = marker_r0 - after_sprites
    combos = []
    for fH in range(*FRAME_H_RANGE):
        if avail < fH + 1:
            continue
        for fW in range(*FRAME_W_RANGE):
            for rk in range(4):
                rh, rw = (fH, fW) if rk % 2 == 0 else (fW, fH)
                if rh * UNIT_SIZE <= grid_h and rw * UNIT_SIZE <= canvas_w:
                    combos.append((fH, fW, rk))
    if not combos:
        return False

    random.shuffle(combos)
    for frame_H, frame_W, rot_k in combos:
        frame_mask = _random_frame_mask(frame_H, frame_W)
        if frame_mask is None:
            continue
        rotated_frame = np.rot90(frame_mask, (-rot_k) % 4)
        comp, tile_rots = _build_alternating_edge_matched_composite(rotated_frame, sprite_a, sprite_b)
        base_comp = _build_checkerboard_composite(rotated_frame, sprite_a, sprite_b)
        active = rotated_frame.astype(bool)
        rr, cc = np.indices(rotated_frame.shape)
        even_mask = active & (((rr + cc) % 2) == 0)
        odd_mask = active & (((rr + cc) % 2) == 1)
        if not np.any(even_mask) or not np.any(odd_mask):
            continue
        if len({int(v) for v in tile_rots[active].ravel()}) < 3:
            continue
        if not np.any(tile_rots[active] != 0):
            continue
        if not np.any(comp != base_comp):
            continue
        leg.update(
            N=UNIT_SIZE,
            legend_w=UNIT_SIZE + 2,
            sprite_a=sprite_a,
            sprite_b=sprite_b,
            legend_sprite_a=legend_sprite_a,
            legend_sprite_b=legend_sprite_b,
            frame_mask=frame_mask,
            frame_H=frame_H,
            frame_W=frame_W,
            frame_color=frame_color,
            rot_k=rot_k,
            rotated_frame=rotated_frame,
            composite=comp,
            tile_rotations=tile_rots,
        )
        return True
    return False


def _comp_size(leg: dict) -> tuple[int, int]:
    comp = leg["composite"]
    return int(comp.shape[0]), int(comp.shape[1])


def _assign_anchors(legends: list[dict], grid_h: int, canvas_w: int) -> bool:
    """Assign anchor positions, avoiding overlap when two legends are present."""
    if len(legends) == 1:
        pr, pc = random.randint(0, 2), random.randint(0, 2)
        ch, cw = _comp_size(legends[0])
        legends[0].update(
            pr=pr,
            pc=pc,
            comp_top=_anchor_to_top_left(pr, grid_h, ch),
            comp_left=_anchor_to_top_left(pc, canvas_w, cw),
        )
        return True

    a, b = legends[0], legends[1]
    a_ch, a_cw = _comp_size(a)
    b_ch, b_cw = _comp_size(b)
    valid_pairs = []
    for apr in range(3):
        for apc in range(3):
            at = _anchor_to_top_left(apr, grid_h, a_ch)
            al = _anchor_to_top_left(apc, canvas_w, a_cw)
            if at < 0 or al < 0 or at + a_ch > grid_h or al + a_cw > canvas_w:
                continue
            for bpr in range(3):
                for bpc in range(3):
                    bt = _anchor_to_top_left(bpr, grid_h, b_ch)
                    bl = _anchor_to_top_left(bpc, canvas_w, b_cw)
                    if bt < 0 or bl < 0 or bt + b_ch > grid_h or bl + b_cw > canvas_w:
                        continue
                    if at < bt + b_ch and bt < at + a_ch and al < bl + b_cw and bl < al + a_cw:
                        continue
                    valid_pairs.append((apr, apc, at, al, bpr, bpc, bt, bl))
    if not valid_pairs:
        return False
    apr, apc, at, al, bpr, bpc, bt, bl = random.choice(valid_pairs)
    a.update(pr=apr, pc=apc, comp_top=at, comp_left=al)
    b.update(pr=bpr, pc=bpc, comp_top=bt, comp_left=bl)
    return True


def _build_banded_canvas(grid_h: int, canvas_w: int, bg1: int, bg2: int) -> np.ndarray:
    canvas = np.zeros((grid_h, canvas_w), dtype=int)
    band_dir = random.choice(["h", "v"])
    dim = grid_h if band_dir == "h" else canvas_w
    bw = random.randint(BAND_WIDTH_MIN, max(BAND_WIDTH_MIN, dim // 2))
    band_indices = np.arange(dim) // bw % 2
    colors = np.array([bg1, bg2])
    if band_dir == "h":
        canvas[:] = colors[band_indices][:, None]
    else:
        canvas[:] = colors[band_indices][None, :]
    return canvas


def _overlay_composites(canvas: np.ndarray, legends: list[dict]) -> np.ndarray:
    output = canvas.copy()
    for leg in legends:
        comp = leg["composite"]
        t, l = leg["comp_top"], leg["comp_left"]
        region = output[t : t + comp.shape[0], l : l + comp.shape[1]]
        write_mask = comp != 0
        region[write_mask] = comp[write_mask]
    return output


def _place_legend(grid: np.ndarray, leg: dict, col_start: int, sep_col: int, marker_r0: int, is_right: bool) -> None:
    """Draw one legend panel: two sprites, frame, marker square, and rotation patch."""
    N = leg["N"]
    grid[:, sep_col] = 6

    mc0 = (col_start + max(0, N - 4)) if not is_right else (sep_col + max(0, N - 3))

    grid[marker_r0 : marker_r0 + 5, mc0 : mc0 + 5] = 0
    grid[marker_r0, mc0 : mc0 + 5] = 6
    grid[marker_r0 + 4, mc0 : mc0 + 5] = 6
    grid[marker_r0 : marker_r0 + 5, mc0] = 6
    grid[marker_r0 : marker_r0 + 5, mc0 + 4] = 6
    grid[marker_r0 + 1 + leg["pr"], mc0 + 1 + leg["pc"]] = 4

    rr0 = marker_r0 + 5
    grid[rr0 : rr0 + 3, mc0 : mc0 + 5] = 0
    grid[rr0 : rr0 + 3, mc0] = 6
    grid[rr0 : rr0 + 3, mc0 + 4] = 6
    cx_r, cx_c = rr0 + 1, mc0 + 2
    dr, dc = ROTATION_DIRS[leg["rot_k"]]
    grid[cx_r, cx_c] = 7
    grid[cx_r + dr, cx_c + dc] = 7

    sp_col = col_start + 1 - mc0
    spr_c0 = mc0 + sp_col
    grid[1 : 1 + N, spr_c0 : spr_c0 + N] = leg["legend_sprite_a"]
    row_b0 = 1 + N + SPRITE_STACK_GAP
    grid[row_b0 : row_b0 + N, spr_c0 : spr_c0 + N] = leg["legend_sprite_b"]

    fH, fW = leg["frame_H"], leg["frame_W"]
    after_sprites = 2 * N + SPRITE_STACK_GAP + 1
    avail = marker_r0 - after_sprites
    gap = random.randint(1, max(1, avail - fH))
    fr_start = after_sprites + gap
    f_col = random.randint(0, max(0, 4 - fW))
    frame_region = grid[fr_start : fr_start + fH, mc0 + f_col : mc0 + f_col + fW]
    frame_region[leg["frame_mask"]] = leg["frame_color"]


def _gen_attempt(num_legends: int) -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)

    legends = [{"N": UNIT_SIZE, "legend_w": UNIT_SIZE + 2} for _ in range(num_legends)]
    left_lw = legends[0]["legend_w"]
    left_total = left_lw + 1
    right_total = (1 + legends[1]["legend_w"]) if num_legends == 2 else 0
    canvas_w = GRID_W - left_total - right_total
    if canvas_w < MIN_CANVAS_W:
        return None

    colors = _allocate_colors(num_legends)
    if colors is None:
        return None
    unit_color_sets, (bg1, bg2), frame_colors = colors

    marker_r0 = grid_h - MARKER_PANEL_HEIGHT
    for li in range(num_legends):
        if not _design_legend(legends[li], unit_color_sets[li], frame_colors[li], marker_r0, grid_h, canvas_w):
            return None

    if not _assign_anchors(legends, grid_h, canvas_w):
        return None

    canvas = _build_banded_canvas(grid_h, canvas_w, bg1, bg2)
    output = _overlay_composites(canvas, legends)

    grid = np.zeros((grid_h, GRID_W), dtype=int)
    canvas_c0 = left_total
    grid[:, canvas_c0 : canvas_c0 + canvas_w] = canvas

    _place_legend(grid, legends[0], col_start=0, sep_col=left_lw, marker_r0=marker_r0, is_right=False)
    if num_legends == 2:
        r_sep = canvas_c0 + canvas_w
        _place_legend(grid, legends[1], col_start=r_sep + 1, sep_col=r_sep, marker_r0=marker_r0, is_right=True)

    try:
        if np.array_equal(main(grid), output):
            return grid
    except Exception:
        pass
    return None


def generate_input() -> np.ndarray:
    num_legends = random.choice(LEGEND_COUNT_CHOICES)
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(num_legends)
        if result is not None:
            return result
    alt_num_legends = LEGEND_COUNT_CHOICES[0] if num_legends == LEGEND_COUNT_CHOICES[1] else LEGEND_COUNT_CHOICES[1]
    for _ in range(max(50, GEN_RETRIES // 5)):
        result = _gen_attempt(alt_num_legends)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {GEN_RETRIES} + fallback attempts")
