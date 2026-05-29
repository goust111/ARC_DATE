# Combo variant of 89565ca0 (index 12)
# combo_spec: D4/3+D11/0
# Name: 干扰框架+仅2个框架

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import bbox, colors_present, crop_bbox, get_bg, label_mask


# Design analysis for D4/3 + D11/0
# a) Concrete behavior and trace
#    - Original 89565ca0 used 3-5 real frame colors plus one scattered noise color. The solver
#      reconstructed each color's rectilinear walls, counted enclosed regions, discarded the
#      unique zero-room color, then output same-color bars sorted by room count.
#    - This combo keeps the same room-count-to-bar mapping, but only two colors are valid frames.
#      The third non-black color is now a structured open rectilinear distractor instead of pixel
#      noise. So the visible input is "three frame-like colors, but only two actually close rooms".
#    - Small trace: if blue encloses 2 rooms, red encloses 5 rooms, and yellow is an open comb,
#      the original task might have several extra real frames plus yellow speckles. Here the input
#      has exactly those three colors; `main()` measures room counts [2, 5, 0], keeps blue/red,
#      and outputs two rows padded with yellow.
#    - Difference: the changed decision is not "ignore visibly random dust" but "among three line
#      drawings, identify which two are genuine closed frames".
#
# b) Complexity calibration
#    - Fixing the real-frame count to two simplifies the scene, so the distractor must stay
#      visually salient. The generator therefore makes the distractor a connected comb/shelf shape
#      spanning a noticeable fraction of the board and usually crossing a true frame on a few wall
#      pixels.
#    - To avoid replacing the original relation with arbitrary tie handling, the two real frames
#      always have distinct room counts. This keeps the rule human-readable from few examples:
#      identify the two closed structures, count their rooms, sort ascending.
#
# c) Implicit-assumption audit of main()
#    - Iteration over `colors_present(..., exclude={0})` now assumes there are exactly three
#      non-black colors. Generator enforces 2 real + 1 distractor.
#    - Cropping one bbox per color assumes each color appears in one connected drawing. Generator
#      renders each color exactly once and avoids self-fragmenting occlusion.
#    - Filtering `rooms > 0` assumes the distractor never accidentally forms a cycle. The comb
#      geometry is built to keep fewer than two dense full-height columns, so reconstruction does
#      not hallucinate a proper frame.
#    - Sorting only by room count assumes the two valid frames do not tie. Generator samples
#      distinct room counts and validates the exact expected output.
#
# d) Placement legality
#    - The two real frames are placed fully in-bounds with a moat, so they do not overlap each
#      other and remain individually readable.
#    - The distractor is also in-bounds. It may overlap true walls, but protected anchor pixels
#      on each real frame's corners and side midpoints are never overwritten, preserving bbox
#      extrema and enough wall evidence for reconstruction.
#
# e) Visual observability
#    - Across examples, the input always shows exactly three non-black colors but the output always
#      has two rows. That makes the "one color is a non-closing distractor" rule directly visible.
#    - Because the distractor is itself made of long line segments, the change is observable in the
#      pixels rather than being hidden in metadata or color ids.
#
# f) Visual cleanliness
#    - Real frames do not overlap each other, which keeps the reduced two-object scene clean.
#      Only the distractor may cross them, and only lightly, so the grid is still easy to parse.
#
# g) Inferability and anti-degeneration target
#    - No color remapping is introduced: surviving colors keep their identity in the output, and
#      the distractor color becomes padding.
#    - Hallmark combo metric: every accepted sample must have exactly one zero-room color among the
#      three non-black colors, and the distractor must visibly overlap at least one real frame
#      pixel. Otherwise the scene would drift toward a trivial "two clean frames plus unrelated
#      doodle" version of the task.
#
# h) New solver assumptions checked against generator
#    - `main()` asserts that there are exactly two positive-room colors and one zero-room color.
#      `generate_input()` guarantees that by construction and by solver-oracle validation.
#    - `main()` assumes room-count order is strict. `generate_input()` samples distinct counts.


def _count_enclosed_regions(walls: np.ndarray) -> int:
    labels, n_labels = label_mask(~walls, connectivity=4)
    h, w = walls.shape
    count = 0
    for k in range(1, n_labels + 1):
        coords = np.argwhere(labels == k)
        assert coords.size > 0
        touches_edge = (
            np.any(coords[:, 0] == 0)
            or np.any(coords[:, 0] == h - 1)
            or np.any(coords[:, 1] == 0)
            or np.any(coords[:, 1] == w - 1)
        )
        if not touches_edge:
            count += 1
    return int(count)


def _reconstruct_walls(mask: np.ndarray) -> np.ndarray:
    h, _w = mask.shape
    col_counts = mask.sum(axis=0)
    v_threshold = max(3, math.ceil(0.45 * h))
    vcols = [int(c) for c, n in enumerate(col_counts) if int(n) >= v_threshold]
    if len(vcols) < 2:
        return mask.copy()

    row_counts = mask.sum(axis=1)
    h_threshold = max(3, len(vcols) + 1)
    hrows = [int(r) for r, n in enumerate(row_counts) if int(n) >= h_threshold]
    assert len(hrows) >= 2

    walls = np.zeros_like(mask, dtype=bool)

    for r in hrows:
        for c0, c1 in zip(vcols, vcols[1:]):
            segment = mask[r, c0 : c1 + 1]
            length = int(c1 - c0 + 1)
            hits = int(segment.sum())
            dense = hits >= max(2, math.ceil(0.45 * length))
            interior_evidence = length > 2 and hits >= 2 and bool(segment[1:-1].any())
            if dense or interior_evidence:
                walls[r, c0 : c1 + 1] = True

    for c in vcols:
        for r0, r1 in zip(hrows, hrows[1:]):
            segment = mask[r0 : r1 + 1, c]
            length = int(r1 - r0 + 1)
            hits = int(segment.sum())
            if hits >= max(2, math.ceil(0.45 * length)):
                walls[r0 : r1 + 1, c] = True

    return walls


def _analyze_color(g: np.ndarray, color: int) -> tuple[int, np.ndarray]:
    b = bbox(g == int(color))
    assert b is not None
    r0, c0, r1, c1 = b
    mask = crop_bbox(g == int(color), b).astype(bool)
    walls = _reconstruct_walls(mask)
    out = np.zeros_like(g, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = walls
    return _count_enclosed_regions(walls), out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Convert the two genuine partitioned frames into horizontal bars whose lengths equal their enclosed-room counts.

    Input structure:
    - Black background with exactly three non-black rectilinear line drawings.
    - Two colors are closed axis-aligned partitioned frames that enclose one or more black regions.
    - The remaining color is an open frame-like distractor made of line segments; it may cross a true frame but never closes a room.

    Rule:
    - For each non-black color, crop its bbox and reconstruct rectilinear walls from dense row/column evidence.
    - Count the 4-connected black regions enclosed by those reconstructed walls.
    - Keep the two colors with positive room counts, sort them by room count ascending, and use the unique zero-room color as output padding.

    Key insight:
    - The decisive test is closure: among three frame-like colors, exactly one stays open after reconstruction and therefore contributes only the padding color, while the other two behave like the original room-count frames.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    assert bg == 0

    counts: list[tuple[int, int, np.ndarray]] = []
    for color in colors_present(g, exclude={bg}):
        rooms, walls = _analyze_color(g, int(color))
        counts.append((int(color), int(rooms), walls))

    assert len(counts) == 3
    drawings = [(color, rooms, walls) for color, rooms, walls in counts if rooms > 0]
    pad_colors = [color for color, rooms, _walls in counts if rooms == 0]
    assert len(drawings) == 2
    assert len(pad_colors) == 1
    assert drawings[0][1] != drawings[1][1]

    pad = int(pad_colors[0])
    ordered = sorted(drawings, key=lambda item: item[1])
    width = max(rooms for _color, rooms, _walls in ordered)

    out = np.full((2, width), pad, dtype=int)
    for r, (color, rooms, _walls) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


# --------------- generator (combo variant) ---------------

_FRAME_COUNT = 2
_ROOM_COUNT_RANGE = (1, 6)
_GRID_H_RANGE = (20, 30)
_GRID_W_RANGE = (20, 30)
_FRAME_MAX_H = 18
_FRAME_MAX_W = 22
_CELL_GAP_RANGE = (3, 13)
_FRAME_PLACEMENT_RETRY = 160
_DISTRACTOR_FRACTION_RANGE = (0.05, 0.10)
_DISTRACTOR_HEIGHT_RANGE = (12, 18)
_DISTRACTOR_RUNG_COUNT_RANGE = (3, 5)
_DISTRACTOR_MAX_WIDTH = 18
_DISTRACTOR_PLACEMENT_RETRY = 300
_DISTRACTOR_TARGET_OVERLAP = (1, 8)
_GEN_RETRIES = 400


def _random_gaps(n: int, min_gap: int, max_gap: int, max_total: int) -> list[int] | None:
    if n * min_gap > max_total:
        return None
    gaps = [min_gap] * n
    remaining = max_total - sum(gaps)
    for i in range(n):
        extra = min(max_gap - gaps[i], remaining)
        if extra > 0:
            add = random.randint(0, extra)
            gaps[i] += add
            remaining -= add
    random.shuffle(gaps)
    return gaps


def _build_frame(target_rooms: int, max_h: int = 22, max_w: int = 25) -> np.ndarray | None:
    if target_rooms == 1:
        gap_h = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_h - 2))
        gap_w = random.randint(_CELL_GAP_RANGE[0], max(_CELL_GAP_RANGE[0], max_w - 2))
        h, w = gap_h + 2, gap_w + 2
        walls = np.zeros((h, w), dtype=bool)
        walls[0, :] = True
        walls[h - 1, :] = True
        walls[:, 0] = True
        walls[:, w - 1] = True
        return walls

    possible_configs = []
    for nv in range(2, 6):
        for nh in range(2, 5):
            cells = (nv - 1) * (nh - 1)
            if target_rooms <= cells <= target_rooms + 4:
                possible_configs.append((nv, nh, cells))
    if not possible_configs:
        for nv in range(2, 7):
            for nh in range(2, 6):
                cells = (nv - 1) * (nh - 1)
                if cells >= target_rooms:
                    possible_configs.append((nv, nh, cells))
    if not possible_configs:
        return None

    n_vcols, n_hrows, total_cells = random.choice(possible_configs)
    n_row_gaps = n_hrows - 1
    n_col_gaps = n_vcols - 1
    min_gap, max_gap = _CELL_GAP_RANGE

    if min_gap * n_row_gaps > max_h - 1 or min_gap * n_col_gaps > max_w - 1:
        return None

    row_gaps = _random_gaps(n_row_gaps, min_gap, max_gap, max_h - 1)
    col_gaps = _random_gaps(n_col_gaps, min_gap, max_gap, max_w - 1)
    if row_gaps is None or col_gaps is None:
        return None

    vcol_positions = [0]
    for gap in col_gaps:
        vcol_positions.append(vcol_positions[-1] + gap)
    hrow_positions = [0]
    for gap in row_gaps:
        hrow_positions.append(hrow_positions[-1] + gap)

    frame_h = hrow_positions[-1] + 1
    frame_w = vcol_positions[-1] + 1
    walls = np.zeros((frame_h, frame_w), dtype=bool)
    for r in hrow_positions:
        walls[r, :] = True
    for c in vcol_positions:
        walls[:, c] = True

    cells_to_merge = total_cells - target_rooms
    if cells_to_merge == 0:
        return walls

    parent = list(range(total_cells))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> bool:
        px, py = find(x), find(y)
        if px == py:
            return False
        parent[px] = py
        return True

    def cell_id(cr: int, cc: int) -> int:
        return cr * (n_vcols - 1) + cc

    internal_segments = []
    for hi in range(1, n_hrows - 1):
        r = hrow_positions[hi]
        for ci in range(n_vcols - 1):
            c0, c1 = vcol_positions[ci], vcol_positions[ci + 1]
            internal_segments.append(("h", r, c0, c1, cell_id(hi - 1, ci), cell_id(hi, ci)))
    for ci in range(1, n_vcols - 1):
        c = vcol_positions[ci]
        for hi in range(n_hrows - 1):
            r0, r1 = hrow_positions[hi], hrow_positions[hi + 1]
            internal_segments.append(("v", c, r0, r1, cell_id(hi, ci - 1), cell_id(hi, ci)))

    random.shuffle(internal_segments)
    merged = 0
    for seg in internal_segments:
        if merged >= cells_to_merge:
            break
        if seg[0] == "h":
            _, r, c0, c1, ca, cb = seg
            if find(ca) != find(cb):
                union(ca, cb)
                walls[r, c0 + 1 : c1] = False
                merged += 1
        else:
            _, c, r0, r1, ca, cb = seg
            if find(ca) != find(cb):
                union(ca, cb)
                walls[r0 + 1 : r1, c] = False
                merged += 1

    if merged < cells_to_merge:
        return None
    if _count_enclosed_regions(walls) != target_rooms:
        return None
    return walls


def _build_distractor(grid_h: int, grid_w: int) -> np.ndarray | None:
    max_h = min(_DISTRACTOR_HEIGHT_RANGE[1], grid_h - 1)
    max_w = min(_DISTRACTOR_MAX_WIDTH, grid_w - 1)
    if max_h < _DISTRACTOR_HEIGHT_RANGE[0] or max_w < 8:
        return None

    for _ in range(100):
        h = random.randint(_DISTRACTOR_HEIGHT_RANGE[0], max_h)
        rung_count = random.randint(_DISTRACTOR_RUNG_COUNT_RANGE[0], min(_DISTRACTOR_RUNG_COUNT_RANGE[1], h - 2))
        backbone_col = random.randint(1, min(3, max_w - 5))
        min_w = max(8, backbone_col + 5)
        if min_w > max_w:
            continue
        w = random.randint(min_w, max_w)

        rows = [0, h - 1]
        if rung_count > 2:
            rows.extend(random.sample(list(range(2, h - 2)), rung_count - 2))
        rows = sorted(rows)

        target_pixels = int(grid_h * grid_w * random.uniform(*_DISTRACTOR_FRACTION_RANGE))
        target_pixels = max(target_pixels, h + 3 * rung_count)
        max_branch = w - backbone_col - 1
        if max_branch < 3:
            continue

        branch_lengths = [3] * rung_count
        remaining = min(target_pixels - h - sum(branch_lengths), rung_count * (max_branch - 3))
        remaining = max(0, remaining)
        order = list(range(rung_count))
        random.shuffle(order)
        for idx in order:
            if remaining <= 0:
                break
            cap = max_branch - branch_lengths[idx]
            add = random.randint(0, min(cap, remaining))
            branch_lengths[idx] += add
            remaining -= add
        while remaining > 0:
            changed = False
            for idx in order:
                if branch_lengths[idx] < max_branch:
                    branch_lengths[idx] += 1
                    remaining -= 1
                    changed = True
                    if remaining == 0:
                        break
            if not changed:
                break

        mask = np.zeros((h, w), dtype=bool)
        mask[:, backbone_col] = True
        lengths = [backbone_col + branch for branch in branch_lengths]
        random.shuffle(lengths)
        for r, end_c in zip(rows, lengths):
            mask[r, backbone_col : end_c + 1] = True

        if _count_enclosed_regions(_reconstruct_walls(mask)) == 0:
            return mask
    return None


def _bbox_conflicts(
    pos_a: tuple[int, int],
    shape_a: tuple[int, int],
    pos_b: tuple[int, int],
    shape_b: tuple[int, int],
    moat: int = 1,
) -> bool:
    ar0, ac0 = pos_a
    br0, bc0 = pos_b
    ah, aw = shape_a
    bh, bw = shape_b
    ar1, ac1 = ar0 + ah - 1, ac0 + aw - 1
    br1, bc1 = br0 + bh - 1, bc0 + bw - 1
    return not (ar1 + moat < br0 or br1 + moat < ar0 or ac1 + moat < bc0 or bc1 + moat < ac0)


def _place_frames(frames: list[tuple[int, int, np.ndarray]], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    order = sorted(range(len(frames)), key=lambda i: -frames[i][2].size)
    positions: list[tuple[int, int] | None] = [None] * len(frames)

    for _ in range(_FRAME_PLACEMENT_RETRY):
        failed = False
        positions = [None] * len(frames)
        for idx in order:
            fh, fw = frames[idx][2].shape
            choices = []
            for _trial in range(60):
                r0 = random.randint(0, grid_h - fh)
                c0 = random.randint(0, grid_w - fw)
                ok = True
                for j, pos_j in enumerate(positions):
                    if pos_j is None:
                        continue
                    if _bbox_conflicts((r0, c0), (fh, fw), pos_j, frames[j][2].shape, moat=1):
                        ok = False
                        break
                if ok:
                    choices.append((r0, c0))
            if not choices:
                failed = True
                break
            positions[idx] = random.choice(choices)
        if not failed and all(pos is not None for pos in positions):
            return [(int(pos[0]), int(pos[1])) for pos in positions if pos is not None]
    return None


def _absolute_mask(local_mask: np.ndarray, pos: tuple[int, int], grid_h: int, grid_w: int) -> np.ndarray:
    out = np.zeros((grid_h, grid_w), dtype=bool)
    rs, cs = np.where(local_mask)
    out[rs + pos[0], cs + pos[1]] = True
    return out


def _protected_mask(frames: list[tuple[int, int, np.ndarray]], positions: list[tuple[int, int]], grid_h: int, grid_w: int) -> np.ndarray:
    protected = np.zeros((grid_h, grid_w), dtype=bool)
    for (_color, _rooms, mask), (r0, c0) in zip(frames, positions):
        h, w = mask.shape
        anchors = {
            (0, 0),
            (0, w - 1),
            (h - 1, 0),
            (h - 1, w - 1),
            (0, w // 2),
            (h - 1, w // 2),
            (h // 2, 0),
            (h // 2, w - 1),
        }
        for rr, cc in anchors:
            if mask[rr, cc]:
                protected[r0 + rr, c0 + cc] = True
    return protected


def _place_distractor(
    distractor_mask: np.ndarray,
    frame_union: np.ndarray,
    protected_overlap: np.ndarray,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int] | None:
    dh, dw = distractor_mask.shape
    rs, cs = np.where(distractor_mask)
    lo, hi = _DISTRACTOR_TARGET_OVERLAP
    fallback_pos: tuple[int, int] | None = None
    fallback_score: tuple[int, int] | None = None

    for _ in range(_DISTRACTOR_PLACEMENT_RETRY):
        r0 = random.randint(0, grid_h - dh)
        c0 = random.randint(0, grid_w - dw)
        if int(protected_overlap[rs + r0, cs + c0].sum()) > 0:
            continue
        overlap = int(frame_union[rs + r0, cs + c0].sum())
        if lo <= overlap <= hi:
            return (r0, c0)
        if overlap <= 0:
            continue
        # Fall back only to light crossings near the target range; never prefer heavy occlusion.
        score = (max(0, overlap - hi), overlap)
        if fallback_score is None or score < fallback_score:
            fallback_score = score
            fallback_pos = (r0, c0)

    if fallback_pos is not None and fallback_score is not None and fallback_score[0] <= 2:
        return fallback_pos
    return None


def _render_shapes(shapes: list[tuple[int, np.ndarray, tuple[int, int]]], grid_h: int, grid_w: int) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for color, mask, pos in shapes:
        rs, cs = np.where(mask)
        grid[rs + pos[0], cs + pos[1]] = color
    return grid


def _expected_output(room_counts: list[int], frame_colors: list[int], pad_color: int) -> np.ndarray:
    ordered = sorted(zip(room_counts, frame_colors))
    width = max(rc for rc, _color in ordered)
    out = np.full((_FRAME_COUNT, width), pad_color, dtype=int)
    for r, (rooms, color) in enumerate(ordered):
        out[r, :rooms] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    room_counts = random.sample(list(range(_ROOM_COUNT_RANGE[0], _ROOM_COUNT_RANGE[1] + 1)), _FRAME_COUNT)
    frame_colors = random.sample(list(range(1, 10)), _FRAME_COUNT + 1)
    distractor_color = int(frame_colors[-1])
    frame_colors = frame_colors[:_FRAME_COUNT]

    grid_h = random.randint(*_GRID_H_RANGE)
    grid_w = random.randint(*_GRID_W_RANGE)

    frames: list[tuple[int, int, np.ndarray]] = []
    for color, target_rooms in zip(frame_colors, room_counts):
        frame_data = _build_frame(target_rooms, max_h=_FRAME_MAX_H, max_w=_FRAME_MAX_W)
        if frame_data is None:
            return None
        fh, fw = frame_data.shape
        if fh > grid_h - 2 or fw > grid_w - 2:
            return None
        frames.append((int(color), int(target_rooms), frame_data))

    positions = _place_frames(frames, grid_h, grid_w)
    if positions is None:
        return None

    distractor_mask = _build_distractor(grid_h, grid_w)
    if distractor_mask is None:
        return None

    real_abs_masks = [_absolute_mask(frames[i][2], positions[i], grid_h, grid_w) for i in range(_FRAME_COUNT)]
    frame_union = np.zeros((grid_h, grid_w), dtype=bool)
    for mask in real_abs_masks:
        frame_union |= mask
    protected = _protected_mask(frames, positions, grid_h, grid_w)

    distractor_pos = _place_distractor(distractor_mask, frame_union, protected, grid_h, grid_w)
    if distractor_pos is None:
        return None

    distractor_abs = _absolute_mask(distractor_mask, distractor_pos, grid_h, grid_w)
    if not bool((distractor_abs & frame_union).any()):
        return None

    shapes = [(frames[i][0], frames[i][2], positions[i]) for i in range(_FRAME_COUNT)]
    random.shuffle(shapes)
    shapes.append((distractor_color, distractor_mask, distractor_pos))
    grid = _render_shapes(shapes, grid_h, grid_w)

    try:
        output = main(grid)
    except Exception:
        return None

    expected = _expected_output(room_counts, frame_colors, distractor_color)
    return grid if np.array_equal(output, expected) else None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
