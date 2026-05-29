# Combo variant of 4c7dc4dd (index 28)
# combo_spec: D1/4+D3/1
# Name: 锚点参数化旋转
#
# Design analysis
# ---------------
# a) Concrete behavior trace:
#    - Base task: two same-palette framed sprites define one raw array transform, and
#      the third non-blank sprite receives the same transform.
#    - Fused variant: every non-blank sprite contains one shared anchor-color pixel.
#      A small orthogonal cue touching that anchor forms a fixed marker: one of two
#      L-shapes or a 3-cell T-like motif. That cue is NOT rotated; it encodes the payload's
#      rotation amount. The remaining body pixels stay at Chebyshev distance >= 2 from
#      the anchor and are rotated around that anchor.
#    - Example on a 5x5 interior: if the anchor is at (1,3) and the cue is at
#      offsets {(0,+1),(+1,0)}, the solver decodes "rotate payload by 90 degrees".
#      A payload pixel at offset (-2,-1) moves to (-1,+2), but the anchor and cue
#      remain fixed. This differs from the original whole-array symmetry logic and
#      also differs from the single D3/1 anchored-transform variant where every
#      non-anchor pixel would have rotated together.
# b) Complexity calibration:
#    - The transform family is restricted to quarter-turn counts {1,2,3}. This keeps
#      the parameter discrete and visible.
#    - The cue uses only three patterns:
#        * southeast L -> 90-degree clockwise payload rotation
#        * east-west bar plus south arm -> 180-degree rotation
#        * northeast L -> 270-degree clockwise payload rotation
#      The payload itself is generated asymmetrically and rejected if reversing the
#      training pair would also satisfy the same forward rule.
# c) Implicit assumptions checked against the old main():
#    - Old color-set matching still works only after removing the shared anchor color.
#    - Old array-level symmetry checks no longer apply because the rotation center is
#      an explicit pixel and the cue must stay fixed.
#    - Any logic that rotates all non-zero cells would be wrong here: the generator
#      reserves the 8-neighborhood of the anchor for cue-only structure, and the
#      solver explicitly preserves that structure while rotating only the payload.
# d) Placement legality:
#    - Frames remain non-overlapping and inside the 30x30 canvas.
#    - The anchor is chosen so both cue cells lie in bounds and enough payload offsets
#      stay valid before and after the encoded rotation.
#    - Payload cells are sampled only from offsets whose rotated images also stay in
#      bounds and avoid the protected anchor/cue neighborhood.
# e) Visual observability:
#    - Humans can spot the anchor immediately because it is the only color shared by
#      all three non-blank sprites and appears exactly once in each.
#    - The fixed cue touching the anchor is small, repeated, and visually distinct
#      from the farther payload. It exposes the angle parameter directly.
#    - The payload visibly swings around an off-center reference point, so the anchor
#      role is not cosmetic.
# f) Cleanliness:
#    - The cue occupies only a tiny subset of orthogonal neighbors of the anchor; the
#      rest of the 3x3 neighborhood around the anchor stays empty.
#    - Frame borders, anchors, cues, and payload cells never overlap.
# g) Inferability and anti-degeneration:
#    - The generator rejects samples where rotating the full non-anchor body would
#      accidentally match the target, because that would collapse back to the plain
#      anchored-rotation variant.
#    - The generator also rejects samples where changing the turn count would leave
#      the test output unchanged, so the parameter cue remains behaviorally active.
# New solver-side assumptions guaranteed by the generator:
#    - Each non-blank sprite has exactly one anchor pixel of the shared anchor color.
#    - The only non-zero cells in the anchor's 8-neighborhood are the cue cells.
#    - The same cue pattern appears in both training sprites and the test sprite.

from __future__ import annotations

import os
import random
import sys

import numpy as np

try:
    from arc_utils import colors_present, paste
except ImportError:
    sys.path.append(os.path.join(os.path.dirname(__file__), "../../../../solvers"))
    from arc_utils import colors_present, paste


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer an anchor-centered parameterized rotation from a framed training pair, then apply it to a test sprite.

    Input structure:
    - The grid contains four same-size hollow square frames on a noisy background, with one blank interior.
    - Each non-blank interior contains one shared anchor-color pixel and a same-palette body.
    - The body has a small fixed cue touching the anchor plus a farther payload that stays outside the anchor's 3x3 neighborhood.

    Rule:
    - Find the shared anchor color, ignore it when pairing the two training sprites by body palette, and treat the remaining non-blank frame as the test source.
    - Decode the rotation amount from the fixed cue pattern around the anchor, verify which training direction obeys "keep cue fixed, rotate only payload around the anchor", then apply that same anchored payload rotation to the test sprite.
    - The anchor and cue remain in place; only the farther body pixels move.

    Key insight:
    - The local pattern attached to the anchor is a visible parameter tag, not part of the rotated payload. The anchor supplies the rotation center, and the cue around it supplies the quarter-turn count.
    """
    g = np.asarray(grid, dtype=int)
    H, W = g.shape

    def find_square_frames():
        cands = []
        for size in range(4, min(H, W) + 1):
            for r0 in range(0, H - size + 1):
                r1 = r0 + size - 1
                for c0 in range(0, W - size + 1):
                    c1 = c0 + size - 1
                    col = int(g[r0, c0])
                    if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                        continue
                    if not (
                        np.all(g[r0, c0 : c1 + 1] == col)
                        and np.all(g[r1, c0 : c1 + 1] == col)
                        and np.all(g[r0 : r1 + 1, c0] == col)
                        and np.all(g[r0 : r1 + 1, c1] == col)
                    ):
                        continue
                    cands.append((size, col, r0, c0, r1, c1))
        cands.sort(key=lambda t: (-t[0], t[2], t[3]))
        picked = []
        for cand in cands:
            _, _, r0, c0, r1, c1 = cand
            overlaps = False
            for _, _, pr0, pc0, pr1, pc1 in picked:
                if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                    overlaps = True
                    break
            if not overlaps:
                picked.append(cand)
            if len(picked) >= 6:
                break
        return picked

    def rotate_offset(dr: int, dc: int, turns: int) -> tuple[int, int]:
        turns %= 4
        if turns == 0:
            return dr, dc
        if turns == 1:
            return dc, -dr
        if turns == 2:
            return -dr, -dc
        return -dc, dr

    marker_to_turns = {
        frozenset({(0, 1), (1, 0)}): 1,
        frozenset({(0, -1), (0, 1), (1, 0)}): 2,
        frozenset({(-1, 0), (0, 1)}): 3,
    }

    def anchor_pos(arr: np.ndarray, anchor_color: int) -> tuple[int, int] | None:
        pts = np.argwhere(arr == anchor_color)
        if pts.shape != (1, 2):
            return None
        return int(pts[0, 0]), int(pts[0, 1])

    def decode_marker(arr: np.ndarray, anchor_color: int):
        pos = anchor_pos(arr, anchor_color)
        if pos is None:
            return None
        ar, ac = pos
        h, w = arr.shape
        offsets = []
        marker_colors = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            rr, cc = ar + dr, ac + dc
            if 0 <= rr < h and 0 <= cc < w and int(arr[rr, cc]) != 0:
                if int(arr[rr, cc]) == anchor_color:
                    return None
                offsets.append((dr, dc))
                marker_colors.append(int(arr[rr, cc]))
        offset_set = frozenset(offsets)
        turns = marker_to_turns.get(offset_set)
        if turns is None or len(marker_colors) != len(offset_set):
            return None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                rr, cc = ar + dr, ac + dc
                if not (0 <= rr < h and 0 <= cc < w):
                    continue
                if int(arr[rr, cc]) == 0:
                    continue
                if (dr, dc) not in offset_set:
                    return None
        return pos, turns, offset_set

    def apply_param_rotation(arr: np.ndarray, anchor_color: int, turns: int) -> np.ndarray | None:
        marker_info = decode_marker(arr, anchor_color)
        if marker_info is None:
            return None
        (ar, ac), decoded_turns, marker_offsets = marker_info
        if decoded_turns != turns:
            return None
        h, w = arr.shape
        protected = {(0, 0)} | set(marker_offsets)
        out = np.zeros_like(arr)
        out[ar, ac] = anchor_color
        for dr, dc in marker_offsets:
            rr, cc = ar + dr, ac + dc
            out[rr, cc] = arr[rr, cc]
        for r, c in np.argwhere(arr != 0):
            rr0 = int(r)
            cc0 = int(c)
            col = int(arr[rr0, cc0])
            dr, dc = rr0 - ar, cc0 - ac
            if (dr, dc) in protected:
                continue
            ndr, ndc = rotate_offset(dr, dc, turns)
            if max(abs(dr), abs(dc)) < 2 or max(abs(ndr), abs(ndc)) < 2:
                return None
            rr, cc = ar + ndr, ac + ndc
            if not (0 <= rr < h and 0 <= cc < w):
                return None
            if out[rr, cc] not in (0, col):
                return None
            out[rr, cc] = col
        return out

    def infer_turns(src: np.ndarray, dst: np.ndarray, anchor_color: int) -> int | None:
        src_info = decode_marker(src, anchor_color)
        dst_info = decode_marker(dst, anchor_color)
        if src_info is None or dst_info is None:
            return None
        src_pos, turns, src_marker = src_info
        dst_pos, dst_turns, dst_marker = dst_info
        if src_pos != dst_pos or turns != dst_turns or src_marker != dst_marker:
            return None
        ar, ac = src_pos
        for dr, dc in src_marker:
            rr, cc = ar + dr, ac + dc
            if int(src[rr, cc]) != int(dst[rr, cc]):
                return None
        pred = apply_param_rotation(src, anchor_color, turns)
        if pred is None or not np.array_equal(pred, dst):
            return None
        return turns

    frames = find_square_frames()
    if len(frames) < 3:
        return g.copy()

    inners = [g[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]

    blank_idx = None
    for i, inner in enumerate(inners):
        if inner.size > 0 and np.all(inner == 0):
            blank_idx = i
            break
    if blank_idx is None:
        blank_idx = int(np.argmin([np.count_nonzero(inner) for inner in inners]))

    nonblank = [i for i in range(len(inners)) if i != blank_idx]
    if len(nonblank) < 2:
        return inners[blank_idx].copy()

    shared = None
    for i in nonblank:
        cols = {int(v) for v in np.unique(inners[i]) if int(v) != 0}
        shared = cols if shared is None else shared & cols
    anchor_candidates = [col for col in sorted(shared or []) if all(np.sum(inners[i] == col) == 1 for i in nonblank)]
    if len(anchor_candidates) != 1:
        return inners[nonblank[0]].copy()
    anchor_color = anchor_candidates[0]

    sigs = []
    for i in nonblank:
        body_cols = sorted(set(int(x) for x in inners[i].ravel() if int(x) not in (0, anchor_color)))
        sigs.append((i, tuple(body_cols)))

    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        counts = [
            (i, int(np.count_nonzero((inners[i] != 0) & (inners[i] != anchor_color))))
            for i in nonblank
        ]
        counts.sort(key=lambda t: t[1])
        pair = (counts[0][0], counts[1][0])

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    src_a = inners[pair[0]]
    src_b = inners[pair[1]]
    test_inner = inners[test_idx]

    best_turns = None
    for src, dst in ((src_a, src_b), (src_b, src_a)):
        turns = infer_turns(src, dst, anchor_color)
        if turns is not None:
            best_turns = turns
            break
    if best_turns is None:
        return test_inner.copy()

    pred = apply_param_rotation(test_inner, anchor_color, best_turns)
    if pred is None:
        return test_inner.copy()
    return pred


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8

TURN_CHOICES = (1, 2, 3)
BODY_COLOR_COUNT_CHOICES = (1, 2)
PAYLOAD_MIN = 3
PAYLOAD_MAX = 9
BUILD_RETRIES = 160
GEN_MAX_ATTEMPTS = 500

TURN_MARKERS = {
    1: ((0, 1), (1, 0)),
    2: ((0, -1), (0, 1), (1, 0)),
    3: ((-1, 0), (0, 1)),
}


def find_square_frames(grid: np.ndarray) -> list[tuple[int, int, int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    cands = []
    for size in range(4, min(h, w) + 1):
        for r0 in range(0, h - size + 1):
            r1 = r0 + size - 1
            for c0 in range(0, w - size + 1):
                c1 = c0 + size - 1
                col = int(g[r0, c0])
                if int(g[r0, c1]) != col or int(g[r1, c0]) != col or int(g[r1, c1]) != col:
                    continue
                if not (
                    np.all(g[r0, c0 : c1 + 1] == col)
                    and np.all(g[r1, c0 : c1 + 1] == col)
                    and np.all(g[r0 : r1 + 1, c0] == col)
                    and np.all(g[r0 : r1 + 1, c1] == col)
                ):
                    continue
                cands.append((size, col, r0, c0, r1, c1))
    cands.sort(key=lambda t: (-t[0], t[2], t[3]))
    picked = []
    for cand in cands:
        _, _, r0, c0, r1, c1 = cand
        overlaps = False
        for _, _, pr0, pc0, pr1, pc1 in picked:
            if not (r1 < pr0 or pr1 < r0 or c1 < pc0 or pc1 < c0):
                overlaps = True
                break
        if not overlaps:
            picked.append(cand)
    return picked


def rotate_offset(dr: int, dc: int, turns: int) -> tuple[int, int]:
    turns %= 4
    if turns == 0:
        return dr, dc
    if turns == 1:
        return dc, -dr
    if turns == 2:
        return -dr, -dc
    return -dc, dr


def anchor_pos(arr: np.ndarray, anchor_color: int) -> tuple[int, int] | None:
    pts = np.argwhere(arr == anchor_color)
    if pts.shape != (1, 2):
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def decode_marker(arr: np.ndarray, anchor_color: int):
    pos = anchor_pos(arr, anchor_color)
    if pos is None:
        return None
    ar, ac = pos
    h, w = arr.shape
    offsets = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = ar + dr, ac + dc
        if 0 <= rr < h and 0 <= cc < w and int(arr[rr, cc]) != 0:
            if int(arr[rr, cc]) == anchor_color:
                return None
            offsets.append((dr, dc))
    offset_set = frozenset(offsets)
    turns = None
    for k, marker in TURN_MARKERS.items():
        if offset_set == frozenset(marker):
            turns = k
            break
    if turns is None or len(offsets) != len(offset_set):
        return None
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            rr, cc = ar + dr, ac + dc
            if not (0 <= rr < h and 0 <= cc < w):
                continue
            if int(arr[rr, cc]) == 0:
                continue
            if (dr, dc) not in offset_set:
                return None
    return pos, turns, offset_set


def apply_param_rotation(arr: np.ndarray, anchor_color: int, turns: int) -> np.ndarray | None:
    marker_info = decode_marker(arr, anchor_color)
    if marker_info is None:
        return None
    (ar, ac), decoded_turns, marker_offsets = marker_info
    if decoded_turns != turns:
        return None
    h, w = arr.shape
    protected = {(0, 0)} | set(marker_offsets)
    out = np.zeros_like(arr)
    out[ar, ac] = anchor_color
    for dr, dc in marker_offsets:
        out[ar + dr, ac + dc] = arr[ar + dr, ac + dc]
    for r, c in np.argwhere(arr != 0):
        rr0 = int(r)
        cc0 = int(c)
        col = int(arr[rr0, cc0])
        dr, dc = rr0 - ar, cc0 - ac
        if (dr, dc) in protected:
            continue
        ndr, ndc = rotate_offset(dr, dc, turns)
        if max(abs(dr), abs(dc)) < 2 or max(abs(ndr), abs(ndc)) < 2:
            return None
        rr, cc = ar + ndr, ac + ndc
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if out[rr, cc] not in (0, col):
            return None
        out[rr, cc] = col
    return out


def infer_turns(src: np.ndarray, dst: np.ndarray, anchor_color: int) -> int | None:
    src_info = decode_marker(src, anchor_color)
    dst_info = decode_marker(dst, anchor_color)
    if src_info is None or dst_info is None:
        return None
    src_pos, turns, src_marker = src_info
    dst_pos, dst_turns, dst_marker = dst_info
    if src_pos != dst_pos or turns != dst_turns or src_marker != dst_marker:
        return None
    ar, ac = src_pos
    for dr, dc in src_marker:
        if int(src[ar + dr, ac + dc]) != int(dst[ar + dr, ac + dc]):
            return None
    pred = apply_param_rotation(src, anchor_color, turns)
    if pred is None or not np.array_equal(pred, dst):
        return None
    return turns


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def assign_body_colors(count: int, colors: list[int], rng: random.Random) -> list[int]:
    if len(colors) == 1:
        return [colors[0]] * count
    assigned = [rng.choice(colors) for _ in range(count)]
    assigned[0] = colors[0]
    assigned[1] = colors[1]
    rng.shuffle(assigned)
    return assigned


def safe_payload_offsets(n: int, anchor: tuple[int, int], turns: int, marker_offsets: tuple[tuple[int, int], ...]) -> list[tuple[int, int]]:
    ar, ac = anchor
    marker_set = set(marker_offsets)
    cells = []
    for dr in range(-(n - 1), n):
        for dc in range(-(n - 1), n):
            if (dr, dc) == (0, 0):
                continue
            if max(abs(dr), abs(dc)) < 2:
                continue
            rr, cc = ar + dr, ac + dc
            if not (0 <= rr < n and 0 <= cc < n):
                continue
            ndr, ndc = rotate_offset(dr, dc, turns)
            if max(abs(ndr), abs(ndc)) < 2:
                continue
            tr, tc = ar + ndr, ac + ndc
            if not (0 <= tr < n and 0 <= tc < n):
                continue
            if (dr, dc) in marker_set or (ndr, ndc) in marker_set:
                continue
            cells.append((dr, dc))
    return cells


def choose_anchor(n: int, turns: int, marker_offsets: tuple[tuple[int, int], ...], rng: random.Random) -> tuple[int, int] | None:
    center = (n // 2, n // 2)
    candidates = []
    for ar in range(n):
        for ac in range(n):
            ok = True
            for dr, dc in marker_offsets:
                rr, cc = ar + dr, ac + dc
                if not (0 <= rr < n and 0 <= cc < n):
                    ok = False
                    break
            if not ok:
                continue
            safe = safe_payload_offsets(n, (ar, ac), turns, marker_offsets)
            if len(safe) >= PAYLOAD_MIN:
                candidates.append((ar, ac, len(safe)))
    if not candidates:
        return None
    candidates.sort(key=lambda t: (-(abs(t[0] - center[0]) + abs(t[1] - center[1])), -t[2]))
    off_center = [t for t in candidates if (t[0], t[1]) != center]
    pool = off_center if off_center else candidates
    best_band = pool[: max(1, min(len(pool), 6))]
    ar, ac, _ = rng.choice(best_band)
    return ar, ac


def build_sprite(
    n: int,
    body_colors: list[int],
    anchor_color: int,
    turns: int,
    rng: random.Random,
    required_anchor: tuple[int, int] | None = None,
):
    marker_offsets = TURN_MARKERS[turns]
    for _ in range(BUILD_RETRIES):
        anchor = required_anchor
        if anchor is None:
            anchor = choose_anchor(n, turns, marker_offsets, rng)
        if anchor is None:
            continue
        safe = safe_payload_offsets(n, anchor, turns, marker_offsets)
        if len(safe) < PAYLOAD_MIN:
            continue
        count = rng.randint(PAYLOAD_MIN, min(PAYLOAD_MAX, len(safe)))
        chosen = rng.sample(safe, count)
        src = np.zeros((n, n), dtype=int)
        ar, ac = anchor
        src[ar, ac] = anchor_color
        marker_color = body_colors[0]
        for dr, dc in marker_offsets:
            src[ar + dr, ac + dc] = marker_color
        for (dr, dc), col in zip(chosen, assign_body_colors(count, body_colors, rng)):
            src[ar + dr, ac + dc] = col
        dst = apply_param_rotation(src, anchor_color, turns)
        if dst is None or np.array_equal(src, dst):
            continue
        if infer_turns(src, dst, anchor_color) != turns:
            continue
        reverse_ok = infer_turns(dst, src, anchor_color) is not None
        if turns != 2 and reverse_ok:
            continue
        full_rot = rotate_all_non_anchor(src, anchor_color, turns)
        if full_rot is not None and np.array_equal(full_rot, dst):
            continue
        return src, dst, anchor
    return None


def rotate_all_non_anchor(arr: np.ndarray, anchor_color: int, turns: int) -> np.ndarray | None:
    pos = anchor_pos(arr, anchor_color)
    if pos is None:
        return None
    ar, ac = pos
    h, w = arr.shape
    out = np.zeros_like(arr)
    out[ar, ac] = anchor_color
    for r, c in np.argwhere(arr != 0):
        rr0 = int(r)
        cc0 = int(c)
        col = int(arr[rr0, cc0])
        if col == anchor_color:
            continue
        dr, dc = rr0 - ar, cc0 - ac
        ndr, ndc = rotate_offset(dr, dc, turns)
        rr, cc = ar + ndr, ac + ndc
        if not (0 <= rr < h and 0 <= cc < w):
            return None
        if out[rr, cc] not in (0, col):
            return None
        out[rr, cc] = col
    return out


def quadrant_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    low, high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    return [
        (rng.randint(low, high), rng.randint(low, high)),
        (rng.randint(low, high), rng.randint(bottom_low, bottom_high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(low, high)),
        (rng.randint(bottom_low, bottom_high), rng.randint(bottom_low, bottom_high)),
    ]


def render_background(
    frame_boxes: list[tuple[int, int, int, int]],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    period = len(bg_palette)
    direction = rng.choice([1, -1])
    rr, cc = np.indices((H, W))
    idx = (rr + direction * cc) % period
    grid = np.asarray(bg_palette, dtype=int)[idx]

    reserved = np.zeros((H, W), dtype=bool)
    for r0, c0, r1, c1 in frame_boxes:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True
    bg_cells = [(int(r), int(c)) for r, c in zip(*np.where(~reserved))]
    perturb_n = int(round(len(bg_cells) * rng.uniform(*BG_PERTURB_FRACTION)))
    noise_choices = list(bg_palette) + [c for c in BG_EXTRA_NOISE if c not in bg_palette]
    for r, c in rng.sample(bg_cells, perturb_n):
        grid[r, c] = rng.choice(noise_choices)
    return grid


def draw_frame(grid: np.ndarray, r0: int, c0: int, s: int, frame_color: int, sprite: np.ndarray) -> None:
    r1, c1 = r0 + s - 1, c0 + s - 1
    grid[r0 : r1 + 1, c0 : c1 + 1] = 0
    grid[r0, c0 : c1 + 1] = frame_color
    grid[r1, c0 : c1 + 1] = frame_color
    grid[r0 : r1 + 1, c0] = frame_color
    grid[r0 : r1 + 1, c1] = frame_color
    paste(grid, sprite, r0 + 1, c0 + 1, transparent=0)


def choose_frame_and_bg(used: set[int], rng: random.Random) -> tuple[int, list[int]]:
    frame_color = rng.choice([c for c in ALL_COLORS if c not in used])
    bg_pool = [c for c in ALL_COLORS if c not in used and c != frame_color]
    size_choices = [k for k in BG_PERIOD_CHOICES if k <= len(bg_pool)]
    if not size_choices:
        raise ValueError("not enough colors for background palette")
    bg_palette = rng.sample(bg_pool, rng.choice(size_choices))
    return frame_color, bg_palette


def assign_slots(positions: list[tuple[int, int]], rng: random.Random) -> tuple[list[int], int]:
    order = sorted(range(4), key=lambda i: positions[i])
    train_slots = sorted(rng.sample(order, 2), key=order.index)
    remaining = [i for i in order if i not in train_slots]
    test_slot = rng.choice(remaining)
    return train_slots, test_slot


def render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    return grid


def validate_scene(grid: np.ndarray, expected_output: np.ndarray) -> bool:
    frames = find_square_frames(grid)
    if len(frames) != 4:
        return False
    sizes = {size for size, _, *_ in frames}
    if len(sizes) != 1 or next(iter(sizes)) not in set(FRAME_SIZE_CHOICES):
        return False
    inners = [grid[r0 + 1 : r1, c0 + 1 : c1] for _, _, r0, c0, r1, c1 in frames]
    if sum(int(np.all(inner == 0)) for inner in inners) != 1:
        return False
    nonblank = [inner for inner in inners if not np.all(inner == 0)]
    if len(nonblank) != 3:
        return False
    shared = None
    for inner in nonblank:
        cols = set(colors_present(inner, exclude={0}))
        shared = cols if shared is None else shared & cols
    anchor_candidates = [col for col in sorted(shared or []) if all(np.sum(inner == col) == 1 for inner in nonblank)]
    if len(anchor_candidates) != 1:
        return False
    anchor_color = anchor_candidates[0]
    body_sigs = [
        tuple(sorted(c for c in colors_present(inner, exclude={0}) if c != anchor_color))
        for inner in nonblank
    ]
    dup_count = sum(
        1 for i in range(len(body_sigs)) for j in range(i + 1, len(body_sigs))
        if body_sigs[i] == body_sigs[j]
    )
    if dup_count != 1:
        return False
    if any(decode_marker(inner, anchor_color) is None for inner in nonblank):
        return False
    return np.array_equal(main(grid), expected_output)


def gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2
    turns = rng.choice(TURN_CHOICES)
    body_color_count = rng.choice(BODY_COLOR_COUNT_CHOICES)

    train_colors = choose_palette(rng, body_color_count, set())
    test_colors = choose_palette(rng, body_color_count, set(train_colors))
    anchor_color = choose_palette(rng, 1, set(train_colors) | set(test_colors))[0]

    built_train = build_sprite(n, train_colors, anchor_color, turns, rng)
    if built_train is None:
        return None
    train_src, train_dst, anchor = built_train

    built_test = build_sprite(n, test_colors, anchor_color, turns, rng, required_anchor=anchor)
    if built_test is None:
        return None
    test_src, test_dst, _ = built_test

    alt_matches = 0
    for alt_turns in TURN_CHOICES:
        pred = apply_param_rotation(test_src, anchor_color, alt_turns)
        if pred is not None and np.array_equal(pred, test_dst):
            alt_matches += 1
    if alt_matches != 1:
        return None

    naive = rotate_all_non_anchor(test_src, anchor_color, turns)
    if naive is not None and np.array_equal(naive, test_dst):
        return None

    used = set(train_colors) | set(test_colors) | {anchor_color}
    frame_color, bg_palette = choose_frame_and_bg(used, rng)

    positions = quadrant_positions(s, rng)
    train_slots, test_slot = assign_slots(positions, rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_slots[0]] = train_src
    sprites[train_slots[1]] = train_dst
    sprites[test_slot] = test_src

    grid = render_scene(s, positions, sprites, frame_color, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
