# Combo variant of 4c7dc4dd (index 9)
# combo_spec: D3/1+D5/2+D11/0
# Name: 锚点引导的定向边界提取
#
# Design analysis
# ---------------
# concretization_and_trace:
#   - The base scene stays close to the original task: a 30x30 noisy background,
#     exactly 4 hollow square frames of one shared size, one blank interior, one
#     training pair, and one test sprite.
#   - D3/1 adds one anchor pixel of a dedicated color to every non-blank interior.
#     The anchor is the only non-zero color shared by all three non-blank frames,
#     appears exactly once per sprite, and stays fixed from source to target.
#   - D5/2 replaces the old 6-way rule family with one fixed rule:
#     extract the 4-connected outer boundary of the sprite's body. The body is the
#     set of non-zero, non-anchor pixels. The anchor survives unchanged but is
#     excluded from the flood-fill boundary computation.
#   - D11/0 removes direction guessing. A thin arrow is drawn in the corridor
#     between the two training frames and points from source frame to destination
#     frame.
#   - Concrete trace:
#     original:
#       1. pair frames are matched by palette
#       2. solver tries both A->B and B->A
#       3. one of several inner transforms is inferred
#     combo:
#       1. find the anchor color from the shared singleton color across all
#          non-blank interiors
#       2. match the training pair by body palette after removing the anchor color
#       3. decode the arrow so direction is fixed before rule inference
#       4. run boundary extraction only on body pixels; keep the anchor as a
#          stable marker in place
#   - Example difference:
#     A notched 4x4 block with an anchor plugged into the notch looks almost filled
#     in the source. The target keeps the anchor but outlines the body as if that
#     anchor cell were empty. This distinguishes "anchor is a stable marker" from
#     "anchor is just another foreground pixel".
#
# complexity_calibration:
#   - I keep only one inner rule, but I make the anchor materially relevant by
#     generating a body notch under the anchor. If the anchor were incorrectly
#     treated as normal foreground, the extracted body boundary would change.
#   - The body palette is restricted to one color per sprite. This is deliberate:
#     after adding a dedicated anchor color, an arrow color, a frame color, and a
#     4-5 color background, allowing two disjoint body palettes would overconsume
#     the 10 ARC colors and force uglier compromises elsewhere.
#   - Training/test sprites still use different body colors so pair matching by
#     body palette remains visible and unambiguous.
#
# implicit_assumptions_audit:
#   - find_square_frames() still works because the arrow never touches a frame.
#   - Blank-frame detection still works because only one interior is all zero.
#   - The original full-palette pair matching is no longer valid because the anchor
#     color is shared by all non-blank frames; the solver must remove the anchor
#     color before comparing signatures.
#   - The original bidirectional rule search is invalid because direction is now
#     explicit and should be decoded from the arrow instead of guessed.
#   - Boundary extraction must ignore the anchor color; otherwise the training rule
#     can silently change on notched samples.
#   - New solver assumptions introduced here:
#     * exactly one anchor color is shared by all non-blank interiors
#     * the training pair is axis-adjacent with a unique arrow component in its gap
#     * the anchor position is unchanged within the training pair
#     The generator enforces all three.
#
# placement_legality:
#   - Frames are placed on a 2x2 structured layout with a guaranteed minimum row
#     and column gap, so the arrow fits cleanly between adjacent training frames.
#   - The arrow is drawn only in the corridor between the chosen pair and never
#     overlaps any frame border or interior.
#   - The body sprite is built inside the n x n interior; the anchor is chosen from
#     one carved notch cell inside that same interior, so no clipping is needed.
#
# visual_observability_and_cleanliness:
#   - The anchor is easy to spot because it is the only non-zero color repeated
#     across all non-blank frames, and it remains fixed while nearby body pixels
#     thin to an outline.
#   - The arrow is a separate unique-color component outside all frames, so it does
#     not compete visually with sprite pixels.
#   - The notched thick body makes boundary extraction visibly 2D rather than a
#     degenerate endpoint task, even on 4x4 to 6x6 interiors.
#   - The body-under-anchor notch guarantees that the anchor does not behave like
#     decorative stable noise: it affects how a wrong solver would trace boundary.
#
# inferability_of_change:
#   - No hidden color ordering is used. Color roles are structural:
#     shared singleton = anchor, isolated corridor component = direction marker,
#     duplicated body palette = training pair.
#   - The rule can be inferred from 2-3 examples because every generated sample
#     repeats the same joint pattern: arrow fixes direction, anchor stays fixed,
#     body collapses to an outer contour while ignoring the anchor cell.

from __future__ import annotations

from collections import deque
import random

import numpy as np

from arc_utils import bbox_coords, colors_present, components_mask, draw_line, paste


H = W = 30

SPRITE_COLORS = [1, 2, 3, 4, 5, 6, 8]
ALL_COLORS = list(range(10))
BG_PERIOD_CHOICES = (4, 5)
BG_PERTURB_FRACTION = (0.30, 0.50)
BG_EXTRA_NOISE = (0,)

FRAME_SIZE_CHOICES = (6, 7, 8)
QUAD_LOW_MIN, QUAD_LOW_MAX = 1, 8
QUAD_BOTTOM_MARGIN = 8
MIN_AXIS_GAP = 4
TRAIN_PAIR_CHOICES = ((0, 1), (2, 3), (0, 2), (1, 3))

SPRITE_NZ_MIN, SPRITE_NZ_MAX = 3, 32
BODY_SRC_MIN = 8
BODY_DST_MIN = 6
BODY_INTERIOR_REMOVED_MIN = 2
BODY_BUILD_RETRIES = 140
SECOND_RECT_PROB = 0.75
GEN_MAX_ATTEMPTS = 500


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


def anchor_pos(arr: np.ndarray, anchor_color: int) -> tuple[int, int] | None:
    pts = np.argwhere(arr == anchor_color)
    if pts.shape != (1, 2):
        return None
    return int(pts[0, 0]), int(pts[0, 1])


def exterior_zero_mask(body: np.ndarray) -> np.ndarray:
    h, w = body.shape
    ext = np.zeros((h, w), dtype=bool)
    q: deque[tuple[int, int]] = deque()

    def seed(r: int, c: int) -> None:
        if not body[r, c] and not ext[r, c]:
            ext[r, c] = True
            q.append((r, c))

    for r in range(h):
        seed(r, 0)
        seed(r, w - 1)
    for c in range(w):
        seed(0, c)
        seed(h - 1, c)

    while q:
        r, c = q.popleft()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < h and 0 <= nc < w and not body[nr, nc] and not ext[nr, nc]:
                ext[nr, nc] = True
                q.append((nr, nc))
    return ext


def body_boundary_mask(body: np.ndarray) -> np.ndarray:
    h, w = body.shape
    if not np.any(body):
        return np.zeros_like(body)
    ext = exterior_zero_mask(body)
    keep = np.zeros((h, w), dtype=bool)
    rr, cc = np.where(body)
    for r, c in zip(rr.tolist(), cc.tolist()):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if nr < 0 or nr >= h or nc < 0 or nc >= w or ext[nr, nc]:
                keep[r, c] = True
                break
    return keep


def boundary_trace(arr: np.ndarray, anchor_color: int) -> np.ndarray:
    body = (arr != 0) & (arr != anchor_color)
    keep = body_boundary_mask(body)
    out = np.zeros_like(arr)
    out[arr == anchor_color] = anchor_color
    out[keep] = arr[keep]
    return out


def boundary_trace_all_nonzero(arr: np.ndarray) -> np.ndarray:
    body = arr != 0
    keep = body_boundary_mask(body)
    out = np.zeros_like(arr)
    out[keep] = arr[keep]
    return out


def infer_boundary_rule(src: np.ndarray, dst: np.ndarray, anchor_color: int) -> bool:
    src_anchor = anchor_pos(src, anchor_color)
    dst_anchor = anchor_pos(dst, anchor_color)
    if src_anchor is None or dst_anchor is None or src_anchor != dst_anchor:
        return False
    return np.array_equal(boundary_trace(src, anchor_color), dst)


def find_anchor_color(inners: list[np.ndarray], nonblank: list[int]) -> int | None:
    shared = None
    for idx in nonblank:
        cols = {int(v) for v in np.unique(inners[idx]) if int(v) != 0}
        shared = cols if shared is None else shared & cols
    candidates = []
    for col in sorted(shared or []):
        if all(np.sum(inners[idx] == col) == 1 for idx in nonblank):
            candidates.append(col)
    if len(candidates) != 1:
        return None
    return candidates[0]


def body_signature(arr: np.ndarray, anchor_color: int) -> tuple[int, ...]:
    return tuple(sorted(int(v) for v in np.unique(arr) if int(v) not in (0, anchor_color)))


def _pair_axis(box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int]) -> str:
    ra = (box_a[0] + box_a[2]) / 2.0
    ca = (box_a[1] + box_a[3]) / 2.0
    rb = (box_b[0] + box_b[2]) / 2.0
    cb = (box_b[1] + box_b[3]) / 2.0
    return "horizontal" if abs(ca - cb) >= abs(ra - rb) else "vertical"


def _direction_from_component(pts: list[tuple[int, int]], axis: str) -> str | None:
    if not pts:
        return None
    if axis == "horizontal":
        col_counts: dict[int, int] = {}
        for _r, c in pts:
            col_counts[c] = col_counts.get(c, 0) + 1
        cols = sorted(col_counts)
        if len(cols) < 2:
            return None
        if col_counts[cols[-1]] == 1 and col_counts[cols[-2]] >= 2:
            return "right"
        if col_counts[cols[0]] == 1 and col_counts[cols[1]] >= 2:
            return "left"
        return None
    row_counts: dict[int, int] = {}
    for r, _c in pts:
        row_counts[r] = row_counts.get(r, 0) + 1
    rows = sorted(row_counts)
    if len(rows) < 2:
        return None
    if row_counts[rows[-1]] == 1 and row_counts[rows[-2]] >= 2:
        return "down"
    if row_counts[rows[0]] == 1 and row_counts[rows[1]] >= 2:
        return "up"
    return None


def decode_training_direction(
    grid: np.ndarray,
    frames: list[tuple[int, int, int, int, int, int]],
    pair: tuple[int, int],
) -> tuple[int, int] | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    reserved = np.zeros((h, w), dtype=bool)
    for _, _, r0, c0, r1, c1 in frames:
        reserved[r0 : r1 + 1, c0 : c1 + 1] = True

    boxes = {
        idx: (frames[idx][2], frames[idx][3], frames[idx][4], frames[idx][5])
        for idx in pair
    }
    box_a = boxes[pair[0]]
    box_b = boxes[pair[1]]
    axis = _pair_axis(box_a, box_b)
    candidates = []

    if axis == "horizontal":
        left_idx, right_idx = sorted(pair, key=lambda i: boxes[i][1])
        left_box = boxes[left_idx]
        right_box = boxes[right_idx]
        gap_c0 = left_box[3] + 1
        gap_c1 = right_box[1] - 1
        mid_r = (left_box[0] + left_box[2]) // 2
        row_lo = max(0, mid_r - 1)
        row_hi = min(h - 1, mid_r + 1)
        for color in colors_present(g, exclude={0}):
            comps = components_mask((g == color) & ~reserved, connectivity=4)
            for comp in comps:
                if comp.size == 0:
                    continue
                pts = [(int(r), int(c)) for r, c in comp]
                r0, c0, r1, c1 = bbox_coords(comp)
                if c0 < gap_c0 or c1 > gap_c1:
                    continue
                if r0 < row_lo or r1 > row_hi:
                    continue
                if (r1 - r0 + 1) > 3 or (c1 - c0 + 1) < 2:
                    continue
                candidates.append((len(pts), pts))
        if not candidates:
            return None
        pts = max(candidates, key=lambda t: t[0])[1]
        arrow = _direction_from_component(pts, axis)
        if arrow == "right":
            return left_idx, right_idx
        if arrow == "left":
            return right_idx, left_idx
        return None

    top_idx, bottom_idx = sorted(pair, key=lambda i: boxes[i][0])
    top_box = boxes[top_idx]
    bottom_box = boxes[bottom_idx]
    gap_r0 = top_box[2] + 1
    gap_r1 = bottom_box[0] - 1
    mid_c = (top_box[1] + top_box[3]) // 2
    col_lo = max(0, mid_c - 1)
    col_hi = min(w - 1, mid_c + 1)
    for color in colors_present(g, exclude={0}):
        comps = components_mask((g == color) & ~reserved, connectivity=4)
        for comp in comps:
            if comp.size == 0:
                continue
            pts = [(int(r), int(c)) for r, c in comp]
            r0, c0, r1, c1 = bbox_coords(comp)
            if r0 < gap_r0 or r1 > gap_r1:
                continue
            if c0 < col_lo or c1 > col_hi:
                continue
            if (c1 - c0 + 1) > 3 or (r1 - r0 + 1) < 2:
                continue
            candidates.append((len(pts), pts))
    if not candidates:
        return None
    pts = max(candidates, key=lambda t: t[0])[1]
    arrow = _direction_from_component(pts, axis)
    if arrow == "down":
        return top_idx, bottom_idx
    if arrow == "up":
        return bottom_idx, top_idx
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Learn an anchor-aware contour rule from a framed training pair with an explicit arrow, then solve the test sprite.

    Input structure:
    - The board contains four same-size hollow square frames on a noisy background.
    - One interior is blank, two non-blank interiors form the training pair, and the remaining non-blank interior is the test source.
    - Every non-blank sprite contains exactly one shared-color anchor pixel, and a thin arrow outside the frames points from the training source frame to the training destination frame.

    Rule:
    - Detect the anchor as the only non-zero singleton color common to all non-blank interiors, and match the training pair by body palette after excluding that anchor color.
    - Use the arrow to decide the training direction, then verify that the destination equals the 4-connected outer boundary of the source body, where the body means all non-zero pixels except the anchor.
    - Apply that same anchor-excluding boundary extraction to the test sprite while keeping the test anchor fixed in place.

    Key insight:
    - The anchor is not ordinary foreground. It stays visible but should be mentally removed before tracing the body's outline, and the arrow removes any need to guess which training frame is the source.
    """
    g = np.asarray(grid, dtype=int)
    frames = find_square_frames(g)
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
    if len(nonblank) < 3:
        return inners[blank_idx].copy()

    anchor_color = find_anchor_color(inners, nonblank)
    if anchor_color is None:
        return inners[nonblank[0]].copy()

    sigs = [(i, body_signature(inners[i], anchor_color)) for i in nonblank]
    pair = None
    for a in range(len(sigs)):
        for b in range(a + 1, len(sigs)):
            if sigs[a][1] == sigs[b][1]:
                pair = (sigs[a][0], sigs[b][0])
                break
        if pair is not None:
            break
    if pair is None:
        return inners[blank_idx].copy()

    remaining = [i for i in nonblank if i not in pair]
    if not remaining:
        return inners[blank_idx].copy()
    test_idx = remaining[0]

    directed_pair = decode_training_direction(g, frames, pair)
    if directed_pair is None:
        return inners[test_idx].copy()

    src_idx, dst_idx = directed_pair
    src = inners[src_idx]
    dst = inners[dst_idx]
    test_inner = inners[test_idx]
    if not infer_boundary_rule(src, dst, anchor_color):
        return test_inner.copy()
    return boundary_trace(test_inner, anchor_color)


def choose_palette(rng: random.Random, count: int, forbidden: set[int]) -> list[int]:
    pool = [c for c in SPRITE_COLORS if c not in forbidden]
    if len(pool) < count:
        raise ValueError("not enough sprite colors")
    return rng.sample(pool, count)


def connected_body(mask: np.ndarray) -> bool:
    comps = components_mask(mask, connectivity=4)
    return len(comps) == 1


def old_endpoints_like(arr: np.ndarray, color: int) -> np.ndarray:
    fg = arr == color
    h, w = arr.shape
    keep = np.zeros((h, w), dtype=bool)
    rr, cc = np.where(fg)
    for r, c in zip(rr.tolist(), cc.tolist()):
        deg = 0
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and fg[nr, nc]:
                    deg += 1
        if deg <= 1:
            keep[r, c] = True
    out = np.zeros_like(arr)
    out[keep] = color
    return out


def body_shape_with_anchor_effect(
    n: int,
    body_color: int,
    anchor_color: int,
    rng: random.Random,
    avoid_anchor: tuple[int, int] | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[int, int], dict[str, int]] | None:
    for _ in range(BODY_BUILD_RETRIES):
        body = np.zeros((n, n), dtype=bool)

        h1 = rng.randint(3, n)
        w1 = rng.randint(3, n)
        r1 = rng.randint(0, n - h1)
        c1 = rng.randint(0, n - w1)
        body[r1 : r1 + h1, c1 : c1 + w1] = True

        if rng.random() < SECOND_RECT_PROB:
            for _ in range(30):
                h2 = rng.randint(2, n)
                w2 = rng.randint(2, n)
                r2 = rng.randint(0, n - h2)
                c2 = rng.randint(0, n - w2)
                rect2 = np.zeros((n, n), dtype=bool)
                rect2[r2 : r2 + h2, c2 : c2 + w2] = True
                if np.any(body & rect2) and np.any(rect2 & ~body):
                    body |= rect2
                    break

        if int(np.count_nonzero(body)) + 1 > SPRITE_NZ_MAX:
            continue

        candidates = []
        rr, cc = np.where(body)
        for r, c in zip(rr.tolist(), cc.tolist()):
            if avoid_anchor is not None and (r, c) == avoid_anchor:
                continue
            body2 = body.copy()
            body2[r, c] = False
            if not np.any(body2):
                continue
            if not connected_body(body2):
                continue

            src = np.zeros((n, n), dtype=int)
            src[body2] = body_color
            src[r, c] = anchor_color
            dst = boundary_trace(src, anchor_color)

            body_src_n = int(np.count_nonzero(src == body_color))
            body_dst_n = int(np.count_nonzero(dst == body_color))
            if body_src_n < BODY_SRC_MIN or body_dst_n < BODY_DST_MIN:
                continue
            if body_src_n - body_dst_n < BODY_INTERIOR_REMOVED_MIN:
                continue

            naive = boundary_trace_all_nonzero(src)
            correct_body = dst == body_color
            naive_body = naive == body_color
            body_diff = int(np.count_nonzero(correct_body ^ naive_body))
            if body_diff < 1:
                continue
            if np.array_equal(correct_body, (old_endpoints_like(src, body_color) == body_color)):
                continue
            if np.array_equal(dst, src):
                continue

            metrics = {
                "body_src_n": body_src_n,
                "body_dst_n": body_dst_n,
                "body_diff_vs_naive": body_diff,
            }
            candidates.append((metrics["body_diff_vs_naive"], body_src_n - body_dst_n, src, dst, (r, c), metrics))

        if candidates:
            candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
            _, _, src, dst, anchor, metrics = candidates[0]
            return src, dst, anchor, metrics
    return None


def structured_positions(s: int, rng: random.Random) -> list[tuple[int, int]]:
    top_low, top_high = QUAD_LOW_MIN, QUAD_LOW_MAX
    bottom_low = H - s - QUAD_BOTTOM_MARGIN
    bottom_high = H - s - 1
    while True:
        top = rng.randint(top_low, top_high)
        bottom = rng.randint(bottom_low, bottom_high)
        left = rng.randint(top_low, top_high)
        right = rng.randint(bottom_low, bottom_high)
        if right - (left + s) < MIN_AXIS_GAP:
            continue
        if bottom - (top + s) < MIN_AXIS_GAP:
            continue
        return [(top, left), (top, right), (bottom, left), (bottom, right)]


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


def draw_direction_arrow(
    grid: np.ndarray,
    src_box: tuple[int, int, int, int],
    dst_box: tuple[int, int, int, int],
    marker_color: int,
) -> None:
    sr0, sc0, sr1, sc1 = src_box
    dr0, dc0, dr1, dc1 = dst_box
    if sc1 < dc0:
        mid = (sr0 + sr1) // 2
        shaft_start = sc1 + 1
        shaft_end = dc0 - 2
        draw_line(grid, mid, shaft_start, end_r=mid, end_c=shaft_end, color=marker_color)
        grid[mid - 1, shaft_end] = marker_color
        grid[mid + 1, shaft_end] = marker_color
        grid[mid, dc0 - 1] = marker_color
        return
    if dc1 < sc0:
        mid = (sr0 + sr1) // 2
        shaft_start = dc1 + 2
        shaft_end = sc0 - 1
        draw_line(grid, mid, shaft_start, end_r=mid, end_c=shaft_end, color=marker_color)
        grid[mid - 1, shaft_start] = marker_color
        grid[mid + 1, shaft_start] = marker_color
        grid[mid, dc1 + 1] = marker_color
        return
    if sr1 < dr0:
        mid = (sc0 + sc1) // 2
        shaft_start = sr1 + 1
        shaft_end = dr0 - 2
        draw_line(grid, shaft_start, mid, end_r=shaft_end, end_c=mid, color=marker_color)
        grid[shaft_end, mid - 1] = marker_color
        grid[shaft_end, mid + 1] = marker_color
        grid[dr0 - 1, mid] = marker_color
        return
    mid = (sc0 + sc1) // 2
    shaft_start = dr1 + 2
    shaft_end = sr0 - 1
    draw_line(grid, shaft_start, mid, end_r=shaft_end, end_c=mid, color=marker_color)
    grid[shaft_start, mid - 1] = marker_color
    grid[shaft_start, mid + 1] = marker_color
    grid[dr1 + 1, mid] = marker_color


def choose_scene_colors(
    train_body_color: int,
    test_body_color: int,
    rng: random.Random,
) -> tuple[int, int, int, list[int]]:
    used = {train_body_color, test_body_color}
    free = [c for c in ALL_COLORS if c not in used]
    if len(free) < 7:
        raise ValueError("not enough colors for anchor/frame/marker/background")
    anchor_color = rng.choice(free)
    free = [c for c in free if c != anchor_color]
    frame_color = rng.choice(free)
    free = [c for c in free if c != frame_color]
    marker_color = rng.choice(free)
    free = [c for c in free if c != marker_color]
    bg_sizes = [p for p in BG_PERIOD_CHOICES if len(free) >= p]
    if not bg_sizes:
        raise ValueError("not enough colors left for background")
    bg_palette = rng.sample(free, rng.choice(bg_sizes))
    return anchor_color, frame_color, marker_color, bg_palette


def assign_slots(rng: random.Random) -> tuple[tuple[int, int], tuple[int, int], int]:
    train_pair = rng.choice(TRAIN_PAIR_CHOICES)
    if rng.random() < 0.5:
        directed = (train_pair[0], train_pair[1])
    else:
        directed = (train_pair[1], train_pair[0])
    remaining = [i for i in range(4) if i not in train_pair]
    test_slot = rng.choice(remaining)
    return train_pair, directed, test_slot


def render_scene(
    s: int,
    positions: list[tuple[int, int]],
    sprites: list[np.ndarray],
    frame_color: int,
    marker_color: int,
    train_direction: tuple[int, int],
    bg_palette: list[int],
    rng: random.Random,
) -> np.ndarray:
    frame_boxes = [(r0, c0, r0 + s - 1, c0 + s - 1) for r0, c0 in positions]
    grid = render_background(frame_boxes, bg_palette, rng)
    for (r0, c0), sprite in zip(positions, sprites):
        draw_frame(grid, r0, c0, s, frame_color, sprite)
    draw_direction_arrow(grid, frame_boxes[train_direction[0]], frame_boxes[train_direction[1]], marker_color)
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

    nonblank = [i for i, inner in enumerate(inners) if not np.all(inner == 0)]
    if len(nonblank) != 3:
        return False
    anchor_color = find_anchor_color(inners, nonblank)
    if anchor_color is None:
        return False

    sigs = [body_signature(inners[i], anchor_color) for i in nonblank]
    dup_count = sum(1 for i in range(len(sigs)) for j in range(i + 1, len(sigs)) if sigs[i] == sigs[j])
    if dup_count != 1:
        return False

    pair = None
    for i in range(len(nonblank)):
        for j in range(i + 1, len(nonblank)):
            if sigs[i] == sigs[j]:
                pair = (nonblank[i], nonblank[j])
                break
        if pair is not None:
            break
    if pair is None or decode_training_direction(grid, frames, pair) is None:
        return False
    return np.array_equal(main(grid), expected_output)


def gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    s = rng.choice(FRAME_SIZE_CHOICES)
    n = s - 2

    train_body_color = choose_palette(rng, 1, set())[0]
    test_body_color = choose_palette(rng, 1, {train_body_color})[0]
    try:
        anchor_color, frame_color, marker_color, bg_palette = choose_scene_colors(
            train_body_color, test_body_color, rng
        )
    except ValueError:
        return None

    built_train = body_shape_with_anchor_effect(n, train_body_color, anchor_color, rng)
    if built_train is None:
        return None
    train_src, train_dst, train_anchor, train_metrics = built_train

    built_test = body_shape_with_anchor_effect(
        n, test_body_color, anchor_color, rng, avoid_anchor=train_anchor
    )
    if built_test is None:
        return None
    test_src, test_dst, _test_anchor, test_metrics = built_test

    if train_metrics["body_diff_vs_naive"] < 1 or test_metrics["body_diff_vs_naive"] < 1:
        return None

    positions = structured_positions(s, rng)
    _train_pair, train_direction, test_slot = assign_slots(rng)

    sprites = [np.zeros((n, n), dtype=int) for _ in range(4)]
    sprites[train_direction[0]] = train_src
    sprites[train_direction[1]] = train_dst
    sprites[test_slot] = test_src

    grid = render_scene(s, positions, sprites, frame_color, marker_color, train_direction, bg_palette, rng)
    if not validate_scene(grid, test_dst):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_MAX_ATTEMPTS} attempts")
