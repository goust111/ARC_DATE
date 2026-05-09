# Combo variant of 8f3a5a89 (index 6)
# combo_spec: D7/3+D9/0+D11/0
# Name: 单个闭合框架+裁剪输出
#
# Design analysis
# ===============
# 1) Concrete behavior trace
#    - Base task: start from the unique corner marker, detect which blue pixels
#      are true structural barriers, flood the marker-reachable side, paint an
#      orange inner boundary there, and erase hidden blue decorations.
#    - This combo collapses the structural side to one connected near-rectangle:
#      the only real wall component is a 1px rectangular frame anchored to the
#      lower-left, with a single bottom-left doorway where the marker enters.
#    - Small trace: if the frame's top edge is row 3 and its right edge is col 8
#      in a 14x13 grid, then the reachable set is the room interior plus the
#      2-cell doorway at the lower-left corner. The solved full-canvas image
#      still paints orange on that room-side inner boundary, but the final output
#      keeps only rows 4..13 and cols 0..7 because those are exactly the rows and
#      cols touched by the reachable set.
#    - Compared with the base task, both "which wall is structural?" and "what
#      is the output size?" now hinge on the same room interpretation.
#
# 2) Complexity calibration
#    - If the frame did not touch the lower-left corner area, the fixed marker at
#      bottom-left would force either extra structural components or an
#      always-reachable exterior, which would destroy the crop. So the room is
#      deliberately anchored to the left/bottom borders and trimmed only from the
#      top and right.
#    - If the crop margins were optional, the new task would often degenerate
#      back to the base same-size output. The generator therefore enforces both a
#      positive top trim and a positive right trim on every sample.
#    - The frame shape is kept to one family instead of many obstacle types so
#      the fusion remains human-inferable: the challenge is doorway/interior
#      reasoning plus crop recovery, not arbitrary wall geometry.
#
# 3) Implicit-assumption review for main()
#    - Marker detection still assumes a unique non-background corner pixel. The
#      generator keeps a single magenta pixel at the bottom-left corner.
#    - Wall-color recovery still assumes the most common non-bg non-marker color
#      is blue. The generator uses only one wall/decor color.
#    - Structural-wall recovery in the base solver looks for wall-color
#      components that touch the border and are adjacent to the first flood. The
#      combo generator preserves this by making exactly one such component: the
#      doorway-broken frame.
#    - The crop step assumes the reachable set is non-empty and includes the
#      marker. That is guaranteed because the marker itself is the flood seed.
#    - Because the crop is by reachable bbox rather than by non-bg bbox,
#      unreachable wall rows/cols above/right of the room are intentionally
#      discarded even though they contain blue pixels.
#
# 4) Placement legality
#    - The structural wall is built as a rectangle border on bbox
#      [frame_top..H-1] x [0..frame_right], then two contiguous cells at the
#      lower-left corner are removed to form one doorway. This keeps the wall
#      4-connected and non-overlapping by construction.
#    - Optional visible decorations are placed only on reachable interior
#      background cells; optional hidden decorations are placed only on the
#      unreachable side, away from the grid border so they cannot be mistaken for
#      the structural frame.
#    - Validation recomputes reachability, checks the frame signature, checks the
#      crop margins, and confirms `main(input)` matches the expected cropped
#      output.
#
# 5) Visual observability / cleanliness
#    - In 2-3 examples a human can see the same motif repeat: one blue room-like
#      rectangle with a small doorway near the bottom-left marker, orange lining
#      on the room side, and outputs cropped tightly to that room.
#    - The generator keeps decoration counts small and shapes tiny so they never
#      visually compete with the dominant frame.
#    - Colors remain the base palette: azure background, blue wall/decor,
#      magenta marker, orange fill. No color-id-based hidden rule is introduced.
#
# 6) New solver assumptions audited against the generator
#    - `main()` assumes every meaningful barrier is recoverable by the base
#      "border-touching and adjacent to first reach" heuristic. The generator
#      enforces exactly one such component.
#    - `main()` assumes the reachable bbox trims at least one top row and one
#      right column if the combo behavior is to be visible. The generator enforces
#      that with explicit frame-top/frame-right ranges.
#    - `main()` treats non-structural blue pixels as passable during the second
#      flood. The generator therefore keeps visible decorations disconnected from
#      the frame so they are not misclassified as structural.

from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np

from arc_utils import components_mask, get_bg, neighbors4, neighbors8, random_free_location


def _flood(start: tuple[int, int], passable, h: int, w: int) -> set[tuple[int, int]]:
    q = deque([start])
    vis = {start}
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in vis and passable(nr, nc):
                vis.add((nr, nc))
                q.append((nr, nc))
    return vis


def _solve_uncropped(grid: np.ndarray) -> tuple[np.ndarray, set[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)
    out = g.copy()

    counts = Counter(int(v) for v in g.ravel())
    corners = [(0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)]
    marker = None
    for r, c in corners:
        v = int(g[r, c])
        if v != bg and counts.get(v, 0) == 1:
            marker = (r, c, v)
            break
    assert marker is not None, "no unique single-pixel corner marker found"

    mr, mc, marker_col = marker
    other_cols = [c for c in counts if c not in (bg, marker_col)]
    assert other_cols, "no non-bg non-marker colors found; cannot determine wall color"
    wall_col = max(other_cols, key=lambda c: counts[c])

    reach0 = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col),
        h,
        w,
    )

    walls: set[tuple[int, int]] = set()
    for comp_arr in components_mask(g == wall_col, connectivity=4):
        comp = {(int(r), int(c)) for r, c in comp_arr}
        touches_border = any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in comp)
        if not touches_border:
            continue
        if any(any((nr, nc) in reach0 for nr, nc in neighbors4(r, c)) for r, c in comp):
            walls |= comp

    reach = _flood(
        (mr, mc),
        lambda r, c: int(g[r, c]) in (bg, marker_col) or (int(g[r, c]) == wall_col and (r, c) not in walls),
        h,
        w,
    )

    for r in range(h):
        for c in range(w):
            if int(out[r, c]) == wall_col and (r, c) not in walls and (r, c) not in reach:
                out[r, c] = bg

    fill_col = 7
    for r, c in reach:
        if int(out[r, c]) != bg:
            continue
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            out[r, c] = fill_col
            continue
        if any((nr, nc) in walls for nr, nc in neighbors8(r, c)):
            out[r, c] = fill_col
            continue
        if any((nr, nc) not in reach for nr, nc in neighbors4(r, c)):
            out[r, c] = fill_col

    return out, reach


def _crop_to_reach(out: np.ndarray, reach: set[tuple[int, int]]) -> np.ndarray:
    pts = np.array(list(reach), dtype=int)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return out[r0 : r1 + 1, c0 : c1 + 1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the doorway-entered room, paint its inner lining, then crop to it.

    Input structure:
    - The background is the most common color and the marker is a single magenta
      pixel fixed on one corner, here bottom-left.
    - Blue pixels contain one border-touching connected component that behaves as
      an almost closed rectangular frame, plus optional small decorative blue
      patches disconnected from that frame.
    - The marker can enter the room only through the frame's one small opening,
      so the room interior is the marker-reachable side of the true wall.

    Rule:
    - Flood from the marker while treating only the border-touching structural
      blue component as blocking; treat disconnected blue decorations as
      passable during reachability recovery.
    - Remove decorative blue patches that lie only on the unreachable side.
    - Paint orange on reachable background cells that lie on the room's inner
      boundary.
    - Crop the solved grid to the smallest rectangle containing all reachable
      cells.

    Key insight: The frame and the crop are coupled: once the solver identifies
    which side of the near-closed rectangle the marker can enter, that same
    reachable set determines both the orange border and the final output size.
    """
    out, reach = _solve_uncropped(np.asarray(grid, dtype=int))
    return _crop_to_reach(out, reach)


BG = 8
WALL = 1
MARKER = 6
FILL = 7

GRID_H_RANGE = (10, 20)
GRID_W_RANGE = (10, 20)

TOP_TRIM_MIN = 2
RIGHT_TRIM_MIN = 2
INTERIOR_H_MIN = 4
INTERIOR_W_MIN = 4

NUM_VISIBLE_DECOR_RANGE = (0, 1)
NUM_HIDDEN_DECOR_RANGE = (0, 2)
DECOR_AREA_RANGE = (1, 3)
DECOR_BORDER = 1
DECOR_PADDING = 1

REACH_RATIO_RANGE = (0.20, 0.72)
GEN_ATTEMPTS = 600


def _flood_mask(start: tuple[int, int], blocked: np.ndarray, h: int, w: int) -> np.ndarray:
    reach = np.zeros((h, w), dtype=bool)
    reach[start] = True
    q = deque([start])
    while q:
        r, c = q.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not reach[nr, nc] and not blocked[nr, nc]:
                reach[nr, nc] = True
                q.append((nr, nc))
    return reach


def _dilate4(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    return out


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = _dilate4(mask)
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _paint_output(structural: np.ndarray, reach: np.ndarray, marker_rc: tuple[int, int]) -> np.ndarray:
    h, w = structural.shape
    out = np.full((h, w), BG, dtype=int)
    out[structural] = WALL
    out[marker_rc] = MARKER

    candidate = reach & (out == BG)

    edge = np.zeros_like(candidate)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True

    near_struct = _dilate8(structural)
    near_unreach = _dilate4(~reach)

    fill = candidate & (edge | near_struct | near_unreach)
    out[fill] = FILL
    return out


def _reachable_bbox(reach: np.ndarray) -> tuple[int, int, int, int]:
    pts = np.argwhere(reach)
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_array(arr: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return arr[r0 : r1 + 1, c0 : c1 + 1]


def _build_structural_frame(h: int, w: int, frame_top: int, frame_right: int) -> np.ndarray:
    structural = np.zeros((h, w), dtype=bool)
    structural[frame_top, : frame_right + 1] = True
    structural[frame_top:, frame_right] = True
    structural[h - 1, : frame_right + 1] = True
    structural[frame_top:, 0] = True

    # One doorway at the lower-left corner: marker cell + one cell above it.
    structural[h - 1, 0] = False
    structural[h - 2, 0] = False
    return structural


def _valid_frame_signature(structural: np.ndarray) -> bool:
    comps = components_mask(structural, connectivity=4)
    if len(comps) != 1:
        return False

    pts = comps[0]
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
    if c0 != 0 or r1 != structural.shape[0] - 1:
        return False
    if r1 - r0 + 1 < INTERIOR_H_MIN + 2 or c1 - c0 + 1 < INTERIOR_W_MIN + 2:
        return False

    per = np.zeros_like(structural, dtype=bool)
    per[r0, c0 : c1 + 1] = True
    per[r1, c0 : c1 + 1] = True
    per[r0 : r1 + 1, c0] = True
    per[r0 : r1 + 1, c1] = True

    if np.any(structural & ~per):
        return False

    missing = set(map(tuple, np.argwhere(per & ~structural)))
    return missing == {(r1 - 1, c0), (r1, c0)}


def _place_decor(allowed_mask: np.ndarray, count: int, rng: random.Random) -> np.ndarray | None:
    h, w = allowed_mask.shape
    placed = np.zeros((h, w), dtype=bool)
    if count == 0:
        return placed

    for _ in range(count):
        area = rng.randint(*DECOR_AREA_RANGE)
        shapes = [(1, area), (area, 1)]
        if area >= 2:
            shapes.append((2, 2))
        rng.shuffle(shapes)
        for ph, pw in shapes:
            if ph > h or pw > w:
                continue
            patch = np.ones((ph, pw), dtype=int)
            canvas = np.where(allowed_mask & ~placed, 0, 9)
            try:
                r0, c0 = random_free_location(
                    canvas,
                    patch,
                    transparent=0,
                    border=DECOR_BORDER,
                    padding=DECOR_PADDING,
                    padding_connectivity=8,
                )
            except ValueError:
                continue
            placed[r0 : r0 + ph, c0 : c0 + pw] = True
            break
        else:
            return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    rng = random
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    marker_rc = (h - 1, 0)

    max_frame_top = h - INTERIOR_H_MIN - 2
    max_frame_right = w - RIGHT_TRIM_MIN
    if max_frame_top < TOP_TRIM_MIN or max_frame_right < INTERIOR_W_MIN + 1:
        return None

    frame_top = rng.randint(TOP_TRIM_MIN, max_frame_top)
    frame_right = rng.randint(INTERIOR_W_MIN + 1, max_frame_right)

    structural = _build_structural_frame(h, w, frame_top, frame_right)
    if not _valid_frame_signature(structural):
        return None

    reach = _flood_mask(marker_rc, structural, h, w)
    if not reach[marker_rc]:
        return None

    reach_bbox = _reachable_bbox(reach)
    r0, c0, r1, c1 = reach_bbox
    if r0 <= 0 or (w - 1 - c1) <= 0:
        return None
    if r0 < TOP_TRIM_MIN or (w - 1 - c1) < RIGHT_TRIM_MIN:
        return None
    if c0 != 0 or r1 != h - 1:
        return None

    interior_h = h - frame_top - 2
    interior_w = frame_right - 1
    if interior_h < INTERIOR_H_MIN or interior_w < INTERIOR_W_MIN:
        return None

    reach_frac = float(reach.mean())
    if not (REACH_RATIO_RANGE[0] <= reach_frac <= REACH_RATIO_RANGE[1]):
        return None

    out_full = _paint_output(structural, reach, marker_rc)
    fill_mask = out_full == FILL
    if not fill_mask.any() or len(components_mask(fill_mask, connectivity=4)) != 1:
        return None

    visible_allowed = reach & (out_full == BG)
    hidden_allowed = (~reach) & (~structural)
    hidden_allowed[0, :] = False
    hidden_allowed[-1, :] = False
    hidden_allowed[:, 0] = False
    hidden_allowed[:, -1] = False

    n_visible = rng.randint(*NUM_VISIBLE_DECOR_RANGE)
    n_hidden = rng.randint(*NUM_HIDDEN_DECOR_RANGE)
    visible_placed = _place_decor(visible_allowed, n_visible, rng)
    if visible_placed is None:
        return None
    hidden_placed = _place_decor(hidden_allowed, n_hidden, rng)
    if hidden_placed is None:
        return None

    out_full[visible_placed] = WALL

    inp = out_full.copy()
    inp[inp == FILL] = BG
    inp[hidden_placed] = WALL

    expected = _crop_array(out_full, reach_bbox)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        try:
            g = _gen_attempt()
        except Exception:
            g = None
        if g is not None:
            return g
    raise RuntimeError("failed to generate valid input for 8f3a5a89 combo variant 6")
