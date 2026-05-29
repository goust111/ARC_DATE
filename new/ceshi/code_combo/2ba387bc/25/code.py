# Combo variant of 2ba387bc (index 25)
# combo_spec: D8/2+D11/0
# Name: 逐步移动+非标尺寸干扰

from __future__ import annotations

import numpy as np

from arc_utils import bbox_h, bbox_w, crop_bbox, find_components, get_bg, paste, random_free_location

# Design analysis for the fused combo variant
# -------------------------------------------
# a) Concrete behavior and trace:
#    - Original 2ba387bc:
#      * input contains only isolated 4x4 objects on a background
#      * each object is either a solid 4x4 square or a 4x4 hollow frame
#      * output discards the source layout and shows one final 8-column stack:
#        frames in the left column, solids in the right column, each group sorted
#        by source row
#    - Fused D8/2 + D11/0:
#      * the input may also contain extra non-4x4 monochrome rectangles; some are
#        attached orthogonally to valid 4x4 squares with a different color
#      * only true 4x4 frame/solid squares are movable targets; every other shape
#        is a distractor and remains in place forever
#      * the output is no longer one final stack. It is a vertical strip of panels.
#        In each panel: left = current source grid, right = the accumulated 8-column
#        destination stack. Start with the untouched input, then move exactly one
#        valid 4x4 square per next panel.
#      * move order is deterministic: first sort frames and solids separately by
#        source (row, col), then read the final two-column destination top-to-bottom,
#        left-to-right. So row-band i contributes frame i first, then solid i.
#    - Concrete mini-trace:
#      * suppose sorted frames are F1 above F2, sorted solids are S1 above S2
#      * panel 0: source still contains F1,F2,S1,S2 plus distractors; destination empty
#      * panel 1: F1 disappears from its source location and appears at destination rows 0:4, cols 0:4
#      * panel 2: S1 disappears and appears at destination rows 0:4, cols 4:8
#      * panel 3: F2 moves to rows 4:8, cols 0:4
#      * panel 4: S2 moves to rows 4:8, cols 4:8
#      * distractors never move in any panel
#    - Difference from original:
#      * D11 breaks the old "whole blob is the object" assumption
#      * D8 makes the final answer depend on visible intermediate states rather than
#        only on the final sorted arrangement
# b) Complexity calibration:
#    - Valid objects stay visually simple: only 4x4 frames and solids.
#    - Distractors use a short rectangle vocabulary such as 1x5, 2x3, 5x2, so a
#      human can see they are "wrong-sized extras" rather than new target classes.
#    - Generator forces 3-5 movable squares, so the panel strip shows several steps;
#      one-step/no-op traces are rejected.
#    - At least one distractor is attached to a valid square, ensuring the D11 change
#      is not visually silent.
# c) Implicit-assumption audit of the original main():
#    - `find_components(... monochromatic=False)` assumed touching different colors
#      should merge into one object. D11 breaks that: an attached distractor must not
#      hide the underlying 4x4 square, so detection must be monochromatic.
#    - `if bbox is 4x4 and not frame -> solid` assumed every 4x4 blob is meaningful.
#      The fused solver keeps an explicit solid test instead of a catch-all branch.
#    - `return one final 8-column stack` assumed processing order was irrelevant.
#      D8 breaks that: the answer is a time trace, so the solver must simulate the
#      move sequence panel by panel.
#    - Sorting by source row still works because blanking earlier moved squares does
#      not change the remaining targets' original coordinates. The solver therefore
#      extracts all valid squares once from the initial input before simulating.
# d) Placement legality:
#    - Valid squares are placed first with a reserved opaque 4x4 footprint, so no
#      distractor can overlap a frame hole or intrude into a source square.
#    - Distractors are monochrome rectangles, never overlap existing colored cells,
#      and attached distractors are allowed to touch exactly one valid square.
#    - Canvas height is chosen large enough that the right-side 8-column destination
#      stack fits entirely inside each panel.
# e) Visual observability:
#    - A human sees the same distractors frozen in every panel while one 4x4 target
#      at a time disappears from the left and appears in the right stack.
#    - The left/right panel layout makes the "take from source, place into output"
#      rule explicit with only 2-4 examples.
#    - No hidden ordering uses color ids; ordering is visible from source positions
#      and destination slot order.
# f) Visual cleanliness:
#    - Panels are separated by one blank row.
#    - Source and destination within each panel are separated by one blank column.
#    - Colors are unique across valid squares and distractors, avoiding accidental
#      same-color mergers and keeping each object easy to track by eye.
# g) Change inferability / anti-degeneracy:
#    - D8 is quantified by `panel_count = valid_count + 1`; generator enforces
#      `valid_count >= 3`, so the output is strictly richer than the original final
#      stack.
#    - D11 is quantified by the number of distractor components touching a valid
#      square; generator enforces at least one such attachment.
# 9) New solver assumptions and generator support:
#    - The solver assumes every movable target is exactly one monochromatic 4x4
#      component whose crop is either a true frame or a true solid; generator uses
#      unique colors and never overlays distractors onto those 4x4 footprints.
#    - The solver assumes the panel layout is: source on the left, one blank column,
#      then the accumulated destination stack. Generator and solver share this layout.
#    - The solver assumes the move order is the reading order of the final two-column
#      stack; generator computes expected outputs with that same rule and rejects
#      anything inconsistent.


_FRAME_MASK = np.ones((4, 4), dtype=bool)
_FRAME_MASK[1:3, 1:3] = False

_DISTRACTOR_SHAPES: tuple[tuple[int, int], ...] = (
    (1, 5),
    (5, 1),
    (2, 3),
    (3, 2),
    (2, 5),
    (5, 2),
)

_PANEL_GAP = 1
_SIDE_GAP = 1


def _make_frame(col: int, bg: int = 0) -> np.ndarray:
    arr = np.full((4, 4), int(col), dtype=int)
    arr[1:3, 1:3] = int(bg)
    return arr


def _make_solid(col: int) -> np.ndarray:
    return np.full((4, 4), int(col), dtype=int)


def _classify_valid_square(sub: np.ndarray, color: int, area: int, bg: int) -> str | None:
    if sub.shape != (4, 4):
        return None

    color = int(color)
    area = int(area)
    bg = int(bg)

    if area == 16 and bool((sub == color).all()):
        return "solid"
    if area == 12 and bool((sub[_FRAME_MASK] == color).all()) and bool((sub[1:3, 1:3] == bg).all()):
        return "frame"
    return None


def _detect_valid_squares(grid: np.ndarray, bg: int) -> tuple[list[tuple[int, int, np.ndarray]], list[tuple[int, int, np.ndarray]]]:
    comps = find_components(grid, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    frames: list[tuple[int, int, np.ndarray]] = []
    solids: list[tuple[int, int, np.ndarray]] = []

    for comp in comps:
        bbox = comp["bbox"]
        if bbox_h(bbox) != 4 or bbox_w(bbox) != 4:
            continue
        r0, c0, _r1, _c1 = bbox
        sub = crop_bbox(grid, bbox)
        kind = _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg)
        if kind == "frame":
            frames.append((int(r0), int(c0), sub))
        elif kind == "solid":
            solids.append((int(r0), int(c0), sub))

    frames.sort(key=lambda t: (int(t[0]), int(t[1])))
    solids.sort(key=lambda t: (int(t[0]), int(t[1])))
    return frames, solids


def _build_move_sequence(
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
) -> list[tuple[int, int, np.ndarray, int, int]]:
    moves: list[tuple[int, int, np.ndarray, int, int]] = []
    for idx in range(max(len(frames), len(solids))):
        if idx < len(frames):
            r0, c0, sub = frames[idx]
            moves.append((int(r0), int(c0), sub, int(idx * 4), 0))
        if idx < len(solids):
            r0, c0, sub = solids[idx]
            moves.append((int(r0), int(c0), sub, int(idx * 4), 4))
    return moves


def _render_panel(source: np.ndarray, stage: np.ndarray, bg: int) -> np.ndarray:
    H, W = map(int, source.shape)
    out = np.full((H, W + _SIDE_GAP + 8), int(bg), dtype=int)
    out[:, :W] = source
    if stage.size:
        out[: stage.shape[0], W + _SIDE_GAP : W + _SIDE_GAP + 8] = stage
    return out


def _stepwise_output(
    grid: np.ndarray,
    frames: list[tuple[int, int, np.ndarray]],
    solids: list[tuple[int, int, np.ndarray]],
    bg: int,
) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, _W = map(int, g.shape)
    moves = _build_move_sequence(frames, solids)
    stage_h = 4 * max(len(frames), len(solids))
    stage = np.full((stage_h, 8), int(bg), dtype=int)
    source = g.copy()

    panels: list[np.ndarray] = [_render_panel(source, stage, bg)]
    for r0, c0, sub, dst_r, dst_c in moves:
        source[r0 : r0 + 4, c0 : c0 + 4] = int(bg)
        paste(stage, sub, int(dst_r), int(dst_c))
        panels.append(_render_panel(source, stage, bg))

    out_h = len(panels) * H + max(0, len(panels) - 1) * _PANEL_GAP
    out_w = int(panels[0].shape[1])
    out = np.full((out_h, out_w), int(bg), dtype=int)
    r = 0
    for idx, panel in enumerate(panels):
        paste(out, panel, int(r), 0)
        r += H
        if idx != len(panels) - 1:
            r += _PANEL_GAP
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused D8/2 + D11/0 solver for 2ba387bc.

    1. Detect only true monochromatic 4x4 frame/solid squares.
    2. Ignore every non-4x4 distractor, even if it touches a valid square.
    3. Build a vertical strip of panels showing the source being emptied one square
       at a time while the right-side destination stack accumulates in reading order.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _detect_valid_squares(g, bg)
    return _stepwise_output(g, frames, solids, bg)


def _bbox_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(int(r0), int(r1) + 1) for c in range(int(c0), int(c1) + 1)}


def _can_place_attached(
    grid: np.ndarray,
    patch: np.ndarray,
    r0: int,
    c0: int,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> bool:
    h, w = map(int, patch.shape)
    H, W = map(int, grid.shape)
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    if bool((grid[r0 : r0 + h, c0 : c0 + w] != int(bg)).any()):
        return False

    target_cells = _bbox_cells(target_bbox)
    touched_target = False
    touched_other = False
    for rr in range(r0, r0 + h):
        for cc in range(c0, c0 + w):
            for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                if not (0 <= nr < H and 0 <= nc < W):
                    continue
                if int(grid[nr, nc]) == int(bg):
                    continue
                if (nr, nc) in target_cells:
                    touched_target = True
                else:
                    touched_other = True
    return touched_target and not touched_other


def _attached_location(
    grid: np.ndarray,
    patch: np.ndarray,
    target_bbox: tuple[int, int, int, int],
    bg: int,
) -> tuple[int, int] | None:
    r0, c0, r1, c1 = target_bbox
    h, w = map(int, patch.shape)
    candidates: list[tuple[int, int]] = []

    if c0 - w >= 0:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((int(rr), int(c0 - w)))
    if c1 + 1 + w <= grid.shape[1]:
        for rr in range(int(r0 - h + 1), int(r1) + 1):
            candidates.append((int(rr), int(c1 + 1)))
    if r0 - h >= 0:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r0 - h), int(cc)))
    if r1 + 1 + h <= grid.shape[0]:
        for cc in range(int(c0 - w + 1), int(c1) + 1):
            candidates.append((int(r1 + 1), int(cc)))

    np.random.shuffle(candidates)
    for rr, cc in candidates:
        if _can_place_attached(grid, patch, int(rr), int(cc), target_bbox, bg):
            return int(rr), int(cc)
    return None


def _place_distractor(
    grid: np.ndarray,
    patch: np.ndarray,
    bg: int,
    anchors: list[tuple[int, int, int, int]],
    force_attach: bool,
) -> tuple[bool, bool]:
    anchor_order = list(range(len(anchors)))
    np.random.shuffle(anchor_order)

    if force_attach:
        for idx in anchor_order:
            loc = _attached_location(grid, patch, anchors[idx], bg)
            if loc is not None:
                paste(grid, patch, int(loc[0]), int(loc[1]), transparent=bg)
                return True, True
        return False, False

    if bool(np.random.rand() < 0.6):
        for idx in anchor_order:
            loc = _attached_location(grid, patch, anchors[idx], bg)
            if loc is not None:
                paste(grid, patch, int(loc[0]), int(loc[1]), transparent=bg)
                return True, True

    try:
        r0, c0 = random_free_location(grid, patch, transparent=bg, border=1, padding=1, padding_connectivity=4)
        paste(grid, patch, int(r0), int(c0), transparent=bg)
        return True, False
    except Exception:
        return False, False


def _count_attached_distractors(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    frames, solids = _detect_valid_squares(g, bg)
    valid_bboxes = [(int(r0), int(c0), int(r0 + 3), int(c0 + 3)) for r0, c0, _sub in frames + solids]
    valid_cells = set()
    for bbox in valid_bboxes:
        valid_cells |= _bbox_cells(bbox)

    comps = find_components(g, bg=bg, connectivity=4, monochromatic=True, return_format="dict")
    attached = 0
    for comp in comps:
        bbox = comp["bbox"]
        sub = crop_bbox(g, bbox)
        if _classify_valid_square(sub, int(comp["color"]), int(comp["area"]), bg) is not None:
            continue
        coords = {(int(r), int(c)) for r, c in comp["coords"]}
        touches = False
        for r, c in coords:
            for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if (int(nr), int(nc)) in valid_cells:
                    touches = True
                    break
            if touches:
                break
        if touches:
            attached += 1
    return int(attached)


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    H = int(np.random.randint(20, 29))
    W = int(np.random.randint(20, 29))
    g = np.full((H, W), bg, dtype=int)

    total_valid = int(np.random.randint(3, 6))
    n_frames = int(np.random.randint(1, total_valid))
    n_solids = int(total_valid - n_frames)
    n_distractors = int(np.random.randint(1, min(4, 10 - total_valid)))

    colors = list(np.random.permutation(np.arange(1, 10)))
    valid_colors = colors[:total_valid]
    distractor_colors = colors[total_valid : total_valid + n_distractors]

    placed_frames: list[tuple[int, int, np.ndarray]] = []
    placed_solids: list[tuple[int, int, np.ndarray]] = []
    anchor_bboxes: list[tuple[int, int, int, int]] = []

    try:
        valid_specs: list[tuple[str, int, np.ndarray]] = []
        for col in valid_colors[:n_frames]:
            valid_specs.append(("frame", int(col), _make_frame(int(col), bg)))
        for col in valid_colors[n_frames:]:
            valid_specs.append(("solid", int(col), _make_solid(int(col))))
        np.random.shuffle(valid_specs)

        for kind, col, patch in valid_specs:
            reserve = np.full((4, 4), int(col), dtype=int)
            r0, c0 = random_free_location(g, reserve, transparent=bg, border=1, padding=2, padding_connectivity=4)
            paste(g, patch, int(r0), int(c0), transparent=bg)
            anchor_bboxes.append((int(r0), int(c0), int(r0 + 3), int(c0 + 3)))
            if kind == "frame":
                placed_frames.append((int(r0), int(c0), patch))
            else:
                placed_solids.append((int(r0), int(c0), patch))

        attached_count = 0
        for idx, dcol in enumerate(distractor_colors):
            shape = _DISTRACTOR_SHAPES[int(np.random.randint(0, len(_DISTRACTOR_SHAPES)))]
            patch = np.full(shape, int(dcol), dtype=int)
            ok, attached = _place_distractor(g, patch, bg, anchor_bboxes, force_attach=(idx == 0))
            if not ok:
                return None
            if attached:
                attached_count += 1

        if attached_count == 0:
            return None

        placed_frames.sort(key=lambda t: (int(t[0]), int(t[1])))
        placed_solids.sort(key=lambda t: (int(t[0]), int(t[1])))
        expected = _stepwise_output(g, placed_frames, placed_solids, bg)
        actual = main(g)
        if actual.shape != expected.shape or not bool((actual == expected).all()):
            return None

        if _count_attached_distractors(g) <= 0:
            return None
        if len(placed_frames) + len(placed_solids) < 3:
            return None

        return g
    except Exception:
        return None


def generate_input() -> np.ndarray:
    for _ in range(500):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 500 attempts")
