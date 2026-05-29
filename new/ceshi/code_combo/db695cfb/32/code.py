# Combo variant of db695cfb (index 32)
# combo_spec: D1/0+D6/3+D10/0
# Name: 直线路径+边缘源+裁剪输出

from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_SOLVER_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVER_DIR) not in sys.path:
    sys.path.append(str(_SOLVER_DIR))

from arc_utils import get_bg


# Design analysis for the combo variant
# =====================================
# combo_summary:
#   - D1/0 changes the carrier path from 45-degree diagonals to axis-aligned
#     horizontal/vertical segments.
#   - D6/3 changes the source objects from free interior endpoint pairs to blue
#     control pixels sitting on opposite grid edges, so the carrier stroke spans
#     a whole row or a whole column.
#   - D10/0 changes the return shape from full-canvas to the tight bounding box
#     of the active construction only.
# concrete_trace:
#   - Original local example:
#       blue at (1,1) and (4,4), magenta at (2,2)
#       => draw the diagonal between the blue pixels, then fire the other
#          diagonal through the magenta cell, keep full canvas.
#   - Combo local example:
#       blue at (3,0) and (3,8), optional magenta on row 3,
#       => draw the full row 3 because the controls sit on opposite edges.
#          If a magenta were on that row, it would fire vertical rays.
#          After drawing, crop to rows/cols touched by the active line system.
#   - Visible difference:
#       the path is orthogonal instead of diagonal, the blue clues live on the
#       border, and the answer is returned in a reduced frame.
# complexity_calibration:
#   - D6 and D10 interact in a nontrivial way: a real trigger on a full-width
#     row or full-height column expands the active bbox to the entire canvas,
#     erasing the crop effect. To keep the crop observable in 2-4 examples, the
#     generator uses zero on-segment triggers while the solver still supports
#     them if such an input is provided.
#   - Multiple true segments remain allowed, but the generator uses one global
#     orientation per sample. Mixing horizontal and vertical edge-spanning
#     segments would also force the active bbox to the full canvas.
# implicit_assumptions_review:
#   - The original diagonal-pair search is invalid. In the combo, valid blue
#     segments are not arbitrary same-row / same-column pairs; they are only
#     opposite-edge pairs: left-right on one row or top-bottom on one column.
#   - A naive row/column grouping would create false border lines because many
#     horizontal segments all place blue pixels on column 0 and column W-1.
#     Therefore the solver must pair endpoints by opposite edges, not by column
#     extremes of all blue pixels.
#   - Cropping must use only active construction cells, not every non-bg cell,
#     otherwise outside distractors would anchor the output frame and D10 would
#     disappear.
#   - Because the generator suppresses true triggers, `main()` now assumes only
#     that at least one opposite-edge blue pair exists; it does not assume any
#     trigger is present.
# placement_legality:
#   - Real segments use distinct interior rows (for horizontal mode) or distinct
#     interior columns (for vertical mode), guaranteeing a non-full crop box.
#   - Same-color distractor blue pixels are kept away from all borders so they
#     cannot accidentally become new opposite-edge control pairs.
#   - Magenta distractors are kept off real segment rows/cols and preferably
#     outside the crop box so they remain visibly irrelevant.
# visual_observability_and_cleanliness:
#   - Humans can see that blue controls touch the border and define entire
#     rows/columns, not diagonals.
#   - Cropping is visually obvious because the output strips empty margins while
#     some distractors may vanish outside the retained band.
#   - The generator avoids overlap between active rows/cols and distractors to
#     keep the construction legible.
# inferability_check:
#   - The only color-role mapping remains blue = control line source and
#     magenta = potential trigger/distractor. No hidden color ordering is used.
#   - Structural cues, not color ids, distinguish real controls: they occur in
#     opposite-edge pairs and repeat across a sample when several segments are
#     present.
# solver_assumption_audit:
#   - `main()` scans each color for opposite-edge pairs and chooses the color
#     with the strongest such support. The generator guarantees distractors of
#     other colors never form opposite-edge pairs.
#   - For each accepted segment, any non-line cell encountered on the path is
#     treated as a trigger. The generator guarantees no unrelated colors are
#     placed on those paths.
#   - `generate_input()` explicitly rejects cases where the active bbox equals
#     the full canvas, matching the solver's crop semantics.


def _bbox_from_cells(cells: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_grid(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _straight_path(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    r0, c0 = p
    r1, c1 = q
    if r0 == r1:
        lo, hi = sorted((c0, c1))
        return [(r0, c) for c in range(lo, hi + 1)]
    if c0 == c1:
        lo, hi = sorted((r0, r1))
        return [(r, c0) for r in range(lo, hi + 1)]
    raise ValueError("points must share a row or column")


def _segment_dirs(p: tuple[int, int], q: tuple[int, int]) -> list[tuple[int, int]]:
    if p[0] == q[0]:
        return [(-1, 0), (1, 0)]
    return [(0, -1), (0, 1)]


def _segments_for_color(
    pts: list[tuple[int, int]], h: int, w: int
) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    pos = set(pts)
    segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for r in range(h):
        left = (r, 0)
        right = (r, w - 1)
        if left in pos and right in pos:
            segments.append((left, right))
    for c in range(w):
        top = (0, c)
        bottom = (h - 1, c)
        if top in pos and bottom in pos:
            segments.append((top, bottom))
    return segments


def main(grid: np.ndarray) -> np.ndarray:
    """
    Connect opposite-edge blue controls with straight full-span strokes, then crop to the active construction.

    Input structure:
    - The most common color is background. One foreground color places paired control pixels on opposite grid edges, so each valid pair occupies either both ends of a row or both ends of a column. Other non-background pixels may appear as off-path distractors, and any non-line pixel lying on a valid stroke acts as a trigger.

    Rule:
    - Find the color that forms opposite-edge control pairs and draw the full horizontal or vertical segment between each such pair.
    - Whenever one of those segments passes through a differently colored pixel, keep that trigger pixel and extend rays from it in the perpendicular orthogonal directions until the canvas boundary, writing only onto background cells.
    - After all drawing finishes, crop the result to the smallest rectangle containing the constructed segments and rays.

    Key insight:
    - The meaningful blue clues are identified by border pairing, not merely by sharing a row or column. Those edge-to-edge strokes define the active geometry, and only that geometry determines the final cropped frame.
    """

    grid = np.asarray(grid, dtype=int)
    h, w = grid.shape
    out = grid.copy()

    bg = get_bg(grid)
    non_bg = [(r, c, int(grid[r, c])) for r in range(h) for c in range(w) if int(grid[r, c]) != bg]
    assert non_bg, "grid must contain at least one non-background pixel"

    by_color: dict[int, list[tuple[int, int]]] = {}
    for r, c, col in non_bg:
        by_color.setdefault(col, []).append((r, c))

    best_score: tuple[int, int, int] | None = None
    line_color: int | None = None
    line_segments: list[tuple[tuple[int, int], tuple[int, int]]] = []
    for col, pts in by_color.items():
        segments = _segments_for_color(pts, h, w)
        if not segments:
            continue
        horiz = sum(1 for p, q in segments if p[0] == q[0])
        vert = len(segments) - horiz
        score = (len(segments), max(horiz, vert), len(pts))
        if best_score is None or score > best_score:
            best_score = score
            line_color = int(col)
            line_segments = segments

    assert line_color is not None, "at least one opposite-edge same-color pair must exist"

    triggers: dict[tuple[int, int, int], set[tuple[int, int]]] = {}
    active_cells: set[tuple[int, int]] = set()

    for p, q in line_segments:
        path = _straight_path(p, q)
        active_cells.update(path)
        perp_dirs = _segment_dirs(p, q)
        for r, c in path:
            v = int(out[r, c])
            if v == bg:
                out[r, c] = int(line_color)
            elif v != int(line_color):
                key = (r, c, v)
                triggers.setdefault(key, set()).update(perp_dirs)

    for (r0, c0, tcol), dirs in triggers.items():
        for ddr, ddc in dirs:
            r, c = r0 + ddr, c0 + ddc
            while 0 <= r < h and 0 <= c < w:
                active_cells.add((r, c))
                if int(out[r, c]) == bg:
                    out[r, c] = int(tcol)
                r += ddr
                c += ddc

    assert active_cells, "at least one active segment must exist"
    return _crop_grid(out, _bbox_from_cells(active_cells))


# --------------- generator (combo) ---------------

# === CFG: canvas and structural bounds ===
_H_RANGE = (8, 25)
_W_RANGE = (9, 21)
_LINE_COLOR = 1
_TRIGGER_COLOR = 6
_SEGMENT_COUNT_CHOICES = [1, 2, 3, 4]
_SEGMENT_COUNT_WEIGHTS = [1, 3, 3, 2]
_ORIENTATIONS = ("h", "v")

# Real triggers are disabled in generation because, under D6 edge-spanning
# strokes, any true trigger would expand the cropped bbox back to the full grid.
_REAL_TRIGGERS_PER_SEG = (0, 0)

# === CFG: distractor counts ===
_UNPAIRED_LINE_PIXELS = (0, 2)
_OFF_PATH_TRIGGERS = (0, 4)

# === CFG: retry limits ===
_DISTRACTOR_RETRIES = 80
_GEN_RETRIES = 400


def _compute_ray_cells(
    triggers: list[tuple[int, int]], perp_dirs: list[tuple[int, int]], H: int, W: int
) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for tr, tc in triggers:
        for ddr, ddc in perp_dirs:
            r, c = tr + ddr, tc + ddc
            while 0 <= r < H and 0 <= c < W:
                cells.add((r, c))
                r += ddr
                c += ddc
    return cells


def _build_expected(
    inp: np.ndarray,
    bg: int,
    segments: list[tuple[str, tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]],
) -> tuple[np.ndarray, tuple[int, int, int, int], set[tuple[int, int]]]:
    H, W = inp.shape
    expected = inp.copy()
    active_cells: set[tuple[int, int]] = set()

    for _, p, q, trigger_positions, perp_dirs in segments:
        path = _straight_path(p, q)
        active_cells.update(path)
        for r, c in path:
            if int(expected[r, c]) == bg:
                expected[r, c] = _LINE_COLOR
        for tr, tc in trigger_positions:
            for ddr, ddc in perp_dirs:
                r, c = tr + ddr, tc + ddc
                while 0 <= r < H and 0 <= c < W:
                    active_cells.add((r, c))
                    if int(expected[r, c]) == bg:
                        expected[r, c] = _TRIGGER_COLOR
                    r += ddr
                    c += ddc

    bbox = _bbox_from_cells(active_cells)
    return _crop_grid(expected, bbox), bbox, active_cells


def _bbox_is_full(bbox: tuple[int, int, int, int], H: int, W: int) -> bool:
    return bbox == (0, 0, H - 1, W - 1)


def _in_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return r0 <= r <= r1 and c0 <= c <= c1


def _has_control_diagonal_conflict(orient: str, indices: list[int], H: int, W: int) -> bool:
    target = (W - 1) if orient == "h" else (H - 1)
    for i, idx0 in enumerate(indices):
        for idx1 in indices[i + 1 :]:
            if abs(idx0 - idx1) == target:
                return True
    return False


def _active_span_indices(orient: str, H: int, W: int, n_segments: int) -> list[int]:
    if orient == "h":
        candidates = list(range(1, H - 1))
    else:
        candidates = list(range(1, W - 1))
    if len(candidates) < n_segments:
        raise ValueError("not enough interior rows/cols for requested segments")
    for _ in range(_DISTRACTOR_RETRIES):
        indices = sorted(random.sample(candidates, n_segments))
        if not _has_control_diagonal_conflict(orient, indices, H, W):
            return indices
    raise ValueError("failed to sample non-diagonal control indices")


def _make_segments(
    orient: str, indices: list[int], H: int, W: int
) -> list[tuple[str, tuple[int, int], tuple[int, int], list[tuple[int, int]], list[tuple[int, int]]]]:
    segments = []
    for idx in indices:
        if orient == "h":
            p = (idx, 0)
            q = (idx, W - 1)
        else:
            p = (0, idx)
            q = (H - 1, idx)
        path = _straight_path(p, q)
        path_interior = path[1:-1]
        max_triggers = min(_REAL_TRIGGERS_PER_SEG[1], len(path_interior))
        n_triggers = random.randint(_REAL_TRIGGERS_PER_SEG[0], max_triggers)
        chosen_triggers = random.sample(path_interior, n_triggers) if n_triggers > 0 else []
        segments.append((orient, p, q, chosen_triggers, _segment_dirs(p, q)))
    return segments


def _place_blue_distractors(
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    active_cells: set[tuple[int, int]],
    orient: str,
    used_indices: set[int],
) -> None:
    H, W = inp.shape
    blue_positions = {tuple(map(int, rc)) for rc in np.argwhere(inp == _LINE_COLOR)}
    for _ in range(random.randint(*_UNPAIRED_LINE_PIXELS)):
        for _ in range(_DISTRACTOR_RETRIES):
            r = random.randint(1, H - 2)
            c = random.randint(1, W - 2)
            pos = (r, c)
            if pos in occupied or pos in active_cells:
                continue
            if orient == "h" and r in used_indices:
                continue
            if orient == "v" and c in used_indices:
                continue
            if any(r == br or c == bc or (abs(r - br) == abs(c - bc) and r != br) for br, bc in blue_positions):
                continue
            inp[pos] = _LINE_COLOR
            occupied.add(pos)
            blue_positions.add(pos)
            break


def _candidate_trigger_cells(
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    active_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    orient: str,
    used_indices: set[int],
) -> list[tuple[int, int]]:
    H, W = inp.shape
    cells: list[tuple[int, int]] = []
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            pos = (r, c)
            if pos in occupied or pos in active_cells:
                continue
            if orient == "h" and r in used_indices:
                continue
            if orient == "v" and c in used_indices:
                continue
            if not _in_bbox(pos, bbox):
                cells.append(pos)
    if cells:
        return cells
    for r in range(1, H - 1):
        for c in range(1, W - 1):
            pos = (r, c)
            if pos in occupied or pos in active_cells:
                continue
            if orient == "h" and r in used_indices:
                continue
            if orient == "v" and c in used_indices:
                continue
            cells.append(pos)
    return cells


def _place_trigger_distractors(
    inp: np.ndarray,
    occupied: set[tuple[int, int]],
    active_cells: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    orient: str,
    used_indices: set[int],
) -> None:
    candidates = _candidate_trigger_cells(inp, occupied, active_cells, bbox, orient, used_indices)
    random.shuffle(candidates)
    want = random.randint(*_OFF_PATH_TRIGGERS)
    placed: list[tuple[int, int]] = []
    for pos in candidates:
        if len(placed) >= want:
            break
        r, c = pos
        if any(r == pr or c == pc or (abs(r - pr) == abs(c - pc) and r != pr) for pr, pc in placed):
            continue
        inp[pos] = _TRIGGER_COLOR
        occupied.add(pos)
        placed.append(pos)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*_H_RANGE)
    W = random.randint(*_W_RANGE)
    bg = random.choice([c for c in range(10) if c not in (_LINE_COLOR, _TRIGGER_COLOR)])

    orient = random.choice(_ORIENTATIONS)
    max_segments = (H - 2) if orient == "h" else (W - 2)
    n_segments = min(random.choices(_SEGMENT_COUNT_CHOICES, weights=_SEGMENT_COUNT_WEIGHTS, k=1)[0], max_segments)
    if n_segments <= 0:
        return None

    try:
        indices = _active_span_indices(orient, H, W, n_segments)
    except ValueError:
        return None

    segments = _make_segments(orient, indices, H, W)

    inp = np.full((H, W), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()
    for _, p, q, trigger_positions, _ in segments:
        inp[p] = _LINE_COLOR
        inp[q] = _LINE_COLOR
        occupied.add(p)
        occupied.add(q)
        for tp in trigger_positions:
            inp[tp] = _TRIGGER_COLOR
            occupied.add(tp)

    expected, bbox, active_cells = _build_expected(inp, bg, segments)
    if _bbox_is_full(bbox, H, W):
        return None

    used_indices = set(indices)
    _place_blue_distractors(inp, occupied, active_cells, orient, used_indices)
    _place_trigger_distractors(inp, occupied, active_cells, bbox, orient, used_indices)

    try:
        result = main(inp)
    except Exception:
        return None

    expected, bbox, active_cells = _build_expected(inp, bg, segments)
    if _bbox_is_full(bbox, H, W):
        return None
    if np.array_equal(result, expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input after {_GEN_RETRIES} attempts")
