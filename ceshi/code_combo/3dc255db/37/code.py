# Combo variant of 3dc255db (index 37)
# combo_spec: D2/2+D3/1+D9/0
# Name: 箭头方向+矩形框架+边缘控制像素

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_SOLVERS = Path("/home/lixinyu/snap/new/solvers")
if str(_SOLVERS) not in sys.path:
    sys.path.append(str(_SOLVERS))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


"""
Design analysis
===============

a) Concrete behavior and trace
   - Original task: a hollow Y/funnel contains source pixels inside the mouth. The tail direction is
     inferred from the cup body itself, then the mouth pixels are erased and shot outward.
   - This combo changes all three of those anchors at once:
       1. the cup body is now a 3-sided rectangular frame,
       2. tail direction comes from a same-color 3-pixel L arrow attached beside the closed side,
       3. the source is not inside the cup at all; it is a set of control pixels on the outer grid edge
          on the mouth side of that cup.
   - Small trace: suppose a cup body is a tail-up rectangle, with a small same-color arrow above it and
     four orange controls on the bottom border under the cup's column span. The solver must remove the
     arrow conceptually, recognize the remaining component as an upward rectangular cup, infer UP from
     the arrow placement, count the four orange edge controls, erase those controls, and draw four orange
     pixels upward from the center of the cup's top edge.
   - The old logic fails at every changed stage:
       1. the merged border component is no longer just the cup body,
       2. the rectangular frame has no "unique edge pixel" spike heuristic,
       3. the mouth interior is empty, so there is no in-cup content to discover.

b) Complexity calibration
   - The arrow is fixed to a 3-cell L triomino, reused across all examples, so direction remains
     visually learnable rather than arbitrary.
   - Rectangular cups still vary in size and orientation, but only over a small bounded family.
   - Edge controls are simple single pixels, but each cup claims exactly one border segment aligned with
     its mouth. That keeps matching visible without making color the sole cue.
   - Content colors are unique across cups in generated data. This is a compatibility choice to avoid
     hidden ambiguity once the source leaves the cup interior.

c) Implicit-assumption audit for the original main()
   - "Full component equals cup body" is false because the arrow is attached to the border component.
   - "Tail is the bbox side with exactly one border pixel" is false because the cup body is rectangular.
   - "Correct candidate flood reaches content inside the cup" is false because the cup interior is empty.
   - "Erased source cells lie inside the mouth" is false because the source is now on the global border.
   - New solver chain:
       1. enumerate 3-cell L arrow candidates on each border component,
       2. remove one candidate and require the remainder to be an exact 3-sided rectangle,
       3. infer direction from the arrow side relative to the recovered cup bbox,
       4. read same-colored control pixels from the opposite outer border segment,
       5. erase those controls and eject from the center of the closed side.

d) Placement legality
   - The generator keeps each cup body at least 1 cell away from the outer border so mouth-side border
     controls never overlap the cup itself. The arrow may approach or touch a non-control border side;
     solver-side cup masking handles that safely.
   - Cup+arrow components keep at least a 1-cell gap from each other via 8-neighborhood dilation.
   - Ejection rays are reserved globally so no cup body, arrow, or later ray occupies another ray.
   - Cups claiming controls from the same outer side are forced to use disjoint projected intervals on
     that side, preventing one control pixel from belonging to multiple cups.

e) Visual observability
   - Across examples a human sees: empty rectangular cups, same-color arrows attached near the closed
     side, and colored edge dots on the opposite outer border that disappear while a ray of that color
     appears from the cup center.
   - Direction is not encoded by numeric color id; it is visible from local arrow placement.
   - Source matching is not purely by color either; the controls sit on the cup's mouth-side border span.

f) Visual cleanliness
   - The mouth-side border-control band stays clean because stray pixels are forbidden from the outermost
     border and cups never occupy their own control side on the border.
   - The cup interior is empty, so the region around each cup is less cluttered than the source task.
   - The arrow sits off the ejection ray, so it remains visible in both input and output.

g) Change inferability
   - The fusion remains structurally inferable in 2-4 examples because each of the three edits is
     directly visible: the cup geometry is rectangular, direction is a local arrow marker, and the
     disappearing source pixels are on the grid edge rather than in the cup.

Post-change solver assumptions and generator guarantees
-------------------------------------------------------
   - Assumption: each relevant border component contains exactly one removable 3-cell L arrow whose
     removal yields an exact rectangular cup. Guaranteed by `_attach_arrow`.
   - Assumption: the arrow lies completely outside exactly one side of the recovered cup bbox. Guaranteed
     by the fixed arrow templates.
   - Assumption: each cup's mouth-side border segment has at least one control cell and is not shared with
     another cup on that same side. Guaranteed by placement and control allocation.
   - Assumption: the cup interior is empty, so any non-bg border-edge pixel outside cup components is a
     control pixel rather than in-cup content. Guaranteed because cup bodies stay off the control border
     segment and stray pixels never use the border band.
"""


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.35
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500

ARROW_OFFSETS = {
    "U": (
        [(-2, -2), (-2, -1), (-1, -1)],
        [(-2, 1), (-2, 2), (-1, 1)],
    ),
    "D": (
        [(1, -1), (2, -2), (2, -1)],
        [(1, 1), (2, 1), (2, 2)],
    ),
    "L": (
        [(-2, -2), (-1, -2), (-1, -1)],
        [(1, -2), (1, -1), (2, -2)],
    ),
    "R": (
        [(-2, 1), (-1, 1), (-1, 2)],
        [(1, 1), (1, 2), (2, 2)],
    ),
}


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str
) -> set[tuple[int, int]]:
    """4-neighbor flood within bbox, seeded from non-border cells on one bbox edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:
        seeds = [(r, c1) for r in range(r0, r1 + 1) if (r, c1) not in border_set]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _is_connected8(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == len(cells)


def _enumerate_arrow_candidates(border_set: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Enumerate unique 3-cell L triominoes found in 2x2 windows of the component bbox."""
    rs = [r for r, _ in border_set]
    cs = [c for _, c in border_set]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    seen: set[frozenset[tuple[int, int]]] = set()
    for wr in range(r0, r1):
        for wc in range(c0, c1):
            window = {(wr, wc), (wr, wc + 1), (wr + 1, wc), (wr + 1, wc + 1)}
            cells = border_set & window
            if len(cells) == 3:
                seen.add(frozenset(cells))
    return [set(x) for x in seen]


def _rect_frame_cells(bbox: tuple[int, int, int, int], tail_dir: str) -> set[tuple[int, int]]:
    """Exact 3-sided rectangular frame cells for a recovered cup bbox and tail direction."""
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if tail_dir == "U":
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif tail_dir == "D":
        cells.update((r1, c) for c in range(c0, c1 + 1))
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r, c1) for r in range(r0, r1 + 1))
    elif tail_dir == "L":
        cells.update((r, c0) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    else:
        cells.update((r, c1) for r in range(r0, r1 + 1))
        cells.update((r0, c) for c in range(c0, c1 + 1))
        cells.update((r1, c) for c in range(c0, c1 + 1))
    return cells


def _spike_from_bbox_center(bbox: tuple[int, int, int, int], tail_dir: str) -> tuple[int, int] | None:
    """Center cell of the closed side; undefined when the closed-side length is even."""
    r0, c0, r1, c1 = bbox
    if tail_dir in ("U", "D"):
        width = c1 - c0 + 1
        if width % 2 == 0:
            return None
        return (r0 if tail_dir == "U" else r1, c0 + width // 2)
    height = r1 - r0 + 1
    if height % 2 == 0:
        return None
    return (r0 + height // 2, c0 if tail_dir == "L" else c1)


def _arrow_side(arrow_cells: set[tuple[int, int]], cup_bbox: tuple[int, int, int, int]) -> str | None:
    ar0, ac0, ar1, ac1 = bbox_coords(np.array(sorted(arrow_cells), dtype=int))
    r0, c0, r1, c1 = cup_bbox
    if ar1 < r0:
        return "U"
    if ar0 > r1:
        return "D"
    if ac1 < c0:
        return "L"
    if ac0 > c1:
        return "R"
    return None


def _edge_segment_for_bbox(
    H: int, W: int, bbox: tuple[int, int, int, int], side: str
) -> list[tuple[int, int]]:
    """Outer border cells aligned with the cup bbox projection on one side."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return [(0, c) for c in range(c0, c1 + 1)]
    if side == "D":
        return [(H - 1, c) for c in range(c0, c1 + 1)]
    if side == "L":
        return [(r, 0) for r in range(r0, r1 + 1)]
    return [(r, W - 1) for r in range(r0, r1 + 1)]


def _recover_cup_from_component(border_set: set[tuple[int, int]]) -> dict | None:
    """Recover the cup body by stripping one arrow candidate from the merged component."""
    passing: list[dict] = []
    for arrow_cells in _enumerate_arrow_candidates(border_set):
        cup_set = border_set - arrow_cells
        if len(cup_set) < 7 or not _is_connected8(cup_set):
            continue

        cup_bbox = bbox_coords(np.array(sorted(cup_set), dtype=int))
        tail_dir = _arrow_side(arrow_cells, cup_bbox)
        if tail_dir is None:
            continue
        if cup_set != _rect_frame_cells(cup_bbox, tail_dir):
            continue

        spike = _spike_from_bbox_center(cup_bbox, tail_dir)
        if spike is None or spike not in cup_set:
            continue

        mouth = _flood_from_bbox_side(cup_set, cup_bbox, OPPOSITE[tail_dir])
        if len(mouth) < 3:
            continue

        passing.append(
            {
                "cup_bbox": cup_bbox,
                "tail_dir": tail_dir,
                "spike": spike,
                "arrow_cells": set(arrow_cells),
                "cup_set": cup_set,
            }
        )

    if len(passing) != 1:
        return None
    return passing[0]


def _find_cups(g: np.ndarray, bg: int) -> list[dict]:
    cups: list[dict] = []
    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 10:
                continue
            border_set = {(int(r), int(c)) for r, c in border_coords}
            recovered = _recover_cup_from_component(border_set)
            if recovered is None:
                continue
            combo_bbox = bbox_coords(border_coords)
            cups.append(
                {
                    "border_color": int(border_col),
                    "border_set": border_set,
                    "combo_bbox": combo_bbox,
                    **recovered,
                }
            )
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Cups are rectangular 3-sided frames.
    - A same-color 3-pixel arrow attached beside the closed side tells the ejection direction.
    - The source pixels are controls on the outer grid edge on the opposite side of the cup.
    - Controls are erased; a line of that color is ejected from the center of the closed side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    cups = _find_cups(g, bg)
    cup_mask = np.zeros((H, W), dtype=bool)
    for cup in cups:
        for r, c in cup["border_set"]:
            cup_mask[r, c] = True

    border_coords = set()
    for c in range(W):
        border_coords.add((0, c))
        border_coords.add((H - 1, c))
    for r in range(H):
        border_coords.add((r, 0))
        border_coords.add((r, W - 1))

    edge_controls: dict[tuple[int, int], int] = {}
    for r, c in border_coords:
        if cup_mask[r, c]:
            continue
        v = int(g[r, c])
        if v != bg:
            edge_controls[(r, c)] = v

    for cup in cups:
        ctrl_side = OPPOSITE[cup["tail_dir"]]
        ctrl_segment = _edge_segment_for_bbox(H, W, cup["cup_bbox"], ctrl_side)
        ctrl_cells = [p for p in ctrl_segment if p in edge_controls]
        if not ctrl_cells:
            continue

        ctrl_colors = {edge_controls[p] for p in ctrl_cells}
        if len(ctrl_colors) != 1:
            continue
        content_col = int(next(iter(ctrl_colors)))
        if content_col == int(cup["border_color"]):
            continue

        for r, c in ctrl_cells:
            out[r, c] = bg

        dr, dc = DIR4[cup["tail_dir"]]
        sr, sc = cup["spike"]
        for t in range(1, len(ctrl_cells) + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = content_col

    return out


def _build_canonical_cup(height: int, width: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a tail-up rectangular cup with an open bottom side."""
    assert height >= 3 and width >= 3 and width % 2 == 1
    g = np.zeros((height, width), dtype=int)
    g[0, :] = color
    g[:, 0] = color
    g[:, width - 1] = color
    return g, (0, width // 2)


def _mouth_interior_coords(
    patch: np.ndarray, border_color: int, opposite_side: str
) -> list[tuple[int, int]]:
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:
        seeds = [(r, w - 1) for r in range(h) if not border_mask[r, w - 1]]

    visited = set(seeds)
    stack = list(seeds)
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], direction: str
) -> tuple[np.ndarray, tuple[int, int]]:
    """Rotate a canonical tail-up patch and spike into the requested direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc)
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr)
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr)
    raise ValueError(direction)


def _sample_cup_shape(
    rng: np.random.Generator, direction: str, compact_level: int = 0
) -> tuple[int, int] | None:
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    shapes = [
        (h, w)
        for h in range(3, max_h_canon + 1)
        for w in range(3, max_w_canon + 1, 2)
        if (h - 1) * (w - 2) >= 3
    ]
    if not shapes:
        return None
    shapes.sort(key=lambda hw: (hw[0] * hw[1], hw[0], hw[1]))
    if compact_level >= 2:
        shapes = shapes[: max(1, len(shapes) // 3)]
    elif compact_level == 1:
        shapes = shapes[: max(1, (2 * len(shapes)) // 3)]
    return shapes[int(rng.integers(0, len(shapes)))]


def _attach_arrow(
    rng: np.random.Generator,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    color: int,
) -> tuple[np.ndarray, tuple[int, int], tuple[int, int, int, int]]:
    """Attach the same-color L arrow beside the closed side of the rectangular cup."""
    border_coords = [(int(r), int(c)) for r, c in np.argwhere(patch == color)]
    arrow_template = ARROW_OFFSETS[direction][int(rng.integers(0, 2))]
    arrow_coords = [(spike[0] + dr, spike[1] + dc) for dr, dc in arrow_template]

    all_coords = border_coords + arrow_coords
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_r = max(r for r, _ in all_coords)
    max_c = max(c for _, c in all_coords)

    combo = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in border_coords:
        combo[r - min_r, c - min_c] = color
    for r, c in arrow_coords:
        combo[r - min_r, c - min_c] = color

    shifted_spike = (spike[0] - min_r, spike[1] - min_c)
    cup_bbox_local = (-min_r, -min_c, -min_r + patch.shape[0] - 1, -min_c + patch.shape[1] - 1)
    return combo, shifted_spike, cup_bbox_local


def _ray_inside_patch_is_clear(patch: np.ndarray, spike: tuple[int, int], direction: str) -> bool:
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    while 0 <= rr < ph and 0 <= cc < pw:
        if patch[rr, cc] != 0:
            return False
        rr += dr
        cc += dc
    return True


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _control_interval_for_bbox(bbox: tuple[int, int, int, int], ctrl_side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    if ctrl_side in ("U", "D"):
        return (c0, c1)
    return (r0, r1)


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    cup_bbox_local: tuple[int, int, int, int],
    occ_combo: np.ndarray,
    occ_path: np.ndarray,
    edge_claims: dict[str, list[tuple[int, int]]],
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int, int, int], tuple[int, int]] | None:
    """Place the combined cup+arrow patch away from the border with a clear reserved ray."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    ctrl_side = OPPOSITE[direction]
    occ_combo_dilated = ndimage.binary_dilation(occ_combo, structure=_STRUCT8)
    patch_mask = patch != 0

    if not _ray_inside_patch_is_clear(patch, spike, direction):
        return None

    min_r0 = max(0, 1 - cup_bbox_local[0])
    max_r0 = min(H - ph, (H - 2) - cup_bbox_local[2])
    min_c0 = max(0, 1 - cup_bbox_local[1])
    max_c0 = min(W - pw, (W - 2) - cup_bbox_local[3])
    if min_r0 > max_r0 or min_c0 > max_c0:
        return None

    row_range = range(min_r0, max_r0 + 1)
    col_range = range(min_c0, max_c0 + 1)
    coords_list = [(r0, c0) for r0 in row_range for c0 in col_range]
    rng.shuffle(coords_list)

    cr0, cc0, cr1, cc1 = cup_bbox_local
    for r0, c0 in coords_list:
        if occ_combo_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if np.any(occ_path[r0:r0 + ph, c0:c0 + pw] & patch_mask):
            continue

        cup_bbox_abs = (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1)
        ctrl_interval = _control_interval_for_bbox(cup_bbox_abs, ctrl_side)
        if any(_intervals_overlap(ctrl_interval, other) for other in edge_claims[ctrl_side]):
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((rr, cc))
            rr += dr
            cc += dc
        if not path_cells:
            continue
        if any(occ_combo[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        return r0, c0, path_cells, cup_bbox_abs, ctrl_interval
    return None


def _stray_candidate_mask(
    H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str
) -> np.ndarray:
    """Cells on the mouth side of the cup bbox, excluding the outer border control band."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, 1:c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:W - 1] = True
    elif tail_dir == "D":
        mask[1:r0, c0:c1 + 1] = True
    else:
        mask[r1 + 1:H - 1, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    forbidden[0, :] = True
    forbidden[H - 1, :] = True
    forbidden[:, 0] = True
    forbidden[:, W - 1] = True
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["combo_bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["cup_bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return

    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[int(r), int(c)] = stray_color
    canvas_out[int(r), int(c)] = stray_color


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_combo: np.ndarray,
    occ_path: np.ndarray,
    edge_claims: dict[str, list[tuple[int, int]]],
    compact_level: int,
) -> dict | None:
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction, compact_level=compact_level)
    if shape is None:
        return None
    height, width = shape

    patch_canon, spike_canon = _build_canonical_cup(height, width, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch_body, spike_body = _rotate_to_direction(patch_canon, spike_canon, direction)
    patch, spike, cup_bbox_local = _attach_arrow(rng, patch_body, spike_body, direction, border_color)

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, cup_bbox_local, occ_combo, occ_path, edge_claims
    )
    if placement is None:
        return None
    r0, c0, path_cells, cup_bbox_abs, ctrl_interval = placement

    ph, pw = patch.shape
    patch_mask = patch != 0

    sub_in = canvas_in[r0:r0 + ph, c0:c0 + pw]
    sub_out = canvas_out[r0:r0 + ph, c0:c0 + pw]
    sub_occ = occ_combo[r0:r0 + ph, c0:c0 + pw]
    sub_in[patch_mask] = border_color
    sub_out[patch_mask] = border_color
    sub_occ[patch_mask] = True

    for r, c in path_cells:
        occ_path[r, c] = True

    edge_claims[OPPOSITE[direction]].append(ctrl_interval)
    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "cup_bbox": cup_bbox_abs,
        "combo_bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "spike_abs": (r0 + spike[0], c0 + spike[1]),
        "path_cells": path_cells,
        "control_side": OPPOSITE[direction],
        "control_interval": ctrl_interval,
    }


def _place_controls_and_ejections(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> bool:
    H, W = canvas_in.shape
    blocked = np.zeros((H, W), dtype=bool)
    for meta in cups_meta:
        for r, c in meta["path_cells"]:
            blocked[r, c] = True

    for meta in cups_meta:
        candidates = [
            (r, c)
            for (r, c) in _edge_segment_for_bbox(H, W, meta["cup_bbox"], meta["control_side"])
            if not blocked[r, c]
        ]
        max_count = min(CONTENT_COUNT_MAX, len(candidates))
        if max_count < 1:
            return False

        count = int(rng.integers(1, max_count + 1))
        idxs = rng.choice(len(candidates), size=count, replace=False)
        ctrl_cells = [candidates[int(i)] for i in idxs]

        for r, c in ctrl_cells:
            canvas_in[r, c] = meta["content_color"]

        meta["content_count"] = count
        meta["control_cells"] = ctrl_cells
        for r, c in meta["path_cells"][:count]:
            canvas_out[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < 2 * num_cups:
        return None

    occ_combo = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    edge_claims = {"U": [], "D": [], "L": [], "R": []}
    compact_level = 2 if num_cups >= 3 else 1 if num_cups == 2 else 0

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        meta = _add_cup(
            rng,
            H,
            W,
            border_color,
            content_color,
            canvas_in,
            canvas_out,
            occ_combo,
            occ_path,
            edge_claims,
            compact_level,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if not _place_controls_and_ejections(rng, canvas_in, canvas_out, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
        return None
    return canvas_in


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups)
        if result is not None:
            return result
    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
