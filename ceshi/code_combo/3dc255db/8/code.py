# Combo variant of 3dc255db (index 8)
# combo_spec: D2/2+D9/1
# Name: 箭头指示方向+L形框架

"""
Fusion design analysis
======================

a) Concretization and behavior trace
------------------------------------
- Each cup is now a 1-pixel-thick axis-aligned L border, plus a separate same-color
  3-pixel L-triomino arrow placed just outside one side of that cup.
- The L body alone no longer determines which endpoint is the spike. For the same
  visual L frame, either endpoint could be the ejecting endpoint; the nearby arrow
  selects which side is active.
- Small trace:
  1. Original task: a funnel/Y frame has one geometric tail spike, so the solver
     reads the frame itself to get direction.
  2. Single D9/1: an L frame replaces the funnel, and the shorter-leg endpoint is
     treated as the spike.
  3. Fused task here: the cup is still an L frame, but the spike endpoint is the
     endpoint on the side indicated by the external arrow component. The content
     rectangle is reconstructed from the L path after orienting the path so that
     the arrow-selected endpoint is first.
  4. The content pixels inside that concave rectangle are erased and re-emitted as
     a straight line from that endpoint in the arrow-indicated direction.
- Difference from original:
  - shape cue changed: funnel -> L frame
  - direction cue changed: inferred from shape -> read from nearby arrow component

b) Complexity calibration
-------------------------
- The arrow vocabulary is tiny: one fixed 3-cell 2x2-L component, placed on one of
  four sides of the cup. That is visually explicit without adding many symbol types.
- The L body keeps only one bend and two endpoints, so the rule stays legible.
- To avoid degeneration back into D9/1, the generator deliberately samples all
  three leg relations for the arrow-selected endpoint: shorter, equal, and longer
  than the other endpoint's first leg. Equal-leg cups especially force the arrow
  to be the deciding cue.

c) Implicit assumption audit against the original main()
--------------------------------------------------------
- Original assumption: one border component is the whole cup body. Broken now:
  each cup color appears as two components, the body and its arrow.
- Original assumption: the tail is a unique non-corner singleton on a bbox edge.
  Broken by the L frame because endpoints are corners.
- D9/1 assumption: the shorter-leg endpoint is always the spike. Broken here by
  construction; the arrow can choose the longer endpoint or an equal-leg case.
- New solver logic therefore is:
  1. find L-body components and L-arrow components separately,
  2. pair an L body with the unique same-color arrow lying completely beyond one
     side of its bbox,
  3. orient the body path so that its tail direction matches that arrow side,
  4. reconstruct the concave content rectangle from that orientation.

d) Placement legality
---------------------
- The generator builds each cup output-first: body + arrow + ejection line, then
  reconstructs the input by scattering the content inside the inferred rectangle.
- Cup+arrow combo bboxes keep a 1-cell gap from each other via dilated placement.
- The ejection ray from the selected endpoint is checked to be clear both inside
  the local combo patch and globally to the grid edge.
- Stray pixels remain outside all combo bboxes and all ejection paths.

e) Visual observability
-----------------------
- Across a few examples, humans can see that the body is always an L and that the
  output line leaves from the endpoint nearest the small same-color arrow.
- Because equal-leg and arrow-to-longer-endpoint cases are common, shape-only
  explanations fail; the arrow is the only stable cue.

f) Visual cleanliness
---------------------
- The arrow is kept close to its cup but separated by at least one empty row/col,
  so the two objects are distinct under 8-connectivity.
- The arrow is offset away from the exact ejection ray, so it remains visible in
  both input and output and does not get overwritten by the emitted line.

g) Change inferability
----------------------
- No color remapping is introduced. The arrow simply reuses the cup border color.
- Border colors are unique per cup, and content colors are chosen disjoint from
  border colors, so humans and the solver can associate each arrow to its cup by
  local color identity plus adjacency.

Post-change solver assumptions and generator guarantees
-------------------------------------------------------
- Assumption: each cup border color has exactly one L-body component and one arrow
  component of that same color. Guaranteed by the generator.
- Assumption: the arrow lies completely beyond exactly one side of the cup body's
  bbox while still overlapping its perpendicular span. Guaranteed by fixed local
  arrow templates.
- Assumption: among the two possible path orientations of the L body, exactly one
  has a tail direction matching the arrow side. Guaranteed by how each cup is built.
- Assumption: the inferred content rectangle contains at least one non-border color
  and has a unique majority color. Guaranteed because each cup gets one dedicated
  content color scattered only inside its own rectangle.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
VEC_TO_DIR = {v: k for k, v in DIR4.items()}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
ORTHOGONAL = {
    "U": ("L", "R"),
    "D": ("L", "R"),
    "L": ("U", "D"),
    "R": ("U", "D"),
}
RELATIONS = ("tail_short", "equal", "tail_long")
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _component_bbox(cell_set: set[tuple[int, int]]) -> tuple[int, int, int, int]:
    return bbox_coords(np.array(sorted(cell_set), dtype=int))


def _neighbors4_in_set(cell: tuple[int, int], cell_set: set[tuple[int, int]]) -> list[tuple[int, int]]:
    r, c = cell
    out = []
    for dr, dc in DIR4.values():
        nxt = (r + dr, c + dc)
        if nxt in cell_set:
            out.append(nxt)
    return out


def _extract_l_path(border_set: set[tuple[int, int]]) -> list[tuple[int, int]] | None:
    """Return the ordered path of a simple 4-connected L component, or None."""
    if not border_set:
        return None

    neighbors = {cell: _neighbors4_in_set(cell, border_set) for cell in border_set}
    endpoints = [cell for cell, ns in neighbors.items() if len(ns) == 1]
    if len(endpoints) != 2:
        return None
    if any(len(ns) not in (1, 2) for ns in neighbors.values()):
        return None

    start, goal = endpoints
    path = [start]
    prev = None
    cur = start
    while cur != goal:
        nxts = [nxt for nxt in neighbors[cur] if nxt != prev]
        if len(nxts) != 1:
            return None
        nxt = nxts[0]
        path.append(nxt)
        prev, cur = cur, nxt
        if len(path) > len(border_set):
            return None

    if len(path) != len(border_set):
        return None

    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    if not dirs or any(d not in VEC_TO_DIR for d in dirs):
        return None

    runs: list[tuple[tuple[int, int], int]] = []
    for d in dirs:
        if runs and runs[-1][0] == d:
            runs[-1] = (d, runs[-1][1] + 1)
        else:
            runs.append((d, 1))
    if len(runs) != 2:
        return None
    if runs[0][0][0] * runs[1][0][0] + runs[0][0][1] * runs[1][0][1] != 0:
        return None
    return path


def _axis_values_excluding_end(a: int, b: int) -> list[int]:
    step = 1 if b > a else -1
    return list(range(a, b, step))


def _axis_values_excluding_start(a: int, b: int) -> list[int]:
    step = 1 if b > a else -1
    return list(range(a + step, b + step, step))


def _candidate_from_path(
    path: list[tuple[int, int]],
) -> tuple[str, str, tuple[int, int], list[tuple[int, int]]] | None:
    """Interpret ``path[0]`` as the spike endpoint and reconstruct its content rectangle."""
    dirs = [(b[0] - a[0], b[1] - a[1]) for a, b in zip(path, path[1:])]
    runs: list[tuple[tuple[int, int], int]] = []
    for d in dirs:
        if runs and runs[-1][0] == d:
            runs[-1] = (d, runs[-1][1] + 1)
        else:
            runs.append((d, 1))
    if len(runs) != 2:
        return None

    d1, first_len = runs[0]
    d2, _ = runs[1]
    spike = path[0]
    elbow = path[first_len]
    mouth = path[-1]

    if d1[0] != 0:
        row_vals = _axis_values_excluding_end(spike[0], elbow[0])
        col_vals = _axis_values_excluding_start(elbow[1], mouth[1])
    else:
        row_vals = _axis_values_excluding_start(elbow[0], mouth[0])
        col_vals = _axis_values_excluding_end(spike[1], elbow[1])

    if not row_vals or not col_vals:
        return None

    interior = [(r, c) for r in row_vals for c in col_vals]
    tail_dir = OPPOSITE[VEC_TO_DIR[d1]]
    mouth_dir = VEC_TO_DIR[d2]
    return tail_dir, mouth_dir, spike, interior


def _is_arrow_component(cell_set: set[tuple[int, int]]) -> bool:
    if len(cell_set) != 3:
        return False
    r0, c0, r1, c1 = _component_bbox(cell_set)
    return (r1 - r0 + 1, c1 - c0 + 1) == (2, 2)


def _arrow_side(
    arrow_cells: set[tuple[int, int]],
    cup_bbox: tuple[int, int, int, int],
) -> str | None:
    ar0, ac0, ar1, ac1 = _component_bbox(arrow_cells)
    r0, c0, r1, c1 = cup_bbox
    col_overlap = max(ac0, c0) <= min(ac1, c1)
    row_overlap = max(ar0, r0) <= min(ar1, r1)
    hits = []
    if ar1 < r0 and col_overlap:
        hits.append("U")
    if ar0 > r1 and col_overlap:
        hits.append("D")
    if ac1 < c0 and row_overlap:
        hits.append("L")
    if ac0 > c1 and row_overlap:
        hits.append("R")
    return hits[0] if len(hits) == 1 else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Each cup border color appears as two separate components: an L-shaped cup body and a
      same-color 3-cell arrow component on one side.
    - The arrow side selects which endpoint of the L is the spike.
    - Content inside the corresponding concave rectangle is erased and ejected from that
      endpoint in the arrow-indicated direction.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        comps = [
            {(int(r), int(c)) for r, c in coords}
            for coords in components_mask(g == int(border_col), connectivity=8)
            if coords.size > 0
        ]
        arrow_components = [comp for comp in comps if _is_arrow_component(comp)]
        body_components = []
        for comp in comps:
            if len(comp) < 5:
                continue
            path = _extract_l_path(comp)
            if path is not None:
                body_components.append((comp, path))

        for body_set, path in body_components:
            body_bbox = _component_bbox(body_set)
            matched_arrows = []
            for arrow_set in arrow_components:
                side = _arrow_side(arrow_set, body_bbox)
                if side is not None:
                    matched_arrows.append((side, arrow_set))
            if len(matched_arrows) != 1:
                continue

            tail_dir = matched_arrows[0][0]
            candidates = []
            for oriented_path in (path, list(reversed(path))):
                cand = _candidate_from_path(oriented_path)
                if cand is not None and cand[0] == tail_dir:
                    candidates.append(cand)
            if len(candidates) != 1:
                continue

            _tail_dir, _mouth_dir, spike, interior = candidates[0]
            cnt = Counter()
            for r, c in interior:
                if not (0 <= r < H and 0 <= c < W):
                    cnt.clear()
                    break
                v = int(g[r, c])
                if v != bg and v != int(border_col):
                    cnt[v] += 1
            if not cnt:
                continue

            top = cnt.most_common(2)
            if len(top) > 1 and top[0][1] == top[1][1]:
                continue
            content_col = int(top[0][0])
            content_cells = [p for p in interior if int(g[p[0], p[1]]) == content_col]
            if not content_cells:
                continue

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)  # body bbox when tail is U or D
CUP_BBOX_MAX_LR = (7, 8)  # body bbox when tail is L or R
CUP_BBOX_MIN = (3, 3)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300


def _relation_from_dims(tail_dir: str, height: int, width: int) -> str:
    first_len = height - 1 if tail_dir in ("U", "D") else width - 1
    second_len = width - 1 if tail_dir in ("U", "D") else height - 1
    if first_len < second_len:
        return "tail_short"
    if first_len > second_len:
        return "tail_long"
    return "equal"


def _sample_body_dims(rng: np.random.Generator, tail_dir: str, relation: str) -> tuple[int, int] | None:
    if tail_dir in ("U", "D"):
        max_h, max_w = CUP_BBOX_MAX_UD
    else:
        max_h, max_w = CUP_BBOX_MAX_LR

    min_h, min_w = CUP_BBOX_MIN
    candidates = []
    for height in range(min_h, max_h + 1):
        for width in range(min_w, max_w + 1):
            if _relation_from_dims(tail_dir, height, width) == relation:
                candidates.append((height, width))
    if not candidates:
        return None
    return candidates[int(rng.integers(0, len(candidates)))]


def _build_body_patch(
    tail_dir: str,
    mouth_dir: str,
    height: int,
    width: int,
    color: int,
) -> tuple[np.ndarray, list[tuple[int, int]], tuple[int, int], list[tuple[int, int]]]:
    d1 = DIR4[OPPOSITE[tail_dir]]
    d2 = DIR4[mouth_dir]
    first_len = height - 1 if tail_dir in ("U", "D") else width - 1
    second_len = width - 1 if tail_dir in ("U", "D") else height - 1

    path_raw = [(0, 0)]
    for i in range(1, first_len + 1):
        path_raw.append((path_raw[0][0] + d1[0] * i, path_raw[0][1] + d1[1] * i))
    elbow = path_raw[-1]
    for j in range(1, second_len + 1):
        path_raw.append((elbow[0] + d2[0] * j, elbow[1] + d2[1] * j))

    min_r = min(r for r, _ in path_raw)
    min_c = min(c for _, c in path_raw)
    path = [(r - min_r, c - min_c) for r, c in path_raw]
    cand = _candidate_from_path(path)
    if cand is None or cand[0] != tail_dir or cand[1] != mouth_dir:
        raise RuntimeError("body construction mismatch")
    _tail, _mouth, spike, interior = cand

    ph = max(r for r, _ in path) + 1
    pw = max(c for _, c in path) + 1
    patch = np.zeros((ph, pw), dtype=int)
    for r, c in path:
        patch[r, c] = color
    return patch, path, spike, interior


def _arrow_templates_for_body(
    tail_dir: str,
    spike: tuple[int, int],
    patch_shape: tuple[int, int],
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sr, sc = spike
    ph, pw = patch_shape
    if tail_dir == "U":
        if sr != 0:
            raise RuntimeError("unexpected spike row for U")
        if sc == 0:
            return [(-3, 1), (-3, 2), (-2, 1)], [(-3, 1), (-3, 2), (-2, 2)]
        if sc == pw - 1:
            return [(-3, -2), (-3, -1), (-2, -2)], [(-3, -2), (-3, -1), (-2, -1)]
    elif tail_dir == "D":
        if sr != ph - 1:
            raise RuntimeError("unexpected spike row for D")
        if sc == 0:
            return [(2, 1), (3, 1), (3, 2)], [(2, 2), (3, 1), (3, 2)]
        if sc == pw - 1:
            return [(2, -2), (3, -2), (3, -1)], [(2, -1), (3, -2), (3, -1)]
    elif tail_dir == "L":
        if sc != 0:
            raise RuntimeError("unexpected spike col for L")
        if sr == 0:
            return [(1, -3), (1, -2), (2, -3)], [(1, -3), (2, -3), (2, -2)]
        if sr == ph - 1:
            return [(-2, -3), (-2, -2), (-1, -3)], [(-2, -3), (-1, -3), (-1, -2)]
    else:  # "R"
        if sc != pw - 1:
            raise RuntimeError("unexpected spike col for R")
        if sr == 0:
            return [(1, 2), (1, 3), (2, 2)], [(1, 2), (2, 2), (2, 3)]
        if sr == ph - 1:
            return [(-2, 2), (-2, 3), (-1, 2)], [(-2, 2), (-1, 2), (-1, 3)]
    raise RuntimeError("unexpected spike corner")


def _attach_arrow(
    rng: np.random.Generator,
    body_patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    tail_dir: str,
    color: int,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]], tuple[int, int, int, int]]:
    body_coords = [(int(r), int(c)) for r, c in np.argwhere(body_patch == color)]
    arrow_template = _arrow_templates_for_body(tail_dir, spike, body_patch.shape)[int(rng.integers(0, 2))]
    arrow_coords = [(spike[0] + dr, spike[1] + dc) for dr, dc in arrow_template]

    all_coords = body_coords + arrow_coords
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_r = max(r for r, _ in all_coords)
    max_c = max(c for _, c in all_coords)

    combo = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in body_coords:
        combo[r - min_r, c - min_c] = color
    for r, c in arrow_coords:
        combo[r - min_r, c - min_c] = color

    shifted_spike = (spike[0] - min_r, spike[1] - min_c)
    shifted_mouth = [(r - min_r, c - min_c) for (r, c) in mouth]
    body_bbox = (-min_r, -min_c, -min_r + body_patch.shape[0] - 1, -min_c + body_patch.shape[1] - 1)
    return combo, shifted_spike, shifted_mouth, body_bbox


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


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    if not _ray_inside_patch_is_clear(patch, spike, direction):
        return None

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    patch_mask = patch != 0
    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if np.any(occ_path[r0:r0 + ph, c0:c0 + pw] & patch_mask):
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
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], mouth_dir: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if mouth_dir == "R":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif mouth_dir == "L":
        mask[r0:r1 + 1, :c0] = True
    elif mouth_dir == "D":
        mask[r1 + 1:, c0:c1 + 1] = True
    else:  # "U"
        mask[:r0, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["combo_bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    H, W = canvas_in.shape
    candidate_mask = _stray_candidate_mask(H, W, meta["body_bbox"], meta["mouth_dir"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return

    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    border_color: int,
    content_color: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    tail_dir = str(rng.choice(["U", "D", "L", "R"]))
    mouth_dir = str(rng.choice(ORTHOGONAL[tail_dir]))
    relation = str(rng.choice(RELATIONS))
    dims = _sample_body_dims(rng, tail_dir, relation)
    if dims is None:
        return None
    height, width = dims

    body_patch, _body_path, spike_body, mouth_body = _build_body_patch(
        tail_dir, mouth_dir, height, width, border_color
    )
    patch, spike, mouth, body_bbox_local = _attach_arrow(
        rng, body_patch, spike_body, mouth_body, tail_dir, border_color
    )
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, tail_dir, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement
    ph, pw = patch.shape

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    br0, bc0, br1, bc1 = body_bbox_local
    return {
        "border_color": border_color,
        "content_color": content_color,
        "tail_dir": tail_dir,
        "mouth_dir": mouth_dir,
        "relation": relation,
        "body_bbox": (r0 + br0, c0 + bc0, r0 + br1, c0 + bc1),
        "combo_bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < 2 * num_cups:
        return None

    border_colors = palette[:num_cups]
    content_colors = palette[num_cups:2 * num_cups]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    cups_meta: list[dict] = []

    for border_color, content_color in zip(border_colors, content_colors):
        meta = _add_cup(
            rng,
            H,
            W,
            int(border_color),
            int(content_color),
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if not _scatter_content(rng, canvas_in, cups_meta):
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
