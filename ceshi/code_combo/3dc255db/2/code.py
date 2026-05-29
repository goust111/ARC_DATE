# Combo variant of 3dc255db (index 2)
# combo_spec: D1/1+D5/1
# Name: 反弹弹射+渐变色轨迹

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[4]
SOLVER_DIR = ROOT / "solvers"
if str(SOLVER_DIR) not in sys.path:
    sys.path.append(str(SOLVER_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


# Design analysis for combo D1/1 + D5/1
# ======================================
# 1. Concretization and behavior trace
#    - Base 3dc255db:
#      detect each hollow cup, erase its interior content, then fire a straight monochrome ray
#      outward from the unique tail spike. The ray stops at the first grid exit.
#    - D1/1 changes only geometry:
#      the ray reflects off the grid boundary and keeps moving. In this task the launch direction
#      is always cardinal, so the reflected path stays on one row or one column and may pass back
#      through the spike and into the cup mouth corridor.
#    - D5/1 changes only coloring:
#      the trajectory is no longer a single flat content color. It now follows an ordered palette
#      from content color toward border color. Near the spike it stays content-colored; after the
#      first bounce the return segment begins with a fixed intermediate color and then tapers to the
#      cup border color at the far end of the visible reflected segment.
#    - Concrete trace:
#      tail = U, spike at row 1, content_count = 5, content color = 2, border color = 7.
#      Reflected path cells are rows [0] | [1,2,3,4], where "|" marks the bounce.
#      Base output: rows [0] only, all color 2.
#      D1-only output: rows [0,1,2,3,4], all color 2.
#      Combo output here: row 0 = 2, rows 1..2 = 5, rows 3..4 = 7.
#      The fused rule is visible in two ways at once:
#        (a) reflected continuation exists on and beyond the spike row;
#        (b) the reflected continuation is not monochrome.
#
# 2. Complexity calibration
#    - The original task bounds are tight: grid size is 10..13 and content count is at most 7.
#      Under those bounds a cardinal reflected prefix can normally show at most one visible bounce,
#      because after the first reflection the opposite boundary is too far away to hit again.
#    - Therefore the color rule is instantiated as a 3-stage palette:
#        content -> fixed middle color 5 -> border
#      rather than an arbitrarily deep list of intermediate colors. This keeps the rule visible and
#      inferable from few examples while still matching the combo-level intent of "advance along the
#      palette after bouncing".
#    - The generator forces short launch distance to the first boundary and enough post-bounce cells
#      so every accepted sample differs from both the original task and the pure D1/1 variant.
#
# 3. Implicit assumption audit for main()
#    - Cup detection assumptions remain valid:
#      spike detection still relies on "exactly one border pixel on one bbox edge", and mouth
#      detection still floods from the opposite bbox side.
#    - The base solver assumed the ejection prefix is a simple non-revisiting run that ends at the
#      first exit. D1/1 breaks that; the solver must trace a step sequence with explicit reflection.
#    - The base solver also assumed every written trajectory cell uses the same content color. D5/1
#      breaks that; rendering must be separated from tracing so that color is assigned segment-wise.
#    - New combo assumption:
#      the palette needs a distinct middle color. The generator guarantees both border and content
#      avoid color 5, so the intended middle color is always visually distinct in generated samples.
#      For robustness on arbitrary inputs the solver falls back to another non-endpoint color if 5 is
#      unavailable.
#
# 4. Placement legality in the generator
#    - Cups still keep a 1-cell bbox gap.
#    - Reflected prefixes may traverse the source cup's own bbox after bouncing, but they may not
#      touch other cups or other paths outside that bbox.
#    - Accepted placements require:
#        direct_run_to_boundary in [1, 2]
#        at least 2 post-bounce cells
#      so the reflected geometry is visible and the gradient reaches the border color.
#
# 5. Visual observability and neatness
#    - Every generated cup shows both combo signals:
#        reflected continuation beyond the boundary hit,
#        a non-monochrome return segment.
#    - The middle color is global and repeated, so the human-visible cue is not tied to hidden color
#      indices or per-object arbitrary mappings.
#    - Paths remain 1-pixel orthogonal traces, and generator-side collision checks keep different cups
#      and path prefixes visually separated.
#
# 6. Quantitative behavior metric used after implementation
#    - A sample is considered combo-visible when:
#        (a) the combo output differs from the D1-only monochrome reflected output, and
#        (b) some reflected trajectory cell is recolored to the cup border color.
#      This detects both the reflection part and the gradient part, preventing silent regression to
#      the original task or to pure D1/1.


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_COUNT_MIN = 4
VISIBLE_DIRECT_RUN_MAX = 2
MIN_RETURN_CELLS = 2

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
MID_COLOR = 5
MID_FALLBACK_ORDER = [5, 8, 4, 7, 6, 3, 2, 1, 9]
NON_MID_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside ``bbox`` from the chosen bbox edge, stopping at border cells."""
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


def _choose_mid_color(content_col: int, border_col: int) -> int:
    for col in MID_FALLBACK_ORDER:
        if int(col) not in {int(content_col), int(border_col), 0}:
            return int(col)
    return int(content_col)


def _bounced_path_segments(
    shape: tuple[int, int], start: tuple[int, int], direction: str, steps: int
) -> list[list[tuple[int, int]]]:
    """Trace a reflected cardinal ray and return its cells grouped by bounce-separated segments."""
    H, W = shape
    r, c = (int(start[0]), int(start[1]))
    dr, dc = DIR4[direction]
    segments: list[list[tuple[int, int]]] = [[]]
    for _ in range(int(steps)):
        nr, nc = r + dr, c + dc
        if not (0 <= nr < H and 0 <= nc < W):
            dr, dc = -dr, -dc
            nr, nc = r + dr, c + dc
            if segments[-1]:
                segments.append([])
        r, c = int(nr), int(nc)
        segments[-1].append((r, c))
    return [seg for seg in segments if seg]


def _segment_ramp(start_col: int, end_col: int, length: int) -> list[int]:
    if length <= 0:
        return []
    if start_col == end_col:
        return [int(start_col)] * int(length)
    if length == 1:
        return [int(end_col)]
    colors = [int(start_col), int(end_col)]
    out: list[int] = []
    for i in range(int(length)):
        idx = (i * (len(colors) - 1)) // (length - 1)
        out.append(int(colors[idx]))
    return out


def _trajectory_color_steps(segments: list[list[tuple[int, int]]], content_col: int, border_col: int) -> list[int]:
    """Assign colors to the traced path.

    Segment 0 stays content-colored.
    Each bounced segment advances one palette stage.
    The final segment tapers from the current stage to the border color.
    """
    if not segments:
        return []
    mid_col = _choose_mid_color(content_col, border_col)
    stage_nodes = [int(content_col)]
    if len(segments) >= 2:
        stage_nodes.append(int(mid_col))
    if len(segments) >= 3:
        used = {int(content_col), int(border_col), int(mid_col), 0}
        extra = [col for col in MID_FALLBACK_ORDER if col not in used]
        need = len(segments) - 2
        stage_nodes.extend(int(extra[min(i, len(extra) - 1)]) for i in range(need))

    colors: list[int] = []
    for seg_idx, seg in enumerate(segments):
        if seg_idx == 0:
            colors.extend([int(content_col)] * len(seg))
            continue
        stage_col = int(stage_nodes[min(seg_idx, len(stage_nodes) - 1)])
        is_last = seg_idx == len(segments) - 1
        if is_last:
            colors.extend(_segment_ramp(stage_col, int(border_col), len(seg)))
        else:
            colors.extend([stage_col] * len(seg))
    return colors


def _paint_trajectory(
    out: np.ndarray, segments: list[list[tuple[int, int]]], content_col: int, border_col: int
) -> list[tuple[int, int]]:
    colors = _trajectory_color_steps(segments, content_col, border_col)
    seq = [cell for seg in segments for cell in seg]
    for (r, c), col in zip(seq, colors):
        out[int(r), int(c)] = int(col)
    return seq


def _direct_run_to_edge(H: int, W: int, spike: tuple[int, int], direction: str) -> int:
    sr, sc = spike
    if direction == "U":
        return int(sr)
    if direction == "D":
        return int(H - 1 - sr)
    if direction == "L":
        return int(sc)
    return int(W - 1 - sc)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1 : L + hL + 1, 0] = color
    if hR:
        g[R + 1 : R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
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
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for (r, c) in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for (r, c) in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for (r, c) in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    L_arm = int(rng.integers(lo, hi + 1))
    R_arm = int(rng.integers(lo, hi + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    wlo, whi = WALL_EXT_RANGE
    hL = int(rng.integers(wlo, whi + 1))
    hR = int(rng.integers(wlo, whi + 1))
    if hL + hR == 0:
        hL = 1
    while max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
        if L_arm + hL >= R_arm + hR and hL > 0:
            hL -= 1
        elif hR > 0:
            hR -= 1
        else:
            break
    if hL + hR == 0:
        return None
    return L_arm, R_arm, hL, hR


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    max_content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, int, list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)
    occ_block = ndimage.binary_dilation(occ_cup | occ_path, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        direct_run = _direct_run_to_edge(H, W, (sr_abs, sc_abs), direction)
        if direct_run < 1 or direct_run > VISIBLE_DIRECT_RUN_MAX:
            continue

        min_visible_count = max(CONTENT_COUNT_MIN, direct_run + MIN_RETURN_CELLS)
        if min_visible_count > max_content_count:
            continue

        segments_full = _bounced_path_segments((H, W), (sr_abs, sc_abs), direction, max_content_count)
        seq_full = [cell for seg in segments_full for cell in seg]
        safe_len = 0
        for r, c in seq_full:
            inside_own_bbox = r0 <= r < r0 + ph and c0 <= c < c0 + pw
            if not inside_own_bbox and occ_block[r, c]:
                break
            safe_len += 1
        if safe_len < min_visible_count:
            continue

        content_count = int(rng.integers(min_visible_count, min(max_content_count, safe_len) + 1))
        segments = _bounced_path_segments((H, W), (sr_abs, sc_abs), direction, content_count)
        if len(segments) < 2 or len(segments[-1]) < MIN_RETURN_CELLS:
            continue
        seq = [cell for seg in segments for cell in seg]
        return r0, c0, content_count, segments, sorted(set(seq))
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0 : r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0 : r1 + 1, c1 + 1 :] = True
    elif tail_dir == "D":
        mask[:r0, c0 : c1 + 1] = True
    else:
        mask[r1 + 1 :, c0 : c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[int(r), int(c)] = int(stray_color)
    canvas_out[int(r), int(c)] = int(stray_color)


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
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < CONTENT_COUNT_MIN:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    max_content_count = min(CONTENT_COUNT_MAX, len(mouth))

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, max_content_count, occ_cup, occ_path
    )
    if placement is None:
        return None
    r0, c0, content_count, segments, path_cells = placement

    patch_mask = patch != 0
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
    _paint_trajectory(canvas_out, segments, content_color, border_color)
    for r, c in path_cells:
        occ_path[int(r), int(c)] = True

    return {
        "border_color": int(border_color),
        "content_color": int(content_color),
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": int(content_count),
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
            canvas_in[int(r), int(c)] = int(meta["content_color"])
    return True


def _solve_grid(grid: np.ndarray, *, gradient: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 5:
                continue

            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            bbox = (r0, c0, r1, c1)
            border_set = {(int(r), int(c)) for r, c in border_coords}
            corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

            edges = (
                ("U", [p for p in border_set if p[0] == r0]),
                ("D", [p for p in border_set if p[0] == r1]),
                ("L", [p for p in border_set if p[1] == c0]),
                ("R", [p for p in border_set if p[1] == c1]),
            )
            candidates = [
                (d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[d])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[int(r), int(c)] = bg

            segments = _bounced_path_segments((H, W), spike, tail_dir, len(content_cells))
            if gradient:
                _paint_trajectory(out, segments, content_col, int(border_col))
            else:
                for rr, cc in [cell for seg in segments for cell in seg]:
                    out[int(rr), int(cc)] = int(content_col)

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Combo solver for reflected cup ejection with a content->mid->border trajectory."""
    return _solve_grid(grid, gradient=True)


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(NON_MID_PALETTE)
    rng.shuffle(palette)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for _ in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
        return None

    # Reject silent collapse to the monochrome D1-only variant.
    d1_only = _solve_grid(canvas_in, gradient=False)
    if np.array_equal(d1_only, canvas_out):
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
