# Variant of 3dc255db
# Dimension: D3 — 源对象类型 (src)
# Change: 杯口内部散布的单色内容像素(1-7个)... → 网格边缘控制像素作为源

from __future__ import annotations

import numpy as np
from scipy import ndimage
from pathlib import Path
import sys

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


# -----------------------------------------------------------------------------
# Design analysis for the D3 variant
#
# variant_spec:
#   core_change:
#     old_source:
#       content pixels (1-7 cells, one color) sit inside each cup mouth.
#       They serve two roles simultaneously:
#       1. identify which candidate tail is real, because only the correct mouth flood reaches them
#       2. provide the ejected color and length
#     new_source:
#       the cup interior is empty in both input and output.
#       Instead, 1-7 control pixels of one color are placed on the outer grid edge.
#       Their color is the cup's content color; their count is the ejection length before overflow.
#       They are consumed: output removes these border controls and draws the same-colored line
#       from the matched cup spike along the tail direction.
#   concrete_geometry:
#     - cups keep the original hollow Y/funnel frame family and unique border colors.
#     - for a cup with tail direction d, its control pixels are placed on the opposite grid edge
#       OPPOSITE[d], inside the projection of the cup bbox onto that edge:
#         tail U -> controls on bottom row, columns c0..c1
#         tail D -> controls on top row, columns c0..c1
#         tail L -> controls on right col, rows r0..r1
#         tail R -> controls on left col, rows r0..r1
#     - this makes the matching visible: the edge controls sit "in front of" the cup mouth.
#   behavior_trace:
#     example:
#       - original: a tail-up cup contains three orange pixels in its mouth.
#         solver floods from bbox bottom, sees orange, picks tail=U, erases those three cells,
#         then writes three orange cells upward from the spike.
#       - variant: the same cup interior is empty. Three orange control pixels now sit on the
#         bottom border within the cup's column span. Solver tests candidate tail=U, inspects the
#         opposite full-grid side (bottom edge) over that same span, finds exactly those three
#         orange pixels, removes them, and ejects three orange cells upward from the spike.
#       - difference: the disambiguation signal moved from inside the mouth to the global border.
#   complexity_calibration:
#     - controls remain low-complexity single pixels, mirroring source task 4a21e3da.
#     - matching is not purely by color: location relative to the cup mouth side matters.
#     - to avoid hidden ambiguity, the generator keeps cups off the border and forbids overlap
#       between cups that would claim the same control edge segment.
#     - content colors are made unique across cups. This is an allowed compatibility tweak:
#       once the source leaves the cup interior, unique colors keep the edge signal legible.
#   implicit_assumption_audit_for_old_main:
#     - old code built `mouth = _flood_from_bbox_side(..., OPPOSITE[d])` and then assumed:
#         any non-bg/non-border cell inside that flood is source content.
#       This no longer holds because the mouth is empty.
#     - old `Counter()` over mouth cells assumed the correct candidate tail is the one whose
#       flood reaches content. In the variant, the flood reaches nothing for every candidate.
#     - old erasure loop assumed the source cells lie inside the cup and should be painted bg.
#       In the variant, the cells to erase live on the global border instead.
#     - line drawing itself remains valid: once `(tail_dir, spike, content_color, count)` are
#       recovered from controls, the ejection stage is unchanged.
#   legality_constraints_for_generation:
#     - cups are kept at least one cell away from all four grid borders so border controls never
#       overlap cup borders.
#     - for cups whose controls would live on the same outer side, their projected intervals on
#       that side are disjoint; otherwise a single border control could belong to multiple cups.
#     - stray pixels, if added, never touch the outer border, so every non-cup border pixel is a
#       true control pixel.
#     - border controls are chosen only from edge cells not occupied by any ejection path cell of
#       another cup, preventing "source pixel also lies on someone else's output ray" clutter.
#   observability_check:
#     - across 2-4 examples, a human sees empty cups in the input, colored dots on the border,
#       and those border dots disappear while same-colored rays appear from nearby cup spikes.
#     - the relation is visible using only position/count/color; no ordering by numeric color id.
#   cleanliness:
#     - no object overlaps the border-control band.
#     - control pixels are confined to one border segment per cup, with 0 stray border pixels.
#     - the cup area stays uncluttered because the mouth is empty.
# -----------------------------------------------------------------------------


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
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


def _edge_segment_for_bbox(H: int, W: int, bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    """Grid-border cells aligned with the bbox projection on the given outer side."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return [(0, c) for c in range(c0, c1 + 1)]
    if side == "D":
        return [(H - 1, c) for c in range(c0, c1 + 1)]
    if side == "L":
        return [(r, 0) for r in range(r0, r1 + 1)]
    return [(r, W - 1) for r in range(r0, r1 + 1)]


def _looks_like_cup(coords: np.ndarray) -> bool:
    """Filter out edge-control runs / stray pixels while keeping cup-shaped border components."""
    if coords.size == 0 or int(coords.shape[0]) < 5:
        return False
    r0, c0, r1, c1 = (int(v) for v in bbox_coords(coords))
    return (r1 - r0 + 1) >= 3 and (c1 - c0 + 1) >= 3


def _find_cups(g: np.ndarray, bg: int) -> list[dict]:
    cups: list[dict] = []
    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if not _looks_like_cup(border_coords):
                continue
            r0, c0, r1, c1 = (int(v) for v in bbox_coords(border_coords))
            cups.append(
                {
                    "border_color": int(border_col),
                    "coords": border_coords,
                    "bbox": (r0, c0, r1, c1),
                    "border_set": {(int(r), int(c)) for r, c in border_coords},
                }
            )
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Cups stay empty.
    - Source content becomes same-colored control pixels on the outer grid edge, on the mouth side
      of the matched cup and within that cup's projected span on the edge.
    - Remove those controls and eject a straight line from the cup spike in the tail direction.
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

    edge_controls: dict[tuple[int, int], int] = {}
    border_coords = set()
    for c in range(W):
        border_coords.add((0, c))
        border_coords.add((H - 1, c))
    for r in range(H):
        border_coords.add((r, 0))
        border_coords.add((r, W - 1))
    for r, c in border_coords:
        if cup_mask[r, c]:
            continue
        v = int(g[r, c])
        if v != bg:
            edge_controls[(r, c)] = v

    for cup in cups:
        border_col = cup["border_color"]
        bbox = cup["bbox"]
        border_set = cup["border_set"]
        r0, c0, r1, c1 = bbox
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

        edges = (
            ("U", [p for p in border_set if p[0] == r0]),
            ("D", [p for p in border_set if p[0] == r1]),
            ("L", [p for p in border_set if p[1] == c0]),
            ("R", [p for p in border_set if p[1] == c1]),
        )
        candidates = [
            (d, cells[0]) for d, cells in edges
            if len(cells) == 1 and cells[0] not in corners
        ]
        if not candidates:
            continue

        passing = []
        for tail_dir, spike in candidates:
            ctrl_side = OPPOSITE[tail_dir]
            ctrl_segment = _edge_segment_for_bbox(H, W, bbox, ctrl_side)
            ctrl_cells = [p for p in ctrl_segment if p in edge_controls]
            if not ctrl_cells:
                continue
            ctrl_colors = {edge_controls[p] for p in ctrl_cells}
            if len(ctrl_colors) != 1:
                continue
            passing.append((tail_dir, spike, int(next(iter(ctrl_colors))), ctrl_cells))

        if len(passing) != 1:
            continue
        tail_dir, spike, content_col, ctrl_cells = passing[0]

        for r, c in ctrl_cells:
            out[r, c] = bg

        dr, dc = DIR4[tail_dir]
        sr, sc = spike
        for t in range(1, len(ctrl_cells) + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = content_col

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.35
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400
# ==================================


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical tail-up hollow funnel/Y patch."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1:L + hL + 1, 0] = color
    if hR:
        g[R + 1:R + hR + 1, w - 1] = color
    return g, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the mouth side for validation."""
    h, w = patch.shape
    border_mask = (patch == border_color)
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
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical tail-up geometry to the requested tail direction."""
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


def _intervals_overlap(a: tuple[int, int], b: tuple[int, int]) -> bool:
    return not (a[1] < b[0] or b[1] < a[0])


def _control_interval_for_placement(r0: int, c0: int, ph: int, pw: int, ctrl_side: str) -> tuple[int, int]:
    if ctrl_side in ("U", "D"):
        return (c0, c0 + pw - 1)
    return (r0, r0 + ph - 1)


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    edge_claims: dict[str, list[tuple[int, int]]],
) -> tuple[int, int, list[tuple[int, int]], tuple[int, int]] | None:
    """Place a cup away from the outer border with clear ejection path and unique control span."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    ctrl_side = OPPOSITE[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    row_range = range(1, H - ph) if H - ph >= 1 else range(0)
    col_range = range(1, W - pw) if W - pw >= 1 else range(0)
    coords_list = [(r0, c0) for r0 in row_range for c0 in col_range]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        ctrl_interval = _control_interval_for_placement(r0, c0, ph, pw, ctrl_side)
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
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells, ctrl_interval
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Mask on the mouth side of a cup, excluding the global border control band."""
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
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
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
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
    edge_claims: dict[str, list[tuple[int, int]]],
) -> dict | None:
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch, spike, _mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, occ_cup, occ_path, edge_claims,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ctrl_interval = placement

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in path_cells:
        occ_path[r, c] = True

    edge_claims[OPPOSITE[direction]].append(ctrl_interval)
    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
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
            for (r, c) in _edge_segment_for_bbox(H, W, meta["bbox"], meta["control_side"])
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
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < 2 * num_cups:
        return None

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    edge_claims = {"U": [], "D": [], "L": [], "R": []}

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
            occ_cup,
            occ_path,
            edge_claims,
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
