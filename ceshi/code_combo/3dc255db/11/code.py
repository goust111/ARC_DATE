# Combo variant of 3dc255db (index 11)
# combo_spec: D3/1+D10/1
# Name: 边缘控制像素+边框数量决定长度

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


# -----------------------------------------------------------------------------
# Design analysis for combo D3/1 + D10/1
#
# a) Concrete behavior trace
#    - Original task: a cup contains k interior pixels of one content color. Those pixels both
#      identify the real mouth/tail pairing and determine the ejection length k.
#    - Fused task here:
#        1. The cup interior is empty in both input and output.
#        2. A single control pixel is placed on the outer grid edge on the mouth side of the cup.
#           Its color is the ejected color.
#        3. The ejection length is no longer tied to source-count. It equals the number of border
#           pixels in that cup component. So color comes from the edge; length comes from the cup.
#      Example: if a tail-up cup has 9 border pixels and one orange control on the bottom edge
#      inside the cup's column span, output removes that orange control and draws 9 orange cells
#      upward from the spike, truncated at the grid edge if necessary.
#    - Difference from D3/1 alone: control pixels do not encode quantity anymore. Using exactly one
#      control pixel per cup makes this visible; otherwise the task could visually regress toward
#      "count the edge dots".
#
# b) Complexity calibration
#    - Each cup has exactly one control pixel. This keeps the color signal simple and prevents a
#      misleading secondary cue where the number of controls also suggests length.
#    - Cups keep varied Y/funnel geometry from the base task, so border counts vary naturally.
#    - Content colors are unique across cups to keep matching legible in multi-cup scenes.
#    - Cups that read controls from the same outer side are forced to have disjoint projected spans,
#      so one border control cannot satisfy two cups.
#
# c) Old-main assumption audit
#    - Old assumption: a flood from the mouth side reaches interior content. Broken; interiors are
#      empty now, so the mouth flood no longer identifies the correct candidate.
#    - Old assumption: source cells to erase live inside the cup. Broken; the consumed source cell
#      now sits on the global border.
#    - Old assumption: source-count equals ejection length. Broken; the new length is
#      `len(border_component)`.
#    - Preserved assumption: the tail spike is still detected as the unique non-corner singleton on
#      one bbox edge. Once the solver knows the correct candidate, drawing the outward ray is the
#      same as before.
#
# d) Placement legality guaranteed by the generator
#    - Cups are placed at least one cell away from the outer border.
#    - Cup bboxes keep a one-cell gap using dilated occupancy checks.
#    - Full spike-to-edge path stays clear of other cups and other paths.
#    - Cups that would read from the same edge have disjoint control intervals.
#    - Control pixels are placed only on cells not used by any path.
#
# e) Visual observability
#    - In input, the cup mouths are empty; the only cue is one colored border dot per cup.
#    - In output, that dot disappears and a much longer same-colored ray appears.
#    - Across examples, the long ray length tracks cup size rather than number of controls, which is
#      visibly fixed at one.
#
# f) Cleanliness
#    - Border controls live only on the global border band.
#    - Optional stray pixels stay off the border so every non-cup border pixel remains a true
#      control.
#    - The cup interior remains uncluttered, making the external-information fusion easy to read.
#
# g) New solver assumptions and how generation satisfies them
#    - Solver assumes each valid cup has exactly one control pixel in its claimed edge segment.
#      Generator enforces one control per cup and disjoint same-side spans.
#    - Solver assumes cup components are the only large 8-connected objects of their color.
#      Generator makes controls and strays isolated single pixels.
#    - Solver assumes the line color can be taken directly from the control pixel.
#      Generator keeps control color different from its own border color and unique across cups.
#    - Solver would visually regress toward plain D3/1 if every realized ray were length 1.
#      Generator rejects those fully degenerate samples and keeps at least one cup with a visible
#      multi-cell ejection.
# -----------------------------------------------------------------------------


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _edge_segment_for_bbox(H: int, W: int, bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    """Grid-border cells aligned with the bbox projection on the chosen outer side."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        return [(0, c) for c in range(c0, c1 + 1)]
    if side == "D":
        return [(H - 1, c) for c in range(c0, c1 + 1)]
    if side == "L":
        return [(r, 0) for r in range(r0, r1 + 1)]
    return [(r, W - 1) for r in range(r0, r1 + 1)]


def _looks_like_cup(coords: np.ndarray) -> bool:
    """Keep cup-shaped border components, reject isolated controls/strays."""
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
            border_set = {(int(r), int(c)) for r, c in border_coords}
            cups.append(
                {
                    "border_color": int(border_col),
                    "bbox": (r0, c0, r1, c1),
                    "border_set": border_set,
                    "border_count": len(border_set),
                }
            )
    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Each cup is empty.
    - One control pixel on the outer edge provides the ejected color.
    - The ejection length equals the number of pixels in that cup's border component.
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
        bbox = cup["bbox"]
        border_set = cup["border_set"]
        border_count = cup["border_count"]
        r0, c0, r1, c1 = bbox
        corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}

        edges = (
            ("U", [p for p in border_set if p[0] == r0]),
            ("D", [p for p in border_set if p[0] == r1]),
            ("L", [p for p in border_set if p[1] == c0]),
            ("R", [p for p in border_set if p[1] == c1]),
        )
        candidates = [
            (tail_dir, cells[0]) for tail_dir, cells in edges
            if len(cells) == 1 and cells[0] not in corners
        ]
        if not candidates:
            continue

        passing = []
        for tail_dir, spike in candidates:
            ctrl_side = OPPOSITE[tail_dir]
            ctrl_segment = _edge_segment_for_bbox(H, W, bbox, ctrl_side)
            ctrl_cells = [p for p in ctrl_segment if p in edge_controls]
            if len(ctrl_cells) != 1:
                continue
            ctrl_cell = ctrl_cells[0]
            passing.append((tail_dir, spike, int(edge_controls[ctrl_cell]), ctrl_cell))

        if len(passing) != 1:
            continue
        tail_dir, spike, content_col, ctrl_cell = passing[0]

        out[ctrl_cell[0], ctrl_cell[1]] = bg

        dr, dc = DIR4[tail_dir]
        sr, sc = spike
        for t in range(1, border_count + 1):
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

STRAY_PROB = 0.30
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500
# ==================================


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical tail-up hollow Y/funnel patch."""
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
    """Place a cup away from the border with a clear full spike-to-edge path."""
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
    """Optional interior clutter that must not be mistaken for a control."""
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
    border_count = int(np.count_nonzero(patch_mask))
    for r, c in path_cells[:border_count]:
        canvas_out[r, c] = content_color

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "path_cells": path_cells,
        "control_side": OPPOSITE[direction],
        "border_count": border_count,
    }


def _place_controls(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    cups_meta: list[dict],
) -> bool:
    """Place exactly one border control pixel per cup; it encodes color only."""
    H, W = canvas_in.shape
    blocked = np.zeros((H, W), dtype=bool)
    for meta in cups_meta:
        for r, c in meta["path_cells"]:
            blocked[r, c] = True

    for meta in cups_meta:
        candidates = [
            (r, c)
            for (r, c) in _edge_segment_for_bbox(H, W, meta["bbox"], meta["control_side"])
            if not blocked[r, c] and canvas_in[r, c] == 0
        ]
        if not candidates:
            return False
        r, c = candidates[int(rng.integers(0, len(candidates)))]
        canvas_in[r, c] = meta["content_color"]
        meta["control_cell"] = (int(r), int(c))
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

    if not any(min(meta["border_count"], len(meta["path_cells"])) >= 2 for meta in cups_meta):
        return None

    if not _place_controls(rng, canvas_in, cups_meta):
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
