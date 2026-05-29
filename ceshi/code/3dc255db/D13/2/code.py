# Variant of 3dc255db
# Dimension: D13 — 弹射线宽度 (trajectory)
# Change: 单像素宽度的射线(1px宽) → 锥形弹射
#
# Design analysis
# ===============
# spec:
#   original_objects:
#     grid: 10-13 rows x 10-13 cols, bg=0
#     cups: 1-3 hollow cup/Y/funnel frames, 8-connected, unique border color per cup
#     content: one color per cup, scattered in the mouth interior
#     original_output: erase the content and draw a 1px ray from the tail spike
#   variant_D13:
#     core_change:
#       The erased content count now controls cone depth instead of ray length.
#       For depth t, layer t sits t cells away from the spike in the tail direction and
#       expands symmetrically on the perpendicular axis. The discrete layer widths are
#       1, 3, 5, ... (= 2*t-1), producing a triangular cone.
#     concrete_trace:
#       Example with tail='U' and 3 content pixels:
#       - original: fill (sr-1,sc), (sr-2,sc), (sr-3,sc)
#       - variant:
#         t=1 -> {(sr-1, sc)}
#         t=2 -> {(sr-2, sc-1), (sr-2, sc), (sr-2, sc+1)}
#         t=3 -> {(sr-3, sc-2), ..., (sr-3, sc+2)}
#       The cup interior is still erased first; only the outbound drawing changes.
#     complexity_calibration:
#       To keep the widening visible without making placement impossible, generator content
#       counts are limited to 2-4 and every placed cup must show at least two visible cone
#       layers with a farthest visible width >= 3. This avoids degenerate samples that look
#       like the original 1px ray.
#     implicit_assumption_audit:
#       - Candidate-tail detection via "exactly one border cell on a bbox edge" still holds
#         because cup borders are unchanged.
#       - Mouth detection via flood from the opposite bbox side still holds because the mouth
#         remains the only open side and content is still entirely inside that region.
#       - The original straight-line draw loop assumed "one content pixel -> one emitted cell".
#         That assumption is no longer valid, so the draw stage must iterate over cone layers.
#       - Unconditional painting into `out` is still safe only if the outbound cone region is
#         clear of other objects. The generator now reserves the whole cone footprint.
#     placement_legality:
#       Generator remains output-first. For each cup it chooses a cone depth, computes the full
#       visible cone cells for the candidate placement, and rejects placements where the cup bbox
#       or cone would overlap existing cups / cone footprints. Stray pixels are forbidden from
#       landing inside those reserved cone cells.
#     visual_observability:
#       Human-solvers can see that each additional content pixel adds a wider, farther layer.
#       Because every generated cup has visible width growth, the change is inferable from 2-3
#       examples and does not collapse back to a thin line.
#     visual_cleanliness:
#       No new markers or colors are introduced. The only added structure is a single-color
#       triangular splash in otherwise empty space, so cups and emitted areas remain separable.
#     color_reasoning:
#       Colors behave exactly as in the original task: border color is unchanged, content color
#       is conserved, and there is no hidden color-dependent branching.
#     solver_assumptions_after_change:
#       - Each acted-on cup still has exactly one unambiguous tail candidate that reaches content.
#       - The outbound cone region intended for that cup is background in the input.
#       - Content count remains the only driver of cone depth.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
PERP4 = {"U": (0, 1), "D": (0, 1), "L": (1, 0), "R": (1, 0)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on the given bbox edge."""
    r0, c0, r1, c1 = bbox
    if side == "U":
        seeds = [(r0, c) for c in range(c0, c1 + 1) if (r0, c) not in border_set]
    elif side == "D":
        seeds = [(r1, c) for c in range(c0, c1 + 1) if (r1, c) not in border_set]
    elif side == "L":
        seeds = [(r, c0) for r in range(r0, r1 + 1) if (r, c0) not in border_set]
    else:  # "R"
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


def _cone_layers_from_spike(
    spike: tuple[int, int], direction: str, depth: int, shape: tuple[int, int],
) -> list[list[tuple[int, int]]]:
    """Cone layers at increasing distance from the spike.

    Layer t is centered t steps away in `direction` and has width 2*t-1 on the
    perpendicular axis. Each layer is clipped to grid bounds.
    """
    H, W = shape
    sr, sc = spike
    dr, dc = DIR4[direction]
    pr, pc = PERP4[direction]
    layers: list[list[tuple[int, int]]] = []
    for t in range(1, depth + 1):
        cr = sr + dr * t
        cc = sc + dc * t
        if not (0 <= cr < H and 0 <= cc < W):
            break
        layer = []
        for offset in range(-(t - 1), t):
            rr = cr + pr * offset
            rc = cc + pc * offset
            if 0 <= rr < H and 0 <= rc < W:
                layer.append((rr, rc))
        if layer:
            layers.append(layer)
    return layers


def _flatten_layers(layers: list[list[tuple[int, int]]]) -> list[tuple[int, int]]:
    return [cell for layer in layers for cell in layer]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: each cup ejects its content as a widening cone instead of a 1px ray.

    The cup-detection logic is unchanged from the original task:
    - find a unique tail spike on one bbox edge
    - flood from the opposite side to identify the mouth and its content color
    - erase that content from the mouth interior

    Changed step:
    - if the cup contains N content pixels, draw N cone layers outward from the spike
      in the tail direction, with layer widths 1, 3, 5, ... .
    """
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
                (direction, cells[0]) for direction, cells in edges
                if len(cells) == 1 and cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for direction, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[direction])
                cnt = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != bg and value != int(border_col):
                        cnt[value] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((direction, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            for layer in _cone_layers_from_spike(spike, tail_dir, len(content_cells), (H, W)):
                for r, c in layer:
                    out[r, c] = content_col

    return out


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MIN = 2
CONTENT_COUNT_MAX = 4
MIN_VISIBLE_CONE_DEPTH = 2
MIN_VISIBLE_CONE_WIDTH = 3

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400
ADD_CUP_ATTEMPTS = 40
# ==================================


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP (spike at top, row 0)."""
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
    """Flood-fill the interior of the cup patch from the mouth side."""
    h, w = patch.shape
    border_mask = patch == border_color
    if opposite_side == "U":
        seeds = [(0, c) for c in range(w) if not border_mask[0, c]]
    elif opposite_side == "D":
        seeds = [(h - 1, c) for c in range(w) if not border_mask[h - 1, c]]
    elif opposite_side == "L":
        seeds = [(r, 0) for r in range(h) if not border_mask[r, 0]]
    else:  # "R"
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
    """Rotate a canonical (tail=U) patch, its spike and mouth coords to the target direction."""
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
    """Pick a canonical cup shape that respects the original bbox bounds after rotation."""
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
    rng: np.random.Generator, H: int, W: int, patch: np.ndarray, spike: tuple[int, int],
    direction: str, content_count: int, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[list[tuple[int, int]]], list[tuple[int, int]]] | None:
    """Find a placement where the cup bbox and its full cone footprint are conflict-free."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        layers = _cone_layers_from_spike((sr_abs, sc_abs), direction, content_count, (H, W))
        if len(layers) < MIN_VISIBLE_CONE_DEPTH:
            continue
        if max(len(layer) for layer in layers) < MIN_VISIBLE_CONE_WIDTH:
            continue

        cone_cells = _flatten_layers(layers)
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in cone_cells):
            continue
        return r0, c0, layers, cone_cells
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of `bbox`, restricted to its perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict],
) -> None:
    """Optionally place an isolated stray pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in cup["cone_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator, H: int, W: int, border_color: int, content_color: int,
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup (shape + placement + cone footprint) and commit it."""
    for _ in range(ADD_CUP_ATTEMPTS):
        direction = str(rng.choice(["U", "D", "L", "R"]))
        shape = _sample_cup_shape(rng, direction)
        if shape is None:
            continue
        L_arm, R_arm, hL, hR = shape

        patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
        mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
        if not mouth_canon:
            continue

        patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
        ph, pw = patch.shape

        max_count = min(CONTENT_COUNT_MAX, len(mouth))
        if max_count < CONTENT_COUNT_MIN:
            continue
        content_count = int(rng.integers(CONTENT_COUNT_MIN, max_count + 1))

        placement = _try_place_cup(
            rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path,
        )
        if placement is None:
            continue
        r0, c0, layers, cone_cells = placement

        patch_mask = patch != 0
        canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
        canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
        occ_cup[r0:r0 + ph, c0:c0 + pw] = True
        for r, c in cone_cells:
            canvas_out[r, c] = content_color
            occ_path[r, c] = True

        return {
            "border_color": border_color,
            "content_color": content_color,
            "direction": direction,
            "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
            "content_count": content_count,
            "cone_layers": layers,
            "cone_cells": cone_cells,
        }
    return None


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content pixels at random inside its mouth interior."""
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
    """Single end-to-end generation attempt with an output-first cone construction."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
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
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path,
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
