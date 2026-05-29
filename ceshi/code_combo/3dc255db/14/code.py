# Combo variant of 3dc255db (index 14)
# combo_spec: D4/1+D5/1
# Name: 目标标记停止+渐变色轨迹
#
# Design analysis
# ===============
# Fusion summary:
# - Keep the original cup/frame detection, mouth flood, and spike-direction logic.
# - Apply D4/1 by placing exactly one same-content-color marker on the outward spike ray; the
#   ejection stops at that marker instead of at the count-based or boundary-based endpoint.
# - Apply D5/1 by recoloring the whole spike->marker segment as a discrete gradient from the
#   mouth content color to the cup border color. The marker cell itself is therefore recolored
#   from content-color in the input to border-color in the output.
# - The two changes are coupled rather than independent: the marker distance now determines both
#   the geometric endpoint and the number of gradient steps.
#
# a) Concrete behavior and trace:
# - Fixed palette convention for this fused task:
#   `GRADIENT_CHAIN = [1, 8, 3, 4, 7]` (blue -> teal -> green -> yellow -> orange).
#   Every cup chooses its content and border colors from this visible chain, with content earlier
#   than border. The output trajectory uses the contiguous subchain between them, stretched across
#   the full spike->marker distance.
# - Example:
#   * Cup A has content color 1, border color 7, three scattered content pixels in the mouth,
#     and a same-color marker six cells ahead of the spike.
#   * Original task: erase the three mouth pixels and paint three cells of color 1.
#   * D4-only: erase the mouth pixels and paint six cells of color 1 up to the marker.
#   * Fused task here: erase the mouth pixels, find the six-cell endpoint from the marker, then
#     paint `[1, 8, 3, 3, 4, 7]` along that segment. The marker cell changes from 1 in the input
#     to 7 in the output.
# - Difference from original: length comes from the marker, not content count.
# - Difference from pure D4: the painted segment is no longer monochrome; endpoint recoloring and
#   internal middle colors depend on distance and the cup's border/content pair.
#
# b) Complexity calibration:
# - The gradient chain is fixed and short (5 colors), so the color order is learnable from a few
#   examples instead of being an invisible numeric interpolation.
# - Border colors are restricted to later chain positions and content colors to earlier positions.
#   This makes all gradients one-way and keeps the non-bg palette at <= 5 colors.
# - Each generated grid has 2-3 cups, and at least one cup uses a long 5-color span. That gives
#   repeated overlapping subchains across a single example, making the middle-color order visible.
# - Marker distance is forced to exceed content count by at least 2, so the task cannot collapse
#   back to the original count-based behavior.
#
# c) Implicit-assumption audit against the original main():
# - `_flood_from_bbox_side(...)` still works because the new marker is outside the cup bbox.
# - `Counter(...).most_common(1)` over mouth cells still works because each cup still has exactly
#   one mouth-content color. D5/1 changes the output trajectory colors, not the input mouth colors.
# - The original ejection loop `range(1, len(content_cells)+1)` is invalid now:
#   `len(content_cells)` no longer determines endpoint or color pattern.
# - The original code assumed any encountered ray cell could be overwritten with content color.
#   Now the solver assumes there is exactly one same-color marker on the reserved spike ray and no
#   earlier same-color distractor there. The generator guarantees that.
# - New assumption: every cup's content and border colors lie on the fixed `GRADIENT_CHAIN` with
#   content strictly earlier than border. The generator constructs only such cups.
#
# d) Placement legality:
# - Cup bboxes still keep a 1-cell moat using dilated occupancy.
# - The full spike ray to the grid edge is reserved in `occ_path`, even though only a prefix up
#   to the marker is used. This prevents cup/path/marker collisions and keeps marker search unique.
# - The marker is never placed on the last ray cell at the edge, so "stop at marker" is distinct
#   from "run until boundary".
# - The chosen marker distance is always at least `gradient_span + 1`, so every intermediate chain
#   color can appear on the output segment.
#
# e) Visual observability:
# - The input visibly contains: cup border, mouth dots, and one aligned same-color marker on the
#   tail ray. The output visibly turns that single marker into the endpoint of a longer segment.
# - Because the endpoint cell changes from content color to border color, and intermediate cells
#   use overlapping colors seen elsewhere in the same grid, a human can infer both the stopping
#   rule and the color progression from 2-4 examples.
#
# f) Visual cleanliness:
# - Only one marker per cup, always on the isolated tail ray.
# - Only 5 non-bg colors are used globally, and cups/paths never overlap.
# - Optional stray pixels remain on the mouth side and never appear on a spike ray, so they do not
#   compete with the target markers.
#
# g) Color inferability:
# - The middle-color order is not derived from hidden color ids. It is a fixed visible chain with
#   repeated overlapping subchains across cups: `[1,8,3]`, `[8,3,4]`, `[1,8,3,4,7]`, etc.
# - The gradient is distance-controlled: the same endpoint pair can produce the same ordered color
#   chain stretched over different numbers of cells, which is human-visible in the outputs.
#
# Post-edit solver assumption audit:
# - `_first_marker_on_ray` assumes the first same-content-color pixel on the tail ray is the unique
#   target marker. The generator reserves the ray and never places stray pixels there.
# - `_gradient_sequence` assumes both endpoint colors are members of `GRADIENT_CHAIN` and that the
#   content color precedes the border color. The generator only samples such pairs.
# - `_choose_count_and_target` and `_try_place_cup` jointly assume the reserved ray is long enough
#   to host a non-edge marker with all chain colors visible. The generator rejects attempts that do
#   not satisfy that geometry.

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


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 400
GEN_RETRIES_FALLBACK = 400

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)

# Fixed visible gradient chain used by every sample.
GRADIENT_CHAIN = (1, 8, 3, 4, 7)
CHAIN_INDEX = {int(color): idx for idx, color in enumerate(GRADIENT_CHAIN)}
BORDER_POSITIONS = (2, 3, 4)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood within bbox, seeded from non-border cells on the chosen edge."""
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


def _first_marker_on_ray(
    g: np.ndarray, spike: tuple[int, int], direction: str, color: int,
) -> tuple[int, int] | None:
    """Return the first same-color marker on the outward spike ray."""
    h, w = g.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    while 0 <= rr < h and 0 <= cc < w:
        if int(g[rr, cc]) == int(color):
            return int(rr), int(cc)
        rr += dr
        cc += dc
    return None


def _gradient_sequence(content_color: int, border_color: int, length: int) -> list[int]:
    """Discrete gradient along the fixed visible chain, stretched to `length` cells."""
    start = CHAIN_INDEX.get(int(content_color))
    end = CHAIN_INDEX.get(int(border_color))
    if start is None or end is None or start >= end or length <= 0:
        return []
    if length == 1:
        return [int(content_color)]
    positions = np.rint(np.linspace(start, end, num=int(length))).astype(int)
    return [int(GRADIENT_CHAIN[int(idx)]) for idx in positions]


def _extract_cups(grid: np.ndarray) -> list[dict]:
    """Parse all cups that satisfy the fused rule assumptions."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    cups: list[dict] = []

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
                content_color = int(cnt.most_common(1)[0][0])
                target = _first_marker_on_ray(g, spike, direction, content_color)
                if target is None:
                    continue
                start_idx = CHAIN_INDEX.get(content_color)
                end_idx = CHAIN_INDEX.get(int(border_col))
                if start_idx is None or end_idx is None or start_idx >= end_idx:
                    continue
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_color]
                distance = abs(target[0] - spike[0]) + abs(target[1] - spike[1])
                grad = _gradient_sequence(content_color, int(border_col), distance)
                if len(grad) != distance:
                    continue
                passing.append(
                    {
                        "tail_dir": direction,
                        "spike": (int(spike[0]), int(spike[1])),
                        "target": target,
                        "border_color": int(border_col),
                        "content_color": content_color,
                        "content_cells": content_cells,
                        "distance": int(distance),
                        "gradient": grad,
                        "gradient_span": int(end_idx - start_idx),
                        "bbox": bbox,
                    }
                )

            if len(passing) == 1:
                cups.append(passing[0])

    return cups


def analyze_grid(grid: np.ndarray) -> list[dict]:
    """Diagnostic helper used by external validation scripts."""
    return _extract_cups(grid)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each cup's content toward its aligned marker, then recolor the whole segment as a
    fixed-chain gradient from content color to border color.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    for cup in _extract_cups(g):
        for r, c in cup["content_cells"]:
            out[r, c] = bg

        dr, dc = DIR4[cup["tail_dir"]]
        rr, cc = cup["spike"]
        for color in cup["gradient"]:
            rr += dr
            cc += dc
            out[rr, cc] = int(color)

    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1 : L + hL + 1, 0] = color
    if hR:
        patch[R + 1 : R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the cup interior from the mouth side."""
    h, w = patch.shape
    border_mask = patch == int(border_color)
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
    return sorted((int(r), int(c)) for r, c in visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate a canonical cup to the desired tail direction."""
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
    """Sample a canonical cup that fits the bbox limits after rotation."""
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
    min_ray_len: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Place a cup whose entire spike ray is clear and long enough."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords)
    for r0, c0 in coords:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((int(rr), int(cc)))
            rr += dr
            cc += dc
        if len(path_cells) < int(min_ray_len):
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return int(r0), int(c0), path_cells
    return None


def _choose_count_and_target(
    rng: np.random.Generator,
    mouth_size: int,
    ray_len: int,
    gradient_span: int,
) -> tuple[int, int] | None:
    """
    Pick `(content_count, target_distance)` so:
    - the marker is strictly before the grid edge,
    - the gradient can show every chain color between content and border,
    - the endpoint is visibly farther than the content count.
    """
    max_count = min(CONTENT_COUNT_MAX, int(mouth_size))
    if max_count < 1 or ray_len < gradient_span + 2:
        return None

    valid_pairs = []
    for target_distance in range(gradient_span + 1, ray_len):
        max_visible_count = min(max_count, target_distance - 2)
        for content_count in range(1, max_visible_count + 1):
            valid_pairs.append((content_count, target_distance))
    if not valid_pairs:
        return None
    return valid_pairs[int(rng.integers(0, len(valid_pairs)))]


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells lying on the mouth side of a cup bbox."""
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
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict],
) -> None:
    """Optionally add one isolated non-ray stray pixel on the mouth side of a cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for item in cups_meta:
        br0, bc0, br1, bc1 = item["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in item["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[int(r), int(c)] = int(stray_color)
    canvas_out[int(r), int(c)] = int(stray_color)


def _palette_pairs_for_grid(rng: np.random.Generator, num_cups: int) -> list[tuple[int, int]]:
    """
    Choose `(content_idx, border_idx)` pairs.
    At least one cup uses the full chain span 0->4 to make the gradient order obvious.
    """
    border_positions = sorted(int(x) for x in rng.choice(np.array(BORDER_POSITIONS, dtype=int), size=num_cups, replace=False))
    pairs: list[tuple[int, int]] = []
    for i, border_idx in enumerate(border_positions):
        if i == len(border_positions) - 1:
            content_idx = 0
        else:
            content_idx = max(0, border_idx - 2)
        pairs.append((int(content_idx), int(border_idx)))
    rng.shuffle(pairs)
    return pairs


def _add_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    content_idx: int,
    border_idx: int,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, place its marker, and commit the expected gradient segment."""
    border_color = int(GRADIENT_CHAIN[int(border_idx)])
    content_color = int(GRADIENT_CHAIN[int(content_idx)])
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None

    L_arm, R_arm, hL, hR = shape
    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    gradient_span = int(border_idx - content_idx)
    placement = _try_place_cup(
        rng,
        H,
        W,
        patch,
        spike,
        direction,
        min_ray_len=gradient_span + 2,
        occ_cup=occ_cup,
        occ_path=occ_path,
    )
    if placement is None:
        return None
    r0, c0, path_cells = placement

    chosen = _choose_count_and_target(rng, len(mouth), len(path_cells), gradient_span)
    if chosen is None:
        return None
    content_count, target_distance = chosen
    target_cell = path_cells[target_distance - 1]
    ejection_cells = path_cells[:target_distance]
    gradient = _gradient_sequence(content_color, border_color, target_distance)
    if len(gradient) != target_distance:
        return None

    ph, pw = patch.shape
    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True

    canvas_in[target_cell] = content_color
    for (r, c), color in zip(ejection_cells, gradient):
        canvas_out[r, c] = int(color)
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": int(content_count),
        "path_cells": path_cells,
        "target_cell": target_cell,
        "target_distance": int(target_distance),
        "gradient": gradient,
        "gradient_span": gradient_span,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's mouth-content pixels inside its mouth interior."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = int(meta["content_count"])
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[int(r), int(c)] = int(meta["content_color"])
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single output-first generation attempt."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    cups_meta: list[dict] = []

    for content_idx, border_idx in _palette_pairs_for_grid(rng, int(num_cups)):
        meta = _add_cup(
            rng,
            H,
            W,
            content_idx,
            border_idx,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != int(num_cups):
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
