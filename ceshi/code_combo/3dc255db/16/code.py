# Combo variant of 3dc255db (index 16)
# combo_spec: D4/1+D6/1
# Name: 目标标记停止+边框擦除
#
# Design analysis
# ===============
# Reference check
# - D4/1 reference code is available and establishes the "same-color marker on
#   the spike ray sets the stopping point" mechanism.
# - A dedicated D6/1 code file is not present in the workspace, so the D6 part
#   is implemented directly from the YAML description: after ejection, the whole
#   cup frame is erased too, so the output keeps only the emitted line.
#
# a) Concrete behavior and flow trace
# - Input objects:
#   * a hollow Y/funnel cup border with a single-pixel spike on one bbox side
#   * scattered content pixels of one color inside the mouth interior
#   * one same-color target marker placed on the outward ray starting at the
#     spike and going along the tail direction
# - Output objects:
#   * only the straight content-colored segment from the spike ray's first cell
#     through the target marker cell
#   * no cup border remains, and no source content remains
# - Small trace:
#   * Original task: 4 content pixels, clear ray to edge -> paint 4 cells, keep
#     the border visible.
#   * D4/1 alone: if the marker is 2 cells away, paint exactly 2 cells and keep
#     the border visible.
#   * This combo: same 2-cell marker stop, but the cup itself disappears, so the
#     output is just that 2-cell line on a blank background.
# - The two changes are coupled: the solver must still detect the erased cup from
#   the input, but the final visible evidence is only the marker-bounded line.
#
# b) Complexity calibration
# - Each cup still has exactly one relevant aligned marker. That is enough to
#   make the D4 change observable without creating a multi-marker matching task.
# - The D6 change is made visually strong by starting the output from a blank
#   background rather than copying the input and deleting piecemeal. This makes
#   the border-erasure effect unambiguous in 2-3 examples.
# - To prevent degeneration toward count-based emission, generation enforces
#   `content_count != target_distance` and also `content_count != target_distance
#   - 1`. The second exclusion matters because a copied marker can accidentally
#   mimic a longer line if only the endpoint cell differs.
#
# c) Implicit-assumption audit of the original solver
# - `_flood_from_bbox_side(...)` assumes the mouth is recoverable by flooding
#   inside the bbox from the side opposite the candidate spike. This still holds:
#   the new marker is outside the bbox.
# - The original dominant-color count inside the flooded mouth assumes the cup's
#   content color is visible inside the mouth. This remains true because the
#   generator always scatters at least one source pixel inside the mouth.
# - The original count-based ejection loop no longer applies. The new invariant
#   is "find the first same-color marker on the spike ray and paint until it".
# - The original `out = g.copy()` assumption breaks under D6/1, because that
#   would preserve borders and unrelated pixels. The combo solver instead starts
#   from an all-background canvas and redraws only the solved rays.
# - Because the solver now searches for the first same-color pixel on the ray,
#   the generator must keep that full ray reserved and obstacle-free.
#
# d) Placement legality
# - Cup bboxes still keep at least a 1-cell gap via dilated occupancy.
# - The entire spike ray to the edge is reserved in `occ_path`, even though only
#   a prefix up to the marker is used in the output. This prevents overlaps with
#   other cups, other markers, and other future rays.
# - Markers are never placed on the last ray cell at the boundary. Therefore the
#   stopping condition is visibly "marker-based" rather than "runs to the edge".
#
# e) Visual observability
# - In the input, the marker is the only same-color pixel outside the mouth and
#   is directionally aligned with the spike.
# - In the output, the border is gone, so the user sees a drastic simplification:
#   each cup collapses into a single straight segment that stops at a pre-existing
#   marker location instead of the canvas edge.
# - Across examples, varying the marker distance while keeping content count
#   different makes the stopping rule human-visible.
#
# f) Visual cleanliness
# - This combo deliberately disables stray pixels. Since the output is supposed to
#   contain only emitted lines, extra isolated pixels would add unexplained noise.
# - Full-ray reservation keeps emitted lines and markers from tangling with other
#   cups, so the input remains legible despite the simplified output.
#
# g) Change inferability
# - No hidden color ordering is used. The marker color equals the cup's content
#   color, and that same color becomes the output line color.
# - The line geometry is determined only by visible structure: cup orientation,
#   spike position, and aligned marker position.

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
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str,
) -> set[tuple[int, int]]:
    """4-connected flood within bbox, seeded from one bbox side and blocked by the border."""
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _first_color_on_ray(
    g: np.ndarray, spike: tuple[int, int], direction: str, color: int,
) -> tuple[int, int] | None:
    """Return the first cell on the outward spike ray whose value equals `color`."""
    H, W = g.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    while 0 <= rr < H and 0 <= cc < W:
        if int(g[rr, cc]) == color:
            return rr, cc
        rr += dr
        cc += dc
    return None


def _ray_distance(spike: tuple[int, int], target: tuple[int, int], direction: str) -> int:
    """Positive number of ray steps from `spike` to `target`."""
    dr, dc = DIR4[direction]
    if dr != 0:
        return int((target[0] - spike[0]) // dr)
    return int((target[1] - spike[1]) // dc)


def _detect_cups(grid: np.ndarray) -> list[dict]:
    """
    Parse cups from the input.

    Each returned dict contains:
    - border_color, content_color
    - direction, spike, target
    - mouth cells and content cells
    - target_distance
    """
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
                (direction, cells[0])
                for direction, cells in edges
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
                target = _first_color_on_ray(g, spike, direction, content_col)
                if target is None:
                    continue

                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                if not content_cells:
                    continue

                passing.append(
                    {
                        "border_color": int(border_col),
                        "content_color": content_col,
                        "direction": direction,
                        "spike": (int(spike[0]), int(spike[1])),
                        "target": (int(target[0]), int(target[1])),
                        "target_distance": _ray_distance(spike, target, direction),
                        "bbox": bbox,
                        "mouth": sorted(mouth),
                        "content_cells": sorted(content_cells),
                        "border_coords": sorted(border_set),
                    }
                )

            if len(passing) == 1:
                cups.append(passing[0])

    return cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Emit a straight line from each cup's spike toward its aligned marker, and erase the cup.

    Output is rebuilt from background so the cup border and source content both disappear.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    for cup in _detect_cups(g):
        dr, dc = DIR4[cup["direction"]]
        rr, cc = cup["spike"][0] + dr, cup["spike"][1] + dc
        while True:
            out[rr, cc] = cup["content_color"]
            if (rr, cc) == cup["target"]:
                break
            rr += dr
            cc += dc

    return out


def _build_canonical_cup(
    L: int, R: int, hL: int, hR: int, color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing up in canonical orientation."""
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


def _mouth_interior_coords(
    patch: np.ndarray, border_color: int, opposite_side: str,
) -> list[tuple[int, int]]:
    """Flood-fill the canonical mouth interior from the mouth side."""
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate the canonical patch, spike, and mouth coordinates to the target direction."""
    h, w = patch.shape
    sr, sc = spike
    if direction == "U":
        return patch.copy(), (sr, sc), list(mouth)
    if direction == "D":
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
    raise ValueError(direction)


def _sample_cup_shape(rng: np.random.Generator, direction: str) -> tuple[int, int, int, int] | None:
    """Pick canonical cup parameters that still fit the original bbox regime after rotation."""
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
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Place a cup so its whole spike ray to the edge is empty and reserved."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    placements = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(placements)

    for r0, c0 in placements:
        if occ_cup_dilated[r0 : r0 + ph, c0 : c0 + pw].any():
            continue
        if occ_path[r0 : r0 + ph, c0 : c0 + pw].any():
            continue

        sr_abs = r0 + spike[0]
        sc_abs = c0 + spike[1]
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((rr, cc))
            rr += dr
            cc += dc

        if len(path_cells) < 3:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells

    return None


def _choose_count_and_target(
    rng: np.random.Generator, mouth_size: int, ray_len: int,
) -> tuple[int, int] | None:
    """
    Pick `(content_count, target_distance)` such that the marker is strictly before the edge
    and the combo does not collapse to a count-based endpoint.
    """
    max_count = min(CONTENT_COUNT_MAX, mouth_size)
    if max_count < 1 or ray_len < 3:
        return None

    target_choices = list(range(2, ray_len))
    pairs = [
        (count, target_distance)
        for count in range(1, max_count + 1)
        for target_distance in target_choices
        if count not in {target_distance, target_distance - 1}
    ]
    if not pairs:
        return None

    return pairs[int(rng.integers(0, len(pairs)))]


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
    """Build one cup, place one aligned marker, and commit the output-only ray."""
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
    ph, pw = patch.shape

    placement = _try_place_cup(rng, H, W, patch, spike, direction, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells = placement

    chosen = _choose_count_and_target(rng, len(mouth), len(path_cells))
    if chosen is None:
        return None
    content_count, target_distance = chosen
    target_cell = path_cells[target_distance - 1]
    ejection_cells = path_cells[:target_distance]

    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_in[target_cell] = content_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "target_cell": target_cell,
        "target_distance": target_distance,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's source content pixels inside its mouth interior."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        for idx in rng.choice(len(mouth_abs), size=count, replace=False):
            r, c = mouth_abs[int(idx)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _sample_color_pairs(rng: np.random.Generator, num_cups: int) -> list[tuple[int, int]]:
    """
    Pick unique border colors and a small disjoint content-color pool.

    Keeping content colors disjoint from border colors prevents same-color markers from
    merging into other cups' 8-connected border components. Using at most two content
    colors also keeps the total palette close to the original task's visual complexity.
    """
    colors = list(range(1, 10))
    rng.shuffle(colors)

    border_colors = [int(c) for c in colors[:num_cups]]
    remaining = [int(c) for c in colors[num_cups:]]
    content_pool_size = 1 if num_cups == 1 else 2
    content_pool = remaining[:content_pool_size]

    assignments = [content_pool[i % len(content_pool)] for i in range(num_cups)]
    rng.shuffle(assignments)
    return [(border_colors[i], int(assignments[i])) for i in range(num_cups)]


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Single generation attempt using output-first construction."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    cups_meta: list[dict] = []

    for border_color, content_color in _sample_color_pairs(rng, num_cups):
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

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, canvas_out):
        return None

    # Enforce that the marker stop is genuinely different from count-based ejection.
    if not all(meta["content_count"] != meta["target_distance"] for meta in cups_meta):
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
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
