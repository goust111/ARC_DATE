# Combo variant of 3dc255db (index 9)
# combo_spec: D3/0+D5/2
# Name: 多色内容+周期颜色循环
#
# Design analysis
# ===============
# fused_variant:
#   rule_summary:
#     - Keep the original cup/frame detection and "erase mouth content then eject from the tail" geometry.
#     - Change the source object from one content color to 2-3 content colors inside the same mouth.
#     - Change the trajectory from a solid block or grouped blocks into a periodic cycle over those
#       content colors only.
#     - The cycle order is inferred from visible mouth position, not from color ids:
#         tail U/D -> order content colors by left-to-right mouth slices
#         tail L/R -> order content colors by top-to-bottom mouth slices
#   concretization_and_trace:
#     - Example: a right-pointing cup contains red pixels in upper mouth rows and blue pixels in lower
#       mouth rows, with 5 total content pixels.
#     - Original task: choose one content color and emit [red red red red red] or [blue ...].
#     - D3-only grouped task: emit [red red blue blue blue] according to per-color counts.
#     - Fused task here: infer order [red, blue], erase all 5 mouth pixels, then emit
#       [red blue red blue red].
#     - Difference: counts only determine total ray length; color order comes from spatial slices and
#       the ray repeats that order periodically.
#   complexity_calibration:
#     - Use only 2-3 content colors per cup.
#     - Require at least 4 content pixels per actionable cup and at least one cup per sample whose ray
#       length is >= 5, so the periodic repetition is visible.
#     - Force each color to occupy its own contiguous slice band in the mouth, so humans can read the
#       cycle order from position without relying on numeric color ids.
#   implicit_assumption_audit_main:
#     - Original `Counter(...).most_common(1)` assumed one content color; invalid now.
#     - Original erase step only removed one color; invalid now because every mouth content pixel must go.
#     - Original write step assumed monochrome output; invalid now because the output ray is periodic.
#     - New solver assumption: different content colors have distinct earliest mouth projections on the
#       reading axis. The generator guarantees this by assigning non-overlapping slice zones.
#   placement_legality:
#     - Keep output-first generation.
#     - Enforce 1-cell bbox gaps between cups and disjoint tail paths.
#     - Enforce path length >= full content count so the whole periodic sequence is visible.
#     - Stray pixels remain outside cup bboxes and all tail paths, so they cannot pollute mouth parsing.
#   visual_observability:
#     - The same ordered color set is seen twice: once as spaced content slices in the mouth, once as a
#       repeating ray. Across 2-4 examples this is enough to infer "read order -> periodic cycle".
#     - Because the ray repeats, the fused behavior is visually distinct from the D3 grouped-block output.
#   visual_cleanliness:
#     - Borders stay unique per cup.
#     - Content colors come from a small shared pool and never reuse border colors.
#     - Cups and rays remain spatially separated, preventing accidental merges.
#   anti_degeneracy_check:
#     - Generator requires total content count > number of content colors for every cup, so the output
#       cannot collapse to a single pass over the colors.
#     - The post-test metric compares the fused solver output against a grouped-block baseline on the same
#       generated inputs; samples are considered healthy only when they differ.
#   generator_guarantees_for_new_solver_assumptions:
#     - Each content color is assigned a dedicated mouth slice zone and appears at least once.
#     - Each cup has 2-3 colors and enough total pixels to show repetition.
#     - At least one cup per sample has a ray length >= 5 to make the cycle obvious.

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


def _flood_from_bbox_side(border_set: set, bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-neighbor flood inside a bbox, blocked by border cells."""
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


def _mouth_projection(cell: tuple[int, int], tail_dir: str) -> int:
    r, c = cell
    return int(c if tail_dir in ("U", "D") else r)


def _other_axis(cell: tuple[int, int], tail_dir: str) -> int:
    r, c = cell
    return int(r if tail_dir in ("U", "D") else c)


def _ordered_content_groups(
    g: np.ndarray,
    mouth: set[tuple[int, int]],
    bg: int,
    border_col: int,
    tail_dir: str,
) -> list[tuple[int, list[tuple[int, int]]]]:
    """Group mouth content by color and sort colors by visible mouth reading order."""
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for r, c in mouth:
        v = int(g[r, c])
        if v != bg and v != border_col:
            by_color[v].append((int(r), int(c)))
    if not by_color:
        return []

    def order_key(item: tuple[int, list[tuple[int, int]]]) -> tuple[float, float, int]:
        color, cells = item
        projections = [_mouth_projection(cell, tail_dir) for cell in cells]
        others = [_other_axis(cell, tail_dir) for cell in cells]
        # The final color tie-break is only for determinism; the generator keeps meaningful ties unlikely.
        return min(projections), sum(projections) / len(projections), min(others) * 100 + int(color)

    return sorted(by_color.items(), key=order_key)


def _extract_cup_actions(grid: np.ndarray) -> list[dict]:
    """Parse cups that carry mouth content and return their actionable metadata."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}
    actions: list[dict] = []

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
            candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
            if not candidates:
                continue

            passing = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, opposite[tail_dir])
                groups = _ordered_content_groups(g, mouth, bg, int(border_col), tail_dir)
                if not groups:
                    continue
                content_cells = [cell for _color, cells in groups for cell in cells]
                passing.append(
                    {
                        "tail_dir": tail_dir,
                        "spike": spike,
                        "groups": groups,
                        "content_cells": content_cells,
                        "border_col": int(border_col),
                    }
                )

            if len(passing) == 1:
                actions.append(passing[0])

    return actions


def _cycled_colors(colors: list[int], total_len: int) -> list[int]:
    return [int(colors[i % len(colors)]) for i in range(int(total_len))]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Detect each cup and its mouth content colors.
    2. Order the content colors by mouth position: left->right for vertical tails, top->bottom for horizontal tails.
    3. Erase all mouth content pixels.
    4. Emit a straight ray whose colors cycle periodically through that ordered color list.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    dir4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

    for action in _extract_cup_actions(g):
        ordered_colors = [int(color) for color, _cells in action["groups"]]
        total_count = len(action["content_cells"])

        for r, c in action["content_cells"]:
            out[int(r), int(c)] = bg

        dr, dc = dir4[action["tail_dir"]]
        sr, sc = action["spike"]
        for step, color in enumerate(_cycled_colors(ordered_colors, total_count), start=1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[int(rr), int(cc)] = int(color)

    return out


# --------------- generator ---------------

GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_GROUP_COUNT_RANGE = (2, 3)
GLOBAL_CONTENT_POOL_SIZE = 3
MIN_CONTENT_PIXELS_PER_CUP = 4
MIN_LONG_RAY = 5

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.4

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical cup whose tail points up."""
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
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
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

    L_arm = int(rng.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1] + 1))
    R_arm = int(rng.integers(ARM_LEN_RANGE[0], ARM_LEN_RANGE[1] + 1))
    while L_arm + R_arm + 1 > max_w_canon:
        if L_arm >= R_arm:
            L_arm -= 1
        else:
            R_arm -= 1

    hL = int(rng.integers(WALL_EXT_RANGE[0], WALL_EXT_RANGE[1] + 1))
    hR = int(rng.integers(WALL_EXT_RANGE[0], WALL_EXT_RANGE[1] + 1))
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


def _split_sorted_values(values: list[int], groups: int, rng: np.random.Generator) -> list[list[int]] | None:
    if len(values) < groups:
        return None
    if groups == 1:
        return [list(values)]
    cut_points = sorted(int(x) for x in rng.choice(np.arange(1, len(values)), size=groups - 1, replace=False))
    out: list[list[int]] = []
    start = 0
    for stop in [*cut_points, len(values)]:
        out.append(list(values[start:stop]))
        start = stop
    return out


def _choose_content_groups(
    rng: np.random.Generator,
    mouth: list[tuple[int, int]],
    direction: str,
    content_palette: list[int],
) -> tuple[list[dict], int] | None:
    unique_proj = sorted({_mouth_projection(cell, direction) for cell in mouth})
    feasible_group_counts: list[int] = []
    max_groups = min(CONTENT_GROUP_COUNT_RANGE[1], len(unique_proj), len(content_palette))
    for group_count in range(CONTENT_GROUP_COUNT_RANGE[0], max_groups + 1):
        min_total = max(MIN_CONTENT_PIXELS_PER_CUP, group_count + 1)
        if min_total <= min(CONTENT_COUNT_MAX, len(mouth)):
            feasible_group_counts.append(int(group_count))
    if not feasible_group_counts:
        return None

    rng.shuffle(feasible_group_counts)
    for num_groups in feasible_group_counts:
        proj_groups = _split_sorted_values(unique_proj, num_groups, rng)
        if proj_groups is None:
            continue

        zones: list[list[tuple[int, int]]] = []
        for proj_values in proj_groups:
            proj_set = set(proj_values)
            zone = [cell for cell in mouth if _mouth_projection(cell, direction) in proj_set]
            zone = sorted(zone, key=lambda rc: (_mouth_projection(rc, direction), _other_axis(rc, direction)))
            if not zone:
                break
            zones.append(zone)
        if len(zones) != num_groups:
            continue

        max_total = min(CONTENT_COUNT_MAX, sum(len(zone) for zone in zones))
        min_total = max(MIN_CONTENT_PIXELS_PER_CUP, num_groups + 1)
        if max_total < min_total:
            continue
        total_count = int(rng.integers(min_total, max_total + 1))

        colors = [int(v) for v in rng.choice(np.array(content_palette, dtype=int), size=num_groups, replace=False)]
        counts = [1] * num_groups
        capacities = [len(zone) - 1 for zone in zones]
        remaining = total_count - num_groups
        while remaining > 0:
            choices = [i for i, cap in enumerate(capacities) if cap > 0]
            if not choices:
                break
            idx = int(rng.choice(np.array(choices, dtype=int)))
            counts[idx] += 1
            capacities[idx] -= 1
            remaining -= 1
        if remaining != 0:
            continue

        groups = [
            {"color": int(colors[i]), "count": int(counts[i]), "zone": zones[i]}
            for i in range(num_groups)
        ]
        return groups, total_count

    return None


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    sequence_len: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
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
        if len(path_cells) < sequence_len:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells, path_cells[:sequence_len]

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
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    if rng.random() < STRAY_USE_BORDER_PROB:
        stray_color = int(meta["border_color"])
    else:
        stray_color = int(rng.choice(np.array([g["color"] for g in meta["content_groups"]], dtype=int)))

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in cup["path_cells"]:
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
    content_palette: list[int],
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
    if not mouth_canon:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    choice = _choose_content_groups(rng, mouth, direction, content_palette)
    if choice is None:
        return None
    content_groups, total_count = choice

    placement = _try_place_cup(rng, H, W, patch, spike, direction, total_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    ph, pw = patch.shape
    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True

    ordered_colors = [int(group["color"]) for group in content_groups]
    for (r, c), color in zip(ejection_cells, _cycled_colors(ordered_colors, total_count)):
        canvas_out[int(r), int(c)] = int(color)
    for r, c in path_cells:
        occ_path[int(r), int(c)] = True

    content_groups_abs = [
        {
            "color": int(group["color"]),
            "count": int(group["count"]),
            "zone_abs": [(r0 + r, c0 + c) for r, c in group["zone"]],
        }
        for group in content_groups
    ]

    return {
        "border_color": int(border_color),
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_groups": content_groups_abs,
        "path_cells": path_cells,
        "content_count": int(total_count),
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    for meta in cups_meta:
        for group in meta["content_groups"]:
            zone_abs = group["zone_abs"]
            count = int(group["count"])
            if len(zone_abs) < count:
                return False
            idxs = rng.choice(len(zone_abs), size=count, replace=False)
            for idx in idxs:
                r, c = zone_abs[int(idx)]
                canvas_in[int(r), int(c)] = int(group["color"])
    return True


def _grouped_block_baseline(grid: np.ndarray) -> np.ndarray:
    """Reference-only baseline used by external validation scripts."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()
    for action in _extract_cup_actions(g):
        for r, c in action["content_cells"]:
            out[int(r), int(c)] = bg
        dr, dc = DIR4[action["tail_dir"]]
        sr, sc = action["spike"]
        seq = [int(color) for color, cells in action["groups"] for _ in range(len(cells))]
        for step, color in enumerate(seq, start=1):
            rr = sr + dr * step
            cc = sc + dc * step
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[int(rr), int(cc)] = int(color)
    return out


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    if len(palette) < num_cups + GLOBAL_CONTENT_POOL_SIZE:
        return None
    border_colors = [int(palette.pop()) for _ in range(num_cups)]
    content_palette = [int(palette.pop()) for _ in range(GLOBAL_CONTENT_POOL_SIZE)]

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)
    cups_meta: list[dict] = []

    for border_color in border_colors:
        meta = _add_cup(rng, H, W, border_color, content_palette, canvas_in, canvas_out, occ_cup, occ_path)
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None

    visible_lengths = [int(meta["content_count"]) for meta in cups_meta]
    if not visible_lengths or max(visible_lengths) < MIN_LONG_RAY:
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
    if np.array_equal(_grouped_block_baseline(canvas_in), canvas_out):
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
