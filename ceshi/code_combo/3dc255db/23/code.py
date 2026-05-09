# Combo variant of 3dc255db (index 23)
# combo_spec: D4/0+D8/0
# Name: 障碍物停止+距离排序处理
#
# Design analysis for this fused variant
# =====================================
# a) Concrete behavior and trace:
# - The source task detects hollow cup/Y frames, finds the single-pixel tail spike, removes the
#   mouth content, and ejects a straight line from the spike.
# - D4/0 changes the stop rule: a ray no longer writes blindly to the grid edge; it halts at the
#   first obstacle cell and does not overwrite it.
# - D8/0 changes the processing model: when several cups exist, they are handled from nearer the
#   grid center to farther. Earlier emitted lines remain on the board and therefore become new
#   obstacles for later cups.
# - This combo makes both effects visible in the same sample. The near-center cup has a static
#   obstacle on its ray, so its own line is shortened. A farther cup has an orthogonal ray that
#   reaches the shortened first line; when processed later, it stops on that already-written line.
#   Small trace: cup A emits rightward, writes 3 cells, then stops at a static obstacle. Cup B
#   emits upward and would reach 4 cells, but after writing 2 cells it encounters one of A's 3
#   emitted cells and halts there. The output differs from both the original task and from either
#   single variant taken alone.
# b) Complexity calibration:
# - The generator always uses at least two cups and forces one cross-ray interaction. This prevents
#   degeneration to an order-insensitive single-cup case.
# - Only one explicit static obstacle is added, keeping the signal strong without cluttering the
#   board with many blocker types.
# - Cup shapes, sizes, orientations, exact crossing point, content counts, and grid size still
#   vary, so the instances are not template-memorization trivial.
# c) Implicit-assumption audit of the original main():
# - Original assumption: cups can be processed immediately while scanning colors/components. Broken:
#   order now matters globally, so all cups must be parsed first and sorted.
# - Original assumption: every in-bounds ray cell is writable until the edge. Broken: static
#   obstacle pixels and earlier emitted lines both block the ray.
# - Original assumption: blocker detection can ignore intermediate writes. Broken: for D8 the
#   current dynamic board must be consulted when ejecting, while cup detection itself must still use
#   the original input grid to avoid polluting mouth detection with earlier output lines.
# d) Placement legality:
# - Cup bboxes keep a 1-cell gap, so border components remain separate.
# - Rays are not allowed to pass through any cup bbox.
# - The two interacting rays cross at exactly one cell outside both bboxes.
# - The static obstacle is placed farther along the first cup's ray than the crossing cell, so it
#   shortens that first line without stealing the second cup's collision.
# e) Visual observability:
# - The ordering key is geometric: bbox center distance to the whole-grid center. No invisible color
#   ordering or hidden metadata is used.
# - The near-center cup is visibly closer to the middle, has a visible lone obstacle ahead of its
#   spike, and leaves a shortened line. The farther cup visibly heads into that line and stops.
# - Two to four examples are enough for a human to notice both "stop before obstacle" and
#   "near-center line blocks far-center line".
# f) Visual cleanliness:
# - Only one crossing pair is used. No stray pixels are added elsewhere.
# - The crossing is orthogonal rather than head-on, so the static obstacle and the dynamic collision
#   do not compete for the same cell.
# g) Color inferability:
# - Each cup keeps its own border/content colors, and the obstacle uses one extra distinct color.
#   The rule is structural, not color-coded.
# h) Generator-side behavior checks:
# - The generator rejects any sample unless at least one cup is statically blocked and at least one
#   cup is dynamically blocked by an earlier emitted line.
# - It also rejects samples whose fused output matches a "static-only" simulation or a
#   "dynamic-only" simulation, ensuring both dimensions matter.
# i) New solver assumptions and matching guarantees:
# - Assumption: every valid cup is still uniquely detectable by the original mouth flood logic.
#   Guarantee: cups remain non-touching, canonical Y/funnel variants with one true spike edge.
# - Assumption: distance ties should not decide behavior. Guarantee: generation enforces distinct
#   center-distance keys for the two cups.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import sys

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (2,)
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
MAX_B_FREE_BEFORE_DYNAMIC_BLOCK = 3
MAX_A_AFTER_CROSS_BEFORE_OBSTACLE = 3
GEN_RETRIES_PRIMARY = 500
GEN_RETRIES_FALLBACK = 500

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
ORTHOGONAL_PATTERNS = (
    ("R", "U"),
    ("R", "D"),
    ("L", "U"),
    ("L", "D"),
    ("U", "R"),
    ("U", "L"),
    ("D", "R"),
    ("D", "L"),
)


def _flood_from_bbox_side(border_set: set, bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood inside bbox, blocked by border cells."""
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


def _cup_order_key(bbox: tuple[int, int, int, int], H: int, W: int) -> tuple[int, int, int]:
    """Sort by squared bbox-center distance to grid center, then visible tie-breakers."""
    r0, c0, r1, c1 = bbox
    dr2 = (r0 + r1) - (H - 1)
    dc2 = (c0 + c1) - (W - 1)
    return dr2 * dr2 + dc2 * dc2, r0, c0


def _detect_cups(grid: np.ndarray) -> list[dict]:
    """Parse all cups from the static input grid."""
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
                (d, cells[0]) for d, cells in edges
                if len(cells) == 1 and cells[0] not in corners
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
            cups.append(
                {
                    "border_color": int(border_col),
                    "bbox": bbox,
                    "tail_dir": tail_dir,
                    "spike": spike,
                    "content_color": content_col,
                    "content_cells": list(content_cells),
                }
            )

    return cups


def _ordered_ejection_cells(
    out: np.ndarray,
    spike: tuple[int, int],
    tail_dir: str,
    content_count: int,
    bg: int,
) -> list[tuple[int, int]]:
    """Emit until count is exhausted, the edge is reached, or a current obstacle is hit."""
    H, W = out.shape
    dr, dc = DIR4[tail_dir]
    rr, cc = spike[0] + dr, spike[1] + dc
    cells: list[tuple[int, int]] = []
    while len(cells) < content_count and 0 <= rr < H and 0 <= cc < W:
        if int(out[rr, cc]) != bg:
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    1. Parse all cups from the original input.
    2. Process cups from nearer-center to farther-center.
    3. Erase each cup's mouth content.
    4. Emit along the tail until either the content count is exhausted, the grid edge is reached,
       or the current board already contains a non-bg obstacle. Earlier emitted lines therefore
       block later cups.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    cups = _detect_cups(g)
    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg
        for r, c in _ordered_ejection_cells(out, cup["spike"], cup["tail_dir"], len(cup["content_cells"]), bg):
            out[r, c] = int(cup["content_color"])

    return out


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical Y/funnel cup with the tail pointing up."""
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    patch = np.zeros((h, w), dtype=int)
    patch[0, L] = color
    patch[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    patch[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        patch[L + 1:L + hL + 1, 0] = color
    if hR:
        patch[R + 1:R + hR + 1, w - 1] = color
    return patch, (0, L)


def _mouth_interior_coords(patch: np.ndarray, border_color: int, opposite_side: str) -> list[tuple[int, int]]:
    """Flood-fill the mouth interior of a patch from the open side."""
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
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate the canonical up-tail patch into the requested tail direction."""
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
    """Sample a cup shape that fits the orientation-specific bbox bounds."""
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


def _build_cup_spec(
    rng: np.random.Generator,
    border_color: int,
    content_color: int,
    direction: str,
) -> dict | None:
    """Sample one cup shape with a fixed tail direction."""
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape
    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None
    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    return {
        "border_color": int(border_color),
        "content_color": int(content_color),
        "direction": str(direction),
        "patch": patch,
        "spike_rel": tuple(spike),
        "mouth_rel": list(mouth),
        "mouth_size": len(mouth),
        "ph": int(patch.shape[0]),
        "pw": int(patch.shape[1]),
    }


def _ray_from_spike(spike_abs: tuple[int, int], direction: str, H: int, W: int) -> list[tuple[int, int]]:
    """Cells from the spike outward to the grid edge."""
    dr, dc = DIR4[direction]
    rr, cc = spike_abs[0] + dr, spike_abs[1] + dc
    cells = []
    while 0 <= rr < H and 0 <= cc < W:
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells


def _finalize_cup(spec: dict, r0: int, c0: int, H: int, W: int) -> dict:
    """Attach absolute placement data to a sampled cup."""
    spike_abs = (r0 + int(spec["spike_rel"][0]), c0 + int(spec["spike_rel"][1]))
    patch = spec["patch"]
    ph, pw = patch.shape
    return {
        "border_color": int(spec["border_color"]),
        "content_color": int(spec["content_color"]),
        "direction": str(spec["direction"]),
        "patch": patch.copy(),
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "spike_abs": spike_abs,
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in spec["mouth_rel"]],
        "mouth_size": int(spec["mouth_size"]),
        "full_ray": _ray_from_spike(spike_abs, str(spec["direction"]), H, W),
    }


def _distance_value(bbox: tuple[int, int, int, int], H: int, W: int) -> int:
    return _cup_order_key(bbox, H, W)[0]


def _bboxes_have_gap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    """True iff two bboxes have at least one empty cell separating them."""
    return (
        a[2] + 1 < b[0]
        or b[2] + 1 < a[0]
        or a[3] + 1 < b[1]
        or b[3] + 1 < a[1]
    )


def _ray_hits_bbox(ray: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = bbox
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r, c in ray)


def _cell_near_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int], pad: int = 1) -> bool:
    r, c = cell
    r0, c0, r1, c1 = bbox
    return (r0 - pad) <= r <= (r1 + pad) and (c0 - pad) <= c <= (c1 + pad)


def _distance_keys_are_unique(cups_meta: list[dict], H: int, W: int) -> bool:
    keys = [_distance_value(meta["bbox"], H, W) for meta in cups_meta]
    return len(keys) == len(set(keys))


def _find_crossing_pair_placement(
    rng: np.random.Generator,
    H: int,
    W: int,
    spec_a: dict,
    spec_b: dict,
) -> tuple[dict, dict, tuple[int, int]] | None:
    """
    Place cup A (nearer the center) and cup B (farther) so:
    - A's ray is statically blocked after passing the crossing point.
    - B's orthogonal ray reaches that crossing and is dynamically blocked by A's emitted line.
    """
    coords_a = [(r0, c0) for r0 in range(H - int(spec_a["ph"]) + 1) for c0 in range(W - int(spec_a["pw"]) + 1)]
    coords_b = [(r0, c0) for r0 in range(H - int(spec_b["ph"]) + 1) for c0 in range(W - int(spec_b["pw"]) + 1)]
    rng.shuffle(coords_a)
    rng.shuffle(coords_b)

    max_a = min(CONTENT_COUNT_MAX, int(spec_a["mouth_size"]))
    max_b = min(CONTENT_COUNT_MAX, int(spec_b["mouth_size"]))
    if max_a < 2 or max_b < 2:
        return None

    candidates = []
    for r_a, c_a in coords_a:
        cup_a = _finalize_cup(spec_a, r_a, c_a, H, W)
        if len(cup_a["full_ray"]) < 3:
            continue
        ray_a_index = {cell: idx for idx, cell in enumerate(cup_a["full_ray"])}
        dist_a = _distance_value(cup_a["bbox"], H, W)

        for r_b, c_b in coords_b:
            cup_b = _finalize_cup(spec_b, r_b, c_b, H, W)
            if not _bboxes_have_gap(cup_a["bbox"], cup_b["bbox"]):
                continue
            if _ray_hits_bbox(cup_a["full_ray"], cup_b["bbox"]) or _ray_hits_bbox(cup_b["full_ray"], cup_a["bbox"]):
                continue

            dist_b = _distance_value(cup_b["bbox"], H, W)
            if dist_a >= dist_b:
                continue

            crossings = [cell for cell in cup_b["full_ray"] if cell in ray_a_index]
            if len(crossings) != 1:
                continue
            cross = crossings[0]
            idx_a = int(ray_a_index[cross])
            idx_b = int(cup_b["full_ray"].index(cross))
            if not (1 <= idx_b <= MAX_B_FREE_BEFORE_DYNAMIC_BLOCK):
                continue

            obs_lo = max(idx_a + 1, 1)
            obs_hi = min(
                len(cup_a["full_ray"]) - 1,
                max_a - 1,
                idx_a + MAX_A_AFTER_CROSS_BEFORE_OBSTACLE,
            )
            if obs_lo > obs_hi:
                continue

            count_b_lo = idx_b + 1
            count_b_hi = min(max_b, idx_b + 3)
            if count_b_lo > count_b_hi:
                continue

            for obs_idx in range(obs_lo, obs_hi + 1):
                obstacle_cell = cup_a["full_ray"][obs_idx]
                if _cell_near_bbox(obstacle_cell, cup_b["bbox"], pad=1):
                    continue
                candidates.append((r_a, c_a, r_b, c_b, obs_idx, count_b_lo, count_b_hi))

    if not candidates:
        return None

    r_a, c_a, r_b, c_b, obs_idx, count_b_lo, count_b_hi = candidates[int(rng.integers(0, len(candidates)))]
    cup_a = _finalize_cup(spec_a, r_a, c_a, H, W)
    cup_b = _finalize_cup(spec_b, r_b, c_b, H, W)
    count_a = int(rng.integers(obs_idx + 1, min(max_a, obs_idx + 2) + 1))
    count_b = int(rng.integers(count_b_lo, count_b_hi + 1))
    cup_a["content_count"] = count_a
    cup_b["content_count"] = count_b
    obstacle_cell = cup_a["full_ray"][obs_idx]
    cup_a["static_obstacle"] = obstacle_cell
    cup_b["static_obstacle"] = None
    return cup_a, cup_b, obstacle_cell


def _draw_borders(canvas: np.ndarray, cups_meta: list[dict]) -> None:
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        patch = meta["patch"]
        canvas[r0:r1 + 1, c0:c1 + 1][patch != 0] = int(meta["border_color"])


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter each cup's content pixels inside its mouth and record their exact locations."""
    for meta in cups_meta:
        mouth_abs = list(meta["mouth_abs"])
        count = int(meta["content_count"])
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        content_abs = []
        for idx in idxs:
            r, c = mouth_abs[int(idx)]
            canvas_in[r, c] = int(meta["content_color"])
            content_abs.append((r, c))
        meta["content_abs"] = content_abs
    return True


def _emit_cells_with_blocker_kind(
    out: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    bg: int,
    drawn_cells: set[tuple[int, int]],
) -> tuple[list[tuple[int, int]], str | None]:
    """Emit on the current board and classify the first blocker as static or dynamic."""
    H, W = out.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    cells: list[tuple[int, int]] = []
    blocker_kind: str | None = None
    while len(cells) < content_count and 0 <= rr < H and 0 <= cc < W:
        if int(out[rr, cc]) != bg:
            blocker_kind = "dynamic" if (rr, cc) in drawn_cells else "static"
            break
        cells.append((rr, cc))
        rr += dr
        cc += dc
    return cells, blocker_kind


def _simulate_fused(canvas_in: np.ndarray, cups_meta: list[dict]) -> tuple[np.ndarray, dict]:
    """Simulate the fused solver and return both the output and behavior metrics."""
    out = np.array(canvas_in, dtype=int)
    H, W = out.shape
    bg = int(get_bg(out))
    cups = list(cups_meta)
    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))

    drawn_cells: set[tuple[int, int]] = set()
    metrics = {"static_blocked": 0, "dynamic_blocked": 0}
    for meta in cups:
        for r, c in meta["content_abs"]:
            out[r, c] = bg
        cells, blocker_kind = _emit_cells_with_blocker_kind(
            out,
            meta["spike_abs"],
            meta["direction"],
            int(len(meta["content_abs"])),
            bg,
            drawn_cells,
        )
        if blocker_kind == "static":
            metrics["static_blocked"] += 1
        elif blocker_kind == "dynamic":
            metrics["dynamic_blocked"] += 1
        for r, c in cells:
            out[r, c] = int(meta["content_color"])
        drawn_cells.update(cells)
    return out, metrics


def _simulate_static_only(canvas_in: np.ndarray, cups_meta: list[dict]) -> np.ndarray:
    """D4-only reference: obstacles come only from the original input, not from earlier lines."""
    g = np.array(canvas_in, dtype=int)
    out = g.copy()
    H, W = out.shape
    bg = int(get_bg(out))
    cups = list(cups_meta)
    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))

    for meta in cups:
        for r, c in meta["content_abs"]:
            out[r, c] = bg
        dr, dc = DIR4[meta["direction"]]
        rr, cc = meta["spike_abs"][0] + dr, meta["spike_abs"][1] + dc
        cells = []
        while len(cells) < len(meta["content_abs"]) and 0 <= rr < H and 0 <= cc < W:
            if int(g[rr, cc]) != bg:
                break
            cells.append((rr, cc))
            rr += dr
            cc += dc
        for r, c in cells:
            out[r, c] = int(meta["content_color"])
    return out


def _simulate_dynamic_only(canvas_in: np.ndarray, cups_meta: list[dict]) -> np.ndarray:
    """D8-only reference on the same input: only earlier emitted lines block, static obstacles do not."""
    out = np.array(canvas_in, dtype=int)
    H, W = out.shape
    bg = int(get_bg(out))
    cups = list(cups_meta)
    cups.sort(key=lambda cup: (*_cup_order_key(cup["bbox"], H, W), int(cup["border_color"])))

    drawn_cells: set[tuple[int, int]] = set()
    for meta in cups:
        for r, c in meta["content_abs"]:
            out[r, c] = bg
        dr, dc = DIR4[meta["direction"]]
        rr, cc = meta["spike_abs"][0] + dr, meta["spike_abs"][1] + dc
        cells = []
        while len(cells) < len(meta["content_abs"]) and 0 <= rr < H and 0 <= cc < W:
            if (rr, cc) in drawn_cells:
                break
            cells.append((rr, cc))
            rr += dr
            cc += dc
        for r, c in cells:
            out[r, c] = int(meta["content_color"])
        drawn_cells.update(cells)
    return out


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Construct one fused sample and validate that both D4 and D8 effects are active."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    if num_cups != 2:
        return None

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))

    palette = list(range(1, 10))
    rng.shuffle(palette)
    border_a, content_a, border_b, content_b, obstacle_color = [int(palette[i]) for i in range(5)]

    dir_a, dir_b = ORTHOGONAL_PATTERNS[int(rng.integers(0, len(ORTHOGONAL_PATTERNS)))]
    spec_a = _build_cup_spec(rng, border_a, content_a, dir_a)
    spec_b = _build_cup_spec(rng, border_b, content_b, dir_b)
    if spec_a is None or spec_b is None:
        return None

    placed = _find_crossing_pair_placement(rng, H, W, spec_a, spec_b)
    if placed is None:
        return None
    cup_a, cup_b, obstacle_cell = placed
    cups_meta = [cup_a, cup_b]

    if not _distance_keys_are_unique(cups_meta, H, W):
        return None

    canvas_in = np.zeros((H, W), dtype=int)
    _draw_borders(canvas_in, cups_meta)
    canvas_in[obstacle_cell] = obstacle_color
    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    fused_out, metrics = _simulate_fused(canvas_in, cups_meta)
    static_only_out = _simulate_static_only(canvas_in, cups_meta)
    dynamic_only_out = _simulate_dynamic_only(canvas_in, cups_meta)

    if metrics["static_blocked"] < 1 or metrics["dynamic_blocked"] < 1:
        return None
    if np.array_equal(fused_out, static_only_out):
        return None
    if np.array_equal(fused_out, dynamic_only_out):
        return None

    try:
        produced = main(canvas_in)
    except Exception:
        return None
    if not np.array_equal(produced, fused_out):
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
