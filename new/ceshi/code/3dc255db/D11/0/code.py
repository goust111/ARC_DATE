# Variant of 3dc255db
# Dimension: D11 — 杯的数量与空间关系 (meta)
# Change: 1-3个杯，不重叠(bbox间至少1格间距)，弹射路径无障碍... → 弹射路径可穿越其他杯
#
# Design analysis
# - Concrete behavior:
#   Input still contains 1-3 hollow cup/Y/funnel borders plus scattered content pixels inside each mouth.
#   Output still removes each cup's content and ejects it along the tail direction.
#   The new visual event is that a straight ray may run through another cup's bbox: when it reaches
#   that cup's border pixels, those border cells stay untouched, but the ray resumes on later cells.
#   Small example: if a rightward ray would hit cells [bg, border, bg, border, bg] and the cup has
#   3 content pixels, the original solver would incorrectly paint the first 3 coordinates; the variant
#   must paint the 1st, 3rd, and 5th coordinates instead.
# - Complexity calibration:
#   The core rule remains one straight tail-directed ejection per cup. Only one new branch is added:
#   "path cell is another cup border" => skip paint but do not consume one content pixel. Generator
#   still uses 1-3 cups and the same cup vocabulary, but it now prefers 2-3 cups and actively creates
#   at least one real skip-through-border case when possible so the difference is visible.
# - Old implicit assumptions that break:
#   1. Original main() used `for t in range(1, len(content_cells)+1)` and wrote every visited cell.
#      This relied on the generator guaranteeing an obstacle-free ray. It is false in D11.
#   2. Original generator rejected any cup placement whose ray touched another cup bbox/path. That
#      removes the new phenomenon entirely, so placement logic must be relaxed and revalidated.
#   3. Original main() erased and painted cup-by-cup. With cross-cup ray interactions, that creates
#      avoidable order sensitivity, so this variant erases all detected contents first, then paints rays.
# - Placement legality:
#   Cup bboxes still stay disjoint with at least one-cell gap. Rays may pass through other cup bboxes,
#   but generator forbids painted ray cells from overlapping other painted ray cells, and it forbids
#   content placement on cells that another cup's ray will occupy. That keeps the input valid and the
#   output unambiguous.
# - Visual observability:
#   The difference is visible because at least one multi-cup sample is forced to have a ray that shows
#   a gap on a foreign border and resumes afterward. Humans can see "line continues past obstacle"
#   from a few examples; no hidden color ordering is involved.
# - Visual cleanliness:
#   Only border pixels are protected. Rays may appear inside another cup's hollow interior, but the
#   border remains intact and content pixels are kept off those traversed cells, so the scene remains
#   parseable instead of becoming a full overlap puzzle.
# - Color inference:
#   Colors keep the original role: each cup has one border color and one content color. The new rule
#   never depends on numeric color ids, only on whether a cell belongs to a detected cup border.
# - New solver assumptions and generator guarantees:
#   main() assumes every obstacle that should be skipped is a true cup-border cell detected from the
#   input, and that different rays do not need tie-breaking on the same painted output cell. The
#   generator guarantees both by avoiding stray/content on rays and rejecting overlapping painted rays.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}


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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if not (r0 <= nr <= r1 and c0 <= nc <= c1):
                continue
            if (nr, nc) in visited or (nr, nc) in border_set:
                continue
            visited.add((nr, nc))
            stack.append((nr, nc))
    return visited


def _detect_cups(grid: np.ndarray) -> tuple[int, list[dict]]:
    """Detect cups exactly as in the original task, but expose border cells for the D11 skip logic."""
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
            cups.append({
                "border_color": int(border_col),
                "bbox": bbox,
                "border_set": border_set,
                "tail_dir": tail_dir,
                "spike": spike,
                "content_col": content_col,
                "content_cells": content_cells,
            })

    return bg, cups


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: eject content pixels along the tail direction, but when the ray reaches border
    pixels of any detected cup, leave those border pixels untouched and keep advancing until enough
    content-colored cells have been placed or the grid edge is reached.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg, cups = _detect_cups(g)
    out = g.copy()

    all_border_cells: set[tuple[int, int]] = set()
    for cup in cups:
        all_border_cells |= cup["border_set"]

    for cup in cups:
        for r, c in cup["content_cells"]:
            out[r, c] = bg

    for cup in cups:
        dr, dc = DIR4[cup["tail_dir"]]
        rr = cup["spike"][0] + dr
        cc = cup["spike"][1] + dc
        remaining = len(cup["content_cells"])
        placed = 0
        while placed < remaining and 0 <= rr < H and 0 <= cc < W:
            if (rr, cc) not in all_border_cells:
                out[rr, cc] = cup["content_col"]
                placed += 1
            rr += dr
            cc += dc

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)

# Primary generation prefers multi-cup scenes so the D11 behavior is visible.
PRIMARY_NUM_CUPS_CHOICES = (2, 3)
FALLBACK_NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

_STRUCT8 = np.ones((3, 3), dtype=bool)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical cup with tail pointing UP."""
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
    """Flood-fill the cup interior from the mouth side inside the patch bbox."""
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]]]:
    """Rotate canonical (tail=U) cup data to the target tail direction."""
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
    """Pick canonical cup parameters that fit the bbox limits after rotation."""
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


def _ray_cells_to_edge(H: int, W: int, spike: tuple[int, int], direction: str) -> list[tuple[int, int]]:
    """All cells strictly beyond the spike, up to the grid edge."""
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    ray: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        ray.append((rr, cc))
        rr += dr
        cc += dc
    return ray


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
    border_color: int,
    content_color: int,
    occ_bbox: np.ndarray,
    existing_border_cells: set[tuple[int, int]],
    existing_ray_cells: set[tuple[int, int]],
    prefer_crossing: bool,
) -> dict | None:
    """Place a cup bbox with >=1-cell gap; optionally prefer placements that create ray-border crossings."""
    ph, pw = patch.shape
    occ_bbox_dilated = ndimage.binary_dilation(occ_bbox, structure=_STRUCT8)
    patch_mask = patch != 0
    patch_coords = np.argwhere(patch_mask)

    candidates: list[dict] = []
    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_bbox_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue

        border_abs = {(r0 + int(r), c0 + int(c)) for r, c in patch_coords}
        spike_abs = (r0 + spike[0], c0 + spike[1])
        full_ray = _ray_cells_to_edge(H, W, spike_abs, direction)
        if not full_ray:
            continue
        mouth_abs = [(r0 + r, c0 + c) for (r, c) in mouth]

        ray_hits_old_border = sum(cell in existing_border_cells for cell in full_ray)
        old_ray_hits_new_border = sum(cell in existing_ray_cells for cell in border_abs)
        path_conflict = sum(
            (cell in existing_ray_cells) and (cell not in existing_border_cells) and (cell not in border_abs)
            for cell in full_ray
        )
        mouth_conflict = sum(cell in existing_ray_cells for cell in mouth_abs)
        candidates.append({
            "border_color": border_color,
            "content_color": content_color,
            "direction": direction,
            "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            "border_abs": border_abs,
            "spike_abs": spike_abs,
            "full_ray": full_ray,
            "mouth_abs": mouth_abs,
            "cross_score": ray_hits_old_border + old_ray_hits_new_border,
            "path_conflict": int(path_conflict),
            "mouth_conflict": int(mouth_conflict),
        })

    if not candidates:
        return None

    if prefer_crossing:
        crossing = [c for c in candidates if c["cross_score"] > 0]
        if crossing:
            crossing.sort(key=lambda c: (-c["cross_score"], c["path_conflict"], c["mouth_conflict"]))
            best = [c for c in crossing if (
                c["cross_score"] == crossing[0]["cross_score"]
                and c["path_conflict"] == crossing[0]["path_conflict"]
                and c["mouth_conflict"] == crossing[0]["mouth_conflict"]
            )]
            return best[int(rng.integers(0, len(best)))]

    candidates.sort(key=lambda c: (c["path_conflict"], c["mouth_conflict"], -c["cross_score"]))
    best = [c for c in candidates if (
        c["path_conflict"] == candidates[0]["path_conflict"]
        and c["mouth_conflict"] == candidates[0]["mouth_conflict"]
        and c["cross_score"] == candidates[0]["cross_score"]
    )]
    return best[int(rng.integers(0, len(best)))]


def _ray_profile(full_ray: list[tuple[int, int]], border_union: set[tuple[int, int]]) -> tuple[list[tuple[int, int]], int | None]:
    """Return paintable cells and the minimum content count needed to force at least one border skip."""
    paintable: list[tuple[int, int]] = []
    min_count_to_use_skip: int | None = None
    for idx, cell in enumerate(full_ray):
        if cell in border_union:
            if min_count_to_use_skip is None:
                for later in full_ray[idx + 1:]:
                    if later not in border_union:
                        min_count_to_use_skip = len(paintable) + 1
                        break
            continue
        paintable.append(cell)
    return paintable, min_count_to_use_skip


def _ejection_cells(
    full_ray: list[tuple[int, int]], border_union: set[tuple[int, int]], content_count: int,
) -> tuple[list[tuple[int, int]], int]:
    """First `content_count` paintable cells on the ray, plus the number of skipped border cells used."""
    ejection: list[tuple[int, int]] = []
    used_skips = 0
    for cell in full_ray:
        if len(ejection) >= content_count:
            break
        if cell in border_union:
            used_skips += 1
            continue
        ejection.append(cell)
    return ejection, used_skips


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, restricted to the cup's perpendicular extent."""
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
    """Optionally place one isolated stray pixel away from bboxes and all full ray cells."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in m["full_ray"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden
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
    occ_bbox: np.ndarray,
    existing_border_cells: set[tuple[int, int]],
    existing_ray_cells: set[tuple[int, int]],
    prefer_crossing: bool,
) -> dict | None:
    """Sample one cup shape and place it on the canvas metadata."""
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
    return _try_place_cup(
        rng, H, W, patch, spike, mouth, direction, border_color, content_color,
        occ_bbox, existing_border_cells, existing_ray_cells, prefer_crossing,
    )


def _choose_content_counts(rng: np.random.Generator, cups_meta: list[dict], require_crossing: bool) -> bool:
    """Assign content counts, forcing one real skip-through-border event when requested."""
    eligible: list[int] = []
    for i, meta in enumerate(cups_meta):
        cap = min(CONTENT_COUNT_MAX, len(meta["mouth_abs"]))
        if cap < 1:
            return False
        meta["content_cap"] = cap
        meta["paintable_ray"], meta["min_skip_count"] = _ray_profile(meta["full_ray"], meta["border_union"])
        if len(meta["paintable_ray"]) == 0:
            return False
        if meta["min_skip_count"] is not None and meta["min_skip_count"] <= cap:
            eligible.append(i)

    target_idx: int | None = None
    if require_crossing:
        if not eligible:
            return False
        target_idx = int(eligible[int(rng.integers(0, len(eligible)))])

    for i, meta in enumerate(cups_meta):
        cap = meta["content_cap"]
        if i == target_idx:
            lo = meta["min_skip_count"]
            meta["content_count"] = int(rng.integers(lo, cap + 1))
        else:
            meta["content_count"] = int(rng.integers(1, cap + 1))
    return True


def _finalize_paths(cups_meta: list[dict], require_crossing: bool) -> bool:
    """Build ejection cells and reject ambiguous/pathological interactions."""
    all_ejection_cells: set[tuple[int, int]] = set()
    total_used_skips = 0
    for meta in cups_meta:
        ejection_cells, used_skips = _ejection_cells(
            meta["full_ray"], meta["border_union"], meta["content_count"],
        )
        meta["ejection_cells"] = ejection_cells
        meta["used_skip_count"] = used_skips
        total_used_skips += used_skips
        if len(all_ejection_cells & set(ejection_cells)) > 0:
            return False
        all_ejection_cells |= set(ejection_cells)

    if require_crossing and total_used_skips == 0:
        return False

    for meta in cups_meta:
        allowed_mouth = [cell for cell in meta["mouth_abs"] if cell not in all_ejection_cells]
        if len(allowed_mouth) < meta["content_count"]:
            return False
        meta["available_mouth_abs"] = allowed_mouth
    return True


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Scatter content pixels inside mouth cells not reserved by other cups' rays."""
    for meta in cups_meta:
        mouth_abs = meta["available_mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for i in idxs:
            r, c = mouth_abs[int(i)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None, require_crossing: bool = True) -> np.ndarray | None:
    """Single output-first generation attempt for the D11 crossing variant."""
    if num_cups is None:
        num_cups = int(rng.choice(FALLBACK_NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    palette = list(range(1, 10))
    rng.shuffle(palette)

    occ_bbox = np.zeros((H, W), dtype=bool)
    existing_border_cells: set[tuple[int, int]] = set()
    existing_ray_cells: set[tuple[int, int]] = set()
    cups_meta: list[dict] = []

    need_crossing = require_crossing and num_cups > 1
    for _ in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng, H, W, border_color, content_color, occ_bbox,
            existing_border_cells, existing_ray_cells, prefer_crossing=(need_crossing and len(cups_meta) > 0),
        )
        if meta is None:
            return None

        r0, c0, r1, c1 = meta["bbox"]
        occ_bbox[r0:r1 + 1, c0:c1 + 1] = True
        existing_border_cells |= meta["border_abs"]
        existing_ray_cells |= set(meta["full_ray"])
        if meta["cross_score"] > 0:
            need_crossing = False
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None

    border_union = set().union(*(meta["border_abs"] for meta in cups_meta))
    for meta in cups_meta:
        meta["border_union"] = border_union

    if not _choose_content_counts(rng, cups_meta, require_crossing=(num_cups > 1 and require_crossing)):
        return None
    if not _finalize_paths(cups_meta, require_crossing=(num_cups > 1 and require_crossing)):
        return None

    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)
    for meta in cups_meta:
        for r, c in meta["border_abs"]:
            canvas_in[r, c] = meta["border_color"]
            canvas_out[r, c] = meta["border_color"]

    if not _scatter_content(rng, canvas_in, cups_meta):
        return None

    for meta in cups_meta:
        for r, c in meta["ejection_cells"]:
            canvas_out[r, c] = meta["content_color"]

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
    num_cups = int(rng.choice(PRIMARY_NUM_CUPS_CHOICES))
    for _ in range(GEN_RETRIES_PRIMARY):
        result = _gen_attempt(rng, num_cups=num_cups, require_crossing=True)
        if result is not None:
            return result

    for _ in range(GEN_RETRIES_FALLBACK):
        result = _gen_attempt(
            rng,
            num_cups=int(rng.choice(FALLBACK_NUM_CUPS_CHOICES)),
            require_crossing=True,
        )
        if result is not None:
            return result

    raise RuntimeError(f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts")
