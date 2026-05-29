# Variant of 3dc255db
# Dimension: D16 — 溢出处理 (dst)
# Change: overflow — 超出网格边界的弹射像素被丢弃... → 溢出回填杯内

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage

# -----------------------------------------------------------------------------
# Design analysis for the D16 variant
#
# a) Concrete behavior and trace
# - The cup border detection stays unchanged: a hollow Y/funnel frame has one tail
#   spike on a bbox edge and an open mouth on the opposite edge.
# - Original rule: erase all mouth content; shoot a straight line from the spike
#   until the grid edge; any remaining pixels are lost.
# - Variant rule: visible ejection still occupies the straight tail ray first, but
#   if content_count > visible_run_to_edge, the overflow pixels are placed back
#   inside the same cup. The refill order is deterministic: fill the mouth cells
#   farthest from the mouth opening first, then proceed row/column in raster order.
# - Example: tail=U, visible path length=2, content_count=5. Original output makes
#   2 pixels above the spike and drops the other 3. Variant output makes the same
#   2 pixels above the spike, then writes 3 pixels back inside the cup on the
#   highest interior row(s), because those cells are deepest/farthest from the
#   bottom mouth opening.
#
# b) Complexity calibration
# - Only one attribute changes: overflow destination.
# - To keep the change visually observable, generation guarantees at least one cup
#   per grid has content_count > visible_run_to_edge, so backfill is actually seen.
# - The backfill order uses only geometry, not hidden metadata or color ids, and
#   has few cases (one per tail direction).
#
# c) Old assumptions that break under D16
# - In main(), the old loop `break`s at the first out-of-bounds cell and implicitly
#   assumes all remaining content disappears. That is the core invalid assumption.
# - Erasing the original content is still valid, but now the solver must count how
#   many pixels were not drawn on-grid and write exactly that many back into mouth
#   cells. Therefore the solver must retain both `len(content_cells)` and the full
#   mouth interior, not just the visible path length.
# - The mouth flood itself still works because the variant does not change object
#   recognition or cup topology.
#
# d) Placement legality
# - Cups still have disjoint bboxes with a 1-cell gap.
# - Tail rays are still kept clear in the input and reserved in placement.
# - Overflow backfill never creates overlap because it uses the cup's own interior
#   cells, which are already inside its reserved bbox.
#
# e) Visual observability
# - Human-visible signature: when a cup is near an edge, the output shows both a
#   truncated line at the edge and a compact block of same-colored pixels reformed
#   deep inside the cup.
# - This differs clearly from the original task, where those extra pixels vanish.
#
# f) Visual cleanliness
# - Refill happens only inside a single cup's hollow interior, so there is no new
#   cross-object clutter.
# - Stray pixels remain outside cup bboxes and cannot obscure the refill region.
#
# g) Color inference
# - No new color mapping is introduced. Refill uses the same content color already
#   present in the input, so the only inference burden is spatial.
#
# New solver assumptions and how generation satisfies them
# - Assumption: each acted-on cup still yields a unique mouth/content color pair
#   under the existing disambiguation rule. Generator preserves the original cup
#   construction and single-color content.
# - Assumption: mouth interior ordering can be read from geometry alone. Generator
#   uses the same ordering helper as the solver when constructing expected outputs.
# - Assumption: at least one cup often overflows, otherwise D16 would be invisible.
#   Generator explicitly forces one cup per grid to have overflow whenever possible.
# -----------------------------------------------------------------------------


def _flood_from_bbox_side(border_set: set, bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on the given bbox edge,
    stopping at border cells. Returns the set of reached cells (includes seeds)."""
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


def _mouth_backfill_order(cells: list[tuple[int, int]] | set[tuple[int, int]], tail_dir: str) -> list[tuple[int, int]]:
    """Order mouth cells from deepest/farthest-from-mouth to nearest-to-mouth.

    Ties use simple raster order on the perpendicular axis so the result is
    deterministic and depends only on visible geometry.
    """
    coords = list(cells)
    if tail_dir == "U":   # mouth is on the bottom edge
        return sorted(coords, key=lambda rc: (rc[0], rc[1]))
    if tail_dir == "D":   # mouth is on the top edge
        return sorted(coords, key=lambda rc: (-rc[0], rc[1]))
    if tail_dir == "L":   # mouth is on the right edge
        return sorted(coords, key=lambda rc: (rc[1], rc[0]))
    if tail_dir == "R":   # mouth is on the left edge
        return sorted(coords, key=lambda rc: (-rc[1], rc[0]))
    raise ValueError(tail_dir)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject content pixels from cup-shaped frames outward along each cup's tail direction.

    D16 variant:
    - Visible ejection still fills the straight tail ray from the spike.
    - If more pixels remain after the ray leaves the grid, they are not dropped.
      Instead they refill the cup interior from the deepest cells first.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

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
                passing.append((d, spike, content_col, content_cells, mouth))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells, mouth = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            visible_count = 0
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col
                visible_count += 1

            overflow = len(content_cells) - visible_count
            if overflow > 0:
                for r, c in _mouth_backfill_order(mouth, tail_dir)[:overflow]:
                    out[r, c] = content_col

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CUP_BBOX_MAX_UD = (8, 7)  # (max_h, max_w) when tail is U or D
CUP_BBOX_MAX_LR = (7, 8)  # (max_h, max_w) when tail is L or R
ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


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
    """Flood-fill the interior of the cup patch from cells on the given bbox edge."""
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
    """Rotate a canonical (tail=U) patch, its spike and mouth coords to the target tail direction."""
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
    """Pick canonical cup parameters, clamped so the rotated patch still fits the bbox spec."""
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
    direction: str, occ_cup: np.ndarray, occ_path: np.ndarray,
    max_content_count: int, require_overflow: bool,
) -> tuple[int, int, list[tuple[int, int]]] | None:
    """Find a placement whose bbox has a 1-cell gap and whose tail ray is clear.

    If require_overflow=True, only accept placements where the on-grid tail ray is
    shorter than max_content_count, so some content can overflow and backfill.
    """
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
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
        rr, cc = sr_abs + dr, sc_abs + dc
        path_cells = []
        while 0 <= rr < H and 0 <= cc < W:
            path_cells.append((rr, cc))
            rr += dr
            cc += dc
        if not path_cells:
            continue
        if require_overflow and len(path_cells) >= max_content_count:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue
        return r0, c0, path_cells
    return None


def _choose_content_count(
    rng: np.random.Generator, max_content_count: int, path_len: int, require_overflow: bool,
) -> int | None:
    """Pick content count, optionally forcing at least one overflow pixel."""
    if require_overflow:
        if path_len >= max_content_count:
            return None
        extra_cap = min(3, max_content_count - path_len)
        overflow = int(rng.integers(1, extra_cap + 1))
        return path_len + overflow

    if path_len < max_content_count and rng.random() < 0.35:
        return int(rng.integers(path_len + 1, max_content_count + 1))
    return int(rng.integers(1, min(path_len, max_content_count) + 1))


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Boolean mask of cells on the mouth side of bbox, restricted to perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if tail_dir == "R":
        mask[r0:r1 + 1, :c0] = True
    elif tail_dir == "L":
        mask[r0:r1 + 1, c1 + 1:] = True
    elif tail_dir == "D":
        mask[:r0, c0:c1 + 1] = True
    else:  # "U"
        mask[r1 + 1:, c0:c1 + 1] = True
    return mask


def _try_place_stray(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict],
) -> None:
    """Optionally place one isolated non-bg pixel on the mouth side of a cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for (r, c) in m["path_cells"]:
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
    require_overflow: bool,
) -> dict | None:
    """Build one cup, place it, and commit both visible ejection and refill into canvas_out."""
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
    ph, pw = patch.shape
    max_content_count = min(CONTENT_COUNT_MAX, len(mouth))

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, occ_cup, occ_path, max_content_count, require_overflow,
    )
    if placement is None:
        return None
    r0, c0, path_cells = placement

    content_count = _choose_content_count(rng, max_content_count, len(path_cells), require_overflow)
    if content_count is None:
        return None

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for (r, c) in path_cells:
        occ_path[r, c] = True

    visible_count = min(content_count, len(path_cells))
    overflow_count = content_count - visible_count
    for (r, c) in path_cells[:visible_count]:
        canvas_out[r, c] = content_color

    mouth_abs = [(r0 + r, c0 + c) for (r, c) in mouth]
    backfill_cells = _mouth_backfill_order(mouth_abs, direction)[:overflow_count]
    for (r, c) in backfill_cells:
        canvas_out[r, c] = content_color

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": mouth_abs,
        "content_count": content_count,
        "path_cells": path_cells,
        "overflow_count": overflow_count,
        "backfill_cells": backfill_cells,
    }


def _scatter_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    """Place each cup's content_count pixels at random inside its mouth interior."""
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
    """Single end-to-end generation attempt. Uses output-first verification."""
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

    overflow_idx = int(rng.integers(0, num_cups))
    cups_meta: list[dict] = []
    for i in range(num_cups):
        if len(palette) < 2:
            return None
        border_color = int(palette.pop())
        content_color = int(palette.pop())
        palette.append(content_color)

        meta = _add_cup(
            rng, H, W, border_color, content_color,
            canvas_in, canvas_out, occ_cup, occ_path,
            require_overflow=(i == overflow_idx),
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not any(m["overflow_count"] > 0 for m in cups_meta):
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
