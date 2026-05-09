# Variant of 3dc255db
# Dimension: D2 — 方向确定机制 (path)
# Change: direction_from_shape — 通过识别Y形精灵的凸出尾部(尖刺)来确定弹射方向
#   -> 箭头像素指示方向
"""Variant of 3dc255db with arrow-indicated ejection direction.

Design analysis
===============
This variant keeps the original cup-content-ejection mechanic but changes how the
tail direction is determined.

a) Concrete behavior and trace
   - Original: a cup border is a hollow Y/funnel. The solver looks at the whole
     border component's bbox and finds the side with exactly one border pixel;
     that side is the tail direction.
   - Variant: every cup still has the same cup body, but now a same-color 3-pixel
     L-shaped arrow is attached near the tail side. The arrow is 8-connected to
     the cup, yet it sits off the direct ejection ray so it remains visible in
     both input and output.
   - The arrow is placed so the old bbox-based spike heuristic is no longer safe:
     the merged border component's outer bbox includes the arrow, so the cup body
     must be recovered by identifying and removing that arrow candidate first.
   - Small trace: suppose a tail-up cup contains three orange pixels. Originally
     the solver sees one top-edge spike and ejects upward. In the variant the same
     cup has a small same-color arrow above it. The merged component's top edge is
     now widened by the arrow, so "unique top-edge spike" is no longer reliable.
     The new solver removes the 3-cell arrow candidate, observes the arrow lies
     above the recovered cup, then uses UP as the ejection direction and ejects
     the three orange pixels upward from the cup's real spike.

b) Complexity calibration
   - Arrow shape is fixed to a 3-cell L triomino. This is enough to be visually
     explicit without introducing too many symbol categories.
   - Each cup always has exactly one arrow, and the arrow always uses the cup's
     border color. That gives repeated local support in multi-cup scenes.
   - The arrow may lean to either perpendicular side of the ray, so the samples do
     not collapse into one memorized offset template.

c) Implicit assumptions review
   - Original `main()` assumes the full border component is exactly the cup body.
     This is false now because the arrow is merged into the same 8-connected
     component.
   - Original edge counting on the full component bbox is no longer valid.
   - Original mouth flood is still valid, but only after stripping the arrow.
   - Original "unique passing candidate" logic is replaced with:
       1. enumerate 3-cell L candidates on the merged component fringe,
       2. remove one candidate,
       3. require the remainder to be an 8-connected cup body,
       4. derive direction from which side the arrow lies on relative to that cup.
   - The new solver assumes each relevant border component contains exactly one
     such valid arrow candidate. The generator enforces that.

d) Placement legality
   - The generator builds the original cup body first, then augments it with a
     same-color arrow offset near the tail side.
   - The arrow is kept off the direct ejection path, so the output line never
     overwrites the arrow.
   - Placement checks use the combined cup+arrow patch, so arrows do not overlap
     other cups or other ejection paths.

e) Visual observability
   - Humans can see a cup plus a small same-color arrow near one side. The output
     line extends from that side's spike direction. Across a few examples, the
     local symbol-to-direction relation is observable.
   - This does not degenerate to the original rule because the arrow is merged
     into the border component and changes the component bbox. Old shape-only
     logic no longer reliably recovers the direction.

f) Visual cleanliness
   - The arrow stays adjacent to its own cup and uses the same border color, so it
     reads as a local instruction attached to that cup rather than as unrelated
     clutter.
   - It is only 3 cells, and it never sits on the ejection ray.

g) Color inference
   - No new color mapping is introduced. The arrow reuses the cup border color.
     Direction is conveyed by local shape/placement, not by color identity.

Post-change solver assumptions and generator guarantees
-------------------------------------------------------
   - Assumption: each processed border component has one removable 3-cell L arrow
     whose removal yields one 8-connected cup body. Guaranteed by generator.
   - Assumption: the arrow lies completely beyond exactly one side of the cup
     body's bbox. Guaranteed by generator's fixed arrow templates.
   - Assumption: after removing the arrow, the cup body still has one spike on the
     arrow-indicated side. Guaranteed because the underlying cup body is the
     original generator's cup patch.
"""

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


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on the given bbox edge."""
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


def _is_connected8(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nxt = (r + dr, c + dc)
                if nxt in cells and nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == len(cells)


def _enumerate_arrow_candidates(border_set: set[tuple[int, int]]) -> list[set[tuple[int, int]]]:
    """Return all unique 3-cell 2x2-L subsets inside the component's bbox."""
    rs = [r for r, _ in border_set]
    cs = [c for _, c in border_set]
    r0, r1 = min(rs), max(rs)
    c0, c1 = min(cs), max(cs)
    seen: set[frozenset[tuple[int, int]]] = set()
    for wr in range(r0, r1):
        for wc in range(c0, c1):
            window = {(wr, wc), (wr, wc + 1), (wr + 1, wc), (wr + 1, wc + 1)}
            cells = border_set & window
            if len(cells) == 3:
                seen.add(frozenset(cells))
    return [set(x) for x in seen]


def _arrow_side(arrow_cells: set[tuple[int, int]], cup_bbox: tuple[int, int, int, int]) -> str | None:
    ar0, ac0, ar1, ac1 = bbox_coords(np.array(sorted(arrow_cells), dtype=int))
    r0, c0, r1, c1 = cup_bbox
    if ar1 < r0:
        return "U"
    if ar0 > r1:
        return "D"
    if ac1 < c0:
        return "L"
    if ac0 > c1:
        return "R"
    return None


def _find_spike(cup_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int] | None:
    r0, c0, r1, c1 = bbox
    corners = {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}
    if direction == "U":
        cells = [p for p in cup_set if p[0] == r0]
    elif direction == "D":
        cells = [p for p in cup_set if p[0] == r1]
    elif direction == "L":
        cells = [p for p in cup_set if p[1] == c0]
    else:
        cells = [p for p in cup_set if p[1] == c1]
    if len(cells) != 1 or cells[0] in corners:
        return None
    return cells[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule:
    - Each cup border component includes a same-color 3-pixel arrow marker.
    - Remove that marker conceptually to recover the cup body.
    - The side on which the arrow sits determines the ejection direction.
    - The content still ejects from the recovered cup's spike in that direction.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < 8:
                continue

            border_set = {(int(r), int(c)) for r, c in border_coords}
            passing: list[tuple[str, tuple[int, int], int, list[tuple[int, int]]]] = []

            for arrow_cells in _enumerate_arrow_candidates(border_set):
                cup_set = border_set - arrow_cells
                if len(cup_set) < 5 or not _is_connected8(cup_set):
                    continue

                cup_bbox = bbox_coords(np.array(sorted(cup_set), dtype=int))
                tail_dir = _arrow_side(arrow_cells, cup_bbox)
                if tail_dir is None:
                    continue

                spike = _find_spike(cup_set, cup_bbox, tail_dir)
                if spike is None:
                    continue

                mouth = _flood_from_bbox_side(cup_set, cup_bbox, OPPOSITE[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    v = int(g[r, c])
                    if v != bg and v != int(border_col):
                        cnt[v] += 1
                if not cnt:
                    continue

                content_col = int(cnt.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                if not content_cells:
                    continue
                passing.append((tail_dir, spike, content_col, content_cells))

            if len(passing) != 1:
                continue

            tail_dir, spike, content_col, content_cells = passing[0]
            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

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

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)

ARROW_OFFSETS = {
    "U": (
        [(-2, -2), (-2, -1), (-1, -1)],
        [(-2, 1), (-2, 2), (-1, 1)],
    ),
    "D": (
        [(1, -1), (2, -2), (2, -1)],
        [(1, 1), (2, 1), (2, 2)],
    ),
    "L": (
        [(-2, -2), (-1, -2), (-1, -1)],
        [(1, -2), (1, -1), (2, -2)],
    ),
    "R": (
        [(-2, 1), (-1, 1), (-1, 2)],
        [(1, 1), (1, 2), (2, 2)],
    ),
}


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a Y/funnel cup with tail pointing UP."""
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


def _attach_arrow(
    rng: np.random.Generator,
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
    color: int,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]], tuple[int, int, int, int]]:
    """Expand the original cup patch with a same-color 3-cell arrow."""
    border_coords = [(int(r), int(c)) for r, c in np.argwhere(patch == color)]
    arrow_template = ARROW_OFFSETS[direction][int(rng.integers(0, 2))]
    arrow_coords = [(spike[0] + dr, spike[1] + dc) for dr, dc in arrow_template]

    all_coords = border_coords + arrow_coords
    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_r = max(r for r, _ in all_coords)
    max_c = max(c for _, c in all_coords)

    combo = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in border_coords:
        combo[r - min_r, c - min_c] = color
    for r, c in arrow_coords:
        combo[r - min_r, c - min_c] = color

    shifted_spike = (spike[0] - min_r, spike[1] - min_c)
    shifted_mouth = [(r - min_r, c - min_c) for (r, c) in mouth]
    cup_bbox = (-min_r, -min_c, -min_r + patch.shape[0] - 1, -min_c + patch.shape[1] - 1)
    return combo, shifted_spike, shifted_mouth, cup_bbox


def _ray_inside_patch_is_clear(patch: np.ndarray, spike: tuple[int, int], direction: str) -> bool:
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    while 0 <= rr < ph and 0 <= cc < pw:
        if patch[rr, cc] != 0:
            return False
        rr += dr
        cc += dc
    return True


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    if not _ray_inside_patch_is_clear(patch, spike, direction):
        return None

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        patch_mask = patch != 0
        if np.any(occ_path[r0:r0 + ph, c0:c0 + pw] & patch_mask):
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

        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
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
    for m in cups_meta:
        br0, bc0, br1, bc1 = m["combo_bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for (r, c) in m["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["cup_bbox"], meta["direction"]) & ~forbidden
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

    patch_body, spike_body, mouth_body = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    patch, spike, mouth, cup_bbox_local = _attach_arrow(
        rng, patch_body, spike_body, mouth_body, direction, border_color
    )
    ph, pw = patch.shape

    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))
    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    cr0, cc0, cr1, cc1 = cup_bbox_local
    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "cup_bbox": (r0 + cr0, c0 + cc0, r0 + cr1, c0 + cc1),
        "combo_bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
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
            canvas_in[r, c] = meta["content_color"]
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
