# Combo variant of 3dc255db (index 29)
# combo_spec: D3/2+D12/2
# Name: 干扰小形状+连通色块内容

from __future__ import annotations

import sys
import numpy as np
from scipy import ndimage

sys.path.append("/home/lixinyu/snap/new/solvers")
from arc_utils import bbox_coords, colors_present, components_mask, get_bg  # noqa: E402


# Design analysis for the fused variant
# ------------------------------------
# a) Concrete behavior:
#    Original: each cup contains 1-7 scattered same-color pixels inside the mouth; an optional
#    stray is a single pixel outside a cup on the mouth side. Output erases the interior pixels
#    and emits a 1-cell-wide straight line from the tail spike, whose length equals the number
#    of interior pixels.
#    Fused variant: the cup interior now contains one 4-connected same-color blob, still fully
#    inside the mouth. The optional stray is upgraded to a separate 2-3 pixel 4-connected shape
#    outside every cup bbox. Example: if a right-pointing cup contains a 3-cell connected blob,
#    the blob disappears and a 3-cell horizontal line is drawn to the right of the spike; a
#    2-cell mouth-side domino elsewhere stays unchanged because it lies outside the flooded mouth.
# b) Complexity calibration:
#    The blob size is still small (2-7), so ejection length remains easy to infer, but it is now
#    visually different from the old scattered pattern. Strays are limited to a few repeated
#    shape types (domino, tromino line, L-tromino) so humans can see "small outside shapes are
#    noise" without the grid becoming chaotic.
# c) Broken assumptions in the original solver:
#    - "Content can be recovered by counting all non-bg non-border cells in the mouth" still
#      holds spatially, but the new generator guarantees something stronger: the true content is
#      the unique 4-connected monochromatic component inside the mouth.
#    - "A stray is only one pixel" no longer holds, so the solver must not use component size to
#      decide globally; it instead restricts attention to cells inside the flooded mouth region.
#    - "All same-colored cells near the cup are content" is now false because the stray may share
#      either the border or content color. Solver therefore relies on cup-relative position.
# d) Placement legality:
#    Generator keeps cup bboxes separated by >=1 cell, reserves the full ejection paths, keeps the
#    blob strictly inside the mouth, and places the stray shape outside all bboxes and paths.
# e/f/g) Human observability and cleanliness:
#    Connected blobs are clearly different from scattered dots, and the 2-3 pixel outside shapes
#    are visually distinct but sparse. No rule depends on numeric color ids; colors are only
#    structural labels reused across cups and strays.


DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-conn flood within bbox, seeded from non-border cells on one bbox edge."""
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


def _extract_mouth_blob(
    g: np.ndarray, bg: int, border_col: int, mouth: set[tuple[int, int]]
) -> tuple[int, list[tuple[int, int]]] | None:
    """Return the unique monochromatic 4-connected component inside ``mouth``."""
    H, W = g.shape
    occupied_mask = np.zeros((H, W), dtype=bool)
    occupied = []
    for r, c in mouth:
        val = int(g[r, c])
        if val != bg and val != border_col:
            occupied_mask[r, c] = True
            occupied.append((r, c))
    if not occupied:
        return None

    good_components: list[tuple[int, list[tuple[int, int]]]] = []
    for comp in components_mask(occupied_mask, connectivity=4):
        cells = [(int(r), int(c)) for r, c in comp]
        cols = {int(g[r, c]) for r, c in cells}
        if len(cols) == 1:
            good_components.append((next(iter(cols)), cells))
    if len(good_components) != 1:
        return None
    color, cells = good_components[0]
    return color, cells


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fused rule:
    - Cup content is a connected blob fully inside the mouth.
    - Cup-external 2-3 pixel shapes on the mouth side are distractors and remain unchanged.
    - Only the mouth blob is erased and ejected in a straight line from the tail spike.
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
            candidates = [(d, cells[0]) for d, cells in edges if len(cells) == 1 and cells[0] not in corners]
            if not candidates:
                continue

            passing = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                blob = _extract_mouth_blob(g, bg, int(border_col), mouth)
                if blob is None:
                    continue
                content_col, content_cells = blob
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
CONTENT_COUNT_MIN = 2
CONTENT_COUNT_MAX = 7
STRAY_SHAPE_PROB = 0.8
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
STRAY_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    assert L >= 1 and R >= 1 and hL >= 0 and hR >= 0 and (hL + hR) >= 1
    w = L + R + 1
    h = max(L + hL, R + hR) + 1
    g = np.zeros((h, w), dtype=int)
    g[0, L] = color
    g[np.arange(1, L + 1), L - np.arange(1, L + 1)] = color
    g[np.arange(1, R + 1), L + np.arange(1, R + 1)] = color
    if hL:
        g[L + 1 : L + hL + 1, 0] = color
    if hR:
        g[R + 1 : R + hR + 1, w - 1] = color
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
        for dr, dc in DIR4.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in visited and not border_mask[nr, nc]:
                visited.add((nr, nc))
                stack.append((nr, nc))
    return sorted(visited)


def _rotate_to_direction(
    patch: np.ndarray, spike: tuple[int, int], mouth: list[tuple[int, int]], direction: str
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
        if not path_cells:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _sample_connected_subset(
    rng: np.random.Generator, cells: list[tuple[int, int]], count: int
) -> list[tuple[int, int]] | None:
    cell_set = set(cells)
    if count < 1 or count > len(cells):
        return None
    seed = cells[int(rng.integers(0, len(cells)))]
    chosen = {seed}
    while len(chosen) < count:
        frontier = []
        for r, c in chosen:
            for dr, dc in DIR4.values():
                nxt = (r + dr, c + dc)
                if nxt in cell_set and nxt not in chosen:
                    frontier.append(nxt)
        frontier = list(dict.fromkeys(frontier))
        if not frontier:
            return None
        chosen.add(frontier[int(rng.integers(0, len(frontier)))])
    return sorted(chosen)


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


def _can_place_shape(
    coords: list[tuple[int, int]], H: int, W: int, occupied_nonbg: np.ndarray, forbidden_mask: np.ndarray
) -> bool:
    coord_set = set(coords)
    for r, c in coords:
        if not (0 <= r < H and 0 <= c < W):
            return False
        if forbidden_mask[r, c] or occupied_nonbg[r, c]:
            return False
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and occupied_nonbg[nr, nc] and (nr, nc) not in coord_set:
                    return False
    return True


def _try_place_stray_shape(
    rng: np.random.Generator, canvas_in: np.ndarray, canvas_out: np.ndarray, cups_meta: list[dict]
) -> int:
    if not cups_meta or rng.random() >= STRAY_SHAPE_PROB:
        return 0

    H, W = canvas_in.shape
    forbidden_mask = np.zeros((H, W), dtype=bool)
    for meta in cups_meta:
        r0, c0, r1, c1 = meta["bbox"]
        forbidden_mask[r0 : r1 + 1, c0 : c1 + 1] = True
        for r, c in meta["path_cells"]:
            forbidden_mask[r, c] = True

    cup_order = list(range(len(cups_meta)))
    rng.shuffle(cup_order)
    shape_order = list(range(len(STRAY_SHAPES)))
    for cup_idx in cup_order:
        meta = cups_meta[cup_idx]
        stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]
        candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & ~forbidden_mask
        anchors = [tuple(map(int, p)) for p in np.argwhere(candidate_mask)]
        if not anchors:
            continue
        rng.shuffle(anchors)
        rng.shuffle(shape_order)
        occupied_nonbg = canvas_in != 0
        for shape_idx in shape_order:
            shape = STRAY_SHAPES[shape_idx]
            for ar, ac in anchors:
                coords = [(ar + dr, ac + dc) for dr, dc in shape]
                if any(not (0 <= r < H and 0 <= c < W) or not candidate_mask[r, c] for r, c in coords):
                    continue
                if not _can_place_shape(coords, H, W, occupied_nonbg, forbidden_mask):
                    continue
                for r, c in coords:
                    canvas_in[r, c] = stray_color
                    canvas_out[r, c] = stray_color
                return len(coords)
    return 0


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
    if len(mouth_canon) < CONTENT_COUNT_MIN:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    max_count = min(CONTENT_COUNT_MAX, len(mouth))
    if max_count < CONTENT_COUNT_MIN:
        return None
    content_count = int(rng.integers(CONTENT_COUNT_MIN, max_count + 1))

    placement = _try_place_cup(rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
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
        "mouth_abs": [(r0 + r, c0 + c) for (r, c) in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _place_connected_content(rng: np.random.Generator, canvas_in: np.ndarray, cups_meta: list[dict]) -> bool:
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        blob = _sample_connected_subset(rng, mouth_abs, count)
        if blob is None:
            return False
        for r, c in blob:
            canvas_in[r, c] = meta["content_color"]
        meta["content_blob"] = blob
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

    if not _place_connected_content(rng, canvas_in, cups_meta):
        return None

    _try_place_stray_shape(rng, canvas_in, canvas_out, cups_meta)

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
