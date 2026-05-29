# Variant of 3dc255db
# Dimension: D15 — 边框颜色唯一性 (meta)
# Change: 每个杯的边框颜色唯一(不同杯颜色不同)... → 边框色与内容色相同
#
# Design analysis for this variant
# a) Concretization and behavior trace
#    - Original inputs use two colors per cup: one for the Y/funnel border and one for the
#      scattered pixels inside the mouth. The solver identifies the border component by color,
#      floods the mouth from the side opposite the spike, counts the differently colored pixels,
#      erases them, and ejects them as a straight line from the spike.
#    - In D15, each cup uses one color for both frame and interior scatter. The visual cue is no
#      longer "different color", but "same-color sparse frame plus detached inner dots".
#    - Small trace: suppose a right-pointing cup of color 4 has three detached color-4 dots in
#      its mouth. The original solver would inspect the color-4 border component, flood the mouth,
#      and look for non-4 colored cells; it would find none and do nothing. The variant solver
#      instead treats only the large 8-connected component as border, floods through the mouth,
#      and counts same-color cells that are inside the mouth but not part of that border component.
#      Those cells are erased and re-drawn as a length-3 ray from the spike.
# b) Complexity calibration
#    - If same-color content touches the border, border and content collapse into one 8-connected
#      component and the distinction becomes ambiguous. To keep the D15 change visible and
#      solvable, the generator places content only at mouth cells that are not 8-neighbors of the
#      border and are mutually 8-separated. This preserves the "frame vs detached inner dots"
#      visual contrast without changing the transformation rule itself.
#    - Different cups still keep distinct colors. This is a deliberate non-D15 simplification so
#      the only new burden is separating border from content spatially, not disentangling multiple
#      cups of one color.
# c) Implicit-assumption audit of the original main()
#    - Looping by `border_col` is still valid because cups remain color-distinct.
#    - `components_mask(g == border_col, connectivity=8)` used to return only border pieces of
#      that color. In this variant it also sees same-color content dots, so the solver must ignore
#      tiny components and only treat large components as cup borders.
#    - `if v != bg and v != border_col` is invalid now: content has `v == border_col`. Replace it
#      with "same color inside mouth but outside the chosen border component".
#    - Unconditional output painting remains safe because the ejection path is still guaranteed to
#      be background in the input.
# d) Placement legality
#    - Cup bboxes still have a 1-cell gap and paths still stay clear to the edge.
#    - Detached content dots are selected before committing the cup, so their count matches the
#      ejection length exactly and they never merge with the frame or with each other.
# e) Visual observability
#    - Across 2-4 examples, a human can see that the inner dots have the same color as the frame
#      and nevertheless get removed while a same-color line appears at the spike side. The spatial
#      role difference is therefore observable directly from the pixels.
# f) Visual cleanliness
#    - Isolated mouth dots keep the interior legible. Strays remain outside all cup bboxes and
#      outside all ejection paths, so they do not visually overlap with the cup logic.
# g) Color-rule inferability
#    - The color change rule is simple: there is no recoloring. The same cup color moves from the
#      mouth interior to the tail ray. Keeping one color per cup and at most three cups prevents
#      the scene from becoming noisy.
# h) Variant verification target
#    - A meaningful metric is how often the original solver and this variant solver disagree on
#      generated inputs. Because the original solver requires border/content color contrast, valid
#      D15 samples should usually change its behavior.
# i) New main() assumptions and generator guarantees
#    - Assumption: each true cup border is an 8-connected component of size >= 5 with a unique
#      spike-edge interpretation after the mouth test. Generator guarantee: it only builds such
#      components.
#    - Assumption: same-color content inside a mouth is not 8-connected to the frame. Generator
#      guarantee: content cells are chosen from border-distant mouth cells and are mutually
#      8-separated.
#    - Assumption: any extra same-color stray pixel lies outside the cup bbox/path. Generator
#      guarantee: stray placement uses the same forbidden-mask logic as the original task.

from __future__ import annotations

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Variant rule: each cup uses one color for both frame and content.

    We still detect the cup by its large 8-connected frame component. Content pixels are the
    same-color cells lying inside the mouth flood but not belonging to that frame component.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    dir4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
    opposite = {"U": "D", "D": "U", "L": "R", "R": "L"}

    for cup_color in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(cup_color), connectivity=8):
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

            passing: list[tuple[str, tuple[int, int], list[tuple[int, int]]]] = []
            for d, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, opposite[d])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == int(cup_color)]
                if not content_cells:
                    continue
                passing.append((d, spike, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            dr, dc = dir4[tail_dir]
            sr, sc = spike
            for t in range(1, len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = int(cup_color)

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
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
_STRUCT8 = np.ones((3, 3), dtype=bool)


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
    """Flood-fill the cup interior from the mouth side, stopping at border cells."""
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
    """Rotate a canonical (tail=U) patch, its spike, and mouth coords to the target tail direction."""
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
    """Sample a canonical cup shape that still fits the bbox limits after rotation."""
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


def _best_isolated_subset(rng: np.random.Generator, cells: list[tuple[int, int]], tries: int = 8) -> list[tuple[int, int]]:
    """Greedy-search for a large subset with pairwise Chebyshev distance >= 2."""
    best: list[tuple[int, int]] = []
    for _ in range(tries):
        order = list(cells)
        rng.shuffle(order)
        chosen: list[tuple[int, int]] = []
        for r, c in order:
            if all(max(abs(r - rr), abs(c - cc)) > 1 for rr, cc in chosen):
                chosen.append((r, c))
        if len(chosen) > len(best):
            best = chosen
    return best


def _safe_mouth_coords(patch: np.ndarray, border_color: int, mouth: list[tuple[int, int]]) -> list[tuple[int, int]]:
    """Keep mouth cells that are not 8-neighbors of the border, so same-color dots stay detached."""
    border_mask = patch == border_color
    dilated = ndimage.binary_dilation(border_mask, structure=_STRUCT8)
    return [(r, c) for (r, c) in mouth if not dilated[r, c]]


def _try_place_cup(
    rng: np.random.Generator, H: int, W: int, patch: np.ndarray, spike: tuple[int, int],
    direction: str, content_count: int, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a placement with a 1-cell bbox gap and a clear ejection path."""
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
        path_cells: list[tuple[int, int]] = []
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
    """Cells on the mouth side of a cup bbox, restricted to the cup's perpendicular extent."""
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
    """Optionally place an isolated same-color stray pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["cup_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
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
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _add_cup(
    rng: np.random.Generator, H: int, W: int, cup_color: int,
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup, choose detached same-color content dots, and commit the output state."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, cup_color)
    mouth_canon = _mouth_interior_coords(patch_canon, cup_color, opposite_side="D")
    if not mouth_canon:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    safe_mouth = _safe_mouth_coords(patch, cup_color, mouth)
    isolated_pool = _best_isolated_subset(rng, safe_mouth)
    if not isolated_pool:
        return None

    max_content = min(CONTENT_COUNT_MAX, len(isolated_pool))
    min_content = 2 if max_content >= 2 else 1
    content_count = int(rng.integers(min_content, max_content + 1))
    chosen_idxs = rng.choice(len(isolated_pool), size=content_count, replace=False)
    content_local = [isolated_pool[int(i)] for i in chosen_idxs]

    ph, pw = patch.shape
    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = cup_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = cup_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True
    for r, c in ejection_cells:
        canvas_out[r, c] = cup_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "cup_color": cup_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_abs": [(r0 + r, c0 + c) for (r, c) in content_local],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(canvas_in: np.ndarray, cups_meta: list[dict]) -> None:
    """Place each cup's detached same-color content dots."""
    for meta in cups_meta:
        for r, c in meta["content_abs"]:
            canvas_in[r, c] = meta["cup_color"]


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Generate one valid D15 input grid and verify it with the solver."""
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
        if not palette:
            return None
        cup_color = int(palette.pop())
        meta = _add_cup(
            rng, H, W, cup_color, canvas_in, canvas_out, occ_cup, occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None

    _scatter_content(canvas_in, cups_meta)
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
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
