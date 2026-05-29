# Combo variant of 3dc255db (index 18)
# combo_spec: D5/3+D6/0
# Name: 弹射线用边框色+内容变色保留

from __future__ import annotations

from collections import Counter
import math

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    Original task: a cup border of color B contains k content pixels of color C.
#    The output erases those k interior pixels to bg and ejects a length-k straight
#    line from the spike in color C. Example with tail up:
#      input:  B-frame with two C cells in the mouth
#      base:   those two mouth C cells disappear; two cells above the spike become C
#    Combo: the same k cells still control the ejection length, but both affected
#    regions are unified to the border color B:
#      combo:  the two mouth C cells stay in place but recolor to B, and the two
#      ejected cells above the spike also become B.
#    So the cup interior and the trajectory are both visibly "absorbed" into the
#    frame color, but one part stays inside the mouth while the other is newly added
#    outside the spike.
# b) Complexity calibration:
#    There is only one new color rule: "for each cup, changed pixels adopt that cup's
#    own border color". No global color lookup table is introduced. To keep this
#    visible, the generator avoids fully filling the mouth, keeps trajectory length >= 1,
#    and rejects samples whose combo output differs from the original output by too few
#    cells.
# c) Hidden assumptions reviewed against the original main():
#    - The original detection logic assumes cup identification happens from the input,
#      where mouth content is a non-border color. That still holds, so cup parsing can
#      remain input-driven.
#    - The original write phase assumed interior content should be erased to bg. That is
#      no longer valid; those cells must instead be recolored to the border color.
#    - The original trajectory write assumed the line color equals content color. That is
#      no longer valid; the line must use the same border color as the cup.
#    - The original solver did not need to distinguish "retained old pixels" from "new
#      trajectory pixels" once colors were chosen. The combo does: retained recolored
#      cells are inside the mouth region, while added trajectory cells are outside the
#      bbox along the tail axis. The generator keeps these regions disjoint and clear.
# d) Placement legality:
#    Cups still obey the original geometry: 1-3 cups, non-overlapping bboxes with a
#    one-cell gap, clear ejection paths, and optional isolated stray pixels. The output
#    first generator writes border, retained recolored mouth cells, and border-colored
#    trajectory cells only after checking that cups and paths remain disjoint.
# e) Visual observability:
#    A human sees each content color in the input only inside a cup, then sees that
#    color disappear from that cup in the output while the border color gains pixels in
#    two structurally linked places: inside the mouth and on the spike ray. The rule is
#    repeated across multiple cups and orientations.
# f) Visual cleanliness:
#    To avoid outputs degenerating into solid blobs, the generator caps mouth occupancy
#    near the original observed density ceiling and never fills the whole mouth. Stray
#    pixels, when present, reuse only border colors so they do not create fake surviving
#    content-color evidence outside cups.
# g) Color-change inferrability:
#    Content colors are always different from all border colors. For multi-cup samples,
#    content colors may repeat across different borders; that is intentional, because it
#    shows the output color depends on the containing frame rather than on a global
#    content-color mapping.
# h) New solver assumptions and generator guarantees:
#    The combo solver still assumes each real cup has an unambiguous spike candidate
#    whose opposite-side flood reaches exactly one dominant content color. The generator
#    preserves that original condition, keeps stray pixels outside cup bboxes and paths,
#    and rejects any sample where the combo output accidentally matches the legacy rule.


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_DENSITY_MAX = 0.45
MIN_VISIBLE_CHANGE_TOTAL = 4

STRAY_PROB = 0.4

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
    """4-connected flood within bbox, seeded from non-border cells on one bbox side."""
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


def _detect_cups(g: np.ndarray, bg: int) -> list[dict]:
    """Recover cups from the input using the original task's geometry cues."""
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

            passing: list[dict] = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                cnt = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != int(bg) and value != int(border_col):
                        cnt[value] += 1
                if not cnt:
                    continue
                content_col = int(cnt.most_common(1)[0][0])
                content_cells = sorted(
                    p for p in mouth if int(g[p[0], p[1]]) == int(content_col)
                )
                passing.append(
                    {
                        "tail_dir": tail_dir,
                        "spike": spike,
                        "content_col": content_col,
                        "content_cells": content_cells,
                        "bbox": bbox,
                    }
                )

            if len(passing) != 1:
                continue

            cup = passing[0].copy()
            cup["border_col"] = int(border_col)
            cups.append(cup)
    return cups


def _solve(grid: np.ndarray, *, keep_recolored_content: bool, line_uses_border: bool) -> np.ndarray:
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for cup in _detect_cups(g, bg):
        border_col = int(cup["border_col"])
        content_col = int(cup["content_col"])
        content_cells = list(cup["content_cells"])

        for r, c in content_cells:
            out[r, c] = int(border_col if keep_recolored_content else bg)

        line_col = int(border_col if line_uses_border else content_col)
        dr, dc = DIR4[str(cup["tail_dir"])]
        sr, sc = cup["spike"]
        for t in range(1, len(content_cells) + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            out[rr, cc] = line_col

    return out


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid, keep_recolored_content=False, line_uses_border=False)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Eject each cup's mouth content along the tail, but unify both affected regions to the
    cup's border color: retained mouth pixels recolor to the frame color, and the ejected
    line also uses that frame color.
    """
    return _solve(grid, keep_recolored_content=True, line_uses_border=True)


def fusion_metrics(grid: np.ndarray) -> dict[str, int]:
    """Quantify whether a sample visibly exercises the fused recolor+retain rule."""
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    retained = 0
    trajectory = 0
    cups = _detect_cups(g, bg)
    for cup in cups:
        count = len(cup["content_cells"])
        retained += count
        dr, dc = DIR4[str(cup["tail_dir"])]
        sr, sc = cup["spike"]
        visible = 0
        for t in range(1, count + 1):
            rr = sr + dr * t
            cc = sc + dc * t
            if not (0 <= rr < H and 0 <= cc < W):
                break
            visible += 1
        trajectory += visible
    variant = main(g)
    legacy = _legacy_main(g)
    return {
        "cups": int(len(cups)),
        "retained_recolored_pixels": int(retained),
        "trajectory_pixels": int(trajectory),
        "legacy_diff": int(np.sum(variant != legacy)),
    }


def _build_canonical_cup(L: int, R: int, hL: int, hR: int, color: int) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical funnel with tail pointing up."""
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
    """Flood-fill the cup interior from the open mouth side."""
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
        for dr, dc in DIR4.values():
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
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
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
    """Find a placement whose bbox is separated from other cups and whose tail path is clear."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

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
    """Cells on the mouth side of a cup bbox, limited to the tail-perpendicular span."""
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
    """Optional isolated stray pixel, using only a border color for visual clarity."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = int(meta["border_color"])

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["bbox"]
        forbidden[br0 : br1 + 1, bc0 : bc1 + 1] = True
        for r, c in cup["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["bbox"], meta["direction"]) & (~forbidden)
    candidates = np.argwhere(candidate_mask)
    if candidates.size == 0:
        return
    r, c = candidates[int(rng.integers(0, len(candidates)))]
    canvas_in[r, c] = stray_color
    canvas_out[r, c] = stray_color


def _assign_cup_colors(rng: np.random.Generator, num_cups: int) -> list[tuple[int, int]]:
    """Choose unique border colors and 1-2 content colors disjoint from all borders."""
    colors = list(range(1, 10))
    rng.shuffle(colors)
    border_colors = [int(colors.pop()) for _ in range(num_cups)]

    max_content_palette = min(2, len(colors))
    if num_cups == 1:
        palette_size = 1
    else:
        palette_size = int(rng.integers(1, max_content_palette + 1))
    content_palette = [int(colors.pop()) for _ in range(palette_size)]

    assignments = [content_palette[i % palette_size] for i in range(num_cups)]
    rng.shuffle(assignments)
    return list(zip(border_colors, assignments))


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
    """Build, place, and commit one cup under the fused output rule."""
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

    density_cap = max(1, int(math.ceil(len(mouth) * CONTENT_DENSITY_MAX)))
    content_count_max = min(CONTENT_COUNT_MAX, density_cap)
    if content_count_max < 1:
        return None
    content_count = int(rng.integers(1, content_count_max + 1))

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, content_count, occ_cup, occ_path
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    mouth_abs = [(r0 + r, c0 + c) for r, c in mouth]
    chosen = rng.choice(len(mouth_abs), size=content_count, replace=False)
    content_abs = [mouth_abs[int(i)] for i in chosen]

    patch_mask = patch != 0
    canvas_in[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    canvas_out[r0 : r0 + ph, c0 : c0 + pw][patch_mask] = border_color
    occ_cup[r0 : r0 + ph, c0 : c0 + pw] = True

    for r, c in content_abs:
        canvas_out[r, c] = border_color
    for r, c in ejection_cells:
        canvas_out[r, c] = border_color
    for r, c in path_cells:
        occ_path[r, c] = True

    return {
        "border_color": int(border_color),
        "content_color": int(content_color),
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "content_abs": content_abs,
        "content_count": int(content_count),
        "path_cells": path_cells,
        "ejection_cells": ejection_cells,
    }


def _scatter_content(canvas_in: np.ndarray, cups_meta: list[dict]) -> None:
    for meta in cups_meta:
        for r, c in meta["content_abs"]:
            canvas_in[r, c] = int(meta["content_color"])


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))
    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    occ_cup = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    cups_meta: list[dict] = []
    for border_color, content_color in _assign_cup_colors(rng, num_cups):
        meta = _add_cup(
            rng,
            H,
            W,
            border_color,
            content_color,
            canvas_in,
            canvas_out,
            occ_cup,
            occ_path,
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None

    visible_change_total = sum(len(meta["content_abs"]) + len(meta["ejection_cells"]) for meta in cups_meta)
    if visible_change_total < MIN_VISIBLE_CHANGE_TOTAL:
        return None

    _scatter_content(canvas_in, cups_meta)
    _try_place_stray(rng, canvas_in, canvas_out, cups_meta)

    try:
        produced = main(canvas_in)
        legacy = _legacy_main(canvas_in)
        metrics = fusion_metrics(canvas_in)
    except Exception:
        return None

    if not np.array_equal(produced, canvas_out):
        return None
    if np.array_equal(produced, legacy):
        return None
    if int(metrics["legacy_diff"]) < MIN_VISIBLE_CHANGE_TOTAL:
        return None
    if int(metrics["retained_recolored_pixels"]) < 1 or int(metrics["trajectory_pixels"]) < 1:
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
