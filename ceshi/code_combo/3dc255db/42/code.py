# Combo variant of 3dc255db (index 42)
# combo_spec: D10/1+D18/2
# Name: 边框数量决定长度+间隔起跳

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


# Design analysis for combo D10/1 + D18/2
# ----------------------------------------
# combo_rule:
#   - Keep the original cup/frame detection logic: each actionable object is a hollow Y/funnel
#     frame with a single-pixel spike on one bbox edge and scattered content pixels in the mouth.
#   - The emitted line now depends on two different structural measurements of the same cup:
#       1. gap_len = abs(left_arm_len - right_arm_len), measured from the two diagonals meeting
#          at the spike. This is the D18/2 "skip N cells before starting" rule.
#       2. eject_len = number of border pixels in the cup component. This is the D10/1 rule,
#          replacing the original "content count determines length".
#
# concrete_example:
#   - Canonical tail=U cup with arm lengths L=3, R=1 and wall extensions hL=2, hR=1.
#   - border pixel count = 1 + L + R + hL + hR = 8, so the output line wants length 8.
#   - gap_len = abs(3 - 1) = 2, so the first two cells above the spike stay bg.
#   - Suppose the input mouth contains only 3 content pixels.
#   - Original task:
#       erase 3 mouth pixels -> paint 3 pixels immediately above the spike.
#   - D18 only:
#       erase 3 mouth pixels -> leave 2-cell runway -> paint 3 pixels after the runway.
#   - This combo:
#       erase 3 mouth pixels -> leave 2-cell runway -> paint 8 pixels after the runway
#       (cropped by the grid edge if needed).
#
# behavior_trace:
#   - Original main() used len(content_cells) for emitted length and assumed start offset=1.
#   - Combo main() must still use content cells only to identify the moving color and what to
#     erase, but the visible line length comes from len(border_set), while the start offset comes
#     from spike-arm asymmetry.
#
# complexity_calibration:
#   - gap_len is intentionally limited to {1, 2}. Zero would visually collapse back toward the
#     original start position, and larger values become too sparse on 10..13 grids.
#   - content_count is capped at 3 in the generator. This is deliberate: it keeps a visible gap
#     between "how many pixels were inside" and "how long the emitted ray becomes", making D10/1
#     observable from a few examples instead of being drowned out by accidental equality.
#   - wall extensions are capped more tightly than the seed generator so border counts stay in a
#     moderate range and are often fully visible before hitting the canvas edge.
#
# hidden_assumptions_review:
#   - In main(), candidate-tail detection still assumes the true spike is a non-corner bbox-edge
#     singleton. The generator preserves that exact geometry.
#   - Mouth discovery still assumes flooding from the side opposite the tail reaches the content
#     region and no other unrelated non-bg pixels inside the bbox dominate the color count. The
#     generator keeps stray pixels outside all cup bboxes.
#   - The new gap-length helper assumes the spike has exactly two diagonal border arms that can be
#     followed straight. The generator therefore stays in the canonical Y/funnel family.
#   - The new length rule assumes border pixels are counted on a single connected cup component.
#     Unique border colors per cup preserve that.
#
# placement_legality:
#   - The generator reserves the whole spike-to-edge ray in occ_path, including the skipped runway,
#     so no other cup or stray pixel can block the emitted line.
#   - Placement requires at least one empty cell after the gap, and more strongly requires the
#     combo output to be longer than the input content count. That prevents D10/1 from becoming
#     visually inert on a sample.
#
# visual_observability:
#   - The gap is visible because asymmetric cups leave 1 or 2 blank cells directly beyond the tip.
#   - The long emitted ray is visible because the mouth usually holds only 1..3 pixels while the
#     border has 5+ pixels. A solver can see that line length is not tracking the number of moved
#     mouth pixels.
#   - Because the two controls come from different geometric cues (arm imbalance vs full border
#     size), the combo is not reducible to a single local measurement around the spike.
#
# visual_cleanliness:
#   - Cups keep a 1-cell bbox gap.
#   - Stray pixels remain only on mouth sides and are forbidden from any reserved path cells.
#   - Moderate wall lengths avoid extremely tall frames that would crowd the small grids.
#
# generator_guarantees_for_new_main:
#   - Every generated cup has gap_len in {1, 2}.
#   - Every generated cup has border_count >= 5.
#   - Every generated cup gets a path where visible emitted length is at least 3 cells and is
#     strictly longer than the content count, so both halves of the combo are actually testable.


# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 2)
GAP_LEN_CHOICES = (1, 2)

MIN_MOUTH_CELLS = 3
CONTENT_COUNT_MAX = 3
MIN_BORDER_PIXELS = 5
MIN_VISIBLE_EJECT_LEN = 3

STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5

GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300
# ==================================

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
_STRUCT8 = np.ones((3, 3), dtype=bool)


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
    """4-connected flood inside bbox, starting from the requested bbox side."""
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


def _gap_len_from_spike_arms(border_set: set[tuple[int, int]], spike: tuple[int, int]) -> int:
    """Return the absolute difference between the two diagonal arm lengths at the spike."""
    sr, sc = spike
    diag_dirs = []
    for dr in (-1, 1):
        for dc in (-1, 1):
            if (sr + dr, sc + dc) in border_set:
                diag_dirs.append((dr, dc))

    if len(diag_dirs) != 2:
        return 0

    lengths = []
    for dr, dc in diag_dirs:
        k = 1
        while (sr + dr * k, sc + dc * k) in border_set:
            k += 1
        lengths.append(k - 1)
    return abs(lengths[0] - lengths[1])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - erase the mouth content as in the original task,
    - skip N cells past the spike, where N is the diagonal-arm length difference,
    - emit a straight ray whose length equals the cup border's pixel count.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        for border_coords in components_mask(g == int(border_col), connectivity=8):
            if border_coords.size == 0 or int(border_coords.shape[0]) < MIN_BORDER_PIXELS:
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
                (direction, edge_cells[0])
                for direction, edge_cells in edges
                if len(edge_cells) == 1 and edge_cells[0] not in corners
            ]
            if not candidates:
                continue

            passing = []
            for tail_dir, spike in candidates:
                mouth = _flood_from_bbox_side(border_set, bbox, OPPOSITE[tail_dir])
                counts = Counter()
                for r, c in mouth:
                    value = int(g[r, c])
                    if value != bg and value != int(border_col):
                        counts[value] += 1
                if not counts:
                    continue
                content_col = int(counts.most_common(1)[0][0])
                content_cells = [p for p in mouth if int(g[p[0], p[1]]) == content_col]
                passing.append((tail_dir, spike, content_col, content_cells))

            if len(passing) != 1:
                continue

            tail_dir, spike, content_col, content_cells = passing[0]
            for r, c in content_cells:
                out[r, c] = bg

            gap_len = _gap_len_from_spike_arms(border_set, spike)
            eject_len = len(border_set)
            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            for t in range(gap_len + 1, gap_len + eject_len + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


def _build_canonical_cup(
    L: int,
    R: int,
    hL: int,
    hR: int,
    color: int,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Build a canonical tail-up Y/funnel cup."""
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


def _mouth_interior_coords(
    patch: np.ndarray,
    border_color: int,
    opposite_side: str,
) -> list[tuple[int, int]]:
    """Return flood-reached mouth interior coordinates inside a canonical patch."""
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
    """Rotate a canonical tail-up cup to the requested tail direction."""
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


def _sample_cup_shape(
    rng: np.random.Generator,
    direction: str,
    target_gap: int,
) -> tuple[int, int, int, int] | None:
    """Choose canonical cup parameters with the requested arm-difference gap."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    wlo, whi = WALL_EXT_RANGE
    shapes = []
    for L_arm in range(lo, hi + 1):
        for R_arm in range(lo, hi + 1):
            if abs(L_arm - R_arm) != target_gap:
                continue
            if (L_arm + R_arm + 1) > max_w_canon:
                continue
            for hL in range(wlo, whi + 1):
                for hR in range(wlo, whi + 1):
                    if (hL + hR) < 1:
                        continue
                    if max(L_arm + hL, R_arm + hR) + 1 > max_h_canon:
                        continue
                    border_count = 1 + L_arm + R_arm + hL + hR
                    if border_count < MIN_BORDER_PIXELS:
                        continue
                    shapes.append((L_arm, R_arm, hL, hR))
    if not shapes:
        return None
    return shapes[int(rng.integers(0, len(shapes)))]


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
    content_count: int,
    gap_len: int,
    border_count: int,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup so its bbox and full reserved ray remain clear."""
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    offsets = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(offsets)

    for r0, c0 in offsets:
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
        if len(path_cells) <= gap_len:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        visible_eject_len = min(border_count, len(path_cells) - gap_len)
        if visible_eject_len < MIN_VISIBLE_EJECT_LEN:
            continue
        if visible_eject_len <= content_count:
            continue

        ejection_cells = path_cells[gap_len:gap_len + visible_eject_len]
        return r0, c0, path_cells, ejection_cells
    return None


def _stray_candidate_mask(
    H: int,
    W: int,
    bbox: tuple[int, int, int, int],
    tail_dir: str,
) -> np.ndarray:
    """Mask of cells on the mouth side of a cup bbox."""
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
    """Optionally place one isolated stray pixel on the mouth side of some cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return

    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

    forbidden = ndimage.binary_dilation(canvas_in != 0, structure=_STRUCT8)
    for cup_meta in cups_meta:
        br0, bc0, br1, bc1 = cup_meta["bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in cup_meta["path_cells"]:
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
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> dict | None:
    """Build one combo cup and commit its border plus emitted line."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    gap_len = int(rng.choice(GAP_LEN_CHOICES))
    shape = _sample_cup_shape(rng, direction, gap_len)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    border_count = int(np.count_nonzero(patch_canon))
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < MIN_MOUTH_CELLS:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape

    max_content = min(CONTENT_COUNT_MAX, len(mouth), border_count - 1)
    if max_content < 1:
        return None
    content_count = int(rng.integers(1, max_content + 1))

    placement = _try_place_cup(
        rng,
        H,
        W,
        patch,
        spike,
        direction,
        content_count,
        gap_len,
        border_count,
        occ_cup,
        occ_path,
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = patch != 0
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

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
        "gap_len": gap_len,
        "border_count": border_count,
        "visible_eject_len": len(ejection_cells),
        "arm_lengths": (L_arm, R_arm),
    }


def _scatter_content(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    cups_meta: list[dict],
) -> bool:
    """Scatter each cup's content color into its mouth interior."""
    for meta in cups_meta:
        mouth_abs = meta["mouth_abs"]
        count = meta["content_count"]
        if len(mouth_abs) < count:
            return False
        idxs = rng.choice(len(mouth_abs), size=count, replace=False)
        for idx in idxs:
            r, c = mouth_abs[int(idx)]
            canvas_in[r, c] = meta["content_color"]
    return True


def _gen_attempt(rng: np.random.Generator, num_cups: int | None = None) -> np.ndarray | None:
    """Output-first generation, validated against the combo solver."""
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_in = np.zeros((H, W), dtype=int)
    canvas_out = np.zeros((H, W), dtype=int)

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
    raise RuntimeError(
        f"generate_input: failed after {GEN_RETRIES_PRIMARY + GEN_RETRIES_FALLBACK} attempts"
    )
