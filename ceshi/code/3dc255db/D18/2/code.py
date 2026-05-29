# Variant of 3dc255db
# Dimension: D18 — 弹射起点 (path)
# Change: 弹射线从尖刺的下一个像素位置开始(尖刺本身保留为边框色)... → 间隔起跳

from __future__ import annotations

from collections import Counter

import numpy as np
from scipy import ndimage

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


# Design analysis for the D18 variant
# ----------------------------------
# variant_spec:
#   core_change:
#     old_rule: content line starts immediately after the spike, at distance 1.
#     new_rule: content line starts only after skipping N empty cells beyond the spike.
#     encoding: N = abs(left_arm_len - right_arm_len), measured on the two diagonal arms
#       that meet at the spike. This keeps the rule visible in the cup geometry itself.
#   concrete_visual_behavior:
#     input:
#       - same cup family as the original generator: hollow Y/funnel frames with a single-pixel
#         spike on one bbox edge and content scattered in the mouth interior.
#       - unlike the original generator, every cup is forced to be arm-asymmetric so N > 0.
#       - example in canonical tail=U orientation:
#         L=3, R=1 => N=2. If the content count is 4, the output leaves the first two cells
#         above the spike empty and begins the emitted line at the third cell above the spike.
#     output:
#       - mouth content is erased as before.
#       - the emitted line has the same color and length logic as before, but its first visible
#         pixel is at spike + (N + 1) * tail_dir, not spike + 1 * tail_dir.
#   behavior_trace:
#     original:
#       - spike found -> erase content -> paint t=1..count from spike outward.
#     variant:
#       - spike found -> measure both diagonal arm lengths at that spike -> N=abs(diff)
#       - erase content -> paint t=(N+1)..(N+count) from spike outward.
#     observable_difference:
#       - each cup now leaves a small bg runway between spike and emitted line.
#       - longer arm imbalance means a longer runway.
#   complexity_calibration:
#     - N is restricted to {1, 2}. This is enough to create visible subcases without making
#       the inference too sparse or too dependent on many discrete values.
#     - generator intentionally mixes cups with both imbalances across samples by sampling the
#       target gap first, then choosing an arm pair that realizes it.
#   hidden_assumptions_review:
#     - original main() assumes the ejection start offset is always 1 in:
#         for t in range(1, len(content_cells) + 1)
#       That line is no longer valid once the offset varies per cup.
#     - original placement logic assumes any clear path from spike to edge is enough.
#       In the variant it must reserve an additional leading gap segment before emitted pixels.
#     - new main() adds one structural assumption: the spike has exactly two diagonal border
#       arms whose lengths can be counted by continuing straight along each diagonal. The
#       generator preserves this by staying within the original canonical Y/funnel family.
#   placement_legality:
#     - generator reserves the entire ray from spike+1 to the edge in occ_path, including the
#       skipped gap cells, so no other cup or stray pixel can occupy the runway.
#     - placement requires available_run > N, guaranteeing at least one emitted pixel remains
#       visible after the skipped segment.
#   visual_observability:
#     - the rule is visible from shape alone: asymmetric cups produce larger gaps.
#     - because the cups are hollow and the path is kept empty in the input, the blank runway
#       in the output is clean and easy to spot.
#   visual_cleanliness:
#     - cups still keep a 1-cell bbox gap from each other.
#     - stray pixels remain on the mouth side only and are forbidden from path cells, so they
#       do not clutter or mask the skipped runway.
#   color_inference:
#     - no new color mapping is introduced. Colors keep the original role separation:
#       border color identifies the cup, content color is simply relocated outward.


def _flood_from_bbox_side(border_set: set[tuple[int, int]], bbox: tuple[int, int, int, int], side: str) -> set[tuple[int, int]]:
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


def _gap_len_from_spike_arms(border_set: set[tuple[int, int]], spike: tuple[int, int]) -> int:
    """Measure the two diagonal arm lengths that meet at the spike and return their
    absolute difference. The generator guarantees a canonical Y/funnel family where the
    spike has exactly two such diagonal neighbors."""
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
    Eject content pixels from cup-shaped frames outward along each cup's tail direction.

    Variant D18:
    - The emitted line no longer starts immediately after the spike.
    - First infer N from the cup geometry: N = abs(diagonal arm lengths at the spike).
    - Then leave N empty bg cells beyond the spike, and start the emitted line after them.
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
                passing.append((d, spike, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            gap_len = _gap_len_from_spike_arms(border_set, spike)
            dr, dc = DIR4[tail_dir]
            sr, sc = spike
            for t in range(gap_len + 1, gap_len + len(content_cells) + 1):
                rr = sr + dr * t
                cc = sc + dc * t
                if not (0 <= rr < H and 0 <= cc < W):
                    break
                out[rr, cc] = content_col

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)

# After rotation, bbox stays within the original 3..8 x 3..7 envelope.
CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)
GAP_LEN_CHOICES = (1, 2)

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
    """Flood-fill the interior of the cup patch from cells on the given bbox edge, stopping at
    the border. Returns the sorted list of interior coordinates (seeds included)."""
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
        return np.flipud(patch).copy(), (h - 1 - sr, sc), [(h - 1 - r, c) for r, c in mouth]
    if direction == "L":
        return np.rot90(patch, k=1).copy(), (w - 1 - sc, sr), [(w - 1 - c, r) for r, c in mouth]
    if direction == "R":
        return np.rot90(patch, k=-1).copy(), (sc, h - 1 - sr), [(c, h - 1 - r) for r, c in mouth]
    raise ValueError(direction)


def _sample_cup_shape(
    rng: np.random.Generator, direction: str, target_gap: int,
) -> tuple[int, int, int, int] | None:
    """Pick canonical cup parameters with a prescribed arm-length difference."""
    if direction in ("U", "D"):
        max_h_canon, max_w_canon = CUP_BBOX_MAX_UD
    else:
        max_h_canon, max_w_canon = CUP_BBOX_MAX_LR

    lo, hi = ARM_LEN_RANGE
    arm_pairs = [
        (L_arm, R_arm)
        for L_arm in range(lo, hi + 1)
        for R_arm in range(lo, hi + 1)
        if abs(L_arm - R_arm) == target_gap and (L_arm + R_arm + 1) <= max_w_canon
    ]
    if not arm_pairs:
        return None

    rng.shuffle(arm_pairs)
    wlo, whi = WALL_EXT_RANGE
    for L_arm, R_arm in arm_pairs:
        wall_pairs = [
            (hL, hR)
            for hL in range(wlo, whi + 1)
            for hR in range(wlo, whi + 1)
            if (hL + hR) >= 1 and max(L_arm + hL, R_arm + hR) + 1 <= max_h_canon
        ]
        if not wall_pairs:
            continue
        idx = int(rng.integers(0, len(wall_pairs)))
        hL, hR = wall_pairs[idx]
        return L_arm, R_arm, hL, hR
    return None


def _try_place_cup(
    rng: np.random.Generator, H: int, W: int, patch: np.ndarray, spike: tuple[int, int],
    direction: str, content_count: int, gap_len: int, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place a cup so its bbox keeps a 1-cell gap and its full ray from spike to edge is clear.

    Returns (r0, c0, full_path_cells, ejection_cells). The full path includes the skipped runway.
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
        if len(path_cells) <= gap_len:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_len = min(content_count, len(path_cells) - gap_len)
        if eject_len <= 0:
            continue
        return r0, c0, path_cells, path_cells[gap_len:gap_len + eject_len]
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Boolean (H, W) mask of cells on the mouth side of ``bbox`` (opposite the tail direction),
    restricted to the cup's perpendicular extent."""
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
    """With probability ``STRAY_PROB``, place an isolated pixel on the mouth side of one cup."""
    if not cups_meta or rng.random() >= STRAY_PROB:
        return
    H, W = canvas_in.shape
    meta = cups_meta[int(rng.integers(0, len(cups_meta)))]
    stray_color = meta["border_color"] if rng.random() < STRAY_USE_BORDER_PROB else meta["content_color"]

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
    rng: np.random.Generator, H: int, W: int, border_color: int, content_color: int,
    canvas_in: np.ndarray, canvas_out: np.ndarray, occ_cup: np.ndarray, occ_path: np.ndarray,
) -> dict | None:
    """Build one cup and commit its border plus delayed ejection line to the canvases."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    gap_len = int(rng.choice(GAP_LEN_CHOICES))
    shape = _sample_cup_shape(rng, direction, gap_len)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < 3:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_cup(
        rng, H, W, patch, spike, direction, content_count, gap_len, occ_cup, occ_path,
    )
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

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
        "gap_len": gap_len,
        "arm_lengths": (L_arm, R_arm),
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
    """Single end-to-end generation attempt."""
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
            rng, H, W, border_color, content_color, canvas_in, canvas_out, occ_cup, occ_path,
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
