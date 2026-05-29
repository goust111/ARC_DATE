# Variant of 3dc255db
# Dimension: D14 — 弹射方向与杯口关系 (path)
# Change: 弹射方向指向尾部(远离杯口)——内容从杯口'倒出'到尾部方向... → 双向弹射
#
# Design analysis for this variant
# a) Concrete behavior and trace
#    - The cup border geometry stays the same: one single-pixel spike on one bbox edge, and the
#      mouth opening on the opposite bbox edge.
#    - Input still shows content pixels scattered only inside the mouth interior.
#    - Output no longer turns all content into one tail-side line. Instead:
#        tail_count = ceil(n / 2)
#        mouth_count = floor(n / 2)
#      where n is the number of content pixels found in the mouth interior.
#    - tail_count pixels form the same outward line from the spike in the tail direction.
#    - mouth_count pixels form a second outward line on the mouth side. Its launch point is the
#      mouth-edge cell directly opposite the spike on the same axis; the drawn line starts one
#      step outside the bbox from that mouth-edge cell.
#    - Example, tail=U and 5 content pixels:
#        original: erase 5 interior pixels, draw 5 upward from the spike
#        variant : erase 5 interior pixels, draw 3 upward from the spike and 2 downward from
#                  the mouth opening's axis line
#      The difference is visible because one output now has two separated rays instead of one.
#
# b) Complexity calibration
#    - The rule uses only two branches (tail-side and mouth-side) and a simple split rule with
#      one asymmetry: odd counts favor the tail by 1.
#    - Generator keeps varied cup orientations and shapes, but requires at least one visible cell
#      on both sides so the bidirectional effect is usually observable.
#    - Content counts are sampled from 2..7 when feasible, which avoids too many degenerate
#      1-pixel cases that would collapse back to the original behavior.
#
# c) Old implicit assumptions that break under the variant
#    - In the original solver, `len(content_cells)` was used as the full tail-line length. That
#      assumes every erased content pixel goes to one side. This is no longer valid.
#    - The original generator reserved only the tail path as guaranteed background. That is not
#      enough now because the mouth-side outward line also needs clear space.
#    - The original output builder wrote only one ejection line. The variant needs two launch
#      anchors: the tail spike and a deterministic mouth-side anchor.
#
# d) Placement legality
#    - Each cup still needs a 1-cell bbox gap from other cups.
#    - Generator now reserves both full outward rays (tail side and mouth side) so later cups and
#      stray pixels cannot block them.
#    - The mouth-side anchor is guaranteed to be an actual open mouth-edge cell by construction of
#      the canonical cup and by keeping the opposite-spike projection inside the flooded mouth set.
#
# e) Visual observability
#    - A human can compare several pairs and see that content no longer exits only through the
#      spike. There is always a second line leaving from the opposite side when n >= 2.
#    - Odd/even counts are both sampled, so examples can show the "tail gets the extra pixel"
#      rule rather than a fixed 50/50 split only.
#
# f) Visual cleanliness
#    - The cup border is unchanged, so object recognition remains stable.
#    - Stray pixels still exist, but they are forbidden from both reserved rays, preventing the
#      new extra mouth-side line from being visually tangled with unrelated noise.
#
# g) Color inferability
#    - No new color mapping is introduced. Both rays use the same single content color already
#      associated with that cup.
#
# Audit of new solver assumptions
# - `main()` assumes the mouth-side launch anchor can be derived from the spike axis and the mouth
#   flood region. The generator guarantees this.
# - `main()` still assumes exactly one candidate tail direction survives the content-bearing flood
#   test. The generator keeps the original single-tail construction.
# - `main()` assumes every acted-on cup has one dominant content color in its mouth interior. The
#   generator still uses one content color per cup.

from __future__ import annotations

from collections import Counter

import numpy as np
from arc_utils import bbox_coords, colors_present, components_mask, get_bg
from scipy import ndimage


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


def _split_content_count(n: int) -> tuple[int, int]:
    """Return (tail_count, mouth_count); odd counts favor the tail."""
    return (n + 1) // 2, n // 2


def _mouth_anchor_from_spike(
    spike: tuple[int, int],
    bbox: tuple[int, int, int, int],
    tail_dir: str,
    mouth_cells: set[tuple[int, int]],
) -> tuple[int, int] | None:
    """Find the mouth-edge cell opposite the spike, preferring exact axial alignment."""
    sr, sc = spike
    r0, c0, r1, c1 = bbox
    if tail_dir == "U":
        preferred = (r1, sc)
        edge_cells = [p for p in mouth_cells if p[0] == r1]
        key = lambda p: (abs(p[1] - sc), p[1])
    elif tail_dir == "D":
        preferred = (r0, sc)
        edge_cells = [p for p in mouth_cells if p[0] == r0]
        key = lambda p: (abs(p[1] - sc), p[1])
    elif tail_dir == "L":
        preferred = (sr, c1)
        edge_cells = [p for p in mouth_cells if p[1] == c1]
        key = lambda p: (abs(p[0] - sr), p[0])
    else:  # "R"
        preferred = (sr, c0)
        edge_cells = [p for p in mouth_cells if p[1] == c0]
        key = lambda p: (abs(p[0] - sr), p[0])

    if preferred in mouth_cells:
        return preferred
    if not edge_cells:
        return None
    return min(edge_cells, key=key)


def _ray_cells_from_anchor(anchor: tuple[int, int], direction: str, H: int, W: int) -> list[tuple[int, int]]:
    """Cells strictly outside the anchor, continuing to the grid edge."""
    dr, dc = DIR4[direction]
    rr, cc = anchor[0] + dr, anchor[1] + dc
    out: list[tuple[int, int]] = []
    while 0 <= rr < H and 0 <= cc < W:
        out.append((rr, cc))
        rr += dr
        cc += dc
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Bidirectional variant of 3dc255db.

    For each detected cup:
    - detect the tail spike as in the original task
    - find the mouth interior as the flooded region from the opposite side
    - erase all content pixels of the cup's single content color
    - split their count into:
        tail_count = ceil(n/2)
        mouth_count = floor(n/2)
    - draw tail_count pixels outward from the spike in the tail direction
    - draw mouth_count pixels outward from the mouth opening on the same axis, away from the cup
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

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
                mouth_anchor = _mouth_anchor_from_spike(spike, bbox, d, mouth)
                if mouth_anchor is None:
                    continue
                passing.append((d, spike, mouth_anchor, content_col, content_cells))

            if len(passing) != 1:
                continue
            tail_dir, spike, mouth_anchor, content_col, content_cells = passing[0]

            for r, c in content_cells:
                out[r, c] = bg

            tail_count, mouth_count = _split_content_count(len(content_cells))
            tail_cells = _ray_cells_from_anchor(spike, tail_dir, H, W)
            mouth_cells = _ray_cells_from_anchor(mouth_anchor, OPPOSITE[tail_dir], H, W)

            for rr, cc in tail_cells[:tail_count]:
                out[rr, cc] = content_col
            for rr, cc in mouth_cells[:mouth_count]:
                out[rr, cc] = content_col

    return out


# --------------- generator ---------------

# ==== Generator configuration ====
GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)

NUM_CUPS_CHOICES = (1, 2, 3)

CUP_BBOX_MAX_UD = (8, 7)
CUP_BBOX_MAX_LR = (7, 8)

ARM_LEN_RANGE = (1, 3)
WALL_EXT_RANGE = (0, 4)

CONTENT_COUNT_MAX = 7
CONTENT_COUNT_MIN_VISIBLE = 2

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
    """Flood-fill the interior of the cup patch from the given bbox edge."""
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
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth: list[tuple[int, int]],
    direction: str,
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
    """Pick (L, R, hL, hR) parameters for a canonical cup that fit after rotation."""
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


def _max_split_count_without_overflow(tail_len: int, mouth_len: int, count_limit: int) -> int:
    """Largest n whose split fits on both reserved rays."""
    for n in range(count_limit, 0, -1):
        tail_count, mouth_count = _split_content_count(n)
        if tail_count <= tail_len and mouth_count <= mouth_len:
            return n
    return 0


def _try_place_cup(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    mouth_anchor_rel: tuple[int, int],
    direction: str,
    occ_cup: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Find a valid placement whose tail ray and mouth ray are both clear."""
    ph, pw = patch.shape
    occ_cup_dilated = ndimage.binary_dilation(occ_cup, structure=_STRUCT8)

    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    mouth_dir = {"U": "D", "D": "U", "L": "R", "R": "L"}[direction]

    for r0, c0 in coords_list:
        if occ_cup_dilated[r0:r0 + ph, c0:c0 + pw].any():
            continue
        if occ_path[r0:r0 + ph, c0:c0 + pw].any():
            continue

        spike_abs = (r0 + spike[0], c0 + spike[1])
        mouth_anchor_abs = (r0 + mouth_anchor_rel[0], c0 + mouth_anchor_rel[1])
        tail_path = _ray_cells_from_anchor(spike_abs, direction, H, W)
        mouth_path = _ray_cells_from_anchor(mouth_anchor_abs, mouth_dir, H, W)
        if not tail_path or not mouth_path:
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in tail_path):
            continue
        if any(occ_cup[r, c] or occ_path[r, c] for r, c in mouth_path):
            continue
        return r0, c0, tail_path, mouth_path
    return None


def _stray_candidate_mask(H: int, W: int, bbox: tuple[int, int, int, int], tail_dir: str) -> np.ndarray:
    """Cells on the mouth side of the cup bbox, within the perpendicular span."""
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
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    canvas_out: np.ndarray,
    cups_meta: list[dict],
) -> None:
    """Optionally place one isolated pixel on the mouth side of one cup."""
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
    """Build one cup, place it, and commit its two output rays."""
    direction = str(rng.choice(["U", "D", "L", "R"]))
    shape = _sample_cup_shape(rng, direction)
    if shape is None:
        return None
    L_arm, R_arm, hL, hR = shape

    patch_canon, spike_canon = _build_canonical_cup(L_arm, R_arm, hL, hR, border_color)
    mouth_canon = _mouth_interior_coords(patch_canon, border_color, opposite_side="D")
    if len(mouth_canon) < CONTENT_COUNT_MIN_VISIBLE:
        return None

    patch, spike, mouth = _rotate_to_direction(patch_canon, spike_canon, mouth_canon, direction)
    ph, pw = patch.shape
    mouth_anchor_rel = _mouth_anchor_from_spike(
        spike,
        (0, 0, ph - 1, pw - 1),
        direction,
        set(mouth),
    )
    if mouth_anchor_rel is None:
        return None

    placement = _try_place_cup(rng, H, W, patch, spike, mouth_anchor_rel, direction, occ_cup, occ_path)
    if placement is None:
        return None
    r0, c0, tail_path, mouth_path = placement

    max_count = _max_split_count_without_overflow(
        len(tail_path),
        len(mouth_path),
        min(CONTENT_COUNT_MAX, len(mouth)),
    )
    if max_count < CONTENT_COUNT_MIN_VISIBLE:
        return None
    content_count = int(rng.integers(CONTENT_COUNT_MIN_VISIBLE, max_count + 1))
    tail_count, mouth_count = _split_content_count(content_count)

    patch_mask = patch != 0
    canvas_out[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    canvas_in[r0:r0 + ph, c0:c0 + pw][patch_mask] = border_color
    occ_cup[r0:r0 + ph, c0:c0 + pw] = True

    for r, c in tail_path[:tail_count]:
        canvas_out[r, c] = content_color
    for r, c in mouth_path[:mouth_count]:
        canvas_out[r, c] = content_color
    for r, c in tail_path:
        occ_path[r, c] = True
    for r, c in mouth_path:
        occ_path[r, c] = True

    return {
        "border_color": border_color,
        "content_color": content_color,
        "direction": direction,
        "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "tail_count": tail_count,
        "mouth_count": mouth_count,
        "path_cells": tail_path + mouth_path,
        "tail_path_len": len(tail_path),
        "mouth_path_len": len(mouth_path),
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
    """Single end-to-end generation attempt using output-first construction."""
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
