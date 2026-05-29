# Combo variant of 3dc255db (index 7)
# combo_spec: D2/2+D9/0
# Name: 箭头指示方向+矩形框架

"""
Fusion variant of 3dc255db: rectangular 3-sided cups plus detached arrow markers.

Design analysis
===============

a) Concrete behavior and trace
   - Original task: a Y/funnel border component contains content; the unique tail spike on the
     border tells the ejection direction.
   - D9/0 alone replaces the Y with a 3-sided rectangular frame whose opening defines the mouth
     and whose opposite closed side center acts as the spike.
   - This fusion breaks that coupling on purpose. The rectangular frame still has exactly one open
     side, but the ejection side is no longer forced to be opposite the opening. Instead, a small
     detached same-color 3-pixel L arrow sits just outside one of the frame's closed sides. The
     mouth is determined from the open side; the spike is the center pixel of the arrow-indicated
     closed side.
   - Example: a 5x5 frame open downward can now eject upward, leftward, or rightward. If the arrow
     sits to the left, content is still inside the downward-opening mouth, but after transformation
     it is erased and drawn as a leftward line from the left-side center pixel. A D9/0-only solver
     would wrongly eject upward from the top-center.

b) Complexity calibration
   - The cup body stays visually simple: strict 3-sided rectangles only.
   - The arrow vocabulary is small: a detached 3-cell L triomino, with two offset variants per
     direction so the dataset does not collapse to one memorized placement.
   - The key freedom is limited but meaningful: for each open side there are three legal closed
     sides, and at least one cup per generated grid uses a side that is not opposite the opening.
     This keeps the fusion observable and prevents degeneration to D9/0.

c) Implicit assumption audit against the original main()
   - Original assumption: one border component contains the full cup body. Broken here because the
     border color is split across a frame component and a detached arrow component.
   - Original assumption: the shape alone determines the tail. Broken here because the rectangle's
     opening only determines the mouth; the arrow chooses among the three closed sides.
   - Original flood-from-opposite-side logic is invalid. The flood must start from the frame's open
     side, not from the side opposite the arrow.
   - New solver assumptions:
       1. each border color belongs to at most one rectangular frame component;
       2. the associated arrow is a detached 3-cell 2x2-L component of the same color;
       3. the arrow lies exactly one blank cell beyond one frame side and overlaps that side's
          perpendicular span, so it can be associated unambiguously;
       4. the chosen tail side is closed and has odd length, so its center pixel is unique.
     The generator explicitly guarantees all four.

d) Placement legality
   - Generator builds the output state first: cup frame + detached arrow + ejected line.
   - Combined occupied cells of cup and arrow keep an 8-neighborhood gap from other cups/arrows.
   - The ray from the chosen spike to the grid edge must be clear of every pre-existing occupied
     cell and pre-existing ray.
   - The arrow is offset away from the ray, so the output line never overwrites the marker.

e) Visual observability
   - The frame opening is plainly visible, so humans can identify the mouth side.
   - The detached arrow is plainly visible on one closed side, so humans can see that ejection is
     not always opposite the opening.
   - Because generator forces at least one non-opposite tail per grid, a few samples are enough to
     show that "opening side" and "ejection side" are separate variables.

f) Visual cleanliness
   - Arrow markers stay one blank cell away from their own cup and never overlap the ray.
   - Cups, arrows, and rays are kept disjoint across objects.
   - Optional stray pixels stay isolated on the mouth side of some cup and do not touch any object.

g) Color inference
   - No new color mapping is introduced. The arrow reuses the frame color, so the association is
     local and structural rather than based on a global color permutation.

Post-change solver assumptions and generator guarantees
------------------------------------------------------
   - Assumption: every solvable border color yields exactly one rectangular frame component with a
     non-empty mouth interior. Guaranteed by the frame constructor.
   - Assumption: exactly one nearby L-arrow of the same color lies one cell beyond a closed side.
     Guaranteed by fixed arrow templates and unique border colors per cup.
   - Assumption: at least one cup per generated grid has tail_dir != opposite(open_side), so the
     fusion is behaviorally different from D9/0. Guaranteed by generation policy.
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


GRID_H_RANGE = (10, 13)
GRID_W_RANGE = (10, 13)
NUM_CUPS_CHOICES = (1, 2, 3)
CONTENT_COUNT_MAX = 7
STRAY_PROB = 0.4
STRAY_USE_BORDER_PROB = 0.5
GEN_RETRIES_PRIMARY = 300
GEN_RETRIES_FALLBACK = 300

DIR4 = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}
OPEN_TO_CLOSED = {
    "U": ("D", "L", "R"),
    "D": ("U", "L", "R"),
    "L": ("U", "D", "R"),
    "R": ("U", "D", "L"),
}
_STRUCT8 = np.ones((3, 3), dtype=bool)

# Cup body sizes are kept odd in both axes so any closed side can provide a unique center spike.
CUP_SHAPES = tuple(
    (h, w)
    for h in (3, 5, 7)
    for w in (3, 5, 7)
    if (h, w) != (3, 3)
)

ARROW_OFFSETS = {
    "U": (
        [(-3, -2), (-3, -1), (-2, -1)],
        [(-3, 1), (-3, 2), (-2, 1)],
    ),
    "D": (
        [(2, -1), (3, -2), (3, -1)],
        [(2, 1), (3, 1), (3, 2)],
    ),
    "L": (
        [(-2, -3), (-1, -3), (-1, -2)],
        [(1, -3), (1, -2), (2, -3)],
    ),
    "R": (
        [(-2, 2), (-1, 2), (-1, 3)],
        [(1, 2), (1, 3), (2, 2)],
    ),
}


def _flood_from_bbox_side(
    border_set: set[tuple[int, int]],
    bbox: tuple[int, int, int, int],
    side: str,
) -> set[tuple[int, int]]:
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


def _rect_frame_cells(
    bbox: tuple[int, int, int, int],
    open_side: str,
) -> set[tuple[int, int]]:
    """Exact cell set of a 3-sided rectangular frame with one open side."""
    r0, c0, r1, c1 = bbox
    cells: set[tuple[int, int]] = set()
    if open_side != "U":
        cells.update((r0, c) for c in range(c0, c1 + 1))
    if open_side != "D":
        cells.update((r1, c) for c in range(c0, c1 + 1))
    if open_side != "L":
        cells.update((r, c0) for r in range(r0, r1 + 1))
    if open_side != "R":
        cells.update((r, c1) for r in range(r0, r1 + 1))
    return cells


def _spike_from_bbox_center(
    bbox: tuple[int, int, int, int],
    side: str,
) -> tuple[int, int] | None:
    """Center pixel of one bbox side; returns None when that side length is even."""
    r0, c0, r1, c1 = bbox
    if side in ("U", "D"):
        width = c1 - c0 + 1
        if width % 2 == 0:
            return None
        return (r0 if side == "U" else r1, c0 + width // 2)
    height = r1 - r0 + 1
    if height % 2 == 0:
        return None
    return (r0 + height // 2, c0 if side == "L" else c1)


def _is_l_arrow_component(cells: set[tuple[int, int]]) -> bool:
    """3-cell detached arrow sprite: an L triomino inside a 2x2 bbox."""
    if len(cells) != 3:
        return False
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    if max(rs) - min(rs) != 1 or max(cs) - min(cs) != 1:
        return False
    window = {
        (min(rs), min(cs)),
        (min(rs), max(cs)),
        (max(rs), min(cs)),
        (max(rs), max(cs)),
    }
    return len(window - cells) == 1


def _overlap_len(a0: int, a1: int, b0: int, b1: int) -> int:
    return max(0, min(a1, b1) - max(a0, b0) + 1)


def _arrow_side(
    arrow_cells: set[tuple[int, int]],
    cup_bbox: tuple[int, int, int, int],
) -> str | None:
    """Infer which cup side the detached arrow marks."""
    ar0, ac0, ar1, ac1 = (int(v) for v in bbox_coords(np.array(sorted(arrow_cells), dtype=int)))
    r0, c0, r1, c1 = cup_bbox
    if ar1 < r0 and r0 - ar1 == 2 and _overlap_len(ac0, ac1, c0, c1) >= 1:
        return "U"
    if ar0 > r1 and ar0 - r1 == 2 and _overlap_len(ac0, ac1, c0, c1) >= 1:
        return "D"
    if ac1 < c0 and c0 - ac1 == 2 and _overlap_len(ar0, ar1, r0, r1) >= 1:
        return "L"
    if ac0 > c1 and ac0 - c1 == 2 and _overlap_len(ar0, ar1, r0, r1) >= 1:
        return "R"
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fusion rule:
    - A cup body is a detached 3-sided rectangular frame.
    - The frame's open side determines the mouth interior.
    - A detached same-color 3-pixel L arrow picks one closed side as the tail.
    - Content inside the mouth is erased and ejected from the center of that chosen side.
    """
    g = np.array(grid, dtype=int)
    H, W = g.shape
    bg = int(get_bg(g))
    out = g.copy()

    for border_col in colors_present(g, exclude={bg}):
        comps = [
            {(int(r), int(c)) for r, c in comp}
            for comp in components_mask(g == int(border_col), connectivity=8)
            if comp.size > 0
        ]
        if not comps:
            continue

        frame_candidates: list[tuple[set[tuple[int, int]], str, tuple[int, int, int, int], set[tuple[int, int]]]] = []
        arrow_candidates: list[set[tuple[int, int]]] = []

        for comp_set in comps:
            if _is_l_arrow_component(comp_set):
                arrow_candidates.append(comp_set)
                continue
            if len(comp_set) < 7:
                continue

            bbox = tuple(int(v) for v in bbox_coords(np.array(sorted(comp_set), dtype=int)))
            r0, c0, r1, c1 = bbox
            if r1 - r0 + 1 < 3 or c1 - c0 + 1 < 3:
                continue

            matches = []
            for open_side in ("U", "D", "L", "R"):
                if comp_set != _rect_frame_cells(bbox, open_side):
                    continue
                mouth = _flood_from_bbox_side(comp_set, bbox, open_side)
                if len(mouth) < 3:
                    continue
                matches.append((open_side, mouth))
            if len(matches) == 1:
                open_side, mouth = matches[0]
                frame_candidates.append((comp_set, open_side, bbox, mouth))

        passing: list[tuple[str, tuple[int, int], int, list[tuple[int, int]]]] = []
        for border_set, open_side, bbox, mouth in frame_candidates:
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

            pairings = []
            for arrow_cells in arrow_candidates:
                tail_dir = _arrow_side(arrow_cells, bbox)
                if tail_dir is None or tail_dir == open_side:
                    continue
                spike = _spike_from_bbox_center(bbox, tail_dir)
                if spike is None or spike not in border_set:
                    continue
                pairings.append((tail_dir, spike))

            if len(pairings) == 1:
                tail_dir, spike = pairings[0]
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


def _build_rect_cup(
    height: int,
    width: int,
    open_side: str,
    color: int,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Build one exact rectangular frame and enumerate its mouth cells."""
    patch = np.zeros((height, width), dtype=int)
    bbox = (0, 0, height - 1, width - 1)
    border_set = _rect_frame_cells(bbox, open_side)
    for r, c in border_set:
        patch[r, c] = color
    mouth = sorted(_flood_from_bbox_side(border_set, bbox, open_side))
    return patch, mouth


def _attach_arrow(
    rng: np.random.Generator,
    body_patch: np.ndarray,
    mouth: list[tuple[int, int]],
    tail_dir: str,
    color: int,
) -> tuple[np.ndarray, tuple[int, int], list[tuple[int, int]], tuple[int, int, int, int]]:
    """Return a combined patch containing the frame and a detached arrow."""
    h, w = body_patch.shape
    body_bbox = (0, 0, h - 1, w - 1)
    spike = _spike_from_bbox_center(body_bbox, tail_dir)
    if spike is None:
        raise ValueError("tail side has no unique center spike")

    body_coords = [(int(r), int(c)) for r, c in np.argwhere(body_patch == color)]
    arrow_template = ARROW_OFFSETS[tail_dir][int(rng.integers(0, 2))]
    arrow_coords = [(spike[0] + dr, spike[1] + dc) for dr, dc in arrow_template]
    all_coords = body_coords + arrow_coords

    min_r = min(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_r = max(r for r, _ in all_coords)
    max_c = max(c for _, c in all_coords)

    combo = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=int)
    for r, c in body_coords:
        combo[r - min_r, c - min_c] = color
    for r, c in arrow_coords:
        combo[r - min_r, c - min_c] = color

    shifted_spike = (spike[0] - min_r, spike[1] - min_c)
    shifted_mouth = [(r - min_r, c - min_c) for r, c in mouth]
    shifted_body_bbox = (-min_r, -min_c, -min_r + h - 1, -min_c + w - 1)
    return combo, shifted_spike, shifted_mouth, shifted_body_bbox


def _ray_inside_patch_is_clear(
    patch: np.ndarray,
    spike: tuple[int, int],
    direction: str,
) -> bool:
    ph, pw = patch.shape
    dr, dc = DIR4[direction]
    rr, cc = spike[0] + dr, spike[1] + dc
    while 0 <= rr < ph and 0 <= cc < pw:
        if patch[rr, cc] != 0:
            return False
        rr += dr
        cc += dc
    return True


def _sample_cup_shape(rng: np.random.Generator) -> tuple[int, int]:
    return CUP_SHAPES[int(rng.integers(0, len(CUP_SHAPES)))]


def _try_place_combo(
    rng: np.random.Generator,
    H: int,
    W: int,
    patch: np.ndarray,
    spike: tuple[int, int],
    tail_dir: str,
    content_count: int,
    occ_combo: np.ndarray,
    occ_path: np.ndarray,
) -> tuple[int, int, list[tuple[int, int]], list[tuple[int, int]]] | None:
    """Place one cup+arrow combo while preserving spacing and a clear ray to the edge."""
    ph, pw = patch.shape
    if not _ray_inside_patch_is_clear(patch, spike, tail_dir):
        return None

    dr, dc = DIR4[tail_dir]
    patch_mask = patch != 0
    occ_combo_dilated = ndimage.binary_dilation(occ_combo, structure=_STRUCT8)
    coords_list = [(r0, c0) for r0 in range(H - ph + 1) for c0 in range(W - pw + 1)]
    rng.shuffle(coords_list)

    for r0, c0 in coords_list:
        if np.any(occ_combo_dilated[r0:r0 + ph, c0:c0 + pw] & patch_mask):
            continue
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
        if any(occ_combo[r, c] or occ_path[r, c] for r, c in path_cells):
            continue

        eject_len = min(content_count, len(path_cells))
        return r0, c0, path_cells, path_cells[:eject_len]
    return None


def _stray_candidate_mask(
    H: int,
    W: int,
    bbox: tuple[int, int, int, int],
    open_side: str,
) -> np.ndarray:
    """Cells on the mouth side of a cup bbox, restricted to the perpendicular extent."""
    r0, c0, r1, c1 = bbox
    mask = np.zeros((H, W), dtype=bool)
    if open_side == "U":
        mask[:r0, c0:c1 + 1] = True
    elif open_side == "D":
        mask[r1 + 1:, c0:c1 + 1] = True
    elif open_side == "L":
        mask[r0:r1 + 1, :c0] = True
    else:
        mask[r0:r1 + 1, c1 + 1:] = True
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
    for cup in cups_meta:
        br0, bc0, br1, bc1 = cup["cup_bbox"]
        forbidden[br0:br1 + 1, bc0:bc1 + 1] = True
        for r, c in cup["path_cells"]:
            forbidden[r, c] = True

    candidate_mask = _stray_candidate_mask(H, W, meta["cup_bbox"], meta["open_side"]) & ~forbidden
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
    occ_combo: np.ndarray,
    occ_path: np.ndarray,
    require_nonopposite: bool,
) -> dict | None:
    open_side = str(rng.choice(["U", "D", "L", "R"]))
    tail_choices = list(OPEN_TO_CLOSED[open_side])
    if require_nonopposite:
        tail_choices = [d for d in tail_choices if d != OPPOSITE[open_side]]
    tail_dir = str(rng.choice(tail_choices))

    height, width = _sample_cup_shape(rng)
    body_patch, mouth_body = _build_rect_cup(height, width, open_side, border_color)
    if len(mouth_body) < 3:
        return None

    combo_patch, spike, mouth, body_bbox_local = _attach_arrow(
        rng, body_patch, mouth_body, tail_dir, border_color
    )
    content_count = int(rng.integers(1, min(CONTENT_COUNT_MAX, len(mouth)) + 1))

    placement = _try_place_combo(
        rng, H, W, combo_patch, spike, tail_dir, content_count, occ_combo, occ_path
    )
    if placement is None:
        return None
    r0, c0, path_cells, ejection_cells = placement

    patch_mask = combo_patch != 0
    canvas_out[r0:r0 + combo_patch.shape[0], c0:c0 + combo_patch.shape[1]][patch_mask] = border_color
    canvas_in[r0:r0 + combo_patch.shape[0], c0:c0 + combo_patch.shape[1]][patch_mask] = border_color
    occ_combo[r0:r0 + combo_patch.shape[0], c0:c0 + combo_patch.shape[1]] |= patch_mask

    for r, c in ejection_cells:
        canvas_out[r, c] = content_color
    for r, c in path_cells:
        occ_path[r, c] = True

    br0, bc0, br1, bc1 = body_bbox_local
    return {
        "border_color": border_color,
        "content_color": content_color,
        "open_side": open_side,
        "tail_dir": tail_dir,
        "cup_bbox": (r0 + br0, c0 + bc0, r0 + br1, c0 + bc1),
        "mouth_abs": [(r0 + r, c0 + c) for r, c in mouth],
        "content_count": content_count,
        "path_cells": path_cells,
    }


def _scatter_content(
    rng: np.random.Generator,
    canvas_in: np.ndarray,
    cups_meta: list[dict],
) -> bool:
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


def _gen_attempt(
    rng: np.random.Generator,
    num_cups: int | None = None,
) -> np.ndarray | None:
    if num_cups is None:
        num_cups = int(rng.choice(NUM_CUPS_CHOICES))

    H = int(rng.integers(GRID_H_RANGE[0], GRID_H_RANGE[1] + 1))
    W = int(rng.integers(GRID_W_RANGE[0], GRID_W_RANGE[1] + 1))
    canvas_out = np.zeros((H, W), dtype=int)
    canvas_in = np.zeros((H, W), dtype=int)

    palette = list(range(1, 10))
    rng.shuffle(palette)
    occ_combo = np.zeros((H, W), dtype=bool)
    occ_path = np.zeros((H, W), dtype=bool)

    force_nonopposite_idx = int(rng.integers(0, num_cups))
    cups_meta: list[dict] = []
    for cup_idx in range(num_cups):
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
            occ_combo,
            occ_path,
            require_nonopposite=(cup_idx == force_nonopposite_idx),
        )
        if meta is None:
            return None
        cups_meta.append(meta)

    if len(cups_meta) != num_cups:
        return None
    if not any(meta["tail_dir"] != OPPOSITE[meta["open_side"]] for meta in cups_meta):
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
