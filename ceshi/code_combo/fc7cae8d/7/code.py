# Combo variant of fc7cae8d (index 7)
# combo_spec: D1/5+D3/3
# Name: 对角对称+十字分割

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

HERE = Path(__file__).resolve()
ARC_UTILS_DIR = None
for parent in HERE.parents:
    candidate = parent / "arc2_solver_select"
    if candidate.is_dir():
        ARC_UTILS_DIR = candidate
        break
if ARC_UTILS_DIR is None:
    raise RuntimeError("arc2_solver_select directory not found")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste


# Design analysis for the combo variant:
# a) Concrete behavior and trace:
#    - Original fc7cae8d used two solid outer frame edges as controls and two long object guide
#      edges as cues. The solver matched side-color to side-color, then cropped only those guide
#      edges away.
#    - This combo moves both pieces of information into a new joint code:
#      1. The frame becomes an internal cross. Its four half-arms carry four distinct colors:
#         `up/down/left/right`.
#      2. Those half-arm colors define four quadrants by pairing one vertical half with one
#         horizontal half: TL=(up,left), TR=(up,right), BL=(down,left), BR=(down,right).
#      3. The object becomes `core + full 1px shell`. Only two opposite shell corners are recolored.
#         The shell corner sitting on the chosen quadrant gets the row-half color of that quadrant
#         (`up` or `down`); the opposite shell corner gets the column-half color (`left` or `right`).
#    - Small trace:
#      cross arms = {up:2, down:8, left:4, right:7}
#      target quadrant = TR
#      aligned object shell markers = {tr:2, bl:7}
#      input object is a rotated/flipped version, e.g. {tl:7, br:2}
#      solver detects cross arm colors, derives the four legal diagonal signatures:
#        TL->{tl:2, br:4}, TR->{tr:2, bl:7}, BL->{bl:8, tr:4}, BR->{br:8, tl:7}
#      then searches the dihedral transforms until the object matches one of those signatures,
#      and finally removes the full shell to output the core.
#    - Difference from the original:
#      the frame control is no longer border-aligned, the object cue is no longer a whole guide
#      edge, and the matched state is now a quadrant-specific diagonal signature.
# b) Complexity calibration:
#    - Four distinct cross-arm colors are used so all four quadrant pairs are different.
#    - The core is forced to be dihedrally asymmetric so rotate/flip reasoning remains visible.
#    - The shell always has exactly two opposite colored corners, keeping the cue simple enough
#      for humans while still making the diagonal aspect visually obvious.
# c) Implicit assumptions review:
#    - Old `detect_controls_frame()` assumed almost-solid outer border edges. Invalid here because
#      the border is empty and the controls sit on an internal cross.
#    - Old `extract_center_object()` choosing the largest interior component is unsafe because the
#      internal cross is also an interior component. The new solver instead chooses the densest
#      interior component, which is the filled rectangle, not the thin cross.
#    - Old `guide_edges()` assumed nearly uniform full sides on the object. Invalid here because
#      only two opposite shell corners differ from the fill.
#    - Old crop logic removed only the two matched sides. Invalid here because the informative
#      carrier is the entire shell, so the correct crop is always `t[1:-1,1:-1]`.
#    - New solver assumptions introduced here:
#      1. exactly one dense rectangular interior object exists;
#      2. the cross row and cross column are the strongest non-background row/column counts;
#      3. each half-arm is monochromatic enough to recover one dominant color;
#      4. the shell markers occupy exactly two opposite corners.
#      The generator below enforces each assumption explicitly.
# d) Placement legality:
#    - The cross is painted first and never touches the outer border.
#    - The object is then placed wholly inside one quadrant with at least one background cell of
#      spacing from both the border and the cross, so there is no overlap and no ambiguity.
# e) Visual observability:
#    - A human sees one large cross with four repeated arm colors and one rectangle with two
#      opposite colored corners. Across examples, the output is always the inner rectangle, so the
#      natural hypothesis is "rotate/flip until the corner colors correspond to one quadrant's arm
#      colors, then remove the shell".
# f) Visual cleanliness:
#    - No border noise is added. The frame is one internal cross, the object is one dense rectangle,
#      and the two shell-corner markers are isolated. This avoids layered clutter.
# g) Color inferability:
#    - Only four control colors appear in the frame and only two of them reappear on the object.
#      The mapping is structurally anchored by quadrant geometry rather than arbitrary recoloring.


CORNER_POS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}
OPPOSITE_CORNER = {"tl": "br", "tr": "bl", "bl": "tr", "br": "tl"}
QUADRANTS = ("tl", "tr", "bl", "br")
CROSS_CENTER_COLOR = 5


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _majority_non_bg(vals: np.ndarray, bg: int) -> int | None:
    seq = [int(v) for v in vals if int(v) != int(bg)]
    if not seq:
        return None
    return int(Counter(seq).most_common(1)[0][0])


def _quadrant_signature(arms: dict[str, int], quadrant: str) -> dict[str, int]:
    row_arm = "up" if quadrant[0] == "t" else "down"
    col_arm = "left" if quadrant[1] == "l" else "right"
    return {
        quadrant: int(arms[row_arm]),
        OPPOSITE_CORNER[quadrant]: int(arms[col_arm]),
    }


def _guide_corners(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    out: dict[str, int] = {}
    for corner, (r, c) in CORNER_POS.items():
        val = int(arr[r, c])
        if val != fill:
            out[corner] = val
    return out


def _matched_quadrants(markers: dict[str, int], arms: dict[str, int]) -> list[str]:
    out: list[str] = []
    for quadrant in QUADRANTS:
        if markers == _quadrant_signature(arms, quadrant):
            out.append(quadrant)
    return out


def _quadrant_capacity(
    grid_shape: tuple[int, int], mid_r: int, mid_c: int, quadrant: str
) -> tuple[int, int]:
    h, w = grid_shape
    if quadrant[0] == "t":
        h_cap = mid_r - 2
    else:
        h_cap = h - mid_r - 3
    if quadrant[1] == "l":
        w_cap = mid_c - 2
    else:
        w_cap = w - mid_c - 3
    return int(h_cap), int(w_cap)


def _fits_quadrant(
    patch_shape: tuple[int, int], grid_shape: tuple[int, int], mid_r: int, mid_c: int, quadrant: str
) -> bool:
    h_cap, w_cap = _quadrant_capacity(grid_shape, mid_r, mid_c, quadrant)
    ph, pw = patch_shape
    return ph <= h_cap and pw <= w_cap


def _matching_transforms(
    obj: np.ndarray, arms: dict[str, int], grid_shape: tuple[int, int], mid_r: int, mid_c: int
) -> list[tuple[str, int, bool]]:
    out: list[tuple[str, int, bool]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            markers = _guide_corners(t)
            for quadrant in _matched_quadrants(markers, arms):
                if _fits_quadrant(t.shape, grid_shape, mid_r, mid_c, quadrant):
                    out.append((quadrant, int(k), bool(flip)))
    return out


def _detect_cross_controls(arr: np.ndarray, bg: int) -> tuple[int, int, dict[str, int]]:
    non_bg = (arr != int(bg))
    row_counts = non_bg.sum(axis=1)
    col_counts = non_bg.sum(axis=0)
    mid_r = int(np.argmax(row_counts))
    mid_c = int(np.argmax(col_counts))

    arms = {
        "left": _majority_non_bg(arr[mid_r, 1:mid_c], bg),
        "right": _majority_non_bg(arr[mid_r, mid_c + 1 : -1], bg),
        "up": _majority_non_bg(arr[1:mid_r, mid_c], bg),
        "down": _majority_non_bg(arr[mid_r + 1 : -1, mid_c], bg),
    }
    if any(v is None for v in arms.values()):
        raise ValueError("failed to detect cross controls")
    return mid_r, mid_c, {k: int(v) for k, v in arms.items()}


def _extract_dense_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best = None
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        ph = int(r1 - r0 + 1)
        pw = int(c1 - c0 + 1)
        if ph < 5 or pw < 5:
            continue
        area = int(comp["area"])
        density = area / float(ph * pw)
        patch = crop_bbox(arr, (r0, c0, r1, c1))
        markers = _guide_corners(patch)
        if len(markers) != 2:
            continue
        score = (density, min(ph, pw), area)
        if best is None or score > best[0]:
            best = (score, patch)
    if best is None:
        raise ValueError("dense guided object not found")
    return best[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a shell-framed rectangle to one quadrant of an internal colored cross.

    Input structure:
    - One internal cross, not on the outer border. Its four half-arms use four distinct colors.
    - One interior rectangle with a full 1px shell. Exactly two opposite shell corners are recolored.

    Rule:
    - Interpret the cross half-arms as quadrant controls.
    - Rotate/mirror the rectangle until its diagonal shell-corner colors match one quadrant's
      signature.
    - Remove the whole shell and output the core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    mid_r, mid_c, arms = _detect_cross_controls(g, bg)
    obj = _extract_dense_object(g, bg)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            markers = _guide_corners(t)
            if len(markers) != 2:
                continue
            for quadrant in _matched_quadrants(markers, arms):
                if _fits_quadrant(t.shape, g.shape, mid_r, mid_c, quadrant):
                    if t.shape[0] > 2 and t.shape[1] > 2:
                        return t[1:-1, 1:-1]
                    return t.copy()

    raise ValueError("no valid transform matched a quadrant signature")


def _is_dihedrally_asymmetric(arr: np.ndarray) -> bool:
    for k in range(4):
        base = np.rot90(arr, int(k))
        for flip in (False, True):
            if k == 0 and not flip:
                continue
            t = np.fliplr(base) if flip else base
            if t.shape == arr.shape and np.array_equal(t, arr):
                return False
    return True


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    for _ in range(100):
        core = np.full((h, w), int(fill), dtype=int)
        area = h * w
        max_nonfill = max(len(pattern_colors) + 2, int(area * 0.40))
        target_nonfill = random.randint(max(len(pattern_colors) + 1, area // 9), max_nonfill)
        painted = 0

        anchor_cells = [(0, 0), (h - 1, min(1, w - 1)), (h // 2, w - 1)]
        for idx, color in enumerate(pattern_colors[: min(3, len(pattern_colors))]):
            r, c = anchor_cells[idx]
            core[r, c] = int(color)
            painted += 1

        while painted < target_nonfill:
            color = random.choice(pattern_colors)
            shape_kind = random.choice(["dot", "line", "rect", "el"])
            if shape_kind == "dot":
                cells = [(random.randrange(h), random.randrange(w))]
            elif shape_kind == "line":
                line_len = random.randint(2, min(5, max(h, w)))
                if random.random() < 0.5:
                    rr = random.randrange(h)
                    c0 = random.randrange(w - line_len + 1)
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(h - line_len + 1)
                    cc = random.randrange(w)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            elif shape_kind == "el":
                arm = random.randint(2, min(4, h, w))
                r0 = random.randrange(h - arm + 1)
                c0 = random.randrange(w - arm + 1)
                cells = [(r0 + d, c0) for d in range(arm)] + [(r0 + arm - 1, c0 + d) for d in range(arm)]
            else:
                mh = random.randint(1, min(3, h))
                mw = random.randint(1, min(3, w))
                r0 = random.randrange(h - mh + 1)
                c0 = random.randrange(w - mw + 1)
                cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

            remaining = target_nonfill - painted
            if remaining <= 0:
                break
            fresh = [(r, c) for r, c in cells if core[r, c] == fill]
            if len(fresh) > remaining:
                fresh = random.sample(fresh, remaining)
            for r, c in fresh:
                core[r, c] = int(color)
            painted += len(fresh)

        if _is_dihedrally_asymmetric(core):
            return core

    raise RuntimeError("failed to build asymmetric core")


def _attach_guides(core: np.ndarray, markers: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    fill = int(Counter(map(int, core.flatten())).most_common(1)[0][0])
    out = np.full((h + 2, w + 2), int(fill), dtype=int)
    paste(out, core, 1, 1)
    for corner, color in markers.items():
        r, c = CORNER_POS[corner]
        out[r, c] = int(color)
    return out


def _paint_cross(grid: np.ndarray, arms: dict[str, int], mid_r: int, mid_c: int) -> tuple[int, int]:
    grid[mid_r, 1:mid_c] = int(arms["left"])
    grid[mid_r, mid_c + 1 : -1] = int(arms["right"])
    grid[1:mid_r, mid_c] = int(arms["up"])
    grid[mid_r + 1 : -1, mid_c] = int(arms["down"])
    grid[mid_r, mid_c] = int(CROSS_CENTER_COLOR)
    return mid_r, mid_c


def _quadrant_top_left_ranges(
    grid_shape: tuple[int, int], patch_shape: tuple[int, int], quadrant: str, mid_r: int, mid_c: int
) -> tuple[range, range] | None:
    h, w = grid_shape
    ph, pw = patch_shape

    if quadrant[0] == "t":
        r_min = 1
        r_max = mid_r - ph - 1
    else:
        r_min = mid_r + 2
        r_max = h - ph - 1

    if quadrant[1] == "l":
        c_min = 1
        c_max = mid_c - pw - 1
    else:
        c_min = mid_c + 2
        c_max = w - pw - 1

    if r_min > r_max or c_min > c_max:
        return None
    return range(r_min, r_max + 1), range(c_min, c_max + 1)


def _place_in_quadrant(grid: np.ndarray, patch: np.ndarray, quadrant: str, mid_r: int, mid_c: int) -> tuple[int, int]:
    ranges = _quadrant_top_left_ranges(grid.shape, patch.shape, quadrant, mid_r, mid_c)
    if ranges is None:
        raise ValueError("patch does not fit in quadrant")
    rr, cc = ranges
    r0 = random.choice(list(rr))
    c0 = random.choice(list(cc))
    paste(grid, patch, r0, c0, transparent=0)
    return r0, c0


def _old_edge_signature_score(grid: np.ndarray, bg: int = 0) -> int:
    sides = {
        "top": grid[0, 1:-1],
        "bottom": grid[-1, 1:-1],
        "left": grid[1:-1, 0],
        "right": grid[1:-1, -1],
    }
    score = 0
    for vals in sides.values():
        seq = [int(v) for v in vals if int(v) != int(bg)]
        if not seq:
            continue
        _col, count = Counter(seq).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            score += 1
    return score


def _fusion_metrics(grid: np.ndarray) -> dict[str, object]:
    bg = int(get_bg(grid))
    mid_r, mid_c, arms = _detect_cross_controls(grid, bg)
    obj = _extract_dense_object(grid, bg)
    markers = _guide_corners(obj)
    border_non_bg = int((grid[0] != bg).sum() + (grid[-1] != bg).sum() + (grid[:, 0] != bg).sum() + (grid[:, -1] != bg).sum())
    return {
        "mid_r": mid_r,
        "mid_c": mid_c,
        "arms": arms,
        "markers": markers,
        "matched_quadrants": _matched_quadrants(markers, arms),
        "matching_transforms": _matching_transforms(obj, arms, grid.shape, mid_r, mid_c),
        "border_non_bg": border_non_bg,
        "old_edge_score": _old_edge_signature_score(grid, bg=bg),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.choice([25, 27, 29, 31])
    mid_r = random.randint(8, n - 9)
    mid_c = random.randint(8, n - 9)

    all_colors = list(range(1, 10))
    arm_palette = random.sample([c for c in all_colors if c != CROSS_CENTER_COLOR], 4)
    arms = {
        "up": int(arm_palette[0]),
        "down": int(arm_palette[1]),
        "left": int(arm_palette[2]),
        "right": int(arm_palette[3]),
    }

    target_quadrant = random.choice(QUADRANTS)
    guide_colors = set(arms.values())
    remaining = [c for c in all_colors if c not in guide_colors and c != CROSS_CENTER_COLOR]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    aligned_h_cap, aligned_w_cap = _quadrant_capacity((n, n), mid_r, mid_c, target_quadrant)
    max_core_h = aligned_h_cap - 2
    max_core_w = aligned_w_cap - 2
    if max_core_h < 5 or max_core_w < 5:
        return None
    size_pairs: list[tuple[int, int]] = []
    for core_h in range(5, max_core_h + 1):
        for core_w in range(5, max_core_w + 1):
            ph = core_h + 2
            pw = core_w + 2
            if ph == pw:
                continue
            fits_aligned = ph <= aligned_h_cap and pw <= aligned_w_cap
            fits_transposed = pw <= aligned_h_cap and ph <= aligned_w_cap
            if fits_aligned and not fits_transposed:
                size_pairs.append((core_h, core_w))
    if not size_pairs:
        return None
    core_h, core_w = random.choice(size_pairs)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    aligned_markers = _quadrant_signature(arms, target_quadrant)
    aligned_object = _attach_guides(core, aligned_markers)

    transforms = [(k, flip) for k in range(4) for flip in (False, True)]
    random.shuffle(transforms)
    input_object = None
    for k, flip in transforms:
        if k == 0 and not flip:
            continue
        candidate = _inverse_transform(aligned_object, k, flip)
        if _guide_corners(candidate) != aligned_markers:
            input_object = candidate
            break
    if input_object is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    _paint_cross(grid, arms, mid_r, mid_c)

    placement_choices = [
        quadrant
        for quadrant in QUADRANTS
        if _fits_quadrant(input_object.shape, grid.shape, mid_r, mid_c, quadrant)
    ]
    if not placement_choices:
        return None
    placement_quadrant = random.choice(placement_choices)
    try:
        _place_in_quadrant(grid, input_object, placement_quadrant, mid_r, mid_c)
    except ValueError:
        return None

    metrics = _fusion_metrics(grid)
    if metrics["old_edge_score"] != 0:
        return None
    if metrics["border_non_bg"] != 0:
        return None
    transform_matches = metrics["matching_transforms"]
    if len(transform_matches) != 1 or transform_matches[0][0] != target_quadrant:
        return None
    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")
