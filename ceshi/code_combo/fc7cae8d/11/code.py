# Combo variant of fc7cae8d (index 11)
# combo_spec: D1/5+D11/2
# Name: 对角对称+正方形核心

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[5]
ARC_UTILS_DIR = ROOT / "arc2_solver_select"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the combo variant:
# a) Concrete behavior and trace:
#    - Original task: two full frame edges encode two target sides, while the object carries two long
#      guide edges. A dihedral transform aligns object-side colors to frame-side colors, then the
#      matched guide edges are cropped away.
#    - D1/5 replaces "long edges" with "diagonal corner signals": the frame now has short inward
#      diagonal rays from selected outer corners, and the object has a 1-cell shell whose informative
#      pixels are shell-corner markers. After alignment the whole shell is removed.
#    - D11/2 forces the inner core to be square. Example:
#         original object:   top=red, left=blue, core size 5x8
#         combo object:      shell corners TL=red, BR=blue, core size 7x7
#         frame control:     red TL diagonal, blue BR diagonal
#      In the combo, rotating the object keeps the outer bbox square, so side lengths no longer reveal
#      orientation. The only explicit orientation signal is which colored marker sits on which diagonal.
# b) Complexity calibration:
#    - The combo uses three corner controls: one opposite-corner diagonal pair plus one extra corner
#      anchor. The extra anchor is the minimal addition that breaks the square-core reflection
#      ambiguity while keeping the puzzle visibly diagonal rather than returning to edge alignment.
#    - The core is square but internally dihedrally asymmetric. This prevents collapse where every
#      transform yields the same output, while still removing the original "non-square silhouette"
#      orientation cue.
#    - Frame diagonals are short and clean, plus limited border noise, so the diagonal rule is visible
#      to humans from a few examples and not mistaken for the original full-edge rule.
# c) Implicit assumptions review against the original solver:
#    - `detect_controls_frame()` is invalid now because controls are no longer almost-solid edges.
#      Replaced by `detect_controls_diagonal()`, which inspects only short corner rays.
#    - `guide_edges()` is invalid because informative pixels no longer span entire sides.
#      Replaced by `_guide_corners()`, which reads only shell corners.
#    - Original crop logic removed only selected sides. Invalid because the combo wraps the core in a
#      full shell whose semantics are "orientation carrier"; after matching, the solver removes the
#      whole 1-cell shell.
#    - New assumptions introduced by `main()`:
#      1. Exactly three strong corner controls exist: an opposite diagonal pair plus one anchor corner.
#      2. The center object is one interior component with a 1-cell shell.
#      3. Only shell corners differ from the shell fill.
#      4. Marker colors are distinct, so corner assignment is directional.
#      The generator enforces all four.
# d) Placement legality:
#    - Frame diagonals are painted first. The object is then placed with
#      `random_free_location(..., transparent=0, border=2)`, so it cannot overlap border controls and
#      also cannot become 4-connected to any colored border cell through immediate adjacency.
#    - The shell is nonzero everywhere, so the whole object remains a single 4-connected component.
# e) Visual observability:
#    - A human sees three colored diagonals entering from corners and three same-colored marker pixels
#      on the shell corners of the central square object. Across examples, the output is always the
#      interior square after the object is reoriented to match those corner controls.
# f) Visual cleanliness:
#    - Controls live only near the outer corners; the object lives strictly inside with a full 1-cell
#      shell. No region mixes multiple semantics on the same cells, so cues remain separable.
# g) Color inferability:
#    - Colors are reused directly rather than remapped. Each marker color appears both on the frame
#      diagonal and on one shell corner, so the color correspondence is structurally signposted.
# h) Fusion non-degeneration target:
#    - Every generated core is square (`h == w`), so the original rectangle aspect-ratio cue is gone.
#    - The input has zero "old-style solid edge controls", so the sample cannot be solved by the
#      original fc7cae8d edge detector.
# i) Post-change solver branch audit:
#    - `detect_controls_diagonal()` assumes each true control dominates its short ray; generator paints
#      four-cell rays without overlap and keeps noise off those cells.
#    - `extract_center_object()` assumes the intended object is the largest interior non-bg component;
#      generator creates only border decorations besides the central object.
#    - The transform loop assumes exactly one dihedral transform matches both corner-color constraints;
#      generator uses distinct colors on opposite corners, which makes the match unique.


CORNER_POS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}

DIAGONAL_PAIRS = [("tl", "br"), ("tr", "bl")]
ANCHOR_BY_PAIR = {
    ("tl", "br"): ["tr", "bl"],
    ("tr", "bl"): ["tl", "br"],
}


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _corner_ray_cells(shape: tuple[int, int], corner: str, length: int = 4) -> list[tuple[int, int]]:
    h, w = shape
    span = min(int(length), int(h), int(w))
    if corner == "tl":
        return [(i, i) for i in range(span)]
    if corner == "tr":
        return [(i, w - 1 - i) for i in range(span)]
    if corner == "bl":
        return [(h - 1 - i, i) for i in range(span)]
    return [(h - 1 - i, w - 1 - i) for i in range(span)]


def _guide_corners(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    out: dict[str, int] = {}
    for corner, (r, c) in CORNER_POS.items():
        val = int(arr[r, c])
        if val != fill:
            out[corner] = val
    return out


def _is_dihedrally_asymmetric(arr: np.ndarray) -> bool:
    for k in range(4):
        base = np.rot90(arr, int(k))
        for flip in [False, True]:
            if k == 0 and not flip:
                continue
            t = np.fliplr(base) if flip else base
            if t.shape == arr.shape and np.array_equal(t, arr):
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a shell-framed square core to diagonal corner controls via dihedral transforms.

    Input structure:
    - Background grid with three short colored diagonals from corners.
    - A central square object: asymmetric square core + 1-cell shell.
    - Exactly three shell corners are recolored as guide markers.

    Rule:
    - Rotate/mirror the square object until its guide-corner colors match the frame's diagonal controls.
    - Remove the entire 1-cell shell and output the square core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def detect_controls_diagonal(arr: np.ndarray) -> list[tuple[str, int]]:
        cand: list[tuple[str, int, int]] = []
        for corner in CORNER_POS:
            cells = _corner_ray_cells(arr.shape, corner, length=4)
            vals = [int(arr[r, c]) for r, c in cells]
            non_bg = [v for v in vals if v != bg]
            if not non_bg:
                continue
            col, count = Counter(non_bg).most_common(1)[0]
            if int(count) >= len(cells) - 1:
                cand.append((corner, int(col), int(count)))
        cand.sort(key=lambda item: item[2], reverse=True)
        return [(corner, col) for corner, col, _count in cand[:3]]

    def extract_center_object(arr: np.ndarray) -> np.ndarray:
        h, w = arr.shape
        comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
        best = None
        for comp in comps:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
                continue
            area = int(comp["area"])
            if best is None or area > best[0]:
                best = (area, (r0, c0, r1, c1))
        if best is None:
            return arr.copy()
        return crop_bbox(arr, best[1])

    desired = {corner: col for corner, col in detect_controls_diagonal(g)}
    obj = extract_center_object(g)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in [False, True]:
            t = np.fliplr(base) if flip else base
            markers = _guide_corners(t)
            if all(markers.get(corner) == col for corner, col in desired.items()):
                if t.shape[0] > 2 and t.shape[1] > 2:
                    return t[1:-1, 1:-1]
                return t.copy()

    if obj.shape[0] > 2 and obj.shape[1] > 2:
        return obj[1:-1, 1:-1]
    return obj


def _build_core(side: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    for _ in range(100):
        core = np.full((side, side), int(fill), dtype=int)
        area = side * side
        max_nonfill = max(len(pattern_colors) + 3, int(area * 0.38))
        target_nonfill = random.randint(max(len(pattern_colors) + 2, area // 8), max_nonfill)
        painted = 0

        anchor_cells = [(0, 0), (side - 1, 1 % side), (side // 2, side - 1), (1 % side, side // 3)]
        for idx, color in enumerate(pattern_colors[: min(4, len(pattern_colors))]):
            r, c = anchor_cells[idx]
            if core[r, c] == fill:
                core[r, c] = int(color)
                painted += 1

        while painted < target_nonfill:
            color = random.choice(pattern_colors)
            shape_kind = random.choice(["dot", "line", "rect", "el"])
            if shape_kind == "dot":
                cells = [(random.randrange(side), random.randrange(side))]
            elif shape_kind == "line":
                line_len = random.randint(2, min(5, side))
                if random.random() < 0.5:
                    rr = random.randrange(side)
                    c0 = random.randrange(side - line_len + 1)
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(side - line_len + 1)
                    cc = random.randrange(side)
                    cells = [(r0 + dr, cc) for dr in range(line_len)]
            elif shape_kind == "el":
                arm = random.randint(2, min(4, side))
                r0 = random.randrange(side - arm + 1)
                c0 = random.randrange(side - arm + 1)
                cells = [(r0 + d, c0) for d in range(arm)] + [(r0 + arm - 1, c0 + d) for d in range(arm)]
            else:
                mh = random.randint(1, min(3, side))
                mw = random.randint(1, min(3, side))
                r0 = random.randrange(side - mh + 1)
                c0 = random.randrange(side - mw + 1)
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

    raise RuntimeError("failed to build asymmetric square core")


def _attach_guides(core: np.ndarray, guide_corners: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    fill = int(Counter(map(int, core.flatten())).most_common(1)[0][0])
    out = np.full((h + 2, w + 2), int(fill), dtype=int)
    paste(out, core, 1, 1)
    for corner, color in guide_corners.items():
        r, c = CORNER_POS[corner]
        out[r, c] = int(color)
    return out


def _paint_frame(grid: np.ndarray, ctrl_corners: dict[str, int]) -> None:
    reserved: set[tuple[int, int]] = set()
    for corner, color in ctrl_corners.items():
        for r, c in _corner_ray_cells(grid.shape, corner, length=4):
            grid[r, c] = int(color)
            reserved.add((r, c))

    border_cells: list[tuple[int, int]] = []
    h, w = grid.shape
    for c in range(w):
        border_cells.append((0, c))
        border_cells.append((h - 1, c))
    for r in range(1, h - 1):
        border_cells.append((r, 0))
        border_cells.append((r, w - 1))
    border_cells = sorted(set(border_cells) - reserved)
    if border_cells:
        min_count = int(np.ceil(0.14 * len(border_cells)))
        max_count = int(np.floor(0.26 * len(border_cells)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(border_cells, noise_count):
            grid[r, c] = 5


def _old_edge_signature_score(grid: np.ndarray, bg: int = 0) -> int:
    sides = {
        "top": grid[0, 1:-1],
        "bottom": grid[-1, 1:-1],
        "left": grid[1:-1, 0],
        "right": grid[1:-1, -1],
    }
    score = 0
    for vals in sides.values():
        vals = list(map(int, vals))
        non_bg = [v for v in vals if v != bg]
        if not non_bg:
            continue
        _col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            score += 1
    return score


def _matching_transform_count(obj: np.ndarray, desired: dict[str, int]) -> int:
    count = 0
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in [False, True]:
            t = np.fliplr(base) if flip else base
            markers = _guide_corners(t)
            if all(markers.get(corner) == col for corner, col in desired.items()):
                count += 1
    return count


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    corner_a, corner_b = random.choice(DIAGONAL_PAIRS)
    anchor_corner = random.choice(ANCHOR_BY_PAIR[(corner_a, corner_b)])

    all_colors = list(range(1, 10))
    ctrl_a, ctrl_b, ctrl_anchor = random.sample([c for c in all_colors if c != 5], 3)
    remaining = [c for c in all_colors if c not in {5, ctrl_a, ctrl_b, ctrl_anchor}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_side = min(18, n - 5)
    if max_side < 5:
        return None
    side = random.randint(5, max_side)
    core = _build_core(side, fill, pattern_colors)

    guide_corners = {corner_a: ctrl_a, corner_b: ctrl_b, anchor_corner: ctrl_anchor}
    aligned_object = _attach_guides(core, guide_corners)

    transforms = [(k, flip) for k in range(4) for flip in [False, True]]
    random.shuffle(transforms)
    input_object = None
    for k, flip in transforms:
        if k == 0 and not flip:
            continue
        candidate = _inverse_transform(aligned_object, k, flip)
        if _guide_corners(candidate) != guide_corners and not np.array_equal(candidate, aligned_object):
            input_object = candidate
            break
    if input_object is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, guide_corners)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=2)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if _old_edge_signature_score(grid, bg=0) != 0:
        return None
    if _matching_transform_count(input_object, guide_corners) != 1:
        return None
    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant")
