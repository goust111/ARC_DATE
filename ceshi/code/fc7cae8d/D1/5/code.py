# Variant of fc7cae8d
# Dimension: D1 — 匹配条件：边颜色对齐 (match_cond)
# Change: 引导边颜色 == 框架控制边颜色（edge_color_match），通过旋转/翻转实现对齐... → 对角对称匹配

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


# Design analysis for the D1 variant:
# a) Concrete behavior:
#    - Original task: the object carries two long colored guide edges (for example top + left),
#      and the frame has two colored control edges on the outer border. The solver aligns sides.
#    - Variant: the object becomes "core + 1-cell shell". The shell is mostly the fill color, but
#      exactly two opposite shell corners are recolored as guide markers, e.g. TL=2 and BR=7.
#      The frame controls are no longer full border edges; instead, each chosen grid corner emits a
#      short inward diagonal streak with the matching control color.
#    - Small trace:
#      original: control says "top is red, left is blue" -> rotate/flip until object's red top edge
#      and blue left edge line up -> crop those guide edges only.
#      variant: control says "top-left diagonal is red, bottom-right diagonal is blue" -> rotate/flip
#      until object's red shell-corner sits at TL and blue shell-corner sits at BR -> crop the whole
#      1-cell shell and keep the core.
# b) Complexity calibration:
#    - Opposite corners are used instead of adjacent sides, so the signal is genuinely diagonal.
#    - Core patterns are forced to be dihedrally asymmetric so the transform is visually meaningful.
#    - Input transforms are chosen so the guide-corner placement differs from the aligned placement,
#      preventing the generator from collapsing into an already-solved view too often.
# c) Implicit assumptions review:
#    - Old `detect_controls_frame()` assumed nearly solid full edges; invalid here because controls are
#      only short corner diagonals.
#    - Old `guide_edges()` assumed the guide color occupies almost an entire side; invalid because the
#      new guide signal is only at shell corners.
#    - Old crop logic removed only the selected sides; invalid because the new guide carrier is a full
#      shell whose informative pixels happen to be at opposite corners.
#    - New solver assumptions: exactly two strong diagonal corner controls exist; the center object is
#      a filled rectangle with a 1-cell shell; only the shell corners can differ from the shell fill.
#      The generator enforces each of those properties.
# d) Placement legality:
#    - Control diagonals are painted first on the big grid, then the object is placed with
#      `random_free_location(..., transparent=0, border=1)`, so it cannot overlap controls or leave
#      the canvas. The object is fully nonzero, so `find_components(..., connectivity=4)` still sees
#      it as one interior component.
# e) Visual observability:
#    - Human solvers can now see that the colored cues sit on diagonals from corners rather than on
#      full edges. Because the core is asymmetric and the shell markers move to different corners,
#      the diagonal-matching rule is visible from a few examples instead of looking like the original
#      edge-alignment puzzle.


CORNER_POS = {
    "tl": (0, 0),
    "tr": (0, -1),
    "bl": (-1, 0),
    "br": (-1, -1),
}

DIAGONAL_PAIRS = [("tl", "br"), ("tr", "bl")]


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


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a shell-framed rectangle to diagonal corner controls via dihedral transforms.

    Input structure:
    - bg grid with two short colored diagonals, each starting from one outer corner.
    - A central filled rectangle with a 1-cell shell; two opposite shell corners are recolored as guides.

    Rule:
    - Rotate/mirror the object so its guide-corner colors match the frame's diagonal corner controls.
    - Remove the outer 1-cell shell and output the core.
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
        cand.sort(key=lambda t: t[2], reverse=True)
        return [(corner, col) for corner, col, _count in cand[:2]]

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


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    for _ in range(80):
        core = np.full((h, w), int(fill), dtype=int)
        area = h * w
        max_nonfill = max(len(pattern_colors) + 2, int(area * 0.40))
        target_nonfill = random.randint(max(len(pattern_colors) + 1, area // 9), max_nonfill)
        painted = 0

        anchor_cells = [(0, 0), (h - 1, 1 % w), (h // 2, w - 1)]
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
                    c0 = random.randrange(max(1, w - line_len + 1))
                    cells = [(rr, c0 + dc) for dc in range(line_len)]
                else:
                    r0 = random.randrange(max(1, h - line_len + 1))
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
        min_count = int(np.ceil(0.18 * len(border_cells)))
        max_count = int(np.floor(0.32 * len(border_cells)))
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


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    corner_a, corner_b = random.choice(DIAGONAL_PAIRS)

    all_colors = list(range(1, 10))
    ctrl_a, ctrl_b = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_a, ctrl_b}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(18, n - 5)
    max_core_w = min(18, n - 5)
    if max_core_h < 5 or max_core_w < 5:
        return None
    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    guide_corners = {corner_a: ctrl_a, corner_b: ctrl_b}
    aligned_object = _attach_guides(core, guide_corners)

    transforms = [(k, flip) for k in range(4) for flip in [False, True]]
    random.shuffle(transforms)
    input_object = None
    for k, flip in transforms:
        if k == 0 and not flip:
            continue
        candidate = _inverse_transform(aligned_object, k, flip)
        if _guide_corners(candidate) != guide_corners:
            input_object = candidate
            break
    if input_object is None:
        return None

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, guide_corners)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if _old_edge_signature_score(grid, bg=0) != 0:
        return None
    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d diagonal variant")
