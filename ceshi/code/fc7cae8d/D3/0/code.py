# Variant of fc7cae8d
# Dimension: D3 — 目标位置与框架结构 (dst)
# Change: 网格边框的两个相邻边为纯色控制边，另两个边为稀疏噪声（color 5），对象对齐到控制边所在角... → 控制边为对边而非相邻边
#
# Design analysis for this variant
# a) Concrete behavior:
#    The frame now has two opposite solid control edges, either top/bottom or left/right.
#    The object also carries two opposite guide edges with the same two colors. The solver
#    must align the object's guide axis to the frame's control axis, then crop off those
#    guide strips and return the inner core. Example:
#    - original: controls at top+left, object uses adjacent guides, so corner placement
#      determines one of eight dihedral transforms.
#    - variant: controls at top+bottom, object guides are also opposite; if the input
#      object shows left/right guides, the solver rotates it onto the vertical axis and
#      crops top+bottom. The decision is axis-based, not corner-based.
# b) Complexity calibration:
#    Opposite guides remove one orientation cue: a reflection parallel to the guide axis
#    keeps the same two guide sides and colors. To avoid underdetermined samples, the
#    generator uses only quarter-turn rotations, never the ambiguous reflections. The core
#    is non-square and keeps a fill-colored border so the axis change is visually obvious.
# c) Original assumptions that break:
#    - detect_controls_frame() in the seed only needed to find any two near-solid edges,
#      because adjacency encoded a corner. Here the solver must prefer an opposite pair.
#    - the seed searched all rotations/reflections; that relied on adjacent guide colors
#      uniquely pinning the transform. With opposite guides, unrestricted reflections could
#      produce the same visible guide configuration but a different core orientation.
#    - corner clearing in _attach_guides() was harmless for adjacent guides; it is no
#      longer relevant because opposite guides never meet at a corner.
# d) Placement legality:
#    The generated object is still placed strictly inside the frame with border=1, so it
#    never overlaps the border controls/noise. The object is a single connected component.
# e) Visual observability:
#    Samples always use a 90 or 270 degree input rotation relative to the aligned state,
#    so the object's guide axis is orthogonal to the frame's control axis in the input.
#    This makes the D3 change readable from a few examples rather than collapsing into a
#    subtle same-axis crop task.

from __future__ import annotations

from collections import Counter
import random
import sys

import numpy as np

ARC_UTILS_DIR = "/home/lixinyu/snap/new/arc2_solver_select"
if ARC_UTILS_DIR not in sys.path:
    sys.path.append(ARC_UTILS_DIR)

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location

OPPOSITE_PAIRS = (("top", "bottom"), ("left", "right"))


def _edge_slices(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }


def _detect_opposite_controls(arr: np.ndarray, bg: int) -> dict[str, int]:
    candidates: dict[str, tuple[int, int]] = {}
    for side, edge in _edge_slices(arr).items():
        vals = [int(x) for x in edge if int(x) != int(bg)]
        if not vals:
            continue
        color, count = Counter(vals).most_common(1)[0]
        if int(count) >= len(edge) - 1:
            candidates[side] = (int(color), int(count))

    for a, b in OPPOSITE_PAIRS:
        if a in candidates and b in candidates:
            return {a: candidates[a][0], b: candidates[b][0]}

    ranked = sorted(candidates.items(), key=lambda kv: kv[1][1], reverse=True)[:2]
    return {side: color for side, (color, _count) in ranked}


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
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


def _guide_edges(arr: np.ndarray) -> dict[str, int]:
    fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
    out: dict[str, int] = {}
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    for side, edge in edges.items():
        vals = [int(x) for x in edge]
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        color, count = Counter(non_fill).most_common(1)[0]
        if int(color) != int(fill) and int(count) >= len(vals) - 1:
            out[side] = int(color)
    return out


def _pick_opposite_guides(edges: dict[str, int], allowed_colors: set[int]) -> dict[str, int]:
    filtered = {side: color for side, color in edges.items() if int(color) in allowed_colors}
    for a, b in OPPOSITE_PAIRS:
        if a in filtered and b in filtered:
            return {a: filtered[a], b: filtered[b]}
    return filtered


def _crop_guides(arr: np.ndarray, sides: dict[str, int]) -> np.ndarray:
    r0 = 1 if "top" in sides else 0
    r1 = arr.shape[0] - (1 if "bottom" in sides else 0)
    c0 = 1 if "left" in sides else 0
    c1 = arr.shape[1] - (1 if "right" in sides else 0)
    return arr[r0:r1, c0:c1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align an object with opposite guide edges to opposite frame controls via rotation.

    This variant intentionally avoids reflection search: opposite-side guide colors do not
    uniquely reveal flips parallel to the guide axis, so the generator only emits quarter-
    turn rotations and the solver inverts that restricted transform family.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_opposite_controls(g, bg)
    obj = _extract_center_object(g, bg)
    allowed_colors = set(map(int, desired.values()))

    for k in range(4):
        rotated = np.rot90(obj, int(k))
        guides = _pick_opposite_guides(_guide_edges(rotated), allowed_colors)
        if set(guides) != set(desired):
            continue
        if all(int(guides[side]) == int(color) for side, color in desired.items()):
            return _crop_guides(rotated, desired)

    return obj


def _rotate_back(arr: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(arr, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    inner_rows = list(range(1, h - 1))
    inner_cols = list(range(1, w - 1))
    inner_cells = [(r, c) for r in inner_rows for c in inner_cols]
    random.shuffle(inner_cells)

    target_nonfill = random.randint(max(len(pattern_colors) + 2, len(inner_cells) // 8), max(len(pattern_colors) + 4, int(len(inner_cells) * 0.35)))
    painted = 0

    for color in pattern_colors:
        if not inner_cells:
            break
        r, c = inner_cells.pop()
        core[r, c] = int(color)
        painted += 1

    while painted < target_nonfill and inner_cells:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [random.choice(inner_cells)]
        elif shape_kind == "line":
            if random.random() < 0.5:
                rr = random.choice(inner_rows)
                line_len = random.randint(2, min(4, w - 2))
                c0 = random.randint(1, w - 1 - line_len)
                cells = [(rr, c0 + dc) for dc in range(line_len)]
            else:
                cc = random.choice(inner_cols)
                line_len = random.randint(2, min(4, h - 2))
                r0 = random.randint(1, h - 1 - line_len)
                cells = [(r0 + dr, cc) for dr in range(line_len)]
        else:
            mh = random.randint(1, min(3, h - 2))
            mw = random.randint(1, min(3, w - 2))
            r0 = random.randint(1, h - 1 - mh)
            c0 = random.randint(1, w - 1 - mw)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        for r, c in cells:
            if painted >= target_nonfill:
                break
            if core[r, c] == int(fill):
                core[r, c] = int(color)
                painted += 1

    return core


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int], bg: int = 0) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.full((h + has_top + has_bottom, w + has_left + has_right), int(bg), dtype=int)
    r0 = 1 if has_top else 0
    c0 = 1 if has_left else 0
    paste(out, core, r0, c0)

    if has_top:
        out[0, c0 : c0 + w] = int(guide_colors["top"])
    if has_bottom:
        out[-1, c0 : c0 + w] = int(guide_colors["bottom"])
    if has_left:
        out[r0 : r0 + h, 0] = int(guide_colors["left"])
    if has_right:
        out[r0 : r0 + h, -1] = int(guide_colors["right"])
    return out


def _paint_frame(grid: np.ndarray, control_sides: tuple[str, str], control_colors: dict[str, int]) -> None:
    n = grid.shape[0]
    for side in control_sides:
        color = int(control_colors[side])
        if side == "top":
            grid[0, :] = color
        elif side == "bottom":
            grid[-1, :] = color
        elif side == "left":
            grid[:, 0] = color
        else:
            grid[:, -1] = color

    for side in ("top", "bottom", "left", "right"):
        if side in control_sides:
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        for r, c in random.sample(coords, random.randint(min_count, max_count)):
            grid[r, c] = 5


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    control_sides = random.choice(OPPOSITE_PAIRS)

    all_colors = list(range(1, 10))
    ctrl_a, ctrl_b = random.sample([c for c in all_colors if c != 5], 2)
    control_colors = {control_sides[0]: ctrl_a, control_sides[1]: ctrl_b}
    remaining = [c for c in all_colors if c not in {5, ctrl_a, ctrl_b}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_core_h = min(16, n - 4)
    max_core_w = min(16, n - 4)
    if max_core_h < 6 or max_core_w < 6:
        return None

    for _ in range(20):
        core_h = random.randint(6, max_core_h)
        core_w = random.randint(6, max_core_w)
        if core_h != core_w:
            break
    else:
        return None

    core = _build_core(core_h, core_w, fill, pattern_colors)
    aligned_object = _attach_guides(core, control_sides, control_colors)

    # Always force an orthogonal-axis input so the D3 change is visible.
    input_object = _rotate_back(aligned_object, random.choice([1, 3]))

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, control_sides, control_colors)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d D3 variant")
