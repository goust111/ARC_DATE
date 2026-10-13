from __future__ import annotations
from collections import Counter
import numpy as np
from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location
import random
from arc_utils import paste, random_free_location

def main(grid: np.ndarray) -> np.ndarray:
    """
    Align a central rectangle to two colored frame-edge controls via rotation/reflection.

    Input structure:
    - bg grid with two solid-colored border edges (distinct non-bg colors).
    - A central filled rectangle with 1px colored guide edges on two sides, colors matching the frame controls.

    Rule:
    - Rotate/mirror the object so its guide edges face the same sides as the matching frame controls.
    - Crop off the 1px guide edges from aligned sides.

    Key insight: Guide-edge colors encode target frame sides; the correct dihedral transform matches both simultaneously.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def detect_controls_frame(arr: np.ndarray) -> list[tuple[str, int]]:
        H, W = arr.shape
        sides = {
            "top": arr[0, 1:-1],
            "bottom": arr[-1, 1:-1],
            "left": arr[1:-1, 0],
            "right": arr[1:-1, -1],
        }
        cand: list[tuple[str, int, int, int]] = []
        for side, v in sides.items():
            v = list(map(int, v))
            v_nb = [x for x in v if x != int(bg)]
            if not v_nb:
                continue
            col, count = Counter(v_nb).most_common(1)[0]
            # allow up to 1 mismatch (overlap near corners with the other control color)
            if int(count) >= len(v) - 1:
                cand.append((side, int(col), int(count), int(len(v))))
        cand.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
        return [(s, c) for s, c, _cnt, _L in cand[:2]]

    def extract_center_object(arr: np.ndarray) -> np.ndarray:
        H, W = arr.shape
        comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
        best = None  # (area, bbox)
        for comp in comps:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if bool(r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1):
                continue
            area = int(comp["area"])
            if best is None or area > best[0]:
                best = (area, (r0, c0, r1, c1))
        if best is None:
            return arr.copy()
        _area, b = best
        return crop_bbox(arr, b)

    def guide_edges(arr: np.ndarray) -> dict[str, int]:
        fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
        edges = {
            "top": arr[0, :],
            "bottom": arr[-1, :],
            "left": arr[:, 0],
            "right": arr[:, -1],
        }
        out: dict[str, int] = {}
        for side, v in edges.items():
            v = list(map(int, v))
            v_nf = [x for x in v if x != int(fill)]
            if not v_nf:
                continue
            col, count = Counter(v_nf).most_common(1)[0]
            if int(count) >= len(v) - 1:
                out[side] = int(col)
        return out

    ctrls = detect_controls_frame(g)
    desired = {side: col for side, col in ctrls}
    obj = extract_center_object(g)

    # Try all dihedral transforms: rotations + optional horizontal flip.
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in [False, True]:
            t = np.fliplr(base) if flip else base
            gedges = guide_edges(t)
            if all(side in gedges and int(gedges[side]) == int(col) for side, col in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                return t[int(top) : int(bottom), int(left) : int(right)]

    # Fallback: should not happen on this puzzle.
    return obj

# --------------- generator (Codex#2) ---------------

def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors), int(area * 0.35))
    target_nonfill = random.randint(max(len(pattern_colors), area // 10), max_nonfill)
    painted = 0

    for color in pattern_colors:
        r = random.randrange(h)
        c = random.randrange(w)
        core[r, c] = int(color)
        painted += 1

    while painted < target_nonfill:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(["dot", "line", "rect"])
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
        else:
            mh = random.randint(1, min(3, h))
            mw = random.randint(1, min(3, w))
            r0 = random.randrange(h - mh + 1)
            c0 = random.randrange(w - mw + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(mh) for dc in range(mw)]

        remaining = target_nonfill - painted
        if remaining <= 0:
            break
        fill_cells = [(r, c) for r, c in cells if core[r, c] == fill]
        if len(fill_cells) > remaining:
            fill_cells = random.sample(fill_cells, remaining)
        for r, c in fill_cells:
            core[r, c] = int(color)
        painted += len(fill_cells)
        for r, c in cells:
            if core[r, c] != fill:
                continue
            if painted >= target_nonfill:
                break
            core[r, c] = int(color)
            painted += 1

    return core


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, str], guide_colors: dict[str, int]) -> np.ndarray:
    h, w = core.shape
    has_top = "top" in guide_sides
    has_bottom = "bottom" in guide_sides
    has_left = "left" in guide_sides
    has_right = "right" in guide_sides
    out = np.zeros((h + has_top + has_bottom, w + has_left + has_right), dtype=int)
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

    if has_top and has_left:
        out[0, 0] = 0
    if has_top and has_right:
        out[0, -1] = 0
    if has_bottom and has_left:
        out[-1, 0] = 0
    if has_bottom and has_right:
        out[-1, -1] = 0
    return out


def _paint_frame(grid: np.ndarray, horiz_side: str, vert_side: str, ctrl_h: int, ctrl_v: int) -> None:
    n = grid.shape[0]
    if horiz_side == "top":
        grid[0, :] = int(ctrl_h)
    else:
        grid[-1, :] = int(ctrl_h)
    if vert_side == "left":
        grid[:, 0] = int(ctrl_v)
    else:
        grid[:, -1] = int(ctrl_v)

    for side in ("top", "bottom", "left", "right"):
        if side in (horiz_side, vert_side):
            continue
        if side in ("top", "bottom"):
            r = 0 if side == "top" else n - 1
            coords = [(r, c) for c in range(1, n - 1)]
        else:
            c = 0 if side == "left" else n - 1
            coords = [(r, c) for r in range(1, n - 1)]
        min_count = int(np.ceil(0.35 * len(coords)))
        max_count = int(np.floor(0.60 * len(coords)))
        noise_count = random.randint(min_count, max_count)
        for r, c in random.sample(coords, noise_count):
            grid[r, c] = 5


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(20, n - 3)
    max_core_w = min(20, n - 3)
    if max_core_h < 5 or max_core_w < 5:
        return None
    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, fill, pattern_colors)

    guide_colors = {horiz_side: ctrl_h, vert_side: ctrl_v}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, ctrl_h, ctrl_v)

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
    raise RuntimeError("failed to generate input for fc7cae8d")
