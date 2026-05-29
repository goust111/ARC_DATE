# Combo variant of fc7cae8d (index 31)
# combo_spec: D9/2+D10/1
# Name: 控制像素+多色噪声

from __future__ import annotations

from collections import Counter
import random

import numpy as np

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


# Design analysis for the fused variant
# a) Concrete behavior:
#    Original: two almost-solid frame edges directly told the target sides, and noise lived only on the
#    two unused edges as color 5. A guide edge of color A simply had to match the solid border edge of
#    color A after a dihedral transform.
#    Fused: the object still has two full guide edges, but the frame no longer has solid controls. Instead:
#    - one isolated control pixel on a horizontal border has the left/right guide color; its horizontal
#      position encodes whether that guide must end up on the left half or right half of the object;
#    - one isolated control pixel on a vertical border has the top/bottom guide color; its vertical
#      position encodes whether that guide must end up on the top half or bottom half of the object.
#    All four frame edges also contain multi-color noise clumps using {5, ctrl_tb, ctrl_lr}, so color
#    uniqueness is gone. Example: a red singleton near the right end of the top row means "red guide -> right";
#    a blue singleton near the bottom of the left column means "blue guide -> bottom". The solver must locate
#    the sparse singletons, decode the target sides, align the object, and then crop off those aligned guides.
# b) Complexity calibration:
#    Only two discrete positional buckets per axis are used (left/right and top/bottom), so the rule is
#    inferable from a few samples. Noise is visually different because it appears in short contiguous runs,
#    while controls are guaranteed isolated by background gaps on both sides. This keeps the fusion visible
#    without adding a hard-to-learn lookup table.
# c) Implicit-assumption audit against the original main():
#    - detect_controls_frame() previously assumed a whole border side was nearly monochrome. That is false now,
#      so it is replaced with singleton detection on border lines.
#    - extract_center_object() assumed the central object is the largest non-border connected component. This
#      still holds because border noise stays on the outer frame and the interior object remains the only large
#      interior component.
#    - guide_edges() assumed the object's guide edges are nearly full monochrome edges. This still holds because
#      the fusion does not alter the object-side encoding.
#    - the transform search assumed the desired mapping is two orthogonal target sides with distinct colors.
#      The generator preserves that: exactly one left/right target and exactly one top/bottom target.
# d) Placement legality:
#    The generator paints controls/noise only on the outer border, reserves a 1-cell gap around each control
#    on its border line so the singleton remains isolated, and places the interior object with border=1 so it
#    never overlaps the frame.
# e) Visual observability:
#    Across samples, the carrier edge can vary independently of the target side, so the human-visible cue is
#    not "which edge hosts the pixel" but "which side of that edge the singleton sits near". The contrast
#    between isolated singletons and noisy clumps provides a stable visual handle.
# f) Visual cleanliness:
#    Noise is limited to short border runs; the interior object remains separated from the frame by background.
#    This avoids ambiguous overlap between the instruction channel and the object channel.
# g) Color-change inferability:
#    Border noise uses only three colors at most: 5 plus the two control/guide colors. Because the same guide
#    colors recur both on the object and in noise, color alone is insufficient by design; position + isolation
#    supplies the missing structure.


def main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))

    def detect_controls(arr: np.ndarray) -> dict[str, int]:
        sides = {
            "top": arr[0, 1:-1],
            "bottom": arr[-1, 1:-1],
            "left": arr[1:-1, 0],
            "right": arr[1:-1, -1],
        }
        horiz_candidates: list[tuple[float, str, int]] = []
        vert_candidates: list[tuple[float, str, int]] = []

        for side, line in sides.items():
            vals = list(map(int, line))
            L = len(vals)
            if L == 0:
                continue
            mid = (L - 1) / 2.0
            for idx, col in enumerate(vals):
                if int(col) == int(bg):
                    continue
                prev_bg = idx == 0 or int(vals[idx - 1]) == int(bg)
                next_bg = idx == L - 1 or int(vals[idx + 1]) == int(bg)
                if not (prev_bg and next_bg):
                    continue
                clarity = abs(float(idx) - mid)
                if side in ("top", "bottom"):
                    target = "left" if idx < L / 2 else "right"
                    horiz_candidates.append((clarity, target, int(col)))
                else:
                    target = "top" if idx < L / 2 else "bottom"
                    vert_candidates.append((clarity, target, int(col)))

        if not horiz_candidates or not vert_candidates:
            raise ValueError("missing isolated control pixels")

        best_lr = max(horiz_candidates, key=lambda t: t[0])
        best_tb = max(vert_candidates, key=lambda t: t[0])
        return {str(best_lr[1]): int(best_lr[2]), str(best_tb[1]): int(best_tb[2])}

    def extract_center_object(arr: np.ndarray) -> np.ndarray:
        H, W = arr.shape
        comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
        best = None
        for comp in comps:
            r0, c0, r1, c1 = map(int, comp["bbox"])
            if r0 == 0 or c0 == 0 or r1 == H - 1 or c1 == W - 1:
                continue
            area = int(comp["area"])
            if best is None or area > best[0]:
                best = (area, (r0, c0, r1, c1))
        if best is None:
            return arr.copy()
        return crop_bbox(arr, best[1])

    def guide_edges(arr: np.ndarray) -> dict[str, int]:
        fill = int(Counter(map(int, arr.flatten())).most_common(1)[0][0])
        edges = {
            "top": arr[0, :],
            "bottom": arr[-1, :],
            "left": arr[:, 0],
            "right": arr[:, -1],
        }
        out: dict[str, int] = {}
        for side, line in edges.items():
            vals = list(map(int, line))
            non_fill = [x for x in vals if int(x) != int(fill)]
            if not non_fill:
                continue
            col, count = Counter(non_fill).most_common(1)[0]
            if int(count) >= len(vals) - 1:
                out[str(side)] = int(col)
        return out

    desired = detect_controls(g)
    obj = extract_center_object(g)

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            t = np.fliplr(base) if flip else base
            gedges = guide_edges(t)
            if all(side in gedges and int(gedges[side]) == int(col) for side, col in desired.items()):
                top = 1 if "top" in desired else 0
                bottom = int(t.shape[0]) - (1 if "bottom" in desired else 0)
                left = 1 if "left" in desired else 0
                right = int(t.shape[1]) - (1 if "right" in desired else 0)
                return t[int(top) : int(bottom), int(left) : int(right)]

    return obj


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
        if int(core[r, c]) == int(fill):
            painted += 1
        core[r, c] = int(color)

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
        fill_cells = [(r, c) for r, c in cells if int(core[r, c]) == int(fill)]
        if len(fill_cells) > remaining:
            fill_cells = random.sample(fill_cells, remaining)
        for r, c in fill_cells:
            core[r, c] = int(color)
        painted += len(fill_cells)

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


def _edge_coords(side: str, n: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, n - 1)]
    if side == "bottom":
        return [(n - 1, c) for c in range(1, n - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, n - 1)]
    return [(r, n - 1) for r in range(1, n - 1)]


def _pick_control_coord(side: str, target_side: str, n: int) -> tuple[int, int]:
    inner = n - 2
    low_band_end = max(3, inner // 3)
    high_band_start = inner - low_band_end + 1
    if side in ("top", "bottom"):
        if target_side == "left":
            idx = random.randint(1, low_band_end)
        else:
            idx = random.randint(high_band_start, inner)
        return (0 if side == "top" else n - 1, idx)

    if target_side == "top":
        idx = random.randint(1, low_band_end)
    else:
        idx = random.randint(high_band_start, inner)
    return (idx, 0 if side == "left" else n - 1)


def _paint_noise_runs(
    grid: np.ndarray,
    side: str,
    palette: list[int],
    forbidden: set[tuple[int, int]],
    density_range: tuple[float, float] = (0.35, 0.55),
) -> None:
    coords = _edge_coords(side, grid.shape[0])
    allowed_idx = [i for i, rc in enumerate(coords) if rc not in forbidden]
    if not allowed_idx:
        return

    target = random.randint(
        max(2, int(np.ceil(density_range[0] * len(allowed_idx)))),
        max(2, int(np.floor(density_range[1] * len(allowed_idx)))),
    )
    filled = set()
    attempts = 0

    while len(filled) < target and attempts < 400:
        attempts += 1
        start = random.choice(allowed_idx)
        if start in filled:
            continue
        run = [start]
        j = start + 1
        while j in allowed_idx and j not in filled and len(run) < 4:
            run.append(j)
            j += 1
        if len(run) < 2:
            continue
        run_len = random.randint(2, len(run))
        use = run[:run_len]
        color = int(random.choice(palette))
        for idx in use:
            r, c = coords[idx]
            grid[r, c] = int(color)
            filled.add(idx)


def _paint_frame_with_controls(
    grid: np.ndarray,
    target_tb: str,
    target_lr: str,
    color_tb: int,
    color_lr: int,
) -> None:
    n = grid.shape[0]
    side_lr = random.choice(["top", "bottom"])
    side_tb = random.choice(["left", "right"])

    ctrl_lr = _pick_control_coord(side_lr, target_lr, n)
    ctrl_tb = _pick_control_coord(side_tb, target_tb, n)

    forbidden: set[tuple[int, int]] = set()
    for side, ctrl in ((side_lr, ctrl_lr), (side_tb, ctrl_tb)):
        r, c = ctrl
        forbidden.add((int(r), int(c)))
        coords = _edge_coords(side, n)
        for i, rc in enumerate(coords):
            if rc != (int(r), int(c)):
                continue
            if i > 0:
                forbidden.add(coords[i - 1])
            if i + 1 < len(coords):
                forbidden.add(coords[i + 1])
            break

    noise_palette = [5, int(color_tb), int(color_lr)]
    for side in ("top", "bottom", "left", "right"):
        _paint_noise_runs(grid, side, noise_palette, forbidden)

    grid[ctrl_lr] = int(color_lr)
    grid[ctrl_tb] = int(color_tb)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    target_tb = random.choice(["top", "bottom"])
    target_lr = random.choice(["left", "right"])

    all_colors = list(range(1, 10))
    color_tb, color_lr = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, color_tb, color_lr}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    pattern_colors = random.sample(remaining, random.randint(2, min(4, len(remaining))))

    max_core_h = min(20, n - 3)
    max_core_w = min(20, n - 3)
    if max_core_h < 5 or max_core_w < 5:
        return None
    core = _build_core(
        random.randint(5, max_core_h),
        random.randint(5, max_core_w),
        fill,
        pattern_colors,
    )

    guide_colors = {target_tb: color_tb, target_lr: color_lr}
    aligned_object = _attach_guides(core, (target_tb, target_lr), guide_colors)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame_with_controls(grid, target_tb, target_lr, color_tb, color_lr)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, core):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D9/2+D10/1")
