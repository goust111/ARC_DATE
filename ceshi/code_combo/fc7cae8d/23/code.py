# Combo variant of fc7cae8d (index 23)
# combo_spec: D3/1+D10/1
# Name: 非实心控制边+多色噪声
#
# Design analysis for this fused variant
# -------------------------------------
# a) Concrete behavior trace:
#    Original task:
#    - One horizontal frame edge and one vertical frame edge are almost perfectly solid
#      control colors.
#    - The interior object carries two 1px guide edges with matching colors.
#    - Solver can identify the frame controls by "near 100% same color", then try D4
#      transforms until the object's guide colors face the same sides, then crop those
#      guide strips away.
#    Fused variant:
#    - The two true control edges keep the same semantic role, but they are only 80-90%
#      filled with their dominant color, with visible gaps.
#    - The two noise edges are no longer single-color grey noise. They contain 35-60%
#      scattered pixels from multiple colors, and each noise edge includes both control
#      colors among those scattered pixels.
#    - Example with a 24x24 grid (inner side length 22):
#      original top control:    2222222222222222222222
#      fused top control:       2220222202222222220222   (18/22 = 0.82 dominant density)
#      original bottom noise:   0050055500505050055005
#      fused bottom noise:      2055075200550257055002   (mixed colors, dominant ratio low)
#      The object logic stays the same: guides are still solid strips. The changed step is
#      only "how to tell which frame sides are the true controls". The fused solver must
#      rank sides by dominant-color density instead of looking for a unique color or a
#      nearly perfect solid edge.
# b) Complexity calibration:
#    - If control fill were too low (<80%), the frame could look like four equally noisy
#      sides and the task would become perceptually unfair. If it were too high (>90%),
#      the original near-solid heuristic would still work too often. The generator keeps
#      control dominant density in [0.80, 0.90].
#    - If noise used too many colors or too high density, the border would become visually
#      messy. The generator limits noise to four colors total on each noise edge, with
#      each individual color staying far below control-side density.
#    - Both noise edges contain both control colors, so "find the unique side containing
#      color X" is invalid in every sample, not just occasionally.
# c) Implicit-assumption audit against the original solver:
#    - Original `detect_controls_frame()` assumed each control side has one dominant color
#      on all but at most one cell. D3/1 breaks that immediately; with 20 inner cells and
#      82% fill, the dominant count is 16-17 rather than 19.
#    - Original `detect_controls_frame()` also benefited from D10/0's grey-only noise:
#      if color 2 appears on the frame, it is probably a control. D10/1 breaks that,
#      because the noise edges also contain scattered control-color pixels.
#    - Original `extract_center_object()` assumes the interior target is still the largest
#      non-border multicolor component. That remains valid because this combo only changes
#      frame-side rendering; it does not add interior distractors.
#    - Original `guide_edges()` assumes the object guides are near-solid and the fill color
#      is the dominant object color. Those assumptions still hold; the generator keeps the
#      guide strips solid and caps interior pattern density so fill remains dominant.
#    - Original transform loop assumes that once the two desired sides are known, matching
#      guide colors is enough to select the correct D4 transform. That still holds because
#      guide colors themselves are unchanged.
# d) Placement legality:
#    - The generator still builds the aligned guided object first, applies an inverse D4
#      transform, and places it with `random_free_location(..., transparent=0, border=1)`.
#      This guarantees the object stays strictly inside the frame, does not overlap border
#      pixels, and remains a single 4-connected component.
#    - Noise lives only on border cells, so it cannot merge with the interior object.
# e) Visual observability:
#    - Across 2-4 examples, a human can still see the same high-level rule: the object's
#      colored guide sides should align to the matching colored frame sides.
#    - What changes visibly is that the "correct" frame sides now look like mostly-solid
#      broken lines, while the wrong sides look like sparse multicolor scatter. The contrast
#      is statistical and structural, not based on color exclusivity.
# f) Visual cleanliness:
#    - Border clutter is restricted to the border only.
#    - Each noise edge uses at most four colors and moderate density, preventing the border
#      from turning into an unreadable rainbow band.
#    - The interior object is still isolated from the frame by at least one background cell.
# g) Color inferability:
#    - No new color remapping is introduced. Guide colors still equal the matching control
#      colors. The human-visible novelty is purely "same colors can appear on noise edges,
#      but only the denser, more continuous sides are controls".
# h) Quantitative non-degeneracy target:
#    - For every generated sample, the minimum control-side dominant density must exceed the
#      maximum noise-side dominant density by a healthy margin.
#    - Both noise edges must contain both control colors.
#    - The original near-solid baseline solver must fail to reproduce the core; otherwise the
#      sample would have silently collapsed back toward the original task.
# i) New solver assumptions introduced here and enforced by the generator:
#    - Exactly one horizontal side and exactly one vertical side have the highest dominant
#      color density, and those are the true controls.
#    - Noise-side dominant-color densities remain well below control-side densities.
#    - Both control colors are distinct and both appear as solid guide strips on the object.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
UTILS_DIR = ROOT / "solvers"
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDE_ORDER = ("top", "bottom", "left", "right")
HORIZ_SIDES = ("top", "bottom")
VERT_SIDES = ("left", "right")


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _frame_side_arrays(arr: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "top": np.array(arr[0, 1:-1], dtype=int),
        "bottom": np.array(arr[-1, 1:-1], dtype=int),
        "left": np.array(arr[1:-1, 0], dtype=int),
        "right": np.array(arr[1:-1, -1], dtype=int),
    }


def _frame_side_stats(arr: np.ndarray, bg: int) -> dict[str, dict[str, object]]:
    stats: dict[str, dict[str, object]] = {}
    for side, values in _frame_side_arrays(arr).items():
        vals = list(map(int, values))
        counts = Counter(v for v in vals if v != int(bg))
        dom_color: int | None
        dom_count = 0
        if counts:
            dom_color, dom_count = counts.most_common(1)[0]
            dom_color = int(dom_color)
            dom_count = int(dom_count)
        else:
            dom_color = None
        length = len(vals)
        nonbg_count = int(sum(counts.values()))
        stats[side] = {
            "length": int(length),
            "counts": dict((int(k), int(v)) for k, v in counts.items()),
            "dom_color": dom_color,
            "dom_count": int(dom_count),
            "dom_density": float(dom_count / length) if length else 0.0,
            "nonbg_density": float(nonbg_count / length) if length else 0.0,
        }
    return stats


def _detect_controls_frame(arr: np.ndarray, bg: int) -> tuple[dict[str, int], dict[str, dict[str, object]]]:
    stats = _frame_side_stats(arr, int(bg))
    desired: dict[str, int] = {}
    for group in (HORIZ_SIDES, VERT_SIDES):
        ranked = sorted(
            group,
            key=lambda side: (
                float(stats[side]["dom_density"]),
                float(stats[side]["nonbg_density"]),
                int(stats[side]["dom_count"]),
            ),
            reverse=True,
        )
        best = ranked[0]
        dom_color = stats[best]["dom_color"]
        if dom_color is not None:
            desired[best] = int(dom_color)
    return desired, stats


def _baseline_detect_controls_frame(arr: np.ndarray, bg: int) -> list[tuple[str, int]]:
    cand: list[tuple[str, int, int, int]] = []
    for side, values in _frame_side_arrays(arr).items():
        vals = list(map(int, values))
        non_bg = [x for x in vals if x != int(bg)]
        if not non_bg:
            continue
        col, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            cand.append((side, int(col), int(count), int(len(vals))))
    cand.sort(key=lambda item: (item[2] / item[3], item[2]), reverse=True)
    return [(side, col) for side, col, _count, _length in cand[:2]]


def _extract_center_object(arr: np.ndarray, bg: int) -> np.ndarray:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    best: tuple[int, tuple[int, int, int, int]] | None = None
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
    edges = {
        "top": np.array(arr[0, :], dtype=int),
        "bottom": np.array(arr[-1, :], dtype=int),
        "left": np.array(arr[:, 0], dtype=int),
        "right": np.array(arr[:, -1], dtype=int),
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = list(map(int, values))
        non_fill = [x for x in vals if x != int(fill)]
        if not non_fill:
            continue
        col, count = Counter(non_fill).most_common(1)[0]
        if int(count) >= len(vals) - 1:
            out[side] = int(col)
    return out


def _crop_aligned_core(arr: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    top = 1 if "top" in desired else 0
    bottom = int(arr.shape[0]) - (1 if "bottom" in desired else 0)
    left = 1 if "left" in desired else 0
    right = int(arr.shape[1]) - (1 if "right" in desired else 0)
    return np.array(arr[int(top) : int(bottom), int(left) : int(right)], dtype=int)


def _solve_with_controls(grid: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    obj = _extract_center_object(g, int(bg))

    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            edges = _guide_edges(transformed)
            if all(edges.get(side) == int(col) for side, col in desired.items()):
                return _crop_aligned_core(transformed, desired)
    return obj


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read the denser horizontal and vertical frame sides as controls, align the
    object's solid guide edges to those sides via a dihedral transform, then crop
    off the aligned guide strips and return the core.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired, _stats = _detect_controls_frame(g, int(bg))
    return _solve_with_controls(g, desired)


def _baseline_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = {side: col for side, col in _baseline_detect_controls_frame(g, int(bg))}
    return _solve_with_controls(g, desired)


def _build_core(h: int, w: int, fill: int, pattern_colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    max_nonfill = max(len(pattern_colors) * 3, int(area * 0.32))
    target_nonfill = random.randint(max(len(pattern_colors) * 2, area // 12), max_nonfill)
    painted = 0

    for color in pattern_colors:
        for _ in range(2):
            r = random.randrange(h)
            c = random.randrange(w)
            if core[r, c] == int(fill):
                core[r, c] = int(color)
                painted += 1

    while painted < target_nonfill:
        color = random.choice(pattern_colors)
        shape_kind = random.choice(("dot", "line", "rect"))
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
        fresh = [(r, c) for r, c in cells if core[r, c] == int(fill)]
        if len(fresh) > remaining:
            fresh = random.sample(fresh, remaining)
        for r, c in fresh:
            core[r, c] = int(color)
        painted += len(fresh)

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


def _side_inner_coords(n: int, side: str) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(1, n - 1)]
    if side == "bottom":
        return [(n - 1, c) for c in range(1, n - 1)]
    if side == "left":
        return [(r, 0) for r in range(1, n - 1)]
    if side == "right":
        return [(r, n - 1) for r in range(1, n - 1)]
    raise ValueError(f"unknown side: {side}")


def _paint_sparse_control_side(grid: np.ndarray, side: str, color: int) -> dict[str, object]:
    coords = _side_inner_coords(int(grid.shape[0]), side)
    length = len(coords)
    min_count = int(np.ceil(0.80 * length))
    max_count = int(np.floor(0.90 * length))
    keep_count = random.randint(min_count, max_count)
    keep_ids = set(random.sample(range(length), keep_count))
    for idx, (r, c) in enumerate(coords):
        if idx in keep_ids:
            grid[r, c] = int(color)
    return {"side": side, "length": int(length), "kept": int(keep_count)}


def _paint_multicolor_noise_side(
    grid: np.ndarray,
    side: str,
    ctrl_h: int,
    ctrl_v: int,
    accent: int,
) -> dict[str, object]:
    coords = _side_inner_coords(int(grid.shape[0]), side)
    length = len(coords)
    min_count = int(np.ceil(0.35 * length))
    max_count = int(np.floor(0.60 * length))
    noise_count = random.randint(min_count, max_count)

    colors = [5, int(ctrl_h), int(ctrl_v), int(accent)]
    counts = {5: 2, int(ctrl_h): 2, int(ctrl_v): 2, int(accent): 1}
    max_per_color = max(3, int(np.floor(0.28 * length)))
    extra = noise_count - sum(counts.values())

    if extra < 0:
        raise ValueError("noise_count too small for mandatory multicolor noise")

    choices = list(colors)
    while extra > 0:
        color = random.choice(choices)
        if counts[color] >= max_per_color:
            choices = [c for c in colors if counts[c] < max_per_color]
            if not choices:
                break
            continue
        counts[color] += 1
        extra -= 1

    pool: list[int] = []
    for color in colors:
        pool.extend([int(color)] * int(counts[color]))
    random.shuffle(pool)

    chosen = random.sample(coords, len(pool))
    for (r, c), color in zip(chosen, pool):
        grid[r, c] = int(color)

    return {"side": side, "length": int(length), "counts": dict(counts)}


def _paint_frame(
    grid: np.ndarray,
    horiz_side: str,
    vert_side: str,
    ctrl_h: int,
    ctrl_v: int,
    accent: int,
) -> None:
    _paint_sparse_control_side(grid, horiz_side, int(ctrl_h))
    _paint_sparse_control_side(grid, vert_side, int(ctrl_v))

    for side in SIDE_ORDER:
        if side in (horiz_side, vert_side):
            continue
        _paint_multicolor_noise_side(grid, side, int(ctrl_h), int(ctrl_v), int(accent))

    n = int(grid.shape[0])
    corner_colors = {
        (0, 0): int(ctrl_v) if vert_side == "left" else (int(ctrl_h) if horiz_side == "top" else 0),
        (0, n - 1): int(ctrl_v) if vert_side == "right" else (int(ctrl_h) if horiz_side == "top" else 0),
        (n - 1, 0): int(ctrl_v) if vert_side == "left" else (int(ctrl_h) if horiz_side == "bottom" else 0),
        (n - 1, n - 1): int(ctrl_v) if vert_side == "right" else (int(ctrl_h) if horiz_side == "bottom" else 0),
    }
    for (r, c), color in corner_colors.items():
        grid[r, c] = int(color)


def _frame_metric_report(
    grid: np.ndarray,
    horiz_side: str,
    vert_side: str,
    ctrl_h: int,
    ctrl_v: int,
) -> dict[str, float | int]:
    bg = int(get_bg(grid))
    stats = _frame_side_stats(grid, int(bg))
    noise_sides = [side for side in SIDE_ORDER if side not in (horiz_side, vert_side)]
    control_min_dom = min(float(stats[horiz_side]["dom_density"]), float(stats[vert_side]["dom_density"]))
    noise_max_dom = max(float(stats[side]["dom_density"]) for side in noise_sides)
    noise_ctrl_pixels = min(
        min(int(stats[side]["counts"].get(int(ctrl_h), 0)), int(stats[side]["counts"].get(int(ctrl_v), 0)))
        for side in noise_sides
    )
    detected, _ = _detect_controls_frame(grid, int(bg))
    return {
        "control_min_dom": float(control_min_dom),
        "noise_max_dom": float(noise_max_dom),
        "dom_gap": float(control_min_dom - noise_max_dom),
        "noise_ctrl_pixels": int(noise_ctrl_pixels),
        "detected_ok": int(detected.get(horiz_side) == int(ctrl_h) and detected.get(vert_side) == int(ctrl_v)),
    }


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(23, 30)
    horiz_side, vert_side = random.choice(
        [("top", "left"), ("top", "right"), ("bottom", "left"), ("bottom", "right")]
    )

    all_colors = list(range(1, 10))
    ctrl_h, ctrl_v = random.sample([c for c in all_colors if c != 5], 2)
    remaining = [c for c in all_colors if c not in {5, ctrl_h, ctrl_v}]
    fill = random.choice(remaining)
    remaining = [c for c in remaining if c != fill]
    accent = random.choice(remaining)
    remaining = [c for c in remaining if c != accent]
    n_pattern_colors = random.randint(2, min(4, len(remaining)))
    pattern_colors = random.sample(remaining, n_pattern_colors)

    max_core_h = min(20, n - 3)
    max_core_w = min(20, n - 3)
    if max_core_h < 5 or max_core_w < 5:
        return None

    core_h = random.randint(5, max_core_h)
    core_w = random.randint(5, max_core_w)
    core = _build_core(core_h, core_w, int(fill), pattern_colors)

    guide_colors = {horiz_side: int(ctrl_h), vert_side: int(ctrl_v)}
    aligned_object = _attach_guides(core, (horiz_side, vert_side), guide_colors)

    k = random.randrange(4)
    flip = bool(random.randrange(2))
    input_object = _inverse_transform(aligned_object, k, flip)

    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, horiz_side, vert_side, int(ctrl_h), int(ctrl_v), int(accent))

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    fused = main(grid)
    if not np.array_equal(fused, core):
        return None

    baseline = _baseline_main(grid)
    if np.array_equal(baseline, core):
        return None

    metrics = _frame_metric_report(grid, horiz_side, vert_side, int(ctrl_h), int(ctrl_v))
    if int(metrics["detected_ok"]) != 1:
        return None
    if float(metrics["control_min_dom"]) < 0.80 or float(metrics["control_min_dom"]) > 0.90:
        return None
    if float(metrics["noise_max_dom"]) > 0.30:
        return None
    if float(metrics["dom_gap"]) < 0.48:
        return None
    if int(metrics["noise_ctrl_pixels"]) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D3/1+D10/1")
