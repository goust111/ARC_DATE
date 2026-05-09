# Combo variant of fc7cae8d (index 5)
# combo_spec: D1/4+D3/2
# Name: 频率消歧+三控制边
#
# Design analysis for this combo
# a) Concrete behavior / trace:
#    - Seed task: two adjacent frame controls + two matching guide edges. One D4 transform
#      aligns them, then those guide strips are cropped away.
#    - Combo task: the frame has three solid control sides and one noisy side. The object has
#      three true guide sides and one unguided side.
#    - To make D1/4 meaningful together with three controls, the unguided side of the aligned
#      object is turned into a decoy: its exposed border copies the color of the opposite
#      control side. Under one axial flip, that decoy side can masquerade as a real guide, so
#      edge-color matching alone yields two valid transforms.
#    - Example: active frame sides = top:red, left:blue, right:yellow, noisy side = bottom.
#      The aligned object really has top/left/right guides in red/blue/yellow, but its bottom
#      exposed edge is also painted red. Therefore both:
#        1. identity (real top guide on top), and
#        2. vertical flip (fake bottom-red edge now appears on top)
#      satisfy the three side-color constraints.
#      The output is chosen by cropping each candidate and taking the one whose non-fill
#      dominant color has the highest frequency. The generator makes the copied control color
#      also be the dominant core color, and the correct orientation keeps two extra corner
#      cells of that color, so the frequency score is uniquely higher.
# b) Complexity calibration:
#    - Three frame controls make the change from the seed visually obvious.
#    - The ambiguity is limited to one deliberate flip candidate, not arbitrary many matches.
#    - The dominant non-fill color is repeated in a thick band plus a few motifs, so the
#      frequency tiebreak is visible rather than a 1-pixel accident.
# c) Hidden assumptions reviewed from the seed:
#    - `detect_controls_frame()` in the seed assumed only two useful sides. Here it must rank
#      all four border sides and keep the top three near-solid controls.
#    - `guide_edges()` assumed a guide edge is almost monochrome and that the fill is the
#      dominant object color. That still holds; the generator keeps fill dominant and the fake
#      guide side is also almost monochrome by construction.
#    - The seed returned the first matching transform. That would silently collapse this combo,
#      because every accepted sample intentionally has at least two color-matching transforms.
#      The solver must therefore score all matching candidates and apply the frequency rule.
#    - New solver assumption: the unique best candidate is the one with the largest count of
#      its most frequent non-fill color after cropping. The generator explicitly checks this.
# d) Placement legality:
#    - The frame is painted first, the object is then placed strictly inside with `border=1`,
#      so it cannot overlap or touch frame controls.
#    - The object remains one connected multicolor component; no other interior components are
#      generated.
# e) Visual observability:
#    - Humans can see three border controls instead of a control corner.
#    - The unguided side visibly carries the same color as one control side, so two mirror
#      readings are plausible from edge colors alone.
#    - Across examples, the kept output is always the crop preserving the larger dominant-color
#      mass, so the second-stage frequency rule is inferable.
# f) Visual cleanliness:
#    - Only one interior object is present.
#    - Noise is confined to the single inactive frame side, and the object is separated from
#      the frame by at least one background cell.
#    - Internal motifs are kept sparse so the dominant band and three guide sides stay legible.
# g) Color-rule interpretability:
#    - No arbitrary color remapping is introduced. The same control color that creates the
#      false guide candidate also becomes the dominant core-pattern color used for tiebreaking.
#      The frequency rule therefore depends on a small, repeated palette cue rather than on
#      opaque color IDs.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
ARC_UTILS_DIR = ROOT / "solvers"
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


SIDES = ("top", "bottom", "left", "right")
MISSING_TO_ANCHOR = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _edge_inner(arr: np.ndarray, side: str) -> np.ndarray:
    if side == "top":
        return arr[0, 1:-1]
    if side == "bottom":
        return arr[-1, 1:-1]
    if side == "left":
        return arr[1:-1, 0]
    return arr[1:-1, -1]


def _detect_controls_frame(arr: np.ndarray, bg: int) -> dict[str, int]:
    candidates: list[tuple[str, int, int, int]] = []
    for side in SIDES:
        edge = [int(x) for x in _edge_inner(arr, side)]
        non_bg = [x for x in edge if x != int(bg)]
        if not non_bg:
            continue
        color, count = Counter(non_bg).most_common(1)[0]
        if int(count) >= len(edge) - 1:
            candidates.append((side, int(color), int(count), int(len(edge))))
    candidates.sort(key=lambda t: (t[2] / t[3], t[2]), reverse=True)
    return {side: color for side, color, _count, _len_edge in candidates[:3]}


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


def _object_fill(arr: np.ndarray, bg: int = 0) -> int:
    if arr.shape[0] > 2 and arr.shape[1] > 2:
        inner = arr[1:-1, 1:-1]
        vals = [int(x) for x in inner.flatten() if int(x) != int(bg)]
        if vals:
            return int(Counter(vals).most_common(1)[0][0])
    vals = [int(x) for x in arr.flatten() if int(x) != int(bg)]
    if not vals:
        return int(bg)
    return int(Counter(vals).most_common(1)[0][0])


def _guide_edges(arr: np.ndarray, bg: int = 0) -> dict[str, int]:
    fill = _object_fill(arr, bg)
    out: dict[str, int] = {}
    for side in SIDES:
        edge = [int(x) for x in _edge_inner(arr, side)]
        signal = [x for x in edge if x not in {int(fill), int(bg)}]
        if not signal:
            continue
        color, count = Counter(signal).most_common(1)[0]
        if int(count) >= len(edge) - 1:
            out[side] = int(color)
    return out


def _crop_guides(arr: np.ndarray, sides: dict[str, int]) -> np.ndarray:
    r0 = 1 if "top" in sides else 0
    r1 = arr.shape[0] - (1 if "bottom" in sides else 0)
    c0 = 1 if "left" in sides else 0
    c1 = arr.shape[1] - (1 if "right" in sides else 0)
    return arr[r0:r1, c0:c1]


def _core_score(core: np.ndarray) -> tuple[int, int]:
    fill = int(Counter(map(int, core.flatten())).most_common(1)[0][0])
    counts = Counter(int(x) for x in core.flatten() if int(x) != int(fill))
    if not counts:
        return (0, 0)
    color, count = max(counts.items(), key=lambda kv: (int(kv[1]), -int(kv[0])))
    return int(count), -int(color)


def _matching_candidates(obj: np.ndarray, desired: dict[str, int], bg: int = 0) -> list[dict[str, object]]:
    matches: list[dict[str, object]] = []
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed, bg=bg)
            if not all(side in guides for side in desired):
                continue
            if any(int(guides[side]) != int(color) for side, color in desired.items()):
                continue
            core = _crop_guides(transformed, desired)
            matches.append(
                {
                    "k": int(k),
                    "flip": bool(flip),
                    "transformed": transformed,
                    "core": core,
                    "score": _core_score(core),
                }
            )
    return matches


def _best_candidate(obj: np.ndarray, desired: dict[str, int], bg: int = 0) -> dict[str, object] | None:
    matches = _matching_candidates(obj, desired, bg=bg)
    if not matches:
        return None
    return max(matches, key=lambda m: m["score"])


def _naive_first_candidate(obj: np.ndarray, desired: dict[str, int], bg: int = 0) -> dict[str, object] | None:
    matches = _matching_candidates(obj, desired, bg=bg)
    return matches[0] if matches else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Align the three-guide rectangle to three frame controls.

    If edge colors alone admit several D4 transforms, choose the candidate whose cropped core
    has the highest-frequency non-fill color occurring most often.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    desired = _detect_controls_frame(g, bg)
    obj = _extract_center_object(g, bg)

    best = _best_candidate(obj, desired, bg=bg)
    if best is not None:
        return np.array(best["core"], dtype=int)
    return obj


def _other_colors(exclude: set[int], count: int) -> list[int]:
    choices = [c for c in range(1, 10) if c not in exclude]
    random.shuffle(choices)
    return choices[:count]


def _build_core(h: int, w: int, fill: int, dom_color: int, accent_colors: list[int], missing_side: str) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)

    if missing_side in ("top", "bottom"):
        band_rows = [0, 1] if missing_side == "top" else [h - 2, h - 1]
        for r in band_rows:
            core[r, :] = int(dom_color)
        inner_span = slice(1, w - 1) if w > 4 else slice(0, w)
        extra_r = 2 if missing_side == "top" and h > 4 else h - 3
        if 0 <= extra_r < h:
            core[extra_r, inner_span] = int(dom_color)
    else:
        band_cols = [0, 1] if missing_side == "left" else [w - 2, w - 1]
        for c in band_cols:
            core[:, c] = int(dom_color)
        inner_span = slice(1, h - 1) if h > 4 else slice(0, h)
        extra_c = 2 if missing_side == "left" and w > 4 else w - 3
        if 0 <= extra_c < w:
            core[inner_span, extra_c] = int(dom_color)

    motifs: list[tuple[int, list[tuple[int, int]]]] = []
    if accent_colors:
        a = int(accent_colors[0])
        motifs.append((a, [(1, 1), (1, min(w - 2, 3)), (min(h - 2, 3), 1)]))
    if len(accent_colors) >= 2:
        b = int(accent_colors[1])
        motifs.append((b, [(h // 2, w // 2), (h // 2, min(w - 2, w // 2 + 1)), (min(h - 2, h // 2 + 1), w // 2)]))
    if len(accent_colors) >= 3:
        c = int(accent_colors[2])
        motifs.append((c, [(max(1, h - 3), max(1, w - 3)), (max(1, h - 4), max(1, w - 2))]))

    for color, cells in motifs:
        for r, c in cells:
            if 0 <= r < h and 0 <= c < w and int(core[r, c]) == int(fill):
                core[r, c] = int(color)

    return core


def _attach_guides(core: np.ndarray, guide_sides: tuple[str, ...], guide_colors: dict[str, int]) -> np.ndarray:
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
    return out


def _notch_anchor_guide(arr: np.ndarray, anchor_side: str) -> None:
    if anchor_side == "top":
        arr[0, arr.shape[1] // 2] = 0
    elif anchor_side == "bottom":
        arr[-1, arr.shape[1] // 2] = 0
    elif anchor_side == "left":
        arr[arr.shape[0] // 2, 0] = 0
    else:
        arr[arr.shape[0] // 2, -1] = 0


def _extend_decoy_corners(arr: np.ndarray, missing_side: str, color: int) -> None:
    if missing_side == "top":
        arr[0, 0] = int(color)
        arr[0, -1] = int(color)
    elif missing_side == "bottom":
        arr[-1, 0] = int(color)
        arr[-1, -1] = int(color)
    elif missing_side == "left":
        arr[0, 0] = int(color)
        arr[-1, 0] = int(color)
    else:
        arr[0, -1] = int(color)
        arr[-1, -1] = int(color)


def _paint_frame(grid: np.ndarray, desired: dict[str, int], noise_color: int = 5) -> None:
    n = grid.shape[0]
    for side, color in desired.items():
        if side == "top":
            grid[0, :] = int(color)
        elif side == "bottom":
            grid[-1, :] = int(color)
        elif side == "left":
            grid[:, 0] = int(color)
        else:
            grid[:, -1] = int(color)

    missing_side = next(side for side in SIDES if side not in desired)
    if missing_side in ("top", "bottom"):
        r = 0 if missing_side == "top" else n - 1
        coords = [(r, c) for c in range(1, n - 1)]
    else:
        c = 0 if missing_side == "left" else n - 1
        coords = [(r, c) for r in range(1, n - 1)]
    min_count = int(np.ceil(0.35 * len(coords)))
    max_count = int(np.floor(0.60 * len(coords)))
    for r, c in random.sample(coords, random.randint(min_count, max_count)):
        grid[r, c] = int(noise_color)


def _expected_aligned_core(aligned_object: np.ndarray, desired: dict[str, int]) -> np.ndarray:
    return _crop_guides(aligned_object, desired)


def _gen_attempt() -> np.ndarray | None:
    n = random.randint(22, 29)
    missing_side = random.choice(SIDES)
    active_sides = tuple(side for side in SIDES if side != missing_side)

    ctrl_colors = random.sample([c for c in range(1, 10) if c != 5], 3)
    desired = {side: color for side, color in zip(active_sides, ctrl_colors)}
    anchor_side = MISSING_TO_ANCHOR[missing_side]
    dom_color = int(desired[anchor_side])

    fill_choices = [c for c in range(1, 10) if c not in set(ctrl_colors) | {5}]
    fill = random.choice(fill_choices)
    accent_colors = _other_colors({5, fill, *ctrl_colors}, count=3)

    max_core_h = min(18, n - 4)
    max_core_w = min(18, n - 4)
    if max_core_h < 6 or max_core_w < 6:
        return None
    core_h = random.randint(7, max_core_h)
    core_w = random.randint(7, max_core_w)
    core = _build_core(core_h, core_w, fill, dom_color, accent_colors, missing_side)

    aligned_object = _attach_guides(core, active_sides, desired)
    _notch_anchor_guide(aligned_object, anchor_side)
    _extend_decoy_corners(aligned_object, missing_side, dom_color)
    expected = _expected_aligned_core(aligned_object, desired)

    viable_inputs: list[np.ndarray] = []
    for k in range(4):
        for flip in (False, True):
            input_object = _inverse_transform(aligned_object, k, flip)
            matches = _matching_candidates(input_object, desired, bg=0)
            if len(matches) < 2:
                continue
            ordered = sorted(matches, key=lambda m: (m["k"], int(bool(m["flip"]))))
            best = max(matches, key=lambda m: m["score"])
            top_score = best["score"]
            if sum(1 for m in matches if m["score"] == top_score) != 1:
                continue
            if not np.array_equal(np.array(best["core"], dtype=int), expected):
                continue
            if np.array_equal(np.array(ordered[0]["core"], dtype=int), expected):
                continue
            viable_inputs.append(np.array(input_object, dtype=int))

    if not viable_inputs:
        return None

    input_object = random.choice(viable_inputs)
    grid = np.zeros((n, n), dtype=int)
    _paint_frame(grid, desired)

    try:
        r0, c0 = random_free_location(grid, input_object, transparent=0, border=1)
    except ValueError:
        return None
    paste(grid, input_object, r0, c0, transparent=0)

    if not np.array_equal(main(grid), expected):
        return None

    bg = int(get_bg(grid))
    obj = _extract_center_object(grid, bg)
    naive = _naive_first_candidate(obj, desired, bg=bg)
    best = _best_candidate(obj, desired, bg=bg)
    if naive is None or best is None:
        return None
    if np.array_equal(np.array(naive["core"], dtype=int), np.array(best["core"], dtype=int)):
        return None
    if len(_detect_controls_frame(grid, bg)) != 3:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo D1/4+D3/2")
