# Combo variant of fc7cae8d (index 37)
# combo_spec: D1/4+D2/3+D10/0
# Name: 频率消歧+多源+干扰
#
# Design analysis for the combo variant
# a) Concrete behavior and trace:
#    - Original task: one interior guided rectangle, two frame controls, and a single dihedral transform whose
#      guide-edge colors land on the two controlled frame sides. Crop the matched guide strips and return the core.
#    - Combo task: the frame is now a full 4-sided frame, but opposite sides repeat colors:
#         top == bottom == horizontal_control
#         left == right == vertical_control
#      so guide colors alone only tell "one horizontal side + one vertical side". For any true object, up to four
#      corner placements can satisfy the color rule: TL / TR / BL / BR.
#    - Each true object still has exactly two adjacent 1px guide strips, one horizontal-control color and one
#      vertical-control color. However, its core contains a highest-frequency non-fill color in a fixed canonical
#      orientation. That dominant color is the tie-break cue: among all guide-color-valid transforms, choose the one
#      whose cropped core restores that dominant-color motif to the same absolute orientation.
#    - Input contains 2-3 such true guided rectangles plus 1-2 plain distractor rectangles without guide strips.
#      Solver must first reject distractors, then resolve each true object's corner by the dominant-color tie-break,
#      then assemble the cropped cores into a 2x2 corner collage.
#    - Small trace:
#         old: "largest interior object -> one transform matches frame colors -> crop -> return"
#         combo: "scan all interior components -> only guided ones yield horizontal+vertical guide colors matching
#         frame colors -> each yields several valid corners because opposite frame sides share colors -> use the
#         most frequent non-fill core color to decide which candidate restores the canonical core orientation ->
#         crop guides -> place into that corner slot in the output collage"
# b) Complexity calibration:
#    - Using repeated opposite-side frame colors creates genuine ambiguity without making the frame unreadable.
#      A human can still see the two frame control colors, but color-only reasoning no longer determines the corner.
#    - The dominant non-fill color is made visibly frequent and concentrated in a fixed top-left motif, so the
#      tie-break is not a hidden numeric trick. It is a coarse visual motif that repeats across samples.
#    - True objects are 2-3 and corners are unique, so the multi-object rule is visible. Distractors are present in
#      every sample and are often comparable in size, so "pick the biggest interior rectangle" fails visibly.
# c) Implicit-assumption audit against the original `main()`:
#    - Original `detect_controls_frame()` assumed exactly two meaningful border sides. That is false now; all four
#      sides are meaningful, and opposite sides intentionally repeat colors.
#    - Original `extract_center_object()` assumed one interior object and picked the largest. That fails because
#      there are multiple true objects plus distractors.
#    - Original `guide_edges()` assumed one successful transform. That fails because repeated frame colors create
#      several guide-color-valid transforms for each true object.
#    - Original early `return` after the first match is invalid. The solver must keep all corner-valid candidates,
#      then rank them by the dominant-color tie-break.
#    - New solver assumption introduced here: the dominant non-fill color is uniquely defined and concentrated toward
#      one corner strongly enough that the best corner score is unique. The generator enforces that.
# d) Placement legality:
#    - The frame occupies only the outer border. All interior objects are placed with `padding=1` and `border=2`,
#      so they do not overlap each other, do not touch the frame, and remain separate 4-connected components.
#    - Guide-corner cells are background, so guided objects still crop cleanly after alignment.
#    - Distractors never use either frame-control color, preventing fake guide matches.
# e) Visual observability:
#    - Across a few samples, a human sees several inner rectangles, notices that only some carry colored guide strips,
#      and notices that opposite frame sides repeat colors, so color alone does not pin down a corner.
#    - The output consistently shows every solved core in the same internal orientation even though the guide colors
#      alone would allow several corners. That makes the frequency tie-break observable instead of hidden.
# f) Visual cleanliness:
#    - No border noise is used; the repeated-color frame is visually simpler than the original noisy border.
#    - True objects and distractors are separated by at least one background cell, avoiding merged silhouettes.
#    - The dominant-color cue is a compact corner cluster rather than scattered statistics, so each core remains easy
#      to parse after cropping.
# g) Color-change inferability:
#    - No arbitrary recoloring is introduced. Frame colors stay fixed on opposite sides; true-object guide colors use
#      those same two colors; distractors avoid them.
#    - The only extra semantic role is frequency prominence of one existing interior color, reinforced by its corner
#      concentration. This is inferable from examples without requiring hidden metadata.
#
# New solver assumptions introduced by this combo and how the generator guarantees them:
# - Every true object has exactly one horizontal guide side and one vertical guide side after any valid alignment.
# - Opposite frame sides share colors, so each true object has multiple color-valid candidate corners.
# - The most frequent non-fill color of each core is unique and yields a unique best canonical-orientation score.
# - Each true object claims a distinct corner; the generator samples unique target corners and verifies the solver.

from __future__ import annotations

from collections import Counter
from pathlib import Path
import random
import sys

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/arc2_solver_select")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import crop_bbox, find_components, get_bg, paste, random_free_location


CORNER_TO_SIDES: dict[str, tuple[str, str]] = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
SIDES_TO_CORNER = {sides: corner for corner, sides in CORNER_TO_SIDES.items()}
CORNER_ORDER = ("tl", "tr", "bl", "br")
SIDE_ORDER = ("top", "bottom", "left", "right")


def _transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.rot90(arr, int(k))
    return np.fliplr(out) if flip else out


def _inverse_transform(arr: np.ndarray, k: int, flip: bool) -> np.ndarray:
    out = np.fliplr(arr) if flip else arr
    return np.rot90(out, -int(k))


def _frame_controls(arr: np.ndarray, bg: int) -> dict[str, int]:
    sides = {
        "top": arr[0, 1:-1],
        "bottom": arr[-1, 1:-1],
        "left": arr[1:-1, 0],
        "right": arr[1:-1, -1],
    }
    controls: dict[str, int] = {}
    for side, values in sides.items():
        vals = list(map(int, values))
        non_bg = [v for v in vals if v != int(bg)]
        if not non_bg or len(set(non_bg)) != 1:
            continue
        if len(non_bg) >= len(vals) - 1:
            controls[side] = int(non_bg[0])
    return controls


def _interior_objects(arr: np.ndarray, bg: int) -> list[np.ndarray]:
    h, w = arr.shape
    comps = find_components(arr, bg=int(bg), connectivity=4, monochromatic=False, return_format="dict")
    objects: list[tuple[tuple[int, int], np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        patch = crop_bbox(arr, (r0, c0, r1, c1))
        objects.append(((r0, c0), patch))
    objects.sort(key=lambda item: item[0])
    return [patch for _pos, patch in objects]


def _most_common_non_fill(arr: np.ndarray, bg: int) -> int | None:
    values = [int(v) for v in arr.flatten() if int(v) != int(bg)]
    if not values:
        return None
    counts = Counter(values)
    fill = int(counts.most_common(1)[0][0])
    non_fill_counts = [(int(color), int(count)) for color, count in counts.items() if int(color) != fill]
    if not non_fill_counts:
        return None
    non_fill_counts.sort(key=lambda item: (item[1], item[0]), reverse=True)
    if len(non_fill_counts) > 1 and non_fill_counts[0][1] == non_fill_counts[1][1]:
        return None
    return int(non_fill_counts[0][0])


def _guide_edges(arr: np.ndarray, bg: int) -> dict[str, int]:
    non_bg_values = [int(v) for v in arr.flatten() if int(v) != int(bg)]
    if not non_bg_values:
        return {}
    fill = int(Counter(non_bg_values).most_common(1)[0][0])
    edges = {
        "top": arr[0, :],
        "bottom": arr[-1, :],
        "left": arr[:, 0],
        "right": arr[:, -1],
    }
    out: dict[str, int] = {}
    for side, values in edges.items():
        vals = list(map(int, values))
        non_bg = [v for v in vals if v != int(bg)]
        if len(non_bg) < len(vals) - 1 or len(set(non_bg)) != 1:
            continue
        color = int(non_bg[0])
        if color != fill:
            out[side] = color
    return out


def _crop_guides(arr: np.ndarray, matched_sides: tuple[str, str]) -> np.ndarray:
    top = 1 if "top" in matched_sides else 0
    bottom = arr.shape[0] - (1 if "bottom" in matched_sides else 0)
    left = 1 if "left" in matched_sides else 0
    right = arr.shape[1] - (1 if "right" in matched_sides else 0)
    return arr[int(top) : int(bottom), int(left) : int(right)].copy()


def _corner_patch(arr: np.ndarray, corner: str, size: int) -> np.ndarray:
    h, w = arr.shape
    rr = min(int(size), int(h))
    cc = min(int(size), int(w))
    if corner == "tl":
        return arr[:rr, :cc]
    if corner == "tr":
        return arr[:rr, w - cc :]
    if corner == "bl":
        return arr[h - rr :, :cc]
    return arr[h - rr :, w - cc :]


def _corner_affinity(core: np.ndarray, corner: str, dominant_color: int) -> tuple[int, int, int]:
    coords = np.argwhere(core == int(dominant_color))
    if len(coords) == 0:
        return (-1, -1, -10**9)
    h, w = core.shape
    if corner == "tl":
        anchor = (0, 0)
    elif corner == "tr":
        anchor = (0, w - 1)
    elif corner == "bl":
        anchor = (h - 1, 0)
    else:
        anchor = (h - 1, w - 1)
    patch2 = int(np.count_nonzero(_corner_patch(core, corner, 2) == int(dominant_color)))
    patch3 = int(np.count_nonzero(_corner_patch(core, corner, 3) == int(dominant_color)))
    dist_sum = int(sum(abs(int(r) - anchor[0]) + abs(int(c) - anchor[1]) for r, c in coords))
    return patch2, patch3, -dist_sum


def _matching_candidates(obj: np.ndarray, controls: dict[str, int], bg: int) -> dict[str, np.ndarray]:
    found: dict[str, np.ndarray] = {}
    for k in range(4):
        base = np.rot90(obj, int(k))
        for flip in (False, True):
            transformed = np.fliplr(base) if flip else base
            guides = _guide_edges(transformed, bg)
            if len(guides) != 2:
                continue
            horiz = [side for side in guides if side in ("top", "bottom")]
            vert = [side for side in guides if side in ("left", "right")]
            if len(horiz) != 1 or len(vert) != 1:
                continue
            sides = (horiz[0], vert[0])
            if any(side not in controls for side in sides):
                continue
            if guides[sides[0]] != controls[sides[0]] or guides[sides[1]] != controls[sides[1]]:
                continue
            corner = SIDES_TO_CORNER[sides]
            if corner not in found:
                found[corner] = _crop_guides(transformed, sides)
    return found


def _select_corner_by_frequency(candidates: dict[str, np.ndarray], bg: int) -> tuple[str, np.ndarray] | None:
    best: tuple[tuple[int, int, int], str, np.ndarray] | None = None
    second_key: tuple[int, int, int] | None = None
    for corner, core in candidates.items():
        dominant_color = _most_common_non_fill(core, bg)
        if dominant_color is None:
            continue
        key = _corner_affinity(core, "tl", dominant_color)
        if best is None or key > best[0]:
            second_key = best[0] if best is not None else second_key
            best = (key, corner, core)
        elif second_key is None or key > second_key:
            second_key = key
    if best is None:
        return None
    if second_key is not None and best[0] == second_key:
        return None
    return best[1], best[2]


def _compose_output(corners: dict[str, np.ndarray], bg: int) -> np.ndarray:
    top_h = max((corners[name].shape[0] for name in ("tl", "tr") if name in corners), default=0)
    bottom_h = max((corners[name].shape[0] for name in ("bl", "br") if name in corners), default=0)
    left_w = max((corners[name].shape[1] for name in ("tl", "bl") if name in corners), default=0)
    right_w = max((corners[name].shape[1] for name in ("tr", "br") if name in corners), default=0)

    row_gap = 1 if top_h > 0 and bottom_h > 0 else 0
    col_gap = 1 if left_w > 0 and right_w > 0 else 0
    out_h = top_h + row_gap + bottom_h
    out_w = left_w + col_gap + right_w
    out = np.full((out_h, out_w), int(bg), dtype=int)

    placements: dict[str, tuple[int, int]] = {}
    if "tl" in corners:
        placements["tl"] = (0, 0)
    if "tr" in corners:
        placements["tr"] = (0, left_w + col_gap + (right_w - corners["tr"].shape[1]))
    if "bl" in corners:
        placements["bl"] = (top_h + row_gap + (bottom_h - corners["bl"].shape[0]), 0)
    if "br" in corners:
        placements["br"] = (
            top_h + row_gap + (bottom_h - corners["br"].shape[0]),
            left_w + col_gap + (right_w - corners["br"].shape[1]),
        )
    for corner, (r0, c0) in placements.items():
        paste(out, corners[corner], int(r0), int(c0))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Combo rule:
    - Read all four frame sides, where top/bottom share one color and left/right share another.
    - Find every interior component.
    - Ignore distractors that do not expose one horizontal and one vertical guide strip.
    - For each guided object, keep every dihedral transform whose guide-strip colors match the repeated frame colors.
    - Use the most frequent non-fill core color as a corner tie-break, crop the guides, and assemble the cores into
      the output corner collage.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    controls = _frame_controls(g, bg)
    objects = _interior_objects(g, bg)

    matched: dict[str, np.ndarray] = {}
    for obj in objects:
        candidates = _matching_candidates(obj, controls, bg)
        if not candidates:
            continue
        selected = _select_corner_by_frequency(candidates, bg)
        if selected is None:
            continue
        corner, core = selected
        if corner not in matched:
            matched[corner] = core

    if not matched:
        return np.full((1, 1), int(bg), dtype=int)
    return _compose_output(matched, bg)


def _build_true_core(h: int, w: int, fill: int, dominant_color: int, accent_colors: list[int]) -> np.ndarray:
    for _ in range(120):
        core = np.full((h, w), int(fill), dtype=int)
        reserved: set[tuple[int, int]] = set()

        motif = {(0, 0), (0, 1), (1, 0), (1, 1), (0, 2), (0, 3), (2, 1)}
        for r, c in motif:
            if 0 <= r < h and 0 <= c < w:
                core[r, c] = int(dominant_color)
                reserved.add((int(r), int(c)))

        free_cells = [(r, c) for r in range(h) for c in range(w) if (r, c) not in reserved]
        random.shuffle(free_cells)
        placed_counts: dict[int, int] = {}
        for color in accent_colors:
            count = random.randint(1, 3)
            placed_counts[int(color)] = count
            for _ in range(count):
                if not free_cells:
                    break
                r, c = free_cells.pop()
                core[r, c] = int(color)

        dominant_count = int(np.count_nonzero(core == int(dominant_color)))
        if dominant_count < 6:
            continue
        if any(dominant_count <= int(count) for count in placed_counts.values()):
            continue
        dominant = _most_common_non_fill(core, bg=0)
        if dominant != int(dominant_color):
            continue
        scores = {corner: _corner_affinity(core, corner, int(dominant_color)) for corner in CORNER_ORDER}
        best_corner = max(scores, key=scores.get)
        ordered_scores = sorted(scores.values(), reverse=True)
        if best_corner == "tl" and ordered_scores[0] != ordered_scores[1]:
            return core
    raise RuntimeError("failed to build true core with unique corner preference")


def _build_distractor_core(h: int, w: int, fill: int, colors: list[int]) -> np.ndarray:
    core = np.full((h, w), int(fill), dtype=int)
    area = h * w
    target_nonfill = random.randint(max(3, area // 8), max(5, int(area * 0.30)))
    painted = 0
    while painted < target_nonfill:
        color = int(random.choice(colors))
        shape_kind = random.choice(["dot", "line", "rect"])
        if shape_kind == "dot":
            cells = [(random.randrange(h), random.randrange(w))]
        elif shape_kind == "line":
            if random.random() < 0.5:
                length = random.randint(2, min(4, w))
                rr = random.randrange(h)
                c0 = random.randrange(w - length + 1)
                cells = [(rr, c0 + dc) for dc in range(length)]
            else:
                length = random.randint(2, min(4, h))
                r0 = random.randrange(h - length + 1)
                cc = random.randrange(w)
                cells = [(r0 + dr, cc) for dr in range(length)]
        else:
            hh = random.randint(1, min(3, h))
            ww = random.randint(1, min(3, w))
            r0 = random.randrange(h - hh + 1)
            c0 = random.randrange(w - ww + 1)
            cells = [(r0 + dr, c0 + dc) for dr in range(hh) for dc in range(ww)]

        for r, c in cells:
            if core[r, c] == int(fill):
                painted += 1
            core[r, c] = int(color)
            if painted >= target_nonfill:
                break
    return core


def _attach_guides(core: np.ndarray, corner: str, h_color: int, v_color: int, bg: int) -> np.ndarray:
    guide_sides = CORNER_TO_SIDES[corner]
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
        out[0, c0 : c0 + w] = int(h_color)
    if has_bottom:
        out[-1, c0 : c0 + w] = int(h_color)
    if has_left:
        out[r0 : r0 + h, 0] = int(v_color)
    if has_right:
        out[r0 : r0 + h, -1] = int(v_color)

    if has_top and has_left:
        out[0, 0] = int(bg)
    if has_top and has_right:
        out[0, -1] = int(bg)
    if has_bottom and has_left:
        out[-1, 0] = int(bg)
    if has_bottom and has_right:
        out[-1, -1] = int(bg)
    return out


def _paint_frame(grid: np.ndarray, h_color: int, v_color: int, bg: int) -> None:
    grid[:, :] = int(bg)
    grid[0, 1:-1] = int(h_color)
    grid[-1, 1:-1] = int(h_color)
    grid[1:-1, 0] = int(v_color)
    grid[1:-1, -1] = int(v_color)
    grid[0, 0] = int(bg)
    grid[0, -1] = int(bg)
    grid[-1, 0] = int(bg)
    grid[-1, -1] = int(bg)


def _object_metric(obj: np.ndarray, controls: dict[str, int], bg: int) -> tuple[int, str | None]:
    candidates = _matching_candidates(obj, controls, bg)
    selected = _select_corner_by_frequency(candidates, bg)
    return len(candidates), None if selected is None else selected[0]


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    n = random.randint(30, 38)
    grid = np.full((n, n), int(bg), dtype=int)

    all_colors = list(range(1, 10))
    h_color, v_color = random.sample(all_colors, 2)
    _paint_frame(grid, h_color, v_color, bg)
    controls = {"top": h_color, "bottom": h_color, "left": v_color, "right": v_color}

    n_true = random.randint(2, 3)
    chosen_corners = random.sample(list(CORNER_TO_SIDES), n_true)
    corners_to_cores: dict[str, np.ndarray] = {}
    source_shapes: list[tuple[int, int]] = []

    for corner in chosen_corners:
        h = random.randint(5, 8)
        w = random.randint(5, 8)
        palette = [c for c in all_colors if c not in {h_color, v_color}]
        fill = random.choice(palette)
        remaining = [c for c in palette if c != fill]
        dominant_color = random.choice(remaining)
        accents = random.sample([c for c in remaining if c != dominant_color], k=2)
        core = _build_true_core(h, w, fill, dominant_color, accents)
        corners_to_cores[corner] = core

        aligned = _attach_guides(core, corner, h_color, v_color, bg)
        k = random.randrange(4)
        flip = bool(random.randrange(2))
        input_object = _inverse_transform(aligned, k, flip)
        source_shapes.append(tuple(map(int, input_object.shape)))

        match_count, selected_corner = _object_metric(input_object, controls, bg)
        if match_count < 4 or selected_corner != corner:
            return None

        try:
            r0, c0 = random_free_location(grid, input_object, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, input_object, r0, c0, transparent=bg)

    n_distractors = random.randint(1, 2)
    max_true_h = max(shape[0] for shape in source_shapes)
    max_true_w = max(shape[1] for shape in source_shapes)
    for idx in range(n_distractors):
        if idx == 0 and random.random() < 0.75:
            dh = random.randint(max(4, max_true_h - 1), max_true_h + 2)
            dw = random.randint(max(4, max_true_w - 1), max_true_w + 2)
        else:
            dh = random.randint(4, 8)
            dw = random.randint(4, 8)
        dh = min(dh, n - 6)
        dw = min(dw, n - 6)
        if dh < 4 or dw < 4:
            return None

        palette = [c for c in all_colors if c not in {h_color, v_color}]
        fill = random.choice(palette)
        colors = [c for c in palette if c != fill]
        distractor = _build_distractor_core(dh, dw, fill, colors)
        if _matching_candidates(distractor, controls, bg):
            return None
        try:
            r0, c0 = random_free_location(grid, distractor, transparent=bg, border=2, padding=1)
        except ValueError:
            return None
        paste(grid, distractor, r0, c0, transparent=bg)

    expected = _compose_output(corners_to_cores, bg)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None

    components = _interior_objects(grid, bg)
    if len(components) != n_true + n_distractors:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate input for fc7cae8d combo variant 37")
