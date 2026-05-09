# Combo variant of 1818057f (index 2)
# combo_spec: D1/2+D2/0
# Name: 结构分解+大小编码着色

from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg

# Design analysis for the combo variant
# a) Concrete behavior and behavior trace:
#    The input still shows one foreground color on a dominant background, with some
#    connected yellow components containing embedded thin '+' motifs. The original
#    task treated every detected '+' identically and painted all 5 cells teal.
#    Here the solver must do three things in sequence:
#    1. detect each '+' center locally;
#    2. measure the size of the 4-connected component containing that '+';
#    3. split the 5-cell recolor by structural role.
#    Example:
#      if a plus sits inside a 9-cell component, output is
#          . 6 .
#          6 8 6
#          . 6 .
#      if the same local plus sits inside an 18-cell component, output is
#          . 7 .
#          7 6 7
#          . 7 .
#    Relative to the original, the changed steps are both "which colors?" and
#    "which plus cells get which color?", and those decisions depend on component
#    size plus center/arm role together.
# b) Complexity calibration:
#    The fused rule uses exactly two size classes and exactly two structural roles.
#    That is enough to make the output visibly different from both the original
#    all-teal task and the single-change variants, while keeping the hypothesis
#    space small enough for ARC-style inference from a few examples.
# c) Implicit-assumption review for main():
#    - `get_bg(...)` still assumes one dominant background; generator preserves that.
#    - taking the only non-bg input color as foreground still works because output
#      colors never appear in the input.
#    - the original unconditional recolor-to-8 is invalid now; recoloring must be
#      deferred until after component size is known.
#    - component size must be read from the untouched input, not from partially
#      recolored output, so detection iterates over component masks built from `g`.
#    - because center and arm colors differ, write order matters in overlapping
#      edge cases; generator avoids multi-center overlap by constructing at most one
#      plus center per designed component.
# d) Placement legality:
#    Plus centers are sampled with at least a one-cell margin, each component is
#    grown in-bounds, and new components are kept 4-disconnected from all occupied
#    cells. This prevents overlap, unintended merges, and border violations.
# e) Visual observability:
#    Each generated sample contains at least one small plus component (<15 cells)
#    and at least one large plus component (>=15 cells). The solver therefore shows
#    both color pairs in the same training distribution, making the size-conditioned
#    mapping observable without hidden metadata.
# f) Visual cleanliness:
#    Designed plus components contain exactly one highlighted '+', while distractors
#    are tiny separate non-plus components. The recolored motifs stay legible because
#    transformed cells are sparse and do not stack across components.
# g) Change inferability:
#    The colors are tied to visible structure and visible size classes:
#    small => center teal / arms magenta; large => center magenta / arms orange.
#    No ordering by color id or other non-visual signal is used.
# h) Generator guarantee:
#    Generation is output-first: create final labeled components, then derive the
#    input by converting all transformed cells back to the single foreground color.
#    Every candidate is validated with `main(input) == output`.
# i) New solver assumptions and generator support:
#    `main()` assumes the relevant component size class is stable and that the input
#    remains two-colored. The generator enforces 4-disconnected components after
#    sizing them, so late distractor placement cannot silently flip a small plus
#    component into the large class.

PLUS_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ARM_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1)]

SIZE_THRESHOLD = 15

SMALL_CENTER_COLOR = 8  # teal
SMALL_ARM_COLOR = 6  # magenta
LARGE_CENTER_COLOR = 6  # magenta
LARGE_ARM_COLOR = 7  # orange


def _color_pair(area: int) -> tuple[int, int]:
    if int(area) < SIZE_THRESHOLD:
        return SMALL_CENTER_COLOR, SMALL_ARM_COLOR
    return LARGE_CENTER_COLOR, LARGE_ARM_COLOR


def _is_plus_center_from_mask(mask: np.ndarray, r: int, c: int) -> bool:
    return all(bool(mask[r + dr, c + dc]) for dr, dc in ARM_OFFSETS)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect embedded '+' motifs in one foreground color and recolor them by part and component size.

    Input structure:
    - bg is the most common color.
    - Exactly one foreground color forms a mix of disconnected shapes on the background.
    - Some foreground components contain a local thin '+' where a center cell has all 4 orthogonal neighbors present.

    Rule:
    - Find every foreground '+' center using the original input geometry.
    - Measure the size of that center's 4-connected foreground component.
    - If the component has size < 15, recolor the center teal (8) and the 4 arm cells magenta (6).
    - If the component has size >= 15, recolor the center magenta (6) and the 4 arm cells orange (7).
    - All non-plus pixels remain unchanged.

    Key insight:
    - The target motif is still detected locally, but the final colors require combining two extra cues:
      the cell's role inside the '+' (center vs arm) and the total size class of the surrounding component.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    h, w = g.shape
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[fg],
        return_format="dict",
    )

    for comp in components:
        coords = comp["coords"]
        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True

        plus_centers: list[tuple[int, int]] = []
        for r, c in coords:
            r = int(r)
            c = int(c)
            if r <= 0 or r >= h - 1 or c <= 0 or c >= w - 1:
                continue
            if _is_plus_center_from_mask(comp_mask, r, c):
                plus_centers.append((r, c))

        if not plus_centers:
            continue

        center_color, arm_color = _color_pair(int(comp["area"]))
        for r, c in plus_centers:
            for dr, dc in ARM_OFFSETS:
                out[r + dr, c + dc] = arm_color
        for r, c in plus_centers:
            out[r, c] = center_color

    return out


def _plus_centers(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    centers: set[tuple[int, int]] = set()
    for r, c in cells:
        if all((r + dr, c + dc) in cells for dr, dc in ARM_OFFSETS):
            centers.add((r, c))
    return centers


def _can_place_component(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
) -> bool:
    for r, c in cells:
        if r < 0 or r >= h or c < 0 or c >= w:
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for dr, dc in ARM_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _grow_plus_component(
    h: int,
    w: int,
    center: tuple[int, int],
    target_area: int,
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    cr, cc = center
    comp = {(cr + dr, cc + dc) for dr, dc in PLUS_OFFSETS}
    if len(comp) != 5 or not _can_place_component(comp, occupied, h, w):
        return None
    if _plus_centers(comp) != {center}:
        return None

    for _ in range(800):
        if len(comp) >= target_area:
            return comp

        frontier: set[tuple[int, int]] = set()
        for r, c in comp:
            for dr, dc in ARM_OFFSETS:
                cand = (r + dr, c + dc)
                if cand in comp or cand in occupied:
                    continue
                nr, nc = cand
                if nr < 0 or nr >= h or nc < 0 or nc >= w:
                    continue
                if any(
                    (ar, ac) in occupied
                    for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1))
                ):
                    continue
                trial = set(comp)
                trial.add(cand)
                if _plus_centers(trial) == {center}:
                    frontier.add(cand)

        if not frontier:
            return None

        ordered = sorted(
            frontier,
            key=lambda rc: (
                -abs(rc[0] - cr) - abs(rc[1] - cc),
                abs(rc[0] - cr + rc[1] - cc),
                rc[0],
                rc[1],
            ),
        )
        comp.add(ordered[int(rng.integers(0, min(len(ordered), 8)))])

    return comp if len(comp) == target_area else None


def _paint_plus_component(
    out: np.ndarray,
    comp: set[tuple[int, int]],
    center: tuple[int, int],
    area: int,
    fg: int,
) -> None:
    for r, c in comp:
        out[r, c] = fg
    cr, cc = center
    center_color, arm_color = _color_pair(area)
    out[cr, cc] = center_color
    for dr, dc in ARM_OFFSETS:
        out[cr + dr, cc + dc] = arm_color


def _place_plus_component(
    out: np.ndarray,
    occupied: set[tuple[int, int]],
    target_area: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    h, w = out.shape
    for _ in range(250):
        cr = int(rng.integers(2, h - 2))
        cc = int(rng.integers(2, w - 2))
        center = (cr, cc)
        comp = _grow_plus_component(h, w, center, target_area, occupied, rng)
        if comp is None:
            continue
        _paint_plus_component(out, comp, center, target_area, fg=4)
        occupied.update(comp)
        return center, comp
    return None


def _grow_distractor(
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
    target_area: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    for _ in range(150):
        start = (int(rng.integers(0, h)), int(rng.integers(0, w)))
        if start in occupied:
            continue
        sr, sc = start
        if any(
            (nr, nc) in occupied
            for nr, nc in ((sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1))
            if 0 <= nr < h and 0 <= nc < w
        ):
            continue

        comp = {start}
        for _ in range(100):
            if len(comp) >= target_area:
                break
            frontier: set[tuple[int, int]] = set()
            for r, c in comp:
                for dr, dc in ARM_OFFSETS:
                    cand = (r + dr, c + dc)
                    nr, nc = cand
                    if nr < 0 or nr >= h or nc < 0 or nc >= w:
                        continue
                    if cand in comp or cand in occupied:
                        continue
                    if any(
                        (ar, ac) in occupied
                        for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1))
                        if 0 <= ar < h and 0 <= ac < w
                    ):
                        continue
                    frontier.add(cand)
            if not frontier:
                break
            comp.add(sorted(frontier)[int(rng.integers(0, len(frontier)))])

        if len(comp) == target_area and not _plus_centers(comp):
            return comp

    return None


def _plus_component_descriptors(grid: np.ndarray, bg: int, fg: int) -> list[dict]:
    h, w = grid.shape
    descriptors: list[dict] = []
    for comp in find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[fg],
        return_format="dict",
    ):
        coords = comp["coords"]
        comp_mask = np.zeros((h, w), dtype=bool)
        comp_mask[coords[:, 0], coords[:, 1]] = True
        centers: list[tuple[int, int]] = []
        for r, c in coords:
            r = int(r)
            c = int(c)
            if 0 < r < h - 1 and 0 < c < w - 1 and _is_plus_center_from_mask(comp_mask, r, c):
                centers.append((r, c))
        if centers:
            descriptors.append({"area": int(comp["area"]), "centers": centers})
    return descriptors


def _sample_plus_areas(total: int, rng: np.random.Generator) -> list[int]:
    max_fg = int(total * 0.41)
    areas = [
        int(rng.integers(7, 13)),
        int(rng.integers(17, 25)),
    ]

    while len(areas) < 4:
        if rng.random() >= 0.55:
            break
        want_large = bool(rng.integers(0, 2))
        if want_large:
            area = int(rng.integers(17, 25))
        else:
            area = int(rng.integers(7, 13))
        if sum(areas) + area > max_fg - 6:
            break
        areas.append(area)

    rng.shuffle(areas)
    return areas


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    h = int(rng.integers(10, 23))
    w = int(rng.integers(10, 23))
    bg, fg = 2, 4

    out = np.full((h, w), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    total = h * w
    max_fg = int(total * 0.41)
    plus_areas = _sample_plus_areas(total, rng)
    if sum(plus_areas) > max_fg:
        return None

    for area in plus_areas:
        if _place_plus_component(out, occupied, area, rng) is None:
            return None

    target_fg = int(total * float(rng.uniform(0.26, 0.39)))
    current_fg = int(np.sum(out != bg))
    if current_fg > max_fg:
        return None

    failed_distractors = 0
    while current_fg < target_fg and failed_distractors < 160:
        d_area = int(rng.integers(1, 5))
        comp = _grow_distractor(h, w, occupied, d_area, rng)
        if comp is None:
            failed_distractors += 1
            continue
        for r, c in comp:
            out[r, c] = fg
        occupied.update(comp)
        current_fg += len(comp)
        failed_distractors = 0

    if current_fg < target_fg or current_fg > max_fg:
        return None

    inp = out.copy()
    inp[(inp == SMALL_CENTER_COLOR) | (inp == SMALL_ARM_COLOR) | (inp == LARGE_ARM_COLOR)] = fg

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None

    descriptors = _plus_component_descriptors(inp, bg, fg)
    if any(len(desc["centers"]) != 1 for desc in descriptors):
        return None
    small_plus = sum(1 for desc in descriptors if int(desc["area"]) < SIZE_THRESHOLD)
    large_plus = sum(1 for desc in descriptors if int(desc["area"]) >= SIZE_THRESHOLD)
    if small_plus < 1 or large_plus < 1:
        return None

    if not (np.any(solved == SMALL_CENTER_COLOR) and np.any(solved == SMALL_ARM_COLOR) and np.any(solved == LARGE_ARM_COLOR)):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
