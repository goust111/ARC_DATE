# Variant of 1818057f
# Dimension: D2 — 变换的目标颜色 (transform)
# Change: 所有'+'字图案统一着色为青色(8/teal)，单一固定目标颜色... → 目标颜色由'+'所在连通分量大小编码

from __future__ import annotations

import numpy as np

from arc_utils import find_components, get_bg

# Design analysis
# ----------------
# variant_spec:
#   core_change:
#     original: every detected plus footprint is recolored to a single fixed color (8).
#     variant: the recolor depends on the size of the plus's 4-connected foreground component.
#       - component size < 15  -> teal (8)
#       - component size >= 15 -> magenta (6)
#   concrete_grid_behavior:
#     - Input still has one foreground color on a most-common red background.
#     - A plus is still "center + 4 orthogonal neighbors all foreground".
#     - The new visible difference is not where pluses are found, but which color each plus receives.
#     - Therefore samples must contain both small and large plus-carrying components; otherwise the
#       task would visually collapse back to the original fixed-teal behavior.
#   behavior_trace:
#     - Original flow on a 12-cell component containing one plus:
#       detect center -> recolor five cells to teal.
#     - Variant flow on that same component:
#       measure component area first.
#       if area = 12, recolor the plus to teal.
#       if area = 18, recolor the plus to magenta.
#     - Difference:
#       detection of plus centers is unchanged, but the write color now depends on a count computed
#       from the entire connected component, so solver and generator must preserve component identity.
#   complexity_calibration:
#     - Only two output colors are used for transformed pluses. This keeps the counting rule visible
#       and learnable from 2-4 examples.
#     - The threshold is binary (<15 vs >=15), again matching ARC's preference for a few repeated cases.
#     - Generator always includes at least one component from each side of the threshold so the rule is
#       observable inside a single sample, not only across multiple samples.
#   implicit_assumption_review:
#     - Original code could safely recolor with an unconditional constant `8`; that is no longer valid.
#     - Original generator could freely sprinkle extra foreground near existing shapes, because component
#       size did not matter. In the variant this would silently flip a small component into a large one.
#     - Therefore solver must read component sizes from the untouched input grid, and generator must keep
#       distractors 4-disconnected from plus components after their target size has been chosen.
#   placement_legality:
#     - Plus centers stay off the border because the 4-neighbor test needs one-cell margin.
#     - Each plus component is grown as a single connected polyomino with exactly one plus center,
#       preventing overlapping plus footprints and keeping the visual story clean.
#     - Distractors are tiny 1-4 cell components, placed with 4-neighbor separation so they neither
#       merge with designed components nor create accidental pluses.
#   visual_observability:
#     - Humans can compare yellow components of different overall sizes and see that the embedded pluses
#       change to different colors in output.
#     - The deciding signal is visible geometry (component area / bulk), not hidden color-id ordering.
#   cleanliness:
#     - Components do not overlap or touch orthogonally.
#     - Each designed component contains one highlighted plus and a thin tail/body, so the output is not
#       cluttered by multiple competing recolor marks inside the same blob.
#   main_input_assumptions:
#     - There is exactly one non-background input color, matching the original task family.
#     - `generate_input()` guarantees at least one small plus component and at least one large plus
#       component, both 4-disconnected from distractors.

PLUS_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]
ARM_OFFSETS = [(1, 0), (-1, 0), (0, 1), (0, -1)]
SMALL_COLOR = 8  # teal
LARGE_COLOR = 6  # magenta/pink
SIZE_THRESHOLD = 15


def _target_color(area: int) -> int:
    return SMALL_COLOR if int(area) < SIZE_THRESHOLD else LARGE_COLOR


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect '+' motifs in one foreground color and recolor them by component size.

    Input structure:
    - bg = most common color.
    - One foreground color forms a mix of small pieces and larger connected shapes.
    - Some foreground components contain an embedded thin '+' pattern.

    Rule:
    - A '+' center is any foreground pixel whose 4 orthogonal neighbors are also foreground.
    - For each detected '+', inspect the size of its 4-connected foreground component in the input.
    - Recolor the 5 cells of that '+' to teal (8) if the component has size < 15, otherwise magenta (6).

    Key insight: The local '+' test still identifies the target motif, but the output color is decided by a
    global property of the motif's own connected component, so counting the component comes before coloring.
    """
    g = np.array(grid, dtype=int)
    out = g.copy()
    bg = int(get_bg(g))

    colors = sorted(set(map(int, g.flatten())) - {bg})
    if not colors:
        return out
    fg = int(colors[0])

    H, W = g.shape
    regions = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[fg],
        return_format="dict",
    )

    for comp in regions:
        target = _target_color(int(comp["area"]))
        comp_mask = np.zeros((H, W), dtype=bool)
        coords = comp["coords"]
        comp_mask[coords[:, 0], coords[:, 1]] = True

        for r, c in coords:
            r = int(r)
            c = int(c)
            if r <= 0 or r >= H - 1 or c <= 0 or c >= W - 1:
                continue
            if all(comp_mask[r + dr, c + dc] for dr, dc in ARM_OFFSETS):
                for dr, dc in PLUS_OFFSETS:
                    out[r + dr, c + dc] = target

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
    H: int,
    W: int,
) -> bool:
    for r, c in cells:
        if r < 0 or r >= H or c < 0 or c >= W:
            return False
        if (r, c) in occupied:
            return False
    for r, c in cells:
        for dr, dc in ARM_OFFSETS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def _grow_plus_component(
    H: int,
    W: int,
    center: tuple[int, int],
    target_area: int,
    occupied: set[tuple[int, int]],
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    cr, cc = center
    comp = {(cr + dr, cc + dc) for dr, dc in PLUS_OFFSETS}
    if len(comp) != 5 or not _can_place_component(comp, occupied, H, W):
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
                if nr < 0 or nr >= H or nc < 0 or nc >= W:
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
        top_k = min(len(ordered), 8)
        comp.add(ordered[int(rng.integers(0, top_k))])

    return comp if len(comp) == target_area else None


def _place_plus_component(
    out: np.ndarray,
    occupied: set[tuple[int, int]],
    target_area: int,
    rng: np.random.Generator,
) -> tuple[tuple[int, int], set[tuple[int, int]]] | None:
    H, W = out.shape
    for _ in range(250):
        cr = int(rng.integers(2, H - 2))
        cc = int(rng.integers(2, W - 2))
        center = (cr, cc)
        comp = _grow_plus_component(H, W, center, target_area, occupied, rng)
        if comp is None:
            continue

        for r, c in comp:
            out[r, c] = 4
        color = _target_color(target_area)
        for dr, dc in PLUS_OFFSETS:
            out[cr + dr, cc + dc] = color
        occupied.update(comp)
        return center, comp
    return None


def _grow_distractor(
    H: int,
    W: int,
    occupied: set[tuple[int, int]],
    target_area: int,
    rng: np.random.Generator,
) -> set[tuple[int, int]] | None:
    for _ in range(150):
        start = (int(rng.integers(0, H)), int(rng.integers(0, W)))
        if start in occupied:
            continue
        sr, sc = start
        if any(
            (nr, nc) in occupied
            for nr, nc in ((sr - 1, sc), (sr + 1, sc), (sr, sc - 1), (sr, sc + 1))
            if 0 <= nr < H and 0 <= nc < W
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
                    if nr < 0 or nr >= H or nc < 0 or nc >= W:
                        continue
                    if cand in comp or cand in occupied:
                        continue
                    if any(
                        (ar, ac) in occupied
                        for ar, ac in ((nr - 1, nc), (nr + 1, nc), (nr, nc - 1), (nr, nc + 1))
                        if 0 <= ar < H and 0 <= ac < W
                    ):
                        continue
                    frontier.add(cand)
            if not frontier:
                break
            comp.add(sorted(frontier)[int(rng.integers(0, len(frontier)))])

        if len(comp) == target_area and not _plus_centers(comp):
            return comp

    return None


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    H = int(rng.integers(10, 23))
    W = int(rng.integers(10, 23))
    bg, fg = 2, 4

    out = np.full((H, W), bg, dtype=int)
    occupied: set[tuple[int, int]] = set()

    total = H * W
    max_fg = int(total * 0.41)
    plus_specs: list[int] = []
    if total < 160:
        n_small = 1
        n_large = 1
    else:
        n_small = int(rng.integers(1, 3))
        n_large = int(rng.integers(1, 3))
    plus_specs.extend(int(rng.integers(6, SIZE_THRESHOLD)) for _ in range(n_small))
    plus_specs.extend(int(rng.integers(SIZE_THRESHOLD, 25)) for _ in range(n_large))
    rng.shuffle(plus_specs)

    if sum(plus_specs) > max_fg:
        return None

    for area in plus_specs:
        if _place_plus_component(out, occupied, area, rng) is None:
            return None

    target_density = float(rng.uniform(0.26, 0.41))
    target_fg = int(total * target_density)
    current_fg = int(np.sum(out != bg))
    if current_fg > max_fg:
        return None

    failed_distractors = 0
    while current_fg < target_fg and failed_distractors < 120:
        d_area = int(rng.integers(1, 5))
        comp = _grow_distractor(H, W, occupied, d_area, rng)
        if comp is None:
            failed_distractors += 1
            continue
        for r, c in comp:
            out[r, c] = fg
        occupied.update(comp)
        current_fg += len(comp)
        failed_distractors = 0

    if current_fg < target_fg:
        return None
    if current_fg > max_fg:
        return None

    inp = out.copy()
    inp[(inp == SMALL_COLOR) | (inp == LARGE_COLOR)] = fg

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if not np.any(solved == SMALL_COLOR) or not np.any(solved == LARGE_COLOR):
        return None

    inp_regions = find_components(
        inp,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[fg],
        return_format="dict",
    )
    small_plus = 0
    large_plus = 0
    for comp in inp_regions:
        comp_mask = np.zeros((H, W), dtype=bool)
        coords = comp["coords"]
        comp_mask[coords[:, 0], coords[:, 1]] = True
        has_plus = False
        for r, c in coords:
            r = int(r)
            c = int(c)
            if 0 < r < H - 1 and 0 < c < W - 1 and all(comp_mask[r + dr, c + dc] for dr, dc in ARM_OFFSETS):
                has_plus = True
                break
        if has_plus:
            if int(comp["area"]) < SIZE_THRESHOLD:
                small_plus += 1
            else:
                large_plus += 1

    if small_plus < 1 or large_plus < 1:
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("generate_input failed after 400 attempts")
