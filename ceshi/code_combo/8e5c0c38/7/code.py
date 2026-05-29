# Combo variant of 8e5c0c38 (index 7)
# combo_spec: D1/0+D12/0
# Name: 共享颜色补全修复

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - Source task logic: group pixels by color, choose one best vertical axis per color layer,
#      then delete pixels that do not already have a same-color mirror partner.
#    - D1 flips the repair direction: an unmatched pixel now means its reflected partner is
#      missing, so the solver must add that partner instead of erasing the visible pixel.
#    - D12 breaks the one-color-one-sprite assumption: one foreground color is reused by two
#      disconnected sprites. Example: color 5 appears in left component A and right component B.
#      A has a pixel at (r, c) whose partner at (r, c') was deleted; B has its own missing pair
#      under a different axis. Old color-only filling would search one compromise axis for A∪B and
#      often fill the wrong places. The fused rule is: split color 5 into A and B by 4-connectivity,
#      infer axis_A and axis_B separately, then fill each component's missing mirrors.
#    - Visible difference from the source task is twofold at once: outputs are larger than inputs,
#      and one color can appear in two separate regions that must be repaired independently.
# b) Complexity calibration
#    - Keep the original 22x22 canvas, 3 sprites, reserved distant slots, vertical symmetry, and
#      axis-gap checks so the task still hinges on per-sprite axis inference.
#    - Limit the color-system change to exactly one duplicated foreground color. That is enough to
#      force connectivity-based grouping without introducing many arbitrary color cases.
#    - Keep deletions at 11%-33% of the clean sprite, capped at 12 cells, so completion is obvious
#      in examples but the true axis remains recoverable from surviving mirrored pairs.
# c) Implicit-assumption audit of the original main()
#    - `for col in colors`: still valid as an outer partition, but no longer sufficient to define
#      one object because a color layer may contain two sprites.
#    - `pts = np.argwhere(g == col)`: broken by D12; this would merge disconnected same-color
#      sprites and can yield a compromise axis unrelated to either real sprite.
#    - Deleting unmatched pixels: broken by D1; unmatched now signals "fill the partner".
#    - Writing while reading the same structure: risky after D1, because newly added pixels could
#      influence later checks. The solver therefore reads component coordinates from the original
#      grid `g` and writes fills into `out`.
# d) Placement legality
#    - Generation remains output-first: first build the fully symmetric output sprites, then delete
#      one side of selected mirrored pairs to form the input. Added pixels therefore stay in-bounds
#      and inside the original placed bbox.
#    - The three slot regions are far apart, so sprites never overlap. The duplicated-color sprites
#      stay disconnected even after flips, and each noisy sprite is kept as one 4-connected input
#      component so solver-side component splitting matches the intended sprite partition.
# e) Visual observability
#    - From a few examples, a human can see that outputs add pixels instead of removing them.
#    - The duplicated color appearing in two distant regions shows that color alone cannot define
#      the repair unit. Because both same-color blobs follow the same local mirror-completion rule,
#      the decomposition by connectivity is visually inferable.
# f) Visual cleanliness
#    - Only one foreground color is reused, and reused sprites are separated by the original slot
#      layout. There is no stacking or overlap, so same-color components remain visually distinct.
#    - Each clean sprite is forced to be a single connected object, keeping the "one blob = one
#      repair axis" interpretation readable.
# g) Color-change audit
#    - No color remapping is introduced. Colors only identify ownership of pixels, with one color
#      intentionally reused for two disconnected sprites.
# h) New assumptions introduced by the fused solver
#    - Because the solver splits input by 4-connected components before choosing axes, the generator
#      must ensure deletions never fragment one sprite into multiple pieces.
#    - The generator also rejects samples where a naive D1-only solver (color-layer fill without
#      component splitting) would accidentally produce the same output, so the D12 contribution is
#      genuinely active in the dataset.


GRID_SIZE = 22
NUM_SPRITES = 3
NUM_FOREGROUND_COLORS = 2
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12

SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

SPRITE_RETRIES = 240
GEN_RETRIES = 500

FLIP_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Return {axis_k: self-mirror score} for all candidate vertical axes."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        scores[k] = score
    return scores


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    """Return the best axis score excluding `true_k`."""
    scores = _axis_scores(coords, width)
    return max(score for k, score in scores.items() if k != true_k)


def _best_axis_for_component(coords: np.ndarray, width: int) -> int:
    """Return the best vertical self-mirror axis for one component."""
    pts = {(int(r), int(c)) for r, c in coords}
    best_score = -1
    best_k = 0
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each disconnected monochromatic sprite by restoring its own vertical mirror symmetry.

    Input structure:
    - One background color fills most cells. There are exactly three near-symmetric sprites, but only two
      foreground colors because one color is shared by two disconnected components.
    - Each visible component is a connected subset of its intended symmetric sprite; the asymmetry comes
      from missing mirror pixels rather than extra protruding noise.

    Rule:
    - For each non-background color, first split that color layer into separate 4-connected components.
    - For each component independently, find the vertical axis that best matches its current pixels and
      fill any missing mirror partner on the reflected side.

    Key insight:
    - The repeated color means color alone no longer identifies one sprite, so spatial separation must come
      first. After that split, any unmatched pixel is evidence that its mirror counterpart should be added.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        comps = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        for comp in comps:
            coords = np.array(comp["coords"], dtype=int)
            if coords.size == 0:
                continue

            k = _best_axis_for_component(coords, w)
            pts = {(int(r), int(c)) for r, c in coords}
            for r, c in pts:
                cc = k - c
                if 0 <= cc < w and (r, cc) not in pts and int(out[int(r), cc]) == int(bg):
                    out[int(r), cc] = int(col)

    return out


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying the base constraints."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                for center_count in range(1, h + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    left_count = (target_count - center_count) // 2
                    if h <= left_count <= h * side_w:
                        options.append((h, w, target_count, center_count))

            if target_count % 2 != 0:
                continue
            for side_w in EVEN_SIDE_W_RANGE:
                w = 2 * side_w
                if w > max_w:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                left_count = target_count // 2
                if h <= left_count <= h * side_w:
                    options.append((h, w, target_count, 0))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick a random feasible shape spec, biased toward denser fills."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> tuple[set[tuple[int, int]], int]:
    """Build a connected left-half scaffold plus random growth."""
    max_extra = left_count - h
    min_backbone = max(0, side_w - 1 - max_extra)
    backbone_col = random.randint(min_backbone, side_w - 1)
    seam_row = random.randrange(h)

    cells: set[tuple[int, int]] = {(r, backbone_col) for r in range(h)}
    for c in range(backbone_col, side_w):
        cells.add((seam_row, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small for connected scaffold")

    while remaining > 0:
        frontier: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in cells:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                cand = (rr, cc)
                if 0 <= rr < h and 0 <= cc < side_w and cand not in cells and cand not in seen:
                    frontier.append(cand)
                    seen.add(cand)
        if not frontier:
            raise ValueError("no frontier while growing connected left half")
        cells.add(random.choice(frontier))
        remaining -= 1

    return cells, seam_row


def _center_rows(h: int, center_count: int, bridge_row: int) -> np.ndarray:
    """Return boolean mask of rows containing a center-column cell."""
    rows = np.zeros(h, dtype=bool)
    if center_count <= 0:
        return rows
    rows[bridge_row] = True
    if center_count > 1:
        other_rows = [r for r in range(h) if r != bridge_row]
        rows[random.sample(other_rows, center_count - 1)] = True
    return rows


def _is_single_component(sprite: np.ndarray, color: int) -> bool:
    """Return whether all nonzero color cells in `sprite` form one 4-connected component."""
    comps = find_components(
        sprite,
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    return len(comps) == 1


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build one connected, vertically symmetric sprite and return (sprite, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left, seam_row = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count, seam_row)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mirror_c = w - 1 - c
            sprite[r, c] = color
            sprite[r, mirror_c] = color
        if center_count > 0:
            sprite[centers, side_w] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if not _is_single_component(sprite, color):
            continue
        if _second_best_score(coords, true_k, width=w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites in the fixed slot layout and record their global geometry."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)

        clean_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        clean_mask[r0 : r0 + h, c0 : c0 + w] = sprite != 0
        spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        spec["clean_mask"] = clean_mask
        spec["clean_coords"] = [tuple(x) for x in np.argwhere(clean_mask)]
        placed.append(spec)

    return out, placed


def _coords_are_connected(coords: list[tuple[int, int]]) -> bool:
    """Return whether the given global coordinates form one 4-connected component."""
    if not coords:
        return False
    pts = {(int(r), int(c)) for r, c in coords}
    start = next(iter(pts))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            cand = (rr, cc)
            if cand in pts and cand not in seen:
                seen.add(cand)
                stack.append(cand)
    return len(seen) == len(pts)


def _removal_candidates(coords: list[tuple[int, int]], true_k: int) -> list[tuple[int, int]]:
    """Return non-axis cells whose mirror currently exists inside the same sprite."""
    pts = {(int(r), int(c)) for r, c in coords}
    candidates: list[tuple[int, int]] = []
    for r, c in pts:
        mirror_c = true_k - c
        if mirror_c == c:
            continue
        if 0 <= mirror_c < GRID_SIZE and (r, mirror_c) in pts:
            candidates.append((r, c))
    return candidates


def _remove_pixels(inp: np.ndarray, bg: int, spec: dict) -> int | None:
    """Delete one side of mirrored pairs while keeping the visible input sprite connected."""
    true_k = int(spec["true_k"])
    clean_coords = list(spec["clean_coords"])
    clean_count = len(clean_coords)
    candidates = _removal_candidates(clean_coords, true_k)
    pair_count = len(candidates) // 2

    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), pair_count)
    if lo > hi:
        return None

    missing_target = random.randint(lo, hi)
    current_coords = list(clean_coords)
    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()

    for _ in range(missing_target):
        viable_scored: list[tuple[int, float, tuple[int, int]]] = []
        for cell in candidates:
            if cell in forbidden or cell not in current_coords:
                continue
            test_coords = [pt for pt in current_coords if pt != cell]
            if not _coords_are_connected(test_coords):
                continue
            alt = _second_best_score(test_coords, true_k, GRID_SIZE)
            viable_scored.append((alt, random.random(), cell))

        if not viable_scored:
            return None

        viable_scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = viable_scored[0][0]
        pool = [cell for alt, _, cell in viable_scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.remove(cell)
        forbidden.add(cell)
        mirror = (cell[0], true_k - cell[1])
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    if not _coords_are_connected(current_coords):
        return None

    scores = _axis_scores(current_coords, GRID_SIZE)
    true_score = scores[true_k]
    if true_score - max(score for k, score in scores.items() if k != true_k) < FINAL_AXIS_GAP_MIN:
        return None

    for r, c in chosen:
        inp[r, c] = bg
    return len(chosen)


def _color_assignment() -> tuple[int, list[int], int]:
    """Return (bg, sprite_colors, shared_color) with exactly one duplicated foreground color."""
    bg, shared_color, unique_color = random.sample(range(10), NUM_FOREGROUND_COLORS + 1)
    sprite_colors = [shared_color, shared_color, unique_color]
    random.shuffle(sprite_colors)
    return bg, sprite_colors, shared_color


def _solve_without_component_split(grid: np.ndarray) -> np.ndarray:
    """Naive D1-only baseline: fill by whole color layer without separating same-color sprites."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        k = _best_axis_for_component(pts, w)
        layer_pts = {(int(r), int(c)) for r, c in pts}
        for r, c in layer_pts:
            cc = k - c
            if 0 <= cc < w and (r, cc) not in layer_pts and int(out[r, cc]) == int(bg):
                out[r, cc] = int(col)
    return out


def _shared_color_has_two_components(grid: np.ndarray, bg: int, shared_color: int) -> bool:
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(shared_color)],
        return_format="dict",
    )
    return len(comps) == 2


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict] | None:
    """Build one fused example and reject degenerate samples."""
    bg, sprite_colors, shared_color = _color_assignment()
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(sprite_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": color, "sprite": sprite, "axis_k_local": axis_k_local})
    except ValueError:
        return None

    placed_result = _slot_place_sprites(bg, sprite_specs, slots)
    if placed_result is None:
        return None
    out, placed_specs = placed_result

    inp = out.copy()
    missing_by_color: dict[int, int] = {}
    for spec in placed_specs:
        removed = _remove_pixels(inp, bg, spec)
        if removed is None:
            return None
        color = int(spec["color"])
        missing_by_color[color] = missing_by_color.get(color, 0) + removed

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    solved = main(inp)
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(inp, out):
        return None
    if not _shared_color_has_two_components(inp, bg, shared_color):
        return None
    if missing_by_color.get(shared_color, 0) < 2:
        return None
    if np.array_equal(_solve_without_component_split(inp), out):
        return None

    meta = {
        "shared_color": int(shared_color),
        "missing_by_color": {int(k): int(v) for k, v in missing_by_color.items()},
        "input_pixels": int(np.count_nonzero(inp != bg)),
        "output_pixels": int(np.count_nonzero(out != bg)),
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
