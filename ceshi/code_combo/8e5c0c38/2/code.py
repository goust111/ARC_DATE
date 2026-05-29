# Combo variant of 8e5c0c38 (index 2)
# combo_spec: D1/0+D2/1
# Name: 自适应方向补全修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# -------------------------------------
# 1) Concrete behavior trace:
#    - Original task: each color is treated as one sprite, only vertical axes are scored,
#      and any unmatched pixel is deleted as noise.
#    - Combo task: each color is still one sprite, but now the solver scores both
#      vertical and horizontal axes and picks the stronger one for that sprite. Once the
#      axis is chosen, an unmatched visible pixel means its mirror partner is missing, so
#      the solver fills the reflected cell instead of erasing the visible cell.
#    - Small example: if a sprite contains (r,c)=(2,4) and its best horizontal axis is
#      k=7, then the mirror is (5,4). The source task would never even test that axis.
#      This combo solver does test it; if (5,4) is absent, it writes the missing pixel.
#      For a vertical sprite with axis k=10, a visible pixel at column 3 now causes a
#      fill at column 7 rather than a deletion at column 3.
# 2) Complexity calibration:
#    - The direction freedom is limited to two choices only: vertical or horizontal.
#      That keeps the task learnable while making the variation visually obvious.
#    - Each generated grid mixes directions so the task cannot collapse back to a
#      single global vertical rule.
#    - Clean sprites are rejected unless their intended axis beats every competing
#      vertical/horizontal axis by a margin, which avoids ambiguous plus-shapes and
#      doubly symmetric blobs that would hide the combo behavior.
# 3) Hidden assumption review:
#    - The original per-color loop still works only because the generator keeps one
#      distinct color per sprite.
#    - The original `cc = k - c` reflection logic assumed column reflection only. The
#      combo solver must branch on direction and use `rr = k - r` for horizontal axes.
#    - The original solver deleted from `out`; this combo solver must instead read
#      evidence from the original grid `g` and only write mirror fills into `out`, so a
#      newly added pixel does not recursively justify more additions.
#    - The generator can no longer store axis positions with `2*c0` unconditionally:
#      horizontal local axes must be translated by `2*r0`.
#    - Removing pixels for input generation must never delete both members of one
#      mirrored pair, and it must never delete axis cells, otherwise the "visible pixel
#      implies missing partner" assumption would break.
# 4) Placement legality:
#    - Generation remains output-first: build a full symmetric sprite, place it into a
#      reserved non-overlapping slot, then delete one side of selected mirror pairs to
#      form the input.
#    - Because the input is a subset of the placed output, placement stays in-bounds,
#      sprite bboxes never expand beyond their slots, and reconstruction by `main()`
#      stays inside the already validated clean output footprint.
# 5) Visual observability:
#    - Across 2-4 examples, humans can see that outputs are larger than inputs, so this
#      is clearly completion rather than denoising-by-deletion.
#    - Mixed vertical and horizontal sprites make the directional choice visible: some
#      objects regrow left-right partners while others regrow up-down partners.
# 6) Visual cleanliness:
#    - The original wide-gap slot layout is preserved, so three sprites stay separated.
#      No extra markers or overlays are added; only the sprite pixels themselves change.
# 7) Change inferability:
#    - Colors do not encode behavior. They only separate sprites, so the rule is
#      recoverable from geometry alone instead of hidden color-value conventions.
# 8) Non-degeneracy target for generation:
#    - Every sample must contain at least one horizontal-axis sprite and at least one
#      vertical-axis sprite, and at least one missing pair per sprite. That prevents
#      accidental collapse back to the original vertical-only problem.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes each non-background color has one dominant axis among the two
#      allowed directions and that visible unmatched pixels correspond to singly deleted
#      mirror pairs.
#    - `generate_input()` enforces those assumptions by constructing unique-axis clean
#      sprites, deleting only one side of a pair, preserving an axis-score margin after
#      deletions, and validating `main(input) == clean_output` before accepting a sample.


def _mirror_cell(
    r: int, c: int, height: int, width: int, direction: str, k: int
) -> tuple[int, int] | None:
    """Return the mirrored cell across the chosen axis, or None if it leaves the grid."""
    if direction == "vertical":
        cc = int(k) - int(c)
        if 0 <= cc < width:
            return int(r), int(cc)
        return None
    if direction == "horizontal":
        rr = int(k) - int(r)
        if 0 <= rr < height:
            return int(rr), int(c)
        return None
    raise ValueError(f"unknown direction: {direction}")


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    """Return the symmetry score for every candidate axis in one direction."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    if direction == "vertical":
        for k in range(2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                mirror = _mirror_cell(r, c, height, width, direction, k)
                if mirror is not None and mirror in pts:
                    score += 1
            scores[k] = score
        return scores
    if direction == "horizontal":
        for k in range(2 * (height - 1) + 1):
            score = 0
            for r, c in pts:
                mirror = _mirror_cell(r, c, height, width, direction, k)
                if mirror is not None and mirror in pts:
                    score += 1
            scores[k] = score
        return scores
    raise ValueError(f"unknown direction: {direction}")


def _best_axis(coords: list[tuple[int, int]], height: int, width: int) -> tuple[str, int, int]:
    """Choose the highest-scoring axis over vertical and horizontal candidates."""
    best_direction = "vertical"
    best_k = 0
    best_score = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if score > best_score:
                best_score = int(score)
                best_direction = direction
                best_k = int(k)
    return best_direction, best_k, best_score


def _best_other_score(
    coords: list[tuple[int, int]], height: int, width: int, true_direction: str, true_k: int
) -> int:
    """Return the best score among all axes except the supplied true axis."""
    best = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if direction == true_direction and k == true_k:
                continue
            if score > best:
                best = int(score)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each colored sprite by restoring its strongest mirror symmetry.

    Input structure:
    - One background color fills most cells, and each non-background color forms one nearly symmetric sprite.
    - Different sprites may prefer different mirror-axis directions: some are left-right symmetric, others are up-down symmetric.
    - The asymmetry comes from missing mirror pixels, so the visible sprite is a partial version of its intended symmetric form.

    Rule:
    - For each non-background color independently, score every vertical axis and every horizontal axis.
    - Pick the direction and axis with the highest symmetry score for that sprite.
    - For every visible pixel whose reflected partner across that chosen axis is absent, fill the missing partner cell.

    Key insight: The task is still per-sprite symmetry repair, but now the solver must first decide the sprite's axis orientation and then treat each unmatched pixel as evidence for a missing mirror counterpart.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts_arr = np.argwhere(g == int(col))
        if pts_arr.size == 0:
            continue
        coords = [(int(r), int(c)) for r, c in pts_arr]
        direction, k, _score = _best_axis(coords, height, width)

        for r, c in coords:
            mirror = _mirror_cell(r, c, height, width, direction, k)
            if mirror is None:
                continue
            rr, cc = mirror
            if int(g[rr, cc]) != int(col) and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(col)

    return out


# --------------- generator (combo D1/0 + D2/1) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# ── CFG: missing mirror pixels ──
MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 200
GEN_RETRIES = 600

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate local clean-sprite specs that fit the slot and density bounds."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_w:
                    continue
                area = h * w
                lo = math.ceil(DENSITY_LO * area)
                hi = math.floor(DENSITY_HI * area)
                if not (lo <= target_count <= hi):
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
                lo = math.ceil(DENSITY_LO * area)
                hi = math.floor(DENSITY_HI * area)
                if not (lo <= target_count <= hi):
                    continue
                left_count = target_count // 2
                if h <= left_count <= h * side_w:
                    options.append((h, w, target_count, 0))
    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Sample one clean-sprite spec, mildly favoring denser bounding boxes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left-half cells with at least one seed per row for mild structure."""
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == inner_anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((r, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    """Return which rows carry the center-column pixel for odd-width sprites."""
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _vertical_candidate(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build one perfectly vertically symmetric candidate sprite."""
    h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
    side_w = w // 2
    left_count = (target_count - center_count) // 2
    left = _left_cells(h, side_w, left_count)
    centers = _center_rows(h, center_count)

    sprite = np.zeros((h, w), dtype=int)
    for r, c in left:
        mirror_c = w - 1 - c
        sprite[r, c] = color
        sprite[r, mirror_c] = color
    if center_count > 0:
        sprite[centers, side_w] = color
    return sprite, w - 1


def _build_clean_sprite(color: int, max_h: int, max_w: int, direction: str) -> tuple[np.ndarray, int]:
    """Build a clean sprite whose dominant axis has the requested direction."""
    for _ in range(SPRITE_RETRIES):
        if direction == "vertical":
            sprite, true_k = _vertical_candidate(color, max_h, max_w)
        elif direction == "horizontal":
            base, true_k = _vertical_candidate(color, max_w, max_h)
            sprite = base.T.copy()
        else:
            raise ValueError(f"unknown direction: {direction}")

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        if not coords:
            continue
        best_direction, best_k, best_score = _best_axis(coords, *sprite.shape)
        other_score = _best_other_score(coords, *sprite.shape, direction, true_k)
        if best_direction != direction or best_k != true_k:
            continue
        if best_score != len(coords):
            continue
        if best_score - other_score < SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _sample_orientations() -> list[str]:
    """Guarantee that each grid mixes vertical and horizontal sprites."""
    sprite_count = NUM_OBJECTS - 1
    horizontal_count = random.choice([1, 2])
    horizontal_idx = set(random.sample(range(sprite_count), horizontal_count))
    return ["horizontal" if i in horizontal_idx else "vertical" for i in range(sprite_count)]


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into separated slots on a fresh clean output grid."""
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
        spec["r0"] = r0
        spec["c0"] = c0
        if spec["axis_dir"] == "vertical":
            spec["true_k"] = spec["axis_k_local"] + 2 * c0
        else:
            spec["true_k"] = spec["axis_k_local"] + 2 * r0
        placed.append(spec)
    return out, placed


def _removal_candidates(clean_mask: np.ndarray, direction: str, true_k: int) -> list[tuple[int, int]]:
    """Return removable cells: one side of a mirrored pair, never self-mirroring axis cells."""
    height, width = clean_mask.shape
    rows, cols = np.where(clean_mask)
    candidates: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        mirror = _mirror_cell(r, c, height, width, direction, true_k)
        if mirror is None or mirror == (r, c):
            continue
        if clean_mask[mirror]:
            candidates.append((r, c))
    return candidates


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Delete one side of mirrored pairs so `main()` must complete them back."""
    bg = int(get_bg(out))
    color = int(spec["color"])
    direction = spec["axis_dir"]
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)
    if clean_count == 0:
        return False

    candidates = _removal_candidates(clean_mask, direction, true_k)
    pair_count = len(candidates) // 2
    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), pair_count)
    if lo > hi:
        return False

    missing_target = random.randint(lo, hi)
    current_pts = set(clean_coords)
    forbidden: set[tuple[int, int]] = set()
    chosen: list[tuple[int, int]] = []

    for _ in range(missing_target):
        viable: list[tuple[int, int, float, tuple[int, int]]] = []
        for cell in candidates:
            if cell in forbidden or cell not in current_pts:
                continue
            test_pts = set(current_pts)
            test_pts.remove(cell)
            test_coords = list(test_pts)
            best_direction, best_k, best_score = _best_axis(test_coords, GRID_SIZE, GRID_SIZE)
            if best_direction != direction or best_k != true_k:
                continue
            other_score = _best_other_score(test_coords, GRID_SIZE, GRID_SIZE, direction, true_k)
            if best_score - other_score < FINAL_AXIS_GAP_MIN:
                continue
            viable.append((other_score, -best_score, random.random(), cell))

        if not viable:
            return False

        viable.sort()
        _other_score, _neg_best, _rand, cell = viable[0]
        current_pts.remove(cell)
        chosen.append(cell)
        forbidden.add(cell)
        mirror = _mirror_cell(cell[0], cell[1], GRID_SIZE, GRID_SIZE, direction, true_k)
        if mirror is not None:
            forbidden.add(mirror)

    final_coords = list(current_pts)
    best_direction, best_k, best_score = _best_axis(final_coords, GRID_SIZE, GRID_SIZE)
    other_score = _best_other_score(final_coords, GRID_SIZE, GRID_SIZE, direction, true_k)
    if best_direction != direction or best_k != true_k:
        return False
    if best_score - other_score < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = bg
    return True


def _gen_attempt() -> np.ndarray | None:
    """Build one combo sample and reject it unless the solver reconstructs the clean output."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)
    directions = _sample_orientations()

    sprite_specs = []
    try:
        for color, slot, direction in zip(obj_colors, slots, directions):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w, direction)
            sprite_specs.append(
                {
                    "color": color,
                    "sprite": sprite,
                    "axis_dir": direction,
                    "axis_k_local": axis_k_local,
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    out, placed_specs = result

    inp = out.copy()
    for spec in placed_specs:
        if not _remove_pixels(inp, out, spec):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if np.array_equal(inp, out):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
