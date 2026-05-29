# Combo variant of 8e5c0c38 (index 1)
# combo_spec: D1/0+D2/0
# Name: 水平轴补全修复

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

for parent in Path(__file__).resolve().parents:
    if (parent / "arc_utils.py").exists():
        sys.path.insert(0, str(parent))
        break

from arc_utils import colors_present, get_bg, paste


# Design analysis for combo D1/0 + D2/0
# --------------------------------------
# a) Concrete behavior trace
# - Original 8e5c0c38 treats each color as an almost vertically symmetric sprite, finds
#   the best vertical axis, then deletes any same-color pixel whose left/right mirror is
#   missing.
# - This combo changes two linked steps at once:
#   1) the symmetry axis becomes horizontal, so reflection is between rows rather than
#      between columns;
#   2) asymmetry means "a mirror partner is absent", so the solver fills the reflected
#      cell instead of erasing the visible one.
# - Small trace: suppose a sprite's true horizontal axis is k=8, so row 2 mirrors to
#   row 6. If input contains color X at (2, 5) but not at (6, 5), then:
#   - original task: would search vertical axes and try to delete unmatched left/right
#     pixels, so it is solving the wrong relation entirely;
#   - D2-only horizontal-denoise variant: finds the horizontal axis correctly but deletes
#     (2, 5);
#   - this combo: finds the same horizontal axis and fills the missing partner at (6, 5).
#   The visible asymmetry is evidence for completion, not evidence for removal.
#
# b) Complexity calibration
# - Keep the original canvas size, exactly 3 object colors, distinct-color sprites, slot
#   layout, pixel-count range, density range, and axis-gap filtering. This preserves the
#   need for per-color axis inference and prevents the puzzle from collapsing into a
#   trivial whole-grid mirror.
# - The generator builds wide horizontal-symmetry sprites so the top/bottom relation is
#   visually salient. Missing cells are 11%-33% of the clean sprite, capped at 12, which
#   makes the output noticeably larger than the input without causing clutter.
#
# c) Hidden assumptions that must change
# - Any use of `cc = k - c` assumes vertical reflection; the combo requires `rr = k - r`.
# - Candidate axes can no longer range over columns; they must range over rows:
#   `k in [0, 2*(H-1)]`.
# - The original delete branch encoded "unmatched means wrong". That assumption is false
#   here. The new rule must read asymmetry from the original grid `g` and write into
#   `out`, otherwise newly filled pixels could create artificial matches during the same
#   pass.
# - Placement translation also changes: a local horizontal axis becomes global through
#   `axis_k_local + 2*r0`, not `+ 2*c0`.
# - Generator corruption must remove one side of a top/bottom mirrored pair, never both,
#   and never a center-row cell; otherwise the fill rule would be underdetermined.
#
# d) Placement legality
# - Generation remains output-first: construct a clean horizontally symmetric sprite,
#   place it inside one reserved slot, then delete one side of selected mirrored pairs to
#   produce the input.
# - Because corruption only removes pixels, every intended filled cell already belongs to
#   the placed clean sprite, so solver completions stay in-bounds and cannot overlap other
#   objects. The slots stay separated, and each non-background color is unique to one
#   sprite, which matches the solver's per-color interpretation.
#
# e) Visual observability
# - Across examples, the consistent cue is that outputs restore top/bottom balance and
#   add pixels below or above existing ones. This differs from both the original task
#   (which shrinks objects) and from a vertical-axis completion task (which would add
#   left/right partners instead).
# - Random flips are still safe: left/right flips preserve horizontal symmetry semantics,
#   while up/down flips simply move the axis to another height.
#
# f) Visual cleanliness
# - Each sprite occupies its own region, colors remain distinct, and removed pixels stay
#   adjacent to the original shape. That keeps each training pair readable: the output is
#   a cleaner, fuller version of the same local object rather than a noisy multi-object
#   overlay.
#
# g) Change inferability
# - No color remapping is introduced. Color only separates the three sprites. The rule is
#   recoverable from shape structure alone: same-color top/bottom counterparts should
#   exist, and when one side is missing the output adds it back.
#
# h) Quantified anti-degeneration plan
# - Besides checking `main(inp) == out`, the generator rejects any sample where a
#   vertical-axis fill solver would also produce `out`. That prevents accidental collapse
#   into the D1-only vertical-completion behavior.
# - Post-generation validation can also measure:
#   1) added pixel count `|out|-|inp|` (should always be > 0),
#   2) disagreement rate with a horizontal-delete baseline (should be 100%),
#   3) disagreement rate with a vertical-fill baseline (should be high).
#
# i) New solver assumptions and generator guarantees
# - `main()` assumes each non-background color is best explained by one horizontal axis
#   and that asymmetry is caused by missing reflected partners rather than extra noise.
# - The generator enforces these assumptions by creating exact horizontal symmetry first,
#   deleting only one side of selected mirrored pairs, and keeping the true axis score at
#   least `FINAL_AXIS_GAP_MIN` above the runner-up after corruption.


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each color layer by restoring horizontal mirror symmetry.

    Input structure:
    - One background color fills most of the grid, and each non-background color forms its own nearly symmetric sprite around some horizontal axis.
    - The asymmetry comes from missing mirror pixels above or below that axis, so the visible sprite is only a partial version of its intended shape.

    Rule:
    - For each non-background color, search all possible horizontal axes independently and keep the axis with the strongest top/bottom mirror agreement.
    - Whenever a same-color pixel lacks its reflected partner across that axis, add the missing partner at the mirrored row.

    Key insight:
    - Each color must be solved separately, and an unmatched pixel is evidence that the sprite should be completed across a horizontal axis rather than pruned.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, _W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    # Search all possible horizontal axes: k in [0, 2*(H-1)] corresponds to axis r = k/2.
    for col in colors:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (H - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k) - int(r)
                if 0 <= rr < H and int(g[rr, int(c)]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        # Fill missing top/bottom mirror partners under the chosen horizontal axis.
        k = int(best_k)
        for r, c in pts:
            rr = int(k) - int(r)
            cc = int(c)
            if 0 <= rr < H and int(g[rr, cc]) != int(col) and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(col)

    return out


# --------------- generator (combo D1/0 + D2/0) ---------------

# -- CFG: canvas & layout --
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# -- CFG: clean sprite geometry --
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
WIDTH_MIN = 3
WIDTH_MAX = 12
ODD_SIDE_H_RANGE = range(1, 5)   # h = 2*side_h + 1 -> 3,5,7,9
EVEN_SIDE_H_RANGE = range(1, 5)  # h = 2*side_h     -> 2,4,6,8
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# -- CFG: missing mirror pixels --
MISSING_FRAC_LO = 0.11
MISSING_FRAC_HI = 0.33
MISSING_MAX = 12

# -- CFG: axis gap thresholds --
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# -- CFG: retry limits --
SPRITE_RETRIES = 120
GEN_RETRIES = 300

# -- CFG: augmentation --
FLIP_PROB = 0.5


def _horizontal_axis_scores(coords: list[tuple[int, int]], height: int = GRID_SIZE) -> dict[int, int]:
    """Return {axis_k: symmetric_pair_count} for every candidate horizontal axis."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (height - 1) + 1):
        score = 0
        for r, c in pts:
            rr = k - r
            if 0 <= rr < height and (rr, c) in pts:
                score += 1
        scores[k] = score
    return scores


def _second_best_horizontal_score(coords: list[tuple[int, int]], true_k: int, height: int = GRID_SIZE) -> int:
    """Return the highest horizontal-axis score excluding the target axis."""
    scores = _horizontal_axis_scores(coords, height)
    return max(score for k, score in scores.items() if k != true_k)


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate feasible (h, w, target_count, center_count) tuples for horizontal symmetry."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    w_max = min(WIDTH_MAX, max_w)

    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for w in range(WIDTH_MIN, w_max + 1):
            for side_h in ODD_SIDE_H_RANGE:
                h = 2 * side_h + 1
                if h > h_max:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                for center_count in range(1, w + 1):
                    if (target_count - center_count) % 2 != 0:
                        continue
                    top_count = (target_count - center_count) // 2
                    if w <= top_count <= side_h * w:
                        options.append((h, w, target_count, center_count))

            if target_count % 2 != 0:
                continue
            for side_h in EVEN_SIDE_H_RANGE:
                h = 2 * side_h
                if h > h_max:
                    continue
                area = h * w
                if not (math.ceil(DENSITY_LO * area) <= target_count <= math.floor(DENSITY_HI * area)):
                    continue
                top_count = target_count // 2
                if w <= top_count <= side_h * w:
                    options.append((h, w, target_count, 0))

    return options


def _choose_shape_spec(max_h: int, max_w: int) -> tuple[int, int, int, int]:
    """Pick a random shape specification, favoring denser placements."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _top_cells(side_h: int, width: int, top_count: int) -> set[tuple[int, int]]:
    """Choose cells from the top half while ensuring every column participates."""
    cells: set[tuple[int, int]] = set()
    inner_anchor_col = random.randrange(width)
    for c in range(width):
        if c == 0:
            r = 0
        elif side_h > 1 and c == inner_anchor_col:
            r = side_h - 1
        else:
            r = random.randrange(side_h)
        cells.add((r, c))

    remaining = top_count - len(cells)
    if remaining < 0:
        raise ValueError("top_count too small")

    pool = [(r, c) for r in range(side_h) for c in range(width) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_cols(width: int, center_count: int) -> np.ndarray:
    """Return a boolean mask indicating which columns get a center-row pixel."""
    cols = np.zeros(width, dtype=bool)
    if center_count > 0:
        cols[random.sample(range(width), center_count)] = True
    return cols


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a horizontally symmetric sprite and return (sprite_array, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_h = h // 2
        top_count = (target_count - center_count) // 2
        top = _top_cells(side_h, w, top_count)
        centers = _center_cols(w, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in top:
            mirror_r = h - 1 - r
            sprite[r, c] = color
            sprite[mirror_r, c] = color
        if center_count > 0:
            sprite[side_h, centers] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        true_k = h - 1
        if len(coords) != target_count:
            continue
        if _second_best_horizontal_score(coords, true_k, h) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k

    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into non-overlapping slot regions."""
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
        spec["true_k"] = spec["axis_k_local"] + 2 * r0
        placed.append(spec)

    return out, placed


def _removal_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Return removable cells: exactly one side of a top/bottom mirrored pair."""
    H, _W = clean_mask.shape
    candidates: list[tuple[int, int]] = []
    rows, cols = np.where(clean_mask)
    for r, c in zip(rows.tolist(), cols.tolist()):
        mirror_r = true_k - r
        if mirror_r == r:
            continue
        if 0 <= mirror_r < H and clean_mask[mirror_r, c]:
            candidates.append((r, c))
    return candidates


def _remove_pixels(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Delete one side of mirrored pairs so the solver must complete them back."""
    bg = int(get_bg(out))
    color = spec["color"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    candidates = _removal_candidates(clean_mask, true_k)
    pair_count = len(candidates) // 2
    lo = max(1, math.ceil(MISSING_FRAC_LO * clean_count))
    hi = min(MISSING_MAX, math.floor(MISSING_FRAC_HI * clean_count), pair_count)
    if lo > hi:
        return False

    missing_target = random.randint(lo, hi)
    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)

    for _ in range(missing_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell in current_coords]
        if not viable:
            return False

        scored = []
        for cell in viable:
            test_coords = [pt for pt in current_coords if pt != cell]
            alt = _second_best_horizontal_score(test_coords, true_k, GRID_SIZE)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)

        chosen.append(cell)
        current_coords.remove(cell)
        r, c = cell
        forbidden.add(cell)
        mirror = (true_k - r, c)
        if 0 <= mirror[0] < GRID_SIZE:
            forbidden.add(mirror)

    scores = _horizontal_axis_scores(current_coords, GRID_SIZE)
    true_score = scores[true_k]
    if true_score - max(score for k, score in scores.items() if k != true_k) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = bg
    return True


def _vertical_fill_baseline(grid: np.ndarray) -> np.ndarray:
    """Reference solver for D1-only completion, used to reject degenerate samples."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _H, W = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (W - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < W and int(g[int(r), cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        for r, c in pts:
            rr = int(r)
            cc = int(best_k) - int(c)
            if 0 <= cc < W and int(g[rr, cc]) != int(col) and int(out[rr, cc]) == int(bg):
                out[rr, cc] = int(col)

    return out


def _horizontal_delete_baseline(grid: np.ndarray) -> np.ndarray:
    """Reference solver for D2-only denoising, used in behavior audits."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, _W = g.shape
    out = g.copy()

    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (H - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k) - int(r)
                if 0 <= rr < H and int(g[rr, int(c)]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        for r, c in pts:
            rr = int(best_k) - int(r)
            if not (0 <= rr < H and int(g[rr, int(c)]) == int(col)):
                out[int(r), int(c)] = int(bg)

    return out


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean sprites, remove mirrors, and reject degenerate samples."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": color, "sprite": sprite, "axis_k_local": axis_k_local})
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
    if np.array_equal(_vertical_fill_baseline(inp), out):
        return None

    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
