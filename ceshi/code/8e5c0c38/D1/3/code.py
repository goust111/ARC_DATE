# Variant of 8e5c0c38
# Dimension: D1 — 对称修复的变换方式 (transform)
# Change: 删除没有镜像对应像素的噪声像素，使精灵恢复垂直镜像对称... → 噪声像素沿对称轴翻转到镜像位置

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the D1 variant
# a) Concrete behavior and trace
#    Each non-background color still forms one almost vertically symmetric sprite.
#    The stable part is a mirror-symmetric core; the unstable part is a set of
#    one-sided extra pixels near that core. Old task: unmatched pixel disappears.
#    New task: unmatched pixel relocates to its reflected column if that cell is
#    inside the grid and empty. Example with axis k=8 (center column 4):
#      clean pair at columns 3 and 5, extra pixel at column 2
#      old output: pixel at 2 is erased
#      new output: pixel at 2 is cleared and a pixel appears at 6
#    So the asymmetry does not vanish; it swaps sides across the inferred axis.
# b) Complexity calibration
#    The axis is still a hidden per-color variable, so the puzzle keeps the same
#    core reasoning load. To make the change observable, every generated sprite
#    gets at least one movable noise pixel, usually several. Noise remains
#    4-adjacent to the object so humans can see it belongs to that sprite.
#    We forbid mirrored noise pairs, because they would cancel the effect and
#    make input/output identical.
# c) Audit of old main() assumptions
#    - Axis search by same-color mirror score is still valid because the clean
#      symmetric core is unchanged and still dominates the score.
#    - The old update "out[r, c] = bg for every unmatched pixel" relied on the
#      repair action being deletion. That is no longer valid.
#    - Counterpart checks must read the original grid g, not the evolving out,
#      otherwise a previously moved pixel could create a fake match.
#    - Movement needs a new occupancy check: only move when the reflected cell is
#      in bounds and background. Old code could safely ignore occupancy because
#      deletion never overwrote anything.
# d) Placement legality
#    Sprites are still generated inside three separated slot layouts. Noise is
#    chosen from cells adjacent to the symmetric core, with an in-bounds mirror,
#    and whose mirror is not part of the core. This keeps both the input-side
#    and output-side noise local to the same sprite and prevents overlap.
# e) Visual observability
#    Across examples, the solver sees one-sided protrusions jump to the opposite
#    side while the main body stays fixed. This is easier to infer than a hidden
#    color rule because the changed cells are spatially tied to the axis.
# f) Visual cleanliness
#    The generator keeps the original sparse three-slot layout, so each color
#    remains isolated. No extra markers or overlays are introduced.
# g) Color inference
#    There is no color remapping. Colors only identify which pixels belong to the
#    same sprite, matching the original task.
# h) Output-first generation plan
#    Build the symmetric clean core first. Then choose output-side noise cells.
#    Reflect those cells across the sprite axis to derive the input-side noise.
#    Oracle check requires main(input) == output.
# i) New solver assumptions and how generation enforces them
#    - One object per non-background color: generator uses distinct colors.
#    - The best axis must remain identifiable: enforced by the same score-gap
#      thresholds as the original task.
#    - Movable noise should have an in-bounds empty mirror: enforced when
#      sampling candidates, so the variant behavior actually occurs.


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Return {axis_k: symmetric_pair_count} for every candidate vertical axis."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = int(k) - int(c)
            if 0 <= cc < width and (int(r), int(cc)) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int = 22) -> int:
    """Return the highest axis score excluding the designated true axis."""
    scores = _axis_scores(coords, width)
    return max(score for k, score in scores.items() if int(k) != int(true_k))


def _mirror_cell(r: int, c: int, axis_k: int) -> tuple[int, int]:
    """Reflect one cell across vertical axis k/2."""
    return int(r), int(axis_k) - int(c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Relocate each unmatched color pixel across its sprite's vertical mirror axis.

    Input structure:
    - bg is the background. Each non-bg color forms one nearly mirror-symmetric sprite around its own vertical axis.
    - A few same-color pixels sit on only one side of that axis, acting as one-sided protrusions rather than matched pairs.

    Rule:
    - For each color independently, find the vertical axis that maximizes same-color mirror matches.
    - If a pixel has no same-color mirror partner, reflect it to the mirrored cell when that cell is inside the grid and currently empty.
    - If the mirrored cell is occupied or outside the grid, leave the original pixel where it is.

    Key insight:
    - The hidden structure is still a per-color symmetry axis, but the repair is a relocation step, not denoising by deletion, so the asymmetric extras switch sides instead of disappearing.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    # Search all possible vertical axes: k in [0, 2*(W-1)] corresponds to axis at c = k/2.
    for col in colors:
        pts_arr = np.argwhere(g == int(col))
        if pts_arr.size == 0:
            continue
        pts = [(int(r), int(c)) for r, c in pts_arr]

        best_score = -1
        best_k = 0
        for k in range(0, 2 * (w - 1) + 1):
            score = 0
            for r, c in pts:
                rr, cc = _mirror_cell(int(r), int(c), int(k))
                if 0 <= cc < w and int(g[rr, cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

        # Move unpaired pixels to their mirrored location when that location is free.
        planned_moves: list[tuple[int, int, int]] = []
        for r, c in pts:
            rr, cc = _mirror_cell(int(r), int(c), int(best_k))
            if 0 <= cc < w and int(g[rr, cc]) == int(col):
                continue
            if 0 <= cc < w and int(out[rr, cc]) == int(bg):
                planned_moves.append((int(r), int(c), int(cc)))

        for r, c, cc in planned_moves:
            out[int(r), int(c)] = int(bg)
            out[int(r), int(cc)] = int(col)

    return out


# --------------- generator (D1 flipped-noise variant) ---------------

# -- CFG: canvas & layout --
GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 sprite colors
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# -- CFG: clean sprite geometry --
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 46
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# -- CFG: movable noise --
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# -- CFG: axis gap thresholds --
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# -- CFG: retry limits --
SPRITE_RETRIES = 120
GEN_RETRIES = 300

# -- CFG: augmentation --
FLIP_PROB = 0.5


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying spec constraints."""
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
    """Pick a random (h, w, target_count, center_count), weighting toward denser fills."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _, _, tc, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left_count cells from the left half-grid while keeping a compact shape."""
    cells: set[tuple[int, int]] = set()
    inner_anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == inner_anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((int(r), int(c)))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(int(r), int(c)) for r in range(h) for c in range(side_w) if (int(r), int(c)) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    """Return boolean mask of which rows have a center-column pixel."""
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a vertically symmetric sprite and return (sprite_array, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mirror_c = w - 1 - c
            sprite[int(r), int(c)] = int(color)
            sprite[int(r), int(mirror_c)] = int(color)
        if center_count > 0:
            sprite[centers, side_w] = int(color)

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if _second_best_score(coords, true_k, width=w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into slot regions on a fresh grid. Returns (grid, placed_specs) or None."""
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, int(r0), int(c0), transparent=0)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        placed.append(spec)
    return out, placed


def _noise_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Cells adjacent to the clean shape whose mirrored location is in-bounds and empty."""
    _h, w = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    cols = np.arange(w)
    mirror_cols = int(true_k) - cols
    on_axis = mirror_cols == cols
    valid_mirror = (mirror_cols >= 0) & (mirror_cols < w) & ~on_axis

    adj[:, ~valid_mirror] = False
    for c in np.where(valid_mirror)[0]:
        adj[:, c] &= ~clean_mask[:, mirror_cols[c]]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_flipped_noise(inp: np.ndarray, out: np.ndarray, clean: np.ndarray, spec: dict) -> bool:
    """Add output-side noise to out and mirrored input-side noise to inp."""
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(clean == color)]
    clean_mask = clean == color
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False
    noise_target = random.randint(lo, hi)

    candidates = _noise_candidates(clean_mask, true_k)
    if len(candidates) < noise_target:
        return False

    chosen_out: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_input_coords = list(clean_coords)

    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen_out]
        if not viable:
            return False

        scored = []
        for cell in viable:
            mirrored_input = _mirror_cell(int(cell[0]), int(cell[1]), true_k)
            alt = _second_best_score(current_input_coords + [mirrored_input], true_k, width=GRID_SIZE)
            scored.append((int(alt), random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)

        chosen_out.append(cell)
        mirrored_input = _mirror_cell(int(cell[0]), int(cell[1]), true_k)
        current_input_coords.append(mirrored_input)
        forbidden.add((int(cell[0]), int(cell[1])))
        forbidden.add((int(mirrored_input[0]), int(mirrored_input[1])))

    if clean_count - _second_best_score(current_input_coords, true_k, width=GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen_out:
        rr, cc = _mirror_cell(int(r), int(c), true_k)
        out[int(r), int(c)] = int(color)
        inp[int(rr), int(cc)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean sprites, add mirrored noise pairs, validate via solver."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": int(color), "sprite": sprite, "axis_k_local": int(axis_k_local)})
    except ValueError:
        return None

    result = _slot_place_sprites(bg, sprite_specs, slots)
    if result is None:
        return None
    clean, placed_specs = result

    inp = clean.copy()
    out = clean.copy()
    for spec in placed_specs:
        if not _add_flipped_noise(inp, out, clean, spec):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
