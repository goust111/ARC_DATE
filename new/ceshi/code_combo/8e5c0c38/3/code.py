# Combo variant of 8e5c0c38 (index 3)
# combo_spec: D1/3+D2/1
# Name: 自适应方向噪声翻转

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    The original task used only vertical axes and deleted unmatched pixels. This
#    combo changes both parts together: each color first chooses whether a
#    vertical or horizontal mirror axis explains it best, then every unmatched
#    pixel is reflected across that chosen axis instead of being erased.
#    Small trace:
#      clean core (vertical axis k=6): pixels at columns 2 and 4 around center 3
#      noisy input adds an unmatched pixel at column 1
#      original task output: pixel at 1 disappears
#      combo output: pixel at 1 is cleared and a pixel appears at column 5
#    The same idea applies to a horizontal axis by swapping rows instead of
#    columns. This means the visible asymmetry migrates to the opposite side.
#    The combo description says "restore perfect symmetry", but D1/3's concrete
#    operation cannot do that: moving a lone protrusion across the axis preserves
#    one-sidedness. The implementation therefore follows the component-level
#    semantics of D1/3, because that is the only internally consistent reading.
# b) Complexity calibration
#    Only two axis directions are allowed, which keeps the search learnable, but
#    every generated grid mixes directions so the orientation choice is visible.
#    Each sprite also receives at least one movable protrusion, so the flip
#    behavior is observable and does not collapse back to deletion.
# c) Hidden assumption audit of the original main()
#    - Iterating per color is still valid only if each sprite owns a unique
#      color; the generator preserves that.
#    - The old `cc = k - c` reflection logic assumed vertical-only symmetry; the
#      combo must branch so horizontal candidates use `rr = k - r`.
#    - The old solver mutated `out` by deleting unmatched pixels. That update is
#      no longer valid; unmatched pixels now require a source-clear plus
#      destination-write.
#    - Match tests must read the original input grid, not the evolving output,
#      otherwise an earlier move could fake a partner for a later pixel.
#    - Movement now assumes the mirrored destination is background; the generator
#      enforces that for the noisy cells it creates.
# d) Placement legality
#    The original three-slot layout is preserved, which keeps the three sprites
#    spatially separated. Horizontal clean sprites are produced by transposing a
#    valid vertical local sprite, so they still satisfy the same size and density
#    bounds and still fit their slots after placement.
# e) Visual observability
#    Samples will show both left-right and up-down flips in the same grid. A
#    human can compare input/output and notice that protrusions jump across the
#    axis rather than vanishing, and that different colors may jump in different
#    directions.
# f) Visual cleanliness
#    Noise remains 4-adjacent to its own sprite and objects remain far apart, so
#    the moved protrusions are easy to attribute to the correct object. No extra
#    markers or color remapping are introduced.
# g) Change inferability
#    Colors are not used as codes for behavior. They only separate sprites. The
#    axis direction is inferred from geometry, and the move target is the mirror
#    of the noisy pixel, which is directly visible in the examples.
# h) Output-first generation plan
#    Build a symmetric clean core first. Choose output-side protrusion cells that
#    are adjacent to the core and have an in-bounds empty mirror under the true
#    axis. Write those cells into the output, then reflect them to create the
#    input-side protrusions. Oracle check requires `main(input) == output`.
# i) New solver assumptions and generator guarantees
#    - Each color corresponds to exactly one dominant-axis sprite.
#    - The true axis must beat every other vertical/horizontal candidate by a
#      margin in both the clean and noisy states.
#    - Every generated protrusion has an empty mirrored destination, so the flip
#      branch actually executes.
#    - Each grid mixes at least one vertical and one horizontal sprite, so D2/1
#      is exercised rather than degenerating back to a single global direction.


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    """Return {axis_k: symmetric partner count} for one reflection direction."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    if direction == "vertical":
        for k in range(2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < width and (int(r), int(cc)) in pts:
                    score += 1
            scores[int(k)] = int(score)
        return scores
    if direction == "horizontal":
        for k in range(2 * (height - 1) + 1):
            score = 0
            for r, c in pts:
                rr = int(k) - int(r)
                if 0 <= rr < height and (int(rr), int(c)) in pts:
                    score += 1
            scores[int(k)] = int(score)
        return scores
    raise ValueError(f"unknown direction: {direction}")


def _best_axis(coords: list[tuple[int, int]], height: int, width: int) -> tuple[str, int, int]:
    """Choose the best mirror axis across both directions."""
    best_direction = "vertical"
    best_k = 0
    best_score = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if int(score) > best_score:
                best_score = int(score)
                best_direction = direction
                best_k = int(k)
    return best_direction, best_k, best_score


def _best_other_score(
    coords: list[tuple[int, int]], height: int, width: int, true_direction: str, true_k: int
) -> int:
    """Highest score among all candidate axes except the designated one."""
    best = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if direction == true_direction and int(k) == int(true_k):
                continue
            if int(score) > best:
                best = int(score)
    return best


def _mirror_cell(r: int, c: int, direction: str, axis_k: int) -> tuple[int, int]:
    """Reflect one cell across the chosen vertical or horizontal axis."""
    if direction == "vertical":
        return int(r), int(axis_k) - int(c)
    if direction == "horizontal":
        return int(axis_k) - int(r), int(c)
    raise ValueError(f"unknown direction: {direction}")


def _paired_under_axis(
    pts: set[tuple[int, int]], r: int, c: int, height: int, width: int, direction: str, axis_k: int
) -> bool:
    """Return True iff the point has a same-color mirror partner under the axis."""
    rr, cc = _mirror_cell(int(r), int(c), direction, int(axis_k))
    return 0 <= rr < height and 0 <= cc < width and (int(rr), int(cc)) in pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Flip each sprite's unmatched protrusions across its own strongest mirror axis.

    Input structure:
    - bg is the background. Each non-bg color forms one nearly mirror-symmetric sprite.
    - Different sprites may favor different axis directions: some are best explained by a vertical axis, others by a horizontal one.
    - A few same-color pixels lack mirror partners and act as one-sided protrusions attached near the symmetric core.

    Rule:
    - For each non-bg color independently, score every vertical axis and every horizontal axis, then choose the highest-scoring direction/axis pair.
    - Any same-color pixel that already has a mirror partner stays where it is.
    - Any same-color pixel without a partner is reflected to its mirrored cell if that mirrored cell is inside the grid and currently background; otherwise it stays put.

    Key insight:
    - The task is no longer "delete bad pixels under a fixed vertical symmetry". Each color first identifies its own preferred symmetry direction, then the asymmetric extras jump across that axis instead of disappearing.
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
        pts = {(int(r), int(c)) for r, c in coords}
        direction, axis_k, _score = _best_axis(coords, height, width)

        planned_moves: list[tuple[int, int, int, int]] = []
        for r, c in coords:
            if _paired_under_axis(pts, int(r), int(c), height, width, direction, int(axis_k)):
                continue
            rr, cc = _mirror_cell(int(r), int(c), direction, int(axis_k))
            if 0 <= rr < height and 0 <= cc < width and int(out[int(rr), int(cc)]) == int(bg):
                planned_moves.append((int(r), int(c), int(rr), int(cc)))

        for r, c, rr, cc in planned_moves:
            out[int(r), int(c)] = int(bg)
            out[int(rr), int(cc)] = int(col)

    return out


# --------------- generator (combo D1/3 + D2/1) ---------------

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

# -- CFG: movable protrusions --
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# -- CFG: dominant-axis thresholds --
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# -- CFG: retry limits --
SPRITE_RETRIES = 180
GEN_RETRIES = 500

# -- CFG: augmentation --
FLIP_PROB = 0.5


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate feasible local symmetric sprite specs within the original bounds."""
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
    """Pick one clean-sprite spec, slightly favoring denser bboxes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left-half cells while keeping the local sprite compact and row-spanning."""
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

    remaining = int(left_count) - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(int(r), int(c)) for r in range(h) for c in range(side_w) if (int(r), int(c)) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    """Return which rows receive a center pixel for odd-width local sprites."""
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _vertical_candidate(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build one perfectly vertically symmetric local sprite."""
    h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
    side_w = w // 2
    left_count = (target_count - center_count) // 2
    left = _left_cells(h, side_w, left_count)
    centers = _center_rows(h, center_count)

    sprite = np.zeros((h, w), dtype=int)
    for r, c in left:
        mirror_c = int(w - 1 - c)
        sprite[int(r), int(c)] = int(color)
        sprite[int(r), int(mirror_c)] = int(color)
    if center_count > 0:
        sprite[centers, side_w] = int(color)
    return sprite, int(w - 1)


def _build_clean_sprite(color: int, max_h: int, max_w: int, direction: str) -> tuple[np.ndarray, int]:
    """Build a clean sprite whose unique best axis has the requested direction."""
    for _ in range(SPRITE_RETRIES):
        if direction == "vertical":
            sprite, true_k = _vertical_candidate(int(color), max_h, max_w)
        elif direction == "horizontal":
            base, true_k = _vertical_candidate(int(color), max_w, max_h)
            sprite = base.T.copy()
        else:
            raise ValueError(f"unknown direction: {direction}")

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        if not coords:
            continue
        best_direction, best_k, best_score = _best_axis(coords, *sprite.shape)
        if best_direction != direction or int(best_k) != int(true_k):
            continue
        if int(best_score) != len(coords):
            continue
        other_score = _best_other_score(coords, *sprite.shape, direction, int(true_k))
        if int(best_score) - int(other_score) < SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite")


def _sample_orientations() -> list[str]:
    """Guarantee that each generated grid mixes vertical and horizontal sprites."""
    sprite_count = NUM_OBJECTS - 1
    horizontal_count = random.choice([1, 2])
    horizontal_idx = set(random.sample(range(sprite_count), horizontal_count))
    return ["horizontal" if i in horizontal_idx else "vertical" for i in range(sprite_count)]


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place clean sprites into separated slots on a fresh background grid."""
    clean = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(clean, sprite, int(r0), int(c0), transparent=0)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        if spec["axis_dir"] == "vertical":
            spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        else:
            spec["true_k"] = int(spec["axis_k_local"] + 2 * r0)
        placed.append(spec)
    return clean, placed


def _noise_candidates(clean_mask: np.ndarray, direction: str, true_k: int) -> list[tuple[int, int]]:
    """Return output-side protrusion cells whose mirrored input-side cell is in-bounds and empty."""
    height, width = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    if direction == "vertical":
        cols = np.arange(width)
        mirror_cols = int(true_k) - cols
        on_axis = mirror_cols == cols
        valid = (mirror_cols >= 0) & (mirror_cols < width) & ~on_axis
        adj[:, ~valid] = False
        for c in np.where(valid)[0]:
            adj[:, c] &= ~clean_mask[:, mirror_cols[c]]
    elif direction == "horizontal":
        rows = np.arange(height)
        mirror_rows = int(true_k) - rows
        on_axis = mirror_rows == rows
        valid = (mirror_rows >= 0) & (mirror_rows < height) & ~on_axis
        adj[~valid, :] = False
        for r in np.where(valid)[0]:
            adj[r, :] &= ~clean_mask[mirror_rows[r], :]
    else:
        raise ValueError(f"unknown direction: {direction}")

    rows_arr, cols_arr = np.where(adj)
    return list(zip(rows_arr.tolist(), cols_arr.tolist()))


def _add_flipped_noise(inp: np.ndarray, out: np.ndarray, clean: np.ndarray, spec: dict) -> bool:
    """Add output-side protrusions to out and mirrored input-side protrusions to inp."""
    color = int(spec["color"])
    direction = spec["axis_dir"]
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(clean == int(color))]
    clean_mask = clean == int(color)
    clean_count = len(clean_coords)
    if clean_count == 0:
        return False

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False
    noise_target = random.randint(lo, hi)

    candidates = _noise_candidates(clean_mask, direction, true_k)
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
            mirrored_input = _mirror_cell(int(cell[0]), int(cell[1]), direction, true_k)
            alt = _best_other_score(current_input_coords + [mirrored_input], GRID_SIZE, GRID_SIZE, direction, true_k)
            scored.append((int(alt), random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = int(scored[0][0])
        pool = [cell for alt, _rnd, cell in scored if int(alt) <= best_alt + 1]
        cell = random.choice(pool)

        chosen_out.append(cell)
        mirrored_input = _mirror_cell(int(cell[0]), int(cell[1]), direction, true_k)
        current_input_coords.append(mirrored_input)
        forbidden.add((int(cell[0]), int(cell[1])))
        forbidden.add((int(mirrored_input[0]), int(mirrored_input[1])))

    best_direction, best_k, best_score = _best_axis(current_input_coords, GRID_SIZE, GRID_SIZE)
    other_score = _best_other_score(current_input_coords, GRID_SIZE, GRID_SIZE, direction, true_k)
    if best_direction != direction or int(best_k) != int(true_k):
        return False
    if int(best_score) != clean_count:
        return False
    if int(best_score) - int(other_score) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen_out:
        rr, cc = _mirror_cell(int(r), int(c), direction, true_k)
        out[int(r), int(c)] = int(color)
        inp[int(rr), int(cc)] = int(color)
    return True


def _gen_attempt() -> np.ndarray | None:
    """Build one combo-valid sample and reject it unless the solver reproduces its output."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)
    directions = _sample_orientations()

    sprite_specs = []
    try:
        for color, slot, direction in zip(obj_colors, slots, directions):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(int(color), max_h, max_w, direction)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "axis_dir": direction,
                    "axis_k_local": int(axis_k_local),
                }
            )
    except ValueError:
        return None

    result = _slot_place_sprites(int(bg), sprite_specs, slots)
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
