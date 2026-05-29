# Combo variant of 8e5c0c38 (index 8)
# combo_spec: D1/3+D4/1
# Name: 单侧噪声翻转修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste


# Design analysis for the combo variant
# a) Concrete behavior and trace
#    Base task: each non-background color is an almost vertically symmetric
#    sprite with a hidden per-sprite axis; old solver deleted unmatched pixels.
#    D1/3 changes the repair action from deletion to reflection-based movement.
#    D4/1 adds that, for any one sprite, all unmatched pixels lie on the same
#    side of that sprite's axis. Small example with axis k=8 (column 4):
#      symmetric core has pixels at columns 3 and 5
#      extra input pixels sit only on the left, say at columns 2 and 1
#      old output would erase 2 and 1
#      combo output clears 2 and 1, and places pixels at 6 and 7 instead
#    The asymmetry does not disappear; it flips to the opposite side. This is
#    the only interpretation consistent with D1/3's explicit "move, don't
#    delete" semantics. A literal "becomes perfectly symmetric" reading would
#    contradict the geometry of reflection because every moved pixel would leave
#    its mirror cell empty.
# b) Complexity calibration
#    The puzzle still revolves around recovering one hidden vertical axis per
#    color, so the core reasoning load is preserved. The new side constraint is
#    strong enough to be visible: in every generated sprite, all noisy protrusions
#    point left or all point right in the input, then they all jump across the
#    axis in the output. We keep the number of extra pixels bounded and local to
#    the sprite so the scene stays readable.
# c) Audit of old main() assumptions
#    - Axis search by same-color mirror score still works: the symmetric core is
#      unchanged and still dominates the score.
#    - The old action "set unmatched pixel to background" is invalid because the
#      new transform is relocation rather than deletion.
#    - Mirror existence checks must read the original grid g, not the evolving
#      output array, otherwise one earlier move could falsely justify a later one.
#    - A move now assumes that the reflected destination is inside the grid and
#      empty in the current output. The generator enforces this for every noisy
#      pixel, and the solver keeps the guard for safety.
#    - Because the combo constrains all noise to one side, the solver should not
#      depend on seeing noisy pixels on both sides. It only needs the best axis.
# d) Placement legality
#    Sprites are still built on isolated slots inside a 22x22 canvas. Noise
#    candidates are 4-adjacent to the clean core, off-axis, have in-bounds
#    mirrors, and their mirrors are not part of the clean core. Choosing from
#    separated slots prevents overlap between different colors both before and
#    after reflection.
# e) Visual observability
#    A human sees the same pattern repeated across three colors: one-sided
#    protrusions in the input all reappear on the opposite side in the output.
#    That is a visible spatial rule; no hidden numeric ordering or color-based
#    metadata is needed.
# f) Visual cleanliness
#    The generator keeps the original sparse three-slot layout and does not add
#    markers or extra scaffolding. Since each color occupies its own region,
#    the moved pixels are easy to attribute to the right sprite.
# g) Color inference
#    Colors are unchanged. They only partition the grid into independent sprites,
#    matching the original task's per-color decomposition.
# h) Output-first generation plan
#    Start from a clean symmetric core. Choose output-side landing cells on one
#    side of the axis, then reflect them to create the input-side noise. This
#    guarantees that running the solver moves those cells back to the chosen
#    landing side. Oracle check is main(input) == output.
# i) New solver assumptions and how generation enforces them
#    - One sprite per non-background color: generator uses three distinct colors.
#    - The intended axis must remain identifiable: score-gap thresholds are kept.
#    - Every noisy pixel must be movable: its mirrored landing cell is in-bounds
#      and background in the clean core.
#    - Noise must be single-sided per sprite, and in practice per whole sample:
#      each generated grid picks one global noisy side before optional whole-grid
#      flips, so the training examples provide repeated evidence for the pattern.


GRID_SIZE = 22
NUM_OBJECTS = 4
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

NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

SPRITE_RETRIES = 120
GEN_RETRIES = 300

FLIP_PROB = 0.5


def _mirror_cell(r: int, c: int, axis_k: int) -> tuple[int, int]:
    return int(r), int(axis_k) - int(c)


def _side_of_col(c: int, axis_k: int) -> int:
    delta = 2 * int(c) - int(axis_k)
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            rr, cc = _mirror_cell(int(r), int(c), int(k))
            if 0 <= cc < width and (rr, cc) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int = GRID_SIZE) -> int:
    scores = _axis_scores(coords, int(width))
    return max(score for k, score in scores.items() if int(k) != int(true_k))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect one-sided extra pixels across each sprite's own vertical symmetry axis.

    Input structure:
    - The background is the dominant color. Each other color forms one nearly vertically symmetric sprite with its own hidden axis.
    - A few same-color pixels are unmatched extras, and for any one sprite those extras all sit on the same side of that axis.

    Rule:
    - Process each non-background color independently and find the vertical axis that maximizes same-color mirror matches.
    - For every pixel without a same-color mirror partner, reflect it to the mirrored cell if that destination is inside the grid and currently background.
    - When such a move happens, the original location becomes background; already matched pixels stay fixed.

    Key insight:
    - The task is still per-color symmetry reasoning, but the repair is not denoising by deletion. The one-sided protrusions identify the noisy side, and the output is obtained by moving those protrusions to the opposite side of the inferred axis.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts_arr = np.argwhere(g == int(col))
        if pts_arr.size == 0:
            continue
        pts = [(int(r), int(c)) for r, c in pts_arr]

        best_score = -1
        best_k = 0
        for k in range(2 * (w - 1) + 1):
            score = 0
            for r, c in pts:
                rr, cc = _mirror_cell(int(r), int(c), int(k))
                if 0 <= cc < w and int(g[rr, cc]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)

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


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
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
    options = _feasible_shape_options(int(max_h), int(max_w))
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _, _, tc, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
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
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(int(max_h), int(max_w))
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(int(h), int(side_w), int(left_count))
        centers = _center_rows(int(h), int(center_count))

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
        spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        placed.append(spec)
    return clean, placed


def _noise_candidates(clean_mask: np.ndarray, true_k: int, desired_side: int) -> list[tuple[int, int]]:
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    cols = np.arange(clean_mask.shape[1])
    mirror_cols = int(true_k) - cols
    on_axis = mirror_cols == cols
    valid_mirror = (mirror_cols >= 0) & (mirror_cols < clean_mask.shape[1]) & ~on_axis

    side_mask = np.array([_side_of_col(int(c), int(true_k)) == int(desired_side) for c in cols], dtype=bool)
    adj[:, ~(valid_mirror & side_mask)] = False

    for c in np.where(valid_mirror & side_mask)[0]:
        adj[:, c] &= ~clean_mask[:, mirror_cols[c]]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_single_sided_flipped_noise(
    inp: np.ndarray,
    out: np.ndarray,
    clean: np.ndarray,
    spec: dict,
    input_side: int,
) -> bool:
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

    output_side = -int(input_side)
    candidates = _noise_candidates(clean_mask, true_k, output_side)
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

        mirrored_input = _mirror_cell(int(cell[0]), int(cell[1]), true_k)
        if _side_of_col(int(mirrored_input[1]), true_k) != int(input_side):
            return False

        chosen_out.append(cell)
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
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(int(color), int(max_h), int(max_w))
            sprite_specs.append({"color": int(color), "sprite": sprite, "axis_k_local": int(axis_k_local)})
    except ValueError:
        return None

    result = _slot_place_sprites(int(bg), sprite_specs, slots)
    if result is None:
        return None
    clean, placed_specs = result

    inp = clean.copy()
    out = clean.copy()
    input_side = random.choice([-1, 1])
    for spec in placed_specs:
        if not _add_single_sided_flipped_noise(inp, out, clean, spec, input_side):
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
