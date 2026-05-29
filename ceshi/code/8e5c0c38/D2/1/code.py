# Variant of 8e5c0c38
# Dimension: D2 — 对称轴方向 (transform)
# Change: 仅搜索垂直对称轴（遍历所有可能的垂直轴位置，选择得分最高的轴）... → 每个精灵独立选择最佳轴方向（垂直或水平）

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste

# Design analysis for the D2 variant
# a) Concrete behavior trace:
#    - The grid still contains exactly 3 colored sprites on one background. Each sprite is almost mirror-symmetric,
#      but now its true mirror axis may be vertical or horizontal.
#    - Example: a color forming {(1,2),(2,2),(3,2)} plus mirrored side pixels across a horizontal axis should keep
#      top/bottom partners and delete only the extra asymmetric noise. The old solver would only scan vertical axes,
#      so it could misread that sprite and erase valid pixels that are only horizontally paired.
#    - New behavior: for each color, score every vertical axis and every horizontal axis, choose the best one for that
#      specific sprite, then delete only pixels lacking a same-color partner across that chosen axis.
# b) Complexity calibration:
#    - Only 2 axis directions are allowed, so the new freedom is visible but still learnable.
#    - Each generated grid deliberately mixes directions (at least one vertical and one horizontal sprite), making the
#      per-object orientation choice visible instead of collapsing back to a single global rule.
#    - Clean sprites are rejected unless their true axis beats every other vertical/horizontal candidate by a margin,
#      preventing ambiguous plus-shapes or doubly-symmetric blobs from hiding the variant.
# c) Hidden assumption audit from the original main():
#    - Iterating per color remains valid only because the generator still uses one distinct color per sprite.
#    - The original `cc = k - c` logic assumed all symmetry checks are column reflections; this must be generalized
#      so horizontal candidates use `rr = k - r` instead.
#    - The original generator stored global axis position as `local_k + 2*c0`; that is only correct for vertical
#      axes. Horizontal sprites require `local_k + 2*r0`.
#    - Noise validity used "mirror column is background"; this must also branch on direction so horizontal noise checks
#      the mirrored row instead.
# d) Placement legality:
#    - The original slot layout is preserved so objects stay separated and never overlap.
#    - Horizontal clean sprites are built by transposing a vertically generated local sprite, which keeps them inside
#      the same slot bounds while reusing the old pixel-count and density regime.
#    - Output-first validation is preserved: generate clean output, add oriented noise, then reject any sample where
#      `main(input)` does not reconstruct the clean grid.
# e) Visual observability:
#    - Because each sample contains both directions, a human can see that some noisy objects repair left/right while
#      others repair up/down. This makes the D2 change observable in just a few examples.
# f) Visual cleanliness:
#    - Objects remain in distant slots, use distinct colors, and only gain local 4-adjacent noise. No extra markers or
#      overlays are introduced, so the axis direction must be inferred from the shape itself.
# g) Color-rule audit:
#    - Colors still only label separate sprites. No sorting, mapping, or behavior choice depends on color id.
# 9) New solver assumptions and generator guarantees:
#    - `main()` now assumes each non-background color is one independent sprite with a single dominant mirror axis.
#    - The generator enforces that dominant-axis margin for both clean and noisy states, and it guarantees a mixed
#      vertical/horizontal orientation set in every generated grid.


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    """Return {axis_k: symmetric_partner_count} for every axis in one direction."""
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    if direction == "vertical":
        for k in range(2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                cc = k - c
                if 0 <= cc < width and (r, cc) in pts:
                    score += 1
            scores[k] = score
    elif direction == "horizontal":
        for k in range(2 * (height - 1) + 1):
            score = 0
            for r, c in pts:
                rr = k - r
                if 0 <= rr < height and (rr, c) in pts:
                    score += 1
            scores[k] = score
    else:
        raise ValueError(f"unknown direction: {direction}")
    return scores


def _best_axis(coords: list[tuple[int, int]], height: int, width: int) -> tuple[str, int, int]:
    """Pick the best axis over both directions; ties stay with the earliest vertical candidate."""
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
    """Highest score among all axes except the supplied true axis."""
    best = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if direction == true_direction and k == true_k:
                continue
            if score > best:
                best = int(score)
    return best


def _paired_under_axis(
    pts: set[tuple[int, int]], r: int, c: int, height: int, width: int, direction: str, k: int
) -> bool:
    """Return True iff (r,c) has a same-color mirror partner across the chosen axis."""
    if direction == "vertical":
        cc = k - c
        return 0 <= cc < width and (r, cc) in pts
    rr = k - r
    return 0 <= rr < height and (rr, c) in pts


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing its strongest mirror symmetry.

    Input structure:
    - bg = background. Several non-bg colors appear, one sprite per color.
    - Each sprite is almost symmetric, but its own best axis may be vertical or horizontal.
    - A few same-color pixels are noise and do not have a mirrored mate across that sprite's true axis.

    Rule:
    - For each non-bg color independently, score every vertical axis and every horizontal axis.
    - Choose the direction/axis pair with the highest score for that color.
    - Remove any pixel whose same-color mirror partner is missing across the chosen axis.

    Key insight: The denoising is still per sprite, but the axis orientation is no longer fixed; each color must first
    decide whether left-right or up-down symmetry explains it better.
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
        pts = {(int(r), int(c)) for r, c in pts_arr}
        coords = [(int(r), int(c)) for r, c in pts_arr]
        direction, k, _ = _best_axis(coords, height, width)

        for r, c in coords:
            if not _paired_under_axis(pts, r, c, height, width, direction, k):
                out[r, c] = bg

    return out


# --------------- generator (codex variant D2) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 sprite colors
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

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3  # true best score beats every other vertical/horizontal candidate by >= this
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 180
GEN_RETRIES = 500

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying the local spec."""
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
    """Pick a random clean-sprite shape spec, slightly favoring denser boxes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left_count cells from the left half-grid, keeping at least one pixel in each row."""
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
    """Return which rows carry a center-column pixel for odd-width sprites."""
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
    """Build a clean sprite whose unique best axis has the requested direction."""
    for _ in range(SPRITE_RETRIES):
        if direction == "vertical":
            sprite, true_k = _vertical_candidate(color, max_h, max_w)
        elif direction == "horizontal":
            base, true_k = _vertical_candidate(color, max_w, max_h)
            sprite = base.T.copy()
        else:
            raise ValueError(f"unknown direction: {direction}")

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        if len(coords) == 0:
            continue
        best_direction, best_k, best_score = _best_axis(coords, *sprite.shape)
        if best_direction != direction or best_k != true_k:
            continue
        if best_score != len(coords):
            continue
        other_score = _best_other_score(coords, *sprite.shape, direction, true_k)
        if best_score - other_score < SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _sample_orientations() -> list[str]:
    """Guarantee each grid contains both axis directions."""
    sprite_count = NUM_OBJECTS - 1
    horizontal_count = random.choice([1, 2])
    horizontal_idx = set(random.sample(range(sprite_count), horizontal_count))
    return ["horizontal" if i in horizontal_idx else "vertical" for i in range(sprite_count)]


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into slot regions on a fresh grid. Returns (grid, placed_specs) or None."""
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


def _noise_candidates(clean_mask: np.ndarray, direction: str, true_k: int) -> list[tuple[int, int]]:
    """Find 4-adjacent background cells whose mirror across the true axis is not on the clean sprite."""
    height, width = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    if direction == "vertical":
        cols = np.arange(width)
        mirror_cols = true_k - cols
        on_axis = mirror_cols == cols
        adj[:, on_axis] = False
        valid = (mirror_cols >= 0) & (mirror_cols < width) & ~on_axis
        for c in np.where(valid)[0]:
            adj[:, c] &= ~clean_mask[:, mirror_cols[c]]
    elif direction == "horizontal":
        rows = np.arange(height)
        mirror_rows = true_k - rows
        on_axis = mirror_rows == rows
        adj[on_axis, :] = False
        valid = (mirror_rows >= 0) & (mirror_rows < height) & ~on_axis
        for r in np.where(valid)[0]:
            adj[r, :] &= ~clean_mask[mirror_rows[r], :]
    else:
        raise ValueError(f"unknown direction: {direction}")

    rows_arr, cols_arr = np.where(adj)
    return list(zip(rows_arr.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Add axis-breaking noise to one sprite in inp. Returns False if uniqueness constraints fail."""
    color = spec["color"]
    direction = spec["axis_dir"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
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

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return False

        scored = []
        for cell in viable:
            alt = _best_other_score(current_coords + [cell], GRID_SIZE, GRID_SIZE, direction, true_k)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        r, c = cell
        forbidden.add(cell)
        if direction == "vertical":
            mirror = (r, true_k - c)
            if 0 <= mirror[1] < GRID_SIZE:
                forbidden.add(mirror)
        else:
            mirror = (true_k - r, c)
            if 0 <= mirror[0] < GRID_SIZE:
                forbidden.add(mirror)

    best_direction, best_k, best_score = _best_axis(current_coords, GRID_SIZE, GRID_SIZE)
    other_score = _best_other_score(current_coords, GRID_SIZE, GRID_SIZE, direction, true_k)
    if best_direction != direction or best_k != true_k:
        return False
    if best_score != clean_count:
        return False
    if best_score - other_score < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean sprites, place them, add noise, validate via solver."""
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
        if not _add_noise(inp, out, spec):
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
