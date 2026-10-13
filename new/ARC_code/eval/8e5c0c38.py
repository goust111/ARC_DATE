from __future__ import annotations
import numpy as np
from arc_utils import colors_present, get_bg
import math
import random
from arc_utils import paste

def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing vertical mirror symmetry.

    Input structure:
    - bg = background. Several non-bg colors, each forming a nearly-symmetric pattern about some vertical axis.
    - Some pixels in each color are "noise" that breaks the symmetry.

    Rule:
    - For each non-bg color, find the vertical axis that best explains its pixel layout.
    - Remove any pixel that lacks a same-color mirror counterpart across that axis.

    Key insight: Each color has its own independent symmetry axis; the task is per-color denoising, not whole-grid symmetry.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    H, W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    # Search all possible vertical axes: k in [0, 2*(W-1)] corresponds to axis at c = k/2.
    for col in colors:
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

        # Delete unpaired pixels under the chosen vertical axis.
        k = int(best_k)
        for r, c in pts:
            rr = int(r)
            cc = int(k) - int(c)
            if not (0 <= cc < W and int(g[rr, cc]) == int(col)):
                out[rr, int(c)] = int(bg)

    return out

# --------------- generator (codex-att3) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_OBJECTS = 4                 # bg + 3 sprite colors
LAYOUTS = [
    [(0, 0, 7, 7), (0, 14, 7, 21), (14, 5, 21, 16)],
    [(0, 0, 8, 5), (0, 16, 8, 21), (15, 5, 21, 16)],
]

# ── CFG: clean sprite geometry ──
PIXEL_COUNT_MIN = 6             # min clean pixel count
PIXEL_COUNT_MAX = 46            # max clean pixel count
HEIGHT_MIN = 2                  # min sprite bbox height
HEIGHT_MAX = 9                  # max sprite bbox height
ODD_SIDE_W_RANGE = range(1, 6) # half-width for odd-width sprites (integer axis)
EVEN_SIDE_W_RANGE = range(2, 7) # half-width for even-width sprites (half-integer axis)
DENSITY_LO = 0.35              # min fill ratio within bbox
DENSITY_HI = 0.89              # max fill ratio within bbox

# ── CFG: noise ──
NOISE_FRAC_LO = 0.11           # min noise pixels as fraction of clean count
NOISE_FRAC_HI = 0.33           # max noise pixels as fraction of clean count
NOISE_MAX = 12                  # absolute max noise pixels per sprite

# ── CFG: axis gap thresholds ──
SPRITE_AXIS_GAP_MIN = 3        # best - 2nd-best score >= this for clean sprite
FINAL_AXIS_GAP_MIN = 2         # best - 2nd-best score >= this after adding noise

# ── CFG: retry limits ──
SPRITE_RETRIES = 120            # attempts to build a single clean sprite
GEN_RETRIES = 300               # outer generate_input retries

# ── CFG: augmentation ──
FLIP_PROB = 0.5                 # probability of each flip (ud / lr)


def _axis_scores(coords: list[tuple[int, int]], width: int = GRID_SIZE) -> dict[int, int]:
    """Return {axis_k: symmetric_pair_count} for every candidate vertical axis."""
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


def _second_best_score(coords: list[tuple[int, int]], true_k: int) -> int:
    """Return the highest axis score excluding the true axis."""
    scores = _axis_scores(coords)
    return max(s for k, s in scores.items() if k != true_k)


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying spec constraints.

    For odd-width sprites (integer axis), center_count >= 1.
    For even-width sprites (half-integer axis), center_count == 0.
    """
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            # Odd-width sprites (integer axis): w = 2*side_w + 1
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
            # Even-width sprites (half-integer axis): w = 2*side_w
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
    """Choose left_count cells from the left half-grid (h x side_w), ensuring connectivity."""
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
    """Return boolean mask of which rows have a center-column pixel."""
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a vertically-symmetric sprite and return (sprite_array, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
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

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if _second_best_score(coords, true_k) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(bg: int, sprite_specs: list[dict],
                        slots: list[tuple[int, int, int, int]]) -> tuple[np.ndarray, list[dict]] | None:
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
        spec["true_k"] = spec["axis_k_local"] + 2 * c0
        placed.append(spec)
    return out, placed


def _noise_candidates(clean_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Find all cells 4-adjacent to the clean shape that are valid noise placements.

    A cell is valid if: not on the clean shape, its mirror across true_k is not on the
    clean shape, and it is not on the symmetry axis itself.
    """
    H, W = clean_mask.shape
    # 4-neighbor dilation minus the shape itself
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask

    # Build mirror-column index: for each column c, mirror_c = true_k - c
    cols = np.arange(W)
    mirror_cols = true_k - cols

    # Exclude cells on the axis (mirror_c == c)
    on_axis = (mirror_cols == cols)
    adj[:, on_axis] = False

    # Exclude cells whose mirror is on the clean shape
    valid_mirror = (mirror_cols >= 0) & (mirror_cols < W) & ~on_axis
    for c in np.where(valid_mirror)[0]:
        adj[:, c] &= ~clean_mask[:, mirror_cols[c]]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict) -> bool:
    """Add noise pixels to one sprite in inp. Returns False if constraints can't be met."""
    color = spec["color"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False
    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, true_k)
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
            alt = _second_best_score(current_coords + [cell], true_k)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda t: (t[0], t[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        r, c = cell
        forbidden.add(cell)
        mirror = (r, true_k - c)
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    if clean_count - _second_best_score(current_coords, true_k) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _gen_attempt() -> np.ndarray | None:
    """Single attempt: build clean sprites, place them, add noise, validate via solver."""
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
