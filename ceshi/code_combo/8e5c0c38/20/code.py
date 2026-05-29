# Combo variant of 8e5c0c38 (index 20)
# combo_spec: D9/1+D11/0
# Name: 大噪声迭代修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste

# Design analysis for the fused variant:
# a) Concrete behavior:
#    Original task can usually pick the correct vertical axis in one pass because noise is sparse.
#    Here each sprite carries 40-60% extra pixels, so deleting every mismatch in one shot is too brittle.
#    The fused solver instead peels only the currently outermost unmatched fringe, then re-estimates the
#    axis. Example: if a symmetric 7-column core has attached noise at distances 6 and 2 from the axis,
#    round 1 removes distance-6 cells, round 2 re-evaluates the smaller pattern, and later rounds remove
#    the inner leftover fringe until the clean core is the stable mirrored subset.
# b) Complexity calibration:
#    The generator keeps exactly 3 colors and fixed slot layouts, so objects remain visually separable.
#    Noise is heavy enough to be visible, but still attached near the sprite so the puzzle is not chaotic.
# c) Implicit assumptions broken by D9/1 + D11/0:
#    The original `delete every unmatched pixel immediately` step assumes one axis estimate is already
#    trustworthy. With heavy noise, this can over-prune. The new solver still searches for a strong
#    mirrored core, but removes only the farthest currently-unmatched layer before recomputing. The
#    repair is still deletion-only, so the generator validates that no clean core pixel ever needs to
#    be restored.
# d) Placement legality:
#    Sprites are placed inside disjoint slots with margins when possible. Noise growth is clipped to the
#    owning slot, so colors never overlap and nothing leaves the grid.
# e) Visual observability:
#    A human can see "same-color blobs become cleaner and more symmetric after repeated pruning". The
#    visible difference from the base task is that one pruning wave is insufficient on accepted samples.
# f) Visual cleanliness:
#    Noise is attached by 4-neighbor growth, preventing isolated confetti across the canvas.
# g) Non-degeneration check:
#    The generator rejects any sample where one round already solves it, or where the iterative solver
#    differs from the clean output. Accepted samples therefore exhibit the intended fused behavior.


def _paired_subset(coords: set[tuple[int, int]], axis_k: int, width: int) -> set[tuple[int, int]]:
    """Return the subset whose mirror exists across axis_k."""
    kept: set[tuple[int, int]] = set()
    for r, c in coords:
        mirror_c = axis_k - c
        if 0 <= mirror_c < width and (r, mirror_c) in coords:
            kept.add((r, c))
    return kept


def _bbox_area(coords: set[tuple[int, int]]) -> int:
    if not coords:
        return 0
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return (max(rows) - min(rows) + 1) * (max(cols) - min(cols) + 1)


def _best_axis(coords: set[tuple[int, int]], width: int) -> tuple[int, set[tuple[int, int]]]:
    """Choose the axis whose mirrored subset is both large and compact."""
    if not coords:
        return 0, set()

    cols = [c for _, c in coords]
    center_hint = min(cols) + max(cols)
    best_key: tuple[int, int, int, int, int] | None = None
    best_axis_k = 0
    best_keep: set[tuple[int, int]] = set()

    for axis_k in range(2 * (width - 1) + 1):
        keep = _paired_subset(coords, axis_k, width)
        pair_count = len(keep)
        if pair_count == 0:
            key = (-10**9, -10**9, -10**9, -10**9, -10**9)
        else:
            area = _bbox_area(keep)
            axis_slack = sum(abs(2 * c - axis_k) for _, c in keep)
            compact_score = 4 * pair_count - area
            key = (compact_score, pair_count, -area, -axis_slack, -abs(axis_k - center_hint))
        if best_key is None or key > best_key:
            best_key = key
            best_axis_k = axis_k
            best_keep = keep

    return best_axis_k, best_keep


def _repair(
    grid: np.ndarray,
    *,
    max_rounds: int | None = None,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, object]]:
    """Iteratively peel the farthest currently-unmatched layer until stable."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    limit = height * width if max_rounds is None else int(max_rounds)
    changed_rounds = 0
    axes_history: list[dict[int, int]] = []

    for _ in range(limit):
        changed = False
        nxt = out.copy()
        round_axes: dict[int, int] = {}

        for color in colors_present(out, exclude={bg}):
            coords = {tuple(map(int, rc)) for rc in np.argwhere(out == int(color))}
            if not coords:
                continue
            axis_k, keep = _best_axis(coords, width)
            round_axes[int(color)] = int(axis_k)
            if len(keep) < len(coords):
                changed = True
                unmatched = list(coords - keep)
                max_dist = max(abs(2 * c - axis_k) for _, c in unmatched)
                for r, c in unmatched:
                    if abs(2 * c - axis_k) == max_dist:
                        nxt[r, c] = bg

        axes_history.append(round_axes)
        out = nxt
        if not changed:
            break
        changed_rounds += 1

    if return_meta:
        return out, {"changed_rounds": changed_rounds, "axes_history": axes_history}
    return out


def _original_one_pass(grid: np.ndarray) -> np.ndarray:
    """Reference baseline: the original task's single-pass raw-pair pruning."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, width = g.shape
    out = g.copy()

    for color in colors_present(g, exclude={bg}):
        coords = np.argwhere(g == int(color))
        if coords.size == 0:
            continue

        best_score = -1
        best_axis_k = 0
        for axis_k in range(2 * (width - 1) + 1):
            score = 0
            for r, c in coords:
                mirror_c = axis_k - int(c)
                if 0 <= mirror_c < width and int(g[int(r), mirror_c]) == int(color):
                    score += 1
            if score > best_score:
                best_score = score
                best_axis_k = axis_k

        for r, c in coords:
            mirror_c = best_axis_k - int(c)
            if not (0 <= mirror_c < width and int(g[int(r), mirror_c]) == int(color)):
                out[int(r), int(c)] = bg

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repeatedly strip each color down to its most plausible vertical mirror core.

    Input structure:
    - One background color fills most cells.
    - Three non-background sprites of different colors appear in separate regions.
    - Each sprite started vertically symmetric, but now carries heavy attached same-color noise.

    Rule:
    - For every color layer, find the vertical axis whose mirrored subset is strongest and most compact.
    - Among the pixels that fail that axis test, delete only the outermost distance band, then recompute.
    - Keep iterating until another full sweep removes nothing.

    Key insight:
    - With large noise, aggressive one-pass deletion is unreliable.
    - Repeatedly peeling the most suspicious fringe lets the axis settle onto the clean symmetric core.
    """
    return _repair(grid)


# --------------- generator ---------------

# Canvas and layout.
GRID_SIZE = 22
NUM_OBJECTS = 4
LAYOUTS = [
    [(0, 0, 9, 9), (0, 12, 9, 21), (12, 4, 21, 17)],
    [(0, 1, 8, 10), (0, 12, 8, 20), (13, 3, 21, 18)],
]

# Clean sprite geometry.
PIXEL_COUNT_MIN = 10
PIXEL_COUNT_MAX = 30
HEIGHT_MIN = 3
HEIGHT_MAX = 8
ODD_SIDE_W_RANGE = range(2, 6)
EVEN_SIDE_W_RANGE = range(2, 5)
DENSITY_LO = 0.35
DENSITY_HI = 0.82

# High-noise fused variant.
NOISE_FRAC_LO = 0.40
NOISE_FRAC_HI = 0.60
NOISE_MAX = 18

# Axis separation and retries.
SPRITE_AXIS_GAP_MIN = 4
SPRITE_RETRIES = 150
NOISE_SEARCH_ATTEMPTS = 120
GEN_RETRIES = 500

# Placement and augmentation.
INNER_MARGIN = 1
FLIP_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for axis_k in range(2 * (width - 1) + 1):
        scores[axis_k] = len(_paired_subset(pts, axis_k, width))
    return scores


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    scores = _axis_scores(coords, width)
    return max(score for axis_k, score in scores.items() if axis_k != true_k)


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
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
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
        cells.add((r, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
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
        if _second_best_score(coords, true_k, w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int,
    sprite_specs: list[dict[str, object]],
    slots: list[tuple[int, int, int, int]],
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict[str, object]] = []

    for spec, slot in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = np.array(spec["sprite"], dtype=int)
        h, w = sprite.shape
        r0s, c0s, r1s, c1s = slot
        slot_h = r1s - r0s + 1
        slot_w = c1s - c0s + 1
        if h > slot_h or w > slot_w:
            return None

        margin = INNER_MARGIN if h + 2 * INNER_MARGIN <= slot_h and w + 2 * INNER_MARGIN <= slot_w else 0
        r0_lo = r0s + margin
        r0_hi = r1s - margin - h + 1
        c0_lo = c0s + margin
        c0_hi = c1s - margin - w + 1
        if r0_lo > r0_hi or c0_lo > c0_hi:
            return None

        r0 = random.randint(r0_lo, r0_hi)
        c0 = random.randint(c0_lo, c0_hi)
        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = r0
        spec["c0"] = c0
        spec["slot"] = slot
        spec["true_k"] = int(spec["axis_k_local"]) + 2 * c0
        placed.append(spec)

    return out, placed


def _in_slot(cell: tuple[int, int], slot: tuple[int, int, int, int]) -> bool:
    r, c = cell
    r0, c0, r1, c1 = slot
    return r0 <= r <= r1 and c0 <= c <= c1


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _true_axis_ok(
    cell: tuple[int, int],
    occupied: set[tuple[int, int]],
    true_k: int,
) -> bool:
    r, c = cell
    if 2 * c == true_k:
        return False
    mirror = (r, true_k - c)
    return not (0 <= mirror[1] < GRID_SIZE and mirror in occupied)


def _frontier_candidates(
    clean: set[tuple[int, int]],
    noise: set[tuple[int, int]],
    slot: tuple[int, int, int, int],
    true_k: int,
) -> list[tuple[int, int]]:
    occupied = clean | noise
    candidates: set[tuple[int, int]] = set()
    for cell in occupied:
        for nxt in _neighbors4(cell):
            if not _in_slot(nxt, slot) or nxt in occupied:
                continue
            if _true_axis_ok(nxt, occupied, true_k):
                candidates.add(nxt)
    return list(candidates)


def _weighted_choice(cells: list[tuple[int, int]], true_k: int, favored_side: int) -> tuple[int, int]:
    weights: list[float] = []
    for cell in cells:
        _, c = cell
        side = -1 if 2 * c < true_k else 1
        side_bias = 1.30 if side == favored_side else 1.0
        dist_bias = 1.0 + 0.25 * abs(2 * c - true_k)
        attach_bias = 1.0 + 0.40 * sum(nei in cells for nei in _neighbors4(cell))
        weights.append(side_bias * dist_bias * attach_bias)
    return random.choices(cells, weights=weights, k=1)[0]


def _propose_noise(
    clean: set[tuple[int, int]],
    slot: tuple[int, int, int, int],
    true_k: int,
    target: int,
) -> set[tuple[int, int]] | None:
    noise: set[tuple[int, int]] = set()
    favored_side = random.choice([-1, 1])

    for _ in range(target * 12):
        if len(noise) >= target:
            break

        frontier = _frontier_candidates(clean, noise, slot, true_k)
        if not frontier:
            return None

        cell = _weighted_choice(frontier, true_k, favored_side)
        noise.add(cell)

    if len(noise) != target:
        return None
    return noise


def _apply_noise(clean_grid: np.ndarray, color: int, noise: set[tuple[int, int]]) -> np.ndarray:
    inp = np.array(clean_grid, copy=True)
    for r, c in noise:
        inp[r, c] = color
    return inp


def _find_noise_for_sprite(
    bg: int,
    color: int,
    clean_grid: np.ndarray,
    slot: tuple[int, int, int, int],
    true_k: int,
) -> tuple[set[tuple[int, int]], dict[str, int]] | None:
    clean = {tuple(map(int, rc)) for rc in np.argwhere(clean_grid == color)}
    clean_count = len(clean)

    lo = max(4, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return None

    for _ in range(NOISE_SEARCH_ATTEMPTS):
        target = random.randint(lo, hi)
        noise = _propose_noise(clean, slot, true_k, target)
        if noise is None:
            continue
        if len({abs(2 * c - true_k) for _, c in noise}) < 2:
            continue

        inp = _apply_noise(clean_grid, color, noise)
        one_step, meta1 = _repair(inp, max_rounds=1, return_meta=True)
        full, meta_full = _repair(inp, return_meta=True)
        if not np.array_equal(full, clean_grid):
            continue
        if np.array_equal(one_step, clean_grid):
            continue
        if int(meta_full["changed_rounds"]) < 2:
            continue

        one_step_extra = int(np.count_nonzero(one_step == color) - clean_count)
        if one_step_extra <= 0:
            continue

        if np.array_equal(_original_one_pass(inp), clean_grid):
            continue

        return noise, {
            "noise_count": target,
            "changed_rounds": int(meta_full["changed_rounds"]),
            "one_step_extra": one_step_extra,
            "one_step_rounds": int(meta1["changed_rounds"]),
        }

    return None


def _gen_attempt() -> np.ndarray | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs: list[dict[str, object]] = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1 - 2 * INNER_MARGIN
            max_w = slot[3] - slot[1] + 1 - 2 * INNER_MARGIN
            max_h = max(max_h, HEIGHT_MIN)
            max_w = max(max_w, 2 * min(ODD_SIDE_W_RANGE) + 1)
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": color, "sprite": sprite, "axis_k_local": axis_k_local})
    except ValueError:
        return None

    placed_result = _slot_place_sprites(bg, sprite_specs, slots)
    if placed_result is None:
        return None
    out, placed_specs = placed_result

    inp = np.array(out, copy=True)
    behavior_hits = 0
    for spec in placed_specs:
        color = int(spec["color"])
        single_clean = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
        paste(single_clean, np.array(spec["sprite"], dtype=int), int(spec["r0"]), int(spec["c0"]), transparent=0)
        noise_result = _find_noise_for_sprite(
            bg,
            color,
            single_clean,
            spec["slot"],  # type: ignore[arg-type]
            int(spec["true_k"]),
        )
        if noise_result is None:
            return None
        noise, metrics = noise_result
        for r, c in noise:
            inp[r, c] = color
        if metrics["changed_rounds"] >= 2 and metrics["one_step_extra"] > 0:
            behavior_hits += 1

    if behavior_hits == 0:
        return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp).copy()
        out = np.flipud(out).copy()
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp).copy()
        out = np.fliplr(out).copy()

    if not np.array_equal(main(inp), out):
        return None
    if np.array_equal(_repair(inp, max_rounds=1), out):
        return None
    if np.array_equal(_original_one_pass(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
