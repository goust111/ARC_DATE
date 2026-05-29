# Combo variant of 8e5c0c38 (index 24)
# combo_spec: D2/1+D14/2
# Name: 轴对齐暗示方向

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste

# Design analysis for the fused D2/1 + D14/2 variant
# a) Concrete behavior trace:
#    - Original task: each color layer was denoised independently, and every layer only searched vertical axes.
#    - D2 alone would let each sprite choose vertical or horizontal independently, so one grid could mix directions.
#    - This combo changes that: the whole grid shares one mirror direction and one aligned axis line. Example:
#      if the three sprites all straddle column 10.5, then even a locally confusing middle sprite should still be
#      repaired left/right because the top and bottom sprites make the shared vertical alignment obvious. Likewise,
#      a horizontal sample stacks three sprites around the same row coordinate.
#    - Solver consequence: do not decide orientation per color in isolation. Score one candidate axis at a time as a
#      grid-wide hypothesis, summing same-color mirror matches across all sprites, then delete unmatched pixels under
#      that single winning axis.
# b) Complexity calibration:
#    - The freedom is still small: only two possible directions and one shared axis line.
#    - To make the fusion visible rather than cosmetic, the generator builds two strong anchor sprites and one weak
#      sprite. The weak sprite gets legal noise that can make its own best local axis disagree with the true one, so
#      the global aligned pattern is actually useful.
#    - The shared-axis search space is still learnable because there are only 43 doubled-axis positions per direction.
# c) Hidden assumption audit:
#    - Iterating by color is still safe only because the generator keeps one sprite per non-bg color.
#    - The original `cc = k - c` reflection assumed vertical-only symmetry; this combo must also support `rr = k - r`.
#    - The D2 single-variant assumption "each sprite picks its own best axis" is no longer valid here. The fused
#      solver assumes one global direction/axis pair for the entire grid.
#    - The original generator stored one axis per sprite independently. Here placement must guarantee all true axes are
#      exactly aligned, so global `k` is chosen first and every sprite is placed around it.
#    - Noise legality also changes subtly: the weak sprite is allowed to look better under a wrong local axis, but the
#      full grid must still prefer the true shared axis by a margin.
# d) Placement legality:
#    - To keep aligned axes exact and objects non-overlapping, vertical cases use three full-width row bands and
#      horizontal cases use three full-height column bands. Each sprite stays inside its own band.
#    - The shared axis is chosen near the canvas center so all parity-compatible sprite widths/heights fit.
#    - Output-first validation remains in force: build the clean output, add legal noise, and reject any sample that
#      the solver does not map back to the clean grid.
# e) Visual observability:
#    - A human can see that every sprite in one sample lines up on the same column or row, so the direction is no
#      longer a per-object guess. The weak sprite makes this useful rather than redundant.
#    - Both vertical and horizontal samples are generated, so the direction change from the original task is visible.
# f) Visual cleanliness:
#    - The three sprites remain color-separated, band-separated, and only receive nearby 4-adjacent noise. There are
#      no extra guide markers, legends, or overlapping decorations.
# g) Change inferability:
#    - No color-dependent rule is introduced. Colors only separate sprites; all reasoning is geometric.
# 9) New solver assumptions and generator guarantees:
#    - `main()` assumes one dominant shared axis candidate exists over the whole grid.
#    - The generator guarantees that all clean sprites use one aligned axis, that the shared candidate beats all other
#      global candidates by a margin, and that at least one noisy sprite is locally misleading in most accepted cases.


GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 sprite colors

# Three separated bands for the aligned-axis layouts.
ROW_BANDS = [(0, 5), (8, 13), (16, 21)]
COL_BANDS = [(0, 5), (8, 13), (16, 21)]

# Clean sprite geometry. The short side is capped at the band width/height; the long side keeps the original feel.
PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 34
SHORT_SIDE_MIN = 2
SHORT_SIDE_MAX = 6
LONG_SIDE_MAX = 12
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

# Noise.
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 10

# Axis discriminability.
ANCHOR_CLEAN_GAP_MIN = 5
WEAK_CLEAN_GAP_MIN = 1
WEAK_CLEAN_GAP_MAX = 4
ANCHOR_FINAL_LOCAL_GAP_MIN = 2
GLOBAL_FINAL_GAP_MIN = 4

# Generation retries.
SPRITE_RETRIES = 250
NOISE_RETRIES = 140
GEN_RETRIES = 700

# Shared axis locations stay central so every compatible sprite can fit.
SHARED_AXIS_CANDIDATES = list(range(17, 26))


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    """Return {axis_k: symmetric-partner count} for one axis direction."""
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
    """Pick the best axis over both directions; ties favor earlier vertical candidates."""
    best_direction = "vertical"
    best_k = 0
    best_score = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if score > best_score:
                best_direction = direction
                best_k = int(k)
                best_score = int(score)
    return best_direction, best_k, best_score


def _best_other_score(
    coords: list[tuple[int, int]], height: int, width: int, true_direction: str, true_k: int
) -> int:
    """Highest score among all axis hypotheses except the supplied one."""
    best = -1
    for direction in ("vertical", "horizontal"):
        for k, score in _axis_scores(coords, height, width, direction).items():
            if direction == true_direction and k == true_k:
                continue
            if score > best:
                best = int(score)
    return best


def _score_under_axis(
    coords: list[tuple[int, int]], height: int, width: int, direction: str, k: int
) -> int:
    """Score one explicit axis candidate for one sprite."""
    pts = {(int(r), int(c)) for r, c in coords}
    score = 0
    if direction == "vertical":
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
    else:
        for r, c in pts:
            rr = k - r
            if 0 <= rr < height and (rr, c) in pts:
                score += 1
    return int(score)


def _paired_under_axis(
    pts: set[tuple[int, int]], r: int, c: int, height: int, width: int, direction: str, k: int
) -> bool:
    """Return True iff (r,c) has a same-color mirror partner under the given axis."""
    if direction == "vertical":
        cc = k - c
        return 0 <= cc < width and (r, cc) in pts
    rr = k - r
    return 0 <= rr < height and (rr, c) in pts


def _collect_color_coords(grid: np.ndarray, bg: int) -> dict[int, list[tuple[int, int]]]:
    """Map each non-bg color to its coordinate list."""
    color_coords: dict[int, list[tuple[int, int]]] = {}
    for col in colors_present(grid, exclude={bg}):
        pts = np.argwhere(grid == int(col))
        if pts.size:
            color_coords[int(col)] = [(int(r), int(c)) for r, c in pts]
    return color_coords


def _best_global_axis(color_coords: dict[int, list[tuple[int, int]]], height: int, width: int) -> tuple[str, int, int]:
    """Pick the shared direction/axis maximizing total same-color mirror support over all sprites."""
    best_direction = "vertical"
    best_k = 0
    best_score = -1
    for direction in ("vertical", "horizontal"):
        limit = 2 * (width - 1) if direction == "vertical" else 2 * (height - 1)
        for k in range(limit + 1):
            score = 0
            for coords in color_coords.values():
                score += _score_under_axis(coords, height, width, direction, k)
            if score > best_score:
                best_direction = direction
                best_k = int(k)
                best_score = int(score)
    return best_direction, best_k, best_score


def _best_other_global_score(
    color_coords: dict[int, list[tuple[int, int]]], height: int, width: int, true_direction: str, true_k: int
) -> int:
    """Highest global score among all shared-axis hypotheses except the true one."""
    best = -1
    for direction in ("vertical", "horizontal"):
        limit = 2 * (width - 1) if direction == "vertical" else 2 * (height - 1)
        for k in range(limit + 1):
            if direction == true_direction and k == true_k:
                continue
            score = 0
            for coords in color_coords.values():
                score += _score_under_axis(coords, height, width, direction, k)
            if score > best:
                best = int(score)
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean the colored sprites by enforcing one aligned mirror axis for the whole grid.

    Input structure:
    - bg is the dominant background color; the other colors form three separated near-symmetric sprites.
    - Every sprite belongs to the same hidden mirror orientation: all are left/right symmetric or all are up/down
      symmetric.
    - Their true axes are aligned on one shared grid line, but each sprite also carries a few same-color noise pixels.

    Rule:
    - Evaluate every vertical axis and every horizontal axis as a single grid-wide hypothesis.
    - For each hypothesis, sum same-color mirror matches over all non-background colors.
    - Keep the direction/axis pair with the highest total score, then remove any pixel whose same-color mirror partner
      is missing across that winning shared axis.

    Key insight:
    - Orientation is no longer a purely local decision. One sprite may look ambiguous by itself, so the reliable cue is
      that all sprites agree on one aligned axis line across the entire grid.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    height, width = g.shape

    out = g.copy()
    color_coords = _collect_color_coords(g, bg)
    assert color_coords, "grid must contain at least one non-background color"

    direction, k, _ = _best_global_axis(color_coords, height, width)
    for coords in color_coords.values():
        pts = set(coords)
        for r, c in coords:
            if not _paired_under_axis(pts, r, c, height, width, direction, k):
                out[r, c] = bg
    return out


def _feasible_shape_options(
    max_short: int, max_long: int, required_axis_parity: int
) -> list[tuple[int, int, int, int]]:
    """Enumerate vertical-base shape specs that match the required local-axis parity."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(SHORT_SIDE_MAX, max_short)
    for target_count in range(PIXEL_COUNT_MIN, PIXEL_COUNT_MAX + 1):
        for h in range(SHORT_SIDE_MIN, h_max + 1):
            for side_w in ODD_SIDE_W_RANGE:
                w = 2 * side_w + 1
                if w > max_long or (w - 1) % 2 != required_axis_parity:
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
                if w > max_long or (w - 1) % 2 != required_axis_parity:
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


def _choose_shape_spec(max_short: int, max_long: int, required_axis_parity: int) -> tuple[int, int, int, int]:
    """Pick a random vertical-base shape spec, slightly favoring denser boxes."""
    options = _feasible_shape_options(max_short, max_long, required_axis_parity)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left-half pixels while keeping at least one pixel in each row."""
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
    """Rows carrying the center-column pixels for odd-width vertical-base sprites."""
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _vertical_candidate(color: int, max_short: int, max_long: int, axis_parity: int) -> tuple[np.ndarray, int]:
    """Build one perfectly vertically symmetric local sprite candidate."""
    h, w, target_count, center_count = _choose_shape_spec(max_short, max_long, axis_parity)
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


def _build_clean_sprite(color: int, direction: str, axis_parity: int, strength: str) -> tuple[np.ndarray, int]:
    """Build a clean sprite whose true axis matches the shared direction and parity."""
    for _ in range(SPRITE_RETRIES):
        if direction == "vertical":
            sprite, true_k = _vertical_candidate(color, SHORT_SIDE_MAX, LONG_SIDE_MAX, axis_parity)
        elif direction == "horizontal":
            base, true_k = _vertical_candidate(color, SHORT_SIDE_MAX, LONG_SIDE_MAX, axis_parity)
            sprite = base.T.copy()
        else:
            raise ValueError(f"unknown direction: {direction}")

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        if not coords:
            continue
        best_direction, best_k, best_score = _best_axis(coords, *sprite.shape)
        if best_direction != direction or best_k != true_k or best_score != len(coords):
            continue

        gap = best_score - _best_other_score(coords, *sprite.shape, direction, true_k)
        if strength == "anchor":
            if gap < ANCHOR_CLEAN_GAP_MIN:
                continue
        elif strength == "weak":
            if not (WEAK_CLEAN_GAP_MIN <= gap <= WEAK_CLEAN_GAP_MAX):
                continue
        else:
            raise ValueError(f"unknown strength: {strength}")
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _place_sprites(
    bg: int, sprite_specs: list[dict], direction: str, shared_k: int
) -> tuple[np.ndarray, list[dict]] | None:
    """Place sprites into aligned row or column bands."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []
    axis_parity = shared_k % 2

    for idx, spec in enumerate(sprite_specs):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if spec["axis_k_local"] % 2 != axis_parity:
            return None

        if direction == "vertical":
            band_r0, band_r1 = ROW_BANDS[idx]
            if h > band_r1 - band_r0 + 1:
                return None
            c0 = (shared_k - spec["axis_k_local"]) // 2
            if 2 * c0 != shared_k - spec["axis_k_local"]:
                return None
            if not (0 <= c0 <= GRID_SIZE - w):
                return None
            r0 = random.randint(band_r0, band_r1 - h + 1)
        else:
            band_c0, band_c1 = COL_BANDS[idx]
            if w > band_c1 - band_c0 + 1:
                return None
            r0 = (shared_k - spec["axis_k_local"]) // 2
            if 2 * r0 != shared_k - spec["axis_k_local"]:
                return None
            if not (0 <= r0 <= GRID_SIZE - h):
                return None
            c0 = random.randint(band_c0, band_c1 - w + 1)

        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["true_k"] = int(shared_k)
        placed.append(spec)

    return out, placed


def _noise_candidates(
    clean_mask: np.ndarray, available_mask: np.ndarray, direction: str, true_k: int
) -> list[tuple[int, int]]:
    """Find 4-adjacent background cells whose mirror under the true axis is not on the clean sprite."""
    height, width = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask
    adj &= available_mask

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


def _noise_bounds(clean_count: int, weak: bool) -> tuple[int, int]:
    """Noise count bounds for one sprite."""
    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if weak:
        lo = max(lo, 2)
    return lo, hi


def _choose_nonmirrored_subset(
    candidates: list[tuple[int, int]], count: int, direction: str, true_k: int
) -> list[tuple[int, int]] | None:
    """Randomly choose count cells without selecting both members of a true-axis mirror pair."""
    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    pool = list(candidates)
    random.shuffle(pool)
    for cell in pool:
        if cell in forbidden:
            continue
        chosen.append(cell)
        r, c = cell
        if direction == "vertical":
            mirror = (r, true_k - c)
            if 0 <= mirror[1] < GRID_SIZE:
                forbidden.add(mirror)
        else:
            mirror = (true_k - r, c)
            if 0 <= mirror[0] < GRID_SIZE:
                forbidden.add(mirror)
        if len(chosen) == count:
            return chosen
    return None


def _orthogonal_pair_bundles(
    candidates: list[tuple[int, int]], direction: str
) -> list[tuple[int, list[tuple[tuple[int, int], tuple[int, int]]]]]:
    """Find candidate-pair bundles that are mirrored across an axis in the wrong direction."""
    pts = set(candidates)
    bundles: dict[int, set[tuple[tuple[int, int], tuple[int, int]]]] = {}
    if direction == "vertical":
        for r, c in candidates:
            for rr in range(GRID_SIZE):
                mate = (rr, c)
                if mate == (r, c) or mate not in pts:
                    continue
                k = r + rr
                pair = tuple(sorted(((r, c), mate)))
                bundles.setdefault(k, set()).add(pair)
    else:
        for r, c in candidates:
            for cc in range(GRID_SIZE):
                mate = (r, cc)
                if mate == (r, c) or mate not in pts:
                    continue
                k = c + cc
                pair = tuple(sorted(((r, c), mate)))
                bundles.setdefault(k, set()).add(pair)
    return [(k, list(pairs)) for k, pairs in bundles.items() if pairs]


def _add_anchor_noise(inp: np.ndarray, out: np.ndarray, spec: dict, direction: str, bg: int) -> bool:
    """Add locally harmless noise to one anchor sprite."""
    color = spec["color"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_count = len(clean_coords)
    if clean_count == 0:
        return False

    lo, hi = _noise_bounds(clean_count, weak=False)
    if lo > hi:
        return False
    available = inp == bg
    candidates = _noise_candidates(out == color, available, direction, true_k)
    if len(candidates) < lo:
        return False

    for _ in range(NOISE_RETRIES):
        noise_target = random.randint(lo, min(hi, len(candidates)))
        chosen: list[tuple[int, int]] = []
        forbidden: set[tuple[int, int]] = set()
        current_coords = list(clean_coords)
        for _step in range(noise_target):
            viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
            if not viable:
                chosen = []
                break
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

        if not chosen:
            continue
        best_direction, best_k, best_score = _best_axis(current_coords, GRID_SIZE, GRID_SIZE)
        other_score = _best_other_score(current_coords, GRID_SIZE, GRID_SIZE, direction, true_k)
        if best_direction != direction or best_k != true_k:
            continue
        if best_score != clean_count:
            continue
        if best_score - other_score < ANCHOR_FINAL_LOCAL_GAP_MIN:
            continue
        for r, c in chosen:
            inp[r, c] = color
        return True
    return False


def _add_weak_noise(inp: np.ndarray, out: np.ndarray, spec: dict, direction: str, bg: int) -> bool:
    """Add noise that can mislead the weak sprite locally while keeping the true axis legal."""
    color = spec["color"]
    true_k = spec["true_k"]
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_count = len(clean_coords)
    if clean_count == 0:
        return False

    lo, hi = _noise_bounds(clean_count, weak=True)
    if lo > hi:
        return False
    available = inp == bg
    candidates = _noise_candidates(out == color, available, direction, true_k)
    if len(candidates) < lo:
        return False

    bundles = _orthogonal_pair_bundles(candidates, direction)
    random.shuffle(bundles)
    for _ in range(NOISE_RETRIES):
        noise_target = random.randint(lo, min(hi, len(candidates)))
        chosen: list[tuple[int, int]] = []

        if bundles:
            _bundle_k, pairs = random.choice(bundles)
            random.shuffle(pairs)
            used: set[tuple[int, int]] = set()
            for a, b in pairs:
                if a in used or b in used:
                    continue
                chosen.extend([a, b])
                used.add(a)
                used.add(b)
                if len(chosen) >= min(noise_target, 4):
                    break

        if len(chosen) > noise_target:
            chosen = chosen[:noise_target]

        remaining = noise_target - len(chosen)
        if remaining > 0:
            extras = [cell for cell in candidates if cell not in chosen]
            extra_subset = _choose_nonmirrored_subset(extras, remaining, direction, true_k)
            if extra_subset is None:
                continue
            chosen.extend(extra_subset)

        if len(chosen) != noise_target or len(set(chosen)) != len(chosen):
            continue

        current_coords = clean_coords + chosen
        best_direction, best_k, best_score = _best_axis(current_coords, GRID_SIZE, GRID_SIZE)
        if best_direction == direction and best_k == true_k:
            continue
        if _score_under_axis(current_coords, GRID_SIZE, GRID_SIZE, direction, true_k) != clean_count:
            continue
        if best_score <= clean_count:
            continue
        for r, c in chosen:
            inp[r, c] = color
        spec["weak_local_best"] = (best_direction, int(best_k), int(best_score))
        spec["weak_local_other"] = int(_best_other_score(current_coords, GRID_SIZE, GRID_SIZE, direction, true_k))
        return True
    return False


def _gen_attempt(return_meta: bool = False):
    """Single generation attempt. Optionally return metadata for diagnostics."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    direction = random.choice(["vertical", "horizontal"])
    shared_k = random.choice(SHARED_AXIS_CANDIDATES)
    axis_parity = shared_k % 2
    weak_index = random.randrange(len(obj_colors))

    sprite_specs: list[dict] = []
    try:
        for idx, color in enumerate(obj_colors):
            strength = "weak" if idx == weak_index else "anchor"
            sprite, axis_k_local = _build_clean_sprite(color, direction, axis_parity, strength)
            sprite_specs.append(
                {
                    "color": color,
                    "sprite": sprite,
                    "axis_k_local": axis_k_local,
                    "strength": strength,
                }
            )
    except ValueError:
        return None

    placed_result = _place_sprites(bg, sprite_specs, direction, shared_k)
    if placed_result is None:
        return None
    out, placed_specs = placed_result

    inp = out.copy()
    for spec in placed_specs:
        if spec["strength"] == "anchor":
            if not _add_anchor_noise(inp, out, spec, direction, bg):
                return None
    weak_spec = placed_specs[weak_index]
    if weak_spec["strength"] != "weak":
        return None
    if not _add_weak_noise(inp, out, weak_spec, direction, bg):
        return None

    if not np.array_equal(main(inp), out):
        return None

    color_coords = _collect_color_coords(inp, bg)
    best_direction, best_k, best_score = _best_global_axis(color_coords, GRID_SIZE, GRID_SIZE)
    if best_direction != direction or best_k != shared_k:
        return None
    global_other = _best_other_global_score(color_coords, GRID_SIZE, GRID_SIZE, direction, shared_k)
    if best_score - global_other < GLOBAL_FINAL_GAP_MIN:
        return None

    local_bests: dict[int, tuple[str, int, int]] = {}
    misleading_colors: list[int] = []
    for spec in placed_specs:
        coords = [tuple(x) for x in np.argwhere(inp == spec["color"])]
        local_best = _best_axis(coords, GRID_SIZE, GRID_SIZE)
        local_bests[spec["color"]] = local_best
        if local_best[:2] != (direction, shared_k):
            misleading_colors.append(spec["color"])

    if weak_spec["color"] not in misleading_colors:
        return None

    meta = {
        "direction": direction,
        "shared_k": int(shared_k),
        "weak_color": int(weak_spec["color"]),
        "misleading_colors": [int(c) for c in misleading_colors],
        "local_bests": {int(c): (d, int(k), int(s)) for c, (d, k, s) in local_bests.items()},
        "global_best": (best_direction, int(best_k), int(best_score)),
        "global_gap": int(best_score - global_other),
    }
    if return_meta:
        return inp, out, meta
    return inp


def generate_input() -> np.ndarray:
    """Generate one valid noisy input for the fused aligned-axis symmetry task."""
    for _ in range(GEN_RETRIES):
        result = _gen_attempt(return_meta=False)
        if result is not None:
            return result
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
