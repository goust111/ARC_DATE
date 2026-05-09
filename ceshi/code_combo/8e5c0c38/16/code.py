# Combo variant of 8e5c0c38 (index 16)
# combo_spec: D2/1+D4/1
# Name: 单侧噪声暗示轴方向

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg, paste

# Design analysis for the D2/1 + D4/1 combo variant
# a) Concrete behavior trace:
#    - Original behavior: each color layer only searched vertical axes, so a nearly symmetric sprite was repaired by
#      deleting pixels with no left-right partner. Noise could sit on either side, so orientation never had to be
#      inferred from the noise pattern itself.
#    - Combo behavior: each sprite may truly be vertical-symmetric or horizontal-symmetric, and every noise pixel of
#      that sprite lies on only one side of the true axis. Example: start from a horizontally symmetric core, add a
#      few same-color pixels only above the axis, and spread those extra pixels to both the left and right halves.
#      Under the true horizontal axis, all unmatched pixels are on one side; under the wrong vertical axis, unmatched
#      pixels appear on both sides. The solver should therefore prefer the axis whose leftovers form a one-sided band.
#    - Difference: axis direction is no longer a fixed global assumption, and the asymmetry pattern itself carries
#      directional evidence instead of being arbitrary clutter.
# b) Complexity calibration:
#    - Only two axis directions are allowed, keeping the search space small enough for ARC.
#    - Each grid mixes directions and also includes one clue-critical sprite whose clean core is symmetric both ways;
#      that sprite cannot be resolved by "best mirror score" alone, so the D4 single-side rule remains visible.
#    - The generator rejects noisy objects unless the fused axis heuristic chooses the intended axis uniquely, which
#      avoids ambiguous failures while still making the variant visually distinct from the original task.
# c) Hidden assumption audit from the original main():
#    - Iterating one color at a time still assumes one sprite per color, so the generator preserves distinct colors.
#    - The old `cc = k - c` reflection logic assumed every candidate axis was vertical; the combo solver must also
#      reflect rows with `rr = k - r` for horizontal candidates.
#    - The old axis selector depended only on partner counts. That assumption breaks here because a doubly symmetric
#      core plus one-sided noise can tie on raw partner count across directions; the new selector therefore inspects
#      where unmatched pixels fall relative to each candidate axis.
#    - The old generator stored axis position with a column offset only; horizontal axes instead need row offsets.
#    - The old noise generator allowed bilateral leftovers. That would erase the D4 clue, so the combo generator
#      explicitly filters noise candidates to one chosen side of the true axis.
# d) Placement legality:
#    - The original slot layout is reused, so sprites stay separated and inside the 22x22 canvas.
#    - Directional sprites are built locally and then pasted into non-overlapping slots.
#    - The clue-critical sprite is still placed inside one slot and validated after noise addition; rejected samples
#      never reach the final dataset.
# e) Visual observability:
#    - A human can inspect an input/output pair and notice that some objects lose left/right protrusions while another
#      loses top-only or bottom-only protrusions. Because the critical sprite's noise spans both halves of the wrong
#      orientation, the axis direction can be inferred from visible asymmetry, not from hidden metadata.
# f) Visual cleanliness:
#    - Only three colored sprites are present, each in its own distant region.
#    - Noise remains 4-adjacent to its sprite and uses the same color, so there are no extra markers or overlays.
#    - Restricting noise to one side makes the perturbations look coherent instead of speckled around the whole object.
# g) Color-rule audit:
#    - Color ids still only separate sprites; no ordering, mapping, or axis choice depends on numeric color value.
# 9) New solver assumptions and generator guarantees:
#    - `main()` now assumes the intended axis is the candidate whose unmatched pixels are all on one side and whose
#      mirror score is strongest among such candidates.
#    - The generator guarantees this by forcing one-sided noise for every sprite and by including at least one sprite
#      where a score-only D2 solver would choose the wrong orientation, proving the fusion behavior is active.

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
NOISE_PICK_RETRIES = 180

# ── CFG: axis distinctness ──
SPRITE_AXIS_GAP_MIN = 3

# ── CFG: retry limits ──
SPRITE_RETRIES = 220
GEN_RETRIES = 700

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _paired_under_axis(
    pts: set[tuple[int, int]], r: int, c: int, height: int, width: int, direction: str, k: int
) -> bool:
    """Return True iff (r,c) has a same-color mirror partner across the given axis."""
    if direction == "vertical":
        cc = k - c
        return 0 <= cc < width and (r, cc) in pts
    if direction == "horizontal":
        rr = k - r
        return 0 <= rr < height and (rr, c) in pts
    raise ValueError(f"unknown direction: {direction}")


def _side_of_axis(direction: str, k: int, r: int, c: int) -> int:
    """Return -1 / 0 / +1 for the side of a point relative to an axis."""
    if direction == "vertical":
        delta = 2 * c - k
    elif direction == "horizontal":
        delta = 2 * r - k
    else:
        raise ValueError(f"unknown direction: {direction}")
    if delta < 0:
        return -1
    if delta > 0:
        return 1
    return 0


def _mirror_cell(direction: str, k: int, cell: tuple[int, int]) -> tuple[int, int]:
    """Reflect one cell across a candidate axis."""
    r, c = cell
    if direction == "vertical":
        return r, k - c
    if direction == "horizontal":
        return k - r, c
    raise ValueError(f"unknown direction: {direction}")


def _axis_scores(
    coords: list[tuple[int, int]], height: int, width: int, direction: str
) -> dict[int, int]:
    """Return {axis_k: partner-count} for every axis in one direction."""
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


def _axis_profile(
    coords: list[tuple[int, int]], height: int, width: int, direction: str, k: int
) -> dict[str, object]:
    """Summarize how well one candidate axis explains a sprite."""
    pts = {(int(r), int(c)) for r, c in coords}
    paired_count = 0
    unmatched: list[tuple[int, int]] = []
    for r, c in coords:
        if _paired_under_axis(pts, r, c, height, width, direction, k):
            paired_count += 1
        else:
            unmatched.append((int(r), int(c)))

    side_set = {_side_of_axis(direction, k, r, c) for r, c in unmatched}
    one_sided = not unmatched or (0 not in side_set and len(side_set) == 1)
    return {
        "direction": direction,
        "k": int(k),
        "paired_count": int(paired_count),
        "unmatched": unmatched,
        "one_sided": bool(one_sided),
        "key": (int(one_sided), int(paired_count), -len(unmatched)),
    }


def _best_axis(coords: list[tuple[int, int]], height: int, width: int) -> tuple[str, int, dict[str, object]]:
    """Pick the best axis, prioritizing one-sided residuals and then mirror score."""
    best_profile: dict[str, object] | None = None
    for direction in ("vertical", "horizontal"):
        limit = width if direction == "vertical" else height
        for k in range(2 * (limit - 1) + 1):
            profile = _axis_profile(coords, height, width, direction, k)
            if best_profile is None or profile["key"] > best_profile["key"]:
                best_profile = profile
    assert best_profile is not None
    return str(best_profile["direction"]), int(best_profile["k"]), best_profile


def _best_axis_by_score(coords: list[tuple[int, int]], height: int, width: int) -> tuple[str, int, int]:
    """Reproduce the old D2-only selector: maximize partner count, keeping earliest vertical ties."""
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
    """Highest raw partner count among all candidate axes except the supplied one."""
    best = -1
    for direction in ("vertical", "horizontal"):
        limit = width if direction == "vertical" else height
        for k in range(2 * (limit - 1) + 1):
            if direction == true_direction and k == true_k:
                continue
            score = _axis_profile(coords, height, width, direction, k)["paired_count"]
            if score > best:
                best = int(score)
    return best


def _best_other_key(
    coords: list[tuple[int, int]], height: int, width: int, true_direction: str, true_k: int
) -> tuple[int, int, int]:
    """Best fused ranking key among all candidate axes except the supplied one."""
    best: tuple[int, int, int] = (-1, -1, -10**9)
    for direction in ("vertical", "horizontal"):
        limit = width if direction == "vertical" else height
        for k in range(2 * (limit - 1) + 1):
            if direction == true_direction and k == true_k:
                continue
            key = _axis_profile(coords, height, width, direction, k)["key"]
            if key > best:
                best = key
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each color layer by enforcing the mirror axis best supported by one-sided noise.

    Input structure:
    - bg = background. Exactly several non-bg colors appear, one sprite per color.
    - Every sprite is near a mirror symmetry, but its true axis may be vertical or horizontal.
    - Same-color noise pixels are attached only on one side of that sprite's true axis.

    Rule:
    - For each non-bg color independently, evaluate every vertical axis and every horizontal axis.
    - Prefer the candidate axis whose unmatched pixels all lie on a single side; among those, keep the one with the
      strongest mirror-partner count.
    - Delete any pixel that has no same-color mirror partner across the chosen axis.

    Key insight: The task is still per-sprite denoising, but axis direction must now be inferred jointly from symmetry
    score and from the fact that true noise forms a one-sided leftover, not a two-sided mismatch cloud.
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
        pts = {(int(r), int(c)) for r, c in pts_arr}
        direction, k, _ = _best_axis(coords, height, width)
        for r, c in coords:
            if not _paired_under_axis(pts, r, c, height, width, direction, k):
                out[r, c] = bg

    return out


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate local clean-sprite specs for a single-axis symmetric sprite."""
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
    """Pick one feasible clean-sprite spec, slightly favoring denser boxes."""
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    """Choose left-half cells while keeping at least one pixel in every row."""
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


def _build_directional_sprite(color: int, max_h: int, max_w: int, direction: str) -> tuple[np.ndarray, int]:
    """Build a clean sprite whose unique strongest axis has the requested direction."""
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
        best_direction, best_k, best_profile = _best_axis(coords, *sprite.shape)
        if best_direction != direction or best_k != true_k:
            continue
        if best_profile["paired_count"] != len(coords):
            continue
        other_score = _best_other_score(coords, *sprite.shape, direction, true_k)
        if len(coords) - other_score < SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build directional sprite")


def _orbit_cells(h: int, w: int, r: int, c: int) -> set[tuple[int, int]]:
    """Return the orbit of one canonical cell under vertical and horizontal reflection."""
    rows = {r, h - 1 - r}
    cols = {c, w - 1 - c}
    return {(rr, cc) for rr in rows for cc in cols}


def _build_bisymmetric_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, dict[str, int]]:
    """Build a clean sprite symmetric about both the central vertical and horizontal axes."""
    h_lo = max(3, HEIGHT_MIN)
    w_lo = 3
    h_hi = min(HEIGHT_MAX, max_h)
    w_hi = min(2 * max(EVEN_SIDE_W_RANGE), max_w)
    if h_lo > h_hi or w_lo > w_hi:
        raise ValueError("slot too small for clue-critical sprite")

    for _ in range(SPRITE_RETRIES):
        h = random.randint(h_lo, h_hi)
        w = random.randint(w_lo, w_hi)
        canonical = [(r, c) for r in range((h + 1) // 2) for c in range((w + 1) // 2)]
        random.shuffle(canonical)

        chosen = {(0, 0)}
        density_pick = random.uniform(0.25, 0.62)
        for cell in canonical:
            if random.random() < density_pick:
                chosen.add(cell)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in chosen:
            for rr, cc in _orbit_cells(h, w, r, c):
                sprite[rr, cc] = color

        coords = [tuple(x) for x in np.argwhere(sprite == color)]
        count = len(coords)
        area = h * w
        if not (PIXEL_COUNT_MIN <= count <= PIXEL_COUNT_MAX):
            continue
        if not (math.ceil(DENSITY_LO * area) <= count <= math.floor(DENSITY_HI * area)):
            continue
        if not any(_side_of_axis("vertical", w - 1, r, c) == -1 for r, c in coords):
            continue
        if not any(_side_of_axis("vertical", w - 1, r, c) == 1 for r, c in coords):
            continue
        if not any(_side_of_axis("horizontal", h - 1, r, c) == -1 for r, c in coords):
            continue
        if not any(_side_of_axis("horizontal", h - 1, r, c) == 1 for r, c in coords):
            continue
        if _axis_profile(coords, h, w, "vertical", w - 1)["paired_count"] != count:
            continue
        if _axis_profile(coords, h, w, "horizontal", h - 1)["paired_count"] != count:
            continue

        other_score = -1
        for direction in ("vertical", "horizontal"):
            limit = w if direction == "vertical" else h
            for k in range(2 * (limit - 1) + 1):
                if direction == "vertical" and k == w - 1:
                    continue
                if direction == "horizontal" and k == h - 1:
                    continue
                other_score = max(
                    other_score,
                    int(_axis_profile(coords, h, w, direction, k)["paired_count"]),
                )
        if count - other_score < SPRITE_AXIS_GAP_MIN:
            continue

        return sprite, {"vertical": w - 1, "horizontal": h - 1}
    raise ValueError("failed to build bi-symmetric sprite")


def _sample_roles() -> list[dict[str, object]]:
    """Guarantee a mixed-direction grid with one clue-critical horizontal sprite."""
    sprite_count = NUM_OBJECTS - 1
    clue_idx = random.randrange(sprite_count)
    roles: list[dict[str, object] | None] = [None] * sprite_count
    roles[clue_idx] = {"mode": "ambiguous", "axis_dir": "horizontal", "force_clue": True}

    remaining = [idx for idx in range(sprite_count) if idx != clue_idx]
    vertical_idx = random.choice(remaining)
    roles[vertical_idx] = {"mode": "directional", "axis_dir": "vertical", "force_clue": False}

    last_idx = next(idx for idx in remaining if idx != vertical_idx)
    roles[last_idx] = {
        "mode": "directional",
        "axis_dir": random.choice(["vertical", "horizontal"]),
        "force_clue": False,
    }
    assert all(role is not None for role in roles)
    return [dict(role) for role in roles if role is not None]


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict[str, object]], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    """Place sprites into slot regions on a fresh grid."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict[str, object]] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = np.array(spec["sprite"], dtype=int)
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        spec["r0"] = r0
        spec["c0"] = c0
        if spec["axis_dir"] == "vertical":
            spec["true_k"] = int(spec["axis_k_local"]) + 2 * c0
        else:
            spec["true_k"] = int(spec["axis_k_local"]) + 2 * r0
        if "cross_axis_dir" in spec and "cross_axis_k_local" in spec:
            if spec["cross_axis_dir"] == "vertical":
                spec["cross_k"] = int(spec["cross_axis_k_local"]) + 2 * c0
            else:
                spec["cross_k"] = int(spec["cross_axis_k_local"]) + 2 * r0
        placed.append(spec)
    return out, placed


def _noise_candidates(
    clean_mask: np.ndarray, direction: str, true_k: int, side_sign: int
) -> list[tuple[int, int]]:
    """Find valid 4-adjacent noise cells on one chosen side of the true axis."""
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
        valid_side = (2 * cols - true_k < 0) if side_sign < 0 else (2 * cols - true_k > 0)
        adj[:, ~valid_side] = False
        adj[:, on_axis] = False
        valid = (mirror_cols >= 0) & (mirror_cols < width) & ~on_axis
        for c in np.where(valid)[0]:
            adj[:, c] &= ~clean_mask[:, mirror_cols[c]]
    elif direction == "horizontal":
        rows = np.arange(height)
        mirror_rows = true_k - rows
        on_axis = mirror_rows == rows
        valid_side = (2 * rows - true_k < 0) if side_sign < 0 else (2 * rows - true_k > 0)
        adj[~valid_side, :] = False
        adj[on_axis, :] = False
        valid = (mirror_rows >= 0) & (mirror_rows < height) & ~on_axis
        for r in np.where(valid)[0]:
            adj[r, :] &= ~clean_mask[mirror_rows[r], :]
    else:
        raise ValueError(f"unknown direction: {direction}")

    rows_arr, cols_arr = np.where(adj)
    return list(zip(rows_arr.tolist(), cols_arr.tolist()))


def _sample_general_noise(candidates: list[tuple[int, int]], noise_target: int) -> list[tuple[int, int]] | None:
    """Randomly sample a one-sided noise set for a directional sprite."""
    if len(candidates) < noise_target:
        return None
    return random.sample(candidates, noise_target)


def _sample_clue_noise(
    candidates: list[tuple[int, int]],
    noise_target: int,
    cross_dir: str,
    cross_k: int,
) -> list[tuple[int, int]] | None:
    """Sample one-sided noise that fools a score-only selector but not the fused selector."""
    left = [cell for cell in candidates if _side_of_axis(cross_dir, cross_k, *cell) == -1]
    right = [cell for cell in candidates if _side_of_axis(cross_dir, cross_k, *cell) == 1]
    if noise_target < 4 or not left or not right:
        return None

    mirrored_pairs = [
        (cell, _mirror_cell(cross_dir, cross_k, cell))
        for cell in left
        if _mirror_cell(cross_dir, cross_k, cell) in candidates
    ]
    if not mirrored_pairs:
        return None
    random.shuffle(mirrored_pairs)

    for left_pair, right_pair in mirrored_pairs:
        left_single_options = [
            cell
            for cell in left
            if cell != left_pair and _mirror_cell(cross_dir, cross_k, cell) != right_pair
        ]
        right_single_options = [
            cell
            for cell in right
            if cell != right_pair and _mirror_cell(cross_dir, cross_k, cell) != left_pair
        ]
        random.shuffle(left_single_options)
        random.shuffle(right_single_options)
        for left_single in left_single_options:
            for right_single in right_single_options:
                if right_single == _mirror_cell(cross_dir, cross_k, left_single):
                    continue
                chosen = [left_pair, right_pair, left_single, right_single]
                forbidden = {
                    left_pair,
                    right_pair,
                    left_single,
                    right_single,
                    _mirror_cell(cross_dir, cross_k, left_single),
                    _mirror_cell(cross_dir, cross_k, right_single),
                }
                pool = [cell for cell in candidates if cell not in forbidden]
                random.shuffle(pool)
                for cell in pool:
                    if len(chosen) >= noise_target:
                        break
                    chosen.append(cell)
                    forbidden.add(cell)
                if len(chosen) != noise_target:
                    continue
                side_set = {_side_of_axis(cross_dir, cross_k, r, c) for r, c in chosen}
                if -1 in side_set and 1 in side_set:
                    return chosen
    return None


def _validate_noisy_object(
    coords: list[tuple[int, int]], spec: dict[str, object], clean_count: int
) -> bool:
    """Check that the noisy sprite really expresses the combo rule."""
    direction = str(spec["axis_dir"])
    true_k = int(spec["true_k"])
    true_profile = _axis_profile(coords, GRID_SIZE, GRID_SIZE, direction, true_k)
    if not bool(true_profile["one_sided"]):
        return False
    if int(true_profile["paired_count"]) != clean_count:
        return False

    best_direction, best_k, best_profile = _best_axis(coords, GRID_SIZE, GRID_SIZE)
    if best_direction != direction or best_k != true_k:
        return False
    if tuple(best_profile["key"]) <= _best_other_key(coords, GRID_SIZE, GRID_SIZE, direction, true_k):
        return False

    if bool(spec.get("force_clue", False)):
        raw_direction, raw_k, _ = _best_axis_by_score(coords, GRID_SIZE, GRID_SIZE)
        if raw_direction == direction and raw_k == true_k:
            return False
        cross_dir = str(spec["cross_axis_dir"])
        cross_k = int(spec["cross_k"])
        if bool(_axis_profile(coords, GRID_SIZE, GRID_SIZE, cross_dir, cross_k)["one_sided"]):
            return False

    return True


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict[str, object]) -> bool:
    """Add same-color noise on one side of the true axis."""
    color = int(spec["color"])
    direction = str(spec["axis_dir"])
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_mask = out == color
    clean_count = len(clean_coords)
    if clean_count == 0:
        return False

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if bool(spec.get("force_clue", False)):
        lo = max(lo, 4)
    if lo > hi:
        return False

    for _ in range(NOISE_PICK_RETRIES):
        side_sign = random.choice([-1, 1])
        noise_target = random.randint(lo, hi)
        candidates = _noise_candidates(clean_mask, direction, true_k, side_sign)
        if len(candidates) < noise_target:
            continue

        if bool(spec.get("force_clue", False)):
            chosen = _sample_clue_noise(
                candidates,
                noise_target,
                str(spec["cross_axis_dir"]),
                int(spec["cross_k"]),
            )
        else:
            chosen = _sample_general_noise(candidates, noise_target)
        if chosen is None:
            continue

        coords = clean_coords + chosen
        if not _validate_noisy_object(coords, spec, clean_count):
            continue

        for r, c in chosen:
            inp[r, c] = color
        return True
    return False


def _gen_attempt() -> np.ndarray | None:
    """Build one candidate input and reject it unless the combo behavior is present."""
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)
    roles = _sample_roles()

    sprite_specs: list[dict[str, object]] = []
    try:
        for color, slot, role in zip(obj_colors, slots, roles):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            direction = str(role["axis_dir"])
            if role["mode"] == "ambiguous":
                sprite, axes = _build_bisymmetric_sprite(color, max_h, max_w)
                sprite_specs.append(
                    {
                        "color": color,
                        "sprite": sprite,
                        "mode": "ambiguous",
                        "axis_dir": direction,
                        "axis_k_local": axes[direction],
                        "cross_axis_dir": "vertical",
                        "cross_axis_k_local": axes["vertical"],
                        "force_clue": True,
                    }
                )
            else:
                sprite, axis_k_local = _build_directional_sprite(color, max_h, max_w, direction)
                sprite_specs.append(
                    {
                        "color": color,
                        "sprite": sprite,
                        "mode": "directional",
                        "axis_dir": direction,
                        "axis_k_local": axis_k_local,
                        "force_clue": False,
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
