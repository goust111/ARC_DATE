# Combo variant of 8e5c0c38 (index 31)
# combo_spec: D9/1+D10/1
# Name: 大噪声小间隙修复

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import colors_present, get_bg, paste


"""
Design analysis for combo variant D9/1 + D10/1

a) Concretization + trace
- Original task: each color forms one near-vertical-symmetric sprite. Noise is modest, so
  the best axis is clearly better than the runner-up, and `main()` can simply pick the
  highest-scoring vertical axis and delete unmatched pixels.
- Fused variant keeps the same input->output rule, but the generator now makes that search
  numerically delicate:
  1. Start from a clean symmetric sprite.
  2. Add many noise pixels: 40-60% of the clean count.
  3. Some early noise is chosen to support a false axis almost as well as the true axis.
  4. Remaining noise is chosen to stay unpaired under the true axis without letting any
     false axis overtake the true one.
- Small concrete example on one row:
  clean row under true axis contains `..x..|..x..`.
  Original generator might add one stray `x` near one flank, so the true axis still wins
  by several pair counts.
  Here we may add many flank pixels such as `..xx.|..x.x`; several of them accidentally
  line up under a neighboring half-column axis. The true axis still wins, but only by 1.
  `main()` must therefore compute scores exactly; a heuristic like "pick the bbox center"
  would fail.

b) Complexity calibration
- The variant would be too weak if only the thresholds changed numerically while typical
  samples still looked almost clean. To prevent that, the generator forces both:
  noise ratio in [0.40, 0.60], and final best-axis gap in [1, 2], usually 1.
- It would be too chaotic if every candidate cell could be noise. The code keeps the
  original human-readable placement rule: noise stays 4-neighbor adjacent to the clean
  sprite and never has a true-axis partner. So the visual pattern is still "corrupted
  symmetric sprite", not arbitrary confetti.
- The ambiguity space is kept small: only vertical axes are considered, and every color
  still owns exactly one sprite. This preserves inferability from a few ARC examples.

c) Implicit-assumption audit against the original main()
- `colors_present(..., exclude={bg})`: assumes one logical sprite per non-bg color.
  Still required; generator keeps distinct colors for the three sprites.
- `pts = np.argwhere(g == col)`: assumes all same-color pixels should be pooled before
  choosing an axis. This remains valid only because generator never splits one color into
  multiple independent sprites.
- `for k in range(...)`: assumes exhaustive score search is necessary and sufficient.
  Under D10/1 this is even more important, because near-ties are intentional.
- `if score > best_score`: ties are broken by first occurrence, so ties would be unsafe.
  Generator therefore guarantees unique best axis with gap >= 1 after noise.
- Deletion uses only mirror existence under the chosen axis. That is unchanged. The fused
  variant increases difficulty by corrupting the axis choice, not by changing deletion.

d) Placement legality
- Generation remains output-first: build clean sprites, place them in separated slots on a
  22x22 canvas, then add noise only onto background cells.
- Noise candidates are filtered against the global background mask, so noisy pixels cannot
  overwrite another object.
- Slots still keep the three sprites far apart. Since noise extends only one cell away
  from the clean body, objects stay disjoint and in-bounds.

e) Visual observability
- A human sees the same kind of "fix each color independently" task as the original, but
  with visibly heavier corruption: many extra flank pixels cling to each sprite.
- The paired examples still teach the rule because the output is always the exact vertical
  mirror-cleaned version of the same shapes. The new difficulty is obvious from the amount
  of junk pixels and from some inputs looking almost symmetric around two nearby axes.

f) Visual cleanliness
- Distinct colors, separated slots, and adjacency-only noise keep each sprite legible.
- The generator avoids near-solid rectangles because those would look muddy under 60%
  noise and would not leave enough legal boundary cells for observable corruption.

g) Change inferability
- No invisible factor such as numeric color order is used. Color only separates sprites.
- The two fused changes are both directly visible:
  high noise is visible as many extra pixels,
  small axis gap is visible because several rows appear almost equally explainable by a
  neighboring axis, yet only one cleaned output is globally consistent.

New solver assumptions introduced here
- Each non-background color appears in exactly one sprite.
- The best vertical axis for each color is unique, though it may beat the runner-up by
  only 1. `generate_input()` explicitly enforces that.
- Noise pixels occupy background cells adjacent to the clean sprite and never form a true-
  mirror pair with the clean sprite or with another chosen noise pixel.
"""


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
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


def _best_axis(coords: list[tuple[int, int]], width: int) -> tuple[int, int, int]:
    scores = _axis_scores(coords, width)
    best_k = 0
    best_score = -1
    second = -1
    for k in range(2 * (width - 1) + 1):
        score = int(scores[int(k)])
        if score > best_score:
            second = best_score
            best_score = score
            best_k = int(k)
        elif score > second:
            second = score
    return int(best_k), int(best_score), int(second)


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    scores = _axis_scores(coords, width)
    return max(int(s) for k, s in scores.items() if int(k) != int(true_k))


def _axis_gap(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    return len(coords) - _second_best_score(coords, true_k, width)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore every colored sprite by selecting its strongest vertical mirror axis.

    Input structure:
    - The background fills most of the 22x22 grid.
    - Three distinct non-background colors each form one heavily corrupted near-symmetric
      sprite; extra pixels may be numerous, but every color still belongs to one object.

    Rule:
    - For each color separately, score every possible vertical axis (integer and half-
      integer column positions) by how many same-color mirror matches it explains.
    - Keep only the pixels that have a same-color mirror counterpart across that color's
      best-scoring axis; delete the rest to background.

    Key insight:
    - The task is still per-color symmetry denoising, but the noise is intentionally large
      enough that nearby false axes can score almost as well. Exact axis scoring matters.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _H, W = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        pts = [tuple(x) for x in np.argwhere(g == int(col))]
        if not pts:
            continue

        best_k, _best_score, _second = _best_axis(pts, W)
        for r, c in pts:
            cc = int(best_k) - int(c)
            if not (0 <= cc < W and int(g[int(r), int(cc)]) == int(col)):
                out[int(r), int(c)] = int(bg)

    return out


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
DENSITY_HI = 0.72

NOISE_FRAC_LO = 0.40
NOISE_FRAC_HI = 0.60

SPRITE_AXIS_GAP_MIN = 3
SPRITE_AXIS_GAP_MAX = 6
FINAL_AXIS_GAP_MIN = 1
FINAL_AXIS_GAP_MAX = 2

SPRITE_RETRIES = 200
NOISE_RETRIES = 120
GEN_RETRIES = 600
FLIP_PROB = 0.5


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
    weights = []
    for h, w, target_count, _center_count in options:
        area = h * w
        weights.append(1.0 / (target_count * area))
    return random.choices(options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    anchor_row = random.randrange(h)
    for r in range(h):
        if r == 0:
            c = 0
        elif side_w > 1 and r == anchor_row:
            c = side_w - 1
        else:
            c = random.randrange(side_w)
        cells.add((int(r), int(c)))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small")

    pool = [(r, c) for r in range(h) for c in range(side_w) if (r, c) not in cells]
    if remaining > len(pool):
        raise ValueError("left_count too large")
    cells.update(random.sample(pool, remaining))
    return cells


def _center_rows(h: int, center_count: int) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        rows[random.sample(range(h), center_count)] = True
    return rows


def _noise_candidates(clean_mask: np.ndarray, true_k: int, free_mask: np.ndarray) -> list[tuple[int, int]]:
    H, W = clean_mask.shape
    adj = np.zeros_like(clean_mask, dtype=bool)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~clean_mask
    adj &= free_mask

    cols = np.arange(W)
    mirror_cols = int(true_k) - cols
    on_axis = mirror_cols == cols
    adj[:, on_axis] = False

    valid_mirror = (mirror_cols >= 0) & (mirror_cols < W) & ~on_axis
    for c in np.where(valid_mirror)[0]:
        adj[:, c] &= ~clean_mask[:, int(mirror_cols[c])]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mc = w - 1 - c
            sprite[int(r), int(c)] = int(color)
            sprite[int(r), int(mc)] = int(color)
        if center_count > 0:
            sprite[centers, side_w] = int(color)

        coords = [tuple(x) for x in np.argwhere(sprite == int(color))]
        true_k = w - 1
        if len(coords) != target_count:
            continue

        gap = _axis_gap(coords, true_k, w)
        if not (SPRITE_AXIS_GAP_MIN <= gap <= SPRITE_AXIS_GAP_MAX):
            continue

        clean_mask = sprite == int(color)
        free_mask = sprite == 0
        min_noise = math.ceil(NOISE_FRAC_LO * target_count)
        if len(_noise_candidates(clean_mask, true_k, free_mask)) < min_noise:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict[str, object]], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    placed: list[dict[str, object]] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        cur = dict(spec)
        sprite = np.array(cur["sprite"], dtype=int)
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)
        cur["r0"] = int(r0)
        cur["c0"] = int(c0)
        cur["true_k"] = int(cur["axis_k_local"]) + 2 * int(c0)
        placed.append(cur)
    return out, placed


def _choose_noise_cells(
    clean_coords: list[tuple[int, int]],
    candidates: list[tuple[int, int]],
    true_k: int,
    width: int,
    noise_target: int,
) -> list[tuple[int, int]] | None:
    clean_count = len(clean_coords)
    best_choice: list[tuple[int, int]] | None = None
    best_key: tuple[int, int, int] | None = None

    for _ in range(NOISE_RETRIES):
        chosen: list[tuple[int, int]] = []
        forbidden: set[tuple[int, int]] = set()
        current = list(clean_coords)
        current_gap = clean_count - _second_best_score(current, true_k, width)

        while len(chosen) < noise_target and current_gap > FINAL_AXIS_GAP_MAX:
            viable = [cell for cell in candidates if cell not in chosen and cell not in forbidden]
            scored = []
            for cell in viable:
                alt = _second_best_score(current + [cell], true_k, width)
                if alt < clean_count:
                    scored.append((int(alt), random.random(), cell))
            if not scored:
                break
            max_alt = max(alt for alt, _rand, _cell in scored)
            pool = [cell for alt, _rand, cell in scored if alt >= max_alt - 1]
            cell = random.choice(pool)
            chosen.append(cell)
            current.append(cell)
            r, c = cell
            mirror = (int(r), int(true_k) - int(c))
            forbidden.add(cell)
            if 0 <= mirror[1] < width:
                forbidden.add(mirror)
            current_gap = clean_count - _second_best_score(current, true_k, width)

        if current_gap < FINAL_AXIS_GAP_MIN:
            continue

        while len(chosen) < noise_target:
            viable = [cell for cell in candidates if cell not in chosen and cell not in forbidden]
            scored = []
            for cell in viable:
                alt = _second_best_score(current + [cell], true_k, width)
                if alt < clean_count:
                    scored.append((int(alt), random.random(), cell))
            if not scored:
                break
            min_alt = min(alt for alt, _rand, _cell in scored)
            pool = [cell for alt, _rand, cell in scored if alt <= min_alt + 1]
            cell = random.choice(pool)
            chosen.append(cell)
            current.append(cell)
            r, c = cell
            mirror = (int(r), int(true_k) - int(c))
            forbidden.add(cell)
            if 0 <= mirror[1] < width:
                forbidden.add(mirror)

        if len(chosen) != noise_target:
            continue

        final_gap = clean_count - _second_best_score(current, true_k, width)
        if not (FINAL_AXIS_GAP_MIN <= final_gap <= FINAL_AXIS_GAP_MAX):
            continue

        key = (0 if final_gap == 1 else 1, int(final_gap), -len(chosen))
        if best_key is None or key < best_key:
            best_key = key
            best_choice = list(chosen)
            if final_gap == 1:
                break

    return best_choice


def _add_noise(inp: np.ndarray, out: np.ndarray, spec: dict[str, object], bg: int) -> bool:
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = [tuple(x) for x in np.argwhere(out == color)]
    clean_count = len(clean_coords)
    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = math.floor(NOISE_FRAC_HI * clean_count)
    if lo > hi:
        return False

    clean_mask = out == color
    free_mask = inp == int(bg)
    candidates = _noise_candidates(clean_mask, true_k, free_mask)
    if len(candidates) < lo:
        return False

    hi = min(hi, len(candidates))
    high_band = list(range(max(lo, math.ceil(0.5 * clean_count)), hi + 1))
    low_band = list(range(lo, max(lo, math.ceil(0.5 * clean_count))))
    random.shuffle(high_band)
    random.shuffle(low_band)
    targets = high_band + low_band

    for noise_target in targets:
        chosen = _choose_noise_cells(clean_coords, candidates, true_k, GRID_SIZE, noise_target)
        if chosen is None:
            continue
        for r, c in chosen:
            inp[int(r), int(c)] = int(color)
        return True
    return False


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)

    sprite_specs: list[dict[str, object]] = []
    try:
        for color, slot in zip(obj_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append({"color": int(color), "sprite": sprite, "axis_k_local": int(axis_k_local)})
    except ValueError:
        return None

    placed = _slot_place_sprites(bg, sprite_specs, slots)
    if placed is None:
        return None
    out, placed_specs = placed
    inp = out.copy()

    for spec in placed_specs:
        if not _add_noise(inp, out, spec, bg):
            return None

    for color in obj_colors:
        coords_in = [tuple(x) for x in np.argwhere(inp == int(color))]
        coords_out = [tuple(x) for x in np.argwhere(out == int(color))]
        true_count = len(coords_out)
        noise_count = len(coords_in) - true_count
        if not (math.ceil(NOISE_FRAC_LO * true_count) <= noise_count <= math.floor(NOISE_FRAC_HI * true_count)):
            return None
        best_k, best_score, second = _best_axis(coords_in, GRID_SIZE)
        if best_score != true_count:
            return None
        if best_score - second not in (1, 2):
            return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            inp, _out = result
            return inp
    raise RuntimeError(f"failed to generate valid combo input after {GEN_RETRIES} attempts")
