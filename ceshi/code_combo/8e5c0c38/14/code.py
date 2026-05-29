# Combo variant of 8e5c0c38 (index 14)
# combo_spec: D3/4+D11/1
# Name: 重叠精灵优先级修复

from __future__ import annotations

import math
import random

import numpy as np

from arc_utils import colors_present, get_bg

# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior:
#    Original task: each color is denoised independently; a pixel survives only if the
#    mirrored cell already contains the same color. In this combo, sprites partially
#    overlap and priority is visible from left to right. Example: a blue sprite on the
#    left overlaps a red sprite on the right, so a red pixel may have its mirrored
#    partner hidden under a confirmed blue pixel. The original logic would delete that
#    red pixel as asymmetric; the combo logic must keep it because the higher-priority
#    blue pixel proves that the missing red partner is merely occluded, not noise.
# b) Complexity calibration:
#    The variant keeps exactly three single-color sprites and one global priority rule:
#    smaller axis-x means higher priority. That is strong enough to create real overlap
#    interactions, but still simple enough that 2-4 examples can support the rule.
# c) Broken original assumptions:
#    - "Each color can be solved independently" no longer holds.
#    - "A missing same-color mirror means noise" no longer holds because the mirror can
#      be owned by an already-fixed higher-priority sprite.
#    - "Object layout is disjoint" no longer holds in the latent clean scene.
#    The new solver therefore fixes sprites sequentially and lets earlier results serve
#    as evidence for later colors.
# d) Placement legality:
#    Generator constructs full clean sprites first, places them fully in-bounds, then
#    renders the visible output by layering low priority to high priority. Adjacent
#    sprites overlap by a small horizontal amount only, so the scene stays readable.
# e) Visual observability:
#    Priority is not tied to color ids. It is the left-to-right order of the sprite
#    axes, which a human can see directly. The visible difference from the original
#    task is that some lower-priority one-sided limbs remain in the output because their
#    missing counterparts are covered by a sprite farther left.
# f) Visual cleanliness:
#    Overlap depth is capped at two layers and only neighboring sprites overlap. Noise
#    is added only on currently visible boundary cells and never inside overlap-owned
#    cells, so the pictures remain legible.
# g) Inferability:
#    There is no recoloring. All structure comes from symmetry, overlap, and the stable
#    left-to-right processing order. The generator rejects samples where the overlap has
#    no effect and the old independent-color solver would produce the same output.


GRID_SIZE = 22
NUM_OBJECTS = 4

PIXEL_COUNT_MIN = 8
PIXEL_COUNT_MAX = 34
HEIGHT_MIN = 3
HEIGHT_MAX = 8
ODD_SIDE_W_RANGE = range(1, 5)
EVEN_SIDE_W_RANGE = range(2, 5)
DENSITY_LO = 0.38
DENSITY_HI = 0.84

NOISE_FRAC_LO = 0.10
NOISE_FRAC_HI = 0.24
NOISE_MAX = 6

SPRITE_RETRIES = 120
GEN_RETRIES = 500
FLIP_PROB = 0.5

OVERLAP_MIN = 1
OVERLAP_MAX = 3
ROW_SHIFT_MAX = 2


def _axis_scores(coords: list[tuple[int, int]], width: int = GRID_SIZE) -> dict[int, int]:
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


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int = GRID_SIZE) -> int:
    scores = _axis_scores(coords, width=width)
    return max(s for k, s in scores.items() if k != true_k)


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
    weights = [1.0 / (h * w) for h, w, _, _ in options]
    return random.choices(options, weights=weights, k=1)[0]


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
        if _second_best_score(coords, true_k, width=w) > target_count - 3:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


def _local_mask(sprite: np.ndarray, color: int) -> np.ndarray:
    return sprite == int(color)


def _bbox_center(mask: np.ndarray) -> float:
    pts = np.argwhere(mask)
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return (c0 + c1) / 2.0


def _visible_center_of_color(grid: np.ndarray, color: int) -> float:
    pts = np.argwhere(grid == int(color))
    c0 = int(pts[:, 1].min())
    c1 = int(pts[:, 1].max())
    return (c0 + c1) / 2.0


def _prelim_order(colors: list[int], grid: np.ndarray) -> list[int]:
    return sorted(colors, key=lambda col: (_visible_center_of_color(grid, int(col)), int(col)))


def _best_axis_with_support(coords: np.ndarray, grid: np.ndarray, fixed_mask: np.ndarray, color: int) -> int:
    _, W = grid.shape
    best_score = -10**9
    best_k = 0
    for k in range(0, 2 * (W - 1) + 1):
        score = 0
        for r, c in coords:
            cc = int(k) - int(c)
            if not (0 <= cc < W):
                score -= 2
                continue
            if int(grid[int(r), cc]) == int(color):
                score += 3 if cc != int(c) else 2
            elif bool(fixed_mask[int(r), cc]):
                score += 1
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def _original_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _, W = g.shape
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
            cc = int(best_k) - int(c)
            if not (0 <= cc < W and int(g[int(r), cc]) == int(col)):
                out[int(r), int(c)] = int(bg)
    return out


def _overlap_support_count(grid: np.ndarray) -> int:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    order = _prelim_order(colors, g)
    fixed_mask = np.zeros_like(g, dtype=bool)
    total = 0
    for col in order:
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        k = _best_axis_with_support(pts, g, fixed_mask, int(col))
        for r, c in pts:
            cc = int(k) - int(c)
            if 0 <= cc < g.shape[1] and bool(fixed_mask[int(r), cc]):
                total += 1
        fixed_mask |= (g == int(col))
    return int(total)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair vertically symmetric sprites when some mirrored partners are hidden by higher-priority overlaps.

    Input structure:
    - One background color plus exactly three non-background sprite colors.
    - Each color belongs to one nearly vertically symmetric sprite, but neighboring sprites can overlap.
    - The sprite whose axis is farther left has higher priority and owns overlap cells; extra same-color noise pixels may also appear.

    Rule:
    - Process colors from leftmost sprite axis to rightmost.
    - For the current color, find the vertical axis best supported by its visible pixels together with already-confirmed higher-priority pixels.
    - Delete a visible pixel only if its mirror is neither the same color nor an already-fixed higher-priority pixel.

    Key insight: A missing same-color mirror is not always noise here; it can be a true mirrored pixel hidden under an earlier sprite, so overlap ownership must be resolved before lower-priority denoising.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    out = g.copy()
    fixed_mask = np.zeros_like(g, dtype=bool)
    order = _prelim_order(colors, g)

    for col in order:
        pts = np.argwhere(out == int(col))
        if pts.size == 0:
            continue

        k = _best_axis_with_support(pts, out, fixed_mask, int(col))
        for r, c in pts:
            cc = int(k) - int(c)
            same_color = 0 <= cc < out.shape[1] and int(out[int(r), cc]) == int(col)
            covered_by_higher = 0 <= cc < out.shape[1] and bool(fixed_mask[int(r), cc])
            if not (same_color or covered_by_higher):
                out[int(r), int(c)] = int(bg)

        fixed_mask |= (out == int(col))

    return out


def _render_from_specs(bg: int, specs: list[dict]) -> tuple[np.ndarray, dict[int, np.ndarray], dict[int, int]]:
    full_masks: dict[int, np.ndarray] = {}
    axis_by_color: dict[int, int] = {}
    for spec in specs:
        color = int(spec["color"])
        sprite = spec["sprite"]
        r0 = int(spec["r0"])
        c0 = int(spec["c0"])
        h, w = sprite.shape
        mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        local = _local_mask(sprite, color)
        mask[r0 : r0 + h, c0 : c0 + w] = local
        full_masks[color] = mask
        axis_by_color[color] = int(spec["axis_k_local"] + 2 * c0)

    out = np.full((GRID_SIZE, GRID_SIZE), int(bg), dtype=int)
    for spec in reversed(specs):
        color = int(spec["color"])
        out[full_masks[color]] = color
    return out, full_masks, axis_by_color


def _choose_positions(specs: list[dict]) -> bool:
    widths = [int(spec["sprite"].shape[1]) for spec in specs]
    heights = [int(spec["sprite"].shape[0]) for spec in specs]

    max_ab = min(OVERLAP_MAX, widths[0] - 2, widths[1] - 2)
    max_bc = min(OVERLAP_MAX, widths[1] - 2, widths[2] - 2)
    if max_ab < OVERLAP_MIN or max_bc < OVERLAP_MIN:
        return False
    overlap_ab = random.randint(OVERLAP_MIN, max_ab)
    overlap_bc = random.randint(OVERLAP_MIN, max_bc)

    total_w = widths[0] + widths[1] + widths[2] - overlap_ab - overlap_bc
    if total_w > GRID_SIZE - 2:
        return False

    c0_a = random.randint(0, GRID_SIZE - total_w)
    c0_b = c0_a + widths[0] - overlap_ab
    c0_c = c0_b + widths[1] - overlap_bc
    if c0_c + widths[2] > GRID_SIZE:
        return False

    r0_a = random.randint(1, GRID_SIZE - heights[0] - 1)

    def _pick_neighbor_row(prev_r0: int, prev_h: int, cur_h: int) -> int | None:
        lo = max(0, prev_r0 - ROW_SHIFT_MAX)
        hi = min(GRID_SIZE - cur_h, prev_r0 + ROW_SHIFT_MAX)
        choices = []
        for cand in range(lo, hi + 1):
            top = max(prev_r0, cand)
            bot = min(prev_r0 + prev_h - 1, cand + cur_h - 1)
            if bot - top + 1 >= 2:
                choices.append(cand)
        if not choices:
            return None
        return random.choice(choices)

    r0_b = _pick_neighbor_row(r0_a, heights[0], heights[1])
    if r0_b is None:
        return False
    r0_c = _pick_neighbor_row(r0_b, heights[1], heights[2])
    if r0_c is None:
        return False

    for spec, r0, c0 in zip(specs, [r0_a, r0_b, r0_c], [c0_a, c0_b, c0_c]):
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
    return True


def _visible_mask_for_color(out: np.ndarray, color: int) -> np.ndarray:
    return out == int(color)


def _noise_candidates(out: np.ndarray, visible: np.ndarray, true_k: int, fixed_higher: np.ndarray) -> list[tuple[int, int]]:
    H, W = out.shape
    adj = np.zeros_like(visible)
    adj[1:, :] |= visible[:-1, :]
    adj[:-1, :] |= visible[1:, :]
    adj[:, 1:] |= visible[:, :-1]
    adj[:, :-1] |= visible[:, 1:]
    adj &= ~visible
    adj &= (out == int(get_bg(out)))

    rows, cols = np.where(adj)
    result: list[tuple[int, int]] = []
    for r, c in zip(rows.tolist(), cols.tolist()):
        cc = int(true_k) - int(c)
        if cc == int(c):
            continue
        if not (0 <= cc < W):
            result.append((int(r), int(c)))
            continue
        if bool(visible[int(r), cc]):
            continue
        if bool(fixed_higher[int(r), cc]):
            continue
        result.append((int(r), int(c)))
    return result


def _add_noise(inp: np.ndarray, out: np.ndarray, specs: list[dict], axis_by_color: dict[int, int]) -> bool:
    bg = int(get_bg(out))
    fixed_higher = np.zeros_like(out, dtype=bool)

    for spec in specs:
        color = int(spec["color"])
        visible = _visible_mask_for_color(out, color)
        visible_count = int(visible.sum())
        lo = max(1, math.ceil(NOISE_FRAC_LO * visible_count))
        hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * visible_count))
        if lo > hi:
            return False

        candidates = _noise_candidates(inp, visible, axis_by_color[color], fixed_higher)
        if len(candidates) < lo:
            return False
        target = random.randint(lo, min(hi, len(candidates)))
        random.shuffle(candidates)

        chosen: list[tuple[int, int]] = []
        forbidden: set[tuple[int, int]] = set()
        for cell in candidates:
            if cell in forbidden:
                continue
            r, c = cell
            cc = axis_by_color[color] - c
            chosen.append((r, c))
            forbidden.add((r, c))
            if 0 <= cc < GRID_SIZE:
                forbidden.add((r, cc))
            if len(chosen) >= target:
                break
        if len(chosen) < lo:
            return False

        for r, c in chosen:
            inp[int(r), int(c)] = int(color)

        fixed_higher |= visible

    return int((inp != out).sum()) > 0


def _occlusion_kept_count(out: np.ndarray, axis_by_color: dict[int, int], ordered_colors: list[int]) -> int:
    fixed_higher = np.zeros_like(out, dtype=bool)
    total = 0
    for color in ordered_colors:
        pts = np.argwhere(out == int(color))
        for r, c in pts:
            cc = axis_by_color[int(color)] - int(c)
            if 0 <= cc < GRID_SIZE and bool(fixed_higher[int(r), cc]):
                total += 1
        fixed_higher |= (out == int(color))
    return int(total)


def _validate_visibility(out: np.ndarray, axis_by_color: dict[int, int], ordered_colors: list[int]) -> bool:
    fixed_higher = np.zeros_like(out, dtype=bool)
    for color in ordered_colors:
        pts = np.argwhere(out == int(color))
        if pts.shape[0] < 5:
            return False
        support = 0
        for r, c in pts:
            cc = axis_by_color[int(color)] - int(c)
            if 0 <= cc < GRID_SIZE and (int(out[int(r), cc]) == int(color) or bool(fixed_higher[int(r), cc])):
                support += 1
        if support < max(4, pts.shape[0] // 3):
            return False
        fixed_higher |= (out == int(color))
    return True


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    ordered_colors = obj_colors[:]

    specs: list[dict] = []
    try:
        for color in ordered_colors:
            sprite, axis_k_local = _build_clean_sprite(int(color), max_h=9, max_w=9)
            specs.append({"color": int(color), "sprite": sprite, "axis_k_local": int(axis_k_local)})
    except ValueError:
        return None

    if not _choose_positions(specs):
        return None

    out, full_masks, axis_by_color = _render_from_specs(bg, specs)

    for color in ordered_colors:
        if abs(_bbox_center(full_masks[int(color)]) - _visible_center_of_color(out, int(color))) > 4.5:
            return None

    if not _validate_visibility(out, axis_by_color, ordered_colors):
        return None

    if _occlusion_kept_count(out, axis_by_color, ordered_colors) < 2:
        return None

    inp = out.copy()
    if not _add_noise(inp, out, specs, axis_by_color):
        return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if np.array_equal(_original_main(inp), out):
        return None
    if _overlap_support_count(out) < 2:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
