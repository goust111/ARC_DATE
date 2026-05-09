# Variant of 8e5c0c38
# Dimension: D12 — 颜色系统与分组规则 (meta)
# Change: 1种背景色+3种不同前景色，每种颜色恰好一个精灵... → 两个精灵共享同一颜色

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import numpy as np

ROOT = Path("/home/lixinyu/snap/new")
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from arc_utils import colors_present, find_components, get_bg, paste


# Design analysis for the D12 variant
# a) Concrete behavior trace
#    - Original task: each non-bg color corresponds to exactly one noisy vertically-symmetric sprite.
#      Solver can therefore treat "all pixels of one color" as one object, search one best vertical axis,
#      and delete unmatched pixels.
#    - Variant: there are still exactly 3 sprites, but only 2 foreground colors. One foreground color is
#      reused by two spatially separate sprites. Input therefore contains one color layer with two disjoint
#      connected components, each with its own local symmetry axis and its own attached noise pixels.
#      Output keeps both same-color sprites, cleaned independently.
#    - Small trace:
#      old logic on shared color S = union(A, B) -> search one axis for all S pixels together -> pixels in A
#      can be judged against B's geometry or against a compromise axis, so valid pixels may be deleted.
#      new logic on shared color S -> split S into 4-connected components A and B -> for A find axis_A and
#      delete only A's unmatched pixels; separately for B find axis_B and delete only B's unmatched pixels.
# b) Complexity calibration
#    - The variant stays close to the source task: still vertical symmetry denoising, still delete-only.
#    - The visible change is real because one sample color now appears in two distant regions. That forces a
#      solver to use color+space grouping rather than color-only grouping.
#    - Freedom is kept limited: exactly 3 sprites, exactly 2 foreground colors, exactly one duplicated color.
# c) Implicit-assumption audit of original main()
#    - `for col in colors`: still valid; colors remain a useful outer partition.
#    - `pts = np.argwhere(g == col)`: no longer valid as the object pixel set, because same-color pixels may
#      belong to two sprites. This is the key broken assumption.
#    - Axis scoring with `g[r, cc] == col`: also breaks, because a mirror match must be searched inside the
#      same connected sprite, not anywhere else in the color layer.
#    - Deleting unmatched pixels into `out`: still valid, since the task remains delete-only and components
#      are spatially separated in the generator.
# d) Placement legality
#    - Generator keeps the original three distant slot regions, so sprite bboxes do not overlap and the two
#      same-color sprites never touch by 4-connectivity.
#    - Noise is added only onto currently background cells adjacent to the owning clean sprite, and never on a
#      mirrored counterpart. This keeps each noisy sprite a single connected component and avoids accidental
#      bridges between sprites.
# e) Visual observability
#    - In the training pairs, one color now appears in two separated places. Both places clean up by the same
#      local "remove unmatched mirror noise" rule, so a human can infer "split same-color blobs first" from
#      just a few examples.
#    - The duplicated color is assigned to varying slots so the rule is not tied to one fixed region.
# f) Visual cleanliness
#    - Only one layer is reused, and reused sprites are far apart. There is no stacking or mixed-color overlap.
#    - Each sprite is forced to be one clean 4-connected component, which keeps the component-level grouping
#      visually readable after the D12 change.
# g) Color-change audit
#    - No color remapping is introduced. The only color-system change is cardinality/grouping: 2 foreground
#      colors encode 3 sprites because one color is intentionally duplicated.


def _best_axis_for_component(coords: np.ndarray, width: int) -> int:
    """Return the vertical axis k (where axis column is k/2) with the best self-mirror score."""
    pts = {(int(r), int(c)) for r, c in coords}
    best_score = -1
    best_k = 0
    for k in range(0, 2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each monochromatic connected sprite by restoring its own vertical mirror symmetry.

    Input structure:
    - One background color fills most cells. There are exactly three noisy sprites, but only two foreground
      colors because one color is reused by two disconnected sprites.
    - Each sprite is almost vertically symmetric around its own axis, with a few extra same-color pixels
      attached as local noise.

    Rule:
    - For each non-background color, first split its pixels into separate 4-connected components.
    - For each component independently, find the vertical axis that best matches that component's pixels and
      erase any pixel whose mirror counterpart is absent within that same component.

    Key insight:
    - Color alone is no longer enough to identify one sprite. The repeated color must be partitioned by spatial
      connectivity before applying the original symmetry-based denoising step component by component.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        comps = find_components(
            g,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        for comp in comps:
            coords = np.array(comp["coords"], dtype=int)
            if coords.size == 0:
                continue

            k = _best_axis_for_component(coords, w)
            pts = {(int(r), int(c)) for r, c in coords}
            for r, c in pts:
                cc = k - c
                if not (0 <= cc < w and (r, cc) in pts):
                    out[int(r), int(c)] = int(bg)

    return out


# --------------- generator (D12 variant) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
NUM_SPRITES = 3
NUM_FOREGROUND_COLORS = 2
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
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 240
GEN_RETRIES = 400

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
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


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    """Return the highest axis score excluding the true axis."""
    scores = _axis_scores(coords, width=width)
    return max(s for k, s in scores.items() if k != true_k)


def _feasible_shape_options(max_h: int, max_w: int) -> list[tuple[int, int, int, int]]:
    """Enumerate all (h, w, target_count, center_count) tuples satisfying the base constraints."""
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


def _left_cells(h: int, side_w: int, left_count: int) -> tuple[set[tuple[int, int]], int]:
    """Choose a connected left half-shape and a seam row that can bridge the mirrored halves."""
    max_extra = left_count - h
    min_backbone = max(0, side_w - 1 - max_extra)
    backbone_col = random.randint(min_backbone, side_w - 1)
    seam_row = random.randrange(h)

    cells: set[tuple[int, int]] = {(r, backbone_col) for r in range(h)}
    for c in range(backbone_col, side_w):
        cells.add((seam_row, c))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count too small for connected scaffold")

    while remaining > 0:
        frontier: list[tuple[int, int]] = []
        seen: set[tuple[int, int]] = set()
        for r, c in cells:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                cand = (rr, cc)
                if 0 <= rr < h and 0 <= cc < side_w and cand not in cells and cand not in seen:
                    frontier.append(cand)
                    seen.add(cand)
        if not frontier:
            raise ValueError("no frontier while growing connected left half")
        cell = random.choice(frontier)
        cells.add(cell)
        remaining -= 1

    return cells, seam_row


def _center_rows(h: int, center_count: int, bridge_row: int) -> np.ndarray:
    """Return boolean mask of rows that contain a center-column pixel."""
    rows = np.zeros(h, dtype=bool)
    if center_count <= 0:
        return rows
    rows[bridge_row] = True
    if center_count > 1:
        other_rows = [r for r in range(h) if r != bridge_row]
        rows[random.sample(other_rows, center_count - 1)] = True
    return rows


def _is_single_component(sprite: np.ndarray, color: int) -> bool:
    comps = find_components(
        sprite,
        bg=0,
        connectivity=4,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    return len(comps) == 1


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    """Build a single 4-connected vertically-symmetric sprite and return (sprite, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left, seam_row = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count, seam_row)

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
        if not _is_single_component(sprite, color):
            continue
        if _second_best_score(coords, true_k, width=w) > target_count - SPRITE_AXIS_GAP_MIN:
            continue
        return sprite, true_k
    raise ValueError("failed to build clean sprite")


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

        clean_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        clean_mask[r0 : r0 + h, c0 : c0 + w] = sprite != 0
        spec["r0"] = r0
        spec["c0"] = c0
        spec["true_k"] = spec["axis_k_local"] + 2 * c0
        spec["clean_mask"] = clean_mask
        spec["clean_coords"] = [tuple(x) for x in np.argwhere(clean_mask)]
        placed.append(spec)
    return out, placed


def _noise_candidates(clean_mask: np.ndarray, occupied_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Find all globally empty cells adjacent to one clean sprite and invalid under its true mirror axis."""
    h, w = clean_mask.shape
    adj = np.zeros_like(clean_mask)
    adj[1:, :] |= clean_mask[:-1, :]
    adj[:-1, :] |= clean_mask[1:, :]
    adj[:, 1:] |= clean_mask[:, :-1]
    adj[:, :-1] |= clean_mask[:, 1:]
    adj &= ~occupied_mask

    cols = np.arange(w)
    mirror_cols = true_k - cols
    on_axis = mirror_cols == cols
    adj[:, on_axis] = False

    valid_mirror = (mirror_cols >= 0) & (mirror_cols < w) & ~on_axis
    for c in np.where(valid_mirror)[0]:
        adj[:, c] &= ~occupied_mask[:, mirror_cols[c]]

    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_noise(inp: np.ndarray, bg: int, spec: dict) -> bool:
    """Add noise pixels to one placed sprite in inp. Returns False if constraints can't be met."""
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = list(spec["clean_coords"])
    clean_mask = np.array(spec["clean_mask"], dtype=bool)
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(NOISE_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return False
    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, inp != bg, true_k)
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
            alt = _second_best_score(current_coords + [cell], true_k, width=GRID_SIZE)
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

    if clean_count - _second_best_score(current_coords, true_k, width=GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _color_assignment() -> list[int]:
    """Return a shuffled 3-sprite color assignment with exactly one duplicated foreground color."""
    bg, shared_color, unique_color = random.sample(range(10), NUM_FOREGROUND_COLORS + 1)
    sprite_colors = [shared_color, shared_color, unique_color]
    random.shuffle(sprite_colors)
    return [bg, *sprite_colors]


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    """Single attempt: build clean sprites, place them, add noise, validate via solver."""
    bg, *sprite_colors = _color_assignment()
    slots = random.choice(LAYOUTS)

    sprite_specs = []
    try:
        for color, slot in zip(sprite_colors, slots):
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
        if not _add_noise(inp, bg, spec):
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
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
