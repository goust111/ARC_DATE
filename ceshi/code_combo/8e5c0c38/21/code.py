# Combo variant of 8e5c0c38 (index 21)
# combo_spec: D9/1+D15/0
# Name: 图例辅助大噪声修复

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


# Design analysis for the combo variant
# -------------------------------------
# a) Concrete behavior trace
#    - Original task:
#      each foreground color is one interior sprite; the solver looks at all pixels of that
#      color, picks the vertical axis with the most mirror matches, and deletes unmatched
#      pixels.
#    - Fused task:
#      each foreground color now appears twice in the input:
#        1) one interior noisy sprite,
#        2) one disconnected 1-cell-thick border bar of the same color.
#      The bar length is an explicit legend: it equals the number of noisy pixels attached
#      to that color's interior sprite.
#    - Small trace:
#      suppose the red interior component has 19 pixels and the red border bar has length 7.
#      Then the clean red sprite must have exactly 12 pixels in the output.
#      Under heavy noise there may be two plausible vertical axes:
#        axis A keeps 14 self-mirrored pixels,
#        axis B keeps 12 self-mirrored pixels.
#      The old rule would choose axis A because 14 > 12.
#      The fused rule must choose axis B because the legend says exactly 7 pixels are noise,
#      so exactly 12 pixels must survive.
#    - Visible difference:
#      the border bar is not decoration; it changes which axis is valid when heavy noise
#      makes geometry alone ambiguous.
# b) Complexity calibration
#    - D9/1 is made visible by pushing every sprite to 40%-60% noise relative to its clean
#      area. This is much larger than the base task and often creates false mirror evidence.
#    - D15/0 is made visible by using only one simple legend vocabulary: per-color border
#      line length. That keeps the symbolic burden small enough for humans to read from
#      2-4 examples.
#    - Freedom is controlled:
#      exactly 3 interior sprites, exactly 3 legend bars, and each legend bar uses the same
#      color as its sprite. The only decoded value is the count.
# c) Implicit-assumption audit of the original main()
#    - `colors_present(... exclude={bg})` still works as the outer partition.
#    - `pts = np.argwhere(g == col)` is no longer the sprite pixel set:
#      it merges the interior sprite with the same-colored border legend.
#    - "pick the axis with the maximum mirror score" is no longer safe:
#      D9/1 intentionally makes some false axes score better than the true axis.
#    - "delete every pixel lacking a mirror under that best axis" is also insufficient:
#      even if several axes produce mirrored subsets, only the axis whose deletion count
#      matches the legend is admissible.
#    - New solver fix:
#      first separate legend component from scene component for each color, then search
#      axes using the exact legend count as a hard constraint.
# d) Placement legality
#    - Interior sprites are placed in three non-overlapping slots fully inside the canvas.
#    - Border legend bars occupy only edge cells and are stamped after scene generation, so
#      they cannot overlap interior sprites.
#    - Noise is added only on background cells inside the owning slot and is kept attached
#      by 4-neighbor growth, so each color still has exactly one interior component.
# e) Visual observability
#    - Humans can see three large noisy blobs plus three same-colored border bars.
#    - Across examples, the bars move under global flips, so their role is inferred from
#      "same color + bar length" rather than from a fixed side of the grid.
#    - The output preserves the legend bars and only cleans the interior blobs, making the
#      legend-to-cleaning relationship easy to compare visually.
# f) Visual cleanliness
#    - There are only two layers per color: one border line and one interior blob.
#      No extra markers, labels, frames, or recolors are added.
#    - Interior slots stay well separated from one another and from the edges, so the scene
#      remains readable even with large noise.
# g) Change inferability
#    - No hidden color ordering is used. The legend key is explicit because the border bar
#      shares the sprite's color.
#    - The numeric content of the legend is visible as run length, a human-observable cue.
# h) Anti-degeneration target
#    - The generator rejects any sample for which the legacy geometry-only cleaner already
#      matches the fused output.
#    - It also rejects any sample where all colors are solvable by geometry-only axis
#      maximization on the interior component. At least one color must require the legend's
#      exact deletion count to disambiguate the axis.
# i) New solver assumptions and generator guarantees
#    - Assumption: every foreground color yields exactly one border-line legend component.
#      Guarantee: generator stamps one disconnected straight edge bar per color.
#    - Assumption: every foreground color yields exactly one interior component.
#      Guarantee: clean sprites are connected and noise grows from the component frontier.
#    - Assumption: the true axis is the unique axis whose mirrored survivor count equals
#      `interior_pixels - legend_count`.
#      Guarantee: generator checks uniqueness with the same oracle used by the solver.


Coord = tuple[int, int]


def _is_border_legend(coords: list[Coord], height: int, width: int) -> bool:
    if not coords:
        return False
    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})
    if len(rows) == 1:
        row = rows[0]
        if row not in {0, height - 1}:
            return False
        return cols == list(range(cols[0], cols[-1] + 1))
    if len(cols) == 1:
        col = cols[0]
        if col not in {0, width - 1}:
            return False
        return rows == list(range(rows[0], rows[-1] + 1))
    return False


def _paired_coords(pts: set[Coord], k: int, width: int) -> set[Coord]:
    paired: set[Coord] = set()
    for r, c in pts:
        cc = int(k) - int(c)
        if 0 <= cc < width and (int(r), int(cc)) in pts:
            paired.add((int(r), int(c)))
    return paired


def _axis_paired_counts(pts: set[Coord], width: int) -> dict[int, int]:
    return {k: len(_paired_coords(pts, k, width)) for k in range(0, 2 * (width - 1) + 1)}


def _choose_axis_with_count(scene_coords: list[Coord], noise_count: int, width: int) -> tuple[int, set[Coord]]:
    pts = {(int(r), int(c)) for r, c in scene_coords}
    target_keep = len(pts) - int(noise_count)
    best_k = 0
    best_keep: set[Coord] = set()
    best_key: tuple[int, int, int] | None = None
    for k in range(0, 2 * (width - 1) + 1):
        keep = _paired_coords(pts, k, width)
        key = (abs(len(keep) - target_keep), -len(keep), int(k))
        if best_key is None or key < best_key:
            best_key = key
            best_k = int(k)
            best_keep = keep
    return best_k, best_keep


def _decode_color_parts(grid: np.ndarray, bg: int, color: int) -> tuple[list[Coord], list[Coord]]:
    height, width = grid.shape
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    legend: list[Coord] | None = None
    scene_parts: list[Coord] = []
    for comp in comps:
        coords = [tuple(map(int, rc)) for rc in np.array(comp["coords"], dtype=int)]
        if _is_border_legend(coords, height, width):
            if legend is not None:
                raise ValueError(f"multiple legend components for color {color}")
            legend = coords
        else:
            scene_parts.extend(coords)
    if legend is None or not scene_parts:
        raise ValueError(f"failed to decode legend+scene for color {color}")
    scene = sorted({(int(r), int(c)) for r, c in scene_parts})
    return legend, scene


def _legacy_main(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _height, width = g.shape
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        pts = np.argwhere(g == int(col))
        if pts.size == 0:
            continue
        best_score = -1
        best_k = 0
        for k in range(0, 2 * (width - 1) + 1):
            score = 0
            for r, c in pts:
                cc = int(k) - int(c)
                if 0 <= cc < width and int(g[int(r), int(cc)]) == int(col):
                    score += 1
            if score > best_score:
                best_score = int(score)
                best_k = int(k)
        for r, c in pts:
            cc = int(best_k) - int(c)
            if not (0 <= cc < width and int(g[int(r), int(cc)]) == int(col)):
                out[int(r), int(c)] = int(bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each interior sprite to perfect vertical symmetry using edge legend counts.

    Input structure:
    - One background color fills most cells. Each foreground color appears as one noisy
      interior sprite plus one disconnected border bar of the same color.
    - The border bar is a legend: its length gives the exact number of noisy pixels that
      must be removed from that color's interior sprite.
    - Noise is heavy, around half of the clean sprite size, so geometry alone can suggest
      the wrong vertical axis.

    Rule:
    - For each foreground color, separate the border legend from the interior component.
    - Search all vertical axes for the interior component and keep the mirrored subset whose
      survivor count matches `interior_pixels - legend_length`.
    - Preserve the legend bar unchanged and erase the other interior pixels to background.

    Key insight:
    - The task is no longer "pick the strongest symmetry axis". The edge bars provide exact
      numeric constraints, so the correct axis is the one that yields the legend-specified
      deletion count while leaving a perfectly mirrored interior shape.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _height, width = g.shape

    out = np.full_like(g, bg)
    colors = colors_present(g, exclude={bg})
    assert colors, "grid must contain at least one non-background color"

    for col in colors:
        legend, scene = _decode_color_parts(g, bg, int(col))
        noise_count = len(legend)
        _best_k, keep = _choose_axis_with_count(scene, noise_count, width)
        for r, c in legend:
            out[int(r), int(c)] = int(col)
        for r, c in keep:
            out[int(r), int(c)] = int(col)

    return out


# --------------- generator (combo D9/1 + D15/0) ---------------

GRID_SIZE = 22
NUM_OBJECTS = 4  # bg + 3 foreground colors
LAYOUTS = [
    [(2, 2, 9, 9), (2, 12, 9, 19), (12, 4, 19, 17)],
    [(2, 2, 10, 8), (2, 13, 10, 19), (12, 5, 19, 16)],
]

PIXEL_COUNT_MIN = 10
PIXEL_COUNT_MAX = 28
HEIGHT_MIN = 3
HEIGHT_MAX = 8
ODD_SIDE_W_RANGE = range(1, 5)
EVEN_SIDE_W_RANGE = range(2, 5)
DENSITY_LO = 0.40
DENSITY_HI = 0.88

NOISE_FRAC_LO = 0.40
NOISE_FRAC_HI = 0.60
LEGEND_MAX = 19

CLEAN_AXIS_GAP_MIN = 3
GEN_RETRIES = 700
SPRITE_RETRIES = 220
NOISE_PICK_RETRIES = 60
FLIP_PROB = 0.5

BASE_LEGEND_SLOTS = [
    ("h", 0, 2),
    ("v", 2, 0),
    ("v", 2, GRID_SIZE - 1),
]


def _second_best_score(coords: list[Coord], true_k: int, width: int) -> int:
    scores = _axis_paired_counts({(int(r), int(c)) for r, c in coords}, width)
    return max(score for k, score in scores.items() if k != int(true_k))


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
    counts = sorted({opt[2] for opt in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> tuple[set[Coord], int]:
    max_extra = left_count - h
    if max_extra < 0:
        raise ValueError("left_count too small")
    min_backbone = max(0, side_w - 1 - max_extra)
    backbone_col = random.randint(min_backbone, side_w - 1)
    seam_row = random.randrange(h)

    cells: set[Coord] = {(int(r), int(backbone_col)) for r in range(h)}
    for c in range(backbone_col, side_w):
        cells.add((int(seam_row), int(c)))

    remaining = left_count - len(cells)
    if remaining < 0:
        raise ValueError("left_count incompatible with scaffold")

    while remaining > 0:
        frontier: set[Coord] = set()
        for r, c in cells:
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < h and 0 <= cc < side_w and (int(rr), int(cc)) not in cells:
                    frontier.add((int(rr), int(cc)))
        if not frontier:
            raise ValueError("frontier exhausted while growing left half")
        cell = random.choice(sorted(frontier))
        cells.add(cell)
        remaining -= 1
    return cells, int(seam_row)


def _center_rows(h: int, center_count: int, required_row: int | None = None) -> np.ndarray:
    rows = np.zeros(h, dtype=bool)
    if center_count > 0:
        pool = list(range(h))
        chosen: set[int] = set()
        if required_row is not None:
            chosen.add(int(required_row))
            pool.remove(int(required_row))
        extra = int(center_count) - len(chosen)
        if extra < 0:
            raise ValueError("required center row exceeds center_count")
        chosen.update(random.sample(pool, extra))
        rows[list(chosen)] = True
    return rows


def _build_clean_sprite(color: int, max_h: int, max_w: int) -> tuple[np.ndarray, int]:
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(max_h, max_w)
        side_w = w // 2
        left_count = (target_count - center_count) // 2
        left, seam_row = _left_cells(h, side_w, left_count)
        centers = _center_rows(h, center_count, required_row=seam_row if center_count > 0 else None)

        sprite = np.zeros((h, w), dtype=int)
        for r, c in left:
            mc = w - 1 - c
            sprite[int(r), int(c)] = int(color)
            sprite[int(r), int(mc)] = int(color)
        if center_count > 0:
            sprite[centers, side_w] = int(color)

        coords = [tuple(map(int, rc)) for rc in np.argwhere(sprite == int(color))]
        true_k = w - 1
        if len(coords) != target_count:
            continue
        if _second_best_score(coords, true_k, w) > target_count - CLEAN_AXIS_GAP_MIN:
            continue
        return sprite, int(true_k)
    raise ValueError("failed to build clean sprite")


def _choose_origin(lo: int, hi: int, size: int, pad: int = 1) -> int:
    start_min = int(lo)
    start_max = int(hi) - int(size) + 1
    if start_min > start_max:
        raise ValueError("slot too small")
    padded_min = start_min + pad
    padded_max = start_max - pad
    if padded_min <= padded_max:
        return random.randint(padded_min, padded_max)
    return random.randint(start_min, start_max)


def _slot_place_sprites(
    bg: int, sprite_specs: list[dict], slots: list[tuple[int, int, int, int]]
) -> tuple[np.ndarray, list[dict]] | None:
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []
    for spec, slot in zip(sprite_specs, slots):
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        r0s, c0s, r1s, c1s = slot
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None
        r0 = _choose_origin(r0s, r1s, h)
        c0 = _choose_origin(c0s, c1s, w)
        paste(out, sprite, r0, c0, transparent=0)
        spec["slot"] = tuple(map(int, slot))
        spec["r0"] = int(r0)
        spec["c0"] = int(c0)
        spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        spec["clean_count"] = int(np.count_nonzero(sprite == int(spec["color"])))
        placed.append(spec)
    return out, placed


def _noise_frontier(
    occupied: set[Coord],
    clean: set[Coord],
    slot: tuple[int, int, int, int],
    true_k: int,
) -> list[Coord]:
    r0, c0, r1, c1 = map(int, slot)
    frontier: set[Coord] = set()
    for r, c in occupied:
        for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if not (r0 <= rr <= r1 and c0 <= cc <= c1):
                continue
            cand = (int(rr), int(cc))
            if cand in occupied:
                continue
            mc = int(true_k) - int(cc)
            if mc == cc:
                continue
            if 0 <= mc < GRID_SIZE and ((int(rr), int(mc)) in clean or (int(rr), int(mc)) in occupied):
                continue
            frontier.add(cand)
    return sorted(frontier)


def _best_other_score(coords: list[Coord], true_k: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    best = -1
    for k in range(0, 2 * (GRID_SIZE - 1) + 1):
        if k == int(true_k):
            continue
        best = max(best, len(_paired_coords(pts, k, GRID_SIZE)))
    return int(best)


def _exact_axes(coords: list[Coord], target_keep: int) -> list[int]:
    pts = {(int(r), int(c)) for r, c in coords}
    return [
        int(k)
        for k in range(0, 2 * (GRID_SIZE - 1) + 1)
        if len(_paired_coords(pts, k, GRID_SIZE)) == int(target_keep)
    ]


def _geom_best_axis(coords: list[Coord]) -> tuple[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    best_k = 0
    best_score = -1
    for k in range(0, 2 * (GRID_SIZE - 1) + 1):
        score = len(_paired_coords(pts, k, GRID_SIZE))
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return int(best_k), int(best_score)


def _add_noise(scene_inp: np.ndarray, spec: dict) -> dict | None:
    color = int(spec["color"])
    clean_coords = [tuple(map(int, rc)) for rc in np.argwhere(scene_inp == color)]
    clean = set(clean_coords)
    clean_count = int(len(clean_coords))
    true_k = int(spec["true_k"])
    slot = tuple(map(int, spec["slot"]))
    adversarial = bool(spec["adversarial"])

    lo = max(1, math.ceil(NOISE_FRAC_LO * clean_count))
    hi = min(LEGEND_MAX, math.floor(NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return None
    noise_target = random.randint(lo, hi)
    target_keep = clean_count

    chosen: list[Coord] = []
    occupied: set[Coord] = set(clean)

    for _ in range(noise_target):
        frontier = _noise_frontier(occupied, clean, slot, true_k)
        scored: list[tuple[int, float, Coord]] = []
        for cell in frontier:
            trial = clean_coords + chosen + [cell]
            if len(_paired_coords(set(trial), true_k, GRID_SIZE)) != clean_count:
                continue
            alt = _best_other_score(trial, true_k)
            scored.append((int(alt), random.random(), cell))
        if not scored:
            return None
        scored.sort(key=lambda item: (item[0], item[1]))
        if adversarial:
            best_alt = scored[-1][0]
            pool = [cell for alt, _, cell in scored if alt >= best_alt - 1]
        else:
            best_alt = scored[0][0]
            pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        occupied.add(cell)

    scene_coords = clean_coords + chosen
    exact = _exact_axes(scene_coords, target_keep)
    if exact != [true_k]:
        return None
    geom_k, geom_score = _geom_best_axis(scene_coords)

    for r, c in chosen:
        scene_inp[int(r), int(c)] = color

    return {
        "color": color,
        "noise_count": int(noise_target),
        "noise_ratio": float(noise_target / clean_count),
        "true_k": int(true_k),
        "geom_best_k": int(geom_k),
        "geom_best_score": int(geom_score),
        "legend_essential": bool(int(geom_k) != int(true_k)),
    }


def _stamp_legend(grid: np.ndarray, legend_entries: list[dict]) -> None:
    for entry in legend_entries:
        axis = str(entry["axis"])
        anchor_r = int(entry["anchor_r"])
        anchor_c = int(entry["anchor_c"])
        color = int(entry["color"])
        count = int(entry["count"])
        if axis == "h":
            for dc in range(count):
                grid[anchor_r, anchor_c + dc] = color
        else:
            for dr in range(count):
                grid[anchor_r + dr, anchor_c] = color


def _legend_entries(placed_specs: list[dict], noise_meta: list[dict]) -> list[dict]:
    color_to_noise = {int(meta["color"]): int(meta["noise_count"]) for meta in noise_meta}
    slots = random.sample(BASE_LEGEND_SLOTS, k=len(placed_specs))
    entries: list[dict] = []
    for spec, (axis, anchor_r, anchor_c) in zip(placed_specs, slots):
        count = color_to_noise[int(spec["color"])]
        entries.append(
            {
                "axis": str(axis),
                "anchor_r": int(anchor_r),
                "anchor_c": int(anchor_c),
                "color": int(spec["color"]),
                "count": int(count),
            }
        )
    return entries


def _generate_case() -> tuple[np.ndarray, np.ndarray, dict] | None:
    bg, *obj_colors = random.sample(range(10), NUM_OBJECTS)
    slots = random.choice(LAYOUTS)
    adversarial_index = random.randrange(len(slots))

    sprite_specs: list[dict] = []
    try:
        for idx, (color, slot) in enumerate(zip(obj_colors, slots)):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(color, max_h, max_w)
            sprite_specs.append(
                {
                    "color": int(color),
                    "sprite": sprite,
                    "axis_k_local": int(axis_k_local),
                    "adversarial": bool(idx == adversarial_index),
                }
            )
    except ValueError:
        return None

    placed_result = _slot_place_sprites(bg, sprite_specs, slots)
    if placed_result is None:
        return None
    scene_out, placed_specs = placed_result
    scene_inp = scene_out.copy()

    noise_meta: list[dict] = []
    for spec in placed_specs:
        meta = None
        for _ in range(NOISE_PICK_RETRIES):
            scene_trial = scene_inp.copy()
            meta = _add_noise(scene_trial, spec)
            if meta is not None:
                scene_inp = scene_trial
                break
        if meta is None:
            return None
        noise_meta.append(meta)

    if sum(int(meta["legend_essential"]) for meta in noise_meta) < 1:
        return None

    inp = scene_inp.copy()
    out = scene_out.copy()
    legend_entries = _legend_entries(placed_specs, noise_meta)
    _stamp_legend(inp, legend_entries)
    _stamp_legend(out, legend_entries)

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None
    if np.array_equal(_legacy_main(inp), out):
        return None

    meta = {
        "noise_ratios": [float(item["noise_ratio"]) for item in noise_meta],
        "legend_essential_colors": int(sum(int(item["legend_essential"]) for item in noise_meta)),
        "noise_counts": [int(item["noise_count"]) for item in noise_meta],
    }
    return inp, out, meta


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        case = _generate_case()
        if case is not None:
            return case[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
