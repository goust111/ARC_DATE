# Combo variant of 8e5c0c38 (index 23)
# combo_spec: D12/0+D15/0
# Name: 图例辅助共享颜色修复

from __future__ import annotations

import itertools
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
# a) Concrete behavior trace
#    - Base task: each foreground color names exactly one noisy near-symmetric sprite, so solver can gather
#      all pixels of one color, pick one vertical axis, and delete unmatched pixels.
#    - D12 already breaks that: one foreground color is reused by two separate sprites, so color is no longer
#      a complete object id. A solver must at least split same-color pixels into disconnected components.
#    - The fusion goes one step further: besides the true sprites, the scene may also contain tiny detached
#      same-color blobs. Those blobs are not real sprites; they are noise components that must disappear.
#      A fixed top legend states, for each foreground color, how many true sprites exist and how many total
#      pixels of that color must be removed.
#    - Small trace:
#      old logic on shared color S -> union(A, B, x) gets one axis and fails badly.
#      D12-only logic -> split into components A, B, x and clean each one independently; singleton x survives
#      because it is trivially symmetric about its own center column.
#      fused logic -> legend says color S has 2 sprites and 5 noise pixels total. Compute the cleaned keep-set
#      of every component, then choose exactly 2 components whose preserved pixels sum to total_pixels - 5.
#      The large components A and B are kept and denoised; tiny component x is discarded entirely.
# b) Complexity calibration
#    - Freedom is intentionally limited: still exactly 3 true sprites, exactly 2 foreground colors, and one
#      duplicated color. The legend also has a fixed readable format instead of a per-example codebook.
#    - The duplicated color always has sprite-count 2 in the legend, and the unique color has sprite-count 1.
#      This repeated low-cardinality evidence makes the new rule visible from a few examples.
#    - Detached noise blobs are tiny and sparse; they create a real need for the legend without flooding the
#      board with arbitrary clutter.
# c) Implicit-assumption audit of the original main()
#    - `pts = np.argwhere(g == col)` is invalid twice over now: the same color may span multiple real sprites,
#      and some same-color components may be disposable legend-announced noise blobs rather than sprites.
#    - Searching mirror partners with `g[r, cc] == col` over the whole color layer is also invalid. Mirror
#      checks must stay inside one candidate component.
#    - The original solver has no notion of symbolic pixels. Here the top 4 rows are a dedicated legend band
#      and must be preserved, not treated as scene objects.
#    - A new subset-selection step is required: after component-level cleaning scores are computed, the solver
#      must keep only as many components as the legend permits and must satisfy the legend's total-noise count.
# d) Placement legality
#    - The generator reserves the top 4 rows for the legend, with row 3 left blank as a separator, so legend
#      pixels never touch the actual sprites.
#    - Real sprites are placed in three separated slot regions below the legend. Attached noise is only added
#      onto background cells adjacent to its owning clean sprite, while detached noise singletons are placed
#      away from every occupied cell so they remain separate components.
#    - Legend rows only use the two foreground colors already present in the scene; no extra instruction color
#      is introduced, which keeps the palette small and readable.
# e) Visual observability
#    - Humans can see a stable top band. Row 0 has one short run per foreground color: its length is the
#      number of real sprites of that color. Rows 1-2 each show one longer monochrome run: its length is the
#      total number of pixels of that color that disappear from input to output.
#    - In the scene, the shared color visibly appears in more than two disconnected places, yet only two of
#      those components survive in the output. That makes the legend's disambiguation role visible.
# f) Visual cleanliness
#    - The legend is spatially isolated by a blank separator row, so it reads as metadata rather than as a
#      fourth object cluster.
#    - Detached noise blobs are tiny singletons, not overlapping stacks or mixed-color tangles. The main
#      sprites still dominate the canvas visually.
# g) Change inferability
#    - No hidden numeric ordering is used. Counts are conveyed by visible run lengths, and component identity
#      is conveyed by connectivity. The solver never relies on raw color ids beyond equality.
# h) New solver assumptions
#    - The input has a fixed 4-row top legend band. Row 0 contains one contiguous run for each foreground
#      color, whose length is that color's true sprite count. Rows 1 and 2 each contain exactly one run whose
#      color identifies the entry and whose length is that color's total noise count. Row 3 is blank.
#    - For each foreground color, there exists a subset of components of the legend-declared size whose
#      cleaned preserved-pixel count matches the legend-declared keep total.
# i) Generator guarantees for those assumptions
#    - The generator writes the legend in exactly that format and rejects any sample for which the solver does
#      not reconstruct the intended output.
#    - At least the shared color always receives one detached singleton noise component, so the legend-guided
#      component-count constraint is behaviorally active rather than decorative.


GRID_SIZE = 22
LEGEND_HEIGHT = 4
SCENE_TOP = LEGEND_HEIGHT
NUM_SPRITES = 3
NUM_FOREGROUND_COLORS = 2

LAYOUTS = [
    [(5, 0, 11, 7), (5, 14, 11, 21), (13, 5, 21, 16)],
    [(5, 0, 12, 5), (5, 16, 12, 21), (14, 5, 21, 16)],
]

PIXEL_COUNT_MIN = 6
PIXEL_COUNT_MAX = 40
HEIGHT_MIN = 2
HEIGHT_MAX = 9
ODD_SIDE_W_RANGE = range(1, 6)
EVEN_SIDE_W_RANGE = range(2, 7)
DENSITY_LO = 0.35
DENSITY_HI = 0.89

ATTACHED_NOISE_FRAC_LO = 0.11
ATTACHED_NOISE_FRAC_HI = 0.24
ATTACHED_NOISE_MAX = 5
DETACHED_SINGLETON_MAX = 1

SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2
SPRITE_RETRIES = 240
GEN_RETRIES = 450
FLIP_LR_PROB = 0.5


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    pts = {(int(r), int(c)) for r, c in coords}
    scores: dict[int, int] = {}
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        scores[int(k)] = int(score)
    return scores


def _second_best_score(coords: list[tuple[int, int]], true_k: int, width: int) -> int:
    scores = _axis_scores(coords, width=width)
    return max(score for k, score in scores.items() if k != int(true_k))


def _best_axis_for_component(coords: np.ndarray, width: int) -> int:
    pts = {(int(r), int(c)) for r, c in coords}
    best_score = -1
    best_k = 0
    for k in range(2 * (width - 1) + 1):
        score = 0
        for r, c in pts:
            cc = k - c
            if 0 <= cc < width and (r, cc) in pts:
                score += 1
        if score > best_score:
            best_score = int(score)
            best_k = int(k)
    return best_k


def _clean_component_points(coords: np.ndarray, width: int) -> set[tuple[int, int]]:
    if coords.size == 0:
        return set()
    pts = {(int(r), int(c)) for r, c in coords}
    k = _best_axis_for_component(coords, width)
    kept: set[tuple[int, int]] = set()
    for r, c in pts:
        cc = k - c
        if 0 <= cc < width and (r, cc) in pts:
            kept.add((int(r), int(c)))
    return kept


def _row_runs(row: np.ndarray, bg: int) -> list[tuple[int, int, int]]:
    runs: list[tuple[int, int, int]] = []
    w = row.shape[0]
    c = 0
    while c < w:
        if int(row[c]) == int(bg):
            c += 1
            continue
        col = int(row[c])
        c0 = c
        while c < w and int(row[c]) == col:
            c += 1
        runs.append((col, int(c - c0), int(c0)))
    return runs


def _parse_legend(g: np.ndarray, bg: int) -> tuple[dict[int, int], dict[int, int]]:
    assert g.shape == (GRID_SIZE, GRID_SIZE), "legend parser expects 22x22 grids"
    legend = g[:LEGEND_HEIGHT, :]

    sprite_runs = _row_runs(legend[0], bg)
    assert len(sprite_runs) == NUM_FOREGROUND_COLORS, "row 0 must contain one run per foreground color"
    sprite_count_by_color = {int(col): int(length) for col, length, _ in sprite_runs}
    assert sorted(sprite_count_by_color.values()) == [1, 2], "legend sprite counts must be 1 and 2"

    noise_count_by_color: dict[int, int] = {}
    for rr in (1, 2):
        runs = _row_runs(legend[rr], bg)
        assert len(runs) == 1, "each noise row must contain exactly one run"
        col, length, _ = runs[0]
        noise_count_by_color[int(col)] = int(length)
    assert set(noise_count_by_color) == set(sprite_count_by_color), "legend rows must reference same colors"
    assert not np.any(legend[3] != bg), "legend separator row must be blank"
    return sprite_count_by_color, noise_count_by_color


def _choose_components_to_keep(
    components: list[dict], sprite_count: int, noise_total: int
) -> list[int]:
    total_area = sum(int(comp["area"]) for comp in components)
    target_keep = int(total_area - noise_total)
    if sprite_count <= 0 or sprite_count > len(components):
        raise ValueError("invalid sprite count for component selection")

    best_exact: tuple[tuple[int, ...], list[int]] | None = None
    best_fallback: tuple[tuple[int, ...], list[int]] | None = None

    for combo in itertools.combinations(range(len(components)), int(sprite_count)):
        keep_sum = sum(int(components[i]["keep_area"]) for i in combo)
        raw_sum = sum(int(components[i]["area"]) for i in combo)
        max_keep = max(int(components[i]["keep_area"]) for i in combo)
        key = (abs(keep_sum - target_keep), -keep_sum, -raw_sum, -max_keep)
        combo_list = [int(i) for i in combo]
        if keep_sum == target_keep:
            exact_key = (-keep_sum, -raw_sum, -max_keep)
            if best_exact is None or exact_key < best_exact[0]:
                best_exact = (exact_key, combo_list)
        if best_fallback is None or key < best_fallback[0]:
            best_fallback = (key, combo_list)

    if best_exact is not None:
        return best_exact[1]
    assert best_fallback is not None
    return best_fallback[1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore the true vertically symmetric sprites while preserving a top legend band.

    Input structure:
    - The top 4 rows are a fixed legend area. Row 0 gives each foreground color's true sprite count as a
      short monochrome run, rows 1-2 give total removable-pixel counts for those colors, and row 3 is blank.
    - Below the legend are exactly three near-symmetric sprites using only two foreground colors because one
      color is shared by two sprites. Extra detached same-color blobs may also appear as noise components.

    Rule:
    - Ignore the legend when analyzing symmetry. For each foreground color in the scene, split pixels into
      4-connected components and compute how many pixels each component would keep after ideal vertical
      symmetry denoising.
    - Use the legend's sprite-count and noise-count constraints to choose which components are the real
      sprites. Keep only those components, and within them erase pixels lacking a mirror counterpart.

    Key insight:
    - Same-color connectivity alone is still insufficient because some disconnected same-color components are
      pure noise. The legend resolves both ambiguities at once: how many components of each color should
      survive, and how many total pixels of that color must vanish.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    h, w = g.shape
    assert (h, w) == (GRID_SIZE, GRID_SIZE), "expected 22x22 grid"

    sprite_count_by_color, noise_count_by_color = _parse_legend(g, bg)
    scene = g[SCENE_TOP:, :]

    out = np.full_like(g, bg)
    out[:LEGEND_HEIGHT, :] = g[:LEGEND_HEIGHT, :]

    for col in sorted(sprite_count_by_color):
        comps = find_components(
            scene,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        assert comps, "every legend color must appear in the scene"

        prepared: list[dict] = []
        for comp in comps:
            coords = np.array(comp["coords"], dtype=int)
            kept = _clean_component_points(coords, width=w)
            prepared.append(
                {
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "kept": kept,
                    "keep_area": len(kept),
                }
            )

        chosen = set(
            _choose_components_to_keep(
                prepared,
                sprite_count=int(sprite_count_by_color[col]),
                noise_total=int(noise_count_by_color[col]),
            )
        )

        for idx, comp in enumerate(prepared):
            if idx not in chosen:
                continue
            for r, c in comp["kept"]:
                out[SCENE_TOP + int(r), int(c)] = int(col)

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
    options = _feasible_shape_options(max_h, max_w)
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({target_count for _, _, target_count, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> tuple[set[tuple[int, int]], int]:
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
        cells.add(random.choice(frontier))
        remaining -= 1

    return cells, seam_row


def _center_rows(h: int, center_count: int, bridge_row: int) -> np.ndarray:
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
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)
    placed: list[dict] = []

    for spec, (r0s, c0s, r1s, c1s) in zip(sprite_specs, slots):
        local = dict(spec)
        sprite = local["sprite"]
        h, w = sprite.shape
        if h > r1s - r0s + 1 or w > c1s - c0s + 1:
            return None

        r0 = random.randint(r0s, r1s - h + 1)
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, r0, c0, transparent=0)

        clean_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        clean_mask[r0 : r0 + h, c0 : c0 + w] = sprite != 0
        local["r0"] = int(r0)
        local["c0"] = int(c0)
        local["true_k"] = int(local["axis_k_local"] + 2 * c0)
        local["clean_mask"] = clean_mask
        local["clean_coords"] = [tuple(x) for x in np.argwhere(clean_mask)]
        placed.append(local)

    return out, placed


def _noise_candidates(clean_mask: np.ndarray, occupied_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
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

    adj[:LEGEND_HEIGHT, :] = False
    rows, cols_arr = np.where(adj)
    return list(zip(rows.tolist(), cols_arr.tolist()))


def _add_attached_noise(inp: np.ndarray, bg: int, spec: dict) -> int | None:
    color = int(spec["color"])
    true_k = int(spec["true_k"])
    clean_coords = list(spec["clean_coords"])
    clean_mask = np.array(spec["clean_mask"], dtype=bool)
    clean_count = len(clean_coords)

    lo = max(1, math.ceil(ATTACHED_NOISE_FRAC_LO * clean_count))
    hi = min(ATTACHED_NOISE_MAX, math.floor(ATTACHED_NOISE_FRAC_HI * clean_count))
    if lo > hi:
        return None
    noise_target = random.randint(lo, hi)
    candidates = _noise_candidates(clean_mask, inp != bg, true_k)
    if len(candidates) < noise_target:
        return None

    chosen: list[tuple[int, int]] = []
    forbidden: set[tuple[int, int]] = set()
    current_coords = list(clean_coords)
    for _ in range(noise_target):
        viable = [cell for cell in candidates if cell not in forbidden and cell not in chosen]
        if not viable:
            return None

        scored = []
        for cell in viable:
            alt = _second_best_score(current_coords + [cell], true_k, width=GRID_SIZE)
            scored.append((alt, random.random(), cell))
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        forbidden.add(cell)
        mirror = (cell[0], true_k - cell[1])
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add((int(mirror[0]), int(mirror[1])))

    if clean_count - _second_best_score(current_coords, true_k, width=GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return None

    for r, c in chosen:
        inp[int(r), int(c)] = color
    return len(chosen)


def _available_detached_cells(inp: np.ndarray, bg: int) -> list[tuple[int, int]]:
    occupied = inp != bg
    avail: list[tuple[int, int]] = []
    for r in range(LEGEND_HEIGHT, GRID_SIZE):
        for c in range(GRID_SIZE):
            if occupied[r, c]:
                continue
            ok = True
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < GRID_SIZE and 0 <= cc < GRID_SIZE and occupied[rr, cc]:
                    ok = False
                    break
            if ok:
                avail.append((int(r), int(c)))
    return avail


def _add_detached_singletons(inp: np.ndarray, bg: int, color: int, count: int) -> bool:
    for _ in range(int(count)):
        avail = _available_detached_cells(inp, bg)
        if not avail:
            return False
        r, c = random.choice(avail)
        inp[int(r), int(c)] = int(color)
    return True


def _color_assignment() -> tuple[int, list[int], int, int]:
    bg, shared_color, unique_color = random.sample(range(10), NUM_FOREGROUND_COLORS + 1)
    sprite_colors = [shared_color, shared_color, unique_color]
    random.shuffle(sprite_colors)
    return int(bg), sprite_colors, int(shared_color), int(unique_color)


def _build_legend(
    bg: int, sprite_count_by_color: dict[int, int], noise_count_by_color: dict[int, int]
) -> np.ndarray | None:
    legend = np.full((LEGEND_HEIGHT, GRID_SIZE), bg, dtype=int)
    colors = list(sprite_count_by_color)

    row0_entries = [(int(col), int(sprite_count_by_color[col])) for col in colors]
    random.shuffle(row0_entries)
    row0_len = sum(length for _, length in row0_entries) + (len(row0_entries) - 1)
    if row0_len > GRID_SIZE:
        return None
    cur = random.randint(0, GRID_SIZE - row0_len)
    for idx, (col, length) in enumerate(row0_entries):
        legend[0, cur : cur + length] = int(col)
        cur += length
        if idx + 1 < len(row0_entries):
            cur += 1

    noise_rows = colors[:]
    random.shuffle(noise_rows)
    for rr, col in zip((1, 2), noise_rows):
        length = int(noise_count_by_color[col])
        if length <= 0 or length > GRID_SIZE:
            return None
        start = random.randint(0, GRID_SIZE - length)
        legend[rr, start : start + length] = int(col)

    return legend


def _legend_component_metric(inp: np.ndarray, bg: int, shared_color: int, unique_color: int) -> tuple[int, int]:
    scene = inp[SCENE_TOP:, :]
    shared_components = len(
        find_components(
            scene,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(shared_color)],
            return_format="dict",
        )
    )
    unique_components = len(
        find_components(
            scene,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(unique_color)],
            return_format="dict",
        )
    )
    return int(shared_components), int(unique_components)


def _base_d12_like_output(grid: np.ndarray) -> np.ndarray:
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    scene = g[SCENE_TOP:, :]
    out = np.full_like(g, bg)
    out[:LEGEND_HEIGHT, :] = g[:LEGEND_HEIGHT, :]

    for col in colors_present(scene, exclude={bg}):
        comps = find_components(
            scene,
            bg=bg,
            connectivity=4,
            monochromatic=True,
            colors=[int(col)],
            return_format="dict",
        )
        for comp in comps:
            coords = np.array(comp["coords"], dtype=int)
            kept = _clean_component_points(coords, width=GRID_SIZE)
            for r, c in kept:
                out[SCENE_TOP + int(r), int(c)] = int(col)
    return out


def _gen_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    bg, sprite_colors, shared_color, unique_color = _color_assignment()
    slots = random.choice(LAYOUTS)

    sprite_specs: list[dict] = []
    try:
        for color, slot in zip(sprite_colors, slots):
            max_h = slot[2] - slot[0] + 1
            max_w = slot[3] - slot[1] + 1
            sprite, axis_k_local = _build_clean_sprite(int(color), max_h, max_w)
            sprite_specs.append({"color": int(color), "sprite": sprite, "axis_k_local": int(axis_k_local)})
    except ValueError:
        return None

    placed_result = _slot_place_sprites(bg, sprite_specs, slots)
    if placed_result is None:
        return None
    out_scene, placed_specs = placed_result

    inp_scene = out_scene.copy()
    noise_count_by_color = {int(shared_color): 0, int(unique_color): 0}
    sprite_count_by_color = {int(shared_color): 2, int(unique_color): 1}

    for spec in placed_specs:
        added = _add_attached_noise(inp_scene, bg, spec)
        if added is None:
            return None
        noise_count_by_color[int(spec["color"])] += int(added)

    detached_shared = random.randint(1, DETACHED_SINGLETON_MAX)
    if not _add_detached_singletons(inp_scene, bg, shared_color, detached_shared):
        return None
    noise_count_by_color[int(shared_color)] += int(detached_shared)

    detached_unique = random.randint(0, DETACHED_SINGLETON_MAX)
    if detached_unique > 0:
        if not _add_detached_singletons(inp_scene, bg, unique_color, detached_unique):
            return None
        noise_count_by_color[int(unique_color)] += int(detached_unique)

    legend = _build_legend(bg, sprite_count_by_color, noise_count_by_color)
    if legend is None:
        return None

    inp = inp_scene.copy()
    out = out_scene.copy()
    inp[:LEGEND_HEIGHT, :] = legend
    out[:LEGEND_HEIGHT, :] = legend

    if random.random() < FLIP_LR_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None

    d12_out = _base_d12_like_output(inp)
    if np.array_equal(d12_out, out):
        return None

    shared_components, unique_components = _legend_component_metric(inp, bg, shared_color, unique_color)
    if shared_components <= sprite_count_by_color[shared_color]:
        return None
    if unique_components < sprite_count_by_color[unique_color]:
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
