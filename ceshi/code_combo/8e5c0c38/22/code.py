# Combo variant of 8e5c0c38 (index 22)
# combo_spec: D11/1+D12/0
# Name: 共享颜色优先级处理

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
# a) Concrete behavior trace
#    - Source task: every non-bg color names exactly one sprite, so `all pixels of color c` is already the
#      right object. The solver can pick one vertical axis per color and delete same-color pixels lacking a
#      mirror partner.
#    - D12 adds a shared color. Here the shared color is stronger than the single-variant reference: the two
#      reused-color sprites sit on the left and right of a compact middle sprite, and a thin same-color bridge
#      connects them above the middle sprite. So the repeated color is not split into two 4-components in the
#      input anymore; connectivity alone is insufficient.
#    - D11 adds sequence. The compact middle color is cleaned first. Its repaired axis is then used as the
#      divider that splits the wide shared-color layer into left and right subsets, and each subset is cleaned
#      with its own local symmetry axis.
#    - Small trace:
#      old per-color logic on shared color S -> one connected set spanning left bridge right -> one best axis
#      is searched for the whole union -> the bridge may be preserved or the two side sprites may be judged
#      against a compromise axis.
#      new combo logic -> clean the unique middle sprite U first -> read U's cleaned vertical axis k_mid ->
#      partition S into pixels left of k_mid and right of k_mid -> denoise those two subsets independently.
# b) Complexity calibration
#    - The visual difference is explicit but controlled: there are still exactly 3 sprites and still only
#      delete-to-clean behavior, but one color now spans two large side lobes plus a thin connector.
#    - Only one sequencing dependency is introduced: the middle unique sprite acts as the reference divider.
#      This avoids an explosion of cases while making the multi-step rule observable from a few examples.
#    - The bridge is short enough that the side sprites remain visually dominant; the task still reads as
#      "restore mirror symmetry", not as arbitrary path editing.
# c) Implicit-assumption audit of the original main()
#    - `for col in colors`: still usable, but colors can no longer all be solved independently.
#    - `pts = np.argwhere(g == col)`: broken for the shared color because the left sprite, right sprite, and
#      connector bridge all share one color and appear as one merged pixel set.
#    - `g[r, cc] == col` mirror testing over the whole color layer: broken for the shared color because valid
#      mirror checks must stay inside one side subset after splitting by the middle reference axis.
#    - One-pass independence across colors: broken by the combo. The middle sprite must be settled first so
#      its cleaned axis can be reused when processing the wide shared color.
# d) Placement legality
#    - Generator uses three non-overlapping horizontal slots: left shared, center unique, right shared.
#    - All three sprites are placed with a common top row, so the shared-color bridge can run on the clear row
#      immediately above them without leaving the grid or touching the middle sprite.
#    - The center sprite is forced to have even width, hence a half-integer axis. That guarantees the divider
#      has no ambiguous grid column and every shared-color bridge pixel lies strictly on one side or the other.
# e) Visual observability
#    - A human sees one compact middle color and one wider color occupying both flanks with a thin connector.
#      Output removes the connector and cleans each flank locally, so the "middle color first, shared color
#      last" dependency is visible from the examples rather than hidden in metadata.
#    - The sequencing cue is geometric, not numeric: the reused color is the horizontally wide layer and the
#      reference color is the compact center sprite.
# f) Visual cleanliness
#    - Colors never overlap. The bridge lives on one empty row above the three sprites, so it is legible as a
#      same-color attachment rather than clutter buried inside other shapes.
#    - The center sprite remains separated from the bridge by one row, keeping the divider role readable.
# g) Color / rule audit
#    - No recoloring is introduced. The only color-system change is that one foreground color intentionally
#      encodes two side sprites plus their connector noise.
#    - The new ordering rule is realizable from visible structure: compact single-sprite color first, wide
#      shared color last.
# h) New implicit assumptions introduced by this combo solver
#    - The first processed color must be the compact reference sprite. Generator guarantees this by making its
#      horizontal span strictly smaller than the shared color's span.
#    - The shared color must place pixels on both sides of the repaired middle axis, and no shared pixel may
#      lie exactly on that axis. Generator guarantees this via a half-integer middle axis and left/right slots.
#    - After splitting by the middle axis, each side subset should correspond to one near-symmetric sprite with
#      only delete-only noise. Generator validates this and rejects degenerate cases where a baseline
#      component-only solver already matches the target output.


def _axis_scores(coords: list[tuple[int, int]], width: int) -> dict[int, int]:
    """Return {axis_k: mirror-match count} for all candidate vertical axes."""
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
    """Return the highest axis score excluding `true_k`."""
    scores = _axis_scores(coords, width=width)
    return max(s for k, s in scores.items() if k != int(true_k))


def _best_axis_for_coords(coords: np.ndarray, width: int) -> int:
    """Return the best vertical axis k for this coordinate set."""
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


def _erase_unmatched(out: np.ndarray, coords: np.ndarray, axis_k: int, bg: int) -> None:
    """Delete pixels from `out` when their mirror partner is absent within `coords`."""
    h, w = out.shape
    pts = {(int(r), int(c)) for r, c in coords}
    for r, c in pts:
        rr = int(r)
        cc = int(axis_k) - int(c)
        if not (0 <= rr < h and 0 <= cc < w and (rr, cc) in pts):
            out[rr, int(c)] = int(bg)


def _clean_color_by_components(out: np.ndarray, color: int, bg: int) -> None:
    """Fallback/base cleaning: split a color into 4-components and clean each independently."""
    _h, w = out.shape
    comps = find_components(
        out,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    for comp in comps:
        coords = np.array(comp["coords"], dtype=int)
        if coords.size == 0:
            continue
        k = _best_axis_for_coords(coords, w)
        _erase_unmatched(out, coords, k, bg)


def _largest_component_axis(grid: np.ndarray, color: int, bg: int) -> int | None:
    """Return the best axis of the largest current component of `color`."""
    comps = find_components(
        grid,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(color)],
        return_format="dict",
    )
    if not comps:
        return None
    largest = max(comps, key=lambda comp: int(comp["area"]))
    coords = np.array(largest["coords"], dtype=int)
    return _best_axis_for_coords(coords, grid.shape[1])


def _baseline_component_solver(grid: np.ndarray) -> np.ndarray:
    """Component-only solver used to reject generator samples that collapse back to D12-only behavior."""
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    out = g.copy()
    for col in colors_present(g, exclude={bg}):
        _clean_color_by_components(out, int(col), bg)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Restore each noisy sprite, but process the compact reference color before splitting the reused color.

    Input structure:
    - One background color fills most cells. There are exactly three near-vertically-symmetric sprites using
      only two foreground colors: one compact middle sprite with its own color, and two side sprites that share
      a color and are linked by a thin same-color bridge.
    - The middle sprite has one local symmetry axis. The wide shared color actually hides two side sprites, one
      on each side of the middle axis, plus unmatched connector noise.

    Rule:
    - Process foreground colors in priority order by compactness: clean the narrow reference color first by the
      usual component-wise vertical symmetry rule.
    - Use the cleaned reference sprite's vertical axis as a divider. Then split the remaining shared-color
      pixels into left-of-divider and right-of-divider subsets, and clean each subset independently by deleting
      pixels without a mirror partner inside that same subset.

    Key insight:
    - The repeated color cannot be solved as one connected component. The compact middle sprite must be repaired
      first because its cleaned axis is the cue that separates the merged shared-color layer into two hidden
      sprites before symmetry denoising can finish the task.
    """
    g = np.array(grid, dtype=int)
    bg = int(get_bg(g))
    _h, w = g.shape

    out = g.copy()
    colors = colors_present(g, exclude={bg})
    assert len(colors) >= 2, "grid must contain at least two non-background colors"

    color_infos = []
    for col in colors:
        coords = np.argwhere(g == int(col))
        if coords.size == 0:
            continue
        cols = coords[:, 1]
        span = int(cols.max() - cols.min() + 1)
        color_infos.append({"color": int(col), "span": span, "area": int(coords.shape[0])})

    assert color_infos, "grid must contain foreground colors"
    color_infos.sort(key=lambda info: (int(info["span"]), int(info["area"]), int(info["color"])))

    reference_axis_k: int | None = None
    for idx, info in enumerate(color_infos):
        col = int(info["color"])
        is_last = idx == len(color_infos) - 1

        current_coords = np.argwhere(out == int(col))
        if current_coords.size == 0:
            continue

        if is_last and reference_axis_k is not None:
            left_coords = current_coords[2 * current_coords[:, 1] < int(reference_axis_k)]
            right_coords = current_coords[2 * current_coords[:, 1] > int(reference_axis_k)]
            if left_coords.size > 0 and right_coords.size > 0:
                for subset in (left_coords, right_coords):
                    k = _best_axis_for_coords(subset, w)
                    _erase_unmatched(out, subset, k, bg)
                continue

        _clean_color_by_components(out, col, bg)
        if reference_axis_k is None:
            reference_axis_k = _largest_component_axis(out, col, bg)

    return out


# --------------- generator (combo variant) ---------------

# ── CFG: canvas & layout ──
GRID_SIZE = 22
LAYOUTS = [
    {
        "row_band": (4, 12),
        "left": (4, 0, 12, 6),
        "center": (4, 8, 12, 13),
        "right": (4, 15, 12, 21),
    },
    {
        "row_band": (6, 14),
        "left": (6, 0, 14, 6),
        "center": (6, 8, 14, 13),
        "right": (6, 15, 14, 21),
    },
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
SHARED_MIN_PIXELS = 14
UNIQUE_MIN_PIXELS = 8

# ── CFG: noise / bridge ──
NOISE_FRAC_LO = 0.11
NOISE_FRAC_HI = 0.33
NOISE_MAX = 12
SPRITE_AXIS_GAP_MIN = 3
FINAL_AXIS_GAP_MIN = 2

# ── CFG: retry limits ──
SPRITE_RETRIES = 260
GEN_RETRIES = 500

# ── CFG: augmentation ──
FLIP_PROB = 0.5


def _feasible_shape_options(
    max_h: int,
    max_w: int,
    *,
    min_pixels: int,
    max_pixels: int,
    force_even_width: bool | None,
) -> list[tuple[int, int, int, int]]:
    """Enumerate feasible (h, w, target_count, center_count) tuples under the variant constraints."""
    options: list[tuple[int, int, int, int]] = []
    h_max = min(HEIGHT_MAX, max_h)
    for target_count in range(int(min_pixels), int(max_pixels) + 1):
        for h in range(HEIGHT_MIN, h_max + 1):
            if force_even_width is not True:
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

            if force_even_width is False:
                continue
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


def _choose_shape_spec(
    max_h: int,
    max_w: int,
    *,
    min_pixels: int,
    max_pixels: int = PIXEL_COUNT_MAX,
    force_even_width: bool | None = None,
) -> tuple[int, int, int, int]:
    """Pick one feasible clean-shape spec, biased toward denser fills."""
    options = _feasible_shape_options(
        max_h,
        max_w,
        min_pixels=min_pixels,
        max_pixels=max_pixels,
        force_even_width=force_even_width,
    )
    if not options:
        raise ValueError("no feasible clean sprite spec")
    counts = sorted({tc for _, _, tc, _ in options})
    target_count = random.choice(counts)
    target_options = [opt for opt in options if opt[2] == target_count]
    weights = [1.0 / (h * w) for h, w, _, _ in target_options]
    return random.choices(target_options, weights=weights, k=1)[0]


def _left_cells(h: int, side_w: int, left_count: int) -> tuple[set[tuple[int, int]], int]:
    """Grow a connected left half plus one seam row so the mirrored sprite stays 4-connected."""
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
    """Choose which rows contain an axis-column cell for odd-width sprites."""
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


def _build_clean_sprite(
    color: int,
    max_h: int,
    max_w: int,
    *,
    min_pixels: int,
    force_even_width: bool | None,
) -> tuple[np.ndarray, int]:
    """Build one connected vertically symmetric sprite and return (sprite, local_axis_k)."""
    for _ in range(SPRITE_RETRIES):
        h, w, target_count, center_count = _choose_shape_spec(
            max_h,
            max_w,
            min_pixels=min_pixels,
            force_even_width=force_even_width,
        )
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


def _place_aligned_sprites(
    bg: int,
    sprite_specs: list[dict],
    layout: dict[str, tuple[int, int, int, int] | tuple[int, int]],
) -> tuple[np.ndarray, list[dict]] | None:
    """Place left / center / right sprites using one shared top row."""
    out = np.full((GRID_SIZE, GRID_SIZE), bg, dtype=int)

    top_band = layout["row_band"]
    top_lo = int(top_band[0])
    top_hi = int(top_band[1])
    max_r0 = top_hi
    for spec, key in zip(sprite_specs, ("left", "center", "right")):
        r0s, _c0s, r1s, _c1s = layout[key]
        h, _w = spec["sprite"].shape
        if h > r1s - r0s + 1:
            return None
        max_r0 = min(max_r0, r1s - h + 1)
    if top_lo > max_r0:
        return None
    common_r0 = random.randint(top_lo, max_r0)

    placed: list[dict] = []
    for spec, key in zip(sprite_specs, ("left", "center", "right")):
        r0s, c0s, r1s, c1s = layout[key]
        spec = dict(spec)
        sprite = spec["sprite"]
        h, w = sprite.shape
        if not (r0s <= common_r0 <= r1s - h + 1):
            return None
        c0 = random.randint(c0s, c1s - w + 1)
        paste(out, sprite, common_r0, c0, transparent=0)

        clean_mask = np.zeros((GRID_SIZE, GRID_SIZE), dtype=bool)
        clean_mask[common_r0 : common_r0 + h, c0 : c0 + w] = sprite != 0
        coords = [tuple(x) for x in np.argwhere(clean_mask)]
        spec["slot_key"] = key
        spec["r0"] = common_r0
        spec["c0"] = c0
        spec["bbox"] = (common_r0, c0, common_r0 + h - 1, c0 + w - 1)
        spec["true_k"] = int(spec["axis_k_local"] + 2 * c0)
        spec["clean_mask"] = clean_mask
        spec["clean_coords"] = coords
        placed.append(spec)
    return out, placed


def _noise_candidates(clean_mask: np.ndarray, occupied_mask: np.ndarray, true_k: int) -> list[tuple[int, int]]:
    """Find globally empty cells adjacent to one clean sprite and invalid under its mirror axis."""
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


def _add_local_noise(inp: np.ndarray, bg: int, spec: dict) -> bool:
    """Add standard one-sided symmetry-breaking noise to one sprite."""
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
        scored.sort(key=lambda item: (item[0], item[1]))
        best_alt = scored[0][0]
        pool = [cell for alt, _, cell in scored if alt <= best_alt + 1]
        cell = random.choice(pool)
        chosen.append(cell)
        current_coords.append(cell)
        forbidden.add(cell)
        mirror = (cell[0], true_k - cell[1])
        if 0 <= mirror[1] < GRID_SIZE:
            forbidden.add(mirror)

    if clean_count - _second_best_score(current_coords, true_k, width=GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return False

    for r, c in chosen:
        inp[r, c] = color
    return True


def _top_bridge_candidates(spec: dict, occupied_mask: np.ndarray) -> list[tuple[int, int]]:
    """Candidate bridge endpoints on the clear row above one shared sprite."""
    r0, _c0, _r1, _c1 = spec["bbox"]
    candidates = _noise_candidates(spec["clean_mask"], occupied_mask, int(spec["true_k"]))
    return [cell for cell in candidates if cell[0] == int(r0) - 1]


def _add_shared_bridge(inp: np.ndarray, bg: int, left_spec: dict, right_spec: dict, divider_k: int) -> list[tuple[int, int]] | None:
    """Connect the two shared-color side sprites with a thin bridge on the row above them."""
    occupied_mask = inp != bg
    left_candidates = _top_bridge_candidates(left_spec, occupied_mask)
    right_candidates = _top_bridge_candidates(right_spec, occupied_mask)
    if not left_candidates or not right_candidates:
        return None

    left_candidates.sort(key=lambda cell: (-cell[1], random.random()))
    right_candidates.sort(key=lambda cell: (cell[1], random.random()))

    for left_cell in left_candidates[:6]:
        for right_cell in right_candidates[:6]:
            if left_cell[0] != right_cell[0]:
                continue
            row = int(left_cell[0])
            c0 = int(left_cell[1])
            c1 = int(right_cell[1])
            if c0 >= c1:
                continue
            path = [(row, c) for c in range(c0, c1 + 1)]
            if any(occupied_mask[r, c] for r, c in path):
                continue
            left_half = [cell for cell in path if 2 * cell[1] < int(divider_k)]
            right_half = [cell for cell in path if 2 * cell[1] > int(divider_k)]
            if not left_half or not right_half:
                continue
            for r, c in path:
                inp[r, c] = int(left_spec["color"])
            return path
    return None


def _color_assignment() -> tuple[int, int, int]:
    """Return (bg, shared_color, unique_color)."""
    bg, shared_color, unique_color = random.sample(range(10), 3)
    return int(bg), int(shared_color), int(unique_color)


def _combo_signature(inp: np.ndarray, out: np.ndarray, shared_color: int) -> dict[str, int | bool]:
    """Quantify whether a sample really exhibits the sequential shared-color behavior."""
    bg = int(get_bg(inp))
    inp_shared_comps = find_components(
        inp,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(shared_color)],
        return_format="dict",
    )
    out_shared_comps = find_components(
        out,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        colors=[int(shared_color)],
        return_format="dict",
    )
    baseline = _baseline_component_solver(inp)
    removed = int(np.sum((inp == int(shared_color)) & (out != int(shared_color))))
    return {
        "input_shared_components": len(inp_shared_comps),
        "output_shared_components": len(out_shared_comps),
        "shared_removed_pixels": removed,
        "baseline_diff": not np.array_equal(baseline, out),
    }


def _gen_attempt() -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]] | None:
    """Build one combo-valid sample and return (input, output, signature)."""
    bg, shared_color, unique_color = _color_assignment()
    layout = random.choice(LAYOUTS)

    try:
        left_sprite, left_axis_local = _build_clean_sprite(
            shared_color,
            layout["left"][2] - layout["left"][0] + 1,
            layout["left"][3] - layout["left"][1] + 1,
            min_pixels=SHARED_MIN_PIXELS,
            force_even_width=None,
        )
        center_sprite, center_axis_local = _build_clean_sprite(
            unique_color,
            layout["center"][2] - layout["center"][0] + 1,
            layout["center"][3] - layout["center"][1] + 1,
            min_pixels=UNIQUE_MIN_PIXELS,
            force_even_width=True,
        )
        right_sprite, right_axis_local = _build_clean_sprite(
            shared_color,
            layout["right"][2] - layout["right"][0] + 1,
            layout["right"][3] - layout["right"][1] + 1,
            min_pixels=SHARED_MIN_PIXELS,
            force_even_width=None,
        )
    except ValueError:
        return None

    sprite_specs = [
        {"color": shared_color, "sprite": left_sprite, "axis_k_local": left_axis_local},
        {"color": unique_color, "sprite": center_sprite, "axis_k_local": center_axis_local},
        {"color": shared_color, "sprite": right_sprite, "axis_k_local": right_axis_local},
    ]

    placed_result = _place_aligned_sprites(bg, sprite_specs, layout)
    if placed_result is None:
        return None
    out, placed_specs = placed_result
    left_spec, center_spec, right_spec = placed_specs

    if center_spec["true_k"] % 2 == 0:
        return None

    inp = out.copy()
    bridge_path = _add_shared_bridge(inp, bg, left_spec, right_spec, int(center_spec["true_k"]))
    if bridge_path is None:
        return None

    if not _add_local_noise(inp, bg, center_spec):
        return None

    left_noise = [cell for cell in bridge_path if 2 * cell[1] < int(center_spec["true_k"])]
    right_noise = [cell for cell in bridge_path if 2 * cell[1] > int(center_spec["true_k"])]
    if not left_noise or not right_noise:
        return None
    if len(left_noise) > NOISE_MAX or len(right_noise) > NOISE_MAX:
        return None

    left_coords = list(left_spec["clean_coords"]) + left_noise
    right_coords = list(right_spec["clean_coords"]) + right_noise
    if len(left_noise) < math.ceil(NOISE_FRAC_LO * len(left_spec["clean_coords"])):
        return None
    if len(right_noise) < math.ceil(NOISE_FRAC_LO * len(right_spec["clean_coords"])):
        return None
    if len(left_spec["clean_coords"]) - _second_best_score(left_coords, int(left_spec["true_k"]), GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return None
    if len(right_spec["clean_coords"]) - _second_best_score(right_coords, int(right_spec["true_k"]), GRID_SIZE) < FINAL_AXIS_GAP_MIN:
        return None

    if random.random() < FLIP_PROB:
        inp = np.flipud(inp)
        out = np.flipud(out)
    if random.random() < FLIP_PROB:
        inp = np.fliplr(inp)
        out = np.fliplr(out)

    if not np.array_equal(main(inp), out):
        return None

    sig = _combo_signature(inp, out, shared_color)
    if sig["input_shared_components"] != 1:
        return None
    if sig["output_shared_components"] != 2:
        return None
    if not bool(sig["baseline_diff"]):
        return None
    if int(sig["shared_removed_pixels"]) < len(left_noise) + len(right_noise):
        return None

    return inp, out, sig


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        result = _gen_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
