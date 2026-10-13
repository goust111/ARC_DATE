from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
LAYER_COUNT_RANGE = (3, 5)
MAX_GENERATION_RETRIES = 500

UNIQUE_SHAPES = (
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 1), (1, 1), (2, 1), (2, 0)),
    ((0, 0), (0, 1), (0, 2), (1, 1)),
    ((0, 1), (1, 0), (1, 1), (1, 2)),
    ((0, 0), (0, 1), (0, 2), (1, 0), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (1, 2), (1, 3)),
)

ZONE_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
)


def _shape_size(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _translate_shape(
    shape: tuple[tuple[int, int], ...], r0: int, c0: int
) -> tuple[tuple[int, int], ...]:
    return tuple((r0 + r, c0 + c) for r, c in shape)


def _cells_to_mask(shape: tuple[int, int], cells: tuple[tuple[int, int], ...]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _overlay_specs(specs: list[dict], order: list[int], band_shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    out = np.zeros(band_shape, dtype=int)
    owner = np.full(band_shape, -1, dtype=int)
    filled = np.zeros(band_shape, dtype=bool)
    for idx in order:
        occ = specs[idx]["mask"]
        claim = occ & (~filled)
        out[claim] = int(specs[idx]["main_color"])
        owner[claim] = int(idx)
        filled |= occ
    return out, owner


def _analyze_band(band: np.ndarray) -> dict | None:
    colors = [int(color) for color in np.unique(band) if int(color) != BG]
    if len(colors) not in {1, 2}:
        return None

    counts = Counter(map(int, band[band != BG].ravel()))
    if len(colors) == 1:
        main_color = colors[0]
        if counts[main_color] < 2:
            return None
        mask = band != BG
        return {"main_color": main_color, "pointer_color": None, "mask": mask}

    one_counts = [color for color in colors if counts[color] == 1]
    if len(one_counts) != 1:
        return None
    pointer_color = one_counts[0]
    main_color = colors[0] if colors[1] == pointer_color else colors[1]
    if counts[main_color] < 2:
        return None
    mask = band != BG
    return {"main_color": main_color, "pointer_color": pointer_color, "mask": mask}


def _valid_partition_specs(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h_total, w = g.shape
    matches = []
    for n_layers in range(LAYER_COUNT_RANGE[0], LAYER_COUNT_RANGE[1] + 1):
        if h_total % n_layers != 0:
            continue
        band_h = h_total // n_layers
        bands = [g[i * band_h : (i + 1) * band_h, :] for i in range(n_layers)]
        specs = []
        for band in bands:
            spec = _analyze_band(band)
            if spec is None:
                break
            specs.append(spec)
        if len(specs) != n_layers:
            continue

        main_colors = [spec["main_color"] for spec in specs]
        if len(set(main_colors)) != n_layers:
            continue
        pointerless = [i for i, spec in enumerate(specs) if spec["pointer_color"] is None]
        if len(pointerless) != 1:
            continue
        if any(spec["pointer_color"] == spec["main_color"] for spec in specs if spec["pointer_color"] is not None):
            continue
        if any(spec["pointer_color"] not in main_colors for spec in specs if spec["pointer_color"] is not None):
            continue

        top_idx = pointerless[0]
        order = [top_idx]
        used = {top_idx}
        current_color = main_colors[top_idx]
        ok = True
        for _ in range(n_layers - 1):
            nxt = [i for i, spec in enumerate(specs) if i not in used and spec["pointer_color"] == current_color]
            if len(nxt) != 1:
                ok = False
                break
            next_idx = nxt[0]
            order.append(next_idx)
            used.add(next_idx)
            current_color = main_colors[next_idx]
        if not ok or len(order) != n_layers:
            continue
        if len(used) != n_layers:
            continue
        if any(i not in used for i in range(n_layers)):
            continue
        matches.append({"n_layers": n_layers, "band_h": band_h, "band_w": w, "specs": specs, "order": order})
    return matches


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct the hidden layer chain from pointer pixels and collapse the horizontal bands into one transparent overlay.

    Input structure:
    - The grid consists of 3 to 5 equal-height horizontal bands on a black background.
    - Each band contains exactly one main foreground color; all but one band also contain one single-pixel foreign color.
    - The foreign pixel's color equals the main color of the layer directly above that band in the true overlay order.

    Rule:
    - Identify the unique band without a foreign pixel as the top layer, then follow pointer colors downward to recover the full layer chain.
    - Treat every band as a transparent layer on the same `(r, c)` coordinate system and collapse them to one band.
    - At each cell, scan from top layer to bottom layer and keep the first non-black pixel; if that visible pixel was a pointer, still output the visible band's main color.

    Key insight:
    - Band position and pixel count are decoys; only the chain of single foreign-color pointers reveals the true stacking priority, and those pointer cells also behave like real opaque foreground.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    matches = _valid_partition_specs(g)
    assert len(matches) == 1, f"expected one valid band partition, got {len(matches)}"
    match = matches[0]
    out, _owner = _overlay_specs(match["specs"], match["order"], (match["band_h"], match["band_w"]))
    return out


def _valid_band_heights(n_layers: int) -> list[int]:
    heights = []
    for band_h in range(4, (30 // n_layers) + 1):
        total_h = n_layers * band_h
        divisors = [n for n in range(LAYER_COUNT_RANGE[0], LAYER_COUNT_RANGE[1] + 1) if total_h % n == 0]
        if divisors == [n_layers]:
            heights.append(band_h)
    return heights


def _fit_shapes(
    band_shape: tuple[int, int],
    n_layers: int,
) -> tuple[list[np.ndarray], list[np.ndarray]] | None:
    h, w = band_shape
    occupied = np.zeros(band_shape, dtype=bool)
    unique_masks: list[np.ndarray | None] = [None] * n_layers
    zone_masks: list[np.ndarray | None] = [None] * (n_layers - 1)

    specs: list[tuple[str, int, tuple[tuple[int, int], ...]]] = []
    for i in range(n_layers):
        specs.append(("u", i, random.choice(UNIQUE_SHAPES)))
    for i in range(n_layers - 1):
        specs.append(("z", i, random.choice(ZONE_SHAPES)))
    specs.sort(key=lambda item: (-len(item[2]), item[0], item[1]))

    for kind, idx, shape in specs:
        candidates = []
        sh, sw = _shape_size(shape)
        for r0 in range(h - sh + 1):
            for c0 in range(w - sw + 1):
                cells = _translate_shape(shape, r0, c0)
                if any(occupied[r, c] for r, c in cells):
                    continue
                candidates.append(cells)
        if not candidates:
            return None
        cells = random.choice(candidates)
        for r, c in cells:
            occupied[r, c] = True
        mask = _cells_to_mask(band_shape, cells)
        if kind == "u":
            unique_masks[idx] = mask
        else:
            zone_masks[idx] = mask

    return [mask for mask in unique_masks if mask is not None], [mask for mask in zone_masks if mask is not None]


def _build_main_masks(unique_masks: list[np.ndarray], zone_masks: list[np.ndarray]) -> list[np.ndarray]:
    n_layers = len(unique_masks)
    band_shape = unique_masks[0].shape
    masks = []
    for layer_idx in range(n_layers):
        mask = unique_masks[layer_idx].copy()
        for zone_idx in range(min(layer_idx, len(zone_masks) - 1) + 1):
            mask |= zone_masks[zone_idx]
        masks.append(mask)
    assert all(int(mask.sum()) >= 2 for mask in masks)
    return masks


def _choose_pointer_positions(
    unique_masks: list[np.ndarray], main_masks: list[np.ndarray]
) -> tuple[list[tuple[int, int] | None], int, int] | None:
    n_layers = len(main_masks)
    pointers: list[tuple[int, int] | None] = [None] * n_layers
    used_positions: set[tuple[int, int]] = set()

    effect_layer = random.randint(1, n_layers - 2)
    effect_target = random.randint(effect_layer + 1, n_layers - 1)
    target_cells = [tuple(map(int, cell)) for cell in np.argwhere(unique_masks[effect_target])]
    if len(target_cells) < 2:
        return None
    effect_pos = random.choice(target_cells)
    pointers[effect_layer] = effect_pos
    used_positions.add(effect_pos)

    h, w = main_masks[0].shape
    for layer_idx in range(1, n_layers):
        if pointers[layer_idx] is not None:
            continue
        lower_unique = np.zeros((h, w), dtype=bool)
        for lower_idx in range(layer_idx + 1, n_layers):
            lower_unique |= unique_masks[lower_idx]
        lower_candidates = [
            (r, c)
            for r, c in np.argwhere(lower_unique & (~main_masks[layer_idx]))
            if (int(r), int(c)) not in used_positions
        ]
        empty_candidates = [
            (r, c)
            for r, c in np.argwhere(~main_masks[layer_idx])
            if (int(r), int(c)) not in used_positions
        ]
        if lower_candidates and random.random() < 0.7:
            choice = random.choice(lower_candidates)
        elif empty_candidates:
            choice = random.choice(empty_candidates)
        elif lower_candidates:
            choice = random.choice(lower_candidates)
        else:
            return None
        pointers[layer_idx] = (int(choice[0]), int(choice[1]))
        used_positions.add(pointers[layer_idx])

    return pointers, effect_layer, effect_target


def _render_bands(
    main_colors: list[int],
    pointer_colors: list[int | None],
    main_masks: list[np.ndarray],
    pointer_positions: list[tuple[int, int] | None],
) -> list[np.ndarray]:
    bands = []
    band_shape = main_masks[0].shape
    for main_color, pointer_color, mask, pointer_pos in zip(main_colors, pointer_colors, main_masks, pointer_positions):
        band = np.zeros(band_shape, dtype=int)
        band[mask] = int(main_color)
        if pointer_pos is not None:
            band[pointer_pos] = int(pointer_color)
        bands.append(band)
    return bands


def _pack_input(bands: list[np.ndarray], input_order: list[int]) -> np.ndarray:
    return np.concatenate([bands[idx] for idx in input_order], axis=0)


def _heuristic_overlay(specs: list[dict], order: list[int], band_shape: tuple[int, int]) -> np.ndarray:
    return _overlay_specs(specs, order, band_shape)[0]


def _case_ok(case: dict) -> bool:
    inp = case["input_grid"]
    expected = case["expected_output"]
    n_layers = case["n_layers"]
    band_shape = case["band_shape"]
    main_colors = case["main_colors"]
    true_specs = case["specs"]
    if inp.shape[0] != n_layers * band_shape[0] or expected.shape != band_shape:
        return False
    if case["input_order"] == list(range(n_layers)):
        return False
    if np.array_equal(expected, np.zeros_like(expected)):
        return False

    matches = _valid_partition_specs(inp)
    if len(matches) != 1 or matches[0]["n_layers"] != n_layers or matches[0]["band_h"] != band_shape[0]:
        return False

    owner = case["owner"]
    for layer_idx in range(n_layers):
        if not np.any(owner == layer_idx):
            return False

    for zone_idx, zone_mask in enumerate(case["zone_masks"]):
        if not np.all(expected[zone_mask] == main_colors[zone_idx]):
            return False

    effect_layer = case["effect_layer"]
    effect_target = case["effect_target"]
    effect_pos = case["pointer_positions"][effect_layer]
    if effect_pos is None:
        return False
    if not case["unique_masks"][effect_target][effect_pos]:
        return False
    if any(case["main_masks"][upper][effect_pos] for upper in range(effect_layer)):
        return False
    if expected[effect_pos] != main_colors[effect_layer]:
        return False
    if not case["main_masks"][effect_target][effect_pos]:
        return False

    predicted = main(inp)
    if not np.array_equal(predicted, expected):
        return False

    stripe_order = list(range(n_layers))
    stripe_out = _heuristic_overlay(matches[0]["specs"], stripe_order, band_shape)
    if np.array_equal(stripe_out, expected):
        return False

    areas = [int(spec["mask"].sum()) for spec in true_specs]
    for reverse in (False, True):
        area_order = sorted(range(n_layers), key=lambda idx: (areas[idx], idx), reverse=reverse)
        area_out = _heuristic_overlay(true_specs, area_order, band_shape)
        if np.array_equal(area_out, expected):
            return False

    density = np.count_nonzero(expected) / expected.size
    if not (0.15 <= density <= 0.75):
        return False

    colors = [int(color) for color in np.unique(expected) if int(color) != BG]
    if len(colors) < min(3, n_layers):
        return False

    if _component_count(expected != BG) > 2 * n_layers:
        return False

    return True


def _make_case() -> dict | None:
    n_layers = random.randint(*LAYER_COUNT_RANGE)
    band_h = random.choice(_valid_band_heights(n_layers))
    band_w = random.randint(max(9, 2 * n_layers + 2), 16)
    band_shape = (band_h, band_w)

    placed = _fit_shapes(band_shape, n_layers)
    if placed is None:
        return None
    unique_masks, zone_masks = placed
    main_masks = _build_main_masks(unique_masks, zone_masks)

    main_colors = random.sample(PALETTE, n_layers)
    pointer_colors = [None] + main_colors[:-1]

    pointer_data = _choose_pointer_positions(unique_masks, main_masks)
    if pointer_data is None:
        return None
    pointer_positions, effect_layer, effect_target = pointer_data

    specs = []
    for layer_idx in range(n_layers):
        occ = main_masks[layer_idx].copy()
        pointer_pos = pointer_positions[layer_idx]
        if pointer_pos is not None:
            occ[pointer_pos] = True
        specs.append(
            {
                "main_color": int(main_colors[layer_idx]),
                "pointer_color": pointer_colors[layer_idx],
                "mask": occ,
            }
        )

    expected_output, owner = _overlay_specs(specs, list(range(n_layers)), band_shape)
    bands = _render_bands(main_colors, pointer_colors, main_masks, pointer_positions)

    input_order = list(range(n_layers))
    random.shuffle(input_order)
    if input_order == list(range(n_layers)):
        return None
    input_grid = _pack_input(bands, input_order)

    return {
        "n_layers": n_layers,
        "band_shape": band_shape,
        "main_colors": main_colors,
        "pointer_colors": pointer_colors,
        "main_masks": main_masks,
        "unique_masks": unique_masks,
        "zone_masks": zone_masks,
        "pointer_positions": pointer_positions,
        "specs": specs,
        "expected_output": expected_output,
        "owner": owner,
        "input_grid": input_grid,
        "input_order": input_order,
        "effect_layer": effect_layer,
        "effect_target": effect_target,
    }


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_RETRIES):
        case = _make_case()
        if case is None:
            continue
        if _case_ok(case):
            return case["input_grid"]
    raise RuntimeError("failed to generate a valid transparent-overlay chain puzzle")
