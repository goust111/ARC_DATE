from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, paste, random_free_location


PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
AREA_RANGE = range(4, 17)
GRID_H_RANGE = (15, 20)
GRID_W_RANGE = (15, 20)
MAX_GENERATION_ATTEMPTS = 400

FRAGMENT_PATTERNS: dict[int, list[np.ndarray]] = {
    1: [
        np.array([[1]], dtype=int),
    ],
    2: [
        np.array([[1, 1]], dtype=int),
        np.array([[1], [1]], dtype=int),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=int),
        np.array([[1], [1], [1]], dtype=int),
        np.array([[1, 1], [1, 0]], dtype=int),
        np.array([[1, 1], [0, 1]], dtype=int),
    ],
}

SUM_TO_FRAGMENT_OPTIONS: dict[int, list[list[int]]] = {
    1: [[1]],
    2: [[2], [1, 1]],
    3: [[3], [1, 2]],
    4: [[1, 3], [2, 2]],
    5: [[2, 3]],
    6: [[3, 3]],
}


def _foreground_colors(grid: np.ndarray) -> list[int]:
    return sorted(int(c) for c in np.unique(grid) if int(c) != Color.BLACK)


def _main_components_by_color(grid: np.ndarray) -> dict[int, dict]:
    comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    by_color: dict[int, list[dict]] = {}
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(comp)

    kept: dict[int, dict] = {}
    for color, entries in by_color.items():
        large = [entry for entry in entries if int(entry["area"]) > 3]
        small = [entry for entry in entries if int(entry["area"]) <= 3]
        assert len(large) == 1, "each color must have exactly one retained main component"
        assert len(small) >= 1, "each color must have at least one distractor fragment"
        kept[color] = large[0]
    return kept


def _main_area_order(grid: np.ndarray) -> list[tuple[int, int]]:
    kept = _main_components_by_color(grid)
    areas = [(int(comp["area"]), int(color)) for color, comp in kept.items()]
    assert len({area for area, _color in areas}) == len(areas), "retained areas must be unique"
    areas.sort(reverse=True)
    return areas


def main(grid: np.ndarray) -> np.ndarray:
    """
    Filter same-colored tiny distractors, count retained rectangle pixels, and render the colors as concentric square rings.

    Input structure:
    - The background is black.
    - There are 3-5 foreground colors, each appearing in one larger filled rectangle plus 1-2 same-colored tiny 4-connected fragments.
    - All components are mutually separated, and only the larger component of each color has area greater than 3.

    Rule:
    - For each foreground color, find its 4-connected monochrome components.
    - Discard every component with area at most 3 and keep the unique larger component as that color's main object.
    - Count the kept pixels for each color, sort colors by this count from largest to smallest, and build a square output of side 2n-1 for n colors.
    - Fill nested centered squares in that order so the first color becomes the outer ring, the next color the next inner ring, and the last color the single center cell.

    Key insight:
    - Same-colored noise cannot be removed by color alone; the correct layer order appears only after filtering components by the hidden area threshold and then ranking colors by retained pixel count.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    ordered = _main_area_order(g)
    n = len(ordered)
    side = 2 * n - 1
    out = np.full((side, side), Color.BLACK, dtype=int)
    for layer, (_area, color) in enumerate(ordered):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs = []
    for h in range(1, area + 1):
        if area % h == 0:
            pairs.append((h, area // h))
    return pairs


def _choose_main_specs(colors: list[int]) -> dict[int, dict] | None:
    n = len(colors)
    for _ in range(200):
        anchor_low = random.randint(4, 13)
        diff = random.choice([1, 2])
        anchor_high = anchor_low + diff
        used = {anchor_low, anchor_high}
        remaining = [area for area in AREA_RANGE if area not in used]
        extra_areas = random.sample(remaining, n - 2)
        areas = [anchor_high, anchor_low, *extra_areas]
        random.shuffle(areas)

        pairs = sorted(areas)
        if not any(b - a <= 2 for a, b in zip(pairs, pairs[1:])):
            continue

        specs: dict[int, dict] = {}
        assigned_target = False
        for color, area in zip(colors, areas):
            dims = random.choice(_factor_pairs(area))
            if len(_factor_pairs(area)) > 1 and random.random() < 0.5:
                dims = dims[::-1]
            specs[int(color)] = {"main_area": int(area), "main_shape": tuple(map(int, dims))}
            if not assigned_target and area == anchor_high:
                specs[int(color)]["target_role"] = "hi"
                assigned_target = True
        for color in colors:
            if specs[int(color)]["main_area"] == anchor_low:
                specs[int(color)]["target_role"] = "lo"
                break
        if sum(spec.get("target_role") == "hi" for spec in specs.values()) == 1 and sum(
            spec.get("target_role") == "lo" for spec in specs.values()
        ) == 1:
            return specs
    return None


def _choose_fragment_partition(total: int, *, require_three: bool = False) -> list[int]:
    options = [list(partition) for partition in SUM_TO_FRAGMENT_OPTIONS[int(total)]]
    random.shuffle(options)
    if require_three:
        options.sort(key=lambda parts: 3 not in parts)
    for parts in options:
        if not require_three or 3 in parts:
            return parts
    return options[0]


def _assign_fragment_specs(specs: dict[int, dict]) -> bool:
    hi_color = next(color for color, spec in specs.items() if spec.get("target_role") == "hi")
    lo_color = next(color for color, spec in specs.items() if spec.get("target_role") == "lo")
    diff = specs[hi_color]["main_area"] - specs[lo_color]["main_area"]
    if diff not in {1, 2}:
        return False

    specs[hi_color]["fragments"] = _choose_fragment_partition(1)
    specs[lo_color]["fragments"] = _choose_fragment_partition(4, require_three=True)

    other_colors = [color for color in specs if color not in {hi_color, lo_color}]
    for color in other_colors:
        total = random.randint(1, 6)
        specs[color]["fragments"] = _choose_fragment_partition(total)

    if all(len(spec["fragments"]) == 1 for spec in specs.values()):
        color = random.choice(list(specs))
        specs[color]["fragments"] = _choose_fragment_partition(random.choice([4, 5, 6]))
    if all(len(spec["fragments"]) == 2 for spec in specs.values()):
        color = random.choice(list(specs))
        specs[color]["fragments"] = [random.choice([1, 2, 3])]

    main_order = tuple(color for _area, color in sorted(
        ((int(spec["main_area"]), int(color)) for color, spec in specs.items()),
        reverse=True,
    ))
    total_order = tuple(color for _area, color in sorted(
        (
            (int(spec["main_area"]) + sum(int(v) for v in spec["fragments"]), int(color))
            for color, spec in specs.items()
        ),
        reverse=True,
    ))
    threshold2_order = tuple(color for _area, color in sorted(
        (
            (
                int(spec["main_area"]) + sum(int(v) for v in spec["fragments"] if int(v) == 3),
                int(color),
            )
            for color, spec in specs.items()
        ),
        reverse=True,
    ))
    total_values = [
        int(spec["main_area"]) + sum(int(v) for v in spec["fragments"])
        for spec in specs.values()
    ]
    if len(set(total_values)) != len(total_values):
        return False
    if total_order == main_order or threshold2_order == main_order:
        return False

    component_counts = [1 + len(spec["fragments"]) for spec in specs.values()]
    if len(set(component_counts)) == 1:
        return False
    area_rank = {color: rank for rank, (_area, color) in enumerate(sorted(
        ((int(spec["main_area"]), int(color)) for color, spec in specs.items()),
        reverse=True,
    ))}
    count_rank = {color: 1 + len(spec["fragments"]) for color, spec in specs.items()}
    inversion = any(
        (area_rank[a] < area_rank[b]) and (count_rank[a] < count_rank[b])
        for a in specs
        for b in specs
        if a != b
    )
    if not inversion:
        return False
    return True


def _make_component_patch(color: int, shape: tuple[int, int] | None = None, area: int | None = None) -> np.ndarray:
    if shape is not None:
        h, w = map(int, shape)
        return np.full((h, w), int(color), dtype=int)
    assert area is not None
    mask = random.choice(FRAGMENT_PATTERNS[int(area)])
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask.astype(bool)] = int(color)
    return patch


def _build_scene(specs: dict[int, dict]) -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    if max(h, w) > 30:
        return None

    canvas = np.full((h, w), Color.BLACK, dtype=int)
    patches: list[tuple[int, np.ndarray]] = []
    for color, spec in specs.items():
        patches.append((int(spec["main_area"]), _make_component_patch(int(color), shape=spec["main_shape"])))
        for frag_area in spec["fragments"]:
            patches.append((int(frag_area), _make_component_patch(int(color), area=int(frag_area))))

    random.shuffle(patches)
    patches.sort(key=lambda item: (-int(item[0]), random.random()))

    for _area, patch in patches:
        try:
            r0, c0 = random_free_location(
                canvas,
                patch,
                transparent=Color.BLACK,
                border=1,
                padding=1,
                padding_connectivity=8,
            )
        except ValueError:
            return None
        paste(canvas, patch, r0, c0, transparent=Color.BLACK)
    return canvas


def _validate_scene(grid: np.ndarray) -> bool:
    colors = _foreground_colors(grid)
    if not (3 <= len(colors) <= 5):
        return False
    if not set(colors).issubset(set(PALETTE)):
        return False

    comps = find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    by_color: dict[int, list[int]] = {}
    component_total = 0
    for comp in comps:
        by_color.setdefault(int(comp["color"]), []).append(int(comp["area"]))
        component_total += 1
    if set(by_color) != set(colors):
        return False
    if len(
        find_components(
            grid,
            bg=Color.BLACK,
            connectivity=8,
            monochromatic=False,
            return_format="dict",
        )
    ) != component_total:
        return False

    main_areas = {}
    total_areas = {}
    threshold2_counts = {}
    component_counts = {}
    seen_three = False
    for color, areas in by_color.items():
        color_comps = [comp for comp in comps if int(comp["color"]) == int(color)]
        large = [comp for comp in color_comps if int(comp["area"]) > 3]
        small = [comp for comp in color_comps if int(comp["area"]) <= 3]
        if len(large) != 1 or len(small) not in {1, 2}:
            return False
        if any(int(comp["area"]) <= 0 for comp in color_comps):
            return False
        main_comp = large[0]
        r0, c0, r1, c1 = map(int, main_comp["bbox"])
        if int(main_comp["area"]) != (r1 - r0 + 1) * (c1 - c0 + 1):
            return False
        if not (4 <= int(main_comp["area"]) <= 16):
            return False
        if any(int(comp["area"]) == 3 for comp in small):
            seen_three = True
        main_areas[color] = int(main_comp["area"])
        total_areas[color] = sum(int(comp["area"]) for comp in color_comps)
        threshold2_counts[color] = int(main_comp["area"]) + sum(
            int(comp["area"]) for comp in small if int(comp["area"]) == 3
        )
        component_counts[color] = len(color_comps)

    if len(set(main_areas.values())) != len(main_areas):
        return False
    if len(set(total_areas.values())) != len(total_areas):
        return False
    if not seen_three:
        return False

    main_order = tuple(color for color, _area in sorted(main_areas.items(), key=lambda item: item[1], reverse=True))
    total_order = tuple(color for color, _area in sorted(total_areas.items(), key=lambda item: item[1], reverse=True))
    threshold2_order = tuple(
        color for color, _area in sorted(threshold2_counts.items(), key=lambda item: item[1], reverse=True)
    )
    if total_order == main_order or threshold2_order == main_order:
        return False

    if len(set(component_counts.values())) == 1:
        return False
    area_rank = {color: i for i, color in enumerate(main_order)}
    if not any(
        area_rank[a] < area_rank[b] and component_counts[a] < component_counts[b]
        for a in colors
        for b in colors
        if a != b
    ):
        return False

    out = main(grid)
    expected_side = 2 * len(colors) - 1
    if out.shape != (expected_side, expected_side):
        return False
    if not np.array_equal(out, _render_expected_output(main_order)):
        return False
    return True


def _render_expected_output(order: tuple[int, ...]) -> np.ndarray:
    n = len(order)
    side = 2 * n - 1
    out = np.full((side, side), Color.BLACK, dtype=int)
    for layer, color in enumerate(order):
        out[layer : side - layer, layer : side - layer] = int(color)
    return out


def _gen_attempt() -> np.ndarray | None:
    n_colors = random.randint(3, 5)
    colors = random.sample(PALETTE, n_colors)
    specs = _choose_main_specs(colors)
    if specs is None or not _assign_fragment_specs(specs):
        return None
    grid = _build_scene(specs)
    if grid is None:
        return None
    return grid if _validate_scene(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
