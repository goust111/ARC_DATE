from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLUE = 1
RED = 2
GREEN = 3
GREY = 5
ORANGE = 7
ALLOWED_COLORS = {BLUE, RED, GREEN, GREY}

COUNT_WEIGHTS = ((3, 4), (4, 4), (5, 2))
GRID_BASE_BY_COUNT = {3: 18, 4: 22, 5: 26}
GRID_JITTER = 4
GEN_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _neighbors8(r: int, c: int) -> list[tuple[int, int]]:
    return [
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    ]


def _nonbg_components(grid: np.ndarray, bg: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors8(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _diamond_outline(center_r: int, center_c: int, radius: int) -> set[tuple[int, int]]:
    return {
        (center_r + dr, center_c + dc)
        for dr in range(-radius, radius + 1)
        for dc in range(-radius, radius + 1)
        if abs(dr) + abs(dc) == radius
    }


def _expanded8(coords: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in coords:
        for nr in range(r - 1, r + 2):
            for nc in range(c - 1, c + 2):
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _fit_diamond(comp: list[tuple[int, int]]) -> tuple[int, int, int]:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height == width and height % 2 == 1, "diamond outline must have an odd square bbox"

    radius = height // 2
    center_r = r0 + radius
    center_c = c0 + radius
    expected = _diamond_outline(center_r, center_c, radius)
    actual = set(comp)
    assert actual == expected, "component must be a full Manhattan-diamond outline"
    return center_r, center_c, radius


def _adjusted_radius(color: int, radius: int) -> int:
    if color == RED:
        return radius - 1
    if color == GREY:
        return radius + 1
    assert color in (BLUE, GREEN)
    return radius


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict[str, int]]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    components = _nonbg_components(g, bg)
    assert 3 <= len(components) <= 5, "expected 3-5 separate diamonds"

    objects: list[dict[str, int]] = []
    for comp in components:
        colors = {int(g[r, c]) for r, c in comp}
        assert len(colors) == 1, "each diamond must be monochrome"
        color = colors.pop()
        assert color in ALLOWED_COLORS, "unexpected diamond color"
        center_r, center_c, radius = _fit_diamond(comp)
        objects.append(
            {
                "color": color,
                "center_r": center_r,
                "center_c": center_c,
                "radius": radius,
                "new_radius": _adjusted_radius(color, radius),
            }
        )

    radii = [obj["radius"] for obj in objects]
    new_radii = [obj["new_radius"] for obj in objects]
    assert len(radii) == len(set(radii)), "input radii must be distinct"
    assert len(new_radii) == len(set(new_radii)), "adjusted radii must be distinct"
    assert min(radii) >= 1
    assert min(new_radii) >= 1
    return bg, objects


def _build_output(bg: int, objects: list[dict[str, int]]) -> np.ndarray:
    transformed = sorted(objects, key=lambda obj: obj["new_radius"], reverse=True)
    max_radius = transformed[0]["new_radius"]
    out = np.full((2 * max_radius + 1, 2 * max_radius + 1), int(bg), dtype=int)
    center = max_radius
    for layer_idx, obj in enumerate(transformed):
        draw_radius = max_radius - layer_idx
        assert draw_radius >= 0
        for r, c in _diamond_outline(center, center, draw_radius):
            out[r, c] = int(obj["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Shrink or dilate colored diamond outlines, then reorder them by the adjusted radius and redraw them as concentric diamonds.

    Input structure:
    - The grid has an orange background and 3-5 separated one-pixel-thick Manhattan-diamond outlines.
    - Each outline is monochrome and uses red, grey, blue, or green.
    - Initial diamond radii are distinct, and each component exactly matches a full set of cells at one Manhattan distance from its center.

    Rule:
    - Detect every diamond outline and read its color and initial radius.
    - Adjust the radius by color: red shrinks by 1, grey grows by 1, and blue/green stay unchanged.
    - Sort the diamonds by adjusted radius from largest to smallest.
    - Create a square output with side `2*R_max+1`, where `R_max` is the largest adjusted radius, and redraw the sorted colors as concentric diamond layers from outside to inside at radii `R_max, R_max-1, ...`.

    Key insight:
    - The color does not only choose a local geometric change; it changes the effective radius that becomes the sorting key for the final concentric layout.
    """
    bg, objects = _extract_objects(np.asarray(grid, dtype=int))
    return _build_output(bg, objects)


def _sample_count(rng: random.Random) -> int:
    choices = [count for count, _ in COUNT_WEIGHTS]
    weights = [weight for _, weight in COUNT_WEIGHTS]
    return rng.choices(choices, weights=weights, k=1)[0]


def _static_colors(n_static: int, rng: random.Random) -> list[int]:
    if n_static == 0:
        return []
    if n_static == 1:
        return [rng.choice([BLUE, GREEN])]
    colors = [BLUE, GREEN]
    while len(colors) < n_static:
        colors.append(rng.choice([BLUE, GREEN]))
    rng.shuffle(colors)
    return colors


def _planned_specs(rng: random.Random) -> list[dict[str, int]]:
    count = _sample_count(rng)
    swap_top = rng.randint(3, 6)
    chosen_radii = [swap_top - 1, swap_top]
    remaining = [radius for radius in range(2, 7) if radius not in chosen_radii]
    chosen_radii.extend(rng.sample(remaining, count - 2))

    specs: list[dict[str, int]] = [
        {"radius": swap_top, "color": RED},
        {"radius": swap_top - 1, "color": GREY},
    ]

    static_radii = [radius for radius in chosen_radii if radius not in (swap_top - 1, swap_top)]
    for radius, color in zip(static_radii, _static_colors(len(static_radii), rng)):
        specs.append({"radius": radius, "color": color})

    for idx, spec in enumerate(specs):
        spec["id"] = idx
        spec["new_radius"] = _adjusted_radius(spec["color"], spec["radius"])

    assert len({spec["radius"] for spec in specs}) == count
    assert len({spec["new_radius"] for spec in specs}) == count

    initial_order = [spec["id"] for spec in sorted(specs, key=lambda spec: spec["radius"], reverse=True)]
    transformed_order = [spec["id"] for spec in sorted(specs, key=lambda spec: spec["new_radius"], reverse=True)]
    assert initial_order != transformed_order, "generator must include a radius-order reversal"

    rng.shuffle(specs)
    return specs


def _sample_shape(specs: list[dict[str, int]], rng: random.Random) -> tuple[int, int]:
    count = len(specs)
    max_radius = max(spec["radius"] for spec in specs)
    base = max(GRID_BASE_BY_COUNT[count], 2 * max_radius + 5)
    h = rng.randint(base, min(30, base + GRID_JITTER))
    w = rng.randint(base, min(30, base + GRID_JITTER))
    return h, w


def _place_specs(specs: list[dict[str, int]], shape: tuple[int, int], rng: random.Random) -> list[dict[str, int]] | None:
    h, w = shape
    forbidden: set[tuple[int, int]] = set()
    placed: list[dict[str, int]] = []
    for spec in sorted(specs, key=lambda spec: spec["radius"], reverse=True):
        radius = spec["radius"]
        candidates = [
            (r, c)
            for r in range(radius, h - radius)
            for c in range(radius, w - radius)
        ]
        rng.shuffle(candidates)
        chosen = None
        chosen_pixels: set[tuple[int, int]] | None = None
        for center_r, center_c in candidates:
            pixels = _diamond_outline(center_r, center_c, radius)
            if pixels & forbidden:
                continue
            chosen = (center_r, center_c)
            chosen_pixels = pixels
            break
        if chosen is None or chosen_pixels is None:
            return None
        placed.append(
            {
                "id": spec["id"],
                "color": spec["color"],
                "radius": radius,
                "new_radius": spec["new_radius"],
                "center_r": chosen[0],
                "center_c": chosen[1],
            }
        )
        forbidden |= _expanded8(chosen_pixels, shape)
    rng.shuffle(placed)
    return placed


def _render_input(shape: tuple[int, int], placed: list[dict[str, int]]) -> np.ndarray:
    grid = np.full(shape, ORANGE, dtype=int)
    for obj in placed:
        for r, c in _diamond_outline(obj["center_r"], obj["center_c"], obj["radius"]):
            grid[r, c] = int(obj["color"])
    return grid


def _validate_semantics(inp: np.ndarray, planned: list[dict[str, int]]) -> np.ndarray | None:
    try:
        bg, extracted = _extract_objects(inp)
    except AssertionError:
        return None

    expected_signature = sorted((obj["color"], obj["radius"], obj["new_radius"]) for obj in planned)
    actual_signature = sorted((obj["color"], obj["radius"], obj["new_radius"]) for obj in extracted)
    if actual_signature != expected_signature:
        return None

    initial_order = [item[:2] for item in sorted(actual_signature, key=lambda item: item[1], reverse=True)]
    transformed_order = [item[:2] for item in sorted(actual_signature, key=lambda item: item[2], reverse=True)]
    if initial_order == transformed_order:
        return None

    expected = _build_output(bg, extracted)
    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None

    if produced.shape[0] > 30 or produced.shape[1] > 30:
        return None
    if len(np.unique(produced)) < 3:
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        specs = _planned_specs(rng)
        shape = _sample_shape(specs, rng)
        placed = _place_specs(specs, shape, rng)
        if placed is None:
            continue
        candidate = _render_input(shape, placed)
        valid = _validate_semantics(candidate, placed)
        if valid is not None:
            return valid
    raise RuntimeError("generate_input: failed to construct a valid puzzle instance")
