from __future__ import annotations

import itertools
import random
from collections import Counter, deque

import numpy as np


PANEL_SIZE = 5
GRID_SHAPE = (5, 17)
PANEL_STARTS = (0, 6, 12)
SEPARATOR_COLS = (5, 11)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
COLOR_POOL = tuple(range(1, 10))
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r + dr, c + dc) for dr, dc in DIRS4]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _overlay(
    components: list[np.ndarray],
    colors: list[int],
    order: tuple[int, int, int],
    bg: int,
) -> np.ndarray:
    out = np.full((PANEL_SIZE, PANEL_SIZE), int(bg), dtype=int)
    for idx in order:
        coords = components[idx]
        out[coords[:, 0], coords[:, 1]] = int(colors[idx])
    return out


def _parse_grid(grid: np.ndarray) -> tuple[int, list[np.ndarray], list[int], list[int]]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, "expected a 5x17 grid"
    for col in SEPARATOR_COLS:
        assert np.all(g[:, col] == 0), "separator columns must be black"

    panels = [g[:, start : start + PANEL_SIZE] for start in PANEL_STARTS]
    shared_colors = [set(map(int, np.unique(panel))) - {0} for panel in panels]
    bg_candidates = set.intersection(*shared_colors)
    assert len(bg_candidates) == 1, "panels must share one common background color"
    bg = int(next(iter(bg_candidates)))

    survivors: list[np.ndarray] = []
    colors: list[int] = []
    areas: list[int] = []
    for panel in panels:
        panel_colors = set(map(int, np.unique(panel)))
        assert 0 not in panel_colors, "black is reserved for separators"
        fg_colors = sorted(panel_colors - {bg})
        assert len(fg_colors) == 1, "each panel must contain exactly one foreground color"
        color = int(fg_colors[0])
        comps = _components(panel == color)
        assert 2 <= len(comps) <= 3, "each panel must contain two or three 4-connected components"
        comp_areas = [int(comp.shape[0]) for comp in comps]
        largest = max(comp_areas)
        assert comp_areas.count(largest) == 1, "largest component must be unique"
        winner = comps[comp_areas.index(largest)]
        survivors.append(winner)
        colors.append(color)
        areas.append(largest)

    assert len(set(colors)) == 3, "panel foreground colors must be distinct"
    assert len(set(areas)) == 3, "largest component areas must be distinct"
    return bg, survivors, colors, areas


def main(grid: np.ndarray) -> np.ndarray:
    """Overlay the largest component from each panel, using component area to decide the layer order.

    Input structure:
    - The input is a 5x17 grid made of three 5x5 panels separated by black columns.
    - Every panel shares the same non-black background color and contains one distinct foreground color.
    - Inside each panel, that foreground color is split into two or three 4-connected components.

    Rule:
    - In each panel, keep only the unique largest foreground component and discard the smaller distractors.
    - Compare the three surviving components by pixel area.
    - Draw them onto one 5x5 canvas at their original local coordinates, from smallest area first to largest area last.
    - Later layers overwrite earlier ones on overlapping cells.

    Key insight:
    - The overlay order is not determined by left, middle, and right panel position; it is determined only by the surviving components' pixel counts.
    """
    bg, survivors, colors, areas = _parse_grid(grid)
    order = tuple(sorted(range(3), key=lambda idx: areas[idx]))
    return _overlay(survivors, colors, order, bg)


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _random_connected_component(area: int, allowed: np.ndarray) -> np.ndarray | None:
    cells = [(r, c) for r in range(PANEL_SIZE) for c in range(PANEL_SIZE) if allowed[r, c]]
    if len(cells) < area:
        return None
    for _ in range(60):
        start = random.choice(cells)
        chosen = {start}
        frontier = {
            (nr, nc)
            for nr, nc in _neighbors4(*start)
            if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE and allowed[nr, nc]
        }
        while len(chosen) < area and frontier:
            nxt = random.choice(tuple(frontier))
            frontier.remove(nxt)
            chosen.add(nxt)
            for nr, nc in _neighbors4(*nxt):
                if (
                    0 <= nr < PANEL_SIZE
                    and 0 <= nc < PANEL_SIZE
                    and allowed[nr, nc]
                    and (nr, nc) not in chosen
                ):
                    frontier.add((nr, nc))
        if len(chosen) != area:
            continue
        coords = np.array(sorted(chosen), dtype=int)
        r0, c0, r1, c1 = _bbox(coords)
        if area > 1 and (r0 == r1 or c0 == c1):
            continue
        return coords
    return None


def _random_main_component(area: int) -> np.ndarray | None:
    for _ in range(80):
        options = []
        for h in range(2, PANEL_SIZE + 1):
            for w in range(2, PANEL_SIZE + 1):
                fill = area / float(h * w)
                if area <= h * w and 0.4 <= fill <= 0.9:
                    options.append((h, w))
        if not options:
            return None
        h, w = random.choice(options)
        r0 = random.randint(0, PANEL_SIZE - h)
        c0 = random.randint(0, PANEL_SIZE - w)
        allowed = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
        allowed[r0 : r0 + h, c0 : c0 + w] = True
        coords = _random_connected_component(area, allowed)
        if coords is None:
            continue
        return coords
    return None


def _overlap_count(a: np.ndarray, b: np.ndarray) -> int:
    return len({(int(r), int(c)) for r, c in a} & {(int(r), int(c)) for r, c in b})


def _visible_counts(
    components: list[np.ndarray],
    colors: list[int],
    order: tuple[int, int, int],
    bg: int,
) -> dict[int, int]:
    out = _overlay(components, colors, order, bg)
    return {idx: int(np.count_nonzero(out == colors[idx])) for idx in range(3)}


def _semantic_ok(
    components: list[np.ndarray],
    areas: list[int],
    overlap_mode: str,
) -> bool:
    order = tuple(sorted(range(3), key=lambda idx: areas[idx]))
    if order in ((0, 1, 2), (2, 1, 0)):
        return False

    overlaps = {(i, j): _overlap_count(components[i], components[j]) for i, j in itertools.combinations(range(3), 2)}
    overlap_pairs = sum(count > 0 for count in overlaps.values())
    if overlap_mode == "single":
        if overlap_pairs != 1:
            return False
    else:
        if overlap_pairs < 2:
            return False

    if sum(overlaps.values()) > 5:
        return False

    largest_idx = max(range(3), key=lambda idx: areas[idx])
    if not any(
        count > 0 and largest_idx in pair
        for pair, count in overlaps.items()
    ):
        return False

    colors = [1, 2, 3]
    correct = _overlay(components, colors, order, 5)
    if not (7 <= int(np.count_nonzero(correct != 5)) <= 16):
        return False

    visible = _visible_counts(components, colors, order, 5)
    if any(visible[idx] == 0 for idx in range(3)):
        return False

    for perm in itertools.permutations(range(3)):
        if perm == order:
            continue
        if np.array_equal(correct, _overlay(components, colors, perm, 5)):
            return False
    return True


def _choose_main_components() -> tuple[list[np.ndarray], list[int]] | None:
    candidate_areas = [4, 5, 6, 7, 8]
    overlap_mode = random.choice(["single", "double"])
    for _ in range(160):
        areas = random.sample(candidate_areas, 3)
        if tuple(sorted(range(3), key=lambda idx: areas[idx])) in ((0, 1, 2), (2, 1, 0)):
            continue
        components = []
        for area in areas:
            comp = _random_main_component(area)
            if comp is None:
                break
            components.append(comp)
        if len(components) != 3:
            continue
        if _semantic_ok(components, areas, overlap_mode):
            return components, areas
    return None


def _forbidden_mask(existing: list[np.ndarray]) -> np.ndarray:
    forbidden = np.zeros((PANEL_SIZE, PANEL_SIZE), dtype=bool)
    for comp in existing:
        for r, c in comp:
            forbidden[int(r), int(c)] = True
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < PANEL_SIZE and 0 <= nc < PANEL_SIZE:
                    forbidden[nr, nc] = True
    return forbidden


def _build_panel(bg: int, color: int, main_comp: np.ndarray, n_components: int) -> np.ndarray | None:
    max_small = min(3, int(main_comp.shape[0]) - 1)
    if max_small < 1:
        return None

    for _ in range(100):
        comps = [main_comp]
        for _extra in range(n_components - 1):
            size = random.randint(1, max_small)
            allowed = ~_forbidden_mask(comps)
            extra = _random_connected_component(size, allowed)
            if extra is None:
                break
            comps.append(extra)
        if len(comps) != n_components:
            continue

        panel = np.full((PANEL_SIZE, PANEL_SIZE), int(bg), dtype=int)
        for comp in comps:
            panel[comp[:, 0], comp[:, 1]] = int(color)

        parsed = _components(panel == color)
        areas = sorted((int(comp.shape[0]) for comp in parsed), reverse=True)
        if len(parsed) != n_components:
            continue
        if areas[0] != int(main_comp.shape[0]) or areas.count(areas[0]) != 1:
            continue
        return panel
    return None


def _assemble_grid(bg: int, panels: list[np.ndarray]) -> np.ndarray:
    grid = np.full(GRID_SHAPE, int(bg), dtype=int)
    for col in SEPARATOR_COLS:
        grid[:, col] = 0
    for start, panel in zip(PANEL_STARTS, panels):
        grid[:, start : start + PANEL_SIZE] = panel
    return grid


def _generator_semantic_check(
    grid: np.ndarray,
    expected_components: list[np.ndarray],
    expected_colors: list[int],
    expected_areas: list[int],
    bg: int,
) -> bool:
    try:
        parsed_bg, survivors, colors, areas = _parse_grid(grid)
    except AssertionError:
        return False
    if parsed_bg != bg or colors != expected_colors or areas != expected_areas:
        return False
    for got, want in zip(survivors, expected_components):
        if {tuple(map(int, p)) for p in got} != {tuple(map(int, p)) for p in want}:
            return False

    expected_order = tuple(sorted(range(3), key=lambda idx: expected_areas[idx]))
    expected_out = _overlay(expected_components, expected_colors, expected_order, bg)
    if not np.array_equal(main(grid), expected_out):
        return False

    left_to_right = _overlay(expected_components, expected_colors, (0, 1, 2), bg)
    if np.array_equal(left_to_right, expected_out):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        chosen = _choose_main_components()
        if chosen is None:
            continue
        main_components, areas = chosen

        fg_colors = random.sample(COLOR_POOL, 3)
        bg_choices = [color for color in COLOR_POOL if color not in fg_colors]
        bg = random.choice(bg_choices)

        panel_component_counts = [random.choice([2, 3]) for _ in range(3)]
        panels: list[np.ndarray] = []
        for color, main_comp, n_components in zip(fg_colors, main_components, panel_component_counts):
            panel = _build_panel(bg, color, main_comp, n_components)
            if panel is None:
                break
            panels.append(panel)
        if len(panels) != 3:
            continue

        grid = _assemble_grid(bg, panels)
        if _generator_semantic_check(grid, main_components, fg_colors, areas, bg):
            return grid
    raise RuntimeError("failed to generate a valid input")
