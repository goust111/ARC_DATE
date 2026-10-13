import random
from collections import deque

import numpy as np


BLACK = 0
PALETTE = (1, 2, 3, 4)
GRID_SHAPE = (10, 10)

CORNER_POSITIONS = {
    "top_left": (0, 0),
    "top_right": (0, GRID_SHAPE[1] - 1),
    "bottom_left": (GRID_SHAPE[0] - 1, 0),
    "bottom_right": (GRID_SHAPE[0] - 1, GRID_SHAPE[1] - 1),
}

CORNER_TEMPLATES = {
    "top_left": ((1, 1), (1, 2), (2, 1), (2, 2)),
    "top_right": ((1, GRID_SHAPE[1] - 3), (1, GRID_SHAPE[1] - 2), (2, GRID_SHAPE[1] - 3), (2, GRID_SHAPE[1] - 2)),
    "bottom_left": ((GRID_SHAPE[0] - 2, 0), (GRID_SHAPE[0] - 3, 0), (GRID_SHAPE[0] - 2, 1)),
    "bottom_right": ((GRID_SHAPE[0] - 2, GRID_SHAPE[1] - 1), (GRID_SHAPE[0] - 3, GRID_SHAPE[1] - 1), (GRID_SHAPE[0] - 2, GRID_SHAPE[1] - 2)),
}

CORNER_ORDER = tuple(CORNER_POSITIONS)
GENERATED_CASES = (
    ("top_left", "bottom_right"),
    ("top_left", "top_right", "bottom_left"),
    ("top_left", "top_right", "bottom_right"),
    ("top_right", "bottom_left"),
    ("top_left", "bottom_left", "bottom_right"),
    ("top_right", "bottom_left", "bottom_right"),
)

_GEN_CALL_COUNT = 0
_STATIC_CHECK_DONE = False


def _has_required_diagonal_pair(active_names: tuple[str, ...]) -> bool:
    active = set(active_names)
    return {"top_left", "bottom_right"} <= active or {"top_right", "bottom_left"} <= active


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(comp)
    return components


def _extract_markers(grid: np.ndarray) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    assert g.shape == GRID_SHAPE, f"expected a {GRID_SHAPE[0]}x{GRID_SHAPE[1]} grid"
    active: dict[str, int] = {}
    corner_cells = set(CORNER_POSITIONS.values())

    for r, c in np.argwhere(g != BLACK):
        cell = (int(r), int(c))
        assert cell in corner_cells, "only corner marker pixels may be non-black in the input"
        color = int(g[cell])
        assert color in PALETTE, "corner markers must use colors from {1,2,3,4}"
        for name, pos in CORNER_POSITIONS.items():
            if pos == cell:
                active[name] = color
                break

    assert 2 <= len(active) <= 4, "expected 2 to 4 active corner markers"
    assert len(set(active.values())) == len(active), "corner marker colors must be distinct"
    assert _has_required_diagonal_pair(tuple(active)), "expected at least one active diagonal corner pair"
    return active


def _template_distance(cell: tuple[int, int], template_cells: tuple[tuple[int, int], ...]) -> int:
    r, c = cell
    return min(abs(r - tr) + abs(c - tc) for tr, tc in template_cells)


def _summed_distances(
    component: list[tuple[int, int]],
    template_cells_by_name: dict[str, tuple[tuple[int, int], ...]],
) -> dict[str, int]:
    return {
        name: sum(_template_distance(cell, template_cells) for cell in component)
        for name, template_cells in template_cells_by_name.items()
    }


def _build_template_grid(active: dict[str, int]) -> tuple[np.ndarray, dict[str, tuple[tuple[int, int], ...]]]:
    stamped = np.zeros(GRID_SHAPE, dtype=int)
    template_cells_by_name: dict[str, tuple[tuple[int, int], ...]] = {}
    for name in CORNER_ORDER:
        color = active.get(name)
        if color is None:
            continue
        cells = tuple(CORNER_TEMPLATES[name])
        template_cells_by_name[name] = cells
        for r, c in cells:
            stamped[r, c] = color
    return stamped, template_cells_by_name


def _solve_details(grid: np.ndarray) -> dict:
    active = _extract_markers(grid)
    stamped, template_cells_by_name = _build_template_grid(active)
    black_components = _components4(stamped == BLACK)

    out = stamped.copy()
    region_owners: list[str] = []
    distance_summaries: list[dict[str, int]] = []
    for component in black_components:
        distances = _summed_distances(component, template_cells_by_name)
        best = min(distances.values())
        winners = [name for name, value in distances.items() if value == best]
        assert len(winners) == 1, "each black component must have a unique closest template"
        owner = winners[0]
        region_owners.append(owner)
        distance_summaries.append(distances)
        fill_color = active[owner]
        for r, c in component:
            out[r, c] = fill_color

    return {
        "active": active,
        "stamped": stamped,
        "templates": template_cells_by_name,
        "components": black_components,
        "owners": region_owners,
        "distances": distance_summaries,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Replace corner markers with corner-specific templates, then recolor each black flood region by its nearest template color.

    Input structure:
    - The grid is 10x10 with black background and 2 to 4 distinct colored single-pixel markers placed only at corners.
    - Marker colors come from {blue(1), red(2), green(3), yellow(4)} and at least one diagonal corner pair is active.
    - No non-corner interior foreground pixels appear in the input.

    Rule:
    - Erase each corner marker back to black, then stamp the fixed template for that corner in the marker's original color.
    - Top corners stamp inward 2x2 blocks; bottom corners stamp inward 3-pixel bent templates mirrored left/right.
    - Treat template pixels as blockers, split the remaining black pixels into 4-connected components, and recolor each component with the color of the template whose template-pixel Manhattan-distance sum over that component is uniquely minimal.

    Key insight:
    - Corner position chooses template shape, and the template shape changes both blocker geometry and distance totals, so marker color alone is insufficient to predict the filled output.
    """
    return _solve_details(grid)["output"]


def _cellwise_unique_labels(
    component: list[tuple[int, int]],
    template_cells_by_name: dict[str, tuple[tuple[int, int], ...]],
) -> set[str]:
    labels: set[str] = set()
    for cell in component:
        distances = {
            name: _template_distance(cell, cells)
            for name, cells in template_cells_by_name.items()
        }
        best = min(distances.values())
        winners = [name for name, value in distances.items() if value == best]
        if len(winners) == 1:
            labels.add(winners[0])
        else:
            labels.add("tie")
    return labels


def _point_template_owner(active_names: tuple[str, ...]) -> list[str]:
    point_templates = {name: (CORNER_POSITIONS[name],) for name in active_names}
    point_blocked = {CORNER_POSITIONS[name] for name in active_names}
    point_components = _components4(
        np.array(
            [[(r, c) not in point_blocked for c in range(GRID_SHAPE[1])] for r in range(GRID_SHAPE[0])],
            dtype=bool,
        )
    )
    owners: list[str] = []
    for component in point_components:
        distances = _summed_distances(component, point_templates)
        best = min(distances.values())
        winners = [name for name, value in distances.items() if value == best]
        owners.append("ambiguous" if len(winners) != 1 else winners[0])
    return owners


def _validate_generated_input(grid: np.ndarray) -> bool:
    try:
        details = _solve_details(grid)
    except AssertionError:
        return False

    active_names = tuple(details["active"])
    if active_names not in GENERATED_CASES:
        return False

    if len(details["components"]) != 1:
        return False

    owner_names = set(details["owners"])
    if len(owner_names) != 1:
        return False

    component = details["components"][0]
    if len(_cellwise_unique_labels(component, details["templates"])) < 2:
        return False

    if _point_template_owner(active_names) == details["owners"]:
        return False

    if np.array_equal(details["stamped"], details["output"]):
        return False

    if np.array_equal(main(grid), grid):
        return False

    return True


def _all_prompt_legal_cases() -> tuple[tuple[str, ...], ...]:
    cases: list[tuple[str, ...]] = []
    for size in (2, 3, 4):
        indices = range(len(CORNER_ORDER))
        for picks in np.array(np.meshgrid(*([indices] * size))).T.reshape(-1, size):
            if len(set(map(int, picks))) != size:
                continue
            case = tuple(CORNER_ORDER[i] for i in sorted(set(map(int, picks))))
            if len(case) != size:
                continue
            if case in cases:
                continue
            if _has_required_diagonal_pair(case):
                cases.append(case)
    return tuple(cases)


def _make_input(case: tuple[str, ...], colors: tuple[int, ...]) -> np.ndarray:
    grid = np.zeros(GRID_SHAPE, dtype=int)
    for name, color in zip(case, colors):
        r, c = CORNER_POSITIONS[name]
        grid[r, c] = int(color)
    return grid


def _run_static_geometry_checks() -> None:
    all_cases = _all_prompt_legal_cases()
    assert set(GENERATED_CASES) < set(all_cases), "generator cases should be a strict, solvable subset of the prompt cases"

    for case in all_cases:
        grid = _make_input(case, tuple(PALETTE[: len(case)]))
        active = _extract_markers(grid)
        stamped, template_cells = _build_template_grid(active)
        components = _components4(stamped == BLACK)
        assert len(components) == 1, "with the stated 10x10 coordinates, legal corner subsets always leave one black 4-connected domain"

        component = components[0]
        labels = _cellwise_unique_labels(component, template_cells)
        if case in GENERATED_CASES:
            distances = _summed_distances(component, template_cells)
            best = min(distances.values())
            winners = [name for name, value in distances.items() if value == best]
            assert len(winners) == 1, "generated cases must remain uniquely solvable"
            assert len(labels) >= 2, "generated cases must make region-level aggregation necessary"
            assert _point_template_owner(case) != winners, "template shape must matter versus using corner points alone"
        else:
            assert len(case) == 4, "only the four-marker prompt case should be excluded"
            distances = _summed_distances(component, template_cells)
            best = min(distances.values())
            winners = [name for name, value in distances.items() if value == best]
            assert len(winners) != 1, "the excluded four-marker case should be ambiguous under the stated rule"


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT, _STATIC_CHECK_DONE

    if not _STATIC_CHECK_DONE:
        _run_static_geometry_checks()
        _STATIC_CHECK_DONE = True

    for _ in range(300):
        case = GENERATED_CASES[_GEN_CALL_COUNT % len(GENERATED_CASES)]
        _GEN_CALL_COUNT += 1
        colors = tuple(random.sample(PALETTE, len(case)))
        grid = _make_input(case, colors)
        if _validate_generated_input(grid):
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 8")
