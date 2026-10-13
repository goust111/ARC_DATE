import random
from typing import Iterable

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8  # ARC has no white; use cyan as the visible substitute for triple-mix white.
MAROON = 9

PRIMARY_COLORS = (RED, BLUE, YELLOW)
PAIR_MIXES = {
    frozenset((RED, BLUE)): MAGENTA,
    frozenset((RED, YELLOW)): ORANGE,
    frozenset((BLUE, YELLOW)): GREEN,
}
LEGEND_ROWS = (
    (RED, BLUE, MAGENTA),
    (RED, YELLOW, ORANGE),
    (BLUE, YELLOW, GREEN),
)
EDGES = ("top", "bottom", "left", "right")

GEN_RETRIES = 400
COMPONENT_RETRIES = 80


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(cells: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    coords = list(cells)
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_legend(grid: np.ndarray) -> tuple[tuple[int, int, int, int], dict[frozenset[int], int], set[int]]:
    maroon_comps = _components(grid == MAROON)
    candidates: list[tuple[int, int, int, int]] = []
    for comp in maroon_comps:
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        ok = True
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                border = r in (r0, r1) or c in (c0, c1)
                if border:
                    if int(grid[r, c]) != MAROON:
                        ok = False
                        break
                else:
                    if int(grid[r, c]) == MAROON:
                        ok = False
                        break
            if not ok:
                break
        if ok:
            candidates.append((r0, c0, r1, c1))

    assert len(candidates) == 1, "expected exactly one maroon legend frame"
    r0, c0, r1, c1 = candidates[0]
    inner = grid[r0 + 1 : r1, c0 + 1 : c1]
    assert inner.shape == (3, 3), "legend interior must encode three 3-pixel rows"

    pair_map: dict[frozenset[int], int] = {}
    primaries: set[int] = set()
    for row in inner:
        a, b, result = map(int, row)
        pair_map[frozenset((a, b))] = result
        primaries.add(a)
        primaries.add(b)

    assert len(pair_map) == 3, "legend must contain three pairwise mixes"
    assert len(primaries) == 3, "legend must expose exactly three primary colors"
    return candidates[0], pair_map, primaries


def _resolve_color(source_colors: set[int], pair_map: dict[frozenset[int], int]) -> int | None:
    if not source_colors:
        return None
    if len(source_colors) == 1:
        return next(iter(source_colors))
    if len(source_colors) == 2:
        return pair_map[frozenset(source_colors)]
    assert len(source_colors) == 3
    return CYAN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Color a grey pipe network by propagating primary-color sources through each connected pipe component.

    The maroon-framed 3x3 legend gives the pairwise mixing table. ARC has no white, so this task uses
    cyan (8) as the visible substitute for the "three colors at once" result.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    (lr0, lc0, lr1, lc1), pair_map, primaries = _find_legend(g)
    legend_mask = np.zeros_like(g, dtype=bool)
    legend_mask[lr0 : lr1 + 1, lc0 : lc1 + 1] = True

    pipe_mask = (g == GRAY) & ~legend_mask
    pipe_components = _components(pipe_mask)
    cell_to_comp: dict[tuple[int, int], int] = {}
    for idx, comp in enumerate(pipe_components):
        for cell in comp:
            cell_to_comp[cell] = idx

    colors_by_comp = [set() for _ in pipe_components]
    h, w = g.shape
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in primaries or legend_mask[r, c]:
                continue
            touched = {
                cell_to_comp[(nr, nc)]
                for nr, nc in _neighbors4(r, c)
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) in cell_to_comp
            }
            if not touched:
                continue
            assert len(touched) == 1, "each entry marker must touch exactly one pipe component"
            colors_by_comp[next(iter(touched))].add(color)

    out = g.copy()
    for idx, comp in enumerate(pipe_components):
        fill = _resolve_color(colors_by_comp[idx], pair_map)
        if fill is None:
            continue
        for r, c in comp:
            out[r, c] = fill
    return out


def _edge_terminal(shape: tuple[int, int], edge: str) -> tuple[int, int]:
    h, w = shape
    if edge == "top":
        return 0, random.randint(1, w - 2)
    if edge == "bottom":
        return h - 1, random.randint(1, w - 2)
    if edge == "left":
        return random.randint(1, h - 2), 0
    assert edge == "right"
    return random.randint(1, h - 2), w - 1


def _marker_for_terminal(terminal: tuple[int, int], edge: str) -> tuple[int, int]:
    r, c = terminal
    if edge in ("top", "bottom"):
        options = [(r, c - 1), (r, c + 1)]
    else:
        options = [(r - 1, c), (r + 1, c)]
    random.shuffle(options)
    return options[0]


def _route_to_hub(shape: tuple[int, int], edge: str, terminal: tuple[int, int], hub: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    hr, hc = hub
    tr, tc = terminal
    cells: set[tuple[int, int]] = {terminal}
    if edge == "top":
        for r in range(0, hr + 1):
            cells.add((r, tc))
        for c in range(min(tc, hc), max(tc, hc) + 1):
            cells.add((hr, c))
    elif edge == "bottom":
        for r in range(hr, h):
            cells.add((r, tc))
        for c in range(min(tc, hc), max(tc, hc) + 1):
            cells.add((hr, c))
    elif edge == "left":
        for c in range(0, hc + 1):
            cells.add((tr, c))
        for r in range(min(tr, hr), max(tr, hr) + 1):
            cells.add((r, hc))
    else:
        for c in range(hc, w):
            cells.add((tr, c))
        for r in range(min(tr, hr), max(tr, hr) + 1):
            cells.add((r, hc))
    return cells


def _dilate(cells: set[tuple[int, int]], shape: tuple[int, int], radius: int = 1) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _pipe_degrees(cells: set[tuple[int, int]]) -> dict[tuple[int, int], int]:
    degrees: dict[tuple[int, int], int] = {}
    for r, c in cells:
        deg = 0
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells:
                deg += 1
        degrees[(r, c)] = deg
    return degrees


def _valid_component(
    shape: tuple[int, int],
    pipe_cells: set[tuple[int, int]],
    markers: dict[tuple[int, int], int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    all_cells = set(pipe_cells) | set(markers)
    if not pipe_cells or _components(_mask_from_cells(shape, pipe_cells)).__len__() != 1:
        return False
    if all_cells & occupied:
        return False
    if _dilate(all_cells, shape, radius=1) & blocked:
        return False

    marker_cells = set(markers)
    if pipe_cells & marker_cells:
        return False

    degrees = _pipe_degrees(pipe_cells)
    for marker in markers:
        mr, mc = marker
        if mr not in (0, shape[0] - 1) and mc not in (0, shape[1] - 1):
            return False
        adjacent = [(nr, nc) for nr, nc in _neighbors4(mr, mc) if (nr, nc) in pipe_cells]
        if len(adjacent) != 1:
            return False
        terminal = adjacent[0]
        tr, tc = terminal
        if tr not in (0, shape[0] - 1) and tc not in (0, shape[1] - 1):
            return False
        if degrees[terminal] != 1:
            return False
    return True


def _mask_from_cells(shape: tuple[int, int], cells: Iterable[tuple[int, int]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _add_spur(
    shape: tuple[int, int],
    pipe_cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> None:
    if random.random() >= 0.45:
        return
    candidates = [cell for cell, deg in _pipe_degrees(pipe_cells).items() if deg >= 1 and cell[0] not in (0, shape[0] - 1) and cell[1] not in (0, shape[1] - 1)]
    if not candidates:
        return
    base = random.choice(candidates)
    directions = list(_neighbors4(0, 0))
    random.shuffle(directions)
    for dr, dc in directions:
        new_cells: list[tuple[int, int]] = []
        length = random.randint(1, 4)
        br, bc = base
        ok = True
        for step in range(1, length + 1):
            nr, nc = br + dr * step, bc + dc * step
            if not (0 < nr < shape[0] - 1 and 0 < nc < shape[1] - 1):
                ok = False
                break
            if (nr, nc) in pipe_cells or (nr, nc) in occupied:
                ok = False
                break
            new_cells.append((nr, nc))
        if not ok:
            continue
        if _dilate(set(new_cells), shape, radius=1) & blocked:
            continue
        pipe_cells.update(new_cells)
        return


def _sample_source_set() -> list[int]:
    kind = random.choice(("single", "pair", "triple"))
    if kind == "single":
        return [random.choice(PRIMARY_COLORS)]
    if kind == "pair":
        return list(random.choice(tuple(PAIR_MIXES)))
    return list(PRIMARY_COLORS)


def _place_active_component(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    source_colors: list[int],
) -> tuple[set[tuple[int, int]], dict[tuple[int, int], int]] | None:
    h, w = shape
    k = len(source_colors)
    for _ in range(COMPONENT_RETRIES):
        edges = random.sample(EDGES, k)
        hub = (random.randint(3, h - 4), random.randint(3, w - 4))
        pipe_cells: set[tuple[int, int]] = set()
        markers: dict[tuple[int, int], int] = {}
        used_marker_cells: set[tuple[int, int]] = set()

        valid = True
        for color, edge in zip(source_colors, edges):
            terminal = _edge_terminal(shape, edge)
            marker = _marker_for_terminal(terminal, edge)
            if marker in used_marker_cells or marker in pipe_cells or marker in occupied:
                valid = False
                break
            path = _route_to_hub(shape, edge, terminal, hub)
            pipe_cells |= path
            used_marker_cells.add(marker)
            markers[marker] = color
        if not valid:
            continue

        for _ in range(random.randint(0, 2)):
            _add_spur(shape, pipe_cells, occupied, blocked)

        if _valid_component(shape, pipe_cells, markers, occupied, blocked):
            return pipe_cells, markers
    return None


def _place_dead_component(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    h, w = shape
    for _ in range(COMPONENT_RETRIES):
        base = (random.randint(2, h - 3), random.randint(2, w - 3))
        pipe_cells: set[tuple[int, int]] = {base}
        arms = random.randint(1, 3)
        directions = random.sample(((1, 0), (-1, 0), (0, 1), (0, -1)), arms)
        valid = True
        for dr, dc in directions:
            length = random.randint(2, 5)
            for step in range(1, length + 1):
                nr, nc = base[0] + dr * step, base[1] + dc * step
                if not (1 <= nr < h - 1 and 1 <= nc < w - 1):
                    valid = False
                    break
                pipe_cells.add((nr, nc))
            if not valid:
                break
        if not valid:
            continue
        if pipe_cells & occupied:
            continue
        if _dilate(pipe_cells, shape, radius=1) & blocked:
            continue
        if _components(_mask_from_cells(shape, pipe_cells)).__len__() != 1:
            continue
        return pipe_cells
    return None


def _place_legend(
    grid: np.ndarray,
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    h, w = grid.shape
    rows = list(LEGEND_ROWS)
    random.shuffle(rows)
    legend = np.full((5, 5), BLACK, dtype=int)
    legend[0, :] = MAROON
    legend[-1, :] = MAROON
    legend[:, 0] = MAROON
    legend[:, -1] = MAROON
    for i, row in enumerate(rows, start=1):
        legend[i, 1:4] = np.array(row, dtype=int)

    for _ in range(COMPONENT_RETRIES):
        r0 = random.randint(1, h - 6)
        c0 = random.randint(1, w - 6)
        cells = {(r0 + r, c0 + c) for r in range(5) for c in range(5)}
        if cells & occupied:
            continue
        if _dilate(cells, (h, w), radius=1) & blocked:
            continue
        for r in range(5):
            for c in range(5):
                grid[r0 + r, c0 + c] = int(legend[r, c])
        occupied |= cells
        blocked |= _dilate(cells, (h, w), radius=1)
        return True
    return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        h = random.randint(18, 28)
        w = random.randint(18, 28)
        grid = np.full((h, w), BLACK, dtype=int)
        occupied: set[tuple[int, int]] = set()
        blocked: set[tuple[int, int]] = set()

        if not _place_legend(grid, occupied, blocked):
            continue

        n_active = random.choice((1, 2, 3))
        used_sets: set[tuple[int, ...]] = set()
        active_ok = True
        for _comp_idx in range(n_active):
            for _try in range(COMPONENT_RETRIES):
                source_colors = _sample_source_set()
                key = tuple(sorted(source_colors))
                if n_active > 1 and key in used_sets and random.random() < 0.8:
                    continue
                placed = _place_active_component((h, w), occupied, blocked, source_colors)
                if placed is None:
                    continue
                pipe_cells, markers = placed
                for r, c in pipe_cells:
                    grid[r, c] = GRAY
                for (r, c), color in markers.items():
                    grid[r, c] = color
                all_cells = set(pipe_cells) | set(markers)
                occupied |= all_cells
                blocked |= _dilate(all_cells, (h, w), radius=1)
                used_sets.add(key)
                break
            else:
                active_ok = False
                break
        if not active_ok:
            continue

        for _ in range(random.choice((0, 1, 1, 2))):
            dead = _place_dead_component((h, w), occupied, blocked)
            if dead is None:
                continue
            for r, c in dead:
                grid[r, c] = GRAY
            occupied |= dead
            blocked |= _dilate(dead, (h, w), radius=1)

        try:
            out = main(grid)
        except AssertionError:
            continue
        if np.array_equal(out, grid):
            continue
        return grid

    raise RuntimeError("failed to generate a valid pipe-mixing instance")
