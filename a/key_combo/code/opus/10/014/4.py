from __future__ import annotations

from functools import lru_cache
import random

import numpy as np


BG = 0
BODY = 3
ORANGE = 7
MARKER_COLORS = (1, 2, 4, 6, 8, 9)
GEN_RETRIES = 300
COMPONENT_RETRIES = 120


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _support_cells(comp_mask: np.ndarray) -> set[tuple[int, int]]:
    h, w = comp_mask.shape
    support: set[tuple[int, int]] = set()
    rows, cols = np.where(comp_mask)
    for r0, c0 in zip(rows, cols):
        r, c = int(r0), int(c0)
        if r == 0 or c == 0 or r == h - 1 or c == w - 1:
            continue
        if comp_mask[r - 1 : r + 2, c - 1 : c + 2].all():
            support.add((r, c))
    return support


def _can_tile_with_2x2(cells: set[tuple[int, int]]) -> bool:
    # Exact-cover check over translated 2x2 squares keeps the rule faithful to the puzzle semantics.
    normalized = frozenset((int(r), int(c)) for r, c in cells)
    if not normalized or len(normalized) % 4 != 0:
        return False

    candidates: list[frozenset[tuple[int, int]]] = []
    seen: set[frozenset[tuple[int, int]]] = set()
    for r, c in normalized:
        for r0 in (r - 1, r):
            for c0 in (c - 1, c):
                block = frozenset(
                    ((r0, c0), (r0 + 1, c0), (r0, c0 + 1), (r0 + 1, c0 + 1))
                )
                if block <= normalized and block not in seen:
                    seen.add(block)
                    candidates.append(block)
    if not candidates:
        return False

    by_cell: dict[tuple[int, int], list[int]] = {cell: [] for cell in normalized}
    for idx, block in enumerate(candidates):
        for cell in block:
            by_cell[cell].append(idx)

    @lru_cache(maxsize=None)
    def search(remaining: frozenset[tuple[int, int]]) -> bool:
        if not remaining:
            return True
        pivot = min(
            remaining,
            key=lambda cell: sum(1 for idx in by_cell[cell] if candidates[idx] <= remaining),
        )
        options = [idx for idx in by_cell[pivot] if candidates[idx] <= remaining]
        if not options:
            return False
        for idx in options:
            if search(remaining - candidates[idx]):
                return True
        return False

    return search(normalized)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each component's solid interior by whether that interior can be tiled by 2x2 squares.

    Input structure:
    - Black background surrounds 3-5 disjoint 4-connected green components.
    - Each green component contains exactly one singleton marker pixel of a distinct non-green color.
    - Every marker sits inside the component's solid interior region, while any corridors remain too thin to create interior support.

    Rule:
    - For each non-background 4-connected component, treat the marker cell as part of the same component and collect every cell whose full 3x3 neighborhood stays inside that component.
    - Test whether that supported interior set can be covered exactly by non-overlapping translated 2x2 squares.
    - If it can, recolor the green supported cells to the component's marker color; otherwise recolor those green supported cells orange.
    - Leave corridors, outer borders, the marker itself, and the background unchanged.

    Key insight:
    - The decision depends on the one-pixel-eroded interior support shape, not on the visible outer blob alone: two blobs can share the same apparent envelope while an interior corner notch breaks 2x2 tileability.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for coords in _components(g != BG):
        values = {int(g[r, c]) for r, c in coords}
        assert BODY in values, "each component must contain green body cells"

        marker_cells = [(r, c) for r, c in coords if int(g[r, c]) != BODY]
        assert len(marker_cells) == 1, "each component must contain exactly one marker"
        marker_r, marker_c = marker_cells[0]
        marker_color = int(g[marker_r, marker_c])

        comp_mask = np.zeros(g.shape, dtype=bool)
        for r, c in coords:
            comp_mask[r, c] = True

        support = _support_cells(comp_mask)
        assert support and (marker_r, marker_c) in support, "marker must lie in the supported interior"
        fill_color = marker_color if _can_tile_with_2x2(support) else ORANGE

        for r, c in support:
            if (r, c) != (marker_r, marker_c) and int(g[r, c]) == BODY:
                out[r, c] = fill_color

    return out


def _rect_cells(height: int, width: int) -> set[tuple[int, int]]:
    return {(r, c) for r in range(height) for c in range(width)}


def _make_support_shape(kind: str) -> tuple[set[tuple[int, int]], bool]:
    if kind == "tile4":
        cells = _rect_cells(4, 4)
    elif kind == "tile_rect":
        cells = _rect_cells(*random.choice(((2, 2), (2, 4), (4, 2), (4, 4))))
    elif kind == "odd_rect":
        cells = _rect_cells(*random.choice(((3, 3), (3, 4), (4, 3), (5, 3), (3, 5))))
    elif kind == "notch4":
        cells = _rect_cells(4, 4)
        cells.remove(random.choice(((0, 0), (0, 3), (3, 0), (3, 3))))
    else:
        raise ValueError(f"unknown support kind: {kind}")

    tileable = _can_tile_with_2x2(cells)
    return cells, tileable


def _dilate8(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                out.add((r + dr, c + dc))
    return out


def _normalize_cells(cells: set[tuple[int, int]]) -> tuple[set[tuple[int, int]], int, int]:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    normalized = {(r - min_r, c - min_c) for r, c in cells}
    return normalized, min_r, min_c


def _perpendicular_dirs(direction: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
    dr, dc = direction
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _boundary_starts(cells: set[tuple[int, int]], direction: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = direction
    starts = [(r, c) for r, c in cells if (r + dr, c + dc) not in cells]
    random.shuffle(starts)
    return starts


def _attach_corridor(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    random.shuffle(directions)
    for direction in directions:
        starts = _boundary_starts(cells, direction)
        for start in starts[:12]:
            new_cells: set[tuple[int, int]] = set()
            pos = start
            ok = True
            for _ in range(random.randint(1, 3)):
                pos = (pos[0] + direction[0], pos[1] + direction[1])
                if pos in cells or pos in new_cells:
                    ok = False
                    break
                new_cells.add(pos)
            if not ok:
                continue

            if random.random() < 0.65:
                turn = random.choice(_perpendicular_dirs(direction))
                for _ in range(random.randint(1, 2)):
                    pos = (pos[0] + turn[0], pos[1] + turn[1])
                    if pos in cells or pos in new_cells:
                        ok = False
                        break
                    new_cells.add(pos)
            if ok:
                return cells | new_cells
    return cells


def _render_patch(cells: set[tuple[int, int]], marker: tuple[int, int], marker_color: int) -> np.ndarray:
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    patch = np.zeros((max_r + 1, max_c + 1), dtype=int)
    for r, c in cells:
        patch[r, c] = BODY
    patch[marker] = marker_color
    return patch


def _patch_matches_design(
    patch: np.ndarray,
    expected_support: set[tuple[int, int]],
    marker: tuple[int, int],
    expected_tileable: bool,
) -> bool:
    comps = _components(patch != BG)
    if len(comps) != 1:
        return False
    support = _support_cells(patch != BG)
    if support != expected_support or marker not in support:
        return False
    marker_cells = [(r, c) for r, c in comps[0] if int(patch[r, c]) != BODY]
    if marker_cells != [marker]:
        return False
    return _can_tile_with_2x2(support) == expected_tileable


def _build_component_patch(kind: str, marker_color: int) -> np.ndarray:
    for _ in range(COMPONENT_RETRIES):
        support, tileable = _make_support_shape(kind)
        marker = random.choice(tuple(support))

        cells = _dilate8(support)
        added_corridor = False
        for _ in range(random.randint(1, 2)):
            grown = _attach_corridor(cells)
            added_corridor |= len(grown) > len(cells)
            cells = grown
        if not added_corridor:
            continue

        norm_cells, min_r, min_c = _normalize_cells(cells)
        norm_support = {(r - min_r, c - min_c) for r, c in support}
        norm_marker = (marker[0] - min_r, marker[1] - min_c)
        patch = _render_patch(norm_cells, norm_marker, marker_color)
        if _patch_matches_design(patch, norm_support, norm_marker, tileable):
            return patch
    raise RuntimeError(f"failed to build component patch for kind={kind}")


def _choose_component_plan() -> list[str]:
    count = random.choices((3, 4, 5), weights=(4, 4, 2), k=1)[0]
    plan = ["tile4", "notch4", "odd_rect"]
    while len(plan) < count:
        plan.append(random.choice(("tile_rect", "odd_rect", "notch4")))
    random.shuffle(plan)
    return plan


def _choose_grid_shape(component_count: int) -> tuple[int, int]:
    base = {3: 20, 4: 22, 5: 24}[component_count]
    limit = min(base + 4, 30)
    return random.randint(base, limit), random.randint(base, limit)


def _can_place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    if top + ph > h or left + pw > w:
        return False
    rows, cols = np.where(patch != BG)
    for r0, c0 in zip(rows, cols):
        gr = top + int(r0)
        gc = left + int(c0)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr, cc = gr + dr, gc + dc
                if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) != BG:
                    return False
    return True


def _paste_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    rows, cols = np.where(patch != BG)
    for r0, c0 in zip(rows, cols):
        grid[top + int(r0), left + int(c0)] = int(patch[int(r0), int(c0)])


def _place_patch_random(grid: np.ndarray, patch: np.ndarray) -> bool:
    h, w = grid.shape
    ph, pw = patch.shape
    positions = [(r, c) for r in range(h - ph + 1) for c in range(w - pw + 1)]
    random.shuffle(positions)
    for top, left in positions:
        if _can_place_patch(grid, patch, top, left):
            _paste_patch(grid, patch, top, left)
            return True
    return False


def _bbox_dims(cells: set[tuple[int, int]]) -> tuple[int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return max(rs) - min(rs) + 1, max(cs) - min(cs) + 1


def _validate_generated_grid(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    comps = _components(grid != BG)
    if not (3 <= len(comps) <= 5):
        return False

    marker_colors_seen: set[int] = set()
    support_box_outcomes: dict[tuple[int, int], set[bool]] = {}
    saw_tileable = False
    saw_nontileable = False
    saw_marker_recolor = False

    for coords in comps:
        marker_cells = [(r, c) for r, c in coords if int(grid[r, c]) != BODY]
        if len(marker_cells) != 1:
            return False
        marker_r, marker_c = marker_cells[0]
        marker_color = int(grid[marker_r, marker_c])
        if marker_color in marker_colors_seen or marker_color in (BG, BODY, ORANGE):
            return False
        marker_colors_seen.add(marker_color)

        comp_mask = np.zeros(grid.shape, dtype=bool)
        for r, c in coords:
            comp_mask[r, c] = True
        support = _support_cells(comp_mask)
        if not support or (marker_r, marker_c) not in support:
            return False

        tileable = _can_tile_with_2x2(support)
        saw_tileable |= tileable
        saw_nontileable |= not tileable
        support_box_outcomes.setdefault(_bbox_dims(support), set()).add(tileable)
        if tileable and any((r, c) != (marker_r, marker_c) for r, c in support):
            saw_marker_recolor = True

    if not (saw_tileable and saw_nontileable and saw_marker_recolor):
        return False
    if not any(outcomes == {False, True} for outcomes in support_box_outcomes.values()):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if not np.any(out == ORANGE):
        return False

    marker_mask = (grid != BG) & (grid != BODY)
    if not np.array_equal(out[marker_mask], grid[marker_mask]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        plan = _choose_component_plan()
        marker_colors = random.sample(MARKER_COLORS, len(plan))
        try:
            patches = [
                _build_component_patch(kind, color)
                for kind, color in zip(plan, marker_colors)
            ]
        except RuntimeError:
            continue

        grid = np.zeros(_choose_grid_shape(len(plan)), dtype=int)
        patches.sort(key=lambda patch: int(np.count_nonzero(patch)), reverse=True)
        if not all(_place_patch_random(grid, patch) for patch in patches):
            continue
        if _validate_generated_grid(grid):
            return grid

    raise RuntimeError("failed to generate a valid input for puzzle 4")
