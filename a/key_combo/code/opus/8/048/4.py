from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

LEGEND_SIZE = 4
LEGEND_BORDER = GREY
MARKER_COLORS = (PINK, TEAL, MAROON)
OBJECT_COLORS = (BLUE, RED, GREEN, YELLOW, ORANGE)

DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _components_from_mask(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_targets(grid: np.ndarray) -> set[int]:
    legend_inner = grid[1:3, 1:3]
    targets = {int(v) for v in np.unique(legend_inner) if int(v) not in (BLACK, LEGEND_BORDER)}
    assert len(targets) == 2
    return targets


def _find_marker_positions(grid: np.ndarray) -> list[tuple[int, int]]:
    h, w = grid.shape
    legend_mask = np.zeros((h, w), dtype=bool)
    legend_mask[:LEGEND_SIZE, :LEGEND_SIZE] = True
    positions: list[tuple[int, int]] = []
    for color in MARKER_COLORS:
        mask = (grid == color) & ~legend_mask
        for comp in _components_from_mask(mask):
            if len(comp) == 1:
                positions.append(comp[0])
    return sorted(positions)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= LEGEND_SIZE and w >= LEGEND_SIZE

    legend = g[:LEGEND_SIZE, :LEGEND_SIZE]
    assert np.all(legend[0, :] == LEGEND_BORDER)
    assert np.all(legend[-1, :] == LEGEND_BORDER)
    assert np.all(legend[:, 0] == LEGEND_BORDER)
    assert np.all(legend[:, -1] == LEGEND_BORDER)
    targets = _extract_targets(g)

    marker_positions = _find_marker_positions(g)
    marker_pos_set = set(marker_positions)

    object_mask = g != BLACK
    object_mask[:LEGEND_SIZE, :LEGEND_SIZE] = False
    for r, c in marker_positions:
        object_mask[r, c] = False

    components = []
    for idx, comp in enumerate(_components_from_mask(object_mask)):
        coords = sorted(comp)
        colors = {int(g[r, c]) for r, c in coords}
        contacts: dict[tuple[int, int], int] = {}
        for mr, mc in marker_positions:
            count = 0
            for r, c in coords:
                if abs(r - mr) + abs(c - mc) == 1:
                    count += 1
            if count:
                contacts[(mr, mc)] = count
        components.append(
            {
                "id": idx,
                "coords": coords,
                "colors": colors,
                "contains_all_targets": targets.issubset(colors),
                "contacts": contacts,
            }
        )

    marker_degrees: dict[tuple[int, int], int] = {pos: 0 for pos in marker_positions}
    for component in components:
        for pos, count in component["contacts"].items():
            marker_degrees[pos] += count

    return {
        "targets": targets,
        "markers": marker_positions,
        "marker_colors": {pos: int(g[pos]) for pos in marker_positions},
        "marker_degrees": marker_degrees,
        "components": components,
    }


def _winning_marker(component: dict, marker_colors: dict[tuple[int, int], int]) -> tuple[int, int]:
    contacts = component["contacts"]
    assert contacts
    best_count = max(contacts.values())
    winners = [pos for pos, count in contacts.items() if count == best_count]
    return min(winners)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Erase legend-selected multicolor components, then recolor each survivor by the side-adjacent marker pixel with the strongest contact.

    Input structure:
    - The grid is mostly black background, with a fixed 4x4 grey-framed legend in the top-left corner.
    - The legend interior contains exactly two target colors, while the rest of the board contains irregular multicolor 4-connected components and isolated marker pixels in colors 6, 8, or 9.
    - Marker pixels are single-cell components; all other non-black, non-legend foreground cells belong to the candidate objects.

    Rule:
    - Read the two target colors from the 2x2 legend interior.
    - Find all 4-connected foreground components after excluding the legend and excluding isolated marker pixels.
    - Erase every component whose color set contains both legend target colors.
    - For each surviving component, count edge adjacencies to each marker pixel and recolor the whole component with the color of the marker having the largest count.
    - Break ties by marker position in row-major order, and leave the legend and marker pixels unchanged.

    Key insight:
    - The legend is not a repaint key but a deletion filter: only after removing components that contain the full legend palette do the remaining components get recolored by per-marker side-contact counts.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_grid(g)
    out = g.copy()

    for component in info["components"]:
        coords = component["coords"]
        if component["contains_all_targets"]:
            for r, c in coords:
                out[r, c] = BLACK
            continue
        winner = _winning_marker(component, info["marker_colors"])
        new_color = info["marker_colors"][winner]
        for r, c in coords:
            out[r, c] = new_color
    return out


def _mask_from_rows(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


BASE_SHAPES = [
    _mask_from_rows(["11", "10", "10"]),
    _mask_from_rows(["111", "010"]),
    _mask_from_rows(["110", "011"]),
    _mask_from_rows(["111", "100"]),
    _mask_from_rows(["11", "11", "10"]),
    _mask_from_rows(["111", "010", "010"]),
    _mask_from_rows(["101", "111"]),
    _mask_from_rows(["111", "101"]),
    _mask_from_rows(["110", "111"]),
    _mask_from_rows(["111", "100", "100"]),
    _mask_from_rows(["111", "100", "111"]),
    _mask_from_rows(["111", "110"]),
    _mask_from_rows(["111", "011"]),
    _mask_from_rows(["111", "101", "100"]),
    _mask_from_rows(["1111", "1010"]),
    _mask_from_rows(["111", "110", "100"]),
    _mask_from_rows(["1111", "0011"]),
    _mask_from_rows(["111", "111", "100"]),
    _mask_from_rows(["1111", "0111"]),
]


def _shape_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen = set()
    for flip in (False, True):
        base = np.fliplr(mask) if flip else mask
        for k in range(4):
            cur = np.rot90(base, k)
            key = cur.tobytes() + bytes(cur.shape)
            if key not in seen:
                seen.add(key)
                variants.append(cur.copy())
    return variants


ALL_SHAPES: list[np.ndarray] = []
for _base in BASE_SHAPES:
    ALL_SHAPES.extend(_shape_variants(_base))


def _component_candidates_for_shape(mask: np.ndarray) -> Counter:
    h, w = mask.shape
    coords = [(r, c) for r in range(h) for c in range(w) if mask[r, c]]
    counts: Counter = Counter()
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                continue
            counts[(nr, nc)] += 1
    return counts


SHAPES_WITH_DOUBLE_CONTACT = [
    shape for shape in ALL_SHAPES if any(v >= 2 for v in _component_candidates_for_shape(shape).values())
]


def _color_set_for_role(role: str, targets: tuple[int, int], non_targets: tuple[int, ...]) -> tuple[int, ...]:
    t1, t2 = targets
    if role == "delete2":
        return tuple(sorted((t1, t2)))
    if role == "delete3":
        extra = random.choice(non_targets)
        return tuple(sorted((t1, t2, extra)))
    if role == "keep_none":
        size = random.randint(1, min(3, len(non_targets)))
        return tuple(sorted(random.sample(non_targets, size)))
    if role == "keep_t1":
        extra_pool = [c for c in non_targets if c != t2]
        size = random.randint(1, min(3, 1 + len(extra_pool)))
        chosen = {t1}
        if size > 1:
            chosen.update(random.sample(extra_pool, size - 1))
        return tuple(sorted(chosen))
    if role == "keep_t2":
        extra_pool = [c for c in non_targets if c != t1]
        size = random.randint(1, min(3, 1 + len(extra_pool)))
        chosen = {t2}
        if size > 1:
            chosen.update(random.sample(extra_pool, size - 1))
        return tuple(sorted(chosen))
    options = []
    for size in (1, 2, 3):
        for _ in range(24):
            chosen = tuple(sorted(random.sample(OBJECT_COLORS, size)))
            if not ({t1, t2} <= set(chosen)):
                options.append(chosen)
    return random.choice(options)


def _paint_component_cells(cells: list[tuple[int, int]], colors: tuple[int, ...]) -> dict[tuple[int, int], int]:
    shuffled = cells[:]
    random.shuffle(shuffled)
    cell_colors: dict[tuple[int, int], int] = {}
    for idx, color in enumerate(colors):
        cell_colors[shuffled[idx]] = color
    for cell in shuffled[len(colors) :]:
        cell_colors[cell] = random.choice(colors)
    return cell_colors


def _fits_component(
    shape: np.ndarray,
    top: int,
    left: int,
    h: int,
    w: int,
    occupied: np.ndarray,
) -> bool:
    sh, sw = shape.shape
    if top < 0 or left < 0 or top + sh > h or left + sw > w:
        return False
    for r in range(sh):
        for c in range(sw):
            if not shape[r, c]:
                continue
            ar, ac = top + r, left + c
            if occupied[ar, ac]:
                return False
            for nr, nc in _neighbors4(ar, ac):
                if _in_bounds((h, w), nr, nc) and occupied[nr, nc]:
                    return False
    return True


def _place_components(
    h: int,
    w: int,
    specs: list[dict],
) -> list[dict] | None:
    order = sorted(
        range(len(specs)),
        key=lambda i: (specs[i]["shape"].sum(), specs[i]["shape"].shape[0] * specs[i]["shape"].shape[1]),
        reverse=True,
    )
    occupied = np.zeros((h, w), dtype=bool)
    occupied[: LEGEND_SIZE + 1, : LEGEND_SIZE + 1] = True
    placed: list[dict | None] = [None] * len(specs)

    def backtrack(k: int) -> bool:
        if k == len(order):
            return True
        idx = order[k]
        spec = specs[idx]
        shape = spec["shape"]
        sh, sw = shape.shape
        positions = [(r, c) for r in range(h - sh + 1) for c in range(w - sw + 1)]
        random.shuffle(positions)
        for top, left in positions[:160]:
            if not _fits_component(shape, top, left, h, w, occupied):
                continue
            cells = []
            for r in range(sh):
                for c in range(sw):
                    if shape[r, c]:
                        ar, ac = top + r, left + c
                        occupied[ar, ac] = True
                        cells.append((ar, ac))
            placed[idx] = {
                "role": spec["role"],
                "shape": shape,
                "cells": sorted(cells),
                "color_set": spec["color_set"],
            }
            if backtrack(k + 1):
                return True
            for ar, ac in cells:
                occupied[ar, ac] = False
            placed[idx] = None
        return False

    if backtrack(0):
        return [item for item in placed if item is not None]
    return None


def _marker_touch_candidates(
    h: int,
    w: int,
    component_cells: list[list[tuple[int, int]]],
    blocked: np.ndarray,
) -> dict[tuple[int, int], dict[int, int]]:
    owners = np.full((h, w), -1, dtype=int)
    for idx, cells in enumerate(component_cells):
        for r, c in cells:
            owners[r, c] = idx

    candidates: dict[tuple[int, int], dict[int, int]] = {}
    for r in range(h):
        for c in range(w):
            if blocked[r, c]:
                continue
            contacts: dict[int, int] = {}
            for nr, nc in _neighbors4(r, c):
                if _in_bounds((h, w), nr, nc) and owners[nr, nc] >= 0:
                    owner = int(owners[nr, nc])
                    contacts[owner] = contacts.get(owner, 0) + 1
            candidates[(r, c)] = contacts
    return candidates


def _marker_cell_ok(cell: tuple[int, int], chosen: dict[tuple[int, int], int]) -> bool:
    r, c = cell
    for mr, mc in chosen:
        if abs(r - mr) + abs(c - mc) == 1:
            return False
    return True


def _allocate_markers(
    h: int,
    w: int,
    placed_components: list[dict],
    branch_type: str,
) -> dict[tuple[int, int], int] | None:
    component_cells = [comp["cells"] for comp in placed_components]
    blocked = np.zeros((h, w), dtype=bool)
    blocked[:LEGEND_SIZE, :LEGEND_SIZE] = True
    for cells in component_cells:
        for r, c in cells:
            blocked[r, c] = True

    candidates = _marker_touch_candidates(h, w, component_cells, blocked)
    survivors = [idx for idx, comp in enumerate(placed_components) if "delete" not in comp["role"]]
    isolated = {
        idx: sorted(
            [(cell, counts[idx]) for cell, counts in candidates.items() if list(counts.keys()) == [idx]],
            key=lambda item: (item[0][0], item[0][1], -item[1]),
        )
        for idx in survivors
    }

    if any(not isolated[idx] for idx in survivors):
        return None

    if branch_type == "tie":
        special_choices = [idx for idx in survivors if len(isolated[idx]) >= 2]
    else:
        special_choices = [
            idx
            for idx in survivors
            if any(cnt >= 2 for _, cnt in isolated[idx]) and len(isolated[idx]) >= 2
        ]
    if not special_choices:
        return None
    special = random.choice(special_choices)

    output_palette = [PINK, TEAL, MAROON]
    desired_outputs = output_palette[:]
    while len(desired_outputs) < len(survivors):
        desired_outputs.extend(output_palette)
    random.shuffle(desired_outputs)
    if len(set(desired_outputs[: len(survivors)])) < 2 and len(survivors) > 1:
        desired_outputs[0] = PINK
        desired_outputs[1] = TEAL
    desired_outputs = desired_outputs[: len(survivors)]

    chosen: dict[tuple[int, int], int] = {}
    survivor_color = {idx: desired_outputs[pos] for pos, idx in enumerate(survivors)}

    for idx in survivors:
        if idx == special:
            continue
        good = [cell for cell, _ in isolated[idx] if _marker_cell_ok(cell, chosen)]
        if not good:
            return None
        chosen[good[0]] = survivor_color[idx]

    if branch_type == "tie":
        options = []
        for cell_a, cnt_a in isolated[special]:
            if not _marker_cell_ok(cell_a, chosen):
                continue
            for cell_b, cnt_b in isolated[special]:
                if cell_a == cell_b or cnt_a != cnt_b:
                    continue
                if not _marker_cell_ok(cell_b, chosen):
                    continue
                if abs(cell_a[0] - cell_b[0]) + abs(cell_a[1] - cell_b[1]) == 1:
                    continue
                options.append(tuple(sorted((cell_a, cell_b))))
        if not options:
            return None
        cell1, cell2 = random.choice(options)
        winner_cell, loser_cell = sorted((cell1, cell2))[0], sorted((cell1, cell2))[1]
        winner_color = survivor_color[special]
        loser_color = random.choice([c for c in MARKER_COLORS if c != winner_color])
        chosen[winner_cell] = winner_color
        chosen[loser_cell] = loser_color
    else:
        double_cells = [cell for cell, cnt in isolated[special] if cnt >= 2 and _marker_cell_ok(cell, chosen)]
        if not double_cells:
            return None
        winner_cell = double_cells[0]
        chosen[winner_cell] = survivor_color[special]
        loser_options = [
            cell
            for cell, cnt in isolated[special]
            if cell != winner_cell and cnt < next(c for c0, c in isolated[special] if c0 == winner_cell) and _marker_cell_ok(cell, chosen)
        ]
        if not loser_options:
            return None
        loser_cell = loser_options[0]
        loser_color = random.choice([c for c in MARKER_COLORS if c != survivor_color[special]])
        chosen[loser_cell] = loser_color

    distractor_cells = [
        cell
        for cell, counts in sorted(candidates.items())
        if not counts and _marker_cell_ok(cell, chosen) and cell[0] >= LEGEND_SIZE + 1 and cell[1] >= LEGEND_SIZE + 1
    ]
    if not distractor_cells:
        return None
    chosen[distractor_cells[0]] = random.choice(MARKER_COLORS)

    return chosen


def _build_grid(
    h: int,
    w: int,
    targets: tuple[int, int],
    placed_components: list[dict],
    markers: dict[tuple[int, int], int],
) -> np.ndarray:
    grid = np.zeros((h, w), dtype=int)
    grid[:LEGEND_SIZE, :LEGEND_SIZE] = LEGEND_BORDER
    legend_cells = [(1, 1), (1, 2), (2, 1), (2, 2)]
    legend_values = [targets[0], targets[1], random.choice(targets), random.choice(targets)]
    random.shuffle(legend_values)
    for cell, value in zip(legend_cells, legend_values):
        grid[cell] = value

    for comp in placed_components:
        painted = _paint_component_cells(comp["cells"], comp["color_set"])
        for cell, color in painted.items():
            grid[cell] = color

    for cell, color in markers.items():
        grid[cell] = color
    return grid


def _validate_generated(grid: np.ndarray, branch_type: str) -> bool:
    info = _analyze_grid(grid)
    components = info["components"]
    markers = info["markers"]

    if not (6 <= len(components) <= 8):
        return False
    if not (5 <= len(markers) <= 8):
        return False

    erased = [comp for comp in components if comp["contains_all_targets"]]
    kept = [comp for comp in components if not comp["contains_all_targets"]]
    if len(erased) < 2 or len(kept) < 3:
        return False

    target_set = info["targets"]
    kept_intersections = [len(comp["colors"] & target_set) for comp in kept]
    if 0 not in kept_intersections or 1 not in kept_intersections:
        return False
    if not any(len(comp["colors"]) == 3 for comp in erased):
        return False

    marker_degrees = info["marker_degrees"]
    if not any(deg == 0 for deg in marker_degrees.values()):
        return False
    if not any(deg > 0 for deg in marker_degrees.values()):
        return False

    for comp in components:
        if not (4 <= len(comp["coords"]) <= 12):
            return False
        if not comp["colors"] <= set(OBJECT_COLORS):
            return False

    for comp in kept:
        if not comp["contacts"]:
            return False

    outputs = main(grid)
    survivor_colors = set()
    for comp in kept:
        coords = comp["coords"]
        colors = {int(outputs[r, c]) for r, c in coords}
        if len(colors) != 1:
            return False
        survivor_colors |= colors
    if len(survivor_colors) < 2:
        return False

    branch_hits = 0
    for comp in kept:
        counts = sorted(comp["contacts"].values(), reverse=True)
        if len(counts) < 2:
            continue
        if branch_type == "tie" and counts[0] == counts[1]:
            winners = [pos for pos, cnt in comp["contacts"].items() if cnt == counts[0]]
            if len(winners) >= 2:
                winner = _winning_marker(comp, info["marker_colors"])
                if winner == min(winners):
                    branch_hits += 1
        if branch_type == "majority" and counts[0] > counts[1] > 0:
            branch_hits += 1
    if branch_hits == 0:
        return False

    return True


def _component_specs(targets: tuple[int, int]) -> list[dict]:
    n_components = random.randint(6, 8)
    n_deleted = random.randint(2, min(3, n_components - 3))
    n_kept = n_components - n_deleted
    non_targets = tuple(c for c in OBJECT_COLORS if c not in targets)

    roles = ["delete3"] + ["delete2"] * (n_deleted - 1)
    keep_roles = ["keep_none", "keep_t1", "keep_t2"]
    while len(keep_roles) < n_kept:
        keep_roles.append("keep_misc")
    random.shuffle(keep_roles)
    roles.extend(keep_roles[:n_kept])
    random.shuffle(roles)

    specs = []
    for role in roles:
        shape_pool = SHAPES_WITH_DOUBLE_CONTACT if random.random() < 0.45 else ALL_SHAPES
        shape = random.choice(shape_pool)
        specs.append(
            {
                "role": role,
                "shape": shape,
                "color_set": _color_set_for_role(role, targets, non_targets),
            }
        )
    return specs


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    targets = tuple(random.sample(OBJECT_COLORS, 2))
    branch_type = random.choice(["tie", "majority"])

    specs = _component_specs(targets)
    placed = _place_components(h, w, specs)
    if placed is None:
        return None

    markers = _allocate_markers(h, w, placed, branch_type)
    if markers is None:
        return None

    grid = _build_grid(h, w, targets, placed, markers)
    if not _validate_generated(grid, branch_type):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
