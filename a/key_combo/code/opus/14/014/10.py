from __future__ import annotations

import itertools
import random
from collections import deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
PINK = 6
ORANGE = 7

OBJECT_COLORS = (BLUE, RED, GREEN, ORANGE)
AXIS_FROM_INDICATOR = {BLUE: "lr", RED: "ud"}
INDICATOR_FROM_AXIS = {"lr": BLUE, "ud": RED}
MASK = PINK

CANVAS_HEIGHTS = tuple(range(13, 18))
CANVAS_WIDTHS = tuple(range(13, 18))
OBJECT_COUNTS = (3, 4)
PARTIAL_DIMS = ((3, 3), (3, 4), (4, 3), (4, 4))
GEN_ATTEMPTS = 600


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
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


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return False
    return len(_components(mask)) == 1


def _touches_all_edges(mask: np.ndarray) -> bool:
    return bool(mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any())


def _reflect_cell(r: int, c: int, h: int, w: int, axis: str) -> tuple[int, int]:
    if axis == "lr":
        return r, w - 1 - c
    return h - 1 - r, c


def _complete_crop(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "lr":
        return mask | np.fliplr(mask)
    return mask | np.flipud(mask)


def _solve_component_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    crop = mask[r0 : r1 + 1, c0 : c1 + 1]
    completed = _complete_crop(crop, axis)
    out = np.zeros_like(mask, dtype=bool)
    out[r0 : r1 + 1, c0 : c1 + 1] = completed
    return out


def _detect_legend(grid: np.ndarray) -> tuple[dict[int, str], set[tuple[int, int]]]:
    h, w = grid.shape
    corners = (
        ("tl", range(0, 4), (0, 1)),
        ("tr", range(0, 4), (w - 2, w - 1)),
        ("bl", range(h - 1, max(-1, h - 5), -1), (0, 1)),
        ("br", range(h - 1, max(-1, h - 5), -1), (w - 2, w - 1)),
    )

    for _name, rows, cols in corners:
        c0, c1 = cols
        for count in (4, 3):
            rows_list = list(rows)[:count]
            if len(rows_list) != count:
                continue
            ok = True
            mapping: dict[int, str] = {}
            cells: set[tuple[int, int]] = set()
            for r in rows_list:
                nz = np.flatnonzero(grid[r] != BG)
                if len(nz) != 2 or int(nz[0]) != c0 or int(nz[1]) != c1:
                    ok = False
                    break
                obj_color = int(grid[r, c0])
                indicator = int(grid[r, c1])
                if obj_color not in OBJECT_COLORS or indicator not in AXIS_FROM_INDICATOR:
                    ok = False
                    break
                if obj_color == indicator:
                    ok = False
                    break
                mapping[obj_color] = AXIS_FROM_INDICATOR[indicator]
                cells.add((int(r), int(c0)))
                cells.add((int(r), int(c1)))
            if ok and len(mapping) == count:
                return mapping, cells
    raise AssertionError("legend not found in a corner")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use a corner legend to choose the reflection axis for each masked monochrome motif, then complete it and erase the legend.

    Input structure:
    - Black background with 3-4 small non-touching motifs, each using one object color plus some pink masked cells.
    - Each motif occupies a tight small bbox and is only partially shown; the pink cells still belong to that motif.
    - One corner contains a legend made of 3-4 horizontal two-pixel pairs: object color on the left, axis-indicator color on the right.

    Rule:
    - Read the legend: blue means complete that color's motif by left-right reflection inside its tight bbox, red means complete it by up-down reflection.
    - For each non-legend component, treat both its visible color cells and pink cells as occupied cells of one partial motif.
    - Mirror the occupied cells across the instructed bbox mid-axis, fill the completed symmetric motif with the motif's solid color, and remove the corner legend to black.

    Key insight:
    - The corner legend does not recolor motifs; it encodes which reflection frame to apply, and the pink cells matter because they anchor the bbox and the partial occupancy before completion.
    """
    g = np.asarray(grid, dtype=int)
    mapping, legend_cells = _detect_legend(g)

    work = g.copy()
    for r, c in legend_cells:
        work[r, c] = BG

    out = np.zeros_like(work)
    for coords in _components(work != BG):
        colors = sorted({int(work[r, c]) for r, c in coords if int(work[r, c]) != MASK})
        assert len(colors) == 1, "each motif must have exactly one solid color"
        color = colors[0]
        assert color in mapping, "missing legend entry for motif color"

        r0, c0, r1, c1 = _bbox_from_coords(coords)
        crop = work[r0 : r1 + 1, c0 : c1 + 1]
        occupied = crop != BG
        completed = _complete_crop(occupied, mapping[color])
        out[r0 : r1 + 1, c0 : c1 + 1][completed] = color

    return out


def _random_connected_mask(h: int, w: int, cells: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True
    while int(mask.sum()) < cells:
        frontier = []
        for r, c in zip(*np.where(mask)):
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True
    return mask


def _mask_subsets(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    cells = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    subsets: list[tuple[tuple[int, int], ...]] = []
    max_k = min(3, len(cells) - 1)
    for k in range(1, max_k + 1):
        subsets.extend(itertools.combinations(cells, k))
    random.shuffle(subsets)
    return subsets


def _valid_mask_choices(mask: np.ndarray) -> dict[str, list[tuple[tuple[int, int], ...]]]:
    h, w = mask.shape
    valid = {"lr": [], "ud": []}
    solved_full = {axis: _solve_component_mask(mask, axis) for axis in ("lr", "ud")}

    for subset in _mask_subsets(mask):
        kept = mask.copy()
        for r, c in subset:
            kept[r, c] = False
        if not kept.any():
            continue
        for axis in ("lr", "ud"):
            mirrors_visible = True
            for r, c in subset:
                mr, mc = _reflect_cell(r, c, h, w, axis)
                if not mask[mr, mc] or (mr, mc) in subset:
                    mirrors_visible = False
                    break
            if not mirrors_visible:
                continue
            solved_kept = _solve_component_mask(kept, axis)
            if np.array_equal(solved_full[axis], solved_kept):
                continue
            valid[axis].append(subset)
    return valid


def _random_partial_spec() -> tuple[np.ndarray, dict[str, list[tuple[tuple[int, int], ...]]], dict[str, np.ndarray]] | None:
    for _ in range(200):
        h, w = random.choice(PARTIAL_DIMS)
        cell_count = random.randint(4, min(6, h * w))
        mask = _random_connected_mask(h, w, cell_count)
        if mask is None or not _is_connected(mask) or not _touches_all_edges(mask):
            continue

        solved = {axis: _solve_component_mask(mask, axis) for axis in ("lr", "ud")}
        if np.array_equal(solved["lr"], solved["ud"]):
            continue
        if not all(4 <= int(solved[axis].sum()) <= 8 for axis in ("lr", "ud")):
            continue
        if not all(_is_connected(solved[axis]) for axis in ("lr", "ud")):
            continue

        choices = _valid_mask_choices(mask)
        if choices["lr"] and choices["ud"]:
            return mask, choices, solved
    return None


def _axis_assignment(colors: list[int]) -> dict[int, str] | None:
    assignment: dict[int, str] = {}
    free_colors: list[int] = []
    for color in colors:
        if color == BLUE:
            assignment[color] = "ud"
        elif color == RED:
            assignment[color] = "lr"
        else:
            free_colors.append(color)

    present = set(assignment.values())
    missing = {"lr", "ud"} - present
    if len(missing) > len(free_colors):
        return None

    random.shuffle(free_colors)
    for axis, color in zip(sorted(missing), free_colors):
        assignment[color] = axis
    for color in free_colors:
        if color not in assignment:
            assignment[color] = random.choice(("lr", "ud"))

    if set(assignment.values()) != {"lr", "ud"}:
        return None
    return assignment


def _legend_layout(h: int, w: int, count: int, corner: str) -> tuple[list[int], tuple[int, int]]:
    if corner.startswith("t"):
        rows = list(range(count))
    else:
        rows = list(range(h - count, h))
    cols = (0, 1) if corner.endswith("l") else (w - 2, w - 1)
    return rows, cols


def _reserve_rows(h: int, count: int, corner: str) -> tuple[int, int]:
    if corner.startswith("t"):
        return count + 1, h - 2
    return 1, h - count - 2


def _can_place(boxes: list[tuple[int, int, int, int]], cand: tuple[int, int, int, int]) -> bool:
    r0, c0, r1, c1 = cand
    for ar0, ac0, ar1, ac1 in boxes:
        if not (r1 + 1 < ar0 or ar1 + 1 < r0 or c1 + 1 < ac0 or ac1 + 1 < c0):
            return False
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    h = random.choice(CANVAS_HEIGHTS)
    w = random.choice(CANVAS_WIDTHS)
    count = random.choice(OBJECT_COUNTS)
    colors = random.sample(list(OBJECT_COLORS), count)
    axis_by_color = _axis_assignment(colors)
    if axis_by_color is None:
        return None

    corner = random.choice(("tl", "tr", "bl", "br"))
    legend_rows, legend_cols = _legend_layout(h, w, count, corner)
    row_min, row_max = _reserve_rows(h, count, corner)
    if row_min > row_max:
        return None

    object_specs = []
    for color in colors:
        spec = _random_partial_spec()
        if spec is None:
            return None
        partial, mask_choices, solved = spec
        axis = axis_by_color[color]
        subset = random.choice(mask_choices[axis])
        object_specs.append(
            {
                "color": color,
                "axis": axis,
                "partial": partial,
                "masked": subset,
                "solved": solved[axis],
            }
        )

    boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = []
    for spec in object_specs:
        ph, pw = spec["partial"].shape
        candidates = []
        for r0 in range(row_min, row_max - ph + 2):
            for c0 in range(2, w - pw - 1):
                cand = (r0, c0, r0 + ph - 1, c0 + pw - 1)
                if _can_place(boxes, cand):
                    candidates.append((r0, c0))
        if not candidates:
            return None
        pos = random.choice(candidates)
        placements.append(pos)
        r0, c0 = pos
        boxes.append((r0, c0, r0 + ph - 1, c0 + pw - 1))

    inp = np.zeros((h, w), dtype=int)
    out = np.zeros((h, w), dtype=int)

    c0, c1 = legend_cols
    legend_order = random.sample(colors, len(colors))
    for r, color in zip(legend_rows, legend_order):
        inp[r, c0] = color
        inp[r, c1] = INDICATOR_FROM_AXIS[axis_by_color[color]]

    for spec, (r0, c0) in zip(object_specs, placements):
        partial = spec["partial"]
        color = spec["color"]
        masked = set(spec["masked"])
        solved = spec["solved"]
        if int(solved.sum()) <= int(partial.sum()):
            return None
        for rr, cc in np.argwhere(partial):
            inp[r0 + int(rr), c0 + int(cc)] = MASK if (int(rr), int(cc)) in masked else color
        for rr, cc in np.argwhere(solved):
            out[r0 + int(rr), c0 + int(cc)] = color

        other_axis = "ud" if spec["axis"] == "lr" else "lr"
        if np.array_equal(spec["solved"], _solve_component_mask(partial, other_axis)):
            return None

    actual = main(inp)
    if not np.array_equal(actual, out):
        return None

    legendless = inp.copy()
    for r in legend_rows:
        legendless[r, c0] = BG
        legendless[r, c1] = BG
    if np.array_equal(out, legendless):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _build_case()
        if case is not None:
            return case[0]
    raise RuntimeError("failed to generate a valid input for puzzle 10")
