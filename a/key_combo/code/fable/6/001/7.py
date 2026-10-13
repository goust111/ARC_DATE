from __future__ import annotations

import random
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import find_components, get_bg


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
INNER_HEIGHT_RANGE = range(3, 6)
INNER_WIDTH_RANGE = range(4, 7)
MIN_USED_PIECES = 3
MAX_USED_PIECES = 5
GRID_HEIGHT_RANGE = range(18, 29)
GRID_WIDTH_RANGE = range(18, 29)


def _bbox_from_coords(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize_coords(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0, c0, _, _ = _bbox_from_coords(tuple(pts))
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_dims(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _is_connected(shape: tuple[tuple[int, int], ...]) -> bool:
    cells = set(shape)
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in cells and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == len(cells)


def _has_hole(shape: tuple[tuple[int, int], ...]) -> bool:
    mask = _shape_mask(shape)
    h, w = mask.shape
    visited = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if r not in (0, h - 1) and c not in (0, w - 1):
                continue
            if mask[r, c] or visited[r, c]:
                continue
            visited[r, c] = True
            stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not visited[nr, nc]:
                visited[nr, nc] = True
                stack.append((nr, nc))
    return bool((~mask & ~visited).any())


def _is_hollow_frame_shape(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(shape)
    if h < 3 or w < 3:
        return False
    cells = set(shape)
    expected = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if r in (0, h - 1) or c in (0, w - 1)
    }
    return cells == expected


def _is_irregular(shape: tuple[tuple[int, int], ...]) -> bool:
    h, w = _shape_dims(shape)
    return len(shape) < h * w


def _component_to_piece(component: dict) -> dict:
    coords = tuple(sorted((int(r), int(c)) for r, c in component["coords"]))
    shape = _normalize_coords(coords)
    h, w = _shape_dims(shape)
    return {
        "color": int(component["color"]),
        "coords": coords,
        "shape": shape,
        "shape_h": h,
        "shape_w": w,
        "area": len(shape),
    }


def _is_frame_component(grid: np.ndarray, component: dict, bg: int) -> bool:
    color = int(component["color"])
    r0, c0, r1, c1 = map(int, component["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if int(component["area"]) != 2 * (h + w) - 4:
        return False
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if not np.all(patch[border] == color):
        return False
    if np.any(patch[~border] == color):
        return False
    if np.any(patch[~border] != int(bg)):
        return False
    return True


def _extract_scene(grid: np.ndarray) -> tuple[int, tuple[int, int, int, int], tuple[int, int], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    components = find_components(
        g,
        bg=bg,
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    frame_candidates = [comp for comp in components if _is_frame_component(g, comp, bg)]
    assert len(frame_candidates) == 1, "expected exactly one hollow rectangular frame"
    frame = frame_candidates[0]
    pieces = [_component_to_piece(comp) for comp in components if comp is not frame]
    r0, c0, r1, c1 = map(int, frame["bbox"])
    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    assert inner_h > 0 and inner_w > 0
    return bg, (r0, c0, r1, c1), (inner_h, inner_w), pieces


def _placement_options(piece: dict, inner_h: int, inner_w: int) -> list[dict]:
    h = int(piece["shape_h"])
    w = int(piece["shape_w"])
    if h > inner_h or w > inner_w:
        return []
    options = []
    for r0 in range(inner_h - h + 1):
        for c0 in range(inner_w - w + 1):
            cells = tuple((r0 + r, c0 + c) for r, c in piece["shape"])
            mask = 0
            for rr, cc in cells:
                mask |= 1 << (rr * inner_w + cc)
            options.append({"mask": mask, "cells": cells, "origin": (r0, c0)})
    return options


def _solve_unique_tiling(
    pieces: list[dict], inner_h: int, inner_w: int, max_solutions: int = 2
) -> list[list[dict]]:
    n = len(pieces)
    total_cells = inner_h * inner_w
    target_mask = (1 << total_cells) - 1
    piece_areas = [int(piece["area"]) for piece in pieces]
    flat_options: list[dict] = []
    by_cell: list[list[int]] = [[] for _ in range(total_cells)]
    option_counts = []

    for piece_idx, piece in enumerate(pieces):
        options = _placement_options(piece, inner_h, inner_w)
        option_counts.append(len(options))
        for option in options:
            oid = len(flat_options)
            flat_options.append(
                {
                    "piece_idx": piece_idx,
                    "mask": int(option["mask"]),
                    "cells": option["cells"],
                    "origin": option["origin"],
                }
            )
            for rr, cc in option["cells"]:
                by_cell[rr * inner_w + cc].append(oid)

    @lru_cache(maxsize=None)
    def _reachable_sums(piece_mask: int) -> frozenset[int]:
        sums = {0}
        work = int(piece_mask)
        while work:
            lsb = work & -work
            idx = lsb.bit_length() - 1
            area = piece_areas[idx]
            sums |= {val + area for val in list(sums)}
            work ^= lsb
        return frozenset(sums)

    @lru_cache(maxsize=None)
    def _remaining_area(piece_mask: int) -> int:
        total = 0
        work = int(piece_mask)
        while work:
            lsb = work & -work
            idx = lsb.bit_length() - 1
            total += piece_areas[idx]
            work ^= lsb
        return total

    solutions: list[list[dict]] = []

    def dfs(filled_mask: int, unused_mask: int, chosen: tuple[int, ...]) -> None:
        if len(solutions) >= max_solutions:
            return
        uncovered_count = total_cells - int(filled_mask.bit_count())
        if uncovered_count == 0:
            sol = []
            for oid in chosen:
                opt = flat_options[oid]
                sol.append(
                    {
                        "piece_idx": int(opt["piece_idx"]),
                        "cells": opt["cells"],
                        "origin": opt["origin"],
                    }
                )
            sol.sort(key=lambda item: item["piece_idx"])
            solutions.append(sol)
            return
        if _remaining_area(unused_mask) < uncovered_count:
            return
        if uncovered_count not in _reachable_sums(unused_mask):
            return

        uncovered_mask = target_mask ^ filled_mask
        best_options: list[int] | None = None
        bits = uncovered_mask
        while bits:
            lsb = bits & -bits
            cell = lsb.bit_length() - 1
            viable = []
            for oid in by_cell[cell]:
                opt = flat_options[oid]
                piece_idx = int(opt["piece_idx"])
                if not ((unused_mask >> piece_idx) & 1):
                    continue
                if opt["mask"] & filled_mask:
                    continue
                viable.append(oid)
            if not viable:
                return
            if best_options is None or len(viable) < len(best_options):
                best_options = viable
                if len(best_options) == 1:
                    break
            bits ^= lsb

        assert best_options is not None
        best_options.sort(
            key=lambda oid: (
                option_counts[int(flat_options[oid]["piece_idx"])],
                -piece_areas[int(flat_options[oid]["piece_idx"])],
            )
        )
        for oid in best_options:
            opt = flat_options[oid]
            piece_idx = int(opt["piece_idx"])
            dfs(
                filled_mask | int(opt["mask"]),
                unused_mask & ~(1 << piece_idx),
                chosen + (oid,),
            )

    dfs(0, (1 << n) - 1, ())
    return solutions


def _paint_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], top: int, left: int, color: int) -> None:
    for r, c in shape:
        grid[top + r, left + c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move exactly the translatable pieces that uniquely tile the hollow frame interior into that frame.

    Input structure:
    - The background is black.
    - One foreground object is a one-pixel-thick monochrome hollow rectangular frame whose interior is empty.
    - The remaining foreground objects are scattered monochrome 4-connected puzzle pieces, each with a distinct color.

    Rule:
    - Find the unique subset of pieces that can be translated, without rotation or reflection, to tile the frame interior exactly as a solid rectangle.
    - Redraw those used pieces inside the frame at their unique non-overlapping translated positions, preserving each piece's color.
    - Erase the used pieces from their original locations, keep every unused distractor where it is, and leave the frame unchanged.

    Key insight:
    - The frame reveals only the target rectangle's boundary; selecting the correct pieces requires exact complementary tiling, not color, size, or source position heuristics.
    """
    g = np.asarray(grid, dtype=int)
    bg, frame_bbox, (inner_h, inner_w), pieces = _extract_scene(g)
    solutions = _solve_unique_tiling(pieces, inner_h, inner_w, max_solutions=2)
    assert len(solutions) == 1, "tiling must be unique"

    r0, c0, r1, c1 = frame_bbox
    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = int(bg)

    for placement in solutions[0]:
        piece = pieces[int(placement["piece_idx"])]
        for rr, cc in piece["coords"]:
            out[rr, cc] = int(bg)
        for rr, cc in placement["cells"]:
            out[r0 + 1 + rr, c0 + 1 + cc] = int(piece["color"])
    return out


def _random_partition(inner_h: int, inner_w: int, n_pieces: int) -> list[tuple[tuple[int, int], ...]] | None:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    if len(cells) < 2 * n_pieces:
        return None
    seeds = random.sample(cells, n_pieces)
    owner = {cell: idx for idx, cell in enumerate(seeds)}
    unassigned = set(cells) - set(seeds)
    sizes = [1] * n_pieces
    frontier = {
        nb
        for cell in seeds
        for nb in _neighbors4(*cell)
        if nb in unassigned
    }

    while unassigned:
        candidates = [
            cell
            for cell in frontier
            if any(nb in owner for nb in _neighbors4(*cell))
        ]
        if not candidates:
            return None
        cell = random.choice(candidates)
        neighbor_ids = sorted({owner[nb] for nb in _neighbors4(*cell) if nb in owner})
        weights = [max(1, inner_h * inner_w // n_pieces + 2 - sizes[idx]) for idx in neighbor_ids]
        piece_idx = random.choices(neighbor_ids, weights=weights, k=1)[0]
        owner[cell] = piece_idx
        sizes[piece_idx] += 1
        unassigned.remove(cell)
        frontier.discard(cell)
        for nb in _neighbors4(*cell):
            if nb in unassigned:
                frontier.add(nb)

    pieces = []
    for piece_idx in range(n_pieces):
        shape = _normalize_coords([cell for cell, idx in owner.items() if idx == piece_idx])
        if len(shape) < 2 or not _is_connected(shape) or _has_hole(shape) or _is_hollow_frame_shape(shape):
            return None
        pieces.append(shape)
    return pieces


def _random_connected_shape(
    area: int,
    max_h: int,
    max_w: int,
    *,
    forbidden: set[tuple[tuple[int, int], ...]],
) -> tuple[tuple[int, int], ...] | None:
    for _ in range(200):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = {
                nb
                for r, c in list(cells)
                for nb in _neighbors4(r, c)
                if nb not in cells and -area <= nb[0] <= area and -area <= nb[1] <= area
            }
            if not frontier:
                break
            cells.add(random.choice(list(frontier)))
        if len(cells) != area:
            continue
        shape = _normalize_coords(cells)
        if shape in forbidden:
            continue
        h, w = _shape_dims(shape)
        if h > max_h or w > max_w:
            continue
        if not _is_connected(shape) or _has_hole(shape) or _is_hollow_frame_shape(shape):
            continue
        return shape
    return None


def _area_replacement_plan(used_areas: list[int]) -> list[int] | None:
    index_order = list(range(len(used_areas)))
    random.shuffle(index_order)
    for idx in index_order:
        area = used_areas[idx]
        if area >= 4 and random.random() < 0.6:
            left = random.randint(2, area - 2)
            return [left, area - left]
        if area >= 2 and random.random() < 0.4:
            return [area]

    pairs = [(i, j) for i in range(len(used_areas)) for j in range(i + 1, len(used_areas))]
    random.shuffle(pairs)
    fallback = None
    for i, j in pairs:
        total = used_areas[i] + used_areas[j]
        fallback = [total]
        if total >= 4 and random.random() < 0.7:
            left = random.randint(2, total - 2)
            return [left, total - left]
    return fallback


def _has_alternative_area_subset(all_areas: list[int], target_area: int, used_count: int) -> bool:
    n = len(all_areas)
    used_mask = (1 << used_count) - 1
    for subset in range(1, 1 << n):
        if subset == used_mask:
            continue
        area = 0
        for idx in range(n):
            if (subset >> idx) & 1:
                area += all_areas[idx]
        if area == target_area:
            return True
    return False


def _shape_to_patch(shape: tuple[tuple[int, int], ...], color: int, bg: int = BG) -> np.ndarray:
    mask = _shape_mask(shape)
    patch = np.full(mask.shape, bg, dtype=int)
    patch[mask] = int(color)
    return patch


def _reserve_box(blocked: np.ndarray, top: int, left: int, h: int, w: int, padding: int) -> None:
    r0 = max(0, top - padding)
    c0 = max(0, left - padding)
    r1 = min(blocked.shape[0], top + h + padding)
    c1 = min(blocked.shape[1], left + w + padding)
    blocked[r0:r1, c0:c1] = True


def _place_patch(
    grid: np.ndarray,
    blocked: np.ndarray,
    patch: np.ndarray,
    *,
    padding: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    candidates = []
    for top in range(grid.shape[0] - h + 1):
        for left in range(grid.shape[1] - w + 1):
            if blocked[top : top + h, left : left + w].any():
                continue
            candidates.append((top, left))
    if not candidates:
        return None
    top, left = random.choice(candidates)
    mask = patch != int(BG)
    grid[top : top + h, left : left + w][mask] = patch[mask]
    _reserve_box(blocked, top, left, h, w, padding)
    return top, left


def _frame_patch(inner_h: int, inner_w: int, color: int) -> np.ndarray:
    patch = np.full((inner_h + 2, inner_w + 2), BG, dtype=int)
    patch[0, :] = int(color)
    patch[-1, :] = int(color)
    patch[:, 0] = int(color)
    patch[:, -1] = int(color)
    return patch


def _expected_output(
    inp: np.ndarray,
    frame_top: int,
    frame_left: int,
    inner_h: int,
    inner_w: int,
    used_specs: list[dict],
) -> np.ndarray:
    out = inp.copy()
    out[frame_top + 1 : frame_top + 1 + inner_h, frame_left + 1 : frame_left + 1 + inner_w] = int(BG)
    for spec in used_specs:
        color = int(spec["color"])
        for rr, cc in spec["orig_abs"]:
            out[rr, cc] = int(BG)
        for rr, cc in spec["target_cells"]:
            out[frame_top + 1 + rr, frame_left + 1 + cc] = color
    return out


def _build_scene(
    inner_h: int,
    inner_w: int,
    used_shapes: list[tuple[tuple[int, int], ...]],
    distractor_shapes: list[tuple[tuple[int, int], ...]],
) -> tuple[np.ndarray, np.ndarray] | None:
    piece_count = len(used_shapes) + len(distractor_shapes)
    assert piece_count + 1 <= len(PALETTE)
    colors = random.sample(list(PALETTE), piece_count + 1)
    frame_color = colors[0]
    used_colors = colors[1 : 1 + len(used_shapes)]
    distractor_colors = colors[1 + len(used_shapes) :]

    grid_h = random.choice(list(GRID_HEIGHT_RANGE))
    grid_w = random.choice(list(GRID_WIDTH_RANGE))
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros_like(grid, dtype=bool)

    frame = _frame_patch(inner_h, inner_w, frame_color)
    frame_pos = _place_patch(grid, blocked, frame, padding=2)
    if frame_pos is None:
        return None
    frame_top, frame_left = frame_pos

    used_piece_dicts = [
        {
            "color": int(color),
            "shape": shape,
            "shape_h": _shape_dims(shape)[0],
            "shape_w": _shape_dims(shape)[1],
            "area": len(shape),
            "coords": shape,
        }
        for color, shape in zip(used_colors, used_shapes)
    ]
    used_solution = _solve_unique_tiling(used_piece_dicts, inner_h, inner_w, max_solutions=2)
    if len(used_solution) != 1:
        return None
    target_by_piece = {
        int(item["piece_idx"]): tuple(item["cells"])
        for item in used_solution[0]
    }

    used_specs = []
    patches = []
    for piece_idx, (shape, color) in enumerate(zip(used_shapes, used_colors)):
        patches.append(("used", piece_idx, shape, color, _shape_to_patch(shape, color)))
    for shape, color in zip(distractor_shapes, distractor_colors):
        patches.append(("distractor", None, shape, color, _shape_to_patch(shape, color)))
    random.shuffle(patches)
    patches.sort(key=lambda item: item[4].shape[0] * item[4].shape[1], reverse=True)

    for kind, piece_idx, shape, color, patch in patches:
        pos = _place_patch(grid, blocked, patch, padding=1)
        if pos is None:
            return None
        top, left = pos
        if kind == "used":
            used_specs.append(
                {
                    "shape": shape,
                    "color": int(color),
                    "orig_abs": tuple((top + r, left + c) for r, c in shape),
                    "target_cells": target_by_piece[int(piece_idx)],
                }
            )

    expected = _expected_output(grid, frame_top, frame_left, inner_h, inner_w, used_specs)
    return grid, expected


def _generator_constraints_ok(
    inner_h: int,
    inner_w: int,
    used_shapes: list[tuple[tuple[int, int], ...]],
    distractor_shapes: list[tuple[tuple[int, int], ...]],
) -> bool:
    used_pieces = [
        {"shape": shape, "shape_h": _shape_dims(shape)[0], "shape_w": _shape_dims(shape)[1], "area": len(shape)}
        for shape in used_shapes
    ]
    used_counts = [len(_placement_options(piece, inner_h, inner_w)) for piece in used_pieces]
    if sum(count > 1 for count in used_counts) < 2:
        return False
    if not any(_is_irregular(shape) for shape in used_shapes):
        return False

    used_dicts = [
        {"color": idx + 1, "shape": shape, "shape_h": _shape_dims(shape)[0], "shape_w": _shape_dims(shape)[1], "area": len(shape), "coords": shape}
        for idx, shape in enumerate(used_shapes)
    ]
    used_solutions = _solve_unique_tiling(used_dicts, inner_h, inner_w, max_solutions=2)
    if len(used_solutions) != 1:
        return False

    distractor_dicts = [
        {"color": len(used_shapes) + idx + 1, "shape": shape, "shape_h": _shape_dims(shape)[0], "shape_w": _shape_dims(shape)[1], "area": len(shape), "coords": shape}
        for idx, shape in enumerate(distractor_shapes)
    ]
    all_dicts = used_dicts + distractor_dicts
    all_solutions = _solve_unique_tiling(all_dicts, inner_h, inner_w, max_solutions=2)
    if len(all_solutions) != 1:
        return False
    used_set = set(range(len(used_shapes)))
    if {item["piece_idx"] for item in all_solutions[0]} != used_set:
        return False

    all_areas = [len(shape) for shape in used_shapes + distractor_shapes]
    if not _has_alternative_area_subset(all_areas, inner_h * inner_w, len(used_shapes)):
        return False

    for shape in distractor_shapes:
        piece = {"shape": shape, "shape_h": _shape_dims(shape)[0], "shape_w": _shape_dims(shape)[1], "area": len(shape)}
        if not _placement_options(piece, inner_h, inner_w):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    inner_h = random.choice(list(INNER_HEIGHT_RANGE))
    inner_w = random.choice(list(INNER_WIDTH_RANGE))
    target_area = inner_h * inner_w
    max_used = min(MAX_USED_PIECES, target_area // 2)
    used_count = random.randint(MIN_USED_PIECES, max_used)

    used_shapes = _random_partition(inner_h, inner_w, used_count)
    if used_shapes is None:
        return None

    replacement_areas = _area_replacement_plan([len(shape) for shape in used_shapes])
    if replacement_areas is None:
        return None

    extra_distractor_count = random.choice([0, 1])
    forbidden = set(used_shapes)
    distractor_shapes = []
    for area in replacement_areas:
        shape = _random_connected_shape(area, inner_h, inner_w, forbidden=forbidden)
        if shape is None:
            return None
        distractor_shapes.append(shape)
        forbidden.add(shape)

    for _ in range(extra_distractor_count):
        area = random.randint(2, max(2, min(target_area - 1, 6)))
        shape = _random_connected_shape(area, inner_h, inner_w, forbidden=forbidden)
        if shape is None:
            return None
        distractor_shapes.append(shape)
        forbidden.add(shape)

    if not _generator_constraints_ok(inner_h, inner_w, used_shapes, distractor_shapes):
        return None

    built = _build_scene(inner_h, inner_w, used_shapes, distractor_shapes)
    if built is None:
        return None
    inp, expected = built

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None

    bg, frame_bbox, _inner_shape, pieces = _extract_scene(inp)
    if len(pieces) != used_count + len(distractor_shapes):
        return None
    fr0, fc0, fr1, fc1 = frame_bbox
    if not np.all(inp[fr0 + 1 : fr1, fc0 + 1 : fc1] == bg):
        return None
    if not np.array_equal(actual[fr0 : fr1 + 1, fc0 : fc1 + 1][1:-1, 1:-1], expected[fr0 : fr1 + 1, fc0 : fc1 + 1][1:-1, 1:-1]):
        return None
    if np.array_equal(actual, inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
