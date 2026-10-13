from __future__ import annotations

import random
from functools import lru_cache

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
ORANGE = 7
TEAL = 8

PIECE_TEMPLATES = {
    "square": ((0, 0), (0, 1), (1, 0), (1, 1)),
    "bar_h": ((0, 0), (0, 1), (0, 2)),
    "bar_v": ((0, 0), (1, 0), (2, 0)),
}
PIECE_DIMS = {"square": (2, 2), "bar_h": (1, 3), "bar_v": (3, 1)}
ORTHO_DIRS = ((1, 0), (-1, 0), (0, 1), (0, -1))

GRID_HEIGHT_RANGE = (18, 25)
GRID_WIDTH_RANGE = (18, 25)
COMPONENT_COUNT_CHOICES = (1, 2, 3)
COMPONENT_COUNT_WEIGHTS = (1, 5, 3)
TOTAL_PIECES_RANGE = (4, 7)
COMPONENT_PIECES_MIN = 2
COMPONENT_PIECES_MAX = 4
COMPONENT_MAX_SPAN = 8
COMPONENT_SEARCH_MARGIN = 3
COMPONENT_ATTEMPTS = 250
SCENE_ATTEMPTS = 400
GRID_BORDER = 2
COMPONENT_PADDING = 2
OBSTACLE_COUNT_RANGE = (3, 6)
RAY_FILL_MAX_FRACTION = 0.33


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _bbox_coords(coords: np.ndarray | list[tuple[int, int]]) -> tuple[int, int, int, int]:
    arr = np.asarray(coords, dtype=int)
    r0, c0 = arr.min(axis=0)
    r1, c1 = arr.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in ORTHO_DIRS:
                nr, nc = rr + dr, cc + dc
                if _in_bounds((h, w), nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        components.append(np.asarray(coords, dtype=int))
    return components


def _piece_cells(kind: str, r0: int, c0: int) -> tuple[tuple[int, int], ...]:
    return tuple((r0 + dr, c0 + dc) for dr, dc in PIECE_TEMPLATES[kind])


def _piece_sort_key(piece: dict) -> tuple[int, int]:
    return int(piece["r0"]), int(piece["c0"])


def _piece_group(piece: dict) -> str:
    return "square" if piece["kind"] == "square" else "bar"


def _piece_emitters(piece: dict) -> list[tuple[int, int, int, int]]:
    r0 = int(piece["r0"])
    c0 = int(piece["c0"])
    kind = piece["kind"]
    if kind == "square":
        return [
            (r0 - 1, c0 - 1, -1, -1),
            (r0 - 1, c0 + 2, -1, 1),
            (r0 + 2, c0 - 1, 1, -1),
            (r0 + 2, c0 + 2, 1, 1),
        ]
    if kind == "bar_h":
        return [(r0, c0 - 1, 0, -1), (r0, c0 + 3, 0, 1)]
    if kind == "bar_v":
        return [(r0 - 1, c0, -1, 0), (r0 + 3, c0, 1, 0)]
    raise ValueError(f"unknown piece kind: {kind}")


def _mask_from_pieces(pieces: list[dict]) -> np.ndarray:
    cells = [cell for piece in pieces for cell in _piece_cells(piece["kind"], int(piece["r0"]), int(piece["c0"]))]
    r0, c0, r1, c1 = _bbox_coords(cells)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in cells:
        out[r - r0, c - c0] = True
    return out


def _enumerate_component_placements(mask: np.ndarray) -> tuple[list[tuple[int, int]], list[dict]]:
    h, w = mask.shape
    cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    cell_index = {cell: idx for idx, cell in enumerate(cells)}
    placements: list[dict] = []
    for kind, offsets in PIECE_TEMPLATES.items():
        max_r = max(dr for dr, _ in offsets)
        max_c = max(dc for _, dc in offsets)
        for r0 in range(h - max_r):
            for c0 in range(w - max_c):
                placed = [(r0 + dr, c0 + dc) for dr, dc in offsets]
                if not all(mask[r, c] for r, c in placed):
                    continue
                bit = 0
                for cell in placed:
                    bit |= 1 << cell_index[cell]
                placements.append({"kind": kind, "r0": int(r0), "c0": int(c0), "bit": bit})
    return cells, placements


def _solve_exact_cover(mask: np.ndarray, limit: int = 2) -> tuple[int, list[dict]]:
    cells, placements = _enumerate_component_placements(mask)
    if not cells:
        return 0, []

    cell_to_placements = [[] for _ in range(len(cells))]
    for idx, placement in enumerate(placements):
        bit = placement["bit"]
        for cell_idx in range(len(cells)):
            if (bit >> cell_idx) & 1:
                cell_to_placements[cell_idx].append(idx)

    full = (1 << len(cells)) - 1

    @lru_cache(None)
    def dfs(remaining: int) -> tuple[int, tuple[int, ...]]:
        if remaining == 0:
            return 1, ()

        best_options: tuple[int, ...] | None = None
        probe = remaining
        while probe:
            lsb = probe & -probe
            cell_idx = lsb.bit_length() - 1
            valid = tuple(
                placement_idx
                for placement_idx in cell_to_placements[cell_idx]
                if placements[placement_idx]["bit"] & remaining == placements[placement_idx]["bit"]
            )
            if not valid:
                return 0, ()
            if best_options is None or len(valid) < len(best_options):
                best_options = valid
                if len(best_options) == 1:
                    break
            probe ^= lsb

        total = 0
        example: tuple[int, ...] = ()
        assert best_options is not None
        for placement_idx in best_options:
            sub_count, sub_solution = dfs(remaining ^ placements[placement_idx]["bit"])
            if sub_count == 0:
                continue
            if total == 0:
                example = (placement_idx,) + sub_solution
            total = min(limit, total + sub_count)
            if total >= limit:
                break
        return total, example

    count, solution_indices = dfs(full)
    solved = []
    for placement_idx in solution_indices:
        solved.append(
            {
                "kind": placements[placement_idx]["kind"],
                "r0": placements[placement_idx]["r0"],
                "c0": placements[placement_idx]["c0"],
            }
        )
    return count, solved


def _decompose_gray_objects(grid: np.ndarray) -> tuple[list[dict], list[list[dict]]]:
    gray_mask = grid == GREY
    components = _components_from_mask(gray_mask)
    assert components, "expected at least one gray component"

    all_pieces: list[dict] = []
    pieces_by_component: list[list[dict]] = []
    for coords in components:
        r0, c0, r1, c1 = _bbox_coords(coords)
        local_mask = gray_mask[r0 : r1 + 1, c0 : c1 + 1]
        count, local_pieces = _solve_exact_cover(local_mask, limit=2)
        assert count == 1, "gray component must have a unique decomposition"
        assert len(local_pieces) >= 2, "each gray component must contain at least two pieces"
        shifted = []
        for piece in local_pieces:
            shifted_piece = {
                "kind": piece["kind"],
                "r0": int(piece["r0"]) + r0,
                "c0": int(piece["c0"]) + c0,
            }
            shifted.append(shifted_piece)
            all_pieces.append(shifted_piece)
        pieces_by_component.append(shifted)
    return all_pieces, pieces_by_component


def _render_scene(
    shape: tuple[int, int],
    pieces: list[dict],
    obstacle_mask: np.ndarray,
    *,
    return_meta: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[dict]]:
    out = np.zeros(shape, dtype=int)
    out[obstacle_mask] = ORANGE

    for piece in pieces:
        color = TEAL if piece["kind"] == "square" else RED
        for r, c in _piece_cells(piece["kind"], int(piece["r0"]), int(piece["c0"])):
            out[r, c] = color

    ray_meta: list[dict] = []
    square_pieces = sorted((piece for piece in pieces if piece["kind"] == "square"), key=_piece_sort_key)
    bar_pieces = sorted((piece for piece in pieces if piece["kind"] != "square"), key=_piece_sort_key)
    for piece in square_pieces + bar_pieces:
        source_kind = _piece_group(piece)
        for start_r, start_c, dr, dc in _piece_emitters(piece):
            painted_cells: list[tuple[int, int]] = []
            painted_colors: list[int] = []
            r, c = start_r, start_c
            while _in_bounds(shape, r, c):
                value = int(out[r, c])
                if value == BLACK:
                    out[r, c] = BLUE
                    painted_cells.append((r, c))
                    painted_colors.append(BLUE)
                elif value == ORANGE:
                    out[r, c] = YELLOW
                    painted_cells.append((r, c))
                    painted_colors.append(YELLOW)
                else:
                    break
                r += dr
                c += dc
            ray_meta.append(
                {
                    "source_kind": source_kind,
                    "piece_kind": piece["kind"],
                    "piece_anchor": (int(piece["r0"]), int(piece["c0"])),
                    "direction": (int(dr), int(dc)),
                    "cells": tuple(painted_cells),
                    "colors": tuple(painted_colors),
                }
            )

    if return_meta:
        return out, ray_meta
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose gray composites into squares and bars, recolor them, then cast shape-dependent rays that turn orange barrier crossings yellow.

    Input structure:
    - The grid is a black background with one or more 4-connected gray composite regions.
    - Each gray region can be uniquely partitioned into non-overlapping 2x2 squares and 1x3 or 3x1 bars.
    - Separate orange horizontal or vertical line segments lie away from every gray pixel.

    Rule:
    - Uniquely decompose every gray region into the allowed primitive pieces.
    - Recolor each square teal and each bar red.
    - For every teal square, cast four outward diagonal rays from just beyond its four corners.
    - For every red bar, cast two outward cardinal rays from just beyond its two ends along the bar axis.
    - Rays paint blue on background cells, paint yellow when they pass through orange barrier cells, and keep extending afterward.
    - If a ray would enter any already non-background non-orange cell, it stops before that cell.
    - Ray order is all squares first by top-left bbox corner, then all bars by the same ordering.

    Key insight:
    - The same hidden decomposition determines both the recoloring and the ray geometry, so a wrong square-vs-bar decision breaks both the painted primitive colors and the later yellow-marked ray crossings.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    pieces, _ = _decompose_gray_objects(g)
    obstacle_mask = g == ORANGE
    return _render_scene(g.shape, pieces, obstacle_mask)


def _shift_pieces(pieces: list[dict], dr: int, dc: int) -> list[dict]:
    return [{"kind": piece["kind"], "r0": int(piece["r0"]) + dr, "c0": int(piece["c0"]) + dc} for piece in pieces]


def _transform_mask(mask: np.ndarray, variant: int) -> np.ndarray:
    variant = int(variant) % 8
    out = mask.copy()
    if variant >= 4:
        out = np.fliplr(out)
        variant -= 4
    return np.rot90(out, k=variant)


def _sample_component_sizes(rng: random.Random) -> list[int]:
    n_components = rng.choices(COMPONENT_COUNT_CHOICES, weights=COMPONENT_COUNT_WEIGHTS, k=1)[0]
    total_lo = max(TOTAL_PIECES_RANGE[0], n_components * COMPONENT_PIECES_MIN)
    total_hi = min(TOTAL_PIECES_RANGE[1], n_components * COMPONENT_PIECES_MAX)
    total_pieces = rng.randint(total_lo, total_hi)

    sizes = []
    remaining = total_pieces
    for idx in range(n_components):
        slots_left = n_components - idx - 1
        lo = max(COMPONENT_PIECES_MIN, remaining - slots_left * COMPONENT_PIECES_MAX)
        hi = min(COMPONENT_PIECES_MAX, remaining - slots_left * COMPONENT_PIECES_MIN)
        sizes.append(rng.randint(lo, hi))
        remaining -= sizes[-1]
    rng.shuffle(sizes)
    return sizes


def _generate_local_component(n_pieces: int, rng: random.Random) -> list[dict] | None:
    for _ in range(COMPONENT_ATTEMPTS):
        pieces = [{"kind": rng.choices(["square", "bar_h", "bar_v"], weights=[4, 3, 3], k=1)[0], "r0": 0, "c0": 0}]
        while len(pieces) < n_pieces:
            occupied = {
                cell
                for piece in pieces
                for cell in _piece_cells(piece["kind"], int(piece["r0"]), int(piece["c0"]))
            }
            r0, c0, r1, c1 = _bbox_coords(list(occupied))
            candidates = []
            for kind in PIECE_TEMPLATES:
                for pr in range(r0 - COMPONENT_SEARCH_MARGIN, r1 + COMPONENT_SEARCH_MARGIN + 1):
                    for pc in range(c0 - COMPONENT_SEARCH_MARGIN, c1 + COMPONENT_SEARCH_MARGIN + 1):
                        cells = _piece_cells(kind, pr, pc)
                        if any(cell in occupied for cell in cells):
                            continue
                        if not any((r + dr, c + dc) in occupied for r, c in cells for dr, dc in ORTHO_DIRS):
                            continue
                        union = list(occupied) + list(cells)
                        ur0, uc0, ur1, uc1 = _bbox_coords(union)
                        if ur1 - ur0 + 1 > COMPONENT_MAX_SPAN or uc1 - uc0 + 1 > COMPONENT_MAX_SPAN:
                            continue
                        candidates.append({"kind": kind, "r0": pr, "c0": pc})
            if not candidates:
                break
            pieces.append(rng.choice(candidates))

        if len(pieces) != n_pieces:
            continue

        mask = _mask_from_pieces(pieces)
        count, solved = _solve_exact_cover(mask, limit=2)
        if count != 1 or len(solved) != n_pieces:
            continue

        transformed_mask = _transform_mask(_mask_from_pieces(solved), rng.randrange(8))
        count, solved = _solve_exact_cover(transformed_mask, limit=2)
        if count != 1 or len(solved) != n_pieces:
            continue
        return solved
    return None


def _find_component_positions(
    shape: tuple[int, int],
    component_pieces: list[dict],
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    h, w = shape
    mask = _mask_from_pieces(component_pieces)
    comp_h, comp_w = mask.shape
    local_cells = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    positions = []
    for r0 in range(GRID_BORDER, h - GRID_BORDER - comp_h + 1):
        for c0 in range(GRID_BORDER, w - GRID_BORDER - comp_w + 1):
            shifted = [(r0 + r, c0 + c) for r, c in local_cells]
            if any(cell in occupied for cell in shifted):
                continue
            if any(
                max(abs(r - rr), abs(c - cc)) <= COMPONENT_PADDING
                for r, c in shifted
                for rr, cc in occupied
            ):
                continue
            positions.append((r0, c0))
    return positions


def _gray_forbidden(shape: tuple[int, int], pieces: list[dict]) -> set[tuple[int, int]]:
    h, w = shape
    forbidden: set[tuple[int, int]] = set()
    for piece in pieces:
        for r, c in _piece_cells(piece["kind"], int(piece["r0"]), int(piece["c0"])):
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w:
                        forbidden.add((nr, nc))
    return forbidden


def _segment_options_through(
    cell: tuple[int, int],
    orientation: str,
    shape: tuple[int, int],
    rng: random.Random,
) -> list[list[tuple[int, int]]]:
    r, c = cell
    h, w = shape
    segments = []
    lengths = [2, 3, 4]
    rng.shuffle(lengths)
    for length in lengths:
        if orientation == "h":
            starts = list(range(c - length + 1, c + 1))
            rng.shuffle(starts)
            for c0 in starts:
                if 0 <= c0 and c0 + length <= w:
                    segments.append([(r, cc) for cc in range(c0, c0 + length)])
        else:
            starts = list(range(r - length + 1, r + 1))
            rng.shuffle(starts)
            for r0 in starts:
                if 0 <= r0 and r0 + length <= h:
                    segments.append([(rr, c) for rr in range(r0, r0 + length)])
    return segments


def _segment_is_clear(
    segment: list[tuple[int, int]],
    blocked_for_obstacles: set[tuple[int, int]],
) -> bool:
    return all(cell not in blocked_for_obstacles for cell in segment)


def _extend_obstacle_blocked(
    shape: tuple[int, int],
    blocked_for_obstacles: set[tuple[int, int]],
    segment: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    h, w = shape
    out = set(blocked_for_obstacles)
    for r, c in segment:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _candidate_hit_segment(
    rays: list[dict],
    desired_group: str,
    shape: tuple[int, int],
    route_cells: set[tuple[int, int]],
    blocked_for_obstacles: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    candidate_rays = [ray for ray in rays if ray["source_kind"] == desired_group and ray["cells"]]
    rng.shuffle(candidate_rays)
    for ray in candidate_rays:
        cells = list(ray["cells"])
        preferred = cells[1:-1] or cells
        probe_cells = preferred + [cell for cell in cells if cell not in preferred]
        rng.shuffle(probe_cells)
        dr, dc = ray["direction"]
        if desired_group == "square":
            orientations = ["h", "v"]
            rng.shuffle(orientations)
        elif dr == 0:
            orientations = ["v"]
        else:
            orientations = ["h"]

        for cell in probe_cells:
            for orientation in orientations:
                for segment in _segment_options_through(cell, orientation, shape, rng):
                    if cell not in segment:
                        continue
                    if sum(seg_cell in route_cells for seg_cell in segment) != 1:
                        continue
                    if not _segment_is_clear(segment, blocked_for_obstacles):
                        continue
                    return segment
    return None


def _candidate_miss_segment(
    shape: tuple[int, int],
    route_cells: set[tuple[int, int]],
    blocked_for_obstacles: set[tuple[int, int]],
    rng: random.Random,
) -> list[tuple[int, int]] | None:
    h, w = shape
    candidates = []
    for orientation in ("h", "v"):
        for length in (2, 3, 4):
            if orientation == "h":
                for r in range(h):
                    for c0 in range(w - length + 1):
                        segment = [(r, c) for c in range(c0, c0 + length)]
                        if any(cell in route_cells for cell in segment):
                            continue
                        if _segment_is_clear(segment, blocked_for_obstacles):
                            candidates.append(segment)
            else:
                for c in range(w):
                    for r0 in range(h - length + 1):
                        segment = [(r, c) for r in range(r0, r0 + length)]
                        if any(cell in route_cells for cell in segment):
                            continue
                        if _segment_is_clear(segment, blocked_for_obstacles):
                            candidates.append(segment)
    if not candidates:
        return None
    return rng.choice(candidates)


def _segments_to_mask(shape: tuple[int, int], segments: list[list[tuple[int, int]]]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for segment in segments:
        for r, c in segment:
            mask[r, c] = True
    return mask


def _build_input(shape: tuple[int, int], pieces: list[dict], segments: list[list[tuple[int, int]]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for piece in pieces:
        for r, c in _piece_cells(piece["kind"], int(piece["r0"]), int(piece["c0"])):
            grid[r, c] = GREY
    for segment in segments:
        for r, c in segment:
            grid[r, c] = ORANGE
    return grid


def _scene_is_semantically_valid(
    inp: np.ndarray,
    expected: np.ndarray,
    rays: list[dict],
    pieces_by_component: list[list[dict]],
    segments: list[list[tuple[int, int]]],
) -> bool:
    if not np.array_equal(main(inp), expected):
        return False

    all_pieces = [piece for component in pieces_by_component for piece in component]
    kinds = {piece["kind"] for piece in all_pieces}
    if "square" not in kinds or "bar_h" not in kinds or "bar_v" not in kinds:
        return False
    if not any(
        any(piece["kind"] == "square" for piece in component)
        and any(piece["kind"] != "square" for piece in component)
        for component in pieces_by_component
    ):
        return False

    yellow_by_group = {"square": 0, "bar": 0}
    total_ray_cells = 0
    for ray in rays:
        total_ray_cells += len(ray["cells"])
        for color in ray["colors"]:
            if color == YELLOW:
                yellow_by_group[ray["source_kind"]] += 1

    if yellow_by_group["square"] == 0 or yellow_by_group["bar"] == 0:
        return False
    if total_ray_cells < 12:
        return False
    if np.count_nonzero(np.isin(expected, (BLUE, YELLOW))) > int(expected.size * RAY_FILL_MAX_FRACTION):
        return False
    if not np.any(expected == BLUE) or not np.any(expected == YELLOW):
        return False
    if not np.any(expected == TEAL) or not np.any(expected == RED):
        return False

    unhit_segments = 0
    for segment in segments:
        if all(expected[r, c] == ORANGE for r, c in segment):
            unhit_segments += 1
    if unhit_segments == 0:
        return False

    _, component_decompositions = _decompose_gray_objects(inp)
    if len(component_decompositions) != len(pieces_by_component):
        return False
    return True


def _generate_scene(rng: random.Random) -> np.ndarray | None:
    component_sizes = _sample_component_sizes(rng)
    local_components = []
    for size in component_sizes:
        component = _generate_local_component(size, rng)
        if component is None:
            return None
        local_components.append(component)

    shape = (
        rng.randint(*GRID_HEIGHT_RANGE),
        rng.randint(*GRID_WIDTH_RANGE),
    )
    occupied: set[tuple[int, int]] = set()
    placed_components: list[list[dict]] = []
    for component in local_components:
        positions = _find_component_positions(shape, component, occupied)
        if not positions:
            return None
        r0, c0 = rng.choice(positions)
        shifted = _shift_pieces(component, r0, c0)
        placed_components.append(shifted)
        for piece in shifted:
            occupied.update(_piece_cells(piece["kind"], int(piece["r0"]), int(piece["c0"])))

    pieces = [piece for component in placed_components for piece in component]
    empty_obstacles = np.zeros(shape, dtype=bool)
    _, no_obstacle_rays = _render_scene(shape, pieces, empty_obstacles, return_meta=True)
    route_cells = {cell for ray in no_obstacle_rays for cell in ray["cells"]}
    if not route_cells:
        return None

    blocked_for_obstacles = _gray_forbidden(shape, pieces)
    segments: list[list[tuple[int, int]]] = []
    for desired_group in ("square", "bar"):
        segment = _candidate_hit_segment(
            no_obstacle_rays,
            desired_group,
            shape,
            route_cells,
            blocked_for_obstacles,
            rng,
        )
        if segment is None:
            return None
        segments.append(segment)
        blocked_for_obstacles = _extend_obstacle_blocked(shape, blocked_for_obstacles, segment)

    miss_segment = _candidate_miss_segment(shape, route_cells, blocked_for_obstacles, rng)
    if miss_segment is None:
        return None
    segments.append(miss_segment)
    blocked_for_obstacles = _extend_obstacle_blocked(shape, blocked_for_obstacles, miss_segment)

    target_count = rng.randint(*OBSTACLE_COUNT_RANGE)
    for _ in range(target_count - len(segments)):
        if rng.random() < 0.45:
            desired_group = rng.choice(["square", "bar"])
            segment = _candidate_hit_segment(
                no_obstacle_rays,
                desired_group,
                shape,
                route_cells,
                blocked_for_obstacles,
                rng,
            )
        else:
            segment = _candidate_miss_segment(shape, route_cells, blocked_for_obstacles, rng)
        if segment is None:
            return None
        segments.append(segment)
        blocked_for_obstacles = _extend_obstacle_blocked(shape, blocked_for_obstacles, segment)

    obstacle_mask = _segments_to_mask(shape, segments)
    expected, rays = _render_scene(shape, pieces, obstacle_mask, return_meta=True)
    inp = _build_input(shape, pieces, segments)
    if not _scene_is_semantically_valid(inp, expected, rays, placed_components, segments):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(SCENE_ATTEMPTS):
        grid = _generate_scene(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
