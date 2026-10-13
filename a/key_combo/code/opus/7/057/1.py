from __future__ import annotations

import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5

PRIMITIVES = (
    {
        "kind": "square",
        "orientation": "square",
        "cells": ((0, 0), (0, 1), (1, 0), (1, 1)),
    },
    {
        "kind": "bar",
        "orientation": "horizontal",
        "cells": ((0, 0), (0, 1), (0, 2)),
    },
    {
        "kind": "bar",
        "orientation": "vertical",
        "cells": ((0, 0), (1, 0), (2, 0)),
    },
)

NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
NEIGHBORS8 = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)

GRID_HEIGHT_RANGE = (11, 14)
GRID_WIDTH_RANGE = (13, 17)
COMPONENT_COUNT_RANGE = (3, 5)
MAX_GENERATION_ATTEMPTS = 400
MAX_LIBRARY_BBOX = 5


def _bbox(cells: tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    return min(rs), min(cs), max(rs), max(cs)


def _normalize(cells: tuple[tuple[int, int], ...] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox(cells)
    return tuple(sorted((r - r0, c - c0) for r, c in cells))


def _translate(cells: tuple[tuple[int, int], ...], dr: int, dc: int) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((r + dr, c + dc) for r, c in cells))


def _touches(existing: set[tuple[int, int]], placed: set[tuple[int, int]]) -> bool:
    for r, c in placed:
        for dr, dc in NEIGHBORS4:
            if (r + dr, c + dc) in existing:
                return True
    return False


def _find_components(mask: np.ndarray) -> list[tuple[tuple[int, int], ...]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[tuple[int, int], ...]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(tuple(sorted(cells)))
    return components


def _candidate_placements(
    cells: tuple[tuple[int, int], ...] | set[tuple[int, int]]
) -> list[dict[str, object]]:
    cell_set = set(cells)
    placements: dict[tuple[tuple[int, int], ...], dict[str, object]] = {}

    for primitive in PRIMITIVES:
        shape = primitive["cells"]
        for anchor_r, anchor_c in cell_set:
            for base_r, base_c in shape:
                dr = anchor_r - base_r
                dc = anchor_c - base_c
                placed = tuple(sorted((r + dr, c + dc) for r, c in shape))
                if set(placed) <= cell_set:
                    placements[placed] = {
                        "kind": primitive["kind"],
                        "orientation": primitive["orientation"],
                        "cells": placed,
                        "bbox": _bbox(placed),
                    }
    return list(placements.values())


def _decompose_unique(
    cells: tuple[tuple[int, int], ...] | set[tuple[int, int]],
    max_solutions: int = 2,
) -> list[dict[str, object]] | None:
    normalized_cells = tuple(sorted(cells))
    cell_set = set(normalized_cells)
    placements = _candidate_placements(normalized_cells)
    cell_to_candidates: dict[tuple[int, int], list[int]] = {cell: [] for cell in normalized_cells}
    for idx, placement in enumerate(placements):
        for cell in placement["cells"]:
            cell_to_candidates[cell].append(idx)

    solutions: list[list[dict[str, object]]] = []

    def search(remaining: set[tuple[int, int]], chosen: list[int]) -> None:
        if len(solutions) >= max_solutions:
            return
        if not remaining:
            solutions.append([placements[idx] for idx in chosen])
            return

        cell = min(
            remaining,
            key=lambda rc: sum(
                1 for idx in cell_to_candidates[rc] if set(placements[idx]["cells"]) <= remaining
            ),
        )
        options = [
            idx for idx in cell_to_candidates[cell] if set(placements[idx]["cells"]) <= remaining
        ]
        if not options:
            return

        for idx in options:
            placement_cells = set(placements[idx]["cells"])
            chosen.append(idx)
            search(remaining - placement_cells, chosen)
            chosen.pop()

    search(set(cell_set), [])
    if len(solutions) != 1:
        return None

    solution = solutions[0]
    solution.sort(key=lambda piece: piece["bbox"])
    return solution


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decompose each gray composite into primitive pieces, then slide squares left and bars right.

    Input structure:
    - The grid is a black background with several separated gray 4-connected regions.
    - Each gray region is a seamless union of 2x2 squares and 1x3 or 3x1 bars.
    - The decomposition of every region into those primitive pieces is unique.

    Rule:
    - Decompose every gray region into non-overlapping primitive pieces.
    - Clear the canvas to black.
    - Slide each 2x2 square horizontally to the left side and recolor it red.
    - Slide each length-three bar horizontally to the right side and recolor it blue.
    - Pieces moving to the same side stack by original distance to that side, with ties broken by bbox top-left; each row keeps its own occupied depth.

    Key insight:
    - The recovered primitive shape is the movement code: squares mean "go left" while bars mean "go right".
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    h, w = g.shape
    components = _find_components(g == GREY)
    assert components, "expected gray foreground components"

    pieces: list[dict[str, object]] = []
    for component in components:
        decomposition = _decompose_unique(component)
        assert decomposition is not None, "each component must have a unique valid decomposition"
        pieces.extend(decomposition)

    left_pieces = [piece for piece in pieces if piece["kind"] == "square"]
    right_pieces = [piece for piece in pieces if piece["kind"] == "bar"]

    left_pieces.sort(key=lambda piece: (piece["bbox"][1], piece["bbox"][0], piece["bbox"][1]))
    right_pieces.sort(
        key=lambda piece: (w - 1 - piece["bbox"][3], piece["bbox"][0], piece["bbox"][1])
    )

    out = np.full((h, w), BLACK, dtype=int)
    left_depth = [0] * h
    right_depth = [0] * h

    def place(piece: dict[str, object], side: str, color: int) -> None:
        r0, c0, r1, c1 = piece["bbox"]
        rows = range(r0, r1 + 1)
        width = c1 - c0 + 1
        if side == "left":
            depth = max(left_depth[r] for r in rows)
            dst_c0 = depth
            for r in rows:
                left_depth[r] = depth + width
        else:
            depth = max(right_depth[r] for r in rows)
            dst_c0 = w - width - depth
            for r in rows:
                right_depth[r] = depth + width

        shift = dst_c0 - c0
        for r, c in piece["cells"]:
            nc = c + shift
            assert 0 <= nc < w
            assert out[r, nc] == BLACK, "shifted pieces must not overlap"
            out[r, nc] = color

    for piece in left_pieces:
        place(piece, "left", RED)
    for piece in right_pieces:
        place(piece, "right", BLUE)

    return out


def _build_component_library() -> list[dict[str, object]]:
    seen: set[tuple[int, tuple[tuple[int, int], ...]]] = set()
    unions: set[tuple[tuple[int, int], ...]] = set()

    def explore(cells: tuple[tuple[int, int], ...], piece_count: int) -> None:
        key = _normalize(cells)
        state = (piece_count, key)
        if state in seen:
            return
        seen.add(state)
        unions.add(key)
        if piece_count == 3:
            return

        existing = set(key)
        for primitive in PRIMITIVES:
            shape = primitive["cells"]
            for dr in range(-4, 5):
                for dc in range(-4, 5):
                    placed = {(r + dr, c + dc) for r, c in shape}
                    if existing & placed:
                        continue
                    if len(existing) + len(placed) > 10:
                        continue
                    if not _touches(existing, placed):
                        continue
                    new_cells = tuple(sorted(existing | placed))
                    r0, c0, r1, c1 = _bbox(new_cells)
                    if r1 - r0 + 1 > MAX_LIBRARY_BBOX or c1 - c0 + 1 > MAX_LIBRARY_BBOX:
                        continue
                    explore(_normalize(new_cells), piece_count + 1)

    for primitive in PRIMITIVES:
        explore(tuple(primitive["cells"]), 1)

    library: list[dict[str, object]] = []
    for cells in sorted(unions):
        solution = _decompose_unique(cells)
        if solution is None:
            continue
        r0, c0, r1, c1 = _bbox(cells)
        if r1 - r0 + 1 > MAX_LIBRARY_BBOX or c1 - c0 + 1 > MAX_LIBRARY_BBOX:
            continue

        square_count = sum(piece["kind"] == "square" for piece in solution)
        bar_count = sum(piece["kind"] == "bar" for piece in solution)
        has_horizontal = any(piece["orientation"] == "horizontal" for piece in solution)
        has_vertical = any(piece["orientation"] == "vertical" for piece in solution)

        library.append(
            {
                "cells": tuple(sorted(cells)),
                "pieces": solution,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
                "area": len(cells),
                "piece_count": len(solution),
                "square_count": square_count,
                "bar_count": bar_count,
                "has_horizontal": has_horizontal,
                "has_vertical": has_vertical,
            }
        )

    return library


COMPONENT_LIBRARY = _build_component_library()
MIXED_COMPONENTS = [
    entry
    for entry in COMPONENT_LIBRARY
    if entry["square_count"] >= 1 and entry["bar_count"] >= 1
]
MIXED_HORIZONTAL = [entry for entry in MIXED_COMPONENTS if entry["has_horizontal"]]
MIXED_VERTICAL = [entry for entry in MIXED_COMPONENTS if entry["has_vertical"]]
COMPOSITE_COMPONENTS = [entry for entry in COMPONENT_LIBRARY if entry["piece_count"] >= 2]


def _candidate_positions(
    entry: dict[str, object],
    half: str,
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    mid = w // 2
    height = int(entry["height"])
    width = int(entry["width"])

    if half == "left":
        col_start, col_end = 0, mid - width
    else:
        col_start, col_end = mid, w - width
    if col_end < col_start:
        return []

    positions: list[tuple[int, int]] = []
    for r0 in range(0, h - height + 1):
        for c0 in range(col_start, col_end + 1):
            translated = _translate(entry["cells"], r0, c0)
            if _fits_with_moat(translated, h, w, occupied):
                positions.append((r0, c0))
    random.shuffle(positions)
    return positions


def _fits_with_moat(
    cells: tuple[tuple[int, int], ...],
    h: int,
    w: int,
    occupied: set[tuple[int, int]],
) -> bool:
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        for dr, dc in NEIGHBORS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) in occupied:
                return False
        if (r, c) in occupied:
            return False
    return True


def _absolute_pieces(
    entry: dict[str, object], r0: int, c0: int
) -> list[dict[str, object]]:
    absolute: list[dict[str, object]] = []
    for piece in entry["pieces"]:
        translated_cells = _translate(piece["cells"], r0, c0)
        absolute.append(
            {
                "kind": piece["kind"],
                "orientation": piece["orientation"],
                "cells": translated_cells,
                "bbox": _bbox(translated_cells),
            }
        )
    return absolute


def _sample_component(composite_bias: float) -> dict[str, object]:
    if random.random() < composite_bias:
        return random.choice(COMPOSITE_COMPONENTS)
    return random.choice(COMPONENT_LIBRARY)


def _valid_primitive_margins(pieces: list[dict[str, object]], width: int) -> bool:
    for piece in pieces:
        _r0, c0, _r1, c1 = piece["bbox"]
        if piece["kind"] == "square" and c0 == 0:
            return False
        if piece["kind"] == "bar" and c1 == width - 1:
            return False
    return True


def _half_features(
    pieces: list[dict[str, object]], width: int
) -> dict[str, set[str]]:
    mid = width // 2
    features = {"left": set(), "right": set()}
    for piece in pieces:
        _r0, c0, _r1, c1 = piece["bbox"]
        if c1 < mid:
            side = "left"
        elif c0 >= mid:
            side = "right"
        else:
            return {}
        features[side].add(str(piece["kind"]))
    return features


def _attempt_generate() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    n_components = random.randint(*COMPONENT_COUNT_RANGE)

    if random.random() < 0.5:
        mandatory = [
            (random.choice(MIXED_HORIZONTAL), "left"),
            (random.choice(MIXED_VERTICAL), "right"),
        ]
    else:
        mandatory = [
            (random.choice(MIXED_VERTICAL), "left"),
            (random.choice(MIXED_HORIZONTAL), "right"),
        ]

    components_to_place = mandatory[:]
    for _ in range(n_components - 2):
        half = random.choice(("left", "right"))
        components_to_place.append((_sample_component(0.7), half))
    components_to_place.sort(key=lambda item: (-item[0]["area"], -item[0]["piece_count"]))

    occupied: set[tuple[int, int]] = set()
    placements: list[tuple[dict[str, object], int, int]] = []
    for entry, half in components_to_place:
        positions = _candidate_positions(entry, half, h, w, occupied)
        if not positions:
            return None
        r0, c0 = positions[0]
        translated = _translate(entry["cells"], r0, c0)
        placements.append((entry, r0, c0))
        occupied.update(translated)

    grid = np.full((h, w), BLACK, dtype=int)
    all_pieces: list[dict[str, object]] = []
    for entry, r0, c0 in placements:
        translated = _translate(entry["cells"], r0, c0)
        for r, c in translated:
            grid[r, c] = GREY
        all_pieces.extend(_absolute_pieces(entry, r0, c0))

    if len(_find_components(grid == GREY)) != n_components:
        return None
    if not _valid_primitive_margins(all_pieces, w):
        return None

    features = _half_features(all_pieces, w)
    if not features:
        return None
    if features["left"] != {"square", "bar"} or features["right"] != {"square", "bar"}:
        return None

    for component in _find_components(grid == GREY):
        if _decompose_unique(component) is None:
            return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if not np.any(out == RED) or not np.any(out == BLUE):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 1")
