import numpy as np


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _ordered_line(coords: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], str]:
    if len(coords) != 3:
        raise AssertionError("expected every object to use exactly 3 cells")

    rows = sorted({int(r) for r, _ in coords})
    cols = sorted({int(c) for _, c in coords})

    if len(rows) == 1 and cols == list(range(cols[0], cols[0] + 3)):
        return [(rows[0], int(c)) for c in cols], "h"
    if len(cols) == 1 and rows == list(range(rows[0], rows[0] + 3)):
        return [(int(r), cols[0]) for r in rows], "v"
    raise AssertionError("expected every object to be a straight 3-cell line")


def _component_records(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    height, width = grid.shape
    records: list[dict[str, object]] = []

    for coords in _find_components(grid != bg):
        ordered, orientation = _ordered_line(coords)
        colors = [int(grid[r, c]) for r, c in ordered]
        border_line = (
            orientation == "h" and ordered[0][0] in (0, height - 1)
        ) or (
            orientation == "v" and ordered[0][1] in (0, width - 1)
        )
        records.append(
            {
                "ordered": ordered,
                "colors": colors,
                "orientation": orientation,
                "start": ordered[0],
                "border_line": border_line,
            }
        )

    return records


def _infer_off_color(lights: list[dict[str, object]]) -> int:
    total_counts: dict[int, int] = {}
    light_counts: dict[int, int] = {}

    for light in lights:
        seen_here: set[int] = set()
        for color in light["colors"]:
            color = int(color)
            total_counts[color] = total_counts.get(color, 0) + 1
            if color not in seen_here:
                light_counts[color] = light_counts.get(color, 0) + 1
                seen_here.add(color)

    best_score = max((total_counts[color], light_counts[color]) for color in total_counts)
    candidates = [
        color
        for color in total_counts
        if (total_counts[color], light_counts[color]) == best_score
    ]
    assert len(candidates) == 1, "off color must be structurally identifiable"
    return int(candidates[0])


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort traffic lights by the number of lit cells and recolor them by palette rank.

    Input structure:
    - Background is black.
    - There is one edge palette: a 3-cell horizontal or vertical strip of three colors.
    - The remaining objects are 3-cell horizontal or vertical traffic lights.
    - In each traffic light, grey cells are off and non-grey cells are lit.

    Rule:
    - Count the lit cells in each traffic light.
    - Sort traffic lights from fewest lit cells to most.
    - Recolor every lit cell in the k-th traffic light with the k-th palette color.
    - Keep off grey cells and all positions unchanged.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    records = _component_records(g, bg)

    palette_candidates = [
        record
        for record in records
        if bool(record["border_line"]) and len(set(int(color) for color in record["colors"])) == 3
    ]
    assert len(palette_candidates) == 1, "expected exactly one edge palette"
    palette = palette_candidates[0]

    lights = [record for record in records if record is not palette]
    assert 1 <= len(lights) <= len(palette["colors"]), "palette must be long enough for all lights"

    off_color = _infer_off_color(lights)
    ranked_lights = []
    for light in lights:
        lit_count = sum(int(color) != off_color for color in light["colors"])
        ranked_lights.append(
            (
                int(lit_count),
                int(light["start"][0]),
                int(light["start"][1]),
                light,
            )
        )
    ranked_lights.sort()

    out = g.copy()
    for rank, (_lit_count, _r, _c, light) in enumerate(ranked_lights):
        new_color = int(palette["colors"][rank])
        for (r, c), color in zip(light["ordered"], light["colors"]):
            if int(color) != off_color:
                out[r, c] = new_color

    return out


_RNG = np.random.default_rng()
_BG = 0
_OFF = 5
_EDGE_OPTIONS = ("top", "bottom", "left", "right")
_ORIENTATIONS = ("h", "v")
_COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)
_VALID_COUNT_SETS = {
    2: ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3)),
    3: ((0, 1, 2), (0, 1, 3), (0, 2, 3), (1, 2, 3)),
}
_GEN_ATTEMPTS = 400


def _palette_cells(height: int, width: int, edge: str, start: int) -> list[tuple[int, int]]:
    if edge == "top":
        return [(0, start + dc) for dc in range(3)]
    if edge == "bottom":
        return [(height - 1, start + dc) for dc in range(3)]
    if edge == "left":
        return [(start + dr, 0) for dr in range(3)]
    return [(start + dr, width - 1) for dr in range(3)]


def _line_cells(orientation: str, start_r: int, start_c: int) -> list[tuple[int, int]]:
    if orientation == "h":
        return [(start_r, start_c + dc) for dc in range(3)]
    return [(start_r + dr, start_c) for dr in range(3)]


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for dr in (-1, 0, 1):
            nr = int(r + dr)
            if not (0 <= nr < height):
                continue
            for dc in (-1, 0, 1):
                nc = int(c + dc)
                if 0 <= nc < width:
                    blocked[nr, nc] = True


def _clear_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    return all(not blocked[r, c] for r, c in cells)


def _candidate_lines(height: int, width: int, orientation: str) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    if orientation == "h":
        for r in range(1, height - 1):
            for c in range(1, width - 3):
                candidates.append(_line_cells("h", r, c))
    else:
        for r in range(1, height - 3):
            for c in range(1, width - 1):
                candidates.append(_line_cells("v", r, c))
    return candidates


def _expected_output(
    grid: np.ndarray,
    palette_colors: list[int],
    lights: list[dict[str, object]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    ranked = sorted(lights, key=lambda light: (int(light["count"]), int(light["cells"][0][0]), int(light["cells"][0][1])))
    for rank, light in enumerate(ranked):
        color = int(palette_colors[rank])
        for idx in light["lit_indices"]:
            r, c = light["cells"][idx]
            out[r, c] = color
    return out


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(9, 15))
    width = int(_RNG.integers(9, 15))
    edge = str(_RNG.choice(_EDGE_OPTIONS))

    if edge in ("top", "bottom"):
        start = int(_RNG.integers(0, width - 2))
    else:
        start = int(_RNG.integers(0, height - 2))

    palette_colors = [int(x) for x in _RNG.choice(_COLOR_POOL, size=3, replace=False)]
    grid = np.full((height, width), _BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)

    palette_cells = _palette_cells(height, width, edge, start)
    for color, (r, c) in zip(palette_colors, palette_cells):
        grid[r, c] = color
    _mark_blocked(blocked, palette_cells)

    light_count = int(_RNG.integers(2, 4))
    count_options = _VALID_COUNT_SETS[light_count]
    chosen_counts = list(count_options[int(_RNG.integers(0, len(count_options)))])
    _RNG.shuffle(chosen_counts)

    total_lit = int(sum(chosen_counts))
    lit_colors = [int(x) for x in _RNG.choice(_COLOR_POOL, size=total_lit, replace=False)]
    lit_cursor = 0
    lights: list[dict[str, object]] = []

    for count in chosen_counts:
        orientation = str(_RNG.choice(_ORIENTATIONS))
        candidates = _candidate_lines(height, width, orientation)
        if not candidates:
            return None
        perm = _RNG.permutation(len(candidates))

        placed_cells = None
        for idx in perm:
            cells = candidates[int(idx)]
            if _clear_cells(blocked, cells):
                placed_cells = cells
                break
        if placed_cells is None:
            return None

        lit_indices = sorted(int(x) for x in _RNG.choice(3, size=count, replace=False))
        colors = [_OFF, _OFF, _OFF]
        for local_idx in lit_indices:
            colors[local_idx] = lit_colors[lit_cursor]
            lit_cursor += 1

        for color, (r, c) in zip(colors, placed_cells):
            grid[r, c] = int(color)
        _mark_blocked(blocked, placed_cells)
        lights.append(
            {
                "cells": placed_cells,
                "count": int(count),
                "lit_indices": lit_indices,
            }
        )

    expected = _expected_output(grid, palette_colors, lights)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    if np.array_equal(grid, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid traffic-light puzzle input")
