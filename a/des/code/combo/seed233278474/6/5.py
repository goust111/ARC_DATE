from __future__ import annotations

import numpy as np


ARROW_COLOR = 9
_RNG = np.random.default_rng()
_DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
_DIRECTIONS = ("right", "left", "up", "down")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_monochrome_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue

            color = int(grid[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in _DIRS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append({"color": color, "coords": coords})

    return comps


def _bbox(coords: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _arrow_coords_from_bbox(direction: str, size: int, bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, bbox)
    coords: set[tuple[int, int]] = set()

    if direction == "right":
        for x in range(size):
            for r in range(r0 + x, r1 - x + 1):
                coords.add((int(r), int(c0 + x)))
    elif direction == "left":
        for x in range(size):
            pad = int(size - 1 - x)
            for r in range(r0 + pad, r1 - pad + 1):
                coords.add((int(r), int(c0 + x)))
    elif direction == "down":
        for y in range(size):
            for c in range(c0 + y, c1 - y + 1):
                coords.add((int(r0 + y), int(c)))
    elif direction == "up":
        for y in range(size):
            pad = int(size - 1 - y)
            for c in range(c0 + pad, c1 - pad + 1):
                coords.add((int(r0 + y), int(c)))
    else:
        raise AssertionError(direction)

    return coords


def _parse_arrow_component(comp: dict[str, object]) -> dict[str, object] | None:
    coords = {tuple(map(int, xy)) for xy in comp["coords"]}  # type: ignore[index]
    area = int(len(coords))
    size = int(round(area ** 0.5))
    if size < 2 or size * size != area:
        return None

    bbox = _bbox(coords)
    r0, c0, r1, c1 = map(int, bbox)
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    matches: list[str] = []

    if height == 2 * size - 1 and width == size:
        for direction in ("right", "left"):
            if coords == _arrow_coords_from_bbox(direction, size, bbox):
                matches.append(direction)
    if height == size and width == 2 * size - 1:
        for direction in ("up", "down"):
            if coords == _arrow_coords_from_bbox(direction, size, bbox):
                matches.append(direction)

    if len(matches) != 1:
        return None

    direction = matches[0]
    if direction == "right":
        tip = (int((r0 + r1) // 2), int(c1))
    elif direction == "left":
        tip = (int((r0 + r1) // 2), int(c0))
    elif direction == "up":
        tip = (int(r0), int((c0 + c1) // 2))
    else:
        tip = (int(r1), int((c0 + c1) // 2))

    return {
        "color": int(comp["color"]),  # type: ignore[index]
        "coords": coords,
        "bbox": bbox,
        "size": size,
        "direction": direction,
        "tip": tip,
    }


def _stripe_cells(arrow: dict[str, object]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    r0, c0, r1, c1 = map(int, arrow["bbox"])  # type: ignore[index]
    direction = str(arrow["direction"])  # type: ignore[index]

    if direction == "right":
        gap = [(int(r), int(c1 + 1)) for r in range(r0, r1 + 1)]
        stripe = [(int(r), int(c1 + 2)) for r in range(r0, r1 + 1)]
    elif direction == "left":
        gap = [(int(r), int(c0 - 1)) for r in range(r0, r1 + 1)]
        stripe = [(int(r), int(c0 - 2)) for r in range(r1, r0 - 1, -1)]
    elif direction == "up":
        gap = [(int(r0 - 1), int(c)) for c in range(c0, c1 + 1)]
        stripe = [(int(r0 - 2), int(c)) for c in range(c0, c1 + 1)]
    elif direction == "down":
        gap = [(int(r1 + 1), int(c)) for c in range(c0, c1 + 1)]
        stripe = [(int(r1 + 2), int(c)) for c in range(c1, c0 - 1, -1)]
    else:
        raise AssertionError(direction)

    return gap, stripe


def _layer_index(arrow: dict[str, object], r: int, c: int) -> int:
    r0, c0, r1, c1 = map(int, arrow["bbox"])  # type: ignore[index]
    direction = str(arrow["direction"])  # type: ignore[index]
    if direction == "right":
        return int(c - c0)
    if direction == "left":
        return int(c1 - c)
    if direction == "up":
        return int(r1 - r)
    if direction == "down":
        return int(r - r0)
    raise AssertionError(direction)


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each arrow's nested triangle layers from the stripe it points at.

    Interpretation of the stripe order is rotation-consistent:
    right -> top-to-bottom, left -> bottom-to-top,
    up -> left-to-right, down -> right-to-left.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = map(int, g.shape)
    bg = _get_bg(g)

    arrows: list[dict[str, object]] = []
    for comp in _find_monochrome_components(g, bg):
        arrow = _parse_arrow_component(comp)
        if arrow is not None:
            arrows.append(arrow)
    assert arrows, "expected at least one arrow"

    out = np.full((h, w), int(bg), dtype=int)

    for arrow in arrows:
        size = int(arrow["size"])  # type: ignore[index]
        gap_cells, stripe_cells = _stripe_cells(arrow)

        for rr, cc in gap_cells + stripe_cells:
            assert 0 <= rr < h and 0 <= cc < w, "stripe must stay inside the grid"
        for rr, cc in gap_cells:
            assert int(g[rr, cc]) == bg, "arrow and stripe must be separated by one background cell"

        stripe_colors = [int(g[rr, cc]) for rr, cc in stripe_cells]
        assert len(stripe_colors) == 2 * size - 1
        assert all(color != bg for color in stripe_colors), "stripe cells must be non-background"
        layer_colors = stripe_colors[:size]

        for rr, cc in arrow["coords"]:  # type: ignore[index]
            layer = _layer_index(arrow, int(rr), int(cc))
            out[int(rr), int(cc)] = int(layer_colors[layer])

    return out


# --------------- generator ---------------

GRID_SIZE_MIN = 18
GRID_SIZE_MAX_EXCLUSIVE = 31
OBJECT_COUNT_MIN = 1
OBJECT_COUNT_MAX_EXCLUSIVE = 5
SIZE_OPTIONS = (2, 3, 4)
MAX_GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 400


def _pair_geometry(direction: str, size: int, top: int, left: int) -> dict[str, object]:
    if direction == "right":
        pair_bbox = (top, left, top + 2 * size - 2, left + size + 1)
        arrow_bbox = (top, left, top + 2 * size - 2, left + size - 1)
        stripe_cells = [(int(top + i), int(left + size + 1)) for i in range(2 * size - 1)]
        gap_cells = [(int(top + i), int(left + size)) for i in range(2 * size - 1)]
    elif direction == "left":
        pair_bbox = (top, left, top + 2 * size - 2, left + size + 1)
        arrow_bbox = (top, left + 2, top + 2 * size - 2, left + size + 1)
        stripe_cells = [(int(top + i), int(left)) for i in range(2 * size - 2, -1, -1)]
        gap_cells = [(int(top + i), int(left + 1)) for i in range(2 * size - 1)]
    elif direction == "up":
        pair_bbox = (top, left, top + size + 1, left + 2 * size - 2)
        arrow_bbox = (top + 2, left, top + size + 1, left + 2 * size - 2)
        stripe_cells = [(int(top), int(left + i)) for i in range(2 * size - 1)]
        gap_cells = [(int(top + 1), int(left + i)) for i in range(2 * size - 1)]
    elif direction == "down":
        pair_bbox = (top, left, top + size + 1, left + 2 * size - 2)
        arrow_bbox = (top, left, top + size - 1, left + 2 * size - 2)
        stripe_cells = [(int(top + size + 1), int(left + i)) for i in range(2 * size - 2, -1, -1)]
        gap_cells = [(int(top + size), int(left + i)) for i in range(2 * size - 1)]
    else:
        raise AssertionError(direction)

    arrow_coords = _arrow_coords_from_bbox(direction, size, arrow_bbox)
    return {
        "direction": direction,
        "size": size,
        "pair_bbox": pair_bbox,
        "arrow_bbox": arrow_bbox,
        "arrow_coords": arrow_coords,
        "stripe_cells": stripe_cells,
        "gap_cells": gap_cells,
    }


def _expanded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, int], pad: int = 1
) -> tuple[int, int, int, int]:
    h, w = map(int, shape)
    r0, c0, r1, c1 = map(int, bbox)
    return max(0, r0 - pad), max(0, c0 - pad), min(h - 1, r1 + pad), min(w - 1, c1 + pad)


def _place_specs(specs: list[dict[str, object]], shape: tuple[int, int]) -> list[dict[str, object]] | None:
    h, w = map(int, shape)
    reserved = np.zeros((h, w), dtype=bool)
    placed: list[dict[str, object]] = []

    order = sorted(
        specs,
        key=lambda spec: (
            max(
                int(spec["size"]) + 2,
                2 * int(spec["size"]) - 1,
            ),
            int(spec["size"]),
        ),
        reverse=True,
    )

    for spec in order:
        size = int(spec["size"])
        direction = str(spec["direction"])
        if direction in {"right", "left"}:
            pair_h = 2 * size - 1
            pair_w = size + 2
        else:
            pair_h = size + 2
            pair_w = 2 * size - 1

        origins = [(r, c) for r in range(h - pair_h + 1) for c in range(w - pair_w + 1)]
        _RNG.shuffle(origins)
        found = False

        for top, left in origins[:PLACEMENT_ATTEMPTS]:
            geom = _pair_geometry(direction, size, int(top), int(left))
            rr0, cc0, rr1, cc1 = _expanded_bbox(geom["pair_bbox"], shape, pad=1)  # type: ignore[index]
            if reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any():
                continue

            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed.append({**spec, **geom})
            found = True
            break

        if not found:
            return None

    return placed


def _render_input(shape: tuple[int, int], bg: int, placed: list[dict[str, object]]) -> np.ndarray:
    grid = np.full(shape, int(bg), dtype=int)
    for spec in placed:
        for rr, cc in spec["arrow_coords"]:  # type: ignore[index]
            grid[int(rr), int(cc)] = ARROW_COLOR
        colors = list(map(int, spec["stripe_colors"]))  # type: ignore[index]
        for (rr, cc), color in zip(spec["stripe_cells"], colors):  # type: ignore[index]
            grid[int(rr), int(cc)] = int(color)
    return grid


def _render_expected(shape: tuple[int, int], bg: int, placed: list[dict[str, object]]) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec in placed:
        arrow = {
            "bbox": spec["arrow_bbox"],
            "direction": spec["direction"],
            "coords": spec["arrow_coords"],
        }
        layer_colors = list(map(int, spec["stripe_colors"]))[: int(spec["size"])]  # type: ignore[index]
        for rr, cc in spec["arrow_coords"]:  # type: ignore[index]
            out[int(rr), int(cc)] = int(layer_colors[_layer_index(arrow, int(rr), int(cc))])
    return out


def _generation_attempt() -> np.ndarray | None:
    bg = int(_RNG.choice([c for c in range(10) if c != ARROW_COLOR]))
    shape = (
        int(_RNG.integers(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE)),
        int(_RNG.integers(GRID_SIZE_MIN, GRID_SIZE_MAX_EXCLUSIVE)),
    )
    count = int(_RNG.integers(OBJECT_COUNT_MIN, OBJECT_COUNT_MAX_EXCLUSIVE))
    directions = list(_RNG.permutation(_DIRECTIONS))[:count]

    specs: list[dict[str, object]] = []
    palette = [c for c in range(10) if c not in {bg, ARROW_COLOR}]
    for direction in directions:
        size = int(_RNG.choice(SIZE_OPTIONS))
        stripe_colors = list(map(int, _RNG.choice(palette, size=2 * size - 1, replace=False)))
        specs.append(
            {
                "direction": direction,
                "size": size,
                "stripe_colors": stripe_colors,
            }
        )

    placed = _place_specs(specs, shape)
    if placed is None:
        return None

    grid = _render_input(shape, bg, placed)
    expected = _render_expected(shape, bg, placed)
    if np.array_equal(main(grid), expected):
        return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generation_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
