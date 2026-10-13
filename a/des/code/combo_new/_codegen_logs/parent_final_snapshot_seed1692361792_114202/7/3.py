from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
ARROW_COLOR = 9
BAND_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
DIRECTIONS = ("left", "right", "up", "down")
GENERATION_ATTEMPTS = 400
_GENERATION_INDEX = 0
_PHASE_DIRECTIONS = (
    ("left", "up"),
    ("right", "up"),
    ("left", "down"),
    ("right", "down"),
)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_bands(grid: np.ndarray) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    bands: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if seen[r, c] or color in (BG, ARROW_COLOR):
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = sorted({rr for rr, _ in coords})
            cols = sorted({cc for _, cc in coords})
            if len(rows) == 1 and len(cols) == 1:
                orientation = "point"
            elif len(rows) == 1:
                orientation = "h"
                assert len(coords) == cols[-1] - cols[0] + 1
            elif len(cols) == 1:
                orientation = "v"
                assert len(coords) == rows[-1] - rows[0] + 1
            else:
                raise AssertionError("foreground components must be straight monochrome bands")

            bands.append(
                {
                    "color": color,
                    "coords": tuple(sorted(coords)),
                    "orientation": orientation,
                    "length": len(coords),
                    "row": rows[0],
                    "col": cols[0],
                    "r0": rows[0],
                    "r1": rows[-1],
                    "c0": cols[0],
                    "c1": cols[-1],
                }
            )

    return bands


def _all_color(grid: np.ndarray, cells: Iterable[tuple[int, int]], color: int) -> bool:
    h, w = grid.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if int(grid[r, c]) != color:
            return False
    return True


def _detect_arrow(grid: np.ndarray, band: dict[str, object]) -> tuple[str, list[tuple[int, int]]] | None:
    orientation = str(band["orientation"])
    candidates: list[tuple[str, list[tuple[int, int]]]] = []

    if orientation in ("h", "point"):
        row = int(band["row"])
        left = int(band["c0"])
        right = int(band["c1"])

        right_arrow = [(row - 1, right + 1), (row, right + 1), (row + 1, right + 1)]
        if _all_color(grid, right_arrow, ARROW_COLOR):
            candidates.append(("right", right_arrow))

        left_arrow = [(row - 1, left - 1), (row, left - 1), (row + 1, left - 1)]
        if _all_color(grid, left_arrow, ARROW_COLOR):
            candidates.append(("left", left_arrow))

    if orientation in ("v", "point"):
        col = int(band["col"])
        top = int(band["r0"])
        bottom = int(band["r1"])

        up_arrow = [(top - 1, col - 1), (top - 1, col), (top - 1, col + 1)]
        if _all_color(grid, up_arrow, ARROW_COLOR):
            candidates.append(("up", up_arrow))

        down_arrow = [(bottom + 1, col - 1), (bottom + 1, col), (bottom + 1, col + 1)]
        if _all_color(grid, down_arrow, ARROW_COLOR):
            candidates.append(("down", down_arrow))

    assert len(candidates) <= 1, "each band may have at most one arrow"
    return candidates[0] if candidates else None


def _extension_cells(band: dict[str, object], direction: str) -> list[tuple[int, int]]:
    length = int(band["length"])
    if direction == "right":
        row = int(band["row"])
        start = int(band["c1"]) + 2
        return [(row, start + i) for i in range(length)]
    if direction == "left":
        row = int(band["row"])
        start = int(band["c0"]) - 2
        return [(row, start - i) for i in range(length)]
    if direction == "down":
        col = int(band["col"])
        start = int(band["r1"]) + 2
        return [(start + i, col) for i in range(length)]
    if direction == "up":
        col = int(band["col"])
        start = int(band["r0"]) - 2
        return [(start - i, col) for i in range(length)]
    raise ValueError(f"unknown direction: {direction}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend each arrow-marked color band outward by a copy of its own length.

    Input structure:
    - The background is black.
    - Foreground bands are straight monochrome horizontal or vertical segments.
    - Some bands have a maroon 3-pixel arrow placed immediately beyond one end,
      with the arrow pointing away from the band.

    Rule:
    - For every band with an adjacent arrow, read the arrow direction and the
      band's pixel length.
    - Remove the arrow itself, then draw a same-color extension in the arrow's
      direction with length exactly equal to the original band length.
    - Leave bands without arrows unchanged.

    Key insight:
    - The maroon marker gives only the extension direction; the required copy
      length must be measured from the original band itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    out = g.copy()
    for band in _find_bands(g):
        match = _detect_arrow(g, band)
        if match is None:
            continue

        direction, arrow_cells = match
        for r, c in arrow_cells:
            out[r, c] = BG

        for r, c in _extension_cells(band, direction):
            assert 0 <= r < h and 0 <= c < w, "extension must stay inside the grid"
            out[r, c] = int(band["color"])

    return out


def _expand_cells(cells: Iterable[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _make_spec(
    orientation: str,
    start_r: int,
    start_c: int,
    length: int,
    color: int,
    arrow_dir: str | None,
) -> dict[str, object]:
    if orientation == "h":
        band = [(start_r, start_c + i) for i in range(length)]
        if arrow_dir == "right":
            arrow = [(start_r - 1, start_c + length), (start_r, start_c + length), (start_r + 1, start_c + length)]
            ext = [(start_r, start_c + length + 1 + i) for i in range(length)]
        elif arrow_dir == "left":
            arrow = [(start_r - 1, start_c - 1), (start_r, start_c - 1), (start_r + 1, start_c - 1)]
            ext = [(start_r, start_c - 2 - i) for i in range(length)]
        else:
            arrow = []
            ext = []
    elif orientation == "v":
        band = [(start_r + i, start_c) for i in range(length)]
        if arrow_dir == "down":
            arrow = [(start_r + length, start_c - 1), (start_r + length, start_c), (start_r + length, start_c + 1)]
            ext = [(start_r + length + 1 + i, start_c) for i in range(length)]
        elif arrow_dir == "up":
            arrow = [(start_r - 1, start_c - 1), (start_r - 1, start_c), (start_r - 1, start_c + 1)]
            ext = [(start_r - 2 - i, start_c) for i in range(length)]
        else:
            arrow = []
            ext = []
    else:
        raise ValueError(f"unknown orientation: {orientation}")

    return {
        "orientation": orientation,
        "length": length,
        "color": color,
        "arrow_dir": arrow_dir,
        "band": band,
        "arrow": arrow,
        "ext": ext,
        "input_cells": band + arrow,
        "output_cells": band + ext,
    }


def _random_arrow_spec(h: int, w: int, direction: str, color: int) -> dict[str, object] | None:
    if direction in ("left", "right"):
        max_len = min(6, (w - 1) // 2)
        if h < 3 or max_len < 2:
            return None
        length = random.randint(2, max_len)
        row = random.randint(1, h - 2)
        if direction == "right":
            start_c = random.randint(0, w - (2 * length + 1))
        else:
            start_c = random.randint(length + 1, w - length)
        return _make_spec("h", row, start_c, length, color, direction)

    max_len = min(6, (h - 1) // 2)
    if w < 3 or max_len < 2:
        return None
    length = random.randint(2, max_len)
    col = random.randint(1, w - 2)
    if direction == "down":
        start_r = random.randint(0, h - (2 * length + 1))
    else:
        start_r = random.randint(length + 1, h - length)
    return _make_spec("v", start_r, col, length, color, direction)


def _random_plain_spec(h: int, w: int, orientation: str, color: int) -> dict[str, object] | None:
    if orientation == "h":
        if w < 2:
            return None
        length = random.randint(2, min(8, w))
        row = random.randint(0, h - 1)
        start_c = random.randint(0, w - length)
        return _make_spec("h", row, start_c, length, color, None)

    if h < 2:
        return None
    length = random.randint(2, min(8, h))
    col = random.randint(0, w - 1)
    start_r = random.randint(0, h - length)
    return _make_spec("v", start_r, col, length, color, None)


def _can_place(
    spec: dict[str, object],
    input_margin: np.ndarray,
    input_exact: np.ndarray,
    output_exact: np.ndarray,
) -> bool:
    h, w = input_margin.shape
    input_cells = list(spec["input_cells"])
    output_cells = list(spec["output_cells"])

    for r, c in input_cells + output_cells:
        if not (0 <= r < h and 0 <= c < w):
            return False

    for r, c in input_cells:
        if input_margin[r, c] or output_exact[r, c]:
            return False

    for r, c in list(spec["arrow"]) + list(spec["ext"]):
        if input_exact[r, c] or output_exact[r, c]:
            return False

    return True


def _place_spec(
    grid: np.ndarray,
    spec: dict[str, object],
    input_margin: np.ndarray,
    input_exact: np.ndarray,
    output_exact: np.ndarray,
) -> None:
    for r, c in spec["band"]:
        grid[r, c] = int(spec["color"])
    for r, c in spec["arrow"]:
        grid[r, c] = ARROW_COLOR

    for r, c in spec["input_cells"]:
        input_exact[r, c] = True
    for r, c in spec["output_cells"]:
        output_exact[r, c] = True
    for r, c in _expand_cells(spec["input_cells"], *grid.shape):
        input_margin[r, c] = True


def _validate_generated(grid: np.ndarray, mandatory_dirs: tuple[str, str]) -> bool:
    bands = _find_bands(grid)
    if len(bands) < 3:
        return False

    arrow_dirs: list[str] = []
    plain_count = 0
    band_orientations = set()
    for band in bands:
        if band["orientation"] != "point":
            band_orientations.add(str(band["orientation"]))
        match = _detect_arrow(grid, band)
        if match is None:
            plain_count += 1
        else:
            arrow_dirs.append(match[0])

    if plain_count < 1:
        return False
    if len(set(band_orientations)) < 2:
        return False
    if mandatory_dirs[0] not in arrow_dirs or mandatory_dirs[1] not in arrow_dirs:
        return False
    if not any(direction in ("left", "right") for direction in arrow_dirs):
        return False
    if not any(direction in ("up", "down") for direction in arrow_dirs):
        return False
    if int(np.count_nonzero(grid == ARROW_COLOR)) != 3 * len(arrow_dirs):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out == ARROW_COLOR)) != 0:
        return False

    return True


def _gen_attempt(call_id: int) -> np.ndarray | None:
    mandatory_dirs = _PHASE_DIRECTIONS[call_id % len(_PHASE_DIRECTIONS)]
    h = random.randint(15, 28)
    w = random.randint(15, 28)
    grid = np.full((h, w), BG, dtype=int)
    input_margin = np.zeros((h, w), dtype=bool)
    input_exact = np.zeros((h, w), dtype=bool)
    output_exact = np.zeros((h, w), dtype=bool)

    arrow_dirs = [mandatory_dirs[0], mandatory_dirs[1]]
    extra_arrow_count = random.randint(0, 3)
    for _ in range(extra_arrow_count):
        arrow_dirs.append(random.choice(DIRECTIONS))
    random.shuffle(arrow_dirs)

    plain_count = random.randint(1, 3)
    plain_orientations = [random.choice(("h", "v"))]
    if plain_count >= 2:
        plain_orientations.append("v" if plain_orientations[0] == "h" else "h")
    while len(plain_orientations) < plain_count:
        plain_orientations.append(random.choice(("h", "v")))
    random.shuffle(plain_orientations)

    tasks = [("arrow", direction) for direction in arrow_dirs] + [("plain", orient) for orient in plain_orientations]
    tasks.sort(key=lambda item: 0 if item[0] == "arrow" else 1)

    for kind, value in tasks:
        placed = False
        for _ in range(120):
            color = random.choice(BAND_COLORS)
            if kind == "arrow":
                spec = _random_arrow_spec(h, w, str(value), color)
            else:
                spec = _random_plain_spec(h, w, str(value), color)
            if spec is None or not _can_place(spec, input_margin, input_exact, output_exact):
                continue
            _place_spec(grid, spec, input_margin, input_exact, output_exact)
            placed = True
            break
        if not placed:
            return None

    return grid if _validate_generated(grid, mandatory_dirs) else None


def generate_input() -> np.ndarray:
    global _GENERATION_INDEX

    call_id = _GENERATION_INDEX
    _GENERATION_INDEX += 1

    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt(call_id)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-band input")
