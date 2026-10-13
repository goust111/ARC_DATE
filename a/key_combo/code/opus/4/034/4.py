from __future__ import annotations

import random
import sys
from collections import deque
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, neighbors8


BG = Color.BLACK
GRAY = Color.GREY
PALETTE_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
CORNER_ORDER = ("tl", "tr", "br", "bl")
QUADRANT_TO_BLOCK = {"tl": (0, 0), "tr": (0, 1), "bl": (1, 0), "br": (1, 1)}
CORNER_STEPS = {"tl": (-1, -1), "tr": (-1, 1), "br": (1, 1), "bl": (1, -1)}


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= int(r) < int(shape[0]) and 0 <= int(c) < int(shape[1])


def _band_start(top: int, left: int, corner: str) -> tuple[int, int]:
    starts = {
        "tl": (top, left),
        "tr": (top, left + 1),
        "br": (top + 1, left + 1),
        "bl": (top + 1, left),
    }
    return starts[corner]


def _band_support_cells(top: int, left: int, corner: str) -> list[tuple[int, int]]:
    if corner == "tl":
        return [(top - 1, left - 1), (top - 1, left), (top, left - 1)]
    if corner == "tr":
        return [(top - 1, left + 1), (top - 1, left + 2), (top, left + 2)]
    if corner == "br":
        return [(top + 1, left + 2), (top + 2, left + 1), (top + 2, left + 2)]
    return [(top + 1, left - 1), (top + 2, left - 1), (top + 2, left)]


def _corner_predicate(top: int, left: int, corner: str):
    starts = {name: _band_start(top, left, name) for name in CORNER_ORDER}
    sr, sc = starts[corner]

    if corner == "tl":
        return lambda r, c: r <= sr and c <= sc
    if corner == "tr":
        return lambda r, c: r <= sr and c >= sc
    if corner == "br":
        return lambda r, c: r >= sr and c >= sc
    return lambda r, c: r >= sr and c <= sc


def _single_components(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=8,
        monochromatic=True,
        colors=PALETTE_COLORS,
        return_format="dict",
    )
    singles = [comp for comp in comps if int(comp["area"]) == 1]
    assert len(singles) == 4, "expected exactly four colored single pixels"
    return singles


def _quadrant_single_map(singles: list[dict], top: int, left: int) -> dict[str, tuple[tuple[int, int], int]]:
    mapping: dict[str, tuple[tuple[int, int], int]] = {}
    for comp in singles:
        r, c = map(int, comp["coords"][0])
        color = int(comp["color"])
        if r < top and c < left:
            key = "tl"
        elif r < top and c > left + 1:
            key = "tr"
        elif r > top + 1 and c < left:
            key = "bl"
        elif r > top + 1 and c > left + 1:
            key = "br"
        else:
            raise AssertionError("single pixel overlaps the block band corridor")
        assert key not in mapping, "expected one single pixel per quadrant"
        mapping[key] = ((r, c), color)
    assert set(mapping) == set(QUADRANT_TO_BLOCK), "missing a quadrant single pixel"
    assert len({color for _pos, color in mapping.values()}) == 4, "quadrant colors must be distinct"
    return mapping


def _find_anchor_block(grid: np.ndarray, singles: list[dict]) -> tuple[int, int]:
    candidates: list[tuple[int, int]] = []
    gray = grid == GRAY
    h, w = grid.shape
    all_gray = {tuple(map(int, rc)) for rc in np.argwhere(gray)}

    for top in range(h - 1):
        for left in range(w - 1):
            if not bool(np.all(gray[top : top + 2, left : left + 2])):
                continue
            try:
                _quadrant_single_map(singles, top, left)
            except AssertionError:
                continue
            bands = {corner: _collect_band(gray, top, left, corner) for corner in CORNER_ORDER}
            if any(_band_start(top, left, corner) not in bands[corner] for corner in CORNER_ORDER):
                continue
            union = set().union(*bands.values())
            if union != all_gray:
                continue
            if sum(len(band) for band in bands.values()) != len(union):
                continue
            candidates.append((top, left))

    assert len(candidates) == 1, "expected a unique 2x2 gray anchor block"
    return candidates[0]


def _collect_band(gray_mask: np.ndarray, top: int, left: int, corner: str) -> set[tuple[int, int]]:
    sr, sc = _band_start(top, left, corner)
    pred = _corner_predicate(top, left, corner)
    todo = deque([(sr, sc)])
    seen: set[tuple[int, int]] = set()

    while todo:
        r, c = todo.popleft()
        if (r, c) in seen:
            continue
        if not _in_bounds(gray_mask.shape, r, c) or not bool(gray_mask[r, c]):
            continue
        if not pred(int(r), int(c)):
            continue
        seen.add((int(r), int(c)))
        for nr, nc in neighbors8(int(r), int(c)):
            todo.append((int(nr), int(nc)))

    return seen


def _paint_solution(
    grid: np.ndarray,
    top: int,
    left: int,
    quadrant_map: dict[str, tuple[tuple[int, int], int]],
) -> np.ndarray:
    gray_mask = grid == GRAY
    out = np.array(grid, dtype=int, copy=True)

    for pos, _color in quadrant_map.values():
        out[pos] = BG

    for quadrant, (dr, dc) in QUADRANT_TO_BLOCK.items():
        out[top + dr, left + dc] = int(quadrant_map[quadrant][1])

    palette = [int(quadrant_map[key][1]) for key in ("tl", "tr", "br", "bl")]
    start_indices = {corner: idx for idx, corner in enumerate(CORNER_ORDER)}

    for corner in CORNER_ORDER:
        sr, sc = _band_start(top, left, corner)
        for r, c in _collect_band(gray_mask, top, left, corner):
            depth = max(abs(int(r) - sr), abs(int(c) - sc))
            out[r, c] = palette[(start_indices[corner] + depth) % 4]

    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Rebuild a clockwise palette from quadrant marker pixels, then color each diagonal band by band-local layer depth.

    Input structure:
    - The background is black, with one central 2x2 gray anchor block.
    - Four distinct colored single pixels lie in the four strict quadrants around that block.
    - Four gray width-2 diagonal bands start from the block's four corner pixels and extend toward the canvas corners.

    Rule:
    - Map the upper-left, upper-right, lower-left, and lower-right single-pixel colors onto the matching cells of the gray 2x2 block.
    - Read the recolored block clockwise as upper-left, upper-right, lower-right, lower-left to form a cyclic palette.
    - For each gray diagonal band, use the palette color at that band's start corner as layer 0, then recolor every band pixel by clockwise palette cycling with Chebyshev depth from the start corner.
    - Delete the original four scattered single pixels and keep the recolored 2x2 block.

    Key insight:
    - The scattered pixels encode both the palette entries and their clockwise order, so the same four colors drive all four bands with different phase offsets.
    """
    grid = np.asarray(input_grid, dtype=int)
    assert grid.ndim == 2

    singles = _single_components(grid)
    top, left = _find_anchor_block(grid, singles)
    quadrant_map = _quadrant_single_map(singles, top, left)
    return _paint_solution(grid, top, left, quadrant_map)


def _generate_band_cells(shape: tuple[int, int], top: int, left: int, corner: str, style: int) -> set[tuple[int, int]]:
    sr, sc = _band_start(top, left, corner)
    dr, dc = CORNER_STEPS[corner]
    max_depth = min(
        sr if dr < 0 else shape[0] - 1 - sr,
        sc if dc < 0 else shape[1] - 1 - sc,
    )
    cells = {(sr, sc)}

    for depth in range(1, max_depth + 1):
        main_r = sr + depth * dr
        main_c = sc + depth * dc
        cells.add((main_r, main_c))
        if corner in {"tl", "br"}:
            side = (main_r, main_c - dc) if style == 0 else (main_r - dr, main_c)
        else:
            side = (main_r, main_c - dc) if style == 0 else (main_r - dr, main_c)
        cells.add(side)

    return {(int(r), int(c)) for r, c in cells if _in_bounds(shape, int(r), int(c))}


def _forbidden_single_cells(gray_cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    forbidden = set(gray_cells)
    for r, c in list(gray_cells):
        for nr, nc in neighbors8(r, c):
            if _in_bounds(shape, nr, nc):
                forbidden.add((int(nr), int(nc)))
    return forbidden


def _quadrant_candidates(
    shape: tuple[int, int],
    top: int,
    left: int,
    forbidden: set[tuple[int, int]],
) -> dict[str, list[tuple[int, int]]]:
    h, w = shape
    out = {key: [] for key in QUADRANT_TO_BLOCK}

    for r in range(h):
        for c in range(w):
            if (r, c) in forbidden:
                continue
            if r < top and c < left:
                out["tl"].append((r, c))
            elif r < top and c > left + 1:
                out["tr"].append((r, c))
            elif r > top + 1 and c < left:
                out["bl"].append((r, c))
            elif r > top + 1 and c > left + 1:
                out["br"].append((r, c))

    return out


def _build_expected_output(
    grid: np.ndarray,
    top: int,
    left: int,
    band_cells: dict[str, set[tuple[int, int]]],
    singles: dict[str, tuple[tuple[int, int], int]],
) -> np.ndarray:
    out = np.array(grid, dtype=int, copy=True)
    palette = [int(singles[key][1]) for key in ("tl", "tr", "br", "bl")]

    for pos, _color in singles.values():
        out[pos] = BG

    for quadrant, (dr, dc) in QUADRANT_TO_BLOCK.items():
        out[top + dr, left + dc] = int(singles[quadrant][1])

    for start_index, corner in enumerate(CORNER_ORDER):
        sr, sc = _band_start(top, left, corner)
        for r, c in band_cells[corner]:
            depth = max(abs(int(r) - sr), abs(int(c) - sc))
            out[r, c] = palette[(start_index + depth) % 4]

    return out


def _semantic_checks(
    grid: np.ndarray,
    expected: np.ndarray,
    top: int,
    left: int,
    band_cells: dict[str, set[tuple[int, int]]],
    singles: dict[str, tuple[tuple[int, int], int]],
) -> bool:
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return False

    if int((grid == GRAY).sum()) == 0 or int((expected == GRAY).sum()) != 0:
        return False

    lengths = []
    palette_set = set(PALETTE_COLORS)
    for corner in CORNER_ORDER:
        sr, sc = _band_start(top, left, corner)
        depths = [max(abs(r - sr), abs(c - sc)) for r, c in band_cells[corner]]
        if max(depths) < 3:
            return False
        lengths.append(max(depths))
        colors = {int(expected[r, c]) for r, c in band_cells[corner]}
        if colors != palette_set:
            return False

    if len(set(lengths)) == 1:
        return False

    for quadrant, (pos, color) in singles.items():
        if int(grid[pos]) != int(color) or int(expected[pos]) != BG:
            return False
        dr, dc = QUADRANT_TO_BLOCK[quadrant]
        if int(expected[top + dr, left + dc]) != int(color):
            return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(13, 19)
    w = random.randint(13, 19)
    grid = np.full((h, w), BG, dtype=int)

    top = random.randint(3, h - 5)
    left = random.randint(3, w - 5)
    lengths = [top, left, h - top - 2, w - left - 2]
    if min(lengths) < 3 or len(set(lengths)) == 1:
        return None

    band_cells: dict[str, set[tuple[int, int]]] = {}
    all_gray: set[tuple[int, int]] = set()
    style_pattern = random.choice(
        [
            {"tl": 0, "tr": 1, "br": 0, "bl": 1},
            {"tl": 1, "tr": 0, "br": 1, "bl": 0},
        ]
    )
    for corner in CORNER_ORDER:
        style = style_pattern[corner]
        band = _generate_band_cells(grid.shape, top, left, corner, style)
        band_cells[corner] = band
        all_gray.update(band)

    for r, c in all_gray:
        grid[r, c] = GRAY
    grid[top : top + 2, left : left + 2] = GRAY

    forbidden = _forbidden_single_cells(all_gray, grid.shape)
    quadrant_slots = _quadrant_candidates(grid.shape, top, left, forbidden)
    if any(not slots for slots in quadrant_slots.values()):
        return None

    colors = list(PALETTE_COLORS)
    random.shuffle(colors)
    singles: dict[str, tuple[tuple[int, int], int]] = {}
    for quadrant, color in zip(("tl", "tr", "bl", "br"), colors):
        pos = random.choice(quadrant_slots[quadrant])
        singles[quadrant] = (pos, int(color))
        grid[pos] = int(color)

    expected = _build_expected_output(grid, top, left, band_cells, singles)
    if not _semantic_checks(grid, expected, top, left, band_cells, singles):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
