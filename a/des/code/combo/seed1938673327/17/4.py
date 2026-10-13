from __future__ import annotations

import random

import numpy as np


BG = 0
BLUE = 1
RED = 2
GREEN = 3
LEGEND = (RED, GREEN, BLUE)
OBJECT_COLORS = (4, 5, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _mask(*rows: str) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(int(v) for v in row) for row in mask.astype(int))


def _shape_variants(mask: np.ndarray) -> tuple[np.ndarray, ...]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            key = _shape_key(variant)
            if key not in seen:
                seen.add(key)
                variants.append(variant.copy())
    return tuple(variants)


def _mask_exposed_edges(mask: np.ndarray) -> int:
    h, w = mask.shape
    total = 0
    for r, c in zip(*np.where(mask)):
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if not (0 <= rr < h and 0 <= cc < w) or not bool(mask[rr, cc]):
                total += 1
    return total


def _mask_category(mask: np.ndarray) -> str:
    area = int(mask.sum())
    exposed = _mask_exposed_edges(mask)
    if exposed >= 3 * area:
        return "high"
    if exposed >= 2 * area:
        return "medium"
    return "low"


def _build_library(bases: tuple[np.ndarray, ...], expected_category: str) -> tuple[np.ndarray, ...]:
    library: list[np.ndarray] = []
    seen: set[tuple[tuple[int, ...], ...]] = set()
    for base in bases:
        assert _mask_category(base) == expected_category
        for variant in _shape_variants(base):
            key = _shape_key(variant)
            if key not in seen:
                seen.add(key)
                library.append(variant.copy())
    return tuple(library)


HIGH_BASES = (
    _mask("#"),
    _mask("##"),
)

MEDIUM_BASES = (
    _mask("###"),
    _mask("##", "#."),
    _mask("##", "##"),
    _mask("####"),
    _mask("###", ".#."),
    _mask("##.", ".##"),
    _mask("###", "#.."),
    _mask("###", "##."),
)

LOW_BASES = (
    _mask("###", "###"),
    _mask("####", "####"),
    _mask("###", "###", "###"),
    _mask("####", "####", "####"),
    _mask("###", "###", "##."),
    _mask("####", "####", "###."),
    _mask("####", "####", "####", "###."),
)

HIGH_LIBRARY = _build_library(HIGH_BASES, "high")
MEDIUM_LIBRARY = _build_library(MEDIUM_BASES, "medium")
LOW_LIBRARY = _build_library(LOW_BASES, "low")
CATEGORY_TO_COLOR = {"high": RED, "medium": GREEN, "low": BLUE}
CORNERS = ("TL", "TR", "BL", "BR")

# --------------- solver ---------------


def _legend_coords(shape: tuple[int, int], corner: str) -> list[tuple[int, int]]:
    h, w = shape
    if corner == "TL":
        return [(0, 0), (1, 0), (2, 0)]
    if corner == "TR":
        return [(0, w - 1), (1, w - 1), (2, w - 1)]
    if corner == "BL":
        return [(h - 3, 0), (h - 2, 0), (h - 1, 0)]
    return [(h - 3, w - 1), (h - 2, w - 1), (h - 1, w - 1)]


def _find_legend_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    assert h >= 3 and w >= 1
    candidates = []
    for corner in CORNERS:
        coords = _legend_coords((h, w), corner)
        colors = tuple(int(grid[r, c]) for r, c in coords)
        if colors == LEGEND:
            candidates.append(coords)
    assert len(candidates) == 1, "expected exactly one corner legend"
    mask = np.zeros_like(grid, dtype=bool)
    for r, c in candidates[0]:
        mask[r, c] = True
    return mask


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _component_exposed_edges(grid: np.ndarray, comp: list[tuple[int, int]]) -> int:
    h, w = grid.shape
    exposed = 0
    for r, c in comp:
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w and int(grid[rr, cc]) == BG:
                exposed += 1
    return exposed


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor each foreground object by its exposed-edge-per-pixel ratio."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    legend_mask = _find_legend_mask(g)
    object_mask = (g != BG) & ~legend_mask

    out = g.copy()
    for comp in _components(object_mask):
        area = len(comp)
        exposed = _component_exposed_edges(g, comp)
        if exposed >= 3 * area:
            color = RED
        elif exposed >= 2 * area:
            color = GREEN
        else:
            color = BLUE
        for r, c in comp:
            out[r, c] = color
    return out


# --------------- generator ---------------

GRID_H_RANGE = (12, 20)
GRID_W_RANGE = (12, 20)
CATEGORY_COUNT_RANGE = (1, 2)
PLACEMENT_TRIES = 300
GEN_ATTEMPTS = 300


def _sample_objects() -> list[tuple[str, np.ndarray]]:
    objects: list[tuple[str, np.ndarray]] = []
    for category, library in (
        ("high", HIGH_LIBRARY),
        ("medium", MEDIUM_LIBRARY),
        ("low", LOW_LIBRARY),
    ):
        count = random.randint(*CATEGORY_COUNT_RANGE)
        for _ in range(count):
            objects.append((category, random.choice(library)))
    random.shuffle(objects)
    objects.sort(key=lambda item: int(item[1].sum()), reverse=True)
    return objects


def _blocked_cells(shape: tuple[int, int], corner: str) -> np.ndarray:
    h, w = shape
    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True
    for r, c in _legend_coords(shape, corner):
        blocked[r, c] = True
        for dr, dc in DIRS4:
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                blocked[rr, cc] = True
    return blocked


def _can_place(mask: np.ndarray, top: int, left: int, occupied: np.ndarray, blocked: np.ndarray) -> bool:
    h, w = occupied.shape
    mh, mw = mask.shape
    for r in range(mh):
        for c in range(mw):
            if not bool(mask[r, c]):
                continue
            rr, cc = top + r, left + c
            if not (0 <= rr < h and 0 <= cc < w):
                return False
            if bool(blocked[rr, cc]) or bool(occupied[rr, cc]):
                return False
            for dr, dc in DIRS4:
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w and bool(occupied[nr, nc]):
                    return False
    return True


def _place_object(
    grid: np.ndarray,
    expected: np.ndarray,
    occupied: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    object_color: int,
    category: str,
) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    positions = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(positions)
    for top, left in positions[:PLACEMENT_TRIES]:
        if not _can_place(mask, top, left, occupied, blocked):
            continue
        paint = CATEGORY_TO_COLOR[category]
        for r in range(mh):
            for c in range(mw):
                if bool(mask[r, c]):
                    rr, cc = top + r, left + c
                    grid[rr, cc] = object_color
                    expected[rr, cc] = paint
                    occupied[rr, cc] = True
        return True
    return False


def _add_legend(grid: np.ndarray, corner: str) -> None:
    for color, (r, c) in zip(LEGEND, _legend_coords(grid.shape, corner)):
        grid[r, c] = color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_H_RANGE)
    w = random.randint(*GRID_W_RANGE)
    corner = random.choice(CORNERS)
    object_color = random.choice(OBJECT_COLORS)
    objects = _sample_objects()

    grid = np.full((h, w), BG, dtype=int)
    expected = np.full((h, w), BG, dtype=int)
    _add_legend(grid, corner)
    _add_legend(expected, corner)

    blocked = _blocked_cells((h, w), corner)
    occupied = np.zeros((h, w), dtype=bool)
    for category, mask in objects:
        if not _place_object(grid, expected, occupied, blocked, mask, object_color, category):
            return None

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return None

    legend_mask = _find_legend_mask(grid)
    output_colors = {int(v) for v in np.unique(expected[(expected != BG) & ~legend_mask])}
    if output_colors != {RED, GREEN, BLUE}:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
