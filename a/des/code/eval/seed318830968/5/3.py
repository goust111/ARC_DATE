from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
MAGENTA = 6
ORANGE = 7
CYAN = 8
BROWN = 9

HORIZONTAL = "horizontal"
VERTICAL = "vertical"
ROTATION = "rotation"
SYMMETRY_TYPES = (HORIZONTAL, VERTICAL, ROTATION)
SYMMETRY_TO_BORDER = {
    HORIZONTAL: RED,
    VERTICAL: BLUE,
    ROTATION: GREEN,
}

FRAME_COLORS = (YELLOW, MAGENTA, CYAN, BROWN)
GRID_SIZE_RANGE = (28, 34)
REGION_COUNT_RANGE = (3, 4)
INTERIOR_SIZE_RANGE = (5, 8)
MAX_GENERATION_ATTEMPTS = 2000


def _apply_symmetry(arr: np.ndarray, kind: str) -> np.ndarray:
    if kind == HORIZONTAL:
        return np.flipud(arr)
    if kind == VERTICAL:
        return np.fliplr(arr)
    if kind == ROTATION:
        return np.rot90(arr, 2)
    raise ValueError(kind)


def _partner(shape: tuple[int, int], r: int, c: int, kind: str) -> tuple[int, int]:
    h, w = shape
    if kind == HORIZONTAL:
        return h - 1 - r, c
    if kind == VERTICAL:
        return r, w - 1 - c
    if kind == ROTATION:
        return h - 1 - r, w - 1 - c
    raise ValueError(kind)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            part: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                part.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            parts.append(part)
    return parts


def _find_frames(grid: np.ndarray) -> list[tuple[int, tuple[int, int, int, int]]]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in (BLACK, ORANGE)]
    frames: list[tuple[int, tuple[int, int, int, int]]] = []
    for color in colors:
        for comp in _components(grid == color):
            rows = [p[0] for p in comp]
            cols = [p[1] for p in comp]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            if h < 3 or w < 3:
                continue
            submask = grid[r0 : r1 + 1, c0 : c1 + 1] == color
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True
            if np.array_equal(submask, border):
                frames.append((color, (r0, c0, r1, c1)))
    frames.sort(key=lambda item: item[1])
    return frames


def _symmetry_scores(inner: np.ndarray) -> dict[str, int]:
    scores: dict[str, int] = {}
    for kind in SYMMETRY_TYPES:
        other = _apply_symmetry(inner, kind)
        known = (inner != ORANGE) & (other != ORANGE)
        scores[kind] = int(np.count_nonzero(known & (inner != other)))
    return scores


def _classify_symmetry(inner: np.ndarray) -> str:
    scores = _symmetry_scores(inner)
    best = min(scores.values())
    winners = [kind for kind, score in scores.items() if score == best]
    assert len(winners) == 1, f"ambiguous symmetry scores: {scores}"
    return winners[0]


def _repair_inner(inner: np.ndarray, kind: str, pattern_color: int) -> np.ndarray:
    fixed = inner.copy()
    original = inner.copy()
    h, w = inner.shape
    for r in range(h):
        for c in range(w):
            if original[r, c] != ORANGE:
                continue
            pr, pc = _partner((h, w), r, c, kind)
            partner_value = int(original[pr, pc])
            fixed[r, c] = pattern_color if partner_value == ORANGE else partner_value
    return fixed


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair noisy symmetric patterns inside framed rectangles and recolor each frame by symmetry type.

    The legend is left unchanged; only hollow rectangle regions are processed.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    frames = _find_frames(g)
    assert frames, "expected at least one rectangular frame"

    for border_color, (r0, c0, r1, c1) in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1].copy()
        inner_colors = set(map(int, np.unique(inner))) - {BLACK, ORANGE}
        assert len(inner_colors) == 1, f"expected one pattern color inside frame, got {inner_colors}"
        pattern_color = int(next(iter(inner_colors)))
        assert pattern_color != border_color

        kind = _classify_symmetry(inner)
        repaired = _repair_inner(inner, kind, pattern_color)
        out[r0 + 1 : r1, c0 + 1 : c1] = repaired

        new_border = SYMMETRY_TO_BORDER[kind]
        out[r0, c0 : c1 + 1] = new_border
        out[r1, c0 : c1 + 1] = new_border
        out[r0 : r1 + 1, c0] = new_border
        out[r0 : r1 + 1, c1] = new_border

    return out


def _orbits(h: int, w: int, kind: str) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[tuple[int, int], ...]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            orbit = tuple(sorted({(r, c), _partner((h, w), r, c, kind)}))
            if orbit not in seen:
                seen.add(orbit)
                orbits.append(orbit)
    return orbits


def _has_adjacent_pair(mask: np.ndarray) -> bool:
    h, w = mask.shape
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            if r + 1 < h and mask[r + 1, c]:
                return True
            if c + 1 < w and mask[r, c + 1]:
                return True
    return False


def _make_symmetric_mask(h: int, w: int, kind: str) -> np.ndarray | None:
    orbit_list = _orbits(h, w, kind)
    area = h * w
    min_cells = max(6, area // 5)
    max_cells = (area * 11) // 20

    for _ in range(250):
        p = random.uniform(0.25, 0.5)
        mask = np.zeros((h, w), dtype=bool)
        for orbit in orbit_list:
            if random.random() < p:
                for r, c in orbit:
                    mask[r, c] = True

        filled = int(mask.sum())
        if not (min_cells <= filled <= max_cells):
            continue
        if len(np.unique(np.argwhere(mask)[:, 0])) < 2 or len(np.unique(np.argwhere(mask)[:, 1])) < 2:
            continue
        if not _has_adjacent_pair(mask):
            continue
        if any(np.array_equal(mask, _apply_symmetry(mask, other)) for other in SYMMETRY_TYPES if other != kind):
            continue
        return mask
    return None


def _mask_to_inner(mask: np.ndarray, orange_cells: set[tuple[int, int]], pattern_color: int = YELLOW) -> np.ndarray:
    inner = np.zeros(mask.shape, dtype=int)
    inner[mask] = pattern_color
    for r, c in orange_cells:
        inner[r, c] = ORANGE
    return inner


def _choose_noise(mask: np.ndarray, kind: str) -> set[tuple[int, int]] | None:
    orbit_list = _orbits(mask.shape[0], mask.shape[1], kind)
    filled_orbits = [orbit for orbit in orbit_list if all(mask[r, c] for r, c in orbit)]
    filled_cells = [cell for orbit in filled_orbits for cell in orbit]
    pair_orbits = [orbit for orbit in filled_orbits if len(orbit) == 2]
    total = int(mask.sum())

    for _ in range(250):
        orange_cells: set[tuple[int, int]] = set()
        if pair_orbits and random.random() < 0.45:
            orange_cells.update(random.choice(pair_orbits))

        remaining_budget = total - len(orange_cells) - 3
        if remaining_budget < 0:
            continue
        extra_max = min(2, remaining_budget)
        extra = random.randint(0, extra_max)
        if not orange_cells and extra == 0:
            extra = 1

        choices = [cell for cell in filled_cells if cell not in orange_cells]
        if extra > len(choices):
            continue
        if extra:
            orange_cells.update(random.sample(choices, extra))

        inner = _mask_to_inner(mask, orange_cells)
        scores = _symmetry_scores(inner)
        best = min(scores.values())
        winners = [name for name, score in scores.items() if score == best]
        if winners != [kind]:
            continue
        if scores[kind] != 0:
            continue
        return orange_cells
    return None


def _legend_patch() -> np.ndarray:
    patch = np.zeros((11, 7), dtype=int)
    items = (
        (0, RED, HORIZONTAL),
        (4, BLUE, VERTICAL),
        (8, GREEN, ROTATION),
    )
    for top, color, kind in items:
        patch[top : top + 3, 0:3] = color
        if kind == HORIZONTAL:
            patch[top + 1, 4:7] = GREY
        elif kind == VERTICAL:
            patch[top : top + 3, 5] = GREY
        else:
            patch[top : top + 2, 4:6] = GREY
    return patch


def _reserve(blocked: np.ndarray, top: int, left: int, height: int, width: int, pad: int = 1) -> None:
    h, w = blocked.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + height + pad)
    c1 = min(w, left + width + pad)
    blocked[r0:r1, c0:c1] = True


def _find_slots(blocked: np.ndarray, height: int, width: int, pad: int = 1) -> list[tuple[int, int]]:
    h, w = blocked.shape
    slots: list[tuple[int, int]] = []
    for top in range(h - height + 1):
        for left in range(w - width + 1):
            r0 = max(0, top - pad)
            c0 = max(0, left - pad)
            r1 = min(h, top + height + pad)
            c1 = min(w, left + width + pad)
            if not blocked[r0:r1, c0:c1].any():
                slots.append((top, left))
    return slots


def _frame_input(spec: dict[str, object]) -> np.ndarray:
    mask = spec["mask"]
    assert isinstance(mask, np.ndarray)
    inner = np.zeros(mask.shape, dtype=int)
    inner[mask] = int(spec["pattern_color"])
    for r, c in spec["orange_cells"]:
        inner[r, c] = ORANGE

    h, w = mask.shape
    frame = np.zeros((h + 2, w + 2), dtype=int)
    border_color = int(spec["border_color"])
    frame[0, :] = border_color
    frame[-1, :] = border_color
    frame[:, 0] = border_color
    frame[:, -1] = border_color
    frame[1:-1, 1:-1] = inner
    return frame


def _draw_frame_output(out: np.ndarray, top: int, left: int, spec: dict[str, object]) -> None:
    mask = spec["mask"]
    assert isinstance(mask, np.ndarray)
    h, w = mask.shape
    border_color = SYMMETRY_TO_BORDER[str(spec["symmetry"])]
    pattern_color = int(spec["pattern_color"])

    out[top, left : left + w + 2] = border_color
    out[top + h + 1, left : left + w + 2] = border_color
    out[top : top + h + 2, left] = border_color
    out[top : top + h + 2, left + w + 1] = border_color

    inner = out[top + 1 : top + h + 1, left + 1 : left + w + 1]
    inner[:] = BLACK
    inner[mask] = pattern_color


def _generate_frame_spec(symmetry: str, border_color: int) -> dict[str, object] | None:
    for _ in range(250):
        h = random.randint(*INTERIOR_SIZE_RANGE)
        w = random.randint(*INTERIOR_SIZE_RANGE)
        mask = _make_symmetric_mask(h, w, symmetry)
        if mask is None:
            continue
        orange_cells = _choose_noise(mask, symmetry)
        if orange_cells is None:
            continue
        pattern_choices = [color for color in FRAME_COLORS if color != border_color]
        return {
            "symmetry": symmetry,
            "border_color": border_color,
            "pattern_color": random.choice(pattern_choices),
            "mask": mask,
            "orange_cells": orange_cells,
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_SIZE_RANGE)
    grid_w = random.randint(*GRID_SIZE_RANGE)
    grid = np.zeros((grid_h, grid_w), dtype=int)

    legend = _legend_patch()
    blocked = np.zeros_like(grid, dtype=bool)
    legend_slots = _find_slots(blocked, legend.shape[0], legend.shape[1], pad=1)
    if not legend_slots:
        return None
    legend_top, legend_left = random.choice(legend_slots)
    grid[legend_top : legend_top + legend.shape[0], legend_left : legend_left + legend.shape[1]] = legend
    _reserve(blocked, legend_top, legend_left, legend.shape[0], legend.shape[1], pad=1)

    n_regions = random.randint(*REGION_COUNT_RANGE)
    symmetries = [HORIZONTAL, VERTICAL, ROTATION]
    while len(symmetries) < n_regions:
        symmetries.append(random.choice(SYMMETRY_TYPES))
    random.shuffle(symmetries)

    border_colors = random.sample(FRAME_COLORS, n_regions)
    specs: list[dict[str, object]] = []
    for symmetry, border_color in zip(symmetries, border_colors):
        spec = _generate_frame_spec(symmetry, border_color)
        if spec is None:
            return None
        specs.append(spec)

    specs.sort(key=lambda item: int(np.prod(_frame_input(item).shape)), reverse=True)
    placements: list[tuple[int, int, dict[str, object]]] = []
    for spec in specs:
        frame = _frame_input(spec)
        slots = _find_slots(blocked, frame.shape[0], frame.shape[1], pad=1)
        if not slots:
            return None
        top, left = random.choice(slots)
        grid[top : top + frame.shape[0], left : left + frame.shape[1]] = frame
        _reserve(blocked, top, left, frame.shape[0], frame.shape[1], pad=1)
        placements.append((top, left, spec))

    expected = grid.copy()
    for top, left, spec in placements:
        _draw_frame_output(expected, top, left, spec)

    try:
        solved = main(grid)
    except Exception:
        return None
    if not np.array_equal(solved, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
