from __future__ import annotations

import random

import numpy as np


RED = 2
BLUE = 1
GREEN = 3
MAGENTA = 6

DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _dominant_color(grid: np.ndarray) -> int:
    colors, counts = np.unique(grid, return_counts=True)
    return int(colors[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(cells, dtype=int))

    return components


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    r1 = int(coords[:, 0].max())
    c1 = int(coords[:, 1].max())
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _symmetry_kind(mask: np.ndarray) -> str | None:
    horizontal = bool(np.array_equal(mask, mask[::-1, :]))
    vertical = bool(np.array_equal(mask, mask[:, ::-1]))
    anti_diagonal = bool(np.array_equal(mask, np.flipud(np.fliplr(mask)).T))
    diagonal = bool(
        mask.shape[0] == mask.shape[1]
        and (np.array_equal(mask, mask.T) or anti_diagonal)
    )

    if sum((horizontal, vertical, diagonal)) != 1:
        return None
    if horizontal:
        return "H"
    if vertical:
        return "V"
    return "D"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor red objects by reflection-symmetry type and erase singleton noise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _dominant_color(g)
    extra_colors = set(map(int, np.unique(g))) - {bg}
    assert extra_colors == {RED}, "expected red to be the only foreground color"

    out = np.full_like(g, bg)
    for coords in _find_components(g == RED):
        if len(coords) == 1:
            continue
        kind = _symmetry_kind(_component_mask(coords))
        assert kind is not None, "every non-singleton object must have exactly one symmetry type"
        color = {"H": BLUE, "V": GREEN, "D": MAGENTA}[kind]
        out[coords[:, 0], coords[:, 1]] = color

    return out


# --------------- generator ---------------

BACKGROUND_CHOICES = (0, 4, 5, 7, 8, 9)
GRID_ROWS = (18, 28)
GRID_COLS = (18, 28)
OBJECT_COUNT_RANGE = (3, 7)
NOISE_COUNT_RANGE = (2, 8)
GENERATION_RETRIES = 400
PLACEMENT_TRIES = 300


def _orbit(kind: str, shape: tuple[int, int], r: int, c: int) -> tuple[tuple[int, int], ...]:
    h, w = shape
    if kind == "H":
        cells = {(r, c), (h - 1 - r, c)}
    elif kind == "V":
        cells = {(r, c), (r, w - 1 - c)}
    else:
        cells = {(r, c), (c, r)}
    return tuple(sorted(cells))


def _touches(mask: np.ndarray, cells: tuple[tuple[int, int], ...]) -> bool:
    h, w = mask.shape
    for r, c in cells:
        if mask[r, c]:
            continue
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                return True
    return False


def _expand_symmetric(mask: np.ndarray, kind: str, steps: int) -> np.ndarray:
    for _ in range(steps):
        choices: list[tuple[tuple[int, int], ...]] = []
        seen: set[tuple[tuple[int, int], ...]] = set()
        h, w = mask.shape
        for r in range(h):
            for c in range(w):
                if mask[r, c]:
                    continue
                orbit = _orbit(kind, mask.shape, r, c)
                if orbit in seen or not _touches(mask, orbit):
                    continue
                seen.add(orbit)
                trial = mask.copy()
                for rr, cc in orbit:
                    trial[rr, cc] = True
                if _symmetry_kind(trial) == kind:
                    choices.append(orbit)
        if not choices:
            break
        for rr, cc in random.choice(choices):
            mask[rr, cc] = True
    return mask


def _horizontal_mask() -> np.ndarray:
    h = random.choice((3, 5, 7, 9))
    w = random.randint(4, 10)
    axis_row = h // 2
    arm_choices = [c for c in range(w) if c != w - 1 - c]
    arm_col = random.choice(arm_choices)

    mask = np.zeros((h, w), dtype=bool)
    mask[axis_row, :] = True
    mask[:, arm_col] = True
    steps = random.randint(0, max(1, (h * w) // 8))
    mask = _expand_symmetric(mask, "H", steps)
    assert _symmetry_kind(mask) == "H"
    return mask


def _vertical_mask() -> np.ndarray:
    h = random.randint(4, 10)
    w = random.choice((3, 5, 7, 9))
    axis_col = w // 2
    arm_choices = [r for r in range(h) if r != h - 1 - r]
    arm_row = random.choice(arm_choices)

    mask = np.zeros((h, w), dtype=bool)
    mask[:, axis_col] = True
    mask[arm_row, :] = True
    steps = random.randint(0, max(1, (h * w) // 8))
    mask = _expand_symmetric(mask, "V", steps)
    assert _symmetry_kind(mask) == "V"
    return mask


def _diagonal_mask() -> np.ndarray:
    n = random.randint(4, 8)
    width = 1 if n <= 5 else random.choice((1, 1, 2))
    rr, cc = np.indices((n, n))
    mask = np.abs(rr - cc) <= width
    steps = random.randint(0, max(1, n // 2))
    mask = _expand_symmetric(mask.copy(), "D", steps)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    assert _symmetry_kind(mask) == "D"
    return mask


def _make_object_mask(kind: str) -> np.ndarray:
    if kind == "H":
        return _horizontal_mask()
    if kind == "V":
        return _vertical_mask()
    return _diagonal_mask()


def _mask_cells(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(mask)
    return [(top + int(r), left + int(c)) for r, c in coords]


def _can_place(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> bool:
    h, w = blocked.shape
    mh, mw = mask.shape
    if top + mh > h or left + mw > w:
        return False
    return all(not blocked[r, c] for r, c in _mask_cells(mask, top, left))


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        blocked[r, c] = True
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_mask(grid: np.ndarray, blocked: np.ndarray, mask: np.ndarray) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    candidates = [(r, c) for r in range(h - mh + 1) for c in range(w - mw + 1)]
    random.shuffle(candidates)

    for top, left in candidates[:PLACEMENT_TRIES]:
        if not _can_place(blocked, mask, top, left):
            continue
        cells = _mask_cells(mask, top, left)
        for r, c in cells:
            grid[r, c] = RED
        _mark_blocked(blocked, cells)
        return True
    return False


def _place_noise(grid: np.ndarray, blocked: np.ndarray, count: int) -> bool:
    h, w = grid.shape
    for _ in range(count):
        cells = [(r, c) for r in range(h) for c in range(w) if not blocked[r, c]]
        if not cells:
            return False
        r, c = random.choice(cells)
        grid[r, c] = RED
        _mark_blocked(blocked, [(r, c)])
    return True


def _sample_kind_list() -> list[str]:
    total = random.randint(*OBJECT_COUNT_RANGE)
    kinds = ["H", "V", "D"]
    while len(kinds) < total:
        kinds.append(random.choice(("H", "V", "D")))
    random.shuffle(kinds)
    return kinds


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    bg = random.choice(BACKGROUND_CHOICES)
    grid = np.full((h, w), bg, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    kinds = _sample_kind_list()
    masks = [_make_object_mask(kind) for kind in kinds]
    masks.sort(key=lambda m: int(m.sum()), reverse=True)
    for mask in masks:
        if not _place_mask(grid, blocked, mask):
            return None

    noise_count = random.randint(*NOISE_COUNT_RANGE)
    if not _place_noise(grid, blocked, noise_count):
        return None

    out = main(grid)
    out_colors = set(map(int, np.unique(out)))
    assert {BLUE, GREEN, MAGENTA}.issubset(out_colors)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
