from __future__ import annotations

import random

import numpy as np


BLACK = 0
AZURE = 8
FOREGROUND_COLORS = (1, 2, 3, 4, 5, 6, 7, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _checkerboard(shape: tuple[int, int], phase: int) -> np.ndarray:
    rows, cols = np.indices(shape)
    return np.where((rows + cols + int(phase)) % 2 == 0, AZURE, BLACK).astype(int)


def _infer_phase(g: np.ndarray) -> int:
    visible = (g == BLACK) | (g == AZURE)
    assert bool(np.any(visible)), "expected visible checkerboard cells"

    valid = []
    for phase in (0, 1):
        board = _checkerboard(g.shape, phase)
        if bool(np.all(g[visible] == board[visible])):
            valid.append(int(phase))

    assert len(valid) == 1, "checkerboard phase must be uniquely determined"
    return valid[0]


def _foreground_components(g: np.ndarray) -> list[tuple[int, np.ndarray]]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in {BLACK, AZURE} or seen[r, c]:
                continue

            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((int(nr), int(nc)))

            components.append((color, np.array(coords, dtype=int)))

    assert components, "expected at least one foreground object"
    return components


def _mirror_horizontally(coords: np.ndarray) -> np.ndarray:
    mirrored = np.array(coords, dtype=int, copy=True)
    c0 = int(coords[:, 1].min())
    c1 = int(coords[:, 1].max())
    mirrored[:, 1] = c0 + c1 - coords[:, 1]
    return mirrored


def _checkerboard_counts(board: np.ndarray, coords: np.ndarray) -> tuple[int, int]:
    colors = board[coords[:, 0], coords[:, 1]]
    azure_count = int(np.count_nonzero(colors == AZURE))
    black_count = int(len(coords) - azure_count)
    return azure_count, black_count


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep, erase, or horizontally mirror each object based on how it covers the hidden checkerboard.

    Input structure:
    - The canvas is an azure/black 1-cell checkerboard, partially covered by foreground objects.
    - Every foreground object is a monochrome 4-connected component whose color is neither azure nor black.
    - Outside the foreground objects, the checkerboard remains directly visible.

    Rule:
    - Infer the checkerboard phase from the visible azure/black cells.
    - For each foreground object, count how many of its covered cells lie on azure checkerboard cells versus black checkerboard cells.
    - If azure > black, keep the object unchanged. If azure < black, remove it and reveal the checkerboard. If azure == black, reflect the object horizontally across the vertical midline of its own bounding box.

    Key insight:
    The checkerboard is a hidden measuring device: each object's fate depends on which background parity it covers more often, not on the object's color, size, or position alone.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    phase = _infer_phase(g)
    board = _checkerboard(g.shape, phase)
    out = board.copy()

    for color, coords in _foreground_components(g):
        azure_count, black_count = _checkerboard_counts(board, coords)
        if azure_count > black_count:
            target = coords
        elif azure_count < black_count:
            continue
        else:
            target = _mirror_horizontally(coords)

        out[target[:, 0], target[:, 1]] = int(color)

    return out


# --------------- generator ---------------

GRID_HEIGHT_BOUNDS = (13, 24)
GRID_WIDTH_BOUNDS = (13, 24)
OBJECT_COUNT_BOUNDS = (3, 6)
RAW_BBOX_HEIGHT_BOUNDS = (2, 6)
RAW_BBOX_WIDTH_BOUNDS = (2, 6)
AREA_BOUNDS = (3, 10)
MASK_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 300
GENERATION_ATTEMPTS = 400


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.nonzero(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1].copy()


def _grow_mask(raw_h: int, raw_w: int, area: int) -> np.ndarray | None:
    seed = (random.randrange(raw_h), random.randrange(raw_w))
    selected = {seed}
    frontier = set()

    def add_frontier(r: int, c: int) -> None:
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < raw_h and 0 <= nc < raw_w and (nr, nc) not in selected:
                frontier.add((nr, nc))

    add_frontier(*seed)
    while len(selected) < area:
        if not frontier:
            return None
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        selected.add(cell)
        add_frontier(*cell)

    mask = np.zeros((raw_h, raw_w), dtype=bool)
    for r, c in selected:
        mask[r, c] = True
    return _trim_mask(mask)


def _randomize_mask(mask: np.ndarray) -> np.ndarray:
    out = np.array(mask, dtype=bool, copy=True)
    out = np.rot90(out, random.randrange(4))
    if random.random() < 0.5:
        out = np.fliplr(out)
    if random.random() < 0.5:
        out = np.flipud(out)
    return np.array(out, dtype=bool, copy=True)


def _mask_parity_counts(mask: np.ndarray) -> tuple[int, int]:
    rows, cols = np.indices(mask.shape)
    even = int(np.count_nonzero(mask & (((rows + cols) % 2) == 0)))
    odd = int(np.count_nonzero(mask)) - even
    return even, odd


def _make_mask(kind: str) -> np.ndarray | None:
    for _ in range(MASK_ATTEMPTS):
        raw_h = random.randint(*RAW_BBOX_HEIGHT_BOUNDS)
        raw_w = random.randint(*RAW_BBOX_WIDTH_BOUNDS)
        max_area = min(raw_h * raw_w, AREA_BOUNDS[1])
        min_area = min(AREA_BOUNDS[0], max_area)
        if max_area < min_area:
            continue

        if kind == "balanced":
            candidates = [area for area in range(max(4, min_area), max_area + 1) if area % 2 == 0]
        else:
            candidates = list(range(min_area, max_area + 1))
        if not candidates:
            continue

        area = random.choice(candidates)
        mask = _grow_mask(raw_h, raw_w, area)
        if mask is None:
            continue
        mask = _randomize_mask(mask)
        even, odd = _mask_parity_counts(mask)

        if kind == "majority" and even != odd:
            return mask
        if kind == "balanced" and even == odd and mask.shape[1] > 1 and not np.array_equal(mask, np.fliplr(mask)):
            return mask

    return None


def _required_origin_sum_parity(mask: np.ndarray, phase: int, branch: str) -> int | None:
    even, odd = _mask_parity_counts(mask)
    if even == odd:
        return None
    if branch == "keep":
        anchor = 0 if even > odd else 1
    else:
        anchor = 1 if even > odd else 0
    return int(anchor ^ int(phase))


def _branch_of(mask: np.ndarray, phase: int, origin: tuple[int, int]) -> str:
    h, w = mask.shape
    board = _checkerboard((h, w), (phase + origin[0] + origin[1]) % 2)
    coords = np.argwhere(mask)
    azure_count, black_count = _checkerboard_counts(board, coords)
    if azure_count > black_count:
        return "keep"
    if azure_count < black_count:
        return "delete"
    return "mirror"


def _place_mask(
    grid: np.ndarray,
    blocked: np.ndarray,
    mask: np.ndarray,
    color: int,
    required_sum_parity: int | None,
) -> tuple[int, int] | None:
    h, w = mask.shape
    H, W = grid.shape
    origins = [
        (r0, c0)
        for r0 in range(H - h + 1)
        for c0 in range(W - w + 1)
        if (required_sum_parity is None or (r0 + c0) % 2 == required_sum_parity)
        and not bool(np.any(blocked[r0 : r0 + h, c0 : c0 + w]))
    ]
    if not origins:
        return None

    random.shuffle(origins)
    for r0, c0 in origins[:PLACEMENT_ATTEMPTS]:
        coords = np.argwhere(mask) + np.array([r0, c0])
        grid[coords[:, 0], coords[:, 1]] = int(color)
        rr0 = max(0, r0 - 1)
        cc0 = max(0, c0 - 1)
        rr1 = min(H, r0 + h + 1)
        cc1 = min(W, c0 + w + 1)
        blocked[rr0:rr1, cc0:cc1] = True
        return int(r0), int(c0)

    return None


def _expected_branches(count: int) -> list[str]:
    branches = ["keep", "delete", "mirror"]
    while len(branches) < count:
        branches.append(random.choice(["keep", "delete", "mirror"]))
    random.shuffle(branches)
    return branches


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_BOUNDS)
    h = random.randint(*GRID_HEIGHT_BOUNDS)
    w = random.randint(*GRID_WIDTH_BOUNDS)
    phase = random.randint(0, 1)

    board = _checkerboard((h, w), phase)
    grid = board.copy()
    blocked = np.zeros((h, w), dtype=bool)
    colors = random.sample(FOREGROUND_COLORS, object_count)
    branches = _expected_branches(object_count)

    specs = []
    for color, branch in zip(colors, branches):
        if branch == "mirror":
            mask = _make_mask("balanced")
            required = None
        else:
            mask = _make_mask("majority")
            if mask is None:
                return None
            required = _required_origin_sum_parity(mask, phase, branch)
        if mask is None:
            return None
        specs.append(
            {
                "branch": branch,
                "color": int(color),
                "mask": mask,
                "required": required,
                "bbox_area": int(mask.shape[0] * mask.shape[1]),
            }
        )

    specs.sort(key=lambda spec: spec["bbox_area"], reverse=True)
    actual_branches = []
    for spec in specs:
        origin = _place_mask(grid, blocked, spec["mask"], spec["color"], spec["required"])
        if origin is None:
            return None
        actual_branches.append(_branch_of(spec["mask"], phase, origin))

    if {"keep", "delete", "mirror"} - set(actual_branches):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape or max(out.shape) > 30:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid checkerboard-object input")
