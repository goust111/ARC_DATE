from __future__ import annotations

import random

import numpy as np


BG = 8  # azure
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
PRIORITY = ("top", "right", "bottom", "left")
OUTPUT_COLOR = {
    "top": 2,     # red
    "right": 1,   # blue
    "bottom": 3,  # green
    "left": 4,    # yellow
}

# --------------- solver ---------------


def _components_monochrome4(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[tuple[int, np.ndarray]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == int(bg):
                continue

            color = int(grid[r, c])
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    stack.append((nr, nc))

            comps.append((color, np.array(coords, dtype=int)))

    return comps


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    assert rows.size > 0 and cols.size > 0, "mask must be non-empty"
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1].copy()


def _visible_run_from_start(line: np.ndarray) -> int:
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return 0
    pos = int(idx[0])
    total = 0
    while pos < line.size and bool(line[pos]):
        total += 1
        pos += 1
    return total


def _visible_run_from_end(line: np.ndarray) -> int:
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return 0
    pos = int(idx[-1])
    total = 0
    while pos >= 0 and bool(line[pos]):
        total += 1
        pos -= 1
    return total


def _visibility_scores(mask: np.ndarray) -> dict[str, int]:
    h, w = mask.shape
    top = 0
    right = 0
    bottom = 0
    left = 0

    for c in range(w):
        col = mask[:, c]
        top += _visible_run_from_start(col)
        bottom += _visible_run_from_end(col)

    for r in range(h):
        row = mask[r, :]
        left += _visible_run_from_start(row)
        right += _visible_run_from_end(row)

    return {
        "top": int(top),
        "right": int(right),
        "bottom": int(bottom),
        "left": int(left),
    }


def _winning_direction(scores: dict[str, int]) -> str:
    best = PRIORITY[0]
    best_score = int(scores[best])
    for direction in PRIORITY[1:]:
        score = int(scores[direction])
        if score > best_score:
            best = direction
            best_score = score
    return best


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each foreground object by the direction from which it shows the
    largest visible area.

    The background is azure. Each foreground object is a monochrome 4-connected
    component. For each object, evaluate four viewing directions on its tight
    bounding box:
    - top / bottom: for each column, count the first contiguous foreground run
      seen from that side
    - left / right: for each row, count the first contiguous foreground run
      seen from that side

    Sum those run lengths over all columns or rows. Repaint the whole object
    with the color of the best-scoring direction. Ties are broken by the fixed
    priority top, then right, then bottom, then left.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    out = g.copy()
    for _color, coords in _components_monochrome4(g, BG):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0 = int(rows.min())
        r1 = int(rows.max())
        c0 = int(cols.min())
        c1 = int(cols.max())

        mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        mask[rows - r0, cols - c0] = True

        direction = _winning_direction(_visibility_scores(mask))
        out[rows, cols] = int(OUTPUT_COLOR[direction])

    return out


# --------------- generator ---------------

GRID_HEIGHT_BOUNDS = (14, 22)
GRID_WIDTH_BOUNDS = (14, 22)
MIN_OBJECTS, MAX_OBJECTS = 2, 5
INPUT_COLORS = (0, 5, 6, 7, 9)
BRANCHES = ("top", "right", "bottom", "left", "tie_top", "tie_right")

MASK_HEIGHT_BOUNDS = (3, 7)
MASK_WIDTH_BOUNDS = (3, 7)
MIN_MASK_AREA = 5
MIN_FILL_RATIO = 0.35
MAX_FILL_RATIO = 0.82

MASK_RETRIES = 1200
GENERATION_RETRIES = 400
BANK_RETURN_PROB = 0.65
BANK_LIMIT = 24

_MASK_BANK: dict[str, list[np.ndarray]] = {branch: [] for branch in BRANCHES}
_MASK_KEYS: dict[str, set[tuple[tuple[int, int], bytes]]] = {branch: set() for branch in BRANCHES}


def _branch_of_scores(scores: dict[str, int]) -> str:
    max_score = max(int(v) for v in scores.values())
    winners = [direction for direction in PRIORITY if int(scores[direction]) == max_score]
    if len(winners) == 1:
        return winners[0]
    if "top" in winners:
        return "tie_top"
    if "right" in winners:
        return "tie_right"
    return "other"


def _all_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], bytes]] = set()

    for k in range(4):
        rotated = np.rot90(mask, k)
        for variant in (rotated, np.fliplr(rotated)):
            cropped = _crop_mask(np.asarray(variant, dtype=bool))
            key = (tuple(map(int, cropped.shape)), cropped.tobytes())
            if key in seen:
                continue
            seen.add(key)
            variants.append(cropped.copy())

    return variants


def _random_connected_mask() -> np.ndarray | None:
    h = random.randint(*MASK_HEIGHT_BOUNDS)
    w = random.randint(*MASK_WIDTH_BOUNDS)
    lo = max(MIN_MASK_AREA, int(np.ceil(MIN_FILL_RATIO * h * w)))
    hi = min(h * w - 1, int(np.floor(MAX_FILL_RATIO * h * w)))
    if lo > hi:
        return None

    target_area = random.randint(lo, hi)
    cells = {(random.randrange(h), random.randrange(w))}

    while len(cells) < target_area:
        frontier: list[tuple[int, int]] = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True

    mask = _crop_mask(mask)
    if min(mask.shape) < 2 or int(mask.sum()) < MIN_MASK_AREA or bool(mask.all()):
        return None
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    mask = np.rot90(mask, random.randrange(4)).copy()
    return _crop_mask(mask)


def _transform_to_branch(mask: np.ndarray, branch: str) -> np.ndarray | None:
    variants = _all_transforms(mask)
    random.shuffle(variants)
    for variant in variants:
        scores = _visibility_scores(variant)
        if len(set(scores.values())) == 1:
            continue
        if _branch_of_scores(scores) == branch:
            return variant.copy()
    return None


def _mask_for_branch(branch: str) -> np.ndarray:
    bank = _MASK_BANK[branch]
    if bank and random.random() < BANK_RETURN_PROB:
        return random.choice(bank).copy()

    for _ in range(MASK_RETRIES):
        raw = _random_connected_mask()
        if raw is None:
            continue
        mask = _transform_to_branch(raw, branch)
        if mask is None:
            continue

        key = (tuple(map(int, mask.shape)), mask.tobytes())
        if key not in _MASK_KEYS[branch]:
            _MASK_KEYS[branch].add(key)
            bank.append(mask.copy())
            if len(bank) > BANK_LIMIT:
                bank.pop(random.randrange(len(bank)))
        return mask

    if bank:
        return random.choice(bank).copy()
    raise RuntimeError(f"failed to generate a mask for branch {branch}")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask, 1, constant_values=False)
    out = np.zeros_like(mask, dtype=bool)
    for dr in range(3):
        for dc in range(3):
            out |= padded[dr : dr + mask.shape[0], dc : dc + mask.shape[1]]
    return out


def _branch_winner(branch: str) -> str:
    if branch == "tie_top":
        return "top"
    if branch == "tie_right":
        return "right"
    return branch


def _sample_branches(count: int) -> list[str]:
    for _ in range(40):
        branches = random.sample(BRANCHES, count)
        if len({_branch_winner(branch) for branch in branches}) >= 2:
            return branches
    return random.sample(BRANCHES, count)


def _place_objects(
    height: int, width: int, objects: list[dict[str, object]]
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, object]] = []

    for obj in sorted(objects, key=lambda item: int(np.sum(item["mask"])), reverse=True):
        mask = np.asarray(obj["mask"], dtype=bool)
        mh, mw = mask.shape
        if mh > height or mw > width:
            return None

        forbidden = _dilate8(occupied)
        positions = [(r0, c0) for r0 in range(height - mh + 1) for c0 in range(width - mw + 1)]
        random.shuffle(positions)

        for r0, c0 in positions:
            if np.any(forbidden[r0 : r0 + mh, c0 : c0 + mw] & mask):
                continue

            coords = np.argwhere(mask) + np.array([int(r0), int(c0)])
            color = int(obj["color"])
            grid[coords[:, 0], coords[:, 1]] = color
            occupied[coords[:, 0], coords[:, 1]] = True

            placed_obj = dict(obj)
            placed_obj["coords"] = coords
            placed.append(placed_obj)
            break
        else:
            return None

    return grid, placed


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_BOUNDS)
    width = random.randint(*GRID_WIDTH_BOUNDS)

    area = height * width
    if area < 240:
        max_objects = 3
    elif area < 320:
        max_objects = 4
    else:
        max_objects = 5
    object_count = random.randint(MIN_OBJECTS, min(MAX_OBJECTS, max_objects))

    branches = _sample_branches(object_count)
    colors = random.sample(INPUT_COLORS, object_count)

    objects: list[dict[str, object]] = []
    for branch, color in zip(branches, colors):
        mask = _mask_for_branch(branch)
        objects.append(
            {
                "branch": branch,
                "winner": _branch_winner(branch),
                "color": int(color),
                "mask": mask,
            }
        )

    placed = _place_objects(height, width, objects)
    if placed is None:
        return None
    grid, placed_objects = placed

    try:
        out = main(grid)
    except Exception:
        return None

    if len(_components_monochrome4(grid, BG)) != object_count:
        return None

    seen_output_colors = set()
    for obj in placed_objects:
        coords = np.asarray(obj["coords"], dtype=int)
        expected = int(OUTPUT_COLOR[str(obj["winner"])])
        if not bool(np.all(out[coords[:, 0], coords[:, 1]] == expected)):
            return None
        seen_output_colors.add(expected)

    if len(seen_output_colors) < 2:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
