import random

import numpy as np


BG = 0
SEED_COLOR = 3
CENTER_COLOR = 7
TEMPLATE_SHAPE = (2, 2)
SPIRAL_DIRS = ((0, -1), (1, 0), (0, 1), (-1, 0))
GRID_SIZE_CHOICES = (11, 13, 15, 17)
TEMPLATE_COLORS = (1, 2, 4, 5, 6, 8, 9)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def _spiral_path(shape: tuple[int, int], start: tuple[int, int]) -> list[tuple[int, int]]:
    path = [start]
    r, c = start
    step_len = 1
    direction_index = 0
    segments_at_len = 0

    while True:
        dr, dc = SPIRAL_DIRS[direction_index]
        for _ in range(step_len):
            nr, nc = r + dr, c + dc
            if not _in_bounds(shape, nr, nc):
                return path
            path.append((nr, nc))
            r, c = nr, nc
        direction_index = (direction_index + 1) % len(SPIRAL_DIRS)
        segments_at_len += 1
        if segments_at_len == 2:
            segments_at_len = 0
            step_len += 1


def _simulate(grid: np.ndarray, quota: int | None = None) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    center_cells = np.argwhere(g == CENTER_COLOR)
    assert center_cells.shape[0] == 1, "expected exactly one center marker"
    start = (int(center_cells[0, 0]), int(center_cells[0, 1]))
    template = g[:2, :2].copy()
    path = _spiral_path(g.shape, start)
    seed_count = sum(int(g[r, c]) == SEED_COLOR for r, c in path)
    limit = seed_count if quota is None else int(quota)

    out = g.copy()
    anchors: list[tuple[int, int]] = []
    skip_nonempty = 0
    skip_oob = 0
    skip_blocked = 0

    for r, c in path:
        if len(anchors) >= limit:
            break
        if int(out[r, c]) != BG:
            skip_nonempty += 1
            continue
        if r + TEMPLATE_SHAPE[0] > h or c + TEMPLATE_SHAPE[1] > w:
            skip_oob += 1
            continue
        if np.any(out[r : r + 2, c : c + 2] != BG):
            skip_blocked += 1
            continue
        out[r : r + 2, c : c + 2] = template
        anchors.append((r, c))

    return {
        "path": path,
        "seed_count": seed_count,
        "anchors": anchors,
        "skip_nonempty": skip_nonempty,
        "skip_oob": skip_oob,
        "skip_blocked": skip_blocked,
        "output": out,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Count spiral seed pixels and stamp the top-left sprite onto the first valid empty spiral cells.

    Input structure:
    - The grid has black background, a 2x2 sprite template in the top-left corner,
      one orange center marker, and several green seed pixels.
    - The seed pixels lie on the square spiral traced from the center marker.
    - The center marker and the template are part of the final output and also act as obstacles.

    Rule:
    - Copy the 2x2 top-left sprite template.
    - Starting at the orange marker, follow the spiral left, down, right, up with
      segment lengths 1, 1, 2, 2, 3, 3, ... and truncate the walk when the next
      step would leave the grid.
    - Count how many green seed pixels lie on that spiral path.
    - Traverse the same spiral again and, for the first k empty black cells
      (where k is the seed count), stamp a copy of the 2x2 template using that
      cell as the stamp's top-left corner whenever the whole 2x2 area is in-bounds
      and still all black; otherwise skip that spiral cell.

    Key insight:
    - The spiral supplies a unique global order, while the green seeds both block
      placement cells and set the number of sprite copies that must be stamped.
    """
    return _simulate(np.asarray(grid, dtype=int))["output"]


def _make_template() -> np.ndarray:
    a, b = random.sample(TEMPLATE_COLORS, 2)
    patterns = (
        np.array([[0, 1], [1, 0]], dtype=int),
        np.array([[0, 1], [0, 1]], dtype=int),
        np.array([[0, 0], [1, 1]], dtype=int),
        np.array([[0, 0], [0, 1]], dtype=int),
    )
    pattern = random.choice(patterns)
    template = np.where(pattern == 0, a, b).astype(int)
    if random.random() < 0.5:
        template = np.rot90(template, k=random.randint(1, 3))
    return template


def _choose_center(size: int, mode: str) -> tuple[int, int]:
    valid = [
        (r, c)
        for r in range(3, size - 3)
        for c in range(3, size - 3)
    ]
    if mode == "central":
        focus = [
            (r, c)
            for r, c in valid
            if abs(r - size // 2) <= 2 and abs(c - size // 2) <= 2
        ]
    elif mode == "bottom_right":
        focus = [
            (r, c)
            for r, c in valid
            if r >= size - 5 or c >= size - 5
        ]
    else:
        focus = [
            (r, c)
            for r, c in valid
            if min(r, c, size - 1 - r, size - 1 - c) <= 4
        ]
    return random.choice(focus or valid)


def _is_template_cell(cell: tuple[int, int]) -> bool:
    r, c = cell
    return r < 2 and c < 2


def _well_spaced(cells: list[tuple[int, int]]) -> bool:
    for i, (r1, c1) in enumerate(cells):
        for r2, c2 in cells[i + 1 :]:
            if max(abs(r1 - r2), abs(c1 - c2)) <= 1:
                return False
    return True


def _pick_seed_cells(
    mode: str,
    k: int,
    usable: list[tuple[int, int]],
    baseline: dict,
) -> list[tuple[int, int]] | None:
    early_success = baseline["anchors"][: min(len(baseline["anchors"]), max(5, k + 2))]
    early_success_set = set(early_success)
    late_pool = usable[len(usable) // 2 :]
    quota_pool = [cell for cell in late_pool if cell not in early_success_set]
    edge_pool = [
        cell for cell in usable if cell[0] in (0, baseline["output"].shape[0] - 1) or cell[1] in (0, baseline["output"].shape[1] - 1)
    ]

    for _ in range(80):
        chosen: list[tuple[int, int]] = []
        if mode == "quota":
            pool = quota_pool if len(quota_pool) >= k else late_pool if len(late_pool) >= k else usable
            if len(pool) < k:
                continue
            chosen = random.sample(pool, k)
        elif mode == "obstacle":
            if not early_success:
                continue
            forced = random.randint(1, min(k, len(early_success)))
            chosen.extend(random.sample(early_success, forced))
            remainder = [cell for cell in (late_pool or usable) if cell not in chosen]
            if len(remainder) < k - forced:
                continue
            chosen.extend(random.sample(remainder, k - forced))
        else:
            if not edge_pool:
                continue
            edge_forced = random.randint(1, min(k, len(edge_pool), 2))
            chosen.extend(random.sample(edge_pool, edge_forced))
            remaining_slots = k - edge_forced
            if remaining_slots > 0 and early_success:
                early_forced = min(remaining_slots, random.randint(1, min(remaining_slots, len(early_success))))
                early_choices = [cell for cell in early_success if cell not in chosen]
                if len(early_choices) < early_forced:
                    continue
                chosen.extend(random.sample(early_choices, early_forced))
            remainder = [cell for cell in usable if cell not in chosen]
            if len(remainder) < k - len(chosen):
                continue
            chosen.extend(random.sample(remainder, k - len(chosen)))

        if _well_spaced(chosen):
            random.shuffle(chosen)
            return chosen
    return None


def _gen_attempt(mode: str | None = None) -> np.ndarray | None:
    mode = random.choice(("quota", "obstacle", "edge")) if mode is None else mode
    size = random.choice((11, 13)) if mode == "edge" else random.choice(GRID_SIZE_CHOICES)
    center_mode = "bottom_right" if mode == "edge" else random.choice(("central", "offset"))
    template = _make_template()
    center = _choose_center(size, center_mode)

    base = np.zeros((size, size), dtype=int)
    base[:2, :2] = template
    base[center] = CENTER_COLOR
    baseline = _simulate(base, quota=size * size)

    usable = [
        cell
        for cell in baseline["path"]
        if cell != center and not _is_template_cell(cell)
    ]
    if len(usable) < 12 or len(baseline["anchors"]) < 10:
        return None

    max_k = min(5, len(baseline["anchors"]) // 3)
    if max_k < 1:
        return None
    if mode == "edge":
        k = random.randint(min(max_k, max(3, max_k - 1)), max_k)
    else:
        k = random.randint(1, max_k)

    for _ in range(100):
        seeds = _pick_seed_cells(mode, k, usable, baseline)
        if seeds is None:
            return None

        inp = base.copy()
        for r, c in seeds:
            inp[r, c] = SEED_COLOR

        final = _simulate(inp)
        if final["seed_count"] != k or len(final["anchors"]) != k:
            continue
        if np.array_equal(final["output"], inp):
            continue

        baseline_prefix = baseline["anchors"][:k]
        if mode == "quota":
            if any(seed in set(baseline_prefix) for seed in seeds):
                continue
            if final["anchors"] != baseline_prefix:
                continue
        elif mode == "obstacle":
            if not any(seed in set(baseline["anchors"][: k + 2]) for seed in seeds):
                continue
            if final["anchors"] == baseline_prefix:
                continue
        else:
            if not any(
                r in (0, size - 1) or c in (0, size - 1) for r, c in seeds
            ):
                continue
            if final["skip_oob"] == 0:
                continue

        if not np.array_equal(main(inp), final["output"]):
            continue
        return inp
    return None


def generate_input() -> np.ndarray:
    preferred_mode = random.choices(
        ("quota", "obstacle", "edge"),
        weights=(3, 3, 4),
        k=1,
    )[0]
    for _ in range(GEN_ATTEMPTS // 2):
        grid = _gen_attempt(preferred_mode)
        if grid is not None:
            return grid
    for _ in range(GEN_ATTEMPTS // 2):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid spiral stamp input")
