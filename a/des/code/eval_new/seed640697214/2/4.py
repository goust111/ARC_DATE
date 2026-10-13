import random

import numpy as np


DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

DIR_ORDER = ["up", "down", "left", "right"]

# ARC palette choices used by the generator.
BG = 3  # green
FENCE = 9  # maroon
ANIMAL_COLORS = (1, 2, 4, 5, 6, 7, 8)

LAYOUTS = {
    2: [(1, 2), (2, 1), (2, 2)],
    3: [(1, 3), (3, 1), (2, 2)],
    4: [(2, 2)],
}

MAX_GENERATION_ATTEMPTS = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rect_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    local = np.zeros((h, w), dtype=bool)
    for r, c in coords:
        local[r - r0, c - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    return bool(np.array_equal(local, expected))


def _find_fences(grid: np.ndarray, bg: int) -> tuple[int, list[tuple[int, int, int, int]]]:
    best_color = -1
    best_frames: list[tuple[int, int, int, int]] = []
    best_area = -1
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != int(bg)):
        frames = []
        for comp in _components(grid == color):
            if _is_rect_frame(comp):
                frames.append(_bbox(comp))
        total_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in frames)
        if len(frames) > len(best_frames) or (len(frames) == len(best_frames) and total_area > best_area):
            best_color = int(color)
            best_frames = sorted(frames)
            best_area = int(total_area)
    if best_color < 0 or not best_frames:
        raise AssertionError("expected at least one rectangular fence")
    return best_color, best_frames


def _inside_any_bbox(pt: tuple[int, int], bboxes: list[tuple[int, int, int, int]]) -> bool:
    r, c = pt
    return any(r0 <= r <= r1 and c0 <= c <= c1 for r0, c0, r1, c1 in bboxes)


def _find_foods(
    grid: np.ndarray, bg: int, fence_color: int, fences: list[tuple[int, int, int, int]]
) -> dict[int, set[tuple[int, int]]]:
    foods: dict[int, set[tuple[int, int]]] = {}
    for color in sorted(int(v) for v in np.unique(grid) if int(v) not in {int(bg), int(fence_color)}):
        for comp in _components(grid == color):
            if len(comp) != 1:
                continue
            pt = comp[0]
            if _inside_any_bbox(pt, fences):
                continue
            foods.setdefault(int(color), set()).add((int(pt[0]), int(pt[1])))
    return foods


def _detect_direction(coords: list[tuple[int, int]]) -> tuple[str, tuple[int, int]]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)
    pts = set(coords)

    def candidate(name: str, line_pts: list[tuple[int, int]]) -> tuple[str, tuple[int, int], int] | None:
        if len(line_pts) != 1:
            return None
        head = line_pts[0]
        dr, dc = DIRS[name]
        inward = (head[0] - dr, head[1] - dc)
        if inward not in pts:
            return None
        if name == "up":
            next_count = sum(r == min_r + 1 for r, _ in coords)
        elif name == "down":
            next_count = sum(r == max_r - 1 for r, _ in coords)
        elif name == "left":
            next_count = sum(c == min_c + 1 for _, c in coords)
        else:
            next_count = sum(c == max_c - 1 for _, c in coords)
        return name, head, int(next_count)

    candidates = []
    for item in (
        candidate("up", [(r, c) for r, c in coords if r == min_r]),
        candidate("down", [(r, c) for r, c in coords if r == max_r]),
        candidate("left", [(r, c) for r, c in coords if c == min_c]),
        candidate("right", [(r, c) for r, c in coords if c == max_c]),
    ):
        if item is not None:
            candidates.append(item)

    if not candidates:
        raise AssertionError("failed to detect animal orientation")

    candidates.sort(key=lambda item: (-item[2], DIR_ORDER.index(item[0])))
    direction, head, _ = candidates[0]
    return direction, (int(head[0]), int(head[1]))


def _first_food_on_ray(
    head: tuple[int, int], direction: str, food_set: set[tuple[int, int]], shape: tuple[int, int]
) -> tuple[int, int] | None:
    dr, dc = DIRS[direction]
    h, w = shape
    r, c = head
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        if (r, c) in food_set:
            return int(r), int(c)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pull each same-color outside food into the fence that its animal is facing.

    Input structure:
    - The background is a dominant grass color.
    - One foreground color forms several one-pixel-thick rectangular fence frames.
    - Each fence interior contains one small monochrome animal-like shape whose orientation is marked by a single-pixel protrusion on one side.
    - Outside the fences there may be isolated food pixels whose color matches one interior animal.

    Rule:
    - Find every rectangular fence and inspect the animal shape inside it.
    - Detect the direction of the unique protrusion (up, down, left, or right).
    - Along the ray that starts at the protrusion and continues outward in that direction, find the first isolated same-color food pixel outside the fence.
    - Remove that food from its original location and place it on the cell immediately in front of the protrusion inside the fence; if no such food exists, leave the fence unchanged.

    Key insight:
    - Color alone does not determine the target: the interior shape's asymmetric one-pixel head selects the only ray along which matching food can be absorbed.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    fence_color, fences = _find_fences(g, bg)
    foods = _find_foods(g, bg, fence_color, fences)

    out = g.copy()
    for r0, c0, r1, c1 in fences:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        colors = [int(v) for v in np.unique(inner) if int(v) not in {int(bg), int(fence_color)}]
        if len(colors) != 1:
            continue
        animal_color = int(colors[0])
        animal_cells = [
            (int(r0 + 1 + rr), int(c0 + 1 + cc))
            for rr, cc in np.argwhere(inner == animal_color)
        ]
        if len(animal_cells) <= 1:
            continue
        direction, head = _detect_direction(animal_cells)
        dr, dc = DIRS[direction]
        target = (head[0] + dr, head[1] + dc)
        if not (r0 + 1 <= target[0] <= r1 - 1 and c0 + 1 <= target[1] <= c1 - 1):
            continue
        food = _first_food_on_ray(head, direction, foods.get(animal_color, set()), g.shape)
        if food is None:
            continue
        out[food] = bg
        out[target] = animal_color
    return out


def _build_animal(direction: str, rng: random.Random) -> tuple[np.ndarray, tuple[int, int]]:
    body_h = rng.randint(2, 3)
    body_w = rng.randint(2, 3)
    cells = {(r, c) for r in range(body_h) for c in range(body_w)}

    if direction == "right":
        head = (rng.randrange(body_h), body_w)
    elif direction == "left":
        head = (rng.randrange(body_h), -1)
    elif direction == "down":
        head = (body_h, rng.randrange(body_w))
    else:
        head = (-1, rng.randrange(body_w))

    cells.add(head)
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    shifted = {(r - min_r, c - min_c) for r, c in cells}
    head_shifted = (head[0] - min_r, head[1] - min_c)
    h = max(r for r, _ in shifted) + 1
    w = max(c for _, c in shifted) + 1
    patch = np.zeros((h, w), dtype=bool)
    for r, c in shifted:
        patch[r, c] = True
    return patch, (int(head_shifted[0]), int(head_shifted[1]))


def _choose_layout(n_fences: int, rng: random.Random) -> tuple[int, int]:
    return rng.choice(LAYOUTS[n_fences])


def _slot_sizes(rows: int, cols: int, rng: random.Random) -> tuple[int, int]:
    max_h = min(13, 30 // rows)
    max_w = min(13, 30 // cols)
    return rng.randint(10, max_h), rng.randint(10, max_w)


def _ray_outside_fence(
    head: tuple[int, int],
    direction: str,
    fence_bbox: tuple[int, int, int, int],
    slot_bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]]:
    dr, dc = DIRS[direction]
    fr0, fc0, fr1, fc1 = fence_bbox
    sr0, sc0, sr1, sc1 = slot_bbox
    r, c = head
    cells = []
    while True:
        r += dr
        c += dc
        if not (sr0 <= r <= sr1 and sc0 <= c <= sc1):
            break
        if not (fr0 <= r <= fr1 and fc0 <= c <= fc1):
            cells.append((int(r), int(c)))
    return cells


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    n_fences = rng.randint(2, 4)
    rows, cols = _choose_layout(n_fences, rng)
    slot_h, slot_w = _slot_sizes(rows, cols, rng)
    grid_h = rows * slot_h
    grid_w = cols * slot_w
    grid = np.full((grid_h, grid_w), BG, dtype=int)

    slot_indices = list(range(rows * cols))
    rng.shuffle(slot_indices)
    chosen_slots = sorted(slot_indices[:n_fences])
    colors = rng.sample(list(ANIMAL_COLORS), n_fences)
    directions = [rng.choice(DIR_ORDER) for _ in range(n_fences)]
    has_target_food = [rng.random() < 0.5 for _ in range(n_fences)]
    if all(has_target_food) or not any(has_target_food):
        has_target_food[rng.randrange(n_fences)] = not has_target_food[rng.randrange(n_fences)]

    for idx, slot_id in enumerate(chosen_slots):
        slot_r = slot_id // cols
        slot_c = slot_id % cols
        sr0 = slot_r * slot_h
        sc0 = slot_c * slot_w
        sr1 = sr0 + slot_h - 1
        sc1 = sc0 + slot_w - 1
        slot_bbox = (sr0, sc0, sr1, sc1)

        animal_mask, head_local = _build_animal(directions[idx], rng)
        animal_h, animal_w = animal_mask.shape
        outer_h = animal_h + 4
        outer_w = animal_w + 4
        if outer_h > slot_h - 2 or outer_w > slot_w - 2:
            return None

        fr0 = rng.randint(sr0 + 1, sr1 - outer_h)
        fc0 = rng.randint(sc0 + 1, sc1 - outer_w)
        fr1 = fr0 + outer_h - 1
        fc1 = fc0 + outer_w - 1

        grid[fr0, fc0 : fc1 + 1] = FENCE
        grid[fr1, fc0 : fc1 + 1] = FENCE
        grid[fr0 : fr1 + 1, fc0] = FENCE
        grid[fr0 : fr1 + 1, fc1] = FENCE

        ar0 = fr0 + 2
        ac0 = fc0 + 2
        color = colors[idx]
        for rr, cc in np.argwhere(animal_mask):
            grid[ar0 + rr, ac0 + cc] = color
        head = (ar0 + head_local[0], ac0 + head_local[1])
        fence_bbox = (fr0, fc0, fr1, fc1)

        if has_target_food[idx]:
            target_cells = _ray_outside_fence(head, directions[idx], fence_bbox, slot_bbox)
            if not target_cells:
                return None
            grid[rng.choice(target_cells)] = color

        for side in DIR_ORDER:
            if side == directions[idx]:
                continue
            if rng.random() >= 0.35:
                continue
            decoy_cells = _ray_outside_fence(head, side, fence_bbox, slot_bbox)
            if not decoy_cells:
                continue
            chosen = rng.choice(decoy_cells)
            if grid[chosen] == BG:
                grid[chosen] = color

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 4")
