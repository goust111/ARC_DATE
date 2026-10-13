import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_foreground_components(grid: np.ndarray) -> list[np.ndarray]:
    h, w = grid.shape
    fg = grid != BLACK
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not fg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and fg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _is_observation_patch(patch: np.ndarray) -> bool:
    if patch.size == 0 or np.any(patch == BLACK):
        return False
    if not np.isin(patch, (BLUE, RED, GREY)).all():
        return False
    blue_coords = np.argwhere(patch == BLUE)
    if len(blue_coords) != 1:
        return False
    br, bc = map(int, blue_coords[0])
    corners = {
        (0, 0),
        (0, patch.shape[1] - 1),
        (patch.shape[0] - 1, 0),
        (patch.shape[0] - 1, patch.shape[1] - 1),
    }
    return (br, bc) in corners


def _rotate_anchor_to_top_left(patch: np.ndarray) -> np.ndarray:
    for k in range(4):
        rotated = np.rot90(patch, k=k)
        if rotated[0, 0] == BLUE and np.count_nonzero(rotated == BLUE) == 1:
            return rotated.copy()
    raise ValueError("observation patch does not have a rotatable corner anchor")


def _canonical_observations(grid: np.ndarray) -> list[np.ndarray]:
    observations = []
    for coords in _find_foreground_components(grid):
        box = _bbox_from_coords(coords)
        patch = _crop_bbox(grid, box)
        if not _is_observation_patch(patch):
            continue
        observations.append(_rotate_anchor_to_top_left(patch))
    return observations


def main(grid: np.ndarray) -> np.ndarray:
    """Align the three anchored observation rectangles and majority-vote their red cells.

    Input structure:
    - A black background contains several separated grey-filled rectangles that are otherwise solid.
    - Each such rectangle contains exactly one blue corner anchor and some red cells on the grey field.
    - The rectangles are rotated views of the same hidden pattern with independent missing or extra red cells.

    Rule:
    - Crop each foreground rectangle and rotate it until its blue anchor sits in the upper-left corner.
    - Compare the aligned rectangles cell by cell and mark a cell red iff a strict majority of the observations show red there.
    - Return one grey rectangle of that aligned size, keep the blue anchor in the upper-left, and leave every non-voted cell grey.

    Key insight:
    - The three rectangles are noisy observations of one source pattern, so the decisive step is anchor-based alignment before majority voting.
    """
    g = np.asarray(grid, dtype=int)
    observations = _canonical_observations(g)
    if not observations:
        return g.copy()

    target_shape = observations[0].shape
    if any(obs.shape != target_shape for obs in observations):
        raise ValueError("aligned observations must share one shape")

    red_votes = np.sum(np.stack([obs == RED for obs in observations], axis=0), axis=0)
    threshold = len(observations) // 2 + 1

    out = np.full(target_shape, GREY, dtype=int)
    out[red_votes >= threshold] = RED
    out[0, 0] = BLUE
    return out


def _draw_segment(mask: np.ndarray, kind: str) -> None:
    h, w = mask.shape
    if kind == "hline":
        row = random.randrange(h)
        c0 = random.randrange(w)
        c1 = random.randrange(c0, w)
        if c1 == c0 and w > 1:
            c1 = min(w - 1, c0 + 1)
        mask[row, c0 : c1 + 1] = True
        return

    if kind == "vline":
        col = random.randrange(w)
        r0 = random.randrange(h)
        r1 = random.randrange(r0, h)
        if r1 == r0 and h > 1:
            r1 = min(h - 1, r0 + 1)
        mask[r0 : r1 + 1, col] = True
        return

    if kind == "block":
        block_h = random.randint(1, min(3, h))
        block_w = random.randint(1, min(3, w))
        if h >= 2:
            block_h = max(block_h, 2)
        if w >= 2:
            block_w = max(block_w, 2)
        r0 = random.randint(0, h - block_h)
        c0 = random.randint(0, w - block_w)
        mask[r0 : r0 + block_h, c0 : c0 + block_w] = True
        return

    elbow_h = random.randint(2, h)
    elbow_w = random.randint(2, w)
    r0 = random.randint(0, h - elbow_h)
    c0 = random.randint(0, w - elbow_w)
    if random.random() < 0.5:
        mask[r0, c0 : c0 + elbow_w] = True
        mask[r0 : r0 + elbow_h, c0] = True
    else:
        mask[r0 + elbow_h - 1, c0 : c0 + elbow_w] = True
        mask[r0 : r0 + elbow_h, c0 + elbow_w - 1] = True


def _make_truth_mask(side: int) -> np.ndarray:
    inner = side - 2
    target_min = max(3, inner * inner // 5)
    target_max = inner * inner - max(3, inner * inner // 5)

    for _ in range(200):
        mask = np.zeros((inner, inner), dtype=bool)
        for _ in range(random.randint(2, 4)):
            _draw_segment(mask, random.choice(("hline", "vline", "block", "elbow")))

        red_coords = np.argwhere(mask)
        red_count = len(red_coords)
        if red_count < target_min or red_count > target_max:
            continue
        rows = {int(r) for r, _ in red_coords}
        cols = {int(c) for _, c in red_coords}
        if len(rows) < 2 or len(cols) < 2:
            continue
        return mask

    raise RuntimeError("failed to generate a non-degenerate hidden pattern")


def _make_truth_sprite(side: int) -> np.ndarray:
    sprite = np.full((side, side), GREY, dtype=int)
    sprite[0, 0] = BLUE
    truth_mask = _make_truth_mask(side)
    sprite[1:-1, 1:-1][truth_mask] = RED
    return sprite


def _noise_plan(true_red: list[tuple[int, int]], true_grey: list[tuple[int, int]]) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    max_misses = max(1, min(len(true_red) // 2, 4))
    max_extras = max(1, min(len(true_grey) // 3, 4))
    miss_count = random.randint(1, max_misses)
    extra_count = random.randint(1, max_extras)
    misses = random.sample(true_red, miss_count)
    extras = random.sample(true_grey, extra_count)
    return misses, extras


def _noisy_observations(truth: np.ndarray) -> list[np.ndarray]:
    inner_truth = truth[1:-1, 1:-1]
    red_cells = [tuple(map(int, rc)) for rc in np.argwhere(inner_truth == RED)]
    grey_cells = [tuple(map(int, rc)) for rc in np.argwhere(inner_truth == GREY)]
    misses, extras = _noise_plan(red_cells, grey_cells)

    observations = [truth.copy() for _ in range(3)]
    events = [("miss", cell) for cell in misses] + [("extra", cell) for cell in extras]
    random.shuffle(events)

    event_targets = []
    for obs_index in range(3):
        event_targets.append(obs_index)
    while len(event_targets) < len(events):
        event_targets.append(random.randrange(3))
    random.shuffle(event_targets)

    for (mode, (rr, cc)), obs_index in zip(events, event_targets):
        r = rr + 1
        c = cc + 1
        if mode == "miss":
            observations[obs_index][r, c] = GREY
        else:
            observations[obs_index][r, c] = RED

    return observations


def _separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _place_rotated_rectangles(rectangles: list[np.ndarray]) -> np.ndarray:
    side = rectangles[0].shape[0]
    for _ in range(200):
        rows = random.randint(max(18, side * 2 + 4), 30)
        cols = random.randint(max(18, side * 2 + 4), 30)
        grid = np.full((rows, cols), BLACK, dtype=int)
        boxes: list[tuple[int, int, int, int]] = []
        placements: list[tuple[int, int]] = []

        success = True
        for rect in rectangles:
            h, w = rect.shape
            placed = False
            for _ in range(200):
                r0 = random.randint(1, rows - h - 1)
                c0 = random.randint(1, cols - w - 1)
                box = (r0, c0, r0 + h - 1, c0 + w - 1)
                if all(_separated(box, prev) for prev in boxes):
                    boxes.append(box)
                    placements.append((r0, c0))
                    placed = True
                    break
            if not placed:
                success = False
                break

        if not success:
            continue

        for rect, (r0, c0) in zip(rectangles, placements):
            h, w = rect.shape
            grid[r0 : r0 + h, c0 : c0 + w] = rect
        return grid

    raise RuntimeError("failed to place the three observation rectangles")


def generate_input() -> np.ndarray:
    for _ in range(200):
        side = random.randint(5, 8)
        truth = _make_truth_sprite(side)
        observations = _noisy_observations(truth)
        rotations = random.sample([0, 1, 2, 3], 3)
        rotated = [np.rot90(obs, k=k) for obs, k in zip(observations, rotations)]
        grid = _place_rotated_rectangles(rotated)

        if not all(np.any(obs == RED) for obs in observations):
            continue
        if not np.array_equal(main(grid), truth):
            continue
        return grid

    raise RuntimeError("failed to generate a valid input grid")
