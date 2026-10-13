import random
from collections import Counter, deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
GREY = 5
CORE_COLORS = (4, 6, 7, 8, 9)
FRAME_COUNT_CHOICES = (1, 2, 3)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    rows = np.where(mask.any(axis=1))[0]
    if rows.size == 0:
        return None
    cols = np.where(mask.any(axis=0))[0]
    return int(rows[0]), int(cols[0]), int(rows[-1]), int(cols[-1])


def _crop_non_bg(arr: np.ndarray, bg: int = BLACK) -> np.ndarray:
    box = _bbox(arr != bg)
    if box is None:
        return np.zeros((0, 0), dtype=int)
    r0, c0, r1, c1 = box
    return arr[r0 : r1 + 1, c0 : c1 + 1].copy()


def _corner_map(r0: int, c0: int, r1: int, c1: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (r0, c0),
        "tr": (r0, c1),
        "br": (r1, c1),
        "bl": (r1, c0),
    }


def _read_frame_instruction(
    arr: np.ndarray, r0: int, c0: int, r1: int, c1: int
) -> tuple[int, str] | None:
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    corners = _corner_map(r0, c0, r1, c1)
    by_pos = {pos: name for name, pos in corners.items()}
    marker: tuple[int, str] | None = None

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            if r not in (r0, r1) and c not in (c0, c1):
                continue
            value = int(arr[r, c])
            corner_name = by_pos.get((r, c))
            if corner_name is not None:
                if value == GREY:
                    continue
                if value == BLACK:
                    return None
                if marker is not None:
                    return None
                marker = (value, corner_name)
            else:
                if value != GREY:
                    return None

    return marker


def _extract_core_and_instructions(grid: np.ndarray) -> tuple[np.ndarray, list[tuple[int, str]]]:
    view = np.asarray(grid, dtype=int)
    instructions: list[tuple[int, str]] = []

    while True:
        box = _bbox(view != BLACK)
        if box is None:
            raise ValueError("grid does not contain a core pattern")
        r0, c0, r1, c1 = box
        instruction = _read_frame_instruction(view, r0, c0, r1, c1)
        if instruction is None:
            return _crop_non_bg(view, BLACK), instructions
        instructions.append(instruction)
        view = view[r0 + 1 : r1, c0 + 1 : c1]


def _least_frequent_color(arr: np.ndarray) -> int:
    counts = Counter(int(v) for v in arr.ravel() if v != BLACK)
    if not counts:
        raise ValueError("core pattern is empty")
    min_count = min(counts.values())
    least = [color for color, count in counts.items() if count == min_count]
    if len(least) != 1:
        raise ValueError("least-frequent color must be unique for green instruction")
    return int(least[0])


def _apply_instruction(arr: np.ndarray, instruction: tuple[int, str]) -> np.ndarray:
    marker_color, corner = instruction

    if marker_color == RED:
        if corner == "tl":
            return np.rot90(arr, -1).copy()
        if corner == "tr":
            return np.rot90(arr, 2).copy()
        if corner == "br":
            return np.rot90(arr, 1).copy()
        if corner == "bl":
            return arr.copy()
    elif marker_color == BLUE:
        if corner in ("tl", "tr"):
            return np.fliplr(arr).copy()
        if corner in ("bl", "br"):
            return np.flipud(arr).copy()
    elif marker_color == GREEN:
        target = _least_frequent_color(arr)
        out = arr.copy()
        out[out == target] = GREEN
        return out

    raise ValueError(f"unsupported instruction: color={marker_color}, corner={corner}")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode nested corner markers on concentric grey frames and apply the resulting
    instruction sequence to the enclosed core pattern.
    """
    core, instructions = _extract_core_and_instructions(np.asarray(grid, dtype=int))
    out = core.copy()
    for instruction in instructions:
        out = _apply_instruction(out, instruction)
    return _crop_non_bg(out, BLACK)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    if r > 0:
        out.append((r - 1, c))
    if r + 1 < h:
        out.append((r + 1, c))
    if c > 0:
        out.append((r, c - 1))
    if c + 1 < w:
        out.append((r, c + 1))
    return out


def _is_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False
    h, w = mask.shape
    start = tuple(map(int, points[0]))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] or (nr, nc) in seen:
                continue
            seen.add((nr, nc))
            queue.append((nr, nc))
    return len(seen) == int(mask.sum())


def _all_transforms(arr: np.ndarray) -> list[np.ndarray]:
    transforms = []
    for k in range(4):
        rotated = np.rot90(arr, k)
        transforms.append(rotated)
        transforms.append(np.fliplr(rotated))
    return transforms


def _is_asymmetric(arr: np.ndarray) -> bool:
    transforms = _all_transforms(arr)
    base = transforms[0]
    return all(not np.array_equal(base, other) for other in transforms[1:])


def _sample_distinct_counts(rng: random.Random, total: int, parts: int) -> list[int]:
    for _ in range(200):
        cuts = sorted(rng.sample(range(1, total), parts - 1))
        counts = []
        prev = 0
        for cut in cuts + [total]:
            counts.append(cut - prev)
            prev = cut
        counts.sort()
        if len(set(counts)) == parts:
            return counts
    raise RuntimeError("failed to sample distinct color counts")


def _random_connected_mask(rng: random.Random, h: int, w: int, target_cells: int) -> np.ndarray | None:
    for _ in range(200):
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = set(_neighbors4(start[0], start[1], h, w))
        while frontier and len(cells) < target_cells:
            cell = rng.choice(tuple(frontier))
            frontier.remove(cell)
            cells.add(cell)
            for nbr in _neighbors4(cell[0], cell[1], h, w):
                if nbr not in cells:
                    frontier.add(nbr)
        if len(cells) != target_cells:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if _bbox(mask) != (0, 0, h - 1, w - 1):
            continue
        if _is_connected(mask):
            return mask
    return None


def _generate_core_pattern(rng: random.Random) -> np.ndarray:
    for _ in range(300):
        h = rng.randint(3, 5)
        w = rng.randint(3, 5)
        min_cells = max(h + w - 1, (h * w + 1) // 2)
        max_cells = h * w - 1
        target_cells = rng.randint(min_cells, max_cells)
        mask = _random_connected_mask(rng, h, w, target_cells)
        if mask is None:
            continue

        color_count = 3 if target_cells >= 6 and rng.random() < 0.5 else 2
        counts = _sample_distinct_counts(rng, target_cells, color_count)
        colors = rng.sample(CORE_COLORS, color_count)
        coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
        rng.shuffle(coords)

        arr = np.zeros((h, w), dtype=int)
        idx = 0
        for color, count in zip(colors, counts):
            for _ in range(count):
                r, c = coords[idx]
                arr[r, c] = color
                idx += 1

        if not _is_asymmetric(arr):
            continue
        return arr

    raise RuntimeError("failed to generate asymmetric core pattern")


def _sample_instruction_categories(rng: random.Random, frame_count: int) -> list[str]:
    cycle = ["rotate", "flip", "replace"]
    rng.shuffle(cycle)
    start = rng.randrange(len(cycle))
    return [cycle[(start + i) % len(cycle)] for i in range(frame_count)]


def _sample_instruction(rng: random.Random, category: str) -> tuple[int, str]:
    corner = rng.choice(["tl", "tr", "br", "bl"])
    if category == "rotate":
        return RED, corner
    if category == "flip":
        return BLUE, corner
    if category == "replace":
        return GREEN, corner
    raise ValueError(f"unknown category: {category}")


def _balanced_pair(rng: random.Random, total: int) -> tuple[int, int]:
    a = total // 2
    b = total - a
    if rng.random() < 0.5:
        return a, b
    return b, a


def _draw_frame(grid: np.ndarray, top: int, left: int, bottom: int, right: int, marker: tuple[int, str]) -> None:
    grid[top, left : right + 1] = GREY
    grid[bottom, left : right + 1] = GREY
    grid[top : bottom + 1, left] = GREY
    grid[top : bottom + 1, right] = GREY

    marker_color, corner = marker
    mr, mc = _corner_map(top, left, bottom, right)[corner]
    grid[mr, mc] = marker_color


def _paste_pattern(grid: np.ndarray, pattern: np.ndarray, top: int, left: int) -> None:
    h, w = pattern.shape
    window = grid[top : top + h, left : left + w]
    mask = pattern != BLACK
    window[mask] = pattern[mask]


def _apply_sequence(pattern: np.ndarray, instructions: list[tuple[int, str]]) -> np.ndarray:
    out = pattern.copy()
    for instruction in instructions:
        out = _apply_instruction(out, instruction)
    return _crop_non_bg(out, BLACK)


def _build_example(rng: random.Random) -> np.ndarray | None:
    core = _generate_core_pattern(rng)
    frame_count = rng.choice(FRAME_COUNT_CHOICES)
    instructions = [_sample_instruction(rng, cat) for cat in _sample_instruction_categories(rng, frame_count)]

    inner_gap = rng.randint(1, 2)
    core_h, core_w = core.shape
    inner_h = core_h + 2 * inner_gap + 2
    inner_w = core_w + 2 * inner_gap + 2
    outer_h = inner_h + 4 * (frame_count - 1)
    outer_w = inner_w + 4 * (frame_count - 1)

    top_margin, bottom_margin = _balanced_pair(rng, rng.randint(2, 6))
    left_margin, right_margin = _balanced_pair(rng, rng.randint(2, 6))
    grid = np.zeros((outer_h + top_margin + bottom_margin, outer_w + left_margin + right_margin), dtype=int)

    outer_top = top_margin
    outer_left = left_margin
    inner_top = outer_top + 2 * (frame_count - 1)
    inner_left = outer_left + 2 * (frame_count - 1)
    core_top = inner_top + 1 + inner_gap
    core_left = inner_left + 1 + inner_gap

    _paste_pattern(grid, core, core_top, core_left)

    for depth, instruction in enumerate(instructions):
        top = outer_top + 2 * depth
        left = outer_left + 2 * depth
        bottom = outer_top + outer_h - 1 - 2 * depth
        right = outer_left + outer_w - 1 - 2 * depth
        _draw_frame(grid, top, left, bottom, right, instruction)

    expected = _apply_sequence(core, instructions)
    actual = main(grid)
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        try:
            grid = _build_example(rng)
        except Exception:
            continue
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input after repeated attempts")
