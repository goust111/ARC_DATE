import numpy as np


BG = 0
SEP = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GAP_TO_ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}
DIR_NAMES = ("up", "right", "down", "left")
MAX_GENERATION_ATTEMPTS = 400
_RNG = np.random.default_rng()


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_separator_col(grid: np.ndarray, bg: int) -> int:
    candidates = [
        c
        for c in range(grid.shape[1])
        if int(grid[0, c]) != bg and np.all(grid[:, c] == grid[0, c])
    ]
    assert len(candidates) == 1, "expected a unique full-height separator"
    return int(candidates[0])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)
    return components


def _ring_direction(coords: list[tuple[int, int]]) -> str | None:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3 or h % 2 == 0 or w % 2 == 0:
        return None

    rel = {(r - r0, c - c0) for r, c in coords}
    border = set()
    for rr in range(h):
        border.add((rr, 0))
        border.add((rr, w - 1))
    for cc in range(w):
        border.add((0, cc))
        border.add((h - 1, cc))

    missing = border - rel
    extra = rel - border
    if extra or len(missing) != 1:
        return None

    gap = next(iter(missing))
    centers = {
        (0, w // 2): "up",
        (h - 1, w // 2): "down",
        (h // 2, 0): "left",
        (h // 2, w - 1): "right",
    }
    return centers.get(gap)


def _extract_ring_instructions(left: np.ndarray, bg: int) -> dict[int, str]:
    instructions: dict[int, str] = {}
    for color in np.unique(left):
        color = int(color)
        if color == bg:
            continue
        directions = []
        for comp in _find_components(left == color):
            direction = _ring_direction(comp)
            if direction is not None:
                directions.append(direction)
        if len(directions) == 1:
            instructions[color] = directions[0]
    assert instructions, "expected at least one instruction ring"
    return instructions


def _color_patch_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(grid == int(color))
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each right-side sprite according to the gap direction of the same-colored ring, then return only the right half.

    Input structure:
    - The background is black, and a full-height one-pixel grey column splits the grid into a left instruction area and a right sprite area.
    - The left side contains several monochrome hollow rectangles of width 1; each ring has exactly one missing border pixel at the center of its top, right, bottom, or left side.
    - The right side contains the same number of monochrome pixel sprites, with colors matching the rings one-to-one; each sprite has a square tight bounding box.

    Rule:
    - Pair every right-side sprite with the left-side ring of the same color.
    - Decode the ring gap as a rotation command: top -> 0 degrees, right -> clockwise 90 degrees, bottom -> 180 degrees, left -> counterclockwise 90 degrees.
    - Rotate each sprite inside its own bounding box and place the rotated result back into the same bbox coordinates.
    - Output only the region strictly to the right of the grey separator; background stays black.

    Key insight:
    - Color establishes which ring controls which sprite, while the single missing border pixel on each ring encodes the sprite rotation direction.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    sep_col = _find_separator_col(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    instructions = _extract_ring_instructions(left, bg)

    out = np.full(right.shape, bg, dtype=int)
    for color, direction in instructions.items():
        bbox = _color_patch_bbox(right, color)
        if bbox is None:
            continue
        r0, c0, r1, c1 = bbox
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        coords = np.argwhere(right == int(color))
        rel = coords - np.array([r0, c0], dtype=int)
        patch[rel[:, 0], rel[:, 1]] = int(color)
        assert patch.shape[0] == patch.shape[1], "sprite bbox must be square"
        rotated = np.rot90(patch, GAP_TO_ROT_K[direction])
        out[r0 : r1 + 1, c0 : c1 + 1] = rotated
    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(lo, hi + 1))


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    return [int(x) for x in _RNG.multinomial(int(total), [1.0 / parts] * parts)]


def _pack_intervals(length: int, sizes: list[int], min_gap: int = 1, margin: int = 1) -> list[tuple[int, int]] | None:
    free = int(length) - sum(int(size) for size in sizes) - int(min_gap) * (len(sizes) - 1) - 2 * int(margin)
    if free < 0:
        return None
    extras = _random_composition(free, len(sizes) + 1)
    pos = int(margin) + extras[0]
    intervals: list[tuple[int, int]] = []
    for i, size in enumerate(sizes):
        size = int(size)
        intervals.append((pos, pos + size - 1))
        pos += size
        if i + 1 < len(sizes):
            pos += int(min_gap) + extras[i + 1]
    return intervals


def _random_path(start: tuple[int, int], end: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = start
    tr, tc = end
    steps = []
    steps.extend([(1 if tr > r else -1, 0)] * abs(tr - r))
    steps.extend([(0, 1 if tc > c else -1)] * abs(tc - c))
    if steps:
        order = _RNG.permutation(len(steps))
        steps = [steps[int(i)] for i in order]
    coords = [(r, c)]
    for dr, dc in steps:
        r += dr
        c += dc
        coords.append((r, c))
    return coords


def _rotation_distinct(mask: np.ndarray) -> bool:
    return all(not np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3))


def _make_sprite_mask(size: int) -> np.ndarray | None:
    for _ in range(200):
        anchors = [
            (0, _randint(0, size - 1)),
            (size - 1, _randint(0, size - 1)),
            (_randint(0, size - 1), 0),
            (_randint(0, size - 1), size - 1),
        ]
        order = [int(i) for i in _RNG.permutation(len(anchors))]
        cells = {anchors[order[0]]}
        cur = anchors[order[0]]
        for idx in order[1:]:
            nxt = anchors[idx]
            cells.update(_random_path(cur, nxt))
            cur = nxt

        min_target = max(len(cells), size + 2)
        max_target = min(size * size - 1, min_target + size + 3)
        if min_target > max_target:
            continue
        target = _randint(min_target, max_target)

        while len(cells) < target:
            frontier = set()
            for r, c in cells:
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < size and 0 <= nc < size and (nr, nc) not in cells:
                        frontier.add((nr, nc))
            if not frontier:
                break
            choices = list(frontier)
            cells.add(choices[_randint(0, len(choices) - 1)])

        mask = np.zeros((size, size), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)
        if not (rows[0] and rows[-1] and cols[0] and cols[-1]):
            continue
        if not _rotation_distinct(mask):
            continue
        return mask
    return None


def _draw_ring(grid: np.ndarray, top: int, left: int, h: int, w: int, color: int, gap_dir: str) -> None:
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color
    gap = {
        "up": (top, left + w // 2),
        "down": (top + h - 1, left + w // 2),
        "left": (top + h // 2, left),
        "right": (top + h // 2, left + w - 1),
    }[gap_dir]
    grid[gap] = BG


def _paint_sprite(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = int(color)
    grid[top : top + patch.shape[0], left : left + patch.shape[1]] = patch


def _gen_attempt() -> np.ndarray | None:
    n_objects = _randint(2, 4)
    ring_sizes = [(_randint(1, 3) * 2 + 1, _randint(1, 3) * 2 + 1) for _ in range(n_objects)]
    box_sizes = [_randint(4, 7) for _ in range(n_objects)]

    min_h = max(
        sum(h for h, _ in ring_sizes) + (n_objects - 1) + 2,
        sum(box_sizes) + (n_objects - 1) + 2,
        14,
    )
    if min_h > 30:
        return None
    height = _randint(min_h, 30)

    left_intervals = _pack_intervals(height, [h for h, _ in ring_sizes], min_gap=1, margin=1)
    right_intervals = _pack_intervals(height, box_sizes, min_gap=1, margin=1)
    if left_intervals is None or right_intervals is None:
        return None

    min_left_w = max(w for _, w in ring_sizes) + 2
    min_right_w = max(box_sizes) + 2
    spare = 30 - (min_left_w + 1 + min_right_w)
    if spare < 0:
        return None
    extra_left = _randint(0, min(5, spare))
    extra_right = _randint(0, min(5, spare - extra_left))
    left_w = min_left_w + extra_left
    right_w = min_right_w + extra_right
    width = left_w + 1 + right_w

    colors = [int(c) for c in _RNG.choice(PALETTE, size=n_objects, replace=False)]
    gap_dirs = {color: DIR_NAMES[_randint(0, 3)] for color in colors}
    ring_order = [colors[int(i)] for i in _RNG.permutation(n_objects)]
    sprite_order = [colors[int(i)] for i in _RNG.permutation(n_objects)]

    sprites: dict[int, dict[str, int | np.ndarray]] = {}
    for slot, color in enumerate(sprite_order):
        size = int(box_sizes[slot])
        mask = _make_sprite_mask(size)
        if mask is None:
            return None
        max_c = right_w - size - 1
        if max_c < 1:
            return None
        sprites[color] = {
            "top": int(right_intervals[slot][0]),
            "left": _randint(1, max_c),
            "size": size,
            "mask": mask,
        }

    inp = np.full((height, width), BG, dtype=int)
    inp[:, left_w] = SEP

    for slot, color in enumerate(ring_order):
        h, w = ring_sizes[slot]
        max_c = left_w - w - 1
        if max_c < 1:
            return None
        _draw_ring(
            inp,
            int(left_intervals[slot][0]),
            _randint(1, max_c),
            int(h),
            int(w),
            int(color),
            gap_dirs[color],
        )

    expected = np.full((height, right_w), BG, dtype=int)
    for color, spec in sprites.items():
        top = int(spec["top"])
        left = int(spec["left"])
        mask = np.asarray(spec["mask"], dtype=bool)
        _paint_sprite(inp, mask, top, left_w + 1 + left, int(color))
        rotated = np.rot90(mask.astype(int), GAP_TO_ROT_K[gap_dirs[color]]).astype(bool)
        _paint_sprite(expected, rotated, top, left, int(color))

    if _get_bg(inp) != BG:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
