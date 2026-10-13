import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
MAX_GENERATION_ATTEMPTS = 400
FENCE_RETRIES = 120
PLACEMENT_PADDING = 1


def _perimeter_coords(height: int, width: int) -> list[tuple[int, int]]:
    coords: list[tuple[int, int]] = []
    for c in range(width):
        coords.append((0, c))
    for r in range(1, height - 1):
        coords.append((r, width - 1))
    for c in range(width - 1, -1, -1):
        coords.append((height - 1, c))
    for r in range(height - 2, 0, -1):
        coords.append((r, 0))
    return coords


def _side_ranges(height: int, width: int) -> list[list[int]]:
    top = list(range(0, width))
    right = list(range(width, width + max(0, height - 2)))
    bottom_start = width + max(0, height - 2)
    bottom = list(range(bottom_start, bottom_start + width))
    left_start = bottom_start + width
    left = list(range(left_start, left_start + max(0, height - 2)))
    return [top, right, bottom, left]


def _proper_divisors(n: int) -> list[int]:
    return [d for d in range(1, n) if n % d == 0]


def _periodic_step(bits: np.ndarray) -> int | None:
    period_len = int(bits.size)
    for step in _proper_divisors(period_len):
        if all(int(bits[i]) == int(bits[(i + step) % period_len]) for i in range(period_len)):
            return step
    return None


def _color_bbox(grid: np.ndarray, color: int) -> tuple[int, int, int, int] | None:
    rows, cols = np.where(grid == color)
    if rows.size == 0:
        return None
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _read_fence_bits(
    grid: np.ndarray, color: int
) -> tuple[tuple[int, int, int, int], list[tuple[int, int]], np.ndarray] | None:
    box = _color_bbox(grid, color)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 3 or width < 3:
        return None

    coords = _perimeter_coords(height, width)
    coord_set = {(r0 + rr, c0 + cc) for rr, cc in coords}
    color_cells = {tuple(map(int, rc)) for rc in np.argwhere(grid == color)}
    if not color_cells:
        return None
    if not color_cells <= coord_set:
        return None

    bits = np.array(
        [1 if grid[r0 + rr, c0 + cc] == color else 0 for rr, cc in coords],
        dtype=int,
    )
    return box, coords, bits


def _enclosed_mask(height: int, width: int) -> np.ndarray:
    walls = np.zeros((height + 2, width + 2), dtype=bool)
    walls[1, 1 : width + 1] = True
    walls[height, 1 : width + 1] = True
    walls[1 : height + 1, 1] = True
    walls[1 : height + 1, width] = True

    seen = np.zeros_like(walls, dtype=bool)
    stack = [(0, 0)]
    while stack:
        r, c = stack.pop()
        if not (0 <= r < height + 2 and 0 <= c < width + 2):
            continue
        if seen[r, c] or walls[r, c]:
            continue
        seen[r, c] = True
        stack.append((r - 1, c))
        stack.append((r + 1, c))
        stack.append((r, c - 1))
        stack.append((r, c + 1))

    interior = ~seen[1 : height + 1, 1 : width + 1]
    interior &= ~walls[1 : height + 1, 1 : width + 1]
    return interior


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the interior of rectangular dashed fences whose perimeter gaps follow a consistent cycle.

    Input structure:
    - A black background contains several one-cell-thick rectangular fence outlines, each in a single color.
    - Every fence is drawn as a dashed perimeter: colored cells and missing cells alternate by some fixed local period.
    - Some rectangles contain one enlarged missing stretch that breaks that period and acts as a real door.

    Rule:
    - For each non-background color, recover the rectangle bbox and read the perimeter cells in clockwise cyclic order.
    - If that binary perimeter signal has a proper repeating period, treat every periodic gap as virtual fence and fill the enclosed rectangle interior with the fence color.
    - If no proper period fits because one gap is too large, leave that fence unchanged.

    Key insight:
    - Pixel connectivity is misleading because every dashed rectangle is physically open; closure must be inferred from cyclic perimeter periodicity before flood-filling the implied solid fence.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        parsed = _read_fence_bits(g, color)
        if parsed is None:
            continue
        (r0, c0, r1, c1), _coords, bits = parsed
        step = _periodic_step(bits)
        if step is None:
            continue

        height = r1 - r0 + 1
        width = c1 - c0 + 1
        interior = _enclosed_mask(height, width)
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[interior] = color

    return out


def _side_is_informative(bits: np.ndarray, height: int, width: int) -> bool:
    for indices in _side_ranges(height, width):
        side = bits[indices]
        if not np.any(side == 1):
            return False
        if not np.any(side == 0):
            return False
    return True


def _render_patch(height: int, width: int, color: int, bits: np.ndarray) -> np.ndarray:
    patch = np.zeros((height, width), dtype=int)
    for bit, (r, c) in zip(bits.tolist(), _perimeter_coords(height, width)):
        if bit:
            patch[r, c] = color
    return patch


def _closed_bits(height: int, width: int, wall_len: int, gap_len: int, phase: int) -> np.ndarray:
    period = wall_len + gap_len
    total = 2 * (height + width) - 4
    return np.array(
        [1 if ((idx + phase) % period) < wall_len else 0 for idx in range(total)],
        dtype=int,
    )


def _door_variants(bits: np.ndarray, height: int, width: int) -> list[np.ndarray]:
    total = int(bits.size)
    variants: list[np.ndarray] = []
    for side_indices in _side_ranges(height, width):
        side_len = len(side_indices)
        for pos in range(1, side_len - 1):
            idx = side_indices[pos]
            if bits[idx] != 1:
                continue
            prev_idx = side_indices[pos - 1]
            next_idx = side_indices[pos + 1]

            if bits[(idx - 1) % total] == 0:
                span = [side_indices[pos]]
                if pos + 1 < side_len - 1 and bits[next_idx] == 1:
                    span.append(next_idx)
                for use_len in (1, 2):
                    if use_len > len(span):
                        continue
                    candidate = bits.copy()
                    candidate[span[:use_len]] = 0
                    variants.append(candidate)

            if bits[(idx + 1) % total] == 0:
                span = [side_indices[pos]]
                if pos - 1 > 0 and bits[prev_idx] == 1:
                    span.append(prev_idx)
                for use_len in (1, 2):
                    if use_len > len(span):
                        continue
                    candidate = bits.copy()
                    candidate[span[:use_len]] = 0
                    variants.append(candidate)
    return variants


def _make_fence_patch(closed: bool, color: int) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(FENCE_RETRIES):
        height = random.randint(6, 11)
        width = random.randint(6, 11)
        perimeter = 2 * (height + width) - 4

        period_options: list[tuple[int, int]] = []
        for wall_len in range(1, 5):
            for gap_len in range(1, 3):
                period = wall_len + gap_len
                if perimeter % period == 0 and perimeter // period >= 4:
                    period_options.append((wall_len, gap_len))
        if not period_options:
            continue

        wall_len, gap_len = random.choice(period_options)
        period = wall_len + gap_len
        phase = random.randrange(period)
        bits = _closed_bits(height, width, wall_len, gap_len, phase)
        if _periodic_step(bits) != period:
            continue
        if not _side_is_informative(bits, height, width):
            continue

        if not closed:
            candidates = [
                candidate
                for candidate in _door_variants(bits, height, width)
                if _periodic_step(candidate) is None and _side_is_informative(candidate, height, width)
            ]
            if not candidates:
                continue
            bits = random.choice(candidates)

        inp = _render_patch(height, width, color, bits)
        out = inp.copy()
        if closed:
            out[1:-1, 1:-1] = color
        return inp, out
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, patch_shape: Iterable[int]) -> bool:
    height, width = map(int, patch_shape)
    grid_h, grid_w = occupied.shape
    if top < 0 or left < 0 or top + height > grid_h or left + width > grid_w:
        return False
    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(grid_h, top + height + PLACEMENT_PADDING)
    c1 = min(grid_w, left + width + PLACEMENT_PADDING)
    return not occupied[r0:r1, c0:c1].any()


def _place_patch(
    inp: np.ndarray,
    out: np.ndarray,
    occupied: np.ndarray,
    patch_in: np.ndarray,
    patch_out: np.ndarray,
) -> bool:
    grid_h, grid_w = inp.shape
    height, width = patch_in.shape
    choices = [
        (top, left)
        for top in range(grid_h - height + 1)
        for left in range(grid_w - width + 1)
        if _can_place(occupied, top, left, patch_in.shape)
    ]
    if not choices:
        return False

    top, left = random.choice(choices)
    inp[top : top + height, left : left + width] = np.maximum(
        inp[top : top + height, left : left + width], patch_in
    )
    out[top : top + height, left : left + width] = np.maximum(
        out[top : top + height, left : left + width], patch_out
    )

    r0 = max(0, top - PLACEMENT_PADDING)
    c0 = max(0, left - PLACEMENT_PADDING)
    r1 = min(grid_h, top + height + PLACEMENT_PADDING)
    c1 = min(grid_w, left + width + PLACEMENT_PADDING)
    occupied[r0:r1, c0:c1] = True
    return True


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    grid_h = random.randint(18, 30)
    grid_w = random.randint(18, 30)
    fence_count = random.randint(2, 4)
    statuses = [True, False]
    while len(statuses) < fence_count:
        statuses.append(bool(random.getrandbits(1)))
    random.shuffle(statuses)

    colors = random.sample(PALETTE, fence_count)
    fences = []
    for closed, color in zip(statuses, colors):
        patch = _make_fence_patch(closed, color)
        if patch is None:
            return None
        fences.append(patch)

    inp = np.zeros((grid_h, grid_w), dtype=int)
    out = np.zeros((grid_h, grid_w), dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    ordering = list(range(fence_count))
    random.shuffle(ordering)
    for idx in ordering:
        patch_in, patch_out = fences[idx]
        if not _place_patch(inp, out, occupied, patch_in, patch_out):
            return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        inp, expected = built
        if np.array_equal(main(inp), expected):
            return inp
    raise RuntimeError("failed to generate a valid dashed-fence puzzle")
