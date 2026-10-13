from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
ROTATION_K = {"top": 0, "right": -1, "bottom": 2, "left": 1}
FRAME_COLORS = (1, 2, 3, 4, 5, 6, 7, 8, 9)
SIZE_CHOICES = (5, 7, 9)
MAX_GENERATION_ATTEMPTS = 300


def _connected_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))
    return components


def _border_mask(size: int) -> np.ndarray:
    border = np.zeros((size, size), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return border


def _gap_coord(size: int, side: str) -> tuple[int, int]:
    mid = size // 2
    if side == "top":
        return 0, mid
    if side == "right":
        return mid, size - 1
    if side == "bottom":
        return size - 1, mid
    if side == "left":
        return mid, 0
    raise ValueError(f"unknown side: {side}")


def _match_gapped_square(local_mask: np.ndarray) -> str | None:
    if local_mask.ndim != 2 or local_mask.shape[0] != local_mask.shape[1]:
        return None
    size = int(local_mask.shape[0])
    if size < 5 or size % 2 == 0:
        return None
    border = _border_mask(size)
    if int(local_mask.sum()) != int(border.sum()) - 1:
        return None
    matches = []
    for side in SIDES:
        expected = border.copy()
        expected[_gap_coord(size, side)] = False
        if np.array_equal(local_mask, expected):
            matches.append(side)
    return matches[0] if len(matches) == 1 else None


def _find_rings(grid: np.ndarray) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    frames: list[dict[str, object]] = []
    for color in sorted(set(map(int, g.flatten())) - {BG}):
        for coords in _connected_components(g == color):
            r0, c0 = map(int, coords.min(axis=0))
            r1, c1 = map(int, coords.max(axis=0))
            if r1 - r0 != c1 - c0:
                continue
            size = int(r1 - r0 + 1)
            if size < 5 or size % 2 == 0:
                continue
            local = np.zeros((size, size), dtype=bool)
            local[coords[:, 0] - r0, coords[:, 1] - c0] = True
            side = _match_gapped_square(local)
            if side is None:
                continue
            interior = g[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0 or np.any(interior == color):
                continue
            interior_colors = sorted(set(map(int, interior.flatten())) - {BG})
            if len(interior_colors) != 1:
                continue
            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "size": size,
                    "color": color,
                    "side": side,
                    "pattern_color": interior_colors[0],
                }
            )
    frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["color"]))
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate each pattern inside a gapped square ring according to the ring's gap direction.

    Input structure:
    - The background is black.
    - Several non-overlapping monochrome square rings appear on the grid; each ring is a 1-cell-thick hollow border of odd side length at least 5.
    - Each ring has exactly one missing border cell at the center of one side, and its interior contains a small monochrome pattern of another color.

    Rule:
    - Detect each square ring and read the missing side-center cell as a rotation command.
    - A top gap keeps the interior unchanged, a right gap rotates the interior 90 degrees clockwise, a bottom gap rotates it 180 degrees, and a left gap rotates it 90 degrees counterclockwise.
    - Apply the rotation inside the same interior square while leaving the surrounding ring, gap, and everything outside the ring unchanged.

    Key insight:
    - The gap is not part of the pattern: it encodes how to rotate the entire interior square patch around its center.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for frame in _find_rings(g):
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = np.rot90(interior, int(ROTATION_K[frame["side"]]))
    return out


def _pattern_signature(mask: np.ndarray) -> bytes:
    variants = [np.rot90(mask, k).tobytes() for k in range(4)]
    return min(variants)


def _random_pattern(size: int, used: set[bytes]) -> np.ndarray | None:
    max_cells = min(size * size - 1, size + 3)
    if max_cells < 3:
        return None
    for _ in range(200):
        target = random.randint(3, max_cells)
        mask = np.zeros((size, size), dtype=bool)
        start = (random.randrange(size), random.randrange(size))
        mask[start] = True
        cells = [start]
        while int(mask.sum()) < target:
            rr, cc = random.choice(cells)
            neighbors = []
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier = []
                for rr, cc in cells:
                    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                        nr, nc = rr + dr, cc + dc
                        if 0 <= nr < size and 0 <= nc < size and not mask[nr, nc]:
                            frontier.append((nr, nc))
                if not frontier:
                    break
                nr, nc = random.choice(frontier)
            else:
                nr, nc = random.choice(neighbors)
            mask[nr, nc] = True
            cells.append((nr, nc))
        if int(mask.sum()) < 3:
            continue
        if any(np.array_equal(mask, np.rot90(mask, k)) for k in (1, 2, 3)):
            continue
        if _match_gapped_square(mask) is not None:
            continue
        sig = _pattern_signature(mask)
        if sig in used:
            continue
        used.add(sig)
        return mask
    return None


def _boxes_conflict(a: tuple[int, int, int], b: tuple[int, int, int], gap: int) -> bool:
    ar, ac, asz = a
    br, bc, bsz = b
    return not (
        ar + asz - 1 + gap < br
        or br + bsz - 1 + gap < ar
        or ac + asz - 1 + gap < bc
        or bc + bsz - 1 + gap < ac
    )


def _place_boxes(sizes: list[int], canvas: int = 30, gap: int = 1) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    placed: list[tuple[int, int, int]] = []
    order = sorted(range(len(sizes)), key=lambda idx: -sizes[idx])
    for idx in order:
        size = sizes[idx]
        placed_here = False
        for _ in range(400):
            r = random.randint(0, canvas - size)
            c = random.randint(0, canvas - size)
            box = (r, c, size)
            if any(_boxes_conflict(box, other, gap) for other in placed):
                continue
            positions[idx] = (r, c)
            placed.append(box)
            placed_here = True
            break
        if not placed_here:
            return None
    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _draw_ring(
    grid: np.ndarray,
    top: int,
    left: int,
    size: int,
    frame_color: int,
    side: str,
    pattern_color: int,
    pattern_mask: np.ndarray,
) -> None:
    bottom = top + size - 1
    right = left + size - 1
    grid[top, left : right + 1] = frame_color
    grid[bottom, left : right + 1] = frame_color
    grid[top : bottom + 1, left] = frame_color
    grid[top : bottom + 1, right] = frame_color
    gap_r, gap_c = _gap_coord(size, side)
    grid[top + gap_r, left + gap_c] = BG
    interior = grid[top + 1 : bottom, left + 1 : right]
    interior[pattern_mask] = pattern_color


def _crop_with_padding(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG)
    r0, c0 = map(int, coords.min(axis=0))
    r1, c1 = map(int, coords.max(axis=0))
    cropped = grid[r0 : r1 + 1, c0 : c1 + 1]
    h, w = map(int, cropped.shape)
    target_h = random.randint(h, min(30, h + 6))
    target_w = random.randint(w, min(30, w + 6))
    pad_h = target_h - h
    pad_w = target_w - w
    top = random.randint(0, pad_h)
    bottom = pad_h - top
    left = random.randint(0, pad_w)
    right = pad_w - left
    return np.pad(cropped, ((top, bottom), (left, right)), constant_values=BG)


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice((4, 5))
    available_colors = list(FRAME_COLORS)
    pattern_color = random.choice(available_colors)
    available_colors.remove(pattern_color)
    frame_colors = random.sample(available_colors, frame_count)

    sizes = [random.choice(SIZE_CHOICES) for _ in range(frame_count)]
    positions = _place_boxes(sizes, canvas=30, gap=1)
    if positions is None:
        return None

    sides = list(SIDES)
    random.shuffle(sides)
    if frame_count == 5:
        sides.append(random.choice(SIDES))
    random.shuffle(sides)

    used_patterns: set[bytes] = set()
    patterns = []
    for size in sizes:
        pattern = _random_pattern(size - 2, used_patterns)
        if pattern is None:
            return None
        patterns.append(pattern)

    grid = np.zeros((30, 30), dtype=int)
    for (top, left), size, frame_color, side, pattern in zip(positions, sizes, frame_colors, sides, patterns):
        _draw_ring(grid, top, left, size, frame_color, side, pattern_color, pattern)

    cropped = _crop_with_padding(grid)
    if cropped.ndim != 2 or max(cropped.shape) > 30:
        return None

    found = _find_rings(cropped)
    if len(found) != frame_count:
        return None
    if set(frame["side"] for frame in found) != set(SIDES):
        return None
    if set(frame["pattern_color"] for frame in found) != {pattern_color}:
        return None

    transformed = main(cropped)
    if np.array_equal(transformed, cropped):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: failed after repeated attempts")
