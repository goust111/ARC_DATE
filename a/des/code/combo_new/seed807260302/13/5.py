import random

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    parts: list[np.ndarray] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < height and 0 <= nc < width and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _color_components(grid: np.ndarray, bg: int) -> list[tuple[int, np.ndarray]]:
    parts: list[tuple[int, np.ndarray]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        for comp in _components(grid == color):
            parts.append((int(color), comp))
    return parts


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_ring(coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 4 or c1 - c0 < 4:
        return False
    coord_set = {tuple(map(int, rc)) for rc in coords}
    border_count = 2 * (r1 - r0 + 1) + 2 * (c1 - c0 + 1) - 4
    if len(coord_set) != border_count:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border != ((r, c) in coord_set):
                return False
    return True


def _kernel_directions(shell_bbox: tuple[int, int, int, int], kernel: np.ndarray) -> tuple[str, ...]:
    r0, c0, r1, c1 = shell_bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0

    up = max(center_r - float(r) for r, _ in kernel)
    down = max(float(r) - center_r for r, _ in kernel)
    left = max(center_c - float(c) for _, c in kernel)
    right = max(float(c) - center_c for _, c in kernel)

    dirs: list[str] = []
    if up > down:
        dirs.append("up")
    elif down > up:
        dirs.append("down")
    if left > right:
        dirs.append("left")
    elif right > left:
        dirs.append("right")
    return tuple(dirs)


def _detect_structures(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    structures: list[dict[str, object]] = []
    for shell_color, comp in _color_components(grid, bg):
        if not _is_rectangular_ring(comp):
            continue

        shell_bbox = _bbox(comp)
        r0, c0, r1, c1 = shell_bbox
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_parts = _components(inner != bg)
        assert len(inner_parts) == 1, "each shell must contain exactly one kernel"

        kernel_local = inner_parts[0]
        kernel = kernel_local + np.array([r0 + 1, c0 + 1], dtype=int)
        kernel_color = int(grid[tuple(kernel[0])])
        assert kernel_color != shell_color, "kernel color must differ from shell color"
        assert int(kernel.shape[0]) in range(2, 7), "kernel area must be 2..6"
        for rr, cc in kernel:
            assert r0 + 1 < int(rr) < r1 - 1, "kernel must stay at least one cell from the shell"
            assert c0 + 1 < int(cc) < c1 - 1, "kernel must stay at least one cell from the shell"
            assert int(grid[rr, cc]) == kernel_color, "kernel must be monochrome"

        structures.append(
            {
                "bbox": shell_bbox,
                "shell_color": shell_color,
                "kernel_color": kernel_color,
                "kernel": kernel,
            }
        )

    structures.sort(key=lambda item: tuple(item["bbox"]))  # type: ignore[arg-type]
    assert structures, "expected at least one nested shell structure"
    return structures


def main(grid: np.ndarray) -> np.ndarray:
    """
    Open selected sides of each hollow rectangle according to the centered kernel's asymmetric reach.

    Input structure:
    - The background is the dominant color.
    - Several non-touching one-pixel-thick rectangular rings appear in various colors.
    - Each ring encloses exactly one smaller 4-connected monochrome kernel object,
      separated from the ring by at least one background cell on every side.

    Rule:
    - For each ring, locate the enclosed kernel and use the ring's midpoint as the
      shared center of the nested structure.
    - Compare how far the kernel reaches upward vs downward from that center, and
      leftward vs rightward from that center.
    - On any axis, erase the full shell edge on the side with the larger reach;
      ties on an axis leave that pair of shell edges unchanged.
    - Keep the kernel itself and any non-erased shell edges unchanged.

    Key insight:
    - The inner object is not content to preserve; it is a direction code whose
      off-center protrusions decide which faces of the outer rectangular shell open up.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    out = g.copy()
    for structure in _detect_structures(g, bg):
        r0, c0, r1, c1 = structure["bbox"]  # type: ignore[misc]
        dirs = _kernel_directions((r0, c0, r1, c1), structure["kernel"])  # type: ignore[arg-type]
        if "up" in dirs:
            out[r0, c0 : c1 + 1] = bg
        if "down" in dirs:
            out[r1, c0 : c1 + 1] = bg
        if "left" in dirs:
            out[r0 : r1 + 1, c0] = bg
        if "right" in dirs:
            out[r0 : r1 + 1, c1] = bg
    return out


# --------------- generator ---------------

GEN_ATTEMPTS = 400
GRID_MIN_BY_COUNT = {2: 18, 3: 22, 4: 25}
GRID_MAX = 30
STRUCTURE_COUNTS = (2, 3, 4)
SIZE_CHOICES = (7, 9, 11)

DIR_PATTERNS = [
    ("up",),
    ("down",),
    ("left",),
    ("right",),
    ("up", "left"),
    ("up", "right"),
    ("down", "left"),
    ("down", "right"),
]

KERNEL_FAMILIES: dict[tuple[str, ...], tuple[tuple[tuple[int, int], ...], ...]] = {
    ("up",): (
        ((0, 0), (-1, 0)),
        ((0, 0), (-1, 0), (-2, 0)),
        ((0, 0), (-1, 0), (-1, -1), (-1, 1)),
        ((0, 0), (-1, 0), (-2, 0), (-1, -1), (-1, 1)),
    ),
    ("down",): (
        ((0, 0), (1, 0)),
        ((0, 0), (1, 0), (2, 0)),
        ((0, 0), (1, 0), (1, -1), (1, 1)),
        ((0, 0), (1, 0), (2, 0), (1, -1), (1, 1)),
    ),
    ("left",): (
        ((0, 0), (0, -1)),
        ((0, 0), (0, -1), (0, -2)),
        ((0, 0), (0, -1), (-1, -1), (1, -1)),
        ((0, 0), (0, -1), (0, -2), (-1, -1), (1, -1)),
    ),
    ("right",): (
        ((0, 0), (0, 1)),
        ((0, 0), (0, 1), (0, 2)),
        ((0, 0), (0, 1), (-1, 1), (1, 1)),
        ((0, 0), (0, 1), (0, 2), (-1, 1), (1, 1)),
    ),
    ("up", "left"): (
        ((0, 0), (-1, 0), (0, -1)),
        ((0, 0), (-1, 0), (0, -1), (-1, -1)),
        ((0, 0), (-1, 0), (0, -1), (-1, -1), (-2, 0)),
        ((0, 0), (-1, 0), (0, -1), (-1, -1), (0, -2)),
    ),
    ("up", "right"): (
        ((0, 0), (-1, 0), (0, 1)),
        ((0, 0), (-1, 0), (0, 1), (-1, 1)),
        ((0, 0), (-1, 0), (0, 1), (-1, 1), (-2, 0)),
        ((0, 0), (-1, 0), (0, 1), (-1, 1), (0, 2)),
    ),
    ("down", "left"): (
        ((0, 0), (1, 0), (0, -1)),
        ((0, 0), (1, 0), (0, -1), (1, -1)),
        ((0, 0), (1, 0), (0, -1), (1, -1), (2, 0)),
        ((0, 0), (1, 0), (0, -1), (1, -1), (0, -2)),
    ),
    ("down", "right"): (
        ((0, 0), (1, 0), (0, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1)),
        ((0, 0), (1, 0), (0, 1), (1, 1), (2, 0)),
        ((0, 0), (1, 0), (0, 1), (1, 1), (0, 2)),
    ),
}


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _dilate_box(box: tuple[int, int, int, int], margin: int, h: int, w: int) -> tuple[int, int, int, int]:
    return (
        max(0, box[0] - margin),
        max(0, box[1] - margin),
        min(h - 1, box[2] + margin),
        min(w - 1, box[3] + margin),
    )


def _offset_extents(offsets: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    up = max(max(-r, 0) for r, _ in offsets)
    down = max(max(r, 0) for r, _ in offsets)
    left = max(max(-c, 0) for _, c in offsets)
    right = max(max(c, 0) for _, c in offsets)
    return up, down, left, right


def _pick_kernel(rng: random.Random, direction: tuple[str, ...], count: int) -> tuple[tuple[tuple[int, int], ...], int, int]:
    candidates = list(KERNEL_FAMILIES[direction])
    rng.shuffle(candidates)
    for offsets in candidates:
        up, down, left, right = _offset_extents(offsets)
        good_heights = [size for size in SIZE_CHOICES if size // 2 - 2 >= max(up, down)]
        good_widths = [size for size in SIZE_CHOICES if size // 2 - 2 >= max(left, right)]
        if count >= 4:
            good_heights = [size for size in good_heights if size <= 9]
            good_widths = [size for size in good_widths if size <= 9]
        if good_heights and good_widths:
            return offsets, rng.choice(good_heights), rng.choice(good_widths)
    raise AssertionError("no shell size fits the chosen kernel")


def _draw_structure(
    grid: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    shell_color: int,
    kernel_color: int,
    offsets: tuple[tuple[int, int], ...],
) -> None:
    bottom = top + height - 1
    right = left + width - 1
    grid[top, left : right + 1] = shell_color
    grid[bottom, left : right + 1] = shell_color
    grid[top : bottom + 1, left] = shell_color
    grid[top : bottom + 1, right] = shell_color

    center_r = top + height // 2
    center_c = left + width // 2
    for dr, dc in offsets:
        grid[center_r + dr, center_c + dc] = kernel_color


def _sample_grid_shape(rng: random.Random, count: int) -> tuple[int, int]:
    minimum = GRID_MIN_BY_COUNT[count]
    return rng.randint(minimum, GRID_MAX), rng.randint(minimum, GRID_MAX)


def _validate_generated(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    counts = {int(v): int(np.count_nonzero(grid == v)) for v in np.unique(grid)}
    if counts[bg] <= max(count for color, count in counts.items() if color != bg):
        return False
    out = main(grid)
    return bool(np.any(out != grid))


def _gen_attempt() -> np.ndarray | None:
    rng = random.Random()
    count = rng.choice(STRUCTURE_COUNTS)
    height, width = _sample_grid_shape(rng, count)

    bg = rng.randrange(10)
    palette = [color for color in range(10) if color != bg]
    rng.shuffle(palette)
    active = palette[: rng.randint(4, min(6, len(palette)))]

    grid = np.full((height, width), bg, dtype=int)
    occupied: list[tuple[int, int, int, int]] = []
    chosen_dirs = rng.sample(DIR_PATTERNS, count)

    for direction in chosen_dirs:
        offsets, shell_h, shell_w = _pick_kernel(rng, direction, count)
        shell_color = rng.choice(active)
        kernel_choices = [color for color in active if color != shell_color]
        kernel_color = rng.choice(kernel_choices)

        placed = False
        for _ in range(200):
            if height - shell_h - 2 < 0 or width - shell_w - 2 < 0:
                return None
            top = rng.randint(1, height - shell_h - 1)
            left = rng.randint(1, width - shell_w - 1)
            box = (top, left, top + shell_h - 1, left + shell_w - 1)
            padded = _dilate_box(box, 1, height, width)
            if any(_boxes_intersect(padded, other) for other in occupied):
                continue

            probe = np.full((shell_h, shell_w), bg, dtype=int)
            _draw_structure(probe, 0, 0, shell_h, shell_w, shell_color, kernel_color, offsets)
            if tuple(_kernel_directions((0, 0, shell_h - 1, shell_w - 1), np.argwhere(probe == kernel_color))) != direction:
                continue

            _draw_structure(grid, top, left, shell_h, shell_w, shell_color, kernel_color, offsets)
            occupied.append(padded)
            placed = True
            break
        if not placed:
            return None

    if not _validate_generated(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-shell puzzle input")
