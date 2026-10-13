import numpy as np


BG = 0
FENCE = 9
PALETTE = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=int)
DIRECTIONS = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
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
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _opening_direction(points: list[tuple[int, int]]) -> str:
    pts = set(points)
    r0, c0, r1, c1 = _bbox(points)
    counts = {
        "up": sum((r0, c) in pts for c in range(c0, c1 + 1)),
        "down": sum((r1, c) in pts for c in range(c0, c1 + 1)),
        "left": sum((r, c0) in pts for r in range(r0, r1 + 1)),
        "right": sum((r, c1) in pts for r in range(r0, r1 + 1)),
    }
    return min(DIRECTIONS, key=lambda d: (counts[d], d))


def _paint_boundary_stack(
    grid: np.ndarray, pixels: list[tuple[int, int, int]], direction: str
) -> None:
    h, w = grid.shape
    if direction in ("up", "down"):
        groups: dict[int, list[tuple[int, int]]] = {}
        for r, c, color in pixels:
            groups.setdefault(c, []).append((r, color))
        reverse = direction == "down"
        for c, items in groups.items():
            items.sort(key=lambda item: item[0], reverse=reverse)
            for offset, (_r, color) in enumerate(items):
                rr = offset if direction == "up" else h - 1 - offset
                grid[rr, c] = color
    else:
        groups = {}
        for r, c, color in pixels:
            groups.setdefault(r, []).append((c, color))
        reverse = direction == "right"
        for r, items in groups.items():
            items.sort(key=lambda item: item[0], reverse=reverse)
            for offset, (_c, color) in enumerate(items):
                cc = offset if direction == "left" else w - 1 - offset
                grid[r, cc] = color


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pour the colored pixels out of each maroon U-shaped fence and stack them on the matching grid edge.

    Input structure:
    - Black background with one or more Maroon U-shaped rectangular fences.
    - Each fence is a one-cell-thick three-sided frame with its opening facing up, down, left, or right.
    - The interior of each fence contains several single colored pixels, none of which are Black or Maroon.

    Rule:
    - Identify each Maroon fence and determine which side is open.
    - Remove every colored pixel from the fence interior and move it straight through the opening toward the outer grid boundary.
    - Pixels that share the same projection line stack against that boundary, keeping their original order along the opening direction: the one closest to the opening ends up closest to the boundary.
    - Leave the Maroon fences unchanged and clear every fence interior cell to Black.

    Key insight:
    - The U-shape itself is the movement instruction: the missing side gives the travel direction, and the interior row/column alignment determines how colors stack once they reach the edge.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()

    for comp in _components(g == FENCE):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 < 2 or c1 - c0 < 2:
            continue
        direction = _opening_direction(comp)
        pixels: list[tuple[int, int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                value = int(g[r, c])
                if value not in (BG, FENCE):
                    pixels.append((r, c, value))
                out[r, c] = BG
        _paint_boundary_stack(out, pixels, direction)

    return out


def _render_fence(
    grid: np.ndarray, bbox: tuple[int, int, int, int], direction: str
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FENCE
    grid[r1, c0 : c1 + 1] = FENCE
    grid[r0 : r1 + 1, c0] = FENCE
    grid[r0 : r1 + 1, c1] = FENCE
    if direction == "up":
        grid[r0, c0 + 1 : c1] = BG
    elif direction == "down":
        grid[r1, c0 + 1 : c1] = BG
    elif direction == "left":
        grid[r0 + 1 : r1, c0] = BG
    else:
        grid[r0 + 1 : r1, c1] = BG


def _sample_group_sizes(
    rng: np.random.Generator, total: int, group_count: int, max_size: int
) -> list[int] | None:
    sizes = [1] * group_count
    remaining = total - group_count
    while remaining > 0:
        candidates = [i for i, size in enumerate(sizes) if size < max_size]
        if not candidates:
            return None
        idx = int(rng.choice(candidates))
        sizes[idx] += 1
        remaining -= 1
    rng.shuffle(sizes)
    return sizes


def _sample_pixels(
    rng: np.random.Generator,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> list[tuple[int, int, int]] | None:
    r0, c0, r1, c1 = bbox
    rows = list(range(r0 + 1, r1))
    cols = list(range(c0 + 1, c1))
    if direction in ("up", "down"):
        lines = cols
        depths = rows
    else:
        lines = rows
        depths = cols

    max_total = min(6, len(PALETTE), len(lines) * len(depths))
    if max_total < 2:
        return None

    stack_branch = bool(rng.integers(0, 2))
    if not stack_branch:
        max_unique = min(5, len(lines), max_total)
        if max_unique < 2:
            stack_branch = True
        else:
            total = int(rng.integers(2, max_unique + 1))
            group_count = total
    if stack_branch:
        total = int(rng.integers(2, max_total + 1))
        max_groups = min(total - 1, len(lines))
        if max_groups < 1:
            return None
        group_count = int(rng.integers(1, max_groups + 1))

    sizes = _sample_group_sizes(rng, total, group_count, len(depths))
    if sizes is None:
        return None

    chosen_lines = [
        int(v)
        for v in rng.choice(np.array(lines, dtype=int), size=group_count, replace=False)
    ]
    colors = [int(v) for v in rng.choice(PALETTE, size=total, replace=False)]
    rng.shuffle(colors)

    pixels: list[tuple[int, int, int]] = []
    color_idx = 0
    for line, size in zip(chosen_lines, sizes):
        chosen_depths = [
            int(v)
            for v in rng.choice(np.array(depths, dtype=int), size=size, replace=False)
        ]
        for depth in chosen_depths:
            if direction in ("up", "down"):
                pixels.append((depth, line, colors[color_idx]))
            else:
                pixels.append((line, depth, colors[color_idx]))
            color_idx += 1
    return pixels


def _used_lines(pixels: list[tuple[int, int, int]], direction: str) -> list[int]:
    if direction in ("up", "down"):
        return sorted({c for _r, c, _color in pixels})
    return sorted({r for r, _c, _color in pixels})


def _corridor_clear(
    blocked: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    pixels: list[tuple[int, int, int]],
) -> bool:
    r0, c0, r1, c1 = bbox
    mr0 = max(0, r0 - 1)
    mc0 = max(0, c0 - 1)
    mr1 = min(blocked.shape[0] - 1, r1 + 1)
    mc1 = min(blocked.shape[1] - 1, c1 + 1)
    if blocked[mr0 : mr1 + 1, mc0 : mc1 + 1].any():
        return False

    for line in _used_lines(pixels, direction):
        if direction == "up" and blocked[0 : r0 + 1, line].any():
            return False
        if direction == "down" and blocked[r1:, line].any():
            return False
        if direction == "left" and blocked[line, 0 : c0 + 1].any():
            return False
        if direction == "right" and blocked[line, c1:].any():
            return False
    return True


def _reserve(
    blocked: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
    pixels: list[tuple[int, int, int]],
) -> None:
    r0, c0, r1, c1 = bbox
    mr0 = max(0, r0 - 1)
    mc0 = max(0, c0 - 1)
    mr1 = min(blocked.shape[0] - 1, r1 + 1)
    mc1 = min(blocked.shape[1] - 1, c1 + 1)
    blocked[mr0 : mr1 + 1, mc0 : mc1 + 1] = True

    for line in _used_lines(pixels, direction):
        if direction == "up":
            blocked[0 : r0 + 1, line] = True
        elif direction == "down":
            blocked[r1:, line] = True
        elif direction == "left":
            blocked[line, 0 : c0 + 1] = True
        else:
            blocked[line, c1:] = True


def _place_one_fence(
    rng: np.random.Generator, h: int, w: int, blocked: np.ndarray, direction: str
) -> dict[str, object] | None:
    max_height = min(9, h - 2)
    max_width = min(9, w - 2)
    if max_height < 4 or max_width < 4:
        return None

    for _ in range(PLACE_ATTEMPTS):
        fh = int(rng.integers(4, max_height + 1))
        fw = int(rng.integers(4, max_width + 1))
        r0 = int(rng.integers(1, h - fh))
        c0 = int(rng.integers(1, w - fw))
        bbox = (r0, c0, r0 + fh - 1, c0 + fw - 1)
        pixels = _sample_pixels(rng, bbox, direction)
        if pixels is None:
            continue
        if not _corridor_clear(blocked, bbox, direction, pixels):
            continue
        _reserve(blocked, bbox, direction, pixels)
        return {"bbox": bbox, "direction": direction, "pixels": pixels}
    return None


def _direction_sequence(rng: np.random.Generator, count: int) -> list[str]:
    pool: list[str] = []
    while len(pool) < count:
        perm = [DIRECTIONS[i] for i in rng.permutation(len(DIRECTIONS))]
        pool.extend(perm)
    return pool[:count]


def _build_grids(h: int, w: int, specs: list[dict[str, object]]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full((h, w), BG, dtype=int)
    for spec in specs:
        bbox = spec["bbox"]
        direction = spec["direction"]
        pixels = spec["pixels"]
        _render_fence(inp, bbox, direction)
        for r, c, color in pixels:
            inp[r, c] = color

    out = inp.copy()
    for spec in specs:
        pixels = spec["pixels"]
        for r, c, _color in pixels:
            out[r, c] = BG
        _paint_boundary_stack(out, pixels, spec["direction"])
    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(16, 31))
    w = int(rng.integers(16, 31))
    area = h * w
    if area < 360:
        max_fences = 2
    elif area < 600:
        max_fences = 3
    else:
        max_fences = 4
    fence_count = int(rng.integers(1, max_fences + 1))

    blocked = np.zeros((h, w), dtype=bool)
    specs: list[dict[str, object]] = []
    for direction in _direction_sequence(rng, fence_count):
        spec = _place_one_fence(rng, h, w, blocked, direction)
        if spec is None:
            return None
        specs.append(spec)

    inp, expected = _build_grids(h, w, specs)
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input failed after repeated placement attempts")
