import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5
FRAME_COLORS = {GREY, RED, BLUE}
PATTERN_COLORS = [3, 4, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ROTATION_MARKS = (
    {"top", "left"},
    {"top", "right"},
    {"bottom", "left"},
    {"bottom", "right"},
)
OUTER_SIZE_CHOICES = [7, 9, 11, 13]
GENERATION_ATTEMPTS = 400
PLACEMENT_ATTEMPTS = 200


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _detect_frame(
    grid: np.ndarray, coords: list[tuple[int, int]]
) -> tuple[tuple[int, int, int, int], str] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None
    if (r1 - r0) % 2 != 0 or (c1 - c0) % 2 != 0:
        return None

    expected = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
    if len(coords) != expected:
        return None

    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            is_frame_color = int(grid[r, c]) in FRAME_COLORS
            if on_border != is_frame_color:
                return None

    mr = (r0 + r1) // 2
    mc = (c0 + c1) // 2
    marks = {
        "top": int(grid[r0, mc]),
        "bottom": int(grid[r1, mc]),
        "left": int(grid[mr, c0]),
        "right": int(grid[mr, c1]),
    }
    blue_sides = {side for side, value in marks.items() if value == BLUE}
    red_sides = {side for side, value in marks.items() if value == RED}

    if blue_sides:
        if red_sides or blue_sides not in ROTATION_MARKS:
            return None
        return (r0, c0, r1, c1), "rotation"
    if red_sides and red_sides <= {"top", "bottom"}:
        return (r0, c0, r1, c1), "horizontal"
    if red_sides and red_sides <= {"left", "right"}:
        return (r0, c0, r1, c1), "vertical"
    return None


def _find_frames(grid: np.ndarray) -> list[tuple[tuple[int, int, int, int], str]]:
    mask = np.isin(grid, list(FRAME_COLORS))
    frames: list[tuple[tuple[int, int, int, int], str]] = []
    for comp in _components(mask):
        found = _detect_frame(grid, comp)
        if found is not None:
            frames.append(found)
    frames.sort(key=lambda item: item[0])
    return frames


def _complete_inner(inner: np.ndarray, mode: str) -> np.ndarray:
    out = inner.copy()
    filled = np.argwhere(inner != BLACK)

    if mode == "horizontal":
        h = inner.shape[0]
        for r, c in filled.tolist():
            out[h - 1 - r, c] = inner[r, c]
        return out

    if mode == "vertical":
        w = inner.shape[1]
        for r, c in filled.tolist():
            out[r, w - 1 - c] = inner[r, c]
        return out

    size_r, size_c = inner.shape
    assert size_r == size_c, "rotation symmetry requires a square interior"
    for r, c in filled.tolist():
        color = int(inner[r, c])
        rr, cc = int(r), int(c)
        for _ in range(4):
            out[rr, cc] = color
            rr, cc = cc, size_r - 1 - rr
    return out


def _solve_grid(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames = _find_frames(g)
    assert frames, "expected at least one marked frame"

    for (r0, c0, r1, c1), mode in frames:
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        out[r0 + 1 : r1, c0 + 1 : c1] = _complete_inner(inner, mode)
        out[r0, c0 : c1 + 1] = GREY
        out[r1, c0 : c1 + 1] = GREY
        out[r0 : r1 + 1, c0] = GREY
        out[r0 : r1 + 1, c1] = GREY

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve_grid(grid)


def _random_blob(
    rng: random.Random, height: int, width: int, min_cells: int = 2
) -> np.ndarray:
    area = height * width
    max_cells = max(min_cells, min(area - 1, int(round(area * 0.55))))
    target = rng.randint(min_cells, max_cells)

    cells = {(rng.randrange(height), rng.randrange(width))}
    while len(cells) < target:
        frontier = []
        for r, c in cells:
            for dr, dc in DIRS4:
                nr = r + dr
                nc = c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.append((nr, nc))
        if not frontier:
            break
        cells.add(rng.choice(frontier))

    if len(cells) < min_cells:
        raise ValueError("blob generation failed")

    mask = np.zeros((height, width), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _build_patch(rng: random.Random, mode: str, color: int, size_choices: list[int]) -> np.ndarray:
    if mode == "rotation":
        outer_h = rng.choice(size_choices)
        outer_w = outer_h
    else:
        outer_h = rng.choice(size_choices)
        outer_w = rng.choice(size_choices)

    patch = np.full((outer_h, outer_w), BLACK, dtype=int)
    patch[0, :] = GREY
    patch[-1, :] = GREY
    patch[:, 0] = GREY
    patch[:, -1] = GREY

    mid_r = outer_h // 2
    mid_c = outer_w // 2
    if mode == "horizontal":
        patch[rng.choice([0, outer_h - 1]), mid_c] = RED
    elif mode == "vertical":
        patch[mid_r, rng.choice([0, outer_w - 1])] = RED
    else:
        corner = rng.choice(["tl", "tr", "bl", "br"])
        if corner == "tl":
            patch[0, mid_c] = BLUE
            patch[mid_r, 0] = BLUE
        elif corner == "tr":
            patch[0, mid_c] = BLUE
            patch[mid_r, -1] = BLUE
        elif corner == "bl":
            patch[-1, mid_c] = BLUE
            patch[mid_r, 0] = BLUE
        else:
            patch[-1, mid_c] = BLUE
            patch[mid_r, -1] = BLUE

    inner = np.zeros((outer_h - 2, outer_w - 2), dtype=int)
    inner_h, inner_w = inner.shape

    if mode == "horizontal":
        half_h = inner_h // 2
        blob = _random_blob(rng, half_h, inner_w)
        top = rng.choice([0, inner_h - half_h])
        left = 0
    elif mode == "vertical":
        half_w = inner_w // 2
        blob = _random_blob(rng, inner_h, half_w)
        top = 0
        left = rng.choice([0, inner_w - half_w])
    else:
        quad = inner_h // 2
        blob = _random_blob(rng, quad, quad)
        top = rng.choice([0, inner_h - quad])
        left = rng.choice([0, inner_w - quad])

    inner[top : top + blob.shape[0], left : left + blob.shape[1]][blob] = color
    patch[1:-1, 1:-1] = inner
    return patch


def _choose_modes(rng: random.Random, count: int) -> list[str]:
    if count == 2:
        modes = list(rng.choice([("horizontal", "vertical"), ("horizontal", "rotation"), ("vertical", "rotation")]))
    elif count == 3:
        modes = ["horizontal", "vertical", "rotation"]
    else:
        modes = ["horizontal", "vertical", "rotation", rng.choice(["horizontal", "vertical", "rotation"])]
    rng.shuffle(modes)
    return modes


def _place_patches(
    rng: random.Random, grid_h: int, grid_w: int, patches: list[np.ndarray]
) -> np.ndarray | None:
    grid = np.full((grid_h, grid_w), BLACK, dtype=int)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    ordered = sorted(patches, key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in ordered:
        ph, pw = patch.shape
        placed = False
        for _ in range(PLACEMENT_ATTEMPTS):
            top = rng.randint(0, grid_h - ph)
            left = rng.randint(0, grid_w - pw)
            rs = slice(max(0, top - 1), min(grid_h, top + ph + 1))
            cs = slice(max(0, left - 1), min(grid_w, left + pw + 1))
            if occupied[rs, cs].any():
                continue
            grid[top : top + ph, left : left + pw] = patch
            occupied[top : top + ph, left : left + pw] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    count = rng.randint(2, 4)
    modes = _choose_modes(rng, count)
    colors = rng.sample(PATTERN_COLORS, count)
    size_choices = OUTER_SIZE_CHOICES[:-1] if count >= 3 else OUTER_SIZE_CHOICES
    patches = [_build_patch(rng, mode, color, size_choices) for mode, color in zip(modes, colors, strict=False)]

    max_h = max(patch.shape[0] for patch in patches)
    max_w = max(patch.shape[1] for patch in patches)
    min_h = max(22, max_h + 2)
    min_w = max(22, max_w + 2)
    if min_h > 30 or min_w > 30:
        return None

    grid_h = rng.randint(min_h, 30)
    grid_w = rng.randint(min_w, 30)
    grid = _place_patches(rng, grid_h, grid_w, patches)
    if grid is None:
        return None

    frames = _find_frames(grid)
    if len(frames) != count:
        return None

    out = _solve_grid(grid)
    if np.array_equal(out, grid):
        return None
    if np.any(out == RED) or np.any(out == BLUE):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
