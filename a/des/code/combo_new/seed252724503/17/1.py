import random
from collections import Counter

import numpy as np


BG = 0
FRAME = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
ROTATION_BY_QUADRANT = {
    "ul": 0,
    "ur": 3,  # 90 degrees clockwise
    "lr": 2,
    "ll": 1,  # 270 degrees clockwise
}


def _components(mask: np.ndarray):
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(cells)
    return comps


def _find_frame(grid: np.ndarray):
    h, w = grid.shape
    for r0 in range(h):
        for c0 in range(w):
            if grid[r0, c0] != FRAME:
                continue
            for r1 in range(r0 + 3, h):
                if grid[r1, c0] != FRAME:
                    continue
                for c1 in range(c0 + 3, w):
                    if grid[r0, c1] != FRAME or grid[r1, c1] != FRAME:
                        continue
                    if np.all(grid[r0, c0 : c1 + 1] == FRAME) and np.all(grid[r1, c0 : c1 + 1] == FRAME):
                        if np.all(grid[r0 : r1 + 1, c0] == FRAME) and np.all(grid[r0 : r1 + 1, c1] == FRAME):
                            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
                            if interior.size and np.count_nonzero(interior != BG) == 1:
                                return r0, c0, r1, c1
    raise ValueError("frame not found")


def _extract_templates(grid: np.ndarray, frame_bbox):
    r0, c0, r1, c1 = frame_bbox
    mask = (grid != BG) & (grid != FRAME)
    mask[r0 + 1 : r1, c0 + 1 : c1] = False
    comps = _components(mask)
    templates = []
    for cells in comps:
        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        rr0, rr1 = min(rows), max(rows)
        cc0, cc1 = min(cols), max(cols)
        patch = grid[rr0 : rr1 + 1, cc0 : cc1 + 1].copy()
        templates.append(patch)
    return templates


def _quadrant(seed_r: int, seed_c: int, h: int, w: int) -> str:
    top = seed_r * 2 < h
    left = seed_c * 2 < w
    if top and left:
        return "ul"
    if top and not left:
        return "ur"
    if not top and left:
        return "ll"
    return "lr"


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the framed region with the rotated outer template selected by the seed color and quadrant.

    Input structure:
    - The background is black, with several small multicolor templates scattered above or left of a larger grey rectangular frame.
    - The frame border is one pixel thick and its interior is mostly background.
    - Exactly one colored seed pixel appears inside one quadrant of the frame interior.

    Rule:
    - Match the seed color to the unique outer template containing that color.
    - Rotate that template according to the seed quadrant: upper-left=0 degrees, upper-right=90 clockwise, lower-right=180 degrees, lower-left=270 clockwise.
    - Scale the rotated template by the integer factor implied by the frame interior size and use it to fill the full interior, covering the seed.

    Key insight:
    - The seed color selects which template to use, while the seed position independently selects how that template must be rotated.
    """
    g = np.asarray(grid, dtype=int)
    frame_bbox = _find_frame(g)
    r0, c0, r1, c1 = frame_bbox
    interior = g[r0 + 1 : r1, c0 + 1 : c1]
    seed_cells = np.argwhere(interior != BG)
    sr, sc = map(int, seed_cells[0])
    seed_color = int(interior[sr, sc])

    template = None
    for candidate in _extract_templates(g, frame_bbox):
        if np.any(candidate == seed_color):
            template = candidate
            break
    if template is None:
        raise ValueError("matching template not found")

    q = _quadrant(sr, sc, interior.shape[0], interior.shape[1])
    rotated = np.rot90(template, k=ROTATION_BY_QUADRANT[q])
    th, tw = rotated.shape
    ih, iw = interior.shape
    scale_h = ih // th
    scale_w = iw // tw
    if scale_h != scale_w or scale_h <= 0 or scale_h * th != ih or scale_w * tw != iw:
        raise ValueError("frame interior is not an integer scale of the selected template")

    out = g.copy()
    out[r0 + 1 : r1, c0 + 1 : c1] = np.kron(rotated, np.ones((scale_h, scale_h), dtype=int))
    return out


def _random_template(rng: random.Random, unique: int, blocked: set[int]):
    size = rng.choice([3, 4])
    filler_pool = [c for c in PALETTE if c != unique and c not in blocked]
    filler_count = rng.randint(1, min(3, len(filler_pool)))
    colors = [unique] + rng.sample(filler_pool, filler_count)
    patch = np.array([[rng.choice(colors) for _ in range(size)] for _ in range(size)], dtype=int)
    patch[patch == unique] = colors[1] if len(colors) > 1 else filler_pool[0]
    pr, pc = rng.randrange(size), rng.randrange(size)
    patch[pr, pc] = unique
    return patch, unique


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int):
    h, w = patch.shape
    if np.any(grid[top : top + h, left : left + w] != BG):
        return False
    grid[top : top + h, left : left + w] = patch
    return True


def _gen_attempt(rng: random.Random):
    template_count = rng.randint(2, 4)
    templates = []
    unique_colors = rng.sample(PALETTE, template_count)
    blocked = set(unique_colors)
    for unique in unique_colors:
        templates.append(_random_template(rng, unique, blocked))

    target_idx = rng.randrange(template_count)
    target_template, seed_color = templates[target_idx]
    scale = rng.randint(1, 3)
    th, tw = target_template.shape
    interior_h = th * scale
    interior_w = tw * scale
    quadrant = rng.choice(["ul", "ur", "ll", "lr"])

    max_template_h = max(t.shape[0] for t, _ in templates)
    max_template_w = max(t.shape[1] for t, _ in templates)
    top_band = max_template_h + 2
    left_band = max_template_w + 2

    frame_h = interior_h + 2
    frame_w = interior_w + 2
    grid_h = top_band + frame_h + rng.randint(2, 4)
    grid_w = left_band + frame_w + rng.randint(2, 4)
    if grid_h > 30 or grid_w > 30:
        return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    frame_top = top_band
    frame_left = left_band
    grid[frame_top, frame_left : frame_left + frame_w] = FRAME
    grid[frame_top + frame_h - 1, frame_left : frame_left + frame_w] = FRAME
    grid[frame_top : frame_top + frame_h, frame_left] = FRAME
    grid[frame_top : frame_top + frame_h, frame_left + frame_w - 1] = FRAME

    occupied = np.zeros_like(grid, dtype=bool)
    occupied[frame_top : frame_top + frame_h, frame_left : frame_left + frame_w] = True

    for patch, _ in templates:
        ph, pw = patch.shape
        placed = False
        options = []
        for top in range(0, grid_h - ph + 1):
            for left in range(0, grid_w - pw + 1):
                in_top_zone = top + ph <= frame_top - 1
                in_left_zone = left + pw <= frame_left - 1
                if in_top_zone or in_left_zone:
                    options.append((top, left))
        rng.shuffle(options)
        for top, left in options:
            if np.any(occupied[top : top + ph, left : left + pw]):
                continue
            grid[top : top + ph, left : left + pw] = patch
            rr0 = max(0, top - 1)
            cc0 = max(0, left - 1)
            rr1 = min(grid_h, top + ph + 1)
            cc1 = min(grid_w, left + pw + 1)
            occupied[rr0:rr1, cc0:cc1] = True
            placed = True
            break
        if not placed:
            return None

    inner_top = frame_top + 1
    inner_left = frame_left + 1
    if quadrant == "ul":
        sr = rng.randrange(0, max(1, interior_h // 2))
        sc = rng.randrange(0, max(1, interior_w // 2))
    elif quadrant == "ur":
        sr = rng.randrange(0, max(1, interior_h // 2))
        sc = rng.randrange(interior_w // 2, interior_w)
    elif quadrant == "ll":
        sr = rng.randrange(interior_h // 2, interior_h)
        sc = rng.randrange(0, max(1, interior_w // 2))
    else:
        sr = rng.randrange(interior_h // 2, interior_h)
        sc = rng.randrange(interior_w // 2, interior_w)
    grid[inner_top + sr, inner_left + sc] = seed_color
    return grid


def generate_input() -> np.ndarray:
    for _ in range(300):
        grid = _gen_attempt(random.Random())
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
