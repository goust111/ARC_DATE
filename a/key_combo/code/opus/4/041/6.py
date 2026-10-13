import importlib.util
import random

import numpy as np


_ARC_UTILS_PATH = "/home/lixinyu/snap/new/solvers/arc_utils.py"
_ARC_SPEC = importlib.util.spec_from_file_location("arc_utils_task_041_6", _ARC_UTILS_PATH)
if _ARC_SPEC is None or _ARC_SPEC.loader is None:
    raise ImportError(f"cannot load arc_utils from {_ARC_UTILS_PATH}")
_ARC_UTILS = importlib.util.module_from_spec(_ARC_SPEC)
_ARC_SPEC.loader.exec_module(_ARC_UTILS)

bbox = _ARC_UTILS.bbox
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
scale_sprite = _ARC_UTILS.scale_sprite


BG = 0
BLUE = 1
RED = 2
GREEN = 3
MAX_ATTEMPTS = 300
SEED_PATTERNS = (
    np.array([[1, 0], [0, 1]], dtype=bool),
    np.array([[0, 1], [1, 0]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
)


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _find_outer_frame(grid: np.ndarray) -> tuple[int, int, int, int]:
    candidates = []
    for comp in find_components(grid, bg=BG, connectivity=4, monochromatic=True, colors=[BLUE], return_format="dict"):
        r0, c0, r1, c1 = comp["bbox"]
        patch = crop_bbox(grid, comp["bbox"])
        frame = _perimeter_mask(*patch.shape)
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        if np.all(patch[frame] == BLUE) and not np.any(patch[~frame] == BLUE):
            candidates.append(comp["bbox"])
    if not candidates:
        raise ValueError("blue outer frame not found")
    return max(candidates, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))


def _find_inner_rect(grid: np.ndarray, outer_bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    br0, bc0, br1, bc1 = outer_bbox
    candidates = []
    for comp in find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=False,
        colors=[RED, GREEN],
        return_format="dict",
    ):
        r0, c0, r1, c1 = comp["bbox"]
        if not (br0 < r0 <= r1 < br1 and bc0 < c0 <= c1 < bc1):
            continue
        if r0 <= br0 + 1 or c0 <= bc0 + 1 or r1 >= br1 - 1 or c1 >= bc1 - 1:
            continue
        patch = crop_bbox(grid, comp["bbox"])
        if np.all(np.isin(patch, (RED, GREEN))) and np.any(patch == RED) and np.any(patch == GREEN):
            candidates.append(comp["bbox"])
    if not candidates:
        raise ValueError("inner red/green rectangle not found")
    return max(candidates, key=lambda b: (b[2] - b[0] + 1) * (b[3] - b[1] + 1))


def _scaled_inner_patch(grid: np.ndarray, inner_bbox: tuple[int, int, int, int]) -> np.ndarray:
    inner = crop_bbox(grid, inner_bbox)
    red_box = bbox(inner == RED)
    if red_box is None:
        raise ValueError("seed pattern not found")
    seed = crop_bbox((inner == RED).astype(int), red_box).astype(bool)
    k = min(inner.shape)
    scaled = scale_sprite(seed.astype(int), k).astype(bool)
    patch = np.full(inner.shape, GREEN, dtype=int)
    h = min(patch.shape[0], scaled.shape[0])
    w = min(patch.shape[1], scaled.shape[1])
    patch[:h, :w] = np.where(scaled[:h, :w], RED, patch[:h, :w])
    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Expand a concentric blue frame outward while enlarging the red seed pattern inside the green core.

    Input structure:
    - The background is black.
    - A single blue one-cell rectangular frame surrounds one compact inner rectangle with a gap of black cells.
    - The inner rectangle is filled with green except for a red seed pattern whose tight bbox is 2x2.

    Rule:
    - Find the green/red inner rectangle and read its short side length k.
    - Crop the red seed to its own bbox, scale each seed pixel into a k-by-k red block, and place that scaled pattern from the inner rectangle's top-left corner, clipping to the inner rectangle; uncovered inner cells stay green.
    - Expand the blue frame outward by the inner rectangle height on top and bottom and by the inner rectangle width on left and right; fill the newly covered outer ring with green and clip at the grid boundary.
    - Keep the original blue frame, the rest of the inner rectangle, and all untouched background cells unchanged.

    Key insight:
    - The inner rectangle size drives two linked effects at once: its short side is the red-pattern zoom factor, while its full height and width determine how far the outer structure grows.
    """
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")

    outer_bbox = _find_outer_frame(g)
    inner_bbox = _find_inner_rect(g, outer_bbox)

    br0, bc0, br1, bc1 = outer_bbox
    ir0, ic0, ir1, ic1 = inner_bbox
    inner_h = ir1 - ir0 + 1
    inner_w = ic1 - ic0 + 1

    out = g.copy()
    er0 = max(0, br0 - inner_h)
    ec0 = max(0, bc0 - inner_w)
    er1 = min(g.shape[0] - 1, br1 + inner_h)
    ec1 = min(g.shape[1] - 1, bc1 + inner_w)
    out[er0 : er1 + 1, ec0 : ec1 + 1] = GREEN
    out[br0 : br1 + 1, bc0 : bc1 + 1] = g[br0 : br1 + 1, bc0 : bc1 + 1]
    out[ir0 : ir1 + 1, ic0 : ic1 + 1] = _scaled_inner_patch(g, inner_bbox)
    return out


def _choose_inner_dims() -> tuple[int, int]:
    k = random.choice((2, 3))
    long_side = k + random.randint(1, 3)
    if random.random() < 0.5:
        return k, long_side
    return long_side, k


def _pick_position(total: int, span: int, grow: int, must_clip: bool) -> int:
    max_start = total - span
    if must_clip:
        ranges = []
        low_hi = min(grow - 1, max_start)
        if low_hi >= 0:
            ranges.append((0, low_hi))
        high_lo = max(0, max_start - grow + 1)
        if high_lo <= max_start:
            ranges.append((high_lo, max_start))
        if not ranges:
            raise ValueError("no clipped placement available")
        lo, hi = random.choice(ranges)
        return random.randint(lo, hi)
    lo = grow
    hi = max_start - grow
    if lo > hi:
        return random.randint(0, max_start)
    return random.randint(lo, hi)


def _gen_attempt() -> np.ndarray | None:
    inner_h, inner_w = _choose_inner_dims()
    gap_r = random.randint(1, 3)
    gap_c = random.randint(1, 3)
    outer_h = inner_h + 2 * gap_r + 2
    outer_w = inner_w + 2 * gap_c + 2

    clipped = random.random() < 0.4
    if clipped:
        clip_r, clip_c = random.choice(((True, False), (False, True), (True, True)))
    else:
        clip_r, clip_c = False, False

    min_h = outer_h if clip_r else outer_h + 2 * inner_h
    min_w = outer_w if clip_c else outer_w + 2 * inner_w
    if min_h > 30 or min_w > 30:
        return None

    grid_h = random.randint(min_h, 30)
    grid_w = random.randint(min_w, 30)

    try:
        br0 = _pick_position(grid_h, outer_h, inner_h, clip_r)
        bc0 = _pick_position(grid_w, outer_w, inner_w, clip_c)
    except ValueError:
        return None
    br1 = br0 + outer_h - 1
    bc1 = bc0 + outer_w - 1

    ir0 = br0 + gap_r + 1
    ic0 = bc0 + gap_c + 1
    ir1 = ir0 + inner_h - 1
    ic1 = ic0 + inner_w - 1

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    grid[br0, bc0 : bc1 + 1] = BLUE
    grid[br1, bc0 : bc1 + 1] = BLUE
    grid[br0 : br1 + 1, bc0] = BLUE
    grid[br0 : br1 + 1, bc1] = BLUE
    grid[ir0 : ir1 + 1, ic0 : ic1 + 1] = GREEN

    seed_pattern = random.choice(SEED_PATTERNS)
    seed_r0 = ir0 + random.randint(0, inner_h - 2)
    seed_c0 = ic0 + random.randint(0, inner_w - 2)
    seed_cells = np.argwhere(seed_pattern)
    for dr, dc in seed_cells:
        grid[seed_r0 + dr, seed_c0 + dc] = RED

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    if get_bg(grid) != BG:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
