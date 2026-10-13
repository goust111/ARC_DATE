from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import bbox_coords, bbox_h, bbox_w, components_mask, find_components, get_bg

BG = 0
COLOR_POOL = (1, 2, 3, 4, 5)
GRID_SIDE_RANGE = (18, 26)
OBJECT_COUNT_RANGE = (3, 5)
AREA_RANGE = (5, 12)
PLACEMENT_GAP = 2
PLACEMENT_RETRIES = 300
SHAPE_RETRIES = 300
GEN_RETRIES = 400


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = bbox_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _local_mask(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (int(r0), int(c0), int(r1), int(c1))


def _mirror_mask(mask: np.ndarray, axis: str) -> np.ndarray:
    if axis == "H":
        return mask[:, ::-1]
    return mask[::-1, :]


def _object_kernel(mask: np.ndarray) -> tuple[np.ndarray, str]:
    axis = "H" if mask.shape[1] >= mask.shape[0] else "V"
    return mask & _mirror_mask(mask, axis), axis


def _is_connected(mask: np.ndarray) -> bool:
    return bool(mask.any()) and len(components_mask(mask, connectivity=4)) == 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract each object's self-axis symmetry kernel by erasing pixels that fail its bbox-directed mirror test.

    Input structure:
    - A black background contains several separated monochromatic 4-connected foreground objects.
    - Each object has its own color and an irregular non-rectangular shape.
    - Different objects can have different bbox aspect ratios, so some are wider than tall and others are taller than wide.

    Rule:
    - Find every 4-connected foreground object and compute its tight bounding box.
    - If an object's bbox width is at least its height, mirror that object left-to-right inside its own bbox; otherwise mirror it top-to-bottom inside its own bbox.
    - Keep only the pixels that are present in both the original object and its mirrored copy, and erase all other pixels to background.
    - Apply that test independently to every object and combine the remaining symmetry kernels on the original canvas.

    Key insight:
    - The mirror axis is not global: each object chooses its own axis from its own bbox proportions, so the output is the intersection of the object with its self-mirror under that locally chosen axis.
    """
    g = np.asarray(grid, dtype=int)
    bg = int(get_bg(g))
    out = np.full_like(g, bg)

    for comp in find_components(g, bg=bg, connectivity=4, monochromatic=True):
        color = int(comp["color"])
        mask, (r0, c0, r1, c1) = _local_mask(comp["coords"])
        kernel, _axis = _object_kernel(mask)
        view = out[r0 : r1 + 1, c0 : c1 + 1]
        view[kernel] = color

    return out


def _centered_bounds(width: int, span: int) -> tuple[int, int]:
    left = (width - span) // 2
    return left, left + span


def _asymmetric_candidates(mask: np.ndarray, side: str) -> list[tuple[int, int]]:
    h, w = mask.shape
    if side == "L":
        allowed = lambda c: c < w // 2
    else:
        allowed = lambda c: c > (w - 1) // 2

    candidates: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for r, c in np.argwhere(mask):
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            nr = int(nr)
            nc = int(nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr, nc] or not allowed(nc):
                continue
            if mask[nr, w - 1 - nc]:
                continue
            cell = (nr, nc)
            if cell not in seen:
                seen.add(cell)
                candidates.append(cell)
    return candidates


def _generate_horizontal_mask(rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        canvas_h = rng.randint(3, 6)
        canvas_w = rng.randint(max(canvas_h, 4), 7)
        span_choices = list(range(1 if canvas_w % 2 else 2, canvas_w, 2))
        if not span_choices:
            continue

        run_h = rng.randint(2, canvas_h)
        row0 = rng.randint(0, canvas_h - run_h)
        spans = [rng.choice(span_choices) for _ in range(run_h)]

        core = np.zeros((canvas_h, canvas_w), dtype=bool)
        for idx, span in enumerate(spans):
            r = row0 + idx
            c0, c1 = _centered_bounds(canvas_w, span)
            core[r, c0:c1] = True

        core_area = int(core.sum())
        if core_area < 2 or core_area >= AREA_RANGE[1]:
            continue

        shape = core.copy()
        goal_area = rng.randint(max(AREA_RANGE[0], core_area + 1), AREA_RANGE[1])
        side = rng.choice(("L", "R"))
        extras = 0
        while int(shape.sum()) < goal_area:
            candidates = _asymmetric_candidates(shape, side)
            if not candidates:
                break
            rr, cc = rng.choice(candidates)
            shape[rr, cc] = True
            extras += 1
            if extras >= 1 and int(shape.sum()) >= AREA_RANGE[0] and rng.random() < 0.25:
                break

        if extras == 0:
            continue

        cropped = _crop_mask(shape)
        area = int(cropped.sum())
        if not (AREA_RANGE[0] <= area <= AREA_RANGE[1]):
            continue
        if cropped.shape[1] <= cropped.shape[0]:
            continue
        if area == cropped.shape[0] * cropped.shape[1]:
            continue
        if not _is_connected(cropped):
            continue

        correct, _axis = _object_kernel(cropped)
        wrong = cropped & _mirror_mask(cropped, "V")
        if int(correct.sum()) < 2 or int(correct.sum()) >= area:
            continue
        if not _is_connected(correct):
            continue
        if np.array_equal(correct, wrong):
            continue
        return cropped

    raise RuntimeError("failed to generate a valid horizontal object")


def _generate_object_mask(axis: str, rng: random.Random) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        mask = _generate_horizontal_mask(rng)
        if axis == "H":
            return mask
        transposed = mask.T.copy()
        correct, actual_axis = _object_kernel(transposed)
        wrong = transposed & _mirror_mask(transposed, "H")
        area = int(transposed.sum())
        if actual_axis != "V":
            continue
        if int(correct.sum()) < 2 or int(correct.sum()) >= area:
            continue
        if not _is_connected(correct):
            continue
        if np.array_equal(correct, wrong):
            continue
        return transposed
    raise RuntimeError(f"failed to generate a valid {axis} object")


def _too_close(r0: int, c0: int, h: int, w: int, placed: list[tuple[int, int, int, int]]) -> bool:
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for pr0, pc0, ph, pw in placed:
        pr1 = pr0 + ph - 1
        pc1 = pc0 + pw - 1
        if not (
            r1 + PLACEMENT_GAP < pr0
            or pr1 + PLACEMENT_GAP < r0
            or c1 + PLACEMENT_GAP < pc0
            or pc1 + PLACEMENT_GAP < c0
        ):
            return True
    return False


def _place_patch(
    grid: np.ndarray,
    patch: np.ndarray,
    placed: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> bool:
    h, w = patch.shape
    H, W = grid.shape
    row_lo = 1
    col_lo = 1
    row_hi = H - h - 1
    col_hi = W - w - 1
    if row_hi < row_lo or col_hi < col_lo:
        return False

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(row_lo, row_hi)
        c0 = rng.randint(col_lo, col_hi)
        if _too_close(r0, c0, h, w, placed):
            continue
        view = grid[r0 : r0 + h, c0 : c0 + w]
        if np.any((patch != BG) & (view != BG)):
            continue
        view[patch != BG] = patch[patch != BG]
        placed.append((r0, c0, h, w))
        return True
    return False


def _validate_grid(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False

    comps = find_components(grid, bg=BG, connectivity=4, monochromatic=True)
    if not (OBJECT_COUNT_RANGE[0] <= len(comps) <= OBJECT_COUNT_RANGE[1]):
        return False

    colors = [int(comp["color"]) for comp in comps]
    if len(set(colors)) != len(colors):
        return False
    if any(color not in COLOR_POOL for color in colors):
        return False

    expected = np.zeros_like(grid)
    has_h = False
    has_v = False
    bboxes: list[tuple[int, int, int, int]] = []

    for comp in comps:
        coords = comp["coords"]
        area = int(coords.shape[0])
        if not (AREA_RANGE[0] <= area <= AREA_RANGE[1]):
            return False

        mask, (r0, c0, r1, c1) = _local_mask(coords)
        if area == mask.shape[0] * mask.shape[1]:
            return False
        if not _is_connected(mask):
            return False

        kernel, axis = _object_kernel(mask)
        wrong_axis = "V" if axis == "H" else "H"
        wrong = mask & _mirror_mask(mask, wrong_axis)

        if axis == "H":
            has_h = True
        else:
            has_v = True

        kernel_area = int(kernel.sum())
        if kernel_area < 2 or kernel_area >= area:
            return False
        if not _is_connected(kernel):
            return False
        if np.array_equal(kernel, wrong):
            return False

        color = int(comp["color"])
        expected[r0 : r1 + 1, c0 : c1 + 1][kernel] = color
        bboxes.append((r0, c0, r1, c1))

    if not (has_h and has_v):
        return False
    if np.array_equal(expected, grid):
        return False

    for i, (r0, c0, r1, c1) in enumerate(bboxes):
        for j in range(i + 1, len(bboxes)):
            rr0, cc0, rr1, cc1 = bboxes[j]
            if not (r1 < rr0 or rr1 < r0 or c1 < cc0 or cc1 < c0):
                return False

    out = main(grid)
    if not np.array_equal(out, expected):
        return False

    out_comps = find_components(out, bg=BG, connectivity=4, monochromatic=True)
    return len(out_comps) == len(comps)


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    H = rng.randint(*GRID_SIDE_RANGE)
    W = rng.randint(*GRID_SIDE_RANGE)
    grid = np.zeros((H, W), dtype=int)

    count = rng.randint(*OBJECT_COUNT_RANGE)
    colors = rng.sample(list(COLOR_POOL), count)
    axes = ["H", "V"] + [rng.choice(("H", "V")) for _ in range(count - 2)]
    rng.shuffle(axes)

    patches: list[np.ndarray] = []
    for color, axis in zip(colors, axes):
        mask = _generate_object_mask(axis, rng)
        patch = np.zeros(mask.shape, dtype=int)
        patch[mask] = color
        patches.append(patch)

    order = list(range(count))
    order.sort(key=lambda idx: patches[idx].shape[0] * patches[idx].shape[1], reverse=True)

    placed: list[tuple[int, int, int, int]] = []
    for idx in order:
        if not _place_patch(grid, patches[idx], placed, rng):
            return None

    return grid if _validate_grid(grid) else None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-kernel input")
