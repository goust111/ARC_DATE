from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, bbox, crop_bbox, find_components


BG = Color.BLACK
FG = Color.GREY
ROTATION_TO_COLOR = {
    0: Color.BLUE,
    1: Color.RED,
    2: Color.GREEN,
    3: Color.YELLOW,
}
DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRY_LIMIT = 400


def _same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    return a.shape == b.shape and np.array_equal(a, b)


def _rotation_index(template_mask: np.ndarray, obj_mask: np.ndarray) -> int | None:
    for k in range(4):
        if _same_shape(np.rot90(template_mask, k), obj_mask):
            return k
    return None


def _is_mirror_variant(template_mask: np.ndarray, obj_mask: np.ndarray) -> bool:
    mirrored = np.fliplr(template_mask)
    return any(_same_shape(np.rot90(mirrored, k), obj_mask) for k in range(4))


def _bbox_inside(inner: tuple[int, int, int, int], outer: tuple[int, int, int, int]) -> bool:
    return outer[0] <= inner[0] <= inner[2] <= outer[2] and outer[1] <= inner[1] <= inner[3] <= outer[3]


def _is_rect_frame_candidate(full_mask: np.ndarray, comp: dict) -> bool:
    r0, c0, r1, c1 = map(int, comp["bbox"])
    local = crop_bbox(full_mask, comp["bbox"])
    if local.shape[0] < 5 or local.shape[1] < 5:
        return False
    if not (local[0, :].all() and local[-1, :].all() and local[:, 0].all() and local[:, -1].all()):
        return False
    inner = full_mask[r0 + 1 : r1, c0 + 1 : c1]
    return inner.size > 0 and inner.any() and (~inner).any()


def _extract_frame_and_template(g: np.ndarray) -> tuple[int, tuple[int, int, int, int], np.ndarray, list[dict]]:
    comps = find_components(g, bg=BG, colors=[FG], connectivity=4, monochromatic=True)
    full_mask = g == FG
    frame_candidates = [comp for comp in comps if _is_rect_frame_candidate(full_mask, comp)]
    assert len(frame_candidates) == 1, "expected exactly one template frame"
    frame_bbox = tuple(map(int, frame_candidates[0]["bbox"]))
    r0, c0, r1, c1 = frame_bbox
    inner_mask = full_mask[r0 + 1 : r1, c0 + 1 : c1]
    inner_bbox = bbox(inner_mask)
    assert inner_bbox is not None, "expected a sample inside the frame"
    template_mask = crop_bbox(inner_mask, inner_bbox)
    return FG, frame_bbox, template_mask, comps


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify each template copy by pose, recolor pure rotations, and erase mirrored impostors.

    Input structure:
    - The background is black, and all visible objects are grey.
    - Exactly one thin rectangular frame appears; inside it sits one grey sample shape.
    - Outside the frame are separate grey objects congruent to that sample, either by pure rotation or only by reflection-plus-rotation.

    Rule:
    - Keep the frame and the sample inside it unchanged.
    - For each grey object outside the frame, compare its cropped mask with the sample mask.
    - If it matches a pure rotation of the sample, recolor it by pose: 0 degrees blue, 90 degrees red, 180 degrees green, and 270 degrees yellow; if it matches only a mirrored version, erase it to black.

    Key insight:
    - The frame supplies one canonical chiral template, so the real task is distinguishing rotation-equivalent copies from reflection-equivalent fakes and then encoding the rotation angle as color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    fg, frame_bbox, template_mask, comps = _extract_frame_and_template(g)
    out = g.copy()

    for comp in comps:
        comp_bbox = tuple(map(int, comp["bbox"]))
        if comp_bbox == frame_bbox or _bbox_inside(comp_bbox, frame_bbox):
            continue

        obj_mask = crop_bbox(g == fg, comp_bbox)
        rot_k = _rotation_index(template_mask, obj_mask)
        coords = comp["coords"]
        if rot_k is not None:
            out[coords[:, 0], coords[:, 1]] = int(ROTATION_TO_COLOR[rot_k])
            continue

        assert _is_mirror_variant(template_mask, obj_mask), "every outer object must be either a rotation or a mirror fake"
        out[coords[:, 0], coords[:, 1]] = int(BG)

    return out


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    b = bbox(mask)
    assert b is not None
    return crop_bbox(mask, b)


def _is_connected(mask: np.ndarray) -> bool:
    comps = find_components(mask.astype(int), bg=0, colors=[1], connectivity=4, monochromatic=True)
    return len(comps) == 1


def _is_chiral(mask: np.ndarray) -> bool:
    for k in (1, 2, 3):
        if _same_shape(mask, np.rot90(mask, k)):
            return False
    mirrored = np.fliplr(mask)
    for k in range(4):
        if _same_shape(mask, np.rot90(mirrored, k)):
            return False
    return True


def _random_template_mask() -> np.ndarray:
    for _ in range(300):
        h = random.randint(3, 5)
        w = random.randint(4, 6)
        target = random.randint(max(5, (h + w) // 2 + 1), min(h * w - 2, 9))
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True

        while int(mask.sum()) < target:
            frontier: set[tuple[int, int]] = set()
            for r, c in np.argwhere(mask):
                for dr, dc in DIRS4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                        frontier.add((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(tuple(frontier))
            mask[nr, nc] = True

        mask = _crop_mask(mask)
        if mask.shape[0] < 3 or mask.shape[1] < 3:
            continue
        if int(mask.sum()) < 5 or int(mask.sum()) >= mask.size - 1:
            continue
        if not _is_connected(mask):
            continue
        if not _is_chiral(mask):
            continue
        return mask

    raise RuntimeError("failed to sample a chiral template")


def _can_place(occupied: np.ndarray, shape: tuple[int, int], r0: int, c0: int, padding: int) -> bool:
    h, w = map(int, shape)
    H, W = occupied.shape
    if r0 < 0 or c0 < 0 or r0 + h > H or c0 + w > W:
        return False
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    return not occupied[rr0:rr1, cc0:cc1].any()


def _pick_location(occupied: np.ndarray, shape: tuple[int, int], padding: int) -> tuple[int, int] | None:
    h, w = map(int, shape)
    H, W = occupied.shape
    spots = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
    random.shuffle(spots)
    for r, c in spots:
        if _can_place(occupied, shape, r, c, padding):
            return r, c
    return None


def _mark_occupied(occupied: np.ndarray, shape: tuple[int, int], r0: int, c0: int, padding: int) -> None:
    h, w = map(int, shape)
    H, W = occupied.shape
    rr0 = max(0, r0 - padding)
    cc0 = max(0, c0 - padding)
    rr1 = min(H, r0 + h + padding)
    cc1 = min(W, c0 + w + padding)
    occupied[rr0:rr1, cc0:cc1] = True


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = int(color)


def _draw_frame_with_template(
    grid: np.ndarray,
    occupied: np.ndarray,
    template_mask: np.ndarray,
) -> tuple[int, int, int, int]:
    th, tw = template_mask.shape
    pad_top = random.randint(1, 2)
    pad_bottom = random.randint(1, 2)
    pad_left = random.randint(1, 2)
    pad_right = random.randint(1, 2)
    frame_h = th + pad_top + pad_bottom + 2
    frame_w = tw + pad_left + pad_right + 2

    loc = _pick_location(occupied, (frame_h, frame_w), padding=2)
    if loc is None:
        raise ValueError("no room for frame")
    fr, fc = loc

    frame_patch = np.zeros((frame_h, frame_w), dtype=bool)
    frame_patch[0, :] = True
    frame_patch[-1, :] = True
    frame_patch[:, 0] = True
    frame_patch[:, -1] = True
    _paint_mask(grid, frame_patch, fr, fc, FG)

    tr = fr + 1 + pad_top
    tc = fc + 1 + pad_left
    _paint_mask(grid, template_mask, tr, tc, FG)
    _mark_occupied(occupied, (frame_h, frame_w), fr, fc, padding=2)
    return fr, fc, frame_h, frame_w


def _gen_attempt() -> np.ndarray | None:
    template_mask = np.rot90(_random_template_mask(), random.randrange(4))
    th, tw = template_mask.shape

    min_h = max(18, th + 12)
    min_w = max(18, tw + 16)
    if min_h > 30 or min_w > 30:
        return None

    H = random.randint(min_h, 30)
    W = random.randint(min_w, 30)
    grid = np.full((H, W), int(BG), dtype=int)
    occupied = np.zeros((H, W), dtype=bool)

    try:
        _draw_frame_with_template(grid, occupied, template_mask)
    except ValueError:
        return None

    specs: list[tuple[str, int]] = [("rot", k) for k in range(4)]
    specs.extend(("mirror", random.randrange(4)) for _ in range(random.randint(1, 3)))
    for _ in range(random.randint(0, 3)):
        if random.random() < 0.6:
            specs.append(("rot", random.randrange(4)))
        else:
            specs.append(("mirror", random.randrange(4)))
    random.shuffle(specs)

    mirrored_base = np.fliplr(template_mask)
    for kind, k in specs:
        obj_mask = np.rot90(template_mask, k) if kind == "rot" else np.rot90(mirrored_base, k)
        loc = _pick_location(occupied, obj_mask.shape, padding=2)
        if loc is None:
            return None
        r0, c0 = loc
        _paint_mask(grid, obj_mask, r0, c0, FG)
        _mark_occupied(occupied, obj_mask.shape, r0, c0, padding=2)

    out = main(grid)
    colors = set(map(int, out.flatten()))
    if not all(color in colors for color in ROTATION_TO_COLOR.values()):
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid pose-classification input")
