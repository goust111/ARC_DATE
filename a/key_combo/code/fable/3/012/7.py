from __future__ import annotations

import random
from pathlib import Path
import sys
from typing import Tuple

import numpy as np

# Ensure arc_utils is importable when this file is loaded directly by SourceFileLoader.
for parent in (Path(__file__).resolve().parent, *Path(__file__).resolve().parents):
    _candidate = parent / "new" / "solvers"
    if (_candidate / "arc_utils.py").exists():
        _ARCSOLVER_PATH = str(_candidate)
        break
else:
    _ARCSOLVER_PATH = "/home/lixinyu/snap/new/solvers"

if _ARCSOLVER_PATH not in sys.path:
    sys.path.append(_ARCSOLVER_PATH)

from arc_utils import (  # type: ignore  # noqa: E402
    Color,
    bbox,
    bbox_coords,
    bbox_h,
    bbox_w,
    colors_present,
    components_mask,
    crop_bbox,
    get_bg,
    paste,
    random_free_location,
    random_sprite,
    scale_sprite,
)


def main(grid: np.ndarray) -> np.ndarray:
    """Scale the unique count-consistent sprite into the hollow frame interior.

    Input structure
    A black background with one single-color square frame and several disjoint single-color
    sprites. A square-bounded sprite can be a geometric candidate if its side divides the
    frame interior.

    Rule
    Detect the frame, compute interior side n, and find every candidate sprite whose square
    bbox side s satisfies n % s == 0. Among those candidates choose the unique sprite with
    pixel count k == n // s, scale it by k, and paint it inside the frame interior.

    Key insight
    Multiple geometry-only fits are given by divisibility, but only one is valid under
    self-consistency when pixel count equals the implied scale.
    """

    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    bg = int(get_bg(g))
    assert bg == Color.BLACK, "expected black background"

    colors = colors_present(g, exclude={bg})
    assert colors, "expected at least one non-background color"

    # Detect the one hollow square frame.
    frame_color = None
    frame_bbox = None
    for color in colors:
        for comp in components_mask(g == color, connectivity=4):
            b = bbox_coords(comp)
            if b is None:
                continue
            h = bbox_h(b)
            w = bbox_w(b)
            if h < 3 or h != w:
                continue
            if len(comp) != 4 * h - 4:
                continue
            r0, c0, r1, c1 = b
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            if inner.size <= 0:
                continue
            if int(np.any(inner == color)):
                continue
            if not np.array_equal(inner, np.full_like(inner, bg)):
                continue

            frame_color = int(color)
            frame_bbox = b
            break
        if frame_color is not None:
            break

    assert frame_color is not None and frame_bbox is not None, "expected exactly one frame"

    r0, c0, r1, c1 = map(int, frame_bbox)
    interior_side = int(r1 - r0 - 1)
    assert interior_side >= 1, "invalid frame interior"

    geometry_candidates: list[tuple[int, int, int, tuple[int, int, int, int], np.ndarray]] = []
    for color in [c for c in colors if c != frame_color]:
        for comp in components_mask(g == color, connectivity=4):
            b = bbox_coords(comp)
            if b is None:
                continue
            h = bbox_h(b)
            w = bbox_w(b)
            if h != w or h < 2:
                continue
            if interior_side % h != 0:
                continue

            sprite = g[b[0] : b[2] + 1, b[1] : b[3] + 1]
            sprite_colors = set(map(int, sprite.flatten())) - {bg}
            if sprite_colors != {int(color)}:
                continue

            k = interior_side // h
            area = int(len(comp))
            geometry_candidates.append((area, k, int(color), b, comp))

    assert len(geometry_candidates) >= 2, "expected at least two geometric candidates"

    selected = [item for item in geometry_candidates if item[0] == item[1]]
    assert len(selected) == 1, "expected exactly one count-consistent candidate"

    selected_area, k, color, b, comp = selected[0]
    assert k >= 1 and len(comp) == selected_area

    out = crop_bbox(g, frame_bbox).copy()
    sprite_bbox = b
    patch = g[sprite_bbox[0] : sprite_bbox[2] + 1, sprite_bbox[1] : sprite_bbox[3] + 1]
    sprite_mask = patch != bg

    sprite = np.full_like(patch, bg)
    sprite[sprite_mask] = color
    scaled = scale_sprite(sprite, k)
    assert scaled.shape == (interior_side, interior_side), "scaled sprite size mismatch"

    paste(out, scaled, 1, 1, transparent=bg)
    return out


def _min_area_for_square_bbox(side: int) -> int:
    return max(3, 2 * side - 1)


def _random_connected_mask_with_square_bbox(side: int, area: int, rng: random.Random) -> np.ndarray:
    side = int(side)
    area = int(area)
    min_area = _min_area_for_square_bbox(side)
    if side < 2 or area < min_area or area > side * side:
        raise ValueError("invalid side or area")

    # Base connected shape with full square bbox.
    mask = np.zeros((side, side), dtype=bool)
    for c in range(side):
        mask[0, c] = True
    for r in range(side):
        mask[r, 0] = True

    current = int(mask.sum())
    if current > area:
        raise ValueError("failed to generate square sprite mask")

    need = area - current
    if need > 0:
        empties = [(int(r), int(c)) for r in range(side) for c in range(side) if not bool(mask[r, c])]
        if need > len(empties):
            raise ValueError("failed to generate square sprite mask")
        rng.shuffle(empties)
        for r, c in empties[:need]:
            mask[r, c] = True

    b = bbox_coords(np.argwhere(mask))
    if b is None or bbox_h(b) != side or bbox_w(b) != side:
        raise ValueError("failed to generate square sprite mask")

    # Randomized orientation for diversity.
    for _ in range(rng.randrange(4)):
        mask = np.rot90(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    if rng.random() < 0.5:
        mask = np.flipud(mask)

    return mask


def _sprite_patch(side: int, area: int, color: int, rng: random.Random) -> np.ndarray:
    mask = _random_connected_mask_with_square_bbox(side, area, rng)
    patch = np.full((side, side), Color.BLACK, dtype=int)
    patch[mask] = color
    return patch


def _frame_patch(side: int, color: int) -> np.ndarray:
    patch = np.full((side, side), Color.BLACK, dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _place_patch(canvas: np.ndarray, patch: np.ndarray, *, padding: int = 1) -> bool:
    try:
        r, c = random_free_location(
            canvas,
            patch,
            transparent=Color.BLACK,
            border=0,
            padding=padding,
            padding_connectivity=8,
        )
    except ValueError:
        return False
    paste(canvas, patch, r, c, transparent=Color.BLACK)
    return True


def _generate_once() -> np.ndarray:
    rng = random.Random(np.random.randint(0, 2**31 - 1))

    interior_side = rng.choice([18, 24, 28])
    selected_side = 4 if interior_side == 28 else 3
    selected_scale = interior_side // selected_side
    if not (_min_area_for_square_bbox(selected_side) <= selected_scale <= selected_side * selected_side):
        raise RuntimeError("invalid selected scale")

    distractor_side = 2
    distractor_scale = interior_side // distractor_side
    distractor_area = 3 if distractor_scale != 3 else 4

    frame_color = rng.choice([c for c in Color.NOT_BLACK if c != Color.TEAL])
    used_colors = {frame_color}

    frame = _frame_patch(interior_side + 2, frame_color)
    canvas = np.zeros((30, 30), dtype=int)

    if not _place_patch(canvas, frame, padding=0):
        raise RuntimeError("failed to place frame")

    selected_color = rng.choice([c for c in Color.NOT_BLACK if c not in used_colors])
    used_colors.add(selected_color)
    selected_patch = _sprite_patch(selected_side, selected_scale, selected_color, rng)
    if not _place_patch(canvas, selected_patch, padding=1):
        raise RuntimeError("failed to place selected sprite")

    distractor_color = rng.choice([c for c in Color.NOT_BLACK if c not in used_colors])
    used_colors.add(distractor_color)
    distractor_patch = _sprite_patch(distractor_side, distractor_area, distractor_color, rng)
    if not _place_patch(canvas, distractor_patch, padding=1):
        raise RuntimeError("failed to place distractor sprite")

    # Optional non-square clutter to increase variance.
    extra_count = rng.randrange(0, 3)
    for _ in range(extra_count):
        remaining = [c for c in Color.NOT_BLACK if c not in used_colors]
        if not remaining:
            break

        color = rng.choice(remaining)
        used_colors.add(color)

        h = rng.randint(2, 5)
        w = rng.randint(2, 5)
        if h == w:
            w = 5 if w < 5 else 2

        patch = random_sprite(
            m=h,
            n=w,
            density=rng.uniform(0.4, 0.8),
            color_palette=[color],
            connectivity=4,
        )
        if not _place_patch(canvas, patch, padding=1):
            continue

    return canvas


def generate_input() -> np.ndarray:
    for _ in range(1200):
        try:
            grid = _generate_once()
            out = main(grid)
            assert out.ndim == 2
            assert out.shape[0] <= 30 and out.shape[1] <= 30
            return grid
        except (AssertionError, ValueError, RuntimeError):
            continue

    raise RuntimeError("failed to generate valid input for puzzle 7")
