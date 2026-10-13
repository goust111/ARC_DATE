from __future__ import annotations

from pathlib import Path
import random
from typing import Optional

import numpy as np

try:
    from arc_utils import (
        Color,
        bbox,
        crop_bbox,
        find_components,
        get_bg,
        paste,
        scale_sprite,
    )
except ModuleNotFoundError:
    import importlib.util

    _paths = [
        Path("/home/lixinyu/snap/new/solvers/arc_utils.py"),
        Path(__file__).resolve().parent / "arc_utils.py",
    ]
    _spec = None
    _arc_utils = None
    for _path in _paths:
        if _path.exists():
            _spec = importlib.util.spec_from_file_location("arc_utils", _path)
            if _spec is not None and _spec.loader is not None:
                _arc_utils = importlib.util.module_from_spec(_spec)
                _spec.loader.exec_module(_arc_utils)
                break
    if _arc_utils is None:
        raise ModuleNotFoundError("Could not load arc_utils from known paths")

    Color = _arc_utils.Color
    bbox = _arc_utils.bbox
    crop_bbox = _arc_utils.crop_bbox
    find_components = _arc_utils.find_components
    get_bg = _arc_utils.get_bg
    paste = _arc_utils.paste
    scale_sprite = _arc_utils.scale_sprite


HOLE_COLOR = Color.GRAY
BACKGROUND = Color.BLACK
CANVAS_MIN = 18
CANVAS_MAX = 30
MAX_GENERATION_ATTEMPTS = 500


def _component_with_holes(grid: np.ndarray, bg: int) -> np.ndarray:
    """Return the most likely template component: a non-bg component with hole color near a corner."""
    comps = find_components(grid, bg=bg, monochromatic=False, return_format="dict", connectivity=4)
    if not comps:
        raise ValueError("no non-background components found")

    h, w = map(int, grid.shape)
    candidates = []
    for comp in comps:
        b = comp["bbox"]
        b0, b1, b2, b3 = b
        sub = crop_bbox(grid, b)
        hole_mask = sub == HOLE_COLOR
        if not hole_mask.any():
            continue
        touches_outer = int(b0 == 0 or b1 == 0 or b2 == h - 1 or b3 == w - 1)
        area = sub.size
        hole_area = int(hole_mask.sum())
        candidates.append((touches_outer, area, -hole_area, sub))

    if not candidates:
        raise ValueError("no component with hole color found")

    _, _, _, template = min(candidates, key=lambda item: (item[0], item[1], item[2]))
    return template.astype(int)


def _template_match_at(grid: np.ndarray, template: np.ndarray, r0: int, c0: int, k: int) -> bool:
    h, w = map(int, template.shape)
    for r in range(h):
        gr0 = r0 + r * k
        for c in range(w):
            gc0 = c0 + c * k
            block = grid[gr0 : gr0 + k, gc0 : gc0 + k]
            if block.shape != (k, k):
                return False
            fill = int(block[0, 0])
            if np.any(block != fill):
                return False
            expected = int(template[r, c])
            if expected != HOLE_COLOR and fill != expected:
                return False
    return True


def _find_matches(grid: np.ndarray, template: np.ndarray) -> list[tuple[int, int, int]]:
    h, w = map(int, template.shape)
    H, W = grid.shape
    max_k = min(H // h, W // w)
    matches: list[tuple[int, int, int]] = []
    for k in range(2, max_k + 1):
        big_h = h * k
        big_w = w * k
        for r0 in range(H - big_h + 1):
            for c0 in range(W - big_w + 1):
                if _template_match_at(grid, template, r0, c0, k):
                    matches.append((k, r0, c0))
    return matches


def _fill_holes_from_match(grid: np.ndarray, template: np.ndarray, k: int, r0: int, c0: int) -> np.ndarray:
    out = template.copy()
    hole_coords = np.argwhere(template == HOLE_COLOR)
    for r, c in hole_coords:
        out[r, c] = int(grid[r0 + r * k, c0 + c * k])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a corner template by reading colors from its hidden complete enlarged replica.

    Input structure:
    - A small multicolor template with gray hole cells appears near a corner.
    - Somewhere in the scene, one complete replica of that template exists, enlarged by an
      integer factor k (2 or more), with no holes.
    - Extra unrelated colorful objects may appear elsewhere.

    Rule:
    - Identify the small template by its corner placement and gray-hole evidence.
    - For every integer scale k and every placement, test whether each non-gray template cell maps to a uniform
      k×k block of the same color.
    - Select the unique valid replica placement, then fill every gray hole in the template using the color of
      its corresponding block in the replica.
    - Return only the completed small template.

    Key insight:
    - The full template information is stored in the enlarged copy; matching by only non-gray witness cells avoids
      distractors and determines the exact scale and position.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    template = _component_with_holes(g, bg)

    matches = _find_matches(g, template)
    if not matches:
        raise AssertionError("no valid scaled replica found")

    # Prefer the largest scale; ties are broken left-to-right for determinism.
    k, r0, c0 = sorted(matches, key=lambda item: (-item[0], item[1], item[2]))[0]
    return _fill_holes_from_match(g, template, k, r0, c0)


def _random_template_core() -> np.ndarray:
    h = random.randint(2, 6)
    w = random.randint(2, 6)
    colors = random.sample([c for c in Color.NOT_BLACK if c != HOLE_COLOR], random.randint(2, 4))
    template = np.zeros((h, w), dtype=int)
    if random.random() < 0.5:
        for r in range(h):
            for c in range(w):
                if random.random() < 0.15:
                    template[r, c] = random.choice(colors)
                else:
                    template[r, c] = colors[(r + c) % len(colors)]
    else:
        template[:, :] = np.random.choice(colors, size=(h, w))
    return template.astype(int)


def _add_holes(template: np.ndarray) -> np.ndarray:
    h, w = map(int, template.shape)
    area = h * w
    out = template.copy()
    hole_count = random.randint(1, max(1, area // 2))
    if hole_count >= area:
        hole_count = area - 1
    holes = random.sample(range(area), hole_count)
    for idx in holes:
        out[idx // w, idx % w] = HOLE_COLOR
    return out


def _pick_corner_positions(h: int, w: int, H: int, W: int) -> list[tuple[int, int]]:
    corners = []
    if h <= H and w <= W:
        corners.append((0, 0))
        corners.append((0, W - w))
        corners.append((H - h, 0))
        corners.append((H - h, W - w))
    return corners


def _free_to_place(occupied: np.ndarray, r0: int, c0: int, h: int, c: int, w: int) -> bool:
    return not occupied[r0 : r0 + h, c0 : c0 + w].any()


def _add_interference(
    grid: np.ndarray,
    occupied: np.ndarray,
    palette: list[int],
) -> None:
    H, W = map(int, grid.shape)
    for _ in range(random.randint(0, 3)):
        if H == 0 or W == 0:
            return
        for _ in range(120):
            color = random.choice(palette)
            obj_h = random.randint(1, 5)
            obj_w = random.randint(1, 5)

            # line-like object branch
            if random.random() < 0.4:
                if random.random() < 0.5:
                    obj_h = 1
                else:
                    obj_w = 1

            r0 = random.randint(0, max(0, H - obj_h))
            c0 = random.randint(0, max(0, W - obj_w))

            if not _free_to_place(occupied, r0, c0, obj_h, c0, obj_w):
                continue
            if random.random() < 0.65:
                grid[r0 : r0 + obj_h, c0 : c0 + obj_w] = color
            else:
                # sparse noise object
                for rr in range(r0, r0 + obj_h):
                    for cc in range(c0, c0 + obj_w):
                        if random.random() < 0.55:
                            grid[rr, cc] = color

            occupied[r0 : r0 + obj_h, c0 : c0 + obj_w] = True
            break


def generate_input() -> np.ndarray:
    """
    Generate a random input with one corner template containing gray holes and one enlarged replica elsewhere.
    """
    for _ in range(MAX_GENERATION_ATTEMPTS):
        H = random.randint(CANVAS_MIN, CANVAS_MAX)
        W = random.randint(CANVAS_MIN, CANVAS_MAX)
        grid = np.full((H, W), BACKGROUND, dtype=int)
        occupied = np.zeros((H, W), dtype=bool)

        core = _random_template_core()
        template = _add_holes(core)
        th, tw = map(int, template.shape)

        corners = _pick_corner_positions(th, tw, H, W)
        if not corners:
            continue

        tr, tc = random.choice(corners)
        paste(grid, template, tr, tc, transparent=BACKGROUND)
        occupied[tr : tr + th, tc : tc + tw] = True

        palette = [c for c in Color.NOT_BLACK if c not in (BACKGROUND, HOLE_COLOR)]
        scale = random.randint(2, 4)
        replica = scale_sprite(core, scale)
        rh, rw = map(int, replica.shape)
        if rh > H or rw > W:
            continue

        copy_placed = False
        for _ in range(180):
            # keep copies away from the corner template to avoid ambiguous touching cases
            if random.random() < 0.7:
                cr = random.randint(0, H - rh)
                cc = random.randint(0, W - rw)
            else:
                cr = random.randint(max(0, H // 3 - rh), min(H - rh, H * 2 // 3))
                cc = random.randint(max(0, W // 3 - rw), min(W - rw, W * 2 // 3))
            if cr < 0 or cc < 0 or cr + rh > H or cc + rw > W:
                continue
            if not _free_to_place(occupied, cr, cc, rh, cc, rw):
                continue
            paste(grid, replica, cr, cc, transparent=None)
            occupied[cr : cr + rh, cc : cc + rw] = True
            copy_placed = True
            break
        if not copy_placed:
            continue

        # Add unrelated distractor objects.
        _add_interference(grid, occupied, palette)

        try:
            matches = _find_matches(grid, template)
        except Exception:
            continue
        if len(matches) != 1:
            continue

        expected = _fill_holes_from_match(grid, template, *matches[0])
        try:
            if not np.array_equal(main(grid), expected):
                continue
        except Exception:
            continue

        # Ensure generated canvas is valid ARC size.
        if max(grid.shape) > 30:
            continue

        return grid
    raise RuntimeError("failed to generate a valid 5.py input")
