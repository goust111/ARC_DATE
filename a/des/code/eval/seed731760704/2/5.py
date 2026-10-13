import random
from collections import deque

import numpy as np


SIDES = ("top", "right", "bottom", "left")
FRAME_COLOR = 5
GEN_RETRIES = 400
INTERIOR_CHOICES = (5, 7, 9)


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((int(cr), int(cc)))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _target_info(grid: np.ndarray, bg: int) -> dict:
    h, w = map(int, grid.shape)
    candidates = []

    for comp in _components(grid != bg):
        r0, c0, r1, c1 = _bbox(comp)
        box_h = int(r1 - r0 + 1)
        box_w = int(c1 - c0 + 1)
        if r0 == 0 or c0 == 0 or r1 == h - 1 or c1 == w - 1:
            continue
        if box_h < 5 or box_w < 5 or box_h % 2 == 0 or box_w % 2 == 0:
            continue

        border_cells = {
            (r0, c) for c in range(c0, c1 + 1)
        } | {
            (r1, c) for c in range(c0, c1 + 1)
        } | {
            (r, c0) for r in range(r0, r1 + 1)
        } | {
            (r, c1) for r in range(r0, r1 + 1)
        }
        comp_set = set(comp)
        if comp_set != border_cells:
            continue
        if not np.all(grid[r0 + 1 : r1, c0 + 1 : c1] == bg):
            continue

        midpoints = {
            "top": (r0, (c0 + c1) // 2),
            "right": ((r0 + r1) // 2, c1),
            "bottom": (r1, (c0 + c1) // 2),
            "left": ((r0 + r1) // 2, c0),
        }
        marker_colors = {side: int(grid[r, c]) for side, (r, c) in midpoints.items()}
        if len(set(marker_colors.values())) != 4:
            continue

        non_marker_border = [int(grid[r, c]) for r, c in border_cells if (r, c) not in midpoints.values()]
        if not non_marker_border or len(set(non_marker_border)) != 1:
            continue
        frame_color = int(non_marker_border[0])
        if frame_color in marker_colors.values():
            continue

        candidates.append(
            {
                "bbox": (r0, c0, r1, c1),
                "frame_color": frame_color,
                "marker_colors": marker_colors,
                "interior_shape": (box_h - 2, box_w - 2),
            }
        )

    assert len(candidates) == 1, "expected exactly one framed target region"
    return candidates[0]


def _source_patches(grid: np.ndarray, bg: int, target_bbox: tuple[int, int, int, int], shape: tuple[int, int]) -> list[dict]:
    r0, c0, r1, c1 = target_bbox
    sources = []
    for comp in _components(grid != bg):
        box = _bbox(comp)
        if box == target_bbox:
            continue
        sr0, sc0, sr1, sc1 = box
        if (sr1 - sr0 + 1, sc1 - sc0 + 1) != shape:
            continue
        if not (sr1 < r0 or sr0 > r1 or sc1 < c0 or sc0 > c1):
            continue
        patch = grid[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()
        colors = {int(v) for v in patch.ravel() if int(v) != bg}
        sources.append({"bbox": box, "patch": patch, "coords": comp, "colors": colors})

    assert len(sources) == 4, "expected exactly four source patterns"
    return sources


def _compose(patches_by_side: dict[str, np.ndarray], bg: int, order: tuple[str, ...] = SIDES) -> np.ndarray:
    sample = next(iter(patches_by_side.values()))
    out = np.full(sample.shape, int(bg), dtype=int)
    for side in order:
        patch = patches_by_side[side]
        mask = patch != bg
        out[mask] = patch[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each source pattern to the target side by its shared marker color, then
    overlay the four patterns inside the framed target in top-right-bottom-left order.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _mode_color(g)

    target = _target_info(g, bg)
    marker_to_side = {color: side for side, color in target["marker_colors"].items()}
    sources = _source_patches(g, bg, target["bbox"], target["interior_shape"])

    patches_by_side: dict[str, np.ndarray] = {}
    for source in sources:
        shared = source["colors"] & set(marker_to_side)
        assert len(shared) == 1, "each source must share exactly one marker color"
        marker_color = int(next(iter(shared)))
        side = marker_to_side[marker_color]
        assert side not in patches_by_side, "marker colors must identify sources uniquely"
        patches_by_side[side] = source["patch"]

    assert set(patches_by_side) == set(SIDES)
    interior = _compose(patches_by_side, bg, SIDES)

    out = g.copy()
    for source in sources:
        for r, c in source["coords"]:
            out[r, c] = bg

    r0, c0, r1, c1 = target["bbox"]
    out[r0 + 1 : r1, c0 + 1 : c1] = interior
    return out


def _make_patch(
    height: int,
    width: int,
    side: str,
    marker_color: int,
    body_color: int,
    bg: int,
    rng: random.Random,
) -> np.ndarray:
    patch = np.full((height, width), int(bg), dtype=int)
    center_r = rng.randint(1, height - 2)
    center_c = rng.randint(1, width - 2)

    patch[:, center_c] = body_color
    patch[center_r, :] = body_color

    if side in ("top", "bottom"):
        extra_r_choices = [r for r in range(height) if r != center_r]
        if extra_r_choices:
            extra_r = rng.choice(extra_r_choices)
            start = rng.randint(0, center_c)
            end = rng.randint(center_c, width - 1)
            patch[extra_r, start : end + 1] = body_color
    else:
        extra_c_choices = [c for c in range(width) if c != center_c]
        if extra_c_choices:
            extra_c = rng.choice(extra_c_choices)
            start = rng.randint(0, center_r)
            end = rng.randint(center_r, height - 1)
            patch[start : end + 1, extra_c] = body_color

    thick = 1 if min(height, width) < 7 or rng.random() < 0.5 else 3
    if side == "top":
        seg = rng.randint(2, min(4, height))
        c0 = max(0, center_c - thick // 2)
        c1 = min(width, c0 + thick)
        c0 = c1 - thick
        patch[:seg, c0:c1] = marker_color
    elif side == "right":
        seg = rng.randint(2, min(4, width))
        r0 = max(0, center_r - thick // 2)
        r1 = min(height, r0 + thick)
        r0 = r1 - thick
        patch[r0:r1, width - seg :] = marker_color
    elif side == "bottom":
        seg = rng.randint(2, min(4, height))
        c0 = max(0, center_c - thick // 2)
        c1 = min(width, c0 + thick)
        c0 = c1 - thick
        patch[height - seg :, c0:c1] = marker_color
    else:
        seg = rng.randint(2, min(4, width))
        r0 = max(0, center_r - thick // 2)
        r1 = min(height, r0 + thick)
        r0 = r1 - thick
        patch[r0:r1, :seg] = marker_color

    return patch


def _placement_slots(
    grid_h: int,
    grid_w: int,
    target_bbox: tuple[int, int, int, int],
    shape: tuple[int, int],
    rng: random.Random,
) -> dict[str, tuple[int, int]]:
    r0, c0, r1, c1 = target_bbox
    box_h, box_w = shape
    return {
        "top_band": (rng.randint(1, r0 - box_h - 1), rng.randint(1, grid_w - box_w - 1)),
        "bottom_band": (rng.randint(r1 + 2, grid_h - box_h - 1), rng.randint(1, grid_w - box_w - 1)),
        "left_band": (rng.randint(r0, r0 + 2), rng.randint(1, c0 - box_w - 1)),
        "right_band": (rng.randint(r0, r0 + 2), rng.randint(c1 + 2, grid_w - box_w - 1)),
    }


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    bg = rng.choice([color for color in range(10) if color != FRAME_COLOR])
    interior_h = rng.choice(INTERIOR_CHOICES)
    interior_w = rng.choice(INTERIOR_CHOICES)
    frame_h = interior_h + 2
    frame_w = interior_w + 2

    grid_h = rng.randint(3 * interior_h + 6, 3 * interior_h + 12)
    grid_w = rng.randint(3 * interior_w + 6, 3 * interior_w + 12)

    target_r0 = rng.randint(interior_h + 2, grid_h - 2 * interior_h - 4)
    target_c0 = rng.randint(interior_w + 2, grid_w - 2 * interior_w - 4)
    target_bbox = (target_r0, target_c0, target_r0 + frame_h - 1, target_c0 + frame_w - 1)

    palette = [color for color in range(10) if color not in (bg, FRAME_COLOR)]
    rng.shuffle(palette)
    marker_colors = palette[:4]
    body_colors = palette[4:8]
    marker_by_side = dict(zip(SIDES, marker_colors))
    body_by_side = dict(zip(SIDES, body_colors))

    grid = np.full((grid_h, grid_w), int(bg), dtype=int)
    r0, c0, r1, c1 = target_bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    midpoints = {
        "top": (r0, (c0 + c1) // 2),
        "right": ((r0 + r1) // 2, c1),
        "bottom": (r1, (c0 + c1) // 2),
        "left": ((r0 + r1) // 2, c0),
    }
    for side, color in marker_by_side.items():
        rr, cc = midpoints[side]
        grid[rr, cc] = color

    patches_by_side = {
        side: _make_patch(interior_h, interior_w, side, marker_by_side[side], body_by_side[side], bg, rng)
        for side in SIDES
    }

    placements = _placement_slots(grid_h, grid_w, target_bbox, (interior_h, interior_w), rng)
    shuffled_sides = list(SIDES)
    rng.shuffle(shuffled_sides)
    for slot, side in zip(placements, shuffled_sides):
        pr, pc = placements[slot]
        patch = patches_by_side[side]
        mask = patch != bg
        area = grid[pr : pr + interior_h, pc : pc + interior_w]
        area[mask] = patch[mask]

    correct = _compose(patches_by_side, bg, SIDES)
    swapped = _compose(patches_by_side, bg, ("right", "top", "bottom", "left"))
    reversed_order = _compose(patches_by_side, bg, tuple(reversed(SIDES)))
    if np.array_equal(correct, swapped) or np.array_equal(correct, reversed_order):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for task 5")
