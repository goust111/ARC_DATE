from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np


ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(ARC_UTILS_DIR))

from arc_utils import bbox_coords, colors_present, components_mask, get_bg


BLACK = 0
ORANGE = 7
CYAN = 8
MARKER_COLORS = (1, 2, 3, 4, 5, 6, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _draw_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, box)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _point_sides(r: int, c: int, box: tuple[int, int, int, int]) -> set[str]:
    r0, c0, r1, c1 = map(int, box)
    sides: set[str] = set()
    if r == r0:
        sides.add("top")
    if r == r1:
        sides.add("bottom")
    if c == c0:
        sides.add("left")
    if c == c1:
        sides.add("right")
    return sides


def _find_marker_groups(grid: np.ndarray) -> list[dict]:
    groups = []
    for color in colors_present(grid, exclude={BLACK, ORANGE, CYAN}):
        coords = np.argwhere(grid == int(color))
        if int(coords.shape[0]) != 3:
            continue
        box = bbox_coords(coords)
        r0, c0, r1, c1 = map(int, box)
        if r0 == r1 or c0 == c1:
            continue
        union_sides: set[str] = set()
        ok = True
        for r, c in coords:
            sides = _point_sides(int(r), int(c), box)
            if not sides:
                ok = False
                break
            union_sides |= sides
        if ok and len(union_sides) >= 3:
            groups.append(
                {
                    "color": int(color),
                    "coords": coords.astype(int),
                    "bbox": tuple(map(int, box)),
                }
            )
    groups.sort(key=lambda g: (g["bbox"][0], g["bbox"][1], g["color"]))
    return groups


def _inner_border_black_seeds(grid: np.ndarray, box: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    seeds: set[tuple[int, int]] = set()
    if r1 - r0 < 2 or c1 - c0 < 2:
        return []
    for c in range(c0 + 1, c1):
        if int(grid[r0 + 1, c]) == BLACK:
            seeds.add((r0 + 1, c))
        if int(grid[r1 - 1, c]) == BLACK:
            seeds.add((r1 - 1, c))
    for r in range(r0 + 1, r1):
        if int(grid[r, c0 + 1]) == BLACK:
            seeds.add((r, c0 + 1))
        if int(grid[r, c1 - 1]) == BLACK:
            seeds.add((r, c1 - 1))
    return sorted(seeds)


def _reachable_black_inside(grid: np.ndarray, box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    seen: set[tuple[int, int]] = set()
    q = deque(_inner_border_black_seeds(grid, box))
    while q:
        r, c = q.popleft()
        if (r, c) in seen or int(grid[r, c]) != BLACK:
            continue
        if not (r0 < r < r1 and c0 < c < c1):
            continue
        seen.add((r, c))
        for dr, dc in DIR4:
            nr, nc = r + dr, c + dc
            if (nr, nc) not in seen and r0 < nr < r1 and c0 < nc < c1 and int(grid[nr, nc]) == BLACK:
                q.append((nr, nc))
    return seen


def _transform_one_frame(grid: np.ndarray, box: tuple[int, int, int, int], color: int) -> None:
    _draw_frame(grid, box, color)
    for r, c in _reachable_black_inside(grid, box):
        grid[r, c] = ORANGE
    r0, c0, r1, c1 = map(int, box)
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior[interior == BLACK] = CYAN


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete each three-marker rectangle, flood orange inward from the frame's inner wall, and mark trapped black dents cyan.

    Input structure:
    - The grid is mostly black background with one or two independent marker groups.
    - Each marker group uses exactly three pixels of one non-orange color, and those pixels lie on the boundary of one latent rectangle.
    - Inside each latent rectangle sits an orange 4-connected blob that stays at least one cell away from the rectangle boundary and contains one or more black dents.

    Rule:
    - For each marker color, take the bbox of its three pixels and draw that full hollow rectangle in the same color.
    - Inside the rectangle, start from every black cell directly adjacent to the inner wall and flood-fill all 4-connected reachable black cells to orange, stopping at the frame and at existing orange cells.
    - Any black cells left inside the rectangle after that flood are unreachable holes inside the orange blob, so recolor them cyan.
    - Leave everything outside the rectangles unchanged.

    Key insight:
    - The frame created by the three markers turns the surrounding black area into the flood source; reachable black becomes orange, while only black cells sealed off by orange remain to be marked.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for group in _find_marker_groups(g):
        _transform_one_frame(out, group["bbox"], group["color"])
    return out


_TEMPLATES_BY_HOLES: dict[int, list[np.ndarray]] = {
    1: [
        np.array(
            [
                [1, 1, 1],
                [1, 0, 1],
                [1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [1, 0, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [0, 1, 1, 0],
            ],
            dtype=int,
        ),
    ],
    2: [
        np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 1, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [0, 1, 1, 0],
                [1, 0, 1, 1],
                [1, 1, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=int,
        ),
    ],
    3: [
        np.array(
            [
                [0, 1, 1, 1, 0],
                [1, 0, 1, 0, 1],
                [0, 1, 1, 1, 0],
                [0, 1, 0, 1, 0],
                [0, 0, 1, 0, 0],
            ],
            dtype=int,
        ),
    ],
}


_CORNER_MARKER_PATTERNS = (
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _rot_flip(rng: np.random.Generator, arr: np.ndarray) -> np.ndarray:
    out = np.rot90(arr, k=int(rng.integers(0, 4)))
    if bool(rng.integers(0, 2)):
        out = np.fliplr(out)
    return out.copy()


def _make_blob_template(rng: np.random.Generator) -> tuple[np.ndarray, int]:
    holes = int(rng.choice([1, 2, 3], p=[0.28, 0.36, 0.36]))
    template = _rot_flip(rng, _TEMPLATES_BY_HOLES[holes][int(rng.integers(0, len(_TEMPLATES_BY_HOLES[holes])))] )
    return template, holes


def _boxes_separate(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 2) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _marker_positions(
    rng: np.random.Generator, box: tuple[int, int, int, int]
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = map(int, box)
    pattern = _CORNER_MARKER_PATTERNS[int(rng.integers(0, len(_CORNER_MARKER_PATTERNS)))]
    corners = {
        (0, 0): (r0, c0),
        (0, 1): (r0, c1),
        (1, 0): (r1, c0),
        (1, 1): (r1, c1),
    }
    return [corners[key] for key in pattern]


def _bbox_mask(shape: tuple[int, int], box: tuple[int, int, int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    r0, c0, r1, c1 = map(int, box)
    mask[r0 : r1 + 1, c0 : c1 + 1] = True
    return mask


def _build_group_spec(rng: np.random.Generator, color: int) -> dict:
    template, holes = _make_blob_template(rng)
    ph, pw = map(int, template.shape)
    top_gap = int(rng.integers(1, 4))
    bottom_gap = int(rng.integers(1, 4))
    left_gap = int(rng.integers(1, 4))
    right_gap = int(rng.integers(1, 4))
    inner_h = ph + top_gap + bottom_gap
    inner_w = pw + left_gap + right_gap
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    return {
        "color": int(color),
        "template": template,
        "holes": int(holes),
        "outer_h": int(outer_h),
        "outer_w": int(outer_w),
        "patch_offset": (1 + top_gap, 1 + left_gap),
    }


def _paint_group(grid: np.ndarray, spec: dict, top: int, left: int, rng: np.random.Generator) -> dict:
    template = np.asarray(spec["template"], dtype=int)
    outer_h = int(spec["outer_h"])
    outer_w = int(spec["outer_w"])
    box = (int(top), int(left), int(top + outer_h - 1), int(left + outer_w - 1))
    patch_r0 = int(top + spec["patch_offset"][0])
    patch_c0 = int(left + spec["patch_offset"][1])

    orange_cells = []
    for r, c in np.argwhere(template == 1):
        rr = int(patch_r0 + r)
        cc = int(patch_c0 + c)
        grid[rr, cc] = ORANGE
        orange_cells.append((rr, cc))

    markers = _marker_positions(rng, box)
    for r, c in markers:
        grid[r, c] = int(spec["color"])

    return {
        "color": int(spec["color"]),
        "bbox": box,
        "markers": markers,
        "orange_cells": orange_cells,
        "holes": int(spec["holes"]),
        "bbox_mask": _bbox_mask(grid.shape, box),
    }


def _connected_black_reachable(grid: np.ndarray, box: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    framed = grid.copy()
    _draw_frame(framed, box, 9)
    return _reachable_black_inside(framed, box)


def _validate_instance(inp: np.ndarray, metas: list[dict]) -> bool:
    if inp.ndim != 2 or max(inp.shape) > 30:
        return False
    groups = _find_marker_groups(inp)
    if len(groups) != len(metas):
        return False
    if np.any(inp == CYAN):
        return False

    orange_components = components_mask(inp == ORANGE, connectivity=4)
    if len(orange_components) != len(metas):
        return False

    used_boxes: list[tuple[int, int, int, int]] = []
    for meta in metas:
        box = meta["bbox"]
        r0, c0, r1, c1 = map(int, box)
        if any(not _boxes_separate(box, other, gap=0) for other in used_boxes):
            return False
        used_boxes.append(box)

        if int((inp == meta["color"]).sum()) != 3:
            return False
        if not np.all(inp[r0 + 1 : r1, c0 + 1 : c1] != meta["color"]):
            return False

        interior = inp[r0 + 1 : r1, c0 + 1 : c1]
        if not np.any(interior == ORANGE):
            return False

        coords = np.argwhere(inp == ORANGE)
        in_box = [
            (int(r), int(c))
            for r, c in coords
            if r0 < int(r) < r1 and c0 < int(c) < c1
        ]
        if len(in_box) != len(meta["orange_cells"]):
            return False

        reachable = _connected_black_reachable(inp, box)
        if not reachable:
            return False

        framed = inp.copy()
        _draw_frame(framed, box, meta["color"])
        after = framed.copy()
        for r, c in reachable:
            after[r, c] = ORANGE
        dents = np.argwhere(after[r0 + 1 : r1, c0 + 1 : c1] == BLACK)
        if int(dents.shape[0]) != int(meta["holes"]):
            return False
        for rr, cc in dents:
            gr = int(r0 + 1 + rr)
            gc = int(c0 + 1 + cc)
            if sum(int(after[gr + dr, gc + dc] == ORANGE) for dr, dc in DIR4) != 4:
                return False

    out = main(inp)
    if out.shape != inp.shape:
        return False

    all_boxes_mask = np.zeros(inp.shape, dtype=bool)
    for meta in metas:
        all_boxes_mask |= meta["bbox_mask"]
        r0, c0, r1, c1 = map(int, meta["bbox"])
        if not np.all(out[r0, c0 : c1 + 1] == meta["color"]):
            return False
        if not np.all(out[r1, c0 : c1 + 1] == meta["color"]):
            return False
        if not np.all(out[r0 : r1 + 1, c0] == meta["color"]):
            return False
        if not np.all(out[r0 : r1 + 1, c1] == meta["color"]):
            return False

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        if np.any(interior == BLACK):
            return False
        if int((interior == CYAN).sum()) != int(meta["holes"]):
            return False

    if not np.array_equal(out[~all_boxes_mask], inp[~all_boxes_mask]):
        return False

    return True


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    n_groups = int(rng.choice([1, 2], p=[0.4, 0.6]))
    colors = rng.choice(np.array(MARKER_COLORS, dtype=int), size=n_groups, replace=False)
    specs = [_build_group_spec(rng, int(color)) for color in colors]

    if n_groups == 1:
        H = int(rng.integers(14, 23))
        W = int(rng.integers(14, 23))
    else:
        H = int(rng.integers(18, 29))
        W = int(rng.integers(18, 29))

    if any(spec["outer_h"] > H - 2 or spec["outer_w"] > W - 2 for spec in specs):
        return None

    grid = np.zeros((H, W), dtype=int)
    metas: list[dict] = []
    boxes: list[tuple[int, int, int, int]] = []

    for spec in sorted(specs, key=lambda s: -(s["outer_h"] * s["outer_w"])):
        placed = False
        for _ in range(120):
            top = int(rng.integers(0, H - spec["outer_h"] + 1))
            left = int(rng.integers(0, W - spec["outer_w"] + 1))
            box = (top, left, top + int(spec["outer_h"]) - 1, left + int(spec["outer_w"]) - 1)
            if all(_boxes_separate(box, other, gap=2) for other in boxes):
                boxes.append(box)
                meta = _paint_group(grid, spec, top, left, rng)
                metas.append(meta)
                placed = True
                break
        if not placed:
            return None

    if not _validate_instance(grid, metas):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
