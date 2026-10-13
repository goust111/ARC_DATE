from __future__ import annotations

import random
from collections import deque

import numpy as np


PALETTE = (1, 2, 3, 4, 7)
GREY = 5
BLACK = 0
TEMPLATE_SPEC = {
    "shape_labels": 4,
    "mask_centers": ((5, 3), (5, 14), (12, 8)),
}


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _component_dicts(mask: np.ndarray) -> list[dict]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "coords": coords,
                    "area": len(coords),
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _crop_mask(coords: list[tuple[int, int]], bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def _nearest_in_row(grid: np.ndarray, r: int, c: int) -> tuple[int | None, int]:
    w = grid.shape[1]
    for dist in range(1, w + 1):
        left = c - dist
        if left >= 0 and grid[r, left] != GREY:
            return int(grid[r, left]), dist
        right = c + dist
        if right < w and grid[r, right] != GREY:
            return int(grid[r, right]), dist
    return None, 10**9


def _nearest_in_col(grid: np.ndarray, r: int, c: int) -> tuple[int | None, int]:
    h = grid.shape[0]
    for dist in range(1, h + 1):
        up = r - dist
        if up >= 0 and grid[up, c] != GREY:
            return int(grid[up, c]), dist
        down = r + dist
        if down < h and grid[down, c] != GREY:
            return int(grid[down, c]), dist
    return None, 10**9


def _repair_puzzle(puzzle_with_legend: np.ndarray) -> np.ndarray:
    g = np.array(puzzle_with_legend, dtype=int)
    puzzle = g[:, :-1].copy()
    source = puzzle.copy()
    grey_coords = np.argwhere(puzzle == GREY)
    for r, c in grey_coords:
        row_color, row_dist = _nearest_in_row(source, int(r), int(c))
        col_color, col_dist = _nearest_in_col(source, int(r), int(c))
        if row_color is None and col_color is None:
            raise ValueError("grey pixel cannot be inferred from its row or column")
        if col_color is None or row_dist <= col_dist:
            puzzle[int(r), int(c)] = int(row_color)
        else:
            puzzle[int(r), int(c)] = int(col_color)
    return puzzle


def _extract_main_objects(puzzle: np.ndarray) -> list[dict]:
    objs = []
    for color in sorted(int(v) for v in np.unique(puzzle)):
        if color in (BLACK, GREY):
            continue
        comps = _component_dicts(puzzle == color)
        if not comps:
            continue
        comp = max(comps, key=lambda d: (int(d["area"]), -int(d["bbox"][0]), -int(d["bbox"][1])))
        objs.append(
            {
                "color": color,
                "area": int(comp["area"]),
                "bbox": tuple(int(x) for x in comp["bbox"]),
                "mask": _crop_mask(comp["coords"], comp["bbox"]),
            }
        )
    objs.sort(key=lambda d: (-int(d["area"]), int(d["bbox"][0]), int(d["bbox"][1])))
    return objs


def _dihedral_variants(mask: np.ndarray) -> list[np.ndarray]:
    variants = []
    seen = set()
    for k in range(4):
        base = np.rot90(mask, k)
        for candidate in (base, np.fliplr(base)):
            key = (candidate.shape, candidate.tobytes())
            if key not in seen:
                seen.add(key)
                variants.append(candidate)
    return variants


def _equiv(mask_a: np.ndarray, mask_b: np.ndarray) -> bool:
    return any(v.shape == mask_b.shape and np.array_equal(v, mask_b) for v in _dihedral_variants(mask_a))


def _draw_table(objects: list[dict]) -> np.ndarray:
    n = len(objects)
    out = np.zeros((3 * n + 2, 3 * n + 2), dtype=int)
    for idx, obj in enumerate(objects):
        start = 3 + 3 * idx
        out[0:2, start : start + 2] = int(obj["color"])
        out[start : start + 2, 0:2] = int(obj["color"])
    for i, obj_i in enumerate(objects):
        for j, obj_j in enumerate(objects):
            if i == j:
                continue
            start_r = 3 + 3 * i
            start_c = 3 + 3 * j
            out[start_r : start_r + 2, start_c : start_c + 2] = 2 if _equiv(obj_i["mask"], obj_j["mask"]) else 5
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair cross-masked puzzle regions, recover the dominant same-color objects, and tabulate their dihedral equivalence.

    Input structure:
    - The last column is a legend column and is ignored for reconstruction and object extraction.
    - The remaining area is fully tiled by colored rectangular cells using colors from the non-grey palette.
    - Several grey 3x3 plus masks hide pixels inside those cells and may sever same-color components before repair.

    Rule:
    - Treat every grey pixel as unknown and refill it from the nearest non-grey pixel on the same row or column, preferring the row when the distances tie.
    - Ignore the legend column, then for each color keep only its largest 4-connected component and discard smaller fragments.
    - Tight-crop each kept component to a boolean mask, sort the components by descending area and then by bbox top-left, and compare every pair up to rotation and reflection.
    - Draw a black table whose top and left labels are 2x2 blocks in the component colors; each off-diagonal cell is a 2x2 red block for dihedral equivalence or a 2x2 grey block otherwise.

    Key insight:
    - The grey crosses hide articulation cells inside uniform rectangles, so repairing them changes which same-color regions are truly whole objects before the equivalence table is built.
    """
    puzzle = _repair_puzzle(np.array(grid, dtype=int))
    objects = _extract_main_objects(puzzle)
    return _draw_table(objects)


def _build_repaired_template(colors: tuple[int, ...]) -> np.ndarray:
    canvas = np.full((18, 18), int(colors[3]), dtype=int)

    # Object A: 3x3 hub with four arms.
    canvas[4:7, 2:5] = int(colors[0])
    canvas[1:4, 3] = int(colors[0])
    canvas[7:11, 3] = int(colors[0])
    canvas[5, 0:2] = int(colors[0])
    canvas[5, 5:7] = int(colors[0])

    # Object B: mirror of A.
    canvas[4:7, 13:16] = int(colors[1])
    canvas[1:4, 14] = int(colors[1])
    canvas[7:11, 14] = int(colors[1])
    canvas[5, 11:13] = int(colors[1])
    canvas[5, 16:18] = int(colors[1])

    # Object C: a different T-shaped hub.
    canvas[11:14, 7:10] = int(colors[2])
    canvas[8:11, 8] = int(colors[2])
    canvas[12, 4:7] = int(colors[2])
    canvas[12, 10:13] = int(colors[2])
    return canvas


def _apply_cross_mask(puzzle: np.ndarray, center_rc: tuple[int, int]) -> None:
    center_r, center_c = center_rc
    puzzle[center_r, center_c] = GREY
    puzzle[center_r - 1, center_c] = GREY
    puzzle[center_r + 1, center_c] = GREY
    puzzle[center_r, center_c - 1] = GREY
    puzzle[center_r, center_c + 1] = GREY


def _with_legend(puzzle: np.ndarray, colors: tuple[int, ...]) -> np.ndarray:
    h = puzzle.shape[0]
    out = np.zeros((h, puzzle.shape[1] + 1), dtype=int)
    out[:, :-1] = puzzle
    rows = np.linspace(2, h - 3, num=len(colors), dtype=int)
    for r, color in zip(rows, colors):
        out[int(r), -1] = int(color)
    return out


def _count_components_of_color(puzzle: np.ndarray, color: int) -> int:
    return len(_component_dicts(puzzle == color))


def _validate_instance(inp: np.ndarray, repaired: np.ndarray, colors: tuple[int, ...]) -> None:
    solved_repair = _repair_puzzle(inp)
    if not np.array_equal(solved_repair, repaired):
        raise ValueError("repair step does not reconstruct the intended puzzle")

    for color in colors[:3]:
        visible_count = _count_components_of_color(inp[:, :-1], color)
        repaired_count = _count_components_of_color(repaired, color)
        if visible_count <= repaired_count:
            raise ValueError("connector mask failed to change connectivity")

    objects = _extract_main_objects(repaired)
    equiv_pairs = 0
    non_equiv_pairs = 0
    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _equiv(objects[i]["mask"], objects[j]["mask"]):
                equiv_pairs += 1
            else:
                non_equiv_pairs += 1
    if equiv_pairs == 0 or non_equiv_pairs == 0:
        raise ValueError("instance does not contain both equivalent and non-equivalent pairs")

    out = main(inp)
    if out.ndim != 2 or max(out.shape) > 30:
        raise ValueError("output grid exceeds ARC size limit")


def generate_input() -> np.ndarray:
    seen = set()
    for _ in range(200):
        colors = tuple(random.sample(PALETTE, TEMPLATE_SPEC["shape_labels"]))
        key = colors
        if key in seen:
            continue
        seen.add(key)

        repaired = _build_repaired_template(colors)
        masked = repaired.copy()
        for center in TEMPLATE_SPEC["mask_centers"]:
            _apply_cross_mask(masked, center)
        inp = _with_legend(masked, colors)
        _validate_instance(inp, repaired, colors)
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
