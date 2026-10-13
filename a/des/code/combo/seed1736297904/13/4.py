from __future__ import annotations

from collections import deque
import random

import numpy as np


DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
BG = 0
GREY = 5
OBJECT_COLORS = tuple(color for color in range(1, 10) if color != GREY)
MAX_SHAPE_ATTEMPTS = 400
MAX_CASE_ATTEMPTS = 400


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((int(nr), int(nc)))
            comps.append(np.array(coords, dtype=int))
    return comps


def _seed_component(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    sr, sc = seed
    assert 0 <= sr < h and 0 <= sc < w and mask[sr, sc]

    seen = np.zeros((h, w), dtype=bool)
    q = deque([(int(sr), int(sc))])
    seen[sr, sc] = True
    coords: list[tuple[int, int]] = []
    while q:
        rr, cc = q.popleft()
        coords.append((rr, cc))
        for dr, dc in DIR4:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((int(nr), int(nc)))
    return np.array(coords, dtype=int)


def _interior_false_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    holes: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if mask[r, c] or seen[r, c]:
                continue
            q = deque([(int(r), int(c))])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_boundary = r in (0, h - 1) or c in (0, w - 1)
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        if nr in (0, h - 1) or nc in (0, w - 1):
                            touches_boundary = True
                        q.append((int(nr), int(nc)))
            if not touches_boundary:
                holes.append(np.array(coords, dtype=int))
    return holes


def _leaf_cells(coords: np.ndarray) -> list[tuple[int, int]]:
    coord_set = {tuple(map(int, cell)) for cell in coords}
    leaves = []
    for cell in coord_set:
        r, c = cell
        degree = sum((r + dr, c + dc) in coord_set for dr, dc in DIR4)
        if degree == 1:
            leaves.append(cell)
    return leaves


def _find_stamp_tasks(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    visited = np.zeros((h, w), dtype=bool)
    tasks: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            coords = _seed_component(g == color, (int(r), int(c)))
            visited[coords[:, 0], coords[:, 1]] = True

            r0, c0, r1, c1 = _bbox(coords)
            patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            patch[coords[:, 0] - r0, coords[:, 1] - c0] = True
            holes = _interior_false_components(patch)
            if len(holes) != 1:
                continue

            hole = holes[0]
            hr0, hc0, hr1, hc1 = _bbox(hole)
            hole_mask = np.zeros((hr1 - hr0 + 1, hc1 - hc0 + 1), dtype=bool)
            hole_mask[hole[:, 0] - hr0, hole[:, 1] - hc0] = True

            leaves = _leaf_cells(coords)
            assert len(leaves) == 1, "expected each stamp object to have one guide-line tip"
            tip_r, tip_c = leaves[0]
            neighbors = [
                (tip_r + dr, tip_c + dc)
                for dr, dc in DIR4
                if 0 <= tip_r + dr < h and 0 <= tip_c + dc < w and int(g[tip_r + dr, tip_c + dc]) == color
            ]
            assert len(neighbors) == 1
            base_r, base_c = neighbors[0]
            dr, dc = tip_r - base_r, tip_c - base_c
            seed_r, seed_c = tip_r + dr, tip_c + dc
            assert 0 <= seed_r < h and 0 <= seed_c < w, "guide line must point to an in-bounds canvas"

            canvas_color = int(g[seed_r, seed_c])
            assert canvas_color not in (bg, color), "guide-line tip must touch a distinct non-background canvas"
            canvas_coords = _seed_component(g == canvas_color, (seed_r, seed_c))
            cr0, cc0, cr1, cc1 = _bbox(canvas_coords)
            canvas_h = cr1 - cr0 + 1
            canvas_w = cc1 - cc0 + 1
            assert len(canvas_coords) == canvas_h * canvas_w, "canvas must be a solid rectangle"
            assert (canvas_h, canvas_w) == hole_mask.shape, "canvas size must match hole bbox size"

            tasks.append(
                {
                    "color": color,
                    "hole_mask": hole_mask,
                    "canvas_bbox": (cr0, cc0, cr1, cc1),
                }
            )

    return tasks


def _solve_with_bg(grid: np.ndarray, bg: int) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    tasks = _find_stamp_tasks(g, bg)
    assert tasks, "expected at least one stamp object"

    for task in tasks:
        cr0, cc0, cr1, cc1 = task["canvas_bbox"]
        hole_mask = task["hole_mask"]
        color = int(task["color"])
        region = out[cr0 : cr1 + 1, cc0 : cc1 + 1]
        region[hole_mask] = color

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Each colored object encodes a stamp via its enclosed hole; copy that hole shape
    onto the same-colored guide line's grey canvas.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _most_common_color(g)
    return _solve_with_bg(g, bg)


def _random_hole_mask(height: int, width: int) -> np.ndarray:
    if height == 1 or width == 1:
        return np.ones((height, width), dtype=bool)

    min_area = max(2, min(height * width, height + width - 1))
    for _ in range(MAX_SHAPE_ATTEMPTS):
        target_area = random.randint(min_area, height * width)
        mask = np.zeros((height, width), dtype=bool)
        start = (random.randrange(height), random.randrange(width))
        mask[start] = True

        while int(mask.sum()) < target_area:
            frontier: list[tuple[int, int]] = []
            filled = np.argwhere(mask)
            for r, c in filled:
                for dr, dc in DIR4:
                    nr, nc = int(r + dr), int(c + dc)
                    if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                        frontier.append((nr, nc))
            if not frontier:
                break
            nr, nc = random.choice(frontier)
            mask[nr, nc] = True

        if not (mask[0, :].any() and mask[-1, :].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask

    return np.ones((height, width), dtype=bool)


def _make_item_patch(color: int) -> tuple[np.ndarray, np.ndarray]:
    for _ in range(MAX_SHAPE_ATTEMPTS):
        hole_h = random.randint(1, 4)
        hole_w = random.randint(1, 4)
        hole_mask = _random_hole_mask(hole_h, hole_w)

        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)
        obj_h = hole_h + top_margin + bottom_margin
        obj_w = hole_w + left_margin + right_margin

        object_cells = {(r, c) for r in range(obj_h) for c in range(obj_w)}
        for hr, hc in np.argwhere(hole_mask):
            object_cells.discard((int(top_margin + hr), int(left_margin + hc)))

        direction = random.choice(("up", "down", "left", "right"))
        line_len = random.randint(1, 5)
        line_cells: list[tuple[int, int]] = []
        canvas_cells: list[tuple[int, int]] = []

        if direction in ("left", "right"):
            attach_r = random.randint(1, obj_h - 2)
            anchor_r = random.randint(0, hole_h - 1)
            if direction == "right":
                line_cells = [(attach_r, obj_w + d) for d in range(line_len)]
                canvas_top = attach_r - anchor_r
                canvas_left = obj_w + line_len
            else:
                line_cells = [(attach_r, -1 - d) for d in range(line_len)]
                canvas_top = attach_r - anchor_r
                canvas_left = -line_len - hole_w
        else:
            attach_c = random.randint(1, obj_w - 2)
            anchor_c = random.randint(0, hole_w - 1)
            if direction == "down":
                line_cells = [(obj_h + d, attach_c) for d in range(line_len)]
                canvas_top = obj_h + line_len
                canvas_left = attach_c - anchor_c
            else:
                line_cells = [(-1 - d, attach_c) for d in range(line_len)]
                canvas_top = -line_len - hole_h
                canvas_left = attach_c - anchor_c

        for r in range(hole_h):
            for c in range(hole_w):
                canvas_cells.append((int(canvas_top + r), int(canvas_left + c)))

        all_cells = list(object_cells) + line_cells + canvas_cells
        min_r = min(r for r, _ in all_cells)
        min_c = min(c for _, c in all_cells)
        max_r = max(r for r, _ in all_cells)
        max_c = max(c for _, c in all_cells)
        patch_h = max_r - min_r + 1
        patch_w = max_c - min_c + 1

        inp = np.full((patch_h, patch_w), BG, dtype=int)
        for r, c in object_cells:
            inp[r - min_r, c - min_c] = color
        for r, c in line_cells:
            inp[r - min_r, c - min_c] = color
        for r, c in canvas_cells:
            rr, cc = r - min_r, c - min_c
            assert inp[rr, cc] == BG
            inp[rr, cc] = GREY

        canvas_coords = np.array([(r - min_r, c - min_c) for r, c in canvas_cells], dtype=int)
        cr0, cc0, cr1, cc1 = _bbox(canvas_coords)
        out = inp.copy()
        region = out[cr0 : cr1 + 1, cc0 : cc1 + 1]
        region[hole_mask] = color

        color_components = _components(inp == color)
        if len(color_components) != 1:
            continue
        if len(_leaf_cells(color_components[0])) != 1:
            continue

        try:
            solved = _solve_with_bg(inp, BG)
        except AssertionError:
            continue
        if not np.array_equal(solved, out):
            continue
        return inp, out

    raise RuntimeError("failed to build a valid stamp object")


def _dilate8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        out[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True
    return out


def _paste_non_bg(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    region = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    mask = patch != BG
    region[mask] = patch[mask]


def _background_is_dominant(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    return int(counts[BG]) > int(counts[1:].max())


def generate_input() -> np.ndarray:
    for _ in range(MAX_CASE_ATTEMPTS):
        object_count = random.randint(1, 3)
        colors = random.sample(list(OBJECT_COLORS), object_count)
        patches = [_make_item_patch(color) for color in colors]
        patches.sort(key=lambda pair: -int(np.count_nonzero(pair[0] != BG)))

        grid_h = random.randint(max(18, 8 + 4 * object_count), 30)
        grid_w = random.randint(max(18, 8 + 4 * object_count), 30)
        grid = np.full((grid_h, grid_w), BG, dtype=int)
        expected = grid.copy()
        occupied = np.zeros((grid_h, grid_w), dtype=bool)

        placed_all = True
        for inp_patch, out_patch in patches:
            ph, pw = inp_patch.shape
            if ph + 2 > grid_h or pw + 2 > grid_w:
                placed_all = False
                break

            blocked = _dilate8(occupied)
            occ_mask = inp_patch != BG
            positions: list[tuple[int, int]] = []
            for top in range(1, grid_h - ph):
                for left in range(1, grid_w - pw):
                    if np.any(blocked[top : top + ph, left : left + pw] & occ_mask):
                        continue
                    positions.append((top, left))

            if not positions:
                placed_all = False
                break

            top, left = random.choice(positions)
            _paste_non_bg(grid, inp_patch, top, left)
            _paste_non_bg(expected, out_patch, top, left)
            occupied[top : top + ph, left : left + pw] |= occ_mask

        if not placed_all:
            continue
        if not _background_is_dominant(grid):
            continue

        try:
            actual = main(grid)
        except AssertionError:
            continue
        if not np.array_equal(actual, expected):
            continue

        tasks = _find_stamp_tasks(grid, BG)
        if len(tasks) != object_count:
            continue
        return grid

    raise RuntimeError("failed to generate a valid ARC input")
