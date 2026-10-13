from __future__ import annotations

from collections import deque

import numpy as np


BG = 8
GREY = 5
PALETTE = (1, 2, 3, 4, 6, 7, 9)
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}


def _components_of_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                pts.append((rr, cc))
                for dr, dc in DIRS.values():
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _build_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color in (BG, GREY):
            continue
        for pts in _components_of_color(grid == color):
            rows = pts[:, 0]
            cols = pts[:, 1]
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            mask[rows - r0, cols - c0] = True
            objects.append(
                {
                    "color": color,
                    "pts": pts,
                    "bbox": (r0, c0, r1, c1),
                    "mask": mask,
                }
            )
    objects.sort(key=lambda obj: obj["bbox"])
    return objects


def _owner_map(shape: tuple[int, int], objects: list[dict]) -> np.ndarray:
    owner = np.full(shape, -1, dtype=int)
    for idx, obj in enumerate(objects):
        pts = obj["pts"]
        owner[pts[:, 0], pts[:, 1]] = idx
    return owner


def _touch_flags(obj: dict, owner: np.ndarray) -> dict[str, bool]:
    h, w = owner.shape
    pts = obj["pts"]
    idx = int(owner[pts[0, 0], pts[0, 1]])
    flags = {
        "ruler_left": False,
        "ruler_top": False,
        "obj_left": False,
        "obj_right": False,
        "obj_top": False,
        "obj_bottom": False,
    }
    for r, c in pts:
        r = int(r)
        c = int(c)
        if c == 1:
            flags["ruler_left"] = True
        if r == 1:
            flags["ruler_top"] = True
        if c > 0 and owner[r, c - 1] not in (-1, idx):
            flags["obj_left"] = True
        if c + 1 < w and owner[r, c + 1] not in (-1, idx):
            flags["obj_right"] = True
        if r > 0 and owner[r - 1, c] not in (-1, idx):
            flags["obj_top"] = True
        if r + 1 < h and owner[r + 1, c] not in (-1, idx):
            flags["obj_bottom"] = True
    return flags


def _clear_dirs(obj: dict, owner: np.ndarray) -> list[str]:
    h, w = owner.shape
    r0, c0, r1, c1 = obj["bbox"]
    idx = int(owner[obj["pts"][0, 0], obj["pts"][0, 1]])
    clear: list[str] = []

    def corridor_is_clear(rows: range, cols: range) -> bool:
        for rr in rows:
            for cc in cols:
                if owner[rr, cc] not in (-1, idx):
                    return False
                if rr == 0 or cc == 0:
                    return False
        return True

    if r0 > 1 and corridor_is_clear(range(1, r0), range(c0, c1 + 1)):
        clear.append("up")
    if corridor_is_clear(range(r1 + 1, h), range(c0, c1 + 1)):
        clear.append("down")
    if c0 > 1 and corridor_is_clear(range(r0, r1 + 1), range(1, c0)):
        clear.append("left")
    if corridor_is_clear(range(r0, r1 + 1), range(c1 + 1, w)):
        clear.append("right")
    return clear


def _detect_direction(obj: dict, owner: np.ndarray) -> str:
    flags = _touch_flags(obj, owner)
    if flags["ruler_left"] and not flags["ruler_top"]:
        return "right"
    if flags["ruler_top"] and not flags["ruler_left"]:
        return "down"
    if flags["obj_right"] and not flags["obj_left"]:
        return "left"
    if flags["obj_bottom"] and not flags["obj_top"]:
        return "up"

    clear = _clear_dirs(obj, owner)
    if len(clear) == 1:
        return clear[0]

    if flags["obj_left"] and not flags["obj_right"]:
        return "right"
    if flags["obj_top"] and not flags["obj_bottom"]:
        return "down"
    if flags["ruler_left"]:
        return "right"
    if flags["ruler_top"]:
        return "down"
    raise AssertionError("unable to infer object direction")


def _shadow_length(grid: np.ndarray, obj: dict, direction: str) -> int:
    pts = obj["pts"]
    if direction in ("left", "right"):
        cols = np.unique(pts[:, 1])
        return int(sum(int(grid[0, int(c)]) == GREY for c in cols))
    rows = np.unique(pts[:, 0])
    return int(sum(int(grid[int(r), 0]) == GREY for r in rows))


def _paint_extension(out: np.ndarray, obj: dict, direction: str, length: int) -> None:
    if length <= 0:
        return
    color = int(obj["color"])
    r0, c0, r1, c1 = obj["bbox"]
    box = obj["mask"]
    bh, bw = box.shape
    h, w = out.shape

    def paint_cell(rr: int, cc: int) -> None:
        if not (1 <= rr < h and 1 <= cc < w):
            return
        if box[(rr - r0) % bh, (cc - c0) % bw]:
            out[rr, cc] = color

    if direction == "right":
        for rr in range(r0, r1 + 1):
            for cc in range(c1 + 1, c1 + length + 1):
                paint_cell(rr, cc)
    elif direction == "left":
        for rr in range(r0, r1 + 1):
            for cc in range(c0 - length, c0):
                paint_cell(rr, cc)
    elif direction == "down":
        for rr in range(r1 + 1, r1 + length + 1):
            for cc in range(c0, c1 + 1):
                paint_cell(rr, cc)
    else:
        for rr in range(r0 - length, r0):
            for cc in range(c0, c1 + 1):
                paint_cell(rr, cc)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read ruler ticks from the top or left edge, then extend each colored object
    along its inferred direction by periodically tiling its original bbox mask.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _build_objects(g)
    owner = _owner_map(g.shape, objects)
    for obj in objects:
        direction = _detect_direction(obj, owner)
        length = _shadow_length(g, obj, direction)
        _paint_extension(out, obj, direction, length)
    return out


def _mask_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if pts.size == 0:
        return False
    seen = set()
    q = deque([tuple(map(int, pts[0]))])
    seen.add(q[0])
    while q:
        r, c = q.popleft()
        for dr, dc in DIRS.values():
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _generate_mask(size: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.ones((size, size), dtype=bool)
    target = int(rng.integers(0, size + 1))
    removed = 0
    for _ in range(size * size * 3):
        if removed >= target:
            break
        r = int(rng.integers(size))
        c = int(rng.integers(size))
        if not mask[r, c]:
            continue
        trial = mask.copy()
        trial[r, c] = False
        if not trial.any():
            continue
        rows = np.argwhere(trial)[:, 0]
        cols = np.argwhere(trial)[:, 1]
        if rows.min() != 0 or rows.max() != size - 1 or cols.min() != 0 or cols.max() != size - 1:
            continue
        if not _mask_connected(trial):
            continue
        mask = trial
        removed += 1
    return mask


def _stamp_mask(grid: np.ndarray, mask: np.ndarray, color: int, top: int, left: int) -> bool:
    h, w = grid.shape
    mh, mw = mask.shape
    if top < 1 or left < 1 or top + mh > h or left + mw > w:
        return False
    cells = np.argwhere(mask)
    for dr, dc in cells:
        rr = top + int(dr)
        cc = left + int(dc)
        if grid[rr, cc] != BG:
            return False
    for dr, dc in cells:
        grid[top + int(dr), left + int(dc)] = color
    return True


def _projection_length_from_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, direction: str) -> int:
    pts = np.argwhere(mask)
    if direction in ("left", "right"):
        cols = np.unique(pts[:, 1] + left)
        return int(sum(int(grid[0, int(c)]) == GREY for c in cols))
    rows = np.unique(pts[:, 0] + top)
    return int(sum(int(grid[int(r), 0]) == GREY for r in rows))


def _extension_cells(mask: np.ndarray, top: int, left: int, direction: str, length: int) -> list[tuple[int, int]]:
    mh, mw = mask.shape
    cells: list[tuple[int, int]] = []
    r0, c0 = top, left
    r1, c1 = top + mh - 1, left + mw - 1
    for rr in range(max(1, r0 - length), r1 + length + 1):
        for cc in range(max(1, c0 - length), c1 + length + 1):
            if r0 <= rr <= r1 and c0 <= cc <= c1:
                continue
            if direction == "right" and not (r0 <= rr <= r1 and c1 < cc <= c1 + length):
                continue
            if direction == "left" and not (r0 <= rr <= r1 and c0 - length <= cc < c0):
                continue
            if direction == "down" and not (c0 <= cc <= c1 and r1 < rr <= r1 + length):
                continue
            if direction == "up" and not (c0 <= cc <= c1 and r0 - length <= rr < r0):
                continue
            if mask[(rr - r0) % mh, (cc - c0) % mw]:
                cells.append((rr, cc))
    return cells


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(400):
        h = int(rng.integers(14, 21))
        w = int(rng.integers(14, 23))
        grid = np.full((h, w), BG, dtype=int)

        top_phase = int(rng.integers(2))
        left_phase = int(rng.integers(2))
        for c in range(w):
            if (c + top_phase) % 2 == 0:
                grid[0, c] = GREY
        for r in range(h):
            if (r + left_phase) % 2 == 0:
                grid[r, 0] = GREY

        colors = rng.choice(np.asarray(PALETTE, dtype=int), size=4, replace=False)
        size_r = int(rng.integers(2, 5))
        size_u = int(rng.integers(2, 5))
        size_d = int(rng.integers(2, 5))
        size_l = int(rng.integers(2, 5))

        row_r_min = max(size_u + 2, h // 2)
        row_r_max = h - size_r - 1
        col_d_min = max(size_l + 2, w // 2)
        col_d_max = w - size_d - 1
        if row_r_min > row_r_max or col_d_min > col_d_max:
            continue

        row_r = int(rng.integers(row_r_min, row_r_max + 1))
        col_r = 1
        row_u = row_r - size_u
        if row_u < 2:
            continue
        col_u_max = max(2, min(col_r + size_r - 1, w - size_u - 1))
        if col_u_max < 2:
            continue
        col_u = int(rng.integers(2, col_u_max + 1))

        col_d = int(rng.integers(col_d_min, col_d_max + 1))
        row_d = 1
        if size_d < 2:
            continue
        row_l_max = min(size_d, h - size_l - 1)
        if row_l_max < 2:
            continue
        row_l = int(rng.integers(2, row_l_max + 1))
        col_l = col_d - size_l
        if col_l < 2:
            continue

        mask_r = _generate_mask(size_r, rng)
        mask_u = _generate_mask(size_u, rng)
        mask_d = _generate_mask(size_d, rng)
        mask_l = _generate_mask(size_l, rng)

        if not _stamp_mask(grid, mask_r, int(colors[0]), row_r, col_r):
            continue
        if not _stamp_mask(grid, mask_u, int(colors[1]), row_u, col_u):
            continue
        if not _stamp_mask(grid, mask_d, int(colors[2]), row_d, col_d):
            continue
        if not _stamp_mask(grid, mask_l, int(colors[3]), row_l, col_l):
            continue

        objects = _build_objects(grid)
        if len(objects) != 4:
            continue
        owner = _owner_map(grid.shape, objects)
        try:
            detected = {int(obj["color"]): _detect_direction(obj, owner) for obj in objects}
        except AssertionError:
            continue
        expected = {
            int(colors[0]): "right",
            int(colors[1]): "up",
            int(colors[2]): "down",
            int(colors[3]): "left",
        }
        if detected != expected:
            continue

        ok = True
        for obj in objects:
            direction = detected[int(obj["color"])]
            length = _shadow_length(grid, obj, direction)
            if length <= 0:
                ok = False
                break
            for rr, cc in _extension_cells(
                obj["mask"],
                int(obj["bbox"][0]),
                int(obj["bbox"][1]),
                direction,
                length,
            ):
                if rr <= 0 or cc <= 0 or rr >= h or cc >= w or grid[rr, cc] != BG:
                    ok = False
                    break
            if not ok:
                break
        if not ok:
            continue

        _ = main(grid)
        return grid

    raise RuntimeError("failed to generate a valid input grid")
