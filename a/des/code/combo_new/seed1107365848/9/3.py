from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BG = 0
DIRS = ("right", "left", "down", "up")
DIR_VECS = {
    "right": (0, 1),
    "left": (0, -1),
    "down": (1, 0),
    "up": (-1, 0),
}
GEN_ATTEMPTS = 400
PLACE_ATTEMPTS = 120


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    mask = grid == color
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _line_info(comp: list[tuple[int, int]]) -> dict | None:
    rows = sorted({r for r, _ in comp})
    cols = sorted({c for _, c in comp})
    if len(rows) == 1:
        row = rows[0]
        c0, c1 = cols[0], cols[-1]
        if len(comp) != c1 - c0 + 1:
            return None
        return {"orientation": "h", "fixed": row, "start": c0, "end": c1, "length": len(comp)}
    if len(cols) == 1:
        col = cols[0]
        r0, r1 = rows[0], rows[-1]
        if len(comp) != r1 - r0 + 1:
            return None
        return {"orientation": "v", "fixed": col, "start": r0, "end": r1, "length": len(comp)}
    return None


def _arrow_from_main(grid: np.ndarray, color: int, info: dict) -> dict | None:
    h, w = grid.shape
    if info["orientation"] == "h":
        r = int(info["fixed"])
        candidates = []
        if info["start"] - 1 >= 0:
            v = int(grid[r, info["start"] - 1])
            if v not in (BG, color):
                candidates.append(
                    {
                        "direction": "left",
                        "arrow_pixel": (r, int(info["start"])),
                        "tail_pos": (r, int(info["start"] - 1)),
                        "tail_color": v,
                    }
                )
        if info["end"] + 1 < w:
            v = int(grid[r, info["end"] + 1])
            if v not in (BG, color):
                candidates.append(
                    {
                        "direction": "right",
                        "arrow_pixel": (r, int(info["end"])),
                        "tail_pos": (r, int(info["end"] + 1)),
                        "tail_color": v,
                    }
                )
    else:
        c = int(info["fixed"])
        candidates = []
        if info["start"] - 1 >= 0:
            v = int(grid[info["start"] - 1, c])
            if v not in (BG, color):
                candidates.append(
                    {
                        "direction": "up",
                        "arrow_pixel": (int(info["start"]), c),
                        "tail_pos": (int(info["start"] - 1), c),
                        "tail_color": v,
                    }
                )
        if info["end"] + 1 < h:
            v = int(grid[info["end"] + 1, c])
            if v not in (BG, color):
                candidates.append(
                    {
                        "direction": "down",
                        "arrow_pixel": (int(info["end"]), c),
                        "tail_pos": (int(info["end"] + 1), c),
                        "tail_color": v,
                    }
                )
    if len(candidates) != 1:
        return None
    return candidates[0]


def _parse_object(grid: np.ndarray, color: int) -> dict | None:
    comps = _components_of_color(grid, color)
    if len(comps) != 2:
        return None

    infos = [_line_info(comp) for comp in comps]
    if any(info is None for info in infos):
        return None

    candidates = []
    for main_idx in (0, 1):
        main_info = infos[main_idx]
        wall_info = infos[1 - main_idx]
        assert main_info is not None and wall_info is not None
        arrow = _arrow_from_main(grid, color, main_info)
        if arrow is None:
            continue

        direction = arrow["direction"]
        dr, dc = DIR_VECS[direction]
        if main_info["orientation"] == "h":
            if wall_info["orientation"] != "v":
                continue
            row = int(main_info["fixed"])
            if not (int(wall_info["start"]) <= row <= int(wall_info["end"])):
                continue
            if direction == "right":
                path_cols = list(range(int(main_info["start"]), int(main_info["end"])))
                front = path_cols[-1]
            elif direction == "left":
                path_cols = list(range(int(main_info["start"]) + 1, int(main_info["end"]) + 1))
                front = path_cols[0]
            else:
                continue
            if len(path_cols) < 3:
                continue
            wall_coord = int(wall_info["fixed"])
            if direction == "right" and wall_coord <= front:
                continue
            if direction == "left" and wall_coord >= front:
                continue
            shift = abs(wall_coord - front) - 1
            if shift < 1:
                continue
            old_path = {(row, c) for c in path_cols}
        else:
            if wall_info["orientation"] != "h":
                continue
            col = int(main_info["fixed"])
            if not (int(wall_info["start"]) <= col <= int(wall_info["end"])):
                continue
            if direction == "down":
                path_rows = list(range(int(main_info["start"]), int(main_info["end"])))
                front = path_rows[-1]
            elif direction == "up":
                path_rows = list(range(int(main_info["start"]) + 1, int(main_info["end"]) + 1))
                front = path_rows[0]
            else:
                continue
            if len(path_rows) < 3:
                continue
            wall_coord = int(wall_info["fixed"])
            if direction == "down" and wall_coord <= front:
                continue
            if direction == "up" and wall_coord >= front:
                continue
            shift = abs(wall_coord - front) - 1
            if shift < 1:
                continue
            old_path = {(r, col) for r in path_rows}

        new_path = {(r + dr * shift, c + dc * shift) for r, c in old_path}
        h, w = grid.shape
        if any(not (0 <= r < h and 0 <= c < w) for r, c in new_path):
            continue

        candidates.append(
            {
                "color": color,
                "tail_color": int(arrow["tail_color"]),
                "direction": direction,
                "shift": int(shift),
                "arrow_pixel": tuple(map(int, arrow["arrow_pixel"])),
                "tail_pos": tuple(map(int, arrow["tail_pos"])),
                "old_path": old_path,
                "new_path": new_path,
            }
        )

    if len(candidates) != 1:
        return None
    return candidates[0]


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects = []
    for color in sorted(set(map(int, np.unique(grid))) - {BG}):
        obj = _parse_object(grid, color)
        if obj is not None:
            objects.append(obj)
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each straight colored path along its arrow direction until it reaches its same-colored wall.

    Input structure:
    - The background is black.
    - Each active color forms a one-pixel-wide straight path plus one attached same-colored arrow pixel at one end.
    - The second arrow pixel, immediately beyond that attached pixel, has a different tail color.
    - Farther along the same direction, after a black gap, a detached same-colored wall segment stands perpendicular to the path.

    Rule:
    - For every path, read the movement direction from the arrow orientation and the refill color from the arrow's second pixel.
    - Remove the two-pixel arrow marker from the scene.
    - Translate the path pixels as one rigid segment along the arrow direction until the leading end is adjacent to the matching wall.
    - Recolor the original path cells that are no longer occupied after the shift with the arrow's tail color; keep the wall unchanged.

    Key insight:
    - The arrow encodes two independent facts at once: which way the path moves, and which color refills the vacated trail.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    objects = _extract_objects(g)

    clears: set[tuple[int, int]] = set()
    tail_fills: list[tuple[int, int, int]] = []
    path_paints: list[tuple[int, int, int]] = []

    for obj in objects:
        old_path = set(obj["old_path"])
        new_path = set(obj["new_path"])
        vacated = old_path - new_path
        clears.update(old_path)
        clears.add(obj["arrow_pixel"])
        clears.add(obj["tail_pos"])
        for r, c in vacated:
            tail_fills.append((int(r), int(c), int(obj["tail_color"])))
        for r, c in new_path:
            path_paints.append((int(r), int(c), int(obj["color"])))

    for r, c in clears:
        out[r, c] = BG
    for r, c, color in tail_fills:
        out[r, c] = color
    for r, c, color in path_paints:
        out[r, c] = color
    return out


def _expand_bbox(bbox: tuple[int, int, int, int], h: int, w: int, margin: int = 1) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    return (
        max(0, r0 - margin),
        max(0, c0 - margin),
        min(h - 1, r1 + margin),
        min(w - 1, c1 + margin),
    )


def _build_object(
    h: int,
    w: int,
    direction: str,
    path_len: int,
    gap: int,
    wall_len: int,
    blocked: np.ndarray,
) -> dict | None:
    for _ in range(PLACE_ATTEMPTS):
        if direction == "right":
            c0_hi = w - path_len - gap - 4
            if c0_hi < 1:
                return None
            row = random.randint(1, h - 2)
            c0 = random.randint(1, c0_hi)
            wall_col = c0 + path_len + gap + 2
            wall_top_lo = max(1, row - wall_len + 1)
            wall_top_hi = min(row, h - wall_len - 1)
            if wall_top_lo > wall_top_hi:
                continue
            wall_top = random.randint(wall_top_lo, wall_top_hi)
            path = {(row, c) for c in range(c0, c0 + path_len)}
            arrow = (row, c0 + path_len)
            tail = (row, c0 + path_len + 1)
            wall = {(r, wall_col) for r in range(wall_top, wall_top + wall_len)}
            shift = gap + 2
            moved = {(row, c + shift) for c in range(c0, c0 + path_len)}
            bbox = (wall_top, c0, wall_top + wall_len - 1, wall_col)
        elif direction == "left":
            c0_lo = gap + 4
            c0_hi = w - path_len - 1
            if c0_lo > c0_hi:
                return None
            row = random.randint(1, h - 2)
            c0 = random.randint(c0_lo, c0_hi)
            wall_col = c0 - gap - 3
            wall_top_lo = max(1, row - wall_len + 1)
            wall_top_hi = min(row, h - wall_len - 1)
            if wall_top_lo > wall_top_hi:
                continue
            wall_top = random.randint(wall_top_lo, wall_top_hi)
            path = {(row, c) for c in range(c0, c0 + path_len)}
            arrow = (row, c0 - 1)
            tail = (row, c0 - 2)
            wall = {(r, wall_col) for r in range(wall_top, wall_top + wall_len)}
            shift = gap + 2
            moved = {(row, c - shift) for c in range(c0, c0 + path_len)}
            bbox = (wall_top, wall_col, wall_top + wall_len - 1, c0 + path_len - 1)
        elif direction == "down":
            r0_hi = h - path_len - gap - 4
            if r0_hi < 1:
                return None
            col = random.randint(1, w - 2)
            r0 = random.randint(1, r0_hi)
            wall_row = r0 + path_len + gap + 2
            wall_left_lo = max(1, col - wall_len + 1)
            wall_left_hi = min(col, w - wall_len - 1)
            if wall_left_lo > wall_left_hi:
                continue
            wall_left = random.randint(wall_left_lo, wall_left_hi)
            path = {(r, col) for r in range(r0, r0 + path_len)}
            arrow = (r0 + path_len, col)
            tail = (r0 + path_len + 1, col)
            wall = {(wall_row, c) for c in range(wall_left, wall_left + wall_len)}
            shift = gap + 2
            moved = {(r + shift, col) for r in range(r0, r0 + path_len)}
            bbox = (r0, wall_left, wall_row, wall_left + wall_len - 1)
        else:
            assert direction == "up"
            r0_lo = gap + 4
            r0_hi = h - path_len - 1
            if r0_lo > r0_hi:
                return None
            col = random.randint(1, w - 2)
            r0 = random.randint(r0_lo, r0_hi)
            wall_row = r0 - gap - 3
            wall_left_lo = max(1, col - wall_len + 1)
            wall_left_hi = min(col, w - wall_len - 1)
            if wall_left_lo > wall_left_hi:
                continue
            wall_left = random.randint(wall_left_lo, wall_left_hi)
            path = {(r, col) for r in range(r0, r0 + path_len)}
            arrow = (r0 - 1, col)
            tail = (r0 - 2, col)
            wall = {(wall_row, c) for c in range(wall_left, wall_left + wall_len)}
            shift = gap + 2
            moved = {(r - shift, col) for r in range(r0, r0 + path_len)}
            bbox = (wall_row, wall_left, r0 + path_len - 1, wall_left + wall_len - 1)

        footprint = set(path) | {arrow, tail} | set(wall)
        corridor = footprint | moved
        if any(not (0 <= r < h and 0 <= c < w) for r, c in corridor):
            continue
        rr0, cc0, rr1, cc1 = _expand_bbox(bbox, h, w, margin=1)
        if bool(blocked[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
            continue
        return {
            "direction": direction,
            "path": path,
            "arrow": arrow,
            "tail": tail,
            "wall": wall,
            "moved": moved,
            "bbox": bbox,
            "blocked_bbox": (rr0, cc0, rr1, cc1),
        }
    return None


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(1, 4)
    min_side = {1: 14, 2: 16, 3: 18, 4: 22}[n_objects]
    h = random.randint(min_side, 30)
    w = random.randint(min_side, 30)

    palette = list(range(1, 10))
    random.shuffle(palette)
    path_colors = palette[:n_objects]
    tail_colors = palette[n_objects : 2 * n_objects]
    directions = random.sample(DIRS, n_objects)

    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    objects = []

    for idx in range(n_objects):
        spec = _build_object(
            h=h,
            w=w,
            direction=directions[idx],
            path_len=random.randint(3, 8),
            gap=random.randint(1, 5),
            wall_len=random.randint(2, 6),
            blocked=blocked,
        )
        if spec is None:
            return None
        color = path_colors[idx]
        tail_color = tail_colors[idx]
        for r, c in spec["path"]:
            grid[r, c] = color
        grid[spec["arrow"]] = color
        grid[spec["tail"]] = tail_color
        for r, c in spec["wall"]:
            grid[r, c] = color
        rr0, cc0, rr1, cc1 = spec["blocked_bbox"]
        blocked[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
        objects.append((spec, color, tail_color))

    parsed = _extract_objects(grid)
    if len(parsed) != n_objects:
        return None
    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"Failed to generate a valid input after {GEN_ATTEMPTS} attempts")
