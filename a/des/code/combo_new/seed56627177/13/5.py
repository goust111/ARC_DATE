from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
WALL = 9
CONTENT_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
OPEN_SIDES = ("up", "down", "left", "right")
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _split_segments(indices: np.ndarray) -> list[np.ndarray]:
    if indices.size == 0:
        return []
    cuts = np.where(np.diff(indices) != 1)[0] + 1
    return [seg for seg in np.split(indices, cuts) if seg.size > 0]


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _mask_component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if seen[r, c] or not bool(mask[r, c]):
                continue
            count += 1
            queue = deque([(r, c)])
            seen[r, c] = True
            while queue:
                cr, cc = queue.popleft()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and bool(mask[nr, nc]):
                        seen[nr, nc] = True
                        queue.append((nr, nc))
    return count


def _find_containers(grid: np.ndarray) -> list[dict[str, object]]:
    containers: list[dict[str, object]] = []
    for coords in _components_of_color(grid, WALL):
        rows = coords[:, 0]
        cols = coords[:, 1]
        r0, r1 = int(rows.min()), int(rows.max())
        c0, c1 = int(cols.min()), int(cols.max())
        h, w = int(r1 - r0 + 1), int(c1 - c0 + 1)
        if h < 3 or w < 3:
            continue

        local_wall = np.zeros((h, w), dtype=bool)
        local_wall[rows - r0, cols - c0] = True

        # A valid container wall stays on the outer border of its bbox.
        if np.any(
            (rows != r0)
            & (rows != r1)
            & (cols != c0)
            & (cols != c1)
        ):
            continue

        inside = ~local_wall
        if not bool(inside.any()):
            continue
        if _mask_component_count(inside) != 1:
            continue

        touched: list[str] = []
        if bool(inside[0, :].any()):
            touched.append("up")
        if bool(inside[-1, :].any()):
            touched.append("down")
        if bool(inside[:, 0].any()):
            touched.append("left")
        if bool(inside[:, -1].any()):
            touched.append("right")
        if len(touched) != 1:
            continue
        opening = touched[0]

        if opening != "up" and not bool(local_wall[0, :].all()):
            continue
        if opening != "down" and not bool(local_wall[-1, :].all()):
            continue
        if opening != "left" and not bool(local_wall[:, 0].all()):
            continue
        if opening != "right" and not bool(local_wall[:, -1].all()):
            continue

        containers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "opening": opening,
                "inside": inside,
            }
        )
    containers.sort(key=lambda item: tuple(int(x) for x in item["bbox"]))  # stable output
    return containers


def _settle_patch(src: np.ndarray, inside: np.ndarray, opening: str) -> np.ndarray:
    gravity = {
        "up": "down",
        "down": "up",
        "left": "right",
        "right": "left",
    }[opening]
    patch = src.copy()
    patch[inside] = BG
    h, w = inside.shape

    if gravity in {"up", "down"}:
        for c in range(w):
            rows = np.flatnonzero(inside[:, c])
            for segment in _split_segments(rows):
                values = [int(src[r, c]) for r in segment if int(src[r, c]) != BG]
                if not values:
                    continue
                targets = segment[: len(values)] if gravity == "up" else segment[-len(values) :]
                for rr, value in zip(targets, values):
                    patch[int(rr), c] = value
    else:
        for r in range(h):
            cols = np.flatnonzero(inside[r, :])
            for segment in _split_segments(cols):
                values = [int(src[r, c]) for c in segment if int(src[r, c]) != BG]
                if not values:
                    continue
                targets = segment[: len(values)] if gravity == "left" else segment[-len(values) :]
                for cc, value in zip(targets, values):
                    patch[r, int(cc)] = value

    return patch


def main(grid: np.ndarray) -> np.ndarray:
    """
    Let the colored pixels inside each open maroon container settle in the direction opposite to that container's opening.

    Input structure:
    - The background is black.
    - One or more single-pixel-thick maroon container walls appear as open U-like frames with an opening on exactly one side.
    - Each container interior contains scattered non-black pixels and empty black cells; unrelated colored noise may also exist outside containers.

    Rule:
    - Detect every maroon container and determine which bbox side is open.
    - Use gravity in the opposite direction of that opening.
    - Within that container interior, slide all non-black pixels along the gravity axis until they pack tightly against the far wall, preserving order within each column or row.
    - Keep the maroon walls and all pixels outside containers unchanged.

    Key insight:
    - The transformation is local to each container: infer its opening from geometry first, then apply a stable one-axis packing inside only that container.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    for container in _find_containers(g):
        r0, c0, r1, c1 = container["bbox"]
        inside = np.asarray(container["inside"], dtype=bool)
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        settled = _settle_patch(patch, inside, str(container["opening"]))
        out[r0 : r1 + 1, c0 : c1 + 1] = settled
    return out


def _make_container(direction: str, inner_h: int, inner_w: int) -> tuple[np.ndarray, np.ndarray]:
    if direction in {"up", "down"}:
        h, w = inner_h + 1, inner_w + 2
    else:
        h, w = inner_h + 2, inner_w + 1

    wall = np.zeros((h, w), dtype=bool)
    if direction == "up":
        wall[:, 0] = True
        wall[:, -1] = True
        wall[-1, :] = True
    elif direction == "down":
        wall[:, 0] = True
        wall[:, -1] = True
        wall[0, :] = True
    elif direction == "left":
        wall[0, :] = True
        wall[-1, :] = True
        wall[:, -1] = True
    elif direction == "right":
        wall[0, :] = True
        wall[-1, :] = True
        wall[:, 0] = True
    else:
        raise ValueError(direction)
    return wall, ~wall


def _try_place_container(
    grid: np.ndarray,
    blocked: np.ndarray,
    direction: str,
) -> dict[str, object] | None:
    h, w = grid.shape
    for _ in range(120):
        inner_h = random.randint(2, 6)
        inner_w = random.randint(2, 6)
        wall, inside = _make_container(direction, inner_h, inner_w)
        ph, pw = wall.shape
        if ph + 2 >= h or pw + 2 >= w:
            continue
        r0 = random.randint(1, h - ph - 1)
        c0 = random.randint(1, w - pw - 1)
        rs = slice(r0 - 1, r0 + ph + 1)
        cs = slice(c0 - 1, c0 + pw + 1)
        if bool(blocked[rs, cs].any()):
            continue

        local = grid[r0 : r0 + ph, c0 : c0 + pw]
        local[wall] = WALL

        inside_cells = np.argwhere(inside)
        area = int(len(inside_cells))
        fill_count = random.randint(max(2, area // 4), max(2, area - 1))
        fill_count = min(fill_count, area - 1)
        if fill_count <= 0:
            continue

        color_count = random.randint(1, min(3, fill_count))
        palette = random.sample(list(CONTENT_COLORS), color_count)
        chosen = random.sample(range(area), fill_count)
        for idx in chosen:
            rr, cc = inside_cells[idx]
            local[int(rr), int(cc)] = random.choice(palette)

        blocked[rs, cs] = True
        return {
            "bbox": (r0, c0, r0 + ph - 1, c0 + pw - 1),
            "direction": direction,
        }
    return None


def _add_noise(grid: np.ndarray, blocked: np.ndarray) -> None:
    h, w = grid.shape
    noise_count = random.randint(0, 4)
    for _ in range(noise_count):
        choices: list[tuple[int, int]] = []
        for r in range(h):
            for c in range(w):
                if bool(blocked[r, c]) or int(grid[r, c]) != BG:
                    continue
                if any(
                    0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) != BG
                    for nr in range(r - 1, r + 2)
                    for nc in range(c - 1, c + 2)
                    if (nr, nc) != (r, c)
                ):
                    continue
                choices.append((r, c))
        if not choices:
            break
        r, c = random.choice(choices)
        grid[r, c] = random.choice(CONTENT_COLORS)
        blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(16, 30)
    w = random.randint(16, 30)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)

    container_count = random.randint(2, 4)
    directions = random.sample(list(OPEN_SIDES), container_count)
    placed: list[dict[str, object]] = []
    for direction in directions:
        info = _try_place_container(grid, blocked, direction)
        if info is None:
            return None
        placed.append(info)

    _add_noise(grid, blocked)

    found = _find_containers(grid)
    if len(found) != len(placed):
        return None
    if {str(item["opening"]) for item in found} != set(directions):
        return None
    if np.array_equal(main(grid), grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _gen_attempt()
        if result is not None:
            return result
    raise RuntimeError("failed to generate a valid open-container gravity input")
