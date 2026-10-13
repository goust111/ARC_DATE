from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
COLORS = tuple(range(10))
SIDES = ("top", "bottom", "left", "right")
OUTER_SIZE_CHOICES = (5, 7, 9, 11)
GEN_RETRY_LIMIT = 500
OBJECT_RETRY_LIMIT = 400


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    mask = grid == int(color)
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or visited[r, c]:
                continue
            stack = [(r, c)]
            visited[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not visited[nr, nc]:
                        visited[nr, nc] = True
                        stack.append((nr, nc))
            components.append(comp)
    return components


def _is_rectangular_frame(coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    if height < 5 or width < 5:
        return False
    expected_area = 2 * (height + width) - 4
    if len(coords) != expected_area:
        return False
    pts = set(coords)
    for c in range(c0, c1 + 1):
        if (r0, c) not in pts or (r1, c) not in pts:
            return False
    for r in range(r0 + 1, r1):
        if (r, c0) not in pts or (r, c1) not in pts:
            return False
    return True


def _seed_direction(
    seed: tuple[int, int],
    bbox: tuple[int, int, int, int],
) -> tuple[int, int] | None:
    sr, sc = seed
    r0, c0, r1, c1 = bbox
    directions: list[tuple[int, int]] = []
    if sr == r0 + 1 and c0 + 1 < sc < c1 - 1:
        directions.append((-1, 0))
    if sr == r1 - 1 and c0 + 1 < sc < c1 - 1:
        directions.append((1, 0))
    if sc == c0 + 1 and r0 + 1 < sr < r1 - 1:
        directions.append((0, -1))
    if sc == c1 - 1 and r0 + 1 < sr < r1 - 1:
        directions.append((0, 1))
    if len(directions) != 1:
        return None
    return directions[0]


def _marker_positions(
    bbox: tuple[int, int, int, int],
    side: str,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0 - 1, c) for c in range(c0 + 1, c1)]
    if side == "bottom":
        return [(r1 + 1, c) for c in range(c0 + 1, c1)]
    if side == "left":
        return [(r, c0 - 1) for r in range(r0 + 1, r1)]
    return [(r, c1 + 1) for r in range(r0 + 1, r1)]


def _count_markers(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> int:
    h, w = grid.shape
    total = 0
    for side in SIDES:
        for r, c in _marker_positions(bbox, side):
            if 0 <= r < h and 0 <= c < w and int(grid[r, c]) == GREY:
                total += 1
    return total


def _ray_start(bbox: tuple[int, int, int, int], direction: tuple[int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) // 2
    center_c = (c0 + c1) // 2
    if direction == (-1, 0):
        return r0 - 1, center_c
    if direction == (1, 0):
        return r1 + 1, center_c
    if direction == (0, -1):
        return center_r, c0 - 1
    return center_r, c1 + 1


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    color: int,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Read a seed pixel inside each hollow frame to choose the ray direction,
    count the grey outer markers to choose the ray length, then draw the ray
    outside the frame in the seed color.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    frames: list[dict[str, object]] = []
    for color in np.unique(g):
        color = int(color)
        if color in (bg, GREY):
            continue
        for comp in _components_of_color(g, color):
            if not _is_rectangular_frame(comp):
                continue
            bbox = _bbox(comp)
            r0, c0, r1, c1 = bbox
            inner = g[r0 + 1 : r1, c0 + 1 : c1]
            seed_rel = np.argwhere(inner != bg)
            if seed_rel.shape[0] != 1:
                continue
            sr = int(seed_rel[0, 0] + r0 + 1)
            sc = int(seed_rel[0, 1] + c0 + 1)
            seed_color = int(g[sr, sc])
            if seed_color in (bg, GREY, color):
                continue
            direction = _seed_direction((sr, sc), bbox)
            if direction is None:
                continue
            length = _count_markers(g, bbox)
            if length <= 0:
                continue
            frames.append(
                {
                    "bbox": bbox,
                    "seed_color": seed_color,
                    "direction": direction,
                    "length": length,
                }
            )

    out = g.copy()
    for frame in frames:
        bbox = frame["bbox"]
        seed_color = int(frame["seed_color"])
        direction = frame["direction"]
        length = int(frame["length"])
        start_r, start_c = _ray_start(bbox, direction)
        dr, dc = direction
        for step in range(length):
            rr = start_r + dr * step
            cc = start_c + dc * step
            if 0 <= rr < h and 0 <= cc < w and int(out[rr, cc]) == bg:
                out[rr, cc] = seed_color
    return out


def _outer_bbox(
    r0: int,
    c0: int,
    height: int,
    width: int,
) -> tuple[int, int, int, int]:
    return r0, c0, r0 + height - 1, c0 + width - 1


def _seed_positions(bbox: tuple[int, int, int, int], side: str) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    if side == "top":
        return [(r0 + 1, c) for c in range(c0 + 2, c1 - 1)]
    if side == "bottom":
        return [(r1 - 1, c) for c in range(c0 + 2, c1 - 1)]
    if side == "left":
        return [(r, c0 + 1) for r in range(r0 + 2, r1 - 1)]
    return [(r, c1 - 1) for r in range(r0 + 2, r1 - 1)]


def _side_capacity(height: int, width: int, side: str) -> int:
    if side in ("top", "bottom"):
        return width - 2
    return height - 2


def _ray_cells(
    bbox: tuple[int, int, int, int],
    direction: str,
    length: int,
) -> list[tuple[int, int]]:
    drdc = {
        "top": (-1, 0),
        "bottom": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }[direction]
    start_r, start_c = _ray_start(bbox, drdc)
    dr, dc = drdc
    return [(start_r + dr * k, start_c + dc * k) for k in range(length)]


def _expand_cells(cells: set[tuple[int, int]], shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w:
                    expanded.add((rr, cc))
    return expanded


def _build_object_spec(
    shape: tuple[int, int],
    frame_color: int,
    seed_color: int,
    direction: str,
    marker_side: str,
    forbidden: set[tuple[int, int]],
) -> dict[str, object] | None:
    h, w = shape
    for _ in range(OBJECT_RETRY_LIMIT):
        height = random.choice(OUTER_SIZE_CHOICES)
        width = random.choice(OUTER_SIZE_CHOICES)

        if direction in ("top", "bottom"):
            max_ray_anywhere = h - height - 1
        else:
            max_ray_anywhere = w - width - 1
        max_length = min(max_ray_anywhere, _side_capacity(height, width, marker_side))
        if max_length < 1:
            continue

        length = random.randint(1, max_length)

        if direction == "top":
            r_min, r_max = max(1, length), h - height - 1
            c_min, c_max = 1, w - width - 1
        elif direction == "bottom":
            r_min, r_max = 1, h - height - length
            c_min, c_max = 1, w - width - 1
        elif direction == "left":
            r_min, r_max = 1, h - height - 1
            c_min, c_max = max(1, length), w - width - 1
        else:
            r_min, r_max = 1, h - height - 1
            c_min, c_max = 1, w - width - length

        if r_min > r_max or c_min > c_max:
            continue

        r0 = random.randint(r_min, r_max)
        c0 = random.randint(c_min, c_max)
        bbox = _outer_bbox(r0, c0, height, width)

        seed_choices = _seed_positions(bbox, direction)
        marker_choices = _marker_positions(bbox, marker_side)
        if not seed_choices or len(marker_choices) < length:
            continue

        seed = random.choice(seed_choices)
        markers = random.sample(marker_choices, length)
        ray = _ray_cells(bbox, direction, length)

        used: set[tuple[int, int]] = set()
        for rr in range(r0, r0 + height):
            for cc in range(c0, c0 + width):
                used.add((rr, cc))
        used.add(seed)
        used.update(markers)
        used.update(ray)

        if any(cell in forbidden for cell in used):
            continue
        expanded = _expand_cells(used, shape)
        if any(cell in forbidden for cell in expanded):
            continue

        return {
            "bbox": bbox,
            "frame_color": frame_color,
            "seed_color": seed_color,
            "direction": direction,
            "seed": seed,
            "markers": markers,
            "length": length,
            "ray": ray,
            "forbidden": expanded,
        }
    return None


def _expected_output(inp: np.ndarray, specs: list[dict[str, object]]) -> np.ndarray:
    out = inp.copy()
    for spec in specs:
        for r, c in spec["ray"]:
            out[r, c] = int(spec["seed_color"])
    return out


def _background_is_dominant(grid: np.ndarray, bg: int) -> bool:
    counts = Counter(map(int, grid.flatten()))
    bg_count = counts[int(bg)]
    return all(bg_count > count for color, count in counts.items() if int(color) != int(bg))


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        height = random.randint(18, 30)
        width = random.randint(18, 30)
        bg = random.choice([c for c in COLORS if c != GREY])
        max_objects = min(4, max(1, (height * width) // 180))
        n_objects = random.randint(1, max_objects)

        palette = [c for c in COLORS if c not in (bg, GREY)]
        random.shuffle(palette)
        if len(palette) < 2 * n_objects:
            continue

        directions = random.sample(list(SIDES), n_objects)
        specs: list[dict[str, object]] = []
        forbidden: set[tuple[int, int]] = set()
        ok = True

        for idx in range(n_objects):
            direction = directions[idx]
            marker_side = random.choice([side for side in SIDES if side != direction])
            spec = _build_object_spec(
                (height, width),
                frame_color=palette[idx],
                seed_color=palette[idx + n_objects],
                direction=direction,
                marker_side=marker_side,
                forbidden=forbidden,
            )
            if spec is None:
                ok = False
                break
            specs.append(spec)
            forbidden.update(spec["forbidden"])

        if not ok:
            continue

        inp = np.full((height, width), bg, dtype=int)
        for spec in specs:
            _draw_frame(inp, spec["bbox"], int(spec["frame_color"]))
            sr, sc = spec["seed"]
            inp[sr, sc] = int(spec["seed_color"])
            for mr, mc in spec["markers"]:
                inp[mr, mc] = GREY

        if not _background_is_dominant(inp, bg):
            continue

        expected = _expected_output(inp, specs)
        if np.array_equal(main(inp), expected):
            return inp

    raise RuntimeError("failed to generate a valid ARC sample")
