import random

import numpy as np


DIR_TO_DELTA = {
    "R": (0, 1),
    "L": (0, -1),
    "D": (1, 0),
    "U": (-1, 0),
}


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((int(cr), int(cc)))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _is_consecutive(values: list[int]) -> bool:
    return bool(values) and values == list(range(values[0], values[-1] + 1))


def _match_arrow_geometry(
    body_coords: list[tuple[int, int]],
    head_coords: list[tuple[int, int]],
) -> dict[str, object] | None:
    body = set(body_coords)
    head = set(head_coords)

    body_rows = sorted({r for r, _ in body})
    body_cols = sorted({c for _, c in body})

    if len(body_rows) == 1 and _is_consecutive(body_cols):
        row = int(body_rows[0])
        c0 = int(body_cols[0])
        c1 = int(body_cols[-1])
        right_head = {(row - 1, c1 + 1), (row, c1 + 1), (row + 1, c1 + 1)}
        left_head = {(row - 1, c0 - 1), (row, c0 - 1), (row + 1, c0 - 1)}
        if head == right_head:
            return {
                "direction": "R",
                "tip": (row, c1 + 1),
                "head_coords": sorted(head),
                "body_coords": sorted(body),
            }
        if head == left_head:
            return {
                "direction": "L",
                "tip": (row, c0 - 1),
                "head_coords": sorted(head),
                "body_coords": sorted(body),
            }

    if len(body_cols) == 1 and _is_consecutive(body_rows):
        col = int(body_cols[0])
        r0 = int(body_rows[0])
        r1 = int(body_rows[-1])
        down_head = {(r1 + 1, col - 1), (r1 + 1, col), (r1 + 1, col + 1)}
        up_head = {(r0 - 1, col - 1), (r0 - 1, col), (r0 - 1, col + 1)}
        if head == down_head:
            return {
                "direction": "D",
                "tip": (r1 + 1, col),
                "head_coords": sorted(head),
                "body_coords": sorted(body),
            }
        if head == up_head:
            return {
                "direction": "U",
                "tip": (r0 - 1, col),
                "head_coords": sorted(head),
                "body_coords": sorted(body),
            }

    return None


def _parse_arrow_component(grid: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object]:
    colors = sorted({int(grid[r, c]) for r, c in coords})
    assert len(colors) == 2, "each arrow must use exactly two colors"

    candidates: list[dict[str, object]] = []
    for body_color in colors:
        head_color = colors[0] if colors[1] == body_color else colors[1]
        body_coords = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == int(body_color)]
        head_coords = [(int(r), int(c)) for r, c in coords if int(grid[r, c]) == int(head_color)]
        if not (2 <= len(body_coords) <= 4 and len(head_coords) == 3):
            continue
        geometry = _match_arrow_geometry(body_coords, head_coords)
        if geometry is None:
            continue
        rows = [r for r, _ in coords]
        cols = [c for _, c in coords]
        geometry["body_color"] = int(body_color)
        geometry["head_color"] = int(head_color)
        geometry["anchor"] = (min(rows), min(cols))
        candidates.append(geometry)

    assert len(candidates) == 1, "component is not a valid arrow"
    return candidates[0]


def _extract_arrows(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    arrows = [
        _parse_arrow_component(grid, comp)
        for comp in _components4(grid != int(bg))
    ]
    arrows.sort(key=lambda arrow: tuple(arrow["anchor"]))
    assert arrows, "expected at least one arrow"
    return arrows


def _trace_arrow_rays(
    grid: np.ndarray,
    bg: int,
    arrow: dict[str, object],
) -> tuple[list[tuple[int, int]], int]:
    h, w = grid.shape
    dr, dc = DIR_TO_DELTA[str(arrow["direction"])]
    painted: list[tuple[int, int]] = []
    blocked_lanes = 0
    for hr, hc in arrow["head_coords"]:
        r = int(hr)
        c = int(hc)
        while True:
            r += dr
            c += dc
            if not (0 <= r < h and 0 <= c < w):
                break
            if int(grid[r, c]) != int(bg):
                blocked_lanes += 1
                break
            painted.append((int(r), int(c)))
    return painted, int(blocked_lanes)


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    arrows = _extract_arrows(g, bg)

    out = g.copy()
    for arrow in arrows:
        ray_cells, _blocked = _trace_arrow_rays(g, bg, arrow)
        for r, c in ray_cells:
            out[r, c] = int(arrow["body_color"])
    return out


def _build_arrow(
    tail_r: int,
    tail_c: int,
    direction: str,
    length: int,
    body_color: int,
    head_color: int,
) -> dict[str, object]:
    if direction == "R":
        body = [(tail_r, tail_c + i) for i in range(length)]
        head = [(tail_r - 1, tail_c + length), (tail_r, tail_c + length), (tail_r + 1, tail_c + length)]
    elif direction == "L":
        body = [(tail_r, tail_c - i) for i in range(length)]
        head = [(tail_r - 1, tail_c - length), (tail_r, tail_c - length), (tail_r + 1, tail_c - length)]
    elif direction == "D":
        body = [(tail_r + i, tail_c) for i in range(length)]
        head = [(tail_r + length, tail_c - 1), (tail_r + length, tail_c), (tail_r + length, tail_c + 1)]
    else:
        assert direction == "U"
        body = [(tail_r - i, tail_c) for i in range(length)]
        head = [(tail_r - length, tail_c - 1), (tail_r - length, tail_c), (tail_r - length, tail_c + 1)]
    return {
        "direction": direction,
        "body_color": int(body_color),
        "head_color": int(head_color),
        "body_coords": body,
        "head_coords": head,
        "coords": body + head,
    }


def _tail_candidates(height: int, width: int, direction: str, length: int) -> list[tuple[int, int]]:
    candidates: list[tuple[int, int]] = []
    if direction == "R":
        for r in range(1, height - 1):
            for c in range(0, width - length - 1):
                candidates.append((int(r), int(c)))
    elif direction == "L":
        for r in range(1, height - 1):
            for c in range(length + 1, width):
                candidates.append((int(r), int(c)))
    elif direction == "D":
        for r in range(0, height - length - 1):
            for c in range(1, width - 1):
                candidates.append((int(r), int(c)))
    else:
        assert direction == "U"
        for r in range(length + 1, height):
            for c in range(1, width - 1):
                candidates.append((int(r), int(c)))
    return candidates


def _mark_forbidden(forbidden: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = forbidden.shape
    for r, c in coords:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                rr = int(r + dr)
                cc = int(c + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    forbidden[rr, cc] = True


def _sample_directions(count: int) -> list[str]:
    if count == 1:
        return [random.choice(list(DIR_TO_DELTA))]

    directions = [random.choice(["L", "R"]), random.choice(["U", "D"])]
    if count >= 4:
        directions = ["L", "R", "U", "D"]
    while len(directions) < count:
        directions.append(random.choice(list(DIR_TO_DELTA)))
    random.shuffle(directions)
    return directions


def _sample_color_pairs(bg: int, count: int) -> list[tuple[int, int]]:
    colors = [color for color in range(10) if color != int(bg)]
    random.shuffle(colors)
    body_colors = [colors[i % len(colors)] for i in range(count)]
    pairs: list[tuple[int, int]] = []
    for body_color in body_colors:
        head_choices = [color for color in colors if color != body_color]
        pairs.append((int(body_color), int(random.choice(head_choices))))
    return pairs


def _place_random_arrows(
    height: int,
    width: int,
    bg: int,
    directions: list[str],
) -> np.ndarray | None:
    grid = np.full((height, width), int(bg), dtype=int)
    forbidden = np.zeros((height, width), dtype=bool)

    lengths = [random.randint(2, 4) for _ in directions]
    color_pairs = _sample_color_pairs(bg, len(directions))
    order = list(range(len(directions)))
    order.sort(key=lambda i: lengths[i], reverse=True)

    for index in order:
        direction = directions[index]
        length = lengths[index]
        body_color, head_color = color_pairs[index]
        candidates: list[dict[str, object]] = []
        tails = _tail_candidates(height, width, direction, length)
        random.shuffle(tails)
        for tail_r, tail_c in tails:
            arrow = _build_arrow(tail_r, tail_c, direction, length, body_color, head_color)
            if all(not forbidden[r, c] for r, c in arrow["coords"]):
                candidates.append(arrow)
        if not candidates:
            return None
        arrow = random.choice(candidates)
        for r, c in arrow["body_coords"]:
            grid[r, c] = int(arrow["body_color"])
        for r, c in arrow["head_coords"]:
            grid[r, c] = int(arrow["head_color"])
        _mark_forbidden(forbidden, arrow["coords"])

    return grid


def _scene_is_valid(grid: np.ndarray, want_blocked: bool) -> bool:
    bg = _background_color(grid)
    if int(np.count_nonzero(grid == bg)) <= grid.size // 2:
        return False

    try:
        arrows = _extract_arrows(grid, bg)
    except AssertionError:
        return False

    all_ray_cells: dict[tuple[int, int], int] = {}
    blocked_lanes = 0
    total_ray_cells = 0
    for arrow in arrows:
        ray_cells, lane_blocks = _trace_arrow_rays(grid, bg, arrow)
        blocked_lanes += int(lane_blocks)
        total_ray_cells += len(ray_cells)
        for cell in ray_cells:
            if cell in all_ray_cells:
                return False
            all_ray_cells[cell] = int(arrow["body_color"])

    if total_ray_cells == 0:
        return False
    if want_blocked != (blocked_lanes > 0):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if int(np.count_nonzero(out != grid)) != total_ray_cells:
        return False
    return True


def _attempt_generate() -> np.ndarray | None:
    want_blocked = random.choice([False, True])
    if want_blocked:
        height = random.randint(11, 15)
        width = random.randint(11, 15)
        arrow_count = random.randint(3, 5)
    else:
        height = random.randint(13, 19)
        width = random.randint(13, 19)
        arrow_count = random.randint(2, 4)

    bg = random.randint(0, 9)
    directions = _sample_directions(arrow_count)
    grid = _place_random_arrows(height, width, bg, directions)
    if grid is None:
        return None
    if not _scene_is_valid(grid, want_blocked):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow scene")
