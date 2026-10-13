from __future__ import annotations

from collections import Counter, defaultdict, deque
import random

import numpy as np


DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
INTERFACES = ("top", "bottom", "left", "right")

QUADRANT_TL = 0
QUADRANT_TR = 1
QUADRANT_BL = 2
QUADRANT_BR = 3

# --------------- solver ---------------


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _find_cross(grid: np.ndarray) -> tuple[int, int, int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _most_common_color(g)

    best: tuple[int, int, int, int, int] | None = None
    min_row_count = max(5, w - 4)
    min_col_count = max(5, h - 4)

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue
        row_counts = np.count_nonzero(g == color, axis=1)
        col_counts = np.count_nonzero(g == color, axis=0)
        row = int(np.argmax(row_counts))
        col = int(np.argmax(col_counts))
        row_count = int(row_counts[row])
        col_count = int(col_counts[col])
        if not (0 < row < h - 1 and 0 < col < w - 1):
            continue
        if row_count < min_row_count or col_count < min_col_count:
            continue
        candidate = (row_count + col_count, -abs(row - h // 2) - abs(col - w // 2), color, row, col)
        if best is None or candidate > best:
            best = candidate

    if best is None:
        raise ValueError("failed to identify separator cross")
    _score, _center_bias, sep, row, col = best
    return bg, sep, row, col


def _monochrome_components(grid: np.ndarray, forbidden: set[int]) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color in forbidden or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": coords})
    return comps


def _path_endpoints(coords: list[tuple[int, int]]) -> list[tuple[int, int]]:
    pts = set(coords)
    endpoints: list[tuple[int, int]] = []
    for cell in coords:
        degree = 0
        for dr, dc in DIRS4:
            if (cell[0] + dr, cell[1] + dc) in pts:
                degree += 1
        if degree == 1:
            endpoints.append(cell)
        elif degree > 2:
            raise ValueError("component is not a simple path")
    if len(endpoints) != 2:
        raise ValueError("expected each path to have exactly two endpoints")
    return endpoints


def _quadrant(point: tuple[int, int], split_r: int, split_c: int) -> int:
    r, c = point
    if r == split_r or c == split_c:
        raise ValueError("path endpoint must lie strictly inside a quadrant")
    if r < split_r and c < split_c:
        return QUADRANT_TL
    if r < split_r and c > split_c:
        return QUADRANT_TR
    if r > split_r and c < split_c:
        return QUADRANT_BL
    return QUADRANT_BR


def _analyze_paths(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    bg, sep, split_r, split_c = _find_cross(g)
    comps = _monochrome_components(g, {bg, sep})
    paths = []
    for comp in comps:
        coords = comp["coords"]
        endpoints = _path_endpoints(coords)  # type: ignore[arg-type]
        qa = _quadrant(endpoints[0], split_r, split_c)
        qb = _quadrant(endpoints[1], split_r, split_c)
        if qa == qb:
            raise ValueError("a valid path must connect different quadrants")
        paths.append(
            {
                "color": int(comp["color"]),
                "coords": list(coords),
                "key": tuple(sorted((qa, qb))),
            }
        )
    return {
        "bg": bg,
        "sep": sep,
        "split_r": split_r,
        "split_c": split_c,
        "paths": paths,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap colors between paired paths that connect the same pair of quadrants.

    The separator cross is detected from the dominant near-solid row and column.
    Every remaining monochrome 4-connected component is treated as one simple
    path. Paths are paired by the quadrants containing their two endpoints.
    """
    g = np.asarray(grid, dtype=int)
    info = _analyze_paths(g)
    out = g.copy()

    by_key: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for path in info["paths"]:  # type: ignore[index]
        by_key[path["key"]].append(path)

    for group in by_key.values():
        if len(group) != 2:
            continue
        color_a = int(group[0]["color"])
        color_b = int(group[1]["color"])
        for r, c in group[0]["coords"]:  # type: ignore[index]
            out[r, c] = color_b
        for r, c in group[1]["coords"]:  # type: ignore[index]
            out[r, c] = color_a

    return out


# --------------- generator ---------------

GRID_HEIGHTS = (15, 17, 19, 21)
GRID_WIDTHS = (15, 17, 19, 21)
COUNT_TEMPLATES = (
    (2, 2, 0, 0),
    (2, 1, 1, 0),
    (2, 1, 0, 0),
    (1, 1, 1, 1),
    (2, 2, 1, 0),
)
MAX_GENERATION_ATTEMPTS = 400


def _interface_quadrants(interface: str) -> tuple[int, int]:
    if interface == "top":
        return QUADRANT_TL, QUADRANT_TR
    if interface == "bottom":
        return QUADRANT_BL, QUADRANT_BR
    if interface == "left":
        return QUADRANT_TL, QUADRANT_BL
    return QUADRANT_TR, QUADRANT_BR


def _routing_bounds(
    quadrant: int,
    split_r: int,
    split_c: int,
    h: int,
    w: int,
) -> tuple[int, int, int, int]:
    if quadrant == QUADRANT_TL:
        return 1, split_r, 1, split_c
    if quadrant == QUADRANT_TR:
        return 1, split_r, split_c + 1, w - 1
    if quadrant == QUADRANT_BL:
        return split_r + 1, h - 1, 1, split_c
    return split_r + 1, h - 1, split_c + 1, w - 1


def _quadrant_mask(
    quadrant: int,
    split_r: int,
    split_c: int,
    h: int,
    w: int,
) -> np.ndarray:
    r0, r1, c0, c1 = _routing_bounds(quadrant, split_r, split_c, h, w)
    mask = np.zeros((h, w), dtype=bool)
    mask[r0:r1, c0:c1] = True
    return mask


def _gap_cells(
    interface: str,
    gap_pos: int,
    split_r: int,
    split_c: int,
) -> tuple[tuple[int, int], tuple[int, int], tuple[int, int]]:
    if interface == "top":
        return (gap_pos, split_c), (gap_pos, split_c - 1), (gap_pos, split_c + 1)
    if interface == "bottom":
        return (gap_pos, split_c), (gap_pos, split_c - 1), (gap_pos, split_c + 1)
    if interface == "left":
        return (split_r, gap_pos), (split_r - 1, gap_pos), (split_r + 1, gap_pos)
    return (split_r, gap_pos), (split_r - 1, gap_pos), (split_r + 1, gap_pos)


def _candidate_gap_positions(interface: str, split_r: int, split_c: int, h: int, w: int) -> list[int]:
    if interface == "top":
        return list(range(1, split_r - 1))
    if interface == "bottom":
        return list(range(split_r + 2, h - 1))
    if interface == "left":
        return list(range(1, split_c - 1))
    return list(range(split_c + 2, w - 1))


def _random_bfs_path(start: tuple[int, int], goal: tuple[int, int], allowed: np.ndarray) -> list[tuple[int, int]] | None:
    if not allowed[start] or not allowed[goal]:
        return None

    q = deque([start])
    prev: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    h, w = allowed.shape

    while q:
        cur = q.popleft()
        if cur == goal:
            break
        dirs = list(DIRS4)
        random.shuffle(dirs)
        for dr, dc in dirs:
            nr = cur[0] + dr
            nc = cur[1] + dc
            nxt = (nr, nc)
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if not allowed[nr, nc] or nxt in prev:
                continue
            prev[nxt] = cur
            q.append(nxt)

    if goal not in prev:
        return None

    path: list[tuple[int, int]] = []
    cur: tuple[int, int] | None = goal
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def _choose_endpoint(allowed: np.ndarray, connector: tuple[int, int]) -> tuple[int, int] | None:
    candidates: list[tuple[int, int]] = []
    h, w = allowed.shape
    for r in range(h):
        for c in range(w):
            if not allowed[r, c]:
                continue
            dist = abs(r - connector[0]) + abs(c - connector[1])
            if dist < 3:
                continue
            candidates.append((r, c))

    random.shuffle(candidates)
    if not candidates:
        return None

    candidates.sort(
        key=lambda p: (
            -abs(p[0] - connector[0]) - abs(p[1] - connector[1]),
            int(p[0] == connector[0] or p[1] == connector[1]),
        )
    )
    best_span = max(1, min(10, len(candidates)))
    return candidates[random.randrange(best_span)]


def _is_simple_chain(path: list[tuple[int, int]]) -> bool:
    if len(path) < 2 or len(set(path)) != len(path):
        return False
    pts = set(path)
    degrees = []
    for r, c in path:
        degree = 0
        for dr, dc in DIRS4:
            if (r + dr, c + dc) in pts:
                degree += 1
        degrees.append(degree)
    return degrees.count(1) == 2 and all(deg <= 2 for deg in degrees)


def _build_interface_path(
    blocked: np.ndarray,
    interface: str,
    gap_pos: int,
    split_r: int,
    split_c: int,
    h: int,
    w: int,
) -> list[tuple[int, int]] | None:
    qa, qb = _interface_quadrants(interface)
    gap, connector_a, connector_b = _gap_cells(interface, gap_pos, split_r, split_c)

    for cell in (gap, connector_a, connector_b):
        if blocked[cell]:
            return None

    allowed_a = _quadrant_mask(qa, split_r, split_c, h, w) & ~blocked
    allowed_b = _quadrant_mask(qb, split_r, split_c, h, w) & ~blocked
    allowed_a[connector_a] = True
    allowed_b[connector_b] = True

    endpoint_a = _choose_endpoint(allowed_a, connector_a)
    endpoint_b = _choose_endpoint(allowed_b, connector_b)
    if endpoint_a is None or endpoint_b is None:
        return None

    half_a = _random_bfs_path(connector_a, endpoint_a, allowed_a)
    half_b = _random_bfs_path(connector_b, endpoint_b, allowed_b)
    if half_a is None or half_b is None:
        return None

    path = list(reversed(half_a)) + [gap] + half_b
    if not _is_simple_chain(path):
        return None
    return path


def _reserve(blocked: np.ndarray, path: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in path:
        blocked[r, c] = True
        for dr, dc in DIRS4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _validate_generated_grid(grid: np.ndarray) -> bool:
    try:
        info = _analyze_paths(grid)
    except Exception:
        return False

    paths = info["paths"]  # type: ignore[index]
    if len(paths) < 2:
        return False

    counts = Counter(path["key"] for path in paths)
    if any(count > 2 for count in counts.values()):
        return False

    sep = int(info["sep"])  # type: ignore[index]
    split_r = int(info["split_r"])  # type: ignore[index]
    split_c = int(info["split_c"])  # type: ignore[index]
    if np.count_nonzero(grid[split_r] == sep) < grid.shape[1] - 4:
        return False
    if np.count_nonzero(grid[:, split_c] == sep) < grid.shape[0] - 4:
        return False

    try:
        out = main(grid)
    except Exception:
        return False

    if out.shape != grid.shape:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.choice(GRID_HEIGHTS)
    w = random.choice(GRID_WIDTHS)
    split_r = h // 2
    split_c = w // 2

    colors = list(range(10))
    bg = random.choice(colors)
    remaining = [c for c in colors if c != bg]
    sep = random.choice(remaining)
    remaining = [c for c in remaining if c != sep]

    counts = list(random.choice(COUNT_TEMPLATES))
    random.shuffle(counts)
    interface_counts = dict(zip(INTERFACES, counts))
    total_paths = sum(interface_counts.values())
    if total_paths > len(remaining):
        return None

    grid = np.full((h, w), bg, dtype=int)
    grid[split_r, :] = sep
    grid[:, split_c] = sep

    blocked = np.zeros((h, w), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    path_specs: list[tuple[str, list[tuple[int, int]]]] = []
    used_gaps: dict[str, list[int]] = {name: [] for name in INTERFACES}

    placement_order = sorted(INTERFACES, key=lambda name: (-interface_counts[name], random.random()))
    for interface in placement_order:
        need = interface_counts[interface]
        for _ in range(need):
            positions = _candidate_gap_positions(interface, split_r, split_c, h, w)
            random.shuffle(positions)
            placed = False
            for gap_pos in positions:
                if any(abs(gap_pos - prev) <= 1 for prev in used_gaps[interface]):
                    continue
                path = _build_interface_path(blocked, interface, gap_pos, split_r, split_c, h, w)
                if path is None:
                    continue
                _reserve(blocked, path)
                used_gaps[interface].append(gap_pos)
                path_specs.append((interface, path))
                placed = True
                break
            if not placed:
                return None

    random.shuffle(remaining)
    chosen_colors = remaining[:total_paths]
    random.shuffle(path_specs)
    for color, (_interface, path) in zip(chosen_colors, path_specs):
        for r, c in path:
            grid[r, c] = color

    return grid if _validate_generated_grid(grid) else None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
