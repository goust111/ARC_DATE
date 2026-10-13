import random
from collections import Counter, defaultdict

import numpy as np


DIRS8_CW = [
    (-1, 0),
    (-1, 1),
    (0, 1),
    (1, 1),
    (1, 0),
    (1, -1),
    (0, -1),
    (-1, -1),
]
BASE_DIR_BY_EDGE = {
    "top": (1, 0),
    "right": (0, -1),
    "bottom": (-1, 0),
    "left": (0, 1),
}
TURN_TYPES = ("straight", "left", "right")


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_cycle(shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    coords: list[tuple[int, int]] = []
    coords.extend((0, c) for c in range(w))
    coords.extend((r, w - 1) for r in range(1, h))
    coords.extend((h - 1, c) for c in range(w - 2, -1, -1))
    coords.extend((r, 0) for r in range(h - 2, 0, -1))
    return coords


def _edge_of(shape: tuple[int, int], coord: tuple[int, int]) -> str | None:
    h, w = shape
    r, c = coord
    if r == 0 and 0 < c < w - 1:
        return "top"
    if c == w - 1 and 0 < r < h - 1:
        return "right"
    if r == h - 1 and 0 < c < w - 1:
        return "bottom"
    if c == 0 and 0 < r < h - 1:
        return "left"
    return None


def _turn_45(base_dir: tuple[int, int], turn: str) -> tuple[int, int]:
    if turn == "straight":
        return base_dir
    idx = DIRS8_CW.index(base_dir)
    if turn == "right":
        return DIRS8_CW[(idx + 1) % len(DIRS8_CW)]
    assert turn == "left"
    return DIRS8_CW[(idx - 1) % len(DIRS8_CW)]


def _trace_ray(
    shape: tuple[int, int], start: tuple[int, int], step: tuple[int, int]
) -> list[tuple[int, int]]:
    h, w = shape
    r, c = start
    path: list[tuple[int, int]] = []
    while 0 <= r < h and 0 <= c < w:
        path.append((r, c))
        r += step[0]
        c += step[1]
    return path


def _infer_grey_color(grid: np.ndarray, bg: int, border: list[tuple[int, int]]) -> int:
    border_colors = [int(grid[coord]) for coord in border if int(grid[coord]) != bg]
    counts = Counter(border_colors)
    candidates = [(count, color) for color, count in counts.items() if count >= 2]
    assert candidates, "expected a repeated indicator color on the border"
    count, color = max(candidates)
    assert count >= 2
    return int(color)


def _extract_rays(grid: np.ndarray) -> tuple[int, list[dict[str, object]]]:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    border = _border_cycle(g.shape)
    grey = _infer_grey_color(g, bg, border)
    border_edges = [_edge_of(g.shape, coord) for coord in border]
    n = len(border)

    rays: list[dict[str, object]] = []
    for idx, coord in enumerate(border):
        color = int(g[coord])
        if color in (bg, grey):
            continue

        edge = border_edges[idx]
        assert edge is not None, "markers must be on non-corner border cells"
        prev_idx = (idx - 1) % n
        next_idx = (idx + 1) % n
        prev_is_grey = (
            border_edges[prev_idx] == edge and int(g[border[prev_idx]]) == grey
        )
        next_is_grey = (
            border_edges[next_idx] == edge and int(g[border[next_idx]]) == grey
        )
        assert not (prev_is_grey and next_is_grey), "marker can have at most one indicator"

        turn = "straight"
        if next_is_grey:
            turn = "right"
        elif prev_is_grey:
            turn = "left"

        step = _turn_45(BASE_DIR_BY_EDGE[edge], turn)
        rays.append(
            {
                "color": color,
                "priority": idx,
                "start": coord,
                "step": step,
                "path": _trace_ray(g.shape, coord, step),
            }
        )

    assert rays, "expected at least one ray"
    return bg, rays


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg, rays = _extract_rays(g)
    visits: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)

    for ray in rays:
        for coord in ray["path"]:
            visits[coord].append(ray)

    out = np.full_like(g, bg)
    for coord, touched in visits.items():
        if len(touched) < 2:
            continue
        chosen = min(touched, key=lambda ray: int(ray["priority"]))
        out[coord] = int(chosen["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Trace edge-launched rays, keep only the intersection cells, and color each
    intersection by the earliest ray marker encountered in clockwise border order.
    """
    return _solve(np.asarray(grid, dtype=int))


BG = 0
GRID_SIDE_MIN = 12
GRID_SIDE_MAX = 20
RAY_COUNT_MIN = 4
RAY_COUNT_MAX = 6
GEN_RETRIES = 400


def _make_branch_list(n_rays: int) -> list[str]:
    branches = ["straight", "left", "right"]
    while len(branches) < n_rays:
        branches.append(random.choice(TURN_TYPES))
    random.shuffle(branches)
    return branches


def _group_indices(
    border: list[tuple[int, int]],
    border_edges: list[str | None],
    marker_idx: int,
    turn: str,
) -> tuple[list[int], int | None]:
    n = len(border)
    edge = border_edges[marker_idx]
    assert edge is not None
    grey_idx = None
    if turn == "right":
        grey_idx = (marker_idx + 1) % n
        if border_edges[grey_idx] != edge:
            raise ValueError("right indicator crossed a corner")
    elif turn == "left":
        grey_idx = (marker_idx - 1) % n
        if border_edges[grey_idx] != edge:
            raise ValueError("left indicator crossed a corner")

    indices = [marker_idx]
    if grey_idx is not None:
        indices.append(grey_idx)
    return indices, grey_idx


def _sample_rays(
    shape: tuple[int, int], branch_list: list[str], colors: list[int]
) -> list[dict[str, object]] | None:
    border = _border_cycle(shape)
    border_edges = [_edge_of(shape, coord) for coord in border]
    n = len(border)
    blocked: set[int] = set()
    rays: list[dict[str, object]] = []

    for turn, color in zip(branch_list, colors):
        candidates: list[tuple[int, int | None, str]] = []
        indices = list(range(n))
        random.shuffle(indices)
        for marker_idx in indices:
            edge = border_edges[marker_idx]
            if edge is None or marker_idx in blocked:
                continue
            try:
                group, grey_idx = _group_indices(border, border_edges, marker_idx, turn)
            except ValueError:
                continue
            if any(idx in blocked for idx in group):
                continue
            candidates.append((marker_idx, grey_idx, edge))

        if not candidates:
            return None

        marker_idx, grey_idx, edge = random.choice(candidates)
        group = [marker_idx]
        if grey_idx is not None:
            group.append(grey_idx)
        for idx in group:
            blocked.add(idx)
            blocked.add((idx - 1) % n)
            blocked.add((idx + 1) % n)

        rays.append(
            {
                "color": color,
                "turn": turn,
                "edge": edge,
                "priority": marker_idx,
                "start": border[marker_idx],
                "grey": None if grey_idx is None else border[grey_idx],
                "step": _turn_45(BASE_DIR_BY_EDGE[edge], turn),
            }
        )

    return rays


def _build_input(
    shape: tuple[int, int], rays: list[dict[str, object]], grey_color: int
) -> np.ndarray:
    grid = np.full(shape, BG, dtype=int)
    for ray in rays:
        sr, sc = ray["start"]
        grid[sr, sc] = int(ray["color"])
        if ray["grey"] is not None:
            gr, gc = ray["grey"]
            grid[gr, gc] = grey_color
    return grid


def _intersection_data(
    shape: tuple[int, int], rays: list[dict[str, object]]
) -> tuple[dict[tuple[int, int], list[int]], list[set[tuple[int, int]]]]:
    visits: dict[tuple[int, int], list[int]] = defaultdict(list)
    path_sets: list[set[tuple[int, int]]] = []
    for idx, ray in enumerate(rays):
        path = _trace_ray(shape, ray["start"], ray["step"])
        ray["path"] = path
        path_set = set(path)
        path_sets.append(path_set)
        for coord in path:
            visits[coord].append(idx)
    return visits, path_sets


def _valid_layout(
    shape: tuple[int, int], rays: list[dict[str, object]], grey_color: int
) -> bool:
    h, w = shape
    visits, path_sets = _intersection_data(shape, rays)
    occupied_border = set()
    for ray in rays:
        occupied_border.add(ray["start"])
        if ray["grey"] is not None:
            occupied_border.add(ray["grey"])

    for ray in rays:
        for coord in ray["path"][1:]:
            if coord in occupied_border:
                return False

    for i in range(len(path_sets)):
        for j in range(i + 1, len(path_sets)):
            if len(path_sets[i] & path_sets[j]) > 1:
                return False

    intersections = {
        coord: ids
        for coord, ids in visits.items()
        if len(ids) >= 2 and 0 < coord[0] < h - 1 and 0 < coord[1] < w - 1
    }
    if not intersections or len(intersections) > 8:
        return False

    involved = {idx for ids in intersections.values() for idx in ids}
    if len(involved) < 3:
        return False

    if not any(rays[idx]["turn"] == "straight" for idx in involved):
        return False
    if not any(rays[idx]["turn"] != "straight" for idx in involved):
        return False

    horizontal_family = {"top", "bottom"}
    vertical_family = {"left", "right"}
    if not any(ray["edge"] in horizontal_family for ray in rays):
        return False
    if not any(ray["edge"] in vertical_family for ray in rays):
        return False

    grid = _build_input(shape, rays, grey_color)
    out = _solve(grid)
    if np.array_equal(out, np.full(shape, BG, dtype=int)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    w = random.randint(GRID_SIDE_MIN, GRID_SIDE_MAX)
    n_rays = random.randint(RAY_COUNT_MIN, RAY_COUNT_MAX)
    grey_color = random.randint(1, 9)
    marker_pool = [color for color in range(10) if color not in (BG, grey_color)]
    random.shuffle(marker_pool)
    marker_colors = marker_pool[:n_rays]
    branch_list = _make_branch_list(n_rays)

    rays = _sample_rays((h, w), branch_list, marker_colors)
    if rays is None:
        return None
    if not _valid_layout((h, w), rays, grey_color):
        return None
    return _build_input((h, w), rays, grey_color)


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ray-intersection input")
