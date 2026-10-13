import random
from collections import defaultdict

import numpy as np


_BG = 5
_PALETTE = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
_CORNERS = ("TL", "TR", "BR", "BL")
_CW_LOCAL = {
    "TL": (0, 1),
    "TR": (1, 1),
    "BR": (1, 0),
    "BL": (0, 0),
}
_CCW_LOCAL = {
    "TL": (1, 0),
    "TR": (0, 0),
    "BR": (0, 1),
    "BL": (1, 1),
}
_AREA_BANDS = (
    (0.45, 0.78),
    (0.30, 0.58),
    (0.20, 0.42),
    (0.12, 0.30),
    (0.08, 0.22),
)
_GEN_ATTEMPTS = 400
_PAIR_ATTEMPTS = 800


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _indicator_origin(shape: tuple[int, int], corner: str) -> tuple[int, int]:
    h, w = shape
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - 2
    if corner == "BR":
        return h - 2, w - 2
    if corner == "BL":
        return h - 2, 0
    raise AssertionError(corner)


def _corner_cells(shape: tuple[int, int], corner: str) -> list[tuple[int, int]]:
    r0, c0 = _indicator_origin(shape, corner)
    return [(r0 + r, c0 + c) for r in range(2) for c in range(2)]


def _path_cells(
    start: tuple[int, int], end: tuple[int, int], order: str
) -> list[tuple[int, int]]:
    sr, sc = start
    er, ec = end
    assert sr < er and sc != ec
    if order == "HV":
        cstep = 1 if ec > sc else -1
        cells = [(sr, c) for c in range(sc, ec + cstep, cstep)]
        cells.extend((r, ec) for r in range(sr + 1, er + 1))
        return cells
    if order == "VH":
        rstep = 1 if er > sr else -1
        cstep = 1 if ec > sc else -1
        cells = [(r, sc) for r in range(sr, er + rstep, rstep)]
        cells.extend((er, c) for c in range(sc + cstep, ec + cstep, cstep))
        return cells
    raise AssertionError(order)


def _path_blocked(
    state: np.ndarray, path: list[tuple[int, int]], bg: int
) -> bool:
    return any(int(state[r, c]) != bg for r, c in path[1:-1])


def _paint_path(
    state: np.ndarray, path: list[tuple[int, int]], color: int, bg: int
) -> None:
    last = len(path) - 1
    for idx, (r, c) in enumerate(path):
        value = int(state[r, c])
        if value == bg:
            state[r, c] = color
            continue
        assert idx in (0, last) and value == color


def _detect_indicator(
    grid: np.ndarray, bg: int
) -> tuple[str, str, set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[str, str, set[tuple[int, int]]]] = []
    for corner in _CORNERS:
        r0, c0 = _indicator_origin((h, w), corner)
        patch = grid[r0 : r0 + 2, c0 : c0 + 2]
        values, counts = np.unique(patch, return_counts=True)
        if len(values) != 2 or sorted(map(int, counts)) != [1, 3]:
            continue
        if bg in set(map(int, values)):
            continue
        single_color = int(values[np.argmin(counts)])
        local = tuple(map(int, np.argwhere(patch == single_color)[0]))
        if local == _CW_LOCAL[corner]:
            default_order = "HV"
        elif local == _CCW_LOCAL[corner]:
            default_order = "VH"
        else:
            continue
        candidates.append((corner, default_order, set(_corner_cells((h, w), corner))))
    assert len(candidates) == 1
    return candidates[0]


def _collect_pairs(
    grid: np.ndarray, bg: int, indicator_cells: set[tuple[int, int]]
) -> list[dict]:
    by_color: dict[int, list[tuple[int, int]]] = defaultdict(list)
    h, w = grid.shape
    for r in range(h):
        for c in range(w):
            if (r, c) in indicator_cells:
                continue
            color = int(grid[r, c])
            if color == bg:
                continue
            by_color[color].append((r, c))

    pairs: list[dict] = []
    for color, pts in by_color.items():
        assert len(pts) == 2
        start, end = sorted(pts)
        assert start[0] < end[0] and start[1] != end[1]
        area = (end[0] - start[0] + 1) * (abs(end[1] - start[1]) + 1)
        pairs.append(
            {
                "color": color,
                "start": start,
                "end": end,
                "area": area,
            }
        )

    assert pairs
    pairs.sort(
        key=lambda item: (
            -item["area"],
            item["start"][0],
            item["start"][1],
            item["end"][0],
            item["end"][1],
        )
    )
    return pairs


def _solve(grid: np.ndarray, return_trace: bool = False):
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    _corner, default_order, indicator_cells = _detect_indicator(g, bg)
    pairs = _collect_pairs(g, bg, indicator_cells)

    state = g.copy()
    trace: list[dict] = []
    for pair in pairs:
        hv_path = _path_cells(pair["start"], pair["end"], "HV")
        vh_path = _path_cells(pair["start"], pair["end"], "VH")
        default_path = hv_path if default_order == "HV" else vh_path
        alt_path = vh_path if default_order == "HV" else hv_path

        default_blocked = _path_blocked(state, default_path, bg)
        alt_blocked = _path_blocked(state, alt_path, bg)
        if default_blocked:
            assert not alt_blocked
            chosen_order = "VH" if default_order == "HV" else "HV"
            chosen_path = alt_path
        else:
            chosen_order = default_order
            chosen_path = default_path

        _paint_path(state, chosen_path, pair["color"], bg)
        if return_trace:
            trace.append(
                {
                    "start": pair["start"],
                    "end": pair["end"],
                    "area": pair["area"],
                    "color": pair["color"],
                    "default_order": default_order,
                    "chosen_order": chosen_order,
                    "default_blocked": default_blocked,
                    "path": chosen_path,
                }
            )

    if return_trace:
        return state, trace
    return state


def main(grid: np.ndarray) -> np.ndarray:
    """Connect each same-color seed pair with a single-turn orthogonal path controlled by a corner indicator."""
    return _solve(grid, return_trace=False)


def _place_indicator(
    grid: np.ndarray, corner: str, default_order: str, l_color: int, single_color: int
) -> set[tuple[int, int]]:
    r0, c0 = _indicator_origin(grid.shape, corner)
    patch = np.full((2, 2), l_color, dtype=int)
    sr, sc = _CW_LOCAL[corner] if default_order == "HV" else _CCW_LOCAL[corner]
    patch[sr, sc] = single_color
    grid[r0 : r0 + 2, c0 : c0 + 2] = patch
    return set(_corner_cells(grid.shape, corner))


def _is_input_clear(
    cell: tuple[int, int],
    h: int,
    w: int,
    occupied_input: set[tuple[int, int]],
    occupied_future: set[tuple[int, int]],
) -> bool:
    r, c = cell
    if not (1 <= r < h - 1 and 1 <= c < w - 1):
        return False
    if cell in occupied_input or cell in occupied_future:
        return False
    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
        if (nr, nc) in occupied_input:
            return False
    return True


def _sample_candidate_pair(
    h: int,
    w: int,
    area_min: int,
    area_max: int,
    default_order: str,
    desired_branch: str,
    occupied_future: set[tuple[int, int]],
    occupied_input: set[tuple[int, int]],
    painted_interior: set[tuple[int, int]],
    min_path_len: int,
) -> dict | None:
    for _ in range(_PAIR_ATTEMPTS):
        r1 = random.randint(1, h - 3)
        r2 = random.randint(r1 + 1, h - 2)
        c1 = random.randint(1, w - 2)
        c2 = random.randint(1, w - 2)
        if c1 == c2:
            continue
        start = (r1, c1)
        end = (r2, c2)
        if not _is_input_clear(start, h, w, occupied_input, occupied_future):
            continue
        if not _is_input_clear(end, h, w, occupied_input, occupied_future):
            continue
        if max(abs(r2 - r1), abs(c2 - c1)) < 3:
            continue

        area = (r2 - r1 + 1) * (abs(c2 - c1) + 1)
        if not (area_min <= area <= area_max):
            continue

        hv_path = _path_cells(start, end, "HV")
        vh_path = _path_cells(start, end, "VH")
        default_path = hv_path if default_order == "HV" else vh_path
        alt_path = vh_path if default_order == "HV" else hv_path

        default_blocked_any = any(cell in occupied_future for cell in default_path[1:-1])
        alt_blocked_any = any(cell in occupied_future for cell in alt_path[1:-1])
        default_hits_painted = any(cell in painted_interior for cell in default_path[1:-1])

        if desired_branch == "default":
            if default_blocked_any:
                continue
            chosen_order = default_order
            chosen_path = default_path
        else:
            if not default_hits_painted or alt_blocked_any:
                continue
            chosen_order = "VH" if default_order == "HV" else "HV"
            chosen_path = alt_path

        if len(chosen_path) < min_path_len:
            continue

        return {
            "start": start,
            "end": end,
            "area": area,
            "chosen_order": chosen_order,
            "chosen_path": chosen_path,
        }
    return None


def _branch_plan(pair_count: int) -> list[str]:
    plan = ["default"]
    for _ in range(pair_count - 1):
        plan.append("switch" if random.random() < 0.5 else "default")
    if "switch" not in plan[1:]:
        plan[random.randint(1, pair_count - 1)] = "switch"
    return plan


def _band_bounds(max_box_area: int, index: int) -> tuple[int, int]:
    low_frac, high_frac = _AREA_BANDS[index]
    area_min = max(6, int(round(max_box_area * low_frac)))
    area_max = max(area_min, int(round(max_box_area * high_frac)))
    return area_min, area_max


def _generate_one() -> np.ndarray | None:
    h = random.randint(11, 17)
    w = random.randint(11, 17)
    pair_count = random.randint(3, 5)
    if pair_count + 3 > len(_PALETTE):
        return None

    grid = np.full((h, w), _BG, dtype=int)
    default_order = random.choice(["HV", "VH"])
    corner = random.choice(_CORNERS)

    non_bg = [color for color in _PALETTE if color != _BG]
    random.shuffle(non_bg)
    l_color, single_color = non_bg[:2]
    pair_colors = non_bg[2 : 2 + pair_count]

    indicator_cells = _place_indicator(grid, corner, default_order, l_color, single_color)
    occupied_input = set(indicator_cells)
    occupied_future = set(indicator_cells)
    painted_interior: set[tuple[int, int]] = set()
    plan = _branch_plan(pair_count)
    max_box_area = (h - 2) * (w - 2)

    placed: list[dict] = []
    prev_area = max_box_area + 1
    for idx in range(pair_count):
        band_min, band_max = _band_bounds(max_box_area, idx)
        area_max = min(band_max, prev_area - 1)
        if area_max < band_min:
            return None
        min_path_len = max(5, (min(h, w) // 2) if idx == 0 else 4)
        candidate = _sample_candidate_pair(
            h=h,
            w=w,
            area_min=band_min,
            area_max=area_max,
            default_order=default_order,
            desired_branch=plan[idx],
            occupied_future=occupied_future,
            occupied_input=occupied_input,
            painted_interior=painted_interior,
            min_path_len=min_path_len,
        )
        if candidate is None:
            return None

        color = pair_colors[idx]
        sr, sc = candidate["start"]
        er, ec = candidate["end"]
        grid[sr, sc] = color
        grid[er, ec] = color
        occupied_input.add(candidate["start"])
        occupied_input.add(candidate["end"])
        occupied_future.update(candidate["chosen_path"])
        painted_interior.update(candidate["chosen_path"][1:-1])
        candidate["color"] = color
        candidate["branch"] = plan[idx]
        placed.append(candidate)
        prev_area = candidate["area"]

    out, trace = _solve(grid, return_trace=True)
    if len(trace) != pair_count:
        return None

    defaults = 0
    switches = 0
    for expected, actual in zip(placed, trace):
        if expected["start"] != actual["start"] or expected["end"] != actual["end"]:
            return None
        if expected["chosen_order"] != actual["chosen_order"]:
            return None
        if actual["chosen_order"] == default_order:
            defaults += 1
        else:
            switches += 1
        if expected["branch"] == "switch" and not actual["default_blocked"]:
            return None
        if expected["branch"] == "default" and actual["default_blocked"]:
            return None

    if defaults == 0 or switches == 0:
        return None
    if np.array_equal(grid, out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_one()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
