import random
import numpy as np


BG = 8
STATION = 5
PATH = 9
MARKER_COLORS = [0, 1, 2, 3, 4, 6, 7]
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]

STATION_RING = np.array(
    [
        [True, True, True],
        [True, False, True],
        [True, True, True],
    ],
    dtype=bool,
)

WORK_H = 32
WORK_W = 32
MAX_GENERATION_ATTEMPTS = 400
PLACE_ATTEMPTS = 250


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in DIR4:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _station_patch(color: int) -> np.ndarray:
    patch = np.full((3, 3), STATION, dtype=int)
    patch[1, 1] = int(color)
    return patch


def _place_station(patch: np.ndarray, center_r: int, center_c: int, color: int) -> None:
    patch[center_r - 1 : center_r + 2, center_c - 1 : center_c + 2] = _station_patch(color)


def _pair_patch_horizontal(color_a: int, color_b: int) -> np.ndarray:
    gap = random.randint(5, 10)
    center_a = (1, 1)
    center_b = (1, 1 + gap)
    patch = np.full((3, gap + 3), BG, dtype=int)
    _place_station(patch, *center_a, color_a)
    _place_station(patch, *center_b, color_b)
    patch[1, center_a[1] + 2 : center_b[1] - 1] = PATH
    return patch


def _pair_patch_vertical(color_a: int, color_b: int) -> np.ndarray:
    gap = random.randint(5, 10)
    center_a = (1, 1)
    center_b = (1 + gap, 1)
    patch = np.full((gap + 3, 3), BG, dtype=int)
    _place_station(patch, *center_a, color_a)
    _place_station(patch, *center_b, color_b)
    patch[center_a[0] + 2 : center_b[0] - 1, 1] = PATH
    return patch


def _pair_patch_l(color_a: int, color_b: int, horizontal_first: bool) -> np.ndarray:
    row_gap = random.randint(5, 10)
    col_gap = random.randint(5, 10)
    center_a = (1, 1)
    center_b = (1 + row_gap, 1 + col_gap)
    patch = np.full((row_gap + 3, col_gap + 3), BG, dtype=int)
    _place_station(patch, *center_a, color_a)
    _place_station(patch, *center_b, color_b)

    if horizontal_first:
        patch[center_a[0], center_a[1] + 2 : center_b[1] + 1] = PATH
        patch[center_a[0] : center_b[0] - 1, center_b[1]] = PATH
    else:
        patch[center_a[0] + 2 : center_b[0] + 1, center_a[1]] = PATH
        patch[center_b[0], center_a[1] : center_b[1] - 1] = PATH

    if random.random() < 0.5:
        patch = np.flipud(patch).copy()
    if random.random() < 0.5:
        patch = np.fliplr(patch).copy()
    return patch


def _pair_patch(color_a: int, color_b: int) -> np.ndarray:
    kind = random.choice(["h", "v", "lh", "lv"])
    if kind == "h":
        return _pair_patch_horizontal(color_a, color_b)
    if kind == "v":
        return _pair_patch_vertical(color_a, color_b)
    if kind == "lh":
        return _pair_patch_l(color_a, color_b, horizontal_first=True)
    return _pair_patch_l(color_a, color_b, horizontal_first=False)


def _isolated_patch(color: int) -> np.ndarray:
    return _station_patch(color)


def _find_stations(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    stations: list[dict] = []

    for r in range(1, h - 1):
        for c in range(1, w - 1):
            center = int(grid[r, c])
            if center in (BG, STATION, PATH):
                continue
            patch = grid[r - 1 : r + 2, c - 1 : c + 2]
            if patch.shape != (3, 3):
                continue
            if np.all(patch[STATION_RING] == STATION):
                edge_cells = [
                    (r - 1, c - 1),
                    (r - 1, c),
                    (r - 1, c + 1),
                    (r, c - 1),
                    (r, c + 1),
                    (r + 1, c - 1),
                    (r + 1, c),
                    (r + 1, c + 1),
                ]
                body_cells = [(rr, cc) for rr in range(r - 1, r + 2) for cc in range(c - 1, c + 2)]
                stations.append(
                    {
                        "center": (r, c),
                        "marker": center,
                        "edge_cells": edge_cells,
                        "body_cells": body_cells,
                    }
                )
    return stations


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify pairs of stations connected by maroon paths and cross-transfer
    their center colors onto the partner station's grey border pixels.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    stations = _find_stations(g)
    assert stations, "expected at least one station"

    body_to_station: dict[tuple[int, int], int] = {}
    for sid, station in enumerate(stations):
        for cell in station["body_cells"]:
            assert cell not in body_to_station, "stations may not overlap"
            body_to_station[cell] = sid

    pairs: list[tuple[int, int]] = []
    seen_station_ids: set[int] = set()
    for comp in _components(g == PATH):
        touched: set[int] = set()
        for r, c in comp:
            for dr, dc in DIR4:
                sid = body_to_station.get((r + dr, c + dc))
                if sid is not None:
                    touched.add(sid)
        assert len(touched) == 2, "each path component must connect exactly two stations"
        a, b = sorted(touched)
        assert a not in seen_station_ids and b not in seen_station_ids, "station may belong to only one pair"
        seen_station_ids.add(a)
        seen_station_ids.add(b)
        pairs.append((a, b))

    out = g.copy()
    for a, b in pairs:
        color_a = int(stations[a]["marker"])
        color_b = int(stations[b]["marker"])
        for r, c in stations[a]["edge_cells"]:
            out[r, c] = color_b
        for r, c in stations[b]["edge_cells"]:
            out[r, c] = color_a
    return out


def _place_patch(canvas: np.ndarray, patch: np.ndarray) -> bool:
    ph, pw = patch.shape
    for _ in range(PLACE_ATTEMPTS):
        top = random.randint(1, WORK_H - ph - 1)
        left = random.randint(1, WORK_W - pw - 1)
        r0 = top - 1
        r1 = top + ph + 1
        c0 = left - 1
        c1 = left + pw + 1
        if np.any(canvas[r0:r1, c0:c1] != BG):
            continue
        region = canvas[top : top + ph, left : left + pw]
        mask = patch != BG
        region[mask] = patch[mask]
        return True
    return False


def _crop_canvas(canvas: np.ndarray) -> np.ndarray:
    coords = np.argwhere(canvas != BG)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    top_margin = random.randint(1, min(4, int(r0)))
    bottom_margin = random.randint(1, min(4, int(canvas.shape[0] - 1 - r1)))
    left_margin = random.randint(1, min(4, int(c0)))
    right_margin = random.randint(1, min(4, int(canvas.shape[1] - 1 - c1)))
    return canvas[
        int(r0) - top_margin : int(r1) + bottom_margin + 1,
        int(c0) - left_margin : int(c1) + right_margin + 1,
    ].copy()


def _build_scene() -> np.ndarray | None:
    pair_count = random.randint(1, 3)
    max_isolated = max(0, 6 - 2 * pair_count)
    isolated_count = 0
    if max_isolated > 0 and random.random() < 0.5:
        isolated_count = random.randint(1, min(2, max_isolated))

    total_stations = 2 * pair_count + isolated_count
    colors = random.sample(MARKER_COLORS, total_stations)
    patches: list[np.ndarray] = []

    idx = 0
    for _ in range(pair_count):
        patches.append(_pair_patch(colors[idx], colors[idx + 1]))
        idx += 2
    for _ in range(isolated_count):
        patches.append(_isolated_patch(colors[idx]))
        idx += 1

    random.shuffle(patches)
    patches.sort(key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)

    canvas = np.full((WORK_H, WORK_W), BG, dtype=int)
    for patch in patches:
        if not _place_patch(canvas, patch):
            return None
    return _crop_canvas(canvas)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_scene()
        if grid is None:
            continue
        try:
            out = main(grid)
        except Exception:
            continue
        if np.array_equal(out, grid):
            continue
        return grid
    raise RuntimeError("failed to generate valid input")
