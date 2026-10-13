from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

try:
    from arc_utils import find_components, neighbors4
except Exception:
    def neighbors4(r: int, c: int):
        yield r - 1, c
        yield r + 1, c
        yield r, c - 1
        yield r, c + 1

    def find_components(
        grid: np.ndarray,
        *,
        bg: int | None = None,
        connectivity: int = 4,
        monochromatic: bool = True,
        colors=None,
        min_area: int = 1,
        return_format: str = "dict",
    ):
        if connectivity != 4:
            raise ValueError("fallback find_components supports only 4-connectivity")
        if return_format != "dict":
            raise ValueError("fallback find_components supports only return_format='dict'")
        if bg is None:
            bg = int(np.bincount(grid.ravel()).argmax())
        if not monochromatic:
            raise ValueError("fallback find_components supports only monochromatic=True")
        if colors is None:
            palette = sorted(set(int(x) for x in grid.ravel()) - {int(bg)})
        else:
            palette = [int(x) for x in colors if int(x) != int(bg)]
        seen = np.zeros(grid.shape, dtype=bool)
        out = []
        for color in palette:
            coords = np.argwhere(grid == color)
            for r0, c0 in coords:
                r0 = int(r0)
                c0 = int(c0)
                if seen[r0, c0]:
                    continue
                stack = [(r0, c0)]
                seen[r0, c0] = True
                comp = []
                while stack:
                    r, c = stack.pop()
                    comp.append((r, c))
                    for nr, nc in neighbors4(r, c):
                        if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                            if not seen[nr, nc] and int(grid[nr, nc]) == color:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
                if len(comp) < int(min_area):
                    continue
                arr = np.array(comp, dtype=int)
                rmin = int(arr[:, 0].min())
                rmax = int(arr[:, 0].max())
                cmin = int(arr[:, 1].min())
                cmax = int(arr[:, 1].max())
                out.append(
                    {
                        "color": color,
                        "coords": arr,
                        "bbox": (rmin, cmin, rmax, cmax),
                        "area": len(comp),
                    }
                )
        return out


BG = 0
RED = 2
GREEN = 3
GREY = 5
MAROON = 9

SIDES = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE = {
    "up": "down",
    "down": "up",
    "left": "right",
    "right": "left",
}

GRID_MIN = 12
GRID_MAX = 16
GEN_RETRIES = 600
OBJECT_TRIES = 80
GROW_STEPS = 40


def _get_bg(grid: np.ndarray) -> int:
    return int(np.bincount(grid.ravel()).argmax())


def _coords_set(coords: np.ndarray | list[tuple[int, int]]) -> set[tuple[int, int]]:
    if isinstance(coords, np.ndarray):
        return {tuple(map(int, xy)) for xy in coords.tolist()}
    return {tuple(map(int, xy)) for xy in coords}


def _components_of_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    comps = find_components(
        grid,
        bg=_get_bg(grid),
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    return [comp["coords"].astype(int, copy=False) for comp in comps]


def _contact_side(coords: np.ndarray, marker_mask: np.ndarray) -> str:
    found = set()
    for r, c in coords:
        rr = int(r)
        cc = int(c)
        for side, (dr, dc) in SIDES.items():
            nr = rr + dr
            nc = cc + dc
            if 0 <= nr < marker_mask.shape[0] and 0 <= nc < marker_mask.shape[1]:
                if bool(marker_mask[nr, nc]):
                    found.add(side)
    assert len(found) == 1, "each object must touch the marker on exactly one side"
    return next(iter(found))


def _steps_to_edge(coords: np.ndarray, move_side: str, shape: tuple[int, int]) -> int:
    if move_side == "up":
        return int(coords[:, 0].min())
    if move_side == "down":
        return int(shape[0] - 1 - coords[:, 0].max())
    if move_side == "left":
        return int(coords[:, 1].min())
    return int(shape[1] - 1 - coords[:, 1].max())


def _shift_coords(coords: np.ndarray, move_side: str, steps: int) -> np.ndarray:
    dr, dc = SIDES[move_side]
    return coords + np.array([dr * steps, dc * steps], dtype=int)


def _landing_anchor(coords: np.ndarray, move_side: str) -> tuple[int, int]:
    pts = np.asarray(coords, dtype=int)
    if move_side == "up":
        frontier = pts[pts[:, 0] == pts[:, 0].min()]
    elif move_side == "down":
        frontier = pts[pts[:, 0] == pts[:, 0].max()]
    elif move_side == "left":
        frontier = pts[pts[:, 1] == pts[:, 1].min()]
    else:
        frontier = pts[pts[:, 1] == pts[:, 1].max()]
    best = sorted((int(r), int(c)) for r, c in frontier.tolist())[0]
    return best


def _path_clear(grid: np.ndarray, coords: np.ndarray, move_side: str, steps: int) -> bool:
    own = _coords_set(coords)
    dr, dc = SIDES[move_side]
    for step in range(1, steps + 1):
        for r, c in coords:
            nr = int(r) + dr * step
            nc = int(c) + dc * step
            if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                return False
            if (nr, nc) in own:
                continue
            if int(grid[nr, nc]) != BG:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    """
    Push each grey object away from the maroon marker, then recolor it from the landing checkerboard parity.

    Input structure:
    - The background is black, with two legend pixels at `(0,0)=red` and `(0,1)=green`.
    - A single maroon 4-connected marker blob sits near the middle of the grid.
    - Several grey 4-connected foreground objects lie on the four cardinal sides of the marker.
    - Every grey object touches the marker on exactly one cardinal side, and its outward path to the border is clear.

    Rule:
    - For each grey object, detect which side of the object touches the maroon marker.
    - Move that object straight in the opposite direction until its front edge reaches the corresponding grid boundary.
    - Choose the landing anchor pixel: among the moved object's pixels closest to the target boundary, pick the one with smallest row, then smallest column.
    - Recolor the whole moved object red if `(anchor_row + anchor_col) % 2 == 0`, otherwise recolor it green.
    - Keep the maroon marker and the two legend pixels unchanged.

    Key insight:
    - The marker contact side determines motion, but the final color is not tied to the object itself: it is decoded from the moved landing position on the grid checkerboard.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert int(g[0, 0]) == RED and int(g[0, 1]) == GREEN, "legend pixels must stay at the top-left"

    marker_mask = (g == MAROON)
    assert bool(marker_mask.any()), "expected a maroon marker blob"
    marker_components = _components_of_color(g, MAROON)
    assert len(marker_components) == 1, "expected exactly one maroon marker component"

    objects = _components_of_color(g, GREY)
    assert objects, "expected at least one grey foreground object"

    out = g.copy()
    out[g == GREY] = BG
    planned: list[tuple[int, np.ndarray]] = []
    for coords in objects:
        side = _contact_side(coords, marker_mask)
        move_side = OPPOSITE[side]
        steps = _steps_to_edge(coords, move_side, g.shape)
        assert steps > 0, "objects must start away from the destination border"
        assert _path_clear(g, coords, move_side, steps), "each outward path must be unobstructed"
        shifted = _shift_coords(coords, move_side, steps)
        anchor = _landing_anchor(shifted, move_side)
        color = RED if ((anchor[0] + anchor[1]) % 2 == 0) else GREEN
        planned.append((color, shifted))

    for color, coords in planned:
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
            assert int(out[rr, cc]) == BG, "moved objects must not overlap markers, legend, or each other"
        out[coords[:, 0], coords[:, 1]] = int(color)

    return out


def _normalize(cells: set[tuple[int, int]]) -> set[tuple[int, int]]:
    rmin = min(r for r, _ in cells)
    cmin = min(c for _, c in cells)
    return {(r - rmin, c - cmin) for r, c in cells}


def _random_polyomino(area: int, rng: random.Random, h_max: int, w_max: int) -> set[tuple[int, int]] | None:
    for _ in range(120):
        cells = {(0, 0)}
        while len(cells) < area:
            frontier = []
            for r, c in list(cells):
                for nr, nc in neighbors4(r, c):
                    frontier.append((int(nr), int(nc)))
            frontier = [pt for pt in set(frontier) if pt not in cells]
            if not frontier:
                break
            pick = rng.choice(frontier)
            cells.add(pick)
            norm = _normalize(cells)
            h = max(r for r, _ in norm) + 1
            w = max(c for _, c in norm) + 1
            if h > h_max or w > w_max:
                cells.remove(pick)
        if len(cells) != area:
            continue
        norm = _normalize(cells)
        h = max(r for r, _ in norm) + 1
        w = max(c for _, c in norm) + 1
        if 2 <= h <= h_max and 2 <= w <= w_max:
            return norm
    return None


def _place_marker(grid: np.ndarray, rng: random.Random) -> set[tuple[int, int]] | None:
    area = rng.randint(4, 8)
    shape = _random_polyomino(area, rng, h_max=4, w_max=4)
    if shape is None:
        return None
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    r0_lo = max(3, grid.shape[0] // 2 - 2)
    r0_hi = min(grid.shape[0] - h - 3, grid.shape[0] // 2 + 1)
    c0_lo = max(3, grid.shape[1] // 2 - 2)
    c0_hi = min(grid.shape[1] - w - 3, grid.shape[1] // 2 + 1)
    if r0_lo > r0_hi or c0_lo > c0_hi:
        return None
    r0 = rng.randint(r0_lo, r0_hi)
    c0 = rng.randint(c0_lo, c0_hi)
    placed = {(r0 + r, c0 + c) for r, c in shape}
    for r, c in placed:
        grid[r, c] = MAROON
    return placed


def _candidate_cells(marker: set[tuple[int, int]], shape: tuple[int, int], side: str) -> list[tuple[int, int]]:
    out = set()
    for r, c in marker:
        dr, dc = SIDES[side]
        nr = r - dr
        nc = c - dc
        if 0 <= nr < shape[0] and 0 <= nc < shape[1] and (nr, nc) not in marker:
            out.add((nr, nc))
    if side in ("up", "down"):
        return sorted(out, key=lambda pt: (pt[1], pt[0]))
    return sorted(out, key=lambda pt: (pt[0], pt[1]))


def _allowed_cell(side: str, seed: tuple[int, int], cell: tuple[int, int], depth: int, span: int) -> bool:
    r, c = cell
    sr, sc = seed
    if side == "up":
        return sr <= r <= sr + depth and abs(c - sc) <= span
    if side == "down":
        return sr - depth <= r <= sr and abs(c - sc) <= span
    if side == "left":
        return sc <= c <= sc + depth and abs(r - sr) <= span
    return sc - depth <= c <= sc and abs(r - sr) <= span


def _touches_other_object(
    cell: tuple[int, int],
    gray_cells: set[tuple[int, int]],
    own_cells: set[tuple[int, int]],
) -> bool:
    for nr, nc in neighbors4(*cell):
        if (int(nr), int(nc)) in gray_cells and (int(nr), int(nc)) not in own_cells:
            return True
    return False


def _build_object(
    seed: tuple[int, int],
    side: str,
    grid: np.ndarray,
    marker_mask: np.ndarray,
    gray_cells: set[tuple[int, int]],
    rng: random.Random,
) -> np.ndarray | None:
    target_area = rng.randint(3, 6)
    depth = rng.randint(1, 3)
    span = rng.randint(1, 2)
    cells = {seed}
    for _ in range(GROW_STEPS):
        if len(cells) >= target_area:
            break
        frontier = []
        for r, c in list(cells):
            for nr, nc in neighbors4(r, c):
                pt = (int(nr), int(nc))
                if pt in cells or pt in gray_cells:
                    continue
                if not (0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]):
                    continue
                if int(grid[nr, nc]) != BG:
                    continue
                if not _allowed_cell(side, seed, pt, depth, span):
                    continue
                if _touches_other_object(pt, gray_cells, cells):
                    continue
                frontier.append(pt)
        if not frontier:
            break
        cells.add(rng.choice(frontier))
    if len(cells) != target_area:
        return None
    coords = np.array(sorted(cells), dtype=int)
    try:
        if _contact_side(coords, marker_mask) != side:
            return None
    except AssertionError:
        return None
    move_side = OPPOSITE[side]
    steps = _steps_to_edge(coords, move_side, grid.shape)
    if steps <= 0 or not _path_clear(grid, coords, move_side, steps):
        return None
    return coords


def _semantic_metadata(grid: np.ndarray) -> list[dict]:
    marker_mask = (grid == MAROON)
    objects = _components_of_color(grid, GREY)
    info = []
    for coords in objects:
        side = _contact_side(coords, marker_mask)
        move_side = OPPOSITE[side]
        steps = _steps_to_edge(coords, move_side, grid.shape)
        shifted = _shift_coords(coords, move_side, steps)
        anchor_before = _landing_anchor(coords, move_side)
        anchor_after = _landing_anchor(shifted, move_side)
        color = RED if ((anchor_after[0] + anchor_after[1]) % 2 == 0) else GREEN
        info.append(
            {
                "coords": coords,
                "side": side,
                "move_side": move_side,
                "steps": steps,
                "anchor_before": anchor_before,
                "anchor_after": anchor_after,
                "color": color,
                "shifted": shifted,
            }
        )
    return info


def _expected_output_from_metadata(grid: np.ndarray, info: list[dict]) -> np.ndarray:
    out = grid.copy()
    out[out == GREY] = BG
    for item in info:
        coords = item["shifted"]
        color = int(item["color"])
        for r, c in coords:
            rr = int(r)
            cc = int(c)
            if not (0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]):
                raise AssertionError("shifted object leaves the grid")
            if int(out[rr, cc]) != BG:
                raise AssertionError("shifted objects overlap")
        out[coords[:, 0], coords[:, 1]] = color
    return out


def _valid_example(grid: np.ndarray) -> bool:
    if grid.shape[0] > 30 or grid.shape[1] > 30:
        return False
    if int(grid[0, 0]) != RED or int(grid[0, 1]) != GREEN:
        return False
    if len(_components_of_color(grid, MAROON)) != 1:
        return False
    marker_area = int(np.sum(grid == MAROON))
    if not (4 <= marker_area <= 8):
        return False
    objects = _components_of_color(grid, GREY)
    if len(objects) < 3:
        return False
    if any(not (3 <= len(comp) <= 6) for comp in objects):
        return False
    try:
        info = _semantic_metadata(grid)
        expected = _expected_output_from_metadata(grid, info)
        actual = main(grid)
    except AssertionError:
        return False
    if not np.array_equal(expected, actual):
        return False

    moved_colors = [int(item["color"]) for item in info]
    if RED not in moved_colors or GREEN not in moved_colors:
        return False
    if not any(int(item["steps"]) % 2 == 0 for item in info):
        return False
    if not any(int(item["steps"]) % 2 == 1 for item in info):
        return False
    if len({item["move_side"] for item in info}) < 2:
        return False

    side_groups: dict[str, list[int]] = {}
    for item in info:
        side_groups.setdefault(item["move_side"], []).append(int(item["color"]))
    if not any(len(colors) >= 2 and len(set(colors)) >= 2 for colors in side_groups.values()):
        return False

    if not any(
        ((item["anchor_before"][0] + item["anchor_before"][1]) % 2)
        != ((item["anchor_after"][0] + item["anchor_after"][1]) % 2)
        for item in info
    ):
        return False

    shifted_sets = [_coords_set(item["shifted"]) for item in info]
    for i in range(len(shifted_sets)):
        for j in range(i + 1, len(shifted_sets)):
            if shifted_sets[i] & shifted_sets[j]:
                return False
    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    height = rng.randint(GRID_MIN, GRID_MAX)
    width = rng.randint(GRID_MIN, GRID_MAX)
    grid = np.zeros((height, width), dtype=int)
    grid[0, 0] = RED
    grid[0, 1] = GREEN

    marker = _place_marker(grid, rng)
    if marker is None:
        return None
    marker_mask = (grid == MAROON)

    counts = {side: 0 for side in SIDES}
    primary_side = rng.choice(list(SIDES))
    counts[primary_side] = 2
    other_sides = [side for side in SIDES if side != primary_side]
    rng.shuffle(other_sides)
    for side in other_sides[: rng.randint(1, 2)]:
        counts[side] += 1
    if sum(counts.values()) < 3:
        return None

    gray_cells: set[tuple[int, int]] = set()
    placed_by_side: dict[str, list[np.ndarray]] = {side: [] for side in SIDES}
    for side in rng.sample(list(SIDES), k=4):
        need = counts[side]
        if need == 0:
            continue
        cands = _candidate_cells(marker, grid.shape, side)
        if len(cands) < need:
            return None
        for _ in range(need):
            placed = None
            for _ in range(OBJECT_TRIES):
                seed = rng.choice(cands)
                if seed in gray_cells or int(grid[seed]) != BG:
                    continue
                if _touches_other_object(seed, gray_cells, {seed}):
                    continue
                coords = _build_object(seed, side, grid, marker_mask, gray_cells, rng)
                if coords is None:
                    continue
                if any(
                    abs(int(seed[1] - other[:, 1].mean())) < 2
                    if side in ("up", "down")
                    else abs(int(seed[0] - other[:, 0].mean())) < 2
                    for other in placed_by_side[side]
                ):
                    continue
                for r, c in coords:
                    grid[int(r), int(c)] = GREY
                    gray_cells.add((int(r), int(c)))
                placed_by_side[side].append(coords)
                placed = coords
                break
            if placed is None:
                return None

    if not _valid_example(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_RETRIES} attempts")
