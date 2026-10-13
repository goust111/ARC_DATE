from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
FRAME = 5
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
COMPONENT_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
FRAME_SIDES = [5, 7, 9, 11]
MAX_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_monochrome_components(
    grid: np.ndarray, ignored: set[int] | None = None
) -> list[dict]:
    ignored = set() if ignored is None else set(int(v) for v in ignored)
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color in ignored or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            cells = []
            while q:
                cr, cc = q.popleft()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            coords = np.asarray(cells, dtype=int)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(len(cells)),
                }
            )
    return comps


def _is_hollow_square_frame(comp: dict) -> bool:
    if int(comp["color"]) != FRAME:
        return False
    coords = comp["coords"]
    r0, c0, r1, c1 = comp["bbox"]
    side_h = r1 - r0 + 1
    side_w = c1 - c0 + 1
    if side_h != side_w or side_h < 5:
        return False
    expected = 4 * side_h - 4
    if int(comp["area"]) != expected:
        return False
    cells = {tuple(map(int, p)) for p in coords.tolist()}
    for rr in range(r0, r1 + 1):
        for cc in range(c0, c1 + 1):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if on_border != ((rr, cc) in cells):
                return False
    return True


def _frame_border_coords(frame: dict) -> np.ndarray:
    r0, c0, r1, c1 = frame["bbox"]
    cells = []
    for rr in range(r0, r1 + 1):
        cells.append((rr, c0))
        cells.append((rr, c1))
    for cc in range(c0 + 1, c1):
        cells.append((r0, cc))
        cells.append((r1, cc))
    return np.asarray(cells, dtype=int)


def _component_to_frame_distance(comp_coords: np.ndarray, frame_coords: np.ndarray) -> int:
    best = None
    for pr, pc in comp_coords:
        dists = np.abs(frame_coords[:, 0] - int(pr)) + np.abs(frame_coords[:, 1] - int(pc))
        cur = int(dists.min())
        if best is None or cur < best:
            best = cur
    assert best is not None
    return int(best)


def _fill_frame_interior(
    out: np.ndarray, bbox: tuple[int, int, int, int], depth: int, color: int
) -> None:
    r0, c0, r1, c1 = bbox
    side = r1 - r0 + 1
    max_depth = side // 2
    depth = min(int(depth), max_depth)
    for rr in range(r0 + 1, r1):
        for cc in range(c0 + 1, c1):
            shell = min(rr - r0, r1 - rr, cc - c0, c1 - cc)
            if shell == 0:
                continue
            if shell <= depth and shell % 2 == 1:
                out[rr, cc] = int(color)
            else:
                out[rr, cc] = BG


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow gray square frame with color-coded concentric interior shells whose depth is set by the nearest outside component's area.

    Input structure:
    - The background is black.
    - There are 2-3 separate one-cell-thick gray hollow square frames with black interiors.
    - There are 2-3 scattered monochrome 4-connected components outside the frames; each component has one non-gray color and a small pixel area.

    Rule:
    - Detect every gray hollow square frame and every non-gray monochrome component.
    - For each frame, find the uniquely nearest component by minimum Manhattan distance from any component pixel to any frame-border pixel.
    - Let N be that component's area and let its color be the fill color; inside the frame, paint N concentric interior shells from outside to inside, using the component color on odd-numbered shells and black on even-numbered shells, clipped to the frame's maximum shell capacity.
    - Keep the gray frame borders and all scattered components unchanged.

    Key insight:
    - Each scattered component encodes two visible parameters at once: its color chooses the colored shell paint, and its pixel count chooses how many interior shells the nearest frame receives.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    comps = _find_monochrome_components(g, ignored={BG})
    frames = [comp for comp in comps if _is_hollow_square_frame(comp)]
    objects = [comp for comp in comps if int(comp["color"]) != FRAME]
    assert 2 <= len(frames) <= 3
    assert len(objects) >= len(frames)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        assert np.all(g[r0 + 1 : r1, c0 + 1 : c1] == BG)

    frame_coords = [_frame_border_coords(frame) for frame in frames]
    out = g.copy()
    for frame, border_coords in zip(frames, frame_coords):
        distances = [
            _component_to_frame_distance(obj["coords"], border_coords) for obj in objects
        ]
        best = min(distances)
        winners = [obj for obj, dist in zip(objects, distances) if dist == best]
        assert len(winners) == 1
        obj = winners[0]
        _fill_frame_interior(out, frame["bbox"], int(obj["area"]), int(obj["color"]))
    return out


def _make_component_patch(area: int, color: int) -> np.ndarray:
    cells = {(0, 0)}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for nr, nc in _neighbors4(r, c):
                if (nr, nc) not in cells:
                    frontier.add((nr, nc))
        cells.add(random.choice(tuple(frontier)))

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    min_r, min_c = min(rows), min(cols)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    h = max(r for r, _ in norm) + 1
    w = max(c for _, c in norm) + 1
    patch = np.zeros((h, w), dtype=int)
    for r, c in norm:
        patch[r, c] = int(color)

    for _ in range(random.randint(0, 3)):
        patch = np.rot90(patch)
    if random.random() < 0.5:
        patch = np.fliplr(patch)
    if random.random() < 0.5:
        patch = np.flipud(patch)
    return patch.copy()


def _draw_frame(grid: np.ndarray, r0: int, c0: int, side: int) -> None:
    r1 = r0 + side - 1
    c1 = c0 + side - 1
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _place_frames(
    height: int, width: int, sides: list[int]
) -> list[dict] | None:
    frames: list[dict] = []
    for side in sorted(sides, reverse=True):
        candidates = []
        for r0 in range(0, height - side + 1):
            for c0 in range(0, width - side + 1):
                box = (r0, c0, r0 + side - 1, c0 + side - 1)
                if all(_boxes_separated(box, old["bbox"], gap=2) for old in frames):
                    candidates.append((r0, c0, box))
        if not candidates:
            return None
        r0, c0, box = random.choice(candidates)
        frames.append({"bbox": box, "side": side, "coords": _frame_border_coords({"bbox": box})})
    random.shuffle(frames)
    return frames


def _patch_cells_at(patch: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = np.argwhere(patch != BG)
    return [(top + int(r), left + int(c)) for r, c in coords]


def _manhattan_to_bbox(cell: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    r, c = cell
    r0, c0, r1, c1 = bbox
    if r0 <= r <= r1 and c0 <= c <= c1:
        return 0
    dr = 0 if r0 <= r <= r1 else min(abs(r - r0), abs(r - r1))
    dc = 0 if c0 <= c <= c1 else min(abs(c - c0), abs(c - c1))
    return int(dr + dc)


def _cell_clear_for_component(
    grid: np.ndarray, rr: int, cc: int, frame_bboxes: list[tuple[int, int, int, int]]
) -> bool:
    h, w = grid.shape
    if not (0 <= rr < h and 0 <= cc < w):
        return False
    if int(grid[rr, cc]) != BG:
        return False
    for nr in range(rr - 1, rr + 2):
        for nc in range(cc - 1, cc + 2):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == FRAME:
                return False
    for bbox in frame_bboxes:
        if _manhattan_to_bbox((rr, cc), bbox) == 0:
            return False
    return True


def _distance_matrix(
    component_cells: list[np.ndarray], frames: list[dict]
) -> np.ndarray:
    matrix = np.zeros((len(component_cells), len(frames)), dtype=int)
    for i, cells in enumerate(component_cells):
        for j, frame in enumerate(frames):
            matrix[i, j] = _component_to_frame_distance(cells, frame["coords"])
    return matrix


def _argmin_unique(values: np.ndarray) -> int | None:
    m = int(values.min())
    idxs = np.where(values == m)[0]
    if len(idxs) != 1:
        return None
    return int(idxs[0])


def _has_distance_interference(matrix: np.ndarray) -> bool:
    for row in matrix:
        ordered = np.sort(row)
        if len(ordered) >= 2 and int(ordered[1]) <= int(ordered[0]) + 3:
            return True
    return False


def _place_components(
    grid: np.ndarray, frames: list[dict], specs: list[dict]
) -> list[np.ndarray] | None:
    frame_bboxes = [frame["bbox"] for frame in frames]
    placed_cells: list[np.ndarray] = []

    for target_idx, spec in enumerate(specs):
        patch = spec["patch"]
        ph, pw = patch.shape
        candidates = []
        for r0 in range(0, grid.shape[0] - ph + 1):
            for c0 in range(0, grid.shape[1] - pw + 1):
                cells = _patch_cells_at(patch, r0, c0)
                if not all(_cell_clear_for_component(grid, rr, cc, frame_bboxes) for rr, cc in cells):
                    continue
                coords = np.asarray(cells, dtype=int)
                dists = np.array(
                    [_component_to_frame_distance(coords, frame["coords"]) for frame in frames],
                    dtype=int,
                )
                if int(dists.min()) < 2:
                    continue
                if _argmin_unique(dists) != target_idx:
                    continue
                if len(dists) >= 2 and np.partition(dists, 1)[1] > dists[target_idx] + 5:
                    continue
                candidates.append((r0, c0, coords, dists))
        if not candidates:
            return None
        r0, c0, coords, _dists = random.choice(candidates)
        for rr, cc in coords:
            grid[int(rr), int(cc)] = int(spec["color"])
        placed_cells.append(coords)

    matrix = _distance_matrix(placed_cells, frames)
    if not _has_distance_interference(matrix):
        return None
    for i in range(len(specs)):
        if _argmin_unique(matrix[i]) != i:
            return None
    for j in range(len(frames)):
        if _argmin_unique(matrix[:, j]) != j:
            return None
    return placed_cells


def _sample_sides(count: int, height: int, width: int) -> list[int]:
    choices = [5, 7, 9] if count == 3 else FRAME_SIDES
    viable = [side for side in choices if side <= min(height, width)]
    return random.sample(viable, count)


def _build_component_specs(frames: list[dict]) -> list[dict]:
    colors = random.sample(COMPONENT_COLORS, len(frames))
    specs = []
    exact_full = random.randint(0, len(frames) - 1)
    for i, (frame, color) in enumerate(zip(frames, colors)):
        max_depth = int(frame["side"] // 2)
        upper = min(5, max_depth)
        if i == exact_full or random.random() < 0.35:
            area = upper
        else:
            area = random.randint(2, upper)
        specs.append({"color": color, "area": area, "patch": _make_component_patch(area, color)})
    random.shuffle(specs)
    return specs


def _attempt_generate() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    count = random.choice([2, 3])
    sides = _sample_sides(count, height, width)
    frames = _place_frames(height, width, sides)
    if frames is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        _draw_frame(grid, r0, c0, r1 - r0 + 1)

    specs = _build_component_specs(frames)
    placed = _place_components(grid, frames, specs)
    if placed is None:
        return None

    if len(_find_monochrome_components(grid, ignored={BG, FRAME})) != count:
        return None
    try:
        out = main(grid)
    except Exception:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle 6 input")
