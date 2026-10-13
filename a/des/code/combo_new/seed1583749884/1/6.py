from __future__ import annotations

from collections import deque
from itertools import combinations
import random

import numpy as np


_STEP4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
_DIR_TO_VEC = {
    "U": (-1, 0),
    "D": (1, 0),
    "L": (0, -1),
    "R": (0, 1),
}
_VEC_TO_DIR = {vec: key for key, vec in _DIR_TO_VEC.items()}


def _infer_checkerboard(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = map(int, grid.shape)
    colors = [int(color) for color in np.unique(grid)]
    parity = np.fromfunction(lambda r, c: (r + c) % 2, (h, w), dtype=int)

    best: tuple[int, int, int, int] | None = None
    for a, b in combinations(colors, 2):
        for phase in (0, 1):
            even_color = a if phase == 0 else b
            odd_color = b if phase == 0 else a
            matches = int(np.count_nonzero(grid[parity == 0] == even_color))
            matches += int(np.count_nonzero(grid[parity == 1] == odd_color))
            candidate = (matches, int(a), int(b), int(phase))
            if best is None or candidate > best:
                best = candidate

    assert best is not None, "expected at least two colors to infer the checkerboard"
    _, a, b, phase = best
    return int(a), int(b), int(phase)


def _expected_checkerboard(
    shape: tuple[int, int], color_a: int, color_b: int, phase: int
) -> np.ndarray:
    h, w = map(int, shape)
    rows, cols = np.indices((h, w))
    use_a = ((rows + cols + int(phase)) % 2) == 0
    expected = np.empty((h, w), dtype=int)
    expected[use_a] = int(color_a)
    expected[~use_a] = int(color_b)
    return expected


def _foreground_components(grid: np.ndarray, expected_bg: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    foreground = grid != expected_bg
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r, c in np.argwhere(foreground):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        color = int(grid[r, c])
        queue = deque([(r, c)])
        seen[r, c] = True
        coords: list[tuple[int, int]] = []

        while queue:
            rr, cc = queue.popleft()
            coords.append((int(rr), int(cc)))
            for dr, dc in _STEP4:
                nr = int(rr + dr)
                nc = int(cc + dc)
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                if seen[nr, nc] or not foreground[nr, nc] or int(grid[nr, nc]) != color:
                    continue
                seen[nr, nc] = True
                queue.append((nr, nc))

        components.append({"color": color, "coords": coords})

    return components


def _unique_leaf(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], str]:
    coord_set = set(coords)
    leaves: list[tuple[tuple[int, int], tuple[int, int]]] = []

    for r, c in coords:
        neighbors = [
            (r + dr, c + dc)
            for dr, dc in _STEP4
            if (r + dr, c + dc) in coord_set
        ]
        if len(neighbors) == 1:
            leaves.append(((int(r), int(c)), neighbors[0]))

    assert len(leaves) == 1, "each object must have exactly one protruding leaf pixel"
    leaf, neighbor = leaves[0]
    direction = _VEC_TO_DIR[(int(leaf[0] - neighbor[0]), int(leaf[1] - neighbor[1]))]
    return leaf, direction


def _project_cells(
    direction: str, center: tuple[int, int], length: int, shape: tuple[int, int]
) -> tuple[list[tuple[int, int]], bool]:
    h, w = map(int, shape)
    if direction in {"U", "D"}:
        anchor = int(center[1])
        start = int(anchor - length // 2)
        stop = int(start + length)
        clipped = start < 0 or stop > w
        row = 0 if direction == "U" else h - 1
        cells = [(int(row), int(col)) for col in range(max(0, start), min(w, stop))]
        return cells, bool(clipped)

    anchor = int(center[0])
    start = int(anchor - length // 2)
    stop = int(start + length)
    clipped = start < 0 or stop > h
    col = 0 if direction == "L" else w - 1
    cells = [(int(row), int(col)) for row in range(max(0, start), min(h, stop))]
    return cells, bool(clipped)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Project each protrusion-marked foreground object to a matching grid edge with an area-length bar.

    Input structure:
    - The background is a two-color black/grey checkerboard.
    - Several 4-connected monochrome foreground objects sit on the checkerboard.
    - Each object has exactly one outward protruding leaf pixel that extends one
      step from the rest of the object in one cardinal direction.

    Rule:
    - Keep the input grid unchanged as the base of the output.
    - For each foreground object, find its unique degree-1 pixel and read the
      direction from the adjacent body pixel toward that protrusion.
    - Count all pixels of the object, including the protrusion itself.
    - Draw a same-color bar on the corresponding outer edge: horizontal on the
      top or bottom, vertical on the left or right, centered on the protrusion
      column or row, and clipped if it extends past the grid boundary.

    Key insight:
    - The single leaf pixel encodes direction, while the object's total area
      independently determines the length of the projected edge marker.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg_a, bg_b, phase = _infer_checkerboard(g)
    expected_bg = _expected_checkerboard(g.shape, bg_a, bg_b, phase)
    components = _foreground_components(g, expected_bg)
    assert components, "expected at least one foreground object"

    out = g.copy()
    for comp in components:
        coords = comp["coords"]
        leaf, direction = _unique_leaf(coords)
        bar_cells, _ = _project_cells(direction, leaf, len(coords), g.shape)
        for r, c in bar_cells:
            out[int(r), int(c)] = int(comp["color"])

    return out


def _mask_connected(mask: np.ndarray) -> bool:
    points = np.argwhere(mask)
    if len(points) == 0:
        return False

    h, w = map(int, mask.shape)
    start = tuple(map(int, points[0]))
    queue = deque([start])
    seen = {start}

    while queue:
        r, c = queue.popleft()
        for dr, dc in _STEP4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))

    return len(seen) == int(mask.sum())


def _min_degree(mask: np.ndarray) -> int:
    coords = [tuple(map(int, coord)) for coord in np.argwhere(mask)]
    if not coords:
        return 0
    coord_set = set(coords)
    return min(
        sum((r + dr, c + dc) in coord_set for dr, dc in _STEP4) for r, c in coords
    )


def _build_body_library() -> dict[int, tuple[np.ndarray, ...]]:
    by_area: dict[int, list[np.ndarray]] = {}

    for h in range(2, 5):
        for w in range(2, 5):
            for bits in range(1, 1 << (h * w)):
                area = int(bits.bit_count())
                if area < 4 or area > 10 or area % 2 == 1:
                    continue

                mask = np.array(
                    [(bits >> idx) & 1 for idx in range(h * w)], dtype=bool
                ).reshape(h, w)
                if not mask[0].any() or not mask[-1].any():
                    continue
                if not mask[:, 0].any() or not mask[:, -1].any():
                    continue
                if not _mask_connected(mask):
                    continue
                if _min_degree(mask) < 2:
                    continue

                by_area.setdefault(area, []).append(mask)

    assert by_area, "body library must not be empty"
    return {area: tuple(masks) for area, masks in by_area.items()}


_BODY_LIBRARY = _build_body_library()
_PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
_GEN_ATTEMPTS = 400


def _balanced_directions(count: int, rng: random.Random) -> list[str]:
    dirs = ["U", "D", "L", "R"]
    rng.shuffle(dirs)
    return dirs[:count]


def _balanced_clip_flags(count: int, rng: random.Random) -> list[bool]:
    flags = [False, True] * ((count + 1) // 2)
    rng.shuffle(flags)
    return flags[:count]


def _checkerboard_canvas(
    h: int, w: int, phase: int, colors: tuple[int, int] = (0, 5)
) -> np.ndarray:
    return _expected_checkerboard((h, w), int(colors[0]), int(colors[1]), int(phase))


def _make_object(
    body_mask: np.ndarray, direction: str, rng: random.Random
) -> dict[str, object]:
    body_coords = [tuple(map(int, coord)) for coord in np.argwhere(body_mask)]
    if direction == "U":
        anchors = [(r, c) for r, c in body_coords if r == 0]
        anchor = rng.choice(anchors)
        protrusion = (int(anchor[0] - 1), int(anchor[1]))
    elif direction == "D":
        max_row = int(body_mask.shape[0] - 1)
        anchors = [(r, c) for r, c in body_coords if r == max_row]
        anchor = rng.choice(anchors)
        protrusion = (int(anchor[0] + 1), int(anchor[1]))
    elif direction == "L":
        anchors = [(r, c) for r, c in body_coords if c == 0]
        anchor = rng.choice(anchors)
        protrusion = (int(anchor[0]), int(anchor[1] - 1))
    else:
        max_col = int(body_mask.shape[1] - 1)
        anchors = [(r, c) for r, c in body_coords if c == max_col]
        anchor = rng.choice(anchors)
        protrusion = (int(anchor[0]), int(anchor[1] + 1))

    coords = body_coords + [protrusion]
    min_row = min(r for r, _ in coords)
    min_col = min(c for _, c in coords)
    shifted = [(int(r - min_row), int(c - min_col)) for r, c in coords]
    shifted_protrusion = (int(protrusion[0] - min_row), int(protrusion[1] - min_col))
    height = max(r for r, _ in shifted) + 1
    width = max(c for _, c in shifted) + 1

    leaf, leaf_dir = _unique_leaf(shifted)
    assert leaf == shifted_protrusion and leaf_dir == direction

    return {
        "coords": shifted,
        "leaf": shifted_protrusion,
        "height": int(height),
        "width": int(width),
        "area": int(len(shifted)),
        "direction": direction,
    }


def _mark_halo(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = map(int, mask.shape)
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    mask[nr, nc] = True


def _candidate_positions(
    obj: dict[str, object],
    shape: tuple[int, int],
    occupied: np.ndarray,
    reserved_edge: set[tuple[int, int]],
    want_clip: bool,
) -> list[tuple[int, int, list[tuple[int, int]]]]:
    h, w = map(int, shape)
    obj_h = int(obj["height"])
    obj_w = int(obj["width"])
    leaf_r, leaf_c = obj["leaf"]
    coords: list[tuple[int, int]] = obj["coords"]  # type: ignore[assignment]
    direction = str(obj["direction"])
    area = int(obj["area"])

    positions: list[tuple[int, int, list[tuple[int, int]]]] = []
    for r0 in range(1, h - obj_h):
        for c0 in range(1, w - obj_w):
            global_cells = [(int(r0 + r), int(c0 + c)) for r, c in coords]
            if any(bool(occupied[r, c]) for r, c in global_cells):
                continue

            leaf = (int(r0 + leaf_r), int(c0 + leaf_c))
            bar_cells, clipped = _project_cells(direction, leaf, area, shape)
            if clipped != want_clip:
                continue
            if any(cell in reserved_edge for cell in bar_cells):
                continue

            positions.append((int(r0), int(c0), bar_cells))

    return positions


def _choose_area_options(h: int, w: int, count: int) -> list[int]:
    options = [4, 6, 8]
    if min(h, w) >= 18 and count <= 3 and 10 in _BODY_LIBRARY:
        options.append(10)
    return [area for area in options if area in _BODY_LIBRARY]


def _generate_attempt(rng: random.Random) -> np.ndarray | None:
    h = rng.randint(14, 24)
    w = rng.randint(14, 24)
    count = rng.randint(2, 4)
    phase = rng.randint(0, 1)

    colors = rng.sample(_PALETTE, count)
    directions = _balanced_directions(count, rng)
    clips = _balanced_clip_flags(count, rng)
    area_options = _choose_area_options(h, w, count)
    if not area_options:
        return None

    specs = []
    for color, direction, want_clip in zip(colors, directions, clips):
        area = rng.choice(area_options)
        body_mask = rng.choice(_BODY_LIBRARY[area])
        obj = _make_object(body_mask, direction, rng)
        specs.append(
            {
                "color": int(color),
                "want_clip": bool(want_clip),
                "object": obj,
            }
        )

    specs.sort(key=lambda spec: int(spec["object"]["area"]), reverse=True)  # type: ignore[index]

    grid = _checkerboard_canvas(h, w, phase)
    occupied = np.zeros((h, w), dtype=bool)
    reserved_edge: set[tuple[int, int]] = set()

    for spec in specs:
        obj = spec["object"]
        positions = _candidate_positions(
            obj, (h, w), occupied, reserved_edge, bool(spec["want_clip"])
        )
        if not positions:
            return None

        r0, c0, bar_cells = rng.choice(positions)
        global_cells = [
            (int(r0 + r), int(c0 + c)) for r, c in obj["coords"]  # type: ignore[index]
        ]

        for r, c in global_cells:
            grid[r, c] = int(spec["color"])
        _mark_halo(occupied, global_cells)
        reserved_edge.update(bar_cells)

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != grid.shape:
        return None
    if max(grid.shape) > 30 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(_GEN_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid projection puzzle input")
