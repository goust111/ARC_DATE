from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


Coord = tuple[int, int]
Shape = tuple[int, int]

PANEL_SIZE = 7
TOP_ONLY_HOLES: tuple[Coord, ...] = (
    (1, 1),
    (1, 4),
    (2, 5),
    (4, 1),
    (5, 2),
    (5, 5),
)
BOTTOM_ONLY_HOLES: tuple[Coord, ...] = (
    (1, 2),
    (1, 5),
    (2, 1),
    (4, 5),
    (5, 1),
    (5, 4),
)
TIE_ORDER = ("up", "right", "down", "left")
ALT_TIE_ORDER = ("left", "up", "right", "down")
DIR_VECS = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}
CORE_MARKERS: tuple[Coord, ...] = ((3, 3), (2, 3))
TIE_MARKERS: tuple[Coord, ...] = ((2, 2), (2, 4), (4, 2), (4, 4))
OPEN_MARKERS: tuple[Coord, ...] = (
    (1, 2),
    (1, 4),
    (2, 1),
    (2, 5),
    (3, 2),
    (3, 4),
    (4, 1),
    (4, 3),
    (4, 5),
)
FILLER_MARKERS: tuple[Coord, ...] = tuple(
    (r, c)
    for r in range(1, PANEL_SIZE - 1)
    for c in range(1, PANEL_SIZE - 1)
    if (r, c) != (1, 3)
)
GEN_RETRIES = 400


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] % 2 == 0, "expected two stacked panels"
    half = g.shape[0] // 2
    top = g[:half]
    bottom = g[half:]
    assert top.shape == bottom.shape, "panels must have equal shape"
    return top, bottom


def _marker_set_from_grid(grid: np.ndarray) -> set[Coord]:
    top, bottom = _split_panels(grid)
    shared_bg = (top == Color.BLACK) & (bottom == Color.BLACK)
    return {(int(r), int(c)) for r, c in np.argwhere(shared_bg)}


def _nearest_boundary(
    cell: Coord, shape: Shape, order: tuple[str, ...] = TIE_ORDER
) -> tuple[str, dict[str, int], tuple[str, ...]]:
    h, w = map(int, shape)
    r, c = map(int, cell)
    distances = {
        "up": r,
        "right": w - 1 - c,
        "down": h - 1 - r,
        "left": c,
    }
    best = min(distances.values())
    tied = tuple(name for name in TIE_ORDER if distances[name] == best)
    for name in order:
        if distances[name] == best:
            return name, distances, tied
    raise AssertionError("unreachable tie resolution")


def _trace_ray(
    markers: set[Coord],
    source: Coord,
    shape: Shape,
    *,
    order: tuple[str, ...] = TIE_ORDER,
    include_edge: bool = False,
    ignore_markers: bool = False,
) -> dict:
    direction, distances, tied = _nearest_boundary(source, shape, order)
    dr, dc = DIR_VECS[direction]
    h, w = map(int, shape)
    r, c = source
    path: list[Coord] = []

    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return {
                "source": source,
                "direction": direction,
                "distances": distances,
                "tied": tied,
                "path": tuple(path),
                "stop": None,
                "stop_reason": "outside",
            }
        if not ignore_markers and (r, c) in markers:
            return {
                "source": source,
                "direction": direction,
                "distances": distances,
                "tied": tied,
                "path": tuple(path),
                "stop": (r, c),
                "stop_reason": "marker",
            }
        if r in (0, h - 1) or c in (0, w - 1):
            if include_edge:
                path.append((r, c))
            return {
                "source": source,
                "direction": direction,
                "distances": distances,
                "tied": tied,
                "path": tuple(path),
                "stop": (r, c),
                "stop_reason": "edge",
            }
        path.append((r, c))


def _render_markers(
    markers: set[Coord],
    shape: Shape,
    *,
    suppress_blockers: bool = False,
    order: tuple[str, ...] = TIE_ORDER,
    include_edge: bool = False,
    ignore_markers: bool = False,
) -> tuple[np.ndarray, dict[Coord, dict], set[Coord], set[Coord]]:
    meta = {
        source: _trace_ray(
            markers,
            source,
            shape,
            order=order,
            include_edge=include_edge,
            ignore_markers=ignore_markers,
        )
        for source in markers
    }
    blockers = {
        info["stop"]
        for info in meta.values()
        if info["stop_reason"] == "marker" and info["stop"] is not None
    }
    emitters = set(markers)
    if suppress_blockers:
        emitters -= blockers

    out = np.full(shape, Color.BLACK, dtype=int)
    for r, c in markers:
        out[r, c] = Color.RED
    for source in emitters:
        for r, c in meta[source]["path"]:
            if (r, c) not in markers:
                out[r, c] = Color.ORANGE
    return out, meta, emitters, blockers


def main(grid: np.ndarray) -> np.ndarray:
    """Compress the two stacked masks into shared-gap markers, then shoot rays from every marker.

    Input structure:
    - The grid contains two equal-sized panels stacked vertically.
    - The upper panel is a blue connected mask on black, and the lower panel is a green connected mask on black.
    - Cells that are black in both panels encode the red output markers after compression.

    Rule:
    - Compare the two panels cell by cell and build an output grid of the panel shape.
    - Put a red marker at every position where both panels are black; all other output cells start black.
    - Each red marker chooses its nearest boundary by Manhattan distance, breaking ties clockwise as up > right > down > left.
    - Every marker extends an orange ray toward its chosen boundary, stopping before the boundary cell or before the next red marker on that path.

    Key insight:
    - The compressed shared-background pattern simultaneously defines the marker set, the ray sources, and the blockers that truncate other rays.
    """
    top, bottom = _split_panels(grid)
    markers = {(int(r), int(c)) for r, c in np.argwhere((top == 0) & (bottom == 0))}
    out, _meta, _emitters, _blockers = _render_markers(markers, top.shape)
    return out


def _build_input_from_markers(markers: set[Coord]) -> np.ndarray:
    top = np.full((PANEL_SIZE, PANEL_SIZE), Color.BLUE, dtype=int)
    bottom = np.full((PANEL_SIZE, PANEL_SIZE), Color.GREEN, dtype=int)

    for cell in markers:
        top[cell] = Color.BLACK
        bottom[cell] = Color.BLACK
    for cell in TOP_ONLY_HOLES:
        if cell not in markers:
            top[cell] = Color.BLACK
    for cell in BOTTOM_ONLY_HOLES:
        if cell not in markers:
            bottom[cell] = Color.BLACK
    return np.vstack([top, bottom])


def _component_ok(panel: np.ndarray, color: int) -> bool:
    comps = find_components(
        panel,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[color],
        return_format="dict",
    )
    return len(comps) == 1 and int(comps[0]["area"]) == int((panel == color).sum())


def _is_valid_instance(grid: np.ndarray) -> bool:
    top, bottom = _split_panels(grid)
    markers = _marker_set_from_grid(grid)
    if not (4 <= len(markers) <= 8):
        return False
    rows = {r for r, _c in markers}
    cols = {c for _r, c in markers}
    if len(rows) == 1 or len(cols) == 1:
        return False

    if set(map(int, np.unique(top))) != {Color.BLACK, Color.BLUE}:
        return False
    if set(map(int, np.unique(bottom))) != {Color.BLACK, Color.GREEN}:
        return False
    if not _component_ok(top, Color.BLUE) or not _component_ok(bottom, Color.GREEN):
        return False
    if np.array_equal(top == Color.BLUE, bottom == Color.GREEN):
        return False

    out, meta, emitters, blockers = _render_markers(markers, top.shape)
    if not blockers:
        return False
    if Color.ORANGE not in set(map(int, np.unique(out))):
        return False
    if int((out == Color.ORANGE).sum()) < 2:
        return False

    emitted_directions = {meta[source]["direction"] for source in emitters}
    if len(emitted_directions) < 2:
        return False
    if not any(direction in emitted_directions for direction in ("left", "right", "down")):
        return False
    if not any(len(meta[source]["tied"]) > 1 for source in emitters):
        return False
    if not any(
        meta[source]["stop_reason"] == "edge" and len(meta[source]["path"]) > 0
        for source in emitters
    ):
        return False

    alt_tie_out, _meta3, _emitters3, _blockers3 = _render_markers(
        markers, top.shape, order=ALT_TIE_ORDER
    )
    if np.array_equal(out, alt_tie_out):
        return False

    inclusive_edge_out, _meta4, _emitters4, _blockers4 = _render_markers(
        markers, top.shape, include_edge=True
    )
    if np.array_equal(out, inclusive_edge_out):
        return False

    return np.array_equal(main(grid), out)


def _sample_marker_set() -> set[Coord]:
    markers = set(CORE_MARKERS)
    markers.add(random.choice(TIE_MARKERS))
    markers.add(random.choice(OPEN_MARKERS))
    target = random.randint(4, 8)
    filler = list(FILLER_MARKERS)
    random.shuffle(filler)
    for cell in filler:
        if len(markers) >= target:
            break
        if cell in markers:
            continue
        if random.random() < 0.3:
            markers.add(cell)
    return markers


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        markers = _sample_marker_set()
        grid = _build_input_from_markers(markers)
        if _is_valid_instance(grid):
            return grid
    raise RuntimeError("failed to generate a valid stacked-mask ray puzzle")
