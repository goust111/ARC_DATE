from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np

_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components, paste, random_free_location


_MOTIF_COLOR_POOL = [
    Color.BLUE,
    Color.RED,
    Color.GREEN,
    Color.YELLOW,
    Color.PINK,
    Color.TEAL,
    Color.MAROON,
]
_MAX_GENERATION_ATTEMPTS = 300

_CONTAINER_OUTER = [
    (0, 0),
    (0, 1),
    (0, 2),
    (0, 3),
    (0, 4),
    (1, 0),
    (2, 0),
    (3, 0),
    (4, 0),
]
_CONTAINER_INNER = [(2, 2), (2, 3), (3, 2)]


def _bbox_area(box: tuple[int, int, int, int]) -> int:
    r0, c0, r1, c1 = box
    return (r1 - r0 + 1) * (c1 - c0 + 1)


def _row_major_key(coords: np.ndarray) -> tuple[int, int]:
    top_r = int(coords[:, 0].min())
    top_c = int(coords[coords[:, 0] == top_r, 1].min())
    return top_r, top_c


def _strictly_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    return outer[0] < inner[0] and outer[1] < inner[1] and outer[2] > inner[2] and outer[3] > inner[3]


def _orange_components(grid: np.ndarray) -> list[dict[str, object]]:
    body = np.asarray(grid[1:], dtype=int)
    raw = find_components(
        body,
        bg=Color.BLACK,
        colors=[Color.ORANGE],
        connectivity=4,
        monochromatic=True,
    )
    comps: list[dict[str, object]] = []
    for comp in raw:
        coords = comp["coords"].copy()
        coords[:, 0] += 1
        r0, c0, r1, c1 = map(int, comp["bbox"])
        comps.append(
            {
                "coords": coords,
                "bbox": (r0 + 1, c0, r1 + 1, c1),
                "area": int(comp["area"]),
            }
        )
    return comps


def _container_map(components: list[dict[str, object]]) -> dict[int, int]:
    parents: dict[int, int] = {}
    for child_idx, child in enumerate(components):
        child_box = child["bbox"]
        containers = [
            outer_idx
            for outer_idx, outer in enumerate(components)
            if outer_idx != child_idx and _strictly_contains(outer["bbox"], child_box)
        ]
        if containers:
            containers.sort(key=lambda idx: _bbox_area(components[idx]["bbox"]))
            parents[child_idx] = containers[0]
    return parents


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the orange components in row-major order using the top-row motif, except bbox-contained components inherit their container's color.

    Input structure:
    - The top row is a gray strip with 2-4 distinct colored motif pixels embedded in it.
    - Rows below contain several orange 4-connected components on a black background.
    - Some smaller orange components have bounding boxes strictly contained inside a larger component's bounding box.

    Rule:
    - Read the non-gray top-row pixels from left to right to obtain a cyclic color motif.
    - Find every orange component below the top row and sort them by row-major position using each component's topmost pixel, then its leftmost pixel on that row.
    - Each component without a container consumes the next motif color cyclically.
    - Each component whose bounding box is strictly inside another component's bounding box inherits the already-assigned color of that container and does not advance the motif index.
    - Keep the top row unchanged.

    Key insight:
    - The cycle runs over the ordered component list, but bbox containment removes some components from the count and turns them into color satellites of their container.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.shape[0] >= 2 and g.shape[1] >= 2

    motif = [int(g[0, c]) for c in range(g.shape[1]) if int(g[0, c]) != Color.GREY]
    assert 2 <= len(motif) <= 4
    assert len(motif) == len(set(motif))

    components = _orange_components(g)
    assert components, "expected at least one orange component"
    parents = _container_map(components)

    order = sorted(range(len(components)), key=lambda idx: _row_major_key(components[idx]["coords"]))

    out = g.copy()
    assigned: dict[int, int] = {}
    motif_index = 0
    for idx in order:
        if idx in parents:
            parent = parents[idx]
            assert parent in assigned, "container must be colored before its contained component"
            color = assigned[parent]
        else:
            color = motif[motif_index % len(motif)]
            motif_index += 1
        assigned[idx] = color
        coords = components[idx]["coords"]
        out[coords[:, 0], coords[:, 1]] = color

    return out


def _sample_motif_columns(width: int, count: int) -> list[int]:
    for _ in range(100):
        cols = sorted(random.sample(range(width), count))
        if all(b - a >= 2 for a, b in zip(cols, cols[1:])):
            return cols
    return sorted(random.sample(range(width), count))


def _render_patch(coords: list[tuple[int, int]], color: int = Color.ORANGE) -> np.ndarray:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, c0 = min(rows), min(cols)
    r1, c1 = max(rows), max(cols)
    patch = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _container_patch() -> np.ndarray:
    base = np.zeros((5, 5), dtype=int)
    for r, c in _CONTAINER_OUTER + _CONTAINER_INNER:
        base[r, c] = Color.ORANGE
    base = np.rot90(base, random.randrange(4))
    if random.random() < 0.5:
        base = np.fliplr(base)
    return base


def _grow_shape(size: int) -> list[tuple[int, int]] | None:
    cells = {(0, 0)}
    frontier = {(1, 0), (-1, 0), (0, 1), (0, -1)}
    while len(cells) < size:
        if not frontier:
            return None
        r, c = random.choice(tuple(frontier))
        frontier.remove((r, c))
        cells.add((r, c))
        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            nbr = (r + dr, c + dc)
            if nbr not in cells:
                frontier.add(nbr)

    rows = [r for r, _c in cells]
    cols = [c for _r, c in cells]
    if max(rows) - min(rows) > 4 or max(cols) - min(cols) > 4:
        return None
    return list(cells)


def _solo_patch() -> np.ndarray:
    for _ in range(100):
        size = random.randint(3, 8)
        coords = _grow_shape(size)
        if coords is None:
            continue
        patch = _render_patch(coords)
        if patch.shape == (5, 5):
            continue
        return patch
    raise RuntimeError("failed to sample a solo component")


def _reserve_and_place(reserve: np.ndarray, body: np.ndarray, patch: np.ndarray) -> bool:
    reserve_patch = np.ones(patch.shape, dtype=int)
    try:
        r0, c0 = random_free_location(
            reserve,
            reserve_patch,
            transparent=Color.BLACK,
            border=0,
            padding=1,
            padding_connectivity=8,
        )
    except ValueError:
        return False

    paste(reserve, reserve_patch, r0, c0)
    paste(body, patch, r0, c0, transparent=Color.BLACK)
    return True


def _input_is_valid(grid: np.ndarray, pair_count: int, total_components: int, motif_len: int) -> bool:
    try:
        comps = _orange_components(grid)
        parents = _container_map(comps)
        if len(comps) != total_components or len(parents) != pair_count:
            return False
        if any(parent in parents for parent in parents.values()):
            return False

        order = sorted(range(len(comps)), key=lambda idx: _row_major_key(comps[idx]["coords"]))
        seen_uncontained = 0
        assigned: set[int] = set()
        motif = [int(grid[0, c]) for c in range(grid.shape[1]) if int(grid[0, c]) != Color.GREY]
        for idx in order:
            if idx in parents:
                parent = parents[idx]
                if parent not in assigned:
                    return False
            else:
                seen_uncontained += 1
            assigned.add(idx)
        if seen_uncontained <= motif_len:
            return False
        _ = main(grid)
    except AssertionError:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    motif_len = random.randint(2, 4)
    pair_count = random.choices([1, 2], weights=[5, 4], k=1)[0]
    min_uncontained = max(motif_len + 1, 5 - pair_count)
    uncontained_count = random.randint(min_uncontained, 6)
    solo_count = uncontained_count - pair_count
    total_components = uncontained_count + pair_count

    height = random.randint(13, 19)
    width = random.randint(max(14, 4 * (pair_count + solo_count)), 24)

    grid = np.zeros((height, width), dtype=int)
    grid[0, :] = Color.GREY

    motif_colors = random.sample(_MOTIF_COLOR_POOL, motif_len)
    for c, color in zip(_sample_motif_columns(width, motif_len), motif_colors):
        grid[0, c] = color

    body = np.zeros((height - 1, width), dtype=int)
    reserve = np.zeros_like(body)

    patches = [_container_patch() for _ in range(pair_count)]
    patches.extend(_solo_patch() for _ in range(solo_count))
    random.shuffle(patches)

    for patch in patches:
        if not _reserve_and_place(reserve, body, patch):
            return None

    grid[1:, :] = body
    if not _input_is_valid(grid, pair_count, total_components, motif_len):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input for puzzle 7")
