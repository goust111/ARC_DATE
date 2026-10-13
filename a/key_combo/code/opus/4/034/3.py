from __future__ import annotations

import random
from importlib.machinery import SourceFileLoader
from pathlib import Path

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_shared",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
find_components = _ARC_UTILS.find_components

BG = Color.BLACK
AXIS = Color.RED
SPRITE = Color.GREY
RANK_COLORS = [Color.BLUE, Color.GREEN, Color.YELLOW, Color.PINK]
GEN_ATTEMPTS = 600


def _bbox_center_twice(comp: dict) -> int:
    _, c0, _, c1 = map(int, comp["bbox"])
    return int(c0 + c1)


def _normalize_coords(coords: np.ndarray) -> frozenset[tuple[int, int]]:
    coords = np.asarray(coords, dtype=int)
    r0 = int(coords[:, 0].min())
    c0 = int(coords[:, 1].min())
    return frozenset((int(r - r0), int(c - c0)) for r, c in coords)


def _bbox_overlap(b1: tuple[int, int, int, int], b2: tuple[int, int, int, int]) -> bool:
    r0a, c0a, r1a, c1a = map(int, b1)
    r0b, c0b, r1b, c1b = map(int, b2)
    return not (r1a < r0b or r1b < r0a or c1a < c0b or c1b < c0a)


def _mirror_coords(coords: np.ndarray, axis_col: int) -> np.ndarray:
    mirrored = np.asarray(coords, dtype=int).copy()
    mirrored[:, 1] = 2 * int(axis_col) - mirrored[:, 1]
    return mirrored


def _paint_component(grid: np.ndarray, coords: np.ndarray, color: int) -> None:
    grid[coords[:, 0], coords[:, 1]] = int(color)


def _analyze_input(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    axis_cols = [c for c in range(w) if bool(np.all(g[:, c] == AXIS))]
    assert len(axis_cols) == 1, "expected exactly one full red axis column"
    axis_col = int(axis_cols[0])

    extra_colors = set(map(int, np.unique(g))) - {BG, AXIS, SPRITE}
    assert not extra_colors, "input must contain only black background, red axis, and grey sprites"

    comps = find_components(
        g,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[SPRITE],
        return_format="dict",
    )
    assert len(comps) == 4, "expected exactly four grey sprites"

    all_coords = np.concatenate([comp["coords"] for comp in comps], axis=0)
    assert not np.any(all_coords[:, 1] == axis_col), "sprites may not touch the axis"
    on_left = bool(np.all(all_coords[:, 1] < axis_col))
    on_right = bool(np.all(all_coords[:, 1] > axis_col))
    assert on_left ^ on_right, "all sprites must lie on exactly one side of the axis"
    source_side = "left" if on_left else "right"

    if source_side == "left":
        assert not np.any(g[:, axis_col + 1 :] == SPRITE), "mirror side must be empty in the input"
    else:
        assert not np.any(g[:, :axis_col] == SPRITE), "mirror side must be empty in the input"

    shape_sigs = set()
    dist_twice_values = set()
    enriched = []
    for comp in comps:
        coords = np.asarray(comp["coords"], dtype=int)
        r0, c0, r1, c1 = map(int, comp["bbox"])
        area = int(comp["area"])
        assert 3 <= area <= 6, "sprite area must be between 3 and 6"

        sig = _normalize_coords(coords)
        assert sig not in shape_sigs, "all sprite shapes must be distinct"
        shape_sigs.add(sig)

        center_twice = _bbox_center_twice(comp)
        dist_twice = abs(center_twice - 2 * axis_col)
        assert dist_twice not in dist_twice_values, "bbox-center distances to the axis must be unique"
        dist_twice_values.add(dist_twice)

        mirrored = _mirror_coords(coords, axis_col)
        assert np.all((0 <= mirrored[:, 1]) & (mirrored[:, 1] < w)), "mirrored sprite must remain in bounds"
        assert np.all(mirrored[:, 1] != axis_col), "mirrored sprite may not overlap the axis"

        enriched.append(
            {
                "coords": coords,
                "bbox": (r0, c0, r1, c1),
                "area": area,
                "dist_twice": int(dist_twice),
                "mirror_coords": mirrored,
            }
        )

    for i in range(len(enriched)):
        for j in range(i + 1, len(enriched)):
            assert not _bbox_overlap(enriched[i]["bbox"], enriched[j]["bbox"]), "sprite bboxes must not overlap"

    return {
        "shape": g.shape,
        "axis_col": axis_col,
        "source_side": source_side,
        "components": enriched,
    }


def main(input_grid: np.ndarray) -> np.ndarray:
    """Rank one-sided sprites by distance to a red vertical axis, recolor them, and mirror them across that axis.

    Input structure:
    - The grid has black background, one full-height red vertical axis, and exactly four grey 4-connected sprite components.
    - All grey sprites lie strictly on one side of the axis; the other side is empty.
    - Each sprite has area 3 to 6, its bbox does not overlap any other sprite bbox, and its bbox-center distance to the axis is unique.

    Rule:
    - Detect the red axis and the four grey sprites.
    - Compute each sprite's horizontal distance from its bbox center to the axis column.
    - Sort sprites from nearest to farthest; recolor ranks 1..4 as blue, green, yellow, magenta.
    - Keep the recolored original sprites in place, reflect each one across the axis to the opposite side with the same color, and preserve the red axis.

    Key insight:
    - The axis is both the mirror line and the ranking reference, so moving the axis changes both the geometry of the mirrored copy and the color assigned to each sprite.
    """
    g = np.asarray(input_grid, dtype=int)
    info = _analyze_input(g)
    out = np.zeros_like(g)
    out[:, info["axis_col"]] = AXIS

    ranked = sorted(info["components"], key=lambda comp: comp["dist_twice"])
    for color, comp in zip(RANK_COLORS, ranked):
        _paint_component(out, comp["coords"], color)
        _paint_component(out, comp["mirror_coords"], color)

    return out


def _random_shape() -> np.ndarray:
    for _ in range(200):
        target_area = random.randint(3, 6)
        cells = {(0, 0)}
        while len(cells) < target_area:
            frontier = set()
            for r, c in cells:
                frontier.update({(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)})
            frontier -= cells
            cells.add(random.choice(tuple(frontier)))

        rs = [r for r, _ in cells]
        cs = [c for _, c in cells]
        r0, c0 = min(rs), min(cs)
        norm = {(r - r0, c - c0) for r, c in cells}
        height = max(r for r, _ in norm) + 1
        width = max(c for _, c in norm) + 1
        if not (2 <= height <= 3 and 2 <= width <= 3):
            continue

        arr = np.zeros((height, width), dtype=int)
        for r, c in norm:
            arr[r, c] = SPRITE
        return arr
    raise RuntimeError("failed to generate a compact sprite")


def _sample_unique_shapes() -> list[np.ndarray]:
    shapes: list[np.ndarray] = []
    sigs: set[frozenset[tuple[int, int]]] = set()
    while len(shapes) < 4:
        shape = _random_shape()
        coords = np.argwhere(shape == SPRITE)
        sig = _normalize_coords(coords)
        if sig in sigs:
            continue
        sigs.add(sig)
        shapes.append(shape)
    return shapes


def _choose_columns(widths: list[int], axis_col: int, source_side: str, total_width: int) -> list[int] | None:
    candidates: list[list[tuple[int, int]]] = []
    for width in widths:
        opts = []
        if source_side == "left":
            for c0 in range(0, axis_col - width + 1):
                dist_twice = abs((2 * c0 + width - 1) - 2 * axis_col)
                opts.append((c0, dist_twice))
        else:
            for c0 in range(axis_col + 1, total_width - width + 1):
                dist_twice = abs((2 * c0 + width - 1) - 2 * axis_col)
                opts.append((c0, dist_twice))
        random.shuffle(opts)
        candidates.append(opts)

    order = list(range(len(widths)))
    order.sort(key=lambda idx: len(candidates[idx]))
    assignment: list[int | None] = [None] * len(widths)
    used_dists: set[int] = set()

    def backtrack(pos: int) -> bool:
        if pos == len(order):
            return True
        idx = order[pos]
        for c0, dist_twice in candidates[idx]:
            if dist_twice in used_dists:
                continue
            assignment[idx] = c0
            used_dists.add(dist_twice)
            if backtrack(pos + 1):
                return True
            used_dists.remove(dist_twice)
            assignment[idx] = None
        return False

    if not backtrack(0):
        return None
    return [int(x) for x in assignment]


def _choose_rows(heights: list[int], total_height: int, order: list[int]) -> list[int]:
    required = sum(heights) + (len(heights) - 1)
    extra = total_height - required
    assert extra >= 0

    slots = [0] * (len(heights) + 1)
    for _ in range(extra):
        slots[random.randrange(len(slots))] += 1

    rows = [0] * len(heights)
    cursor = slots[0]
    for pos, idx in enumerate(order):
        rows[idx] = cursor
        cursor += heights[idx]
        if pos < len(heights) - 1:
            cursor += 1 + slots[pos + 1]
    return rows


def _rank_signature(items: list[dict], key) -> tuple[int, ...]:
    ranked = sorted(range(len(items)), key=lambda idx: key(items[idx]))
    return tuple(int(idx) for idx in ranked)


def _validate_output(input_grid: np.ndarray, output_grid: np.ndarray, info: dict) -> None:
    expected = np.zeros_like(input_grid)
    expected[:, info["axis_col"]] = AXIS
    ranked = sorted(info["components"], key=lambda comp: comp["dist_twice"])
    for color, comp in zip(RANK_COLORS, ranked):
        _paint_component(expected, comp["coords"], color)
        _paint_component(expected, comp["mirror_coords"], color)
    assert np.array_equal(output_grid, expected), "output does not match the puzzle rule"


def _gen_attempt() -> np.ndarray | None:
    shapes = _sample_unique_shapes()
    dims = [{"h": int(shape.shape[0]), "w": int(shape.shape[1]), "area": int(np.count_nonzero(shape))} for shape in shapes]

    min_height = sum(d["h"] for d in dims) + 3
    if min_height > 16:
        return None
    total_height = random.randint(max(11, min_height), 16)

    short_width = random.randint(6, 8)
    long_min = max(short_width + 2, total_height - short_width + 1, 8)
    long_max = min(14, 29 - short_width)
    if long_min > long_max:
        return None
    long_width = random.randint(long_min, long_max)
    total_width = short_width + 1 + long_width
    if total_width <= total_height:
        return None

    source_side = random.choice(["left", "right"])
    axis_col = short_width if source_side == "left" else long_width

    widths = [d["w"] for d in dims]
    cols = _choose_columns(widths, axis_col, source_side, total_width)
    if cols is None:
        return None

    row_order = list(range(4))
    random.shuffle(row_order)
    rows = _choose_rows([d["h"] for d in dims], total_height, row_order)

    grid = np.zeros((total_height, total_width), dtype=int)
    grid[:, axis_col] = AXIS
    for idx, shape in enumerate(shapes):
        h, w = shape.shape
        r0 = rows[idx]
        c0 = cols[idx]
        mask = shape == SPRITE
        grid[r0 : r0 + h, c0 : c0 + w][mask] = SPRITE

    try:
        info = _analyze_input(grid)
        dist_rank = _rank_signature(info["components"], key=lambda comp: comp["dist_twice"])
        row_rank = _rank_signature(info["components"], key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
        area_rank = _rank_signature(info["components"], key=lambda comp: (comp["area"], comp["bbox"][0], comp["bbox"][1]))
        if row_rank == dist_rank or area_rank == dist_rank:
            return None
        out = main(grid)
        _validate_output(grid, out, info)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
