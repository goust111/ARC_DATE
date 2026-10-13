from __future__ import annotations

import random
from collections import deque
from importlib.machinery import SourceFileLoader

import numpy as np

_ARC_UTILS = SourceFileLoader(
    "arc_utils_shared_056_10", "/home/lixinyu/snap/new/solvers/arc_utils.py"
).load_module()

Color = _ARC_UTILS.Color
bbox_coords = _ARC_UTILS.bbox_coords
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
neighbors4 = _ARC_UTILS.neighbors4

BG = int(Color.BLACK)
FILL = int(Color.RED)
BODY = int(Color.GREY)

GEN_ATTEMPTS = 500
GRID_SIDE_RANGE = (16, 23)
_SEEN_OUTPUT_SIGNATURES: set[tuple[tuple[int, int], bytes]] = set()


def _sorted_reading_order(components: list[dict]) -> list[dict]:
    return sorted(components, key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))


def _component_connector(grid: np.ndarray, component: dict) -> tuple[int, int]:
    coords = component["coords"]
    red_coords = coords[grid[coords[:, 0], coords[:, 1]] == FILL]
    assert len(red_coords) == 1, "each sprite must have exactly one red connector"
    return int(red_coords[0, 0]), int(red_coords[0, 1])


def _component_bounds(component: dict) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = component["bbox"]
    return int(r0), int(c0), int(r1), int(c1)


def _render_docked(base_cells: set[tuple[int, int]], insert_cells: set[tuple[int, int]]) -> np.ndarray:
    merged = np.array(list(base_cells | insert_cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(merged)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in base_cells | insert_cells:
        out[r - r0, c - c0] = BODY
    return out


def _enclosed_mask(grid: np.ndarray) -> np.ndarray:
    h, w = grid.shape
    exterior = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if grid[r, c] == BG and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if grid[r, c] == BG and not exterior[r, c]:
                exterior[r, c] = True
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for nr, nc in neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == BG and not exterior[nr, nc]:
                exterior[nr, nc] = True
                queue.append((nr, nc))

    return (grid == BG) & (~exterior)


def _hole_count(grid: np.ndarray) -> int:
    enclosed = _enclosed_mask(grid)
    if not np.any(enclosed):
        return 0
    seen = np.zeros_like(enclosed, dtype=bool)
    holes = 0
    for r, c in np.argwhere(enclosed):
        rr, cc = int(r), int(c)
        if seen[rr, cc]:
            continue
        holes += 1
        queue = deque([(rr, cc)])
        seen[rr, cc] = True
        while queue:
            cr, cc = queue.popleft()
            for nr, nc in neighbors4(cr, cc):
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and enclosed[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    queue.append((nr, nc))
    return holes


def _tight_body_grid(cells: set[tuple[int, int]]) -> np.ndarray:
    coords = np.array(list(cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(coords)
    out = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in cells:
        out[r - r0, c - c0] = BODY
    return out


def _normalize(cells: set[tuple[int, int]], connector: tuple[int, int]) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    coords = np.array(list(cells), dtype=int)
    r0, c0, _r1, _c1 = bbox_coords(coords)
    shifted = {(r - r0, c - c0) for r, c in cells}
    return shifted, (connector[0] - r0, connector[1] - c0)


def _base_shape_single(rng: random.Random) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    height = rng.choice((5, 6))
    width = rng.choice((6, 7, 8)) if height == 5 else rng.choice((6, 7))
    cells = {(0, c) for c in range(width)}
    cells |= {(r, 0) for r in range(1, height - 1)}
    cells |= {(r, width - 1) for r in range(1, height - 1)}
    if width + 2 * (height - 2) <= 14 and rng.random() < 0.65:
        stub_row = rng.randint(1, height - 2)
        cells.add((stub_row, -1))
    connector = (height - 2, width - 1)
    return _normalize(cells, connector)


def _base_shape_double(rng: random.Random) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    height = rng.choice((5, 6))
    width = rng.choice((6, 7))
    cells = {(0, c) for c in range(width)}
    cells |= {(r, 0) for r in range(1, height - 1)}
    cells |= {(r, width - 1) for r in range(1, height - 1)}
    mid = width // 2
    cells |= {(r, mid) for r in range(1, height - 2)}
    connector = (height - 2, width - 1)
    return _normalize(cells, connector)


def _insert_shape(rng: random.Random, width_hint: int, hole_family: str) -> tuple[set[tuple[int, int]], tuple[int, int]]:
    body_width = max(5, width_hint)
    cells = {(1, c) for c in range(body_width)}
    connector = (0, body_width - 1)
    cells.add(connector)

    leg_columns = list(range(body_width - 1))
    rng.shuffle(leg_columns)
    leg_count = 2 if hole_family == "double" else rng.randint(2, 4)
    leg_count = min(leg_count, len(leg_columns))
    for c in leg_columns[:leg_count]:
        cells.add((2, c))

    if hole_family == "double" and body_width >= 6:
        cells.add((2, body_width - 2))

    return _normalize(cells, connector)


def _build_docked_pair(rng: random.Random) -> tuple[set[tuple[int, int]], tuple[int, int], set[tuple[int, int]], tuple[int, int], int]:
    family = rng.choice(("single", "double"))
    if family == "single":
        base_cells, base_connector = _base_shape_single(rng)
    else:
        base_cells, base_connector = _base_shape_double(rng)

    base_width = max(c for _r, c in base_cells) + 1
    insert_cells, insert_connector = _insert_shape(rng, base_width - 1, family)

    if rng.random() < 0.5:
        base_width = max(c for _r, c in base_cells) + 1
        insert_width = max(c for _r, c in insert_cells) + 1
        base_cells = {(r, base_width - 1 - c) for r, c in base_cells}
        insert_cells = {(r, insert_width - 1 - c) for r, c in insert_cells}
        base_connector = (base_connector[0], base_width - 1 - base_connector[1])
        insert_connector = (insert_connector[0], insert_width - 1 - insert_connector[1])

    dr = base_connector[0] - 1 - insert_connector[0]
    dc = base_connector[1] - insert_connector[1]
    shifted_insert = {(r + dr, c + dc) for r, c in insert_cells}

    base_grid = _tight_body_grid(base_cells)
    insert_grid = _tight_body_grid(insert_cells)
    docked = _render_docked(base_cells, shifted_insert)

    assert 8 <= len(base_cells) <= 15
    assert 8 <= len(insert_cells) <= 15
    assert _hole_count(base_grid) == 0
    assert _hole_count(insert_grid) == 0
    holes = _hole_count(docked)
    assert holes >= 1
    if family == "double":
        assert holes >= 2
    return base_cells, base_connector, insert_cells, insert_connector, holes


def _paint_sprite(
    grid: np.ndarray,
    top: int,
    left: int,
    cells: set[tuple[int, int]],
    connector: tuple[int, int],
) -> None:
    for r, c in cells:
        grid[top + r, left + c] = BODY
    grid[top + connector[0], left + connector[1]] = FILL


def _non_touching_positions(
    rng: random.Random,
    grid_shape: tuple[int, int],
    base_shape: tuple[int, int],
    insert_shape: tuple[int, int],
) -> tuple[tuple[int, int], tuple[int, int]]:
    h, w = grid_shape
    base_h, base_w = base_shape
    insert_h, insert_w = insert_shape

    base_positions = [
        (r, c)
        for r in range(1, h - base_h - 1)
        for c in range(1, w - base_w - 1)
    ]
    rng.shuffle(base_positions)

    for br, bc in base_positions:
        candidates = []
        for ir in range(1, h - insert_h - 1):
            for ic in range(1, w - insert_w - 1):
                if (ir, ic) <= (br, bc):
                    continue
                if ir + insert_h + 1 < br or br + base_h + 1 < ir or ic + insert_w + 1 < bc or bc + base_w + 1 < ic:
                    candidates.append((ir, ic))
        if candidates:
            return (br, bc), rng.choice(candidates)
    raise AssertionError("failed to place sprites with base earlier in reading order")


def main(grid: np.ndarray) -> np.ndarray:
    """Dock the two connector-marked grey sprites, then fill the enclosed cavity with the connector color.

    Input structure:
    - Black background contains exactly two disconnected sprites made of grey body
      pixels plus one red connector pixel per sprite.
    - The sprite whose bbox top-left is earlier in reading order is the base; its
      red connector lies on the sprite's lower bbox edge.
    - The other sprite is the insert; its red connector lies on the sprite's upper
      bbox edge.

    Rule:
    - Translate the insert so its red connector lands exactly one cell above the
      base connector.
    - Recolor both red connector pixels to grey and merge the two sprites.
    - Tight-crop the merged non-black pixels, mark border-reachable black cells as
      exterior by 4-neighbor flood fill, and recolor the remaining enclosed black
      cells red.

    Key insight:
    - The docking step changes the topology: the two hole-free sprites together
      create a closed cavity, and the same red connector color that chose the
      docking alignment becomes the cavity fill color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    components = _sorted_reading_order(find_components(g, bg=BG, connectivity=4, monochromatic=False))
    assert len(components) == 2, "expected exactly two sprites"
    base, insert = components

    base_connector = _component_connector(g, base)
    insert_connector = _component_connector(g, insert)
    base_bbox = _component_bounds(base)
    insert_bbox = _component_bounds(insert)
    assert base_connector[0] == base_bbox[2], "base connector must lie on the lower bbox edge"
    assert insert_connector[0] == insert_bbox[0], "insert connector must lie on the upper bbox edge"

    dr = base_connector[0] - 1 - insert_connector[0]
    dc = base_connector[1] - insert_connector[1]

    base_cells = {tuple(map(int, cell)) for cell in base["coords"]}
    insert_cells = {(int(r + dr), int(c + dc)) for r, c in insert["coords"]}
    merged_cells = base_cells | insert_cells

    merged_coords = np.array(list(merged_cells), dtype=int)
    r0, c0, r1, c1 = bbox_coords(merged_coords)
    out = crop_bbox(np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int), (0, 0, r1 - r0, c1 - c0))
    for r, c in merged_cells:
        out[r - r0, c - c0] = BODY

    enclosed = _enclosed_mask(out)
    assert np.any(enclosed), "docked shape must enclose at least one cavity"
    out[enclosed] = FILL
    return out


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        rng = random.Random()
        side = rng.randint(*GRID_SIDE_RANGE)
        grid = np.full((side, side), BG, dtype=int)

        try:
            base_cells, base_connector, insert_cells, insert_connector, expected_holes = _build_docked_pair(rng)
        except AssertionError:
            continue

        base_h = max(r for r, _c in base_cells) + 1
        base_w = max(c for _r, c in base_cells) + 1
        insert_h = max(r for r, _c in insert_cells) + 1
        insert_w = max(c for _r, c in insert_cells) + 1

        try:
            (base_top, base_left), (insert_top, insert_left) = _non_touching_positions(
                rng,
                grid.shape,
                (base_h, base_w),
                (insert_h, insert_w),
            )
        except AssertionError:
            continue

        _paint_sprite(grid, base_top, base_left, base_cells, base_connector)
        _paint_sprite(grid, insert_top, insert_left, insert_cells, insert_connector)

        components = _sorted_reading_order(find_components(grid, bg=BG, connectivity=4, monochromatic=False))
        if len(components) != 2:
            continue

        try:
            out = main(grid)
        except AssertionError:
            continue

        fill_components = find_components(
            np.where(out == FILL, FILL, BG),
            bg=BG,
            connectivity=4,
            monochromatic=True,
            colors=[FILL],
        )
        if not np.any(out == FILL):
            continue
        if len(fill_components) != expected_holes:
            continue
        if np.count_nonzero(out == FILL) < expected_holes:
            continue
        out_sig = (tuple(map(int, out.shape)), out.tobytes())
        if out_sig in _SEEN_OUTPUT_SIGNATURES:
            continue
        _SEEN_OUTPUT_SIGNATURES.add(out_sig)
        return grid

    raise RuntimeError("failed to generate valid input for new puzzle 10")
