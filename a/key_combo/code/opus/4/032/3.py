import itertools
import random

import numpy as np


class Color:
    BLACK = 0
    BLUE = 1
    RED = 2
    GREEN = 3
    YELLOW = 4
    GRAY = 5
    PINK = 6
    ORANGE = 7


PAIR_COLORS = [
    Color.BLUE,
    Color.RED,
    Color.YELLOW,
    Color.GRAY,
    Color.PINK,
    Color.ORANGE,
]
MAX_RETRIES = 400


def neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def bbox(mask: np.ndarray):
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box):
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    return np.kron(sprite, np.ones((factor, factor), dtype=sprite.dtype))


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        comp = []
        while stack:
            rr, cc = stack.pop()
            comp.append((rr, cc))
            for nr, nc in neighbors4(rr, cc):
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(comp, dtype=int))
    return comps


def _regions_between(lines: list[int], size: int) -> list[tuple[int, int]]:
    regions = []
    start = 0
    for line in sorted(lines):
        if start <= line - 1:
            regions.append((start, line - 1))
        start = line + 1
    if start <= size - 1:
        regions.append((start, size - 1))
    return regions


def _mask_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _spans_both_axes(mask: np.ndarray) -> bool:
    rows = np.any(mask, axis=1).sum()
    cols = np.any(mask, axis=0).sum()
    return int(rows) >= 2 and int(cols) >= 2


def _is_full_bbox(mask: np.ndarray) -> bool:
    box = bbox(mask)
    return box == (0, 0, mask.shape[0] - 1, mask.shape[1] - 1)


def _is_irregular(mask: np.ndarray) -> bool:
    return bool(mask.any()) and not bool(mask.all())


def _enumerate_masks(shape: tuple[int, int]) -> list[np.ndarray]:
    h, w = shape
    masks = []
    for bits in range(1, 1 << (h * w)):
        mask = np.array([(bits >> i) & 1 for i in range(h * w)], dtype=bool).reshape(h, w)
        if mask.sum() < 3:
            continue
        if not _is_irregular(mask):
            continue
        if not _spans_both_axes(mask):
            continue
        if not _is_full_bbox(mask):
            continue
        if len(components_mask(mask)) != 1:
            continue
        masks.append(mask)
    return masks


SQUARE_SMALL_MASKS = _enumerate_masks((2, 2)) + _enumerate_masks((3, 3))
RECT_SMALL_MASKS = _enumerate_masks((2, 3)) + _enumerate_masks((3, 2))


def _random_connected_mask(
    shape: tuple[int, int],
    area_range: tuple[int, int],
    *,
    require_full_bbox: bool = True,
    tries: int = 200,
) -> np.ndarray | None:
    h, w = shape
    lo, hi = area_range
    for _ in range(tries):
        target = random.randint(lo, hi)
        start = (random.randrange(h), random.randrange(w))
        coords = {start}
        while len(coords) < target:
            frontier = []
            for r, c in coords:
                for nr, nc in neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in coords:
                        frontier.append((nr, nc))
            if not frontier:
                break
            coords.add(random.choice(frontier))
        if len(coords) != target:
            continue
        mask = np.zeros((h, w), dtype=bool)
        for r, c in coords:
            mask[r, c] = True
        if not _is_irregular(mask):
            continue
        if not _spans_both_axes(mask):
            continue
        if require_full_bbox and not _is_full_bbox(mask):
            continue
        if len(components_mask(mask)) != 1:
            continue
        return mask
    return None


def _cell_bounds(cell_size: int, cell: tuple[int, int]) -> tuple[int, int, int, int]:
    rr, cc = cell
    r0 = rr * (cell_size + 1)
    c0 = cc * (cell_size + 1)
    return r0, c0, r0 + cell_size - 1, c0 + cell_size - 1


def _cell_category(cell: tuple[int, int]) -> str:
    r, c = cell
    if r == 1 and c == 1:
        return "center"
    if r in (0, 2) and c in (0, 2):
        return "corner"
    return "edge"


def _position_color(cell: tuple[int, int]) -> int:
    cat = _cell_category(cell)
    if cat == "corner":
        return Color.RED
    if cat == "center":
        return Color.BLUE
    return Color.YELLOW


def _extract_objects(grid: np.ndarray) -> tuple[list[dict], list[tuple[int, int]], list[tuple[int, int]]]:
    g = np.asarray(grid, dtype=int)
    full_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == Color.GREEN)]
    full_cols = [c for c in range(g.shape[1]) if np.all(g[:, c] == Color.GREEN)]
    row_regions = _regions_between(full_rows, g.shape[0])
    col_regions = _regions_between(full_cols, g.shape[1])
    objects = []
    for row_i, (r0, r1) in enumerate(row_regions):
        for col_i, (c0, c1) in enumerate(col_regions):
            cell = g[r0 : r1 + 1, c0 : c1 + 1]
            mask = (cell != Color.BLACK) & (cell != Color.GREEN)
            comps = components_mask(mask)
            if not comps:
                continue
            assert len(comps) == 1, "expected at most one object per cell"
            coords = comps[0]
            box = bbox(mask)
            assert box is not None
            tight_mask = crop_bbox(mask, box)
            colors = {int(cell[r, c]) for r, c in coords}
            assert len(colors) == 1
            global_coords = np.array([[int(r0 + r), int(c0 + c)] for r, c in coords], dtype=int)
            objects.append(
                {
                    "cell": (row_i, col_i),
                    "cell_bbox": (r0, c0, r1, c1),
                    "color": int(next(iter(colors))),
                    "coords": global_coords,
                    "mask": tight_mask.astype(bool),
                    "bbox": box,
                }
            )
    return objects, row_regions, col_regions


def _match_small_large(obj_a: dict, obj_b: dict):
    mask_a = obj_a["mask"]
    mask_b = obj_b["mask"]
    if tuple(mask_b.shape) == (mask_a.shape[0] * 2, mask_a.shape[1] * 2):
        if np.array_equal(scale_sprite(mask_a.astype(int), 2).astype(bool), mask_b):
            return obj_a, obj_b
    if tuple(mask_a.shape) == (mask_b.shape[0] * 2, mask_b.shape[1] * 2):
        if np.array_equal(scale_sprite(mask_b.astype(int), 2).astype(bool), mask_a):
            return obj_b, obj_a
    return None


def _best_matching(objects: list[dict]) -> list[tuple[int, int]]:
    edges = []
    for i, j in itertools.combinations(range(len(objects)), 2):
        pair = _match_small_large(objects[i], objects[j])
        if pair is None:
            continue
        if pair[0] is objects[i]:
            edges.append((i, j))
        else:
            edges.append((j, i))

    best_size = -1
    best_matchings: list[tuple[tuple[int, int], ...]] = []

    def rec(idx: int, used: set[int], chosen: list[tuple[int, int]]) -> None:
        nonlocal best_size, best_matchings
        if idx == len(edges):
            key = tuple(sorted(chosen))
            if len(chosen) > best_size:
                best_size = len(chosen)
                best_matchings = [key]
            elif len(chosen) == best_size and key not in best_matchings:
                best_matchings.append(key)
            return
        rec(idx + 1, used, chosen)
        small, large = edges[idx]
        if small not in used and large not in used:
            used.add(small)
            used.add(large)
            chosen.append((small, large))
            rec(idx + 1, used, chosen)
            chosen.pop()
            used.remove(small)
            used.remove(large)

    rec(0, set(), [])
    assert len(best_matchings) == 1, "matching must be unique"
    return list(best_matchings[0])


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    objects, row_regions, col_regions = _extract_objects(g)
    assert len(row_regions) == 3 and len(col_regions) == 3, "expected a 3x3 table"

    matching = _best_matching(objects)
    assert len(matching) == 2, "expected exactly two valid cross-scale pairs"

    out = g.copy()
    out[(out != Color.BLACK) & (out != Color.GREEN)] = Color.BLACK

    for small_idx, _large_idx in matching:
        obj = objects[small_idx]
        new_color = _position_color(obj["cell"])
        out[obj["coords"][:, 0], obj["coords"][:, 1]] = new_color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find cross-scale matching objects inside a green 3x3 table, keep only the smaller matches, and recolor them by cell position.

    Input structure:
    - A black background contains complete green separator rows and columns that split the canvas into a 3x3 table.
    - Some cells are empty and the occupied cells each contain one irregular 4-connected monochrome object in a non-green color.
    - Exactly two pairs consist of the same shape at scales 1x and 2x, while the remaining occupied cells are distractors.

    Rule:
    - Detect the table cells from the full green lines and extract each cell's foreground object while ignoring black background and green separators.
    - Compare objects by tight masks: if scaling one mask by 2 reproduces another mask exactly, those two cells form a pair and only the smaller object survives.
    - Erase the larger partner of each pair and erase every unmatched distractor object.
    - Recolor each surviving smaller object by its own table position: corner cell -> red, edge non-corner cell -> yellow, center cell -> blue.

    Key insight:
    - Shape matching decides which objects survive, but the final color is determined independently by the survivor's location in the 3x3 table.
    """
    return _solve(grid)


def _choose_small_pair_cells() -> list[tuple[int, int]]:
    corner_cells = [(0, 0), (0, 2), (2, 0), (2, 2)]
    edge_cells = [(0, 1), (1, 0), (1, 2), (2, 1)]
    category_to_cells = {
        "corner": corner_cells,
        "edge": edge_cells,
        "center": [(1, 1)],
    }
    cats = random.sample(["corner", "edge", "center"], 2)
    chosen = []
    for cat in cats:
        chosen.append(random.choice(category_to_cells[cat]))
    return chosen


def _assign_colors(small_cells: list[tuple[int, int]]) -> dict[str, int] | None:
    roles = [
        "small0",
        "large0",
        "small1",
        "large1",
        "distractor_small",
        "distractor_large",
    ]
    target0 = _position_color(small_cells[0])
    target1 = _position_color(small_cells[1])
    for _ in range(100):
        perm = random.sample(PAIR_COLORS, len(PAIR_COLORS))
        color_map = dict(zip(roles, perm))
        if color_map["small0"] == target0:
            continue
        if color_map["small1"] == target1:
            continue
        return color_map
    return None


def _paint_mask(grid: np.ndarray, cell_size: int, cell: tuple[int, int], mask: np.ndarray, color: int) -> None:
    r0, c0, _r1, _c1 = _cell_bounds(cell_size, cell)
    h, w = mask.shape
    dr = random.randint(0, cell_size - h)
    dc = random.randint(0, cell_size - w)
    patch = grid[r0 + dr : r0 + dr + h, c0 + dc : c0 + dc + w]
    patch[mask] = color


def _semantic_check(grid: np.ndarray) -> None:
    objects, row_regions, col_regions = _extract_objects(grid)
    assert len(row_regions) == 3 and len(col_regions) == 3
    assert len(objects) == 6
    occupied = {obj["cell"] for obj in objects}
    assert len(occupied) == 6

    matching = _best_matching(objects)
    assert len(matching) == 2
    matched_small = {small for small, _ in matching}
    matched_large = {large for _, large in matching}
    assert len(matched_small) == 2 and len(matched_large) == 2

    unmatched = [idx for idx in range(len(objects)) if idx not in matched_small and idx not in matched_large]
    assert len(unmatched) == 2
    assert any(max(objects[idx]["mask"].shape) <= 3 for idx in unmatched), "need a small distractor"
    assert any(max(objects[idx]["mask"].shape) >= 4 for idx in unmatched), "need a large distractor"

    survivor_categories = {_cell_category(objects[idx]["cell"]) for idx in matched_small}
    assert len(survivor_categories) == 2, "survivors should cover two different position classes"

    for small, large in matching:
        assert objects[small]["color"] != objects[large]["color"]

    out = main(grid)
    assert out.shape == grid.shape
    assert np.all(out[g := (grid == Color.GREEN)] == Color.GREEN)

    expected = grid.copy()
    expected[(expected != Color.BLACK) & (expected != Color.GREEN)] = Color.BLACK
    for idx in matched_small:
        obj = objects[idx]
        expected[obj["coords"][:, 0], obj["coords"][:, 1]] = _position_color(obj["cell"])
    assert np.array_equal(out, expected)
    assert any(np.any(out == color) for color in (Color.RED, Color.YELLOW, Color.BLUE))


def _gen_attempt() -> np.ndarray | None:
    cell_size = random.randint(6, 8)
    side = cell_size * 3 + 2
    grid = np.zeros((side, side), dtype=int)
    for cut in (cell_size, cell_size * 2 + 1):
        grid[cut, :] = Color.GREEN
        grid[:, cut] = Color.GREEN

    pair_small_masks = random.sample(SQUARE_SMALL_MASKS, 2)
    if _mask_key(scale_sprite(pair_small_masks[0].astype(int), 2)) == _mask_key(scale_sprite(pair_small_masks[1].astype(int), 2)):
        return None
    distractor_small = random.choice(RECT_SMALL_MASKS)
    distractor_large = _random_connected_mask(
        random.choice([(5, 4), (4, 5), (5, 5)]),
        (7, 12),
    )
    if distractor_large is None:
        return None

    pair_large_masks = [scale_sprite(mask.astype(int), 2).astype(bool) for mask in pair_small_masks]

    small_cells = _choose_small_pair_cells()
    remaining = [cell for cell in itertools.product(range(3), repeat=2) if cell not in small_cells]
    random.shuffle(remaining)
    large_cells = remaining[:2]
    distractor_cells = remaining[2:4]

    color_map = _assign_colors(small_cells)
    if color_map is None:
        return None

    placements = [
        (small_cells[0], pair_small_masks[0], color_map["small0"]),
        (large_cells[0], pair_large_masks[0], color_map["large0"]),
        (small_cells[1], pair_small_masks[1], color_map["small1"]),
        (large_cells[1], pair_large_masks[1], color_map["large1"]),
        (distractor_cells[0], distractor_small, color_map["distractor_small"]),
        (distractor_cells[1], distractor_large, color_map["distractor_large"]),
    ]

    for cell, mask, color in placements:
        if mask.shape[0] > cell_size or mask.shape[1] > cell_size:
            return None
        _paint_mask(grid, cell_size, cell, mask.astype(bool), color)

    try:
        _semantic_check(grid)
    except Exception:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
