from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
TRANSFORMS = ("lr", "ud", "transpose")
GEN_ATTEMPTS = 400
MASK_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def colors_present(grid: np.ndarray, *, exclude: set[int] | None = None) -> list[int]:
    colors = set(map(int, np.unique(grid)))
    if exclude is not None:
        colors -= set(map(int, exclude))
    return sorted(colors)


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _mask_from_coords(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _transform_coords(coords: np.ndarray, mode: str) -> np.ndarray:
    r0, c0, r1, c1 = bbox_coords(coords)
    rr = coords[:, 0].copy()
    cc = coords[:, 1].copy()
    if mode == "lr":
        new = np.column_stack([rr, c0 + c1 - cc])
    elif mode == "ud":
        new = np.column_stack([r0 + r1 - rr, cc])
    elif mode == "transpose":
        if (r1 - r0) != (c1 - c0):
            raise AssertionError("transpose legend requires a square bbox")
        new = np.column_stack([r0 + (cc - c0), c0 + (rr - r0)])
    else:
        raise ValueError(mode)
    return new.astype(int)


def _transform_mask(mask: np.ndarray, mode: str) -> np.ndarray:
    if mode == "lr":
        return np.fliplr(mask)
    if mode == "ud":
        return np.flipud(mask)
    if mode == "transpose":
        return mask.T.copy()
    raise ValueError(mode)


def _decode_indicator(offset: tuple[int, int]) -> str:
    dr, dc = offset
    if dr == 0 and dc != 0:
        return "lr"
    if dr != 0 and dc == 0:
        return "ud"
    if dr != 0 and dc != 0:
        return "transpose"
    raise AssertionError("indicator cannot coincide with legend center")


def _find_legends(grid: np.ndarray) -> dict[int, dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    legends: dict[int, dict[str, object]] = {}
    seen_windows: set[tuple[int, int]] = set()

    for r0 in range(h - 2):
        for c0 in range(w - 2):
            if not (r0 == 0 or c0 == 0 or r0 == h - 3 or c0 == w - 3):
                continue
            if (r0, c0) in seen_windows:
                continue
            patch = g[r0 : r0 + 3, c0 : c0 + 3]
            if int(patch[1, 1]) != GREY:
                continue
            grey_coords = np.argwhere(patch == GREY)
            if len(grey_coords) != 2:
                continue
            non_grey = patch[patch != GREY]
            if non_grey.size != 7:
                continue
            fill_colors = set(map(int, non_grey))
            if len(fill_colors) != 1:
                continue
            color = next(iter(fill_colors))
            if color in {BG, GREY}:
                continue
            coords = {(r0 + r, c0 + c) for r in range(3) for c in range(3)}
            center = (r0 + 1, c0 + 1)
            indicators = [(r0 + int(r), c0 + int(c)) for r, c in grey_coords if not (int(r) == 1 and int(c) == 1)]
            if len(indicators) != 1:
                continue
            ind = indicators[0]
            offset = (ind[0] - center[0], ind[1] - center[1])
            mode = _decode_indicator(offset)
            if color in legends:
                raise AssertionError("each legend color must be unique")
            legends[color] = {"mode": mode, "coords": coords}
            seen_windows.add((r0, c0))

    return legends


def _parse_objects(grid: np.ndarray, legends: dict[int, dict[str, object]]) -> dict[int, np.ndarray]:
    g = np.asarray(grid, dtype=int).copy()
    for info in legends.values():
        for r, c in info["coords"]:
            g[r, c] = BG

    objects: dict[int, np.ndarray] = {}
    for color in colors_present(g, exclude={BG, GREY}):
        coords = np.argwhere(g == color)
        if len(coords) == 0:
            continue
        objects[color] = coords.astype(int)
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reflect each colored mask inside its own bounding box using the axis encoded by its matching border legend.

    Input structure:
    - A black background contains three non-black irregular masks, each in a different color and each separated from the others.
    - The outer border also contains three 3x3 legends whose fill colors match the three masks; each legend has a grey center cell and one extra grey indicator cell.
    - The indicator's position relative to the center encodes one reflection type for that color: horizontal neighbor means left-right mirror, vertical neighbor means up-down mirror, diagonal neighbor means main-diagonal transpose.

    Rule:
    - Find the three border legends and decode a transform for each fill color from the center-to-indicator offset.
    - Remove the legends, then for each matching interior color take all pixels of that color, compute their bounding box, and transform those pixels only within that box.
    - Left-right mirror maps (r, c) to (r, c0 + c1 - c); up-down mirror maps (r, c) to (r0 + r1 - r, c); transpose maps (r, c) to (r0 + (c - c0), c0 + (r - r0)).

    Key insight:
    - The legends do not mark destinations directly; they parameterize which geometric reflection each color mask must undergo, so solving requires first matching legend color to mask color and only then applying the per-color transform.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    legends = _find_legends(g)
    objects = _parse_objects(g, legends)

    out = g.copy()
    for info in legends.values():
        for r, c in info["coords"]:
            out[r, c] = bg

    for color, coords in objects.items():
        out[coords[:, 0], coords[:, 1]] = bg

    for color, coords in objects.items():
        if color not in legends:
            raise AssertionError("each mask color must have a matching legend")
        moved = _transform_coords(coords, str(legends[color]["mode"]))
        if len({(int(r), int(c)) for r, c in moved}) != len(moved):
            raise AssertionError("transform must stay one-to-one inside the bbox")
        for r, c in moved:
            if out[r, c] != bg:
                raise AssertionError("transformed masks must not overlap")
            out[r, c] = color

    return out


def _connected_mask(shape: tuple[int, int], area: int) -> np.ndarray | None:
    h, w = shape
    seed = (random.randrange(h), random.randrange(w))
    cells = {seed}
    frontier = set()
    for nr, nc in neighbors4(*seed):
        if 0 <= nr < h and 0 <= nc < w:
            frontier.add((nr, nc))

    while len(cells) < area and frontier:
        cell = random.choice(tuple(frontier))
        frontier.remove(cell)
        cells.add(cell)
        for nr, nc in neighbors4(*cell):
            if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                frontier.add((nr, nc))

    if len(cells) != area:
        return None

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _valid_shape(mask: np.ndarray, *, needs_square_tests: bool) -> bool:
    if mask is None:
        return False
    coords = np.argwhere(mask)
    if len(coords) < 5 or len(coords) > 12:
        return False
    if len(components_mask(mask)) != 1:
        return False
    r0, c0, r1, c1 = bbox_coords(coords)
    if (r0, c0) != (0, 0) or (r1, c1) != (mask.shape[0] - 1, mask.shape[1] - 1):
        return False
    if not mask[0].any() or not mask[-1].any() or not mask[:, 0].any() or not mask[:, -1].any():
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    if needs_square_tests and np.array_equal(mask, mask.T):
        return False
    return True


def _generate_shape(h: int, w: int, *, transpose_mode: bool) -> np.ndarray:
    max_area = min(12, h * w)
    for _ in range(MASK_ATTEMPTS):
        area = random.randint(5, max_area)
        mask = _connected_mask((h, w), area)
        if not _valid_shape(mask, needs_square_tests=transpose_mode):
            continue
        if np.array_equal(mask, _transform_mask(mask, "lr")):
            continue
        if np.array_equal(mask, _transform_mask(mask, "ud")):
            continue
        if transpose_mode and np.array_equal(mask, _transform_mask(mask, "transpose")):
            continue
        return mask
    raise RuntimeError("failed to generate a non-degenerate mask")


def _bboxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    return a[2] + gap < b[0] or b[2] + gap < a[0] or a[3] + gap < b[1] or b[3] + gap < a[1]


def _choose_bboxes(h: int, w: int, assignments: dict[int, str]) -> dict[int, tuple[int, int, int, int]] | None:
    specs: list[tuple[int, int, int]] = []
    for color, mode in assignments.items():
        if mode == "transpose":
            side = random.choice((3, 4, 5))
            specs.append((color, side, side))
        else:
            dims = random.choice(((3, 4), (4, 3), (3, 5), (5, 3), (4, 5), (5, 4)))
            specs.append((color, dims[0], dims[1]))
    specs.sort(key=lambda item: item[1] * item[2], reverse=True)

    bboxes: dict[int, tuple[int, int, int, int]] = {}
    for color, bh, bw in specs:
        placed = False
        row_choices = list(range(4, h - bh - 3))
        col_choices = list(range(4, w - bw - 3))
        random.shuffle(row_choices)
        random.shuffle(col_choices)
        for r0 in row_choices:
            for c0 in col_choices:
                cand = (r0, c0, r0 + bh - 1, c0 + bw - 1)
                if all(_bboxes_separated(cand, other, gap=1) for other in bboxes.values()):
                    bboxes[color] = cand
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None
    return bboxes


def _legend_origin(side: str, h: int, w: int) -> tuple[int, int]:
    if side == "top":
        return 0, random.randint(3, w - 6)
    if side == "bottom":
        return h - 3, random.randint(3, w - 6)
    if side == "left":
        return random.randint(3, h - 6), 0
    if side == "right":
        return random.randint(3, h - 6), w - 3
    raise ValueError(side)


def _make_legend(color: int, mode: str) -> np.ndarray:
    patch = np.full((3, 3), color, dtype=int)
    patch[1, 1] = GREY
    if mode == "lr":
        rr, cc = random.choice(((1, 0), (1, 2)))
    elif mode == "ud":
        rr, cc = random.choice(((0, 1), (2, 1)))
    elif mode == "transpose":
        rr, cc = random.choice(((0, 0), (0, 2), (2, 0), (2, 2)))
    else:
        raise ValueError(mode)
    patch[rr, cc] = GREY
    return patch


def _place_mask(grid: np.ndarray, color: int, bbox: tuple[int, int, int, int], mask: np.ndarray) -> None:
    r0, c0, r1, c1 = bbox
    assert mask.shape == (r1 - r0 + 1, c1 - c0 + 1)
    grid[r0 : r1 + 1, c0 : c1 + 1][mask] = color


def _expected_output_from_parse(grid: np.ndarray) -> np.ndarray:
    legends = _find_legends(grid)
    objects = _parse_objects(grid, legends)
    out = np.full_like(grid, BG)
    for color, coords in objects.items():
        moved = _transform_coords(coords, str(legends[color]["mode"]))
        out[moved[:, 0], moved[:, 1]] = color
    return out


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        legends = _find_legends(grid)
        if len(legends) != 3:
            return False
        modes = {str(info["mode"]) for info in legends.values()}
        if modes != set(TRANSFORMS):
            return False

        objects = _parse_objects(grid, legends)
        if set(objects) != set(legends):
            return False

        boxes = []
        for color, coords in objects.items():
            if len(coords) < 5 or len(coords) > 12:
                return False
            if len(components_mask(_mask_from_coords(coords, bbox_coords(coords)))) != 1:
                return False
            box = bbox_coords(coords)
            boxes.append(box)
            mode = str(legends[color]["mode"])
            if mode == "transpose" and (box[2] - box[0]) != (box[3] - box[1]):
                return False
            moved = _transform_coords(coords, mode)
            if {(int(r), int(c)) for r, c in moved} == {(int(r), int(c)) for r, c in coords}:
                return False

        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                if not _bboxes_separated(boxes[i], boxes[j], gap=1):
                    return False

        expected = _expected_output_from_parse(grid)
        actual = main(grid)
        if not np.array_equal(actual, expected):
            return False
        if np.array_equal(actual, grid):
            return False
        if np.any(actual == GREY):
            return False
        return True
    except Exception:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        h = random.randint(15, 19)
        w = random.randint(15, 19)
        colors = random.sample(COLOR_POOL, 3)
        modes = list(TRANSFORMS)
        random.shuffle(modes)
        assignments = {color: mode for color, mode in zip(colors, modes)}

        bboxes = _choose_bboxes(h, w, assignments)
        if bboxes is None:
            continue

        grid = np.full((h, w), BG, dtype=int)

        ok = True
        for color in colors:
            r0, c0, r1, c1 = bboxes[color]
            mode = assignments[color]
            mask = _generate_shape(r1 - r0 + 1, c1 - c0 + 1, transpose_mode=(mode == "transpose"))
            _place_mask(grid, color, bboxes[color], mask)

        if not ok:
            continue

        sides = random.sample(("top", "bottom", "left", "right"), 3)
        occupied_legend_cells: set[tuple[int, int]] = set()
        for color, side in zip(colors, sides):
            patch = _make_legend(color, assignments[color])
            placed = False
            for _legend_try in range(40):
                r0, c0 = _legend_origin(side, h, w)
                coords = {(r0 + r, c0 + c) for r in range(3) for c in range(3)}
                if coords & occupied_legend_cells:
                    continue
                occupied_legend_cells |= coords
                grid[r0 : r0 + 3, c0 : c0 + 3] = patch
                placed = True
                break
            if not placed:
                ok = False
                break

        if not ok:
            continue

        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid puzzle instance")
