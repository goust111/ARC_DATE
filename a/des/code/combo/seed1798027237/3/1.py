import random
from typing import Any

import numpy as np


def _majority_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r0 in range(h):
        for c0 in range(w):
            if not mask[r0, c0] or seen[r0, c0]:
                continue
            stack = [(int(r0), int(c0))]
            seen[r0, c0] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((r, c))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    start = None
    for i, flag in enumerate(mask.tolist()):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            out.append((int(start), int(i)))
            start = None
    if start is not None:
        out.append((int(start), int(mask.size)))
    return out


def _horizontal_legend(
    grid: np.ndarray,
    row0: int,
    row1: int,
    bg: int,
) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    same = (grid[row0] == grid[row1]) & (grid[row0] != bg)
    candidates = [(a, b) for a, b in _runs(same) if b - a >= 2 and (b - a) % 2 == 0]
    assert candidates, "expected a horizontal legend strip"
    start, end = max(candidates, key=lambda it: (it[1] - it[0], -it[0]))
    values = grid[row0, start:end]
    mapping: dict[int, int] = {}
    for i in range(0, len(values), 2):
        keep = int(values[i])
        replace = int(values[i + 1])
        assert keep != replace, "legend pair must contain two different colors"
        assert replace not in mapping, "replacement colors must be unique within a legend"
        mapping[replace] = keep
    mask = np.zeros((h, w), dtype=bool)
    mask[row0 : row1 + 1, start:end] = True
    return mapping, mask


def _vertical_legend(
    grid: np.ndarray,
    col0: int,
    col1: int,
    bg: int,
) -> tuple[dict[int, int], np.ndarray]:
    h, w = grid.shape
    same = (grid[:, col0] == grid[:, col1]) & (grid[:, col0] != bg)
    candidates = [(a, b) for a, b in _runs(same) if b - a >= 2 and (b - a) % 2 == 0]
    assert candidates, "expected a vertical legend strip"
    start, end = max(candidates, key=lambda it: (it[1] - it[0], -it[0]))
    values = grid[start:end, col0]
    mapping: dict[int, int] = {}
    for i in range(0, len(values), 2):
        keep = int(values[i])
        replace = int(values[i + 1])
        assert keep != replace, "legend pair must contain two different colors"
        assert replace not in mapping, "replacement colors must be unique within a legend"
        mapping[replace] = keep
    mask = np.zeros((h, w), dtype=bool)
    mask[start:end, col0 : col1 + 1] = True
    return mapping, mask


def _parse_legends(grid: np.ndarray, bg: int) -> tuple[dict[str, dict[int, int]], np.ndarray]:
    h, w = grid.shape
    assert h >= 6 and w >= 6, "grid must be large enough to hold all four legends"

    top, top_mask = _horizontal_legend(grid, 0, 1, bg)
    bottom, bottom_mask = _horizontal_legend(grid, h - 2, h - 1, bg)
    left, left_mask = _vertical_legend(grid, 0, 1, bg)
    right, right_mask = _vertical_legend(grid, w - 2, w - 1, bg)

    legend_mask = top_mask | bottom_mask | left_mask | right_mask
    legends = {
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }
    return legends, legend_mask


def _perimeter_cells(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0 + 1, r1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def _frame_info(coords: np.ndarray) -> dict[str, Any] | None:
    r0, c0, r1, c1 = _bbox(coords)
    if r1 - r0 < 3 or c1 - c0 < 3:
        return None

    coord_set = {tuple(map(int, pt)) for pt in coords}
    perimeter = _perimeter_cells((r0, c0, r1, c1))
    if len(coord_set) != len(perimeter) - 1:
        return None
    if not coord_set <= perimeter:
        return None

    missing = list(perimeter - coord_set)
    if len(missing) != 1:
        return None
    gap_r, gap_c = missing[0]
    if (gap_r, gap_c) in {(r0, c0), (r0, c1), (r1, c0), (r1, c1)}:
        return None

    if gap_r == r0:
        direction = "top"
    elif gap_r == r1:
        direction = "bottom"
    elif gap_c == c0:
        direction = "left"
    else:
        direction = "right"

    return {"bbox": (r0, c0, r1, c1), "direction": direction}


def _find_frames(grid: np.ndarray, bg: int, legend_mask: np.ndarray) -> list[dict[str, Any]]:
    frames: list[dict[str, Any]] = []
    colors = [int(color) for color in np.unique(grid[~legend_mask]) if int(color) != bg]
    for color in colors:
        comps = _components((grid == color) & (~legend_mask))
        for coords in comps:
            info = _frame_info(coords)
            if info is None:
                continue
            info["color"] = color
            frames.append(info)
    frames.sort(key=lambda item: item["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    根据边框缺口方向选择对应的边界图例，并把框内匹配到“替换色”的对象改成同组“保留色”。
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _majority_color(g)
    legends, legend_mask = _parse_legends(g, bg)
    frames = _find_frames(g, bg, legend_mask)
    assert frames, "expected at least one gapped rectangular frame"

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mapping = legends[frame["direction"]]
        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        recolored = inner.copy()
        for replace, keep in mapping.items():
            recolored[inner == replace] = keep
        out[r0 + 1 : r1, c0 + 1 : c1] = recolored

    return out


# --------------- generator ---------------

BG = 0
FRAME_COLOR = 5
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
MIN_GRID = 22
MAX_GRID = 30
FRAME_COUNT_CHOICES = (4, 5)
FRAME_H_MIN = 6
FRAME_H_MAX = 9
FRAME_W_MIN = 6
FRAME_W_MAX = 9
LEGEND_PAIR_CHOICES = (2, 3)
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
OBJECT_RETRIES = 120

SHAPES = [
    np.array([[1]], dtype=bool),
    np.array([[1, 1]], dtype=bool),
    np.array([[1], [1]], dtype=bool),
    np.array([[1, 1, 1]], dtype=bool),
    np.array([[1], [1], [1]], dtype=bool),
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1], [0, 1]], dtype=bool),
    np.array([[1, 0], [1, 1]], dtype=bool),
    np.array([[0, 1], [1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return not (
        a[2] + pad < b[0]
        or b[2] + pad < a[0]
        or a[3] + pad < b[1]
        or b[3] + pad < a[1]
    )


def _random_mapping() -> list[tuple[int, int]]:
    pair_count = random.choice(LEGEND_PAIR_CHOICES)
    colors = random.sample(GEN_COLORS, 2 * pair_count)
    return [(int(colors[2 * i]), int(colors[2 * i + 1])) for i in range(pair_count)]


def _draw_horizontal_legend(grid: np.ndarray, row0: int, start: int, pairs: list[tuple[int, int]]) -> None:
    values = np.array([value for pair in pairs for value in pair], dtype=int)
    grid[row0 : row0 + 2, start : start + len(values)] = values[np.newaxis, :]


def _draw_vertical_legend(grid: np.ndarray, col0: int, start: int, pairs: list[tuple[int, int]]) -> None:
    values = np.array([value for pair in pairs for value in pair], dtype=int)
    grid[start : start + len(values), col0 : col0 + 2] = values[:, np.newaxis]


def _place_legends(grid: np.ndarray) -> dict[str, list[tuple[int, int]]]:
    h, w = grid.shape
    legends = {
        "top": _random_mapping(),
        "bottom": _random_mapping(),
        "left": _random_mapping(),
        "right": _random_mapping(),
    }

    for side in ("top", "bottom"):
        length = 2 * len(legends[side])
        start = random.randint(2, w - 2 - length)
        row0 = 0 if side == "top" else h - 2
        _draw_horizontal_legend(grid, row0, start, legends[side])

    for side in ("left", "right"):
        length = 2 * len(legends[side])
        start = random.randint(2, h - 2 - length)
        col0 = 0 if side == "left" else w - 2
        _draw_vertical_legend(grid, col0, start, legends[side])

    return legends


def _random_shape(max_h: int, max_w: int) -> np.ndarray:
    for _ in range(40):
        shape = random.choice(SHAPES)
        transformed = np.rot90(shape, random.randrange(4))
        if random.random() < 0.5:
            transformed = np.fliplr(transformed)
        if random.random() < 0.5:
            transformed = np.flipud(transformed)
        if transformed.shape[0] <= max_h and transformed.shape[1] <= max_w:
            return transformed.copy()
    return np.ones((1, 1), dtype=bool)


def _choose_object_colors(
    pairs: list[tuple[int, int]],
    max_objects: int,
) -> list[int]:
    keeps = [keep for keep, _replace in pairs]
    replaces = [replace for _keep, replace in pairs]
    active_keep, active_replace = random.choice(pairs)

    count = 2 if max_objects == 2 else random.choice([2, 3])
    colors = [int(active_replace)]

    preferred = []
    if active_keep not in colors:
        preferred.extend([int(active_keep), int(active_keep)])
    preferred.extend(int(c) for c in replaces if c != active_replace)
    preferred.extend(int(c) for c in keeps if c != active_keep)
    preferred.extend(
        int(c) for c in GEN_COLORS if c not in keeps and c not in replaces and c not in colors
    )
    preferred.extend(int(c) for c in GEN_COLORS if c not in colors)

    for color in preferred:
        if color not in colors:
            colors.append(color)
        if len(colors) == count:
            break

    return colors


def _place_objects(ih: int, iw: int, colors: list[int]) -> list[dict[str, Any]] | None:
    occupied = np.zeros((ih, iw), dtype=bool)
    placed: list[dict[str, Any]] = []

    for color in colors:
        for _ in range(OBJECT_RETRIES):
            shape = _random_shape(min(3, ih), min(3, iw))
            sh, sw = map(int, shape.shape)
            if sh > ih or sw > iw:
                continue
            r0 = random.randint(0, ih - sh)
            c0 = random.randint(0, iw - sw)
            cells = [(r0 + int(r), c0 + int(c)) for r, c in np.argwhere(shape)]

            valid = True
            for rr, cc in cells:
                r_lo = max(0, rr - 1)
                r_hi = min(ih, rr + 2)
                c_lo = max(0, cc - 1)
                c_hi = min(iw, cc + 2)
                if occupied[r_lo:r_hi, c_lo:c_hi].any():
                    valid = False
                    break
            if not valid:
                continue

            for rr, cc in cells:
                occupied[rr, cc] = True
            placed.append({"color": int(color), "row": int(r0), "col": int(c0), "shape": shape})
            break
        else:
            return None

    return placed


def _draw_frame(
    grid: np.ndarray,
    bbox: tuple[int, int, int, int],
    direction: str,
) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = FRAME_COLOR
    grid[r1, c0 : c1 + 1] = FRAME_COLOR
    grid[r0 : r1 + 1, c0] = FRAME_COLOR
    grid[r0 : r1 + 1, c1] = FRAME_COLOR

    if direction == "top":
        gap = (r0, random.randint(c0 + 1, c1 - 1))
    elif direction == "bottom":
        gap = (r1, random.randint(c0 + 1, c1 - 1))
    elif direction == "left":
        gap = (random.randint(r0 + 1, r1 - 1), c0)
    else:
        gap = (random.randint(r0 + 1, r1 - 1), c1)
    grid[gap] = BG


def _place_frames(
    h: int,
    w: int,
    directions: list[str],
) -> list[dict[str, Any]] | None:
    specs = []
    for direction in directions:
        specs.append(
            {
                "direction": direction,
                "height": random.randint(FRAME_H_MIN, FRAME_H_MAX),
                "width": random.randint(FRAME_W_MIN, FRAME_W_MAX),
            }
        )
    specs.sort(key=lambda item: item["height"] * item["width"], reverse=True)

    placed_boxes: list[tuple[int, int, int, int]] = []
    for spec in specs:
        height = int(spec["height"])
        width = int(spec["width"])
        max_r0 = h - height - 3
        max_c0 = w - width - 3
        if max_r0 < 3 or max_c0 < 3:
            return None
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(3, max_r0)
            c0 = random.randint(3, max_c0)
            bbox = (r0, c0, r0 + height - 1, c0 + width - 1)
            if any(_boxes_overlap(bbox, other, pad=1) for other in placed_boxes):
                continue
            placed_boxes.append(bbox)
            spec["bbox"] = bbox
            break
        else:
            return None

    return specs


def _populate_frames(
    grid: np.ndarray,
    legends: dict[str, list[tuple[int, int]]],
    frames: list[dict[str, Any]],
) -> list[dict[str, Any]] | None:
    populated: list[dict[str, Any]] = []
    for frame in frames:
        bbox = frame["bbox"]
        direction = frame["direction"]
        r0, c0, r1, c1 = bbox
        _draw_frame(grid, bbox, direction)

        ih = r1 - r0 - 1
        iw = c1 - c0 - 1
        max_objects = 2 if ih * iw < 25 else 3
        colors = _choose_object_colors(legends[direction], max_objects)
        objects = _place_objects(ih, iw, colors)
        if objects is None:
            return None

        for obj in objects:
            shape = obj["shape"]
            for rr, cc in np.argwhere(shape):
                grid[r0 + 1 + obj["row"] + int(rr), c0 + 1 + obj["col"] + int(cc)] = obj["color"]

        frame_copy = dict(frame)
        frame_copy["objects"] = objects
        populated.append(frame_copy)
    return populated


def _expected_output(
    grid: np.ndarray,
    legends: dict[str, list[tuple[int, int]]],
    frames: list[dict[str, Any]],
) -> np.ndarray:
    out = grid.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        mapping = {replace: keep for keep, replace in legends[frame["direction"]]}
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        recolored = inner.copy()
        for replace, keep in mapping.items():
            recolored[inner == replace] = keep
        out[r0 + 1 : r1, c0 + 1 : c1] = recolored
    return out


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(MIN_GRID, MAX_GRID)
    w = random.randint(MIN_GRID, MAX_GRID)
    grid = np.full((h, w), BG, dtype=int)
    legends = _place_legends(grid)

    frame_count = random.choice(FRAME_COUNT_CHOICES)
    directions = ["top", "bottom", "left", "right"]
    if frame_count == 5:
        directions.append(random.choice(directions))
    random.shuffle(directions)

    frames = _place_frames(h, w, directions)
    if frames is None:
        return None
    frames = _populate_frames(grid, legends, frames)
    if frames is None:
        return None

    expected = _expected_output(grid, legends, frames)
    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
