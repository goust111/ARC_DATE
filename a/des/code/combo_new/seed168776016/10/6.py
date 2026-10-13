from __future__ import annotations

from collections import Counter, deque
import random
import numpy as np


BG_COLOR = 0
DIRECTIONS = ("up", "down", "left", "right")
GEN_RETRIES = 400
SCENE_TRIES = 40
PLACEMENT_TRIES = 250
OBJECT_COUNT_CHOICES = (1, 2, 3, 4)
OBJECT_COUNT_WEIGHTS = (1, 3, 3, 1)


def _bg_color(grid: np.ndarray) -> int:
    counts = Counter(map(int, np.asarray(grid, dtype=int).flat))
    return int(counts.most_common(1)[0][0])


def _neighbors8(r: int, c: int, height: int, width: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            nr, nc = r + dr, c + dc
            if 0 <= nr < height and 0 <= nc < width:
                yield nr, nc


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    height, width = grid.shape
    seen = np.zeros((height, width), dtype=bool)
    out: list[dict] = []
    for r in range(height):
        for c in range(width):
            color = int(grid[r, c])
            if seen[r, c] or color == int(bg):
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((int(rr), int(cc)))
                for nr, nc in _neighbors8(int(rr), int(cc), height, width):
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((int(nr), int(nc)))
            out.append({"color": color, "coords": coords})
    return out


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    return int(r0), int(c0), int(r1 - r0 + 1), int(c1 - c0 + 1)


def _arrow_cells(direction: str, bbox: tuple[int, int, int, int]) -> list[tuple[int, int]]:
    r0, c0, h, w = bbox
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    mid_r = r0 + h // 2
    mid_c = c0 + w // 2
    if direction == "right":
        return [(mid_r, c1 + 2), (mid_r - 1, c1 + 1), (mid_r + 1, c1 + 1)]
    if direction == "left":
        return [(mid_r, c0 - 2), (mid_r - 1, c0 - 1), (mid_r + 1, c0 - 1)]
    if direction == "down":
        return [(r1 + 2, mid_c), (r1 + 1, mid_c - 1), (r1 + 1, mid_c + 1)]
    return [(r0 - 2, mid_c), (r0 - 1, mid_c - 1), (r0 - 1, mid_c + 1)]


def _candidate_block_bbox(comp_bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int, int, int] | None:
    r0, c0, h, w = comp_bbox
    if direction == "right":
        return (r0, c0, h, w - 2) if w >= 5 else None
    if direction == "left":
        return (r0, c0 + 2, h, w - 2) if w >= 5 else None
    if direction == "down":
        return (r0, c0, h - 2, w) if h >= 5 else None
    return (r0 + 2, c0, h - 2, w) if h >= 5 else None


def _in_bbox(r: int, c: int, bbox: tuple[int, int, int, int]) -> bool:
    r0, c0, h, w = bbox
    return r0 <= r < r0 + h and c0 <= c < c0 + w


def _validate_candidate(
    grid: np.ndarray,
    color: int,
    bg: int,
    coords: list[tuple[int, int]],
    direction: str,
    bbox: tuple[int, int, int, int],
) -> list[tuple[int, int]] | None:
    r0, c0, h, w = bbox
    if not (3 <= h <= 5 and 3 <= w <= 5):
        return None

    component = set(coords)
    outside = {pt for pt in component if not _in_bbox(pt[0], pt[1], bbox)}
    expected_arrow = set(_arrow_cells(direction, bbox))
    if outside != expected_arrow:
        return None

    holes: list[tuple[int, int]] = []
    for r in range(r0, r0 + h):
        for c in range(c0, c0 + w):
            value = int(grid[r, c])
            is_border = r in (r0, r0 + h - 1) or c in (c0, c0 + w - 1)
            if is_border:
                if value != color:
                    return None
                continue
            if value == color:
                continue
            if value == bg:
                holes.append((int(r), int(c)))
                continue
            return None

    expected_size = h * w - len(holes) + 3
    if len(component) != expected_size or len(holes) > 3:
        return None
    return holes


def _parse_component(grid: np.ndarray, comp: dict, bg: int) -> dict:
    comp_bbox = _bbox_from_coords(comp["coords"])
    matches = []
    for direction in DIRECTIONS:
        block_bbox = _candidate_block_bbox(comp_bbox, direction)
        if block_bbox is None:
            continue
        holes = _validate_candidate(
            grid=grid,
            color=int(comp["color"]),
            bg=int(bg),
            coords=comp["coords"],
            direction=direction,
            bbox=block_bbox,
        )
        if holes is None:
            continue
        matches.append(
            {
                "color": int(comp["color"]),
                "bbox": block_bbox,
                "direction": direction,
                "copy_count": len(holes),
                "arrow": _arrow_cells(direction, block_bbox),
            }
        )
    assert len(matches) == 1, "expected each component to be exactly one block-plus-arrow object"
    return matches[0]


def _copy_bboxes(bbox: tuple[int, int, int, int], direction: str, copy_count: int):
    r0, c0, h, w = bbox
    for idx in range(copy_count):
        if direction == "right":
            yield r0, c0 + w + idx * (w + 1), h, w
        elif direction == "left":
            yield r0, c0 - w - idx * (w + 1), h, w
        elif direction == "down":
            yield r0 + h + idx * (h + 1), c0, h, w
        else:
            yield r0 - h - idx * (h + 1), c0, h, w


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each arrow-marked rectangular block along its arrow direction, with the number of solid copies equal to the number of internal holes.

    Input structure:
    - The background is black.
    - Each foreground object is a monochrome 3x3 to 5x5 rectangle whose border is solid and whose interior may contain 0 to 3 black hole pixels.
    - Exactly one 3-pixel arrow of the same color is attached outside one side of each rectangle, pointing up, down, left, or right.

    Rule:
    - For every block, count the number of black interior pixels inside the rectangle.
    - Remove the external arrow marker.
    - Keep the original rectangle unchanged, including its holes.
    - Along the arrow direction, place that many solid copies of the rectangle immediately beyond the source block, with a 1-cell black gap between consecutive copies.

    Key insight:
    - The rectangle interior encodes the repetition count, while the attached arrow independently encodes the translation direction.
    """
    g = np.asarray(grid, dtype=int)
    bg = _bg_color(g)
    out = g.copy()

    objects = [_parse_component(g, comp, bg) for comp in _components(g, bg)]
    for obj in objects:
        color = int(obj["color"])
        for r, c in obj["arrow"]:
            out[int(r), int(c)] = int(bg)
        for r0, c0, h, w in _copy_bboxes(obj["bbox"], obj["direction"], obj["copy_count"]):
            out[int(r0) : int(r0) + h, int(c0) : int(c0) + w] = color

    return out


def _shape_options(direction: str, holes: int) -> list[tuple[int, int]]:
    if direction in ("left", "right"):
        heights = (3, 5)
        widths = (3, 4, 5)
    else:
        heights = (3, 4, 5)
        widths = (3, 5)
    return [(h, w) for h in heights for w in widths if (h - 2) * (w - 2) >= holes]


def _sample_holes(h: int, w: int, hole_count: int) -> set[tuple[int, int]]:
    interior = [(r, c) for r in range(1, h - 1) for c in range(1, w - 1)]
    return set(random.sample(interior, hole_count))


def _relative_copy_origins(direction: str, h: int, w: int, count: int) -> list[tuple[int, int]]:
    origins = []
    for idx in range(count):
        if direction == "right":
            origins.append((0, w + idx * (w + 1)))
        elif direction == "left":
            origins.append((0, -w - idx * (w + 1)))
        elif direction == "down":
            origins.append((h + idx * (h + 1), 0))
        else:
            origins.append((-h - idx * (h + 1), 0))
    return origins


def _build_spec(color: int, direction: str, h: int, w: int, holes: set[tuple[int, int]]) -> dict:
    block_cells = {
        (r, c)
        for r in range(h)
        for c in range(w)
        if (r, c) not in holes
    }
    arrow_cells = set(_arrow_cells(direction, (0, 0, h, w)))
    copy_cells: set[tuple[int, int]] = set()
    for rr, cc in _relative_copy_origins(direction, h, w, len(holes)):
        for r in range(rr, rr + h):
            for c in range(cc, cc + w):
                copy_cells.add((int(r), int(c)))

    union_cells = block_cells | arrow_cells | copy_cells
    rows = [r for r, _c in union_cells]
    cols = [c for _r, c in union_cells]
    min_r, max_r = min(rows), max(rows)
    min_c, max_c = min(cols), max(cols)

    return {
        "color": int(color),
        "direction": direction,
        "h": int(h),
        "w": int(w),
        "holes": set(holes),
        "block_cells": block_cells,
        "arrow_cells": arrow_cells,
        "copy_cells": copy_cells,
        "input_cells": block_cells | arrow_cells,
        "output_cells": block_cells | copy_cells,
        "span_h": int(max_r - min_r + 1),
        "span_w": int(max_c - min_c + 1),
        "min_r": int(min_r),
        "max_r": int(max_r),
        "min_c": int(min_c),
        "max_c": int(max_c),
    }


def _stamp_cells(grid: np.ndarray, coords: set[tuple[int, int]], top: int, left: int, color: int) -> None:
    for r, c in coords:
        grid[int(top + r), int(left + c)] = int(color)


def _try_place_scene(specs: list[dict], height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.full((height, width), BG_COLOR, dtype=int)
    out = np.full((height, width), BG_COLOR, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    indexed_specs = list(enumerate(specs))
    random.shuffle(indexed_specs)
    indexed_specs.sort(
        key=lambda item: item[1]["span_h"] * item[1]["span_w"],
        reverse=True,
    )

    for _idx, spec in indexed_specs:
        row_lo = -spec["min_r"]
        row_hi = height - 1 - spec["max_r"]
        col_lo = -spec["min_c"]
        col_hi = width - 1 - spec["max_c"]
        if row_lo > row_hi or col_lo > col_hi:
            return None

        placed = False
        for _ in range(PLACEMENT_TRIES):
            top = random.randint(row_lo, row_hi)
            left = random.randint(col_lo, col_hi)
            rr0 = max(0, top + spec["min_r"] - 1)
            rr1 = min(height - 1, top + spec["max_r"] + 1)
            cc0 = max(0, left + spec["min_c"] - 1)
            cc1 = min(width - 1, left + spec["max_c"] + 1)
            if bool(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1].any()):
                continue
            if any(inp[top + r, left + c] != BG_COLOR for r, c in spec["input_cells"]):
                continue
            if any(out[top + r, left + c] != BG_COLOR for r, c in spec["output_cells"]):
                continue

            _stamp_cells(inp, spec["block_cells"], top, left, spec["color"])
            _stamp_cells(inp, spec["arrow_cells"], top, left, spec["color"])
            _stamp_cells(out, spec["block_cells"], top, left, spec["color"])
            _stamp_cells(out, spec["copy_cells"], top, left, spec["color"])
            reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True
            placed = True
            break

        if not placed:
            return None

    return inp, out


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.choices(OBJECT_COUNT_CHOICES, weights=OBJECT_COUNT_WEIGHTS, k=1)[0]
    colors = random.sample(list(range(1, 10)), n_objects)
    specs = []
    for color in colors:
        holes = random.randint(0, 3)
        direction = random.choice(DIRECTIONS)
        shape_options = _shape_options(direction, holes)
        if not shape_options:
            return None
        h, w = random.choice(shape_options)
        specs.append(_build_spec(color, direction, h, w, _sample_holes(h, w, holes)))

    min_h = max(spec["span_h"] for spec in specs)
    min_w = max(spec["span_w"] for spec in specs)
    low_h = max(min_h + 2, 8 + 3 * n_objects)
    low_w = max(min_w + 2, 8 + 4 * n_objects)
    if low_h > 30 or low_w > 30:
        return None

    for _ in range(SCENE_TRIES):
        height = random.randint(low_h, 30)
        width = random.randint(low_w, 30)
        placed = _try_place_scene(specs, height, width)
        if placed is None:
            continue
        inp, expected = placed
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, expected):
            return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid arrow-copy ARC input")
