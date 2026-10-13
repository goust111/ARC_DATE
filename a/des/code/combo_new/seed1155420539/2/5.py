from collections import deque

import numpy as np


_RNG = np.random.default_rng()
_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_GENERATION_RETRIES = 400

_SHAPES = (
    ((0, 0),),
    ((0, 0), (0, 1)),
    ((0, 0), (1, 0)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (1, 0), (2, 0)),
    ((0, 0), (0, 1), (1, 0)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (1, 0), (1, 1)),
    ((0, 1), (1, 0), (1, 1)),
)


def _background(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros(mask.shape, dtype=bool)
    comps = []
    for sr in range(h):
        for sc in range(w):
            if not mask[sr, sc] or seen[sr, sc]:
                continue
            seen[sr, sc] = True
            q = deque([(sr, sc)])
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_frame_component(grid: np.ndarray, coords: np.ndarray, color: int, bg: int) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 4 or w < 3 or h - 2 not in (2, 3):
        return False

    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True

    actual = np.zeros((h, w), dtype=bool)
    actual[coords[:, 0] - r0, coords[:, 1] - c0] = True
    if not np.array_equal(actual, expected):
        return False

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    return bool(np.all(interior == bg) and np.all(grid[coords[:, 0], coords[:, 1]] == color))


def _shape_from_component(coords: np.ndarray) -> dict:
    r0, c0, r1, c1 = _bbox(coords)
    rel = coords - np.array([r0, c0], dtype=int)
    return {
        "coords": coords,
        "bbox": (r0, c0, r1, c1),
        "rel": rel,
        "height": r1 - r0 + 1,
        "width": c1 - c0 + 1,
        "source_col": c0,
        "center_row": float(coords[:, 0].mean()),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort colored fragments into same-colored storage frames by their vertical side.

    Input structure:
    - The background is the most common color.
    - Each storage bin is a one-pixel-thick hollow rectangle whose interior is background and whose border color is unique.
    - Outside the bins are small 4-connected fragments of one to three pixels; every fragment has the color of exactly one bin.

    Rule:
    - Match every fragment to the bin with the same border color and clear its original cells to background.
    - A fragment whose center row is above its bin center is stamped into the top interior row; a lower fragment is stamped into the bottom interior row.
    - Within each side of a bin, fragments are ordered by their original left column and packed from the interior left edge with one background column between bounding boxes.
    - Bin borders remain unchanged.

    Key insight:
    - The color selects the destination frame, while the fragment's global up/down position selects the local interior row and its original horizontal order selects the packed slot.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _background(g)

    frames = {}
    fragments_by_color = {}
    for color in [int(c) for c in np.unique(g) if int(c) != bg]:
        comps = _components(g == color)
        frame_indices = [i for i, comp in enumerate(comps) if _is_frame_component(g, comp, color, bg)]
        assert len(frame_indices) == 1, "each active color must have one storage frame"

        frame_index = frame_indices[0]
        frame_bbox = _bbox(comps[frame_index])
        frames[color] = frame_bbox
        fragments = []
        for i, comp in enumerate(comps):
            if i == frame_index:
                continue
            assert 1 <= len(comp) <= 3, "external fragments must have one to three cells"
            fragments.append(_shape_from_component(comp))
        fragments_by_color[color] = fragments

    out = g.copy()
    for color, fragments in fragments_by_color.items():
        for fragment in fragments:
            coords = fragment["coords"]
            out[coords[:, 0], coords[:, 1]] = bg

    for color, frame_bbox in frames.items():
        r0, c0, r1, c1 = frame_bbox
        frame_center = (r0 + r1) / 2.0
        interior_top = r0 + 1
        interior_bottom = r1 - 1
        interior_left = c0 + 1
        interior_right = c1 - 1

        by_side = {"top": [], "bottom": []}
        for fragment in fragments_by_color[color]:
            assert fragment["center_row"] != frame_center
            side = "top" if fragment["center_row"] < frame_center else "bottom"
            by_side[side].append(fragment)

        for side in ("top", "bottom"):
            cursor = interior_left
            ordered = sorted(by_side[side], key=lambda x: (x["source_col"], x["bbox"][0]))
            for fragment in ordered:
                height = fragment["height"]
                width = fragment["width"]
                if side == "top":
                    dest_r0 = interior_top
                else:
                    dest_r0 = interior_bottom - height + 1
                dest_c0 = cursor

                rel = fragment["rel"]
                rows = dest_r0 + rel[:, 0]
                cols = dest_c0 + rel[:, 1]
                assert rows.min() >= interior_top and rows.max() <= interior_bottom
                assert cols.min() >= interior_left and cols.max() <= interior_right

                target = out[rows, cols]
                assert bool(np.all((target == bg) | (target == color)))
                out[rows, cols] = color
                cursor += width + 1

    return out


def _shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    rows = [p[0] for p in shape]
    cols = [p[1] for p in shape]
    return max(rows) + 1, max(cols) + 1


def _row_width(shapes: list[tuple[tuple[int, int], ...]]) -> int:
    if not shapes:
        return 0
    return sum(_shape_dims(shape)[1] for shape in shapes) + len(shapes) - 1


def _layout_ok(interior_h: int, interior_w: int, top_shapes, bottom_shapes) -> bool:
    occ = np.zeros((interior_h, interior_w), dtype=bool)
    for side, shapes in (("top", top_shapes), ("bottom", bottom_shapes)):
        cursor = 0
        for shape in shapes:
            sh, sw = _shape_dims(shape)
            if sh > interior_h or cursor + sw > interior_w:
                return False
            row0 = 0 if side == "top" else interior_h - sh
            rows = np.array([row0 + r for r, _c in shape], dtype=int)
            cols = np.array([cursor + c for _r, c in shape], dtype=int)
            if np.any(occ[rows, cols]):
                return False
            occ[rows, cols] = True
            cursor += sw + 1
    return True


def _sample_shapes(count: int, interior_h: int) -> list[tuple[tuple[int, int], ...]]:
    choices = [shape for shape in _SHAPES if _shape_dims(shape)[0] <= interior_h]
    indices = _RNG.integers(0, len(choices), size=count)
    return [choices[int(i)] for i in indices]


def _sample_specs(n_boxes: int) -> list[dict] | None:
    specs = []
    has_double = False
    count_options = ((1, 1), (2, 1), (1, 2))
    for _ in range(n_boxes):
        interior_h = int(_RNG.integers(2, 4))
        top_count, bottom_count = count_options[int(_RNG.integers(0, len(count_options)))]
        has_double = has_double or top_count == 2 or bottom_count == 2

        for _try in range(100):
            top_shapes = _sample_shapes(top_count, interior_h)
            bottom_shapes = _sample_shapes(bottom_count, interior_h)
            min_w = max(_row_width(top_shapes), _row_width(bottom_shapes), 4)
            interior_w = min_w + int(_RNG.integers(0, 4))
            if _layout_ok(interior_h, interior_w, top_shapes, bottom_shapes):
                specs.append(
                    {
                        "interior_h": interior_h,
                        "interior_w": interior_w,
                        "top_shapes": top_shapes,
                        "bottom_shapes": bottom_shapes,
                    }
                )
                break
        else:
            return None

    if not has_double:
        return None
    return specs


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color


def _shape_cells(shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> tuple[np.ndarray, np.ndarray]:
    rows = np.array([r0 + r for r, _c in shape], dtype=int)
    cols = np.array([c0 + c for _r, c in shape], dtype=int)
    return rows, cols


def _can_stamp(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, bg: int) -> bool:
    h, w = grid.shape
    sh, sw = _shape_dims(shape)
    if r0 < 1 or c0 < 1 or r0 + sh > h - 1 or c0 + sw > w - 1:
        return False

    rows, cols = _shape_cells(shape, r0, c0)
    if np.any(grid[rows, cols] != bg):
        return False

    for r, c in zip(rows, cols):
        if np.any(grid[r - 1 : r + 2, c - 1 : c + 2] != bg):
            return False
    return True


def _stamp_shape(grid: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int, color: int) -> None:
    rows, cols = _shape_cells(shape, r0, c0)
    grid[rows, cols] = color


def _source_columns(frame_bbox: tuple[int, int, int, int], shapes) -> list[int]:
    _r0, c0, _r1, c1 = frame_bbox
    widths = [_shape_dims(shape)[1] for shape in shapes]
    min_total = sum(widths) + max(0, len(widths) - 1)
    interior_w = c1 - c0 - 1
    slack = interior_w - min_total
    assert slack >= 0

    left_slack = int(_RNG.integers(0, slack + 1))
    gap_count = max(0, len(widths) - 1)
    gaps = [1] * gap_count
    for _ in range(slack - left_slack):
        if gap_count:
            gaps[int(_RNG.integers(0, gap_count))] += 1

    cols = []
    cursor = c0 + 1 + left_slack
    for i, width in enumerate(widths):
        cols.append(cursor)
        cursor += width
        if i < gap_count:
            cursor += gaps[i]
    return cols


def _place_external_side(grid: np.ndarray, frame_bbox, shapes, color: int, side: str) -> bool:
    h, _w = grid.shape
    r0, _c0, r1, _c1 = frame_bbox
    cols = _source_columns(frame_bbox, shapes)
    for shape, col in zip(shapes, cols):
        sh, _sw = _shape_dims(shape)
        if side == "top":
            low, high = 1, r0 - sh - 1
        else:
            low, high = r1 + 2, h - sh - 1
        if low > high:
            return False

        placed = False
        row_choices = list(range(low, high + 1))
        _RNG.shuffle(row_choices)
        for row in row_choices[:30]:
            if _can_stamp(grid, shape, row, col, _BG):
                _stamp_shape(grid, shape, row, col, color)
                placed = True
                break
        if not placed:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(18, 31))
    width = int(_RNG.integers(18, 31))
    n_boxes = int(_RNG.integers(2, 4))

    specs = _sample_specs(n_boxes)
    if specs is None:
        return None

    outer_widths = [spec["interior_w"] + 2 for spec in specs]
    gaps = [int(_RNG.integers(2, 5)) for _ in range(n_boxes - 1)]
    total_width = sum(outer_widths) + sum(gaps)
    if total_width > width - 2:
        return None

    max_left = width - total_width - 1
    if max_left < 1:
        return None
    start_col = int(_RNG.integers(1, max_left + 1))

    colors = [int(c) for c in _RNG.choice(_COLOR_POOL, size=n_boxes, replace=False)]
    grid = np.full((height, width), _BG, dtype=int)

    frame_bboxes = []
    cursor = start_col
    for spec, outer_w in zip(specs, outer_widths):
        outer_h = spec["interior_h"] + 2
        max_top = height - outer_h - 6
        if max_top < 6:
            return None
        top = int(_RNG.integers(6, max_top + 1))
        bbox = (top, cursor, top + outer_h - 1, cursor + outer_w - 1)
        frame_bboxes.append(bbox)
        cursor += outer_w
        if len(frame_bboxes) <= len(gaps):
            cursor += gaps[len(frame_bboxes) - 1]

    for bbox, color in zip(frame_bboxes, colors):
        _draw_frame(grid, bbox, color)

    order = np.arange(n_boxes)
    _RNG.shuffle(order)
    for idx in order:
        spec = specs[int(idx)]
        bbox = frame_bboxes[int(idx)]
        color = colors[int(idx)]
        if not _place_external_side(grid, bbox, spec["top_shapes"], color, "top"):
            return None
        if not _place_external_side(grid, bbox, spec["bottom_shapes"], color, "bottom"):
            return None

    if _background(grid) != _BG:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid storage-frame sorting input")
