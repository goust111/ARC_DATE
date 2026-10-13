from __future__ import annotations

import random
import sys
from pathlib import Path

import numpy as np


_ARC_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_ARC_UTILS_DIR) not in sys.path:
    sys.path.append(str(_ARC_UTILS_DIR))

from arc_utils import Color, find_components


TOP = "top"
BOTTOM = "bottom"
LEFT = "left"
RIGHT = "right"

ROLE_TO_SIDE = {
    "left": RIGHT,
    "right": LEFT,
    "top": BOTTOM,
    "bottom": TOP,
}


def _mask_from_coords(shape: tuple[int, int], coords: np.ndarray) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    mask[coords[:, 0], coords[:, 1]] = True
    return mask


def _longest_contiguous_span(indices: np.ndarray) -> tuple[int, int]:
    assert indices.size > 0
    best_lo = best_hi = int(indices[0])
    run_lo = int(indices[0])
    prev = int(indices[0])
    best_len = 1
    for raw in indices[1:]:
        value = int(raw)
        if value != prev + 1:
            run_len = prev - run_lo + 1
            if run_len > best_len:
                best_len = run_len
                best_lo, best_hi = run_lo, prev
            run_lo = value
        prev = value
    run_len = prev - run_lo + 1
    if run_len > best_len:
        best_lo, best_hi = run_lo, prev
    return best_lo, best_hi


def _centered_span(source_lo: int, source_hi: int, length: int) -> tuple[int, int]:
    center2 = int(source_lo) + int(source_hi)
    lo = (center2 - (int(length) - 1)) // 2
    hi = (center2 + (int(length) - 1)) // 2
    assert hi - lo + 1 == int(length)
    return int(lo), int(hi)


def _side_spans(mask: np.ndarray) -> dict[str, tuple[int, int]]:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    top = _longest_contiguous_span(np.flatnonzero(mask[r0, :]))
    bottom = _longest_contiguous_span(np.flatnonzero(mask[r1, :]))
    left = _longest_contiguous_span(np.flatnonzero(mask[:, c0]))
    right = _longest_contiguous_span(np.flatnonzero(mask[:, c1]))
    return {
        TOP: top,
        BOTTOM: bottom,
        LEFT: left,
        RIGHT: right,
    }


def _stripe_mask(mask: np.ndarray, blocked: np.ndarray) -> np.ndarray:
    spans = _side_spans(mask)
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    r0 = int(r0)
    c0 = int(c0)
    r1 = int(r1)
    c1 = int(c1)

    out = np.zeros_like(mask, dtype=bool)
    height, width = mask.shape

    def paint_row(row: int, lo: int, hi: int) -> None:
        if not 0 <= int(row) < height:
            return
        for col in range(int(lo), int(hi) + 1):
            if 0 <= col < width and not blocked[int(row), col]:
                out[int(row), col] = True

    def paint_col(col: int, lo: int, hi: int) -> None:
        if not 0 <= int(col) < width:
            return
        for row in range(int(lo), int(hi) + 1):
            if 0 <= row < height and not blocked[row, int(col)]:
                out[row, int(col)] = True

    for side, (span_lo, span_hi) in spans.items():
        source_len = int(span_hi) - int(span_lo) + 1
        k = 1
        while True:
            length = source_len - 2 * k
            if length <= 0:
                break
            lo, hi = _centered_span(span_lo, span_hi, length)
            if side == TOP:
                paint_row(r0 - k, lo, hi)
            elif side == BOTTOM:
                paint_row(r1 + k, lo, hi)
            elif side == LEFT:
                paint_col(c0 - k, lo, hi)
            else:
                paint_col(c1 + k, lo, hi)
            k += 1
    return out


def _extract_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    components = find_components(
        g,
        bg=Color.BLACK,
        colors=[Color.TEAL],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    components.sort(key=lambda comp: tuple(map(int, comp["bbox"])))
    return components


def _coverage_masks(grid: np.ndarray) -> tuple[list[dict], list[np.ndarray]]:
    g = np.asarray(grid, dtype=int)
    objects = _extract_objects(g)
    blocked = g == Color.TEAL
    covers = []
    for obj in objects:
        mask = _mask_from_coords(g.shape, obj["coords"])
        covers.append(_stripe_mask(mask, blocked))
    return objects, covers


def main(grid: np.ndarray) -> np.ndarray:
    """Mark collision zones where shrinking four-way stripe expansions from teal objects overlap.

    Input structure:
    - The grid is a black background with 3-4 disconnected teal 4-connected objects.
    - Each object is compact and rounded enough that its top, bottom, left, and right boundary runs are contiguous and at least three pixels long.
    - Objects do not touch each other, and only black background surrounds them.

    Rule:
    - Keep every teal object pixel unchanged.
    - For each object separately, measure the longest contiguous teal run on its topmost row, bottommost row, leftmost column, and rightmost column.
    - From each side, emit outward shrinking stripes: layer k has length side_run - 2k, stays centered on that side run, and is painted only on background cells.
    - Background cells covered by stripes from exactly one object become yellow; background cells covered by stripes from two or more different objects become red.

    Key insight:
    - Red cells are not tied to any single object; they emerge only after computing every object's full shrinking expansion and then intersecting those independent coverage regions.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert max(g.shape) <= 30
    assert set(map(int, np.unique(g))) <= {Color.BLACK, Color.TEAL}

    objects, covers = _coverage_masks(g)
    assert 3 <= len(objects) <= 4

    out = g.copy()
    cover_count = np.zeros(g.shape, dtype=int)
    for cover in covers:
        cover_count += cover.astype(int)

    bg_mask = g == Color.BLACK
    out[bg_mask & (cover_count == 1)] = Color.YELLOW
    out[bg_mask & (cover_count >= 2)] = Color.RED
    return out


def _rect_mask(height: int, width: int) -> np.ndarray:
    return np.ones((int(height), int(width)), dtype=bool)


def _cut_mask(height: int, width: int, holes: list[tuple[int, int]]) -> np.ndarray:
    mask = np.ones((int(height), int(width)), dtype=bool)
    for r, c in holes:
        mask[int(r), int(c)] = False
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    seen = {start}
    stack = [start]
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc]:
                nxt = (nr, nc)
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
    return len(seen) == int(mask.sum())


def _shape_library() -> list[dict]:
    masks = [
        ("rect3x3", _rect_mask(3, 3)),
        ("rect3x4", _rect_mask(3, 4)),
        ("rect4x3", _rect_mask(4, 3)),
        ("rect3x5", _rect_mask(3, 5)),
        ("rect5x3", _rect_mask(5, 3)),
        ("rect4x4", _rect_mask(4, 4)),
        ("diag_a", _cut_mask(4, 4, [(0, 0), (3, 3)])),
        ("diag_b", _cut_mask(4, 4, [(0, 3), (3, 0)])),
        ("cut_tl", _cut_mask(4, 4, [(0, 0)])),
        ("cut_tr", _cut_mask(4, 4, [(0, 3)])),
        ("cut_bl", _cut_mask(4, 4, [(3, 0)])),
        ("cut_br", _cut_mask(4, 4, [(3, 3)])),
    ]

    catalog = []
    for name, mask in masks:
        area = int(mask.sum())
        if not (8 <= area <= 16):
            continue
        if not _is_connected(mask):
            continue
        spans = _side_spans(mask)
        if min(hi - lo + 1 for lo, hi in spans.values()) < 3:
            continue
        coords = np.argwhere(mask)
        bbox = (
            int(coords[:, 0].min()),
            int(coords[:, 1].min()),
            int(coords[:, 0].max()),
            int(coords[:, 1].max()),
        )
        side_info = {}
        for side, (lo, hi) in spans.items():
            length = int(hi) - int(lo) + 1
            side_info[side] = {
                "lo": int(lo),
                "hi": int(hi),
                "length": length,
                "reach": (length - 1) // 2,
                "center": (int(lo) + int(hi)) // 2,
                "odd": length % 2 == 1,
            }
        catalog.append(
            {
                "name": name,
                "mask": mask,
                "bbox": bbox,
                "h": int(mask.shape[0]),
                "w": int(mask.shape[1]),
                "area": area,
                "side_info": side_info,
                "is_rect": area == int(mask.shape[0] * mask.shape[1]),
            }
        )
    return catalog


_SHAPES = _shape_library()


def _role_catalog(role: str) -> list[dict]:
    inward_side = ROLE_TO_SIDE[role]
    return [
        shape
        for shape in _SHAPES
        if shape["side_info"][inward_side]["odd"] and shape["side_info"][inward_side]["reach"] >= 1
    ]


def _paint_object(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int) -> None:
    rr, cc = np.argwhere(mask).T
    grid[rr + int(r0), cc + int(c0)] = Color.TEAL


def _object_pixels(mask: np.ndarray, r0: int, c0: int) -> set[tuple[int, int]]:
    return {
        (int(r + r0), int(c + c0))
        for r, c in np.argwhere(mask)
    }


def _min_pixel_distance(a: set[tuple[int, int]], b: set[tuple[int, int]]) -> int:
    return min(abs(ar - br) + abs(ac - bc) for ar, ac in a for br, bc in b)


def _bbox_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _placed_bbox(shape: dict, r0: int, c0: int) -> tuple[int, int, int, int]:
    return int(r0), int(c0), int(r0 + shape["h"] - 1), int(c0 + shape["w"] - 1)


def _pair_collision_count(covers: list[np.ndarray]) -> int:
    count = 0
    for i in range(len(covers)):
        for j in range(i + 1, len(covers)):
            if bool((covers[i] & covers[j]).any()):
                count += 1
    return count


def _input_signature(grid: np.ndarray) -> tuple:
    objects = _extract_objects(grid)
    sig = []
    for obj in objects:
        mask = _mask_from_coords(grid.shape, obj["coords"])
        coords = obj["coords"]
        r0, c0 = coords.min(axis=0)
        local = sorted((int(r - r0), int(c - c0)) for r, c in coords)
        spans = _side_spans(mask)
        sig.append((tuple(local), tuple(sorted(int(hi - lo + 1) for lo, hi in spans.values()))))
    return tuple(sig)


def _validate_example(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if set(map(int, np.unique(grid))) != {Color.BLACK, Color.TEAL}:
        return False

    objects, covers = _coverage_masks(grid)
    if len(objects) < 3:
        return False

    spans_per_object = []
    local_signatures = set()
    non_rect = 0
    pixels = []
    bboxes = []
    for obj in objects:
        if not (8 <= int(obj["area"]) <= 16):
            return False
        mask = _mask_from_coords(grid.shape, obj["coords"])
        spans = _side_spans(mask)
        lengths = tuple(int(hi - lo + 1) for lo, hi in (spans[TOP], spans[BOTTOM], spans[LEFT], spans[RIGHT]))
        if min(lengths) < 3:
            return False
        spans_per_object.append(lengths)
        local = obj["coords"] - obj["coords"].min(axis=0)
        local_key = tuple(sorted((int(r), int(c)) for r, c in local))
        local_signatures.add(local_key)
        if int(obj["area"]) < int((obj["bbox"][2] - obj["bbox"][0] + 1) * (obj["bbox"][3] - obj["bbox"][1] + 1)):
            non_rect += 1
        pix = {(int(r), int(c)) for r, c in obj["coords"]}
        pixels.append(pix)
        bboxes.append(tuple(map(int, obj["bbox"])))

    for i in range(len(objects)):
        for j in range(i + 1, len(objects)):
            if _bbox_overlap(bboxes[i], bboxes[j]):
                return False
            if _min_pixel_distance(pixels[i], pixels[j]) < 2:
                return False

    out = main(grid)
    red = int((out == Color.RED).sum())
    yellow = int((out == Color.YELLOW).sum())
    if red < 1 or yellow < 6:
        return False
    if red > yellow * 2 or red > 18:
        return False
    if _pair_collision_count(covers) < 2:
        return False
    if len(local_signatures) < 2:
        return False
    if len(set(spans_per_object)) < 2:
        return False
    if len(objects) == 4 and non_rect == 0:
        return False
    if len(set(map(int, out.flatten()))) < 4:
        return False
    return True


def _build_candidate() -> np.ndarray | None:
    height = random.randint(18, 24)
    width = random.randint(18, 24)
    grid = np.zeros((height, width), dtype=int)

    object_count = random.choice([3, 4])
    orientation = random.choice(["horizontal", "vertical"])

    if orientation == "horizontal":
        left_catalog = [shape for shape in _SHAPES if shape["side_info"][RIGHT]["odd"] and shape["side_info"][RIGHT]["reach"] >= 1]
        right_catalog = [shape for shape in _SHAPES if shape["side_info"][LEFT]["odd"] and shape["side_info"][LEFT]["reach"] >= 1]
        middle_catalog = [
            shape
            for shape in _SHAPES
            if shape["side_info"][LEFT]["odd"]
            and shape["side_info"][RIGHT]["odd"]
            and shape["side_info"][LEFT]["reach"] >= 1
            and shape["side_info"][RIGHT]["reach"] >= 1
            and shape["side_info"][LEFT]["center"] == shape["side_info"][RIGHT]["center"]
        ]
        weights = lambda shape: 4 + (0 if shape["is_rect"] else 4) + max(shape["side_info"][LEFT]["reach"], shape["side_info"][RIGHT]["reach"])
        left_shape = random.choice([shape for shape in left_catalog for _ in range(weights(shape))])
        middle_shape = random.choice([shape for shape in middle_catalog for _ in range(weights(shape))])
        right_shape = random.choice([shape for shape in right_catalog for _ in range(weights(shape))])

        max_gap_1 = left_shape["side_info"][RIGHT]["reach"] + middle_shape["side_info"][LEFT]["reach"] - 1
        max_gap_2 = middle_shape["side_info"][RIGHT]["reach"] + right_shape["side_info"][LEFT]["reach"] - 1
        if max_gap_1 < 1 or max_gap_2 < 1:
            return None
        gap_1 = random.randint(1, max_gap_1)
        gap_2 = random.randint(1, max_gap_2)

        row_min = max(
            left_shape["side_info"][RIGHT]["center"],
            middle_shape["side_info"][LEFT]["center"],
            right_shape["side_info"][LEFT]["center"],
        )
        row_max = min(
            height - 1 - (left_shape["h"] - 1 - left_shape["side_info"][RIGHT]["center"]),
            height - 1 - (middle_shape["h"] - 1 - middle_shape["side_info"][LEFT]["center"]),
            height - 1 - (right_shape["h"] - 1 - right_shape["side_info"][LEFT]["center"]),
        )
        c0_min = left_shape["w"] + gap_1
        c0_max = width - middle_shape["w"] - gap_2 - right_shape["w"]
        if row_min > row_max or c0_min > c0_max:
            return None
        common_row = random.randint(row_min, row_max)
        middle_c0 = random.randint(c0_min, c0_max)

        placements = {
            "left": (
                common_row - left_shape["side_info"][RIGHT]["center"],
                middle_c0 - gap_1 - left_shape["w"],
            ),
            "middle": (
                common_row - middle_shape["side_info"][LEFT]["center"],
                middle_c0,
            ),
            "right": (
                common_row - right_shape["side_info"][LEFT]["center"],
                middle_c0 + middle_shape["w"] + gap_2,
            ),
        }
        core_shapes = {
            "left": left_shape,
            "middle": middle_shape,
            "right": right_shape,
        }
    else:
        top_catalog = [shape for shape in _SHAPES if shape["side_info"][BOTTOM]["odd"] and shape["side_info"][BOTTOM]["reach"] >= 1]
        bottom_catalog = [shape for shape in _SHAPES if shape["side_info"][TOP]["odd"] and shape["side_info"][TOP]["reach"] >= 1]
        middle_catalog = [
            shape
            for shape in _SHAPES
            if shape["side_info"][TOP]["odd"]
            and shape["side_info"][BOTTOM]["odd"]
            and shape["side_info"][TOP]["reach"] >= 1
            and shape["side_info"][BOTTOM]["reach"] >= 1
            and shape["side_info"][TOP]["center"] == shape["side_info"][BOTTOM]["center"]
        ]
        weights = lambda shape: 4 + (0 if shape["is_rect"] else 4) + max(shape["side_info"][TOP]["reach"], shape["side_info"][BOTTOM]["reach"])
        top_shape = random.choice([shape for shape in top_catalog for _ in range(weights(shape))])
        middle_shape = random.choice([shape for shape in middle_catalog for _ in range(weights(shape))])
        bottom_shape = random.choice([shape for shape in bottom_catalog for _ in range(weights(shape))])

        max_gap_1 = top_shape["side_info"][BOTTOM]["reach"] + middle_shape["side_info"][TOP]["reach"] - 1
        max_gap_2 = middle_shape["side_info"][BOTTOM]["reach"] + bottom_shape["side_info"][TOP]["reach"] - 1
        if max_gap_1 < 1 or max_gap_2 < 1:
            return None
        gap_1 = random.randint(1, max_gap_1)
        gap_2 = random.randint(1, max_gap_2)

        col_min = max(
            top_shape["side_info"][BOTTOM]["center"],
            middle_shape["side_info"][TOP]["center"],
            bottom_shape["side_info"][TOP]["center"],
        )
        col_max = min(
            width - 1 - (top_shape["w"] - 1 - top_shape["side_info"][BOTTOM]["center"]),
            width - 1 - (middle_shape["w"] - 1 - middle_shape["side_info"][TOP]["center"]),
            width - 1 - (bottom_shape["w"] - 1 - bottom_shape["side_info"][TOP]["center"]),
        )
        r0_min = top_shape["h"] + gap_1
        r0_max = height - middle_shape["h"] - gap_2 - bottom_shape["h"]
        if col_min > col_max or r0_min > r0_max:
            return None
        common_col = random.randint(col_min, col_max)
        middle_r0 = random.randint(r0_min, r0_max)

        placements = {
            "top": (
                middle_r0 - gap_1 - top_shape["h"],
                common_col - top_shape["side_info"][BOTTOM]["center"],
            ),
            "middle": (
                middle_r0,
                common_col - middle_shape["side_info"][TOP]["center"],
            ),
            "bottom": (
                middle_r0 + middle_shape["h"] + gap_2,
                common_col - bottom_shape["side_info"][TOP]["center"],
            ),
        }
        core_shapes = {
            "top": top_shape,
            "middle": middle_shape,
            "bottom": bottom_shape,
        }

    core_pixels: list[set[tuple[int, int]]] = []
    for name, shape in core_shapes.items():
        r0, c0 = placements[name]
        if r0 < 0 or c0 < 0 or r0 + shape["h"] > height or c0 + shape["w"] > width:
            return None
        core_pixels.append(_object_pixels(shape["mask"], r0, c0))

    for i in range(len(core_pixels)):
        for j in range(i + 1, len(core_pixels)):
            if core_pixels[i] & core_pixels[j]:
                return None
            if _min_pixel_distance(core_pixels[i], core_pixels[j]) < 2:
                return None

    for name, shape in core_shapes.items():
        _paint_object(grid, shape["mask"], *placements[name])

    if object_count == 4:
        occupied = set().union(*core_pixels)
        non_rect_shapes = [shape for shape in _SHAPES if not shape["is_rect"]]
        random.shuffle(non_rect_shapes)
        for shape in non_rect_shapes:
            candidates = [
                (0, 0),
                (0, width - shape["w"]),
                (height - shape["h"], 0),
                (height - shape["h"], width - shape["w"]),
            ]
            random.shuffle(candidates)
            for base_r, base_c in candidates:
                for jitter_r in range(3):
                    for jitter_c in range(3):
                        r0 = base_r + jitter_r if base_r == 0 else base_r - jitter_r
                        c0 = base_c + jitter_c if base_c == 0 else base_c - jitter_c
                        if r0 < 0 or c0 < 0 or r0 + shape["h"] > height or c0 + shape["w"] > width:
                            continue
                        pix = _object_pixels(shape["mask"], r0, c0)
                        if occupied & pix:
                            continue
                        if min(_min_pixel_distance(pix, prev) for prev in core_pixels) < 4:
                            continue
                        _paint_object(grid, shape["mask"], r0, c0)
                        return grid
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        candidate = _build_candidate()
        if candidate is None:
            continue
        if _validate_example(candidate):
            return candidate
    raise RuntimeError("failed to generate a valid non-degenerate ARC input")
