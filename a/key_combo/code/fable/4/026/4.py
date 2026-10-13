import random
from collections import Counter, deque
from dataclasses import dataclass

import numpy as np


DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
SIDES = ("top", "bottom", "left", "right")
HORIZONTAL_SIDES = {"top", "bottom"}
VERTICAL_SIDES = {"left", "right"}
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
BACKGROUND = 0

GRID_SIZE_MAX = 30
OBJECT_GAP = 1
GEN_RETRIES = 400
SPEC_RETRIES = 120
BASE_HEIGHT_RANGE = (5, 9)
BASE_WIDTH_RANGE = (5, 9)
FEATURE_LENGTH_CAP = 4
BACKGROUND_FRACTION_MIN = 0.45


@dataclass(frozen=True)
class Feature:
    side: str
    kind: str
    start: int
    length: int


@dataclass(frozen=True)
class ObjectSpec:
    color: int
    features: tuple[Feature, ...]
    input_coords: tuple[tuple[int, int], ...]
    output_coords: tuple[tuple[int, int], ...]
    input_bbox: tuple[int, int, int, int]
    output_bbox: tuple[int, int, int, int]
    overall_bbox: tuple[int, int, int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    return 0 <= r < shape[0] and 0 <= c < shape[1]


def bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def relative_bbox(coords: tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def union_bbox(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> tuple[int, int, int, int]:
    return min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])


def translate_bbox(
    bbox: tuple[int, int, int, int], anchor: tuple[int, int]
) -> tuple[int, int, int, int]:
    r0, c0 = anchor
    return bbox[0] + r0, bbox[1] + c0, bbox[2] + r0, bbox[3] + c0


def bbox_area(bbox: tuple[int, int, int, int]) -> int:
    return (bbox[2] - bbox[0] + 1) * (bbox[3] - bbox[1] + 1)


def boxes_separated(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int
) -> bool:
    return (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = map(int, grid.shape)
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []
    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == bg:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if (
                        in_bounds((h, w), nr, nc)
                        and not visited[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        visited[nr, nc] = True
                        queue.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {"color": color, "coords": coords, "bbox": bbox_from_coords(coords)}
            )
    return components


def coords_connected(coords: tuple[tuple[int, int], ...]) -> bool:
    cells = set(coords)
    start = next(iter(cells))
    seen = {start}
    queue = deque([start])
    while queue:
        r, c = queue.popleft()
        for dr, dc in DIRS4:
            nxt = (r + dr, c + dc)
            if nxt in cells and nxt not in seen:
                seen.add(nxt)
                queue.append(nxt)
    return len(seen) == len(cells)


def _contiguous_span(line: np.ndarray) -> tuple[int, int] | None:
    idx = np.flatnonzero(line)
    if idx.size == 0:
        return None
    if int(idx[-1] - idx[0] + 1) != int(idx.size):
        raise ValueError("feature cells must form one contiguous segment")
    return int(idx[0]), int(idx.size)


def _feature_span_on_line(line: np.ndarray) -> tuple[int, int] | None:
    line = np.asarray(line, dtype=bool).reshape(-1)
    if line.size < 3:
        if line.any():
            raise ValueError("feature cannot touch rectangle corners")
        return None
    span = _contiguous_span(line)
    if span is None:
        return None
    start, length = span
    if start == 0 or start + length == line.size:
        raise ValueError("feature segment must lie on the middle part of a side")
    return start, length


def _recover_object(
    component: dict,
) -> tuple[tuple[int, int, int, int], tuple[Feature, ...]]:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    coords = component["coords"]
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    h, w = map(int, mask.shape)

    candidates = []
    for top_shift in (0, 1):
        for bottom_shift in (0, 1):
            for left_shift in (0, 1):
                for right_shift in (0, 1):
                    top = top_shift
                    bottom = h - 1 - bottom_shift
                    left = left_shift
                    right = w - 1 - right_shift
                    if top > bottom or left > right:
                        continue

                    rect = np.zeros_like(mask, dtype=bool)
                    rect[top : bottom + 1, left : right + 1] = True
                    missing = rect & ~mask
                    extra = mask & ~rect
                    covered_missing = np.zeros_like(mask, dtype=bool)
                    covered_extra = np.zeros_like(mask, dtype=bool)
                    features: list[Feature] = []

                    try:
                        for side in SIDES:
                            if side == "top":
                                tab_line = extra[top - 1, left : right + 1] if top > 0 else np.zeros(right - left + 1, dtype=bool)
                                notch_line = missing[top, left : right + 1]
                            elif side == "bottom":
                                tab_line = extra[bottom + 1, left : right + 1] if bottom + 1 < h else np.zeros(right - left + 1, dtype=bool)
                                notch_line = missing[bottom, left : right + 1]
                            elif side == "left":
                                tab_line = extra[top : bottom + 1, left - 1] if left > 0 else np.zeros(bottom - top + 1, dtype=bool)
                                notch_line = missing[top : bottom + 1, left]
                            else:
                                tab_line = extra[top : bottom + 1, right + 1] if right + 1 < w else np.zeros(bottom - top + 1, dtype=bool)
                                notch_line = missing[top : bottom + 1, right]

                            tab_span = _feature_span_on_line(tab_line)
                            notch_span = _feature_span_on_line(notch_line)
                            if tab_span is not None and notch_span is not None:
                                raise ValueError("a side cannot be both a tab and a notch")

                            if tab_span is not None:
                                start, length = tab_span
                                features.append(Feature(side, "tab", start, length))
                                if side == "top":
                                    covered_extra[top - 1, left + start : left + start + length] = True
                                elif side == "bottom":
                                    covered_extra[bottom + 1, left + start : left + start + length] = True
                                elif side == "left":
                                    covered_extra[top + start : top + start + length, left - 1] = True
                                else:
                                    covered_extra[top + start : top + start + length, right + 1] = True
                            elif notch_span is not None:
                                start, length = notch_span
                                features.append(Feature(side, "notch", start, length))
                                if side == "top":
                                    covered_missing[top, left + start : left + start + length] = True
                                elif side == "bottom":
                                    covered_missing[bottom, left + start : left + start + length] = True
                                elif side == "left":
                                    covered_missing[top + start : top + start + length, left] = True
                                else:
                                    covered_missing[top + start : top + start + length, right] = True
                    except (AssertionError, ValueError):
                        continue

                    if not np.array_equal(missing, covered_missing):
                        continue
                    if not np.array_equal(extra, covered_extra):
                        continue
                    if not (1 <= len(features) <= 2):
                        continue

                    base_bbox = (r0 + top, c0 + left, r0 + bottom, c0 + right)
                    signature = tuple((f.side, f.kind, f.start, f.length) for f in sorted(features, key=lambda x: (x.side, x.kind, x.start, x.length)))
                    candidates.append((bbox_area((top, left, bottom, right)), len(features), signature, base_bbox, tuple(features)))

    if not candidates:
        raise AssertionError("failed to recover rectangle and features")
    candidates.sort(key=lambda item: (item[0], item[1], item[2]), reverse=True)
    _area, _count, _signature, base_bbox, features = candidates[0]
    return base_bbox, features


def _transform_bbox(
    base_bbox: tuple[int, int, int, int], features: tuple[Feature, ...]
) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = base_bbox
    for feature in features:
        k = feature.length
        if feature.side == "top":
            r1 += k if feature.kind == "notch" else -k
        elif feature.side == "bottom":
            r0 -= k if feature.kind == "notch" else k
        elif feature.side == "left":
            c1 += k if feature.kind == "notch" else -k
        else:
            c0 -= k if feature.kind == "notch" else k
    assert r0 <= r1 and c0 <= c1
    return r0, c0, r1, c1


def main(grid: np.ndarray) -> np.ndarray:
    """Treat each boundary tab or notch as an instruction that changes the opposite side of its own rectangle.

    Input structure:
    - The grid has a black background and several separated monochrome objects.
    - Each object is one solid axis-aligned rectangle with one or two side features on different edges.
    - A tab is a one-cell-deep outward strip on the middle of a side, while a notch is a one-cell-deep inward gap on the middle of a side.

    Rule:
    - Recover each object's underlying plain rectangle by removing tabs and filling notches.
    - For a tab of length k, erase the tab itself and shave k whole layers from the opposite side of that rectangle.
    - For a notch of length k, fill the notch and grow the rectangle outward by k whole layers on the opposite side.

    Key insight:
    - The visible side feature is not the part to preserve: it is an instruction whose sign flips tab versus notch and whose direction flips to the opposite side, with the feature's sidewise length becoming the depth of the opposite-side edit.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = get_bg(g)
    out = np.full_like(g, bg)

    for component in find_components(g, bg):
        base_bbox, features = _recover_object(component)
        tr0, tc0, tr1, tc1 = _transform_bbox(base_bbox, features)
        assert 0 <= tr0 <= tr1 < g.shape[0]
        assert 0 <= tc0 <= tc1 < g.shape[1]
        color = int(component["color"])
        assert np.all(out[tr0 : tr1 + 1, tc0 : tc1 + 1] == bg)
        out[tr0 : tr1 + 1, tc0 : tc1 + 1] = color

    return out


def _feature_segment(side: str, start: int, length: int, h: int, w: int) -> list[tuple[int, int]]:
    if side == "top":
        return [(0, c) for c in range(start, start + length)]
    if side == "bottom":
        return [(h - 1, c) for c in range(start, start + length)]
    if side == "left":
        return [(r, 0) for r in range(start, start + length)]
    return [(r, w - 1) for r in range(start, start + length)]


def _build_input_coords(h: int, w: int, features: tuple[Feature, ...]) -> tuple[tuple[int, int], ...]:
    cells = {(r, c) for r in range(h) for c in range(w)}
    for feature in features:
        if feature.kind == "tab":
            if feature.side == "top":
                cells.update((-1, c) for c in range(feature.start, feature.start + feature.length))
            elif feature.side == "bottom":
                cells.update((h, c) for c in range(feature.start, feature.start + feature.length))
            elif feature.side == "left":
                cells.update((r, -1) for r in range(feature.start, feature.start + feature.length))
            else:
                cells.update((r, w) for r in range(feature.start, feature.start + feature.length))
        else:
            for cell in _feature_segment(feature.side, feature.start, feature.length, h, w):
                cells.remove(cell)
    return tuple(sorted(cells))


def _build_output_coords(h: int, w: int, features: tuple[Feature, ...]) -> tuple[tuple[int, int], ...]:
    top, left, bottom, right = 0, 0, h - 1, w - 1
    for feature in features:
        k = feature.length
        if feature.side == "top":
            bottom += k if feature.kind == "notch" else -k
        elif feature.side == "bottom":
            top -= k if feature.kind == "notch" else k
        elif feature.side == "left":
            right += k if feature.kind == "notch" else -k
        else:
            left -= k if feature.kind == "notch" else k
    if bottom - top + 1 < 2 or right - left + 1 < 2:
        raise ValueError("output rectangle became too small")
    return tuple((r, c) for r in range(top, bottom + 1) for c in range(left, right + 1))


def _make_spec(color: int, feature_kinds: list[str]) -> ObjectSpec | None:
    for _ in range(SPEC_RETRIES):
        h = random.randint(*BASE_HEIGHT_RANGE)
        w = random.randint(*BASE_WIDTH_RANGE)
        sides = random.sample(SIDES, len(feature_kinds))
        features: list[Feature] = []
        vertical_shave = 0
        horizontal_shave = 0

        for side, kind in zip(sides, feature_kinds):
            axis_len = w if side in HORIZONTAL_SIDES else h
            max_k = min(FEATURE_LENGTH_CAP, axis_len - 2)
            if kind == "tab" and side in HORIZONTAL_SIDES:
                max_k = min(max_k, h - 1)
            if kind == "tab" and side in VERTICAL_SIDES:
                max_k = min(max_k, w - 1)
            if max_k < 1:
                break
            length = random.randint(1, max_k)
            start = random.randint(1, axis_len - length - 1)
            features.append(Feature(side, kind, start, length))
            if kind == "tab" and side in HORIZONTAL_SIDES:
                vertical_shave += length
            if kind == "tab" and side in VERTICAL_SIDES:
                horizontal_shave += length
        else:
            if vertical_shave >= h or horizontal_shave >= w:
                continue
            feature_tuple = tuple(features)
            try:
                input_coords = _build_input_coords(h, w, feature_tuple)
                output_coords = _build_output_coords(h, w, feature_tuple)
            except ValueError:
                continue
            if not coords_connected(input_coords):
                continue
            input_bbox = relative_bbox(input_coords)
            output_bbox = relative_bbox(output_coords)
            return ObjectSpec(
                color=color,
                features=feature_tuple,
                input_coords=input_coords,
                output_coords=output_coords,
                input_bbox=input_bbox,
                output_bbox=output_bbox,
                overall_bbox=union_bbox(input_bbox, output_bbox),
            )
    return None


def _sample_feature_counts(n_objects: int) -> list[int]:
    counts = [2, 1]
    for _ in range(n_objects - 2):
        counts.append(random.choice([1, 2]))
    random.shuffle(counts)
    return counts


def _sample_specs(n_objects: int) -> list[ObjectSpec] | None:
    counts = _sample_feature_counts(n_objects)
    total_features = sum(counts)
    kinds = ["tab", "notch"] + [random.choice(["tab", "notch"]) for _ in range(total_features - 2)]
    random.shuffle(kinds)
    colors = random.sample(PALETTE, n_objects)

    specs: list[ObjectSpec] = []
    offset = 0
    for color, count in zip(colors, counts):
        spec = _make_spec(color, kinds[offset : offset + count])
        if spec is None:
            return None
        specs.append(spec)
        offset += count
    return specs


def _sample_grid_shape(n_objects: int) -> tuple[int, int]:
    min_dim = 12 + 3 * n_objects
    return random.randint(min_dim, GRID_SIZE_MAX), random.randint(min_dim, GRID_SIZE_MAX)


def _place_specs(shape: tuple[int, int], specs: list[ObjectSpec]) -> dict[int, tuple[int, int]] | None:
    anchors: dict[int, tuple[int, int]] = {}
    input_boxes: list[tuple[int, int, int, int]] = []
    output_boxes: list[tuple[int, int, int, int]] = []
    order = sorted(
        range(len(specs)),
        key=lambda idx: max(bbox_area(specs[idx].input_bbox), bbox_area(specs[idx].output_bbox)),
        reverse=True,
    )

    for idx in order:
        spec = specs[idx]
        rr0, rc0, rr1, rc1 = spec.overall_bbox
        row_lo = -rr0
        row_hi = shape[0] - 1 - rr1
        col_lo = -rc0
        col_hi = shape[1] - 1 - rc1
        if row_lo > row_hi or col_lo > col_hi:
            return None

        candidates = [
            (r, c)
            for r in range(row_lo, row_hi + 1)
            for c in range(col_lo, col_hi + 1)
        ]
        random.shuffle(candidates)

        chosen = None
        for anchor in candidates:
            input_box = translate_bbox(spec.input_bbox, anchor)
            output_box = translate_bbox(spec.output_bbox, anchor)
            if not all(boxes_separated(input_box, other, OBJECT_GAP) for other in input_boxes):
                continue
            if not all(boxes_separated(output_box, other, OBJECT_GAP) for other in output_boxes):
                continue
            chosen = anchor
            break
        if chosen is None:
            return None
        anchors[idx] = chosen
        input_boxes.append(translate_bbox(spec.input_bbox, chosen))
        output_boxes.append(translate_bbox(spec.output_bbox, chosen))

    return anchors


def _paint_coords(
    grid: np.ndarray, coords: tuple[tuple[int, int], ...], anchor: tuple[int, int], color: int
) -> None:
    r0, c0 = anchor
    for dr, dc in coords:
        grid[r0 + dr, c0 + dc] = color


def _gen_attempt() -> np.ndarray | None:
    n_objects = random.randint(2, 4)
    specs = _sample_specs(n_objects)
    if specs is None:
        return None

    shape = _sample_grid_shape(n_objects)
    anchors = _place_specs(shape, specs)
    if anchors is None:
        return None

    inp = np.full(shape, BACKGROUND, dtype=int)
    out = np.full(shape, BACKGROUND, dtype=int)
    for idx, spec in enumerate(specs):
        anchor = anchors[idx]
        _paint_coords(inp, spec.input_coords, anchor, spec.color)
        _paint_coords(out, spec.output_coords, anchor, spec.color)

    if np.mean(inp == BACKGROUND) < BACKGROUND_FRACTION_MIN:
        return None
    if np.mean(out == BACKGROUND) < BACKGROUND_FRACTION_MIN:
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
