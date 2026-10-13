from __future__ import annotations

from collections import Counter
import random

import numpy as np


_NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
_BG = 0
_GRID_HEIGHT_RANGE = (18, 30)
_GRID_WIDTH_RANGE = (18, 30)
_COLOR_COUNT_RANGE = (1, 4)
_SIZE_MIN = 3
_SIZE_CAP = 12
_PLACEMENT_PAD = 1
_OVERLAP_PROB = 0.5
_MAX_GENERATION_ATTEMPTS = 500


def _unique_mode(values: np.ndarray) -> int:
    counts = Counter(map(int, values.reshape(-1)))
    ranked = counts.most_common()
    assert ranked, "expected a nonempty grid"
    assert len(ranked) == 1 or ranked[0][1] > ranked[1][1], "background must be uniquely dominant"
    return int(ranked[0][0])


def _color_components(grid: np.ndarray, color: int) -> list[dict]:
    mask = np.asarray(grid == int(color), dtype=bool)
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[dict] = []

    for row in range(height):
        for col in range(width):
            if not bool(mask[row, col]) or bool(seen[row, col]):
                continue

            stack = [(int(row), int(col))]
            seen[row, col] = True
            coords: list[tuple[int, int]] = []
            while stack:
                r, c = stack.pop()
                coords.append((int(r), int(c)))
                for dr, dc in _NEIGHBORS4:
                    rr, cc = r + dr, c + dc
                    if 0 <= rr < height and 0 <= cc < width and bool(mask[rr, cc]) and not bool(seen[rr, cc]):
                        seen[rr, cc] = True
                        stack.append((int(rr), int(cc)))

            arr = np.array(coords, dtype=int)
            r0 = int(arr[:, 0].min())
            c0 = int(arr[:, 1].min())
            r1 = int(arr[:, 0].max())
            c1 = int(arr[:, 1].max())
            components.append(
                {
                    "coords": arr,
                    "bbox": (r0, c0, r1, c1),
                    "area": int(arr.shape[0]),
                    "color": int(color),
                }
            )

    return components


def _classify_component(component: dict) -> tuple[str, int]:
    r0, c0, r1, c1 = component["bbox"]
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    area = int(component["area"])

    if height == 2 and width == 2 and area == 4:
        return "anchor", 2
    if height == 1 and area == width and width >= 2:
        return "horizontal", int(width)
    if width == 1 and area == height and height >= 2:
        return "vertical", int(height)
    raise AssertionError(f"unexpected component shape for color {component['color']}: {component['bbox']}, area={area}")


def _extract_specs(grid: np.ndarray, bg: int) -> list[dict]:
    grid = np.asarray(grid, dtype=int)
    height, width = map(int, grid.shape)
    specs: list[dict] = []

    for color in sorted(int(c) for c in np.unique(grid) if int(c) != int(bg)):
        anchors: list[tuple[int, int]] = []
        horizontal_lengths: list[int] = []
        vertical_lengths: list[int] = []

        for component in _color_components(grid, int(color)):
            kind, length = _classify_component(component)
            if kind == "anchor":
                r0, c0, _r1, _c1 = component["bbox"]
                anchors.append((int(r0), int(c0)))
            elif kind == "horizontal":
                horizontal_lengths.append(int(length))
            else:
                vertical_lengths.append(int(length))

        assert len(anchors) == 1, f"expected one anchor for color {color}, got {anchors}"
        assert len(horizontal_lengths) == 2, f"expected two horizontal segments for color {color}"
        assert len(vertical_lengths) == 2, f"expected two vertical segments for color {color}"
        size = int(horizontal_lengths[0])
        assert size >= 2, "square side length must be at least 2"
        assert all(length == size for length in horizontal_lengths), f"horizontal lengths mismatch for color {color}"
        assert all(length == size for length in vertical_lengths), f"vertical lengths mismatch for color {color}"

        anchor_row, anchor_col = anchors[0]
        assert anchor_row + size <= height and anchor_col + size <= width, "restored frame must fit inside the grid"
        specs.append({"color": int(color), "size": int(size), "anchor": (int(anchor_row), int(anchor_col))})

    return specs


def _spec_sort_key(spec: dict) -> tuple[int, int, int]:
    anchor_row, anchor_col = spec["anchor"]
    return (-int(spec["size"]), int(anchor_row), int(anchor_col))


def _draw_frame(grid: np.ndarray, anchor_row: int, anchor_col: int, size: int, color: int) -> None:
    grid[int(anchor_row), int(anchor_col) : int(anchor_col + size)] = int(color)
    grid[int(anchor_row + size - 1), int(anchor_col) : int(anchor_col + size)] = int(color)
    grid[int(anchor_row) : int(anchor_row + size), int(anchor_col)] = int(color)
    grid[int(anchor_row) : int(anchor_row + size), int(anchor_col + size - 1)] = int(color)


def _render_output(shape: tuple[int, int], specs: list[dict], bg: int) -> np.ndarray:
    out = np.full(shape, int(bg), dtype=int)
    for spec in sorted(specs, key=_spec_sort_key):
        anchor_row, anchor_col = spec["anchor"]
        _draw_frame(out, int(anchor_row), int(anchor_col), int(spec["size"]), int(spec["color"]))
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble same-color scattered line segments into square frames anchored by same-color 2x2 blocks.

    Input structure:
    - Black is the background.
    - Each non-background color has exactly one solid 2x2 anchor block.
    - The same color also has four detached 1-pixel-thick line segments:
      two horizontal and two vertical, all with the same length S.

    Rule:
    - Interpret the four segments as the four edges of a hollow SxS square.
    - Place that square so its top-left corner matches the anchor block's top-left pixel.
    - Remove the original scattered pieces and anchors.
    - Draw larger squares first, then smaller ones so later smaller frames overwrite overlaps.
    """
    grid = np.asarray(grid, dtype=int)
    assert grid.ndim == 2, "expected a 2D grid"

    bg = _unique_mode(grid)
    specs = _extract_specs(grid, bg)
    return _render_output(grid.shape, specs, bg)


def _frame_bbox(spec: dict) -> tuple[int, int, int, int]:
    row, col = spec["anchor"]
    size = int(spec["size"])
    return int(row), int(col), int(row + size - 1), int(col + size - 1)


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 0) -> bool:
    return (
        int(a[2]) + int(pad) < int(b[0])
        or int(b[2]) + int(pad) < int(a[0])
        or int(a[3]) + int(pad) < int(b[1])
        or int(b[3]) + int(pad) < int(a[1])
    )


def _sample_sizes(count: int, max_size: int, want_overlap: bool) -> list[int] | None:
    if max_size < _SIZE_MIN or max_size - _SIZE_MIN + 1 < count:
        return None

    if want_overlap:
        if count < 2 or max_size < _SIZE_MIN + 3:
            return None
        big = random.randint(_SIZE_MIN + 3, max_size)
        small = random.randint(_SIZE_MIN, big - 3)
        remaining_pool = [size for size in range(_SIZE_MIN, max_size + 1) if size not in {big, small}]
        if len(remaining_pool) < count - 2:
            return None
        sizes = [big, small, *random.sample(remaining_pool, count - 2)]
        return sorted(sizes, reverse=True)

    return sorted(random.sample(range(_SIZE_MIN, max_size + 1), count), reverse=True)


def _sample_nonoverlap_anchor(size: int, height: int, width: int, existing: list[dict]) -> tuple[int, int] | None:
    positions = [(row, col) for row in range(height - size + 1) for col in range(width - size + 1)]
    random.shuffle(positions)
    existing_boxes = [_frame_bbox(spec) for spec in existing]

    for row, col in positions:
        bbox = (int(row), int(col), int(row + size - 1), int(col + size - 1))
        if all(_boxes_separated(bbox, other, pad=1) for other in existing_boxes):
            return int(row), int(col)
    return None


def _overlap_pair_specs(big: int, small: int, height: int, width: int) -> list[dict] | None:
    sides = ["top", "bottom", "left", "right"]
    for _ in range(200):
        side = random.choice(sides)
        big_row = random.randint(0, height - big)
        big_col = random.randint(0, width - big)

        if side in {"top", "bottom"}:
            lo = big_col + 3
            hi = big_col + big - small
            if lo > hi:
                continue
            small_col = random.randint(lo, hi)
            small_row = big_row if side == "top" else big_row + big - small
        else:
            lo = big_row + 3
            hi = big_row + big - small
            if lo > hi:
                continue
            small_row = random.randint(lo, hi)
            small_col = big_col if side == "left" else big_col + big - small

        return [
            {"size": int(big), "anchor": (int(big_row), int(big_col))},
            {"size": int(small), "anchor": (int(small_row), int(small_col))},
        ]
    return None


def _place_frame_specs(height: int, width: int, colors: list[int], sizes: list[int], want_overlap: bool) -> list[dict] | None:
    specs: list[dict] = []
    color_iter = iter(colors)

    if want_overlap:
        pair = _overlap_pair_specs(int(sizes[0]), int(sizes[1]), int(height), int(width))
        if pair is None:
            return None
        for color, partial in zip([next(color_iter), next(color_iter)], pair):
            specs.append(
                {
                    "color": int(color),
                    "size": int(partial["size"]),
                    "anchor": tuple(map(int, partial["anchor"])),
                }
            )
        start_index = 2
    else:
        start_index = 0

    for idx in range(start_index, len(sizes)):
        anchor = _sample_nonoverlap_anchor(int(sizes[idx]), int(height), int(width), specs)
        if anchor is None:
            return None
        specs.append({"color": int(next(color_iter)), "size": int(sizes[idx]), "anchor": anchor})

    return specs


def _can_place(occupied: np.ndarray, row: int, col: int, patch_h: int, patch_w: int, pad: int = _PLACEMENT_PAD) -> bool:
    height, width = occupied.shape
    if row < 0 or col < 0 or row + patch_h > height or col + patch_w > width:
        return False
    r0 = max(0, int(row) - int(pad))
    c0 = max(0, int(col) - int(pad))
    r1 = min(height, int(row + patch_h + pad))
    c1 = min(width, int(col + patch_w + pad))
    return not bool(occupied[r0:r1, c0:c1].any())


def _mark_rect(occupied: np.ndarray, grid: np.ndarray, row: int, col: int, patch_h: int, patch_w: int, color: int) -> None:
    grid[int(row) : int(row + patch_h), int(col) : int(col + patch_w)] = int(color)
    occupied[int(row) : int(row + patch_h), int(col) : int(col + patch_w)] = True


def _random_slot(occupied: np.ndarray, patch_h: int, patch_w: int) -> tuple[int, int] | None:
    height, width = occupied.shape
    positions = [(row, col) for row in range(height - patch_h + 1) for col in range(width - patch_w + 1)]
    random.shuffle(positions)
    for row, col in positions:
        if _can_place(occupied, int(row), int(col), int(patch_h), int(patch_w)):
            return int(row), int(col)
    return None


def _derive_input(height: int, width: int, specs: list[dict], bg: int) -> np.ndarray | None:
    inp = np.full((height, width), int(bg), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    for spec in sorted(specs, key=lambda item: (item["anchor"][0], item["anchor"][1], item["size"])):
        anchor_row, anchor_col = spec["anchor"]
        if not _can_place(occupied, int(anchor_row), int(anchor_col), 2, 2):
            return None
        _mark_rect(occupied, inp, int(anchor_row), int(anchor_col), 2, 2, int(spec["color"]))

    segments: list[tuple[int, int, int]] = []
    for spec in specs:
        size = int(spec["size"])
        color = int(spec["color"])
        segments.extend(
            [
                (color, 1, size),
                (color, 1, size),
                (color, size, 1),
                (color, size, 1),
            ]
        )

    random.shuffle(segments)
    segments.sort(key=lambda item: max(int(item[1]), int(item[2])), reverse=True)

    for color, patch_h, patch_w in segments:
        slot = _random_slot(occupied, int(patch_h), int(patch_w))
        if slot is None:
            return None
        row, col = slot
        _mark_rect(occupied, inp, int(row), int(col), int(patch_h), int(patch_w), int(color))

    return inp


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*_GRID_HEIGHT_RANGE)
    width = random.randint(*_GRID_WIDTH_RANGE)
    want_overlap = bool(random.random() < _OVERLAP_PROB)

    min_colors = 2 if want_overlap else 1
    color_count = random.randint(min_colors, _COLOR_COUNT_RANGE[1])
    if color_count > len(range(1, 10)):
        return None

    max_size = min(_SIZE_CAP, height - 4, width - 4)
    sizes = _sample_sizes(int(color_count), int(max_size), want_overlap)
    if sizes is None:
        return None

    colors = random.sample(list(range(1, 10)), int(color_count))
    specs = _place_frame_specs(int(height), int(width), colors, sizes, want_overlap)
    if specs is None:
        return None

    inp = _derive_input(int(height), int(width), specs, _BG)
    if inp is None:
        return None

    expected = _render_output(inp.shape, specs, _BG)
    try:
        if np.array_equal(main(inp), expected):
            return inp
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(_MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 7")
