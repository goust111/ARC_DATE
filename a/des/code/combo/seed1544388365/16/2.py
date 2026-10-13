from __future__ import annotations

import numpy as np


BG = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
RECT_SIDE_MIN = 3
RECT_SIDE_MAX = 7
RECT_COUNT_MIN = 2
RECT_COUNT_MAX = 4
INPUT_SIDE_MIN = 18
INPUT_SIDE_MAX = 30
OUTPUT_WIDTH_MAX = 30
PLACEMENT_TRIES = 500
SIZE_TRIES = 40
MAX_GENERATION_ATTEMPTS = 400

_RNG = np.random.default_rng()


def _neighbors4(r: int, c: int, height: int, width: int):
    for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < height and 0 <= nc < width:
            yield nr, nc


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert coords.size > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _component_masks_for_color(grid: np.ndarray, color: int) -> list[np.ndarray]:
    comps = _find_components(grid == color)
    assert len(comps) == 2, f"expected exactly two fragments for color {color}"

    masks: list[np.ndarray] = []
    for coords in comps:
        mask = np.zeros_like(grid, dtype=bool)
        mask[coords[:, 0], coords[:, 1]] = True
        masks.append(_crop_mask(mask))
    return masks


def _scan_join_candidates(mask_a: np.ndarray, mask_b: np.ndarray, axis: str) -> set[tuple[int, int]]:
    area = int(mask_a.sum() + mask_b.sum())
    coords_a = set(map(tuple, np.argwhere(mask_a)))
    coords_b = np.argwhere(mask_b)
    found: set[tuple[int, int]] = set()

    if axis == "vertical":
        if mask_a.shape[0] != mask_b.shape[0]:
            return found
        for dx in range(-mask_b.shape[1], mask_a.shape[1] + 1):
            shifted_b = {(int(r), int(c) + dx) for r, c in coords_b}
            if coords_a & shifted_b:
                continue
            union = coords_a | shifted_b
            rows = [r for r, _ in union]
            cols = [c for _, c in union]
            height = max(rows) - min(rows) + 1
            width = max(cols) - min(cols) + 1
            if height * width == area:
                found.add((height, width))
        return found

    if mask_a.shape[1] != mask_b.shape[1]:
        return found
    for dy in range(-mask_b.shape[0], mask_a.shape[0] + 1):
        shifted_b = {(int(r) + dy, int(c)) for r, c in coords_b}
        if coords_a & shifted_b:
            continue
        union = coords_a | shifted_b
        rows = [r for r, _ in union]
        cols = [c for _, c in union]
        height = max(rows) - min(rows) + 1
        width = max(cols) - min(cols) + 1
        if height * width == area:
            found.add((height, width))
    return found


def _candidate_rectangles(mask_a: np.ndarray, mask_b: np.ndarray) -> set[tuple[int, int]]:
    candidates: set[tuple[int, int]] = set()

    for flip_mode in range(3):
        ma = mask_a.copy()
        mb = mask_b.copy()
        if flip_mode == 1:
            ma = np.fliplr(ma)
        elif flip_mode == 2:
            mb = np.fliplr(mb)
        candidates |= _scan_join_candidates(ma, mb, axis="vertical")

    for flip_mode in range(3):
        ma = mask_a.copy()
        mb = mask_b.copy()
        if flip_mode == 1:
            ma = np.flipud(ma)
        elif flip_mode == 2:
            mb = np.flipud(mb)
        candidates |= _scan_join_candidates(ma, mb, axis="horizontal")

    return candidates


def _reconstruct_rectangle(mask_a: np.ndarray, mask_b: np.ndarray) -> tuple[int, int]:
    candidates = _candidate_rectangles(mask_a, mask_b)
    assert len(candidates) == 1, f"expected one rectangle, got {sorted(candidates)}"
    return next(iter(candidates))


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble each color's two fragments into a solid rectangle, then sort by area.

    Input structure
    - Background is black.
    - Each non-black color appears in exactly two disconnected fragments.
    - The two fragments of one color can be joined into one solid rectangle by
      translation plus an optional flip along the join axis; rotation is not allowed.

    Rule
    - Reconstruct the unique rectangle for each color.
    - Sort the rectangles by area from small to large.
    - Place them left to right with one black separator column and bottom alignment.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rectangles = []
    for color in map(int, np.unique(g)):
        if color == BG:
            continue
        mask_a, mask_b = _component_masks_for_color(g, color)
        height, width = _reconstruct_rectangle(mask_a, mask_b)
        rectangles.append(
            {
                "color": color,
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )

    assert rectangles, "expected at least one color pair"
    rectangles.sort(key=lambda item: (item["area"], item["height"], item["width"]))

    out_height = max(item["height"] for item in rectangles)
    out_width = sum(item["width"] for item in rectangles) + len(rectangles) - 1
    out = np.full((out_height, out_width), BG, dtype=int)

    col = 0
    for item in rectangles:
        h = item["height"]
        w = item["width"]
        out[out_height - h :, col : col + w] = item["color"]
        col += w + 1

    return out


def _random_profile(length: int, limit: int) -> list[int]:
    vals = [int(_RNG.integers(1, limit))]
    for _ in range(length - 1):
        step = int(_RNG.integers(-1, 2))
        vals.append(int(np.clip(vals[-1] + step, 1, limit - 1)))

    # Avoid fully straight cuts, which are more likely to create ambiguous joins.
    if len(set(vals)) == 1 and length > 1 and limit > 2:
        idx = int(_RNG.integers(0, length))
        if vals[idx] == 1:
            vals[idx] = 2
        elif vals[idx] == limit - 1:
            vals[idx] = limit - 2
        else:
            vals[idx] += 1 if bool(_RNG.integers(0, 2)) else -1
    return vals


def _split_rectangle(height: int, width: int) -> tuple[np.ndarray, np.ndarray]:
    if bool(_RNG.integers(0, 2)):
        cut = _random_profile(height, width)
        first = np.zeros((height, width), dtype=bool)
        second = np.zeros((height, width), dtype=bool)
        for r, boundary in enumerate(cut):
            first[r, :boundary] = True
            second[r, boundary:] = True
        flip_mode = int(_RNG.integers(0, 3))
        if flip_mode == 1:
            first = np.fliplr(first)
        elif flip_mode == 2:
            second = np.fliplr(second)
        return first, second

    cut = _random_profile(width, height)
    first = np.zeros((height, width), dtype=bool)
    second = np.zeros((height, width), dtype=bool)
    for c, boundary in enumerate(cut):
        first[:boundary, c] = True
        second[boundary:, c] = True
    flip_mode = int(_RNG.integers(0, 3))
    if flip_mode == 1:
        first = np.flipud(first)
    elif flip_mode == 2:
        second = np.flipud(second)
    return first, second


def _dilate8(mask: np.ndarray) -> np.ndarray:
    dilated = mask.copy()
    dilated[1:, :] |= mask[:-1, :]
    dilated[:-1, :] |= mask[1:, :]
    dilated[:, 1:] |= mask[:, :-1]
    dilated[:, :-1] |= mask[:, 1:]
    dilated[1:, 1:] |= mask[:-1, :-1]
    dilated[1:, :-1] |= mask[:-1, 1:]
    dilated[:-1, 1:] |= mask[1:, :-1]
    dilated[:-1, :-1] |= mask[1:, 1:]
    return dilated


def _place_patches(patches: list[tuple[np.ndarray, int]], height: int, width: int) -> np.ndarray | None:
    order = sorted(range(len(patches)), key=lambda idx: int(patches[idx][0].sum()), reverse=True)
    grid = np.full((height, width), BG, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)
    dilated = np.zeros((height, width), dtype=bool)

    for idx in order:
        mask, color = patches[idx]
        h, w = mask.shape
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = int(_RNG.integers(0, height - h + 1))
            c0 = int(_RNG.integers(0, width - w + 1))
            if bool((dilated[r0 : r0 + h, c0 : c0 + w] & mask).any()):
                continue
            window = grid[r0 : r0 + h, c0 : c0 + w]
            window[mask] = color
            occupied[r0 : r0 + h, c0 : c0 + w] |= mask
            dilated = _dilate8(occupied)
            placed = True
            break
        if not placed:
            return None

    return grid


def _render_expected_output(rectangles: list[dict[str, int]]) -> np.ndarray:
    ordered = sorted(rectangles, key=lambda item: (item["area"], item["height"], item["width"]))
    out_height = max(item["height"] for item in ordered)
    out_width = sum(item["width"] for item in ordered) + len(ordered) - 1
    out = np.full((out_height, out_width), BG, dtype=int)

    col = 0
    for item in ordered:
        h = item["height"]
        w = item["width"]
        out[out_height - h :, col : col + w] = item["color"]
        col += w + 1

    return out


def _generate_attempt() -> np.ndarray | None:
    rect_count = int(_RNG.integers(RECT_COUNT_MIN, RECT_COUNT_MAX + 1))
    colors = _RNG.choice(COLOR_POOL, size=rect_count, replace=False)

    rectangles: list[dict[str, int]] = []
    patches: list[tuple[np.ndarray, int]] = []
    used_areas: set[int] = set()

    for color in map(int, colors):
        accepted = False
        for _ in range(200):
            height = int(_RNG.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
            width = int(_RNG.integers(RECT_SIDE_MIN, RECT_SIDE_MAX + 1))
            area = int(height * width)
            if area in used_areas:
                continue

            first, second = _split_rectangle(height, width)
            if len(_candidate_rectangles(first, second)) != 1:
                continue

            rectangles.append(
                {
                    "color": color,
                    "height": height,
                    "width": width,
                    "area": area,
                }
            )
            patches.append((first, color))
            patches.append((second, color))
            used_areas.add(area)
            accepted = True
            break
        if not accepted:
            return None

    if sum(item["width"] for item in rectangles) + len(rectangles) - 1 > OUTPUT_WIDTH_MAX:
        return None

    total_fg = sum(int(mask.sum()) for mask, _ in patches)
    max_patch_h = max(mask.shape[0] for mask, _ in patches)
    max_patch_w = max(mask.shape[1] for mask, _ in patches)
    expected = _render_expected_output(rectangles)

    min_h = max(INPUT_SIDE_MIN, max_patch_h + 2)
    min_w = max(INPUT_SIDE_MIN, max_patch_w + 2)
    if min_h > INPUT_SIDE_MAX or min_w > INPUT_SIDE_MAX:
        return None

    for _ in range(SIZE_TRIES):
        height = int(_RNG.integers(min_h, INPUT_SIDE_MAX + 1))
        width = int(_RNG.integers(min_w, INPUT_SIDE_MAX + 1))
        if height * width < total_fg * 3:
            continue
        grid = _place_patches(patches, height, width)
        if grid is None:
            continue
        try:
            actual = main(grid)
        except AssertionError:
            continue
        if np.array_equal(actual, expected):
            return grid

    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
