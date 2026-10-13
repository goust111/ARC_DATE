import numpy as np


_RNG = np.random.default_rng()
_DIRS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))

# Generator configuration.
_BG = 0
_COLOR_POOL = tuple(range(1, 10))
_GEN_RETRIES = 300
_CONTAINER_COUNT_RANGE = (3, 5)
_CANVAS_SIDE_RANGE = (20, 30)
_LARGEST_INNER_RANGE = (8, 12)
_SMALLER_INNER_MIN = 5
_PATTERN_MARGIN = 1
_PATTERN_RETRIES = 80
_PLACEMENT_RETRIES = 200


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _is_exact_frame(mask: np.ndarray) -> bool:
    height, width = mask.shape
    if height < 3 or width < 3:
        return False
    return bool(np.array_equal(mask, _border_mask(height, width)))


def _frame_candidate(grid: np.ndarray, color: int) -> dict | None:
    mask = grid == int(color)
    box = _bbox(mask)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    height = int(r1 - r0 + 1)
    width = int(c1 - c0 + 1)
    if height < 3 or width < 3:
        return None

    local = mask[r0 : r1 + 1, c0 : c1 + 1]
    if not np.array_equal(local, _border_mask(height, width)):
        return None

    return {
        "color": int(color),
        "r0": int(r0),
        "c0": int(c0),
        "r1": int(r1),
        "c1": int(c1),
        "height": int(height),
        "width": int(width),
        "inner_area": int((height - 2) * (width - 2)),
    }


def _strictly_contains(outer: dict, inner: dict) -> bool:
    return (
        int(outer["r0"]) < int(inner["r0"])
        and int(outer["c0"]) < int(inner["c0"])
        and int(outer["r1"]) > int(inner["r1"])
        and int(outer["c1"]) > int(inner["c1"])
    )


def _find_containers(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = _most_common_color(grid)
    candidates: list[dict] = []
    for color in sorted(map(int, np.unique(grid))):
        if int(color) == int(bg):
            continue
        candidate = _frame_candidate(grid, int(color))
        if candidate is not None:
            candidates.append(candidate)

    containers = [
        cand
        for cand in candidates
        if not any(_strictly_contains(other, cand) for other in candidates if other is not cand)
    ]
    containers.sort(
        key=lambda item: (
            int(item["inner_area"]),
            int(item["height"]),
            int(item["width"]),
            int(item["r0"]),
            int(item["c0"]),
        )
    )
    return int(bg), containers


def _extract_pattern(interior: np.ndarray, bg: int) -> np.ndarray:
    box = _bbox(interior != int(bg))
    assert box is not None, "each container must contain a pattern"
    r0, c0, r1, c1 = box
    return np.array(interior[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _paste_center(dst: np.ndarray, patch: np.ndarray, bg: int) -> None:
    ph, pw = map(int, patch.shape)
    dh, dw = map(int, dst.shape)
    assert ph <= dh and pw <= dw, "pattern must fit inside the largest container interior"
    top = (dh - ph) // 2
    left = (dw - pw) // 2
    window = dst[top : top + ph, left : left + pw]
    mask = patch != int(bg)
    window[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extract the patterns from colored rectangular containers and center-stack them inside the largest one.

    Input structure:
    - The background is the most common color.
    - Several disjoint 1-pixel rectangular frames appear, each frame color is unique and marks one container.
    - Every container has a non-background pixel pattern inside it, separated from the border by background cells.

    Rule:
    - Detect all top-level rectangular frame containers and sort them by interior area.
    - Output only the largest container, keeping its original border.
    - Crop each smaller container's interior pattern to its non-background bounding box and paste it centered into the largest container interior from smallest to largest.
    - After all smaller patterns are pasted, restore the original pattern of the largest container on top; smaller borders never appear in the output.

    Key insight:
    - The meaningful objects are the top-level colored frames, while the actual transferred content is only each frame's interior motif cropped to its tight bounding box and layered by container size.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, containers = _find_containers(g)
    assert containers, "expected at least one rectangular container"

    largest = containers[-1]
    out = np.full((int(largest["height"]), int(largest["width"])), int(bg), dtype=int)
    border_color = int(largest["color"])
    out[0, :] = border_color
    out[-1, :] = border_color
    out[:, 0] = border_color
    out[:, -1] = border_color

    inner = out[1:-1, 1:-1]
    for container in containers[:-1]:
        interior = g[
            int(container["r0"]) + 1 : int(container["r1"]),
            int(container["c0"]) + 1 : int(container["c1"]),
        ]
        patch = _extract_pattern(interior, bg)
        _paste_center(inner, patch, bg)

    largest_interior = g[
        int(largest["r0"]) + 1 : int(largest["r1"]),
        int(largest["c0"]) + 1 : int(largest["c1"]),
    ]
    keep = largest_interior != int(bg)
    inner[keep] = largest_interior[keep]
    return out


def _crop_nonzero(patch: np.ndarray) -> np.ndarray | None:
    box = _bbox(patch != 0)
    if box is None:
        return None
    r0, c0, r1, c1 = box
    return np.array(patch[r0 : r1 + 1, c0 : c1 + 1], dtype=int)


def _grow_connected_mask(height: int, width: int, target: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    start = (_randint(0, height - 1), _randint(0, width - 1))
    mask[start] = True

    while int(mask.sum()) < int(target):
        frontier: list[tuple[int, int]] = []
        for r, c in np.argwhere(mask):
            for dr, dc in _DIRS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if 0 <= nr < height and 0 <= nc < width and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        nr, nc = frontier[_randint(0, len(frontier) - 1)]
        mask[nr, nc] = True
    return mask


def _make_pattern_patch(max_h: int, max_w: int, color: int) -> np.ndarray:
    for _ in range(_PATTERN_RETRIES):
        mode = _randint(0, 4)

        if mode == 0:
            ph = _randint(1, max_h)
            pw = _randint(1, max_w)
            patch = np.full((ph, pw), int(color), dtype=int)
        elif mode == 1:
            if max_h == 1 and max_w == 1:
                patch = np.full((1, 1), int(color), dtype=int)
            elif _randint(0, 1) == 0:
                patch = np.full((1, _randint(1, max_w)), int(color), dtype=int)
            else:
                patch = np.full((_randint(1, max_h), 1), int(color), dtype=int)
        elif mode == 2:
            if max_h < 2 or max_w < 2:
                continue
            ph = _randint(2, max_h)
            pw = _randint(2, max_w)
            patch = np.zeros((ph, pw), dtype=int)
            patch[0, :] = int(color)
            patch[:, 0] = int(color)
            if _randint(0, 1) == 1:
                patch = np.fliplr(patch)
            if _randint(0, 1) == 1:
                patch = np.flipud(patch)
        elif mode == 3:
            if max_h < 3 or max_w < 3:
                continue
            ph = _randint(3, max_h)
            pw = _randint(3, max_w)
            patch = np.zeros((ph, pw), dtype=int)
            patch[ph // 2, :] = int(color)
            patch[:, pw // 2] = int(color)
        else:
            ph = _randint(2 if max_h >= 2 else 1, max_h)
            pw = _randint(2 if max_w >= 2 else 1, max_w)
            patch = np.zeros((ph, pw), dtype=int)
            area = int(ph * pw)
            target = _randint(max(1, area // 3), area)
            patch[_grow_connected_mask(ph, pw, target)] = int(color)

        patch = _crop_nonzero(patch)
        if patch is None:
            continue
        if _is_exact_frame(patch != 0):
            continue
        return patch

    return np.full((1, 1), int(color), dtype=int)


def _sample_inner_sizes(count: int) -> list[tuple[int, int]] | None:
    largest_h = _randint(*_LARGEST_INNER_RANGE)
    largest_w = _randint(*_LARGEST_INNER_RANGE)
    largest_area = int(largest_h * largest_w)

    candidates = [
        (h, w)
        for h in range(_SMALLER_INNER_MIN, largest_h + 1)
        for w in range(_SMALLER_INNER_MIN, largest_w + 1)
        if h * w < largest_area
    ]
    if len(candidates) < count - 1:
        return None

    order = _RNG.permutation(len(candidates))
    used_areas = {largest_area}
    sizes = [(largest_h, largest_w)]
    for idx in order:
        h, w = candidates[int(idx)]
        area = int(h * w)
        if area in used_areas:
            continue
        sizes.append((int(h), int(w)))
        used_areas.add(area)
        if len(sizes) == count:
            return sizes
    return None


def _boxes_touch(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (
        ar1 + 1 < br0
        or br1 + 1 < ar0
        or ac1 + 1 < bc0
        or bc1 + 1 < ac0
    )


def _background_dominates(grid: np.ndarray) -> bool:
    bg_count = int(np.count_nonzero(grid == _BG))
    for color in _COLOR_POOL:
        if int(np.count_nonzero(grid == int(color))) >= bg_count:
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    count = _randint(*_CONTAINER_COUNT_RANGE)
    inner_sizes = _sample_inner_sizes(count)
    if inner_sizes is None:
        return None

    containers = []
    border_colors = list(map(int, _RNG.choice(_COLOR_POOL, size=count, replace=False)))
    pattern_palette = [color for color in _COLOR_POOL if color not in border_colors]
    for idx, (inner_h, inner_w) in enumerate(inner_sizes):
        max_patch_h = int(inner_h - 2 * _PATTERN_MARGIN)
        max_patch_w = int(inner_w - 2 * _PATTERN_MARGIN)
        if max_patch_h < 1 or max_patch_w < 1:
            return None
        pattern_color = int(pattern_palette[_randint(0, len(pattern_palette) - 1)])
        patch = _make_pattern_patch(max_patch_h, max_patch_w, pattern_color)
        containers.append(
            {
                "inner_h": int(inner_h),
                "inner_w": int(inner_w),
                "outer_h": int(inner_h + 2),
                "outer_w": int(inner_w + 2),
                "border_color": int(border_colors[idx]),
                "pattern_color": int(pattern_color),
                "patch": patch,
                "inner_area": int(inner_h * inner_w),
            }
        )

    max_outer_h = max(int(item["outer_h"]) for item in containers)
    max_outer_w = max(int(item["outer_w"]) for item in containers)
    min_h = max(int(max_outer_h + 2), int(_CANVAS_SIDE_RANGE[0]))
    min_w = max(int(max_outer_w + 2), int(_CANVAS_SIDE_RANGE[0]))
    if min_h > _CANVAS_SIDE_RANGE[1] or min_w > _CANVAS_SIDE_RANGE[1]:
        return None

    canvas_h = _randint(min_h, _CANVAS_SIDE_RANGE[1])
    canvas_w = _randint(min_w, _CANVAS_SIDE_RANGE[1])
    total_outer_area = sum(int(item["outer_h"]) * int(item["outer_w"]) for item in containers)
    if total_outer_area > int(0.42 * canvas_h * canvas_w):
        return None

    grid = np.full((canvas_h, canvas_w), _BG, dtype=int)
    placed_boxes: list[tuple[int, int, int, int]] = []
    placement_order = sorted(
        range(count),
        key=lambda idx: int(containers[idx]["outer_h"]) * int(containers[idx]["outer_w"]),
        reverse=True,
    )

    for idx in placement_order:
        item = containers[idx]
        outer_h = int(item["outer_h"])
        outer_w = int(item["outer_w"])
        for _ in range(_PLACEMENT_RETRIES):
            top = _randint(0, canvas_h - outer_h)
            left = _randint(0, canvas_w - outer_w)
            box = (int(top), int(left), int(top + outer_h - 1), int(left + outer_w - 1))
            if any(_boxes_touch(box, other) for other in placed_boxes):
                continue

            border_color = int(item["border_color"])
            grid[top : top + outer_h, left] = border_color
            grid[top : top + outer_h, left + outer_w - 1] = border_color
            grid[top, left : left + outer_w] = border_color
            grid[top + outer_h - 1, left : left + outer_w] = border_color

            patch = np.asarray(item["patch"], dtype=int)
            ph, pw = map(int, patch.shape)
            inner_h = int(item["inner_h"])
            inner_w = int(item["inner_w"])
            patch_top = _randint(_PATTERN_MARGIN, inner_h - ph - _PATTERN_MARGIN)
            patch_left = _randint(_PATTERN_MARGIN, inner_w - pw - _PATTERN_MARGIN)
            inner = grid[top + 1 : top + 1 + inner_h, left + 1 : left + 1 + inner_w]
            mask = patch != 0
            inner[patch_top : patch_top + ph, patch_left : patch_left + pw][mask] = patch[mask]

            item["top"] = int(top)
            item["left"] = int(left)
            placed_boxes.append(box)
            break
        else:
            return None

    if not _background_dominates(grid):
        return None

    bg, found = _find_containers(grid)
    if int(bg) != _BG or len(found) != count:
        return None
    found_areas = [int(item["inner_area"]) for item in found]
    if len(set(found_areas)) != count:
        return None

    out = main(grid)
    if out.ndim != 2 or max(out.shape) > 30:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid container-stacking input")
