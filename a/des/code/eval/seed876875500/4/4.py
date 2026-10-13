import numpy as np


RNG = np.random.default_rng()

BG = 0
ANCHOR_COLOR = 5
COLOR_POOL = np.array([1, 2, 3, 4, 6, 7, 8, 9], dtype=int)

RECT_COUNT_RANGE = (2, 4)
LARGEST_HEIGHT_RANGE = (4, 7)
LARGEST_WIDTH_RANGE = (5, 8)
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 30)
PARTITION_ATTEMPTS = 120
RECTANGLE_ATTEMPTS = 120
SCATTER_ATTEMPTS = 250
GEN_ATTEMPTS = 400


def _randint(low: int, high: int) -> int:
    return int(RNG.integers(low, high + 1))


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    result: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            result.append(cells)
    return result


def _normalize_fragment(cells: list[tuple[int, int]]) -> dict[str, object]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0, c0 = min(rows), min(cols)
    local = tuple(sorted((r - r0, c - c0) for r, c in cells))
    height = max(rows) - r0 + 1
    width = max(cols) - c0 + 1
    return {
        "cells": local,
        "height": int(height),
        "width": int(width),
        "size": int(len(cells)),
    }


def _factor_pairs(area: int) -> list[tuple[int, int]]:
    pairs: list[tuple[int, int]] = []
    for h in range(1, int(area ** 0.5) + 1):
        if area % h != 0:
            continue
        w = area // h
        pairs.append((h, w))
        if h != w:
            pairs.append((w, h))
    pairs.sort()
    return pairs


def _placement_masks(fragment: dict[str, object], height: int, width: int) -> list[int]:
    frag_h = int(fragment["height"])
    frag_w = int(fragment["width"])
    if frag_h > height or frag_w > width:
        return []
    cells = fragment["cells"]
    placements: list[int] = []
    for r0 in range(height - frag_h + 1):
        for c0 in range(width - frag_w + 1):
            mask = 0
            for r, c in cells:
                bit = (r0 + r) * width + (c0 + c)
                mask |= 1 << bit
            placements.append(mask)
    return placements


def _count_tilings_for_dims(
    fragments: list[dict[str, object]],
    height: int,
    width: int,
    limit: int = 2,
) -> int:
    placements = [_placement_masks(fragment, height, width) for fragment in fragments]
    if any(not options for options in placements):
        return 0

    full_mask = (1 << (height * width)) - 1
    total = 0

    def search(remaining: tuple[int, ...], occupied: int) -> None:
        nonlocal total
        if total >= limit:
            return
        if not remaining:
            total += 1
            return

        target_mask = (~occupied) & full_mask
        target_mask &= -target_mask
        best_index = -1
        best_options: list[int] | None = None

        for index in remaining:
            options = [
                mask
                for mask in placements[index]
                if (mask & target_mask) and not (occupied & mask)
            ]
            if not options:
                continue
            if best_options is None or len(options) < len(best_options):
                best_index = index
                best_options = options
                if len(best_options) == 1:
                    break

        if best_options is None:
            return

        next_remaining = tuple(index for index in remaining if index != best_index)
        for mask in best_options:
            search(next_remaining, occupied | mask)
            if total >= limit:
                return

    search(tuple(range(len(fragments))), 0)
    return total


def _rectangle_shapes(fragments: list[dict[str, object]], limit: int = 2) -> list[tuple[int, int]]:
    area = sum(int(fragment["size"]) for fragment in fragments)
    solutions: list[tuple[int, int]] = []
    for height, width in _factor_pairs(area):
        if any(int(fragment["height"]) > height or int(fragment["width"]) > width for fragment in fragments):
            continue
        if _count_tilings_for_dims(fragments, height, width, limit=1):
            solutions.append((height, width))
            if len(solutions) >= limit:
                break
    return solutions


def _find_anchor_color(grid: np.ndarray, bg: int) -> int:
    height, width = grid.shape
    corners = [(0, 0), (0, width - 1), (height - 1, 0), (height - 1, width - 1)]
    values = [int(grid[r, c]) for r, c in corners if int(grid[r, c]) != bg]
    assert len(values) == 1, "expected exactly one non-background corner anchor"
    anchor = values[0]
    assert int(np.count_nonzero(grid == anchor)) == 1, "anchor color must appear exactly once"
    return anchor


def _stack_rectangles(rectangles: list[dict[str, int]], bg: int) -> np.ndarray:
    out_h = max(int(rectangle["height"]) for rectangle in rectangles)
    out_w = max(int(rectangle["width"]) for rectangle in rectangles)
    out = np.full((out_h, out_w), bg, dtype=int)
    for rectangle in sorted(rectangles, key=lambda item: item["area"], reverse=True):
        h = int(rectangle["height"])
        w = int(rectangle["width"])
        out[out_h - h :, :w] = int(rectangle["color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group same-colored fragments, reconstruct one solid rectangle per color by translation only,
    then stack the rectangles from largest area to smallest with a shared lower-left anchor.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _most_common_color(g)
    anchor_color = _find_anchor_color(g, bg)
    rectangles: list[dict[str, int]] = []

    for color in np.unique(g):
        color = int(color)
        if color in (bg, anchor_color):
            continue
        comps = _components(g == color)
        assert 2 <= len(comps) <= 4, "each color must appear as 2 to 4 separate fragments"
        fragments = [_normalize_fragment(component) for component in comps]
        shapes = _rectangle_shapes(fragments, limit=2)
        assert len(shapes) == 1, "same-colored fragments must reconstruct one unique rectangle shape"
        height, width = shapes[0]
        rectangles.append(
            {
                "color": color,
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
            }
        )

    assert rectangles, "expected at least one colored rectangle family"
    areas = [int(rectangle["area"]) for rectangle in rectangles]
    assert len(set(areas)) == len(areas), "rectangle areas must be distinct"
    return _stack_rectangles(rectangles, bg)


def _sample_nested_dims(count: int) -> list[tuple[int, int]] | None:
    for _ in range(200):
        dims: list[tuple[int, int]] = []
        used_areas: set[int] = set()
        cur_h = _randint(*LARGEST_HEIGHT_RANGE)
        cur_w = _randint(*LARGEST_WIDTH_RANGE)
        dims.append((cur_h, cur_w))
        used_areas.add(cur_h * cur_w)

        ok = True
        for _ in range(count - 1):
            candidates: list[tuple[int, int]] = []
            for h in range(2, cur_h + 1):
                for w in range(2, cur_w + 1):
                    area = h * w
                    if area >= cur_h * cur_w:
                        continue
                    if area in used_areas:
                        continue
                    if h == cur_h and w == cur_w:
                        continue
                    candidates.append((h, w))
            if not candidates:
                ok = False
                break
            next_h, next_w = candidates[_randint(0, len(candidates) - 1)]
            dims.append((next_h, next_w))
            used_areas.add(next_h * next_w)
            cur_h, cur_w = next_h, next_w
        if ok:
            return dims
    return None


def _grow_partition(height: int, width: int, part_count: int) -> list[dict[str, object]] | None:
    area = height * width
    if area < part_count * 2:
        return None

    for _ in range(PARTITION_ATTEMPTS):
        labels = np.full((height, width), -1, dtype=int)
        seeds = RNG.choice(area, size=part_count, replace=False)
        frontiers = [set() for _ in range(part_count)]
        sizes = [1] * part_count

        def add_frontier(index: int, r: int, c: int) -> None:
            for nr, nc in _neighbors4(r, c, height, width):
                if labels[nr, nc] == -1:
                    frontiers[index].add((nr, nc))

        for index, seed in enumerate(seeds):
            r, c = divmod(int(seed), width)
            labels[r, c] = index

        for index, seed in enumerate(seeds):
            r, c = divmod(int(seed), width)
            add_frontier(index, r, c)

        unassigned = area - part_count
        while unassigned > 0:
            active = [index for index in range(part_count) if frontiers[index]]
            if not active:
                break
            weights = np.array([1.0 / sizes[index] for index in active], dtype=float)
            weights /= weights.sum()
            chosen = int(active[int(RNG.choice(len(active), p=weights))])
            choices = tuple(frontiers[chosen])
            r, c = choices[_randint(0, len(choices) - 1)]
            frontiers[chosen].discard((r, c))
            if labels[r, c] != -1:
                continue
            labels[r, c] = chosen
            sizes[chosen] += 1
            unassigned -= 1
            for frontier in frontiers:
                frontier.discard((r, c))
            add_frontier(chosen, r, c)

        if unassigned != 0 or min(sizes) < 2:
            continue

        fragments: list[dict[str, object]] = []
        for index in range(part_count):
            cells = [(r, c) for r in range(height) for c in range(width) if labels[r, c] == index]
            fragments.append(_normalize_fragment(cells))

        if len({fragment["cells"] for fragment in fragments}) != part_count:
            continue
        return fragments
    return None


def _compose_positive_parts(total: int, count: int) -> list[int] | None:
    if count <= 0 or total < count:
        return None
    if count == 1:
        return [total]
    cuts = sorted(int(x) for x in RNG.choice(np.arange(1, total), size=count - 1, replace=False))
    values: list[int] = []
    prev = 0
    for cut in cuts + [total]:
        values.append(cut - prev)
        prev = cut
    return values


def _fragment_from_box(r0: int, c0: int, height: int, width: int) -> dict[str, object]:
    cells = [(r, c) for r in range(r0, r0 + height) for c in range(c0, c0 + width)]
    return _normalize_fragment(cells)


def _structured_partition(height: int, width: int, part_count: int) -> list[dict[str, object]] | None:
    modes: list[str] = []
    if width >= part_count:
        modes.append("vertical")
    if height >= part_count:
        modes.append("horizontal")
    if part_count >= 3 and width >= 2 and height >= part_count - 1:
        modes.append("strip_plus_rows")
    if part_count >= 3 and height >= 2 and width >= part_count - 1:
        modes.append("strip_plus_cols")
    if not modes:
        return None

    mode = modes[_randint(0, len(modes) - 1)]
    fragments: list[dict[str, object]] = []

    if mode == "vertical":
        widths = _compose_positive_parts(width, part_count)
        if widths is None:
            return None
        c0 = 0
        for frag_w in widths:
            fragments.append(_fragment_from_box(0, c0, height, frag_w))
            c0 += frag_w
        return fragments

    if mode == "horizontal":
        heights = _compose_positive_parts(height, part_count)
        if heights is None:
            return None
        r0 = 0
        for frag_h in heights:
            fragments.append(_fragment_from_box(r0, 0, frag_h, width))
            r0 += frag_h
        return fragments

    if mode == "strip_plus_rows":
        strip_w = _randint(1, width - 1)
        row_heights = _compose_positive_parts(height, part_count - 1)
        if row_heights is None:
            return None
        fragments.append(_fragment_from_box(0, 0, height, strip_w))
        r0 = 0
        for frag_h in row_heights:
            fragments.append(_fragment_from_box(r0, strip_w, frag_h, width - strip_w))
            r0 += frag_h
        return fragments

    strip_h = _randint(1, height - 1)
    col_widths = _compose_positive_parts(width, part_count - 1)
    if col_widths is None:
        return None
    fragments.append(_fragment_from_box(height - strip_h, 0, strip_h, width))
    c0 = 0
    for frag_w in col_widths:
        fragments.append(_fragment_from_box(0, c0, height - strip_h, frag_w))
        c0 += frag_w
    return fragments


def _build_fragments_for_rectangle(height: int, width: int) -> list[dict[str, object]] | None:
    area = height * width
    counts = [count for count in range(2, 5) if area >= count]
    if not counts:
        return None

    for _ in range(RECTANGLE_ATTEMPTS):
        part_count = counts[_randint(0, len(counts) - 1)]
        fragments = _structured_partition(height, width, part_count)
        if fragments is None:
            continue
        shapes = _rectangle_shapes(fragments, limit=2)
        if shapes != [(height, width)]:
            continue
        return fragments
    return None


def _mark_blocked(blocked: np.ndarray, cells: tuple[tuple[int, int], ...]) -> None:
    height, width = blocked.shape
    for r, c in cells:
        for nr in range(max(0, r - 1), min(height, r + 2)):
            for nc in range(max(0, c - 1), min(width, c + 2)):
                blocked[nr, nc] = True


def _scatter_fragments(rectangles: list[dict[str, object]]) -> np.ndarray | None:
    grid_h = _randint(*GRID_HEIGHT_RANGE)
    grid_w = _randint(*GRID_WIDTH_RANGE)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    corners = [(0, 0), (0, grid_w - 1), (grid_h - 1, 0), (grid_h - 1, grid_w - 1)]
    anchor_r, anchor_c = corners[_randint(0, 3)]
    grid[anchor_r, anchor_c] = ANCHOR_COLOR
    _mark_blocked(blocked, ((anchor_r, anchor_c),))

    items: list[tuple[int, dict[str, object]]] = []
    for rectangle in rectangles:
        color = int(rectangle["color"])
        for fragment in rectangle["fragments"]:
            items.append((color, fragment))

    RNG.shuffle(items)
    items.sort(key=lambda item: int(item[1]["size"]), reverse=True)

    for color, fragment in items:
        frag_h = int(fragment["height"])
        frag_w = int(fragment["width"])
        if grid_h - frag_h - 2 < 1 or grid_w - frag_w - 2 < 1:
            return None

        placed = False
        for _ in range(SCATTER_ATTEMPTS):
            r0 = _randint(1, grid_h - frag_h - 1)
            c0 = _randint(1, grid_w - frag_w - 1)
            translated = tuple((r0 + r, c0 + c) for r, c in fragment["cells"])
            if any(blocked[r, c] for r, c in translated):
                continue
            for r, c in translated:
                grid[r, c] = color
            _mark_blocked(blocked, translated)
            placed = True
            break
        if not placed:
            return None
    return grid


def _expected_output(rectangles: list[dict[str, object]]) -> np.ndarray:
    simplified = [
        {
            "color": int(rectangle["color"]),
            "height": int(rectangle["height"]),
            "width": int(rectangle["width"]),
            "area": int(rectangle["area"]),
        }
        for rectangle in rectangles
    ]
    return _stack_rectangles(simplified, BG)


def _gen_attempt() -> np.ndarray | None:
    rect_count = _randint(*RECT_COUNT_RANGE)
    dims = _sample_nested_dims(rect_count)
    if dims is None:
        return None

    colors = list(RNG.choice(COLOR_POOL, size=rect_count, replace=False))
    rectangles: list[dict[str, object]] = []
    for color, (height, width) in zip(colors, dims):
        fragments = _build_fragments_for_rectangle(height, width)
        if fragments is None:
            return None
        rectangles.append(
            {
                "color": int(color),
                "height": int(height),
                "width": int(width),
                "area": int(height * width),
                "fragments": fragments,
            }
        )

    grid = _scatter_fragments(rectangles)
    if grid is None:
        return None

    expected = _expected_output(rectangles)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
