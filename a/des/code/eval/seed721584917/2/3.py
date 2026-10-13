from __future__ import annotations

from collections import defaultdict
import random

import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            for rr, cc in cells:
                mask[rr - r0, cc - c0] = True

            components.append({"color": color, "mask": mask})

    return components


def _assemble_pair(mask_a: np.ndarray, color_a: int, mask_b: np.ndarray, color_b: int, bg: int) -> np.ndarray:
    assert mask_a.shape == mask_b.shape
    h, w = map(int, mask_a.shape)

    cells_a = [(int(r), int(c)) for r, c in np.argwhere(mask_a)]
    cells_b = [(int(r), int(c)) for r, c in np.argwhere(mask_b)]
    assert cells_a and cells_b

    candidates: list[tuple[int, int, int, int, int, np.ndarray]] = []
    for dr in range(-(h - 1), h):
        for dc in range(-(w - 1), w):
            shifted_b = [(r + dr, c + dc) for r, c in cells_b]
            occupied = set(cells_a)
            overlap = False
            for cell in shifted_b:
                if cell in occupied:
                    overlap = True
                    break
                occupied.add(cell)
            if overlap:
                continue

            rows = [r for r, _ in occupied]
            cols = [c for _, c in occupied]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            out_h = r1 - r0 + 1
            out_w = c1 - c0 + 1
            if len(occupied) != out_h * out_w:
                continue

            rect = np.full((out_h, out_w), bg, dtype=int)
            for r, c in cells_a:
                rect[r - r0, c - c0] = color_a
            for r, c in shifted_b:
                rect[r - r0, c - c0] = color_b
            candidates.append((out_h * out_w, out_h, out_w, abs(dr) + abs(dc), dr, dc, rect))

    assert candidates, "pair must admit a complementary rectangular assembly"
    candidates.sort(key=lambda item: (item[0], item[1], item[2], item[3], item[4], item[5]))
    return candidates[0][-1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair fragments by identical local bounding-box size, translate each pair into a shared
    rectangle without overlap, then stack the resulting rectangles by area and width.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    bg = _get_bg(g)
    components = _find_components(g, bg)
    assert components and len(components) % 2 == 0

    groups: dict[tuple[int, int], list[dict[str, object]]] = defaultdict(list)
    for comp in components:
        mask = np.asarray(comp["mask"], dtype=bool)
        groups[tuple(map(int, mask.shape))].append(comp)

    rectangles: list[np.ndarray] = []
    for shape, pair in groups.items():
        assert len(pair) == 2, f"expected exactly two fragments for size group {shape}"
        first, second = pair
        color_a = int(first["color"])
        color_b = int(second["color"])
        assert color_a != color_b
        rect = _assemble_pair(
            np.asarray(first["mask"], dtype=bool),
            color_a,
            np.asarray(second["mask"], dtype=bool),
            color_b,
            bg,
        )
        rectangles.append(rect)

    rectangles.sort(key=lambda rect: (int(rect.shape[0] * rect.shape[1]), int(rect.shape[1])))

    out_h = sum(int(rect.shape[0]) for rect in rectangles) + len(rectangles) - 1
    out_w = max(int(rect.shape[1]) for rect in rectangles)
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for rect in rectangles:
        h, w = map(int, rect.shape)
        out[row : row + h, :w] = rect
        row += h + 1

    return out


PALETTE = tuple(range(10))
PAIR_COUNT_RANGE = (2, 4)
LOCAL_DIM_MIN = 2
LOCAL_DIM_MAX = 6
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
PLACEMENT_PADDING = 1
PLACEMENT_RETRIES = 300
MAX_GENERATION_ATTEMPTS = 400
TIE_AREA_PROB = 0.5


def _component_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    count = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            count += 1
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                cr, cc = stack.pop()
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
    return count


def _random_symmetry(mask_a: np.ndarray, mask_b: np.ndarray, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    if rng.random() < 0.5:
        mask_a = np.flipud(mask_a)
        mask_b = np.flipud(mask_b)
    if rng.random() < 0.5:
        mask_a = np.fliplr(mask_a)
        mask_b = np.fliplr(mask_b)
    return np.array(mask_a, copy=True), np.array(mask_b, copy=True)


def _make_horizontal_pair(h: int, w: int, shift: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    overlap_w = w - shift
    mask_a = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)

    mask_a[:, :shift] = True
    mask_b[:, overlap_w:] = True

    cuts = [rng.randint(0, h - 2) for _ in range(overlap_w)]
    for j, cut in enumerate(cuts):
        mask_a[: cut + 1, shift + j] = True
        mask_b[cut + 1 :, j] = True

    assert _component_count(mask_a) == 1
    assert _component_count(mask_b) == 1
    return _random_symmetry(mask_a, mask_b, rng)


def _make_vertical_pair(h: int, w: int, shift: int, rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    overlap_h = h - shift
    mask_a = np.zeros((h, w), dtype=bool)
    mask_b = np.zeros((h, w), dtype=bool)

    mask_a[:shift, :] = True
    mask_b[overlap_h:, :] = True

    cuts = [rng.randint(0, w - 2) for _ in range(overlap_h)]
    for i, cut in enumerate(cuts):
        mask_a[shift + i, : cut + 1] = True
        mask_b[i, cut + 1 :] = True

    assert _component_count(mask_a) == 1
    assert _component_count(mask_b) == 1
    return _random_symmetry(mask_a, mask_b, rng)


def _spec_catalog() -> list[dict[str, object]]:
    specs: list[dict[str, object]] = []
    for h in range(LOCAL_DIM_MIN, LOCAL_DIM_MAX + 1):
        for w in range(LOCAL_DIM_MIN, LOCAL_DIM_MAX + 1):
            for shift in range(1, w):
                specs.append(
                    {
                        "bbox": (h, w),
                        "axis": "h",
                        "shift": shift,
                        "out_shape": (h, w + shift),
                    }
                )
            for shift in range(1, h):
                specs.append(
                    {
                        "bbox": (h, w),
                        "axis": "v",
                        "shift": shift,
                        "out_shape": (h + shift, w),
                    }
                )
    return specs


def _choose_specs(pair_count: int, force_tie: bool, rng: random.Random) -> list[dict[str, object]] | None:
    specs = _spec_catalog()

    if force_tie and pair_count >= 2:
        tie_pairs: list[tuple[dict[str, object], dict[str, object]]] = []
        for i, first in enumerate(specs):
            area_a = int(first["out_shape"][0] * first["out_shape"][1])
            width_a = int(first["out_shape"][1])
            for second in specs[i + 1 :]:
                area_b = int(second["out_shape"][0] * second["out_shape"][1])
                width_b = int(second["out_shape"][1])
                if (
                    first["bbox"] != second["bbox"]
                    and area_a == area_b
                    and width_a != width_b
                ):
                    tie_pairs.append((first, second))
        if not tie_pairs:
            return None

        first, second = rng.choice(tie_pairs)
        selected = [dict(first), dict(second)]
        if pair_count == 2:
            return selected

        used_bbox = {tuple(first["bbox"]), tuple(second["bbox"])}
        used_areas = {int(first["out_shape"][0] * first["out_shape"][1])}
        pool = [dict(spec) for spec in specs if tuple(spec["bbox"]) not in used_bbox]
        rng.shuffle(pool)
        for spec in pool:
            area = int(spec["out_shape"][0] * spec["out_shape"][1])
            bbox = tuple(spec["bbox"])
            if area in used_areas or bbox in used_bbox:
                continue
            selected.append(spec)
            used_bbox.add(bbox)
            used_areas.add(area)
            if len(selected) == pair_count:
                return selected
        return None

    rng.shuffle(specs)
    selected: list[dict[str, object]] = []
    used_bbox: set[tuple[int, int]] = set()
    used_areas: set[int] = set()
    for spec in specs:
        bbox = tuple(spec["bbox"])
        area = int(spec["out_shape"][0] * spec["out_shape"][1])
        if bbox in used_bbox or area in used_areas:
            continue
        selected.append(dict(spec))
        used_bbox.add(bbox)
        used_areas.add(area)
        if len(selected) == pair_count:
            return selected
    return None


def _build_pair(spec: dict[str, object], rng: random.Random) -> tuple[np.ndarray, np.ndarray]:
    h, w = map(int, spec["bbox"])
    shift = int(spec["shift"])
    if spec["axis"] == "h":
        return _make_horizontal_pair(h, w, shift, rng)
    return _make_vertical_pair(h, w, shift, rng)


def _place_fragment(
    grid: np.ndarray,
    reserved: np.ndarray,
    mask: np.ndarray,
    color: int,
    rng: random.Random,
) -> bool:
    gh, gw = grid.shape
    h, w = mask.shape

    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(0, gh - h)
        c0 = rng.randint(0, gw - w)
        rr0 = max(0, r0 - PLACEMENT_PADDING)
        rr1 = min(gh, r0 + h + PLACEMENT_PADDING)
        cc0 = max(0, c0 - PLACEMENT_PADDING)
        cc1 = min(gw, c0 + w + PLACEMENT_PADDING)
        if reserved[rr0:rr1, cc0:cc1].any():
            continue
        grid[r0 : r0 + h, c0 : c0 + w][mask] = color
        reserved[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(PAIR_COUNT_RANGE[0], PAIR_COUNT_RANGE[1])
    force_tie = rng.random() < TIE_AREA_PROB
    specs = _choose_specs(pair_count, force_tie, rng)
    if specs is None:
        return None

    bg = rng.choice(PALETTE)
    colors = rng.sample([color for color in PALETTE if color != bg], 2 * pair_count)

    fragments: list[dict[str, object]] = []
    assembled_rects: list[np.ndarray] = []
    for i, spec in enumerate(specs):
        mask_a, mask_b = _build_pair(spec, rng)
        if rng.random() < 0.5:
            mask_a, mask_b = mask_b, mask_a
        fragments.append({"mask": mask_a, "color": colors[2 * i]})
        fragments.append({"mask": mask_b, "color": colors[2 * i + 1]})
        rect = _assemble_pair(mask_a, colors[2 * i], mask_b, colors[2 * i + 1], bg)
        assembled_rects.append(rect)

    areas = [int(rect.shape[0] * rect.shape[1]) for rect in assembled_rects]
    widths = [int(rect.shape[1]) for rect in assembled_rects]
    tie_found = False
    for i in range(len(areas)):
        for j in range(i + 1, len(areas)):
            if areas[i] == areas[j] and widths[i] != widths[j]:
                tie_found = True
                break
        if tie_found:
            break
    if tie_found != force_tie:
        return None
    if not force_tie and len(areas) != len(set(areas)):
        return None

    reserve_area = sum((int(item["mask"].shape[0]) + 2) * (int(item["mask"].shape[1]) + 2) for item in fragments)
    min_side = max(
        GRID_SIDE_MIN,
        max(int(item["mask"].shape[0]) for item in fragments) + 4,
        max(int(item["mask"].shape[1]) for item in fragments) + 4,
        int(np.ceil((reserve_area * 1.2) / GRID_SIDE_MAX)),
    )
    if min_side > GRID_SIDE_MAX:
        return None

    grid_h = rng.randint(min_side, GRID_SIDE_MAX)
    min_width = max(min_side, int(np.ceil((reserve_area * 1.1) / grid_h)))
    if min_width > GRID_SIDE_MAX:
        return None
    grid_w = rng.randint(min_width, GRID_SIDE_MAX)
    if rng.random() < 0.5:
        grid_h, grid_w = grid_w, grid_h

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    rng.shuffle(fragments)
    fragments.sort(key=lambda item: int(item["mask"].shape[0] * item["mask"].shape[1]), reverse=True)
    for fragment in fragments:
        if not _place_fragment(grid, reserved, np.asarray(fragment["mask"], dtype=bool), int(fragment["color"]), rng):
            return None

    values, counts = np.unique(grid, return_counts=True)
    bg_idx = np.where(values == bg)[0]
    assert len(bg_idx) == 1
    bg_count = int(counts[bg_idx[0]])
    if any(int(count) >= bg_count for value, count in zip(values, counts) if int(value) != bg):
        return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.size == 0:
        return None

    components = _find_components(grid, bg)
    if len(components) != 2 * pair_count:
        return None
    groups: dict[tuple[int, int], int] = defaultdict(int)
    for comp in components:
        groups[tuple(map(int, np.asarray(comp["mask"], dtype=bool).shape))] += 1
    if sorted(groups.values()) != [2] * pair_count:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
