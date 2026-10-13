from __future__ import annotations

from collections import deque
import numpy as np


CLASS_ORDER = ("H", "V", "R")
TEMPLATE_AREA_RANGE = (4, 8)
GRID_H_RANGE = (18, 25)  # high-exclusive
GRID_W_RANGE = (18, 27)  # high-exclusive
OBJECT_COUNT_RANGE = (3, 7)  # high-exclusive
BOTTOM_STRIP_HEIGHT = 2
BOTTOM_BUFFER_ROWS = 1
PLACEMENT_GAP = 1
MARKER_GAP_RANGE = (1, 3)  # high-exclusive
MAX_GENERATION_ATTEMPTS = 300
_RNG = np.random.default_rng()
_TEMPLATE_CACHE: dict[str, list[np.ndarray]] | None = None


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue

            visited[r, c] = True
            queue = deque([(r, c)])
            coords: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in coords]
            cols = [cc for _, cc in coords]
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows) + 1, max(cols) + 1),
                }
            )

    return components


def _mask_from_component(component: dict) -> np.ndarray:
    r0, c0, r1, c1 = component["bbox"]
    mask = np.zeros((r1 - r0, c1 - c0), dtype=bool)
    for r, c in component["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _is_connected(mask: np.ndarray) -> bool:
    coords = np.argwhere(mask)
    if len(coords) == 0:
        return False

    h, w = mask.shape
    start = tuple(map(int, coords[0]))
    seen = {start}
    queue = deque([start])

    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                queue.append((nr, nc))

    return len(seen) == len(coords)


def _classify_mask(mask: np.ndarray) -> str:
    horizontal = np.array_equal(mask, np.flipud(mask))
    vertical = np.array_equal(mask, np.fliplr(mask))
    rotational = np.array_equal(mask, np.rot90(mask, 2))

    if horizontal and not vertical and not rotational:
        return "H"
    if vertical and not horizontal and not rotational:
        return "V"
    if rotational and not horizontal and not vertical:
        return "R"
    raise ValueError("mask does not belong to exactly one supported symmetry class")


def _extract_marker_map(grid: np.ndarray, bg: int) -> tuple[dict[str, int], np.ndarray]:
    strip = np.asarray(grid[-BOTTOM_STRIP_HEIGHT:, :], dtype=int)
    components = _find_components(strip, bg)
    if len(components) != 3:
        raise ValueError("expected exactly three bottom marker blocks")

    markers: list[tuple[int, int]] = []
    for component in components:
        r0, c0, r1, c1 = component["bbox"]
        mask = _mask_from_component(component)
        if len(component["coords"]) != 4 or (r1 - r0, c1 - c0) != (2, 2) or not mask.all():
            raise ValueError("bottom markers must be solid 2x2 blocks")
        markers.append((c0, int(component["color"])))

    markers.sort()
    return {CLASS_ORDER[i]: color for i, (_, color) in enumerate(markers)}, strip != bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each monochrome sprite according to its symmetry type using the bottom marker legend.

    Input structure:
    - The background is the dominant color.
    - Several non-overlapping monochrome sprites sit above a bottom legend area.
    - The bottom legend contains exactly three solid 2x2 color blocks in one row, read left to right as: horizontal-symmetry class, vertical-symmetry class, and 180-degree-rotation class.

    Rule:
    - Identify each sprite's bbox-local binary shape and classify it as exactly one of: horizontal-axis-only symmetry, vertical-axis-only symmetry, or 180-degree rotational symmetry without any axis symmetry.
    - A horizontal-symmetry sprite is flipped vertically in place; a vertical-symmetry sprite keeps its shape; a rotational-symmetry sprite keeps its shape.
    - Repaint each transformed sprite with the color from the corresponding legend block.
    - Erase the entire bottom legend back to the background color.

    Key insight:
    - The bottom blocks are not ordinary objects; they are a fixed left-to-right color lookup table keyed by symmetry class, so the task is first to classify each sprite's symmetry and only then apply the class-specific recolor.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    marker_map, marker_mask = _extract_marker_map(g, bg)

    working = g.copy()
    working[-BOTTOM_STRIP_HEIGHT:, :][marker_mask] = bg
    components = _find_components(working, bg)
    if not components:
        raise ValueError("expected at least one sprite above the legend")

    out = working.copy()
    for component in components:
        mask = _mask_from_component(component)
        sym_class = _classify_mask(mask)
        transformed = np.flipud(mask) if sym_class == "H" else mask
        r0, c0, r1, c1 = component["bbox"]
        patch = out[r0:r1, c0:c1]
        patch[:] = bg
        patch[transformed] = marker_map[sym_class]

    return out


def _mask_from_code(code: int, h: int, w: int) -> np.ndarray:
    bits = [(code >> i) & 1 for i in range(h * w)]
    return np.array(bits, dtype=bool).reshape((h, w))


def _build_templates() -> dict[str, list[np.ndarray]]:
    templates: dict[str, list[np.ndarray]] = {name: [] for name in CLASS_ORDER}
    seen: dict[str, set[tuple[tuple[int, int], bytes]]] = {name: set() for name in CLASS_ORDER}

    for h in range(2, 6):
        for w in range(2, 6):
            if h * w > 16:
                continue
            for code in range(1 << (h * w)):
                area = code.bit_count()
                if not (TEMPLATE_AREA_RANGE[0] <= area <= TEMPLATE_AREA_RANGE[1]):
                    continue

                mask = _mask_from_code(code, h, w)
                if not np.any(mask[0]) or not np.any(mask[-1]) or not np.any(mask[:, 0]) or not np.any(mask[:, -1]):
                    continue
                if not _is_connected(mask):
                    continue

                try:
                    sym_class = _classify_mask(mask)
                except ValueError:
                    continue

                key = (mask.shape, mask.tobytes())
                if key in seen[sym_class]:
                    continue
                seen[sym_class].add(key)
                templates[sym_class].append(mask)

    for sym_class, masks in templates.items():
        if not masks:
            raise RuntimeError(f"no templates found for class {sym_class}")
    return templates


def _get_templates() -> dict[str, list[np.ndarray]]:
    global _TEMPLATE_CACHE
    if _TEMPLATE_CACHE is None:
        _TEMPLATE_CACHE = _build_templates()
    return _TEMPLATE_CACHE


def _sample_template(sym_class: str) -> np.ndarray:
    mask = _get_templates()[sym_class][int(_RNG.integers(0, len(_get_templates()[sym_class])))]
    sampled = mask.copy()

    if sym_class == "H":
        if bool(_RNG.integers(0, 2)):
            sampled = np.fliplr(sampled)
    elif sym_class == "V":
        if bool(_RNG.integers(0, 2)):
            sampled = np.flipud(sampled)
    else:
        for _ in range(int(_RNG.integers(0, 4))):
            sampled = np.rot90(sampled)
        if bool(_RNG.integers(0, 2)):
            sampled = np.fliplr(sampled)

    assert _classify_mask(sampled) == sym_class
    return sampled


def _sample_classes(count: int) -> list[str]:
    classes = list(CLASS_ORDER)
    extras = count - len(CLASS_ORDER)
    if extras > 0:
        perm = list(_RNG.permutation(np.array(CLASS_ORDER, dtype=object)))
        classes.extend(str(item) for item in perm[:extras])
    _RNG.shuffle(classes)
    return classes


def _can_place(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    hh, ww = occupied.shape
    if not (0 <= r0 and 0 <= c0 and r0 + h <= hh and c0 + w <= ww):
        return False
    rs = max(0, r0 - PLACEMENT_GAP)
    re = min(hh, r0 + h + PLACEMENT_GAP)
    cs = max(0, c0 - PLACEMENT_GAP)
    ce = min(ww, c0 + w + PLACEMENT_GAP)
    return not occupied[rs:re, cs:ce].any()


def _paint_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    patch = grid[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
    patch[mask] = color


def _choose_colors(count: int) -> tuple[int, dict[str, int], list[int]]:
    colors = list(_RNG.permutation(np.arange(10)))
    bg = int(colors[0])
    marker_map = {cls: int(colors[i + 1]) for i, cls in enumerate(CLASS_ORDER)}
    object_colors = [int(color) for color in colors[4 : 4 + count]]
    return bg, marker_map, object_colors


def _place_markers(grid: np.ndarray, marker_map: dict[str, int]) -> None:
    h, w = grid.shape
    gap1 = int(_RNG.integers(*MARKER_GAP_RANGE))
    gap2 = int(_RNG.integers(*MARKER_GAP_RANGE))
    total_width = 6 + gap1 + gap2
    left = int(_RNG.integers(0, w - total_width + 1))
    starts = (
        left,
        left + 2 + gap1,
        left + 4 + gap1 + gap2,
    )
    for start, cls in zip(starts, CLASS_ORDER):
        grid[h - 2 : h, start : start + 2] = marker_map[cls]


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(*GRID_H_RANGE))
    w = int(_RNG.integers(*GRID_W_RANGE))
    object_count = int(_RNG.integers(*OBJECT_COUNT_RANGE))
    usable_h = h - BOTTOM_BUFFER_ROWS - BOTTOM_STRIP_HEIGHT
    if usable_h < 4:
        return None

    bg, marker_map, object_colors = _choose_colors(object_count)
    classes = _sample_classes(object_count)
    masks = [_sample_template(sym_class) for sym_class in classes]

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((usable_h, w), dtype=bool)
    placed: list[tuple[str, np.ndarray]] = []

    for sym_class, mask, color in zip(classes, masks, object_colors):
        mh, mw = mask.shape
        candidates = [
            (int(r0), int(c0))
            for r0 in range(usable_h - mh + 1)
            for c0 in range(w - mw + 1)
            if _can_place(occupied, int(r0), int(c0), mh, mw)
        ]
        if not candidates:
            return None

        r0, c0 = candidates[int(_RNG.integers(0, len(candidates)))]
        _paint_mask(grid, mask, r0, c0, color)
        occupied[r0 : r0 + mh, c0 : c0 + mw] |= mask
        placed.append((sym_class, mask))

    _place_markers(grid, marker_map)

    try:
        parsed_marker_map, marker_mask = _extract_marker_map(grid, bg)
    except ValueError:
        return None
    if parsed_marker_map != marker_map:
        return None
    if np.any(grid[usable_h : h - BOTTOM_STRIP_HEIGHT, :] != bg):
        return None

    working = grid.copy()
    working[-BOTTOM_STRIP_HEIGHT:, :][marker_mask] = bg
    components = _find_components(working, bg)
    if len(components) != object_count:
        return None

    class_counts = {cls: 0 for cls in CLASS_ORDER}
    for component in components:
        mask = _mask_from_component(component)
        if not (TEMPLATE_AREA_RANGE[0] <= int(mask.sum()) <= TEMPLATE_AREA_RANGE[1]):
            return None
        try:
            sym_class = _classify_mask(mask)
        except ValueError:
            return None
        class_counts[sym_class] += 1

    if any(class_counts[cls] == 0 for cls in CLASS_ORDER):
        return None

    try:
        out = main(grid)
    except Exception:
        return None
    if out.shape != grid.shape or np.any(out[-BOTTOM_STRIP_HEIGHT:, :] != bg):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid symmetry-legend ARC input")
