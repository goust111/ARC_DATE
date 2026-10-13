import random

import numpy as np


GREY = 5
GRID_MIN = 24
GRID_MAX = 30
GEN_RETRIES = 400
SHAPE_RETRIES = 300
PLACE_RETRIES = 300
GAP = 1


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            comp: list[tuple[int, int]] = []
            stack = [(r, c)]
            seen[r, c] = True
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _cropped_component_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask


def _hole_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    bg = ~mask
    seen = np.zeros((h, w), dtype=bool)
    holes = 0
    for r in range(h):
        for c in range(w):
            if not bg[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            touches_edge = False
            while stack:
                rr, cc = stack.pop()
                if rr == 0 or cc == 0 or rr == h - 1 or cc == w - 1:
                    touches_edge = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and bg[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            if not touches_edge:
                holes += 1
    return holes


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _touches_border(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _legend_mapping(grid: np.ndarray, bg: int) -> dict[int, int]:
    h, w = grid.shape
    mapping: dict[int, int] = {}
    for color in map(int, np.unique(grid)):
        if color in (bg, GREY):
            continue
        for comp in _components(grid == color):
            if _touches_border(comp, h, w):
                continue
            r0, c0, r1, c1 = _bbox(comp)
            if (r1 - r0 + 1) < 2 or (c1 - c0 + 1) < 2:
                continue
            holes = _hole_count(_cropped_component_mask(comp))
            if holes in mapping and mapping[holes] != color:
                raise AssertionError("ambiguous legend mapping")
            mapping[holes] = color
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor grey objects by matching their 4-connected hole counts to the color legend.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    legend = _legend_mapping(g, bg)
    out = g.copy()

    for comp in _components(g == GREY):
        holes = _hole_count(_cropped_component_mask(comp))
        new_color = int(legend.get(holes, bg))
        for r, c in comp:
            out[r, c] = new_color

    return out


def _crop(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.where(mask)
    return mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]


def _random_orientation(mask: np.ndarray) -> np.ndarray:
    mask = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask.copy()


def _sample_hole_cells(height: int, width: int, holes: int) -> list[tuple[int, int]] | None:
    if holes == 0:
        return []
    interior = [(r, c) for r in range(1, height - 1) for c in range(1, width - 1)]
    random.shuffle(interior)
    chosen: list[tuple[int, int]] = []
    for r, c in interior:
        if all(abs(r - rr) + abs(c - cc) > 1 for rr, cc in chosen):
            chosen.append((r, c))
            if len(chosen) == holes:
                return chosen
    return None


def _build_shape(holes: int, compact: bool) -> np.ndarray:
    for _ in range(SHAPE_RETRIES):
        if compact:
            height = random.randint(max(4, holes + 3), max(5, holes + 4))
            width = random.randint(max(4, holes + 3), max(5, holes + 5))
            arm_count = random.randint(0, 2)
        else:
            height = random.randint(max(4, holes + 3), max(6, holes + 5))
            width = random.randint(max(4, holes + 3), max(6, holes + 6))
            arm_count = random.randint(0, 3)

        hole_cells = _sample_hole_cells(height, width, holes)
        if hole_cells is None:
            continue

        base = np.ones((height, width), dtype=bool)
        for r, c in hole_cells:
            base[r, c] = False

        canvas = np.zeros((height + 6, width + 6), dtype=bool)
        off = 3
        canvas[off : off + height, off : off + width] = base

        for _ in range(arm_count):
            side = random.choice("tblr")
            if side in "tb":
                arm_h = 1
                arm_w = random.randint(1, max(1, width - 1))
                c0 = off + random.randint(0, width - arm_w)
                if side == "t":
                    canvas[off - arm_h : off, c0 : c0 + arm_w] = True
                else:
                    canvas[off + height : off + height + arm_h, c0 : c0 + arm_w] = True
            else:
                arm_w = 1
                arm_h = random.randint(1, max(1, height - 1))
                r0 = off + random.randint(0, height - arm_h)
                if side == "l":
                    canvas[r0 : r0 + arm_h, off - arm_w : off] = True
                else:
                    canvas[r0 : r0 + arm_h, off + width : off + width + arm_w] = True

        mask = _random_orientation(_crop(canvas))
        if _hole_count(mask) == holes:
            return mask

    raise RuntimeError("failed to build shape")


def _band_thickness(side: str, grid_h: int, grid_w: int, legend_masks: list[np.ndarray]) -> int:
    total_area = sum(int(mask.sum()) for mask in legend_masks)
    max_h = max(mask.shape[0] for mask in legend_masks)
    max_w = max(mask.shape[1] for mask in legend_masks)
    if side in ("top", "bottom"):
        usable = max(4, grid_w - 2)
        packed = int(np.ceil(1.6 * total_area / usable)) + 2
        return max(max_h + 2, packed)
    usable = max(4, grid_h - 2)
    packed = int(np.ceil(1.6 * total_area / usable)) + 2
    return max(max_w + 2, packed)


def _regions(side: str, grid_h: int, grid_w: int, band: int) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    if side == "top":
        return (1, 1, band, grid_w - 1), (band + 1, 1, grid_h - 1, grid_w - 1)
    if side == "bottom":
        return (grid_h - band, 1, grid_h - 1, grid_w - 1), (1, 1, grid_h - band - 1, grid_w - 1)
    if side == "left":
        return (1, 1, grid_h - 1, band), (1, band + 1, grid_h - 1, grid_w - 1)
    return (1, grid_w - band, grid_h - 1, grid_w - 1), (1, 1, grid_h - 1, grid_w - band - 1)


def _can_place(mask: np.ndarray, occupied: np.ndarray, top: int, left: int, gap: int) -> bool:
    h, w = occupied.shape
    rows, cols = np.where(mask)
    for dr, dc in zip(rows.tolist(), cols.tolist(), strict=False):
        r = top + int(dr)
        c = left + int(dc)
        if occupied[r, c]:
            return False
        r0 = max(0, r - gap)
        r1 = min(h, r + gap + 1)
        c0 = max(0, c - gap)
        c1 = min(w, c + gap + 1)
        if occupied[r0:r1, c0:c1].any():
            return False
    return True


def _stamp(mask: np.ndarray, occupied: np.ndarray, top: int, left: int) -> None:
    rows, cols = np.where(mask)
    occupied[top + rows, left + cols] = True


def _place_items(
    items: list[dict[str, object]],
    region: tuple[int, int, int, int],
    occupied: np.ndarray,
) -> list[dict[str, object]] | None:
    rmin, cmin, rmax, cmax = region
    placed: list[dict[str, object]] = []
    for item in sorted(items, key=lambda x: -int(np.sum(x["mask"]))):
        mask = np.asarray(item["mask"], dtype=bool)
        mh, mw = mask.shape
        if mh > (rmax - rmin) or mw > (cmax - cmin):
            return None
        max_top = rmax - mh
        max_left = cmax - mw
        if max_top < rmin or max_left < cmin:
            return None

        success = False
        for _ in range(PLACE_RETRIES):
            top = random.randint(rmin, max_top)
            left = random.randint(cmin, max_left)
            if _can_place(mask, occupied, top, left, GAP):
                _stamp(mask, occupied, top, left)
                new_item = dict(item)
                new_item["top"] = top
                new_item["left"] = left
                placed.append(new_item)
                success = True
                break
        if not success:
            return None
    return placed


def _paint(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    rows, cols = np.where(mask)
    grid[top + rows, left + cols] = int(color)


def _grey_hole_counts(legend_holes: list[int]) -> list[int]:
    counts = list(legend_holes)
    for _ in range(random.randint(0, 2)):
        counts.append(random.choice(legend_holes))

    remaining = [h for h in range(4) if h not in legend_holes]
    if remaining and random.random() < 0.75:
        counts.append(random.choice(remaining))
        if random.random() < 0.35:
            counts.append(random.choice(remaining))

    random.shuffle(counts)
    return counts


def _valid_layout(grid_h: int, grid_w: int, band: int, side: str, grey_masks: list[np.ndarray]) -> bool:
    if side in ("top", "bottom"):
        free_h = grid_h - band - 2
        free_w = grid_w - 2
    else:
        free_h = grid_h - 2
        free_w = grid_w - band - 2
    if free_h < 6 or free_w < 6:
        return False
    return all(mask.shape[0] <= free_h and mask.shape[1] <= free_w for mask in grey_masks)


def _generate_case() -> np.ndarray | None:
    grid_h = random.randint(GRID_MIN, GRID_MAX)
    grid_w = random.randint(GRID_MIN, GRID_MAX)
    side = random.choice(["top", "bottom", "left", "right"])

    bg = random.choice([c for c in range(10) if c != GREY])
    legend_holes = random.sample(range(4), random.randint(2, 4))
    legend_colors = random.sample([c for c in range(10) if c not in (bg, GREY)], len(legend_holes))
    grey_holes = _grey_hole_counts(legend_holes)

    legend_items = [
        {"holes": holes, "color": color, "mask": _build_shape(holes, compact=True)}
        for holes, color in zip(legend_holes, legend_colors, strict=False)
    ]
    grey_items = [{"holes": holes, "mask": _build_shape(holes, compact=False)} for holes in grey_holes]

    band = _band_thickness(side, grid_h, grid_w, [np.asarray(item["mask"]) for item in legend_items])
    if not _valid_layout(grid_h, grid_w, band, side, [np.asarray(item["mask"]) for item in grey_items]):
        return None

    legend_region, object_region = _regions(side, grid_h, grid_w, band)
    occupied = np.zeros((grid_h, grid_w), dtype=bool)

    placed_legends = _place_items(legend_items, legend_region, occupied)
    if placed_legends is None:
        return None

    placed_greys = _place_items(grey_items, object_region, occupied)
    if placed_greys is None:
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for item in placed_legends:
        _paint(grid, np.asarray(item["mask"], dtype=bool), int(item["top"]), int(item["left"]), int(item["color"]))
    for item in placed_greys:
        _paint(grid, np.asarray(item["mask"], dtype=bool), int(item["top"]), int(item["left"]), GREY)

    legend = _legend_mapping(grid, bg)
    if set(legend.keys()) != set(legend_holes):
        return None

    actual_grey_holes = sorted(_hole_count(_cropped_component_mask(comp)) for comp in _components(grid == GREY))
    if actual_grey_holes != sorted(grey_holes):
        return None

    out = main(grid)
    for comp in _components(grid == GREY):
        holes = _hole_count(_cropped_component_mask(comp))
        expected = legend.get(holes, bg)
        if any(int(out[r, c]) != int(expected) for r, c in comp):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
