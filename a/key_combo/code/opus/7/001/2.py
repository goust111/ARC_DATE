import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREY = 5


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    """BBox of an (N,2) coords array in (r, c) order."""
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a grid to an inclusive bbox."""
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def scale_sprite(sprite: np.ndarray, factor: int) -> np.ndarray:
    """Nearest-neighbor integer pixel upscale."""
    k = int(factor)
    if k <= 0:
        raise ValueError("scale factor must be positive")
    return np.kron(sprite, np.ones((k, k), dtype=sprite.dtype))


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    """Return connected components of a boolean mask as coord arrays."""
    if connectivity != 4:
        raise ValueError("this task uses only 4-connectivity")

    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue

            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))

            comps.append(np.array(coords, dtype=int))

    return comps


def main(input_grid: np.ndarray) -> np.ndarray:
    """
    Count the area-two blue components below the divider and use that count as the red template's upscale factor.

    Input structure:
    - A black grid contains exactly one full-width gray horizontal divider.
    - Above the divider, red pixels form one 4-connected template shape on black background.
    - Below the divider, blue pixels form several separated 4-connected components of mixed areas.

    Rule:
    - Split the grid at the gray divider into an upper template region and a lower marker region.
    - Count how many blue 4-connected components in the lower region have area exactly 2.
    - Crop the upper red template to its tight bounding box, keeping any black gaps inside that box.
    - Replace each cropped pixel by an NxN block, where N is the area-two component count.

    Key insight:
    - The blue markers encode a number by component area filtering: only domino-sized components count, and that count controls the pixel-upscale multiplier.
    """
    grid = np.asarray(input_grid, dtype=int)
    if grid.ndim != 2:
        raise ValueError("expected a 2D grid")

    h, w = grid.shape
    divider_rows = [r for r in range(h) if np.all(grid[r, :] == GREY)]
    if len(divider_rows) != 1:
        raise ValueError("expected exactly one full-width gray divider row")

    split = divider_rows[0]
    if split == 0 or split == h - 1:
        raise ValueError("divider must leave a region above and below")

    top = grid[:split, :]
    bottom = grid[split + 1 :, :]

    template_coords = np.argwhere(top == RED)
    if template_coords.size == 0:
        raise ValueError("missing red template")

    template = crop_bbox(top, bbox_coords(template_coords))
    factor = sum(comp.shape[0] == 2 for comp in components_mask(bottom == BLUE, connectivity=4))
    if factor <= 0:
        raise ValueError("expected at least one area-two blue component")

    return scale_sprite(template, factor)


AREA_SHAPES: dict[int, list[np.ndarray]] = {
    1: [np.array([[1]], dtype=bool)],
    2: [
        np.array([[1, 1]], dtype=bool),
        np.array([[1], [1]], dtype=bool),
    ],
    3: [
        np.array([[1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1]], dtype=bool),
        np.array([[1, 1], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1]], dtype=bool),
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1, 1]], dtype=bool),
        np.array([[1], [1], [1], [1]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
        np.array([[0, 1], [0, 1], [1, 1]], dtype=bool),
        np.array([[1, 1], [1, 0], [1, 0]], dtype=bool),
        np.array([[1, 1], [0, 1], [0, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
        np.array([[0, 1], [1, 1], [0, 1]], dtype=bool),
    ],
}


BLUE_SCENARIOS = {
    2: ([2, 2, 1, 4], [2, 2, 1, 3, 4]),
    3: ([2, 2, 2, 1, 4], [2, 2, 2, 3, 4], [2, 2, 2, 1, 1, 4]),
}


def _random_template_mask(max_factor: int) -> np.ndarray | None:
    dims: list[tuple[int, int]] = []
    for hh in range(2, 7):
        for ww in range(3, 8):
            if hh * max_factor <= 30 and ww * max_factor <= 30 and hh * ww >= 5:
                dims.append((hh, ww))
    random.shuffle(dims)

    for box_h, box_w in dims:
        upper_area = min(12, box_h * box_w - 1)
        if upper_area < 4:
            continue
        target = random.randint(4, upper_area)

        for _ in range(120):
            mask = np.zeros((box_h, box_w), dtype=bool)
            start = (random.randrange(box_h), random.randrange(box_w))
            mask[start] = True

            while int(mask.sum()) < target:
                frontier = set()
                for r, c in np.argwhere(mask):
                    for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                        if 0 <= nr < box_h and 0 <= nc < box_w and not mask[nr, nc]:
                            frontier.add((int(nr), int(nc)))
                if not frontier:
                    break
                nr, nc = random.choice(tuple(frontier))
                mask[nr, nc] = True

            if int(mask.sum()) != target:
                continue

            rows = np.where(mask.any(axis=1))[0]
            cols = np.where(mask.any(axis=0))[0]
            if rows.size == 0 or cols.size == 0:
                continue
            if rows[0] != 0 or rows[-1] != box_h - 1 or cols[0] != 0 or cols[-1] != box_w - 1:
                continue
            if np.all(mask):
                continue
            if len(components_mask(mask, connectivity=4)) != 1:
                continue
            return mask

    return None


def _try_place_shapes(height: int, width: int, shapes: list[np.ndarray]) -> np.ndarray | None:
    grid = np.full((height, width), BLACK, dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = sorted(shapes, key=lambda shp: (-int(shp.shape[0] * shp.shape[1]), -int(shp.sum())))
    for shape in order:
        sh, sw = shape.shape
        placements: list[list[tuple[int, int]]] = []

        for r0 in range(height - sh + 1):
            for c0 in range(width - sw + 1):
                cells = [(r0 + r, c0 + c) for r, c in np.argwhere(shape)]
                ok = True
                for rr, cc in cells:
                    if occupied[rr, cc]:
                        ok = False
                        break
                    for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                        if 0 <= nr < height and 0 <= nc < width and occupied[nr, nc]:
                            ok = False
                            break
                    if not ok:
                        break
                if ok:
                    placements.append(cells)

        if not placements:
            return None

        for rr, cc in random.choice(placements):
            occupied[rr, cc] = True
            grid[rr, cc] = BLUE

    return grid


def _build_blue_region(n_pairs: int, height: int, width: int) -> tuple[np.ndarray, list[int]] | None:
    areas = list(random.choice(BLUE_SCENARIOS[n_pairs]))
    random.shuffle(areas)
    shapes = [random.choice(AREA_SHAPES[area]).copy() for area in areas]

    for _ in range(80):
        random.shuffle(shapes)
        region = _try_place_shapes(height, width, shapes)
        if region is None:
            continue
        return region, areas

    return None


def _validate_instance(grid: np.ndarray, expected_factor: int, expected_template: np.ndarray) -> None:
    out = main(grid)
    expected = scale_sprite(expected_template, expected_factor)
    if not np.array_equal(out, expected):
        raise AssertionError("main() does not match the constructed rule")
    if max(grid.shape) > 30 or max(out.shape) > 30:
        raise AssertionError("grid exceeds ARC size limit")

    divider_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == GREY)]
    if len(divider_rows) != 1:
        raise AssertionError("expected one divider row")
    split = divider_rows[0]

    top = grid[:split, :]
    bottom = grid[split + 1 :, :]
    red_mask = top == RED
    red_comps = components_mask(red_mask, connectivity=4)
    if len(red_comps) != 1:
        raise AssertionError("expected one red template component")
    if not (4 <= int(red_mask.sum()) <= 12):
        raise AssertionError("template area out of range")

    template = crop_bbox(top, bbox_coords(np.argwhere(red_mask)))
    if np.all(template != RED):
        raise AssertionError("cropped template lost the red shape")
    if not np.any(template == BLACK):
        raise AssertionError("template bbox must contain black gaps")
    if template.shape != expected_template.shape or not np.array_equal(template, expected_template):
        raise AssertionError("template crop mismatch")

    blue_comps = components_mask(bottom == BLUE, connectivity=4)
    sizes = sorted(int(comp.shape[0]) for comp in blue_comps)
    if sizes.count(2) != expected_factor:
        raise AssertionError("wrong number of area-two components")
    if len(blue_comps) == expected_factor:
        raise AssertionError("count-all-components shortcut not ruled out")
    if sizes.count(1) == expected_factor or sizes.count(3) == expected_factor or sizes.count(4) == expected_factor:
        raise AssertionError("wrong area-count shortcut not ruled out")
    if 1 not in sizes or not ({3, 4} & set(sizes)):
        raise AssertionError("missing distractor coverage")


def generate_input() -> np.ndarray:
    for _ in range(400):
        factor = random.choice((2, 3))
        template_mask = _random_template_mask(max_factor=factor)
        if template_mask is None:
            continue

        template_h, template_w = template_mask.shape
        top_h = random.randint(template_h + 1, min(10, template_h + 4))
        width = random.randint(max(14, template_w + 6), 22)
        lower_h = random.randint(7, 12)
        total_h = top_h + 1 + lower_h
        if total_h > 24:
            continue

        built = _build_blue_region(factor, lower_h, width)
        if built is None:
            continue
        lower_region, _areas = built

        grid = np.full((total_h, width), BLACK, dtype=int)
        grid[top_h, :] = GREY

        r0 = random.randint(0, top_h - template_h)
        c0 = random.randint(0, width - template_w)
        template = np.full((template_h, template_w), BLACK, dtype=int)
        template[template_mask] = RED
        grid[r0 : r0 + template_h, c0 : c0 + template_w] = template
        grid[top_h + 1 :, :] = lower_region

        expected_template = crop_bbox(grid[:top_h, :], bbox_coords(np.argwhere(grid[:top_h, :] == RED)))
        if expected_template.shape != template.shape or not np.array_equal(expected_template, template):
            continue

        try:
            _validate_instance(grid, factor, expected_template)
        except Exception:
            continue

        return grid

    raise RuntimeError("failed to generate a valid ARC input")
