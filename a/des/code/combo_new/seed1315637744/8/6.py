import random
from collections import deque

import numpy as np


BG = 0
GREY = 5
SEP = 9
MARKER_COLORS = [1, 2, 3, 4, 6, 7, 8]
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _signature_from_coords(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    min_r = min(r for r, _ in coords)
    min_c = min(c for _, c in coords)
    return tuple(sorted((r - min_r, c - min_c) for r, c in coords))


def _signature_from_mask(mask: np.ndarray) -> tuple[tuple[int, int], ...]:
    return tuple(sorted((int(r), int(c)) for r, c in np.argwhere(mask)))


def _objects_of_color(panel: np.ndarray, color: int) -> list[dict[str, object]]:
    objects: list[dict[str, object]] = []
    for comp in _components(panel == color):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        objects.append(
            {
                "coords": comp,
                "bbox": (r0, c0, r1, c1),
                "sig": _signature_from_coords(comp),
            }
        )
    return objects


def _split_panels(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    h, w = grid.shape
    sep_rows = [r for r in range(h) if np.all(grid[r, :] == SEP)]
    sep_cols = [c for c in range(w) if np.all(grid[:, c] == SEP)]
    candidates: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    if len(sep_rows) == 2:
        top_sep, bottom_sep = sep_rows
        panel_h = top_sep
        if panel_h > 0 and bottom_sep == 2 * panel_h + 1 and h == 3 * panel_h + 2:
            candidates.append(
                (
                    grid[:top_sep, :],
                    grid[top_sep + 1 : bottom_sep, :],
                    grid[bottom_sep + 1 :, :],
                )
            )

    if len(sep_cols) == 2:
        left_sep, right_sep = sep_cols
        panel_w = left_sep
        if panel_w > 0 and right_sep == 2 * panel_w + 1 and w == 3 * panel_w + 2:
            candidates.append(
                (
                    grid[:, :left_sep],
                    grid[:, left_sep + 1 : right_sep],
                    grid[:, right_sep + 1 :],
                )
            )

    assert len(candidates) == 1, "expected one valid 3-panel separator layout"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor the translated target objects by using palette markers to identify template shapes.

    Input structure:
    - A maroon one-cell separator splits the grid into three equal panels in one row or one column.
    - The first panel is a black template field containing several different grey 4-connected objects.
    - The second panel is a black palette field with one colored singleton marker for each template object; each marker lies within that object's bounding box under aligned panel coordinates.
    - The third panel is a black target field containing the same object shapes as the template, still grey but translated to new positions.

    Rule:
    - Use each palette marker position to determine which template-object bounding box it falls inside, thereby assigning a color to that template shape.
    - Normalize every grey object in the template and target panels to translation-invariant shape signatures.
    - Match each target object to the template object with the same shape signature and repaint the entire target object with that assigned palette color.
    - Return only the recolored target panel, without separators or the other two panels.

    Key insight:
    - The palette panel does not color the target directly; it first points to template objects by aligned position, and only then can the translated target objects be recolored by shape matching.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30

    template, palette, target = _split_panels(g)
    assert template.shape == palette.shape == target.shape
    assert set(map(int, np.unique(template))).issubset({BG, GREY})
    assert set(map(int, np.unique(target))).issubset({BG, GREY})
    assert not np.any(palette == GREY), "palette panel should not contain grey objects"

    template_objects = _objects_of_color(template, GREY)
    target_objects = _objects_of_color(target, GREY)
    assert template_objects and len(template_objects) == len(target_objects)

    template_sigs = [obj["sig"] for obj in template_objects]
    assert len(set(template_sigs)) == len(template_sigs), "template shapes must be distinct"
    assert set(obj["sig"] for obj in target_objects) == set(template_sigs), "target shapes must match template shapes"

    marker_points = [(int(r), int(c), int(palette[r, c])) for r, c in np.argwhere(palette != BG)]
    assert len(marker_points) == len(template_objects), "expected one palette marker per template object"

    shape_to_color: dict[tuple[tuple[int, int], ...], int] = {}
    for r, c, color in marker_points:
        matches = []
        for obj in template_objects:
            r0, c0, r1, c1 = obj["bbox"]
            if r0 <= r <= r1 and c0 <= c <= c1:
                matches.append(obj)
        assert len(matches) == 1, "each marker must select exactly one template bounding box"
        sig = matches[0]["sig"]
        assert sig not in shape_to_color, "each template object must receive exactly one marker"
        shape_to_color[sig] = color

    assert len(shape_to_color) == len(template_objects)

    out = np.full(target.shape, BG, dtype=int)
    for obj in target_objects:
        color = shape_to_color[obj["sig"]]
        for r, c in obj["coords"]:
            out[r, c] = color
    return out


def _normalize_mask(cells: set[tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    max_r = max(r for r, _ in cells)
    max_c = max(c for _, c in cells)
    mask = np.zeros((max_r - min_r + 1, max_c - min_c + 1), dtype=bool)
    for r, c in cells:
        mask[r - min_r, c - min_c] = True
    return mask


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(80):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        area_hi = min(8, h * w)
        if area_hi < 3:
            continue
        area = random.randint(3, area_hi)

        seed = (random.randrange(h), random.randrange(w))
        cells = {seed}
        while len(cells) < area:
            frontier = []
            for r, c in cells:
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))

        if len(cells) != area:
            continue

        mask = _normalize_mask(cells)
        if min(mask.shape) < 2 or max(mask.shape) > max(max_h, max_w):
            continue
        if mask.sum() < 3:
            continue
        if int(mask.sum()) == mask.size and random.random() < 0.8:
            continue

        for k in range(random.randrange(4)):
            mask = np.rot90(mask)
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask.copy()
    return None


def _sample_shapes(count: int, panel_shape: tuple[int, int]) -> list[np.ndarray] | None:
    max_h = min(5, panel_shape[0] - 2)
    max_w = min(5, panel_shape[1] - 2)
    if max_h < 2 or max_w < 2:
        return None

    shapes: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], ...]] = set()
    for _ in range(300):
        mask = _random_shape(max_h, max_w)
        if mask is None:
            continue
        if mask.shape[0] > panel_shape[0] - 2 or mask.shape[1] > panel_shape[1] - 2:
            continue
        sig = _signature_from_mask(mask)
        if sig in seen:
            continue
        seen.add(sig)
        shapes.append(mask)
        if len(shapes) == count:
            return shapes
    return None


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> bool:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    return not np.any(occupied[r0:r1, c0:c1])


def _mark_occupied(occupied: np.ndarray, top: int, left: int, h: int, w: int, pad: int = 1) -> None:
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(occupied.shape[0], top + h + pad)
    c1 = min(occupied.shape[1], left + w + pad)
    occupied[r0:r1, c0:c1] = True


def _place_shapes(shapes: list[np.ndarray], panel_shape: tuple[int, int]) -> list[tuple[int, int]] | None:
    occupied = np.zeros(panel_shape, dtype=bool)
    positions: list[tuple[int, int] | None] = [None] * len(shapes)
    order = sorted(
        range(len(shapes)),
        key=lambda i: (shapes[i].shape[0] * shapes[i].shape[1], int(shapes[i].sum())),
        reverse=True,
    )

    for idx in order:
        h, w = shapes[idx].shape
        candidates = [
            (r, c)
            for r in range(panel_shape[0] - h + 1)
            for c in range(panel_shape[1] - w + 1)
            if _can_place(occupied, r, c, h, w)
        ]
        if not candidates:
            return None
        top, left = random.choice(candidates)
        positions[idx] = (top, left)
        _mark_occupied(occupied, top, left, h, w)

    return [(int(r), int(c)) for r, c in positions if r is not None and c is not None]


def _paint_mask(panel: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    window = panel[top : top + mask.shape[0], left : left + mask.shape[1]]
    window[mask] = color


def _choose_marker(mask: np.ndarray) -> tuple[int, int]:
    filled = [(int(r), int(c)) for r, c in np.argwhere(mask)]
    holes = [
        (r, c)
        for r in range(mask.shape[0])
        for c in range(mask.shape[1])
        if not mask[r, c]
    ]
    if holes and random.random() < 0.5:
        return random.choice(holes)
    return random.choice(filled)


def _build_case(
    panel_shape: tuple[int, int],
    side_by_side: bool,
    shapes: list[np.ndarray],
    template_pos: list[tuple[int, int]],
    target_pos: list[tuple[int, int]],
    marker_offsets: list[tuple[int, int]],
    colors: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    template = np.full(panel_shape, BG, dtype=int)
    palette = np.full(panel_shape, BG, dtype=int)
    target = np.full(panel_shape, BG, dtype=int)
    expected = np.full(panel_shape, BG, dtype=int)

    for mask, (tr, tc), (rr, rc), (mr, mc), color in zip(
        shapes, template_pos, target_pos, marker_offsets, colors
    ):
        _paint_mask(template, mask, tr, tc, GREY)
        palette[tr + mr, tc + mc] = color
        _paint_mask(target, mask, rr, rc, GREY)
        _paint_mask(expected, mask, rr, rc, color)

    if side_by_side:
        sep = np.full((panel_shape[0], 1), SEP, dtype=int)
        inp = np.hstack([template, sep, palette, sep.copy(), target])
    else:
        sep = np.full((1, panel_shape[1]), SEP, dtype=int)
        inp = np.vstack([template, sep, palette, sep.copy(), target])
    return inp, expected


def _gen_attempt() -> np.ndarray | None:
    side_by_side = random.random() < 0.5
    if side_by_side:
        panel_shape = (random.randint(9, 15), random.randint(7, 9))
    else:
        panel_shape = (random.randint(7, 9), random.randint(9, 15))

    count_choices = [2, 3, 4]
    if panel_shape[0] * panel_shape[1] < 80:
        count_choices = [2, 3]
    count = random.choice(count_choices)

    shapes = _sample_shapes(count, panel_shape)
    if shapes is None:
        return None

    template_pos = _place_shapes(shapes, panel_shape)
    if template_pos is None:
        return None
    target_pos = _place_shapes(shapes, panel_shape)
    if target_pos is None or all(a == b for a, b in zip(template_pos, target_pos)):
        return None

    colors = random.sample(MARKER_COLORS, count)
    marker_offsets = [_choose_marker(mask) for mask in shapes]
    inp, expected = _build_case(
        panel_shape,
        side_by_side,
        shapes,
        template_pos,
        target_pos,
        marker_offsets,
        colors,
    )

    if np.array_equal(main(inp), expected):
        return inp
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
