from __future__ import annotations

import random

import numpy as np


BG = 0
TEMPLATE_SIZE = 4
OBJECT_COUNT_RANGE = (2, 4)
GRID_MAX = 30
TEMPLATE_ATTEMPTS = 200
PLACEMENT_ATTEMPTS = 80
GENERATION_ATTEMPTS = 400
COLOR_POOL = tuple(range(1, 10))

_DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
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
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in _DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _is_connected(mask: np.ndarray) -> bool:
    return int(mask.sum()) > 0 and len(_components(mask)) == 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    lo = coords.min(axis=0)
    hi = coords.max(axis=0)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def _spans_full_template(mask: np.ndarray) -> bool:
    return bool(mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any())


def _candidate_missing_positions(template: np.ndarray) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in np.argwhere(template):
        test = template.copy()
        test[int(r), int(c)] = False
        if _spans_full_template(test) and _is_connected(test):
            out.append((int(r), int(c)))
    return out


def _sample_template(object_count: int) -> tuple[np.ndarray, list[tuple[int, int]]]:
    min_cells = max(7, object_count + 4)
    for _ in range(TEMPLATE_ATTEMPTS):
        target_cells = random.randint(min_cells, 13)
        template = np.ones((TEMPLATE_SIZE, TEMPLATE_SIZE), dtype=bool)

        while int(template.sum()) > target_cells:
            removable_now = []
            for r, c in np.argwhere(template):
                test = template.copy()
                test[int(r), int(c)] = False
                if _spans_full_template(test) and _is_connected(test):
                    removable_now.append((int(r), int(c)))
            if not removable_now:
                break
            rr, cc = random.choice(removable_now)
            template[rr, cc] = False

        if int(template.sum()) != target_cells:
            continue

        removable = _candidate_missing_positions(template)
        if len(removable) < object_count:
            continue

        return template, random.sample(removable, object_count)

    raise RuntimeError("failed to sample a valid 4x4 template")


def _render_object(template: np.ndarray, missing: tuple[int, int], color: int) -> np.ndarray:
    patch = np.full((TEMPLATE_SIZE, TEMPLATE_SIZE), BG, dtype=int)
    patch[template] = int(color)
    patch[missing[0], missing[1]] = BG
    return patch


def _territory_bbox(top: int, left: int) -> tuple[int, int, int, int]:
    return top - 1, left - 1, top + TEMPLATE_SIZE, left + TEMPLATE_SIZE


def _overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _paint_border(out: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = bbox
    tr0 = max(0, r0 - 1)
    tc0 = max(0, c0 - 1)
    tr1 = min(out.shape[0] - 1, r1 + 1)
    tc1 = min(out.shape[1] - 1, c1 + 1)

    for rr in range(tr0, tr1 + 1):
        for cc in range(tc0, tc1 + 1):
            inside_object = r0 <= rr <= r1 and c0 <= cc <= c1
            if not inside_object and int(out[rr, cc]) == BG:
                out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reconstruct a shared 4x4 template from several near-identical objects, then color each object's outer border from a bottom palette using its missing pixel as the lookup key.

    Input structure:
    - The background is black.
    - Several separated monochrome objects appear in the grid; each one fits exactly inside a 4x4 bounding box.
    - All objects share the same hidden 4x4 template, but each object is missing exactly one template pixel, and different objects miss different pixels.
    - The bottommost non-background row is a palette: isolated single pixels arranged left to right with black gaps between them.

    Rule:
    - Ignore object colors and take the union of all 4x4-normalized object shapes to recover the complete template.
    - Among template cells that are absent from at least one object, sort those candidate missing positions in reading order.
    - For each object, find which candidate position is missing from that object and use the palette color at the same index.
    - Around that object's 4x4 bounding box, expand by one cell in every direction and fill the outer ring's black cells with the selected palette color.

    Key insight:
    - No single object identifies its border color directly; the color only becomes determined after jointly reconstructing the common template and turning the object's missing cell into a palette index.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    non_bg = np.argwhere(g != BG)
    assert non_bg.size > 0

    palette_row = int(non_bg[:, 0].max())
    palette_cols = [int(c) for c in np.flatnonzero(g[palette_row] != BG)]
    assert len(palette_cols) >= 2
    assert all(b - a >= 2 for a, b in zip(palette_cols, palette_cols[1:]))
    palette_colors = [int(g[palette_row, c]) for c in palette_cols]

    body = g.copy()
    body[palette_row, :] = BG

    object_colors = [int(color) for color in np.unique(body) if int(color) != BG]
    assert len(object_colors) == len(palette_colors)

    objects = []
    for color in object_colors:
        coords = np.argwhere(body == color)
        assert coords.size > 0
        bbox = _bbox_from_coords(coords)
        r0, c0, r1, c1 = bbox
        assert r1 - r0 + 1 == TEMPLATE_SIZE and c1 - c0 + 1 == TEMPLATE_SIZE
        shape = {(int(r - r0), int(c - c0)) for r, c in coords}
        objects.append({"bbox": bbox, "shape": shape})

    template_cells = set().union(*(obj["shape"] for obj in objects))
    key_positions = sorted(
        cell for cell in template_cells if any(cell not in obj["shape"] for obj in objects)
    )
    assert len(key_positions) == len(palette_colors)
    color_by_key = {cell: palette_colors[i] for i, cell in enumerate(key_positions)}

    out = g.copy()
    for obj in objects:
        missing = [cell for cell in key_positions if cell not in obj["shape"]]
        assert len(missing) == 1
        _paint_border(out, obj["bbox"], color_by_key[missing[0]])

    return out


def _choose_grid_shape(object_count: int) -> tuple[int, int]:
    min_dim = 12 + 2 * (object_count - 2)
    palette_width = 2 * object_count - 1
    height = random.randint(min_dim, GRID_MAX)
    width = random.randint(max(min_dim, palette_width + 1), GRID_MAX)
    return height, width


def _place_object_positions(height: int, width: int, object_count: int) -> list[tuple[int, int]] | None:
    candidates = [
        (r, c)
        for r in range(1, height - 6)
        for c in range(1, width - 4)
    ]
    if len(candidates) < object_count:
        return None

    for _ in range(PLACEMENT_ATTEMPTS):
        random.shuffle(candidates)
        picked: list[tuple[int, int]] = []
        territories: list[tuple[int, int, int, int]] = []
        for top, left in candidates:
            territory = _territory_bbox(top, left)
            if any(_overlap(territory, other) for other in territories):
                continue
            picked.append((top, left))
            territories.append(territory)
            if len(picked) == object_count:
                return picked
    return None


def _expected_output(
    inp: np.ndarray,
    placements: list[tuple[int, int]],
    missing_positions: list[tuple[int, int]],
    palette_colors: list[int],
) -> np.ndarray:
    order = sorted(missing_positions)
    color_by_missing = {cell: palette_colors[i] for i, cell in enumerate(order)}
    out = inp.copy()
    for (top, left), missing in zip(placements, missing_positions):
        bbox = (top, left, top + TEMPLATE_SIZE - 1, left + TEMPLATE_SIZE - 1)
        _paint_border(out, bbox, color_by_missing[missing])
    return out


def _gen_attempt() -> np.ndarray | None:
    object_count = random.randint(*OBJECT_COUNT_RANGE)
    template, missing_positions = _sample_template(object_count)

    height, width = _choose_grid_shape(object_count)
    placements = _place_object_positions(height, width, object_count)
    if placements is None:
        return None

    colors = list(COLOR_POOL)
    object_colors = random.sample(colors, object_count)
    remaining = [color for color in colors if color not in object_colors]
    palette_colors = random.sample(remaining, object_count)

    grid = np.full((height, width), BG, dtype=int)

    specs = list(zip(object_colors, missing_positions, placements))
    random.shuffle(specs)
    shuffled_missing = []
    shuffled_placements = []
    for color, missing, (top, left) in specs:
        patch = _render_object(template, missing, color)
        grid[top : top + TEMPLATE_SIZE, left : left + TEMPLATE_SIZE] = patch
        shuffled_missing.append(missing)
        shuffled_placements.append((top, left))

    palette_width = 2 * object_count - 1
    palette_start = random.randint(0, width - palette_width)
    for i, color in enumerate(palette_colors):
        grid[height - 1, palette_start + 2 * i] = int(color)

    expected = _expected_output(grid, shuffled_placements, shuffled_missing, palette_colors)
    try:
        if np.array_equal(main(grid), expected):
            return grid
    except Exception:
        return None
    return None


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
