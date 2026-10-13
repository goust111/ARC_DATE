from __future__ import annotations

import random
from collections import deque

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
ALL_COLORS = list(range(10))

# Generator configuration.
GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_RANGE = (2, 5)
MAX_GENERATION_ATTEMPTS = 500
FAMILY_NAMES = ("rect", "l", "plus", "donut", "t", "poly")


def _get_bg(grid: np.ndarray) -> int:
    vals, cnts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(cnts)])


def _boundary_pixels(mask: np.ndarray) -> int:
    h, w = mask.shape
    total = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            for dr, dc in DIR4:
                nr = r + dr
                nc = c + dc
                if not (0 <= nr < h and 0 <= nc < w) or not mask[nr, nc]:
                    total += 1
                    break
    return total


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _mask_connected(mask: np.ndarray) -> bool:
    cells = np.argwhere(mask)
    if cells.size == 0:
        return False
    start = tuple(map(int, cells[0]))
    q: deque[tuple[int, int]] = deque([start])
    seen = {start}
    h, w = mask.shape
    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                q.append((nr, nc))
    return len(seen) == int(mask.sum())


def _transform_mask(mask: np.ndarray) -> np.ndarray:
    out = np.rot90(mask, random.randrange(4))
    if random.random() < 0.5:
        out = out[::-1, :]
    if random.random() < 0.5:
        out = out[:, ::-1]
    return _crop_mask(out)


def _make_rect() -> np.ndarray:
    h = random.randint(2, 6)
    w = random.randint(2, 6)
    return np.ones((h, w), dtype=bool)


def _make_l() -> np.ndarray:
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    t = random.randint(1, min(2, h - 2, w - 2))
    mask = np.zeros((h, w), dtype=bool)
    mask[:t, :] = True
    mask[:, :t] = True
    return mask


def _make_plus() -> np.ndarray:
    t = random.randint(1, 2)
    top = random.randint(1, 3)
    bottom = random.randint(1, 3)
    left = random.randint(1, 3)
    right = random.randint(1, 3)
    h = top + t + bottom
    w = left + t + right
    mask = np.zeros((h, w), dtype=bool)
    mask[top : top + t, :] = True
    mask[:, left : left + t] = True
    return mask


def _make_donut() -> np.ndarray:
    h = random.randint(4, 7)
    w = random.randint(4, 7)
    thickness_options = [1]
    if min(h, w) >= 6:
        thickness_options.append(2)
    t = random.choice(thickness_options)
    mask = np.ones((h, w), dtype=bool)
    mask[t : h - t, t : w - t] = False
    return mask


def _make_t() -> np.ndarray:
    bar_t = random.randint(1, 2)
    bar_w = random.randint(4, 7)
    stem_t = random.randint(1, min(2, bar_w - 2))
    stem_h = random.randint(2, 5)
    mask = np.zeros((bar_t + stem_h, bar_w), dtype=bool)
    mask[:bar_t, :] = True
    c0 = (bar_w - stem_t) // 2
    mask[bar_t:, c0 : c0 + stem_t] = True
    return mask


def _make_poly() -> np.ndarray:
    target = random.randint(5, 12)
    cells = {(0, 0)}
    while len(cells) < target:
        frontier: dict[tuple[int, int], int] = {}
        for r, c in cells:
            for dr, dc in DIR4:
                cand = (r + dr, c + dc)
                if cand in cells:
                    continue
                frontier[cand] = frontier.get(cand, 0) + 1
        choices = list(frontier.items())
        pos = random.choices(
            [p for p, _ in choices],
            weights=[w * w for _, w in choices],
            k=1,
        )[0]
        cells.add(pos)

    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    r0 = min(rows)
    c0 = min(cols)
    h = max(rows) - r0 + 1
    w = max(cols) - c0 + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r - r0, c - c0] = True
    return mask


SHAPE_BUILDERS = {
    "rect": _make_rect,
    "l": _make_l,
    "plus": _make_plus,
    "donut": _make_donut,
    "t": _make_t,
    "poly": _make_poly,
}


def _build_shape(family: str) -> dict[str, object]:
    for _ in range(80):
        mask = _transform_mask(SHAPE_BUILDERS[family]())
        area = int(mask.sum())
        boundary = _boundary_pixels(mask)
        if area < 4 or boundary < 4:
            continue
        if not _mask_connected(mask):
            continue
        return {
            "family": family,
            "mask": mask,
            "area": area,
            "boundary": boundary,
        }
    raise AssertionError(f"failed to build shape family {family}")


def _choose_shape_set(n_objects: int) -> list[dict[str, object]] | None:
    for _ in range(120):
        families = random.sample(FAMILY_NAMES, k=min(n_objects, len(FAMILY_NAMES)))
        while len(families) < n_objects:
            families.append(random.choice(FAMILY_NAMES))
        random.shuffle(families)

        shapes: list[dict[str, object]] = []
        seen_boundaries: set[int] = set()
        for family in families:
            picked = None
            for _ in range(80):
                candidate = _build_shape(family)
                boundary = int(candidate["boundary"])
                if boundary in seen_boundaries:
                    continue
                picked = candidate
                seen_boundaries.add(boundary)
                break
            if picked is None:
                break
            shapes.append(picked)

        if len(shapes) != n_objects:
            continue

        area_order = sorted(
            range(n_objects),
            key=lambda i: (-int(shapes[i]["area"]), -int(shapes[i]["boundary"])),
        )
        boundary_order = sorted(
            range(n_objects),
            key=lambda i: (-int(shapes[i]["boundary"]), -int(shapes[i]["area"])),
        )
        if area_order == boundary_order:
            continue
        return shapes
    return None


def _place_shapes(
    h: int,
    w: int,
    bg: int,
    shapes: list[dict[str, object]],
    colors: list[int],
) -> np.ndarray | None:
    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    order = sorted(
        range(len(shapes)),
        key=lambda i: (
            -max(np.asarray(shapes[i]["mask"]).shape),
            -int(np.asarray(shapes[i]["mask"]).sum()),
        ),
    )

    for idx in order:
        mask = np.asarray(shapes[idx]["mask"], dtype=bool)
        mh, mw = mask.shape
        positions: list[tuple[int, int]] = []
        for r in range(1, h - mh):
            for c in range(1, w - mw):
                r0 = r - 1
                r1 = r + mh + 1
                c0 = c - 1
                c1 = c + mw + 1
                if occupied[r0:r1, c0:c1].any():
                    continue
                positions.append((r, c))
        if not positions:
            return None

        r, c = random.choice(positions)
        grid[r : r + mh, c : c + mw][mask] = colors[idx]
        occupied[r - 1 : r + mh + 1, c - 1 : c + mw + 1] = True

    return grid


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            r0 = r1 = r
            c0 = c1 = c
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                r0 = min(r0, rr)
                c0 = min(c0, cc)
                r1 = max(r1, rr)
                c1 = max(c1, cc)
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(g[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))

            boundary = 0
            for rr, cc in coords:
                for dr, dc in DIR4:
                    nr = rr + dr
                    nc = cc + dc
                    if not (0 <= nr < h and 0 <= nc < w) or int(g[nr, nc]) == bg:
                        boundary += 1
                        break

            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": len(coords),
                    "boundary": boundary,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    return objects


def _render_output(bg: int, ordered_objects: list[dict[str, object]]) -> np.ndarray:
    n = len(ordered_objects)
    size = 2 * n - 1
    out = np.full((size, size), bg, dtype=int)
    for layer, obj in enumerate(ordered_objects):
        color = int(obj["color"])
        lo = layer
        hi = size - layer - 1
        out[lo, lo : hi + 1] = color
        out[hi, lo : hi + 1] = color
        out[lo : hi + 1, lo] = color
        out[lo : hi + 1, hi] = color
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Extract each object's contour size and redraw the colors as nested square rings."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = _get_bg(g)
    objects = _find_components(g, bg)
    assert objects, "expected at least one foreground object"

    ordered = sorted(
        objects,
        key=lambda obj: (
            -int(obj["boundary"]),
            int(obj["bbox"][0]),
            int(obj["bbox"][1]),
            -int(obj["area"]),
        ),
    )
    return _render_output(bg, ordered)


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        n_objects = random.randint(*OBJECT_COUNT_RANGE)
        bg = random.choice(ALL_COLORS)
        palette = [color for color in ALL_COLORS if color != bg]
        colors = random.sample(palette, k=n_objects)

        shapes = _choose_shape_set(n_objects)
        if shapes is None:
            continue

        h = random.randint(*GRID_H_RANGE)
        w = random.randint(*GRID_W_RANGE)
        grid = _place_shapes(h, w, bg, shapes, colors)
        if grid is None:
            continue

        counts = {int(color): int(np.count_nonzero(grid == color)) for color in colors}
        if any(counts[color] >= int(np.count_nonzero(grid == bg)) for color in colors):
            continue

        detected = _find_components(grid, bg)
        if len(detected) != n_objects:
            continue

        boundaries = [int(obj["boundary"]) for obj in detected]
        if len(set(boundaries)) != n_objects:
            continue

        areas = [int(obj["area"]) for obj in detected]
        if sorted(range(n_objects), key=lambda i: (-areas[i], -boundaries[i])) == sorted(
            range(n_objects),
            key=lambda i: (-boundaries[i], -areas[i]),
        ):
            continue

        expected = _render_output(
            bg,
            sorted(detected, key=lambda obj: -int(obj["boundary"])),
        )
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue

        return grid

    raise RuntimeError("failed to generate a valid ARC sample")
