from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
COLORS = tuple(range(10))
RNG = random.Random()
GEN_ATTEMPTS = 300


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _interior_from_boundary(boundary: np.ndarray) -> np.ndarray:
    h, w = boundary.shape
    outside = np.zeros_like(boundary, dtype=bool)
    q: deque[tuple[int, int]] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if not boundary[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not boundary[r, c] and not outside[r, c]:
                outside[r, c] = True
                q.append((r, c))

    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and not boundary[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                q.append((nr, nc))

    return (~boundary) & (~outside)


def _extract_palette(grid: np.ndarray) -> tuple[list[int], np.ndarray]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, np.ndarray]] = []
    for col in (0, w - 1):
        column = grid[:, col]
        rows = np.flatnonzero(column != BG)
        if rows.size == 0:
            continue
        prefix_len = 0
        while prefix_len < h and int(column[prefix_len]) != BG:
            prefix_len += 1
        prefix_ok = int(np.all(column[prefix_len:] == BG))
        distinct = len({int(v) for v in column if int(v) != BG})
        candidates.append((prefix_ok, distinct, int(rows.size), column.copy()))

    assert candidates, "expected a palette column on the left or right edge"
    candidates.sort(reverse=True)
    column = candidates[0][3]
    palette = [int(v) for v in column if int(v) != BG]
    assert palette, "palette must be non-empty"
    if np.array_equal(column, grid[:, 0]):
        return palette, grid[:, 1:].copy()
    return palette, grid[:, :-1].copy()


def _solve(content: np.ndarray, palette: list[int], border_color: int) -> np.ndarray:
    out = content.copy()
    for comp in _components(content == border_color):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)

        local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in comp:
            local[r - r0, c - c0] = True

        perimeter = len(comp)
        assert perimeter >= 4 and (perimeter - 4) % 2 == 0, "invalid perimeter for palette lookup"
        palette_idx = (perimeter - 4) // 2
        assert palette_idx < len(palette), "perimeter exceeds palette range"

        interior = _interior_from_boundary(local)
        if interior.any():
            patch = out[r0 : r1 + 1, c0 : c1 + 1]
            patch[interior & (patch == BG)] = palette[palette_idx]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hollow outline with the palette color indexed by its perimeter length.

    Input structure:
    - The background is black.
    - Several simple closed outlines of one shared border color appear in the main field; each outline encloses black interior cells.
    - A single vertical palette column sits on the left or right edge; its top-to-bottom entries encode colors for perimeters 4, 6, 8, ... in order.

    Rule:
    - Remove the palette column from the output.
    - Count the number of border-color pixels in each closed outline to get its perimeter P.
    - Look up the fill color at palette index (P - 4) / 2, keep the outline unchanged, and fill the enclosed black interior with that color.

    Key insight:
    - The palette is keyed by boundary length, not by area or by the outline color, so the solver must count contour pixels and use their ordered palette position.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    palette, content = _extract_palette(g)
    non_bg = [int(v) for v in np.unique(content) if int(v) != BG]
    assert non_bg, "expected at least one outline color"

    if len(non_bg) == 1:
        border_color = non_bg[0]
    else:
        border_color = max(non_bg, key=lambda color: int((content == color).sum()))

    return _solve(content, palette, border_color)


def _trace_vertices(vertices: list[tuple[int, int]]) -> np.ndarray:
    cells: set[tuple[int, int]] = set()
    for i, (r0, c0) in enumerate(vertices):
        r1, c1 = vertices[(i + 1) % len(vertices)]
        dr = (r1 > r0) - (r1 < r0)
        dc = (c1 > c0) - (c1 < c0)
        assert dr == 0 or dc == 0, "segments must be axis-aligned"
        length = abs(r1 - r0) + abs(c1 - c0)
        for step in range(length + 1):
            cells.add((r0 + dr * step, c0 + dc * step))

    h = max(r for r, _ in cells) + 1
    w = max(c for _, c in cells) + 1
    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _shape_from_mask(mask: np.ndarray) -> dict[str, np.ndarray | int] | None:
    interior = _interior_from_boundary(mask)
    if not interior.any():
        return None
    perimeter = int(mask.sum())
    if (perimeter - 4) % 2 != 0:
        return None
    return {
        "mask": mask,
        "interior": interior,
        "height": int(mask.shape[0]),
        "width": int(mask.shape[1]),
        "perimeter": perimeter,
    }


def _make_rectangle(max_perimeter: int, max_h: int, max_w: int) -> dict[str, np.ndarray | int] | None:
    if max_h < 3 or max_w < 3:
        return None
    for _ in range(40):
        h = RNG.randint(3, max_h)
        w = RNG.randint(3, max_w)
        perimeter = 2 * h + 2 * w - 4
        if perimeter > max_perimeter:
            continue
        mask = np.zeros((h, w), dtype=bool)
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        return _shape_from_mask(mask)
    return None


def _make_notched(max_perimeter: int, max_h: int, max_w: int) -> dict[str, np.ndarray | int] | None:
    for _ in range(60):
        side = RNG.choice(("top", "bottom", "left", "right"))
        if side in ("top", "bottom"):
            if max_h < 5 or max_w < 6:
                continue
            h = RNG.randint(5, max_h)
            w = RNG.randint(6, max_w)
            depth = RNG.randint(1, h - 3)
            start = RNG.randint(1, w - 4)
            end = RNG.randint(start + 2, w - 2)
            if side == "top":
                vertices = [
                    (0, 0),
                    (0, start),
                    (depth, start),
                    (depth, end),
                    (0, end),
                    (0, w - 1),
                    (h - 1, w - 1),
                    (h - 1, 0),
                ]
            else:
                vertices = [
                    (0, 0),
                    (0, w - 1),
                    (h - 1, w - 1),
                    (h - 1, end),
                    (h - 1 - depth, end),
                    (h - 1 - depth, start),
                    (h - 1, start),
                    (h - 1, 0),
                ]
        else:
            if max_h < 6 or max_w < 5:
                continue
            h = RNG.randint(6, max_h)
            w = RNG.randint(5, max_w)
            depth = RNG.randint(1, w - 3)
            start = RNG.randint(1, h - 4)
            end = RNG.randint(start + 2, h - 2)
            if side == "left":
                vertices = [
                    (0, 0),
                    (0, w - 1),
                    (h - 1, w - 1),
                    (h - 1, 0),
                    (end, 0),
                    (end, depth),
                    (start, depth),
                    (start, 0),
                ]
            else:
                vertices = [
                    (0, 0),
                    (0, w - 1),
                    (start, w - 1),
                    (start, w - 1 - depth),
                    (end, w - 1 - depth),
                    (end, w - 1),
                    (h - 1, w - 1),
                    (h - 1, 0),
                ]

        mask = _trace_vertices(vertices)
        shape = _shape_from_mask(mask)
        if shape is not None and int(shape["perimeter"]) <= max_perimeter:
            return shape
    return None


def _make_shape(max_perimeter: int, max_h: int, max_w: int) -> dict[str, np.ndarray | int] | None:
    builders = [_make_rectangle, _make_notched]
    RNG.shuffle(builders)
    for builder in builders:
        shape = builder(max_perimeter, max_h, max_w)
        if shape is not None:
            return shape
    return None


def _palette_sequence(border_color: int, length: int) -> list[int]:
    colors = [c for c in COLORS if c not in (BG, border_color)]
    RNG.shuffle(colors)
    palette = [colors[i % len(colors)] for i in range(length)]
    if length >= 2 and palette[0] == palette[1]:
        palette[1] = colors[(colors.index(palette[1]) + 1) % len(colors)]
    return palette


def _place_shapes(
    height: int,
    width: int,
    n_objects: int,
) -> list[dict[str, np.ndarray | int]] | None:
    max_perimeter = 2 * height + 2
    max_h = min(10, height - 2)
    max_w = min(10, width - 2)
    reserved = np.zeros((height, width), dtype=bool)
    placed: list[dict[str, np.ndarray | int]] = []

    for _ in range(n_objects):
        shape = _make_shape(max_perimeter=max_perimeter, max_h=max_h, max_w=max_w)
        if shape is None:
            return None

        sh = int(shape["height"])
        sw = int(shape["width"])
        if sh > height - 2 or sw > width - 2:
            return None

        options: list[tuple[int, int]] = []
        for top in range(1, height - sh):
            for left in range(1, width - sw):
                r0 = max(0, top - 1)
                r1 = min(height, top + sh + 1)
                c0 = max(0, left - 1)
                c1 = min(width, left + sw + 1)
                if not reserved[r0:r1, c0:c1].any():
                    options.append((top, left))
        if not options:
            return None

        top, left = RNG.choice(options)
        reserved[top : top + sh, left : left + sw] = True
        placed.append(
            {
                **shape,
                "top": top,
                "left": left,
            }
        )

    perimeters = {int(item["perimeter"]) for item in placed}
    if len(placed) > 1 and len(perimeters) < 2:
        return None
    return placed


def _gen_attempt() -> np.ndarray | None:
    height = RNG.randint(12, 24)
    content_width = RNG.randint(13, 29)
    if content_width + 1 > 30:
        content_width = 29

    n_objects = RNG.randint(2, 4 if height * content_width < 420 else 5)
    placed = _place_shapes(height, content_width, n_objects)
    if placed is None:
        return None

    border_color = RNG.choice([c for c in COLORS if c != BG])
    palette_len_needed = max((int(item["perimeter"]) - 4) // 2 + 1 for item in placed)
    if palette_len_needed > height:
        return None
    extra = RNG.randint(0, min(3, height - palette_len_needed))
    palette = _palette_sequence(border_color, palette_len_needed + extra)
    palette_len = len(palette)

    content = np.full((height, content_width), BG, dtype=int)
    expected = content.copy()
    for item in placed:
        top = int(item["top"])
        left = int(item["left"])
        mask = np.asarray(item["mask"], dtype=bool)
        interior = np.asarray(item["interior"], dtype=bool)
        h, w = mask.shape
        patch = content[top : top + h, left : left + w]
        patch[mask] = border_color

        fill_color = palette[(int(item["perimeter"]) - 4) // 2]
        expected_patch = expected[top : top + h, left : left + w]
        expected_patch[mask] = border_color
        expected_patch[interior] = fill_color

    if not np.any(expected != content):
        return None

    side = RNG.choice(("left", "right"))
    full = np.full((height, content_width + 1), BG, dtype=int)
    if side == "left":
        full[:, 1:] = content
        full[:palette_len, 0] = palette
    else:
        full[:, :-1] = content
        full[:palette_len, -1] = palette

    try:
        produced = main(full)
    except Exception:
        return None
    if produced.shape != expected.shape or not np.array_equal(produced, expected):
        return None
    return full


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GEN_ATTEMPTS} attempts")
