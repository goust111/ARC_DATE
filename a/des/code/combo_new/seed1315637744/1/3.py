from __future__ import annotations

from collections import deque
import random

import numpy as np


BACKGROUND = 0
GEN_FRAME_COLOR = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def _parse_mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "#" for ch in row] for row in rows], dtype=bool)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(cells, dtype=int))
    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _component_mask(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox(coords)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0], dtype=int)
    out[rel[:, 0], rel[:, 1]] = True
    return out


def _hole_pixel_count(mask: np.ndarray) -> int:
    h, w = mask.shape
    outside = np.zeros((h, w), dtype=bool)
    queue: deque[tuple[int, int]] = deque()
    for r in range(h):
        for c in (0, w - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    for c in range(w):
        for r in (0, h - 1):
            if not mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                queue.append((r, c))
    while queue:
        r, c = queue.popleft()
        for nr, nc in _neighbors4(r, c, h, w):
            if not mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                queue.append((nr, nc))
    holes = (~mask) & (~outside)
    return int(holes.sum())


def _unique_transforms(mask: np.ndarray) -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    seen: set[tuple[tuple[int, int], tuple[bool, ...]]] = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for candidate in (rotated, np.fliplr(rotated)):
            trimmed = _trim_mask(candidate)
            key = (trimmed.shape, tuple(bool(v) for v in trimmed.ravel()))
            if key not in seen:
                seen.add(key)
                variants.append(trimmed.copy())
    return variants


def _build_template_library() -> dict[int, list[np.ndarray]]:
    raw_templates = {
        0: [
            ["##", "##"],
            ["###", ".##"],
            [".#.", "###", ".#."],
            ["###", "..#", "..#"],
        ],
        1: [
            ["###", "#.#", "###"],
            ["###", "#.#", "###", "..#"],
        ],
        2: [
            ["####", "#..#", "####"],
            ["####", "#..#", "####", "..#."],
            [".#..", "####", "#..#", "####"],
        ],
        3: [
            ["#####", "#...#", "#####"],
            ["#####", "#...#", "#####", "..#.."],
        ],
        4: [
            ["####", "#..#", "#..#", "####"],
            ["####", "#..#", "#..#", "####", "..#."],
        ],
        5: [
            ["#####", "#.#.#", "#...#", "#####"],
        ],
        6: [
            ["#####", "#...#", "#...#", "#####"],
            ["#####", "#...#", "#...#", "#####", "..#.."],
        ],
    }

    library: dict[int, list[np.ndarray]] = {}
    for hole_count, patterns in raw_templates.items():
        variants: list[np.ndarray] = []
        seen: set[tuple[tuple[int, int], tuple[bool, ...]]] = set()
        for rows in patterns:
            base = _parse_mask(rows)
            for variant in _unique_transforms(base):
                key = (variant.shape, tuple(bool(v) for v in variant.ravel()))
                if key in seen:
                    continue
                if len(_mask_components(variant)) != 1:
                    continue
                if int(variant.sum()) > 15:
                    continue
                if _hole_pixel_count(variant) != hole_count:
                    continue
                seen.add(key)
                variants.append(variant.copy())
        assert variants, f"missing template variants for hole count {hole_count}"
        library[hole_count] = variants
    return library


TEMPLATE_LIBRARY = _build_template_library()


def _is_rectangular_frame(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    perimeter = 2 * h + 2 * w - 4
    if int(mask.sum()) != perimeter:
        return False
    if not np.all(mask[0, :]) or not np.all(mask[-1, :]):
        return False
    if not np.all(mask[:, 0]) or not np.all(mask[:, -1]):
        return False
    return not np.any(mask[1:-1, 1:-1])


def _find_palette(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[int]]:
    candidates: list[tuple[tuple[int, int, int, int], list[int]]] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != BACKGROUND):
        for coords in _mask_components(grid == color):
            mask = _component_mask(coords)
            if not _is_rectangular_frame(mask):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.shape[0] < 1 or interior.shape[1] < 2:
                continue
            if np.any(interior == BACKGROUND):
                continue
            if not np.all(interior == interior[0:1, :]):
                continue
            palette_colors = [int(v) for v in interior[0]]
            candidates.append(((r0, c0, r1, c1), palette_colors))
    assert len(candidates) == 1, "expected one framed palette"
    return candidates[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each lower template by using its enclosed black hole-pixel count as a palette index.

    Input structure:
    - The background is black.
    - The upper part contains one grey rectangular frame around a horizontal palette of 1-column color stripes.
    - The lower part contains several disjoint small 4-connected monochrome template objects that may enclose black hole pixels.

    Rule:
    - Keep the framed palette unchanged.
    - For each lower template, count the black pixels inside its bbox that are enclosed by the template and do not connect to the bbox edge.
    - Use that count N as a zero-based left-to-right index into the palette stripes, and recolor every template foreground pixel to the selected stripe color.
    - Leave enclosed hole pixels black.

    Key insight:
    The template outline is only a stencil; the decisive signal is the number of enclosed black hole pixels, which points to a palette position rather than to a color identity already present in the template.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    (pr0, pc0, pr1, pc1), palette_colors = _find_palette(g)
    remaining = g != BACKGROUND
    remaining[pr0 : pr1 + 1, pc0 : pc1 + 1] = False

    templates = _mask_components(remaining)
    assert templates, "expected at least one lower template"
    for coords in templates:
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1
        mask = _component_mask(coords)
        holes = _hole_pixel_count(mask)
        assert 0 <= holes < len(palette_colors)
        target_color = palette_colors[holes]
        out[coords[:, 0], coords[:, 1]] = target_color
    return out


def _expanded_box(
    r0: int,
    c0: int,
    h: int,
    w: int,
    pad: int,
    grid_h: int,
    grid_w: int,
) -> tuple[int, int, int, int]:
    return (
        max(0, r0 - pad),
        max(0, c0 - pad),
        min(grid_h, r0 + h + pad),
        min(grid_w, c0 + w + pad),
    )


def _paint_palette(
    grid: np.ndarray,
    top: int,
    left: int,
    stripe_h: int,
    colors: list[int],
    frame_color: int,
) -> None:
    palette_h = stripe_h + 2
    palette_w = len(colors) + 2
    grid[top, left : left + palette_w] = frame_color
    grid[top + palette_h - 1, left : left + palette_w] = frame_color
    grid[top : top + palette_h, left] = frame_color
    grid[top : top + palette_h, left + palette_w - 1] = frame_color
    for idx, color in enumerate(colors):
        grid[top + 1 : top + 1 + stripe_h, left + 1 + idx] = color


def _paint_template(
    grid: np.ndarray,
    top: int,
    left: int,
    mask: np.ndarray,
    color: int,
) -> None:
    h, w = mask.shape
    patch = grid[top : top + h, left : left + w]
    patch[mask] = color


def _choose_hole_counts(palette_len: int, template_count: int) -> list[int]:
    counts = [0, random.randint(1, palette_len - 1)]
    unique_target = random.randint(2, min(template_count, palette_len))
    while len(set(counts)) < unique_target:
        counts.append(random.randint(0, palette_len - 1))
    while len(counts) < template_count:
        counts.append(random.choice(counts + list(range(palette_len))))
    random.shuffle(counts)
    return counts[:template_count]


def _place_templates(
    grid: np.ndarray,
    palette_bbox: tuple[int, int, int, int],
    templates: list[tuple[np.ndarray, int, int]],
) -> list[tuple[int, int]] | None:
    h, w = grid.shape
    blocked = np.zeros((h, w), dtype=bool)
    pr0, pc0, pr1, pc1 = palette_bbox
    er0, ec0, er1, ec1 = _expanded_box(pr0, pc0, pr1 - pr0 + 1, pc1 - pc0 + 1, 1, h, w)
    blocked[er0:er1, ec0:ec1] = True

    lower_top = max(h // 2, pr1 + 3)
    order = sorted(
        range(len(templates)),
        key=lambda idx: templates[idx][0].shape[0] * templates[idx][0].shape[1],
        reverse=True,
    )

    placements: list[tuple[int, int] | None] = [None] * len(templates)
    for idx in order:
        mask, _holes, _color = templates[idx]
        th, tw = mask.shape
        if lower_top + th > h:
            return None
        positions = [
            (r, c)
            for r in range(lower_top, h - th + 1)
            for c in range(0, w - tw + 1)
        ]
        random.shuffle(positions)
        placed = False
        for r0, c0 in positions:
            br0, bc0, br1, bc1 = _expanded_box(r0, c0, th, tw, 1, h, w)
            if np.any(blocked[br0:br1, bc0:bc1]):
                continue
            placements[idx] = (r0, c0)
            blocked[br0:br1, bc0:bc1] = True
            placed = True
            break
        if not placed:
            return None

    for (mask, _holes, color), pos in zip(templates, placements):
        assert pos is not None
        r0, c0 = pos
        _paint_template(grid, r0, c0, mask, color)
    return [(int(r0), int(c0)) for r0, c0 in placements if r0 is not None and c0 is not None]


def _gen_attempt() -> np.ndarray | None:
    palette_len = random.randint(4, 7)
    palette_colors = random.sample(COLOR_POOL, palette_len)
    source_pool = [color for color in COLOR_POOL if color not in palette_colors]
    source_colors = random.sample(source_pool, random.randint(1, min(3, len(source_pool))))

    template_count = random.randint(3, 6)
    hole_counts = _choose_hole_counts(palette_len, template_count)
    template_specs: list[tuple[np.ndarray, int, int]] = []
    for holes in hole_counts:
        mask = random.choice(TEMPLATE_LIBRARY[holes]).copy()
        color = random.choice(source_colors)
        template_specs.append((mask, holes, color))

    stripe_h = random.randint(2, 5)
    palette_h = stripe_h + 2
    palette_w = palette_len + 2
    max_template_h = max(mask.shape[0] for mask, _holes, _color in template_specs)
    max_template_w = max(mask.shape[1] for mask, _holes, _color in template_specs)

    grid_h = random.randint(max(14, palette_h + max_template_h + 6), 30)
    grid_w = random.randint(max(12, palette_w + 2, max_template_w + 2), 30)
    palette_top = random.randint(0, min(3, max(0, grid_h // 4)))
    palette_left = random.randint(0, grid_w - palette_w)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    _paint_palette(grid, palette_top, palette_left, stripe_h, palette_colors, GEN_FRAME_COLOR)
    palette_bbox = (
        palette_top,
        palette_left,
        palette_top + palette_h - 1,
        palette_left + palette_w - 1,
    )

    placements = _place_templates(grid, palette_bbox, template_specs)
    if placements is None:
        return None

    expected = grid.copy()
    for (mask, holes, _source_color), (r0, c0) in zip(template_specs, placements):
        patch = expected[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]]
        patch[mask] = palette_colors[holes]

    non_palette = grid != BACKGROUND
    pr0, pc0, pr1, pc1 = palette_bbox
    non_palette[pr0 : pr1 + 1, pc0 : pc1 + 1] = False
    if len(_mask_components(non_palette)) != template_count:
        return None

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
