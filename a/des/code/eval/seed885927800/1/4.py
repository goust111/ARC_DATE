from __future__ import annotations

import random
from collections import deque

import numpy as np


BG = 0
DIVIDER = 5
MAX_GENERATION_ATTEMPTS = 400
NEIGHBORS8 = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _mask(rows: list[str]) -> np.ndarray:
    return np.array([[ch == "1" for ch in row] for row in rows], dtype=bool)


def _classify_mask(mask: np.ndarray) -> str:
    vertical = np.array_equal(mask, np.fliplr(mask))
    horizontal = np.array_equal(mask, np.flipud(mask))
    if vertical and horizontal:
        return "both"
    if vertical:
        return "vertical"
    if horizontal:
        return "horizontal"
    return "none"


VERTICAL_SHAPES = [
    _mask(["111", "010"]),
    _mask(["111", "010", "010"]),
    _mask(["101", "111"]),
    _mask(["010", "111", "111"]),
    _mask(["11111", "00100"]),
]

HORIZONTAL_SHAPES = [shape.T for shape in VERTICAL_SHAPES]

BOTH_SHAPES = [
    _mask(["11", "11"]),
    _mask(["010", "111", "010"]),
    _mask(["111"]),
    _mask(["111", "111"]),
    _mask(["010", "111", "111", "010"]),
    _mask(["0110", "1111", "0110"]),
]

NONE_SHAPES = [
    _mask(["10", "10", "11"]),
    _mask(["110", "011"]),
    _mask(["111", "100"]),
    _mask(["10", "11", "01"]),
    _mask(["01", "11", "10"]),
]

ICON_LIBRARY = {
    "vertical": [
        _mask(["111", "010"]),
        _mask(["101", "111", "010"]),
    ],
    "horizontal": [
        _mask(["11", "10", "11"]),
        _mask(["10", "11", "10"]),
    ],
    "both": [
        _mask(["11", "11"]),
        _mask(["010", "111", "010"]),
    ],
}


def _validate_libraries() -> None:
    for shape in VERTICAL_SHAPES:
        assert _classify_mask(shape) == "vertical"
        assert 1 < int(shape.sum()) <= 8
    for shape in HORIZONTAL_SHAPES:
        assert _classify_mask(shape) == "horizontal"
        assert 1 < int(shape.sum()) <= 8
    for shape in BOTH_SHAPES:
        assert _classify_mask(shape) == "both"
        assert 1 < int(shape.sum()) <= 8
    for shape in NONE_SHAPES:
        assert _classify_mask(shape) == "none"
        assert 1 < int(shape.sum()) <= 8
    for kind, shapes in ICON_LIBRARY.items():
        for shape in shapes:
            assert _classify_mask(shape) == kind


_validate_libraries()


def _find_components(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    components: list[dict[str, object]] = []

    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or visited[r, c]:
                continue
            visited[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            cells: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                cells.append((rr, cc))
                for dr, dc in NEIGHBORS8:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        q.append((nr, nc))

            coords = np.array(cells, dtype=int)
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (r0, c0, r1, c1),
                }
            )
    return components


def _component_mask(component: dict[str, object]) -> np.ndarray:
    coords = np.asarray(component["coords"], dtype=int)
    r0, c0, r1, c1 = map(int, component["bbox"])
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask


def _component_sprite(component: dict[str, object], bg: int) -> np.ndarray:
    mask = _component_mask(component)
    color = int(component["color"])
    sprite = np.full(mask.shape, bg, dtype=int)
    sprite[mask] = color
    return sprite


def _pack_sprites(sprites: list[np.ndarray], width: int, bg: int) -> np.ndarray:
    if not sprites:
        return np.full((1, width), bg, dtype=int)

    placements: list[tuple[np.ndarray, int, int]] = []
    x = 0
    y = 0
    row_height = 0
    total_height = 0

    for sprite in sprites:
        sh, sw = sprite.shape
        assert sw <= width
        if x > 0 and x + sw > width:
            y += row_height + 1
            x = 0
            row_height = 0
        placements.append((sprite, y, x))
        x += sw + 1
        row_height = max(row_height, sh)
        total_height = max(total_height, y + sh)

    canvas = np.full((total_height, width), bg, dtype=int)
    for sprite, top, left in placements:
        sh, sw = sprite.shape
        patch = canvas[top : top + sh, left : left + sw]
        mask = sprite != bg
        patch[mask] = sprite[mask]
    return canvas


def _find_marker_layout(grid: np.ndarray, bg: int) -> tuple[int, int, int, tuple[int, int]]:
    h, w = grid.shape
    bottom_cells = [(c, int(grid[h - 1, c])) for c in range(w) if int(grid[h - 1, c]) != bg]
    assert len(bottom_cells) == 2
    divider_cols = tuple(sorted(int(c) for c, _color in bottom_cells))
    divider_color = int(bottom_cells[0][1])
    assert all(int(color) == divider_color for _c, color in bottom_cells)

    left_divider, right_divider = divider_cols
    region_w = left_divider
    assert right_divider - left_divider - 1 == region_w
    assert w - right_divider - 1 == region_w

    band_top = h - 1
    while (
        band_top > 0
        and int(grid[band_top - 1, left_divider]) == divider_color
        and int(grid[band_top - 1, right_divider]) == divider_color
    ):
        band_top -= 1
    assert band_top < h - 1
    return band_top, region_w, divider_color, divider_cols


def _region_bounds(region_w: int) -> list[tuple[int, int]]:
    return [
        (0, region_w),
        (region_w + 1, 2 * region_w + 1),
        (2 * region_w + 2, 3 * region_w + 2),
    ]


def _category_icons(marker_band: np.ndarray, bg: int, divider_color: int, region_w: int) -> list[str]:
    categories: list[str] = []
    for c0, c1 in _region_bounds(region_w):
        region = marker_band[:, c0:c1].copy()
        region[region == divider_color] = bg
        components = _find_components(region, bg)
        assert len(components) == 1
        category = _classify_mask(_component_mask(components[0]))
        assert category != "none"
        categories.append(category)
    assert len(set(categories)) == 3
    return categories


def _render_output_from_groups(
    groups: dict[str, list[np.ndarray]],
    partition_categories: list[str],
    region_w: int,
    bg: int,
    divider_color: int,
) -> np.ndarray:
    partitions = [_pack_sprites(groups[category], region_w, bg) for category in partition_categories]
    out_h = max(part.shape[0] for part in partitions)
    out_w = 3 * region_w + 2
    out = np.full((out_h, out_w), bg, dtype=int)
    out[:, region_w] = divider_color
    out[:, 2 * region_w + 1] = divider_color

    for idx, part in enumerate(partitions):
        start = idx * (region_w + 1)
        if idx == 2:
            start -= 0
        patch = out[: part.shape[0], start : start + region_w]
        mask = part != bg
        patch[mask] = part[mask]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Classify scattered monochrome shapes by the symmetry type indicated by the bottom icons.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0

    colors, counts = np.unique(g, return_counts=True)
    bg = int(colors[np.argmax(counts)])
    band_top, region_w, divider_color, _divider_cols = _find_marker_layout(g, bg)
    marker_band = g[band_top:, :]
    scene = g[:band_top, :]

    partition_categories = _category_icons(marker_band, bg, divider_color, region_w)
    groups: dict[str, list[np.ndarray]] = {"vertical": [], "horizontal": [], "both": []}

    components = _find_components(scene, bg)
    components.sort(key=lambda comp: (int(comp["bbox"][0]), int(comp["bbox"][1])))
    for component in components:
        category = _classify_mask(_component_mask(component))
        if category == "none":
            continue
        groups[category].append(_component_sprite(component, bg))

    return _render_output_from_groups(groups, partition_categories, region_w, bg, divider_color)


def _sample_shape(kind: str) -> np.ndarray:
    if kind == "vertical":
        mask = random.choice(VERTICAL_SHAPES).copy()
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    if kind == "horizontal":
        mask = random.choice(HORIZONTAL_SHAPES).copy()
        if random.random() < 0.5:
            mask = np.fliplr(mask)
        return mask
    if kind == "both":
        mask = random.choice(BOTH_SHAPES).copy()
        k = random.randrange(4)
        if k:
            mask = np.rot90(mask, k)
        if random.random() < 0.5:
            mask = np.flipud(mask)
        return mask
    mask = random.choice(NONE_SHAPES).copy()
    k = random.randrange(4)
    if k:
        mask = np.rot90(mask, k)
    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _sample_icon(kind: str) -> np.ndarray:
    return random.choice(ICON_LIBRARY[kind]).copy()


def _mark_neighbors(blocked: np.ndarray, mask: np.ndarray, top: int, left: int) -> None:
    h, w = blocked.shape
    rows, cols = np.where(mask)
    for rr, cc in zip(rows + top, cols + left):
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = int(rr + dr)
                nc = int(cc + dc)
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_objects(
    scene_h: int, width: int, objects: list[dict[str, object]]
) -> tuple[np.ndarray, list[dict[str, object]]] | None:
    scene = np.full((scene_h, width), BG, dtype=int)
    blocked = np.zeros((scene_h, width), dtype=bool)
    placements: list[dict[str, object]] = []

    order = sorted(
        objects,
        key=lambda item: (-(int(np.asarray(item["mask"], dtype=bool).shape[0]) * int(np.asarray(item["mask"], dtype=bool).shape[1]))),
    )

    for item in order:
        mask = np.asarray(item["mask"], dtype=bool)
        color = int(item["color"])
        sh, sw = mask.shape
        positions = [
            (r, c)
            for r in range(scene_h - sh + 1)
            for c in range(width - sw + 1)
        ]
        random.shuffle(positions)
        placed = False
        for top, left in positions:
            rows, cols = np.where(mask)
            if np.any(blocked[rows + top, cols + left]):
                continue
            scene[top + rows, left + cols] = color
            _mark_neighbors(blocked, mask, top, left)
            placements.append(
                {
                    "kind": item["kind"],
                    "color": color,
                    "mask": mask,
                    "top": top,
                    "left": left,
                }
            )
            placed = True
            break
        if not placed:
            return None

    return scene, placements


def _render_marker_band(region_w: int, icon_colors: list[int]) -> np.ndarray:
    band_h = 5
    width = 3 * region_w + 2
    band = np.full((band_h, width), BG, dtype=int)
    band[:, region_w] = DIVIDER
    band[:, 2 * region_w + 1] = DIVIDER

    for idx, category in enumerate(("vertical", "horizontal", "both")):
        icon = _sample_icon(category)
        ih, iw = icon.shape
        c0, c1 = _region_bounds(region_w)[idx]
        top = 1 + random.randrange(band_h - ih - 1)
        left = c0 + (region_w - iw) // 2
        band[top : top + ih, left : left + iw][icon] = int(icon_colors[idx])
    return band


def _expected_output(region_w: int, placements: list[dict[str, object]]) -> np.ndarray:
    groups: dict[str, list[np.ndarray]] = {"vertical": [], "horizontal": [], "both": []}
    ordered = sorted(placements, key=lambda item: (int(item["top"]), int(item["left"])))
    for item in ordered:
        kind = str(item["kind"])
        if kind == "none":
            continue
        mask = np.asarray(item["mask"], dtype=bool)
        sprite = np.full(mask.shape, BG, dtype=int)
        sprite[mask] = int(item["color"])
        groups[kind].append(sprite)
    return _render_output_from_groups(groups, ["vertical", "horizontal", "both"], region_w, BG, DIVIDER)


def _sample_object_specs() -> list[dict[str, object]]:
    palette = [color for color in range(10) if color not in {BG, DIVIDER}]
    objects: list[dict[str, object]] = []
    for kind in ("vertical", "horizontal", "both"):
        for _ in range(random.randint(1, 4)):
            objects.append(
                {
                    "kind": kind,
                    "mask": _sample_shape(kind),
                    "color": random.choice(palette),
                }
            )
    for _ in range(random.randint(1, 3)):
        objects.append(
            {
                "kind": "none",
                "mask": _sample_shape("none"),
                "color": random.choice(palette),
            }
        )
    return objects


def _gen_attempt() -> np.ndarray | None:
    region_w = random.randint(7, 10)
    width = 3 * region_w + 2
    scene_h = random.randint(10, 16)

    objects = _sample_object_specs()
    placed = _place_objects(scene_h, width, objects)
    if placed is None:
        return None
    scene, placements = placed

    icon_palette = random.sample([color for color in range(10) if color not in {BG, DIVIDER}], 3)
    marker_band = _render_marker_band(region_w, icon_palette)
    spacer = np.full((1, width), BG, dtype=int)
    grid = np.vstack([scene, spacer, marker_band])

    expected = _expected_output(region_w, placements)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
