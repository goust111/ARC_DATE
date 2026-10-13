from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
TRANSFORMS = ("hflip", "vflip", "rot180")


def _neighbors(r: int, c: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]
    return [
        (r + dr, c + dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if dr != 0 or dc != 0
    ]


def _components(mask: np.ndarray, connectivity: int) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
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
                for nr, nc in _neighbors(rr, cc, connectivity):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _normalize(coords: list[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    r0, c0, _, _ = _bbox(coords)
    return tuple(sorted((r - r0, c - c0) for r, c in coords))


def _transform_local_coords(
    coords: list[tuple[int, int]], height: int, width: int, kind: str
) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for r, c in coords:
        if kind == "hflip":
            out.append((r, width - 1 - c))
        elif kind == "vflip":
            out.append((height - 1 - r, c))
        elif kind == "rot180":
            out.append((height - 1 - r, width - 1 - c))
        else:
            raise ValueError(f"unknown transform {kind}")
    return out


def _decode_transform(source: tuple[int, int], target: tuple[int, int]) -> str:
    dr = target[0] - source[0]
    dc = target[1] - source[1]
    if dr == 0 and abs(dc) == 1:
        return "hflip"
    if dc == 0 and abs(dr) == 1:
        return "vflip"
    if abs(dr) == 1 and abs(dc) == 1:
        return "rot180"
    raise AssertionError("legend pair must be 8-neighbor adjacent")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each legend pixel pair as a transform instruction and recolor the matching sprite.

    Input structure:
    - A full gray row and full gray column split the black background grid into four quadrants.
    - Three quadrants each contain one small single-color 4-connected sprite.
    - The remaining quadrant is a legend with three isolated 8-neighbor pixel pairs; each pair contains one sprite color and one new target color.

    Rule:
    - Use the gray cross to extract the four quadrants and identify the legend as the only quadrant with multiple foreground colors.
    - In each legend pair, the pixel whose color matches a sprite selects that sprite, the other pixel gives the recolor target, and their relative position selects the transform.
    - Horizontal adjacency means horizontal mirror, vertical adjacency means vertical mirror, and diagonal adjacency means 180-degree rotation, all applied inside the sprite's own bounding box.
    - Clear the legend and original sprite pixels, keep the gray separators, and draw the transformed sprite in the target color.

    Key insight:
    - The legend's geometry is itself the instruction code: color identity chooses which sprite to edit, while pair orientation chooses how to transform it.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    sep_rows = np.where(np.all(g == GREY, axis=1))[0]
    sep_cols = np.where(np.all(g == GREY, axis=0))[0]
    assert len(sep_rows) == 1 and len(sep_cols) == 1
    sep_r = int(sep_rows[0])
    sep_c = int(sep_cols[0])

    quad_specs = [
        ((0, 0), g[:sep_r, :sep_c]),
        ((0, sep_c + 1), g[:sep_r, sep_c + 1 :]),
        ((sep_r + 1, 0), g[sep_r + 1 :, :sep_c]),
        ((sep_r + 1, sep_c + 1), g[sep_r + 1 :, sep_c + 1 :]),
    ]

    sprites: dict[int, dict[str, object]] = {}
    legend_sub: np.ndarray | None = None
    legend_origin: tuple[int, int] | None = None
    legend_count = 0

    for origin, sub in quad_specs:
        colors = [int(c) for c in np.unique(sub) if int(c) not in (BLACK, GREY)]
        assert colors, "each quadrant must contain foreground content"
        if len(colors) == 1:
            color = colors[0]
            comps = _components(sub == color, 4)
            assert len(comps) == 1
            coords = comps[0]
            assert 3 <= len(coords) <= 6
            r0, c0, r1, c1 = _bbox(coords)
            height = r1 - r0 + 1
            width = c1 - c0 + 1
            local = [(r - r0, c - c0) for r, c in coords]
            sprites[color] = {
                "origin": origin,
                "bbox": (r0, c0, r1, c1),
                "local": local,
                "height": height,
                "width": width,
            }
        else:
            legend_sub = sub
            legend_origin = origin
            legend_count += 1

    assert legend_count == 1 and legend_sub is not None and legend_origin is not None
    assert len(sprites) == 3

    source_colors = set(sprites)
    legend_comps = _components(legend_sub != BLACK, 8)
    assert len(legend_comps) == 3

    instructions: dict[int, tuple[str, int]] = {}
    seen_targets: set[int] = set()
    for comp in legend_comps:
        assert len(comp) == 2
        (r1, c1), (r2, c2) = comp
        color1 = int(legend_sub[r1, c1])
        color2 = int(legend_sub[r2, c2])
        src1 = color1 in source_colors
        src2 = color2 in source_colors
        assert src1 ^ src2
        if src1:
            source_color = color1
            source_pos = (r1, c1)
            target_color = color2
            target_pos = (r2, c2)
        else:
            source_color = color2
            source_pos = (r2, c2)
            target_color = color1
            target_pos = (r1, c1)
        assert target_color not in source_colors
        assert target_color not in seen_targets
        seen_targets.add(target_color)
        instructions[source_color] = (_decode_transform(source_pos, target_pos), target_color)

    assert set(instructions) == source_colors

    out = np.where(g == GREY, GREY, BLACK)
    for source_color, sprite in sprites.items():
        kind, target_color = instructions[source_color]
        r0, c0, r1, c1 = sprite["bbox"]  # type: ignore[assignment]
        base_r, base_c = sprite["origin"]  # type: ignore[assignment]
        height = int(sprite["height"])  # type: ignore[arg-type]
        width = int(sprite["width"])  # type: ignore[arg-type]
        local = sprite["local"]  # type: ignore[assignment]
        transformed = _transform_local_coords(local, height, width, kind)
        for rr, cc in transformed:
            out[base_r + r0 + rr, base_c + c0 + cc] = target_color

    return out


SHAPES = [
    ((0, 0), (1, 0), (1, 1)),
    ((0, 0), (1, 0), (2, 0), (2, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 1), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (2, 2)),
    ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (2, 3)),
]

PAIR_LAYOUTS: dict[str, list[tuple[tuple[int, int], tuple[int, int], tuple[int, int]]]] = {
    "hflip": [
        ((0, 0), (0, 1), (1, 2)),
        ((0, 1), (0, 0), (1, 2)),
    ],
    "vflip": [
        ((0, 0), (1, 0), (2, 1)),
        ((1, 0), (0, 0), (2, 1)),
    ],
    "rot180": [
        ((0, 0), (1, 1), (2, 2)),
        ((1, 1), (0, 0), (2, 2)),
        ((0, 1), (1, 0), (2, 2)),
        ((1, 0), (0, 1), (2, 2)),
    ],
}

MAX_ATTEMPTS = 400


def _validate_shapes() -> None:
    for shape in SHAPES:
        coords = list(shape)
        h = max(r for r, _ in coords) + 1
        w = max(c for _, c in coords) + 1
        variants = {_normalize(_transform_local_coords(coords, h, w, kind)) for kind in TRANSFORMS}
        assert _normalize(coords) not in variants
        assert len(variants) == 3


_validate_shapes()


def _paint_coords(grid: np.ndarray, coords: list[tuple[int, int]], color: int) -> None:
    for r, c in coords:
        grid[r, c] = color


def _place_sprite(
    quad_h: int, quad_w: int, shape: tuple[tuple[int, int], ...]
) -> tuple[list[tuple[int, int]], tuple[int, int, int, int]]:
    h = max(r for r, _ in shape) + 1
    w = max(c for _, c in shape) + 1
    margin_r = 1 if quad_h - h >= 2 else 0
    margin_c = 1 if quad_w - w >= 2 else 0
    r0 = random.randint(margin_r, quad_h - h - margin_r)
    c0 = random.randint(margin_c, quad_w - w - margin_c)
    coords = [(r0 + r, c0 + c) for r, c in shape]
    return coords, (r0, c0, r0 + h - 1, c0 + w - 1)


def _can_place_cells(blocked: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = blocked.shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if blocked[r, c]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, cells: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        blocked[max(0, r - 1) : min(h, r + 2), max(0, c - 1) : min(w, c + 2)] = True


def _place_legend(
    quad_h: int,
    quad_w: int,
    specs: list[tuple[int, int, str]],
) -> np.ndarray | None:
    legend = np.zeros((quad_h, quad_w), dtype=int)
    blocked = np.zeros((quad_h, quad_w), dtype=bool)
    order = list(specs)
    random.shuffle(order)

    for source_color, target_color, kind in order:
        layouts = list(PAIR_LAYOUTS[kind])
        random.shuffle(layouts)
        placed = False
        for source_rel, target_rel, (ph, pw) in layouts:
            anchors = [(r, c) for r in range(quad_h - ph + 1) for c in range(quad_w - pw + 1)]
            random.shuffle(anchors)
            for r0, c0 in anchors:
                source = (r0 + source_rel[0], c0 + source_rel[1])
                target = (r0 + target_rel[0], c0 + target_rel[1])
                cells = [source, target]
                if not _can_place_cells(blocked, cells):
                    continue
                legend[source] = source_color
                legend[target] = target_color
                _mark_blocked(blocked, cells)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None
    return legend


def _assemble_case() -> np.ndarray | None:
    top_h = random.randint(7, 10)
    bottom_h = random.randint(7, 10)
    left_w = random.randint(7, 10)
    right_w = random.randint(7, 10)
    height = top_h + bottom_h + 1
    width = left_w + right_w + 1

    quad_dims = {
        0: (top_h, left_w),
        1: (top_h, right_w),
        2: (bottom_h, left_w),
        3: (bottom_h, right_w),
    }
    quad_origins = {
        0: (0, 0),
        1: (0, left_w + 1),
        2: (top_h + 1, 0),
        3: (top_h + 1, left_w + 1),
    }

    legend_idx = random.randrange(4)
    sprite_quads = [idx for idx in range(4) if idx != legend_idx]

    colors = random.sample(PALETTE, 6)
    sprite_colors = colors[:3]
    target_colors = colors[3:]
    shapes = random.sample(SHAPES, 3)
    transforms = list(TRANSFORMS)
    random.shuffle(transforms)

    grid = np.zeros((height, width), dtype=int)
    grid[top_h, :] = GREY
    grid[:, left_w] = GREY

    for quad_idx, color, shape in zip(sprite_quads, sprite_colors, shapes):
        quad_h, quad_w = quad_dims[quad_idx]
        local_coords, bbox = _place_sprite(quad_h, quad_w, shape)
        r0, c0 = quad_origins[quad_idx]
        global_coords = [(r0 + r, c0 + c) for r, c in local_coords]
        _paint_coords(grid, global_coords, color)

        local_bbox_coords = [(r - bbox[0], c - bbox[1]) for r, c in local_coords]
        h = bbox[2] - bbox[0] + 1
        w = bbox[3] - bbox[1] + 1
        kind = transforms[sprite_colors.index(color)]
        moved = _normalize(_transform_local_coords(local_bbox_coords, h, w, kind))
        assert moved != _normalize(local_bbox_coords)

    legend_specs = list(zip(sprite_colors, target_colors, transforms))
    legend_h, legend_w = quad_dims[legend_idx]
    legend = _place_legend(legend_h, legend_w, legend_specs)
    if legend is None:
        return None

    lr0, lc0 = quad_origins[legend_idx]
    grid[lr0 : lr0 + legend_h, lc0 : lc0 + legend_w] = legend

    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _assemble_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
