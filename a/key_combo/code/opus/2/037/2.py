from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 8
GREY = 5
RED = 2
ORANGE = 7


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flat)).most_common(1)[0][0])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _neighbors8(r: int, c: int):
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            yield r + dr, c + dc


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != transparent
    view[mask] = patch[mask]


def _label_components(mask: np.ndarray, connectivity: int) -> tuple[np.ndarray, list[np.ndarray]]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    components: list[np.ndarray] = []
    label_id = 0
    neighbors = _neighbors4 if connectivity == 4 else _neighbors8

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            label_id += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = label_id
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in neighbors(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = label_id
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return labels, components


def _normalized_shape(component: np.ndarray) -> tuple[tuple[int, int], ...]:
    r0, c0, _r1, _c1 = _bbox_coords(component)
    pts = sorted((int(r - r0), int(c - c0)) for r, c in component)
    return tuple(pts)


def _component_touches_edge(coords: np.ndarray, shape: tuple[int, int]) -> bool:
    h, w = shape
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    bg_labels, bg_components = _label_components(g == bg, connectivity=4)
    bg_touches_edge = {
        label_id: _component_touches_edge(coords, g.shape)
        for label_id, coords in enumerate(bg_components, start=1)
    }

    grey_labels, grey_components = _label_components(g == GREY, connectivity=8)

    sprites: list[dict] = []
    prototypes: dict[int, np.ndarray] = {}
    prototype_shapes: dict[int, tuple[tuple[int, int], ...]] = {}

    for color in (RED, ORANGE):
        _labels, comps = _label_components(g == color, connectivity=4)
        if not comps:
            continue
        comps = sorted(comps, key=lambda pts: _bbox_coords(pts))
        bbox = _bbox_coords(comps[0])
        prototypes[color] = _crop_bbox(g, bbox)
        prototype_shapes[color] = _normalized_shape(comps[0])

        for comp in comps:
            bbox = _bbox_coords(comp)
            adjacent_bg_labels = set()
            for r, c in comp:
                for nr, nc in _neighbors4(int(r), int(c)):
                    if _in_bounds(g.shape, nr, nc) and bg_labels[nr, nc] > 0:
                        adjacent_bg_labels.add(int(bg_labels[nr, nc]))

            enclosed = bool(adjacent_bg_labels) and all(
                not bg_touches_edge[label_id] for label_id in adjacent_bg_labels
            )

            ring_ids: set[int] = set()
            cavity_labels = {label_id for label_id in adjacent_bg_labels if not bg_touches_edge[label_id]}
            if enclosed:
                for label_id in cavity_labels:
                    for r, c in bg_components[label_id - 1]:
                        for nr, nc in _neighbors8(int(r), int(c)):
                            if _in_bounds(g.shape, nr, nc) and grey_labels[nr, nc] > 0:
                                ring_ids.add(int(grey_labels[nr, nc]))

            sprites.append(
                {
                    "color": color,
                    "coords": comp,
                    "bbox": bbox,
                    "shape": _normalized_shape(comp),
                    "adjacent_bg_labels": adjacent_bg_labels,
                    "enclosed": enclosed,
                    "ring_ids": ring_ids,
                }
            )

    return {
        "grid": g,
        "bg": bg,
        "sprites": sprites,
        "prototypes": prototypes,
        "prototype_shapes": prototype_shapes,
        "grey_components": grey_components,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Swap only the sprites trapped inside closed grey loops, then recolor those loops to the swapped sprite color.

    Input structure:
    - The background is teal, with small red and orange 4-connected sprite instances of two fixed prototypes.
    - Some sprites sit inside irregular grey closed loops with at least one background cell separating loop and sprite.
    - Other sprites may have nearby grey fragments, but those fragments do not topologically close off the adjacent background.

    Rule:
    - Detect every red and orange sprite component and recover the red prototype and orange prototype from their existing instances.
    - For each sprite, inspect the 4-connected background region touching the sprite: if that region cannot reach the grid edge, the sprite is enclosed.
    - Enclosed red sprites are replaced by the orange prototype at the same top-left anchor, enclosed orange sprites are replaced by the red prototype, and the enclosing grey loop is recolored to the new sprite color.
    - Open sprites, open grey fragments, and all remaining background stay unchanged.

    Key insight:
    - The decisive test is not whether grey is nearby, but whether the sprite-adjacent background is cut off from the outer background; only those topologically closed cases trigger both prototype swapping and boundary recoloring.
    """
    info = _analyze_grid(grid)
    g = info["grid"]
    bg = info["bg"]
    out = g.copy()

    if RED not in info["prototypes"] or ORANGE not in info["prototypes"]:
        return out

    for sprite in info["sprites"]:
        if not sprite["enclosed"]:
            continue

        src_color = int(sprite["color"])
        dst_color = ORANGE if src_color == RED else RED
        dst_proto = info["prototypes"][dst_color]

        for r, c in sprite["coords"]:
            out[int(r), int(c)] = bg

        r0, c0, _r1, _c1 = sprite["bbox"]
        _paste(out, dst_proto, r0, c0, transparent=bg)

        for ring_id in sprite["ring_ids"]:
            coords = info["grey_components"][ring_id - 1]
            out[coords[:, 0], coords[:, 1]] = dst_color

    return out


_SHAPE_LIBRARY: dict[int, list[tuple[tuple[int, int], ...]]] = {
    4: [
        ((0, 0), (1, 0), (2, 0), (2, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1)),
        ((0, 0), (0, 1), (1, 1), (2, 1)),
        ((0, 0), (1, 0), (1, 1), (1, 2)),
    ],
    5: [
        ((0, 0), (1, 0), (2, 0), (3, 0), (3, 1)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2)),
        ((0, 2), (1, 0), (1, 1), (1, 2), (2, 0)),
        ((0, 0), (0, 1), (1, 1), (1, 2), (2, 2)),
    ],
    6: [
        ((0, 0), (1, 0), (2, 0), (2, 1), (2, 2), (3, 2)),
        ((0, 1), (0, 2), (1, 0), (1, 1), (2, 0), (3, 0)),
        ((0, 0), (0, 1), (1, 1), (2, 1), (2, 2), (3, 2)),
        ((0, 0), (1, 0), (1, 1), (1, 2), (2, 2), (3, 2)),
    ],
}


def _shape_to_patch(coords: tuple[tuple[int, int], ...], color: int) -> np.ndarray:
    pts = np.array(coords, dtype=int)
    r0, c0, r1, c1 = _bbox_coords(pts)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), BG, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = color
    return patch


def _choose_prototypes() -> tuple[np.ndarray, np.ndarray]:
    area = random.choice(sorted(_SHAPE_LIBRARY))
    shapes = random.sample(_SHAPE_LIBRARY[area], 2)
    return _shape_to_patch(shapes[0], RED), _shape_to_patch(shapes[1], ORANGE)


def _boundary_from_fill(fill: np.ndarray) -> np.ndarray:
    h, w = fill.shape
    boundary = np.zeros_like(fill, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not fill[r, c]:
                continue
            for nr, nc in _neighbors4(r, c):
                if not _in_bounds(fill.shape, nr, nc) or not fill[nr, nc]:
                    boundary[r, c] = True
                    break
    return boundary


def _make_enclosed_patch(sprite: np.ndarray, max_dims: tuple[int, int]) -> np.ndarray:
    sprite_h, sprite_w = sprite.shape
    max_h, max_w = max_dims

    extra_top = random.randint(1, 2)
    extra_bottom = random.randint(1, 2)
    extra_left = random.randint(1, 2)
    extra_right = random.randint(1, 2)
    if extra_top + extra_bottom + extra_left + extra_right == 0:
        extra_top = 1

    base_h = max_h + 4 + random.randint(0, 1)
    base_w = max_w + 4 + random.randint(0, 1)
    patch_h = base_h + extra_top + extra_bottom
    patch_w = base_w + extra_left + extra_right

    fill = np.zeros((patch_h, patch_w), dtype=bool)
    base_r0 = extra_top
    base_c0 = extra_left
    fill[base_r0 : base_r0 + base_h, base_c0 : base_c0 + base_w] = True

    protrusion_count = random.randint(1, 3)
    sides = ["top", "bottom", "left", "right"]
    random.shuffle(sides)
    for side in sides[:protrusion_count]:
        if side == "top" and extra_top > 0:
            ph = random.randint(1, extra_top)
            pw = random.randint(max(2, base_w // 3), max(2, base_w - 1))
            start = random.randint(0, base_w - pw)
            fill[base_r0 - ph : base_r0, base_c0 + start : base_c0 + start + pw] = True
        elif side == "bottom" and extra_bottom > 0:
            ph = random.randint(1, extra_bottom)
            pw = random.randint(max(2, base_w // 3), max(2, base_w - 1))
            start = random.randint(0, base_w - pw)
            fill[base_r0 + base_h : base_r0 + base_h + ph, base_c0 + start : base_c0 + start + pw] = True
        elif side == "left" and extra_left > 0:
            pw = random.randint(1, extra_left)
            ph = random.randint(max(2, base_h // 3), max(2, base_h - 1))
            start = random.randint(0, base_h - ph)
            fill[base_r0 + start : base_r0 + start + ph, base_c0 - pw : base_c0] = True
        elif side == "right" and extra_right > 0:
            pw = random.randint(1, extra_right)
            ph = random.randint(max(2, base_h // 3), max(2, base_h - 1))
            start = random.randint(0, base_h - ph)
            fill[base_r0 + start : base_r0 + start + ph, base_c0 + base_w : base_c0 + base_w + pw] = True

    patch = np.full((patch_h, patch_w), BG, dtype=int)
    boundary = _boundary_from_fill(fill)
    patch[boundary] = GREY

    slot_r = base_r0 + 2 + random.randint(0, base_h - max_h - 4)
    slot_c = base_c0 + 2 + random.randint(0, base_w - max_w - 4)
    _paste(patch, sprite, slot_r, slot_c, transparent=BG)
    return patch


def _draw_fragment_line(patch: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        if _in_bounds(patch.shape, r, c) and patch[r, c] == BG:
            patch[r, c] = GREY


def _make_open_patch(sprite: np.ndarray) -> np.ndarray:
    h, w = sprite.shape
    pad_top = random.randint(2, 4)
    pad_bottom = random.randint(2, 4)
    pad_left = random.randint(2, 4)
    pad_right = random.randint(2, 4)
    patch = np.full((h + pad_top + pad_bottom, w + pad_left + pad_right), BG, dtype=int)
    sr = pad_top
    sc = pad_left
    _paste(patch, sprite, sr, sc, transparent=BG)

    side_groups = [
        ("top", random.randint(max(1, w - 1), w + 1)),
        ("bottom", random.randint(max(1, w - 1), w + 1)),
        ("left", random.randint(max(1, h - 1), h + 1)),
        ("right", random.randint(max(1, h - 1), h + 1)),
    ]
    random.shuffle(side_groups)
    side_count = random.randint(1, 2)
    for side, span in side_groups[:side_count]:
        if side == "top":
            row = sr - random.choice([1, 2])
            start = sc - random.randint(0, 1)
            cells = [(row, cc) for cc in range(start, min(start + span, patch.shape[1]))]
            if random.random() < 0.5:
                cells.append((row + 1, min(cells[-1][1], patch.shape[1] - 1)))
        elif side == "bottom":
            row = sr + h - 1 + random.choice([1, 2])
            start = sc - random.randint(0, 1)
            cells = [(row, cc) for cc in range(start, min(start + span, patch.shape[1]))]
            if random.random() < 0.5:
                cells.append((row - 1, min(cells[-1][1], patch.shape[1] - 1)))
        elif side == "left":
            col = sc - random.choice([1, 2])
            start = sr - random.randint(0, 1)
            cells = [(rr, col) for rr in range(start, min(start + span, patch.shape[0]))]
            if random.random() < 0.5:
                cells.append((min(cells[-1][0], patch.shape[0] - 1), col + 1))
        else:
            col = sc + w - 1 + random.choice([1, 2])
            start = sr - random.randint(0, 1)
            cells = [(rr, col) for rr in range(start, min(start + span, patch.shape[0]))]
            if random.random() < 0.5:
                cells.append((min(cells[-1][0], patch.shape[0] - 1), col - 1))
        _draw_fragment_line(patch, cells)
    return patch


def _find_location(reserved: np.ndarray, patch_shape: tuple[int, int], *, border: int, padding: int) -> tuple[int, int]:
    h, w = reserved.shape
    ph, pw = patch_shape
    candidates: list[tuple[int, int]] = []
    for r0 in range(border, h - ph - border + 1):
        for c0 in range(border, w - pw - border + 1):
            rr0 = max(0, r0 - padding)
            cc0 = max(0, c0 - padding)
            rr1 = min(h, r0 + ph + padding)
            cc1 = min(w, c0 + pw + padding)
            if not reserved[rr0:rr1, cc0:cc1].any():
                candidates.append((r0, c0))
    if not candidates:
        raise ValueError("no placement location")
    return random.choice(candidates)


def _reserve_bbox(reserved: np.ndarray, r0: int, c0: int, patch_shape: tuple[int, int]) -> None:
    ph, pw = patch_shape
    reserved[r0 : r0 + ph, c0 : c0 + pw] = True


def _generate_layout() -> np.ndarray:
    proto_red, proto_orange = _choose_prototypes()
    max_dims = (
        max(proto_red.shape[0], proto_orange.shape[0]),
        max(proto_red.shape[1], proto_orange.shape[1]),
    )

    specs: list[tuple[int, bool]] = []
    for color in (RED, ORANGE):
        count = random.randint(2, 3)
        statuses = [True, False]
        if count == 3:
            statuses.append(random.choice([True, False]))
        random.shuffle(statuses)
        specs.extend((color, enclosed) for enclosed in statuses)
    random.shuffle(specs)

    patches: list[np.ndarray] = []
    for color, enclosed in specs:
        sprite = proto_red if color == RED else proto_orange
        patch = _make_enclosed_patch(sprite, max_dims) if enclosed else _make_open_patch(sprite)
        patches.append(patch)

    total_area = sum(int(p.shape[0] * p.shape[1]) for p in patches)
    min_h = max(max(p.shape[0] for p in patches) + 2, 18)
    min_w = max(max(p.shape[1] for p in patches) + 2, 18)

    size_choices = [
        (h, w)
        for h in range(min_h, 31)
        for w in range(min_w, 31)
        if h * w >= total_area + 80
    ]
    if not size_choices:
        raise ValueError("no feasible grid size")

    grid_h, grid_w = random.choice(size_choices)
    grid = np.full((grid_h, grid_w), BG, dtype=int)
    reserved = np.zeros((grid_h, grid_w), dtype=bool)

    for patch in sorted(patches, key=lambda arr: arr.shape[0] * arr.shape[1], reverse=True):
        r0, c0 = _find_location(reserved, patch.shape, border=1, padding=1)
        _paste(grid, patch, r0, c0, transparent=BG)
        _reserve_bbox(reserved, r0, c0, patch.shape)

    return grid


def _validate_generated(grid: np.ndarray) -> bool:
    if grid.ndim != 2 or max(grid.shape) > 30:
        return False

    info = _analyze_grid(grid)
    sprites = info["sprites"]
    if RED not in info["prototypes"] or ORANGE not in info["prototypes"]:
        return False

    counts = {RED: {"enclosed": 0, "open": 0}, ORANGE: {"enclosed": 0, "open": 0}}
    for sprite in sprites:
        bucket = "enclosed" if sprite["enclosed"] else "open"
        counts[int(sprite["color"])][bucket] += 1
        if sprite["enclosed"] and not sprite["ring_ids"]:
            return False

    if any(counts[color]["enclosed"] < 1 or counts[color]["open"] < 1 for color in (RED, ORANGE)):
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid = _generate_layout()
        except ValueError:
            continue
        if _validate_generated(grid):
            return grid
    raise RuntimeError("failed to generate a valid input")
