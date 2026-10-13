from __future__ import annotations

import random
from typing import Iterable

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
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


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr = int(rr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: Iterable[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [int(r) for r, _ in coords]
    cols = [int(c) for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _mask_from_coords(coords: Iterable[tuple[int, int]], shape: tuple[int, int]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for r, c in coords:
        mask[int(r), int(c)] = True
    return mask


def _enclosed_region(outline_mask: np.ndarray) -> np.ndarray:
    h, w = outline_mask.shape
    outside = np.zeros((h, w), dtype=bool)
    stack: list[tuple[int, int]] = []

    for r in range(h):
        for c in (0, w - 1):
            if not outline_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((int(r), int(c)))
    for c in range(w):
        for r in (0, h - 1):
            if not outline_mask[r, c] and not outside[r, c]:
                outside[r, c] = True
                stack.append((int(r), int(c)))

    while stack:
        r, c = stack.pop()
        for dr, dc in NEIGHBORS4:
            nr = int(r + dr)
            nc = int(c + dc)
            if 0 <= nr < h and 0 <= nc < w and not outline_mask[nr, nc] and not outside[nr, nc]:
                outside[nr, nc] = True
                stack.append((nr, nc))

    return (~outline_mask) & (~outside)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _erode8(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = np.zeros_like(mask, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not mask[r, c]:
                continue
            keep = True
            for dr, dc in NEIGHBORS8:
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w and mask[nr, nc]):
                    keep = False
                    break
            if keep:
                out[r, c] = True
    return out


def _find_separator(grid: np.ndarray, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        col = grid[:, c]
        if int(c) * 2 + 1 != int(w):
            continue
        if np.all(col == col[0]) and int(col[0]) != int(bg):
            candidates.append((int(c), int(col[0])))
    assert len(candidates) == 1, "expected one centered full-height separator"
    return candidates[0]


def _extract_outline_objects(panel: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for coords in _components(panel != bg):
        r0, c0, r1, c1 = _bbox(coords)
        patch = panel[r0 : r1 + 1, c0 : c1 + 1]
        local_coords = [(int(r - r0), int(c - c0)) for r, c in coords]
        outline_mask = _mask_from_coords(local_coords, patch.shape)
        interior = _enclosed_region(outline_mask)
        if not interior.any():
            continue

        payload: list[tuple[int, int, int]] = []
        ys, xs = np.where(interior & (patch != int(bg)))
        for rr, cc in zip(ys, xs):
            payload.append((int(rr), int(cc), int(patch[int(rr), int(cc)])))

        objects.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "shape": outline_mask,
                "interior": interior,
                "key": _shape_key(outline_mask),
                "payload": payload,
            }
        )
    return objects


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    sep_col, _sep_color = _find_separator(g, bg)

    left_panel = g[:, :sep_col]
    right_panel = g[:, sep_col + 1 :]
    left_objects = _extract_outline_objects(left_panel, bg)
    right_objects = _extract_outline_objects(right_panel, bg)

    assert len(left_objects) == len(right_objects) and left_objects, "panels must contain matching outlines"

    left_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    right_by_key: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for obj in left_objects:
        left_by_key.setdefault(obj["key"], []).append(obj)
    for obj in right_objects:
        right_by_key.setdefault(obj["key"], []).append(obj)

    assert left_by_key.keys() == right_by_key.keys(), "shape sets must match across panels"

    out = g.copy()
    for key, left_group in left_by_key.items():
        right_group = right_by_key[key]
        assert len(left_group) == len(right_group), "each outline shape must occur equally often"

        left_group = sorted(left_group, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))
        right_group = sorted(right_group, key=lambda obj: (obj["bbox"][0], obj["bbox"][1]))

        for left_obj, right_obj in zip(left_group, right_group):
            h, w = left_obj["shape"].shape
            assert right_obj["shape"].shape == (h, w)
            rr0, rc0, _rr1, _rc1 = right_obj["bbox"]
            for r, c, value in left_obj["payload"]:
                mc = int(w - 1 - c)
                assert right_obj["interior"][r, mc], "mirrored texture must remain inside the matched outline"
                out[int(rr0 + r), int(sep_col + 1 + rc0 + mc)] = int(value)

    return out


BG_COLOR = 0
TEXTURE_COLOR = 4
SEPARATOR_COLOR = 5
OUTLINE_COLORS = (1, 2, 3, 6, 7, 8, 9)
SHAPE_FAMILIES = ("rectangle",)

OBJECT_COUNT_RANGE = (2, 3)
PANEL_HEIGHT_RANGE = (14, 19)
PANEL_WIDTH_RANGE = (11, 14)
OUTER_HEIGHT_RANGE = (6, 10)
OUTER_WIDTH_CHOICES = (7, 9)
PLACEMENT_MARGIN = 1
GEN_ATTEMPTS = 400
TEXTURE_ATTEMPTS = 100


def _outline_from_filled(filled: np.ndarray) -> np.ndarray:
    h, w = filled.shape
    outline = np.zeros_like(filled, dtype=bool)
    for r in range(h):
        for c in range(w):
            if not filled[r, c]:
                continue
            for dr, dc in NEIGHBORS4:
                nr = int(r + dr)
                nc = int(c + dc)
                if not (0 <= nr < h and 0 <= nc < w) or not filled[nr, nc]:
                    outline[r, c] = True
                    break
    return outline


def _make_filled_rectangle() -> np.ndarray:
    h = random.randint(*OUTER_HEIGHT_RANGE)
    w = random.choice(OUTER_WIDTH_CHOICES)
    return np.ones((h, w), dtype=bool)


def _texture_mask(interior: np.ndarray) -> np.ndarray | None:
    h, w = interior.shape
    cells = [tuple(map(int, xy)) for xy in np.argwhere(interior)]
    if len(cells) < 2:
        return None

    for _ in range(TEXTURE_ATTEMPTS):
        mode = random.choice(("blob", "scatter"))
        count = random.randint(1, min(6, len(cells)))
        chosen: set[tuple[int, int]] = set()

        if mode == "scatter":
            chosen = set(random.sample(cells, count))
        else:
            chosen = {random.choice(cells)}
            while len(chosen) < count:
                frontier: list[tuple[int, int]] = []
                for r, c in chosen:
                    for dr, dc in NEIGHBORS4:
                        nr = int(r + dr)
                        nc = int(c + dc)
                        if 0 <= nr < h and 0 <= nc < w and interior[nr, nc] and (nr, nc) not in chosen:
                            frontier.append((nr, nc))
                if frontier:
                    chosen.add(random.choice(frontier))
                else:
                    remaining = [cell for cell in cells if cell not in chosen]
                    if not remaining:
                        break
                    chosen.add(random.choice(remaining))

        mask = np.zeros_like(interior, dtype=bool)
        for r, c in chosen:
            mask[int(r), int(c)] = True

        if np.array_equal(mask, np.fliplr(mask)):
            continue
        return mask

    return None


def _make_shape() -> dict | None:
    family = "rectangle"
    filled = _make_filled_rectangle()

    outline = _outline_from_filled(filled)
    interior = _enclosed_region(outline)
    if interior.sum() < 2:
        return None

    texture_zone = _erode8(interior)
    texture = _texture_mask(texture_zone)
    if texture is None:
        return None

    key = _shape_key(outline)
    return {
        "family": family,
        "outline": outline,
        "interior": interior,
        "texture": texture,
        "key": key,
        "shape": outline.shape,
    }


def _render_patch(shape: dict, outline_color: int, with_texture: bool) -> np.ndarray:
    patch = np.full(shape["shape"], BG_COLOR, dtype=int)
    patch[shape["outline"]] = int(outline_color)
    if with_texture:
        patch[shape["texture"]] = TEXTURE_COLOR
    return patch


def _place_shapes(panel_h: int, panel_w: int, shapes: list[dict]) -> list[tuple[int, int]] | None:
    order = sorted(range(len(shapes)), key=lambda i: shapes[i]["shape"][0] * shapes[i]["shape"][1], reverse=True)
    occupied = np.zeros((panel_h, panel_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(shapes)

    for idx in order:
        h, w = shapes[idx]["shape"]
        if h > panel_h or w > panel_w:
            return None
        placed = False
        for _ in range(200):
            r0 = random.randint(0, panel_h - h)
            c0 = random.randint(0, panel_w - w)
            rr0 = max(0, r0 - PLACEMENT_MARGIN)
            cc0 = max(0, c0 - PLACEMENT_MARGIN)
            rr1 = min(panel_h, r0 + h + PLACEMENT_MARGIN)
            cc1 = min(panel_w, c0 + w + PLACEMENT_MARGIN)
            if occupied[rr0:rr1, cc0:cc1].any():
                continue
            placements[idx] = (int(r0), int(c0))
            occupied[r0 : r0 + h, c0 : c0 + w] = True
            placed = True
            break
        if not placed:
            return None

    return [(int(r), int(c)) for r, c in placements if r is not None and c is not None]


def _non_identity_permutation(n: int) -> list[int]:
    perm = list(range(n))
    if n <= 1:
        return perm
    while True:
        random.shuffle(perm)
        if any(i != perm[i] for i in range(n)):
            return perm


def _compose_input(
    panel_h: int,
    panel_w: int,
    shapes: list[dict],
    left_positions: list[tuple[int, int]],
    right_positions: list[tuple[int, int]],
    left_colors: list[int],
    right_colors: list[int],
) -> np.ndarray:
    grid = np.full((panel_h, 2 * panel_w + 1), BG_COLOR, dtype=int)
    grid[:, panel_w] = SEPARATOR_COLOR

    for i, shape in enumerate(shapes):
        lr, lc = left_positions[i]
        rr, rc = right_positions[i]
        left_patch = _render_patch(shape, left_colors[i], with_texture=True)
        right_patch = _render_patch(shape, right_colors[i], with_texture=False)
        lh, lw = left_patch.shape
        rh, rw = right_patch.shape
        grid[lr : lr + lh, lc : lc + lw] = left_patch
        grid[rr : rr + rh, panel_w + 1 + rc : panel_w + 1 + rc + rw] = right_patch

    return grid


def _expected_output(
    grid: np.ndarray,
    panel_w: int,
    shapes: list[dict],
    right_positions: list[tuple[int, int]],
) -> np.ndarray:
    out = np.array(grid, copy=True)
    for shape, (r0, c0) in zip(shapes, right_positions):
        texture = np.fliplr(shape["texture"])
        h, w = texture.shape
        patch = out[r0 : r0 + h, panel_w + 1 + c0 : panel_w + 1 + c0 + w]
        patch[texture] = TEXTURE_COLOR
    return out


def _generate_case() -> np.ndarray | None:
    n_objects = random.randint(*OBJECT_COUNT_RANGE)

    shapes: list[dict] = []
    keys: set[tuple[tuple[int, int], bytes]] = set()
    for _ in range(200):
        if len(shapes) == n_objects:
            break
        shape = _make_shape()
        if shape is None or shape["key"] in keys:
            continue
        shapes.append(shape)
        keys.add(shape["key"])
    if len(shapes) != n_objects:
        return None

    max_h = max(shape["shape"][0] for shape in shapes)
    max_w = max(shape["shape"][1] for shape in shapes)
    min_h = max(PANEL_HEIGHT_RANGE[0], max_h + 2)
    min_w = max(PANEL_WIDTH_RANGE[0], max_w + 2)
    if min_h > PANEL_HEIGHT_RANGE[1] or min_w > PANEL_WIDTH_RANGE[1]:
        return None

    panel_h = random.randint(min_h, PANEL_HEIGHT_RANGE[1])
    panel_w = random.randint(min_w, PANEL_WIDTH_RANGE[1])
    if sum((shape["shape"][0] + 2) * (shape["shape"][1] + 2) for shape in shapes) > panel_h * panel_w:
        return None

    left_positions = _place_shapes(panel_h, panel_w, shapes)
    right_positions = _place_shapes(panel_h, panel_w, shapes)
    if left_positions is None or right_positions is None:
        return None

    left_colors = random.sample(list(OUTLINE_COLORS), n_objects)
    perm = _non_identity_permutation(n_objects)
    right_colors = [left_colors[i] for i in perm]

    grid = _compose_input(panel_h, panel_w, shapes, left_positions, right_positions, left_colors, right_colors)
    bg_count = int((grid == BG_COLOR).sum())
    if bg_count <= max(int((grid == color).sum()) for color in np.unique(grid) if int(color) != BG_COLOR):
        return None
    expected = _expected_output(grid, panel_w, shapes, right_positions)
    produced = main(grid)
    if not np.array_equal(produced, expected):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        case = _generate_case()
        if case is not None:
            return case
    raise RuntimeError("failed to generate a valid puzzle instance")
