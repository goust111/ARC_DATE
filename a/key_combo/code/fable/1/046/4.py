from __future__ import annotations

import random
from typing import Iterable

import numpy as np


BG = 0
PALETTE = list(range(1, 10))
MAX_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(values).argmax())


def _bbox_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue
            color = int(g[r, c])
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            touches_border = False
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            bbox = _bbox_coords(coords)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": bbox,
                    "touches_border": touches_border,
                    "patch": _crop_bbox(g, bbox),
                }
            )
    return comps


def _rect_sum(prefix: np.ndarray, r0: int, c0: int, r1: int, c1: int) -> int:
    return int(
        prefix[r1 + 1, c1 + 1]
        - prefix[r0, c1 + 1]
        - prefix[r1 + 1, c0]
        + prefix[r0, c0]
    )


def _largest_background_rectangle(
    grid: np.ndarray, bg: int
) -> tuple[tuple[int, int, int, int], int]:
    occ = (np.asarray(grid, dtype=int) != bg).astype(int)
    h, w = occ.shape
    prefix = np.pad(occ, ((1, 0), (1, 0)), constant_values=0).cumsum(0).cumsum(1)

    best_bbox: tuple[int, int, int, int] | None = None
    best_area = -1
    tie_count = 0

    for r0 in range(h - 1):
        for r1 in range(r0 + 1, h):
            rect_h = r1 - r0 + 1
            for c0 in range(w - 1):
                for c1 in range(c0 + 1, w):
                    rect_w = c1 - c0 + 1
                    area = rect_h * rect_w
                    if area < best_area:
                        continue
                    if _rect_sum(prefix, r0, c0, r1, c1) != 0:
                        continue
                    if area > best_area:
                        best_area = area
                        best_bbox = (r0, c0, r1, c1)
                        tie_count = 1
                    else:
                        tie_count += 1

    assert best_bbox is not None, "expected at least one all-background rectangle"
    return best_bbox, tie_count


def _paste_transparent(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, bg: int) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    mask = patch != bg
    view[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move the uniquely size-matching floating sprite into the largest empty background cavity.

    Input structure:
    - The grid has one background color, several border-touching solid clutter pieces that partition space, and several monochrome floating sprites.
    - Among all axis-aligned all-background rectangles with both sides at least 2, one maximum-area cavity is unique.
    - Exactly one floating sprite has a tight bounding box whose dimensions match the cavity, possibly after a 90-degree rotation.

    Rule:
    - Search all all-background rectangles of height and width at least 2 and select the unique maximum-area one.
    - Among floating monochrome components, find the unique sprite whose bounding-box dimensions equal the cavity dimensions, allowing a 90-degree rotation.
    - Erase that sprite from its original position and redraw it so its bounding box exactly coincides with the cavity, rotating it by 90 degrees when needed.

    Key insight:
    - The target is identified by a negative-space measurement, not by color or salience: first locate the largest empty rectangle, then match only by bounding-box dimensions.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)

    cavity_bbox, _ = _largest_background_rectangle(g, bg)
    cr0, cc0, cr1, cc1 = cavity_bbox
    cavity_shape = (cr1 - cr0 + 1, cc1 - cc0 + 1)

    candidates: list[dict] = []
    for comp in _components(g, bg):
        if comp["touches_border"]:
            continue
        patch = comp["patch"]
        if patch.shape == cavity_shape or patch.shape == cavity_shape[::-1]:
            candidates.append(comp)

    assert len(candidates) == 1, "expected one size-matching floating sprite"
    target = candidates[0]

    out = g.copy()
    for r, c in target["coords"]:
        out[r, c] = bg

    patch = target["patch"]
    if patch.shape != cavity_shape:
        patch = np.rot90(patch)
    _paste_transparent(out, patch, cr0, cc0, bg)
    return out


def _room_box(name: str, h1: int, w1: int, h2: int, w2: int, w3: int) -> tuple[int, int, int, int]:
    boxes = {
        "cavity": (1, 1, h1, w1),
        "top_mid": (1, w1 + 2, h1, w2),
        "top_right": (1, w1 + w2 + 3, h1, w3),
        "bottom_left": (h1 + 2, 1, h2, w1),
        "bottom_mid": (h1 + 2, w1 + 2, h2, w2),
        "bottom_right": (h1 + 2, w1 + w2 + 3, h2, w3),
    }
    return boxes[name]


def _iter_orbits_180(h: int, w: int) -> list[tuple[tuple[int, int], ...]]:
    seen: set[tuple[int, int]] = set()
    orbits: list[tuple[tuple[int, int], ...]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) in seen:
                continue
            mate = (h - 1 - r, w - 1 - c)
            orbit = tuple(sorted({(r, c), mate}))
            seen.update(orbit)
            orbits.append(orbit)
    return orbits


def _is_connected(mask: np.ndarray) -> bool:
    cells = list(map(tuple, np.argwhere(mask)))
    if not cells:
        return False
    start = cells[0]
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc))
                stack.append((nr, nc))
    return len(seen) == int(mask.sum())


def _random_sprite_mask(h: int, w: int, require_hole: bool) -> np.ndarray:
    interior_orbits = [
        orbit
        for orbit in _iter_orbits_180(h, w)
        if all(0 < r < h - 1 and 0 < c < w - 1 for r, c in orbit)
    ]

    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        if interior_orbits:
            max_remove = max(1, min(len(interior_orbits), (h * w) // 4))
            remove_count = random.randint(1 if require_hole else 0, max_remove)
            chosen = random.sample(interior_orbits, remove_count)
            for orbit in chosen:
                for r, c in orbit:
                    mask[r, c] = False
        if require_hole and mask.all():
            continue
        if not _is_connected(mask):
            continue
        if np.all(mask[0]) and np.all(mask[-1]) and np.all(mask[:, 0]) and np.all(mask[:, -1]):
            return mask
        if mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any():
            return mask
    raise RuntimeError("failed to generate a connected sprite mask")


def _render_patch(mask: np.ndarray, color: int) -> np.ndarray:
    return np.where(mask, int(color), BG).astype(int)


def _serialize_patch(patch: np.ndarray) -> tuple:
    mask = patch != BG
    return (mask.shape, tuple(map(int, mask.ravel())))


def _paste_room_sprite(
    grid: np.ndarray,
    room: tuple[int, int, int, int],
    patch: np.ndarray,
) -> tuple[int, int]:
    r0, c0, _h, _w = room
    _paste_transparent(grid, patch, r0 + 1, c0 + 1, BG)
    return r0 + 1, c0 + 1


def _apply_global_symmetry(grid: np.ndarray, rotation_k: int, flip_lr: bool) -> np.ndarray:
    out = np.rot90(grid, rotation_k)
    if flip_lr:
        out = np.fliplr(out)
    return out.copy()


def _build_instance() -> tuple[np.ndarray, np.ndarray]:
    while True:
        h1, w1 = random.randint(4, 7), random.randint(4, 7)
        if h1 == w1:
            continue
        rotate_needed = bool(random.getrandbits(1))
        target_h, target_w = (w1, h1) if rotate_needed else (h1, w1)
        h2 = target_h + 2
        w2 = target_w + 2
        w3_choices = [v for v in range(4, 9) if v - 2 not in {w1, h1, target_w}]
        if not w3_choices:
            continue
        w3 = random.choice(w3_choices)
        candidate_shapes = {(h1, w1), (w1, h1)}
        room_dims = {
            "top_mid": (h1 - 2, w2 - 2),
            "top_right": (h1 - 2, w3 - 2),
            "bottom_left": (h2 - 2, w1 - 2),
            "bottom_mid": (h2 - 2, w2 - 2),
            "bottom_right": (h2 - 2, w3 - 2),
        }
        if len(set(room_dims.values())) != len(room_dims):
            continue
        if all(dim not in candidate_shapes for name, dim in room_dims.items() if name != "bottom_mid"):
            break

    grid_h = h1 + h2 + 3
    grid_w = w1 + w2 + w3 + 4
    assert grid_h <= 30 and grid_w <= 30

    colors = random.sample(PALETTE, 9)
    border_color, vwall_1, vwall_2, hwall_color = colors[:4]
    sprite_colors = colors[4:]

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    inp[0, :] = border_color
    inp[-1, :] = border_color
    inp[:, 0] = border_color
    inp[:, -1] = border_color
    inp[:, w1 + 1] = vwall_1
    inp[:, w1 + w2 + 2] = vwall_2
    inp[h1 + 1, :] = hwall_color

    room_names = ["top_mid", "top_right", "bottom_left", "bottom_mid", "bottom_right"]
    room_boxes = {name: _room_box(name, h1, w1, h2, w2, w3) for name in ["cavity", *room_names]}

    used_shapes: set[tuple] = set()
    patches: dict[str, np.ndarray] = {}

    for idx, name in enumerate(room_names):
        rr, cc, rh, rw = room_boxes[name]
        bbox_h = rh - 2
        bbox_w = rw - 2
        require_hole = name == "bottom_mid" or (bbox_h >= 4 and bbox_w >= 4 and random.random() < 0.75)
        for _ in range(200):
            mask = _random_sprite_mask(bbox_h, bbox_w, require_hole=require_hole)
            patch = _render_patch(mask, sprite_colors[idx])
            sig = _serialize_patch(patch)
            if sig in used_shapes:
                continue
            if patch.shape in {(h1, w1), (w1, h1)} and name != "bottom_mid":
                continue
            used_shapes.add(sig)
            patches[name] = patch
            break
        else:
            raise RuntimeError("failed to create unique sprite")
        _paste_room_sprite(inp, room_boxes[name], patches[name])

    out = inp.copy()
    target_room = room_boxes["bottom_mid"]
    target_patch = patches["bottom_mid"]
    tr0, tc0, _th, _tw = target_room
    out[tr0 + 1 : tr0 + 1 + target_patch.shape[0], tc0 + 1 : tc0 + 1 + target_patch.shape[1]][
        target_patch != BG
    ] = BG

    placed_patch = np.rot90(target_patch) if rotate_needed else target_patch
    cr0, cc0, _ch, _cw = room_boxes["cavity"]
    _paste_transparent(out, placed_patch, cr0, cc0, BG)

    rotation_k = random.randrange(4)
    flip_lr = bool(random.getrandbits(1))
    return _apply_global_symmetry(inp, rotation_k, flip_lr), _apply_global_symmetry(out, rotation_k, flip_lr)


def _validate_instance(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(expected.shape) > 30:
        return False
    bg = _get_bg(inp)
    cavity_bbox, tie_count = _largest_background_rectangle(inp, bg)
    if tie_count != 1:
        return False
    cr0, cc0, cr1, cc1 = cavity_bbox
    if cr1 - cr0 + 1 < 2 or cc1 - cc0 + 1 < 2:
        return False

    cavity_shape = (cr1 - cr0 + 1, cc1 - cc0 + 1)
    candidates = []
    for comp in _components(inp, bg):
        if comp["touches_border"]:
            continue
        patch = comp["patch"]
        if patch.shape == cavity_shape or patch.shape == cavity_shape[::-1]:
            candidates.append(comp)
    if len(candidates) != 1:
        return False
    if np.array_equal(inp, expected):
        return False
    return bool(np.array_equal(main(inp), expected))


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        inp, expected = _build_instance()
        if _validate_instance(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid negative-space lock-key instance")
