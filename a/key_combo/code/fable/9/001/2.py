from __future__ import annotations

import random
from collections import Counter, defaultdict
from typing import Iterable

import numpy as np


BBox = tuple[int, int, int, int]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> BBox:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    step4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    step8 = step4 + [(-1, -1), (-1, 1), (1, -1), (1, 1)]
    steps = step4 if connectivity == 4 else step8
    out: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in steps:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.array(coords, dtype=int))
    return out


def _shape_signature(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(mask.shape), mask.astype(np.uint8).tobytes()


def _draw_room(
    patch: np.ndarray,
    top: int,
    left: int,
    height: int,
    width: int,
    frame_color: int,
    sprite_mask: np.ndarray,
    sprite_color: int,
) -> None:
    patch[top, left : left + width] = frame_color
    patch[top + height - 1, left : left + width] = frame_color
    patch[top : top + height, left] = frame_color
    patch[top : top + height, left + width - 1] = frame_color

    inner_h = height - 2
    inner_w = width - 2
    sh, sw = sprite_mask.shape
    sprite_top = top + 1 + random.randint(0, inner_h - sh)
    sprite_left = left + 1 + random.randint(0, inner_w - sw)
    sprite = patch[sprite_top : sprite_top + sh, sprite_left : sprite_left + sw]
    sprite[sprite_mask] = sprite_color


def _is_hollow_room(grid: np.ndarray, color: int, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return False
    frame = np.zeros((h, w), dtype=bool)
    frame[0, :] = True
    frame[-1, :] = True
    frame[:, 0] = True
    frame[:, -1] = True
    return np.array_equal(grid[r0 : r1 + 1, c0 : c1 + 1] == color, frame)


def _find_rooms(grid: np.ndarray, bg: int) -> tuple[int, list[dict]]:
    colors = [int(v) for v in np.unique(grid) if int(v) != bg]
    candidates: list[tuple[int, list[dict]]] = []
    for color in colors:
        rooms = []
        for coords in _components(grid == color, connectivity=4):
            if _is_hollow_room(grid, color, coords):
                rooms.append({"bbox": _bbox_from_coords(coords)})
        if len(rooms) >= 2 and len(rooms) % 2 == 0:
            candidates.append((color, rooms))

    assert candidates, "expected hollow rooms"
    max_count = max(len(rooms) for _color, rooms in candidates)
    winners = [(color, rooms) for color, rooms in candidates if len(rooms) == max_count]
    assert len(winners) == 1, "room color must be unique"
    frame_color, rooms = winners[0]
    return frame_color, rooms


def _room_sprite_mask(grid: np.ndarray, room: dict, bg: int, frame_color: int) -> np.ndarray:
    r0, c0, r1, c1 = room["bbox"]
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    mask = (interior != bg) & (interior != frame_color)
    comps = _components(mask, connectivity=8)
    assert len(comps) == 1, "each room must contain one sprite"
    sr0, sc0, sr1, sc1 = _bbox_from_coords(comps[0])
    return mask[sr0 : sr1 + 1, sc0 : sc1 + 1].copy()


def _connect_pair(out: np.ndarray, bg: int, frame_color: int, a: dict, b: dict) -> None:
    ar0, ac0, ar1, ac1 = a["bbox"]
    br0, bc0, br1, bc1 = b["bbox"]

    if ac1 < bc0 or bc1 < ac0:
        left, right = (a, b) if ac1 < bc0 else (b, a)
        lr0, lc0, lr1, lc1 = left["bbox"]
        rr0, rc0, rr1, rc1 = right["bbox"]
        ov0 = max(lr0 + 1, rr0 + 1)
        ov1 = min(lr1 - 1, rr1 - 1)
        gap0 = lc1 + 1
        gap1 = rc0 - 1
        assert ov1 - ov0 + 1 >= 3 and gap0 <= gap1
        assert np.all(out[ov0 : ov1 + 1, gap0 : gap1 + 1] == bg)
        out[ov0, gap0 : gap1 + 1] = frame_color
        out[ov1, gap0 : gap1 + 1] = frame_color
        out[ov0 + 1 : ov1, lc1] = bg
        out[ov0 + 1 : ov1, rc0] = bg
        return

    if ar1 < br0 or br1 < ar0:
        top, bottom = (a, b) if ar1 < br0 else (b, a)
        tr0, tc0, tr1, tc1 = top["bbox"]
        br0, bc0, br1, bc1 = bottom["bbox"]
        ov0 = max(tc0 + 1, bc0 + 1)
        ov1 = min(tc1 - 1, bc1 - 1)
        gap0 = tr1 + 1
        gap1 = br0 - 1
        assert ov1 - ov0 + 1 >= 3 and gap0 <= gap1
        assert np.all(out[gap0 : gap1 + 1, ov0 : ov1 + 1] == bg)
        out[gap0 : gap1 + 1, ov0] = frame_color
        out[gap0 : gap1 + 1, ov1] = frame_color
        out[tr1, ov0 + 1 : ov1] = bg
        out[br0, ov0 + 1 : ov1] = bg
        return

    raise AssertionError("matched rooms must be separated horizontally or vertically")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair hollow rooms by interior sprite shape and connect each pair with an overlap-defined corridor.

    Input structure:
    - Several hollow rectangular rooms of one shared frame color sit on a background.
    - Each room contains one small interior sprite; matching is by sprite shape only, while
      same-shape pairs use different sprite colors and repeated colors belong to different shapes.
    - For every true pair, the two rooms are separated horizontally or vertically, and their
      interior intervals overlap across a clear background gap.

    Rule:
    - Detect the hollow rooms and crop each interior sprite to its shape mask, ignoring color.
    - Pair the two rooms whose sprite masks are identical.
    - For each pair, use the overlap of the two room interiors to position a corridor: paint the
      two boundary rows or columns of that overlap across the gap in the room color, then erase
      the facing wall segment strictly between those boundaries so the interiors connect.

    Key insight:
    - Neither sprite color nor nearest distance identifies the partner room; only the normalized
      interior sprite shape determines the pairing, and the corridor geometry comes from interior
      overlap rather than outer-frame alignment.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    frame_color, rooms = _find_rooms(g, bg)

    groups: dict[tuple[tuple[int, int], bytes], list[dict]] = defaultdict(list)
    for room in rooms:
        sprite_mask = _room_sprite_mask(g, room, bg, frame_color)
        groups[_shape_signature(sprite_mask)].append(room)

    assert groups and all(len(group) == 2 for group in groups.values())

    out = g.copy()
    for pair in groups.values():
        _connect_pair(out, bg, frame_color, pair[0], pair[1])
    return out


_SHAPES = [
    np.array([[1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[1, 1, 1], [1, 0, 0]], dtype=bool),
    np.array([[0, 1, 0], [1, 1, 1]], dtype=bool),
    np.array([[1, 1], [1, 1]], dtype=bool),
]


def _sample_pair_colors(n_pairs: int, palette: list[int]) -> list[tuple[int, int]]:
    base = random.sample(palette, 3)
    if n_pairs == 2:
        pairs = [(base[0], base[1]), (base[0], base[2])]
    else:
        pairs = [(base[0], base[1]), (base[0], base[2]), (base[1], base[2])]
    random.shuffle(pairs)
    return [pair if random.random() < 0.5 else (pair[1], pair[0]) for pair in pairs]


def _make_horizontal_pair(shape: np.ndarray, frame_color: int, colors: tuple[int, int]) -> np.ndarray:
    sh, sw = shape.shape
    ih1 = random.randint(max(4, sh), 7)
    iw1 = random.randint(max(4, sw), 7)
    ih2 = random.randint(max(4, sh), 7)
    iw2 = random.randint(max(4, sw), 7)
    overlap = random.randint(3, min(ih1, ih2))
    extra1_top = random.randint(0, ih1 - overlap)
    extra2_top = random.randint(0, ih2 - overlap)
    extra1_bottom = ih1 - overlap - extra1_top
    extra2_bottom = ih2 - overlap - extra2_top
    overlap_top = max(extra1_top, extra2_top) + 1

    room1_top = overlap_top - extra1_top - 1
    room2_top = overlap_top - extra2_top - 1
    room1_h = ih1 + 2
    room2_h = ih2 + 2
    room1_w = iw1 + 2
    room2_w = iw2 + 2
    gap = random.randint(2, 6)

    patch_h = max(room1_top + room1_h, room2_top + room2_h)
    patch_w = room1_w + gap + room2_w
    patch = np.zeros((patch_h, patch_w), dtype=int)
    _draw_room(patch, room1_top, 0, room1_h, room1_w, frame_color, shape, colors[0])
    _draw_room(patch, room2_top, room1_w + gap, room2_h, room2_w, frame_color, shape, colors[1])
    return patch


def _make_vertical_pair(shape: np.ndarray, frame_color: int, colors: tuple[int, int]) -> np.ndarray:
    sh, sw = shape.shape
    ih1 = random.randint(max(4, sh), 7)
    iw1 = random.randint(max(4, sw), 7)
    ih2 = random.randint(max(4, sh), 7)
    iw2 = random.randint(max(4, sw), 7)
    overlap = random.randint(3, min(iw1, iw2))
    extra1_left = random.randint(0, iw1 - overlap)
    extra2_left = random.randint(0, iw2 - overlap)
    extra1_right = iw1 - overlap - extra1_left
    extra2_right = iw2 - overlap - extra2_left
    overlap_left = max(extra1_left, extra2_left) + 1

    room1_left = overlap_left - extra1_left - 1
    room2_left = overlap_left - extra2_left - 1
    room1_h = ih1 + 2
    room2_h = ih2 + 2
    room1_w = iw1 + 2
    room2_w = iw2 + 2
    gap = random.randint(2, 6)

    patch_h = room1_h + gap + room2_h
    patch_w = max(room1_left + room1_w, room2_left + room2_w)
    patch = np.zeros((patch_h, patch_w), dtype=int)
    _draw_room(patch, 0, room1_left, room1_h, room1_w, frame_color, shape, colors[0])
    _draw_room(patch, room1_h + gap, room2_left, room2_h, room2_w, frame_color, shape, colors[1])
    return patch


def _can_place(blocked: np.ndarray, top: int, left: int, patch: np.ndarray, pad: int) -> bool:
    h, w = patch.shape
    rr0 = max(0, top - pad)
    cc0 = max(0, left - pad)
    rr1 = min(blocked.shape[0], top + h + pad)
    cc1 = min(blocked.shape[1], left + w + pad)
    return not blocked[rr0:rr1, cc0:cc1].any()


def _place_patch(canvas: np.ndarray, blocked: np.ndarray, patch: np.ndarray, pad: int = 1) -> bool:
    h, w = patch.shape
    positions = [(r, c) for r in range(canvas.shape[0] - h + 1) for c in range(canvas.shape[1] - w + 1)]
    random.shuffle(positions)
    mask = patch != 0
    for top, left in positions:
        if not _can_place(blocked, top, left, patch, pad):
            continue
        region = canvas[top : top + h, left : left + w]
        if np.any(mask & (region != 0)):
            continue
        region[mask] = patch[mask]
        rr0 = max(0, top - pad)
        cc0 = max(0, left - pad)
        rr1 = min(blocked.shape[0], top + h + pad)
        cc1 = min(blocked.shape[1], left + w + pad)
        blocked[rr0:rr1, cc0:cc1] = True
        return True
    return False


def _crop_with_margin(grid: np.ndarray, bg: int) -> np.ndarray:
    coords = np.argwhere(grid != bg)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    top_margin = random.randint(0, min(3, r0))
    left_margin = random.randint(0, min(3, c0))
    bottom_margin = random.randint(0, min(3, grid.shape[0] - 1 - r1))
    right_margin = random.randint(0, min(3, grid.shape[1] - 1 - c1))
    return grid[
        r0 - top_margin : r1 + bottom_margin + 1,
        c0 - left_margin : c1 + right_margin + 1,
    ].copy()


def _build_candidate() -> np.ndarray:
    bg = 0
    canvas = np.zeros((30, 30), dtype=int)
    blocked = np.zeros((30, 30), dtype=bool)

    n_pairs = random.choice([2, 3])
    room_color = random.choice([c for c in range(1, 10)])
    sprite_palette = [c for c in range(1, 10) if c != room_color]
    pair_colors = _sample_pair_colors(n_pairs, sprite_palette)
    shapes = random.sample(_SHAPES, n_pairs)

    orientations = ["h", "v"]
    if n_pairs == 3:
        orientations.append(random.choice(["h", "v"]))
    random.shuffle(orientations)

    patches = []
    for shape, colors, orient in zip(shapes, pair_colors, orientations):
        patch = (
            _make_horizontal_pair(shape, room_color, colors)
            if orient == "h"
            else _make_vertical_pair(shape, room_color, colors)
        )
        patches.append(patch)

    patches.sort(key=lambda p: p.shape[0] * p.shape[1], reverse=True)
    for patch in patches:
        if not _place_patch(canvas, blocked, patch, pad=1):
            raise RuntimeError("placement failed")

    return _crop_with_margin(canvas, bg)


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid = _build_candidate()
            out = main(grid)
        except Exception:
            continue

        if np.array_equal(grid, out):
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        colors = [int(v) for v in np.unique(grid) if int(v) != 0]
        if len(colors) < 4:
            continue
        counts = Counter(int(v) for v in grid.ravel() if int(v) != 0)
        if counts.most_common(1)[0][1] < 2:
            continue
        return grid
    raise RuntimeError("failed to generate a valid paired-corridor puzzle")
