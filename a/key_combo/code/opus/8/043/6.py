from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
FRAME_COLOR = 3
SPRITE_COLORS = (1, 2, 4, 6, 7, 8, 9)
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIR8 = [(dr, dc) for dr in (-1, 0, 1) for dc in (-1, 0, 1) if (dr, dc) != (0, 0)]
MAX_GEN_ATTEMPTS = 400


def _background_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return grid[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray, connectivity: int) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    deltas = DIR4 if connectivity == 4 else DIR8
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q: deque[tuple[int, int]] = deque([(r, c)])
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in deltas:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _color_components(grid: np.ndarray, bg: int, connectivity: int) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, np.unique(grid))) - {int(bg)}):
        for coords in _components(grid == color, connectivity):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(len(coords)),
                }
            )
    return out


def _tight_mask(coords: np.ndarray) -> tuple[tuple[int, int], np.ndarray]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return (int(r0), int(c0)), mask


def _orientations(mask: np.ndarray) -> list[np.ndarray]:
    seen = set()
    outs: list[np.ndarray] = []
    base = mask.astype(bool)
    for flipped in (False, True):
        flipped_mask = np.fliplr(base) if flipped else base
        for k in range(4):
            cur = np.rot90(flipped_mask, k)
            cur = np.ascontiguousarray(cur)
            key = (cur.shape, cur.astype(np.uint8, copy=False).tobytes())
            if key not in seen:
                seen.add(key)
                outs.append(cur)
    return outs


def _canonical_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return min((m.shape, m.astype(np.uint8, copy=False).tobytes()) for m in _orientations(mask))


def _paint_mask(grid: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    patch = grid[top : top + mask.shape[0], left : left + mask.shape[1]]
    patch[mask] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match outside sprites to frame holes, then fill each hole with the next smaller hole's matched color.

    Input structure:
    - The background is black, and one large green frame object occupies the center of the grid.
    - Inside the frame are 3-4 black holes, each a different 4-connected shape with a unique pixel area.
    - Outside the frame are 3-4 monochrome sprites of distinct colors; each sprite matches exactly one hole up to rotation or reflection.

    Rule:
    - Find the largest monochrome object and crop the output to its bounding box.
    - Match each hole inside the frame to the uniquely shape-equivalent outside sprite, allowing all rotations and reflections.
    - Sort holes by pixel area from largest to smallest, then fill each hole with the matched sprite color of the next hole in that area ranking, wrapping the smallest hole to the largest hole's color.

    Key insight:
    - The final color of a hole depends on two linked inferences: shape matching identifies which sprite color belongs to each hole, and area ranking decides how those colors are cyclically reassigned.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)

    comps = _color_components(g, bg, connectivity=8)
    assert comps, "expected foreground components"
    frame = max(comps, key=lambda comp: comp["area"])
    frame_color = int(frame["color"])
    frame_box = frame["bbox"]
    fr0, fc0, fr1, fc1 = frame_box

    sub = _crop(g, frame_box)
    hole_infos = []
    for coords in _components(sub == bg, connectivity=4):
        r0, c0, r1, c1 = _bbox(coords)
        assert r0 > 0 and c0 > 0 and r1 < sub.shape[0] - 1 and c1 < sub.shape[1] - 1
        (top, left), mask = _tight_mask(coords)
        hole_infos.append(
            {
                "top": int(top),
                "left": int(left),
                "mask": mask,
                "area": int(mask.sum()),
                "key": _canonical_key(mask),
            }
        )

    assert 3 <= len(hole_infos) <= 4
    assert len({hole["area"] for hole in hole_infos}) == len(hole_infos)

    outside = g.copy()
    outside[fr0 : fr1 + 1, fc0 : fc1 + 1] = bg
    sprite_map: dict[tuple[tuple[int, int], bytes], int] = {}
    for comp in _color_components(outside, bg, connectivity=8):
        (_, _), mask = _tight_mask(comp["coords"])
        key = _canonical_key(mask)
        assert key not in sprite_map, "expected one sprite per hole shape"
        sprite_map[key] = int(comp["color"])

    assert len(sprite_map) == len(hole_infos)
    ordered = sorted(hole_infos, key=lambda hole: hole["area"], reverse=True)
    matched_colors = [sprite_map[hole["key"]] for hole in ordered]
    shifted_colors = matched_colors[1:] + matched_colors[:1]

    out = np.full(sub.shape, frame_color, dtype=int)
    for hole, color in zip(ordered, shifted_colors):
        _paint_mask(out, hole["top"], hole["left"], hole["mask"], color)
    return out


SHAPES_BY_AREA = {
    3: [
        np.array([[1, 0], [1, 1]], dtype=bool),
        np.array([[1, 1, 1]], dtype=bool),
    ],
    4: [
        np.array([[1, 1], [1, 1]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 0], [1, 0], [1, 1]], dtype=bool),
    ],
    5: [
        np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
        np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
        np.array([[1, 1], [1, 1], [1, 0]], dtype=bool),
    ],
    6: [
        np.array([[1, 1, 1], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 0, 0]], dtype=bool),
        np.array([[1, 1, 1], [0, 1, 0], [0, 1, 1]], dtype=bool),
    ],
    7: [
        np.array([[1, 1, 1], [0, 1, 0], [1, 1, 1]], dtype=bool),
        np.array([[1, 1, 0], [1, 1, 1], [1, 1, 0]], dtype=bool),
        np.array([[1, 1, 1], [1, 0, 1], [1, 1, 0]], dtype=bool),
    ],
}


def _pick_shapes(area_count: int) -> list[dict]:
    areas = sorted(random.sample([3, 4, 5, 6, 7], area_count), reverse=True)
    chosen = []
    asymmetric_index = None
    for idx, area in enumerate(areas):
        mask = random.choice(SHAPES_BY_AREA[area])
        chosen.append({"area": area, "base": mask})
        if len(_orientations(mask)) > 1 and asymmetric_index is None:
            asymmetric_index = idx
    if asymmetric_index is None:
        return []

    twist_done = False
    for idx, item in enumerate(chosen):
        hole_mask = random.choice(_orientations(item["base"]))
        sprite_choices = _orientations(item["base"])
        if idx == asymmetric_index:
            alt = [m for m in sprite_choices if m.shape != hole_mask.shape or not np.array_equal(m, hole_mask)]
            if not alt:
                return []
            sprite_mask = random.choice(alt)
            twist_done = True
        else:
            sprite_mask = random.choice(sprite_choices)
        item["hole_mask"] = hole_mask
        item["sprite_mask"] = sprite_mask
    return chosen if twist_done else []


def _hole_slots(frame_h: int, frame_w: int) -> list[tuple[int, int, int, int]]:
    mid_r = frame_h // 2
    mid_c = frame_w // 2
    return [
        (1, 1, mid_r - 2, mid_c - 2),
        (1, mid_c + 1, mid_r - 2, frame_w - 2),
        (mid_r + 1, 1, frame_h - 2, mid_c - 2),
        (mid_r + 1, mid_c + 1, frame_h - 2, frame_w - 2),
    ]


def _place_holes(frame_h: int, frame_w: int, shapes: list[dict]) -> list[dict] | None:
    slots = _hole_slots(frame_h, frame_w)
    slot_indices = random.sample(range(4), len(shapes))
    placed = []
    for item, slot_idx in zip(shapes, slot_indices):
        mask = item["hole_mask"]
        sr0, sc0, sr1, sc1 = slots[slot_idx]
        sh, sw = mask.shape
        max_r = sr1 - sh + 1
        max_c = sc1 - sw + 1
        if max_r < sr0 or max_c < sc0:
            return None
        top = random.randint(sr0, max_r)
        left = random.randint(sc0, max_c)
        placed.append(
            {
                **item,
                "slot": int(slot_idx),
                "top": int(top),
                "left": int(left),
            }
        )
    return placed


def _sprite_slots(grid_h: int, grid_w: int, frame_box: tuple[int, int, int, int]) -> list[tuple[int, int, int, int]]:
    fr0, fc0, fr1, fc1 = frame_box
    return [
        (0, 0, fr0 - 2, fc0 - 2),
        (0, fc0, fr0 - 2, fc1),
        (0, fc1 + 2, fr0 - 2, grid_w - 1),
        (fr0, 0, fr1, fc0 - 2),
        (fr0, fc1 + 2, fr1, grid_w - 1),
        (fr1 + 2, 0, grid_h - 1, fc0 - 2),
        (fr1 + 2, fc0, grid_h - 1, fc1),
        (fr1 + 2, fc1 + 2, grid_h - 1, grid_w - 1),
    ]


def _place_sprites(
    grid_h: int,
    grid_w: int,
    frame_box: tuple[int, int, int, int],
    placed_holes: list[dict],
) -> list[dict] | None:
    slots = _sprite_slots(grid_h, grid_w, frame_box)
    slot_order = list(range(len(slots)))
    random.shuffle(slot_order)
    used_slots = set()
    sprites = []

    for hole in placed_holes:
        mask = hole["sprite_mask"]
        sh, sw = mask.shape
        fit_slots = []
        for idx in slot_order:
            if idx in used_slots:
                continue
            sr0, sc0, sr1, sc1 = slots[idx]
            if sr1 - sr0 + 1 >= sh and sc1 - sc0 + 1 >= sw:
                fit_slots.append(idx)
        if not fit_slots:
            return None
        slot_idx = random.choice(fit_slots)
        used_slots.add(slot_idx)
        sr0, sc0, sr1, sc1 = slots[slot_idx]
        top = random.randint(sr0, sr1 - sh + 1)
        left = random.randint(sc0, sc1 - sw + 1)
        sprites.append({**hole, "sprite_top": int(top), "sprite_left": int(left), "sprite_slot": int(slot_idx)})
    return sprites


def _render_expected(frame_h: int, frame_w: int, placed: list[dict]) -> np.ndarray:
    out = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    ordered = sorted(placed, key=lambda item: item["area"], reverse=True)
    colors = [int(item["color"]) for item in ordered]
    rotated = colors[1:] + colors[:1]
    for item, color in zip(ordered, rotated):
        _paint_mask(out, item["top"], item["left"], item["hole_mask"], color)
    return out


def _render_identity(frame_h: int, frame_w: int, placed: list[dict]) -> np.ndarray:
    out = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    for item in placed:
        _paint_mask(out, item["top"], item["left"], item["hole_mask"], item["color"])
    return out


def _render_top_left_shift(frame_h: int, frame_w: int, placed: list[dict]) -> np.ndarray:
    out = np.full((frame_h, frame_w), FRAME_COLOR, dtype=int)
    ordered = sorted(placed, key=lambda item: (item["top"], item["left"]))
    colors = [int(item["color"]) for item in ordered]
    rotated = colors[1:] + colors[:1]
    for item, color in zip(ordered, rotated):
        _paint_mask(out, item["top"], item["left"], item["hole_mask"], color)
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = random.choice([3, 4])
    shapes = _pick_shapes(hole_count)
    if not shapes:
        return None

    frame_h = random.randint(10, 14)
    frame_w = random.randint(10, 14)
    if frame_h < 10 or frame_w < 10:
        return None

    placed_holes = _place_holes(frame_h, frame_w, shapes)
    if placed_holes is None:
        return None

    grid_h = frame_h + random.randint(8, min(10, 22 - frame_h))
    grid_w = frame_w + random.randint(8, min(10, 22 - frame_w))
    frame_top = random.randint(4, grid_h - frame_h - 4)
    frame_left = random.randint(4, grid_w - frame_w - 4)
    frame_box = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    placed = _place_sprites(grid_h, grid_w, frame_box, placed_holes)
    if placed is None:
        return None

    colors = random.sample(SPRITE_COLORS, hole_count)
    for item, color in zip(placed, colors):
        item["color"] = int(color)

    grid = np.zeros((grid_h, grid_w), dtype=int)
    fr0, fc0, _, _ = frame_box
    grid[fr0 : fr0 + frame_h, fc0 : fc0 + frame_w] = FRAME_COLOR

    hole_keys = set()
    for item in placed:
        hole_key = _canonical_key(item["hole_mask"])
        sprite_key = _canonical_key(item["sprite_mask"])
        if hole_key != sprite_key or hole_key in hole_keys:
            return None
        hole_keys.add(hole_key)
        _paint_mask(grid, fr0 + item["top"], fc0 + item["left"], item["hole_mask"], BG)
        _paint_mask(grid, item["sprite_top"], item["sprite_left"], item["sprite_mask"], item["color"])

    expected = _render_expected(frame_h, frame_w, placed)
    identity = _render_identity(frame_h, frame_w, placed)
    top_left_shift = _render_top_left_shift(frame_h, frame_w, placed)

    if np.array_equal(expected, identity):
        return None
    if np.array_equal(expected, top_left_shift):
        return None

    ranked_by_area = [item["area"] for item in sorted(placed, key=lambda item: item["area"], reverse=True)]
    ranked_by_pos = [item["area"] for item in sorted(placed, key=lambda item: (item["top"], item["left"]))]
    if ranked_by_area == ranked_by_pos:
        return None

    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        grid, expected = built
        solved = main(grid)
        if solved.shape != expected.shape or not np.array_equal(solved, expected):
            continue

        bg = _background_color(grid)
        comps = _color_components(grid, bg, connectivity=8)
        frame = max(comps, key=lambda comp: comp["area"])
        if int(frame["color"]) != FRAME_COLOR:
            continue
        sub = _crop(grid, frame["bbox"])
        holes = _components(sub == bg, connectivity=4)
        if not (3 <= len(holes) <= 4):
            continue
        if len({len(coords) for coords in holes}) != len(holes):
            continue

        outside = grid.copy()
        r0, c0, r1, c1 = frame["bbox"]
        outside[r0 : r1 + 1, c0 : c1 + 1] = bg
        sprites = _color_components(outside, bg, connectivity=8)
        if len(sprites) != len(holes):
            continue
        if len({int(comp["color"]) for comp in sprites}) != len(sprites):
            continue

        hole_keys = {_canonical_key(_tight_mask(coords)[1]) for coords in holes}
        sprite_keys = {_canonical_key(_tight_mask(comp["coords"])[1]) for comp in sprites}
        if hole_keys != sprite_keys:
            continue
        if not (3 <= len(set(map(int, grid.flatten())) - {BG, FRAME_COLOR}) <= 4):
            continue
        return grid
    raise RuntimeError("failed to generate a valid input")
