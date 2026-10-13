from __future__ import annotations

from collections import Counter, defaultdict
import random

import numpy as np


START_OFFSETS = ((0, 0), (-1, -1), (-1, 1), (1, -1), (1, 1))
MAX_GENERATION_ATTEMPTS = 600
BACKGROUND = 0


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, box: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = box
    return np.asarray(grid, dtype=int)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _neighbors4(r: int, c: int, shape: tuple[int, int]):
    h, w = shape
    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            yield nr, nc


def _components_mask(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    parts: list[np.ndarray] = []
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
                for nr, nc in _neighbors4(rr, cc, mask.shape):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            parts.append(np.array(coords, dtype=int))
    return parts


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    comps = []
    for coords in _components_mask(np.asarray(grid, dtype=int) != int(bg)):
        comps.append(
            {
                "coords": coords,
                "bbox": bbox_coords(coords),
                "area": int(coords.shape[0]),
            }
        )
    return comps


def _signature(arr: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(map(int, row)) for row in np.asarray(arr, dtype=int))


def _draw_period2_column(out: np.ndarray, start_r: int, c: int, color: int) -> None:
    h, w = out.shape
    if not (0 <= c < w) or start_r >= h:
        return
    r = int(start_r)
    while r < 0:
        r += 2
    while r < h:
        out[r, c] = int(color)
        r += 2


def _render_from_selected_crop(
    grid_shape: tuple[int, int],
    crop: np.ndarray,
    top_left: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(grid_shape, dtype=int)
    r0, c0 = map(int, top_left)
    h, w = crop.shape
    out[r0 : r0 + h, c0 : c0 + w] = np.asarray(crop, dtype=int)
    for dr in range(h):
        for dc in range(w):
            color = int(crop[dr, dc])
            if color == BACKGROUND:
                continue
            seed_r = r0 + dr
            seed_c = c0 + dc
            for off_r, off_c in START_OFFSETS:
                _draw_period2_column(out, seed_r + off_r, seed_c + off_c, color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep the unique multicolor sprite and grow period-2 diagonal-column trails from every pixel it contains.

    Input structure:
    - The grid has a black background.
    - Several separated 4-connected multicolor sprites appear in the upper part of the grid.
    - All sprites share the same bounding-box size and foreground mask, but exactly one sprite has a different colored crop.

    Rule:
    - Split all non-black pixels into 4-connected sprites and crop each sprite to its tight bounding box.
    - Compare the exact cropped color patterns, including black cells inside the crop, and find the only pattern that occurs once.
    - Erase every other sprite.
    - For each non-black pixel of the unique sprite, scanned from top-left to bottom-right inside the crop, draw vertical period-2 trails of that pixel's color starting from the pixel itself and from its four diagonal neighbors; later seeds overwrite earlier ones on overlaps.

    Key insight:
    - The singleton sprite is identified by exact repeated-pattern matching, and its internal multicolor layout becomes a whole family of colored growth seeds rather than a single selected object.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    objects = _find_components(g, bg)
    crops = [crop_bbox(g, obj["bbox"]) for obj in objects]
    counts = Counter(_signature(crop) for crop in crops)
    unique = [sig for sig, n in counts.items() if n == 1]
    assert len(unique) == 1
    target_sig = unique[0]
    for obj, crop in zip(objects, crops):
        if _signature(crop) == target_sig:
            r0, c0, _r1, _c1 = obj["bbox"]
            return _render_from_selected_crop(g.shape, crop, (r0, c0))
    raise AssertionError("no unique sprite found")


def _mask_signature(mask: np.ndarray) -> tuple[tuple[int, ...], ...]:
    return _signature(mask.astype(int))


def _mask_connected(mask: np.ndarray) -> bool:
    parts = _components_mask(mask)
    return len(parts) == 1 and int(parts[0].shape[0]) == int(np.count_nonzero(mask))


def _random_mask(h: int, w: int) -> np.ndarray | None:
    max_area = min(8, h * w - 1)
    for _ in range(160):
        area = random.randint(4, max_area)
        picks = random.sample(range(h * w), area)
        mask = np.zeros((h, w), dtype=bool)
        for idx in picks:
            mask[idx // w, idx % w] = True
        if not _mask_connected(mask):
            continue
        if bbox_coords(np.argwhere(mask)) != (0, 0, h - 1, w - 1):
            continue
        return mask
    return None


def _random_color_counts(area: int, num_colors: int) -> list[int]:
    cuts = sorted(random.sample(range(1, area), num_colors - 1))
    parts = []
    prev = 0
    for cut in cuts + [area]:
        parts.append(cut - prev)
        prev = cut
    random.shuffle(parts)
    return parts


def _colorize_mask(mask: np.ndarray, palette: list[int], counts: list[int]) -> np.ndarray:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    color_list = []
    for color, count in zip(palette, counts):
        color_list.extend([int(color)] * int(count))
    random.shuffle(color_list)
    sprite = np.zeros(mask.shape, dtype=int)
    for (r, c), color in zip(coords, color_list):
        sprite[r, c] = color
    return sprite


def _placement_candidates(h: int, w: int, box_h: int, box_w: int, max_row: int) -> list[tuple[int, int]]:
    return [(r, c) for r in range(max_row + 1) for c in range(w - box_w + 1)]


def _mark_reserved(reserved: np.ndarray, r: int, c: int, box_h: int, box_w: int) -> None:
    h, w = reserved.shape
    r0 = max(0, r - 1)
    c0 = max(0, c - 1)
    r1 = min(h, r + box_h + 1)
    c1 = min(w, c + box_w + 1)
    reserved[r0:r1, c0:c1] = True


def _can_place(reserved: np.ndarray, r: int, c: int, box_h: int, box_w: int) -> bool:
    return not bool(reserved[r : r + box_h, c : c + box_w].any())


def _place_bboxes(
    grid_shape: tuple[int, int],
    box_shape: tuple[int, int],
    count: int,
    anomaly_pos: tuple[int, int],
) -> list[tuple[int, int]] | None:
    h, w = grid_shape
    box_h, box_w = box_shape
    max_row = max(box_h, h // 2) - box_h
    candidates = _placement_candidates(h, w, box_h, box_w, max_row)
    random.shuffle(candidates)
    reserved = np.zeros(grid_shape, dtype=bool)
    _mark_reserved(reserved, anomaly_pos[0], anomaly_pos[1], box_h, box_w)
    positions = [anomaly_pos]
    for r, c in candidates:
        if len(positions) == count:
            break
        if (r, c) == anomaly_pos:
            continue
        if _can_place(reserved, r, c, box_h, box_w):
            positions.append((r, c))
            _mark_reserved(reserved, r, c, box_h, box_w)
    if len(positions) != count:
        return None
    random.shuffle(positions[1:])
    return positions


def _track_contributions(
    sprite: np.ndarray,
    top_left: tuple[int, int],
    grid_shape: tuple[int, int],
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    hits: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    r0, c0 = map(int, top_left)
    seed_index = 0
    for dr in range(sprite.shape[0]):
        for dc in range(sprite.shape[1]):
            color = int(sprite[dr, dc])
            if color == BACKGROUND:
                continue
            sr = r0 + dr
            sc = c0 + dc
            for off_r, off_c in START_OFFSETS:
                rr = sr + off_r
                cc = sc + off_c
                if not (0 <= cc < grid_shape[1]) or rr >= grid_shape[0]:
                    continue
                while rr < 0:
                    rr += 2
                while rr < grid_shape[0]:
                    hits[(rr, cc)].append((seed_index, color))
                    rr += 2
            seed_index += 1
    return hits


def _has_multicolor_overlap(sprite: np.ndarray, top_left: tuple[int, int], grid_shape: tuple[int, int]) -> bool:
    hits = _track_contributions(sprite, top_left, grid_shape)
    for values in hits.values():
        if len({seed for seed, _color in values}) >= 2 and len({color for _seed, color in values}) >= 2:
            return True
    return False


def _make_expected(grid_shape: tuple[int, int], anomaly_sprite: np.ndarray, anomaly_pos: tuple[int, int]) -> np.ndarray:
    return _render_from_selected_crop(grid_shape, anomaly_sprite, anomaly_pos)


def _validate_input_structure(grid: np.ndarray) -> tuple[bool, tuple[int, int] | None]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    if bg != BACKGROUND:
        return False, None
    objs = _find_components(g, bg)
    if not (5 <= len(objs) <= 7):
        return False, None
    crops = [crop_bbox(g, obj["bbox"]) for obj in objs]
    boxes = [crop.shape for crop in crops]
    if len(set(boxes)) != 1:
        return False, None
    if any(not (4 <= int(np.count_nonzero(crop)) <= 8) for crop in crops):
        return False, None
    if any(len(np.unique(crop[crop != BACKGROUND])) not in (2, 3) for crop in crops):
        return False, None
    mask_sigs = {_mask_signature(crop != BACKGROUND) for crop in crops}
    if len(mask_sigs) != 1:
        return False, None
    counts = Counter(_signature(crop) for crop in crops)
    if sorted(counts.values()) != [1, len(crops) - 1]:
        return False, None
    target_sig = next(sig for sig, n in counts.items() if n == 1)
    target_obj = next(obj for obj, crop in zip(objs, crops) if _signature(crop) == target_sig)
    r0, _c0, r1, _c1 = target_obj["bbox"]
    if r0 > g.shape[0] // 3:
        return False, None
    if g.shape[0] - 1 - r1 < 10:
        return False, None
    return True, (int(r0), int(target_obj["bbox"][1]))


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(20, 26)
    grid_w = random.randint(20, 26)
    box_h, box_w = random.choice(((3, 3), (3, 4), (4, 3), (4, 4)))
    num_sprites = random.randint(5, 7)
    mask = _random_mask(box_h, box_w)
    if mask is None:
        return None

    num_colors = 2 if random.random() < 0.55 else 3
    palette = random.sample(range(1, 10), num_colors)
    counts = _random_color_counts(int(np.count_nonzero(mask)), num_colors)
    repeated_sprite = _colorize_mask(mask, palette, counts)

    anomaly_sprite = repeated_sprite.copy()
    for _ in range(120):
        anomaly_sprite = _colorize_mask(mask, palette, counts)
        if not np.array_equal(anomaly_sprite, repeated_sprite):
            break
    if np.array_equal(anomaly_sprite, repeated_sprite):
        return None

    max_anomaly_row = min(grid_h // 3, grid_h - box_h - 10)
    if max_anomaly_row < 0:
        return None
    anomaly_row = random.randint(0, max_anomaly_row)
    edge_clip = random.random() < 0.4
    if edge_clip:
        anomaly_col = random.choice((0, grid_w - box_w))
    else:
        anomaly_col = random.randint(0, grid_w - box_w)
    anomaly_pos = (anomaly_row, anomaly_col)

    positions = _place_bboxes((grid_h, grid_w), (box_h, box_w), num_sprites, anomaly_pos)
    if positions is None:
        return None

    grid = np.zeros((grid_h, grid_w), dtype=int)
    for idx, (r, c) in enumerate(positions):
        sprite = anomaly_sprite if idx == 0 else repeated_sprite
        grid[r : r + box_h, c : c + box_w] = sprite

    expected = _make_expected(grid.shape, anomaly_sprite, anomaly_pos)
    valid, inferred_pos = _validate_input_structure(grid)
    if not valid or inferred_pos != anomaly_pos:
        return None
    if not np.array_equal(main(grid), expected):
        return None

    want_overlap = random.random() < 0.6
    has_overlap = _has_multicolor_overlap(anomaly_sprite, anomaly_pos, grid.shape)
    if want_overlap and not has_overlap:
        return None
    if edge_clip:
        if anomaly_col not in (0, grid_w - box_w):
            return None
    else:
        if anomaly_col in (0, grid_w - box_w):
            return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 2")
