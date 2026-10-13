from __future__ import annotations

from collections import Counter
import random

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _coords_to_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[tuple[int, int], bytes]:
    return tuple(map(int, mask.shape)), mask.astype(np.uint8).tobytes()


def _dilate8(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[:-1, :] |= mask[1:, :]
    out[1:, :] |= mask[:-1, :]
    out[:, :-1] |= mask[:, 1:]
    out[:, 1:] |= mask[:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    return out


def _paste_mask(grid: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    view = grid[top : top + h, left : left + w]
    view[mask] = int(color)


def _place_masks(
    canvas_shape: tuple[int, int],
    masks: list[np.ndarray],
    initial_occupied: np.ndarray | None = None,
) -> list[tuple[int, int]] | None:
    h, w = canvas_shape
    occupied = np.zeros((h, w), dtype=bool) if initial_occupied is None else initial_occupied.copy()
    order = sorted(range(len(masks)), key=lambda i: masks[i].shape[0] * masks[i].shape[1], reverse=True)
    placements: list[tuple[int, int] | None] = [None] * len(masks)

    for idx in order:
        mask = masks[idx]
        mh, mw = mask.shape
        blocked = _dilate8(occupied)
        candidates: list[tuple[int, int]] = []
        for top in range(h - mh + 1):
            for left in range(w - mw + 1):
                if not bool((blocked[top : top + mh, left : left + mw] & mask).any()):
                    candidates.append((int(top), int(left)))
        if not candidates:
            return None
        top, left = random.choice(candidates)
        placements[idx] = (top, left)
        occupied[top : top + mh, left : left + mw] |= mask

    return [p for p in placements if p is not None]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each hole in the large rectangle using the color of the outside piece with the same shape.
    """
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    comps = _components(g != bg)
    assert comps, "expected at least one foreground component"

    comp_infos = []
    for coords in comps:
        mask, bbox = _coords_to_mask(coords)
        colors = [int(g[r, c]) for r, c in coords]
        color = int(Counter(colors).most_common(1)[0][0])
        comp_infos.append(
            {
                "coords": coords,
                "mask": mask,
                "bbox": bbox,
                "key": _shape_key(mask),
                "color": color,
                "area": int(len(coords)),
            }
        )

    frame = max(comp_infos, key=lambda item: item["area"])
    fr0, fc0, fr1, fc1 = frame["bbox"]
    sub = g[fr0 : fr1 + 1, fc0 : fc1 + 1]

    holes = []
    for local_coords in _components(sub == bg):
        if any(r == 0 or c == 0 or r == sub.shape[0] - 1 or c == sub.shape[1] - 1 for r, c in local_coords):
            continue
        hole_mask, local_bbox = _coords_to_mask(local_coords)
        hr0, hc0, _, _ = local_bbox
        abs_coords = [(fr0 + r, fc0 + c) for r, c in local_coords]
        holes.append(
            {
                "coords": abs_coords,
                "anchor": (fr0 + hr0, fc0 + hc0),
                "key": _shape_key(hole_mask),
            }
        )

    pieces = [info for info in comp_infos if info is not frame]
    piece_map: dict[tuple[tuple[int, int], bytes], list[dict]] = {}
    for piece in pieces:
        piece_map.setdefault(piece["key"], []).append(piece)

    out = g.copy()
    for piece in pieces:
        for r, c in piece["coords"]:
            out[r, c] = bg

    holes.sort(key=lambda item: item["anchor"])
    for hole in holes:
        matches = piece_map.get(hole["key"], [])
        assert matches, "every hole must have a matching piece"
        piece = matches.pop()
        for r, c in hole["coords"]:
            out[r, c] = int(piece["color"])

    return out


MIN_GRID_SIZE = 18
MAX_GRID_SIZE = 30
INNER_H_RANGE = (6, 13)
INNER_W_RANGE = (6, 13)
HOLE_COUNT_RANGE = (2, 5)
SHAPE_BOX_RANGE = (1, 5)
SHAPE_AREA_RANGE = (1, 8)
MAX_GENERATION_ATTEMPTS = 400
SHAPE_ATTEMPTS = 60


def _grow_shape(height: int, width: int, area: int) -> np.ndarray | None:
    cells = {(random.randrange(height), random.randrange(width))}
    while len(cells) < area:
        frontier = set()
        for r, c in cells:
            for dr, dc in NEIGHBORS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        cells.add(random.choice(tuple(frontier)))

    rs = [r for r, _ in cells]
    cs = [c for _, c in cells]
    mask = np.zeros((max(rs) - min(rs) + 1, max(cs) - min(cs) + 1), dtype=bool)
    for r, c in cells:
        mask[r - min(rs), c - min(cs)] = True
    return mask


def _random_shape(max_h: int, max_w: int) -> np.ndarray | None:
    for _ in range(SHAPE_ATTEMPTS):
        box_h = random.randint(1, min(SHAPE_BOX_RANGE[1], max_h))
        box_w = random.randint(1, min(SHAPE_BOX_RANGE[1], max_w))
        max_area = min(SHAPE_AREA_RANGE[1], box_h * box_w)
        if max_area <= 0:
            continue
        if max_area == 1:
            area = 1
        else:
            lo = 1 if random.random() < 0.15 else 2
            area = random.randint(lo, max_area)
        mask = _grow_shape(box_h, box_w, area)
        if mask is None:
            continue
        if area > 1 and bool(mask.all()) and random.random() < 0.65:
            continue
        return mask
    return None


def _sample_hole_masks(inner_h: int, inner_w: int, count: int) -> list[np.ndarray] | None:
    masks: list[np.ndarray] = []
    seen = set()
    for _ in range(count):
        mask = None
        for _ in range(SHAPE_ATTEMPTS):
            candidate = _random_shape(inner_h, inner_w)
            if candidate is None:
                continue
            if candidate.shape[0] > inner_h or candidate.shape[1] > inner_w:
                continue
            key = _shape_key(candidate)
            if key in seen:
                continue
            mask = candidate
            seen.add(key)
            break
        if mask is None:
            return None
        masks.append(mask)
    return masks


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    hole_count = random.randint(HOLE_COUNT_RANGE[0], HOLE_COUNT_RANGE[1])
    inner_h = random.randint(INNER_H_RANGE[0], INNER_H_RANGE[1])
    inner_w = random.randint(INNER_W_RANGE[0], INNER_W_RANGE[1])
    outer_h = inner_h + 2
    outer_w = inner_w + 2

    hole_masks = _sample_hole_masks(inner_h, inner_w, hole_count)
    if hole_masks is None:
        return None
    hole_positions = _place_masks((inner_h, inner_w), hole_masks)
    if hole_positions is None:
        return None

    grid_candidates: list[tuple[int, int]] = []
    min_h = max(MIN_GRID_SIZE, outer_h + 6)
    min_w = max(MIN_GRID_SIZE, outer_w + 6)
    for h in range(min_h, MAX_GRID_SIZE + 1):
        for w in range(min_w, MAX_GRID_SIZE + 1):
            if h * w > 2 * outer_h * outer_w + 10:
                grid_candidates.append((h, w))
    if not grid_candidates:
        return None
    grid_h, grid_w = random.choice(grid_candidates)

    frame_r = random.randint(2, grid_h - outer_h - 2)
    frame_c = random.randint(2, grid_w - outer_w - 2)

    colors = random.sample(range(10), hole_count + 2)
    bg = int(colors[0])
    frame_color = int(colors[1])
    piece_colors = [int(c) for c in colors[2:]]

    inp = np.full((grid_h, grid_w), bg, dtype=int)
    out = np.full((grid_h, grid_w), bg, dtype=int)
    inp[frame_r : frame_r + outer_h, frame_c : frame_c + outer_w] = frame_color
    out[frame_r : frame_r + outer_h, frame_c : frame_c + outer_w] = frame_color

    frame_block = np.zeros((grid_h, grid_w), dtype=bool)
    frame_block[frame_r : frame_r + outer_h, frame_c : frame_c + outer_w] = True

    for mask, (top, left), color in zip(hole_masks, hole_positions, piece_colors, strict=True):
        _paste_mask(inp, mask, frame_r + 1 + top, frame_c + 1 + left, bg)
        _paste_mask(out, mask, frame_r + 1 + top, frame_c + 1 + left, color)

    piece_positions = _place_masks((grid_h, grid_w), hole_masks, initial_occupied=frame_block)
    if piece_positions is None:
        return None
    for mask, (top, left), color in zip(hole_masks, piece_positions, piece_colors, strict=True):
        _paste_mask(inp, mask, top, left, color)

    counts = Counter(int(x) for x in inp.flat)
    other_counts = [count for color, count in counts.items() if color != bg]
    if not other_counts or counts[bg] <= max(other_counts):
        return None

    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        built = _build_case()
        if built is not None:
            inp, _ = built
            return inp
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
