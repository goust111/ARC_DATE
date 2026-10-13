from __future__ import annotations

import random

import numpy as np


BG = 0
YELLOW = 4
GREY = 5
SPRITE_COLORS = [1, 2, 3, 6, 7, 8, 9]
N4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _is_sparse_rect_perimeter(coords: np.ndarray) -> bool:
    if coords.size == 0:
        return False
    r0, c0, r1, c1 = _bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    if np.any(rel % 2 != 0):
        return False

    h = (r1 - r0) // 2 + 1
    w = (c1 - c0) // 2 + 1
    expected = 2 * h + 2 * w - 4
    if coords.shape[0] != expected:
        return False

    points = {tuple(map(int, p)) for p in coords}
    for rr in range(r0, r1 + 1, 2):
        for cc in range(c0, c1 + 1, 2):
            on_border = rr in (r0, r1) or cc in (c0, c1)
            if ((rr, cc) in points) != on_border:
                return False
    return True


def _solidify_sparse_coords(coords: np.ndarray) -> np.ndarray:
    r0, c0, r1, c1 = _bbox_coords(coords)
    rel = coords - np.array([r0, c0])
    assert np.all(rel % 2 == 0), "sparse sprite must lie on a two-step lattice"

    h = (r1 - r0) // 2 + 1
    w = (c1 - c0) // 2 + 1
    logical = np.zeros((h, w), dtype=bool)
    for dr, dc in rel:
        logical[int(dr) // 2, int(dc) // 2] = True
    return np.kron(logical, np.ones((2, 2), dtype=bool))[: 2 * h - 1, : 2 * w - 1]


def _rotate_cw(mask: np.ndarray, turns: int) -> np.ndarray:
    k = (-int(turns)) % 4
    return np.rot90(mask, k=k)


def _turns_from_marker(marker: tuple[int, int], bbox: tuple[int, int, int, int]) -> int:
    mr, mc = marker
    r0, c0, r1, c1 = bbox
    if mr == r0 - 1 and c0 <= mc <= c1:
        return 0
    if mc == c1 + 1 and r0 <= mr <= r1:
        return 1
    if mr == r1 + 1 and c0 <= mc <= c1:
        return 2
    if mc == c0 - 1 and r0 <= mr <= r1:
        return 3
    raise AssertionError("marker must sit exactly one step outside one side of the sprite bbox")


def _paint_mask(out: np.ndarray, mask: np.ndarray, top: int, left: int, color: int) -> None:
    h, w = mask.shape
    patch = out[top : top + h, left : left + w]
    patch[mask] = int(color)


def _parse_scene(grid: np.ndarray) -> tuple[tuple[int, int, int, int], list[dict]]:
    frame_coords = np.argwhere(grid == GREY)
    assert _is_sparse_rect_perimeter(frame_coords), "expected one sparse grey rectangular frame"
    frame_bbox = _bbox_coords(frame_coords)
    r0, c0, r1, c1 = frame_bbox
    assert (c1 - c0) > (r1 - r0), "frame must be wide"

    marker_points = [tuple(map(int, p)) for p in np.argwhere(grid == YELLOW)]
    remaining_markers = set(marker_points)

    sprite_colors = [int(c) for c in np.unique(grid) if int(c) not in {BG, YELLOW, GREY}]
    assert 2 <= len(sprite_colors) <= 4, "expected 2-4 external sprites"

    sprites = []
    for color in sprite_colors:
        coords = np.argwhere(grid == color)
        bbox = _bbox_coords(coords)
        marker_matches = [
            mp
            for mp in remaining_markers
            if (
                (mp[0] == bbox[0] - 1 and bbox[1] <= mp[1] <= bbox[3])
                or (mp[1] == bbox[3] + 1 and bbox[0] <= mp[0] <= bbox[2])
                or (mp[0] == bbox[2] + 1 and bbox[1] <= mp[1] <= bbox[3])
                or (mp[1] == bbox[1] - 1 and bbox[0] <= mp[0] <= bbox[2])
            )
        ]
        assert len(marker_matches) == 1, "each sprite needs exactly one side marker"
        marker = marker_matches[0]
        remaining_markers.remove(marker)

        dense = _solidify_sparse_coords(coords)
        turns = _turns_from_marker(marker, bbox)
        rotated = _rotate_cw(dense, turns)
        sprites.append(
            {
                "color": color,
                "bbox": bbox,
                "marker": marker,
                "dense": dense,
                "turns": turns,
                "rotated": rotated,
            }
        )

    assert not remaining_markers, "all yellow markers must belong to a sprite"
    sprites.sort(key=lambda sp: (sp["bbox"][1], sp["bbox"][0]))
    return frame_bbox, sprites


def _render_output(shape: tuple[int, int], frame_bbox: tuple[int, int, int, int], sprites: list[dict]) -> np.ndarray:
    out = np.full(shape, BG, dtype=int)
    r0, c0, r1, c1 = frame_bbox
    out[r0, c0 : c1 + 1] = GREY
    out[r1, c0 : c1 + 1] = GREY
    out[r0 : r1 + 1, c0] = GREY
    out[r0 : r1 + 1, c1] = GREY

    inner_h = r1 - r0 - 1
    inner_w = c1 - c0 - 1
    total_w = sum(sp["rotated"].shape[1] for sp in sprites)
    gap = (inner_w - total_w) // (len(sprites) + 1)
    assert gap >= 0, "frame interior is too narrow"
    cursor = c0 + 1 + gap

    for sp in sprites:
        mask = sp["rotated"]
        ph, pw = mask.shape
        top = r0 + 1 + (inner_h - ph) // 2
        assert top >= r0 + 1 and top + ph <= r1
        _paint_mask(out, mask, top, cursor, sp["color"])
        cursor += pw + gap

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Densify the external sparse sprites, rotate each by the side of its yellow marker, and place them into the thickened frame.

    Input structure:
    - Black background with one wide sparse grey rectangular frame drawn on a two-step lattice.
    - Outside the frame are 2-4 monochrome sparse sprites, each in its own non-grey non-yellow color on the same lattice.
    - Each sprite has exactly one yellow pixel placed one cell outside one side of that sprite's bounding box.

    Rule:
    - Replace the sparse grey frame by a solid one-pixel-wide grey rectangle on the same bounding box.
    - Densify every sparse sprite by filling the one-cell gaps implied by the two-step lattice.
    - Convert the yellow marker side into a clockwise rotation: above=0, right=90, below=180, left=270.
    - Sort sprites by source bounding-box minimum column, breaking ties by minimum row, then place the rotated dense sprites left-to-right inside the frame with equal floor-divided gaps and vertical centering.

    Key insight:
    - The yellow pixels do not carry different colors; their relative side position around each sparse sprite is the hidden code for the sprite's rotation before the ordered in-frame packing step.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frame_bbox, sprites = _parse_scene(g)
    return _render_output(g.shape, frame_bbox, sprites)


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out = []
    for dr, dc in N4:
        rr = r + dr
        cc = c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _random_connected_mask(h: int, w: int) -> np.ndarray:
    for _ in range(200):
        target = random.randint(max(3, (h * w) // 2), h * w - 1)
        mask = np.zeros((h, w), dtype=bool)
        seed = (random.randrange(h), random.randrange(w))
        mask[seed] = True
        frontier = set(_neighbors4(seed[0], seed[1], h, w))

        while int(mask.sum()) < target and frontier:
            rr, cc = random.choice(tuple(frontier))
            frontier.remove((rr, cc))
            if mask[rr, cc]:
                continue
            if not any(mask[nr, nc] for nr, nc in _neighbors4(rr, cc, h, w)):
                continue
            mask[rr, cc] = True
            for nxt in _neighbors4(rr, cc, h, w):
                if not mask[nxt]:
                    frontier.add(nxt)

        if int(mask.sum()) != target:
            continue
        if not (mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()):
            continue
        return mask
    raise RuntimeError("failed to generate a connected logical sprite mask")


def _mask_to_sparse_coords(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    coords = []
    for rr, cc in np.argwhere(mask):
        coords.append((top + 2 * int(rr), left + 2 * int(cc)))
    return coords


def _stamp_sparse_frame(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = bbox
    for rr in range(r0, r1 + 1, 2):
        grid[rr, c0] = GREY
        grid[rr, c1] = GREY
    for cc in range(c0, c1 + 1, 2):
        grid[r0, cc] = GREY
        grid[r1, cc] = GREY


def _odd_between(lo: int, hi: int) -> int:
    vals = [v for v in range(lo, hi + 1) if v % 2 == 1]
    if not vals:
        raise ValueError("no odd value in range")
    return random.choice(vals)


def _build_input(
    logical_masks: list[np.ndarray],
    colors: list[int],
    turns: list[int],
    frame_bbox: tuple[int, int, int, int],
    sprite_positions: list[tuple[int, int]],
) -> np.ndarray:
    h = max(frame_bbox[2] + 2, max(r + 2 * mask.shape[0] for (r, _), mask in zip(sprite_positions, logical_masks)))
    w = max(frame_bbox[3] + 2, max(c + 2 * mask.shape[1] for (_, c), mask in zip(sprite_positions, logical_masks)))
    grid = np.full((h, w), BG, dtype=int)
    _stamp_sparse_frame(grid, frame_bbox)

    for mask, color, turn, (top, left) in zip(logical_masks, colors, turns, sprite_positions):
        sparse_coords = _mask_to_sparse_coords(mask, top, left)
        for rr, cc in sparse_coords:
            grid[rr, cc] = int(color)

        r0, c0 = top, left
        r1 = top + 2 * (mask.shape[0] - 1)
        c1 = left + 2 * (mask.shape[1] - 1)
        if turn == 0:
            marker = (r0 - 1, random.randrange(c0 + 1, c1))
        elif turn == 1:
            marker = (random.randrange(r0 + 1, r1), c1 + 1)
        elif turn == 2:
            marker = (r1 + 1, random.randrange(c0 + 1, c1))
        else:
            marker = (random.randrange(r0 + 1, r1), c0 - 1)
        grid[marker] = YELLOW

    return grid


def _choose_dims(n_sprites: int) -> tuple[int, int]:
    options = {
        2: [(2, 3), (2, 4), (2, 5), (3, 4), (3, 5)],
        3: [(2, 3), (2, 4), (3, 4)],
        4: [(2, 3), (2, 4)],
    }
    return random.choice(options[n_sprites])


def _generate_case() -> tuple[np.ndarray, np.ndarray]:
    n_sprites = random.randint(2, 4)
    logical_h, logical_w = _choose_dims(n_sprites)
    sparse_h = 2 * logical_h - 1
    sparse_w = 2 * logical_w - 1

    logical_masks = [_random_connected_mask(logical_h, logical_w) for _ in range(n_sprites)]
    colors = random.sample(SPRITE_COLORS, n_sprites)
    while True:
        turns = random.sample([0, 1, 2, 3], n_sprites)
        if any(turn in {1, 3} for turn in turns):
            break
    dense_masks = [_solidify_sparse_coords(np.argwhere(mask) * 2) for mask in logical_masks]
    rotated_masks = [_rotate_cw(mask, turn) for mask, turn in zip(dense_masks, turns)]

    max_rot_h = max(mask.shape[0] for mask in rotated_masks)
    total_rot_w = sum(mask.shape[1] for mask in rotated_masks)
    gap = random.randint(1, 2)
    tail_extra = random.randint(0, n_sprites)
    inner_h = _odd_between(max_rot_h, min(max_rot_h + 4, 11))
    inner_w = total_rot_w + (n_sprites + 1) * gap + tail_extra
    if inner_w % 2 == 0:
        inner_w += 1
    frame_h = inner_h + 2
    frame_w = inner_w + 2

    strip_gap_choices = [1, 2] if n_sprites >= 3 else [1, 2, 3]
    strip_gap = random.choice(strip_gap_choices)
    strip_w = n_sprites * sparse_w + (n_sprites - 1) * strip_gap
    band_h = sparse_h + 2
    v_gap = random.randint(1, 3)
    side_margin = random.randint(1, 2)
    min_w = max(frame_w + 2, strip_w + 2 * side_margin)
    min_h = frame_h + band_h + v_gap + 2
    if min_w > 30 or min_h > 30:
        raise RuntimeError("layout exceeds ARC size limit")

    canvas_w = random.randint(min_w, 30)
    canvas_h = random.randint(min_h, 30)
    strip_left = random.randint(side_margin, canvas_w - strip_w - side_margin)
    strip_top_mode = random.choice(["above", "below"])

    if strip_top_mode == "above":
        band_top = random.randint(1, canvas_h - frame_h - band_h - v_gap - 1)
        frame_top_min = band_top + band_h + v_gap
        frame_top = random.randint(frame_top_min, canvas_h - frame_h - 1)
    else:
        frame_top = random.randint(1, canvas_h - frame_h - band_h - v_gap - 1)
        band_top_min = frame_top + frame_h + v_gap
        band_top = random.randint(band_top_min, canvas_h - band_h - 1)

    frame_left = random.randint(1, canvas_w - frame_w - 1)
    sprite_positions = [(band_top + 1, strip_left + i * (sparse_w + strip_gap)) for i in range(n_sprites)]
    frame_bbox = (frame_top, frame_left, frame_top + frame_h - 1, frame_left + frame_w - 1)

    inp = _build_input(logical_masks, colors, turns, frame_bbox, sprite_positions)
    assert inp.shape[0] <= 30 and inp.shape[1] <= 30

    frame_out_bbox, parsed_sprites = _parse_scene(inp)
    expected = _render_output(inp.shape, frame_out_bbox, parsed_sprites)

    no_rotate = []
    for sp in parsed_sprites:
        no_rotate.append({**sp, "rotated": sp["dense"]})
    no_rotate_out = _render_output(inp.shape, frame_out_bbox, no_rotate)
    assert not np.array_equal(no_rotate_out, expected), "markers must affect the result"

    color_sorted = sorted(parsed_sprites, key=lambda sp: sp["color"])
    color_sorted_out = _render_output(inp.shape, frame_out_bbox, color_sorted)
    x_order_colors = [sp["color"] for sp in parsed_sprites]
    assert x_order_colors != sorted(x_order_colors), "source order must not coincide with ascending color order"
    assert x_order_colors != sorted(x_order_colors, reverse=True), "source order must not coincide with descending color order"
    assert not np.array_equal(color_sorted_out, expected), "position order must affect the result"

    out = main(inp)
    assert np.array_equal(out, expected)
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, _expected = _generate_case()
        except Exception:
            continue
        return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
