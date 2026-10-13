import random
from collections import deque

import numpy as np


BACKGROUND = 0
FRAME_COLOR = 5
NOISE_COLOR = 6
CORNERS = ("TL", "TR", "BR", "BL")
CORNER_INDEX = {corner: idx for idx, corner in enumerate(CORNERS)}
SPRITE_COLORS = (1, 2, 3, 4)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GEN_ATTEMPTS = 500


FRAME_TEMPLATES = {
    1: [
        np.array(
            [
                [5, 5, 5, 5],
                [5, 0, 0, 5],
                [5, 0, 0, 5],
                [5, 5, 5, 5],
            ],
            dtype=int,
        ),
        np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 0, 0, 0, 5],
                [5, 0, 0, 0, 5],
                [5, 0, 0, 0, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=int,
        ),
    ],
    2: [
        np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 0, 5, 0, 5],
                [5, 0, 5, 0, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=int,
        ),
        np.array(
            [
                [5, 5, 5, 5],
                [5, 0, 0, 5],
                [5, 5, 5, 5],
                [5, 0, 0, 5],
                [5, 5, 5, 5],
            ],
            dtype=int,
        ),
    ],
    3: [
        np.array(
            [
                [5, 5, 5, 5, 5],
                [5, 0, 5, 0, 5],
                [5, 5, 5, 0, 5],
                [5, 0, 5, 0, 5],
                [5, 5, 5, 5, 5],
            ],
            dtype=int,
        )
    ],
}


CANONICAL_SPRITES = [
    {(0, 0), (0, 1), (1, 0)},
    {(0, 0), (1, 0), (1, 1)},
    {(0, 0), (0, 1), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (2, 0)},
    {(0, 0), (0, 1), (1, 0), (1, 1), (2, 1)},
    {(0, 0), (1, 0), (1, 1), (1, 2)},
    {(0, 0), (0, 1), (0, 2), (1, 1)},
    {(0, 0), (0, 1), (1, 1), (1, 2), (2, 2)},
]


def _bbox(coords):
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _connected_components(mask):
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _corner_from_bbox(bbox, h, w):
    r0, c0, r1, c1 = bbox
    if r0 == 0 and c0 == 0:
        return "TL"
    if r0 == 0 and c1 == w - 1:
        return "TR"
    if r1 == h - 1 and c1 == w - 1:
        return "BR"
    if r1 == h - 1 and c0 == 0:
        return "BL"
    return None


def _corner_origin(corner, sprite_h, sprite_w, h, w):
    if corner == "TL":
        return 0, 0
    if corner == "TR":
        return 0, w - sprite_w
    if corner == "BR":
        return h - sprite_h, w - sprite_w
    return h - sprite_h, 0


def _assign_corner_by_proximity(bbox, h, w):
    r0, c0, r1, c1 = bbox
    center_r = (r0 + r1) / 2.0
    center_c = (c0 + c1) / 2.0
    corner_points = {
        "TL": (0.0, 0.0),
        "TR": (0.0, float(w - 1)),
        "BR": (float(h - 1), float(w - 1)),
        "BL": (float(h - 1), 0.0),
    }
    return min(
        CORNERS,
        key=lambda corner: abs(center_r - corner_points[corner][0]) + abs(center_c - corner_points[corner][1]),
    )


def _component_infos_for_color(grid, color):
    comps = _connected_components(grid == color)
    infos = []
    for coords in comps:
        bbox = _bbox(coords)
        infos.append(
            {
                "coords": coords,
                "bbox": bbox,
                "color": int(color),
                "area": len(coords),
            }
        )
    return infos


def _count_rooms_in_crop(crop):
    black_comps = _connected_components(crop == BACKGROUND)
    count = 0
    crop_h, crop_w = crop.shape
    for comp in black_comps:
        if any(r == 0 or r == crop_h - 1 or c == 0 or c == crop_w - 1 for r, c in comp):
            continue
        count += 1
    return count


def _count_rooms_for_frame(grid, frame_info):
    r0, c0, r1, c1 = frame_info["bbox"]
    crop = grid[r0 : r1 + 1, c0 : c1 + 1]
    return _count_rooms_in_crop(crop)


def _extract_corner_sprites(grid):
    h, w = grid.shape
    sprites = {}
    for color in np.unique(grid):
        color = int(color)
        if color in (BACKGROUND, FRAME_COLOR, NOISE_COLOR):
            continue
        for comp in _component_infos_for_color(grid, color):
            corner = _corner_from_bbox(comp["bbox"], h, w)
            if corner is None:
                continue
            r0, c0, r1, c1 = comp["bbox"]
            local = sorted((r - r0, c - c0) for r, c in comp["coords"])
            sprites[corner] = {
                "color": color,
                "local_coords": local,
                "height": r1 - r0 + 1,
                "width": c1 - c0 + 1,
            }
    assert set(sprites) == set(CORNERS), "expected one colored corner sprite at each corner"
    return sprites


def _extract_corner_frames(grid):
    h, w = grid.shape
    frames = {}
    for comp in _component_infos_for_color(grid, FRAME_COLOR):
        corner = _assign_corner_by_proximity(comp["bbox"], h, w)
        assert corner not in frames, "expected one frame per corner quadrant"
        frames[corner] = comp
    assert set(frames) == set(CORNERS), "expected four grey partition frames"
    return frames


def _stamp_sprite(out, sprite, corner):
    h, w = out.shape
    r0, c0 = _corner_origin(corner, sprite["height"], sprite["width"], h, w)
    for dr, dc in sprite["local_coords"]:
        out[r0 + dr, c0 + dc] = sprite["color"]


def _render_output_from_parts(shape, sprites, room_counts):
    out = np.zeros(shape, dtype=int)
    for corner in CORNERS:
        _stamp_sprite(out, sprites[corner], corner)
    for corner in CORNERS:
        dst = CORNERS[(CORNER_INDEX[corner] + room_counts[corner]) % 4]
        _stamp_sprite(out, sprites[corner], dst)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Use each corner frame's enclosed-room count as a clockwise jump distance for copying the corner sprite.

    Input structure
    - A black background contains exactly four colored 4-connected sprites, one anchored at each true grid corner.
    - Near each corner sprite sits a separate grey orthogonal frame component whose walls partition enclosed black rooms.
    - Optional magenta distractor pixels may appear in the central area and do not participate.

    Rule
    - Ignore every magenta distractor.
    - For each grey frame, count how many 4-connected black regions are fully enclosed by its walls; this room count is an integer from 1 to 3.
    - Let the corners be ordered clockwise as top-left, top-right, bottom-right, bottom-left. Copy each corner sprite, preserving its exact pixel pattern and orientation, to the corner reached by jumping forward by its own room count in that order.
    - Keep all original sprites, stamp the copies in clockwise source-corner order, and erase every grey frame and magenta distractor from the final grid.

    Key insight
    - The grey frames are not output objects: they encode per-corner jump distances, so the visible answer is obtained only after counting enclosed rooms and then using that count to route each corner sprite clockwise.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    sprites = _extract_corner_sprites(g)
    frames = _extract_corner_frames(g)
    room_counts = {corner: _count_rooms_for_frame(g, frames[corner]) for corner in CORNERS}
    assert all(1 <= count <= 3 for count in room_counts.values())
    return _render_output_from_parts(g.shape, sprites, room_counts)


def _transform_local_coords(coords, corner):
    h = max(r for r, _ in coords) + 1
    w = max(c for _, c in coords) + 1
    if corner == "TL":
        transformed = coords
    elif corner == "TR":
        transformed = {(r, w - 1 - c) for r, c in coords}
    elif corner == "BR":
        transformed = {(h - 1 - r, w - 1 - c) for r, c in coords}
    else:
        transformed = {(h - 1 - r, c) for r, c in coords}
    return transformed, h, w


def _random_sprite_for_corner(corner):
    base = set(random.choice(CANONICAL_SPRITES))
    transformed, h, w = _transform_local_coords(base, corner)
    return sorted(transformed), h, w


def _place_sprite(grid, corner, color):
    local_coords, h, w = _random_sprite_for_corner(corner)
    r0, c0 = _corner_origin(corner, h, w, *grid.shape)
    absolute = [(r0 + dr, c0 + dc) for dr, dc in local_coords]
    for r, c in absolute:
        grid[r, c] = color
    return {
        "color": int(color),
        "local_coords": local_coords,
        "height": h,
        "width": w,
        "coords": absolute,
    }


def _frame_variants_for_count(count):
    if count != 3:
        return FRAME_TEMPLATES[count]
    base = FRAME_TEMPLATES[3][0]
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        if not any(np.array_equal(rotated, prev) for prev in variants):
            variants.append(rotated.copy())
    return variants


def _frame_candidates(corner, sprite_h, sprite_w, frame_h, frame_w, h, w):
    candidates = []
    for gap in (1, 2, 3):
        for shift in (0, 1, 2):
            if corner == "TL":
                candidates.extend(
                    [
                        (shift, sprite_w + gap),
                        (sprite_h + gap, shift),
                        (sprite_h + gap, sprite_w + gap),
                    ]
                )
            elif corner == "TR":
                candidates.extend(
                    [
                        (shift, w - sprite_w - gap - frame_w),
                        (sprite_h + gap, w - frame_w - shift),
                        (sprite_h + gap, w - sprite_w - gap - frame_w),
                    ]
                )
            elif corner == "BR":
                candidates.extend(
                    [
                        (h - frame_h - shift, w - sprite_w - gap - frame_w),
                        (h - sprite_h - gap - frame_h, w - frame_w - shift),
                        (h - sprite_h - gap - frame_h, w - sprite_w - gap - frame_w),
                    ]
                )
            else:
                candidates.extend(
                    [
                        (h - frame_h - shift, sprite_w + gap),
                        (h - sprite_h - gap - frame_h, shift),
                        (h - sprite_h - gap - frame_h, sprite_w + gap),
                    ]
                )
    return candidates


def _bboxes_overlap(a, b):
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 < br0 or br1 < ar0 or ac1 < bc0 or bc1 < ac0)


def _place_frame(grid, occupancy, corner, sprite_info, room_count):
    template = random.choice(_frame_variants_for_count(room_count))
    frame_h, frame_w = template.shape
    h, w = grid.shape
    sprite_bbox = _bbox(sprite_info["coords"])
    for r0, c0 in _frame_candidates(corner, sprite_info["height"], sprite_info["width"], frame_h, frame_w, h, w):
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
            continue
        frame_bbox = (r0, c0, r1, c1)
        if _bboxes_overlap(sprite_bbox, frame_bbox):
            continue
        ok = True
        for rr in range(frame_h):
            for cc in range(frame_w):
                if template[rr, cc] != FRAME_COLOR:
                    continue
                r = r0 + rr
                c = c0 + cc
                if occupancy[r, c]:
                    ok = False
                    break
                for dr, dc in NEIGHBORS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and grid[nr, nc] == FRAME_COLOR:
                        ok = False
                        break
                if not ok:
                    break
            if not ok:
                break
        if not ok:
            continue
        grid[r0 : r1 + 1, c0 : c1 + 1] = np.where(template == FRAME_COLOR, FRAME_COLOR, grid[r0 : r1 + 1, c0 : c1 + 1])
        occupancy[r0 : r1 + 1, c0 : c1 + 1] |= template == FRAME_COLOR
        return {"bbox": frame_bbox, "room_count": room_count, "template": template.copy()}
    return None


def _noise_shapes():
    return [
        [(0, 0)],
        [(0, 0), (0, 1)],
        [(0, 0), (1, 0)],
        [(0, 0), (0, 1), (1, 0)],
    ]


def _add_noise(grid, occupancy):
    h, w = grid.shape
    if h < 16 or w < 16:
        return
    n_clusters = random.choices([0, 1, 2, 3, 4], weights=[1, 2, 3, 2, 1])[0]
    min_r, max_r = 4, h - 5
    min_c, max_c = 4, w - 5
    if min_r > max_r or min_c > max_c:
        return
    for _ in range(n_clusters):
        placed = False
        for _ in range(40):
            shape = random.choice(_noise_shapes())
            base_r = random.randint(min_r, max_r)
            base_c = random.randint(min_c, max_c)
            cells = [(base_r + dr, base_c + dc) for dr, dc in shape]
            if any(not (min_r <= r <= max_r and min_c <= c <= max_c) for r, c in cells):
                continue
            if any(occupancy[r, c] for r, c in cells):
                continue
            for r, c in cells:
                grid[r, c] = NOISE_COLOR
                occupancy[r, c] = True
            placed = True
            break
        if not placed:
            continue


def _build_case():
    h = random.randint(18, 24)
    w = random.randint(18, 24)
    grid = np.zeros((h, w), dtype=int)
    occupancy = np.zeros((h, w), dtype=bool)

    counts = [1, 2, 3, random.choice([1, 2, 3])]
    random.shuffle(counts)
    room_counts = {corner: counts[idx] for idx, corner in enumerate(CORNERS)}

    colors = list(SPRITE_COLORS)
    random.shuffle(colors)
    sprites = {}
    for corner, color in zip(CORNERS, colors):
        sprite = _place_sprite(grid, corner, color)
        sprites[corner] = {
            "color": sprite["color"],
            "local_coords": sprite["local_coords"],
            "height": sprite["height"],
            "width": sprite["width"],
        }
        for r, c in sprite["coords"]:
            occupancy[r, c] = True

    frames = {}
    for corner in CORNERS:
        frame = _place_frame(grid, occupancy, corner, sprites[corner] | {"coords": [
            (_corner_origin(corner, sprites[corner]["height"], sprites[corner]["width"], h, w)[0] + dr,
             _corner_origin(corner, sprites[corner]["height"], sprites[corner]["width"], h, w)[1] + dc)
            for dr, dc in sprites[corner]["local_coords"]
        ]}, room_counts[corner])
        if frame is None:
            return None
        frames[corner] = frame

    _add_noise(grid, occupancy)
    expected = _render_output_from_parts(grid.shape, sprites, room_counts)
    return grid, expected, sprites, room_counts, frames


def _sprite_only_grid(shape, sprites):
    out = np.zeros(shape, dtype=int)
    for corner in CORNERS:
        _stamp_sprite(out, sprites[corner], corner)
    return out


def _semantic_checks(grid, expected, sprites, room_counts):
    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False
    if np.any(solved == FRAME_COLOR) or np.any(solved == NOISE_COLOR):
        return False
    if np.array_equal(solved, _sprite_only_grid(grid.shape, sprites)):
        return False
    if len(set(room_counts.values())) < 3:
        return False

    parsed_sprites = _extract_corner_sprites(grid)
    parsed_frames = _extract_corner_frames(grid)
    parsed_counts = {corner: _count_rooms_for_frame(grid, parsed_frames[corner]) for corner in CORNERS}
    if parsed_counts != room_counts:
        return False
    for corner in CORNERS:
        if parsed_sprites[corner]["color"] != sprites[corner]["color"]:
            return False
        if parsed_sprites[corner]["local_coords"] != sprites[corner]["local_coords"]:
            return False

    without_noise = grid.copy()
    without_noise[without_noise == NOISE_COLOR] = BACKGROUND
    if not np.array_equal(main(without_noise), solved):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        built = _build_case()
        if built is None:
            continue
        grid, expected, sprites, room_counts, _frames = built
        if _semantic_checks(grid, expected, sprites, room_counts):
            return grid
    raise RuntimeError("failed to generate a valid puzzle input")
