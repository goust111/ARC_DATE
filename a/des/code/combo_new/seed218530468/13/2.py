import random
from collections import deque

import numpy as np


AZURE = 8
ARC_MAX = 30
MAX_GENERATION_ATTEMPTS = 400
ORTHOGONAL_STEPS = ((1, 0), (-1, 0), (0, 1), (0, -1))
COLOR_POOL = (0, 1, 2, 3, 4, 5, 6, 7, 9)


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = map(int, mask.shape)
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []
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
                for dr, dc in ORTHOGONAL_STEPS:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _is_connected(mask: np.ndarray) -> bool:
    return len(_component_coords(mask)) == 1


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _extract_objects(grid: np.ndarray) -> list[dict]:
    h, w = map(int, grid.shape)
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == AZURE or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ORTHOGONAL_STEPS:
                    nr, nc = rr + dr, cc + dc
                    if (
                        0 <= nr < h
                        and 0 <= nc < w
                        and not seen[nr, nc]
                        and int(grid[nr, nc]) == color
                    ):
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0, r1, c1 = _bbox_from_coords(arr)
            local = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
            local[arr[:, 0] - r0, arr[:, 1] - c0] = True
            objects.append(
                {
                    "color": color,
                    "bbox": (r0, c0, r1, c1),
                    "mask": local,
                }
            )
    return objects


def _corner_offsets(full_h: int, full_w: int, h: int, w: int) -> list[tuple[int, int]]:
    offsets = [
        (0, 0),
        (0, full_w - w),
        (full_h - h, 0),
        (full_h - h, full_w - w),
    ]
    ordered: list[tuple[int, int]] = []
    seen = set()
    for offset in offsets:
        if offset not in seen:
            seen.add(offset)
            ordered.append(offset)
    return ordered


def _place_mask(mask: np.ndarray, full_h: int, full_w: int, offset: tuple[int, int]) -> np.ndarray:
    h, w = map(int, mask.shape)
    r0, c0 = offset
    canvas = np.zeros((full_h, full_w), dtype=bool)
    canvas[r0 : r0 + h, c0 : c0 + w] = mask
    return canvas


def _pair_match(a: dict, b: dict) -> dict | None:
    if int(a["color"]) == int(b["color"]):
        return None

    ah, aw = map(int, a["mask"].shape)
    bh, bw = map(int, b["mask"].shape)
    full_h = max(ah, bh)
    full_w = max(aw, bw)

    for off_a in _corner_offsets(full_h, full_w, ah, aw):
        placed_a = _place_mask(a["mask"], full_h, full_w, off_a)
        for off_b in _corner_offsets(full_h, full_w, bh, bw):
            placed_b = _place_mask(b["mask"], full_h, full_w, off_b)
            if np.any(placed_a & placed_b):
                continue
            if not np.all(placed_a | placed_b):
                continue
            return {
                "full_shape": (full_h, full_w),
                "placed_a": placed_a,
                "placed_b": placed_b,
            }
    return None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair complementary colored fragments, rebuild each full rectangle, and stack the rectangles by area.

    Input structure:
    - The background is azure.
    - The grid contains an even number of separated monochrome 4-connected fragments.
    - Fragments come in pairs: after aligning their local bounding rectangles, the occupied cells are complementary parts of one solid rectangle.
    - Different pairs correspond to different rectangle sizes.

    Rule:
    - Extract every non-background connected component.
    - Find pairs whose aligned local masks fill one rectangle without overlap or gaps.
    - Repaint each paired rectangle with the original two fragment colors.
    - Sort the rebuilt rectangles by area from small to large and stack them top-to-bottom, left aligned, with one azure row between rectangles.

    Key insight:
    - Colors do not determine the pairing; the decisive signal is that two fragments become exact complementary masks once their local frames are aligned into one common rectangle.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = _extract_objects(g)
    assert objects and len(objects) % 2 == 0, "expected an even number of fragments"

    used: set[int] = set()
    rebuilt: list[np.ndarray] = []
    for i, obj_a in enumerate(objects):
        if i in used:
            continue
        match_index = None
        match_data = None
        for j in range(i + 1, len(objects)):
            if j in used:
                continue
            candidate = _pair_match(obj_a, objects[j])
            if candidate is not None:
                match_index = j
                match_data = candidate
                break
        assert match_index is not None and match_data is not None, "every fragment must have exactly one complement"

        full_h, full_w = match_data["full_shape"]
        rect = np.full((full_h, full_w), AZURE, dtype=int)
        rect[match_data["placed_a"]] = int(obj_a["color"])
        rect[match_data["placed_b"]] = int(objects[match_index]["color"])
        assert np.all(rect != AZURE), "paired fragments must fill a solid rectangle"

        rebuilt.append(rect)
        used.add(i)
        used.add(match_index)

    assert len(used) == len(objects), "all fragments must be paired"

    rebuilt.sort(key=lambda arr: (int(arr.shape[0] * arr.shape[1]), int(arr.shape[0]), int(arr.shape[1])))
    out_h = sum(int(arr.shape[0]) for arr in rebuilt) + len(rebuilt) - 1
    out_w = max(int(arr.shape[1]) for arr in rebuilt)
    assert out_h <= ARC_MAX and out_w <= ARC_MAX

    out = np.full((out_h, out_w), AZURE, dtype=int)
    r = 0
    for arr in rebuilt:
        h, w = map(int, arr.shape)
        out[r : r + h, :w] = arr
        r += h + 1
    return out


def _draw_path(mask: np.ndarray, start: tuple[int, int], end: tuple[int, int]) -> None:
    r, c = start
    tr, tc = end
    if random.random() < 0.5:
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
    else:
        while c != tc:
            mask[r, c] = True
            c += 1 if tc > c else -1
        while r != tr:
            mask[r, c] = True
            r += 1 if tr > r else -1
    mask[tr, tc] = True


def _grow_blob(sub_h: int, sub_w: int) -> np.ndarray | None:
    if sub_h < 2 or sub_w < 2:
        return None

    mask = np.zeros((sub_h, sub_w), dtype=bool)
    bottom_target = (sub_h - 1, random.randrange(sub_w))
    right_target = (random.randrange(sub_h), sub_w - 1)
    _draw_path(mask, (0, 0), bottom_target)

    existing = [tuple(map(int, rc)) for rc in np.argwhere(mask)]
    _draw_path(mask, random.choice(existing), right_target)

    current = int(mask.sum())
    upper = min(sub_h * sub_w - 1, max(current, (sub_h * sub_w * 2) // 3))
    target = current if upper <= current else random.randint(current, upper)
    while int(mask.sum()) < target:
        frontier = []
        for r, c in np.argwhere(mask):
            for dr, dc in ORTHOGONAL_STEPS:
                nr, nc = int(r + dr), int(c + dc)
                if 0 <= nr < sub_h and 0 <= nc < sub_w and not mask[nr, nc]:
                    frontier.append((nr, nc))
        if not frontier:
            break
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True

    if not _is_connected(mask):
        return None
    if not (
        np.any(mask[0, :])
        and np.any(mask[-1, :])
        and np.any(mask[:, 0])
        and np.any(mask[:, -1])
    ):
        return None
    if int(mask.sum()) >= sub_h * sub_w:
        return None
    return mask


def _orient_corner(mask: np.ndarray, corner: str) -> np.ndarray:
    if corner == "tl":
        return mask.copy()
    if corner == "tr":
        return np.fliplr(mask)
    if corner == "bl":
        return np.flipud(mask)
    if corner == "br":
        return np.flipud(np.fliplr(mask))
    raise ValueError(f"unknown corner {corner}")


def _embed_corner(mask: np.ndarray, full_h: int, full_w: int, corner: str) -> np.ndarray:
    h, w = map(int, mask.shape)
    full = np.zeros((full_h, full_w), dtype=bool)
    if corner == "tl":
        full[:h, :w] = mask
    elif corner == "tr":
        full[:h, full_w - w :] = mask
    elif corner == "bl":
        full[full_h - h :, :w] = mask
    elif corner == "br":
        full[full_h - h :, full_w - w :] = mask
    else:
        raise ValueError(f"unknown corner {corner}")
    return full


def _random_pair(full_h: int, full_w: int) -> tuple[np.ndarray, np.ndarray] | None:
    corners = ("tl", "tr", "bl", "br")
    for _ in range(120):
        sub_h = random.randint(2, full_h - 1)
        sub_w = random.randint(2, full_w - 1)
        blob = _grow_blob(sub_h, sub_w)
        if blob is None:
            continue

        corner = random.choice(corners)
        piece_a = _orient_corner(blob, corner)
        full_a = _embed_corner(piece_a, full_h, full_w, corner)
        piece_b = ~full_a
        if not _is_connected(piece_b):
            continue
        return piece_a, piece_b
    return None


def _place_piece(grid: np.ndarray, piece: np.ndarray, color: int, top: int, left: int) -> None:
    h, w = map(int, piece.shape)
    region = grid[top : top + h, left : left + w]
    region[piece] = color


def _can_place(blocked: np.ndarray, piece_h: int, piece_w: int, top: int, left: int) -> bool:
    h, w = map(int, blocked.shape)
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + piece_h + 1)
    c1 = min(w, left + piece_w + 1)
    return not blocked[r0:r1, c0:c1].any()


def _mark_blocked(blocked: np.ndarray, piece_h: int, piece_w: int, top: int, left: int) -> None:
    h, w = map(int, blocked.shape)
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(h, top + piece_h + 1)
    c1 = min(w, left + piece_w + 1)
    blocked[r0:r1, c0:c1] = True


def _expected_output(pairs: list[dict]) -> np.ndarray:
    rects = []
    for pair in pairs:
        full_h = int(pair["piece_b"].shape[0])
        full_w = int(pair["piece_b"].shape[1])
        rect = np.full((full_h, full_w), AZURE, dtype=int)
        rect[pair["placed_a"]] = int(pair["color_a"])
        rect[pair["piece_b"]] = int(pair["color_b"])
        rects.append(rect)

    rects.sort(key=lambda arr: (int(arr.shape[0] * arr.shape[1]), int(arr.shape[0]), int(arr.shape[1])))
    out_h = sum(int(arr.shape[0]) for arr in rects) + len(rects) - 1
    out_w = max(int(arr.shape[1]) for arr in rects)
    out = np.full((out_h, out_w), AZURE, dtype=int)
    r = 0
    for arr in rects:
        h, w = map(int, arr.shape)
        out[r : r + h, :w] = arr
        r += h + 1
    return out


def _gen_attempt() -> np.ndarray | None:
    pair_count = random.randint(2, 3)

    dims: list[tuple[int, int]] = []
    used_areas: set[int] = set()
    total_height = 0
    for _ in range(pair_count):
        choices = []
        for full_h in range(4, 9):
            for full_w in range(4, 9):
                area = full_h * full_w
                if area in used_areas:
                    continue
                if total_height + full_h + len(dims) > ARC_MAX:
                    continue
                choices.append((full_h, full_w))
        if not choices:
            return None
        full_h, full_w = random.choice(choices)
        dims.append((full_h, full_w))
        used_areas.add(full_h * full_w)
        total_height += full_h

    needed_colors = 2 * pair_count
    colors = random.sample(COLOR_POOL, needed_colors)

    pairs: list[dict] = []
    for index, (full_h, full_w) in enumerate(dims):
        pair_masks = _random_pair(full_h, full_w)
        if pair_masks is None:
            return None
        piece_a, piece_b = pair_masks

        match = _pair_match(
            {"color": colors[2 * index], "mask": piece_a},
            {"color": colors[2 * index + 1], "mask": piece_b},
        )
        if match is None:
            return None

        pairs.append(
            {
                "piece_a": piece_a,
                "piece_b": piece_b,
                "color_a": colors[2 * index],
                "color_b": colors[2 * index + 1],
                "placed_a": match["placed_a"],
            }
        )

    min_side = max(max(piece["piece_a"].shape[0], piece["piece_b"].shape[0]) for piece in pairs) + 4
    grid_h = random.randint(max(18, min_side), ARC_MAX)
    grid_w = random.randint(max(18, min_side), ARC_MAX)
    grid = np.full((grid_h, grid_w), AZURE, dtype=int)
    blocked = np.zeros((grid_h, grid_w), dtype=bool)

    placements = []
    pieces_to_place = []
    for pair in pairs:
        pieces_to_place.append((pair["piece_a"], int(pair["color_a"])))
        pieces_to_place.append((pair["piece_b"], int(pair["color_b"])))
    random.shuffle(pieces_to_place)

    for piece, color in pieces_to_place:
        ph, pw = map(int, piece.shape)
        candidates = [(r, c) for r in range(grid_h - ph + 1) for c in range(grid_w - pw + 1)]
        random.shuffle(candidates)
        placed = False
        for top, left in candidates:
            if _can_place(blocked, ph, pw, top, left):
                _place_piece(grid, piece, color, top, left)
                _mark_blocked(blocked, ph, pw, top, left)
                placements.append((piece, color, top, left))
                placed = True
                break
        if not placed:
            return None

    expected = _expected_output(pairs)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
