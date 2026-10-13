from __future__ import annotations

import numpy as np


BG = 0
COLOR_POOL = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=int)
ROOM_COUNT_CHOICES = (2, 3, 4, 5)
INNER_DIM_MIN = 4
INNER_DIM_MAX = 8
GRID_SIDE_MIN = 18
GRID_SIDE_MAX = 30
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400
MAX_PATTERN_ATTEMPTS = 250
RNG = np.random.default_rng()


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _is_rectangular_ring(mask: np.ndarray, coords: np.ndarray) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False

    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    if not np.all(sub[0, :]) or not np.all(sub[-1, :]):
        return False
    if not np.all(sub[:, 0]) or not np.all(sub[:, -1]):
        return False
    if h > 2 and w > 2 and np.any(sub[1:-1, 1:-1]):
        return False
    return True


def _find_rooms(grid: np.ndarray) -> list[dict[str, object]]:
    candidates: list[tuple[tuple[int, int], list[tuple[int, int, int, int]]]] = []

    for color in map(int, np.unique(grid)):
        if color == BG:
            continue
        color_mask = grid == color
        comps = _components(color_mask)
        if not comps:
            continue
        boxes: list[tuple[int, int, int, int]] = []
        ok = True
        for comp in comps:
            if not _is_rectangular_ring(color_mask, comp):
                ok = False
                break
            boxes.append(_bbox(comp))
        if ok:
            total_box_area = sum((r1 - r0 + 1) * (c1 - c0 + 1) for r0, c0, r1, c1 in boxes)
            candidates.append(((len(boxes), total_box_area), boxes))

    assert candidates, "expected at least one rectangular frame color"
    best_score = max(score for score, _boxes in candidates)
    best = [boxes for score, boxes in candidates if score == best_score]
    assert len(best) == 1, "frame color must be uniquely identifiable"

    rooms: list[dict[str, object]] = []
    for r0, c0, r1, c1 in best[0]:
        inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        inner_h, inner_w = map(int, inner.shape)
        assert inner_h > 0 and inner_w > 0

        pattern_coords = np.argwhere(inner != BG)
        assert len(pattern_coords) > 0, "every room must contain a seed pattern"
        colors = np.unique(inner[inner != BG])
        assert len(colors) == 1, "each room pattern must be monochrome"

        pr0, pc0 = pattern_coords.min(axis=0)
        pr1, pc1 = pattern_coords.max(axis=0)
        patch = inner[pr0 : pr1 + 1, pc0 : pc1 + 1].copy()
        assert patch.shape[0] < inner_h and patch.shape[1] < inner_w

        rooms.append(
            {
                "outer_top": int(r0),
                "outer_left": int(c0),
                "inner_top": int(r0 + 1),
                "inner_left": int(c0 + 1),
                "inner_h": inner_h,
                "inner_w": inner_w,
                "area": inner_h * inner_w,
                "patch": patch,
            }
        )

    return rooms


def _rotate_clockwise(patch: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(patch, -int(quarter_turns) % 4)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Identify rectangular rooms, rank them by interior area, rotate each centered seed pattern,
    and write the rotated pattern back into the room interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    rooms = _find_rooms(g)
    ranked = sorted(
        rooms,
        key=lambda room: (
            int(room["area"]),
            int(room["outer_top"]),
            int(room["outer_left"]),
        ),
    )

    out = g.copy()
    for rank, room in enumerate(ranked):
        inner_top = int(room["inner_top"])
        inner_left = int(room["inner_left"])
        inner_h = int(room["inner_h"])
        inner_w = int(room["inner_w"])
        patch = np.asarray(room["patch"], dtype=int)
        rotated = _rotate_clockwise(patch, rank % 4)
        rot_h, rot_w = map(int, rotated.shape)
        assert rot_h <= inner_h and rot_w <= inner_w

        out[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = BG
        dst_top = inner_top + (inner_h - rot_h) // 2
        dst_left = inner_left + (inner_w - rot_w) // 2
        out[dst_top : dst_top + rot_h, dst_left : dst_left + rot_w] = rotated

    return out


def _all_room_dim_options() -> dict[int, list[tuple[int, int]]]:
    by_area: dict[int, list[tuple[int, int]]] = {}
    for inner_h in range(INNER_DIM_MIN, INNER_DIM_MAX + 1):
        for inner_w in range(INNER_DIM_MIN, INNER_DIM_MAX + 1):
            if inner_h % 2 != inner_w % 2:
                continue
            by_area.setdefault(inner_h * inner_w, []).append((inner_h, inner_w))
    return by_area


ROOM_DIM_OPTIONS = _all_room_dim_options()


def _is_asymmetric(mask: np.ndarray) -> bool:
    base = mask.astype(np.uint8)
    variants = []
    for k in range(4):
        rotated = np.rot90(base, k)
        variants.append(rotated)
        variants.append(np.fliplr(rotated))
    for variant in variants[1:]:
        if variant.shape == base.shape and np.array_equal(variant, base):
            return False
    return True


def _random_pattern_mask(box_h: int, box_w: int) -> np.ndarray:
    base = np.zeros((box_h, box_w), dtype=bool)
    base[:, 0] = True
    base[-1, :] = True

    for _ in range(MAX_PATTERN_ATTEMPTS):
        mask = base.copy()
        max_extra = min(box_h * box_w - int(mask.sum()) - 1, max(0, (box_h * box_w) // 3))
        add_count = int(RNG.integers(max_extra + 1)) if max_extra > 0 else 0
        for _extra in range(add_count):
            frontier: list[tuple[int, int]] = []
            for r in range(box_h):
                for c in range(box_w):
                    if mask[r, c]:
                        continue
                    for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                        nr, nc = r + dr, c + dc
                        if 0 <= nr < box_h and 0 <= nc < box_w and mask[nr, nc]:
                            frontier.append((r, c))
                            break
            if not frontier:
                break
            r, c = frontier[int(RNG.integers(len(frontier)))]
            mask[r, c] = True

        if mask.all():
            continue
        if not mask[0, :].any() or not mask[-1, :].any():
            continue
        if not mask[:, 0].any() or not mask[:, -1].any():
            continue
        comps = _components(mask)
        if len(comps) != 1 or len(comps[0]) != int(mask.sum()):
            continue
        if _is_rectangular_ring(mask, np.argwhere(mask)):
            continue
        if not _is_asymmetric(mask):
            continue

        patch = mask.astype(int)
        k = int(RNG.integers(4))
        patch = np.rot90(patch, k)
        if int(RNG.integers(2)) == 1:
            patch = np.fliplr(patch)
        return patch.astype(bool)

    patch = base.astype(int)
    k = int(RNG.integers(4))
    patch = np.rot90(patch, k)
    if int(RNG.integers(2)) == 1:
        patch = np.fliplr(patch)
    return patch.astype(bool)


def _sample_room_specs() -> list[dict[str, object]]:
    room_count = int(RNG.choice(np.array(ROOM_COUNT_CHOICES)))
    chosen_areas = RNG.choice(np.array(sorted(ROOM_DIM_OPTIONS)), size=room_count, replace=False)

    specs: list[dict[str, object]] = []
    for area in map(int, chosen_areas):
        inner_h, inner_w = ROOM_DIM_OPTIONS[area][int(RNG.integers(len(ROOM_DIM_OPTIONS[area])))]
        parity = inner_h % 2
        limit = min(inner_h, inner_w) - 1
        bbox_options = [
            (box_h, box_w)
            for box_h in range(2, limit + 1)
            for box_w in range(2, limit + 1)
            if box_h % 2 == parity and box_w % 2 == parity
        ]
        assert bbox_options
        box_h, box_w = bbox_options[int(RNG.integers(len(bbox_options)))]
        specs.append(
            {
                "inner_h": inner_h,
                "inner_w": inner_w,
                "outer_h": inner_h + 2,
                "outer_w": inner_w + 2,
                "mask": _random_pattern_mask(box_h, box_w),
            }
        )

    RNG.shuffle(specs)
    return specs


def _place_rooms(room_specs: list[dict[str, object]], grid_h: int, grid_w: int) -> list[tuple[int, int]] | None:
    occupied = np.zeros((grid_h, grid_w), dtype=bool)
    placements: list[tuple[int, int] | None] = [None] * len(room_specs)
    order = sorted(
        range(len(room_specs)),
        key=lambda idx: int(room_specs[idx]["outer_h"]) * int(room_specs[idx]["outer_w"]),
        reverse=True,
    )

    for idx in order:
        outer_h = int(room_specs[idx]["outer_h"])
        outer_w = int(room_specs[idx]["outer_w"])
        candidates: list[tuple[int, int]] = []
        for top in range(1, grid_h - outer_h):
            for left in range(1, grid_w - outer_w):
                r0 = max(0, top - PLACEMENT_GAP)
                c0 = max(0, left - PLACEMENT_GAP)
                r1 = min(grid_h, top + outer_h + PLACEMENT_GAP)
                c1 = min(grid_w, left + outer_w + PLACEMENT_GAP)
                if not occupied[r0:r1, c0:c1].any():
                    candidates.append((top, left))
        if not candidates:
            return None
        placements[idx] = candidates[int(RNG.integers(len(candidates)))]
        top, left = placements[idx]
        occupied[top : top + outer_h, left : left + outer_w] = True

    return [(int(top), int(left)) for top, left in placements if top is not None and left is not None]


def _draw_room(grid: np.ndarray, room: dict[str, object], frame_color: int) -> None:
    top = int(room["top"])
    left = int(room["left"])
    inner_h = int(room["inner_h"])
    inner_w = int(room["inner_w"])
    outer_h = int(room["outer_h"])
    outer_w = int(room["outer_w"])
    patch = np.asarray(room["patch"], dtype=int)

    grid[top, left : left + outer_w] = frame_color
    grid[top + outer_h - 1, left : left + outer_w] = frame_color
    grid[top : top + outer_h, left] = frame_color
    grid[top : top + outer_h, left + outer_w - 1] = frame_color

    ph, pw = map(int, patch.shape)
    inner_top = top + 1
    inner_left = left + 1
    dst_top = inner_top + (inner_h - ph) // 2
    dst_left = inner_left + (inner_w - pw) // 2
    grid[dst_top : dst_top + ph, dst_left : dst_left + pw] = patch


def _expected_output(inp: np.ndarray, rooms: list[dict[str, object]]) -> np.ndarray:
    out = np.asarray(inp, dtype=int).copy()
    ranked = sorted(
        rooms,
        key=lambda room: (
            int(room["inner_h"]) * int(room["inner_w"]),
            int(room["top"]),
            int(room["left"]),
        ),
    )

    for rank, room in enumerate(ranked):
        inner_top = int(room["top"]) + 1
        inner_left = int(room["left"]) + 1
        inner_h = int(room["inner_h"])
        inner_w = int(room["inner_w"])
        rotated = _rotate_clockwise(np.asarray(room["patch"], dtype=int), rank % 4)
        rot_h, rot_w = map(int, rotated.shape)
        out[inner_top : inner_top + inner_h, inner_left : inner_left + inner_w] = BG
        dst_top = inner_top + (inner_h - rot_h) // 2
        dst_left = inner_left + (inner_w - rot_w) // 2
        out[dst_top : dst_top + rot_h, dst_left : dst_left + rot_w] = rotated

    return out


def _gen_attempt() -> np.ndarray | None:
    room_specs = _sample_room_specs()
    total_outer_area = sum(int(room["outer_h"]) * int(room["outer_w"]) for room in room_specs)
    max_room_h = max(int(room["outer_h"]) for room in room_specs)
    max_room_w = max(int(room["outer_w"]) for room in room_specs)

    min_side = max(
        GRID_SIDE_MIN,
        max_room_h + 2,
        max_room_w + 2,
        int(np.ceil(np.sqrt(total_outer_area * 1.8))),
    )
    if min_side > GRID_SIDE_MAX:
        return None

    grid_h = int(RNG.integers(min_side, GRID_SIDE_MAX + 1))
    grid_w = int(RNG.integers(min_side, GRID_SIDE_MAX + 1))
    placements = _place_rooms(room_specs, grid_h, grid_w)
    if placements is None:
        return None

    colors = RNG.choice(COLOR_POOL, size=len(room_specs) + 1, replace=False)
    frame_color = int(colors[0])
    pattern_colors = list(map(int, colors[1:]))

    rooms: list[dict[str, object]] = []
    for idx, spec in enumerate(room_specs):
        patch = np.where(np.asarray(spec["mask"], dtype=bool), pattern_colors[idx], BG).astype(int)
        room = {
            "top": int(placements[idx][0]),
            "left": int(placements[idx][1]),
            "inner_h": int(spec["inner_h"]),
            "inner_w": int(spec["inner_w"]),
            "outer_h": int(spec["outer_h"]),
            "outer_w": int(spec["outer_w"]),
            "patch": patch,
        }
        rooms.append(room)

    inp = np.full((grid_h, grid_w), BG, dtype=int)
    for room in rooms:
        _draw_room(inp, room, frame_color)

    expected = _expected_output(inp, rooms)
    try:
        actual = main(inp)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        inp = _gen_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid room-ranking puzzle")
