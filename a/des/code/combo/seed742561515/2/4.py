from __future__ import annotations

import random

import numpy as np


BG = 0
GREY = 5
ROOM_KEYS = ("tl", "tr", "bl", "br")
ROOM_SIDES = {
    "tl": ("top", "left"),
    "tr": ("top", "right"),
    "bl": ("bottom", "left"),
    "br": ("bottom", "right"),
}
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_RETRIES = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    out: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(np.asarray(pts, dtype=int))
    return out


def _bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _cropped_mask_from_coords(coords: np.ndarray) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox_coords(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    mask[coords[:, 0] - r0, coords[:, 1] - c0] = True
    return mask, (r0, c0, r1, c1)


def _shape_key(mask: np.ndarray) -> tuple[int, int, bytes]:
    return int(mask.shape[0]), int(mask.shape[1]), mask.astype(np.uint8).tobytes()


def _connected(mask: np.ndarray) -> bool:
    if not np.any(mask):
        return False
    return len(_components(mask)) == 1


def _edge_touches(mask: np.ndarray) -> tuple[bool, bool, bool, bool]:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False, False, False, False
    top = bool(np.any(mask[0, 1:-1]))
    bottom = bool(np.any(mask[-1, 1:-1]))
    left = bool(np.any(mask[1:-1, 0]))
    right = bool(np.any(mask[1:-1, -1]))
    return top, bottom, left, right


def _valid_room_mask(mask: np.ndarray) -> bool:
    h, w = mask.shape
    area = int(mask.sum())
    total = h * w
    if area <= 0 or area >= total:
        return False
    if area < max(8, int(np.ceil(total * 0.45))):
        return False
    if area > int(np.floor(total * 0.85)):
        return False
    if not _connected(mask):
        return False
    if _edge_touches(mask) != (True, True, True, True):
        return False
    if np.array_equal(mask, np.fliplr(mask)):
        return False
    if np.array_equal(mask, np.flipud(mask)):
        return False
    return True


def _random_room_mask(h: int, w: int) -> np.ndarray | None:
    area = h * w
    min_area = max(8, int(np.ceil(area * 0.45)))
    max_area = int(np.floor(area * 0.85))
    cells = [(r, c) for r in range(h) for c in range(w)]
    max_carves = min(area - min_area, max(4, area // 3))
    if max_carves <= 0:
        return None
    for _ in range(200):
        mask = np.ones((h, w), dtype=bool)
        target_carves = random.randint(2, max_carves)
        order = cells[:]
        random.shuffle(order)
        carved = 0
        for r, c in order:
            if carved >= target_carves:
                break
            trial = mask.copy()
            trial[r, c] = False
            if int(trial.sum()) < min_area:
                continue
            if not _connected(trial):
                continue
            if _edge_touches(trial) != (True, True, True, True):
                continue
            mask = trial
            carved += 1
        if carved >= 2 and _valid_room_mask(mask):
            return mask
    return None


def _transform_mask(mask: np.ndarray, side: str | None) -> np.ndarray:
    if side in ("top", "bottom"):
        return np.fliplr(mask)
    if side in ("left", "right"):
        return np.flipud(mask)
    return mask.copy()


def _marker_position(
    mask: np.ndarray,
    bbox: tuple[int, int, int, int],
    side: str,
    template_shape: tuple[int, int],
) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    th, tw = template_shape
    if side == "top":
        cols = [c0 + int(x) for x in np.where(mask[0, 1:-1])[0] + 1]
        return 0, cols[len(cols) // 2]
    if side == "bottom":
        cols = [c0 + int(x) for x in np.where(mask[-1, 1:-1])[0] + 1]
        return th - 1, cols[len(cols) // 2]
    if side == "left":
        rows = [r0 + int(x) for x in np.where(mask[1:-1, 0])[0] + 1]
        return rows[len(rows) // 2], 0
    if side == "right":
        rows = [r0 + int(x) for x in np.where(mask[1:-1, -1])[0] + 1]
        return rows[len(rows) // 2], tw - 1
    raise AssertionError(f"unknown side {side}")


def _detect_grey(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid[grid != BG], return_counts=True)
    assert len(vals) > 0, "grid must contain non-background colors"
    return int(vals[int(np.argmax(counts))])


def _slot_marker_side(
    slot_mask: np.ndarray,
    slot_bbox: tuple[int, int, int, int],
    markers: set[tuple[int, int]],
    template_shape: tuple[int, int],
) -> str | None:
    r0, c0, r1, c1 = slot_bbox
    th, tw = template_shape
    matches: list[str] = []
    if r0 == 1:
        for dc in np.where(slot_mask[0])[0]:
            if (0, c0 + int(dc)) in markers:
                matches.append("top")
                break
    if r1 == th - 2:
        for dc in np.where(slot_mask[-1])[0]:
            if (th - 1, c0 + int(dc)) in markers:
                matches.append("bottom")
                break
    if c0 == 1:
        for dr in np.where(slot_mask[:, 0])[0]:
            if (r0 + int(dr), 0) in markers:
                matches.append("left")
                break
    if c1 == tw - 2:
        for dr in np.where(slot_mask[:, -1])[0]:
            if (r0 + int(dr), tw - 1) in markers:
                matches.append("right")
                break
    assert len(matches) <= 1, "each slot must see at most one border marker"
    return matches[0] if matches else None


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble pieces into grey template slots, then mirror the filled slot mask when the
    slot's outer border segment carries a colored marker.
    """
    g = np.asarray(grid, dtype=int)
    grey = _detect_grey(g)

    grey_coords = np.argwhere(g == grey)
    assert len(grey_coords) > 0, "template must contain grey cells"
    tr0, tc0, tr1, tc1 = _bbox_coords(grey_coords)
    template = g[tr0 : tr1 + 1, tc0 : tc1 + 1].copy()
    th, tw = template.shape

    slots: list[dict] = []
    for coords in _components(template == BG):
        if np.any(coords[:, 0] == 0) or np.any(coords[:, 0] == th - 1) or np.any(coords[:, 1] == 0) or np.any(coords[:, 1] == tw - 1):
            continue
        mask, bbox = _cropped_mask_from_coords(coords)
        slots.append({"mask": mask, "bbox": bbox})
    assert slots, "template must contain enclosed black slots"

    markers: set[tuple[int, int]] = set()
    pieces: list[dict] = []
    fg_mask = (g != BG) & (g != grey)
    for coords in _components(fg_mask):
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "each colored component must be monochrome"
        color = next(iter(colors))
        mask, bbox = _cropped_mask_from_coords(coords)
        r0, c0, r1, c1 = bbox
        inside_template = tr0 <= r0 <= r1 <= tr1 and tc0 <= c0 <= c1 <= tc1
        on_outer_border = (
            len(coords) == 1
            and inside_template
            and (r0 in (tr0, tr1) or c0 in (tc0, tc1))
        )
        if on_outer_border:
            markers.add((r0 - tr0, c0 - tc0))
        else:
            pieces.append({"mask": mask, "color": color})

    assert len(pieces) == len(slots), "piece count must equal slot count"
    piece_by_key: dict[tuple[int, int, bytes], dict] = {}
    for piece in pieces:
        key = _shape_key(piece["mask"])
        assert key not in piece_by_key, "piece shapes must be unique"
        piece_by_key[key] = piece

    out = template.copy()
    for slot in slots:
        key = _shape_key(slot["mask"])
        assert key in piece_by_key, "every slot must have an exact matching piece"
        piece = piece_by_key.pop(key)
        side = _slot_marker_side(slot["mask"], slot["bbox"], markers, (th, tw))
        draw_mask = _transform_mask(slot["mask"], side)
        r0, c0, r1, c1 = slot["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[draw_mask] = int(piece["color"])
    assert not piece_by_key, "all pieces must be used exactly once"
    return out


def _room_layout(
    top_h: int, bottom_h: int, left_w: int, right_w: int
) -> dict[str, tuple[int, int, int, int]]:
    return {
        "tl": (1, 1, top_h, left_w),
        "tr": (1, left_w + 2, top_h, right_w),
        "bl": (top_h + 2, 1, bottom_h, left_w),
        "br": (top_h + 2, left_w + 2, bottom_h, right_w),
    }


def _make_template_case() -> tuple[np.ndarray, list[dict], list[dict]] | None:
    top_h = random.randint(5, 7)
    bottom_h = random.randint(5, 7)
    left_w = random.randint(5, 7)
    right_w = random.randint(5, 7)
    template_h = top_h + bottom_h + 3
    template_w = left_w + right_w + 3
    template = np.full((template_h, template_w), GREY, dtype=int)
    rooms = _room_layout(top_h, bottom_h, left_w, right_w)
    active_rooms = set(random.sample(ROOM_KEYS, random.choice((3, 4))))

    slot_specs: list[dict] = []
    seen_keys: set[tuple[int, int, bytes]] = set()
    for room_key in ROOM_KEYS:
        r0, c0, h, w = rooms[room_key]
        r1 = r0 + h - 1
        c1 = c0 + w - 1
        if room_key not in active_rooms:
            continue
        mask = _random_room_mask(h, w)
        if mask is None:
            return None
        key = _shape_key(mask)
        if key in seen_keys:
            return None
        seen_keys.add(key)
        template[r0 : r1 + 1, c0 : c1 + 1][mask] = BG
        slot_specs.append(
            {
                "room": room_key,
                "bbox": (r0, c0, r1, c1),
                "mask": mask,
                "sides": ROOM_SIDES[room_key],
            }
        )

    if len(slot_specs) < 3:
        return None

    piece_colors = random.sample(COLOR_POOL, len(slot_specs))
    remaining = [c for c in COLOR_POOL if c not in piece_colors]

    marker_count = random.randint(1, len(slot_specs))
    marked_rooms = random.sample(slot_specs, marker_count)
    side_by_room: dict[str, str] = {}

    if marker_count >= 2:
        horiz_spec = random.choice(marked_rooms)
        side_by_room[horiz_spec["room"]] = horiz_spec["sides"][0]

        vertical_choices = [spec for spec in marked_rooms if spec["room"] != horiz_spec["room"]]
        vert_spec = random.choice(vertical_choices)
        side_by_room[vert_spec["room"]] = vert_spec["sides"][1]

    for spec in marked_rooms:
        if spec["room"] in side_by_room:
            continue
        side_by_room[spec["room"]] = random.choice(spec["sides"])

    marker_colors = random.sample(remaining, len(side_by_room))
    spec_by_room = {spec["room"]: spec for spec in slot_specs}
    markers: list[dict] = []
    for color, room in zip(marker_colors, side_by_room):
        spec = spec_by_room[room]
        side = side_by_room[room]
        mr, mc = _marker_position(spec["mask"], spec["bbox"], side, template.shape)
        template[mr, mc] = int(color)
        markers.append({"room": room, "side": side, "pos": (mr, mc), "color": int(color)})

    random.shuffle(piece_colors)
    for color, spec in zip(piece_colors, slot_specs):
        spec["piece_color"] = int(color)
        spec["marker_side"] = side_by_room.get(spec["room"])
        transformed = _transform_mask(spec["mask"], spec["marker_side"])
        spec["out_mask"] = transformed
        if spec["marker_side"] is not None and np.array_equal(transformed, spec["mask"]):
            return None

    return template, slot_specs, markers


def _render_expected(template: np.ndarray, slot_specs: list[dict]) -> np.ndarray:
    out = template.copy()
    for spec in slot_specs:
        r0, c0, r1, c1 = spec["bbox"]
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[spec["out_mask"]] = int(spec["piece_color"])
    return out


def _scatter_input(template: np.ndarray, slot_specs: list[dict]) -> np.ndarray | None:
    piece_order = slot_specs[:]
    random.shuffle(piece_order)

    piece_width_total = sum(int(spec["mask"].shape[1]) for spec in piece_order)
    gaps = [random.randint(1, 3) for _ in range(max(0, len(piece_order) - 1))]
    gap_total = sum(gaps)
    side_margin = random.randint(2, 4)
    grid_w_min = max(template.shape[1] + random.randint(4, 8), piece_width_total + gap_total + side_margin * 2)
    grid_w = grid_w_min + random.randint(0, 4)

    max_piece_h = max(int(spec["mask"].shape[0]) for spec in piece_order)
    top_band = max_piece_h + random.randint(4, 7)
    bottom_margin = random.randint(1, 3)
    grid_h = top_band + template.shape[0] + bottom_margin

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    template_r0 = top_band
    max_c0 = grid_w - template.shape[1] - 1
    if max_c0 < 1:
        return None
    template_c0 = random.randint(1, max_c0)
    grid[
        template_r0 : template_r0 + template.shape[0],
        template_c0 : template_c0 + template.shape[1],
    ] = template

    cursor = random.randint(1, max(1, grid_w - (piece_width_total + gap_total) - 1))
    for idx, spec in enumerate(piece_order):
        mask = spec["mask"]
        ph, pw = mask.shape
        max_r0 = top_band - ph - 1
        if max_r0 < 1:
            return None
        r0 = random.randint(1, max_r0)
        c0 = cursor
        if c0 + pw >= grid_w - 1:
            return None
        grid[r0 : r0 + ph, c0 : c0 + pw][mask] = int(spec["piece_color"])
        cursor = c0 + pw
        if idx + 1 < len(piece_order):
            cursor += gaps[idx]
    return grid


def _build_case() -> tuple[np.ndarray, np.ndarray] | None:
    base = _make_template_case()
    if base is None:
        return None
    template, slot_specs, _markers = base
    expected = _render_expected(template, slot_specs)
    grid = _scatter_input(template, slot_specs)
    if grid is None:
        return None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        built = _build_case()
        if built is None:
            continue
        grid, expected = built
        actual = main(grid)
        if not np.array_equal(actual, expected):
            continue
        return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
