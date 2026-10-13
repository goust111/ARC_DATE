import numpy as np


BG_COLOR = 0
STAMP_COLOR = 1
ARROW_COLOR = 9
MAX_RETRIES = 300
TEMP_SIDE = 30

UP = "up"
DOWN = "down"
LEFT = "left"
RIGHT = "right"

_RNG = np.random.default_rng()

_SHAPE_TO_DIR = {
    ((0, 0), (0, 2), (1, 1)): UP,
    ((0, 1), (1, 0), (1, 2)): DOWN,
    ((0, 0), (1, 1), (2, 0)): LEFT,
    ((0, 1), (1, 0), (2, 1)): RIGHT,
}

_FAMILIES = [
    (UP, DOWN),
    (LEFT, RIGHT),
    (UP, LEFT),
    (UP, RIGHT),
    (DOWN, LEFT),
    (DOWN, RIGHT),
]


def neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def neighbors8(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return (
        (r - 1, c - 1),
        (r - 1, c),
        (r - 1, c + 1),
        (r, c - 1),
        (r, c + 1),
        (r + 1, c - 1),
        (r + 1, c),
        (r + 1, c + 1),
    )


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    h, w = mask.shape
    if connectivity == 4:
        neigh = neighbors4
    elif connectivity == 8:
        neigh = neighbors8
    else:
        raise ValueError(connectivity)
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for nr, nc in neigh(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    rows = coords[:, 0]
    cols = coords[:, 1]
    return int(rows.min()), int(cols.min()), int(rows.max()), int(cols.max())


def _classify_arrow(coords: np.ndarray) -> dict[str, int | str] | None:
    if coords.shape != (3, 2):
        return None
    r0, c0, r1, c1 = bbox_coords(coords)
    norm = tuple(sorted((int(r - r0), int(c - c0)) for r, c in coords))
    direction = _SHAPE_TO_DIR.get(norm)
    if direction is None:
        return None
    info: dict[str, int | str] = {
        "direction": direction,
        "r0": r0,
        "c0": c0,
        "r1": r1,
        "c1": c1,
    }
    if direction in (UP, DOWN):
        info["anchor"] = c0 + 1
    else:
        info["anchor"] = r0 + 1
    return info


def _find_arrows(grid: np.ndarray, bg: int) -> tuple[int, list[dict[str, int | str]]]:
    candidates: list[tuple[int, list[dict[str, int | str]]]] = []
    for color in sorted(int(x) for x in np.unique(grid) if int(x) != bg):
        infos: list[dict[str, int | str]] = []
        for comp in components_mask(grid == color, connectivity=8):
            info = _classify_arrow(comp)
            if info is None:
                infos = []
                break
            infos.append(info)
        if infos:
            candidates.append((color, infos))
    assert len(candidates) == 1, "expected exactly one arrow color"
    return candidates[0]


def _find_stamps(
    grid: np.ndarray, bg: int, arrow_color: int
) -> list[dict[str, int | np.ndarray]]:
    stamp_colors = [int(x) for x in np.unique(grid) if int(x) not in (bg, arrow_color)]
    assert len(stamp_colors) == 1, "expected exactly one stamp color"
    stamp_color = stamp_colors[0]
    comps = components_mask(grid == stamp_color)
    assert comps, "expected stamp components"
    stamps: list[dict[str, int | np.ndarray]] = []
    for comp in comps:
        r0, c0, r1, c1 = bbox_coords(comp)
        crop = grid[r0 : r1 + 1, c0 : c1 + 1]
        stamps.append(
            {
                "r0": r0,
                "c0": c0,
                "r1": r1,
                "c1": c1,
                "crop": crop,
            }
        )
    shapes = {tuple(map(int, stamp["crop"].shape)) for stamp in stamps}
    assert len(shapes) == 1, "all stamps must share one bbox size"
    return stamps


def _match_arrow_to_stamp(
    arrow: dict[str, int | str], stamps: list[dict[str, int | np.ndarray]]
) -> int:
    direction = str(arrow["direction"])
    matches: list[int] = []
    for idx, stamp in enumerate(stamps):
        r0 = int(stamp["r0"])
        c0 = int(stamp["c0"])
        r1 = int(stamp["r1"])
        c1 = int(stamp["c1"])
        if direction == DOWN:
            ok = r0 == int(arrow["r1"]) + 1 and c0 <= int(arrow["anchor"]) <= c1
        elif direction == UP:
            ok = r1 + 1 == int(arrow["r0"]) and c0 <= int(arrow["anchor"]) <= c1
        elif direction == RIGHT:
            ok = c0 == int(arrow["c1"]) + 1 and r0 <= int(arrow["anchor"]) <= r1
        elif direction == LEFT:
            ok = c1 + 1 == int(arrow["c0"]) and r0 <= int(arrow["anchor"]) <= r1
        else:
            ok = False
        if ok:
            matches.append(idx)
    assert len(matches) == 1, "each arrow must match exactly one adjacent stamp"
    return matches[0]


def _active_half_mask(direction: str, h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    if direction == UP:
        mask[: (h + 1) // 2, :] = True
    elif direction == DOWN:
        mask[h // 2 :, :] = True
    elif direction == LEFT:
        mask[:, : (w + 1) // 2] = True
    elif direction == RIGHT:
        mask[:, w // 2 :] = True
    else:
        raise ValueError(direction)
    return mask


def main(grid: np.ndarray) -> np.ndarray:
    """
    Merge repeated stamp sprites by overlaying only the arrow-selected half from each copy.

    Input structure:
    - The background is black.
    - Several identical blue stamp sprites appear as separated rectangular crops with the same bbox size.
    - Each stamp has one adjacent maroon 3-pixel arrow on its top, bottom, left, or right side, and the arrow points into the stamp.

    Rule:
    - Identify every 3-pixel arrow and decode whether it points up, down, left, or right.
    - Pair each arrow with the unique adjacent stamp on the side it points toward.
    - Create one output grid with the same size as a single stamp crop, keep only the selected half of each stamp
      (top/bottom/left/right, including the center row or column when the size is odd), and overlay those non-black pixels.

    Key insight:
    The repeated sprites are not copied directly; each nearby arrow selects one half of its own stamp, and those halves combine in shared local coordinates.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg = BG_COLOR if np.any(g == BG_COLOR) else get_bg(g)
    arrow_color, arrows = _find_arrows(g, bg)
    stamps = _find_stamps(g, bg, arrow_color)
    assert len(arrows) == len(stamps) >= 2, "expected one arrow per stamp"

    h, w = map(int, stamps[0]["crop"].shape)
    out = np.full((h, w), bg, dtype=int)
    used: set[int] = set()

    for arrow in arrows:
        stamp_idx = _match_arrow_to_stamp(arrow, stamps)
        assert stamp_idx not in used, "a stamp cannot receive multiple arrows"
        used.add(stamp_idx)

        crop = np.asarray(stamps[stamp_idx]["crop"], dtype=int)
        active = _active_half_mask(str(arrow["direction"]), h, w)
        paint = active & (crop != bg)
        out[paint] = crop[paint]

    assert len(used) == len(stamps)
    return out


def _top_end(h: int) -> int:
    return (h + 1) // 2


def _bottom_start(h: int) -> int:
    return h // 2


def _left_end(w: int) -> int:
    return (w + 1) // 2


def _right_start(w: int) -> int:
    return w // 2


def _cell_members(r: int, c: int, h: int, w: int) -> set[str]:
    members: set[str] = set()
    if r < _top_end(h):
        members.add(UP)
    if r >= _bottom_start(h):
        members.add(DOWN)
    if c < _left_end(w):
        members.add(LEFT)
    if c >= _right_start(w):
        members.add(RIGHT)
    return members


def _safe_cells(h: int, w: int, directions: tuple[str, str]) -> np.ndarray:
    selected = set(directions)
    safe = np.zeros((h, w), dtype=bool)
    for r in range(h):
        for c in range(w):
            if len(_cell_members(r, c, h, w) & selected) <= 1:
                safe[r, c] = True
    return safe


def _draw_segment(mask: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        mask[r, c] = True


def _make_base_pattern(h: int, w: int, directions: tuple[str, str]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    top_end = _top_end(h)
    bottom_start = _bottom_start(h)
    left_end = _left_end(w)
    right_start = _right_start(w)

    if directions == (UP, DOWN):
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        if _RNG.random() < 0.5:
            mask[:, int(_RNG.integers(1, w - 1))] = True
        else:
            mask[int(_RNG.integers(1, h - 1)), :] = True
    elif directions == (LEFT, RIGHT):
        mask[0, :] = True
        mask[-1, :] = True
        mask[:, 0] = True
        mask[:, -1] = True
        if _RNG.random() < 0.5:
            mask[:, int(_RNG.integers(1, w - 1))] = True
        else:
            mask[int(_RNG.integers(1, h - 1)), :] = True
    elif directions == (UP, LEFT):
        mask[0, left_end:] = True
        mask[:, -1] = True
        mask[-1, :] = True
        mask[top_end:, 0] = True
    elif directions == (UP, RIGHT):
        mask[0, :right_start] = True
        mask[:, 0] = True
        mask[-1, :] = True
        mask[top_end:, -1] = True
    elif directions == (DOWN, LEFT):
        mask[0, :] = True
        mask[:, -1] = True
        mask[-1, left_end:] = True
        mask[:bottom_start, 0] = True
    elif directions == (DOWN, RIGHT):
        mask[0, :] = True
        mask[:, 0] = True
        mask[-1, :right_start] = True
        mask[:bottom_start, -1] = True
    else:
        raise ValueError(directions)
    return mask


def _grow_pattern(mask: np.ndarray, safe: np.ndarray) -> np.ndarray:
    out = mask.copy()
    candidates = [(int(r), int(c)) for r, c in np.argwhere(safe & ~out)]
    _RNG.shuffle(candidates)
    target = int(_RNG.integers(out.sum() + 2, min(int(safe.sum()), out.sum() + max(3, safe.size // 4)) + 1))
    for r, c in candidates:
        if int(out.sum()) >= target:
            break
        if any(0 <= nr < out.shape[0] and 0 <= nc < out.shape[1] and out[nr, nc] for nr, nc in neighbors4(r, c)):
            if _RNG.random() < 0.7:
                out[r, c] = True
    return out


def _make_pattern(h: int, w: int, directions: tuple[str, str]) -> np.ndarray | None:
    safe = _safe_cells(h, w, directions)
    base = _make_base_pattern(h, w, directions)
    if not np.all(base <= safe):
        return None
    mask = _grow_pattern(base, safe)
    if len(components_mask(mask)) != 1:
        return None
    if bbox_coords(np.argwhere(mask)) != (0, 0, h - 1, w - 1):
        return None
    for direction in directions:
        active = mask & _active_half_mask(direction, h, w)
        if not np.any(active):
            return None
    overlap = (mask & _active_half_mask(directions[0], h, w)) & (mask & _active_half_mask(directions[1], h, w))
    if np.any(overlap):
        return None
    return mask


def _sample_dims(directions: tuple[str, str]) -> tuple[int, int]:
    if directions == (UP, DOWN):
        h = int(_RNG.choice([6, 8]))
        w = int(_RNG.integers(5, 10))
    elif directions == (LEFT, RIGHT):
        h = int(_RNG.integers(5, 10))
        w = int(_RNG.choice([6, 8]))
    else:
        h = int(_RNG.integers(5, 10))
        w = int(_RNG.integers(5, 10))
    return h, w


def _arrow_cells(direction: str, r0: int, c0: int, h: int, w: int) -> np.ndarray:
    if direction == DOWN:
        start_c = c0 + int(_RNG.integers(0, w - 2))
        cells = [(r0 - 2, start_c + 1), (r0 - 1, start_c), (r0 - 1, start_c + 2)]
    elif direction == UP:
        start_c = c0 + int(_RNG.integers(0, w - 2))
        cells = [(r0 + h, start_c), (r0 + h, start_c + 2), (r0 + h + 1, start_c + 1)]
    elif direction == RIGHT:
        start_r = r0 + int(_RNG.integers(0, h - 2))
        cells = [(start_r + 1, c0 - 2), (start_r, c0 - 1), (start_r + 2, c0 - 1)]
    elif direction == LEFT:
        start_r = r0 + int(_RNG.integers(0, h - 2))
        cells = [(start_r, c0 + w), (start_r + 2, c0 + w), (start_r + 1, c0 + w + 1)]
    else:
        raise ValueError(direction)
    return np.array(cells, dtype=int)


def _expanded_bbox(stamp_r0: int, stamp_c0: int, h: int, w: int, arrow: np.ndarray) -> tuple[int, int, int, int]:
    stamp = np.array(
        [
            (stamp_r0, stamp_c0),
            (stamp_r0 + h - 1, stamp_c0 + w - 1),
        ],
        dtype=int,
    )
    union = np.vstack([stamp, arrow])
    r0, c0, r1, c1 = bbox_coords(union)
    return r0 - 1, c0 - 1, r1 + 1, c1 + 1


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_stamps(h: int, w: int, directions: tuple[str, str]) -> list[dict[str, int | str | np.ndarray]] | None:
    placed: list[dict[str, int | str | np.ndarray]] = []
    boxes: list[tuple[int, int, int, int]] = []
    for direction in directions:
        for _ in range(80):
            if direction == DOWN:
                r0 = int(_RNG.integers(2, TEMP_SIDE - h + 1))
                c0 = int(_RNG.integers(0, TEMP_SIDE - w + 1))
            elif direction == UP:
                r0 = int(_RNG.integers(0, TEMP_SIDE - h - 1))
                c0 = int(_RNG.integers(0, TEMP_SIDE - w + 1))
            elif direction == RIGHT:
                r0 = int(_RNG.integers(0, TEMP_SIDE - h + 1))
                c0 = int(_RNG.integers(2, TEMP_SIDE - w + 1))
            elif direction == LEFT:
                r0 = int(_RNG.integers(0, TEMP_SIDE - h + 1))
                c0 = int(_RNG.integers(0, TEMP_SIDE - w - 1))
            else:
                raise ValueError(direction)
            arrow = _arrow_cells(direction, r0, c0, h, w)
            box = _expanded_bbox(r0, c0, h, w, arrow)
            if box[0] < 0 or box[1] < 0 or box[2] >= TEMP_SIDE or box[3] >= TEMP_SIDE:
                continue
            if any(_boxes_overlap(box, other) for other in boxes):
                continue
            placed.append(
                {
                    "direction": direction,
                    "r0": r0,
                    "c0": c0,
                    "arrow": arrow,
                }
            )
            boxes.append(box)
            break
        else:
            return None
    return placed


def _expected_output(mask: np.ndarray, directions: tuple[str, str]) -> np.ndarray:
    h, w = mask.shape
    out = np.full((h, w), BG_COLOR, dtype=int)
    for direction in directions:
        active = mask & _active_half_mask(direction, h, w)
        out[active] = STAMP_COLOR
    return out


def _crop_with_margin(grid: np.ndarray) -> np.ndarray:
    coords = np.argwhere(grid != BG_COLOR)
    r0, c0, r1, c1 = bbox_coords(coords)
    top = int(_RNG.integers(0, min(3, r0 + 1)))
    left = int(_RNG.integers(0, min(3, c0 + 1)))
    bottom = int(_RNG.integers(0, min(3, TEMP_SIDE - r1)))
    right = int(_RNG.integers(0, min(3, TEMP_SIDE - c1)))
    return grid[r0 - top : r1 + bottom + 1, c0 - left : c1 + right + 1]


def _gen_attempt() -> np.ndarray | None:
    directions = tuple(_FAMILIES[int(_RNG.integers(0, len(_FAMILIES)))])
    h, w = _sample_dims(directions)
    pattern = _make_pattern(h, w, directions)
    if pattern is None:
        return None

    placements = _place_stamps(h, w, directions)
    if placements is None:
        return None

    grid = np.full((TEMP_SIDE, TEMP_SIDE), BG_COLOR, dtype=int)
    for item in placements:
        r0 = int(item["r0"])
        c0 = int(item["c0"])
        grid[r0 : r0 + h, c0 : c0 + w][pattern] = STAMP_COLOR
        arrow = np.asarray(item["arrow"], dtype=int)
        grid[arrow[:, 0], arrow[:, 1]] = ARROW_COLOR

    cropped = _crop_with_margin(grid)
    expected = _expected_output(pattern, directions)
    if not np.array_equal(main(cropped), expected):
        return None
    return cropped


def generate_input() -> np.ndarray:
    for _ in range(MAX_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid stamp-half input")
