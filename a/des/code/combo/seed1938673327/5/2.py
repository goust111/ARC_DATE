import random
from collections import Counter, deque

import numpy as np


BG = 0
BLUE = 1
RED = 2
OBJECT_COLORS = (3, 4, 5, 6, 7, 8, 9)
NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
GEN_RETRIES = 300


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _find_components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            queue = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while queue:
                cr, cc = queue.popleft()
                coords.append((cr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _edge_color(values: np.ndarray, bg: int) -> int:
    colors = [int(v) for v in np.ravel(values) if int(v) != bg]
    assert colors, "frame edge must contain a non-background color"
    return int(Counter(colors).most_common(1)[0][0])


def _mode_for_colors(colors: set[int]) -> str:
    if RED in colors and BLUE in colors:
        return "rot180"
    if RED in colors:
        return "hflip"
    if BLUE in colors:
        return "vflip"
    return "none"


def _apply_mode(patch: np.ndarray, mode: str) -> np.ndarray:
    if mode == "hflip":
        return np.fliplr(patch)
    if mode == "vflip":
        return np.flipud(patch)
    if mode == "rot180":
        return np.rot90(patch, 2)
    return patch.copy()


def _extract_frame(grid: np.ndarray, bg: int) -> tuple[tuple[int, int, int, int], dict[str, int]]:
    fg = np.argwhere(grid != bg)
    assert fg.size > 0, "expected a non-empty frame"
    r0, c0, r1, c1 = _bbox_from_coords(fg)
    assert r1 - r0 >= 2 and c1 - c0 >= 2, "frame must be at least 3x3"
    side_colors = {
        "top": _edge_color(grid[r0, c0 : c1 + 1], bg),
        "bottom": _edge_color(grid[r1, c0 : c1 + 1], bg),
        "left": _edge_color(grid[r0 : r1 + 1, c0], bg),
        "right": _edge_color(grid[r0 : r1 + 1, c1], bg),
    }
    return (r0, c0, r1, c1), side_colors


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    (r0, c0, r1, c1), side_colors = _extract_frame(g, bg)

    inner = g[r0 + 1 : r1, c0 + 1 : c1]
    inner_h, inner_w = inner.shape
    components = _find_components(inner != bg)

    placements: list[tuple[int, int, np.ndarray]] = []
    out = g.copy()

    for coords in components:
        rows = coords[:, 0]
        cols = coords[:, 1]
        touched_colors: set[int] = set()
        if np.any(rows == 0):
            touched_colors.add(side_colors["top"])
        if np.any(rows == inner_h - 1):
            touched_colors.add(side_colors["bottom"])
        if np.any(cols == 0):
            touched_colors.add(side_colors["left"])
        if np.any(cols == inner_w - 1):
            touched_colors.add(side_colors["right"])

        rr0, cc0, rr1, cc1 = _bbox_from_coords(coords)
        patch = np.full((rr1 - rr0 + 1, cc1 - cc0 + 1), bg, dtype=int)
        for rr, cc in coords:
            patch[rr - rr0, cc - cc0] = inner[rr, cc]
            out[r0 + 1 + rr, c0 + 1 + cc] = bg

        transformed = _apply_mode(patch, _mode_for_colors(touched_colors))
        placements.append((rr0, cc0, transformed))

    for rr0, cc0, patch in placements:
        view = out[r0 + 1 + rr0 : r0 + 1 + rr0 + patch.shape[0], c0 + 1 + cc0 : c0 + 1 + cc0 + patch.shape[1]]
        mask = patch != bg
        view[mask] = patch[mask]

    return out


def main(grid: np.ndarray) -> np.ndarray:
    return _solve(grid)


def _random_frame_sides() -> dict[str, int]:
    names = ("top", "bottom", "left", "right")
    while True:
        colors = {name: random.choice((RED, BLUE)) for name in names}
        if len(set(colors.values())) == 2:
            return colors


def _slot_specs(side_colors: dict[str, int]) -> list[dict[str, object]]:
    slots: list[dict[str, object]] = []
    for side in ("top", "bottom", "left", "right"):
        colors = {int(side_colors[side])}
        slots.append({"kind": "side", "name": side, "colors": colors, "mode": _mode_for_colors(colors)})

    corner_sides = {
        "tl": ("top", "left"),
        "tr": ("top", "right"),
        "bl": ("bottom", "left"),
        "br": ("bottom", "right"),
    }
    for corner, sides in corner_sides.items():
        colors = {int(side_colors[sides[0]]), int(side_colors[sides[1]])}
        slots.append({"kind": "corner", "name": corner, "colors": colors, "mode": _mode_for_colors(colors)})
    return slots


def _grow_mask(h: int, w: int, area: int) -> np.ndarray | None:
    mask = np.zeros((h, w), dtype=bool)
    start = (random.randrange(h), random.randrange(w))
    mask[start] = True

    while int(mask.sum()) < area:
        frontier: set[tuple[int, int]] = set()
        for r, c in np.argwhere(mask):
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r) + dr, int(c) + dc
                if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                    frontier.add((nr, nc))
        if not frontier:
            return None
        nr, nc = random.choice(tuple(frontier))
        mask[nr, nc] = True

    return _crop_mask(mask)


def _shape_changes(mask: np.ndarray, mode: str) -> bool:
    transformed = _apply_mode(mask.astype(int), mode) > 0
    return not np.array_equal(mask, transformed)


def _make_object_mask(mode: str, max_h: int, max_w: int) -> np.ndarray | None:
    if max_h < 2 or max_w < 2:
        return None
    for _ in range(200):
        h = random.randint(2, max_h)
        w = random.randint(2, max_w)
        max_area = min(8, h * w)
        if max_area < 3:
            continue
        area = random.randint(3, max_area)
        mask = _grow_mask(h, w, area)
        if mask is None or min(mask.shape) < 2:
            continue
        if _shape_changes(mask, mode):
            return mask
    return None


def _position_candidates(slot: dict[str, object], shape: tuple[int, int], inner_h: int, inner_w: int) -> list[tuple[int, int]]:
    h, w = shape
    name = str(slot["name"])
    kind = str(slot["kind"])

    if kind == "corner":
        if h >= inner_h or w >= inner_w:
            return []
        if name == "tl":
            return [(0, 0)]
        if name == "tr":
            return [(0, inner_w - w)]
        if name == "bl":
            return [(inner_h - h, 0)]
        return [(inner_h - h, inner_w - w)]

    if name == "top":
        if h >= inner_h or w > inner_w - 2:
            return []
        return [(0, c0) for c0 in range(1, inner_w - w)]
    if name == "bottom":
        if h >= inner_h or w > inner_w - 2:
            return []
        return [(inner_h - h, c0) for c0 in range(1, inner_w - w)]
    if name == "left":
        if w >= inner_w or h > inner_h - 2:
            return []
        return [(r0, 0) for r0 in range(1, inner_h - h)]
    if w >= inner_w or h > inner_h - 2:
        return []
    return [(r0, inner_w - w) for r0 in range(1, inner_h - h)]


def _bbox_has_gap(occupied: np.ndarray, r0: int, c0: int, h: int, w: int) -> bool:
    rr0 = max(0, r0 - 1)
    cc0 = max(0, c0 - 1)
    rr1 = min(occupied.shape[0], r0 + h + 1)
    cc1 = min(occupied.shape[1], c0 + w + 1)
    return not bool(occupied[rr0:rr1, cc0:cc1].any())


def _draw_frame(grid: np.ndarray, r0: int, c0: int, h: int, w: int, side_colors: dict[str, int]) -> None:
    grid[r0, c0 + 1 : c0 + w - 1] = side_colors["top"]
    grid[r0 + h - 1, c0 + 1 : c0 + w - 1] = side_colors["bottom"]
    grid[r0 + 1 : r0 + h - 1, c0] = side_colors["left"]
    grid[r0 + 1 : r0 + h - 1, c0 + w - 1] = side_colors["right"]

    grid[r0, c0] = random.choice((side_colors["top"], side_colors["left"]))
    grid[r0, c0 + w - 1] = random.choice((side_colors["top"], side_colors["right"]))
    grid[r0 + h - 1, c0] = random.choice((side_colors["bottom"], side_colors["left"]))
    grid[r0 + h - 1, c0 + w - 1] = random.choice((side_colors["bottom"], side_colors["right"]))


def _colored_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.full(mask.shape, BG, dtype=int)
    patch[mask] = color
    return patch


def _render_expected(
    grid: np.ndarray,
    frame_top: int,
    frame_left: int,
    objects: list[dict[str, object]],
) -> np.ndarray:
    out = grid.copy()
    inner_top = frame_top + 1
    inner_left = frame_left + 1

    for obj in objects:
        patch = np.asarray(obj["patch"], dtype=int)
        r0, c0 = obj["pos"]
        mask = patch != BG
        view = out[inner_top + r0 : inner_top + r0 + patch.shape[0], inner_left + c0 : inner_left + c0 + patch.shape[1]]
        view[mask] = BG

    for obj in objects:
        patch = _apply_mode(np.asarray(obj["patch"], dtype=int), str(obj["mode"]))
        r0, c0 = obj["pos"]
        mask = patch != BG
        view = out[inner_top + r0 : inner_top + r0 + patch.shape[0], inner_left + c0 : inner_left + c0 + patch.shape[1]]
        view[mask] = patch[mask]

    return out


def _choose_slots(slots: list[dict[str, object]]) -> list[dict[str, object]] | None:
    by_mode: dict[str, list[dict[str, object]]] = {"hflip": [], "vflip": [], "rot180": []}
    for slot in slots:
        mode = str(slot["mode"])
        if mode in by_mode:
            by_mode[mode].append(slot)
    if not all(by_mode.values()):
        return None

    chosen = [random.choice(by_mode["hflip"]), random.choice(by_mode["vflip"]), random.choice(by_mode["rot180"])]
    seen = {id(slot) for slot in chosen}
    remaining = [slot for slot in slots if id(slot) not in seen]
    target_count = random.randint(3, min(5, len(slots)))
    extra_count = min(target_count - 3, len(remaining))
    if extra_count > 0:
        chosen.extend(random.sample(remaining, extra_count))

    chosen.sort(key=lambda slot: 0 if slot["kind"] == "corner" else 1)
    return chosen


def _gen_attempt() -> np.ndarray | None:
    frame_h = random.randint(9, 15)
    frame_w = random.randint(9, 15)
    inner_h = frame_h - 2
    inner_w = frame_w - 2

    side_colors = _random_frame_sides()
    slots = _slot_specs(side_colors)
    chosen_slots = _choose_slots(slots)
    if chosen_slots is None:
        return None

    margin_top = random.randint(1, 3)
    margin_bottom = random.randint(1, 3)
    margin_left = random.randint(1, 3)
    margin_right = random.randint(1, 3)
    grid_h = frame_h + margin_top + margin_bottom
    grid_w = frame_w + margin_left + margin_right
    frame_top = margin_top
    frame_left = margin_left

    occupied = np.zeros((inner_h, inner_w), dtype=bool)
    object_colors = random.sample(OBJECT_COLORS, len(chosen_slots))
    objects: list[dict[str, object]] = []

    for slot, color in zip(chosen_slots, object_colors):
        max_h = min(5, inner_h - 1)
        max_w = min(5, inner_w - 1)
        if slot["kind"] == "side" and slot["name"] in ("top", "bottom"):
            max_w = min(max_w, inner_w - 2)
        if slot["kind"] == "side" and slot["name"] in ("left", "right"):
            max_h = min(max_h, inner_h - 2)

        placed = False
        for _ in range(200):
            mask = _make_object_mask(str(slot["mode"]), max_h, max_w)
            if mask is None:
                return None
            candidates = _position_candidates(slot, mask.shape, inner_h, inner_w)
            random.shuffle(candidates)
            for r0, c0 in candidates:
                if _bbox_has_gap(occupied, r0, c0, mask.shape[0], mask.shape[1]):
                    occupied[r0 : r0 + mask.shape[0], c0 : c0 + mask.shape[1]] = True
                    patch = _colored_patch(mask, color)
                    objects.append({"patch": patch, "pos": (r0, c0), "mode": slot["mode"]})
                    placed = True
                    break
            if placed:
                break
        if not placed:
            return None

    grid = np.full((grid_h, grid_w), BG, dtype=int)
    _draw_frame(grid, frame_top, frame_left, frame_h, frame_w, side_colors)

    inner_top = frame_top + 1
    inner_left = frame_left + 1
    for obj in objects:
        patch = np.asarray(obj["patch"], dtype=int)
        r0, c0 = obj["pos"]
        mask = patch != BG
        view = grid[inner_top + r0 : inner_top + r0 + patch.shape[0], inner_left + c0 : inner_left + c0 + patch.shape[1]]
        view[mask] = patch[mask]

    expected = _render_expected(grid, frame_top, frame_left, objects)
    if np.array_equal(expected, grid):
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
