import random
import numpy as np


_DIRS = ("up", "right", "down", "left")
_ROT_K = {"up": 0, "right": 3, "down": 2, "left": 1}
_NEIGHBORS8 = [
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
]


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _connected_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in _NEIGHBORS8:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _extract_patch(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
    for r, c in coords:
        patch[r - r0, c - c0] = int(grid[r, c])
    return patch, (r0, c0, r1, c1)


def _direction_from_marker(marker_r: int, marker_c: int, bbox: tuple[int, int, int, int]) -> str:
    r0, c0, r1, c1 = bbox
    cr = (r0 + r1) / 2.0
    cc = (c0 + c1) / 2.0
    dr = marker_r - cr
    dc = marker_c - cc
    assert abs(dr) != abs(dc), "marker direction must be unambiguous"
    if abs(dr) > abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _stamp(out: np.ndarray, patch: np.ndarray, top: int, left: int, bg: int) -> None:
    h, w = patch.shape
    assert 0 <= top and 0 <= left
    assert top + h <= out.shape[0] and left + w <= out.shape[1]
    region = out[top : top + h, left : left + w]
    mask = patch != bg
    region[mask] = patch[mask]


def _sprite_info(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict:
    patch, bbox = _extract_patch(grid, coords, bg)
    colors, counts = np.unique([int(grid[r, c]) for r, c in coords], return_counts=True)
    assert len(colors) == 2, "each source sprite must use one body color plus one marker color"
    singleton = [int(color) for color, count in zip(colors, counts) if int(count) == 1]
    assert len(singleton) == 1, "each source sprite must contain exactly one embedded marker pixel"
    marker_color = singleton[0]
    marker_local = tuple(np.argwhere(patch == marker_color)[0])
    direction = _direction_from_marker(marker_local[0] + bbox[0], marker_local[1] + bbox[1], bbox)
    return {
        "patch": patch,
        "bbox": bbox,
        "marker_color": marker_color,
        "direction": direction,
    }


def _parse_scene(grid: np.ndarray) -> tuple[int, list[dict], dict[int, list[tuple[int, int]]]]:
    bg = _get_bg(grid)
    comps = _connected_components(grid != bg)
    sprites: list[dict] = []
    isolated: dict[int, list[tuple[int, int]]] = {}
    for comp in comps:
        if len(comp) == 1:
            r, c = comp[0]
            isolated.setdefault(int(grid[r, c]), []).append((r, c))
        else:
            sprites.append(_sprite_info(grid, comp, bg))
    assert sprites, "expected at least one source sprite"
    marker_colors = [sprite["marker_color"] for sprite in sprites]
    assert len(marker_colors) == len(set(marker_colors)), "marker colors must identify sprites uniquely"
    sprites.sort(key=lambda sprite: sprite["bbox"])
    for coords in isolated.values():
        coords.sort()
    return bg, sprites, isolated


def main(grid: np.ndarray) -> np.ndarray:
    """
    Copy each source sprite onto every matching isolated marker, aligning by the marker pixel
    and rotating according to the source marker's direction relative to the sprite bbox center.
    """
    g = np.asarray(grid, dtype=int)
    bg, sprites, isolated = _parse_scene(g)
    out = g.copy()

    for sprite in sprites:
        patch = sprite["patch"]
        marker_color = int(sprite["marker_color"])
        rotated = np.rot90(patch, _ROT_K[sprite["direction"]])
        marker_pos = tuple(np.argwhere(rotated == marker_color)[0])
        for target_r, target_c in isolated.get(marker_color, []):
            top = int(target_r - marker_pos[0])
            left = int(target_c - marker_pos[1])
            _stamp(out, rotated, top, left, bg)

    return out


# --------------- generator ---------------

_GEN_COLORS = tuple(range(10))
_GEN_MAX_ATTEMPTS = 400
_GEN_MAX_SIDE = 32


def _middle_indices(length: int) -> list[int]:
    if length % 2 == 1:
        return [length // 2]
    return [length // 2 - 1, length // 2]


def _step_path(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    tr, tc = b
    cells = [(r, c)]
    while (r, c) != (tr, tc):
        if r < tr:
            r += 1
        elif r > tr:
            r -= 1
        if c < tc:
            c += 1
        elif c > tc:
            c -= 1
        cells.append((r, c))
    return cells


def _neighbors_in_bounds(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    cells = []
    for dr, dc in _NEIGHBORS8:
        nr = r + dr
        nc = c + dc
        if 0 <= nr < h and 0 <= nc < w:
            cells.append((nr, nc))
    return cells


def _grow_body(
    h: int,
    w: int,
    allowed: set[tuple[int, int]],
    mandatory: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    body = set()
    root = mandatory[0]
    for point in mandatory:
        for cell in _step_path(root, point):
            assert cell in allowed
            body.add(cell)

    target = random.randint(max(len(body), 6), len(allowed))
    while len(body) < target:
        frontier = []
        for cell in allowed:
            if cell in body:
                continue
            if any(nei in body for nei in _neighbors_in_bounds(cell[0], cell[1], h, w)):
                frontier.append(cell)
        if not frontier:
            break
        body.add(random.choice(frontier))
    return body


def _generate_sprite(direction: str, bg: int, body_color: int, marker_color: int) -> np.ndarray:
    if direction in ("up", "down"):
        h = random.randint(4, 6)
        w = random.randint(3, 6)
        mid_c = random.choice(_middle_indices(w))
        if direction == "up":
            marker = (0, mid_c)
            allowed = {(r, c) for r in range(1, h) for c in range(w)}
            mandatory = [
                (1, mid_c),
                (random.randint(1, h - 1), 0),
                (random.randint(1, h - 1), w - 1),
                (h - 1, random.randint(0, w - 1)),
            ]
        else:
            marker = (h - 1, mid_c)
            allowed = {(r, c) for r in range(0, h - 1) for c in range(w)}
            mandatory = [
                (h - 2, mid_c),
                (random.randint(0, h - 2), 0),
                (random.randint(0, h - 2), w - 1),
                (0, random.randint(0, w - 1)),
            ]
    else:
        h = random.randint(3, 6)
        w = random.randint(4, 6)
        mid_r = random.choice(_middle_indices(h))
        if direction == "left":
            marker = (mid_r, 0)
            allowed = {(r, c) for r in range(h) for c in range(1, w)}
            mandatory = [
                (mid_r, 1),
                (0, random.randint(1, w - 1)),
                (h - 1, random.randint(1, w - 1)),
                (random.randint(0, h - 1), w - 1),
            ]
        else:
            marker = (mid_r, w - 1)
            allowed = {(r, c) for r in range(h) for c in range(0, w - 1)}
            mandatory = [
                (mid_r, w - 2),
                (0, random.randint(0, w - 2)),
                (h - 1, random.randint(0, w - 2)),
                (random.randint(0, h - 1), 0),
            ]

    body = _grow_body(h, w, allowed, mandatory)
    patch = np.full((h, w), bg, dtype=int)
    for r, c in body:
        patch[r, c] = body_color
    patch[marker] = marker_color
    return patch


def _footprint(patch: np.ndarray, bg: int, top: int, left: int) -> list[tuple[int, int]]:
    return [(top + int(r), left + int(c)) for r, c in np.argwhere(patch != bg)]


def _isolated(occupied: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    h, w = occupied.shape
    cell_set = set(cells)
    for r, c in cell_set:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if occupied[r, c] and (r, c) not in cell_set:
            return False
        for nr in range(max(0, r - 1), min(h, r + 2)):
            for nc in range(max(0, c - 1), min(w, c + 2)):
                if occupied[nr, nc] and (nr, nc) not in cell_set:
                    return False
    return True


def _mark(occupied: np.ndarray, cells: list[tuple[int, int]]) -> None:
    for r, c in cells:
        occupied[r, c] = True


def _candidate_positions(height: int, width: int, patch_h: int, patch_w: int) -> list[tuple[int, int]]:
    positions = [(r, c) for r in range(height - patch_h + 1) for c in range(width - patch_w + 1)]
    random.shuffle(positions)
    return positions


def _try_place_patch(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    bg: int,
) -> tuple[int, int] | None:
    h, w = patch.shape
    for top, left in _candidate_positions(grid.shape[0], grid.shape[1], h, w):
        cells = _footprint(patch, bg, top, left)
        if not _isolated(occupied, cells):
            continue
        _stamp(grid, patch, top, left, bg)
        _mark(occupied, cells)
        return top, left
    return None


def _place_patch_at(
    grid: np.ndarray,
    occupied: np.ndarray,
    patch: np.ndarray,
    bg: int,
    top: int,
    left: int,
) -> None:
    cells = _footprint(patch, bg, top, left)
    assert _isolated(occupied, cells)
    _stamp(grid, patch, top, left, bg)
    _mark(occupied, cells)


def _try_place_target_marker(
    inp: np.ndarray,
    expected: np.ndarray,
    input_occupied: np.ndarray,
    output_occupied: np.ndarray,
    patch: np.ndarray,
    marker_color: int,
    bg: int,
) -> bool:
    h, w = patch.shape
    marker_r, marker_c = tuple(np.argwhere(patch == marker_color)[0])
    for top, left in _candidate_positions(inp.shape[0], inp.shape[1], h, w):
        copy_cells = _footprint(patch, bg, top, left)
        target = (top + int(marker_r), left + int(marker_c))
        if not _isolated(output_occupied, copy_cells):
            continue
        if not _isolated(input_occupied, [target]):
            continue
        inp[target] = marker_color
        expected_top = target[0] - int(marker_r)
        expected_left = target[1] - int(marker_c)
        _stamp(expected, patch, expected_top, expected_left, bg)
        _mark(input_occupied, [target])
        _mark(output_occupied, copy_cells)
        return True
    return False


def _try_place_single_pixel(
    grid: np.ndarray,
    expected: np.ndarray,
    input_occupied: np.ndarray,
    output_occupied: np.ndarray,
    color: int,
) -> bool:
    positions = [(r, c) for r in range(grid.shape[0]) for c in range(grid.shape[1])]
    random.shuffle(positions)
    for r, c in positions:
        if not _isolated(input_occupied, [(r, c)]):
            continue
        if not _isolated(output_occupied, [(r, c)]):
            continue
        grid[r, c] = color
        expected[r, c] = color
        input_occupied[r, c] = True
        output_occupied[r, c] = True
        return True
    return False


def _build_attempt() -> np.ndarray | None:
    bg = random.choice(_GEN_COLORS)
    n_sources = random.randint(2, 4)
    target_counts = [random.choice((0, 1, 2)) for _ in range(n_sources)]
    if sum(target_counts) == 0:
        target_counts[random.randrange(n_sources)] = 1

    usable = [color for color in _GEN_COLORS if color != bg]
    random.shuffle(usable)
    max_unmatched = min(2, len(usable) - 2 * n_sources)
    if max_unmatched < 0:
        return None
    n_unmatched = random.randint(0, max_unmatched)

    body_colors = usable[:n_sources]
    marker_colors = usable[n_sources : 2 * n_sources]
    unmatched_colors = usable[2 * n_sources : 2 * n_sources + n_unmatched]

    min_side = 16 + n_sources + sum(target_counts)
    if min_side > _GEN_MAX_SIDE:
        return None
    height = random.randint(min_side, _GEN_MAX_SIDE)
    width = random.randint(min_side, _GEN_MAX_SIDE)

    inp = np.full((height, width), bg, dtype=int)
    expected = inp.copy()
    input_occupied = np.zeros((height, width), dtype=bool)
    output_occupied = np.zeros((height, width), dtype=bool)

    directions = random.sample(list(_DIRS), n_sources)
    specs = []
    for body_color, marker_color, direction, target_count in zip(body_colors, marker_colors, directions, target_counts):
        patch = _generate_sprite(direction, bg, body_color, marker_color)
        rotated = np.rot90(patch, _ROT_K[direction])
        specs.append(
            {
                "patch": patch,
                "rotated": rotated,
                "marker_color": marker_color,
                "direction": direction,
                "target_count": target_count,
            }
        )

    specs.sort(key=lambda spec: int(np.count_nonzero(spec["patch"] != bg)), reverse=True)

    for spec in specs:
        source_pos = _try_place_patch(inp, input_occupied, spec["patch"], bg)
        if source_pos is None:
            return None
        _place_patch_at(expected, output_occupied, spec["patch"], bg, source_pos[0], source_pos[1])

    target_jobs = []
    for spec in specs:
        for _ in range(spec["target_count"]):
            target_jobs.append(spec)
    random.shuffle(target_jobs)
    target_jobs.sort(key=lambda spec: int(np.count_nonzero(spec["rotated"] != bg)), reverse=True)

    for spec in target_jobs:
        if not _try_place_target_marker(
            inp,
            expected,
            input_occupied,
            output_occupied,
            spec["rotated"],
            int(spec["marker_color"]),
            bg,
        ):
            return None

    for color in unmatched_colors:
        if not _try_place_single_pixel(inp, expected, input_occupied, output_occupied, color):
            return None

    produced = main(inp)
    if not np.array_equal(produced, expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(_GEN_MAX_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the marker-anchored sprite-copy task")
