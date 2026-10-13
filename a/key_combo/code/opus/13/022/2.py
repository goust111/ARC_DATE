import numpy as np
from collections import Counter, defaultdict, deque


BG = 0
BODY = 1
_RNG = np.random.default_rng()

DIRS = {"T": (-1, 0), "B": (1, 0), "L": (0, -1), "R": (0, 1)}
OPPOSITE = {"T": "B", "B": "T", "L": "R", "R": "L"}


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def paste(dst: np.ndarray, patch: np.ndarray, r0: int, c0: int, *, transparent: int | None = None) -> None:
    h, w = patch.shape
    view = dst[r0 : r0 + h, c0 : c0 + w]
    if transparent is None:
        view[:] = patch
        return
    mask = patch != int(transparent)
    view[mask] = patch[mask]


def find_components(grid: np.ndarray, *, bg: int = 0) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    out = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or g[r, c] == bg:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] != bg:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            out.append({"coords": arr, "bbox": bbox_coords(arr), "area": int(arr.shape[0])})
    return out


def dominant_body_color(grid: np.ndarray, bg: int) -> int:
    fg = np.asarray(grid, dtype=int)
    vals = fg[fg != bg]
    counts = Counter(map(int, vals))
    return counts.most_common(1)[0][0]


def complete_sprite(sprite: np.ndarray) -> np.ndarray:
    out = np.asarray(sprite, dtype=int).copy()
    h, w = out.shape
    progress = True
    while progress and np.any(out == BG):
        progress = False
        zeros = np.argwhere(out == BG)
        for rr, cc in zeros:
            r = int(rr)
            c = int(cc)
            orbit = {
                (r, c),
                (h - 1 - r, c),
                (r, w - 1 - c),
                (h - 1 - r, w - 1 - c),
            }
            vals = {int(out[orow, ocol]) for orow, ocol in orbit if out[orow, ocol] != BG}
            if len(vals) == 1:
                value = vals.pop()
                for orow, ocol in orbit:
                    if out[orow, ocol] == BG:
                        out[orow, ocol] = value
                        progress = True
    assert not np.any(out == BG), "every missing pixel must be solvable from its mirror orbit"
    return out


def boundary_side(shape: tuple[int, int], r: int, c: int) -> str:
    h, w = shape
    sides = []
    if r == 0:
        sides.append("T")
    if r == h - 1:
        sides.append("B")
    if c == 0:
        sides.append("L")
    if c == w - 1:
        sides.append("R")
    assert len(sides) == 1, "connectors must lie on a single exposed side"
    return sides[0]


def marker_map(sprite: np.ndarray, body: int) -> dict[int, list[tuple[int, int, tuple[int, int]]]]:
    out: dict[int, list[tuple[int, int, tuple[int, int]]]] = defaultdict(list)
    h, w = sprite.shape
    for r in range(h):
        for c in range(w):
            color = int(sprite[r, c])
            if color in (BG, body):
                continue
            side = boundary_side((h, w), r, c)
            out[color].append((r, c, DIRS[side]))
    return dict(out)


def candidate_offsets(
    anchor_markers: dict[int, list[tuple[int, int, tuple[int, int]]]],
    new_markers: dict[int, list[tuple[int, int, tuple[int, int]]]],
    color: int,
) -> list[tuple[int, int]]:
    offsets = set()
    for ar, ac, adir in anchor_markers[color]:
        for br, bc, bdir in new_markers[color]:
            if bdir != (-adir[0], -adir[1]):
                continue
            offsets.add((ar + adir[0] - br, ac + adir[1] - bc))
    assert offsets, f"no opposite-facing marker pair for color {color}"
    return sorted(offsets)


def sprite_cells(sprite: np.ndarray, top_left: tuple[int, int], *, bg: int = 0) -> set[tuple[int, int]]:
    base_r, base_c = top_left
    cells = set()
    for rr, cc in np.argwhere(np.asarray(sprite) != bg):
        cells.add((base_r + int(rr), base_c + int(cc)))
    return cells


def overlaps(sprite: np.ndarray, top_left: tuple[int, int], occupied: set[tuple[int, int]], *, bg: int = 0) -> bool:
    return any(cell in occupied for cell in sprite_cells(sprite, top_left, bg=bg))


def compose_output(
    sprites: list[np.ndarray],
    placements: dict[int, tuple[int, int]],
    *,
    bg: int = 0,
) -> np.ndarray:
    all_cells = []
    for idx, sprite in enumerate(sprites):
        r0, c0 = placements[idx]
        for rr, cc in np.argwhere(sprite != bg):
            all_cells.append((r0 + int(rr), c0 + int(cc)))
    min_r = min(r for r, _ in all_cells)
    min_c = min(c for _, c in all_cells)
    max_r = max(r for r, _ in all_cells)
    max_c = max(c for _, c in all_cells)
    out = np.full((max_r - min_r + 1, max_c - min_c + 1), bg, dtype=int)
    for idx, sprite in enumerate(sprites):
        r0, c0 = placements[idx]
        paste(out, sprite, r0 - min_r, c0 - min_c, transparent=bg)
    return out


def assemble_completed_sprites(
    sprites: list[np.ndarray],
    *,
    bg: int = 0,
    body: int = BODY,
) -> tuple[np.ndarray, dict[int, tuple[int, int]]]:
    markers = [marker_map(sprite, body) for sprite in sprites]
    color_to_objects: dict[int, list[int]] = defaultdict(list)
    for idx, obj_markers in enumerate(markers):
        for color in obj_markers:
            color_to_objects[color].append(idx)

    ordered_colors = sorted(color_to_objects)
    for color in ordered_colors:
        assert len(color_to_objects[color]) == 2, "each connector color must belong to exactly two objects"

    def choose_initial_pair(a: int, b: int) -> tuple[int, int]:
        degree_a = len(markers[a])
        degree_b = len(markers[b])
        if degree_a != degree_b:
            return (a, b) if degree_a < degree_b else (b, a)
        tie_a = (sprites[a].shape[0] * sprites[a].shape[1], sprites[a].shape, a)
        tie_b = (sprites[b].shape[0] * sprites[b].shape[1], sprites[b].shape, b)
        return (a, b) if tie_a <= tie_b else (b, a)

    def search(
        step: int,
        placements: dict[int, tuple[int, int]],
        occupied: set[tuple[int, int]],
    ) -> dict[int, tuple[int, int]] | None:
        if step == len(ordered_colors):
            return dict(placements) if len(placements) == len(sprites) else None

        color = ordered_colors[step]
        a, b = color_to_objects[color]

        if a in placements and b in placements:
            current = (placements[b][0] - placements[a][0], placements[b][1] - placements[a][1])
            if current in candidate_offsets(markers[a], markers[b], color):
                return search(step + 1, placements, occupied)
            return None

        if not placements:
            anchor, new = choose_initial_pair(a, b)
            next_placements = {anchor: (0, 0)}
            next_occupied = sprite_cells(sprites[anchor], (0, 0), bg=bg)
            for dr, dc in candidate_offsets(markers[anchor], markers[new], color):
                pos = (dr, dc)
                if overlaps(sprites[new], pos, next_occupied, bg=bg):
                    continue
                placed = dict(next_placements)
                placed[new] = pos
                result = search(step + 1, placed, next_occupied | sprite_cells(sprites[new], pos, bg=bg))
                if result is not None:
                    return result
            return None

        if (a in placements) ^ (b in placements):
            anchor, new = (a, b) if a in placements else (b, a)
            candidate_positions = []
            for dr, dc in candidate_offsets(markers[anchor], markers[new], color):
                pos = (placements[anchor][0] + dr, placements[anchor][1] + dc)
                if not overlaps(sprites[new], pos, occupied, bg=bg):
                    candidate_positions.append(pos)
            for pos in sorted(set(candidate_positions)):
                placed = dict(placements)
                placed[new] = pos
                result = search(step + 1, placed, occupied | sprite_cells(sprites[new], pos, bg=bg))
                if result is not None:
                    return result
            return None

        return None

    placements = search(0, {}, set())
    assert placements is not None, "failed to find a valid assembly"
    return compose_output(sprites, placements, bg=bg), placements


def main(grid: np.ndarray) -> np.ndarray:
    """Complete each broken symmetric object to reveal its hidden connector markers, then dock equal-colored connectors and crop the assembly.

    Input structure:
    - The grid has black background and 3-4 separated 4-connected foreground objects.
    - Each object is mostly blue, lies in an odd-sized bounding box, and is symmetric across its horizontal and vertical midlines before some pixels were erased to black.
    - Non-blue boundary pixels are connector markers; a connector color belongs to exactly two objects globally.

    Rule:
    - Crop every foreground object to its bounding box and infer the two reflection axes from that box.
    - For each black cell inside the box, gather its horizontal, vertical, and double-reflection orbit cells; if the known non-black colors in that orbit are unique, fill the missing cell with that color.
    - After completion, treat each boundary non-blue pixel as a connector marker and process connector colors in ascending palette order.
    - For each color, translate the new object so one of its opposite-facing same-colored markers touches the already assembled group edge-to-edge, preserving orientations, then return the tight crop of the final assembly.

    Key insight:
    - Symmetry completion is the decoding step: the erased black pixels hide the connector markers, and only the restored mirror orbit reveals how the objects should be assembled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = BG
    body = dominant_body_color(g, bg)

    components = find_components(g, bg=bg)
    assert 3 <= len(components) <= 4

    completed = []
    for comp in components:
        sprite = crop_bbox(g, comp["bbox"])
        completed.append(complete_sprite(sprite))

    out, _placements = assemble_completed_sprites(completed, bg=bg, body=body)
    return out


def connector_cells(shape: tuple[int, int], axis: str) -> list[tuple[int, int]]:
    h, w = shape
    if axis == "H":
        return [(h // 2, 0), (h // 2, w - 1)]
    assert axis == "V"
    return [(0, w // 2), (h - 1, w // 2)]


def build_completed_object(shape: tuple[int, int], connectors: list[tuple[int, str]]) -> np.ndarray:
    sprite = np.full(shape, BODY, dtype=int)
    for color, axis in connectors:
        for r, c in connector_cells(shape, axis):
            sprite[r, c] = int(color)
    return sprite


def orbit_of(shape: tuple[int, int], r: int, c: int) -> list[tuple[int, int]]:
    h, w = shape
    orbit = {
        (r, c),
        (h - 1 - r, c),
        (r, w - 1 - c),
        (h - 1 - r, w - 1 - c),
    }
    return sorted(orbit)


def masked_object(completed: np.ndarray, connectors: list[tuple[int, str]]) -> np.ndarray:
    h, w = completed.shape
    connector_cells_flat = {cell for _color, axis in connectors for cell in connector_cells((h, w), axis)}

    for _ in range(120):
        sprite = completed.copy()
        missing: set[tuple[int, int]] = set()

        for color, axis in connectors:
            cells = connector_cells((h, w), axis)
            missing.add(cells[int(_RNG.integers(0, len(cells)))])

        target_missing = int(_RNG.integers(max(2, len(missing)), 5))
        blue_candidates = []
        for r in range(h):
            for c in range(w):
                if (r, c) in connector_cells_flat:
                    continue
                if sprite[r, c] != BODY:
                    continue
                blue_candidates.append((r, c))
        _RNG.shuffle(blue_candidates)

        for r, c in blue_candidates:
            if len(missing) >= target_missing:
                break
            orbit = set(orbit_of((h, w), r, c))
            if sum((cell in missing) for cell in orbit) >= len(orbit) - 1:
                continue
            missing.add((r, c))

        if not (2 <= len(missing) <= 4):
            continue
        for r, c in missing:
            sprite[r, c] = BG
        if len(find_components(sprite, bg=BG)) == 1:
            return sprite

    raise AssertionError("failed to mask object without breaking connectivity")


def boxes_far_enough(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 3) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    if ar1 + gap < br0 or br1 + gap < ar0:
        return True
    if ac1 + gap < bc0 or bc1 + gap < ac0:
        return True
    return False


def scatter_sprites(sprites: list[np.ndarray]) -> np.ndarray | None:
    max_h = max(sprite.shape[0] for sprite in sprites)
    max_w = max(sprite.shape[1] for sprite in sprites)
    min_side = max(22, max_h * 2 + 5, max_w * 2 + 5)
    canvas_h = int(_RNG.integers(min_side, 31))
    canvas_w = int(_RNG.integers(min_side, 31))
    grid = np.full((canvas_h, canvas_w), BG, dtype=int)

    order = list(_RNG.permutation(len(sprites)))
    boxes = []
    for idx in order:
        sprite = sprites[idx]
        h, w = sprite.shape
        placed = False
        for _ in range(300):
            r0 = int(_RNG.integers(0, canvas_h - h + 1))
            c0 = int(_RNG.integers(0, canvas_w - w + 1))
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(boxes_far_enough(box, other, gap=3) for other in boxes):
                paste(grid, sprite, r0, c0, transparent=BG)
                boxes.append(box)
                placed = True
                break
        if not placed:
            return None
    return grid


def generate_case() -> tuple[np.ndarray, np.ndarray]:
    object_count = int(_RNG.choice([3, 4]))
    colors = list(range(2, 2 + object_count - 1))
    start_axis = str(_RNG.choice(["H", "V"]))
    other_axis = "V" if start_axis == "H" else "H"
    axes = [start_axis if i % 2 == 0 else other_axis for i in range(object_count - 1)]

    connector_specs = [[] for _ in range(object_count)]
    for edge_idx, color in enumerate(colors):
        axis = axes[edge_idx]
        connector_specs[edge_idx].append((color, axis))
        connector_specs[edge_idx + 1].append((color, axis))

    shapes = []
    for _ in range(object_count):
        shapes.append((int(_RNG.choice([5, 7])), int(_RNG.choice([5, 7]))))

    completed = [build_completed_object(shape, specs) for shape, specs in zip(shapes, connector_specs)]
    expected, _placements = assemble_completed_sprites(completed, bg=BG, body=BODY)
    assert max(expected.shape) <= 30

    masked = [masked_object(sprite, specs) for sprite, specs in zip(completed, connector_specs)]
    grid = scatter_sprites(masked)
    assert grid is not None
    return grid, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            grid, expected = generate_case()
        except AssertionError:
            continue
        produced = main(grid)
        if np.array_equal(produced, expected):
            return grid
    raise RuntimeError("failed to generate a valid symmetric-docking ARC input")
