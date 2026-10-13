import random

import numpy as np


BACKGROUND = 0
WALL_COLOR = 1
PALETTE = tuple(range(2, 10))

SYM_LR = "mirror_lr"
SYM_UD = "mirror_ud"
SYM_ROT = "rot180"
SYM_NONE = "none"
SYMMETRY_TYPES = (SYM_LR, SYM_UD, SYM_ROT, SYM_NONE)
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

# Generator configuration.
LEGEND_HEIGHT_CHOICES = (4, 5)
ROOM_ROW_RANGE = (2, 4)
ROOM_COL_RANGE = (3, 4)
ROOM_SIZE_RANGE = (5, 8)
LEGEND_MAX_CELLS = 4
SEED_MAX_CELLS = 6
MAX_SHAPE_ATTEMPTS = 300
MAX_SAMPLE_ATTEMPTS = 300


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    assert len(coords) > 0, "expected a nonempty mask"
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return np.asarray(mask[r0 : r1 + 1, c0 : c1 + 1], dtype=bool)


def _classify_mask(mask: np.ndarray) -> str:
    m = _tight_mask(np.asarray(mask, dtype=bool))
    lr = np.array_equal(m, np.fliplr(m))
    ud = np.array_equal(m, np.flipud(m))
    rot = np.array_equal(m, np.rot90(m, 2))

    if lr and not ud and not rot:
        return SYM_LR
    if ud and not lr and not rot:
        return SYM_UD
    if rot and not lr and not ud:
        return SYM_ROT
    if not lr and not ud and not rot:
        return SYM_NONE
    raise AssertionError("shape has ambiguous symmetry")


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(coords, dtype=int))
    return comps


def _find_wall_system(g: np.ndarray) -> tuple[int, int]:
    candidates = []
    for r in range(g.shape[0]):
        vals = np.unique(g[r, :])
        if vals.size == 1 and int(vals[0]) != BACKGROUND:
            candidates.append((r, int(vals[0])))
    assert candidates, "expected at least one full non-background row"

    top_wall, wall_color = candidates[0]
    wall_rows = [r for r in range(top_wall, g.shape[0]) if np.all(g[r, :] == wall_color)]
    wall_cols = [c for c in range(g.shape[1]) if np.all(g[top_wall:, c] == wall_color)]
    assert len(wall_rows) >= 2 and len(wall_cols) >= 2, "expected a rectangular wall system"
    return int(top_wall), int(wall_color)


def _legend_mapping(g: np.ndarray, top_wall: int, wall_color: int) -> dict[str, int]:
    band = g[:top_wall, :]
    colors = [int(color) for color in np.unique(band) if int(color) not in {BACKGROUND, wall_color}]
    mapping: dict[str, int] = {}
    for color in colors:
        kind = _classify_mask(band == color)
        assert kind not in mapping, "legend must contain exactly one pattern per symmetry type"
        mapping[kind] = color
    assert set(mapping) == set(SYMMETRY_TYPES), "legend must encode all four symmetry types"
    return mapping


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each room according to the symmetry class of its seed pattern.

    The top band is a legend: each legend color draws a small example shape whose
    own symmetry class determines which room-seed class that color represents.
    Below the first full wall row, blue walls partition the canvas into rooms.
    Every room contains exactly one monochrome seed pattern on black cells.
    Replace the black cells in that room with the legend color for the matching
    symmetry class, while preserving the seed pixels and the walls.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    top_wall, wall_color = _find_wall_system(g)
    legend = _legend_mapping(g, top_wall, wall_color)

    room_mask = g[top_wall + 1 :, :] != wall_color
    rooms = _components(room_mask)
    assert rooms, "expected at least one room"

    out = g.copy()
    row_offset = top_wall + 1
    for room in rooms:
        abs_rows = room[:, 0] + row_offset
        abs_cols = room[:, 1]
        values = g[abs_rows, abs_cols]
        seed_colors = {int(v) for v in values if int(v) != BACKGROUND}
        assert len(seed_colors) == 1, "each room must contain exactly one seed color"
        seed_color = next(iter(seed_colors))

        seed_coords = [(int(r), int(c)) for r, c in zip(abs_rows, abs_cols) if int(g[r, c]) == seed_color]
        rs = [r for r, _ in seed_coords]
        cs = [c for _, c in seed_coords]
        r0, r1 = min(rs), max(rs)
        c0, c1 = min(cs), max(cs)
        seed_mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
        for r, c in seed_coords:
            seed_mask[r - r0, c - c0] = True

        fill_color = legend[_classify_mask(seed_mask)]
        for r, c in zip(abs_rows, abs_cols):
            if int(g[r, c]) == BACKGROUND:
                out[int(r), int(c)] = fill_color

    return out


def _orbit_groups(height: int, width: int, kind: str) -> list[list[tuple[int, int]]]:
    seen: set[tuple[int, int]] = set()
    groups: list[list[tuple[int, int]]] = []
    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue
            if kind == SYM_LR:
                orbit = {(r, c), (r, width - 1 - c)}
            elif kind == SYM_UD:
                orbit = {(r, c), (height - 1 - r, c)}
            elif kind == SYM_ROT:
                orbit = {(r, c), (height - 1 - r, width - 1 - c)}
            else:
                raise AssertionError("unknown symmetry kind")
            orbit_list = sorted(orbit)
            seen.update(orbit_list)
            groups.append(orbit_list)
    return groups


def _fallback_masks(kind: str) -> list[np.ndarray]:
    pools = {
        SYM_LR: (
            np.array([[1, 0, 1], [0, 1, 0]], dtype=bool),
            np.array([[1, 1, 1], [0, 1, 0]], dtype=bool),
        ),
        SYM_UD: (
            np.array([[1, 0], [0, 1], [1, 0]], dtype=bool),
            np.array([[1, 0], [1, 1], [1, 0]], dtype=bool),
        ),
        SYM_ROT: (
            np.array([[1, 0], [0, 1]], dtype=bool),
            np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=bool),
        ),
        SYM_NONE: (
            np.array([[1, 1], [1, 0]], dtype=bool),
            np.array([[1, 0, 0], [1, 1, 0]], dtype=bool),
        ),
    }
    return [mask.copy() for mask in pools[kind]]


def _fits(mask: np.ndarray, max_cells: int, max_h: int, max_w: int) -> bool:
    return int(mask.sum()) <= max_cells and mask.shape[0] <= max_h and mask.shape[1] <= max_w


def _random_mask(kind: str, max_cells: int, max_h: int, max_w: int) -> np.ndarray:
    min_h = 2
    min_w = 2
    if kind == SYM_LR:
        min_w = 3
    if kind == SYM_UD:
        min_h = 3
    assert max_h >= min_h and max_w >= min_w, "bbox is too small for the requested symmetry"

    for _ in range(MAX_SHAPE_ATTEMPTS):
        height = random.randint(min_h, max_h)
        width = random.randint(min_w, max_w)
        prob = random.uniform(0.25, 0.75)
        mask = np.zeros((height, width), dtype=bool)

        if kind == SYM_NONE:
            for r in range(height):
                for c in range(width):
                    if random.random() < prob:
                        mask[r, c] = True
        else:
            for orbit in _orbit_groups(height, width, kind):
                if random.random() < prob:
                    for r, c in orbit:
                        mask[r, c] = True

        if not mask.any():
            continue
        mask = _tight_mask(mask)
        cells = int(mask.sum())
        if not (2 <= cells <= max_cells):
            continue
        try:
            if _classify_mask(mask) == kind:
                return mask
        except AssertionError:
            continue

    candidates = [mask for mask in _fallback_masks(kind) if _fits(mask, max_cells, max_h, max_w)]
    assert candidates, "no fallback shape fits the requested bounds"
    return random.choice(candidates).copy()


def _split_extra(total: int, parts: int) -> list[int]:
    buckets = [0] * parts
    for _ in range(total):
        buckets[random.randrange(parts)] += 1
    random.shuffle(buckets)
    return buckets


def _build_spans(sizes: list[int], first_wall: int) -> tuple[list[tuple[int, int]], list[int], int]:
    spans = []
    walls = [first_wall]
    cursor = first_wall + 1
    for size in sizes:
        end = cursor + int(size)
        spans.append((cursor, end))
        walls.append(end)
        cursor = end + 1
    total = walls[-1] + 1
    return spans, walls, total


def _draw_mask(canvas: np.ndarray, top: int, left: int, mask: np.ndarray, color: int) -> None:
    patch = canvas[top : top + mask.shape[0], left : left + mask.shape[1]]
    assert patch.shape == mask.shape
    assert np.all(patch[mask] == BACKGROUND)
    patch[mask] = color


def _make_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    legend_height = random.choice(LEGEND_HEIGHT_CHOICES)
    room_rows = random.randint(*ROOM_ROW_RANGE)
    room_cols = random.randint(*ROOM_COL_RANGE)
    room_heights = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_rows)]
    room_widths = [random.randint(*ROOM_SIZE_RANGE) for _ in range(room_cols)]

    row_spans, wall_rows, total_h = _build_spans(room_heights, legend_height)
    col_spans, wall_cols, total_w = _build_spans(room_widths, 0)

    grid = np.full((total_h, total_w), BACKGROUND, dtype=int)
    grid[wall_rows, :] = WALL_COLOR
    grid[legend_height:, wall_cols] = WALL_COLOR

    color_order = random.sample(PALETTE, len(PALETTE))
    legend_palette = color_order[:4]
    seed_palette = color_order[4:]
    legend_kind_order = random.sample(list(SYMMETRY_TYPES), len(SYMMETRY_TYPES))
    legend_color_by_kind = {kind: color for kind, color in zip(legend_kind_order, legend_palette)}
    legend_masks = {
        kind: _random_mask(kind, LEGEND_MAX_CELLS, min(3, legend_height), 3)
        for kind in SYMMETRY_TYPES
    }

    ordered_masks = [legend_masks[kind] for kind in legend_kind_order]
    min_gaps = [1] * (len(ordered_masks) + 1)
    required_width = sum(mask.shape[1] for mask in ordered_masks) + sum(min_gaps)
    if required_width > total_w:
        return None
    gaps = [base + extra for base, extra in zip(min_gaps, _split_extra(total_w - required_width, len(min_gaps)))]

    cursor = gaps[0]
    for index, kind in enumerate(legend_kind_order):
        mask = legend_masks[kind]
        top = random.randint(0, legend_height - mask.shape[0])
        _draw_mask(grid, top, cursor, mask, legend_color_by_kind[kind])
        cursor += mask.shape[1] + gaps[index + 1]

    room_positions = [(ri, ci) for ri in range(room_rows) for ci in range(room_cols)]
    room_kinds = list(SYMMETRY_TYPES)
    while len(room_kinds) < len(room_positions):
        room_kinds.append(random.choice(SYMMETRY_TYPES))
    random.shuffle(room_kinds)

    out = grid.copy()
    for (ri, ci), kind in zip(room_positions, room_kinds):
        r0, r1 = row_spans[ri]
        c0, c1 = col_spans[ci]
        room_h = r1 - r0
        room_w = c1 - c0
        max_h = min(4, room_h - 2)
        max_w = min(4, room_w - 2)
        if max_h < 2 or max_w < 2:
            return None

        seed_mask = _random_mask(kind, SEED_MAX_CELLS, max_h, max_w)
        seed_color = random.choice(seed_palette)
        top = r0 + random.randint(1, room_h - seed_mask.shape[0] - 1)
        left = c0 + random.randint(1, room_w - seed_mask.shape[1] - 1)
        _draw_mask(grid, top, left, seed_mask, seed_color)

        room_in = grid[r0:r1, c0:c1]
        room_out = out[r0:r1, c0:c1]
        room_out[room_in == BACKGROUND] = legend_color_by_kind[kind]
        room_out[room_in == seed_color] = seed_color

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return grid, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_SAMPLE_ATTEMPTS):
        sample = _make_attempt()
        if sample is not None:
            return sample[0]
    raise RuntimeError("failed to generate a valid input")
