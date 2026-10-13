import random

import numpy as np


BLACK = 0
GREY = 5
SHAPES = ("plus", "T", "L")

GEN_ATTEMPTS = 300
GEN_MARKER_SIZE = 3
GEN_STAMP_SIZES = (5, 7, 9)
GEN_CONTAINER_COUNTS = (3, 4)
GEN_EDGE_SIDES = ("top", "bottom", "left", "right")
GEN_COLORS = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_BUFFER = 2


def _most_common_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
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
                rr, cc = stack.pop()
                comp.append((rr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rs = [r for r, _ in coords]
    cs = [c for _, c in coords]
    return min(rs), min(cs), max(rs), max(cs)


def _touches_edge(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _perimeter_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _shape_mask(kind: str, size: int) -> np.ndarray:
    assert size >= 3 and size % 2 == 1
    mid = size // 2
    mask = np.zeros((size, size), dtype=bool)
    if kind == "plus":
        mask[mid, :] = True
        mask[:, mid] = True
    elif kind == "T":
        mask[0, :] = True
        mask[:, mid] = True
    elif kind == "L":
        mask[:, 0] = True
        mask[-1, :] = True
    else:
        raise AssertionError(f"unknown shape kind: {kind}")
    return mask


def _classify_shape(mask: np.ndarray) -> str:
    mask = np.asarray(mask, dtype=bool)
    coords = np.argwhere(mask)
    assert len(coords) > 0, "shape mask must be non-empty"
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    cropped = mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]
    h, w = cropped.shape
    assert h == w, "shape bbox must be square"
    for kind in SHAPES:
        base = _shape_mask(kind, h)
        for k in range(4):
            if np.array_equal(cropped, np.rot90(base, k)):
                return kind
    raise AssertionError("unrecognized shape")


def _rotate_clockwise(arr: np.ndarray, quarter_turns: int) -> np.ndarray:
    return np.rot90(arr, (-quarter_turns) % 4)


def _find_templates(grid: np.ndarray, bg: int) -> dict[str, list[np.ndarray]]:
    h, w = grid.shape
    templates: dict[str, list[np.ndarray]] = {}
    colors = [int(v) for v in np.unique(grid) if int(v) not in (bg, BLACK, GREY)]
    for color in colors:
        for coords in _components(grid == color):
            if not _touches_edge(coords, h, w):
                continue
            r0, c0, r1, c1 = _bbox(coords)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
            kind = _classify_shape(patch == color)
            templates.setdefault(kind, []).append(patch)
    return templates


def _container_gap_count(
    grid: np.ndarray,
    bg: int,
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
) -> int | None:
    h, w = grid.shape
    if r0 < 0 or c0 < 0 or r0 + outer_h > h or c0 + outer_w > w:
        return None
    patch = grid[r0 : r0 + outer_h, c0 : c0 + outer_w]
    border = _perimeter_mask(outer_h, outer_w)
    border_vals = patch[border]
    if np.any((border_vals != BLACK) & (border_vals != bg)):
        return None
    if np.any(patch[1:-1, 1:-1] == BLACK):
        return None
    gap_count = int(border.sum() - np.count_nonzero(border_vals == BLACK))
    if not 1 <= gap_count <= 3:
        return None
    return gap_count


def _find_containers(
    grid: np.ndarray,
    bg: int | None = None,
    templates: dict[str, list[np.ndarray]] | None = None,
) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g) if bg is None else int(bg)
    templates = _find_templates(g, bg) if templates is None else templates

    containers: list[dict[str, object]] = []
    for coords in _components(g == GREY):
        mr0, mc0, mr1, mc1 = _bbox(coords)
        marker_mask = np.zeros((mr1 - mr0 + 1, mc1 - mc0 + 1), dtype=bool)
        for r, c in coords:
            marker_mask[r - mr0, c - mc0] = True
        kind = _classify_shape(marker_mask)
        marker_h, marker_w = marker_mask.shape

        candidates: list[dict[str, object]] = []
        for patch in templates.get(kind, []):
            inner_h, inner_w = patch.shape
            if inner_h < marker_h or inner_w < marker_w:
                continue
            if (inner_h - marker_h) % 2 != 0 or (inner_w - marker_w) % 2 != 0:
                continue
            r0 = mr0 - 1 - (inner_h - marker_h) // 2
            c0 = mc0 - 1 - (inner_w - marker_w) // 2
            gap_count = _container_gap_count(g, bg, r0, c0, inner_h + 2, inner_w + 2)
            if gap_count is None:
                continue
            candidates.append(
                {
                    "bbox": (r0, c0, r0 + inner_h + 1, c0 + inner_w + 1),
                    "gaps": gap_count,
                    "kind": kind,
                    "template": patch,
                }
            )

        assert len(candidates) == 1, "each grey marker must identify exactly one container"
        containers.append(candidates[0])

    containers.sort(key=lambda d: (d["bbox"][0], d["bbox"][1]))
    return containers


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    templates = _find_templates(g, bg)
    assert all(kind in templates for kind in SHAPES), "missing stamp template"

    out = g.copy()
    for container in _find_containers(g, bg=bg, templates=templates):
        r0, c0, r1, c1 = container["bbox"]
        gap_count = int(container["gaps"])
        patch = np.asarray(container["template"], dtype=int)
        out[r0 + 1 : r1, c0 + 1 : c1] = _rotate_clockwise(patch, gap_count)

    return out


def _paste(grid: np.ndarray, patch: np.ndarray, r0: int, c0: int) -> None:
    h, w = patch.shape
    grid[r0 : r0 + h, c0 : c0 + w] = patch


def _mark_rect(mask: np.ndarray, r0: int, c0: int, h: int, w: int, buffer_size: int) -> None:
    rr0 = max(0, r0 - buffer_size)
    cc0 = max(0, c0 - buffer_size)
    rr1 = min(mask.shape[0], r0 + h + buffer_size)
    cc1 = min(mask.shape[1], c0 + w + buffer_size)
    mask[rr0:rr1, cc0:cc1] = True


def _rect_clear(mask: np.ndarray, r0: int, c0: int, h: int, w: int, buffer_size: int) -> bool:
    rr0 = max(0, r0 - buffer_size)
    cc0 = max(0, c0 - buffer_size)
    rr1 = min(mask.shape[0], r0 + h + buffer_size)
    cc1 = min(mask.shape[1], c0 + w + buffer_size)
    return not mask[rr0:rr1, cc0:cc1].any()


def _colored_shape_patch(kind: str, size: int, bg: int, color: int) -> np.ndarray:
    patch = np.full((size, size), bg, dtype=int)
    patch[_shape_mask(kind, size)] = color
    return patch


def _choose_gap_cells(box_size: int, count: int, rng: random.Random) -> list[tuple[int, int]]:
    positions = list(range(2, box_size - 2))
    assert positions, "container must be large enough for non-corner gaps"
    sides = rng.sample(["top", "right", "bottom", "left"], count)
    chosen: list[tuple[int, int]] = []
    for side in sides:
        pos = rng.choice(positions)
        if side == "top":
            chosen.append((0, pos))
        elif side == "right":
            chosen.append((pos, box_size - 1))
        elif side == "bottom":
            chosen.append((box_size - 1, pos))
        else:
            chosen.append((pos, 0))
    return chosen


def _template_layout(
    rng: random.Random,
    h: int,
    w: int,
    size: int,
    side: str,
) -> list[tuple[int, int]]:
    gap1 = rng.randint(1, 3)
    gap2 = rng.randint(1, 3)
    if side in ("top", "bottom"):
        left_max = w - (3 * size + gap1 + gap2)
        if left_max < 0:
            raise RuntimeError("grid too small for top/bottom templates")
        left = rng.randint(0, left_max)
        row = 0 if side == "top" else h - size
        return [
            (row, left),
            (row, left + size + gap1),
            (row, left + 2 * size + gap1 + gap2),
        ]

    top_max = h - (3 * size + gap1 + gap2)
    if top_max < 0:
        raise RuntimeError("grid too small for left/right templates")
    top = rng.randint(0, top_max)
    col = 0 if side == "left" else w - size
    return [
        (top, col),
        (top + size + gap1, col),
        (top + 2 * size + gap1 + gap2, col),
    ]


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    stamp_size = rng.choice(GEN_STAMP_SIZES)
    container_count = rng.choice(GEN_CONTAINER_COUNTS)
    side = rng.choice(GEN_EDGE_SIDES)
    box_size = stamp_size + 2
    reserve = 2 * box_size + 2 * GEN_BUFFER * 2 + 4
    if side in ("top", "bottom"):
        min_h = stamp_size + 3 + reserve
        min_w = max(3 * stamp_size + 8, reserve)
        h = rng.randint(min_h, min_h + 8)
        w = rng.randint(min_w, min_w + 10)
    else:
        min_h = max(3 * stamp_size + 8, reserve)
        min_w = stamp_size + 3 + reserve
        h = rng.randint(min_h, min_h + 10)
        w = rng.randint(min_w, min_w + 8)

    bg = rng.choice(GEN_COLORS)
    template_colors = rng.sample([c for c in GEN_COLORS if c != bg], 3)

    grid = np.full((h, w), bg, dtype=int)
    occupied = np.zeros((h, w), dtype=bool)

    shape_order = list(SHAPES)
    rng.shuffle(shape_order)
    template_positions = _template_layout(rng, h, w, stamp_size, side)
    templates_by_kind: dict[str, np.ndarray] = {}
    for kind, color, (r0, c0) in zip(shape_order, template_colors, template_positions):
        patch = _colored_shape_patch(kind, stamp_size, bg, color)
        _paste(grid, patch, r0, c0)
        templates_by_kind[kind] = patch
        _mark_rect(occupied, r0, c0, stamp_size, stamp_size, GEN_BUFFER)

    positions = [
        (r0, c0)
        for r0 in range(1, h - box_size)
        for c0 in range(1, w - box_size)
        if _rect_clear(occupied, r0, c0, box_size, box_size, GEN_BUFFER)
    ]
    rng.shuffle(positions)

    chosen_positions: list[tuple[int, int]] = []
    for r0, c0 in positions:
        if not _rect_clear(occupied, r0, c0, box_size, box_size, GEN_BUFFER):
            continue
        chosen_positions.append((r0, c0))
        _mark_rect(occupied, r0, c0, box_size, box_size, GEN_BUFFER)
        if len(chosen_positions) == container_count:
            break
    if len(chosen_positions) != container_count:
        return None

    kinds = ["plus", "T", "L"]
    if container_count == 4:
        kinds.append(rng.choice(SHAPES))
    rng.shuffle(kinds)

    gap_counts = [1, 2, 3]
    if container_count == 4:
        gap_counts.append(rng.choice([1, 2, 3]))
    rng.shuffle(gap_counts)

    marker_start = (stamp_size - GEN_MARKER_SIZE) // 2
    container_specs: list[tuple[int, int, str, int]] = []

    for (r0, c0), kind, gap_count in zip(chosen_positions, kinds, gap_counts):
        grid[r0, c0 : c0 + box_size] = BLACK
        grid[r0 + box_size - 1, c0 : c0 + box_size] = BLACK
        grid[r0 : r0 + box_size, c0] = BLACK
        grid[r0 : r0 + box_size, c0 + box_size - 1] = BLACK

        for gr, gc in _choose_gap_cells(box_size, gap_count, rng):
            grid[r0 + gr, c0 + gc] = bg

        marker = _shape_mask(kind, GEN_MARKER_SIZE)
        if kind != "plus":
            marker = np.rot90(marker, rng.randint(0, 3))
        mr = r0 + 1 + marker_start
        mc = c0 + 1 + marker_start
        marker_view = grid[mr : mr + GEN_MARKER_SIZE, mc : mc + GEN_MARKER_SIZE]
        marker_view[marker] = GREY
        container_specs.append((r0, c0, kind, gap_count))

    expected = grid.copy()
    for r0, c0, kind, gap_count in container_specs:
        rotated = _rotate_clockwise(templates_by_kind[kind], gap_count)
        expected[r0 + 1 : r0 + 1 + stamp_size, c0 + 1 : c0 + 1 + stamp_size] = rotated

    if set(_find_templates(grid, bg)) != set(SHAPES):
        return None
    if len(_find_containers(grid)) != container_count:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
