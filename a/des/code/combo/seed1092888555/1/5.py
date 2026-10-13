import numpy as np


BG = 9  # maroon
PALETTE = (1, 2, 3, 4, 5, 6, 7, 8)
DIRS = ("up", "right", "down", "left")
NEIGHBORS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))

BASE_MASKS = (
    np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [1, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 1],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=bool,
    ),
)

_RNG = np.random.default_rng()


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
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
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def crop_mask(coords: list[tuple[int, int]]) -> np.ndarray:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    out = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        out[r - r0, c - c0] = True
    return out


def centroid_direction(mask: np.ndarray) -> str:
    coords = np.argwhere(mask)
    center = np.array([(mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0], dtype=float)
    delta = coords.mean(axis=0) - center
    dr, dc = float(delta[0]), float(delta[1])
    if abs(dr) >= abs(dc):
        return "up" if dr < 0 else "down"
    return "left" if dc < 0 else "right"


def rotate_to_direction(mask: np.ndarray, target_dir: str) -> np.ndarray:
    src_dir = centroid_direction(mask)
    src_idx = DIRS.index(src_dir)
    dst_idx = DIRS.index(target_dir)
    k = (dst_idx - src_idx) % 4
    return np.rot90(mask, -k)


def nearest_edge_direction(r: int, c: int, h: int, w: int) -> str | None:
    dists = {
        "up": r,
        "down": h - 1 - r,
        "left": c,
        "right": w - 1 - c,
    }
    min_dist = min(dists.values())
    winners = [name for name, dist in dists.items() if dist == min_dist]
    if len(winners) != 1:
        return None
    return winners[0]


def mask_cells(mask: np.ndarray, anchor_r: int, anchor_c: int) -> list[tuple[int, int]]:
    center_r = mask.shape[0] // 2
    center_c = mask.shape[1] // 2
    cells: list[tuple[int, int]] = []
    for r, c in np.argwhere(mask):
        cells.append((anchor_r + int(r) - center_r, anchor_c + int(c) - center_c))
    return cells


def stamp(grid: np.ndarray, mask: np.ndarray, anchor_r: int, anchor_c: int, color: int) -> None:
    for r, c in mask_cells(mask, anchor_r, anchor_c):
        if 0 <= r < grid.shape[0] and 0 <= c < grid.shape[1]:
            grid[r, c] = color


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    out = np.full_like(g, bg)

    for color in sorted(int(x) for x in np.unique(g) if int(x) != bg):
        comps = find_components(g == color)
        source_comps = [comp for comp in comps if len(comp) > 1]
        marker_comps = [comp for comp in comps if len(comp) == 1]
        assert len(source_comps) == 1
        assert 1 <= len(marker_comps) <= 2

        source_mask = crop_mask(source_comps[0])
        for marker in marker_comps:
            r, c = marker[0]
            edge_dir = nearest_edge_direction(int(r), int(c), *g.shape)
            assert edge_dir is not None
            rotated = rotate_to_direction(source_mask, edge_dir)
            stamp(out, rotated, int(r), int(c), color)

    return out


def _shape_variants() -> list[np.ndarray]:
    variants: list[np.ndarray] = []
    for base in BASE_MASKS:
        for k in range(4):
            mask = np.rot90(base, k)
            if centroid_direction(mask) not in DIRS:
                continue
            if not np.array_equal(mask, mask[::-1, ::-1]):
                variants.append(mask.copy())
    return variants


MASK_LIBRARY = _shape_variants()


def _reserve_cells(blocked: np.ndarray, cells: list[tuple[int, int]], pad: int) -> None:
    h, w = blocked.shape
    for r, c in cells:
        for rr in range(max(0, r - pad), min(h, r + pad + 1)):
            for cc in range(max(0, c - pad), min(w, c + pad + 1)):
                blocked[rr, cc] = True


def _choose_marker_anchor(
    blocked_output: np.ndarray,
    mask: np.ndarray,
    edge_dir: str,
) -> tuple[int, int] | None:
    h, w = blocked_output.shape
    center_r = mask.shape[0] // 2
    center_c = mask.shape[1] // 2
    candidates: list[tuple[int, int]] = []
    for r in range(center_r, h - (mask.shape[0] - 1 - center_r)):
        for c in range(center_c, w - (mask.shape[1] - 1 - center_c)):
            if nearest_edge_direction(r, c, h, w) != edge_dir:
                continue
            cells = mask_cells(mask, r, c)
            if any(blocked_output[rr, cc] for rr, cc in cells):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[_RNG.integers(len(candidates))]


def _place_source(
    blocked_input: np.ndarray,
    mask: np.ndarray,
) -> tuple[int, int] | None:
    h, w = blocked_input.shape
    center_r = mask.shape[0] // 2
    center_c = mask.shape[1] // 2
    candidates: list[tuple[int, int]] = []
    for r in range(center_r, h - (mask.shape[0] - 1 - center_r)):
        for c in range(center_c, w - (mask.shape[1] - 1 - center_c)):
            if r in (0, h - 1) or c in (0, w - 1):
                continue
            cells = mask_cells(mask, r, c)
            if any(blocked_input[rr, cc] for rr, cc in cells):
                continue
            candidates.append((r, c))
    if not candidates:
        return None
    return candidates[_RNG.integers(len(candidates))]


def _gen_attempt() -> np.ndarray | None:
    h = int(_RNG.integers(13, 21))
    w = int(_RNG.integers(13, 21))
    color_count = int(_RNG.integers(2, 5))
    colors = list(_RNG.choice(PALETTE, size=color_count, replace=False))

    blocked_output = np.zeros((h, w), dtype=bool)
    planned: list[dict[str, object]] = []
    expected = np.full((h, w), BG, dtype=int)

    for color in colors:
        source_mask = MASK_LIBRARY[int(_RNG.integers(len(MASK_LIBRARY)))].copy()
        marker_count = 1 + int(_RNG.integers(2))
        marker_dirs = [DIRS[int(_RNG.integers(4))] for _ in range(marker_count)]

        markers: list[tuple[int, int]] = []
        for edge_dir in marker_dirs:
            rotated = rotate_to_direction(source_mask, edge_dir)
            anchor = _choose_marker_anchor(blocked_output, rotated, edge_dir)
            if anchor is None:
                return None
            cells = mask_cells(rotated, *anchor)
            _reserve_cells(blocked_output, cells, pad=1)
            markers.append(anchor)
            stamp(expected, rotated, anchor[0], anchor[1], int(color))

        planned.append(
            {
                "color": int(color),
                "source_mask": source_mask,
                "markers": markers,
            }
        )

    grid = np.full((h, w), BG, dtype=int)
    blocked_input = np.zeros((h, w), dtype=bool)

    # Place markers first so sources can avoid them.
    for item in planned:
        color = int(item["color"])
        for r, c in item["markers"]:
            if blocked_input[r, c]:
                return None
            grid[r, c] = color
            _reserve_cells(blocked_input, [(r, c)], pad=1)

    for item in planned:
        color = int(item["color"])
        source_mask = np.asarray(item["source_mask"], dtype=bool)
        anchor = _place_source(blocked_input, source_mask)
        if anchor is None:
            return None
        cells = mask_cells(source_mask, anchor[0], anchor[1])
        for r, c in cells:
            grid[r, c] = color
        _reserve_cells(blocked_input, cells, pad=1)

    # Validate the hidden assumptions the solver relies on.
    try:
        produced = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(produced, expected):
        return None
    if len(np.unique(grid)) < 3:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(500):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample")
