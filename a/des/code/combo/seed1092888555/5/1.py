import numpy as np


BG = 0
PALETTE = tuple(range(1, 10))
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MAX_GEN_ATTEMPTS = 400
MAX_PLACE_ATTEMPTS = 200


def _color_components(grid: np.ndarray, bg: int = BG) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": (min(rows), min(cols), max(rows), max(cols)),
                }
            )
    return comps


def _is_frame(comp: dict) -> bool:
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border = set()
    for c in range(c0, c1 + 1):
        border.add((r0, c))
        border.add((r1, c))
    for r in range(r0, r1 + 1):
        border.add((r, c0))
        border.add((r, c1))
    return set(comp["coords"]) == border


def _extract_groups(grid: np.ndarray) -> list[dict]:
    comps = _color_components(grid, BG)
    frames = [comp for comp in comps if _is_frame(comp)]
    frame_by_bbox = {tuple(comp["bbox"]): comp for comp in frames}
    roots: list[tuple[int, int, int, int]] = []
    for comp in frames:
        r0, c0, r1, c1 = comp["bbox"]
        inner_bbox = (r0 + 1, c0 + 1, r1 - 1, c1 - 1)
        if inner_bbox not in frame_by_bbox:
            roots.append(tuple(comp["bbox"]))

    groups = []
    frame_cells = {coord for comp in frames for coord in comp["coords"]}
    for root_bbox in sorted(roots):
        chain = []
        bbox = root_bbox
        while bbox in frame_by_bbox:
            chain.append(frame_by_bbox[bbox])
            r0, c0, r1, c1 = bbox
            bbox = (r0 - 1, c0 - 1, r1 + 1, c1 + 1)
        inner = chain[0]["bbox"]
        ir0, ic0, ir1, ic1 = inner
        object_coords = []
        for r in range(ir0 + 1, ir1):
            for c in range(ic0 + 1, ic1):
                if int(grid[r, c]) != BG and (r, c) not in frame_cells:
                    object_coords.append((r, c))
        if not object_coords:
            continue
        groups.append({"frames": chain, "object_coords": object_coords})
    return groups


def main(grid: np.ndarray) -> np.ndarray:
    """Rotate each center object clockwise once per nested frame layer."""
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    groups = _extract_groups(g)
    for group in groups:
        turns = len(group["frames"]) % 4
        inner_bbox = group["frames"][0]["bbox"]
        fr0, fc0, fr1, fc1 = inner_bbox
        cavity = g[fr0 + 1 : fr1, fc0 + 1 : fc1]
        out[fr0 + 1 : fr1, fc0 + 1 : fc1] = np.rot90(cavity, -turns)
    return out


def _mask_to_patch(mask: np.ndarray, color: int) -> np.ndarray:
    patch = np.zeros(mask.shape, dtype=int)
    patch[mask] = color
    return patch


def _connected_shape(size: int, cells: int, rng: np.random.Generator) -> np.ndarray:
    while True:
        mask = np.zeros((size, size), dtype=bool)
        seed = (int(rng.integers(size)), int(rng.integers(size)))
        mask[seed] = True
        frontier = {seed}
        while int(mask.sum()) < cells:
            base = list(frontier)[int(rng.integers(len(frontier)))]
            dr, dc = DIRS4[int(rng.integers(4))]
            nr, nc = base[0] + dr, base[1] + dc
            if 0 <= nr < size and 0 <= nc < size:
                mask[nr, nc] = True
                frontier.add((nr, nc))
        rows, cols = np.where(mask)
        mask = mask[rows.min() : rows.max() + 1, cols.min() : cols.max() + 1]
        h, w = mask.shape
        square = np.zeros((size, size), dtype=bool)
        off_r = (size - h) // 2
        off_c = (size - w) // 2
        square[off_r : off_r + h, off_c : off_c + w] = mask
        rot1 = np.rot90(square, -1)
        rot2 = np.rot90(square, -2)
        rot3 = np.rot90(square, -3)
        if np.array_equal(square, rot1) or np.array_equal(square, rot2) or np.array_equal(square, rot3):
            continue
        if _is_single_pixel_frame(square):
            continue
        return square


def _is_single_pixel_frame(mask: np.ndarray) -> bool:
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return False
    r0, c0, r1, c1 = rows.min(), cols.min(), rows.max(), cols.max()
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    sub = mask[r0 : r1 + 1, c0 : c1 + 1]
    return np.array_equal(sub, border)


def _draw_structure(grid: np.ndarray, top: int, left: int, layers: int, patch: np.ndarray, frame_colors: list[int]) -> None:
    size = patch.shape[0]
    for idx in range(layers):
        side = size + 2 + 2 * idx
        r0 = top + (layers - 1 - idx)
        c0 = left + (layers - 1 - idx)
        r1 = r0 + side - 1
        c1 = c0 + side - 1
        color = int(frame_colors[idx])
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
    inner_top = top + layers
    inner_left = left + layers
    grid[inner_top : inner_top + size, inner_left : inner_left + size] = np.maximum(
        grid[inner_top : inner_top + size, inner_left : inner_left + size], patch
    )


def _can_place(occ: np.ndarray, top: int, left: int, side: int, pad: int = 2) -> bool:
    h, w = occ.shape
    r0 = max(0, top - pad)
    c0 = max(0, left - pad)
    r1 = min(h, top + side + pad)
    c1 = min(w, left + side + pad)
    return not bool(occ[r0:r1, c0:c1].any())


def _mark_occ(occ: np.ndarray, top: int, left: int, side: int) -> None:
    occ[top : top + side, left : left + side] = True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(18, 31))
    w = int(rng.integers(18, 31))
    grid = np.zeros((h, w), dtype=int)
    occ = np.zeros((h, w), dtype=bool)

    group_count = int(rng.integers(1, 5))
    placed = 0
    seen_layers = set()
    for _ in range(group_count * 3):
        if placed >= group_count:
            break
        layers = int(rng.integers(1, 4))
        size = int(rng.choice([3, 4, 5]))
        cells = int(rng.integers(3, min(7, size * size)))
        mask = _connected_shape(size, cells, rng)
        colors = list(rng.choice(PALETTE, size=layers + 1, replace=False))
        obj_color = int(colors[0])
        frame_colors = [int(c) for c in colors[1:]]
        side = size + 2 * layers
        patch = _mask_to_patch(mask, obj_color)
        for _ in range(MAX_PLACE_ATTEMPTS):
            top = int(rng.integers(1, h - side))
            left = int(rng.integers(1, w - side))
            if not _can_place(occ, top, left, side, pad=2):
                continue
            _draw_structure(grid, top, left, layers, patch, frame_colors)
            _mark_occ(occ, top, left, side)
            placed += 1
            seen_layers.add(layers)
            break

    if placed == 0:
        return None
    if len(seen_layers) == 1 and placed >= 2:
        return None
    solved = main(grid)
    if np.array_equal(solved, grid):
        return None
    groups = _extract_groups(grid)
    if len(groups) != placed:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid nested-frame rotation puzzle")
