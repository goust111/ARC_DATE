import numpy as np


DIR_ORDER = ("up", "right", "down", "left")
DIR_TO_STEP = {
    "up": (-1, 0),
    "right": (0, 1),
    "down": (1, 0),
    "left": (0, -1),
}

FRAME_SIZE = 5
INTERIOR_SIZE = FRAME_SIZE - 2

BASE_SEED_MASKS = (
    np.array(
        [
            [0, 1, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
    np.array(
        [
            [0, 1, 0],
            [1, 0, 1],
            [0, 0, 0],
        ],
        dtype=bool,
    ),
)

GEN_RETRIES = 500
PLACEMENT_RETRIES = 250


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _dir_index(name: str) -> int:
    return DIR_ORDER.index(name)


def _rotate_mask(mask: np.ndarray, from_dir: str, to_dir: str) -> np.ndarray:
    turns = (_dir_index(from_dir) - _dir_index(to_dir)) % 4
    return np.rot90(mask, turns)


def _tight_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _frame_border_cells(r0: int, c0: int, h: int, w: int) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for c in range(c0, c0 + w):
        cells.add((r0, c))
        cells.add((r0 + h - 1, c))
    for r in range(r0, r0 + h):
        cells.add((r, c0))
        cells.add((r, c0 + w - 1))
    return cells


def _color_components(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != color:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(coords)

    return components


def _infer_head_direction(seed_coords: list[tuple[int, int]], shape: tuple[int, int]) -> str:
    rows = np.array([r for r, _ in seed_coords], dtype=float)
    cols = np.array([c for _, c in seed_coords], dtype=float)
    h, w = shape
    row_scale = max(1, h - 1)
    col_scale = max(1, w - 1)

    scores = {
        "up": float(np.sum((h - 1 - rows) / row_scale)),
        "down": float(np.sum(rows / row_scale)),
        "left": float(np.sum((w - 1 - cols) / col_scale)),
        "right": float(np.sum(cols / col_scale)),
    }
    best = max(scores.values())
    winners = [name for name, value in scores.items() if abs(value - best) < 1e-9]
    if len(winners) == 1:
        return winners[0]

    center_r = (h - 1) / 2.0
    center_c = (w - 1) / 2.0
    bias_r = (rows.mean() - center_r) / max(center_r, 0.5)
    bias_c = (cols.mean() - center_c) / max(center_c, 0.5)
    if abs(bias_r) > abs(bias_c):
        return "down" if bias_r > 0 else "up"
    return "right" if bias_c > 0 else "left"


def _bud_position(frame: dict, side: str) -> tuple[int, int]:
    r0, c0, r1, c1 = frame["bbox"]
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if side == "up":
        return (r0 - 1, mid_c)
    if side == "down":
        return (r1 + 1, mid_c)
    if side == "left":
        return (mid_r, c0 - 1)
    return (mid_r, c1 + 1)


def _growth_cells(
    frame: dict,
    grid_shape: tuple[int, int],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    h, w = grid_shape
    trunk_cells: set[tuple[int, int]] = set()
    seed_cells: set[tuple[int, int]] = set()
    frame_color = int(frame["frame_color"])
    seed_color = int(frame["seed_color"])
    _ = frame_color, seed_color  # kept for symmetry with caller-side rendering

    tight_mask = frame["tight_seed_mask"]
    seed_len = int(np.count_nonzero(tight_mask))
    head_dir = frame["head_dir"]

    for side in frame["buds"]:
        bud_r, bud_c = _bud_position(frame, side)
        dr, dc = DIR_TO_STEP[side]
        for step in range(1, seed_len + 1):
            rr = bud_r + dr * step
            cc = bud_c + dc * step
            if 0 <= rr < h and 0 <= cc < w:
                trunk_cells.add((rr, cc))

        tip_r = bud_r + dr * seed_len
        tip_c = bud_c + dc * seed_len
        rotated = _rotate_mask(tight_mask, head_dir, side)
        patch_h, patch_w = rotated.shape
        if side == "up":
            top = tip_r - patch_h
            left = tip_c - patch_w // 2
        elif side == "down":
            top = tip_r + 1
            left = tip_c - patch_w // 2
        elif side == "left":
            top = tip_r - patch_h // 2
            left = tip_c - patch_w
        else:
            top = tip_r - patch_h // 2
            left = tip_c + 1

        for rr, cc in np.argwhere(rotated):
            ar = top + int(rr)
            ac = left + int(cc)
            if 0 <= ar < h and 0 <= ac < w:
                seed_cells.add((ar, ac))

    return trunk_cells, seed_cells


def _apply_growth(out: np.ndarray, frame: dict) -> None:
    trunk_cells, seed_cells = _growth_cells(frame, out.shape)
    for r, c in trunk_cells:
        out[r, c] = int(frame["frame_color"])
    for r, c in seed_cells:
        out[r, c] = int(frame["seed_color"])


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames: list[dict] = []
    for value in np.unique(grid):
        color = int(value)
        if color == bg:
            continue
        for coords in _color_components(grid, color):
            coord_set = set(coords)
            rows = [r for r, _ in coords]
            cols = [c for _, c in coords]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            frame_h = r1 - r0 + 1
            frame_w = c1 - c0 + 1
            if frame_h < 3 or frame_w < 3:
                continue
            if coord_set != _frame_border_cells(r0, c0, frame_h, frame_w):
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            if interior.size == 0:
                continue
            seed_mask = (interior != bg) & (interior != color)
            if not seed_mask.any():
                continue
            seed_colors = np.unique(interior[seed_mask])
            if len(seed_colors) != 1:
                continue
            seed_color = int(seed_colors[0])
            seed_coords = [tuple(map(int, coord)) for coord in np.argwhere(interior == seed_color)]
            head_dir = _infer_head_direction(seed_coords, interior.shape)
            tight_seed_mask = _tight_mask(interior == seed_color)

            buds: list[str] = []
            for side in DIR_ORDER:
                br, bc = _bud_position({"bbox": (r0, c0, r1, c1)}, side)
                if 0 <= br < grid.shape[0] and 0 <= bc < grid.shape[1]:
                    if int(grid[br, bc]) not in (bg, color, seed_color):
                        buds.append(side)

            if not buds:
                continue

            frames.append(
                {
                    "bbox": (r0, c0, r1, c1),
                    "frame_color": color,
                    "seed_color": seed_color,
                    "head_dir": head_dir,
                    "tight_seed_mask": tight_seed_mask,
                    "buds": tuple(buds),
                }
            )

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow fixed-length frame-colored branches from bud markers and attach rotated seed copies at each branch tip.

    Input structure:
    - The background is black, and the foreground contains several single-color rectangular frames.
    - Inside each frame sits a 1-3 pixel seed pattern of a second color.
    - One or more single-pixel bud markers of a third color sit just outside side midpoints of the frame.

    Rule:
    - For each frame, infer the seed's head direction from the side that its occupied pixels lean toward inside the frame.
    - For every bud, grow a straight frame-colored branch outward along that side's normal direction; the branch length equals the seed pixel count.
    - At the cell just beyond the branch tip, copy the seed's tight shape, rotating it so the inferred head direction matches the branch direction.
    - Keep the original frame, original seed, and bud markers unchanged; clip any grown pixels that would leave the grid.

    Key insight:
    - The seed plays two roles at once: its asymmetry encodes a cardinal direction, and its pixel count sets the branch length before a rotated copy is stamped at the end.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    out = g.copy()

    for frame in _find_frames(g, bg):
        _apply_growth(out, frame)

    return out


def _expand_cells(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    expanded.add((nr, nc))
    return expanded


def _sample_buds(rng: np.random.Generator) -> tuple[str, ...]:
    bud_count = int(rng.integers(1, 5))
    order = list(rng.permutation(DIR_ORDER))
    return tuple(sorted(order[:bud_count], key=_dir_index))


def _seed_mask_for_direction(template_idx: int, head_dir: str) -> np.ndarray:
    return _rotate_mask(BASE_SEED_MASKS[template_idx], "up", head_dir)


def _object_input_cells(frame: dict) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cells = _frame_border_cells(r0, c0, r1 - r0 + 1, c1 - c0 + 1)
    for rr, cc in np.argwhere(frame["seed_mask"]):
        cells.add((r0 + 1 + int(rr), c0 + 1 + int(cc)))
    for side in frame["buds"]:
        cells.add(_bud_position(frame, side))
    return cells


def _object_output_cells(frame: dict, grid_shape: tuple[int, int]) -> set[tuple[int, int]]:
    cells = set(_object_input_cells(frame))
    trunk_cells, seed_cells = _growth_cells(frame, grid_shape)
    cells.update(trunk_cells)
    cells.update(seed_cells)
    return cells


def _render_input(specs: list[dict], shape: tuple[int, int], bud_color: int) -> np.ndarray:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    for spec in specs:
        r0, c0, r1, c1 = spec["bbox"]
        for r, c in _frame_border_cells(r0, c0, r1 - r0 + 1, c1 - c0 + 1):
            grid[r, c] = int(spec["frame_color"])
        for rr, cc in np.argwhere(spec["seed_mask"]):
            grid[r0 + 1 + int(rr), c0 + 1 + int(cc)] = int(spec["seed_color"])
        for side in spec["buds"]:
            br, bc = _bud_position(spec, side)
            grid[br, bc] = bud_color
    return grid


def _render_expected_output(specs: list[dict], shape: tuple[int, int], bud_color: int) -> np.ndarray:
    out = _render_input(specs, shape, bud_color)
    for spec in specs:
        _apply_growth(out, spec)
    return out


def _sample_spec(
    rng: np.random.Generator,
    shape: tuple[int, int],
    frame_color: int,
    seed_color: int,
    blocked: set[tuple[int, int]],
) -> dict | None:
    h, w = shape
    for _ in range(PLACEMENT_RETRIES):
        head_dir = DIR_ORDER[int(rng.integers(0, 4))]
        buds = _sample_buds(rng)
        template_idx = int(rng.integers(0, len(BASE_SEED_MASKS)))
        r0 = int(rng.integers(0, h - FRAME_SIZE + 1))
        c0 = int(rng.integers(0, w - FRAME_SIZE + 1))

        spec = {
            "bbox": (r0, c0, r0 + FRAME_SIZE - 1, c0 + FRAME_SIZE - 1),
            "frame_color": frame_color,
            "seed_color": seed_color,
            "head_dir": head_dir,
            "seed_mask": _seed_mask_for_direction(template_idx, head_dir),
            "tight_seed_mask": _tight_mask(_seed_mask_for_direction(template_idx, head_dir)),
            "buds": buds,
        }

        input_cells = _object_input_cells(spec)
        if not all(0 <= r < h and 0 <= c < w for r, c in input_cells):
            continue

        output_cells = _object_output_cells(spec, shape)
        if output_cells & blocked:
            continue
        return spec
    return None


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    object_count = int(rng.integers(1, 4))
    min_dim = {1: 13, 2: 18, 3: 22}[object_count]
    h = int(rng.integers(min_dim, 29))
    w = int(rng.integers(min_dim, 29))

    palette = list(range(1, 10))
    perm = list(rng.permutation(palette))
    bud_color = int(perm[0])
    specs: list[dict] = []
    blocked: set[tuple[int, int]] = set()

    for idx in range(object_count):
        frame_color = int(perm[1 + 2 * idx])
        seed_color = int(perm[2 + 2 * idx])
        spec = _sample_spec(rng, (h, w), frame_color, seed_color, blocked)
        if spec is None:
            return None
        specs.append(spec)
        blocked |= _expand_cells(_object_output_cells(spec, (h, w)), h, w, pad=1)

    grid = _render_input(specs, (h, w), bud_color)
    expected = _render_expected_output(specs, (h, w), bud_color)
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input grid")
