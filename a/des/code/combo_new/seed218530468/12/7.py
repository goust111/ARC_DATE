import numpy as np
from collections import deque


BG = 0
PALETTE = [1, 2, 3, 4, 5, 6, 7, 8, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
GEN_RETRIES = 400


def _components_monochrome(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == BG or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((int(cr), int(cc)))
                for dr, dc in DIRS4:
                    nr, nc = int(cr + dr), int(cc + dc)
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))
            comps.append({"color": color, "coords": np.array(coords, dtype=int)})
    return comps


def _mask_points(mask: np.ndarray, top: int, left: int) -> list[tuple[int, int]]:
    pts = np.argwhere(mask)
    return [(int(top + r), int(left + c)) for r, c in pts]


def _random_shape(min_area: int = 2, max_area: int = 6) -> np.ndarray:
    for _ in range(200):
        h = int(RNG.integers(1, 5))
        w = int(RNG.integers(1, 5))
        hi = min(max_area, h * w)
        lo = min(min_area, hi)
        area = int(RNG.integers(lo, hi + 1))

        start = (int(RNG.integers(0, h)), int(RNG.integers(0, w)))
        cells = {start}
        frontier = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    frontier.add((nr, nc))

        add_frontier(*start)
        while len(cells) < area and frontier:
            idx = int(RNG.integers(0, len(frontier)))
            pick = list(frontier)[idx]
            frontier.remove(pick)
            cells.add(pick)
            add_frontier(*pick)

        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True

        rows = np.where(mask.any(axis=1))[0]
        cols = np.where(mask.any(axis=0))[0]
        mask = mask[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]

        # Avoid always generating solid rectangles.
        if mask.sum() == mask.shape[0] * mask.shape[1] and mask.size > 4:
            if int(RNG.integers(0, 4)) != 0:
                continue
        return mask

    return np.array([[True], [True]], dtype=bool)


def _separated(candidate: list[tuple[int, int]], occupied: set[tuple[int, int]], pad: int = 1) -> bool:
    for r, c in candidate:
        for dr in range(-pad, pad + 1):
            for dc in range(-pad, pad + 1):
                if (int(r + dr), int(c + dc)) in occupied:
                    return False
    return True


def _draw_reference_column(grid: np.ndarray, colors: list[int], rows: list[int]) -> None:
    for color, row in zip(colors, rows):
        grid[int(row), 0] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each colored object downward by the rank of its color in the left reference column.

    Input structure:
    - The background is black.
    - Column 0 contains a vertical reference list: distinct non-black single pixels appear from top to bottom, possibly separated by black gap rows.
    - The remaining columns contain several 4-connected monochrome objects; every object color also appears in the reference column.

    Rule:
    - Scan the reference column from top to bottom and rank its non-black colors by encounter order: first non-black color -> shift 1, second -> shift 2, and so on.
    - Remove the reference column and remove all original object pixels.
    - For each object, translate every pixel straight downward by the shift assigned to that object's color.
    - Pixels translated below the bottom edge are clipped and not drawn.

    Key insight:
    - The reference column is not an object to copy; its vertical color order encodes the per-color downward offset shared by all objects of that color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape

    shifts: dict[int, int] = {}
    order = 0
    for r in range(h):
        color = int(g[r, 0])
        if color == BG or color in shifts:
            continue
        order += 1
        shifts[color] = int(order)

    work = g.copy()
    work[:, 0] = BG
    out = np.full((h, w), BG, dtype=int)

    for comp in _components_monochrome(work):
        color = int(comp["color"])
        assert color in shifts
        dr = int(shifts[color])
        for r, c in comp["coords"]:
            nr = int(r + dr)
            if 0 <= nr < h:
                out[nr, int(c)] = color
    return out


def _specs_for_sample(h: int) -> tuple[list[int], dict[int, int], list[dict], bool, bool]:
    num_colors = int(RNG.integers(2, min(6, h - 2)))
    colors = list(RNG.choice(PALETTE, size=num_colors, replace=False))
    ref_rows = sorted(int(x) for x in RNG.choice(np.arange(h), size=num_colors, replace=False))
    shifts = {int(color): idx + 1 for idx, color in enumerate(colors)}
    clip_mode = bool(int(RNG.integers(0, 2)))
    duplicate_mode = bool(int(RNG.integers(0, 2)))

    counts = {int(color): 1 for color in colors}
    if duplicate_mode:
        extra = int(RNG.integers(1, min(4, num_colors + 1)))
        for _ in range(extra):
            color = int(colors[int(RNG.integers(0, len(colors)))])
            counts[color] += 1

    specs: list[dict] = []
    for color in colors:
        for _ in range(counts[int(color)]):
            mask = _random_shape()
            specs.append(
                {
                    "color": int(color),
                    "shift": int(shifts[int(color)]),
                    "mask": mask,
                    "clip": False,
                }
            )

    if clip_mode:
        candidates = [spec for spec in specs if spec["mask"].shape[0] >= 2]
        if not candidates:
            spec = specs[int(RNG.integers(0, len(specs)))]
            spec["mask"] = _random_shape(min_area=2, max_area=6)
            while spec["mask"].shape[0] < 2:
                spec["mask"] = _random_shape(min_area=2, max_area=6)
            candidates = [spec]
        num_clipped = int(RNG.integers(1, min(3, len(candidates)) + 1))
        chosen_idx = RNG.choice(np.arange(len(candidates)), size=num_clipped, replace=False)
        for idx in np.atleast_1d(chosen_idx):
            candidates[int(idx)]["clip"] = True

    return ref_rows, shifts, specs, clip_mode, duplicate_mode


def _place_specs(h: int, w: int, specs: list[dict]) -> list[dict] | None:
    placed: list[dict] = []
    input_occ: set[tuple[int, int]] = set()
    output_occ: set[tuple[int, int]] = set()

    order = sorted(
        specs,
        key=lambda spec: (
            0 if spec["clip"] else 1,
            -(int(spec["mask"].shape[0]) * int(spec["mask"].shape[1])),
            -int(spec["shift"]),
        ),
    )

    for spec in order:
        mask = spec["mask"]
        sh, sw = map(int, mask.shape)
        shift = int(spec["shift"])

        col_max = w - sw
        if col_max < 2:
            return None

        if spec["clip"]:
            row_lo = max(0, h - shift - sh)
            row_hi = min(h - sh, h - shift - 1)
        else:
            row_lo = 0
            row_hi = h - sh - shift
        if row_lo > row_hi:
            return None

        candidates = [
            (r, c)
            for r in range(int(row_lo), int(row_hi) + 1)
            for c in range(2, int(col_max) + 1)
        ]
        RNG.shuffle(candidates)

        placed_here = None
        for top, left in candidates:
            input_pts = _mask_points(mask, int(top), int(left))
            output_pts = [(int(r + shift), int(c)) for r, c in input_pts if int(r + shift) < h]
            if not output_pts:
                continue
            if not _separated(input_pts, input_occ, pad=1):
                continue
            if not _separated(output_pts, output_occ, pad=1):
                continue
            placed_here = {
                "color": int(spec["color"]),
                "shift": shift,
                "mask": mask,
                "top": int(top),
                "left": int(left),
                "clip": bool(spec["clip"]),
            }
            placed.append(placed_here)
            input_occ.update(input_pts)
            output_occ.update(output_pts)
            break

        if placed_here is None:
            return None
    return placed


def _build_grid(h: int, w: int, ref_rows: list[int], shifts: dict[int, int], placed: list[dict]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    colors_in_order = sorted(shifts, key=lambda color: shifts[color])
    _draw_reference_column(grid, colors_in_order, ref_rows)
    for item in placed:
        for r, c in _mask_points(item["mask"], item["top"], item["left"]):
            grid[int(r), int(c)] = int(item["color"])
    return grid


def _validate_grid(grid: np.ndarray, clip_mode: bool, duplicate_mode: bool) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    if np.count_nonzero(grid[:, 1]) != 0:
        return False

    ref_colors = [int(v) for v in grid[:, 0] if int(v) != BG]
    if len(ref_colors) < 2 or len(ref_colors) != len(set(ref_colors)):
        return False

    work = grid.copy()
    work[:, 0] = BG
    comps = _components_monochrome(work)
    if len(comps) < 2:
        return False
    comp_colors = [int(comp["color"]) for comp in comps]
    if not set(comp_colors).issubset(set(ref_colors)):
        return False
    if duplicate_mode and len(comp_colors) == len(set(comp_colors)):
        return False

    out = main(grid)
    if out.shape != grid.shape or np.array_equal(out, grid):
        return False
    if np.any(out[:, 0] != BG):
        return False
    if clip_mode and np.count_nonzero(out) >= np.count_nonzero(work):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    h = int(RNG.integers(12, 25))
    w = int(RNG.integers(12, 26))
    ref_rows, shifts, specs, clip_mode, duplicate_mode = _specs_for_sample(h)
    placed = _place_specs(h, w, specs)
    if placed is None:
        return None
    grid = _build_grid(h, w, ref_rows, shifts, placed)
    if not _validate_grid(grid, clip_mode, duplicate_mode):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid sample for task 7")
