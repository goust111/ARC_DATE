import random

import numpy as np


BG = 0
FRAME_COLOR = 5
LEGEND_FRAME_SIZE = 5
LEGEND_INNER_LEFT = (1, 0)
LEGEND_INNER_RIGHT = (1, 2)
GEN_RETRIES = 400
SHAPE_RETRIES = 200
PLACEMENT_RETRIES = 300
COLOR_POOL = tuple(c for c in range(10) if c not in {BG, FRAME_COLOR})


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors(r: int, c: int, h: int, w: int, connectivity: int) -> list[tuple[int, int]]:
    if connectivity == 4:
        deltas = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    else:
        deltas = [
            (-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1),
        ]
    out = []
    for dr, dc in deltas:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray, connectivity: int = 8) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                pts.append((cr, cc))
                for nr, nc in _neighbors(cr, cc, h, w, connectivity):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.asarray(pts, dtype=int))
    return comps


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    return (
        int(coords[:, 0].min()),
        int(coords[:, 1].min()),
        int(coords[:, 0].max()),
        int(coords[:, 1].max()),
    )


def _trim_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0, r1, c1 = _bbox_from_coords(coords)
    return mask[r0 : r1 + 1, c0 : c1 + 1]


def _find_legend_cells(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    border_mask = np.ones((LEGEND_FRAME_SIZE, LEGEND_FRAME_SIZE), dtype=bool)
    border_mask[1:-1, 1:-1] = False
    candidates: list[dict] = []

    for r in range(h - LEGEND_FRAME_SIZE + 1):
        for c in range(w - LEGEND_FRAME_SIZE + 1):
            win = grid[r : r + LEGEND_FRAME_SIZE, c : c + LEGEND_FRAME_SIZE]
            frame_color = int(win[0, 0])
            if frame_color == bg:
                continue
            if not np.all(win[border_mask] == frame_color):
                continue

            inner = win[1:-1, 1:-1]
            nz = np.argwhere(inner != bg)
            if len(nz) != 1:
                continue
            ir, ic = map(int, nz[0])
            if (ir, ic) not in {LEGEND_INNER_LEFT, LEGEND_INNER_RIGHT}:
                continue
            pixel_color = int(inner[ir, ic])
            if pixel_color in {bg, frame_color}:
                continue
            inner_copy = inner.copy()
            inner_copy[ir, ic] = bg
            if not np.all(inner_copy == bg):
                continue

            candidates.append(
                {
                    "bbox": (r, c, r + LEGEND_FRAME_SIZE - 1, c + LEGEND_FRAME_SIZE - 1),
                    "row": r,
                    "col": c,
                    "frame_color": frame_color,
                    "new_color": pixel_color,
                    "axis": "lr" if (ir, ic) == LEGEND_INNER_LEFT else "ud",
                }
            )

    grouped: dict[tuple[int, int], list[dict]] = {}
    for cand in candidates:
        key = (cand["row"], cand["frame_color"])
        grouped.setdefault(key, []).append(cand)

    valid_rows: list[list[dict]] = []
    for (_row, _frame_color), row_cands in grouped.items():
        row_cands = sorted(row_cands, key=lambda x: x["col"])
        if not (2 <= len(row_cands) <= 3):
            continue
        if len({cand["new_color"] for cand in row_cands}) != len(row_cands):
            continue
        if all(
            row_cands[i + 1]["col"] - row_cands[i]["col"] >= LEGEND_FRAME_SIZE + 1
            for i in range(len(row_cands) - 1)
        ):
            valid_rows.append(row_cands)

    assert valid_rows, "could not find legend row"
    valid_rows.sort(key=lambda row: row[0]["row"])
    legend = valid_rows[0]
    top_row = legend[0]["row"]
    assert all(cell["row"] == top_row for cell in legend)
    return legend


def main(grid: np.ndarray) -> np.ndarray:
    """Recolor sprite groups by legend index and mirror inside each sprite bbox."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    legend = _find_legend_cells(g, bg)
    legend_map = {i + 1: (cell["new_color"], cell["axis"]) for i, cell in enumerate(sorted(legend, key=lambda x: x["col"]))}

    legend_mask = np.zeros_like(g, dtype=bool)
    for cell in legend:
        r0, c0, r1, c1 = cell["bbox"]
        legend_mask[r0 : r1 + 1, c0 : c1 + 1] = True

    sprite_mask = (g != bg) & (~legend_mask)
    comps = _components(sprite_mask, connectivity=8)
    assert comps, "expected sprite components below legend"

    groups: dict[int, list[dict]] = {}
    for coords in comps:
        r0, c0, r1, c1 = _bbox_from_coords(coords)
        colors = {int(g[r, c]) for r, c in coords}
        assert len(colors) == 1, "sprite component must be monochrome"
        color = next(iter(colors))
        patch = g[r0 : r1 + 1, c0 : c1 + 1] == color
        groups.setdefault(color, []).append({"bbox": (r0, c0, r1, c1), "mask": patch})

    out = g.copy()
    used_counts: set[int] = set()
    for color, copies in groups.items():
        _ = color
        count = len(copies)
        assert count in legend_map, "sprite multiplicity must index a legend cell"
        assert count not in used_counts, "sprite group counts must be unique"
        used_counts.add(count)

        shape = copies[0]["mask"]
        for copy in copies[1:]:
            assert copy["mask"].shape == shape.shape and np.array_equal(copy["mask"], shape), "copies in a group must match"

        new_color, axis = legend_map[count]
        for copy in copies:
            r0, c0, r1, c1 = copy["bbox"]
            patch = copy["mask"]
            transformed = np.fliplr(patch) if axis == "lr" else np.flipud(patch)
            sub = out[r0 : r1 + 1, c0 : c1 + 1]
            sub[patch] = bg
            sub[transformed] = new_color

    return out


def _rect_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + pad < br0 or br1 + pad < ar0 or ac1 + pad < bc0 or bc1 + pad < ac0


def _random_shape(rng: random.Random, seen: set[tuple]) -> np.ndarray | None:
    for _ in range(SHAPE_RETRIES):
        h = rng.randint(2, 4)
        w = rng.randint(2, 5)
        area = h * w
        cells_target = rng.randint(3, min(area - 1, 7))

        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        while len(cells) < cells_target:
            base_r, base_c = rng.choice(tuple(cells))
            candidates = []
            for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                rr, cc = base_r + dr, base_c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    candidates.append((rr, cc))
            if not candidates:
                continue
            cells.add(rng.choice(candidates))

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        mask = _trim_mask(mask)
        if mask.shape[0] < 2 or mask.shape[1] < 2:
            continue
        if np.array_equal(mask, np.fliplr(mask)) or np.array_equal(mask, np.flipud(mask)):
            continue
        if len(_components(mask, connectivity=4)) != 1:
            continue

        sig = (mask.shape, tuple(mask.astype(int).ravel().tolist()))
        if sig in seen:
            continue
        seen.add(sig)
        return mask
    return None


def _legend_axes(rng: random.Random, count: int) -> list[str]:
    if count == 2:
        axes = ["lr", "ud"]
    else:
        axes = ["lr", "lr", "ud"] if rng.random() < 0.5 else ["lr", "ud", "ud"]
    rng.shuffle(axes)
    return axes


def _place_shape(
    grid: np.ndarray,
    mask: np.ndarray,
    color: int,
    main_top: int,
    blocked_rects: list[tuple[int, int, int, int]],
    rng: random.Random,
) -> tuple[int, int, int, int] | None:
    h, w = grid.shape
    ph, pw = mask.shape
    for _ in range(PLACEMENT_RETRIES):
        r0 = rng.randint(main_top, h - ph)
        c0 = rng.randint(0, w - pw)
        rect = (r0, c0, r0 + ph - 1, c0 + pw - 1)
        if not all(_rect_separated(rect, other, pad=1) for other in blocked_rects):
            continue
        sub = grid[r0 : r0 + ph, c0 : c0 + pw]
        if np.any(sub[mask] != BG):
            continue
        sub[mask] = color
        blocked_rects.append(rect)
        return rect
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    legend_count = rng.choice([2, 3])
    height = rng.randint(20, 25)
    width = rng.randint(19, 26)
    grid = np.full((height, width), BG, dtype=int)

    gaps = [rng.randint(1, 3 if legend_count == 2 else 2) for _ in range(legend_count - 1)]
    legend_span = legend_count * LEGEND_FRAME_SIZE + sum(gaps)
    if legend_span > width:
        return None
    start_col = rng.randint(0, width - legend_span)

    axes = _legend_axes(rng, legend_count)
    palette = list(COLOR_POOL)
    rng.shuffle(palette)
    legend_colors = palette[:legend_count]
    sprite_colors = palette[legend_count : 2 * legend_count]

    legend_rects: list[tuple[int, int, int, int]] = []
    col = start_col
    for i in range(legend_count):
        r0 = 0
        c0 = col
        grid[r0 : r0 + LEGEND_FRAME_SIZE, c0 : c0 + LEGEND_FRAME_SIZE] = BG
        grid[r0, c0 : c0 + LEGEND_FRAME_SIZE] = FRAME_COLOR
        grid[r0 + LEGEND_FRAME_SIZE - 1, c0 : c0 + LEGEND_FRAME_SIZE] = FRAME_COLOR
        grid[r0 : r0 + LEGEND_FRAME_SIZE, c0] = FRAME_COLOR
        grid[r0 : r0 + LEGEND_FRAME_SIZE, c0 + LEGEND_FRAME_SIZE - 1] = FRAME_COLOR
        pixel_col = c0 + 1 if axes[i] == "lr" else c0 + 3
        grid[r0 + 2, pixel_col] = legend_colors[i]
        legend_rects.append((r0, c0, r0 + LEGEND_FRAME_SIZE - 1, c0 + LEGEND_FRAME_SIZE - 1))
        if i < legend_count - 1:
            col += LEGEND_FRAME_SIZE + gaps[i]

    main_top = 6 + rng.randint(0, 1)
    blocked_rects = legend_rects.copy()
    seen_shapes: set[tuple] = set()

    group_specs = []
    counts = list(range(legend_count, 0, -1))
    for count in counts:
        shape = _random_shape(rng, seen_shapes)
        if shape is None:
            return None
        group_specs.append({"count": count, "shape": shape, "color": sprite_colors[count - 1]})

    for spec in group_specs:
        for _ in range(spec["count"]):
            rect = _place_shape(grid, spec["shape"], spec["color"], main_top, blocked_rects, rng)
            if rect is None:
                return None

    try:
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
