import numpy as np


BG = 0
SPRITE_SIZE = 3
SHAPE_LIBRARY = [
    np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool),
    np.array([[1, 1, 1], [0, 1, 0], [0, 1, 0]], dtype=bool),
    np.array([[1, 0, 0], [1, 0, 0], [1, 1, 1]], dtype=bool),
    np.array([[0, 0, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]], dtype=bool),
    np.array([[0, 1, 1], [1, 1, 0], [1, 0, 0]], dtype=bool),
    np.array([[1, 1, 0], [0, 1, 0], [0, 1, 1]], dtype=bool),
    np.array([[0, 1, 1], [0, 1, 0], [1, 1, 0]], dtype=bool),
]


def _bg_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) == bg:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) != bg:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            pts = np.asarray(coords, dtype=int)
            colors = {int(grid[pr, pc]) for pr, pc in pts}
            assert len(colors) == 1, "each sprite must be monochromatic"
            r0, c0, r1, c1 = _bbox(pts)
            assert (r1 - r0 + 1, c1 - c0 + 1) == (SPRITE_SIZE, SPRITE_SIZE), "every sprite must occupy a 3x3 bbox"
            mask = np.zeros((SPRITE_SIZE, SPRITE_SIZE), dtype=bool)
            mask[pts[:, 0] - r0, pts[:, 1] - c0] = True
            components.append(
                {
                    "color": int(next(iter(colors))),
                    "bbox": (r0, c0, r1, c1),
                    "row": int(r0),
                    "col": int(c0),
                    "shape": mask,
                }
            )
    return components


def _shape_key(mask: np.ndarray) -> tuple[int, ...]:
    return tuple(int(x) for x in mask.astype(int).ravel())


def _majority(items: list):
    counts = {}
    for item in items:
        counts[item] = counts.get(item, 0) + 1
    ordered = sorted(counts.items(), key=lambda kv: kv[1], reverse=True)
    assert len(ordered) >= 1
    if len(ordered) > 1:
        assert ordered[0][1] > ordered[1][1], f"majority must be unique: {counts}"
    return ordered[0][0]


def _paint_sprite(grid: np.ndarray, r: int, c: int, shape: np.ndarray, color: int) -> None:
    patch = grid[r : r + SPRITE_SIZE, c : c + SPRITE_SIZE]
    patch[:] = BG
    patch[shape] = int(color)


def _analyze_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    bg = _bg_color(g)
    components = _find_components(g, bg)
    assert components, "expected at least one sprite"

    rows = sorted({comp["row"] for comp in components})
    cols = sorted({comp["col"] for comp in components})
    assert len(rows) >= 3 and len(cols) >= 3, "need a nontrivial row/column lattice"

    row_entries = {r: [comp for comp in components if comp["row"] == r] for r in rows}
    col_entries = {c: [comp for comp in components if comp["col"] == c] for c in cols}
    assert all(row_entries[r] for r in rows)
    assert all(col_entries[c] for c in cols)

    row_colors = {r: int(_majority([comp["color"] for comp in row_entries[r]])) for r in rows}
    col_shapes = {}
    for c in cols:
        shape_keys = [_shape_key(comp["shape"]) for comp in col_entries[c]]
        col_shapes[c] = np.array(_majority(shape_keys), dtype=int).reshape(SPRITE_SIZE, SPRITE_SIZE).astype(bool)

    occupied = {(comp["row"], comp["col"]) for comp in components}
    missing = [(r, c) for r in rows for c in cols if (r, c) not in occupied]
    assert missing, "expected at least one empty lattice slot"

    suspicious = []
    for comp in components:
        expected_color = row_colors[comp["row"]]
        expected_shape = col_shapes[comp["col"]]
        color_ok = int(comp["color"]) == int(expected_color)
        shape_ok = np.array_equal(comp["shape"], expected_shape)
        if not (color_ok and shape_ok):
            suspicious.append(
                {
                    "row": comp["row"],
                    "col": comp["col"],
                    "color_ok": color_ok,
                    "shape_ok": shape_ok,
                    "actual_color": int(comp["color"]),
                    "expected_color": int(expected_color),
                    "actual_shape": comp["shape"],
                    "expected_shape": expected_shape,
                }
            )

    assert len(suspicious) == 1, f"expected exactly one impostor, got {len(suspicious)}"
    imp = suspicious[0]
    assert imp["color_ok"] ^ imp["shape_ok"], "the impostor must fake exactly one attribute"

    return {
        "bg": bg,
        "components": components,
        "rows": rows,
        "cols": cols,
        "row_colors": row_colors,
        "col_shapes": col_shapes,
        "missing": missing,
        "impostor": imp,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """Complete a row-color/column-shape sprite table and correct the single impostor.

    Input structure:
    - A black background contains monochromatic sprites placed on an un-drawn row/column lattice.
    - Every sprite occupies a 3x3 mask; sprites in the same row share one color and sprites in the same column share one shape.
    - Some lattice slots are empty, and exactly one existing sprite borrows either another row's color or another column's shape.

    Rule:
    - Read the intended row color from the unique majority color among the existing sprites of each row.
    - Read the intended column shape from the unique majority 3x3 mask among the existing sprites of each column.
    - Identify the only sprite that disagrees with its row code or column code, correct that wrong attribute, and fill every empty lattice slot with the sprite formed by that row's color and that column's shape.

    Key insight:
    - A sprite's identity factorizes into orthogonal row and column attributes, so crossing row-majority color evidence with column-majority shape evidence uniquely exposes the impostor and reconstructs the full Cartesian-product table.
    """
    info = _analyze_grid(grid)
    out = np.full_like(np.asarray(grid, dtype=int), info["bg"])
    for r in info["rows"]:
        for c in info["cols"]:
            _paint_sprite(out, r, c, info["col_shapes"][c], info["row_colors"][r])
    return out


def _sample_axis(rng: np.random.Generator, count: int) -> tuple[list[int], int] | None:
    gaps = rng.integers(5, 8, size=count - 1)
    margin0 = int(rng.integers(1, 4))
    margin1 = int(rng.integers(1, 4))
    positions = [margin0]
    for gap in gaps:
        positions.append(positions[-1] + int(gap))
    size = positions[-1] + SPRITE_SIZE + margin1
    if size > 30:
        return None
    return positions, size


def _occupancy_valid(
    occupied: np.ndarray,
    imp_r: int,
    imp_c: int,
    wrong_attr: str,
) -> bool:
    if not occupied[imp_r, imp_c]:
        return False
    row_counts = occupied.sum(axis=1)
    col_counts = occupied.sum(axis=0)
    for r, count in enumerate(row_counts):
        need = 3 if wrong_attr == "color" and r == imp_r else 2
        if int(count) < need:
            return False
    for c, count in enumerate(col_counts):
        need = 3 if wrong_attr == "shape" and c == imp_c else 2
        if int(count) < need:
            return False
    return int(np.sum(~occupied)) >= 1


def _sample_occupancy(
    rng: np.random.Generator,
    n_rows: int,
    n_cols: int,
    imp_r: int,
    imp_c: int,
    wrong_attr: str,
) -> np.ndarray | None:
    occupied = np.ones((n_rows, n_cols), dtype=bool)
    cells = [(r, c) for r in range(n_rows) for c in range(n_cols) if (r, c) != (imp_r, imp_c)]
    rng.shuffle(cells)
    max_remove = max(1, min(len(cells), (n_rows * n_cols) // 2))
    target_remove = int(rng.integers(1, max_remove + 1))
    removed = 0
    for r, c in cells:
        if removed >= target_remove:
            break
        if rng.random() < 0.65:
            occupied[r, c] = False
            if _occupancy_valid(occupied, imp_r, imp_c, wrong_attr):
                removed += 1
            else:
                occupied[r, c] = True
    if removed == 0:
        for r, c in cells:
            occupied[r, c] = False
            if _occupancy_valid(occupied, imp_r, imp_c, wrong_attr):
                removed = 1
                break
            occupied[r, c] = True
    return occupied if _occupancy_valid(occupied, imp_r, imp_c, wrong_attr) else None


def _build_full_grid(
    height: int,
    width: int,
    row_positions: list[int],
    col_positions: list[int],
    row_colors: list[int],
    col_shapes: list[np.ndarray],
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r_idx, r0 in enumerate(row_positions):
        for c_idx, c0 in enumerate(col_positions):
            _paint_sprite(grid, r0, c0, col_shapes[c_idx], row_colors[r_idx])
    return grid


def _build_input_grid(
    height: int,
    width: int,
    row_positions: list[int],
    col_positions: list[int],
    row_colors: list[int],
    col_shapes: list[np.ndarray],
    occupied: np.ndarray,
    imp_r: int,
    imp_c: int,
    wrong_attr: str,
    fake_row: int,
    fake_col: int,
) -> np.ndarray:
    grid = np.full((height, width), BG, dtype=int)
    for r_idx, r0 in enumerate(row_positions):
        for c_idx, c0 in enumerate(col_positions):
            if not occupied[r_idx, c_idx]:
                continue
            color = row_colors[r_idx]
            shape = col_shapes[c_idx]
            if (r_idx, c_idx) == (imp_r, imp_c):
                if wrong_attr == "color":
                    color = row_colors[fake_row]
                else:
                    shape = col_shapes[fake_col]
            _paint_sprite(grid, r0, c0, shape, color)
    return grid


def _semantic_check(
    inp: np.ndarray,
    out: np.ndarray,
    row_colors: list[int],
    col_shapes: list[np.ndarray],
    imp_r: int,
    imp_c: int,
    wrong_attr: str,
    fake_row: int,
    fake_col: int,
) -> bool:
    info = _analyze_grid(inp)
    if not np.array_equal(main(inp), out):
        return False
    if len(set(row_colors)) != len(row_colors):
        return False
    shape_keys = [_shape_key(shape) for shape in col_shapes]
    if len(set(shape_keys)) != len(shape_keys):
        return False
    if len(info["rows"]) != len(row_colors) or len(info["cols"]) != len(col_shapes):
        return False
    imp = info["impostor"]
    if info["rows"][imp_r] != imp["row"] or info["cols"][imp_c] != imp["col"]:
        return False
    if wrong_attr == "color":
        if imp["color_ok"] or not imp["shape_ok"]:
            return False
        borrowed = int(row_colors[fake_row])
        if imp["actual_color"] != borrowed or borrowed == imp["expected_color"]:
            return False
    else:
        if not imp["color_ok"] or imp["shape_ok"]:
            return False
        borrowed_key = _shape_key(col_shapes[fake_col])
        if _shape_key(imp["actual_shape"]) != borrowed_key or borrowed_key == _shape_key(imp["expected_shape"]):
            return False
    return True


def _gen_attempt() -> np.ndarray | None:
    rng = np.random.default_rng()
    n_rows = int(rng.integers(3, 5))
    n_cols = int(rng.integers(3, 6))
    row_axis = _sample_axis(rng, n_rows)
    col_axis = _sample_axis(rng, n_cols)
    if row_axis is None or col_axis is None:
        return None
    row_positions, height = row_axis
    col_positions, width = col_axis
    if height > 30 or width > 30:
        return None

    row_colors = rng.choice(np.arange(1, 10), size=n_rows, replace=False).astype(int).tolist()
    shape_indices = rng.choice(len(SHAPE_LIBRARY), size=n_cols, replace=False)
    col_shapes = [SHAPE_LIBRARY[int(idx)].copy() for idx in shape_indices]

    imp_r = int(rng.integers(0, n_rows))
    imp_c = int(rng.integers(0, n_cols))
    wrong_attr = "color" if rng.random() < 0.5 else "shape"
    fake_row = int(rng.choice([i for i in range(n_rows) if i != imp_r]))
    fake_col = int(rng.choice([i for i in range(n_cols) if i != imp_c]))

    occupied = _sample_occupancy(rng, n_rows, n_cols, imp_r, imp_c, wrong_attr)
    if occupied is None:
        return None

    out = _build_full_grid(height, width, row_positions, col_positions, row_colors, col_shapes)
    inp = _build_input_grid(
        height,
        width,
        row_positions,
        col_positions,
        row_colors,
        col_shapes,
        occupied,
        imp_r,
        imp_c,
        wrong_attr,
        fake_row,
        fake_col,
    )
    if not _semantic_check(inp, out, row_colors, col_shapes, imp_r, imp_c, wrong_attr, fake_row, fake_col):
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid row-color/column-shape impostor table")
