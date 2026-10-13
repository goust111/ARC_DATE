import random
from collections import Counter, defaultdict, deque

import numpy as np


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.reshape(-1))).most_common(1)[0][0])


def find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def lattice_info(grid: np.ndarray, comp: list[tuple[int, int]], frame_color: int) -> dict | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]

    h_lines = [i for i in range(patch.shape[0]) if bool((patch[i, :] == frame_color).all())]
    v_lines = [j for j in range(patch.shape[1]) if bool((patch[:, j] == frame_color).all())]
    if len(h_lines) < 2 or len(v_lines) < 2:
        return None
    if h_lines[0] != 0 or h_lines[-1] != patch.shape[0] - 1:
        return None
    if v_lines[0] != 0 or v_lines[-1] != patch.shape[1] - 1:
        return None

    row_gaps = [b - a for a, b in zip(h_lines, h_lines[1:])]
    col_gaps = [b - a for a, b in zip(v_lines, v_lines[1:])]
    if min(row_gaps) < 2 or min(col_gaps) < 2:
        return None
    if len(set(row_gaps)) != 1 or len(set(col_gaps)) != 1:
        return None

    expected = np.zeros_like(patch, dtype=bool)
    expected[h_lines, :] = True
    expected[:, v_lines] = True
    if not np.array_equal(patch == frame_color, expected):
        return None

    return {
        "bbox": (r0, c0, r1, c1),
        "h_lines": [r0 + x for x in h_lines],
        "v_lines": [c0 + x for x in v_lines],
        "rows": len(h_lines) - 1,
        "cols": len(v_lines) - 1,
    }


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    counts = Counter(map(int, g.reshape(-1)))
    non_bg = [color for color, _ in counts.most_common() if color != bg]
    assert non_bg, "expected at least one foreground color"
    frame_color = int(non_bg[0])

    lattices = []
    for comp in find_components(g == frame_color):
        info = lattice_info(g, comp, frame_color)
        if info is not None:
            lattices.append(info)
    assert lattices, "no lattice frames found"

    shape_counts = Counter((lat["rows"], lat["cols"]) for lat in lattices)
    target_shape = shape_counts.most_common(1)[0][0]
    lattices = [lat for lat in lattices if (lat["rows"], lat["cols"]) == target_shape]
    assert lattices, "no consistent lattice shape found"

    slot_colors: dict[tuple[int, int], int] = {}
    for lat in lattices:
        for rr in range(lat["rows"]):
            for cc in range(lat["cols"]):
                rs = slice(lat["h_lines"][rr] + 1, lat["h_lines"][rr + 1])
                cs = slice(lat["v_lines"][cc] + 1, lat["v_lines"][cc + 1])
                cell = g[rs, cs]
                vals = {int(v) for v in cell.reshape(-1) if int(v) not in (bg, frame_color)}
                if not vals:
                    continue
                assert len(vals) == 1, "slot contains conflicting colors"
                color = next(iter(vals))
                prev = slot_colors.get((rr, cc))
                if prev is not None:
                    assert prev == color, "same logical slot has multiple colors"
                slot_colors[(rr, cc)] = color

    assert len(slot_colors) == target_shape[0] * target_shape[1], "incomplete slot evidence"

    out = g.copy()
    for lat in lattices:
        for rr in range(lat["rows"]):
            for cc in range(lat["cols"]):
                color = slot_colors[(rr, cc)]
                rs = slice(lat["h_lines"][rr] + 1, lat["h_lines"][rr + 1])
                cs = slice(lat["v_lines"][cc] + 1, lat["v_lines"][cc + 1])
                out[rs, cs] = color
    return out


MAX_ATTEMPTS = 300


def _place_rectangles(
    canvas_h: int, canvas_w: int, rect_h: int, rect_w: int, count: int
) -> list[tuple[int, int]] | None:
    positions: list[tuple[int, int]] = []
    for _ in range(500):
        positions.clear()
        ok = True
        for _k in range(count):
            placed = False
            choices = [
                (r, c)
                for r in range(0, canvas_h - rect_h + 1)
                for c in range(0, canvas_w - rect_w + 1)
                if all(
                    not (
                        r < pr + rect_h + 2
                        and pr < r + rect_h + 2
                        and c < pc + rect_w + 2
                        and pc < c + rect_w + 2
                    )
                    for pr, pc in positions
                )
            ]
            if not choices:
                ok = False
                break
            positions.append(random.choice(choices))
            placed = True
            if not placed:
                ok = False
                break
        if ok:
            return positions[:]
    return None


def _gen_attempt() -> np.ndarray | None:
    bg = 0
    frame_color = random.choice([1, 2, 3, 4, 5, 6, 7, 8, 9])
    rows = random.randint(2, 4)
    cols = random.randint(2, 4)
    cell_h = random.randint(2, 4)
    cell_w = random.randint(2, 4)
    lattice_h = rows * cell_h + (rows + 1)
    lattice_w = cols * cell_w + (cols + 1)
    copies = random.randint(2, 5)

    grid_h = max(lattice_h + 2, random.randint(lattice_h + 2, lattice_h * copies + 10))
    grid_w = max(lattice_w + 2, random.randint(lattice_w + 2, lattice_w * copies + 10))
    positions = _place_rectangles(grid_h, grid_w, lattice_h, lattice_w, copies)
    if positions is None:
        return None

    palette = [c for c in range(1, 10) if c != frame_color]
    random.shuffle(palette)
    need = rows * cols
    if len(palette) < need:
        colors = [random.choice(palette) for _ in range(need)]
    else:
        colors = palette[:need]
    pattern = np.array(colors, dtype=int).reshape(rows, cols)

    reveals = np.zeros((copies, rows, cols), dtype=bool)
    indices = [(r, c) for r in range(rows) for c in range(cols)]
    random.shuffle(indices)
    for r, c in indices:
        reveals[random.randrange(copies), r, c] = True
    for k in range(copies):
        extra = random.randint(1, max(1, rows * cols - 1))
        picks = random.sample(indices, extra)
        for r, c in picks:
            reveals[k, r, c] = True
    if any(reveals[k].all() for k in range(copies)):
        full_idx = next(k for k in range(copies) if reveals[k].all())
        r, c = random.choice(indices)
        reveals[full_idx, r, c] = False
    if any(not reveals[k].any() for k in range(copies)):
        return None

    grid = np.full((grid_h, grid_w), bg, dtype=int)
    for k, (r0, c0) in enumerate(positions):
        patch = np.full((lattice_h, lattice_w), bg, dtype=int)
        for rr in range(rows + 1):
            patch[rr * (cell_h + 1), :] = frame_color
        for cc in range(cols + 1):
            patch[:, cc * (cell_w + 1)] = frame_color
        for rr in range(rows):
            for cc in range(cols):
                if not reveals[k, rr, cc]:
                    continue
                rs = slice(rr * (cell_h + 1) + 1, rr * (cell_h + 1) + 1 + cell_h)
                cs = slice(cc * (cell_w + 1) + 1, cc * (cell_w + 1) + 1 + cell_w)
                patch[rs, cs] = int(pattern[rr, cc])
        grid[r0 : r0 + lattice_h, c0 : c0 + lattice_w] = patch

    if Counter(map(int, grid.reshape(-1)))[frame_color] <= max(Counter(map(int, grid.reshape(-1))).values()):
        pass

    try:
        out = main(grid)
    except Exception:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
