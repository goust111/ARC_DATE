from __future__ import annotations

import random
from collections import deque

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

TEMPLATE_COLORS = (BLUE, RED, GREEN)
RIGHT_COLORS = (YELLOW, PINK, ORANGE, TEAL, MAROON)
NOISE_COLORS = (BLUE, RED, GREEN, YELLOW, PINK, ORANGE, TEAL, MAROON, MAROON)
MAX_ATTEMPTS = 400
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIRS8 = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1),           (0, 1),
    (1, -1),  (1, 0),  (1, 1),
)


def _normalize_shape(coords) -> tuple[tuple[int, int], ...]:
    pts = [(int(r), int(c)) for r, c in coords]
    r0 = min(r for r, _ in pts)
    c0 = min(c for _, c in pts)
    return tuple(sorted((r - r0, c - c0) for r, c in pts))


def _shape_bbox(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return max(r for r, _ in shape) + 1, max(c for _, c in shape) + 1


def _shape_mask(shape: tuple[tuple[int, int], ...]) -> np.ndarray:
    h, w = _shape_bbox(shape)
    mask = np.zeros((h, w), dtype=bool)
    for r, c in shape:
        mask[r, c] = True
    return mask


def _dihedral_variants(shape: tuple[tuple[int, int], ...]) -> list[tuple[tuple[int, int], ...]]:
    mask = _shape_mask(shape)
    out = []
    seen = set()
    for k in range(4):
        rotated = np.rot90(mask, k)
        for arr in (rotated, np.fliplr(rotated)):
            key = _normalize_shape(np.argwhere(arr))
            if key not in seen:
                seen.add(key)
                out.append(key)
    return out


def _is_connected(coords) -> bool:
    cells = {tuple(map(int, p)) for p in coords}
    start = next(iter(cells))
    stack = [start]
    seen = {start}
    while stack:
        r, c = stack.pop()
        for dr, dc in DIRS4:
            nb = (r + dr, c + dc)
            if nb in cells and nb not in seen:
                seen.add(nb)
                stack.append(nb)
    return seen == cells


def _enumerate_shapes() -> list[tuple[tuple[int, int], ...]]:
    cells = [(r, c) for r in range(3) for c in range(3)]
    shapes = set()
    for bits in range(1 << len(cells)):
        area = bits.bit_count()
        if not 2 <= area <= 6:
            continue
        coords = [cells[i] for i in range(len(cells)) if (bits >> i) & 1]
        if _is_connected(coords):
            shapes.add(_normalize_shape(coords))
    return sorted(shapes, key=lambda shp: (len(shp), _shape_bbox(shp), shp))


ALL_SHAPES = _enumerate_shapes()


def _fits(shape: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    sh, sw = _shape_bbox(shape)
    return sh <= h and sw <= w


def _build_line_positions(sizes: list[int]) -> tuple[list[int], list[tuple[int, int]], int]:
    lines = [0]
    ranges = []
    cur = 0
    for size in sizes:
        ranges.append((cur + 1, cur + size))
        cur += size + 1
        lines.append(cur)
    return lines, ranges, cur + 1


def _find_components(
    grid: np.ndarray,
    *,
    colors: list[int] | tuple[int, ...] | None = None,
    exclude: set[int] | None = None,
) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    exclude = set() if exclude is None else {int(x) for x in exclude}
    if colors is None:
        palette = sorted(set(map(int, g.flatten())) - exclude)
    else:
        palette = [int(c) for c in colors if int(c) not in exclude]

    comps = []
    for color in palette:
        starts = np.argwhere((g == color) & ~seen)
        for sr, sc in starts:
            sr, sc = int(sr), int(sc)
            if seen[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            coords = []
            while q:
                r, c = q.popleft()
                coords.append((r, c))
                for dr, dc in DIRS4:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and g[nr, nc] == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": np.array(coords, dtype=int),
                    "shape": _normalize_shape(coords),
                }
            )
    return comps


def _extract_templates(grid: np.ndarray) -> tuple[dict[tuple[tuple[int, int], ...], int], tuple[int, int, int, int]]:
    g = np.asarray(grid, dtype=int)
    grey_coords = np.argwhere(g == GREY)
    assert grey_coords.size > 0, "expected a grey lattice on the left"
    r0 = int(grey_coords[:, 0].min())
    c0 = int(grey_coords[:, 1].min())
    r1 = int(grey_coords[:, 0].max())
    c1 = int(grey_coords[:, 1].max())

    lattice = g[r0 : r1 + 1, c0 : c1 + 1]
    sep_rows = [r for r in range(lattice.shape[0]) if np.all(lattice[r, :] == GREY)]
    sep_cols = [c for c in range(lattice.shape[1]) if np.all(lattice[:, c] == GREY)]
    assert len(sep_rows) >= 4 and len(sep_cols) >= 4, "expected a bordered 3x3 grey lattice"

    row_ranges = [(sep_rows[i] + 1, sep_rows[i + 1] - 1) for i in range(len(sep_rows) - 1)]
    col_ranges = [(sep_cols[i] + 1, sep_cols[i + 1] - 1) for i in range(len(sep_cols) - 1)]
    assert len(row_ranges) == 3 and len(col_ranges) == 3, "expected exactly 3x3 cells"

    templates: dict[tuple[tuple[int, int], ...], int] = {}
    for rr0, rr1 in row_ranges:
        for cc0, cc1 in col_ranges:
            cell = lattice[rr0 : rr1 + 1, cc0 : cc1 + 1]
            coords = np.argwhere(cell != BLACK)
            if coords.size == 0:
                continue
            colors = sorted(set(map(int, cell[cell != BLACK].flatten())))
            if len(colors) != 1 or len(coords) < 2:
                continue
            shape = _normalize_shape(coords)
            assert shape not in templates, "template shapes must be unique"
            templates[shape] = colors[0]

    assert 2 <= len(templates) <= 3, "expected 2-3 valid template cells"
    return templates, (r0, c0, r1, c1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor right-side objects whose normalized shape matches a template extracted from a grey lattice.

    Input structure:
    - The left side contains a grey separator lattice split into 3x3 cells on a black background.
    - Exactly 2-3 cells are valid template cells: their non-black pixels use one color only and form a small 4-connected shape.
    - The remaining lattice cells are multicolor distractors, while the right side contains scattered monochrome 4-connected objects.

    Rule:
    - Extract the left-side template shapes by reading each cell interior and keeping only cells whose non-black pixels are all one color with at least two pixels.
    - Normalize each valid cell shape by translation and map it to that cell's color.
    - Find every monochrome 4-connected object to the right of the lattice, normalize its shape, and recolor it if it exactly matches one extracted template.

    Key insight:
    - The template is the precise non-black pixel pattern inside a valid cell, not the cell rectangle itself, so the lattice must be parsed before shape matching can happen.
    """
    g = np.asarray(grid, dtype=int)
    templates, (_r0, _c0, _r1, lattice_c1) = _extract_templates(g)
    out = g.copy()
    right = g[:, lattice_c1 + 1 :]
    comps = _find_components(right, exclude={BLACK})
    for comp in comps:
        color = templates.get(comp["shape"])
        if color is None:
            continue
        rr = comp["coords"][:, 0]
        cc = comp["coords"][:, 1] + lattice_c1 + 1
        out[rr, cc] = int(color)
    return out


def _make_noise_cell(h: int, w: int) -> np.ndarray:
    cell = np.zeros((h, w), dtype=int)
    positions = [(r, c) for r in range(h) for c in range(w)]
    k = random.randint(2, min(len(positions), 4))
    picked = random.sample(positions, k)
    colors = random.sample(list(dict.fromkeys(NOISE_COLORS)), 2)
    cell[picked[0]] = colors[0]
    cell[picked[1]] = colors[1]
    for pos in picked[2:]:
        cell[pos] = random.choice(colors if random.random() < 0.7 else list(dict.fromkeys(NOISE_COLORS)))
    assert len(set(map(int, cell[cell != BLACK].flatten()))) >= 2
    return cell


def _choose_templates(row_sizes: list[int], col_sizes: list[int], count: int):
    all_cells = [(r, c) for r in range(3) for c in range(3)]
    valid_cells = random.sample(all_cells, count)
    colors = random.sample(TEMPLATE_COLORS, count)
    templates = []
    used_shapes = set()

    for i, ((ri, ci), color) in enumerate(zip(valid_cells, colors)):
        h, w = row_sizes[ri], col_sizes[ci]
        candidates = [s for s in ALL_SHAPES if _fits(s, h, w) and len(s) < h * w and s not in used_shapes]
        if i == 0:
            strong = [s for s in candidates if any(v != s for v in _dihedral_variants(s))]
            if strong:
                candidates = strong
        if not candidates:
            return None
        shape = random.choice(candidates)
        sh, sw = _shape_bbox(shape)
        off_r = random.randint(0, h - sh)
        off_c = random.randint(0, w - sw)
        templates.append(
            {
                "cell": (ri, ci),
                "color": int(color),
                "shape": shape,
                "offset": (off_r, off_c),
                "cell_size": (h, w),
            }
        )
        used_shapes.add(shape)

    decoy_shape = None
    decoy_from = None
    for template in templates:
        variants = [v for v in _dihedral_variants(template["shape"]) if v != template["shape"] and v not in used_shapes]
        if variants:
            decoy_shape = random.choice(variants)
            decoy_from = template["shape"]
            break
    if decoy_shape is None:
        return None

    return templates, decoy_shape, decoy_from


def _render_left(row_sizes: list[int], col_sizes: list[int], templates: list[dict]) -> np.ndarray:
    row_lines, row_ranges, h = _build_line_positions(row_sizes)
    col_lines, col_ranges, w = _build_line_positions(col_sizes)
    left = np.zeros((h, w), dtype=int)
    left[row_lines, :] = GREY
    left[:, col_lines] = GREY

    by_cell = {tpl["cell"]: tpl for tpl in templates}
    for ri, (r0, r1) in enumerate(row_ranges):
        for ci, (c0, c1) in enumerate(col_ranges):
            cell_h = r1 - r0 + 1
            cell_w = c1 - c0 + 1
            if (ri, ci) in by_cell:
                tpl = by_cell[(ri, ci)]
                off_r, off_c = tpl["offset"]
                for sr, sc in tpl["shape"]:
                    left[r0 + off_r + sr, c0 + off_c + sc] = tpl["color"]
            else:
                left[r0 : r1 + 1, c0 : c1 + 1] = _make_noise_cell(cell_h, cell_w)
    return left


def _make_object_specs(templates: list[dict], decoy_shape, decoy_from):
    n_templates = len(templates)
    while True:
        extra_matches = random.randint(1, 3 if n_templates == 2 else 2)
        unmatched_count = random.randint(2, 4)
        total = n_templates + extra_matches + unmatched_count
        if 6 <= total <= 10:
            break

    specs = []
    for tpl in templates:
        specs.append(
            {
                "shape": tpl["shape"],
                "src_color": random.choice(RIGHT_COLORS),
                "matched_color": tpl["color"],
                "matched": True,
                "decoy": False,
            }
        )

    for _ in range(extra_matches):
        tpl = random.choice(templates)
        specs.append(
            {
                "shape": tpl["shape"],
                "src_color": random.choice(RIGHT_COLORS),
                "matched_color": tpl["color"],
                "matched": True,
                "decoy": False,
            }
        )

    specs.append(
        {
            "shape": decoy_shape,
            "src_color": random.choice(RIGHT_COLORS),
            "matched_color": None,
            "matched": False,
            "decoy": True,
            "same_area_as": len(decoy_from),
        }
    )

    template_shapes = {tpl["shape"] for tpl in templates}
    while len(specs) < total:
        candidates = [s for s in ALL_SHAPES if s not in template_shapes]
        shape = random.choice(candidates)
        specs.append(
            {
                "shape": shape,
                "src_color": random.choice(RIGHT_COLORS),
                "matched_color": None,
                "matched": False,
                "decoy": False,
            }
        )

    random.shuffle(specs)
    specs.sort(key=lambda spec: (-len(spec["shape"]), _shape_bbox(spec["shape"])))
    return specs


def _can_place(blocked: np.ndarray, shape: tuple[tuple[int, int], ...], r0: int, c0: int) -> bool:
    h, w = blocked.shape
    for dr, dc in shape:
        rr, cc = r0 + dr, c0 + dc
        if not (0 <= rr < h and 0 <= cc < w) or blocked[rr, cc]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, coords: list[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in coords:
        blocked[r, c] = True
        for dr, dc in DIRS8:
            nr, nc = r + dr, c + dc
            if 0 <= nr < h and 0 <= nc < w:
                blocked[nr, nc] = True


def _place_right_objects(height: int, width: int, specs: list[dict]):
    right = np.zeros((height, width), dtype=int)
    expected = np.zeros((height, width), dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    placements = []

    for spec in specs:
        sh, sw = _shape_bbox(spec["shape"])
        placed = None
        for _ in range(250):
            r0 = random.randint(0, height - sh)
            c0 = random.randint(0, width - sw)
            if not _can_place(blocked, spec["shape"], r0, c0):
                continue
            placed = [(r0 + dr, c0 + dc) for dr, dc in spec["shape"]]
            break
        if placed is None:
            return None

        for r, c in placed:
            right[r, c] = spec["src_color"]
            expected[r, c] = spec["matched_color"] if spec["matched"] else spec["src_color"]
        _mark_blocked(blocked, placed)
        placements.append({**spec, "coords": placed})

    return right, expected, placements


def _validate_case(
    grid: np.ndarray,
    expected: np.ndarray,
    templates: list[dict],
    placements: list[dict],
    left_width: int,
) -> bool:
    if grid.ndim != 2 or expected.ndim != 2 or max(grid.shape) > 30 or max(expected.shape) > 30:
        return False
    if np.array_equal(grid, expected):
        return False
    if not np.array_equal(grid[:, :left_width], expected[:, :left_width]):
        return False
    if len({tpl["shape"] for tpl in templates}) != len(templates):
        return False
    if not any(len(tpl["shape"]) < tpl["cell_size"][0] * tpl["cell_size"][1] for tpl in templates):
        return False
    if not any(obj["matched"] for obj in placements) or not any(not obj["matched"] for obj in placements):
        return False
    if not any(obj.get("decoy", False) for obj in placements):
        return False
    if not any(obj.get("decoy", False) and obj["same_area_as"] == len(obj["shape"]) for obj in placements):
        return False

    for tpl in templates:
        if not any(obj["matched"] and obj["shape"] == tpl["shape"] for obj in placements):
            return False

    right_components = _find_components(grid[:, left_width + 1 :], exclude={BLACK})
    if len(right_components) != len(placements):
        return False

    solved = main(grid)
    if not np.array_equal(solved, expected):
        return False

    changed_pixels = int(np.count_nonzero(grid != expected))
    matched_area = sum(len(obj["shape"]) for obj in placements if obj["matched"])
    return changed_pixels == matched_area


def _generate_case():
    row_sizes = [random.randint(2, 3) for _ in range(3)]
    col_sizes = [random.randint(2, 3) for _ in range(3)]
    template_count = random.choice((2, 3))

    chosen = _choose_templates(row_sizes, col_sizes, template_count)
    if chosen is None:
        return None
    templates, decoy_shape, decoy_from = chosen
    left = _render_left(row_sizes, col_sizes, templates)

    left_h, left_w = left.shape
    right_w = random.randint(10, 16)
    height = random.randint(max(left_h, 16), 26)
    if left_w + 1 + right_w > 30:
        return None

    specs = _make_object_specs(templates, decoy_shape, decoy_from)
    placed = _place_right_objects(height, right_w, specs)
    if placed is None:
        return None
    right, expected_right, placements = placed

    grid = np.zeros((height, left_w + 1 + right_w), dtype=int)
    expected = np.zeros_like(grid)
    grid[:left_h, :left_w] = left
    expected[:left_h, :left_w] = left
    grid[:, left_w + 1 :] = right
    expected[:, left_w + 1 :] = expected_right

    if not _validate_case(grid, expected, templates, placements, left_w):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _generate_case()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid puzzle instance")
