from __future__ import annotations

from collections import deque
import random

import numpy as np


NEIGHBORS4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
SIDES = ("top", "bottom", "left", "right")
SHAPE_KINDS = ("blob", "path", "el", "cross")
GEN_RETRIES = 400
MAX_GRID_SIZE = 30


def _background_color(grid: np.ndarray) -> int:
    vals, counts = np.unique(grid, return_counts=True)
    return int(vals[np.argmax(counts)])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in NEIGHBORS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _component_mask(coords: list[tuple[int, int]]) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    r0, c0, r1, c1 = _bbox(coords)
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in coords:
        mask[r - r0, c - c0] = True
    return mask, (r0, c0, r1, c1)


def _touches_boundary(coords: list[tuple[int, int]], h: int, w: int) -> bool:
    return any(r == 0 or c == 0 or r == h - 1 or c == w - 1 for r, c in coords)


def _touch_sides(coords: list[tuple[int, int]], h: int, w: int) -> set[str]:
    sides: set[str] = set()
    for r, c in coords:
        if r == 0:
            sides.add("top")
        if r == h - 1:
            sides.add("bottom")
        if c == 0:
            sides.add("left")
        if c == w - 1:
            sides.add("right")
    return sides


def _is_connected(mask: np.ndarray) -> bool:
    pts = np.argwhere(mask)
    if len(pts) == 0:
        return False
    seen = np.zeros_like(mask, dtype=bool)
    q = deque([tuple(map(int, pts[0]))])
    seen[q[0]] = True
    count = 0
    while q:
        r, c = q.popleft()
        count += 1
        for dr, dc in NEIGHBORS4:
            nr, nc = r + dr, c + dc
            if 0 <= nr < mask.shape[0] and 0 <= nc < mask.shape[1] and mask[nr, nc] and not seen[nr, nc]:
                seen[nr, nc] = True
                q.append((nr, nc))
    return count == int(mask.sum())


def _crop_mask(mask: np.ndarray) -> np.ndarray:
    coords = np.argwhere(mask)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return mask[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _is_rectangular_border(mask: np.ndarray) -> bool:
    h, w = mask.shape
    if h < 3 or w < 3:
        return False
    border = np.zeros_like(mask, dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    return bool(np.array_equal(mask, border))


def _extract_templates(grid: np.ndarray, bg: int) -> dict[int, np.ndarray]:
    h, w = grid.shape
    templates: dict[int, np.ndarray] = {}
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        comps = _components(grid == color)
        boundary = [coords for coords in comps if _touches_boundary(coords, h, w)]
        if not boundary:
            continue
        assert len(boundary) == 1, "each template color must have exactly one boundary-touching component"
        mask, _ = _component_mask(boundary[0])
        templates[color] = mask
    return templates


def _find_fences(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    fences: list[dict[str, object]] = []
    for color in map(int, np.unique(grid)):
        if color == bg:
            continue
        for coords in _components(grid == color):
            if _touches_boundary(coords, h, w):
                continue
            mask, (r0, c0, r1, c1) = _component_mask(coords)
            if not _is_rectangular_border(mask):
                continue

            interior = grid[r0 + 1 : r1, c0 + 1 : c1]
            foreign = np.argwhere((interior != bg) & (interior != color))
            if len(foreign) != 1:
                continue

            seed_r, seed_c = map(int, foreign[0])
            seed_r += r0 + 1
            seed_c += c0 + 1
            seed_color = int(grid[seed_r, seed_c])
            assert seed_color != color
            fences.append(
                {
                    "border_color": color,
                    "bbox": (r0, c0, r1, c1),
                    "seed": (seed_r, seed_c),
                    "seed_color": seed_color,
                }
            )
    return fences


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each fenced interior with the same-color edge template after block scaling.

    Input structure:
    - The background is the most common color.
    - Several closed rectangular fences sit away from the boundary; each contains exactly
      one colored seed pixel.
    - A template library sits on one grid edge; each compact monochrome template uses the
      same color as one seed.

    Rule:
    - For each fence, use the seed color to find the matching template on the grid edge.
    - Compute the integer block scale from the fence interior size to the template bbox.
    - Expand each template pixel into a k x k block and paint that scaled shape inside the
      fence using the seed color.
    - The seed must lie inside one foreground block of the scaled template.
    """
    g = np.asarray(grid, dtype=int)
    bg = _background_color(g)
    templates = _extract_templates(g, bg)
    fences = _find_fences(g, bg)

    assert templates, "expected boundary templates"
    assert fences, "expected at least one fenced region"

    out = g.copy()
    for fence in fences:
        r0, c0, r1, c1 = fence["bbox"]  # type: ignore[assignment]
        seed_r, seed_c = fence["seed"]  # type: ignore[assignment]
        seed_color = int(fence["seed_color"])
        assert seed_color in templates, "every seed color must have a matching template"

        template = templates[seed_color]
        th, tw = template.shape
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        assert inner_h > 0 and inner_w > 0
        assert inner_h % th == 0 and inner_w % tw == 0
        scale_h = inner_h // th
        scale_w = inner_w // tw
        assert scale_h == scale_w
        scale = scale_h

        scaled = np.kron(template, np.ones((scale, scale), dtype=bool))
        rel_r = seed_r - (r0 + 1)
        rel_c = seed_c - (c0 + 1)
        assert 0 <= rel_r < inner_h and 0 <= rel_c < inner_w
        assert bool(scaled[rel_r, rel_c]), "seed must land on a foreground block"

        interior = out[r0 + 1 : r1, c0 + 1 : c1]
        interior[:, :] = bg
        interior[scaled] = seed_color

    return out


def _grow_blob(rng: random.Random, h: int, w: int, target: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    start = (rng.randrange(h), rng.randrange(w))
    mask[start] = True
    frontier = {start}
    while int(mask.sum()) < target and frontier:
        cell = rng.choice(tuple(frontier))
        nbrs = []
        for dr, dc in NEIGHBORS4:
            nr, nc = cell[0] + dr, cell[1] + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if nbrs:
            nxt = rng.choice(nbrs)
            mask[nxt] = True
            frontier.add(nxt)
        else:
            frontier.remove(cell)
    return _crop_mask(mask)


def _shape_blob(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    max_area = h * w
    target = rng.randint(max(3, max_area // 3), max(max_area - 1, 3))
    return _grow_blob(rng, h, w, target)


def _shape_path(rng: random.Random) -> np.ndarray:
    canvas_h = rng.randint(3, 6)
    canvas_w = rng.randint(3, 6)
    length = rng.randint(3, min(canvas_h * canvas_w, 9))
    r = rng.randrange(canvas_h)
    c = rng.randrange(canvas_w)
    cells = [(r, c)]
    used = {cells[0]}
    while len(cells) < length:
        cur_r, cur_c = cells[-1]
        options = []
        for dr, dc in NEIGHBORS4:
            nr, nc = cur_r + dr, cur_c + dc
            if 0 <= nr < canvas_h and 0 <= nc < canvas_w and (nr, nc) not in used:
                options.append((nr, nc))
        if not options:
            break
        nxt = rng.choice(options)
        cells.append(nxt)
        used.add(nxt)
    mask = np.zeros((canvas_h, canvas_w), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    return _crop_mask(mask)


def _shape_el(rng: random.Random) -> np.ndarray:
    h = rng.randint(2, 5)
    w = rng.randint(2, 5)
    mask = np.zeros((h, w), dtype=bool)
    corner = rng.choice(((0, 0), (0, w - 1), (h - 1, 0), (h - 1, w - 1)))
    if corner[1] == 0:
        mask[:, 0] = True
    else:
        mask[:, -1] = True
    if corner[0] == 0:
        mask[0, :] = True
    else:
        mask[-1, :] = True
    if h >= 3 and rng.random() < 0.5:
        if corner[1] == 0:
            mask[1 : h - 1, 1] = rng.random() < 0.5
        else:
            mask[1 : h - 1, -2] = rng.random() < 0.5
    if w >= 3 and rng.random() < 0.5:
        if corner[0] == 0:
            mask[1, 1 : w - 1] = rng.random() < 0.5
        else:
            mask[-2, 1 : w - 1] = rng.random() < 0.5
    return _crop_mask(mask)


def _shape_cross(rng: random.Random) -> np.ndarray:
    h = rng.choice((3, 5))
    w = rng.choice((3, 5))
    mask = np.zeros((h, w), dtype=bool)
    cr = h // 2
    cc = w // 2
    mask[cr, :] = True
    mask[:, cc] = True
    if rng.random() < 0.5:
        tip = rng.choice(((0, cc), (h - 1, cc), (cr, 0), (cr, w - 1)))
        mask[tip] = False
    return _crop_mask(mask)


def _valid_template(mask: np.ndarray) -> bool:
    if mask.size == 0:
        return False
    h, w = mask.shape
    if not (1 <= h <= 5 and 1 <= w <= 5):
        return False
    if int(mask.sum()) < 2:
        return False
    if not _is_connected(mask):
        return False
    if h > 1 and w > 1 and bool(mask.all()):
        return False
    if _is_rectangular_border(mask):
        return False
    return True


def _random_template(rng: random.Random) -> np.ndarray:
    makers = {
        "blob": _shape_blob,
        "path": _shape_path,
        "el": _shape_el,
        "cross": _shape_cross,
    }
    for _ in range(100):
        kind = rng.choice(SHAPE_KINDS)
        mask = makers[kind](rng)
        if _valid_template(mask):
            return mask
    raise RuntimeError("failed to generate template shape")


def _layout_rows(
    rng: random.Random, sizes: list[tuple[int, int]]
) -> tuple[int, int, list[tuple[int, int]]]:
    n = len(sizes)
    row_count = rng.randint(1, min(n, 3 if n >= 4 else 2))
    order = list(range(n))
    rng.shuffle(order)
    rows = [[] for _ in range(row_count)]
    for i, idx in enumerate(order):
        rows[i % row_count].append(idx)
    for row in rows:
        rng.shuffle(row)

    horiz_gaps = [[rng.randint(2, 4) for _ in range(len(row) - 1)] for row in rows]
    vert_gaps = [rng.randint(2, 4) for _ in range(row_count - 1)]
    row_widths = []
    row_heights = []
    for row, gaps in zip(rows, horiz_gaps):
        row_widths.append(sum(sizes[idx][1] for idx in row) + sum(gaps))
        row_heights.append(max(sizes[idx][0] for idx in row))

    left_pad = rng.randint(1, 3)
    right_pad = rng.randint(1, 3)
    top_pad = rng.randint(1, 3)
    bottom_pad = rng.randint(1, 3)
    extra_w = rng.randint(0, 2)
    extra_h = rng.randint(0, 2)
    main_w = max(row_widths) + left_pad + right_pad + extra_w
    main_h = sum(row_heights) + sum(vert_gaps) + top_pad + bottom_pad + extra_h

    positions = [(-1, -1)] * n
    y = top_pad
    for row_idx, row in enumerate(rows):
        row_width = row_widths[row_idx]
        row_height = row_heights[row_idx]
        x = rng.randint(left_pad, main_w - right_pad - row_width)
        for j, idx in enumerate(row):
            h, w = sizes[idx]
            positions[idx] = (y + rng.randint(0, row_height - h), x)
            if j < len(row) - 1:
                x += w + horiz_gaps[row_idx][j]
        if row_idx < len(rows) - 1:
            y += row_height + vert_gaps[row_idx]
    return main_h, main_w, positions


def _layout_cols(
    rng: random.Random, sizes: list[tuple[int, int]]
) -> tuple[int, int, list[tuple[int, int]]]:
    n = len(sizes)
    col_count = rng.randint(1, min(n, 3 if n >= 4 else 2))
    order = list(range(n))
    rng.shuffle(order)
    cols = [[] for _ in range(col_count)]
    for i, idx in enumerate(order):
        cols[i % col_count].append(idx)
    for col in cols:
        rng.shuffle(col)

    vert_gaps = [[rng.randint(2, 4) for _ in range(len(col) - 1)] for col in cols]
    horiz_gaps = [rng.randint(2, 4) for _ in range(col_count - 1)]
    col_widths = []
    col_heights = []
    for col, gaps in zip(cols, vert_gaps):
        col_widths.append(max(sizes[idx][1] for idx in col))
        col_heights.append(sum(sizes[idx][0] for idx in col) + sum(gaps))

    left_pad = rng.randint(1, 3)
    right_pad = rng.randint(1, 3)
    top_pad = rng.randint(1, 3)
    bottom_pad = rng.randint(1, 3)
    extra_w = rng.randint(0, 2)
    extra_h = rng.randint(0, 2)
    main_w = sum(col_widths) + sum(horiz_gaps) + left_pad + right_pad + extra_w
    main_h = max(col_heights) + top_pad + bottom_pad + extra_h

    positions = [(-1, -1)] * n
    x = left_pad
    for col_idx, col in enumerate(cols):
        col_width = col_widths[col_idx]
        y = rng.randint(top_pad, main_h - bottom_pad - col_heights[col_idx])
        for j, idx in enumerate(col):
            h, w = sizes[idx]
            positions[idx] = (y, x + rng.randint(0, col_width - w))
            if j < len(col) - 1:
                y += h + vert_gaps[col_idx][j]
        if col_idx < len(cols) - 1:
            x += col_width + horiz_gaps[col_idx]
    return main_h, main_w, positions


def _layout_fences(
    rng: random.Random, sizes: list[tuple[int, int]]
) -> tuple[int, int, list[tuple[int, int]]]:
    if rng.random() < 0.5:
        return _layout_rows(rng, sizes)
    return _layout_cols(rng, sizes)


def _place_mask(grid: np.ndarray, mask: np.ndarray, r0: int, c0: int, color: int) -> None:
    h, w = mask.shape
    patch = grid[r0 : r0 + h, c0 : c0 + w]
    patch[mask] = color


def _rand_edge_offset(rng: random.Random, slack: int) -> int:
    if slack <= 0:
        return 0
    if slack == 1:
        return rng.randint(0, 1)
    return rng.randint(1, slack - 1)


def _make_library_positions(
    rng: random.Random, side: str, dims: list[tuple[int, int]], height: int, width: int
) -> list[tuple[int, int]]:
    order = list(range(len(dims)))
    rng.shuffle(order)
    positions = [(-1, -1)] * len(dims)
    if side in ("left", "right"):
        gaps = [rng.randint(1, 3) for _ in range(len(order) - 1)]
        span = sum(dims[idx][0] for idx in order) + sum(gaps)
        start_r = _rand_edge_offset(rng, height - span)
        cur = start_r
        for j, idx in enumerate(order):
            h, w = dims[idx]
            c = 0 if side == "left" else width - w
            positions[idx] = (cur, c)
            if j < len(order) - 1:
                cur += h + gaps[j]
    else:
        gaps = [rng.randint(1, 3) for _ in range(len(order) - 1)]
        span = sum(dims[idx][1] for idx in order) + sum(gaps)
        start_c = _rand_edge_offset(rng, width - span)
        cur = start_c
        for j, idx in enumerate(order):
            h, w = dims[idx]
            r = 0 if side == "top" else height - h
            positions[idx] = (r, cur)
            if j < len(order) - 1:
                cur += w + gaps[j]
    return positions


def _generate_case(rng: random.Random | None = None) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random() if rng is None else rng

    num_fences = rng.choices([2, 3, 4], weights=[4, 4, 1], k=1)[0]
    bg = rng.randrange(10)
    remaining = [c for c in range(10) if c != bg]
    border_color = rng.choice(remaining)
    seed_colors = rng.sample([c for c in remaining if c != border_color], num_fences)
    side = rng.choice(SIDES)

    templates: list[dict[str, object]] = []
    scale_cap = {2: 4, 3: 3, 4: 2}[num_fences]
    for color in seed_colors:
        mask = _random_template(rng)
        th, tw = mask.shape
        local_cap = scale_cap
        if max(th, tw) >= 5:
            local_cap = min(local_cap, 2)
        scale = rng.randint(1, local_cap)
        templates.append(
            {
                "color": color,
                "mask": mask,
                "scale": scale,
                "inner_h": th * scale,
                "inner_w": tw * scale,
                "outer_h": th * scale + 2,
                "outer_w": tw * scale + 2,
            }
        )

    if max(int(t["scale"]) for t in templates) == 1 and scale_cap > 1:
        candidate = rng.randrange(num_fences)
        mask = templates[candidate]["mask"]  # type: ignore[index]
        th, tw = mask.shape  # type: ignore[assignment]
        upper = min(scale_cap, 2 if max(th, tw) >= 5 else scale_cap)
        if upper > 1:
            scale = rng.randint(2, upper)
            templates[candidate]["scale"] = scale
            templates[candidate]["inner_h"] = th * scale
            templates[candidate]["inner_w"] = tw * scale
            templates[candidate]["outer_h"] = th * scale + 2
            templates[candidate]["outer_w"] = tw * scale + 2

    fence_sizes = [(int(t["outer_h"]), int(t["outer_w"])) for t in templates]
    main_h, main_w, fence_positions = _layout_fences(rng, fence_sizes)
    lib_dims = [tuple(map(int, t["mask"].shape)) for t in templates]  # type: ignore[union-attr]

    if side in ("left", "right"):
        strip_w = max(w for _, w in lib_dims)
        gap = rng.randint(2, 3)
        extra_h = rng.randint(2, 4)
        extra_w = rng.randint(2, 4)
        lib_span = sum(h for h, _ in lib_dims) + (num_fences - 1) * 3
        height = max(main_h, lib_span) + extra_h
        width = strip_w + gap + main_w + extra_w
        if width > MAX_GRID_SIZE or height > MAX_GRID_SIZE:
            raise RuntimeError("grid too large")
        main_y = _rand_edge_offset(rng, height - main_h)
        if side == "left":
            main_x = strip_w + gap
        else:
            main_x = rng.randint(0, extra_w)
        lib_positions = _make_library_positions(rng, side, lib_dims, height, width)
    else:
        strip_h = max(h for h, _ in lib_dims)
        gap = rng.randint(2, 3)
        extra_h = rng.randint(2, 4)
        extra_w = rng.randint(2, 4)
        lib_span = sum(w for _, w in lib_dims) + (num_fences - 1) * 3
        height = strip_h + gap + main_h + extra_h
        width = max(main_w, lib_span) + extra_w
        if width > MAX_GRID_SIZE or height > MAX_GRID_SIZE:
            raise RuntimeError("grid too large")
        main_x = _rand_edge_offset(rng, width - main_w)
        if side == "top":
            main_y = strip_h + gap + rng.randint(0, extra_h)
        else:
            main_y = rng.randint(0, extra_h)
        lib_positions = _make_library_positions(rng, side, lib_dims, height, width)

    grid = np.full((height, width), bg, dtype=int)
    out = grid.copy()

    for idx, tpl in enumerate(templates):
        color = int(tpl["color"])
        mask = tpl["mask"]  # type: ignore[assignment]
        lr, lc = lib_positions[idx]
        _place_mask(grid, mask, lr, lc, color)
        _place_mask(out, mask, lr, lc, color)

    for idx, tpl in enumerate(templates):
        color = int(tpl["color"])
        mask = tpl["mask"]  # type: ignore[assignment]
        scale = int(tpl["scale"])
        outer_h = int(tpl["outer_h"])
        outer_w = int(tpl["outer_w"])
        fr, fc = fence_positions[idx]
        fr += main_y
        fc += main_x
        grid[fr, fc : fc + outer_w] = border_color
        grid[fr + outer_h - 1, fc : fc + outer_w] = border_color
        grid[fr : fr + outer_h, fc] = border_color
        grid[fr : fr + outer_h, fc + outer_w - 1] = border_color
        out[fr, fc : fc + outer_w] = border_color
        out[fr + outer_h - 1, fc : fc + outer_w] = border_color
        out[fr : fr + outer_h, fc] = border_color
        out[fr : fr + outer_h, fc + outer_w - 1] = border_color

        inner_r = fr + 1
        inner_c = fc + 1
        scaled = np.kron(mask, np.ones((scale, scale), dtype=bool))
        out[inner_r : inner_r + scaled.shape[0], inner_c : inner_c + scaled.shape[1]][scaled] = color

        foreground = np.argwhere(mask)
        anchor_r, anchor_c = map(int, foreground[rng.randrange(len(foreground))])
        seed_r = inner_r + anchor_r * scale + rng.randrange(scale)
        seed_c = inner_c + anchor_c * scale + rng.randrange(scale)
        grid[seed_r, seed_c] = color

    assert _background_color(grid) == bg

    template_side_sets = []
    for color in seed_colors:
        boundary = [comp for comp in _components(grid == color) if _touches_boundary(comp, height, width)]
        assert len(boundary) == 1
        template_side_sets.append(_touch_sides(boundary[0], height, width))
    common_side = set(SIDES)
    for side_set in template_side_sets:
        common_side &= side_set
    assert len(common_side) == 1

    generated = main(grid)
    assert np.array_equal(generated, out)
    return grid, out


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        try:
            grid, _ = _generate_case(rng)
        except Exception:
            continue
        return grid
    raise RuntimeError(f"failed to generate input after {GEN_RETRIES} attempts")
