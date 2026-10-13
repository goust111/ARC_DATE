from __future__ import annotations

import random

import numpy as np


BG = 0
BRANCHES = ("tb", "lr", "both", "up", "down", "left", "right")
QUADRANTS = ("TL", "TR", "BL", "BR")
ROW_SIDE = {"TL": 0, "TR": 0, "BL": 1, "BR": 1}
COL_SIDE = {"TL": 0, "TR": 1, "BL": 0, "BR": 1}
ROW_MIRROR = {"TL": "BL", "TR": "BR", "BL": "TL", "BR": "TR"}
COL_MIRROR = {"TL": "TR", "TR": "TL", "BL": "BR", "BR": "BL"}
TRANSLATION_TARGETS = {
    "up": {"BL": "TL", "BR": "TR"},
    "down": {"TL": "BL", "TR": "BR"},
    "left": {"TR": "TL", "BR": "BL"},
    "right": {"TL": "TR", "BL": "BR"},
}

MAX_GENERATION_ATTEMPTS = 400
GRID_RANGE_BY_COUNT = {2: (18, 26), 3: (22, 28), 4: (26, 30)}


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            stack = [(r, c)]
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _quadrant_slices(interior_shape: tuple[int, int]) -> dict[str, tuple[slice, slice]]:
    h, w = map(int, interior_shape)
    assert h >= 5 and w >= 5 and h % 2 == 1 and w % 2 == 1, "interior must have a center row and column"
    qh = h // 2
    qw = w // 2
    return {
        "TL": (slice(0, qh), slice(0, qw)),
        "TR": (slice(0, qh), slice(qw + 1, w)),
        "BL": (slice(qh + 1, h), slice(0, qw)),
        "BR": (slice(qh + 1, h), slice(qw + 1, w)),
    }


def _branch_from_marks(marks: set[str]) -> str | None:
    if marks == {"top", "bottom"}:
        return "tb"
    if marks == {"left", "right"}:
        return "lr"
    if marks == {"top", "bottom", "left", "right"}:
        return "both"
    if len(marks) == 1:
        return next(iter(marks))
    return None


def _frame_candidate(mask: np.ndarray, coords: list[tuple[int, int]]) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 7 or w < 7 or h % 2 == 0 or w % 2 == 0:
        return None

    local = mask[r0 : r1 + 1, c0 : c1 + 1]
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.all(local[expected]):
        return None

    extras = local & ~expected
    mid_r = h // 2
    mid_c = w // 2
    mark_positions = {
        "top": (1, mid_c),
        "bottom": (h - 2, mid_c),
        "left": (mid_r, 1),
        "right": (mid_r, w - 2),
    }

    marks: set[str] = set()
    for name, (rr, cc) in mark_positions.items():
        if extras[rr, cc]:
            marks.add(name)
            extras[rr, cc] = False
    if np.any(extras):
        return None

    branch = _branch_from_marks(marks)
    if branch is None:
        return None

    return {"bbox": (r0, c0, r1, c1), "branch": branch, "marks": marks}


def _transform_patch(source_patch: np.ndarray, source_quad: str, target_quad: str) -> np.ndarray:
    patch = np.asarray(source_patch, dtype=bool)
    if ROW_SIDE[source_quad] != ROW_SIDE[target_quad]:
        patch = np.flipud(patch)
    if COL_SIDE[source_quad] != COL_SIDE[target_quad]:
        patch = np.fliplr(patch)
    return patch


def _target_patches(source_patch: np.ndarray, source_quad: str, branch: str) -> dict[str, np.ndarray]:
    source_patch = np.asarray(source_patch, dtype=bool)
    if branch == "tb":
        target = ROW_MIRROR[source_quad]
        return {target: _transform_patch(source_patch, source_quad, target)}
    if branch == "lr":
        target = COL_MIRROR[source_quad]
        return {target: _transform_patch(source_patch, source_quad, target)}
    if branch == "both":
        return {
            quad: _transform_patch(source_patch, source_quad, quad)
            for quad in QUADRANTS
            if quad != source_quad
        }
    assert branch in TRANSLATION_TARGETS, "single-edge branch must be directional"
    assert source_quad in TRANSLATION_TARGETS[branch], "source quadrant must support translation direction"
    target = TRANSLATION_TARGETS[branch][source_quad]
    return {target: source_patch.copy()}


def _solve_frame(out: np.ndarray, grid: np.ndarray, frame_color: int, candidate: dict[str, object]) -> None:
    r0, c0, r1, c1 = candidate["bbox"]  # type: ignore[index]
    branch = candidate["branch"]  # type: ignore[index]

    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    interior_out = out[r0 + 1 : r1, c0 + 1 : c1]
    quads = _quadrant_slices(interior.shape)

    non_quadrant = np.ones(interior.shape, dtype=bool)
    for rs, cs in quads.values():
        non_quadrant[rs, cs] = False
    cross_colors = set(map(int, np.unique(interior[non_quadrant])))
    assert cross_colors <= {BG, int(frame_color)}, "center cross may only contain background or frame markers"

    quadrant_colors = set(map(int, np.unique(interior[~non_quadrant])))
    quadrant_colors.discard(BG)
    quadrant_colors.discard(int(frame_color))
    assert len(quadrant_colors) == 1, "exactly one pattern color must appear across quadrants"
    pattern_color = quadrant_colors.pop()

    source_quads: list[str] = []
    source_patch: np.ndarray | None = None
    for name, (rs, cs) in quads.items():
        quad = interior[rs, cs]
        values = set(map(int, np.unique(quad)))
        assert int(frame_color) not in values, "frame color markers must stay on the center cross"
        if pattern_color in values:
            assert values <= {BG, pattern_color}, "source quadrant must contain one pattern color"
            source_quads.append(name)
            source_patch = quad == pattern_color
        else:
            assert values <= {BG}, "non-source quadrants must start empty"

    assert len(source_quads) == 1 and source_patch is not None, "exactly one source quadrant is required"
    source_quad = source_quads[0]

    for target_quad, patch in _target_patches(source_patch, source_quad, str(branch)).items():
        rs, cs = quads[target_quad]
        quad_view = interior_out[rs, cs]
        assert set(map(int, np.unique(quad_view))) <= {BG}, "target quadrants must start empty"
        quad_view[patch] = int(pattern_color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete the interior pattern inside marked rectangular frames.

    Each frame contains a single source quadrant and inward midpoint markers on selected edges.
    The marker layout encodes either horizontal mirroring, vertical mirroring, both-axis mirroring,
    or a one-step translation toward the marked edge.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a nonempty 2D grid"

    out = g.copy()
    frame_found = False
    for color in sorted(int(v) for v in np.unique(g) if int(v) != BG):
        for coords in _components(g == color):
            candidate = _frame_candidate(g == color, coords)
            if candidate is None:
                continue
            frame_found = True
            _solve_frame(out, g, int(color), candidate)

    assert frame_found, "expected at least one marked frame"
    return out


def _random_connected_patch(height: int, width: int, branch: str) -> np.ndarray | None:
    area = height * width
    min_cells = max(2, area // 3)
    max_cells = max(min_cells, area - 1)

    for _ in range(80):
        target = random.randint(min_cells, max_cells)
        seed = (random.randrange(height), random.randrange(width))
        cells = {seed}
        frontier = [seed]
        while len(cells) < target and frontier:
            r, c = random.choice(frontier)
            options = [(nr, nc) for nr, nc in _neighbors4(r, c) if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells]
            if not options:
                frontier.remove((r, c))
                continue
            cell = random.choice(options)
            cells.add(cell)
            frontier.append(cell)

        if len(cells) != target:
            continue

        patch = np.zeros((height, width), dtype=bool)
        for r, c in cells:
            patch[r, c] = True

        if branch == "tb" and np.array_equal(patch, np.flipud(patch)):
            continue
        if branch == "lr" and np.array_equal(patch, np.fliplr(patch)):
            continue
        if branch == "both" and np.array_equal(patch, np.flipud(patch)) and np.array_equal(patch, np.fliplr(patch)):
            continue
        return patch
    return None


def _compatible_sources(branch: str) -> tuple[str, ...]:
    if branch in TRANSLATION_TARGETS:
        return tuple(sorted(TRANSLATION_TARGETS[branch]))
    return QUADRANTS


def _draw_frame(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int) -> None:
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _draw_marks(grid: np.ndarray, r0: int, c0: int, r1: int, c1: int, color: int, branch: str) -> None:
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if branch in ("tb", "both", "top"):
        grid[r0 + 1, mid_c] = int(color)
    if branch in ("tb", "both", "bottom"):
        grid[r1 - 1, mid_c] = int(color)
    if branch in ("lr", "both", "left"):
        grid[mid_r, c0 + 1] = int(color)
    if branch in ("lr", "both", "right"):
        grid[mid_r, c1 - 1] = int(color)


def _quadrant_origin(r0: int, c0: int, qh: int, qw: int, quadrant: str) -> tuple[int, int]:
    row = r0 + 1 if ROW_SIDE[quadrant] == 0 else r0 + qh + 2
    col = c0 + 1 if COL_SIDE[quadrant] == 0 else c0 + qw + 2
    return row, col


def _paint_patch(grid: np.ndarray, top: int, left: int, patch: np.ndarray, color: int) -> None:
    view = grid[top : top + patch.shape[0], left : left + patch.shape[1]]
    view[patch] = int(color)


def _render_object(inp: np.ndarray, out: np.ndarray, spec: dict[str, object]) -> None:
    r0 = int(spec["r0"])
    c0 = int(spec["c0"])
    qh = int(spec["qh"])
    qw = int(spec["qw"])
    r1 = r0 + 2 * qh + 2
    c1 = c0 + 2 * qw + 2
    frame_color = int(spec["frame_color"])
    pattern_color = int(spec["pattern_color"])
    branch = str(spec["branch"])
    source_quad = str(spec["source_quad"])
    source_patch = np.asarray(spec["source_patch"], dtype=bool)

    _draw_frame(inp, r0, c0, r1, c1, frame_color)
    _draw_frame(out, r0, c0, r1, c1, frame_color)
    _draw_marks(inp, r0, c0, r1, c1, frame_color, branch)
    _draw_marks(out, r0, c0, r1, c1, frame_color, branch)

    sr, sc = _quadrant_origin(r0, c0, qh, qw, source_quad)
    _paint_patch(inp, sr, sc, source_patch, pattern_color)
    _paint_patch(out, sr, sc, source_patch, pattern_color)

    for target_quad, patch in _target_patches(source_patch, source_quad, branch).items():
        tr, tc = _quadrant_origin(r0, c0, qh, qw, target_quad)
        _paint_patch(out, tr, tc, patch, pattern_color)


def _place_objects(specs: list[dict[str, object]], height: int, width: int) -> bool:
    occupied = np.zeros((height, width), dtype=bool)
    order = sorted(range(len(specs)), key=lambda idx: -(2 * int(specs[idx]["qh"]) + 3) * (2 * int(specs[idx]["qw"]) + 3))
    for idx in order:
        qh = int(specs[idx]["qh"])
        qw = int(specs[idx]["qw"])
        obj_h = 2 * qh + 3
        obj_w = 2 * qw + 3
        positions: list[tuple[int, int]] = []
        for r0 in range(1, height - obj_h):
            for c0 in range(1, width - obj_w):
                r1 = r0 + obj_h - 1
                c1 = c0 + obj_w - 1
                rr0 = max(0, r0 - 1)
                cc0 = max(0, c0 - 1)
                rr1 = min(height, r1 + 2)
                cc1 = min(width, c1 + 2)
                if not occupied[rr0:rr1, cc0:cc1].any():
                    positions.append((r0, c0))
        if not positions:
            return False
        r0, c0 = random.choice(positions)
        specs[idx]["r0"] = int(r0)
        specs[idx]["c0"] = int(c0)
        occupied[r0 : r0 + obj_h, c0 : c0 + obj_w] = True
    return True


def _build_specs() -> tuple[int, int, list[dict[str, object]]] | None:
    n_objects = random.randint(2, 4)
    min_side, max_side = GRID_RANGE_BY_COUNT[n_objects]
    height = random.randint(min_side, max_side)
    width = random.randint(min_side, max_side)

    frame_pattern_colors = random.sample(list(range(1, 10)), 2 * n_objects)
    branches = random.sample(list(BRANCHES), n_objects)
    max_half = 3 if n_objects == 4 else 4

    specs: list[dict[str, object]] = []
    for idx in range(n_objects):
        branch = branches[idx]
        qh = random.randint(2, max_half)
        qw = random.randint(2, max_half)
        patch = _random_connected_patch(qh, qw, branch)
        if patch is None:
            return None
        specs.append(
            {
                "frame_color": int(frame_pattern_colors[2 * idx]),
                "pattern_color": int(frame_pattern_colors[2 * idx + 1]),
                "branch": branch,
                "source_quad": random.choice(_compatible_sources(branch)),
                "source_patch": patch,
                "qh": qh,
                "qw": qw,
            }
        )

    if not _place_objects(specs, height, width):
        return None
    return height, width, specs


def _gen_attempt() -> np.ndarray | None:
    built = _build_specs()
    if built is None:
        return None
    height, width, specs = built

    inp = np.full((height, width), BG, dtype=int)
    out = np.full((height, width), BG, dtype=int)
    for spec in specs:
        _render_object(inp, out, spec)

    try:
        actual = main(inp)
    except Exception:
        return None
    if not np.array_equal(actual, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid marked-frame symmetry task")
