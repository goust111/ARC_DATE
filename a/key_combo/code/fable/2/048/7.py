from __future__ import annotations

import numpy as np


BACKGROUND = 0
ARROW_COLOR = 2
BODY_COLOR = 3
EDGE_COLOR_POOL = np.array([1, 4, 5, 6, 7, 8, 9], dtype=int)

SIDES = ("top", "right", "bottom", "left")
SIDE_TO_DIR = {
    "top": (-1, 0),
    "right": (0, 1),
    "bottom": (1, 0),
    "left": (0, -1),
}
DIR_TO_SIDE = {value: key for key, value in SIDE_TO_DIR.items()}

GRID_MIN_SIDE = 14
GRID_MAX_SIDE = 22
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid, dtype=int), return_counts=True)
    return int(values[counts.argmax()])


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def _neighbors4(r: int, c: int) -> list[tuple[int, int]]:
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            out.append(comp)
    return out


def _rotate_point(point: tuple[int, int], k: int) -> tuple[int, int]:
    r, c = map(int, point)
    k %= 4
    if k == 0:
        return r, c
    if k == 1:
        return c, -r
    if k == 2:
        return -r, -c
    return -c, r


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(int(a[0]) - int(b[0])), abs(int(a[1]) - int(b[1])))


def _edge_side_colors(grid: np.ndarray, bg: int) -> dict[str, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert h >= 3 and w >= 3, "grid must have at least one interior row and column"

    side_slices = {
        "top": g[0, 1:-1],
        "bottom": g[-1, 1:-1],
        "left": g[1:-1, 0],
        "right": g[1:-1, -1],
    }

    side_colors: dict[str, int] = {}
    for side, segment in side_slices.items():
        assert segment.size > 0, "border side segment must be non-empty"
        color = int(segment[0])
        assert np.all(segment == color), f"{side} edge segment must be uniform"
        if color != bg:
            side_colors[side] = color

    assert side_colors, "expected at least one colored edge line"
    assert len(set(side_colors.values())) == len(side_colors), "colored edge lines must use distinct colors"
    return side_colors


def _parse_pointers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    g = np.asarray(grid, dtype=int)
    mask = (g != bg).copy()
    mask[0, :] = False
    mask[-1, :] = False
    mask[:, 0] = False
    mask[:, -1] = False

    comps = _components(mask)
    assert comps, "expected internal pointer components"

    pointers: list[dict[str, object]] = []
    for comp in comps:
        assert len(comp) == 2, "each internal component must be a 2-cell pointer"
        colors = [int(g[r, c]) for r, c in comp]
        assert sorted(colors) == [ARROW_COLOR, BODY_COLOR], "each pointer must contain one red arrow and one green body"

        body = next((r, c) for r, c in comp if int(g[r, c]) == BODY_COLOR)
        arrow = next((r, c) for r, c in comp if int(g[r, c]) == ARROW_COLOR)
        dr = int(arrow[0] - body[0])
        dc = int(arrow[1] - body[1])
        assert (dr, dc) in DIR_TO_SIDE, "pointer cells must be 4-neighbor adjacent"

        pointers.append(
            {
                "body": (int(body[0]), int(body[1])),
                "arrow": (int(arrow[0]), int(arrow[1])),
                "direction": (dr, dc),
            }
        )

    assert int(np.sum(g == BODY_COLOR)) == len(pointers), "every green cell must be a pointer body"
    assert int(np.sum(g == ARROW_COLOR)) == len(pointers), "every red cell must be a pointer arrow"
    return pointers


def _annotate_targets(
    grid: np.ndarray,
    pointers: list[dict[str, object]],
    side_colors: dict[str, int],
    bg: int,
) -> None:
    cell_to_pointer: dict[tuple[int, int], int] = {}
    for index, pointer in enumerate(pointers):
        for coord in (pointer["body"], pointer["arrow"]):
            assert coord not in cell_to_pointer, "pointer cells must not overlap"
            cell_to_pointer[coord] = int(index)

    for index, pointer in enumerate(pointers):
        dr, dc = pointer["direction"]
        br, bc = pointer["body"]
        r = int(br + 2 * dr)
        c = int(bc + 2 * dc)
        side = DIR_TO_SIDE[(int(dr), int(dc))]

        while _in_bounds(grid.shape, r, c):
            color = int(grid[r, c])
            if color == bg:
                r += int(dr)
                c += int(dc)
                continue

            coord = (int(r), int(c))
            if coord in cell_to_pointer:
                target_index = int(cell_to_pointer[coord])
                assert target_index != index, "pointer sightline cannot hit itself"
                target_pointer = pointers[target_index]
                hit_part = "body" if coord == target_pointer["body"] else "arrow"
                pointer["target_kind"] = "pointer"
                pointer["target_index"] = target_index
                pointer["hit_part"] = hit_part
                break

            hit_border = (
                (side == "top" and r == 0)
                or (side == "bottom" and r == grid.shape[0] - 1)
                or (side == "left" and c == 0)
                or (side == "right" and c == grid.shape[1] - 1)
            )
            assert hit_border, "non-background non-pointer targets must be edge lines"
            assert side in side_colors, "pointer must only terminate on a colored border side"
            assert color == side_colors[side], "sightline edge color must match its border side"
            pointer["target_kind"] = "edge"
            pointer["target_side"] = side
            pointer["target_color"] = color
            break
        else:
            raise AssertionError("every pointer sightline must hit a pointer or an edge line")


def _resolve_pointer_color(
    pointers: list[dict[str, object]],
    index: int,
    visiting: set[int],
) -> int:
    pointer = pointers[int(index)]
    cached = pointer.get("resolved_color")
    if cached is not None:
        return int(cached)

    if index in visiting:
        raise AssertionError("pointer dependencies must be acyclic")
    visiting.add(int(index))

    if pointer["target_kind"] == "edge":
        color = int(pointer["target_color"])
        depth = 0
    else:
        target_index = int(pointer["target_index"])
        color = _resolve_pointer_color(pointers, target_index, visiting)
        depth = 1 + int(pointers[target_index]["dependency_depth"])

    visiting.remove(int(index))
    pointer["resolved_color"] = int(color)
    pointer["dependency_depth"] = int(depth)
    return int(color)


def _analyze_scene(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert max(g.shape) <= 30, "ARC grids must fit within 30x30"

    bg = _get_bg(g)
    side_colors = _edge_side_colors(g, bg)
    pointers = _parse_pointers(g, bg)
    _annotate_targets(g, pointers, side_colors, bg)

    for index in range(len(pointers)):
        _resolve_pointer_color(pointers, index, set())

    return {
        "bg": int(bg),
        "side_colors": side_colors,
        "pointers": pointers,
    }


def _render_output(grid: np.ndarray, analysis: dict[str, object]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(analysis["bg"])
    for pointer in analysis["pointers"]:
        ar, ac = pointer["arrow"]
        br, bc = pointer["body"]
        out[int(ar), int(ac)] = bg
        out[int(br), int(bc)] = int(pointer["resolved_color"])
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Resolve each green pointer's inherited line-of-sight color, then erase its red arrow.

    Input structure:
    - The four border sides are solid colored edge lines on a background field.
    - Interior foreground objects are 2-cell pointers: one green body cell plus one
      adjacent red arrow cell whose relative position gives one of the four cardinal directions.
    - Some pointers look straight at a border line, while others first see another pointer cell.

    Rule:
    - For each pointer body, continue straight from the arrow side through background cells.
    - The first non-background thing on that ray decides the body's color: hitting a border
      line copies that side's color, and hitting any cell of another pointer copies that
      target pointer's final resolved color.
    - Resolve these dependencies without cycles, then output the same grid with red arrow
      cells erased to background and green body cells recolored to their resolved colors.

    Key insight:
    - Color propagates by directed visibility, not by nearest border: a pointer may inherit
      indirectly through another pointer, and hitting an arrow cell counts exactly like hitting its body.
    """
    analysis = _analyze_scene(np.asarray(grid, dtype=int))
    return _render_output(np.asarray(grid, dtype=int), analysis)


def _paint_edges(grid: np.ndarray, side_colors: dict[str, int]) -> None:
    grid[0, 1:-1] = int(side_colors["top"])
    grid[-1, 1:-1] = int(side_colors["bottom"])
    grid[1:-1, 0] = int(side_colors["left"])
    grid[1:-1, -1] = int(side_colors["right"])


def _pointer_path_to_border(body: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    r = int(body[0] + 2 * dr)
    c = int(body[1] + 2 * dc)
    cells: list[tuple[int, int]] = []
    while _in_bounds(shape, r, c):
        if r == 0 or c == 0 or r == shape[0] - 1 or c == shape[1] - 1:
            return cells
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    raise AssertionError("direct pointer path should reach a border")


def _cells_before_target(
    body: tuple[int, int],
    direction: tuple[int, int],
    target_cell: tuple[int, int],
) -> list[tuple[int, int]]:
    dr, dc = map(int, direction)
    r = int(body[0] + 2 * dr)
    c = int(body[1] + 2 * dc)
    cells: list[tuple[int, int]] = []
    while (r, c) != (int(target_cell[0]), int(target_cell[1])):
        cells.append((int(r), int(c)))
        r += dr
        c += dc
    return cells


def _can_place_pointer(
    body: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> bool:
    br, bc = map(int, body)
    dr, dc = map(int, direction)
    arrow = (br + dr, bc + dc)
    cells = [body, arrow]

    for r, c in cells:
        if not (1 <= int(r) < shape[0] - 1 and 1 <= int(c) < shape[1] - 1):
            return False
        if (int(r), int(c)) in occupied or (int(r), int(c)) in blocked:
            return False
        for other in occupied:
            if _chebyshev((int(r), int(c)), other) <= 1:
                return False
    return True


def _ray_clear_to_border(
    body: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
) -> bool:
    for cell in _pointer_path_to_border(body, direction, shape):
        if cell in occupied:
            return False
    return True


def _build_motif(
    rng: np.random.Generator,
    shape: tuple[int, int],
) -> tuple[list[dict[str, object]], set[tuple[int, int]], set[tuple[int, int]]] | None:
    hgap = int(rng.integers(3, 6))
    vgap = int(rng.integers(2, 5))
    rotation = int(rng.integers(4))

    canonical = [
        {"name": "root", "body": (0, 0), "arrow": (-1, 0)},
        {"name": "mid", "body": (0, hgap), "arrow": (0, hgap - 1)},
        {"name": "tail", "body": (-vgap - 1, hgap - 1), "arrow": (-vgap, hgap - 1)},
    ]

    transformed = []
    all_coords: list[tuple[int, int]] = []
    for pointer in canonical:
        body = _rotate_point(pointer["body"], rotation)
        arrow = _rotate_point(pointer["arrow"], rotation)
        transformed.append({"name": pointer["name"], "body": body, "arrow": arrow})
        all_coords.extend([body, arrow])

    min_r = min(r for r, _ in all_coords)
    max_r = max(r for r, _ in all_coords)
    min_c = min(c for _, c in all_coords)
    max_c = max(c for _, c in all_coords)

    root_dir = (
        int(transformed[0]["arrow"][0] - transformed[0]["body"][0]),
        int(transformed[0]["arrow"][1] - transformed[0]["body"][1]),
    )
    r_low = 1 - min_r
    r_high = shape[0] - 2 - max_r
    c_low = 1 - min_c
    c_high = shape[1] - 2 - max_c

    if root_dir == (-1, 0):
        r_low = max(r_low, 3)
    elif root_dir == (1, 0):
        r_high = min(r_high, shape[0] - 4)
    elif root_dir == (0, -1):
        c_low = max(c_low, 3)
    elif root_dir == (0, 1):
        c_high = min(c_high, shape[1] - 4)

    if r_low > r_high or c_low > c_high:
        return None

    root_r = int(rng.integers(r_low, r_high + 1))
    root_c = int(rng.integers(c_low, c_high + 1))

    pointers: list[dict[str, object]] = []
    occupied: set[tuple[int, int]] = set()
    blocked: set[tuple[int, int]] = set()

    for pointer in transformed:
        body = (int(root_r + pointer["body"][0]), int(root_c + pointer["body"][1]))
        arrow = (int(root_r + pointer["arrow"][0]), int(root_c + pointer["arrow"][1]))
        direction = (int(arrow[0] - body[0]), int(arrow[1] - body[1]))
        if not _can_place_pointer(body, direction, shape, occupied, blocked):
            return None
        pointers.append({"name": pointer["name"], "body": body, "arrow": arrow, "direction": direction})
        occupied.add(body)
        occupied.add(arrow)

    root = next(pointer for pointer in pointers if pointer["name"] == "root")
    mid = next(pointer for pointer in pointers if pointer["name"] == "mid")
    tail = next(pointer for pointer in pointers if pointer["name"] == "tail")

    blocked.update(_pointer_path_to_border(root["body"], root["direction"], shape))
    blocked.update(_cells_before_target(mid["body"], mid["direction"], root["body"]))
    blocked.update(_cells_before_target(tail["body"], tail["direction"], mid["arrow"]))
    return pointers, occupied, blocked


def _direct_pointer_candidates(
    side: str,
    shape: tuple[int, int],
) -> list[tuple[int, int]]:
    h, w = map(int, shape)
    if side == "top":
        return [(r, c) for r in range(3, h - 1) for c in range(1, w - 1)]
    if side == "bottom":
        return [(r, c) for r in range(1, h - 3) for c in range(1, w - 1)]
    if side == "left":
        return [(r, c) for r in range(1, h - 1) for c in range(3, w - 1)]
    return [(r, c) for r in range(1, h - 1) for c in range(1, w - 3)]


def _place_direct_pointer(
    rng: np.random.Generator,
    side: str,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
) -> dict[str, object] | None:
    direction = SIDE_TO_DIR[side]
    candidates = _direct_pointer_candidates(side, shape)
    order = rng.permutation(len(candidates))

    for idx in order:
        body = tuple(map(int, candidates[int(idx)]))
        if not _can_place_pointer(body, direction, shape, occupied, blocked):
            continue
        if not _ray_clear_to_border(body, direction, shape, occupied):
            continue
        arrow = (int(body[0] + direction[0]), int(body[1] + direction[1]))
        return {"name": f"direct_{side}", "body": body, "arrow": arrow, "direction": direction}
    return None


def _paint_pointers(grid: np.ndarray, pointers: list[dict[str, object]]) -> None:
    for pointer in pointers:
        br, bc = pointer["body"]
        ar, ac = pointer["arrow"]
        grid[int(br), int(bc)] = BODY_COLOR
        grid[int(ar), int(ac)] = ARROW_COLOR


def _direction_only_output(grid: np.ndarray, analysis: dict[str, object]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    side_colors = analysis["side_colors"]
    bg = int(analysis["bg"])
    for pointer in analysis["pointers"]:
        side = DIR_TO_SIDE[tuple(pointer["direction"])]
        br, bc = pointer["body"]
        ar, ac = pointer["arrow"]
        out[int(ar), int(ac)] = bg
        out[int(br), int(bc)] = int(side_colors[side])
    return out


def _resolve_body_only_color(
    pointers: list[dict[str, object]],
    index: int,
    body_lookup: dict[tuple[int, int], int],
    side_colors: dict[str, int],
    bg: int,
    grid: np.ndarray,
    memo: dict[int, int],
    visiting: set[int],
) -> int:
    if index in memo:
        return int(memo[index])
    if index in visiting:
        raise AssertionError("body-only shortcut should also stay acyclic on generated scenes")

    visiting.add(int(index))
    pointer = pointers[int(index)]
    dr, dc = pointer["direction"]
    br, bc = pointer["body"]
    r = int(br + 2 * dr)
    c = int(bc + 2 * dc)
    side = DIR_TO_SIDE[(int(dr), int(dc))]

    while _in_bounds(grid.shape, r, c):
        color = int(grid[r, c])
        if color == bg:
            r += int(dr)
            c += int(dc)
            continue

        coord = (int(r), int(c))
        if coord in body_lookup:
            result = _resolve_body_only_color(
                pointers,
                body_lookup[coord],
                body_lookup,
                side_colors,
                bg,
                grid,
                memo,
                visiting,
            )
            memo[index] = int(result)
            visiting.remove(int(index))
            return int(result)

        border_hit = (
            (side == "top" and r == 0)
            or (side == "bottom" and r == grid.shape[0] - 1)
            or (side == "left" and c == 0)
            or (side == "right" and c == grid.shape[1] - 1)
        )
        if border_hit:
            memo[index] = int(side_colors[side])
            visiting.remove(int(index))
            return int(side_colors[side])

        r += int(dr)
        c += int(dc)

    raise AssertionError("body-only shortcut should always reach a border on generated scenes")


def _body_only_output(grid: np.ndarray, analysis: dict[str, object]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    bg = int(analysis["bg"])
    side_colors = analysis["side_colors"]
    pointers = analysis["pointers"]
    body_lookup = {tuple(pointer["body"]): int(index) for index, pointer in enumerate(pointers)}
    memo: dict[int, int] = {}

    for index, pointer in enumerate(pointers):
        color = _resolve_body_only_color(
            pointers,
            index,
            body_lookup,
            side_colors,
            bg,
            np.asarray(grid, dtype=int),
            memo,
            set(),
        )
        br, bc = pointer["body"]
        ar, ac = pointer["arrow"]
        out[int(ar), int(ac)] = bg
        out[int(br), int(bc)] = int(color)
    return out


def _validate_generated_scene(grid: np.ndarray, analysis: dict[str, object]) -> None:
    pointers = analysis["pointers"]
    assert len(pointers) >= 5, "generator should create several pointers"

    direct = [pointer for pointer in pointers if pointer["target_kind"] == "edge"]
    indirect = [pointer for pointer in pointers if pointer["target_kind"] == "pointer"]
    assert len(direct) >= 3, "generator should include multiple direct edge hits"
    assert len(indirect) >= 2, "generator should include multiple inherited pointers"

    hit_parts = {pointer["hit_part"] for pointer in indirect}
    assert hit_parts == {"arrow", "body"}, "generated scenes must cover both body-hit and arrow-hit inheritance"

    depths = [int(pointer["dependency_depth"]) for pointer in pointers]
    assert max(depths) >= 2, "generator must include a dependency chain of length at least two"

    resolved_colors = {int(pointer["resolved_color"]) for pointer in pointers}
    assert len(resolved_colors) >= 2, "output should not collapse to a single pointer color"

    correct = _render_output(grid, analysis)
    assert not np.array_equal(correct, grid), "output must differ from input"
    assert not np.array_equal(correct, _direction_only_output(grid, analysis)), "inheritance must matter beyond raw direction"
    assert not np.array_equal(correct, _body_only_output(grid, analysis)), "arrow-cell hits must affect the result"


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(GRID_MIN_SIDE, GRID_MAX_SIDE + 1))
    w = int(rng.integers(GRID_MIN_SIDE, GRID_MAX_SIDE + 1))
    grid = np.full((h, w), BACKGROUND, dtype=int)

    side_palette = rng.choice(EDGE_COLOR_POOL, size=4, replace=False)
    side_colors = {side: int(color) for side, color in zip(SIDES, side_palette)}
    _paint_edges(grid, side_colors)

    motif = _build_motif(rng, grid.shape)
    if motif is None:
        return None
    pointers, occupied, blocked = motif

    root_side = DIR_TO_SIDE[tuple(next(pointer for pointer in pointers if pointer["name"] == "root")["direction"])]
    available_sides = [side for side in SIDES if side != root_side]
    rng.shuffle(available_sides)
    extra_count = int(rng.integers(2, 4))
    extra_sides = available_sides[:extra_count]

    for side in extra_sides:
        placed = _place_direct_pointer(rng, side, grid.shape, occupied, blocked)
        if placed is None:
            return None
        pointers.append(placed)
        occupied.add(placed["body"])
        occupied.add(placed["arrow"])
        blocked.update(_pointer_path_to_border(placed["body"], placed["direction"], grid.shape))

    _paint_pointers(grid, pointers)

    try:
        analysis = _analyze_scene(grid)
        _validate_generated_scene(grid, analysis)
    except AssertionError:
        return None

    if not np.array_equal(main(grid), _render_output(grid, analysis)):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid line-of-sight inheritance puzzle")
