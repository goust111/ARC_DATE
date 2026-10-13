import random
import numpy as np


BG = 0
GREY = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_RETRIES = 500


def _neighbors4(r: int, c: int, h: int, w: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < h:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < w:
        yield r, c + 1


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc, h, w):
                    if mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _border_cells(top: int, left: int, bottom: int, right: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for c in range(left, right + 1):
        cells.append((top, c))
        if bottom != top:
            cells.append((bottom, c))
    for r in range(top + 1, bottom):
        cells.append((r, left))
        if right != left:
            cells.append((r, right))
    return cells


def _contains_point(rect: dict, pt: tuple[int, int]) -> bool:
    r, c = pt
    return rect["top"] < r < rect["bottom"] and rect["left"] < c < rect["right"]


def _contains_rect(outer: dict, inner: dict) -> bool:
    return (
        outer["top"] < inner["top"]
        and outer["left"] < inner["left"]
        and outer["bottom"] > inner["bottom"]
        and outer["right"] > inner["right"]
    )


def _find_legend(grid: np.ndarray) -> tuple[dict[int, int], set[tuple[int, int]]]:
    h, w = grid.shape
    candidates: list[tuple[int, np.ndarray]] = []
    for c in range(w - 1):
        left_non_bg = grid[:, c] != BG
        right_non_bg = grid[:, c + 1] != BG
        paired = left_non_bg & right_non_bg
        if not paired.any():
            continue
        if not np.array_equal(left_non_bg, paired):
            continue
        if not np.array_equal(right_non_bg, paired):
            continue
        rows = np.flatnonzero(paired)
        if np.any(grid[rows, c] == GREY) or np.any(grid[rows, c + 1] == GREY):
            continue
        candidates.append((c, rows))

    if not candidates:
        return {}, set()

    col, rows = max(candidates, key=lambda item: item[0])
    mapping: dict[int, int] = {}
    cells: set[tuple[int, int]] = set()
    for r in rows:
        rr = int(r)
        mapping[int(grid[rr, col])] = int(grid[rr, col + 1])
        cells.add((rr, col))
        cells.add((rr, col + 1))
    return mapping, cells


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    rects: list[dict] = []
    for comp in _components(grid == GREY):
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        top, bottom = min(rows), max(rows)
        left, right = min(cols), max(cols)
        if bottom - top < 2 or right - left < 2:
            continue
        border = set(_border_cells(top, left, bottom, right))
        if border != set(comp):
            continue
        rects.append(
            {
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "border": list(border),
                "area": (bottom - top + 1) * (right - left + 1),
            }
        )
    for idx, rect in enumerate(rects):
        rect["id"] = idx
        rect["depth"] = sum(_contains_rect(other, rect) for other in rects)
    return rects


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangular border using the seed-to-color legend on the right.

    Nested rectangles are handled from inner to outer. Each seed belongs to the
    smallest rectangle that contains it, so outer rectangles ignore seeds that are
    inside nested descendants. All seeds and the legend are erased in the output.
    """
    g = np.asarray(grid, dtype=int)
    legend_map, legend_cells = _find_legend(g)
    rects = _find_rectangles(g)

    seed_cells: list[tuple[int, int]] = []
    seed_for_rect: dict[int, int] = {}
    color_mask = (g != BG) & (g != GREY)
    for r, c in np.argwhere(color_mask):
        pt = (int(r), int(c))
        if pt in legend_cells:
            continue
        seed_cells.append(pt)
        containers = [rect for rect in rects if _contains_point(rect, pt)]
        if not containers:
            continue
        owner = min(containers, key=lambda rect: rect["area"])
        seed_for_rect[owner["id"]] = int(g[pt])

    out = g.copy()
    for rect in sorted(rects, key=lambda rect: (-rect["depth"], rect["area"])):
        seed_color = seed_for_rect.get(rect["id"])
        fill_color = legend_map.get(seed_color)
        if fill_color is None:
            continue
        for r, c in rect["border"]:
            if out[r, c] == GREY:
                out[r, c] = fill_color

    for r, c in seed_cells:
        out[r, c] = BG
    for r, c in legend_cells:
        out[r, c] = BG
    return out


def _min_side(depth: int) -> int:
    return 3 + 4 * (depth - 1)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], pad: int) -> bool:
    at, al, ab, ar = a
    bt, bl, bb, br = b
    return not (ab + pad < bt or bb + pad < at or ar + pad < bl or br + pad < al)


def _sample_structure(rng: random.Random) -> tuple[int, ...]:
    if rng.random() < 0.5:
        return rng.choice(((1, 1), (1, 1, 1)))
    return rng.choice(((2, 1), (2, 2), (3,)))


def _sample_root_sizes(rng: random.Random, structure: tuple[int, ...]) -> list[tuple[int, int]]:
    sizes: list[tuple[int, int]] = []
    for depth in structure:
        base = _min_side(depth)
        slack = 3 if depth >= 2 else 4
        height = base + rng.randint(0, slack)
        width = base + rng.randint(0, slack)
        sizes.append((height, width))
    return sizes


def _sample_grid_shape(
    rng: random.Random, root_sizes: list[tuple[int, int]]
) -> tuple[int, int] | None:
    min_h = max(16, max(height for height, _ in root_sizes) + 2)
    min_w = max(18, max(width for _, width in root_sizes) + 5)
    if min_h > 24 or min_w > 26:
        return None

    area_need = sum((height + 3) * (width + 3) for height, width in root_sizes)
    for _ in range(80):
        h = rng.randint(min_h, 24)
        w = rng.randint(min_w, 26)
        if h * (w - 3) >= area_need:
            return h, w
    return None


def _place_root_boxes(
    rng: random.Random, h: int, w: int, root_sizes: list[tuple[int, int]]
) -> list[tuple[int, int, int, int]] | None:
    field_w = w - 3
    ordered = sorted(
        enumerate(root_sizes), key=lambda item: item[1][0] * item[1][1], reverse=True
    )
    placed: list[tuple[int, int, int, int]] = []
    result: list[tuple[int, int, int, int] | None] = [None] * len(root_sizes)
    for idx, (height, width) in ordered:
        if height > h - 2 or width > field_w - 2:
            return None
        for _ in range(250):
            top = rng.randint(1, h - height - 1)
            left = rng.randint(1, field_w - width - 1)
            box = (top, left, top + height - 1, left + width - 1)
            if all(not _boxes_too_close(box, other, 2) for other in placed):
                placed.append(box)
                result[idx] = box
                break
        else:
            return None
    return [box for box in result if box is not None]


def _build_chain_boxes(
    rng: random.Random, root_box: tuple[int, int, int, int], depth: int
) -> list[tuple[int, int, int, int]] | None:
    chain = [root_box]
    current = root_box
    for subtree_depth in range(depth - 1, 0, -1):
        top, left, bottom, right = current
        cur_h = bottom - top + 1
        cur_w = right - left + 1
        min_child_h = _min_side(subtree_depth)
        min_child_w = _min_side(subtree_depth)
        max_child_h = cur_h - 4
        max_child_w = cur_w - 4
        if max_child_h < min_child_h or max_child_w < min_child_w:
            return None

        child_h = rng.randint(min_child_h, max_child_h)
        child_w = rng.randint(min_child_w, max_child_w)
        child_top = rng.randint(top + 2, bottom - child_h - 1)
        child_left = rng.randint(left + 2, right - child_w - 1)
        child = (
            child_top,
            child_left,
            child_top + child_h - 1,
            child_left + child_w - 1,
        )
        chain.append(child)
        current = child
    return chain


def _flatten_chains(
    chains: list[list[tuple[int, int, int, int]]], colors: list[tuple[int, int]]
) -> list[dict]:
    rects: list[dict] = []
    color_iter = iter(colors)
    for chain in chains:
        prev_id: int | None = None
        for depth, box in enumerate(chain):
            seed_color, fill_color = next(color_iter)
            top, left, bottom, right = box
            rect = {
                "id": len(rects),
                "top": top,
                "left": left,
                "bottom": bottom,
                "right": right,
                "depth": depth,
                "child": None,
                "seed_color": seed_color,
                "fill_color": fill_color,
                "border": _border_cells(top, left, bottom, right),
            }
            rects.append(rect)
            if prev_id is not None:
                rects[prev_id]["child"] = rect["id"]
            prev_id = rect["id"]
    return rects


def _direct_cells(rect: dict, child: dict | None) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for r in range(rect["top"] + 1, rect["bottom"]):
        for c in range(rect["left"] + 1, rect["right"]):
            if child is not None and (
                child["top"] <= r <= child["bottom"] and child["left"] <= c <= child["right"]
            ):
                continue
            cells.append((r, c))
    return cells


def _touches_existing_seed(pt: tuple[int, int], used: list[tuple[int, int]]) -> bool:
    r, c = pt
    return any(max(abs(r - ur), abs(c - uc)) <= 1 for ur, uc in used)


def _place_seeds(rng: random.Random, grid: np.ndarray, rects: list[dict]) -> bool:
    used: list[tuple[int, int]] = []
    for rect in sorted(rects, key=lambda item: -item["depth"]):
        child = rects[rect["child"]] if rect["child"] is not None else None
        candidates = _direct_cells(rect, child)
        rng.shuffle(candidates)
        chosen = None
        for pt in candidates:
            if not _touches_existing_seed(pt, used):
                chosen = pt
                break
        if chosen is None:
            return False
        rect["seed_pos"] = chosen
        used.append(chosen)
        grid[chosen] = rect["seed_color"]
    return True


def _place_legend(rng: random.Random, grid: np.ndarray, rects: list[dict]) -> set[tuple[int, int]]:
    h, w = grid.shape
    start = rng.randint(0, h - len(rects))
    pairs = [(rect["seed_color"], rect["fill_color"]) for rect in rects]
    rng.shuffle(pairs)
    legend_cells: set[tuple[int, int]] = set()
    for offset, (seed_color, fill_color) in enumerate(pairs):
        r = start + offset
        grid[r, w - 2] = seed_color
        grid[r, w - 1] = fill_color
        legend_cells.add((r, w - 2))
        legend_cells.add((r, w - 1))
    return legend_cells


def _expected_output(
    grid: np.ndarray, rects: list[dict], legend_cells: set[tuple[int, int]]
) -> np.ndarray:
    out = grid.copy()
    for rect in sorted(rects, key=lambda item: -item["depth"]):
        for r, c in rect["border"]:
            if out[r, c] == GREY:
                out[r, c] = rect["fill_color"]
    for rect in rects:
        out[rect["seed_pos"]] = BG
    for r, c in legend_cells:
        out[r, c] = BG
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    structure = _sample_structure(rng)
    root_sizes = _sample_root_sizes(rng, structure)
    shape = _sample_grid_shape(rng, root_sizes)
    if shape is None:
        return None
    h, w = shape

    roots = _place_root_boxes(rng, h, w, root_sizes)
    if roots is None or len(roots) != len(structure):
        return None

    chains: list[list[tuple[int, int, int, int]]] = []
    for root_box, depth in zip(roots, structure):
        chain = _build_chain_boxes(rng, root_box, depth)
        if chain is None:
            return None
        chains.append(chain)

    total_rects = sum(structure)
    colors = list(COLOR_POOL)
    rng.shuffle(colors)
    pairs = list(zip(colors[:total_rects], colors[total_rects : 2 * total_rects]))
    rects = _flatten_chains(chains, pairs)

    grid = np.zeros((h, w), dtype=int)
    for rect in rects:
        for r, c in rect["border"]:
            grid[r, c] = GREY

    if not _place_seeds(rng, grid, rects):
        return None
    legend_cells = _place_legend(rng, grid, rects)

    if len(_find_rectangles(grid)) != len(rects):
        return None
    legend_map, parsed_legend_cells = _find_legend(grid)
    if len(legend_map) != len(rects) or parsed_legend_cells != legend_cells:
        return None

    solver_out = main(grid)
    expected = _expected_output(grid, rects, legend_cells)
    if not np.array_equal(solver_out, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
