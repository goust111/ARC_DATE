from __future__ import annotations

from collections import Counter

import numpy as np


NEIGHBORS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
RNG = np.random.default_rng()
MAX_GENERATION_ATTEMPTS = 300


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def colors_present(grid: np.ndarray, exclude: set[int] | None = None) -> list[int]:
    excluded = set() if exclude is None else set(exclude)
    return [int(color) for color in np.unique(grid) if int(color) not in excluded]


def components_of_color(grid: np.ndarray, color: int) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(grid[r, c]) != int(color):
                continue
            stack = [(int(r), int(c))]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((int(rr), int(cc)))
                for dr, dc in NEIGHBORS4:
                    nr, nc = int(rr + dr), int(cc + dc)
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == int(color):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            components.append(cells)
    return components


def normalize_shape(cells: list[tuple[int, int]] | set[tuple[int, int]]) -> tuple[tuple[int, int], ...]:
    rs = [r for r, _c in cells]
    cs = [c for _r, c in cells]
    r0, c0 = min(rs), min(cs)
    return tuple(sorted((int(r - r0), int(c - c0)) for r, c in cells))


def shape_dims(shape: tuple[tuple[int, int], ...]) -> tuple[int, int]:
    return int(max(r for r, _c in shape) + 1), int(max(c for _r, c in shape) + 1)


def count_tilings(
    shapes: list[tuple[tuple[int, int], ...]],
    rect_h: int,
    rect_w: int,
    limit: int = 2,
) -> int:
    shape_counts = Counter(tuple(shape) for shape in shapes)
    unique_shapes = list(shape_counts)
    placements: list[list[int]] = []
    for shape in unique_shapes:
        piece_h, piece_w = shape_dims(shape)
        if piece_h > rect_h or piece_w > rect_w:
            return 0
        masks: list[int] = []
        for dr in range(rect_h - piece_h + 1):
            for dc in range(rect_w - piece_w + 1):
                mask = 0
                for r, c in shape:
                    mask |= 1 << ((int(r) + dr) * rect_w + int(c) + dc)
                masks.append(mask)
        placements.append(masks)

    full_mask = (1 << (rect_h * rect_w)) - 1
    init_state = tuple(int(shape_counts[shape]) for shape in unique_shapes)
    memo: dict[tuple[int, tuple[int, ...]], int] = {}

    def dfs(used_mask: int, state: tuple[int, ...]) -> int:
        key = (int(used_mask), state)
        if key in memo:
            return int(memo[key])
        if used_mask == full_mask:
            return 1

        first_free = 0
        while (used_mask >> first_free) & 1:
            first_free += 1
        first_bit = 1 << first_free

        total = 0
        mutable_state = list(state)
        for shape_idx, remaining in enumerate(state):
            if remaining == 0:
                continue
            mutable_state[shape_idx] -= 1
            next_state = tuple(mutable_state)
            mutable_state[shape_idx] += 1
            for mask in placements[shape_idx]:
                if not (mask & first_bit):
                    continue
                if mask & used_mask:
                    continue
                total += dfs(used_mask | mask, next_state)
                if total >= limit:
                    memo[key] = int(limit)
                    return int(limit)
        memo[key] = int(total)
        return int(total)

    return int(dfs(0, init_state))


def valid_rectangle_dims(shapes: list[tuple[tuple[int, int], ...]]) -> list[tuple[int, int]]:
    area = sum(len(shape) for shape in shapes)
    dims: list[tuple[int, int]] = []
    for rect_h in range(1, area + 1):
        if area % rect_h != 0:
            continue
        rect_w = area // rect_h
        if count_tilings(shapes, int(rect_h), int(rect_w), limit=1) > 0:
            dims.append((int(rect_h), int(rect_w)))
    return dims


def infer_rectangle_dims(shapes: list[tuple[tuple[int, int], ...]]) -> tuple[int, int]:
    dims = valid_rectangle_dims(shapes)
    assert len(dims) == 1, f"ambiguous rectangle dimensions: {dims}"
    return dims[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Group fragments by color, reassemble each color into a solid rectangle, then stack by area.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)

    first_seen: dict[int, tuple[int, int]] = {}
    for r in range(g.shape[0]):
        for c in range(g.shape[1]):
            color = int(g[r, c])
            if color != bg and color not in first_seen:
                first_seen[color] = (int(r), int(c))

    groups: list[dict[str, object]] = []
    for color in colors_present(g, exclude={bg}):
        components = components_of_color(g, int(color))
        assert 2 <= len(components) <= 4
        shapes = [normalize_shape(component) for component in components]
        assert all(1 <= len(shape) <= 6 for shape in shapes)
        rect_h, rect_w = infer_rectangle_dims(shapes)
        groups.append(
            {
                "color": int(color),
                "h": int(rect_h),
                "w": int(rect_w),
                "area": int(rect_h * rect_w),
                "first": first_seen[int(color)],
            }
        )

    assert groups
    groups.sort(key=lambda item: (-int(item["area"]), -int(item["h"]), -int(item["w"]), item["first"][0], item["first"][1]))

    out_h = sum(int(group["h"]) for group in groups) + max(0, len(groups) - 1)
    out_w = max(int(group["w"]) for group in groups)
    out = np.full((out_h, out_w), bg, dtype=int)

    row = 0
    for idx, group in enumerate(groups):
        h = int(group["h"])
        w = int(group["w"])
        out[row : row + h, :w] = int(group["color"])
        row += h
        if idx != len(groups) - 1:
            row += 1
    return out


def partition_rectangle(rect_h: int, rect_w: int, piece_count: int, tries: int = 300) -> list[tuple[tuple[int, int], ...]] | None:
    cells = [(int(r), int(c)) for r in range(rect_h) for c in range(rect_w)]
    for _ in range(tries):
        seed_indices = RNG.choice(len(cells), size=piece_count, replace=False)
        pieces = [{cells[int(idx)]} for idx in seed_indices]
        owners = {cells[int(idx)]: piece_idx for piece_idx, idx in enumerate(seed_indices)}
        frontiers = [set() for _ in range(piece_count)]

        def add_frontier(cell: tuple[int, int], piece_idx: int) -> None:
            r, c = cell
            for dr, dc in NEIGHBORS4:
                nr, nc = int(r + dr), int(c + dc)
                candidate = (int(nr), int(nc))
                if 0 <= nr < rect_h and 0 <= nc < rect_w and candidate not in owners:
                    frontiers[piece_idx].add(candidate)

        for piece_idx, idx in enumerate(seed_indices):
            add_frontier(cells[int(idx)], int(piece_idx))

        ok = True
        while len(owners) < len(cells):
            choices = [piece_idx for piece_idx in range(piece_count) if len(pieces[piece_idx]) < 6 and frontiers[piece_idx]]
            if not choices:
                ok = False
                break
            weights = np.array([7 - len(pieces[piece_idx]) for piece_idx in choices], dtype=float)
            weights /= weights.sum()
            piece_idx = int(RNG.choice(choices, p=weights))
            frontier = list(frontiers[piece_idx])
            cell = frontier[int(RNG.integers(len(frontier)))]
            owners[cell] = int(piece_idx)
            pieces[piece_idx].add(cell)
            for other_frontier in frontiers:
                other_frontier.discard(cell)
            add_frontier(cell, int(piece_idx))

        if ok and all(1 <= len(piece) <= 6 for piece in pieces):
            return [normalize_shape(piece) for piece in pieces]
    return None


def build_group_spec(used_areas: set[int], tries_per_dim: int = 300) -> dict[str, object] | None:
    candidates: list[tuple[int, int, list[int]]] = []
    for rect_h in range(1, 7):
        for rect_w in range(1, 7):
            area = rect_h * rect_w
            if area < 4 or area > 18 or area in used_areas:
                continue
            piece_counts = [count for count in range(2, 5) if count <= area <= 6 * count]
            if piece_counts:
                candidates.append((int(rect_h), int(rect_w), piece_counts))
    RNG.shuffle(candidates)

    for rect_h, rect_w, piece_counts in candidates:
        for _ in range(tries_per_dim):
            piece_count = int(RNG.choice(piece_counts))
            shapes = partition_rectangle(int(rect_h), int(rect_w), int(piece_count))
            if shapes is None:
                continue
            if valid_rectangle_dims(shapes) == [(int(rect_h), int(rect_w))]:
                return {
                    "h": int(rect_h),
                    "w": int(rect_w),
                    "area": int(rect_h * rect_w),
                    "pieces": shapes,
                }
    return None


def can_place_fragment(shape: tuple[tuple[int, int], ...], top: int, left: int, occupied: np.ndarray) -> bool:
    h, w = occupied.shape
    for r, c in shape:
        rr, cc = int(top + r), int(left + c)
        if occupied[rr, cc]:
            return False
        if occupied[max(0, rr - 1) : min(h, rr + 2), max(0, cc - 1) : min(w, cc + 2)].any():
            return False
    return True


def scatter_fragments(groups: list[dict[str, object]], bg: int) -> np.ndarray | None:
    fragments: list[tuple[int, tuple[tuple[int, int], ...]]] = []
    for group in groups:
        color = int(group["color"])
        for shape in group["pieces"]:
            fragments.append((int(color), shape))

    fragments.sort(key=lambda item: (-(shape_dims(item[1])[0] * shape_dims(item[1])[1]), -len(item[1])))
    total_cells = sum(len(shape) for _color, shape in fragments)
    min_side = max(15, int(np.ceil(np.sqrt(total_cells * 7))))

    for _ in range(200):
        high = min(31, min_side + 8)
        if high <= min_side:
            high = min_side + 1
        grid_h = int(RNG.integers(min_side, high))
        grid_w = int(RNG.integers(min_side, high))
        grid = np.full((grid_h, grid_w), bg, dtype=int)
        occupied = np.zeros((grid_h, grid_w), dtype=bool)
        ok = True

        for color, shape in fragments:
            piece_h, piece_w = shape_dims(shape)
            row_margin = 1 if grid_h - piece_h >= 3 else 0
            col_margin = 1 if grid_w - piece_w >= 3 else 0
            row_start = row_margin
            row_end = grid_h - piece_h - row_margin
            col_start = col_margin
            col_end = grid_w - piece_w - col_margin
            positions = [(int(r), int(c)) for r in range(row_start, row_end + 1) for c in range(col_start, col_end + 1)]
            RNG.shuffle(positions)

            placed = False
            for top, left in positions:
                if not can_place_fragment(shape, int(top), int(left), occupied):
                    continue
                for r, c in shape:
                    rr, cc = int(top + r), int(left + c)
                    occupied[rr, cc] = True
                    grid[rr, cc] = int(color)
                placed = True
                break

            if not placed:
                ok = False
                break

        if ok and int((grid == bg).sum()) > total_cells:
            return grid
    return None


def generate_input() -> np.ndarray:
    all_colors = list(range(10))
    for _ in range(MAX_GENERATION_ATTEMPTS):
        bg = int(RNG.integers(10))
        palette = [color for color in all_colors if color != bg]
        RNG.shuffle(palette)

        group_count = int(RNG.integers(2, 5))
        used_areas: set[int] = set()
        groups: list[dict[str, object]] = []
        ok = True
        for group_idx in range(group_count):
            group = build_group_spec(used_areas)
            if group is None:
                ok = False
                break
            group["color"] = int(palette[group_idx])
            used_areas.add(int(group["area"]))
            groups.append(group)
        if not ok:
            continue

        grid = scatter_fragments(groups, int(bg))
        if grid is None:
            continue

        out = main(grid)
        assert isinstance(out, np.ndarray)
        return grid
    raise RuntimeError(f"failed to generate valid input after {MAX_GENERATION_ATTEMPTS} attempts")
