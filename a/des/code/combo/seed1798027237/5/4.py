from __future__ import annotations

import numpy as np


BLACK = 0
BG = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
STEP_CHOICES = (6, 7)
SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))
DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
MAX_GENERATION_ATTEMPTS = 400


def _get_bg(grid: np.ndarray) -> int:
    values = np.asarray(grid, dtype=int).ravel()
    counts = np.bincount(values)
    return int(np.argmax(counts))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _pointer_direction(r0: int, c0: int, r1: int, c1: int, pr: int, pc: int) -> str:
    candidates: list[str] = []
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2

    if h % 2 == 1:
        if pc == c0 and pr == mid_r:
            candidates.append("left")
        if pc == c1 and pr == mid_r:
            candidates.append("right")
    if w % 2 == 1:
        if pr == r0 and pc == mid_c:
            candidates.append("up")
        if pr == r1 and pc == mid_c:
            candidates.append("down")

    assert len(candidates) == 1, "pointer must lie on exactly one edge midpoint"
    return candidates[0]


def _extract_objects(grid: np.ndarray) -> tuple[list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _get_bg(g)
    objects: list[dict[str, object]] = []

    for obj_id, coords in enumerate(_components(g != bg)):
        r0, c0 = map(int, coords.min(axis=0))
        r1, c1 = map(int, coords.max(axis=0))
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch != bg), "each object must be a solid rectangle"

        black_cells = np.argwhere(patch == BLACK)
        assert black_cells.shape == (1, 2), "each rectangle must contain one black pointer"
        colors = set(map(int, np.unique(patch)))
        colors.discard(BLACK)
        assert len(colors) == 1, "each rectangle must have exactly one body color"
        color = colors.pop()

        pr, pc = map(int, black_cells[0] + np.array([r0, c0]))
        direction = _pointer_direction(r0, c0, r1, c1, pr, pc)
        objects.append(
            {
                "id": obj_id,
                "bbox": (r0, c0, r1, c1),
                "color": color,
                "pointer": (pr, pc),
                "direction": direction,
            }
        )

    colors = [int(obj["color"]) for obj in objects]
    assert len(colors) == len(set(colors)), "input rectangle colors must be unique"
    return objects, bg


def _ray_distance(src: dict[str, object], dst: dict[str, object]) -> int | None:
    if src["id"] == dst["id"]:
        return None

    sr0, sc0, sr1, sc1 = src["bbox"]  # type: ignore[assignment]
    dr0, dc0, dr1, dc1 = dst["bbox"]  # type: ignore[assignment]
    pr, pc = src["pointer"]  # type: ignore[assignment]
    direction = src["direction"]

    if direction == "right" and dc0 > sc1 + 1 and dr0 <= pr <= dr1:
        return int(dc0 - sc1)
    if direction == "left" and dc1 < sc0 - 1 and dr0 <= pr <= dr1:
        return int(sc0 - dc1)
    if direction == "down" and dr0 > sr1 + 1 and dc0 <= pc <= dc1:
        return int(dr0 - sr1)
    if direction == "up" and dr1 < sr0 - 1 and dc0 <= pc <= dc1:
        return int(sr0 - dr1)
    return None


def _successors(objects: list[dict[str, object]]) -> dict[int, int | None]:
    succ: dict[int, int | None] = {}
    for src in objects:
        hits: list[tuple[int, int]] = []
        for dst in objects:
            dist = _ray_distance(src, dst)
            if dist is not None:
                hits.append((dist, int(dst["id"])))
        if not hits:
            succ[int(src["id"])] = None
            continue
        hits.sort()
        assert len(hits) == 1 or hits[0][0] < hits[1][0], "successor ray is ambiguous"
        succ[int(src["id"])] = hits[0][1]
    return succ


def _chains(objects: list[dict[str, object]], succ: dict[int, int | None]) -> list[list[int]]:
    indegree = {int(obj["id"]): 0 for obj in objects}
    for dst in succ.values():
        if dst is not None:
            indegree[dst] += 1
    assert all(v <= 1 for v in indegree.values()), "chains must not branch"

    starts = [obj_id for obj_id, deg in indegree.items() if deg == 0]
    assert starts, "expected at least one chain start"

    chains: list[list[int]] = []
    visited: set[int] = set()
    for start in starts:
        chain: list[int] = []
        cur = start
        local: set[int] = set()
        while cur is not None:
            assert cur not in local, "cycles are not allowed"
            assert cur not in visited, "chains must be disjoint"
            local.add(cur)
            visited.add(cur)
            chain.append(cur)
            cur = succ[cur]
        chains.append(chain)

    assert len(visited) == len(objects), "all objects must belong to chains"
    return chains


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer each rectangle's pointing direction from its black edge-midpoint pixel,
    follow the directed chains, and copy each chain head's color onto its tail.
    All black pointer pixels disappear into their rectangle's final color.
    """
    g = np.asarray(grid, dtype=int)
    objects, bg = _extract_objects(g)
    succ = _successors(objects)
    chains = _chains(objects, succ)
    by_id = {int(obj["id"]): obj for obj in objects}

    final_colors = {int(obj["id"]): int(obj["color"]) for obj in objects}
    for chain in chains:
        final_colors[chain[-1]] = int(by_id[chain[0]]["color"])

    out = np.full_like(g, bg)
    for obj in objects:
        r0, c0, r1, c1 = obj["bbox"]  # type: ignore[assignment]
        out[r0 : r1 + 1, c0 : c1 + 1] = final_colors[int(obj["id"])]
    return out


def _path_has_turn(path: list[tuple[int, int]]) -> bool:
    if len(path) < 3:
        return False
    dirs = {
        (path[i + 1][0] - path[i][0], path[i + 1][1] - path[i][1])
        for i in range(len(path) - 1)
    }
    return len(dirs) > 1


def _neighbors(pos: tuple[int, int], rows: int, cols: int) -> list[tuple[int, int]]:
    r, c = pos
    out: list[tuple[int, int]] = []
    for dr, dc in DIR_TO_DELTA.values():
        nr = r + dr
        nc = c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            out.append((nr, nc))
    return out


def _shuffled(seq: list[tuple[int, int]], rng: np.random.Generator) -> list[tuple[int, int]]:
    order = list(seq)
    rng.shuffle(order)
    return order


def _dfs_path(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    path: list[tuple[int, int]],
    target_len: int,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    if len(path) == target_len:
        return path.copy()

    current = path[-1]
    taken = occupied | set(path)
    candidates = [nxt for nxt in _neighbors(current, rows, cols) if nxt not in taken]
    for nxt in _shuffled(candidates, rng):
        path.append(nxt)
        result = _dfs_path(rows, cols, occupied, path, target_len, rng)
        if result is not None:
            return result
        path.pop()
    return None


def _sample_path(
    rows: int,
    cols: int,
    occupied: set[tuple[int, int]],
    length: int,
    want_turn: bool,
    rng: np.random.Generator,
) -> list[tuple[int, int]] | None:
    starts = [(r, c) for r in range(rows) for c in range(cols) if (r, c) not in occupied]
    for start in _shuffled(starts, rng):
        result = _dfs_path(rows, cols, occupied, [start], length, rng)
        if result is not None and (not want_turn or _path_has_turn(result)):
            return result
    return None


def _step_to_direction(a: tuple[int, int], b: tuple[int, int]) -> str:
    dr = b[0] - a[0]
    dc = b[1] - a[1]
    for direction, delta in DIR_TO_DELTA.items():
        if (dr, dc) == delta:
            return direction
    raise AssertionError("invalid path step")


def _terminal_directions(pos: tuple[int, int], occupied: set[tuple[int, int]]) -> list[str]:
    r, c = pos
    valids: list[str] = []
    if not any(rr < r and cc == c for rr, cc in occupied):
        valids.append("up")
    if not any(rr > r and cc == c for rr, cc in occupied):
        valids.append("down")
    if not any(rr == r and cc < c for rr, cc in occupied):
        valids.append("left")
    if not any(rr == r and cc > c for rr, cc in occupied):
        valids.append("right")
    return valids


def _pointer_cell(bbox: tuple[int, int, int, int], direction: str) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    mid_r = (r0 + r1) // 2
    mid_c = (c0 + c1) // 2
    if direction == "up":
        return r0, mid_c
    if direction == "down":
        return r1, mid_c
    if direction == "left":
        return mid_r, c0
    if direction == "right":
        return mid_r, c1
    raise AssertionError("unknown direction")


def _sample_chain_lengths(chain_count: int, rng: np.random.Generator) -> list[int]:
    min_total = 2 * chain_count
    max_total = min(len(PALETTE), 4 * chain_count)
    total = int(rng.integers(min_total, max_total + 1))
    lengths = [2] * chain_count
    extra = total - min_total
    while extra > 0:
        idx = int(rng.integers(0, chain_count))
        if lengths[idx] < 4:
            lengths[idx] += 1
            extra -= 1
    rng.shuffle(lengths)
    return lengths


def _generate_attempt(rng: np.random.Generator) -> np.ndarray | None:
    chain_count = int(rng.integers(1, 4))
    lengths = _sample_chain_lengths(chain_count, rng)
    lattice_rows = int(rng.integers(4, 6))
    lattice_cols = int(rng.integers(4, 6))
    occupied: set[tuple[int, int]] = set()
    paths: list[list[tuple[int, int]]] = []

    for length in lengths:
        want_turn = length >= 3 and bool(rng.integers(0, 2))
        path = _sample_path(lattice_rows, lattice_cols, occupied, length, want_turn, rng)
        if path is None:
            return None
        paths.append(path)
        occupied.update(path)

    specs: list[dict[str, object]] = []
    for path in paths:
        tail_dirs = _terminal_directions(path[-1], occupied)
        if not tail_dirs:
            return None
        tail_dir = tail_dirs[int(rng.integers(0, len(tail_dirs)))]
        for idx, pos in enumerate(path):
            direction = _step_to_direction(pos, path[idx + 1]) if idx + 1 < len(path) else tail_dir
            height, width = SHAPES[int(rng.integers(0, len(SHAPES)))]
            specs.append(
                {
                    "pos": pos,
                    "direction": direction,
                    "shape": (height, width),
                    "is_tail": idx + 1 == len(path),
                }
            )

    colors = list(map(int, rng.choice(PALETTE, size=len(specs), replace=False)))
    rng.shuffle(colors)
    for spec, color in zip(specs, colors):
        spec["color"] = color

    step = int(STEP_CHOICES[int(rng.integers(0, len(STEP_CHOICES)))])
    for spec in specs:
        lr, lc = spec["pos"]  # type: ignore[assignment]
        height, width = spec["shape"]  # type: ignore[assignment]
        center_r = lr * step
        center_c = lc * step
        half_h = height // 2
        half_w = width // 2
        spec["bbox0"] = (center_r - half_h, center_c - half_w, center_r + half_h, center_c + half_w)

    min_r = min(spec["bbox0"][0] for spec in specs)  # type: ignore[index]
    min_c = min(spec["bbox0"][1] for spec in specs)  # type: ignore[index]
    max_r = max(spec["bbox0"][2] for spec in specs)  # type: ignore[index]
    max_c = max(spec["bbox0"][3] for spec in specs)  # type: ignore[index]

    top = int(rng.integers(1, 4))
    bottom = int(rng.integers(1, 4))
    left = int(rng.integers(1, 4))
    right = int(rng.integers(1, 4))
    row_shift = top - min_r
    col_shift = left - min_c
    height = max_r - min_r + 1 + top + bottom
    width = max_c - min_c + 1 + left + right
    grid = np.full((height, width), BG, dtype=int)

    for spec in specs:
        r0, c0, r1, c1 = spec["bbox0"]  # type: ignore[assignment]
        bbox = (r0 + row_shift, c0 + col_shift, r1 + row_shift, c1 + col_shift)
        spec["bbox"] = bbox
        color = int(spec["color"])
        grid[bbox[0] : bbox[2] + 1, bbox[1] : bbox[3] + 1] = color
        pr, pc = _pointer_cell(bbox, str(spec["direction"]))
        grid[pr, pc] = BLACK

    expected_succ: dict[int, int | None] = {}
    idx = 0
    for path in paths:
        chain_specs = specs[idx : idx + len(path)]
        idx += len(path)
        for j, spec in enumerate(chain_specs):
            color = int(spec["color"])
            expected_succ[color] = int(chain_specs[j + 1]["color"]) if j + 1 < len(chain_specs) else None

    try:
        objects, _ = _extract_objects(grid)
        succ = _successors(objects)
        chains = _chains(objects, succ)
        actual_color_by_id = {int(obj["id"]): int(obj["color"]) for obj in objects}
        actual_succ = {
            int(obj["color"]): (actual_color_by_id[succ[int(obj["id"])]] if succ[int(obj["id"])] is not None else None)
            for obj in objects
        }
        if actual_succ != expected_succ:
            return None
        if any(len(chain) < 2 for chain in chains):
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.any(out == BLACK):
        return None
    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _generate_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
