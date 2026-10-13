import numpy as np


BG = 0
YELLOW = 4
PRISM = 6
TOWER_PALETTE = (1, 2, 3, 5, 7, 8, 9)
DIRS = ((-1, 0), (0, 1), (1, 0), (0, -1))

_RNG = np.random.default_rng()
_CATEGORY_CURSOR = 0


def _in_bounds(cell: tuple[int, int], height: int, width: int) -> bool:
    r, c = cell
    return 0 <= r < height and 0 <= c < width


def _add(cell: tuple[int, int], direction: tuple[int, int], steps: int = 1) -> tuple[int, int]:
    return (cell[0] + direction[0] * steps, cell[1] + direction[1] * steps)


def _cw(direction: tuple[int, int]) -> tuple[int, int]:
    dr, dc = direction
    return (dc, -dr)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    height, width = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for r in range(height):
        for c in range(width):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for dr, dc in DIRS:
                    nr = int(cr + dr)
                    nc = int(cc + dc)
                    if 0 <= nr < height and 0 <= nc < width and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _extract_towers(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    height, width = g.shape
    towers: list[dict] = []

    for color in sorted(int(c) for c in np.unique(g) if int(c) not in (BG, YELLOW, PRISM)):
        for comp in _components(g == int(color)):
            pts = np.array(comp, dtype=int)
            length = int(len(comp))
            assert 2 <= length <= 5

            rows = pts[:, 0]
            cols = pts[:, 1]
            candidates: list[tuple[tuple[int, int], tuple[int, int]]] = []

            if int(rows.min()) == int(rows.max()):
                row = int(rows[0])
                c0 = int(cols.min())
                c1 = int(cols.max())
                assert length == c1 - c0 + 1
                if c0 - 1 >= 0 and int(g[row, c0 - 1]) == YELLOW:
                    candidates.append(((row, c0 - 1), (0, -1)))
                if c1 + 1 < width and int(g[row, c1 + 1]) == YELLOW:
                    candidates.append(((row, c1 + 1), (0, 1)))
            elif int(cols.min()) == int(cols.max()):
                col = int(cols[0])
                r0 = int(rows.min())
                r1 = int(rows.max())
                assert length == r1 - r0 + 1
                if r0 - 1 >= 0 and int(g[r0 - 1, col]) == YELLOW:
                    candidates.append(((r0 - 1, col), (-1, 0)))
                if r1 + 1 < height and int(g[r1 + 1, col]) == YELLOW:
                    candidates.append(((r1 + 1, col), (1, 0)))
            else:
                raise AssertionError("tower body must be straight")

            assert len(candidates) == 1
            yellow, direction = candidates[0]
            towers.append(
                {
                    "color": int(color),
                    "length": int(length),
                    "yellow": (int(yellow[0]), int(yellow[1])),
                    "direction": (int(direction[0]), int(direction[1])),
                }
            )

    assert towers
    towers.sort(key=lambda tower: (tower["yellow"][0], tower["yellow"][1], tower["color"]))
    return towers


def _fire_beam(out: np.ndarray, yellow: tuple[int, int], direction: tuple[int, int], length: int, color: int) -> None:
    height, width = out.shape
    r, c = yellow
    dr, dc = direction
    remaining = int(length)

    while remaining > 0:
        nr = int(r + dr)
        nc = int(c + dc)
        if not (0 <= nr < height and 0 <= nc < width):
            break

        cell = int(out[nr, nc])
        if cell == PRISM:
            out[nr, nc] = int(color)
            r, c = nr, nc
            dr, dc = _cw((dr, dc))
            continue

        if cell != BG:
            break

        out[nr, nc] = int(color)
        r, c = nr, nc
        remaining -= 1


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fire colored rays from tower heads; prisms rotate the rays clockwise and do not consume range.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    out = g.copy()
    for tower in _extract_towers(g):
        _fire_beam(out, tower["yellow"], tower["direction"], tower["length"], tower["color"])
    return out


def _tower_back_cells(start: tuple[int, int], direction: tuple[int, int], length: int) -> tuple[tuple[int, int], list[tuple[int, int]]]:
    yellow = _add(start, direction, -1)
    body = [_add(start, direction, -(step + 2)) for step in range(length)]
    return yellow, body


def _steps_until_exit(start: tuple[int, int], direction: tuple[int, int], height: int, width: int) -> int:
    steps = 0
    cell = start
    while _in_bounds(cell, height, width):
        steps += 1
        cell = _add(cell, direction, 1)
    return steps


def _positive_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(int(v) for v in _RNG.choice(np.arange(1, total), size=parts - 1, replace=False))
    pieces: list[int] = []
    prev = 0
    for cut in cuts + [int(total)]:
        pieces.append(int(cut - prev))
        prev = int(cut)
    return pieces


def _input_cells_ok(
    cells: list[tuple[int, int]],
    height: int,
    width: int,
    occupied_input: set[tuple[int, int]],
    reserved_beam: set[tuple[int, int]],
) -> bool:
    cell_set = set(cells)
    if len(cell_set) != len(cells):
        return False
    for cell in cells:
        if not _in_bounds(cell, height, width):
            return False
        if cell in occupied_input or cell in reserved_beam:
            return False
        for dr, dc in DIRS:
            neighbor = (cell[0] + dr, cell[1] + dc)
            if neighbor in occupied_input:
                return False
    return True


def _beam_cells_ok(
    beam_cells: list[tuple[int, int]],
    prism_cells: list[tuple[int, int]],
    tower_cells: list[tuple[int, int]],
    height: int,
    width: int,
    occupied_input: set[tuple[int, int]],
    reserved_beam: set[tuple[int, int]],
) -> bool:
    beam_set = set(beam_cells)
    prism_set = set(prism_cells)
    if len(beam_set) != len(beam_cells):
        return False
    if not beam_set.isdisjoint(set(tower_cells)):
        return False
    for cell in beam_cells:
        if not _in_bounds(cell, height, width):
            return False
        if cell in reserved_beam:
            return False
        if cell in occupied_input and cell not in prism_set:
            return False
    return True


def _build_plan(
    category: str,
    height: int,
    width: int,
    occupied_input: set[tuple[int, int]],
    occupied_nonprism: set[tuple[int, int]],
    reserved_beam: set[tuple[int, int]],
) -> dict | None:
    attempts = 300 if category == "turn" else 220

    for _ in range(attempts):
        direction = DIRS[int(_RNG.integers(len(DIRS)))]
        length = int(_RNG.integers(2, 6))
        prism_cells: list[tuple[int, int]] = []
        beam_cells: list[tuple[int, int]] = []

        if category == "straight":
            start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
            beam_cells = [_add(start, direction, step) for step in range(length)]
        elif category == "edge":
            start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
            run = _steps_until_exit(start, direction, height, width)
            if not (1 <= run < length):
                continue
            beam_cells = [_add(start, direction, step) for step in range(run)]
        elif category == "turn":
            max_turns = min(2, length - 1)
            turns = int(_RNG.integers(1, max_turns + 1))
            segments = _positive_composition(length, turns + 1)
            start = (int(_RNG.integers(height)), int(_RNG.integers(width)))
            cursor = start
            cur_dir = direction
            for idx, seg_len in enumerate(segments):
                beam_cells.extend(_add(cursor, cur_dir, step) for step in range(seg_len))
                cursor = _add(cursor, cur_dir, seg_len)
                if idx + 1 < len(segments):
                    prism = cursor
                    prism_cells.append(prism)
                    beam_cells.append(prism)
                    cur_dir = _cw(cur_dir)
                    cursor = _add(prism, cur_dir, 1)
        elif category == "block":
            if not occupied_nonprism:
                continue
            blockers = list(occupied_nonprism)
            blocker = blockers[int(_RNG.integers(len(blockers)))]
            free_run = int(_RNG.integers(1, 5))
            if free_run >= length:
                length = free_run + 1
            if length > 5:
                continue
            start = _add(blocker, direction, -free_run)
            beam_cells = [_add(start, direction, step) for step in range(free_run)]
        else:
            raise AssertionError("unknown category")

        yellow, body = _tower_back_cells(start, direction, length)
        tower_cells = [yellow] + body
        input_cells = tower_cells + prism_cells

        if not _input_cells_ok(input_cells, height, width, occupied_input, reserved_beam):
            continue
        if not _beam_cells_ok(beam_cells, prism_cells, tower_cells, height, width, occupied_input, reserved_beam):
            continue

        return {
            "category": category,
            "direction": direction,
            "length": length,
            "yellow": yellow,
            "body": body,
            "prisms": prism_cells,
            "beam": beam_cells,
        }

    return None


def _choose_categories(n_towers: int) -> list[str]:
    global _CATEGORY_CURSOR
    base = ["straight", "turn", "edge", "block"]
    if n_towers == 4:
        offset = _CATEGORY_CURSOR % 4
        cats = base[offset:] + base[:offset]
    else:
        omit = _CATEGORY_CURSOR % 4
        cats = [cat for idx, cat in enumerate(base) if idx != omit]
    _CATEGORY_CURSOR += 1
    _RNG.shuffle(cats)
    return cats


def _place_extra_prisms(
    height: int,
    width: int,
    occupied_input: set[tuple[int, int]],
    reserved_beam: set[tuple[int, int]],
) -> list[tuple[int, int]]:
    extra: list[tuple[int, int]] = []
    target = int(_RNG.integers(0, 3))
    for _ in range(target):
        placed = False
        for _ in range(100):
            cell = (int(_RNG.integers(height)), int(_RNG.integers(width)))
            if not _input_cells_ok([cell], height, width, occupied_input, reserved_beam):
                continue
            occupied_input.add(cell)
            extra.append(cell)
            placed = True
            break
        if not placed:
            break
    return extra


def _gen_attempt() -> np.ndarray | None:
    height = int(_RNG.integers(14, 23))
    width = int(_RNG.integers(14, 23))
    n_towers = 4 if float(_RNG.random()) < 0.55 else 3
    categories = _choose_categories(n_towers)
    colors = [int(v) for v in _RNG.choice(np.array(TOWER_PALETTE), size=n_towers, replace=False)]

    occupied_input: set[tuple[int, int]] = set()
    occupied_nonprism: set[tuple[int, int]] = set()
    reserved_beam: set[tuple[int, int]] = set()
    plans: list[dict] = []

    indexed = list(enumerate(categories))
    indexed.sort(key=lambda item: item[1] == "block")

    for idx, category in indexed:
        plan = _build_plan(category, height, width, occupied_input, occupied_nonprism, reserved_beam)
        if plan is None:
            return None
        plan["color"] = colors[idx]
        plans.append(plan)

        tower_cells = [plan["yellow"]] + list(plan["body"])
        occupied_nonprism.update(tower_cells)
        occupied_input.update(tower_cells)
        occupied_input.update(plan["prisms"])
        reserved_beam.update(plan["beam"])

    extra_prisms = _place_extra_prisms(height, width, occupied_input, reserved_beam)

    grid = np.zeros((height, width), dtype=int)
    for plan in plans:
        for r, c in plan["body"]:
            grid[r, c] = int(plan["color"])
        yr, yc = plan["yellow"]
        grid[yr, yc] = YELLOW
        for r, c in plan["prisms"]:
            grid[r, c] = PRISM

    for r, c in extra_prisms:
        grid[r, c] = PRISM

    try:
        towers = _extract_towers(grid)
        if len(towers) != n_towers:
            return None
        out = main(grid)
    except AssertionError:
        return None

    if np.array_equal(out, grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(400):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid signal-tower sample")
