from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
TOP_COLOR = 1
RIGHT_COLOR = 2
BOTTOM_COLOR = 3
LEFT_COLOR = 4
MARKER_COLOR = 5
HINT_COLOR = 6
MAIN_COLOR = 7

SIDE_TO_COLOR = {
    "top": TOP_COLOR,
    "bottom": BOTTOM_COLOR,
    "left": LEFT_COLOR,
    "right": RIGHT_COLOR,
}
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
SIDES = tuple(SIDE_TO_COLOR)

GRID_ROWS = (13, 16)
GRID_COLS = (13, 16)
BLOB_AREA = (15, 25)
GEN_ATTEMPTS = 400


def _in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = shape
    return 0 <= r < h and 0 <= c < w


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
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if _in_bounds(mask.shape, nr, nc) and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _largest_component(mask: np.ndarray) -> set[tuple[int, int]]:
    comps = _components(mask)
    assert comps, "expected at least one component"
    comps.sort(key=len, reverse=True)
    if len(comps) > 1:
        assert len(comps[0]) > len(comps[1]), "largest component must be unique"
    return set(comps[0])


def _is_connected(cells: set[tuple[int, int]]) -> bool:
    if not cells:
        return False
    start = next(iter(cells))
    q = deque([start])
    seen = {start}
    while q:
        r, c = q.popleft()
        for nb in _neighbors4(r, c):
            if nb in cells and nb not in seen:
                seen.add(nb)
                q.append(nb)
    return len(seen) == len(cells)


def _frame_grid(h: int, w: int) -> np.ndarray:
    g = np.full((h, w), BG, dtype=int)
    g[0, 1:-1] = TOP_COLOR
    g[-1, 1:-1] = BOTTOM_COLOR
    g[1:-1, 0] = LEFT_COLOR
    g[1:-1, -1] = RIGHT_COLOR
    return g


def _side_color(side: str) -> int:
    return SIDE_TO_COLOR[side]


def _find_side_hints(g: np.ndarray) -> list[tuple[int, int, str]]:
    h, w = g.shape
    hints: list[tuple[int, int, str]] = []
    for c in range(1, w - 1):
        if int(g[0, c]) == HINT_COLOR:
            hints.append((0, c, "top"))
        if int(g[h - 1, c]) == HINT_COLOR:
            hints.append((h - 1, c, "bottom"))
    for r in range(1, h - 1):
        if int(g[r, 0]) == HINT_COLOR:
            hints.append((r, 0, "left"))
        if int(g[r, w - 1]) == HINT_COLOR:
            hints.append((r, w - 1, "right"))
    return hints


def _aligned_hints(g: np.ndarray, r: int, c: int) -> list[tuple[int, int, str]]:
    out = []
    for hr, hc, side in _find_side_hints(g):
        if hr == 0 or hr == g.shape[0] - 1:
            if hc == c:
                out.append((hr, hc, side))
        else:
            if hr == r:
                out.append((hr, hc, side))
    return out


def _marker_destination(r: int, c: int, side: str, steps: int, shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return max(1, r - steps), c
    if side == "bottom":
        return min(h - 2, r + steps), c
    if side == "left":
        return r, max(1, c - steps)
    return r, min(w - 2, c + steps)


def _orange_neighbors(g: np.ndarray, r: int, c: int) -> int:
    return sum(
        1
        for nr, nc in _neighbors4(r, c)
        if _in_bounds(g.shape, nr, nc) and int(g[nr, nc]) == MAIN_COLOR
    )


def _parse_roles(g: np.ndarray) -> tuple[set[tuple[int, int]], list[tuple[int, int]], list[tuple[int, int]], dict[tuple[int, int], str]]:
    main_cells = _largest_component(g == MAIN_COLOR)
    grey_cells = [tuple(map(int, rc)) for rc in np.argwhere(g == MARKER_COLOR)]
    embedded: list[tuple[int, int]] = []
    isolated: list[tuple[int, int]] = []
    directions: dict[tuple[int, int], str] = {}
    for r, c in grey_cells:
        if _orange_neighbors(g, r, c) >= 3:
            embedded.append((r, c))
        else:
            isolated.append((r, c))
            matches = _aligned_hints(g, r, c)
            assert len(matches) == 1, "isolated marker must match exactly one hint"
            directions[(r, c)] = matches[0][2]
    return main_cells, embedded, isolated, directions


def main(grid: np.ndarray) -> np.ndarray:
    """
    Repair embedded defect pixels, count them, and move isolated markers toward the frame side indicated by their aligned hint.

    Input structure:
    - The grid has a one-cell-thick rectangular frame on the outer border; the
      corners are black, and the top/bottom/left/right sides have fixed distinct
      colors except for a few magenta hint pixels replacing side pixels.
    - Inside the frame there is one irregular orange 4-connected main object and
      several single grey marker pixels.
    - Some grey pixels are embedded defects inside the orange object, while the
      remaining grey pixels are isolated in the black background and each is aligned
      with exactly one magenta hint on the frame.

    Rule:
    - Find the largest orange 4-connected component as the main object.
    - Classify each grey pixel: if it has at least three orange 4-neighbors, it is
      an embedded marker and is repaired to orange; otherwise it is an isolated marker.
    - Let N be the number of repaired embedded markers.
    - Move every isolated marker N cells toward the frame side containing its unique
      aligned magenta hint, clamping the endpoint to stay inside the frame interior.
    - Restore every magenta hint pixel back to the original color of its frame side.

    Key insight:
    The embedded grey pixels serve two roles at once: they are repaired defects, and
    their repair count is the hidden displacement distance used by the separate isolated markers.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h >= 5 and w >= 5

    out = g.copy()
    _main_cells, embedded, isolated, directions = _parse_roles(g)
    steps = len(embedded)
    assert 1 <= steps <= 3, "embedded repair count must encode the move distance"

    for r, c in embedded:
        out[r, c] = MAIN_COLOR

    for r, c in isolated:
        out[r, c] = BG
    destinations: list[tuple[int, int]] = []
    for r, c in isolated:
        nr, nc = _marker_destination(r, c, directions[(r, c)], steps, g.shape)
        destinations.append((nr, nc))
    assert len(destinations) == len(set(destinations)), "translated markers must not collide"
    for nr, nc in destinations:
        assert 1 <= nr < h - 1 and 1 <= nc < w - 1
        out[nr, nc] = MARKER_COLOR

    for hr, hc, side in _find_side_hints(g):
        out[hr, hc] = _side_color(side)

    return out


def _degree(blob: set[tuple[int, int]], cell: tuple[int, int]) -> int:
    r, c = cell
    return sum((nr, nc) in blob for nr, nc in _neighbors4(r, c))


def _grow_blob(h: int, w: int, area: int) -> set[tuple[int, int]] | None:
    allowed = {(r, c) for r in range(2, h - 2) for c in range(2, w - 2)}
    if len(allowed) < area:
        return None
    start = random.choice(tuple(allowed))
    blob = {start}
    frontier = {nb for nb in _neighbors4(*start) if nb in allowed}
    while len(blob) < area and frontier:
        choices = list(frontier)
        weights = [(_degree(blob, cell) + 0.35) ** 3 for cell in choices]
        cell = random.choices(choices, weights=weights, k=1)[0]
        frontier.remove(cell)
        blob.add(cell)
        for nb in _neighbors4(*cell):
            if nb in allowed and nb not in blob:
                frontier.add(nb)
    if len(blob) != area:
        return None
    rows = [r for r, _c in blob]
    cols = [c for _r, c in blob]
    if max(rows) - min(rows) + 1 < 4 or max(cols) - min(cols) + 1 < 4:
        return None
    if sum(_degree(blob, cell) >= 3 for cell in blob) < 3:
        return None
    return blob


def _choose_embedded(blob: set[tuple[int, int]], want: int) -> list[tuple[int, int]] | None:
    candidates = [cell for cell in blob if _degree(blob, cell) >= 3]
    random.shuffle(candidates)
    chosen: list[tuple[int, int]] = []
    removed: set[tuple[int, int]] = set()
    for cell in candidates:
        if any(nb in removed for nb in _neighbors4(*cell)):
            continue
        trial = removed | {cell}
        if _is_connected(blob - trial):
            chosen.append(cell)
            removed.add(cell)
        if len(chosen) == want:
            return chosen
    return None


def _is_free_bg(grid: np.ndarray, r: int, c: int, blob: set[tuple[int, int]]) -> bool:
    if int(grid[r, c]) != BG:
        return False
    return all((nr, nc) not in blob for nr, nc in _neighbors4(r, c))


def _candidate_positions(
    grid: np.ndarray,
    blob: set[tuple[int, int]],
    side: str,
    steps: int,
    force_clamp: bool,
    used_rows: set[int],
    used_cols: set[int],
    reserved_cells: set[tuple[int, int]],
) -> list[tuple[int, int, int, int]]:
    h, w = grid.shape
    candidates: list[tuple[int, int, int, int]] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            if r in used_rows or c in used_cols:
                continue
            if not _is_free_bg(grid, r, c, blob):
                continue
            if (r, c) in reserved_cells:
                continue
            nr, nc = _marker_destination(r, c, side, steps, grid.shape)
            if (nr, nc) == (r, c):
                continue
            if not _is_free_bg(grid, nr, nc, blob):
                continue
            if (nr, nc) in reserved_cells:
                continue
            if force_clamp:
                if side == "top" and not (1 < r <= steps):
                    continue
                if side == "bottom" and not (h - 2 - steps < r < h - 2):
                    continue
                if side == "left" and not (1 < c <= steps):
                    continue
                if side == "right" and not (w - 2 - steps < c < w - 2):
                    continue
            else:
                if side == "top" and r <= steps:
                    continue
                if side == "bottom" and r > h - 2 - steps:
                    continue
                if side == "left" and c <= steps:
                    continue
                if side == "right" and c > w - 2 - steps:
                    continue
            candidates.append((r, c, nr, nc))
    random.shuffle(candidates)
    return candidates


def _expected_output(
    inp: np.ndarray,
    embedded: list[tuple[int, int]],
    isolated_records: list[dict],
) -> np.ndarray:
    out = inp.copy()
    for r, c in embedded:
        out[r, c] = MAIN_COLOR
    for rec in isolated_records:
        sr, sc = rec["src"]
        dr, dc = rec["dst"]
        out[sr, sc] = BG
        out[dr, dc] = MARKER_COLOR
        hr, hc = rec["hint"]
        out[hr, hc] = _side_color(rec["side"])
    return out


def _validate_generated_case(
    inp: np.ndarray,
    blob: set[tuple[int, int]],
    embedded: list[tuple[int, int]],
    isolated_records: list[dict],
) -> bool:
    h, w = inp.shape
    if not (inp.ndim == 2 and h <= 30 and w <= 30):
        return False
    if {int(inp[0, 0]), int(inp[0, -1]), int(inp[-1, 0]), int(inp[-1, -1])} != {BG}:
        return False

    for c in range(1, w - 1):
        if int(inp[0, c]) not in {TOP_COLOR, HINT_COLOR}:
            return False
        if int(inp[-1, c]) not in {BOTTOM_COLOR, HINT_COLOR}:
            return False
    for r in range(1, h - 1):
        if int(inp[r, 0]) not in {LEFT_COLOR, HINT_COLOR}:
            return False
        if int(inp[r, -1]) not in {RIGHT_COLOR, HINT_COLOR}:
            return False

    orange = _largest_component(inp == MAIN_COLOR)
    if orange != (blob - set(embedded)):
        return False
    grey_positions = {tuple(map(int, rc)) for rc in np.argwhere(inp == MARKER_COLOR)}
    if grey_positions != {tuple(x) for x in embedded} | {tuple(rec["src"]) for rec in isolated_records}:
        return False

    parsed_main, parsed_embedded, parsed_isolated, directions = _parse_roles(inp)
    if parsed_main != orange:
        return False
    if Counter(parsed_embedded) != Counter(embedded):
        return False
    if len(parsed_isolated) != len(isolated_records):
        return False

    rec_by_src = {tuple(rec["src"]): rec for rec in isolated_records}
    for src in parsed_isolated:
        if src not in rec_by_src:
            return False
        rec = rec_by_src[src]
        if directions[src] != rec["side"]:
            return False
        if _marker_destination(src[0], src[1], rec["side"], len(embedded), inp.shape) != tuple(rec["dst"]):
            return False

    if len({tuple(rec["dst"]) for rec in isolated_records}) != len(isolated_records):
        return False

    expected = _expected_output(inp, embedded, isolated_records)
    solved = main(inp)
    if not np.array_equal(solved, expected):
        return False
    if np.any(solved == HINT_COLOR):
        return False
    if np.array_equal(solved, inp):
        return False

    changed = int(np.sum(solved != inp))
    if changed < len(embedded) + len(isolated_records):
        return False
    return True


def _build_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_ROWS)
    w = random.randint(*GRID_COLS)
    inp = _frame_grid(h, w)

    blob = _grow_blob(h, w, random.randint(*BLOB_AREA))
    if blob is None:
        return None
    for r, c in blob:
        inp[r, c] = MAIN_COLOR

    desired_embedded = random.choices([1, 2, 3], weights=[1, 3, 3], k=1)[0]
    embedded = _choose_embedded(blob, desired_embedded)
    if embedded is None:
        return None
    for r, c in embedded:
        inp[r, c] = MARKER_COLOR
    steps = len(embedded)

    isolated_count = random.choice([2, 3, 4])
    unique_side_count = random.randint(2, min(4, isolated_count))
    side_pool = random.sample(SIDES, unique_side_count)
    sides = side_pool + random.choices(side_pool, k=isolated_count - unique_side_count)
    random.shuffle(sides)

    clamp_index = random.randrange(isolated_count) if random.random() < 0.45 else None
    used_rows: set[int] = set()
    used_cols: set[int] = set()
    reserved_cells = set(blob) | set(embedded)
    isolated_records: list[dict] = []

    for idx, side in enumerate(sides):
        force_clamp = idx == clamp_index
        candidates = _candidate_positions(
            inp,
            blob,
            side,
            steps,
            force_clamp,
            used_rows,
            used_cols,
            reserved_cells,
        )
        if not candidates:
            return None
        r, c, nr, nc = candidates[0]
        used_rows.add(r)
        used_cols.add(c)
        reserved_cells.add((r, c))
        reserved_cells.add((nr, nc))
        inp[r, c] = MARKER_COLOR
        if side == "top":
            hint = (0, c)
        elif side == "bottom":
            hint = (h - 1, c)
        elif side == "left":
            hint = (r, 0)
        else:
            hint = (r, w - 1)
        inp[hint] = HINT_COLOR
        isolated_records.append({"src": (r, c), "dst": (nr, nc), "hint": hint, "side": side})

    if not _validate_generated_case(inp, blob, embedded, isolated_records):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
