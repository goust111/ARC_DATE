from __future__ import annotations

import random
from collections import Counter, deque

import numpy as np


BLACK = 0
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5

GRID_SIZE = 20
FRAME_LEFT = 1
FRAME_RIGHT = 16
INTERIOR_LEFT = FRAME_LEFT + 1
INTERIOR_RIGHT = FRAME_RIGHT - 1
INTERIOR_WIDTH = INTERIOR_RIGHT - INTERIOR_LEFT + 1

TRIOMINO_COLORS = (RED, GREY)
MISSING_CORNERS = ((0, 0), (0, 1), (1, 0), (1, 1))

GLYPH_BASES = (
    np.array(
        [
            [1, 1, 1, 1, 1],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [1, 1, 1, 1, 0, 0],
            [1, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 0, 0, 1, 0],
            [1, 1, 1, 1, 1],
        ],
        dtype=int,
    ),
    np.array(
        [
            [1, 1, 1, 1],
            [1, 0, 0, 0],
            [1, 1, 1, 0],
            [1, 0, 1, 0],
            [1, 1, 1, 1],
        ],
        dtype=int,
    ),
)


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, allowed: set[int]) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color not in allowed or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": len(coords),
                }
            )
    return comps


def _detect_frame(grid: np.ndarray) -> tuple[int, int, int]:
    h, w = grid.shape
    candidates: list[tuple[int, int]] = []
    for c in range(w):
        color = int(grid[0, c])
        if color != BLACK and np.all(grid[:, c] == color):
            candidates.append((c, color))
    colors = Counter(color for _, color in candidates)
    frame_colors = [color for color, count in colors.items() if count == 2]
    assert len(frame_colors) == 1, "expected one frame color spanning exactly two full columns"
    frame_color = frame_colors[0]
    cols = sorted(c for c, color in candidates if color == frame_color)
    left, right = cols
    assert right - left >= 6, "frame interior must be non-trivial"
    return frame_color, left, right


def _source_cells(r0: int, c0: int, missing: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        (r0 + rr, c0 + cc)
        for rr in (0, 1)
        for cc in (0, 1)
        if (rr, cc) != missing
    )


def _missing_corner_direction(comp: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = comp["bbox"]
    assert comp["area"] == 3 and r1 - r0 == 1 and c1 - c0 == 1
    present = {(r - r0, c - c0) for r, c in comp["coords"]}
    missing = set(MISSING_CORNERS) - present
    assert len(missing) == 1
    mr, mc = next(iter(missing))
    return (-1 if mr == 0 else 1, -1 if mc == 0 else 1)


def _copy_offset(color: int, missing: tuple[int, int]) -> tuple[int, int]:
    mr, mc = missing
    dr = -1 if mr == 0 else 1
    dc = -1 if mc == 0 else 1
    if color == RED:
        dr, dc = -dr, -dc
    return dr, dc


def _copy_from_component(comp: dict) -> tuple[int, list[tuple[int, int]]] | None:
    dr, dc = _missing_corner_direction(comp)
    if int(comp["color"]) == RED:
        dr, dc = -dr, -dc
        dst_color = GREEN
    else:
        dst_color = YELLOW
    copied = [(r + dr, c + dc) for r, c in comp["coords"]]
    if all(0 <= r < GRID_SIZE and INTERIOR_LEFT <= c <= INTERIOR_RIGHT for r, c in copied):
        return dst_color, copied
    return None


def _endpoint_job(dst_color: int, copied: list[tuple[int, int]]) -> tuple[int, int]:
    counts = Counter(r for r, _ in copied)
    best_row = max(counts, key=lambda r: (counts[r], -abs(r - GRID_SIZE // 2), -r))
    row_cols = [c for r, c in copied if r == best_row]
    assert len(row_cols) >= 2
    return dst_color, best_row


def _glyph_mask_from_component(comp: dict) -> np.ndarray:
    r0, c0, r1, c1 = comp["bbox"]
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    for r, c in comp["coords"]:
        mask[r - r0, c - c0] = True
    return mask


def _glyph_cells(mask: np.ndarray, anchor_row: int, left: int, right: int) -> set[tuple[int, int]]:
    h, w = mask.shape
    c0 = left + 1 + (right - left - 1 - w) // 2
    r0 = anchor_row - h // 2
    cells: set[tuple[int, int]] = set()
    for r in range(h):
        for c in range(w):
            if mask[r, c]:
                rr = r0 + r
                cc = c0 + c
                assert 0 <= rr < GRID_SIZE
                assert left < cc < right
                cells.add((rr, cc))
    return cells


def _reconstruct_jobs(
    glyph_mask: np.ndarray,
    jobs: list[tuple[int, int]],
    left: int,
    right: int,
) -> dict[tuple[int, int], int]:
    painted: dict[tuple[int, int], int] = {}
    for color, anchor_row in jobs:
        mask = glyph_mask if color == GREEN else np.fliplr(glyph_mask)
        for cell in _glyph_cells(mask, anchor_row, left, right):
            prev = painted.get(cell)
            assert prev is None or prev == color, "conflicting reconstructed glyphs"
            painted[cell] = color
    return painted


def _parse_scene(grid: np.ndarray) -> tuple[int, int, int, np.ndarray, list[tuple[int, int]], list[dict]]:
    frame_color, left, right = _detect_frame(grid)
    allowed = set(int(v) for v in np.unique(grid) if int(v) not in (BLACK, frame_color))
    comps = _find_components(grid, allowed)

    triominoes = [comp for comp in comps if int(comp["color"]) in TRIOMINO_COLORS]
    glyphs = [comp for comp in comps if int(comp["color"]) not in TRIOMINO_COLORS]

    assert glyphs, "expected one complete glyph"
    assert len(glyphs) == 1, "expected exactly one complete glyph"
    glyph = glyphs[0]
    assert glyph["bbox"][1] > left and glyph["bbox"][3] < right
    glyph_mask = _glyph_mask_from_component(glyph)
    assert glyph_mask.shape[0] % 2 == 1, "glyph height must be odd"

    jobs: set[tuple[int, int]] = set()
    for comp in triominoes:
        copied = _copy_from_component(comp)
        if copied is None:
            continue
        jobs.add(_endpoint_job(*copied))

    return frame_color, left, right, glyph_mask, sorted(jobs), triominoes


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rebuild centered glyphs from diagonal L-triomino copies that survive inside the frame.

    Input structure:
    - The grid is 20x20 with black background and two same-color full-height vertical frame bars.
    - Inside the frame there is one complete single-color glyph plus 2-3 red/grey L triominoes, and some triominoes may sit outside the frame.
    - Each triomino occupies three cells of a 2x2 box, so its missing corner is well defined.

    Rule:
    - For every grey triomino, translate a yellow copy one diagonal step toward its missing corner; for every red triomino, translate a green copy one diagonal step away from its missing corner.
    - Discard any translated copy that touches the frame or lands outside the frame interior, then keep only the anchor row given by the copy's densest row.
    - Erase the original glyph and all triominoes, preserve the frame bars, and reconstruct a centered glyph on every anchor row: green uses the original glyph mask and yellow uses its horizontal mirror.

    Key insight:
    - The L triominoes are only intermediate encoders: their missing-corner direction and source color jointly decide which anchored glyph version must be redrawn after the scene is cleared.
    """
    g = np.asarray(grid, dtype=int)
    assert g.shape == (GRID_SIZE, GRID_SIZE)

    frame_color, left, right, glyph_mask, jobs, _triominoes = _parse_scene(g)
    painted = _reconstruct_jobs(glyph_mask, jobs, left, right)

    out = np.zeros_like(g)
    out[:, left] = frame_color
    out[:, right] = frame_color
    for (r, c), color in painted.items():
        out[r, c] = color
    return out


def _expand_with_margin(cells: set[tuple[int, int]], radius: int = 1) -> set[tuple[int, int]]:
    expanded: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                nr, nc = r + dr, c + dc
                if 0 <= nr < GRID_SIZE and 0 <= nc < GRID_SIZE:
                    expanded.add((nr, nc))
    return expanded


def _random_glyph_mask(rng: random.Random) -> np.ndarray:
    mask = np.array(rng.choice(GLYPH_BASES), copy=True)
    if rng.random() < 0.5:
        mask = np.flipud(mask)
    if rng.random() < 0.5:
        mask = np.fliplr(mask)
    assert not np.array_equal(mask, np.fliplr(mask))
    return mask.astype(bool)


def _valid_anchor_rows(mask: np.ndarray) -> list[int]:
    half = mask.shape[0] // 2
    return list(range(half, GRID_SIZE - half))


def _triomino_candidate(
    color: int,
    missing: tuple[int, int],
    r0: int,
    c0: int,
    glyph_cells: set[tuple[int, int]],
) -> dict | None:
    src = set(_source_cells(r0, c0, missing))
    if any(c in (FRAME_LEFT, FRAME_RIGHT) for _, c in src):
        return None
    if src & glyph_cells:
        return None

    dr, dc = _copy_offset(color, missing)
    copied = {(r + dr, c + dc) for r, c in src}
    fully_inside = all(0 <= r < GRID_SIZE and INTERIOR_LEFT <= c <= INTERIOR_RIGHT for r, c in copied)

    row_counts = Counter(r for r, _ in copied if 0 <= r < GRID_SIZE)
    anchor_row = None
    if fully_inside:
        anchor_row = _endpoint_job(GREEN if color == RED else YELLOW, list(copied))[1]

    return {
        "color": color,
        "missing": missing,
        "src": src,
        "copied": copied,
        "valid": fully_inside,
        "anchor_row": anchor_row,
        "dst_color": GREEN if color == RED else YELLOW,
    }


def _candidate_output_cells(glyph_mask: np.ndarray, cand: dict) -> set[tuple[int, int]] | None:
    if not cand["valid"]:
        return None
    mask = glyph_mask if cand["dst_color"] == GREEN else np.fliplr(glyph_mask)
    half = mask.shape[0] // 2
    row = cand["anchor_row"]
    if row is None or row - half < 0 or row + half >= GRID_SIZE:
        return None
    return _glyph_cells(mask, row, FRAME_LEFT, FRAME_RIGHT)


def _render_input(
    frame_color: int,
    glyph_color: int,
    glyph_cells: set[tuple[int, int]],
    objects: list[dict],
) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[:, FRAME_LEFT] = frame_color
    grid[:, FRAME_RIGHT] = frame_color
    for r, c in glyph_cells:
        grid[r, c] = glyph_color
    for obj in objects:
        for r, c in obj["src"]:
            grid[r, c] = obj["color"]
    return grid


def _expected_output(frame_color: int, glyph_mask: np.ndarray, jobs: list[tuple[int, int]]) -> np.ndarray:
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=int)
    grid[:, FRAME_LEFT] = frame_color
    grid[:, FRAME_RIGHT] = frame_color
    for (r, c), color in _reconstruct_jobs(glyph_mask, jobs, FRAME_LEFT, FRAME_RIGHT).items():
        grid[r, c] = color
    return grid


def _pick_valid_objects(
    rng: random.Random,
    glyph_mask: np.ndarray,
    glyph_cells: set[tuple[int, int]],
    total_objects: int,
) -> list[dict] | None:
    all_candidates: list[dict] = []
    for color in TRIOMINO_COLORS:
        for missing in MISSING_CORNERS:
            for r0 in range(GRID_SIZE - 1):
                for c0 in range(GRID_SIZE - 1):
                    cand = _triomino_candidate(color, missing, r0, c0, glyph_cells)
                    if cand is None or not cand["valid"]:
                        continue
                    output_cells = _candidate_output_cells(glyph_mask, cand)
                    if output_cells is None:
                        continue
                    cand["output_cells"] = output_cells
                    all_candidates.append(cand)

    if not all_candidates:
        return None

    rng.shuffle(all_candidates)
    desired_valid = 2 if total_objects == 3 or rng.random() < 0.55 else 1
    desired_valid = min(desired_valid, total_objects)

    preferred_colors = [rng.choice(TRIOMINO_COLORS)]
    if desired_valid >= 2:
        preferred_colors = [RED, GREY] if rng.random() < 0.75 else [rng.choice(TRIOMINO_COLORS)] * desired_valid

    selected: list[dict] = []
    occupied = _expand_with_margin(glyph_cells)
    output_used: set[tuple[int, int]] = set()
    jobs_seen: set[tuple[int, int]] = set()

    def can_add(cand: dict) -> bool:
        if cand["src"] & occupied:
            return False
        job = (cand["dst_color"], cand["anchor_row"])
        if job in jobs_seen:
            return False
        if cand["output_cells"] & output_used:
            return False
        return True

    for want_color in preferred_colors:
        options = [cand for cand in all_candidates if cand["color"] == want_color and can_add(cand)]
        if not options:
            return None
        cand = rng.choice(options)
        selected.append(cand)
        occupied |= _expand_with_margin(cand["src"])
        output_used |= cand["output_cells"]
        jobs_seen.add((cand["dst_color"], cand["anchor_row"]))

    while len(selected) < desired_valid:
        options = [cand for cand in all_candidates if can_add(cand)]
        if not options:
            return None
        cand = rng.choice(options)
        selected.append(cand)
        occupied |= _expand_with_margin(cand["src"])
        output_used |= cand["output_cells"]
        jobs_seen.add((cand["dst_color"], cand["anchor_row"]))

    return selected


def _pick_invalid_objects(
    rng: random.Random,
    glyph_cells: set[tuple[int, int]],
    selected: list[dict],
    total_objects: int,
) -> list[dict] | None:
    need = total_objects - len(selected)
    if need <= 0:
        return []

    invalid_candidates: list[dict] = []
    for color in TRIOMINO_COLORS:
        for missing in MISSING_CORNERS:
            for r0 in range(GRID_SIZE - 1):
                for c0 in range(GRID_SIZE - 1):
                    cand = _triomino_candidate(color, missing, r0, c0, glyph_cells)
                    if cand is None or cand["valid"]:
                        continue
                    invalid_candidates.append(cand)

    occupied = _expand_with_margin(glyph_cells)
    for cand in selected:
        occupied |= _expand_with_margin(cand["src"])

    chosen: list[dict] = []
    rng.shuffle(invalid_candidates)
    for cand in invalid_candidates:
        if cand["src"] & occupied:
            continue
        chosen.append(cand)
        occupied |= _expand_with_margin(cand["src"])
        if len(chosen) == need:
            break

    return chosen if len(chosen) == need else None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    glyph_mask = _random_glyph_mask(rng)
    glyph_color_choices = [1, 6, 7, 8, 9]
    frame_color_choices = [1, 6, 7, 8, 9]

    glyph_color = rng.choice(glyph_color_choices)
    frame_color = rng.choice([c for c in frame_color_choices if c != glyph_color])

    anchor_row = rng.choice(_valid_anchor_rows(glyph_mask))
    glyph_cells = _glyph_cells(glyph_mask, anchor_row, FRAME_LEFT, FRAME_RIGHT)

    total_objects = rng.choice((2, 3))
    valid_objects = _pick_valid_objects(rng, glyph_mask, glyph_cells, total_objects)
    if valid_objects is None:
        return None
    invalid_objects = _pick_invalid_objects(rng, glyph_cells, valid_objects, total_objects)
    if invalid_objects is None:
        return None

    objects = valid_objects + invalid_objects
    rng.shuffle(objects)
    inp = _render_input(frame_color, glyph_color, glyph_cells, objects)

    jobs = sorted({(obj["dst_color"], obj["anchor_row"]) for obj in valid_objects})
    expected = _expected_output(frame_color, glyph_mask, jobs)
    try:
        actual = main(inp)
    except Exception:
        return None

    if not np.array_equal(actual, expected):
        return None

    try:
        parsed_frame, left, right, parsed_mask, parsed_jobs, parsed_triominoes = _parse_scene(inp)
    except Exception:
        return None

    if parsed_frame != frame_color or left != FRAME_LEFT or right != FRAME_RIGHT:
        return None
    if not np.array_equal(parsed_mask, glyph_mask):
        return None
    if parsed_jobs != jobs:
        return None
    if len(parsed_triominoes) != total_objects:
        return None
    if len({job[0] for job in jobs}) == 1 and total_objects == 3 and rng.random() < 0.35:
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = random.Random(random.randrange(1 << 30))
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
