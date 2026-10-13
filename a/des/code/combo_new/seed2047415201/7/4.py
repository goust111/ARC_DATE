from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
DIR4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
DIR8 = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)


def _sign(value: int) -> int:
    if value < 0:
        return -1
    if value > 0:
        return 1
    return 0


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []

    for sr, sc in np.argwhere(mask):
        sr, sc = int(sr), int(sc)
        if seen[sr, sc]:
            continue
        q: deque[tuple[int, int]] = deque([(sr, sc)])
        seen[sr, sc] = True
        comp: list[tuple[int, int]] = []
        while q:
            r, c = q.popleft()
            comp.append((r, c))
            for dr, dc in DIR4:
                rr, cc = r + dr, c + dc
                if 0 <= rr < h and 0 <= cc < w and bool(mask[rr, cc]) and not seen[rr, cc]:
                    seen[rr, cc] = True
                    q.append((rr, cc))
        comps.append(comp)
    return comps


def _outer_outline(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    padded = np.pad(mask.astype(bool), 1, constant_values=False)
    exterior = np.zeros_like(padded, dtype=bool)
    q: deque[tuple[int, int]] = deque([(0, 0)])
    exterior[0, 0] = True

    while q:
        r, c = q.popleft()
        for dr, dc in DIR4:
            rr, cc = r + dr, c + dc
            if (
                0 <= rr < h + 2
                and 0 <= cc < w + 2
                and not exterior[rr, cc]
                and not bool(padded[rr, cc])
            ):
                exterior[rr, cc] = True
                q.append((rr, cc))

    outline = np.zeros((h, w), dtype=bool)
    for r, c in np.argwhere(mask):
        r, c = int(r), int(c)
        pr, pc = r + 1, c + 1
        outline[r, c] = any(exterior[pr + dr, pc + dc] for dr, dc in DIR4)
    return outline


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each multicolor sprite into body pixels plus direction-coded outline projections.

    Input structure:
    - Black background with several 4-connected multicolor sprites.
    - Each sprite has two or three colors: one majority body color and one or two singleton marker colors.
    - The singleton markers lie at visible positions inside the sprite, away from the exact sprite center.

    Rule:
    - Keep every body-color pixel in place and recolor each singleton marker pixel to the sprite body color.
    - For each marker, compare its position with the sprite bounding-box center to get an up, down, left, right, or diagonal direction.
    - Shift the sprite silhouette one full bounding-box span plus one blank pixel in that direction, and draw only the silhouette's outer outline in the marker color.

    Key insight:
    - Marker colors are not moved as objects; their relative positions inside the sprite encode the direction where a colored outline copy must appear.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert np.all((0 <= g) & (g <= 9))

    out = g.copy()
    paint: dict[tuple[int, int], int] = {}
    comps = _components(g != BG)
    assert comps, "expected at least one multicolor sprite"

    h, w = g.shape
    for comp in comps:
        rows = [r for r, _c in comp]
        cols = [c for _r, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        bh, bw = r1 - r0 + 1, c1 - c0 + 1

        colors, counts = np.unique([int(g[r, c]) for r, c in comp], return_counts=True)
        assert len(colors) in (2, 3), "each sprite must contain body color plus one or two marker colors"
        max_count = int(counts.max())
        winners = [int(color) for color, count in zip(colors, counts) if int(count) == max_count]
        assert len(winners) == 1 and max_count > 1, "body color must be the unique majority"
        body = winners[0]

        local_mask = np.zeros((bh, bw), dtype=bool)
        for r, c in comp:
            local_mask[r - r0, c - c0] = True
        outline = _outer_outline(local_mask)

        marker_cells: list[tuple[int, int, int]] = []
        for color, count in zip(colors, counts):
            color, count = int(color), int(count)
            if color == body:
                continue
            assert count == 1, "each non-body color must be a singleton marker"
            coords = [(r, c) for r, c in comp if int(g[r, c]) == color]
            assert len(coords) == 1
            mr, mc = coords[0]
            marker_cells.append((mr, mc, color))
            out[mr, mc] = body

        center2_r = r0 + r1
        center2_c = c0 + c1
        for mr, mc, color in marker_cells:
            dr = _sign(2 * mr - center2_r)
            dc = _sign(2 * mc - center2_c)
            assert (dr, dc) != (0, 0), "marker cannot sit at the sprite center"

            shift_r = dr * (bh + 1) if dr else 0
            shift_c = dc * (bw + 1) if dc else 0
            tr0, tc0 = r0 + shift_r, c0 + shift_c
            tr1, tc1 = tr0 + bh - 1, tc0 + bw - 1
            assert 0 <= tr0 <= tr1 < h and 0 <= tc0 <= tc1 < w, "projection must fit on the canvas"

            for lr, lc in np.argwhere(outline):
                rr, cc = tr0 + int(lr), tc0 + int(lc)
                assert int(g[rr, cc]) == BG, "projection must land on background"
                previous = paint.setdefault((rr, cc), color)
                assert previous == color, "outline projections cannot conflict"

    for (r, c), color in paint.items():
        out[r, c] = color
    return out


# --------------- generator ---------------

GEN_COLORS = list(range(1, 10))
GEN_SIZE_RANGE = (16, 30)
GEN_OBJECT_RANGE = (2, 4)
GEN_SHAPES = ((3, 3), (3, 5), (5, 3), (5, 5))
GEN_MAX_ATTEMPTS = 500
GEN_PLACE_ATTEMPTS = 80


def _cells_for_direction(h: int, w: int, direction: tuple[int, int]) -> list[tuple[int, int]]:
    cr, cc = h // 2, w // 2
    dr, dc = direction
    out: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r, c) == (cr, cc):
                continue
            if _sign(r - cr) == dr and _sign(c - cc) == dc:
                out.append((r, c))
    return out


def _path_between(a: tuple[int, int], b: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = a
    er, ec = b
    cells = [(r, c)]
    while (r, c) != (er, ec):
        steps: list[tuple[int, int]] = []
        if r != er:
            steps.append((_sign(er - r), 0))
        if c != ec:
            steps.append((0, _sign(ec - c)))
        dr, dc = random.choice(steps)
        r += dr
        c += dc
        cells.append((r, c))
    return cells


def _grow_mask(h: int, w: int, marker_positions: list[tuple[int, int]]) -> np.ndarray:
    cr, cc = h // 2, w // 2
    required = {(cr, cc), (0, cc), (h - 1, cc), (cr, 0), (cr, w - 1), *marker_positions}
    cells: set[tuple[int, int]] = set()
    for target in required:
        cells.update(_path_between((cr, cc), target))

    target_size = random.randint(max(len(cells), 6), h * w)
    while len(cells) < target_size:
        frontier = sorted(
            {
                (r + dr, c + dc)
                for r, c in cells
                for dr, dc in DIR4
                if 0 <= r + dr < h and 0 <= c + dc < w and (r + dr, c + dc) not in cells
            }
        )
        if not frontier:
            break
        cells.add(random.choice(frontier))

    mask = np.zeros((h, w), dtype=bool)
    for r, c in cells:
        mask[r, c] = True
    return mask


def _make_sprite_spec() -> dict[str, object]:
    h, w = random.choice(GEN_SHAPES)
    marker_count = random.choice((1, 2))
    directions = random.sample(DIR8, marker_count)
    marker_positions = [random.choice(_cells_for_direction(h, w, direction)) for direction in directions]
    mask = _grow_mask(h, w, marker_positions)
    assert int(mask.sum()) - marker_count > 1
    return {
        "mask": mask,
        "directions": directions,
        "marker_positions": marker_positions,
    }


def _spec_relative_cells(spec: dict[str, object]) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    mask = spec["mask"]
    directions = spec["directions"]
    assert isinstance(mask, np.ndarray)
    assert isinstance(directions, list)
    h, w = mask.shape
    outline = _outer_outline(mask)

    source = {(int(r), int(c)) for r, c in np.argwhere(mask)}
    output = set(source)
    for dr, dc in directions:
        assert isinstance(dr, int) and isinstance(dc, int)
        shift_r = dr * (h + 1) if dr else 0
        shift_c = dc * (w + 1) if dc else 0
        for r, c in np.argwhere(outline):
            output.add((int(r) + shift_r, int(c) + shift_c))
    return source, output


def _can_place(
    top: int,
    left: int,
    output_cells: set[tuple[int, int]],
    blocked: np.ndarray,
) -> bool:
    h, w = blocked.shape
    for r, c in output_cells:
        rr, cc = top + r, left + c
        if not (0 <= rr < h and 0 <= cc < w):
            return False
        if blocked[rr, cc]:
            return False
    return True


def _mark_blocked(blocked: np.ndarray, top: int, left: int, cells: set[tuple[int, int]]) -> None:
    h, w = blocked.shape
    for r, c in cells:
        rr, cc = top + r, left + c
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr, nc = rr + dr, cc + dc
                if 0 <= nr < h and 0 <= nc < w:
                    blocked[nr, nc] = True


def _place_spec(spec: dict[str, object], blocked: np.ndarray) -> tuple[int, int, set[tuple[int, int]], set[tuple[int, int]]] | None:
    source_cells, output_cells = _spec_relative_cells(spec)
    min_r = min(r for r, _c in output_cells)
    max_r = max(r for r, _c in output_cells)
    min_c = min(c for _r, c in output_cells)
    max_c = max(c for _r, c in output_cells)
    h, w = blocked.shape

    top_lo, top_hi = -min_r, h - 1 - max_r
    left_lo, left_hi = -min_c, w - 1 - max_c
    if top_lo > top_hi or left_lo > left_hi:
        return None

    positions = [(r, c) for r in range(top_lo, top_hi + 1) for c in range(left_lo, left_hi + 1)]
    random.shuffle(positions)
    for top, left in positions[:GEN_PLACE_ATTEMPTS]:
        if _can_place(top, left, output_cells, blocked):
            return top, left, source_cells, output_cells
    for top, left in positions[GEN_PLACE_ATTEMPTS:]:
        if _can_place(top, left, output_cells, blocked):
            return top, left, source_cells, output_cells
    return None


def _render_sprite(grid: np.ndarray, top: int, left: int, spec: dict[str, object]) -> None:
    mask = spec["mask"]
    marker_positions = spec["marker_positions"]
    assert isinstance(mask, np.ndarray)
    assert isinstance(marker_positions, list)

    body = random.choice(GEN_COLORS)
    marker_colors = random.sample([color for color in GEN_COLORS if color != body], len(marker_positions))
    for r, c in np.argwhere(mask):
        grid[top + int(r), left + int(c)] = body
    for (r, c), color in zip(marker_positions, marker_colors):
        grid[top + int(r), left + int(c)] = color


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(*GEN_SIZE_RANGE)
    w = random.randint(*GEN_SIZE_RANGE)
    grid = np.full((h, w), BG, dtype=int)
    blocked = np.zeros((h, w), dtype=bool)
    object_count = random.randint(*GEN_OBJECT_RANGE)

    for _ in range(object_count):
        placed = False
        for _try in range(40):
            spec = _make_sprite_spec()
            placement = _place_spec(spec, blocked)
            if placement is None:
                continue
            top, left, _source_cells, output_cells = placement
            _render_sprite(grid, top, left, spec)
            _mark_blocked(blocked, top, left, output_cells)
            placed = True
            break
        if not placed:
            return None

    try:
        solved = main(grid)
    except Exception:
        return None
    if np.array_equal(grid, solved):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid seed2047415201/7/4 input")
