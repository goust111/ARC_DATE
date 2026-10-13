import random
from collections import deque

import numpy as np


BLACK = 0
MAROON = 9
COLOR_POOL = [1, 2, 3, 4, 5, 6, 7, 8]
DIRS4 = [(-1, 0), (0, 1), (1, 0), (0, -1)]
SEGMENT_TEMPLATES = [
    ((0, 0), (0, 1)),
    ((0, 0), (0, 1), (0, 2)),
    ((0, 0), (0, 1), (0, 2), (0, 3)),
    ((0, 0), (0, 1), (1, 1)),
    ((0, 0), (0, 1), (1, 1), (2, 1)),
    ((0, 0), (0, 1), (0, 2), (1, 2)),
]

GEN_RETRIES = 400
GRID_H_RANGE = (16, 24)
GRID_W_RANGE = (18, 28)
SEGMENT_COUNT_CHOICES = (3, 4, 5)
LEGEND_COL_CHOICES = (1, 2)
LEGEND_GAP_EXTRAS = (1, 2)
MIN_PATH_COL = 5
SCATTER_TRIES = 300
FIRST_CANDIDATE_LIMIT = 240


def _neighbors4(r: int, c: int):
    for dr, dc in DIRS4:
        yield r + dr, c + dc


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
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rotate_point(pt: tuple[int, int], turns: int) -> tuple[int, int]:
    r, c = pt
    for _ in range(turns % 4):
        r, c = c, -r
    return int(r), int(c)


def _rotate_coords(coords: tuple[tuple[int, int], ...], turns: int) -> tuple[tuple[int, int], ...]:
    return tuple(_rotate_point(pt, turns) for pt in coords)


def _translate(coords: tuple[tuple[int, int], ...], anchor: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    ar, ac = anchor
    return tuple((ar + r, ac + c) for r, c in coords)


def _in_bounds(coords: tuple[tuple[int, int], ...], h: int, w: int) -> bool:
    return all(0 <= r < h and 0 <= c < w for r, c in coords)


def _touch_pairs(
    coords: tuple[tuple[int, int], ...], occupied: set[tuple[int, int]]
) -> set[tuple[tuple[int, int], tuple[int, int]]]:
    pairs: set[tuple[tuple[int, int], tuple[int, int]]] = set()
    for r, c in coords:
        for nr, nc in _neighbors4(r, c):
            if (nr, nc) in occupied:
                pairs.add(((r, c), (nr, nc)))
    return pairs


def _expand(cells: set[tuple[int, int]], h: int, w: int, pad: int = 1) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for rr in range(r - pad, r + pad + 1):
            for cc in range(c - pad, c + pad + 1):
                if 0 <= rr < h and 0 <= cc < w:
                    out.add((rr, cc))
    return out


def _sample_spaced_positions(
    count: int, lo: int, hi: int, extra_gap: int, rng: random.Random
) -> list[int] | None:
    span = hi - lo + 1
    needed = count + (count - 1) * extra_gap
    if span < needed:
        return None
    compressed = span - (count - 1) * extra_gap
    picks = sorted(rng.sample(range(compressed), count))
    return [lo + picks[i] + i * extra_gap for i in range(count)]


def _parse_segment_component(grid: np.ndarray, comp: list[tuple[int, int]]) -> dict:
    colors = {int(grid[r, c]) for r, c in comp}
    maroon_cells = [(int(r), int(c)) for r, c in comp if int(grid[r, c]) == MAROON]
    assert len(maroon_cells) == 1, "each segment needs exactly one maroon entry"
    other_colors = colors - {MAROON}
    assert len(other_colors) == 1, "each segment must have one visible color"
    color = int(next(iter(other_colors)))

    comp_set = {(int(r), int(c)) for r, c in comp}
    degrees: dict[tuple[int, int], int] = {}
    for cell in comp_set:
        degrees[cell] = sum((nr, nc) in comp_set for nr, nc in _neighbors4(*cell))
    endpoints = [cell for cell, deg in degrees.items() if deg == 1]
    assert len(comp_set) >= 2 and len(endpoints) == 2, "segment must be a simple path"

    entry = maroon_cells[0]
    assert entry in endpoints, "the maroon cell must be an endpoint"
    exit_cell = endpoints[0] if endpoints[1] == entry else endpoints[1]
    er, ec = entry
    rel_coords = tuple(sorted((r - er, c - ec) for r, c in comp_set))
    exit_rel = (exit_cell[0] - er, exit_cell[1] - ec)
    return {
        "color": color,
        "entry": entry,
        "exit": exit_cell,
        "rel_coords": rel_coords,
        "exit_rel": exit_rel,
        "abs_coords": tuple(sorted(comp_set)),
    }


def _extract_order_and_segments(grid: np.ndarray) -> tuple[list[tuple[int, int, int]], dict[int, dict]]:
    comps = _components(grid != BLACK)
    legend: list[tuple[int, int, int]] = []
    segments: dict[int, dict] = {}
    for comp in comps:
        if len(comp) == 1:
            r, c = comp[0]
            color = int(grid[r, c])
            assert color != MAROON, "legend pixels cannot be maroon"
            legend.append((int(r), int(c), color))
            continue
        assert 2 <= len(comp) <= 4, "segments must contain 2 to 4 cells"
        seg = _parse_segment_component(grid, comp)
        assert seg["color"] not in segments, "segment colors must be unique"
        segments[seg["color"]] = seg

    assert legend, "a left legend is required"
    legend.sort(key=lambda item: item[0])
    legend_cols = {c for _, c, _ in legend}
    assert len(legend_cols) == 1, "legend must be a single vertical column"
    order = [color for _, _, color in legend]
    assert set(order) == set(segments), "legend colors must match segment colors"
    return legend, segments


def _segment_candidates(
    segment: dict,
    current_exit: tuple[int, int],
    occupied: set[tuple[int, int]],
    shape: tuple[int, int],
) -> list[dict]:
    h, w = shape
    results: list[dict] = []
    for turns in range(4):
        rel_coords = _rotate_coords(segment["rel_coords"], turns)
        exit_rel = _rotate_point(segment["exit_rel"], turns)
        for dr, dc in DIRS4:
            entry_abs = (current_exit[0] + dr, current_exit[1] + dc)
            abs_coords = _translate(rel_coords, entry_abs)
            if not _in_bounds(abs_coords, h, w):
                continue
            if occupied.intersection(abs_coords):
                continue
            if _touch_pairs(abs_coords, occupied) != {(entry_abs, current_exit)}:
                continue
            exit_abs = (entry_abs[0] + exit_rel[0], entry_abs[1] + exit_rel[1])
            results.append(
                {
                    "coords": abs_coords,
                    "entry": entry_abs,
                    "exit": exit_abs,
                    "turns": turns,
                }
            )
    return results


def _draw_segment(out: np.ndarray, coords: tuple[tuple[int, int], ...], color: int) -> None:
    for r, c in coords:
        out[r, c] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Reassemble the colored path pieces into one ordered polyline while preserving the legend.

    Input structure:
    - Black is the background.
    - On the left there is one vertical column of isolated single legend pixels, each with a distinct non-Maroon color.
    - On the right there is exactly one 2-to-4-cell path segment per legend color; each segment is straight or L-shaped and has one Maroon endpoint marking its entry.

    Rule:
    - Read the legend colors from top to bottom to get the segment order.
    - Clear all segment cells from the canvas, keep the legend fixed, and keep the first ordered segment at its original position.
    - For each later segment, try the four right-angle rotations and place it so its Maroon entry cell sits next to the previous segment's exit cell while the growing drawing stays a single non-branching polyline.
    - Redraw the placed segment in its own color, not Maroon, and continue from its exit endpoint.

    Key insight:
    - The legend gives the placement order, the Maroon endpoint gives the traversal direction, and the valid rotation is the one that lets each new piece attach to the current open end without creating extra contacts.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    legend, segments = _extract_order_and_segments(g)

    out = np.full_like(g, BLACK)
    occupied = {(r, c) for r, c, _ in legend}
    for r, c, color in legend:
        out[r, c] = color

    first = segments[legend[0][2]]
    first_coords = tuple(sorted(first["abs_coords"]))
    _draw_segment(out, first_coords, first["color"])
    occupied.update(first_coords)
    current_exit = first["exit"]

    for _r, _c, color in legend[1:]:
        seg = segments[color]
        candidates = _segment_candidates(seg, current_exit, occupied, g.shape)
        assert len(candidates) == 1, "each next segment must have a unique legal placement"
        chosen = candidates[0]
        _draw_segment(out, chosen["coords"], color)
        occupied.update(chosen["coords"])
        current_exit = chosen["exit"]

    return out


def _first_segment_candidates(
    shape: tuple[int, int], legend_cells: set[tuple[int, int]]
) -> list[dict]:
    h, w = shape
    candidates: list[dict] = []
    occupied = set(legend_cells)
    for template in SEGMENT_TEMPLATES:
        base_segment = {
            "rel_coords": template,
            "exit_rel": template[-1],
        }
        for turns in range(4):
            rel_coords = _rotate_coords(base_segment["rel_coords"], turns)
            exit_rel = _rotate_point(base_segment["exit_rel"], turns)
            rows = [r for r, _ in rel_coords]
            cols = [c for _, c in rel_coords]
            row_lo, row_hi = -min(rows), h - 1 - max(rows)
            col_lo, col_hi = -min(cols), w - 1 - max(cols)
            for entry_r in range(row_lo, row_hi + 1):
                for entry_c in range(max(col_lo, MIN_PATH_COL), col_hi + 1):
                    coords = _translate(rel_coords, (entry_r, entry_c))
                    if occupied.intersection(coords):
                        continue
                    if _touch_pairs(coords, occupied):
                        continue
                    exit_abs = (entry_r + exit_rel[0], entry_c + exit_rel[1])
                    if not (
                        exit_abs[0] == 0
                        or exit_abs[0] == h - 1
                        or exit_abs[1] == 0
                        or exit_abs[1] == w - 1
                    ):
                        continue
                    candidates.append(
                        {
                            "rel_coords": tuple(template),
                            "exit_rel": template[-1],
                            "coords": coords,
                            "entry": (entry_r, entry_c),
                            "exit": exit_abs,
                        }
                    )
    return candidates


def _build_ordered_output(rng: random.Random):
    h = rng.randint(*GRID_H_RANGE)
    w = rng.randint(*GRID_W_RANGE)
    n_segments = rng.choice(SEGMENT_COUNT_CHOICES)
    if h < 2 * n_segments + 2:
        return None

    legend_col = rng.choice(LEGEND_COL_CHOICES)
    extra_gap = rng.choice(LEGEND_GAP_EXTRAS)
    legend_rows = _sample_spaced_positions(n_segments, 1, h - 2, extra_gap, rng)
    if legend_rows is None:
        return None
    colors = rng.sample(COLOR_POOL, n_segments)
    legend = [(legend_rows[i], legend_col, colors[i]) for i in range(n_segments)]
    legend_cells = {(r, c) for r, c, _ in legend}

    first_choices = _first_segment_candidates((h, w), legend_cells)
    rng.shuffle(first_choices)
    first_choices = first_choices[:FIRST_CANDIDATE_LIMIT]
    if not first_choices:
        return None

    def extend_path(
        index: int,
        occupied: set[tuple[int, int]],
        current_exit: tuple[int, int],
        pieces: list[dict],
    ) -> list[dict] | None:
        if index == n_segments:
            return pieces

        template_order = list(SEGMENT_TEMPLATES)
        rng.shuffle(template_order)
        for template in template_order:
            seg = {"rel_coords": tuple(template), "exit_rel": template[-1]}
            candidates = _segment_candidates(seg, current_exit, occupied, (h, w))
            if len(candidates) != 1:
                continue
            candidate = candidates[0]
            coords = candidate["coords"]
            if min(c for _, c in coords) < MIN_PATH_COL:
                continue
            next_piece = {
                "color": colors[index],
                "rel_coords": tuple(template),
                "exit_rel": template[-1],
                "coords": coords,
                "entry": candidate["entry"],
                "exit": candidate["exit"],
            }
            result = extend_path(
                index + 1,
                occupied | set(coords),
                candidate["exit"],
                pieces + [next_piece],
            )
            if result is not None:
                return result
        return None

    for first in first_choices:
        pieces = [
            {
                "color": colors[0],
                "rel_coords": first["rel_coords"],
                "exit_rel": first["exit_rel"],
                "coords": first["coords"],
                "entry": first["entry"],
                "exit": first["exit"],
            }
        ]
        result = extend_path(
            1,
            legend_cells | set(first["coords"]),
            first["exit"],
            pieces,
        )
        if result is None:
            continue

        output = np.full((h, w), BLACK, dtype=int)
        for r, c, color in legend:
            output[r, c] = color
        for piece in result:
            _draw_segment(output, piece["coords"], piece["color"])
        return {
            "shape": (h, w),
            "legend": legend,
            "pieces": result,
            "output": output,
        }
    return None


def _scatter_segments(spec: dict, rng: random.Random) -> np.ndarray | None:
    h, w = spec["shape"]
    grid = np.full((h, w), BLACK, dtype=int)
    legend = spec["legend"]
    pieces = spec["pieces"]

    for r, c, color in legend:
        grid[r, c] = color

    first = pieces[0]
    for r, c in first["coords"]:
        grid[r, c] = first["color"]
    grid[first["entry"]] = MAROON

    blocked = _expand({(r, c) for r, c, _ in legend} | set(first["coords"]), h, w)
    blocked_output_sets = [set(piece["coords"]) for piece in pieces]

    for idx, piece in enumerate(pieces[1:], start=1):
        turns_order = [0, 1, 2, 3]
        rng.shuffle(turns_order)
        placed = False
        for turns in turns_order:
            rel_coords = _rotate_coords(piece["rel_coords"], turns)
            rows = [r for r, _ in rel_coords]
            cols = [c for _, c in rel_coords]
            row_lo, row_hi = -min(rows), h - 1 - max(rows)
            col_lo, col_hi = -min(cols), w - 1 - max(cols)
            anchors = [
                (r, c)
                for r in range(row_lo, row_hi + 1)
                for c in range(max(col_lo, MIN_PATH_COL), col_hi + 1)
            ]
            rng.shuffle(anchors)
            for entry_r, entry_c in anchors[:SCATTER_TRIES]:
                coords = _translate(rel_coords, (entry_r, entry_c))
                coord_set = set(coords)
                if coord_set & blocked:
                    continue
                if coord_set == blocked_output_sets[idx]:
                    continue
                if _touch_pairs(coords, blocked - coord_set):
                    continue
                for r, c in coords:
                    grid[r, c] = piece["color"]
                grid[entry_r, entry_c] = MAROON
                blocked |= _expand(coord_set, h, w)
                placed = True
                break
            if placed:
                break
        if not placed:
            return None

    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    spec = _build_ordered_output(rng)
    if spec is None:
        return None
    grid = _scatter_segments(spec, rng)
    if grid is None:
        return None
    if not np.array_equal(main(grid), spec["output"]):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for the ordered path-assembly task")
