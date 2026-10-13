from __future__ import annotations

import itertools
from collections import Counter

import numpy as np


BACKGROUND = 0
MARKER = 2
SEPARATOR = 5
CELL_SIZE = 3
ABSTRACT_SIZE = 5
COLOR_POOL = (1, 3, 4, 6, 7, 8, 9)
ORDERED_NEIGHBOR_OFFSETS = [
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
]
MAX_GENERATION_ATTEMPTS = 500
RNG = np.random.default_rng()
CALL_INDEX = 0


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _groups(indices: list[int]) -> list[tuple[int, int]]:
    groups: list[tuple[int, int]] = []
    start = prev = indices[0]
    for idx in indices[1:]:
        if idx == prev + 1:
            prev = idx
        else:
            groups.append((start, prev))
            start = prev = idx
    groups.append((start, prev))
    return groups


def _spans_between(line_groups: list[tuple[int, int]], limit: int) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    start = 0
    for line_start, line_end in line_groups:
        if start < line_start:
            spans.append((start, line_start - 1))
        start = line_end + 1
    if start < limit:
        spans.append((start, limit - 1))
    assert spans
    return spans


def _separator_color(grid: np.ndarray) -> int:
    row_colors = {
        int(grid[r, 0])
        for r in range(grid.shape[0])
        if np.all(grid[r, :] == grid[r, 0])
    }
    col_colors = {
        int(grid[0, c])
        for c in range(grid.shape[1])
        if np.all(grid[:, c] == grid[0, c])
    }
    candidates = row_colors & col_colors
    assert len(candidates) == 1
    return candidates.pop()


def _compress_lattice(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], np.ndarray]:
    sep = _separator_color(grid)
    row_lines = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    col_lines = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    assert row_lines and col_lines

    row_spans = _spans_between(_groups(row_lines), grid.shape[0])
    col_spans = _spans_between(_groups(col_lines), grid.shape[1])

    cells = np.zeros((len(row_spans), len(col_spans)), dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            block = grid[r0 : r1 + 1, c0 : c1 + 1]
            values = np.unique(block)
            assert len(values) == 1
            cells[r, c] = int(values[0])
    return row_spans, col_spans, cells


def _expand_lattice(cells: np.ndarray, row_spans: list[tuple[int, int]], col_spans: list[tuple[int, int]], sep: int) -> np.ndarray:
    h = row_spans[-1][1] + 1
    w = col_spans[-1][1] + 1
    out = np.full((h, w), sep, dtype=int)
    for r, (r0, r1) in enumerate(row_spans):
        for c, (c0, c1) in enumerate(col_spans):
            out[r0 : r1 + 1, c0 : c1 + 1] = int(cells[r, c])
    return out


def _uniform_spans(size: int, cell_size: int) -> list[tuple[int, int]]:
    return [(i * (cell_size + 1), i * (cell_size + 1) + cell_size - 1) for i in range(size)]


def _ordered_template(cells: np.ndarray, center: tuple[int, int], marker: int = MARKER) -> list[tuple[int, int, int]]:
    h, w = cells.shape
    r, c = center
    bg = _get_bg(cells)
    template: list[tuple[int, int, int]] = []
    for dr, dc in ORDERED_NEIGHBOR_OFFSETS:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            value = int(cells[rr, cc])
            if value not in (bg, marker):
                template.append((dr, dc, value))
    return template


def _find_exemplar(cells: np.ndarray) -> tuple[tuple[int, int], list[tuple[int, int, int]]]:
    markers = [tuple(map(int, rc)) for rc in np.argwhere(cells == MARKER)]
    assert 4 <= len(markers) <= 5

    decorated: list[tuple[tuple[int, int], list[tuple[int, int, int]]]] = []
    for center in markers:
        template = _ordered_template(cells, center)
        if template:
            decorated.append((center, template))
    assert len(decorated) == 1
    exemplar, template = decorated[0]
    colors = [color for _dr, _dc, color in template]
    assert 2 <= len(colors) <= 4
    assert len(set(colors)) == len(colors)
    return exemplar, template


def _shift_right(seq: list[int], amount: int) -> list[int]:
    if not seq:
        return []
    k = amount % len(seq)
    if k == 0:
        return list(seq)
    return list(seq[-k:] + seq[:-k])


def _solve_cells(cells: np.ndarray) -> np.ndarray:
    out = cells.copy()
    exemplar, template = _find_exemplar(cells)
    er, ec = exemplar
    sequence = [color for _dr, _dc, color in template]
    h, w = cells.shape

    for r, c in [tuple(map(int, rc)) for rc in np.argwhere(cells == MARKER)]:
        distance = max(abs(r - er), abs(c - ec))
        shifted = _shift_right(sequence, distance)
        for (dr, dc, _old_color), new_color in zip(template, shifted):
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                out[rr, cc] = int(new_color)
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """Stamp the exemplar neighborhood onto every red marker with a distance-driven cyclic color shift.

    Input structure:
    - A black background is split into square lattice cells by full grey separator lines.
    - Exactly one red marker cell is decorated by 2-4 distinct non-background neighbors in its 8-neighborhood.
    - The same red marker color appears 3-4 more times as isolated cells elsewhere in the lattice.

    Rule:
    - Compress the lattice to one color per cell and locate the unique decorated red marker.
    - Read its non-background neighbors from top-left to bottom-right to get both the stamp offsets and the base color sequence.
    - For every red marker, compute its Chebyshev distance from the exemplar in lattice coordinates, cyclically right-shift the color sequence by that distance, and stamp the shifted colors at the same offsets, clipping at lattice edges.

    Key insight:
    - The exemplar contributes the stamp shape, but each target's colors are not copied directly: the palette rotates by an amount encoded by that target's lattice distance from the exemplar.
    """
    g = np.asarray(grid, dtype=int)
    row_spans, col_spans, cells = _compress_lattice(g)
    solved_cells = _solve_cells(cells)
    return _expand_lattice(solved_cells, row_spans, col_spans, _separator_color(g))


def _stamp_cells(center: tuple[int, int], offsets: list[tuple[int, int, int]], size: int = ABSTRACT_SIZE) -> tuple[set[tuple[int, int]], bool]:
    r, c = center
    in_bounds: set[tuple[int, int]] = set()
    clipped = False
    for dr, dc, _color in offsets:
        rr, cc = r + dr, c + dc
        if 0 <= rr < size and 0 <= cc < size:
            in_bounds.add((rr, cc))
        else:
            clipped = True
    return in_bounds, clipped


def _chebyshev(a: tuple[int, int], b: tuple[int, int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _build_input_cells(
    exemplar: tuple[int, int],
    offsets: list[tuple[int, int, int]],
    targets: list[tuple[int, int]],
    size: int = ABSTRACT_SIZE,
) -> np.ndarray:
    cells = np.zeros((size, size), dtype=int)
    cells[exemplar] = MARKER
    er, ec = exemplar
    for dr, dc, color in offsets:
        rr, cc = er + dr, ec + dc
        if 0 <= rr < size and 0 <= cc < size:
            cells[rr, cc] = int(color)
    for r, c in targets:
        cells[r, c] = MARKER
    return cells


def _build_output_cells(
    exemplar: tuple[int, int],
    offsets: list[tuple[int, int, int]],
    targets: list[tuple[int, int]],
    size: int = ABSTRACT_SIZE,
) -> np.ndarray:
    cells = np.zeros((size, size), dtype=int)
    centers = [exemplar, *targets]
    sequence = [color for _dr, _dc, color in offsets]
    er, ec = exemplar

    for r, c in centers:
        cells[r, c] = MARKER
    for r, c in centers:
        shifted = _shift_right(sequence, _chebyshev((r, c), exemplar))
        for (dr, dc, _old_color), new_color in zip(offsets, shifted):
            rr, cc = r + dr, c + dc
            if 0 <= rr < size and 0 <= cc < size:
                cells[rr, cc] = int(new_color)
    return cells


def _render_cells(cells: np.ndarray) -> np.ndarray:
    spans = _uniform_spans(cells.shape[0], CELL_SIZE)
    return _expand_lattice(cells, spans, spans, SEPARATOR)


def _exemplar_choices() -> list[tuple[int, int]]:
    interior = [(r, c) for r in range(1, ABSTRACT_SIZE - 1) for c in range(1, ABSTRACT_SIZE - 1)]
    center = (ABSTRACT_SIZE // 2, ABSTRACT_SIZE // 2)
    return [pos for pos in interior if pos != center] + [center]


def _choose_offsets(template_len: int) -> list[tuple[int, int, int]]:
    chosen = RNG.choice(len(ORDERED_NEIGHBOR_OFFSETS), size=template_len, replace=False)
    chosen_offsets = [ORDERED_NEIGHBOR_OFFSETS[int(i)] for i in sorted(chosen)]
    colors = RNG.choice(COLOR_POOL, size=template_len, replace=False)
    return [(dr, dc, int(color)) for (dr, dc), color in zip(chosen_offsets, colors)]


def _candidate_targets(
    exemplar: tuple[int, int],
    offsets: list[tuple[int, int, int]],
) -> list[dict]:
    exemplar_stamp, _ = _stamp_cells(exemplar, offsets)
    exemplar_input_occ = {exemplar, *exemplar_stamp}
    exemplar_output_occ = {exemplar, *exemplar_stamp}
    candidates: list[dict] = []

    for r in range(ABSTRACT_SIZE):
        for c in range(ABSTRACT_SIZE):
            center = (r, c)
            if center == exemplar:
                continue
            if any(_chebyshev(center, cell) <= 1 for cell in exemplar_input_occ):
                continue
            stamp_occ, clipped = _stamp_cells(center, offsets)
            output_occ = {center, *stamp_occ}
            if output_occ & exemplar_output_occ:
                continue
            candidates.append(
                {
                    "center": center,
                    "output_occ": output_occ,
                    "distance": _chebyshev(center, exemplar),
                    "clipped": clipped,
                }
            )
    return candidates


def _valid_target_combo(combo: tuple[dict, ...], template_len: int, want_clipping: bool) -> bool:
    centers = [cand["center"] for cand in combo]
    if len({cand["distance"] for cand in combo}) < 2:
        return False
    if len({cand["distance"] % template_len for cand in combo}) < 2:
        return False
    if want_clipping:
        if not any(cand["clipped"] for cand in combo):
            return False
        if len(combo) > 1 and not any(not cand["clipped"] for cand in combo):
            return False
    else:
        if any(cand["clipped"] for cand in combo):
            return False

    for i, center_a in enumerate(centers):
        for center_b in centers[i + 1 :]:
            if _chebyshev(center_a, center_b) <= 1:
                return False
    for i, cand_a in enumerate(combo):
        for cand_b in combo[i + 1 :]:
            if cand_a["output_occ"] & cand_b["output_occ"]:
                return False
    return True


def _choose_targets(
    exemplar: tuple[int, int],
    offsets: list[tuple[int, int, int]],
    target_count: int,
    want_clipping: bool,
) -> list[tuple[int, int]] | None:
    candidates = _candidate_targets(exemplar, offsets)
    RNG.shuffle(candidates)
    valid: list[list[tuple[int, int]]] = []
    for combo in itertools.combinations(candidates, target_count):
        if _valid_target_combo(combo, len(offsets), want_clipping):
            valid.append([cand["center"] for cand in combo])
    if not valid:
        return None
    return valid[int(RNG.integers(len(valid)))]


def _extract_case_facts(cells: np.ndarray) -> dict:
    exemplar, template = _find_exemplar(cells)
    targets = [tuple(map(int, rc)) for rc in np.argwhere(cells == MARKER) if tuple(map(int, rc)) != exemplar]
    distances = [_chebyshev(target, exemplar) for target in targets]
    clipped = [_stamp_cells(target, template, cells.shape[0])[1] for target in targets]
    shifted_patterns = [
        tuple(_shift_right([color for _dr, _dc, color in template], dist))
        for dist in distances
    ]
    return {
        "exemplar": exemplar,
        "template": template,
        "targets": targets,
        "distances": distances,
        "clipped": clipped,
        "shifted_patterns": shifted_patterns,
    }


def _validate_case(inp: np.ndarray, expected_out: np.ndarray, want_clipping: bool) -> bool:
    row_spans, col_spans, cells = _compress_lattice(inp)
    if cells.shape != (ABSTRACT_SIZE, ABSTRACT_SIZE):
        return False
    if any((r1 - r0 + 1) != CELL_SIZE for r0, r1 in row_spans):
        return False
    if any((c1 - c0 + 1) != CELL_SIZE for c0, c1 in col_spans):
        return False

    facts = _extract_case_facts(cells)
    template = facts["template"]
    if not (2 <= len(template) <= 4):
        return False
    if len(facts["targets"]) not in (3, 4):
        return False
    if len(set(facts["distances"])) < 2:
        return False
    if len(set(d % len(template) for d in facts["distances"])) < 2:
        return False
    if len(set(facts["shifted_patterns"])) < 2:
        return False
    if want_clipping:
        if not any(facts["clipped"]) or not any(not flag for flag in facts["clipped"]):
            return False
    else:
        if any(facts["clipped"]):
            return False
    if not np.array_equal(main(inp), expected_out):
        return False
    return True


def _gen_attempt(template_len: int, target_count: int, want_clipping: bool) -> np.ndarray | None:
    exemplar = _exemplar_choices()[int(RNG.integers(len(_exemplar_choices())))]
    offsets = _choose_offsets(template_len)
    targets = _choose_targets(exemplar, offsets, target_count, want_clipping)
    if targets is None:
        return None

    input_cells = _build_input_cells(exemplar, offsets, targets)
    output_cells = _build_output_cells(exemplar, offsets, targets)
    inp = _render_cells(input_cells)
    expected_out = _render_cells(output_cells)

    if not _validate_case(inp, expected_out, want_clipping):
        return None
    return inp


def generate_input() -> np.ndarray:
    global CALL_INDEX
    schedule = [
        (2, 3, False),
        (3, 4, True),
        (4, 3, True),
        (3, 3, False),
        (2, 4, True),
        (4, 4, True),
    ]

    template_len, target_count, want_clipping = schedule[CALL_INDEX % len(schedule)]

    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(template_len, target_count, want_clipping)
        if grid is not None:
            CALL_INDEX += 1
            return grid
    raise RuntimeError("failed to generate a valid input")
