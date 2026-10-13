from __future__ import annotations

import random

import numpy as np


COPY = 0
MIRROR = 1
ALL_COLORS = tuple(range(10))
MAX_GENERATION_ATTEMPTS = 400


def _apply_mode(left: np.ndarray, mode: int) -> np.ndarray:
    return left.copy() if mode == COPY else left[::-1].copy()


def _is_asymmetric(row: np.ndarray) -> bool:
    return not np.array_equal(row, row[::-1])


def _assignment_score(
    left_rows: np.ndarray, opcodes: np.ndarray, right_rows: np.ndarray, assignment: dict[int, int]
) -> tuple[int, int]:
    supported_colors: set[int] = set()
    exact_matches = 0
    for left, opcode, right in zip(left_rows, opcodes, right_rows, strict=True):
        if not _is_asymmetric(left):
            continue
        expected = _apply_mode(left, assignment[int(opcode)])
        if np.array_equal(right, expected):
            supported_colors.add(int(opcode))
            exact_matches += 1
    return len(supported_colors), exact_matches


def _infer_assignment(grid: np.ndarray) -> dict[int, int]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    assert w % 2 == 1 and w >= 3, "grid must have an odd width with a center column"

    mid = w // 2
    left_rows = g[:, :mid]
    opcodes = g[:, mid]
    right_rows = g[:, mid + 1 :]

    opcode_colors = sorted(set(map(int, opcodes.tolist())))
    assert len(opcode_colors) == 2, "expected exactly two opcode colors in the center column"
    a, b = opcode_colors
    candidates = [{a: COPY, b: MIRROR}, {a: MIRROR, b: COPY}]

    scored = [(_assignment_score(left_rows, opcodes, right_rows, cand), cand) for cand in candidates]
    best_score = max(score for score, _cand in scored)
    winners = [cand for score, cand in scored if score == best_score]
    assert len(winners) == 1, "opcode mapping must be uniquely decodable from asymmetric intact rows"
    assert best_score[0] == 2, "both opcode colors must be evidenced by asymmetric intact rows"
    return winners[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode the center-column opcodes, then rewrite each right half from the left half.

    Input structure:
    - The grid has an odd number of columns; the middle column contains exactly two opcode colors.
    - The left and right halves have equal width, with one row-wise pattern per half.
    - Most rows are intact: for each row, the right half is either the left half copied as-is or mirrored.

    Rule:
    - Use only asymmetric left halves to infer, within this input, which opcode color means copy and which means mirror.
    - After the opcode mapping is decoded, regenerate every row's right half from its left half according to that row's center color.
    - Keep the left half and the center column unchanged.

    Key insight:
    - A damaged row can perfectly masquerade as the other operation, so the opcode meanings must be inferred globally from the intact majority rather than guessed row by row.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    h, w = g.shape
    assert h >= 1 and w >= 3 and w % 2 == 1
    mid = w // 2

    assignment = _infer_assignment(g)

    out = g.copy()
    for r in range(h):
        opcode = int(g[r, mid])
        left = g[r, :mid]
        out[r, mid + 1 :] = _apply_mode(left, assignment[opcode])
    return out


def _random_palette(bg: int, opcode_colors: tuple[int, int]) -> list[int]:
    usable = [c for c in ALL_COLORS if c not in opcode_colors]
    palette = random.sample(usable, random.randint(3, 5))
    if bg not in palette and random.random() < 0.7:
        palette[random.randrange(len(palette))] = bg
    random.shuffle(palette)
    return palette


def _sample_row(width: int, palette: list[int]) -> np.ndarray:
    return np.array([random.choice(palette) for _ in range(width)], dtype=int)


def _make_asymmetric_half(width: int, palette: list[int]) -> np.ndarray:
    for _ in range(120):
        row = _sample_row(width, palette)
        if len(set(map(int, row.tolist()))) >= 2 and _is_asymmetric(row):
            return row
    for i in range(width):
        for j in range(width):
            if i == j:
                continue
            row = np.full(width, palette[0], dtype=int)
            row[i] = palette[min(1, len(palette) - 1)]
            row[j] = palette[min(2, len(palette) - 1)]
            if _is_asymmetric(row):
                return row
    raise RuntimeError("failed to build an asymmetric half-row")


def _make_symmetric_half(width: int, palette: list[int]) -> np.ndarray:
    half = (width + 1) // 2
    for _ in range(120):
        seed = np.array([random.choice(palette) for _ in range(half)], dtype=int)
        row = np.empty(width, dtype=int)
        row[:half] = seed
        row[width - half :] = seed[::-1]
        if len(set(map(int, row.tolist()))) >= 2:
            return row
    row = np.full(width, palette[0], dtype=int)
    row[0] = palette[min(1, len(palette) - 1)]
    row[-1] = palette[min(1, len(palette) - 1)]
    return row


def _make_neither_half(left: np.ndarray, palette: list[int]) -> np.ndarray:
    copy_row = _apply_mode(left, COPY)
    mirror_row = _apply_mode(left, MIRROR)
    for _ in range(120):
        right = _sample_row(left.size, palette)
        if not np.array_equal(right, copy_row) and not np.array_equal(right, mirror_row):
            return right
    right = copy_row.copy()
    for c in range(left.size):
        for value in palette:
            if value != copy_row[c] and value != mirror_row[c]:
                right[c] = value
                if not np.array_equal(right, copy_row) and not np.array_equal(right, mirror_row):
                    return right
    raise RuntimeError("failed to build a damaged right half")


def _row_record(left: np.ndarray, opcode: int, mode: int, kind: str, palette: list[int]) -> dict[str, object]:
    expected = _apply_mode(left, mode)
    if kind in {"intact_asym", "intact_sym"}:
        given = expected.copy()
    elif kind == "damaged_disguised":
        given = _apply_mode(left, 1 - mode)
    elif kind == "damaged_wrong":
        given = _make_neither_half(left, palette)
    else:
        raise ValueError(f"unknown row kind: {kind}")
    return {
        "left": left.copy(),
        "opcode": int(opcode),
        "mode": int(mode),
        "kind": kind,
        "given_right": given,
        "expected_right": expected,
    }


def _render_grid(rows: list[dict[str, object]], width: int) -> tuple[np.ndarray, np.ndarray]:
    h = len(rows)
    grid = np.zeros((h, width * 2 + 1), dtype=int)
    solved = np.zeros_like(grid)
    for r, row in enumerate(rows):
        left = np.asarray(row["left"], dtype=int)
        opcode = int(row["opcode"])
        given_right = np.asarray(row["given_right"], dtype=int)
        expected_right = np.asarray(row["expected_right"], dtype=int)
        grid[r, :width] = left
        grid[r, width] = opcode
        grid[r, width + 1 :] = given_right
        solved[r, :width] = left
        solved[r, width] = opcode
        solved[r, width + 1 :] = expected_right
    return grid, solved


def _validate_instance(
    grid: np.ndarray, solved: np.ndarray, rows: list[dict[str, object]], assignment: dict[int, int]
) -> bool:
    if grid.ndim != 2 or solved.ndim != 2 or max(grid.shape) > 30 or max(solved.shape) > 30:
        return False
    if grid.shape != solved.shape or np.array_equal(grid, solved):
        return False

    try:
        inferred = _infer_assignment(grid)
        observed = main(grid)
    except Exception:
        return False
    if inferred != assignment or not np.array_equal(observed, solved):
        return False

    mid = grid.shape[1] // 2
    left_rows = grid[:, :mid]
    opcodes = grid[:, mid]
    right_rows = grid[:, mid + 1 :]
    colors = sorted(set(map(int, opcodes.tolist())))
    alt = {colors[0]: 1 - assignment[colors[0]], colors[1]: 1 - assignment[colors[1]]}
    true_score = _assignment_score(left_rows, opcodes, right_rows, assignment)
    alt_score = _assignment_score(left_rows, opcodes, right_rows, alt)
    if not (true_score[0] == 2 and true_score[1] >= alt_score[1] + 2):
        return False

    by_color: dict[int, dict[str, int]] = {
        color: {"intact": 0, "disguised": 0, "wrong": 0, "asym_intact": 0} for color in assignment
    }
    has_symmetric = False
    changed_rows = 0
    for row, inp_row, out_row in zip(rows, grid, solved, strict=True):
        opcode = int(row["opcode"])
        left = np.asarray(row["left"], dtype=int)
        if np.array_equal(inp_row, out_row):
            if row["kind"] not in {"intact_asym", "intact_sym"}:
                return False
        else:
            changed_rows += 1
        if row["kind"] == "intact_sym":
            has_symmetric = True
        if _is_asymmetric(left):
            if np.array_equal(inp_row[mid + 1 :], _apply_mode(left, assignment[opcode])):
                by_color[opcode]["intact"] += 1
            elif np.array_equal(inp_row[mid + 1 :], _apply_mode(left, 1 - assignment[opcode])):
                by_color[opcode]["disguised"] += 1
            else:
                by_color[opcode]["wrong"] += 1
            if row["kind"] == "intact_asym":
                by_color[opcode]["asym_intact"] += 1
    if not has_symmetric or changed_rows < 2:
        return False
    for color, stats in by_color.items():
        if stats["asym_intact"] < 1 or stats["intact"] <= stats["disguised"]:
            return False
    if sum(stats["disguised"] for stats in by_color.values()) < 2:
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    half_width = random.randint(4, 7)
    bg = random.choice(ALL_COLORS)
    opcode_colors = tuple(random.sample([c for c in ALL_COLORS if c != bg], 2))
    palette = _random_palette(bg, opcode_colors)

    first_mode = random.choice((COPY, MIRROR))
    assignment = {opcode_colors[0]: first_mode, opcode_colors[1]: 1 - first_mode}

    intact_counts = {color: random.randint(2, 3) for color in opcode_colors}
    disguised_counts = {
        color: random.randint(1, min(2, intact_counts[color] - 1)) for color in opcode_colors
    }
    symmetric_count = random.randint(1, 3)
    wrong_count = 1 if random.random() < 0.45 else 0
    if random.random() < 0.15:
        wrong_count += 1

    rows: list[dict[str, object]] = []
    for opcode in opcode_colors:
        mode = assignment[opcode]
        for _ in range(intact_counts[opcode]):
            rows.append(_row_record(_make_asymmetric_half(half_width, palette), opcode, mode, "intact_asym", palette))
        for _ in range(disguised_counts[opcode]):
            rows.append(
                _row_record(_make_asymmetric_half(half_width, palette), opcode, mode, "damaged_disguised", palette)
            )
    for _ in range(symmetric_count):
        opcode = random.choice(opcode_colors)
        rows.append(_row_record(_make_symmetric_half(half_width, palette), opcode, assignment[opcode], "intact_sym", palette))
    for _ in range(wrong_count):
        opcode = random.choice(opcode_colors)
        rows.append(_row_record(_make_asymmetric_half(half_width, palette), opcode, assignment[opcode], "damaged_wrong", palette))

    random.shuffle(rows)
    grid, solved = _render_grid(rows, half_width)
    if not _validate_instance(grid, solved, rows, assignment):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for center-column opcode rewriting")
