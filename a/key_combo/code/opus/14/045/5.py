from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
MAROON = 9

TEMPLATE_A_ROW = 0
TEMPLATE_B_ROW = 2
PARTIAL_START_ROW = 5

PREFIX_COLORS = [BLUE, RED, GREEN, YELLOW, GREY, PINK, ORANGE, MAROON]
ROW_COLOR_PAIRS = [
    (GREY, PINK),
    (GREY, ORANGE),
    (GREY, MAROON),
    (PINK, ORANGE),
    (PINK, MAROON),
    (ORANGE, MAROON),
    (BLUE, GREY),
    (RED, PINK),
    (GREEN, ORANGE),
    (YELLOW, MAROON),
]

GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _components4(mask: np.ndarray) -> tuple[np.ndarray, int]:
    h, w = mask.shape
    labels = np.zeros((h, w), dtype=int)
    comp_id = 0
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or labels[r, c] != 0:
                continue
            comp_id += 1
            q: deque[tuple[int, int]] = deque([(r, c)])
            labels[r, c] = comp_id
            while q:
                rr, cc = q.popleft()
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and labels[nr, nc] == 0:
                        labels[nr, nc] = comp_id
                        q.append((nr, nc))
    return labels, comp_id


def _prefix_columns(row: np.ndarray) -> np.ndarray:
    end = 0
    while end < len(row) and int(row[end]) not in (BLACK, TEAL):
        end += 1
    if end == 0:
        return np.array([], dtype=int)
    tail = np.asarray(row[end:], dtype=int)
    if np.any((tail != BLACK) & (tail != TEAL)):
        raise AssertionError("partial rows may only have a left non-teal prefix")
    return np.arange(end, dtype=int)


def _maybe_prefix_columns(row: np.ndarray) -> np.ndarray:
    end = 0
    while end < len(row) and int(row[end]) not in (BLACK, TEAL):
        end += 1
    if end == 0:
        return np.array([], dtype=int)
    tail = np.asarray(row[end:], dtype=int)
    if np.any((tail != BLACK) & (tail != TEAL)):
        return np.array([], dtype=int)
    return np.arange(end, dtype=int)


def _template_rows(grid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    template_a = np.asarray(grid[TEMPLATE_A_ROW], dtype=int).copy()
    template_b = np.asarray(grid[TEMPLATE_B_ROW], dtype=int).copy()
    template_a[template_a == TEAL] = BLACK
    template_b[template_b == TEAL] = BLACK
    return template_a, template_b


def _partial_rows(grid: np.ndarray) -> list[int]:
    rows: list[int] = []
    for r in range(PARTIAL_START_ROW, grid.shape[0]):
        row = np.asarray(grid[r], dtype=int)
        prefix = _maybe_prefix_columns(row)
        if len(prefix) == 0:
            continue
        vals = {int(v) for v in row[prefix]}
        if TEAL in vals:
            raise AssertionError("partial rows may not contain bridge pixels")
        if len(vals) != 2:
            raise AssertionError("each partial row must use exactly two colors in its prefix")
        rows.append(r)
    return rows


def _decode_mapping(template: np.ndarray, row: np.ndarray, prefix_cols: np.ndarray) -> dict[int, int]:
    mapping: dict[int, int] = {}
    seen_template_colors = {int(template[c]) for c in prefix_cols}
    if BLACK in seen_template_colors or len(seen_template_colors) != 2:
        raise AssertionError("prefix must only cover the two template colors")
    for c in map(int, prefix_cols):
        role = int(template[c])
        color = int(row[c])
        if role in mapping and mapping[role] != color:
            raise AssertionError("template color must map consistently")
        mapping[role] = color
    if len(set(mapping.values())) != 2 or set(mapping) != seen_template_colors:
        raise AssertionError("prefix must define a bijection on the template colors")
    return mapping


def _fill_from_template(template: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.array(template, copy=True)
    for c, value in enumerate(template):
        value = int(value)
        out[c] = BLACK if value == BLACK else mapping[value]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Route each partial row to the correct template by 4-connected teal bridges, then complete it via prefix decoding.

    Input structure:
    - Black background with two wide two-color template rows near the top, one based on red/blue and one based on green/yellow.
    - Lower rows contain contiguous left prefixes made of exactly two non-black colors, with empty rows between them.
    - Single-pixel-wide teal bridge paths connect each partial row upward to exactly one template region.

    Rule:
    - Build 4-connected components over all non-black cells.
    - For each partial row, find the component containing that row and determine whether it reaches the upper red/blue template row or the upper green/yellow template row.
    - Decode the row's two-color mapping by pairing each prefix cell with the chosen template color at the same column, then replay the whole template row through that mapping.
    - Keep the templates and all teal bridge pixels unchanged.

    Key insight:
    - The prefix alone is intentionally compatible with both templates; only the bridge connectivity tells you which full template sequence to replay.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    template_a, template_b = _template_rows(g)
    assert set(map(int, template_a)) == {RED, BLUE}, "row 0 must be the red/blue template"
    assert {int(v) for v in template_b if int(v) != BLACK} == {GREEN, YELLOW}, "row 2 must be the green/yellow template"

    labels, n_comp = _components4(g != BLACK)
    assert n_comp > 0

    comp_has_a = np.zeros(n_comp + 1, dtype=bool)
    comp_has_b = np.zeros(n_comp + 1, dtype=bool)
    for c in range(g.shape[1]):
        lid = int(labels[TEMPLATE_A_ROW, c])
        if lid and int(g[TEMPLATE_A_ROW, c]) in (RED, BLUE):
            comp_has_a[lid] = True
        lid = int(labels[TEMPLATE_B_ROW, c])
        if lid and int(g[TEMPLATE_B_ROW, c]) in (GREEN, YELLOW):
            comp_has_b[lid] = True

    out = g.copy()
    partial_rows = _partial_rows(g)
    assert partial_rows, "expected at least one partial row"
    for r in partial_rows:
        prefix_cols = _prefix_columns(g[r])
        comp_ids = {int(labels[r, c]) for c in prefix_cols}
        assert len(comp_ids) == 1
        comp_id = next(iter(comp_ids))
        has_a = bool(comp_has_a[comp_id])
        has_b = bool(comp_has_b[comp_id])
        assert has_a ^ has_b, "each partial row must connect to exactly one template"

        template = template_a if has_a else template_b
        mapping = _decode_mapping(template, g[r], prefix_cols)
        out[r] = _fill_from_template(template, mapping)

    return out


def _build_role_sequences(width: int, shared_len: int, rng: random.Random) -> tuple[list[int], list[int]]:
    prefix = [0, 1]
    while len(prefix) < shared_len:
        prefix.append(rng.randrange(2))
    if len(set(prefix)) != 2:
        prefix[1] = 1 - prefix[0]

    suffix_len = width - shared_len
    assert suffix_len >= 3
    while True:
        suffix_a = [rng.randrange(2) for _ in range(suffix_len)]
        suffix_b = [rng.randrange(2) for _ in range(suffix_len)]
        if suffix_a != suffix_b and len(set(suffix_a)) == 2 and len(set(suffix_b)) == 2:
            break
    return prefix + suffix_a, prefix + suffix_b


def _render_template(roles: list[int], colors: tuple[int, int], holes: set[int] | None = None) -> np.ndarray:
    holes = holes or set()
    row = np.zeros(len(roles), dtype=int)
    for c, role in enumerate(roles):
        if c in holes:
            row[c] = BLACK
        else:
            row[c] = int(colors[role])
    return row


def _candidate_partial_rows(count: int, rng: random.Random) -> list[int]:
    row = PARTIAL_START_ROW
    rows: list[int] = []
    for _ in range(count):
        rows.append(row)
        row += 3 + rng.randrange(2)
    return rows


def _attach_path(grid: np.ndarray, row: int, attach_col: int, lane_col: int, target: str) -> None:
    bridge_row = row - 1
    for c in range(min(attach_col, lane_col), max(attach_col, lane_col) + 1):
        grid[bridge_row, c] = TEAL
    top_row = 1 if target == "A" else 3
    for r in range(top_row, bridge_row + 1):
        grid[r, lane_col] = TEAL


def _template_mapping_compatible(template: np.ndarray, prefix: np.ndarray) -> bool:
    prefix_cols = np.arange(len(prefix), dtype=int)
    try:
        _decode_mapping(template, prefix, prefix_cols)
    except AssertionError:
        return False
    return True


def _row_output_for_template(template: np.ndarray, prefix: np.ndarray) -> np.ndarray:
    prefix_cols = np.arange(len(prefix), dtype=int)
    mapping = _decode_mapping(template, prefix, prefix_cols)
    return _fill_from_template(template, mapping)


def _validate_instance(grid: np.ndarray, meta: dict) -> bool:
    h, w = grid.shape
    if max(h, w) > 30:
        return False

    partial_rows = _partial_rows(grid)
    if partial_rows != meta["rows"]:
        return False

    solved = main(grid)
    if solved.shape != grid.shape:
        return False

    template_a, template_b = _template_rows(grid)

    # Every prefix must be compatible with both templates, but the completed rows must differ.
    for r in partial_rows:
        prefix_cols = _prefix_columns(grid[r])
        prefix = np.asarray(grid[r, prefix_cols], dtype=int)
        if not _template_mapping_compatible(template_a[: len(prefix)], prefix):
            return False
        if not _template_mapping_compatible(template_b[: len(prefix)], prefix):
            return False
        out_a = _row_output_for_template(template_a, prefix)
        out_b = _row_output_for_template(template_b, prefix)
        if np.array_equal(out_a, out_b):
            return False

    # Route balance: both templates must appear in the same grid.
    if len(set(meta["targets"])) != 2:
        return False

    # Non-overlapping lanes and B holes exactly on A lanes.
    a_lanes = {lane for lane, target in zip(meta["lanes"], meta["targets"]) if target == "A"}
    if not a_lanes.issubset(set(meta["holes_b"])):
        return False

    # Check component routing semantics explicitly.
    labels, _ = _components4(grid != BLACK)
    for r, target in zip(partial_rows, meta["targets"]):
        prefix_cols = _prefix_columns(grid[r])
        comp_id = int(labels[r, int(prefix_cols[0])])
        touches_a = np.any((labels[TEMPLATE_A_ROW] == comp_id) & np.isin(grid[TEMPLATE_A_ROW], [RED, BLUE]))
        touches_b = np.any((labels[TEMPLATE_B_ROW] == comp_id) & np.isin(grid[TEMPLATE_B_ROW], [GREEN, YELLOW]))
        if target == "A" and (not touches_a or touches_b):
            return False
        if target == "B" and (touches_a or not touches_b):
            return False

    # Distinct completed rows avoid many-to-one degeneracy inside one example.
    completed = [tuple(map(int, solved[r])) for r in partial_rows]
    if len(completed) != len(set(completed)):
        return False

    return True


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    row_count = rng.randint(4, 5)
    prefix_min = rng.randint(2, 4)
    prefix_lengths = list(range(prefix_min, prefix_min + row_count))
    max_prefix = prefix_lengths[-1]

    width = rng.randint(max_prefix + 3 * row_count + 4, max_prefix + 3 * row_count + 8)
    if width > 30:
        return None

    rows = _candidate_partial_rows(row_count, rng)
    height = rows[-1] + 2
    if height > 26:
        return None

    lanes = [max_prefix + 2 + 3 * i for i in range(row_count)]
    if lanes[-1] >= width:
        return None

    shared_len = max_prefix
    if shared_len + 3 >= width:
        return None
    roles_a, roles_b = _build_role_sequences(width, shared_len, rng)

    targets = ["A"] * rng.randint(2, min(3, row_count - 1)) + ["B"] * 1
    while len(targets) < row_count:
        targets.append(rng.choice(["A", "B"]))
    rng.shuffle(targets)
    if len(set(targets)) != 2:
        return None

    holes_b: set[int] = set()
    for lane, target in zip(lanes, targets):
        if target == "A":
            holes_b.update({lane - 1, lane, lane + 1})
    template_a = _render_template(roles_a, (RED, BLUE))
    template_b = _render_template(roles_b, (GREEN, YELLOW), holes=holes_b)

    grid = np.zeros((height, width), dtype=int)
    grid[TEMPLATE_A_ROW] = template_a
    grid[TEMPLATE_B_ROW] = template_b

    pair_pool = ROW_COLOR_PAIRS[:]
    rng.shuffle(pair_pool)

    for idx, (r, prefix_len, lane_col, target) in enumerate(zip(rows, prefix_lengths, lanes, targets)):
        template = template_a if target == "A" else template_b
        if BLACK in template[:prefix_len]:
            return None

        pair = pair_pool[idx]
        mapping = {
            (RED if target == "A" else GREEN): pair[0],
            (BLUE if target == "A" else YELLOW): pair[1],
        }
        prefix = np.array([mapping[int(v)] for v in template[:prefix_len]], dtype=int)
        if len(set(map(int, prefix))) != 2:
            return None
        grid[r, :prefix_len] = prefix
        attach_col = rng.randrange(1, prefix_len)
        _attach_path(grid, r, attach_col, lane_col, target)

    meta = {
        "rows": rows,
        "targets": targets,
        "lanes": lanes,
        "holes_b": sorted(holes_b),
    }
    if not _validate_instance(grid, meta):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        seed = random.randrange(10**9)
        rng = random.Random(seed)
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid bridge-routed template-completion input")
