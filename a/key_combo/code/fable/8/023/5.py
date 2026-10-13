import random
import sys
from pathlib import Path

import numpy as np


for _arc_utils_dir in (Path("/home/lixinyu/snap/new/solvers"),):
    if _arc_utils_dir.is_dir():
        sys.path.append(str(_arc_utils_dir))

try:
    from arc_utils import Color
except Exception:
    class Color:
        BLACK = 0
        BLUE = 1
        RED = 2
        GREEN = 3
        YELLOW = 4
        GREY = 5
        GRAY = 5
        PINK = 6
        ORANGE = 7
        TEAL = 8
        MAROON = 9
        NOT_BLACK = [1, 2, 3, 4, 5, 6, 7, 8, 9]


BG = Color.BLACK
PALETTE = tuple(int(c) for c in Color.NOT_BLACK)

MIN_MAIN_H = 8
MAX_MAIN_H = 16
MIN_MAIN_W = 8
MAX_MAIN_W = 16
MIN_EMITTERS = 2
MAX_EMITTERS = 4
MAX_GATES_PER_RAY = 4
MAX_DISTRACTORS = 8
MAX_ATTEMPTS = 500


def _sample_spaced_positions(limit: int, count: int, min_gap: int) -> list[int]:
    positions = list(range(1, limit + 1))
    random.shuffle(positions)
    chosen: list[int] = []
    for pos in positions:
        if all(abs(pos - old) >= min_gap for old in chosen):
            chosen.append(pos)
            if len(chosen) == count:
                break
    return sorted(chosen)


def _extract_emitters(grid: np.ndarray) -> tuple[dict[int, int], dict[int, int]]:
    row_emitters = {
        int(r): int(grid[r, 0])
        for r in range(1, grid.shape[0])
        if int(grid[r, 0]) != BG
    }
    col_emitters = {
        int(c): int(grid[0, c])
        for c in range(1, grid.shape[1])
        if int(grid[0, c]) != BG
    }
    return row_emitters, col_emitters


def _analyze_grid(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    assert min(g.shape) >= 3, "grid is too small for emitters plus main area"
    assert int(g[0, 0]) == BG, "top-left corner must be black"

    row_emitters, col_emitters = _extract_emitters(g)
    assert row_emitters, "expected at least one row emitter in the left column"
    assert col_emitters, "expected at least one column emitter in the top row"

    row_open = np.zeros_like(g, dtype=bool)
    col_open = np.zeros_like(g, dtype=bool)
    row_gate = np.zeros_like(g, dtype=bool)
    col_gate = np.zeros_like(g, dtype=bool)

    for c, color in col_emitters.items():
        is_open = False
        for r in range(1, g.shape[0]):
            if int(g[r, c]) == color:
                col_gate[r, c] = True
                is_open = not is_open
            elif is_open:
                col_open[r, c] = True

    for r, color in row_emitters.items():
        is_open = False
        for c in range(1, g.shape[1]):
            if int(g[r, c]) == color:
                row_gate[r, c] = True
                is_open = not is_open
            elif is_open:
                row_open[r, c] = True

    return {
        "grid": g,
        "row_emitters": row_emitters,
        "col_emitters": col_emitters,
        "row_open": row_open,
        "col_open": col_open,
        "row_gate": row_gate,
        "col_gate": col_gate,
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Extend top and left emitter rays through same-color gate pixels, then cancel row/column overlaps back to black.

    Input structure:
    - The top row contains non-black column emitters and the left column contains non-black row emitters; the top-left corner is black.
    - The interior region has a black background plus scattered colored pixels.
    - Any interior pixel that matches its aligned emitter's color is a gate for that row ray or column ray.

    Rule:
    - Each column emitter scans downward from row 1 with state initially off; every same-color pixel in that column toggles the state, and only non-gate cells traversed while on receive a column-open mark.
    - Each row emitter scans rightward from column 1 under the analogous rule, producing row-open marks.
    - Output keeps emitters fixed; interior cells with only a column-open mark become the column emitter's color, cells with only a row-open mark become the row emitter's color, cells with both marks become black, and unmarked cells stay unchanged.

    Key insight:
    - Same-color aligned pixels are not paint targets but parity switches, so the visible colored spans are the alternating intervals between gates, while row/column overlaps disappear by XOR-like cancellation.
    """
    info = _analyze_grid(grid)
    g = info["grid"]
    row_emitters = info["row_emitters"]
    col_emitters = info["col_emitters"]
    row_open = info["row_open"]
    col_open = info["col_open"]

    out = g.copy()
    for r in range(1, g.shape[0]):
        row_color = row_emitters.get(r)
        for c in range(1, g.shape[1]):
            if row_open[r, c] and col_open[r, c]:
                out[r, c] = BG
            elif col_open[r, c]:
                out[r, c] = int(col_emitters[c])
            elif row_open[r, c]:
                out[r, c] = int(row_color)
    return out


def _render_base_scene(
    main_h: int,
    main_w: int,
    row_emitters: dict[int, int],
    col_emitters: dict[int, int],
    row_gates: dict[int, list[int]],
    col_gates: dict[int, list[int]],
) -> np.ndarray:
    grid = np.full((main_h + 1, main_w + 1), BG, dtype=int)
    for r, color in row_emitters.items():
        grid[r, 0] = int(color)
    for c, color in col_emitters.items():
        grid[0, c] = int(color)
    for r, cols in row_gates.items():
        for c in cols:
            grid[r, c] = int(row_emitters[r])
    for c, rows in col_gates.items():
        for r in rows:
            assert grid[r, c] == BG
            grid[r, c] = int(col_emitters[c])
    return grid


def _sample_counts(row_keys: list[int], col_keys: list[int], anchor_row: int, anchor_col: int) -> tuple[dict[int, int], dict[int, int]]:
    count_options = [0, 1, 2, 3, 4]
    weights = [1, 4, 4, 3, 2]

    row_counts = {r: random.choices(count_options, weights=weights, k=1)[0] for r in row_keys}
    col_counts = {c: random.choices(count_options, weights=weights, k=1)[0] for c in col_keys}

    row_counts[anchor_row] = random.choice([1, 2, 3, 4])
    col_counts[anchor_col] = random.choice([1, 2, 3, 4])

    complex_is_row = random.choice([True, False])
    if complex_is_row:
        row_counts[random.choice(row_keys)] = random.choice([3, 4])
    else:
        col_counts[random.choice(col_keys)] = random.choice([3, 4])

    if all(v == 0 for v in row_counts.values()):
        row_counts[random.choice(row_keys)] = random.choice([1, 2, 3])
    if all(v == 0 for v in col_counts.values()):
        col_counts[random.choice(col_keys)] = random.choice([1, 2, 3])
    return row_counts, col_counts


def _sample_anchor_positions(anchor: int, limit: int, count: int, blocked: set[int] | None = None) -> list[int] | None:
    blocked = set() if blocked is None else set(blocked)
    before = [p for p in range(1, anchor) if p not in blocked]
    after = [p for p in range(anchor + 1, limit + 1) if p not in blocked]
    if not before or len(after) < count - 1:
        return None
    positions = [random.choice(before)]
    if count > 1:
        positions.extend(random.sample(after, count - 1))
    return sorted(positions)


def _sample_regular_positions(limit: int, count: int, blocked: set[int] | None = None) -> list[int] | None:
    blocked = set() if blocked is None else set(blocked)
    choices = [p for p in range(1, limit + 1) if p not in blocked]
    if count > len(choices):
        return None
    return sorted(random.sample(choices, count))


def _sample_gates(
    main_h: int,
    main_w: int,
    row_keys: list[int],
    col_keys: list[int],
    row_counts: dict[int, int],
    col_counts: dict[int, int],
    anchor_row: int,
    anchor_col: int,
) -> tuple[dict[int, list[int]], dict[int, list[int]]] | None:
    col_gates: dict[int, list[int]] = {}
    row_gates: dict[int, list[int]] = {}
    occupied: set[tuple[int, int]] = set()

    for c in random.sample(col_keys, len(col_keys)):
        count = int(col_counts[c])
        if count == 0:
            col_gates[c] = []
            continue
        if count > main_h:
            return None
        if c == anchor_col:
            rows = _sample_anchor_positions(anchor_row, main_h, count)
        else:
            rows = _sample_regular_positions(main_h, count)
        if rows is None:
            return None
        for r in rows:
            occupied.add((r, c))
        col_gates[c] = rows

    for r in random.sample(row_keys, len(row_keys)):
        count = int(row_counts[r])
        if count == 0:
            row_gates[r] = []
            continue
        blocked = {c for rr, c in occupied if rr == r}
        if count > main_w - len(blocked):
            return None
        if r == anchor_row:
            cols = _sample_anchor_positions(anchor_col, main_w, count, blocked)
        else:
            cols = _sample_regular_positions(main_w, count, blocked)
        if cols is None:
            return None
        for c in cols:
            occupied.add((r, c))
        row_gates[r] = cols

    return row_gates, col_gates


def _safe_colors_for_cell(r: int, c: int, row_emitters: dict[int, int], col_emitters: dict[int, int]) -> list[int]:
    forbidden = {row_emitters.get(r, BG), col_emitters.get(c, BG), BG}
    return [color for color in PALETTE if color not in forbidden]


def _pick_witness(mask: np.ndarray) -> tuple[int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r, c = random.choice(coords.tolist())
    return int(r), int(c)


def _add_safe_pixel(
    grid: np.ndarray,
    pos: tuple[int, int],
    row_emitters: dict[int, int],
    col_emitters: dict[int, int],
    prefer_emitter_color: bool,
) -> bool:
    r, c = pos
    safe = _safe_colors_for_cell(r, c, row_emitters, col_emitters)
    if not safe:
        return False
    if prefer_emitter_color:
        emitter_colors = set(row_emitters.values()) | set(col_emitters.values())
        reuse = [color for color in safe if color in emitter_colors]
        if reuse:
            safe = reuse
    grid[r, c] = int(random.choice(safe))
    return True


def _decorate_scene(base: np.ndarray, info: dict[str, object]) -> np.ndarray | None:
    grid = base.copy()
    row_emitters = info["row_emitters"]
    col_emitters = info["col_emitters"]
    row_open = info["row_open"]
    col_open = info["col_open"]
    row_gate = info["row_gate"]
    col_gate = info["col_gate"]

    gate_union = row_gate | col_gate
    row_only = row_open & ~col_open
    col_only = col_open & ~row_open
    both = row_open & col_open
    unmarked = ~gate_union & ~row_open & ~col_open
    unmarked[0, :] = False
    unmarked[:, 0] = False

    witness_specs = [
        (row_only, False),
        (col_only, False),
        (both, False),
        (unmarked, False),
        (unmarked, True),
    ]
    used: set[tuple[int, int]] = set()
    for mask, reuse_color in witness_specs:
        available = mask.copy()
        for pos in used:
            available[pos] = False
        pos = _pick_witness(available)
        if pos is None:
            return None
        used.add(pos)
        if not _add_safe_pixel(grid, pos, row_emitters, col_emitters, prefer_emitter_color=reuse_color):
            return None

    remaining = [
        (r, c)
        for r in range(1, grid.shape[0])
        for c in range(1, grid.shape[1])
        if (r, c) not in used and not gate_union[r, c]
    ]
    random.shuffle(remaining)
    extra_count = random.randint(2, MAX_DISTRACTORS)
    for r, c in remaining[:extra_count]:
        if random.random() < 0.5:
            grid[r, c] = BG
            continue
        if not _add_safe_pixel(
            grid,
            (r, c),
            row_emitters,
            col_emitters,
            prefer_emitter_color=(random.random() < 0.35),
        ):
            return None
    return grid


def _scene_is_rich(grid: np.ndarray) -> bool:
    info = _analyze_grid(grid)
    row_open = info["row_open"]
    col_open = info["col_open"]
    row_gate = info["row_gate"]
    col_gate = info["col_gate"]
    row_emitters = info["row_emitters"]
    col_emitters = info["col_emitters"]

    gate_union = row_gate | col_gate
    row_only = row_open & ~col_open
    col_only = col_open & ~row_open
    both = row_open & col_open
    marked = row_open | col_open

    if gate_union.sum() < 5:
        return False
    if not row_only.any() or not col_only.any() or not both.any():
        return False
    if np.any(gate_union & marked):
        return False
    if marked.sum() < 8:
        return False

    main_area = (grid.shape[0] - 1) * (grid.shape[1] - 1)
    if int(marked.sum()) > int(0.55 * main_area):
        return False
    if int(both.sum()) > int(0.45 * marked.sum()):
        return False

    gate_counts = [int(row_gate[r, 1:].sum()) for r in row_emitters] + [int(col_gate[1:, c].sum()) for c in col_emitters]
    positive = [count for count in gate_counts if count > 0]
    if not positive:
        return False
    if not any(count % 2 == 1 for count in positive):
        return False
    if not any(count % 2 == 0 for count in positive):
        return False
    if not any(count >= 3 for count in positive):
        return False

    return True


def _has_same_color_distractor(grid: np.ndarray) -> bool:
    info = _analyze_grid(grid)
    gate_union = info["row_gate"] | info["col_gate"]
    emitter_colors = set(info["row_emitters"].values()) | set(info["col_emitters"].values())
    for r in range(1, grid.shape[0]):
        for c in range(1, grid.shape[1]):
            if gate_union[r, c]:
                continue
            cell = int(grid[r, c])
            if cell != BG and cell in emitter_colors:
                return True
    return False


def _sample_scene() -> np.ndarray | None:
    main_h = random.randint(MIN_MAIN_H, MAX_MAIN_H)
    main_w = random.randint(MIN_MAIN_W, MAX_MAIN_W)

    row_count = random.randint(MIN_EMITTERS, min(MAX_EMITTERS, main_h - 1))
    col_count = random.randint(MIN_EMITTERS, min(MAX_EMITTERS, main_w - 1))
    min_gap = random.choice([2, 3])
    row_keys = _sample_spaced_positions(main_h, row_count, min_gap)
    col_keys = _sample_spaced_positions(main_w, col_count, min_gap)
    if len(row_keys) < row_count or len(col_keys) < col_count:
        return None

    inner_rows = [r for r in row_keys if 2 <= r <= main_h - 1]
    inner_cols = [c for c in col_keys if 2 <= c <= main_w - 1]
    if not inner_rows or not inner_cols:
        return None
    anchor_row = random.choice(inner_rows)
    anchor_col = random.choice(inner_cols)

    palette = random.sample(PALETTE, random.randint(3, 6))
    row_emitters = {r: random.choice(palette) for r in row_keys}
    col_emitters = {c: random.choice(palette) for c in col_keys}
    if len(set(row_emitters.values()) | set(col_emitters.values())) < 2:
        return None

    row_counts, col_counts = _sample_counts(row_keys, col_keys, anchor_row, anchor_col)
    sampled = _sample_gates(main_h, main_w, row_keys, col_keys, row_counts, col_counts, anchor_row, anchor_col)
    if sampled is None:
        return None
    row_gates, col_gates = sampled

    base = _render_base_scene(main_h, main_w, row_emitters, col_emitters, row_gates, col_gates)
    if not _scene_is_rich(base):
        return None

    decorated = _decorate_scene(base, _analyze_grid(base))
    if decorated is None:
        return None
    if not _scene_is_rich(decorated) or not _has_same_color_distractor(decorated):
        return None
    return decorated


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _sample_scene()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for fable/8/5")


def _has_influential_gate(inp: np.ndarray, out: np.ndarray, gate_mask: np.ndarray) -> bool:
    coords = [tuple(map(int, rc)) for rc in np.argwhere(gate_mask)]
    random.shuffle(coords)
    for r, c in coords:
        mutated = inp.copy()
        mutated[r, c] = BG
        if not np.array_equal(main(mutated), out):
            return True
    return False


def _has_influential_emitter(inp: np.ndarray, out: np.ndarray, emitters: dict[int, int], axis: str) -> bool:
    items = list(emitters.items())
    random.shuffle(items)
    for idx, color in items:
        alt = next(candidate for candidate in PALETTE if candidate != color)
        mutated = inp.copy()
        if axis == "row":
            mutated[idx, 0] = alt
        else:
            mutated[0, idx] = alt
        if not np.array_equal(main(mutated), out):
            return True
    return False


def _validate_generated_example(inp: np.ndarray, out: np.ndarray) -> bool:
    if not isinstance(inp, np.ndarray) or not isinstance(out, np.ndarray):
        return False
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False

    info = _analyze_grid(inp)
    row_emitters = info["row_emitters"]
    col_emitters = info["col_emitters"]
    row_open = info["row_open"]
    col_open = info["col_open"]
    row_gate = info["row_gate"]
    col_gate = info["col_gate"]
    gate_union = row_gate | col_gate

    expected = inp.copy()
    for r in range(1, inp.shape[0]):
        for c in range(1, inp.shape[1]):
            if row_open[r, c] and col_open[r, c]:
                expected[r, c] = BG
            elif col_open[r, c]:
                expected[r, c] = int(col_emitters[c])
            elif row_open[r, c]:
                expected[r, c] = int(row_emitters[r])
    if not np.array_equal(expected, out):
        return False
    if not _scene_is_rich(inp) or not _has_same_color_distractor(inp):
        return False

    if not np.any(row_gate) or not np.any(col_gate):
        return False
    if not _has_influential_gate(inp, out, row_gate):
        return False
    if not _has_influential_gate(inp, out, col_gate):
        return False
    if not _has_influential_emitter(inp, out, row_emitters, "row"):
        return False
    if not _has_influential_emitter(inp, out, col_emitters, "col"):
        return False
    return True
