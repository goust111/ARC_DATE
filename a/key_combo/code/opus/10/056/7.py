from __future__ import annotations

import random
from itertools import product

import numpy as np


BG = 0
FRAME = 5
PATTERN_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
GEN_ATTEMPTS = 400


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


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
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for nr, nc in _neighbors4(cr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _line_indices(lengths: list[int]) -> list[int]:
    lines = [0]
    pos = 0
    for length in lengths:
        pos += int(length) + 1
        lines.append(pos)
    return lines


def _parse_frames(grid: np.ndarray) -> tuple[list[dict], int, int]:
    frames: list[dict] = []
    for comp in _components(grid == FRAME):
        r0, c0, r1, c1 = _bbox(comp)
        sub = grid[r0 : r1 + 1, c0 : c1 + 1] == FRAME
        row_lines = [r for r in range(sub.shape[0]) if bool(sub[r, :].all())]
        col_lines = [c for c in range(sub.shape[1]) if bool(sub[:, c].all())]
        assert len(row_lines) >= 3 and len(col_lines) >= 3, "frame must define at least 2x2 slots"
        assert row_lines[0] == 0 and row_lines[-1] == sub.shape[0] - 1
        assert col_lines[0] == 0 and col_lines[-1] == sub.shape[1] - 1
        assert all(a + 1 < b for a, b in zip(row_lines, row_lines[1:]))
        assert all(a + 1 < b for a, b in zip(col_lines, col_lines[1:]))

        expected = np.zeros_like(sub, dtype=bool)
        expected[row_lines, :] = True
        expected[:, col_lines] = True
        assert np.array_equal(sub, expected), "grey component must be a complete lattice"

        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "rows": [r0 + r for r in row_lines],
                "cols": [c0 + c for c in col_lines],
            }
        )

    frames.sort(key=lambda frame: frame["bbox"])
    assert frames, "expected at least one lattice frame"
    n_slot_rows = len(frames[0]["rows"]) - 1
    n_slot_cols = len(frames[0]["cols"]) - 1
    for frame in frames:
        assert len(frame["rows"]) - 1 == n_slot_rows
        assert len(frame["cols"]) - 1 == n_slot_cols
    return frames, n_slot_rows, n_slot_cols


def _slot_patch(grid: np.ndarray, frame: dict, slot_r: int, slot_c: int) -> np.ndarray:
    rows = frame["rows"]
    cols = frame["cols"]
    return grid[
        rows[slot_r] + 1 : rows[slot_r + 1],
        cols[slot_c] + 1 : cols[slot_c + 1],
    ]


def _cycle_for_k(slot_patches: list[np.ndarray], period: int) -> tuple[int, ...] | None:
    residue_to_color: dict[int, int] = {}
    for patch in slot_patches:
        for r, c in np.argwhere(patch != BG):
            color = int(patch[r, c])
            residue = (int(r) + int(c)) % int(period)
            if residue in residue_to_color and residue_to_color[residue] != color:
                return None
            residue_to_color[residue] = color
    if len(residue_to_color) != int(period):
        return None
    return tuple(residue_to_color[idx] for idx in range(int(period)))


def _matching_cycles(slot_patches: list[np.ndarray]) -> list[tuple[int, tuple[int, ...]]]:
    matches: list[tuple[int, tuple[int, ...]]] = []
    for period in (2, 3):
        cycle = _cycle_for_k(slot_patches, period)
        if cycle is not None:
            matches.append((period, cycle))
    return matches


def _infer_row_cycle(slot_patches: list[np.ndarray]) -> tuple[int, tuple[int, ...]]:
    matches = _matching_cycles(slot_patches)
    assert matches, "row must expose a valid diagonal cycle"
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Complete missing slot pixels by inferring each lattice-row's diagonal color cycle.

    Input structure:
    - Black background surrounds several non-overlapping grey one-pixel lattice frames.
    - Every frame is subdivided into the same logical row-by-column set of rectangular slots, though slot sizes vary by frame.
    - Inside each slot, non-grey colored pixels follow a local diagonal rule based on (local_row + local_col) modulo 2 or 3, with some slot pixels replaced by black.

    Rule:
    - Parse every grey lattice and recover its slot boundaries.
    - For each slot row of each frame, pool the visible non-black pixels from all slots in that row and test periods 2 then 3.
    - Choose the smallest period whose diagonal residue classes are color-consistent, then fill every black slot pixel from that row's residue-to-color cycle.
    - Preserve the grey frame lines, all already visible colored pixels, and the black background outside slots.

    Key insight:
    - A single slot can be missing an entire diagonal residue class, so the cycle must be inferred by comparing sibling slots within the same lattice row rather than by solving each slot independently.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    frames, n_slot_rows, n_slot_cols = _parse_frames(g)
    out = g.copy()
    for frame in frames:
        for slot_r in range(n_slot_rows):
            patches = [_slot_patch(g, frame, slot_r, slot_c) for slot_c in range(n_slot_cols)]
            period, cycle = _infer_row_cycle(patches)
            for slot_c, patch in enumerate(patches):
                dst = _slot_patch(out, frame, slot_r, slot_c)
                for r, c in np.argwhere(patch == BG):
                    dst[r, c] = cycle[(int(r) + int(c)) % int(period)]
    return out


def _row_periods(n_rows: int) -> list[int]:
    periods = [2, 3]
    while len(periods) < n_rows:
        periods.append(random.choice((2, 3)))
    random.shuffle(periods)
    return periods


def _choose_row_specs(n_rows: int) -> list[tuple[int, tuple[int, ...]]]:
    specs: list[tuple[int, tuple[int, ...]]] = []
    used: set[tuple[int, tuple[int, ...]]] = set()
    for period in _row_periods(n_rows):
        while True:
            cycle = tuple(random.sample(PATTERN_COLORS, period))
            spec = (period, cycle)
            if spec not in used:
                used.add(spec)
                specs.append(spec)
                break
    return specs


def _slot_size_bounds(n_slot_rows: int, n_slot_cols: int, n_frames: int) -> tuple[int, int]:
    max_dim = max(n_slot_rows, n_slot_cols)
    if max_dim == 4:
        return 2, 3
    if n_frames == 3:
        return 2, 4
    return 2, 5


def _frame_spec(n_slot_rows: int, n_slot_cols: int, n_frames: int) -> tuple[list[int], list[int]]:
    lo, hi = _slot_size_bounds(n_slot_rows, n_slot_cols, n_frames)
    heights = [random.randint(lo, hi) for _ in range(n_slot_rows)]
    widths = [random.randint(lo, hi) for _ in range(n_slot_cols)]
    return heights, widths


def _frame_size(spec: tuple[list[int], list[int]]) -> tuple[int, int]:
    heights, widths = spec
    return sum(heights) + len(heights) + 1, sum(widths) + len(widths) + 1


def _rect_clear(
    rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int
) -> bool:
    r0, c0, r1, c1 = rect
    for pr0, pc0, pr1, pc1 in placed:
        if not (r1 + gap < pr0 or pr1 + gap < r0 or c1 + gap < pc0 or pc1 + gap < c0):
            return False
    return True


def _place_frames(
    shape: tuple[int, int], specs: list[tuple[list[int], list[int]]]
) -> list[tuple[int, int]] | None:
    h, w = shape
    sizes = [_frame_size(spec) for spec in specs]
    order = sorted(range(len(specs)), key=lambda idx: sizes[idx][0] * sizes[idx][1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(specs)
    placed: list[tuple[int, int, int, int]] = []

    def search(i: int) -> bool:
        if i == len(order):
            return True
        idx = order[i]
        fh, fw = sizes[idx]
        candidates = [
            (r0, c0)
            for r0 in range(1, h - fh)
            for c0 in range(1, w - fw)
            if _rect_clear((r0, c0, r0 + fh - 1, c0 + fw - 1), placed, gap=1)
        ]
        random.shuffle(candidates)
        for r0, c0 in candidates:
            placed.append((r0, c0, r0 + fh - 1, c0 + fw - 1))
            positions[idx] = (r0, c0)
            if search(i + 1):
                return True
            placed.pop()
            positions[idx] = None
        return False

    if not search(0):
        return None
    return [pos for pos in positions if pos is not None]


def _draw_frame(
    grid: np.ndarray,
    out_meta: list[dict],
    top: int,
    left: int,
    spec: tuple[list[int], list[int]],
    row_specs: list[tuple[int, tuple[int, ...]]],
) -> None:
    heights, widths = spec
    row_lines = _line_indices(heights)
    col_lines = _line_indices(widths)
    abs_rows = [top + idx for idx in row_lines]
    abs_cols = [left + idx for idx in col_lines]
    grid[np.array(abs_rows), left : abs_cols[-1] + 1] = FRAME
    grid[top : abs_rows[-1] + 1, np.array(abs_cols)] = FRAME

    for slot_r, (period, cycle) in enumerate(row_specs):
        for slot_c in range(len(widths)):
            r0 = abs_rows[slot_r] + 1
            r1 = abs_rows[slot_r + 1]
            c0 = abs_cols[slot_c] + 1
            c1 = abs_cols[slot_c + 1]
            slot = grid[r0:r1, c0:c1]
            rr, cc = np.indices(slot.shape)
            slot[:, :] = np.vectorize(lambda r, c: cycle[(int(r) + int(c)) % int(period)])(rr, cc)

    out_meta.append({"rows": abs_rows, "cols": abs_cols})


def _residue_counts(shape: tuple[int, int], period: int) -> dict[int, int]:
    h, w = shape
    rr, cc = np.indices((h, w))
    residues = (rr + cc) % int(period)
    return {res: int(np.count_nonzero(residues == res)) for res in range(int(period))}


def _hide_choices(shape: tuple[int, int], period: int) -> list[int]:
    area = int(shape[0] * shape[1])
    counts = _residue_counts(shape, period)
    return [res for res, count in counts.items() if count <= area // 2]


def _row_hides(
    frame: dict, slot_r: int, n_slot_cols: int, period: int
) -> tuple[int, ...] | None:
    options: list[list[int]] = []
    present_sets: list[set[int]] = []
    for slot_c in range(n_slot_cols):
        slot = _slot_patch(np.zeros((frame["rows"][-1] + 1, frame["cols"][-1] + 1), dtype=int), frame, slot_r, slot_c)
        shape = slot.shape
        choices = _hide_choices(shape, period)
        if not choices:
            return None
        options.append(choices)
        present_sets.append(set(range(period)))

    valid: list[tuple[int, ...]] = []
    for combo in product(*options):
        visible = set()
        for residues, hidden in zip(present_sets, combo):
            visible |= residues - {hidden}
        if len(visible) == period and len(set(combo)) >= 2:
            valid.append(tuple(int(x) for x in combo))
    if not valid:
        return None
    return random.choice(valid)


def _apply_holes(inp: np.ndarray, frame: dict, slot_r: int, hides: tuple[int, ...], period: int) -> None:
    for slot_c, hidden in enumerate(hides):
        slot = _slot_patch(inp, frame, slot_r, slot_c)
        rr, cc = np.indices(slot.shape)
        slot[(rr + cc) % int(period) == int(hidden)] = BG


def _slot_visible_residues(slot: np.ndarray, period: int) -> set[int]:
    out: set[int] = set()
    for r, c in np.argwhere(slot != BG):
        out.add((int(r) + int(c)) % int(period))
    return out


def _build_instance() -> tuple[np.ndarray, np.ndarray, list[tuple[int, tuple[int, ...]]]] | None:
    n_slot_rows = random.randint(2, 4)
    n_slot_cols = random.randint(2, 4)
    n_frames = random.randint(2, 3)
    row_specs = _choose_row_specs(n_slot_rows)
    specs = [_frame_spec(n_slot_rows, n_slot_cols, n_frames) for _ in range(n_frames)]
    max_h = max(_frame_size(spec)[0] for spec in specs)
    max_w = max(_frame_size(spec)[1] for spec in specs)
    grid_h = random.randint(max(20, max_h + 2), 30)
    grid_w = random.randint(max(20, max_w + 2), 30)
    positions = _place_frames((grid_h, grid_w), specs)
    if positions is None:
        return None

    out = np.zeros((grid_h, grid_w), dtype=int)
    frame_meta: list[dict] = []
    for (top, left), spec in zip(positions, specs):
        _draw_frame(out, frame_meta, top, left, spec, row_specs)

    inp = out.copy()
    for frame in frame_meta:
        for slot_r, (period, _cycle) in enumerate(row_specs):
            hides = _row_hides(frame, slot_r, n_slot_cols, period)
            if hides is None:
                return None
            _apply_holes(inp, frame, slot_r, hides, period)

    return inp, out, row_specs


def _validate_semantics(
    inp: np.ndarray, out: np.ndarray, row_specs: list[tuple[int, tuple[int, ...]]]
) -> bool:
    if not (inp.ndim == 2 and out.ndim == 2 and inp.shape == out.shape and max(inp.shape) <= 30):
        return False
    try:
        frames, n_slot_rows, n_slot_cols = _parse_frames(inp)
        if n_slot_rows != len(row_specs):
            return False
        if len({spec for spec in row_specs}) != len(row_specs):
            return False
        if {period for period, _cycle in row_specs} != {2, 3}:
            return False

        for frame in frames:
            for slot_r, expected in enumerate(row_specs):
                period, cycle = expected
                patches = [_slot_patch(inp, frame, slot_r, slot_c) for slot_c in range(n_slot_cols)]
                matches = _matching_cycles(patches)
                if not matches or matches[0] != expected:
                    return False
                if period == 3 and len(matches) != 1:
                    return False
                for patch in patches:
                    if int(np.count_nonzero(patch)) * 2 < patch.size:
                        return False
                    if _slot_visible_residues(patch, period) == set(range(period)):
                        return False
                    if _cycle_for_k([patch], period) is not None:
                        return False
                full = [_slot_patch(out, frame, slot_r, slot_c) for slot_c in range(n_slot_cols)]
                if _matching_cycles(full)[:1] != [expected]:
                    return False

        solved = main(inp)
        return np.array_equal(solved, out)
    except AssertionError:
        return False


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        built = _build_instance()
        if built is None:
            continue
        inp, out, row_specs = built
        if _validate_semantics(inp, out, row_specs):
            return inp
    raise RuntimeError("failed to generate a valid lattice-diagonal ARC input")
