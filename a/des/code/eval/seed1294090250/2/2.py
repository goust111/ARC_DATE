from __future__ import annotations

from collections import Counter, deque
import math

import numpy as np


BG = 0
LATTICE_COLOR = 2
ROW_COLOR_POOL = np.array([1, 3, 4, 5, 6, 7, 8, 9], dtype=int)
MAX_SIDE = 30
MAX_GENERATION_ATTEMPTS = 500


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    coords = np.argwhere(mask)
    assert len(coords) > 0
    (r0, c0), (r1, c1) = coords.min(axis=0), coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _neighbors4(r: int, c: int, height: int, width: int):
    if r > 0:
        yield r - 1, c
    if r + 1 < height:
        yield r + 1, c
    if c > 0:
        yield r, c - 1
    if c + 1 < width:
        yield r, c + 1


def _most_common_color(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _cell_spans(separators: list[int]) -> list[tuple[int, int]]:
    return [(separators[i] + 1, separators[i + 1] - 1) for i in range(len(separators) - 1)]


def _find_lattice(grid: np.ndarray, bg: int) -> dict[str, object]:
    best: dict[str, object] | None = None
    best_score: tuple[int, int] | None = None

    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue

        mask = grid == color
        r0, c0, r1, c1 = _bbox(mask)
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]

        full_rows = [r0 + rr for rr in range(patch.shape[0]) if np.all(patch[rr, :] == color)]
        full_cols = [c0 + cc for cc in range(patch.shape[1]) if np.all(patch[:, cc] == color)]
        if len(full_rows) < 3 or len(full_cols) < 3:
            continue
        if full_rows[0] != r0 or full_rows[-1] != r1 or full_cols[0] != c0 or full_cols[-1] != c1:
            continue

        row_steps = np.diff(full_rows)
        col_steps = np.diff(full_cols)
        if len(set(map(int, row_steps))) != 1 or len(set(map(int, col_steps))) != 1:
            continue
        if int(row_steps[0]) <= 1 or int(col_steps[0]) <= 1:
            continue

        row_set = {r - r0 for r in full_rows}
        col_set = {c - c0 for c in full_cols}
        ok = True
        for rr, cc in np.argwhere(patch == color):
            if int(rr) not in row_set and int(cc) not in col_set:
                ok = False
                break
        if not ok:
            continue

        score = ((len(full_rows) - 1) * (len(full_cols) - 1), patch.shape[0] * patch.shape[1])
        if best is None or score > best_score:
            best = {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "sep_rows": full_rows,
                "sep_cols": full_cols,
                "cell_h": int(row_steps[0] - 1),
                "cell_w": int(col_steps[0] - 1),
            }
            best_score = score

    assert best is not None, "failed to identify lattice skeleton"
    return best


def _extract_row_specs(
    grid: np.ndarray, bg: int, lattice_color: int, sep_rows: list[int], sep_cols: list[int]
) -> list[dict[str, object]]:
    row_spans = _cell_spans(sep_rows)
    col_spans = _cell_spans(sep_cols)
    specs: list[dict[str, object]] = []

    for ri, (r0, r1) in enumerate(row_spans):
        row_color: int | None = None
        empty_cols: list[int] = []
        for ci, (c0, c1) in enumerate(col_spans):
            cell = grid[r0 : r1 + 1, c0 : c1 + 1]
            vals = {
                int(v)
                for v in np.unique(cell)
                if int(v) != bg and int(v) != lattice_color
            }
            assert lattice_color not in vals
            if not vals:
                empty_cols.append(ci)
                continue
            assert len(vals) == 1, "each filled cell must be monochrome with bg holes only"
            color = next(iter(vals))
            if row_color is None:
                row_color = color
            else:
                assert row_color == color, "every lattice row must use one color"

        assert row_color is not None, "each lattice row needs at least one filled cell"
        specs.append({"row_index": ri, "row_color": row_color, "empty_cols": empty_cols})

    row_colors = [int(spec["row_color"]) for spec in specs]
    assert len(set(row_colors)) == len(row_colors), "row colors must be distinct"
    return specs


def _extract_strips(
    grid: np.ndarray, bg: int, lattice_bbox: tuple[int, int, int, int], cell_h: int, cell_w: int
) -> list[dict[str, object]]:
    height, width = grid.shape
    r0, c0, r1, c1 = lattice_bbox
    visited = np.zeros_like(grid, dtype=bool)
    strips: list[dict[str, object]] = []

    def outside_lattice(rr: int, cc: int) -> bool:
        return not (r0 <= rr <= r1 and c0 <= cc <= c1)

    for sr in range(height):
        for sc in range(width):
            if visited[sr, sc] or not outside_lattice(sr, sc) or int(grid[sr, sc]) == bg:
                continue

            color = int(grid[sr, sc])
            q = deque([(sr, sc)])
            visited[sr, sc] = True
            coords: list[tuple[int, int]] = []

            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc, height, width):
                    if visited[nr, nc] or not outside_lattice(nr, nc):
                        continue
                    if int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            rows = [p[0] for p in coords]
            cols = [p[1] for p in coords]
            br0, br1 = min(rows), max(rows)
            bc0, bc1 = min(cols), max(cols)
            patch = grid[br0 : br1 + 1, bc0 : bc1 + 1]

            assert patch.shape == (cell_h, cell_w), "strip bbox must match cell size"
            vals = {int(v) for v in np.unique(patch)}
            assert vals <= {bg, color}, "strip must be monochrome with bg holes only"

            hole_points = [
                (int(rr), int(cc))
                for rr, cc in np.argwhere(patch == bg)
                if 0 < int(rr) < cell_h - 1 and 0 < int(cc) < cell_w - 1
            ]
            assert hole_points, "strip must have at least one interior hole pixel"
            hole_signature = tuple(sorted({cc for _, cc in hole_points}))

            strips.append(
                {
                    "color": color,
                    "bbox": (br0, bc0, br1, bc1),
                    "patch": patch.copy(),
                    "hole_signature": hole_signature,
                }
            )

    return strips


def main(grid: np.ndarray) -> np.ndarray:
    """Place coded monochrome strips back into the correct lattice cells."""
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _most_common_color(g)
    lattice = _find_lattice(g, bg)
    lattice_color = int(lattice["color"])
    sep_rows = list(map(int, lattice["sep_rows"]))
    sep_cols = list(map(int, lattice["sep_cols"]))
    cell_h = int(lattice["cell_h"])
    cell_w = int(lattice["cell_w"])

    row_specs = _extract_row_specs(g, bg, lattice_color, sep_rows, sep_cols)
    strips = _extract_strips(g, bg, tuple(map(int, lattice["bbox"])), cell_h, cell_w)

    by_color: dict[int, list[dict[str, object]]] = {}
    for strip in strips:
        by_color.setdefault(int(strip["color"]), []).append(strip)

    out = g.copy()
    for strip in strips:
        br0, bc0, br1, bc1 = map(int, strip["bbox"])
        out[br0 : br1 + 1, bc0 : bc1 + 1] = bg

    row_spans = _cell_spans(sep_rows)
    col_spans = _cell_spans(sep_cols)
    used = 0
    for spec in row_specs:
        ri = int(spec["row_index"])
        row_color = int(spec["row_color"])
        empty_cols = list(map(int, spec["empty_cols"]))
        row_strips = by_color.get(row_color, [])
        assert len(row_strips) == len(empty_cols), "strip count must match empty cells in the row"

        if len(empty_cols) <= 1:
            ordered = row_strips
        else:
            signatures = [tuple(strip["hole_signature"]) for strip in row_strips]
            assert len(set(signatures)) == len(signatures), "row strip codes must be unique"
            ordered = sorted(row_strips, key=lambda strip: tuple(strip["hole_signature"]))

        for ci, strip in zip(sorted(empty_cols), ordered):
            rr0, rr1 = row_spans[ri]
            cc0, cc1 = col_spans[ci]
            out[rr0 : rr1 + 1, cc0 : cc1 + 1] = np.asarray(strip["patch"], dtype=int)
            used += 1

    assert used == len(strips)
    return out


def _paste(canvas: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    h, w = patch.shape
    canvas[top : top + h, left : left + w] = patch


def _make_patch(
    rng: np.random.Generator,
    color: int,
    cell_h: int,
    cell_w: int,
    *,
    hole_col: int | None,
    require_hole: bool,
) -> np.ndarray:
    patch = np.full((cell_h, cell_w), int(color), dtype=int)
    if not require_hole and rng.random() < 0.35:
        return patch

    if hole_col is None:
        hole_col = int(rng.integers(1, cell_w - 1))
    hole_len = int(rng.integers(1, cell_h - 1))
    start = int(rng.integers(1, cell_h - hole_len))
    patch[start : start + hole_len, hole_col] = BG
    return patch


def _branch_types(rng: np.random.Generator, n_rows: int) -> list[str]:
    branches = ["single", "multi"]
    for _ in range(n_rows - 2):
        branches.append("single" if rng.random() < 0.5 else "multi")
    rng.shuffle(branches)
    return branches


def _zone_layout(
    rng: np.random.Generator, n_strips: int, cell_h: int, cell_w: int, orientation: str
) -> tuple[int, int, list[tuple[int, int]]]:
    candidates: list[tuple[int, int]] = []
    if orientation in {"left", "right"}:
        for cols in range(1, min(3, n_strips) + 1):
            candidates.append((math.ceil(n_strips / cols), cols))
    else:
        for rows in range(1, min(3, n_strips) + 1):
            candidates.append((rows, math.ceil(n_strips / rows)))
    candidates = list(dict.fromkeys(candidates))
    rows, cols = candidates[int(rng.integers(0, len(candidates)))]

    gap_h = int(rng.integers(1, 3))
    gap_w = int(rng.integers(1, 3))
    pitch_h = cell_h + gap_h
    pitch_w = cell_w + gap_w
    zone_h = 1 + rows * pitch_h
    zone_w = 1 + cols * pitch_w

    slots = [(r, c) for r in range(rows) for c in range(cols)]
    chosen = rng.choice(len(slots), size=n_strips, replace=False)
    positions: list[tuple[int, int]] = []
    for idx in chosen:
        slot_r, slot_c = slots[int(idx)]
        jitter_r = int(rng.integers(0, gap_h))
        jitter_c = int(rng.integers(0, gap_w))
        top = 1 + slot_r * pitch_h + jitter_r
        left = 1 + slot_c * pitch_w + jitter_c
        positions.append((top, left))
    return zone_h, zone_w, positions


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    rng = np.random.default_rng()

    n_rows = int(rng.integers(2, 5))
    n_cols = int(rng.integers(3, 5))
    cell_h = int(rng.integers(3, 5))
    cell_w = int(rng.integers(4, 6))
    max_multi = min(n_cols - 1, cell_w - 2)
    if max_multi < 2:
        return None

    row_colors = rng.choice(ROW_COLOR_POOL, size=n_rows, replace=False).astype(int)
    lattice_h = n_rows * cell_h + (n_rows + 1)
    lattice_w = n_cols * cell_w + (n_cols + 1)

    lattice_in = np.full((lattice_h, lattice_w), BG, dtype=int)
    lattice_out = lattice_in.copy()
    sep_rows = [i * (cell_h + 1) for i in range(n_rows + 1)]
    sep_cols = [i * (cell_w + 1) for i in range(n_cols + 1)]
    lattice_in[sep_rows, :] = LATTICE_COLOR
    lattice_in[:, sep_cols] = LATTICE_COLOR
    lattice_out[sep_rows, :] = LATTICE_COLOR
    lattice_out[:, sep_cols] = LATTICE_COLOR

    strips: list[dict[str, object]] = []
    branches = _branch_types(rng, n_rows)

    for ri, row_color in enumerate(row_colors):
        if branches[ri] == "single":
            missing_count = 1
        else:
            missing_count = int(rng.integers(2, max_multi + 1))

        empty_cols = sorted(rng.choice(n_cols, size=missing_count, replace=False).tolist())
        code_cols = sorted(rng.choice(np.arange(1, cell_w - 1), size=missing_count, replace=False).tolist())
        code_by_col = dict(zip(empty_cols, code_cols))

        rr0 = sep_rows[ri] + 1
        rr1 = sep_rows[ri + 1]
        for ci in range(n_cols):
            cc0 = sep_cols[ci] + 1
            cc1 = sep_cols[ci + 1]
            if ci in code_by_col:
                patch = _make_patch(
                    rng,
                    int(row_color),
                    cell_h,
                    cell_w,
                    hole_col=int(code_by_col[ci]),
                    require_hole=True,
                )
                strips.append({"color": int(row_color), "patch": patch})
                lattice_out[rr0:rr1, cc0:cc1] = patch
            else:
                patch = _make_patch(
                    rng,
                    int(row_color),
                    cell_h,
                    cell_w,
                    hole_col=None,
                    require_hole=False,
                )
                lattice_in[rr0:rr1, cc0:cc1] = patch
                lattice_out[rr0:rr1, cc0:cc1] = patch

    orientation = str(rng.choice(np.array(["left", "right", "top", "bottom"], dtype=object)))
    zone_h, zone_w, local_positions = _zone_layout(rng, len(strips), cell_h, cell_w, orientation)

    top_margin = int(rng.integers(1, 4))
    bottom_margin = int(rng.integers(1, 4))
    left_margin = int(rng.integers(1, 4))
    right_margin = int(rng.integers(1, 4))
    between_gap = int(rng.integers(2, 5))
    extra_h = int(rng.integers(0, 3))
    extra_w = int(rng.integers(0, 3))

    if orientation in {"left", "right"}:
        total_h = max(lattice_h, zone_h) + top_margin + bottom_margin + extra_h
        total_w = lattice_w + zone_w + left_margin + right_margin + between_gap + extra_w
    else:
        total_h = lattice_h + zone_h + top_margin + bottom_margin + between_gap + extra_h
        total_w = max(lattice_w, zone_w) + left_margin + right_margin + extra_w

    if total_h > MAX_SIDE or total_w > MAX_SIDE:
        return None

    inp = np.full((total_h, total_w), BG, dtype=int)
    out = inp.copy()

    if orientation == "left":
        lattice_left = left_margin
        zone_left = lattice_left + lattice_w + between_gap
        usable_h = total_h - top_margin - bottom_margin
        lattice_top = top_margin + int(rng.integers(0, usable_h - lattice_h + 1))
        zone_top = top_margin + int(rng.integers(0, usable_h - zone_h + 1))
    elif orientation == "right":
        zone_left = left_margin
        lattice_left = zone_left + zone_w + between_gap
        usable_h = total_h - top_margin - bottom_margin
        lattice_top = top_margin + int(rng.integers(0, usable_h - lattice_h + 1))
        zone_top = top_margin + int(rng.integers(0, usable_h - zone_h + 1))
    elif orientation == "top":
        lattice_top = top_margin
        zone_top = lattice_top + lattice_h + between_gap
        usable_w = total_w - left_margin - right_margin
        lattice_left = left_margin + int(rng.integers(0, usable_w - lattice_w + 1))
        zone_left = left_margin + int(rng.integers(0, usable_w - zone_w + 1))
    else:
        zone_top = top_margin
        lattice_top = zone_top + zone_h + between_gap
        usable_w = total_w - left_margin - right_margin
        lattice_left = left_margin + int(rng.integers(0, usable_w - lattice_w + 1))
        zone_left = left_margin + int(rng.integers(0, usable_w - zone_w + 1))

    _paste(inp, lattice_in, lattice_top, lattice_left)
    _paste(out, lattice_out, lattice_top, lattice_left)

    order = rng.permutation(len(strips))
    for idx, strip_idx in enumerate(order):
        patch = np.asarray(strips[int(strip_idx)]["patch"], dtype=int)
        local_top, local_left = local_positions[idx]
        _paste(inp, patch, zone_top + local_top, zone_left + local_left)

    counts = Counter(map(int, inp.ravel()))
    if any(color != BG and count >= counts[BG] for color, count in counts.items()):
        return None

    try:
        solved = main(inp)
    except Exception:
        return None
    if not np.array_equal(solved, out):
        return None

    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        result = _build_attempt()
        if result is not None:
            return result[0]
    raise RuntimeError("failed to generate a valid coded-strip lattice sample")
