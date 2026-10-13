from __future__ import annotations

import numpy as np


BG = 0
SEP = 5
PALETTE = [1, 2, 3, 4, 6, 7, 8, 9]
GEN_ATTEMPTS = 400


def _solid_lines(grid: np.ndarray, axis: int) -> list[tuple[int, int]]:
    lines: list[tuple[int, int]] = []
    length = grid.shape[axis]
    for idx in range(length):
        line = grid[idx, :] if axis == 0 else grid[:, idx]
        color = int(line[0])
        if color != BG and np.all(line == color):
            lines.append((idx, color))
    return lines


def _content_bands(separators: list[int], size: int) -> list[tuple[int, int]]:
    cuts = sorted(set(int(x) for x in separators))
    if not cuts or cuts[0] != 0:
        cuts = [-1] + cuts
    if cuts[-1] != size - 1:
        cuts = cuts + [size]
    else:
        cuts = cuts + [size]

    bands: list[tuple[int, int]] = []
    for a, b in zip(cuts, cuts[1:]):
        if b - a > 1:
            bands.append((a + 1, b))
    return bands


def _panel_layout(grid: np.ndarray) -> tuple[list[tuple[int, int]], list[tuple[int, int]], int]:
    row_lines = _solid_lines(grid, axis=0)
    col_lines = _solid_lines(grid, axis=1)
    row_colors = {color for _, color in row_lines}
    col_colors = {color for _, color in col_lines}
    common = row_colors & col_colors
    assert common, "expected a shared separator color"
    sep = max(common, key=lambda color: sum(c == color for _, c in row_lines) + sum(c == color for _, c in col_lines))

    row_bands = _content_bands([idx for idx, color in row_lines if color == sep], grid.shape[0])
    col_bands = _content_bands([idx for idx, color in col_lines if color == sep], grid.shape[1])
    heights = {r1 - r0 for r0, r1 in row_bands}
    widths = {c1 - c0 for c0, c1 in col_bands}
    assert len(heights) == 1 and len(widths) == 1, "panels must have equal size"
    assert heights == widths, "panels must be square"
    return row_bands, col_bands, int(sep)


def _axis_stats(panel: np.ndarray, axis: str) -> tuple[int, int, int]:
    size = panel.shape[0]
    mismatches = 0
    non_bg_matches = 0
    matches = 0

    if axis == "v":
        for r in range(size):
            for c in range(size // 2):
                c2 = size - 1 - c
                a = int(panel[r, c])
                b = int(panel[r, c2])
                if a == b:
                    matches += 1
                    if a != BG:
                        non_bg_matches += 1
                else:
                    mismatches += 1
    else:
        for r in range(size // 2):
            r2 = size - 1 - r
            for c in range(size):
                a = int(panel[r, c])
                b = int(panel[r2, c])
                if a == b:
                    matches += 1
                    if a != BG:
                        non_bg_matches += 1
                else:
                    mismatches += 1

    return mismatches, non_bg_matches, matches


def _choose_axis(panel: np.ndarray) -> str:
    vertical = _axis_stats(panel, "v")
    horizontal = _axis_stats(panel, "h")
    if (vertical[0], -vertical[1], -vertical[2]) <= (horizontal[0], -horizontal[1], -horizontal[2]):
        return "v"
    return "h"


def _repair_panel(panel: np.ndarray) -> np.ndarray:
    axis = _choose_axis(panel)
    out = panel.copy()
    size = panel.shape[0]

    if axis == "v":
        for r in range(size):
            for c in range(size // 2):
                c2 = size - 1 - c
                if out[r, c] != out[r, c2]:
                    left = int(out[r, c])
                    out[r, c] = int(out[r, c2])
                    out[r, c2] = left
    else:
        for r in range(size // 2):
            r2 = size - 1 - r
            for c in range(size):
                if out[r, c] != out[r2, c]:
                    top = int(out[r, c])
                    out[r, c] = int(out[r2, c])
                    out[r2, c] = top
    return out


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    row_bands, col_bands, sep = _panel_layout(g)
    out = g.copy()
    out[g == sep] = sep

    for r0, r1 in row_bands:
        for c0, c1 in col_bands:
            out[r0:r1, c0:c1] = _repair_panel(g[r0:r1, c0:c1])

    return out


def _wrong_axis_mismatches(panel: np.ndarray, axis: str) -> int:
    other = "h" if axis == "v" else "v"
    return _axis_stats(panel, other)[0]


def _random_base_panel(size: int, axis: str) -> np.ndarray | None:
    panel = np.zeros((size, size), dtype=int)
    colors = list(np.random.choice(PALETTE, size=int(np.random.randint(2, 5)), replace=False))

    if axis == "v":
        orbits = [(r, c) for r in range(size) for c in range((size + 1) // 2)]
    else:
        orbits = [(r, c) for r in range((size + 1) // 2) for c in range(size)]

    np.random.shuffle(orbits)
    target = int(np.random.randint(max(4, size), max(5, (size * size) // 2 + 1)))
    colored_orbits = 0
    for r, c in orbits[:target]:
        if np.random.rand() < 0.35:
            continue
        color = int(np.random.choice(colors))
        if axis == "v":
            c2 = size - 1 - c
            panel[r, c] = color
            panel[r, c2] = color
        else:
            r2 = size - 1 - r
            panel[r, c] = color
            panel[r2, c] = color
        colored_orbits += 1

    if colored_orbits < 4:
        return None
    if np.count_nonzero(panel) < size + 2:
        return None
    if _wrong_axis_mismatches(panel, axis) < 3:
        return None
    return panel


def _markable_pairs(panel: np.ndarray, axis: str) -> list[tuple[tuple[int, int], tuple[int, int]]]:
    size = panel.shape[0]
    pairs: list[tuple[tuple[int, int], tuple[int, int]]] = []
    if axis == "v":
        for r in range(size):
            for c in range(size // 2):
                c2 = size - 1 - c
                if int(panel[r, c]) != BG:
                    pairs.append(((r, c), (r, c2)))
    else:
        for r in range(size // 2):
            r2 = size - 1 - r
            for c in range(size):
                if int(panel[r, c]) != BG:
                    pairs.append(((r, c), (r2, c)))
    return pairs


def _inject_markers(base: np.ndarray, axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    pairs = _markable_pairs(base, axis)
    if len(pairs) < 2:
        return None

    order = np.random.permutation(len(pairs))
    mark_count = 1 if np.random.rand() < 0.5 else 2
    if len(pairs) < mark_count:
        return None

    inp = base.copy()
    out = base.copy()
    used_markers: set[int] = set()

    for idx in order[:mark_count]:
        a, b = pairs[int(idx)]
        original = int(base[a])
        marker_choices = [color for color in PALETTE if color != original and color not in used_markers]
        if not marker_choices:
            return None
        marker = int(np.random.choice(marker_choices))
        used_markers.add(marker)

        if np.random.rand() < 0.5:
            marked, mirrored = a, b
        else:
            marked, mirrored = b, a

        inp[marked] = marker
        out[mirrored] = marker

    if _choose_axis(inp) != axis:
        return None
    if _axis_stats(inp, axis)[0] != mark_count:
        return None
    if _wrong_axis_mismatches(inp, axis) <= mark_count:
        return None
    return inp, out


def _build_panel(size: int, axis: str) -> tuple[np.ndarray, np.ndarray] | None:
    for _ in range(80):
        base = _random_base_panel(size, axis)
        if base is None:
            continue
        result = _inject_markers(base, axis)
        if result is not None:
            return result
    return None


def _assemble_grid(panels_in: list[list[np.ndarray]], sep: int) -> np.ndarray:
    rows = len(panels_in)
    cols = len(panels_in[0])
    size = panels_in[0][0].shape[0]
    height = rows * size + rows + 1
    width = cols * size + cols + 1
    grid = np.full((height, width), sep, dtype=int)

    for pr in range(rows):
        for pc in range(cols):
            r0 = 1 + pr * (size + 1)
            c0 = 1 + pc * (size + 1)
            grid[r0 : r0 + size, c0 : c0 + size] = panels_in[pr][pc]
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        size = int(np.random.randint(5, 9))
        panel_rows = int(np.random.randint(2, 5))
        panel_cols = int(np.random.randint(2, 5))
        total = panel_rows * panel_cols

        axes = np.array(["v"] * total, dtype=object)
        axes[: total // 2] = "h"
        np.random.shuffle(axes)

        panels_in: list[list[np.ndarray]] = []
        panels_out: list[list[np.ndarray]] = []
        ok = True
        cursor = 0

        for _r in range(panel_rows):
            row_in: list[np.ndarray] = []
            row_out: list[np.ndarray] = []
            for _c in range(panel_cols):
                built = _build_panel(size, str(axes[cursor]))
                cursor += 1
                if built is None:
                    ok = False
                    break
                pin, pout = built
                row_in.append(pin)
                row_out.append(pout)
            if not ok:
                break
            panels_in.append(row_in)
            panels_out.append(row_out)

        if not ok:
            continue

        grid_in = _assemble_grid(panels_in, SEP)
        expected = _assemble_grid(panels_out, SEP)
        if np.array_equal(main(grid_in), expected):
            return grid_in

    raise RuntimeError(f"generate_input: failed after {GEN_ATTEMPTS} attempts")
