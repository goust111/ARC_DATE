from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Iterable

import numpy as np


def _load_arc_utils() -> object:
    """Load arc_utils from the repository root without requiring package imports."""
    current = Path(__file__).resolve().parent
    for _ in range(16):
        candidate = current / "new" / "solvers" / "arc_utils.py"
        if candidate.is_file():
            spec = importlib.util.spec_from_file_location("_arc_utils_scan_diamond", candidate)
            if spec is None or spec.loader is None:
                raise ImportError("cannot load arc_utils specification")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            return module
        if current.parent == current:
            break
        current = current.parent
    raise ImportError("arc_utils.py not found relative to solver file")


try:  # pragma: no cover
    _arc_utils = _load_arc_utils()
    get_bg = _arc_utils.get_bg
except Exception:  # pragma: no cover
    from collections import Counter

    def get_bg(grid: np.ndarray) -> int:
        return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _run_segments(indices: np.ndarray) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    runs: list[tuple[int, int]] = []
    start = int(indices[0])
    prev = int(indices[0])
    for idx in map(int, indices[1:]):
        if idx == prev + 1:
            prev = idx
        else:
            runs.append((start, prev))
            start = idx
            prev = idx
    runs.append((start, prev))
    return runs


def _decode_single_pair(
    g: np.ndarray,
    bg: int,
    line_color: int,
    diamond_color: int,
) -> tuple[int, int, int] | None:
    """Try decode `(line_color, diamond_color)` into `(center_col, center_row, radius)`."""
    h, w = g.shape
    row_infos: list[tuple[int, int, int, int]] = []  # (r, width, start, end)
    rows_with_cells: list[tuple[int, int]] = []
    center_col: int | None = None

    for r in range(h):
        row = g[r]
        non_bg = row != bg
        if not bool(np.any(non_bg)):
            continue

        vals = {int(v) for v in row[non_bg]}
        if not vals.issubset({int(line_color), int(diamond_color)}):
            return None

        line_mask = row == line_color
        diamond_mask = row == diamond_color
        diamond_idx = np.flatnonzero(diamond_mask)

        if diamond_idx.size == 0:
            # Scan row without intersection at this row.
            if vals and vals != {int(line_color)}:
                return None
            row_infos.append((r, 0, -1, -1))
            continue

        runs = _run_segments(diamond_idx)
        if len(runs) != 1:
            return None

        start, end = runs[0]
        width = end - start + 1
        if width % 2 == 0:
            return None

        # All non-bg cells must be either line color or diamond color.
        if not np.array_equal((line_mask | diamond_mask), non_bg):
            return None

        # Diamond segment must be contiguous and its center must be integer.
        if ((start + end) % 2) != 0:
            return None
        c = (start + end) // 2
        if center_col is None:
            center_col = c
        elif center_col != c:
            return None

        if c < 0 or c >= w:
            return None

        row_infos.append((r, width, start, end))
        rows_with_cells.append((r, (width - 1) // 2))

    if not rows_with_cells:
        return None

    assert center_col is not None

    candidates: list[tuple[int, int]] = []
    for candidate_r in range(h):
        # radius determined from any intersection row
        r0, h0 = rows_with_cells[0]
        candidate_r_abs = abs(r0 - candidate_r)
        radius = h0 + candidate_r_abs

        # consistency check: every intersection row must match the diamond law.
        for rr, hh in rows_with_cells:
            if (radius - abs(rr - candidate_r)) < 0:
                break
            if (radius - abs(rr - candidate_r)) * 2 + 1 != 2 * hh + 1:
                break
            if abs(rr - candidate_r) + hh != radius:
                break
        else:
            # rows without diamond on scan lines must be outside the reconstructed span.
            ok = True
            for rr, width, _s, _e in row_infos:
                if width == 0 and abs(rr - candidate_r) <= radius:
                    ok = False
                    break
            if ok:
                candidates.append((candidate_r, radius))

    if len(candidates) != 1:
        return None

    center_r, radius = candidates[0]
    if radius <= 0:
        return None
    if center_col - radius < 0 or center_col + radius >= w:
        return None

    return center_col, center_r, radius


def main(grid: np.ndarray) -> np.ndarray:
    """Infer the hidden solid diamond from horizontal scanline segments and draw it as red.

    Input structure:
    - A black background with several full-width horizontal scan lines of a second color.
    - On rows where the hidden diamond intersects the scan line, one contiguous red segment appears, all on the same center column.
    - On rows without intersection, the scan line remains unbroken.

    Rule:
    - Identify which two non-background colors are the scanline color and the red trace color.
    - For each candidate pair, verify every non-background row is either a full scan line or a scan line with one contiguous red segment.
    - Use equal segment centers and the Manhattan width law (segment width grows by 2 when moving one row closer to center) to decode the exact center and radius.
    - Output the complete red filled diamond at the same grid size and erase scan lines.

    Key insight:
    - The cross-section widths are the only evidence of the hidden object, and linearity of
    |row - center_row| + (width-1)/2 uniquely determines center and radius.
    """
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = get_bg(g)

    colors = sorted(int(c) for c in np.unique(g) if int(c) != bg)
    if len(colors) < 2:
        raise ValueError("expected at least two non-background colors")

    candidates: list[tuple[int, int, int, int, int]] = []
    for i, c0 in enumerate(colors):
        for c1 in colors[i + 1 :]:
            for line_color, diamond_color in ((c0, c1), (c1, c0)):
                decoded = _decode_single_pair(g, bg, line_color, diamond_color)
                if decoded is None:
                    continue
                center_col, center_r, radius = decoded
                # Stronger check to avoid accidental aliasing when two colors pass weakly.
                row_with_cells = 0
                line_rows = 0
                for r in range(h):
                    row = g[r]
                    nz = row != bg
                    if not np.any(nz):
                        continue
                    row_with_cells += 1
                    if np.any(row[nz] == diamond_color):
                        d = abs(r - center_r)
                        expected_width = 2 * (radius - d) + 1
                        if expected_width <= 0:
                            row_with_cells = -10
                            break
                        seg_idx = np.flatnonzero(row == diamond_color)
                        if seg_idx.size != expected_width:
                            row_with_cells = -10
                            break
                        if not np.all(np.diff(seg_idx) == 1):
                            row_with_cells = -10
                            break
                    else:
                        line_rows += 1
                if row_with_cells < 0:
                    continue
                if row_with_cells == 0:
                    continue
                candidates.append((row_with_cells, line_rows, center_col, center_r, radius))

    if len(candidates) != 1:
        raise ValueError(f"expected one unique decode, got {len(candidates)}")

    _, _, center_col, center_r, radius = candidates[0]

    out = np.full((h, w), int(bg), dtype=int)
    for r in range(h):
        d = abs(r - center_r)
        if d > radius:
            continue
        half = radius - d
        c0 = center_col - half
        c1 = center_col + half
        out[r, c0 : c1 + 1] = 2

    return out


def _draw_diamond(h: int, w: int, center_r: int, center_c: int, radius: int, bg: int, color: int) -> np.ndarray:
    out = np.full((h, w), bg, dtype=int)
    for r in range(h):
        d = abs(r - center_r)
        if d > radius:
            continue
        half = radius - d
        c0 = center_c - half
        c1 = center_c + half
        out[r, c0 : c1 + 1] = color
    return out


def generate_input() -> np.ndarray:
    """Randomly sample a valid hidden-diamond scanline image under ARC bounds."""
    bg = 0
    diamond_color = 2
    line_color_choices = [5, 6, 7, 8, 9]

    for _ in range(400):
        h = int(np.random.randint(12, 29))
        w = int(np.random.randint(12, 29))

        max_radius = min(h, w) // 2 - 2
        if max_radius < 2:
            continue

        radius = int(np.random.randint(2, max_radius + 1))

        # Random geometry, with occasional margin from border but always valid.
        if np.random.random() < 0.5 and h > 2 * radius + 4 and w > 2 * radius + 4:
            center_r = int(np.random.randint(radius + 2, h - radius - 2))
            center_c = int(np.random.randint(radius + 2, w - radius - 2))
        else:
            center_r = int(np.random.randint(radius, h - radius))
            center_c = int(np.random.randint(radius, w - radius))

        line_color = int(np.random.choice(line_color_choices))
        if line_color == diamond_color:
            line_color = 5

        # Choose intersecting rows with branchy patterns.
        all_intersect_rows = list(range(center_r - radius, center_r + radius + 1))
        if not all_intersect_rows:
            continue

        mode = int(np.random.choice([0, 1, 2]))
        if mode == 0:
            # All sampled rows.
            sampled = all_intersect_rows
        elif mode == 1:
            # Sparse random subset, keep center.
            min_keep = min(3, len(all_intersect_rows))
            keep = int(np.random.randint(min_keep, len(all_intersect_rows) + 1))
            sampled = sorted(np.random.choice(all_intersect_rows, size=keep, replace=False).tolist())
        else:
            # Arithmetic stride branch to create irregular sampling.
            stride = int(np.random.choice([1, 2, 3]))
            sampled = all_intersect_rows[::stride]

        sampled = set(int(r) for r in sampled)
        sampled.add(center_r)

        # Optional outside lines that do not intersect the diamond.
        if np.random.random() < 0.75:
            spare = [r for r in range(h) if r not in sampled and 0 <= r < h]
            extra = int(np.random.randint(0, min(3, len(spare)) + 1)) if spare else 0
            sampled.update(int(x) for x in np.random.choice(spare, size=extra, replace=False).tolist())

        # Ensure at least 3 scan rows for stable reconstruction.
        if len(sampled) < 3:
            continue

        grid = np.full((h, w), bg, dtype=int)
        for r in sampled:
            grid[r, :] = line_color
            if center_r - radius <= r <= center_r + radius:
                d = abs(r - center_r)
                half = radius - d
                c0 = center_c - half
                c1 = center_c + half
                grid[r, c0 : c1 + 1] = diamond_color

        expected = _draw_diamond(h, w, center_r, center_c, radius, bg, diamond_color)
        try:
            decoded = main(grid)
        except Exception:
            continue
        if not np.array_equal(decoded, expected):
            continue

        return grid

    raise RuntimeError("failed to sample a valid hidden diamond input after retries")
