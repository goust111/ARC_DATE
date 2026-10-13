from __future__ import annotations

from collections import Counter
import random

import numpy as np


GREY = 5
MAROON = 9
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _most_common_color(grid: np.ndarray) -> int:
    counts = np.bincount(np.asarray(grid, dtype=int).ravel(), minlength=10)
    return int(np.argmax(counts))


def _neighbors4(r: int, c: int, h: int, w: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for dr, dc in DIRS4:
        rr = int(r + dr)
        cc = int(c + dc)
        if 0 <= rr < h and 0 <= cc < w:
            out.append((rr, cc))
    return out


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((int(cr), int(cc)))
                for nr, nc in _neighbors4(int(cr), int(cc), h, w):
                    if bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((int(nr), int(nc)))
            comps.append(comp)
    return comps


def _bbox(cells: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in cells]
    cols = [c for _, c in cells]
    return min(rows), min(cols), max(rows), max(cols)


def _stamp_cells(
    shape_cells: list[tuple[int, int]],
    anchor: tuple[int, int],
    height: int,
    width: int,
) -> tuple[set[tuple[int, int]], bool]:
    cells: set[tuple[int, int]] = set()
    clipped = False
    ar, ac = anchor
    for sr, sc in shape_cells:
        rr = int(ar + sr)
        cc = int(ac + sc)
        if 0 <= rr < height and 0 <= cc < width:
            cells.add((rr, cc))
        else:
            clipped = True
    return cells, clipped


def _separate_from(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    height: int,
    width: int,
) -> bool:
    if cells & occupied:
        return False
    for r, c in cells:
        for nr, nc in _neighbors4(int(r), int(c), height, width):
            if (nr, nc) in occupied and (nr, nc) not in cells:
                return False
    return True


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    bg = _most_common_color(g)

    line_mask = (g != bg) & (g != GREY) & (g != MAROON)
    region_comps = _components(~line_mask)
    out = g.copy()
    out[g == MAROON] = bg

    for comp in region_comps:
        grey_cells = [(r, c) for r, c in comp if int(g[r, c]) == GREY]
        if not grey_cells:
            continue

        r0, c0, r1, c1 = _bbox(comp)
        assert len(comp) == (r1 - r0 + 1) * (c1 - c0 + 1), "each region interior must be rectangular"

        boundary_counts: Counter[int] = Counter()
        boundary_seen: set[tuple[int, int]] = set()
        for r, c in comp:
            for nr, nc in _neighbors4(int(r), int(c), h, w):
                if bool(line_mask[nr, nc]) and (nr, nc) not in boundary_seen:
                    boundary_seen.add((nr, nc))
                    boundary_counts[int(g[nr, nc])] += 1
        assert boundary_counts, "each region must touch divider lines"

        max_count = max(boundary_counts.values())
        dominant = [color for color, count in boundary_counts.items() if count == max_count]
        assert len(dominant) == 1, "region border colors must have a unique dominant count"
        stamp_color = int(dominant[0])

        region = g[r0 : r1 + 1, c0 : c1 + 1]
        template_comps = _components(region == GREY)
        assert len(template_comps) == 1, "each region must contain exactly one grey template"

        template = template_comps[0]
        tr0, tc0, tr1, tc1 = _bbox(template)
        rel_cells = [(int(r - tr0), int(c - tc0)) for r, c in template]

        for rr, cc in template:
            out[r0 + rr, c0 + cc] = stamp_color

        markers = [(int(r), int(c)) for r, c in np.argwhere(region == MAROON)]
        for mr, mc in markers:
            for sr, sc in rel_cells:
                rr = int(mr + sr)
                cc = int(mc + sc)
                if 0 <= rr < region.shape[0] and 0 <= cc < region.shape[1]:
                    out[r0 + rr, c0 + cc] = stamp_color

    return out


REGION_ROWS_RANGE = (2, 4)
REGION_COLS_RANGE = (2, 4)
REGION_HEIGHT_RANGE = (6, 10)
REGION_WIDTH_RANGE = (6, 10)
LINE_COLOR_COUNT_RANGE = (3, 5)
LINE_COLOR_ATTEMPTS = 120
REGION_CONTENT_ATTEMPTS = 160
GENERATE_ATTEMPTS = 400


def _random_template() -> np.ndarray:
    for _ in range(80):
        canvas_h = random.randint(1, 4)
        canvas_w = random.randint(1, 4)
        max_area = min(canvas_h * canvas_w, 7)
        area = random.randint(1, max_area)
        start = (random.randrange(canvas_h), random.randrange(canvas_w))
        cells = {start}
        while len(cells) < area:
            frontier: list[tuple[int, int]] = []
            for r, c in cells:
                for nr, nc in _neighbors4(int(r), int(c), canvas_h, canvas_w):
                    if (nr, nc) not in cells:
                        frontier.append((nr, nc))
            if not frontier:
                break
            cells.add(random.choice(frontier))
        if len(cells) != area:
            continue

        rows = [r for r, _ in cells]
        cols = [c for _, c in cells]
        r0, c0 = min(rows), min(cols)
        cropped = {(r - r0, c - c0) for r, c in cells}
        h = max(r for r, _ in cropped) + 1
        w = max(c for _, c in cropped) + 1
        mask = np.zeros((h, w), dtype=bool)
        for r, c in cropped:
            mask[r, c] = True
        return mask
    raise RuntimeError("failed to create a template")


def _choose_line_colors(
    row_heights: list[int],
    col_widths: list[int],
    bg: int,
) -> tuple[list[int], list[int], list[list[int]]] | None:
    available = [c for c in range(10) if c not in (bg, GREY, MAROON)]
    min_count = min(LINE_COLOR_COUNT_RANGE[0], len(available))
    max_count = min(LINE_COLOR_COUNT_RANGE[1], len(available))
    for _ in range(LINE_COLOR_ATTEMPTS):
        palette = random.sample(available, random.randint(min_count, max_count))
        hcols = [random.choice(palette) for _ in range(len(row_heights) + 1)]
        vcols = [random.choice(palette) for _ in range(len(col_widths) + 1)]
        if len(set(hcols + vcols)) < min(3, len(available)):
            continue

        dominants: list[list[int]] = []
        ok = True
        for i, rh in enumerate(row_heights):
            row: list[int] = []
            for j, cw in enumerate(col_widths):
                counts: Counter[int] = Counter()
                counts[int(hcols[i])] += int(cw)
                counts[int(hcols[i + 1])] += int(cw)
                counts[int(vcols[j])] += int(rh)
                counts[int(vcols[j + 1])] += int(rh)
                top = max(counts.values())
                winners = [color for color, count in counts.items() if count == top]
                if len(winners) != 1:
                    ok = False
                    break
                row.append(int(winners[0]))
            if not ok:
                break
            dominants.append(row)
        if ok:
            return hcols, vcols, dominants
    return None


def _valid_marker_candidates(
    height: int,
    width: int,
    shape_cells: list[tuple[int, int]],
    input_occupied: set[tuple[int, int]],
    output_occupied: set[tuple[int, int]],
) -> list[tuple[tuple[int, int], set[tuple[int, int]], bool]]:
    candidates: list[tuple[tuple[int, int], set[tuple[int, int]], bool]] = []
    for ar in range(1, height - 1):
        for ac in range(1, width - 1):
            anchor = (int(ar), int(ac))
            if anchor in input_occupied or anchor in output_occupied:
                continue
            if any(nb in input_occupied for nb in _neighbors4(int(ar), int(ac), height, width)):
                continue
            stamp, clipped = _stamp_cells(shape_cells, anchor, height, width)
            if not stamp:
                continue
            if not _separate_from(stamp, output_occupied, height, width):
                continue
            candidates.append((anchor, stamp, clipped))
    return candidates


def _build_region_contents(
    height: int,
    width: int,
    bg: int,
    stamp_color: int,
    require_clipped: bool,
    prefer_clipped: bool,
) -> tuple[np.ndarray, np.ndarray, bool] | None:
    for _ in range(REGION_CONTENT_ATTEMPTS):
        template_mask = _random_template()
        th, tw = map(int, template_mask.shape)
        if height < th + 2 or width < tw + 2:
            continue

        top = random.randint(1, height - th - 1)
        left = random.randint(1, width - tw - 1)
        template_rel = [(int(r), int(c)) for r, c in np.argwhere(template_mask)]
        template_abs = {(int(top + r), int(left + c)) for r, c in template_rel}
        template_box_rel = [(int(r), int(c)) for r in range(th) for c in range(tw) if template_mask[r, c]]

        input_patch = np.full((height, width), bg, dtype=int)
        output_patch = np.full((height, width), bg, dtype=int)
        for r, c in template_abs:
            input_patch[r, c] = GREY
            output_patch[r, c] = int(stamp_color)

        markers: list[tuple[int, int]] = []
        input_occupied = set(template_abs)
        output_occupied = set(template_abs)
        used_clipped = False

        initial = _valid_marker_candidates(height, width, template_box_rel, input_occupied, output_occupied)
        if not initial:
            continue
        if require_clipped and not any(clipped for _anchor, _stamp, clipped in initial):
            continue

        target_count = random.randint(1, min(4, len(initial)))
        if require_clipped:
            target_count = max(1, target_count)

        while len(markers) < target_count:
            candidates = _valid_marker_candidates(height, width, template_box_rel, input_occupied, output_occupied)
            if not candidates:
                break

            clipped_candidates = [cand for cand in candidates if cand[2]]
            if require_clipped and not used_clipped:
                chosen = random.choice(clipped_candidates)
            elif prefer_clipped and clipped_candidates and random.random() < 0.55:
                chosen = random.choice(clipped_candidates)
            else:
                chosen = random.choice(candidates)

            anchor, stamp_cells, clipped = chosen
            markers.append(anchor)
            input_occupied.add(anchor)
            output_occupied |= stamp_cells
            used_clipped = used_clipped or bool(clipped)

            input_patch[anchor] = MAROON
            for r, c in stamp_cells:
                output_patch[r, c] = int(stamp_color)

        if not markers:
            continue
        if require_clipped and not used_clipped:
            continue
        return input_patch, output_patch, used_clipped
    return None


def _line_positions(sizes: list[int]) -> list[int]:
    positions = [0]
    cursor = 0
    for size in sizes:
        cursor += int(size) + 1
        positions.append(int(cursor))
    return positions


def _build_attempt() -> tuple[np.ndarray, np.ndarray] | None:
    region_rows = random.randint(*REGION_ROWS_RANGE)
    region_cols = random.randint(*REGION_COLS_RANGE)
    row_heights = [random.randint(*REGION_HEIGHT_RANGE) for _ in range(region_rows)]
    col_widths = [random.randint(*REGION_WIDTH_RANGE) for _ in range(region_cols)]

    bg_choices = [c for c in range(10) if c not in (GREY, MAROON)]
    bg = random.choice(bg_choices)

    line_choice = _choose_line_colors(row_heights, col_widths, bg)
    if line_choice is None:
        return None
    hcols, vcols, dominants = line_choice

    row_lines = _line_positions(row_heights)
    col_lines = _line_positions(col_widths)
    height = int(row_lines[-1] + 1)
    width = int(col_lines[-1] + 1)

    inp = np.full((height, width), bg, dtype=int)
    out = np.full((height, width), bg, dtype=int)

    for idx, r in enumerate(row_lines):
        inp[r, :] = int(hcols[idx])
        out[r, :] = int(hcols[idx])
    for idx, c in enumerate(col_lines):
        inp[:, c] = int(vcols[idx])
        out[:, c] = int(vcols[idx])

    force_clipping = random.random() < 0.5
    clip_region = None
    if force_clipping:
        clip_region = (random.randrange(region_rows), random.randrange(region_cols))

    used_any_clipping = False
    for i in range(region_rows):
        for j in range(region_cols):
            r0 = int(row_lines[i] + 1)
            r1 = int(row_lines[i + 1] - 1)
            c0 = int(col_lines[j] + 1)
            c1 = int(col_lines[j + 1] - 1)
            rh = int(r1 - r0 + 1)
            cw = int(c1 - c0 + 1)

            require_clipped = clip_region == (i, j)
            prefer_clipped = random.random() < 0.35
            region = _build_region_contents(
                rh,
                cw,
                bg,
                int(dominants[i][j]),
                require_clipped=require_clipped,
                prefer_clipped=prefer_clipped,
            )
            if region is None:
                return None

            input_patch, output_patch, clipped = region
            used_any_clipping = used_any_clipping or bool(clipped)
            inp[r0 : r1 + 1, c0 : c1 + 1] = input_patch
            out[r0 : r1 + 1, c0 : c1 + 1] = output_patch

    if force_clipping and not used_any_clipping:
        return None

    counts = Counter(map(int, inp.ravel()))
    bg_count = counts[int(bg)]
    if any(color != int(bg) and count >= bg_count for color, count in counts.items()):
        return None
    if not np.array_equal(main(inp), out):
        return None
    return inp, out


def generate_input() -> np.ndarray:
    for _ in range(GENERATE_ATTEMPTS):
        built = _build_attempt()
        if built is not None:
            return built[0]
    raise RuntimeError(f"generate_input: failed after {GENERATE_ATTEMPTS} attempts")
