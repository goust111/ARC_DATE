import random
from collections import deque

import numpy as np


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _aspect_ratio(height: int, width: int) -> float:
    long_side = max(int(height), int(width))
    short_side = min(int(height), int(width))
    assert short_side > 0
    return long_side / short_side


def _separator_color(grid: np.ndarray, bg: int) -> int:
    candidates = []
    for color0 in np.unique(grid):
        color = int(color0)
        if color == bg:
            continue
        if np.any(np.all(grid == color, axis=1)) or np.any(np.all(grid == color, axis=0)):
            candidates.append(color)
    assert len(candidates) == 1, "expected a unique separator color"
    return candidates[0]


def _spans(length: int, separators: list[int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(separators) + [length]
    spans: list[tuple[int, int]] = []
    for lo, hi in zip(bounds, bounds[1:]):
        if hi - lo > 1:
            spans.append((lo + 1, hi - 1))
    assert spans
    return spans


def _cell_spans(grid: np.ndarray, sep: int) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    sep_rows = [r for r in range(grid.shape[0]) if np.all(grid[r, :] == sep)]
    sep_cols = [c for c in range(grid.shape[1]) if np.all(grid[:, c] == sep)]
    assert sep_rows or sep_cols, "expected at least one full separator line"
    return _spans(grid.shape[0], sep_rows), _spans(grid.shape[1], sep_cols)


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    out: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            queue = deque([(r, c)])
            comp = []
            while queue:
                cr, cc = queue.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
            out.append(comp)
    return out


def _extract_samples(cell: np.ndarray, bg: int, sep: int) -> list[dict]:
    sample_mask = (cell != bg) & (cell != sep)
    comps = _components(sample_mask)
    assert comps, "library must contain at least one sample"

    samples = []
    for comp in comps:
        colors = {int(cell[r, c]) for r, c in comp}
        assert len(colors) == 1, "each library sample must be monochrome"
        rows = [r for r, _ in comp]
        cols = [c for _, c in comp]
        r0, r1 = min(rows), max(rows)
        c0, c1 = min(cols), max(cols)
        rel = sorted((r - r0, c - c0) for r, c in comp)
        height = r1 - r0 + 1
        width = c1 - c0 + 1
        samples.append(
            {
                "coords": rel,
                "height": height,
                "width": width,
                "ratio": _aspect_ratio(height, width),
            }
        )

    samples.sort(key=lambda sample: (sample["ratio"], sample["height"], sample["width"], len(sample["coords"])))
    ratios = [sample["ratio"] for sample in samples]
    for i in range(len(ratios) - 1):
        assert abs(ratios[i] - ratios[i + 1]) > 1e-9, "sample ratios must be distinct"
    return samples


def _parse_grid(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0
    bg = _mode_color(g)
    sep = _separator_color(g, bg)
    row_spans, col_spans = _cell_spans(g, sep)

    library_candidates = []
    seeds = []
    blanks = 0

    for r0, r1 in row_spans:
        for c0, c1 in col_spans:
            patch = g[r0 : r1 + 1, c0 : c1 + 1]
            colored = np.argwhere((patch != bg) & (patch != sep))
            if len(colored) == 0:
                blanks += 1
                continue
            if len(colored) == 1:
                rr, cc = map(int, colored[0])
                seeds.append(
                    {
                        "bounds": (r0, r1, c0, c1),
                        "anchor": (r0 + rr, c0 + cc),
                        "color": int(patch[rr, cc]),
                    }
                )
            else:
                library_candidates.append((r0, r1, c0, c1))

    assert len(library_candidates) == 1, "expected exactly one library region"
    lr0, lr1, lc0, lc1 = library_candidates[0]
    samples = _extract_samples(g[lr0 : lr1 + 1, lc0 : lc1 + 1], bg, sep)

    return {
        "grid": g,
        "bg": bg,
        "sep": sep,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "library_bounds": (lr0, lr1, lc0, lc1),
        "samples": samples,
        "seeds": seeds,
        "blank_count": blanks,
    }


def _best_match_index(value: float, targets: list[float]) -> int:
    assert targets
    diffs = [abs(value - target) for target in targets]
    order = sorted(range(len(targets)), key=lambda idx: (diffs[idx], targets[idx]))
    if len(order) > 1:
        assert diffs[order[0]] + 1e-9 < diffs[order[1]], "expected a unique closest match"
    return order[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Match each seeded region to the library sample with the closest bbox aspect ratio,
    then stamp that sample into the region using the seed pixel as the sample's top-left anchor.
    """
    parsed = _parse_grid(np.asarray(grid, dtype=int))
    g = parsed["grid"]
    bg = parsed["bg"]
    sep = parsed["sep"]
    sample_ratios = [sample["ratio"] for sample in parsed["samples"]]

    out = g.copy()
    for seed in parsed["seeds"]:
        r0, r1, c0, c1 = seed["bounds"]
        anchor_r, anchor_c = seed["anchor"]
        cell_ratio = _aspect_ratio(r1 - r0 + 1, c1 - c0 + 1)
        sample = parsed["samples"][_best_match_index(cell_ratio, sample_ratios)]

        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[(patch != bg) & (patch != sep)] = bg
        for dr, dc in sample["coords"]:
            rr = anchor_r + dr
            cc = anchor_c + dc
            assert r0 <= rr <= r1 and c0 <= cc <= c1, "stamped sample must stay inside its region"
            assert out[rr, cc] != sep
            out[rr, cc] = seed["color"]

    return out


SAMPLE_BOXES = [(2, 2), (2, 3), (3, 5), (2, 4), (2, 5), (2, 6)]
SIDE_CHOICES = ("left", "right", "top", "bottom")
GEN_RETRIES = 500


def _oriented_box(box: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    h, w = box
    return (w, h) if rng.random() < 0.5 else (h, w)


def _make_sample_mask(height: int, width: int, rng: random.Random) -> np.ndarray:
    cells = {(0, 0)}
    path_steps = [0] * (height - 1) + [1] * (width - 1)
    rng.shuffle(path_steps)
    r = c = 0
    for step in path_steps:
        if step == 0:
            r += 1
        else:
            c += 1
        cells.add((r, c))

    max_area = height * width
    extra_budget = max(0, min(max_area - len(cells), (max_area + 1) // 3))
    target_area = len(cells) + rng.randint(0, extra_budget)
    while len(cells) < target_area:
        candidates = []
        for cr, cc in cells:
            for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                if 0 <= nr < height and 0 <= nc < width and (nr, nc) not in cells:
                    candidates.append((nr, nc))
        if not candidates:
            break
        cells.add(rng.choice(candidates))

    mask = np.zeros((height, width), dtype=bool)
    for rr, cc in cells:
        mask[rr, cc] = True
    assert mask[0, 0] and mask[height - 1, width - 1]
    return mask


def _layout_requirement(samples: list[dict], side: str) -> tuple[int, int]:
    if side in ("left", "right"):
        height = 2 + 2 + sum(sample["height"] for sample in samples) + (len(samples) - 1)
        width = 2 + 2 + max(sample["width"] for sample in samples)
    else:
        height = 2 + 2 + max(sample["height"] for sample in samples)
        width = 2 + 2 + sum(sample["width"] for sample in samples) + (len(samples) - 1)
    return height, width


def _spans_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans = []
    separators = []
    pos = 0
    for idx, size in enumerate(sizes):
        spans.append((pos, pos + size - 1))
        pos += size
        if idx != len(sizes) - 1:
            separators.append(pos)
            pos += 1
    return spans, separators, pos


def _library_index(side: str, row_count: int, col_count: int, rng: random.Random) -> tuple[int, int]:
    if side == "left":
        return rng.randrange(row_count), 0
    if side == "right":
        return rng.randrange(row_count), col_count - 1
    if side == "top":
        return 0, rng.randrange(col_count)
    return row_count - 1, rng.randrange(col_count)


def _paint_library(
    grid: np.ndarray,
    bounds: tuple[int, int, int, int],
    samples: list[dict],
    sample_colors: list[int],
    side: str,
    rng: random.Random,
    sep: int,
) -> None:
    r0, r1, c0, c1 = bounds
    grid[r0:r1 + 1, c0] = sep
    grid[r0:r1 + 1, c1] = sep
    grid[r0, c0:c1 + 1] = sep
    grid[r1, c0:c1 + 1] = sep

    inner_r0, inner_r1 = r0 + 1, r1 - 1
    inner_c0, inner_c1 = c0 + 1, c1 - 1
    assert inner_r0 <= inner_r1 and inner_c0 <= inner_c1

    order = list(range(len(samples)))
    rng.shuffle(order)
    if side in ("left", "right"):
        cursor = inner_r0 + 1
        for idx in order:
            sample = samples[idx]
            color = sample_colors[idx]
            max_col = inner_c1 - sample["width"] + 1
            col = rng.randint(inner_c0 + 1, max_col)
            for dr, dc in sample["coords"]:
                grid[cursor + dr, col + dc] = color
            cursor += sample["height"] + 1
    else:
        cursor = inner_c0 + 1
        for idx in order:
            sample = samples[idx]
            color = sample_colors[idx]
            max_row = inner_r1 - sample["height"] + 1
            row = rng.randint(inner_r0 + 1, max_row)
            for dr, dc in sample["coords"]:
                grid[row + dr, cursor + dc] = color
            cursor += sample["width"] + 1


def _fit_positions(bounds: tuple[int, int, int, int], sample: dict) -> list[tuple[int, int]]:
    r0, r1, c0, c1 = bounds
    max_r = r1 - sample["height"] + 1
    max_c = c1 - sample["width"] + 1
    if max_r < r0 or max_c < c0:
        return []
    return [(r, c) for r in range(r0, max_r + 1) for c in range(c0, max_c + 1)]


def _candidate_seed_cells(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    library_idx: tuple[int, int],
    samples: list[dict],
) -> list[dict]:
    sample_ratios = [sample["ratio"] for sample in samples]
    jobs = []
    for ri, (r0, r1) in enumerate(row_spans):
        for ci, (c0, c1) in enumerate(col_spans):
            if (ri, ci) == library_idx:
                continue
            ratio = _aspect_ratio(r1 - r0 + 1, c1 - c0 + 1)
            try:
                match = _best_match_index(ratio, sample_ratios)
            except AssertionError:
                continue
            positions = _fit_positions((r0, r1, c0, c1), samples[match])
            if not positions:
                continue
            jobs.append(
                {
                    "cell_index": (ri, ci),
                    "bounds": (r0, r1, c0, c1),
                    "match": match,
                    "positions": positions,
                }
            )
    return jobs


def _sample_specs(rng: random.Random) -> list[dict]:
    count = rng.randint(3, 4)
    while True:
        boxes = rng.sample(SAMPLE_BOXES, count)
        ratios = sorted(_aspect_ratio(h, w) for h, w in boxes)
        if all(b - a >= 0.2 for a, b in zip(ratios, ratios[1:])):
            break

    specs = []
    for box in boxes:
        height, width = _oriented_box(box, rng)
        mask = _make_sample_mask(height, width, rng)
        coords = sorted(map(tuple, np.argwhere(mask)))
        specs.append(
            {
                "coords": [(int(r), int(c)) for r, c in coords],
                "height": height,
                "width": width,
                "ratio": _aspect_ratio(height, width),
            }
        )
    return specs


def _generate_candidate(rng: random.Random) -> np.ndarray | None:
    bg, sep = rng.sample(list(range(10)), 2)
    free_colors = [color for color in range(10) if color not in {bg, sep}]

    samples = _sample_specs(rng)
    side = rng.choice(SIDE_CHOICES)
    req_h, req_w = _layout_requirement(samples, side)

    row_count = rng.randint(2, 4)
    col_count = rng.randint(2, 4)
    library_idx = _library_index(side, row_count, col_count, rng)
    lib_row, lib_col = library_idx

    row_sizes = [rng.randint(4, 9) for _ in range(row_count)]
    col_sizes = [rng.randint(4, 9) for _ in range(col_count)]
    row_sizes[lib_row] = rng.randint(req_h, req_h + 3)
    col_sizes[lib_col] = rng.randint(req_w, req_w + 3)

    row_spans, sep_rows, total_h = _spans_from_sizes(row_sizes)
    col_spans, sep_cols, total_w = _spans_from_sizes(col_sizes)
    if not (14 <= total_h <= 30 and 14 <= total_w <= 30):
        return None

    candidates = _candidate_seed_cells(row_spans, col_spans, library_idx, samples)
    if len(candidates) < 2:
        return None
    if len({job["match"] for job in candidates}) < 2:
        return None

    desired = min(len(candidates), rng.randint(2, max(2, len(candidates))))
    rng.shuffle(candidates)
    chosen = []
    used_matches = set()
    for job in candidates:
        if len(chosen) < 2 and job["match"] not in used_matches:
            chosen.append(job)
            used_matches.add(job["match"])
    for job in candidates:
        if len(chosen) >= desired:
            break
        if job in chosen:
            continue
        chosen.append(job)
    chosen = chosen[:desired]
    if len(chosen) < 2 or len({job["match"] for job in chosen}) < 2:
        return None

    sample_colors = rng.sample(free_colors, len(samples))
    seed_palette = [color for color in free_colors if color not in sample_colors]
    if not seed_palette:
        return None

    grid = np.full((total_h, total_w), bg, dtype=int)
    for r in sep_rows:
        grid[r, :] = sep
    for c in sep_cols:
        grid[:, c] = sep

    lr0, lr1 = row_spans[lib_row]
    lc0, lc1 = col_spans[lib_col]
    _paint_library(grid, (lr0, lr1, lc0, lc1), samples, sample_colors, side, rng, sep)

    blank_cells = {
        (ri, ci)
        for ri in range(row_count)
        for ci in range(col_count)
        if (ri, ci) != library_idx and all(job["cell_index"] != (ri, ci) for job in chosen)
    }
    if not blank_cells:
        return None

    for job in chosen:
        anchor_r, anchor_c = rng.choice(job["positions"])
        grid[anchor_r, anchor_c] = rng.choice(seed_palette)

    if _mode_color(grid) != bg:
        return None

    try:
        parsed = _parse_grid(grid)
        if len(parsed["seeds"]) != len(chosen):
            return None
        if parsed["blank_count"] < 1:
            return None
        _ = main(grid)
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _generate_candidate(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
