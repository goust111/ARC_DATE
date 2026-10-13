import random
from collections import Counter, deque

import numpy as np


BACKGROUND = 0
BLUE = 1
RED = 2
YELLOW = 4
GREY = 5
PINK = 6
ORANGE = 7
TEAL = 8
PANEL_COLORS = (BLUE, GREY, ORANGE, TEAL)

MIN_BAR_AREA = 4
RECOLOR_PERIOD = 3
GEN_RETRIES = 600


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _components_from_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _find_monochromatic_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    out: list[dict] = []
    for color in sorted(set(map(int, grid.ravel())) - {bg}):
        for coords in _components_from_mask(grid == color):
            out.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": _bbox(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return out


def _find_foreground_components(grid: np.ndarray, bg: int = BACKGROUND) -> list[dict]:
    out: list[dict] = []
    for coords in _components_from_mask(grid != bg):
        out.append(
            {
                "coords": coords,
                "bbox": _bbox(coords),
                "area": int(coords.shape[0]),
            }
        )
    return out


def _clean_panel(panel: np.ndarray) -> tuple[np.ndarray, dict]:
    comps = _find_monochromatic_components(panel, bg=BACKGROUND)
    large = [comp for comp in comps if comp["area"] >= MIN_BAR_AREA]
    small = [comp for comp in comps if comp["area"] < MIN_BAR_AREA]

    owner = np.full(panel.shape, -1, dtype=int)
    for idx, comp in enumerate(large):
        coords = comp["coords"]
        owner[coords[:, 0], coords[:, 1]] = idx

    out = panel.copy()
    stats = {"merged": 0, "erased": 0, "ties": 0}
    for comp in small:
        counts: Counter[int] = Counter()
        for r, c in comp["coords"]:
            for nr, nc in _neighbors4(int(r), int(c)):
                if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1]:
                    idx = int(owner[nr, nc])
                    if idx >= 0:
                        counts[idx] += 1

        coords = comp["coords"]
        if sum(counts.values()) >= 2 and counts:
            best_count = max(counts.values())
            candidates = [idx for idx, cnt in counts.items() if cnt == best_count]
            if len(candidates) > 1:
                stats["ties"] += 1
            winner = min(candidates, key=lambda idx: (large[idx]["bbox"][0], large[idx]["bbox"][1]))
            out[coords[:, 0], coords[:, 1]] = large[winner]["color"]
            stats["merged"] += 1
        else:
            out[coords[:, 0], coords[:, 1]] = BACKGROUND
            stats["erased"] += 1
    return out, stats


def _overlay_panels(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    out = bottom.copy()
    out[top != BACKGROUND] = top[top != BACKGROUND]
    return out


def _recolor_every_third(grid: np.ndarray) -> np.ndarray:
    comps = sorted(_find_foreground_components(grid, bg=BACKGROUND), key=lambda comp: (comp["bbox"][0], comp["bbox"][1]))
    out = grid.copy()
    for idx, comp in enumerate(comps):
        if idx % RECOLOR_PERIOD == 0:
            coords = comp["coords"]
            out[coords[:, 0], coords[:, 1]] = RED
    return out


def _solve_with_details(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    sep_rows = [r for r in range(g.shape[0]) if np.all(g[r, :] == PINK)]
    assert len(sep_rows) == 1, "expected one full-width pink separator row"
    sep = int(sep_rows[0])
    top = g[:sep, :]
    bottom = g[sep + 1 :, :]
    assert top.shape == bottom.shape and top.size > 0, "expected two equal panels"

    top_clean, top_stats = _clean_panel(top)
    bottom_clean, bottom_stats = _clean_panel(bottom)
    overlay = _overlay_panels(top_clean, bottom_clean)
    out = _recolor_every_third(overlay)
    details = {
        "top_clean": top_clean,
        "bottom_clean": bottom_clean,
        "overlay": overlay,
        "top_stats": top_stats,
        "bottom_stats": bottom_stats,
        "components": sorted(
            _find_foreground_components(overlay, bg=BACKGROUND),
            key=lambda comp: (comp["bbox"][0], comp["bbox"][1]),
        ),
    }
    return out, details


def main(grid: np.ndarray) -> np.ndarray:
    """
    Clean each panel's small yellow noise, overlay the two panels, then recolor every third merged stripe red.

    Input structure:
    - The grid has black background and one full-width pink separator row.
    - The separator splits the grid into equal-height upper and lower panels.
    - Each panel contains several horizontal colored bars plus yellow 1-2 pixel noise components.

    Rule:
    - In each panel separately, find 4-connected monochromatic components.
    - Treat components with area < 4 as noise and larger ones as true bars.
    - For each noise component, count its 4-neighbor contacts with the large components:
      recolor it to the most-contacted bar color if the total contact count is at least 2,
      breaking ties toward the upper bar; otherwise erase it to black.
    - Remove the separator row, align the cleaned panels cell by cell, and keep the upper-panel color
      wherever both panels have foreground.
    - In the merged panel, sort all 4-connected foreground components by bbox top row then left column,
      and recolor ranks 1, 4, 7, ... fully red.

    Key insight:
    - Noise cleaning changes the shapes that enter the overlay, and the overlay creates a new set of merged
      foreground components whose sorted order controls the final periodic recoloring.
    """
    return _solve_with_details(grid)[0]


def _row_overlap(a: dict, b: dict) -> bool:
    return max(a["r0"], b["r0"]) <= min(a["r0"] + a["h"] - 1, b["r0"] + b["h"] - 1)


def _interval_overlap_len(c0: int, w0: int, c1: int, w1: int) -> int:
    left = max(c0, c1)
    right = min(c0 + w0 - 1, c1 + w1 - 1)
    return max(0, right - left + 1)


def _sample_row_layout(height: int, n_bars: int) -> list[tuple[int, int]] | None:
    for _ in range(120):
        heights = [1 + int(random.random() < 0.35) for _ in range(n_bars)]
        min_total = sum(heights) + (n_bars - 1)
        if min_total > height:
            continue
        extra = height - min_total
        gaps = [0] * (n_bars + 1)
        for _ in range(extra):
            gaps[random.randrange(n_bars + 1)] += 1

        rows = []
        r = gaps[0]
        for i, h in enumerate(heights):
            rows.append((r, h))
            r += h
            if i < n_bars - 1:
                r += 1 + gaps[i + 1]
        return rows
    return None


def _sample_partial_overlap_start(width: int, other_c0: int, other_w: int, my_w: int) -> int | None:
    starts = []
    for c0 in range(width - my_w + 1):
        overlap = _interval_overlap_len(c0, my_w, other_c0, other_w)
        if 2 <= overlap < min(my_w, other_w):
            starts.append(c0)
    if not starts:
        return None
    return random.choice(starts)


def _sample_disjoint_start(width: int, my_w: int, forbids: list[tuple[int, int]]) -> int | None:
    starts = list(range(width - my_w + 1))
    random.shuffle(starts)
    for c0 in starts:
        if all(_interval_overlap_len(c0, my_w, f0, fw) == 0 for f0, fw in forbids):
            return c0
    return None


def _render_panel(shape: tuple[int, int], bars: list[dict]) -> np.ndarray:
    panel = np.zeros(shape, dtype=int)
    for bar in bars:
        panel[bar["r0"] : bar["r0"] + bar["h"], bar["c0"] : bar["c0"] + bar["w"]] = bar["color"]
    return panel


def _sample_panel_pair(height: int, width: int) -> tuple[list[dict], list[dict]] | None:
    n_top = random.choice((3, 4))
    n_bottom = random.choice((3, 4))
    top_rows = _sample_row_layout(height, n_top)
    bottom_rows = _sample_row_layout(height, n_bottom)
    if top_rows is None or bottom_rows is None:
        return None

    top_colors = random.sample(list(PANEL_COLORS), n_top)
    bottom_colors = random.sample(list(PANEL_COLORS), n_bottom)

    top_bars = []
    for (r0, h), color in zip(top_rows, top_colors):
        w = random.randint(4, 8)
        c0 = random.randint(0, width - w)
        top_bars.append({"r0": r0, "h": h, "w": w, "c0": c0, "color": color})

    bottom_specs = []
    for (r0, h), color in zip(bottom_rows, bottom_colors):
        w = random.randint(4, 8)
        bottom_specs.append({"r0": r0, "h": h, "w": w, "color": color})

    overlap_pairs = [(i, j) for i, top_bar in enumerate(top_bars) for j, bot_bar in enumerate(bottom_specs) if _row_overlap(top_bar, bot_bar)]
    if not overlap_pairs:
        return None
    overlap_i, overlap_j = random.choice(overlap_pairs)

    chosen_c0 = _sample_partial_overlap_start(width, top_bars[overlap_i]["c0"], top_bars[overlap_i]["w"], bottom_specs[overlap_j]["w"])
    if chosen_c0 is None:
        return None

    bottom_bars: list[dict] = []
    for j, spec in enumerate(bottom_specs):
        bar = dict(spec)
        if j == overlap_j:
            bar["c0"] = chosen_c0
        else:
            forbids = []
            for i, top_bar in enumerate(top_bars):
                if _row_overlap(top_bar, bar):
                    forbids.append((top_bar["c0"], top_bar["w"]))
            c0 = _sample_disjoint_start(width, bar["w"], forbids)
            if c0 is None:
                c0 = random.randint(0, width - bar["w"])
            bar["c0"] = c0
        bottom_bars.append(bar)

    return top_bars, bottom_bars


def _contact_counts_from_base(base_panel: np.ndarray, cells: list[tuple[int, int]]) -> Counter[int]:
    large = [comp for comp in _find_monochromatic_components(base_panel, bg=BACKGROUND) if comp["area"] >= MIN_BAR_AREA]
    owner = np.full(base_panel.shape, -1, dtype=int)
    for idx, comp in enumerate(large):
        coords = comp["coords"]
        owner[coords[:, 0], coords[:, 1]] = idx
    counts: Counter[int] = Counter()
    for r, c in cells:
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < base_panel.shape[0] and 0 <= nc < base_panel.shape[1]:
                idx = int(owner[nr, nc])
                if idx >= 0:
                    counts[idx] += 1
    return counts


def _can_place_noise_component(panel: np.ndarray, cells: list[tuple[int, int]]) -> bool:
    cell_set = set(cells)
    for r, c in cells:
        if not (0 <= r < panel.shape[0] and 0 <= c < panel.shape[1]):
            return False
        if int(panel[r, c]) != BACKGROUND:
            return False
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < panel.shape[0] and 0 <= nc < panel.shape[1] and (nr, nc) not in cell_set and int(panel[nr, nc]) == YELLOW:
                return False
    return True


def _domino_merge_candidates(panel: np.ndarray, bars: list[dict]) -> list[list[tuple[int, int]]]:
    h, w = panel.shape
    candidates: list[list[tuple[int, int]]] = []
    for bar in bars:
        r0, bh, c0, bw = bar["r0"], bar["h"], bar["c0"], bar["w"]
        if r0 > 0:
            for c in range(c0, c0 + bw - 1):
                cells = [(r0 - 1, c), (r0 - 1, c + 1)]
                if _can_place_noise_component(panel, cells):
                    candidates.append(cells)
        if r0 + bh < h:
            for c in range(c0, c0 + bw - 1):
                cells = [(r0 + bh, c), (r0 + bh, c + 1)]
                if _can_place_noise_component(panel, cells):
                    candidates.append(cells)
        if bh >= 2 and c0 > 0:
            for r in range(r0, r0 + bh - 1):
                cells = [(r, c0 - 1), (r + 1, c0 - 1)]
                if _can_place_noise_component(panel, cells):
                    candidates.append(cells)
        if bh >= 2 and c0 + bw < w:
            for r in range(r0, r0 + bh - 1):
                cells = [(r, c0 + bw), (r + 1, c0 + bw)]
                if _can_place_noise_component(panel, cells):
                    candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _tie_merge_candidates(panel: np.ndarray, bars: list[dict]) -> list[list[tuple[int, int]]]:
    candidates: list[list[tuple[int, int]]] = []
    ordered = sorted(bars, key=lambda bar: (bar["r0"], bar["c0"]))
    for upper in ordered:
        for lower in ordered:
            if upper is lower:
                continue
            upper_r1 = upper["r0"] + upper["h"] - 1
            if lower["r0"] != upper_r1 + 2:
                continue
            overlap_left = max(upper["c0"], lower["c0"])
            overlap_right = min(upper["c0"] + upper["w"] - 1, lower["c0"] + lower["w"] - 1)
            for c in range(overlap_left, overlap_right + 1):
                cells = [(upper_r1 + 1, c)]
                if _can_place_noise_component(panel, cells):
                    candidates.append(cells)
    random.shuffle(candidates)
    return candidates


def _place_merge_noise(panel: np.ndarray, base_panel: np.ndarray, bars: list[dict], target_count: int, allow_tie: bool) -> int:
    placed = 0
    if allow_tie:
        for cells in _tie_merge_candidates(panel, bars):
            counts = _contact_counts_from_base(base_panel, cells)
            if sum(counts.values()) >= 2:
                for r, c in cells:
                    panel[r, c] = YELLOW
                placed += 1
                break
    while placed < target_count:
        progressed = False
        for cells in _domino_merge_candidates(panel, bars):
            counts = _contact_counts_from_base(base_panel, cells)
            if sum(counts.values()) >= 2:
                for r, c in cells:
                    panel[r, c] = YELLOW
                placed += 1
                progressed = True
                break
        if not progressed:
            break
    return placed


def _random_erase_candidate(panel: np.ndarray, base_panel: np.ndarray) -> list[tuple[int, int]] | None:
    h, w = panel.shape
    for _ in range(250):
        size = 1 if random.random() < 0.55 else 2
        r = random.randrange(h)
        c = random.randrange(w)
        if size == 1:
            cells = [(r, c)]
        else:
            dr, dc = random.choice(((1, 0), (0, 1)))
            cells = [(r, c), (r + dr, c + dc)]
        if not _can_place_noise_component(panel, cells):
            continue
        counts = _contact_counts_from_base(base_panel, cells)
        if sum(counts.values()) < 2:
            return cells
    return None


def _add_noise(panel: np.ndarray, bars: list[dict]) -> bool:
    base_panel = panel.copy()
    target_total = random.randint(4, 6)
    merge_target = random.randint(1, 3)
    erase_target = target_total - merge_target
    allow_tie = random.random() < 0.45

    if _place_merge_noise(panel, base_panel, bars, merge_target, allow_tie) < merge_target:
        return False
    for _ in range(erase_target):
        cells = _random_erase_candidate(panel, base_panel)
        if cells is None:
            return False
        for r, c in cells:
            panel[r, c] = YELLOW
    return True


def _stack_input(top: np.ndarray, bottom: np.ndarray) -> np.ndarray:
    separator = np.full((1, top.shape[1]), PINK, dtype=int)
    return np.vstack([top, separator, bottom])


def _panel_noise_components(panel: np.ndarray) -> list[dict]:
    return [comp for comp in _find_monochromatic_components(panel, bg=BACKGROUND) if comp["area"] < MIN_BAR_AREA]


def _validate_instance(inp: np.ndarray) -> bool:
    solved, details = _solve_with_details(inp)
    sep = int(np.where(np.all(inp == PINK, axis=1))[0][0])
    top = inp[:sep, :]
    bottom = inp[sep + 1 :, :]

    top_small = _panel_noise_components(top)
    bottom_small = _panel_noise_components(bottom)
    if not (4 <= len(top_small) <= 8 and 4 <= len(bottom_small) <= 8):
        return False
    if any(comp["color"] != YELLOW or comp["area"] not in (1, 2) for comp in top_small + bottom_small):
        return False
    if details["top_stats"]["merged"] < 1 or details["top_stats"]["erased"] < 1:
        return False
    if details["bottom_stats"]["merged"] < 1 or details["bottom_stats"]["erased"] < 1:
        return False

    top_clean = details["top_clean"]
    bottom_clean = details["bottom_clean"]
    overlay = details["overlay"]
    raw_overlay = _overlay_panels(top, bottom)

    overlap = int(np.sum((top_clean != BACKGROUND) & (bottom_clean != BACKGROUND)))
    top_only = int(np.sum((top_clean != BACKGROUND) & (bottom_clean == BACKGROUND)))
    bottom_only = int(np.sum((top_clean == BACKGROUND) & (bottom_clean != BACKGROUND)))
    if overlap == 0 or top_only == 0 or bottom_only == 0:
        return False
    if np.array_equal(raw_overlay, overlay):
        return False

    comps = details["components"]
    if len(comps) < 4:
        return False
    recolored = _recolor_every_third(overlay)
    if not np.array_equal(recolored, solved):
        return False
    if not np.any(solved == RED):
        return False
    if len(set(map(int, inp.ravel()))) < 5:
        return False
    return True


def _generate_attempt() -> np.ndarray | None:
    height = random.randint(8, 11)
    width = random.randint(18, 26)
    pair = _sample_panel_pair(height, width)
    if pair is None:
        return None
    top_bars, bottom_bars = pair
    top = _render_panel((height, width), top_bars)
    bottom = _render_panel((height, width), bottom_bars)
    if not _add_noise(top, top_bars):
        return None
    if not _add_noise(bottom, bottom_bars):
        return None

    inp = _stack_input(top, bottom)
    try:
        out = main(inp)
    except Exception:
        return None
    if out.shape != (height, width):
        return None
    if not _validate_instance(inp):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        inp = _generate_attempt()
        if inp is not None:
            return inp
    raise RuntimeError("failed to generate a valid puzzle instance")
