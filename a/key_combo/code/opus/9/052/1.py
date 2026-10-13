from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
WALL = 5
SEED_PALETTE = (1, 2, 3, 4)


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _draw_h(grid: np.ndarray, row: int, c0: int, c1: int, color: int = WALL) -> None:
    grid[int(row), int(c0) : int(c1) + 1] = int(color)


def _draw_v(grid: np.ndarray, col: int, r0: int, r1: int, color: int = WALL) -> None:
    grid[int(r0) : int(r1) + 1, int(col)] = int(color)


def _zero_regions(grid: np.ndarray) -> tuple[np.ndarray, list[dict[str, object]]]:
    h, w = grid.shape
    labels = np.full((h, w), -1, dtype=int)
    regions: list[dict[str, object]] = []
    region_id = 0
    for r in range(h):
        for c in range(w):
            if int(grid[r, c]) != BG or int(labels[r, c]) >= 0:
                continue
            q = deque([(r, c)])
            labels[r, c] = region_id
            coords: list[tuple[int, int]] = []
            touches_border = False
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                if rr == 0 or rr == h - 1 or cc == 0 or cc == w - 1:
                    touches_border = True
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BG and int(labels[nr, nc]) < 0:
                        labels[nr, nc] = region_id
                        q.append((nr, nc))
            regions.append(
                {
                    "coords": coords,
                    "touches_border": touches_border,
                }
            )
            region_id += 1
    return labels, regions


def _seed_columns(grid: np.ndarray) -> list[int]:
    return [int(c) for c in range(grid.shape[1]) if int(grid[0, c]) in SEED_PALETTE]


def _simulate_seed_drop(grid: np.ndarray, col: int) -> tuple[int, int]:
    h = grid.shape[0]
    c = int(col)
    r = 0
    while r + 1 < h and int(grid[r + 1, c]) == BG:
        r += 1
    return int(r), c


def _solve_details(grid: np.ndarray) -> dict[str, object]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    labels, regions = _zero_regions(g)
    seeds: list[dict[str, int]] = []
    cavity_claims: dict[int, dict[str, int]] = {}
    region_winners: dict[int, int] = {}

    for c in _seed_columns(g):
        color = int(g[0, c])
        stop_r, stop_c = _simulate_seed_drop(g, c)
        info = {
            "color": color,
            "col": c,
            "stop_r": stop_r,
            "stop_c": stop_c,
            "drop": stop_r,
            "region_id": -1,
            "is_closed": 0,
            "wins": 0,
        }
        if int(stop_r) <= 0:
            seeds.append(info)
            continue
        region_id = int(labels[stop_r, stop_c])
        info["region_id"] = region_id
        if region_id < 0:
            seeds.append(info)
            continue
        closed = not bool(regions[region_id]["touches_border"])
        info["is_closed"] = int(closed)
        if closed:
            incumbent = cavity_claims.get(region_id)
            key = (int(stop_r), -int(c))
            if incumbent is None or key > (int(incumbent["drop"]), -int(incumbent["col"])):
                cavity_claims[region_id] = info
        seeds.append(info)

    for region_id, winner in cavity_claims.items():
        region_winners[int(region_id)] = int(winner["color"])
        winner["wins"] = 1

    out = g.copy()
    for c in _seed_columns(g):
        out[0, c] = BG
    for region_id, winner_color in region_winners.items():
        for r, c in regions[region_id]["coords"]:
            out[r, c] = int(winner_color)
    for seed in seeds:
        if seed["stop_r"] <= 0:
            continue
        region_id = int(seed["region_id"])
        if region_id < 0:
            out[seed["stop_r"], seed["stop_c"]] = int(seed["color"])
            continue
        if bool(regions[region_id]["touches_border"]):
            out[seed["stop_r"], seed["stop_c"]] = int(seed["color"])

    return {
        "grid": g,
        "out": out,
        "labels": labels,
        "regions": regions,
        "seeds": seeds,
        "region_winners": region_winners,
        "shape": (h, w),
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop top-row seeds straight down and flood-fill only the enclosed cavity each seed lands in.

    Input structure:
    - The background is black (0), and grey (5) pixels form a connected orthogonal line drawing.
    - The grey drawing creates several 4-connected black regions, some enclosed and some still border-connected.
    - The top row contains 2-4 distinct single-pixel seeds in colors {1,2,3,4}; each seed sits above a black vertical channel.

    Rule:
    - Each seed falls straight down through black cells until the next cell would be grey or outside the grid, then the seed stops on the last black cell.
    - Classify the seed's stop cell by the original input's 4-connected black region: if that region is enclosed, fill the whole region with the seed color; if it touches the border, place only one pixel at the stop cell.
    - If multiple seeds stop inside the same enclosed region, the seed with the larger drop distance wins; equal drop distances break toward the smaller column index.
    - Grey walls stay unchanged, and the top-row seed pixels are vacated as the seeds fall.

    Key insight:
    - The fill target is determined by vertical reachability into a capped cavity, and the final drop depth is not just motion history but also the priority signal that resolves multi-seed conflicts.
    """
    return np.asarray(_solve_details(grid)["out"], dtype=int)


def _partition_sum(total: int, parts: int, min_each: int) -> list[int]:
    values = [int(min_each)] * int(parts)
    remainder = int(total - parts * min_each)
    if remainder < 0:
        raise ValueError("partition impossible")
    for _ in range(remainder):
        values[random.randrange(parts)] += 1
    random.shuffle(values)
    return values


def _add_ledge(
    grid: np.ndarray,
    span: tuple[int, int],
    top: int,
    bottom: int,
    *,
    force_side: str | None = None,
) -> dict[str, int] | None:
    c0, c1 = map(int, span)
    if c1 - c0 + 1 < 4 or bottom - top < 6:
        return None
    row = random.randint(top + 3, bottom - 3)
    side = force_side or random.choice(["left", "right"])
    if side == "left":
        end = random.randint(c0 + 1, c1 - 1)
        _draw_h(grid, row, c0, end)
        return {"row": row, "side": 0, "solid_lo": c0, "solid_hi": end, "gap_lo": end + 1, "gap_hi": c1}
    start = random.randint(c0 + 1, c1 - 1)
    _draw_h(grid, row, start, c1)
    return {"row": row, "side": 1, "solid_lo": start, "solid_hi": c1, "gap_lo": c0, "gap_hi": start - 1}


def _pick_two_distinct(lo: int, hi: int) -> tuple[int, int]:
    a, b = sorted(random.sample(range(lo, hi + 1), 2))
    return int(a), int(b)


def _build_attempt() -> tuple[np.ndarray, dict[str, object]] | None:
    h = random.randint(15, 22)
    k = random.choice([2, 3])
    min_width = 4
    min_total_w = 5 * k + 4
    w = random.randint(max(15, min_total_w), 24)

    top = 1
    bottom = random.randint(h - 4, h - 2)
    left = random.randint(1, 2)
    right = random.randint(w - 3, w - 2)
    if bottom - top < 8 or right - left < 8:
        return None

    free_total = (right - left - 1) - (k - 1)
    if free_total < k * min_width:
        return None
    widths = _partition_sum(free_total, k, min_width)

    grid = np.zeros((h, w), dtype=int)
    _draw_h(grid, top, left, right)
    _draw_h(grid, bottom, left, right)
    _draw_v(grid, left, top, bottom)
    _draw_v(grid, right, top, bottom)

    spans: list[tuple[int, int]] = []
    dividers: list[int] = []
    cursor = left + 1
    for idx, width in enumerate(widths):
        spans.append((cursor, cursor + width - 1))
        cursor += width
        if idx != k - 1:
            dividers.append(cursor)
            _draw_v(grid, cursor, top, bottom)
            cursor += 1

    want_competition = random.random() < 0.65
    want_open = random.random() < 0.55

    eligible_competition = [i for i, (c0, c1) in enumerate(spans) if c1 - c0 + 1 >= 5]
    competition_comp = random.choice(eligible_competition) if want_competition and eligible_competition else None
    max_seed_budget = 4
    min_seed_budget = 2
    base_seeds = k
    extras = int(competition_comp is not None) + int(want_open)
    if base_seeds + extras > max_seed_budget:
        if competition_comp is not None and want_open and random.random() < 0.5:
            want_open = False
        else:
            competition_comp = None
    if base_seeds + int(competition_comp is not None) + int(want_open) < min_seed_budget:
        return None

    ledges: dict[int, dict[str, int]] = {}
    openings: list[list[int]] = [[] for _ in range(k)]
    competition_mode: str | None = None

    if competition_comp is not None:
        comp_span = spans[competition_comp]
        competition_mode = random.choice(["distance", "tie"])
        if competition_mode == "distance":
            ledge = _add_ledge(grid, comp_span, top, bottom)
            if ledge is None:
                return None
            ledges[competition_comp] = ledge
            if random.random() < 0.5:
                short_col = random.randint(ledge["solid_lo"], ledge["solid_hi"])
                long_col = random.randint(ledge["gap_lo"], ledge["gap_hi"])
            else:
                long_col = random.randint(ledge["solid_lo"], ledge["solid_hi"])
                short_col = random.randint(ledge["gap_lo"], ledge["gap_hi"])
            openings[competition_comp] = [short_col, long_col]
        else:
            c0, c1 = comp_span
            a, b = _pick_two_distinct(c0, c1)
            openings[competition_comp] = [a, b]

    for idx, span in enumerate(spans):
        c0, c1 = span
        if openings[idx]:
            continue
        if idx not in ledges and random.random() < 0.35:
            ledge = _add_ledge(grid, span, top, bottom)
            if ledge is not None:
                ledges[idx] = ledge
        if idx in ledges and random.random() < 0.55:
            ledge = ledges[idx]
            lo = ledge["solid_lo"]
            hi = ledge["solid_hi"]
            if lo <= hi:
                open_col = random.randint(lo, hi)
            else:
                open_col = random.randint(c0, c1)
        else:
            open_col = random.randint(c0, c1)
        openings[idx] = [open_col]

    for cols in openings:
        for col in cols:
            grid[top, int(col)] = BG

    open_seed_cols: list[int] = []
    if want_open:
        side = random.choice(["left", "right"])
        if side == "left" and left > 0:
            open_col = random.randint(0, left - 1)
            if random.random() < 0.5:
                bar_row = random.randint(top + 4, h - 2)
                _draw_h(grid, bar_row, 0, left)
            open_seed_cols.append(open_col)
        elif side == "right" and right < w - 1:
            open_col = random.randint(right + 1, w - 1)
            if random.random() < 0.5:
                bar_row = random.randint(top + 4, h - 2)
                _draw_h(grid, bar_row, right, w - 1)
            open_seed_cols.append(open_col)

    seed_columns = [col for cols in openings for col in cols] + open_seed_cols
    if not (2 <= len(seed_columns) <= 4):
        return None
    if len(set(seed_columns)) != len(seed_columns):
        return None

    seed_colors = random.sample(SEED_PALETTE, len(seed_columns))
    random.shuffle(seed_colors)
    seed_by_col = dict(zip(seed_columns, seed_colors))
    for col, color in seed_by_col.items():
        grid[0, int(col)] = int(color)

    meta: dict[str, object] = {
        "spans": spans,
        "openings": openings,
        "open_seed_cols": open_seed_cols,
        "competition_comp": competition_comp,
        "competition_mode": competition_mode,
        "seed_by_col": seed_by_col,
        "top": top,
        "bottom": bottom,
        "left": left,
        "right": right,
    }
    return grid, meta


def _validate_semantics(grid: np.ndarray, meta: dict[str, object]) -> bool:
    details = _solve_details(grid)
    out = np.asarray(details["out"], dtype=int)
    regions = details["regions"]
    seeds = {int(seed["col"]): seed for seed in details["seeds"]}
    openings: list[list[int]] = meta["openings"]  # type: ignore[assignment]
    open_seed_cols: list[int] = meta["open_seed_cols"]  # type: ignore[assignment]
    seed_by_col: dict[int, int] = meta["seed_by_col"]  # type: ignore[assignment]
    competition_comp = meta["competition_comp"]
    competition_mode = meta["competition_mode"]

    if not (2 <= len(seed_by_col) <= 4):
        return False
    if len(set(seed_by_col.values())) != len(seed_by_col):
        return False
    if any(int(grid[0, c]) != int(seed_by_col[c]) for c in seed_by_col):
        return False
    if any(int(out[0, c]) != BG for c in seed_by_col):
        return False

    comp_region_ids: list[int] = []
    winners_by_comp: list[int] = []
    for cols in openings:
        region_ids = {int(seeds[c]["region_id"]) for c in cols}
        if len(region_ids) != 1:
            return False
        region_id = region_ids.pop()
        if region_id < 0 or bool(regions[region_id]["touches_border"]):
            return False
        comp_region_ids.append(region_id)
        contenders = [seeds[c] for c in cols]
        winner = max(contenders, key=lambda item: (int(item["drop"]), -int(item["col"])))
        winners_by_comp.append(int(winner["color"]))
        region_colors = {int(out[r, c]) for r, c in regions[region_id]["coords"]}
        if region_colors != {int(winner["color"])}:
            return False

    if len(set(comp_region_ids)) != len(comp_region_ids):
        return False

    for col in open_seed_cols:
        seed = seeds[col]
        if int(seed["region_id"]) < 0:
            return False
        if not bool(regions[int(seed["region_id"])]["touches_border"]):
            return False
        color = int(seed["color"])
        if int(out[int(seed["stop_r"]), int(seed["stop_c"])]) != color:
            return False
        if int(np.count_nonzero(out == color)) != 1:
            return False

    if competition_comp is not None:
        cols = openings[int(competition_comp)]
        if len(cols) < 2:
            return False
        contenders = [seeds[c] for c in cols]
        drops = [int(seed["drop"]) for seed in contenders]
        winner = max(contenders, key=lambda item: (int(item["drop"]), -int(item["col"])))
        if competition_mode == "distance":
            if len(set(drops)) < 2:
                return False
        elif competition_mode == "tie":
            best_drop = max(drops)
            tied_cols = [int(seed["col"]) for seed in contenders if int(seed["drop"]) == best_drop]
            if len(tied_cols) < 2:
                return False
            if int(winner["col"]) != min(tied_cols):
                return False
        losing = [seed for seed in contenders if int(seed["color"]) != int(winner["color"])]
        for seed in losing:
            if int(np.count_nonzero(out == int(seed["color"]))) != 0:
                return False

    changed = int(np.count_nonzero(out != grid))
    if changed < len(seed_by_col) + 4:
        return False
    if len(comp_region_ids) < 2:
        return False
    if len(set(winners_by_comp + [seed_by_col[c] for c in open_seed_cols])) < 2:
        return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(500):
        built = _build_attempt()
        if built is None:
            continue
        grid, meta = built
        if _validate_semantics(grid, meta):
            return grid
    raise RuntimeError("failed to generate a valid gravity-cavity puzzle")
