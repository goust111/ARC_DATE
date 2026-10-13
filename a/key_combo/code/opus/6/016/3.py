import random

import numpy as np


BG = 0
BLUE = 1
RED = 2

MIN_BAR_HEIGHT = 2
MAX_BAR_HEIGHT = 5
MAX_CHAIN_LEN = 4
MAX_GRID_SIDE = 30
MAX_GENERATION_ATTEMPTS = 600


def _parse_bars(grid: np.ndarray) -> tuple[int, list[dict]]:
    g = np.asarray(grid, dtype=int)
    cells = np.argwhere(g != BG)
    if cells.size == 0:
        raise ValueError("expected at least one bar")

    baseline = int(cells[:, 0].max())
    bars = []
    for col in range(g.shape[1]):
        rows = np.flatnonzero(g[:, col] != BG)
        if rows.size == 0:
            continue
        color = int(g[rows[0], col])
        if color not in (BLUE, RED):
            raise ValueError("unexpected bar color")
        if not np.all(g[rows, col] == color):
            raise ValueError("bar must be monochromatic")
        if int(rows[-1]) != baseline:
            raise ValueError("bars must share one baseline")
        if int(rows[-1] - rows[0] + 1) != int(rows.size):
            raise ValueError("bar must be vertically contiguous")
        bars.append(
            {
                "color": color,
                "col": int(col),
                "top": int(rows[0]),
                "bottom": baseline,
                "height": int(rows.size),
            }
        )
    return baseline, bars


def _extract_chains(bars: list[dict]) -> tuple[list[dict], list[dict]]:
    ordered = sorted(bars, key=lambda bar: bar["col"])
    by_col = {bar["col"]: bar for bar in ordered}

    chains = []
    covered_red_cols = set()
    for bar in ordered:
        if bar["color"] != BLUE:
            continue
        members = [bar]
        current = bar
        while True:
            next_col = current["col"] + current["height"]
            next_bar = by_col.get(next_col)
            if next_bar is None or next_bar["color"] != RED:
                break
            members.append(next_bar)
            covered_red_cols.add(next_bar["col"])
            current = next_bar

        chains.append(
            {
                "start_col": int(members[0]["col"]),
                "end_col": int(members[-1]["col"] + members[-1]["height"]),
                "total": int(sum(member["height"] for member in members)),
                "color": int(members[0]["color"]),
                "members": members,
            }
        )

    unchained_red = [
        bar for bar in ordered if bar["color"] == RED and bar["col"] not in covered_red_cols
    ]
    return chains, unchained_red


def _beats(chain_a: dict, chain_b: dict) -> bool:
    if chain_a["total"] != chain_b["total"]:
        return bool(chain_a["total"] > chain_b["total"])
    return bool(chain_a["start_col"] < chain_b["start_col"])


def _segments_overlap(chain_a: dict, chain_b: dict) -> bool:
    return max(chain_a["start_col"], chain_b["start_col"]) < min(
        chain_a["end_col"], chain_b["end_col"]
    )


def _cutoff_for_member(member: dict, owner: dict, chains: list[dict]) -> int:
    start = int(member["col"])
    end = start + int(member["height"])
    cutoff = end
    for other in chains:
        if other is owner or not _beats(other, owner):
            continue
        overlap_start = max(start, int(other["start_col"]))
        overlap_end = min(end, int(other["end_col"]))
        if overlap_start < overlap_end:
            cutoff = min(cutoff, overlap_start)
    return cutoff


def main(grid: np.ndarray) -> np.ndarray:
    """
    Rotate blue-started bar chains into horizontal segments and truncate losers where different chains collide.

    Input structure:
    - The background is black and the foreground consists of one-cell-wide vertical bars ending on one shared baseline.
    - Blue bars mark chain starts; red bars may continue a chain when the next bar's column equals the current bar's column plus its height.
    - There may also be red bars that are not reached from any blue start.

    Rule:
    - Identify every blue-started chain by repeatedly following the exact height-as-offset linkage rule through red bars.
    - Erase each chain member's vertical pixels, then rotate every member into a horizontal bar drawn rightward from its original bottom cell, using the starter blue color.
    - When horizontal bars from different chains overlap, compare whole-chain pixel totals; the stronger chain keeps the overlap and the weaker bar is cut at its first collision column, while unchained red bars stay vertical.

    Key insight:
    - A bar's height simultaneously encodes the next chain link and the rotated width, so collisions happen only after rotation and are resolved by chain-level totals rather than by individual bar lengths.
    """
    g = np.asarray(grid, dtype=int)
    baseline, bars = _parse_bars(g)
    chains, _unchained_red = _extract_chains(bars)

    out = g.copy()
    for chain in chains:
        for member in chain["members"]:
            out[int(member["top"]) : baseline + 1, int(member["col"])] = BG

    for chain in chains:
        for member in chain["members"]:
            start = int(member["col"])
            cutoff = _cutoff_for_member(member, chain, chains)
            if cutoff > start:
                out[baseline, start:cutoff] = int(chain["color"])

    return out


def _sample_heights() -> list[int]:
    length = random.choices([1, 2, 3, 4], weights=[2, 4, 3, 2])[0]
    return [random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT) for _ in range(length)]


def _make_chain(start_col: int, heights: list[int]) -> dict:
    members = []
    col = int(start_col)
    for index, height in enumerate(heights):
        members.append({"col": col, "height": int(height), "color": BLUE if index == 0 else RED})
        col += int(height)
    return {
        "start_col": int(start_col),
        "end_col": int(col),
        "total": int(sum(heights)),
        "members": members,
    }


def _member_cols(chain: dict) -> set[int]:
    return {int(member["col"]) for member in chain["members"]}


def _place_relative_chain(
    heights: list[int], anchor: dict, used_cols: set[int], mode: str
) -> dict | None:
    total = int(sum(heights))
    if mode == "inside":
        candidates = list(range(int(anchor["start_col"]) + 1, int(anchor["end_col"]) - 1))
    elif mode == "left_overlap":
        lo = max(0, int(anchor["start_col"]) - total + 1)
        hi = int(anchor["start_col"]) - 1
        candidates = list(range(lo, hi + 1))
    else:
        raise ValueError("unknown placement mode")

    random.shuffle(candidates)
    for start_col in candidates:
        chain = _make_chain(start_col, heights)
        if chain["end_col"] > MAX_GRID_SIDE:
            continue
        if _member_cols(chain) & used_cols:
            continue
        if not _segments_overlap(chain, anchor):
            continue
        return chain
    return None


def _place_disjoint_chain(heights: list[int], chains: list[dict], used_cols: set[int]) -> dict | None:
    total = int(sum(heights))
    candidates = list(range(0, MAX_GRID_SIDE - total + 1))
    random.shuffle(candidates)
    for start_col in candidates:
        chain = _make_chain(start_col, heights)
        if _member_cols(chain) & used_cols:
            continue
        if any(_segments_overlap(chain, other) for other in chains):
            continue
        return chain
    return None


def _sample_chain_with_relation(
    reference_total: int, forbidden_totals: set[int], relation: str
) -> list[int] | None:
    for _ in range(200):
        heights = _sample_heights()
        total = int(sum(heights))
        if total in forbidden_totals:
            continue
        if relation == "stronger" and total <= reference_total:
            continue
        if relation == "weaker" and total >= reference_total:
            continue
        return heights
    return None


def _build_chains() -> list[dict] | None:
    for _ in range(300):
        first = _sample_heights()
        second = _sample_heights()
        if sum(first) == sum(second):
            continue

        first_wins = random.random() < 0.5
        if first_wins != (sum(first) > sum(second)):
            first, second = second, first

        start_first = random.randint(2, 7)
        chain_first = _make_chain(start_first, first)
        if chain_first["end_col"] > MAX_GRID_SIDE - 1:
            continue

        mode = "left_overlap" if first_wins else "inside"
        chain_second = _place_relative_chain(second, chain_first, _member_cols(chain_first), mode)
        if chain_second is None:
            continue

        chains = [chain_first, chain_second]
        used_cols = _member_cols(chain_first) | _member_cols(chain_second)
        if random.random() < 0.55:
            anchor = random.choice(chains)
            stronger = random.random() < 0.5
            relation = "stronger" if stronger else "weaker"
            heights_third = _sample_chain_with_relation(
                int(anchor["total"]), {chain["total"] for chain in chains}, relation
            )
            if heights_third is None:
                continue
            mode = "inside" if stronger else "left_overlap"
            if random.random() < 0.25:
                chain_third = _place_disjoint_chain(heights_third, chains, used_cols)
            else:
                chain_third = _place_relative_chain(heights_third, anchor, used_cols, mode)
            if chain_third is None:
                continue
            chains.append(chain_third)

        totals = [int(chain["total"]) for chain in chains]
        if len(set(totals)) != len(totals):
            continue
        if not any(_segments_overlap(chains[i], chains[j]) for i in range(len(chains)) for j in range(i + 1, len(chains))):
            continue
        return chains
    return None


def _place_unchained_reds(chains: list[dict]) -> list[dict]:
    used_cols = set()
    blocked_cols = set()
    for chain in chains:
        used_cols |= _member_cols(chain)
        blocked_cols.update(range(int(chain["start_col"]), int(chain["end_col"])))
        for member in chain["members"]:
            blocked_cols.add(int(member["col"]) + int(member["height"]))

    candidates = [col for col in range(MAX_GRID_SIDE) if col not in used_cols and col not in blocked_cols]
    random.shuffle(candidates)

    extras = []
    for _ in range(random.choices([0, 1, 2], weights=[3, 4, 2])[0]):
        if not candidates:
            break
        col = candidates.pop()
        extras.append({"col": int(col), "height": random.randint(MIN_BAR_HEIGHT, MAX_BAR_HEIGHT)})
    return extras


def _render_input(chains: list[dict], extras: list[dict]) -> np.ndarray:
    all_heights = [member["height"] for chain in chains for member in chain["members"]]
    all_heights.extend(extra["height"] for extra in extras)
    max_height = max(all_heights)

    top_padding = random.randint(5, 14)
    height = min(MAX_GRID_SIDE, max_height + top_padding)
    baseline = height - 1

    rightmost = max(
        [int(chain["end_col"]) for chain in chains]
        + [int(extra["col"]) + 1 for extra in extras]
    )
    width = min(MAX_GRID_SIDE, max(rightmost, 8) + random.randint(0, min(2, MAX_GRID_SIDE - max(rightmost, 8))))

    grid = np.zeros((height, width), dtype=int)
    for chain in chains:
        for member in chain["members"]:
            col = int(member["col"])
            bar_height = int(member["height"])
            color = int(member["color"])
            grid[baseline - bar_height + 1 : baseline + 1, col] = color
    for extra in extras:
        col = int(extra["col"])
        bar_height = int(extra["height"])
        grid[baseline - bar_height + 1 : baseline + 1, col] = RED
    return grid


def _scene_ok(grid: np.ndarray) -> bool:
    try:
        baseline, bars = _parse_bars(grid)
        chains, extras = _extract_chains(bars)
    except Exception:
        return False

    if not (2 <= len(chains) <= 3):
        return False
    totals = [int(chain["total"]) for chain in chains]
    if len(set(totals)) != len(totals):
        return False
    if not any(_segments_overlap(chains[i], chains[j]) for i in range(len(chains)) for j in range(i + 1, len(chains))):
        return False

    if any(not (0 <= extra["col"] < grid.shape[1]) for extra in extras):
        return False
    for extra in extras:
        if any(int(chain["start_col"]) <= int(extra["col"]) < int(chain["end_col"]) for chain in chains):
            return False

    partial_cut = False
    for chain in chains:
        for member in chain["members"]:
            start = int(member["col"])
            end = start + int(member["height"])
            cutoff = _cutoff_for_member(member, chain, chains)
            if start < cutoff < end:
                partial_cut = True
    if not partial_cut:
        return False

    out = main(grid)
    if out.shape != grid.shape or max(out.shape) > MAX_GRID_SIDE:
        return False
    if baseline < 0:
        return False
    if np.array_equal(out, grid):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        chains = _build_chains()
        if chains is None:
            continue
        extras = _place_unchained_reds(chains)
        grid = _render_input(chains, extras)
        if _scene_ok(grid):
            return grid
    raise RuntimeError("failed to generate a valid collision-truncation chain puzzle")
