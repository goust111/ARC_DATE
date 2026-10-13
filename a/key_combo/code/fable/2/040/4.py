from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
SHELF = 5
BLOCK_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)


def _components_by_color(g: np.ndarray, colors: set[int]) -> list[dict]:
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in colors or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc]:
                        if int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            q.append((nr, nc))
            arr = np.array(coords, dtype=int)
            r0, c0 = arr.min(axis=0)
            r1, c1 = arr.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": arr,
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                    "area": int(arr.shape[0]),
                }
            )
    return comps


def _find_blocks(g: np.ndarray) -> list[dict]:
    colors = set(map(int, np.unique(g))) - {BG, SHELF}
    assert colors, "expected at least one colored square"
    blocks: list[dict] = []
    for comp in _components_by_color(g, colors):
        r0, c0, r1, c1 = comp["bbox"]
        n = r1 - r0 + 1
        assert n == c1 - c0 + 1, "each movable object must have a square bbox"
        assert comp["area"] == n * n, "each movable square must be solid"
        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(patch == int(comp["color"])), "square bbox must be filled by one color"
        blocks.append(
            {
                "color": int(comp["color"]),
                "r": int(r0),
                "c": int(c0),
                "n": int(n),
                "bottom": int(r1),
            }
        )

    bottoms = [b["bottom"] for b in blocks]
    assert len(bottoms) == len(set(bottoms)), "initial falling order must be unambiguous"
    return sorted(blocks, key=lambda b: (-int(b["bottom"]), int(b["c"])))


def _gap_runs(row: np.ndarray) -> list[tuple[int, int]]:
    w = row.shape[0]
    gaps: list[tuple[int, int]] = []
    c = 0
    while c < w:
        if int(row[c]) != BG:
            assert int(row[c]) == SHELF, "shelf rows may contain only shelf and background cells"
            c += 1
            continue
        start = c
        while c < w and int(row[c]) == BG:
            c += 1
        end = c - 1
        if start > 0 and end + 1 < w and int(row[start - 1]) == SHELF and int(row[end + 1]) == SHELF:
            gaps.append((int(start), int(end)))
    return gaps


def _shelves_and_gaps(g: np.ndarray) -> tuple[np.ndarray, dict[int, list[tuple[int, int]]]]:
    shelf_rows = [r for r in range(g.shape[0]) if np.any(g[r] == SHELF)]
    assert shelf_rows, "expected at least one grey shelf"

    shelf_mask = np.zeros(g.shape, dtype=bool)
    gaps_by_row: dict[int, list[tuple[int, int]]] = {}
    for r in shelf_rows:
        row_colors = set(map(int, np.unique(g[r])))
        assert row_colors <= {BG, SHELF}, "shelf rows must not contain square colors"
        gaps = _gap_runs(g[r])
        assert gaps, "each shelf must contain at least one bounded gap"
        gaps_by_row[int(r)] = gaps
        shelf_mask[r] = g[r] == SHELF
    return shelf_mask, gaps_by_row


def _area_hits(mask: np.ndarray, r: int, c: int, n: int) -> bool:
    return bool(mask[int(r) : int(r) + int(n), int(c) : int(c) + int(n)].any())


def _fits_gap(c: int, n: int, gaps: list[tuple[int, int]]) -> bool:
    right = int(c) + int(n) - 1
    return any(gl <= int(c) and right <= gr for gl, gr in gaps)


def _drop_one(
    block: dict,
    shape: tuple[int, int],
    shelf_mask: np.ndarray,
    gaps_by_row: dict[int, list[tuple[int, int]]],
    settled: np.ndarray,
) -> tuple[int, int, list[dict]]:
    h, w = shape
    r = int(block["r"])
    c = int(block["c"])
    n = int(block["n"])
    events: list[dict] = []
    assert 0 <= r <= h - n and 0 <= c <= w - n

    while True:
        if r == h - n:
            events.append({"kind": "bottom_stop", "r": r, "c": c})
            return r, c, events

        nr = r + 1
        if _area_hits(settled, nr, c, n):
            events.append({"kind": "vertical_collision", "r": r, "c": c})
            return r, c, events

        if _area_hits(shelf_mask, nr, c, n):
            local = shelf_mask[nr : nr + n, c : c + n]
            hit_rows = np.flatnonzero(local.any(axis=1)) + nr
            assert len(hit_rows) == 1, "shelf spacing must expose one shelf at a time"
            shelf_r = int(hit_rows[0])
            assert shelf_r == r + n, "a falling square should touch a shelf with its bottom edge"
            events.append({"kind": "land_shelf", "shelf": shelf_r, "r": r, "c": c})

            candidates: list[tuple[int, int, tuple[int, int]]] = []
            for gl, gr in gaps_by_row[shelf_r]:
                if gr - gl + 1 >= n:
                    target_c = int(gl)  # align the square's left edge with the gap's left edge
                    candidates.append((abs(target_c - c), target_c, (gl, gr)))

            if not candidates:
                events.append({"kind": "no_gap_stop", "shelf": shelf_r, "r": r, "c": c})
                return r, c, events

            best_dist = min(item[0] for item in candidates)
            nearest = [item for item in candidates if item[0] == best_dist]
            assert len(nearest) == 1, "nearest passable gap must be unique"
            _dist, target_c, gap = nearest[0]
            assert 0 <= target_c <= w - n

            if target_c != c:
                step = 1 if target_c > c else -1
                while c != target_c:
                    nc = c + step
                    if _area_hits(settled, r, nc, n):
                        events.append({"kind": "horizontal_collision", "shelf": shelf_r, "r": r, "c": c})
                        return r, c, events
                    c = nc
                events.append({"kind": "slide", "shelf": shelf_r, "r": r, "c": c, "gap": gap})

            assert _fits_gap(c, n, gaps_by_row[shelf_r])
            events.append({"kind": "pass_gap", "shelf": shelf_r, "r": r, "c": c, "gap": gap})
            continue

        r = nr


def _solve_with_log(grid: np.ndarray) -> tuple[np.ndarray, list[dict]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    h, w = g.shape
    assert h <= 30 and w <= 30

    shelf_mask, gaps_by_row = _shelves_and_gaps(g)
    blocks = _find_blocks(g)

    out = g.copy()
    for b in blocks:
        r, c, n = int(b["r"]), int(b["c"]), int(b["n"])
        out[r : r + n, c : c + n] = BG

    settled = np.zeros((h, w), dtype=bool)
    logs: list[dict] = []
    for b in blocks:
        fr, fc, events = _drop_one(b, (h, w), shelf_mask, gaps_by_row, settled)
        n = int(b["n"])
        assert not _area_hits(shelf_mask, fr, fc, n), "final square cannot overwrite a shelf"
        assert not _area_hits(settled, fr, fc, n), "final squares cannot overlap"
        out[fr : fr + n, fc : fc + n] = int(b["color"])
        settled[fr : fr + n, fc : fc + n] = True
        logs.append(
            {
                "color": int(b["color"]),
                "size": n,
                "start": (int(b["r"]), int(b["c"])),
                "final": (int(fr), int(fc)),
                "events": events,
            }
        )
    return out, logs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop solid squares through the nearest shelf gap they can fit through.

    Input structure:
    - The background is black, and grey horizontal shelf rows contain bounded black gaps.
    - Above the shelves are several separated, solid, monochrome square blocks of different colors and sizes.
    - The initial bottom rows of the blocks are distinct, giving a unique bottom-to-top falling order.

    Rule:
    - Clear the blocks from their starting positions, then process them from lowest initial block to highest.
    - A block falls vertically until it touches a shelf, a previously settled block, or the grid bottom.
    - On a shelf it searches that shelf's gaps whose width is at least the block side length; if any exist,
      it slides to the unique nearest such gap, aligning its left edge to the gap's left edge, then continues falling.
    - If no passable gap exists, or if the vertical/horizontal path hits an earlier settled block, the block stops there.
    - Shelves stay fixed, and each block is repainted at its final resting position.

    Key insight:
    The visible gap widths and the falling order both matter: each square repeatedly seeks only holes it can fit through,
    while earlier resting squares become new obstacles for later squares.
    """
    out, _logs = _solve_with_log(grid)
    return out


def _sample_gap_specs(width: int, gap_widths: list[int]) -> list[tuple[int, int]] | None:
    for _ in range(200):
        gaps: list[tuple[int, int]] = []
        for gw in sorted(gap_widths, reverse=True):
            choices = list(range(1, width - gw))
            random.shuffle(choices)
            placed = False
            for start in choices:
                end = start + gw - 1
                if all(end < a - 1 or start > b + 1 for a, b in gaps):
                    gaps.append((start, end))
                    placed = True
                    break
            if not placed:
                break
        else:
            return sorted(gaps)
    return None


def _draw_shelves(
    shape: tuple[int, int],
    rows: list[int],
    gap_specs: dict[int, list[tuple[int, int]]],
) -> np.ndarray:
    h, w = shape
    g = np.full((h, w), BG, dtype=int)
    for r in rows:
        g[r, :] = SHELF
        for a, b in gap_specs[r]:
            g[r, a : b + 1] = BG
    return g


def _overlaps_existing(blocks: list[dict], r: int, c: int, n: int, pad: int = 0) -> bool:
    r0, c0, r1, c1 = r - pad, c - pad, r + n - 1 + pad, c + n - 1 + pad
    for b in blocks:
        br0, bc0 = int(b["r"]), int(b["c"])
        br1, bc1 = br0 + int(b["n"]) - 1, bc0 + int(b["n"]) - 1
        if not (r1 < br0 or br1 < r0 or c1 < bc0 or bc1 < c0):
            return True
    return False


def _add_block(
    blocks: list[dict],
    color: int,
    n: int,
    r_choices: list[int],
    c_choices: list[int],
    first_shelf: int,
    pad: int = 1,
) -> bool:
    random.shuffle(r_choices)
    random.shuffle(c_choices)
    for r in r_choices:
        if r < 0 or r + n > first_shelf:
            continue
        for c in c_choices:
            if c < 0:
                continue
            if _overlaps_existing(blocks, r, c, n, pad=pad):
                continue
            blocks.append({"color": int(color), "n": int(n), "r": int(r), "c": int(c)})
            return True
    return False


def _choose_pass_start(n: int, gaps: list[tuple[int, int]], width: int, avoid: tuple[int, int]) -> list[int]:
    starts: list[int] = []
    avoid_a, avoid_b = avoid
    for gl, gr in gaps:
        if gr - gl + 1 < n:
            continue
        target = gl
        for delta in (-7, -6, -5, -4, 4, 5, 6, 7):
            c = target + delta
            if 0 <= c <= width - n and not _fits_gap(c, n, gaps):
                lo = min(c, target)
                hi = max(c + n - 1, target + n - 1)
                if hi < avoid_a or lo > avoid_b:
                    starts.append(c)
    random.shuffle(starts)
    return starts


def _gen_attempt() -> np.ndarray | None:
    h = random.randint(23, 30)
    w = random.randint(19, 30)
    first_shelf = random.randint(9, min(11, h - 13))
    shelf_count = random.choice([3, 4])
    rows = [first_shelf]
    for _ in range(shelf_count - 1):
        rows.append(rows[-1] + random.randint(4, 6))
    if rows[-1] > h - 4:
        return None

    gap_specs: dict[int, list[tuple[int, int]]] = {}
    top_widths = [1, 2, random.choice([1, 2])]
    top_gaps = _sample_gap_specs(w, top_widths)
    if top_gaps is None:
        return None
    gap_specs[rows[0]] = top_gaps

    for r in rows[1:]:
        widths = [1, 2, 3]
        if random.random() < 0.55:
            widths.append(random.choice([1, 2, 3, 4]))
        gaps = _sample_gap_specs(w, widths)
        if gaps is None:
            return None
        gap_specs[r] = gaps

    palette = list(BLOCK_COLORS)
    random.shuffle(palette)
    blocks: list[dict] = []

    big_n = 3
    big_r = first_shelf - big_n - random.randint(2, 3)
    if big_r < 1:
        return None
    big_c_choices = list(range(2, w - big_n - 1))
    random.shuffle(big_c_choices)
    if not big_c_choices:
        return None
    big_c = big_c_choices[0]
    blocks.append({"color": palette.pop(), "n": big_n, "r": big_r, "c": big_c})

    collide_n = random.choice([1, 2])
    collide_c = big_c + random.randint(0, big_n - collide_n)
    collide_rows = list(range(0, max(0, big_r - collide_n)))
    if not collide_rows:
        return None
    collide_r = random.choice(collide_rows)
    if _overlaps_existing(blocks, collide_r, collide_c, collide_n, pad=0):
        return None
    blocks.append({"color": palette.pop(), "n": collide_n, "r": collide_r, "c": collide_c})

    pass_count = random.choice([2, 3])
    for i in range(pass_count):
        n = random.choice([1, 2])
        starts = _choose_pass_start(n, top_gaps, w, (big_c, big_c + big_n - 1))
        if not starts:
            starts = list(range(0, w - n + 1))
        row_choices = list(range(0, first_shelf - n))
        if not _add_block(blocks, palette.pop(), n, row_choices, starts, first_shelf, pad=1):
            row_choices = list(range(0, first_shelf - n))
            col_choices = [c for c in range(0, w - n + 1) if c < big_c - n - 1 or c > big_c + big_n]
            if not _add_block(blocks, palette.pop(), n, row_choices, col_choices, first_shelf, pad=1):
                return None

    bottoms = [int(b["r"]) + int(b["n"]) - 1 for b in blocks]
    if len(bottoms) != len(set(bottoms)):
        return None

    g = _draw_shelves((h, w), rows, gap_specs)
    for b in blocks:
        r, c, n = int(b["r"]), int(b["c"]), int(b["n"])
        if np.any(g[r : r + n, c : c + n] != BG):
            return None
        g[r : r + n, c : c + n] = int(b["color"])

    try:
        out, logs = _solve_with_log(g)
    except AssertionError:
        return None

    event_kinds = [event["kind"] for log in logs for event in log["events"]]
    pass_counts = [sum(event["kind"] == "pass_gap" for event in log["events"]) for log in logs]
    moved = sum(log["start"] != log["final"] for log in logs)

    if "no_gap_stop" not in event_kinds:
        return None
    if not any(kind in event_kinds for kind in ("vertical_collision", "horizontal_collision")):
        return None
    if "slide" not in event_kinds or "pass_gap" not in event_kinds:
        return None
    if max(pass_counts, default=0) < 2:
        return None
    if moved < 3 or np.array_equal(g, out):
        return None

    return g


def generate_input() -> np.ndarray:
    for _ in range(600):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid fable/2/040/4 input")
