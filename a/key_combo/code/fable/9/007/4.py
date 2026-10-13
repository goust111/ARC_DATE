from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
COLOR_POOL = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))
MAX_GENERATION_ATTEMPTS = 500


def _components(mask: np.ndarray) -> tuple[list[np.ndarray], np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    labels = np.full((h, w), -1, dtype=int)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            cid = len(comps)
            while q:
                rr, cc = q.popleft()
                labels[rr, cc] = cid
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps, labels


def _unique_majority(counts: Counter[int]) -> int:
    assert counts, "expected at least one vote"
    best = max(counts.values())
    winners = [color for color, count in counts.items() if count == best]
    assert len(winners) == 1, f"expected a unique majority, got {counts}"
    return int(winners[0])


def _infer_bg_wall(grid: np.ndarray) -> tuple[int, int]:
    flat = Counter(map(int, grid.ravel()))
    bg = int(flat.most_common(1)[0][0])
    inner = grid[1:-1, 1:-1] if min(grid.shape) > 2 else grid
    wall_counts = Counter(int(v) for v in inner.ravel() if int(v) != bg)
    assert wall_counts, "expected a visible wall color"
    best = max(wall_counts.values())
    winners = [color for color, count in wall_counts.items() if count == best]
    assert len(winners) == 1, f"expected a unique wall color, got {wall_counts}"
    return bg, int(winners[0])


def _marker_direction(h: int, w: int, r: int, c: int) -> tuple[int, int]:
    on_top = r == 0
    on_bottom = r == h - 1
    on_left = c == 0
    on_right = c == w - 1
    assert on_top or on_bottom or on_left or on_right, "marker must be on the outer ring"
    assert (on_top + on_bottom + on_left + on_right) == 1, "corner markers are ambiguous"
    if on_top:
        return 1, 0
    if on_bottom:
        return -1, 0
    if on_left:
        return 0, 1
    return 0, -1


def _solve_data(grid: np.ndarray) -> tuple[int, int, list[np.ndarray], np.ndarray, dict[int, Counter[int]], dict[tuple[int, int], bool]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"
    h, w = g.shape
    bg, wall = _infer_bg_wall(g)

    chamber_mask = g != wall
    chambers, chamber_id = _components(chamber_mask)
    assert chambers, "expected at least one non-wall chamber"

    marker_colors = set(map(int, g.ravel())) - {bg, wall}
    votes: dict[int, Counter[int]] = {cid: Counter() for cid in range(len(chambers))}
    blocked: dict[tuple[int, int], bool] = {}

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color not in marker_colors:
                continue
            if not (r == 0 or r == h - 1 or c == 0 or c == w - 1):
                continue
            dr, dc = _marker_direction(h, w, r, c)
            nr, nc = r + dr, c + dc
            assert 0 <= nr < h and 0 <= nc < w
            if int(g[nr, nc]) == wall:
                blocked[(r, c)] = True
                continue
            cid = int(chamber_id[nr, nc])
            assert cid >= 0, "ray must enter a chamber"
            rr, cc = nr, nc
            while 0 <= rr < h and 0 <= cc < w and int(g[rr, cc]) != wall:
                assert int(chamber_id[rr, cc]) == cid, "a ray should stay within one chamber"
                rr += dr
                cc += dc
            votes[cid][color] += 1
            blocked[(r, c)] = False

    return bg, wall, chambers, chamber_id, votes, blocked


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each wall-separated chamber with the majority color of the edge markers that can see into it.

    Input structure:
    - A dominant background is cut into 4-connected chambers by a single wall color.
    - Colored single-pixel markers lie only on the outer ring, and chamber connectivity treats those marker cells as background.
    - Some markers sit above or beside wall roots, so their inward ray is blocked immediately.

    Rule:
    - Each marker fires straight inward, perpendicular to the boundary edge it sits on.
    - If the very first inward cell is a wall, that marker casts no vote; otherwise its ray stays in the first chamber it enters until it hits a wall or leaves the opposite side, and that chamber gets one vote of the marker's color.
    - Recolor every chamber uniformly by the unique majority among its received votes; chambers with no votes stay background, walls stay unchanged, and marker pixels are erased into their chamber's final fill.

    Key insight:
    - Chamber ownership is decided by straight line-of-sight through non-wall cells, not by adjacency to the marker cell; blocked boundary markers are deliberate false clues that belong to a chamber visually but contribute zero votes.
    """
    bg, wall, chambers, _chamber_id, votes, _blocked = _solve_data(grid)
    out = np.full_like(np.asarray(grid, dtype=int), bg)
    out[np.asarray(grid, dtype=int) == wall] = wall
    for cid, coords in enumerate(chambers):
        fill = bg if not votes[cid] else _unique_majority(votes[cid])
        out[coords[:, 0], coords[:, 1]] = fill
    return out


def _sample_segment_widths(n_segments: int) -> list[int]:
    widths = [random.randint(2, 5) for _ in range(n_segments)]
    if widths[0] < 3:
        widths[0] += 1
    if widths[-1] < 3:
        widths[-1] += 1
    return widths


def _build_layout() -> tuple[np.ndarray, dict[str, object]]:
    top_h = random.randint(3, 5)
    mid_h = random.randint(4, 7)
    bottom_h = random.randint(3, 5)
    n_middle = random.randint(3, 5)
    widths = _sample_segment_widths(n_middle)

    h = top_h + 1 + mid_h + 1 + bottom_h
    w = sum(widths) + (n_middle - 1)
    assert h <= 30 and w <= 30

    top_wall = top_h
    bottom_wall = top_h + 1 + mid_h

    g = np.full((h, w), BG, dtype=int)
    g[top_wall, :] = WALL
    g[bottom_wall, :] = WALL

    part_cols: list[int] = []
    col_ranges: list[tuple[int, int]] = []
    cur = 0
    for idx, width in enumerate(widths):
        col_ranges.append((cur, cur + width - 1))
        cur += width
        if idx != len(widths) - 1:
            part_cols.append(cur)
            g[top_wall : bottom_wall + 1, cur] = WALL
            cur += 1

    stub_candidates = [c for c in range(1, w - 1) if c not in part_cols]
    random.shuffle(stub_candidates)
    top_stub_count = random.randint(3, min(5, len(stub_candidates) - 1))
    top_stubs = sorted(stub_candidates[:top_stub_count])
    for c in top_stubs:
        g[1 : top_wall + 1, c] = WALL

    bottom_stubs: list[int] = []
    if random.random() < 0.65:
        remaining = [c for c in stub_candidates if c not in top_stubs]
        if len(remaining) >= 2:
            random.shuffle(remaining)
            bottom_stub_count = random.randint(2, min(4, len(remaining)))
            bottom_stubs = sorted(remaining[:bottom_stub_count])
            for c in bottom_stubs:
                g[bottom_wall : h - 1, c] = WALL

    chambers, labels = _components(g != WALL)
    assert len(chambers) >= n_middle + 2

    meta = {
        "top_wall": top_wall,
        "bottom_wall": bottom_wall,
        "part_cols": part_cols,
        "top_stubs": top_stubs,
        "bottom_stubs": bottom_stubs,
        "col_ranges": col_ranges,
        "labels": labels,
        "n_middle": n_middle,
    }
    return g, meta


def _pick_fill_colors() -> tuple[int, int, int, int, int]:
    palette = random.sample(COLOR_POOL, 5)
    return palette[0], palette[1], palette[2], palette[3], palette[4]


def _sample_positions(positions: list[tuple[int, int]], count: int) -> list[tuple[int, int]]:
    assert count <= len(positions), (count, len(positions))
    return random.sample(positions, count)


def _place_markers(
    grid: np.ndarray,
    meta: dict[str, object],
) -> tuple[np.ndarray, dict[int, Counter[int]], dict[str, object]]:
    g = grid.copy()
    h, w = g.shape
    top_wall = int(meta["top_wall"])
    bottom_wall = int(meta["bottom_wall"])
    labels = np.asarray(meta["labels"], dtype=int)
    top_stubs = set(meta["top_stubs"])
    bottom_stubs = set(meta["bottom_stubs"])
    col_ranges = list(meta["col_ranges"])

    top_id = int(labels[0, 1])
    bottom_id = int(labels[h - 1, 1])
    left_id = int(labels[top_wall + 1, 0])
    right_id = int(labels[top_wall + 1, w - 1])

    top_open = [(0, c) for c in range(1, w - 1) if c not in top_stubs]
    top_blocked = [(0, c) for c in sorted(top_stubs)]
    bottom_open = [(h - 1, c) for c in range(1, w - 1) if c not in bottom_stubs]
    bottom_blocked = [(h - 1, c) for c in sorted(bottom_stubs)]
    left_open = [(r, 0) for r in range(top_wall + 1, bottom_wall)]
    right_open = [(r, w - 1) for r in range(top_wall + 1, bottom_wall)]

    top_fill, top_minor, top_block_decoy, bottom_fill, side_fill = _pick_fill_colors()
    right_fill = side_fill if random.random() < 0.35 else random.choice([c for c in COLOR_POOL if c not in {top_fill, top_minor, top_block_decoy, bottom_fill}])
    left_fill = random.choice([c for c in COLOR_POOL if c != right_fill])

    planned: dict[int, Counter[int]] = {}
    markers: dict[tuple[int, int], int] = {}

    def add_votes(cid: int, coords: list[tuple[int, int]], colors: list[int]) -> None:
        for pos, color in zip(coords, colors):
            markers[pos] = int(color)
            planned.setdefault(cid, Counter())[int(color)] += 1

    assert len(top_open) >= 3, "top chamber needs enough open markers for a strict majority"
    top_majority = 2
    top_minor_count = 1
    top_coords = _sample_positions(top_open, top_majority + top_minor_count)
    top_colors = [top_fill] * top_majority + [top_minor]
    random.shuffle(top_colors)
    add_votes(top_id, top_coords, top_colors)

    blocked_needed = min(len(top_blocked), max(3, top_majority + 1))
    blocked_coords = _sample_positions(top_blocked, blocked_needed)
    for pos in blocked_coords:
        markers[pos] = top_block_decoy

    bottom_voted = random.random() < 0.85
    if bottom_voted:
        bottom_count = random.randint(1, min(3, len(bottom_open)))
        bottom_coords = _sample_positions(bottom_open, bottom_count)
        bottom_colors = [bottom_fill] * bottom_count
        if bottom_count >= 3 and random.random() < 0.5:
            bottom_colors[-1] = random.choice([c for c in COLOR_POOL if c != bottom_fill])
        add_votes(bottom_id, bottom_coords, bottom_colors)

    left_count = random.randint(1, min(3, len(left_open)))
    left_coords = _sample_positions(left_open, left_count)
    left_colors = [left_fill] * left_count
    if left_count >= 3 and random.random() < 0.4:
        left_colors[-1] = random.choice([c for c in COLOR_POOL if c != left_fill])
    add_votes(left_id, left_coords, left_colors)

    right_count = random.randint(1, min(3, len(right_open)))
    right_coords = _sample_positions(right_open, right_count)
    right_colors = [right_fill] * right_count
    if right_count >= 3 and random.random() < 0.4:
        right_colors[-1] = random.choice([c for c in COLOR_POOL if c != right_fill])
    add_votes(right_id, right_coords, right_colors)

    if bottom_blocked and random.random() < 0.6:
        decoy_color = random.choice([c for c in COLOR_POOL if c != bottom_fill])
        for pos in _sample_positions(bottom_blocked, random.randint(1, len(bottom_blocked))):
            markers[pos] = decoy_color

    for pos, color in markers.items():
        g[pos] = int(color)

    info = {
        "planned": planned,
        "markers": markers,
        "top_id": top_id,
        "bottom_id": bottom_id,
        "left_id": left_id,
        "right_id": right_id,
        "interior_ids": sorted(set(range(int(labels.max()) + 1)) - {top_id, bottom_id, left_id, right_id}),
        "top_fill": top_fill,
        "top_block_decoy": top_block_decoy,
    }
    return g, planned, info


def _membership_marker_counts(labels: np.ndarray, markers: dict[tuple[int, int], int]) -> dict[int, Counter[int]]:
    counts: dict[int, Counter[int]] = {}
    for (r, c), color in markers.items():
        cid = int(labels[r, c])
        counts.setdefault(cid, Counter())[int(color)] += 1
    return counts


def _validate_generated(grid: np.ndarray, planned: dict[int, Counter[int]], info: dict[str, object]) -> bool:
    labels = np.asarray(info["labels"], dtype=int) if "labels" in info else None
    if labels is None:
        return False
    actual = main(grid)
    solved_bg, solved_wall, chambers, chamber_id, votes, blocked = _solve_data(grid)
    if solved_bg != BG or solved_wall != WALL:
        return False

    if not any(blocked.values()) or not any(not flag for flag in blocked.values()):
        return False

    interior_ids = list(info["interior_ids"])
    if not interior_ids:
        return False
    if not any(not votes[cid] for cid in interior_ids):
        return False

    top_id = int(info["top_id"])
    top_votes = planned.get(top_id, Counter())
    if len(top_votes) < 2:
        return False

    chamber_marker_counts = _membership_marker_counts(chamber_id, info["markers"])
    if _unique_majority(chamber_marker_counts[top_id]) == _unique_majority(top_votes):
        return False

    for cid, coords in enumerate(chambers):
        expected_fill = BG if not planned.get(cid) else _unique_majority(planned[cid])
        if not np.all(actual[coords[:, 0], coords[:, 1]] == expected_fill):
            return False

    if not np.array_equal(actual[grid == WALL], np.full(int((grid == WALL).sum()), WALL, dtype=int)):
        return False

    return True


def _gen_attempt() -> np.ndarray | None:
    base, meta = _build_layout()
    placed, planned, info = _place_markers(base, meta)
    info["labels"] = meta["labels"]
    if _validate_generated(placed, planned, info):
        return placed
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"generate_input failed after {MAX_GENERATION_ATTEMPTS} attempts")
