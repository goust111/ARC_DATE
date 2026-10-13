from __future__ import annotations

import random
from collections import Counter, defaultdict

import numpy as np


BACKGROUND = 0
GEN_WALL_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)

ROOM_COUNT_OPTIONS = (1, 2, 3)
ROOM_COUNT_WEIGHTS = (1, 4, 3)
ROOM_SIZE_RANGE = (5, 8)
ROOM_COLOR_OPTIONS = (0, 1, 2, 3)
ROOM_COLOR_WEIGHTS_WITH_GAPS = (2, 4, 2, 1)
ROOM_COLOR_WEIGHTS_NO_GAPS = (4, 3, 1, 0)
MAX_GENERATION_ATTEMPTS = 500


def _separator_lines_for_color(
    grid: np.ndarray,
    color: int,
) -> tuple[dict[int, int], dict[int, int]]:
    h, w = grid.shape
    row_gaps: dict[int, int] = {}
    col_gaps: dict[int, int] = {}

    for r in range(h):
        row = grid[r]
        if not bool(np.all((row == BACKGROUND) | (row == int(color)))):
            continue
        if int(np.count_nonzero(row == int(color))) != w - 1:
            continue
        gap_cols = np.flatnonzero(row == BACKGROUND)
        if int(gap_cols.size) == 1:
            row_gaps[int(r)] = int(gap_cols[0])

    for c in range(w):
        col = grid[:, c]
        if not bool(np.all((col == BACKGROUND) | (col == int(color)))):
            continue
        if int(np.count_nonzero(col == int(color))) != h - 1:
            continue
        gap_rows = np.flatnonzero(col == BACKGROUND)
        if int(gap_rows.size) == 1:
            col_gaps[int(c)] = int(gap_rows[0])

    return row_gaps, col_gaps


def _find_wall_color(grid: np.ndarray) -> tuple[int, dict[int, int], dict[int, int]]:
    best_color = None
    best_rows: dict[int, int] = {}
    best_cols: dict[int, int] = {}
    best_score = None

    for color in sorted(int(c) for c in np.unique(grid) if int(c) != BACKGROUND):
        row_gaps, col_gaps = _separator_lines_for_color(grid, color)
        if not row_gaps and not col_gaps:
            continue
        score = (
            int(len(row_gaps) + len(col_gaps)),
            int(len(row_gaps) * grid.shape[1] + len(col_gaps) * grid.shape[0]),
        )
        if best_score is None or score > best_score:
            best_color = int(color)
            best_rows = row_gaps
            best_cols = col_gaps
            best_score = score

    assert best_color is not None, "expected a separator color forming full-width/full-height lines"
    return int(best_color), best_rows, best_cols


def _spans_from_lines(size: int, separators: dict[int, int]) -> list[tuple[int, int]]:
    bounds = [-1] + sorted(int(x) for x in separators) + [int(size)]
    spans: list[tuple[int, int]] = []
    for left, right in zip(bounds, bounds[1:]):
        start = int(left + 1)
        end = int(right - 1)
        assert start <= end, "separator lines must leave non-empty rooms between them"
        spans.append((start, end))
    return spans


def _room_gaps(
    row_spans: list[tuple[int, int]],
    col_spans: list[tuple[int, int]],
    row_gaps: dict[int, int],
    col_gaps: dict[int, int],
    row_idx: int,
    col_idx: int,
) -> list[dict]:
    r0, r1 = row_spans[row_idx]
    c0, c1 = col_spans[col_idx]
    gaps: list[dict] = []

    if row_idx > 0:
        sep_r = int(r0 - 1)
        gap_c = row_gaps.get(sep_r)
        if gap_c is not None and c0 <= int(gap_c) <= c1:
            gaps.append(
                {
                    "gap": (sep_r, int(gap_c)),
                    "dir": (-1, 0),
                    "neighbor": (row_idx - 1, col_idx),
                }
            )
    if row_idx + 1 < len(row_spans):
        sep_r = int(r1 + 1)
        gap_c = row_gaps.get(sep_r)
        if gap_c is not None and c0 <= int(gap_c) <= c1:
            gaps.append(
                {
                    "gap": (sep_r, int(gap_c)),
                    "dir": (1, 0),
                    "neighbor": (row_idx + 1, col_idx),
                }
            )
    if col_idx > 0:
        sep_c = int(c0 - 1)
        gap_r = col_gaps.get(sep_c)
        if gap_r is not None and r0 <= int(gap_r) <= r1:
            gaps.append(
                {
                    "gap": (int(gap_r), sep_c),
                    "dir": (0, -1),
                    "neighbor": (row_idx, col_idx - 1),
                }
            )
    if col_idx + 1 < len(col_spans):
        sep_c = int(c1 + 1)
        gap_r = col_gaps.get(sep_c)
        if gap_r is not None and r0 <= int(gap_r) <= r1:
            gaps.append(
                {
                    "gap": (int(gap_r), sep_c),
                    "dir": (0, 1),
                    "neighbor": (row_idx, col_idx + 1),
                }
            )

    return gaps


def _analyze_structure(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape

    wall_color, row_gaps, col_gaps = _find_wall_color(g)
    row_spans = _spans_from_lines(h, row_gaps)
    col_spans = _spans_from_lines(w, col_gaps)

    rooms: list[dict] = []
    room_lookup: dict[tuple[int, int], dict] = {}
    for row_idx, (r0, r1) in enumerate(row_spans):
        for col_idx, (c0, c1) in enumerate(col_spans):
            room = {
                "index": (int(row_idx), int(col_idx)),
                "bounds": (int(r0), int(c0), int(r1), int(c1)),
                "gaps": _room_gaps(row_spans, col_spans, row_gaps, col_gaps, row_idx, col_idx),
            }
            rooms.append(room)
            room_lookup[(int(row_idx), int(col_idx))] = room

    return {
        "wall_color": int(wall_color),
        "row_gaps": row_gaps,
        "col_gaps": col_gaps,
        "row_spans": row_spans,
        "col_spans": col_spans,
        "rooms": rooms,
        "room_lookup": room_lookup,
    }


def _room_color_counts(grid: np.ndarray, room: dict, wall_color: int) -> Counter:
    r0, c0, r1, c1 = room["bounds"]
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    counts = Counter(
        int(value)
        for value in patch.flat
        if int(value) not in (BACKGROUND, int(wall_color))
    )
    return counts


def _collect_sources(grid: np.ndarray, structure: dict) -> list[dict]:
    wall_color = int(structure["wall_color"])
    sources: list[dict] = []
    for room in structure["rooms"]:
        counts = _room_color_counts(grid, room, wall_color)
        if not counts or not room["gaps"]:
            continue
        for color, length in counts.items():
            for gap_info in room["gaps"]:
                gap_r, gap_c = gap_info["gap"]
                dr, dc = gap_info["dir"]
                sources.append(
                    {
                        "room": room["index"],
                        "target_room": gap_info["neighbor"],
                        "color": int(color),
                        "length": int(length),
                        "start": (int(gap_r + dr), int(gap_c + dc)),
                        "dir": (int(dr), int(dc)),
                        "active": True,
                    }
                )
    return sources


def main(grid: np.ndarray) -> np.ndarray:
    """
    Grow each room's seed colors through separator gaps into neighboring rooms by count-defined lengths.

    Input structure:
    - The background is black.
    - One separator color forms an orthogonal network of full-width and/or full-height lines that partitions the grid into rectangular rooms.
    - Every separator line has exactly one one-cell background gap, and rooms may contain sparse colored seed pixels away from the separators.

    Rule:
    - For each room, count how many pixels of each seed color appear inside that room.
    - For every color count N and for every gap on that room's boundary, cast a straight ray through the gap into the adjacent room, starting one cell past the gap and extending up to N cells.
    - Growth keeps the original walls and seed pixels unchanged; a ray can paint background cells and can pass through same-colored cells, but it stops before any different-colored pixel.

    Key insight:
    - The visible count of a color inside one room controls how far that color appears in other rooms, because each room exports rays through its own boundary gaps instead of changing internally.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    structure = _analyze_structure(g)
    sources = _collect_sources(g, structure)
    out = g.copy()
    h, w = g.shape

    max_length = max((int(source["length"]) for source in sources), default=0)
    for step in range(max_length):
        proposals: dict[tuple[int, int], list[int]] = defaultdict(list)
        for idx, source in enumerate(sources):
            if not bool(source["active"]) or step >= int(source["length"]):
                continue

            dr, dc = source["dir"]
            sr, sc = source["start"]
            r = int(sr + step * dr)
            c = int(sc + step * dc)
            if not (0 <= r < h and 0 <= c < w):
                source["active"] = False
                continue

            cell = int(out[r, c])
            color = int(source["color"])
            if cell not in (BACKGROUND, color):
                source["active"] = False
                continue
            proposals[(r, c)].append(idx)

        blocked_sources: set[int] = set()
        for _, indices in proposals.items():
            colors = {int(sources[idx]["color"]) for idx in indices}
            if len(colors) > 1:
                blocked_sources.update(int(idx) for idx in indices)

        for idx in blocked_sources:
            sources[idx]["active"] = False

        for (r, c), indices in proposals.items():
            valid = [idx for idx in indices if idx not in blocked_sources]
            if not valid:
                continue
            out[r, c] = int(sources[valid[0]]["color"])

    return out


def _layout_spans_from_sizes(sizes: list[int]) -> tuple[list[tuple[int, int]], list[int], int]:
    spans: list[tuple[int, int]] = []
    separators: list[int] = []
    cursor = 0
    for idx, size in enumerate(sizes):
        start = int(cursor)
        end = int(cursor + size - 1)
        spans.append((start, end))
        cursor = end + 1
        if idx + 1 < len(sizes):
            separators.append(int(cursor))
            cursor += 1
    return spans, separators, int(cursor)


def _build_base_layout() -> tuple[np.ndarray, dict] | None:
    row_count = int(random.choices(ROOM_COUNT_OPTIONS, weights=ROOM_COUNT_WEIGHTS)[0])
    col_count = int(random.choices(ROOM_COUNT_OPTIONS, weights=ROOM_COUNT_WEIGHTS)[0])
    if row_count == 1 and col_count == 1:
        return None

    row_sizes = [random.randint(*ROOM_SIZE_RANGE) for _ in range(row_count)]
    col_sizes = [random.randint(*ROOM_SIZE_RANGE) for _ in range(col_count)]
    row_spans, row_separators, height = _layout_spans_from_sizes(row_sizes)
    col_spans, col_separators, width = _layout_spans_from_sizes(col_sizes)
    if height > 30 or width > 30:
        return None

    grid = np.full((height, width), BACKGROUND, dtype=int)
    for r in row_separators:
        grid[r, :] = GEN_WALL_COLOR
    for c in col_separators:
        grid[:, c] = GEN_WALL_COLOR

    free_gap_cols = [c for c in range(width) if c not in set(col_separators)]
    free_gap_rows = [r for r in range(height) if r not in set(row_separators)]
    if (row_separators and not free_gap_cols) or (col_separators and not free_gap_rows):
        return None

    for r in row_separators:
        grid[r, random.choice(free_gap_cols)] = BACKGROUND
    for c in col_separators:
        grid[random.choice(free_gap_rows), c] = BACKGROUND

    structure = _analyze_structure(grid)
    if len(structure["row_spans"]) != len(row_spans) or len(structure["col_spans"]) != len(col_spans):
        return None
    return grid, structure


def _room_seed_candidates(grid: np.ndarray, room: dict) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = room["bounds"]
    candidates: list[tuple[int, int]] = []
    for r in range(int(r0 + 1), int(r1)):
        for c in range(int(c0 + 1), int(c1)):
            if int(grid[r, c]) == BACKGROUND:
                candidates.append((int(r), int(c)))
    return candidates


def _sample_counts(num_colors: int, capacity: int) -> list[int] | None:
    for _ in range(60):
        counts = [random.randint(1, 5) for _ in range(num_colors)]
        if sum(counts) <= int(capacity):
            return counts
    return None


def _place_room_seeds(grid: np.ndarray, room: dict) -> None:
    candidates = _room_seed_candidates(grid, room)
    if not candidates:
        return

    has_gap = bool(room["gaps"])
    weights = ROOM_COLOR_WEIGHTS_WITH_GAPS if has_gap else ROOM_COLOR_WEIGHTS_NO_GAPS
    color_count = int(random.choices(ROOM_COLOR_OPTIONS, weights=weights)[0])
    color_count = min(color_count, len(SEED_COLORS), len(candidates))
    if color_count == 0:
        return

    counts = _sample_counts(color_count, len(candidates))
    if counts is None:
        return

    chosen_colors = random.sample(SEED_COLORS, color_count)
    random.shuffle(candidates)
    offset = 0
    for color, count in zip(chosen_colors, counts):
        cells = candidates[offset : offset + count]
        offset += count
        for r, c in cells:
            grid[r, c] = int(color)


def _ensure_some_seeds(grid: np.ndarray, structure: dict) -> bool:
    if bool(np.any((grid != BACKGROUND) & (grid != GEN_WALL_COLOR))):
        return True

    rooms = [room for room in structure["rooms"] if _room_seed_candidates(grid, room)]
    if not rooms:
        return False
    room = random.choice(rooms)
    cells = _room_seed_candidates(grid, room)
    count = random.randint(1, min(3, len(cells)))
    for r, c in random.sample(cells, count):
        grid[r, c] = int(random.choice(SEED_COLORS))
    return True


def _force_multicolor_room(grid: np.ndarray, structure: dict) -> bool:
    enriched_rooms: list[tuple[dict, Counter]] = []
    for room in structure["rooms"]:
        counts = _room_color_counts(grid, room, GEN_WALL_COLOR)
        if len(counts) >= 2:
            return True
        if not counts:
            continue
        if _room_seed_candidates(grid, room):
            enriched_rooms.append((room, counts))

    if enriched_rooms:
        room, counts = random.choice(enriched_rooms)
        extra_colors = [color for color in SEED_COLORS if color not in counts]
        if not extra_colors:
            return False
        r, c = random.choice(_room_seed_candidates(grid, room))
        grid[r, c] = int(random.choice(extra_colors))
        return True

    empty_rooms = [room for room in structure["rooms"] if len(_room_seed_candidates(grid, room)) >= 2]
    if not empty_rooms:
        return False
    room = random.choice(empty_rooms)
    cells = random.sample(_room_seed_candidates(grid, room), 2)
    colors = random.sample(SEED_COLORS, 2)
    for (r, c), color in zip(cells, colors):
        grid[r, c] = int(color)
    return True


def _force_blocker(grid: np.ndarray) -> bool:
    structure = _analyze_structure(grid)
    sources = _collect_sources(grid, structure)
    random.shuffle(sources)
    for source in sources:
        if int(source["length"]) < 2:
            continue
        target_room = structure["room_lookup"][tuple(source["target_room"])]
        r0, c0, r1, c1 = target_room["bounds"]
        dr, dc = source["dir"]
        sr, sc = source["start"]
        for step in range(2, int(source["length"]) + 1):
            r = int(sr + (step - 1) * dr)
            c = int(sc + (step - 1) * dc)
            if not (r0 + 1 <= r <= r1 - 1 and c0 + 1 <= c <= c1 - 1):
                continue
            if int(grid[r, c]) != BACKGROUND:
                continue
            blocker_colors = [color for color in SEED_COLORS if int(color) != int(source["color"])]
            grid[r, c] = int(random.choice(blocker_colors))
            return True
    return False


def _has_multicolor_room(grid: np.ndarray, structure: dict) -> bool:
    return any(len(_room_color_counts(grid, room, GEN_WALL_COLOR)) >= 2 for room in structure["rooms"])


def _growth_cell_count(inp: np.ndarray, out: np.ndarray) -> int:
    return int(np.count_nonzero(inp != out))


def _gen_attempt() -> np.ndarray | None:
    built = _build_base_layout()
    if built is None:
        return None
    grid, structure = built

    for room in structure["rooms"]:
        _place_room_seeds(grid, room)
    if not _ensure_some_seeds(grid, structure):
        return None

    want_multicolor = bool(random.getrandbits(1))
    want_blocker = bool(random.getrandbits(1))

    structure = _analyze_structure(grid)
    if want_multicolor and not _has_multicolor_room(grid, structure):
        _force_multicolor_room(grid, structure)

    if want_blocker:
        _force_blocker(grid)

    structure = _analyze_structure(grid)
    try:
        out = main(grid)
    except Exception:
        return None

    if out.shape != grid.shape:
        return None
    if not np.any(out != grid):
        return None

    seed_pixels = int(np.count_nonzero((grid != BACKGROUND) & (grid != GEN_WALL_COLOR)))
    grown_pixels = _growth_cell_count(grid, out)
    if seed_pixels < 2:
        return None
    if grown_pixels < 3:
        return None
    if grown_pixels > max(18, grid.size // 3):
        return None

    if want_multicolor and not _has_multicolor_room(grid, structure):
        return None

    if not any(room["gaps"] for room in structure["rooms"]):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid room-gap growth puzzle")
