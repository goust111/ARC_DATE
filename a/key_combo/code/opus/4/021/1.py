from __future__ import annotations

from collections import Counter, deque
from pathlib import Path
import random
import sys

import numpy as np

_UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(_UTILS_DIR) not in sys.path:
    sys.path.append(str(_UTILS_DIR))

from arc_utils import Color, find_components, in_bounds


TOKEN_COLORS = (Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW)
GEN_TRIES = 500


def _frame_border_mask(height: int, width: int, gap_col: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    mask[0, gap_col] = False
    return mask


def _detect_frames(grid: np.ndarray) -> list[dict]:
    comps = find_components(
        grid,
        bg=Color.BLACK,
        colors=[Color.GREY],
        connectivity=4,
        monochromatic=True,
        return_format="dict",
    )
    frames: list[dict] = []
    for comp in comps:
        r0, c0, r1, c1 = comp["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        grey = sub == Color.GREY
        if np.any(grey[1:-1, 1:-1]):
            continue
        if not np.all(grey[:, 0]) or not np.all(grey[:, -1]) or not np.all(grey[-1, :]):
            continue
        if not grey[0, 0] or not grey[0, -1]:
            continue
        gap_cols = [cc for cc in range(1, w - 1) if not grey[0, cc]]
        if not gap_cols:
            continue
        if any(grey[0, cc] == grey[0, cc - 1] == False for cc in range(2, w - 1)):
            # Multi-cell openings are allowed by the puzzle family, but this task's
            # generator uses a single-cell opening to keep the landing unique.
            continue
        if any(grey[0, cc] for cc in gap_cols):
            continue
        if any(int(sub[1, cc]) != Color.BLACK for cc in gap_cols):
            continue
        frames.append(
            {
                "bbox": (int(r0), int(c0), int(r1), int(c1)),
                "coords": comp["coords"],
                "gap_cols": [int(c0 + cc) for cc in gap_cols],
                "landing_cells": [(int(r0 + 1), int(c0 + cc)) for cc in gap_cols],
                "interior": {
                    (rr, cc)
                    for rr in range(r0 + 1, r1)
                    for cc in range(c0 + 1, c1)
                },
            }
        )
    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _reachable_landings(
    passable: np.ndarray,
    start: tuple[int, int],
    landing_to_frame: dict[tuple[int, int], int],
) -> list[tuple[int, int]]:
    queue: deque[tuple[int, int]] = deque([start])
    seen = {start}
    found: list[tuple[int, int]] = []
    while queue:
        r, c = queue.popleft()
        if (r, c) in landing_to_frame:
            found.append((r, c))
            continue
        for dr, dc in ((0, -1), (0, 1), (1, 0)):
            rr, cc = r + dr, c + dc
            if not in_bounds(passable.shape, rr, cc):
                continue
            if not passable[rr, cc] or (rr, cc) in seen:
                continue
            seen.add((rr, cc))
            queue.append((rr, cc))
    return found


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, dict]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    frames = _detect_frames(g)
    assert frames, "expected at least one hollow grey frame"

    tokens = [
        {"start": (int(r), int(c)), "color": int(g[r, c])}
        for r, c in np.argwhere(np.isin(g, TOKEN_COLORS))
    ]
    assert tokens, "expected colored tokens"

    color_counts = Counter(token["color"] for token in tokens)
    landing_to_frame: dict[tuple[int, int], int] = {}
    blocked_inside: set[tuple[int, int]] = set()
    for frame_id, frame in enumerate(frames):
        for landing in frame["landing_cells"]:
            landing_to_frame[landing] = frame_id
        blocked_inside |= frame["interior"] - set(frame["landing_cells"])

    passable = (g != Color.PINK) & (g != Color.GREY)
    for r, c in blocked_inside:
        passable[r, c] = False

    candidates_per_frame: dict[int, list[dict]] = {idx: [] for idx in range(len(frames))}
    token_results: list[dict] = []
    for token in tokens:
        reached = _reachable_landings(passable, token["start"], landing_to_frame)
        frame_ids = {landing_to_frame[cell] for cell in reached}
        assert len(frame_ids) <= 1, "a token should not reach multiple frames"
        if not frame_ids:
            token_results.append({**token, "frame": None, "landing": None, "winner": False})
            continue
        frame_id = next(iter(frame_ids))
        landing = min(
            (cell for cell in reached if landing_to_frame[cell] == frame_id),
            key=lambda rc: (rc[0], rc[1]),
        )
        info = {**token, "frame": frame_id, "landing": landing, "winner": False}
        candidates_per_frame[frame_id].append(info)
        token_results.append(info)

    out = g.copy()
    out[np.isin(out, TOKEN_COLORS)] = Color.BLACK

    for frame_id, frame in enumerate(frames):
        candidates = candidates_per_frame[frame_id]
        if not candidates:
            continue
        winner = min(
            candidates,
            key=lambda item: (color_counts[item["color"]], item["start"][1]),
        )
        winner["winner"] = True
        rr, cc = frame["coords"].T
        out[rr, cc] = winner["color"]
        lr, lc = winner["landing"]
        out[lr, lc] = winner["color"]

    meta = {
        "frames": frames,
        "tokens": token_results,
        "competition_frames": [
            frame_id for frame_id, items in candidates_per_frame.items() if len(items) >= 2
        ],
        "activated_frames": [
            frame_id for frame_id, items in candidates_per_frame.items() if len(items) >= 1
        ],
    }
    return out, meta


def main(grid: np.ndarray) -> np.ndarray:
    """
    Drop corridor tokens into open-top frames and recolor each reached frame by the winning token color.

    Input structure:
    - The grid has a black background, thin pink wall segments, 3x3 to 5x5 grey hollow frames near the bottom, and 3 to 6 single-pixel tokens in blue/red/green/yellow near the top.
    - Each grey frame is a hollow rectangle whose top edge has a single black opening aligned with a wall-bounded corridor.
    - Some corridors lead to a frame opening, while others terminate in sealed dead ends.

    Rule:
    - Treat pink cells and grey frame borders as barriers; a token may move only downward or sideways through black corridor cells until it either reaches a frame landing cell or gets trapped.
    - A token that reaches a frame stops on the first interior row directly under that frame's top opening; a token that reaches no frame is erased.
    - Recolor the entire border of each reached grey frame with the winner's token color, keeping only the winning token at its landing cell.
    - If multiple tokens reach the same frame, the color with the smaller global token count in the input wins; ties are broken by the smaller original token column.

    Key insight:
    - The important object is not the nearest token but the set of tokens whose wall-bounded corridor can reach each frame opening; once that set is known, global color rarity resolves any competition.
    """
    out, _meta = _simulate(grid)
    return out


def _rand_partition(total: int, parts: int) -> list[int]:
    bins = [0] * parts
    for _ in range(total):
        bins[random.randrange(parts)] += 1
    return bins


def _place_frames(height: int, width: int, count: int) -> list[dict] | None:
    frame_ws = [random.randint(3, 5) for _ in range(count)]
    frame_hs = [random.randint(3, 5) for _ in range(count)]
    min_gap = 2
    need = sum(frame_ws) + min_gap * (count - 1) + 4
    if need > width:
        return None
    extra = width - need
    margins = _rand_partition(extra, count + 1)
    frames: list[dict] = []
    c = 2 + margins[0]
    baseline = height - 2
    for idx in range(count):
        w = frame_ws[idx]
        h = frame_hs[idx]
        c0 = c
        c1 = c0 + w - 1
        r1 = baseline
        r0 = r1 - h + 1
        if r0 < 8:
            return None
        gap_col = random.randint(c0 + 1, c1 - 1)
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "gap_col": gap_col,
                "landing": (r0 + 1, gap_col),
                "interior": {
                    (rr, cc)
                    for rr in range(r0 + 1, r1)
                    for cc in range(c0 + 1, c1)
                },
                "gap": (r0, gap_col),
            }
        )
        c = c1 + 1 + min_gap + margins[idx + 1]
    return frames


def _polyline_cells(points: list[tuple[int, int]]) -> set[tuple[int, int]]:
    cells: set[tuple[int, int]] = set()
    for (r0, c0), (r1, c1) in zip(points, points[1:]):
        if r0 == r1:
            step = 1 if c1 >= c0 else -1
            for cc in range(c0, c1 + step, step):
                cells.add((r0, cc))
        else:
            step = 1 if r1 >= r0 else -1
            for rr in range(r0, r1 + step, step):
                cells.add((rr, c0))
    return cells


def _neighbors4(cell: tuple[int, int]) -> list[tuple[int, int]]:
    r, c = cell
    return [(r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)]


def _try_thicken_vertical(
    cells: set[tuple[int, int]],
    segment: tuple[int, int, int],
    width: int,
    shape: tuple[int, int],
    blocked: set[tuple[int, int]],
) -> set[tuple[int, int]] | None:
    r0, r1, c = segment
    if width <= 1:
        return set(cells)
    if r1 - r0 < 3:
        return set(cells)
    choices = []
    if width == 2:
        choices = [[0, -1], [0, 1]]
    else:
        choices = [[-1, 0, 1]]
    offsets = random.choice(choices)
    expanded = set(cells)
    for rr in range(r0, r1):
        for dc in offsets:
            cc = c + dc
            if not in_bounds(shape, rr, cc):
                return None
            if (rr, cc) in blocked:
                return None
            expanded.add((rr, cc))
    return expanded


def _network_valid(
    cells: set[tuple[int, int]],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    frame_gaps: set[tuple[int, int]],
    allowed_gaps: set[tuple[int, int]],
    forbidden_landings: set[tuple[int, int]],
    shape: tuple[int, int],
) -> bool:
    for cell in cells:
        r, c = cell
        if not in_bounds(shape, r, c):
            return False
        if cell in blocked or cell in occupied:
            return False
        if cell in frame_gaps and cell not in allowed_gaps:
            return False
        for nb in _neighbors4(cell):
            if nb in forbidden_landings:
                return False
            if nb in occupied and nb not in cells:
                return False
    return True


def _make_simple_network(
    frame: dict,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    frame_gaps: set[tuple[int, int]],
    all_landings: set[tuple[int, int]],
) -> dict | None:
    height, width = shape
    r0, _c0, _r1, _c1 = frame["bbox"]
    gap_r, gap_c = frame["gap"]
    for _ in range(100):
        start_row = random.choice([0, 1, 1, 2])
        bend_row = random.randint(start_row + 3, gap_r - 2)
        start_col = random.randint(max(1, gap_c - 7), min(width - 2, gap_c + 7))
        points = [(start_row, start_col), (bend_row, start_col), (bend_row, gap_c), (gap_r, gap_c)]
        cells = _polyline_cells(points)
        upper_seg = (start_row, bend_row, start_col)
        lower_seg = (bend_row, gap_r, gap_c)
        if random.random() < 0.7:
            width_choice = random.choice([2, 3])
            seg = random.choice([upper_seg, lower_seg])
            thicker = _try_thicken_vertical(cells, seg, width_choice, shape, blocked)
            if thicker is not None:
                cells = thicker
        if not _network_valid(
            cells,
            occupied,
            blocked,
            frame_gaps,
            {frame["gap"]},
            all_landings - {frame["landing"]},
            shape,
        ):
            continue
        return {"cells": cells, "tokens": [(start_row, start_col)], "frame": frame}
    return None


def _make_competition_network(
    frame: dict,
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    frame_gaps: set[tuple[int, int]],
    all_landings: set[tuple[int, int]],
) -> dict | None:
    height, width = shape
    gap_r, gap_c = frame["gap"]
    for _ in range(120):
        start_row = random.choice([0, 1, 1, 2])
        merge_row = random.randint(start_row + 4, gap_r - 3)
        left_lo = max(1, gap_c - 9)
        left_hi = gap_c - 3
        right_lo = gap_c + 3
        right_hi = min(width - 2, gap_c + 9)
        if left_lo > left_hi or right_lo > right_hi:
            return None
        left_col = random.randint(left_lo, left_hi)
        right_col = random.randint(right_lo, right_hi)
        cells = set()
        cells |= _polyline_cells([(start_row, left_col), (merge_row, left_col), (merge_row, gap_c)])
        cells |= _polyline_cells([(start_row, right_col), (merge_row, right_col), (merge_row, gap_c)])
        cells |= _polyline_cells([(merge_row, gap_c), (gap_r, gap_c)])
        if random.random() < 0.8:
            thicker = _try_thicken_vertical(
                cells,
                (merge_row, gap_r, gap_c),
                random.choice([2, 3]),
                shape,
                blocked,
            )
            if thicker is not None:
                cells = thicker
        if not _network_valid(
            cells,
            occupied,
            blocked,
            frame_gaps,
            {frame["gap"]},
            all_landings - {frame["landing"]},
            shape,
        ):
            continue
        return {
            "cells": cells,
            "tokens": [(start_row, left_col), (start_row, right_col)],
            "frame": frame,
        }
    return None


def _make_dead_network(
    shape: tuple[int, int],
    occupied: set[tuple[int, int]],
    blocked: set[tuple[int, int]],
    frame_gaps: set[tuple[int, int]],
    all_landings: set[tuple[int, int]],
) -> dict | None:
    height, width = shape
    for _ in range(120):
        start_row = random.choice([0, 1, 1, 2])
        start_col = random.randint(1, width - 2)
        end_row = random.randint(start_row + 5, height - 7)
        cells = _polyline_cells([(start_row, start_col), (end_row, start_col)])
        if random.random() < 0.6:
            side = random.choice([-1, 1])
            reach = random.randint(2, 4)
            end_col = max(1, min(width - 2, start_col + side * reach))
            cells |= _polyline_cells([(end_row, start_col), (end_row, end_col)])
        if random.random() < 0.5:
            thicker = _try_thicken_vertical(
                cells,
                (start_row, end_row, start_col),
                random.choice([2, 3]),
                shape,
                blocked,
            )
            if thicker is not None:
                cells = thicker
        if not _network_valid(
            cells,
            occupied,
            blocked,
            frame_gaps,
            set(),
            all_landings,
            shape,
        ):
            continue
        return {"cells": cells, "tokens": [(start_row, start_col)], "frame": None}
    return None


def _assign_colors(networks: list[dict], include_comp: bool, tie_case: bool) -> dict[tuple[int, int], int]:
    starts = [start for net in networks for start in net["tokens"]]
    pos_to_color: dict[tuple[int, int], int] = {}
    colors = list(TOKEN_COLORS)
    random.shuffle(colors)

    comp_net = next((net for net in networks if len(net["tokens"]) == 2), None)
    extra_slots = [start for start in starts if comp_net is None or start not in comp_net["tokens"]]

    if include_comp and comp_net is not None:
        left_start, right_start = sorted(comp_net["tokens"], key=lambda rc: rc[1])
        color_a, color_b = colors[:2]
        if tie_case:
            pos_to_color[left_start] = color_a
            pos_to_color[right_start] = color_b
            remaining_palette = [col for col in TOKEN_COLORS if col not in (color_a, color_b)]
            for start in extra_slots:
                pos_to_color[start] = random.choice(remaining_palette)
            return pos_to_color

        winner_start = random.choice([left_start, right_start])
        loser_start = right_start if winner_start == left_start else left_start
        winner_color = color_a
        loser_color = color_b
        pos_to_color[winner_start] = winner_color
        pos_to_color[loser_start] = loser_color
        assert extra_slots, "rarity case needs at least one extra token"
        booster = random.choice(extra_slots)
        pos_to_color[booster] = loser_color
        for start in extra_slots:
            if start == booster:
                continue
            pos_to_color[start] = random.choice([col for col in TOKEN_COLORS if col != winner_color])
        return pos_to_color

    for start in starts:
        pos_to_color[start] = random.choice(TOKEN_COLORS)
    return pos_to_color


def _build_grid(shape: tuple[int, int], frames: list[dict], networks: list[dict], pos_to_color: dict) -> np.ndarray:
    height, width = shape
    grid = np.zeros((height, width), dtype=int)

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        gap_col = frame["gap"][1] - c0
        border = _frame_border_mask(h, w, gap_col)
        grid[r0 : r1 + 1, c0 : c1 + 1][border] = Color.GREY

    open_cells = set()
    for net in networks:
        open_cells |= net["cells"]

    for r, c in open_cells:
        for rr, cc in _neighbors4((r, c)):
            if not in_bounds(shape, rr, cc):
                continue
            if (rr, cc) in open_cells:
                continue
            if any((rr, cc) == frame["landing"] for frame in frames):
                continue
            if grid[rr, cc] == Color.GREY:
                continue
            grid[rr, cc] = Color.PINK

    for pos, color in pos_to_color.items():
        grid[pos] = color

    return grid


def _generate_candidate() -> np.ndarray | None:
    frame_count = random.randint(2, 4)
    width = random.randint(18, 28)
    height = random.randint(17, 27)
    frames = _place_frames(height, width, frame_count)
    if frames is None:
        return None

    blocked = set()
    frame_gaps = set()
    all_landings = set()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        blocked |= {(rr, cc) for rr in range(r0, r1 + 1) for cc in range(c0, c1 + 1)}
        blocked -= {frame["gap"]}
        frame_gaps.add(frame["gap"])
        all_landings.add(frame["landing"])

    networks: list[dict] = []
    occupied: set[tuple[int, int]] = set()

    include_comp = frame_count >= 2 and random.random() < 0.7
    tie_case = include_comp and random.random() < 0.4
    shuffled_frames = frames[:]
    random.shuffle(shuffled_frames)

    if include_comp:
        comp_frame = shuffled_frames.pop()
        comp_net = _make_competition_network(
            comp_frame,
            (height, width),
            occupied,
            blocked,
            frame_gaps,
            all_landings,
        )
        if comp_net is None:
            return None
        networks.append(comp_net)
        occupied |= comp_net["cells"]

    simple_goal = random.randint(1, min(2, len(shuffled_frames)))
    for frame in shuffled_frames[:simple_goal]:
        net = _make_simple_network(
            frame,
            (height, width),
            occupied,
            blocked,
            frame_gaps,
            all_landings,
        )
        if net is None:
            return None
        networks.append(net)
        occupied |= net["cells"]

    token_count = sum(len(net["tokens"]) for net in networks)
    dead_goal = max(1, random.randint(3, 6) - token_count)
    for _ in range(dead_goal):
        dead_net = _make_dead_network(
            (height, width),
            occupied,
            blocked,
            frame_gaps,
            all_landings,
        )
        if dead_net is None:
            return None
        networks.append(dead_net)
        occupied |= dead_net["cells"]

    pos_to_color = _assign_colors(networks, include_comp, tie_case)
    grid = _build_grid((height, width), frames, networks, pos_to_color)

    try:
        out, meta = _simulate(grid)
    except AssertionError:
        return None

    if len(meta["activated_frames"]) == 0:
        return None
    if not any(token["frame"] is None for token in meta["tokens"]):
        return None
    if include_comp and len(meta["competition_frames"]) == 0:
        return None
    if not np.any(grid != out):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_TRIES):
        grid = _generate_candidate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid corridor-frame ARC input")
