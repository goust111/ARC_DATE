from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
ARC2_DIR = ROOT / "arc2_solver_select"
if str(ARC2_DIR) not in sys.path:
    sys.path.insert(0, str(ARC2_DIR))

from arc_utils import components_mask


BG = 7
WATER = 6
SHELF = 2
DRAIN = 1
SPECIAL_SHELF = 3
DEPTH_COLORS = (6, 1, 8, 3)
FIXED_SPREAD = 3
FIXED_HIT_LIMIT = 3
EXPAND_PAD_X = 4
EXPAND_PAD_Y = 6


@dataclass(frozen=True)
class ComboConfig:
    combo_spec: str
    diagonal: bool = False
    single_edge: bool = False
    holes: bool = False
    width_constraint: bool = False
    intermittent: bool = False
    depth_color: bool = False
    fixed_spread: bool = False
    single_source: bool = False
    drain_stop: bool = False
    distance_limit: bool = False
    hit_limit: bool = False
    hit_limit_from_source: bool = False
    dissolve: bool = False
    expand_grid: bool = False
    ordered_sources: bool = False
    holey_shelves: bool = False
    four_color: bool = False
    alt_lr: bool = False
    entry_side: bool = False


@dataclass
class SourceInfo:
    row: int
    col: int
    dc: int
    area: int
    cells: set[tuple[int, int]]


@dataclass
class BranchState:
    row: int
    col: int
    dc: int
    depth: int
    hits: int
    steps_left: int | None
    paint_phase: int


def _parse_combo_spec(combo_spec: str) -> ComboConfig:
    parts = set(combo_spec.split("+"))
    return ComboConfig(
        combo_spec=combo_spec,
        diagonal="D1/0" in parts,
        single_edge="D2/0" in parts,
        holes="D2/1" in parts,
        width_constraint="D3/1" in parts,
        intermittent="D4/0" in parts,
        depth_color="D4/1" in parts,
        fixed_spread="D5/2" in parts,
        single_source="D6/0" in parts,
        drain_stop="D7/0" in parts,
        distance_limit="D7/1" in parts,
        hit_limit="D7/3" in parts or "D11/0" in parts,
        hit_limit_from_source="D11/0" in parts,
        dissolve="D8/0" in parts,
        expand_grid="D9/2" in parts,
        ordered_sources="D10/0" in parts,
        holey_shelves="D12/3" in parts,
        four_color="D13/0" in parts,
        alt_lr="D14/2" in parts,
        entry_side="D14/3" in parts,
    )


def _copy_with_padding(grid: np.ndarray, cfg: ComboConfig) -> tuple[np.ndarray, np.ndarray, tuple[int, int]]:
    g = np.array(grid, dtype=int)
    if not cfg.expand_grid:
        return g.copy(), g.copy(), (0, 0)
    H, W = g.shape
    world = np.full((H + EXPAND_PAD_Y, W + 2 * EXPAND_PAD_X), int(BG), dtype=int)
    out = world.copy()
    world[:H, EXPAND_PAD_X:EXPAND_PAD_X + W] = g
    out[:H, EXPAND_PAD_X:EXPAND_PAD_X + W] = g
    return world, out, (0, EXPAND_PAD_X)


def _trim_output(out: np.ndarray, original: np.ndarray, cfg: ComboConfig, offset: tuple[int, int]) -> np.ndarray:
    if not cfg.expand_grid:
        return out
    row0, col0 = offset
    H, W = original.shape
    mask = out != int(BG)
    if not bool(mask.any()):
        return out[:H, col0:col0 + W]
    rows, cols = np.where(mask)
    r1 = max(H - 1, int(rows.max()))
    c0 = min(int(cols.min()), col0)
    c1 = max(col0 + W - 1, int(cols.max()))
    return out[:r1 + 1, c0:c1 + 1]


def _detect_roles(g: np.ndarray, cfg: ComboConfig) -> tuple[int, int, set[int], set[int]]:
    cnt = Counter(map(int, g.flatten()))
    bg = int(cnt.most_common(1)[0][0])
    non_bg = [int(c) for c in cnt if int(c) != int(bg)]
    assert non_bg

    touching_top = [int(c) for c in non_bg if bool((g[0] == int(c)).any())]
    if touching_top:
        water = int(min(touching_top, key=lambda c: (int(cnt[int(c)]), int(c))))
    else:
        water = int(min(non_bg, key=lambda c: (int(cnt[int(c)]), int(c))))

    rest = [int(c) for c in non_bg if int(c) != int(water)]
    assert rest

    def has_horizontal_run(color: int) -> bool:
        for r in range(g.shape[0]):
            for c in range(g.shape[1] - 1):
                if int(g[r, c]) == int(color) and int(g[r, c + 1]) == int(color):
                    return True
        return False

    shelf_colors = {int(c) for c in rest if has_horizontal_run(int(c))}
    if not shelf_colors:
        shelf_colors = {int(max(rest, key=lambda c: (int(cnt[int(c)]), -int(c))))}
    drain_colors = {int(c) for c in rest if int(c) not in shelf_colors}
    if cfg.four_color and not drain_colors and len(rest) > 1:
        drain_colors.add(int(min(rest)))
    return bg, water, shelf_colors, drain_colors


def _extract_sources(g: np.ndarray, water: int, cfg: ComboConfig) -> list[SourceInfo]:
    sources: list[SourceInfo] = []
    for coords in components_mask(g == int(water), connectivity=8):
        if coords.size == 0 or not bool((coords[:, 0] == 0).any()):
            continue
        rows = coords[:, 0]
        cols = coords[:, 1]
        area = int(coords.shape[0])
        rmax = int(rows.max())
        bottom_cols = sorted({int(c) for r, c in coords if int(r) == int(rmax)})
        col = int(bottom_cols[len(bottom_cols) // 2])
        dc = 0
        if cfg.diagonal:
            top_cols = [int(c) for r, c in coords if int(r) == int(rows.min())]
            if top_cols:
                dc = int(np.sign(col - int(round(sum(top_cols) / len(top_cols)))))
            if dc == 0:
                dc = 1 if col < g.shape[1] // 2 else -1
        cells = {(int(r), int(c)) for r, c in coords}
        sources.append(SourceInfo(row=rmax, col=col, dc=dc, area=area, cells=cells))
    return sources


def _depth_color(depth: int) -> int:
    return int(DEPTH_COLORS[min(int(depth), len(DEPTH_COLORS) - 1)])


def _should_paint_step(cfg: ComboConfig, phase: int) -> bool:
    if not cfg.intermittent:
        return True
    return int(phase) % 2 == 0


def _is_shelf(value: int, shelf_colors: set[int]) -> bool:
    return int(value) in shelf_colors


def _row_segments(world: np.ndarray, row: int, shelf_colors: set[int]) -> list[tuple[int, int, int]]:
    W = world.shape[1]
    segs: list[tuple[int, int, int]] = []
    c = 0
    while c < W:
        val = int(world[row, c])
        if not _is_shelf(val, shelf_colors):
            c += 1
            continue
        c0 = c
        while c + 1 < W and int(world[row, c + 1]) == int(val):
            c += 1
        segs.append((int(c0), int(c), int(val)))
        c += 1
    return segs


def _find_segment(segs: list[tuple[int, int, int]], col: int) -> tuple[int, int, int] | None:
    for a, b, color in segs:
        if int(a) <= int(col) <= int(b):
            return (int(a), int(b), int(color))
    return None


def _find_gap_route(
    segs: list[tuple[int, int, int]],
    col: int,
    cfg: ComboConfig,
) -> tuple[int, int, tuple[int, int, int]] | None:
    hit = _find_segment(segs, col)
    if hit is None:
        return None
    idx = segs.index(hit)
    max_gap = 2 if cfg.holey_shelves else 1
    candidates: list[tuple[int, int, tuple[int, int, int]]] = []
    for other_idx in (idx - 1, idx + 1):
        if not (0 <= other_idx < len(segs)):
            continue
        a1, b1, _ = hit
        a2, b2, _ = segs[other_idx]
        if int(a2) > int(b1):
            gap_lo, gap_hi = int(b1 + 1), int(a2 - 1)
        else:
            gap_lo, gap_hi = int(b2 + 1), int(a1 - 1)
        if gap_lo > gap_hi:
            continue
        if int(gap_hi - gap_lo + 1) > max_gap:
            continue
        candidates.append((int(gap_lo), int(gap_hi), hit))
    if not candidates:
        return None
    if cfg.width_constraint:
        candidates = [item for item in candidates if int(item[1] - item[0] + 1) >= 2]
        if not candidates:
            return None
    return min(candidates, key=lambda item: abs(((item[0] + item[1]) // 2) - int(col)))


def _entry_side(c: int, a: int, b: int) -> str:
    length = int(b - a + 1)
    offset = int(c - a)
    if length % 2 == 1 and offset == length // 2:
        return "center"
    if offset < length / 2:
        return "left"
    return "right"


def _edge_side(col: int, width: int) -> str:
    return "left" if int(col) * 2 <= int(width - 1) else "right"


def _horizontal_targets(cfg: ComboConfig, impact_col: int, a: int, b: int, width: int, hits: int) -> list[int]:
    if cfg.fixed_spread:
        left = int(impact_col - FIXED_SPREAD)
        right = int(impact_col + FIXED_SPREAD)
    else:
        left = int(a - 1)
        right = int(b + 1)
    if cfg.alt_lr:
        return [int(left if hits % 2 == 0 else right)]
    if cfg.entry_side:
        side = _entry_side(impact_col, a, b)
        if side == "left":
            return [int(left)]
        if side == "right":
            return [int(right)]
        return [int(left), int(right)]
    if cfg.single_edge:
        return [int(left if _edge_side(impact_col, width) == "left" else right)]
    return [int(left), int(right)]


def _paint_cell(out: np.ndarray, world: np.ndarray, row: int, col: int, color: int, drain_colors: set[int]) -> None:
    if not (0 <= int(row) < out.shape[0] and 0 <= int(col) < out.shape[1]):
        return
    if int(world[row, col]) in drain_colors:
        return
    out[int(row), int(col)] = int(color)


def _paint_run(
    out: np.ndarray,
    world: np.ndarray,
    start_row: int,
    start_col: int,
    stop_col: int,
    color: int,
    bg: int,
    drain_colors: set[int],
    blocker: np.ndarray | None,
) -> tuple[bool, int]:
    step = 1 if int(stop_col) >= int(start_col) else -1
    c = int(start_col)
    last = int(start_col)
    while True:
        if not (0 <= c < out.shape[1]):
            return False, int(last)
        if blocker is not None and blocker[int(start_row), int(c)] and c != int(start_col):
            return False, int(last)
        val = int(world[int(start_row), int(c)])
        if c != int(start_col) and val != int(bg):
            return False, int(last)
        _paint_cell(out, world, int(start_row), int(c), int(color), drain_colors)
        last = int(c)
        if int(c) == int(stop_col):
            return True, int(last)
        c += step


def _erase_segment(world: np.ndarray, out: np.ndarray, row: int, seg: tuple[int, int, int], bg: int) -> None:
    a, b, _ = seg
    for c in range(int(a), int(b) + 1):
        world[int(row), int(c)] = int(bg)
        out[int(row), int(c)] = int(bg)


def _limit_from_source(source: SourceInfo) -> int:
    return int(max(2, min(4, source.area)))


def _distance_budget(source: SourceInfo) -> int:
    return int(max(4, source.area * 3))


def run_main(grid: np.ndarray, combo_spec: str) -> np.ndarray:
    cfg = _parse_combo_spec(combo_spec)
    original = np.array(grid, dtype=int)
    world, out, offset = _copy_with_padding(original, cfg)
    bg, water, shelf_colors, drain_colors = _detect_roles(world, cfg)
    sources = _extract_sources(world, water, cfg)

    if cfg.ordered_sources:
        sources = sorted(sources, key=lambda src: (src.col, src.row))

    blocker = np.zeros_like(world, dtype=bool) if cfg.ordered_sources else None

    for source in sources:
        queue: deque[BranchState] = deque([
            BranchState(
                row=int(source.row),
                col=int(source.col),
                dc=int(source.dc if cfg.diagonal else 0),
                depth=0,
                hits=0,
                steps_left=_distance_budget(source) if cfg.distance_limit else None,
                paint_phase=0,
            )
        ])

        visited: set[tuple[int, int, int, int, int, int]] = set()
        before = out.copy()

        while queue:
            state = queue.popleft()
            key = (
                int(state.row),
                int(state.col),
                int(state.dc),
                int(state.depth),
                int(state.hits),
                -1 if state.steps_left is None else int(state.steps_left),
            )
            if key in visited:
                continue
            visited.add(key)

            color = _depth_color(state.depth) if cfg.depth_color else int(water)
            rr, cc = int(state.row), int(state.col)
            phase = int(state.paint_phase)
            if _should_paint_step(cfg, phase):
                _paint_cell(out, world, rr, cc, color, drain_colors)

            while True:
                nr = int(rr + 1)
                nc = int(cc + (state.dc if cfg.diagonal else 0))
                if nr >= world.shape[0]:
                    break
                if nc < 0 or nc >= world.shape[1]:
                    break
                if blocker is not None and blocker[nr, nc]:
                    break
                if cfg.drain_stop and int(world[nr, nc]) in drain_colors:
                    break
                if cfg.distance_limit and state.steps_left is not None and int(state.steps_left) <= 0:
                    break

                row_segs = _row_segments(world, nr, shelf_colors)
                seg = _find_segment(row_segs, nc)
                next_hits = int(state.hits + 1)
                next_depth = int(state.depth + 1)
                next_steps = None if state.steps_left is None else int(state.steps_left - 1)

                if seg is not None:
                    hit_limit = None
                    if cfg.hit_limit_from_source:
                        hit_limit = _limit_from_source(source)
                    elif cfg.hit_limit:
                        hit_limit = FIXED_HIT_LIMIT

                    gap_route = _find_gap_route(row_segs, nc, cfg) if (cfg.holes or cfg.holey_shelves) else None
                    if cfg.dissolve:
                        _erase_segment(world, out, nr, seg, bg)

                    if gap_route is not None:
                        gap_lo, gap_hi, touched = gap_route
                        gap_col = int((gap_lo + gap_hi) // 2)
                        ok, last = _paint_run(out, world, rr, cc, gap_col, color, bg, drain_colors, blocker)
                        if ok and (hit_limit is None or next_hits < hit_limit):
                            queue.append(
                                BranchState(
                                    row=int(rr),
                                    col=int(last),
                                    dc=int(state.dc if cfg.diagonal else 0),
                                    depth=next_depth,
                                    hits=next_hits,
                                    steps_left=next_steps,
                                    paint_phase=int(phase + abs(last - cc)),
                                )
                            )
                        break

                    a, b, _ = seg
                    targets = _horizontal_targets(cfg, nc, a, b, world.shape[1], state.hits)
                    spread_phase = int(phase)
                    children: list[BranchState] = []
                    for target in targets:
                        ok, last = _paint_run(out, world, rr, cc, target, color, bg, drain_colors, blocker)
                        spread_phase = int(spread_phase + abs(last - cc))
                        if not ok:
                            continue
                        child_dc = 0
                        if cfg.diagonal:
                            if int(last) < int(nc):
                                child_dc = -1
                            elif int(last) > int(nc):
                                child_dc = 1
                            else:
                                child_dc = int(state.dc or 1)
                        if hit_limit is None or next_hits < hit_limit:
                            children.append(
                                BranchState(
                                    row=int(rr),
                                    col=int(last),
                                    dc=int(child_dc),
                                    depth=next_depth,
                                    hits=next_hits,
                                    steps_left=next_steps,
                                    paint_phase=spread_phase,
                                )
                            )
                    if cfg.dissolve and (hit_limit is None or next_hits < hit_limit):
                        _paint_cell(out, world, nr, nc, color, drain_colors)
                        children.append(
                            BranchState(
                                row=int(nr),
                                col=int(nc),
                                dc=int(state.dc if cfg.diagonal else 0),
                                depth=next_depth,
                                hits=next_hits,
                                steps_left=next_steps,
                                paint_phase=int(phase + 1),
                            )
                        )
                    for child in children:
                        queue.append(child)
                    break

                rr, cc = nr, nc
                phase += 1
                if cfg.distance_limit and state.steps_left is not None:
                    state = BranchState(
                        row=state.row,
                        col=state.col,
                        dc=state.dc,
                        depth=state.depth,
                        hits=state.hits,
                        steps_left=int(state.steps_left - 1),
                        paint_phase=phase,
                    )
                    if int(state.steps_left) < 0:
                        break
                if _should_paint_step(cfg, phase):
                    _paint_cell(out, world, rr, cc, color, drain_colors)

        if blocker is not None:
            new_cells = (out != before) & (out != int(bg))
            blocker |= new_cells

    return _trim_output(out, original, cfg, offset)


def _choose_source_specs(cfg: ComboConfig, width: int) -> list[tuple[int, int, int]]:
    if cfg.ordered_sources:
        count = 2 if width < 18 else random.choice([2, 3])
    else:
        count = 1
        if not cfg.single_source and not cfg.diagonal and random.random() < 0.35:
            count = 2
    cols: list[int] = []
    gap = 5 if cfg.ordered_sources else 4
    for _ in range(200):
        cols = sorted(random.sample(range(1, width - 1), count))
        if all(b - a >= gap for a, b in zip(cols, cols[1:])):
            break
    dirs = []
    for idx, col in enumerate(cols):
        if cfg.diagonal:
            dc = 1 if idx % 2 == 0 else -1
            if col <= 1:
                dc = 1
            if col >= width - 2:
                dc = -1
        else:
            dc = 0
        dirs.append(dc)
    return [(int(col), int(dc), idx) for idx, (col, dc) in enumerate(zip(cols, dirs))]


def _place_sources(grid: np.ndarray, cfg: ComboConfig) -> list[SourceInfo]:
    H, W = grid.shape
    infos: list[SourceInfo] = []
    for col, dc, _ in _choose_source_specs(cfg, W):
        if cfg.diagonal:
            top_col = int(col - dc)
            grid[0, top_col] = WATER
            grid[1, col] = WATER
            cells = {(0, int(top_col)), (1, int(col))}
            infos.append(SourceInfo(row=1, col=int(col), dc=int(dc), area=2, cells=cells))
            continue

        height = 1 if cfg.single_source else 2
        if cfg.distance_limit or cfg.hit_limit_from_source:
            height = random.randint(2, 4)
        for r in range(height):
            grid[int(r), int(col)] = WATER
        cells = {(int(r), int(col)) for r in range(height)}
        infos.append(SourceInfo(row=height - 1, col=int(col), dc=0, area=height, cells=cells))
    return infos


def _choose_solid_segment(cfg: ComboConfig, width: int, impact_col: int, hits: int) -> tuple[int, int]:
    for _ in range(80):
        if cfg.expand_grid and random.random() < 0.35:
            edge = random.choice(["left", "right"])
            length = random.randint(2, min(6, width))
            if edge == "left":
                a, b = 0, length - 1
            else:
                a, b = width - length, width - 1
            if int(a) <= int(impact_col) <= int(b):
                return int(a), int(b)

        if cfg.entry_side:
            mode = random.choice(["left", "right", "center"])
            if mode == "center":
                length = random.choice([n for n in (3, 5, 7) if n <= width] or [3])
                a = int(impact_col - length // 2)
                b = int(a + length - 1)
                if 0 <= a and b < width:
                    return int(a), int(b)
                continue

        length = random.randint(3, min(8, width))
        offset = random.randint(0, length - 1)
        a = int(impact_col - offset)
        b = int(a + length - 1)
        if 0 <= a and b < width:
            return int(a), int(b)
    raise RuntimeError("failed to choose solid shelf")


def _choose_hole_parts(cfg: ComboConfig, width: int, impact_col: int) -> tuple[list[tuple[int, int]], int]:
    gap_width = 2 if cfg.width_constraint else random.choice([1, 2] if cfg.holey_shelves else [1])
    left_len = random.randint(2, 4)
    right_len = random.randint(2, 4)
    for _ in range(80):
        side = random.choice(["left", "right"])
        if side == "right":
            offset = random.randint(0, left_len - 1)
            a1 = int(impact_col - offset)
            b1 = int(a1 + left_len - 1)
            gap_lo = int(b1 + 1)
            gap_hi = int(gap_lo + gap_width - 1)
            a2 = int(gap_hi + 1)
            b2 = int(a2 + right_len - 1)
            if 0 <= a1 and b2 < width:
                return [(int(a1), int(b1)), (int(a2), int(b2))], int((gap_lo + gap_hi) // 2)
        else:
            offset = random.randint(0, right_len - 1)
            a2 = int(impact_col - offset)
            b2 = int(a2 + right_len - 1)
            gap_hi = int(a2 - 1)
            gap_lo = int(gap_hi - gap_width + 1)
            b1 = int(gap_lo - 1)
            a1 = int(b1 - left_len + 1)
            if 0 <= a1 and b2 < width:
                return [(int(a1), int(b1)), (int(a2), int(b2))], int((gap_lo + gap_hi) // 2)
    raise RuntimeError("failed to choose hole shelf")


def _mark_clear(cells: set[tuple[int, int]], row: int, col0: int, col1: int) -> None:
    lo, hi = sorted((int(col0), int(col1)))
    for c in range(lo, hi + 1):
        cells.add((int(row), int(c)))


def _base_combo_spec(combo_spec: str, *remove: str) -> str:
    parts = [part for part in combo_spec.split("+") if part not in remove]
    return "+".join(parts)


def _fallback_sources(grid: np.ndarray, cfg: ComboConfig) -> list[SourceInfo]:
    H, W = grid.shape
    infos: list[SourceInfo] = []
    cols = [W // 2]
    if cfg.ordered_sources or cfg.distance_limit:
        cols = [max(2, W // 4), min(W - 3, (3 * W) // 4)]
    for idx, col in enumerate(cols):
        if cfg.diagonal:
            dc = 1 if idx % 2 == 0 else -1
            if col <= 2:
                dc = 1
            if col >= W - 3:
                dc = -1
            grid[0, col - dc] = WATER
            grid[1, col] = WATER
            infos.append(SourceInfo(row=1, col=col, dc=dc, area=2, cells={(0, col - dc), (1, col)}))
            continue

        height = 1 if cfg.single_source else 2
        if cfg.distance_limit or cfg.hit_limit_from_source:
            height = 3 if idx == 0 else 2
        for r in range(height):
            grid[r, col] = WATER
        infos.append(SourceInfo(row=height - 1, col=col, dc=0, area=height, cells={(r, col) for r in range(height)}))
    return infos


def _fallback_attempt(combo_spec: str) -> np.ndarray | None:
    cfg = _parse_combo_spec(combo_spec)
    H = 12 if cfg.expand_grid else 20
    W = 8 if cfg.expand_grid else (22 if cfg.ordered_sources else 18)
    grid = np.full((H, W), int(BG), dtype=int)
    source_infos = _fallback_sources(grid, cfg)
    shelf_cells: set[tuple[int, int]] = set()
    queue: deque[BranchState] = deque(
        BranchState(
            row=int(src.row),
            col=int(src.col),
            dc=int(src.dc if cfg.diagonal else 0),
            depth=0,
            hits=0,
            steps_left=_distance_budget(src) if cfg.distance_limit else None,
            paint_phase=0,
        )
        for src in source_infos
    )

    max_levels = 3 if not cfg.hit_limit else 4
    if cfg.expand_grid:
        max_levels = 2

    while queue:
        state = queue.popleft()
        if state.depth >= max_levels:
            continue
        drop = 3
        shelf_row = int(state.row + drop)
        if shelf_row >= H - 1:
            continue
        impact_col = int(state.col + state.dc * drop)
        impact_col = max(1, min(W - 2, impact_col))
        color = SPECIAL_SHELF if cfg.four_color and not cfg.drain_stop and state.depth % 2 == 1 else SHELF

        parts: list[tuple[int, int]]
        targets: list[int]
        if cfg.holes or cfg.holey_shelves:
            gap_width = 2 if (cfg.width_constraint or cfg.holey_shelves) else 1
            left_len = right_len = 3
            if impact_col <= W // 2:
                a1 = max(1, impact_col - 1)
                b1 = min(W - 3 - gap_width, a1 + left_len - 1)
                gap_lo = b1 + 1
                gap_hi = gap_lo + gap_width - 1
                a2 = gap_hi + 1
                b2 = min(W - 2, a2 + right_len - 1)
                if b2 >= W - 1:
                    continue
            else:
                b2 = min(W - 2, impact_col + 1)
                a2 = max(2 + gap_width, b2 - right_len + 1)
                gap_hi = a2 - 1
                gap_lo = gap_hi - gap_width + 1
                b1 = gap_lo - 1
                a1 = max(1, b1 - left_len + 1)
                if a1 <= 0:
                    continue
            parts = [(int(a1), int(b1)), (int(a2), int(b2))]
            targets = [int((gap_lo + gap_hi) // 2)]
        else:
            if cfg.ordered_sources and state.depth == 0 and len(source_infos) >= 2:
                src_cols = sorted(src.col for src in source_infos)
                a = max(0, src_cols[0] - 2)
                b = min(W - 1, src_cols[-1] + 2)
            else:
                half = 2
                a = max(0, impact_col - half)
                b = min(W - 1, impact_col + half)
                if cfg.entry_side and state.depth == 0:
                    a = max(0, impact_col - 2)
                    b = min(W - 1, impact_col + 2)
            parts = [(int(a), int(b))]
            targets = _horizontal_targets(cfg, impact_col, a, b, W, state.hits)

        bad = False
        for a0, b0 in parts:
            for cc in range(a0, b0 + 1):
                if (shelf_row, cc) in shelf_cells:
                    bad = True
                    break
            if bad:
                break
        if bad:
            continue

        for a0, b0 in parts:
            for cc in range(a0, b0 + 1):
                shelf_cells.add((shelf_row, cc))
                grid[shelf_row, cc] = int(color)

        next_hits = int(state.hits + 1)
        next_depth = int(state.depth + 1)
        hit_limit = None
        if cfg.hit_limit_from_source:
            src_area = source_infos[0].area
            hit_limit = int(max(2, min(4, src_area)))
        elif cfg.hit_limit:
            hit_limit = FIXED_HIT_LIMIT

        if hit_limit is not None and next_hits >= hit_limit:
            continue

        for target in targets:
            child_dc = 0 if not cfg.diagonal else (-1 if target < impact_col else 1 if target > impact_col else state.dc)
            queue.append(
                BranchState(
                    row=int(shelf_row - 1),
                    col=int(max(0, min(W - 1, target))),
                    dc=int(child_dc),
                    depth=next_depth,
                    hits=next_hits,
                    steps_left=None if state.steps_left is None else int(max(1, state.steps_left - drop)),
                    paint_phase=0,
                )
            )
        if cfg.dissolve:
            queue.append(
                BranchState(
                    row=int(shelf_row),
                    col=int(impact_col),
                    dc=int(state.dc),
                    depth=next_depth,
                    hits=next_hits,
                    steps_left=None if state.steps_left is None else int(max(1, state.steps_left - drop)),
                    paint_phase=0,
                )
            )

    if cfg.drain_stop:
        base_spec = _base_combo_spec(combo_spec, "D7/0")
        base_out = run_main(grid.copy(), base_spec)
        points = [(int(r), int(c)) for r, c in zip(*np.where(base_out != BG)) if 1 < int(r) < grid.shape[0] - 1 and 0 < int(c) < grid.shape[1] - 1 and int(grid[int(r), int(c)]) == int(BG)]
        if len(points) < 2:
            return None
        for idx, (r, c) in enumerate(points[:2]):
            grid[int(r), int(c)] = int(DRAIN if not cfg.four_color or idx == 0 else SPECIAL_SHELF)

    out = run_main(grid.copy(), combo_spec)
    if np.array_equal(out, grid):
        return None
    return grid


def _special_input(combo_spec: str) -> np.ndarray | None:
    if combo_spec == "D6/0+D10/0":
        grid = np.full((14, 12), int(BG), dtype=int)
        grid[0, 3] = WATER
        grid[0, 6] = WATER
        grid[3, 2:6] = SHELF
        grid[7, 5:9] = SHELF
        out = run_main(grid.copy(), combo_spec)
        if np.array_equal(out, grid):
            return None
        return grid

    if combo_spec == "D2/1+D3/1":
        grid = np.full((16, 16), int(BG), dtype=int)
        grid[0, 7] = WATER
        grid[1, 7] = WATER
        grid[5, 5:8] = SHELF
        grid[5, 9:12] = SHELF
        grid[11, 6:11] = SHELF
        out = run_main(grid.copy(), combo_spec)
        if np.array_equal(out, grid):
            return None
        return grid

    return None


def _place_attempt(combo_spec: str) -> np.ndarray | None:
    cfg = _parse_combo_spec(combo_spec)
    H = random.randint(8, 14) if cfg.expand_grid else random.randint(12, 28)
    W = random.randint(7, 11) if cfg.expand_grid else random.randint(10, 28)
    grid = np.full((H, W), int(BG), dtype=int)
    source_infos = _place_sources(grid, cfg)

    reserved_clear = {cell for src in source_infos for cell in src.cells}
    shelf_cells: set[tuple[int, int]] = set()
    queue: deque[BranchState] = deque()
    source_order = list(source_infos)
    if cfg.ordered_sources:
        source_order.sort(key=lambda src: src.col)
    for src in source_order:
        queue.append(
            BranchState(
                row=int(src.row),
                col=int(src.col),
                dc=int(src.dc if cfg.diagonal else 0),
                depth=0,
                hits=0,
                steps_left=_distance_budget(src) if cfg.distance_limit else None,
                paint_phase=0,
            )
        )

    min_depth = 3
    if cfg.hit_limit and not cfg.hit_limit_from_source:
        min_depth = FIXED_HIT_LIMIT + 1
    max_depth = 4 if not cfg.diagonal else 5
    if cfg.distance_limit:
        max_depth = 3

    shelves_placed = 0
    used_hole = False
    used_center = False

    while queue:
        state = queue.popleft()
        row, col, dc = int(state.row), int(state.col), int(state.dc if cfg.diagonal else 0)
        depth = int(state.depth)
        steps_left = state.steps_left

        if depth >= max_depth or row + 3 >= H:
            continue
        if depth >= min_depth and random.random() < 0.25:
            continue

        placed = False
        for _ in range(120):
            drop = random.randint(2, 4)
            if steps_left is not None and drop >= int(steps_left):
                drop = max(1, int(steps_left) - 1)
            if drop < 1:
                continue
            shelf_row = int(row + drop)
            if shelf_row >= H - 1:
                continue
            spread_row = int(shelf_row - 1)
            impact_col = int(col + dc * drop)
            if not (1 <= impact_col < W - 1):
                continue

            blocked = False
            cur_col = int(col)
            for rr in range(row + 1, shelf_row):
                cur_col += dc
                if not (0 <= cur_col < W) or (int(rr), int(cur_col)) in shelf_cells:
                    blocked = True
                    break
            if blocked:
                continue

            use_hole = cfg.holes or cfg.holey_shelves
            if use_hole and random.random() < 0.75:
                parts, gap_col = _choose_hole_parts(cfg, W, impact_col)
                spread_targets = [int(gap_col)]
            else:
                a, b = _choose_solid_segment(cfg, W, impact_col, state.hits)
                parts = [(int(a), int(b))]
                spread_targets = _horizontal_targets(cfg, impact_col, a, b, W, state.hits)
                if cfg.entry_side and len(spread_targets) == 2:
                    used_center = True

            conflict = False
            for a0, b0 in parts:
                for cc in range(a0, b0 + 1):
                    if (int(shelf_row), int(cc)) in shelf_cells or (int(shelf_row), int(cc)) in reserved_clear:
                        conflict = True
                        break
                if conflict:
                    break
            if conflict:
                continue

            spread_cells = {(int(spread_row), int(impact_col))}
            for target in spread_targets:
                lo, hi = sorted((int(impact_col), int(target)))
                for cc in range(lo, hi + 1):
                    spread_cells.add((int(spread_row), int(cc)))
            if any(cell in shelf_cells for cell in spread_cells):
                continue

            for rr in range(row + 1, shelf_row):
                reserved_clear.add((int(rr), int(col + dc * (rr - row))))
            for cell in spread_cells:
                reserved_clear.add(cell)

            color = SPECIAL_SHELF if cfg.four_color and not cfg.drain_stop and shelves_placed % 3 == 2 else SHELF
            for a0, b0 in parts:
                for cc in range(a0, b0 + 1):
                    shelf_cells.add((int(shelf_row), int(cc)))
                    grid[int(shelf_row), int(cc)] = int(color)

            next_depth = int(depth + 1)
            next_hits = int(state.hits + 1)
            next_steps = None if steps_left is None else int(steps_left - drop)
            if use_hole and random.random() < 0.75:
                used_hole = True
                queue.append(
                    BranchState(
                        row=int(spread_row),
                        col=int(spread_targets[0]),
                        dc=int(dc),
                        depth=next_depth,
                        hits=next_hits,
                        steps_left=next_steps,
                        paint_phase=int(state.paint_phase + abs(spread_targets[0] - impact_col)),
                    )
                )
            else:
                hit_limit = None
                if cfg.hit_limit_from_source:
                    src_area = next((src.area for src in source_infos if src.col == col or src.col == impact_col), source_infos[0].area)
                    hit_limit = int(max(2, min(4, src_area)))
                elif cfg.hit_limit:
                    hit_limit = FIXED_HIT_LIMIT

                if hit_limit is None or next_hits < hit_limit:
                    for target in spread_targets:
                        child_dc = 0 if not cfg.diagonal else (-1 if target < impact_col else 1 if target > impact_col else dc)
                        queue.append(
                            BranchState(
                                row=int(spread_row),
                                col=int(target),
                                dc=int(child_dc),
                                depth=next_depth,
                                hits=next_hits,
                                steps_left=next_steps,
                                paint_phase=int(state.paint_phase + abs(target - impact_col)),
                            )
                        )
                    if cfg.dissolve:
                        queue.append(
                            BranchState(
                                row=int(shelf_row),
                                col=int(impact_col),
                                dc=int(dc),
                                depth=next_depth,
                                hits=next_hits,
                                steps_left=next_steps,
                                paint_phase=int(state.paint_phase + 1),
                            )
                        )
            shelves_placed += 1
            placed = True
            break

        if not placed and depth < min_depth - 1:
            return None

    if shelves_placed < 2:
        return None
    if (cfg.holes or cfg.holey_shelves) and not used_hole:
        return None
    if cfg.entry_side and not used_center and random.random() < 0.6:
        return None

    if cfg.drain_stop:
        base_spec = _base_combo_spec(combo_spec, "D7/0")
        base_out = run_main(grid.copy(), base_spec)
        candidates: list[tuple[int, int]] = []
        h2, w2 = min(grid.shape[0], base_out.shape[0]), min(grid.shape[1], base_out.shape[1])
        for r in range(2, h2 - 1):
            for c in range(1, w2 - 1):
                if int(grid[r, c]) != int(BG) or int(base_out[r, c]) == int(BG):
                    continue
                candidates.append((int(r), int(c)))
        if len(candidates) < 2:
            return None
        random.shuffle(candidates)
        for idx, (r, c) in enumerate(candidates[: random.choice([2, 3])]):
            grid[int(r), int(c)] = int(DRAIN if idx % 2 == 0 else (SPECIAL_SHELF if cfg.four_color else DRAIN))

    out = run_main(grid.copy(), combo_spec)
    if np.array_equal(out, grid):
        return None
    if cfg.depth_color and len(set(int(v) for v in np.unique(out) if int(v) != int(BG))) < 2:
        return None
    if cfg.dissolve and int(np.sum((grid == SHELF) | (grid == SPECIAL_SHELF))) <= int(np.sum((out == SHELF) | (out == SPECIAL_SHELF))):
        return None
    if cfg.expand_grid and not (out.shape[0] > grid.shape[0] or out.shape[1] > grid.shape[1]):
        return None
    if cfg.distance_limit and len(source_infos) >= 2:
        path_lengths = [int(np.sum(out[:, src.col] != BG)) for src in source_infos if src.col < out.shape[1]]
        if len(set(path_lengths)) < 2:
            return None
    return grid


def generate_input_for_combo(combo_spec: str) -> np.ndarray:
    cfg = _parse_combo_spec(combo_spec)
    special = _special_input(combo_spec)
    if special is not None:
        return special
    if cfg.fixed_spread or cfg.four_color:
        for _ in range(16):
            result = _fallback_attempt(combo_spec)
            if result is not None:
                return result
    for _ in range(80 if (cfg.fixed_spread or cfg.four_color) else 120):
        result = _place_attempt(combo_spec)
        if result is not None:
            return result
    for _ in range(40):
        result = _fallback_attempt(combo_spec)
        if result is not None:
            return result
    raise RuntimeError(f"Failed to generate valid input for {combo_spec} after fallback attempts")


def build_combo_functions(combo_spec: str):
    def main(grid: np.ndarray) -> np.ndarray:
        return run_main(grid, combo_spec)

    def generate_input() -> np.ndarray:
        return generate_input_for_combo(combo_spec)

    return main, generate_input
