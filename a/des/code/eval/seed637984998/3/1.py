import random

import numpy as np


BG_COLOR = 0
DEFAULT_FRAME_COLOR = 5
SEED_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
ORTHO_DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))

GRID_HEIGHT_RANGE = (18, 30)
GRID_WIDTH_RANGE = (18, 30)
INNER_HEIGHT_RANGE = (3, 5)
INNER_WIDTH_RANGE = (3, 5)
GEN_RETRY_LIMIT = 300
FRAME_RETRY_LIMIT = 180

_GEN_CALL_COUNT = 0


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                comp.append((cr, cc))
                for dr, dc in ORTHO_DIRS:
                    nr = cr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _perpendiculars(dr: int, dc: int) -> tuple[tuple[int, int], tuple[int, int]]:
    if dr != 0:
        return (0, -1), (0, 1)
    return (-1, 0), (1, 0)


def _draw_segment(
    grid: np.ndarray,
    bg: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    color: int,
    include_start: bool,
) -> list[tuple[int, int]]:
    h, w = grid.shape
    base_r, base_c = start
    dr, dc = direction
    pts: list[tuple[int, int]] = []
    for step in range(length):
        offset = step if include_start else step + 1
        rr = base_r + dr * offset
        cc = base_c + dc * offset
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if int(grid[rr, cc]) != bg:
            break
        grid[rr, cc] = color
        pts.append((rr, cc))
    return pts


def _draw_segment_on_set(
    occupied: set[tuple[int, int]],
    h: int,
    w: int,
    start: tuple[int, int],
    direction: tuple[int, int],
    length: int,
    include_start: bool,
) -> list[tuple[int, int]]:
    base_r, base_c = start
    dr, dc = direction
    pts: list[tuple[int, int]] = []
    for step in range(length):
        offset = step if include_start else step + 1
        rr = base_r + dr * offset
        cc = base_c + dc * offset
        if not (0 <= rr < h and 0 <= cc < w):
            break
        if (rr, cc) in occupied:
            break
        occupied.add((rr, cc))
        pts.append((rr, cc))
    return pts


def _paint_tree(
    out: np.ndarray,
    bg: int,
    gap: tuple[int, int],
    grow_dir: tuple[int, int],
    layer_colors: list[int],
) -> None:
    trunk = _draw_segment(out, bg, gap, grow_dir, 3, layer_colors[0], include_start=True)
    if not trunk or len(layer_colors) < 2:
        return

    secondary_color = layer_colors[1]
    secondary_tips: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    tip = trunk[-1]
    for branch_dir in _perpendiculars(*grow_dir):
        branch = _draw_segment(out, bg, tip, branch_dir, 2, secondary_color, include_start=False)
        secondary_tips.append((branch_dir, branch))

    if len(layer_colors) < 3:
        return

    tertiary_color = layer_colors[2]
    for branch_dir, branch in secondary_tips:
        if not branch:
            continue
        branch_tip = branch[-1]
        for leaf_dir in _perpendiculars(*branch_dir):
            _draw_segment(out, bg, branch_tip, leaf_dir, 1, tertiary_color, include_start=False)


def _frame_info(
    grid: np.ndarray,
    comp: list[tuple[int, int]],
    frame_color: int,
    bg: int,
) -> dict[str, object] | None:
    rows = [r for r, _ in comp]
    cols = [c for _, c in comp]
    r0, r1 = min(rows), max(rows)
    c0, c1 = min(cols), max(cols)
    if r1 - r0 < 2 or c1 - c0 < 2:
        return None

    border = {
        (r, c)
        for r in range(r0, r1 + 1)
        for c in range(c0, c1 + 1)
        if r in (r0, r1) or c in (c0, c1)
    }
    frame_cells = set(comp)
    if not frame_cells.issubset(border):
        return None

    missing = sorted(border - frame_cells)
    if len(missing) != 1:
        return None
    gap = missing[0]
    gap_r, gap_c = gap
    if int(grid[gap_r, gap_c]) != bg:
        return None
    if gap_r in (r0, r1) and gap_c in (c0, c1):
        return None

    if gap_r == r0:
        grow_dir = (-1, 0)
    elif gap_r == r1:
        grow_dir = (1, 0)
    elif gap_c == c0:
        grow_dir = (0, -1)
    elif gap_c == c1:
        grow_dir = (0, 1)
    else:
        return None

    seeds: list[tuple[int, int]] = []
    for r in range(r0 + 1, r1):
        for c in range(c0 + 1, c1):
            value = int(grid[r, c])
            if value == frame_color:
                return None
            if value != bg:
                seeds.append((r, c))

    if not (1 <= len(seeds) <= 3):
        return None

    seeds.sort()
    colors = [int(grid[r, c]) for r, c in seeds]
    if len(set(colors)) != len(colors):
        return None
    for i, (r1s, c1s) in enumerate(seeds):
        for r2s, c2s in seeds[i + 1 :]:
            if abs(r1s - r2s) <= 1 and abs(c1s - c2s) <= 1:
                return None

    return {
        "bbox": (r0, c0, r1, c1),
        "gap": gap,
        "grow_dir": grow_dir,
        "frame_color": frame_color,
        "seed_cells": seeds,
        "seed_colors": colors,
    }


def _extract_frames(grid: np.ndarray) -> tuple[list[dict[str, object]], int]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    colors = [int(v) for v in np.unique(g) if int(v) != bg]

    best_frames: list[dict[str, object]] = []
    best_score: tuple[int, int, int] | None = None
    for color in colors:
        frames = []
        for comp in _components(g == color):
            info = _frame_info(g, comp, color, bg)
            if info is not None:
                frames.append(info)
        if not frames:
            continue
        score = (len(frames), sum(len(f["seed_cells"]) for f in frames), color == DEFAULT_FRAME_COLOR)
        if best_score is None or score > best_score:
            best_score = score
            best_frames = frames

    best_frames.sort(key=lambda item: (item["bbox"][0], item["bbox"][1], item["gap"][0], item["gap"][1]))
    return best_frames, bg


def main(grid: np.ndarray) -> np.ndarray:
    """
    Detect hollow frames with one-pixel gaps, then grow a tree from each gap.

    The gap side determines the trunk direction. Seed pixels inside the frame are
    read in row-major order; the first seed colors the trunk, the second colors the
    two perpendicular branches, and the third colors one-pixel branches
    perpendicular to each second-level branch.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    frames, bg = _extract_frames(g)
    for frame in frames:
        _paint_tree(
            out,
            bg,
            frame["gap"],
            frame["grow_dir"],
            list(frame["seed_colors"]),
        )
    return out


def _tree_layout(
    h: int,
    w: int,
    gap: tuple[int, int],
    grow_dir: tuple[int, int],
    seed_count: int,
) -> tuple[set[tuple[int, int]], dict[str, list[int] | int]]:
    occupied: set[tuple[int, int]] = set()
    lengths: dict[str, list[int] | int] = {"trunk": 0, "secondary": [], "tertiary": []}

    trunk = _draw_segment_on_set(occupied, h, w, gap, grow_dir, 3, include_start=True)
    lengths["trunk"] = len(trunk)
    if not trunk or seed_count < 2:
        return occupied, lengths

    tip = trunk[-1]
    secondary: list[tuple[tuple[int, int], list[tuple[int, int]]]] = []
    for branch_dir in _perpendiculars(*grow_dir):
        branch = _draw_segment_on_set(occupied, h, w, tip, branch_dir, 2, include_start=False)
        lengths["secondary"].append(len(branch))
        secondary.append((branch_dir, branch))

    if seed_count < 3:
        return occupied, lengths

    for branch_dir, branch in secondary:
        if not branch:
            continue
        branch_tip = branch[-1]
        for leaf_dir in _perpendiculars(*branch_dir):
            leaf = _draw_segment_on_set(occupied, h, w, branch_tip, leaf_dir, 1, include_start=False)
            lengths["tertiary"].append(len(leaf))
    return occupied, lengths


def _has_full_tree(lengths: dict[str, list[int] | int], seed_count: int) -> bool:
    if int(lengths["trunk"]) != 3:
        return False
    if seed_count >= 2 and list(lengths["secondary"]) != [2, 2]:
        return False
    if seed_count >= 3 and list(lengths["tertiary"]) != [1, 1, 1, 1]:
        return False
    return True


def _bbox_cells(r0: int, c0: int, outer_h: int, outer_w: int) -> set[tuple[int, int]]:
    return {
        (r, c)
        for r in range(r0, r0 + outer_h)
        for c in range(c0, c0 + outer_w)
    }


def _dilate(cells: set[tuple[int, int]], h: int, w: int) -> set[tuple[int, int]]:
    out: set[tuple[int, int]] = set()
    for r, c in cells:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = r + dr
                nc = c + dc
                if 0 <= nr < h and 0 <= nc < w:
                    out.add((nr, nc))
    return out


def _sample_seed_offsets(inner_h: int, inner_w: int, count: int) -> list[tuple[int, int]] | None:
    cells = [(r, c) for r in range(inner_h) for c in range(inner_w)]
    for _ in range(80):
        random.shuffle(cells)
        chosen: list[tuple[int, int]] = []
        for cell in cells:
            if all(abs(cell[0] - r) > 1 or abs(cell[1] - c) > 1 for r, c in chosen):
                chosen.append(cell)
                if len(chosen) == count:
                    return sorted(chosen)
    return None


def _frame_cells_for_spec(
    r0: int,
    c0: int,
    outer_h: int,
    outer_w: int,
    gap: tuple[int, int],
) -> set[tuple[int, int]]:
    cells = set()
    for r in range(r0, r0 + outer_h):
        for c in range(c0, c0 + outer_w):
            if r in (r0, r0 + outer_h - 1) or c in (c0, c0 + outer_w - 1):
                if (r, c) != gap:
                    cells.add((r, c))
    return cells


def _spec_signature(spec: dict[str, object]) -> tuple[object, ...]:
    return (
        spec["bbox"],
        spec["gap"],
        spec["grow_dir"],
        tuple(spec["seed_cells"]),
        tuple(spec["seed_colors"]),
    )


def _expected_output(grid: np.ndarray, specs: list[dict[str, object]], bg: int) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    for spec in sorted(specs, key=lambda item: (item["bbox"][0], item["bbox"][1])):
        _paint_tree(out, bg, spec["gap"], spec["grow_dir"], list(spec["seed_colors"]))
    return out


def _primary_seed_count() -> int:
    global _GEN_CALL_COUNT
    count = (1, 2, 3)[_GEN_CALL_COUNT % 3]
    _GEN_CALL_COUNT += 1
    return count


def _place_frame(
    h: int,
    w: int,
    blocked: set[tuple[int, int]],
    required_seed_count: int,
) -> tuple[dict[str, object], set[tuple[int, int]]] | None:
    for _ in range(FRAME_RETRY_LIMIT):
        inner_h = random.randint(*INNER_HEIGHT_RANGE)
        inner_w = random.randint(*INNER_WIDTH_RANGE)
        outer_h = inner_h + 2
        outer_w = inner_w + 2
        grow_dir = random.choice(ORTHO_DIRS)

        max_r0 = h - outer_h
        max_c0 = w - outer_w
        if max_r0 < 0 or max_c0 < 0:
            return None
        r0 = random.randint(0, max_r0)
        c0 = random.randint(0, max_c0)

        if grow_dir == (-1, 0):
            gap = (r0, c0 + random.randint(1, outer_w - 2))
        elif grow_dir == (1, 0):
            gap = (r0 + outer_h - 1, c0 + random.randint(1, outer_w - 2))
        elif grow_dir == (0, -1):
            gap = (r0 + random.randint(1, outer_h - 2), c0)
        else:
            gap = (r0 + random.randint(1, outer_h - 2), c0 + outer_w - 1)

        seed_offsets = _sample_seed_offsets(inner_h, inner_w, required_seed_count)
        if seed_offsets is None:
            continue
        seed_palette = random.sample(SEED_COLORS, required_seed_count)
        seed_cells = [(r0 + 1 + dr, c0 + 1 + dc) for dr, dc in seed_offsets]

        tree_cells, lengths = _tree_layout(h, w, gap, grow_dir, required_seed_count)
        if not _has_full_tree(lengths, required_seed_count):
            continue

        bbox_cells = _bbox_cells(r0, c0, outer_h, outer_w)
        candidate_cells = bbox_cells | tree_cells
        if candidate_cells & blocked:
            continue

        spec = {
            "bbox": (r0, c0, r0 + outer_h - 1, c0 + outer_w - 1),
            "gap": gap,
            "grow_dir": grow_dir,
            "frame_color": DEFAULT_FRAME_COLOR,
            "seed_cells": seed_cells,
            "seed_colors": seed_palette,
            "frame_cells": _frame_cells_for_spec(r0, c0, outer_h, outer_w, gap),
        }
        new_blocked = _dilate(candidate_cells, h, w)
        return spec, new_blocked
    return None


def _stamp_frame(grid: np.ndarray, spec: dict[str, object]) -> None:
    for r, c in spec["frame_cells"]:
        grid[r, c] = DEFAULT_FRAME_COLOR
    for (r, c), color in zip(spec["seed_cells"], spec["seed_colors"], strict=False):
        grid[r, c] = color


def _generate_attempt() -> np.ndarray | None:
    h = random.randint(*GRID_HEIGHT_RANGE)
    w = random.randint(*GRID_WIDTH_RANGE)
    grid = np.full((h, w), BG_COLOR, dtype=int)

    max_frames = 4 if h * w >= 520 else 3
    target_frames = random.randint(1, max_frames)
    seed_counts = [_primary_seed_count()]
    while len(seed_counts) < target_frames:
        seed_counts.append(random.choice((1, 2, 3)))
    random.shuffle(seed_counts)

    blocked: set[tuple[int, int]] = set()
    specs: list[dict[str, object]] = []
    for count in seed_counts:
        placed = _place_frame(h, w, blocked, count)
        if placed is None:
            if not specs:
                return None
            break
        spec, block_update = placed
        _stamp_frame(grid, spec)
        blocked |= block_update
        specs.append(spec)

    extracted, bg = _extract_frames(grid)
    if len(extracted) != len(specs):
        return None
    if sorted(_spec_signature(spec) for spec in extracted) != sorted(_spec_signature(spec) for spec in specs):
        return None

    expected = _expected_output(grid, specs, bg)
    actual = main(grid)
    if np.array_equal(actual, grid):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRY_LIMIT):
        grid = _generate_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
