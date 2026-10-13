from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
MAROON = 9
RING_COLORS = (1, 2, 3, 4, 5, 6, 7, 8)
CELL_STEPS = (4, 5)
MAX_GEN_TRIES = 500
MAX_DIR_TRIES = 32

DIR_TO_DELTA = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

DIR_TO_GAP = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

BORDER_OFFSETS = (
    (-1, -1),
    (-1, 0),
    (-1, 1),
    (0, -1),
    (0, 1),
    (1, -1),
    (1, 0),
    (1, 1),
)


def _infer_bg(grid: np.ndarray) -> int:
    g = np.asarray(grid, dtype=int)
    corners = [int(g[0, 0]), int(g[0, -1]), int(g[-1, 0]), int(g[-1, -1])]
    bg = int(Counter(corners).most_common(1)[0][0])
    if int((g == bg).sum()) * 2 < int(g.size):
        bg = int(Counter(map(int, g.ravel())).most_common(1)[0][0])
    return bg


def _detect_rings(grid: np.ndarray, bg: int | None = None) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    if bg is None:
        bg = _infer_bg(g)
    bg = int(bg)

    rings: list[dict] = []
    for r in range(1, h - 1):
        for c in range(1, w - 1):
            inner_color = int(g[r, c])
            if inner_color == bg:
                continue

            valid: list[tuple[str, int]] = []
            for direction, gap in DIR_TO_GAP.items():
                gap_r, gap_c = int(r + gap[0]), int(c + gap[1])
                if int(g[gap_r, gap_c]) != bg:
                    continue

                ring_color = None
                ok = True
                for dr, dc in BORDER_OFFSETS:
                    rr, cc = int(r + dr), int(c + dc)
                    value = int(g[rr, cc])
                    if (dr, dc) == gap:
                        if value != bg:
                            ok = False
                            break
                        continue
                    if value == bg or value == inner_color:
                        ok = False
                        break
                    if ring_color is None:
                        ring_color = int(value)
                    elif value != ring_color:
                        ok = False
                        break

                if ok and ring_color is not None:
                    valid.append((direction, int(ring_color)))

            if len(valid) != 1:
                continue

            direction, ring_color = valid[0]
            rings.append(
                {
                    "center": (int(r), int(c)),
                    "inner_color": int(inner_color),
                    "ring_color": int(ring_color),
                    "direction": direction,
                    "bbox": (int(r - 1), int(c - 1), int(r + 1), int(c + 1)),
                }
            )

    rings.sort(key=lambda ring: ring["center"])
    return rings


def _simulate(
    grid: np.ndarray,
    rings: list[dict] | None = None,
    *,
    strict: bool,
) -> dict | None:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    bg = _infer_bg(g)
    if rings is None:
        rings = _detect_rings(g, bg=bg)

    counts = Counter(int(ring["ring_color"]) for ring in rings)

    scene = g.copy()
    for ring in rings:
        cr, cc = ring["center"]
        scene[int(cr), int(cc)] = bg
    static_scene = scene.copy()

    moves: list[dict] = []
    for ring in rings:
        start = tuple(map(int, ring["center"]))
        if strict and int(scene[start]) != bg:
            return None

        dr, dc = DIR_TO_DELTA[str(ring["direction"])]
        steps = int(counts[int(ring["ring_color"])])
        r, c = start
        traveled = 0
        path: list[tuple[int, int]] = []
        stop_reason = "distance"

        for _ in range(steps):
            nr, nc = int(r + dr), int(c + dc)
            if not (0 <= nr < h and 0 <= nc < w):
                stop_reason = "boundary"
                break
            if int(scene[nr, nc]) != bg:
                if int(static_scene[nr, nc]) != bg:
                    stop_reason = "static_block"
                else:
                    stop_reason = "moved_block"
                break
            r, c = nr, nc
            path.append((int(r), int(c)))
            traveled += 1

        scene[int(r), int(c)] = int(ring["inner_color"])
        moves.append(
            {
                "start": start,
                "end": (int(r), int(c)),
                "path": path,
                "steps": int(steps),
                "traveled": int(traveled),
                "stop_reason": stop_reason,
                "direction": str(ring["direction"]),
                "ring_color": int(ring["ring_color"]),
            }
        )

    return {"output": scene, "moves": moves, "rings": rings, "counts": counts, "bg": bg}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Move each center square along the opening of its ring by the count of same-colored rings.
    """
    g = np.asarray(grid, dtype=int)
    rings = _detect_rings(g)
    result = _simulate(g, rings, strict=False)
    assert result is not None
    return np.asarray(result["output"], dtype=int)


def _random_composition(total: int, parts: int) -> list[int]:
    if parts == 1:
        return [int(total)]
    cuts = sorted(random.sample(range(1, total), parts - 1))
    prev = 0
    out: list[int] = []
    for cut in cuts + [total]:
        out.append(int(cut - prev))
        prev = cut
    return out


def _sample_group_sizes(total_rings: int) -> list[int]:
    max_groups = min(4, total_rings - 1)
    for _ in range(80):
        num_groups = random.randint(2, max_groups)
        sizes = _random_composition(total_rings, num_groups)
        if max(sizes) > 1 and len(set(sizes)) > 1:
            random.shuffle(sizes)
            return sizes
    return [1, total_rings - 1]


def _pick_positions(cell_rows: int, cell_cols: int, total_rings: int) -> list[tuple[int, int]]:
    cells = [(r, c) for r in range(cell_rows) for c in range(cell_cols)]
    boundary = [
        cell
        for cell in cells
        if cell[0] in (0, cell_rows - 1) or cell[1] in (0, cell_cols - 1)
    ]
    chosen = {random.choice(boundary)}
    while len(chosen) < total_rings:
        chosen.add(random.choice(cells))
    out = list(chosen)
    random.shuffle(out)
    return out


def _cell_to_center(
    cell: tuple[int, int],
    top_margin: int,
    left_margin: int,
    cell_step: int,
) -> tuple[int, int]:
    return (
        int(top_margin + 1 + cell[0] * cell_step),
        int(left_margin + 1 + cell[1] * cell_step),
    )


def _build_grid(
    specs: list[dict],
    *,
    cell_rows: int,
    cell_cols: int,
    cell_step: int,
    top_margin: int,
    bottom_margin: int,
    left_margin: int,
    right_margin: int,
) -> np.ndarray:
    h = int(top_margin + bottom_margin + cell_step * (cell_rows - 1) + 3)
    w = int(left_margin + right_margin + cell_step * (cell_cols - 1) + 3)
    grid = np.full((h, w), BG, dtype=int)

    for spec in specs:
        r, c = map(int, spec["center"])
        grid[int(r), int(c)] = MAROON
        gap = DIR_TO_GAP[str(spec["direction"])]
        for dr, dc in BORDER_OFFSETS:
            if (dr, dc) == gap:
                continue
            grid[int(r + dr), int(c + dc)] = int(spec["ring_color"])

    return grid


def _validate_instance(grid: np.ndarray, expected_rings: int) -> bool:
    rings = _detect_rings(grid, bg=BG)
    if len(rings) != expected_rings or len(rings) < 4:
        return False

    group_counts = Counter(int(ring["ring_color"]) for ring in rings)
    sizes = sorted(group_counts.values())
    if len(group_counts) < 2 or max(sizes) <= 1 or len(set(sizes)) <= 1:
        return False

    result = _simulate(grid, rings, strict=True)
    if result is None:
        return False
    moves = list(result["moves"])

    if not any(int(move["traveled"]) == int(move["steps"]) for move in moves):
        return False
    if not any(int(move["traveled"]) < int(move["steps"]) for move in moves):
        return False
    if any(str(move["stop_reason"]) == "moved_block" for move in moves):
        return False

    starts = {tuple(move["start"]) for move in moves}
    ends = [tuple(move["end"]) for move in moves]
    if len(set(ends)) != len(ends):
        return False
    if any(end in starts and end != start for start, end in zip((m["start"] for m in moves), ends)):
        return False

    ring_bboxes = [(tuple(ring["center"]), tuple(ring["bbox"])) for ring in rings]
    for move in moves:
        start = tuple(move["start"])
        for pos in move["path"]:
            rr, cc = map(int, pos)
            for center, bbox in ring_bboxes:
                if center == start:
                    continue
                r0, c0, r1, c1 = bbox
                if r0 <= rr <= r1 and c0 <= cc <= c1:
                    return False

    if not np.array_equal(main(grid), result["output"]):
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_TRIES):
        cell_step = int(random.choice(CELL_STEPS))
        max_cells = 6 if cell_step == 4 else 5
        cell_rows = random.randint(3, max_cells)
        cell_cols = random.randint(3, max_cells)
        total_cells = int(cell_rows * cell_cols)
        total_rings = random.randint(4, min(8, total_cells - 1))

        group_sizes = _sample_group_sizes(total_rings)
        colors = random.sample(RING_COLORS, len(group_sizes))

        specs: list[dict] = []
        for ring_color, size in zip(colors, group_sizes):
            for _ in range(size):
                specs.append({"ring_color": int(ring_color), "count": int(size)})
        random.shuffle(specs)

        positions = _pick_positions(cell_rows, cell_cols, total_rings)
        top_margin = random.randint(1, 3)
        bottom_margin = random.randint(1, 3)
        left_margin = random.randint(1, 3)
        right_margin = random.randint(1, 3)

        for spec, cell in zip(specs, positions):
            spec["center"] = _cell_to_center(cell, top_margin, left_margin, cell_step)

        for _ in range(MAX_DIR_TRIES):
            for spec in specs:
                spec["direction"] = random.choice(tuple(DIR_TO_DELTA))
            grid = _build_grid(
                specs,
                cell_rows=cell_rows,
                cell_cols=cell_cols,
                cell_step=cell_step,
                top_margin=top_margin,
                bottom_margin=bottom_margin,
                left_margin=left_margin,
                right_margin=right_margin,
            )
            if _validate_instance(grid, total_rings):
                return grid

    raise RuntimeError("failed to generate a valid ARC instance")
