import random
from collections import defaultdict, deque

import numpy as np


BG = 0
COLORS = tuple(range(1, 10))
DIR4 = ((1, 0), (-1, 0), (0, 1), (0, -1))
MAX_GENERATION_ATTEMPTS = 300


def _find_rectangles(grid: np.ndarray) -> list[dict]:
    h, w = grid.shape
    visited = np.zeros((h, w), dtype=bool)
    rectangles: list[dict] = []

    for r in range(h):
        for c in range(w):
            if visited[r, c] or int(grid[r, c]) == BG:
                continue
            color = int(grid[r, c])
            queue = deque([(r, c)])
            visited[r, c] = True
            cells: list[tuple[int, int]] = []

            while queue:
                rr, cc = queue.popleft()
                cells.append((rr, cc))
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not visited[nr, nc] and int(grid[nr, nc]) == color:
                        visited[nr, nc] = True
                        queue.append((nr, nc))

            rows = [rr for rr, _ in cells]
            cols = [cc for _, cc in cells]
            r0, r1 = min(rows), max(rows)
            c0, c1 = min(cols), max(cols)
            patch = grid[r0 : r1 + 1, c0 : c1 + 1]
            if np.all(patch == color):
                rectangles.append(
                    {
                        "color": color,
                        "r0": r0,
                        "r1": r1,
                        "c0": c0,
                        "c1": c1,
                        "h": r1 - r0 + 1,
                        "w": c1 - c0 + 1,
                    }
                )

    return rectangles


def _pair_candidates(grid: np.ndarray, rectangles: list[dict]) -> list[dict]:
    candidates: list[dict] = []

    for i in range(len(rectangles)):
        a = rectangles[i]
        for j in range(i + 1, len(rectangles)):
            b = rectangles[j]
            if a["color"] != b["color"] or a["h"] != b["h"] or a["w"] != b["w"]:
                continue

            if a["r0"] == b["r0"] and a["r1"] == b["r1"]:
                left, right = (a, b) if a["c0"] < b["c0"] else (b, a)
                gap = right["c0"] - left["c1"] - 1
                if gap > 0 and np.all(grid[left["r0"] : left["r1"] + 1, left["c1"] + 1 : right["c0"]] == BG):
                    candidates.append(
                        {"i": i, "j": j, "orientation": "horizontal", "gap": gap, "first": left, "second": right}
                    )

            if a["c0"] == b["c0"] and a["c1"] == b["c1"]:
                top, bottom = (a, b) if a["r0"] < b["r0"] else (b, a)
                gap = bottom["r0"] - top["r1"] - 1
                if gap > 0 and np.all(grid[top["r1"] + 1 : bottom["r0"], top["c0"] : top["c1"] + 1] == BG):
                    candidates.append(
                        {"i": i, "j": j, "orientation": "vertical", "gap": gap, "first": top, "second": bottom}
                    )

    return candidates


def main(grid: np.ndarray) -> np.ndarray:
    """
    Find pairs of equal solid rectangles and fill half of the gap from both sides.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    rectangles = _find_rectangles(g)
    candidates = _pair_candidates(g, rectangles)

    by_rect: dict[int, list[int]] = defaultdict(list)
    for idx, cand in enumerate(candidates):
        by_rect[cand["i"]].append(idx)
        by_rect[cand["j"]].append(idx)

    used_rects: set[int] = set()
    chosen: list[dict] = []
    for cand in sorted(candidates, key=lambda item: (item["gap"], item["orientation"], item["first"]["r0"], item["first"]["c0"])):
        if cand["i"] in used_rects or cand["j"] in used_rects:
            continue
        chosen.append(cand)
        used_rects.add(cand["i"])
        used_rects.add(cand["j"])

    for cand in chosen:
        first = cand["first"]
        second = cand["second"]
        half = cand["gap"] // 2
        if half == 0:
            continue

        if cand["orientation"] == "horizontal":
            out[first["r0"] : first["r1"] + 1, first["c1"] + 1 : first["c1"] + 1 + half] = first["color"]
            out[second["r0"] : second["r1"] + 1, second["c0"] - half : second["c0"]] = second["color"]
        else:
            out[first["r1"] + 1 : first["r1"] + 1 + half, first["c0"] : first["c1"] + 1] = first["color"]
            out[second["r0"] - half : second["r0"], second["c0"] : second["c1"] + 1] = second["color"]

    return out


def _paint_pair(
    grid: np.ndarray,
    reserved: np.ndarray,
    color: int,
    orientation: str,
    top: int,
    left: int,
    rect_h: int,
    rect_w: int,
    gap: int,
) -> None:
    if orientation == "horizontal":
        grid[top : top + rect_h, left : left + rect_w] = color
        right_left = left + rect_w + gap
        grid[top : top + rect_h, right_left : right_left + rect_w] = color
        r0, r1 = top, top + rect_h - 1
        c0, c1 = left, right_left + rect_w - 1
    else:
        grid[top : top + rect_h, left : left + rect_w] = color
        bottom_top = top + rect_h + gap
        grid[bottom_top : bottom_top + rect_h, left : left + rect_w] = color
        r0, r1 = top, bottom_top + rect_h - 1
        c0, c1 = left, left + rect_w - 1

    rr0 = max(0, r0 - 1)
    rr1 = min(grid.shape[0] - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(grid.shape[1] - 1, c1 + 1)
    reserved[rr0 : rr1 + 1, cc0 : cc1 + 1] = True


def _can_place(
    reserved: np.ndarray,
    orientation: str,
    top: int,
    left: int,
    rect_h: int,
    rect_w: int,
    gap: int,
) -> bool:
    if orientation == "horizontal":
        r0, r1 = top, top + rect_h - 1
        c0, c1 = left, left + 2 * rect_w + gap - 1
    else:
        r0, r1 = top, top + 2 * rect_h + gap - 1
        c0, c1 = left, left + rect_w - 1
    rr0 = max(0, r0 - 1)
    rr1 = min(reserved.shape[0] - 1, r1 + 1)
    cc0 = max(0, c0 - 1)
    cc1 = min(reserved.shape[1] - 1, c1 + 1)
    return not np.any(reserved[rr0 : rr1 + 1, cc0 : cc1 + 1])


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(12, 24)
    width = random.randint(12, 24)
    grid = np.full((height, width), BG, dtype=int)
    reserved = np.zeros((height, width), dtype=bool)

    pair_count = random.randint(2, 4)
    colors = random.sample(COLORS, pair_count)

    orientations = [random.choice(("horizontal", "vertical")) for _ in range(pair_count)]
    if pair_count >= 2:
        orientations[0] = "horizontal"
        orientations[1] = "vertical"
        random.shuffle(orientations)

    parities = [random.choice((0, 1)) for _ in range(pair_count)]
    if pair_count >= 2:
        parities[0] = 0
        parities[1] = 1
        random.shuffle(parities)

    specs = list(zip(colors, orientations, parities))
    random.shuffle(specs)

    for color, orientation, parity in specs:
        placed = False
        for _ in range(200):
            rect_h = random.randint(1, 4)
            rect_w = random.randint(1, 4)
            max_gap = 7
            possible_gaps = [g for g in range(1, max_gap + 1) if g % 2 == parity]
            if not possible_gaps:
                continue
            gap = random.choice(possible_gaps)

            if orientation == "horizontal":
                total_h = rect_h
                total_w = 2 * rect_w + gap
            else:
                total_h = 2 * rect_h + gap
                total_w = rect_w

            if total_h > height or total_w > width:
                continue

            top = random.randint(0, height - total_h)
            left = random.randint(0, width - total_w)
            if not _can_place(reserved, orientation, top, left, rect_h, rect_w, gap):
                continue

            _paint_pair(grid, reserved, color, orientation, top, left, rect_h, rect_w, gap)
            placed = True
            break

        if not placed:
            return None

    rectangles = _find_rectangles(grid)
    if len(rectangles) != 2 * pair_count:
        return None

    candidates = _pair_candidates(grid, rectangles)
    if len(candidates) != pair_count:
        return None

    degrees = defaultdict(int)
    for cand in candidates:
        degrees[cand["i"]] += 1
        degrees[cand["j"]] += 1
    if any(degrees[idx] != 1 for idx in range(len(rectangles))):
        return None

    orientations_seen = {cand["orientation"] for cand in candidates}
    if pair_count >= 2 and orientations_seen != {"horizontal", "vertical"}:
        return None

    parities_seen = {cand["gap"] % 2 for cand in candidates}
    if pair_count >= 2 and parities_seen != {0, 1}:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
