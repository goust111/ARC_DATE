from __future__ import annotations

import random

import numpy as np


BLACK = 0
GREY = 5
COLOR_POOL = [1, 2, 3, 4, 6, 7, 8, 9]

# Generator configuration.
GRID_HEIGHT_RANGE = (18, 28)
GRID_WIDTH_RANGE = (18, 30)
RECT_COUNT_RANGE = (2, 4)
RECT_HEIGHT_RANGE = (7, 10)
RECT_WIDTH_RANGE = (7, 10)
RECT_PAD = 2
MAX_MINOR_PIXELS = 4
MAX_EXTRA_MAJOR = 4
PLACEMENT_RETRIES = 150
GENERATION_RETRIES = 400


def _components4(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    components: list[np.ndarray] = []

    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.asarray(coords, dtype=int))

    return components


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _boxes_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _expand_box(box: tuple[int, int, int, int], pad: int, h: int, w: int) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return (
        max(0, int(r0) - int(pad)),
        max(0, int(c0) - int(pad)),
        min(int(h) - 1, int(r1) + int(pad)),
        min(int(w) - 1, int(c1) + int(pad)),
    )


def _frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    grey_components = _components4(grid == GREY)
    assert grey_components, "expected at least one grey frame"

    frames: list[tuple[int, int, int, int]] = []
    for coords in grey_components:
        r0, c0, r1, c1 = _bbox(coords)
        assert r1 - r0 >= 4 and c1 - c0 >= 4, "frame must contain an interior"

        sub = grid[r0 : r1 + 1, c0 : c1 + 1]
        assert np.all(sub[0, :] == GREY) and np.all(sub[-1, :] == GREY), "frame rows must be solid grey"
        assert np.all(sub[:, 0] == GREY) and np.all(sub[:, -1] == GREY), "frame columns must be solid grey"
        assert not np.any(sub[1:-1, 1:-1] == GREY), "grey must only appear on the border"

        perim = 2 * ((r1 - r0 + 1) + (c1 - c0 + 1)) - 4
        assert len(coords) == perim, "grey component must be a one-cell-thick rectangle"
        frames.append((r0, c0, r1, c1))

    frames.sort()
    return frames


def _side_distances(r: int, c: int, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = box
    return int(r - r0), int(c - c0), int(r1 - r), int(c1 - c)


def _escape_side(r: int, c: int, box: tuple[int, int, int, int]) -> str:
    top, left, bottom, right = _side_distances(r, c, box)
    return min(
        (
            (int(top), 0, "top"),
            (int(left), 1, "left"),
            (int(bottom), 2, "bottom"),
            (int(right), 3, "right"),
        )
    )[2]


def _escape_target(r: int, c: int, side: str, box: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = box
    if side == "top":
        return int(r0 - 1), int(c)
    if side == "left":
        return int(r), int(c0 - 1)
    if side == "bottom":
        return int(r1 + 1), int(c)
    return int(r), int(c1 + 1)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recolor each grey rectangle border by the majority interior color, then let the minority pixels escape outside.

    - Every grey rectangular frame is processed independently.
    - The more frequent of the two interior colors repaints the border and stays in place.
    - Each minority pixel is removed from the interior and copied to the cell just outside its nearest side.
    - Ties prefer top, then left, then bottom, then right.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    out = g.copy()
    for box in _frame_boxes(g):
        r0, c0, r1, c1 = box
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        colored = interior[(interior != BLACK) & (interior != GREY)]
        colors, counts = np.unique(colored, return_counts=True)
        assert colors.size == 2, "expected exactly two interior colors"
        assert counts[0] != counts[1], "expected a strict majority color"

        major_idx = int(np.argmax(counts))
        minor_idx = 1 - major_idx
        major = int(colors[major_idx])
        minor = int(colors[minor_idx])

        out[r0, c0 : c1 + 1] = major
        out[r1, c0 : c1 + 1] = major
        out[r0 : r1 + 1, c0] = major
        out[r0 : r1 + 1, c1] = major

        for rr, cc in np.argwhere(interior == minor):
            r = int(r0 + 1 + rr)
            c = int(c0 + 1 + cc)
            side = _escape_side(r, c, box)
            tr, tc = _escape_target(r, c, side, box)
            assert 0 <= tr < g.shape[0] and 0 <= tc < g.shape[1], "escape target must stay in bounds"
            out[r, c] = BLACK
            out[tr, tc] = minor

    return out


def _minority_candidates(box: tuple[int, int, int, int]) -> list[dict]:
    r0, c0, r1, c1 = box
    candidates: list[dict] = []
    for r in range(r0 + 2, r1 - 1):
        for c in range(c0 + 2, c1 - 1):
            top, left, bottom, right = _side_distances(r, c, box)
            dmin = min(top, left, bottom, right)
            side = _escape_side(r, c, box)
            candidates.append(
                {
                    "cell": (int(r), int(c)),
                    "side": side,
                    "dest": _escape_target(r, c, side, box),
                    "tie": sum(d == dmin for d in (top, left, bottom, right)) > 1,
                }
            )
    return candidates


def _choose_minority(candidates: list[dict]) -> list[dict] | None:
    usable = len(candidates)
    unique_dests = len({cand["dest"] for cand in candidates})
    max_minor = min(MAX_MINOR_PIXELS, unique_dests, (usable - 1) // 2)
    if max_minor < 1:
        return None

    target = random.randint(1, max_minor)
    chosen: list[dict] = []
    used_dests: set[tuple[int, int]] = set()

    if any(cand["tie"] for cand in candidates) and random.random() < 0.5:
        tie_pool = [cand for cand in candidates if cand["tie"]]
        random.shuffle(tie_pool)
        for cand in tie_pool:
            if cand["dest"] not in used_dests:
                chosen.append(cand)
                used_dests.add(cand["dest"])
                break

    shuffled = candidates[:]
    random.shuffle(shuffled)
    for cand in shuffled:
        if len(chosen) >= target:
            break
        if cand["dest"] in used_dests:
            continue
        chosen.append(cand)
        used_dests.add(cand["dest"])

    if len(chosen) != target:
        return None
    return chosen


def _paint_rectangle(grid: np.ndarray, box: tuple[int, int, int, int], major: int, minor: int) -> bool:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY

    candidates = _minority_candidates(box)
    minority = _choose_minority(candidates)
    if minority is None:
        return False

    minority_cells = {cand["cell"] for cand in minority}
    free_cells = [cand["cell"] for cand in candidates if cand["cell"] not in minority_cells]
    major_min = len(minority) + 1
    major_max = min(len(free_cells), major_min + MAX_EXTRA_MAJOR)
    if major_max < major_min:
        return False

    major_count = random.randint(major_min, major_max)
    for r, c in random.sample(free_cells, major_count):
        grid[r, c] = major
    for cand in minority:
        r, c = cand["cell"]
        grid[r, c] = minor

    return True


def _sample_boxes(height: int, width: int) -> list[tuple[int, int, int, int]] | None:
    count = random.randint(*RECT_COUNT_RANGE)
    size_choices = [
        (
            random.randint(*RECT_HEIGHT_RANGE),
            random.randint(*RECT_WIDTH_RANGE),
        )
        for _ in range(count)
    ]
    size_choices.sort(key=lambda hw: hw[0] * hw[1], reverse=True)

    boxes: list[tuple[int, int, int, int]] = []
    for rect_h, rect_w in size_choices:
        if rect_h + 2 > height or rect_w + 2 > width:
            return None

        placed = False
        for _ in range(PLACEMENT_RETRIES):
            r0 = random.randint(1, height - rect_h - 1)
            c0 = random.randint(1, width - rect_w - 1)
            box = (r0, c0, r0 + rect_h - 1, c0 + rect_w - 1)
            expanded = _expand_box(box, RECT_PAD, height, width)
            if any(_boxes_overlap(expanded, _expand_box(other, RECT_PAD, height, width)) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None

    boxes.sort()
    return boxes


def _gen_attempt() -> np.ndarray | None:
    height = random.randint(*GRID_HEIGHT_RANGE)
    width = random.randint(*GRID_WIDTH_RANGE)
    boxes = _sample_boxes(height, width)
    if boxes is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for box in boxes:
        major, minor = random.sample(COLOR_POOL, 2)
        if not _paint_rectangle(grid, box, int(major), int(minor)):
            return None

    _ = main(grid)
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GENERATION_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {GENERATION_RETRIES} attempts")
