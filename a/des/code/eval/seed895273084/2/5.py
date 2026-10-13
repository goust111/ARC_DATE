from __future__ import annotations

import random

import numpy as np


AZURE = 8
GREY = 5
PALETTE = [0, 1, 2, 3, 4, 6, 7, 9]
SIDES = ("top", "right", "bottom", "left")
GEN_RETRIES = 400
PLACEMENT_RETRIES = 200
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while stack:
                cr, cc = stack.pop()
                coords.append((cr, cc))
                for dr, dc in DIRS4:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _border_mask(h: int, w: int) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _missing_side(r: int, c: int, h: int, w: int) -> str | None:
    sides = []
    if r == 0:
        sides.append("top")
    if r == h - 1:
        sides.append("bottom")
    if c == 0:
        sides.append("left")
    if c == w - 1:
        sides.append("right")
    if len(sides) != 1:
        return None
    return sides[0]


def _rotate_for_side(pattern: np.ndarray, side: str) -> np.ndarray:
    ccw_turns = {"top": 0, "right": 3, "bottom": 2, "left": 1}[side]
    return np.rot90(pattern, ccw_turns)


def _extract_objects(grid: np.ndarray) -> tuple[dict[int, np.ndarray], list[dict[str, object]]]:
    grey_components = _components(grid == GREY)
    stamps: dict[int, np.ndarray] = {}
    papers: list[dict[str, object]] = []

    for comp in grey_components:
        r0, c0, r1, c1 = _bbox(comp)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 3 and w >= 3

        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        grey_mask = patch == GREY
        border = _border_mask(h, w)
        assert not np.any(grey_mask & ~border)

        missing = np.argwhere(border & ~grey_mask)
        assert len(missing) in (0, 1)

        if len(missing) == 0:
            mid_c = c0 + w // 2
            assert r0 > 0
            label = int(grid[r0 - 1, mid_c])
            assert label not in (AZURE, GREY)
            pattern = patch[1:-1, 1:-1].copy()
            assert np.any(pattern != AZURE)
            assert label not in stamps
            stamps[label] = pattern
            continue

        gap_r, gap_c = map(int, missing[0])
        side = _missing_side(gap_r, gap_c, h, w)
        assert side is not None
        if side == "top":
            mr, mc = r0 - 1, c0 + gap_c
        elif side == "bottom":
            mr, mc = r1 + 1, c0 + gap_c
        elif side == "left":
            mr, mc = r0 + gap_r, c0 - 1
        else:
            mr, mc = r0 + gap_r, c1 + 1
        assert 0 <= mr < grid.shape[0] and 0 <= mc < grid.shape[1]
        mark = int(grid[mr, mc])
        assert mark not in (AZURE, GREY)
        assert np.all(patch[1:-1, 1:-1] == AZURE)
        papers.append({"bbox": (r0, c0, r1, c1), "side": side, "mark": mark})

    assert stamps and papers
    return stamps, papers


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair stamps to blank papers by matching label colors, then rotate the stamp
    according to the paper's notch side and print the rotated pattern centered
    inside the paper.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    stamps, papers = _extract_objects(g)

    for paper in papers:
        mark = int(paper["mark"])
        assert mark in stamps
        pattern = _rotate_for_side(stamps[mark], str(paper["side"]))
        r0, c0, r1, c1 = paper["bbox"]
        inner_h = r1 - r0 - 1
        inner_w = c1 - c0 - 1
        ph, pw = pattern.shape
        assert ph <= inner_h and pw <= inner_w
        top = r0 + 1 + (inner_h - ph) // 2
        left = c0 + 1 + (inner_w - pw) // 2
        view = out[top : top + ph, left : left + pw]
        mask = pattern != AZURE
        view[mask] = pattern[mask]

    return out


def _random_asymmetric_mask(h: int, w: int, rng: random.Random) -> np.ndarray:
    min_cells = max(3, (h * w) // 4)
    max_cells = max(min_cells, min(h * w - 1, (2 * h * w) // 3))
    for _ in range(200):
        target = rng.randint(min_cells, max_cells)
        start = (rng.randrange(h), rng.randrange(w))
        cells = {start}
        frontier = [start]
        while len(cells) < target:
            if frontier and rng.random() < 0.85:
                base_r, base_c = rng.choice(frontier)
            else:
                base_r, base_c = rng.choice(tuple(cells))
            neighbors = []
            for dr, dc in DIRS4:
                nr, nc = base_r + dr, base_c + dc
                if 0 <= nr < h and 0 <= nc < w and (nr, nc) not in cells:
                    neighbors.append((nr, nc))
            if not neighbors:
                frontier = [pt for pt in frontier if any(
                    0 <= pt[0] + dr < h and 0 <= pt[1] + dc < w and (pt[0] + dr, pt[1] + dc) not in cells
                    for dr, dc in DIRS4
                )]
                if not frontier:
                    break
                continue
            nxt = rng.choice(neighbors)
            cells.add(nxt)
            frontier.append(nxt)

        if len(cells) < min_cells:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        if np.all(mask) or not np.any(mask):
            continue
        rows, cols = np.argwhere(mask).T
        if rows.min() == rows.max() or cols.min() == cols.max():
            continue
        if any(np.array_equal(mask, np.rot90(mask, turns)) for turns in (1, 2, 3)):
            continue
        return mask
    raise RuntimeError("failed to create asymmetric stamp mask")


def _draw_frame(arr: np.ndarray, top: int, left: int, h: int, w: int) -> None:
    arr[top, left : left + w] = GREY
    arr[top + h - 1, left : left + w] = GREY
    arr[top : top + h, left] = GREY
    arr[top : top + h, left + w - 1] = GREY


def _make_stamp(inner_h: int, inner_w: int, label_color: int, pattern_color: int, mask: np.ndarray) -> np.ndarray:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    patch = np.full((outer_h + 1, outer_w), AZURE, dtype=int)
    _draw_frame(patch, 1, 0, outer_h, outer_w)
    patch[0, outer_w // 2] = label_color
    interior = patch[2 : 2 + inner_h, 1 : 1 + inner_w]
    interior[mask] = pattern_color
    return patch


def _make_paper(inner_h: int, inner_w: int, label_color: int, side: str) -> np.ndarray:
    outer_h = inner_h + 2
    outer_w = inner_w + 2
    extra_top = 1 if side == "top" else 0
    extra_bottom = 1 if side == "bottom" else 0
    extra_left = 1 if side == "left" else 0
    extra_right = 1 if side == "right" else 0
    patch = np.full((outer_h + extra_top + extra_bottom, outer_w + extra_left + extra_right), AZURE, dtype=int)
    frame_top = extra_top
    frame_left = extra_left
    _draw_frame(patch, frame_top, frame_left, outer_h, outer_w)

    mid_r = outer_h // 2
    mid_c = outer_w // 2
    if side == "top":
        patch[frame_top, frame_left + mid_c] = AZURE
        patch[frame_top - 1, frame_left + mid_c] = label_color
    elif side == "right":
        patch[frame_top + mid_r, frame_left + outer_w - 1] = AZURE
        patch[frame_top + mid_r, frame_left + outer_w] = label_color
    elif side == "bottom":
        patch[frame_top + outer_h - 1, frame_left + mid_c] = AZURE
        patch[frame_top + outer_h, frame_left + mid_c] = label_color
    else:
        patch[frame_top + mid_r, frame_left] = AZURE
        patch[frame_top + mid_r, frame_left - 1] = label_color
    return patch


def _can_place(occupied: np.ndarray, top: int, left: int, h: int, w: int) -> bool:
    r0 = max(0, top - 1)
    c0 = max(0, left - 1)
    r1 = min(occupied.shape[0], top + h + 1)
    c1 = min(occupied.shape[1], left + w + 1)
    return not np.any(occupied[r0:r1, c0:c1])


def _place_objects(canvas_h: int, canvas_w: int, objects: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    grid = np.full((canvas_h, canvas_w), AZURE, dtype=int)
    occupied = np.zeros((canvas_h, canvas_w), dtype=bool)
    order = sorted(range(len(objects)), key=lambda i: objects[i].shape[0] * objects[i].shape[1], reverse=True)

    for idx in order:
        patch = objects[idx]
        h, w = patch.shape
        placed = False
        for _ in range(PLACEMENT_RETRIES):
            top = rng.randint(0, canvas_h - h)
            left = rng.randint(0, canvas_w - w)
            if not _can_place(occupied, top, left, h, w):
                continue
            grid[top : top + h, left : left + w] = patch
            occupied[top : top + h, left : left + w] = True
            placed = True
            break
        if not placed:
            return None
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 3)
    label_colors = rng.sample(PALETTE, pair_count)
    sides = rng.sample(list(SIDES), pair_count)
    objects: list[np.ndarray] = []

    for idx in range(pair_count):
        label_color = label_colors[idx]
        pattern_choices = [c for c in PALETTE if c != label_color]
        pattern_color = rng.choice(pattern_choices)
        inner_h = rng.choice([3, 5, 7])
        inner_w = rng.choice([3, 5, 7])
        mask = _random_asymmetric_mask(inner_h, inner_w, rng)
        stamp = _make_stamp(inner_h, inner_w, label_color, pattern_color, mask)

        max_dim = max(inner_h, inner_w)
        paper_inner_h = max_dim + 2 * rng.randint(1, 3)
        paper_inner_w = max_dim + 2 * rng.randint(1, 3)
        paper = _make_paper(paper_inner_h, paper_inner_w, label_color, sides[idx])

        objects.append(stamp)
        objects.append(paper)

    total_area = sum((obj.shape[0] + 2) * (obj.shape[1] + 2) for obj in objects)
    min_side = max(
        max(obj.shape[0] for obj in objects) + 1,
        max(obj.shape[1] for obj in objects) + 1,
        int(np.ceil(np.sqrt(total_area))) + 1,
    )
    if min_side > 30:
        return None

    for _ in range(50):
        canvas_h = rng.randint(min_side, 30)
        canvas_w = rng.randint(min_side, 30)
        placed = _place_objects(canvas_h, canvas_w, objects, rng)
        if placed is None:
            continue
        try:
            out = main(placed)
        except AssertionError:
            continue
        if np.array_equal(out, placed):
            continue
        return placed
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {GEN_RETRIES} attempts")
