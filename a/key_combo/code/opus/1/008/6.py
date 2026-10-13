from __future__ import annotations

import numpy as np


OCCLUDER = 5
FILL_BY_COUNT = {1: 1, 2: 2, 3: 3, 4: 4}
BASE_COLORS = (6, 7, 8, 9)
GEN_ATTEMPTS = 500


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
            coords: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_boxes(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    boxes: list[tuple[int, int, int, int]] = []
    for coords in _components(grid == OCCLUDER):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        expected_area = 2 * (h + w) - 4
        if h < 3 or w < 3 or len(coords) != expected_area:
            raise AssertionError("gray components must be single-pixel hollow rectangles")
        border = set(coords)
        for rr in range(r0, r1 + 1):
            if (rr, c0) not in border or (rr, c1) not in border:
                raise AssertionError("frame border is incomplete")
        for cc in range(c0, c1 + 1):
            if (r0, cc) not in border or (r1, cc) not in border:
                raise AssertionError("frame border is incomplete")
        boxes.append((r0, c0, r1, c1))
    boxes.sort()
    assert boxes, "expected at least one gray hollow frame"
    return boxes


def _infer_axis_period(grid: np.ndarray, axis: int) -> int:
    n = grid.shape[axis]
    for period in range(1, n):
        if axis == 0:
            left = grid[:-period, :]
            right = grid[period:, :]
        else:
            left = grid[:, :-period]
            right = grid[:, period:]
        mask = (left != OCCLUDER) & (right != OCCLUDER)
        if np.any(mask) and np.all(left[mask] == right[mask]):
            return period
    raise AssertionError("failed to infer translational period from visible cells")


def _infer_periods(grid: np.ndarray) -> tuple[int, int]:
    py = _infer_axis_period(grid, axis=0)
    px = _infer_axis_period(grid, axis=1)
    residue_colors: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(grid != OCCLUDER):
        key = (int(r) % py, int(c) % px)
        color = int(grid[r, c])
        prev = residue_colors.get(key)
        if prev is not None and prev != color:
            raise AssertionError("visible cells disagree within one symmetry orbit")
        residue_colors[key] = color
    for rr in range(py):
        for cc in range(px):
            if (rr, cc) not in residue_colors:
                raise AssertionError("every symmetry orbit must have a visible witness")
    return py, px


def _restore_pattern(grid: np.ndarray, py: int, px: int) -> np.ndarray:
    residue_colors: dict[tuple[int, int], int] = {}
    for r, c in np.argwhere(grid != OCCLUDER):
        residue_colors[(int(r) % py, int(c) % px)] = int(grid[r, c])
    restored = np.array(grid, copy=True)
    for r, c in np.argwhere(grid == OCCLUDER):
        restored[r, c] = residue_colors[(int(r) % py, int(c) % px)]
    return restored


def _solve(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    frames = _frame_boxes(g)
    py, px = _infer_periods(g)
    restored = _restore_pattern(g, py, px)

    out = g.copy()
    for r0, c0, r1, c1 in frames:
        count = len(set(map(int, restored[r0 : r1 + 1, c0 : c1 + 1].ravel())))
        if count not in FILL_BY_COUNT:
            raise AssertionError("restored color count must map to a fill color")
        out[r0 + 1 : r1, c0 + 1 : c1] = FILL_BY_COUNT[count]
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Recover the periodic pattern hidden under gray hollow frames, then fill each frame interior by the restored color count.

    Input structure:
    - The whole canvas is covered by one small multicolor translational pattern; there is no true background.
    - One to three gray (5) hollow rectangular frames overwrite only their own one-pixel borders.
    - The strict interior of each frame still shows the underlying pattern, while the border cells are occluded.

    Rule:
    - Treat gray cells as occluders and infer the smallest vertical and horizontal translation periods from all visible non-gray cells.
    - Restore every gray border cell from any visible cell in the same translation orbit.
    - For each frame, count the distinct restored colors enclosed by the frame, including the cells hidden under its gray border.
    - Recolor the strict interior uniformly by that count: 1 -> blue(1), 2 -> red(2), 3 -> green(3), 4 -> yellow(4).
    - Keep every gray border cell unchanged.

    Key insight:
    - The frame interiors are not filled from what is directly visible; the hidden border cells belong to the periodic pattern too, and they can change the count that determines the final fill color.
    """
    return _solve(grid)


def _axis_period_of_labels(labels: np.ndarray) -> int:
    n = len(labels)
    for period in range(1, n + 1):
        if all(int(labels[i]) == int(labels[i % period]) for i in range(n)):
            return period
    raise AssertionError("unreachable")


def _sample_tile(
    rng: np.random.Generator, py: int, px: int, palette: tuple[int, ...]
) -> np.ndarray | None:
    color_count = len(palette)
    for _ in range(200):
        row_labels = rng.integers(0, color_count, size=py)
        col_labels = rng.integers(0, color_count, size=px)
        if _axis_period_of_labels(row_labels) != py or _axis_period_of_labels(col_labels) != px:
            continue
        tile_ids = (row_labels[:, None] + col_labels[None, :]) % color_count
        if len(set(map(int, tile_ids.ravel()))) != color_count:
            continue
        return np.vectorize(lambda x: palette[int(x)], otypes=[int])(tile_ids)
    return None


def _candidate_frames(base: np.ndarray) -> list[dict[str, int | tuple[int, int, int, int]]]:
    h, w = base.shape
    candidates: list[dict[str, int | tuple[int, int, int, int]]] = []
    for fh in range(5, 10):
        for fw in range(5, 10):
            if fh > h or fw > w:
                continue
            for r0 in range(h - fh + 1):
                for c0 in range(w - fw + 1):
                    r1 = r0 + fh - 1
                    c1 = c0 + fw - 1
                    full_count = len(set(map(int, base[r0 : r1 + 1, c0 : c1 + 1].ravel())))
                    visible_count = len(set(map(int, base[r0 + 1 : r1, c0 + 1 : c1].ravel())))
                    if 1 <= full_count <= 4 and full_count > visible_count:
                        candidates.append(
                            {
                                "box": (r0, c0, r1, c1),
                                "count": full_count,
                                "visible_count": visible_count,
                                "height": fh,
                                "width": fw,
                            }
                        )
    return candidates


def _boxes_too_close(
    box_a: tuple[int, int, int, int], box_b: tuple[int, int, int, int], gap: int = 1
) -> bool:
    ar0, ac0, ar1, ac1 = box_a
    br0, bc0, br1, bc1 = box_b
    return not (ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0)


def _select_frames(
    rng: np.random.Generator, candidates: list[dict[str, int | tuple[int, int, int, int]]]
) -> list[dict[str, int | tuple[int, int, int, int]]] | None:
    by_count: dict[int, list[dict[str, int | tuple[int, int, int, int]]]] = {}
    for cand in candidates:
        by_count.setdefault(int(cand["count"]), []).append(cand)
    counts = sorted(by_count)
    if not counts:
        return None

    for target_n in rng.permutation(np.array([3, 2, 1], dtype=int)):
        distinct_needed = 1 if target_n == 1 else min(2, len(counts))
        for _ in range(400):
            base_counts = rng.choice(counts, size=distinct_needed, replace=False).tolist()
            plan = base_counts[:]
            while len(plan) < target_n:
                plan.append(int(rng.choice(counts)))
            rng.shuffle(plan)

            selected: list[dict[str, int | tuple[int, int, int, int]]] = []
            used_sizes: set[tuple[int, int]] = set()
            ok = True
            for count in plan:
                pool = by_count[count][:]
                rng.shuffle(pool)
                pick = None
                for cand in pool:
                    box = cand["box"]
                    assert isinstance(box, tuple)
                    if any(_boxes_too_close(box, prev["box"]) for prev in selected):
                        continue
                    size = (int(cand["height"]), int(cand["width"]))
                    if len(selected) < 2 and size in used_sizes and len(pool) > 1:
                        continue
                    pick = cand
                    break
                if pick is None:
                    ok = False
                    break
                selected.append(pick)
                used_sizes.add((int(pick["height"]), int(pick["width"])))
            if ok and len({int(c["count"]) for c in selected}) >= distinct_needed:
                return selected
    return None


def _apply_frames(base: np.ndarray, frames: list[dict[str, int | tuple[int, int, int, int]]]) -> np.ndarray:
    grid = np.array(base, copy=True)
    for frame in frames:
        r0, c0, r1, c1 = frame["box"]
        grid[r0, c0 : c1 + 1] = OCCLUDER
        grid[r1, c0 : c1 + 1] = OCCLUDER
        grid[r0 : r1 + 1, c0] = OCCLUDER
        grid[r0 : r1 + 1, c1] = OCCLUDER
    return grid


def _expected_output(
    occluded: np.ndarray, frames: list[dict[str, int | tuple[int, int, int, int]]]
) -> np.ndarray:
    out = np.array(occluded, copy=True)
    for frame in frames:
        r0, c0, r1, c1 = frame["box"]
        out[r0 + 1 : r1, c0 + 1 : c1] = FILL_BY_COUNT[int(frame["count"])]
    return out


def _validate_generated(
    base: np.ndarray,
    occluded: np.ndarray,
    frames: list[dict[str, int | tuple[int, int, int, int]]],
    py: int,
    px: int,
) -> bool:
    detected_boxes = _frame_boxes(occluded)
    expected_boxes = sorted(frame["box"] for frame in frames)
    if detected_boxes != expected_boxes:
        return False
    if _infer_periods(occluded) != (py, px):
        return False

    restored = _restore_pattern(occluded, py, px)
    shortcut_fails = False
    for frame in frames:
        r0, c0, r1, c1 = frame["box"]
        full_count = len(set(map(int, restored[r0 : r1 + 1, c0 : c1 + 1].ravel())))
        visible_count = len(set(map(int, base[r0 + 1 : r1, c0 + 1 : c1].ravel())))
        if full_count != int(frame["count"]) or visible_count != int(frame["visible_count"]):
            return False
        if full_count != visible_count:
            shortcut_fails = True

    if not shortcut_fails:
        return False

    expected = _expected_output(occluded, frames)
    if not np.array_equal(_solve(occluded), expected):
        return False
    return True


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(17, 22))
    w = int(rng.integers(17, 22))

    if rng.random() < 0.5:
        py = int(rng.choice([6, 7]))
        px = int(rng.choice([4, 5, 6]))
    else:
        py = int(rng.choice([4, 5, 6]))
        px = int(rng.choice([6, 7]))

    color_count = 3 if rng.random() < 0.45 else 4
    palette_arr = rng.choice(BASE_COLORS, size=color_count, replace=False)
    palette = tuple(int(x) for x in palette_arr)
    tile = _sample_tile(rng, py, px, palette)
    if tile is None:
        return None

    reps_r = h // py + 2
    reps_c = w // px + 2
    base = np.tile(tile, (reps_r, reps_c))[:h, :w]
    candidates = _candidate_frames(base)
    if len({int(c["count"]) for c in candidates}) < 2:
        return None

    frames = _select_frames(rng, candidates)
    if frames is None:
        return None

    occluded = _apply_frames(base, frames)
    if not _validate_generated(base, occluded, frames, py, px):
        return None
    return occluded


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
