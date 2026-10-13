from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import itertools

import numpy as np


DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
BG = 3  # green
FG_COLORS = [1, 2, 4, 5, 6, 7, 8, 9]

GRID_H_RANGE = (15, 23)  # inclusive lower, exclusive upper for rng.integers
GRID_W_RANGE = (15, 23)
OBJECT_COUNT_RANGE = (2, 6)
PLACEMENT_GAP = 1
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Component:
    color: int
    coords: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class PlacedObject:
    patch: np.ndarray
    orientation: str
    top: int
    left: int


def _dominant_color(grid: np.ndarray) -> int:
    flat = np.asarray(grid, dtype=int).ravel()
    return int(np.bincount(flat, minlength=10).argmax())


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[Component]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[Component] = []

    for r in range(h):
        for c in range(w):
            if seen[r, c] or int(g[r, c]) == bg:
                continue

            color = int(g[r, c])
            q: deque[tuple[int, int]] = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []

            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIR4:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
                        continue
                    if seen[nr, nc] or int(g[nr, nc]) != color:
                        continue
                    seen[nr, nc] = True
                    q.append((nr, nc))

            comps.append(Component(color=color, coords=tuple(coords), bbox=_bbox_from_coords(coords)))

    return comps


def _best_bg_run(line: np.ndarray, bg: int) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for idx, value in enumerate(line):
        if int(value) == bg:
            if start is None:
                start = idx
        elif start is not None:
            run = (start, idx)
            if best is None or (run[1] - run[0]) > (best[1] - best[0]):
                best = run
            start = None
    if start is not None:
        run = (start, len(line))
        if best is None or (run[1] - run[0]) > (best[1] - best[0]):
            best = run
    return best


def _detect_slit(patch: np.ndarray, bg: int, color: int) -> tuple[str, int]:
    h, w = patch.shape
    best_score = -1.0
    best: tuple[str, int] | None = None

    for r in range(1, h - 1):
        run = _best_bg_run(patch[r, :], bg)
        if run is None:
            continue
        c0, c1 = run
        run_len = c1 - c0
        if c0 == 0 or c1 == w:
            continue
        if run_len < max(2, (w - 2) // 2):
            continue
        if int(patch[r, c0 - 1]) != color or int(patch[r, c1]) != color:
            continue
        if not np.any(patch[r - 1, c0:c1] == color):
            continue
        if not np.any(patch[r + 1, c0:c1] == color):
            continue
        score = run_len / max(1, w - 2)
        if score > best_score:
            best_score = score
            best = ("h", r)

    for c in range(1, w - 1):
        run = _best_bg_run(patch[:, c], bg)
        if run is None:
            continue
        r0, r1 = run
        run_len = r1 - r0
        if r0 == 0 or r1 == h:
            continue
        if run_len < max(2, (h - 2) // 2):
            continue
        if int(patch[r0 - 1, c]) != color or int(patch[r1, c]) != color:
            continue
        if not np.any(patch[r0:r1, c - 1] == color):
            continue
        if not np.any(patch[r0:r1, c + 1] == color):
            continue
        score = run_len / max(1, h - 2)
        if score > best_score:
            best_score = score
            best = ("v", c)

    assert best is not None, "expected one internal slit per object"
    return best


def _paste(dest: np.ndarray, patch: np.ndarray, top: int, left: int, transparent: int) -> None:
    h, w = patch.shape
    region = dest[top : top + h, left : left + w]
    mask = patch != transparent
    region[mask] = patch[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split each slit-marked foreground object into two halves and move them apart.

    Input structure:
    - The background is green.
    - Each foreground object is a monochrome 4-connected component.
    - Every object's bbox is close to rectangular but contains one internal
      one-cell-wide background slit, either horizontal or vertical.

    Rule:
    - Detect the slit orientation inside each object.
    - If the slit is horizontal, move the upper part up by one row and the
      lower part down by one row.
    - If the slit is vertical, move the left part left by one column and the
      right part right by one column.
    - Clear the original object footprint and redraw the moved halves in the
      same color.
    """
    g = np.asarray(grid, dtype=int)
    bg = _dominant_color(g)
    out = np.full_like(g, bg)

    for comp in _find_components(g, bg):
        r0, c0, r1, c1 = comp.bbox
        patch = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
        for r, c in comp.coords:
            patch[r - r0, c - c0] = comp.color

        orientation, slit_idx = _detect_slit(patch, bg, comp.color)
        if orientation == "h":
            upper = patch[: slit_idx + 1, :]
            lower = patch[slit_idx + 1 :, :]
            _paste(out, upper, r0 - 1, c0, bg)
            _paste(out, lower, r0 + slit_idx + 2, c0, bg)
        else:
            left_half = patch[:, : slit_idx + 1]
            right_half = patch[:, slit_idx + 1 :]
            _paste(out, left_half, r0, c0 - 1, bg)
            _paste(out, right_half, r0, c0 + slit_idx + 2, bg)

    return out


def _sample_balanced_orientations(rng: np.random.Generator, count: int) -> list[str]:
    if count == 1:
        return [str(rng.choice(["h", "v"]))]
    n_h = count // 2 + int(count % 2 and rng.integers(0, 2))
    n_v = count - n_h
    items = ["h"] * n_h + ["v"] * n_v
    rng.shuffle(items)
    return items


def _horizontal_patch(rng: np.random.Generator, color: int) -> np.ndarray:
    width = int(rng.integers(5, 11))
    top_h = int(rng.integers(2, 5))
    delta = int(rng.choice(np.array([-1, 0, 1])))
    bottom_h = min(4, max(2, top_h + delta))
    height = top_h + 1 + bottom_h

    patch = np.full((height, width), color, dtype=int)
    patch[top_h, 1:-1] = BG
    return patch


def _vertical_patch(rng: np.random.Generator, color: int) -> np.ndarray:
    height = int(rng.integers(5, 11))
    left_w = int(rng.integers(2, 5))
    delta = int(rng.choice(np.array([-1, 0, 1])))
    right_w = min(4, max(2, left_w + delta))
    width = left_w + 1 + right_w

    patch = np.full((height, width), color, dtype=int)
    patch[1:-1, left_w] = BG
    return patch


def _movement_union_bbox(obj: PlacedObject) -> tuple[int, int, int, int]:
    h, w = obj.patch.shape
    if obj.orientation == "h":
        return obj.top - 1, obj.left, obj.top + h, obj.left + w - 1
    return obj.top, obj.left - 1, obj.top + h - 1, obj.left + w


def _expand_bbox(box: tuple[int, int, int, int], gap: int) -> tuple[int, int, int, int]:
    return box[0] - gap, box[1] - gap, box[2] + gap, box[3] + gap


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _candidate_positions(h: int, w: int, patch: np.ndarray, orientation: str) -> list[tuple[int, int]]:
    ph, pw = patch.shape
    if orientation == "h":
        row_range = range(1, h - ph)
        col_range = range(1, w - pw)
    else:
        row_range = range(1, h - ph)
        col_range = range(1, w - pw)
    return list(itertools.product(row_range, col_range))


def _place_objects(
    rng: np.random.Generator,
    h: int,
    w: int,
    patches: list[tuple[np.ndarray, str]],
) -> list[PlacedObject] | None:
    order = sorted(range(len(patches)), key=lambda idx: patches[idx][0].size, reverse=True)
    placed: dict[int, PlacedObject] = {}
    reserved: list[tuple[int, int, int, int]] = []

    for idx in order:
        patch, orientation = patches[idx]
        candidates = _candidate_positions(h, w, patch, orientation)
        rng.shuffle(candidates)

        chosen: PlacedObject | None = None
        for top, left in candidates:
            obj = PlacedObject(patch=patch, orientation=orientation, top=top, left=left)
            union_box = _movement_union_bbox(obj)
            if union_box[0] < 0 or union_box[1] < 0 or union_box[2] >= h or union_box[3] >= w:
                continue
            safe_box = _expand_bbox(union_box, PLACEMENT_GAP)
            if any(_boxes_intersect(safe_box, other) for other in reserved):
                continue
            chosen = obj
            reserved.append(safe_box)
            break

        if chosen is None:
            return None
        placed[idx] = chosen

    return [placed[i] for i in range(len(patches))]


def _render_input(h: int, w: int, objects: list[PlacedObject]) -> np.ndarray:
    grid = np.full((h, w), BG, dtype=int)
    for obj in objects:
        _paste(grid, obj.patch, obj.top, obj.left, BG)
    return grid


def _valid_background_dominance(grid: np.ndarray) -> bool:
    counts = np.bincount(grid.ravel(), minlength=10)
    bg_count = int(counts[BG])
    return bg_count > int(counts[[c for c in range(10) if c != BG]].max(initial=0))


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_H_RANGE))
    w = int(rng.integers(*GRID_W_RANGE))
    object_count = int(rng.integers(*OBJECT_COUNT_RANGE))
    orientations = _sample_balanced_orientations(rng, object_count)
    colors = [int(c) for c in rng.choice(np.array(FG_COLORS), size=object_count, replace=False)]

    patches: list[tuple[np.ndarray, str]] = []
    for orientation, color in zip(orientations, colors):
        patch = _horizontal_patch(rng, color) if orientation == "h" else _vertical_patch(rng, color)
        patches.append((patch, orientation))

    placed = _place_objects(rng, h, w, patches)
    if placed is None:
        return None

    inp = _render_input(h, w, placed)
    if not _valid_background_dominance(inp):
        return None

    try:
        out = main(inp)
    except AssertionError:
        return None

    if out.shape != inp.shape:
        return None
    if np.array_equal(inp, out):
        return None
    return inp


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        rng = np.random.default_rng()
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GENERATION_ATTEMPTS} attempts")
