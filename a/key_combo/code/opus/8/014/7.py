from __future__ import annotations

from collections import Counter, deque
import random

import numpy as np


BG = 0
WALL = 5
PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _neighbors4(r: int, c: int):
    yield r - 1, c
    yield r + 1, c
    yield r, c - 1
    yield r, c + 1


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _bbox_w(bbox: tuple[int, int, int, int]) -> int:
    return int(bbox[3] - bbox[1] + 1)


def _mask_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for sr in range(h):
        for sc in range(w):
            if seen[sr, sc] or not mask[sr, sc]:
                continue
            q = deque([(sr, sc)])
            seen[sr, sc] = True
            cells = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for nr, nc in _neighbors4(r, c):
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and mask[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray) -> list[dict]:
    objects: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if v not in (BG, WALL)):
        for coords in _mask_components(grid == color):
            bbox = _bbox(coords)
            objects.append(
                {
                    "color": int(color),
                    "coords": coords,
                    "bbox": bbox,
                    "width": _bbox_w(bbox),
                }
            )
    objects.sort(key=lambda obj: obj["bbox"])
    return objects


def _extract_chambers(grid: np.ndarray) -> list[dict]:
    chambers: list[dict] = []
    for coords in _mask_components(grid == WALL):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 4 or c1 - c0 < 5:
            continue

        comp = {tuple(int(x) for x in pt) for pt in coords.tolist()}
        for c in range(c0, c1 + 1):
            assert (r0, c) in comp and (r1, c) in comp
        for r in range(r0, r1 + 1):
            assert (r, c1) in comp

        slit_rows = []
        for r in range(r0 + 1, r1):
            if (r, c0) in comp:
                continue
            slit_rows.append(r)
        assert slit_rows, "expected a left-wall slit"
        assert slit_rows == list(range(slit_rows[0], slit_rows[-1] + 1)), "slit must be contiguous"
        assert len(slit_rows) == 1, "generator uses a single-cell entry slit"
        for r in range(r0 + 1, r1):
            if r not in slit_rows:
                assert (r, c0) in comp

        interior = grid[r0 + 1 : r1, c0 + 1 : c1]
        assert interior.size > 0 and not np.any(interior == WALL), "interior must be wall-free"

        chambers.append(
            {
                "bbox": (r0, c0, r1, c1),
                "slit_rows": tuple(slit_rows),
                "interior_bbox": (r0 + 1, c0 + 1, r1 - 1, c1 - 1),
            }
        )

    chambers.sort(key=lambda chamber: chamber["bbox"])
    return chambers


def _match_chambers_and_objects(grid: np.ndarray) -> list[tuple[dict, dict]]:
    chambers = _extract_chambers(grid)
    objects = _extract_objects(grid)
    assert len(chambers) == len(objects) >= 1

    used: set[int] = set()
    matches: list[tuple[dict, dict]] = []
    for chamber in chambers:
        r0, c0, r1, c1 = chamber["bbox"]
        slit_rows = set(chamber["slit_rows"])
        candidates = []
        for idx, obj in enumerate(objects):
            br0, bc0, br1, bc1 = obj["bbox"]
            width = int(obj["width"])
            if bc1 != c0 - 1:
                continue
            if not (r0 + 1 <= br0 <= br1 <= r1 - 1):
                continue
            if not set(range(br0, br1 + 1)).issubset(slit_rows):
                continue
            if bc0 + width != c0:
                continue
            if bc1 + width >= c1:
                continue
            candidates.append(idx)
        assert len(candidates) == 1, "each chamber needs one unique entry object"
        idx = candidates[0]
        assert idx not in used
        used.add(idx)
        matches.append((chamber, objects[idx]))

    assert len(used) == len(objects)
    return matches


def _fill_black_component(grid: np.ndarray, start: tuple[int, int], color: int) -> None:
    h, w = grid.shape
    sr, sc = start
    assert 0 <= sr < h and 0 <= sc < w and int(grid[sr, sc]) == BG
    q = deque([(sr, sc)])
    grid[sr, sc] = int(color)
    while q:
        r, c = q.popleft()
        for nr, nc in _neighbors4(r, c):
            if 0 <= nr < h and 0 <= nc < w and int(grid[nr, nc]) == BG:
                grid[nr, nc] = int(color)
                q.append((nr, nc))


def main(grid: np.ndarray) -> np.ndarray:
    """Shift each outside object into its chamber by its own width, then flood-fill that chamber.

    Input structure:
    - The background is black, and separate grey wall components form rectangular chambers.
    - Each chamber has a one-column slit on its left wall, and a small monochrome 4-connected object sits immediately to the slit's left.
    - Every object's bounding-box width matches the rightward translation needed for it to enter the adjacent chamber.

    Rule:
    - Identify each colored object outside the walls and compute its bounding-box width.
    - Erase the object from its original cells and shift it horizontally right by exactly that width so it occupies the slit and the chamber interior.
    - From the moved object, find the black chamber cell just inside the chamber and 4-connected flood-fill that enclosed black region with the object's color.
    - Keep grey walls unchanged and leave all other exterior black cells unchanged.

    Key insight:
    - The horizontal shift is not arbitrary: moving by the object's own width places it exactly at the chamber entrance, and that moved marker is what triggers the chamber-only flood fill.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    matches = _match_chambers_and_objects(g)

    out = g.copy()
    for _chamber, obj in matches:
        coords = obj["coords"]
        out[coords[:, 0], coords[:, 1]] = BG

    for chamber, obj in matches:
        r0, c0, r1, c1 = chamber["bbox"]
        color = int(obj["color"])
        shift = int(obj["width"])
        dst = obj["coords"] + np.array([0, shift], dtype=int)

        assert np.all(dst[:, 0] >= 0) and np.all(dst[:, 0] < g.shape[0])
        assert np.all(dst[:, 1] >= 0) and np.all(dst[:, 1] < g.shape[1])
        assert not np.any(g[dst[:, 0], dst[:, 1]] == WALL)
        out[dst[:, 0], dst[:, 1]] = color

        seed = None
        for rr, cc in dst.tolist():
            for nr, nc in _neighbors4(int(rr), int(cc)):
                if r0 + 1 <= nr <= r1 - 1 and c0 + 1 <= nc <= c1 - 1 and int(out[nr, nc]) == BG:
                    seed = (nr, nc)
                    break
            if seed is not None:
                break
        assert seed is not None, "moved object must expose an interior black neighbor"
        _fill_black_component(out, seed, color)

    return out


def _render_layout(layout: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    out = np.full(shape, BG, dtype=int)

    for item in layout:
        r0, c0, r1, c1 = item["bbox"]
        color = int(item["color"])
        width = int(item["width"])
        row = int(item["entry_row"])

        inp[r0 : r1 + 1, c0 : c1 + 1] = BG
        out[r0 : r1 + 1, c0 : c1 + 1] = BG

        inp[r0, c0 : c1 + 1] = WALL
        inp[r1, c0 : c1 + 1] = WALL
        inp[r0 : r1 + 1, c0] = WALL
        inp[r0 : r1 + 1, c1] = WALL

        out[r0, c0 : c1 + 1] = WALL
        out[r1, c0 : c1 + 1] = WALL
        out[r0 : r1 + 1, c0] = WALL
        out[r0 : r1 + 1, c1] = WALL

        inp[row, c0] = BG
        out[row, c0] = BG

        inp[row, c0 - width : c0] = color

        out[r0 + 1 : r1, c0 + 1 : c1] = color
        out[row, c0] = color

    return inp, out


def _sample_layout() -> tuple[np.ndarray, np.ndarray] | None:
    chamber_count = random.randint(3, 4)
    widths = [random.randint(2, 4) for _ in range(chamber_count)]
    inner_heights = [random.randint(3, 5) for _ in range(chamber_count)]
    outer_heights = [h + 2 for h in inner_heights]
    gaps = [random.randint(1, 2) for _ in range(chamber_count - 1)]
    top = random.randint(1, 2)
    bottom = random.randint(1, 2)
    total_h = top + sum(outer_heights) + sum(gaps) + bottom
    if total_h > 30:
        return None

    base_left = max(widths) + random.randint(2, 4)
    c0_values = []
    for width in widths:
        lo = max(width + 1, base_left - 1)
        hi = base_left + 1
        c0_values.append(random.randint(lo, hi))

    colors = random.sample(PALETTE, chamber_count)
    row = top
    layout = []
    right_edge = 0
    for idx in range(chamber_count):
        width = widths[idx]
        inner_h = inner_heights[idx]
        outer_h = outer_heights[idx]
        inner_w = width + 2
        outer_w = inner_w + 2
        r0 = row
        r1 = r0 + outer_h - 1
        c0 = c0_values[idx]
        c1 = c0 + outer_w - 1
        right_edge = max(right_edge, c1)

        entry_row = r0 + 1 + random.randint(0, inner_h - 1)
        layout.append(
            {
                "bbox": (r0, c0, r1, c1),
                "width": width,
                "entry_row": entry_row,
                "color": colors[idx],
            }
        )
        row = r1 + 1 + (gaps[idx] if idx < len(gaps) else 0)

    total_w = right_edge + 1 + random.randint(1, 3)
    if total_w > 30:
        return None

    return _render_layout(layout, (total_h, total_w))


def _semantic_checks(inp: np.ndarray, expected: np.ndarray) -> bool:
    if inp.ndim != 2 or expected.ndim != 2 or max(inp.shape) > 30 or max(expected.shape) > 30:
        return False

    if not np.array_equal(main(inp), expected):
        return False

    matches = _match_chambers_and_objects(inp)
    if len(matches) not in (3, 4):
        return False

    for chamber, obj in matches:
        r0, c0, r1, c1 = chamber["bbox"]
        width = int(obj["width"])
        row = int(obj["bbox"][0])
        if obj["bbox"][0] != obj["bbox"][2]:
            return False
        if width != int(obj["coords"].shape[0]):
            return False
        if width not in (2, 3, 4):
            return False
        if row not in chamber["slit_rows"]:
            return False
        if c1 - c0 - 1 != width + 2:
            return False
        shifted = obj["coords"] + np.array([0, width], dtype=int)
        if shifted[:, 1].min() != c0 or shifted[:, 1].max() >= c1:
            return False
        if int(expected[row, c0]) != int(obj["color"]):
            return False
        if not np.all(expected[r0 + 1 : r1, c0 + 1 : c1] == int(obj["color"])):
            return False

    counts = Counter(map(int, inp.ravel()))
    if counts[WALL] <= 0 or counts[BG] <= 0:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        sample = _sample_layout()
        if sample is None:
            continue
        inp, expected = sample
        if _semantic_checks(inp, expected):
            return inp
    raise RuntimeError("failed to generate a valid input")
