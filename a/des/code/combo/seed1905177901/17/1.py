from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))


def _find_objects(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    h, w = g.shape
    seen = np.zeros((h, w), dtype=bool)
    objects: list[dict] = []

    for r in range(h):
        for c in range(w):
            color = int(g[r, c])
            if color == BG or seen[r, c]:
                continue

            seen[r, c] = True
            q = deque([(r, c)])
            area = 0
            r0 = r1 = r
            c0 = c1 = c

            while q:
                rr, cc = q.popleft()
                area += 1
                r0 = min(r0, rr)
                r1 = max(r1, rr)
                c0 = min(c0, cc)
                c1 = max(c1, cc)

                for dr, dc in CARDINALS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(g[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))

            objects.append(
                {
                    "color": color,
                    "area": area,
                    "bbox": (r0, c0, r1, c1),
                }
            )

    assert objects, "expected at least one foreground object"
    return objects


def _sort_key(obj: dict) -> tuple[int, int, int, int]:
    r0, c0, _r1, _c1 = obj.get("bbox", (0, 0, 0, 0))
    return (int(obj["area"]), int(r0), int(c0), int(obj["color"]))


def _render_output(objects: list[dict]) -> np.ndarray:
    wide = sorted([obj for obj in objects if obj["kind"] == "wide"], key=_sort_key)
    tall = sorted([obj for obj in objects if obj["kind"] == "tall"], key=_sort_key)

    wide_block_h = 0 if not wide else 2 * len(wide) - 1
    wide_block_w = max((int(obj["area"]) for obj in wide), default=0)
    tall_block_h = max((int(obj["area"]) for obj in tall), default=0)
    tall_block_w = 0 if not tall else 2 * len(tall) - 1
    block_gap = 1 if wide and tall else 0

    out_h = max(wide_block_h, tall_block_h)
    out_w = wide_block_w + block_gap + tall_block_w
    out = np.full((out_h, out_w), BG, dtype=int)

    for i, obj in enumerate(wide):
        row = 2 * i
        out[row, : int(obj["area"])] = int(obj["color"])

    col0 = wide_block_w + block_gap
    for i, obj in enumerate(tall):
        col = col0 + 2 * i
        out[: int(obj["area"]), col] = int(obj["color"])

    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Split foreground objects by bbox aspect ratio, flatten wide ones into rows,
    compress tall ones into columns, and pack them by increasing area.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    objects = []
    for obj in _find_objects(g):
        r0, c0, r1, c1 = obj["bbox"]
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h != w, "square bounding boxes are outside the task specification"
        objects.append(
            {
                "color": int(obj["color"]),
                "area": int(obj["area"]),
                "bbox": obj["bbox"],
                "kind": "wide" if w > h else "tall",
            }
        )

    return _render_output(objects)


# --------------- generator ---------------

COLORS = tuple(range(1, 10))
OBJECT_COUNTS = (2, 4, 6)
SHORT_SIDE_MIN = 1
SHORT_SIDE_MAX = 4
LONG_SIDE_MAX = 6
AREA_CAP = 16
GRID_LIMIT = 30
MAX_GENERATION_ATTEMPTS = 400


def _make_connected_mask(h: int, w: int, area: int) -> np.ndarray:
    assert h >= 1 and w >= 1 and h != w
    min_area = h + w - 1
    assert min_area <= area <= h * w

    mask = np.zeros((h, w), dtype=bool)
    r = c = 0
    mask[r, c] = True

    steps = ["D"] * (h - 1) + ["R"] * (w - 1)
    random.shuffle(steps)
    for step in steps:
        if step == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    while int(mask.sum()) < area:
        frontier = []
        for rr in range(h):
            for cc in range(w):
                if mask[rr, cc]:
                    continue
                for dr, dc in CARDINALS:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((rr, cc))
                        break
        assert frontier
        rr, cc = random.choice(frontier)
        mask[rr, cc] = True

    if random.random() < 0.5:
        mask = np.flipud(mask)
    if random.random() < 0.5:
        mask = np.fliplr(mask)
    return mask


def _sample_object(kind: str, color: int, used_areas: set[int]) -> dict | None:
    for _ in range(120):
        short = random.randint(SHORT_SIDE_MIN, SHORT_SIDE_MAX)
        long_min = short + 1
        if long_min > LONG_SIDE_MAX:
            continue
        long = random.randint(long_min, LONG_SIDE_MAX)
        h, w = (short, long) if kind == "wide" else (long, short)

        min_area = h + w - 1
        max_area = min(h * w, AREA_CAP)
        choices = [area for area in range(min_area, max_area + 1) if area not in used_areas]
        if not choices:
            continue

        roll = random.random()
        if roll < 0.25:
            area = choices[0]
        elif roll < 0.5:
            area = choices[-1]
        else:
            area = random.choice(choices)

        return {
            "color": int(color),
            "area": int(area),
            "kind": kind,
            "h": int(h),
            "w": int(w),
            "mask": _make_connected_mask(h, w, area),
        }
    return None


def _valid_positions(blocked: np.ndarray, h: int, w: int) -> list[tuple[int, int]]:
    height, width = blocked.shape
    positions = []
    for r in range(1, height - h):
        for c in range(1, width - w):
            if not blocked[r - 1 : r + h + 1, c - 1 : c + w + 1].any():
                positions.append((r, c))
    return positions


def _place_objects(specs: list[dict], height: int, width: int) -> np.ndarray | None:
    grid = np.full((height, width), BG, dtype=int)
    blocked = np.zeros((height, width), dtype=bool)
    blocked[0, :] = True
    blocked[-1, :] = True
    blocked[:, 0] = True
    blocked[:, -1] = True

    placed = sorted(specs, key=lambda obj: int(obj["h"]) * int(obj["w"]), reverse=True)
    for obj in placed:
        positions = _valid_positions(blocked, int(obj["h"]), int(obj["w"]))
        if not positions:
            return None
        r, c = random.choice(positions)
        patch = grid[r : r + int(obj["h"]), c : c + int(obj["w"])]
        patch[obj["mask"]] = int(obj["color"])
        blocked[r - 1 : r + int(obj["h"]) + 1, c - 1 : c + int(obj["w"]) + 1] = True

    return grid


def _sample_specs() -> list[dict] | None:
    n = random.choice(OBJECT_COUNTS)
    kinds = ["wide"] * (n // 2) + ["tall"] * (n // 2)
    random.shuffle(kinds)
    colors = random.sample(COLORS, n)

    specs: list[dict] = []
    used_areas: set[int] = set()
    for kind, color in zip(kinds, colors):
        obj = _sample_object(kind, color, used_areas)
        if obj is None:
            return None
        specs.append(obj)
        used_areas.add(int(obj["area"]))
    return specs


def _gen_attempt() -> np.ndarray | None:
    specs = _sample_specs()
    if specs is None:
        return None

    expected = _render_output(specs)
    if expected.shape[0] > GRID_LIMIT or expected.shape[1] > GRID_LIMIT:
        return None

    total_fg = sum(int(obj["area"]) for obj in specs)
    n = len(specs)
    min_side = max(10, 8 + 2 * n, max(int(obj["h"]) for obj in specs) + 2, max(int(obj["w"]) for obj in specs) + 2)
    max_side = min(GRID_LIMIT, min_side + 6)

    for _ in range(40):
        height = random.randint(min_side, max_side)
        width = random.randint(min_side, max_side)
        if total_fg * 2 >= height * width:
            continue
        grid = _place_objects(specs, height, width)
        if grid is None:
            continue
        if np.array_equal(main(grid), expected):
            return grid
    return None


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
