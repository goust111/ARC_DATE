from __future__ import annotations

from collections import Counter
import random

import numpy as np


BG = 0
PALETTE = [1, 2, 3, 4, 5, 7]
MAX_GEN_ATTEMPTS = 500


def _get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    H, W = g.shape
    seen = np.zeros((H, W), dtype=bool)
    components: list[dict] = []

    for r in range(H):
        for c in range(W):
            color = int(g[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        if dr == 0 and dc == 0:
                            continue
                        nr = rr + dr
                        nc = cc + dc
                        if 0 <= nr < H and 0 <= nc < W and not seen[nr, nc] and int(g[nr, nc]) == color:
                            seen[nr, nc] = True
                            stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            components.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": int(coords.shape[0]),
                }
            )
    return components


def _is_hollow_frame(patch: np.ndarray, color: int, bg: int) -> bool:
    h, w = patch.shape
    if h < 3 or w < 3:
        return False
    if not np.all(patch[0, :] == color):
        return False
    if not np.all(patch[-1, :] == color):
        return False
    if not np.all(patch[:, 0] == color):
        return False
    if not np.all(patch[:, -1] == color):
        return False
    inner = patch[1:-1, 1:-1]
    return inner.size > 0 and bool(np.all(inner == bg))


def _transform_frame(patch: np.ndarray) -> np.ndarray:
    h, w = patch.shape
    if w > h:
        return np.concatenate([patch, np.fliplr(patch)], axis=1)
    if h > w:
        return np.concatenate([patch, np.flipud(patch)], axis=0)
    return patch.copy()


def _build_output(items: list[tuple[int, int, np.ndarray]], bg: int) -> np.ndarray:
    ordered = sorted(items, key=lambda item: (item[0], item[1]))
    out_h = max(int(patch.shape[0]) for _, _, patch in ordered)
    out_w = sum(int(patch.shape[1]) for _, _, patch in ordered)
    out = np.full((out_h, out_w), bg, dtype=int)
    c0 = 0
    for _, _, patch in ordered:
        h, w = patch.shape
        out[:h, c0 : c0 + w] = patch
        c0 += w
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Keep only hollow rectangular frames, mirror each by its own aspect ratio, then sort and concatenate them.

    Input structure:
    - The grid has a black background and several separated single-color 8-connected objects.
    - Some objects are one-pixel-thick hollow rectangular frames; the rest are solid square distractors.
    - Each object has its own color, and objects do not touch each other.

    Rule:
    - Find every monochromatic 8-connected object and crop it to its bounding box.
    - Discard solid objects; keep only crops that are hollow rectangular frames with background-only interiors.
    - For each kept frame: if width > height, append its left-right mirror on the right; if height > width, append its up-down mirror below; if square, keep it unchanged.
    - Count each transformed frame's non-background pixels, sort by that area ascending, break ties by the original bbox's left column, and place the sorted results side by side with top alignment.

    Key insight:
    - The mirror direction is chosen independently per frame from its own aspect ratio, so sorting can only be done after transforming each surviving frame.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    kept: list[tuple[int, int, np.ndarray]] = []

    for comp in _find_components(g, bg):
        patch = _crop_bbox(g, comp["bbox"])
        color = int(comp["color"])
        if not _is_hollow_frame(patch, color, bg):
            continue
        transformed = _transform_frame(patch)
        area = int(np.count_nonzero(transformed != bg))
        kept.append((area, int(comp["bbox"][1]), transformed))

    if not kept:
        return np.zeros((1, 1), dtype=int)
    return _build_output(kept, bg)


def _make_frame(h: int, w: int, color: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _make_square(side: int, color: int) -> np.ndarray:
    return np.full((side, side), color, dtype=int)


def _transformed_dims(h: int, w: int) -> tuple[int, int]:
    if w > h:
        return h, 2 * w
    if h > w:
        return 2 * h, w
    return h, w


def _sample_frame_specs(frame_count: int) -> list[dict] | None:
    types = ["wide", "tall"]
    while len(types) < frame_count:
        pick = random.choices(["square", "wide", "tall"], weights=[4, 3, 3], k=1)[0]
        types.append(pick)
    random.shuffle(types)

    specs: list[dict] = []
    for kind in types:
        if kind == "wide":
            h = random.randint(3, 5)
            w = random.randint(h + 1, min(7, h + 4))
        elif kind == "tall":
            w = random.randint(3, 5)
            h = random.randint(w + 1, min(7, w + 4))
        else:
            side = random.randint(3, 6)
            h = side
            w = side
        out_h, out_w = _transformed_dims(h, w)
        specs.append({"kind": kind, "h": h, "w": w, "out_h": out_h, "out_w": out_w})

    transformed_areas = []
    for spec in specs:
        border = 2 * (spec["h"] + spec["w"]) - 4
        factor = 2 if spec["kind"] != "square" else 1
        transformed_areas.append(border * factor)
    if len(set(transformed_areas)) < 2:
        return None

    total_out_w = sum(spec["out_w"] for spec in specs)
    max_out_h = max(spec["out_h"] for spec in specs)
    if total_out_w > 30 or max_out_h > 30:
        return None

    return specs


def _try_place_objects(grid: np.ndarray, objects: list[dict]) -> bool:
    H, W = grid.shape
    blocked = np.zeros((H, W), dtype=bool)

    for obj in sorted(objects, key=lambda item: item["patch"].shape[0] * item["patch"].shape[1], reverse=True):
        patch = obj["patch"]
        h, w = patch.shape
        candidates = [(r, c) for r in range(H - h + 1) for c in range(W - w + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            if blocked[r0 : r0 + h, c0 : c0 + w].any():
                continue
            grid[r0 : r0 + h, c0 : c0 + w] = np.maximum(grid[r0 : r0 + h, c0 : c0 + w], patch)
            rr0 = max(0, r0 - 1)
            cc0 = max(0, c0 - 1)
            rr1 = min(H, r0 + h + 1)
            cc1 = min(W, c0 + w + 1)
            blocked[rr0:rr1, cc0:cc1] = True
            obj["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
            placed = True
            break
        if not placed:
            return False
    return True


def _expected_output(frame_objects: list[dict]) -> np.ndarray:
    items: list[tuple[int, int, np.ndarray]] = []
    for obj in frame_objects:
        transformed = _transform_frame(obj["patch"])
        area = int(np.count_nonzero(transformed != BG))
        items.append((area, int(obj["bbox"][1]), transformed))
    return _build_output(items, BG)


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(22, 28)
    W = random.randint(22, 28)
    total_objects = random.randint(5, 6)
    frame_count = random.randint(3, 4)
    if frame_count > total_objects:
        frame_count = total_objects
    square_count = total_objects - frame_count
    colors = random.sample(PALETTE, total_objects)

    frame_specs = _sample_frame_specs(frame_count)
    if frame_specs is None:
        return None

    frame_objects: list[dict] = []
    for spec, color in zip(frame_specs, colors[:frame_count]):
        frame_objects.append(
            {
                "patch": _make_frame(spec["h"], spec["w"], color),
                "kind": spec["kind"],
                "color": color,
            }
        )

    square_objects: list[dict] = []
    for color in colors[frame_count:]:
        side = random.randint(2, 4)
        square_objects.append({"patch": _make_square(side, color), "color": color})

    grid = np.zeros((H, W), dtype=int)
    objects = frame_objects + square_objects
    if not _try_place_objects(grid, objects):
        return None

    if not any(obj["patch"].shape[1] > obj["patch"].shape[0] for obj in frame_objects):
        return None
    if not any(obj["patch"].shape[0] > obj["patch"].shape[1] for obj in frame_objects):
        return None

    expected = _expected_output(frame_objects)
    if expected.shape[0] > 30 or expected.shape[1] > 30:
        return None
    if not np.array_equal(main(grid), expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate valid input")
