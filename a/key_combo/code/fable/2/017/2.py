import random
from collections import Counter

import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRS = ("U", "D", "L", "R")
DIR_VECS = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}

GRID_H_RANGE = (18, 30)
GRID_W_RANGE = (18, 30)
OBJECT_COUNT_CHOICES = (3, 4)
PAIR_SHAPES = (
    ((1, 4), (2, 2)),
    ((1, 6), (2, 3)),
    ((1, 8), (2, 4)),
    ((1, 9), (3, 3)),
    ((2, 6), (3, 4)),
    ((2, 8), (4, 4)),
)
RANDOM_INNER_SIZES = (
    (1, 1),
    (1, 2),
    (1, 3),
    (1, 5),
    (2, 2),
    (2, 4),
    (2, 5),
    (3, 2),
    (3, 4),
    (3, 5),
    (4, 3),
    (4, 4),
)
PLACE_RETRIES = 120
GEN_RETRIES = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.flatten())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray, connectivity: int = 4) -> list[np.ndarray]:
    if connectivity != 4:
        raise ValueError("only 4-connectivity is supported")
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(coords, dtype=int))
    return comps


def _border_mask(height: int, width: int) -> np.ndarray:
    mask = np.zeros((height, width), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _ray_cells(frame: dict, shape: tuple[int, int]) -> list[tuple[int, int]]:
    h, w = shape
    dr, dc = frame["dir"]
    r0, c0 = frame["gap"]
    coords = []
    for step in range(1, frame["area"] + 1):
        rr = r0 + dr * step
        cc = c0 + dc * step
        if not (0 <= rr < h and 0 <= cc < w):
            break
        coords.append((int(rr), int(cc)))
    return coords


def _analyze_frame(grid: np.ndarray, coords: np.ndarray, bg: int) -> dict:
    colors = {int(grid[r, c]) for r, c in coords}
    assert len(colors) == 1
    color = next(iter(colors))
    assert color != bg

    r0, c0, r1, c1 = bbox_coords(coords)
    height = r1 - r0 + 1
    width = c1 - c0 + 1
    assert height >= 3 and width >= 3

    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = _border_mask(height, width)
    border_cells = int(border.sum())
    assert coords.shape[0] == border_cells - 1
    assert np.all(patch[~border] == bg)

    missing = np.argwhere(border & (patch == bg))
    assert missing.shape == (1, 2)
    gr, gc = map(int, missing[0])
    assert 0 < gr < height - 1 or 0 < gc < width - 1
    patch_without_gap = patch.copy()
    patch_without_gap[gr, gc] = color
    assert np.all(patch_without_gap[border] == color)

    if gr == 0:
        direction = DIR_VECS["U"]
    elif gr == height - 1:
        direction = DIR_VECS["D"]
    elif gc == 0:
        direction = DIR_VECS["L"]
    else:
        assert gc == width - 1
        direction = DIR_VECS["R"]

    return {
        "color": int(color),
        "bbox": (int(r0), int(c0), int(r1), int(c1)),
        "gap": (int(r0 + gr), int(c0 + gc)),
        "dir": (int(direction[0]), int(direction[1])),
        "area": int((height - 2) * (width - 2)),
        "inner_shape": (int(height - 2), int(width - 2)),
    }


def _extract_frames(grid: np.ndarray) -> tuple[int, list[dict]]:
    bg = get_bg(grid)
    assert bg == BG
    comps = components_mask(grid != bg, connectivity=4)
    assert comps
    frames = [_analyze_frame(grid, comp, bg) for comp in comps]
    frames.sort(key=lambda d: (d["bbox"][0], d["bbox"][1], d["bbox"][2], d["bbox"][3]))
    return bg, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Leak each gapped hollow rectangle by shooting a grey ray whose length equals its sealed cavity area.

    Input structure:
    - The background is black.
    - Each foreground object is a monochrome one-cell-thick rectangular frame with exactly one missing border pixel.
    - The missing pixel lies on a non-corner border cell, and the enclosed interior is otherwise empty black space.

    Rule:
    - For every frame, recover its full bounding rectangle and compute the interior cavity area n.
    - Fill the whole rectangle footprint with the frame color, which both repairs the gap and solidifies the interior.
    - From the cell immediately outside the gap, extend a straight grey ray along the outward normal for n cells, clipping anything that would leave the grid.

    Key insight:
    - The gap determines both the repair location and the ejection direction, while the invisible closed cavity area determines the ray length.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg, frames = _extract_frames(g)

    out = g.copy()
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = frame["color"]

    for frame in frames:
        for r, c in _ray_cells(frame, g.shape):
            out[r, c] = GREY

    return out


def _rotate_shape(shape: tuple[int, int]) -> tuple[int, int]:
    return shape if random.random() < 0.5 else (shape[1], shape[0])


def _expand_bbox_cells(r0: int, c0: int, r1: int, c1: int, shape: tuple[int, int]) -> set[tuple[int, int]]:
    h, w = shape
    cells = set()
    for r in range(max(0, r0 - 1), min(h - 1, r1 + 1) + 1):
        for c in range(max(0, c0 - 1), min(w - 1, c1 + 1) + 1):
            cells.add((int(r), int(c)))
    return cells


def _make_descriptor(inner_h: int, inner_w: int, side: str, color: int, clipped: bool) -> dict:
    return {
        "inner_h": int(inner_h),
        "inner_w": int(inner_w),
        "outer_h": int(inner_h + 2),
        "outer_w": int(inner_w + 2),
        "side": side,
        "color": int(color),
        "area": int(inner_h * inner_w),
        "clipped": bool(clipped),
    }


def _visible_capacity(desc: dict, r0: int, c0: int, grid_h: int, grid_w: int) -> int:
    if desc["side"] == "U":
        return int(r0)
    if desc["side"] == "D":
        return int(grid_h - (r0 + desc["outer_h"]))
    if desc["side"] == "L":
        return int(c0)
    return int(grid_w - (c0 + desc["outer_w"]))


def _place_object(desc: dict, forbidden: set[tuple[int, int]], grid_h: int, grid_w: int) -> dict | None:
    for _ in range(PLACE_RETRIES):
        r0 = random.randint(0, grid_h - desc["outer_h"])
        c0 = random.randint(0, grid_w - desc["outer_w"])
        capacity = _visible_capacity(desc, r0, c0, grid_h, grid_w)
        if desc["clipped"]:
            if not (1 <= capacity < desc["area"]):
                continue
        else:
            if capacity < desc["area"]:
                continue

        if desc["side"] in ("U", "D"):
            gap = (r0 if desc["side"] == "U" else r0 + desc["outer_h"] - 1, c0 + random.randint(1, desc["outer_w"] - 2))
        else:
            gap = (r0 + random.randint(1, desc["outer_h"] - 2), c0 if desc["side"] == "L" else c0 + desc["outer_w"] - 1)

        frame = {
            "color": desc["color"],
            "bbox": (int(r0), int(c0), int(r0 + desc["outer_h"] - 1), int(c0 + desc["outer_w"] - 1)),
            "gap": (int(gap[0]), int(gap[1])),
            "dir": DIR_VECS[desc["side"]],
            "area": desc["area"],
            "inner_shape": (desc["inner_h"], desc["inner_w"]),
        }
        ray = set(_ray_cells(frame, (grid_h, grid_w)))
        if not ray:
            continue
        r1, c1 = frame["bbox"][2], frame["bbox"][3]
        reserved = _expand_bbox_cells(r0, c0, r1, c1, (grid_h, grid_w)) | ray
        if reserved & forbidden:
            continue
        placed = dict(desc)
        placed.update(frame)
        placed["reserved"] = reserved
        return placed
    return None


def _draw_input(grid_h: int, grid_w: int, placed: list[dict]) -> np.ndarray:
    grid = np.zeros((grid_h, grid_w), dtype=int)
    for desc in placed:
        r0, c0, r1, c1 = desc["bbox"]
        color = desc["color"]
        grid[r0, c0 : c1 + 1] = color
        grid[r1, c0 : c1 + 1] = color
        grid[r0 : r1 + 1, c0] = color
        grid[r0 : r1 + 1, c1] = color
        gr, gc = desc["gap"]
        grid[gr, gc] = BG
    return grid


def _has_same_area_diff_shape(frames: list[dict]) -> bool:
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            if a["area"] == b["area"] and a["inner_shape"] != b["inner_shape"]:
                return True
    return False


def _has_same_direction_diff_area(frames: list[dict]) -> bool:
    for i, a in enumerate(frames):
        for b in frames[i + 1 :]:
            if a["dir"] == b["dir"] and a["area"] != b["area"]:
                return True
    return False


def _validate_semantics(inp: np.ndarray) -> None:
    bg, frames = _extract_frames(inp)
    assert len(frames) >= 3
    assert _has_same_area_diff_shape(frames)
    assert _has_same_direction_diff_area(frames)
    assert len({frame["dir"] for frame in frames}) >= 2

    out = main(inp)
    assert out.shape == inp.shape
    solid = {}
    expected_grey = set()

    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                key = (r, c)
                assert key not in solid
                solid[key] = frame["color"]
        expected_grey.update(_ray_cells(frame, inp.shape))

    for (r, c), color in solid.items():
        assert out[r, c] == color

    grey_cells = set(map(tuple, np.argwhere(out == GREY)))
    assert grey_cells == expected_grey
    assert expected_grey.isdisjoint(solid)

    for r in range(inp.shape[0]):
        for c in range(inp.shape[1]):
            if (r, c) in solid or (r, c) in expected_grey:
                continue
            assert out[r, c] == bg


def _gen_attempt() -> np.ndarray | None:
    grid_h = random.randint(*GRID_H_RANGE)
    grid_w = random.randint(*GRID_W_RANGE)
    count = random.choice(OBJECT_COUNT_CHOICES)

    colors = random.sample(FRAME_COLORS, count)
    pair_a, pair_b = random.choice(PAIR_SHAPES)
    shape1 = _rotate_shape(pair_a)
    shape2 = _rotate_shape(pair_b)
    pair_area = shape1[0] * shape1[1]

    side1 = random.choice(DIRS)
    side2_choices = [side for side in DIRS if side != side1]
    side2 = random.choice(side2_choices)

    frames = [
        _make_descriptor(shape1[0], shape1[1], side1, colors[0], clipped=False),
        _make_descriptor(shape2[0], shape2[1], side2, colors[1], clipped=False),
    ]

    extra_shape = random.choice([shape for shape in RANDOM_INNER_SIZES if shape[0] * shape[1] != pair_area])
    frames.append(_make_descriptor(extra_shape[0], extra_shape[1], side1, colors[2], clipped=False))

    if count == 4:
        clip_shape_candidates = [shape for shape in RANDOM_INNER_SIZES if shape[0] * shape[1] >= 2]
        shape4 = random.choice(clip_shape_candidates)
        clipped = random.random() < 0.5
        frames.append(_make_descriptor(shape4[0], shape4[1], random.choice(DIRS), colors[3], clipped=clipped))

    forbidden: set[tuple[int, int]] = set()
    placed = []
    for frame in frames:
        placed_frame = _place_object(frame, forbidden, grid_h, grid_w)
        if placed_frame is None:
            return None
        placed.append(placed_frame)
        forbidden |= placed_frame["reserved"]

    grid = _draw_input(grid_h, grid_w, placed)
    try:
        _validate_semantics(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
