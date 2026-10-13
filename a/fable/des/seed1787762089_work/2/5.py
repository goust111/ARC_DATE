from __future__ import annotations

import numpy as np


PALETTE = (1, 2, 3, 4, 6, 7, 8, 9)
SIZE_OPTIONS = (2, 3, 4)
GRID_MIN = 16
GRID_MAX = 22
PLACEMENT_GAP = 1
GEN_ATTEMPTS = 500

_RNG = np.random.default_rng()


def _most_common_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[int(np.argmax(counts))])


def _square_border_mask(side: int) -> np.ndarray:
    mask = np.zeros((side, side), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    return mask


def _classify_objects(grid: np.ndarray) -> tuple[int, dict[int, dict], list[dict]]:
    g = np.asarray(grid, dtype=int)
    bg = _most_common_color(g)
    frames: dict[int, dict] = {}
    blocks: list[dict] = []

    for color in np.unique(g):
        color = int(color)
        if color == bg:
            continue

        coords = np.argwhere(g == color)
        assert coords.size > 0, "foreground color must appear"
        r0, c0 = coords.min(axis=0)
        r1, c1 = coords.max(axis=0)
        r0, c0, r1, c1 = map(int, (r0, c0, r1, c1))
        h = int(r1 - r0 + 1)
        w = int(c1 - c0 + 1)
        assert h == w, "each object must be square"

        sub = g[r0 : r1 + 1, c0 : c1 + 1]
        mask = sub == color
        side = int(h)

        if bool(mask.all()):
            blocks.append({"color": color, "side": side, "bbox": (r0, c0, r1, c1)})
            continue

        border = _square_border_mask(side)
        assert side >= 4, "frame outer side must leave an interior"
        assert np.array_equal(mask, border), "foreground objects must be solid squares or hollow frames"
        assert bool((sub[1:-1, 1:-1] == bg).all()), "frame cavity must be background"
        inner = int(side - 2)
        assert inner not in frames, "frame inner sizes must be unique"
        frames[inner] = {"color": color, "bbox": (r0, c0, r1, c1)}

    return bg, frames, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Embed each solid square into the hollow frame whose cavity side matches it,
    then swap the colors of the block and the frame.
    """
    g = np.asarray(grid, dtype=int)
    bg, frames, blocks = _classify_objects(g)

    block_sizes = [int(block["side"]) for block in blocks]
    assert 2 <= len(blocks) <= 3, "expected 2 or 3 solid blocks"
    assert len(block_sizes) == len(set(block_sizes)), "block sizes must be unique"
    assert set(block_sizes) == set(frames.keys()), "each block must match one frame by size"

    out = g.copy()

    for block in blocks:
        r0, c0, r1, c1 = map(int, block["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = bg

    for block in blocks:
        side = int(block["side"])
        frame = frames[side]
        r0, c0, r1, c1 = map(int, frame["bbox"])
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[1:-1, 1:-1] = int(frame["color"])
        patch[0, :] = int(block["color"])
        patch[-1, :] = int(block["color"])
        patch[:, 0] = int(block["color"])
        patch[:, -1] = int(block["color"])

    return out


def _randint(lo: int, hi: int) -> int:
    return int(_RNG.integers(int(lo), int(hi) + 1))


def _choice(seq):
    return seq[_randint(0, len(seq) - 1)]


def _sample(seq, k: int) -> list[int]:
    idx = _RNG.choice(len(seq), size=int(k), replace=False)
    return [int(seq[int(i)]) for i in idx]


def _shuffled(seq) -> list:
    vals = list(seq)
    _RNG.shuffle(vals)
    return vals


def _boxes_are_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int) -> bool:
    ar0, ac0, ar1, ac1 = map(int, a)
    br0, bc0, br1, bc1 = map(int, b)
    return (
        ar1 + gap < br0
        or br1 + gap < ar0
        or ac1 + gap < bc0
        or bc1 + gap < ac0
    )


def _place_objects(height: int, width: int, specs: list[dict]) -> list[dict] | None:
    placed_boxes: list[tuple[int, int, int, int]] = []
    placed_specs: list[dict] = []

    order = _shuffled(specs)
    order.sort(key=lambda item: int(item["span"]), reverse=True)

    for spec in order:
        span = int(spec["span"])
        candidates: list[tuple[int, int]] = []
        for r in range(1, int(height - span)):
            for c in range(1, int(width - span)):
                bbox = (r, c, r + span - 1, c + span - 1)
                if all(_boxes_are_separated(bbox, other, PLACEMENT_GAP) for other in placed_boxes):
                    candidates.append((int(r), int(c)))

        if not candidates:
            return None

        top, left = candidates[_randint(0, len(candidates) - 1)]
        bbox = (top, left, top + span - 1, left + span - 1)
        placed_boxes.append(bbox)
        placed_specs.append({**spec, "bbox": bbox})

    return placed_specs


def _draw_frame(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0, c0 : c1 + 1] = int(color)
    grid[r1, c0 : c1 + 1] = int(color)
    grid[r0 : r1 + 1, c0] = int(color)
    grid[r0 : r1 + 1, c1] = int(color)


def _draw_block(grid: np.ndarray, bbox: tuple[int, int, int, int], color: int) -> None:
    r0, c0, r1, c1 = map(int, bbox)
    grid[r0 : r1 + 1, c0 : c1 + 1] = int(color)


def _expected_output(grid: np.ndarray, placed_specs: list[dict]) -> np.ndarray:
    bg = _most_common_color(grid)
    out = np.array(grid, copy=True)
    frames = {int(item["size"]): item for item in placed_specs if item["kind"] == "frame"}
    blocks = [item for item in placed_specs if item["kind"] == "block"]

    for block in blocks:
        r0, c0, r1, c1 = map(int, block["bbox"])
        out[r0 : r1 + 1, c0 : c1 + 1] = int(bg)

    for block in blocks:
        frame = frames[int(block["size"])]
        r0, c0, r1, c1 = map(int, frame["bbox"])
        patch = out[r0 : r1 + 1, c0 : c1 + 1]
        patch[1:-1, 1:-1] = int(frame["color"])
        patch[0, :] = int(block["color"])
        patch[-1, :] = int(block["color"])
        patch[:, 0] = int(block["color"])
        patch[:, -1] = int(block["color"])

    return out


def _gen_attempt() -> np.ndarray | None:
    height = _randint(GRID_MIN, GRID_MAX)
    width = _randint(GRID_MIN, GRID_MAX)
    count = _choice((2, 3))
    sizes = _sample(SIZE_OPTIONS, count)
    colors = _sample(PALETTE, 2 * count)

    frame_colors = colors[:count]
    block_colors = colors[count:]
    _RNG.shuffle(frame_colors)
    _RNG.shuffle(block_colors)

    specs: list[dict] = []
    for size, color in zip(sizes, frame_colors, strict=False):
        specs.append({"kind": "frame", "size": int(size), "span": int(size + 2), "color": int(color)})
    for size, color in zip(sizes, block_colors, strict=False):
        specs.append({"kind": "block", "size": int(size), "span": int(size), "color": int(color)})

    placed_specs = _place_objects(height, width, specs)
    if placed_specs is None:
        return None

    grid = np.zeros((height, width), dtype=int)
    for item in placed_specs:
        if item["kind"] == "frame":
            _draw_frame(grid, item["bbox"], int(item["color"]))
        else:
            _draw_block(grid, item["bbox"], int(item["color"]))

    expected = _expected_output(grid, placed_specs)

    try:
        bg, frames, blocks = _classify_objects(grid)
        if int(bg) != 0:
            return None
        if len(frames) != count or len(blocks) != count:
            return None
        if set(frames.keys()) != {int(size) for size in sizes}:
            return None
        if not np.array_equal(main(grid), expected):
            return None
    except AssertionError:
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
