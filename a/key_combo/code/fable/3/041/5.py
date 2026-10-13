from __future__ import annotations

from itertools import combinations
import random

import numpy as np


BLACK = 0
RED = 2
BLOCK_COLORS = (1, 3, 4, 5, 6, 7, 8, 9)
DIRS4 = ((-1, 0), (1, 0), (0, -1), (0, 1))


def _components(mask: np.ndarray) -> list[np.ndarray]:
    mask = np.asarray(mask, dtype=bool)
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for sr, sc in np.argwhere(mask):
        sr = int(sr)
        sc = int(sc)
        if seen[sr, sc]:
            continue
        stack = [(sr, sc)]
        seen[sr, sc] = True
        coords: list[tuple[int, int]] = []
        while stack:
            r, c = stack.pop()
            coords.append((r, c))
            for dr, dc in DIRS4:
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _bbox(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _count_chambers_patch(patch: np.ndarray) -> int:
    interior = patch[1:-1, 1:-1]
    return len(_components(interior == BLACK))


def _find_frames(g: np.ndarray) -> list[dict]:
    frames: list[dict] = []
    for coords in _components(g == RED):
        r0, c0, r1, c1 = _bbox(coords)
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 4 and w >= 4, "red frame is too small"

        patch = g[r0 : r1 + 1, c0 : c1 + 1]
        assert bool(np.all(patch[0, :] == RED))
        assert bool(np.all(patch[-1, :] == RED))
        assert bool(np.all(patch[:, 0] == RED))
        assert bool(np.all(patch[:, -1] == RED))
        assert int(np.count_nonzero(patch == RED)) == int(coords.shape[0])

        interior = patch[1:-1, 1:-1]
        assert bool(np.all((interior == BLACK) | (interior == RED)))
        chambers = _count_chambers_patch(patch)
        assert chambers >= 1, "each frame must contain at least one black chamber"
        frames.append(
            {
                "bbox": (r0, c0, r1, c1),
                "coords": coords,
                "chambers": chambers,
                "area": h * w,
            }
        )

    assert frames, "expected at least one red hollow frame"
    assert len({f["chambers"] for f in frames}) == len(frames), "frame chamber counts must be distinct"
    return frames


def _find_blocks(g: np.ndarray) -> list[dict]:
    blocks: list[dict] = []
    for color in sorted(set(map(int, np.unique(g))) - {BLACK, RED}):
        color_comps = _components(g == int(color))
        for coords in color_comps:
            r0, c0, r1, c1 = _bbox(coords)
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h == w and h >= 2, "free blocks must be solid squares"
            assert int(coords.shape[0]) == h * w, "free block must fill its square bbox"
            assert bool(np.all(g[r0 : r1 + 1, c0 : c1 + 1] == int(color)))
            blocks.append(
                {
                    "color": int(color),
                    "bbox": (r0, c0, r1, c1),
                    "coords": coords,
                    "area": int(coords.shape[0]),
                }
            )

    assert blocks, "expected free colored blocks"
    assert len({b["color"] for b in blocks}) == len(blocks), "block colors must be distinct"
    assert len({b["area"] for b in blocks}) == len(blocks), "block sizes must be distinct"
    return blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill each red frame's chambers with the color of the equally ranked free block.

    Input structure:
    - A black background contains several separated red hollow rectangular frames.
    - Inside each frame are red wall segments; wall segments reaching two opposite red boundaries split chambers, while dangling one-ended segments leave a gap.
    - The same number of separated solid monochrome square blocks appear outside the frames, with distinct sizes and distinct non-red colors.

    Rule:
    - Count each frame's internal black chambers by 4-connectivity inside that frame.
    - Sort frames by chamber count descending and sort free square blocks by area descending; pair objects with equal rank.
    - Erase the free blocks to black, then recolor every black cell inside each paired frame to the paired block's color while preserving all red frame and wall pixels.

    Key insight:
    - The color assignment is cross-set rank pairing: it depends on chamber topology and block size, not on position, proximity, frame size, or the raw number of red wall segments.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _find_frames(g)
    blocks = _find_blocks(g)
    assert len(frames) == len(blocks), "frame and free-block counts must match"

    out = g.copy()
    for block in blocks:
        coords = block["coords"]
        out[coords[:, 0], coords[:, 1]] = BLACK

    ranked_frames = sorted(frames, key=lambda item: item["chambers"], reverse=True)
    ranked_blocks = sorted(blocks, key=lambda item: item["area"], reverse=True)
    for frame, block in zip(ranked_frames, ranked_blocks):
        r0, c0, r1, c1 = frame["bbox"]
        interior = g[r0 + 1 : r1, c0 + 1 : c1]
        fill_mask = interior == BLACK
        view = out[r0 + 1 : r1, c0 + 1 : c1]
        view[fill_mask] = int(block["color"])

    return out


# --------------- generator ---------------

MAX_GENERATION_ATTEMPTS = 500
GRID_SIDE_RANGE = (24, 30)
FRAME_COUNT_CHOICES = (3, 3, 4)
CHAMBER_POOL = (1, 2, 3, 4, 5)
BLOCK_SIDES = (2, 3, 4, 5)
PLACEMENT_PAD = 1


def _wall_positions(span: int, count: int) -> list[int] | None:
    if count == 0:
        return []
    candidates = [
        combo
        for combo in combinations(range(2, int(span) - 2), int(count))
        if all(combo[i + 1] - combo[i] >= 2 for i in range(len(combo) - 1))
    ]
    if not candidates:
        return None
    return list(random.choice(candidates))


def _empty_frame(h: int, w: int) -> np.ndarray:
    patch = np.zeros((h, w), dtype=int)
    patch[0, :] = RED
    patch[-1, :] = RED
    patch[:, 0] = RED
    patch[:, -1] = RED
    return patch


def _add_dangling_wall(patch: np.ndarray) -> bool:
    h, w = patch.shape
    old_chambers = _count_chambers_patch(patch)
    starts: list[tuple[int, int, int, int]] = []
    for c in range(1, w - 1):
        starts.append((0, c, 1, 0))
        starts.append((h - 1, c, -1, 0))
    for r in range(1, h - 1):
        starts.append((r, 0, 0, 1))
        starts.append((r, w - 1, 0, -1))
    random.shuffle(starts)

    for ar, ac, dr, dc in starts:
        cells: list[tuple[int, int]] = []
        r, c = ar + dr, ac + dc
        while 0 < r < h - 1 and 0 < c < w - 1 and int(patch[r, c]) == BLACK:
            cells.append((r, c))
            r += dr
            c += dc
        if len(cells) < 3:
            continue
        lengths = list(range(2, len(cells)))
        random.shuffle(lengths)
        for length in lengths:
            trial = patch.copy()
            for rr, cc in cells[:length]:
                trial[rr, cc] = RED
            if _count_chambers_patch(trial) == old_chambers:
                patch[:, :] = trial
                return True
    return False


def _make_frame_patch(chambers: int) -> tuple[np.ndarray, dict] | None:
    full_walls = int(chambers) - 1
    desired_dangling = max(1, 5 - int(chambers))

    for _ in range(80):
        orient = random.choice(("V", "H"))
        if full_walls == 0:
            h = random.randint(7, 12)
            w = random.randint(7, 12)
        elif orient == "V":
            min_w = max(7, 2 * full_walls + 3)
            w = random.randint(min_w, min(13, min_w + 3))
            h = random.randint(7, 11)
        else:
            min_h = max(7, 2 * full_walls + 3)
            h = random.randint(min_h, min(13, min_h + 3))
            w = random.randint(7, 11)

        patch = _empty_frame(h, w)
        if orient == "V":
            positions = _wall_positions(w, full_walls)
            if positions is None:
                continue
            for c in positions:
                patch[1 : h - 1, c] = RED
        else:
            positions = _wall_positions(h, full_walls)
            if positions is None:
                continue
            for r in positions:
                patch[r, 1 : w - 1] = RED

        if _count_chambers_patch(patch) != int(chambers):
            continue

        dangling = 0
        for _attempt in range(50):
            if dangling >= desired_dangling:
                break
            if _add_dangling_wall(patch):
                dangling += 1
        if dangling == 0:
            continue
        if _count_chambers_patch(patch) != int(chambers):
            continue

        return patch, {
            "chambers": int(chambers),
            "full_walls": int(full_walls),
            "dangling_walls": int(dangling),
            "wall_segments": int(full_walls + dangling),
            "orient": orient,
        }
    return None


def _can_place(reserved: np.ndarray, r: int, c: int, h: int, w: int, pad: int) -> bool:
    rows, cols = reserved.shape
    rr0 = max(0, r - pad)
    cc0 = max(0, c - pad)
    rr1 = min(rows, r + h + pad)
    cc1 = min(cols, c + w + pad)
    return not bool(reserved[rr0:rr1, cc0:cc1].any())


def _place_items(items: list[dict], rows: int, cols: int) -> tuple[np.ndarray, list[dict]] | None:
    grid = np.zeros((rows, cols), dtype=int)
    reserved = np.zeros((rows, cols), dtype=bool)
    placed: list[dict] = []
    order = list(range(len(items)))
    random.shuffle(order)
    order.sort(key=lambda idx: items[idx]["patch"].size, reverse=True)

    for idx in order:
        item = dict(items[idx])
        patch = item["patch"]
        h, w = patch.shape
        locs = [
            (r, c)
            for r in range(rows - h + 1)
            for c in range(cols - w + 1)
            if _can_place(reserved, r, c, h, w, PLACEMENT_PAD)
        ]
        if not locs:
            return None
        r0, c0 = random.choice(locs)
        mask = patch != BLACK
        view = grid[r0 : r0 + h, c0 : c0 + w]
        view[mask] = patch[mask]
        reserved[r0 : r0 + h, c0 : c0 + w] = True
        item["bbox"] = (r0, c0, r0 + h - 1, c0 + w - 1)
        placed.append(item)

    return grid, placed


def _expected_output(grid: np.ndarray, placed: list[dict]) -> np.ndarray:
    out = grid.copy()
    frames = [item for item in placed if item["kind"] == "frame"]
    blocks = [item for item in placed if item["kind"] == "block"]

    for block in blocks:
        r0, c0, r1, c1 = block["bbox"]
        out[r0 : r1 + 1, c0 : c1 + 1] = BLACK

    ranked_frames = sorted(frames, key=lambda item: item["chambers"], reverse=True)
    ranked_blocks = sorted(blocks, key=lambda item: item["area"], reverse=True)
    for frame, block in zip(ranked_frames, ranked_blocks):
        r0, c0, _r1, _c1 = frame["bbox"]
        patch = frame["patch"]
        mask = patch[1:-1, 1:-1] == BLACK
        view = out[r0 + 1 : r0 + patch.shape[0] - 1, c0 + 1 : c0 + patch.shape[1] - 1]
        view[mask] = int(block["color"])
    return out


def _center(box: tuple[int, int, int, int]) -> tuple[float, float]:
    r0, c0, r1, c1 = box
    return (r0 + r1) / 2.0, (c0 + c1) / 2.0


def _semantic_checks(placed: list[dict]) -> bool:
    frames = [item for item in placed if item["kind"] == "frame"]
    blocks = [item for item in placed if item["kind"] == "block"]
    if len(frames) != len(blocks):
        return False
    if not any(frame["dangling_walls"] > 0 for frame in frames):
        return False

    chamber_rank = [id(item) for item in sorted(frames, key=lambda item: item["chambers"], reverse=True)]
    area_rank = [id(item) for item in sorted(frames, key=lambda item: (item["patch"].size, item["bbox"]), reverse=True)]
    if len({item["patch"].size for item in frames}) == len(frames) and chamber_rank == area_rank:
        return False

    segment_rank = [id(item) for item in sorted(frames, key=lambda item: item["wall_segments"], reverse=True)]
    if len({item["wall_segments"] for item in frames}) == len(frames) and chamber_rank == segment_rank:
        return False

    ranked_frames = sorted(frames, key=lambda item: item["chambers"], reverse=True)
    ranked_blocks = sorted(blocks, key=lambda item: item["area"], reverse=True)
    rank_pair = {id(frame): id(block) for frame, block in zip(ranked_frames, ranked_blocks)}
    nearest_pair = {}
    for frame in frames:
        fr, fc = _center(frame["bbox"])
        nearest = min(
            blocks,
            key=lambda block: abs(fr - _center(block["bbox"])[0]) + abs(fc - _center(block["bbox"])[1]),
        )
        nearest_pair[id(frame)] = id(nearest)
    if all(rank_pair[id(frame)] == nearest_pair[id(frame)] for frame in frames):
        return False
    return True


def _gen_attempt() -> np.ndarray | None:
    frame_count = random.choice(FRAME_COUNT_CHOICES)
    chambers = [1] + random.sample([n for n in CHAMBER_POOL if n != 1], frame_count - 1)
    random.shuffle(chambers)

    items: list[dict] = []
    for chamber_count in chambers:
        made = _make_frame_patch(chamber_count)
        if made is None:
            return None
        patch, meta = made
        item = {"kind": "frame", "patch": patch, **meta}
        items.append(item)

    sides = random.sample(BLOCK_SIDES, frame_count)
    colors = random.sample(BLOCK_COLORS, frame_count)
    for side, color in zip(sides, colors):
        patch = np.full((side, side), int(color), dtype=int)
        items.append(
            {
                "kind": "block",
                "patch": patch,
                "color": int(color),
                "side": int(side),
                "area": int(side * side),
            }
        )

    rows = random.randint(*GRID_SIDE_RANGE)
    cols = random.randint(*GRID_SIDE_RANGE)
    rendered = _place_items(items, rows, cols)
    if rendered is None:
        return None
    grid, placed = rendered

    if not _semantic_checks(placed):
        return None

    expected = _expected_output(grid, placed)
    try:
        actual = main(grid)
        found_frames = _find_frames(grid)
        found_blocks = _find_blocks(grid)
    except AssertionError:
        return None
    if len(found_frames) != frame_count or len(found_blocks) != frame_count:
        return None
    if sorted(frame["chambers"] for frame in found_frames) != sorted(chambers):
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid chamber-rank pairing input")
