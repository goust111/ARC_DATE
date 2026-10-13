from __future__ import annotations

from collections import deque
import random

import numpy as np


BG = 0
GREEN = 3
GREY = 5
ORANGE = 7
PAYLOAD_COLORS = (1, 2, 4, 6, 8, 9)
DIRS = [(-1, 0), (0, 1), (1, 0), (0, -1)]
MAX_GEN_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                pts.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(pts)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _frame_from_component(
    grid: np.ndarray, coords: list[tuple[int, int]], color: int
) -> dict[str, object] | None:
    r0, c0, r1, c1 = _bbox(coords)
    h, w = r1 - r0 + 1, c1 - c0 + 1
    if h < 5 or w < 5:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True
    if len(coords) != int(border.sum()):
        return None
    if not np.all(patch[border] == color):
        return None
    if np.any(patch[~border] == color):
        return None
    return {"bbox": (r0, c0, r1, c1)}


def _find_frames(grid: np.ndarray, color: int) -> list[dict[str, object]]:
    frames = []
    for comp in _components(grid == color):
        frame = _frame_from_component(grid, comp, color)
        if frame is not None:
            frames.append(frame)
    frames.sort(key=lambda item: item["bbox"])  # type: ignore[index]
    return frames


def _owner_map(shape: tuple[int, int], frames: list[dict[str, object]]) -> np.ndarray:
    owners = np.full(shape, -1, dtype=int)
    for idx, frame in enumerate(frames):
        r0, c0, r1, c1 = frame["bbox"]  # type: ignore[index]
        owners[r0, c0 : c1 + 1] = idx
        owners[r1, c0 : c1 + 1] = idx
        owners[r0 : r1 + 1, c0] = idx
        owners[r0 : r1 + 1, c1] = idx
    return owners


def _pairings(
    grid: np.ndarray, sources: list[dict[str, object]], molds: list[dict[str, object]]
) -> list[dict[str, int]]:
    h, w = grid.shape
    source_owner = _owner_map(grid.shape, sources)
    mold_owner = _owner_map(grid.shape, molds)
    pairs: list[dict[str, int]] = []
    used_sources: set[int] = set()
    used_molds: set[int] = set()

    for comp in _components(grid == GREY):
        source_dirs: dict[int, set[int]] = {}
        mold_dirs: dict[int, set[int]] = {}
        source_contacts: dict[int, set[tuple[int, int]]] = {}
        mold_contacts: dict[int, set[tuple[int, int]]] = {}
        for r, c in comp:
            for d, (dr, dc) in enumerate(DIRS):
                nr, nc = r + dr, c + dc
                if not (0 <= nr < h and 0 <= nc < w):
                    continue
                sid = int(source_owner[nr, nc])
                if sid >= 0:
                    source_dirs.setdefault(sid, set()).add((d + 2) % 4)
                    source_contacts.setdefault(sid, set()).add((r, c))
                mid = int(mold_owner[nr, nc])
                if mid >= 0:
                    mold_dirs.setdefault(mid, set()).add(d)
                    mold_contacts.setdefault(mid, set()).add((r, c))

        assert len(source_dirs) == 1, "each grey path must touch exactly one source"
        assert len(mold_dirs) == 1, "each grey path must touch exactly one mold"

        source_id = next(iter(source_dirs))
        mold_id = next(iter(mold_dirs))
        assert len(source_dirs[source_id]) == 1
        assert len(mold_dirs[mold_id]) == 1
        assert len(source_contacts[source_id]) == 1
        assert len(mold_contacts[mold_id]) == 1
        assert source_id not in used_sources
        assert mold_id not in used_molds

        dep = next(iter(source_dirs[source_id]))
        arr = next(iter(mold_dirs[mold_id]))
        pairs.append({"source": source_id, "mold": mold_id, "rotation": (arr - dep) % 4})
        used_sources.add(source_id)
        used_molds.add(mold_id)

    assert len(pairs) == len(sources) == len(molds)
    return pairs


def _analyze_scene(
    grid: np.ndarray,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, int]]]:
    sources = _find_frames(grid, GREEN)
    molds = _find_frames(grid, ORANGE)
    assert sources, "expected at least one green-framed source"
    assert molds, "expected at least one orange-framed mold"
    pairs = _pairings(grid, sources, molds)
    return sources, molds, pairs


def main(grid: np.ndarray) -> np.ndarray:
    """
    Pair each green source frame with its orange mold via the grey path, rotate
    the source interior by the path's net clockwise direction change, and write
    the rotated pattern into the mold cavity.
    """
    g = np.asarray(grid, dtype=int)
    sources, molds, pairs = _analyze_scene(g)
    out = g.copy()

    for pair in pairs:
        sr0, sc0, sr1, sc1 = sources[pair["source"]]["bbox"]  # type: ignore[index]
        mr0, mc0, mr1, mc1 = molds[pair["mold"]]["bbox"]  # type: ignore[index]
        payload = g[sr0 + 1 : sr1, sc0 + 1 : sc1]
        rotated = np.rot90(payload, -pair["rotation"])
        mold_h = mr1 - mr0 - 1
        mold_w = mc1 - mc0 - 1
        assert rotated.shape == (mold_h, mold_w)
        out[mr0 + 1 : mr1, mc0 + 1 : mc1] = rotated

    return out


def _random_pattern(size: int, rng: random.Random) -> np.ndarray:
    for _ in range(200):
        color_count = rng.randint(2, min(4, len(PAYLOAD_COLORS)))
        palette = rng.sample(PAYLOAD_COLORS, color_count)
        pattern = np.array(
            [[rng.choice(palette) for _ in range(size)] for _ in range(size)],
            dtype=int,
        )
        if len(set(map(int, pattern.flatten()))) < 2:
            continue
        if any(np.array_equal(pattern, np.rot90(pattern, -k)) for k in (1, 2, 3)):
            continue
        return pattern
    raise RuntimeError("failed to build a non-symmetric payload")


def _outside_source_cell(outer: int, direction: int, offset: int) -> tuple[int, int]:
    if direction == 0:
        return -1, offset
    if direction == 1:
        return offset, outer
    if direction == 2:
        return outer, offset
    return offset, -1


def _mold_bbox_from_endpoint(
    endpoint: tuple[int, int], outer: int, direction: int, offset: int
) -> tuple[int, int, int, int]:
    r, c = endpoint
    if direction == 0:
        r0, c0 = r - outer, c - offset
    elif direction == 1:
        r0, c0 = r - offset, c + 1
    elif direction == 2:
        r0, c0 = r + 1, c - offset
    else:
        r0, c0 = r - offset, c - outer
    return r0, c0, r0 + outer - 1, c0 + outer - 1


def _boxes_overlap(
    a: tuple[int, int, int, int], b: tuple[int, int, int, int]
) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _build_pair_block(rng: random.Random) -> np.ndarray:
    for _ in range(300):
        inner = rng.randint(3, 5)
        outer = inner + 2
        payload = _random_pattern(inner, rng)
        dep = rng.randrange(4)
        rotation = rng.randrange(4)
        arr = (dep + rotation) % 4

        source_offset = rng.randint(1, outer - 2)
        mold_offset = rng.randint(1, outer - 2)
        start = _outside_source_cell(outer, dep, source_offset)

        if rotation == 0 and rng.random() < 0.5:
            dirs = [dep]
            lengths = [rng.randint(2, 5)]
        elif rotation == 0:
            mid = rng.choice([(dep + 1) % 4, (dep + 3) % 4])
            dirs = [dep, mid, dep]
            lengths = [rng.randint(1, 3), rng.randint(2, 5), rng.randint(1, 3)]
        elif rotation in (1, 3):
            dirs = [dep, arr]
            lengths = [rng.randint(2, 5), rng.randint(2, 5)]
        else:
            mid = rng.choice([(dep + 1) % 4, (dep + 3) % 4])
            dirs = [dep, mid, arr]
            lengths = [rng.randint(1, 3), rng.randint(2, 5), rng.randint(1, 3)]

        path = [start]
        cr, cc = start
        for direction, length in zip(dirs, lengths):
            dr, dc = DIRS[direction]
            for _step in range(length):
                cr += dr
                cc += dc
                path.append((cr, cc))
        end = path[-1]

        source_bbox = (0, 0, outer - 1, outer - 1)
        mold_bbox = _mold_bbox_from_endpoint(end, outer, arr, mold_offset)
        if _boxes_overlap(source_bbox, mold_bbox):
            continue
        if len(path) != len(set(path)):
            continue

        all_rows = [0, outer - 1, mold_bbox[0], mold_bbox[2], *[r for r, _ in path]]
        all_cols = [0, outer - 1, mold_bbox[1], mold_bbox[3], *[c for _, c in path]]
        shift_r = -min(all_rows)
        shift_c = -min(all_cols)

        shifted_source = (
            source_bbox[0] + shift_r,
            source_bbox[1] + shift_c,
            source_bbox[2] + shift_r,
            source_bbox[3] + shift_c,
        )
        shifted_mold = (
            mold_bbox[0] + shift_r,
            mold_bbox[1] + shift_c,
            mold_bbox[2] + shift_r,
            mold_bbox[3] + shift_c,
        )
        shifted_path = [(r + shift_r, c + shift_c) for r, c in path]

        h = max([shifted_source[2], shifted_mold[2], *[r for r, _ in shifted_path]]) + 1
        w = max([shifted_source[3], shifted_mold[3], *[c for _, c in shifted_path]]) + 1
        grid = np.zeros((h, w), dtype=int)

        sr0, sc0, sr1, sc1 = shifted_source
        mr0, mc0, mr1, mc1 = shifted_mold
        grid[sr0, sc0 : sc1 + 1] = GREEN
        grid[sr1, sc0 : sc1 + 1] = GREEN
        grid[sr0 : sr1 + 1, sc0] = GREEN
        grid[sr0 : sr1 + 1, sc1] = GREEN
        grid[sr0 + 1 : sr1, sc0 + 1 : sc1] = payload

        grid[mr0, mc0 : mc1 + 1] = ORANGE
        grid[mr1, mc0 : mc1 + 1] = ORANGE
        grid[mr0 : mr1 + 1, mc0] = ORANGE
        grid[mr0 : mr1 + 1, mc1] = ORANGE

        for r, c in shifted_path:
            if grid[r, c] != BG:
                break
            grid[r, c] = GREY
        else:
            try:
                _, _, pairs = _analyze_scene(grid)
            except AssertionError:
                continue
            if len(pairs) != 1 or pairs[0]["rotation"] != rotation:
                continue
            solved = main(grid)
            expected = np.rot90(payload, -rotation)
            if not np.array_equal(solved[mr0 + 1 : mr1, mc0 + 1 : mc1], expected):
                continue
            return grid

    raise RuntimeError("failed to build a valid source-path-mold block")


def _inflate_mask(mask: np.ndarray) -> np.ndarray:
    h, w = mask.shape
    out = mask.copy()
    for dr in (-1, 0, 1):
        for dc in (-1, 0, 1):
            if dr == 0 and dc == 0:
                continue
            src_r0 = max(0, -dr)
            src_r1 = min(h, h - dr)
            src_c0 = max(0, -dc)
            src_c1 = min(w, w - dc)
            dst_r0 = max(0, dr)
            dst_r1 = min(h, h + dr)
            dst_c0 = max(0, dc)
            dst_c1 = min(w, w + dc)
            out[dst_r0:dst_r1, dst_c0:dst_c1] |= mask[src_r0:src_r1, src_c0:src_c1]
    return out


def _place_blocks(blocks: list[np.ndarray], rng: random.Random) -> np.ndarray | None:
    block_masks = [block != BG for block in blocks]
    max_h = max(block.shape[0] for block in blocks)
    max_w = max(block.shape[1] for block in blocks)
    area = sum((mask.shape[0] + 2) * (mask.shape[1] + 2) for mask in block_masks)
    side = max(max_h + 2, max_w + 2, int(np.ceil(np.sqrt(area * 1.25))))

    for _ in range(120):
        h = rng.randint(side, side + 8)
        w = rng.randint(side, side + 8)
        if h <= max_h + 1 or w <= max_w + 1:
            continue
        grid = np.zeros((h, w), dtype=int)
        occupied = np.zeros((h, w), dtype=bool)
        order = list(range(len(blocks)))
        order.sort(key=lambda idx: blocks[idx].shape[0] * blocks[idx].shape[1], reverse=True)
        placed = True

        for idx in order:
            block = blocks[idx]
            mask = block_masks[idx]
            bh, bw = block.shape
            forbidden = _inflate_mask(occupied)
            candidates: list[tuple[int, int]] = []
            for r0 in range(1, h - bh):
                for c0 in range(1, w - bw):
                    if np.any(mask & forbidden[r0 : r0 + bh, c0 : c0 + bw]):
                        continue
                    candidates.append((r0, c0))
            if not candidates:
                placed = False
                break
            r0, c0 = rng.choice(candidates)
            patch = grid[r0 : r0 + bh, c0 : c0 + bw]
            patch[mask] = block[mask]
            occupied[r0 : r0 + bh, c0 : c0 + bw] |= mask

        if placed:
            rows, cols = np.where(occupied)
            r0, r1 = int(rows.min()), int(rows.max())
            c0, c1 = int(cols.min()), int(cols.max())
            cropped = grid[r0 : r1 + 1, c0 : c1 + 1]
            pad_top = rng.randint(1, 3)
            pad_bottom = rng.randint(1, 3)
            pad_left = rng.randint(1, 3)
            pad_right = rng.randint(1, 3)
            out = np.zeros(
                (
                    cropped.shape[0] + pad_top + pad_bottom,
                    cropped.shape[1] + pad_left + pad_right,
                ),
                dtype=int,
            )
            out[pad_top : pad_top + cropped.shape[0], pad_left : pad_left + cropped.shape[1]] = cropped
            return out
    return None


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    pair_count = rng.randint(1, 3)
    blocks = [_build_pair_block(rng) for _ in range(pair_count)]
    rng.shuffle(blocks)
    grid = _place_blocks(blocks, rng)
    if grid is None:
        return None
    try:
        sources, molds, pairs = _analyze_scene(grid)
    except AssertionError:
        return None
    if len(sources) != pair_count or len(molds) != pair_count or len(pairs) != pair_count:
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_GEN_ATTEMPTS} attempts")
