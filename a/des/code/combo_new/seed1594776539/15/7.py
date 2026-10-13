from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import numpy as np


BG = 0
AZURE = 8
PALETTE = [1, 2, 3, 4, 5, 6, 7, 9]
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]


@dataclass(frozen=True)
class Component:
    color: int
    coords: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]


@dataclass(frozen=True)
class Container:
    bbox: tuple[int, int, int, int]
    side: str
    gap: tuple[int, int]


def _components(grid: np.ndarray, mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for dr, dc in DIRS4:
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]] | tuple[tuple[int, int], ...]) -> tuple[int, int, int, int]:
    rows = [r for r, _c in coords]
    cols = [c for _r, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _missing_segments(side: np.ndarray) -> list[tuple[int, int]]:
    segs: list[tuple[int, int]] = []
    start: int | None = None
    for i, present in enumerate(side.tolist()):
        if not present and start is None:
            start = i
        elif present and start is not None:
            segs.append((start, i - 1))
            start = None
    if start is not None:
        segs.append((start, len(side) - 1))
    return segs


def _detect_containers(grid: np.ndarray) -> list[Container]:
    containers: list[Container] = []
    for coords in _components(grid, grid == AZURE):
        r0, c0, r1, c1 = _bbox(coords)
        if r1 - r0 < 4 or c1 - c0 < 4:
            continue

        top = grid[r0, c0 : c1 + 1] == AZURE
        bottom = grid[r1, c0 : c1 + 1] == AZURE
        left = grid[r0 : r1 + 1, c0] == AZURE
        right = grid[r0 : r1 + 1, c1] == AZURE

        side_segments = {
            "top": _missing_segments(top),
            "bottom": _missing_segments(bottom),
            "left": _missing_segments(left),
            "right": _missing_segments(right),
        }
        gap_sides = []
        for side, segs in side_segments.items():
            if len(segs) != 1:
                continue
            start, end = segs[0]
            if start == 0 or end == (len(top) - 1 if side in {"top", "bottom"} else len(left) - 1):
                continue
            if 2 <= end - start + 1 <= 3:
                gap_sides.append((side, start, end))

        if len(gap_sides) != 1:
            continue

        side, start, end = gap_sides[0]
        other_sides = [s for s in ("top", "bottom", "left", "right") if s != side]
        if any(side_segments[s] for s in other_sides):
            continue

        if side in {"top", "bottom"}:
            gap = (c0 + start, c0 + end)
        else:
            gap = (r0 + start, r0 + end)
        containers.append(Container(bbox=(r0, c0, r1, c1), side=side, gap=gap))

    containers.sort(key=lambda item: item.bbox)
    return containers


def _detect_blocks(grid: np.ndarray) -> list[Component]:
    mask = (grid != BG) & (grid != AZURE)
    blocks: list[Component] = []
    for coords_list in _components(grid, mask):
        coords = tuple(sorted((int(r), int(c)) for r, c in coords_list))
        color = int(grid[coords[0][0], coords[0][1]])
        blocks.append(Component(color=color, coords=coords, bbox=_bbox(coords)))
    blocks.sort(key=lambda item: item.bbox)
    return blocks


def _match_block(container: Container, blocks: list[Component], used: set[int]) -> int | None:
    r0, c0, r1, c1 = container.bbox
    gap0, gap1 = container.gap
    strict: list[tuple[int, int]] = []
    loose: list[tuple[int, int]] = []

    for idx, block in enumerate(blocks):
        if idx in used:
            continue
        br0, bc0, br1, bc1 = block.bbox
        if container.side == "top":
            if br1 >= r0:
                continue
            strict_ok = bc0 >= gap0 and bc1 <= gap1
            loose_ok = max(bc0, gap0) <= min(bc1, gap1) and (bc1 - bc0 + 1) <= (gap1 - gap0 + 1)
            if strict_ok:
                strict.append((r0 - br1, idx))
            elif loose_ok:
                loose.append((r0 - br1, idx))
        elif container.side == "bottom":
            if br0 <= r1:
                continue
            strict_ok = bc0 >= gap0 and bc1 <= gap1
            loose_ok = max(bc0, gap0) <= min(bc1, gap1) and (bc1 - bc0 + 1) <= (gap1 - gap0 + 1)
            if strict_ok:
                strict.append((br0 - r1, idx))
            elif loose_ok:
                loose.append((br0 - r1, idx))
        elif container.side == "left":
            if bc1 >= c0:
                continue
            strict_ok = br0 >= gap0 and br1 <= gap1
            loose_ok = max(br0, gap0) <= min(br1, gap1) and (br1 - br0 + 1) <= (gap1 - gap0 + 1)
            if strict_ok:
                strict.append((c0 - bc1, idx))
            elif loose_ok:
                loose.append((c0 - bc1, idx))
        else:
            if bc0 <= c1:
                continue
            strict_ok = br0 >= gap0 and br1 <= gap1
            loose_ok = max(br0, gap0) <= min(br1, gap1) and (br1 - br0 + 1) <= (gap1 - gap0 + 1)
            if strict_ok:
                strict.append((bc0 - c1, idx))
            elif loose_ok:
                loose.append((bc0 - c1, idx))

    pool = strict if strict else loose
    if not pool:
        return None
    pool.sort()
    return pool[0][1]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each aligned outside block through an open azure frame into the opposite inner wall.

    Input structure:
    - Background is black.
    - Several hollow azure rectangular frames appear on the grid; each frame has one 2-3 pixel opening on exactly one side.
    - Outside the frames are solid colored blocks, each monochrome and distinct from black and azure.
    - For every frame, exactly one outside block lies along the opening normal and fits through that opening.

    Rule:
    - Detect each azure frame, its missing side, and the opening interval on that side.
    - Search outward from the opening direction to find the aligned block assigned to that frame.
    - Translate that block only along the opening normal, moving it through the opening until it touches the opposite inner wall.
    - Clear the block's original position to black and keep all colors unchanged elsewhere.

    Key insight:
    - The missing border segment is not damage: it encodes which external block the frame captures and which direction that block must slide.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    containers = _detect_containers(g)
    blocks = _detect_blocks(g)

    moves: list[tuple[Component, list[tuple[int, int]]]] = []
    used: set[int] = set()

    for container in containers:
        block_idx = _match_block(container, blocks, used)
        if block_idx is None:
            continue
        used.add(block_idx)
        block = blocks[block_idx]
        r0, c0, r1, c1 = container.bbox
        br0, bc0, br1, bc1 = block.bbox
        bh = br1 - br0 + 1
        bw = bc1 - bc0 + 1

        if container.side == "top":
            dest_r0 = r1 - bh
            dest_c0 = bc0
        elif container.side == "bottom":
            dest_r0 = r0 + 1
            dest_c0 = bc0
        elif container.side == "left":
            dest_r0 = br0
            dest_c0 = c1 - bw
        else:
            dest_r0 = br0
            dest_c0 = c0 + 1

        new_coords = []
        for r, c in block.coords:
            new_coords.append((dest_r0 + (r - br0), dest_c0 + (c - bc0)))
        moves.append((block, new_coords))

    for block, _new_coords in moves:
        for r, c in block.coords:
            out[r, c] = BG
    for block, new_coords in moves:
        for r, c in new_coords:
            out[r, c] = block.color
    return out


def _draw_container_patch(
    frame_h: int,
    frame_w: int,
    side: str,
    gap_start: int,
    gap_len: int,
) -> np.ndarray:
    patch = np.zeros((frame_h, frame_w), dtype=int)
    patch[0, :] = AZURE
    patch[-1, :] = AZURE
    patch[:, 0] = AZURE
    patch[:, -1] = AZURE
    if side == "top":
        patch[0, gap_start : gap_start + gap_len] = BG
    elif side == "bottom":
        patch[-1, gap_start : gap_start + gap_len] = BG
    elif side == "left":
        patch[gap_start : gap_start + gap_len, 0] = BG
    else:
        patch[gap_start : gap_start + gap_len, -1] = BG
    return patch


def _make_item(rng: np.random.Generator, side: str, color: int) -> dict[str, object]:
    gap_len = int(rng.integers(2, 4))
    separation = int(rng.integers(1, 5))

    if side in {"top", "bottom"}:
        block_h = int(rng.integers(1, 4))
        block_w = int(rng.integers(1, gap_len + 1))
        interior_h = block_h + int(rng.integers(1, 5))
        interior_w = gap_len + int(rng.integers(1, 6))
        frame_h = interior_h + 2
        frame_w = interior_w + 2
        gap_start = int(rng.integers(1, frame_w - gap_len))
        block_offset = int(rng.integers(0, gap_len - block_w + 1))
        block_c = gap_start + block_offset

        if side == "top":
            patch_h = block_h + separation + frame_h
            patch_w = frame_w
            container_r = block_h + separation
            container_c = 0
            block_r = 0
            final_block_r = container_r + frame_h - 1 - block_h
        else:
            patch_h = frame_h + separation + block_h
            patch_w = frame_w
            container_r = 0
            container_c = 0
            block_r = frame_h + separation
            final_block_r = 1

        block_patch = np.full((block_h, block_w), color, dtype=int)
        patch = np.zeros((patch_h, patch_w), dtype=int)
        frame = _draw_container_patch(frame_h, frame_w, side, gap_start, gap_len)
        patch[container_r : container_r + frame_h, container_c : container_c + frame_w] = frame
        patch[block_r : block_r + block_h, block_c : block_c + block_w] = block_patch

        final_patch = frame.copy()
        final_patch[final_block_r : final_block_r + block_h, block_c : block_c + block_w] = color
        return {
            "patch": patch,
            "final_patch": final_patch,
            "container_offset": (container_r, container_c),
            "frame_shape": (frame_h, frame_w),
        }

    block_h = int(rng.integers(1, gap_len + 1))
    block_w = int(rng.integers(1, 4))
    interior_h = gap_len + int(rng.integers(1, 6))
    interior_w = block_w + int(rng.integers(1, 5))
    frame_h = interior_h + 2
    frame_w = interior_w + 2
    gap_start = int(rng.integers(1, frame_h - gap_len))
    block_offset = int(rng.integers(0, gap_len - block_h + 1))
    block_r = gap_start + block_offset

    if side == "left":
        patch_h = frame_h
        patch_w = block_w + separation + frame_w
        container_r = 0
        container_c = block_w + separation
        block_c = 0
        final_block_c = container_c + frame_w - 1 - block_w
    else:
        patch_h = frame_h
        patch_w = frame_w + separation + block_w
        container_r = 0
        container_c = 0
        block_c = frame_w + separation
        final_block_c = 1

    block_patch = np.full((block_h, block_w), color, dtype=int)
    patch = np.zeros((patch_h, patch_w), dtype=int)
    frame = _draw_container_patch(frame_h, frame_w, side, gap_start, gap_len)
    patch[container_r : container_r + frame_h, container_c : container_c + frame_w] = frame
    patch[block_r : block_r + block_h, block_c : block_c + block_w] = block_patch

    final_patch = frame.copy()
    final_patch[block_r : block_r + block_h, final_block_c : final_block_c + block_w] = color
    return {
        "patch": patch,
        "final_patch": final_patch,
        "container_offset": (container_r, container_c),
        "frame_shape": (frame_h, frame_w),
    }


def _paste(dst: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    mask = patch != BG
    sub = dst[top : top + patch.shape[0], left : left + patch.shape[1]]
    sub[mask] = patch[mask]


def _place_items(rng: np.random.Generator, items: list[dict[str, object]], height: int, width: int) -> tuple[np.ndarray, np.ndarray] | None:
    inp = np.zeros((height, width), dtype=int)
    out = np.zeros((height, width), dtype=int)
    occupied = np.zeros((height, width), dtype=bool)

    order = sorted(range(len(items)), key=lambda i: -int(np.prod(items[i]["patch"].shape)))  # type: ignore[index]
    placements: dict[int, tuple[int, int]] = {}

    for idx in order:
        patch = items[idx]["patch"]  # type: ignore[index]
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r in range(height - ph + 1):
            for c in range(width - pw + 1):
                if occupied[r : r + ph, c : c + pw].any():
                    continue
                candidates.append((r, c))
        if not candidates:
            return None
        top, left = candidates[int(rng.integers(0, len(candidates)))]
        placements[idx] = (top, left)
        _paste(inp, patch, top, left)

        frame = items[idx]["final_patch"]  # type: ignore[index]
        container_r, container_c = items[idx]["container_offset"]  # type: ignore[index]
        fh, fw = items[idx]["frame_shape"]  # type: ignore[index]
        if not np.any(out[top + container_r : top + container_r + fh, left + container_c : left + container_c + fw]):
            out[top + container_r : top + container_r + fh, left + container_c : left + container_c + fw] = frame

        rr0 = max(0, top - 1)
        cc0 = max(0, left - 1)
        rr1 = min(height, top + ph + 1)
        cc1 = min(width, left + pw + 1)
        occupied[rr0:rr1, cc0:cc1] = True

    return inp, out


def _build_example(rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray] | None:
    n_containers = int(rng.integers(2, 5))
    sides = list(rng.permutation(np.array(["top", "bottom", "left", "right"], dtype=object))[:n_containers])
    colors = [int(c) for c in rng.choice(np.array(PALETTE, dtype=int), size=n_containers, replace=False)]
    items = [_make_item(rng, str(side), color) for side, color in zip(sides, colors)]

    max_h = max(int(item["patch"].shape[0]) for item in items)  # type: ignore[index]
    max_w = max(int(item["patch"].shape[1]) for item in items)  # type: ignore[index]
    low_h = max(18, max_h + 2, 14 + 2 * n_containers)
    low_w = max(18, max_w + 2, 14 + 2 * n_containers)
    if low_h > 30 or low_w > 30:
        return None

    height = int(rng.integers(low_h, 31))
    width = int(rng.integers(low_w, 31))
    placed = _place_items(rng, items, height, width)
    if placed is None:
        return None
    inp, expected = placed

    detected = _detect_containers(inp)
    blocks = _detect_blocks(inp)
    if len(detected) != n_containers or len(blocks) != n_containers:
        return None
    if not np.array_equal(main(inp), expected):
        return None
    return inp, expected


def generate_input() -> np.ndarray:
    for _ in range(400):
        rng = np.random.default_rng()
        built = _build_example(rng)
        if built is not None:
            return built[0]
    raise RuntimeError("failed to generate a valid input")
