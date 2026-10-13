from __future__ import annotations

import random
from collections import Counter

import numpy as np


BG = 0
DIR4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
TRANSFORMS = (
    "id",
    "rot90",
    "rot180",
    "rot270",
    "flip_h",
    "flip_v",
    "transpose",
    "anti_transpose",
)
PRESERVE_TRANSFORMS = ("id", "rot180", "flip_h", "flip_v")
SWAP_TRANSFORMS = ("rot90", "rot270", "transpose", "anti_transpose")
GEN_ATTEMPTS = 500


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, np.asarray(grid).ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def crop_bbox(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    return np.asarray(grid)[r0 : r1 + 1, c0 : c1 + 1].copy()


def find_components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components: list[np.ndarray] = []
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
                for dr, dc in DIR4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            components.append(np.array(coords, dtype=int))
    return components


def _transform_array(arr: np.ndarray, name: str) -> np.ndarray:
    if name == "id":
        return arr.copy()
    if name == "rot90":
        return np.rot90(arr, 1).copy()
    if name == "rot180":
        return np.rot90(arr, 2).copy()
    if name == "rot270":
        return np.rot90(arr, 3).copy()
    if name == "flip_h":
        return np.flipud(arr).copy()
    if name == "flip_v":
        return np.fliplr(arr).copy()
    if name == "transpose":
        return arr.T.copy()
    if name == "anti_transpose":
        return np.flipud(np.fliplr(arr.T)).copy()
    raise ValueError(f"unknown transform: {name}")


def _signature_cross(signature: tuple[int, int, int, int]) -> np.ndarray:
    top, right, bottom, left = map(int, signature)
    cross = np.zeros((3, 3), dtype=int)
    cross[0, 1] = top
    cross[1, 2] = right
    cross[2, 1] = bottom
    cross[1, 0] = left
    return cross


def _transform_signature(signature: tuple[int, int, int, int], name: str) -> tuple[int, int, int, int]:
    cross = _transform_array(_signature_cross(signature), name)
    return int(cross[0, 1]), int(cross[1, 2]), int(cross[2, 1]), int(cross[1, 0])


def _perimeter_mask(shape: tuple[int, int]) -> np.ndarray:
    h, w = shape
    out = np.zeros((h, w), dtype=bool)
    out[0, :] = True
    out[-1, :] = True
    out[:, 0] = True
    out[:, -1] = True
    return out


def _frame_signature(grid: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    r0, c0, r1, c1 = bbox
    top_vals = {int(v) for v in grid[r0, c0 : c1 + 1]}
    bottom_vals = {int(v) for v in grid[r1, c0 : c1 + 1]}
    left_vals = {int(v) for v in grid[r0 + 1 : r1, c0]}
    right_vals = {int(v) for v in grid[r0 + 1 : r1, c1]}
    if min(len(top_vals), len(bottom_vals), len(left_vals), len(right_vals)) != 1:
        raise AssertionError("frame sides must each have a single visible color")
    signature = (
        next(iter(top_vals)),
        next(iter(right_vals)),
        next(iter(bottom_vals)),
        next(iter(left_vals)),
    )
    if len(set(signature)) != 4:
        raise AssertionError("frame side colors must be pairwise distinct")
    return signature


def _find_frames(grid: np.ndarray, bg: int) -> list[dict]:
    frames = []
    for coords in find_components(grid != bg):
        bbox = bbox_coords(coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        local = np.zeros((h, w), dtype=bool)
        local[coords[:, 0] - r0, coords[:, 1] - c0] = True
        if not np.array_equal(local, _perimeter_mask((h, w))):
            continue
        frames.append(
            {
                "bbox": bbox,
                "signature": _frame_signature(grid, bbox),
                "interior": grid[r0 + 1 : r1, c0 + 1 : c1].copy(),
            }
        )
    frames.sort(key=lambda item: item["bbox"])
    if not frames:
        raise AssertionError("expected at least one rectangular frame")
    return frames


def _match_transform(
    sample_signature: tuple[int, int, int, int],
    sample_content: np.ndarray,
    target_signature: tuple[int, int, int, int],
    target_shape: tuple[int, int],
) -> tuple[str, np.ndarray]:
    matches = []
    for name in TRANSFORMS:
        if _transform_signature(sample_signature, name) != target_signature:
            continue
        patch = _transform_array(sample_content, name)
        if patch.shape == target_shape:
            matches.append((name, patch))
    if len(matches) != 1:
        raise AssertionError("target frame must match exactly one pose")
    return matches[0]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Decode each empty frame's border-color pose and stamp the transformed sample content into it.

    Input structure:
    - A black background contains several one-cell-thick rectangular frames.
    - Each frame uses the same four distinct border colors, arranged as a visible
      `(top, right, bottom, left)` pose code; corners belong to the top or bottom rows.
    - Exactly one frame contains a multicolor sample pattern inside it, while every
      other frame has an empty black interior of the same size or the transposed size.

    Rule:
    - Read the sample frame's side-color order.
    - For every empty frame, find the unique rotation or mirror transform that maps
      the sample side order to that frame's side order.
    - Apply the same transform to the sample interior and write that transformed
      pattern into the empty frame, leaving all border pixels unchanged.

    Key insight:
    - The border colors are not decoration; their side order is a pose code that
      tells you exactly how the sample content must be rotated or mirrored.
    """
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    frames = _find_frames(g, bg)
    sample_frames = [frame for frame in frames if np.any(frame["interior"] != bg)]
    if len(sample_frames) != 1:
        raise AssertionError("expected exactly one sample frame")
    sample = sample_frames[0]
    sample_signature = sample["signature"]
    sample_content = sample["interior"]

    out = g.copy()
    for frame in frames:
        if frame is sample:
            continue
        if np.any(frame["interior"] != bg):
            raise AssertionError("non-sample frames must start empty")
        _name, patch = _match_transform(
            sample_signature,
            sample_content,
            frame["signature"],
            frame["interior"].shape,
        )
        r0, c0, r1, c1 = frame["bbox"]
        out[r0 + 1 : r1, c0 + 1 : c1] = patch
    return out


def _content_variants(content: np.ndarray) -> set[tuple[tuple[int, int], bytes]]:
    return {(tuple(_transform_array(content, name).shape), _transform_array(content, name).tobytes()) for name in TRANSFORMS}


def _generate_content(interior_h: int, interior_w: int, palette: list[int]) -> np.ndarray:
    inner_cells = [(r, c) for r in range(1, interior_h - 1) for c in range(1, interior_w - 1)]
    inner_area = len(inner_cells)
    for _ in range(600):
        content = np.zeros((interior_h, interior_w), dtype=int)
        color_count = random.randint(2, min(4, len(palette)))
        colors = random.sample(palette, color_count)
        lo = max(4, inner_area // 3)
        hi = max(lo, min(inner_area, (2 * inner_area + 1) // 3))
        picked = random.sample(inner_cells, random.randint(lo, hi))
        offset = random.randrange(32)
        for idx, (r, c) in enumerate(picked):
            content[r, c] = colors[(2 * r + 3 * c + idx + offset) % color_count]

        fg = np.argwhere(content != BG)
        if len(fg) < 4 or len({int(v) for v in content[content != BG]}) < 2:
            continue
        r0, c0, r1, c1 = bbox_coords(fg)
        if r1 == r0 or c1 == c0:
            continue
        if len(_content_variants(content)) != 8:
            continue
        return content
    raise RuntimeError("failed to generate asymmetric sample content")


def _choose_target_transforms() -> list[str]:
    target_count = random.randint(2, 4)
    chosen = [random.choice(PRESERVE_TRANSFORMS), random.choice(SWAP_TRANSFORMS)]
    pool = [name for name in TRANSFORMS if name not in chosen]
    random.shuffle(pool)
    while len(chosen) < target_count:
        chosen.append(pool.pop())
    random.shuffle(chosen)
    return chosen


def _boxes_intersect(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    return not (a[2] < b[0] or b[2] < a[0] or a[3] < b[1] or b[3] < a[1])


def _place_boxes(sizes: list[tuple[int, int]], grid_h: int, grid_w: int, gap: int = 2) -> list[tuple[int, int]] | None:
    order = sorted(range(len(sizes)), key=lambda i: sizes[i][0] * sizes[i][1], reverse=True)
    positions: list[tuple[int, int] | None] = [None] * len(sizes)
    blocked: list[tuple[int, int, int, int]] = []
    for idx in order:
        box_h, box_w = sizes[idx]
        for _ in range(800):
            r0 = random.randint(0, grid_h - box_h)
            c0 = random.randint(0, grid_w - box_w)
            expanded = (r0 - gap, c0 - gap, r0 + box_h - 1 + gap, c0 + box_w - 1 + gap)
            if any(_boxes_intersect(expanded, other) for other in blocked):
                continue
            positions[idx] = (r0, c0)
            blocked.append(expanded)
            break
        else:
            return None
    return [pos for pos in positions if pos is not None]


def _paint_frame(
    grid: np.ndarray,
    top_left: tuple[int, int],
    interior: np.ndarray,
    signature: tuple[int, int, int, int],
) -> None:
    r0, c0 = top_left
    h, w = interior.shape
    r1 = r0 + h + 1
    c1 = c0 + w + 1
    top, right, bottom, left = signature
    grid[r0, c0 : c1 + 1] = top
    grid[r1, c0 : c1 + 1] = bottom
    grid[r0 + 1 : r1, c0] = left
    grid[r0 + 1 : r1, c1] = right
    grid[r0 + 1 : r1, c0 + 1 : c1] = interior


def _build_attempt() -> np.ndarray | None:
    square = random.random() < 0.35
    if square:
        interior_h = interior_w = random.randint(4, 7)
    else:
        interior_h = random.randint(4, 8)
        interior_w = random.randint(4, 8)
        if interior_h == interior_w:
            interior_w = 4 + (interior_w % 5)
            if interior_w == interior_h:
                interior_w = 8 if interior_h != 8 else 4

    edge_colors = random.sample(list(range(1, 10)), 4)
    content_palette = [color for color in range(1, 10) if color not in edge_colors]
    sample_signature = tuple(random.sample(edge_colors, 4))
    sample_content = _generate_content(interior_h, interior_w, content_palette)
    target_transforms = _choose_target_transforms()

    target_contents = [_transform_array(sample_content, name) for name in target_transforms]
    target_signatures = [_transform_signature(sample_signature, name) for name in target_transforms]
    sizes = [(interior_h + 2, interior_w + 2)] + [(patch.shape[0] + 2, patch.shape[1] + 2) for patch in target_contents]
    total_area = sum(h * w for h, w in sizes)

    for _ in range(120):
        grid_h = random.randint(max(max(h for h, _ in sizes) + 2, 16), 30)
        grid_w = random.randint(max(max(w for _, w in sizes) + 2, 16), 30)
        if grid_h * grid_w < total_area * 3:
            continue
        positions = _place_boxes(sizes, grid_h, grid_w, gap=2)
        if positions is None:
            continue

        grid = np.zeros((grid_h, grid_w), dtype=int)
        _paint_frame(grid, positions[0], sample_content, sample_signature)
        for pos, signature, patch in zip(positions[1:], target_signatures, target_contents):
            _paint_frame(grid, pos, np.zeros_like(patch), signature)
        if _validate_generated_input(grid):
            return grid
    return None


def _validate_generated_input(grid: np.ndarray) -> bool:
    if not isinstance(grid, np.ndarray) or grid.ndim != 2 or max(grid.shape) > 30:
        return False
    bg = get_bg(grid)
    frames = _find_frames(grid, bg)
    sample_frames = [frame for frame in frames if np.any(frame["interior"] != bg)]
    if len(sample_frames) != 1 or len(frames) < 3:
        return False
    sample = sample_frames[0]
    if len(_content_variants(sample["interior"])) != 8:
        return False
    seen_transforms = set()
    for frame in frames:
        if frame is sample:
            continue
        if np.any(frame["interior"] != bg):
            return False
        try:
            transform_name, patch = _match_transform(
                sample["signature"],
                sample["interior"],
                frame["signature"],
                frame["interior"].shape,
            )
        except AssertionError:
            return False
        seen_transforms.add(transform_name)
        if patch.shape != frame["interior"].shape:
            return False
    return len(seen_transforms) >= 2


def generate_input() -> np.ndarray:
    for _ in range(GEN_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid edge-frame pose puzzle")
