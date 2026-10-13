import random
from collections import deque

import numpy as np


BG = 0
FILL_COLOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
MAX_GENERATION_ATTEMPTS = 400


def _bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def _crop(mask: np.ndarray) -> np.ndarray:
    box = _bbox(mask)
    assert box is not None, "expected a non-empty mask to crop"
    r0, c0, r1, c1 = box
    return mask[r0 : r1 + 1, c0 : c1 + 1].copy()


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            seen[r, c] = True
            q = deque([(r, c)])
            comp: list[tuple[int, int]] = []
            while q:
                rr, cc = q.popleft()
                comp.append((rr, cc))
                for nr, nc in ((rr - 1, cc), (rr + 1, cc), (rr, cc - 1), (rr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _rectangular_frame_present(region: np.ndarray, color: int) -> bool:
    if region.shape[0] < 3 or region.shape[1] < 3:
        return False
    return bool(
        np.all(region[0, :] == color)
        and np.all(region[-1, :] == color)
        and np.all(region[:, 0] == color)
        and np.all(region[:, -1] == color)
    )


def _count_enclosed_bg_components(region: np.ndarray, bg: int) -> int:
    comps = _components(region == bg)
    h, w = region.shape
    count = 0
    for comp in comps:
        touches_border = any(r == 0 or r == h - 1 or c == 0 or c == w - 1 for r, c in comp)
        if not touches_border:
            count += 1
    return count


def _extract_frames(grid: np.ndarray, bg: int = BG) -> list[dict]:
    frames: list[dict] = []
    for color in sorted(int(v) for v in np.unique(grid) if int(v) != bg):
        mask = grid == color
        box = _bbox(mask)
        if box is None:
            continue
        r0, c0, r1, c1 = box
        region = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not _rectangular_frame_present(region, color):
            continue
        inner = region[1:-1, 1:-1]
        motif_mask = inner == color
        frames.append(
            {
                "color": color,
                "bbox": box,
                "region": region,
                "inner_shape": inner.shape,
                "motif_mask": motif_mask,
                "enclosed_bg_components": _count_enclosed_bg_components(region, bg),
            }
        )
    return frames


def _tile_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    h, w = mask.shape
    out_h, out_w = shape
    reps_h = (out_h + h - 1) // h
    reps_w = (out_w + w - 1) // w
    return np.tile(mask, (reps_h, reps_w))[:out_h, :out_w]


def main(grid: np.ndarray) -> np.ndarray:
    """Find the only framed rectangle whose inner same-color barrier splits its cavity, then crop and tile that barrier into the other frames.

    Input structure:
    - The grid has a black background and 3-5 separated single-color hollow rectangular frames.
    - All frames have the same interior size; exactly one frame also contains extra same-color pixels inside its cavity.

    Rule:
    - Detect the rectangular frames by their full monochrome outer borders.
    - Select the unique frame whose full bounding box contains more than one enclosed background component, meaning its inner same-color pixels split the cavity.
    - Extract only those inner same-color pixels, crop them to their tight bounding box, and use that cropped binary motif as a tile.
    - Leave the source frame unchanged, and fill every other frame interior by repeating the cropped motif from the interior's top-left corner in grey.

    Key insight:
    - The source is identified by enclosure structure, but the propagated content is the cropped geometry of the inner barrier itself.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"
    bg = BG

    frames = _extract_frames(g, bg)
    assert len(frames) >= 2, "expected multiple framed objects"

    source_candidates = [frame for frame in frames if frame["motif_mask"].any() and frame["enclosed_bg_components"] > 1]
    if len(source_candidates) != 1:
        source_candidates = [frame for frame in frames if frame["motif_mask"].any()]
    assert len(source_candidates) == 1, "expected exactly one source frame"
    source = source_candidates[0]

    motif = _crop(source["motif_mask"])
    out = g.copy()
    for frame in frames:
        if frame["bbox"] == source["bbox"]:
            continue
        r0, c0, r1, c1 = frame["bbox"]
        fill_mask = _tile_mask(motif, frame["inner_shape"])
        inner = out[r0 + 1 : r1, c0 + 1 : c1]
        inner[fill_mask] = FILL_COLOR
    return out


def _make_empty_frame(color: int, inner_h: int, inner_w: int) -> np.ndarray:
    patch = np.zeros((inner_h + 2, inner_w + 2), dtype=int)
    patch[0, :] = color
    patch[-1, :] = color
    patch[:, 0] = color
    patch[:, -1] = color
    return patch


def _source_vertical_mask() -> np.ndarray:
    base = np.array(
        [
            [1, 0],
            [1, 0],
            [1, 1],
            [0, 1],
        ],
        dtype=bool,
    )
    variant = base.copy()
    if random.random() < 0.5:
        variant = np.fliplr(variant)
    if random.random() < 0.5:
        variant = np.flipud(variant)
    return variant


def _source_horizontal_mask() -> np.ndarray:
    variant = np.rot90(_source_vertical_mask(), 3)
    return variant.astype(bool)


def _make_source_frame(color: int, orientation: str, inner_h: int, inner_w: int) -> np.ndarray:
    patch = _make_empty_frame(color, inner_h, inner_w)
    if orientation == "vertical":
        motif = _source_vertical_mask()
        col0 = random.randint(1, inner_w - motif.shape[1] - 1)
        row0 = 0
    else:
        motif = _source_horizontal_mask()
        row0 = random.randint(1, inner_h - motif.shape[0] - 1)
        col0 = 0
    inner = patch[1:-1, 1:-1]
    inner[row0 : row0 + motif.shape[0], col0 : col0 + motif.shape[1]][motif] = color
    return patch


def _boxes_separated(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return ar1 + gap < br0 or br1 + gap < ar0 or ac1 + gap < bc0 or bc1 + gap < ac0


def _random_layout(rows: int, cols: int, patches: list[np.ndarray]) -> np.ndarray | None:
    placed_boxes: list[tuple[int, int, int, int]] = []
    placements: list[tuple[int, int]] = [(-1, -1)] * len(patches)
    order = list(range(len(patches)))
    random.shuffle(order)
    grid = np.zeros((rows, cols), dtype=int)

    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        candidates = [(r, c) for r in range(rows - h + 1) for c in range(cols - w + 1)]
        random.shuffle(candidates)
        placed = False
        for r0, c0 in candidates:
            box = (r0, c0, r0 + h - 1, c0 + w - 1)
            if all(_boxes_separated(box, prev, gap=1) for prev in placed_boxes):
                placements[idx] = (r0, c0)
                placed_boxes.append(box)
                placed = True
                break
        if not placed:
            return None

    for patch, (r0, c0) in zip(patches, placements):
        h, w = patch.shape
        region = grid[r0 : r0 + h, c0 : c0 + w]
        mask = patch != BG
        region[mask] = patch[mask]
    return grid


def _has_nontrivial_symmetry(mask: np.ndarray) -> bool:
    return bool(
        np.array_equal(mask, np.fliplr(mask))
        or np.array_equal(mask, np.flipud(mask))
        or np.array_equal(mask, np.rot90(mask, 2))
    )


def _build_attempt() -> np.ndarray | None:
    n_frames = random.randint(3, 5)
    orientation = random.choice(("vertical", "horizontal"))
    if orientation == "vertical":
        inner_h = 4
        inner_w = random.choice((4, 6))
    else:
        inner_h = random.choice((4, 6))
        inner_w = 4

    colors = random.sample(FRAME_COLORS, n_frames)
    source_index = random.randrange(n_frames)
    patches: list[np.ndarray] = []
    for idx, color in enumerate(colors):
        if idx == source_index:
            patches.append(_make_source_frame(color, orientation, inner_h, inner_w))
        else:
            patches.append(_make_empty_frame(color, inner_h, inner_w))

    outer_h, outer_w = patches[0].shape
    min_rows = max(16, outer_h * ((n_frames + 1) // 2) + 4)
    min_cols = max(16, outer_w * 2 + 4)
    if min_rows > 30 or min_cols > 30:
        return None
    rows = random.randint(min_rows, 30)
    cols = random.randint(min_cols, 30)
    grid = _random_layout(rows, cols, patches)
    if grid is None:
        return None

    frames = _extract_frames(grid, BG)
    if len(frames) != n_frames:
        return None
    inner_shapes = {frame["inner_shape"] for frame in frames}
    if len(inner_shapes) != 1:
        return None

    source_candidates = [frame for frame in frames if frame["motif_mask"].any() and frame["enclosed_bg_components"] > 1]
    if len(source_candidates) != 1:
        return None
    source = source_candidates[0]
    motif = _crop(source["motif_mask"])
    if int(motif.sum()) not in (4, 5):
        return None
    if _has_nontrivial_symmetry(motif):
        return None
    if source["motif_mask"][0, 0] or motif.shape == source["inner_shape"]:
        return None
    if not (source["inner_shape"][0] > motif.shape[0] or source["inner_shape"][1] > motif.shape[1]):
        return None

    out = main(grid)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        inp_inner = grid[r0 + 1 : r1, c0 + 1 : c1]
        out_inner = out[r0 + 1 : r1, c0 + 1 : c1]
        if frame["bbox"] == source["bbox"]:
            if not np.array_equal(inp_inner, out_inner):
                return None
            continue
        expected = np.zeros(frame["inner_shape"], dtype=int)
        expected[_tile_mask(motif, frame["inner_shape"])] = FILL_COLOR
        if not np.array_equal(out_inner, expected):
            return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _build_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid ARC input")
