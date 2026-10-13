from __future__ import annotations

from collections import Counter
import numpy as np


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _components_of_color(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                cr, cc = stack.pop()
                cells.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(cells, dtype=int))
    return comps


def _extract_objects(grid: np.ndarray, bg: int) -> list[dict]:
    objects: list[dict] = []
    for color in np.unique(grid):
        color = int(color)
        if color == bg:
            continue
        for coords in _components_of_color(grid == color):
            r0 = int(np.min(coords[:, 0]))
            r1 = int(np.max(coords[:, 0]))
            c0 = int(np.min(coords[:, 1]))
            c1 = int(np.max(coords[:, 1]))
            sprite = np.full((r1 - r0 + 1, c1 - c0 + 1), bg, dtype=int)
            sprite[coords[:, 0] - r0, coords[:, 1] - c0] = color
            objects.append(
                {
                    "color": color,
                    "coords": coords,
                    "sprite": sprite,
                    "bbox": (r0, c0, r1, c1),
                    "bbox_area": int(sprite.shape[0] * sprite.shape[1]),
                    "pixel_area": int(coords.shape[0]),
                }
            )
    return objects


def _center_offsets(outer_shape: tuple[int, int], inner_shape: tuple[int, int]) -> tuple[int, int]:
    oh, ow = outer_shape
    ih, iw = inner_shape
    return (oh - ih) // 2, (ow - iw) // 2


def _paste_sprite(canvas: np.ndarray, sprite: np.ndarray, bg: int) -> None:
    r0, c0 = _center_offsets(canvas.shape, sprite.shape)
    mask = sprite != bg
    canvas[r0 : r0 + sprite.shape[0], c0 : c0 + sprite.shape[1]][mask] = sprite[mask]


def main(grid: np.ndarray) -> np.ndarray:
    """
    Sort scattered sprites by bounding-box area and center-stack them into a nested composition.

    The output canvas is the largest bounding box enlarged by two pixels per additional object.
    Objects are pasted from largest bbox-area to smallest, so the smallest appears on top.
    """
    g = np.asarray(grid, dtype=int)
    bg = _get_bg(g)
    objects = _extract_objects(g, bg)
    assert objects, "expected at least one foreground object"

    bbox_areas = [obj["bbox_area"] for obj in objects]
    assert len(bbox_areas) == len(set(bbox_areas)), "bbox areas must be unique"

    ordered = sorted(objects, key=lambda obj: obj["bbox_area"], reverse=True)
    largest_h, largest_w = ordered[0]["sprite"].shape
    layers = len(ordered) - 1
    out_h = largest_h + 2 * layers
    out_w = largest_w + 2 * layers
    out = np.full((out_h, out_w), bg, dtype=int)

    for obj in ordered:
        _paste_sprite(out, obj["sprite"], bg)
    return out


_GRID_H_RANGE = (18, 26)
_GRID_W_RANGE = (18, 26)
_OBJECT_COUNT_CHOICES = (3, 4, 5)
_BBOX_HEIGHT_RANGE = (2, 7)
_BBOX_WIDTH_RANGE = (2, 7)
_PLACEMENT_PADDING = 1
_GEN_RETRIES = 400
_PALETTE = tuple(range(10))


def _random_monotone_path_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    moves = ["D"] * (h - 1) + ["R"] * (w - 1)
    rng.shuffle(moves)
    r = c = 0
    mask[r, c] = True
    for move in moves:
        if move == "D":
            r += 1
        else:
            c += 1
        mask[r, c] = True

    additions = int(rng.integers(0, max(1, (h * w) // 4) + 1))
    for _ in range(additions):
        frontier = []
        for rr in range(h):
            for cc in range(w):
                if mask[rr, cc]:
                    continue
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc]:
                        frontier.append((rr, cc))
                        break
        if not frontier:
            break
        rr, cc = frontier[int(rng.integers(len(frontier)))]
        mask[rr, cc] = True
    return mask


def _orient(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    out = np.rot90(mask, int(rng.integers(4)))
    if bool(rng.integers(2)):
        out = np.fliplr(out)
    if bool(rng.integers(2)):
        out = np.flipud(out)
    return out.copy()


def _make_shape_mask(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    styles = ["solid", "ell", "tee", "plus", "path"]
    if h >= 3 and w >= 3:
        styles.append("frame")
    style = styles[int(rng.integers(len(styles)))]

    if style == "solid":
        return np.ones((h, w), dtype=bool)
    if style == "frame":
        mask = np.ones((h, w), dtype=bool)
        mask[1:-1, 1:-1] = False
        return mask
    if style == "plus":
        mask = np.zeros((h, w), dtype=bool)
        mask[h // 2, :] = True
        mask[:, w // 2] = True
        return mask
    if style == "ell":
        base = np.zeros((h, w), dtype=bool)
        base[0, :] = True
        base[:, 0] = True
        return _orient(base, rng)
    if style == "tee":
        base = np.zeros((h, w), dtype=bool)
        base[0, :] = True
        base[:, w // 2] = True
        return _orient(base, rng)
    return _orient(_random_monotone_path_mask(h, w, rng), rng)


def _sample_bbox_specs(n: int, rng: np.random.Generator) -> list[tuple[int, int]]:
    candidates = [
        (h, w)
        for h in range(_BBOX_HEIGHT_RANGE[0], _BBOX_HEIGHT_RANGE[1] + 1)
        for w in range(_BBOX_WIDTH_RANGE[0], _BBOX_WIDTH_RANGE[1] + 1)
    ]
    rng.shuffle(candidates)
    used_areas: set[int] = set()
    specs: list[tuple[int, int]] = []
    for h, w in candidates:
        area = h * w
        if area in used_areas:
            continue
        specs.append((h, w))
        used_areas.add(area)
        if len(specs) == n:
            break
    if len(specs) != n:
        raise RuntimeError("unable to sample distinct bbox areas")
    return specs


def _expanded_rect_slice(r0: int, c0: int, h: int, w: int, H: int, W: int) -> tuple[slice, slice]:
    return (
        slice(max(0, r0 - _PLACEMENT_PADDING), min(H, r0 + h + _PLACEMENT_PADDING)),
        slice(max(0, c0 - _PLACEMENT_PADDING), min(W, c0 + w + _PLACEMENT_PADDING)),
    )


def _place_objects(
    grid: np.ndarray,
    patches: list[np.ndarray],
    colors: list[int],
    rng: np.random.Generator,
) -> bool:
    H, W = grid.shape
    occupied = np.zeros((H, W), dtype=bool)
    order = sorted(range(len(patches)), key=lambda i: patches[i].shape[0] * patches[i].shape[1], reverse=True)

    for idx in order:
        patch = patches[idx]
        h, w = patch.shape
        positions = []
        for r0 in range(H - h + 1):
            for c0 in range(W - w + 1):
                sr, sc = _expanded_rect_slice(r0, c0, h, w, H, W)
                if occupied[sr, sc].any():
                    continue
                positions.append((r0, c0))
        if not positions:
            return False
        r0, c0 = positions[int(rng.integers(len(positions)))]
        color = colors[idx]
        grid[r0 : r0 + h, c0 : c0 + w][patch] = color
        sr, sc = _expanded_rect_slice(r0, c0, h, w, H, W)
        occupied[sr, sc] = True
    return True


def _oracle_valid(grid: np.ndarray) -> bool:
    bg = _get_bg(grid)
    objects = _extract_objects(grid, bg)
    if len(objects) < 3 or len(objects) > 5:
        return False
    if len({obj["color"] for obj in objects}) != len(objects):
        return False
    bbox_areas = [obj["bbox_area"] for obj in objects]
    if len(set(bbox_areas)) != len(bbox_areas):
        return False
    try:
        out = main(grid)
    except AssertionError:
        return False
    return isinstance(out, np.ndarray) and out.ndim == 2


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    H = int(rng.integers(_GRID_H_RANGE[0], _GRID_H_RANGE[1] + 1))
    W = int(rng.integers(_GRID_W_RANGE[0], _GRID_W_RANGE[1] + 1))
    n = int(rng.choice(_OBJECT_COUNT_CHOICES))
    bg = 0
    grid = np.full((H, W), bg, dtype=int)

    specs = _sample_bbox_specs(n, rng)
    patches = [_make_shape_mask(h, w, rng) for h, w in specs]
    colors = list(rng.choice(np.array([c for c in _PALETTE if c != bg], dtype=int), size=n, replace=False))

    total_bbox_area = sum(h * w for h, w in specs)
    if total_bbox_area > (H * W) // 2:
        return None

    if not _place_objects(grid, patches, colors, rng):
        return None
    if not _oracle_valid(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(_GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate valid input after {_GEN_RETRIES} attempts")
