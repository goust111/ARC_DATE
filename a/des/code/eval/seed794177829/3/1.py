import random
from collections import Counter, deque

import numpy as np


SIDES = ("top", "right", "bottom", "left")

# --------------- shared helpers ---------------


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _find_components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _bbox_shape(bbox: tuple[int, int, int, int]) -> tuple[int, int]:
    r0, c0, r1, c1 = bbox
    return r1 - r0 + 1, c1 - c0 + 1


def _border_points(bbox: tuple[int, int, int, int]) -> set[tuple[int, int]]:
    r0, c0, r1, c1 = bbox
    pts = set()
    for c in range(c0, c1 + 1):
        pts.add((r0, c))
        pts.add((r1, c))
    for r in range(r0, r1 + 1):
        pts.add((r, c0))
        pts.add((r, c1))
    return pts


def _anchor_side(anchor: tuple[int, int], bbox: tuple[int, int, int, int]) -> str | None:
    r, c = anchor
    r0, c0, r1, c1 = bbox
    if r == r0 and c0 < c < c1:
        return "top"
    if r == r1 and c0 < c < c1:
        return "bottom"
    if c == c0 and r0 < r < r1:
        return "left"
    if c == c1 and r0 < r < r1:
        return "right"
    return None


def _child_offset(cavity_shape: tuple[int, int], child_shape: tuple[int, int], side: str) -> tuple[int, int]:
    cav_h, cav_w = cavity_shape
    ch, cw = child_shape
    extra_r = cav_h - ch
    extra_c = cav_w - cw
    assert extra_r >= 0 and extra_c >= 0, "child must fit inside parent cavity"
    if side == "top":
        return 0, extra_c // 2
    if side == "bottom":
        return extra_r, extra_c // 2
    if side == "left":
        return extra_r // 2, 0
    if side == "right":
        return extra_r // 2, extra_c
    raise AssertionError(f"unknown side: {side}")


def _assemble_layer(frames: list[dict]) -> np.ndarray:
    child = None
    for frame in reversed(frames):
        h, w = frame["outer_shape"]
        layer = np.full((h, w), int(frame["frame_color"]), dtype=int)
        if child is not None:
            dr, dc = _child_offset(frame["cavity_shape"], child.shape, frame["side"])
            layer[1 + dr : 1 + dr + child.shape[0], 1 + dc : 1 + dc + child.shape[1]] = child
        child = layer
    assert child is not None
    return child


def _render_output(grid_shape: tuple[int, int], bg: int, frames: list[dict]) -> np.ndarray:
    out = np.full(grid_shape, int(bg), dtype=int)
    assembled = _assemble_layer(frames)
    outer = frames[0]
    r0, c0, r1, c1 = outer["bbox"]
    assert assembled.shape == (r1 - r0 + 1, c1 - c0 + 1)
    out[r0 : r1 + 1, c0 : c1 + 1] = assembled
    return out


def _parse_frame(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    bbox = _bbox(coords)
    h, w = _bbox_shape(bbox)
    if h < 4 or w < 4:
        return None

    coord_set = set(coords)
    border = _border_points(bbox)
    if coord_set != border:
        return None

    r0, c0, r1, c1 = bbox
    interior = grid[r0 + 1 : r1, c0 + 1 : c1]
    if interior.size == 0 or not np.all(interior == bg):
        return None

    colors = [int(grid[r, c]) for r, c in coords]
    counts = Counter(colors)
    if len(counts) != 2:
        return None

    frame_color, frame_count = counts.most_common(1)[0]
    if frame_count != len(coords) - 1:
        return None

    anchors = [(r, c) for r, c in coords if int(grid[r, c]) != int(frame_color)]
    if len(anchors) != 1:
        return None

    side = _anchor_side(anchors[0], bbox)
    if side is None:
        return None

    cavity_shape = (h - 2, w - 2)
    return {
        "bbox": bbox,
        "outer_shape": (h, w),
        "cavity_shape": cavity_shape,
        "cavity_area": cavity_shape[0] * cavity_shape[1],
        "frame_color": int(frame_color),
        "anchor_color": int(grid[anchors[0]]),
        "anchor": anchors[0],
        "side": side,
    }


def _parse_block(grid: np.ndarray, coords: list[tuple[int, int]], bg: int) -> dict | None:
    bbox = _bbox(coords)
    h, w = _bbox_shape(bbox)
    r0, c0, r1, c1 = bbox
    if len(coords) != h * w:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1]
    colors = np.unique(patch)
    if len(colors) != 1 or int(colors[0]) == int(bg):
        return None
    return {
        "bbox": bbox,
        "shape": (h, w),
        "color": int(colors[0]),
    }


def _extract_objects(grid: np.ndarray) -> tuple[int, list[dict], list[dict]]:
    bg = _get_bg(grid)
    comps = _find_components(grid != bg)
    frames: list[dict] = []
    blocks: list[dict] = []

    for coords in comps:
        frame = _parse_frame(grid, coords, bg)
        if frame is not None:
            frames.append(frame)
            continue
        block = _parse_block(grid, coords, bg)
        if block is not None:
            blocks.append(block)
            continue
        raise AssertionError("every foreground component must be either a frame or a solid block")

    return bg, frames, blocks


def main(grid: np.ndarray) -> np.ndarray:
    """
    Assemble scattered hollow frames and matching fill blocks into a nested stack.

    Each frame contributes a solid fill in its own color (via its matching cavity-sized
    block), then the next smaller assembled frame is inserted into that filled cavity and
    aligned flush against the wall indicated by the frame's anchor side.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2, "expected a 2D grid"

    bg, frames, blocks = _extract_objects(g)
    assert len(frames) >= 2, "expected at least two frames"
    assert len(frames) == len(blocks), "expected exactly one solid block per frame"

    frames.sort(key=lambda item: item["cavity_area"], reverse=True)
    cavity_shapes = [frame["cavity_shape"] for frame in frames]
    assert len(set(cavity_shapes)) == len(cavity_shapes), "cavity shapes must be unique"

    block_by_shape: dict[tuple[int, int], dict] = {}
    for block in blocks:
        key = tuple(block["shape"])
        assert key not in block_by_shape, "block shapes must be unique"
        block_by_shape[key] = block

    for frame in frames:
        assert frame["cavity_shape"] in block_by_shape, "every frame cavity must have a matching block"

    return _render_output(g.shape, bg, frames)


# --------------- generator ---------------

BG = 0
GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_COUNT_CHOICES = (2, 3)
INNER_OUTER_H_RANGE = (4, 7)
INNER_OUTER_W_RANGE = (4, 7)
PRIMARY_EXTRA_RANGE = (1, 4)
SECONDARY_EXTRA_RANGE = (0, 4)
PLACEMENT_GAP = 1
PLACEMENT_TRIES = 200
GEN_RETRIES = 400


def _objects_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = PLACEMENT_GAP) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _pick_colors(rng: random.Random, n_frames: int) -> tuple[list[int], list[int], list[int]]:
    colors = list(range(1, 10))
    rng.shuffle(colors)
    chosen = colors[: 3 * n_frames]
    return chosen[:n_frames], chosen[n_frames : 2 * n_frames], chosen[2 * n_frames :]


def _make_frame_specs(rng: random.Random) -> list[dict]:
    n_frames = rng.choice(FRAME_COUNT_CHOICES)
    frame_colors, block_colors, anchor_colors = _pick_colors(rng, n_frames)

    sides_inner = [rng.choice(SIDES) for _ in range(n_frames)]
    shapes_inner: list[tuple[int, int]] = [
        (
            rng.randint(*INNER_OUTER_H_RANGE),
            rng.randint(*INNER_OUTER_W_RANGE),
        )
    ]
    cur_h, cur_w = shapes_inner[0]
    for i in range(1, n_frames):
        side = sides_inner[i]
        if side in ("top", "bottom"):
            cav_h = cur_h + rng.randint(*PRIMARY_EXTRA_RANGE)
            cav_w = cur_w + rng.randint(*SECONDARY_EXTRA_RANGE)
        else:
            cav_h = cur_h + rng.randint(*SECONDARY_EXTRA_RANGE)
            cav_w = cur_w + rng.randint(*PRIMARY_EXTRA_RANGE)
        cur_h, cur_w = cav_h + 2, cav_w + 2
        shapes_inner.append((cur_h, cur_w))

    frames: list[dict] = []
    for idx, (shape, side) in enumerate(zip(reversed(shapes_inner), reversed(sides_inner), strict=True)):
        h, w = shape
        frames.append(
            {
                "outer_shape": (h, w),
                "cavity_shape": (h - 2, w - 2),
                "cavity_area": (h - 2) * (w - 2),
                "side": side,
                "frame_color": frame_colors[idx],
                "block_color": block_colors[idx],
                "anchor_color": anchor_colors[idx],
            }
        )
    return frames


def _place_objects(
    rng: random.Random,
    grid_shape: tuple[int, int],
    frames: list[dict],
) -> dict[tuple[str, int], tuple[int, int]]:
    gh, gw = grid_shape
    objects: list[tuple[tuple[str, int], tuple[int, int]]] = []
    for i, frame in enumerate(frames):
        objects.append((("frame", i), frame["outer_shape"]))
        objects.append((("block", i), frame["cavity_shape"]))
    objects.sort(key=lambda item: item[1][0] * item[1][1], reverse=True)

    placements: dict[tuple[str, int], tuple[int, int]] = {}
    occupied: list[tuple[int, int, int, int]] = []

    for key, (h, w) in objects:
        if gh - h - 2 < 0 or gw - w - 2 < 0:
            raise AssertionError("grid too small for object placement")
        placed = False
        for _ in range(PLACEMENT_TRIES):
            r0 = rng.randint(1, gh - h - 1)
            c0 = rng.randint(1, gw - w - 1)
            bbox = (r0, c0, r0 + h - 1, c0 + w - 1)
            if any(_objects_too_close(bbox, other) for other in occupied):
                continue
            placements[key] = (r0, c0)
            occupied.append(bbox)
            placed = True
            break
        if not placed:
            raise RuntimeError("failed to place all objects")

    return placements


def _anchor_position(side: str, top: int, left: int, shape: tuple[int, int], rng: random.Random) -> tuple[int, int]:
    h, w = shape
    if side == "top":
        return top, left + rng.randint(1, w - 2)
    if side == "bottom":
        return top + h - 1, left + rng.randint(1, w - 2)
    if side == "left":
        return top + rng.randint(1, h - 2), left
    if side == "right":
        return top + rng.randint(1, h - 2), left + w - 1
    raise AssertionError(f"unknown side: {side}")


def _draw_frame(grid: np.ndarray, top: int, left: int, frame: dict, rng: random.Random) -> tuple[int, int, int, int]:
    h, w = frame["outer_shape"]
    color = int(frame["frame_color"])
    grid[top, left : left + w] = color
    grid[top + h - 1, left : left + w] = color
    grid[top : top + h, left] = color
    grid[top : top + h, left + w - 1] = color
    anchor = _anchor_position(frame["side"], top, left, frame["outer_shape"], rng)
    grid[anchor] = int(frame["anchor_color"])
    return top, left, top + h - 1, left + w - 1


def _render_input(
    grid_shape: tuple[int, int],
    frames: list[dict],
    placements: dict[tuple[str, int], tuple[int, int]],
    rng: random.Random,
) -> np.ndarray:
    grid = np.full(grid_shape, BG, dtype=int)
    for i, frame in enumerate(frames):
        fr, fc = placements[("frame", i)]
        frame["bbox"] = _draw_frame(grid, fr, fc, frame, rng)
        br, bc = placements[("block", i)]
        bh, bw = frame["cavity_shape"]
        grid[br : br + bh, bc : bc + bw] = int(frame["block_color"])
    return grid


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frames = _make_frame_specs(rng)
    outer_h, outer_w = frames[0]["outer_shape"]

    min_h = max(GRID_H_RANGE[0], outer_h + 4)
    min_w = max(GRID_W_RANGE[0], outer_w + 4)
    if min_h > GRID_H_RANGE[1] or min_w > GRID_W_RANGE[1]:
        return None

    gh = rng.randint(min_h, GRID_H_RANGE[1])
    gw = rng.randint(min_w, GRID_W_RANGE[1])

    try:
        placements = _place_objects(rng, (gh, gw), frames)
    except RuntimeError:
        return None

    grid = _render_input((gh, gw), frames, placements, rng)
    if _get_bg(grid) != BG:
        return None

    expected = _render_output(grid.shape, BG, frames)
    try:
        actual = main(grid)
    except AssertionError:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("generate_input: exhausted retries")
