import random
from collections import Counter

import numpy as np


RED = 2
INNER_SIZE = 3
FRAME_SIZE = INNER_SIZE + 2
CORNER_TO_CW = {
    "tl": 0,
    "tr": 1,
    "br": 2,
    "bl": 3,
}
CW_TO_CORNER = {value: key for key, value in CORNER_TO_CW.items()}


def get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _corner_offsets(size: int) -> dict[str, tuple[int, int]]:
    return {
        "tl": (0, 0),
        "tr": (0, size - 1),
        "br": (size - 1, size - 1),
        "bl": (size - 1, 0),
    }


def _detect_frame_groups(grid: np.ndarray, bg: int) -> list[list[dict]]:
    h, w = grid.shape
    groups: dict[tuple[int, int], list[dict]] = {}

    for size in range(3, min(h, w) + 1):
        border_mask = np.zeros((size, size), dtype=bool)
        border_mask[0, :] = True
        border_mask[-1, :] = True
        border_mask[:, 0] = True
        border_mask[:, -1] = True

        corners = _corner_offsets(size)
        corner_mask = np.zeros((size, size), dtype=bool)
        for rr, cc in corners.values():
            corner_mask[rr, cc] = True

        for r0 in range(h - size + 1):
            for c0 in range(w - size + 1):
                patch = grid[r0 : r0 + size, c0 : c0 + size]
                border = patch[border_mask]

                if np.count_nonzero(border == RED) != 1:
                    continue
                red_coords = np.argwhere((patch == RED) & border_mask)
                if red_coords.shape[0] != 1:
                    continue
                red_r, red_c = map(int, red_coords[0])
                if not corner_mask[red_r, red_c]:
                    continue

                red_corner = None
                for name, (rr, cc) in corners.items():
                    if (red_r, red_c) == (rr, cc):
                        red_corner = name
                        break
                if red_corner is None:
                    continue

                frame_colors = set(map(int, border)) - {RED}
                if len(frame_colors) != 1:
                    continue
                frame_color = frame_colors.pop()
                if frame_color == bg:
                    continue
                if not np.all((border == frame_color) | (border == RED)):
                    continue
                if np.any((patch == RED) & ~border_mask):
                    continue

                inner = patch[1:-1, 1:-1]
                if inner.size == 0:
                    continue
                if np.any((inner == RED) | (inner == frame_color)):
                    continue

                hole = bool(np.all(inner == bg))
                if not hole:
                    if np.any(inner == bg):
                        continue
                    inner_colors = set(map(int, np.unique(inner)))
                    if len(inner_colors) != 2:
                        continue
                groups.setdefault((size, frame_color), []).append(
                    {
                        "r0": int(r0),
                        "c0": int(c0),
                        "size": int(size),
                        "frame_color": int(frame_color),
                        "red_corner": red_corner,
                        "hole": hole,
                        "inner": inner.copy(),
                    }
                )

    valid_groups = []
    for group in groups.values():
        boxes = []
        overlap = False
        for frame in group:
            r0, c0, size = frame["r0"], frame["c0"], frame["size"]
            box = (r0, c0, r0 + size - 1, c0 + size - 1)
            for or0, oc0, or1, oc1 in boxes:
                if not (box[2] < or0 or or1 < box[0] or box[3] < oc0 or oc1 < box[1]):
                    overlap = True
                    break
            if overlap:
                break
            boxes.append(box)
        if overlap:
            continue
        hole_count = sum(1 for frame in group if frame["hole"])
        if len(group) >= 2 and hole_count == 1:
            valid_groups.append(group)

    valid_groups.sort(key=lambda group: (-len(group), -group[0]["size"]))
    return valid_groups


def _extract_candidate_tile(grid: np.ndarray, bg: int, frames: list[dict]) -> np.ndarray:
    mask = np.zeros(grid.shape, dtype=bool)
    size = frames[0]["size"]
    frame_color = int(frames[0]["frame_color"])
    inner_size = size - 2

    for frame in frames:
        r0, c0 = int(frame["r0"]), int(frame["c0"])
        mask[r0 : r0 + size, c0 : c0 + size] = True

    remain = (grid != bg) & ~mask
    pts = np.argwhere(remain)
    assert pts.size > 0, "candidate tile missing"

    r0, c0 = map(int, pts.min(axis=0))
    r1, c1 = map(int, pts.max(axis=0))
    if (r1 - r0 + 1, c1 - c0 + 1) != (inner_size, inner_size):
        raise AssertionError("candidate tile has wrong shape")
    if int(np.count_nonzero(remain)) != inner_size * inner_size:
        raise AssertionError("unexpected extra foreground outside frames")

    tile = grid[r0 : r1 + 1, c0 : c1 + 1].copy()
    if np.any(tile == bg):
        raise AssertionError("candidate tile must be solid")
    tile_colors = set(map(int, np.unique(tile)))
    if len(tile_colors) != 2:
        raise AssertionError("candidate tile must use exactly two colors")
    if RED in tile_colors or frame_color in tile_colors:
        raise AssertionError("candidate tile colors must differ from border colors")
    return tile


def _find_frames_and_tile(grid: np.ndarray, bg: int) -> tuple[list[dict], np.ndarray]:
    for frames in _detect_frame_groups(grid, bg):
        try:
            tile = _extract_candidate_tile(grid, bg, frames)
        except AssertionError:
            continue

        valid = True
        for frame in frames:
            rotation = CORNER_TO_CW[frame["red_corner"]]
            expected = np.rot90(tile, -rotation)
            if frame["hole"]:
                continue
            if not np.array_equal(frame["inner"], expected):
                valid = False
                break
        if valid:
            return frames, tile
    raise AssertionError("failed to identify consistent frames and candidate tile")


def main(grid: np.ndarray) -> np.ndarray:
    """
    Fill the empty framed slot by rotating the edge candidate tile according to
    the red corner marker on that frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and g.size > 0, "expected a non-empty 2D grid"

    bg = get_bg(g)
    frames, tile = _find_frames_and_tile(g, bg)
    hole = next(frame for frame in frames if frame["hole"])
    rotation = CORNER_TO_CW[hole["red_corner"]]

    out = g.copy()
    r0, c0 = int(hole["r0"]), int(hole["c0"])
    out[r0 + 1 : r0 + 1 + tile.shape[0], c0 + 1 : c0 + 1 + tile.shape[1]] = np.rot90(tile, -rotation)
    return out


def _random_tile(rng: random.Random, color_a: int, color_b: int) -> np.ndarray | None:
    for _ in range(200):
        mask = np.array([rng.random() < 0.5 for _ in range(INNER_SIZE * INNER_SIZE)], dtype=bool)
        count = int(mask.sum())
        if count < 2 or count > INNER_SIZE * INNER_SIZE - 2:
            continue
        tile = np.where(mask.reshape(INNER_SIZE, INNER_SIZE), color_a, color_b)
        rotations = {np.rot90(tile, k).tobytes() for k in range(4)}
        if len(rotations) != 4:
            continue
        return tile
    return None


def _can_place(rect: tuple[int, int, int, int], placed: list[tuple[int, int, int, int]], gap: int) -> bool:
    r0, c0, h, w = rect
    r1 = r0 + h - 1
    c1 = c0 + w - 1
    for or0, oc0, oh, ow in placed:
        or1 = or0 + oh - 1
        oc1 = oc0 + ow - 1
        if not (r1 + gap < or0 or or1 + gap < r0 or c1 + gap < oc0 or oc1 + gap < c0):
            return False
    return True


def _edge_rects(height: int, width: int) -> list[tuple[str, int, int]]:
    rects = []
    for c0 in range(width - INNER_SIZE + 1):
        rects.append(("top", 0, c0))
        rects.append(("bottom", height - INNER_SIZE, c0))
    for r0 in range(1, height - INNER_SIZE):
        rects.append(("left", r0, 0))
        rects.append(("right", r0, width - INNER_SIZE))
    return rects


def _paint_frame(
    grid: np.ndarray,
    r0: int,
    c0: int,
    frame_color: int,
    rotation: int,
    tile: np.ndarray | None,
    bg: int,
) -> None:
    grid[r0 : r0 + FRAME_SIZE, c0 : c0 + FRAME_SIZE] = bg
    grid[r0, c0 : c0 + FRAME_SIZE] = frame_color
    grid[r0 + FRAME_SIZE - 1, c0 : c0 + FRAME_SIZE] = frame_color
    grid[r0 : r0 + FRAME_SIZE, c0] = frame_color
    grid[r0 : r0 + FRAME_SIZE, c0 + FRAME_SIZE - 1] = frame_color
    corner_name = CW_TO_CORNER[int(rotation)]
    corner_r, corner_c = _corner_offsets(FRAME_SIZE)[corner_name]
    grid[r0 + corner_r, c0 + corner_c] = RED
    if tile is not None:
        grid[r0 + 1 : r0 + 1 + INNER_SIZE, c0 + 1 : c0 + 1 + INNER_SIZE] = np.rot90(tile, -rotation)


def _expected_from_meta(meta: dict) -> np.ndarray:
    out = meta["grid"].copy()
    hole_r0, hole_c0 = meta["hole_pos"]
    rotation = int(meta["hole_rotation"])
    out[hole_r0 + 1 : hole_r0 + 1 + INNER_SIZE, hole_c0 + 1 : hole_c0 + 1 + INNER_SIZE] = np.rot90(
        meta["candidate_tile"], -rotation
    )
    return out


def _gen_attempt(rng: random.Random, return_meta: bool = False):
    palette = [color for color in range(10) if color != RED]
    bg, frame_color, color_a, color_b = rng.sample(palette, 4)
    tile = _random_tile(rng, color_a, color_b)
    if tile is None:
        return None

    frame_count = rng.choice([5, 6])
    hole_rotation = rng.randrange(4)
    rotations = [0, 1, 2, 3, hole_rotation]
    if frame_count == 6:
        rotations.append(rng.randrange(4))
    rng.shuffle(rotations)

    height = rng.randint(18, 24)
    width = rng.randint(18, 24)
    grid = np.full((height, width), bg, dtype=int)

    candidate_options = _edge_rects(height, width)
    rng.shuffle(candidate_options)
    candidate_rect = None
    placed: list[tuple[int, int, int, int]] = []
    for _side, r0, c0 in candidate_options:
        rect = (r0, c0, INNER_SIZE, INNER_SIZE)
        candidate_rect = rect
        placed.append(rect)
        break
    if candidate_rect is None:
        return None

    frame_slots = [(r0, c0) for r0 in range(1, height - FRAME_SIZE) for c0 in range(1, width - FRAME_SIZE)]
    rng.shuffle(frame_slots)

    frame_positions = []
    for r0, c0 in frame_slots:
        rect = (r0, c0, FRAME_SIZE, FRAME_SIZE)
        if _can_place(rect, placed, gap=1):
            frame_positions.append((r0, c0))
            placed.append(rect)
            if len(frame_positions) == frame_count:
                break
    if len(frame_positions) != frame_count:
        return None

    hole_candidates = [idx for idx, rotation in enumerate(rotations) if rotation == hole_rotation]
    hole_index = rng.choice(hole_candidates)

    cand_r0, cand_c0, _, _ = candidate_rect
    grid[cand_r0 : cand_r0 + INNER_SIZE, cand_c0 : cand_c0 + INNER_SIZE] = tile

    hole_pos = None
    for idx, (r0, c0) in enumerate(frame_positions):
        rotation = int(rotations[idx])
        if idx == hole_index:
            _paint_frame(grid, r0, c0, frame_color, rotation, None, bg)
            hole_pos = (r0, c0)
        else:
            _paint_frame(grid, r0, c0, frame_color, rotation, tile, bg)

    if hole_pos is None:
        return None

    if int(np.count_nonzero(grid == bg)) <= int(np.count_nonzero(grid != bg)):
        return None

    if return_meta:
        return {
            "grid": grid,
            "candidate_tile": tile.copy(),
            "hole_rotation": int(hole_rotation),
            "hole_pos": hole_pos,
        }
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(400):
        result = _gen_attempt(rng)
        if result is not None:
            return result
    raise RuntimeError("generate_input: failed after 400 attempts")
