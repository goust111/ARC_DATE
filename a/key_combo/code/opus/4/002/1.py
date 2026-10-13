from __future__ import annotations

from collections import Counter
import random

import numpy as np


BLACK = 0
BLUE = 1
RED = 2
GREEN = 3
YELLOW = 4
GRAY = 5

SIDES = ("up", "down", "left", "right")
DIRS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
RING_OFFSETS = tuple(
    (dr, dc)
    for dr in (-1, 0, 1)
    for dc in (-1, 0, 1)
    if not (dr == 0 and dc == 0)
)
SHAPE_BOXES = ((1, 3), (2, 2), (2, 3), (3, 1), (3, 2), (3, 3))
SPRITE_COLORS = (RED, GREEN, YELLOW)


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def bbox_coords(coords: np.ndarray) -> tuple[int, int, int, int]:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    return int(r0), int(c0), int(r1), int(c1)


def components_mask(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []

    for r, c in np.argwhere(mask):
        r = int(r)
        c = int(c)
        if seen[r, c]:
            continue
        stack = [(r, c)]
        seen[r, c] = True
        coords: list[tuple[int, int]] = []
        while stack:
            rr, cc = stack.pop()
            coords.append((rr, cc))
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                nr = rr + dr
                nc = cc + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                    seen[nr, nc] = True
                    stack.append((nr, nc))
        comps.append(np.array(coords, dtype=int))
    return comps


def _gap_side(gap: tuple[int, int], bbox: tuple[int, int, int, int]) -> str:
    r, c = gap
    r0, c0, r1, c1 = bbox
    if r == r0:
        return "up"
    if r == r1:
        return "down"
    if c == c0:
        return "left"
    if c == c1:
        return "right"
    raise AssertionError("gap must lie on a frame side")


def _slide_distance(
    sprite_bbox: tuple[int, int, int, int],
    frame_bbox: tuple[int, int, int, int],
    side: str,
) -> int:
    sr0, sc0, sr1, sc1 = sprite_bbox
    fr0, fc0, fr1, fc1 = frame_bbox
    if side == "up":
        return int(sr0 - (fr0 + 1))
    if side == "down":
        return int((fr1 - 1) - sr1)
    if side == "left":
        return int(sc0 - (fc0 + 1))
    if side == "right":
        return int((fc1 - 1) - sc1)
    raise AssertionError("invalid side")


def _move_coords(coords: np.ndarray, side: str, dist: int) -> np.ndarray:
    dr, dc = DIRS[side]
    return coords + np.array([dr * int(dist), dc * int(dist)], dtype=int)


def _extract_frames(grid: np.ndarray) -> list[dict]:
    g = np.asarray(grid, dtype=int)
    blue_components = components_mask(g == BLUE)
    assert blue_components, "expected blue frame components"

    frames: list[dict] = []
    for frame_coords in blue_components:
        bbox = bbox_coords(frame_coords)
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        assert h >= 5 and w >= 5, "frame must have a nontrivial interior"
        assert all(
            int(r) in (r0, r1) or int(c) in (c0, c1)
            for r, c in frame_coords
        ), "blue pixels must lie on a rectangular border"

        missing: list[tuple[int, int]] = []
        for r in range(r0, r1 + 1):
            for c in range(c0, c1 + 1):
                if r in (r0, r1) or c in (c0, c1):
                    if int(g[r, c]) != BLUE:
                        missing.append((r, c))
        assert len(missing) == 1, "each frame must have exactly one gap"
        gap = missing[0]
        assert gap[0] not in (r0, r1) or gap[1] not in (c0, c1), "gap cannot be a corner"
        side = _gap_side(gap, bbox)

        inner = g[r0 + 1 : r1, c0 + 1 : c1]
        sprite_mask = (inner != BLACK) & (inner != BLUE)
        sprite_components = components_mask(sprite_mask)
        assert len(sprite_components) == 1, "each frame must contain exactly one sprite"
        sprite_coords = sprite_components[0] + np.array([r0 + 1, c0 + 1], dtype=int)
        sprite_colors = {int(g[r, c]) for r, c in sprite_coords}
        assert len(sprite_colors) == 1, "sprite must be monochrome"
        sprite_color = sprite_colors.pop()
        sprite_bbox = bbox_coords(sprite_coords)
        assert sprite_bbox[0] >= r0 + 2 and sprite_bbox[1] >= c0 + 2
        assert sprite_bbox[2] <= r1 - 2 and sprite_bbox[3] <= c1 - 2

        frames.append(
            {
                "bbox": bbox,
                "gap": gap,
                "side": side,
                "sprite_color": int(sprite_color),
                "sprite_coords": sprite_coords,
            }
        )

    frames.sort(key=lambda frame: frame["bbox"])
    return frames


def _render(
    grid: np.ndarray,
    frames: list[dict],
    *,
    side_overrides: dict[int, str] | None = None,
    move_sprite: bool = True,
    draw_outline: bool = True,
    clip_to_frame: bool = True,
) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    side_overrides = {} if side_overrides is None else dict(side_overrides)

    for idx, frame in enumerate(frames):
        sprite = frame["sprite_coords"]
        out[sprite[:, 0], sprite[:, 1]] = BLACK

        side = side_overrides.get(idx, frame["side"])
        moved = sprite.copy()
        if move_sprite:
            dist = _slide_distance(bbox_coords(sprite), frame["bbox"], side)
            moved = _move_coords(sprite, side, dist)

        moved_set = {tuple(map(int, coord)) for coord in moved.tolist()}
        out[moved[:, 0], moved[:, 1]] = int(frame["sprite_color"])

        if not draw_outline:
            continue

        r0, c0, r1, c1 = frame["bbox"]
        for r_raw, c_raw in moved:
            r = int(r_raw)
            c = int(c_raw)
            for dr, dc in RING_OFFSETS:
                rr = r + dr
                cc = c + dc
                if (rr, cc) in moved_set:
                    continue
                if clip_to_frame and not (r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1):
                    continue
                if 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]:
                    out[rr, cc] = GRAY
    return out


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each sprite toward its frame gap side, then draw a frame-clipped gray halo around the parked sprite.

    Input structure:
    - The grid has a black background and 2 to 4 separate blue hollow rectangular frames.
    - Every blue frame is one pixel thick and has exactly one missing non-corner border cell.
    - Each frame contains one monochrome 4-connected sprite of 3 to 5 cells, initially at least one background cell away from every wall.

    Rule:
    - For each frame, read the missing border cell as a direction: top, bottom, left, or right.
    - Translate the enclosed sprite straight in that direction until its bounding box touches the frame's inner wall, while keeping the sprite inside the frame.
    - Clear the sprite's old location, keep the moved sprite in its original color, and paint gray 3x3 hollow-square neighbors around every moved sprite cell.
    - Draw gray only inside the frame interior; any ring pixel that would land on the blue wall or outside the frame is omitted, and gray never replaces the sprite itself.

    Key insight:
    - The border gap does not act as an exit; it only encodes which wall the sprite should slide toward, and the visible gray outline is determined only after that slide and then clipped by the frame interior.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == BLACK
    frames = _extract_frames(g)
    return _render(g, frames)


def _random_connected_shape() -> np.ndarray:
    for _ in range(200):
        area = random.randint(3, 5)
        choices = [(h, w) for h, w in SHAPE_BOXES if h * w >= area]
        h, w = random.choice(choices)
        cells = {(random.randrange(h), random.randrange(w))}
        frontier: set[tuple[int, int]] = set()

        def add_frontier(r: int, c: int) -> None:
            for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                rr = r + dr
                cc = c + dc
                if 0 <= rr < h and 0 <= cc < w and (rr, cc) not in cells:
                    frontier.add((rr, cc))

        add_frontier(*next(iter(cells)))
        while len(cells) < area and frontier:
            pick = random.choice(tuple(frontier))
            frontier.remove(pick)
            cells.add(pick)
            add_frontier(*pick)
        if len(cells) != area:
            continue

        mask = np.zeros((h, w), dtype=bool)
        for r, c in cells:
            mask[r, c] = True
        coords = np.argwhere(mask)
        r0, c0, r1, c1 = bbox_coords(coords)
        trimmed = mask[r0 : r1 + 1, c0 : c1 + 1]
        if trimmed.sum() != area:
            continue
        return trimmed
    raise RuntimeError("failed to sample a sprite shape")


def _build_frame_patch(side: str, color: int) -> np.ndarray | None:
    for _ in range(200):
        shape = _random_connected_shape()
        sh, sw = shape.shape

        min_h = max(6, sh + 4)
        min_w = max(6, sw + 4)
        if min_h > 8 or min_w > 8:
            continue
        h = random.randint(min_h, 8)
        w = random.randint(min_w, 8)

        safe_rows = list(range(2, h - sh - 1))
        safe_cols = list(range(2, w - sw - 1))
        if not safe_rows or not safe_cols:
            continue

        if side == "up":
            r = random.choice(safe_rows)
            c = random.choice(safe_cols)
            final_r = 1
            gap_choices = [cc for cc in range(1, w - 1) if not (c <= cc <= c + sw - 1)]
            gap = (0, random.choice(gap_choices))
        elif side == "down":
            r = random.choice(safe_rows)
            c = random.choice(safe_cols)
            final_r = h - 1 - sh
            gap_choices = [cc for cc in range(1, w - 1) if not (c <= cc <= c + sw - 1)]
            gap = (h - 1, random.choice(gap_choices))
        elif side == "left":
            r = random.choice(safe_rows)
            c = random.choice(safe_cols)
            final_c = 1
            gap_choices = [rr for rr in range(1, h - 1) if not (r <= rr <= r + sh - 1)]
            gap = (random.choice(gap_choices), 0)
        elif side == "right":
            r = random.choice(safe_rows)
            c = random.choice(safe_cols)
            final_c = w - 1 - sw
            gap_choices = [rr for rr in range(1, h - 1) if not (r <= rr <= r + sh - 1)]
            gap = (random.choice(gap_choices), w - 1)
        else:
            raise AssertionError("invalid side")

        patch = np.zeros((h, w), dtype=int)
        patch[0, :] = BLUE
        patch[-1, :] = BLUE
        patch[:, 0] = BLUE
        patch[:, -1] = BLUE
        patch[gap] = BLACK

        coords = np.argwhere(shape) + np.array([r, c], dtype=int)
        patch[coords[:, 0], coords[:, 1]] = int(color)

        frame = _extract_frames(patch)[0]
        dist = _slide_distance(bbox_coords(frame["sprite_coords"]), frame["bbox"], frame["side"])
        if dist <= 0:
            continue
        moved = _move_coords(frame["sprite_coords"], frame["side"], dist)
        moved_bbox = bbox_coords(moved)
        gap_r, gap_c = frame["gap"]
        if frame["side"] in ("up", "down"):
            if moved_bbox[1] <= gap_c <= moved_bbox[3]:
                continue
            if frame["side"] == "up" and moved_bbox[0] != final_r:
                continue
            if frame["side"] == "down" and moved_bbox[2] != final_r + sh - 1:
                continue
        else:
            if moved_bbox[0] <= gap_r <= moved_bbox[2]:
                continue
            if frame["side"] == "left" and moved_bbox[1] != final_c:
                continue
            if frame["side"] == "right" and moved_bbox[3] != final_c + sw - 1:
                continue
        return patch
    return None


def _place_patches(shape: tuple[int, int], patches: list[np.ndarray]) -> np.ndarray | None:
    h, w = shape
    grid = np.zeros((h, w), dtype=int)
    reserved = np.zeros((h, w), dtype=bool)

    ordered = sorted(patches, key=lambda patch: patch.shape[0] * patch.shape[1], reverse=True)
    for patch in ordered:
        ph, pw = patch.shape
        candidates: list[tuple[int, int]] = []
        for r0 in range(1, h - ph):
            for c0 in range(1, w - pw):
                rr0 = max(0, r0 - 2)
                cc0 = max(0, c0 - 2)
                rr1 = min(h, r0 + ph + 2)
                cc1 = min(w, c0 + pw + 2)
                if reserved[rr0:rr1, cc0:cc1].any():
                    continue
                candidates.append((r0, c0))
        if not candidates:
            return None

        r0, c0 = random.choice(candidates)
        grid[r0 : r0 + ph, c0 : c0 + pw] = patch
        rr0 = max(0, r0 - 2)
        cc0 = max(0, c0 - 2)
        rr1 = min(h, r0 + ph + 2)
        cc1 = min(w, c0 + pw + 2)
        reserved[rr0:rr1, cc0:cc1] = True
    return grid


def _validate_instance(grid: np.ndarray) -> bool:
    try:
        frames = _extract_frames(grid)
    except AssertionError:
        return False
    if not 2 <= len(frames) <= 4:
        return False

    try:
        out = main(grid)
    except AssertionError:
        return False

    if np.array_equal(out, grid):
        return False
    if np.array_equal(_render(grid, frames, move_sprite=False), out):
        return False
    if np.array_equal(_render(grid, frames, draw_outline=False), out):
        return False
    if np.array_equal(_render(grid, frames, clip_to_frame=False), out):
        return False

    for idx, frame in enumerate(frames):
        sprite_bbox = bbox_coords(frame["sprite_coords"])
        dist = _slide_distance(sprite_bbox, frame["bbox"], frame["side"])
        if dist <= 0:
            return False

        moved = _move_coords(frame["sprite_coords"], frame["side"], dist)
        moved_bbox = bbox_coords(moved)
        gap_r, gap_c = frame["gap"]
        if frame["side"] in ("up", "down"):
            if moved_bbox[1] <= gap_c <= moved_bbox[3]:
                return False
        else:
            if moved_bbox[0] <= gap_r <= moved_bbox[2]:
                return False

        gray_added = 0
        r0, c0, r1, c1 = frame["bbox"]
        moved_set = {tuple(map(int, coord)) for coord in moved.tolist()}
        for r_raw, c_raw in moved:
            r = int(r_raw)
            c = int(c_raw)
            for dr, dc in RING_OFFSETS:
                rr = r + dr
                cc = c + dc
                if (rr, cc) in moved_set:
                    continue
                if r0 + 1 <= rr <= r1 - 1 and c0 + 1 <= cc <= c1 - 1:
                    gray_added += 1
        if gray_added == 0:
            return False

        for wrong_side in SIDES:
            if wrong_side == frame["side"]:
                continue
            if np.array_equal(_render(grid, frames, side_overrides={idx: wrong_side}), out):
                return False

    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        frame_count = random.randint(2, 4)
        directions = random.sample(SIDES, k=frame_count)

        colors = list(SPRITE_COLORS)
        random.shuffle(colors)
        while len(colors) < frame_count:
            colors.append(random.choice(SPRITE_COLORS))

        patches: list[np.ndarray] = []
        failed = False
        for side, color in zip(directions, colors):
            patch = _build_frame_patch(side, color)
            if patch is None:
                failed = True
                break
            patches.append(patch)
        if failed:
            continue

        base = 18 + 2 * (frame_count - 2)
        h = random.randint(base, 28)
        w = random.randint(base, 28)
        grid = _place_patches((h, w), patches)
        if grid is None:
            continue
        if _validate_instance(grid):
            return grid

    raise RuntimeError("failed to generate a valid input")
