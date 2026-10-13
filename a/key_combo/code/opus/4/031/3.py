from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random
import sys

import numpy as np

UTILS_DIR = Path("/home/lixinyu/snap/new/solvers")
if str(UTILS_DIR) not in sys.path:
    sys.path.append(str(UTILS_DIR))

from arc_utils import Color, bbox_anchor, bbox_coords, find_components, get_bg, paste


FRAME_COLOR = Color.GREY
MULTI_FILL = Color.PINK
PALETTE = [Color.BLUE, Color.RED, Color.GREEN, Color.YELLOW, Color.ORANGE]
MAX_GEN_ATTEMPTS = 400


@dataclass
class Frame:
    bbox: tuple[int, int, int, int]

    @property
    def r0(self) -> int:
        return int(self.bbox[0])

    @property
    def c0(self) -> int:
        return int(self.bbox[1])

    @property
    def r1(self) -> int:
        return int(self.bbox[2])

    @property
    def c1(self) -> int:
        return int(self.bbox[3])

    def contains_center(self, r: int, c: int) -> bool:
        return self.r0 < int(r) < self.r1 and self.c0 < int(c) < self.c1


@dataclass
class Obj:
    color: int
    bbox: tuple[int, int, int, int]
    patch: np.ndarray
    mask: np.ndarray
    closed: bool

    @property
    def r0(self) -> int:
        return int(self.bbox[0])

    @property
    def c0(self) -> int:
        return int(self.bbox[1])

    @property
    def height(self) -> int:
        return int(self.patch.shape[0])

    @property
    def width(self) -> int:
        return int(self.patch.shape[1])


def _mask_from_component(coords: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    r0, c0, r1, c1 = bbox
    mask = np.zeros((r1 - r0 + 1, c1 - c0 + 1), dtype=bool)
    rel = coords - np.array([r0, c0])
    mask[rel[:, 0], rel[:, 1]] = True
    return mask


def _has_enclosed_hole(mask: np.ndarray) -> bool:
    h, w = mask.shape
    exterior = np.zeros_like(mask, dtype=bool)
    stack: list[tuple[int, int]] = []
    for r in range(h):
        for c in range(w):
            if (r in (0, h - 1) or c in (0, w - 1)) and not mask[r, c]:
                exterior[r, c] = True
                stack.append((r, c))
    while stack:
        r, c = stack.pop()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr = r + dr
            nc = c + dc
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc] and not exterior[nr, nc]:
                exterior[nr, nc] = True
                stack.append((nr, nc))
    return bool(np.any((~mask) & (~exterior)))


def _detect_frames(grid: np.ndarray) -> list[Frame]:
    frames: list[Frame] = []
    for comp in find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME_COLOR],
        return_format="dict",
    ):
        bbox = tuple(map(int, comp["bbox"]))
        r0, c0, r1, c1 = bbox
        h = r1 - r0 + 1
        w = c1 - c0 + 1
        if h < 3 or w < 3:
            continue
        patch = grid[r0 : r1 + 1, c0 : c1 + 1]
        per = 2 * (h + w) - 4
        if int(np.sum(patch == FRAME_COLOR)) != per:
            continue
        if not (patch[0, :] == FRAME_COLOR).all():
            continue
        if not (patch[-1, :] == FRAME_COLOR).all():
            continue
        if not (patch[:, 0] == FRAME_COLOR).all():
            continue
        if not (patch[:, -1] == FRAME_COLOR).all():
            continue
        frames.append(Frame(bbox))
    frames.sort(key=lambda f: (f.c0, f.r0, f.c1, f.r1))
    return frames


def _detect_objects(grid: np.ndarray) -> list[Obj]:
    colors = [int(c) for c in np.unique(grid) if int(c) not in {Color.BLACK, FRAME_COLOR}]
    objects: list[Obj] = []
    for comp in find_components(
        grid,
        bg=Color.BLACK,
        connectivity=4,
        monochromatic=True,
        colors=colors,
        return_format="dict",
    ):
        bbox = tuple(map(int, comp["bbox"]))
        color = int(comp["color"])
        mask = _mask_from_component(comp["coords"], bbox)
        patch = np.full(mask.shape, Color.BLACK, dtype=int)
        patch[mask] = color
        objects.append(Obj(color=color, bbox=bbox, patch=patch, mask=mask, closed=_has_enclosed_hole(mask)))
    return objects


def _frame_openings(frames: list[Frame], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    open_for_up = np.zeros(shape, dtype=bool)
    open_for_down = np.zeros(shape, dtype=bool)
    for frame in frames:
        open_for_up[frame.r1, frame.c0 : frame.c1 + 1] = True
        open_for_down[frame.r0, frame.c0 : frame.c1 + 1] = True
    return open_for_up, open_for_down


def _can_place(out: np.ndarray, obj: Obj, row: int, open_mask: np.ndarray) -> bool:
    h, w = obj.patch.shape
    c0 = obj.c0
    if row < 0 or row + h > out.shape[0] or c0 < 0 or c0 + w > out.shape[1]:
        return False
    blocked = out != Color.BLACK
    blocked[open_mask] = False
    return not np.any(blocked[row : row + h, c0 : c0 + w] & obj.mask)


def _simulate(grid: np.ndarray) -> tuple[np.ndarray, list[dict], list[Frame]]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    assert get_bg(g) == Color.BLACK

    frames = _detect_frames(g)
    objects = _detect_objects(g)
    open_for_up, open_for_down = _frame_openings(frames, g.shape)

    out = np.zeros_like(g)
    out[g == FRAME_COLOR] = FRAME_COLOR

    settled: list[dict] = []
    up_objs = [obj for obj in objects if obj.closed]
    down_objs = [obj for obj in objects if not obj.closed]
    up_objs.sort(key=lambda obj: (obj.bbox[0], obj.bbox[1], obj.bbox[2], obj.bbox[3]))
    down_objs.sort(key=lambda obj: (-obj.bbox[2], obj.bbox[1], -obj.bbox[0], obj.bbox[3]))

    for upward, group in ((True, up_objs), (False, down_objs)):
        step = -1 if upward else 1
        open_mask = open_for_up if upward else open_for_down
        for obj in group:
            row = obj.r0
            while _can_place(out, obj, row + step, open_mask):
                row += step
            paste(out, obj.patch, row, obj.c0, transparent=Color.BLACK)
            final_bbox = (row, obj.c0, row + obj.height - 1, obj.c0 + obj.width - 1)
            center = bbox_anchor(final_bbox, anchor="center")
            settled.append(
                {
                    "color": obj.color,
                    "closed": obj.closed,
                    "start_bbox": obj.bbox,
                    "final_bbox": final_bbox,
                    "center": center,
                }
            )

    counts: list[list[dict]] = [[] for _ in frames]
    for item in settled:
        cr, cc = item["center"]
        for idx, frame in enumerate(frames):
            if frame.contains_center(cr, cc):
                counts[idx].append(item)
                break

    for idx, frame in enumerate(frames):
        fill_color = None
        if len(counts[idx]) == 1:
            fill_color = int(counts[idx][0]["color"])
        elif len(counts[idx]) >= 2:
            fill_color = MULTI_FILL
        if fill_color is None:
            continue
        inner = out[frame.r0 + 1 : frame.r1, frame.c0 + 1 : frame.c1]
        inner[inner == Color.BLACK] = int(fill_color)

    return out, settled, frames


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide hole objects upward and holeless objects downward, then color each gray collector by how many objects it captures.

    Input structure:
    - Black background with several small monochrome 4-connected objects in distinct colors.
    - Some objects enclose at least one black hole in their own bbox; others do not.
    - There are also 2-3 gray 1px hollow rectangular collector frames in the middle area.

    Rule:
    - Classify each non-gray object by topology: hole objects move upward, holeless objects move downward.
    - Move upward objects from top to bottom and downward objects from bottom to top, sliding one row at a time until blocked by the edge, a settled object, or a gray frame boundary.
    - After all objects stop, count a frame as collecting an object iff that object's final bbox center lies strictly inside the frame interior.
    - Leave frame borders and object pixels unchanged; fill only the remaining black interior cells of each frame.
    - A frame with 0 collected objects stays black inside; with exactly 1 object it uses that object's color; with 2 or more objects it fills with magenta.

    Key insight:
    - The answer depends on a full dependency chain: hole detection decides direction, sliding decides final placement, final placement decides per-frame counts, and those counts decide the fill color.
    """
    out, _settled, _frames = _simulate(np.asarray(grid, dtype=int))
    return out


def _patch_closed(color: int) -> np.ndarray:
    mask = np.array(
        [
            [1, 1, 1],
            [1, 0, 1],
            [1, 1, 1],
        ],
        dtype=int,
    )
    patch = np.zeros_like(mask)
    patch[mask == 1] = int(color)
    return patch


def _patch_open(color: int) -> np.ndarray:
    options = [
        np.array(
            [
                [0, 1, 0],
                [1, 1, 1],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 1],
                [0, 1, 0],
                [0, 1, 0],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 0, 0],
                [1, 0, 0],
                [1, 1, 1],
            ],
            dtype=int,
        ),
        np.array(
            [
                [1, 1, 0],
                [0, 1, 0],
                [0, 1, 1],
            ],
            dtype=int,
        ),
    ]
    mask = random.choice(options)
    mask = np.rot90(mask, k=random.randint(0, 3))
    patch = np.zeros_like(mask)
    patch[mask == 1] = int(color)
    return patch


def _draw_frame(grid: np.ndarray, frame: Frame) -> None:
    grid[frame.r0, frame.c0 : frame.c1 + 1] = FRAME_COLOR
    grid[frame.r1, frame.c0 : frame.c1 + 1] = FRAME_COLOR
    grid[frame.r0 : frame.r1 + 1, frame.c0] = FRAME_COLOR
    grid[frame.r0 : frame.r1 + 1, frame.c1] = FRAME_COLOR


def _safe_place(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> bool:
    h, w = patch.shape
    if top < 0 or left < 0 or top + h > grid.shape[0] or left + w > grid.shape[1]:
        return False
    mask = patch != Color.BLACK
    occupied = grid != Color.BLACK
    for rr, cc in np.argwhere(mask):
        ar = int(top + rr)
        ac = int(left + cc)
        if occupied[ar, ac]:
            return False
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                nr = ar + dr
                nc = ac + dc
                if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1] and occupied[nr, nc]:
                    return False
    return True


def _place_patch(grid: np.ndarray, patch: np.ndarray, top: int, left: int) -> None:
    if not _safe_place(grid, patch, top, left):
        raise ValueError("invalid placement")
    paste(grid, patch, top, left, transparent=Color.BLACK)


def _rows_above(frame: Frame, count: int, height: int) -> list[int]:
    rows: list[int] = []
    bottom = frame.r0 - 2
    for _ in range(count):
        top = bottom - height + 1
        rows.append(top)
        bottom = top - 2
    return rows


def _rows_below(frame: Frame, count: int, height: int) -> list[int]:
    rows: list[int] = []
    top = frame.r1 + 2
    for _ in range(count):
        rows.append(top)
        top += height + 1
    return rows


def _build_frames() -> tuple[np.ndarray, list[Frame], list[int]]:
    n_frames = 3 if random.random() < 0.55 else 2
    counts = [3, 1] + ([0] if n_frames == 3 else [])
    random.shuffle(counts)

    widths: list[int] = []
    heights: list[int] = []
    for count in counts:
        if count == 3:
            widths.append(9)
            heights.append(11)
        else:
            widths.append(7 if n_frames == 3 else random.choice([7, 9]))
            heights.append(random.choice([7, 9]))

    gaps = [random.randint(2, 3 if n_frames == 3 else 5) for _ in range(n_frames - 1)]
    left_margin = random.randint(1, 2)
    right_margin = random.randint(1, 2)
    width = left_margin + sum(widths) + sum(gaps) + right_margin
    if width > 30:
        raise ValueError("frame layout too wide")

    height = 29
    top = 10
    grid = np.zeros((height, width), dtype=int)
    frames: list[Frame] = []
    col = left_margin
    for w, h in zip(widths, heights):
        frame = Frame((top, col, top + h - 1, col + w - 1))
        _draw_frame(grid, frame)
        frames.append(frame)
        col += w
        if len(frames) < n_frames:
            col += gaps[len(frames) - 1]
    return grid, frames, counts


def _initial_center_in_any_frame(top: int, left: int, patch: np.ndarray, frames: list[Frame]) -> bool:
    bbox = (top, left, top + patch.shape[0] - 1, left + patch.shape[1] - 1)
    cr, cc = bbox_anchor(bbox, anchor="center")
    return any(frame.contains_center(cr, cc) for frame in frames)


def _frame_collections(frames: list[Frame], settled: list[dict]) -> list[int]:
    counts = [0] * len(frames)
    for item in settled:
        cr, cc = item["center"]
        for idx, frame in enumerate(frames):
            if frame.contains_center(cr, cc):
                counts[idx] += 1
                break
    return counts


def generate_input() -> np.ndarray:
    for _ in range(MAX_GEN_ATTEMPTS):
        try:
            grid, frames, targets = _build_frames()
        except ValueError:
            continue

        colors = random.sample(PALETTE, 4)
        multi_idx = targets.index(3)
        single_idx = targets.index(1)

        if random.random() < 0.5:
            multi_dirs = ["up", "up", "down"]
        else:
            multi_dirs = ["down", "down", "up"]
        single_dir = random.choice(["up", "down"])

        plan = [
            (multi_idx, multi_dirs[0], colors[0]),
            (multi_idx, multi_dirs[1], colors[1]),
            (multi_idx, multi_dirs[2], colors[2]),
            (single_idx, single_dir, colors[3]),
        ]
        random.shuffle(plan)

        per_frame: dict[int, list[tuple[str, np.ndarray]]] = {idx: [] for idx in range(len(frames))}
        for frame_idx, direction, color in plan:
            patch = _patch_closed(color) if direction == "up" else _patch_open(color)
            if patch.shape[1] > (frames[frame_idx].c1 - frames[frame_idx].c0 - 1):
                break
            per_frame[frame_idx].append((direction, patch))
        else:
            try:
                for frame_idx, items in per_frame.items():
                    if not items:
                        continue
                    frame = frames[frame_idx]
                    ups = [(direction, patch) for direction, patch in items if direction == "up"]
                    downs = [(direction, patch) for direction, patch in items if direction == "down"]

                    up_rows = _rows_below(frame, len(ups), 3)
                    down_rows = _rows_above(frame, len(downs), 3)

                    if len(items) == 3:
                        lane_a = frame.c0 + 1
                        lane_b = frame.c0 + 5
                        if random.random() < 0.5:
                            lane_a, lane_b = lane_b, lane_a
                        up_lefts = [lane_a] * len(ups)
                        down_lefts = [lane_b] * len(downs)
                        if len(ups) == 1 and len(downs) == 2:
                            up_lefts = [lane_a]
                            down_lefts = [lane_b] * 2
                        elif len(ups) == 2 and len(downs) == 1:
                            up_lefts = [lane_a] * 2
                            down_lefts = [lane_b]
                    else:
                        shared_left = random.randint(frame.c0 + 1, frame.c1 - 3)
                        up_lefts = [shared_left] * len(ups)
                        down_lefts = [shared_left] * len(downs)

                    for (direction, patch), top, left in zip(ups, up_rows, up_lefts):
                        if _initial_center_in_any_frame(top, left, patch, frames):
                            raise ValueError("up object starts inside a frame")
                        _place_patch(grid, patch, top, left)

                    for (direction, patch), top, left in zip(downs, down_rows, down_lefts):
                        if _initial_center_in_any_frame(top, left, patch, frames):
                            raise ValueError("down object starts inside a frame")
                        _place_patch(grid, patch, top, left)
            except ValueError:
                continue

            out, settled, detected_frames = _simulate(grid)
            counts = _frame_collections(detected_frames, settled)
            initial_inside = 0
            for obj in _detect_objects(grid):
                center = bbox_anchor(obj.bbox, anchor="center")
                initial_inside += sum(frame.contains_center(*center) for frame in detected_frames)

            if counts.count(3) != 1:
                continue
            if counts.count(1) != 1:
                continue
            if len(detected_frames) == 3 and 0 not in counts:
                continue
            if initial_inside != 0:
                continue
            if not any(item["closed"] for item in settled):
                continue
            if not any(not item["closed"] for item in settled):
                continue
            if not any(item["start_bbox"] != item["final_bbox"] for item in settled):
                continue
            if np.array_equal(grid, out):
                continue

            multi_frame_idx = counts.index(3)
            multi_inner = out[
                detected_frames[multi_frame_idx].r0 + 1 : detected_frames[multi_frame_idx].r1,
                detected_frames[multi_frame_idx].c0 + 1 : detected_frames[multi_frame_idx].c1,
            ]
            if MULTI_FILL not in set(map(int, np.unique(multi_inner))):
                continue
            single_frame_idx = counts.index(1)
            single_colors = [item["color"] for item in settled if detected_frames[single_frame_idx].contains_center(*item["center"])]
            single_inner = out[
                detected_frames[single_frame_idx].r0 + 1 : detected_frames[single_frame_idx].r1,
                detected_frames[single_frame_idx].c0 + 1 : detected_frames[single_frame_idx].c1,
            ]
            if len(single_colors) != 1:
                continue
            if MULTI_FILL in set(map(int, np.unique(single_inner))):
                continue
            if single_colors[0] not in set(map(int, np.unique(single_inner))):
                continue

            return grid

    raise RuntimeError("failed to generate a valid ARC input")
