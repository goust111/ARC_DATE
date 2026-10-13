from __future__ import annotations

from dataclasses import dataclass
from importlib.machinery import SourceFileLoader
from pathlib import Path
import random

import numpy as np


_ARC_UTILS = SourceFileLoader(
    "_arc_utils_fable_1_024_6",
    str(Path("/home/lixinyu/snap/new/solvers/arc_utils.py")),
).load_module()

Color = _ARC_UTILS.Color
colors_present = _ARC_UTILS.colors_present
crop_bbox = _ARC_UTILS.crop_bbox
find_components = _ARC_UTILS.find_components
get_bg = _ARC_UTILS.get_bg
in_bounds = _ARC_UTILS.in_bounds
paste = _ARC_UTILS.paste
random_free_location = _ARC_UTILS.random_free_location


BG = Color.BLACK
FRAME_COLOR = Color.GREY
PIECE_COLORS = [c for c in Color.ALL_COLORS if c not in (BG, FRAME_COLOR)]
SIDE_TO_STEP = {
    "top": (-1, 0),
    "bottom": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}
OPPOSITE_SIDE = {
    "top": "bottom",
    "bottom": "top",
    "left": "right",
    "right": "left",
}
SIDES = tuple(SIDE_TO_STEP)

GRID_H_RANGE = (22, 30)
GRID_W_RANGE = (22, 30)
FRAME_SIZE_RANGE = (5, 8)
FRAME_PADDING = 2
MIN_OUTSIDE_CELLS = 2
MAX_ATTEMPTS = 400


@dataclass(frozen=True)
class Frame:
    r0: int
    c0: int
    h: int
    w: int
    side: str
    gap_index: int
    label: str

    @property
    def r1(self) -> int:
        return self.r0 + self.h - 1

    @property
    def c1(self) -> int:
        return self.c0 + self.w - 1

    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r1, self.c1)

    def gap_local(self) -> tuple[int, int]:
        if self.side == "top":
            return (0, self.gap_index)
        if self.side == "bottom":
            return (self.h - 1, self.gap_index)
        if self.side == "left":
            return (self.gap_index, 0)
        return (self.gap_index, self.w - 1)

    def gap_abs(self) -> tuple[int, int]:
        rr, cc = self.gap_local()
        return (self.r0 + rr, self.c0 + cc)

    def lane_ids(self) -> list[int]:
        if self.side in ("top", "bottom"):
            return list(range(1, self.w - 1))
        return list(range(1, self.h - 1))

    def lane_capacity(self) -> int:
        if self.side in ("top", "bottom"):
            return self.h - 2
        return self.w - 2

    def leak_lane(self) -> int:
        return self.gap_index

    def patch(self) -> np.ndarray:
        return _frame_patch(self.h, self.w, self.side, self.gap_index)


def _frame_patch(h: int, w: int, side: str, gap_index: int) -> np.ndarray:
    patch = np.full((h, w), BG, dtype=int)
    patch[0, :] = FRAME_COLOR
    patch[h - 1, :] = FRAME_COLOR
    patch[:, 0] = FRAME_COLOR
    patch[:, w - 1] = FRAME_COLOR

    if side == "top":
        patch[0, gap_index] = BG
    elif side == "bottom":
        patch[h - 1, gap_index] = BG
    elif side == "left":
        patch[gap_index, 0] = BG
    else:
        patch[gap_index, w - 1] = BG
    return patch


def _sample_frame_dims() -> tuple[int, int]:
    return (
        random.randint(*FRAME_SIZE_RANGE),
        random.randint(*FRAME_SIZE_RANGE),
    )


def _sample_gap_index(side: str, h: int, w: int) -> int:
    if side in ("top", "bottom"):
        return random.randint(1, w - 2)
    return random.randint(1, h - 2)


def _point_in_bbox(frame: Frame, r: int, c: int) -> bool:
    return frame.r0 <= r <= frame.r1 and frame.c0 <= c <= frame.c1


def _is_frame_wall(frame: Frame, r: int, c: int) -> bool:
    if not _point_in_bbox(frame, r, c):
        return False
    lr = r - frame.r0
    lc = c - frame.c0
    if (lr, lc) == frame.gap_local():
        return False
    return lr in (0, frame.h - 1) or lc in (0, frame.w - 1)


def _trace_corridor(
    frame: Frame,
    frames: list[Frame],
    shape: tuple[int, int],
) -> tuple[list[tuple[int, int]] | None, str | None, str | None]:
    dr, dc = SIDE_TO_STEP[frame.side]
    rr, cc = frame.gap_abs()
    rr += dr
    cc += dc
    path: list[tuple[int, int]] = []

    while in_bounds(shape, rr, cc):
        blocker = None
        for other in frames:
            if other.label == frame.label:
                continue
            if _point_in_bbox(other, rr, cc):
                blocker = other
                break

        if blocker is not None:
            if not _is_frame_wall(blocker, rr, cc):
                return None, None, None
            return path, "frame", blocker.label

        path.append((rr, cc))
        rr += dr
        cc += dc

    return path, "edge", None


def _detect_frames(grid: np.ndarray) -> list[Frame]:
    comps = find_components(
        grid,
        bg=BG,
        connectivity=4,
        monochromatic=True,
        colors=[FRAME_COLOR],
        return_format="dict",
    )
    frames: list[Frame] = []
    for comp in comps:
        r0, c0, r1, c1 = map(int, comp["bbox"])
        patch = crop_bbox(grid, (r0, c0, r1, c1))
        h, w = patch.shape
        assert h >= 5 and w >= 5
        assert not np.any(patch[1:-1, 1:-1] == FRAME_COLOR)

        missing = []
        for cc in range(w):
            if patch[0, cc] != FRAME_COLOR:
                missing.append((0, cc))
            if patch[h - 1, cc] != FRAME_COLOR:
                missing.append((h - 1, cc))
        for rr in range(1, h - 1):
            if patch[rr, 0] != FRAME_COLOR:
                missing.append((rr, 0))
            if patch[rr, w - 1] != FRAME_COLOR:
                missing.append((rr, w - 1))

        assert len(missing) == 1
        mr, mc = missing[0]
        assert patch[mr, mc] == BG
        assert (mr in (0, h - 1)) ^ (mc in (0, w - 1))
        assert 0 < mr < h - 1 or 0 < mc < w - 1

        if mr == 0:
            side = "top"
            gap_index = mc
        elif mr == h - 1:
            side = "bottom"
            gap_index = mc
        elif mc == 0:
            side = "left"
            gap_index = mr
        else:
            side = "right"
            gap_index = mr

        frames.append(Frame(r0, c0, h, w, side, gap_index, f"det_{r0}_{c0}"))

    frames.sort(key=lambda f: (f.r0, f.c0))
    return frames


def _lane_items(patch: np.ndarray, frame: Frame, lane: int) -> list[tuple[int, int]]:
    items: list[tuple[int, int]] = []
    if frame.side in ("top", "bottom"):
        for rr in range(1, frame.h - 1):
            color = int(patch[rr, lane])
            if color not in (BG, FRAME_COLOR):
                items.append((rr, color))
    else:
        for cc in range(1, frame.w - 1):
            color = int(patch[lane, cc])
            if color not in (BG, FRAME_COLOR):
                items.append((cc, color))
    items.sort(key=lambda t: t[0], reverse=frame.side in ("bottom", "right"))
    return items


def _settle_frame(out: np.ndarray, src: np.ndarray, frame: Frame) -> None:
    patch = crop_bbox(src, frame.bbox())
    for lane in frame.lane_ids():
        items = _lane_items(patch, frame, lane)
        colors = [color for _axis, color in items]
        if not colors:
            continue

        if lane == frame.leak_lane():
            dr, dc = SIDE_TO_STEP[frame.side]
            start_r, start_c = frame.gap_abs()
            for color in colors:
                rr, cc = start_r, start_c
                while True:
                    nr = rr + dr
                    nc = cc + dc
                    if not in_bounds(out.shape, nr, nc) or int(out[nr, nc]) != BG:
                        break
                    rr, cc = nr, nc
                out[rr, cc] = int(color)
            continue

        if frame.side == "top":
            targets = [(frame.r0 + 1 + idx, frame.c0 + lane) for idx in range(len(colors))]
        elif frame.side == "bottom":
            targets = [(frame.r1 - 1 - idx, frame.c0 + lane) for idx in range(len(colors))]
        elif frame.side == "left":
            targets = [(frame.r0 + lane, frame.c0 + 1 + idx) for idx in range(len(colors))]
        else:
            targets = [(frame.r0 + lane, frame.c1 - 1 - idx) for idx in range(len(colors))]

        for (rr, cc), color in zip(targets, colors):
            out[rr, cc] = int(color)


def main(grid: np.ndarray) -> np.ndarray:
    """
    Slide each box's loose cells toward its one-cell gap, letting the aligned lane spill out through that opening.

    Input structure:
    - A black background contains several grey hollow rectangular frames.
    - Each frame has exactly one non-corner gap on one side.
    - Colored single-cell pieces appear only inside the frames.

    Rule:
    - For each frame, the side containing the gap defines that frame's gravity direction.
    - In every row or column lane perpendicular to that gravity, pieces slide toward the gap side and stack against the inner wall while preserving their original order within the lane.
    - The lane aligned with the gap is not stopped by a wall: its pieces leave through the gap in nearest-to-gap order, then keep sliding outside until blocked by the grid boundary, another frame, or earlier leaked pieces.
    - The grey frame outline and the gap itself stay unchanged.

    Key insight:
    - The missing border cell simultaneously points to the local gravity direction and marks the only lane that can escape the frame.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = int(get_bg(g))
    assert bg == BG

    frames = _detect_frames(g)
    assert frames

    out = g.copy()
    out[(out != bg) & (out != FRAME_COLOR)] = bg

    for frame in frames:
        _settle_frame(out, g, frame)
    return out


def _build_blocked_pair(shape: tuple[int, int]) -> list[Frame] | None:
    H, W = shape
    for _ in range(200):
        source_side = random.choice(list(SIDES))
        blocker_side = random.choice([side for side in SIDES if side != OPPOSITE_SIDE[source_side]])
        hs, ws = _sample_frame_dims()
        hb, wb = _sample_frame_dims()
        gap_len = random.randint(2, 5)

        if source_side == "bottom":
            if hs + hb + gap_len + 2 >= H:
                continue
            blocker_r0 = random.randint(hs + gap_len + 1, H - hb - 1)
            blocker_c0 = random.randint(1, W - wb - 1)
            lane_abs = random.randint(blocker_c0 + 1, blocker_c0 + wb - 2)
            source_gap = random.randint(1, ws - 2)
            source_c0 = lane_abs - source_gap
            source_r0 = blocker_r0 - gap_len - hs
        elif source_side == "top":
            if hs + hb + gap_len + 2 >= H:
                continue
            blocker_r0 = random.randint(1, H - hb - hs - gap_len - 1)
            blocker_c0 = random.randint(1, W - wb - 1)
            lane_abs = random.randint(blocker_c0 + 1, blocker_c0 + wb - 2)
            source_gap = random.randint(1, ws - 2)
            source_c0 = lane_abs - source_gap
            source_r0 = blocker_r0 + hb + gap_len
        elif source_side == "right":
            if ws + wb + gap_len + 2 >= W:
                continue
            blocker_r0 = random.randint(1, H - hb - 1)
            blocker_c0 = random.randint(ws + gap_len + 1, W - wb - 1)
            lane_abs = random.randint(blocker_r0 + 1, blocker_r0 + hb - 2)
            source_gap = random.randint(1, hs - 2)
            source_r0 = lane_abs - source_gap
            source_c0 = blocker_c0 - gap_len - ws
        else:
            if ws + wb + gap_len + 2 >= W:
                continue
            blocker_r0 = random.randint(1, H - hb - 1)
            blocker_c0 = random.randint(1, W - wb - ws - gap_len - 1)
            lane_abs = random.randint(blocker_r0 + 1, blocker_r0 + hb - 2)
            source_gap = random.randint(1, hs - 2)
            source_r0 = lane_abs - source_gap
            source_c0 = blocker_c0 + wb + gap_len

        blocker_gap = _sample_gap_index(blocker_side, hb, wb)
        source = Frame(source_r0, source_c0, hs, ws, source_side, source_gap, "source")
        blocker = Frame(blocker_r0, blocker_c0, hb, wb, blocker_side, blocker_gap, "blocker")

        ok = (
            1 <= source.r0 <= H - source.h - 1
            and 1 <= source.c0 <= W - source.w - 1
            and 1 <= blocker.r0 <= H - blocker.h - 1
            and 1 <= blocker.c0 <= W - blocker.w - 1
        )
        if not ok:
            continue

        path, stop, blocker_label = _trace_corridor(source, [source, blocker], shape)
        if path is None or stop != "frame" or blocker_label != "blocker":
            continue
        if len(path) < MIN_OUTSIDE_CELLS:
            continue
        return [source, blocker]
    return None


def _bbox_canvas(shape: tuple[int, int], frames: list[Frame]) -> np.ndarray:
    canvas = np.zeros(shape, dtype=int)
    for frame in frames:
        canvas[frame.r0 : frame.r1 + 1, frame.c0 : frame.c1 + 1] = 1
    return canvas


def _bbox_cells(frame: Frame) -> set[tuple[int, int]]:
    return {
        (rr, cc)
        for rr in range(frame.r0, frame.r1 + 1)
        for cc in range(frame.c0, frame.c1 + 1)
    }


def _place_free_frame(
    shape: tuple[int, int],
    frames: list[Frame],
    reserved_path_cells: set[tuple[int, int]],
    label: str,
    side_choices: list[str],
) -> Frame | None:
    occ = _bbox_canvas(shape, frames)
    for _ in range(200):
        h, w = _sample_frame_dims()
        side = random.choice(side_choices)
        gap = _sample_gap_index(side, h, w)
        block = np.ones((h, w), dtype=int)
        try:
            r0, c0 = random_free_location(
                occ,
                block,
                transparent=0,
                border=1,
                padding=FRAME_PADDING,
            )
        except ValueError:
            return None

        frame = Frame(r0, c0, h, w, side, gap, label)
        if _bbox_cells(frame) & reserved_path_cells:
            continue

        path, stop, blocker = _trace_corridor(frame, frames + [frame], shape)
        if path is None or stop != "edge" or blocker is not None:
            continue
        if len(path) < MIN_OUTSIDE_CELLS:
            continue
        if set(path) & reserved_path_cells:
            continue
        return frame
    return None


def _analyze_corridors(
    frames: list[Frame],
    shape: tuple[int, int],
) -> dict[str, dict[str, object]] | None:
    used: set[tuple[int, int]] = set()
    info: dict[str, dict[str, object]] = {}
    for frame in frames:
        path, stop, blocker = _trace_corridor(frame, frames, shape)
        if path is None:
            return None
        path_set = set(path)
        if path_set & used:
            return None
        used |= path_set
        info[frame.label] = {"path": path, "stop": stop, "blocker": blocker}
    return info


def _axis_to_local(frame: Frame, lane: int, axis_pos: int) -> tuple[int, int]:
    if frame.side in ("top", "bottom"):
        return (axis_pos, lane)
    return (lane, axis_pos)


def _bias_axis_positions(frame: Frame, axis_positions: list[int]) -> list[int]:
    half = max(1, frame.lane_capacity() // 2)
    if frame.side in ("bottom", "right"):
        favored = [pos for pos in axis_positions if pos <= half]
    else:
        favored = [pos for pos in axis_positions if pos > frame.lane_capacity() - half]
    return favored or axis_positions


def _sample_lane_cells(
    frame: Frame,
    lane: int,
    count: int,
    used: set[tuple[int, int]],
    *,
    bias_away_from_wall: bool,
) -> list[tuple[int, int]]:
    candidates = []
    for axis_pos in range(1, frame.lane_capacity() + 1):
        cell = _axis_to_local(frame, lane, axis_pos)
        if cell not in used:
            candidates.append(axis_pos)
    if len(candidates) < count:
        raise ValueError("lane capacity exhausted")

    pool = _bias_axis_positions(frame, candidates) if bias_away_from_wall else candidates
    if len(pool) < count:
        pool = candidates
    chosen_axes = random.sample(pool, count)
    cells = [_axis_to_local(frame, lane, axis_pos) for axis_pos in chosen_axes]
    used.update(cells)
    return cells


def _interior_cells(frame: Frame) -> list[tuple[int, int]]:
    return [
        (rr, cc)
        for rr in range(1, frame.h - 1)
        for cc in range(1, frame.w - 1)
    ]


def _paint_frame_pieces(
    grid: np.ndarray,
    frame: Frame,
    leak_count: int,
    palette: list[int],
    *,
    ensure_stack: bool,
) -> None:
    used: set[tuple[int, int]] = set()
    piece_cells: list[tuple[int, int]] = []
    lanes = frame.lane_ids()
    leak_lane = frame.leak_lane()

    if leak_count > 0:
        piece_cells.extend(
            _sample_lane_cells(
                frame,
                leak_lane,
                leak_count,
                used,
                bias_away_from_wall=False,
            )
        )

    non_leak_lanes = [lane for lane in lanes if lane != leak_lane]
    if ensure_stack and non_leak_lanes:
        stack_lane = random.choice(non_leak_lanes)
        piece_cells.extend(
            _sample_lane_cells(
                frame,
                stack_lane,
                min(2, frame.lane_capacity()),
                used,
                bias_away_from_wall=True,
            )
        )

    available = [cell for cell in _interior_cells(frame) if cell not in used]
    max_total = min(9, len(_interior_cells(frame)))
    min_total = max(4, len(piece_cells))
    target_total = random.randint(min_total, max_total)
    extra = min(len(available), target_total - len(piece_cells))
    if extra > 0:
        piece_cells.extend(random.sample(available, extra))

    random.shuffle(piece_cells)
    colors = [random.choice(palette) for _ in piece_cells]
    for (lr, lc), color in zip(piece_cells, colors):
        grid[frame.r0 + lr, frame.c0 + lc] = int(color)


def _lane_color_sequence(patch: np.ndarray, frame: Frame, lane: int) -> list[int]:
    return [color for _axis, color in _lane_items(patch, frame, lane)]


def _lane_output_positions(frame: Frame, lane: int, count: int) -> list[int]:
    if lane == frame.leak_lane():
        return []
    if frame.side in ("top", "left"):
        return list(range(1, count + 1))
    return list(range(frame.lane_capacity(), frame.lane_capacity() - count, -1))


def _output_lane_sequence(patch: np.ndarray, frame: Frame, lane: int) -> list[int]:
    items = _lane_items(patch, frame, lane)
    return [color for _axis, color in items]


def _frame_mask_from_frames(shape: tuple[int, int], frames: list[Frame]) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    for frame in frames:
        mask[frame.r0 : frame.r1 + 1, frame.c0 : frame.c1 + 1] = True
    return mask


def _semantic_ok(
    inp: np.ndarray,
    out: np.ndarray,
    frames: list[Frame],
    corridors: dict[str, dict[str, object]],
    leak_counts: dict[str, int],
) -> bool:
    if np.array_equal(inp, out):
        return False

    try:
        detected = _detect_frames(inp)
    except Exception:
        return False
    if len(detected) != len(frames):
        return False

    detected_boxes = sorted(frame.bbox() for frame in detected)
    planned_boxes = sorted(frame.bbox() for frame in frames)
    if detected_boxes != planned_boxes:
        return False

    if not np.array_equal(inp == FRAME_COLOR, out == FRAME_COLOR):
        return False

    all_piece_colors = set(colors_present(inp, exclude={BG, FRAME_COLOR}))
    all_piece_colors |= set(colors_present(out, exclude={BG, FRAME_COLOR}))
    for color in all_piece_colors:
        if int(np.count_nonzero(inp == color)) != int(np.count_nonzero(out == color)):
            return False

    frame_area = _frame_mask_from_frames(inp.shape, frames)
    if not bool(((out != BG) & (out != FRAME_COLOR) & ~frame_area).any()):
        return False

    for frame in frames:
        in_patch = crop_bbox(inp, frame.bbox())
        out_patch = crop_bbox(out, frame.bbox())
        path = list(corridors[frame.label]["path"])
        leak_count = int(leak_counts[frame.label])

        for lane in frame.lane_ids():
            input_seq = _lane_color_sequence(in_patch, frame, lane)
            output_seq = _output_lane_sequence(out_patch, frame, lane)

            if lane == frame.leak_lane():
                if output_seq:
                    return False
                occupied_path = [cell for cell in path if int(out[cell]) not in (BG, FRAME_COLOR)]
                if len(occupied_path) != leak_count:
                    return False
                if leak_count:
                    expected_cells = path[-leak_count:]
                    if occupied_path != expected_cells:
                        return False
                    expected_colors = input_seq
                    leaked_colors = [int(out[cell]) for cell in reversed(expected_cells)]
                    if leaked_colors != expected_colors:
                        return False
                elif occupied_path:
                    return False
                continue

            if len(input_seq) != len(output_seq):
                return False
            if input_seq != output_seq:
                return False

            expected_axes = _lane_output_positions(frame, lane, len(input_seq))
            actual_axes = [axis for axis, _color in _lane_items(out_patch, frame, lane)]
            if actual_axes != expected_axes:
                return False

    return True


def _gen_attempt() -> np.ndarray | None:
    H = random.randint(*GRID_H_RANGE)
    W = random.randint(*GRID_W_RANGE)
    shape = (H, W)

    pair = _build_blocked_pair(shape)
    if pair is None:
        return None
    frames = pair[:]

    pair_info = _analyze_corridors(frames, shape)
    if pair_info is None:
        return None
    reserved_paths = set(pair_info["source"]["path"])

    free_side_choices = [side for side in SIDES if side != frames[0].side]
    free_frame = _place_free_frame(shape, frames, reserved_paths, "free", free_side_choices)
    if free_frame is None:
        return None
    frames.append(free_frame)

    corridors = _analyze_corridors(frames, shape)
    if corridors is None:
        return None
    if corridors["source"]["stop"] != "frame" or corridors["source"]["blocker"] != "blocker":
        return None
    if corridors["free"]["stop"] != "edge":
        return None
    if len(corridors["source"]["path"]) < MIN_OUTSIDE_CELLS:
        return None
    if len(corridors["free"]["path"]) < MIN_OUTSIDE_CELLS:
        return None

    if random.random() < 0.45:
        reserved = set()
        for data in corridors.values():
            reserved.update(data["path"])
        extra = _place_free_frame(shape, frames, reserved, "extra", list(SIDES))
        if extra is not None:
            frames.append(extra)
            corridors = _analyze_corridors(frames, shape)
            if corridors is None:
                return None

    grid = np.full(shape, BG, dtype=int)
    for frame in frames:
        paste(grid, frame.patch(), frame.r0, frame.c0, transparent=BG)

    palette_size = random.randint(3, min(6, len(PIECE_COLORS)))
    palette = random.sample(PIECE_COLORS, palette_size)

    leak_counts: dict[str, int] = {}
    for frame in frames:
        path_len = len(corridors[frame.label]["path"])
        if frame.label == "source":
            leak_counts[frame.label] = random.randint(2, min(3, path_len))
        elif frame.label == "free":
            leak_counts[frame.label] = random.randint(1, min(3, path_len))
        else:
            leak_counts[frame.label] = random.randint(0, min(2, path_len))

    for frame in frames:
        ensure_stack = frame.label in {"source", "free"} or random.random() < 0.7
        _paint_frame_pieces(grid, frame, leak_counts[frame.label], palette, ensure_stack=ensure_stack)

    out = main(grid)
    if not _semantic_ok(grid, out, frames, corridors, leak_counts):
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid input after {MAX_ATTEMPTS} attempts")
