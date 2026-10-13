from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np


BG = 0
FRAME = 1
CONTROL = 2

DIRS = [
    ("up", (-1, 0)),
    ("right", (0, 1)),
    ("down", (1, 0)),
    ("left", (0, -1)),
]
DIR_INDEX = {name: i for i, (name, _delta) in enumerate(DIRS)}

GRID_SIDE_RANGE = (20, 25)
FRAME_COUNT_CHOICES = (3, 4)
FRAME_SIDE_MIN = 4
WINNER_SIDE_MAX = 16
OTHER_SIDE_MAX = 10
FRAME_GAP = 2
GENERATION_RETRY_LIMIT = 600
SELECTION_RETRY_LIMIT = 120

_MODE_CYCLE = (
    {"prefer_tie": "up", "winner_not_largest": True},
    {"prefer_tie": "right", "winner_not_largest": False},
    {"prefer_tie": "down", "winner_not_largest": True},
    {"prefer_tie": None, "winner_not_largest": False},
)
_GEN_CALL_COUNT = 0
_USED_OUTPUT_SHAPES: set[tuple[int, int]] = set()


@dataclass(frozen=True)
class Frame:
    index: int
    r0: int
    c0: int
    r1: int
    c1: int

    @property
    def height(self) -> int:
        return self.r1 - self.r0 + 1

    @property
    def width(self) -> int:
        return self.c1 - self.c0 + 1

    @property
    def interior_area(self) -> int:
        return (self.height - 2) * (self.width - 2)


@dataclass(frozen=True)
class ControlVote:
    coord: tuple[int, int]
    target: tuple[int, int]
    frame_index: int
    chosen_dir: str
    min_dist: int
    tie_dirs: tuple[str, ...]


def _components(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
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


def _frame_from_coords(coords: np.ndarray, index: int) -> Frame | None:
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    h = int(r1 - r0 + 1)
    w = int(c1 - c0 + 1)
    if h < 3 or w < 3:
        return None
    if len(coords) != 2 * h + 2 * w - 4:
        return None

    local = np.zeros((h, w), dtype=bool)
    local[coords[:, 0] - r0, coords[:, 1] - c0] = True
    expected = np.zeros((h, w), dtype=bool)
    expected[0, :] = True
    expected[-1, :] = True
    expected[:, 0] = True
    expected[:, -1] = True
    if not np.array_equal(local, expected):
        return None
    return Frame(index=index, r0=int(r0), c0=int(c0), r1=int(r1), c1=int(c1))


def _extract_frames(grid: np.ndarray) -> tuple[list[Frame], np.ndarray, np.ndarray]:
    blue = grid == FRAME
    comps = _components(blue)
    frames: list[Frame] = []
    frame_lookup = np.full(grid.shape, -1, dtype=int)
    bbox_mask = np.zeros(grid.shape, dtype=bool)

    for comp in comps:
        frame = _frame_from_coords(comp, len(frames))
        assert frame is not None, "all blue components must be single-pixel rectangular frames"
        frames.append(frame)
        frame_lookup[comp[:, 0], comp[:, 1]] = frame.index
        bbox_mask[frame.r0 : frame.r1 + 1, frame.c0 : frame.c1 + 1] = True

    assert len(frames) >= 3, "expected at least three frames"
    areas = [frame.interior_area for frame in frames]
    assert len(set(areas)) == len(areas), "frame interior areas must be unique"
    return frames, frame_lookup, bbox_mask


def _vote_for_position(
    grid: np.ndarray,
    frame_lookup: np.ndarray,
    bbox_mask: np.ndarray,
    r: int,
    c: int,
    *,
    strict: bool,
) -> ControlVote | None:
    expected_color = CONTROL if strict else BG
    if int(grid[r, c]) != expected_color or bool(bbox_mask[r, c]):
        return None

    visible: list[tuple[int, str, int]] = []
    h, w = grid.shape
    for name, (dr, dc) in DIRS:
        rr = r + dr
        cc = c + dc
        dist = 1
        while 0 <= rr < h and 0 <= cc < w:
            if int(grid[rr, cc]) == FRAME:
                visible.append((dist, name, int(frame_lookup[rr, cc])))
                break
            rr += dr
            cc += dc
            dist += 1

    if not visible:
        return None

    min_dist = min(dist for dist, _name, _idx in visible)
    tie_dirs = tuple(name for name, _delta in DIRS if any(dist == min_dist and d == name for dist, d, _idx in visible))
    chosen_dir = tie_dirs[0]
    dr, dc = DIRS[DIR_INDEX[chosen_dir]][1]
    target = (r + dr, c + dc)
    if min_dist < 2 or int(grid[target]) != BG:
        if strict:
            raise AssertionError("control pixels must extend into a background cell before reaching the frame")
        return None
    frame_index = next(idx for dist, name, idx in visible if dist == min_dist and name == chosen_dir)
    return ControlVote(
        coord=(r, c),
        target=target,
        frame_index=frame_index,
        chosen_dir=chosen_dir,
        min_dist=min_dist,
        tie_dirs=tie_dirs,
    )


def _analyze_grid(grid: np.ndarray) -> tuple[list[Frame], list[ControlVote], Frame]:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2 and max(g.shape) <= 30
    assert set(np.unique(g)).issubset({BG, FRAME, CONTROL})

    frames, frame_lookup, bbox_mask = _extract_frames(g)
    controls: list[ControlVote] = []
    votes = [0] * len(frames)

    control_coords = np.argwhere(g == CONTROL)
    assert len(control_coords) > 0, "expected at least one control pixel"
    for r0, c0 in control_coords:
        r = int(r0)
        c = int(c0)
        assert not bool(bbox_mask[r, c]), "control pixels must be outside all frame bounding boxes"
        vote = _vote_for_position(g, frame_lookup, bbox_mask, r, c, strict=True)
        assert vote is not None, "every control must see at least one frame edge"
        controls.append(vote)
        votes[vote.frame_index] += 1

    scored = sorted(
        ((votes[frame.index], frame.interior_area, -frame.index, frame) for frame in frames),
        reverse=True,
    )
    assert scored[0][0] > scored[1][0], "generated inputs must have a unique vote winner"
    return frames, controls, scored[0][3]


def _crop_frame(grid: np.ndarray, frame: Frame) -> np.ndarray:
    crop = np.array(grid[frame.r0 : frame.r1 + 1, frame.c0 : frame.c1 + 1], dtype=int, copy=True)
    crop[crop == FRAME] = CONTROL
    return crop


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the most-voted rectangular frame and recolor its border to the control color.

    Input structure:
    - The grid uses black background, several blue one-pixel-thick hollow rectangular frames,
      and red single control pixels outside every frame bbox.
    - Each control lies on at least one row or column that can see a frame border pixel.
    - Frames do not overlap, and their enclosed interior areas are pairwise distinct.

    Rule:
    - For each red control pixel, look up, right, down, and left for the nearest visible blue
      frame border pixel in each direction.
    - Choose the shortest such distance, breaking ties by direction priority
      `up > right > down > left`; the control casts one vote to that frame and conceptually
      extends one red cell toward it.
    - Select the frame with the most votes, breaking ties by larger enclosed interior area,
      crop that frame's bbox, and recolor every blue border pixel in the crop to red.

    Key insight:
    - The red pixels do not mark the output directly; they induce directional votes whose
      winner decides which existing frame is cropped, so frame geometry and control placement
      must be interpreted together.
    """
    g = np.asarray(grid, dtype=int)
    _frames, _controls, winner = _analyze_grid(g)
    return _crop_frame(g, winner)


def _output_shape_pool(h: int, w: int) -> list[tuple[int, int]]:
    return [
        (fh, fw)
        for fh in range(FRAME_SIDE_MIN, min(WINNER_SIDE_MAX, h - 4) + 1)
        for fw in range(FRAME_SIDE_MIN, min(WINNER_SIDE_MAX, w - 4) + 1)
    ]


def _choose_winner_shape(h: int, w: int, winner_not_largest: bool) -> tuple[int, int]:
    choices = _output_shape_pool(h, w)
    if winner_not_largest:
        max_other_area = (min(OTHER_SIDE_MAX, h - 4) - 2) * (min(OTHER_SIDE_MAX, w - 4) - 2)
        choices = [shape for shape in choices if (shape[0] - 2) * (shape[1] - 2) < max_other_area]
    choices = [shape for shape in choices if shape not in _USED_OUTPUT_SHAPES]
    if not choices:
        _USED_OUTPUT_SHAPES.clear()
        choices = _output_shape_pool(h, w)
        if winner_not_largest:
            max_other_area = (min(OTHER_SIDE_MAX, h - 4) - 2) * (min(OTHER_SIDE_MAX, w - 4) - 2)
            choices = [shape for shape in choices if (shape[0] - 2) * (shape[1] - 2) < max_other_area]
    return random.choice(choices)


def _random_bbox(shape: tuple[int, int], height: int, width: int) -> tuple[int, int, int, int]:
    h, w = shape
    r0 = random.randint(0, h - height)
    c0 = random.randint(0, w - width)
    return (r0, c0, r0 + height - 1, c0 + width - 1)


def _boxes_too_close(a: tuple[int, int, int, int], b: tuple[int, int, int, int], gap: int = FRAME_GAP) -> bool:
    return not (
        a[2] + gap < b[0]
        or b[2] + gap < a[0]
        or a[3] + gap < b[1]
        or b[3] + gap < a[1]
    )


def _paint_frame(grid: np.ndarray, box: tuple[int, int, int, int]) -> None:
    r0, c0, r1, c1 = box
    grid[r0, c0 : c1 + 1] = FRAME
    grid[r1, c0 : c1 + 1] = FRAME
    grid[r0 : r1 + 1, c0] = FRAME
    grid[r0 : r1 + 1, c1] = FRAME


def _choose_frame_dims(
    h: int,
    w: int,
    frame_count: int,
    winner_shape: tuple[int, int],
    winner_not_largest: bool,
) -> tuple[list[tuple[int, int]], int] | None:
    winner_area = (winner_shape[0] - 2) * (winner_shape[1] - 2)
    used_areas = {winner_area}
    dims = [winner_shape]

    if winner_not_largest:
        larger_choices = [
            (fh, fw)
            for fh in range(FRAME_SIDE_MIN, min(OTHER_SIDE_MAX, h - 4) + 1)
            for fw in range(FRAME_SIDE_MIN, min(OTHER_SIDE_MAX, w - 4) + 1)
            if (fh - 2) * (fw - 2) > winner_area
        ]
        if not larger_choices:
            return None
        larger_shape = random.choice(larger_choices)
        dims.append(larger_shape)
        used_areas.add((larger_shape[0] - 2) * (larger_shape[1] - 2))

    while len(dims) < frame_count:
        fh = random.randint(FRAME_SIDE_MIN, min(OTHER_SIDE_MAX, h - 4))
        fw = random.randint(FRAME_SIDE_MIN, min(OTHER_SIDE_MAX, w - 4))
        area = (fh - 2) * (fw - 2)
        if area in used_areas:
            continue
        dims.append((fh, fw))
        used_areas.add(area)

    random.shuffle(dims)
    winner_index = dims.index(winner_shape)
    return dims, winner_index


def _place_frames(shape: tuple[int, int], dims: list[tuple[int, int]]) -> list[tuple[int, int, int, int]] | None:
    boxes: list[tuple[int, int, int, int]] = []
    for height, width in sorted(dims, key=lambda x: x[0] * x[1], reverse=True):
        placed = False
        for _ in range(120):
            box = _random_bbox(shape, height, width)
            if any(_boxes_too_close(box, other) for other in boxes):
                continue
            boxes.append(box)
            placed = True
            break
        if not placed:
            return None
    return boxes


def _build_frame_grid(shape: tuple[int, int], boxes: list[tuple[int, int, int, int]]) -> np.ndarray:
    grid = np.zeros(shape, dtype=int)
    for box in boxes:
        _paint_frame(grid, box)
    return grid


def _candidate_controls(frame_grid: np.ndarray) -> tuple[list[list[ControlVote]], list[ControlVote]]:
    frames, frame_lookup, bbox_mask = _extract_frames(frame_grid)
    by_frame: list[list[ControlVote]] = [[] for _ in frames]
    all_votes: list[ControlVote] = []
    for r in range(frame_grid.shape[0]):
        for c in range(frame_grid.shape[1]):
            vote = _vote_for_position(frame_grid, frame_lookup, bbox_mask, r, c, strict=False)
            if vote is None:
                continue
            by_frame[vote.frame_index].append(vote)
            all_votes.append(vote)
    return by_frame, all_votes


def _choose_vote_counts(frame_count: int, winner_index: int) -> list[int]:
    counts = [1] * frame_count
    counts[winner_index] = random.randint(4, 6)
    for i in range(frame_count):
        if i == winner_index:
            continue
        counts[i] += random.randint(0, 2)
    runner = max(counts[i] for i in range(frame_count) if i != winner_index)
    if counts[winner_index] <= runner:
        counts[winner_index] = runner + 1
    return counts


def _conflicts(candidate: ControlVote, chosen: list[ControlVote]) -> bool:
    cr, cc = candidate.coord
    tr, tc = candidate.target
    for other in chosen:
        orr, occ = other.coord
        otr, otc = other.target
        if candidate.coord == other.coord or candidate.coord == other.target:
            return True
        if candidate.target == other.coord or candidate.target == other.target:
            return True
        if max(abs(cr - orr), abs(cc - occ)) <= 1:
            return True
        if max(abs(tr - otr), abs(tc - otc)) <= 0:
            return True
    return False


def _select_controls(
    by_frame: list[list[ControlVote]],
    counts: list[int],
    prefer_tie: str | None,
) -> list[ControlVote] | None:
    for _ in range(SELECTION_RETRY_LIMIT):
        chosen: list[ControlVote] = []
        remaining = counts[:]

        if prefer_tie is not None:
            tie_pool = [
                vote
                for votes in by_frame
                for vote in votes
                if len(vote.tie_dirs) > 1 and vote.chosen_dir == prefer_tie and remaining[vote.frame_index] > 0
            ]
            random.shuffle(tie_pool)
            special = next((vote for vote in tie_pool if not _conflicts(vote, chosen)), None)
            if special is None:
                continue
            chosen.append(special)
            remaining[special.frame_index] -= 1

        order = sorted(range(len(by_frame)), key=lambda i: (-remaining[i], i))
        ok = True
        for frame_index in order:
            if remaining[frame_index] <= 0:
                continue
            pool = by_frame[frame_index][:]
            random.shuffle(pool)
            frame_picks = 0
            for vote in pool:
                if _conflicts(vote, chosen):
                    continue
                chosen.append(vote)
                frame_picks += 1
                if frame_picks == remaining[frame_index]:
                    break
            if frame_picks != remaining[frame_index]:
                ok = False
                break

        if ok:
            return chosen
    return None


def _grid_from_controls(frame_grid: np.ndarray, controls: list[ControlVote]) -> np.ndarray:
    grid = np.array(frame_grid, copy=True)
    for vote in controls:
        r, c = vote.coord
        grid[r, c] = CONTROL
    return grid


def _validate_generated(
    grid: np.ndarray,
    prefer_tie: str | None,
    winner_not_largest: bool,
) -> bool:
    try:
        frames, controls, winner = _analyze_grid(grid)
    except Exception:
        return False

    if prefer_tie is not None and not any(len(v.tie_dirs) > 1 and v.chosen_dir == prefer_tie for v in controls):
        return False

    if winner_not_largest and winner.interior_area == max(frame.interior_area for frame in frames):
        return False

    vote_receivers = {vote.frame_index for vote in controls}
    if len(vote_receivers) < len(frames):
        return False

    expected = _crop_frame(grid, winner)
    return bool(np.array_equal(main(grid), expected))


def _attempt_generate(mode: dict[str, object]) -> np.ndarray | None:
    h = random.randint(*GRID_SIDE_RANGE)
    w = random.randint(*GRID_SIDE_RANGE)
    winner_shape = _choose_winner_shape(h, w, bool(mode["winner_not_largest"]))
    frame_count = random.choice(FRAME_COUNT_CHOICES)

    dims_info = _choose_frame_dims(
        h,
        w,
        frame_count,
        winner_shape,
        bool(mode["winner_not_largest"]),
    )
    if dims_info is None:
        return None
    dims, winner_index = dims_info

    boxes = _place_frames((h, w), dims)
    if boxes is None:
        return None

    frame_grid = _build_frame_grid((h, w), boxes)
    frames, _frame_lookup, _bbox_mask = _extract_frames(frame_grid)
    shape_to_index = {(frame.height, frame.width): frame.index for frame in frames}
    winner_index = shape_to_index[winner_shape]

    by_frame, all_votes = _candidate_controls(frame_grid)
    if not all(len(votes) >= 2 for votes in by_frame):
        return None

    if mode["prefer_tie"] is not None:
        if not any(len(v.tie_dirs) > 1 and v.chosen_dir == mode["prefer_tie"] for v in all_votes):
            return None

    counts = _choose_vote_counts(len(frames), winner_index)
    controls = _select_controls(by_frame, counts, mode["prefer_tie"])
    if controls is None:
        return None

    grid = _grid_from_controls(frame_grid, controls)
    if not _validate_generated(grid, mode["prefer_tie"], bool(mode["winner_not_largest"])):
        return None
    return grid


def generate_input() -> np.ndarray:
    global _GEN_CALL_COUNT
    mode = _MODE_CYCLE[_GEN_CALL_COUNT % len(_MODE_CYCLE)]
    for _ in range(GENERATION_RETRY_LIMIT):
        grid = _attempt_generate(mode)
        if grid is None:
            continue
        out_shape = tuple(map(int, main(grid).shape))
        if out_shape in _USED_OUTPUT_SHAPES:
            continue
        _USED_OUTPUT_SHAPES.add(out_shape)
        _GEN_CALL_COUNT += 1
        return grid
    raise RuntimeError("failed to generate a valid input for puzzle 6")
