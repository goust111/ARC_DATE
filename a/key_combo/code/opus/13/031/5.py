from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import random

import numpy as np


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    interior_area: int

    @property
    def signature(self) -> tuple[int, int, int, int, int]:
        r0, c0, r1, c1 = self.bbox
        return (self.color, r0, c0, r1, c1)


@dataclass(frozen=True)
class Slot:
    r0: int
    c0: int
    h: int
    w: int

    @property
    def bbox(self) -> tuple[int, int, int, int]:
        return (self.r0, self.c0, self.r0 + self.h - 1, self.c0 + self.w - 1)

    @property
    def interior_area(self) -> int:
        return (self.h - 2) * (self.w - 2)


BACKGROUND = 0
SEPARATOR = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
PANEL_SIZE_RANGE = (11, 14)
GEN_RETRIES = 800
LAYOUT_RETRIES = 400
MAX_FRAME_DIM = 7


def _get_bg(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _neighbors4(r: int, c: int) -> tuple[tuple[int, int], ...]:
    return ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1))


def _separator_col(grid: np.ndarray, bg: int) -> int:
    h, w = map(int, grid.shape)
    candidates: list[int] = []
    for col in range(w):
        color = int(grid[0, col])
        if color == bg or not np.all(grid[:, col] == color):
            continue
        if col == 0 or col == w - 1 or col != w - col - 1:
            continue
        if np.any(np.concatenate((grid[:, :col], grid[:, col + 1 :]), axis=1) == color):
            continue
        candidates.append(col)
    assert len(candidates) == 1, "expected one unique separator column"
    return candidates[0]


def _component_coords(mask: np.ndarray) -> list[np.ndarray]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[np.ndarray] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            pts: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                pts.append((rr, cc))
                for nr, nc in _neighbors4(rr, cc):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(np.array(pts, dtype=int))
    return comps


def _find_frames(panel: np.ndarray, bg: int) -> list[Frame]:
    frames: list[Frame] = []
    for color in sorted(int(v) for v in np.unique(panel) if int(v) != bg):
        for coords in _component_coords(panel == color):
            r0 = int(coords[:, 0].min())
            c0 = int(coords[:, 1].min())
            r1 = int(coords[:, 0].max())
            c1 = int(coords[:, 1].max())
            h = r1 - r0 + 1
            w = c1 - c0 + 1
            assert h >= 3 and w >= 3, "every component must be a frame with non-empty interior"

            sub = panel[r0 : r1 + 1, c0 : c1 + 1]
            border = np.zeros((h, w), dtype=bool)
            border[0, :] = True
            border[-1, :] = True
            border[:, 0] = True
            border[:, -1] = True

            assert np.all(sub[border] == color), "frame border must be solid"
            assert np.all(sub[~border] == bg), "frame interior must be background"
            assert int(coords.shape[0]) == int(border.sum()), "component must be exactly the border"

            frames.append(Frame(color=color, bbox=(r0, c0, r1, c1), interior_area=(h - 2) * (w - 2)))
    return frames


def _analyze(grid: np.ndarray) -> dict:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _get_bg(g)
    sep_col = _separator_col(g, bg)
    left = g[:, :sep_col]
    right = g[:, sep_col + 1 :]
    assert left.shape == right.shape

    left_frames = _find_frames(left, bg)
    right_frames = _find_frames(right, bg)
    assert 3 <= len(left_frames) <= 4 and 3 <= len(right_frames) <= 4

    left_sigs = {frame.signature for frame in left_frames}
    right_sigs = {frame.signature for frame in right_frames}
    shared_sigs = left_sigs & right_sigs
    left_exclusive = [frame for frame in left_frames if frame.signature not in right_sigs]
    right_exclusive = [frame for frame in right_frames if frame.signature not in left_sigs]

    assert shared_sigs, "need at least one shared frame"
    assert left_exclusive and right_exclusive, "need exclusives on both sides"

    left_total = sum(frame.interior_area for frame in left_exclusive)
    right_total = sum(frame.interior_area for frame in right_exclusive)
    assert left_total != right_total, "exclusive totals must be different"

    winner_side = "left" if left_total > right_total else "right"
    winner_exclusive = left_exclusive if winner_side == "left" else right_exclusive
    winner_frames = left_frames if winner_side == "left" else right_frames
    largest_exclusive_area = max(frame.interior_area for frame in winner_exclusive)
    largest_exclusive = [frame for frame in winner_exclusive if frame.interior_area == largest_exclusive_area]
    assert len(largest_exclusive) == 1, "winner must have a unique largest exclusive frame"

    return {
        "bg": bg,
        "sep_col": sep_col,
        "left": left,
        "right": right,
        "left_frames": left_frames,
        "right_frames": right_frames,
        "shared_sigs": shared_sigs,
        "left_exclusive": left_exclusive,
        "right_exclusive": right_exclusive,
        "left_total": left_total,
        "right_total": right_total,
        "winner_side": winner_side,
        "winner_frames": winner_frames,
        "winner_largest_exclusive": largest_exclusive[0],
    }


def main(grid: np.ndarray) -> np.ndarray:
    """
    Compare two framed panels, ignore shared frames, and output the winning exclusive frame color.

    Input structure:
    - The grid contains two equal square panels separated by one solid vertical divider column.
    - The background is black, the divider color appears only in that middle column, and each panel contains 3 or 4 one-cell-thick hollow rectangular frames.
    - A frame is shared only when its color and panel-relative rectangle position are exactly the same on both sides.

    Rule:
    - Remove the divider and align the two panels cell by cell.
    - Mark each frame as shared or exclusive by checking whether the other panel contains the same colored rectangle at the same relative coordinates.
    - Sum the strictly interior areas of the exclusive frames on each side, choose the side with the larger sum, then choose that side's unique exclusive frame with the largest interior area.
    - Return a 2x2 solid block using that winning frame's color.

    Key insight:
    - Shared frames are decoys: they stay visible in both panels but must be excluded before any area comparison, and the output color comes from the winner's largest exclusive frame rather than from the largest visible frame overall.
    """
    info = _analyze(np.asarray(grid, dtype=int))
    color = info["winner_largest_exclusive"].color
    return np.full((2, 2), color, dtype=int)


def _draw_frame(panel: np.ndarray, slot: Slot, color: int) -> None:
    r0, c0, r1, c1 = slot.bbox
    panel[r0 : r1 + 1, c0] = color
    panel[r0 : r1 + 1, c1] = color
    panel[r0, c0 : c1 + 1] = color
    panel[r1, c0 : c1 + 1] = color


def _boxes_conflict(a: Slot, b: Slot, pad: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return not (
        ar1 + pad < br0
        or br1 + pad < ar0
        or ac1 + pad < bc0
        or bc1 + pad < ac0
    )


def _random_partition(total: int, parts: int, minimum: int) -> list[int]:
    extra = total - parts * minimum
    if extra < 0:
        raise ValueError("partition impossible")
    buckets = [minimum] * parts
    for _ in range(extra):
        buckets[random.randrange(parts)] += 1
    random.shuffle(buckets)
    return buckets


def _random_slots(panel_size: int) -> list[Slot] | None:
    max_dim = min(MAX_FRAME_DIM, panel_size)
    for _ in range(LAYOUT_RETRIES):
        layout = random.choice(("3x2", "2x3"))
        try:
            if layout == "3x2":
                row_heights = _random_partition(panel_size - 2, 3, 3)
                col_widths = _random_partition(panel_size - 1, 2, 4)
            else:
                row_heights = _random_partition(panel_size - 1, 2, 4)
                col_widths = _random_partition(panel_size - 2, 3, 3)
        except ValueError:
            continue

        row_starts: list[int] = []
        cur = 0
        for i, height in enumerate(row_heights):
            row_starts.append(cur)
            cur += height + (1 if i < len(row_heights) - 1 else 0)

        col_starts: list[int] = []
        cur = 0
        for i, width in enumerate(col_widths):
            col_starts.append(cur)
            cur += width + (1 if i < len(col_widths) - 1 else 0)

        slots: list[Slot] = []
        for r0_cell, cell_h in zip(row_starts, row_heights):
            for c0_cell, cell_w in zip(col_starts, col_widths):
                h = random.randint(3, min(cell_h, max_dim))
                w = random.randint(3, min(cell_w, max_dim))
                r0 = r0_cell + random.randint(0, cell_h - h)
                c0 = c0_cell + random.randint(0, cell_w - w)
                slots.append(Slot(r0=r0, c0=c0, h=h, w=w))

        if all(not _boxes_conflict(a, b, pad=1) for a, b in combinations(slots, 2)):
            return slots
    return None


def _choose_roles(slots: list[Slot], shared_count: int) -> dict | None:
    idxs = range(len(slots))
    candidates: list[dict] = []
    for shared_idxs in combinations(idxs, shared_count):
        remaining = [idx for idx in idxs if idx not in shared_idxs]
        for winner_idxs in combinations(remaining, 2):
            rem_after_winner = [idx for idx in remaining if idx not in winner_idxs]
            for loser_idxs in combinations(rem_after_winner, 2):
                shared = [slots[idx] for idx in shared_idxs]
                winner = [slots[idx] for idx in winner_idxs]
                loser = [slots[idx] for idx in loser_idxs]

                shared_big = max(slot.interior_area for slot in shared)
                winner_areas = sorted((slot.interior_area for slot in winner), reverse=True)
                loser_areas = sorted((slot.interior_area for slot in loser), reverse=True)

                if winner_areas[0] >= shared_big:
                    continue
                if loser_areas[0] <= winner_areas[0]:
                    continue
                if sum(winner_areas) <= sum(loser_areas):
                    continue
                if winner_areas[0] == winner_areas[1]:
                    continue

                candidates.append(
                    {
                        "shared_idxs": tuple(shared_idxs),
                        "winner_idxs": tuple(winner_idxs),
                        "loser_idxs": tuple(loser_idxs),
                        "score": (
                            int(shared_big - winner_areas[0]),
                            int(sum(winner_areas) - sum(loser_areas)),
                            -int(loser_areas[0] - winner_areas[0]),
                        ),
                    }
                )
    if not candidates:
        return None
    best_gap = min(candidate["score"][1] for candidate in candidates)
    tight = [candidate for candidate in candidates if candidate["score"][1] == best_gap]
    return random.choice(tight)


def _assign_colors(shared_count: int) -> dict[str, list[int]]:
    palette = list(FRAME_COLORS)
    random.shuffle(palette)
    shared_colors = [palette.pop() for _ in range(shared_count)]
    cross_color = palette.pop()
    winner_unique = palette.pop()
    loser_unique = palette.pop()
    return {
        "shared": shared_colors,
        "winner": [cross_color, winner_unique],
        "loser": [cross_color, loser_unique],
    }


def _build_grid(panel_size: int, slots: list[Slot], roles: dict) -> np.ndarray:
    left = np.full((panel_size, panel_size), BACKGROUND, dtype=int)
    right = np.full((panel_size, panel_size), BACKGROUND, dtype=int)
    colors = _assign_colors(len(roles["shared_idxs"]))

    shared_slots = [slots[idx] for idx in roles["shared_idxs"]]
    winner_slots = [slots[idx] for idx in roles["winner_idxs"]]
    loser_slots = [slots[idx] for idx in roles["loser_idxs"]]
    winner_side = random.choice(("left", "right"))

    for slot, color in zip(shared_slots, colors["shared"]):
        _draw_frame(left, slot, color)
        _draw_frame(right, slot, color)

    random.shuffle(winner_slots)
    random.shuffle(loser_slots)
    random.shuffle(colors["winner"])
    random.shuffle(colors["loser"])

    if winner_side == "left":
        for slot, color in zip(winner_slots, colors["winner"]):
            _draw_frame(left, slot, color)
        for slot, color in zip(loser_slots, colors["loser"]):
            _draw_frame(right, slot, color)
    else:
        for slot, color in zip(winner_slots, colors["winner"]):
            _draw_frame(right, slot, color)
        for slot, color in zip(loser_slots, colors["loser"]):
            _draw_frame(left, slot, color)

    sep = np.full((panel_size, 1), SEPARATOR, dtype=int)
    return np.concatenate((left, sep, right), axis=1)


def _semantic_checks(grid: np.ndarray) -> None:
    info = _analyze(grid)
    left_frames = info["left_frames"]
    right_frames = info["right_frames"]
    left_ex = info["left_exclusive"]
    right_ex = info["right_exclusive"]

    assert len(left_ex) == 2 and len(right_ex) == 2, "generator should force equal exclusive counts"
    assert 1 <= len(info["shared_sigs"]) <= 2

    winner_ex = left_ex if info["winner_side"] == "left" else right_ex
    loser_ex = right_ex if info["winner_side"] == "left" else left_ex
    winner_frames = info["winner_frames"]
    winner_color = info["winner_largest_exclusive"].color

    winner_overall = max(winner_frames, key=lambda frame: frame.interior_area)
    assert winner_overall.signature in info["shared_sigs"], "shared frame must dominate the visible winner panel"
    assert winner_overall.color != winner_color, "output color must differ from the largest visible winner frame"

    winner_max = max(frame.interior_area for frame in winner_ex)
    loser_max = max(frame.interior_area for frame in loser_ex)
    assert loser_max > winner_max, "largest single exclusive frame should belong to the losing side"
    assert sum(frame.interior_area for frame in winner_ex) > sum(frame.interior_area for frame in loser_ex)

    assert any(
        left_frame.color == right_frame.color and left_frame.signature != right_frame.signature
        for left_frame in left_ex
        for right_frame in right_ex
    ), "same-color exclusive distractor should appear on both sides"

    out = main(grid)
    assert out.shape == (2, 2)
    assert np.all(out == winner_color)


def _attempt() -> np.ndarray | None:
    panel_size = random.randint(*PANEL_SIZE_RANGE)
    shared_count = random.choice((1, 2))
    slots = _random_slots(panel_size)
    if slots is None:
        return None

    roles = _choose_roles(slots, shared_count)
    if roles is None:
        return None

    grid = _build_grid(panel_size, slots, roles)
    try:
        _semantic_checks(grid)
    except AssertionError:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input for new puzzle 5")
