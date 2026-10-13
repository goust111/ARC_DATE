from __future__ import annotations

import random

import numpy as np


_RECEIVER_COLOR = 5  # ARC grey
_RING = (
    (0, 0), (0, 1), (0, 2),
    (1, 0),         (1, 2),
    (2, 0), (2, 1), (2, 2),
)
_MARKER_TO_DIR = {
    (0, 1): (1, 0),
    (2, 1): (-1, 0),
    (1, 0): (0, 1),
    (1, 2): (0, -1),
}
_DIR_TO_MARKER = {direction: marker for marker, direction in _MARKER_TO_DIR.items()}
_GEN_ATTEMPTS = 300


def _mode_color(grid: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(grid), return_counts=True)
    return int(values[counts.argmax()])


def _find_receivers(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    receivers: list[dict[str, object]] = []
    for top in range(h - 2):
        for left in range(w - 2):
            patch = grid[top : top + 3, left : left + 3]
            if int(patch[1, 1]) != bg:
                continue
            frame = [int(patch[r, c]) for r, c in _RING]
            frame_color = frame[0]
            if frame_color == bg:
                continue
            if all(color == frame_color for color in frame):
                receivers.append(
                    {
                        "top": top,
                        "left": left,
                        "frame_color": frame_color,
                        "center": (top + 1, left + 1),
                    }
                )
    return receivers


def _find_transmitters(grid: np.ndarray, bg: int) -> list[dict[str, object]]:
    h, w = grid.shape
    transmitters: list[dict[str, object]] = []
    for top in range(h - 2):
        for left in range(w - 2):
            patch = grid[top : top + 3, left : left + 3]
            colors, counts = np.unique(patch, return_counts=True)
            if len(colors) != 2:
                continue

            color_counts = {int(color): int(count) for color, count in zip(colors, counts)}
            if sorted(color_counts.values()) != [1, 8]:
                continue

            marker_color = next(color for color, count in color_counts.items() if count == 1)
            body_color = next(color for color, count in color_counts.items() if count == 8)
            if marker_color == bg or body_color == bg:
                continue

            singleton = np.argwhere(patch == marker_color)
            if singleton.shape != (1, 2):
                continue
            mr, mc = map(int, singleton[0])
            if (mr, mc) not in _MARKER_TO_DIR:
                continue

            if any(
                int(patch[r, c]) != body_color
                for r in range(3)
                for c in range(3)
                if (r, c) != (mr, mc)
            ):
                continue

            dr, dc = _MARKER_TO_DIR[(mr, mc)]
            start = (top + 1 + 2 * dr, left + 1 + 2 * dc)
            transmitters.append(
                {
                    "top": top,
                    "left": left,
                    "body_color": body_color,
                    "marker_color": marker_color,
                    "direction": (dr, dc),
                    "start": start,
                }
            )
    return transmitters


def _receiver_frame_lookup(receivers: list[dict[str, object]]) -> dict[tuple[int, int], tuple[int, int]]:
    frame_lookup: dict[tuple[int, int], tuple[int, int]] = {}
    for receiver in receivers:
        top = int(receiver["top"])
        left = int(receiver["left"])
        center = receiver["center"]
        assert isinstance(center, tuple)
        for rr in range(top, top + 3):
            for cc in range(left, left + 3):
                if rr == top + 1 and cc == left + 1:
                    continue
                previous = frame_lookup.get((rr, cc))
                assert previous is None or previous == center
                frame_lookup[(rr, cc)] = center
    return frame_lookup


def main(grid: np.ndarray) -> np.ndarray:
    """
    In a grid containing 3x3 transmitters and 3x3 receivers, each transmitter emits
    a straight colored ray opposite its edge marker; the ray paints its path and, if
    it reaches a receiver frame, fills that receiver's center with the ray color.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    bg = _mode_color(g)
    receivers = _find_receivers(g, bg)
    transmitters = _find_transmitters(g, bg)
    assert receivers, "expected at least one receiver"
    assert transmitters, "expected at least one transmitter"

    frame_lookup = _receiver_frame_lookup(receivers)
    out = g.copy()
    h, w = g.shape

    for transmitter in sorted(transmitters, key=lambda item: (int(item["top"]), int(item["left"]))):
        rr, cc = transmitter["start"]
        dr, dc = transmitter["direction"]
        color = int(transmitter["marker_color"])

        while 0 <= rr < h and 0 <= cc < w:
            receiver_center = frame_lookup.get((rr, cc))
            if receiver_center is not None:
                out[receiver_center[0], receiver_center[1]] = color
                break

            if int(g[rr, cc]) != bg:
                break

            out[rr, cc] = color
            rr += dr
            cc += dc

    return out


def _draw_receiver(grid: np.ndarray, top: int, left: int, bg: int) -> tuple[int, int]:
    patch = np.full((3, 3), bg, dtype=int)
    patch[0, :] = _RECEIVER_COLOR
    patch[2, :] = _RECEIVER_COLOR
    patch[:, 0] = _RECEIVER_COLOR
    patch[:, 2] = _RECEIVER_COLOR
    grid[top : top + 3, left : left + 3] = patch
    return top + 1, left + 1


def _draw_transmitter(
    grid: np.ndarray,
    top: int,
    left: int,
    body_color: int,
    marker_color: int,
    direction: tuple[int, int],
) -> tuple[int, int]:
    patch = np.full((3, 3), body_color, dtype=int)
    mr, mc = _DIR_TO_MARKER[direction]
    patch[mr, mc] = marker_color
    grid[top : top + 3, left : left + 3] = patch
    dr, dc = direction
    return top + 1 + 2 * dr, left + 1 + 2 * dc


def _sample_lane_centers(length: int, count: int) -> list[int] | None:
    centers = list(range(2, length - 1, 4))
    if len(centers) < count:
        return None
    return random.sample(centers, count)


def _place_spare_receiver(grid: np.ndarray, axis: str, lane: int, bg: int) -> tuple[int, int]:
    h, w = grid.shape
    if axis == "h":
        top = lane - 1
        left = random.randint(1, w - 3)
    else:
        top = random.randint(1, h - 3)
        left = lane - 1
    return _draw_receiver(grid, top, left, bg)


def _place_emitter(
    grid: np.ndarray,
    axis: str,
    lane: int,
    direction: tuple[int, int],
    hit_receiver: bool,
    body_color: int,
    marker_color: int,
    bg: int,
) -> dict[str, object]:
    h, w = grid.shape
    receiver_center: tuple[int, int] | None = None

    if axis == "h":
        row = lane
        if direction == (0, 1):
            top = row - 1
            left = random.randint(1, w - 7) if hit_receiver else random.randint(1, w - 5)
            if hit_receiver:
                receiver_left = random.randint(left + 4, w - 3)
                receiver_center = _draw_receiver(grid, row - 1, receiver_left, bg)
        else:
            top = row - 1
            left = random.randint(5, w - 4) if hit_receiver else random.randint(2, w - 4)
            if hit_receiver:
                receiver_left = random.randint(1, left - 4)
                receiver_center = _draw_receiver(grid, row - 1, receiver_left, bg)
    else:
        col = lane
        if direction == (1, 0):
            left = col - 1
            top = random.randint(1, h - 7) if hit_receiver else random.randint(1, h - 5)
            if hit_receiver:
                receiver_top = random.randint(top + 4, h - 3)
                receiver_center = _draw_receiver(grid, receiver_top, col - 1, bg)
        else:
            left = col - 1
            top = random.randint(5, h - 4) if hit_receiver else random.randint(2, h - 4)
            if hit_receiver:
                receiver_top = random.randint(1, top - 4)
                receiver_center = _draw_receiver(grid, receiver_top, col - 1, bg)

    start = _draw_transmitter(grid, top, left, body_color, marker_color, direction)
    return {
        "top": top,
        "left": left,
        "direction": direction,
        "marker_color": marker_color,
        "start": start,
        "hit": hit_receiver,
        "receiver_center": receiver_center,
    }


def _last_boundary_cell(start: tuple[int, int], direction: tuple[int, int], shape: tuple[int, int]) -> tuple[int, int]:
    h, w = shape
    rr, cc = start
    dr, dc = direction
    while 0 <= rr + dr < h and 0 <= cc + dc < w:
        rr += dr
        cc += dc
    return rr, cc


def _validate_generated(grid: np.ndarray, specs: list[dict[str, object]], expected_receivers: int) -> bool:
    bg = _mode_color(grid)
    transmitters = _find_transmitters(grid, bg)
    receivers = _find_receivers(grid, bg)
    if len(transmitters) != len(specs) or len(receivers) != expected_receivers:
        return False

    out = main(grid)
    if np.array_equal(out, grid):
        return False

    hit_centers = set()
    for spec in specs:
        marker_color = int(spec["marker_color"])
        if bool(spec["hit"]):
            center = spec["receiver_center"]
            if center is None or int(out[center[0], center[1]]) != marker_color:
                return False
            hit_centers.add(center)
        else:
            boundary = _last_boundary_cell(spec["start"], spec["direction"], grid.shape)
            if int(out[boundary[0], boundary[1]]) != marker_color:
                return False

    for receiver in receivers:
        center = receiver["center"]
        if center not in hit_centers and int(out[center[0], center[1]]) != bg:
            return False

    return True


def _attempt_generate() -> np.ndarray | None:
    axis = random.choice(("h", "v"))
    n_emitters = random.choice((2, 4))
    extra_receivers = random.choice((1, 2)) if n_emitters == 2 else random.choice((0, 1))
    n_lanes = n_emitters + extra_receivers

    if axis == "h":
        min_h = 4 * (n_lanes - 1) + 5
        h = random.randint(min_h, min_h + 6)
        w = random.randint(16, 26)
        lane_centers = _sample_lane_centers(h, n_lanes)
    else:
        min_w = 4 * (n_lanes - 1) + 5
        h = random.randint(16, 26)
        w = random.randint(min_w, min_w + 6)
        lane_centers = _sample_lane_centers(w, n_lanes)

    if lane_centers is None:
        return None

    bg_choices = [color for color in range(10) if color != _RECEIVER_COLOR]
    bg = random.choice(bg_choices)
    palette = [color for color in range(10) if color not in (bg, _RECEIVER_COLOR)]
    random.shuffle(palette)
    body_colors = palette[:n_emitters]
    marker_colors = palette[n_emitters : 2 * n_emitters]

    grid = np.full((h, w), bg, dtype=int)
    random.shuffle(lane_centers)
    emitter_lanes = lane_centers[:n_emitters]
    spare_receiver_lanes = lane_centers[n_emitters:]

    for lane in spare_receiver_lanes:
        _place_spare_receiver(grid, axis, lane, bg)

    hit_flags = [True] * (n_emitters // 2) + [False] * (n_emitters // 2)
    random.shuffle(hit_flags)

    if axis == "h":
        directions = [(0, 1)] * (n_emitters // 2) + [(0, -1)] * (n_emitters // 2)
    else:
        directions = [(1, 0)] * (n_emitters // 2) + [(-1, 0)] * (n_emitters // 2)
    random.shuffle(directions)

    specs: list[dict[str, object]] = []
    for lane, direction, hit_receiver, body_color, marker_color in zip(
        emitter_lanes, directions, hit_flags, body_colors, marker_colors
    ):
        specs.append(
            _place_emitter(
                grid=grid,
                axis=axis,
                lane=lane,
                direction=direction,
                hit_receiver=hit_receiver,
                body_color=body_color,
                marker_color=marker_color,
                bg=bg,
            )
        )

    expected_receivers = extra_receivers + sum(bool(spec["hit"]) for spec in specs)
    if not _validate_generated(grid, specs, expected_receivers):
        return None

    return grid


def generate_input() -> np.ndarray:
    for _ in range(_GEN_ATTEMPTS):
        grid = _attempt_generate()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid transmitter/receiver ARC input")
