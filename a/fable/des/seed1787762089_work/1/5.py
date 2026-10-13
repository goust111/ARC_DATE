import random

import numpy as np


BG = 0
SHAPE_COLOR = 2
SEP_COLOR = 5
MIRROR_COLOR = 8
GEN_RETRIES = 400


def _rot_cw(offsets):
    return frozenset((c, 1 - r) for r, c in offsets)


BASE_ORIENTATION = frozenset({(0, 0), (1, 0), (1, 1)})
ORIENTATIONS = [BASE_ORIENTATION]
for _ in range(3):
    ORIENTATIONS.append(_rot_cw(ORIENTATIONS[-1]))
ORIENTATION_TO_INDEX = {shape: idx for idx, shape in enumerate(ORIENTATIONS)}


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    return int(values[np.argmax(counts)])


def _split_panels(grid: np.ndarray, bg: int):
    h, w = grid.shape
    assert (w - 3) % 4 == 0
    panel_w = (w - 3) // 4
    sep_cols = [panel_w, 2 * panel_w + 1, 3 * panel_w + 2]
    sep_color = int(grid[0, sep_cols[0]])
    assert sep_color != bg
    for col in sep_cols:
        assert np.all(grid[:, col] == sep_color)
    panels = []
    for idx in range(4):
        c0 = idx * (panel_w + 1)
        panels.append(grid[:, c0 : c0 + panel_w])
    return panels, sep_color


def _extract_l(panel: np.ndarray, bg: int):
    coords = np.argwhere(panel != bg)
    assert coords.shape[0] == 3
    color = int(panel[coords[0, 0], coords[0, 1]])
    assert np.all(panel[coords[:, 0], coords[:, 1]] == color)
    r0, c0 = coords.min(axis=0)
    r1, c1 = coords.max(axis=0)
    assert int(r1 - r0) == 1 and int(c1 - c0) == 1
    offsets = frozenset((int(r - r0), int(c - c0)) for r, c in coords)
    assert offsets in ORIENTATION_TO_INDEX
    return color, (int(r0), int(c0)), offsets


def _find_mirror_line(panel: np.ndarray, bg: int):
    non_bg = int(np.count_nonzero(panel != bg))
    vert = [
        c
        for c in range(panel.shape[1])
        if int(panel[0, c]) != bg and np.all(panel[:, c] == panel[0, c])
    ]
    hori = [
        r
        for r in range(panel.shape[0])
        if int(panel[r, 0]) != bg and np.all(panel[r, :] == panel[r, 0])
    ]
    if len(vert) == 1 and non_bg == panel.shape[0]:
        c = vert[0]
        return "v", int(c), int(panel[0, c])
    if len(hori) == 1 and non_bg == panel.shape[1]:
        r = hori[0]
        return "h", int(r), int(panel[r, 0])
    raise AssertionError("template panel must contain exactly one mirror line")


def main(grid: np.ndarray) -> np.ndarray:
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2
    bg = _dominant_color(g)
    panels, _sep_color = _split_panels(g, bg)

    frames = [_extract_l(panel, bg) for panel in panels[:3]]
    shape_color = frames[0][0]
    assert all(frame[0] == shape_color for frame in frames)

    positions = [frame[1] for frame in frames]
    delta1 = (
        positions[1][0] - positions[0][0],
        positions[1][1] - positions[0][1],
    )
    delta2 = (
        positions[2][0] - positions[1][0],
        positions[2][1] - positions[1][1],
    )
    assert delta1 == delta2
    next_pos = (positions[2][0] + delta1[0], positions[2][1] + delta1[1])

    ori = [ORIENTATION_TO_INDEX[frame[2]] for frame in frames]
    rot1 = (ori[1] - ori[0]) % 4
    rot2 = (ori[2] - ori[1]) % 4
    assert rot1 == rot2 and rot1 in (1, 3)
    next_ori = (ori[2] + rot1) % 4

    out = panels[3].copy()
    axis, line_idx, mirror_color = _find_mirror_line(out, bg)

    for dr, dc in ORIENTATIONS[next_ori]:
        rr = next_pos[0] + dr
        cc = next_pos[1] + dc
        assert 0 <= rr < out.shape[0] and 0 <= cc < out.shape[1]
        assert (axis == "v" and cc != line_idx) or (axis == "h" and rr != line_idx)
        out[rr, cc] = shape_color

        if axis == "v":
            mr, mc = rr, 2 * line_idx - cc
        else:
            mr, mc = 2 * line_idx - rr, cc
        assert 0 <= mr < out.shape[0] and 0 <= mc < out.shape[1]
        assert out[mr, mc] == bg
        out[mr, mc] = mirror_color

    return out


def _valid_template_positions(height: int, width: int, axis: str, line_idx: int):
    positions = []
    for r0 in range(height - 1):
        for c0 in range(width - 1):
            if axis == "v":
                if not (c0 + 1 < line_idx or c0 > line_idx):
                    continue
                mirror_c0 = 2 * line_idx - c0 - 1
                if 0 <= mirror_c0 <= width - 2:
                    positions.append((r0, c0))
            else:
                if not (r0 + 1 < line_idx or r0 > line_idx):
                    continue
                mirror_r0 = 2 * line_idx - r0 - 1
                if 0 <= mirror_r0 <= height - 2:
                    positions.append((r0, c0))
    return positions


def _draw_shape(panel: np.ndarray, top_left, orientation_idx: int, color: int):
    r0, c0 = top_left
    for dr, dc in ORIENTATIONS[orientation_idx]:
        panel[r0 + dr, c0 + dc] = color


def _gen_attempt():
    height = random.choice([6, 7, 8])
    width = random.choice([8, 9, 10])
    axis = random.choice(["v", "h"])
    line_idx = random.randint(2, (width if axis == "v" else height) - 3)
    positions4 = _valid_template_positions(height, width, axis, line_idx)
    if not positions4:
        return None

    deltas = [
        (dr, dc)
        for dr in (-1, 0, 1)
        for dc in (-1, 0, 1)
        if (dr, dc) != (0, 0)
    ] + [(-2, 0), (2, 0), (0, -2), (0, 2)]
    random.shuffle(positions4)
    random.shuffle(deltas)

    chosen = None
    for pos4 in positions4:
        for dr, dc in deltas:
            pos1 = (pos4[0] - 3 * dr, pos4[1] - 3 * dc)
            if not (0 <= pos1[0] <= height - 2 and 0 <= pos1[1] <= width - 2):
                continue
            pos2 = (pos1[0] + dr, pos1[1] + dc)
            pos3 = (pos2[0] + dr, pos2[1] + dc)
            if not (
                0 <= pos2[0] <= height - 2
                and 0 <= pos2[1] <= width - 2
                and 0 <= pos3[0] <= height - 2
                and 0 <= pos3[1] <= width - 2
            ):
                continue
            chosen = (pos1, pos2, pos3, pos4, (dr, dc))
            break
        if chosen is not None:
            break
    if chosen is None:
        return None

    pos1, pos2, pos3, _pos4, _delta = chosen
    start_ori = random.randrange(4)
    rot_step = random.choice([1, -1])
    orientations = [
        start_ori,
        (start_ori + rot_step) % 4,
        (start_ori + 2 * rot_step) % 4,
    ]

    grid = np.full((height, 4 * width + 3), BG, dtype=int)
    for col in (width, 2 * width + 1, 3 * width + 2):
        grid[:, col] = SEP_COLOR

    for panel_idx, (top_left, ori_idx) in enumerate(zip((pos1, pos2, pos3), orientations)):
        c0 = panel_idx * (width + 1)
        panel = grid[:, c0 : c0 + width]
        _draw_shape(panel, top_left, ori_idx, SHAPE_COLOR)

    template_c0 = 3 * (width + 1)
    if axis == "v":
        grid[:, template_c0 + line_idx] = MIRROR_COLOR
    else:
        grid[line_idx, template_c0 : template_c0 + width] = MIRROR_COLOR

    try:
        out = main(grid)
    except AssertionError:
        return None

    if out.shape != (height, width):
        return None
    if int(np.count_nonzero(out == SHAPE_COLOR)) != 3:
        return None
    expected_mirror = (height if axis == "v" else width) + 3
    if int(np.count_nonzero(out == MIRROR_COLOR)) != expected_mirror:
        return None
    return grid


def generate_input() -> np.ndarray:
    for _ in range(GEN_RETRIES):
        grid = _gen_attempt()
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid input")
