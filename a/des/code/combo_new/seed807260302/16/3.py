import random
from typing import List, Tuple

import numpy as np


BG = 0
SIDES = ("top", "right", "bottom", "left")
OPPOSITE = {"top": "bottom", "bottom": "top", "left": "right", "right": "left"}
DIRS4 = [(-1, 0), (1, 0), (0, -1), (0, 1)]
GEN_RETRIES = 400


def _components_mask(mask: np.ndarray) -> List[List[Tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: List[List[Tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            coords: List[Tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                coords.append((rr, cc))
                for dr, dc in DIRS4:
                    nr, nc = rr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(coords)
    return comps


def _bbox(coords: List[Tuple[int, int]]) -> Tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _false_runs(line: np.ndarray) -> List[Tuple[int, int]]:
    runs: List[Tuple[int, int]] = []
    start = None
    for idx, value in enumerate(line.tolist()):
        if not value and start is None:
            start = idx
        elif value and start is not None:
            runs.append((start, idx - 1))
            start = None
    if start is not None:
        runs.append((start, len(line) - 1))
    return runs


def _make_frame_mask(h: int, w: int, ports: List[dict]) -> np.ndarray:
    mask = np.zeros((h, w), dtype=bool)
    mask[0, :] = True
    mask[-1, :] = True
    mask[:, 0] = True
    mask[:, -1] = True
    for port in ports:
        side = port["side"]
        a0 = int(port["start"])
        a1 = int(port["end"])
        if side == "top":
            mask[0, a0 : a1 + 1] = False
        elif side == "bottom":
            mask[-1, a0 : a1 + 1] = False
        elif side == "left":
            mask[a0 : a1 + 1, 0] = False
        elif side == "right":
            mask[a0 : a1 + 1, -1] = False
        else:
            raise AssertionError(f"unknown side: {side}")
    return mask


def _detect_ports(sprite: np.ndarray) -> List[dict]:
    h, w = sprite.shape
    assert h >= 4 and w >= 4
    assert bool(sprite[0, 0] and sprite[0, -1] and sprite[-1, 0] and sprite[-1, -1])
    assert not bool(sprite[1:-1, 1:-1].any())

    ports: List[dict] = []
    checks = [
        ("top", sprite[0, :]),
        ("bottom", sprite[-1, :]),
        ("left", sprite[:, 0]),
        ("right", sprite[:, -1]),
    ]
    for side, line in checks:
        inner = line[1:-1]
        runs = _false_runs(inner)
        if not runs:
            assert bool(inner.all())
            continue
        assert len(runs) == 1
        start, end = runs[0]
        if start > 0:
            assert bool(inner[:start].all())
        if end + 1 < len(inner):
            assert bool(inner[end + 1 :].all())
        ports.append(
            {
                "side": side,
                "start": int(start + 1),
                "end": int(end + 1),
                "width": int(end - start + 1),
            }
        )

    recon = _make_frame_mask(h, w, ports)
    assert np.array_equal(recon, sprite)
    assert 1 <= len(ports) <= 2
    return ports


def _crop_non_bg(grid: np.ndarray, bg: int = BG) -> np.ndarray:
    pts = np.argwhere(grid != bg)
    assert len(pts) > 0
    r0, c0 = pts.min(axis=0)
    r1, c1 = pts.max(axis=0)
    return grid[int(r0) : int(r1) + 1, int(c0) : int(c1) + 1]


def _extract_ordered_frames(grid: np.ndarray) -> List[dict]:
    colors = sorted(set(map(int, grid.flatten())) - {BG})
    assert colors

    frames_by_color = {}
    hints = []
    for color in colors:
        comps = _components_mask(grid == int(color))
        assert len(comps) >= 2
        hint_comps = [comp for comp in comps if len(comp) == 1]
        frame_comps = [comp for comp in comps if len(comp) > 1]
        assert len(hint_comps) == 1
        assert len(frame_comps) >= 1

        hr, hc = hint_comps[0][0]
        hints.append((int(hr), int(hc), int(color)))

        coords = [pt for comp in frame_comps for pt in comp]
        r0, c0, r1, c1 = _bbox(coords)
        sprite = (grid[r0 : r1 + 1, c0 : c1 + 1] == int(color))
        ports = _detect_ports(sprite)
        frames_by_color[int(color)] = {
            "color": int(color),
            "sprite": sprite,
            "ports": ports,
            "h": int(sprite.shape[0]),
            "w": int(sprite.shape[1]),
        }

    hints.sort(key=lambda item: (item[0], item[1]))
    order = [color for _r, _c, color in hints]
    assert set(order) == set(frames_by_color)
    return [frames_by_color[color] for color in order]


def _choose_links(frames: List[dict]) -> List[Tuple[int, int]]:
    used = [set() for _ in frames]
    links: List[Tuple[int, int]] = []
    for idx in range(len(frames) - 1):
        candidates = []
        for a_idx, a_port in enumerate(frames[idx]["ports"]):
            if a_idx in used[idx]:
                continue
            for b_idx, b_port in enumerate(frames[idx + 1]["ports"]):
                if b_idx in used[idx + 1]:
                    continue
                if OPPOSITE[a_port["side"]] != b_port["side"]:
                    continue
                if int(a_port["width"]) != int(b_port["width"]):
                    continue
                candidates.append((a_idx, b_idx))
        assert len(candidates) == 1
        a_idx, b_idx = candidates[0]
        used[idx].add(a_idx)
        used[idx + 1].add(b_idx)
        links.append((a_idx, b_idx))
    return links


def _place_frames(frames: List[dict], links: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
    positions: List[Tuple[int, int]] = [(0, 0)]
    for idx, (a_idx, b_idx) in enumerate(links):
        frame_a = frames[idx]
        frame_b = frames[idx + 1]
        port_a = frame_a["ports"][a_idx]
        port_b = frame_b["ports"][b_idx]
        ar, ac = positions[idx]

        if port_a["side"] == "right":
            br = ar + int(port_a["start"]) - int(port_b["start"])
            bc = ac + int(frame_a["w"]) + 1
        elif port_a["side"] == "left":
            br = ar + int(port_a["start"]) - int(port_b["start"])
            bc = ac - int(frame_b["w"]) - 1
        elif port_a["side"] == "bottom":
            br = ar + int(frame_a["h"]) + 1
            bc = ac + int(port_a["start"]) - int(port_b["start"])
        elif port_a["side"] == "top":
            br = ar - int(frame_b["h"]) - 1
            bc = ac + int(port_a["start"]) - int(port_b["start"])
        else:
            raise AssertionError(f"unknown side: {port_a['side']}")
        positions.append((int(br), int(bc)))
    return positions


def _render_chain(frames: List[dict], links: List[Tuple[int, int]], positions: List[Tuple[int, int]]) -> np.ndarray:
    min_r = min(r for r, _c in positions)
    min_c = min(c for _r, c in positions)
    norm_positions = [(r - min_r, c - min_c) for r, c in positions]
    max_r = max(r + int(frame["h"]) for (r, _c), frame in zip(norm_positions, frames))
    max_c = max(c + int(frame["w"]) for (_r, c), frame in zip(norm_positions, frames))
    out = np.full((int(max_r), int(max_c)), BG, dtype=int)

    for frame, (r, c) in zip(frames, norm_positions):
        sprite = frame["sprite"]
        window = out[r : r + frame["h"], c : c + frame["w"]]
        window[sprite] = int(frame["color"])

    for idx, (a_idx, _b_idx) in enumerate(links):
        frame_a = frames[idx]
        frame_b = frames[idx + 1]
        port_a = frame_a["ports"][a_idx]
        r, c = norm_positions[idx]
        bridge_color = int(frame_b["color"])

        if port_a["side"] == "right":
            out[r + port_a["start"] : r + port_a["end"] + 1, c + frame_a["w"]] = bridge_color
        elif port_a["side"] == "left":
            out[r + port_a["start"] : r + port_a["end"] + 1, c - 1] = bridge_color
        elif port_a["side"] == "bottom":
            out[r + frame_a["h"], c + port_a["start"] : c + port_a["end"] + 1] = bridge_color
        elif port_a["side"] == "top":
            out[r - 1, c + port_a["start"] : c + port_a["end"] + 1] = bridge_color
        else:
            raise AssertionError(f"unknown side: {port_a['side']}")

    return _crop_non_bg(out, BG)


def main(grid: np.ndarray) -> np.ndarray:
    """Assemble the gapped rectangular frames into an ordered bridge-linked chain.

    Input structure:
    - The background is black.
    - Several differently colored hollow rectangular frames are scattered on the canvas.
    - Each frame has a continuous 2-3 cell gap on one border side, and the corner hint column contains one single-pixel sample for every frame color.

    Rule:
    - Read the hint pixels from top to bottom to determine the frame order.
    - For each consecutive pair, translate the later frame so its gap faces the current frame's gap from the opposite side, keeping the two gaps aligned and leaving a one-cell bridge space.
    - Fill that bridge space with the later frame's color, preserve each frame's shape, and crop the final result to the smallest bounding box covering the full assembled chain.

    Key insight:
    - The corner color column is not decoration; it fixes the linking order, while each frame gap acts as a directional port that determines where the next frame must attach and what bridge span gets filled.
    """
    g = np.asarray(grid, dtype=int)
    assert g.ndim == 2

    frames = _extract_ordered_frames(g)
    links = _choose_links(frames)
    positions = _place_frames(frames, links)
    out = _render_chain(frames, links, positions)
    assert out.ndim == 2 and max(out.shape) <= 30
    return out


def _sample_chain_spec(rng: random.Random):
    n_frames = rng.choice([2, 3, 4])
    first_side = rng.choice(list(SIDES))
    opposite_side = OPPOSITE[first_side]
    gap_width = rng.choice([2, 3])
    horizontal = first_side in ("left", "right")

    for _ in range(100):
        if horizontal:
            widths = [rng.randint(5, 7 if n_frames <= 3 else 6) for _ in range(n_frames)]
            heights = [rng.randint(max(5, gap_width + 2), 8) for _ in range(n_frames)]
            if sum(widths) + (n_frames - 1) > 24:
                continue
            max_gap_start = min(h - gap_width - 1 for h in heights)
        else:
            heights = [rng.randint(5, 7 if n_frames <= 3 else 6) for _ in range(n_frames)]
            widths = [rng.randint(max(5, gap_width + 2), 8) for _ in range(n_frames)]
            if sum(heights) + (n_frames - 1) > 24:
                continue
            max_gap_start = min(w - gap_width - 1 for w in widths)
        if max_gap_start < 1:
            continue

        gap_start = rng.randint(1, max_gap_start)
        colors = rng.sample(list(range(1, 10)), n_frames)
        frames = []
        for idx in range(n_frames):
            if idx == 0:
                sides = [first_side]
            elif idx == n_frames - 1:
                sides = [opposite_side]
            else:
                sides = [opposite_side, first_side]
            ports = [
                {
                    "side": side,
                    "start": gap_start,
                    "end": gap_start + gap_width - 1,
                    "width": gap_width,
                }
                for side in sides
            ]
            h = heights[idx]
            w = widths[idx]
            frames.append(
                {
                    "color": int(colors[idx]),
                    "ports": ports,
                    "h": int(h),
                    "w": int(w),
                    "sprite": _make_frame_mask(int(h), int(w), ports),
                }
            )

        links = _choose_links(frames)
        positions = _place_frames(frames, links)
        out = _render_chain(frames, links, positions)
        if max(out.shape) <= 30:
            return frames
    return None


def _reserve(occ: np.ndarray, r: int, c: int, h: int, w: int, pad: int = 1) -> bool:
    hh, ww = occ.shape
    r0 = max(0, r - pad)
    c0 = max(0, c - pad)
    r1 = min(hh, r + h + pad)
    c1 = min(ww, c + w + pad)
    if occ[r0:r1, c0:c1].any():
        return False
    occ[r0:r1, c0:c1] = True
    return True


def _hint_positions(h: int, w: int, n: int, rng: random.Random) -> List[Tuple[int, int]]:
    step = rng.choice([1, 2])
    height = 1 + (n - 1) * step
    corner = rng.choice(["tl", "tr", "bl", "br"])
    if corner in ("tl", "tr"):
        row0 = 0
    else:
        row0 = h - height
    col = 0 if corner in ("tl", "bl") else w - 1
    return [(int(row0 + idx * step), int(col)) for idx in range(n)]


def _scatter_frames(frames: List[dict], rng: random.Random):
    n = len(frames)
    for _ in range(120):
        canvas_h = rng.randint(20, 30)
        canvas_w = rng.randint(20, 30)
        hint_pts = _hint_positions(canvas_h, canvas_w, n, rng)
        occ = np.zeros((canvas_h, canvas_w), dtype=bool)
        hint_rows = [r for r, _ in hint_pts]
        hint_cols = [c for _, c in hint_pts]
        if not _reserve(
            occ,
            min(hint_rows),
            min(hint_cols),
            max(hint_rows) - min(hint_rows) + 1,
            max(hint_cols) - min(hint_cols) + 1,
            pad=1,
        ):
            continue

        ok = True
        placements = {}
        order = list(range(n))
        rng.shuffle(order)
        for idx in sorted(order, key=lambda ii: frames[ii]["h"] * frames[ii]["w"], reverse=True):
            frame = frames[idx]
            placed = False
            for _ in range(200):
                r = rng.randint(0, canvas_h - frame["h"])
                c = rng.randint(0, canvas_w - frame["w"])
                if _reserve(occ, r, c, frame["h"], frame["w"], pad=1):
                    placements[idx] = (int(r), int(c))
                    placed = True
                    break
            if not placed:
                ok = False
                break
        if not ok:
            continue

        grid = np.zeros((canvas_h, canvas_w), dtype=int)
        for idx, frame in enumerate(frames):
            r, c = placements[idx]
            window = grid[r : r + frame["h"], c : c + frame["w"]]
            window[frame["sprite"]] = int(frame["color"])
        for (r, c), frame in zip(hint_pts, frames):
            grid[r, c] = int(frame["color"])
        return grid
    return None


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(GEN_RETRIES):
        frames = _sample_chain_spec(rng)
        if frames is None:
            continue
        grid = _scatter_frames(frames, rng)
        if grid is None:
            continue
        try:
            out = main(grid)
        except AssertionError:
            continue
        if grid.ndim != 2 or out.ndim != 2:
            continue
        if max(grid.shape) > 30 or max(out.shape) > 30:
            continue
        return grid
    raise RuntimeError("generate_input failed to sample a valid chain puzzle")
