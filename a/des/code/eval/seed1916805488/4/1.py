import random
from collections import deque

import numpy as np


BLACK = 0
GREY = 5
MAROON = 9
PALETTE_POOL = [1, 2, 3, 4, 6, 7, 8]
POSITIONS = ("tl", "tr", "bl", "br")
MAX_GEN_ATTEMPTS = 400


def _components4(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not mask[r, c] or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                comp.append((cr, cc))
                for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
                    if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] and not seen[nr, nc]:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(points: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in points]
    cols = [c for _, c in points]
    return min(rows), min(cols), max(rows), max(cols)


def _find_frames(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    frames: list[tuple[int, int, int, int]] = []
    for comp in _components4(grid == GREY):
        r0, c0, r1, c1 = _bbox(comp)
        if r1 - r0 + 1 < 6 or c1 - c0 + 1 < 6:
            continue
        box = grid[r0 : r1 + 1, c0 : c1 + 1]
        if not np.all(box[0, :] == GREY) or not np.all(box[-1, :] == GREY):
            continue
        if not np.all(box[:, 0] == GREY) or not np.all(box[:, -1] == GREY):
            continue
        if np.any(box[1:-1, 1:-1] == GREY):
            continue
        frames.append((r0, c0, r1, c1))
    frames.sort()
    return frames


def _quadrant_slices(frame: tuple[int, int, int, int]) -> dict[str, tuple[slice, slice]]:
    r0, c0, r1, c1 = frame
    ih = r1 - r0 - 1
    iw = c1 - c0 - 1
    assert ih % 2 == 0 and iw % 2 == 0, "frame interior must split evenly into 2x2"
    qh, qw = ih // 2, iw // 2
    ir0, ic0 = r0 + 1, c0 + 1
    return {
        "tl": (slice(ir0, ir0 + qh), slice(ic0, ic0 + qw)),
        "tr": (slice(ir0, ir0 + qh), slice(ic0 + qw, ic0 + 2 * qw)),
        "bl": (slice(ir0 + qh, ir0 + 2 * qh), slice(ic0, ic0 + qw)),
        "br": (slice(ir0 + qh, ir0 + 2 * qh), slice(ic0 + qw, ic0 + 2 * qw)),
    }


def _palette_colors(grid: np.ndarray) -> set[int]:
    edge_values = []
    edge_values.extend(int(v) for v in grid[0, :])
    edge_values.extend(int(v) for v in grid[-1, :])
    edge_values.extend(int(v) for v in grid[:, 0])
    edge_values.extend(int(v) for v in grid[:, -1])
    return {v for v in edge_values if v not in {BLACK, GREY, MAROON}}


def _analyze(grid: np.ndarray):
    g = np.asarray(grid, dtype=int)
    frames = _find_frames(g)
    assert len(frames) >= 2, "expected multiple grey frames"
    palette = _palette_colors(g)
    assert len(palette) >= 4, "palette colors not found on grid edge"

    frame_infos = []
    templates: dict[str, list[np.ndarray]] = {pos: [] for pos in POSITIONS}
    hidden = None

    for idx, frame in enumerate(frames):
        pos_info: dict[str, dict[str, object]] = {}
        used_colors: set[int] = set()
        for pos, (rs, cs) in _quadrant_slices(frame).items():
            sub = g[rs, cs]
            if np.any(sub == MAROON):
                assert hidden is None, "expected exactly one hidden quadrant"
                hidden = (idx, pos)
                pos_info[pos] = {"hidden": True, "mask": None, "color": None, "slices": (rs, cs)}
                continue
            colors = {int(v) for v in np.unique(sub) if v != BLACK}
            assert len(colors) == 1, "visible quadrant must contain one fragment color"
            color = next(iter(colors))
            mask = sub == color
            assert mask.any(), "visible fragment cannot be empty"
            pos_info[pos] = {"hidden": False, "mask": mask, "color": color, "slices": (rs, cs)}
            used_colors.add(color)
            templates[pos].append(mask)
        frame_infos.append({"frame": frame, "positions": pos_info, "used_colors": used_colors})

    assert hidden is not None, "no hidden quadrant found"

    for pos, masks in templates.items():
        assert masks, f"missing visible template for {pos}"
        base = masks[0]
        for other in masks[1:]:
            assert other.shape == base.shape and np.array_equal(other, base), "shape template mismatch across frames"

    return g, frames, palette, frame_infos, hidden, {pos: masks[0] for pos, masks in templates.items()}


def main(grid: np.ndarray) -> np.ndarray:
    """
    Infer the hidden fragment color using frame-local uniqueness and cross-frame
    position uniqueness, then restore the occluded quadrant with the shared shape.
    """
    g, _frames, palette, frame_infos, hidden, templates = _analyze(grid)
    hidden_frame_idx, hidden_pos = hidden
    hidden_frame = frame_infos[hidden_frame_idx]

    frame_missing = palette - hidden_frame["used_colors"]
    assert len(frame_missing) == 1, "frame-local color constraint must identify one missing color"

    pos_used = {
        info["positions"][hidden_pos]["color"]
        for info in frame_infos
        if not info["positions"][hidden_pos]["hidden"]
    }
    pos_missing = palette - pos_used
    candidates = frame_missing & pos_missing
    assert len(candidates) == 1, "combined constraints must yield one color"
    fill_color = next(iter(candidates))

    out = g.copy()
    rs, cs = hidden_frame["positions"][hidden_pos]["slices"]
    out[rs, cs] = BLACK
    out[rs, cs][templates[hidden_pos]] = fill_color
    return out


def _grow_connected_mask(h: int, w: int, seed: tuple[int, int], rng: random.Random) -> np.ndarray:
    area = h * w
    target = rng.randint(max(3, area // 3), max(3, area - 1))
    mask = np.zeros((h, w), dtype=bool)
    mask[seed] = True
    frontier = {seed}
    while int(mask.sum()) < target and frontier:
        cell = rng.choice(tuple(frontier))
        cr, cc = cell
        nbrs = []
        for nr, nc in ((cr - 1, cc), (cr + 1, cc), (cr, cc - 1), (cr, cc + 1)):
            if 0 <= nr < h and 0 <= nc < w and not mask[nr, nc]:
                nbrs.append((nr, nc))
        if not nbrs:
            frontier.remove(cell)
            continue
        nr, nc = rng.choice(nbrs)
        mask[nr, nc] = True
        frontier.add((nr, nc))
    return mask


def _make_templates(qh: int, qw: int, rng: random.Random) -> dict[str, np.ndarray]:
    seeds = {
        "tl": (qh - 1, qw - 1),
        "tr": (qh - 1, 0),
        "bl": (0, qw - 1),
        "br": (0, 0),
    }
    templates = {pos: _grow_connected_mask(qh, qw, seed, rng) for pos, seed in seeds.items()}
    for pos, mask in templates.items():
        assert mask.any() and not np.all(mask), f"degenerate template at {pos}"
    return templates


def _latin_assignment(colors: list[int], rng: random.Random) -> list[list[int]]:
    symbols = colors[:]
    rng.shuffle(symbols)
    shifts = [0, 1, 2, 3]
    rng.shuffle(shifts)
    perm = [0, 1, 2, 3]
    rng.shuffle(perm)
    rows = []
    for shift in shifts:
        rows.append([symbols[(perm[col] + shift) % 4] for col in range(4)])
    return rows


def _place_frames(
    h: int, w: int, frame_size: int, rng: random.Random, palette_edge: str
) -> list[tuple[int, int]]:
    placed: list[tuple[int, int]] = []
    tries = 0
    while len(placed) < 4 and tries < 2000:
        tries += 1
        r0 = rng.randint(1, h - frame_size - 1)
        c0 = rng.randint(1, w - frame_size - 1)
        if palette_edge == "top" and r0 <= 1:
            continue
        if palette_edge == "bottom" and r0 + frame_size >= h - 1:
            continue
        if palette_edge == "left" and c0 <= 1:
            continue
        if palette_edge == "right" and c0 + frame_size >= w - 1:
            continue
        ok = True
        for pr, pc in placed:
            if not (
                r0 + frame_size + 1 < pr
                or pr + frame_size + 1 < r0
                or c0 + frame_size + 1 < pc
                or pc + frame_size + 1 < c0
            ):
                ok = False
                break
        if ok:
            placed.append((r0, c0))
    if len(placed) != 4:
        raise RuntimeError("failed to place four disjoint frames")
    return placed


def _draw_frame(grid: np.ndarray, r0: int, c0: int, size: int) -> None:
    r1, c1 = r0 + size - 1, c0 + size - 1
    grid[r0 : r1 + 1, c0] = GREY
    grid[r0 : r1 + 1, c1] = GREY
    grid[r0, c0 : c1 + 1] = GREY
    grid[r1, c0 : c1 + 1] = GREY


def _draw_palette(grid: np.ndarray, colors: list[int], edge: str, rng: random.Random) -> None:
    colors = colors[:]
    rng.shuffle(colors)
    if edge in {"top", "bottom"}:
        row = 0 if edge == "top" else grid.shape[0] - 1
        start = rng.randint(1, grid.shape[1] - len(colors) - 1)
        for i, color in enumerate(colors):
            grid[row, start + i] = color
    else:
        col = 0 if edge == "left" else grid.shape[1] - 1
        start = rng.randint(1, grid.shape[0] - len(colors) - 1)
        for i, color in enumerate(colors):
            grid[start + i, col] = color


def _expected_output(
    inp: np.ndarray,
    templates: dict[str, np.ndarray],
    hidden_frame_idx: int,
    hidden_pos: str,
    hidden_color: int,
    frames: list[tuple[int, int]],
) -> np.ndarray:
    out = inp.copy()
    r0, c0 = frames[hidden_frame_idx]
    size = (templates["tl"].shape[0] * 2) + 2
    rs, cs = _quadrant_slices((r0, c0, r0 + size - 1, c0 + size - 1))[hidden_pos]
    out[rs, cs] = BLACK
    out[rs, cs][templates[hidden_pos]] = hidden_color
    return out


def _gen_attempt(rng: random.Random) -> np.ndarray | None:
    frame_size = rng.choice([8, 10, 12])
    qh = qw = (frame_size - 2) // 2
    templates = _make_templates(qh, qw, rng)
    colors = rng.sample(PALETTE_POOL, 4)
    assignments = _latin_assignment(colors, rng)
    hidden_frame_idx = rng.randrange(4)
    hidden_pos = rng.choice(list(POSITIONS))
    hidden_pos_idx = POSITIONS.index(hidden_pos)
    hidden_color = assignments[hidden_frame_idx][hidden_pos_idx]

    palette_edge = rng.choice(["top", "bottom", "left", "right"])
    h = rng.randint(max(22, frame_size + 8), 32)
    w = rng.randint(max(22, frame_size + 8), 32)

    try:
        frame_origins = _place_frames(h, w, frame_size, rng, palette_edge)
    except RuntimeError:
        return None

    grid = np.full((h, w), BLACK, dtype=int)
    for idx, (r0, c0) in enumerate(frame_origins):
        _draw_frame(grid, r0, c0, frame_size)
        slices = _quadrant_slices((r0, c0, r0 + frame_size - 1, c0 + frame_size - 1))
        for pos_idx, pos in enumerate(POSITIONS):
            rs, cs = slices[pos]
            if idx == hidden_frame_idx and pos == hidden_pos:
                grid[rs, cs] = MAROON
            else:
                grid[rs, cs][templates[pos]] = assignments[idx][pos_idx]

    _draw_palette(grid, colors, palette_edge, rng)
    expected = _expected_output(grid, templates, hidden_frame_idx, hidden_pos, hidden_color, frame_origins)

    try:
        actual = main(grid)
    except Exception:
        return None
    if not np.array_equal(actual, expected):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = random.Random()
    for _ in range(MAX_GEN_ATTEMPTS):
        candidate = _gen_attempt(rng)
        if candidate is not None:
            return candidate
    raise RuntimeError("failed to generate a valid input")
