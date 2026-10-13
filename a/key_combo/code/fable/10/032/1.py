from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
import numpy as np


DIRS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
OPPOSITE = {(-1, 0): (1, 0), (1, 0): (-1, 0), (0, -1): (0, 1), (0, 1): (0, -1)}
MARKER_TEMPLATES = {
    (-1, 0): np.array([[1, 0, 1], [1, 0, 1], [1, 1, 1]], dtype=bool),
    (1, 0): np.array([[1, 1, 1], [1, 0, 1], [1, 0, 1]], dtype=bool),
    (0, -1): np.array([[1, 1, 1], [0, 0, 1], [1, 1, 1]], dtype=bool),
    (0, 1): np.array([[1, 1, 1], [1, 0, 0], [1, 1, 1]], dtype=bool),
}

GRID_HEIGHT_RANGE = (22, 31)
GRID_WIDTH_RANGE = (22, 31)
FRAME_SIZE_OPTIONS = (5, 7, 9)
CHAIN_LENGTH_OPTIONS = (2, 3, 4)
NOISE_COLOR_COUNTS = (2, 3)
MAX_GENERATION_ATTEMPTS = 400


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: tuple[int, int, int, int]
    direction: tuple[int, int]
    gap: tuple[int, int]
    border_cells: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class Marker:
    color: int
    bbox: tuple[int, int, int, int]
    direction: tuple[int, int]
    gap: tuple[int, int]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def _bbox_from_coords(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _find_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            q = deque([(r, c)])
            seen[r, c] = True
            coords: list[tuple[int, int]] = []
            while q:
                cr, cc = q.popleft()
                coords.append((cr, cc))
                for dr, dc in DIRS:
                    nr, nc = cr + dr, cc + dc
                    if 0 <= nr < h and 0 <= nc < w and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        q.append((nr, nc))
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "bbox": _bbox_from_coords(coords),
                    "area": len(coords),
                }
            )
    return comps


def _marker_candidate(comp: dict, color_counts: Counter[int], grid: np.ndarray) -> Marker | None:
    if color_counts[int(comp["color"])] != int(comp["area"]):
        return None
    r0, c0, r1, c1 = comp["bbox"]
    if (r1 - r0 + 1, c1 - c0 + 1) != (3, 3) or int(comp["area"]) != 7:
        return None
    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == int(comp["color"])
    for direction, template in MARKER_TEMPLATES.items():
        if np.array_equal(patch, template):
            if direction == (-1, 0):
                gap = (r0, c0 + 1)
            elif direction == (1, 0):
                gap = (r1, c0 + 1)
            elif direction == (0, -1):
                gap = (r0 + 1, c0)
            else:
                gap = (r0 + 1, c1)
            return Marker(int(comp["color"]), comp["bbox"], direction, gap)
    return None


def _frame_candidate(comp: dict, color_counts: Counter[int], grid: np.ndarray) -> Frame | None:
    if color_counts[int(comp["color"])] != int(comp["area"]):
        return None
    color = int(comp["color"])
    r0, c0, r1, c1 = comp["bbox"]
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 5 or w < 5:
        return None

    patch = grid[r0 : r1 + 1, c0 : c1 + 1] == color
    border = np.zeros((h, w), dtype=bool)
    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    candidates: list[tuple[int, int, tuple[int, int]]] = []
    if w % 2 == 1:
        mid_c = w // 2
        candidates.append((0, mid_c, (-1, 0)))
        candidates.append((h - 1, mid_c, (1, 0)))
    if h % 2 == 1:
        mid_r = h // 2
        candidates.append((mid_r, 0, (0, -1)))
        candidates.append((mid_r, w - 1, (0, 1)))

    for gr, gc, direction in candidates:
        expected = border.copy()
        expected[gr, gc] = False
        if np.array_equal(patch, expected):
            border_cells = tuple((r0 + rr, c0 + cc) for rr, cc in np.argwhere(expected))
            return Frame(color, comp["bbox"], direction, (r0 + gr, c0 + gc), border_cells)
    return None


def _extract_structures(grid: np.ndarray) -> tuple[Marker, list[Frame]]:
    g = np.asarray(grid, dtype=int)
    bg = get_bg(g)
    color_counts = Counter(map(int, g.ravel()))
    comps = _find_components(g, bg)

    marker_candidates: list[Marker] = []
    frame_candidates: list[Frame] = []
    for comp in comps:
        marker = _marker_candidate(comp, color_counts, g)
        if marker is not None:
            marker_candidates.append(marker)
        frame = _frame_candidate(comp, color_counts, g)
        if frame is not None:
            frame_candidates.append(frame)

    assert len(marker_candidates) == 1, "expected exactly one start marker"
    assert len(frame_candidates) >= 2, "expected at least two gapped frames"
    return marker_candidates[0], frame_candidates


def _ray_hit(
    start: tuple[int, int],
    direction: tuple[int, int],
    shape: tuple[int, int],
    border_to_frame: dict[tuple[int, int], int],
) -> int | None:
    h, w = shape
    dr, dc = direction
    r, c = start
    while True:
        r += dr
        c += dc
        if not (0 <= r < h and 0 <= c < w):
            return None
        frame_idx = border_to_frame.get((r, c))
        if frame_idx is not None:
            return frame_idx


def _trace_chain(marker: Marker, frames: list[Frame], shape: tuple[int, int]) -> list[int]:
    border_to_frame: dict[tuple[int, int], int] = {}
    for idx, frame in enumerate(frames):
        for cell in frame.border_cells:
            border_to_frame[cell] = idx

    first = _ray_hit(marker.gap, marker.direction, shape, border_to_frame)
    assert first is not None, "marker must point to a frame"

    order: list[int] = []
    seen: set[int] = set()
    current = first
    while True:
        assert current not in seen, "frame pointer chain must be acyclic"
        seen.add(current)
        order.append(current)
        nxt = _ray_hit(frames[current].gap, frames[current].direction, shape, border_to_frame)
        if nxt is None:
            return order
        current = nxt


def main(grid: np.ndarray) -> np.ndarray:
    """Crop the interior of the terminal frame reached by following the gap-pointer chain.

    Input structure:
    - A noisy canvas contains several isolated monochrome one-cell-thick rectangular
      frames, each in a distinct color.
    - Every frame has exactly one centered gap on one side; the outward orthogonal
      ray from that gap is that frame's pointer.
    - One separate 3x3 start marker has a single opening direction whose sightline
      reaches the first frame in the chain.

    Rule:
    - Find the frame first indicated by the start marker.
    - Repeatedly follow the current frame's gap pointer to the next frame hit on that
      straight ray.
    - Stop when a frame's pointer leaves the grid without hitting any later frame,
      then crop out that terminal frame's strict interior, unchanged.

    Key insight:
    - The missing border cell is a directional pointer, not damage: the correct frame
      is selected only by traversing the chain, because off-chain frames may also point
      out of the grid.
    """
    g = np.asarray(grid, dtype=int)
    marker, frames = _extract_structures(g)
    chain = _trace_chain(marker, frames, g.shape)
    terminal = frames[chain[-1]]
    r0, c0, r1, c1 = terminal.bbox
    return g[r0 + 1 : r1, c0 + 1 : c1].copy()


def _rand_palette(rng: np.random.Generator, total_frames: int) -> tuple[list[int], list[int], int]:
    noise_count = int(rng.choice(NOISE_COLOR_COUNTS))
    count = total_frames + noise_count + 1
    colors = [int(x) for x in rng.choice(np.arange(1, 10), size=count, replace=False)]
    frame_colors = colors[:total_frames]
    noise_colors = colors[total_frames : total_frames + noise_count]
    marker_color = colors[-1]
    return frame_colors, noise_colors, marker_color


def _make_noise_grid(rng: np.random.Generator, h: int, w: int, noise_colors: list[int]) -> np.ndarray:
    palette = np.array([0] + noise_colors, dtype=int)
    weights = rng.random(len(palette))
    weights[0] += 2.5
    weights /= weights.sum()
    return rng.choice(palette, size=(h, w), p=weights)


def _frame_bbox(frame: dict) -> tuple[int, int, int, int]:
    return frame["r0"], frame["c0"], frame["r0"] + frame["h"] - 1, frame["c0"] + frame["w"] - 1


def _frame_gap(frame: dict) -> tuple[int, int]:
    r0, c0, r1, c1 = _frame_bbox(frame)
    if frame["dir"] == (-1, 0):
        return r0, (c0 + c1) // 2
    if frame["dir"] == (1, 0):
        return r1, (c0 + c1) // 2
    if frame["dir"] == (0, -1):
        return (r0 + r1) // 2, c0
    return (r0 + r1) // 2, c1


def _expanded_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int], margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def _can_place_frame(frame: dict, placed: list[dict], h: int, w: int) -> bool:
    r0, c0, r1, c1 = _frame_bbox(frame)
    if r0 < 0 or c0 < 0 or r1 >= h or c1 >= w:
        return False
    bbox = (r0, c0, r1, c1)
    return all(not _expanded_overlap(bbox, _frame_bbox(other)) for other in placed)


def _sample_frame_dims(rng: np.random.Generator) -> tuple[int, int]:
    return int(rng.choice(FRAME_SIZE_OPTIONS)), int(rng.choice(FRAME_SIZE_OPTIONS))


def _place_chain(rng: np.random.Generator, h: int, w: int, chain_len: int) -> tuple[list[dict], dict] | None:
    for _ in range(120):
        first_h, first_w = _sample_frame_dims(rng)
        first = {
            "r0": int(rng.integers(2, h - first_h - 1)),
            "c0": int(rng.integers(2, w - first_w - 1)),
            "h": first_h,
            "w": first_w,
            "dir": None,
        }
        frames = [first]
        ok = True

        for i in range(chain_len - 1):
            current = frames[i]
            prev_dir = frames[i - 1]["dir"] if i > 0 else None
            dir_options = DIRS[:]
            rng.shuffle(dir_options)
            if prev_dir is not None:
                dir_options = [d for d in dir_options if d != OPPOSITE[prev_dir]]

            placed_next = False
            for direction in dir_options:
                next_h, next_w = _sample_frame_dims(rng)
                bbox = _frame_bbox(current)
                r0, c0, r1, c1 = bbox
                center_r = (r0 + r1) // 2
                center_c = (c0 + c1) // 2
                gap = int(rng.integers(2, 7))

                if direction == (0, 1):
                    candidate = {"r0": center_r - next_h // 2, "c0": c1 + gap, "h": next_h, "w": next_w, "dir": None}
                elif direction == (0, -1):
                    candidate = {
                        "r0": center_r - next_h // 2,
                        "c0": c0 - gap - next_w + 1,
                        "h": next_h,
                        "w": next_w,
                        "dir": None,
                    }
                elif direction == (1, 0):
                    candidate = {"r0": r1 + gap, "c0": center_c - next_w // 2, "h": next_h, "w": next_w, "dir": None}
                else:
                    candidate = {
                        "r0": r0 - gap - next_h + 1,
                        "c0": center_c - next_w // 2,
                        "h": next_h,
                        "w": next_w,
                        "dir": None,
                    }

                if not _can_place_frame(candidate, frames, h, w):
                    continue

                current["dir"] = direction
                frames.append(candidate)
                placed_next = True
                break

            if not placed_next:
                ok = False
                break

        if not ok:
            continue

        terminal_prev = frames[-2]["dir"] if chain_len > 1 else None
        terminal_dirs = DIRS[:]
        rng.shuffle(terminal_dirs)
        if terminal_prev is not None:
            terminal_dirs = [d for d in terminal_dirs if d != OPPOSITE[terminal_prev]]
        frames[-1]["dir"] = terminal_dirs[0]

        first_dirs = DIRS[:]
        rng.shuffle(first_dirs)
        first_dirs = [d for d in first_dirs if d != OPPOSITE[frames[0]["dir"]]]
        if not first_dirs:
            continue
        marker_dir = first_dirs[0]
        fr0, fc0, fr1, fc1 = _frame_bbox(frames[0])
        center_r = (fr0 + fr1) // 2
        center_c = (fc0 + fc1) // 2
        marker_gap = int(rng.integers(2, 7))
        if marker_dir == (0, 1):
            marker = {"r0": center_r - 1, "c0": fc0 - marker_gap - 2, "dir": marker_dir}
        elif marker_dir == (0, -1):
            marker = {"r0": center_r - 1, "c0": fc1 + marker_gap, "dir": marker_dir}
        elif marker_dir == (1, 0):
            marker = {"r0": fr0 - marker_gap - 2, "c0": center_c - 1, "dir": marker_dir}
        else:
            marker = {"r0": fr1 + marker_gap, "c0": center_c - 1, "dir": marker_dir}

        mr0, mc0 = marker["r0"], marker["c0"]
        marker_bbox = (mr0, mc0, mr0 + 2, mc0 + 2)
        if mr0 < 0 or mc0 < 0 or mr0 + 2 >= h or mc0 + 2 >= w:
            continue
        if any(_expanded_overlap(marker_bbox, _frame_bbox(frame)) for frame in frames):
            continue
        return frames, marker

    return None


def _place_decoys(
    rng: np.random.Generator,
    placed: list[dict],
    h: int,
    w: int,
    decoy_count: int,
) -> list[dict] | None:
    frames = list(placed)
    decoys: list[dict] = []
    for _ in range(decoy_count):
        placed_one = False
        for _ in range(120):
            fh, fw = _sample_frame_dims(rng)
            direction = DIRS[int(rng.integers(0, len(DIRS)))]
            if direction == (-1, 0):
                candidate = {
                    "r0": int(rng.integers(1, max(2, min(4, h - fh + 1)))),
                    "c0": int(rng.integers(1, w - fw - 1)),
                    "h": fh,
                    "w": fw,
                    "dir": direction,
                }
            elif direction == (1, 0):
                r0_min = max(1, h - fh - 3)
                candidate = {
                    "r0": int(rng.integers(r0_min, h - fh)),
                    "c0": int(rng.integers(1, w - fw - 1)),
                    "h": fh,
                    "w": fw,
                    "dir": direction,
                }
            elif direction == (0, -1):
                candidate = {
                    "r0": int(rng.integers(1, h - fh - 1)),
                    "c0": int(rng.integers(1, max(2, min(4, w - fw + 1)))),
                    "h": fh,
                    "w": fw,
                    "dir": direction,
                }
            else:
                c0_min = max(1, w - fw - 3)
                candidate = {
                    "r0": int(rng.integers(1, h - fh - 1)),
                    "c0": int(rng.integers(c0_min, w - fw)),
                    "h": fh,
                    "w": fw,
                    "dir": direction,
                }
            if _can_place_frame(candidate, frames, h, w):
                frames.append(candidate)
                decoys.append(candidate)
                placed_one = True
                break
        if not placed_one:
            return None
    return decoys


def _fill_unique_interiors(
    rng: np.random.Generator,
    grid: np.ndarray,
    frames: list[dict],
    noise_colors: list[int],
) -> bool:
    seen: set[bytes] = set()
    palette = np.array([0] + noise_colors, dtype=int)
    weights = rng.random(len(palette))
    weights[0] += 1.8
    weights /= weights.sum()
    for frame in frames:
        r0, c0, r1, c1 = _frame_bbox(frame)
        patch = rng.choice(palette, size=(frame["h"] - 2, frame["w"] - 2), p=weights)
        sig = patch.tobytes() + bytes([patch.shape[0], patch.shape[1]])
        if sig in seen:
            return False
        seen.add(sig)
        grid[r0 + 1 : r1, c0 + 1 : c1] = patch
    return True


def _draw_frame(grid: np.ndarray, frame: dict, color: int) -> None:
    r0, c0, r1, c1 = _frame_bbox(frame)
    grid[r0, c0 : c1 + 1] = color
    grid[r1, c0 : c1 + 1] = color
    grid[r0 : r1 + 1, c0] = color
    grid[r0 : r1 + 1, c1] = color
    gr, gc = _frame_gap(frame)
    grid[gr, gc] = grid[gr, gc]


def _draw_marker(grid: np.ndarray, marker: dict, color: int) -> None:
    template = MARKER_TEMPLATES[marker["dir"]]
    r0, c0 = marker["r0"], marker["c0"]
    patch = grid[r0 : r0 + 3, c0 : c0 + 3]
    patch[template] = color


def _semantic_check(grid: np.ndarray) -> bool:
    try:
        marker, frames = _extract_structures(grid)
        chain = _trace_chain(marker, frames, grid.shape)
        if len(chain) < 2:
            return False
        terminal = frames[chain[-1]]
        terminal_area = (terminal.bbox[2] - terminal.bbox[0] - 1) * (terminal.bbox[3] - terminal.bbox[1] - 1)

        border_to_frame: dict[tuple[int, int], int] = {}
        for idx, frame in enumerate(frames):
            for cell in frame.border_cells:
                border_to_frame[cell] = idx

        off_chain = [idx for idx in range(len(frames)) if idx not in set(chain)]
        if not off_chain:
            return False
        if not any(_ray_hit(frames[idx].gap, frames[idx].direction, grid.shape, border_to_frame) is None for idx in off_chain):
            return False

        other_areas = [
            (frames[idx].bbox[2] - frames[idx].bbox[0] - 1) * (frames[idx].bbox[3] - frames[idx].bbox[1] - 1)
            for idx in range(len(frames))
            if idx != chain[-1]
        ]
        if max(other_areas) < terminal_area:
            return False

        terminal_out = main(grid)
        matches = 0
        for frame in frames:
            r0, c0, r1, c1 = frame.bbox
            if np.array_equal(grid[r0 + 1 : r1, c0 + 1 : c1], terminal_out):
                matches += 1
        if matches != 1:
            return False
        return True
    except Exception:
        return False


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_HEIGHT_RANGE))
    w = int(rng.integers(*GRID_WIDTH_RANGE))
    chain_len = int(rng.choice(CHAIN_LENGTH_OPTIONS))
    total_frames = chain_len + int(rng.integers(1, min(3, 6 - chain_len)))
    decoy_count = total_frames - chain_len

    placement = _place_chain(rng, h, w, chain_len)
    if placement is None:
        return None
    chain_frames, marker = placement
    decoys = _place_decoys(rng, chain_frames, h, w, decoy_count)
    if decoys is None:
        return None
    all_frames = chain_frames + decoys

    frame_colors, noise_colors, marker_color = _rand_palette(rng, total_frames)
    grid = _make_noise_grid(rng, h, w, noise_colors)

    if not _fill_unique_interiors(rng, grid, all_frames, noise_colors):
        return None

    for frame, color in zip(all_frames, frame_colors):
        _draw_frame(grid, frame, color)
        gr, gc = _frame_gap(frame)
        grid[gr, gc] = 0 if rng.random() < 0.6 else int(rng.choice(noise_colors))

    _draw_marker(grid, marker, marker_color)
    mr0, mc0 = marker["r0"], marker["c0"]
    gap_r = mr0 + (0 if marker["dir"] == (-1, 0) else 2 if marker["dir"] == (1, 0) else 1)
    gap_c = mc0 + (0 if marker["dir"] == (0, -1) else 2 if marker["dir"] == (0, 1) else 1)
    grid[gap_r, gap_c] = 0 if rng.random() < 0.6 else int(rng.choice(noise_colors))

    if not _semantic_check(grid):
        return None
    return grid


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_GENERATION_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError("failed to generate a valid gap-pointer-chain sample")
