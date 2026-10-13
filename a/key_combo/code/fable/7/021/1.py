from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass

import numpy as np


Coord = tuple[int, int]
Dir = tuple[int, int]
BBox = tuple[int, int, int, int]

BLACK = 0
DIR_UP: Dir = (-1, 0)
DIR_DOWN: Dir = (1, 0)
DIR_LEFT: Dir = (0, -1)
DIR_RIGHT: Dir = (0, 1)
DIR_TO_SIDE = {
    DIR_UP: "top",
    DIR_DOWN: "bottom",
    DIR_LEFT: "left",
    DIR_RIGHT: "right",
}


@dataclass(frozen=True)
class Marker:
    color: int
    coord: Coord
    direction: Dir


@dataclass(frozen=True)
class Frame:
    color: int
    bbox: BBox
    gap: Coord
    exit_dir: Dir
    wall_cells: frozenset[Coord]


def get_bg(grid: np.ndarray) -> int:
    return int(Counter(map(int, grid.ravel())).most_common(1)[0][0])


def in_bounds(shape: tuple[int, int], r: int, c: int) -> bool:
    h, w = map(int, shape)
    return 0 <= int(r) < h and 0 <= int(c) < w


def neighbors4(r: int, c: int) -> tuple[Coord, Coord, Coord, Coord]:
    return (r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)


def monochrome_components(grid: np.ndarray, bg: int) -> list[dict]:
    h, w = grid.shape
    seen = np.zeros((h, w), dtype=bool)
    comps: list[dict] = []
    for r in range(h):
        for c in range(w):
            color = int(grid[r, c])
            if color == bg or seen[r, c]:
                continue
            stack = [(r, c)]
            seen[r, c] = True
            cells = []
            while stack:
                rr, cc = stack.pop()
                cells.append((rr, cc))
                for nr, nc in neighbors4(rr, cc):
                    if in_bounds((h, w), nr, nc) and not seen[nr, nc] and int(grid[nr, nc]) == color:
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            coords = np.array(cells, dtype=int)
            r0, c0 = coords.min(axis=0)
            r1, c1 = coords.max(axis=0)
            comps.append(
                {
                    "color": color,
                    "coords": coords,
                    "area": int(coords.shape[0]),
                    "bbox": (int(r0), int(c0), int(r1), int(c1)),
                }
            )
    return comps


def bbox_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    return {(r, c) for r in range(r0, r1 + 1) for c in range(c0, c1 + 1)}


def ring_cells(bbox: BBox) -> set[Coord]:
    r0, c0, r1, c1 = bbox
    cells = set()
    for c in range(c0, c1 + 1):
        cells.add((r0, c))
        cells.add((r1, c))
    for r in range(r0, r1 + 1):
        cells.add((r, c0))
        cells.add((r, c1))
    return cells


def gap_exit_dir(bbox: BBox, gap: Coord) -> Dir | None:
    r0, c0, r1, c1 = bbox
    r, c = gap
    if r == r0 and c0 < c < c1:
        return DIR_UP
    if r == r1 and c0 < c < c1:
        return DIR_DOWN
    if c == c0 and r0 < r < r1:
        return DIR_LEFT
    if c == c1 and r0 < r < r1:
        return DIR_RIGHT
    return None


def inward_dir(coord: Coord, shape: tuple[int, int]) -> Dir | None:
    r, c = coord
    h, w = shape
    if r == 0 and 0 < c < w - 1:
        return DIR_DOWN
    if r == h - 1 and 0 < c < w - 1:
        return DIR_UP
    if c == 0 and 0 < r < h - 1:
        return DIR_RIGHT
    if c == w - 1 and 0 < r < h - 1:
        return DIR_LEFT
    return None


def parse_scene(grid: np.ndarray) -> tuple[int, list[Marker], list[Frame]]:
    g = np.asarray(grid, dtype=int)
    if g.ndim != 2:
        raise ValueError("expected a 2D grid")
    bg = get_bg(g)
    markers: list[Marker] = []
    frames: list[Frame] = []
    for comp in monochrome_components(g, bg):
        coords = comp["coords"]
        if comp["area"] == 1:
            coord = tuple(map(int, coords[0]))
            direction = inward_dir(coord, g.shape)
            if direction is None:
                raise ValueError("singleton foreground cells must be non-corner boundary markers")
            markers.append(Marker(color=int(comp["color"]), coord=coord, direction=direction))
            continue

        frame = analyze_frame_component(comp)
        if frame is None:
            raise ValueError("foreground component is neither a boundary marker nor a valid gap frame")
        frames.append(frame)

    if not markers or not frames:
        raise ValueError("expected at least one marker and one frame")
    markers.sort(key=lambda marker: (marker.coord[0], marker.coord[1], marker.color))
    frames.sort(key=lambda frame: (frame.bbox[0], frame.bbox[1], frame.color))
    return bg, markers, frames


def analyze_frame_component(comp: dict) -> Frame | None:
    coords = {tuple(map(int, coord)) for coord in comp["coords"]}
    r0, c0, r1, c1 = map(int, comp["bbox"])
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return None

    ring = ring_cells((r0, c0, r1, c1))
    if not coords <= ring:
        return None
    missing = ring - coords
    if len(missing) != 1 or len(coords) != len(ring) - 1:
        return None

    gap = next(iter(missing))
    exit_dir = gap_exit_dir((r0, c0, r1, c1), gap)
    if exit_dir is None:
        return None

    return Frame(
        color=int(comp["color"]),
        bbox=(r0, c0, r1, c1),
        gap=gap,
        exit_dir=exit_dir,
        wall_cells=frozenset(coords),
    )


def first_hit(current: Coord, direction: Dir, frames: list[Frame]) -> tuple[int, Frame] | None:
    r, c = current
    best: tuple[int, Frame] | None = None
    for frame in frames:
        distances = []
        for wr, wc in frame.wall_cells:
            if direction == DIR_RIGHT and wr == r and wc > c:
                distances.append(wc - c)
            elif direction == DIR_LEFT and wr == r and wc < c:
                distances.append(c - wc)
            elif direction == DIR_DOWN and wc == c and wr > r:
                distances.append(wr - r)
            elif direction == DIR_UP and wc == c and wr < r:
                distances.append(r - wr)
        if distances:
            dist = min(distances)
            if best is None or dist < best[0]:
                best = (dist, frame)
    return best


def cells_until(current: Coord, direction: Dir, steps: int) -> list[Coord]:
    r, c = current
    dr, dc = direction
    return [(r + dr * t, c + dc * t) for t in range(steps)]


def cells_to_boundary(current: Coord, direction: Dir, shape: tuple[int, int]) -> list[Coord]:
    r, c = current
    dr, dc = direction
    out = []
    while in_bounds(shape, r, c):
        out.append((r, c))
        r += dr
        c += dc
    return out


def trace_ray(marker: Marker, frames: list[Frame], shape: tuple[int, int]) -> list[Coord]:
    current = marker.coord
    direction = marker.direction
    path: list[Coord] = []
    visited = set()
    while True:
        state = (current, direction)
        if state in visited:
            raise ValueError("portal cycle detected")
        visited.add(state)

        hit = first_hit(current, direction, frames)
        if hit is None:
            path.extend(cells_to_boundary(current, direction, shape))
            return path

        dist, frame = hit
        path.extend(cells_until(current, direction, dist))
        path.append(frame.gap)
        current = frame.gap
        direction = frame.exit_dir


def render_output(grid: np.ndarray, bg: int, markers: list[Marker], frames: list[Frame]) -> np.ndarray:
    out = np.asarray(grid, dtype=int).copy()
    wall_cells = set()
    for frame in frames:
        wall_cells.update(frame.wall_cells)
    for marker in markers:
        for r, c in trace_ray(marker, frames, out.shape):
            if (r, c) not in wall_cells:
                out[r, c] = marker.color
    return out


def main(input_grid: np.ndarray) -> np.ndarray:
    """Trace each boundary marker's ray through gap-port frames until it leaves the grid.

    Input structure:
    - The background is black.
    - Several unique-colored single-pixel markers lie on the outer boundary, each at a non-corner boundary cell.
    - The interior contains one-pixel-thick hollow rectangles; each rectangle is monochromatic and has exactly one missing border cell as its gap.

    Rule:
    - Each boundary marker emits a ray inward, perpendicular to the boundary edge it sits on, and the ray paints every visited background cell with the marker color.
    - When a ray would next touch any frame wall cell, it stops before that wall and immediately resumes from that frame's unique gap cell, painting the gap cell too; the resumed direction is the outward normal of the gap side.
    - The same ray may be redirected by multiple frames in sequence until a final segment reaches the grid boundary; frame wall cells themselves never change color, and different rays are interpreted independently.

    Key insight:
    - A hollow rectangle is not an obstacle to bounce off: it is a portal whose single missing border cell uniquely decides where and in which direction the hit ray continues, regardless of which wall cell was hit.
    """
    bg, markers, frames = parse_scene(np.asarray(input_grid, dtype=int))
    return render_output(np.asarray(input_grid, dtype=int), bg, markers, frames)


def opposite(direction: Dir) -> Dir:
    return (-direction[0], -direction[1])


def rotate_dir(direction: Dir, k: int) -> Dir:
    dr, dc = direction
    k %= 4
    for _ in range(k):
        dr, dc = -dc, dr
    return dr, dc


def dir_name(direction: Dir) -> str:
    return DIR_TO_SIDE[direction]


def paint_frame(grid: np.ndarray, frame: Frame) -> None:
    for r, c in frame.wall_cells:
        grid[r, c] = frame.color


def segment_is_free(cells: list[Coord], occupied: set[Coord]) -> bool:
    return all(cell not in occupied for cell in cells)


def generate_frame_candidates(
    current: Coord,
    direction_in: Dir,
    direction_out: Dir,
    band_top: int,
    band_bottom: int,
    shape: tuple[int, int],
    occupied: set[Coord],
) -> list[dict]:
    h_lim, w_lim = shape
    r, c = current
    candidates = []
    height_choices = [3, 4, 5]
    width_choices = [3, 4, 5, 6]

    for fh in height_choices:
        for fw in width_choices:
            if direction_in == DIR_RIGHT:
                for r0 in range(max(1, band_top), min(r - 1, band_bottom - fh + 1) + 1):
                    r1 = r0 + fh - 1
                    if not (r0 < r < r1):
                        continue
                    for c0 in range(c + 2, w_lim - fw):
                        c1 = c0 + fw - 1
                        hit = (r, c0)
                        candidates.extend(
                            frame_variants(
                                bbox=(r0, c0, r1, c1),
                                hit=hit,
                                current=current,
                                direction_in=direction_in,
                                direction_out=direction_out,
                                occupied=occupied,
                            )
                        )
            elif direction_in == DIR_LEFT:
                for r0 in range(max(1, band_top), min(r - 1, band_bottom - fh + 1) + 1):
                    r1 = r0 + fh - 1
                    if not (r0 < r < r1):
                        continue
                    for c0 in range(1, c - fw):
                        c1 = c0 + fw - 1
                        if c1 > c - 2:
                            continue
                        hit = (r, c1)
                        candidates.extend(
                            frame_variants(
                                bbox=(r0, c0, r1, c1),
                                hit=hit,
                                current=current,
                                direction_in=direction_in,
                                direction_out=direction_out,
                                occupied=occupied,
                            )
                        )
            elif direction_in == DIR_DOWN:
                for c0 in range(1, w_lim - fw):
                    c1 = c0 + fw - 1
                    if not (c0 < c < c1):
                        continue
                    for r0 in range(r + 2, band_bottom - fh + 2):
                        r1 = r0 + fh - 1
                        if r1 >= h_lim - 1:
                            continue
                        hit = (r0, c)
                        candidates.extend(
                            frame_variants(
                                bbox=(r0, c0, r1, c1),
                                hit=hit,
                                current=current,
                                direction_in=direction_in,
                                direction_out=direction_out,
                                occupied=occupied,
                            )
                        )
            else:
                for c0 in range(1, w_lim - fw):
                    c1 = c0 + fw - 1
                    if not (c0 < c < c1):
                        continue
                    for r0 in range(max(1, band_top), r - fh):
                        r1 = r0 + fh - 1
                        if r1 > r - 2:
                            continue
                        hit = (r1, c)
                        candidates.extend(
                            frame_variants(
                                bbox=(r0, c0, r1, c1),
                                hit=hit,
                                current=current,
                                direction_in=direction_in,
                                direction_out=direction_out,
                                occupied=occupied,
                            )
                        )
    random.shuffle(candidates)
    return candidates


def frame_variants(
    bbox: BBox,
    hit: Coord,
    current: Coord,
    direction_in: Dir,
    direction_out: Dir,
    occupied: set[Coord],
) -> list[dict]:
    r0, c0, r1, c1 = bbox
    frame_cells = bbox_cells(bbox)
    if any(cell in occupied for cell in frame_cells):
        return []

    if direction_in == DIR_RIGHT:
        steps = hit[1] - current[1]
    elif direction_in == DIR_LEFT:
        steps = current[1] - hit[1]
    elif direction_in == DIR_DOWN:
        steps = hit[0] - current[0]
    else:
        steps = current[0] - hit[0]
    segment = cells_until(current, direction_in, steps)
    if not segment_is_free(segment[1:], occupied):
        return []

    variants = []
    if direction_out == DIR_UP:
        gaps = [(r0, cc) for cc in range(c0 + 1, c1)]
    elif direction_out == DIR_DOWN:
        gaps = [(r1, cc) for cc in range(c0 + 1, c1)]
    elif direction_out == DIR_LEFT:
        gaps = [(rr, c0) for rr in range(r0 + 1, r1)]
    else:
        gaps = [(rr, c1) for rr in range(r0 + 1, r1)]

    for gap in gaps:
        if gap == hit:
            continue
        if gap in occupied:
            continue
        variants.append(
            {
                "bbox": bbox,
                "gap": gap,
                "segment": segment,
                "new_current": gap,
                "new_direction": direction_out,
                "occupied_add": frame_cells | set(segment[1:]),
            }
        )
    return variants


def free_path_to_boundary(current: Coord, direction: Dir, shape: tuple[int, int], occupied: set[Coord]) -> list[Coord] | None:
    cells = cells_to_boundary(current, direction, shape)
    if segment_is_free(cells[1:], occupied):
        return cells
    return None


def build_chain(
    band_top: int,
    band_bottom: int,
    width: int,
    template: list[Dir],
    start_side: str,
    mid_row: int,
) -> dict | None:
    start = (mid_row, 0) if start_side == "left" else (mid_row, width - 1)
    start_dir = DIR_RIGHT if start_side == "left" else DIR_LEFT
    occupied = {start}

    def rec(current: Coord, direction: Dir, idx: int, occ: set[Coord], frames: list[dict]) -> list[dict] | None:
        if idx == len(template):
            if direction not in (DIR_LEFT, DIR_RIGHT):
                return None
            if free_path_to_boundary(current, direction, (band_bottom + 1, width), occ) is None:
                return None
            return frames

        for cand in generate_frame_candidates(current, direction, template[idx], band_top, band_bottom, (band_bottom + 1, width), occ):
            nxt = rec(cand["new_current"], cand["new_direction"], idx + 1, occ | cand["occupied_add"], frames + [cand])
            if nxt is not None:
                return nxt
        return None

    placed = rec(start, start_dir, 0, occupied, [])
    if placed is None:
        return None
    return {"start": start, "start_dir": start_dir, "frames": placed}


def choose_templates(n_chains: int) -> list[list[Dir]]:
    templates: list[list[Dir]] = []
    vertical = random.choice([DIR_UP, DIR_DOWN])
    horizontal_finish = random.choice([DIR_LEFT, DIR_RIGHT])
    templates.append([vertical, horizontal_finish])

    one_portal_dir = random.choice([DIR_LEFT, DIR_RIGHT])
    templates.append([one_portal_dir])

    if n_chains == 3:
        options = [[], [random.choice([DIR_LEFT, DIR_RIGHT])], [random.choice([DIR_UP, DIR_DOWN]), random.choice([DIR_LEFT, DIR_RIGHT])]]
        templates.append(random.choice(options))

    random.shuffle(templates)
    return templates


def render_case(chains: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, dict]:
    h, w = shape
    total_frames = sum(len(chain["frames"]) for chain in chains)
    colors = list(range(1, 10))
    random.shuffle(colors)
    marker_colors = colors[: len(chains)]
    frame_colors = colors[len(chains) : len(chains) + total_frames]
    grid = np.full((h, w), BLACK, dtype=int)

    frame_idx = 0
    metadata = {"chain_templates": [], "markers": 0, "frames": total_frames}
    for chain, marker_color in zip(chains, marker_colors):
        sr, sc = chain["start"]
        grid[sr, sc] = marker_color
        metadata["markers"] += 1
        metadata["chain_templates"].append(len(chain["frames"]))
        for placed in chain["frames"]:
            color = frame_colors[frame_idx]
            frame_idx += 1
            bbox = placed["bbox"]
            gap = placed["gap"]
            ring = ring_cells(bbox)
            walls = frozenset(ring - {gap})
            frame = Frame(color=color, bbox=bbox, gap=gap, exit_dir=gap_exit_dir(bbox, gap), wall_cells=walls)
            paint_frame(grid, frame)
    return grid, metadata


def rotate_scene(grid: np.ndarray, k: int) -> np.ndarray:
    return np.rot90(grid, k=k)


def generate_case() -> tuple[np.ndarray, np.ndarray, dict]:
    n_chains = random.choice([2, 3])
    band_h = 9
    sep = 1
    height = n_chains * band_h + (n_chains - 1) * sep
    width = random.randint(20, 28)
    templates = choose_templates(n_chains)

    chains = []
    for idx in range(n_chains):
        band_top = idx * (band_h + sep)
        band_bottom = band_top + band_h - 1
        mid_row = random.randint(band_top + 2, band_bottom - 2)
        start_side = random.choice(["left", "right"])
        chain = build_chain(band_top, band_bottom, width, templates[idx], start_side, mid_row)
        if chain is None:
            raise RuntimeError("failed to build chain")
        chains.append(chain)

    canonical, metadata = render_case(chains, (height, width))
    k = random.randint(0, 3)
    inp = rotate_scene(canonical, k)
    out = rotate_scene(main(canonical), k)
    metadata["rotation"] = k
    metadata["shape"] = tuple(map(int, inp.shape))
    return inp, out, metadata


def validate_generated_case(inp: np.ndarray, out: np.ndarray, metadata: dict) -> bool:
    if inp.ndim != 2 or out.ndim != 2:
        return False
    if max(inp.shape) > 30 or max(out.shape) > 30:
        return False
    try:
        bg, markers, frames = parse_scene(inp)
    except Exception:
        return False
    if bg != BLACK:
        return False
    if len(markers) != metadata["markers"] or len(frames) != metadata["frames"]:
        return False
    if not any(length >= 2 for length in metadata["chain_templates"]):
        return False
    recomputed = render_output(inp, bg, markers, frames)
    if not np.array_equal(recomputed, out):
        return False
    if np.array_equal(inp, out):
        return False
    colored = int(np.count_nonzero(out != BLACK))
    if colored < len(markers) + 12:
        return False
    return True


def generate_input() -> np.ndarray:
    for _ in range(400):
        try:
            inp, out, metadata = generate_case()
        except RuntimeError:
            continue
        try:
            actual = main(inp)
        except Exception:
            continue
        if np.array_equal(actual, out) and validate_generated_case(inp, out, metadata):
            return inp
    raise RuntimeError("failed to generate a valid ray-gap ARC input")
