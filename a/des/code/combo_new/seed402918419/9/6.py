import numpy as np


BG = 0
GREY = 5
FRAME_COLORS = (1, 2, 3, 4, 6, 7, 8, 9)
DIRECTIONS = ("up", "right", "down", "left")
TURN_BY_DIRECTION = {"up": 0, "right": 1, "down": 2, "left": 3}
PATTERN_LIBRARY = (
    ((0, 0), (-1, 0)),
    ((0, 0), (-1, 0), (-2, 0)),
    ((0, 0), (-1, 0), (-1, -1)),
    ((0, 0), (-1, 0), (-1, 1)),
    ((0, 0), (-1, -1), (-1, 0), (-1, 1)),
    ((0, 0), (-1, 0), (-2, 0), (-2, -1)),
    ((0, 0), (-1, 0), (-2, 0), (-2, 1)),
    ((0, 0), (-1, 0), (-2, 0), (-1, -1)),
    ((0, 0), (-1, 0), (-2, 0), (-1, 1)),
)
FRAME_SIZES = (7, 9)
GRID_RANGE = (27, 31)
FRAME_SLOTS = ("tl", "tr", "bl", "br")
MAX_ATTEMPTS = 400


def _components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps: list[list[tuple[int, int]]] = []
    for r in range(h):
        for c in range(w):
            if not bool(mask[r, c]) or bool(seen[r, c]):
                continue
            stack = [(r, c)]
            seen[r, c] = True
            comp: list[tuple[int, int]] = []
            while stack:
                rr, cc = stack.pop()
                comp.append((int(rr), int(cc)))
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nr = rr + dr
                    nc = cc + dc
                    if 0 <= nr < h and 0 <= nc < w and bool(mask[nr, nc]) and not bool(seen[nr, nc]):
                        seen[nr, nc] = True
                        stack.append((nr, nc))
            comps.append(comp)
    return comps


def _bbox(coords: list[tuple[int, int]]) -> tuple[int, int, int, int]:
    rows = [r for r, _ in coords]
    cols = [c for _, c in coords]
    return min(rows), min(cols), max(rows), max(cols)


def _is_rectangular_frame(grid: np.ndarray, color: int, coords: list[tuple[int, int]]) -> bool:
    r0, c0, r1, c1 = _bbox(coords)
    h = r1 - r0 + 1
    w = c1 - c0 + 1
    if h < 3 or w < 3:
        return False
    if len(coords) != 2 * (h + w) - 4:
        return False
    for r in range(r0, r1 + 1):
        for c in range(c0, c1 + 1):
            on_border = r in (r0, r1) or c in (c0, c1)
            if on_border:
                if int(grid[r, c]) != int(color):
                    return False
            elif int(grid[r, c]) == int(color):
                return False
    return True


def _rotate_offsets(offsets: list[tuple[int, int]], turns: int) -> list[tuple[int, int]]:
    out = [(int(r), int(c)) for r, c in offsets]
    for _ in range(int(turns) % 4):
        out = [(c, -r) for r, c in out]
    return out


def _dominant_direction(center: tuple[int, int], point: tuple[int, int]) -> str:
    dr = int(point[0]) - int(center[0])
    dc = int(point[1]) - int(center[1])
    if abs(dr) >= abs(dc):
        return "down" if dr > 0 else "up"
    return "right" if dc > 0 else "left"


def _extract_frames(grid: np.ndarray) -> list[dict]:
    infos: list[dict] = []
    for color in map(int, np.unique(grid)):
        if color in (BG, GREY):
            continue
        comps = _components(grid == color)
        frame_comps = [comp for comp in comps if _is_rectangular_frame(grid, color, comp)]
        if len(frame_comps) != 1:
            continue
        frame_coords = frame_comps[0]
        frame_set = set(frame_coords)
        markers = [comp[0] for comp in comps if len(comp) == 1 and comp[0] not in frame_set]
        r0, c0, r1, c1 = _bbox(frame_coords)
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        pattern_offsets: list[tuple[int, int]] = []
        for r in range(r0 + 1, r1):
            for c in range(c0 + 1, c1):
                if int(grid[r, c]) == GREY:
                    pattern_offsets.append((int(r - center[0]), int(c - center[1])))
        if not pattern_offsets:
            continue
        infos.append(
            {
                "color": color,
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "pattern": pattern_offsets,
                "markers": sorted((int(r), int(c)) for r, c in markers),
            }
        )
    infos.sort(key=lambda info: info["bbox"])
    return infos


def main(grid: np.ndarray) -> np.ndarray:
    """
    Stamp each grey pattern from its colored frame onto same-colored marker pixels, rotating by relative direction.

    Input structure:
    - Black background contains 2-3 small rectangular colored frames with border width 1.
    - Each frame encloses a simple Grey pattern on Black interior, with the pattern anchored at the frame center.
    - Elsewhere, scattered single-pixel markers reuse the frame colors.

    Rule:
    - For each frame color, extract the Grey pattern as offsets from that frame's center.
    - Find all singleton markers of the same color outside the frame.
    - Determine each marker's dominant direction relative to the source frame center: up, right, down, or left.
    - Rotate the pattern accordingly and repaint it in Grey centered on the marker pixel, while leaving the source frame unchanged.

    Key insight:
    - Frame color identifies which pattern to copy, while the marker's position relative to the frame supplies the rotation parameter.
    """
    g = np.asarray(grid, dtype=int)
    out = g.copy()
    h, w = g.shape

    for info in _extract_frames(g):
        for marker in info["markers"]:
            direction = _dominant_direction(info["center"], marker)
            rotated = _rotate_offsets(info["pattern"], TURN_BY_DIRECTION[direction])
            for dr, dc in rotated:
                rr = int(marker[0] + dr)
                cc = int(marker[1] + dc)
                if 0 <= rr < h and 0 <= cc < w:
                    out[rr, cc] = GREY

    return out


def _dilate(mask: np.ndarray) -> np.ndarray:
    out = mask.copy()
    out[1:, :] |= mask[:-1, :]
    out[:-1, :] |= mask[1:, :]
    out[:, 1:] |= mask[:, :-1]
    out[:, :-1] |= mask[:, 1:]
    out[1:, 1:] |= mask[:-1, :-1]
    out[1:, :-1] |= mask[:-1, 1:]
    out[:-1, 1:] |= mask[1:, :-1]
    out[:-1, :-1] |= mask[1:, 1:]
    return out


def _slot_origin(slot: str, h: int, w: int, frame_h: int, frame_w: int) -> tuple[int, int]:
    top = 4
    left = 4
    if slot == "tl":
        return top, left
    if slot == "tr":
        return top, w - frame_w - left
    if slot == "bl":
        return h - frame_h - top, left
    if slot == "br":
        return h - frame_h - top, w - frame_w - left
    raise AssertionError(f"unknown slot {slot}")


def _build_direction_lists(rng: np.random.Generator, n_frames: int) -> list[list[str]]:
    if n_frames == 2:
        count_options = ((2, 2), (2, 3), (3, 2), (3, 3))
    else:
        count_options = (
            (2, 1, 1),
            (1, 2, 1),
            (1, 1, 2),
            (2, 2, 1),
            (2, 1, 2),
            (1, 2, 2),
            (2, 2, 2),
            (3, 1, 1),
            (1, 3, 1),
            (1, 1, 3),
        )
    counts = count_options[int(rng.integers(len(count_options)))]
    order = list(DIRECTIONS)
    rng.shuffle(order)
    per_frame = [[] for _ in range(n_frames)]
    if n_frames == 2:
        per_frame[0].extend(order[:2])
        per_frame[1].extend(order[2:])
    else:
        for i in range(3):
            per_frame[i].append(order[i])
        extra_hosts = [i for i, count in enumerate(counts) if count > 1]
        host = extra_hosts[int(rng.integers(len(extra_hosts)))]
        per_frame[host].append(order[3])
    for i, count in enumerate(counts):
        while len(per_frame[i]) < count:
            options = [d for d in DIRECTIONS if d not in per_frame[i]]
            if not options:
                options = list(DIRECTIONS)
            per_frame[i].append(options[int(rng.integers(len(options)))])
        rng.shuffle(per_frame[i])
    return per_frame


def _candidate_points(
    frame: dict,
    direction: str,
    rng: np.random.Generator,
) -> list[tuple[int, int]]:
    r0, c0, r1, c1 = frame["bbox"]
    cr, cc = frame["center"]
    gaps = list(range(2, 7))
    minors = list(range(-3, 4))
    rng.shuffle(gaps)
    rng.shuffle(minors)
    points: list[tuple[int, int]] = []
    for gap in gaps:
        for minor in minors:
            if direction == "up":
                points.append((r0 - gap, cc + minor))
            elif direction == "down":
                points.append((r1 + gap, cc + minor))
            elif direction == "left":
                points.append((cr + minor, c0 - gap))
            else:
                points.append((cr + minor, c1 + gap))
    return points


def _footprint(marker: tuple[int, int], offsets: list[tuple[int, int]]) -> list[tuple[int, int]]:
    mr, mc = marker
    return [(int(mr + dr), int(mc + dc)) for dr, dc in offsets]


def _fits(
    cells: list[tuple[int, int]],
    blocked: np.ndarray,
    shape: tuple[int, int],
) -> bool:
    h, w = shape
    for r, c in cells:
        if not (0 <= r < h and 0 <= c < w):
            return False
        if bool(blocked[r, c]):
            return False
    return True


def _render_input_and_output(frames: list[dict], shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    inp = np.full(shape, BG, dtype=int)
    for frame in frames:
        color = int(frame["color"])
        r0, c0, r1, c1 = frame["bbox"]
        inp[r0, c0 : c1 + 1] = color
        inp[r1, c0 : c1 + 1] = color
        inp[r0 : r1 + 1, c0] = color
        inp[r0 : r1 + 1, c1] = color
        cr, cc = frame["center"]
        for dr, dc in frame["pattern"]:
            inp[cr + dr, cc + dc] = GREY
        for marker in frame["markers"]:
            inp[marker] = color

    out = inp.copy()
    for frame in frames:
        for marker, direction in zip(frame["markers"], frame["directions"]):
            rotated = _rotate_offsets(frame["pattern"], TURN_BY_DIRECTION[direction])
            for r, c in _footprint(marker, rotated):
                out[r, c] = GREY
    return inp, out


def _gen_attempt(rng: np.random.Generator) -> np.ndarray | None:
    h = int(rng.integers(*GRID_RANGE))
    w = int(rng.integers(*GRID_RANGE))
    n_frames = int(rng.integers(2, 4))
    colors = [int(v) for v in rng.choice(FRAME_COLORS, size=n_frames, replace=False)]
    slots = list(rng.choice(np.array(FRAME_SLOTS), size=n_frames, replace=False))
    rng.shuffle(slots)
    directions_per_frame = _build_direction_lists(rng, n_frames)

    frames: list[dict] = []
    for idx in range(n_frames):
        frame_h = int(rng.choice(FRAME_SIZES))
        frame_w = int(rng.choice(FRAME_SIZES))
        pattern = [tuple(map(int, cell)) for cell in PATTERN_LIBRARY[int(rng.integers(len(PATTERN_LIBRARY)))]]
        r0, c0 = _slot_origin(slots[idx], h, w, frame_h, frame_w)
        r1 = r0 + frame_h - 1
        c1 = c0 + frame_w - 1
        center = ((r0 + r1) // 2, (c0 + c1) // 2)
        frames.append(
            {
                "color": colors[idx],
                "bbox": (r0, c0, r1, c1),
                "center": center,
                "pattern": pattern,
                "directions": directions_per_frame[idx],
                "markers": [],
            }
        )

    reserved = np.zeros((h, w), dtype=bool)
    for frame in frames:
        r0, c0, r1, c1 = frame["bbox"]
        reserved[r0 : r1 + 1, c0 : c1 + 1] |= False
        reserved[r0, c0 : c1 + 1] = True
        reserved[r1, c0 : c1 + 1] = True
        reserved[r0 : r1 + 1, c0] = True
        reserved[r0 : r1 + 1, c1] = True
        cr, cc = frame["center"]
        for dr, dc in frame["pattern"]:
            reserved[cr + dr, cc + dc] = True

    for frame in frames:
        for direction in frame["directions"]:
            rotated = _rotate_offsets(frame["pattern"], TURN_BY_DIRECTION[direction])
            blocked = _dilate(reserved)
            placed = None
            for marker in _candidate_points(frame, direction, rng):
                cells = _footprint(marker, rotated)
                if not _fits(cells, blocked, (h, w)):
                    continue
                if _dominant_direction(frame["center"], marker) != direction:
                    continue
                placed = marker
                break
            if placed is None:
                return None
            frame["markers"].append(placed)
            for cell in _footprint(placed, rotated):
                reserved[cell] = True

    inp, expected = _render_input_and_output(frames, (h, w))
    if not np.array_equal(main(inp), expected):
        return None
    return inp


def generate_input() -> np.ndarray:
    rng = np.random.default_rng()
    for _ in range(MAX_ATTEMPTS):
        grid = _gen_attempt(rng)
        if grid is not None:
            return grid
    raise RuntimeError(f"failed to generate a valid sample after {MAX_ATTEMPTS} attempts")
